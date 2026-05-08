#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified, dictionary-driven plotting engine for detector-related plots.

Provides :class:`PlotGenerator`, a single reusable backend that all
high-level plot-recipe modules instantiate.  Supporting axis-polish
helpers are also defined here.
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import transforms
from matplotlib.ticker import (
    AutoMinorLocator,
    FormatStrFormatter,
    StrMethodFormatter,
)
from scipy.stats import norm

logger = logging.getLogger(__name__)

__version__ = "1.3.0"


# ---------------------------------------------------------------------------
# Private axis-annotation helpers
# ---------------------------------------------------------------------------

def _label_hline(
    ax: plt.Axes,
    y: float,
    *,
    color: str | None = None,
    value_fmt: str = "{:.3g}",
    value_pos: str = "right",
    x_offset: int = 4,
    fontsize: int = 9,
    label_zorder: int = 10,
) -> None:
    """
    Annotate a horizontal line with its numeric value.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    y : float
        Data-coordinate value of the line.
    color : str, optional
        Text colour; defaults to the line colour if *None*.
    value_fmt : str
        Python format string used to render *y* (``"{:.3g}"`` style).
    value_pos : {"right", "left"}
        Which spine to anchor the label against.
    x_offset : int
        Horizontal offset in points away from the spine.
    fontsize : int
        Label font size in points.
    label_zorder : int
        Drawing z-order for the annotation.
    """
    x = 1.0 if value_pos == "right" else 0.0
    ha = "left" if value_pos == "right" else "right"
    trans = ax.get_yaxis_transform()
    ax.annotate(
        value_fmt.format(y),
        (x, y),
        xycoords=trans,
        textcoords="offset points",
        xytext=(x_offset if value_pos == "right" else -x_offset, 0),
        va="center",
        ha=ha,
        fontsize=fontsize,
        color=color,
        zorder=label_zorder,
    )


def _label_vline(
    ax: plt.Axes,
    x: float,
    *,
    color: str | None = None,
    value_fmt: str = "{:.3g}",
    value_pos: str = "top",
    y_offset: int = 4,
    fontsize: int = 9,
    label_zorder: int = 10,
) -> None:
    """
    Annotate a vertical line with its numeric value.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    x : float
        Data-coordinate value of the line.
    color : str, optional
        Text colour; defaults to the line colour if *None*.
    value_fmt : str
        Python format string used to render *x*.
    value_pos : {"top", "bottom"}
        Which spine to anchor the label against.
    y_offset : int
        Vertical offset in points away from the spine.
    fontsize : int
        Label font size in points.
    label_zorder : int
        Drawing z-order for the annotation.
    """
    y = 1.0 if value_pos == "top" else 0.0
    va = "bottom" if value_pos == "top" else "top"
    trans = ax.get_xaxis_transform()
    ax.annotate(
        value_fmt.format(x),
        (x, y),
        xycoords=trans,
        textcoords="offset points",
        xytext=(0, y_offset if value_pos == "top" else -y_offset),
        va=va,
        ha="center",
        fontsize=fontsize,
        color=color,
        zorder=label_zorder,
    )


def _polish_axes(
    ax: plt.Axes,
    *,
    thousands: bool = True,
    minor: bool = True,
    headroom: float = 0.03,
    yformat: str | None = None,
    xformat: str | None = None,
) -> None:
    """
    Apply finishing touches to a matplotlib axes object.

    Adds minor ticks, optional thousands-separator formatting, and a
    small amount of headroom at the top of the y-axis so data lines do
    not touch the frame.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    thousands : bool
        If *True*, apply thousands-separator formatting to the y-axis
        when values are large enough to warrant it.  Automatically
        suppressed for small value ranges.
    minor : bool
        If *True*, add :class:`~matplotlib.ticker.AutoMinorLocator` to
        both axes (linear scales only).
    headroom : float
        Fraction of the y-axis range to add as padding at the top.
    yformat : str, optional
        Custom format string for the y-axis tick labels
        (``"%.2f"`` or ``"{x:.2f}"`` styles are both accepted).
        When provided this overrides automatic formatting.
    xformat : str, optional
        Custom format string for the x-axis tick labels.
        When provided this overrides automatic formatting.
    """
    # Minor ticks — linear scales only
    try:
        if minor and ax.get_xscale() == "linear":
            ax.xaxis.set_minor_locator(AutoMinorLocator())
        if minor and ax.get_yscale() == "linear":
            ax.yaxis.set_minor_locator(AutoMinorLocator())
    except Exception:
        pass

    # Y-axis tick formatting
    try:
        if ax.get_yscale() == "linear":
            if yformat is not None:
                if yformat.startswith("%"):
                    ax.yaxis.set_major_formatter(FormatStrFormatter(yformat))
                else:
                    ax.yaxis.set_major_formatter(StrMethodFormatter(yformat))
            elif thousands:
                y_lo, y_hi = ax.get_ylim()
                y_range = abs(y_hi - y_lo)
                y_max = max(abs(y_lo), abs(y_hi))
                if y_max >= 1000 and y_range >= 10:
                    ax.yaxis.set_major_formatter(
                        StrMethodFormatter("{x:,.0f}")
                    )
                elif y_max < 10:
                    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
                elif y_max < 100:
                    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    except Exception:
        pass

    # X-axis tick formatting (only when explicitly requested)
    try:
        if xformat is not None and ax.get_xscale() == "linear":
            if xformat.startswith("%"):
                ax.xaxis.set_major_formatter(FormatStrFormatter(xformat))
            else:
                ax.xaxis.set_major_formatter(StrMethodFormatter(xformat))
    except Exception:
        pass

    # Top headroom — keep data lines away from the frame
    try:
        y_lo, y_hi = ax.get_ylim()
        if y_hi > y_lo and headroom and ax.get_yscale() == "linear":
            ax.set_ylim(y_lo, y_hi + headroom * (y_hi - y_lo))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# PlotGenerator
# ---------------------------------------------------------------------------

class PlotGenerator:
    """
    Unified, dictionary-driven plotting engine for detector characterisation.

    All high-level plot-recipe functions (e.g. in ``detector_plots``)
    create a single shared instance of this class and delegate all
    rendering to :meth:`plot`.

    Parameters
    ----------
    default_figure : dict, optional
        Default figure configuration overrides.  Keys: ``"size"``,
        ``"dpi"``, ``"tight_layout"``.
    default_axes : dict, optional
        Default axes configuration overrides.  Keys: ``"xlabel"``,
        ``"ylabel"``, ``"xlim"``, ``"ylim"``, ``"xscale"``,
        ``"yscale"``.
    default_grid : dict, optional
        Default grid configuration overrides.  Keys: ``"visible"``,
        ``"linestyle"``, ``"color"``, ``"alpha"``.
    default_legend : dict, optional
        Default legend configuration overrides.  Keys: ``"loc"``,
        ``"fontsize"``, ``"frameon"``.
    """

    def __init__(
        self,
        default_figure: dict[str, Any] | None = None,
        default_axes: dict[str, Any] | None = None,
        default_grid: dict[str, Any] | None = None,
        default_legend: dict[str, Any] | None = None,
    ) -> None:
        self.default_figure: dict[str, Any] = default_figure or {
            "size": (10, 8),
            "dpi": 300,
            "tight_layout": True,
        }
        self.default_axes: dict[str, Any] = default_axes or {
            "xlabel": "X",
            "ylabel": "Y",
            "xlim": None,
            "ylim": None,
            "xscale": "linear",
            "yscale": "linear",
        }
        self.default_grid: dict[str, Any] = default_grid or {
            "visible": True,
            "linestyle": ":",
            "color": "gray",
            "alpha": 0.6,
        }
        self.default_legend: dict[str, Any] = default_legend or {
            "loc": "best",
            "fontsize": None,
            "frameon": True,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plot(
        self,
        output_path: str | None = None,
        datasets: list[dict[str, Any]] | None = None,
        horizontal_lines: list[dict[str, Any]] | None = None,
        vertical_lines: list[dict[str, Any]] | None = None,
        annotations: list[dict[str, Any]] | None = None,
        shaded_regions: list[dict[str, Any]] | None = None,
        error_bands: list[dict[str, Any]] | None = None,
        results_boxes: list[dict[str, Any]] | None = None,
        text_boxes: list[dict[str, Any]] | None = None,
        legend: dict[str, Any] | None = None,
        axes: dict[str, Any] | None = None,
        figure: dict[str, Any] | None = None,
        grid: dict[str, Any] | None = None,
        colorbar: dict[str, Any] | None = None,
        overwrite: bool = True,
        show: bool = False,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Render a figure from declarative dataset and style dictionaries.

        Parameters
        ----------
        output_path : str, optional
            File path at which to save the figure.  When *None* the
            figure is created but not written to disk.
        datasets : list of dict, optional
            Each element describes one data series.  Required key:
            ``"type"`` (``"line"``, ``"scatter"``, ``"errorbar"``,
            ``"hist"``, or ``"imshow"``); ``"x"`` and ``"y"`` arrays;
            plus any style keys (``"color"``, ``"marker"``, etc.).
        horizontal_lines : list of dict, optional
            Each element draws an ``axhline``.  Required key: ``"y"``.
            Optional keys: ``"color"``, ``"linestyle"``, ``"linewidth"``,
            ``"label"``, ``"show_value"``, ``"value_fmt"``,
            ``"value_pos"``, ``"zorder"``.
        vertical_lines : list of dict, optional
            Each element draws an ``axvline``.  Required key: ``"x"``.
            Same optional keys as *horizontal_lines*.
        annotations : list of dict, optional
            Free-text or arrowed annotations.  When ``"xy"`` is present
            an arrowed annotation is drawn (data coords); otherwise a
            plain text box is placed using ``"x"``/``"y"`` (axes or
            data coords per ``"transform"``).
        shaded_regions : list of dict, optional
            Each element draws a vertical ``axvspan``.  Required keys:
            ``"xmin"``, ``"xmax"``.  Optional: ``"color"``, ``"alpha"``,
            ``"label"``.
        error_bands : list of dict, optional
            Each element draws a ``fill_between`` band.  Required keys:
            ``"x"``, ``"y1"``, ``"y2"``.  Optional: ``"color"``,
            ``"alpha"``, ``"label"``.
        results_boxes : list of dict, optional
            Deprecated alias for *text_boxes*.  Ignored when
            *text_boxes* is also provided.
        text_boxes : list of dict, optional
            Styled text boxes placed in axes or figure coordinates.
            Required key: ``"text"`` (or ``"lines"`` for a list).
            Optional keys: ``"xy"``, ``"loc"``, ``"ha"``, ``"va"``,
            ``"fontsize"``, ``"facecolor"``, ``"alpha"``,
            ``"transform"`` (``"axes"`` | ``"figure"``), ``"zorder"``.
        legend : dict, optional
            Legend configuration; merged with *default_legend*.
            Keys: ``"loc"``, ``"fontsize"``, ``"frameon"``, ``"ncol"``.
        axes : dict, optional
            Axes configuration; merged with *default_axes*.
        figure : dict, optional
            Figure configuration; merged with *default_figure*.
        grid : dict, optional
            Grid configuration; merged with *default_grid*.
        colorbar : dict, optional
            Colorbar configuration.  Keys: ``"visible"``, ``"label"``,
            ``"orientation"``, ``"shrink"``.
        overwrite : bool
            Currently unused at this level (handled by callers).
        show : bool
            If *True*, call ``plt.show()`` after rendering.

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        """
        datasets = datasets or []
        horizontal_lines = horizontal_lines or []
        vertical_lines = vertical_lines or []
        annotations = annotations or []
        shaded_regions = shaded_regions or []
        error_bands = error_bands or []
        results_boxes = results_boxes or []

        legend_cfg: dict[str, Any] = {
            **self.default_legend,
            **(legend or {}),
        }
        axes_cfg: dict[str, Any] = {
            **self.default_axes,
            **(axes or {}),
        }
        figure_cfg: dict[str, Any] = {
            **self.default_figure,
            **(figure or {}),
        }
        grid_cfg: dict[str, Any] = {
            **self.default_grid,
            **(grid or {}),
        }
        colorbar_cfg: dict[str, Any] = colorbar or {
            "visible": True,
            "label": None,
            "orientation": "vertical",
            "shrink": 1.0,
        }

        fig, ax = plt.subplots(
            figsize=figure_cfg.get("size"),
            dpi=figure_cfg.get("dpi"),
        )

        # --- datasets ---------------------------------------------------
        for ds in datasets:
            self._plot_dataset(ax, ds, fig)

        # --- error bands ------------------------------------------------
        for band in error_bands:
            bx = np.asarray(band["x"])
            by1 = np.asarray(band["y1"])
            by2 = np.asarray(band["y2"])
            ax.fill_between(
                bx, by1, by2,
                color=band.get("color", "gray"),
                alpha=band.get("alpha", 0.3),
                label=band.get("label"),
            )

        # --- shaded vertical regions ------------------------------------
        for region in shaded_regions:
            ax.axvspan(
                region["xmin"],
                region["xmax"],
                color=region.get("color", "yellow"),
                alpha=region.get("alpha", 0.2),
                label=region.get("label"),
            )

        # --- horizontal guide lines -------------------------------------
        for line in horizontal_lines:
            yval: float = line["y"]
            ax.axhline(
                y=yval,
                color=line.get("color", "r"),
                linestyle=line.get("linestyle", "--"),
                linewidth=line.get("linewidth", 1),
                zorder=line.get("zorder", 0),
            )
            if not line.get("show_value", True):
                continue

            line_color = line.get("color", "black")
            line_label = line.get("label", "")
            value_fmt = line.get("value_fmt", "{:.1f}")
            xpos: str = line.get("xpos", "right")
            x_offset: int = line.get("x_offset", 0)
            base_y_offset: int = line.get("y_offset", 6)

            bt = transforms.blended_transform_factory(
                ax.transAxes, ax.transData
            )
            x_frac = {"left": 0.01, "center": 0.5, "right": 0.99}.get(
                xpos, 0.99
            )
            ha = {"left": "left", "center": "center", "right": "right"}[
                xpos
            ]

            yl0, yl1 = ax.get_ylim()
            y_offset = base_y_offset
            if yval > yl0 + 0.85 * (yl1 - yl0):
                y_offset = -abs(base_y_offset)
            elif yval < yl0 + 0.15 * (yl1 - yl0):
                y_offset = abs(base_y_offset)
            va = "bottom" if y_offset >= 0 else "top"

            ann_text = (
                f"{line_label}: {value_fmt.format(yval)}"
                if line_label
                else value_fmt.format(yval)
            )
            ax.annotate(
                ann_text,
                (x_frac, yval),
                xycoords=bt,
                textcoords="offset points",
                xytext=(x_offset, y_offset),
                ha=ha,
                va=va,
                fontsize=line.get("fontsize", 9),
                color=line.get("text_color", line_color),
                zorder=line.get("label_zorder", 10),
                clip_on=True,
                bbox=line.get(
                    "bbox",
                    {"fc": "white", "alpha": 0.6, "ec": "none", "pad": 1.5},
                ),
            )

        # --- vertical guide lines ---------------------------------------
        for line in vertical_lines:
            xval: float = line["x"]
            ax.axvline(
                x=xval,
                color=line.get("color", "r"),
                linestyle=line.get("linestyle", "--"),
                linewidth=line.get("linewidth", 1),
                zorder=line.get("zorder", 0),
            )
            if line.get("show_value", True):
                _label_vline(
                    ax,
                    xval,
                    color=line.get("text_color") or line.get("color", "r"),
                    value_fmt=line.get("value_fmt", "{:.3g}"),
                    value_pos=line.get("value_pos", "top"),
                    y_offset=line.get("y_offset", 4),
                    fontsize=line.get("fontsize", 9),
                )

        # --- free-text / arrowed annotations ----------------------------
        for ann in annotations:
            ann_text = ann.get("text", "")
            fontsize: int = ann.get("fontsize", 10)
            color: str = ann.get("color", "black")
            z: int = ann.get("zorder", 10)

            if "xy" in ann:
                # Arrowed annotation anchored to a data point
                ax.annotate(
                    ann_text,
                    xy=tuple(ann["xy"]),
                    xytext=ann.get("xytext", (20, 12)),
                    textcoords=ann.get("textcoords", "offset points"),
                    ha=ann.get("ha", "left"),
                    va=ann.get("va", "bottom"),
                    fontsize=fontsize,
                    color=color,
                    arrowprops=ann.get(
                        "arrowprops",
                        {"arrowstyle": "->", "color": color, "lw": 1.0},
                    ),
                    zorder=z,
                )
            else:
                # Plain text placed in axes or data coordinates
                transform = (
                    ax.transAxes
                    if ann.get("transform") == "axes"
                    else ax.transData
                )
                ax.text(
                    ann["x"],
                    ann["y"],
                    ann_text,
                    transform=transform,
                    fontsize=fontsize,
                    ha=ann.get("ha", "left"),
                    va=ann.get("va", "top"),
                    color=color,
                    zorder=z,
                )

        # --- text / results boxes ---------------------------------------
        # Prefer text_boxes when provided; fall back to deprecated results_boxes.
        active_text_boxes: list[dict[str, Any]] = (
            text_boxes if text_boxes is not None else results_boxes
        )

        _LOC_AXES: dict[str, tuple[float, float, str, str]] = {
            "upper left":  (0.05, 0.95, "left",   "top"),
            "upper right": (0.95, 0.95, "left",   "top"),
            "lower left":  (0.05, 0.05, "left",   "bottom"),
            "lower right": (0.95, 0.05, "right",  "bottom"),
            "center":      (0.50, 0.50, "center", "center"),
        }

        for tb in active_text_boxes:
            if "text" in tb:
                tb_text = tb["text"]
            else:
                lines = tb.get("lines", [])
                tb_text = (
                    "\n".join(lines)
                    if isinstance(lines, (list, tuple))
                    else str(lines)
                )

            tb_fontsize: int = tb.get("fontsize", 10)
            tb_bbox: dict[str, Any] = tb.get(
                "bbox",
                {
                    "boxstyle": tb.get("boxstyle", "round,pad=0.5"),
                    "fc": tb.get("facecolor", "white"),
                    "alpha": tb.get("alpha", 1.0),
                },
            )

            transform_name: str = tb.get("transform", "axes")
            loc: str | None = tb.get("loc")
            xy: tuple[float, float] | None = tb.get("xy")

            if transform_name == "figure":
                default_xy = (0.5, 0.02)
                default_ha, default_va = "center", "bottom"
            else:
                default_xy = (0.98, 0.98)
                default_ha, default_va = "right", "top"

            if xy is None and loc:
                loc_tuple = _LOC_AXES.get(loc)
                if loc_tuple:
                    tb_x, tb_y, tb_ha, tb_va = loc_tuple
                else:
                    tb_x, tb_y = default_xy
                    tb_ha, tb_va = default_ha, default_va
            else:
                tb_x, tb_y = xy or default_xy
                tb_ha = tb.get("ha", default_ha)
                tb_va = tb.get("va", default_va)

            if transform_name == "figure":
                fig.text(
                    tb_x, tb_y, tb_text,
                    fontsize=tb_fontsize,
                    ha=tb_ha, va=tb_va,
                    bbox=tb_bbox,
                    zorder=tb.get("zorder", 10),
                    multialignment="left",
                )
            else:
                ax.text(
                    tb_x, tb_y, tb_text,
                    transform=ax.transAxes,
                    fontsize=tb_fontsize,
                    ha=tb_ha, va=tb_va,
                    bbox=tb_bbox,
                    zorder=tb.get("zorder", 10),
                    multialignment="left",
                )

        # --- axis labels, scales, limits --------------------------------
        ax.set_xlabel(axes_cfg.get("xlabel"))
        ax.set_ylabel(axes_cfg.get("ylabel"))
        ax.set_xscale(axes_cfg.get("xscale", "linear"))
        ax.set_yscale(axes_cfg.get("yscale", "linear"))

        if axes_cfg.get("xlim") is not None:
            ax.set_xlim(*axes_cfg["xlim"])
        if axes_cfg.get("ylim") is not None:
            ax.set_ylim(*axes_cfg["ylim"])

        # --- grid -------------------------------------------------------
        if grid_cfg.get("visible", True):
            ax.grid(
                True,
                linestyle=grid_cfg.get("linestyle", ":"),
                color=grid_cfg.get("color", "gray"),
                alpha=grid_cfg.get("alpha", 0.6),
            )

        # --- legend -----------------------------------------------------
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                loc=legend_cfg.get("loc", "best"),
                fontsize=legend_cfg.get("fontsize"),
                frameon=legend_cfg.get("frameon", True),
                ncol=legend_cfg.get("ncol", 1),
            )

        # --- colorbar ---------------------------------------------------
        if colorbar_cfg.get("visible", False):
            images = ax.get_images()
            if images:
                cbar = fig.colorbar(
                    images[-1],
                    ax=ax,
                    orientation=colorbar_cfg.get("orientation", "vertical"),
                    shrink=colorbar_cfg.get("shrink", 1.0),
                )
                if colorbar_cfg.get("label"):
                    cbar.set_label(colorbar_cfg["label"])

        # --- title / subtitle -------------------------------------------
        plot_title: str | None = axes_cfg.get("title")
        subtitle: str | None = axes_cfg.get("subtitle")
        title_fontsize: int = axes_cfg.get("title_fontsize", 12)
        subtitle_fontsize: int = axes_cfg.get("subtitle_fontsize", 10)

        if plot_title and subtitle:
            fig.suptitle(plot_title, fontsize=title_fontsize, fontweight="bold")
            ax.set_title(subtitle, fontsize=subtitle_fontsize, color="dimgray")
        elif plot_title:
            ax.set_title(plot_title, fontsize=title_fontsize)

        # --- layout / polish --------------------------------------------
        if figure_cfg.get("tight_layout", True):
            plt.tight_layout()

        if plot_title and subtitle:
            plt.subplots_adjust(top=0.90)

        yformat: str | None = (
            axes_cfg.get("yformat")
            or axes_cfg.get("yaxis_format")
            or axes_cfg.get("yaxis_major_formatter")
        )
        xformat: str | None = (
            axes_cfg.get("xformat")
            or axes_cfg.get("xaxis_format")
            or axes_cfg.get("xaxis_major_formatter")
        )
        _polish_axes(
            ax,
            thousands=True,
            minor=True,
            headroom=0.03,
            yformat=yformat,
            xformat=xformat,
        )

        # --- save / display ---------------------------------------------
        if output_path:
            try:
                plt.savefig(
                    output_path,
                    dpi=figure_cfg.get("dpi"),
                    bbox_inches="tight",
                )
                logger.info("Saved plot to: %s", output_path)
            except Exception as exc:
                logger.error("Failed to save plot %s: %s", output_path, exc)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig, ax

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _plot_dataset(
        self,
        ax: plt.Axes,
        ds: dict[str, Any],
        fig: plt.Figure | None = None,
    ) -> None:
        """
        Render a single dataset dict onto *ax*.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Target axes.
        ds : dict
            Dataset descriptor.  Required keys: ``"y"`` (array-like).
            Optional keys: ``"x"``, ``"type"`` (default ``"line"``),
            and any Matplotlib style keyword (``"color"``,
            ``"marker"``, ``"linestyle"``, ``"linewidth"``,
            ``"label"``, ``"zorder"``, ``"alpha"``).
        fig : matplotlib.figure.Figure, optional
            Parent figure; used only for ``"imshow"`` datasets when a
            per-dataset colorbar is requested.
        """
        y = np.asarray(ds["y"])
        x = np.asarray(ds.get("x", np.arange(len(y))))
        ptype: str = ds.get("type") or ds.get("plot_type") or "line"

        style: dict[str, Any] = {
            k: ds[k]
            for k in (
                "color", "marker", "linestyle", "linewidth",
                "label", "zorder", "alpha",
            )
            if k in ds
        }

        if ptype in ("line", "plot"):
            ax.plot(x, y, **style)

        elif ptype == "step":
            # Step plot: each pixel value displayed as a horizontal flat
            # centred on its x coordinate, with vertical transitions between
            # adjacent pixels.  This is the honest representation of discrete
            # sampled data — no interpolation is implied between samples.
            # 'where=mid' places the step edge at the midpoint between
            # adjacent x values, so each sample occupies equal width.
            step_style = {k: v for k, v in style.items() if k != "marker"}
            ax.step(x, y, where="mid", **step_style)
            # Overlay markers at sample centres if explicitly requested.
            if "marker" in style and style["marker"] not in (None, "None", ""):
                marker_style = {
                    k: style[k] for k in ("color", "marker", "zorder", "alpha")
                    if k in style
                }
                ax.plot(x, y, linestyle="none", **marker_style)

        elif ptype == "scatter":
            scatter_style = {
                k: v for k, v in style.items()
                if k not in ("linestyle", "linewidth")
            }
            if "s" in ds:
                scatter_style["s"] = ds["s"]
            ax.scatter(x, y, **scatter_style)

        elif ptype == "errorbar":
            err_style = dict(style)
            if "fmt" in ds:
                err_style.pop("marker", None)
                err_style.pop("linestyle", None)
            ax.errorbar(
                x, y,
                yerr=ds.get("yerr"),
                fmt=ds.get("fmt", ""),
                capsize=ds.get("capsize", 0),
                **err_style,
            )

        elif ptype == "hist":
            bins = ds.get("bins", "auto")
            ax.hist(y, bins=bins, label=ds.get("label"), alpha=ds.get("alpha", 0.7))
            if ds.get("show_fit", True):
                mu, sigma = norm.fit(y)
                x_lo, x_hi = ax.get_xlim()
                xx = np.linspace(x_lo, x_hi, 200)
                pdf = norm.pdf(xx, mu, sigma)
                ax.plot(
                    xx,
                    pdf * len(y) * (x_hi - x_lo) / 100.0,
                    ds.get("fit_color", "k"),
                    linewidth=2,
                )
                ax.axvline(mu, color="r", linestyle="dashed", linewidth=1)

        elif ptype in ("imshow", "image"):
            im = ax.imshow(
                y,
                origin=ds.get("origin", "lower"),
                cmap=ds.get("cmap", "viridis"),
                aspect=ds.get("aspect", "auto"),
                vmin=ds.get("vmin"),
                vmax=ds.get("vmax"),
            )
            if fig is not None and ds.get("colorbar", True):
                fig.colorbar(im, ax=ax)
