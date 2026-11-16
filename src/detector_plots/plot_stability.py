#!/usr/bin/env python3
#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"
"""
Stability plot recipes.

Views of a long time series of repeated measurements, such as a flat taken
every few minutes for days:

- :func:`plot_stability_time_series` — stacked panels sharing a time axis,
  one per quantity, with optional binned means and linear trend.
- :func:`plot_allan_deviation` — Allan deviation against averaging time on
  log-log axes, with an optional white-noise reference slope.
- :func:`plot_stability_tile_map` — a grid of per-tile values (for example
  the change in signal of each region of the frame), annotated in each cell.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .helpers import (
    _build_subtitle,
    _ensure_parent_dir,
    _merge_title,
    _parse_plot_kwargs,
    _warn_unused_kwargs,
)

logger = logging.getLogger(__name__)


def _save(fig: Any, output_path: str, figure: dict[str, Any], overwrite: bool, show: bool) -> None:
    outp = Path(output_path)
    if outp.exists() and not overwrite:
        plt.close("all")
        raise FileExistsError(f"Refusing to overwrite existing file: {output_path}")
    fig.savefig(output_path, dpi=figure.get("dpi"), bbox_inches="tight")
    if show:
        plt.show()
    plt.close("all")


def plot_stability_time_series(
    time_hours: np.ndarray,
    panels: list[dict[str, Any]],
    title: str,
    output_path: str,
    header_info: dict[str, Any] | None = None,
    hdu: int | str | None = None,
    xlabel: str = "Elapsed time (h)",
    shade_before_hours: float | None = None,
    **kwargs: Any,
) -> None:
    """
    Plot several quantities against time, one panel each, sharing the x-axis.

    Parameters
    ----------
    time_hours : np.ndarray
        Time of each sample in hours (e.g. elapsed since the first sample).
    panels : list of dict
        One dict per panel, top to bottom. Keys:

        - ``"y"`` (required): values, same length as *time_hours*.
        - ``"ylabel"`` (required): axis label, including units.
        - ``"label"``: legend label for the samples.
        - ``"binned_x"``, ``"binned_y"``: binned means drawn as a line with
          markers over the samples.
        - ``"trend"``: ``(slope_per_hour, intercept)`` of a linear fit, drawn
          as a dashed line and annotated.
        - ``"trend_unit"``: unit for the slope annotation (default ``"/h"``).
        - ``"color"``: matplotlib colour for the samples.
        - ``"reference"``: y value for a horizontal reference line (e.g. 0).
    title : str
        Figure title.
    output_path : str
        Destination file path.
    header_info : dict, optional
        Metadata for the subtitle (``instrument``, ``detector``,
        ``date_obs``, ``mode``).
    hdu : int or str, optional
        FITS extension identifier for the subtitle.
    xlabel : str, optional
        Label of the shared time axis.
    shade_before_hours : float, optional
        Shade the region before this time in every panel (for example a
        settling period excluded from the statistics).
    **kwargs
        Layout keywords understood by ``_parse_plot_kwargs`` (``figsize_x``,
        ``figsize_y``, ``dpi``, ``xlim``, ``overwrite``, ``show``, ...).
    """
    if not panels:
        raise ValueError("plot_stability_time_series needs at least one panel.")
    _ensure_parent_dir(output_path)
    title1, title2 = _build_subtitle(header_info or {}, hdu)
    figure, axes_extra, _legend, _grid, _cb, overwrite, show, remaining = _parse_plot_kwargs(kwargs)
    _warn_unused_kwargs(remaining, "plot_stability_time_series")

    t = np.asarray(time_hours, dtype=float)
    size = figure.get("size", (11, 2.1 * len(panels) + 1.2))
    fig, axes = plt.subplots(len(panels), 1, sharex=True, figsize=size, squeeze=False)
    for ax, panel in zip(axes[:, 0], panels, strict=True):
        y = np.asarray(panel["y"], dtype=float)
        ax.plot(t, y, ".", ms=3, alpha=0.6, color=panel.get("color", "tab:blue"),
                label=panel.get("label"))
        if panel.get("binned_x") is not None and panel.get("binned_y") is not None:
            ax.plot(np.asarray(panel["binned_x"]), np.asarray(panel["binned_y"]), "o-",
                    ms=4, lw=1.2, color="black", label="binned mean", zorder=5)
        trend = panel.get("trend")
        if trend is not None:
            slope, intercept = trend
            ax.plot(t, slope * t + intercept, "--", lw=1.2, color="tab:red", zorder=6,
                    label=f"trend {slope:+.3g} {panel.get('trend_unit', '/h')}")
        if panel.get("reference") is not None:
            ax.axhline(panel["reference"], color="gray", lw=0.8, ls=":")
        if shade_before_hours is not None and shade_before_hours > 0:
            ax.axvspan(t.min() if t.size else 0.0, shade_before_hours, color="0.85", zorder=0)
        ax.set_ylabel(panel["ylabel"])
        ax.grid(True, alpha=0.3)
        if panel.get("label") or trend is not None or panel.get("binned_y") is not None:
            ax.legend(loc="best", fontsize=8, frameon=False)
    axes[-1, 0].set_xlabel(xlabel)
    if "xlim" in axes_extra:
        axes[-1, 0].set_xlim(axes_extra["xlim"])
    fig.suptitle(_merge_title(title, title1, title2))
    fig.tight_layout()
    _save(fig, output_path, figure, overwrite, show)
    logger.debug("Stability time series saved to: %s", output_path)


def plot_allan_deviation(
    series: list[dict[str, Any]],
    title: str,
    output_path: str,
    header_info: dict[str, Any] | None = None,
    hdu: int | str | None = None,
    xlabel: str = "Averaging time (h)",
    ylabel: str = "Allan deviation (%)",
    show_white_noise_slope: bool = True,
    **kwargs: Any,
) -> None:
    """
    Plot Allan deviation against averaging time for one or more series.

    Parameters
    ----------
    series : list of dict
        One dict per curve. Keys: ``"tau"`` (averaging times),
        ``"adev"`` (Allan deviations), ``"label"``, and optionally
        ``"adev_err"`` (error bars) and ``"color"``.
    title : str
        Plot title.
    output_path : str
        Destination file path.
    header_info : dict, optional
        Metadata for the subtitle.
    hdu : int or str, optional
        FITS extension identifier for the subtitle.
    xlabel, ylabel : str, optional
        Axis labels.
    show_white_noise_slope : bool, optional
        Draw a tau^-1/2 reference line through the first point of the first
        series: white noise follows it, drift rises above it.
    **kwargs
        Layout keywords understood by ``_parse_plot_kwargs``.
    """
    _ensure_parent_dir(output_path)
    title1, title2 = _build_subtitle(header_info or {}, hdu)
    figure, axes_extra, _legend, _grid, _cb, overwrite, show, remaining = _parse_plot_kwargs(kwargs)
    _warn_unused_kwargs(remaining, "plot_allan_deviation")

    fig, ax = plt.subplots(figsize=figure.get("size", (9, 6)))
    first: tuple[float, float] | None = None
    for s in series:
        tau = np.asarray(s["tau"], dtype=float)
        adev = np.asarray(s["adev"], dtype=float)
        ok = np.isfinite(tau) & np.isfinite(adev) & (tau > 0) & (adev > 0)
        if not ok.any():
            continue
        err = s.get("adev_err")
        if err is not None:
            ax.errorbar(tau[ok], adev[ok], yerr=np.asarray(err, dtype=float)[ok], fmt="o-",
                        ms=4, capsize=2, label=s.get("label"), color=s.get("color"))
        else:
            ax.plot(tau[ok], adev[ok], "o-", ms=4, label=s.get("label"), color=s.get("color"))
        if first is None:
            first = (float(tau[ok][0]), float(adev[ok][0]))
    if show_white_noise_slope and first is not None:
        t0, a0 = first
        all_tau = np.concatenate([np.asarray(s["tau"], dtype=float) for s in series])
        tt = np.array([t0, np.nanmax(all_tau)])
        ax.plot(tt, a0 * np.sqrt(t0 / tt), ":", color="gray", label=r"white noise ($\tau^{-1/2}$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for key in ("xlim", "ylim"):
        if key in axes_extra:
            getattr(ax, f"set_{key}")(axes_extra[key])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax.set_title(_merge_title(title, title1, title2))
    fig.tight_layout()
    _save(fig, output_path, figure, overwrite, show)
    logger.debug("Allan deviation plot saved to: %s", output_path)


def plot_stability_tile_map(
    values: np.ndarray,
    title: str,
    output_path: str,
    colorbar_label: str = "Change (%)",
    header_info: dict[str, Any] | None = None,
    hdu: int | str | None = None,
    value_format: str = "{:+.3f}",
    symmetric: bool = True,
    **kwargs: Any,
) -> None:
    """
    Show a small grid of per-tile values (rows x columns of the frame).

    Parameters
    ----------
    values : np.ndarray
        2-D array, one value per tile; row 0 is drawn at the bottom, as the
        first rows of the frame.
    title : str
        Plot title.
    output_path : str
        Destination file path.
    colorbar_label : str, optional
        Label of the colour bar.
    header_info : dict, optional
        Metadata for the subtitle.
    hdu : int or str, optional
        FITS extension identifier for the subtitle.
    value_format : str, optional
        Format for the number written in each cell.
    symmetric : bool, optional
        Centre the colour scale on zero (diverging colour map).
    **kwargs
        Layout keywords understood by ``_parse_plot_kwargs``.
    """
    _ensure_parent_dir(output_path)
    title1, title2 = _build_subtitle(header_info or {}, hdu)
    figure, _axes, _legend, _grid, _cb, overwrite, show, remaining = _parse_plot_kwargs(kwargs)
    _warn_unused_kwargs(remaining, "plot_stability_tile_map")

    v = np.asarray(values, dtype=float)
    if v.ndim != 2:
        raise ValueError(f"values must be 2-D, got shape {v.shape}")
    fig, ax = plt.subplots(figsize=figure.get("size", (8, 5)))
    finite = v[np.isfinite(v)]
    if symmetric and finite.size:
        lim = float(np.max(np.abs(finite))) or 1.0
        im = ax.imshow(v, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    else:
        im = ax.imshow(v, origin="lower", cmap="viridis", aspect="auto")
    norm = im.norm
    for (r, c), val in np.ndenumerate(v):
        if np.isfinite(val):
            strong = abs(float(norm(val)) - 0.5) > 0.3 if symmetric else float(norm(val)) < 0.35
            ax.text(c, r, value_format.format(val), ha="center", va="center", fontsize=9,
                    color="white" if strong else "black")
    ax.set_xticks(range(v.shape[1]))
    ax.set_yticks(range(v.shape[0]))
    ax.set_xlabel("Tile column (serial direction)")
    ax.set_ylabel("Tile row (parallel direction)")
    fig.colorbar(im, ax=ax, label=colorbar_label)
    ax.set_title(_merge_title(title, title1, title2))
    fig.tight_layout()
    _save(fig, output_path, figure, overwrite, show)
    logger.debug("Stability tile map saved to: %s", output_path)
