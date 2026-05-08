#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Charge Transfer Efficiency (CTE) and EPER plot recipes.

Provides five diagnostic plots for CTE/CTI characterisation via the
Extended Pixel Edge Response (EPER) method:

- :func:`plot_cte_vs_signal` — CTE per transfer vs. signal level.
- :func:`plot_cti_vs_signal` — CTI per transfer vs. signal level
  (log y-scale).
- :func:`plot_deferred_charge_vs_signal` — total deferred charge vs.
  signal level with optional error bars.
- :func:`plot_eper_edge_profiles` — trailing-edge and overscan profiles
  at multiple signal levels (the classic EPER signature plot).
- :func:`plot_eper_trailing_detail` — detailed trailing-charge view at
  a single signal level.
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .plot_generator import PlotGenerator
from .helpers import (
    __version__,  # noqa: F401 — re-exported as package version
    _build_subtitle,
    _ensure_parent_dir,
    _make_text_boxes,
    _merge_title,
    _parse_plot_kwargs,
    _warn_unused_kwargs,
)

logger = logging.getLogger(__name__)

_PLOTGEN = PlotGenerator()


def plot_cte_vs_signal(
    serial_signals: np.ndarray | None = None,
    serial_cte: np.ndarray | None = None,
    parallel_signals: np.ndarray | None = None,
    parallel_cte: np.ndarray | None = None,
    title: str = "CTE vs Signal Level",
    output_path: str = "cte_vs_signal.png",
    header_info: dict[str, Any] | None = None,
    hdu: int | str | None = None,
    signal_unit: str = "ADU",
    **kwargs: Any,
) -> None:
    """
    Plot Charge Transfer Efficiency vs. signal level.

    Parameters
    ----------
    serial_signals : np.ndarray, optional
        Signal levels for serial CTE measurements.  Must already be in
        *signal_unit* (caller is responsible for any unit conversion).
    serial_cte : np.ndarray, optional
        Serial CTE values (per transfer).
    parallel_signals : np.ndarray, optional
        Signal levels for parallel CTE measurements.
    parallel_cte : np.ndarray, optional
        Parallel CTE values (per transfer).
    title : str, optional
        Plot title.
    output_path : str, optional
        Destination file path.
    header_info : dict, optional
        Metadata for the subtitle.
    hdu : int or str, optional
        FITS extension identifier for the subtitle.
    signal_unit : str, optional
        Unit label for the x-axis (e.g. ``"ADU"`` or ``"e⁻"``).
        Data must already be in these units.  Default ``"ADU"``.
    **kwargs
        Additional layout keywords forwarded to
        :func:`._helpers._parse_plot_kwargs`.
    """
    _ensure_parent_dir(output_path)

    header_info = header_info or {}
    title1, title2 = _build_subtitle(header_info, hdu)

    figure, axes_extra, legend, grid, colorbar, overwrite, show, remaining = (
        _parse_plot_kwargs(kwargs)
    )
    _warn_unused_kwargs(remaining, "plot_cte_vs_signal")
    figure.setdefault("size", (10, 6))

    datasets: list[dict[str, Any]] = []

    if (serial_signals is not None
            and serial_cte is not None
            and len(serial_signals) > 0):
        datasets.append(
            {
                "type":      "line",
                "x":         np.asarray(serial_signals),
                "y":         np.asarray(serial_cte),
                "marker":    "o",
                "linestyle": "-",
                "color":     "steelblue",
                "linewidth": 1.5,
                "label":     "Serial CTE",
                "zorder":    3,
            }
        )

    if (parallel_signals is not None
            and parallel_cte is not None
            and len(parallel_signals) > 0):
        datasets.append(
            {
                "type":      "line",
                "x":         np.asarray(parallel_signals),
                "y":         np.asarray(parallel_cte),
                "marker":    "s",
                "linestyle": "-",
                "color":     "coral",
                "linewidth": 1.5,
                "label":     "Parallel CTE",
                "zorder":    3,
            }
        )

    result_parts: list[str] = []
    if serial_cte is not None and len(serial_cte) > 0:
        arr = np.asarray(serial_cte)
        valid = np.isfinite(arr)
        if np.any(valid):
            result_parts.append(
                f"Serial CTE: {np.mean(arr[valid]):.8f} (mean)"
            )
    if parallel_cte is not None and len(parallel_cte) > 0:
        arr = np.asarray(parallel_cte)
        valid = np.isfinite(arr)
        if np.any(valid):
            result_parts.append(
                f"Parallel CTE: {np.mean(arr[valid]):.8f} (mean)"
            )
    result1 = "\n".join(result_parts) if result_parts else None

    axes: dict[str, Any] = {
        "title":  _merge_title(title, title1, title2),
        "xlabel": f"Signal ({signal_unit})",
        "ylabel": "CTE per transfer",
        **axes_extra,
    }

    hlines: list[dict[str, Any]] = [
        {
            "y":          1.0,
            "color":      "gray",
            "linestyle":  "--",
            "linewidth":  0.8,
            "label":      "Perfect CTE",
            "show_value": False,
            "zorder":     1,
        },
    ]

    _PLOTGEN.plot(
        output_path=output_path,
        datasets=datasets,
        horizontal_lines=hlines,
        axes=axes,
        figure=figure or None,
        legend={"loc": "lower right", **(legend or {})},
        grid=grid or None,
        colorbar=colorbar or None,
        text_boxes=_make_text_boxes(
            result1=result1, result1_position="lower right"
        ),
        overwrite=overwrite,
        show=show,
    )
    plt.close("all")
    logger.debug("CTE vs signal plot saved to: %s", output_path)


def plot_cti_vs_signal(
    serial_signals: np.ndarray | None = None,
    serial_cti: np.ndarray | None = None,
    parallel_signals: np.ndarray | None = None,
    parallel_cti: np.ndarray | None = None,
    title: str = "CTI vs Signal Level",
    output_path: str = "cti_vs_signal.png",
    header_info: dict[str, Any] | None = None,
    hdu: int | str | None = None,
    signal_unit: str = "ADU",
    **kwargs: Any,
) -> None:
    """
    Plot Charge Transfer Inefficiency vs. signal level (log y-scale).

    Non-positive CTI values are excluded before plotting to keep the
    logarithmic y-axis well-defined.

    Parameters
    ----------
    serial_signals : np.ndarray, optional
        Signal levels for serial CTI measurements.  Must already be in
        *signal_unit*.
    serial_cti : np.ndarray, optional
        Serial CTI values (per transfer).
    parallel_signals : np.ndarray, optional
        Signal levels for parallel CTI measurements.
    parallel_cti : np.ndarray, optional
        Parallel CTI values (per transfer).
    title : str, optional
        Plot title.
    output_path : str, optional
        Destination file path.
    header_info : dict, optional
        Metadata for the subtitle.
    hdu : int or str, optional
        FITS extension identifier for the subtitle.
    signal_unit : str, optional
        Unit label for the x-axis (e.g. ``"ADU"`` or ``"e⁻"``).
        Data must already be in these units.  Default ``"ADU"``.
    **kwargs
        Additional layout keywords forwarded to
        :func:`._helpers._parse_plot_kwargs`.
    """
    _ensure_parent_dir(output_path)

    header_info = header_info or {}
    title1, title2 = _build_subtitle(header_info, hdu)

    figure, axes_extra, legend, grid, colorbar, overwrite, show, remaining = (
        _parse_plot_kwargs(kwargs)
    )
    _warn_unused_kwargs(remaining, "plot_cti_vs_signal")
    figure.setdefault("size", (10, 6))

    datasets: list[dict[str, Any]] = []

    if (serial_signals is not None
            and serial_cti is not None
            and len(serial_signals) > 0):
        s_arr = np.asarray(serial_signals)
        c_arr = np.asarray(serial_cti)
        mask = c_arr > 0
        if np.any(mask):
            datasets.append(
                {
                    "type":      "line",
                    "x":         s_arr[mask],
                    "y":         c_arr[mask],
                    "marker":    "o",
                    "linestyle": "-",
                    "color":     "steelblue",
                    "linewidth": 1.5,
                    "label":     "Serial CTI",
                    "zorder":    3,
                }
            )

    if (parallel_signals is not None
            and parallel_cti is not None
            and len(parallel_signals) > 0):
        s_arr = np.asarray(parallel_signals)
        c_arr = np.asarray(parallel_cti)
        mask = c_arr > 0
        if np.any(mask):
            datasets.append(
                {
                    "type":      "line",
                    "x":         s_arr[mask],
                    "y":         c_arr[mask],
                    "marker":    "s",
                    "linestyle": "-",
                    "color":     "coral",
                    "linewidth": 1.5,
                    "label":     "Parallel CTI",
                    "zorder":    3,
                }
            )

    result_parts: list[str] = []
    if serial_cti is not None and len(serial_cti) > 0:
        arr = np.asarray(serial_cti)
        valid = np.isfinite(arr) & (arr > 0)
        if np.any(valid):
            result_parts.append(
                f"Serial CTI: {np.mean(arr[valid]):.2e} (mean)"
            )
    if parallel_cti is not None and len(parallel_cti) > 0:
        arr = np.asarray(parallel_cti)
        valid = np.isfinite(arr) & (arr > 0)
        if np.any(valid):
            result_parts.append(
                f"Parallel CTI: {np.mean(arr[valid]):.2e} (mean)"
            )
    result1 = "\n".join(result_parts) if result_parts else None

    axes: dict[str, Any] = {
        "title":  _merge_title(title, title1, title2),
        "xlabel": f"Signal ({signal_unit})",
        "ylabel": "CTI per transfer",
        "yscale": "log",
        **axes_extra,
    }

    _PLOTGEN.plot(
        output_path=output_path,
        datasets=datasets,
        axes=axes,
        figure=figure or None,
        legend=legend or None,
        grid=grid or None,
        colorbar=colorbar or None,
        text_boxes=_make_text_boxes(
            result1=result1, result1_position="upper right"
        ),
        overwrite=overwrite,
        show=show,
    )
    plt.close("all")
    logger.debug("CTI vs signal plot saved to: %s", output_path)


def plot_deferred_charge_vs_signal(
    serial_signals: np.ndarray | None = None,
    serial_deferred: np.ndarray | None = None,
    serial_deferred_std: np.ndarray | None = None,
    parallel_signals: np.ndarray | None = None,
    parallel_deferred: np.ndarray | None = None,
    parallel_deferred_std: np.ndarray | None = None,
    title: str = "Deferred Charge vs Signal Level",
    output_path: str = "deferred_charge.png",
    header_info: dict[str, Any] | None = None,
    hdu: int | str | None = None,
    signal_unit: str = "ADU",
    deferred_unit: str | None = None,
    **kwargs: Any,
) -> None:
    """
    Plot total deferred charge vs. signal level with optional error bars.

    Parameters
    ----------
    serial_signals : np.ndarray, optional
        Signal levels for serial EPER measurements.  Must already be
        in *signal_unit*.
    serial_deferred : np.ndarray, optional
        Total deferred charge per signal level (serial).
    serial_deferred_std : np.ndarray, optional
        Uncertainty on *serial_deferred*.
    parallel_signals : np.ndarray, optional
        Signal levels for parallel EPER measurements.
    parallel_deferred : np.ndarray, optional
        Total deferred charge per signal level (parallel).
    parallel_deferred_std : np.ndarray, optional
        Uncertainty on *parallel_deferred*.
    title : str, optional
        Plot title.
    output_path : str, optional
        Destination file path.
    header_info : dict, optional
        Metadata for the subtitle.
    hdu : int or str, optional
        FITS extension identifier for the subtitle.
    signal_unit : str, optional
        Unit label for the x-axis.  Data must already be in these
        units.  Default ``"ADU"``.
    deferred_unit : str, optional
        Unit label for the y-axis.  Defaults to *signal_unit*.
    **kwargs
        Additional layout keywords forwarded to
        :func:`._helpers._parse_plot_kwargs`.
    """
    _ensure_parent_dir(output_path)

    if deferred_unit is None:
        deferred_unit = signal_unit

    header_info = header_info or {}
    title1, title2 = _build_subtitle(header_info, hdu)

    figure, axes_extra, legend, grid, colorbar, overwrite, show, remaining = (
        _parse_plot_kwargs(kwargs)
    )
    _warn_unused_kwargs(remaining, "plot_deferred_charge_vs_signal")
    figure.setdefault("size", (10, 6))

    datasets: list[dict[str, Any]] = []

    if (serial_signals is not None
            and serial_deferred is not None
            and len(serial_signals) > 0):
        ds: dict[str, Any] = {
            "type":      "errorbar",
            "x":         np.asarray(serial_signals),
            "y":         np.asarray(serial_deferred),
            "color":     "steelblue",
            "label":     "Serial",
            "fmt":       "o-",
            "capsize":   3,
            "linewidth": 1.5,
            "zorder":    3,
        }
        if serial_deferred_std is not None:
            ds["yerr"] = np.asarray(serial_deferred_std)
        datasets.append(ds)

    if (parallel_signals is not None
            and parallel_deferred is not None
            and len(parallel_signals) > 0):
        ds = {
            "type":      "errorbar",
            "x":         np.asarray(parallel_signals),
            "y":         np.asarray(parallel_deferred),
            "color":     "coral",
            "label":     "Parallel",
            "fmt":       "s-",
            "capsize":   3,
            "linewidth": 1.5,
            "zorder":    3,
        }
        if parallel_deferred_std is not None:
            ds["yerr"] = np.asarray(parallel_deferred_std)
        datasets.append(ds)

    axes: dict[str, Any] = {
        "title":  _merge_title(title, title1, title2),
        "xlabel": f"Signal ({signal_unit})",
        "ylabel": f"Deferred Charge ({deferred_unit})",
        **axes_extra,
    }

    _PLOTGEN.plot(
        output_path=output_path,
        datasets=datasets,
        axes=axes,
        figure=figure or None,
        legend=legend or None,
        grid=grid or None,
        colorbar=colorbar or None,
        overwrite=overwrite,
        show=show,
    )
    plt.close("all")
    logger.debug(
        "Deferred charge vs signal plot saved to: %s", output_path
    )


def plot_eper_edge_profiles(
    edge_profiles: list[list[float]],
    trailing_profiles: list[list[float]],
    signal_levels: np.ndarray,
    direction: str = "Serial",
    title: str = "EPER Edge Profiles",
    output_path: str = "eper_edge_profiles.png",
    header_info: dict[str, Any] | None = None,
    hdu: int | str | None = None,
    signal_unit: str = "ADU",
    **kwargs: Any,
) -> None:
    """
    Plot trailing-edge and overscan profiles at multiple signal levels.

    This is the classic EPER signature plot: signal drops from the
    active-area level to the overscan level, with deferred charge
    visible in the first few overscan pixels.  Profiles are coloured
    by signal level using the viridis colourmap.

    X-axis convention
    -----------------
    Active-area pixels are numbered ..., -2, -1 (the last active pixel
    is -1).  Overscan pixels are numbered 1, 2, 3, ... (the first
    overscan pixel is 1).  The boundary between active area and overscan
    falls at x = 0, marked by a vertical dashed line.  This convention
    means pixel indices directly reflect physical distance from the
    active/overscan boundary with no ambiguity.

    Parameters
    ----------
    edge_profiles : list of list of float
        Per-pixel profiles from the trailing edge of the active area.
        One list per signal level.  Displayed as step plots.
    trailing_profiles : list of list of float
        Per-pixel profiles from the overscan trailing region.
        One list per signal level.  Displayed as step plots.
    signal_levels : np.ndarray
        Mean signal level for each profile.  Must already be in
        *signal_unit* (caller is responsible for any unit conversion).
    direction : {"Serial", "Parallel"}
        Readout direction; controls axis labelling.
    title : str, optional
        Plot title.
    output_path : str, optional
        Destination file path.
    header_info : dict, optional
        Metadata for the subtitle.
    hdu : int or str, optional
        FITS extension identifier for the subtitle.
    signal_unit : str, optional
        Unit label for axis annotations and legend entries.
        Data must already be in these units.  Default ``"ADU"``.
    **kwargs
        Additional layout keywords forwarded to
        :func:`._helpers._parse_plot_kwargs`.
    """
    _ensure_parent_dir(output_path)

    header_info = header_info or {}
    title1, title2 = _build_subtitle(header_info, hdu)

    figure, axes_extra, legend, grid, colorbar, overwrite, show, remaining = (
        _parse_plot_kwargs(kwargs)
    )
    _warn_unused_kwargs(remaining, "plot_eper_edge_profiles")
    figure.setdefault("size", (12, 7))

    pixel_label = "Column" if direction.lower() == "serial" else "Row"
    cmap = plt.cm.viridis
    n_profiles = len(signal_levels)

    datasets: list[dict[str, Any]] = []
    for i in range(n_profiles):
        edge = np.asarray(edge_profiles[i])
        trail = np.asarray(trailing_profiles[i])

        n_edge = len(edge)
        n_trail = len(trail)

        # Active-area pixels: -n_edge, ..., -1  (last active pixel = -1)
        # Overscan pixels:     1, 2, ..., n_trail (first overscan pixel = 1)
        # Boundary is at x = 0; there is no pixel at x = 0.
        x_active  = np.arange(-n_edge, 0)          # -n_edge … -1
        x_overscan = np.arange(1, n_trail + 1)      #  1 … n_trail
        x_full = np.concatenate([x_active, x_overscan])
        y_full = np.concatenate([edge, trail])

        color = cmap(i / max(n_profiles - 1, 1))

        datasets.append(
            {
                "type":      "step",
                "x":         x_full,
                "y":         y_full,
                "color":     color,
                "linewidth": 1.4,
                "label":     f"{signal_levels[i]:.0f} {signal_unit}",
                "zorder":    2 + i,
            }
        )

    # Boundary line at x = 0 — the gap between pixel -1 and pixel 1.
    vlines: list[dict[str, Any]] = [
        {
            "x":          0,
            "color":      "black",
            "linestyle":  "--",
            "linewidth":  1.0,
            "show_value": False,
            "zorder":     10,
        },
    ]

    axes: dict[str, Any] = {
        "title":  _merge_title(title, title1, title2),
        "xlabel": f"{pixel_label} index (negative = active area, positive = overscan)",
        "ylabel": f"Signal ({signal_unit}, baseline-subtracted)",
        **axes_extra,
    }

    # "Active Area" and "Overscan" labels placed just inside the plot
    # at the top of the frame, flanking the boundary line at x=0.
    # Compute x position of the boundary in axes coordinates [0,1] from
    # the actual data range so the labels track the boundary correctly
    # regardless of how many active vs overscan pixels are shown.
    all_x = np.concatenate([
        np.arange(-len(edge_profiles[i]), 0) for i in range(n_profiles)
    ] + [
        np.arange(1, len(trailing_profiles[i]) + 1) for i in range(n_profiles)
    ])
    x_min = float(np.min(all_x)) - 0.5   # half-pixel margin for step plot
    x_max = float(np.max(all_x)) + 0.5
    x_range = x_max - x_min
    # x=0 in axes coordinates (boundary is between pixel -1 and pixel 1)
    boundary_axes = (0.0 - x_min) / x_range
    label_offset = 0.02   # small gap either side of the boundary line

    annotations: list[dict[str, Any]] = [
        {
            "text":      "◄ Active Area",
            "x":         boundary_axes - label_offset,
            "y":         0.97,
            "transform": "axes",
            "ha":        "right",
            "va":        "top",
            "fontsize":  10,
            "color":     "dimgray",
        },
        {
            "text":      "Overscan ►",
            "x":         boundary_axes + label_offset,
            "y":         0.97,
            "transform": "axes",
            "ha":        "left",
            "va":        "top",
            "fontsize":  10,
            "color":     "dimgray",
        },
    ]

    legend_cfg: dict[str, Any] = {
        "loc":      "upper left",
        "fontsize": 9,
        "ncol":     1,
        **(legend or {}),
    }

    _PLOTGEN.plot(
        output_path=output_path,
        datasets=datasets,
        vertical_lines=vlines,
        annotations=annotations,
        axes=axes,
        figure=figure or None,
        legend=legend_cfg,
        grid=grid or None,
        colorbar=colorbar or None,
        overwrite=overwrite,
        show=show,
    )
    plt.close("all")
    logger.debug("EPER edge profiles plot saved: %s", output_path)

def plot_eper_trailing_detail(
    trailing_profile: list[float],
    baseline_profile: list[float],
    signal_level: float,
    cte: float,
    cti: float,
    direction: str = "Serial",
    title: str = "Trailing Charge Detail",
    output_path: str = "eper_trailing_detail.png",
    header_info: dict[str, Any] | None = None,
    hdu: int | str | None = None,
    signal_unit: str = "ADU",
    **kwargs: Any,
) -> None:
    """
    Detailed view of trailing charge in the overscan at one signal level.

    Shows baseline-subtracted excess signal in the first few overscan
    pixels as a step plot (one flat per pixel, honest discrete sampling).
    Pixel 1 is the first overscan pixel immediately after the active area.
    A zero-reference line is drawn when *baseline_profile* is provided.

    Parameters
    ----------
    trailing_profile : list of float
        Per-pixel excess signal in the overscan trailing region
        (baseline-subtracted).  Must already be in *signal_unit*.
        Element 0 corresponds to overscan pixel 1.
    baseline_profile : list of float
        Per-pixel signal in the far overscan (raw, for reference).
        When non-empty, a horizontal zero reference line is drawn.
    signal_level : float
        Mean active-area signal for this measurement, in *signal_unit*.
    cte : float
        CTE value for this measurement.
    cti : float
        CTI value for this measurement.
    direction : {"Serial", "Parallel"}
        Readout direction; controls axis labelling.
    title : str, optional
        Plot title.
    output_path : str, optional
        Destination file path.
    header_info : dict, optional
        Metadata for the subtitle.
    hdu : int or str, optional
        FITS extension identifier for the subtitle.
    signal_unit : str, optional
        Unit label for axis and annotation text.
        Data must already be in these units.  Default ``"ADU"``.
    **kwargs
        Additional layout keywords forwarded to _parse_plot_kwargs.
    """
    _ensure_parent_dir(output_path)

    header_info = header_info or {}
    title1, title2 = _build_subtitle(header_info, hdu)

    figure, axes_extra, legend, grid, colorbar, overwrite, show, remaining = (
        _parse_plot_kwargs(kwargs)
    )
    _warn_unused_kwargs(remaining, "plot_eper_trailing_detail")
    figure.setdefault("size", (10, 6))

    pixel_label = "Column" if direction.lower() == "serial" else "Row"
    trail = np.asarray(trailing_profile)
    n_trail = len(trail)

    # Pixel 1 = first overscan pixel immediately adjacent to active area.
    x = np.arange(1, n_trail + 1)

    datasets: list[dict[str, Any]] = [
        {
            "type":      "step",
            "x":         x,
            "y":         trail,
            "color":     "steelblue",
            "linewidth": 1.5,
            "label":     f"Trailing charge (baseline-subtracted)",
            "zorder":    3,
        },
    ]

    # Zero reference line — only when far-overscan baseline is available
    hlines: list[dict[str, Any]] = []
    if baseline_profile and len(baseline_profile) > 0:
        hlines.append(
            {
                "y":          0.0,
                "color":      "gray",
                "linestyle":  "--",
                "linewidth":  0.8,
                "show_value": False,
                "zorder":     1,
            }
        )

    # Flag anomalous baseline columns (> 5 sigma from median)
    baseline_note = ""
    if baseline_profile and len(baseline_profile) > 2:
        bp = np.asarray(baseline_profile)
        bp_med = np.median(bp)
        bp_mad = np.median(np.abs(bp - bp_med)) * 1.4826
        if bp_mad > 0:
            n_anomalous = int(np.sum(np.abs(bp - bp_med) > 5 * bp_mad))
            if n_anomalous > 0:
                baseline_note = f"\n⚠ {n_anomalous} anomalous baseline col(s)"

    total_deferred = float(np.sum(trail))

    # Show the CTI calculation explicitly so the reader can verify it
    if cti > 0 and signal_level > 0:
        # n_transfers is implicit in cti = deferred / (signal * n_transfers)
        # Back-calculate n_transfers for display
        n_transfers_display = round(total_deferred / (signal_level * cti))
        cti_check = f"CTI = {total_deferred:.2f} / ({signal_level:.1f} × {n_transfers_display}) = {cti:.3e}"
    else:
        cti_check = ""

    result_parts = [
        f"Signal: {signal_level:.1f} {signal_unit}",
        f"Deferred (first {n_trail} overscan pixels): {total_deferred:.2f} {signal_unit}",
        cti_check,
        f"{direction} CTI: {cti:.3e}",
        f"{direction} CTE: {cte:.8f}",
    ]
    if baseline_note:
        result_parts.append(baseline_note)
    result1 = "\n".join(r for r in result_parts if r)

    # Set x-axis to start cleanly at 0.5 so pixel 1 step is fully visible
    axes: dict[str, Any] = {
        "title":  _merge_title(title, title1, title2),
        "xlabel": f"Overscan {pixel_label} (1 = first pixel after active area)",
        "ylabel": f"Excess Signal ({signal_unit})",
        "xlim":   (0.5, n_trail + 0.5),
        **axes_extra,
    }

    _PLOTGEN.plot(
        output_path=output_path,
        datasets=datasets,
        horizontal_lines=hlines,
        axes=axes,
        figure=figure or None,
        legend=legend or None,
        grid=grid or None,
        colorbar=colorbar or None,
        text_boxes=_make_text_boxes(
            result1=result1, result1_position="upper right"
        ),
        overwrite=overwrite,
        show=show,
    )
    plt.close("all")
    logger.debug("EPER trailing detail plot saved: %s", output_path)
