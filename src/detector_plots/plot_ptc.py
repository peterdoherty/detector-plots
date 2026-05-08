#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photon Transfer Curve (PTC) plot recipe.

Provides :func:`plot_ptc`, which renders a PTC in either ADU or
electron units, with optional log-log scaling and shot-noise reference
lines.
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


def plot_ptc(
    all_mean_signals: np.ndarray,
    all_variances: np.ndarray,
    fit_mean_signals: np.ndarray,
    fit_variances: np.ndarray,
    excluded_mean_signals: np.ndarray,
    excluded_variances: np.ndarray,
    ptc_gain_e_per_adu: float,
    ptc_slope: float,
    ptc_intercept: float,
    read_noise_e: float | None = None,
    read_noise_from_bias_e: float | None = None,
    title: str = "Photon Transfer Curve",
    output_path: str = "ptc_plot.png",
    header_info: dict[str, Any] | None = None,
    hdu: int | str | None = None,
    units: str = "ADU",
    x_max: float | None = None,
    loglog: bool = False,
    show_reference_lines: bool = True,
    **kwargs: Any,
) -> None:
    """
    Render a Photon Transfer Curve (PTC) plot.

    Handles unit conversion to electrons, optional x-axis zoom, and
    optional log-log scaling internally.  Both ADU and electron units
    produce consistent styling.

    Parameters
    ----------
    all_mean_signals : np.ndarray
        All mean signal values in ADU (before any unit conversion).
    all_variances : np.ndarray
        All variance values in ADU².
    fit_mean_signals : np.ndarray
        Binned mean signal values used for the linear fit (ADU).
    fit_variances : np.ndarray
        Binned variance values used for the linear fit (ADU²).
    excluded_mean_signals : np.ndarray
        Binned mean signal values excluded from the fit (ADU).
    excluded_variances : np.ndarray
        Binned variance values excluded from the fit (ADU²).
    ptc_gain_e_per_adu : float
        System gain in e⁻/ADU.  Must be positive.
    ptc_slope : float
        Slope of the linear PTC fit (ADU² / ADU).
    ptc_intercept : float
        Intercept of the linear PTC fit (ADU²).
    read_noise_e : float, optional
        Read noise in electrons (for annotation and reference lines).
    read_noise_from_bias_e : float, optional
        Read noise derived from bias frames; takes precedence over
        *read_noise_e* for display when provided.
    title : str, optional
        Plot title.  Default ``"Photon Transfer Curve"``.
    output_path : str, optional
        Destination file path.  Default ``"ptc_plot.png"``.
    header_info : dict, optional
        Metadata for the subtitle.  Recognised keys: ``"instrument"``,
        ``"detector"``, ``"date_obs"``, ``"mode"``.
    hdu : int or str, optional
        FITS extension identifier for the subtitle.
    units : str, optional
        Display units: ``"ADU"`` (default) or any of ``"e-"``,
        ``"electrons"``, ``"e"`` to convert to electrons.
    x_max : float, optional
        Upper x-axis limit for zoomed plots (in the selected units).
    loglog : bool, optional
        If *True*, render both axes on a logarithmic scale.
    show_reference_lines : bool, optional
        If *True*, overlay shot-noise (and optionally read-noise)
        reference lines.
    **kwargs
        Additional layout keywords forwarded to
        :func:`._helpers._parse_plot_kwargs`.

    Raises
    ------
    ValueError
        If *ptc_gain_e_per_adu* is not positive.
    """
    _ensure_parent_dir(output_path)

    if ptc_gain_e_per_adu is None or ptc_gain_e_per_adu <= 0:
        raise ValueError(
            f"Invalid gain: {ptc_gain_e_per_adu}. Must be positive."
        )

    # Unit conversion
    is_electrons = units.lower() in ("e-", "electrons", "e")
    gain = ptc_gain_e_per_adu if is_electrons else 1.0
    gain_sq = gain ** 2
    sig_unit = "e⁻" if is_electrons else "ADU"
    var_unit = "e⁻²" if is_electrons else "ADU²"

    all_x = np.asarray(all_mean_signals, dtype=float) * gain
    all_y = np.asarray(all_variances, dtype=float) * gain_sq
    fit_x = np.asarray(fit_mean_signals, dtype=float) * gain
    fit_y = np.asarray(fit_variances, dtype=float) * gain_sq
    exc_x = np.asarray(excluded_mean_signals, dtype=float) * gain
    exc_y = np.asarray(excluded_variances, dtype=float) * gain_sq

    def _filter(
        arr_x: np.ndarray, arr_y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        mask = np.isfinite(arr_x) & np.isfinite(arr_y)
        if loglog:
            mask &= (arr_x > 0) & (arr_y > 0)
        if x_max is not None:
            mask &= arr_x <= x_max
        return arr_x[mask], arr_y[mask]

    all_x, all_y = _filter(all_x, all_y)
    fit_x, fit_y = _filter(fit_x, fit_y)
    exc_x, exc_y = _filter(exc_x, exc_y)

    if len(all_x) == 0:
        logger.warning(
            "No valid data points remain for PTC plot %s", output_path
        )
        return

    header_info = header_info or {}
    title1, title2_base = _build_subtitle(header_info, hdu)
    extra_parts: list[str] = []
    if x_max is not None:
        extra_parts.append(f"Zoomed: 0-{x_max:.0f} {sig_unit}")
    if title2_base:
        extra_parts.append(title2_base)
    title2 = " | ".join(extra_parts)

    rn_display = (
        read_noise_from_bias_e
        if read_noise_from_bias_e is not None
        else read_noise_e
    )
    res_parts = [f"Gain: {ptc_gain_e_per_adu:.4f} e⁻/ADU"]
    if rn_display is not None and np.isfinite(rn_display):
        res_parts.append(f"Read Noise: {rn_display:.2f} e⁻")
    if loglog:
        res_parts.append("Slope=1 → Shot noise")
    result1 = "\n".join(res_parts)

    datasets: list[dict[str, Any]] = [
        {
            "type": "scatter", "x": all_x, "y": all_y,
            "marker": ".", "color": "lightgray",
            "label": "Raw Data", "zorder": 1,
        },
        {
            "type": "line", "x": fit_x, "y": fit_y,
            "marker": "o", "linestyle": "--", "linewidth": 1.5,
            "color": "orange", "label": "Binned Data (Fit)", "zorder": 3,
        },
    ]
    if len(exc_x) > 0:
        datasets.append(
            {
                "type": "line", "x": exc_x, "y": exc_y,
                "marker": "o", "linestyle": "", "color": "darkblue",
                "label": "Binned Data (Excluded)", "zorder": 2,
            }
        )

    x_line = np.linspace(
        max(0.0, float(np.min(all_x))), float(np.max(all_x)), 200
    )
    if loglog and show_reference_lines:
        y_ref = x_line if is_electrons else ptc_slope * x_line
        datasets.append(
            {
                "type": "line", "x": x_line, "y": y_ref,
                "linestyle": "--", "color": "darkred",
                "label": "Shot Noise (Slope=1)", "zorder": 4,
                "show_value": False,
            }
        )
    elif show_reference_lines:
        if is_electrons:
            datasets.append(
                {
                    "type": "line", "x": x_line, "y": x_line,
                    "linestyle": "--", "color": "darkred",
                    "label": "Shot Noise (var=mean)", "zorder": 4,
                    "show_value": False,
                }
            )
            if rn_display is not None and rn_display > 0:
                datasets.append(
                    {
                        "type": "line", "x": x_line,
                        "y": x_line + rn_display ** 2,
                        "linestyle": ":", "color": "purple",
                        "label": f"Shot + RN ({rn_display:.2f} e⁻)",
                        "zorder": 4,
                        "show_value": False,
                    }
                )
        else:
            datasets.append(
                {
                    "type": "line", "x": x_line,
                    "y": ptc_slope * x_line + ptc_intercept,
                    "linestyle": "--", "color": "darkred",
                    "label": "Linear Fit", "zorder": 4,
                    "show_value": False,
                }
            )

    figure, axes_extra, legend, grid, colorbar, overwrite, show, remaining = (
        _parse_plot_kwargs(kwargs)
    )
    _warn_unused_kwargs(remaining, "plot_ptc")
    figure.setdefault("size", (10, 6))

    axes: dict[str, Any] = {
        "title":  _merge_title(title, title1, title2),
        "xlabel": f"Mean Signal ({sig_unit})",
        "ylabel": f"Variance ({var_unit})",
        "xscale": "log" if loglog else "linear",
        "yscale": "log" if loglog else "linear",
        **axes_extra,
    }
    if x_max is not None and not loglog:
        axes["xlim"] = (0, x_max)

    _PLOTGEN.plot(
        output_path=output_path,
        datasets=datasets,
        axes=axes,
        figure=figure or None,
        legend=legend or None,
        grid=grid or None,
        colorbar=colorbar or None,
        text_boxes=_make_text_boxes(
            result1=result1, result1_position="lower right"
        ),
        overwrite=overwrite,
        show=show,
    )
    plt.close("all")
    logger.debug("PTC plot saved to: %s", output_path)
