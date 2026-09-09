#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"
"""
Variance-ratio plot recipes.

Provides diagnostic plots that compare measured variance to expected
(shot + read) noise variance, either as a function of signal level or
exposure time.  Ratios departing from unity indicate noise sources
beyond the shot-noise + read-noise model.
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


def plot_variance_ratio(
    mean_signals: np.ndarray,
    measured_variances: np.ndarray,
    expected_variances: np.ndarray,
    title: str,
    output_path: str,
    gain_e_per_adu: float | None = None,
    read_noise_e: float | None = None,
    header_info: dict[str, Any] | None = None,
    hdu: int | str | None = None,
    xlabel: str = "Mean Signal (ADU)",
    ylabel: str = "Variance Ratio (Measured / Expected)",
    ylim: tuple[float, float] | None = None,
    xlim: tuple[float, float] | None = None,
    show_unity_line: bool = True,
    show_signal_ranges: bool = True,
    signal_range_markers_adu: tuple[float, float] = (100.0, 500.0),
    **kwargs: Any,
) -> None:
    """
    Plot the ratio of measured variance to expected variance vs. signal level.

    This diagnostic reveals signal-dependent noise sources beyond shot
    noise, such as illumination instability, fixed pattern noise, or
    sensor non-linearity.

    Parameters
    ----------
    mean_signals : np.ndarray
        Mean signal values in ADU.
    measured_variances : np.ndarray
        Measured variance values in ADU².
    expected_variances : np.ndarray
        Expected variance in ADU² (typically ``signal/gain + RN_adu²``).
    title : str
        Plot title.
    output_path : str
        Destination file path.
    gain_e_per_adu : float, optional
        System gain in e⁻/ADU (for annotation only).
    read_noise_e : float, optional
        Read noise in electrons (for annotation only).
    header_info : dict, optional
        Metadata for the subtitle.  Recognised keys: ``"instrument"``,
        ``"detector"``, ``"date_obs"``, ``"mode"``.
    hdu : int or str, optional
        FITS extension identifier for the subtitle.
    xlabel : str, optional
        X-axis label.  Default ``"Mean Signal (ADU)"``.
    ylabel : str, optional
        Y-axis label.  Default ``"Variance Ratio (Measured / Expected)"``.
    ylim : tuple of float, optional
        Y-axis limits ``(min, max)``.
    xlim : tuple of float, optional
        X-axis limits ``(min, max)``.
    show_unity_line : bool, optional
        Overlay a horizontal line at ratio = 1.  Default *True*.
    show_signal_ranges : bool, optional
        Overlay vertical reference lines at ``signal_range_markers_adu``.
        Default *True*.
    signal_range_markers_adu : tuple of float, optional
        ADU positions of the two vertical reference lines drawn when
        *show_signal_ranges* is *True*. Default ``(100.0, 500.0)`` — pick
        values appropriate to this sensor's actual signal range (e.g. its
        read-noise floor and a representative mid-range signal) rather than
        relying on the default, which was tuned for a specific instrument.
    **kwargs
        Additional layout keywords forwarded to
        :func:`._helpers._parse_plot_kwargs`.

    Notes
    -----
    Common interpretations of the ratio:

    - ≈ 1.0 — noise dominated by shot noise + read noise (ideal).
    - > 1.0 at high signal — illumination instability or uncancelled FPN.
    - < 1.0 at low signal — possible over-subtraction or correlation.
    - Increasing with signal — percentage-level light-source variation.
    """
    _ensure_parent_dir(output_path)

    mean_signals = np.asarray(mean_signals)
    measured_variances = np.asarray(measured_variances)
    expected_variances = np.asarray(expected_variances)

    valid_mask = expected_variances > 0
    variance_ratio = np.full_like(measured_variances, np.nan)
    variance_ratio[valid_mask] = (
        measured_variances[valid_mask] / expected_variances[valid_mask]
    )

    header_info = header_info or {}
    title1, title2 = _build_subtitle(header_info, hdu)

    figure, axes_extra, legend, grid, colorbar, overwrite, show, remaining = (
        _parse_plot_kwargs(kwargs)
    )
    _warn_unused_kwargs(remaining, "plot_variance_ratio")
    figure.setdefault("size", (10, 6))

    datasets: list[dict[str, Any]] = [
        {
            "type":   "scatter",
            "x":      mean_signals,
            "y":      variance_ratio,
            "marker": "o",
            "color":  "tab:blue",
            "label":  "Variance Ratio",
            "zorder": 5,
        },
    ]

    horizontal_lines: list[dict[str, Any]] = []
    if show_unity_line:
        horizontal_lines.append(
            {
                "y":          1.0,
                "color":      "green",
                "linestyle":  "-",
                "linewidth":  2,
                "label":      "Expected (ratio=1)",
                "show_value": False,
            }
        )

    vertical_lines: list[dict[str, Any]] = []
    if show_signal_ranges:
        marker_low, marker_high = signal_range_markers_adu
        vertical_lines += [
            {
                "x":          marker_low,
                "color":      "gray",
                "linestyle":  ":",
                "alpha":      0.7,
                "label":      f"{marker_low:g} ADU",
                "show_value": False,
            },
            {
                "x":          marker_high,
                "color":      "gray",
                "linestyle":  "--",
                "alpha":      0.7,
                "label":      f"{marker_high:g} ADU",
                "show_value": False,
            },
        ]

    result_parts: list[str] = []
    if gain_e_per_adu is not None:
        result_parts.append(f"Gain: {gain_e_per_adu:.3f} e⁻/ADU")
    if read_noise_e is not None:
        result_parts.append(f"Read Noise: {read_noise_e:.2f} e⁻")
    valid_ratios = variance_ratio[valid_mask]
    if len(valid_ratios) > 0:
        result_parts.append(f"Mean Ratio: {np.mean(valid_ratios):.3f}")
        result_parts.append(f"Std Ratio: {np.std(valid_ratios):.3f}")
    result_text = "\n".join(result_parts) if result_parts else None

    axes: dict[str, Any] = {
        "title":  _merge_title(title, title1, title2),
        "xlabel": xlabel,
        "ylabel": ylabel,
        **axes_extra,
    }
    if xlim is not None:
        axes["xlim"] = xlim
    if ylim is not None:
        axes["ylim"] = ylim

    _PLOTGEN.plot(
        output_path=output_path,
        datasets=datasets,
        axes=axes,
        figure=figure or None,
        legend={"loc": "upper left", **(legend or {})},
        grid=grid or None,
        colorbar=colorbar or None,
        horizontal_lines=horizontal_lines,
        vertical_lines=vertical_lines,
        text_boxes=_make_text_boxes(
            result1=result_text, result1_position="upper right"
        ),
        overwrite=overwrite,
        show=show,
    )
    plt.close("all")
    logger.debug("Variance ratio plot saved to: %s", output_path)


def plot_variance_ratio_by_exposure(
    exposure_times: np.ndarray,
    measured_variances: np.ndarray,
    expected_variances: np.ndarray,
    title: str,
    output_path: str,
    gain_e_per_adu: float | None = None,
    read_noise_e: float | None = None,
    header_info: dict[str, Any] | None = None,
    hdu: int | str | None = None,
    xlabel: str = "Exposure Time (s)",
    ylabel: str = "Variance Ratio (Measured / Expected)",
    ylim: tuple[float, float] | None = None,
    xlim: tuple[float, float] | None = None,
    show_unity_line: bool = True,
    **kwargs: Any,
) -> None:
    """
    Plot measured-to-expected variance ratio vs. exposure time.

    Similar to :func:`plot_variance_ratio` but with exposure time on
    the x-axis, useful for identifying timing-related noise sources
    such as shutter jitter or light-source oscillation correlated with
    exposure duration.

    Parameters
    ----------
    exposure_times : np.ndarray
        Exposure time values in seconds.
    measured_variances : np.ndarray
        Measured variance values in ADU².
    expected_variances : np.ndarray
        Expected variance values in ADU².
    title : str
        Plot title.
    output_path : str
        Destination file path.
    gain_e_per_adu : float, optional
        System gain in e⁻/ADU (for annotation only).
    read_noise_e : float, optional
        Read noise in electrons (for annotation only).
    header_info : dict, optional
        Metadata for the subtitle.
    hdu : int or str, optional
        FITS extension identifier for the subtitle.
    xlabel : str, optional
        X-axis label.  Default ``"Exposure Time (s)"``.
    ylabel : str, optional
        Y-axis label.  Default ``"Variance Ratio (Measured / Expected)"``.
    ylim : tuple of float, optional
        Y-axis limits.
    xlim : tuple of float, optional
        X-axis limits.
    show_unity_line : bool, optional
        Overlay a horizontal line at ratio = 1.  Default *True*.
    **kwargs
        Additional layout keywords forwarded to
        :func:`._helpers._parse_plot_kwargs`.
    """
    _ensure_parent_dir(output_path)

    exposure_times = np.asarray(exposure_times)
    measured_variances = np.asarray(measured_variances)
    expected_variances = np.asarray(expected_variances)

    valid_mask = expected_variances > 0
    variance_ratio = np.full_like(measured_variances, np.nan)
    variance_ratio[valid_mask] = (
        measured_variances[valid_mask] / expected_variances[valid_mask]
    )

    header_info = header_info or {}
    title1, title2 = _build_subtitle(header_info, hdu)

    figure, axes_extra, legend, grid, colorbar, overwrite, show, remaining = (
        _parse_plot_kwargs(kwargs)
    )
    _warn_unused_kwargs(remaining, "plot_variance_ratio_by_exposure")
    figure.setdefault("size", (10, 6))

    datasets: list[dict[str, Any]] = [
        {
            "type":   "scatter",
            "x":      exposure_times,
            "y":      variance_ratio,
            "marker": "o",
            "color":  "tab:blue",
            "label":  "Variance Ratio",
            "zorder": 5,
        },
    ]

    horizontal_lines: list[dict[str, Any]] = []
    if show_unity_line:
        horizontal_lines.append(
            {
                "y":          1.0,
                "color":      "green",
                "linestyle":  "-",
                "linewidth":  2,
                "label":      "Expected (ratio=1)",
                "show_value": False,
            }
        )

    result_parts: list[str] = []
    if gain_e_per_adu is not None:
        result_parts.append(f"Gain: {gain_e_per_adu:.3f} e⁻/ADU")
    if read_noise_e is not None:
        result_parts.append(f"Read Noise: {read_noise_e:.2f} e⁻")
    valid_ratios = variance_ratio[valid_mask]
    if len(valid_ratios) > 0:
        result_parts.append(f"Mean Ratio: {np.mean(valid_ratios):.3f}")
        result_parts.append(f"Std Ratio: {np.std(valid_ratios):.3f}")
    result_text = "\n".join(result_parts) if result_parts else None

    axes: dict[str, Any] = {
        "title":  _merge_title(title, title1, title2),
        "xlabel": xlabel,
        "ylabel": ylabel,
        **axes_extra,
    }
    if xlim is not None:
        axes["xlim"] = xlim
    if ylim is not None:
        axes["ylim"] = ylim

    _PLOTGEN.plot(
        output_path=output_path,
        datasets=datasets,
        axes=axes,
        figure=figure or None,
        legend={"loc": "upper left", **(legend or {})},
        grid=grid or None,
        colorbar=colorbar or None,
        horizontal_lines=horizontal_lines,
        text_boxes=_make_text_boxes(
            result1=result_text, result1_position="upper right"
        ),
        overwrite=overwrite,
        show=show,
    )
    plt.close("all")
    logger.debug(
        "Variance ratio by exposure plot saved to: %s", output_path
    )
