#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"
"""
Linearity plot recipes.

Provides three complementary views of detector linearity:

- :func:`plot_linearity_curve` — signal vs. exposure time with linear
  and polynomial fit overlays.
- :func:`plot_linearity_residuals` — percent deviation from the global
  linear fit over the full signal range.
- :func:`plot_linearity_residuals_low` — percent deviation in the
  low-signal regime, with optional local-refit comparison.
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
    _get_ordinal_suffix,
    _make_text_boxes,
    _merge_title,
    _parse_plot_kwargs,
    _warn_unused_kwargs,
)

logger = logging.getLogger(__name__)

_PLOTGEN = PlotGenerator()


def plot_linearity_curve(
    exposure_times: np.ndarray,
    mean_signals: np.ndarray,
    fit_exposure_times: np.ndarray,
    fit_mean_signals: np.ndarray,
    title: str,
    output_path: str,
    slope: float | None = None,
    intercept: float | None = None,
    r_squared: float | None = None,
    linear_range_start: float | None = None,
    linear_range_end: float | None = None,
    poly_fit_y: np.ndarray | None = None,
    extended_fit_y: np.ndarray | None = None,
    polynomial_order: int | None = None,
    piecewise_overlays: list[dict[str, Any]] | None = None,
    header_info: dict[str, Any] | None = None,
    hdu: int | str | None = None,
    ylabel: str = "Mean Signal (ADU)",
    slope_unit: str = "ADU/s",
    xlim: tuple[float, float] | None = None,
    vertical_lines: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> None:
    """
    Plot detector linearity (mean signal vs. exposure time).

    Parameters
    ----------
    exposure_times : np.ndarray
        All exposure time values in seconds.
    mean_signals : np.ndarray
        Corresponding mean signal values (ADU or electrons).
    fit_exposure_times : np.ndarray
        Exposure times used for the linear fit.
    fit_mean_signals : np.ndarray
        Signal values evaluated along the linear fit at
        *fit_exposure_times*.
    title : str
        Plot title.
    output_path : str
        Destination file path.
    slope : float, optional
        Linear fit slope for annotation.
    intercept : float, optional
        Linear fit intercept for annotation.
    r_squared : float, optional
        Coefficient of determination for annotation.
    linear_range_start : float, optional
        Lower signal bound of the linear range (horizontal reference).
    linear_range_end : float, optional
        Upper signal bound of the linear range (horizontal reference).
    poly_fit_y : np.ndarray, optional
        Polynomial fit evaluated at *exposure_times*.
    extended_fit_y : np.ndarray, optional
        Extended linear fit evaluated at *exposure_times*.
    polynomial_order : int, optional
        Degree of the polynomial fit (used to label the curve).
    piecewise_overlays : list of dict, optional
        Additional line overlays for piecewise fits.  Each dict must
        contain ``"x"``, ``"y"``, and optionally ``"label"``,
        ``"color"``, ``"linestyle"``.
    header_info : dict, optional
        Metadata for the subtitle.
    hdu : int or str, optional
        FITS extension identifier for the subtitle.
    ylabel : str, optional
        Y-axis label.  Default ``"Mean Signal (ADU)"``.
    slope_unit : str, optional
        Unit string appended to the slope annotation.
        Default ``"ADU/s"``.
    xlim : tuple of float, optional
        X-axis limits ``(min, max)``.
    vertical_lines : list of dict, optional
        Vertical line specifications passed directly to
        :meth:`~.plot_generator.PlotGenerator.plot`.
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
    _warn_unused_kwargs(remaining, "plot_linearity_curve")
    figure.setdefault("size", (10, 6))

    datasets: list[dict[str, Any]] = [
        {
            "type":   "scatter",
            "x":      np.asarray(exposure_times),
            "y":      np.asarray(mean_signals),
            "marker": "o",
            "color":  "tab:blue",
            "label":  "Measured Data",
            "zorder": 5,
        },
        {
            "type":      "line",
            "x":         np.asarray(fit_exposure_times),
            "y":         np.asarray(fit_mean_signals),
            "linestyle": "-",
            "color":     "tab:green",
            "label":     "Linear Fit",
            "zorder":    4,
        },
    ]

    if poly_fit_y is not None:
        order_str = (
            f"{polynomial_order}{_get_ordinal_suffix(polynomial_order)} Order"
            if polynomial_order
            else "Polynomial"
        )
        datasets.append(
            {
                "type":      "line",
                "x":         np.asarray(exposure_times),
                "y":         np.asarray(poly_fit_y),
                "linestyle": "--",
                "color":     "tab:red",
                "label":     f"{order_str} Poly Fit",
                "zorder":    3,
            }
        )

    if extended_fit_y is not None:
        datasets.append(
            {
                "type":      "line",
                "x":         np.asarray(exposure_times),
                "y":         np.asarray(extended_fit_y),
                "linestyle": ":",
                "color":     "tab:orange",
                "label":     "Extended Linear",
                "zorder":    2,
            }
        )

    if piecewise_overlays:
        for ov in piecewise_overlays:
            datasets.append(
                {
                    "type":      "line",
                    "x":         np.asarray(ov["x"]),
                    "y":         np.asarray(ov["y"]),
                    "linestyle": ov.get("linestyle", "--"),
                    "color":     ov.get("color", "tab:orange"),
                    "label":     ov.get("label", "Piecewise"),
                    "zorder":    2,
                }
            )

    horizontal_lines: list[dict[str, Any]] = []
    if linear_range_start is not None:
        horizontal_lines.append(
            {
                "y":          linear_range_start,
                "color":      "magenta",
                "linestyle":  ":",
                "label":      "Linear start",
                "show_value": False,
            }
        )
    if linear_range_end is not None:
        horizontal_lines.append(
            {
                "y":          linear_range_end,
                "color":      "purple",
                "linestyle":  ":",
                "label":      "Linear end",
                "show_value": False,
            }
        )

    result_parts: list[str] = []
    if slope is not None:
        result_parts.append(f"slope={slope:.3f} {slope_unit}")
    if r_squared is not None:
        result_parts.append(f"R²={r_squared:.4f}")
    result_text = ",  ".join(result_parts) if result_parts else None

    axes: dict[str, Any] = {
        "title":  _merge_title(title, title1, title2),
        "xlabel": "Exposure Time (s)",
        "ylabel": ylabel,
        **axes_extra,
    }
    if xlim is not None:
        axes["xlim"] = xlim

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
            result1=result_text, result1_position="lower right"
        ),
        overwrite=overwrite,
        show=show,
    )
    plt.close("all")
    logger.debug("Linearity curve plot saved to: %s", output_path)


def plot_linearity_residuals(
    exposure_times: np.ndarray,
    residuals_percent: np.ndarray,
    title: str,
    output_path: str,
    threshold_percent: float = 1.0,
    linear_start_time: float | None = None,
    linear_end_time: float | None = None,
    header_info: dict[str, Any] | None = None,
    hdu: int | str | None = None,
    xlabel: str = "Exposure Time (s)",
    ylabel: str = "Deviation from Linear (%)",
    ylim: tuple[float, float] | None = (-5.0, 5.0),
    xlim: tuple[float, float] | None = None,
    **kwargs: Any,
) -> None:
    """
    Plot linearity residuals (percent deviation from fit vs. exposure time).

    Parameters
    ----------
    exposure_times : np.ndarray
        Exposure time values in seconds.
    residuals_percent : np.ndarray
        Percent deviation from the linear fit at each exposure time.
    title : str
        Plot title.
    output_path : str
        Destination file path.
    threshold_percent : float, optional
        Linearity specification threshold; symmetric ± reference lines
        are drawn at this value.  Default 1.0.
    linear_start_time : float, optional
        Exposure time at the lower edge of the linear range (vertical
        reference line).
    linear_end_time : float, optional
        Exposure time at the upper edge of the linear range (vertical
        reference line).
    header_info : dict, optional
        Metadata for the subtitle.
    hdu : int or str, optional
        FITS extension identifier for the subtitle.
    xlabel : str, optional
        X-axis label.  Default ``"Exposure Time (s)"``.
    ylabel : str, optional
        Y-axis label.  Default ``"Deviation from Linear (%)"``.
    ylim : tuple of float, optional
        Y-axis limits.  Default ``(-5.0, 5.0)``.
    xlim : tuple of float, optional
        X-axis limits.
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
    _warn_unused_kwargs(remaining, "plot_linearity_residuals")
    figure.setdefault("size", (10, 6))

    datasets: list[dict[str, Any]] = [
        {
            "type":   "scatter",
            "x":      np.asarray(exposure_times),
            "y":      np.asarray(residuals_percent),
            "marker": "o",
            "color":  "tab:blue",
            "label":  "Residuals",
            "zorder": 5,
        }
    ]

    horizontal_lines: list[dict[str, Any]] = [
        {
            "y":          threshold_percent,
            "color":      "red",
            "linestyle":  "--",
            "label":      f"±{threshold_percent}% threshold",
            "show_value": False,
        },
        {
            "y":          -threshold_percent,
            "color":      "red",
            "linestyle":  "--",
            "show_value": False,
        },
        {
            "y":          0,
            "color":      "black",
            "linestyle":  ":",
            "linewidth":  1,
            "show_value": False,
        },
    ]

    vertical_lines: list[dict[str, Any]] = []
    if linear_start_time is not None:
        vertical_lines.append(
            {
                "x":          linear_start_time,
                "color":      "magenta",
                "linestyle":  ":",
                "label":      "Linear range",
                "show_value": False,
            }
        )
    if linear_end_time is not None:
        vertical_lines.append(
            {
                "x":          linear_end_time,
                "color":      "purple",
                "linestyle":  ":",
                "show_value": False,
            }
        )

    axes: dict[str, Any] = {
        "title":  _merge_title(title, title1, title2),
        "xlabel": xlabel,
        "ylabel": ylabel,
        **axes_extra,
    }
    if ylim is not None:
        axes["ylim"] = ylim
    if xlim is not None:
        axes["xlim"] = xlim

    _PLOTGEN.plot(
        output_path=output_path,
        datasets=datasets,
        axes=axes,
        figure=figure or None,
        legend={"loc": "upper left", **(legend or {})},
        grid=grid or None,
        colorbar=colorbar or None,
        horizontal_lines=horizontal_lines,
        vertical_lines=vertical_lines or None,
        overwrite=overwrite,
        show=show,
    )
    plt.close("all")
    logger.debug("Linearity residuals plot saved to: %s", output_path)


def plot_linearity_residuals_low(
    exposure_times: np.ndarray,
    residuals_global: np.ndarray,
    title: str,
    output_path: str,
    residuals_local: np.ndarray | None = None,
    threshold_percent: float = 1.0,
    slope_global: float | None = None,
    intercept_global: float | None = None,
    slope_local: float | None = None,
    intercept_local: float | None = None,
    header_info: dict[str, Any] | None = None,
    hdu: int | str | None = None,
    ylim: tuple[float, float] | None = (-100.0, 20.0),
    xlim: tuple[float, float] | None = None,
    **kwargs: Any,
) -> None:
    """
    Plot low-signal linearity residuals with optional local-refit comparison.

    Parameters
    ----------
    exposure_times : np.ndarray
        Exposure time values (already windowed to the low-signal region).
    residuals_global : np.ndarray
        Percent deviation from the global linear fit.
    title : str
        Plot title.
    output_path : str
        Destination file path.
    residuals_local : np.ndarray, optional
        Percent deviation from a local (low-signal region) linear fit.
    threshold_percent : float, optional
        Linearity specification threshold for reference lines.
        Default 1.0.
    slope_global : float, optional
        Global fit slope for annotation.
    intercept_global : float, optional
        Global fit intercept for annotation.
    slope_local : float, optional
        Local fit slope for annotation.
    intercept_local : float, optional
        Local fit intercept for annotation.
    header_info : dict, optional
        Metadata for the subtitle.
    hdu : int or str, optional
        FITS extension identifier for the subtitle.
    ylim : tuple of float, optional
        Y-axis limits.  Default ``(-100.0, 20.0)``.
    xlim : tuple of float, optional
        X-axis limits.
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
    _warn_unused_kwargs(remaining, "plot_linearity_residuals_low")
    figure.setdefault("size", (10, 6))

    datasets: list[dict[str, Any]] = [
        {
            "type":   "scatter",
            "x":      np.asarray(exposure_times),
            "y":      np.asarray(residuals_global),
            "marker": "o",
            "color":  "tab:blue",
            "label":  "Residuals vs global fit",
            "zorder": 5,
        }
    ]

    if residuals_local is not None:
        datasets.append(
            {
                "type":   "scatter",
                "x":      np.asarray(exposure_times),
                "y":      np.asarray(residuals_local),
                "marker": "o",
                "color":  "tab:orange",
                "label":  "Residuals vs low-fit",
                "zorder": 4,
            }
        )

    horizontal_lines: list[dict[str, Any]] = [
        {
            "y":          threshold_percent,
            "color":      "red",
            "linestyle":  "--",
            "label":      f"±{threshold_percent}% threshold",
            "show_value": False,
        },
        {
            "y":          -threshold_percent,
            "color":      "red",
            "linestyle":  "--",
            "show_value": False,
        },
        {
            "y":          0,
            "color":      "black",
            "linestyle":  ":",
            "linewidth":  1,
            "show_value": False,
        },
    ]

    result_parts: list[str] = []
    if slope_global is not None and intercept_global is not None:
        result_parts.append(
            f"global: m={slope_global:.3g}, b={intercept_global:.3g}"
        )
    if slope_local is not None and intercept_local is not None:
        result_parts.append(
            f"low10: m={slope_local:.3g}, b={intercept_local:.3g}"
        )
        if slope_global is not None and slope_global != 0:
            gain_ratio = slope_local / slope_global
            result_parts.append(f"(m_low/m_global={gain_ratio:.4f})")
    result_text = "\n".join(result_parts) if result_parts else None

    axes: dict[str, Any] = {
        "title":  _merge_title(title, title1, title2),
        "xlabel": "Exposure Time (s)",
        "ylabel": "Deviation from Linear (%)",
        **axes_extra,
    }
    if ylim is not None:
        axes["ylim"] = ylim
    if xlim is not None:
        axes["xlim"] = xlim

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
            result1=result_text, result1_position="lower right"
        ),
        overwrite=overwrite,
        show=show,
    )
    plt.close("all")
    logger.debug(
        "Linearity residuals (low) plot saved to: %s", output_path
    )
