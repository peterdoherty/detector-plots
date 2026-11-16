#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"
"""
Image and pixel-distribution plot recipes.

Covers two related groups of diagnostics:

**2-D spatial images**
    :func:`plot_image`, :func:`plot_superbias_image`,
    :func:`plot_deviation_map`, :func:`plot_2d_histogram`,
    :func:`plot_clipped_mask`

**1-D pixel statistics and profiles**
    :func:`plot_histogram_log`, :func:`plot_noise_histogram`,
    :func:`plot_row_profile`, :func:`plot_column_profile`,
    :func:`plot_radial_profile`, :func:`plot_column_means`,
    :func:`plot_row_medians`, :func:`plot_full_well_vs_gain`
"""

from __future__ import annotations

import logging
from pathlib import Path
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


# ---------------------------------------------------------------------------
# 2-D spatial image plots
# ---------------------------------------------------------------------------

def plot_image(
    data: np.ndarray,
    title: str,
    output_path: str,
    cmap: str = "viridis",
    **kwargs: Any,
) -> None:
    """
    Render a 2-D image plot from a pixel array.

    Parameters
    ----------
    data : np.ndarray
        2-D pixel array.
    title : str
        Plot title.
    output_path : str
        Destination file path.
    cmap : str, optional
        Matplotlib colourmap name.  Default ``"viridis"``.
    **kwargs
        ``"origin"``, ``"aspect"``, ``"dataset_colorbar"``, ``"vmin"``,
        and ``"vmax"`` are consumed directly; remaining keys are
        forwarded to :func:`._helpers._parse_plot_kwargs`.

    Raises
    ------
    ValueError
        If *data* is not 2-D.
    """
    _ensure_parent_dir(output_path)

    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("plot_image expects a 2D array.")

    mean = float(np.mean(data))
    std_dev = float(np.std(data))
    result_text = f"Mean: {mean:.2f}  StdDev: {std_dev:.2f}"

    origin = kwargs.pop("origin", "lower")
    aspect = kwargs.pop("aspect", "auto")
    ds_colorbar = kwargs.pop("dataset_colorbar", True)
    vmin = kwargs.pop("vmin", None)
    vmax = kwargs.pop("vmax", None)

    figure, axes_extra, legend, grid, colorbar, overwrite, show, remaining = (
        _parse_plot_kwargs(kwargs)
    )
    _warn_unused_kwargs(remaining, "plot_image")

    datasets: list[dict[str, Any]] = [
        {
            "type":     "imshow",
            "y":        data,
            "cmap":     cmap,
            "origin":   origin,
            "aspect":   aspect,
            "colorbar": ds_colorbar,
            "vmin":     vmin,
            "vmax":     vmax,
        }
    ]

    axes: dict[str, Any] = {"title": title, **axes_extra}

    # Disable plot-level colorbar to avoid duplicating the per-dataset one.
    if not colorbar:
        colorbar = {"visible": False}

    _PLOTGEN.plot(
        output_path=output_path,
        datasets=datasets,
        axes=axes,
        figure=figure or None,
        legend=legend or None,
        grid=grid or None,
        colorbar=colorbar,
        text_boxes=_make_text_boxes(result1=result_text),
        overwrite=overwrite,
        show=show,
    )
    plt.close("all")
    logger.debug("Image plot saved to: %s", output_path)


def plot_superbias_image(
    data: np.ndarray,
    title: str,
    output_path: str,
    vmin: float | None = None,
    vmax: float | None = None,
    use_tight_scale: bool = True,
    **kwargs: Any,
) -> None:
    """
    Plot a 2-D image of a bias or dark calibration frame.

    Parameters
    ----------
    data : np.ndarray
        2-D pixel array.
    title : str
        Plot title.
    output_path : str
        Destination file path.
    vmin : float, optional
        Lower colour-scale limit.  Derived automatically when *None*
        (see *use_tight_scale*).
    vmax : float, optional
        Upper colour-scale limit.
    use_tight_scale : bool, optional
        If *True*, set limits to mean ± 3σ.  If *False*, use the
        0.5–99.5 percentile range.  Ignored when both *vmin* and
        *vmax* are provided.  Default *True*.
    **kwargs
        Additional layout keywords forwarded to
        :func:`._helpers._parse_plot_kwargs`.
    """
    _ensure_parent_dir(output_path)

    mean = float(np.mean(data))
    std = float(np.std(data))

    if vmin is None or vmax is None:
        if use_tight_scale:
            vmin, vmax = mean - 3 * std, mean + 3 * std
        else:
            vmin = float(np.percentile(data, 0.5))
            vmax = float(np.percentile(data, 99.5))

    figure, axes_extra, legend, grid, colorbar, overwrite, show, remaining = (
        _parse_plot_kwargs(kwargs)
    )
    _warn_unused_kwargs(remaining, "plot_superbias_image")

    datasets: list[dict[str, Any]] = [
        {
            "type":   "imshow",
            "y":      data,
            "cmap":   "viridis",
            "aspect": "auto",
            # The plot-level colorbar below is authoritative.  Suppress the
            # PlotGenerator's per-dataset default to avoid two colorbars.
            "colorbar": False,
            "vmin":   vmin,
            "vmax":   vmax,
        }
    ]
    tboxes = _make_text_boxes(
        result1=f"Mean: {mean:.2f} ADU\nStd Dev: {std:.2f} ADU",
        result2=f"Scale: [{vmin:.2f}, {vmax:.2f}] ADU",
        result2_position="lower left",
    )

    _PLOTGEN.plot(
        output_path=output_path,
        datasets=datasets,
        text_boxes=tboxes,
        colorbar={
            "visible": True,
            "label":   "Signal Level (ADU)",
            **(colorbar or {}),
        },
        axes={
            "title":  title,
            "xlabel": "X",
            "ylabel": "Y",
            **axes_extra,
        },
        figure=figure or {"size": (10, 8)},
        overwrite=overwrite,
        show=show,
    )
    plt.close("all")
    logger.debug("Superbias image plot saved to: %s", output_path)


def plot_deviation_map(
    deviation_data: np.ndarray,
    title: str,
    output_path: str,
    vmin: float | None = None,
    vmax: float | None = None,
    percentile_clip: float = 99.5,
    **kwargs: Any,
) -> None:
    """
    Plot a 2-D map of pixel deviations from median/mean.

    The colour scale is symmetric about zero and clipped at the
    *percentile_clip* percentile of the absolute deviation.

    Parameters
    ----------
    deviation_data : np.ndarray
        2-D array of per-pixel deviations.
    title : str
        Plot title.
    output_path : str
        Destination file path.
    vmin : float, optional
        Explicit lower colour-scale limit.  Derived from
        *percentile_clip* when *None*.
    vmax : float, optional
        Explicit upper colour-scale limit.
    percentile_clip : float, optional
        Percentile of the absolute deviation used to set the automatic
        symmetric colour scale.  Default 99.5.
    **kwargs
        Additional layout keywords forwarded to
        :func:`._helpers._parse_plot_kwargs`.
    """
    _ensure_parent_dir(output_path)

    median_dev = float(np.median(deviation_data))
    std_dev = float(np.std(deviation_data))

    if vmin is None or vmax is None:
        abs_max = float(
            np.percentile(np.abs(deviation_data), percentile_clip)
        )
        vmin, vmax = -abs_max, abs_max
    else:
        abs_max = max(abs(vmin), abs(vmax))

    rms = float(np.sqrt(np.mean(deviation_data ** 2)))
    pct_in_range = (
        100.0 * np.sum(np.abs(deviation_data) <= abs_max) / deviation_data.size
    )

    textstr = (
        f"Median: {median_dev:.4f}\n"
        f"Std: {std_dev:.4f}\n"
        f"RMS: {rms:.4f}\n"
        f"Scale: ±{abs_max:.4f}"
    )
    scale_info = f"{pct_in_range:.1f}% of pixels in range"

    figure, axes_extra, legend, grid, colorbar, overwrite, show, remaining = (
        _parse_plot_kwargs(kwargs)
    )
    _warn_unused_kwargs(remaining, "plot_deviation_map")

    datasets: list[dict[str, Any]] = [
        {
            "type":   "imshow",
            "y":      deviation_data,
            "cmap":   "RdBu_r",
            "aspect": "auto",
            "colorbar": False,
            "vmin":   vmin,
            "vmax":   vmax,
        }
    ]
    tboxes = _make_text_boxes(
        result1=textstr,
        result2=scale_info,
        result1_position="upper left",
        result2_position="lower right",
    )

    _PLOTGEN.plot(
        output_path=output_path,
        datasets=datasets,
        text_boxes=tboxes,
        colorbar={
            "visible": True,
            "label":   "Deviation from Unity",
            **(colorbar or {}),
        },
        axes={
            "title":  title,
            "xlabel": "Column",
            "ylabel": "Row",
            **axes_extra,
        },
        figure=figure or {"size": (12, 9)},
        overwrite=overwrite,
        show=show,
    )
    plt.close("all")
    logger.debug("Deviation map plot saved to: %s", output_path)


def plot_2d_histogram(
    data: np.ndarray,
    title: str,
    output_path: str,
    bins: int = 50,
    colorbar_label: str = "Signal Level (ADU)",
    **kwargs: Any,
) -> None:
    """
    Plot a 2-D spatial map of signal levels (downsampled to *bins* × *bins*).

    Parameters
    ----------
    data : np.ndarray
        2-D pixel array with shape ``(rows, cols)``.
    title : str
        Plot title.
    output_path : str
        Destination file path.
    bins : int, optional
        Target grid size for downsampling.  The image is rescaled so
        that neither dimension exceeds *bins* pixels.  Default 50.
    colorbar_label : str, optional
        Colorbar axis label.  Default ``"Signal Level (ADU)"``.
    **kwargs
        Additional layout keywords forwarded to
        :func:`._helpers._parse_plot_kwargs`.
    """
    _ensure_parent_dir(output_path)

    ny, nx = data.shape
    if nx > bins or ny > bins:
        from scipy.ndimage import zoom
        factor = min(bins / ny, bins / nx)
        data_downsampled = zoom(data, factor, order=1)
    else:
        data_downsampled = data

    figure, axes_extra, legend, grid, colorbar, overwrite, show, remaining = (
        _parse_plot_kwargs(kwargs)
    )
    _warn_unused_kwargs(remaining, "plot_2d_histogram")

    datasets: list[dict[str, Any]] = [
        {
            "type":   "imshow",
            "y":      data_downsampled,
            "cmap":   "viridis",
            "aspect": "auto",
            "colorbar": False,
        }
    ]
    colorbar_config: dict[str, Any] = {
        "visible": True,
        "label":   colorbar_label,
        **(colorbar or {}),
    }

    _PLOTGEN.plot(
        output_path=output_path,
        datasets=datasets,
        colorbar=colorbar_config,
        axes={
            "title":  title,
            "xlabel": "Column (binned)",
            "ylabel": "Row (binned)",
            **axes_extra,
        },
        figure=figure or {"size": (10, 8)},
        overwrite=overwrite,
        show=show,
    )
    plt.close("all")
    logger.debug("2D histogram plot saved to: %s", output_path)


def plot_clipped_mask(
    mask_data: np.ndarray,
    title: str,
    output_path: str,
    **kwargs: Any,
) -> None:
    """
    Plot a 2-D map of the sigma-clipped bad-pixel mask.

    Parameters
    ----------
    mask_data : np.ndarray
        2-D boolean or integer array where non-zero values indicate
        clipped (bad) pixels.
    title : str
        Plot title.
    output_path : str
        Destination file path.
    **kwargs
        ``"cmap"``, ``"origin"``, ``"aspect"``, and
        ``"dataset_colorbar"`` are consumed directly; remaining keys
        are forwarded to :func:`._helpers._parse_plot_kwargs`.
    """
    _ensure_parent_dir(output_path)

    mask_data = np.asarray(mask_data)
    num_clipped = int(np.sum(mask_data))
    total_pixels = int(mask_data.size)
    percent_clipped = (num_clipped / total_pixels) * 100.0
    result1 = f"Clipped Pixels: {num_clipped} ({percent_clipped:.4f}%)"

    cmap = kwargs.pop("cmap", "viridis")
    origin = kwargs.pop("origin", "lower")
    aspect = kwargs.pop("aspect", "auto")
    ds_colorbar = kwargs.pop("dataset_colorbar", True)

    figure, axes_extra, legend, grid, colorbar, overwrite, show, remaining = (
        _parse_plot_kwargs(kwargs)
    )
    _warn_unused_kwargs(remaining, "plot_clipped_mask")

    datasets: list[dict[str, Any]] = [
        {
            "type":     "imshow",
            "y":        mask_data,
            "cmap":     cmap,
            "origin":   origin,
            "aspect":   aspect,
            "colorbar": ds_colorbar,
        }
    ]

    axes: dict[str, Any] = {"title": title, **axes_extra}
    if not colorbar:
        colorbar = {"visible": False}

    _PLOTGEN.plot(
        output_path=output_path,
        datasets=datasets,
        axes=axes,
        figure=figure or None,
        legend=legend or None,
        grid=grid or None,
        colorbar=colorbar,
        text_boxes=_make_text_boxes(result1=result1),
        overwrite=overwrite,
        show=show,
    )
    plt.close("all")
    logger.debug("Clipped mask plot saved to: %s", output_path)


# ---------------------------------------------------------------------------
# 1-D pixel statistics and profile plots
# ---------------------------------------------------------------------------

def plot_histogram_log(
    data: np.ndarray,
    title: str,
    output_path: str,
    bins: int = 200,
    percentile_range: tuple[float, float] = (0.1, 99.9),
    xlabel: str = "Signal Level (ADU)",
    **kwargs: Any,
) -> None:
    """
    Plot a pixel-value histogram with a logarithmic y-axis.

    Data are clipped to the specified percentile range before binning
    to prevent sparse tails from dominating the x-axis scale.

    Parameters
    ----------
    data : np.ndarray
        Pixel data array (any shape; flattened internally).
    title : str
        Plot title.
    output_path : str
        Destination file path.
    bins : int, optional
        Number of histogram bins.  Default 200.
    percentile_range : tuple of float, optional
        ``(lower, upper)`` percentile clip limits.
        Default ``(0.1, 99.9)``.
    xlabel : str, optional
        X-axis label.  Default ``"Signal Level (ADU)"``.
    **kwargs
        Additional layout keywords forwarded to
        :func:`._helpers._parse_plot_kwargs`.
    """
    _ensure_parent_dir(output_path)

    median = float(np.median(data))
    mean = float(np.mean(data))
    std = float(np.std(data))
    vmin, vmax = np.percentile(data, list(percentile_range))
    filtered_data = data[(data >= vmin) & (data <= vmax)]

    figure, axes_extra, legend, grid, colorbar, overwrite, show, remaining = (
        _parse_plot_kwargs(kwargs)
    )
    _warn_unused_kwargs(remaining, "plot_histogram_log")

    datasets: list[dict[str, Any]] = [
        {
            "type":  "hist",
            "y":     filtered_data,
            "bins":  bins,
            "color": "steelblue",
            "alpha": 0.7,
        }
    ]
    vlines: list[dict[str, Any]] = [
        {
            "x": median, "color": "red", "linestyle": "-",
            "linewidth": 2, "label": "Median", "show_value": False,
        },
        {
            "x": mean, "color": "darkred", "linestyle": "--",
            "linewidth": 1.5, "label": "Mean", "show_value": False,
        },
        {
            "x": mean - std, "color": "orange", "linestyle": ":",
            "linewidth": 1, "label": "±1σ", "show_value": False,
        },
        {
            "x": mean + std, "color": "orange", "linestyle": ":",
            "linewidth": 1, "show_value": False,
        },
        {
            "x": mean - 3 * std, "color": "yellow", "linestyle": ":",
            "linewidth": 1, "label": "±3σ", "show_value": False,
        },
        {
            "x": mean + 3 * std, "color": "yellow", "linestyle": ":",
            "linewidth": 1, "show_value": False,
        },
    ]

    textstr = (
        f"Median: {median:.2f} ADU\n"
        f"Mean: {mean:.2f} ADU\n"
        f"Std: {std:.2f} ADU"
    )

    _PLOTGEN.plot(
        output_path=output_path,
        datasets=datasets,
        vertical_lines=vlines,
        axes={
            "title":  title,
            "xlabel": xlabel,
            "ylabel": "Number of Pixels (log scale)",
            "yscale": "log",
            **axes_extra,
        },
        text_boxes=_make_text_boxes(result1=textstr),
        figure=figure or None,
        overwrite=overwrite,
        show=show,
    )
    plt.close("all")
    logger.debug("Log histogram plot saved to: %s", output_path)


def plot_noise_histogram(
    data: np.ndarray,
    title: str,
    output_path: str,
    bins: int = 100,
    show_gaussian_fit: bool = True,
    percentile_range: tuple[float, float] | None = None,
    **kwargs: Any,
) -> None:
    """
    Plot a pixel-value histogram with an optional Gaussian fit overlay.

    Parameters
    ----------
    data : np.ndarray
        Pixel data array (any shape; flattened internally).
    title : str
        Plot title.
    output_path : str
        Destination file path.
    bins : int, optional
        Number of histogram bins.  Default 100.
    show_gaussian_fit : bool, optional
        If *True*, overlay a fitted Gaussian curve and mark its mean.
        Default *True*.
    percentile_range : tuple of float, optional
        ``(lower, upper)`` percentile clip limits applied before
        binning.  No clipping when *None*.
    **kwargs
        Additional layout keywords forwarded to
        :func:`._helpers._parse_plot_kwargs`.
    """
    _ensure_parent_dir(output_path)

    if percentile_range is not None:
        lo, hi = np.percentile(data, list(percentile_range))
        data = data[(data >= lo) & (data <= hi)]

    mean = float(np.mean(data))
    std = float(np.std(data))

    figure, axes_extra, legend, grid, colorbar, overwrite, show, remaining = (
        _parse_plot_kwargs(kwargs)
    )
    _warn_unused_kwargs(remaining, "plot_noise_histogram")

    datasets: list[dict[str, Any]] = [
        {
            "type":     "hist",
            "y":        data,
            "bins":     bins,
            "color":    "steelblue",
            "alpha":    0.7,
            "label":    "Data",
            "show_fit": show_gaussian_fit,
        }
    ]
    vlines: list[dict[str, Any]] = [
        {
            "x": mean, "color": "red", "linestyle": ":",
            "alpha": 0.5, "show_value": False,
        },
        {
            "x": mean - std, "color": "orange", "linestyle": ":",
            "alpha": 0.5, "show_value": False,
        },
        {
            "x": mean + std, "color": "orange", "linestyle": ":",
            "alpha": 0.5, "show_value": False,
        },
    ]

    textstr = (
        f"Mean: {mean:.2f} ADU\n"
        f"Std Dev: {std:.2f} ADU\n"
        f"Pixels: {len(data):,}"
    )

    _PLOTGEN.plot(
        output_path=output_path,
        datasets=datasets,
        vertical_lines=vlines,
        axes={
            "title":  title,
            "xlabel": "Signal Level (ADU)",
            "ylabel": "Number of Pixels",
            **axes_extra,
        },
        text_boxes=_make_text_boxes(result1=textstr),
        figure=figure or None,
        overwrite=overwrite,
        show=show,
    )
    plt.close("all")
    logger.debug("Noise histogram plot saved to: %s", output_path)


def plot_row_profile(
    data: np.ndarray,
    title: str,
    output_path: str,
    ylabel: str = "Signal Level (ADU)",
    show_both: bool = True,
    **kwargs: Any,
) -> None:
    """
    Plot row-averaged signal profiles (median and/or mean vs. row number).

    Parameters
    ----------
    data : np.ndarray
        2-D pixel array with shape ``(rows, cols)``.
    title : str
        Plot title.
    output_path : str
        Destination file path.
    ylabel : str, optional
        Y-axis label.  Default ``"Signal Level (ADU)"``.
    show_both : bool, optional
        If *True*, overlay both median and mean row profiles.
        If *False*, plot only the median.  Default *True*.
    **kwargs
        Additional layout keywords forwarded to
        :func:`._helpers._parse_plot_kwargs`.
    """
    _ensure_parent_dir(output_path)

    row_medians = np.median(data, axis=1)
    row_means = np.mean(data, axis=1)
    rows = np.arange(len(row_medians))

    datasets: list[dict[str, Any]] = [
        {
            "type":      "line",
            "x":         rows,
            "y":         row_medians,
            "color":     "steelblue",
            "linewidth": 1.5,
            "label":     "Median",
        }
    ]
    if show_both:
        datasets.append(
            {
                "type":      "line",
                "x":         rows,
                "y":         row_means,
                "color":     "coral",
                "linewidth": 1.5,
                "alpha":     0.7,
                "label":     "Mean",
            }
        )

    global_median = float(np.median(data))
    hlines: list[dict[str, Any]] = [
        {
            "y":         global_median,
            "color":     "gray",
            "linestyle": "--",
            "alpha":     0.5,
            "label":     "Global Median",
            "value_fmt": "{:.2f}",
        }
    ]

    figure, axes_extra, legend, grid, colorbar, overwrite, show, remaining = (
        _parse_plot_kwargs(kwargs)
    )
    _warn_unused_kwargs(remaining, "plot_row_profile")

    _PLOTGEN.plot(
        output_path=output_path,
        datasets=datasets,
        horizontal_lines=hlines,
        axes={
            "title":  title,
            "xlabel": "Row Number",
            "ylabel": ylabel,
            **axes_extra,
        },
        figure=figure or None,
        overwrite=overwrite,
        show=show,
    )
    plt.close("all")
    logger.debug("Row profile plot saved to: %s", output_path)


def plot_column_profile(
    data: np.ndarray,
    title: str,
    output_path: str,
    ylabel: str = "Normalized Response",
    show_both: bool = True,
    **kwargs: Any,
) -> None:
    """
    Plot column-averaged signal profiles (median and/or mean vs. column number).

    Parameters
    ----------
    data : np.ndarray
        2-D pixel array with shape ``(rows, cols)``.
    title : str
        Plot title.
    output_path : str
        Destination file path.
    ylabel : str, optional
        Y-axis label.  Default ``"Normalized Response"``.
    show_both : bool, optional
        If *True*, overlay both median and mean column profiles.
        If *False*, plot only the median.  Default *True*.
    **kwargs
        Additional layout keywords forwarded to
        :func:`._helpers._parse_plot_kwargs`.
    """
    _ensure_parent_dir(output_path)

    col_medians = np.median(data, axis=0)
    col_means = np.mean(data, axis=0)
    cols = np.arange(len(col_medians))

    datasets: list[dict[str, Any]] = [
        {
            "type":      "line",
            "x":         cols,
            "y":         col_medians,
            "color":     "steelblue",
            "linewidth": 1.5,
            "label":     "Median",
        }
    ]
    if show_both:
        datasets.append(
            {
                "type":      "line",
                "x":         cols,
                "y":         col_means,
                "color":     "coral",
                "linewidth": 1.5,
                "alpha":     0.7,
                "label":     "Mean",
            }
        )

    global_median = float(np.median(data))
    hlines: list[dict[str, Any]] = [
        {
            "y":         global_median,
            "color":     "gray",
            "linestyle": "--",
            "alpha":     0.5,
            "label":     "Global Median",
            "value_fmt": "{:.2f}",
        }
    ]

    figure, axes_extra, legend, grid, colorbar, overwrite, show, remaining = (
        _parse_plot_kwargs(kwargs)
    )
    _warn_unused_kwargs(remaining, "plot_column_profile")

    _PLOTGEN.plot(
        output_path=output_path,
        datasets=datasets,
        horizontal_lines=hlines,
        axes={
            "title":  title,
            "xlabel": "Column Number",
            "ylabel": ylabel,
            **axes_extra,
        },
        figure=figure or None,
        overwrite=overwrite,
        show=show,
    )
    plt.close("all")
    logger.debug("Column profile plot saved to: %s", output_path)


def plot_radial_profile(
    radius: np.ndarray,
    median: np.ndarray,
    std: np.ndarray,
    title: str,
    output_path: str,
    ylabel: str = "Normalized Response",
    **kwargs: Any,
) -> None:
    """
    Plot the radial median profile with a ±1σ uncertainty band.

    Parameters
    ----------
    radius : np.ndarray
        Radial distance in pixels for each annular bin.
    median : np.ndarray
        Median normalised response in each bin.
    std : np.ndarray
        Standard deviation of the normalised response in each bin.
    title : str
        Plot title.
    output_path : str
        Destination file path.
    ylabel : str, optional
        Y-axis label.  Default ``"Normalized Response"``.
    **kwargs
        Additional layout keywords forwarded to
        :func:`._helpers._parse_plot_kwargs`.
    """
    _ensure_parent_dir(output_path)

    finite_mask = np.isfinite(median)
    radius = radius[finite_mask]
    median = median[finite_mask]
    std = std[finite_mask]

    figure, axes_extra, legend, grid, colorbar, overwrite, show, remaining = (
        _parse_plot_kwargs(kwargs)
    )
    _warn_unused_kwargs(remaining, "plot_radial_profile")

    datasets: list[dict[str, Any]] = [
        {
            "type":  "line",
            "x":     radius,
            "y":     median,
            "color": "steelblue",
            "label": "Median",
        }
    ]
    bands: list[dict[str, Any]] = [
        {
            "x":     radius,
            "y1":    median - std,
            "y2":    median + std,
            "color": "steelblue",
            "alpha": 0.2,
            "label": "±1σ",
        }
    ]

    _PLOTGEN.plot(
        output_path=output_path,
        datasets=datasets,
        error_bands=bands,
        axes={
            "title":  title,
            "xlabel": "Radius (pixels)",
            "ylabel": ylabel,
            **axes_extra,
        },
        figure=figure or None,
        overwrite=overwrite,
        show=show,
    )
    plt.close("all")
    logger.debug("Radial profile plot saved to: %s", output_path)


def plot_column_means(
    data: np.ndarray,
    title: str,
    output_path: str,
) -> None:
    """
    Plot the per-column mean for a 2-D pixel array.

    Parameters
    ----------
    data : np.ndarray
        2-D pixel array with shape ``(rows, cols)``.
    title : str
        Plot title.
    output_path : str
        Destination file path.

    Raises
    ------
    ValueError
        If *data* is not 2-D.
    """
    _ensure_parent_dir(output_path)

    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("plot_column_means expects a 2D array.")

    col_means = np.mean(data, axis=0)
    result_text = (
        f"Mean: {np.mean(col_means):.2f}  "
        f"StdDev: {np.std(col_means):.2f}"
    )

    datasets: list[dict[str, Any]] = [
        {
            "type":      "line",
            "y":         col_means,
            "marker":    "",
            "linestyle": "-",
            "label":     "Column Mean",
        }
    ]

    _PLOTGEN.plot(
        output_path=output_path,
        datasets=datasets,
        axes={
            "title":  title,
            "xlabel": "Column Index",
            "ylabel": "Mean Pixel Value",
        },
        text_boxes=_make_text_boxes(result1=result_text),
    )
    plt.close("all")
    logger.debug("Column means plot saved to: %s", output_path)


def plot_row_medians(
    data: np.ndarray,
    title: str,
    output_path: str,
) -> None:
    """
    Plot the per-row median for a 2-D pixel array.

    Parameters
    ----------
    data : np.ndarray
        2-D pixel array with shape ``(rows, cols)``.
    title : str
        Plot title.
    output_path : str
        Destination file path.

    Raises
    ------
    ValueError
        If *data* is not 2-D.
    """
    _ensure_parent_dir(output_path)

    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("plot_row_medians expects a 2D array.")

    row_medians = np.median(data, axis=1)
    result_text = (
        f"Median: {np.median(row_medians):.2f}  "
        f"StdDev: {np.std(row_medians):.2f}"
    )

    datasets: list[dict[str, Any]] = [
        {
            "type":      "line",
            "y":         row_medians,
            "marker":    "",
            "linestyle": "-",
            "label":     "Row Median",
        }
    ]

    _PLOTGEN.plot(
        output_path=output_path,
        datasets=datasets,
        axes={
            "title":  title,
            "xlabel": "Row Index",
            "ylabel": "Median Pixel Value",
        },
        text_boxes=_make_text_boxes(result1=result_text),
    )
    plt.close("all")
    logger.debug("Row medians plot saved to: %s", output_path)


def plot_full_well_vs_gain(
    gain_labels: list[str],
    full_well_values: np.ndarray,
    output_path: str,
    header_info: dict[str, Any] | None = None,
    hdu: int | str | None = None,
    **kwargs: Any,
) -> None:
    """
    Plot full-well capacity vs. gain setting.

    Because the gain axis carries string labels, this function renders
    the figure via :meth:`~.plot_generator.PlotGenerator.plot` without
    saving, then manually applies tick labels before writing the file.

    Parameters
    ----------
    gain_labels : list of str
        Ordered gain-setting labels for the x-axis ticks.
    full_well_values : np.ndarray
        Full-well capacity in electrons for each gain setting.
    output_path : str
        Destination file path.
    header_info : dict, optional
        Metadata for the subtitle.  Recognised keys: ``"instrument"``,
        ``"detector"``, ``"date_obs"``, ``"mode"``.
    hdu : int or str, optional
        FITS extension identifier for the subtitle.
    **kwargs
        Additional layout keywords forwarded to
        :func:`._helpers._parse_plot_kwargs`.
    """
    _ensure_parent_dir(output_path)

    header_info = header_info or {}
    title1, _ = _build_subtitle(header_info, hdu)

    logger.debug(
        "plot_full_well_vs_gain: %d points", len(full_well_values)
    )

    x = np.arange(len(gain_labels), dtype=float)
    y = np.asarray(full_well_values)

    figure, axes_extra, legend, grid, colorbar, overwrite, show, remaining = (
        _parse_plot_kwargs(kwargs)
    )
    _warn_unused_kwargs(remaining, "plot_full_well_vs_gain")

    datasets: list[dict[str, Any]] = [
        {
            "type":      "line",
            "x":         x,
            "y":         y,
            "marker":    "o",
            "linestyle": "-",
            "label":     "Full Well",
        }
    ]

    axes: dict[str, Any] = {
        "title":  _merge_title(
            "Full Well Capacity vs. Gain Setting", title1, ""
        ),
        "xlabel": "Gain Setting",
        "ylabel": "Full Well Capacity (e⁻)",
        **axes_extra,
    }

    # Render without saving so we can apply string tick labels manually.
    _PLOTGEN.plot(
        output_path=None,
        datasets=datasets,
        axes=axes,
        figure=figure or None,
        legend=legend or None,
        grid=grid or None,
        colorbar=colorbar or None,
        overwrite=overwrite,
        show=False,
    )

    fig = plt.gcf()
    ax = plt.gca()
    ax.set_xticks(np.arange(len(gain_labels)))
    ax.set_xticklabels(gain_labels, rotation=45, ha="right")
    fig.tight_layout()

    outp = Path(output_path)
    if outp.exists() and not overwrite:
        plt.close("all")
        raise FileExistsError(
            f"Refusing to overwrite existing file: {output_path}"
        )

    fig.savefig(
        output_path,
        dpi=figure.get("dpi") if figure else None,
        bbox_inches="tight",
    )
    if show:
        plt.show()
    plt.close("all")
    logger.debug("Full well vs gain plot saved to: %s", output_path)
