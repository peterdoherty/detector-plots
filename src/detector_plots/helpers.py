#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared private helpers for the detector plot-recipe modules.

Nothing in this module is part of the public API.  All names are
prefixed with an underscore and should be imported only by sibling
modules within the ``plots`` package.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

# Single authoritative version for the entire plots package.
__version__ = "1.3.0"


def _ensure_parent_dir(output_path: str) -> None:
    """
    Create parent directories for *output_path* if they do not exist.

    Parameters
    ----------
    output_path : str
        Destination file path whose parent directory should exist.
    """
    p = Path(output_path)
    if p.parent and str(p.parent) not in ("", "."):
        p.parent.mkdir(parents=True, exist_ok=True)


def _get_ordinal_suffix(n: int) -> str:
    """
    Return the English ordinal suffix for integer *n*.

    Parameters
    ----------
    n : int
        Non-negative integer.

    Returns
    -------
    str
        One of ``"st"``, ``"nd"``, ``"rd"``, or ``"th"``.
    """
    if 11 <= (n % 100) <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def _merge_title(title0: str, title1: str = "", title2: str = "") -> str:
    """
    Join up to three title lines, omitting any empty parts.

    Parameters
    ----------
    title0 : str
        Primary (top) title line.
    title1 : str, optional
        Secondary title line.
    title2 : str, optional
        Tertiary title line.

    Returns
    -------
    str
        Newline-joined non-empty title parts.
    """
    parts = [p for p in (title0, title1, title2) if p]
    return "\n".join(parts)


def _make_text_boxes(
    result1: str | None = None,
    result2: str | None = None,
    result1_position: str = "upper right",
    result2_position: str = "upper left",
) -> list[dict[str, Any]]:
    """
    Build a list of text-box descriptors for :meth:`PlotGenerator.plot`.

    Parameters
    ----------
    result1 : str, optional
        Content for the first text box (typically metric annotations).
    result2 : str, optional
        Content for the second text box.
    result1_position : str, optional
        Axes-fraction position for *result1*.  One of
        ``"upper right"``, ``"upper left"``, ``"lower right"``,
        ``"lower left"``.  Default ``"upper right"``.
    result2_position : str, optional
        Axes-fraction position for *result2*.  Default ``"upper left"``.

    Returns
    -------
    list of dict
        Ready-to-pass ``text_boxes`` list.
    """
    _positions: dict[str, tuple[float, float, str, str]] = {
        "upper right": (0.98, 0.98, "right", "top"),
        "upper left":  (0.02, 0.98, "left",  "top"),
        "lower right": (0.98, 0.02, "right", "bottom"),
        "lower left":  (0.02, 0.02, "left",  "bottom"),
    }

    boxes: list[dict[str, Any]] = []
    for text, position in (
        (result1, result1_position),
        (result2, result2_position),
    ):
        if not text:
            continue
        px, py, ha, va = _positions.get(position, _positions["upper right"])
        boxes.append(
            {
                "text":      text,
                "xy":        (px, py),
                "ha":        ha,
                "va":        va,
                "transform": "axes",
                "fontsize":  10,
                "facecolor": "white",
                "alpha":     1.0,
            }
        )
    return boxes


def _parse_plot_kwargs(
    kwargs: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    bool,
    bool,
    dict[str, Any],
]:
    """
    Extract recognised layout keyword arguments from a ``**kwargs`` dict.

    Recognised keys are consumed (popped) from *kwargs* in-place.  Any
    keys not listed below are returned in *remaining_kwargs* so callers
    can decide how to handle them.

    Parameters
    ----------
    kwargs : dict
        Raw keyword arguments forwarded from a plot-recipe function.

    Returns
    -------
    figure : dict
        Figure-level configuration (``"size"``, ``"dpi"``).
    axes : dict
        Axes-level configuration (``"xlim"``, ``"ylim"``,
        ``"xscale"``, ``"yscale"``).
    legend : dict
        Legend configuration (``"loc"``, ``"fontsize"``,
        ``"frameon"``, ``"ncol"``).
    grid : dict
        Grid configuration (``"visible"``, ``"linestyle"``,
        ``"color"``, ``"alpha"``).
    colorbar : dict
        Colorbar configuration (``"visible"``, ``"label"``,
        ``"orientation"``, ``"shrink"``).
    overwrite : bool
        Whether to overwrite an existing output file.
    show : bool
        Whether to call ``plt.show()`` after saving.
    remaining_kwargs : dict
        Any kwargs not consumed by this function.
    """
    figure: dict[str, Any] = {}
    axes: dict[str, Any] = {}
    legend: dict[str, Any] = {}
    grid: dict[str, Any] = {}
    colorbar: dict[str, Any] = {}

    overwrite = bool(kwargs.pop("overwrite", True))
    show = bool(kwargs.pop("show", False))

    fx = kwargs.pop("figsize_x", None)
    fy = kwargs.pop("figsize_y", None)
    if fx is not None and fy is not None:
        figure["size"] = (float(fx), float(fy))

    dpi = kwargs.pop("dpi", None)
    if dpi is not None:
        figure["dpi"] = int(dpi)

    for key in ("xlim", "ylim", "xscale", "yscale"):
        if key in kwargs:
            axes[key] = kwargs.pop(key)

    for key in ("loc", "fontsize", "frameon", "ncol"):
        kwkey = f"legend_{key}"
        if kwkey in kwargs:
            legend[key] = kwargs.pop(kwkey)

    for key in ("visible", "linestyle", "color", "alpha"):
        kwkey = f"grid_{key}"
        if kwkey in kwargs:
            grid[key] = kwargs.pop(kwkey)

    for key in ("visible", "label", "orientation", "shrink"):
        kwkey = f"colorbar_{key}"
        if kwkey in kwargs:
            colorbar[key] = kwargs.pop(kwkey)

    return figure, axes, legend, grid, colorbar, overwrite, show, kwargs


def _warn_unused_kwargs(remaining: dict[str, Any], func_name: str) -> None:
    """
    Emit a debug-level warning for any unrecognised keyword arguments.

    Call this at the end of every plot-recipe function after
    :func:`_parse_plot_kwargs` has consumed its recognised keys.

    Parameters
    ----------
    remaining : dict
        Leftover kwargs returned by :func:`_parse_plot_kwargs`.
    func_name : str
        Name of the calling function, used in the log message.
    """
    if remaining:
        logger.debug(
            "%s: ignoring unsupported kwargs: %s",
            func_name,
            sorted(remaining.keys()),
        )


def _build_subtitle(
    header_info: dict[str, Any],
    hdu: int | str | None,
) -> tuple[str, str]:
    """
    Construct first and second subtitle lines from header metadata.

    Parameters
    ----------
    header_info : dict
        Metadata dict with optional keys ``"instrument"``,
        ``"detector"``, ``"date_obs"``, and ``"mode"``.
    hdu : int or str or None
        FITS extension identifier; included in the subtitle when not
        *None*.

    Returns
    -------
    title1 : str
        Instrument / detector / date line (may be empty).
    title2 : str
        Extension / mode line (may be empty).
    """
    instrume = header_info.get("instrument", "")
    detector = header_info.get("detector", "")
    date_obs = header_info.get("date_obs", "")
    mode = header_info.get("mode", "")

    title1 = " | ".join(p for p in (instrume, detector, date_obs) if p)

    parts: list[str] = []
    if hdu is not None:
        parts.append(f"Extension: {hdu}")
    if mode and mode != "UNKNOWN":
        parts.append(f"Mode: {mode}")
    title2 = " | ".join(parts)

    return title1, title2
