#!/usr/bin/env python3
# Copyright (c) 2026, Unitree RL Lab contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Plot recorded robot signals as one time-series panel per component."""

from __future__ import annotations

import math
import os
import pathlib
from xml.sax.saxutils import escape

import numpy as np


GRID_ROWS = 3
GRID_COLS = 10


def shared_series_y_limits(*value_matrices) -> np.ndarray:
    """Return padded per-series y limits shared by all supplied recordings."""
    matrices = [np.asarray(matrix, dtype=float) for matrix in value_matrices]
    if not matrices:
        raise ValueError("At least one torque matrix is required.")
    if any(matrix.ndim != 2 for matrix in matrices):
        raise ValueError("All value matrices must be 2-D.")
    series_count = matrices[0].shape[1]
    if any(matrix.shape[1] != series_count for matrix in matrices):
        raise ValueError("All value matrices must have the same series count.")

    y_min = np.minimum.reduce([np.nanmin(matrix, axis=0) for matrix in matrices])
    y_max = np.maximum.reduce([np.nanmax(matrix, axis=0) for matrix in matrices])
    span = y_max - y_min
    padding = np.maximum(span * 0.08, 0.1)
    flat = span <= 0.0
    y_min = np.where(flat, y_min - 0.5, y_min - padding)
    y_max = np.where(flat, y_max + 0.5, y_max + padding)
    return np.column_stack((y_min, y_max))


def shared_joint_y_limits(*torque_matrices) -> np.ndarray:
    """Backward-compatible torque-specific name."""
    return shared_series_y_limits(*torque_matrices)


def plot_timeseries_grid(
    path: pathlib.Path,
    series_names: list[str],
    times,
    value_matrix,
    *,
    title: str,
    ylabel: str,
    color: str = "#2563eb",
    y_limits=None,
    grid_shape: tuple[int, int] = (GRID_ROWS, GRID_COLS),
) -> str:
    """Plot one time-series panel per component."""
    path = pathlib.Path(path)
    times = np.asarray(times, dtype=float)
    value_matrix = np.asarray(value_matrix, dtype=float)
    grid_rows, grid_cols = grid_shape
    grid_size = grid_rows * grid_cols
    if grid_rows < 1 or grid_cols < 1:
        raise ValueError(f"Grid dimensions must be positive, got {grid_shape}.")
    if not 0 < len(series_names) <= grid_size:
        raise ValueError(f"The {grid_rows}x{grid_cols} grid supports 1 to {grid_size} series.")
    if times.ndim != 1 or value_matrix.shape != (len(times), len(series_names)):
        raise ValueError(
            f"Expected value shape ({len(times)}, {len(series_names)}), got {value_matrix.shape}."
        )
    if y_limits is None:
        y_limits = shared_series_y_limits(value_matrix)
    else:
        y_limits = np.asarray(y_limits, dtype=float)
        if y_limits.shape != (len(series_names), 2):
            raise ValueError(f"Expected y_limits shape ({len(series_names)}, 2), got {y_limits.shape}.")

    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        svg_path = path.with_suffix(".svg")
        _write_svg_grid(
            svg_path,
            series_names,
            times,
            value_matrix,
            title,
            ylabel,
            color,
            y_limits,
            grid_shape,
        )
        print(f"[WARN] matplotlib unavailable ({exc}); wrote SVG grid instead: {svg_path}")
        return str(svg_path)

    figure_width = max(12.0, grid_cols * 2.8)
    figure_height = max(3.8, grid_rows * 3.2 + 0.4)
    fig, axes = plt.subplots(
        grid_rows,
        grid_cols,
        figsize=(figure_width, figure_height),
        sharex=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()
    for series_idx, series_name in enumerate(series_names):
        ax = flat_axes[series_idx]
        ax.plot(times, value_matrix[:, series_idx], color=color, linewidth=0.9)
        ax.axhline(0.0, color="#111827", linewidth=0.6)
        ax.set_ylim(y_limits[series_idx])
        ax.set_title(series_name, fontsize=9)
        ax.grid(color="#e5e7eb", linewidth=0.5, alpha=0.8)
        ax.margins(x=0.0)
        ax.tick_params(axis="both", labelsize=7)

    for ax in flat_axes[len(series_names) :]:
        ax.axis("off")

    fig.suptitle(title, fontsize=15)
    fig.supxlabel("Simulation time (s)", fontsize=12)
    fig.supylabel(ylabel, fontsize=12)
    fig.tight_layout(rect=(0.025, 0.03, 1.0, 0.95))
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def plot_torque_timeseries_grid(
    path: pathlib.Path,
    joint_names: list[str],
    times,
    torque_matrix,
    *,
    title: str,
    color: str = "#2563eb",
    y_limits=None,
) -> str:
    """Plot one torque time-series panel per joint in a 3-by-10 figure."""
    return plot_timeseries_grid(
        path,
        joint_names,
        times,
        torque_matrix,
        title=title,
        ylabel="Applied joint torque (N*m)",
        color=color,
        y_limits=y_limits,
    )


def _write_svg_grid(path, series_names, times, value_matrix, title, ylabel, color, y_limits, grid_shape):
    grid_rows, grid_cols = grid_shape
    width = max(900, grid_cols * 240)
    height = max(360, grid_rows * 270 + 110)
    left = 70
    right = 25
    top = 55
    bottom = 55
    gap_x = 12
    gap_y = 18
    cell_width = (width - left - right - gap_x * (grid_cols - 1)) / grid_cols
    cell_height = (height - top - bottom - gap_y * (grid_rows - 1)) / grid_rows
    x_min = float(np.nanmin(times))
    x_max = float(np.nanmax(times))
    if x_max <= x_min:
        x_max = x_min + 1.0

    sample_stride = max(1, int(math.ceil(len(times) / 1200)))
    sample_indices = list(range(0, len(times), sample_stride))
    if sample_indices[-1] != len(times) - 1:
        sample_indices.append(len(times) - 1)

    with pathlib.Path(path).open("w") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">\n')
        f.write('<rect width="100%" height="100%" fill="white"/>\n')
        f.write(
            f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" '
            f'font-size="20" font-family="sans-serif">{escape(title)}</text>\n'
        )

        for series_idx, series_name in enumerate(series_names):
            row = series_idx // grid_cols
            col = series_idx % grid_cols
            cell_x = left + col * (cell_width + gap_x)
            cell_y = top + row * (cell_height + gap_y)
            plot_left = cell_x + 38
            plot_right = cell_x + cell_width - 6
            plot_top = cell_y + 22
            plot_bottom = cell_y + cell_height - 25
            y_min, y_max = y_limits[series_idx]

            def scale_x(value):
                return plot_left + (float(value) - x_min) / (x_max - x_min) * (plot_right - plot_left)

            def scale_y(value):
                return plot_top + (y_max - float(value)) / (y_max - y_min) * (plot_bottom - plot_top)

            f.write(
                f'<text x="{cell_x + cell_width / 2:.1f}" y="{cell_y + 12:.1f}" text-anchor="middle" '
                f'font-size="11" font-family="sans-serif">{escape(series_name)}</text>\n'
            )
            f.write(
                f'<rect x="{plot_left:.1f}" y="{plot_top:.1f}" '
                f'width="{plot_right - plot_left:.1f}" height="{plot_bottom - plot_top:.1f}" '
                'fill="none" stroke="#9ca3af" stroke-width="0.7"/>\n'
            )

            for tick_idx in range(3):
                fraction = tick_idx / 2
                grid_x = plot_left + fraction * (plot_right - plot_left)
                grid_y = plot_top + fraction * (plot_bottom - plot_top)
                x_value = x_min + fraction * (x_max - x_min)
                y_value = y_max - fraction * (y_max - y_min)
                f.write(
                    f'<line x1="{grid_x:.1f}" y1="{plot_top:.1f}" x2="{grid_x:.1f}" '
                    f'y2="{plot_bottom:.1f}" stroke="#e5e7eb" stroke-width="0.7"/>\n'
                )
                f.write(
                    f'<line x1="{plot_left:.1f}" y1="{grid_y:.1f}" x2="{plot_right:.1f}" '
                    f'y2="{grid_y:.1f}" stroke="#e5e7eb" stroke-width="0.7"/>\n'
                )
                f.write(
                    f'<text x="{grid_x:.1f}" y="{plot_bottom + 15:.1f}" text-anchor="middle" '
                    f'font-size="8" font-family="sans-serif">{x_value:.1f}</text>\n'
                )
                f.write(
                    f'<text x="{plot_left - 4:.1f}" y="{grid_y + 3:.1f}" text-anchor="end" '
                    f'font-size="8" font-family="sans-serif">{y_value:.1f}</text>\n'
                )

            if y_min <= 0.0 <= y_max:
                zero_y = scale_y(0.0)
                f.write(
                    f'<line x1="{plot_left:.1f}" y1="{zero_y:.1f}" x2="{plot_right:.1f}" '
                    f'y2="{zero_y:.1f}" stroke="#111827" stroke-width="0.7"/>\n'
                )

            points = " ".join(
                f"{scale_x(times[index]):.1f},{scale_y(value_matrix[index, series_idx]):.1f}"
                for index in sample_indices
            )
            f.write(
                f'<polyline points="{points}" fill="none" stroke="{color}" '
                'stroke-width="1.0"/>\n'
            )

        f.write(
            f'<text x="{width / 2:.1f}" y="{height - 12}" text-anchor="middle" '
            'font-size="13" font-family="sans-serif">Simulation time (s)</text>\n'
        )
        f.write(
            f'<text x="16" y="{height / 2:.1f}" text-anchor="middle" '
            f'transform="rotate(-90 16 {height / 2:.1f})" '
            f'font-size="13" font-family="sans-serif">{escape(ylabel)}</text>\n'
        )
        f.write("</svg>\n")
