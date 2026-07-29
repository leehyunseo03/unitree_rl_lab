#!/usr/bin/env python3
# Copyright (c) 2026, Unitree RL Lab contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Compare two torque_summary.csv files with one overlaid range graph."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
from datetime import datetime


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = REPO_ROOT / "metrics" / "shared_runs"

from g1_effort_limits import g1_29dof_effort_limit_nm
from torque_timeseries_plot import plot_timeseries_grid, shared_series_y_limits


parser = argparse.ArgumentParser(description="Compare two ONNX torque metric runs.")
parser.add_argument(
    "--run-a",
    type=str,
    default=None,
    help="First run directory, torque_summary.csv path, or run folder name under metrics/shared_runs.",
)
parser.add_argument(
    "--run-b",
    type=str,
    default=None,
    help="Second run directory, torque_summary.csv path, or run folder name under metrics/shared_runs.",
)
parser.add_argument("--label-a", type=str, default=None, help="Legend label for run A.")
parser.add_argument("--label-b", type=str, default=None, help="Legend label for run B.")
parser.add_argument(
    "--run-root",
    type=str,
    default=str(DEFAULT_RUN_ROOT),
    help="Directory containing torque metric run folders.",
)
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Output image path. Defaults to metrics/shared_runs/comparisons/<timestamp>_torque_compare.png.",
)
args = parser.parse_args()


def _resolve_path(path_value: str | None, run_root: pathlib.Path) -> pathlib.Path | None:
    if path_value is None:
        return None

    path = pathlib.Path(path_value).expanduser()
    if not path.is_absolute():
        if path.exists():
            path = path.resolve()
        else:
            path = (run_root / path).resolve()

    if path.is_dir():
        path = path / "torque_summary.csv"
    if path.name != "torque_summary.csv":
        raise ValueError(f"Expected a run directory or torque_summary.csv path, got: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Missing torque summary: {path}")
    return path


def _find_latest_two_summaries(run_root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    candidates = []
    for path in run_root.glob("*/torque_summary.csv"):
        if path.parent.name == "comparisons":
            continue
        candidates.append(path)
    candidates.sort(key=lambda path: path.parent.stat().st_mtime, reverse=True)
    if len(candidates) < 2:
        raise FileNotFoundError(f"Need at least two torque_summary.csv files under {run_root}")
    return candidates[0], candidates[1]


def _load_metadata(summary_path: pathlib.Path) -> dict:
    metadata_path = summary_path.parent / "metadata.json"
    if not metadata_path.exists():
        return {}
    with metadata_path.open() as f:
        return json.load(f)


def _default_label(summary_path: pathlib.Path) -> str:
    metadata = _load_metadata(summary_path)
    task = metadata.get("task")
    if task == "Unitree-G1-29dof-Velocity-Backpack":
        return "Backpack"
    if task == "Unitree-G1-29dof-Velocity":
        return "Plain G1"
    if task:
        return task
    return summary_path.parent.name


def _float_or_nan(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _load_summary(summary_path: pathlib.Path) -> list[dict]:
    rows = []
    with summary_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "joint_index": int(row["joint_index"]),
                    "joint_name": row["joint_name"],
                    "min_torque_nm": _float_or_nan(row["min_torque_nm"]),
                    "max_torque_nm": _float_or_nan(row["max_torque_nm"]),
                    "peak_abs_torque_nm": _float_or_nan(row.get("peak_abs_torque_nm", "")),
                    "effort_limit_nm": _float_or_nan(row.get("effort_limit_nm", "")),
                    "peak_abs_over_limit_pct": _float_or_nan(row.get("peak_abs_over_limit_pct", "")),
                }
            )
    return sorted(rows, key=lambda row: row["joint_index"])


def _apply_official_g1_effort_limits(rows: list[dict]):
    for row in rows:
        limit = g1_29dof_effort_limit_nm(row["joint_name"])
        if limit is None:
            continue
        row["effort_limit_nm"] = float(limit)
        peak = row["peak_abs_torque_nm"]
        row["peak_abs_over_limit_pct"] = peak / limit * 100.0 if limit > 0.0 else float("nan")


def _validate_joint_order(rows_a: list[dict], rows_b: list[dict]):
    names_a = [(row["joint_index"], row["joint_name"]) for row in rows_a]
    names_b = [(row["joint_index"], row["joint_name"]) for row in rows_b]
    if names_a != names_b:
        raise ValueError("The two summaries do not have the same joint order.")


def _load_timeseries(
    summary_path: pathlib.Path,
    filename: str,
    expected_series_names: list[str] | None = None,
):
    timeseries_path = summary_path.parent / filename
    if not timeseries_path.exists():
        raise FileNotFoundError(f"Missing time series: {timeseries_path}")

    times = []
    values = []
    with timeseries_path.open(newline="") as f:
        reader = csv.DictReader(f)
        series_names = [name for name in (reader.fieldnames or []) if name not in {"frame", "time_s"}]
        if expected_series_names is not None and series_names != expected_series_names:
            raise ValueError(f"Unexpected series order in: {timeseries_path}")
        for row in reader:
            times.append(float(row["time_s"]))
            values.append([float(row[series_name]) for series_name in series_names])

    if not times:
        raise ValueError(f"No samples in: {timeseries_path}")
    return timeseries_path, times, values, series_names


def _write_delta_csv(path: pathlib.Path, rows_a: list[dict], rows_b: list[dict], label_a: str, label_b: str):
    fieldnames = [
        "joint_index",
        "joint_name",
        f"{label_a}_peak_abs_torque_nm",
        f"{label_b}_peak_abs_torque_nm",
        "peak_abs_delta_nm",
        "peak_abs_delta_pct_of_a",
        f"{label_a}_limit_usage_pct",
        f"{label_b}_limit_usage_pct",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row_a, row_b in zip(rows_a, rows_b):
            peak_a = row_a["peak_abs_torque_nm"]
            peak_b = row_b["peak_abs_torque_nm"]
            delta = peak_b - peak_a
            delta_pct = delta / peak_a * 100.0 if peak_a else float("nan")
            writer.writerow(
                {
                    "joint_index": row_a["joint_index"],
                    "joint_name": row_a["joint_name"],
                    f"{label_a}_peak_abs_torque_nm": f"{peak_a:.4f}",
                    f"{label_b}_peak_abs_torque_nm": f"{peak_b:.4f}",
                    "peak_abs_delta_nm": f"{delta:.4f}",
                    "peak_abs_delta_pct_of_a": f"{delta_pct:.2f}",
                    f"{label_a}_limit_usage_pct": f"{row_a['peak_abs_over_limit_pct']:.2f}",
                    f"{label_b}_limit_usage_pct": f"{row_b['peak_abs_over_limit_pct']:.2f}",
                }
            )


def _ensure_output_path(output_arg: str | None, run_root: pathlib.Path) -> pathlib.Path:
    if output_arg:
        output_path = pathlib.Path(output_arg).expanduser()
        if not output_path.is_absolute():
            output_path = (pathlib.Path.cwd() / output_path).resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = run_root / "comparisons"
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(output_dir, 0o777)
        except OSError:
            pass
        output_path = output_dir / f"{timestamp}_torque_range_compare.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _save_chmod(path: pathlib.Path):
    try:
        os.chmod(path, 0o666)
    except OSError:
        pass


def _slugify_output_label(label: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "_" for char in label)
    return "_".join(part for part in slug.split("_") if part) or "run"


def _timeseries_output_paths(
    range_output_path: pathlib.Path,
    label_a: str,
    label_b: str,
    signal_name: str,
) -> tuple[pathlib.Path, pathlib.Path]:
    range_suffix = "_torque_range_compare"
    if range_output_path.stem.endswith(range_suffix):
        stem = range_output_path.stem[: -len(range_suffix)]
    else:
        stem = range_output_path.stem
    output_a = range_output_path.with_name(
        f"{stem}_{_slugify_output_label(label_a)}_{signal_name}_timeseries_grid.png"
    )
    output_b = range_output_path.with_name(
        f"{stem}_{_slugify_output_label(label_b)}_{signal_name}_timeseries_grid.png"
    )
    return output_a, output_b


def _plot_separate_timeseries_grids(
    output_a: pathlib.Path,
    output_b: pathlib.Path,
    summary_a: pathlib.Path,
    summary_b: pathlib.Path,
    label_a: str,
    label_b: str,
    *,
    filename: str,
    expected_series_names: list[str] | None,
    graph_title: str,
    ylabel: str,
    grid_shape: tuple[int, int],
):
    import numpy as np

    timeseries_a, times_a, values_a, series_names_a = _load_timeseries(
        summary_a,
        filename,
        expected_series_names,
    )
    timeseries_b, times_b, values_b, series_names_b = _load_timeseries(
        summary_b,
        filename,
        expected_series_names,
    )
    if series_names_a != series_names_b:
        raise ValueError(f"The two {filename} files do not have the same series order.")
    times_a = np.asarray(times_a, dtype=float)
    times_b = np.asarray(times_b, dtype=float)
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    y_limits = shared_series_y_limits(values_a, values_b)

    graph_a = plot_timeseries_grid(
        output_a,
        series_names_a,
        times_a,
        values_a,
        title=f"{graph_title} - {label_a}",
        ylabel=ylabel,
        color="#2563eb",
        y_limits=y_limits,
        grid_shape=grid_shape,
    )
    graph_b = plot_timeseries_grid(
        output_b,
        series_names_b,
        times_b,
        values_b,
        title=f"{graph_title} - {label_b}",
        ylabel=ylabel,
        color="#dc2626",
        y_limits=y_limits,
        grid_shape=grid_shape,
    )
    return timeseries_a, timeseries_b, pathlib.Path(graph_a), pathlib.Path(graph_b)


def _plot_png(output_path: pathlib.Path, rows_a, rows_b, label_a: str, label_b: str):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    joint_names = [row["joint_name"] for row in rows_a]
    y = np.arange(len(joint_names))
    offset = 0.16

    a_min = np.array([row["min_torque_nm"] for row in rows_a])
    a_max = np.array([row["max_torque_nm"] for row in rows_a])
    b_min = np.array([row["min_torque_nm"] for row in rows_b])
    b_max = np.array([row["max_torque_nm"] for row in rows_b])
    limits = np.array([row["effort_limit_nm"] for row in rows_a])

    fig_height = max(8.0, len(joint_names) * 0.36)
    fig, ax = plt.subplots(figsize=(13.5, fig_height))

    finite_limits = np.isfinite(limits)
    if finite_limits.any():
        ax.hlines(
            y[finite_limits],
            -limits[finite_limits],
            limits[finite_limits],
            color="#d1d5db",
            linewidth=1.4,
            label="official G1 effort limit",
            zorder=0,
        )

    ax.hlines(y - offset, a_min, a_max, color="#2563eb", linewidth=4, label=label_a)
    ax.scatter(a_min, y - offset, color="#1d4ed8", s=16)
    ax.scatter(a_max, y - offset, color="#1d4ed8", s=16)
    ax.hlines(y + offset, b_min, b_max, color="#dc2626", linewidth=4, label=label_b)
    ax.scatter(b_min, y + offset, color="#b91c1c", s=16)
    ax.scatter(b_max, y + offset, color="#b91c1c", s=16)

    ax.axvline(0.0, color="#111827", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(joint_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Torque (N*m)")
    ax.set_title(f"Joint Torque Range Comparison: {label_a} vs {label_b}")
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax.legend(loc="lower right")

    values = [a_min, a_max, b_min, b_max]
    x_min = min(float(np.nanmin(values_array)) for values_array in values)
    x_max = max(float(np.nanmax(values_array)) for values_array in values)
    if finite_limits.any():
        x_min = min(x_min, float(np.nanmin(-limits[finite_limits])))
        x_max = max(x_max, float(np.nanmax(limits[finite_limits])))
    pad = max((x_max - x_min) * 0.05, 1.0)
    ax.set_xlim(x_min - pad, x_max + pad)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_svg(output_path: pathlib.Path, rows_a, rows_b, label_a: str, label_b: str):
    joint_names = [row["joint_name"] for row in rows_a]
    width = 1200
    row_h = 28
    left = 290
    right = 40
    top = 55
    height = top + row_h * len(joint_names) + 55

    values = []
    for row in rows_a + rows_b:
        values.extend([row["min_torque_nm"], row["max_torque_nm"]])
        if row["effort_limit_nm"] == row["effort_limit_nm"]:
            values.extend([-row["effort_limit_nm"], row["effort_limit_nm"]])
    x_min = min(values)
    x_max = max(values)
    pad = max((x_max - x_min) * 0.05, 1.0)
    x_min -= pad
    x_max += pad

    def scale_x(value):
        return left + (float(value) - x_min) / (x_max - x_min) * (width - left - right)

    zero_x = scale_x(0.0)
    with output_path.open("w") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">\n')
        f.write('<rect width="100%" height="100%" fill="white"/>\n')
        f.write(f'<text x="{left}" y="24" font-size="18" font-family="sans-serif">Joint Torque Range Comparison</text>\n')
        f.write(f'<line x1="{zero_x:.1f}" y1="38" x2="{zero_x:.1f}" y2="{height - 25}" stroke="#111827"/>\n')
        f.write(f'<text x="{left}" y="45" font-size="12" fill="#2563eb">{label_a}</text>\n')
        f.write(f'<text x="{left + 130}" y="45" font-size="12" fill="#dc2626">{label_b}</text>\n')
        for idx, (row_a, row_b) in enumerate(zip(rows_a, rows_b)):
            base_y = top + idx * row_h
            f.write(f'<text x="8" y="{base_y + 5}" font-size="12" font-family="monospace">{row_a["joint_name"]}</text>\n')
            limit = row_a["effort_limit_nm"]
            if limit == limit:
                f.write(
                    f'<line x1="{scale_x(-limit):.1f}" y1="{base_y}" x2="{scale_x(limit):.1f}" '
                    f'y2="{base_y}" stroke="#d1d5db" stroke-width="1.4"/>\n'
                )
            y_a = base_y - 5
            y_b = base_y + 5
            f.write(
                f'<line x1="{scale_x(row_a["min_torque_nm"]):.1f}" y1="{y_a}" '
                f'x2="{scale_x(row_a["max_torque_nm"]):.1f}" y2="{y_a}" stroke="#2563eb" stroke-width="4"/>\n'
            )
            f.write(
                f'<line x1="{scale_x(row_b["min_torque_nm"]):.1f}" y1="{y_b}" '
                f'x2="{scale_x(row_b["max_torque_nm"]):.1f}" y2="{y_b}" stroke="#dc2626" stroke-width="4"/>\n'
            )
        f.write("</svg>\n")


def main():
    run_root = pathlib.Path(args.run_root).expanduser().resolve()
    summary_a = _resolve_path(args.run_a, run_root)
    summary_b = _resolve_path(args.run_b, run_root)
    if summary_a is None or summary_b is None:
        summary_a, summary_b = _find_latest_two_summaries(run_root)

    rows_a = _load_summary(summary_a)
    rows_b = _load_summary(summary_b)
    _apply_official_g1_effort_limits(rows_a)
    _apply_official_g1_effort_limits(rows_b)
    _validate_joint_order(rows_a, rows_b)

    label_a = args.label_a or _default_label(summary_a)
    label_b = args.label_b or _default_label(summary_b)
    output_path = _ensure_output_path(args.output, run_root)

    try:
        _plot_png(output_path, rows_a, rows_b, label_a, label_b)
    except Exception as exc:  # noqa: BLE001
        output_path = output_path.with_suffix(".svg")
        print(f"[WARN] matplotlib PNG output failed ({exc}); writing SVG instead: {output_path}")
        _plot_svg(output_path, rows_a, rows_b, label_a, label_b)

    delta_csv = output_path.with_name(output_path.stem + "_delta.csv")
    _write_delta_csv(delta_csv, rows_a, rows_b, label_a, label_b)
    _save_chmod(output_path)
    _save_chmod(delta_csv)

    joint_names = [row["joint_name"] for row in rows_a]
    signal_specs = [
        {
            "name": "torque",
            "filename": "torque_timeseries.csv",
            "expected_series_names": joint_names,
            "graph_title": "Joint Torque over Time",
            "ylabel": "Applied joint torque (N*m)",
            "grid_shape": (3, 10),
        },
        {
            "name": "joint_position",
            "filename": "joint_position_timeseries.csv",
            "expected_series_names": joint_names,
            "graph_title": "Joint Position over Time",
            "ylabel": "Joint position (rad)",
            "grid_shape": (3, 10),
        },
        {
            "name": "joint_velocity",
            "filename": "joint_velocity_timeseries.csv",
            "expected_series_names": joint_names,
            "graph_title": "Joint Velocity over Time",
            "ylabel": "Joint velocity (rad/s)",
            "grid_shape": (3, 10),
        },
        {
            "name": "base_angular_velocity",
            "filename": "base_angular_velocity_timeseries.csv",
            "expected_series_names": ["x_roll", "y_pitch", "z_yaw"],
            "graph_title": "Base Angular Velocity over Time",
            "ylabel": "Base angular velocity in body frame (rad/s)",
            "grid_shape": (1, 3),
        },
        {
            "name": "policy_action",
            "filename": "policy_action_timeseries.csv",
            "expected_series_names": joint_names,
            "graph_title": "Raw Policy Action over Time",
            "ylabel": "Raw policy action",
            "grid_shape": (3, 10),
        },
    ]
    grid_outputs = []
    for signal_spec in signal_specs:
        timeseries_output_a, timeseries_output_b = _timeseries_output_paths(
            output_path,
            label_a,
            label_b,
            signal_spec["name"],
        )
        try:
            timeseries_a, timeseries_b, timeseries_output_a, timeseries_output_b = (
                _plot_separate_timeseries_grids(
                    timeseries_output_a,
                    timeseries_output_b,
                    summary_a,
                    summary_b,
                    label_a,
                    label_b,
                    filename=signal_spec["filename"],
                    expected_series_names=signal_spec["expected_series_names"],
                    graph_title=signal_spec["graph_title"],
                    ylabel=signal_spec["ylabel"],
                    grid_shape=signal_spec["grid_shape"],
                )
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Could not create {signal_spec['name']} grids ({exc}).")
            continue
        _save_chmod(timeseries_output_a)
        _save_chmod(timeseries_output_b)
        grid_outputs.append(
            (signal_spec["name"], timeseries_a, timeseries_b, timeseries_output_a, timeseries_output_b)
        )

    print(f"[INFO] Run A: {summary_a.parent}")
    print(f"[INFO] Run B: {summary_b.parent}")
    print(f"[INFO] Saved comparison graph: {output_path}")
    print(f"[INFO] Saved peak torque delta table: {delta_csv}")
    for signal_name, timeseries_a, timeseries_b, graph_a, graph_b in grid_outputs:
        print(f"[INFO] {signal_name} source A: {timeseries_a}")
        print(f"[INFO] {signal_name} source B: {timeseries_b}")
        print(f"[INFO] Saved run A {signal_name} grid: {graph_a}")
        print(f"[INFO] Saved run B {signal_name} grid: {graph_b}")


if __name__ == "__main__":
    main()
