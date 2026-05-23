#!/usr/bin/env python3
"""
Generate report-ready SVG charts from the final comparison CSV files.

Inputs:
    docs/final_performance_summary.csv
    docs/final_performance_comparison.csv
    docs/final_accuracy_comparison.csv

Outputs:
    docs/charts/*.svg
"""

from __future__ import annotations

import csv
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS = PROJECT_ROOT / "docs"
CHARTS = DOCS / "charts"
PERF_SUMMARY = DOCS / "final_performance_summary.csv"
PERF_RAW = DOCS / "final_performance_comparison.csv"
ACC_RAW = DOCS / "final_accuracy_comparison.csv"

FAMILY_LABELS = {
    "serial_300d": "Serial",
    "openmp_300d": "OpenMP",
    "mpi_300d": "MPI",
    "mpi_openmp_hybrid_300d": "MPI+OpenMP",
    "cuda_300d": "CUDA",
    "cuda_openmp_hybrid_300d": "CUDA+OpenMP",
}

FAMILY_ORDER = [
    "serial_300d",
    "openmp_300d",
    "mpi_300d",
    "mpi_openmp_hybrid_300d",
    "cuda_300d",
    "cuda_openmp_hybrid_300d",
]

COLORS = {
    "serial_300d": "#111827",
    "openmp_300d": "#2563eb",
    "mpi_300d": "#dc2626",
    "mpi_openmp_hybrid_300d": "#059669",
    "cuda_300d": "#7c3aed",
    "cuda_openmp_hybrid_300d": "#ea580c",
    "load_s": "#94a3b8",
    "covariance_s": "#2563eb",
    "jacobi_s": "#dc2626",
    "whitening_s": "#059669",
    "vectorize_s": "#f59e0b",
    "positional_s": "#14b8a6",
    "attention_s": "#7c3aed",
    "scoring_s": "#ea580c",
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(value: object) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def text(x: float, y: float, value: str, size: int = 13, anchor: str = "middle",
         weight: str = "400", fill: str = "#111827") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}">{escape(value)}</text>'
    )


def nice_ticks(max_value: float, count: int = 6) -> List[float]:
    if max_value <= 0:
        return [0.0]
    raw_step = max_value / max(1, count - 1)
    power = 10 ** math.floor(math.log10(raw_step))
    step = power
    for mult in (1, 2, 5, 10):
        if raw_step <= mult * power:
            step = mult * power
            break
    top = math.ceil(max_value / step) * step
    ticks: List[float] = []
    value = 0.0
    while value <= top + step * 0.5:
        ticks.append(round(value, 10))
        value += step
    return ticks


def valid_perf_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [
        row for row in rows
        if row.get("status") == "ok"
        and row.get("family") in FAMILY_LABELS
        and fnum(row.get("compute_total_s")) is not None
        and fnum(row.get("total_workers")) is not None
    ]


def best_by_family(rows: List[Dict[str, str]], metric: str = "compute_total_s") -> Dict[str, Dict[str, str]]:
    best: Dict[str, Dict[str, str]] = {}
    for row in valid_perf_rows(rows):
        family = row["family"]
        value = fnum(row.get(metric))
        if value is None:
            continue
        if family not in best or value < (fnum(best[family].get(metric)) or float("inf")):
            best[family] = row
    return best


def svg_frame(width: int, height: int, title: str, subtitle: str = "") -> List[str]:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        text(width / 2, 34, title, 23, "middle", "700"),
    ]
    return parts


def setup_label(row: Dict[str, str]) -> str:
    family = row.get("family", "")
    ranks = row.get("mpi_ranks") or "1"
    threads = row.get("omp_threads") or row.get("actual_omp_threads") or "1"

    if family == "serial_300d":
        return "1 process"
    if family == "openmp_300d":
        return f"{threads} threads"
    if family == "mpi_300d":
        return f"{ranks} ranks"
    if family == "mpi_openmp_hybrid_300d":
        return f"{ranks} ranks x {threads} threads"
    if family == "cuda_300d":
        return "GPU baseline"
    if family == "cuda_openmp_hybrid_300d":
        return f"GPU + {threads} CPU threads"
    return ""


def write_line_chart_data(data: Dict[str, List[Tuple[int, float, str]]],
                          out_path: Path, title: str, y_label: str,
                          include_families: Sequence[str], subtitle: str,
                          metric_label: str) -> None:
    data = {family: sorted(points) for family, points in data.items() if points}
    if not data:
        return

    width, height = 1180, 720
    left, right, top, bottom = 92, 310, 72, 92
    plot_w, plot_h = width - left - right, height - top - bottom
    max_workers = max(worker for points in data.values() for worker, _, _ in points)
    max_y = max(value for points in data.values() for _, value, _ in points)
    y_ticks = nice_ticks(max_y)
    y_max = y_ticks[-1]

    def sx(worker: int) -> float:
        return left if max_workers <= 1 else left + (worker - 1) / (max_workers - 1) * plot_w

    def sy(value: float) -> float:
        return top + plot_h - (value / y_max * plot_h if y_max else 0.0)

    parts = svg_frame(width, height, title, subtitle)
    x_ticks = sorted({1, 2, 4, 8, 16, 24, 32} | {w for points in data.values() for w, _, _ in points})
    for tick in y_ticks:
        y = sy(tick)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(text(left - 12, y + 4, f"{tick:g}", 12, "end", "400", "#374151"))
    for tick in x_ticks:
        if tick > max_workers:
            continue
        x = sx(tick)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#f3f4f6"/>')
        parts.append(text(x, top + plot_h + 24, str(tick), 12, "middle", "400", "#374151"))
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.4"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.4"/>')
    parts.append(text(left + plot_w / 2, height - 32, "Number of Processing Workers", 14, "middle", "600"))
    parts.append(f'<text x="25" y="{top + plot_h / 2:.1f}" text-anchor="middle" font-size="14" font-weight="600" fill="#111827" transform="rotate(-90 25 {top + plot_h / 2:.1f})">{escape(y_label)}</text>')

    legend_x, legend_y = left + plot_w + 34, top + 10
    parts.append(text(legend_x, legend_y, "Implementation", 15, "start", "700"))
    legend_y += 26
    for family in include_families:
        points = data.get(family)
        if not points:
            continue
        color = COLORS[family]
        coords = " ".join(f"{sx(w):.1f},{sy(v):.1f}" for w, v, _ in points)
        if len(points) > 1:
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>')
        for worker, value, run_id in points:
            parts.append(f'<circle cx="{sx(worker):.1f}" cy="{sy(value):.1f}" r="5.2" fill="{color}" stroke="#ffffff" stroke-width="1.4"><title>{escape(run_id)} | workers={worker} | {escape(metric_label)}={value:.4g}</title></circle>')
        parts.append(f'<line x1="{legend_x}" y1="{legend_y:.1f}" x2="{legend_x + 26}" y2="{legend_y:.1f}" stroke="{color}" stroke-width="3"/>')
        parts.append(text(legend_x + 36, legend_y + 4, FAMILY_LABELS[family], 13, "start"))
        legend_y += 24
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def line_chart(rows: List[Dict[str, str]], metric: str, out_path: Path, title: str,
               y_label: str, include_families: Sequence[str]) -> None:
    data: Dict[str, List[Tuple[int, float, str]]] = {family: [] for family in include_families}
    for row in valid_perf_rows(rows):
        family = row["family"]
        if family not in include_families:
            continue
        y_value = fnum(row.get(metric))
        workers = fnum(row.get("total_workers"))
        if y_value is None or workers is None:
            continue
        data[family].append((int(workers), y_value, row.get("run_id", "")))
    write_line_chart_data(
        data,
        out_path,
        title,
        y_label,
        include_families,
        "",
        metric,
    )


def best_compute_by_family_worker(rows: List[Dict[str, str]],
                                  include_families: Sequence[str]) -> Dict[str, List[Tuple[int, float, str]]]:
    best: Dict[Tuple[str, int], Tuple[float, str]] = {}
    for row in valid_perf_rows(rows):
        family = row["family"]
        if family not in include_families:
            continue
        workers = fnum(row.get("total_workers"))
        compute_total = fnum(row.get("compute_total_s"))
        if workers is None or compute_total is None:
            continue
        key = (family, int(workers))
        current = best.get(key)
        if current is None or compute_total < current[0]:
            best[key] = (compute_total, row.get("run_id", ""))

    grouped: Dict[str, List[Tuple[int, float, str]]] = {family: [] for family in include_families}
    for (family, workers), (compute_total, run_id) in best.items():
        grouped[family].append((workers, compute_total, run_id))
    return {family: sorted(points) for family, points in grouped.items() if points}


def scaling_chart(rows: List[Dict[str, str]], out_path: Path, title: str,
                  y_label: str, include_families: Sequence[str],
                  efficiency: bool = False) -> None:
    runtime_data = best_compute_by_family_worker(rows, include_families)
    data: Dict[str, List[Tuple[int, float, str]]] = {}
    for family, points in runtime_data.items():
        if len(points) < 2:
            continue
        base_workers, base_time, _ = min(points, key=lambda item: item[0])
        scaled: List[Tuple[int, float, str]] = []
        for workers, compute_total, run_id in points:
            if compute_total <= 0:
                continue
            speedup = base_time / compute_total
            value = speedup
            if efficiency:
                value = 100.0 * speedup / max(1.0, workers / base_workers)
            scaled.append((workers, value, f"{run_id}; baseline_workers={base_workers}"))
        data[family] = scaled

    metric_label = "scaling efficiency %" if efficiency else "scaling speedup"
    write_line_chart_data(data, out_path, title, y_label, include_families, "", metric_label)


def bar_chart(items: List[Tuple[str, float, str, str]], out_path: Path, title: str, y_label: str) -> None:
    if not items:
        return
    width, height = 1100, 680
    left, right, top, bottom = 92, 40, 76, 130
    plot_w, plot_h = width - left - right, height - top - bottom
    max_y = max(value for _, value, _, _ in items)
    y_ticks = nice_ticks(max_y)
    y_max = y_ticks[-1]
    bar_gap = 24
    bar_w = max(36, (plot_w - bar_gap * (len(items) + 1)) / len(items))

    def sy(value: float) -> float:
        return top + plot_h - (value / y_max * plot_h if y_max else 0.0)

    parts = svg_frame(width, height, title)
    for tick in y_ticks:
        y = sy(tick)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(text(left - 12, y + 4, f"{tick:g}", 12, "end", "400", "#374151"))
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827"/>')
    parts.append(f'<text x="25" y="{top + plot_h / 2:.1f}" text-anchor="middle" font-size="14" font-weight="600" fill="#111827" transform="rotate(-90 25 {top + plot_h / 2:.1f})">{escape(y_label)}</text>')

    for idx, (label, value, color_key, setup) in enumerate(items):
        x = left + bar_gap + idx * (bar_w + bar_gap)
        y = sy(value)
        h = top + plot_h - y
        color = COLORS.get(color_key, "#2563eb")
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}" rx="4"/>')
        parts.append(text(x + bar_w / 2, y - 8, f"{value:.3g}", 12, "middle", "600"))
        label_x = x + bar_w / 2
        parts.append(text(label_x, top + plot_h + 28, label, 12, "middle"))
        if setup:
            parts.append(text(label_x, top + plot_h + 46, setup, 11, "middle", "400", "#4b5563"))
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def load_compute_chart() -> None:
    script = PROJECT_ROOT / "scripts" / "generate_compute_time_chart.py"
    if script.exists():
        subprocess.run([sys.executable, str(script)], cwd=PROJECT_ROOT, check=False)


def best_runtime_chart(rows: List[Dict[str, str]]) -> None:
    best = best_by_family(rows)
    items = []
    for family in FAMILY_ORDER:
        row = best.get(family)
        if row:
            value = fnum(row.get("compute_total_s"))
            if value is not None:
                items.append((FAMILY_LABELS[family], value, family, setup_label(row)))
    bar_chart(items, CHARTS / "best_compute_runtime_by_family.svg", "Best Compute Runtime by Implementation", "Compute Time (seconds)")


def memory_chart(rows: List[Dict[str, str]]) -> None:
    best = best_by_family(rows)
    items = []
    for family in FAMILY_ORDER:
        row = best.get(family)
        if row:
            value = fnum(row.get("peak_memory_mb"))
            if value is not None:
                items.append((FAMILY_LABELS[family], value, family, setup_label(row)))
    bar_chart(items, CHARTS / "peak_memory_by_best_family.svg", "Peak Memory of Best Runtime Setup", "Peak Memory (MB)")


def load_vs_compute_chart(rows: List[Dict[str, str]]) -> None:
    best = best_by_family(rows)
    selected = [(family, best[family]) for family in FAMILY_ORDER if family in best]
    if not selected:
        return
    width, height = 1180, 720
    left, right, top, bottom = 92, 40, 76, 160
    plot_w, plot_h = width - left - right, height - top - bottom
    max_y = max((fnum(row.get("load_s")) or 0) + (fnum(row.get("compute_total_s")) or 0) for _, row in selected)
    ticks, y_max = nice_ticks(max_y), nice_ticks(max_y)[-1]
    bar_w = max(50, plot_w / max(1, len(selected)) * 0.55)
    slot = plot_w / max(1, len(selected))

    def sy(value: float) -> float:
        return top + plot_h - (value / y_max * plot_h if y_max else 0.0)

    parts = svg_frame(width, height, "Load Time vs Compute Time")
    for tick in ticks:
        y = sy(tick)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(text(left - 12, y + 4, f"{tick:g}", 12, "end", "400", "#374151"))
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827"/>')
    parts.append(f'<text x="25" y="{top + plot_h / 2:.1f}" text-anchor="middle" font-size="14" font-weight="600" fill="#111827" transform="rotate(-90 25 {top + plot_h / 2:.1f})">Time (seconds)</text>')

    for idx, (family, row) in enumerate(selected):
        load = fnum(row.get("load_s")) or 0.0
        compute = fnum(row.get("compute_total_s")) or 0.0
        x = left + idx * slot + (slot - bar_w) / 2
        base = top + plot_h
        compute_h = base - sy(compute)
        load_h = base - sy(load)
        parts.append(f'<rect x="{x:.1f}" y="{base - compute_h:.1f}" width="{bar_w:.1f}" height="{compute_h:.1f}" fill="#2563eb" rx="3"><title>compute={compute:.4f}s</title></rect>')
        parts.append(f'<rect x="{x:.1f}" y="{base - compute_h - load_h:.1f}" width="{bar_w:.1f}" height="{load_h:.1f}" fill="#94a3b8" rx="3"><title>load={load:.4f}s</title></rect>')
        label_x = x + bar_w / 2
        parts.append(text(label_x, top + plot_h + 28, FAMILY_LABELS[family], 12, "middle"))
        parts.append(text(label_x, top + plot_h + 46, setup_label(row), 11, "middle", "400", "#4b5563"))
    parts.append(text(width - 170, top + 18, "GloVe loading", 13, "start"))
    parts.append(f'<rect x="{width - 205}" y="{top + 7}" width="18" height="12" fill="#94a3b8"/>')
    parts.append(text(width - 170, top + 40, "Functional compute", 13, "start"))
    parts.append(f'<rect x="{width - 205}" y="{top + 29}" width="18" height="12" fill="#2563eb"/>')
    parts.append("</svg>")
    (CHARTS / "load_vs_compute_best_family.svg").write_text("\n".join(parts), encoding="utf-8")


def stage_breakdown_chart(rows: List[Dict[str, str]]) -> None:
    best = best_by_family(rows)
    stages = ["covariance_s", "jacobi_s", "whitening_s", "vectorize_s", "positional_s", "attention_s", "scoring_s"]
    stage_labels = {
        "covariance_s": "Covariance",
        "jacobi_s": "Jacobi Eigen",
        "whitening_s": "Whitening Matrix",
        "vectorize_s": "Vector Lookup",
        "positional_s": "Positional Encoding",
        "attention_s": "Self Attention",
        "scoring_s": "Sentence Scoring",
    }
    selected = [(family, best[family]) for family in FAMILY_ORDER if family in best]
    if not selected:
        return
    width, height = 1240, 720
    left, right, top, bottom = 92, 250, 76, 160
    plot_w, plot_h = width - left - right, height - top - bottom
    max_y = max(sum(fnum(row.get(stage)) or 0.0 for stage in stages) for _, row in selected)
    ticks, y_max = nice_ticks(max_y), nice_ticks(max_y)[-1]
    slot = plot_w / max(1, len(selected))
    bar_w = max(52, slot * 0.58)

    def sy(value: float) -> float:
        return top + plot_h - (value / y_max * plot_h if y_max else 0.0)

    parts = svg_frame(width, height, "Stage Time Breakdown")
    for tick in ticks:
        y = sy(tick)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(text(left - 12, y + 4, f"{tick:g}", 12, "end", "400", "#374151"))
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827"/>')
    parts.append(f'<text x="25" y="{top + plot_h / 2:.1f}" text-anchor="middle" font-size="14" font-weight="600" fill="#111827" transform="rotate(-90 25 {top + plot_h / 2:.1f})">Time (seconds)</text>')

    for idx, (family, row) in enumerate(selected):
        x = left + idx * slot + (slot - bar_w) / 2
        current = 0.0
        for stage in stages:
            value = fnum(row.get(stage)) or 0.0
            if value <= 0:
                continue
            y_top = sy(current + value)
            y_bottom = sy(current)
            parts.append(f'<rect x="{x:.1f}" y="{y_top:.1f}" width="{bar_w:.1f}" height="{y_bottom - y_top:.1f}" fill="{COLORS[stage]}"><title>{escape(stage_labels[stage])}={value:.4f}s</title></rect>')
            current += value
        label_x = x + bar_w / 2
        parts.append(text(label_x, top + plot_h + 28, FAMILY_LABELS[family], 12, "middle"))
        parts.append(text(label_x, top + plot_h + 46, setup_label(row), 11, "middle", "400", "#4b5563"))

    lx, ly = left + plot_w + 30, top + 10
    parts.append(text(lx, ly, "Stage", 15, "start", "700"))
    ly += 24
    for stage in stages:
        parts.append(f'<rect x="{lx}" y="{ly - 10}" width="18" height="12" fill="{COLORS[stage]}"/>')
        parts.append(text(lx + 28, ly, stage_labels[stage], 12, "start"))
        ly += 22
    parts.append("</svg>")
    (CHARTS / "stage_time_breakdown_best_family.svg").write_text("\n".join(parts), encoding="utf-8")


def accuracy_chart(acc_rows: List[Dict[str, str]], perf_rows: List[Dict[str, str]]) -> None:
    best = best_by_family(perf_rows)
    selected_run_ids = {row.get("run_id"): family for family, row in best.items()}
    best_by_run_id = {row.get("run_id"): row for row in best.values()}
    metric = "token_centrality_rmse"
    items = []
    for row in acc_rows:
        run_id = row.get("run_id")
        family = selected_run_ids.get(run_id)
        if not family or family == "serial_300d":
            continue
        value = fnum(row.get(metric))
        if value is not None:
            items.append((FAMILY_LABELS[family], value, family, setup_label(best_by_run_id.get(run_id, {}))))
    bar_chart(items, CHARTS / "accuracy_token_centrality_rmse_best_family.svg", "Accuracy Error vs Serial Baseline", "Token Centrality RMSE")


def main() -> int:
    CHARTS.mkdir(parents=True, exist_ok=True)
    perf_rows = read_csv(PERF_SUMMARY) or read_csv(PERF_RAW)
    acc_rows = read_csv(ACC_RAW)
    if not perf_rows:
        raise SystemExit("No performance CSV data found.")

    load_compute_chart()
    line_chart(
        perf_rows,
        "compute_speedup_vs_best_serial300",
        CHARTS / "compute_speedup_vs_workers.svg",
        "Compute Speedup vs Serial Baseline",
        "Speedup Compared with Serial",
        [family for family in FAMILY_ORDER if family != "serial_300d"],
    )
    line_chart(
        perf_rows,
        "compute_efficiency_percent",
        CHARTS / "compute_efficiency_vs_workers.svg",
        "CPU Parallel Efficiency vs Serial Baseline",
        "Parallel Efficiency (%)",
        ["openmp_300d", "mpi_300d", "mpi_openmp_hybrid_300d"],
    )
    scaling_chart(
        perf_rows,
        CHARTS / "scaling_speedup_vs_workers.svg",
        "Scaling Speedup vs Worker Count",
        "Speedup Compared with Own Baseline",
        ["openmp_300d", "mpi_300d", "mpi_openmp_hybrid_300d", "cuda_openmp_hybrid_300d"],
    )
    scaling_chart(
        perf_rows,
        CHARTS / "scaling_efficiency_vs_workers.svg",
        "Scaling Efficiency vs Worker Count",
        "Scaling Efficiency (%)",
        ["openmp_300d", "mpi_300d", "mpi_openmp_hybrid_300d", "cuda_openmp_hybrid_300d"],
        efficiency=True,
    )
    best_runtime_chart(perf_rows)
    load_vs_compute_chart(perf_rows)
    stage_breakdown_chart(perf_rows)
    memory_chart(perf_rows)
    accuracy_chart(acc_rows, perf_rows)

    print("Created report charts in docs/charts:")
    for path in sorted(CHARTS.glob("*.svg")):
        print(f"  {path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
