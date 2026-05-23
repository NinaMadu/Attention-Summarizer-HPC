#!/usr/bin/env python3
"""
Generate an SVG chart of compute_total_s versus worker count.

Input:
    docs/final_performance_summary.csv if available, otherwise
    docs/final_performance_comparison.csv

Output:
    docs/charts/compute_total_vs_workers.svg
    docs/charts/compute_total_vs_workers_unmatched.csv
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_CSV = PROJECT_ROOT / "docs" / "final_performance_summary.csv"
RAW_CSV = PROJECT_ROOT / "docs" / "final_performance_comparison.csv"
OUT_DIR = PROJECT_ROOT / "docs" / "charts"
OUT_SVG = OUT_DIR / "compute_total_vs_workers.svg"
OUT_UNMATCHED = OUT_DIR / "compute_total_vs_workers_unmatched.csv"


FAMILY_LABELS = {
    "serial_300d": "Serial 300D",
    "openmp_300d": "OpenMP 300D",
    "mpi_300d": "MPI 300D",
    "mpi_openmp_hybrid_300d": "MPI+OpenMP Hybrid 300D",
    "cuda_300d": "CUDA 300D",
    "cuda_openmp_hybrid_300d": "CUDA+OpenMP Hybrid 300D",
}

COLORS = {
    "serial_300d": "#111827",
    "openmp_300d": "#2563eb",
    "mpi_300d": "#dc2626",
    "mpi_openmp_hybrid_300d": "#059669",
    "cuda_300d": "#7c3aed",
    "cuda_openmp_hybrid_300d": "#ea580c",
}


def to_float(value: str) -> float | None:
    try:
        if value == "":
            return None
        number = float(value)
        if not math.isfinite(number):
            return None
        return number
    except ValueError:
        return None


def to_int(value: str) -> int | None:
    try:
        if value == "":
            return None
        return int(float(value))
    except ValueError:
        return None


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
    ticks = []
    value = 0.0
    while value <= top + step * 0.5:
        ticks.append(round(value, 10))
        value += step
    return ticks


def svg_text(x: float, y: float, text: str, size: int = 13, anchor: str = "middle",
             weight: str = "400", fill: str = "#111827") -> str:
    escaped = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}">{escaped}</text>'
    )


def main() -> int:
    input_csv = SUMMARY_CSV if SUMMARY_CSV.exists() else RAW_CSV
    if not input_csv.exists():
        raise SystemExit(f"Missing input CSV: {input_csv}")

    rows = list(csv.DictReader(input_csv.open(newline="", encoding="utf-8")))
    valid: List[Dict[str, str]] = []
    unmatched: List[Dict[str, str]] = []

    for row in rows:
        reason = ""
        family = row.get("family", "")
        status = row.get("status", "")
        workers = to_int(row.get("total_workers", ""))
        compute_total = to_float(row.get("compute_total_s", ""))

        if status != "ok":
            reason = f"status is {status!r}, not 'ok'"
        elif family not in FAMILY_LABELS:
            reason = f"family {family!r} is not a comparable 300D family"
        elif workers is None:
            reason = "missing/invalid total_workers"
        elif compute_total is None:
            reason = "missing/invalid compute_total_s"
        elif workers < 1:
            reason = "total_workers must be >= 1"

        if reason:
            copied = dict(row)
            copied["chart_skip_reason"] = reason
            unmatched.append(copied)
        else:
            valid.append(row)

    if not valid:
        raise SystemExit("No valid rows found for chart.")

    grouped: Dict[str, List[Tuple[int, float, str]]] = defaultdict(list)
    for row in valid:
        grouped[row["family"]].append(
            (
                int(float(row["total_workers"])),
                float(row["compute_total_s"]),
                row["run_id"],
            )
        )

    for family in grouped:
        grouped[family].sort(key=lambda item: (item[0], item[2]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if unmatched:
        fieldnames = list(rows[0].keys()) + ["chart_skip_reason"]
        with OUT_UNMATCHED.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in unmatched:
                writer.writerow({field: row.get(field, "") for field in fieldnames})
    elif OUT_UNMATCHED.exists():
        OUT_UNMATCHED.unlink()

    max_workers = max(int(float(row["total_workers"])) for row in valid)
    max_time = max(float(row["compute_total_s"]) for row in valid)
    y_ticks = nice_ticks(max_time)
    y_max = y_ticks[-1] if y_ticks else max_time

    width, height = 1180, 720
    left, right, top, bottom = 92, 310, 70, 92
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(workers: int) -> float:
        if max_workers <= 1:
            return left
        return left + (workers - 1) / (max_workers - 1) * plot_w

    def sy(seconds: float) -> float:
        if y_max <= 0:
            return top + plot_h
        return top + plot_h - seconds / y_max * plot_h

    x_ticks = sorted({1, 2, 4, 8, 16, 24, 32} | {int(float(r["total_workers"])) for r in valid})
    x_ticks = [tick for tick in x_ticks if 1 <= tick <= max_workers]

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(width / 2, 34, "Compute Time vs Worker Count", 24, "middle", "700"),
        svg_text(width / 2, 58, f"compute_total_s excludes GloVe loading time | source: {input_csv.relative_to(PROJECT_ROOT).as_posix()}", 13, "middle", "400", "#4b5563"),
    ]

    # Grid and axes
    for tick in y_ticks:
        y = sy(tick)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(svg_text(left - 12, y + 4, f"{tick:g}", 12, "end", "400", "#374151"))
    for tick in x_ticks:
        x = sx(tick)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#f3f4f6" stroke-width="1"/>')
        parts.append(svg_text(x, top + plot_h + 24, str(tick), 12, "middle", "400", "#374151"))

    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.4"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.4"/>')
    parts.append(svg_text(left + plot_w / 2, height - 32, "Worker count (total_workers: OpenMP threads, MPI ranks, or ranks x threads)", 14, "middle", "600"))
    parts.append(f'<text x="25" y="{top + plot_h / 2:.1f}" text-anchor="middle" font-size="14" font-weight="600" fill="#111827" transform="rotate(-90 25 {top + plot_h / 2:.1f})">compute_total_s (seconds)</text>')

    # Lines and markers
    order = [
        "serial_300d",
        "openmp_300d",
        "mpi_300d",
        "mpi_openmp_hybrid_300d",
        "cuda_300d",
        "cuda_openmp_hybrid_300d",
    ]
    for family in order:
        points = grouped.get(family)
        if not points:
            continue
        color = COLORS[family]
        coords = " ".join(f"{sx(workers):.1f},{sy(seconds):.1f}" for workers, seconds, _ in points)
        if len(points) > 1:
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>')
        for workers, seconds, run_id in points:
            x, y = sx(workers), sy(seconds)
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.2" fill="{color}" stroke="#ffffff" stroke-width="1.4"/>')
            parts.append(f'<title>{FAMILY_LABELS[family]} | workers={workers} | compute_total_s={seconds:.4f}s | {run_id}</title>')

    # Legend
    legend_x = left + plot_w + 34
    legend_y = top + 8
    parts.append(svg_text(legend_x, legend_y, "Implementation", 15, "start", "700"))
    legend_y += 24
    for family in order:
        if family not in grouped:
            continue
        color = COLORS[family]
        parts.append(f'<line x1="{legend_x}" y1="{legend_y:.1f}" x2="{legend_x + 26}" y2="{legend_y:.1f}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>')
        parts.append(f'<circle cx="{legend_x + 13}" cy="{legend_y:.1f}" r="4.5" fill="{color}" stroke="#ffffff" stroke-width="1"/>')
        parts.append(svg_text(legend_x + 36, legend_y + 4, FAMILY_LABELS[family], 13, "start"))
        legend_y += 24

    note_y = height - 52
    skipped_text = f"Skipped rows: {len(unmatched)}"
    if unmatched:
        skipped_text += f" (see {OUT_UNMATCHED.relative_to(PROJECT_ROOT).as_posix()})"
    parts.append(svg_text(width - 24, note_y, skipped_text, 12, "end", "400", "#6b7280"))
    parts.append("</svg>")

    OUT_SVG.write_text("\n".join(parts), encoding="utf-8")

    print(f"Created {OUT_SVG.relative_to(PROJECT_ROOT)}")
    if unmatched:
        print(f"Skipped {len(unmatched)} unmatched row(s):")
        for row in unmatched:
            print(f"  {row.get('run_id')} ({row.get('family')}): {row.get('chart_skip_reason')}")
        print(f"Details written to {OUT_UNMATCHED.relative_to(PROJECT_ROOT)}")
    else:
        print("No unmatched rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
