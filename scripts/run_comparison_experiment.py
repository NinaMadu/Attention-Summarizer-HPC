#!/usr/bin/env python3
"""
Run summarizer performance/accuracy experiments and generate compact CSVs.

Run from WSL Ubuntu at the project root, for example:

    python3 scripts/run_comparison_experiment.py --plan i9-quick --input-file examples/paragraphs.txt

Outputs:
    docs/final_performance_comparison.csv
    docs/final_accuracy_comparison.csv
    results/comparison_runs/<experiment_id>/*.log
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RunCase:
    run_name: str
    family: str
    label: str
    build_target: str
    command: List[str]
    glove_file: str
    mpi_ranks: int = 1
    omp_threads: int = 1
    cuda_enabled: str = "no"
    config: str = "default"
    notes: str = ""

    @property
    def total_workers(self) -> int:
        if self.cuda_enabled == "yes":
            return max(1, self.omp_threads)
        return max(1, self.mpi_ranks * self.omp_threads)


def csv_int_list(value: str) -> List[int]:
    values: List[int] = []
    for item in value.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values


def hybrid_list(value: str) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if "x" not in item:
            raise argparse.ArgumentTypeError("Hybrid values must look like 2x8,4x4")
        ranks, threads = item.split("x", 1)
        pairs.append((int(ranks), int(threads)))
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all summarizer modes and create performance/accuracy CSVs."
    )
    parser.add_argument("--input-file", default="examples/paragraphs.txt")
    parser.add_argument("--input-text", default=None)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--input-id", default="INPUT001")
    parser.add_argument(
        "--plan",
        choices=["single", "i9-quick", "i9-full", "custom"],
        default="i9-quick",
        help="single keeps one config; i9-quick is recommended; i9-full is longer.",
    )
    parser.add_argument("--mpi-ranks", type=int, default=2, help="Used by --plan single.")
    parser.add_argument("--omp-threads", type=int, default=4, help="Used by --plan single.")
    parser.add_argument("--openmp-list", type=csv_int_list, default=None)
    parser.add_argument("--mpi-list", type=csv_int_list, default=None)
    parser.add_argument("--hybrid-list", type=hybrid_list, default=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--skip-cuda", action="store_true")
    parser.add_argument("--include-serial-200d", action="store_true")
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--list-plan",
        action="store_true",
        help="Print the selected run plan and exit without building or running programs.",
    )
    parser.add_argument("--results-dir", default="results/comparison_runs")
    parser.add_argument(
        "--nvcc-flags",
        default="-O2",
        help="Flags passed to nvcc through Makefile NVCC_FLAGS. Default avoids hard-coded GPU architecture.",
    )

    parser.add_argument("--cpu-model", default="Intel Core i9-14900K")
    parser.add_argument("--cpu-cores", type=int, default=24)
    parser.add_argument("--cpu-threads", type=int, default=32)
    parser.add_argument("--gpu-model", default="NVIDIA RTX 5090")
    parser.add_argument("--gpu-memory-gb", type=int, default=32)
    parser.add_argument("--cuda-version", default="13.2")
    parser.add_argument("--ram-gb", type=int, default=64)
    return parser.parse_args()


def unique_ints(values: Iterable[int]) -> List[int]:
    seen = set()
    result = []
    for value in values:
        if value < 1:
            continue
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def plan_values(args: argparse.Namespace) -> Tuple[List[int], List[int], List[Tuple[int, int]]]:
    if args.plan == "single":
        return [args.omp_threads], [args.mpi_ranks], [(args.mpi_ranks, args.omp_threads)]

    if args.plan == "i9-quick":
        openmp_values = [1, 4, 8, 16, 24, 32]
        mpi_values = [1, 2, 4, 8, 16]
        hybrid_values = [(2, 8), (2, 16), (4, 4), (4, 8), (8, 2), (8, 4), (16, 2)]
    elif args.plan == "i9-full":
        openmp_values = [1, 2, 4, 8, 16, 24, 32]
        mpi_values = [1, 2, 4, 8, 16, 24, 32]
        hybrid_values = [
            (1, 8), (1, 16), (1, 24), (1, 32),
            (2, 4), (2, 8), (2, 12), (2, 16),
            (4, 2), (4, 4), (4, 6), (4, 8),
            (8, 2), (8, 4),
            (16, 1), (16, 2),
        ]
    else:
        openmp_values = args.openmp_list or [args.omp_threads]
        mpi_values = args.mpi_list or [args.mpi_ranks]
        hybrid_values = args.hybrid_list or [(args.mpi_ranks, args.omp_threads)]

    if args.openmp_list is not None:
        openmp_values = args.openmp_list
    if args.mpi_list is not None:
        mpi_values = args.mpi_list
    if args.hybrid_list is not None:
        hybrid_values = args.hybrid_list

    openmp_values = unique_ints(v for v in openmp_values if v <= args.cpu_threads)
    mpi_values = unique_ints(v for v in mpi_values if v <= args.cpu_threads)
    hybrid_values = [
        (r, t) for r, t in hybrid_values
        if r >= 1 and t >= 1 and r * t <= args.cpu_threads
    ]
    return openmp_values, mpi_values, hybrid_values


def make_base_cases(args: argparse.Namespace) -> List[RunCase]:
    openmp_values, mpi_values, hybrid_values = plan_values(args)
    cases: List[RunCase] = []

    if args.include_serial_200d:
        cases.append(
            RunCase(
                run_name="serial_200d",
                family="serial_200d",
                label="Serial 200D CPU",
                build_target="serial-200d",
                command=["./serial/summarizer_200d"],
                glove_file="glove.6B.200d.txt",
                notes="Optional legacy 200D baseline.",
            )
        )

    cases.append(
        RunCase(
            run_name="serial_300d",
            family="serial_300d",
            label="Serial 300D CPU",
            build_target="serial-300d",
            command=["./serial/summarizer_300d"],
            glove_file="glove.6B.300d.txt",
            notes="Main 300D baseline.",
        )
    )

    for threads in openmp_values:
        cases.append(
            RunCase(
                run_name=f"openmp_300d_t{threads}",
                family="openmp_300d",
                label=f"OpenMP 300D CPU, {threads} threads",
                build_target="openmp-mode",
                command=["./openmp/summarizer_openmp_mode"],
                glove_file="glove.6B.300d.txt",
                omp_threads=threads,
                config=f"threads={threads}",
            )
        )

    for ranks in mpi_values:
        cases.append(
            RunCase(
                run_name=f"mpi_300d_r{ranks}",
                family="mpi_300d",
                label=f"MPI 300D CPU, {ranks} ranks",
                build_target="mpi-mode",
                command=["mpirun", "-np", str(ranks), "./mpi/summarizer_mpi_mode"],
                glove_file="glove.6B.300d.txt",
                mpi_ranks=ranks,
                config=f"ranks={ranks}",
            )
        )

    for ranks, threads in hybrid_values:
        cases.append(
            RunCase(
                run_name=f"hybrid_300d_r{ranks}_t{threads}",
                family="mpi_openmp_hybrid_300d",
                label=f"Hybrid 300D, {ranks} ranks x {threads} threads",
                build_target="mpi_openmp_hybrid",
                command=["mpirun", "-np", str(ranks), "./mpi/mpi_openmp_hybrid"],
                glove_file="glove.6B.300d.txt",
                mpi_ranks=ranks,
                omp_threads=threads,
                config=f"ranks={ranks};threads={threads}",
            )
        )

    if not args.skip_cuda:
        cases.append(
            RunCase(
                run_name="cuda_300d",
                family="cuda_300d",
                label="CUDA 300D GPU",
                build_target="cuda_app",
                command=["./cuda/summarizer_cuda"],
                glove_file="glove.6B.300d.txt",
                cuda_enabled="yes",
                config="gpu=rtx5090",
                notes="CUDA implementation using the same 300D GloVe configuration as the CPU modes.",
            )
        )
        for threads in openmp_values:
            cases.append(
                RunCase(
                    run_name=f"cuda_openmp_hybrid_300d_t{threads}",
                    family="cuda_openmp_hybrid_300d",
                    label=f"CUDA + OpenMP Hybrid 300D, {threads} CPU threads",
                    build_target="cuda_openmp_hybrid",
                    command=["./cuda/summarizer_cuda_openmp_hybrid"],
                    glove_file="glove.6B.300d.txt",
                    omp_threads=threads,
                    cuda_enabled="yes",
                    config=f"gpu=rtx5090;threads={threads}",
                    notes="CUDA kernels for covariance and attention, OpenMP for CPU-side preprocessing and scoring.",
                )
            )

    return cases


def expand_repeats(cases: List[RunCase], repeats: int) -> List[Tuple[RunCase, int, str]]:
    expanded: List[Tuple[RunCase, int, str]] = []
    for case in cases:
        for repeat in range(1, max(1, repeats) + 1):
            suffix = f"_rep{repeat}" if repeats > 1 else ""
            expanded.append((case, repeat, f"{case.run_name}{suffix}"))
    return expanded


def read_input_text(args: argparse.Namespace) -> str:
    if args.input_text is not None:
        return args.input_text.strip() + "\n"
    input_path = PROJECT_ROOT / args.input_file
    text = input_path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        raise SystemExit(f"Input file is empty: {input_path}")
    return text + "\n"


def run_command(
    command: List[str],
    input_text: Optional[str] = None,
    env_extra: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )


def parse_float(text: str) -> Optional[float]:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def parse_int(text: str) -> Optional[int]:
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def parse_output(output: str) -> Dict[str, object]:
    data: Dict[str, object] = {"stages": {}, "stage_ops": {}}

    stage_re = re.compile(
        r"^\s*([A-Za-z0-9_+\-]+)\s+([0-9]+(?:\.[0-9]+)?)\s+([0-9]+)\s+",
        re.MULTILINE,
    )
    total_re = re.compile(r"^\s*TOTAL\s+([0-9]+(?:\.[0-9]+)?)\s*$", re.MULTILINE)

    stages: Dict[str, float] = {}
    stage_ops: Dict[str, int] = {}
    for match in stage_re.finditer(output):
        stages[match.group(1)] = float(match.group(2))
        stage_ops[match.group(1)] = int(match.group(3))
    data["stages"] = stages
    data["stage_ops"] = stage_ops

    total_match = total_re.search(output)
    if total_match:
        data["total_s"] = float(total_match.group(1))

    resource_patterns = {
        "functional_wall_s": r"Functional wall time\s+([0-9]+(?:\.[0-9]+)?)\s+s",
        "user_cpu_s": r"User CPU time\s+([0-9]+(?:\.[0-9]+)?)\s+s",
        "system_cpu_s": r"System CPU time\s+([0-9]+(?:\.[0-9]+)?)\s+s",
        "total_cpu_all_ranks_s": r"Total CPU time, all ranks\s+([0-9]+(?:\.[0-9]+)?)\s+s",
        "avg_cpu_per_rank_s": r"Average CPU time per rank\s+([0-9]+(?:\.[0-9]+)?)\s+s",
        "max_cpu_one_rank_s": r"Max CPU time on one rank\s+([0-9]+(?:\.[0-9]+)?)\s+s",
        "cpu_util_percent": r"Approx(?: total)? CPU utilization\s+([0-9]+(?:\.[0-9]+)?)\s+%",
        "peak_rss_mb": r"Peak resident memory\s+([0-9]+(?:\.[0-9]+)?)\s+MB",
        "avg_peak_rss_mb": r"Average peak RSS per rank\s+([0-9]+(?:\.[0-9]+)?)\s+MB",
        "max_peak_rss_mb": r"Max peak RSS on one rank\s+([0-9]+(?:\.[0-9]+)?)\s+MB",
        "sum_peak_rss_mb": r"Sum peak RSS across ranks\s+([0-9]+(?:\.[0-9]+)?)\s+MB",
    }
    for key, pattern in resource_patterns.items():
        match = re.search(pattern, output)
        if match:
            data[key] = float(match.group(1))

    gpu_match = re.search(r"GPU:\s+(.+?)\s+\(SM\s+([0-9]+)\.([0-9]+),\s+([0-9]+)\s+SMs,\s+([0-9]+)\s+MB VRAM\)", output)
    if gpu_match:
        data["detected_gpu_name"] = gpu_match.group(1).strip()
        data["detected_gpu_sm"] = f"{gpu_match.group(2)}.{gpu_match.group(3)}"
        data["detected_gpu_sms"] = parse_int(gpu_match.group(4))
        data["detected_gpu_memory_mb"] = parse_int(gpu_match.group(5))

    for line in output.splitlines():
        line = line.strip()
        if not line.startswith("COMPARE_"):
            continue

        parts = line.split(",")
        tag = parts[0]
        if tag == "COMPARE_CONFIG":
            for i in range(1, len(parts) - 1, 2):
                value = parse_int(parts[i + 1])
                data[parts[i].lower()] = value if value is not None else parts[i + 1]
        elif tag == "COMPARE_COUNTS":
            for i in range(1, len(parts) - 1, 2):
                data[parts[i].lower()] = parse_int(parts[i + 1])
        elif tag == "COMPARE_SELECTED":
            rank = parts[2]
            data[f"rank{rank}_sentence_id"] = parse_int(parts[4])
            data[f"rank{rank}_score"] = parse_float(parts[6])
        elif tag == "COMPARE_ATTENTION":
            for i in range(1, len(parts) - 1, 2):
                data[f"attention_{parts[i].lower()}"] = parse_float(parts[i + 1])
        elif tag == "COMPARE_OUTPUT":
            for i in range(1, len(parts) - 1, 2):
                data[f"output_{parts[i].lower()}"] = parse_float(parts[i + 1])

    return data


def pick_stage(stages: Dict[str, float], *names: str) -> Optional[float]:
    for name in names:
        if name in stages:
            return stages[name]
    return None


def pick_stage_ops(stage_ops: Dict[str, int], *names: str) -> Optional[int]:
    for name in names:
        if name in stage_ops:
            return stage_ops[name]
    return None


def safe_diff(current: object, baseline: object) -> Optional[float]:
    if current is None or baseline is None:
        return None
    try:
        return abs(float(current) - float(baseline))
    except (TypeError, ValueError):
        return None


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


def fmt(value: object) -> object:
    return "" if value is None else value


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field)) for field in fieldnames})


def build_targets(cases: List[RunCase], args: argparse.Namespace) -> Dict[str, str]:
    statuses: Dict[str, str] = {}
    if args.no_build:
        for case in cases:
            statuses[case.build_target] = "skipped"
        return statuses

    for target in sorted({case.build_target for case in cases}):
        env_extra = {}
        if target == "cuda_app":
            env_extra["NVCC_FLAGS"] = args.nvcc_flags
        elif target == "cuda_openmp_hybrid":
            env_extra["NVCC_OMP_FLAGS"] = f"{args.nvcc_flags} -Xcompiler -fopenmp"
        print(f"=== build {target} ===")
        build = run_command(["make", target], env_extra=env_extra)
        log_path = PROJECT_ROOT / "results" / "build_logs"
        log_path.mkdir(parents=True, exist_ok=True)
        (log_path / f"{target}.log").write_text(build.stdout, encoding="utf-8", errors="replace")
        statuses[target] = "ok" if build.returncode == 0 else "build_failed"
        print(statuses[target])
        if build.returncode != 0 and not args.continue_on_error:
            print(build.stdout)
            raise SystemExit(f"Build failed for target {target}")
    return statuses


def run_env_for(case: RunCase) -> Dict[str, str]:
    env = {
        "OMP_NUM_THREADS": str(case.omp_threads),
        "OMP_PROC_BIND": "close",
        "OMP_PLACES": "cores",
    }
    return env


def main() -> int:
    args = parse_args()
    experiment_id = args.experiment_id or datetime.now().strftime("EXP%Y%m%d_%H%M%S")
    input_text = read_input_text(args)
    base_cases = make_base_cases(args)
    run_cases = expand_repeats(base_cases, args.repeats)

    if args.list_plan:
        print(f"Plan: {args.plan}")
        print(f"Machine: {args.cpu_model}, {args.cpu_cores} cores / {args.cpu_threads} threads, {args.gpu_model}")
        print(f"Repeats: {args.repeats}")
        print()
        for case, repeat, run_id in run_cases:
            print(
                f"{run_id}: family={case.family}, config={case.config}, "
                f"mpi_ranks={case.mpi_ranks}, omp_threads={case.omp_threads}, "
                f"workers={case.total_workers}, cuda={case.cuda_enabled}"
            )
        return 0

    results_dir = PROJECT_ROOT / args.results_dir / experiment_id
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment: {experiment_id}")
    print(f"Plan:       {args.plan}")
    print(f"Machine:    {args.cpu_model}, {args.cpu_cores} cores / {args.cpu_threads} threads, {args.gpu_model}")
    print(f"Runs:       {len(run_cases)}")
    print(f"Logs:       {results_dir.relative_to(PROJECT_ROOT)}")
    print()

    build_statuses = build_targets(base_cases, args)
    parsed_runs: List[Dict[str, object]] = []

    for case, repeat, run_id in run_cases:
        build_status = build_statuses.get(case.build_target, "unknown")
        output = ""
        returncode = None
        status = build_status

        print(f"=== run {run_id} ===")
        if build_status in {"ok", "skipped"}:
            run = run_command(case.command, input_text=input_text, env_extra=run_env_for(case))
            output = run.stdout
            returncode = run.returncode
            status = "ok" if run.returncode == 0 else "run_failed"
            print(status)
        else:
            print("skipped because build failed")

        log_file = results_dir / f"{run_id}.log"
        log_file.write_text(output, encoding="utf-8", errors="replace")

        parsed = parse_output(output)
        parsed.update(
            {
                "case": case,
                "repeat": repeat,
                "run_id": run_id,
                "status": status,
                "returncode": returncode,
                "log_file": str(log_file.relative_to(PROJECT_ROOT)),
            }
        )
        parsed_runs.append(parsed)

        if status != "ok" and not args.continue_on_error:
            print(f"Stopping after failed run: {run_id}")
            break
        print()

    serial_runs = [r for r in parsed_runs if r["status"] == "ok" and r["case"].family == "serial_300d"]
    baseline = min(serial_runs, key=lambda r: float(r.get("total_s") or float("inf"))) if serial_runs else {}
    baseline_total = baseline.get("total_s")
    baseline_selected = (baseline.get("rank1_sentence_id"), baseline.get("rank2_sentence_id"))
    baseline_attention_checksum = baseline.get("attention_checksum")
    baseline_output_checksum = baseline.get("output_checksum")
    baseline_embed_dim = baseline.get("embed_dim")
    baseline_num_heads = baseline.get("num_heads")
    baseline_head_dim = baseline.get("head_dim")

    performance_rows: List[Dict[str, object]] = []
    accuracy_rows: List[Dict[str, object]] = []

    for result in parsed_runs:
        case: RunCase = result["case"]  # type: ignore[assignment]
        stages = result.get("stages", {})
        stage_ops = result.get("stage_ops", {})
        assert isinstance(stages, dict)
        assert isinstance(stage_ops, dict)

        current_total = result.get("total_s") or result.get("functional_wall_s")
        speedup = None
        if baseline_total and current_total:
            speedup = float(baseline_total) / float(current_total)

        efficiency = None
        if speedup is not None and case.total_workers > 0 and case.cuda_enabled != "yes":
            efficiency = 100.0 * speedup / case.total_workers

        peak_memory = (
            result.get("peak_rss_mb")
            or result.get("max_peak_rss_mb")
            or result.get("avg_peak_rss_mb")
        )

        performance_rows.append(
            {
                "experiment_id": experiment_id,
                "input_id": args.input_id,
                "run_id": result.get("run_id"),
                "family": case.family,
                "config": case.config,
                "repeat": result.get("repeat"),
                "status": result.get("status"),
                "cpu_model": args.cpu_model,
                "cpu_cores": args.cpu_cores,
                "cpu_threads_available": args.cpu_threads,
                "gpu_model": args.gpu_model,
                "cuda_version": args.cuda_version,
                "ram_gb": args.ram_gb,
                "glove_file": case.glove_file,
                "embed_dim": result.get("embed_dim"),
                "tokens": result.get("tokens"),
                "sentences": result.get("sentences"),
                "mpi_ranks": case.mpi_ranks,
                "omp_threads": case.omp_threads,
                "total_workers": case.total_workers,
                "cuda": case.cuda_enabled,
                "total_s": current_total,
                "load_s": pick_stage(stages, "load_glove", "load_glove_replicated"),
                "covariance_s": pick_stage(stages, "covariance_matrix", "covariance_matrix_mpi", "covariance_matrix_hybrid"),
                "jacobi_s": pick_stage(stages, "jacobi_eigen"),
                "whitening_s": pick_stage(stages, "build_whitening_W", "build_whitening_W_hybrid"),
                "vectorize_s": pick_stage(stages, "tokenize+vectorize", "tokenize+vectorize_mpi", "tokenize+vectorize_hybrid"),
                "positional_s": pick_stage(stages, "positional_encoding"),
                "attention_s": pick_stage(stages, "self_attention", "self_attention_mpi", "self_attention_hybrid"),
                "scoring_s": pick_stage(stages, "sentence_scoring"),
                "cpu_util_percent": result.get("cpu_util_percent"),
                "peak_memory_mb": peak_memory,
                "attention_ops": pick_stage_ops(stage_ops, "self_attention", "self_attention_mpi", "self_attention_hybrid"),
                "speedup_vs_best_serial300": speedup,
                "efficiency_percent": efficiency,
                "log_file": result.get("log_file"),
                "notes": case.notes,
            }
        )

        selected = (result.get("rank1_sentence_id"), result.get("rank2_sentence_id"))
        selected_match = selected == baseline_selected
        config_match = (
            result.get("embed_dim") == baseline_embed_dim
            and result.get("num_heads") == baseline_num_heads
            and result.get("head_dim") == baseline_head_dim
        )
        attention_diff = safe_diff(result.get("attention_checksum"), baseline_attention_checksum)
        output_diff = safe_diff(result.get("output_checksum"), baseline_output_checksum)

        row_min = result.get("attention_row_sum_min")
        row_max = result.get("attention_row_sum_max")
        row_sum_ok = False
        if row_min is not None and row_max is not None:
            row_sum_ok = abs(float(row_min) - 1.0) <= 1e-4 and abs(float(row_max) - 1.0) <= 1e-4

        attention_close = attention_diff is not None and attention_diff <= 1e-4
        output_close = output_diff is not None and output_diff <= 1e-4

        if case.family == "serial_300d":
            accuracy_status = "baseline"
        elif not config_match:
            accuracy_status = "different_config"
        elif selected_match and attention_close and output_close and row_sum_ok:
            accuracy_status = "match"
        elif selected_match and row_sum_ok:
            accuracy_status = "same_summary_numeric_diff"
        else:
            accuracy_status = "different_output"

        accuracy_rows.append(
            {
                "experiment_id": experiment_id,
                "input_id": args.input_id,
                "run_id": result.get("run_id"),
                "family": case.family,
                "config": case.config,
                "repeat": result.get("repeat"),
                "status": result.get("status"),
                "glove_file": case.glove_file,
                "embed_dim": result.get("embed_dim"),
                "tokens": result.get("tokens"),
                "sentences": result.get("sentences"),
                "selected_sentences": ";".join(str(x) for x in selected if x is not None),
                "rank1_score": result.get("rank1_score"),
                "rank2_score": result.get("rank2_score"),
                "same_selected_as_serial300": yes_no(selected_match),
                "config_same_as_serial300": yes_no(config_match),
                "attention_checksum": result.get("attention_checksum"),
                "attention_checksum_diff": attention_diff,
                "output_checksum": result.get("output_checksum"),
                "output_checksum_diff": output_diff,
                "attention_row_sum_min": row_min,
                "attention_row_sum_max": row_max,
                "attention_rows_valid": yes_no(row_sum_ok),
                "accuracy_status": accuracy_status,
                "log_file": result.get("log_file"),
                "notes": case.notes,
            }
        )

    perf_fields = [
        "experiment_id", "input_id", "run_id", "family", "config", "repeat", "status",
        "cpu_model", "cpu_cores", "cpu_threads_available", "gpu_model", "cuda_version", "ram_gb",
        "glove_file", "embed_dim", "tokens", "sentences",
        "mpi_ranks", "omp_threads", "total_workers", "cuda",
        "total_s", "load_s", "covariance_s", "jacobi_s", "whitening_s",
        "vectorize_s", "positional_s", "attention_s", "scoring_s",
        "cpu_util_percent", "peak_memory_mb", "attention_ops",
        "speedup_vs_best_serial300", "efficiency_percent", "log_file", "notes",
    ]

    accuracy_fields = [
        "experiment_id", "input_id", "run_id", "family", "config", "repeat", "status",
        "glove_file", "embed_dim", "tokens", "sentences",
        "selected_sentences", "rank1_score", "rank2_score",
        "same_selected_as_serial300", "config_same_as_serial300",
        "attention_checksum", "attention_checksum_diff",
        "output_checksum", "output_checksum_diff",
        "attention_row_sum_min", "attention_row_sum_max", "attention_rows_valid",
        "accuracy_status", "log_file", "notes",
    ]

    perf_path = PROJECT_ROOT / "docs/final_performance_comparison.csv"
    acc_path = PROJECT_ROOT / "docs/final_accuracy_comparison.csv"
    write_csv(perf_path, perf_fields, performance_rows)
    write_csv(acc_path, accuracy_fields, accuracy_rows)

    print("Created:")
    print(f"  {perf_path.relative_to(PROJECT_ROOT)}")
    print(f"  {acc_path.relative_to(PROJECT_ROOT)}")
    print("Raw logs:")
    print(f"  {results_dir.relative_to(PROJECT_ROOT)}")

    failed = [row["run_id"] for row in performance_rows if row["status"] != "ok"]
    if failed:
        print("Failed runs:")
        for run_id in failed:
            print(f"  {run_id}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
