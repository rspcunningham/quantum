"""Multi-backend apples-to-apples benchmark comparison runner."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from benchmarks.backends import create_backend, known_backends
from benchmarks.backends.base import BackendAdapter
from benchmarks.cases import ALL_CASES, BenchmarkCase
from benchmarks.ir import build_circuit_ir
from benchmarks.run import SHOT_COUNTS, check_correctness, count_ops, get_git_hash, infer_resources

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "KMP_DUPLICATE_LIB_OK",
)


def _environment_metadata() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "thread_env": {name: os.environ.get(name) for name in THREAD_ENV_VARS if os.environ.get(name) is not None},
    }


def _sync_if_native(adapter: BackendAdapter, prepared_case: Any) -> None:
    if adapter.name != "native":
        return
    device = getattr(prepared_case, "device", None)
    if not isinstance(device, torch.device):
        return
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _case_qubits(case: BenchmarkCase) -> int:
    return case.n_qubits if case.n_qubits is not None else infer_resources(case.circuit)[0]


def _run_case_backend(
    case: BenchmarkCase,
    *,
    backend: BackendAdapter,
    repetitions: int,
    verbose: bool,
    suite: str,
    git_hash: str,
    env_meta: dict[str, Any],
) -> dict[str, Any]:
    case_ir = build_circuit_ir(case.circuit, n_qubits=case.n_qubits)
    supported, support_reason = backend.supports(case_ir)
    ops = count_ops(case.circuit)
    display_qubits = _case_qubits(case)

    base_row: dict[str, Any] = {
        "backend": backend.name,
        "backend_version": backend.version_info(),
        "case": case.name,
        "suite": suite,
        "git_hash": git_hash,
        "env": env_meta,
        "n_qubits": display_qubits,
        "ops": ops,
        "is_dynamic": case_ir.is_dynamic,
        "supported": supported,
        "support_reason": support_reason,
        "shot_counts": SHOT_COUNTS,
        "repetitions": repetitions,
        "compile_time_s": None,
        "warmup_time_s": None,
        "times_s": {},
        "times_min_s": {},
        "times_max_s": {},
        "metric_shots": None,
        "correct": None,
        "errors": [],
    }

    if not supported:
        if verbose:
            print(f"\n{case.name} [{backend.name}]")
            print(f"  supported: NO ({support_reason})")
        return base_row

    try:
        compile_start = time.perf_counter()
        prepared = backend.prepare(case_ir)
        base_row["compile_time_s"] = round(time.perf_counter() - compile_start, 4)
    except Exception as error:
        base_row["supported"] = False
        base_row["support_reason"] = f"prepare failed: {error}"
        base_row["errors"] = [str(error)]
        if verbose:
            print(f"\n{case.name} [{backend.name}]")
            print(f"  prepare failed: {error}")
        return base_row

    # Warmup (not timed in per-shot metrics)
    try:
        _sync_if_native(backend, prepared)
        warmup_start = time.perf_counter()
        _ = backend.run(prepared, 1, warmup=True)
        _sync_if_native(backend, prepared)
        base_row["warmup_time_s"] = round(time.perf_counter() - warmup_start, 4)
    except Exception as error:
        base_row["correct"] = False
        base_row["errors"] = [f"warmup failed: {error}"]
        if verbose:
            print(f"\n{case.name} [{backend.name}]")
            print(f"  warmup failed: {error}")
        return base_row

    result_max: dict[str, int] | None = None
    run_errors: list[str] = []

    for shots in SHOT_COUNTS:
        shot_times: list[float] = []
        last_result: dict[str, int] | None = None
        for _rep in range(repetitions):
            try:
                _sync_if_native(backend, prepared)
                t0 = time.perf_counter()
                result = backend.run(prepared, shots)
                _sync_if_native(backend, prepared)
                dt = time.perf_counter() - t0
            except Exception as error:
                run_errors.append(f"{shots} shots: {error}")
                shot_times = []
                last_result = None
                break
            shot_times.append(dt)
            last_result = result

        if not shot_times or last_result is None:
            break

        base_row["times_s"][str(shots)] = round(statistics.median(shot_times), 4)
        base_row["times_min_s"][str(shots)] = round(min(shot_times), 4)
        base_row["times_max_s"][str(shots)] = round(max(shot_times), 4)

        if shots == max(SHOT_COUNTS):
            result_max = last_result

    completed_shots = [s for s in SHOT_COUNTS if str(s) in base_row["times_s"]]
    if completed_shots:
        metric_shots = max(completed_shots)
        base_row["metric_shots"] = metric_shots
        if metric_shots == max(SHOT_COUNTS):
            assert result_max is not None
            correct, errors = check_correctness(result_max, case.expected, case.tolerance)
        else:
            correct = False
            errors = [f"incomplete shot ladder, highest completed={metric_shots}"]
        base_row["correct"] = correct
        base_row["errors"] = [*errors, *run_errors]
    else:
        base_row["correct"] = False
        base_row["errors"] = run_errors if run_errors else ["no shots completed"]

    if verbose:
        print(f"\n{case.name} [{backend.name}] ({display_qubits} qubits)")
        if base_row["times_s"]:
            shots_str = "    ".join(
                f"{s} -> {base_row['times_s'][str(s)]:.3f}s"
                if str(s) in base_row["times_s"] else
                f"{s} -> n/a"
                for s in SHOT_COUNTS
            )
            print(f"  shots:    {shots_str}")
        else:
            print("  shots:    none")
        print(
            "  compile:  "
            f"{base_row['compile_time_s']:.3f}s | warmup: {base_row['warmup_time_s']:.3f}s"
            if base_row["compile_time_s"] is not None and base_row["warmup_time_s"] is not None else
            "  compile:  n/a"
        )
        if base_row["correct"] is True:
            print("  correct:  PASS")
        else:
            error_text = "; ".join(base_row["errors"]) if base_row["errors"] else "unknown"
            print(f"  correct:  FAIL — {error_text}")

    return base_row


def _select_cases(
    *,
    requested_cases: list[str] | None,
    suite: str,
) -> list[BenchmarkCase]:
    case_map = {case_fn().name: case_fn for case_fn in ALL_CASES}
    if len(case_map) != len(ALL_CASES):
        raise RuntimeError("Duplicate benchmark case names detected.")

    if requested_cases is None:
        selected = [case_fn() for case_fn in ALL_CASES]
    else:
        unknown = [name for name in requested_cases if name not in case_map]
        if unknown:
            raise RuntimeError(f"Unknown case(s): {', '.join(unknown)}")
        selected = [case_map[name]() for name in requested_cases]

    if suite == "full":
        return selected

    static_cases: list[BenchmarkCase] = []
    for case in selected:
        ir = build_circuit_ir(case.circuit, n_qubits=case.n_qubits)
        if not ir.is_dynamic:
            static_cases.append(case)
    return static_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare benchmark results across simulator backends")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["native", "aer"],
        help=f"Backend names to run. Known: {', '.join(known_backends())}",
    )
    parser.add_argument(
        "--suite",
        choices=["full", "static"],
        default="full",
        help="Case subset to run: full suite or static-only cases.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="Optional explicit case-name subset.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Timed repetitions per shot count (median reported).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Per-case backend details.")
    args = parser.parse_args()

    if args.repetitions <= 0:
        parser.error("--repetitions must be >= 1")

    try:
        cases = _select_cases(requested_cases=args.cases, suite=args.suite)
    except RuntimeError as error:
        parser.error(str(error))

    if not cases:
        parser.error("No cases selected after applying suite/case filters.")

    backends: list[BackendAdapter] = []
    for backend_name in args.backends:
        try:
            backend = create_backend(backend_name)
        except ValueError as error:
            parser.error(str(error))
        backends.append(backend)

    git_hash = get_git_hash()
    env_meta = _environment_metadata()
    print(f"Backends: {[b.name for b in backends]} | Git: {git_hash} | Suite: {args.suite} | Reps: {args.repetitions}")
    print(f"Cases: {[case.name for case in cases]}")
    print(f"Env: python={env_meta['python_version']} | platform={env_meta['platform']}")

    for backend in backends:
        availability = backend.availability()
        status = "available" if availability.available else f"unavailable ({availability.reason})"
        version = backend.version_info()
        print(f"  - {backend.name}: {status} | version={version}")

    results: list[dict[str, Any]] = []
    failures: list[str] = []

    for case in cases:
        for backend in backends:
            result = _run_case_backend(
                case,
                backend=backend,
                repetitions=args.repetitions,
                verbose=args.verbose,
                suite=args.suite,
                git_hash=git_hash,
                env_meta=env_meta,
            )
            results.append(result)
            if result["supported"] and result["correct"] is False:
                failures.append(f"{case.name}:{backend.name}")

    # Summary by backend and shot ladder.
    print("\nTotals by backend (execution median sums):")
    for backend in backends:
        backend_rows = [r for r in results if r["backend"] == backend.name and r["supported"]]
        if not backend_rows:
            print(f"  {backend.name}: no supported cases")
            continue
        parts: list[str] = []
        for shots in SHOT_COUNTS:
            key = str(shots)
            if all(key in row["times_s"] for row in backend_rows):
                total = sum(float(row["times_s"][key]) for row in backend_rows)
                parts.append(f"{shots} -> {total:.2f}s")
            else:
                parts.append(f"{shots} -> n/a")
        print(f"  {backend.name}: " + "    ".join(parts))

    rusage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss_mb = rusage.ru_maxrss / (1024 * 1024) if sys.platform == "darwin" else rusage.ru_maxrss / 1024
    print(f"Peak RSS: {peak_rss_mb:.0f} MB")

    if failures:
        print(f"\nFAILED: {', '.join(failures)}")

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    output_path = results_dir / f"compare-{timestamp}.jsonl"
    with output_path.open("w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {output_path}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
