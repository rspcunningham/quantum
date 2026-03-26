"""Benchmark runner for the Aer reference backend.

Runs the full case suite through Qiskit Aer and writes results to
benchmarks/results/<timestamp>-aer.jsonl. Used to regenerate the
pinned aer-reference.jsonl, not part of the optimization loop.
"""

import contextlib
import json
import signal
import time
from datetime import datetime
from pathlib import Path

from benchmarks.cases import ALL_CASES
from benchmarks.backends.aer_adapter import AerAdapter
from benchmarks.ir import build_circuit_ir
from benchmarks.run import (
    SHOTS, TIMEOUT, count_ops, classify_workload, check_correctness,
    get_git_hash, get_memory_limit_gb, estimate_memory_gb, get_peak_rss_mb,
    print_results,
)
from quantum import infer_resources


class _CaseTimeout(Exception):
    pass


@contextlib.contextmanager
def _time_limit(seconds: int):
    """SIGALRM timeout — works for Aer because it releases the GIL."""
    def _raise(signum, frame):
        raise _CaseTimeout()
    old = signal.signal(signal.SIGALRM, _raise)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def main() -> None:
    aer = AerAdapter()
    avail = aer.availability()
    if not avail.available:
        print(f"Aer unavailable: {avail.reason}")
        return

    memory_limit_gb = get_memory_limit_gb()
    git_hash = get_git_hash()
    print(f"Backend: aer | Git: {git_hash} | Shots: {SHOTS} | Memory limit: {memory_limit_gb:.0f} GB")

    instantiated = [case_fn() for case_fn in ALL_CASES]
    instantiated.sort(key=lambda c: c.n_qubits if c.n_qubits is not None else infer_resources(c.circuit)[0])

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    output_path = results_dir / f"{timestamp}-aer.jsonl"

    results: list[dict] = []

    with open(output_path, "w") as jsonl_f:
        for case in instantiated:
            n_qubits = case.n_qubits
            if n_qubits is None:
                n_qubits = infer_resources(case.circuit)[0]
            ops = count_ops(case.circuit)
            workload = classify_workload(case.circuit)

            abort_reason = None
            result = None
            wall_elapsed = 0.0
            cpu_elapsed = 0.0

            estimated_gb = estimate_memory_gb(n_qubits)
            if estimated_gb > memory_limit_gb:
                abort_reason = f"skipped: {estimated_gb:.1f} GB estimated > {memory_limit_gb:.1f} GB limit ({n_qubits} qubits)"
            else:
                case_ir = build_circuit_ir(case.circuit, n_qubits=n_qubits)
                supported, reason = aer.supports(case_ir)
                if not supported:
                    abort_reason = f"aer unsupported: {reason}"
                else:
                    try:
                        prepared = aer.prepare(case_ir)
                        wall_start = time.perf_counter()
                        cpu_start = time.process_time()
                        with _time_limit(TIMEOUT):
                            result = aer.run(prepared, SHOTS)
                        wall_elapsed = time.perf_counter() - wall_start
                        cpu_elapsed = time.process_time() - cpu_start
                    except _CaseTimeout:
                        wall_elapsed = time.perf_counter() - wall_start
                        cpu_elapsed = time.process_time() - cpu_start
                        abort_reason = f"timeout: exceeded {TIMEOUT}s"
                    except Exception as error:
                        abort_reason = str(error).strip()

            if abort_reason is not None:
                correct = False
                errors = [abort_reason]
            elif result is not None:
                correct, errors = check_correctness(result, case.expected, case.tolerance)
            else:
                correct = False
                errors = ["no result"]

            total_ops = ops["gates"] + ops["measurements"] + ops["conditional"]
            ops_per_sec = round(total_ops / wall_elapsed, 1) if wall_elapsed > 0 else 0.0
            shots_per_sec = round(SHOTS / wall_elapsed, 1) if wall_elapsed > 0 else 0.0

            row = {
                "case": case.name,
                "backend": "aer",
                "git_hash": git_hash,
                "n_qubits": n_qubits,
                "workload": workload,
                "ops": ops,
                "shots": SHOTS,
                "time_s": round(wall_elapsed, 4),
                "cpu_s": round(cpu_elapsed, 4),
                "ops_per_sec": ops_per_sec,
                "shots_per_sec": shots_per_sec,
                "correct": correct,
                "errors": errors,
                "aborted": abort_reason is not None,
            }
            results.append(row)
            jsonl_f.write(json.dumps(row) + "\n")
            jsonl_f.flush()
            status = "PASS" if row["correct"] else ("ABORT" if row["aborted"] else "FAIL")
            print(f"  {row['case']}: {row['time_s']:.4f}s [{status}]")

    print(f"Wrote {output_path}")
    print_results(results)


if __name__ == "__main__":
    main()
