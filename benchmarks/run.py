"""Benchmark runner for the quantum simulator."""

import atexit
import argparse
import gc
import json
import multiprocessing as mp
import os
import resource
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import datetime
from multiprocessing.connection import Connection
from pathlib import Path

import torch

from benchmarks.backends.aer_adapter import AerAdapter
from benchmarks.cases import BenchmarkCase, ALL_CASES, CORE_CASES
from benchmarks.ir import build_circuit_ir
from quantum import run_simulation, infer_resources
from quantum.gates import Gate, Measurement, ConditionalGate, Circuit
from quantum.system import _circuit_compilation_cache

SHOT_COUNTS = [1000, 10000]
OP_KINDS = ("gates", "measurements", "conditional")
_CASE_WORKER: "_CaseWorker | None" = None
_CASE_WORKER_CONFIG: tuple[str, bool, str, str] | None = None
_CASE_WORKER_ATEXIT_REGISTERED = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def clear_device_cache(device: torch.device) -> None:
    """Best-effort cache cleanup after OOM to keep benchmark process alive."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def is_oom_error(error: BaseException) -> bool:
    msg = str(error).lower()
    return "out of memory" in msg or "oom" in msg


def get_git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def count_ops(circuit: Circuit) -> dict[str, int]:
    """Count operations in a circuit by type."""
    counts = {"gates": 0, "measurements": 0, "conditional": 0}

    def _walk(ops: Sequence[Gate | Measurement | ConditionalGate | Circuit]) -> None:
        for op in ops:
            if isinstance(op, Circuit):
                _walk(op.operations)
            elif isinstance(op, Gate):
                counts["gates"] += 1
            elif isinstance(op, Measurement):
                counts["measurements"] += 1
            else:
                counts["conditional"] += 1

    _walk(circuit.operations)
    return counts


def analyze_workload(circuit: Circuit) -> dict[str, bool | str]:
    """Classify circuit as static or dynamic based on control-flow structure."""
    flattened: list[Gate | Measurement | ConditionalGate] = []

    def _flatten(ops: Sequence[Gate | Measurement | ConditionalGate | Circuit]) -> None:
        for op in ops:
            if isinstance(op, Circuit):
                _flatten(op.operations)
            else:
                flattened.append(op)

    _flatten(circuit.operations)

    has_conditional = False
    has_non_terminal_measurement = False
    seen_measurement = False

    for op in flattened:
        if isinstance(op, ConditionalGate):
            has_conditional = True
            continue
        if isinstance(op, Measurement):
            seen_measurement = True
            continue
        if seen_measurement:
            has_non_terminal_measurement = True

    workload_class = "dynamic" if (has_conditional or has_non_terminal_measurement) else "static"
    return {
        "has_conditional": has_conditional,
        "has_non_terminal_measurement": has_non_terminal_measurement,
        "workload_class": workload_class,
    }


def get_memory_stats(device: torch.device) -> dict[str, float]:
    """Get GPU memory stats in MB. Returns empty dict for CPU."""
    stats: dict[str, float] = {}
    if device.type == "cuda":
        stats["allocated_mb"] = torch.cuda.memory_allocated(device) / 1024 / 1024
        stats["max_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 1024 / 1024
        stats["reserved_mb"] = torch.cuda.memory_reserved(device) / 1024 / 1024
    elif device.type == "mps":
        stats["allocated_mb"] = torch.mps.current_allocated_memory() / 1024 / 1024
        stats["driver_mb"] = torch.mps.driver_allocated_memory() / 1024 / 1024
    return stats


def get_peak_rss_mb() -> float:
    """Process peak RSS in MB (macOS reports bytes, Linux reports KB)."""
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    return rusage.ru_maxrss / (1024 * 1024) if sys.platform == "darwin" else rusage.ru_maxrss / 1024


def get_process_rss_mb() -> float | None:
    """Current process RSS in MB via `ps`; returns None when unavailable."""
    try:
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            capture_output=True,
            text=True,
            check=True,
        )
        rss_kb = float(result.stdout.strip())
        return rss_kb / 1024.0
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
        return None


def add_process_memory_stats(memory_stats: dict[str, float], *, rss_before: float | None, rss_after: float | None) -> None:
    if rss_before is not None:
        memory_stats["process_rss_before_mb"] = rss_before
    if rss_after is not None:
        memory_stats["process_rss_after_mb"] = rss_after
    if rss_before is not None and rss_after is not None:
        memory_stats["process_rss_delta_mb"] = rss_after - rss_before
    memory_stats["process_peak_rss_mb"] = get_peak_rss_mb()


def check_correctness(
    result: dict[str, int],
    expected: dict[str, float],
    tolerance: float,
) -> tuple[bool, list[str]]:
    total = sum(result.values())
    observed = {k: v / total for k, v in result.items()}

    errors: list[str] = []
    for outcome, expected_freq in expected.items():
        observed_freq = observed.get(outcome, 0.0)
        if abs(observed_freq - expected_freq) > tolerance:
            errors.append(f'"{outcome}": expected {expected_freq:.2f}, got {observed_freq:.2f}')

    for outcome, freq in observed.items():
        if outcome not in expected and freq > tolerance:
            errors.append(f'"{outcome}": unexpected, got {freq:.2f}')

    return (len(errors) == 0, errors)


def _totals_for_rows(results: list[dict]) -> dict[str, float | None]:
    totals: dict[str, float | None] = {}
    for shot in SHOT_COUNTS:
        key = str(shot)
        if results and all(key in row["times_s"] for row in results):
            totals[key] = round(sum(float(row["times_s"][key]) for row in results), 4)
        else:
            totals[key] = None
    return totals


def _format_totals_line(totals: dict[str, float | None]) -> str:
    labels = {SHOT_COUNTS[0]: "cold", **{s: "warm" for s in SHOT_COUNTS[1:]}}
    parts: list[str] = []
    for shot in SHOT_COUNTS:
        key = str(shot)
        value = totals.get(key)
        label = labels[shot]
        if value is None:
            parts.append(f"{label}@{shot} \u2192 n/a")
        else:
            parts.append(f"{label}@{shot} \u2192 {value:.2f}s")
    return "    ".join(parts)


def _category_hotspot_summary(
    rows: list[dict],
    *,
    shot_key: str,
    top_n: int = 5,
) -> dict[str, list[dict[str, float | int | str]]]:
    rows_with_shot = [row for row in rows if shot_key in row["times_s"]]
    if not rows_with_shot:
        return {"cases": [], "operators_proxy": []}

    total_time = sum(float(row["times_s"][shot_key]) for row in rows_with_shot)
    case_entries: list[dict[str, float | int | str]] = []
    for row in sorted(rows_with_shot, key=lambda r: float(r["times_s"][shot_key]), reverse=True)[:top_n]:
        case_time = float(row["times_s"][shot_key])
        case_entries.append({
            "case": str(row["case"]),
            "time_s": round(case_time, 4),
            "share_pct": round((100.0 * case_time / total_time) if total_time > 0 else 0.0, 2),
            "gates": int(row["ops"]["gates"]),
            "measurements": int(row["ops"]["measurements"]),
            "conditional": int(row["ops"]["conditional"]),
        })

    op_proxy_entries: list[dict[str, float | int | str]] = []
    scored: list[tuple[float, str, dict]] = []
    for row in rows_with_shot:
        row_time = float(row["times_s"][shot_key])
        total_ops = int(row["ops"]["gates"] + row["ops"]["measurements"] + row["ops"]["conditional"])
        if total_ops <= 0:
            continue
        for op_kind in OP_KINDS:
            op_count = int(row["ops"][op_kind])
            if op_count <= 0:
                continue
            score = row_time * (op_count / total_ops)
            scored.append((score, op_kind, row))

    for score, op_kind, row in sorted(scored, key=lambda item: item[0], reverse=True)[:top_n]:
        op_proxy_entries.append({
            "op_kind": op_kind,
            "case": str(row["case"]),
            "proxy_time_s": round(float(score), 4),
            "case_time_s": round(float(row["times_s"][shot_key]), 4),
            "op_count": int(row["ops"][op_kind]),
            "case_total_ops": int(row["ops"]["gates"] + row["ops"]["measurements"] + row["ops"]["conditional"]),
        })

    return {"cases": case_entries, "operators_proxy": op_proxy_entries}


def _print_hotspot_block(
    rows: list[dict],
    *,
    label: str,
    shot_key: str,
    top_n: int = 5,
) -> None:
    summary = _category_hotspot_summary(rows, shot_key=shot_key, top_n=top_n)
    if not summary["cases"]:
        print(f"\nTop hotspots ({label}, {shot_key} shots): n/a")
        return

    print(f"\nTop case hotspots ({label}, {shot_key} shots):")
    for idx, entry in enumerate(summary["cases"], start=1):
        print(
            f"  {idx}. {entry['case']}: {entry['time_s']:.4f}s "
            f"({entry['share_pct']:.2f}%) | ops g/m/c="
            f"{entry['gates']}/{entry['measurements']}/{entry['conditional']}"
        )

    print(f"Top operator hotspots ({label}, {shot_key} shots, proxy):")
    for idx, entry in enumerate(summary["operators_proxy"], start=1):
        print(
            f"  {idx}. {entry['op_kind']} in {entry['case']}: "
            f"proxy {entry['proxy_time_s']:.4f}s (case {entry['case_time_s']:.4f}s, "
            f"count {entry['op_count']}/{entry['case_total_ops']})"
        )


def _print_case_row_verbose(row: dict) -> None:
    labels = {SHOT_COUNTS[0]: "cold", **{s: "warm" for s in SHOT_COUNTS[1:]}}
    times = row["times_s"]
    cpu_times = row["cpu_times_s"]
    metric_shots = row.get("metric_shots")
    total_ops = int(row["ops"]["gates"] + row["ops"]["measurements"] + row["ops"]["conditional"])
    shots_str = "    ".join(
        (
            f"{labels[s]}@{s} -> {times[str(s)]:.3f}s (cpu: {cpu_times[str(s)]:.3f}s)"
            if str(s) in times else
            f"{labels[s]}@{s} -> skipped"
        )
        for s in SHOT_COUNTS
    )
    mem = row.get("memory", {})
    mem_parts = [f"{k}: {float(v):.1f}" for k, v in mem.items()]
    mem_str = ", ".join(mem_parts) if mem_parts else "n/a"
    metric_label = f"at {metric_shots} shots" if metric_shots is not None else "n/a"
    print(f"\n{row['case']} [{row['backend']}] ({row['n_qubits']} qubits, {total_ops} ops, {row['workload_class']})")
    print(f"  shots:    {shots_str}")
    print(f"  ops/s:    {row['ops_per_sec']}  |  shots/s: {row['shots_per_sec']}  |  cpu util: {row['cpu_util']} ({metric_label})")
    print(f"  memory:   {mem_str}")
    if row.get("aborted"):
        status = f"ABORT — {row.get('abort_reason')}"
    else:
        status = "PASS" if row.get("correct") else f"FAIL — {'; '.join(row.get('errors', []))}"
    print(f"  correct:  {status}")


def _terminate_process(process: mp.Process, *, grace_seconds: float = 1.0) -> None:
    if not process.is_alive():
        process.join(timeout=0)
        return
    process.terminate()
    process.join(timeout=grace_seconds)
    if process.is_alive():
        process.kill()
        process.join(timeout=grace_seconds)


def _build_timeout_row(
    case: BenchmarkCase,
    *,
    backend: str,
    device: torch.device,
    git_hash: str,
    abort_reason: str,
    times: dict[str, float],
    cpu_times: dict[str, float],
) -> dict:
    row = build_aborted_row(
        case,
        backend=backend,
        device=device,
        git_hash=git_hash,
        abort_reason=abort_reason,
        memory_stats={},
    )
    row["times_s"] = {k: round(v, 4) for k, v in times.items()}
    row["cpu_times_s"] = {k: round(v, 4) for k, v in cpu_times.items()}

    completed_shots = [s for s in SHOT_COUNTS if str(s) in row["times_s"]]
    metric_shots = max(completed_shots) if completed_shots else None
    row["metric_shots"] = metric_shots
    if metric_shots is not None:
        total_ops = int(row["ops"]["gates"] + row["ops"]["measurements"] + row["ops"]["conditional"])
        wall_max = float(row["times_s"][str(metric_shots)])
        cpu_max = float(row["cpu_times_s"].get(str(metric_shots), 0.0))
        row["ops_per_sec"] = round(total_ops / wall_max, 1) if wall_max > 0 else 0.0
        row["shots_per_sec"] = round(metric_shots / wall_max, 1) if wall_max > 0 else 0.0
        row["cpu_util"] = round(cpu_max / wall_max, 3) if wall_max > 0 else 0.0
    return row


def _run_case_worker_loop(
    *,
    task_conn: Connection,
    event_conn: Connection,
    device_type: str,
    verbose: bool,
    git_hash: str,
    backend: str,
) -> None:
    device = torch.device(device_type)
    aer_adapter: AerAdapter | None = AerAdapter() if backend == "aer" else None
    cases_by_name: dict[str, BenchmarkCase] = {}
    try:
        while True:
            try:
                message = task_conn.recv()
            except EOFError:
                break
            kind = message.get("kind")
            if kind == "shutdown":
                break
            if kind != "run_case":
                continue

            incoming_case: BenchmarkCase = message["case"]
            case = cases_by_name.get(incoming_case.name)
            if case is None:
                case = incoming_case
                cases_by_name[case.name] = case
            case_timeout = message.get("case_timeout")
            try:
                result = _run_case_local(
                    case,
                    device,
                    verbose,
                    git_hash,
                    backend=backend,
                    case_timeout=case_timeout,
                    aer_adapter=aer_adapter,
                    progress_conn=event_conn,
                )
                event_conn.send({"kind": "result", "result": result})
            except BaseException as error:
                try:
                    event_conn.send({
                        "kind": "error",
                        "error": str(error).strip(),
                        "is_oom": is_oom_error(error),
                    })
                except Exception:
                    pass
    finally:
        try:
            task_conn.close()
        except Exception:
            pass
        try:
            event_conn.close()
        except Exception:
            pass


class _CaseWorker:
    process: mp.Process
    _task_send_conn: Connection
    _event_recv_conn: Connection

    def __init__(
        self,
        *,
        device_type: str,
        verbose: bool,
        git_hash: str,
        backend: str,
    ):
        ctx = mp.get_context("spawn")
        task_recv_conn, task_send_conn = ctx.Pipe(duplex=False)
        event_recv_conn, event_send_conn = ctx.Pipe(duplex=False)
        process = ctx.Process(
            target=_run_case_worker_loop,
            kwargs={
                "task_conn": task_recv_conn,
                "event_conn": event_send_conn,
                "device_type": device_type,
                "verbose": verbose,
                "git_hash": git_hash,
                "backend": backend,
            },
        )
        process.start()

        # Parent keeps task send + event recv ends.
        task_recv_conn.close()
        event_send_conn.close()

        self.process = process
        self._task_send_conn = task_send_conn
        self._event_recv_conn = event_recv_conn

    def is_alive(self) -> bool:
        return self.process.is_alive()

    def send_case(self, case: BenchmarkCase, case_timeout: float | None) -> None:
        self._task_send_conn.send({
            "kind": "run_case",
            "case": case,
            "case_timeout": case_timeout,
        })

    def poll_event(self, timeout_s: float) -> bool:
        return self._event_recv_conn.poll(timeout_s)

    def recv_event(self) -> dict:
        return self._event_recv_conn.recv()

    def drain_events(self) -> list[dict]:
        events: list[dict] = []
        while True:
            try:
                if not self._event_recv_conn.poll():
                    break
                events.append(self._event_recv_conn.recv())
            except EOFError:
                break
        return events

    def close(self) -> None:
        try:
            self._task_send_conn.send({"kind": "shutdown"})
        except Exception:
            pass

        _terminate_process(self.process)

        try:
            self._task_send_conn.close()
        except Exception:
            pass
        try:
            self._event_recv_conn.close()
        except Exception:
            pass


def _close_case_worker() -> None:
    global _CASE_WORKER
    global _CASE_WORKER_CONFIG
    if _CASE_WORKER is None:
        return
    _CASE_WORKER.close()
    _CASE_WORKER = None
    _CASE_WORKER_CONFIG = None


def _ensure_case_worker(
    *,
    device: torch.device,
    verbose: bool,
    git_hash: str,
    backend: str,
) -> _CaseWorker:
    global _CASE_WORKER
    global _CASE_WORKER_CONFIG
    global _CASE_WORKER_ATEXIT_REGISTERED

    config = (device.type, verbose, git_hash, backend)
    if _CASE_WORKER is not None and (_CASE_WORKER_CONFIG != config or not _CASE_WORKER.is_alive()):
        _close_case_worker()
    if _CASE_WORKER is None:
        _CASE_WORKER = _CaseWorker(
            device_type=device.type,
            verbose=verbose,
            git_hash=git_hash,
            backend=backend,
        )
        _CASE_WORKER_CONFIG = config
        if not _CASE_WORKER_ATEXIT_REGISTERED:
            atexit.register(_close_case_worker)
            _CASE_WORKER_ATEXIT_REGISTERED = True
    return _CASE_WORKER


def _run_case_local(
    case: BenchmarkCase,
    device: torch.device,
    verbose: bool,
    git_hash: str,
    *,
    backend: str = "native",
    case_timeout: float | None = None,
    aer_adapter: AerAdapter | None = None,
    progress_conn: Connection | None = None,
) -> dict:
    n_qubits = case.n_qubits
    display_qubits = n_qubits if n_qubits is not None else infer_resources(case.circuit)[0]

    ops = count_ops(case.circuit)
    workload = analyze_workload(case.circuit)
    total_ops = ops["gates"] + ops["measurements"] + ops["conditional"]

    times: dict[str, float] = {}
    cpu_times: dict[str, float] = {}
    result_max: dict[str, int] | None = None
    memory_stats: dict[str, float] = {}
    max_shots = max(SHOT_COUNTS)
    abort_reason: str | None = None
    rss_before = get_process_rss_mb()

    if backend == "native":
        for shot_idx, shots in enumerate(SHOT_COUNTS):
            # Clear compilation cache before the first (cold) call
            if shot_idx == 0:
                _circuit_compilation_cache.pop(id(case.circuit), None)

            try:
                if progress_conn is not None:
                    progress_conn.send({"kind": "shot_start", "shots": shots})
                sync_device(device)
                wall_start = time.perf_counter()
                cpu_start = time.process_time()
                result = run_simulation(case.circuit, shots, n_qubits=n_qubits, device=device)
                sync_device(device)
                wall_elapsed = time.perf_counter() - wall_start
                cpu_elapsed = time.process_time() - cpu_start
                if progress_conn is not None:
                    progress_conn.send({
                        "kind": "shot_done",
                        "shots": shots,
                        "wall_elapsed": wall_elapsed,
                        "cpu_elapsed": cpu_elapsed,
                    })
            except RuntimeError as error:
                if is_oom_error(error):
                    abort_reason = f"OOM at {shots} shots: {str(error).strip()}"
                    clear_device_cache(device)
                else:
                    abort_reason = f"runtime error at {shots} shots: {str(error).strip()}"
                break
            except Exception as error:
                abort_reason = f"error at {shots} shots: {str(error).strip()}"
                break

            times[str(shots)] = round(wall_elapsed, 4)
            cpu_times[str(shots)] = round(cpu_elapsed, 4)

            if case_timeout is not None and wall_elapsed > case_timeout:
                abort_reason = f"timeout at {shots} shots ({wall_elapsed:.1f}s > {case_timeout:.0f}s)"
                break

            if shots == max_shots:
                result_max = result

        memory_stats = get_memory_stats(device)
    else:
        if aer_adapter is None:
            raise RuntimeError("Aer adapter is required when backend='aer'.")

        case_ir = build_circuit_ir(case.circuit, n_qubits=n_qubits)
        supported, support_reason = aer_adapter.supports(case_ir)
        if not supported:
            abort_reason = f"aer unsupported: {support_reason or 'unknown reason'}"
        else:
            try:
                prepared_case = aer_adapter.prepare(case_ir)
            except Exception as error:  # pragma: no cover - depends on environment
                abort_reason = f"aer prepare failed: {str(error).strip()}"
                prepared_case = None

            if abort_reason is None and prepared_case is not None:
                for shots in SHOT_COUNTS:
                    try:
                        if progress_conn is not None:
                            progress_conn.send({"kind": "shot_start", "shots": shots})
                        wall_start = time.perf_counter()
                        cpu_start = time.process_time()
                        result = aer_adapter.run(prepared_case, shots)
                        wall_elapsed = time.perf_counter() - wall_start
                        cpu_elapsed = time.process_time() - cpu_start
                        if progress_conn is not None:
                            progress_conn.send({
                                "kind": "shot_done",
                                "shots": shots,
                                "wall_elapsed": wall_elapsed,
                                "cpu_elapsed": cpu_elapsed,
                            })
                    except Exception as error:  # pragma: no cover - depends on environment
                        abort_reason = f"aer runtime abort at {shots} shots: {str(error).strip()}"
                        break

                    times[str(shots)] = round(wall_elapsed, 4)
                    cpu_times[str(shots)] = round(cpu_elapsed, 4)

                    if case_timeout is not None and wall_elapsed > case_timeout:
                        abort_reason = f"timeout at {shots} shots ({wall_elapsed:.1f}s > {case_timeout:.0f}s)"
                        break

                    if shots == max_shots:
                        result_max = result

    rss_after = get_process_rss_mb()
    add_process_memory_stats(memory_stats, rss_before=rss_before, rss_after=rss_after)

    completed_shots = [s for s in SHOT_COUNTS if str(s) in times]
    metric_shots = max(completed_shots) if completed_shots else None

    if abort_reason is not None:
        correct = False
        errors = [abort_reason]
    elif result_max is not None and metric_shots == max_shots:
        correct, errors = check_correctness(result_max, case.expected, case.tolerance)
    elif metric_shots is not None:
        correct = False
        errors = [f"incomplete shot ladder, highest completed={metric_shots}"]
    else:
        correct = False
        errors = ["no shot count completed"]

    # Derived metrics at highest completed shot count
    if metric_shots is not None:
        wall_max = times[str(metric_shots)]
        cpu_max = cpu_times[str(metric_shots)]
        ops_per_sec = round(total_ops / wall_max, 1) if wall_max > 0 else 0
        shots_per_sec = round(metric_shots / wall_max, 1) if wall_max > 0 else 0
        cpu_util = round(cpu_max / wall_max, 3) if wall_max > 0 else 0
    else:
        ops_per_sec = 0.0
        shots_per_sec = 0.0
        cpu_util = 0.0

    row = {
        "backend": backend,
        "case": case.name,
        "git_hash": git_hash,
        "device": device.type if backend == "native" else "cpu",
        "n_qubits": display_qubits,
        "workload_class": workload["workload_class"],
        "has_conditional": workload["has_conditional"],
        "has_non_terminal_measurement": workload["has_non_terminal_measurement"],
        "ops": ops,
        "times_s": times,
        "cpu_times_s": cpu_times,
        "shot_counts": SHOT_COUNTS,
        "metric_shots": metric_shots,
        "ops_per_sec": ops_per_sec,
        "shots_per_sec": shots_per_sec,
        "cpu_util": cpu_util,
        "memory": {k: round(v, 2) for k, v in memory_stats.items()},
        "correct": correct,
        "aborted": abort_reason is not None,
        "abort_reason": abort_reason,
        "errors": errors,
    }
    if verbose:
        _print_case_row_verbose(row)
    return row


def run_case(
    case: BenchmarkCase,
    device: torch.device,
    verbose: bool,
    git_hash: str,
    *,
    backend: str = "native",
    case_timeout: float | None = None,
    aer_adapter: AerAdapter | None = None,
) -> dict:
    if case_timeout is None:
        return _run_case_local(
            case,
            device,
            verbose,
            git_hash,
            backend=backend,
            case_timeout=None,
            aer_adapter=aer_adapter,
        )

    worker = _ensure_case_worker(
        device=device,
        verbose=verbose,
        git_hash=git_hash,
        backend=backend,
    )
    try:
        worker.send_case(case, case_timeout)
    except (BrokenPipeError, EOFError, OSError):
        _close_case_worker()
        worker = _ensure_case_worker(
            device=device,
            verbose=verbose,
            git_hash=git_hash,
            backend=backend,
        )
        worker.send_case(case, case_timeout)

    in_flight_shot: int | None = None
    shot_start_monotonic: float | None = None
    partial_times: dict[str, float] = {}
    partial_cpu_times: dict[str, float] = {}
    worker_result: dict | None = None
    worker_error: dict | None = None

    def _consume_event(msg: dict) -> bool:
        nonlocal in_flight_shot
        nonlocal shot_start_monotonic
        nonlocal worker_result
        nonlocal worker_error
        kind = msg.get("kind")
        if kind == "shot_start":
            in_flight_shot = int(msg["shots"])
            shot_start_monotonic = time.perf_counter()
            return False
        if kind == "shot_done":
            shot_value = int(msg["shots"])
            partial_times[str(shot_value)] = float(msg["wall_elapsed"])
            partial_cpu_times[str(shot_value)] = float(msg["cpu_elapsed"])
            in_flight_shot = None
            shot_start_monotonic = None
            return False
        if kind == "result":
            worker_result = msg["result"]
            return True
        if kind == "error":
            worker_error = msg
            return True
        return False

    while True:
        try:
            if worker.poll_event(0.05):
                msg = worker.recv_event()
                if _consume_event(msg):
                    break
        except EOFError:
            pass

        if (
            in_flight_shot is not None
            and shot_start_monotonic is not None
            and (time.perf_counter() - shot_start_monotonic) > case_timeout
        ):
            _close_case_worker()
            abort_reason = f"timeout at {in_flight_shot} shots (>{case_timeout:.0f}s hard limit)"
            timed_out_row = _build_timeout_row(
                case,
                backend=backend,
                device=device,
                git_hash=git_hash,
                abort_reason=abort_reason,
                times=partial_times,
                cpu_times=partial_cpu_times,
            )
            if verbose:
                _print_case_row_verbose(timed_out_row)
            return timed_out_row

        if not worker.is_alive():
            for msg in worker.drain_events():
                _consume_event(msg)
            _close_case_worker()
            break

    if worker_result is not None:
        return worker_result

    if worker_error is not None:
        error_text = str(worker_error.get("error", "")).strip()
        is_oom = bool(worker_error.get("is_oom"))
        if is_oom:
            return _build_timeout_row(
                case,
                backend=backend,
                device=device,
                git_hash=git_hash,
                abort_reason=f"outer OOM: {error_text}",
                times=partial_times,
                cpu_times=partial_cpu_times,
            )
        raise RuntimeError(error_text or "worker error")

    raise RuntimeError("worker exited without result")


def build_aborted_row(
    case: BenchmarkCase,
    *,
    backend: str,
    device: torch.device,
    git_hash: str,
    abort_reason: str,
    memory_stats: dict[str, float] | None = None,
) -> dict:
    display_qubits = case.n_qubits if case.n_qubits is not None else infer_resources(case.circuit)[0]
    workload = analyze_workload(case.circuit)
    ops = count_ops(case.circuit)
    memory_payload = {k: round(v, 2) for k, v in (memory_stats or {}).items()}
    return {
        "backend": backend,
        "case": case.name,
        "git_hash": git_hash,
        "device": device.type if backend == "native" else "cpu",
        "n_qubits": display_qubits,
        "workload_class": workload["workload_class"],
        "has_conditional": workload["has_conditional"],
        "has_non_terminal_measurement": workload["has_non_terminal_measurement"],
        "ops": ops,
        "times_s": {},
        "cpu_times_s": {},
        "shot_counts": SHOT_COUNTS,
        "metric_shots": None,
        "ops_per_sec": 0.0,
        "shots_per_sec": 0.0,
        "cpu_util": 0.0,
        "memory": memory_payload,
        "correct": False,
        "aborted": True,
        "abort_reason": abort_reason,
        "errors": [abort_reason],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantum simulator benchmark")
    parser.add_argument("-v", "--verbose", action="store_true", help="Per-case timing and correctness details")
    parser.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="Run only the selected benchmark case names.",
    )
    parser.add_argument(
        "--core",
        action="store_true",
        help="Run only the core-6 cases (bell_state, simple_grovers, real_grovers, ghz_state, qft, teleportation).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Max wall-clock seconds per case per shot count. Cases exceeding this are aborted (default: 30).",
    )
    parser.add_argument(
        "--backend",
        choices=["native", "aer"],
        default="native",
        help="Execution backend (default: native).",
    )
    args = parser.parse_args()

    if args.core and args.cases:
        parser.error("Cannot use both --core and --cases.")

    if args.core:
        cases_to_run = CORE_CASES
    elif args.cases is not None:
        all_case_map = {case_fn().name: case_fn for case_fn in ALL_CASES}
        if len(all_case_map) != len(ALL_CASES):
            raise RuntimeError("Duplicate benchmark case names detected.")
        unknown_cases = [name for name in args.cases if name not in all_case_map]
        if unknown_cases:
            parser.error(f"Unknown case(s): {', '.join(unknown_cases)}")
        cases_to_run = [all_case_map[name] for name in args.cases]
    else:
        cases_to_run = ALL_CASES

    device = get_device()
    git_hash = get_git_hash()
    aer_adapter: AerAdapter | None = None
    if args.backend == "aer":
        aer_adapter = AerAdapter()
        availability = aer_adapter.availability()
        if not availability.available:
            parser.error(f"Aer backend unavailable: {availability.reason}")
        print(f"Backend: aer | Host device: {device.type} | Git: {git_hash} | Schedule: cold@{SHOT_COUNTS[0]} warm@{SHOT_COUNTS[-1]}")
    else:
        print(f"Backend: native | Device: {device.type} | Git: {git_hash} | Schedule: cold@{SHOT_COUNTS[0]} warm@{SHOT_COUNTS[-1]}")

    # Sort cases by qubit count ascending so small/fast cases run and save first
    instantiated = [case_fn() for case_fn in cases_to_run]
    instantiated.sort(key=lambda c: c.n_qubits if c.n_qubits is not None else infer_resources(c.circuit)[0])

    # Incremental JSONL output — each result flushed immediately for OOM resilience
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    output_path = results_dir / f"{timestamp}.jsonl"

    results: list[dict] = []
    failures: list[str] = []

    with open(output_path, "w") as jsonl_f:
        for case in instantiated:
            outer_non_oom_error = False
            try:
                result = run_case(
                    case,
                    device,
                    args.verbose,
                    git_hash,
                    backend=args.backend,
                    case_timeout=args.timeout,
                    aer_adapter=aer_adapter,
                )
            except Exception as e:
                print(f"\nERROR in {case.name}: {e}")
                error_text = str(e).strip()
                is_oom = is_oom_error(e)
                abort_reason = f"outer {'OOM' if is_oom else 'error'}: {error_text}"
                outer_non_oom_error = not is_oom
                rss_now = get_process_rss_mb()
                fallback_memory: dict[str, float] = get_memory_stats(device) if args.backend == "native" else {}
                add_process_memory_stats(fallback_memory, rss_before=rss_now, rss_after=rss_now)
                result = build_aborted_row(
                    case,
                    backend=args.backend,
                    device=device,
                    git_hash=git_hash,
                    abort_reason=abort_reason,
                    memory_stats=fallback_memory,
                )
            results.append(result)
            jsonl_f.write(json.dumps(result) + "\n")
            jsonl_f.flush()
            if outer_non_oom_error:
                failures.append(result["case"])
            elif not result["correct"] and not result.get("aborted"):
                failures.append(result["case"])

    print(f"Wrote {output_path}")

    # Summary — only include cases that completed all shot counts
    if results:
        complete_results = [r for r in results if not r.get("aborted") and all(str(s) in r["times_s"] for s in SHOT_COUNTS)]
        aborted_count = sum(1 for r in results if r.get("aborted"))

        static_rows = [row for row in complete_results if row["workload_class"] == "static"]
        dynamic_rows = [row for row in complete_results if row["workload_class"] == "dynamic"]

        totals_all = _totals_for_rows(complete_results)
        totals_static = _totals_for_rows(static_rows)
        totals_dynamic = _totals_for_rows(dynamic_rows)

        if aborted_count > 0:
            print(f"\nAborted: {aborted_count} case(s) (excluded from totals)")
        print(f"Complete: {len(complete_results)}/{len(results)} cases")

        print(f"\nTotal time by shot count (all):\n    {_format_totals_line(totals_all)}")
        print(f"Total time by shot count (static):\n    {_format_totals_line(totals_static)}")
        print(f"Total time by shot count (dynamic):\n    {_format_totals_line(totals_dynamic)}")

        cold_key = str(SHOT_COUNTS[0])
        warm_key = str(SHOT_COUNTS[-1])
        _print_hotspot_block(static_rows, label="static cold", shot_key=cold_key, top_n=5)
        _print_hotspot_block(dynamic_rows, label="dynamic cold", shot_key=cold_key, top_n=5)
        _print_hotspot_block(static_rows, label="static warm", shot_key=warm_key, top_n=5)
        _print_hotspot_block(dynamic_rows, label="dynamic warm", shot_key=warm_key, top_n=5)

        # Process-level stats
        peak_rss_mb = get_peak_rss_mb()
        print(f"Peak RSS: {peak_rss_mb:.0f} MB")

    _close_case_worker()

    if failures:
        print(f"\nFAILED: {', '.join(failures)}")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
