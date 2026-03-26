"""Benchmark runner for the quantum simulator."""

import json
import multiprocessing
import os
import pickle
import resource
import struct
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from benchmarks.cases import BenchmarkCase, ALL_CASES
from quantum import run_simulation, infer_resources
from quantum.gates import Gate, Measurement, ConditionalGate, Circuit

SHOTS = 10_000
TIMEOUT = 10


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
    counts = {"gates": 0, "measurements": 0, "conditional": 0}

    def _walk(ops) -> None:
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


def classify_workload(circuit: Circuit) -> str:
    flattened = []

    def _flatten(ops) -> None:
        for op in ops:
            if isinstance(op, Circuit):
                _flatten(op.operations)
            else:
                flattened.append(op)

    _flatten(circuit.operations)

    seen_measurement = False
    for op in flattened:
        if isinstance(op, ConditionalGate):
            return "dynamic"
        if isinstance(op, Measurement):
            seen_measurement = True
        elif seen_measurement:
            return "dynamic"
    return "static"


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


def get_peak_rss_mb() -> float:
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    return rusage.ru_maxrss / (1024 * 1024) if sys.platform == "darwin" else rusage.ru_maxrss / 1024


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def estimate_memory_gb(n_qubits: int) -> float:
    """Estimate GPU memory needed: 4 state buffers + 2 sampling prob buffers + histogram + shots."""
    dim = 2 ** n_qubits
    state_bytes = dim * 4 * 4           # 4 state buffers (ping-pong re/im), float32
    sampling_bytes = dim * 4 * 2        # 2 probability buffers (a/b), float32
    shots_bytes = SHOTS * 4             # sampled_codes, uint32
    hist_size = _next_pow2(max(1, SHOTS * 2))
    hist_bytes = hist_size * 4 * 2      # keys + counts, uint32 each
    return (state_bytes + sampling_bytes + shots_bytes + hist_bytes) / (1024 ** 3)


def get_memory_limit_gb() -> float:
    """Use half of physical RAM as the limit."""
    total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return total / (1024 ** 3) / 2


# ---------------------------------------------------------------------------
# Long-lived worker subprocess
#
# Spawned once. Receives (circuit, n_qubits, shots, timeout) over stdin,
# calls run_simulation with the native C++ timeout, sends back the result.
# On timeout the C++ layer throws immediately; the worker exits so the
# parent can spawn a fresh one with a clean Metal context.
# ---------------------------------------------------------------------------

_WORKER_SCRIPT = """\
import pickle, struct, sys, time, os

# Import only the simulator — not benchmarks.
from quantum import run_simulation

stdin = sys.stdin.buffer
stdout = sys.stdout.buffer

# Signal ready.
stdout.write(b"R")
stdout.flush()

while True:
    hdr = stdin.read(4)
    if len(hdr) < 4:
        break
    length = struct.unpack(">I", hdr)[0]
    payload = stdin.read(length)
    if len(payload) < length:
        break
    circuit, n_qubits, shots, timeout = pickle.loads(payload)
    error = None
    wall = 0.0
    cpu = 0.0
    result = None
    gpu_timeout = False
    try:
        t0 = time.perf_counter()
        c0 = time.process_time()
        result = run_simulation(circuit, shots, n_qubits=n_qubits, timeout=timeout)
        wall = time.perf_counter() - t0
        cpu = time.process_time() - c0
    except Exception as e:
        wall = time.perf_counter() - t0
        cpu = time.process_time() - c0
        error = str(e).strip()
        if "timeout" in error.lower():
            gpu_timeout = True
    resp = pickle.dumps({"result": result, "wall": wall, "cpu": cpu,
                         "error": error, "gpu_timeout": gpu_timeout})
    stdout.write(struct.pack(">I", len(resp)))
    stdout.write(resp)
    stdout.flush()
    if gpu_timeout:
        # GPU work is still in flight — exit cleanly so the OS does
        # orderly Metal teardown before the parent spawns a fresh worker.
        break
"""


def _send(pipe, obj):
    data = pickle.dumps(obj)
    os.write(pipe, struct.pack(">I", len(data)) + data)


def _recv(pipe, deadline):
    """Read a length-prefixed pickle from a raw fd, respecting deadline."""
    import select

    def _read_exact(n):
        buf = bytearray()
        while len(buf) < n:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            ready, _, _ = select.select([pipe], [], [], remaining)
            if not ready:
                return None
            chunk = os.read(pipe, n - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    hdr = _read_exact(4)
    if hdr is None:
        return None
    length = struct.unpack(">I", hdr)[0]
    payload = _read_exact(length)
    if payload is None:
        return None
    return pickle.loads(payload)


class _Worker:
    def __init__(self):
        self._proc = None
        self._stdin_fd = None
        self._stdout_fd = None

    def _spawn(self):
        proc = subprocess.Popen(
            [sys.executable, "-u", "-c", _WORKER_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Wait for ready signal.
        ready = proc.stdout.read(1)
        if ready != b"R":
            proc.kill()
            proc.wait()
            raise RuntimeError("worker failed to start")
        self._proc = proc
        self._stdin_fd = proc.stdin.fileno()
        self._stdout_fd = proc.stdout.fileno()

    def _ensure(self):
        if self._proc is None or self._proc.poll() is not None:
            self._spawn()

    def _kill(self):
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()
            self._proc.wait()
        self._proc = None
        self._stdin_fd = None
        self._stdout_fd = None

    def run(self, circuit, n_qubits, shots, timeout):
        self._ensure()
        try:
            _send(self._stdin_fd, (circuit, n_qubits, shots, timeout))
        except OSError:
            self._proc = None
            raise RuntimeError("worker died before receiving task")

        # Give the worker timeout + generous margin for pickle overhead.
        deadline = time.monotonic() + timeout + 5.0
        resp = _recv(self._stdout_fd, deadline)

        if resp is None:
            # Worker died or hung beyond the margin — force kill.
            self._kill()
            return {"result": None, "wall": timeout, "cpu": 0.0,
                    "error": f"timeout: killed after {timeout}s", "gpu_timeout": True}

        if resp.get("gpu_timeout"):
            # Worker is exiting on its own — wait for clean shutdown.
            self._proc.wait(timeout=5)
            self._proc = None
            self._stdin_fd = None
            self._stdout_fd = None

        return resp

    def close(self):
        if self._proc is not None and self._proc.poll() is None:
            self._proc.stdin.close()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        self._proc = None
        self._stdin_fd = None
        self._stdout_fd = None


def run_case(case: BenchmarkCase, git_hash: str, *, memory_limit_gb: float, worker: _Worker) -> dict:
    n_qubits = case.n_qubits
    if n_qubits is None:
        n_qubits = infer_resources(case.circuit)[0]
    ops = count_ops(case.circuit)
    workload = classify_workload(case.circuit)
    total_ops = ops["gates"] + ops["measurements"] + ops["conditional"]

    abort_reason = None
    result = None
    wall_elapsed = 0.0
    cpu_elapsed = 0.0

    estimated_gb = estimate_memory_gb(n_qubits)
    if estimated_gb > memory_limit_gb:
        abort_reason = f"skipped: {estimated_gb:.1f} GB estimated > {memory_limit_gb:.1f} GB limit ({n_qubits} qubits)"
    else:
        resp = worker.run(case.circuit, n_qubits, SHOTS, TIMEOUT)
        wall_elapsed = resp["wall"]
        cpu_elapsed = resp["cpu"]
        if resp["error"] is not None:
            abort_reason = resp["error"]
        else:
            result = resp["result"]

    if abort_reason is not None:
        correct = False
        errors = [abort_reason]
    elif result is not None:
        correct, errors = check_correctness(result, case.expected, case.tolerance)
    else:
        correct = False
        errors = ["no result"]

    ops_per_sec = round(total_ops / wall_elapsed, 1) if wall_elapsed > 0 else 0.0
    shots_per_sec = round(SHOTS / wall_elapsed, 1) if wall_elapsed > 0 else 0.0

    return {
        "case": case.name,
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


def print_results(results: list[dict]) -> None:
    static = [r for r in results if r["workload"] == "static" and not r["aborted"]]
    dynamic = [r for r in results if r["workload"] == "dynamic" and not r["aborted"]]
    aborted = [r for r in results if r["aborted"]]

    def _print_table(rows: list[dict], label: str) -> None:
        if not rows:
            return
        total_time = sum(r["time_s"] for r in rows)
        print(f"\n{label} ({len(rows)} cases, {total_time:.3f}s total):")
        print(f"  {'case':<30} {'qubits':>6} {'ops':>6} {'time_s':>8} {'shots/s':>10} {'correct':>8}")
        print(f"  {'-'*30} {'-'*6} {'-'*6} {'-'*8} {'-'*10} {'-'*8}")
        for r in sorted(rows, key=lambda x: x["time_s"], reverse=True):
            total_ops = r["ops"]["gates"] + r["ops"]["measurements"] + r["ops"]["conditional"]
            status = "PASS" if r["correct"] else "FAIL"
            print(
                f"  {r['case']:<30} {r['n_qubits']:>6} {total_ops:>6} "
                f"{r['time_s']:>8.4f} {r['shots_per_sec']:>10.0f} {status:>8}"
            )

    _print_table(static, "Static circuits")
    _print_table(dynamic, "Dynamic circuits")

    if aborted:
        print(f"\nAborted ({len(aborted)}):")
        for r in aborted:
            print(f"  {r['case']}: {r['errors'][0]}")

    failed = [r for r in results if not r["correct"] and not r["aborted"]]
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for r in failed:
            print(f"  {r['case']}: {'; '.join(r['errors'])}")

    complete = [r for r in results if not r["aborted"]]
    if complete:
        total = sum(r["time_s"] for r in complete)
        print(f"\nTotal: {len(complete)}/{len(results)} cases in {total:.3f}s | Peak RSS: {get_peak_rss_mb():.0f} MB")


def main() -> None:
    memory_limit_gb = get_memory_limit_gb()
    git_hash = get_git_hash()
    print(f"Git: {git_hash} | Shots: {SHOTS} | Memory limit: {memory_limit_gb:.0f} GB")

    instantiated = [case_fn() for case_fn in ALL_CASES]
    instantiated.sort(key=lambda c: c.n_qubits if c.n_qubits is not None else infer_resources(c.circuit)[0])

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    output_path = results_dir / f"{timestamp}.jsonl"

    results: list[dict] = []
    worker = _Worker()

    try:
        with open(output_path, "w") as jsonl_f:
            for case in instantiated:
                try:
                    row = run_case(case, git_hash, memory_limit_gb=memory_limit_gb, worker=worker)
                except Exception as e:
                    print(f"\nERROR in {case.name}: {e}")
                    row = {
                        "case": case.name, "git_hash": git_hash,
                        "n_qubits": case.n_qubits or 0, "workload": "unknown",
                        "ops": count_ops(case.circuit), "shots": SHOTS,
                        "time_s": 0, "cpu_s": 0, "ops_per_sec": 0, "shots_per_sec": 0,
                        "correct": False, "errors": [str(e)], "aborted": True,
                    }
                results.append(row)
                jsonl_f.write(json.dumps(row) + "\n")
                jsonl_f.flush()
                status = "PASS" if row["correct"] else ("ABORT" if row["aborted"] else "FAIL")
                print(f"  {row['case']}: {row['time_s']:.4f}s [{status}]")
    finally:
        worker.close()

    print(f"Wrote {output_path}")
    print_results(results)

    if any(not r["correct"] and not r["aborted"] for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
