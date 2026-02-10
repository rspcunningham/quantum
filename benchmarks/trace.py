"""Targeted profiling for a single benchmark case."""

import argparse
import sys
from pathlib import Path

import torch

from benchmarks.cases import ALL_CASES
from quantum import run_simulation, infer_resources


CASE_MAP = {case_fn().name: case_fn for case_fn in ALL_CASES}


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


def main() -> None:
    case_names = list(CASE_MAP.keys())

    parser = argparse.ArgumentParser(
        description="Profile a single benchmark case with torch.profiler",
        epilog=f"Available cases: {', '.join(case_names)}",
    )
    parser.add_argument("case", choices=case_names, help="Benchmark case to profile")
    parser.add_argument("shots", type=int, help="Number of shots")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output trace path (default: benchmarks/results/trace_{case}_{shots}.json)")
    parser.add_argument("--no-stack", action="store_true", help="Disable stack traces (smaller output files)")
    args = parser.parse_args()

    device = get_device()
    case = CASE_MAP[args.case]()
    n_qubits = case.n_qubits

    if n_qubits is None:
        n_qubits = infer_resources(case.circuit)[0]

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    trace_path = Path(args.output) if args.output else output_dir / f"trace_{args.case}_{args.shots}.json"

    activities = [
        torch.profiler.ProfilerActivity.CPU,
        *(
            [torch.profiler.ProfilerActivity.CUDA]
            if device.type == "cuda" else []
        ),
    ]

    print(f"Profiling {args.case} ({n_qubits} qubits, {args.shots} shots) on {device.type}")

    # Warmup
    run_simulation(case.circuit, 1, n_qubits=n_qubits, device=device)
    sync_device(device)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=not args.no_stack,
    ) as prof:
        run_simulation(case.circuit, args.shots, n_qubits=n_qubits, device=device)
        sync_device(device)

    prof.export_chrome_trace(str(trace_path))
    print(f"\nTrace: {trace_path} ({trace_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"\nOpen in Chrome: chrome://tracing")
    print(f"Open in Perfetto: https://ui.perfetto.dev\n")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))


if __name__ == "__main__":
    main()
