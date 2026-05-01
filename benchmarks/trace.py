"""Targeted profiling for a single benchmark case."""

import argparse
import cProfile
import gc
import pstats
import sys
from pathlib import Path

from benchmarks.cases import ALL_CASES
from quantum import compile as compile_circuit, infer_resources


CASE_MAP = {case_fn().name: case_fn for case_fn in ALL_CASES}


def clear_runtime_caches() -> None:
    """Clear simulator caches before a cold profile."""
    gc.collect()


def main() -> None:
    case_names = list(CASE_MAP.keys())

    parser = argparse.ArgumentParser(
        description="Profile a single benchmark case with cProfile",
        epilog=f"Available cases: {', '.join(case_names)}",
    )
    parser.add_argument("case", choices=case_names, help="Benchmark case to profile")
    parser.add_argument("shots", type=int, help="Number of shots")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output profile path (default: benchmarks/results/trace_{case}_{shots}.prof)")
    parser.add_argument(
        "--cold",
        action="store_true",
        help="Profile true first-call behavior: clear caches and skip warmup",
    )
    args = parser.parse_args()

    case = CASE_MAP[args.case]()
    n_qubits = case.n_qubits

    if n_qubits is None:
        n_qubits = infer_resources(case.circuit)[0]

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    prof_path = Path(args.output) if args.output else output_dir / f"trace_{args.case}_{args.shots}.prof"

    profile_mode = "cold" if args.cold else "warm"
    print(
        f"Profiling {args.case} ({n_qubits} qubits, {args.shots} shots) [{profile_mode}]"
    )

    if args.cold:
        clear_runtime_caches()
        profiler = cProfile.Profile()
        profiler.enable()
        with compile_circuit(case.circuit, n_qubits=n_qubits) as compiled:
            compiled.run(args.shots)
        profiler.disable()
    else:
        with compile_circuit(case.circuit, n_qubits=n_qubits) as compiled:
            compiled.run(1)
            profiler = cProfile.Profile()
            profiler.enable()
            compiled.run(args.shots)
            profiler.disable()

    profiler.dump_stats(str(prof_path))
    print(f"\nProfile: {prof_path} ({prof_path.stat().st_size / 1024:.1f} KB)")
    print(f"View with: python -m pstats {prof_path}\n")

    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(20)


if __name__ == "__main__":
    main()
