"""Plot benchmark performance over time from all JSONL result files."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


SHOT_COUNTS = [1, 10, 100, 1000]

# Consistent colors per case across all plots
CASE_COLORS = {
    "bell_state": "#1f77b4",
    "simple_grovers": "#ff7f0e",
    "real_grovers": "#d62728",
    "ghz_state": "#2ca02c",
    "qft": "#9467bd",
    "teleportation": "#8c564b",
}


def load_all_runs(results_dir: Path) -> list[tuple[str, list[dict]]]:
    """Load all JSONL files, sorted by timestamp. Returns [(label, results), ...]."""
    runs: list[tuple[str, list[dict]]] = []
    for path in sorted(results_dir.glob("*.jsonl")):
        with open(path) as f:
            results = [json.loads(line) for line in f if line.strip()]
        if not results:
            continue
        # Label: git hash if available, otherwise timestamp from filename
        git_hash = results[0].get("git_hash", "")
        timestamp = path.stem  # e.g. "2026-02-10T171139"
        short_ts = timestamp[5:16]  # "02-10T17:11" — drop year, keep date+time
        label = f"{git_hash} ({short_ts})" if git_hash else short_ts
        runs.append((label, results))
    return runs


def plot_over_time(runs: list[tuple[str, list[dict]]], output_path: Path) -> None:
    """Plot performance trends across all runs."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for ax_idx, shots in enumerate(SHOT_COUNTS):
        ax = axes[ax_idx // 2][ax_idx % 2]
        shot_key = str(shots)

        # Collect all case names across all runs
        all_cases: list[str] = []
        for _, results in runs:
            for r in results:
                if r["case"] not in all_cases:
                    all_cases.append(r["case"])

        # Plot each case as a line over time
        for case_name in all_cases:
            x_positions: list[int] = []
            times: list[float] = []
            markers: list[str] = []

            for run_idx, (_, results) in enumerate(runs):
                for r in results:
                    if r["case"] == case_name and shot_key in r["times_s"]:
                        x_positions.append(run_idx)
                        times.append(r["times_s"][shot_key])
                        markers.append("x" if not r["correct"] else "o")

            if not times:
                continue

            color = CASE_COLORS.get(case_name, "gray")
            ax.plot(x_positions, times, "-o", color=color, label=case_name,
                    markersize=5, linewidth=1.5)

            # Mark failures with red X
            for x, t, m in zip(x_positions, times, markers):
                if m == "x":
                    ax.plot(x, t, "x", color="red", markersize=10, markeredgewidth=2)

        ax.set_yscale("log")
        ax.set_title(f"{shots} shots")
        ax.set_ylabel("Time (s)")
        ax.set_xticks(range(len(runs)))
        ax.set_xticklabels([label for label, _ in runs], rotation=45, ha="right", fontsize=7)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.3g}"))
        ax.grid(True, alpha=0.3)

    # Single legend for all subplots
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(all_cases), fontsize=9,
               bbox_to_anchor=(0.5, 1.0))

    fig.suptitle("Benchmark Performance Over Time", fontsize=14, y=1.03)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark performance over time")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output PNG path (default: benchmarks/results/performance.png)")
    args = parser.parse_args()

    results_dir = Path(__file__).parent / "results"
    runs = load_all_runs(results_dir)

    if not runs:
        print("No results found in benchmarks/results/")
        sys.exit(1)

    output_path = Path(args.output) if args.output else results_dir / "performance.png"
    plot_over_time(runs, output_path)


if __name__ == "__main__":
    main()
