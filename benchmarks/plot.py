"""Plot benchmark results from a JSONL file."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def plot(results: list[dict], output_path: Path) -> None:
    shot_counts = [1, 10, 100, 1000]
    cases = [r["case"] for r in results]

    fig, (ax_bars, ax_total) = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: grouped bar chart per case ──
    x = np.arange(len(cases))
    width = 0.18
    offsets = np.arange(len(shot_counts)) - (len(shot_counts) - 1) / 2

    for i, shots in enumerate(shot_counts):
        times = [r["times_s"][str(shots)] for r in results]
        bars = ax_bars.bar(x + offsets[i] * width, times, width, label=f"{shots} shots")

        # Mark failed cases
        for j, r in enumerate(results):
            if not r["correct"]:
                bars[j].set_edgecolor("red")
                bars[j].set_linewidth(2)

    ax_bars.set_xlabel("Benchmark case")
    ax_bars.set_ylabel("Time (s)")
    ax_bars.set_title("Time per case")
    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels([f"{r['case']}\n({r['n_qubits']}q)" for r in results])
    ax_bars.legend()
    ax_bars.set_yscale("log")

    # Add PASS/FAIL labels
    for j, r in enumerate(results):
        label = "FAIL" if not r["correct"] else "PASS"
        color = "red" if not r["correct"] else "green"
        ax_bars.text(x[j], ax_bars.get_ylim()[0] * 1.5, label,
                     ha="center", va="bottom", fontsize=9, fontweight="bold", color=color)

    # ── Right: total time by shot count ──
    totals = [sum(r["times_s"][str(s)] for r in results) for s in shot_counts]
    bars = ax_total.bar([str(s) for s in shot_counts], totals, color="steelblue")
    ax_total.set_xlabel("Shot count")
    ax_total.set_ylabel("Total time (s)")
    ax_total.set_title("Total time by shot count")

    for bar, t in zip(bars, totals):
        ax_total.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                      f"{t:.1f}s", ha="center", va="bottom", fontsize=10)

    fig.suptitle(f"Benchmark results \u2014 {output_path.stem}", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument("file", nargs="?", help="Path to a .jsonl results file (default: latest)")
    args = parser.parse_args()

    if args.file:
        path = Path(args.file)
    else:
        results_dir = Path(__file__).parent / "results"
        files = sorted(results_dir.glob("*.jsonl"))
        if not files:
            print("No results found in benchmarks/results/")
            sys.exit(1)
        path = files[-1]

    results = load_results(path)
    png_path = path.with_suffix(".png")
    plot(results, png_path)


if __name__ == "__main__":
    main()
