"""Regenerate the optimization progress chart from docs/progress-data.md."""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

DOCS = Path(__file__).resolve().parent.parent / "docs"
DATA_PATH = DOCS / "progress-data.md"
OUTPUT_PATH = DOCS / "images" / "progress.png"

SHOT_COLS = ["total_1_s", "total_10_s", "total_100_s", "total_1000_s", "total_10000_s"]
SHOT_LABELS = ["@1", "@10", "@100", "@1000", "@10000"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def parse_table(text: str) -> list[dict[str, str | float | None]]:
    rows: list[dict[str, str | float | None]] = []
    in_table = False
    headers: list[str] = []

    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            if in_table:
                break
            continue

        cells = [c.strip() for c in line.split("|")[1:-1]]

        if not in_table:
            headers = cells
            in_table = True
            continue

        if all(re.fullmatch(r"-+:?", c) or re.fullmatch(r":?-+:?", c) for c in cells):
            continue

        row: dict[str, str | float | None] = {}
        for header, cell in zip(headers, cells):
            if cell.lower() == "null" or cell == "":
                row[header] = None
            else:
                try:
                    row[header] = float(cell)
                except ValueError:
                    row[header] = cell

        rows.append(row)

    return rows


def main() -> None:
    text = DATA_PATH.read_text()
    rows = parse_table(text)

    if not rows:
        raise SystemExit(f"No data rows found in {DATA_PATH}")

    fig, ax = plt.subplots(figsize=(16, 7))

    xs = list(range(len(rows)))
    labels = [str(r.get("label_x", r.get("idx", i))).replace("\\n", "\n") for i, r in enumerate(rows)]

    for col, shot_label, color in zip(SHOT_COLS, SHOT_LABELS, COLORS):
        x_vals, y_vals = [], []
        for i, r in enumerate(rows):
            val = r.get(col)
            if val is not None and isinstance(val, (int, float)):
                x_vals.append(i)
                y_vals.append(val)
        ax.plot(x_vals, y_vals, "-o", color=color, label=shot_label, markersize=4, linewidth=1.5)

    for i, r in enumerate(rows):
        annotation = r.get("annotation")
        if annotation and isinstance(annotation, str) and annotation.strip():
            y_val = r.get("total_1000_s") or r.get("total_1_s")
            if y_val is not None:
                ax.annotate(
                    annotation,
                    xy=(i, y_val),
                    xytext=(0, 12),
                    textcoords="offset points",
                    fontsize=7,
                    ha="center",
                    fontweight="bold",
                    color="#333333",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="wheat", alpha=0.7, edgecolor="none"),
                )

    ax.set_yscale("log")
    ax.set_ylabel("Core-6 Total Time (s)", fontsize=11)
    ax.set_xlabel("Checkpoint", fontsize=11)
    ax.set_title("Optimization Progress — Core-6 Suite", fontsize=13, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6.5)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.3g}"))
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
