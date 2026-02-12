# Optimization Progress Raw Data (Full Suite)

This file contains all points used to generate:

- `docs/progress.png`

Data source files are in `benchmarks/results/` and are listed below in plotting order.

## Cases Included

All benchmark cases (24 hand-coded + 132 QASM-loaded). Cases that OOM are excluded from totals — only cases that complete all 5 shot counts are summed.

## Plot Metadata

- Y-axis: log scale
- Series: totals for shot counts `1`, `10`, `100`, `1000`, `10000`
- `label_x`: exact x-axis text (`git_hash` + compact timestamp)

| idx | checkpoint_jsonl | git_hash | label_x | total_1_s | total_10_s | total_100_s | total_1000_s | total_10000_s | cases_complete |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 0 | `2026-02-11T172533.jsonl` | `654bc2c` | `654bc2c\n02-11T1725` | 71.1667 | 68.3367 | 68.7975 | 67.8941 | 68.4366 | 146 |
| 1 | `2026-02-12T070101.jsonl` | `6331041` | `6331041\n02-12T0701` | 7.2060 | 4.0211 | 4.3185 | 4.5478 | 4.5890 | 156 |
| 2 | `2026-02-12T072716.jsonl` | `6827078` | `6827078\n02-12T0727` | 4.7374 | 1.6530 | 1.8440 | 2.0580 | 2.1434 | 156 |
| 3 | `2026-02-12T073646.jsonl` | `3f5cb37` | `3f5cb37\n02-12T0736` | 3.9303 | 0.8914 | 0.9436 | 0.9530 | 1.0722 | 156 |
| 4 | `2026-02-12T074313.jsonl` | `1f50330` | `1f50330\n02-12T0743` | 0.7353 | 0.8326 | 0.9136 | 0.9810 | 1.1260 | 156 |

## SOTA Reference — Qiskit Aer (Full Suite)

Horizontal reference lines for the progress chart. These are Aer's full-suite totals per shot count — a fixed target measured once. **Do not re-run** the comparison each iteration; just read the values below.

| shot_count | aer_total_s |
|---:|---:|
| 1 | null |
| 10 | null |
| 100 | null |
| 1000 | null |
| 10000 | null |

## Notes

- `null` means the checkpoint did not record that shot count for at least one case, or the SOTA comparison has not been run yet.
- Cases that OOM at lower shot counts are excluded entirely from totals.
- The `cases_complete` column tracks how many cases completed all 5 shot counts.
- SOTA reference values of `null` mean the comparison has not been run yet. Once populated, they are plotted as horizontal dashed lines on `docs/progress.png`.
