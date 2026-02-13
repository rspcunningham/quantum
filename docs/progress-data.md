# Optimization Progress Raw Data (Full Suite)

This file contains all points used to generate:

- `docs/progress.png`

Data source files are in `benchmarks/results/` and are listed below in plotting order.

## Cases Included

All benchmark cases (hand-coded + QASM-loaded). Cases that OOM are excluded from totals — only cases that complete both shot counts are summed.

## Era 1 — 5-shot ladder (idx 0-11)

Benchmark schedule: `[1, 10, 100, 1000, 10000]` with warmup call before shot ladder. All shots measured warm (cached).

| idx | checkpoint_jsonl | git_hash | label_x | total_1_s | total_10_s | total_100_s | total_1000_s | total_10000_s | cases_complete |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 0 | `2026-02-11T172533.jsonl` | `654bc2c` | `654bc2c\n02-11T1725` | 71.1667 | 68.3367 | 68.7975 | 67.8941 | 68.4366 | 146 |
| 1 | `2026-02-12T070101.jsonl` | `6331041` | `6331041\n02-12T0701` | 7.2060 | 4.0211 | 4.3185 | 4.5478 | 4.5890 | 156 |
| 2 | `2026-02-12T072716.jsonl` | `6827078` | `6827078\n02-12T0727` | 4.7374 | 1.6530 | 1.8440 | 2.0580 | 2.1434 | 156 |
| 3 | `2026-02-12T073646.jsonl` | `3f5cb37` | `3f5cb37\n02-12T0736` | 3.9303 | 0.8914 | 0.9436 | 0.9530 | 1.0722 | 156 |
| 4 | `2026-02-12T074313.jsonl` | `1f50330` | `1f50330\n02-12T0743` | 0.7353 | 0.8326 | 0.9136 | 0.9810 | 1.1260 | 156 |
| 5 | `2026-02-12T075645.jsonl` | `98fb061` | `98fb061\n02-12T0756` | 0.1100 | 0.2200 | 0.3000 | 0.3400 | 0.4000 | 156 |
| 6 | `2026-02-12T080848.jsonl` | `a13e93e` | `a13e93e\n02-12T0808` | 0.1000 | 0.2000 | 0.2600 | 0.3100 | 0.3400 | 156 |
| 7 | `2026-02-12T083949.jsonl` | `c921e03` | `c921e03\n02-12T0839` | 0.0300 | 0.0300 | 0.0300 | 0.0500 | 0.0800 | 156 |
| 8 | `2026-02-12T085627.jsonl` | `7c8bacb` | `7c8bacb\n02-12T0856` | 0.0287 | 0.0248 | 0.0270 | 0.0483 | 0.0760 | 156 |
| 9 | `2026-02-12T090356.jsonl` | `8464055` | `8464055\n02-12T0903` | 0.0198 | 0.0175 | 0.0197 | 0.0431 | 0.0688 | 156 |
| 10 | `2026-02-12T091156.jsonl` | `74183cf` | `74183cf\n02-12T0911` | 0.0187 | 0.0152 | 0.0194 | 0.0389 | 0.0629 | 156 |
| 11 | `2026-02-12T091718.jsonl` | `751b889` | `751b889\n02-12T0917` | 0.0153 | 0.0123 | 0.0169 | 0.0377 | 0.0639 | 156 |

## Era 2 — Cold/warm schedule (idx 12+)

Benchmark schedule: cold @1K (cache cleared) + warm @10K per circuit. Two calls total.

| idx | checkpoint_jsonl | git_hash | label_x | cold_1000_s | warm_10000_s | cases_complete |
|---:|---|---|---|---:|---:|---:|
| 12 | `2026-02-12T154156.jsonl` | `4ed771f` | `4ed771f\n02-12T1541` | 68.7304 | 0.1753 | 216 |
| 13 | `2026-02-12T185245.jsonl` | `5f6ec69` | `5f6ec69\n02-12T1852` | 86.1442 | 0.1929 | 229 |
| 14 | `2026-02-12T192423.jsonl` | `cd36f38` | `cd36f38\n02-12T1924` | 81.1741 | 0.1898 | 229 |
| 15 | `2026-02-12T201222.jsonl` | `382f2f1` | `382f2f1\n02-12T2012` | 113.4620 | 0.2287 | 232 |
| 16 | `2026-02-12T205755.jsonl` | `adf2705` | `adf2705\n02-12T2057` | 131.5244 | 0.2453 | 232 |

## SOTA Reference — Qiskit Aer (Full Suite)

Horizontal reference lines for the progress chart. These are Aer's full-suite totals — a fixed target measured once under the cold/warm schedule. **Do not re-run** the comparison each iteration; just read the values below.

| metric | aer_total_s |
|---|---:|
| cold_1000 | 135.0238 |
| warm_10000 | 145.5009 |

## Notes

- `null` means the checkpoint did not record that shot count for at least one case, or the SOTA comparison has not been run yet.
- Cases that OOM are excluded entirely from totals.
- The `cases_complete` column tracks how many cases completed both shot counts.
- SOTA reference values of `null` mean the comparison has not been run yet. Once populated, they are plotted as horizontal dashed lines on `docs/progress.png`.
- Era 1 data is preserved for historical context but uses a different methodology (all-warm, 5 shot counts). Direct comparison between eras is not meaningful.
