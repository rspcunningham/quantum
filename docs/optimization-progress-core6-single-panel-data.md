# Optimization Progress Single-Panel Raw Data (Core-6)

This file contains all points used to generate:

- `docs/images/optimization-progress-core6-single-panel.png`

Data source files are in `benchmarks/results/` and are listed below in plotting order.

## Core-6 Cases Included

- `bell_state`
- `simple_grovers`
- `real_grovers`
- `ghz_state`
- `qft`
- `teleportation`

## Plot Metadata

- Y-axis: log scale
- Series: totals for shot counts `1`, `10`, `100`, `1000`, `10000`
- `label_x`: exact x-axis text (`git_hash` + compact timestamp)
- `annotation`: callout labels drawn on the chart

| idx | checkpoint_jsonl | git_hash | label_x | total_1_s | total_10_s | total_100_s | total_1000_s | total_10000_s | annotation |
|---:|---|---|---|---:|---:|---:|---:|---:|---|
| 0 | `2026-02-10T195547.jsonl` | `3df121d` | `3df121d\n02-10T1955` | 0.4152 | 4.4596 | 36.8003 | 370.9927 | null | Baseline |
| 1 | `2026-02-10T195724.jsonl` | `587a03f` | `587a03f\n02-10T1957` | 0.1967 | 0.4135 | 0.9643 | 7.7868 | null |  |
| 2 | `2026-02-10T195839.jsonl` | `eb476df` | `eb476df\n02-10T1958` | 0.1312 | 0.3496 | 0.8244 | 6.5297 | null |  |
| 3 | `2026-02-10T195955.jsonl` | `0c3186d` | `0c3186d\n02-10T1959` | 0.1120 | 0.6522 | 0.7629 | 6.4103 | null |  |
| 4 | `2026-02-10T201437.jsonl` | `03a0dbb` | `03a0dbb\n02-10T2014` | 0.0872 | 0.2855 | 0.7337 | 6.3512 | null |  |
| 5 | `2026-02-10T204227.jsonl` | `9701dc6` | `9701dc6\n02-10T2042` | 0.0925 | 0.2698 | 0.7693 | 6.8295 | 68.4851 |  |
| 6 | `2026-02-10T210225.jsonl` | `63d943f` | `63d943f\n02-10T2102` | 0.0948 | 0.2702 | 0.7829 | 6.8750 | 67.1232 |  |
| 7 | `2026-02-10T214958.jsonl` | `63d943f` | `63d943f\n02-10T2149` | 0.0924 | 0.2498 | 0.3415 | 2.0002 | 18.0746 |  |
| 8 | `2026-02-10T215408.jsonl` | `63d943f` | `63d943f\n02-10T2154` | 0.0954 | 0.2475 | 0.3378 | 1.9856 | 17.8640 |  |
| 9 | `2026-02-10T230611.jsonl` | `e1fa5b0` | `e1fa5b0\n02-10T2306` | 0.0851 | 0.2479 | 0.3373 | 2.0226 | 18.7998 |  |
| 10 | `2026-02-10T234124.jsonl` | `4e4b2cb` | `4e4b2cb\n02-10T2341` | 0.0764 | 0.2400 | 0.1916 | 0.1999 | 0.2809 | H0 |
| 11 | `2026-02-10T235128.jsonl` | `7fbe06b` | `7fbe06b\n02-10T2351` | 0.0642 | 0.1198 | 0.1017 | 0.1057 | 0.1419 | H0.1 |
| 12 | `2026-02-11T010850.jsonl` | `c6d7826` | `c6d7826\n02-11T0108` | 0.0759 | 0.0817 | 0.0820 | 0.0850 | 0.0852 | H2-v1 |
| 13 | `2026-02-11T011947.jsonl` | `8fe4ed4` | `8fe4ed4\n02-11T0119` | 0.0854 | 0.0932 | 0.0926 | 0.0899 | 0.0932 |  |
| 14 | `2026-02-11T012820.jsonl` | `8fe4ed4` | `8fe4ed4\n02-11T0128` | 0.0744 | 0.0820 | 0.0799 | 0.0811 | 0.0798 | H2-v2 |
| 15 | `2026-02-11T031009.jsonl` | `bab05d4` | `bab05d4\n02-11T0310` | 0.0832 | 0.0865 | 0.0858 | 0.0859 | 0.0884 |  |
| 16 | `2026-02-11T112001.jsonl` | `a85bf9c` | `a85bf9c\n02-11T1120` | 0.0784 | 0.0748 | 0.0799 | 0.0766 | 0.0805 | Current |

## Notes

- `null` means the checkpoint did not record that shot count for at least one core-6 case.
- The plotted line for a series is therefore discontinuous where `null` appears.
