# Optimization Progress - 2026-02-10

## Environment

- Machine: Apple M1 Max, 32 GB
- Backend: PyTorch MPS (`torch>=2.9.0`)
- Benchmark command: `uv run bench -v`
- Baseline run timestamp: `2026-02-10T195547.jsonl`
- Latest run timestamp: `2026-02-10T195955.jsonl`

## Optimization Loop Summary

All iterations preserved benchmark correctness (all cases PASS).

| Commit | Change | Total @ 1000 shots | Delta vs previous |
|---|---|---:|---:|
| `3df121d` | Baseline after H1/H2 | 370.99s | - |
| `587a03f` | Vectorize batched measurement collapse (remove per-shot projection loop) | 7.79s | 47.6x faster |
| `eb476df` | Cache gate tensors on device in batched path | 6.53s | 1.19x faster |
| `0c3186d` | Use cached measurement weight masks for p(|1>) matmul | 6.41s | 1.02x faster |

Net result from baseline (`3df121d`) to latest (`0c3186d`): **57.9x faster** at 1000 shots.

## Per-Case Impact at 1000 Shots

| Case | Baseline (`3df121d`) | Latest (`0c3186d`) | Speedup |
|---|---:|---:|---:|
| `bell_state` | 1.462s | 0.038s | 38.7x |
| `simple_grovers` | 3.095s | 0.037s | 84.3x |
| `real_grovers` | 198.469s | 5.731s | 34.6x |
| `ghz_state` | 144.913s | 0.175s | 829.0x |
| `qft` | 20.832s | 0.388s | 53.7x |
| `teleportation` | 2.221s | 0.043s | 51.8x |

## Profiler Notes

Targeted trace command:

```bash
uv run bench-trace real_grovers 1000 --no-stack
```

Key observations after the latest changes:

- `aten::mm` is now a major visible cost (real compute), around 43% self CPU in trace tables.
- Measurement loop bottleneck from Python/per-shot projection is removed.
- Transfer/sync-attributed `aten::to` / `aten::copy_` remains prominent in profiler totals.
  - This includes synchronization effects and host-device boundaries (for example final result extraction).

## Next Candidates

1. Reduce host-device synchronization and copy overhead in measurement/result extraction.
2. Specialize diagonal gates (`Z`, `S`, `T`, `RZ`, controlled phase) as elementwise multiplies.
3. Evaluate selective conditional gate application strategy to avoid clone/restore overhead.

## Benchmark Expansion (2026-02-11)

The benchmark suite was expanded from 16 to 24 configured cases (22 currently runnable on MPS due to existing rank-limit failures in `ghz_state_16` and `ghz_state_18`).

Added always-on synthetic families:
- `reversible_mix_13`
- `reversible_mix_15`
- `clifford_scrambler_14`
- `brickwork_entangler_15`
- `random_universal_12`
- `random_universal_14`
- `diagonal_mesh_15`
- `adaptive_feedback_5q`

All synthetic cases are deterministic round-trips (`U` then `U^-1`) with expected output `|0...0>` to preserve strict correctness checks while broadening structural coverage.

Expanded-suite baseline run:
- Run timestamp: `2026-02-10T230611.jsonl`
- Commit: `e1fa5b0`
- Total @1000 shots: `60.33s`
- Total @10000 shots: `592.91s`
- Correctness: PASS on all 22 completed cases
- Known backend limits: `ghz_state_16`, `ghz_state_18` still fail on MPS tensor-dimension limit (`<=16`)

Largest added-case costs @10000 shots:
- `random_universal_14`: `186.21s`
- `brickwork_entangler_15`: `158.61s`
- `clifford_scrambler_14`: `63.00s`
- `random_universal_12`: `31.39s`
