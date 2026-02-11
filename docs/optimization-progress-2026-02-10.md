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
