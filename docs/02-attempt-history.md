# Attempt History

This document is the canonical record of what has been tried, what worked, and what did not.

## Environment

- Machine: Apple M1 Max, 32 GB
- Backend: PyTorch MPS (`torch>=2.9.0`)
- Benchmark command: `uv run bench -v`

## Attempt Ledger

| # | Commit / State | Hypothesis | Result | Verdict | Evidence |
|---|---|---|---|---|---|
| 1 | `587a03f` | Vectorize batched measurement collapse to remove per-shot projection loop. | Total @1000 (6-case suite): `370.99s -> 7.79s` (`47.6x`). | Worked | `2026-02-10T195547.jsonl` vs `2026-02-10T195724.jsonl` |
| 2 | `eb476df` | Cache gate tensors on device to remove repeated transfers. | Total @1000 (6-case): `7.79s -> 6.53s` (`1.19x`). | Worked | `2026-02-10T195724.jsonl` vs `2026-02-10T195839.jsonl` |
| 3 | `0c3186d` | Cache measurement weight masks for p(|1>) matmul. | Total @1000 (6-case): `6.53s -> 6.41s` (`1.02x`). | Worked (small) | `2026-02-10T195839.jsonl` vs `2026-02-10T195955.jsonl` |
| 4 | `63d943f` | Add diagonal gate metadata and diagonal fast path. | Total @10k (14-case suite): `509.54s -> 181.28s` (`2.81x`). | Worked | `2026-02-10T204227.jsonl` vs `2026-02-10T210225.jsonl` |
| 5 | `e1fa5b0` (bench run at equivalent working tree) | Add monomial/permutation gate metadata + permutation fast path. | Total @10k (14-case): `181.28s -> 94.92s` (`1.91x`). | Worked | `2026-02-10T210225.jsonl` vs `2026-02-10T215408.jsonl` |
| 6 | Uncommitted experiment, reverted | Replace global permutation gather with local-axis `index_select` path. | Regressed strongly on MPS: `real_grovers` `15.67s -> 22.99s`; `toffoli_oracle_13` `10.85s -> 17.55s` at 10k. | Did not work (reverted) | `2026-02-10T221843.jsonl` vs `2026-02-10T215408.jsonl` |
| 7 | `a6c5d1b` | Expand benchmark to broad synthetic always-on families (no core/extended split). | Suite expanded from 16 configured cases to 24 configured cases (22 runnable on MPS). New expanded baseline @10k: `592.91s`. | Worked (coverage) | `2026-02-10T230611.jsonl` |

## Current Benchmark Scope

Always-on suite now includes:

- Original functional/algorithmic cases
- Extended sweeps (`qft_14`, `phase_ladder_13`, `toffoli_oracle_13`, etc.)
- New synthetic families:
  - `reversible_mix_13`
  - `reversible_mix_15`
  - `clifford_scrambler_14`
  - `brickwork_entangler_15`
  - `random_universal_12`
  - `random_universal_14`
  - `diagonal_mesh_15`
  - `adaptive_feedback_5q`

## Latest Baseline (Expanded Suite)

Run: `benchmarks/results/2026-02-10T230611.jsonl`

- Completed cases: 22
- Correctness: 22/22 PASS
- Total @1000: `60.33s`
- Total @10000: `592.91s`

Known backend-limit failures (not OOM):

- `ghz_state_16`: `MPS supports tensors with dimensions <= 16, but got 17`
- `ghz_state_18`: `MPS supports tensors with dimensions <= 16, but got 19`

## Key Takeaways

What worked:

1. Removing asymptotic waste (`O(4^n)` matrix build paths) produced the biggest wins.
2. Structural gate specialization (diagonal, then permutation) produced the next biggest wins.
3. Cache cleanup/memory-transfer cleanup gave smaller but real gains.

What did not:

1. Local-axis permutation `index_select` looked promising conceptually but regressed on MPS in practice and was reverted.

What changed strategically:

1. Benchmarking is now broad and always-on; optimization work is measured against general simulator behavior, not narrow case subsets.
