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
| 8 | `4e4b2cb` | Terminal-measurement sampling fast path: for circuits without conditionals/non-terminal measurements, evolve once and sample many shots from final distribution. | Expanded suite totals: @1000 `60.33s -> 4.76s` (`12.67x`), @10000 `592.91s -> 5.23s` (`113.35x`). 22/22 runnable cases still PASS. | Worked (major) | `2026-02-10T230611.jsonl` vs `2026-02-10T234124.jsonl` |
| 9 | `7fbe06b` | On MPS terminal-measurement fast path, sample/count on CPU instead of MPS (`multinomial`/`bincount` offload) to avoid backend sampler overhead and sync stalls. | Expanded suite totals: @1000 `4.76s -> 2.96s` (`1.61x`), @10000 `5.23s -> 3.17s` (`1.65x`). 22/22 runnable cases still PASS. | Worked | `2026-02-10T234124.jsonl` vs `2026-02-10T235128.jsonl` |
| 10 | `c6d7826` (working tree benchmark state) | Implement H2 dynamic branch engine for conditional/mid-circuit execution with branch splitting/merging and fallback heuristics (enabled by default for dynamic circuits with `n_qubits >= 3`). | Expanded suite totals: @1000 `2.96s -> 2.80s` (`1.06x`), @10000 `3.17s -> 3.00s` (`1.06x`). Compare-vs-Aer (full suite) improved from native/aer ratio `5.79x -> 4.99x` at 1000 shots and `1.47x -> 1.05x` at 10000 shots. | Worked (targeted dynamic gain) | `2026-02-10T235128.jsonl` vs `2026-02-11T010850.jsonl`; `compare-2026-02-11T005222.jsonl` vs `compare-2026-02-11T010956.jsonl` |
| 11 | `8fe4ed4` (working tree benchmark state) | H2 pass 2: remove dynamic measurement boolean index writes (`index_put_/nonzero`) via mask multiplication and replace `allclose` merge checks with compact state-signature merge bucketing. | Controlled A/B (same commit, full suite, native only, repetitions=3): branch engine enabled vs disabled gave `@1000 3.97s -> 2.68s` (`1.48x`) and `@10000 4.30s -> 2.78s` (`1.55x`). Dynamic-only speedup: `1.69x` (@1000), `1.78x` (@10000). | Worked (major dynamic-path gain) | `compare-2026-02-11T012341.jsonl` vs `compare-2026-02-11T012324.jsonl` |
| 12 | `d310883` (working tree benchmark state) | Replace hard branch-engine gate (`n_qubits >= 3`) with workload-aware dispatch: use branch mode for low-qubit circuits when dynamic op count is high (env-tunable thresholds). | Controlled A/B (same commit, full suite, native only): low-qubit branch dispatch enabled vs disabled gave `@1000 2.00s -> 1.82s` (`1.10x`) and `@10000 2.13s -> 1.83s` (`1.16x`). Dynamic-only speedup: `1.17x` (@1000), `1.29x` (@10000). | Worked (general dynamic dispatch gain) | `2026-02-11T014954.jsonl` vs `2026-02-11T014743.jsonl`; profiler: `trace_adaptive_feedback_120_10000_current.json` vs `trace_adaptive_feedback_120_10000_h3_default.json` |
| 13 | `d310883` (working tree benchmark state) | Remove dynamic dispatch heuristics and always route dynamic circuits through the branch engine. | Simplified policy; on the current benchmark suite, this is behavior-equivalent because all dynamic cases already satisfied the prior routing rule (`n_qubits >= 3` or `dynamic_ops >= 128`). | Worked (simplification, neutral expected impact) | Dynamic-case op audit from `2026-02-11T020119.jsonl` confirms all current dynamic cases would branch under old and new policy. |
| 14 | Uncommitted analysis experiment | Force static circuits through branch engine (disable terminal-sampling fast path + force dynamic plan) to test unified-engine viability. | Regressed: across 18 static cases at 10000 shots, median total `0.9727s -> 1.4996s` (`1.54x` slower). Worst slowdowns include `ghz_state` (`3.14x`) and `simple_grovers` (`2.51x`). | Did not work | Direct monkeypatch timing experiment (shots=10000, reps=2) run on current tree. |

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

Run: `benchmarks/results/2026-02-11T014743.jsonl`

- Completed cases: 22
- Correctness: 22/22 PASS
- Total @1000: `1.82s`
- Total @10000: `1.83s`
- Speedup vs expanded-suite baseline (`2026-02-10T230611.jsonl`):
  - @1000: `33.09x`
  - @10000: `323.20x`
- Speedup vs post-H0.1 baseline (`2026-02-10T235128.jsonl`):
  - @1000: `1.62x`
  - @10000: `1.73x`

Known backend-limit failures (not OOM):

- `ghz_state_16`: `MPS supports tensors with dimensions <= 16, but got 17`
- `ghz_state_18`: `MPS supports tensors with dimensions <= 16, but got 19`

## Key Takeaways

What worked:

1. Removing asymptotic waste (`O(4^n)` matrix build paths) produced the first major wins.
2. Structural gate specialization (diagonal, then permutation) produced the next major wins.
3. Terminal-measurement sampling fast path produced the largest single step on the expanded suite by removing shot-scaled unitary replay on static circuits.
4. On MPS, terminal-measurement sampling/counting offload to CPU removed a backend-specific sampler bottleneck and delivered another broad step improvement.
5. Cache cleanup/memory-transfer cleanup gave smaller but real gains.
6. Dynamic branch execution improved dominant dynamic cases (especially `adaptive_feedback_5q` and `teleportation`) and narrowed the external full-suite gap to near parity at 10000 shots.
7. Branch-engine pass-2 (mask multiplication + signature-based merge) removed the remaining dynamic indexing and `allclose` merge bottlenecks and delivered another large dynamic speedup in controlled A/B.
8. Workload-aware dynamic dispatch (op-count-based threshold) preserved the general branch-engine strategy while unlocking additional speedups on high-feedback low-qubit circuits without hard-coding benchmark case IDs.
9. Always routing dynamic circuits to the branch engine simplifies policy and avoids overfitting dispatch knobs; on current cases, it is behavior-equivalent to the previous thresholded routing.

What did not:

1. Local-axis permutation `index_select` looked promising conceptually but regressed on MPS in practice and was reverted.
2. Forcing static circuits through the branch engine (instead of terminal-measurement sampling fast path) regressed materially (`~1.54x` slower aggregate in static-case experiment).

What changed strategically:

1. Benchmarking is now broad and always-on; optimization work is measured against general simulator behavior, not narrow case subsets.
2. Post-H0/H0.1 bottlenecks shifted decisively to dynamic/feedback execution paths; static terminal-measurement circuits are now relatively cheap.
