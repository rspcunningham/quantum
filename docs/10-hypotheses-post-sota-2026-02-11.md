# Hypotheses After SOTA Comparison (2026-02-11)

This document updates optimization hypotheses after running apples-to-apples external comparisons.
It explicitly compares conclusions against `docs/08-design-h1-dynamic-branch-engine.md`.

## 1) Inputs

Benchmark comparison artifacts:

1. `benchmarks/results/compare-2026-02-11T005222.jsonl` (`suite=full`, backends: native + Aer, repetitions=3)
2. `benchmarks/results/compare-2026-02-11T005251.jsonl` (`suite=static`, backends: native + Aer + qsim, repetitions=3)
3. `benchmarks/results/compare-2026-02-11T005222.md` (merged report)

Targeted post-comparison profiler traces:

1. `benchmarks/results/trace_adaptive_feedback_5q_1000_postsota.json`
2. `benchmarks/results/trace_random_universal_14_10000_postsota.json`

Reference design compared:

1. `docs/08-design-h1-dynamic-branch-engine.md`

## 2) What We Learned From SOTA Comparison

### 2.1 Aggregate gap to current external baselines

Full suite (`native` vs `aer`):

- `@1000`: `3.532s` vs `0.610s` (`5.79x` slower)
- `@10000`: `4.320s` vs `2.935s` (`1.47x` slower)

Static intersection (`native` vs `aer` vs `qsim`):

- `@1000`: native `0.982s`, aer `0.213s`, qsim `0.212s`
- `@10000`: native `1.014s`, aer `0.281s`, qsim `0.219s`

### 2.2 Gap decomposition (full suite only, supported cases)

At `1000` shots:

- Static gap (`native - aer`): `0.655s`
- Dynamic gap (`native - aer`): `2.267s`
- Dynamic share of gap: `77.6%`

At `10000` shots:

- Static gap (`native - aer`): `0.599s`
- Dynamic gap (`native - aer`): `0.786s`
- Dynamic share of gap: `56.8%`

Interpretation:

- Dynamic is the largest gap driver at `1000`.
- By `10000`, static and dynamic are both material; dynamic-only work is not sufficient.

### 2.3 Per-case gap concentration

Largest `@10000` full-suite gaps include:

1. `adaptive_feedback_5q` (`+0.543s`, dynamic)
2. `adaptive_feedback_120` (`+0.179s`, dynamic)
3. `phase_ladder_13` (`+0.126s`, static)
4. `diagonal_mesh_15` (`+0.094s`, static)
5. `phase_ladder` (`+0.090s`, static)

This is mixed dynamic + static concentration, not one-family-only.

### 2.4 Post-SOTA profiler evidence

`adaptive_feedback_5q`, 1000 shots:

- `aten::nonzero`: `73.42%` total CPU
- `aten::index_put_`: `38.11%` total CPU
- `aten::index`: `36.65%` total CPU
- `aten::item` + `aten::_local_scalar_dense`: `34.71%` total CPU

`random_universal_14`, 10000 shots:

- `aten::copy_`: `78.92%` total CPU
- `aten::to` + `aten::_to_copy`: `78.25%` / `78.13%` total CPU
- `aten::mm`: `9.95%` total CPU

Interpretation:

- Dynamic bottleneck remains conditional-mask/index/scalar-sync overhead.
- Static bottleneck is dominated by movement/conversion overhead, not matrix multiply throughput.

## 3) Comparison Against `docs/08-design-h1-dynamic-branch-engine.md`

### 3.1 What still holds

`docs/08` core diagnosis for dynamic circuits remains correct:

1. Conditional path is dominated by `nonzero`/advanced indexing/scalar sync.
2. Current mask-and-restore strategy is intrinsically expensive.
3. A branch-based dynamic executor is a valid general fix direction.

### 3.2 What changed

`docs/08` positioned dynamic rewrite as the highest-priority single path.

Updated evidence says:

1. Dynamic should remain top priority for `@1000`.
2. Static movement/copy overhead is now co-priority for closing `@10000` SOTA gap.
3. Branch engine alone cannot close total SOTA delta.

So `docs/08` is still necessary, but no longer sufficient.

## 4) Updated Hypothesis Set (v2)

## H1. Compiled Execution Plan With Persistent Device Artifacts

Hypothesis:

- Precompiling circuit operations into a reusable plan with device-resident tensors and precomputed index metadata will remove most `to/_to_copy/copy_` overhead and improve both static and dynamic paths.

Implementation sketch:

1. Compile flattened ops once into plan steps.
2. Pre-stage gate tensor/diagonal/permutation tensors on device once.
3. Precompute per-gate metadata (`targets`, `non_targets`, `perm`, `inv_perm`, subindex maps).
4. Reuse plan across runs for the same `(circuit, n_qubits, n_bits, device)`.

Why it is general:

- This is infrastructure for all workloads, not tied to one benchmark family.

Acceptance targets:

1. In `random_universal_14` trace, `aten::to + aten::_to_copy` share reduced below `25%`.
2. Static total at `@10000` improves from `~1.01s` toward `<=0.60s` on current machine.

## H2. Dynamic Branch Engine (Carry Forward `docs/08`, Refined Scope)

Hypothesis:

- Replacing conditional mask+restore with branch-state execution will remove dynamic indexing/scalar-sync hotspots and close most `@1000` dynamic gap.

Implementation sketch:

1. Use branch states keyed by classical register value with shot counts.
2. Measurement splits branch counts via binomial sampling.
3. Conditionals apply only to matching branches.
4. Keep existing path as fallback behind a safety switch.

Why it is general:

- Preserves exact semantics for all dynamic-feedback circuits.

Acceptance targets:

1. In `adaptive_feedback_5q` trace, combined `nonzero/index/index_put/item` share reduced below `20%`.
2. Dynamic total at `@1000` improves from `~2.66s` toward `<=1.20s`.

## H3. Static Unitary Segment Fusion + Layout Tracking

Hypothesis:

- Reducing full-state passes by fusing compatible gate chains and tracking logical layout (instead of repeated full permute/restore) will materially reduce static deep-circuit cost.

Implementation sketch:

1. Identify unitary segments and apply bounded fusion (`<=3` or `<=4` qubits).
2. Fuse diagonal chains and permutation chains aggressively when safe.
3. Introduce logical-to-physical qubit map to defer expensive layout restoration.

Why it is general:

- Targets structural inefficiency in deep static circuits (`qft*`, `phase_ladder*`, `random_universal*`, `diagonal_mesh*`).

Acceptance targets:

1. `phase_ladder_13` and `random_universal_14` each improve by `>=2x` at `@10000`.
2. Static total at `@10000` moves below `0.45s`.

## H4. MPS Rank-Limit-Safe High-Qubit Path

Hypothesis:

- A shape-safe execution path that avoids rank-`>16` tensor views on MPS can restore coverage for `ghz_state_16` and `ghz_state_18` without regressing existing paths.

Implementation sketch:

1. Avoid rank-`(batch + qubit_axes)` representations above backend limits.
2. Use reshaped 2D/blocked contractions or a fallback kernel for high-qubit cases.

Acceptance targets:

1. `ghz_state_16` and `ghz_state_18` become supported on native MPS in compare harness.
2. No correctness regressions in existing 22 runnable cases.

## 5) Priority Order

1. H1 (plan + device artifact residency): highest leverage, low semantic risk, prerequisite for broader wins.
2. H2 (dynamic branch engine): direct carry-forward from `docs/08`, now scoped as second pillar.
3. H3 (fusion + layout): major static gap closer after plan infrastructure.
4. H4 (rank-limit path): coverage and robustness.

## 6) Decision

Compared to `docs/08`, we keep the branch-engine strategy, but update overall optimization strategy to a two-pillar program:

1. static movement-path reduction (H1 + H3),
2. dynamic control-flow rewrite (H2).

This is the minimum coherent strategy to close both `@1000` and `@10000` SOTA gaps.
