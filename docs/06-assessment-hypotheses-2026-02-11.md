# Assessment and Hypotheses (2026-02-11)

This document captures the current state of evidence before the next implementation step.
It is intentionally analysis-first and hypothesis-driven.

Inputs used:

- Benchmark artifact: `benchmarks/results/2026-02-10T230611.jsonl`
- Targeted traces:
  - `benchmarks/results/trace_random_universal_14_100.json`
  - `benchmarks/results/trace_qft_14_100.json`
  - `benchmarks/results/trace_reversible_mix_15_100.json`
  - `benchmarks/results/trace_adaptive_feedback_5q_1000.json`
- External references (primary sources): links listed in "SOTA Research Notes" below.

## 1) Current Baseline Decomposition

From `2026-02-10T230611.jsonl`:

- Completed cases: 22
- Correctness: 22/22 PASS
- Total @1000: `60.33s`
- Total @10000: `592.91s`

At 10000 shots, time concentration is extreme:

| Rank | Case | Time @10000 | Share |
|---|---|---:|---:|
| 1 | `random_universal_14` | 186.21s | 31.41% |
| 2 | `brickwork_entangler_15` | 158.61s | 26.75% |
| 3 | `clifford_scrambler_14` | 63.00s | 10.63% |
| 4 | `qft_14` | 36.89s | 6.22% |
| 5 | `random_universal_12` | 31.39s | 5.29% |

Concentration summary:

- Top 2 = `58.16%`
- Top 5 = `80.30%`
- Top 8 = `91.73%`

## 2) Structural Split: Static vs Dynamic Circuits

Using operation structure over the 22 passing cases:

- 18/22 cases have no conditionals and no non-terminal measurements.
- These 18 "static terminal-measurement" cases account for `590.55s` / `592.91s` = `99.6%` of total @10000.
- The 4 dynamic/mid-circuit cases account for only `2.36s` (`0.4%`) @10000.

Interpretation:

- The current suite is overwhelmingly dominated by unitary evolution workloads.
- Any optimization that removes shot-scaling from static circuits will dominate total-suite impact.

## 3) Profiler Evidence

### 3.1 Dense/large unitary cases: copy/movement dominates

`random_universal_14`, 100 shots:

- `aten::copy_`: 95.31% self CPU
- `aten::to` / `aten::_to_copy`: 95.23% total CPU
- `aten::mm`: 1.48% total CPU

`qft_14`, 100 shots:

- `aten::copy_`: 84.07% self CPU
- `aten::to` / `aten::_to_copy`: 84.19% total CPU
- `aten::mm`: 3.24% total CPU

### 3.2 Permutation-heavy case: still copy-heavy with indexing overhead

`reversible_mix_15`, 100 shots:

- `aten::copy_`: 55.62% self CPU
- `aten::to` / `aten::_to_copy`: 55.50% total CPU
- `aten::arange`: 11.15% total CPU

### 3.3 Dynamic feedback case: conditional/indexing path dominates

`adaptive_feedback_5q`, 1000 shots:

- `aten::nonzero`: 50.02% self CPU, 73.42% total CPU
- `aten::index_put_`: 38.02% total CPU
- `aten::index`: 36.79% total CPU
- `aten::item` + `aten::_local_scalar_dense`: 33.88% total CPU

Inference (not a direct profiler fact): dynamic performance is limited by boolean indexing/scalar sync behavior in conditional execution, not linear algebra throughput.

## 4) First-Principles Gap

For ideal state-vector simulation with terminal measurements:

- Unitary evolution should be independent of shot count.
- Shot count should primarily affect only the final sampling step.

Current behavior in our benchmark is near-linear in shots for dominant cases (1000 -> 10000 often ~10x), which indicates we are still scaling unitary evolution with shot count in static circuits.

That is an algorithmic mismatch, not a micro-optimization problem.

## 5) Prototype Check (Analysis-Only, Not Yet Productized)

A quick prototype path was timed for eligible static circuits:

1. execute gate sequence once with `batch_size=1`
2. skip terminal measurement collapse during evolution
3. sample `num_shots` outcomes from the final probability vector

Aggregate estimate over the 18 static cases at 10000 shots:

- Baseline: `590.5529s`
- Prototype: `3.0194s`
- Estimated aggregate speedup: `195.6x`

Selected case deltas:

| Case | Baseline @10000 | Prototype | Est. Speedup |
|---|---:|---:|---:|
| `random_universal_14` | 186.21s | 0.14s | 1350.7x |
| `brickwork_entangler_15` | 158.61s | 0.10s | 1555.9x |
| `qft_14` | 36.89s | 0.32s | 113.8x |
| `real_grovers` | 16.55s | 0.09s | 178.7x |
| `phase_ladder_13` | 24.78s | 0.15s | 166.5x |

Prototype correctness spot-checks passed on:

- `real_grovers`
- `qft_14`
- `random_universal_14`
- `diagonal_mesh_15`
- `toffoli_oracle_13`

Important caveat:

- This prototype still used Python-side count aggregation and was not integrated into production code.
- Numbers are directional but strong enough to prioritize implementation.

## 6) SOTA Research Notes (Guide, Not Dogma)

We should use external systems as directional evidence and adapt only what survives our measurements.

### qsim (Google)

- qsim documents runtime as roughly proportional to `g * 2^n` for circuits without intermediate measurements.
- qsim also notes that many samples can be drawn at minimal additional cost when measurements are terminal.
- qsim exposes explicit gate fuser components and fused-gate size controls.

References:

- https://quantumai.google/qsim/choose_hw
- https://quantumai.google/reference/cc/qsim/class/qsim/basic-gate-fuser
- https://quantumai.google/reference/cc/qsim/namespace/qsim
- https://github.com/quantumlib/qsim

### Qiskit Aer

- AerSimulator statevector-style methods explicitly support sampling outcomes from ideal circuits with measurements at end.
- Aer release notes describe shot-branching as a dedicated dynamic-circuit optimization.

References:

- https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html
- https://qiskit.github.io/qiskit-aer/release_notes.html

### NVIDIA cuStateVec

- cuStateVec provides dedicated sampler preprocess/sample APIs and generalized permutation matrix application APIs.
- This matches the pattern of separating state evolution from repeated sampling work.

Reference:

- https://docs.nvidia.com/cuda/cuquantum/latest/custatevec/api/functions.html

## 7) Ranked Hypotheses

### H0. Terminal-Measurement Sampling Engine (Highest Priority)

Implement a fast path for circuits with no conditionals and no non-terminal measurements:

- evolve once
- sample many

Why this is general:

- It is mathematically equivalent for ideal simulation.
- It applies to most algorithmic benchmarks and real workloads without mid-circuit feedback.

Expected impact:

- Very high on current suite totals.

Main risks:

- Ensuring bit-order correctness for arbitrary measurement maps.
- Maintaining parity with existing API behavior.

### H1. Execution-Plan Compilation

Compile circuits into reusable execution plans:

- segment boundaries at measurement/conditional barriers
- pre-lowered/cached per-op artifacts (tensors, indices, dispatch kind)
- reduced runtime branching

Expected impact:

- Broad moderate gain.
- Enables cleaner fusion and dynamic-path work.

### H2. Gate Fusion in Unitary Segments

Fuse adjacent compatible ops with strict safety rules and bounded fused width:

- single-qubit chains on same wire
- commuting diagonal chains
- permutation chains
- bounded dense fusion (explicit qubit cap)

Expected impact:

- High on deep circuits with many small gates.

### H3. Dense/Permutation Copy-Path Reduction

Reduce full-state copy/layout churn in gate application:

- minimize permute/materialize/restore patterns
- consider lazy logical-physical layout tracking
- remove redundant global gathers where possible

Expected impact:

- Moderate-to-high for dense/permutation-heavy workloads.

### H4. Dynamic Conditional Path Rewrite

Improve dynamic feedback path:

- remove scalar-sync-heavy condition checks
- avoid expensive boolean advanced indexing restore patterns
- evaluate branch-group execution strategy

Expected impact:

- Moderate for adaptive-feedback workloads.
- Smaller impact on current total benchmark share.

### H5. MPS Rank-Limit-Safe High-Qubit Path

Address known MPS rank failures (`ghz_state_16`, `ghz_state_18`) with a shape-safe execution path.

Expected impact:

- Coverage/correctness expansion.

## 8) Suggested Experiment Order

1. Implement H0 fast path with strict eligibility checks and thorough correctness validation.
2. Re-benchmark full suite and record per-case deltas.
3. If H0 lands as expected, implement H1 plan compilation to unify static and dynamic execution paths.
4. Add H2 fusion on top of compiled segments.
5. Tackle H3/H4 based on post-H0 profiler shape.

Acceptance criteria for each accepted optimization:

- 22/22 current passing cases remain correct.
- Full-suite improvement, not isolated-case improvement.
- Attempt recorded in `docs/02-attempt-history.md`.
