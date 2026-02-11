# Findings (Current)

This document captures stable conclusions from completed experiments.

For chronological attempt details and exact metrics, see:
`docs/02-attempt-history.md`.

## First-Principles Model

For state-vector simulation, most nontrivial gates fundamentally require touching `O(2^n)` amplitudes.

Practical consequence:

- Large wins come from eliminating accidental `O(4^n)` work and avoiding unnecessary full-state passes.
- For static (terminal-measurement) circuits, shot count should not scale unitary evolution cost.
- After asymptotics and shot-scaling mismatches are fixed, memory movement, indexing, and dispatch overhead dominate.

## What Worked

### 1. Removing full-matrix construction paths

Replacing Kronecker/swap-heavy gate application with tensor-contraction style execution produced the largest gains.

Why it worked:

- Removed avoidable exponential-overhead constants and matrix build churn.

### 2. Vectorized measurement collapse

Eliminating per-shot projection loops produced major batch-scaling improvements.

Why it worked:

- Replaced Python loops and matrix builds with bulk tensor operations.

### 3. Structural gate specialization

Diagonal and permutation/monomial metadata with dedicated fast paths produced another major step change.

Why it worked:

- Avoided generic dense matmul for gates that only scale or permute amplitudes.

### 4. Device/cache hygiene

Caching gate tensors and measurement masks gave smaller but consistent wins.

Why it worked:

- Reduced repeated device transfer and setup overhead.

### 5. Terminal-measurement sampling fast path

For circuits without conditionals and without non-terminal measurements, evolving once and sampling many shots from the final distribution produced a major broad win.

Why it worked:

- Removed unnecessary `O(num_shots * gates * 2^n)` replay in static circuits.
- Restored first-principles shot behavior: one unitary evolution plus independent sampling.

## What Did Not Work

### Local-axis permutation `index_select` path on MPS

Attempt:

- Replace global permutation gather with reshape/permute/local-axis `index_select`.

Outcome:

- Clear regressions on permutation-heavy cases; reverted.

Likely reason:

- On MPS, the local-axis approach induced more expensive layout/copy behavior than expected.

Conclusion:

- “Smaller local indexing” is not automatically faster; backend-specific memory behavior must be measured, not inferred.

## Backend-Specific Reality (MPS)

Current traces indicate:

- Pre-H0 heavy static cases were dominated by `aten::copy_` / `aten::to`.
- Post-H0, dominant costs are now concentrated in:
  - one-shot unitary evolution overhead for deep/high-qubit cases
  - dynamic-feedback indexing/scalar-sync paths (`nonzero` / `index` / `item`)

This shifts optimization focus to:

1. reduce one-shot unitary overhead (fewer passes, fewer expensive index-map builds)
2. reduce dynamic conditional/indexing overhead
3. reduce runtime dispatch overhead with compiled execution plans

## Benchmark Strategy Finding

A narrow benchmark can reward local hacks that do not generalize.

Now adopted:

- one always-on, broadened suite with synthetic and structured families
- deterministic correctness checks (round-trip constructions where possible)
- optimizations judged on broad-family behavior, not single-case wins

## Open Directions

Most promising next directions:

1. execution-plan compilation (segment + precompute)
2. gate fusion inside unitary segments
3. dynamic conditional path optimization
4. MPS rank-limit-safe execution for higher-qubit coverage
