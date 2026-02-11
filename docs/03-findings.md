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

### 6. MPS sampler offload (terminal fast path)

On MPS, offloading terminal sampling/counting from device kernels to CPU improved broad static-case performance.

Why it worked:

- `torch.multinomial` / `bincount` on MPS incurred significant backend overhead and sync cost at these shapes.
- Final-distribution transfer size is small relative to full-state gate evolution cost, so CPU sampling won overall.

## What Did Not Work

### Local-axis permutation `index_select` path on MPS

Attempt:

- Replace global permutation gather with reshape/permute/local-axis `index_select`.

Outcome:

- Clear regressions on permutation-heavy cases; reverted.

Likely reason:

- On MPS, the local-axis approach induced more expensive layout/copy behavior than expected.

Conclusion:

- "Smaller local indexing" is not automatically faster; backend-specific memory behavior must be measured, not inferred.

### Alternative gate contraction paths on MPS

Tested `torch.einsum`, `torch.tensordot`, and flat-strided indexing as replacements for permute→reshape→matmul in dense gate application.

Results:

- `einsum`: consistently 5-20% slower than current path on MPS. Hard-crashes at 15+ qubits (MPS rank-16 limit: batch + 15 qubit dims + gate output dim = 17). Not viable.
- `tensordot`: performance-neutral (within noise). Produces cleaner code by eliminating explicit permute/inverse-permute and the axis permutation cache. Adopted for clarity.
- Flat-strided indexing: slower at low qubit counts, marginally faster at 15q. Not worth the complexity.

Conclusion:

- The `copy_` overhead in dense gate application on MPS is the inherent cost of the operation, not an artifact of the contraction strategy. Reducing gate count (fusion) is higher leverage than changing how each gate is applied.

## Backend-Specific Reality (MPS)

Current traces indicate:

- Pre-H0 heavy static cases were dominated by `aten::copy_` / `aten::to`.
- Post-H0, dominant costs are now concentrated in:
  - dynamic-feedback indexing/scalar-sync paths (`nonzero` / `index` / `item`)
  - one-shot unitary evolution overhead for selected deep/high-qubit cases

This shifts optimization focus to:

1. reduce dynamic conditional/indexing overhead
2. reduce one-shot unitary overhead (fewer passes, fewer expensive index-map builds)
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
2. dynamic conditional path optimization
3. gate fusion inside unitary segments
4. MPS rank-limit-safe execution for higher-qubit coverage
