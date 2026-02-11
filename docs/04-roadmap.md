# Roadmap (Current)

This plan is based on the completed attempt history in `docs/02-attempt-history.md`.

## Objective

Improve state-vector simulation performance generally, not for a narrow benchmark shape.

Constraints:

- Keep one always-on benchmark suite (no core/extended split).
- Preserve correctness.
- Prefer approaches that improve whole circuit families, not one-off case hacks.

## What Is Already Done

Implemented and validated:

1. Tensor-contraction gate path (removed full-matrix gate construction in batched path).
2. Vectorized batched measurement collapse.
3. Gate tensor and measurement-weight caching.
4. Diagonal gate metadata + diagonal fast path.
5. Monomial/permutation metadata + permutation fast path.

Attempted and rejected:

1. Local-axis permutation `index_select` replacement for global gather.
Reason: significant regressions on MPS; reverted.

## Current Bottleneck View

Post-diagonal/permutation traces show copy/transfer attribution dominating (`aten::copy_`, `aten::to`) with math (`aten::mm`) no longer primary in targeted heavy cases.

Implication:

- Next wins likely come from reducing memory movement and dispatch overhead per applied operation.

## Next Hypotheses (Not Implemented Yet)

### H1. Execution-Plan Compilation (Circuit IR)

Compile a circuit into execution segments and reusable per-op artifacts:

- segment by measurement/conditional boundaries
- precompute operation metadata, device tensors, index maps, and dispatch kind
- reduce runtime Python branching and repeated cache lookups

Expected impact: broad, moderate improvement across deep circuits.

### H2. Gate Fusion Within Unitary Segments

Within safe boundaries, fuse adjacent compatible gates to reduce full-state passes:

- single-qubit chain fusion on same wire
- commuting diagonal block fusion
- permutation-chain fusion where valid

Expected impact: high for deep circuits with many small gates.

### H3. Reduce Result/Measurement Sync Overhead

Minimize host-device synchronization and copy churn in measurement/result collection:

- audit `.cpu()`/sync barriers
- keep operations device-side until required

Expected impact: small-to-moderate, especially for dynamic/measurement-heavy cases.

### H4. Address MPS Rank-Limit Failure Path

Investigate application path for high-qubit cases that currently fail due tensor rank >16 on MPS.

Expected impact: expands runnable benchmark space; correctness/coverage improvement rather than raw speedup.

## Validation Protocol

For each hypothesis:

1. Commit code before benchmark run.
2. Run full suite: `uv run bench -v`.
3. Compare totals and per-family deltas vs latest baseline artifact.
4. Accept only if no correctness regressions and broad-family performance improves.
5. Record outcome in `docs/02-attempt-history.md` as worked / did not work.
