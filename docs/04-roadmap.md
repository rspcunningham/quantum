# Roadmap (Current)

This plan is based on the completed attempt history in `docs/02-attempt-history.md`.
Latest assessment context: `docs/06-assessment-hypotheses-2026-02-11.md`.

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

From `benchmarks/results/2026-02-10T230611.jsonl`:

- Total @10000 is `592.91s`.
- Top 2 cases contribute `58.16%`.
- Top 5 cases contribute `80.30%`.
- 18/22 passing cases are unitary-until-terminal-measurement and contribute `590.55s` (`99.6%`) of @10000 runtime.

From targeted profiler traces:

- Heavy unitary cases are dominated by copy/transfer attribution (`aten::copy_`, `aten::to`) rather than math (`aten::mm`).
- Dynamic-feedback cases are dominated by boolean indexing and scalar sync (`aten::nonzero`, `aten::index`, `aten::item` / `_local_scalar_dense`).

Implications:

- The biggest remaining win is algorithmic: avoid scaling unitary evolution with shot count when measurements are terminal.
- The next tier is reducing full-state copy/layout churn and runtime dispatch overhead.
- Dynamic-circuit performance requires dedicated conditional/branch handling, but this is currently a much smaller share of total suite time.

## Next Hypotheses (Not Implemented Yet)

### H0. Terminal-Measurement Sampling Fast Path (Highest Priority)

For circuits with:

- no conditional gates, and
- no non-terminal measurements

execute the unitary portion once (single state vector), then sample outcomes for `num_shots` from the final probability distribution.

Implementation design: `docs/07-design-h0-terminal-sampling.md`.

Expected impact: very high, broad, and principled. Removes unnecessary `O(num_shots * gates * 2^n)` evolution for static circuits.

### H1. Execution-Plan Compilation (Circuit IR)

Compile a circuit into reusable execution segments and artifacts:

- segment by measurement/conditional boundaries
- precompute operation metadata, device tensors, index maps, and dispatch kernel choice
- reduce runtime Python branching and repeated cache lookups

Expected impact: broad, moderate improvement across deep circuits.

### H2. Gate Fusion Within Unitary Segments

Within safe boundaries, fuse adjacent compatible gates to reduce full-state passes:

- single-qubit chain fusion on same wire
- commuting diagonal block fusion
- permutation-chain fusion where valid
- bounded dense fusion with explicit width cap (for example, <= 3 or <= 4 qubits)

Expected impact: high for deep circuits with many small gates.

### H3. Reduce Copy/Layout Overhead in Gate Application

Reduce full-state layout thrash in dense/permutation application:

- avoid unnecessary permute->reshape materialization patterns
- investigate lazy logical-to-physical qubit layout tracking
- reduce repeated full-state gathers where a cheaper formulation exists on MPS

Expected impact: moderate-to-high on dense-heavy workloads.

### H4. Dynamic-Circuit Conditional Path Rewrite

Target adaptive-feedback bottlenecks:

- avoid scalar sync in condition checks (`.item()` / Python bool hot path)
- avoid expensive boolean advanced indexing restore patterns
- investigate branch-group/shot-branching style execution for repeated feedback loops

Expected impact: moderate on feedback workloads; lower aggregate impact on current suite totals.

### H5. Address MPS Rank-Limit Failure Path

Investigate application path for high-qubit cases that currently fail due tensor rank >16 on MPS.

Expected impact: expands runnable benchmark space; correctness/coverage improvement rather than raw speedup.

## Validation Protocol

For each hypothesis:

1. Commit code before benchmark run.
2. Run full suite: `uv run bench -v`.
3. Compare totals and per-family deltas vs latest baseline artifact.
4. Accept only if no correctness regressions and broad-family performance improves.
5. Record outcome in `docs/02-attempt-history.md` as worked / did not work.
