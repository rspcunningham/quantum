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
6. Terminal-measurement sampling fast path (single-shot evolution + multi-shot sampling for eligible static circuits).

Attempted and rejected:

1. Local-axis permutation `index_select` replacement for global gather.
Reason: significant regressions on MPS; reverted.

## Current Bottleneck View

From `benchmarks/results/2026-02-10T234124.jsonl`:

- Total @1000 is `4.76s` (from `60.33s`, `12.67x` faster).
- Total @10000 is `5.23s` (from `592.91s`, `113.35x` faster).
- 22/22 runnable cases remain correct (same known MPS rank-limit failures on `ghz_state_16` and `ghz_state_18`).

Post-H0 time concentration at 10000 shots:

- `reversible_mix_15`: `1.45s` (`27.8%`)
- `adaptive_feedback_5q`: `1.26s` (`24.0%`)
- `adaptive_feedback_120`: `0.87s` (`16.7%`)
- `adaptive_feedback`: `0.32s` (`6.1%`)

Implications:

- Shot-scaled static replay is no longer the dominant issue.
- Remaining hotspots are:
  - one-shot heavy unitary execution overhead (notably deep 15-qubit reversible mix)
  - dynamic-feedback conditional/indexing overhead
  - persistent backend rank-limit coverage gap

## Next Hypotheses (Not Implemented Yet)

### H1. Execution-Plan Compilation (Circuit IR)

Compile a circuit into reusable execution segments and artifacts:

- segment by measurement/conditional boundaries
- precompute operation metadata, device tensors, index maps, and dispatch kernel choice
- reduce runtime Python branching and repeated cache lookups

Expected impact: broad, moderate-to-high improvement, especially for deep one-shot unitary cases where index-map/setup overhead is now visible.

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

Expected impact: moderate on remaining dense/permutation-heavy one-shot hotspots.

### H4. Dynamic-Circuit Conditional Path Rewrite

Target adaptive-feedback bottlenecks:

- avoid scalar sync in condition checks (`.item()` / Python bool hot path)
- avoid expensive boolean advanced indexing restore patterns
- investigate branch-group/shot-branching style execution for repeated feedback loops

Expected impact: moderate-to-high on feedback workloads, which are now a large share of post-H0 totals.

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
