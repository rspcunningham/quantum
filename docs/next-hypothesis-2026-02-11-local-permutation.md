# Next Hypothesis - 2026-02-11

## H1: Replace global gather with local-axis permutation application

Status: design only (not implemented)

## Context

Latest commit with permutation fast path: `e1fa5b0`

Latest benchmark comparison:
- Baseline: `benchmarks/results/2026-02-10T210225.jsonl`
- Current: `benchmarks/results/2026-02-10T215408.jsonl`

Observed impact after current permutation path:
- Major wins on permutation-heavy cases (`real_grovers`, `toffoli_oracle`, `toffoli_oracle_13`)
- New top profiler cost is transfer/copy attribution (`aten::copy_` / `aten::to`) rather than math (`aten::mm`)

Profiler evidence:
- `benchmarks/results/trace_real_grovers_1000.json`
- `benchmarks/results/trace_toffoli_oracle_13_1000.json`

## Problem statement

Current permutation execution in `BatchedQuantumSystem._apply_permutation_gate` uses a full-state gather:
1. Build/cache a global source index tensor of shape `(2^n,)`
2. Reindex full state as `state_vectors[:, source_indices]`
3. Apply optional phase factors via a second full-length lookup

This is correct and much faster than dense matmul, but it likely creates heavy memory traffic on the full state vector and is consistent with copy-dominated traces.

## Hypothesis

Applying permutation gates in local gate basis (size `2^k`) after axis reordering will reduce full-state gather/copy traffic and further improve permutation-heavy benchmarks.

## Proposed algorithm (no code yet)

For a gate with `k` targets in an `n`-qubit state:
1. Reshape: `(batch, 2^n)` -> `(batch, 2, ..., 2)`
2. Permute targets to the trailing axes (same strategy as dense path)
3. Flatten: `(-1, 2^k)`
4. Reindex only the local axis using gate permutation:
   - `flat = flat.index_select(1, local_source_index)` or equivalent gather
5. If monomial factors exist, apply factors on local axis:
   - `flat *= local_factors.unsqueeze(0)`
6. Reshape and inverse-permute back to `(batch, 2^n)`

Why this should help:
- Reindexing a local axis of width `2^k` avoids constructing/applying a global `2^n` gather map
- For common gates (`X`, `CX`, `CCX`), `k` is small (1-3), so the permutation work stays compact
- Keeps dispatch structure aligned with existing dense path, reducing special-case surface area

## Expected impact

Primary targets:
- `real_grovers`
- `toffoli_oracle`
- `toffoli_oracle_13`

Secondary effect:
- Neutral to slight improvement on mixed suites (`qft*`, `phase_ladder*`), since these are already dominated by diagonal path

Quantitative expectation (at 10000 shots):
- `real_grovers`: additional 1.2x-1.6x
- `toffoli_oracle*`: additional 1.2x-1.6x
- Full-suite total: additional 1.1x-1.3x

## Risks and constraints

1. Backend behavior risk (MPS)
- `index_select`/gather on non-contiguous layouts can still trigger hidden copies.
- Need to verify with profiler instead of assuming.

2. Correctness risk in target-bit ordering
- Must preserve the same target significance convention already used by diagonal and dense paths.

3. Small-case overhead
- Extra `permute`/reshape overhead could hurt tiny circuits if not carefully dispatched.

## Validation plan

Correctness:
1. Run full benchmark correctness checks (`uv run bench -v`).
2. Add targeted equivalence checks for representative gates: `X`, `CX`, `CCX`, and a monomial-with-phase case.

Performance A/B:
1. Compare new run vs `benchmarks/results/2026-02-10T215408.jsonl`.
2. Focus metrics:
   - per-case `times_s["10000"]` for permutation-heavy cases
   - suite totals at `1000` and `10000`
3. Re-profile `real_grovers 1000` and `toffoli_oracle_13 1000` and confirm copy-dominance decreases.

Acceptance criteria:
- No correctness regressions on passing cases
- >=15% speedup on at least two permutation-heavy cases at 10000 shots
- No >5% regressions on diagonal-dominant cases (`qft*`, `phase_ladder*`)

## Fallback if hypothesis fails

If local-axis permutation does not lower copy overhead:
1. Keep current global-gather path for permutation gates.
2. Move to buffer reuse/ping-pong allocation strategy to reduce allocation/copy churn.
3. Evaluate circuit-level precompiled execution plans to reduce runtime dispatch overhead.
