# Experiment Record - 2026-02-11

## H1: Replace global gather with local-axis permutation application

Status: attempted, measured, rejected (reverted)

## Original Idea

Replace permutation-gate execution based on global `2^n` source-index gather with:

1. reshape state to `(batch, 2, ..., 2)`
2. permute targets to trailing axes
3. flatten to `(-1, 2^k)`
4. apply local-axis permutation via `index_select`
5. apply optional monomial factors
6. inverse-permute and reshape back

Rationale at the time:

- operate in local gate basis (`2^k`) instead of building global gather map
- potentially reduce full-state gather pressure

## What Was Implemented

Implemented in `src/quantum/system.py` as a replacement `_apply_permutation_gate` path, then benchmarked.

## Measured Outcome

Comparison:

- Baseline before experiment: `benchmarks/results/2026-02-10T215408.jsonl`
- Experimental run: `benchmarks/results/2026-02-10T221843.jsonl`

Regression at 10000 shots:

- `real_grovers`: `15.67s -> 22.99s` (0.68x)
- `toffoli_oracle`: `1.78s -> 2.85s` (0.62x)
- `toffoli_oracle_13`: `10.85s -> 17.55s` (0.62x)
- `qft_14`: `30.94s -> 35.01s` (0.88x)
- `phase_ladder_13`: `21.36s -> 24.33s` (0.88x)

## Decision

Rejected and reverted.

## Takeaway

On MPS, the local-axis `index_select` formulation produced worse memory/layout behavior than the global gather path for these workloads. Backend-measured behavior overruled theoretical intuition.

This record is kept to avoid retesting the same idea without a materially different execution strategy.
