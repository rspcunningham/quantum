# Plan: Unify bench and bench-compare

## Problem

`bench` (run.py) and `bench-compare` (compare.py) are two separate runners that do fundamentally the same thing — run benchmark cases at multiple shot counts and record timings. They diverged in features:

| Feature | `bench` | `bench-compare` |
|---|---|---|
| Backends | native only | native + aer + qsim (pluggable) |
| Reps per shot | 1 | 3 (median reported) |
| OOM recovery | yes | no |
| Timeout (`--timeout`) | yes (30s default) | no |
| Incremental JSONL | yes | no (writes all at end) |
| Qubit-sorted execution | yes | no |
| Hotspot analysis | yes | no |
| Correctness check | at max shots | at max shots |
| Backend abstraction | calls `run_simulation()` directly | goes through `BackendAdapter` IR layer |

## Proposal

Add `--backend` flag to `bench`. Default: `native`. When `--backend aer` or `--backend native aer` is passed, use the `BackendAdapter` layer from compare.py. Deprecate `bench-compare` as a separate entry point.

### Unified interface

```bash
uv run bench -v                              # native, full suite, 1 rep
uv run bench -v --backend native aer         # native + aer side-by-side
uv run bench -v --backend aer --reps 3       # aer only, 3 reps (median)
uv run bench --core -v --backend native aer  # core-6, both backends
```

### Implementation

1. Move `BackendAdapter` integration into `run.py`:
   - When backend is `native` (default): use current `run_case()` path (direct `run_simulation()`)
   - When backend is non-native or multiple: use `BackendAdapter.prepare()` + `BackendAdapter.run()`
   - Apply timeout, OOM recovery, incremental writes to all backends

2. Add `--reps` flag (default 1, used for median when >1)

3. Keep `BackendAdapter` abstraction in `benchmarks/backends/` unchanged

4. Deprecate `bench-compare` entry point (keep as thin wrapper that prints deprecation notice and calls `bench --backend native aer --reps 3`)

5. Update `bench-compare-report` to work with the unified JSONL format

### Output format

Unified JSONL rows include a `backend` field. When multiple backends are used, each case produces one row per backend. Summary totals are printed per-backend.

### Migration

- Existing `bench-compare` JSONL files remain readable (already have `backend` field)
- `bench` JSONL gains a `backend: "native"` field for consistency
- Progress tracking (`docs/progress-data.md`) continues to use native-backend totals only

## Status

Not started. Lower priority than simulation optimization work.
