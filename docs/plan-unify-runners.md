# Plan: Migrate To Single `bench` Runner With `--backend`

## Goal

Keep one benchmark CLI: `bench` (`benchmarks/run.py`), with a configurable backend:

- `--backend native` (default)
- `--backend aer`

We only care about per-case:

- runtime by shot count
- CPU time / CPU utilization
- memory metadata
- status (`PASS`, `FAIL`, `ABORT`) and errors

`bench-compare` should be removed after migration.

## Non-Goals

- No multi-backend run in a single command.
- No median-of-reps feature (`--reps`) in this migration.
- No compare-style markdown report tooling.

## Target CLI

```bash
uv run bench -v
uv run bench -v --backend native
uv run bench -v --backend aer
uv run bench --core -v --backend aer
uv run bench --cases qft real_grovers --backend aer
```

## Target JSONL Contract

Current `bench` JSONL schema stays intact, plus:

- `backend`: `"native"` or `"aer"`

All existing fields used by optimization workflow remain unchanged (`times_s`, `cpu_times_s`, `cpu_util`, `memory`, `correct`, `aborted`, `abort_reason`, `errors`, etc.).

## Implementation Plan

1. Add backend flag to `bench`
   - In `benchmarks/run.py`, add `--backend` with `choices=["native", "aer"]`, default `"native"`.
   - Print selected backend in run header.

2. Introduce backend runner abstraction inside `run.py`
   - Keep current native path behavior unchanged.
   - Add Aer execution path that mirrors native instrumentation:
     - warmup call
     - per-shot timing (`time.perf_counter`)
     - per-shot CPU timing (`time.process_time`)
     - timeout/abort behavior
     - correctness check at highest completed shot
   - Use existing Aer adapter (`benchmarks/backends/aer_adapter.py`) for circuit prep + execution.

3. Preserve status behavior and resilience
   - Keep OOM handling for native path.
   - For Aer path, treat runtime exceptions as `ABORT` with explicit `abort_reason`.
   - Keep incremental JSONL writes (flush each case).
   - Keep case sorting by qubits and existing summary generation.

4. Keep memory metadata for both backends
   - Native: existing device memory stats.
   - Aer: record per-case process memory telemetry (before/after RSS, delta, and process peak RSS).
   - Also record process memory telemetry for native path to maximize comparability.
   - Ensure schema consistency so downstream parsing is stable.

5. Remove `bench-compare` surface area
   - Remove `bench-compare` entry point from `pyproject.toml`.
   - Delete `benchmarks/compare.py`.
   - Remove `bench-compare-report` entry point and delete `benchmarks/compare_report.py`.
   - Remove qsim support code for now:
     - delete `benchmarks/backends/qsim_adapter.py`
     - remove qsim from `benchmarks/backends/__init__.py` registry
     - remove qsim dependency references from docs and dependency groups where not needed
   - Update docs (`README.md`, `OPTIMIZE.md`) to use `bench --backend aer` for Aer runs.

6. Compatibility and validation
   - Run smoke tests:
     - `uv run bench --cases bell_state --backend native -v`
     - `uv run bench --cases bell_state --backend aer -v`
     - `uv run bench --core --backend native`
   - Verify JSONL rows include backend field and expected status/metrics fields.
   - Verify progress workflow metrics can be computed directly from JSONL.

## Acceptance Criteria

- `bench` runs successfully with both `--backend native` and `--backend aer`.
- Output retains required per-case metrics: wall time, CPU, memory, status, errors.
- Incremental JSONL output remains functional.
- `bench-compare` command no longer exists.
- qsim is removed from active benchmark tooling.
- README/OPTIMIZE docs no longer reference `bench-compare`.

## Migration Order

1. Implement `--backend` in `benchmarks/run.py`.
2. Validate outputs and summary compatibility.
3. Remove compare CLI/files and update docs.
4. Final smoke run for native + aer.

## Status

Implemented (2026-02-11).
