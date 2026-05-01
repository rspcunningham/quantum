# Plan: Repo-Local Benchmark Runners

## Goal

Keep benchmark tooling in this repository, but outside the installable `quantum`
package. The simulator wheel should not install benchmark modules, benchmark
commands, or Qiskit/Aer dependencies.

## Commands

```bash
uv run --group bench python -m benchmarks.run
uv run --group bench python -m benchmarks.run_aer
uv run --group bench python -m benchmarks.trace <case_name> <shots>
```

## Current Runner Shape

- `benchmarks/run.py` runs the native simulator suite.
- `benchmarks/run_aer.py` runs the Aer reference suite.
- `benchmarks/trace.py` profiles one benchmark case.
- Benchmark scripts are internal repo tooling and are not public package API.

## Packaging Boundary

- `src/quantum` is the only packaged Python module.
- `benchmarks/` stays in the repo checkout for development workflows.
- Benchmark-only dependencies live in the `bench` dependency group.

## Validation

- Simulator tests should pass without the `bench` dependency group.
- Repo-local benchmark imports should work with `uv run --group bench`.
- Built wheels should not contain `benchmarks/`.
