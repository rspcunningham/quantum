# Optimization Guide

Self-contained instructions for running the optimization loop on this quantum simulator. This is the only file you need to read before starting work.

## Philosophy

This is a **general-purpose quantum simulator**. The benchmark suite exists to quantify progress, not to define it. Any optimization must improve general performance across diverse circuit structures with dense output distributions (real-world circuits), not just sparse ones (textbook circuits).

Think big. Read `results.tsv` to understand what's been tried, what worked, and what failed — then think about what hasn't been tried yet. The biggest wins have historically come from rethinking the execution model, not from making existing code faster.

## Scope

- **Optimization targets**: The simulation engine and native runtime. This includes `src/quantum/system.py`, `src/quantum/gates.py`, `src/quantum/metal_exec.py`, `native/src/*.cpp`, `native/src/*.mm`, `native/src/*.metal`, and `native/src/*.hpp`. Do not modify benchmark cases or the user-facing API (gate constructors, `run_simulation` signature, `Circuit`/`QuantumRegister` interface).
- **Code quality**: Keep source clean and readable. Prefer structural improvements over micro-hacks. No dead code, no commented-out experiments, no special-case branches for specific benchmark cases. No caching of simulation results or output distributions between calls — the simulator must do the work each time.
- **Platform constraints**: This runs on Apple Silicon (M3 Ultra, 64 GB). Solutions must be callable from Python and optimized for this hardware. How you achieve that — PyTorch, native code extensions, GPU shaders, anything — is up to you.
- **Rebuilding**: After editing any native C++ / Objective-C / Metal source under `native/src/`, you must rebuild before benchmarking: `uv sync --reinstall-package quantum`.

## Setup

1. **Agree on a run tag** with the user (e.g. `mar23`). The branch `optimize/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b optimize/<tag>` from current main.
3. **Read the in-scope files**: `src/quantum/system.py`, `src/quantum/gates.py`, and `results.tsv` (if it exists from a prior run) for context on what's been tried.
4. **Read the latest native JSONL** in `benchmarks/results/` and the pinned `benchmarks/results/aer-reference.jsonl` to understand current per-case performance.
5. **Initialize `results.tsv`**: create it with just the header row. The baseline will be recorded after the first benchmark.
6. **Establish baseline**: run the benchmark as-is (no code changes) and record the first row with status `keep`.
7. **Confirm and go**: confirm setup looks good, then kick off the loop.

## The loop

```
LOOP FOREVER:

1. Analyze   — read latest results, identify where time is spent
2. Hypothesize — state what you're changing and why
3. Implement — edit system.py and/or gates.py
4. Commit    — git commit (creates revert point)
5. Benchmark — uv run bench
6. Record    — read results, append to results.tsv
7. Decide    — keep (advance) or discard (git reset)
```

### 1. Analyze

Read the latest benchmark JSONL to identify where time is spent. The per-case wall times *are* your profile — which cases are slowest, which families dominate the total.

Use `bench-trace` only when you need to go deeper on a specific bottleneck:

```bash
uv run bench-trace <case_name> <shots>
uv run bench-trace random_universal_14 10000
uv run bench-trace random_universal_14 10000 --no-stack  # smaller trace
```

If 3+ consecutive experiments are discarded, stop and research before the next attempt. Use web search for papers and techniques, study other simulators, question fundamental assumptions.

### 2. Hypothesize

Before writing code, state:
- **What** you're changing
- **Why** you expect it to help (with evidence)
- **Which circuit families** should improve (not just one case)
- **What could go wrong**

The best hypotheses eliminate work entirely rather than making existing work faster.

### 3. Implement

Edit `src/quantum/system.py` and/or `src/quantum/gates.py`. Priorities:
- Structural clarity over micro-optimization
- General solutions over case-specific fixes
- Fewer full-state passes over faster individual passes

### 4. Commit

Always commit **before** running benchmarks. This is your revert point if the experiment fails.

### 5. Benchmark

```bash
uv run bench
```

The harness runs each circuit at 10K shots and measures wall time. Cases exceeding 10 seconds are aborted (via a native Metal timeout) and excluded from totals. Results are written incrementally to `benchmarks/results/<timestamp>.jsonl` — read this file directly for per-case data.

The last line of output is a machine-readable summary:

```
SUMMARY	<total_seconds>	<completed>/<total>	<fail_count> FAIL
```

To regenerate the Aer reference (not part of the loop): `uv run bench-aer`.

### 6. Record

Append a row to `results.tsv` (tab-separated). The TSV has 5 columns:

```
commit	total	cases	status	description
```

1. git commit hash (short, 7 chars)
2. total wall time in seconds for complete cases (e.g. `12.34`). Use `0.00` for crashes.
3. cases complete as fraction (e.g. `241/249`). Use `0/249` for crashes.
4. status: `keep`, `discard`, or `crash`
5. short description of what was tried

### 7. Decide

- **Any correctness FAIL** → discard, `git reset --hard HEAD~1`
- **Previously-completing case now aborts** → treat as regression, likely discard
- Otherwise, use your judgment. Consider the total time, cases complete, and what the change does to the architecture. A small improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome — that's a simplification win.

If you discard: `git reset --hard HEAD~1` and iterate.
If you keep: the branch advances and you iterate.

**NEVER STOP.** Once the loop begins, do not pause to ask if you should continue. The human may be asleep. If you run out of ideas, think harder — re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you.

## Key files

| File | Purpose |
|------|---------|
| `src/quantum/system.py` | Simulation engine — Python entry point |
| `src/quantum/gates.py` | Gate types and circuit API — internals modifiable, public API frozen |
| `src/quantum/metal_exec.py` | Metal execution bridge — Python ↔ native |
| `native/src/runtime.mm` | Native Metal runtime — GPU dispatch, sampling, timeout |
| `native/src/runtime.hpp` | Native runtime header |
| `native/src/py_module.cpp` | pybind11 bindings — circuit compilation, `run_circuit` entry |
| `native/src/shaders.metal` | Metal GPU shaders — gate kernels, sampling, histogram |
| `src/quantum/qasm.py` | QASM 2.0 parser |
| `benchmarks/run.py` | Benchmark harness (`bench`) — do not modify |
| `benchmarks/trace.py` | Profiler (`bench-trace`) |
| `benchmarks/run_aer.py` | Aer reference runner (`bench-aer`) — not part of the loop |
| `benchmarks/cases/` | Benchmark case definitions — do not modify |
| `benchmarks/circuits/` | QASM circuit files (auto-discovered) |
| `benchmarks/expected/` | Expected distributions from Aer |
| `benchmarks/results/aer-reference.jsonl` | Pinned Aer timings for comparison |
| `results.tsv` | Experiment log — created per run, not committed |

## Anti-patterns

- **Benchmark hacking**: special-casing code for a specific test case name, structure, or output distribution. If an optimization only helps circuits with sparse outputs, it's not helping real-world circuits.
- **Result caching**: caching output distributions, CDFs, or any simulation results between calls. The simulator must compute from scratch each time. Compile-time preprocessing (gate fusion, circuit simplification) is fine — runtime result caching is not.
- **Regressing correctness**: never trade correctness for speed.
- **Complexity without payoff**: if an optimization adds significant code complexity for <5% broad improvement, it's probably not worth it. Conversely, simplifying code while maintaining performance is always a win.
- **Re-trying failed ideas**: read `results.tsv` before hypothesizing. If something was tried and discarded, don't try it again unless you have a specific reason it would work differently now.
