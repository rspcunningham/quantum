# Optimization Guide

Self-contained instructions for running the optimization loop on this quantum simulator. This is the only file you need to read before starting work.

## Philosophy

This is a **general-purpose quantum simulator**. The benchmark suite exists to quantify progress, not to define it. Solving for one special case doesn't help — any optimization must improve the simulator's general performance across diverse circuit structures. Clean, DRY, modern Python is paramount; clever-but-messy code that saves a few percent is not worth the maintenance cost.

## Scope

- **Optimization target**: `src/quantum/system.py` only. Do not modify `gates.py` or benchmark cases.
- **Code quality**: Keep `system.py` clean and readable. Prefer structural improvements over micro-hacks. No dead code, no commented-out experiments, no special-case branches for specific benchmark cases.
- **Backend**: Apple Silicon MPS via PyTorch. This runs on a MacBook — MPS is the primary target, not CUDA.

## The loop

```
1. Profile    — identify where time is actually spent
2. Hypothesize — form a concrete, falsifiable prediction
3. Implement  — apply the change in src/quantum/system.py
4. Commit     — commit before benchmarking (creates clean audit trail)
5. Benchmark  — run full suite, evaluate correctness + timing
6. Evaluate   — compare against prior run, accept or revert
7. Record     — log outcome and update progress tracking
```

Each step in detail:

### 1. Profile

Use `bench-trace` to identify the actual bottleneck before guessing.

```bash
# Profile a specific case at a specific shot count
uv run bench-trace <case_name> <shots>

# Examples:
uv run bench-trace random_universal_14 1000
uv run bench-trace adaptive_feedback_5q 10000

# Smaller trace files (no stack traces):
uv run bench-trace random_universal_14 1000 --no-stack
```

Output: a `torch.profiler` table (top 20 ops by CPU time) printed to stdout, plus a Chrome/Perfetto trace JSON at `benchmarks/results/trace_<case>_<shots>.json`. Open traces at `chrome://tracing` or `https://ui.perfetto.dev`.

Focus on:
- Which `aten::*` ops dominate self CPU time
- Whether the bottleneck is compute (`mm`, `mul`) or movement (`copy_`, `to`, `_to_copy`)
- Whether the bottleneck is indexing/control flow (`nonzero`, `index_put_`, `item`)

### 2. Hypothesize

Before writing any code, state:
- **What** you're changing
- **Why** you expect it to help (linked to profiler evidence)
- **Which circuit families** should improve (not just one case)
- **What could go wrong** (correctness risk, regression on other families)

Bad hypothesis: "Make `adaptive_feedback_5q` faster by caching its specific branch pattern."
Good hypothesis: "Replacing per-measurement `nonzero` calls with multiplicative masking should reduce dynamic-circuit overhead across all feedback workloads, because profiler shows `nonzero` at 73% of CPU time in the dynamic path."

### 3. Implement

Edit `src/quantum/system.py`. Priorities:
- Structural clarity over micro-optimization
- General solutions over case-specific fixes
- Fewer full-state passes over faster individual passes

### 4. Commit

Always commit **before** running benchmarks. This creates a 1:1 mapping between code states and benchmark artifacts for reproducibility and bisection.

### 5. Benchmark

```bash
# Full suite with per-case detail (primary command)
uv run bench -v

# Run specific cases only (for quick iteration during development)
uv run bench -v --cases real_grovers qft adaptive_feedback_5q
```

The harness runs each case at shot counts `[1, 10, 100, 1000, 10000]` and checks correctness at the highest completed count against expected output distributions.

**Output**:
- Per-case: wall time, CPU time, ops/sec, memory, correctness (PASS/FAIL)
- Summary: totals split by static/dynamic, hotspot analysis
- Files: `benchmarks/results/<timestamp>.jsonl` + `.summary.json`

### 6. Evaluate

Compare the new run against the prior baseline:
- **Correctness**: all 22 cases must PASS. A faster but incorrect simulator is useless.
- **Broad improvement**: check static totals AND dynamic totals at both @1000 and @10000. An optimization that helps one family but regresses another is suspect.
- **Shot scaling**: compare @1000 vs @10000 for static circuits. If they scale linearly, unitary evolution is leaking into the shot loop (an algorithmic bug, not a micro-optimization problem).

For SOTA comparison against Qiskit Aer and Google qsim:

```bash
# Full suite: native vs Aer (3 reps, median reported)
uv run bench-compare -v

# Static-only: native vs Aer vs qsim
uv run bench-compare --suite static --backends native aer qsim -v

# Specific cases only
uv run bench-compare --cases bell_state qft -v

# Generate markdown report from a compare JSONL
uv run bench-compare-report benchmarks/results/compare-<timestamp>.jsonl
```

### 7. Record

After each iteration, update three things:

**a) Attempt history** — append a row to `docs/02-attempt-history.md` following the existing ledger format. Include: commit hash, hypothesis, measured result, verdict (worked / did not work), evidence artifact paths.

**b) Progress data table** — if the iteration was successful (worked), append a row to `docs/progress-data.md` with the core-6 totals from the new benchmark run. The core-6 cases are: `bell_state`, `simple_grovers`, `real_grovers`, `ghz_state`, `qft`, `teleportation`. Extract their per-shot-count totals from the JSONL and add a new row. Update the `annotation` on the previous "Current" row to blank and mark the new row as "Current".

**c) Progress chart** — first, **read the existing `docs/images/progress.png`** to see what the current chart looks like. Then regenerate it from the updated `docs/progress-data.md`. Write a one-off Python script that reads the table, plots the series (log-scale Y, one line per shot count), and saves the PNG. Use `matplotlib` (available in the project venv). Don't commit the script — just run it ephemerally and commit the resulting image. After generating, **read the new image** to verify it looks correct and to inform your next hypothesis — the shape of the curves tells you where the remaining headroom is.

## Interpreting results

**Shot counts** `[1, 10, 100, 1000, 10000]` surface different behaviors:
- Low shots (1, 10): dominated by fixed overhead (compilation, device transfer, warmup)
- High shots (1000, 10000): dominated by per-shot or per-evolution costs

**Static vs dynamic**: the harness classifies circuits automatically:
- **Static**: no conditionals, terminal-only measurements → eligible for evolve-once/sample-many fast path
- **Dynamic**: mid-circuit measurements and/or conditional gates → requires branch-based execution

**Known failures**: `ghz_state_16` and `ghz_state_18` fail on MPS due to a tensor rank limit (MPS supports rank ≤ 16). These are backend-limit failures, not correctness bugs.

## Key files

| File | Purpose |
|------|---------|
| `src/quantum/system.py` | Simulation engine — the optimization target |
| `src/quantum/gates.py` | Gate types and circuit API — do not modify |
| `benchmarks/run.py` | Benchmark harness (`bench`) |
| `benchmarks/trace.py` | Profiler (`bench-trace`) |
| `benchmarks/compare.py` | SOTA comparison (`bench-compare`) |
| `benchmarks/cases/` | Benchmark case definitions — do not modify |
| `docs/02-attempt-history.md` | Experiment log (what was tried, what worked, what didn't) |
| `docs/progress-data.md` | Raw data for the progress chart |

## Anti-patterns

- **Benchmark hacking**: special-casing code for a specific test case name or structure. The benchmark is a proxy for general performance, not the goal.
- **Premature micro-optimization**: tuning constants or unrolling loops before fixing algorithmic issues (e.g., shot-scaled evolution).
- **Guessing without profiling**: always run `bench-trace` before hypothesizing. Intuition about GPU bottlenecks is frequently wrong.
- **Regressing correctness**: never trade correctness for speed. The tolerance check is the hard gate.
- **Complexity without payoff**: if an optimization adds significant code complexity for <5% broad improvement, it's probably not worth it.
