# Optimization Guide

Self-contained instructions for running the optimization loop on this quantum simulator. This is the only file you need to read before starting work.

## Philosophy

This is a **general-purpose quantum simulator**. The benchmark suite exists to quantify progress, not to define it. Solving for one special case doesn't help — any optimization must improve the simulator's general performance across diverse circuit structures, with dense output distributions (real-world circuits), not just sparse ones (textbook circuits).

The external reference backend is **Qiskit Aer**, but for the optimization loop it is treated as a fixed baseline dataset. The main question is not "did one case get faster?" but "did native improve broadly while keeping or improving completion rate vs the pinned Aer baseline?"

Think big. Read `docs/experiment-log.md` to understand what's been tried, what worked, and what failed — then think about what hasn't been tried yet. The biggest wins have historically come from rethinking the execution model, not from making existing code faster.

## Scope

- **Optimization targets**: `src/quantum/system.py` and `src/quantum/gates.py`. Do not modify benchmark cases or the user-facing API (gate constructors, `run_simulation` signature, `Circuit`/`QuantumRegister` interface).
- **Code quality**: Keep source clean and readable. Prefer structural improvements over micro-hacks. No dead code, no commented-out experiments, no special-case branches for specific benchmark cases.
- **Platform constraints**: This runs on Apple Silicon (M1 Max, 32 GB). Solutions must be callable from Python and optimized for this hardware. How you achieve that — PyTorch, native code extensions, GPU shaders, anything — is up to you.
- **Benchmark execution in scope**: run `native` only during normal optimization iterations. `aer` is a pinned reference JSONL used for comparison graphics.

## The loop

```
1. Profile    — identify where time is actually spent
2. Hypothesize — form a concrete, falsifiable prediction
3. Implement  — apply the change
4. Commit     — commit before benchmarking (creates clean audit trail)
5. Benchmark  — run full suite, evaluate correctness + timing
6. Evaluate   — compare against prior run, accept or revert
7. Record     — log outcome and refresh comparison artifacts
```

Each step in detail:

### 1. Profile

**Start here, not with a benchmark run.** The most recent native and Aer results are already in `benchmarks/results/` as JSONL files (one JSON object per line per case). Read the latest native JSONL and the pinned `benchmarks/results/aer-reference.jsonl` to understand current performance before doing anything else. Running a full benchmark takes 10+ minutes — don't waste that time until you have a change to measure.

Use `bench-trace` to identify actual bottlenecks before guessing. Be deliberate about what you profile — a cached call measures different things than a cold call.

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

### 2. Hypothesize

Before writing any code, state:
- **What** you're changing
- **Why** you expect it to help (with evidence — profiling, measurement, or architectural reasoning)
- **Which circuit families** should improve (not just one case)
- **What could go wrong** (correctness risk, regression on other families)

A hypothesis doesn't have to come from the profiler. It can come from rethinking the architecture, studying other simulators, or questioning a fundamental assumption. The best hypotheses often eliminate work entirely rather than making existing work faster.

### 2b. Research

Before implementing, use external sources to inform your approach. The goal is not just to match SOTA but to surpass it — look for techniques that haven't been tried yet. Use DeepWiki to query open-source codebases, web search for papers and documentation, and WebFetch to read specific sources.

### 3. Implement

Edit `src/quantum/system.py` and/or `src/quantum/gates.py`. Priorities:
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

`--backend` defaults to `native`. In the optimization loop, do not run `--backend aer`; use the pinned Aer reference JSONL during analysis instead.

The harness runs each circuit twice:
- **cold @1K** — compilation cache cleared before this call. Measures full cold-start cost (any preprocessing + execution + sampling at 1000 shots).
- **warm @10K** — reuses cached state from the cold call. Measures pure execution + sampling throughput at 10000 shots.

Correctness is checked at @10K (highest shot count). Cases are sorted by qubit count ascending (small/fast first). Each result is flushed to disk immediately for crash resilience.

Cases exceeding `--timeout` seconds (default 30) on any shot count are aborted — remaining shots are skipped and the case is excluded from totals.

**Output**:
- Per-case: wall time (cold and warm), CPU time, ops/sec, memory, correctness, abort metadata
- Terminal summary: totals split by cold/warm and static/dynamic, hotspot analysis for both
- File: `benchmarks/results/<timestamp>.jsonl`

### 6. Evaluate

Compare the new run against the prior baseline:
- **Correctness**: all complete cases must PASS. Any correctness failure (FAIL) is a hard blocker — revert immediately. Aborted cases (OOM or timeout) are not failures; they're excluded from totals and don't cause exit code 1. However, the long-term goal is **zero aborted cases** — every case should complete within the timeout. If a previously-passing case now gets aborted, that's a performance regression.
- **Cases complete**: compare against the prior run's `cases_complete` count. More complete cases = progress. Fewer = regression.
- **Head-to-head vs Aer (hero metric)**: compare the latest native JSONL against the pinned Aer reference JSONL. Compute per-cell ratio `aer_runtime / native_runtime` for each `(case, shot_count)` cell. Aggregate with geometric mean by shot and overall. Values `>1` mean native is faster. For aborted/missing cells, use timeout-censoring at the run timeout (default 30s) so aborts are penalized instead of silently dropped.
- **Coverage**: track the fraction of cases with a concrete runtime at each shot count for each backend. Higher coverage means fewer aborts/timeouts.

### 7. Record

After each optimization iteration, update the artifacts that drive decisions:

**a) Experiment log** — append a row to `docs/experiment-log.md` matching the existing table format. Include: idx (next sequential), commit hash, what changed, result metrics, verdict. The result metrics are cold @1K total and warm @10K total for complete cases.

**b) Native vs Aer comparison graphic** — generate `docs/native-vs-aer.png` with the committed tool:
1. Run:
   - `uv run bench-heatmap`
2. Optional explicit inputs:
   - `uv run bench-heatmap --native benchmarks/results/<native>.jsonl --reference benchmarks/results/aer-reference.jsonl --output docs/native-vs-aer.png --timeout 30`
3. The generator enforces comparison policy:
   - **Always include all test cases** (full case union plus expected-case coverage).
   - **Any aborted/missing cell and any runtime >= timeout** is treated as a fail and plotted at timeout (`30s` by default), for both backends.
   - Plot type is a **single timeout-parity scatter**:
     - x = Aer runtime, y = native runtime, log-log axes.
     - parity line `y=x`.
     - timeout boundaries at `x=30`, `y=30`.
   - Marker semantics:
     - color = qubit bucket (`<=8`, `9-16`, `17-24`, `25-30+`),
     - shape = status (`both complete`, `native fail`, `Aer fail`, `both fail`),
     - size = shot row (cold @1K vs warm @10K).
4. Read the generated image and confirm it renders correctly.
5. Only refresh the pinned Aer JSONL in a separate maintenance pass (e.g., benchmark suite/harness/environment change), not during normal optimization iterations.

**c) Progress chart** — `docs/progress-data.md` tracks cold and warm totals across optimization checkpoints.

## Interpreting results

The benchmark measures two distinct things per circuit:
- **Cold @1K**: end-to-end latency including any preprocessing, first compilation, and execution. This is what a user experiences the first time they run a circuit.
- **Warm @10K**: throughput after the first call. This is what a user experiences on repeated calls or parameter sweeps.

Both matter. Cold performance determines interactive responsiveness. Warm performance determines batch throughput.

**Abort handling**: Cases that OOM or exceed `--timeout` at any shot count break out early. Aborted cases are excluded from totals — only cases completing both shot counts are summed. Results are written incrementally to JSONL, so even if a large case causes an OS-level kill (exit 137), all prior results are preserved.

## Key files

| File | Purpose |
|------|---------|
| `src/quantum/system.py` | Simulation engine — the optimization target |
| `src/quantum/gates.py` | Gate types and circuit API — internals modifiable, public API frozen |
| `src/quantum/qasm.py` | QASM 2.0 parser |
| `benchmarks/run.py` | Benchmark harness (`bench`) |
| `benchmarks/trace.py` | Profiler (`bench-trace`) |
| `benchmarks/cases/` | Hand-coded benchmark case definitions — do not modify |
| `benchmarks/circuits/` | QASM circuit files (auto-discovered) |
| `benchmarks/expected/` | Expected distributions from Aer |
| `benchmarks/generate_circuits.py` | QASM circuit generator |
| `benchmarks/generate_expected.py` | Expected distribution generator (via Aer) |
| `docs/experiment-log.md` | Experiment log (what was tried, what worked, what didn't) |
| `docs/native-vs-aer.png` | Native-vs-Aer heatmap comparison |
| `docs/progress-data.md` | Full-suite progress chart data |

## Anti-patterns

- **Benchmark hacking**: special-casing code for a specific test case name, structure, or output distribution. The benchmark is a proxy for general performance, not the goal. If an optimization only helps circuits with sparse outputs (few distinct measurement outcomes), it's not helping real-world quantum circuits.
- **Regressing correctness**: never trade correctness for speed. The tolerance check is the hard gate.
- **Complexity without payoff**: if an optimization adds significant code complexity for <5% broad improvement, it's probably not worth it.
