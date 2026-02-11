# Optimization Guide

Self-contained instructions for running the optimization loop on this quantum simulator. This is the only file you need to read before starting work.

## Philosophy

This is a **general-purpose quantum simulator**. The benchmark suite exists to quantify progress, not to define it. Solving for one special case doesn't help — any optimization must improve the simulator's general performance across diverse circuit structures. Clean, DRY, modern Python is paramount; clever-but-messy code that saves a few percent is not worth the maintenance cost.

## Scope

- **Optimization targets**: `src/quantum/system.py` and `src/quantum/gates.py`. Do not modify benchmark cases or the user-facing API (gate constructors, `run_simulation` signature, `Circuit`/`QuantumRegister` interface).
- **Code quality**: Keep source clean and readable. Prefer structural improvements over micro-hacks. No dead code, no commented-out experiments, no special-case branches for specific benchmark cases.
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

### 2b. Research

Before implementing, use external sources to inform your approach. The goal is not just to match SOTA but to surpass it — look for techniques that haven't been tried yet.

**DeepWiki** — query open-source codebases for design patterns and implementation details:

```
# How does the SOTA target implement something?
deepwiki ask Qiskit/qiskit-aer "How does Aer's gate fusion work? What is fusion_max_qubit?"

# How does the backend we're targeting actually work?
deepwiki ask pytorch/pytorch "How does MPS dispatch element-wise ops vs gather/scatter?"

# What does another fast simulator do differently?
deepwiki ask quantumlib/qsim "How does qsim apply 2-qubit gates to the statevector?"
```

Key repos to query:
- **`Qiskit/qiskit-aer`** — our SOTA comparison target. Gate fusion (up to 5q), AVX2 SIMD kernels, OpenMP parallelization, interleaved real/imag memory layout.
- **`quantumlib/qsim`** — Google's simulator. Aggressive gate fusion, AVX/FMA vectorization, multi-threaded statevector updates.
- **`pytorch/pytorch`** — MPS backend internals. How Metal kernels are dispatched, gather/scatter implementation, threshold-based dispatch between custom Metal kernels vs MPSGraph.

**Web search** — find papers, docs, and source code for novel techniques:

```
# Academic papers on simulation techniques
WebSearch "quantum statevector simulator gate fusion optimization 2025 2026"

# PyTorch MPS performance details not in DeepWiki
WebSearch "pytorch MPS backend Metal kernel dispatch performance"

# Pull and read a specific paper or source file
WebFetch <url> "Extract the key optimization techniques described..."
```

Known leads from prior research:
- **DiaQ** (arxiv 2405.01250): sparse diagonal format for statevector simulation. Most gates touch only a few diagonals of the full unitary. `O(d*N)` instead of dense matmul. 40-69% speedups reported. Our stride-based slicing is a hand-rolled version of this for 1q/2q — the DiaQ framework suggests generalizing further.
- **QMin** (Springer 2025): cost-aware fusion — deciding *when* fusion helps vs hurts based on gate count and qubit count.
- **BQSim** (ASPLOS 2025): decision-diagram based batch simulation exploiting gate matrix regularity/sparsity.
- **MPS gather/scatter**: implemented as custom Metal kernels with 1 thread per element via `mtl_dispatch1DJob`. No explicit WAR hazard tracking — relies on Metal's automatic tracking, which doesn't catch intra-buffer dependencies (explains F6 failure).

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

# Core-6 only (legacy — saturated at Aer parity)
uv run bench --core -v

# Custom timeout (default 30s per case per shot count)
uv run bench -v --timeout 60

# Run specific cases only (for quick iteration during development)
uv run bench -v --cases real_grovers qft adaptive_feedback_5q
```

The harness runs each case at shot counts `[1, 10, 100, 1000, 10000]` and checks correctness at the highest completed count against expected output distributions. Cases are sorted by qubit count ascending (small/fast first). Each result is flushed to disk immediately for crash resilience.

Cases exceeding `--timeout` seconds (default 30) on any shot count are aborted — remaining shots are skipped and the case is excluded from totals. This prevents 20-24q cases from dominating runtime.

**Output**:
- Per-case: wall time, CPU time, ops/sec, memory, correctness (PASS/FAIL/ABORT)
- Summary: totals for complete cases only (aborted/OOM cases excluded), split by static/dynamic, hotspot analysis
- Files: `benchmarks/results/<timestamp>.jsonl` + `.summary.json`

### 6. Evaluate

Compare the new run against the prior baseline:
- **Correctness**: all cases must PASS. A faster but incorrect simulator is useless.
- **Broad improvement**: check static totals AND dynamic totals at both @1000 and @10000. An optimization that helps one family but regresses another is suspect.
- **Shot scaling**: compare @1000 vs @10000 for static circuits. If they scale linearly, unitary evolution is leaking into the shot loop (an algorithmic bug, not a micro-optimization problem).

For SOTA comparison against Qiskit Aer and Google qsim:

```bash
# Full suite: native vs Aer (3 reps, median reported)
uv run bench-compare -v

# Core-6 only
uv run bench-compare --core -v

# Static-only: native vs Aer vs qsim
uv run bench-compare --suite static --backends native aer qsim -v

# Specific cases only
uv run bench-compare --cases bell_state qft -v

# Generate markdown report from a compare JSONL
uv run bench-compare-report benchmarks/results/compare-<timestamp>.jsonl
```

### 7. Record

After each iteration, update three things:

**a) Experiment log** — append a row to `docs/experiment-log.md` matching the existing table format. Include: idx (next sequential), commit hash, what changed, result metric, verdict. The result metric is the full-suite total @1000 for complete cases (format: `Xs (N cases)`).

**b) Progress data table** — if the iteration was successful (worked), append a row to `docs/progress-data.md` with the full-suite totals from the new benchmark run. Only include cases that completed all 5 shot counts in the totals. Cases that OOM are excluded. Record the `cases_complete` count.

For core-6 tracking (legacy, saturated): the core-6 data lives in `docs/progress-data-core.md`. The core-6 cases are: `bell_state`, `simple_grovers`, `real_grovers`, `ghz_state`, `qft`, `teleportation`.

**c) Progress chart** — first, **read the existing `docs/progress.png`** to see what the current chart looks like. Then regenerate it from the updated `docs/progress-data.md`. Write a one-off Python script that reads the table, plots the series (log-scale Y, one line per shot count), and saves the PNG. Use `matplotlib` (available in the project venv). Don't commit the script — just run it ephemerally and commit the resulting image. After generating, **read the new image** to verify it looks correct and to inform your next hypothesis — the shape of the curves tells you where the remaining headroom is.

The chart must include **SOTA reference lines** from the "SOTA Reference — Qiskit Aer (Full Suite)" section in `docs/progress-data.md`. For each shot count that has a non-null `aer_total_s` value, plot a horizontal dashed line at that Y value spanning the full X range. Use the same color as the corresponding shot-count series line, with `linestyle='--'`, `alpha=0.4`, `linewidth=1`. Do **not** add the SOTA lines to the legend — instead, place a single italic "Aer" text annotation at the right edge of the plot, vertically centered on the geometric mean of the Aer values (using `ax.annotate` with `xycoords=('axes fraction', 'data')`). If all Aer values are `null`, skip the SOTA lines silently.

## Interpreting results

**Shot counts** `[1, 10, 100, 1000, 10000]` surface different behaviors:
- Low shots (1, 10): dominated by fixed overhead (compilation, device transfer, warmup)
- High shots (1000, 10000): dominated by per-shot or per-evolution costs

**Static vs dynamic**: the harness classifies circuits automatically:
- **Static**: no conditionals, terminal-only measurements → eligible for evolve-once/sample-many fast path
- **Dynamic**: mid-circuit measurements and/or conditional gates → requires branch-based execution

**Known failures**: `ghz_state_16` and `ghz_state_18` fail on MPS due to a tensor rank limit (MPS supports rank ≤ 16). These are backend-limit failures, not correctness bugs.

**Abort handling**: Cases that OOM or exceed `--timeout` at any shot count break out of the shot ladder early. Aborted cases are excluded from totals — only cases completing all 5 shot counts are summed. Results are written incrementally to JSONL, so even if a large case causes an OS-level kill (exit 137), all prior results are preserved.

## Key files

| File | Purpose |
|------|---------|
| `src/quantum/system.py` | Simulation engine — the optimization target |
| `src/quantum/gates.py` | Gate types and circuit API — internals modifiable, public API frozen |
| `src/quantum/qasm.py` | QASM 2.0 parser |
| `benchmarks/run.py` | Benchmark harness (`bench`) |
| `benchmarks/trace.py` | Profiler (`bench-trace`) |
| `benchmarks/compare.py` | SOTA comparison (`bench-compare`) |
| `benchmarks/cases/` | Hand-coded benchmark case definitions — do not modify |
| `benchmarks/circuits/` | QASM circuit files (auto-discovered) |
| `benchmarks/expected/` | Expected distributions from Aer |
| `benchmarks/generate_circuits.py` | QASM circuit generator |
| `benchmarks/generate_expected.py` | Expected distribution generator (via Aer) |
| `docs/experiment-log.md` | Experiment log (what was tried, what worked, what didn't) |
| `docs/progress-data.md` | Full-suite progress chart data |
| `docs/progress-data-core.md` | Core-6 progress chart data (legacy, saturated) |

## Anti-patterns

- **Benchmark hacking**: special-casing code for a specific test case name or structure. The benchmark is a proxy for general performance, not the goal.
- **Premature micro-optimization**: tuning constants or unrolling loops before fixing algorithmic issues (e.g., shot-scaled evolution).
- **Guessing without profiling**: always run `bench-trace` before hypothesizing. Intuition about GPU bottlenecks is frequently wrong.
- **Regressing correctness**: never trade correctness for speed. The tolerance check is the hard gate.
- **Complexity without payoff**: if an optimization adds significant code complexity for <5% broad improvement, it's probably not worth it.
