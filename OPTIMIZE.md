# Optimization Guide

Self-contained instructions for running the optimization loop on this quantum simulator. This is the only file you need to read before starting work.

## Philosophy

This is a **general-purpose quantum simulator**. The benchmark suite exists to quantify progress, not to define it. Solving for one special case doesn't help - any optimization must improve the simulator's general performance across diverse circuit structures. Clean, DRY, modern Python is paramount; clever-but-messy code that saves a few percent is not worth the maintenance cost.

The external reference backend is **Qiskit Aer**, but for the optimization loop it is treated as a fixed baseline dataset. The main question is not "did one case get faster?" but "did native improve broadly while keeping or improving completion rate vs the pinned Aer baseline?"

## Scope

- **Optimization targets**: `src/quantum/system.py` and `src/quantum/gates.py`. Do not modify benchmark cases or the user-facing API (gate constructors, `run_simulation` signature, `Circuit`/`QuantumRegister` interface).
- **Code quality**: Keep source clean and readable. Prefer structural improvements over micro-hacks. No dead code, no commented-out experiments, no special-case branches for specific benchmark cases.
- **Backend**: Apple Silicon MPS via PyTorch. This runs on a MacBook - MPS is the primary target, not CUDA.
- **Benchmark execution in scope**: run `native` only during normal optimization iterations. `aer` is a pinned reference JSONL used for comparison graphics. qsim is intentionally out of scope for now.

## The loop

```
1. Profile    — identify where time is actually spent
2. Hypothesize — form a concrete, falsifiable prediction
3. Implement  — apply the change in src/quantum/system.py
4. Commit     — commit before benchmarking (creates clean audit trail)
5. Benchmark  — run full suite, evaluate correctness + timing
6. Evaluate   — compare against prior run, accept or revert
7. Record     — log outcome and refresh comparison artifacts
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
# How does the backend we're targeting actually work?
deepwiki ask pytorch/pytorch "How does MPS dispatch element-wise ops vs gather/scatter?"

```

Key repos to query:
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

# Custom timeout (default 30s per case per shot count)
uv run bench -v --timeout 60

# Run specific cases only (for quick iteration during development)
uv run bench -v --cases real_grovers qft adaptive_feedback_5q
```

`--backend` defaults to `native`. In the optimization loop, do not run `--backend aer`; use the pinned Aer reference JSONL during analysis instead. The harness runs each case at shot counts `[1, 10, 100, 1000, 10000]` and checks correctness at the highest completed count against expected output distributions. Cases are sorted by qubit count ascending (small/fast first). Each result is flushed to disk immediately for crash resilience.

Cases exceeding `--timeout` seconds (default 30) on any shot count are aborted — remaining shots are skipped and the case is excluded from totals. This prevents 20-24q cases from dominating runtime.

**Output**:
- Per-case: wall time, CPU time, ops/sec, memory, correctness (`correct` + `errors`), abort metadata
- Terminal summary: totals for complete cases only (aborted/OOM cases excluded), split by static/dynamic, hotspot analysis
- File: `benchmarks/results/<timestamp>.jsonl`

### 6. Evaluate

Compare the new run against the prior baseline:
- **Correctness**: all complete cases must PASS. Any correctness failure (FAIL) is a hard blocker — revert immediately. Aborted cases (OOM or timeout) are not failures; they're excluded from totals and don't cause exit code 1. However, the long-term goal is **zero aborted cases** — every case should complete within the timeout. If a previously-passing case now gets aborted, that's a performance regression.
- **Broad improvement**: check static totals AND dynamic totals at both @1000 and @10000. An optimization that helps one family but regresses another is suspect.
- **Shot scaling**: compare @1000 vs @10000 for static circuits. If they scale linearly, unitary evolution is leaking into the shot loop (an algorithmic bug, not a micro-optimization problem).
- **Cases complete**: compare against the prior run's `cases_complete` count. More complete cases = progress. Fewer = regression.
- **Head-to-head vs Aer (hero metric)**: compare the latest native JSONL against the pinned Aer reference JSONL. Compute per-cell ratio `aer_runtime / native_runtime` for each `(case, shots)` cell. Aggregate with geometric mean by shot and overall. Values `>1` mean native is faster. For aborted/missing shot cells, use timeout-censoring at the run timeout (default 30s) so aborts are penalized instead of silently dropped.
- **Coverage**: track the fraction of cases with a concrete runtime at each shot count for each backend. Higher coverage means fewer aborts/timeouts.

### 7. Record

After each optimization iteration, update the artifacts that drive decisions:

**a) Experiment log** - append a row to `docs/experiment-log.md` matching the existing table format. Include: idx (next sequential), commit hash, what changed, result metric, verdict. The result metric is the full-suite total @1000 for complete cases (format: `Xs (N cases)`).

**b) Native vs Aer comparison graphic (required for README narrative)**:
1. Run full suite for native only (`uv run bench -v`) and use the latest native JSONL.
2. Use the pinned Aer reference JSONL: `benchmarks/results/2026-02-11T211659.jsonl`.
3. Read both JSONL files directly.
4. Generate `docs/native-vs-aer.png` with one-off, ephemeral analysis code. This is a single heatmap figure (landscape, ~16:9):
   - **X-axis**: test circuits (sorted by qubit count, no tick labels, labeled "test circuit").
   - **Y-axis**: shot counts (@1, @10, @100, @1K, @10K).
   - **Color**: `log2(aer_time / native_time)` with a diverging scale (green = faster than SOTA, red = Aer faster).
   - **Title**: overall geometric mean ratio.
5. Read the image after generation and verify it renders correctly.
6. Update the README section that explains the graph with concrete numbers from the latest pair of JSONLs.
7. Only refresh the pinned Aer JSONL in a separate maintenance pass (e.g., benchmark suite/harness/environment change), not in normal optimization iterations.

Do not commit helper scripts for this. Keep the analysis ephemeral and data-driven.

**c) Progress chart (legacy tracking, optional)** - `docs/progress-data.md` / `docs/progress.png` remain useful for historical trend lines across optimization checkpoints, but they are not the primary Aer comparison artifact.

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
| `benchmarks/cases/` | Hand-coded benchmark case definitions — do not modify |
| `benchmarks/circuits/` | QASM circuit files (auto-discovered) |
| `benchmarks/expected/` | Expected distributions from Aer |
| `benchmarks/generate_circuits.py` | QASM circuit generator |
| `benchmarks/generate_expected.py` | Expected distribution generator (via Aer) |
| `docs/experiment-log.md` | Experiment log (what was tried, what worked, what didn't) |
| `docs/native-vs-aer.png` | Native-vs-Aer heatmap comparison |
| `docs/progress-data.md` | Full-suite progress chart data |
| `docs/progress-data-core.md` | Core-6 progress chart data (legacy, saturated) |

## Anti-patterns

- **Benchmark hacking**: special-casing code for a specific test case name or structure. The benchmark is a proxy for general performance, not the goal.
- **Premature micro-optimization**: tuning constants or unrolling loops before fixing algorithmic issues (e.g., shot-scaled evolution).
- **Guessing without profiling**: always run `bench-trace` before hypothesizing. Intuition about GPU bottlenecks is frequently wrong.
- **Regressing correctness**: never trade correctness for speed. The tolerance check is the hard gate.
- **Complexity without payoff**: if an optimization adds significant code complexity for <5% broad improvement, it's probably not worth it.
