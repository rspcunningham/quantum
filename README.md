# Quantum Simulator

A state-vector quantum circuit simulator built from scratch with PyTorch, targeting Apple Silicon (MPS) as a first-class backend.

## Why this exists

1. **There isn't a good MPS-native quantum simulator.** Qiskit and Cirq target CUDA. This project aims to build a performant simulator that runs well on Apple GPUs via PyTorch's MPS backend.

2. **This is a testbed for coding-agent-driven optimization.** The simulator ships with a benchmark harness, and the development workflow uses Claude Code to iteratively propose, implement, and validate performance improvements. The goal is to see how far a coding agent can push the performance of a real codebase through repeated optimize-measure-evaluate cycles.

## Architecture

**API layer** (`src/quantum/gates.py`) — circuit-building primitives. Gates (`H`, `X`, `CX`, etc.), parametric gates (`RX`, `RY`, `RZ`), arbitrary controlled gates via `ControlledGateType`, measurements, conditional gates, quantum registers, and `Circuit` composition with `+`, `*`, and `.inverse()`.

**Simulation layer** (`src/quantum/system.py`) — executes circuits against state vectors. `BatchedQuantumSystem` runs many shots in parallel as a single `(batch_size, 2^n)` tensor. `run_simulation()` is the main entry point.

```python
from quantum import QuantumRegister, H, CX, run_simulation, measure_all

qr = QuantumRegister(2)
circuit = H(qr[0]) + CX(qr[0], qr[1]) + measure_all(qr)
result = run_simulation(circuit, 1000)
# {'00': 503, '11': 497}
```

Big-endian qubit ordering (qubit 0 is the leftmost bit). Supports CUDA, MPS, and CPU backends (auto-detected).

## Benchmark

```bash
uv run bench          # run all cases, print totals
uv run bench -v       # verbose: per-case timing + correctness details
uv run bench --cases real_grovers qft
uv run bench-plot     # plot the most recent results
```

Each benchmark case defines a circuit and its theoretical output distribution. The harness runs the full suite at fixed shot counts: 1, 10, 100, 1000, and 10000. Correctness is verified at the highest completed shot count by comparing the observed distribution against the expected one within a tolerance. If a case hits out-of-memory, the runner records it as an OOM failure and continues.

### Cases

| Case | Qubits | What it tests |
|------|--------|---------------|
| `bell_state` | 2 | Minimal circuit, baseline overhead |
| `simple_grovers` | 5 | Multi-controlled gates (Grover's search) |
| `real_grovers` | 13 | Deep circuit with hash oracle (Grover's preimage search) |
| `ghz_state` | 12 | Qubit scaling with shallow circuit (H + CX chain) |
| `qft` | 10 | Parametric controlled phase gates (QFT round-trip) |
| `teleportation` | 3 | Mid-circuit measurement and conditional gates |
| `phase_ladder` | 11 | Deep diagonal-gate stress (RZ + controlled phase round-trip) |
| `toffoli_oracle` | 11 | Toffoli-heavy nonlinear oracle round-trip |
| `adaptive_feedback` | 2 | Repeated mid-circuit measurement and conditional feedback |
| `ghz_state_16` | 16 | Larger GHZ scaling |
| `ghz_state_18` | 18 | Capacity-push GHZ scaling |
| `qft_12` | 12 | Larger QFT round-trip |
| `qft_14` | 14 | Deeper/larger phase-gate pressure |
| `phase_ladder_13` | 13 | Larger diagonal-gate stress |
| `toffoli_oracle_13` | 13 | Larger Toffoli-heavy oracle round-trip |
| `adaptive_feedback_120` | 2 | Long repeated measurement/conditional feedback |
| `reversible_mix_13` | 13 | Random reversible logic mix (X/CX/CCX round-trip) |
| `reversible_mix_15` | 15 | Larger reversible logic stress near backend limits |
| `clifford_scrambler_14` | 14 | Random Clifford scrambling + inverse |
| `brickwork_entangler_15` | 15 | Nearest-neighbor brickwork entangling pattern |
| `random_universal_12` | 12 | Random universal circuit (RX/RY/RZ/CX/CCX) round-trip |
| `random_universal_14` | 14 | Larger random universal round-trip |
| `diagonal_mesh_15` | 15 | Random long-range diagonal phase mesh round-trip |
| `adaptive_feedback_5q` | 5 | Mid-circuit feedback stress on larger state vectors |

Cases live in `benchmarks/cases/`.

Expanded synthetic families (randomized but reproducible with fixed seeds) are intentionally included in the default suite to broaden structural coverage across reversible logic, Clifford scrambling, universal gate mixes, diagonal phase meshes, and larger dynamic-feedback workloads.

## Optimization workflow

### Philosophy

This is a **general-purpose quantum simulator**. The benchmark suite exists to quantify progress, not to define it. Solving for one special case doesn't help — any optimization must improve the simulator's general performance across diverse circuit structures. Clean, DRY, modern Python is paramount; clever-but-messy code that saves a few percent is not worth the maintenance cost.

### Scope

- **Optimization target**: `src/quantum/system.py` only. Do not modify `gates.py` or benchmark cases.
- **Code quality**: Keep `system.py` clean and readable. Prefer structural improvements over micro-hacks. No dead code, no commented-out experiments, no special-case branches for specific benchmark cases.

### The loop

```
1. Profile    — identify where time is actually spent
2. Hypothesize — form a concrete, falsifiable prediction
3. Implement  — apply the change in src/quantum/system.py
4. Commit     — commit before benchmarking (creates clean audit trail)
5. Benchmark  — run full suite, evaluate correctness + timing
6. Evaluate   — compare against prior run, accept or revert
7. Record     — log outcome in docs/02-attempt-history.md
```

Each step in detail:

#### 1. Profile

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

#### 2. Hypothesize

Before writing any code, state:
- **What** you're changing
- **Why** you expect it to help (linked to profiler evidence)
- **Which circuit families** should improve (not just one case)
- **What could go wrong** (correctness risk, regression on other families)

Bad hypothesis: "Make `adaptive_feedback_5q` faster by caching its specific branch pattern."
Good hypothesis: "Replacing per-measurement `nonzero` calls with multiplicative masking should reduce dynamic-circuit overhead across all feedback workloads, because profiler shows `nonzero` at 73% of CPU time in the dynamic path."

#### 3. Implement

Edit `src/quantum/system.py`. Priorities:
- Structural clarity over micro-optimization
- General solutions over case-specific fixes
- Fewer full-state passes over faster individual passes

#### 4. Commit

Always commit **before** running benchmarks. This creates a 1:1 mapping between code states and benchmark artifacts for reproducibility and bisection.

#### 5. Benchmark

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

#### 6. Evaluate

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

#### 7. Record

Log the outcome in `docs/02-attempt-history.md` following the existing ledger format. Include: commit hash, hypothesis, measured result, verdict (worked / did not work), evidence artifact paths.

### Interpreting results

**Shot counts** `[1, 10, 100, 1000, 10000]` surface different behaviors:
- Low shots (1, 10): dominated by fixed overhead (compilation, device transfer, warmup)
- High shots (1000, 10000): dominated by per-shot or per-evolution costs

**Static vs dynamic**: the harness classifies circuits automatically:
- **Static**: no conditionals, terminal-only measurements → eligible for evolve-once/sample-many fast path
- **Dynamic**: mid-circuit measurements and/or conditional gates → requires branch-based execution

**Known failures**: `ghz_state_16` and `ghz_state_18` fail on MPS due to a tensor rank limit (MPS supports rank ≤ 16). These are backend-limit failures, not correctness bugs.

### Key files

| File | Purpose |
|------|---------|
| `src/quantum/system.py` | Simulation engine — the optimization target |
| `src/quantum/gates.py` | Gate types and circuit API — do not modify |
| `benchmarks/run.py` | Benchmark harness (`bench`) |
| `benchmarks/trace.py` | Profiler (`bench-trace`) |
| `benchmarks/compare.py` | SOTA comparison (`bench-compare`) |
| `benchmarks/cases/` | Benchmark case definitions — do not modify |
| `docs/02-attempt-history.md` | Canonical ledger of what was tried and what happened |
| `docs/03-findings.md` | Stable validated conclusions |
| `docs/04-roadmap.md` | Current hypotheses and priorities |
| `docs/10-hypotheses-post-sota-2026-02-11.md` | Latest hypothesis set with SOTA context |

### Anti-patterns

- **Benchmark hacking**: special-casing code for a specific test case name or structure. The benchmark is a proxy for general performance, not the goal.
- **Premature micro-optimization**: tuning constants or unrolling loops before fixing algorithmic issues (e.g., shot-scaled evolution).
- **Guessing without profiling**: always run `bench-trace` before hypothesizing. Intuition about GPU bottlenecks is frequently wrong.
- **Regressing correctness**: never trade correctness for speed. The tolerance check is the hard gate.
- **Complexity without payoff**: if an optimization adds significant code complexity for <5% broad improvement, it's probably not worth it.

## Performance tracker

Latest full benchmark artifact:

- Run: `benchmarks/results/2026-02-11T112001.jsonl`
- Commit: `a85bf9c`
- Completed correctness: 22/22 PASS (with known MPS rank-limit failures on `ghz_state_16` and `ghz_state_18`)

### Progress

![Optimization progress (single panel)](docs/images/optimization-progress-core6-single-panel.png)

Raw data: `docs/optimization-progress-core6-single-panel-data.md`.

Progress summary:

| Scope | Baseline | Latest | Speedup |
|---|---:|---:|---:|
| Core-6 suite total @1000 (`3df121d` -> latest) | 370.99s | 0.086s | 4318.9x |
| Expanded suite total @1000 (`2026-02-10T230611` -> latest) | 60.33s | 2.48s | 24.3x |
| Expanded suite total @10000 (`2026-02-10T230611` -> latest) | 592.91s | 2.48s | 239.1x |

### Latest SOTA Comparison

| Scope | Shot count | Native | Aer | qsim | Native vs Aer | Native vs qsim |
|---|---:|---:|---:|---:|---:|---:|
| Full suite | 1000 | 2.5505s | 0.6116s | n/a | 4.17x slower | n/a |
| Full suite | 10000 | 2.6881s | 2.9460s | n/a | 1.10x faster | n/a |
| Static intersection | 1000 | 0.9772s | 0.2119s | 0.2121s | 4.61x slower | 4.61x slower |
| Static intersection | 10000 | 0.9614s | 0.2814s | 0.2188s | 3.42x slower | 4.39x slower |
| Dynamic subset (full) | 1000 | 1.6367s | 0.3979s | n/a | 4.11x slower | n/a |
| Dynamic subset (full) | 10000 | 1.7734s | 2.6633s | n/a | 1.50x faster | n/a |

Detailed run log and profiler notes: `docs/02-attempt-history.md`.
Narrative docs index: `docs/README.md`.

## Setup

```bash
uv sync
```

## Examples

See `examples/` for standalone scripts: a Bell state, a simple Grover's search, and a full Grover's hash-preimage search.
