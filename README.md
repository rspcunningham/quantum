# Quantum Simulator

A state-vector quantum circuit simulator built from scratch with PyTorch.

## Architecture

The simulator has two layers:

**API layer** (`gates.py`) — defines the circuit-building primitives. Gates (`H`, `X`, `CX`, etc.), parametric gates (`RX`, `RY`, `RZ`), arbitrary controlled gates via `ControlledGateType`, measurements, conditional gates, quantum registers, and `Circuit` composition with `+`, `*`, and `.inverse()`.

**Simulation layer** (`system.py`) — executes circuits against state vectors. `QuantumSystem` handles single-shot simulation. `BatchedQuantumSystem` runs many shots in parallel as a single `(batch_size, 2^n)` tensor, making it efficient on GPU. `run_simulation()` is the main entry point.

```python
from quantum import QuantumRegister, H, CX, run_simulation, measure_all

qr = QuantumRegister(2)
circuit = H(qr[0]) + CX(qr[0], qr[1]) + measure_all(qr)
result = run_simulation(circuit, 1000)
# {'00': 503, '11': 497}
```

Gates are unitary matrices applied via Kronecker products. Measurements are projective collapses. All operations maintain normalized state vectors. Big-endian qubit ordering (qubit 0 is the leftmost bit).

Supports CUDA, MPS, and CPU backends (auto-detected).

## Benchmark

The `benchmarks/` directory contains a harness for evaluating simulator performance:

```bash
uv run bench          # run all cases, print totals
uv run bench -v       # verbose: per-case timing + correctness details
uv run bench-plot     # plot the most recent results
```

Each benchmark case defines a circuit and its theoretical output distribution. The harness runs every case at 1, 10, 100, and 1000 shots, timing each. Correctness is verified at 1000 shots by comparing the observed distribution against the expected one within a tolerance.

Results are written as JSONL to `benchmarks/results/`, one file per run, one line per case:

```json
{"case": "bell_state", "n_qubits": 2, "times_s": {"1": 0.01, "10": 0.04, "100": 0.19, "1000": 1.84}, "correct": true}
{"case": "simple_grovers", "n_qubits": 5, "times_s": {"1": 0.15, "10": 0.20, "100": 0.57, "1000": 4.26}, "correct": true}
{"case": "real_grovers", "n_qubits": 13, "times_s": {"1": 137.1, "10": 139.0, "100": 175.4, "1000": 528.5}, "correct": true}
```

Current cases: Bell state (2 qubits), Grover's search (5 qubits), and Grover's with a hash oracle (13 qubits).

## Optimization workflow

The benchmark exists to support an iterative optimization loop driven by a coding agent (Claude Code). The cycle is:

1. Run `uv run bench -v` to establish a baseline
2. Have the agent propose and implement a performance optimization in `system.py`
3. Re-run the benchmark to measure the impact
4. If correctness holds and times improve, keep the change; otherwise revert

The multiple shot counts (1, 10, 100, 1000) surface optimizations that behave differently at different batch sizes. The correctness check guards against regressions.

## Setup

```bash
uv sync
```

## Examples

See `examples/` for standalone scripts: a Bell state, a simple Grover's search, and a full Grover's hash-preimage search.
