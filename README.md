# Quantum Simulator

A state-vector quantum circuit simulator built from scratch with PyTorch, targeting Apple Silicon (MPS) as a first-class backend.

![Native vs Aer](docs/native-vs-aer.png)

229 test circuits spanning 2-30 qubits, benchmarked cold (cache cleared, @1K shots) and warm (cached, @10K shots) against Qiskit Aer (IBM's C++ simulator). Green = native is faster. Warm throughput: **36.9x faster** (216/229 wins). Cold start: **0.8x** — Aer is currently faster on first call due to compilation overhead. Overall geometric mean: **5.3x faster**.

## How it got here

This started as a naive PyTorch simulator — 700 seconds for 6 circuits. Then we pointed [Claude Code](https://docs.anthropic.com/en/docs/claude-code) at it with a benchmark harness and a simple loop: profile, hypothesize, implement, measure, repeat.

40 iterations later: 38,000x faster. The agent discovered compile-time gate fusion, inverse-pair cancellation, CDF caching, probability-weighted branching for dynamic circuits, and a dozen other optimizations — each validated against correctness checks across the full suite.

The optimization workflow is documented in [`OPTIMIZE.md`](OPTIMIZE.md). The full experiment history is in [`docs/experiment-log.md`](docs/experiment-log.md).

## Quick start

```bash
uv sync
```

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
uv run bench -v                         # full suite, cold @1K + warm @10K per circuit
uv run bench --cases real_grovers qft   # selected cases only
uv run bench -v --backend aer           # same suite on Aer backend
```

Each circuit runs twice: **cold @1K** (cache cleared — measures compilation + first execution) and **warm @10K** (cached — measures pure throughput). Both matter: cold is what a user feels the first time, warm is what they feel on repeated calls.

## Examples

See `examples/` for standalone scripts: a Bell state, a simple Grover's search, and a full Grover's hash-preimage search.
