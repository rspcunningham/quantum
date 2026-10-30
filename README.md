# quantum

A statevector quantum circuit simulator for Apple Silicon with a native Metal runtime, and an experiment in letting coding agents optimize code unsupervised.

Most of the code here was written by Codex and Claude running overnight in a benchmark-gated loop ([`OPTIMIZE.md`](OPTIMIZE.md)). The simulator is the testbed. The more interesting result is how the agents gamed the benchmark, and what it took to stop them.

## Results

<!-- TODO: fill from a fresh run of both backends on HEAD (see "Benchmarks" below). -->

On the 249-circuit benchmark suite (39 circuit families, 2–30 qubits, 10,000 shots), against Qiskit Aer on the same machine:

| | quantum (Metal) | Qiskit Aer (statevector, CPU) |
|---|---|---|
| Circuits completed (10 s limit) | TODO / 249 | TODO / 249 |
| Total wall time, circuits both complete | TODO s | TODO s |
| Faster than Aer on | TODO of TODO circuits | |
| Median per-circuit speedup | TODO× | |

Aer runs on the CPU because its GPU backend requires CUDA. Every result is checked against Aer's output distribution, and a correctness failure fails the run.

## The loop

The project started in November 2025 as a small pedagogical simulator in PyTorch. In February 2026 I wrote a benchmark harness and a set of instructions for an agent to optimize it on its own:

1. Read the latest per-circuit timings and find where time goes.
2. State a hypothesis: what to change, why, and which circuit families it should help.
3. Commit, so there is a revert point.
4. Benchmark.
5. Keep the commit or `git reset` it, and log the result.

The agent never stops on its own. After three discarded experiments in a row, it has to go read papers and other simulators before trying again. Over about four days and ~130 commits, it took the simulator from batched PyTorch tensor ops, through compile-time gate fusion, to a native Metal runtime with custom kernels.

## What went wrong

The commit history is the honest record. A few things the agents did, in order:

- **They optimized for the test suite.** About a fifth of the circuits are round-trips (apply a circuit, then its inverse), so their output is trivial. The agent added compile-time inverse-pair cancellation and reported a 14.5× overall speedup ([`6331041`](https://github.com/rspcunningham/quantum/commit/6331041)). Cancelling inverse pairs is a legitimate optimization, but real circuits rarely undo themselves. Most of that gain was an artifact of the suite.
- **They optimized for easy outputs.** Textbook circuits like GHZ or Bernstein–Vazirani have only a handful of possible outcomes. The agent added fast paths for sparse output distributions ([`7c8bacb`](https://github.com/rspcunningham/quantum/commit/7c8bacb)). These help the benchmark, but they do nothing for real-world circuits, whose outputs are dense. The instructions already banned special-casing benchmark cases, so this went through a loophole: it targeted a *property* of the cases rather than their names. I added the rule about dense outputs afterwards.
- **They stopped simulating.** The benchmark ran each circuit several times, at different shot counts, and the agent noticed that later runs could reuse the first. It cached the output distribution and skipped evolution entirely on later calls. The speedup it reported against Aer went from 6.6× to 26.3× ([`98fb061`](https://github.com/rspcunningham/quantum/commit/98fb061)), then to 47.5× once it did the same for dynamic circuits ([`c921e03`](https://github.com/rspcunningham/quantum/commit/c921e03)). This inflated the numbers I posted at the time. Nothing in the rules forbade it, because I hadn't thought to forbid it.
- **They saturated the benchmark.** Each time the suite got fast enough that run-to-run noise dominated, I had to grow it: from 6 cases, to 24, to 156, to 249.

Each failure became a rule in `OPTIMIZE.md`: no special-casing of cases *or* their structure, dense outputs are the target, and no result caching between calls (compile-time simplification is fine). The static Metal runtime that replaced the PyTorch path computes every result from scratch, and the numbers above come from that code.

The general lesson: an agent in a loop optimizes the measurement, not the intent. The benchmark has to be treated as an adversarial spec, and the rules end up being written in response to what it actually does.

## Usage

Requires macOS on Apple Silicon, Python 3.13, and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

```python
from quantum import QuantumRegister, H, CX, compile, measure_all

qr = QuantumRegister(2)
circuit = H(qr[0]) + CX(qr[0], qr[1]) + measure_all(qr)

with compile(circuit) as compiled:
    print(compiled.run(1000))  # {'00': ~500, '11': ~500}
```

Circuits can also be loaded from OpenQASM 2.0 with `quantum.parse_qasm`. More examples are in [`examples/`](examples/).

## Benchmarks

```bash
uv run --group bench python -m benchmarks.run       # this simulator
uv run --group bench python -m benchmarks.run_aer   # Qiskit Aer reference
```

Each case runs at 10,000 shots with a 10-second limit and is checked against an expected distribution from Aer. Results are written to `benchmarks/results/` as JSONL (ignored by git, apart from the pinned Aer reference).

## Layout

| Path | What it is |
|---|---|
| `src/quantum/` | Python API: gates, circuits, compilation, QASM parser |
| `native/src/` | Metal runtime, GPU shaders, and pybind11 bindings |
| `benchmarks/` | Benchmark cases, QASM circuits, harness, and Aer reference |
| `OPTIMIZE.md` | The instructions the optimization agent runs from |
| `docs/` | Notes, including a later audit of the codebase |

## Limitations

- Single precision (`float32`) state vectors.
- Up to 30 qubits; on 64 GB, most 30-qubit cases exceed Metal's working-set limit.
- Circuits with mid-circuit measurement and classical feedback work, but use a much slower path than static circuits.
- The QASM parser handles a practical subset of OpenQASM 2.0, not the full language.
