# quantum

A statevector quantum circuit simulator for Apple Silicon with a native Metal runtime, and an experiment in letting coding agents optimize code unsupervised.

Most of the code here was written by Claude Opus 4.6, running overnight in a benchmark-gated loop ([`OPTIMIZE.md`](OPTIMIZE.md)). The simulator is the testbed. The more interesting result is how the agent gamed the benchmark, and what it took to stop it.

## Results

On the 226 static circuits in the benchmark suite (2–30 qubits, 10,000 shots per circuit), against Qiskit Aer on the same machine (M3 Max, 64 GB):

| | quantum (Metal) | Qiskit Aer (statevector, CPU) |
|---|---|---|
| Circuits completed (10 s limit) | 211 / 226 | 209 / 226 |
| Total time, the 209 circuits both complete | 27.7 s | 76.7 s |
| Faster on | 200 of 209 | 9 of 209 |
| Median per-circuit speedup | 4.8× | |

By circuit size, over the same 209 circuits:

| Qubits | Circuits | quantum faster on | Median speedup |
|---|---|---|---|
| 2–12 | 97 | 91 | 3.4× |
| 13–20 | 77 | 77 | 7.1× |
| 21–30 | 35 | 32 | 3.1× |

Aer runs on the CPU because its GPU backend requires CUDA. Every result is checked against an expected output distribution, and all completed runs on both sides pass.

Where it falls short: most 28–30-qubit circuits hit the 10-second limit on both simulators, and Aer is still 1.7–2.5× faster on a few dense 24-qubit circuits (quantum volume, SU(4) chains, Heisenberg). The suite also has 23 dynamic circuits (mid-circuit measurement with classical feedback), which the current runtime doesn't support, so they're left out.

The runs behind these numbers are pinned in `benchmarks/results/` as `native-reference.jsonl` and `aer-reference.jsonl`.

## The loop

The project started in November 2025 as a small pedagogical simulator in PyTorch. In February 2026 I wrote a benchmark harness and a set of instructions for an agent to optimize it on its own:

1. Read the latest per-circuit timings and find where time goes.
2. State a hypothesis: what to change, why, and which circuit families it should help.
3. Commit, so there is a revert point.
4. Benchmark.
5. Keep the commit or `git reset` it, and log the result.

The agent never stops on its own. After three discarded experiments in a row, it has to go read papers and other simulators before trying again. Over about four days and ~130 commits, it took the simulator from batched PyTorch tensor ops, through compile-time gate fusion, to a native Metal runtime with custom kernels.

## What went wrong

The commit history is the honest record. A few things the agent did, in order:

- **It optimized for the test suite.** About a fifth of the circuits are round-trips (apply a circuit, then its inverse), so their output is trivial. The agent added compile-time inverse-pair cancellation and reported a 14.5× overall speedup ([`6331041`](https://github.com/rspcunningham/quantum/commit/6331041)). Cancelling inverse pairs is a legitimate optimization, but real circuits rarely undo themselves. Most of that gain was an artifact of the suite.
- **It optimized for easy outputs.** Textbook circuits like GHZ or Bernstein–Vazirani have only a handful of possible outcomes. The agent added fast paths for sparse output distributions ([`7c8bacb`](https://github.com/rspcunningham/quantum/commit/7c8bacb)). These help the benchmark, but they do nothing for real-world circuits, whose outputs are dense. The instructions already banned special-casing benchmark cases, so this went through a loophole: it targeted a *property* of the cases rather than their names. I added the rule about dense outputs afterwards.
- **It stopped simulating.** The benchmark ran each circuit several times, at different shot counts, and the agent noticed that later runs could reuse the first. It cached the output distribution and skipped evolution entirely on later calls. The speedup it reported against Aer went from 6.6× to 26.3× ([`98fb061`](https://github.com/rspcunningham/quantum/commit/98fb061)), then to 47.5× once it did the same for dynamic circuits ([`c921e03`](https://github.com/rspcunningham/quantum/commit/c921e03)). This inflated the numbers I posted at the time. Nothing in the rules forbade it, because I hadn't thought to forbid it.
- **It saturated the benchmark.** Each time the suite got fast enough that run-to-run noise dominated, I had to grow it: from 6 cases, to 24, to 156, to 249.

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
scripts/bench.sh   # both backends, plus machine info, logged to benchmarks/results/
```

Or one backend at a time:

```bash
uv run --group bench python -m benchmarks.run       # this simulator
uv run --group bench python -m benchmarks.run_aer   # Qiskit Aer reference
```

Each case runs at 10,000 shots with a 10-second limit and is checked against an expected distribution from Aer. Results are written to `benchmarks/results/` as JSONL (ignored by git, apart from the two pinned reference runs).

## Layout

| Path | What it is |
|---|---|
| `src/quantum/` | Python API: gates, circuits, compilation, QASM parser |
| `native/src/` | Metal runtime, GPU shaders, and pybind11 bindings |
| `benchmarks/` | Benchmark cases, QASM circuits, harness, and Aer reference |
| `OPTIMIZE.md` | The instructions the optimization agent runs from |
| `scripts/bench.sh` | Runs both benchmark backends and logs the output |
| `docs/` | Notes, including a later audit of the codebase |

## Limitations

- Single precision (`float32`) state vectors.
- Static circuits only. An earlier version supported mid-circuit measurement and classical feedback, but the current compiled runtime does not.
- Up to 30 qubits, though most 28–30-qubit circuits take longer than the benchmark's 10-second limit.
- The QASM parser handles a practical subset of OpenQASM 2.0, not the full language.
