# Quantum Simulator Audit Log

Date: 2026-05-01

Goal: assess the repository for performance, modern best practices, low overhead, and robustness, with the target of extremely performant and lightweight statevector simulation on Apple M-series chips.

## Scope Notes

- Source surface found so far: Python package in `src/quantum`, native Metal extension in `native/src`, benchmark harnesses in `benchmarks`, examples, one pytest file, CMake/pyproject metadata, and a large QASM/expected-output benchmark corpus.
- Initial `git status --short` was clean before this log was created.

## Running Observations

- Repository appears intentionally lightweight: no large framework, no frontend, no service layer.
- Need to verify whether the native backend is truly used by default and whether Python fallback paths are still competitive or mainly correctness/reference paths.
- Need to check measurement/reset behavior carefully because statevector collapse tends to be a robustness hotspot.
- Need to inspect benchmark methodology before trusting any speed claims.

## Audit Commands / Verification

- `git status --short` was clean before creating this file; after audit work, only `log.md` is untracked.
- `uv run pytest -q` failed because `pytest` is not installed.
- `uv run python -m unittest -q` ran 0 tests; `uv run python -m unittest discover -s tests -q` ran 7 tests and passed.
- `uv run python tests/test_native_compile_isolation.py` ran 7 tests and passed.
- `uv run basedpyright` failed with 190 errors and 638 warnings. Most errors are typing/API-shape issues around `QuantumRegister.__getitem__`, import cycles in benchmark cases, and QASM AST constant typing.
- Current `HEAD` is `91ead40`. The newest full native benchmark result I inspected records git hash `04db5be`, so it is stale relative to this checkout.
- Targeted current-head check: teleportation dynamic circuit produced the expected roughly uniform `001/011/101/111` distribution over 2000 shots.
- Targeted current-head check: `RX(0.3) + Circuit([RX(0.3)]).inverse()` measured about 8.7% `1`, confirming `Circuit.inverse()` is not a real adjoint for parametric/non-self-inverse gates.
- Parsed all 225 QASM files under `benchmarks/circuits`; current parser accepted all of them and produced 127,539 top-level operations total.

## High-Impact Findings

### Import / Packaging Weight

- `src/quantum/__init__.py` imports `plot_results`, and `plot_results` imports seaborn/matplotlib at module import time. A measured `python -X importtime -c "import quantum"` cold import took about 21.9 seconds in this environment and dragged in NumPy, Matplotlib, Seaborn, SciPy, Pillow, etc.
- `pyproject.toml` puts `rich`, `matplotlib`, `seaborn`, and `qiskit-aer` in core runtime dependencies. For a lightweight simulator, these should be optional extras such as `quantum[viz]` and `quantum[bench]`.
- `pyproject.toml` packages `benchmarks` into the wheel. This conflicts with the goal of a lightweight runtime. If benchmark assets are included, the package is bloated; if data files are excluded, the installed benchmark console scripts are likely broken.
- `src/quantum/gates.py` eagerly imports `quantum_native_runtime`, so even constructing a `QuantumRegister` or `Circuit` requires the compiled extension. `metal_exec.py` attempts lazy native error reporting, but `gates.py` defeats that.

### Correctness / API Robustness

- `Circuit.inverse()` only reverses operation order and returns the same gate objects. It does not conjugate-transpose gates, negate rotation angles, or reject measurements/conditionals. This is a correctness bug for any non-self-inverse gate.
- `QuantumRegister.__getitem__` returns `int | QuantumRegister`, which makes every `qr[0]` statically appear as possibly a `QuantumRegister`. This causes the bulk of pyright argument errors. Runtime works for integer indexing, but the annotation needs overloads. Slice handling is also semantically wrong for strides/reversed slices because it returns a contiguous register from the first selected index.
- `Gate.if_(classical_bit: int)` is named as if it accepts a bit index, but the native and benchmark semantics are full classical-register integer equality. This is easy to misuse.
- `infer_resources()` cannot infer needed classical bits from `ConditionalGate(condition=0)` because `0.bit_length() == 0`. A circuit containing only `X(...).if_(0)` may infer `n_bits=0`.
- `GateType.dimension` says it returns number of target qubits, but returns matrix dimension (`2`, `4`, `8`). This is harmless internally for `on()`, but misleading public API.
- `CustomGateType` is imported in `__init__.py` but omitted from `__all__`, causing static export warnings and making the public surface unclear.

### Static Runtime Hot Path

- `run_simulation()` always flattens the Python circuit and calls native `run_circuit()`.
- Native `run_circuit()` compiles a static program, executes it, and frees it every call. The runtime has persistent per-program state buffers, but the public Python path immediately discards the program handle, so those buffers do not amortize across repeated runs.
- There is a native `compile_circuit` / `execute_static_program` path, but no Python-level compiled-circuit API. For repeated use, this is probably the largest low-risk improvement.
- The benchmark profiler contains dead cache-clearing code for `_circuit_compilation_cache`, which no longer exists. That suggests a previous caching architecture was removed but tooling was not updated.

### Metal Runtime / Apple M-Series Performance

- All Metal buffers are created with `MTLResourceStorageModeShared`, including large state and scratch buffers. Apple’s Metal guidance for macOS says GPU-exclusive buffers should use `Private`; large data initialized once by CPU and frequently read by GPU should be staged from shared/managed into private. On Apple Silicon this needs measurement, but current code is not following Apple’s general macOS buffer guidance.
- Static sampling computes probabilities, then builds a full CDF using a Hillis-Steele style scan: one full-state kernel pass per `offset <<= 1`. For 28 qubits, that is 28 full passes over the state just for the CDF. This likely dominates low-depth high-qubit circuits.
- The CDF is `float`, and RNG uses only 24 random bits in `uniform01_from_u64`. That is likely acceptable for coarse benchmark tolerances but weak for robust sampling over large, dense distributions.
- State representation uses split real/imag `float` arrays and ping-pong buffers. This is memory-bandwidth friendly for simple elementwise kernels but doubles buffer binding and makes dense gate kernels read scattered pairs. It should be benchmarked against `float2` packed complex for dense1/dense2 and fused kernels.
- `n_qubits > 30` is rejected. With 64 GB unified memory, 30 qubits already requires roughly 16 GiB for four float state buffers plus roughly 8 GiB for two probability buffers before overhead. The current working-set guard rejects many 30-qubit benchmark cases, which is prudent but means “30q on 64 GB” is not consistently available.
- Pipeline states are lazy-created on first use. Apple recommends building known pipelines up front/asynchronously to avoid runtime stalls. This matters less for long benchmark cases but matters for interactive use and first-shot latency.
- Command buffer use is generally good for static execution: gate dispatch and sampling are encoded into one command buffer, matching Apple’s guidance to use few command buffers. Dynamic execution breaks this by running segment after segment through separate compilation/execution.

### Dynamic Runtime

- Current `HEAD` includes a dynamic branch-tree implementation, and a small teleportation case works.
- The dynamic runtime is not production-grade for performance: it stores full CPU state vectors per branch, copies states for branch forks, recompiles segments per branch/classical state, and round-trips through Metal for each segment.
- `execute_dynamic_circuit()` computes `1ULL << n_qubits` and `1ULL << n_bits` without validating bounds first. Large or invalid inputs can overflow or attempt impossible allocations.
- Dynamic scratch buffers are allocated with `calloc`, not checked for null, and freed only on the normal path. Exceptions leak them. They are also passed into `execute_gates_only()` but that function currently allocates its own Metal scratch buffers and does not use the provided scratch pointers.
- Branch merging compares every pair of states with a full-vector scan, so it is O(branches^2 * dim). It is safe for tiny feedback cases but will not scale.

### QASM Parser

- Parser is intentionally small and regex-based. That is fine for controlled benchmark QASM, but not robust enough for a public parser.
- `_parse_gate_line()` silently returns `[]` on unmatched syntax instead of raising, so malformed QASM can be accepted without notice.
- Comments are only ignored when the stripped line starts with `//`; inline comments can break parsing.
- Whole-register operations such as `h q;` are not supported.
- Parameter evaluation supports a small AST subset but lacks arity validation for function calls and does not support exponentiation. Error messages are raw AST dumps or IndexError/KeyError in several malformed cases.
- Gate arity and parameter counts are not validated before indexing, so bad QASM often raises `IndexError` instead of a useful parser error.

### Benchmarks / Results

- Historical benchmark results under `benchmarks/results` consume about 511 MB for the repo; trace JSON files dominate. These should not live in the runtime package and probably should not live in the main repo unless intentionally tracked as research artifacts.
- Newest historical full run at `04db5be` shows strong 28-qubit static performance on some cases, but 30-qubit cases mostly fail the recommended Metal working-set guard. Dynamic cases in that run were marked unsupported, which is stale relative to current `HEAD`.
- Benchmark correctness uses distribution tolerances, not exact state validation. This is practical for sampling, but it can hide coherent phase errors in circuits whose final distribution is simple.
- `check_correctness()` divides by `total` without guarding against empty/zero result maps.
- Benchmark harness uses a long-lived subprocess, which is a good robustness strategy for GPU timeouts, but the installed package should separate this from runtime dependencies.

### Tests / Typing

- Only one smoke test module exists. It dynamically creates 7 core benchmark tests and validates sampled distributions.
- There are no direct tests for QASM malformed inputs, bit-order conventions, inverse semantics, dynamic branch robustness, compiled program reuse, error boundaries, 0-shot/negative-shot behavior, or packaging/import weight.
- `pytest` is not a dependency, but `uv run pytest -q` is a natural command and currently fails. Either add pytest to dev dependencies or document/use unittest explicitly.
- Basedpyright is configured but currently fails heavily. The largest actionable typing fix is to overload `QuantumRegister.__getitem__`.

## External Source Notes

- Apple Metal Best Practices: resource storage modes should be chosen deliberately; for macOS buffers accessed exclusively by GPU, Apple recommends `Private`, and for large data initialized once and frequently used by GPU, Apple recommends staging into `Private`.
- Apple Metal Best Practices: submit the fewest command buffers practical, preferably one or two, to balance CPU/GPU work and avoid synchronization stalls.
- Apple Metal Best Practices: build known render/compute pipelines up front/asynchronously and avoid lazy loading.
- Apple Accelerate overview: vDSP/vForce/BLAS/LAPACK provide optimized CPU vector, arithmetic, reduction, and linear algebra routines. This is relevant for CPU-side dynamic collapse/sampling paths if they remain CPU-based.
- Source links:
  - https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/ResourceOptions.html
  - https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/CommandBuffers.html
  - https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/Pipelines.html
  - https://developer.apple.com/accelerate/

## Priority Recommendations

1. Split runtime from optional features: make `import quantum` avoid visualization, Qiskit, Rich, and benchmark imports. Move plotting and Aer into extras. This is the clearest lightweight win.
2. Add a Python `CompiledCircuit` API backed by native `compile_circuit`, `execute_static_program`, and `free_program`; keep `run_simulation()` as convenience. This lets state buffers and native program lowering amortize across repeated runs.
3. Replace static sampling scan with a work-efficient hierarchical GPU prefix sum, or add specialized samplers for common sparse/structured cases. Current full CDF scan is an obvious high-qubit bottleneck.
4. Fix `Circuit.inverse()` or remove/rename it until correct. Incorrect inverse semantics are dangerous because benchmarks/examples use the name heavily.
5. Add input validation across Python and native boundaries: qubit/bit bounds, duplicate targets, dense matrix shape, negative shots, large `n_bits`, malformed QASM, measurement bit bounds, and condition width.
6. Rework Metal buffer storage modes and allocation reuse based on measured Apple Silicon behavior. At minimum, separate GPU-only state/scratch buffers from CPU-read histogram/output buffers.
7. Stabilize dynamic circuits by caching compiled segments within a run, using RAII vectors/buffers, bounding branch growth, and deciding whether dynamic support is a first-class performance target or a correctness-only fallback.
8. Tighten tests: add deterministic seeded tests, inverse tests, QASM parser tests, dynamic tests, and small native unit tests around edge cases. Add a benchmark smoke suite that runs in seconds.
