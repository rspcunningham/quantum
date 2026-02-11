# Design: SOTA Comparison Harness

Date: 2026-02-11  
Scope: benchmark infrastructure and adapter layer (no simulator-core changes)  
Status: implemented (first production pass)

## 0) Implementation Snapshot

Implemented files:

1. `benchmarks/ir.py`
2. `benchmarks/backends/base.py`
3. `benchmarks/backends/native_adapter.py`
4. `benchmarks/backends/aer_adapter.py`
5. `benchmarks/backends/qsim_adapter.py`
6. `benchmarks/backends/__init__.py`
7. `benchmarks/compare.py`
8. `benchmarks/compare_report.py`

Implemented commands:

1. `uv sync --group compare`
2. `uv run bench-compare --backends native aer --suite full --repetitions 1`
3. `uv run bench-compare --backends native aer qsim --suite static --repetitions 1`
4. `uv run bench-compare-report --input <compare-jsonl...>`

Known constraints in current environment:

1. qsim supports static-only workloads in this harness (dynamic cases marked `UNSUPPORTED`).
2. On macOS, qsim is loaded with `KMP_DUPLICATE_LIB_OK=TRUE` to avoid OpenMP runtime duplication aborts in mixed torch/qsim environments.
3. Native MPS backend marks circuits requiring tensor rank > 16 as `UNSUPPORTED` (e.g., `ghz_state_16`, `ghz_state_18`) instead of failing late.

## 1) Objective

Build an apples-to-apples benchmark comparison pipeline between this simulator and external SOTA baselines, using the existing case suite and correctness criteria.

The output must answer:

1. Where we stand on static-circuit workloads.
2. Where we stand on dynamic-circuit workloads.
3. Which performance gaps are algorithmic vs backend/tooling artifacts.

## 2) Comparator Selection

Use two comparators with complementary strengths:

1. **Qiskit Aer** (`AerSimulator`, statevector family)
   - includes dynamic-circuit support and mature runtime.
2. **qsim** (via Cirq integration)
   - high-performance state-vector baseline for static/unitary-focused circuits.

Non-goal for first pass:

- GPU-only NVIDIA paths (cuQuantum) because local target environment is Apple Silicon.

## 3) Comparison Matrix

Two benchmark reports:

1. **Full-suite report**:
   - backends: `native`, `aer`
   - includes dynamic cases.
2. **Static-intersection report**:
   - backends: `native`, `aer`, `qsim`
   - excludes unsupported dynamic features in qsim.

All exclusions must be explicit and documented as `UNSUPPORTED`, not silently omitted.

## 4) Fairness Protocol (Strict)

### 4.1 Environment

1. Same machine/session for all backends.
2. Record:
   - CPU model, RAM
   - Python version
   - package versions
   - thread env vars (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, etc.)
3. Pin versions in a dedicated comparison lockfile or dependency group.

### 4.2 Timing

For each case/shot/backend, collect:

1. `compile_or_translate_time_s`
2. `execution_time_s`
3. `total_time_s`

Default comparison metric:

- `execution_time_s` (primary)
- `total_time_s` (secondary)

Rationale:

- prevents hiding translation/transpile costs while still isolating runtime engine behavior.

### 4.3 Warmup and Repetition

1. One warmup run per case/backend (not timed).
2. Timed runs repeated `N` times (suggest `N=5`).
3. Report median and min/max.

### 4.4 Correctness

Reuse existing expected-distribution tolerance checks from `benchmarks/run.py`.

Any backend output key ordering/endian mismatch must be normalized before correctness check.

## 5) Circuit Translation Strategy

## 5.1 Intermediate Representation (IR)

Add a backend-neutral IR extracted from current `Circuit`:

1. op types: `gate`, `measurement`, `conditional`, `circuit_boundary`
2. fields:
   - target qubits
   - gate matrix or symbolic id
   - measurement qubit/bit
   - condition integer

This avoids duplicated ad-hoc translators.

### 5.2 Endianness and Bit Mapping

Must define a single conversion contract:

1. Native simulator:
   - big-endian qubit and classical-string convention.
2. External backend outputs:
   - normalize to native convention before scoring.

Add explicit normalization helper and validation micro-tests for mapping correctness.

### 5.3 Gate Coverage

Minimum gate mapping to support existing suite:

1. `H`, `X`, `Y`, `Z`
2. `RX`, `RY`, `RZ`
3. `CX`, `CCX`
4. controlled-phase / generic controlled gate forms used in synthetic cases

Fallback rule:

- if exact gate mapping unavailable in backend API, inject explicit unitary matrix for that op.

### 5.4 Dynamic Semantics

1. `Measurement(qubit, bit)` must map to explicit classical register target.
2. `ConditionalGate(condition)` must map to full-register equality semantics, not single-bit shorthand.

## 6) Adapter Architecture

New files:

1. `benchmarks/backends/base.py`
   - `BackendAdapter` protocol.
2. `benchmarks/backends/native_adapter.py`
   - wraps existing `run_simulation`.
3. `benchmarks/backends/aer_adapter.py`
4. `benchmarks/backends/qsim_adapter.py`
5. `benchmarks/ir.py`
   - IR extraction and normalization helpers.
6. `benchmarks/compare.py`
   - multi-backend orchestrator.
7. `benchmarks/compare_report.py`
   - merge JSONL outputs and generate summary tables.

### 6.1 Adapter Interface

Implemented methods:

1. `name() -> str`
2. `version_info() -> dict[str, str]`
3. `supports(case_ir) -> tuple[bool, str | None]`
4. `prepare(case_ir) -> PreparedCase`
5. `run(prepared_case, shots, warmup=False) -> BackendRunResult`

Result payload fields:

1. counts dict
2. timing breakdown
3. metadata (backend config, thread settings)

## 7) Implementation Sequence

### Phase A: Harness Plumbing

1. Add backend adapter interface and IR extraction.
2. Keep native adapter first to prove no regression in current benchmark behavior.

### Phase B: Aer Integration

1. Implement gate + measurement + conditional translation.
2. Add output normalization to native bitstring convention.
3. Validate on full suite.

### Phase C: qsim Integration

1. Implement static-circuit translation.
2. Mark dynamic cases as unsupported with explicit reason.
3. Validate static intersection suite.

### Phase D: Reporting

1. Generate per-backend JSONL artifacts.
2. Generate merged comparison markdown/CSV:
   - per-case speedups
   - family-level aggregates
   - unsupported-case matrix

## 8) Commands We Support

1. `uv run bench-compare --backends native aer --suite full -v`
2. `uv run bench-compare --backends native aer qsim --suite static -v`
3. `uv run bench-compare-report --input benchmarks/results/compare-*.jsonl`

## 9) Risk Register

1. **Python version compatibility** (especially external packages):
   - Mitigation: dedicated compare environment pinning; version check command in harness startup.
2. **Transpiler/circuit rewrite bias**:
   - Mitigation: record both translation and execution times; expose backend transpiler settings.
3. **Endianness mismatch**:
   - Mitigation: centralized normalization and mapping tests.
4. **Dynamic semantics drift**:
   - Mitigation: targeted conditional/measurement fixture cases before full-suite run.
5. **Threading unfairness**:
   - Mitigation: enforce and record thread env vars in result metadata.

## 10) Validation Plan

1. Sanity fixtures:
   - bell state
   - repeated classical-bit overwrite
   - simple conditional toggle loop
2. Full-suite correctness for backends that claim support.
3. Compare native-vs-native adapter parity first.
4. Publish first baseline comparison artifact with reproducibility metadata.

## 11) Deliverables

1. `benchmarks/results/compare-<timestamp>.jsonl`
2. `benchmarks/results/compare-<timestamp>.md`
3. Capability matrix table (`supported`, `unsupported`, `reason`)
4. Reproduction command block in report header.

## 12) Definition of Done

1. We can run one command to produce a reproducible multi-backend report.
2. Full-suite native vs Aer comparison exists with correctness checks.
3. Static-intersection native vs Aer vs qsim comparison exists.
4. Unsupported cases are explicitly enumerated with reasons.
5. Report is enough to guide next optimization priorities.
