# Design: H1 Dynamic Branch Engine

Date: 2026-02-11  
Scope: `src/quantum/system.py` only  
Status: design only, no implementation yet

## 1) Objective

Replace the current conditional-gate execution strategy in batched simulation with a branch-based dynamic execution engine that:

1. preserves circuit semantics exactly,
2. eliminates per-shot boolean indexing hot paths, and
3. improves dynamic-feedback workloads (`adaptive_feedback*`, `teleportation`) without regressing static-circuit paths.

## 2) Why This Is Next

Post-H0/H0.1 profiling shows dynamic execution is now dominant:

- `adaptive_feedback_5q` and `adaptive_feedback_120` are dominated by:
  - `aten::nonzero`
  - `aten::index` / `aten::index_put_`
  - `aten::item` / `aten::_local_scalar_dense`
- Static terminal-measurement workloads are comparatively cheap after H0 + sampler offload.

Current root cause in code:

- `BatchedQuantumSystem.apply_one()` (`src/quantum/system.py`) handles `ConditionalGate` by:
  1. building mask over all shots,
  2. cloning/restoring non-matching states,
  3. applying gate to whole batch anyway.

This is inherently expensive and sync-heavy for dynamic feedback loops.

## 3) Non-Goals (V1)

1. No symbolic simulator or stabilizer-mode rewrite.
2. No gate-definition API changes (`gates.py` untouched).
3. No benchmark case edits.
4. No approximation methods that alter statistical correctness.

## 4) High-Level Architecture

Introduce a new dynamic execution path in `run_simulation()` for circuits containing:

- `ConditionalGate`, or
- non-terminal measurements.

Core idea: simulate **classical branches** with shot counts, not full per-shot batches.

### 4.1 Branch State

Each branch stores:

1. `state_vector`: complex tensor shape `(1, 2^n)` (or `(2^n,)`) on device.
2. `classical_value`: integer encoding current classical register.
3. `shots`: integer count of shots represented by this branch.

Interpretation: this branch represents `shots` identical trajectories sharing the same quantum state and classical register value at current execution point.

### 4.2 Branch Map

Maintain `dict[int, BranchState]` keyed by `classical_value`.

Invariant:

- all branches have strictly positive `shots`.
- total shots across branches equals `num_shots`.

### 4.3 Transition Types

1. `Gate`:
   - apply once per branch state.
2. `Measurement(qubit, bit)`:
   - for each branch, compute `p1`.
   - sample `k1 ~ Binomial(shots, p1)` (on CPU RNG; deterministic seed plumbing later).
   - split branch into up to two children (`outcome=0/1`) with projected/renormalized state.
   - update classical register bit in child key.
3. `ConditionalGate(gate, condition)`:
   - apply gate only on branches where `classical_value == condition`.

## 5) Semantic Requirements

The new path must preserve:

1. Big-endian classical register formatting.
2. Sequential measurement semantics.
3. Classical bit overwrite semantics.
4. Conditional gate equality semantics against full register value.
5. Output type: `dict[str, int]`.
6. Same correctness tolerances under benchmark harness.

Important note:

- Shot-level trajectories will differ from current implementation due to different random draw grouping, but output distribution must remain equivalent.

## 6) Detailed Implementation Plan

### Phase A: Circuit Analysis + Execution Plan

Add helpers:

1. `flatten_circuit_ops(circuit) -> list[Op]`
2. `has_dynamic_behavior(ops) -> bool`
3. `compile_dynamic_plan(ops) -> list[PlanStep]`

`PlanStep` kinds:

1. `unitary_segment` (contiguous `Gate` ops)
2. `measurement`
3. `conditional_gate`

Why:

- avoids repeated `isinstance` checks and recursive traversal in hot loop.

### Phase B: Branch Engine Core (MVP)

Add internal executor:

`_run_dynamic_branch_simulation(circuit, num_shots, n_qubits, n_bits, device) -> dict[str, int] | None`

Flow:

1. Build dynamic plan; return `None` if circuit is static (fallback to existing static path).
2. Initialize one branch:
   - state = `|0...0>`
   - classical=0
   - shots=`num_shots`
3. Execute each plan step:
   1. `unitary_segment`: apply gates to each branch.
   2. `measurement`: split/merge branches by outcome and updated classical value.
   3. `conditional_gate`: apply only to matching branches.
4. Materialize final counts from branch shot totals and terminal measurements.

### Phase C: Measurement Split Mechanics

For each branch on measurement:

1. Compute `p1` from branch state.
2. Draw `k1` with binomial using branch `shots`.
3. `k0 = shots - k1`.
4. If `k0 > 0`, create/update branch for outcome 0.
5. If `k1 > 0`, create/update branch for outcome 1.

Branch merge rule:

- if child classical key already exists and state tensors are mathematically identical for this transition context, add shot counts.
- V1 safe merge scope: only immediate siblings from same parent/outcome path.

### Phase D: Conditional Step

For each branch:

1. if `branch.classical_value == condition`, apply gate to branch state.
2. else no-op.

No masks, no advanced indexing, no restore copies.

### Phase E: Result Materialization

At end of plan:

1. If final measurements already consumed, branch `shots` directly map to classical bitstrings.
2. If trailing terminal measurements remain, optionally reuse static terminal sampling helper per branch.
3. Aggregate counts in Python dict.

## 7) Fallbacks and Safety

Fallback policy in `run_simulation()`:

1. Try static terminal fast path (existing).
2. Else try dynamic branch engine.
3. Else use current batched engine.

Debug safety flag:

- Add internal env flag (for development) to force old path for A/B correctness checks.

## 8) Performance Strategy

1. Keep branch count low:
   - prune zero-shot branches immediately.
   - cap/monitor branch explosion; emergency fallback to old path if cap exceeded.
2. Keep quantum states on device.
3. Keep branch metadata (`shots`, classical keys) on CPU.
4. Avoid `.item()` in hot loop except unavoidable scalar extraction points.
5. Batch unitary application across branches when branch count > 1 (phase 2 optimization).

## 9) Data Structures and APIs (Planned)

Planned internal types (in `system.py` only):

1. `_BranchState`
   - `state: torch.Tensor`
   - `shots: int`
   - `classical_value: int`
2. `_PlanStep`
   - `kind: Literal["unitary_segment","measurement","conditional"]`
   - payload fields per kind

Planned helpers:

1. `_compile_dynamic_plan(...)`
2. `_apply_unitary_segment_to_branch(...)`
3. `_measure_branch_and_split(...)`
4. `_apply_conditional_to_branches(...)`
5. `_materialize_branch_counts(...)`

## 10) Validation Protocol

### 10.1 Correctness

1. Full `uv run bench -v` must keep 22/22 runnable PASS.
2. Focus checks:
   - `teleportation`
   - `adaptive_feedback`
   - `adaptive_feedback_120`
   - `adaptive_feedback_5q`
3. Add deterministic custom circuits for:
   - repeated bit overwrite
   - repeated conditionals with canceling gates
   - mixed terminal/non-terminal measurements

### 10.2 Performance

Primary success criteria:

1. Dynamic-family improvement at 1000/10000 shots.
2. No meaningful regressions on static-family totals.
3. Trace-level reduction in:
   - `aten::nonzero`
   - `aten::index` / `index_put_`
   - `aten::item` / `_local_scalar_dense`

## 11) Rollout Sequence

1. Implement plan compiler scaffolding.
2. Implement dynamic branch MVP without batching.
3. Validate correctness (targeted + full bench).
4. Profile dynamic cases.
5. Add branch-batch unitary optimization if needed.
6. Update `docs/02-attempt-history.md` with measured result.

## 12) Risks and Mitigations

1. Risk: branch explosion for large `n_bits`.
   - Mitigation: branch cap + fallback.
2. Risk: numerical drift from repeated projection/renorm.
   - Mitigation: norm guards + existing clamp behavior.
3. Risk: subtle classical encoding mismatch.
   - Mitigation: shared bit-encoding helpers and edge-case tests.
4. Risk: dynamic engine overhead exceeds benefit on tiny cases.
   - Mitigation: heuristic threshold to use old path for very small workloads.

## 13) Definition of Done

1. New dynamic path enabled by default.
2. 22/22 runnable correctness maintained.
3. Dynamic workloads show clear wins.
4. No broad static regression.
5. Attempt logged and findings/roadmap updated.
