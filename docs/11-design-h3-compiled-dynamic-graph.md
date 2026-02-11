# Design: H3 Compiled Dynamic Graph Executor

Date: 2026-02-11  
Scope: `src/quantum/system.py`

## 1) Goal

Unify dynamic execution around a graph model:

1. static-quality unitary execution on edges,
2. explicit control-flow nodes for measurement and conditionals,
3. branch/path bookkeeping decoupled from tensor evolution.

The target is general dynamic-circuit acceleration without benchmark-specific case logic.

## 2) Core Model

Dynamic circuit as a directed graph over a linear program:

1. **Edge steps**:
   - contiguous unconditional gate segments,
   - contiguous conditional-gate segments for one classical condition.
2. **Node steps**:
   - measurement barriers that may split path shot counts.

Runtime state is represented as:

1. `state_id -> state_vector` arena,
2. path counts keyed by `(state_id, classical_value)`.

This separates quantum state storage from classical-trajectory multiplicity.

## 3) Implementation

Implemented in `src/quantum/system.py`:

1. `_compile_dynamic_execution_graph(circuit)`
   - flattens circuit,
   - groups gates into `_DynamicSegmentStep` / `_DynamicConditionalStep`,
   - emits `_DynamicMeasurementStep` nodes,
   - records `node_count`,
   - compiles static and dynamic circuits into one graph format (static graphs have `node_count == 0`).
2. `_run_dynamic_branch_simulation(...)`
   - is the primary executor path from `run_simulation` for both static and dynamic circuits,
   - uses a zero-node fast path (`node_count == 0`) to execute edges once and then sink-sample terminal measurements,
   - uses node traversal with branch/path state for `node_count > 0`,
   - reuses edge transitions per unique source state (`transition_cache`),
   - splits at measurement with binomial shot allocation.
3. Measurement merge strategy:
   - structural caching per `(source_state_id, outcome)` within node,
   - compact state fingerprint merge bucketing (`_state_merge_signature`) to compress equivalent child states across parents.

## 4) Why This Is Better Structurally

Compared to operation-by-operation dynamic handling, this gives:

1. explicit separation between edge execution and node branching,
2. per-edge transition reuse for shared states,
3. one orchestration model for static and dynamic execution while preserving static sink-speed behavior,
4. a clear place to add future analyses:
   - measurement deferral (liveness),
   - selective branch materialization,
   - path pruning and merge policies.

## 5) Current Limits

Profiling (`trace_adaptive_feedback_120_10000_graphir.json`) shows remaining hotspots are still:

1. copy/transfer-heavy ops (`aten::copy_`, `aten::to`, `aten::_to_copy`),
2. scalar sync (`aten::item`, `aten::_local_scalar_dense`),
3. merge fingerprint overhead.

So the architecture is cleaner, but merge machinery is still expensive.

## 6) Next Pass (H3.1)

Highest-yield follow-up from this design:

1. Replace expensive amplitude fingerprinting with cheaper deterministic merge keys where safe.
2. Apply fingerprinting only when branch pressure exceeds threshold.
3. Add measurement-bit liveness analysis so unused classical bits do not force branch-state distinctions.
4. Batch gate application across states that share segment characteristics.

## 7) Validation Snapshot

Evidence artifacts:

1. Full suite post-refactor: `benchmarks/results/2026-02-11T021936.jsonl`
2. Dynamic-only compare, same code:
   - branch engine disabled: `benchmarks/results/2026-02-11T022010.jsonl`
   - graph branch engine enabled: `benchmarks/results/2026-02-11T021916.jsonl`
3. Trace: `benchmarks/results/trace_adaptive_feedback_120_10000_graphir.json`
