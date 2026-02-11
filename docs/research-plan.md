# Research Plan

## Status Update (2026-02-10)

- ✅ **H1/H2 completed**: tensor-contraction gate path landed (commit `86fb8c5`).
- ✅ **H3 completed**: measurement collapse vectorized in batched path (commit `587a03f`), with follow-up improvements in commits `eb476df` and `0c3186d`.
- ⏳ **H4 pending**: batched gate path no longer renormalizes per gate; single-shot `QuantumSystem` still does.
- ⏳ **H5 ongoing**: MPS dispatch/sync overhead remains visible after core algorithmic fixes.

Current measured status (M1 Max, MPS):
- Baseline after H1/H2 (`3df121d`): total 1000-shot benchmark `370.99s`
- Current (`0c3186d`): total 1000-shot benchmark `6.41s` (`57.9x` faster)

Detailed run-by-run numbers: `docs/optimization-progress-2026-02-10.md`

## H1: Full matrix construction dominates — the GPU is mostly idle

The current code builds a complete 2^n × 2^n matrix for every gate, even single-qubit gates. For 13 qubits, applying a 2×2 Hadamard gate means:
 1. _gate_to_qubit: chain of torch.kron calls building up to 8192×8192
 2. _get_swap_matrix: Python for loop over 8192 iterations to build another 8192×8192 matrix
 3. A dense 8192×8192 matmul to apply it

This is O(4^n) per gate when it should be O(2^n).

Confirming data: cpu_util ≈ 1.0 for real_grovers and ghz_state (CPU is the bottleneck, not GPU). In the profiler, aten::kron and Python overhead should dominate. The actual aten::mm (the useful work) should be a small fraction of total time.

Strategy: Reshape state vectors from (batch, 2^n) to (batch, 2, 2, ..., 2) and apply gates via torch.einsum or torch.tensordot directly on the target qubit dimensions. No Kronecker products, no swap matrices. A single-qubit gate becomes a contraction of a 2×2 tensor with one axis of the state tensor — O(2^n) instead of O(4^n).

---

## H2: Swap matrix construction is the worst single function

```python
_get_swap_matrix builds an 8192×8192 permutation matrix using a Python for-loop:
for x in range(dim):       # 8192 iterations
S[y, x] = 1            # one element at a time
```

This isn't even a torch operation — it's pure Python setting individual tensor elements. And it's called for every gate whose target qubits aren't already in positions 0,1,...,k. For a circuit with 180 gates (real_grovers), most need at least one swap.

Confirming data: The profiler won't directly show this since it's Python overhead, not a torch op. But if we see large gaps between torch ops in the Chrome trace, or if Self CPU time on non-torch functions is high, that's the signal. Also: real_grovers at 1 shot takes 140s for 184 ops — that's ~0.76s per op, which is absurdly slow for what should be a matrix multiply.

Strategy: Eliminated entirely by H1's tensor contraction approach. If we wanted a half-measure, we could vectorize the swap matrix construction using torch.arange and bitwise ops instead of the Python loop — but this is polishing the wrong abstraction.

---

## H3: The measurement loop is the batch-scaling bottleneck

```python
BatchedQuantumSystem.apply_measurement loops over every shot in Python:
for i in range(self.batch_size):      # 1000 iterations at 1000 shots
P_full = self._gate_to_qubit(P, ...)  # builds full 2^n × 2^n projection
self.state_vectors[i] = self.state_vectors[i] @ P_full.T
```

Each iteration constructs a full projection matrix and does a matmul. At 1000 shots with 13 qubits, that's 1000 × (8192×8192 matrix construction + matmul) per measurement.

Confirming data: Compare the 1→1000 shot scaling ratio across cases. From the last run:
 - bell_state: 0.007→1.83 (261x) — 2 measurements dominate the tiny circuit
 - real_grovers: 140→527 (3.8x) — gate application dominates, measurements are noise
 - ghz_state: 2.2→161 (73x) — 12 measurements on a shallow 12-gate circuit

The measurement loop's impact scales with measurements × batch_size / total_ops. For shallow circuits with many measured qubits, it's the main bottleneck at high shot counts.

Strategy: Vectorize: instead of looping over batch elements, compute projections for all outcomes in one batched operation. With tensor contraction, projection becomes zeroing out one index of the qubit dimension — no matrix construction at all.

---

## H4: Per-gate renormalization is pure waste

Both gate application methods compute norm = sqrt(sum(abs(sv)^2)) and divide after every gate. Unitary gates preserve norms by definition. This is O(batch × 2^n) per gate — cheap compared to the matmuls, but entirely unnecessary.

Confirming data: aten::sum, aten::sqrt, aten::abs, aten::div appearing in the profiler for every gate. Their total time as a fraction of overall time tells us the magnitude.

Strategy: Remove renormalization from gate application entirely. Only normalize before measurements (where floating point drift could affect outcome probabilities). This is a one-line change with zero risk — if it saves even 2%, it's free.

---

## H5: MPS dispatch overhead compounds across many small ops

Every torch call on MPS tensors goes through Python → PyTorch dispatch → Metal. With hundreds of gates, each requiring ~n kron calls + swap matmuls + the gate matmul + normalization, there are thousands of tiny Metal dispatches per circuit execution.

Confirming data: If after implementing tensor contraction (H1) we see that wall time is still higher than expected from the raw compute, and the profiler shows many ops with very small self-times but large cumulative overhead, that's dispatch latency.

Strategy: This becomes relevant after H1. Possible mitigations: gate fusion (combine consecutive single-qubit gates on the same qubit into one), circuit compilation (pre-compute the sequence of tensor contractions), or moving the circuit execution loop into a single torch operation.

---

## Prioritization

H1 and H2 are the same fix (tensor contraction) and represent the vast majority of the problem. I'd estimate 10-100x speedup on real_grovers from this alone. H3 is the second priority — matters most for high shot counts on shallow circuits. H4 is a quick free win. H5 is a later-stage concern.
