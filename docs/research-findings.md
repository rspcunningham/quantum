# Research Findings: Optimization Techniques for State-Vector Quantum Simulation

Compiled during baseline profiling run.

---

## Update: Implemented Results (2026-02-10)

The highest-priority recommendations in this document have now been implemented and benchmarked:

1. Tensor contraction gate application (H1/H2) — already landed before this session.
2. Vectorized measurement collapse (H3) — implemented in `BatchedQuantumSystem`.
3. Gate tensor device cache — avoids repeated `tensor.to(device)` in hot path.
4. Cached measurement probability weights — replaces boolean-index sum with matmul.

Measured impact on full benchmark suite (`1000` shots, M1 Max, MPS):
- Baseline (commit `3df121d`): `370.99s`
- Current (commit `0c3186d`): `6.41s`
- Net: `57.9x` faster with correctness checks still passing.

Detailed timing tables and profiler snapshots are tracked in:
`docs/optimization-progress-2026-02-10.md`

---

## 1. Tensor Contraction (replaces Kronecker products + swap matrices)

**The core problem:** Our simulator builds a full `2^n x 2^n` matrix for every gate via `torch.kron`, then applies it with dense matmul. For 13 qubits that's 8192x8192 per gate — O(4^n) when it should be O(2^n).

**The fix:** Reshape the state vector from `(batch, 2^n)` to `(batch, 2, 2, ..., 2)` — one axis per qubit. Apply gates by contracting only along the target qubit dimensions.

### Complexity comparison

| Operation | Current (Kronecker) | Tensor Contraction |
|---|---|---|
| 1-qubit gate, n qubits | O(4^n) build + O(4^n) matmul | O(2^n) contraction |
| 2-qubit gate, n qubits | O(4^n) build + O(4^n) matmul | O(2 * 2^n) contraction |
| Swap routing | O(4^n) per swap | **Eliminated entirely** |
| Memory per gate | O(4^n) | O(4) for 1q, O(16) for 2q |

For 13 qubits: ~8000x reduction in work per gate.

### Implementation: einsum approach

```python
def apply_gate_einsum(state, gate, targets, n_qubits):
    """General gate application via einsum. Any number of target qubits."""
    k = len(targets)
    gate_reshaped = gate.reshape([2] * (2 * k))

    # Build einsum string dynamically
    batch_char = 'Z'
    state_chars = [chr(ord('a') + i) for i in range(n_qubits)]
    gate_in_chars = [state_chars[t] for t in targets]
    gate_out_chars = [chr(ord('a') + n_qubits + i) for i in range(k)]

    gate_str = ''.join(gate_out_chars + gate_in_chars)
    state_str = batch_char + ''.join(state_chars)

    result_chars = list(state_chars)
    for i, t in enumerate(targets):
        result_chars[t] = gate_out_chars[i]
    result_str = batch_char + ''.join(result_chars)

    subscripts = f"{gate_str},{state_str}->{result_str}"
    return torch.einsum(subscripts, gate_reshaped, state)
```

### Implementation: tensordot approach (safer on MPS)

```python
def apply_gate_tensordot(state, gate, targets, n_qubits):
    """MPS-friendly gate application using tensordot + permute."""
    k = len(targets)
    gate_reshaped = gate.reshape([2] * (2 * k))

    non_targets = [i for i in range(n_qubits) if i not in targets]
    perm_to_end = non_targets + list(targets)
    state = state.permute([0] + [p + 1 for p in perm_to_end])  # batch dim first

    # Contract gate input axes with state's last k axes
    contract_gate_axes = list(range(k, 2 * k))
    contract_state_axes = list(range(n_qubits - k + 1, n_qubits + 1))
    state = torch.tensordot(gate_reshaped, state, dims=(contract_gate_axes, contract_state_axes))
    # First k dims are gate outputs, then batch, then non-targets

    # Permute back to (batch, qubit_0, qubit_1, ..., qubit_{n-1})
    inv_perm = [0] * (n_qubits + 1)
    inv_perm[0] = k  # batch dim
    for i, t in enumerate(targets):
        inv_perm[t + 1] = i
    for i, nt in enumerate(non_targets):
        inv_perm[nt + 1] = k + 1 + i
    state = state.permute(inv_perm)

    return state
```

### Note on einsum vs tensordot on MPS

- `einsum` is cleaner but may fall back to CPU on MPS for complex patterns
- `tensordot` reduces to reshape + matmul + reshape internally, which is well-supported on MPS
- **Recommendation:** try einsum first, benchmark both, fall back to tensordot if needed

### References

- PennyLane `default.qubit` — uses `np.tensordot` + `np.transpose` (most readable reference impl)
- Cirq `state_vector_simulation_state.py` — uses `np.einsum`
- Qulacs (arXiv:2011.13524) — describes "update function" approach
- Google qsim — aggressive gate fusion + tensor contraction

---

## 2. Vectorized Measurement (replaces Python loop + projection matrices)

**The core problem:** `apply_measurement` loops over each batch element in Python, building a full `2^n x 2^n` projection matrix for each, then doing matmul. At 1000 shots on 13 qubits: 1000 x (8192x8192 construction + matmul).

**The fix:** Projection is just zeroing out amplitudes where the qubit has the "wrong" value. No matrix needed — use a broadcasted boolean mask.

### Flat-vector approach (drop-in replacement)

```python
def apply_measurement(self, measurement):
    qubit = measurement.qubit
    bit = measurement.bit
    bitpos = self.n_qubits - 1 - qubit

    indices = torch.arange(1 << self.n_qubits, device=self.device)
    mask_1 = ((indices >> bitpos) & 1).bool()  # (2^n,)

    probs = torch.abs(self.state_vectors) ** 2
    p1 = probs[:, mask_1].sum(dim=1)

    outcomes = (torch.rand(self.batch_size, device=self.device) < p1).int()
    self.bit_registers[:, bit] = outcomes

    # Vectorized projection: no loop, no matrix
    outcomes_col = outcomes.unsqueeze(1).bool()       # (batch, 1)
    mask_1_row = mask_1.unsqueeze(0)                  # (1, 2^n)
    keep = (mask_1_row == outcomes_col)                # (batch, 2^n)

    self.state_vectors = self.state_vectors * keep

    norms = torch.sqrt((torch.abs(self.state_vectors) ** 2).sum(dim=1, keepdim=True))
    self.state_vectors = self.state_vectors / norms
    return self
```

### Tensor-form approach (if state is `(batch, 2, 2, ..., 2)`)

```python
# Measurement on qubit k = slicing axis k+1 in the batched tensor
# Build a selector of shape (batch, 1, ..., 2, ..., 1) with 1 at measured outcome
selector = torch.zeros(batch_size, 2, device=device, dtype=state.dtype)
selector[torch.arange(batch_size), outcomes] = 1.0

view_shape = [1] * (n_qubits + 1)
view_shape[0] = batch_size
view_shape[qubit + 1] = 2
selector = selector.view(view_shape)

state = state * selector  # zeros out wrong branch, O(batch * 2^n)
```

### Complexity

| Approach | Cost | Python Loop? | Matrix Built? |
|---|---|---|---|
| Current code | O(batch * 4^n) | Yes | Yes, 2^n x 2^n per element |
| Flat mask + broadcast | O(batch * 2^n) | No | No |
| Tensor-form selector | O(batch * 2^n) | No | No |

---

## 3. Additional Optimization Tricks

### Diagonal gate specialization

Gates like Z, S, T, RZ, controlled-phase are diagonal — they scale amplitudes without mixing basis states. These can be applied as elementwise multiply, even cheaper than tensor contraction:

```python
def apply_diagonal_gate(state_flat, diagonal_values, target, n_qubits):
    bitpos = n_qubits - 1 - target
    indices = torch.arange(1 << n_qubits, device=state_flat.device)
    qubit_val = (indices >> bitpos) & 1
    phases = diagonal_values[qubit_val]
    return state_flat * phases
```

### SWAP gates are free in tensor form

A SWAP between qubits a and b is just `state.permute(...)` — changes tensor strides, no data copy.

### Gate fusion

Consecutive single-qubit gates on the same qubit can be fused by multiplying their 2x2 matrices before applying. Reduces number of tensor contractions. Google's qsim does this aggressively.

### Skip renormalization after unitary gates

Unitary gates preserve norm by definition. Only renormalize before measurements. Current code renormalizes after every gate — pure waste. (Quick free win, identified as H4 in research-plan.md.)

### Gate tensor caching

Move gate tensors to device once and cache the reshaped `(2,)*2k` versions to avoid repeated `.to(device).reshape(...)` calls.

---

## 4. MPS Backend (Metal Performance Shaders) — PyTorch 2.9

Verified against the installed PyTorch 2.9.0 source.

### Native vs composite operations

| Operation | Native MPS kernel? | Notes |
|---|---|---|
| `mm`, `bmm`, `addmm` | **Yes** | Core matmul — fast path |
| Element-wise (`add`, `mul`, `abs`, `sqrt`) | **Yes** | |
| `view`, `reshape`, `permute` | **Yes** (metadata-only) | |
| FFT (`_fft_c2c`, etc.) | **Yes** | |
| `view_as_complex`/`view_as_real` | **Yes** (zero-copy) | |
| `torch.kron` | **No** — composite | Decomposes to reshape + broadcast multiply. Each call in a loop generates multiple dispatches. |
| `torch.einsum` | **No** — composite | Decomposes to permute + reshape + mm. The mm core hits native kernel. |
| `torch.tensordot` | **No** — composite | Same as einsum — reshape + mm + reshape. |

**Key implication:** `einsum`/`tensordot` are fine because their inner `mm` is native. The extra reshape/permute overhead is small relative to the matmul. `torch.kron` in a loop (our current `_gate_to_qubit`) is much worse — each iteration creates growing intermediate tensors with multiple dispatches.

### MPS dispatch overhead

- Each PyTorch op translates to an MPSGraph operation, compiled and dispatched as a Metal command buffer
- No automatic multi-operation fusion at dispatch level (unlike CUDA kernel fusion)
- **`.item()` and `.cpu()` force GPU-CPU sync barriers** — our measurement loop calls `.item()` per batch element, serializing all GPU work
- MPS naturally batches commands — the GPU works ahead of the CPU as long as you don't force sync
- Tensor allocation has higher overhead on MPS than CUDA — avoid repeated small allocations in hot paths

### complex64 support

- Metal Shading Language has **no native complex type**
- `mm` with complex inputs decomposes internally: `(a+bi)(c+di) = (ac-bd) + (ad+bc)i` using real matmuls
- `complex128` is **not supported** on MPS (Metal lacks float64) — our complex64 usage is correct
- For maximum performance, could manually represent state as `float32` with `(..., 2)` for real/imag — avoids internal decomposition overhead. Significant refactoring, only worth it if benchmarks show complex decomposition is a bottleneck.

### torch.compile on MPS (Inductor)

- Available in PyTorch 2.9 via `torch/_inductor/codegen/mps.py`
- Generates Metal shader source code for fused pointwise/reduction kernels
- **Early prototype** — only handles elementwise ops and some reductions, NOT matmul or kron
- Could help with normalization / probability calculation steps
- The `DTYPE_TO_METAL` map has no complex entry — complex ops cannot be directly represented in compiled path

### Useful MPS APIs

- `torch.mps.synchronize()` — full CPU-GPU sync barrier. Already used in our benchmark timing. Avoid in hot loops.
- `torch.mps.Event` — record GPU-side timing without full sync. Supports `elapsed_time()`.
- `torch.mps.profiler.metal_capture()` — GPU trace capture for Xcode Instruments analysis
- `torch.mps.compile_shader()` — write custom Metal compute shaders callable from Python. Could be used for fused kernels later.
- `torch.mps.current_allocated_memory()` / `driver_allocated_memory()` — already used in our benchmark

---

## 5. Implementation Priority

1. **Tensor contraction for gates** (H1+H2) — reshape state, einsum/tensordot gate application. Eliminates Kronecker products + swap matrices. Expected 100-8000x per gate.
2. **Vectorized measurement** (H3) — broadcast mask instead of loop + projection matrix. Expected 10-100x for measurement-heavy circuits at high shot counts.
3. **Remove per-gate renormalization** (H4) — one-line change, 2-5% free win.
4. **Gate fusion** (H5) — compile-time optimization, 1.5-3x for deep circuits.
5. **Diagonal gate specialization** — further constant-factor improvement.
