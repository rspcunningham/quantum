# Quantum Simulator Code Review - TODO

## Mathematical Correctness

### ✅ **Gates Implementation (gates.py)**

**Correct implementations:**
- Identity, Hadamard, Pauli gates (X, Y, Z) are mathematically correct
- Phase gates (S, T) are correct
- CNOT (CX), CZ, SWAP are correct
- Toffoli (CCX) is correct
- Rotation gates (RX, RY, RZ) formulas are correct

**Issue with RZ gate (lines 128-131):**
```python
RZ = ParametricSingleQubitGate(lambda theta: torch.tensor(
    [[torch.cos(theta / 2) - 1j * torch.sin(theta / 2), 0],
    [0, torch.cos(theta / 2) + 1j * torch.sin(theta / 2)]],
    dtype=torch.complex64))
```
This should be:
```python
[[torch.exp(-1j * theta / 2), 0],
 [0, torch.exp(1j * theta / 2)]]
```
Your current implementation is mathematically equivalent but unnecessarily verbose. The standard form uses `exp(±iθ/2)`.

**Issue with Controlled gate (lines 168-180):**
```python
class ControlledGate:
    def __call__(self, gate_matrix: torch.Tensor, targets: list[int]) -> Gate:
        # ...
        controlled_matrix = torch.tensor(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, gate_matrix[0, 0], gate_matrix[0, 1]],
             [0, 0, gate_matrix[1, 0], gate_matrix[1, 1]]],
            dtype=torch.complex64)
```
**Problems:**
1. This only works for single-qubit gates, but there's no validation
2. Using `torch.tensor()` with gate_matrix elements breaks gradient tracking (if you ever want to differentiate through circuits)
3. Should use `torch.zeros()` + indexing or `torch.block_diag()`

### ✅ **State Evolution (system.py)**

**The swap-based gate application (lines 112-144) is mathematically sound:**
- Moving target qubits to leftmost positions via swaps
- Applying gate via Kronecker product
- Undoing swaps
- This correctly handles arbitrary target qubits

**Measurement implementation (lines 60-92) is correct:**
- Properly calculates probabilities using Born rule
- Correctly identifies basis states with qubit=|1⟩ using bit masking
- Projection operator is correct: `P = diag(1-outcome, outcome)`
- Normalization after measurement is correct

**Minor issue with endianness (lines 63, 73):**
```python
# In big-endian convention, qubit i corresponds to bit position (n_qubits - 1 - i).
```
You're using big-endian (qubit 0 is leftmost in |q0q1q2⟩), which is fine, but:
- The comment on line 73 says "python bit operations are little-endian" which is confusing
- Python's bit operations are actually big-endian when you use `>>`: the rightmost bit is bit 0
- Your conversion `bitpos = self.n_qubits - 1 - qubit` is correct for your convention

## DRY / Coding Best Practices

### 🔴 **Major: Repetitive Gate Definitions**

Lines 74-165 in gates.py contain extremely repetitive tensor definitions. Consider:

```python
# Current approach - lots of repetition
I = SingleQubitGate(torch.tensor([[1, 0], [0, 1]], dtype=torch.complex64))
X = SingleQubitGate(torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64))
# ... etc
```

**Suggestion:** Create a helper function:
```python
def _complex_matrix(data: list[list[complex | int | float]]) -> torch.Tensor:
    return torch.tensor(data, dtype=torch.complex64)

I = SingleQubitGate(_complex_matrix([[1, 0], [0, 1]]))
X = SingleQubitGate(_complex_matrix([[0, 1], [1, 0]]))
```

### 🔴 **Major: Inconsistent Gate API**

```python
# Single qubit: takes int
H(0)  # Returns Gate

# Two qubit: takes list
CX([0, 1])  # Returns Gate

# Parametric: takes tensor, then int
RX(torch.tensor(np.pi/4))(0)  # Returns Gate

# Controlled: completely different API
Controlled(H._matrix, [0, 1])  # Returns Gate - exposes internal _matrix!
```

**Problems:**
1. Inconsistent argument types (int vs list)
2. Parametric gates require double call
3. Controlled gate exposes private `_matrix` attribute
4. No way to do `CX(0, 1)` - must use list

**Suggestions:**
- Make all gates accept `*targets` for consistency: `CX(0, 1)` and `CX([0, 1])` both work
- Create a better API for controlled gates: `Controlled(H)(0, 1)` or `H.controlled([0, 1])`
- Consider making parametric gates simpler: maybe `RX(target=0, theta=np.pi/4)`

### 🟡 **Medium: Type Hints Could Be Stronger**

```python
# gates.py line 44
def __call__(self, targets: list[int]) -> Gate:
```

Use `Sequence[int]` to accept tuples too, or keep `list[int]` but document that tuples aren't supported.

```python
# system.py line 12
operations: list[Gate | ConditionalGate | Measurement | Circuit]
```

This is recursive - consider creating a type alias:
```python
Operation = Gate | ConditionalGate | Measurement | Circuit
operations: list[Operation]
```

### 🟡 **Medium: Magic Numbers**

```python
# system.py line 214
if magnitude < 1e-10:
    continue
```

Consider making thresholds configurable or documented constants:
```python
DISPLAY_THRESHOLD = 1e-10
```

### 🟢 **Good: Separation of Concerns**

The gate/system split is clean. Gate definitions are separate from simulation logic.

### 🟢 **Good: Validation**

Excellent validation throughout (lines 27-30, 45-47, 65-68, 98-110 in system.py).

## API Usability

### 🔴 **Major: Verbose Circuit Construction**

Current usage likely looks like:
```python
qs = QuantumSystem(3, 3)
qs.apply_quantum_gate(H._matrix, [0])  # Exposing _matrix is bad!
qs.apply_quantum_gate(CX._matrix, [0, 1])
```

Or with Circuit:
```python
circuit = Circuit([
    H(0),
    CX([0, 1]),
    Measurement(0, 0)
])
qs.apply_circuit(circuit)
```

**Issues:**
1. No fluent/chainable API
2. Circuit construction is verbose
3. No operator overloading for common patterns

**Suggestions:**

```python
# Option 1: Fluent API
qs = QuantumSystem(3, 3)
qs.H(0).CX(0, 1).measure(0, 0)

# Option 2: Circuit builder
circuit = Circuit()
circuit.H(0).CX(0, 1).measure(0, 0)
qs.apply(circuit)

# Option 3: Context manager
with QuantumSystem(3, 3) as qs:
    qs.H(0)
    qs.CX(0, 1)
    qs.measure(0, 0)
```

### 🔴 **Major: No Circuit Visualization/Inspection**

Consider adding:
```python
def __repr__(self) -> str:  # for Circuit
    # Return ASCII circuit diagram

def depth(self) -> int:
    # Return circuit depth

def gate_count(self) -> dict[str, int]:
    # Return gate statistics
```

### 🟡 **Medium: Measurement API is Confusing**

```python
qs.measure(qubit=0, output=0)
```

The `output` parameter is the classical register index. This naming is unclear. Consider:
```python
qs.measure(qubit=0, classical_bit=0)
# or
qs.measure(qubit=0, creg=0)
# or even
qs.measure_z(qubit=0, store_in=0)
```

Also, there's no measurement-without-storage option (just collapse the state without saving to classical register).

### 🟡 **Medium: No Basis Measurement Options**

Only Z-basis measurement is supported. Consider adding:
```python
qs.measure_x(qubit=0, output=0)  # Measure in X basis
qs.measure_y(qubit=0, output=0)  # Measure in Y basis
```

### 🟡 **Medium: ConditionalGate API**

```python
ConditionalGate(gate=H(0), classical_target=0)
```

This is reasonable but consider a more fluent API:
```python
H(0).if_classical(0)
# or
If(classical_bit=0).then(H(0))
```

### 🟢 **Good: State Representation**

The `__repr__` method (lines 196-249) is excellent! Clear, readable quantum state display.

### 🟢 **Good: Sampling Interface**

```python
samples = qs.sample(num_shots=1000)
```

Clean and intuitive.

## Additional Observations

### 🔴 **Gate Parameterization Type Inconsistency**

```python
# In gates.py
RX(theta: torch.Tensor) -> SingleQubitGate

# But users might expect to write:
RX(np.pi/4)(0)  # Will fail! Need torch.tensor()
```

Consider accepting `float | torch.Tensor` and converting internally.

### 🟡 **No Global Phase Tracking**

Quantum states have global phase freedom, but sometimes you want to track it (e.g., for controlled gates). This is fine for most applications but worth noting.

### 🟡 **Device Management**

```python
# system.py lines 41-45
self.device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)
```

**Issues:**
1. No way for user to specify device
2. Automatic MPS selection might not always be desired (it can be slower for small circuits)

**Suggestion:**
```python
def __init__(self, n_qubits: int, n_bits: int = 0,
             state_vector: torch.Tensor | None = None,
             device: str | torch.device | None = None):
    if device is None:
        device = torch.device(...)  # auto-select
    else:
        device = torch.device(device)
```

### 🟡 **Normalization Assertion**

```python
# system.py line 142
assert torch.allclose(norm, torch.tensor(1.0, device=self.device), atol=1e-5), f"Norm drift: {norm}"
```

**Issues:**
1. Assertions can be disabled with `python -O`
2. `1e-5` is quite loose for complex64
3. Creating a tensor just for comparison is wasteful

**Suggestion:**
```python
if not torch.allclose(norm, torch.ones(1, device=self.device), atol=1e-6):
    raise RuntimeError(f"Norm drift detected: {norm.item():.10f}")
```

### 🟢 **Swap Matrix Implementation**

The `_get_swap_matrix` method (lines 173-194) is clever and efficient. Nice work!

## Summary Recommendations

### High Priority:
 - [ ] 1. **Fix RZ gate** to use standard exponential form
 - [ ] 2. **Fix ControlledGate** to handle gradients and validate input
 - [ ] 3. **Unify gate API** - consistent calling convention across all gate types
 - [ ] 4. **Add fluent API** for circuit building
 - [ ] 5. **Improve measurement API** naming and add basis options

### Medium Priority:
 - [ ] 6. **DRY up gate definitions** with helper functions
 - [ ] 7. **Allow device specification** by user
 - [ ] 8. **Add circuit visualization** and inspection methods
 - [ ] 9. **Handle float/numpy inputs** in parametric gates
 - [ ] 10. **Better error messages** with helpful suggestions

### Low Priority (nice-to-have):
 - [ ] 11. Add circuit optimization methods (gate fusion, etc.)
 - [ ] 12. Add common circuit patterns (QFT, Grover, etc.)
 - [ ] 13. Add statevector fidelity/distance metrics
 - [ ] 14. Add density matrix support for mixed states
 - [ ] 15. Add circuit serialization (to/from QASM, JSON, etc.)
