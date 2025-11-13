# Quantum Simulator

A pedagogical quantum circuit simulator built from first principles using PyTorch.

## What is this?

A pure state-vector quantum simulator that exposes the underlying linear algebra without abstractions. Built for learning how quantum computing actually works at the mathematical level.

## Features

- **Pure implementation** - No external quantum libraries, just PyTorch tensors
- **Full state vector simulation** - Explicit Kronecker products and unitary evolution
- **GPU-accelerated batch simulation** - Run thousands of shots in parallel
- **Hybrid quantum-classical** - Quantum registers + classical bit storage with conditional gates
- **Device flexibility** - Automatic selection of CUDA, MPS, or CPU
- **Clear visualization** - Dirac notation state representation + matplotlib plotting
- **Type-safe** - Full type annotations with jaxtyping for tensor shapes

## Quick Start

### Single Shot Simulation

```python
from quantum import QuantumSystem, Circuit
from quantum.gates import H, CX, Measurement

# Create a 2-qubit system with 2 classical bits
qs = QuantumSystem(n_qubits=2, n_bits=2)

# Build a Bell state circuit
circuit = Circuit([
    H(0),                 # Hadamard on qubit 0
    CX(0, 1),             # CNOT: control=0, target=1
    Measurement(0, 0),    # Measure qubit 0 → classical bit 0
    Measurement(1, 1),    # Measure qubit 1 → classical bit 1
])

# Run the circuit
qs.apply_circuit(circuit)
print(qs)  # See the quantum state in Dirac notation
```

### Multi-Shot Simulation with Visualization

```python
from quantum import QuantumSystem, Circuit, run_simulation
from quantum.visualization import plot_results
from quantum.gates import H, CX, RY, Measurement
import math

# Create circuit
circuit = Circuit([
    H(0),
    CX(0, 1),
    RY(math.pi/4)(0),     # Parametric gate: angle first, then target
    Measurement(0, 0),
    Measurement(1, 1)
])

# Run 1000 shots with GPU acceleration
initial_system = QuantumSystem(n_qubits=2, n_bits=2)
counts = run_simulation(initial_system, circuit, num_shots=1000)

# Visualize results
plot_results(counts, title="Bell State Measurement")

# Print distribution
for state, count in sorted(counts.items()):
    print(f"{state}: {count} ({count/10:.1f}%)")
```

## Available Gates

### Single-Qubit Gates
**Predefined:** `I`, `H`, `X`, `Y`, `Z`, `S`, `T`

```python
from quantum.gates import H, X, Y, Z, S, T
H(0)  # Apply Hadamard to qubit 0
```

### Parametric Rotation Gates
**Rotations:** `RX(θ)`, `RY(θ)`, `RZ(θ)`

```python
from quantum.gates import RX, RY, RZ
import math
RY(math.pi/2)(0)  # First call with angle, then with target qubit

# Alternatively, define a gate for the specific angle needed
RYT = RY(math.pi/2)
RYT(0)
```

### Multi-Qubit Gates
**Predefined:** `CX` (CNOT), `CCX` (Toffoli)

```python
from quantum.gates import CX, CCX
CX(0, 1)     # Control: qubit 0, Target: qubit 1
CCX(0, 1, 2) # Controls: qubits 0,1, Target: qubit 2
```

### Creating Controlled Gates
**Any gate can be made controlled:**

```python
from quantum.gates import ControlledGateType, Z, Y
CZ = ControlledGateType(Z)  # Create controlled-Z
CY = ControlledGateType(Y)  # Create controlled-Y
CZ(0, 1)  # Apply controlled-Z
```

### Conditional Gates
**Execute gates based on classical bits:**

```python
from quantum.gates import X, H
X(0).if_(0)      # Apply X to qubit 0 only if classical bit register == 0 (ie 00000)
H(1).if_(2)      # Apply H to qubit 1 only if classical bit register == 2 (ie 00010)
```

### Measurements
**Collapse quantum state to classical:**

```python
from quantum import Measurement
Measurement(qubit_index, classical_bit_index)
```

## Core API

### QuantumSystem

Main class for single quantum state simulation.

```python
qs = QuantumSystem(n_qubits=3, n_bits=3)
qs.apply_gate(H(0))                    # Apply single gate
qs.apply_measurement(Measurement(0, 0)) # Measure and collapse
qs.apply_circuit(circuit)               # Apply entire circuit
print(qs)                               # Pretty-print state
probs = qs.get_distribution()           # Get probability distribution
samples = qs.sample(num_shots=100)      # Sample from current state
```

### Circuit

Container for quantum operations.

```python
from quantum import Circuit
circuit = Circuit([
    H(0),
    CX(0, 1),
    X(2).if_(0),         # Conditional gate
    Measurement(1, 1),
    Circuit([...])        # Circuits can be nested
])
```

### run_simulation

High-level function for GPU-accelerated multi-shot simulation.

```python
from quantum import run_simulation

counts = run_simulation(
    initial_system=QuantumSystem(n_qubits=2, n_bits=2),
    circuit=circuit,
    num_shots=1000
)
# Returns: {'00': 503, '11': 497, ...}
```

### Visualization

Plot measurement results with matplotlib.

```python
from quantum.visualization import plot_results

fig, ax = plot_results(
    results=counts,
    title="Measurement Results",
    show=True  # Display immediately
)
```

## How It Works

This simulator implements quantum mechanics using linear algebra:

1. **State vectors**: Complex-valued tensors of shape (2^n, 1)
2. **Gates**: Unitary matrices applied via Kronecker products
3. **Measurements**: Projection operators with probabilistic collapse
4. **Batching**: Parallel simulation of multiple shots on GPU

All operations preserve unitarity and maintain normalized states. The implementation uses big-endian qubit ordering (qubit 0 is leftmost bit).

## Purpose

This is an educational tool for understanding quantum mechanics and quantum computing fundamentals. The implementation prioritizes clarity over performance.

**Use this to:**
- Learn how quantum gates work mathematically
- Prototype small quantum algorithms
- Teach quantum computing concepts
- Understand state vector simulation

**Don't use this for:**
- Large-scale simulations (>14 qubits depending on the size of your gpu)
- Production code
- Performance benchmarking

## Requirements

- Python 3.13+
- PyTorch >= 2.9.0
- NumPy >= 2.3.4
- Matplotlib >= 3.9.2
- Seaborn >= 0.13.2

## Installation

Using uv is reccomended:
```bash
uv sync
```

to run the demo script:
```bash
uv run main.py
```
