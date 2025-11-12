# Quantum Simulator

A pedagogical quantum circuit simulator built from first principles using PyTorch.

## What is this?

A pure state-vector quantum simulator that exposes the underlying linear algebra without abstractions. Built for learning how quantum computing actually works at the mathematical level.

## Features

- **Pure implementation** - No external quantum libraries, just PyTorch tensors
- **Full state vector simulation** - Explicit Kronecker products and unitary evolution
- **Hybrid quantum-classical** - Quantum registers + classical bit storage with conditional gates
- **Device flexibility** - Runs on CPU, CUDA, or MPS
- **Clear visualization** - Human-readable quantum state representation

## Quick Start

```python
from quantum import QuantumSystem, Circuit, gates, Measurement

# Create a 2-qubit system with 2 classical bits
qs = QuantumSystem(n_qubits=2, n_bits=2)

# Build a Bell state circuit
circuit = Circuit([
    gates.H(0),           # Hadamard on qubit 0
    gates.CX([0, 1]),     # CNOT from qubit 0 to 1
    Measurement(0, 0),    # Measure qubit 0 → classical bit 0
    Measurement(1, 1),    # Measure qubit 1 → classical bit 1
])

# Run the circuit
qs.apply_circuit(circuit)
print(qs)  # See the quantum state in Dirac notation
```

## Available Gates

**Single-qubit:** I, H, X, Y, Z, S, T

**Parametric:** RX(θ), RY(θ), RZ(θ)

**Two-qubit:** CX, CZ, SWAP

**Three-qubit:** CCX (Toffoli)

**General:** Controlled(gate), ConditionalGate(gate, classical_bit)

**Measurement:** Measurement(qubit, classical_bit)

## Purpose

This is an educational tool for understanding quantum mechanics and quantum computing fundamentals. It's intentionally unoptimized to make the mathematics transparent.

**Use this to:** Learn how quantum gates work, prototype small algorithms, teach quantum computing

**Don't use this for:** Large-scale simulations, production code, performance-critical applications

## Requirements

- Python 3.10+
- PyTorch
- NumPy
