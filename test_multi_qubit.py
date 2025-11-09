from system import QuantumSystem
from gates import *

print("=== Testing general apply_gate method ===\n")

# Test 1: Single-qubit gates still work
print("Test 1: Single-qubit Hadamard on qubit 0")
state = QuantumSystem(2)
state.apply_gate(H, [0])
print(state)
print()

# Test 2: CNOT gate on qubits (0, 1) - control=0, target=1
print("Test 2: CNOT with control=0, target=1")
state = QuantumSystem(2)
state.apply_gate(H, [0])  # Create superposition on qubit 0
print("After H on qubit 0:", state)
state.apply_gate(CNOT, [0, 1])  # Entangle qubits
print("After CNOT(0,1):", state)
print("Note: This creates a Bell state |00⟩ + |11⟩")
print()

# Test 3: CNOT on non-adjacent qubits
print("Test 3: CNOT with control=0, target=2 (non-adjacent)")
state = QuantumSystem(3)
state.apply_gate(H, [0])
print("After H on qubit 0:", state)
state.apply_gate(CNOT, [0, 2])  # Control on 0, target on 2
print("After CNOT(0,2):", state)
print()

# Test 4: SWAP gate
print("Test 4: SWAP gate")
state = QuantumSystem(2)
state.apply_gate(X, [0])  # Set qubit 0 to |1⟩
print("After X on qubit 0:", state)
state.apply_gate(SWAP, [0, 1])
print("After SWAP(0,1):", state)
print("Note: Now qubit 1 is |1⟩ and qubit 0 is |0⟩")
print()

# Test 5: Creating entanglement across multiple qubits
print("Test 5: Creating GHZ state (|000⟩ + |111⟩)")
state = QuantumSystem(3)
state.apply_gate(H, [0])
state.apply_gate(CNOT, [0, 1])
state.apply_gate(CNOT, [0, 2])
print(state)
print()

# Test 6: Measuring an entangled state
print("Test 6: Sampling from Bell state")
state = QuantumSystem(2)
state.apply_gate(H, [0])
state.apply_gate(CNOT, [0, 1])
print("State:", state)
samples = state.sample(1000)
count_00 = sum(1 for s in samples if s == 0)  # |00⟩
count_11 = sum(1 for s in samples if s == 3)  # |11⟩
count_other = sum(1 for s in samples if s not in [0, 3])
print(f"Samples: |00⟩: {count_00}, |11⟩: {count_11}, other: {count_other}")
print("Expected: ~500 for |00⟩ and |11⟩, ~0 for others")
