from quantum import Circuit, QuantumSystem
from quantum.gates import H, CX


circuit = Circuit([
    H(0),
    CX(0, 1)
])

qs = QuantumSystem(2)

_ = qs.apply_circuit(circuit)

result = qs.get_distribution()
print(result)

import matplotlib.pyplot as plt
# Plot probabilities
n_qubits = 2
basis_states = [f"|{bin(i)[2:].zfill(n_qubits)}⟩" for i in range(2**n_qubits)]
probabilities = result.squeeze().cpu().numpy()

_ = plt.figure(figsize=(10, 6))
_ = plt.bar(basis_states, probabilities)
_ = plt.xlabel('Computational Basis State')
_ = plt.ylabel('Probability')
_ = plt.title('Quantum State Probability Distribution')
_ = plt.ylim([0, 1])
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
