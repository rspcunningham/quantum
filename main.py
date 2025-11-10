from quantum import QuantumSystem, Circuit, Gate, gates
import torch

theta = torch.tensor(torch.pi / 2)

print("------------------------------------------------------------------------")

sys = QuantumSystem(2)
print(sys)
c = Circuit(
    n_qubits=2,
    gates=[
        Gate(gates.H, [0]),
        Gate(gates.CX, [0, 1]),
        Gate(gates.RY(theta), [0])
    ])

sys.apply_circuit(c)
print(sys)

print("------------------------------------------------------------------------")
sys = QuantumSystem(2)
sys.apply_gate(gates.H, [0])     # Should put qubit 0 in superposition
print(sys)
