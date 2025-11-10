from quantum import QuantumSystem, Circuit, Gate, gates
import torch

theta = torch.pi / 2

sys = QuantumSystem(2)
print(sys)
sys.apply_gate(gates.H, [0])
print(sys)
sys.apply_gate(gates.CX, [0, 1])
print(sys)
sys.apply_gate(gates.RY(theta), [0])
print(sys)

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
