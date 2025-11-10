from quantum import QuantumSystem, Circuit, Gate, gates
import torch

from quantum.system import Measurement

theta = torch.tensor(torch.pi / 2)

print("------------------------------------------------------------------------")

sys = QuantumSystem(2)
print(sys)
c = Circuit([
        Gate(gates.H, [0]),
        Gate(gates.CX, [0, 1]),
        Gate(gates.RY(theta), [0]),
        Measurement([1, 0])
    ])

result = sys.apply_circuit(c)
print(result)

print("------------------------------------------------------------------------")

for i in range(10):
      sys = QuantumSystem(2)
      result = sys.apply_circuit(c)
      print(f"Run {i}: {result}")
