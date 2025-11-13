from quantum import QuantumSystem, Circuit
from quantum.gates import H, CX, RY, Measurement
import math

qc = Circuit([
    H(0),
    CX(0, 1),
    RY(math.pi/2)(0),
    Measurement(0, 0),
    Measurement(1, 1)
])

counts = {"00": 0, "01": 0, "10": 0, "11": 0}
num_shots = 100

for _ in range(num_shots):
    sys = QuantumSystem(2, 2)
    sys = sys.apply_circuit(qc)
    key = "".join(map(str, sys.bit_register))
    counts[key] += 1

print(sys)
print(f"Results from {num_shots} shots:")
for state, count in counts.items():
    percentage = (count / num_shots) * 100
    print(f"  {state}: {count:4d} ({percentage:5.1f}%)")
print()
