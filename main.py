from quantum import QuantumSystem, Circuit, gates
from quantum.system import Measurement

print("Test Bell state correlation (should only see 00 or 11)")
bell_circuit = Circuit([
    gates.H(0),
    gates.CX([0, 1]),
    Measurement(0, 0),
    Measurement(1, 1),
    Measurement(2, 2),
    Measurement(3, 3)
])

counts = {"0000": 0, "0001": 0, "0010": 0, "0011": 0, "0100": 0, "0101": 0, "0110": 0, "0111": 0, "1000": 0, "1001": 0, "1010": 0, "1011": 0, "1100": 0, "1101": 0, "1110": 0, "1111": 0}
num_shots = 100

for _ in range(num_shots):
    sys = QuantumSystem(4, 4)
    sys = sys.apply_circuit(bell_circuit)
    key = "".join(map(str, sys.bit_register))
    counts[key] += 1

print(sys)
print(f"Results from {num_shots} shots:")
for state, count in counts.items():
    percentage = (count / num_shots) * 100
    print(f"  {state}: {count:4d} ({percentage:5.1f}%)")
print()
