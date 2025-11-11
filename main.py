from quantum import QuantumSystem, Circuit, gates

from quantum.system import Measurement

print("Test Bell state correlation (should only see 00 or 11)")
bell_circuit = Circuit([
    gates.H(0),
    gates.CX([0, 1]),
    Measurement(0, 0),
    Measurement(1, 1)
])

counts = {"00": 0, "01": 0, "10": 0, "11": 0}
num_shots = 1000

for _ in range(num_shots):
    sys = QuantumSystem(2, 2)
    sys = sys.apply_circuit(bell_circuit)
    key = "".join(map(str, sys.bit_register))
    counts[key] += 1

print(f"Results from {num_shots} shots:")
for state, count in counts.items():
    percentage = (count / num_shots) * 100
    print(f"  |{state}⟩: {count:4d} ({percentage:5.1f}%)")
print()
