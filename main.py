from quantum import QuantumSystem, Circuit, gates, Measurement, ConditionalGate

print("Test Bell state correlation (should only see 00 or 11)")
bell_circuit = Circuit([
    gates.H(0),
    gates.CX([0, 1]),
    Measurement(0, 0),
    Measurement(1, 1),
    ConditionalGate(gates.H(2), 0),
    Measurement(2, 2),
])

counts = {"000": 0, "001": 0, "010": 0, "011": 0, "100": 0, "101": 0, "110": 0, "111": 0}
num_shots = 100

for _ in range(num_shots):
    sys = QuantumSystem(3, 3)
    sys = sys.apply_circuit(bell_circuit)
    key = "".join(map(str, sys.bit_register))
    counts[key] += 1

print(sys)
print(f"Results from {num_shots} shots:")
for state, count in counts.items():
    percentage = (count / num_shots) * 100
    print(f"  {state}: {count:4d} ({percentage:5.1f}%)")
print()
