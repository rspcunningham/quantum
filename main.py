from quantum import QuantumSystem, Circuit, gates, Measurement, ConditionalGate

print("Test Bell state correlation (should only see 00 or 11)")
circuit = Circuit([
    gates.H(0),
    gates.CX([0, 1]),
    Measurement(0, 0),
    Measurement(1, 1),
    #ConditionalGate(gates.H(2), 0),
    #Measurement(2, 2),
])

counts = {"00": 0, "01": 0, "10": 0, "11": 0}
num_shots = 100

for _ in range(num_shots):
    sys = QuantumSystem(2, 2)
    sys = sys.apply_circuit(circuit)
    key = "".join(map(str, sys.bit_register))
    counts[key] += 1

print(sys)
print(f"Results from {num_shots} shots:")
for state, count in counts.items():
    percentage = (count / num_shots) * 100
    print(f"  {state}: {count:4d} ({percentage:5.1f}%)")
print()
