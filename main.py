from quantum import QuantumSystem, gates

sys = QuantumSystem(2)
print(sys)
sys.apply_gate(gates.H, [0])
print(sys)
