from quantum import Circuit, QuantumRegister, run_simulation, registers, H, X, I, CX, ControlledGateType, GateType, measure_all, plot_results
import math
import time

#######################
# Utilities
#######################

def xor(in_1: int, in_2: int, out: int) -> Circuit:
    return CX(in_1, out) + CX(in_2, out)

def get_controller(n_controls: int, gate_type: GateType | ControlledGateType) -> GateType | ControlledGateType:
    if n_controls == 0: return gate_type
    return get_controller(n_controls - 1, ControlledGateType(gate_type))

def get_if_qubitstring_gate(test_qubits: QuantumRegister, if_value: list[int], store_qubit: int) -> Circuit:
    assert len(test_qubits) == len(if_value)

    def test_gate(if_value: int) -> GateType:
        if if_value == 0: return X
        return I

    test_gates = Circuit([test_gate(if_value[i])(test_qubits[i]) for i in range(len(test_qubits))])

    return test_gates + get_controller(len(test_qubits), X)(*test_qubits, store_qubit) + test_gates.inverse()

#######################
# Functions
#######################

def classical_hash(input_bits: list[int]) -> list[int]:
    assert len(input_bits) == 4
    extra_bits = [0] * 4
    hash_bits = [0] * 4

    extra_bits[0] = input_bits[0] ^ input_bits[2]
    extra_bits[1] = input_bits[1] ^ input_bits[3]
    extra_bits[2] = input_bits[0] ^ input_bits[1]
    extra_bits[3] = input_bits[2] ^ input_bits[3]

    hash_bits[0] = extra_bits[0] ^ input_bits[3]
    hash_bits[1] = extra_bits[1] ^ input_bits[0]
    hash_bits[2] = extra_bits[2] ^ input_bits[2]
    hash_bits[3] = extra_bits[3] ^ input_bits[1]

    return hash_bits

def get_qhash(input_reg: QuantumRegister, working_reg: QuantumRegister, hash_reg: QuantumRegister) -> Circuit:
    return (
        xor(input_reg[0], input_reg[2], working_reg[0])
        + xor(input_reg[1], input_reg[3], working_reg[1])
        + xor(input_reg[0], input_reg[1], working_reg[2])
        + xor(input_reg[2], input_reg[3], working_reg[3])
        + xor(working_reg[0], input_reg[3], hash_reg[0])
        + xor(working_reg[1], input_reg[0], hash_reg[1])
        + xor(working_reg[2], input_reg[2], hash_reg[2])
        + xor(working_reg[3], input_reg[1], hash_reg[3])
    )

def get_oracle(
    input_reg: QuantumRegister,
    working_reg: QuantumRegister,
    hash_reg: QuantumRegister,
    target_hash: list[int],
    ancilla: int
) -> Circuit:
    assert len(target_hash) == len(hash_reg)
    qhash = get_qhash(input_reg, working_reg, hash_reg)
    return qhash + get_if_qubitstring_gate(hash_reg, target_hash, ancilla) + qhash.inverse()

def get_diffuser(input_reg: QuantumRegister, ancilla: int) -> Circuit:
    controller = get_controller(len(input_reg), X)(*input_reg, ancilla)
    return H.on(input_reg) + X.on(input_reg) + controller + X.on(input_reg) + H.on(input_reg)

#######################
# Implementation
#######################

input_reg, working_reg, hash_reg, ancilla = registers(4, 4, 4, 1)
anc = ancilla[0]

target_hash = [0, 1, 1, 0]

total_qubits = len(input_reg) + len(working_reg) + len(hash_reg) + len(ancilla)
search_space = 1 << len(input_reg)
iterations = math.floor(math.pi / 4 * math.sqrt(search_space))

init = H.on(input_reg) + X(anc) + H(anc)
oracle = get_oracle(input_reg, working_reg, hash_reg, target_hash, anc)
diffuser = get_diffuser(input_reg, anc)

circuit = init + (oracle + diffuser) * iterations + measure_all(input_reg)

print(f"Applying {iterations} iterations")

start_time = time.time()
result = run_simulation(circuit, 100, n_qubits=total_qubits)
end_time = time.time()

print(f"time: {end_time - start_time}")

most_likely = max(result, key=lambda x: result[x])
most_likely = bits = [int(c) for c in most_likely]
print(f"Most likely solution: {most_likely}")
hashed_most_likely = classical_hash(most_likely)
print(f"Hashed most likely solution: {hashed_most_likely}")
print(f"Target hash: {target_hash} | Solution hash: {hashed_most_likely} | Match: {hashed_most_likely == target_hash}")
_ = plot_results(result)
