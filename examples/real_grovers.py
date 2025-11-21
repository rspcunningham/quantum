from quantum import QuantumSystem, Circuit, run_simulation
from quantum.gates import H, X, I, CX, Measurement, ControlledGateType, GateType
from quantum.visualization import plot_results
import math
import time

#######################
# Utilities
#######################

def xor(in_1: int, in_2: int, out: int) -> Circuit:
    return Circuit([CX(in_1, out), CX(in_2, out)])

def get_controller(n_controls: int, gate_type: GateType | ControlledGateType) -> GateType | ControlledGateType:
    if n_controls == 0: return gate_type
    return get_controller(n_controls - 1, ControlledGateType(gate_type))

def get_if_qubitstring_gate(test_qubits: list[int], if_value: list[int], store_qubit: int) -> Circuit:

    assert len(test_qubits) == len(if_value)

    def test_gate(if_value: int) -> GateType:
        if if_value == 0: return X
        return I

    test_gates = Circuit([test_gate(if_value[i])(test_qubits[i]) for i in range(len(test_qubits))])

    return Circuit([
        test_gates,
        get_controller(len(test_qubits), X)( *test_qubits, store_qubit ),
        test_gates.uncomputed()
    ])

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

def get_qhash(input_register: list[int], working_register: list[int], hash_register: list[int]) -> Circuit:

     return Circuit([
        xor(input_register[0], input_register[2], working_register[0]),
        xor(input_register[1], input_register[3], working_register[1]),
        xor(input_register[0], input_register[1], working_register[2]),
        xor(input_register[2], input_register[3], working_register[3]),
        xor(working_register[0], input_register[3], hash_register[0]),
        xor(working_register[1], input_register[0], hash_register[1]),
        xor(working_register[2], input_register[2], hash_register[2]),
        xor(working_register[3], input_register[1], hash_register[3])
    ])

def get_init(input_register: list[int], ancilla: int) -> Circuit:
    ops_list = [ *[H(i) for i in input_register], X(ancilla), H(ancilla)]
    return Circuit(ops_list)

def get_oracle(
    input_register: list[int],
    working_register: list[int],
    hash_register: list[int],
    target_hash: list[int],
    ancilla: int
) -> Circuit:
    assert len(target_hash) == len(hash_register)

    qhash = get_qhash(input_register, working_register, hash_register)

    return Circuit([
        qhash,
        get_if_qubitstring_gate(hash_register, target_hash, ancilla),
        qhash.uncomputed()
    ])

def get_diffuser(input_register: list[int]) -> Circuit:
    h_list = [H(i) for i in input_register]
    x_list = [X(i) for i in input_register]
    controller = get_controller(len(input_register), X)( *input_register, len(input_register) )

    return Circuit(h_list + x_list + [controller] + x_list + h_list)

def get_measure_all(qubits_to_measure: list[int]) -> Circuit:
    ops_list = [Measurement(n, i) for i, n in enumerate(qubits_to_measure)]
    return Circuit(ops_list)

#######################
# Implementation
#######################

# 4 bit search register
# 4 bit hash intermediate register
# 4 bit hash output register
# 1 bit ancilla

input_register = [0, 1, 2, 3]
working_register = [4, 5, 6, 7]
hash_register = [8, 9, 10, 11]

target_hash = [0, 1, 1, 0]

total_qubits = len(input_register) + len(working_register) + len(hash_register) + 1

search_space = 1 << len(input_register) # == 2^len(input_register)
iterations = math.floor(math.pi / 4 * math.sqrt(search_space))

init = get_init(input_register, 12)
oracle = get_oracle(input_register, working_register, hash_register, target_hash, 12)
diffuser = get_diffuser(input_register)
measurement = get_measure_all(input_register)

grover_iters = [oracle, diffuser] * iterations

circuit = Circuit([
    init,
    *grover_iters,
    measurement
])

qs = QuantumSystem(total_qubits, len(input_register))

print(f"Applying {iterations} iterations")

start_time = time.time()
result = run_simulation(qs, circuit, 100)
end_time = time.time()

print(f"time: {end_time - start_time}")

most_likely = max(result, key=lambda x: result[x])
most_likely = bits = [int(c) for c in most_likely]
print(f"Most likely solution: {most_likely}")
hashed_most_likely = classical_hash(most_likely)
print(f"Hashed most likely solution: {hashed_most_likely}")
print(f"Target hash: {target_hash} | Solution hash: {hashed_most_likely} | Match: {hashed_most_likely == target_hash}")
_ = plot_results(result)
