from quantum.gates import CX, X, I, Gate, GateType, ControlledGateType
from quantum import Circuit

MasterCX = ControlledGateType(ControlledGateType(ControlledGateType(CX)))

def xor(in_1: int, in_2: int, out: int) -> list[Gate]:
    return [CX(in_1, out), CX(in_2, out)]

q_hash = Circuit([
    *xor(0, 2, 4),
    *xor(1, 3, 5),
    *xor(0, 1, 6),
    *xor(2, 3, 7),
    *xor(4, 3, 8),
    *xor(5, 0, 9),
    *xor(6, 2, 10),
    *xor(7, 1, 11)
])


def uncompute(circuit: Circuit) -> Circuit:
    ops_list = circuit.operations
    return Circuit(list(reversed(ops_list)))

def get_if_qubitstring_gate(test_qubits: list[int], if_value: list[int], store_qubit) -> Gate:

    def gate_if_not(if_value: int) -> GateType:
        if target_bit == 0: return X
        return I

    return Circuit([
        gate_from_target_bit()
    ])

def get_oracle(target_hash: list[int]) -> Circuit:
    assert len(target_hash) == 4
    assert min(target_hash) == 0
    assert max(target_hash) == 1

    def gate_from_target_bit(target_bit: int) -> GateType:
        if target_bit == 0: return X
        return I

    return Circuit([
        q_hash,
        gate_from_target_bit(target_hash[0])(8),
        gate_from_target_bit(target_hash[1])(9),
        gate_from_target_bit(target_hash[2])(10),
        gate_from_target_bit(target_hash[3])(11),
        MasterCX(8, 9, 10, 11, 12), # uses 12 as our ancilla for the phase shift
        gate_from_target_bit(target_hash[0])(11),
        gate_from_target_bit(target_hash[1])(10),
        gate_from_target_bit(target_hash[2])(9),
        gate_from_target_bit(target_hash[3])(8),
        uncompute(q_hash)
    ])
