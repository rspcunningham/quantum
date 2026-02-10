"""Grover's hash-preimage search: 13 qubits, deep oracle circuit."""

import math

from quantum import (
    Circuit,
    QuantumRegister,
    registers,
    H, X, I,
    CX,
    ControlledGateType,
    GateType,
    measure_all,
)
from benchmarks.cases import BenchmarkCase


def _xor(in_1: int, in_2: int, out: int) -> Circuit:
    return CX(in_1, out) + CX(in_2, out)


def _get_controller(
    n_controls: int, gate_type: GateType | ControlledGateType
) -> GateType | ControlledGateType:
    if n_controls == 0:
        return gate_type
    return _get_controller(n_controls - 1, ControlledGateType(gate_type))


def _get_if_qubitstring_gate(
    test_qubits: QuantumRegister, if_value: list[int], store_qubit: int
) -> Circuit:
    assert len(test_qubits) == len(if_value)

    def test_gate(v: int) -> GateType:
        return X if v == 0 else I

    test_gates = Circuit([test_gate(if_value[i])(test_qubits[i]) for i in range(len(test_qubits))])
    return (
        test_gates
        + _get_controller(len(test_qubits), X)(*test_qubits, store_qubit)
        + test_gates.inverse()
    )


def _classical_hash(input_bits: list[int]) -> list[int]:
    assert len(input_bits) == 4
    extra = [0] * 4
    h = [0] * 4
    extra[0] = input_bits[0] ^ input_bits[2]
    extra[1] = input_bits[1] ^ input_bits[3]
    extra[2] = input_bits[0] ^ input_bits[1]
    extra[3] = input_bits[2] ^ input_bits[3]
    h[0] = extra[0] ^ input_bits[3]
    h[1] = extra[1] ^ input_bits[0]
    h[2] = extra[2] ^ input_bits[2]
    h[3] = extra[3] ^ input_bits[1]
    return h


def _get_qhash(
    input_reg: QuantumRegister, working_reg: QuantumRegister, hash_reg: QuantumRegister
) -> Circuit:
    return (
        _xor(input_reg[0], input_reg[2], working_reg[0])
        + _xor(input_reg[1], input_reg[3], working_reg[1])
        + _xor(input_reg[0], input_reg[1], working_reg[2])
        + _xor(input_reg[2], input_reg[3], working_reg[3])
        + _xor(working_reg[0], input_reg[3], hash_reg[0])
        + _xor(working_reg[1], input_reg[0], hash_reg[1])
        + _xor(working_reg[2], input_reg[2], hash_reg[2])
        + _xor(working_reg[3], input_reg[1], hash_reg[3])
    )


def _get_oracle(
    input_reg: QuantumRegister,
    working_reg: QuantumRegister,
    hash_reg: QuantumRegister,
    target_hash: list[int],
    ancilla: int,
) -> Circuit:
    qhash = _get_qhash(input_reg, working_reg, hash_reg)
    return qhash + _get_if_qubitstring_gate(hash_reg, target_hash, ancilla) + qhash.inverse()


def _get_diffuser(input_reg: QuantumRegister, ancilla: int) -> Circuit:
    controller = _get_controller(len(input_reg), X)(*input_reg, ancilla)
    return H.on(input_reg) + X.on(input_reg) + controller + X.on(input_reg) + H.on(input_reg)


def _compute_grovers_expected(target_hash: list[int]) -> dict[str, float]:
    """Brute-force the classical hash to find preimages, compute Grover's expected distribution."""
    n_search = 4
    N = 1 << n_search

    preimages: list[str] = []
    for i in range(N):
        bits = [(i >> (n_search - 1 - b)) & 1 for b in range(n_search)]
        if _classical_hash(bits) == target_hash:
            preimages.append("".join(str(b) for b in bits))

    M = len(preimages)
    if M == 0:
        return {}

    iterations = math.floor(math.pi / 4 * math.sqrt(N / M))
    theta = math.asin(math.sqrt(M / N))
    success_prob = math.sin((2 * iterations + 1) * theta) ** 2
    per_solution = success_prob / M

    return {preimage: per_solution for preimage in preimages}


def real_grovers() -> BenchmarkCase:
    input_reg, working_reg, hash_reg, ancilla = registers(4, 4, 4, 1)
    anc = ancilla[0]
    target_hash = [0, 1, 1, 0]

    total_qubits = len(input_reg) + len(working_reg) + len(hash_reg) + len(ancilla)
    search_space = 1 << len(input_reg)
    iterations = math.floor(math.pi / 4 * math.sqrt(search_space))

    init = H.on(input_reg) + X(anc) + H(anc)
    oracle = _get_oracle(input_reg, working_reg, hash_reg, target_hash, anc)
    diffuser = _get_diffuser(input_reg, anc)

    circuit = init + (oracle + diffuser) * iterations + measure_all(input_reg)
    expected = _compute_grovers_expected(target_hash)

    return BenchmarkCase(
        name="real_grovers",
        circuit=circuit,
        expected=expected,
        n_qubits=total_qubits,
        tolerance=0.06,
    )
