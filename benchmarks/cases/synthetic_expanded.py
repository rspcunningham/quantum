"""Expanded synthetic benchmark families for broad simulator coverage."""

from __future__ import annotations

import cmath
import math
import random

import torch

from quantum import (
    Circuit,
    QuantumRegister,
    ControlledGateType,
    GateType,
    H,
    X,
    Y,
    Z,
    RX,
    RY,
    RZ,
    CX,
    CCX,
    Measurement,
    measure_all,
)
from benchmarks.cases import BenchmarkCase


GateDescriptor = tuple[str, tuple[int, ...], float]


def _phase_gate(phi: float) -> GateType:
    matrix = torch.tensor([[1, 0], [0, cmath.exp(1j * phi)]], dtype=torch.complex64)
    return GateType(matrix)


def _materialize_descriptors(
    qr: QuantumRegister,
    descriptors: list[GateDescriptor],
    *,
    inverse: bool = False,
) -> list:
    ordered = reversed(descriptors) if inverse else descriptors
    ops = []

    for tag, qubits, param in ordered:
        angle = -param if inverse and tag in {"rx", "ry", "rz", "cp"} else param

        if tag == "h":
            ops.append(H(qr[qubits[0]]))
        elif tag == "x":
            ops.append(X(qr[qubits[0]]))
        elif tag == "y":
            ops.append(Y(qr[qubits[0]]))
        elif tag == "z":
            ops.append(Z(qr[qubits[0]]))
        elif tag == "rx":
            ops.append(RX(angle)(qr[qubits[0]]))
        elif tag == "ry":
            ops.append(RY(angle)(qr[qubits[0]]))
        elif tag == "rz":
            ops.append(RZ(angle)(qr[qubits[0]]))
        elif tag == "cx":
            ops.append(CX(qr[qubits[0]], qr[qubits[1]]))
        elif tag == "ccx":
            ops.append(CCX(qr[qubits[0]], qr[qubits[1]], qr[qubits[2]]))
        elif tag == "cp":
            cp = ControlledGateType(_phase_gate(angle))
            ops.append(cp(qr[qubits[0]], qr[qubits[1]]))
        else:
            raise ValueError(f"Unsupported descriptor tag: {tag}")

    return ops


def _roundtrip_case_from_descriptors(
    *,
    name: str,
    n_qubits: int,
    descriptors: list[GateDescriptor],
    tolerance: float = 0.06,
) -> BenchmarkCase:
    qr = QuantumRegister(n_qubits)
    forward = Circuit(_materialize_descriptors(qr, descriptors, inverse=False))
    backward = Circuit(_materialize_descriptors(qr, descriptors, inverse=True))
    circuit = forward + backward + measure_all(qr)

    return BenchmarkCase(
        name=name,
        circuit=circuit,
        expected={"0" * n_qubits: 1.0},
        n_qubits=n_qubits,
        tolerance=tolerance,
    )


def _reversible_mix_descriptors(n_qubits: int, n_ops: int, *, seed: int) -> list[GateDescriptor]:
    rng = random.Random(seed)
    desc: list[GateDescriptor] = []

    for _ in range(n_ops):
        pick = rng.random()
        if pick < 0.2:
            q = rng.randrange(n_qubits)
            desc.append(("x", (q,), 0.0))
        elif pick < 0.78:
            c, t = rng.sample(range(n_qubits), 2)
            desc.append(("cx", (c, t), 0.0))
        else:
            a, b, t = rng.sample(range(n_qubits), 3)
            desc.append(("ccx", (a, b, t), 0.0))

    return desc


def _clifford_scrambler_descriptors(n_qubits: int, n_layers: int, *, seed: int) -> list[GateDescriptor]:
    rng = random.Random(seed)
    desc: list[GateDescriptor] = []

    one_qubit_tags = ("h", "x", "y", "z")

    for layer in range(n_layers):
        for q in range(n_qubits):
            desc.append((one_qubit_tags[rng.randrange(len(one_qubit_tags))], (q,), 0.0))

        start = layer % 2
        for q in range(start, n_qubits - 1, 2):
            if rng.random() < 0.9:
                if rng.random() < 0.5:
                    desc.append(("cx", (q, q + 1), 0.0))
                else:
                    desc.append(("cx", (q + 1, q), 0.0))

        if rng.random() < 0.35:
            c, t = rng.sample(range(n_qubits), 2)
            desc.append(("cx", (c, t), 0.0))

    return desc


def _brickwork_entangler_descriptors(n_qubits: int, n_layers: int) -> list[GateDescriptor]:
    desc: list[GateDescriptor] = []

    for layer in range(n_layers):
        if layer % 3 == 0:
            for q in range(0, n_qubits, 2):
                desc.append(("h", (q,), 0.0))
        elif layer % 3 == 1:
            for q in range(1, n_qubits, 2):
                desc.append(("x", (q,), 0.0))
        else:
            for q in range(0, n_qubits, 3):
                desc.append(("z", (q,), 0.0))

        start = layer % 2
        for q in range(start, n_qubits - 1, 2):
            desc.append(("cx", (q, q + 1), 0.0))

        if layer % 4 == 0 and n_qubits >= 6:
            desc.append(("cx", (0, n_qubits - 1), 0.0))

    return desc


def _random_universal_descriptors(n_qubits: int, n_layers: int, *, seed: int) -> list[GateDescriptor]:
    rng = random.Random(seed)
    desc: list[GateDescriptor] = []

    for layer in range(n_layers):
        for q in range(n_qubits):
            pick = rng.random()
            angle = rng.uniform(-math.pi, math.pi)
            if pick < 0.15:
                desc.append(("h", (q,), 0.0))
            elif pick < 0.45:
                desc.append(("rx", (q,), angle))
            elif pick < 0.75:
                desc.append(("ry", (q,), angle))
            else:
                desc.append(("rz", (q,), angle))

        qubits = list(range(n_qubits))
        rng.shuffle(qubits)
        for i in range(0, n_qubits - 1, 2):
            c = qubits[i]
            t = qubits[i + 1]
            if rng.random() < 0.75:
                desc.append(("cx", (c, t), 0.0))
            else:
                desc.append(("cx", (t, c), 0.0))

        if n_qubits >= 3 and layer % 3 == 0:
            a, b, t = rng.sample(range(n_qubits), 3)
            desc.append(("ccx", (a, b, t), 0.0))

    return desc


def _diagonal_mesh_descriptors(n_qubits: int, n_layers: int, *, seed: int) -> list[GateDescriptor]:
    rng = random.Random(seed)
    desc: list[GateDescriptor] = []

    for layer in range(n_layers):
        for q in range(n_qubits):
            base = 0.013 * (layer + 1) * (q + 1)
            jitter = rng.uniform(-0.2, 0.2)
            desc.append(("rz", (q,), base + jitter))

        pair_count = max(1, n_qubits // 3)
        for _ in range(pair_count):
            c, t = rng.sample(range(n_qubits), 2)
            desc.append(("cp", (c, t), rng.uniform(-0.35, 0.35)))

        if n_qubits >= 6:
            c = layer % n_qubits
            t = (layer * 5 + 3) % n_qubits
            if c != t:
                desc.append(("cp", (c, t), 0.07 * math.cos(layer + 1)))

    return desc


def reversible_mix_13() -> BenchmarkCase:
    return _roundtrip_case_from_descriptors(
        name="reversible_mix_13",
        n_qubits=13,
        descriptors=_reversible_mix_descriptors(13, 150, seed=1301),
        tolerance=0.06,
    )


def reversible_mix_15() -> BenchmarkCase:
    return _roundtrip_case_from_descriptors(
        name="reversible_mix_15",
        n_qubits=15,
        descriptors=_reversible_mix_descriptors(15, 190, seed=1501),
        tolerance=0.07,
    )


def clifford_scrambler_14() -> BenchmarkCase:
    return _roundtrip_case_from_descriptors(
        name="clifford_scrambler_14",
        n_qubits=14,
        descriptors=_clifford_scrambler_descriptors(14, 9, seed=1402),
        tolerance=0.06,
    )


def brickwork_entangler_15() -> BenchmarkCase:
    return _roundtrip_case_from_descriptors(
        name="brickwork_entangler_15",
        n_qubits=15,
        descriptors=_brickwork_entangler_descriptors(15, 14),
        tolerance=0.06,
    )


def random_universal_12() -> BenchmarkCase:
    return _roundtrip_case_from_descriptors(
        name="random_universal_12",
        n_qubits=12,
        descriptors=_random_universal_descriptors(12, 8, seed=1207),
        tolerance=0.08,
    )


def random_universal_14() -> BenchmarkCase:
    return _roundtrip_case_from_descriptors(
        name="random_universal_14",
        n_qubits=14,
        descriptors=_random_universal_descriptors(14, 9, seed=1407),
        tolerance=0.08,
    )


def diagonal_mesh_15() -> BenchmarkCase:
    return _roundtrip_case_from_descriptors(
        name="diagonal_mesh_15",
        n_qubits=15,
        descriptors=_diagonal_mesh_descriptors(15, 10, seed=1510),
        tolerance=0.07,
    )


def adaptive_feedback_5q() -> BenchmarkCase:
    qr = QuantumRegister(5)
    rounds = 50
    ops = []

    for _ in range(rounds):
        ops.append(H(qr[0]))
        ops.append(Measurement(qr[0], 0))
        ops.append(X(qr[0]).if_(0))

        for q in range(1, 5):
            ops.append(X(qr[q]).if_(1))
            ops.append(X(qr[q]).if_(1))

        ops.append(CX(qr[1], qr[2]).if_(1))
        ops.append(CX(qr[1], qr[2]).if_(1))

    ops.append(Measurement(qr[0], 0))

    return BenchmarkCase(
        name="adaptive_feedback_5q",
        circuit=Circuit(ops),
        expected={"1": 1.0},
        n_qubits=5,
        tolerance=0.02,
    )
