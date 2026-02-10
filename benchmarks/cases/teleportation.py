"""Quantum teleportation: 3 qubits. Tests mid-circuit measurement and conditional gates.

Teleports |1> from qubit 0 to qubit 2 via a Bell pair on qubits 1,2.
After Bell measurement on qubits 0,1 and conditional corrections on qubit 2,
qubit 2 should always measure as 1.

The conditional gate API checks the full classical register as an integer.
With 3 classical bits, the register value is 4*bit0 + 2*bit1 + bit2.
At the point of corrections, bit2=0, so the register values for each
measurement outcome (bit0=A, bit1=M) are: 00->0, 01->2, 10->4, 11->6.
"""

from quantum import Circuit, QuantumRegister, H, X, Z, CX, Measurement, measure_all
from benchmarks.cases import BenchmarkCase


def teleportation() -> BenchmarkCase:
    qr = QuantumRegister(3)

    circuit = Circuit([
        # Prepare qubit 0 in |1>
        X(qr[0]),
        # Create Bell pair on qubits 1,2
        H(qr[1]),
        CX(qr[1], qr[2]),
        # Bell measurement on qubits 0,1
        CX(qr[0], qr[1]),
        H(qr[0]),
        Measurement(qr[0], 0),
        Measurement(qr[1], 1),
        # Conditional corrections on qubit 2
        # (register values account for 3-bit register with bit2=0)
        X(qr[2]).if_(2),   # M=1 (bit1 set): register = 0*4 + 1*2 + 0 = 2
        X(qr[2]).if_(6),   # A=1,M=1: register = 1*4 + 1*2 + 0 = 6
        Z(qr[2]).if_(4),   # A=1 (bit0 set): register = 1*4 + 0*2 + 0 = 4
        Z(qr[2]).if_(6),   # A=1,M=1: register = 1*4 + 1*2 + 0 = 6
        # Measure teleported qubit
        Measurement(qr[2], 2),
    ])

    # Bit 2 (teleported qubit) is always 1; bits 0,1 are uniformly random
    return BenchmarkCase(
        name="teleportation",
        circuit=circuit,
        expected={"001": 0.25, "011": 0.25, "101": 0.25, "111": 0.25},
        tolerance=0.06,
    )
