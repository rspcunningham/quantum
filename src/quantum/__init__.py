from typing import TYPE_CHECKING

from quantum.gates import (
    Gate,
    GateType,
    ParametricGateType,
    CustomGateType,
    ConditionalGate,
    Measurement,
    QuantumRegister,
    registers,
    Circuit,
    H, X, Y, Z, S, Sdg, T, Tdg, SX, I,
    CX, CZ, CCX, SWAP,
    RX, RY, RZ, CP,
)
from quantum.metal_exec import CompiledCircuit, compile
from quantum.system import measure_all, infer_resources

if TYPE_CHECKING:
    from quantum.qasm import ParsedCircuit, parse_qasm


def __getattr__(name: str):
    if name == "parse_qasm":
        from quantum.qasm import parse_qasm

        return parse_qasm
    if name == "ParsedCircuit":
        from quantum.qasm import ParsedCircuit

        return ParsedCircuit
    raise AttributeError(f"module 'quantum' has no attribute {name!r}")


__all__ = [
    # Core
    "Circuit", "CompiledCircuit", "compile", "measure_all", "infer_resources",
    # Registers
    "QuantumRegister", "registers",
    # Gates
    "H", "X", "Y", "Z", "S", "Sdg", "T", "Tdg", "SX", "I",
    "CX", "CZ", "CCX", "SWAP",
    "RX", "RY", "RZ", "CP",
    # Gate types
    "Gate", "GateType", "ParametricGateType", "CustomGateType", "ConditionalGate", "Measurement",
    # QASM
    "parse_qasm", "ParsedCircuit",
]
