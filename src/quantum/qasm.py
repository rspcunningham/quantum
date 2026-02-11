"""QASM 2.0 parser → native Circuit."""
from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass

import torch

from quantum.gates import (
    Circuit, Gate, GateType, Measurement, ConditionalGate,
    H, X, Y, Z, S, T, CX, CCX, RX, RY, RZ,
)


@dataclass
class ParsedCircuit:
    circuit: Circuit
    n_qubits: int
    n_bits: int


# Pre-built gate types for non-standard-library gates
_Sdg = GateType(diagonal=torch.tensor([1, -1j], dtype=torch.complex64))
_Tdg = GateType(diagonal=torch.tensor([1, (1 - 1j) / math.sqrt(2)], dtype=torch.complex64))
_CZ = GateType(diagonal=torch.tensor([1, 1, 1, -1], dtype=torch.complex64))
_SWAP = GateType(torch.tensor(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
    dtype=torch.complex64,
))
_SX = GateType(torch.tensor(
    [[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=torch.complex64,
) / 2)


def _eval_param(expr: str) -> float:
    expr = expr.strip()
    tree = ast.parse(expr, mode='eval')
    return float(_safe_eval(tree.body))


def _safe_eval(node: ast.expr) -> float:
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return -_safe_eval(node.operand)
        if isinstance(node.op, ast.UAdd):
            return _safe_eval(node.operand)
    if isinstance(node, ast.BinOp):
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    if isinstance(node, ast.Name):
        if node.id == 'pi':
            return math.pi
        raise ValueError(f"Unknown name: {node.id}")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        arg = _safe_eval(node.args[0])
        fn_map = {
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'exp': math.exp, 'sqrt': math.sqrt, 'ln': math.log,
            'acos': math.acos, 'asin': math.asin, 'atan': math.atan,
        }
        if node.func.id in fn_map:
            return fn_map[node.func.id](arg)
    raise ValueError(f"Unsupported AST node: {ast.dump(node)}")


def _make_u3(theta: float, phi: float, lam: float) -> GateType:
    ct, st = math.cos(theta / 2), math.sin(theta / 2)
    return GateType(torch.tensor([
        [ct, -complex(math.cos(lam), math.sin(lam)) * st],
        [complex(math.cos(phi), math.sin(phi)) * st,
         complex(math.cos(phi + lam), math.sin(phi + lam)) * ct],
    ], dtype=torch.complex64))


def _make_u1(lam: float) -> GateType:
    return GateType(diagonal=torch.tensor(
        [1, complex(math.cos(lam), math.sin(lam))], dtype=torch.complex64))


def _make_cp(lam: float) -> GateType:
    return GateType(diagonal=torch.tensor(
        [1, 1, 1, complex(math.cos(lam), math.sin(lam))], dtype=torch.complex64))


def _make_cu3(theta: float, phi: float, lam: float) -> GateType:
    ct, st = math.cos(theta / 2), math.sin(theta / 2)
    u00 = complex(ct, 0)
    u01 = -complex(math.cos(lam), math.sin(lam)) * st
    u10 = complex(math.cos(phi), math.sin(phi)) * st
    u11 = complex(math.cos(phi + lam), math.sin(phi + lam)) * ct
    return GateType(torch.tensor([
        [1, 0, 0, 0], [0, 1, 0, 0],
        [0, 0, u00, u01], [0, 0, u10, u11],
    ], dtype=torch.complex64))


def _make_rzz(theta: float) -> GateType:
    e_neg = complex(math.cos(theta / 2), -math.sin(theta / 2))
    e_pos = complex(math.cos(theta / 2), math.sin(theta / 2))
    return GateType(diagonal=torch.tensor([e_neg, e_pos, e_pos, e_neg], dtype=torch.complex64))


def _make_rxx(theta: float) -> GateType:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return GateType(torch.tensor([
        [c, 0, 0, complex(0, -s)],
        [0, c, complex(0, -s), 0],
        [0, complex(0, -s), c, 0],
        [complex(0, -s), 0, 0, c],
    ], dtype=torch.complex64))


# Regex patterns
_QREG_RE = re.compile(r'qreg\s+(\w+)\s*\[(\d+)\]\s*;')
_CREG_RE = re.compile(r'creg\s+(\w+)\s*\[(\d+)\]\s*;')
_MEASURE_RE = re.compile(r'measure\s+(\w+)\[(\d+)\]\s*->\s*(\w+)\[(\d+)\]\s*;')
_IF_RE = re.compile(r'if\s*\(\s*(\w+)\s*==\s*(\d+)\s*\)\s*(.*)')
_GATE_RE = re.compile(r'(\w+)(?:\(([^)]*)\))?\s+(.*?)\s*;')
_QUBIT_RE = re.compile(r'(\w+)\[(\d+)\]')
_GATE_DEF_RE = re.compile(r'gate\s+(\w+)\s*(?:\(([^)]*)\))?\s+([\w\s,]+)\s*\{')


@dataclass
class _GateDef:
    param_names: list[str]
    qubit_names: list[str]
    body_lines: list[str]


def parse_qasm(qasm_string: str) -> ParsedCircuit:
    lines = qasm_string.strip().split('\n')

    qregs: dict[str, tuple[int, int]] = {}
    cregs: dict[str, tuple[int, int]] = {}
    gate_defs: dict[str, _GateDef] = {}
    total_qubits = 0
    total_bits = 0
    operations: list[Gate | ConditionalGate | Measurement] = []

    in_gate_def: str | None = None
    gate_def_body: list[str] = []
    gate_def_params: list[str] = []
    gate_def_qubits: list[str] = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith('//') or line.startswith('OPENQASM') or line.startswith('include'):
            continue

        # Collect gate definition bodies
        if in_gate_def is not None:
            if '}' in line:
                before_brace = line[:line.index('}')].strip()
                if before_brace:
                    gate_def_body.append(before_brace)
                gate_defs[in_gate_def] = _GateDef(gate_def_params, gate_def_qubits, gate_def_body)
                in_gate_def = None
                gate_def_body = []
            else:
                line_stripped = line.rstrip(';').strip()
                if line_stripped:
                    gate_def_body.append(line_stripped)
            continue

        # Start of gate definition
        m = _GATE_DEF_RE.match(line)
        if m:
            gname = m.group(1)
            pnames = [p.strip() for p in m.group(2).split(',')] if m.group(2) else []
            qnames = [q.strip() for q in m.group(3).strip().split(',')]
            # Single-line definition?
            rest = line[m.end():]
            if '}' in rest:
                body_text = rest[:rest.index('}')].strip()
                body = [s.strip() for s in body_text.split(';') if s.strip()]
                gate_defs[gname] = _GateDef(pnames, qnames, body)
            else:
                in_gate_def = gname
                gate_def_params = pnames
                gate_def_qubits = qnames
                gate_def_body = []
                body_on_same_line = line[m.end():].strip()
                if body_on_same_line:
                    for s in body_on_same_line.split(';'):
                        s = s.strip()
                        if s:
                            gate_def_body.append(s)
            continue

        if line.startswith('barrier'):
            continue

        m = _QREG_RE.match(line)
        if m:
            name, size = m.group(1), int(m.group(2))
            qregs[name] = (size, total_qubits)
            total_qubits += size
            continue

        m = _CREG_RE.match(line)
        if m:
            name, size = m.group(1), int(m.group(2))
            cregs[name] = (size, total_bits)
            total_bits += size
            continue

        m = _MEASURE_RE.match(line)
        if m:
            qubit = qregs[m.group(1)][1] + int(m.group(2))
            bit = cregs[m.group(3)][1] + int(m.group(4))
            operations.append(Measurement(qubit, bit))
            continue

        m = _IF_RE.match(line)
        if m:
            creg_name, val_str, rest = m.group(1), m.group(2), m.group(3)
            val = int(val_str)
            creg_size = cregs[creg_name][0]
            reversed_val = int(format(val, f'0{creg_size}b')[::-1], 2) if val > 0 else 0
            gate_ops = _parse_gate_line(rest.strip(), qregs, gate_defs)
            for op in gate_ops:
                if isinstance(op, Gate):
                    operations.append(op.if_(reversed_val))
            continue

        ops = _parse_gate_line(line, qregs, gate_defs)
        operations.extend(ops)

    return ParsedCircuit(circuit=Circuit(operations), n_qubits=total_qubits, n_bits=total_bits)


def _parse_gate_line(
    line: str,
    qregs: dict[str, tuple[int, int]],
    gate_defs: dict[str, _GateDef],
) -> list[Gate]:
    line_clean = line.rstrip(';').strip()
    if not line_clean:
        return []
    m = _GATE_RE.match(line_clean + ';')
    if not m:
        return []

    gate_name = m.group(1)
    params_str = m.group(2)
    qubits_str = m.group(3)

    if gate_name in ('barrier', 'id'):
        return []

    qubit_matches = _QUBIT_RE.findall(qubits_str)
    qubits: list[int] = []
    qubit_refs: list[str] = []
    for reg_name, idx_str in qubit_matches:
        qubits.append(qregs[reg_name][1] + int(idx_str))
        qubit_refs.append(f"{reg_name}[{idx_str}]")

    params: list[float] = []
    if params_str:
        for p in params_str.split(','):
            params.append(_eval_param(p))

    result = _make_gate(gate_name, params, qubits)
    if result is not None:
        return result

    # Fallback: expand gate definition
    if gate_name in gate_defs:
        return _expand_gate_def(gate_defs[gate_name], params, qubit_refs, qregs, gate_defs)

    raise ValueError(f"Unknown QASM gate: {gate_name}")


def _make_gate(name: str, params: list[float], qubits: list[int]) -> list[Gate] | None:
    if name == 'h':
        return [H(qubits[0])]
    if name == 'x':
        return [X(qubits[0])]
    if name == 'y':
        return [Y(qubits[0])]
    if name == 'z':
        return [Z(qubits[0])]
    if name == 's':
        return [S(qubits[0])]
    if name == 't':
        return [T(qubits[0])]
    if name == 'sdg':
        return [_Sdg(qubits[0])]
    if name == 'tdg':
        return [_Tdg(qubits[0])]
    if name == 'sx':
        return [_SX(qubits[0])]
    if name == 'rx':
        return [RX(params[0])(qubits[0])]
    if name == 'ry':
        return [RY(params[0])(qubits[0])]
    if name == 'rz':
        return [RZ(params[0])(qubits[0])]
    if name in ('u1', 'p'):
        return [_make_u1(params[0])(qubits[0])]
    if name == 'u2':
        return [_make_u3(math.pi / 2, params[0], params[1])(qubits[0])]
    if name in ('u3', 'u'):
        return [_make_u3(params[0], params[1], params[2])(qubits[0])]
    if name in ('cx', 'CX'):
        return [CX(qubits[0], qubits[1])]
    if name == 'cz':
        return [_CZ(qubits[0], qubits[1])]
    if name == 'swap':
        return [_SWAP(qubits[0], qubits[1])]
    if name in ('cp', 'cu1', 'cphase'):
        return [_make_cp(params[0])(qubits[0], qubits[1])]
    if name == 'ccx':
        return [CCX(qubits[0], qubits[1], qubits[2])]
    if name == 'cu3':
        return [_make_cu3(params[0], params[1], params[2])(qubits[0], qubits[1])]
    if name == 'rzz':
        return [_make_rzz(params[0])(qubits[0], qubits[1])]
    if name == 'rxx':
        return [_make_rxx(params[0])(qubits[0], qubits[1])]
    if name == 'cy':
        return [_make_cu3(math.pi, math.pi / 2, math.pi / 2)(qubits[0], qubits[1])]
    if name == 'ch':
        ct = 1 / math.sqrt(2)
        return [GateType(torch.tensor([
            [1, 0, 0, 0], [0, 1, 0, 0],
            [0, 0, ct, ct], [0, 0, ct, -ct],
        ], dtype=torch.complex64))(qubits[0], qubits[1])]
    if name == 'crz':
        e_neg = complex(math.cos(params[0] / 2), -math.sin(params[0] / 2))
        e_pos = complex(math.cos(params[0] / 2), math.sin(params[0] / 2))
        return [GateType(diagonal=torch.tensor(
            [1, 1, e_neg, e_pos], dtype=torch.complex64))(qubits[0], qubits[1])]
    return None


def _expand_gate_def(
    gate_def: _GateDef,
    actual_params: list[float],
    actual_qubit_refs: list[str],
    qregs: dict[str, tuple[int, int]],
    gate_defs: dict[str, _GateDef],
) -> list[Gate]:
    param_map = dict(zip(gate_def.param_names, actual_params))
    qubit_map = dict(zip(gate_def.qubit_names, actual_qubit_refs))
    ops: list[Gate] = []
    for body_line in gate_def.body_lines:
        expanded = body_line
        # Replace parameters with evaluated values
        for pname, pval in param_map.items():
            expanded = re.sub(rf'\b{re.escape(pname)}\b', repr(pval), expanded)
        # Replace qubit names with actual references
        for qname, qref in qubit_map.items():
            expanded = re.sub(rf'\b{re.escape(qname)}\b', qref, expanded)
        ops.extend(_parse_gate_line(expanded, qregs, gate_defs))
    return ops
