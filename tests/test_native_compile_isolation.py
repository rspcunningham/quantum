from __future__ import annotations

import os
import math
import unittest

import torch
import quantum_native_runtime as native

from quantum import metal_exec
from quantum.gates import Circuit, ConditionalGate, Gate, H, Measurement, RZ, X


class NativeCompileIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        module_path = getattr(native, "__file__", "")
        if module_path:
            native.set_module_file_path(module_path)

        override_path = os.environ.get("QUANTUM_METAL_METALLIB")
        if override_path:
            native.set_metallib_path_override(override_path)

    def _compile(self, circuit: Circuit, n_qubits: int, n_bits: int) -> int:
        handle = int(native.compile_static_circuit(circuit, n_qubits, n_bits))
        self.addCleanup(native.free_program, handle)
        return handle

    def test_dynamic_rejected(self) -> None:
        circuit = Circuit([
            H(0),
            Measurement(0, 0),
            ConditionalGate(X(0), 1),
            Measurement(0, 0),
        ])
        with self.assertRaisesRegex(RuntimeError, "Dynamic circuits are temporarily unsupported"):
            native.compile_static_circuit(circuit, 1, 1)

    def test_dense_arity_over_6_rejected(self) -> None:
        dense = torch.eye(1 << 7, dtype=torch.complex64)
        s = 1.0 / math.sqrt(2.0)
        dense[0, 0] = complex(s, 0.0)
        dense[0, 1] = complex(s, 0.0)
        dense[1, 0] = complex(s, 0.0)
        dense[1, 1] = complex(-s, 0.0)
        gate = Gate(dense, 0, 1, 2, 3, 4, 5, 6)
        circuit = Circuit([gate])
        with self.assertRaisesRegex(RuntimeError, "Dense gates with arity > 6"):
            native.compile_static_circuit(circuit, 7, 0)

    def test_inverse_cancellation_reduces_to_zero_ops(self) -> None:
        circuit = Circuit([
            H(0),
            H(0),
            Measurement(0, 0),
        ])
        handle = self._compile(circuit, 1, 1)
        stats = native.get_program_stats(handle)
        self.assertEqual(int(stats["op_count"]), 0)

    def test_local_diagonal_compaction(self) -> None:
        circuit = Circuit([
            RZ(0.17)(0),
            RZ(-0.23)(0),
            RZ(0.11)(0),
            Measurement(0, 0),
        ])
        handle = self._compile(circuit, 1, 1)
        stats = native.get_program_stats(handle)
        self.assertEqual(int(stats["op_count"]), 1)

    def test_monomial_stream_packing_and_execution(self) -> None:
        old_disable = os.environ.get("QUANTUM_DISABLE_MONOMIAL_STREAM")
        old_min = os.environ.get("QUANTUM_MONOMIAL_MIN_RUN")
        try:
            os.environ["QUANTUM_DISABLE_MONOMIAL_STREAM"] = "0"
            os.environ["QUANTUM_MONOMIAL_MIN_RUN"] = "8"

            circuit = Circuit([
                X(0), X(1), X(2), X(3), X(4), X(5), X(6), X(7),
                Measurement(0, 0),
                Measurement(1, 1),
                Measurement(2, 2),
                Measurement(3, 3),
                Measurement(4, 4),
                Measurement(5, 5),
                Measurement(6, 6),
                Measurement(7, 7),
            ])
            handle = self._compile(circuit, 8, 8)
            stats = native.get_program_stats(handle)
            self.assertEqual(int(stats["op_count"]), 1)
            self.assertEqual(int(stats["monomial_spec_count"]), 1)

            counts = native.execute_static_program(handle, 64, 1234)
            self.assertEqual(int(sum(counts.values())), 64)
            self.assertEqual(int(counts.get("11111111", 0)), 64)
        finally:
            if old_disable is None:
                os.environ.pop("QUANTUM_DISABLE_MONOMIAL_STREAM", None)
            else:
                os.environ["QUANTUM_DISABLE_MONOMIAL_STREAM"] = old_disable
            if old_min is None:
                os.environ.pop("QUANTUM_MONOMIAL_MIN_RUN", None)
            else:
                os.environ["QUANTUM_MONOMIAL_MIN_RUN"] = old_min

    def test_dispatch_grouping_compile_only(self) -> None:
        old_disable = os.environ.get("QUANTUM_DISABLE_MONOMIAL_STREAM")
        try:
            os.environ["QUANTUM_DISABLE_MONOMIAL_STREAM"] = "1"
            circuit = Circuit([
                X(0),
                X(1),
                X(2),
                Measurement(0, 0),
                Measurement(1, 1),
                Measurement(2, 2),
            ])
            handle = self._compile(circuit, 3, 3)
            stats = native.get_program_stats(handle)
            self.assertEqual(int(stats["op_count"]), 3)
            self.assertEqual(int(stats["dispatch_count"]), 1)
        finally:
            if old_disable is None:
                os.environ.pop("QUANTUM_DISABLE_MONOMIAL_STREAM", None)
            else:
                os.environ["QUANTUM_DISABLE_MONOMIAL_STREAM"] = old_disable

    def test_grouped_permutation_runtime_execution(self) -> None:
        old_disable = os.environ.get("QUANTUM_DISABLE_MONOMIAL_STREAM")
        try:
            os.environ["QUANTUM_DISABLE_MONOMIAL_STREAM"] = "1"
            circuit = Circuit([
                X(0),
                X(1),
                X(2),
                Measurement(0, 0),
                Measurement(1, 1),
                Measurement(2, 2),
            ])
            handle = self._compile(circuit, 3, 3)
            stats = native.get_program_stats(handle)
            self.assertEqual(int(stats["op_count"]), 3)
            self.assertEqual(int(stats["dispatch_count"]), 1)

            counts = native.execute_static_program(handle, 64, 123)
            self.assertEqual(int(sum(counts.values())), 64)
            self.assertEqual(int(counts.get("111", 0)), 64)
        finally:
            if old_disable is None:
                os.environ.pop("QUANTUM_DISABLE_MONOMIAL_STREAM", None)
            else:
                os.environ["QUANTUM_DISABLE_MONOMIAL_STREAM"] = old_disable

    def test_grouped_diagonal_runtime_execution(self) -> None:
        old_disable = os.environ.get("QUANTUM_DISABLE_MONOMIAL_STREAM")
        try:
            os.environ["QUANTUM_DISABLE_MONOMIAL_STREAM"] = "1"
            circuit = Circuit([
                H(0),
                H(1),
                RZ(math.pi)(0),
                RZ(math.pi)(1),
                H(0),
                H(1),
                Measurement(0, 0),
                Measurement(1, 1),
            ])
            handle = self._compile(circuit, 2, 2)
            stats = native.get_program_stats(handle)
            self.assertEqual(int(stats["op_count"]), 6)
            self.assertEqual(int(stats["dispatch_count"]), 5)

            counts = native.execute_static_program(handle, 64, 321)
            self.assertEqual(int(sum(counts.values())), 64)
            self.assertEqual(int(counts.get("11", 0)), 64)
        finally:
            if old_disable is None:
                os.environ.pop("QUANTUM_DISABLE_MONOMIAL_STREAM", None)
            else:
                os.environ["QUANTUM_DISABLE_MONOMIAL_STREAM"] = old_disable

    def test_execute_static_circuit_uses_native_packed_run_entrypoint(self) -> None:
        import quantum.metal_compile as metal_compile

        original_native_run = getattr(native, "run_static_circuit")
        original_native_run_packed = getattr(native, "run_static_packed")
        original_native_compile = getattr(native, "compile_static_circuit")
        original_canonical_compile = getattr(native, "compile_static_canonical")
        original_py_compile = metal_compile.compile_to_metal_program

        calls = {"native_run": 0, "native_run_packed": 0, "native_compile": 0, "canonical_compile": 0}

        def _wrapped_native_run(
            circuit: object,
            n_qubits: int,
            n_bits: int,
            num_shots: int,
            seed: object,
        ) -> dict[str, int]:
            calls["native_run"] += 1
            return dict(original_native_run(
                circuit,
                n_qubits,
                n_bits,
                num_shots,
                seed,
            ))

        def _wrapped_native_run_packed(
            payload: bytes,
            num_shots: int,
            seed: object,
        ) -> dict[str, int]:
            calls["native_run_packed"] += 1
            return dict(original_native_run_packed(payload, num_shots, seed))

        def _native_compile_should_not_run(*_args: object, **_kwargs: object) -> int:
            calls["native_compile"] += 1
            raise RuntimeError("compile_static_circuit should not run in Python wrapper path")

        def _canonical_compile_should_not_run(*_args: object, **_kwargs: object) -> int:
            calls["canonical_compile"] += 1
            raise RuntimeError("compile_static_canonical should not run in native run-static path")

        def _python_compile_should_not_run(*_args: object, **_kwargs: object) -> object:
            raise RuntimeError("compile_to_metal_program should not run in native compiler-core path")

        native.run_static_circuit = _wrapped_native_run  # type: ignore[assignment]
        native.run_static_packed = _wrapped_native_run_packed  # type: ignore[assignment]
        native.compile_static_circuit = _native_compile_should_not_run  # type: ignore[assignment]
        native.compile_static_canonical = _canonical_compile_should_not_run  # type: ignore[assignment]
        metal_compile.compile_to_metal_program = _python_compile_should_not_run  # type: ignore[assignment]
        self.addCleanup(setattr, native, "run_static_circuit", original_native_run)
        self.addCleanup(setattr, native, "run_static_packed", original_native_run_packed)
        self.addCleanup(setattr, native, "compile_static_circuit", original_native_compile)
        self.addCleanup(setattr, native, "compile_static_canonical", original_canonical_compile)
        self.addCleanup(setattr, metal_compile, "compile_to_metal_program", original_py_compile)

        circuit = Circuit([
            H(0),
            Measurement(0, 0),
        ])
        counts = metal_exec.execute_static_circuit(
            circuit,
            n_qubits=1,
            n_bits=1,
            num_shots=32,
            seed=7,
        )
        self.assertEqual(int(sum(counts.values())), 32)
        self.assertEqual(int(calls["native_run"]), 0)
        self.assertEqual(int(calls["native_run_packed"]), 1)
        self.assertEqual(int(calls["native_compile"]), 0)
        self.assertEqual(int(calls["canonical_compile"]), 0)

    def test_circuit_static_payload_reused_until_gate_changes(self) -> None:
        gate = H(0)
        circuit = Circuit([
            gate,
            Measurement(0, 0),
        ])
        payload_a = circuit.build_native_static_payload(n_qubits=1, n_bits=1)
        payload_b = circuit.build_native_static_payload(n_qubits=1, n_bits=1)
        self.assertIs(payload_a, payload_b)

        gate.tensor = X(0).tensor.clone()
        payload_c = circuit.build_native_static_payload(n_qubits=1, n_bits=1)
        self.assertNotEqual(payload_a, payload_c)

    def test_compile_static_circuit_uses_gate_canonical_cache(self) -> None:
        gate = H(0)
        gate._tensor = None  # type: ignore[attr-defined]
        circuit = Circuit([
            gate,
            Measurement(0, 0),
        ])

        handle = self._compile(circuit, 1, 1)
        stats = native.get_program_stats(handle)
        self.assertEqual(int(stats["op_count"]), 1)

    def test_compile_prefers_native_dense_abi_over_legacy_sequences(self) -> None:
        gate = H(0)
        gate._canonical_coeff_re = ("bad",)  # type: ignore[attr-defined]
        gate._canonical_coeff_im = ("bad",)  # type: ignore[attr-defined]
        circuit = Circuit([
            gate,
            Measurement(0, 0),
        ])
        handle = self._compile(circuit, 1, 1)
        counts = native.execute_static_program(handle, 64, 5)
        self.assertEqual(int(sum(counts.values())), 64)

    def test_compile_prefers_native_permutation_abi_over_legacy_sequences(self) -> None:
        gate = X(0)
        gate._canonical_perm = ("bad",)  # type: ignore[attr-defined]
        circuit = Circuit([
            gate,
            Measurement(0, 0),
        ])
        handle = self._compile(circuit, 1, 1)
        counts = native.execute_static_program(handle, 64, 5)
        self.assertEqual(int(sum(counts.values())), 64)
        self.assertEqual(int(counts.get("1", 0)), 64)

    def test_state_buffers_reused_per_program_handle(self) -> None:
        circuit = Circuit([
            H(0),
            H(1),
            Measurement(0, 0),
            Measurement(1, 1),
        ])
        handle = self._compile(circuit, 2, 2)
        before = native.get_program_stats(handle)
        self.assertEqual(int(before.get("state_buffer_allocations", -1)), 0)

        counts_1 = native.execute_static_program(handle, 32, 1)
        self.assertEqual(int(sum(counts_1.values())), 32)
        after_first = native.get_program_stats(handle)
        self.assertEqual(int(after_first.get("state_buffer_allocations", -1)), 1)
        self.assertEqual(int(after_first.get("state_buffer_dim", -1)), 4)

        counts_2 = native.execute_static_program(handle, 32, 2)
        self.assertEqual(int(sum(counts_2.values())), 32)
        after_second = native.get_program_stats(handle)
        self.assertEqual(int(after_second.get("state_buffer_allocations", -1)), 1)

    def test_gpu_histogram_sampling_covers_all_shots(self) -> None:
        circuit = Circuit([
            H(0),
            H(1),
            H(2),
            H(3),
            H(4),
            Measurement(0, 0),
            Measurement(1, 1),
            Measurement(2, 2),
            Measurement(3, 3),
            Measurement(4, 4),
        ])
        handle = self._compile(circuit, 5, 5)
        counts = native.execute_static_program(handle, 4096, 4242)
        self.assertEqual(int(sum(counts.values())), 4096)
        self.assertGreaterEqual(len(counts), 16)


if __name__ == "__main__":
    unittest.main()
