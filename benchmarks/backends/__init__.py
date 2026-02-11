"""Backend adapter registry for comparison harness."""

from __future__ import annotations

import os
import sys

from benchmarks.backends.base import BackendAdapter

KNOWN_BACKENDS = ["native", "aer", "qsim"]


def _configure_qsim_runtime() -> None:
    """Avoid OpenMP duplicate-runtime aborts on macOS when torch is already loaded."""
    if sys.platform != "darwin":
        return
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def create_backend(name: str) -> BackendAdapter:
    key = name.lower()
    if key == "native":
        from benchmarks.backends.native_adapter import NativeAdapter

        return NativeAdapter()
    if key == "aer":
        from benchmarks.backends.aer_adapter import AerAdapter

        return AerAdapter()
    if key == "qsim":
        _configure_qsim_runtime()
        from benchmarks.backends.qsim_adapter import QsimAdapter

        return QsimAdapter()
    raise ValueError(f"Unknown backend: {name}")


def known_backends() -> list[str]:
    return list(KNOWN_BACKENDS)
