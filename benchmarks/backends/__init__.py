"""Backend adapter registry for comparison harness."""

from __future__ import annotations

from benchmarks.backends.base import BackendAdapter

KNOWN_BACKENDS = ["native", "aer"]


def create_backend(name: str) -> BackendAdapter:
    key = name.lower()
    if key == "native":
        from benchmarks.backends.native_adapter import NativeAdapter

        return NativeAdapter()
    if key == "aer":
        from benchmarks.backends.aer_adapter import AerAdapter

        return AerAdapter()
    raise ValueError(f"Unknown backend: {name}")


def known_backends() -> list[str]:
    return list(KNOWN_BACKENDS)
