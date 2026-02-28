"""Backend detection and dispatch."""

from __future__ import annotations

from meshfix.types import Backend, FixStepName

_HAS_PYMESHLAB = False
try:
    import pymeshlab  # noqa: F401

    _HAS_PYMESHLAB = True
except ImportError:
    pass

_HAS_MANIFOLD3D = False
try:
    import manifold3d  # noqa: F401

    _HAS_MANIFOLD3D = True
except ImportError:
    pass

# Maps each step to its preferred backend(s) in priority order.
_STEP_BACKENDS: dict[FixStepName, list[Backend]] = {
    FixStepName.REMOVE_DEGENERATES: [Backend.NUMPY],
    FixStepName.REMOVE_FLOATING_VERTICES: [Backend.NUMPY],
    FixStepName.REMOVE_SMALL_SHELLS: [Backend.NUMPY],
    FixStepName.FIX_WINDING: [Backend.TRIMESH],
    FixStepName.FIX_NORMALS: [Backend.TRIMESH],
}


def available_backends() -> dict[Backend, bool]:
    """Check which backends are installed."""
    return {
        Backend.NUMPY: True,
        Backend.TRIMESH: True,
        Backend.PYMESHLAB: _HAS_PYMESHLAB,
        Backend.MANIFOLD3D: _HAS_MANIFOLD3D,
    }


def get_backend_for_step(step: FixStepName) -> Backend:
    """Return the best available backend for a step."""
    candidates = _STEP_BACKENDS.get(step, [])
    avail = available_backends()
    for b in candidates:
        if avail[b]:
            return b
    raise RuntimeError(f"No available backend for {step.name}")
