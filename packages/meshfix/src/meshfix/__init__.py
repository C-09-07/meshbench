"""meshfix — Targeted mesh repair driven by meshbench analysis.

Public API:
    fix(mesh, report?, config?)  → (Trimesh, FixResult)
    plan(report, config?)        → list[FixStepName]
    available_backends()         → dict[Backend, bool]

Individual operations (all return new Trimesh, never mutate input):
    remove_degenerates(mesh, face_indices?) → Trimesh
    remove_floating_vertices(mesh, vert_indices?) → Trimesh
    remove_small_shells(mesh, min_face_count?, min_face_ratio?) → Trimesh
    fix_winding(mesh) → Trimesh
    fix_normals(mesh) → Trimesh
"""

from __future__ import annotations

import trimesh

from meshbench.types import MeshReport

from meshfix._backends import available_backends
from meshfix._normal_ops import fix_normals, fix_winding
from meshfix._numpy_ops import (
    remove_degenerates,
    remove_floating_vertices,
    remove_small_shells,
)
from meshfix._pipeline import run_pipeline
from meshfix._plan import plan_steps
from meshfix.types import (
    Backend,
    FixConfig,
    FixResult,
    FixStep,
    FixStepName,
)


def fix(
    mesh: trimesh.Trimesh,
    report: MeshReport | None = None,
    config: FixConfig | None = None,
) -> tuple[trimesh.Trimesh, FixResult]:
    """Smart repair pipeline. Returns (new_mesh, result).

    Args:
        mesh: Input mesh (not mutated).
        report: Pre-computed MeshReport with defects. If None, audit() is called.
        config: Repair configuration. If None, defaults are used.

    Returns:
        Tuple of (repaired_mesh, FixResult with before/after reports).
    """
    return run_pipeline(mesh, report, config)


def plan(
    report: MeshReport,
    config: FixConfig | None = None,
) -> list[FixStepName]:
    """Dry-run: what steps would fix() execute?

    Args:
        report: MeshReport to analyze for defects.
        config: Repair configuration. If None, defaults are used.

    Returns:
        Ordered list of fix steps that would be executed.
    """
    if config is None:
        config = FixConfig()
    return plan_steps(report, config)


__all__ = [
    "available_backends",
    "fix",
    "fix_normals",
    "fix_winding",
    "plan",
    "remove_degenerates",
    "remove_floating_vertices",
    "remove_small_shells",
    "Backend",
    "FixConfig",
    "FixResult",
    "FixStep",
    "FixStepName",
]
