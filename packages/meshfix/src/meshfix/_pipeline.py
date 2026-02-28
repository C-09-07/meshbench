"""Pipeline executor: runs fix steps in order, re-audits."""

from __future__ import annotations

import time

import trimesh

import meshbench
from meshbench.types import MeshReport

from meshfix._backends import get_backend_for_step
from meshfix._numpy_ops import (
    remove_degenerates,
    remove_floating_vertices,
    remove_small_shells,
)
from meshfix._normal_ops import fix_normals, fix_winding
from meshfix._plan import plan_steps
from meshfix.types import FixConfig, FixResult, FixStep, FixStepName


_STEP_FNS = {
    FixStepName.REMOVE_DEGENERATES: "remove_degenerates",
    FixStepName.REMOVE_FLOATING_VERTICES: "remove_floating_vertices",
    FixStepName.REMOVE_SMALL_SHELLS: "remove_small_shells",
    FixStepName.FIX_WINDING: "fix_winding",
    FixStepName.FIX_NORMALS: "fix_normals",
}


def _execute_step(
    step_name: FixStepName,
    current: trimesh.Trimesh,
    report: MeshReport,
    config: FixConfig,
) -> tuple[trimesh.Trimesh, FixStep]:
    """Execute a single fix step and record results."""
    backend = get_backend_for_step(step_name)
    v_before = current.vertices.shape[0]
    f_before = current.faces.shape[0]

    t0 = time.perf_counter()

    if step_name == FixStepName.REMOVE_DEGENERATES:
        indices = None
        if report.defects is not None:
            indices = report.defects.degenerate_faces
        result_mesh = remove_degenerates(current, face_indices=indices)

    elif step_name == FixStepName.REMOVE_FLOATING_VERTICES:
        indices = None
        if report.defects is not None:
            indices = report.defects.floating_vertices
        result_mesh = remove_floating_vertices(current, vert_indices=indices)

    elif step_name == FixStepName.REMOVE_SMALL_SHELLS:
        result_mesh = remove_small_shells(
            current,
            min_face_count=config.min_shell_face_count,
            min_face_ratio=config.min_shell_face_ratio,
        )

    elif step_name == FixStepName.FIX_WINDING:
        result_mesh = fix_winding(current)

    elif step_name == FixStepName.FIX_NORMALS:
        result_mesh = fix_normals(current)

    else:
        raise ValueError(f"Unknown step: {step_name}")

    elapsed_ms = (time.perf_counter() - t0) * 1000

    v_after = result_mesh.vertices.shape[0]
    f_after = result_mesh.faces.shape[0]

    step = FixStep(
        name=step_name,
        applied=True,
        backend=backend,
        vertices_removed=max(0, v_before - v_after),
        vertices_added=max(0, v_after - v_before),
        faces_removed=max(0, f_before - f_after),
        faces_added=max(0, f_after - f_before),
        duration_ms=round(elapsed_ms, 2),
    )
    return result_mesh, step


def run_pipeline(
    mesh: trimesh.Trimesh,
    report: MeshReport | None = None,
    config: FixConfig | None = None,
) -> tuple[trimesh.Trimesh, FixResult]:
    """Execute the full repair pipeline.

    Args:
        mesh: Input mesh (not mutated).
        report: Pre-computed MeshReport. If None, audit() is called automatically.
        config: Repair configuration. If None, defaults are used.

    Returns:
        (fixed_mesh, FixResult) with before/after reports.
    """
    if config is None:
        config = FixConfig()

    if report is None:
        report = meshbench.audit(mesh, include_defects=True)

    v_before = mesh.vertices.shape[0]
    f_before = mesh.faces.shape[0]

    steps_to_run = plan_steps(report, config)

    t_start = time.perf_counter()
    current = mesh
    executed_steps: list[FixStep] = []
    use_defect_indices = True

    for step_name in steps_to_run:
        # After face/vertex removal, defect indices are stale
        if not use_defect_indices:
            # Clear defect indices for index-based steps
            stale_report = MeshReport(
                vertex_count=current.vertices.shape[0],
                face_count=current.faces.shape[0],
                edge_count=0,
                fingerprint=report.fingerprint,
                manifold=report.manifold,
                normals=report.normals,
                topology=report.topology,
                density=report.density,
                cell=report.cell,
                defects=None,
            )
            current, step = _execute_step(step_name, current, stale_report, config)
        else:
            current, step = _execute_step(step_name, current, report, config)

        executed_steps.append(step)

        # Once we remove faces/vertices, defect indices are invalidated
        if step.faces_removed > 0 or step.vertices_removed > 0:
            use_defect_indices = False

    total_ms = (time.perf_counter() - t_start) * 1000

    # Re-audit the final mesh
    after_report = meshbench.audit(current, include_defects=True)

    result = FixResult(
        steps=executed_steps,
        before=report,
        after=after_report,
        total_duration_ms=round(total_ms, 2),
        vertices_before=v_before,
        vertices_after=current.vertices.shape[0],
        faces_before=f_before,
        faces_after=current.faces.shape[0],
    )
    return current, result
