"""Planner: reads a MeshReport and decides which fix steps to run."""

from __future__ import annotations

from meshbench.types import MeshReport

from meshfix.types import FixConfig, FixStepName


def plan_steps(report: MeshReport, config: FixConfig) -> list[FixStepName]:
    """Decide which steps to run based on defects and config flags.

    Steps are returned in execution order:
    1. Remove degenerates (index-based, safe to run first)
    2. Remove floating vertices (index-based)
    3. Remove small shells (topology-based, runs on current mesh state)
    4. Fix winding (operates on current mesh state)
    5. Fix normals (always, cheap safety net)
    """
    steps: list[FixStepName] = []

    defects = report.defects

    if config.remove_degenerates:
        if defects is not None and defects.degenerate_faces is not None:
            steps.append(FixStepName.REMOVE_DEGENERATES)
        elif report.topology.degenerate_face_count > 0:
            steps.append(FixStepName.REMOVE_DEGENERATES)

    if config.remove_floating_vertices:
        if defects is not None and defects.floating_vertices is not None:
            steps.append(FixStepName.REMOVE_FLOATING_VERTICES)
        elif report.topology.floating_vertex_count > 0:
            steps.append(FixStepName.REMOVE_FLOATING_VERTICES)

    if config.remove_small_shells:
        if report.topology.component_count > 1:
            steps.append(FixStepName.REMOVE_SMALL_SHELLS)

    if config.fix_winding:
        if defects is not None and defects.flipped_face_pairs is not None:
            steps.append(FixStepName.FIX_WINDING)
        elif report.normals.flipped_count > 0:
            steps.append(FixStepName.FIX_WINDING)

    # Always run fix_normals as a cheap safety net
    if config.fix_normals:
        steps.append(FixStepName.FIX_NORMALS)

    return steps
