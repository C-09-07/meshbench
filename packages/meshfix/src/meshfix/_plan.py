"""Planner: reads a MeshReport and decides which fix steps to run."""

from __future__ import annotations

from meshbench.types import MeshReport

from meshfix.types import FixConfig, FixStepName


def plan_steps(report: MeshReport, config: FixConfig) -> list[FixStepName]:
    """Decide which steps to run based on defects and config flags.

    Steps are returned in execution order:
    1. Remove degenerates (index-based, safe to run first)
    2. Remove floating vertices (index-based)
    3. Remove small shells (topology-based)
    4. Split non-manifold edges (Phase 3 topology repair)
    5. Split non-manifold vertices (Phase 3)
    6. Fill holes (Phase 3)
    7. Resolve self-intersections (Phase 3, last — needs manifold-ish input)
    8. Fix winding (re-normalize after structural changes)
    9. Fix normals (always, cheap safety net)
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

    # Phase 3: topology repair (after cleanup, before normalization)
    if config.split_non_manifold_edges:
        if report.manifold.non_manifold_edge_count > 0:
            steps.append(FixStepName.SPLIT_NON_MANIFOLD_EDGES)

    if config.split_non_manifold_vertices:
        if report.manifold.non_manifold_vertex_count > 0:
            steps.append(FixStepName.SPLIT_NON_MANIFOLD_VERTICES)

    if config.fill_holes:
        if report.manifold.boundary_loop_count > 0:
            steps.append(FixStepName.FILL_HOLES)

    if config.resolve_self_intersections:
        if report.manifold.self_intersection_count > 0:
            steps.append(FixStepName.RESOLVE_SELF_INTERSECTIONS)

    if config.fix_winding:
        if defects is not None and defects.flipped_face_pairs is not None:
            steps.append(FixStepName.FIX_WINDING)
        elif report.normals.flipped_count > 0:
            steps.append(FixStepName.FIX_WINDING)

    # Always run fix_normals as a cheap safety net
    if config.fix_normals:
        steps.append(FixStepName.FIX_NORMALS)

    return steps
