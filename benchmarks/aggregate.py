"""Per-source aggregate stats and delta computation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np


@dataclass
class AggregateStats:
    """Per-source summary statistics across all meshes."""

    mesh_count: int
    # manifold
    watertight_ratio: float  # fraction watertight
    mean_boundary_loops: float
    mean_non_manifold_edges: float
    # normals
    mean_winding_consistency: float
    std_winding_consistency: float
    mean_backface_divergence: float
    # topology
    mean_genus: float
    mean_valence_mean: float
    mean_valence_std: float
    mean_degenerate_faces: float
    mean_floating_vertices: float
    # density
    mean_vertices_per_area: float
    mean_face_density_variance: float
    # fingerprint
    mean_hollowness: float
    mean_thinness: float
    mean_normal_entropy: float


def _safe_mean(values: list[float | int]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def _safe_std(values: list[float | int]) -> float:
    if not values:
        return 0.0
    return float(np.std(values))


def compute_aggregate(reports: dict[str, dict]) -> AggregateStats:
    """Compute aggregate stats from a dict of {name: report_dict}."""
    n = len(reports)
    if n == 0:
        return AggregateStats(
            mesh_count=0,
            watertight_ratio=0.0,
            mean_boundary_loops=0.0,
            mean_non_manifold_edges=0.0,
            mean_winding_consistency=0.0,
            std_winding_consistency=0.0,
            mean_backface_divergence=0.0,
            mean_genus=0.0,
            mean_valence_mean=0.0,
            mean_valence_std=0.0,
            mean_degenerate_faces=0.0,
            mean_floating_vertices=0.0,
            mean_vertices_per_area=0.0,
            mean_face_density_variance=0.0,
            mean_hollowness=0.0,
            mean_thinness=0.0,
            mean_normal_entropy=0.0,
        )

    rs = list(reports.values())

    watertight_count = sum(1 for r in rs if r.get("manifold", {}).get("is_watertight", False))
    winding_vals = [r["normals"]["winding_consistency"] for r in rs]

    return AggregateStats(
        mesh_count=n,
        watertight_ratio=watertight_count / n,
        mean_boundary_loops=_safe_mean([r["manifold"]["boundary_loop_count"] for r in rs]),
        mean_non_manifold_edges=_safe_mean([r["manifold"]["non_manifold_edge_count"] for r in rs]),
        mean_winding_consistency=_safe_mean(winding_vals),
        std_winding_consistency=_safe_std(winding_vals),
        mean_backface_divergence=_safe_mean([r["normals"]["backface_divergence"] for r in rs]),
        mean_genus=_safe_mean([r["topology"]["genus"] for r in rs]),
        mean_valence_mean=_safe_mean([r["topology"]["valence_mean"] for r in rs]),
        mean_valence_std=_safe_mean([r["topology"]["valence_std"] for r in rs]),
        mean_degenerate_faces=_safe_mean([r["topology"]["degenerate_face_count"] for r in rs]),
        mean_floating_vertices=_safe_mean([r["topology"]["floating_vertex_count"] for r in rs]),
        mean_vertices_per_area=_safe_mean([r["density"]["vertices_per_area"] for r in rs]),
        mean_face_density_variance=_safe_mean([r["density"]["face_density_variance"] for r in rs]),
        mean_hollowness=_safe_mean([r["fingerprint"]["hollowness"] for r in rs]),
        mean_thinness=_safe_mean([r["fingerprint"]["thinness"] for r in rs]),
        mean_normal_entropy=_safe_mean([r["fingerprint"]["normal_entropy"] for r in rs]),
    )


def compute_delta(generated: AggregateStats, reference: AggregateStats) -> dict[str, float]:
    """Compute raw difference (generated - reference) for all float fields."""
    delta = {}
    for f in fields(AggregateStats):
        if f.name == "mesh_count":
            continue
        gen_val = getattr(generated, f.name)
        ref_val = getattr(reference, f.name)
        delta[f.name] = round(gen_val - ref_val, 6)
    return delta
