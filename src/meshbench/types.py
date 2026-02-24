"""Data types for mesh quality reports."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Fingerprint:
    """Geometric shape descriptors extracted from mesh geometry."""

    hollowness: float  # mesh_vol / hull_vol
    thinness: float  # surface_area / bbox_vol
    aspect_ratio: float  # PC1 / PC2 extent
    symmetry_score: float  # min(PC2,PC3) / max(PC2,PC3)
    normal_entropy: float  # 32-bin NDH Shannon entropy
    component_count: int  # connected components
    taper_ratio: float | None = None  # cross-section area change along PC1
    taper_direction: np.ndarray | None = None  # unit vector thick→thin


@dataclass
class ManifoldReport:
    """Manifold analysis: boundary edges, non-manifold geometry, watertightness."""

    is_watertight: bool  # boundary_edge_count == 0
    boundary_edge_count: int  # edges shared by exactly 1 face
    boundary_loop_count: int  # closed loops of boundary edges (= holes)
    non_manifold_edge_count: int  # edges shared by >2 faces
    non_manifold_vertex_count: int  # vertices where face fan isn't a disc


@dataclass
class NormalReport:
    """Normal consistency analysis."""

    entropy: float  # full-mesh normal entropy
    flipped_count: int  # adjacent face pairs with opposing normals
    winding_consistency: float  # 1.0 - (flipped / total_adjacent_pairs)
    backface_divergence: float  # max entropy difference across PCA axis splits
    dominant_normal: np.ndarray | None = None  # area-weighted peak bin normal


@dataclass
class TopologyReport:
    """Topological analysis: genus, valence distribution, degenerates."""

    genus: int  # topological genus from Euler: V-E+F=2(1-g)
    valence_mean: float
    valence_std: float
    valence_histogram: dict[int, int]
    quad_compatible_ratio: float  # % vertices at valence 4
    valence_outlier_count: int  # vertices with valence >= 10
    degenerate_face_count: int  # zero/near-zero area triangles
    floating_vertex_count: int  # vertices not in any face
    component_count: int
    component_sizes: list[tuple[int, int]] = field(
        default_factory=list
    )  # [(face_count, vert_count), ...] sorted desc


@dataclass
class DensityReport:
    """Polycount efficiency metrics."""

    vertices_per_area: float  # vertex_count / surface_area
    face_density_variance: float  # normalized variance of per-face area


@dataclass
class MeshReport:
    """Complete mesh quality report."""

    vertex_count: int
    face_count: int
    edge_count: int
    fingerprint: Fingerprint
    manifold: ManifoldReport
    normals: NormalReport
    topology: TopologyReport
    density: DensityReport
