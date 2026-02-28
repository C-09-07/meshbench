"""Data types for mesh quality reports."""

from __future__ import annotations

from dataclasses import dataclass, fields, field

import numpy as np


def _to_dict_value(obj: object) -> object:
    """Recursively convert a value to JSON-safe types."""
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, _ReportMixin):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {k: _to_dict_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_dict_value(v) for v in obj]
    return obj


class _ReportMixin:
    """Mixin providing JSON-safe serialization for report dataclasses."""

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dict."""
        return {f.name: _to_dict_value(getattr(self, f.name)) for f in fields(self)}  # type: ignore[arg-type]


@dataclass
class Fingerprint(_ReportMixin):
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
class ManifoldReport(_ReportMixin):
    """Manifold analysis: boundary edges, non-manifold geometry, watertightness."""

    is_watertight: bool  # boundary_edge_count == 0
    boundary_edge_count: int  # edges shared by exactly 1 face
    boundary_loop_count: int  # closed loops of boundary edges (= holes)
    non_manifold_edge_count: int  # edges shared by >2 faces
    non_manifold_vertex_count: int  # vertices where face fan isn't a disc
    non_manifold_edge_ratio: float = 0.0
    non_manifold_vertex_ratio: float = 0.0
    boundary_edge_ratio: float = 0.0
    self_intersection_count: int = 0
    self_intersection_ratio: float = 0.0


@dataclass
class NormalReport(_ReportMixin):
    """Normal consistency analysis."""

    entropy: float  # full-mesh normal entropy
    flipped_count: int  # adjacent face pairs with opposing normals
    winding_consistency: float  # 1.0 - (flipped / total_adjacent_pairs)
    backface_divergence: float  # max entropy difference across PCA axis splits
    dominant_normal: np.ndarray | None = None  # area-weighted peak bin normal


@dataclass
class TopologyReport(_ReportMixin):
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
    degenerate_face_ratio: float = 0.0
    floating_vertex_ratio: float = 0.0


@dataclass
class DensityReport(_ReportMixin):
    """Polycount efficiency metrics."""

    vertices_per_area: float  # vertex_count / surface_area
    face_density_variance: float  # normalized variance of per-face area


@dataclass
class CellReport(_ReportMixin):
    """Per-face (cell) quality metrics."""

    aspect_ratio_mean: float
    aspect_ratio_std: float
    aspect_ratio_max: float
    aspect_ratio_p95: float
    min_angle_mean: float  # degrees
    min_angle_std: float
    min_angle_min: float
    min_angle_p05: float
    max_angle_mean: float = 0.0  # degrees
    max_angle_std: float = 0.0
    max_angle_max: float = 0.0
    max_angle_p95: float = 0.0
    skewness_mean: float = 0.0  # equiangle skewness [0,1]
    skewness_std: float = 0.0
    skewness_max: float = 0.0
    skewness_p95: float = 0.0


@dataclass
class Check(_ReportMixin):
    """Single metric check result within a ScoreCard."""

    metric: str  # e.g. "is_watertight"
    status: str  # "pass", "warn", "fail"
    value: float | int | bool  # actual value
    threshold_warn: float | None
    threshold_fail: float | None
    message: str  # e.g. "12 non-manifold vertices (ratio 0.003)"


@dataclass
class ScoreCard(_ReportMixin):
    """Application-specific quality score from a profile evaluation."""

    profile: str  # e.g. "print"
    grade: str  # A-F
    score: float  # 0-100
    pass_count: int
    warn_count: int
    fail_count: int
    checks: list[Check] = field(default_factory=list)


@dataclass
class Defects(_ReportMixin):
    """Per-element defect indices and per-face metric arrays."""

    # Manifold defects (index arrays)
    boundary_edges: np.ndarray | None = None  # (N, 2) vertex pairs
    non_manifold_edges: np.ndarray | None = None  # (N, 2) vertex pairs
    non_manifold_vertices: np.ndarray | None = None  # (M,) vertex indices
    self_intersection_pairs: np.ndarray | None = None  # (K, 2) face pairs

    # Normal defects
    flipped_face_pairs: np.ndarray | None = None  # (N, 2) adjacent face pairs with opposing normals

    # Topology defects (index arrays)
    degenerate_faces: np.ndarray | None = None  # (D,) face indices
    floating_vertices: np.ndarray | None = None  # (V,) vertex indices
    high_valence_vertices: np.ndarray | None = None  # (H,) vertex indices with valence >= 10

    # Per-face metric arrays
    aspect_ratio: np.ndarray | None = None  # (F,)
    min_angle: np.ndarray | None = None  # (F,) degrees
    max_angle: np.ndarray | None = None  # (F,) degrees
    skewness: np.ndarray | None = None  # (F,) [0,1]


@dataclass
class FeatureReport(_ReportMixin):
    """Feature edge analysis from dihedral angles."""

    feature_edge_count: int
    smooth_edge_count: int
    boundary_edge_count: int
    feature_angle_deg: float  # threshold used
    feature_edge_ratio: float  # feature / total_interior
    dihedral_angles: np.ndarray | None = None  # (E_interior,) degrees, opt-in


@dataclass
class CurvatureReport(_ReportMixin):
    """Discrete curvature statistics."""

    mean_curvature_abs_avg: float
    mean_curvature_abs_max: float
    gaussian_curvature_avg: float
    gaussian_curvature_std: float
    flat_vertex_ratio: float  # |H| < epsilon


@dataclass
class MeshReport(_ReportMixin):
    """Complete mesh quality report."""

    vertex_count: int
    face_count: int
    edge_count: int
    fingerprint: Fingerprint
    manifold: ManifoldReport
    normals: NormalReport
    topology: TopologyReport
    density: DensityReport
    cell: CellReport | None = None
    defects: Defects | None = None
