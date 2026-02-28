"""Data types for mesh repair results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from meshbench.types import MeshReport, _ReportMixin


class FixStepName(Enum):
    REMOVE_DEGENERATES = auto()
    REMOVE_FLOATING_VERTICES = auto()
    REMOVE_SMALL_SHELLS = auto()
    FIX_WINDING = auto()
    FIX_NORMALS = auto()
    # Phase 2+:
    # SPLIT_NON_MANIFOLD_EDGES = auto()
    # SPLIT_NON_MANIFOLD_VERTICES = auto()
    # RESOLVE_SELF_INTERSECTIONS = auto()
    # FILL_HOLES = auto()
    # REMESH = auto()


class Backend(Enum):
    NUMPY = auto()
    TRIMESH = auto()
    PYMESHLAB = auto()
    MANIFOLD3D = auto()


@dataclass
class FixStep(_ReportMixin):
    """Record of a single repair operation."""

    name: FixStepName
    applied: bool
    backend: Backend
    skipped_reason: str | None = None
    vertices_removed: int = 0
    vertices_added: int = 0
    faces_removed: int = 0
    faces_added: int = 0
    duration_ms: float = 0.0


@dataclass
class FixConfig:
    """Configuration for the repair pipeline."""

    remove_degenerates: bool = True
    remove_floating_vertices: bool = True
    remove_small_shells: bool = True
    min_shell_face_count: int = 10
    min_shell_face_ratio: float = 0.01
    fix_winding: bool = True
    fix_normals: bool = True


@dataclass
class DecimateConfig:
    """Configuration for QEM decimation."""

    target_faces: int | None = None
    target_ratio: float | None = None  # 0.0–1.0
    feature_angle: float = 30.0  # degrees: sharper edges preserved
    preserve_boundary: bool = True
    quality_threshold: float = 0.3


@dataclass
class DecimateResult(_ReportMixin):
    """Result of a decimation operation."""

    backend: Backend
    faces_before: int
    faces_after: int
    vertices_before: int
    vertices_after: int
    feature_edges_before: int
    feature_edges_after: int
    duration_ms: float


@dataclass
class FixResult(_ReportMixin):
    """Result of a full repair pipeline run."""

    steps: list[FixStep] = field(default_factory=list)
    before: MeshReport | None = None
    after: MeshReport | None = None
    total_duration_ms: float = 0.0
    vertices_before: int = 0
    vertices_after: int = 0
    faces_before: int = 0
    faces_after: int = 0
