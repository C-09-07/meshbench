"""Shape fingerprint: PCA, hollowness, thinness, symmetry, taper.

Ported from customuse core/geometry.py with modifications:
- Drops is_watertight field
- Returns meshbench.types.Fingerprint
"""

from __future__ import annotations

import logging
import numpy as np
import trimesh

from meshbench._entropy import _histogram_entropy
from meshbench.types import Fingerprint

logger = logging.getLogger(__name__)


def compute_pca(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """Compute PCA axes and extents for a mesh.

    Returns:
        (axes, extents): axes is (3, 3) where rows are principal components
                         ordered by decreasing variance. extents is (3,).
    """
    verts = mesh.vertices
    centered = verts - verts.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    order = np.argsort(-eigenvalues)
    axes = eigenvectors[:, order].T  # rows = principal components
    extents = np.sqrt(eigenvalues[order]) * 2

    return axes, extents


def compute_fingerprint(
    mesh: trimesh.Trimesh,
    _pca: tuple[np.ndarray, np.ndarray] | None = None,
    _component_count: int | None = None,
) -> Fingerprint:
    """Extract geometric shape descriptors from a mesh.

    Computes hollowness, thinness, aspect ratio, symmetry, normal entropy,
    component count, and optional taper analysis.

    Args:
        _pca: Optional pre-computed (axes, extents) from compute_pca().
        _component_count: Optional pre-computed component count (avoids
            duplicate mesh.split() call when topology already computed it).
    """
    pca_axes, pca_extents = _pca if _pca is not None else compute_pca(mesh)

    # Hollowness: mesh_vol / hull_vol
    try:
        hull_vol = abs(mesh.convex_hull.volume)
        mesh_vol = abs(mesh.volume)
        hollowness = min(mesh_vol / max(hull_vol, 1e-12), 1.0)
    except ValueError:
        logger.warning("Degenerate convex hull for hollowness, returning 0.0")
        hollowness = 0.0

    # Thinness: surface_area / bbox_volume
    try:
        area = mesh.area
        bbox_vol = float(np.prod(mesh.bounding_box.extents))
        thinness = area / max(bbox_vol, 1e-12)
    except ValueError:
        logger.warning("Degenerate bounding box for thinness, returning 0.0")
        thinness = 0.0

    # Connected components
    if _component_count is not None:
        component_count = _component_count
    else:
        try:
            components = mesh.split()
            component_count = len(components) if components else 1
        except ModuleNotFoundError:
            logger.warning("networkx not installed, assuming 1 component")
            component_count = 1

    # Symmetry: ratio of 2nd/3rd PCA extents
    if pca_extents[2] > 1e-6:
        smaller = min(pca_extents[1], pca_extents[2])
        larger = max(pca_extents[1], pca_extents[2])
        symmetry_score = smaller / larger
    else:
        symmetry_score = 0.0

    aspect_ratio = pca_extents[0] / max(pca_extents[1], 1e-6)

    # Normal entropy (32-bin NDH)
    normal_entropy = _normal_entropy(mesh, pca_axes)

    # Taper analysis
    taper_result = compute_taper_vector(mesh, pca_axes)
    taper_ratio: float | None = None
    taper_direction: np.ndarray | None = None
    if taper_result is not None:
        taper_direction, taper_ratio = taper_result

    return Fingerprint(
        hollowness=round(hollowness, 4),
        thinness=round(thinness, 2),
        aspect_ratio=round(aspect_ratio, 2),
        symmetry_score=round(symmetry_score, 4),
        normal_entropy=round(normal_entropy, 4),
        component_count=component_count,
        taper_ratio=round(taper_ratio, 4) if taper_ratio is not None else None,
        taper_direction=taper_direction,
    )


def _normal_entropy(mesh: trimesh.Trimesh, pca_axes: np.ndarray) -> float:
    """Compute Shannon entropy of face normals projected onto the minor PCA axis."""
    if mesh.faces.shape[0] == 0:
        return 0.0
    minor_axis = pca_axes[2]
    dots = np.dot(mesh.face_normals, minor_axis)
    return _histogram_entropy(dots, bins=32)


def _perpendicular_spread(verts: np.ndarray, axis: np.ndarray) -> float:
    """Mean perpendicular distance from centroid of a vertex set."""
    if len(verts) < 3:
        return 0.0
    centroid = verts.mean(axis=0)
    offsets = verts - centroid
    axial = np.outer(offsets @ axis, axis)
    perpendicular = offsets - axial
    return float(np.mean(np.linalg.norm(perpendicular, axis=1)))


def compute_taper_vector(
    mesh: trimesh.Trimesh,
    pca_axes: np.ndarray,
) -> tuple[np.ndarray, float] | None:
    """Find the thick-to-thin direction along the primary PCA axis.

    Returns:
        (direction, taper_ratio): unit vector toward thick base, and the
        ratio thick/thin (>1 means tapered). None if no significant taper.
    """
    if mesh.vertices.shape[0] < 20:
        return None

    pca_axis = pca_axes[0]
    projections = mesh.vertices @ pca_axis
    proj_min = float(projections.min())
    proj_max = float(projections.max())
    proj_range = proj_max - proj_min

    if proj_range < 1e-6:
        return None

    min_sample = max(10, len(mesh.vertices) // 20)
    slice_width = proj_range * 0.10

    mask_lo = projections < (proj_min + slice_width)
    mask_hi = projections > (proj_max - slice_width)

    if int(mask_lo.sum()) < min_sample:
        idx_lo = np.argpartition(projections, min_sample)[:min_sample]
        mask_lo = np.zeros(len(projections), dtype=bool)
        mask_lo[idx_lo] = True

    if int(mask_hi.sum()) < min_sample:
        idx_hi = np.argpartition(-projections, min_sample)[:min_sample]
        mask_hi = np.zeros(len(projections), dtype=bool)
        mask_hi[idx_hi] = True

    spread_lo = _perpendicular_spread(mesh.vertices[mask_lo], pca_axis)
    spread_hi = _perpendicular_spread(mesh.vertices[mask_hi], pca_axis)

    thick = max(spread_lo, spread_hi)
    thin = min(spread_lo, spread_hi)

    if thin < 1e-6:
        return None

    taper_ratio = thick / thin
    if taper_ratio < 2.0:
        return None

    if spread_hi >= spread_lo:
        direction = pca_axis.copy()
    else:
        direction = -pca_axis.copy()

    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        return None

    return direction / norm, taper_ratio
