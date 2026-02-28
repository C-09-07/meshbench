"""Normal consistency analysis: entropy, winding, flipped faces, backface divergence.

Ports normal_entropy and extract_dominant_normal from customuse core/geometry.py.
Adds flipped_normal_count, winding_consistency, backface_divergence.
"""

from __future__ import annotations

import numpy as np
import trimesh

from meshbench._entropy import _histogram_entropy
from meshbench.fingerprint import compute_pca
from meshbench.types import NormalReport


def normal_entropy(
    mesh: trimesh.Trimesh,
    _pca: tuple[np.ndarray, np.ndarray] | None = None,
) -> float:
    """Compute Shannon entropy of face normals projected onto the minor PCA axis.

    Uses a 32-bin histogram over dot products in [-1, 1].
    Higher entropy = more diffuse normal distribution.
    """
    if mesh.faces.shape[0] == 0:
        return 0.0

    pca_axes, _ = _pca if _pca is not None else compute_pca(mesh)
    minor_axis = pca_axes[2]
    dots = np.dot(mesh.face_normals, minor_axis)
    return _histogram_entropy(dots, bins=32)


def dominant_normal(
    mesh: trimesh.Trimesh,
    pca_axes: np.ndarray | None = None,
) -> np.ndarray | None:
    """Extract the dominant normal direction via area-weighted NDH.

    Returns the area-weighted mean 3D normal of faces in the peak bin,
    or None if no clear facing direction exists.
    """
    if mesh.faces.shape[0] == 0:
        return None

    if pca_axes is None:
        pca_axes, _ = compute_pca(mesh)

    minor_axis = pca_axes[2]
    face_normals = mesh.face_normals
    face_areas = mesh.area_faces

    dots = face_normals @ minor_axis
    bin_edges = np.linspace(-1.0, 1.0, 33)
    bin_indices = np.digitize(dots, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, 31)

    weighted_counts = np.bincount(bin_indices, weights=face_areas, minlength=32)[:32]

    total = weighted_counts.sum()
    if total < 1e-12:
        return None

    peak_bin = int(np.argmax(weighted_counts))
    mean_count = total / 32
    if weighted_counts[peak_bin] < 2.0 * mean_count:
        return None

    peak_mask = bin_indices == peak_bin
    peak_normals = face_normals[peak_mask]
    peak_areas = face_areas[peak_mask]

    weighted_normal = (peak_normals * peak_areas[:, np.newaxis]).sum(axis=0)
    norm = float(np.linalg.norm(weighted_normal))
    if norm < 1e-12:
        return None

    return weighted_normal / norm


def flipped_face_pairs(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return (N, 2) array of adjacent face pairs with opposing normals (dot < 0).

    Each row is a pair of face indices whose normals disagree.
    Returns empty (0, 2) array if none found.
    """
    if mesh.faces.shape[0] < 2:
        return np.empty((0, 2), dtype=np.intp)

    adj = mesh.face_adjacency
    if len(adj) == 0:
        return np.empty((0, 2), dtype=np.intp)

    normals = mesh.face_normals
    dots = np.sum(normals[adj[:, 0]] * normals[adj[:, 1]], axis=1)
    mask = dots < 0
    return adj[mask]


def flipped_normal_count(mesh: trimesh.Trimesh) -> int:
    """Count adjacent face pairs with opposing normals (dot < 0)."""
    return len(flipped_face_pairs(mesh))


def winding_consistency(
    mesh: trimesh.Trimesh,
    _flipped: int | None = None,
) -> float:
    """Fraction of adjacent face pairs with consistent winding (normals agree).

    Returns 1.0 for perfectly consistent winding, 0.0 for all flipped.

    Args:
        _flipped: Optional pre-computed flipped count to avoid recomputation.
    """
    if mesh.faces.shape[0] < 2:
        return 1.0

    adj = mesh.face_adjacency
    total_pairs = len(adj)
    if total_pairs == 0:
        return 1.0

    n_flipped = _flipped if _flipped is not None else flipped_normal_count(mesh)
    return 1.0 - (n_flipped / total_pairs)


def backface_divergence(
    mesh: trimesh.Trimesh,
    _pca: tuple[np.ndarray, np.ndarray] | None = None,
) -> float:
    """View-independent backface divergence.

    Splits faces along each PCA axis into front/back hemispheres,
    computes entropy of each half, and reports the max entropy
    difference across all 3 axes. High values indicate one side has
    more varied normals than the other (common in single-sided models).
    """
    if mesh.faces.shape[0] < 2:
        return 0.0

    pca_axes, _ = _pca if _pca is not None else compute_pca(mesh)
    face_normals = mesh.face_normals
    face_centers = mesh.triangles_center
    mesh_center = mesh.centroid

    max_diff = 0.0
    for i in range(3):
        axis = pca_axes[i]
        projections = (face_centers - mesh_center) @ axis
        pos_mask = projections >= 0
        neg_mask = ~pos_mask

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue

        ent_pos = _histogram_entropy(np.dot(face_normals[pos_mask], axis), bins=16)
        ent_neg = _histogram_entropy(np.dot(face_normals[neg_mask], axis), bins=16)
        max_diff = max(max_diff, abs(ent_pos - ent_neg))

    return round(max_diff, 4)


def backface_coherence(mesh: trimesh.Trimesh, view_dir: np.ndarray) -> float:
    """View-dependent backface coherence.

    Lower-level function: computes entropy difference between faces
    facing toward vs away from the given view direction.
    """
    if mesh.faces.shape[0] < 2:
        return 0.0

    view_dir = np.asarray(view_dir, dtype=float)
    norm = np.linalg.norm(view_dir)
    if norm < 1e-12:
        return 0.0
    view_dir = view_dir / norm

    face_normals = mesh.face_normals
    dots = face_normals @ view_dir

    front_mask = dots >= 0
    back_mask = ~front_mask

    if front_mask.sum() == 0 or back_mask.sum() == 0:
        return 0.0

    ent_front = _histogram_entropy(dots[front_mask], bins=16)
    ent_back = _histogram_entropy(dots[back_mask], bins=16)
    return round(abs(ent_front - ent_back), 4)


def check_and_fix_inverted_normals(
    mesh: trimesh.Trimesh,
    dominant_normal_vec: np.ndarray,
) -> bool:
    """Detect and fix inside-out mesh normals.

    Returns True if normals were flipped.
    """
    centroid = mesh.centroid
    face_normals = mesh.face_normals
    face_areas = mesh.area_faces

    alignment = face_normals @ dominant_normal_vec
    threshold = float(np.percentile(alignment, 75))
    aligned_mask = alignment >= threshold

    if not np.any(aligned_mask):
        return False

    aligned_centroids = mesh.triangles_center[aligned_mask]
    aligned_areas = face_areas[aligned_mask]
    total_area = float(aligned_areas.sum())
    if total_area < 1e-12:
        return False

    surface_pt = (
        aligned_centroids * aligned_areas[:, np.newaxis]
    ).sum(axis=0) / total_area

    to_centroid = centroid - surface_pt
    if float(np.dot(dominant_normal_vec, to_centroid)) > 0:
        mesh.faces = np.fliplr(mesh.faces)
        return True

    return False


def compute_normal_report(
    mesh: trimesh.Trimesh,
    _pca: tuple[np.ndarray, np.ndarray] | None = None,
) -> NormalReport:
    """Compute a full NormalReport for a mesh.

    Args:
        _pca: Optional pre-computed (axes, extents) from compute_pca().
    """
    pca = _pca if _pca is not None else compute_pca(mesh)
    ent = normal_entropy(mesh, _pca=pca)
    flipped = flipped_normal_count(mesh)
    consistency = winding_consistency(mesh, _flipped=flipped)
    divergence = backface_divergence(mesh, _pca=pca)
    dom = dominant_normal(mesh, pca_axes=pca[0])

    return NormalReport(
        entropy=round(ent, 4),
        flipped_count=flipped,
        winding_consistency=round(consistency, 4),
        backface_divergence=divergence,
        dominant_normal=dom,
    )
