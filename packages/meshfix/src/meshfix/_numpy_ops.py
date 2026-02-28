"""Pure numpy mesh repair operations."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as sparse_cc
import trimesh


def remove_degenerates(
    mesh: trimesh.Trimesh,
    face_indices: np.ndarray | None = None,
    area_threshold: float = 1e-10,
) -> trimesh.Trimesh:
    """Remove degenerate (zero/near-zero area) faces.

    Args:
        mesh: Input mesh (not mutated).
        face_indices: Pre-computed degenerate face indices from Defects.
            If None, detects via area threshold.
        area_threshold: Minimum face area to keep (used only if face_indices is None).

    Returns:
        New Trimesh with degenerate faces removed and vertices compacted.
    """
    if face_indices is None:
        areas = mesh.area_faces
        face_indices = np.where(areas < area_threshold)[0]

    if len(face_indices) == 0:
        return trimesh.Trimesh(
            vertices=mesh.vertices.copy(),
            faces=mesh.faces.copy(),
            process=False,
        )

    keep_mask = np.ones(mesh.faces.shape[0], dtype=bool)
    keep_mask[face_indices] = False
    new_faces = mesh.faces[keep_mask]

    return _compact(mesh.vertices, new_faces)


def remove_floating_vertices(
    mesh: trimesh.Trimesh,
    vert_indices: np.ndarray | None = None,
) -> trimesh.Trimesh:
    """Remove vertices not referenced by any face.

    Args:
        mesh: Input mesh (not mutated).
        vert_indices: Pre-computed floating vertex indices from Defects.
            If None, detects via np.setdiff1d.

    Returns:
        New Trimesh with floating vertices removed and faces remapped.
    """
    if vert_indices is None:
        if mesh.faces.shape[0] == 0:
            vert_indices = np.arange(mesh.vertices.shape[0])
        else:
            referenced = np.unique(mesh.faces)
            all_verts = np.arange(mesh.vertices.shape[0])
            vert_indices = np.setdiff1d(all_verts, referenced)

    if len(vert_indices) == 0:
        return trimesh.Trimesh(
            vertices=mesh.vertices.copy(),
            faces=mesh.faces.copy(),
            process=False,
        )

    return _compact(mesh.vertices, mesh.faces)


def remove_small_shells(
    mesh: trimesh.Trimesh,
    min_face_count: int = 10,
    min_face_ratio: float = 0.01,
) -> trimesh.Trimesh:
    """Remove small disconnected components.

    Keeps components where face_count >= min_face_count AND
    face_count / total_faces >= min_face_ratio. Always keeps the
    largest component regardless of thresholds.

    Args:
        mesh: Input mesh (not mutated).
        min_face_count: Minimum faces to keep a component.
        min_face_ratio: Minimum face ratio to keep a component.

    Returns:
        New Trimesh with small shells removed.
    """
    n_faces = mesh.faces.shape[0]
    if n_faces == 0:
        return trimesh.Trimesh(
            vertices=mesh.vertices.copy(),
            faces=mesh.faces.copy(),
            process=False,
        )

    adj_pairs = mesh.face_adjacency
    if len(adj_pairs) == 0:
        # Triangle soup — keep all
        return trimesh.Trimesh(
            vertices=mesh.vertices.copy(),
            faces=mesh.faces.copy(),
            process=False,
        )

    rows = np.concatenate([adj_pairs[:, 0], adj_pairs[:, 1]])
    cols = np.concatenate([adj_pairs[:, 1], adj_pairs[:, 0]])
    data = np.ones(len(rows), dtype=np.int8)
    graph = csr_matrix((data, (rows, cols)), shape=(n_faces, n_faces))
    n_comp, labels = sparse_cc(graph, directed=False)

    if n_comp <= 1:
        return trimesh.Trimesh(
            vertices=mesh.vertices.copy(),
            faces=mesh.faces.copy(),
            process=False,
        )

    face_counts = np.bincount(labels, minlength=n_comp)
    largest_comp = int(np.argmax(face_counts))

    keep_comps = set()
    for comp_id in range(n_comp):
        fc = face_counts[comp_id]
        if comp_id == largest_comp:
            keep_comps.add(comp_id)
        elif fc >= min_face_count and fc / n_faces >= min_face_ratio:
            keep_comps.add(comp_id)

    keep_mask = np.isin(labels, list(keep_comps))
    new_faces = mesh.faces[keep_mask]

    return _compact(mesh.vertices, new_faces)


def _compact(vertices: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
    """Compact vertices: keep only referenced vertices, remap face indices."""
    if faces.shape[0] == 0:
        return trimesh.Trimesh(vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=int), process=False)

    used = np.unique(faces)
    remap = np.full(vertices.shape[0], -1, dtype=np.intp)
    remap[used] = np.arange(len(used))
    new_verts = vertices[used]
    new_faces = remap[faces]

    return trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)
