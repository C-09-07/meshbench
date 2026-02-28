"""Vectorized edge utilities shared across meshbench modules.

All edge extraction is done via NumPy operations on mesh.faces — no Python loops.
"""

from __future__ import annotations

import numpy as np
import trimesh

# Pre-computed edge data: (edges (E,2), counts (E,))
EdgeData = tuple[np.ndarray, np.ndarray]

# Rich edge→face mapping: (sorted_edges (3F,2), sorted_face_ids (3F,),
#                           edge_starts (E,), edge_ends (E,))
EdgeFaceMap = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _build_sorted_edges(faces: np.ndarray) -> np.ndarray:
    """Extract all edges from faces and sort each row to (min, max).

    Returns (3F, 2) array of sorted edge pairs.
    """
    e0 = faces[:, [0, 1]]
    e1 = faces[:, [1, 2]]
    e2 = faces[:, [2, 0]]
    all_edges = np.vstack([e0, e1, e2])
    all_edges.sort(axis=1)
    return all_edges


def build_edge_face_map(faces: np.ndarray) -> EdgeFaceMap:
    """Build sorted edge-to-face mapping with group boundaries.

    Returns (sorted_edges, sorted_face_ids, edge_starts, edge_ends) where
    edges are sorted by packed key and grouped into unique-edge runs.
    Used by non_manifold_vertices to avoid redundant sorting.
    """
    n_faces = faces.shape[0]
    all_edges = _build_sorted_edges(faces)
    face_ids = np.tile(np.arange(n_faces), 3)

    max_v = int(faces.max()) + 1
    edge_keys = all_edges[:, 0].astype(np.int64) * max_v + all_edges[:, 1].astype(np.int64)

    order = np.argsort(edge_keys)
    sorted_keys = edge_keys[order]
    sorted_face_ids = face_ids[order]
    sorted_edges = all_edges[order]

    changes = np.where(np.diff(sorted_keys) != 0)[0] + 1
    edge_starts = np.concatenate([[0], changes])
    edge_ends = np.concatenate([changes, [len(sorted_keys)]])

    return sorted_edges, sorted_face_ids, edge_starts, edge_ends


def edge_data_from_map(efm: EdgeFaceMap) -> EdgeData:
    """Derive (unique_edges, counts) from an EdgeFaceMap."""
    sorted_edges, _, edge_starts, edge_ends = efm
    edges = sorted_edges[edge_starts]
    counts = (edge_ends - edge_starts).astype(np.int64)
    return edges, counts


def unique_edges(
    mesh: trimesh.Trimesh,
    _edge_data: EdgeData | None = None,
) -> np.ndarray:
    """Return sorted unique edges as an (E, 2) int array.

    Each row is (min_vertex, max_vertex).
    """
    if _edge_data is not None:
        return _edge_data[0]
    edges, _ = edge_face_counts(mesh)
    return edges


def edge_face_counts(mesh: trimesh.Trimesh) -> EdgeData:
    """Return (edges, counts) where edges is (E, 2) and counts is (E,).

    counts[i] = number of faces sharing edges[i].
    Uses 1D key packing for faster dedup than np.unique(axis=0).
    """
    if mesh.faces.shape[0] == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int64)

    all_edges = _build_sorted_edges(mesh.faces)
    max_v = int(mesh.faces.max()) + 1
    keys = all_edges[:, 0].astype(np.int64) * max_v + all_edges[:, 1].astype(np.int64)
    unique_keys, counts = np.unique(keys, return_counts=True)
    edges = np.column_stack([unique_keys // max_v, unique_keys % max_v])
    return edges, counts


def vertex_valences(
    mesh: trimesh.Trimesh,
    _edge_data: EdgeData | None = None,
) -> np.ndarray:
    """Return an array of length V where valences[v] = valence of vertex v.

    Valence = number of unique edges incident to v.
    """
    n_verts = mesh.vertices.shape[0]
    edges = unique_edges(mesh, _edge_data)
    if len(edges) == 0:
        return np.zeros(n_verts, dtype=int)
    return (
        np.bincount(edges[:, 0], minlength=n_verts)
        + np.bincount(edges[:, 1], minlength=n_verts)
    )
