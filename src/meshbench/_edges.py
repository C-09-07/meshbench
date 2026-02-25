"""Vectorized edge utilities shared across meshbench modules.

All edge extraction is done via NumPy operations on mesh.faces — no Python loops.
"""

from __future__ import annotations

import numpy as np
import trimesh

# Pre-computed edge data: (edges (E,2), counts (E,))
EdgeData = tuple[np.ndarray, np.ndarray]


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


def unique_edges(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return sorted unique edges as an (E, 2) int array.

    Each row is (min_vertex, max_vertex).
    """
    all_edges = _build_sorted_edges(mesh.faces)
    return np.unique(all_edges, axis=0)


def edge_face_counts(mesh: trimesh.Trimesh) -> EdgeData:
    """Return (edges, counts) where edges is (E, 2) and counts is (E,).

    counts[i] = number of faces sharing edges[i].
    """
    all_edges = _build_sorted_edges(mesh.faces)
    edges, counts = np.unique(all_edges, axis=0, return_counts=True)
    return edges, counts


def vertex_valences(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return an array of length V where valences[v] = valence of vertex v.

    Valence = number of unique edges incident to v.
    """
    n_verts = mesh.vertices.shape[0]
    edges = unique_edges(mesh)

    valences = np.zeros(n_verts, dtype=int)
    np.add.at(valences, edges[:, 0], 1)
    np.add.at(valences, edges[:, 1], 1)
    return valences
