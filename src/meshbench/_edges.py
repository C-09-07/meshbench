"""Vectorized edge utilities shared across meshbench modules.

All edge extraction is done via NumPy operations on mesh.faces — no Python loops.
"""

from __future__ import annotations

import numpy as np
import trimesh


def unique_edges(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return sorted unique edges as an (E, 2) int array.

    Each row is (min_vertex, max_vertex). No Python loops.
    """
    faces = mesh.faces  # (F, 3)
    # Extract all 3 edges per triangle: (0,1), (1,2), (2,0)
    e0 = faces[:, [0, 1]]
    e1 = faces[:, [1, 2]]
    e2 = faces[:, [2, 0]]
    all_edges = np.vstack([e0, e1, e2])  # (3F, 2)

    # Sort each edge so (min, max)
    all_edges.sort(axis=1)

    # Unique rows
    return np.unique(all_edges, axis=0)


def edge_face_counts(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """Return (edges, counts) where edges is (E, 2) and counts is (E,).

    counts[i] = number of faces sharing edges[i]. No Python loops.
    """
    faces = mesh.faces
    e0 = faces[:, [0, 1]]
    e1 = faces[:, [1, 2]]
    e2 = faces[:, [2, 0]]
    all_edges = np.vstack([e0, e1, e2])
    all_edges.sort(axis=1)

    edges, counts = np.unique(all_edges, axis=0, return_counts=True)
    return edges, counts


def vertex_valences(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return an array of length V where valences[v] = valence of vertex v.

    Valence = number of unique edges incident to v. No Python loops.
    """
    n_verts = mesh.vertices.shape[0]
    edges = unique_edges(mesh)  # (E, 2) sorted

    # Each edge contributes +1 valence to both endpoints
    valences = np.zeros(n_verts, dtype=int)
    np.add.at(valences, edges[:, 0], 1)
    np.add.at(valences, edges[:, 1], 1)
    return valences
