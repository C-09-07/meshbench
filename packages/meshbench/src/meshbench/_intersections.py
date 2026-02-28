"""Self-intersection detection via sweep-and-prune + SAT narrow phase.

Broad phase: sort triangle AABBs along longest-extent axis, sweep to find
overlapping non-adjacent pairs.  Narrow phase: Separating Axis Theorem (SAT)
on 11 candidate axes (2 face normals + 9 edge cross products).

Optional numba acceleration for the batch SAT inner loop.
"""

from __future__ import annotations

import numpy as np
import trimesh


try:
    from meshbench._numba_kernels import _sat_batch_numba

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


# ---------------------------------------------------------------------------
# AABB helpers
# ---------------------------------------------------------------------------


def _triangle_aabbs(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-face axis-aligned bounding boxes.

    Returns ``(mins, maxs)`` each of shape ``(F, 3)``.
    """
    triangles = vertices[faces]  # (F, 3, 3)
    mins = triangles.min(axis=1)  # (F, 3)
    maxs = triangles.max(axis=1)  # (F, 3)
    return mins, maxs


# ---------------------------------------------------------------------------
# Broad phase — sweep-and-prune
# ---------------------------------------------------------------------------


def _build_adjacency_set(faces: np.ndarray) -> set[int]:
    """Build a set of packed keys for face pairs sharing at least one vertex.

    Key = ``min(fi, fj) * n_faces + max(fi, fj)`` so lookup is O(1).
    """
    n_faces = faces.shape[0]

    # Invert the vertex→face relationship: for every vertex, collect its faces
    vert_indices = faces.ravel()
    face_indices = np.repeat(np.arange(n_faces, dtype=np.int64), 3)

    order = np.argsort(vert_indices)
    sv = vert_indices[order]
    sf = face_indices[order]

    changes = np.where(np.diff(sv) != 0)[0] + 1
    starts = np.concatenate([[0], changes])
    ends = np.concatenate([changes, [len(sv)]])

    adj: set[int] = set()
    for k in range(len(starts)):
        group = sf[starts[k] : ends[k]]
        n = len(group)
        if n < 2:
            continue
        # All pairs in this group share a vertex
        for i in range(n):
            fi = int(group[i])
            for j in range(i + 1, n):
                fj = int(group[j])
                lo, hi = (fi, fj) if fi < fj else (fj, fi)
                adj.add(lo * n_faces + hi)

    return adj


def _find_candidate_pairs(mesh: trimesh.Trimesh) -> np.ndarray:
    """Find non-adjacent face pairs with overlapping AABBs.

    Uses sweep-and-prune along the axis of greatest AABB extent.

    Returns ``(N, 2)`` int64 array of candidate face-index pairs, or an
    empty ``(0, 2)`` array when there are no candidates.
    """
    faces = mesh.faces
    n_faces = faces.shape[0]

    if n_faces < 2:
        return np.empty((0, 2), dtype=np.int64)

    vertices = mesh.vertices
    mins, maxs = _triangle_aabbs(vertices, faces)

    # Choose sweep axis = axis with greatest extent range
    global_extent = maxs.max(axis=0) - mins.min(axis=0)
    axis = int(np.argmax(global_extent))

    # Sort triangles by AABB min along sweep axis
    sort_order = np.argsort(mins[:, axis])
    sorted_min = mins[sort_order]
    sorted_max = maxs[sort_order]

    # Build adjacency set for filtering
    adj = _build_adjacency_set(faces)

    # Sweep
    pairs: list[tuple[int, int]] = []
    seen_keys: set[int] = set()

    for ii in range(n_faces):
        i = int(sort_order[ii])
        i_max_axis = sorted_max[ii, axis]

        jj = ii + 1
        while jj < n_faces and sorted_min[jj, axis] <= i_max_axis:
            j = int(sort_order[jj])

            # Check remaining 2 axes for AABB overlap
            other_axes = [a for a in range(3) if a != axis]
            overlap = True
            for ax in other_axes:
                if mins[i, ax] > maxs[j, ax] or mins[j, ax] > maxs[i, ax]:
                    overlap = False
                    break

            if overlap:
                lo, hi = (i, j) if i < j else (j, i)
                key = lo * n_faces + hi

                # Filter out adjacent pairs and duplicates
                if key not in adj and key not in seen_keys:
                    seen_keys.add(key)
                    pairs.append((lo, hi))

            jj += 1

    if not pairs:
        return np.empty((0, 2), dtype=np.int64)
    return np.array(pairs, dtype=np.int64)


# ---------------------------------------------------------------------------
# Narrow phase — Separating Axis Theorem
# ---------------------------------------------------------------------------


def _project_triangle(tri: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
    """Project a triangle (3, 3) onto an axis, return (min, max) of projections."""
    dots = tri @ axis
    return float(dots.min()), float(dots.max())


def _sat_test_pair(tri_a: np.ndarray, tri_b: np.ndarray) -> bool:
    """SAT intersection test for a single triangle pair.

    Parameters
    ----------
    tri_a, tri_b : ndarray of shape (3, 3)
        Vertex positions for each triangle.

    Returns
    -------
    bool
        True if the triangles intersect (no separating axis found).
    """
    edges_a = np.array(
        [tri_a[1] - tri_a[0], tri_a[2] - tri_a[1], tri_a[0] - tri_a[2]]
    )
    edges_b = np.array(
        [tri_b[1] - tri_b[0], tri_b[2] - tri_b[1], tri_b[0] - tri_b[2]]
    )

    # Face normals
    normal_a = np.cross(edges_a[0], edges_a[1])
    normal_b = np.cross(edges_b[0], edges_b[1])

    # All candidate separating axes: 2 normals + 9 edge cross products
    axes: list[np.ndarray] = [normal_a, normal_b]
    for ea in edges_a:
        for eb in edges_b:
            axes.append(np.cross(ea, eb))

    for ax in axes:
        length_sq = float(ax @ ax)
        if length_sq < 1e-30:
            # Degenerate axis (parallel edges or degenerate triangle) — skip
            continue

        min_a, max_a = _project_triangle(tri_a, ax)
        min_b, max_b = _project_triangle(tri_b, ax)

        if max_a < min_b or max_b < min_a:
            return False  # Separating axis found — no intersection

    return True  # No separating axis — triangles intersect


def _batch_sat_test(triangles: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Test multiple triangle pairs for intersection.

    Parameters
    ----------
    triangles : ndarray of shape (F, 3, 3)
        Vertex positions for every face.
    pairs : ndarray of shape (N, 2)
        Face-index pairs to test.

    Returns
    -------
    ndarray of bool, length N
        True where the pair intersects.
    """
    n = pairs.shape[0]
    if n == 0:
        return np.empty(0, dtype=bool)

    if _HAS_NUMBA:
        return _sat_batch_numba(
            triangles.astype(np.float64),
            pairs[:, 0].astype(np.int64),
            pairs[:, 1].astype(np.int64),
        )

    results = np.empty(n, dtype=bool)
    for k in range(n):
        results[k] = _sat_test_pair(
            triangles[pairs[k, 0]], triangles[pairs[k, 1]]
        )
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_self_intersections(
    mesh: trimesh.Trimesh,
) -> tuple[int, np.ndarray]:
    """Detect self-intersecting (non-adjacent) face pairs.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh.

    Returns
    -------
    count : int
        Number of self-intersecting face pairs.
    pairs : ndarray of shape (K, 2), dtype int64
        Intersecting face-index pairs, or empty ``(0, 2)`` if none.
    """
    n_faces = mesh.faces.shape[0]
    if n_faces < 2:
        return 0, np.empty((0, 2), dtype=np.int64)

    candidates = _find_candidate_pairs(mesh)
    if candidates.shape[0] == 0:
        return 0, np.empty((0, 2), dtype=np.int64)

    triangles = mesh.vertices[mesh.faces]  # (F, 3, 3)
    hits = _batch_sat_test(triangles, candidates)
    intersecting = candidates[hits]
    return int(intersecting.shape[0]), intersecting
