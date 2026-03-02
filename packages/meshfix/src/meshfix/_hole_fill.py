"""Hole filling: extract boundary loops, triangulate via ear clipping.

numpy backend: boundary walk + ear-clip triangulation.
pymeshlab backend: meshing_close_holes().
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import trimesh

from meshbench._edges import edge_face_counts


# ---------------------------------------------------------------------------
# Boundary loop extraction
# ---------------------------------------------------------------------------


def extract_boundary_loops(
    mesh: trimesh.Trimesh,
    boundary_edge_pairs: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Extract ordered boundary loops from boundary edges.

    Each loop is an array of vertex indices forming a closed chain.
    A manifold boundary vertex has exactly 2 boundary neighbours.

    Args:
        mesh: Input mesh.
        boundary_edge_pairs: Pre-computed boundary edges (N,2).
            If None, detected automatically.

    Returns:
        List of 1-D arrays, each an ordered loop of vertex indices.
    """
    if boundary_edge_pairs is None:
        edges, counts = edge_face_counts(mesh)
        boundary_edge_pairs = edges[counts == 1]

    if len(boundary_edge_pairs) == 0:
        return []

    # Build adjacency: vertex → set of neighbours via boundary edges
    adj: dict[int, list[int]] = defaultdict(list)
    for i in range(len(boundary_edge_pairs)):
        v0, v1 = int(boundary_edge_pairs[i, 0]), int(boundary_edge_pairs[i, 1])
        adj[v0].append(v1)
        adj[v1].append(v0)

    visited_edges: set[tuple[int, int]] = set()
    loops: list[np.ndarray] = []

    for start in adj:
        if not adj[start]:
            continue

        for first_neighbor in list(adj[start]):
            edge_key = (min(start, first_neighbor), max(start, first_neighbor))
            if edge_key in visited_edges:
                continue

            # Walk a loop
            loop = [start]
            visited_edges.add(edge_key)
            prev, current = start, first_neighbor

            while current != start:
                loop.append(current)
                # Find next: the neighbor of current that isn't prev
                # and whose edge hasn't been visited yet
                found_next = False
                for nb in adj[current]:
                    ek = (min(current, nb), max(current, nb))
                    if nb == prev:
                        # Only skip prev if we haven't used the reverse direction
                        if ek not in visited_edges:
                            pass  # Could still go back (non-manifold case)
                        else:
                            continue
                    if ek in visited_edges:
                        continue
                    visited_edges.add(ek)
                    prev, current = current, nb
                    found_next = True
                    break

                if not found_next:
                    break  # Broken chain, skip

            if current == start and len(loop) >= 3:
                loops.append(np.array(loop, dtype=np.intp))

    return loops


# ---------------------------------------------------------------------------
# Ear-clipping triangulation
# ---------------------------------------------------------------------------


def _project_to_2d(vertices_3d: np.ndarray) -> np.ndarray:
    """Project loop vertices onto best-fit plane via SVD.

    Returns (N, 2) array of 2D coordinates.
    """
    centroid = vertices_3d.mean(axis=0)
    centered = vertices_3d - centroid

    # SVD to find best-fit plane
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    # First two right singular vectors span the plane
    u_axis = vt[0]
    v_axis = vt[1]

    proj_2d = np.column_stack([centered @ u_axis, centered @ v_axis])
    return proj_2d


def _signed_area_2d(pts: np.ndarray) -> float:
    """Signed area of a 2D polygon (positive = CCW)."""
    n = len(pts)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _point_in_triangle_2d(
    p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> bool:
    """Check if point p is strictly inside triangle abc (2D)."""
    def cross_2d(o, aa, bb):
        return (aa[0] - o[0]) * (bb[1] - o[1]) - (aa[1] - o[1]) * (bb[0] - o[0])

    d1 = cross_2d(p, a, b)
    d2 = cross_2d(p, b, c)
    d3 = cross_2d(p, c, a)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)


def ear_clip_triangulate(loop_indices: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Triangulate a boundary loop via ear clipping.

    Args:
        loop_indices: Ordered vertex indices of the boundary loop.
        vertices: Full vertex array (N, 3).

    Returns:
        (T, 3) array of face indices (triangles filling the hole).
    """
    n = len(loop_indices)
    if n < 3:
        return np.empty((0, 3), dtype=np.intp)
    if n == 3:
        return loop_indices.reshape(1, 3)

    loop_verts_3d = vertices[loop_indices]
    pts_2d = _project_to_2d(loop_verts_3d)

    # Ensure CCW winding
    if _signed_area_2d(pts_2d) < 0:
        loop_indices = loop_indices[::-1].copy()
        pts_2d = pts_2d[::-1].copy()

    # Work with a mutable list of active indices
    active = list(range(n))
    triangles: list[tuple[int, int, int]] = []

    max_iters = n * n  # safety limit
    iters = 0

    while len(active) > 2 and iters < max_iters:
        iters += 1
        found_ear = False
        m = len(active)

        for i in range(m):
            prev_i = active[(i - 1) % m]
            curr_i = active[i]
            next_i = active[(i + 1) % m]

            a = pts_2d[prev_i]
            b = pts_2d[curr_i]
            c = pts_2d[next_i]

            # Check convexity (cross product > 0 for CCW)
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if cross <= 0:
                continue

            # Check no other active vertex inside this triangle
            is_ear = True
            for j in range(m):
                if j == (i - 1) % m or j == i or j == (i + 1) % m:
                    continue
                if _point_in_triangle_2d(pts_2d[active[j]], a, b, c):
                    is_ear = False
                    break

            if is_ear:
                triangles.append((
                    int(loop_indices[prev_i]),
                    int(loop_indices[curr_i]),
                    int(loop_indices[next_i]),
                ))
                active.pop(i)
                found_ear = True
                break

        if not found_ear:
            # Degenerate polygon — force-clip to avoid infinite loop
            if len(active) >= 3:
                triangles.append((
                    int(loop_indices[active[0]]),
                    int(loop_indices[active[1]]),
                    int(loop_indices[active[2]]),
                ))
                active.pop(1)
            else:
                break

    if not triangles:
        return np.empty((0, 3), dtype=np.intp)
    return np.array(triangles, dtype=np.intp)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _centroid_fan(
    loop_indices: np.ndarray,
    vertices: np.ndarray,
    centroid_idx: int,
) -> np.ndarray:
    """Create fan triangulation from boundary loop to a centroid vertex.

    Returns (N, 3) face array connecting each consecutive edge pair
    to the centroid.
    """
    n = len(loop_indices)
    tris = np.empty((n, 3), dtype=np.intp)
    tris[:, 0] = loop_indices
    tris[:, 1] = np.roll(loop_indices, -1)
    tris[:, 2] = centroid_idx
    return tris


def _ring_fill(
    loop_indices: np.ndarray,
    verts: list,
) -> np.ndarray:
    """Fill a hole with concentric rings converging to a centroid.

    Automatically chooses the number of intermediate rings so that
    radial edge lengths are comparable to boundary edge lengths,
    producing near-equilateral triangles throughout the patch.

    Args:
        loop_indices: Ordered boundary vertex indices.
        verts: Mutable vertex list (new vertices appended in place).

    Returns:
        (T, 3) face array for the fill patch.
    """
    n = len(loop_indices)
    boundary_pos = np.array([verts[i] for i in loop_indices])
    centroid = boundary_pos.mean(axis=0)

    # Compute average spoke and boundary edge lengths
    spoke_vecs = boundary_pos - centroid
    avg_spoke = float(np.mean(np.linalg.norm(spoke_vecs, axis=1)))
    edge_vecs = np.roll(boundary_pos, -1, axis=0) - boundary_pos
    avg_edge = float(np.mean(np.linalg.norm(edge_vecs, axis=1)))

    if avg_edge < 1e-30:
        return np.empty((0, 3), dtype=np.intp)

    # Number of rings so radial step ≈ avg boundary edge length
    n_rings = max(0, round(avg_spoke / avg_edge) - 1)
    n_rings = min(n_rings, 2)  # cap — diminishing returns beyond 2

    if n_rings == 0:
        # Simple fan — spoke is already comparable to boundary edge
        centroid_idx = len(verts)
        verts.append(centroid)
        return _centroid_fan(loop_indices, verts, centroid_idx)

    # Build rings: interpolate from boundary toward centroid
    rings: list[np.ndarray] = [loop_indices]  # ring 0 = boundary

    for r in range(1, n_rings + 1):
        t = r / (n_rings + 1)  # 0 at boundary, 1 at centroid
        ring_indices = np.empty(n, dtype=np.intp)
        for i in range(n):
            pos = (1 - t) * np.array(verts[loop_indices[i]]) + t * centroid
            ring_indices[i] = len(verts)
            verts.append(pos)
        rings.append(ring_indices)

    # Add centroid
    centroid_idx = len(verts)
    verts.append(centroid)

    tris: list[tuple[int, int, int]] = []

    # Triangle strips between consecutive rings (same vertex count)
    for r in range(len(rings) - 1):
        outer = rings[r]
        inner = rings[r + 1]
        for i in range(n):
            j = (i + 1) % n
            # Two triangles per quad
            tris.append((int(outer[i]), int(outer[j]), int(inner[i])))
            tris.append((int(inner[i]), int(outer[j]), int(inner[j])))

    # Final fan from innermost ring to centroid
    innermost = rings[-1]
    for i in range(n):
        j = (i + 1) % n
        tris.append((int(innermost[i]), int(innermost[j]), centroid_idx))

    return np.array(tris, dtype=np.intp)


_RING_FILL_THRESHOLD = 8  # use ring fill for holes with >= this many edges


def fill_holes(
    mesh: trimesh.Trimesh,
    boundary_edge_pairs: np.ndarray | None = None,
    max_hole_edges: int = 100,
) -> trimesh.Trimesh:
    """Fill holes by triangulating boundary loops.

    Small holes (< 8 edges) use ear-clipping for exact boundary-only fill.
    Larger holes use centroid-fan with iterative refinement: inserts a centroid
    vertex and fans from each boundary edge to it, then subdivides any
    triangles with aspect ratio > 2.5 by splitting their longest edge.

    Args:
        mesh: Input mesh (not mutated).
        boundary_edge_pairs: Pre-computed boundary edges (N,2) from
            Defects.boundary_edges. If None, detected automatically.
        max_hole_edges: Skip holes with more edges than this.

    Returns:
        New Trimesh with holes filled.
    """
    faces = mesh.faces
    vertices = mesh.vertices

    if faces.shape[0] == 0:
        return trimesh.Trimesh(
            vertices=vertices.copy(), faces=faces.copy(), process=False
        )

    loops = extract_boundary_loops(mesh, boundary_edge_pairs)

    if not loops:
        return trimesh.Trimesh(
            vertices=vertices.copy(), faces=faces.copy(), process=False
        )

    new_verts = list(vertices)  # mutable copy, may grow
    new_face_blocks: list[np.ndarray] = [faces.copy()]

    for loop in loops:
        if len(loop) > max_hole_edges:
            continue

        if len(loop) >= _RING_FILL_THRESHOLD:
            # Concentric ring fill with adaptive ring count
            tri = _ring_fill(loop, new_verts)
        else:
            tri = ear_clip_triangulate(loop, vertices)

        if len(tri) > 0:
            new_face_blocks.append(tri)

    if len(new_face_blocks) == 1:
        return trimesh.Trimesh(
            vertices=vertices.copy(), faces=faces.copy(), process=False
        )

    all_verts = np.array(new_verts, dtype=np.float64)
    all_faces = np.vstack(new_face_blocks)
    return trimesh.Trimesh(vertices=all_verts, faces=all_faces, process=False)
