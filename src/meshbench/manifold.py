"""Manifold analysis: boundary edges, non-manifold geometry, watertightness."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as sparse_cc
import trimesh

from meshbench._edges import (
    EdgeData,
    EdgeFaceMap,
    _build_sorted_edges,
    edge_face_counts as _vec_edge_face_counts,
)
from meshbench._intersections import detect_self_intersections
from meshbench.types import ManifoldReport

try:
    from meshbench._numba_kernels import _check_vertices_numba
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def _get_edge_data(
    mesh: trimesh.Trimesh,
    _edge_data: EdgeData | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (edges, counts), using pre-computed data if available."""
    if _edge_data is not None:
        return _edge_data
    return _vec_edge_face_counts(mesh)


def boundary_edges(
    mesh: trimesh.Trimesh,
    _edge_data: EdgeData | None = None,
) -> np.ndarray:
    """Return edges shared by exactly 1 face (boundary/open edges).

    Returns (N, 2) array of boundary edge vertex pairs.
    """
    edges, counts = _get_edge_data(mesh, _edge_data)
    return edges[counts == 1]


def boundary_loops(
    mesh: trimesh.Trimesh,
    _edge_data: EdgeData | None = None,
) -> int:
    """Count closed loops formed by boundary edges (= number of holes).

    Uses scipy.sparse connected_components for O(V+E) compiled C performance
    instead of Python-level BFS.
    """
    b_edges = boundary_edges(mesh, _edge_data)
    if len(b_edges) == 0:
        return 0

    # Remap vertex IDs to compact 0..n-1 range for sparse matrix
    unique_verts, inv = np.unique(b_edges, return_inverse=True)
    remapped = inv.reshape(b_edges.shape)  # (B, 2) with compact IDs
    n = len(unique_verts)

    rows = np.concatenate([remapped[:, 0], remapped[:, 1]])
    cols = np.concatenate([remapped[:, 1], remapped[:, 0]])
    data = np.ones(len(rows), dtype=np.int8)

    adj = csr_matrix((data, (rows, cols)), shape=(n, n))
    n_components, _ = sparse_cc(adj, directed=False)
    return int(n_components)


def non_manifold_edges(
    mesh: trimesh.Trimesh,
    _edge_data: EdgeData | None = None,
) -> np.ndarray:
    """Return edges shared by more than 2 faces.

    Returns (N, 2) array of non-manifold edge vertex pairs.
    """
    edges, counts = _get_edge_data(mesh, _edge_data)
    return edges[counts > 2]


def non_manifold_vertices(
    mesh: trimesh.Trimesh,
    _edge_face_map: EdgeFaceMap | None = None,
) -> list[int]:
    """Return vertices where the face fan doesn't form a single disc/cone.

    A vertex is non-manifold if the faces around it form more than one
    connected component when considering only face-adjacency through that vertex.

    Strategy: build a global edge→face-pair lookup (vectorized), then for each
    vertex that has >1 incident face, check connectivity of its face fan using
    the pre-built lookup and a simple union-find.

    Args:
        _edge_face_map: Optional pre-computed EdgeFaceMap to avoid redundant
            edge sorting (from build_edge_face_map).
    """
    faces = mesh.faces
    n_faces = faces.shape[0]
    if n_faces == 0:
        return []

    # --- Build edge → face pairs (vectorized) ---
    if _edge_face_map is not None:
        sorted_edges, sorted_face_ids, edge_starts, edge_ends = _edge_face_map
    else:
        all_edges = _build_sorted_edges(faces)  # (3F, 2)
        face_ids = np.tile(np.arange(n_faces), 3)

        max_v = int(faces.max()) + 1
        edge_keys = all_edges[:, 0].astype(np.int64) * max_v + all_edges[:, 1].astype(np.int64)

        order = np.argsort(edge_keys)
        sorted_face_ids = face_ids[order]
        sorted_edges = all_edges[order]

        sorted_keys = edge_keys[order]
        changes = np.where(np.diff(sorted_keys) != 0)[0] + 1
        edge_starts = np.concatenate([[0], changes])
        edge_ends = np.concatenate([changes, [len(sorted_keys)]])

    edge_sizes = edge_ends - edge_starts

    # --- Vectorized: edges with exactly 2 faces (the common case) ---
    # For each such edge, faces f1 and f2 are adjacent through both endpoints.
    exactly2 = edge_sizes == 2
    e2_idx = np.where(exactly2)[0]

    if len(e2_idx) > 0:
        e2_starts = edge_starts[e2_idx]
        f1 = sorted_face_ids[e2_starts]
        f2 = sorted_face_ids[e2_starts + 1]
        ev0 = sorted_edges[e2_starts, 0]
        ev1 = sorted_edges[e2_starts, 1]

        # Create adjacency arrays: (vertex, face_a, face_b) triples
        # Each edge contributes 2 triples (one per endpoint)
        adj_v = np.concatenate([ev0, ev1])
        adj_fa = np.concatenate([f1, f1])
        adj_fb = np.concatenate([f2, f2])
    else:
        adj_v = np.array([], dtype=np.intp)
        adj_fa = np.array([], dtype=np.intp)
        adj_fb = np.array([], dtype=np.intp)

    # Handle edges with >2 faces (non-manifold edges).
    # Only need a linear chain of unions (f[0]-f[1], f[1]-f[2], ...) per edge
    # to establish full connectivity — O(N) instead of O(N^2) pairs.
    gt2_idx = np.where(edge_sizes > 2)[0]
    if len(gt2_idx) > 0:
        extra_v, extra_fa, extra_fb = [], [], []
        for ei in gt2_idx:
            s, e = int(edge_starts[ei]), int(edge_ends[ei])
            efaces = sorted_face_ids[s:e]
            v0, v1 = int(sorted_edges[s, 0]), int(sorted_edges[s, 1])
            for ii in range(len(efaces) - 1):
                for vv in (v0, v1):
                    extra_v.append(vv)
                    extra_fa.append(int(efaces[ii]))
                    extra_fb.append(int(efaces[ii + 1]))
        if extra_v:
            adj_v = np.concatenate([adj_v, np.array(extra_v, dtype=np.intp)])
            adj_fa = np.concatenate([adj_fa, np.array(extra_fa, dtype=np.intp)])
            adj_fb = np.concatenate([adj_fb, np.array(extra_fb, dtype=np.intp)])

    # No shared edges → no adjacency info. Every vertex with >1 face
    # would be non-manifold, but we check below via check_indices.
    if len(adj_v) == 0:
        # Fall through: every vertex with >1 incident face is non-manifold
        vert_indices = faces.ravel()
        face_indices = np.repeat(np.arange(n_faces), 3)
        v_order = np.argsort(vert_indices)
        sv = vert_indices[v_order]
        v_changes = np.where(np.diff(sv) != 0)[0] + 1
        v_starts = np.concatenate([[0], v_changes])
        v_ends = np.concatenate([v_changes, [len(sv)]])
        v_group_verts = sv[v_starts]
        v_group_sizes = v_ends - v_starts
        return [int(v_group_verts[i]) for i in range(len(v_group_verts)) if v_group_sizes[i] > 1]

    # Sort adjacency by vertex for slicing
    a_order = np.argsort(adj_v)
    adj_v = adj_v[a_order]
    adj_fa = adj_fa[a_order]
    adj_fb = adj_fb[a_order]

    a_changes = np.where(np.diff(adj_v) != 0)[0] + 1
    a_starts = np.concatenate([[0], a_changes])
    a_ends = np.concatenate([a_changes, [len(adj_v)]])
    a_verts = adj_v[a_starts]

    # --- Build vertex → incident faces ---
    vert_indices = faces.ravel()
    face_indices = np.repeat(np.arange(n_faces), 3)
    v_order = np.argsort(vert_indices)
    sv = vert_indices[v_order]
    sf = face_indices[v_order]
    v_changes = np.where(np.diff(sv) != 0)[0] + 1
    v_starts = np.concatenate([[0], v_changes])
    v_ends = np.concatenate([v_changes, [len(sv)]])
    v_group_verts = sv[v_starts]
    v_group_sizes = v_ends - v_starts

    check_indices = np.where(v_group_sizes > 1)[0]
    if len(check_indices) == 0:
        return []

    # --- Check connectivity per vertex ---
    if _HAS_NUMBA:
        # JIT-compiled path: binary search on sorted arrays, no Python dicts
        result = _check_vertices_numba(
            check_indices.astype(np.int64),
            v_group_verts.astype(np.int64),
            v_starts.astype(np.int64),
            v_ends.astype(np.int64),
            sf.astype(np.int64),
            a_verts.astype(np.int64),
            a_starts.astype(np.int64),
            a_ends.astype(np.int64),
            adj_fa.astype(np.int64),
            adj_fb.astype(np.int64),
        )
        return result.tolist()

    # Python fallback: dict-based lookup + inline union-find
    a_lookup: dict[int, tuple[int, int]] = {}
    for i in range(len(a_verts)):
        a_lookup[int(a_verts[i])] = (int(a_starts[i]), int(a_ends[i]))

    non_manifold = []
    for idx in check_indices:
        v = int(v_group_verts[idx])
        incident = sf[v_starts[idx]:v_ends[idx]]
        n = len(incident)
        if n <= 1:
            continue

        bounds = a_lookup.get(v)
        if bounds is None:
            non_manifold.append(v)
            continue

        s, e = bounds
        local_fa = adj_fa[s:e]
        local_fb = adj_fb[s:e]

        # Map global face → local index
        face_map: dict[int, int] = {}
        for i in range(n):
            face_map[int(incident[i])] = i

        # Inline union-find
        parent = list(range(n))

        for k in range(len(local_fa)):
            la = face_map.get(int(local_fa[k]), -1)
            lb = face_map.get(int(local_fb[k]), -1)
            if la >= 0 and lb >= 0:
                x = la
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                ra = x
                x = lb
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                rb = x
                if ra != rb:
                    parent[ra] = rb

        roots = set()
        for i in range(n):
            x = i
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            roots.add(x)
        if len(roots) > 1:
            non_manifold.append(v)

    return non_manifold


def compute_manifold_report(
    mesh: trimesh.Trimesh,
    _edge_data: EdgeData | None = None,
    _edge_face_map: EdgeFaceMap | None = None,
) -> ManifoldReport:
    """Compute a full ManifoldReport for a mesh."""
    edge_data = _get_edge_data(mesh, _edge_data)
    total_edges = len(edge_data[0])
    total_verts = mesh.vertices.shape[0]

    n_faces = mesh.faces.shape[0]

    b_edges = boundary_edges(mesh, edge_data)
    b_loops = boundary_loops(mesh, edge_data)
    nm_edges = non_manifold_edges(mesh, edge_data)
    nm_verts = non_manifold_vertices(mesh, _edge_face_map=_edge_face_map)
    si_count, _si_pairs = detect_self_intersections(mesh)

    return ManifoldReport(
        is_watertight=len(b_edges) == 0,
        boundary_edge_count=len(b_edges),
        boundary_loop_count=b_loops,
        non_manifold_edge_count=len(nm_edges),
        non_manifold_vertex_count=len(nm_verts),
        non_manifold_edge_ratio=len(nm_edges) / total_edges if total_edges > 0 else 0.0,
        non_manifold_vertex_ratio=len(nm_verts) / total_verts if total_verts > 0 else 0.0,
        boundary_edge_ratio=len(b_edges) / total_edges if total_edges > 0 else 0.0,
        self_intersection_count=si_count,
        self_intersection_ratio=si_count / n_faces if n_faces > 0 else 0.0,
    )
