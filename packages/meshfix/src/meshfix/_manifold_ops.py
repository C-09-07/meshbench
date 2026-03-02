"""Non-manifold topology repair: split NM edges and NM vertices.

numpy backend: vertex duplication + face remapping.
pymeshlab backend: delegated repair methods.
"""

from __future__ import annotations

import numpy as np
import trimesh

from meshbench._edges import (
    EdgeFaceMap,
    build_edge_face_map,
    edge_face_counts,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fan_components(
    vertex: int,
    incident_faces: np.ndarray,
    adj_fa: np.ndarray,
    adj_fb: np.ndarray,
) -> list[list[int]]:
    """Decompose the face fan around *vertex* into connected components.

    Uses union-find over face adjacency (same algorithm as
    ``manifold.py:non_manifold_vertices`` lines 258-284, but returns
    component groups instead of just a boolean).

    Args:
        vertex: The vertex index (unused here, kept for clarity).
        incident_faces: Array of face indices incident to this vertex.
        adj_fa: Array of face-A indices from adjacency triples for this vertex.
        adj_fb: Array of face-B indices from adjacency triples for this vertex.

    Returns:
        List of face-index groups (each group is a connected component).
    """
    n = len(incident_faces)
    if n <= 1:
        return [incident_faces.tolist()]

    # Map global face index → local index
    face_map: dict[int, int] = {}
    for i in range(n):
        face_map[int(incident_faces[i])] = i

    # Union-find
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for k in range(len(adj_fa)):
        la = face_map.get(int(adj_fa[k]), -1)
        lb = face_map.get(int(adj_fb[k]), -1)
        if la >= 0 and lb >= 0:
            union(la, lb)

    # Collect components
    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(int(incident_faces[i]))
    return list(groups.values())


def _build_vertex_adjacency(faces: np.ndarray, edge_face_map: EdgeFaceMap):
    """Build per-vertex adjacency arrays from an EdgeFaceMap.

    Returns (adj_v, adj_fa, adj_fb, a_starts, a_ends, a_verts) —
    the same structure used by manifold.py for NM vertex detection.
    """
    sorted_edges, sorted_face_ids, edge_starts, edge_ends = edge_face_map
    edge_sizes = edge_ends - edge_starts

    # Edges with exactly 2 faces
    exactly2 = edge_sizes == 2
    e2_idx = np.where(exactly2)[0]

    if len(e2_idx) > 0:
        e2_starts = edge_starts[e2_idx]
        f1 = sorted_face_ids[e2_starts]
        f2 = sorted_face_ids[e2_starts + 1]
        ev0 = sorted_edges[e2_starts, 0]
        ev1 = sorted_edges[e2_starts, 1]

        adj_v = np.concatenate([ev0, ev1])
        adj_fa = np.concatenate([f1, f1])
        adj_fb = np.concatenate([f2, f2])
    else:
        adj_v = np.array([], dtype=np.intp)
        adj_fa = np.array([], dtype=np.intp)
        adj_fb = np.array([], dtype=np.intp)

    # Edges with >2 faces (NM edges) — chain unions
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

    if len(adj_v) == 0:
        return adj_v, adj_fa, adj_fb, np.array([], dtype=np.intp), np.array([], dtype=np.intp), np.array([], dtype=np.intp)

    # Sort by vertex for slicing
    a_order = np.argsort(adj_v)
    adj_v = adj_v[a_order]
    adj_fa = adj_fa[a_order]
    adj_fb = adj_fb[a_order]

    a_changes = np.where(np.diff(adj_v) != 0)[0] + 1
    a_starts = np.concatenate([[0], a_changes])
    a_ends = np.concatenate([a_changes, [len(adj_v)]])
    a_verts = adj_v[a_starts]

    return adj_v, adj_fa, adj_fb, a_starts, a_ends, a_verts


# ---------------------------------------------------------------------------
# split_non_manifold_edges
# ---------------------------------------------------------------------------


def split_non_manifold_edges(
    mesh: trimesh.Trimesh,
    edge_pairs: np.ndarray | None = None,
) -> trimesh.Trimesh:
    """Split non-manifold edges by duplicating shared vertices.

    For each NM edge (shared by >2 faces), faces are grouped into "sheets"
    by local connectivity through non-NM shared edges. The first sheet keeps
    original vertices; each additional sheet gets copies.

    Args:
        mesh: Input mesh (not mutated).
        edge_pairs: Pre-computed NM edge pairs (N,2) from Defects.non_manifold_edges.
            If None, detected automatically.

    Returns:
        New Trimesh with NM edges resolved.
    """
    faces = mesh.faces
    vertices = mesh.vertices
    n_faces = faces.shape[0]

    if n_faces == 0:
        return trimesh.Trimesh(
            vertices=vertices.copy(), faces=faces.copy(), process=False
        )

    # Detect NM edges
    if edge_pairs is None:
        edges, counts = edge_face_counts(mesh)
        nm_mask = counts > 2
        if not np.any(nm_mask):
            return trimesh.Trimesh(
                vertices=vertices.copy(), faces=faces.copy(), process=False
            )
        edge_pairs = edges[nm_mask]
    elif len(edge_pairs) == 0:
        return trimesh.Trimesh(
            vertices=vertices.copy(), faces=faces.copy(), process=False
        )

    # Build edge→face map for grouping
    efm = build_edge_face_map(faces)
    sorted_edges, sorted_face_ids, edge_starts, edge_ends = efm

    # Build a set of NM edge keys for fast lookup
    max_v = int(faces.max()) + 1
    nm_keys = set(
        (int(edge_pairs[i, 0]) * max_v + int(edge_pairs[i, 1]))
        for i in range(len(edge_pairs))
    )

    # For each NM edge, get its face list and group faces into sheets
    # by connectivity through non-NM edges only
    new_faces = faces.copy()
    extra_verts: list[np.ndarray] = []
    next_vert_id = vertices.shape[0]

    for i in range(len(edge_pairs)):
        v0, v1 = int(edge_pairs[i, 0]), int(edge_pairs[i, 1])
        ekey = v0 * max_v + v1

        # Find this edge in the sorted map
        target_key = np.int64(v0) * max_v + np.int64(v1)
        sorted_keys = sorted_edges[:, 0].astype(np.int64) * max_v + sorted_edges[:, 1].astype(np.int64)

        # Binary search: find where this edge group starts
        group_mask = np.zeros(len(edge_starts), dtype=bool)
        for gi in range(len(edge_starts)):
            if sorted_keys[edge_starts[gi]] == target_key:
                group_mask[gi] = True
                break

        if not np.any(group_mask):
            continue

        gi = int(np.where(group_mask)[0][0])
        s, e = int(edge_starts[gi]), int(edge_ends[gi])
        face_list = sorted_face_ids[s:e]

        if len(face_list) <= 2:
            continue

        # Group faces into sheets via adjacency through non-NM edges.
        # Two faces in face_list are in the same sheet if they share
        # a non-NM edge that is NOT this NM edge.
        n = len(face_list)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        face_to_local: dict[int, int] = {int(face_list[k]): k for k in range(n)}

        # Check all non-NM edges of each face for shared faces in face_list
        for k in range(n):
            fi = int(face_list[k])
            f_verts = new_faces[fi]
            for ei_local in range(3):
                ev0 = int(min(f_verts[ei_local], f_verts[(ei_local + 1) % 3]))
                ev1 = int(max(f_verts[ei_local], f_verts[(ei_local + 1) % 3]))
                ek = ev0 * max_v + ev1
                if ek == ekey:
                    continue  # Skip the NM edge itself

                # Find faces sharing this edge via the efm
                ek64 = np.int64(ev0) * max_v + np.int64(ev1)
                for gi2 in range(len(edge_starts)):
                    if sorted_keys[edge_starts[gi2]] == ek64:
                        s2, e2 = int(edge_starts[gi2]), int(edge_ends[gi2])
                        for fi2 in sorted_face_ids[s2:e2]:
                            fi2 = int(fi2)
                            if fi2 in face_to_local and fi2 != fi:
                                union(face_to_local[fi], face_to_local[fi2])
                        break

        # Collect sheets
        sheets: dict[int, list[int]] = {}
        for k in range(n):
            root = find(k)
            sheets.setdefault(root, []).append(int(face_list[k]))

        sheet_list = list(sheets.values())
        if len(sheet_list) <= 1:
            continue

        # First sheet keeps original vertices. Each additional sheet
        # gets copies of v0 and v1.
        for sheet in sheet_list[1:]:
            new_v0 = next_vert_id
            new_v1 = next_vert_id + 1
            extra_verts.append(vertices[v0:v0+1])
            extra_verts.append(vertices[v1:v1+1])
            next_vert_id += 2

            for fi in sheet:
                f = new_faces[fi]
                for vi in range(3):
                    if f[vi] == v0:
                        new_faces[fi, vi] = new_v0
                    elif f[vi] == v1:
                        new_faces[fi, vi] = new_v1

    if extra_verts:
        all_verts = np.vstack([vertices, *extra_verts])
    else:
        all_verts = vertices.copy()

    return trimesh.Trimesh(vertices=all_verts, faces=new_faces, process=False)


# ---------------------------------------------------------------------------
# split_non_manifold_vertices
# ---------------------------------------------------------------------------


def split_non_manifold_vertices(
    mesh: trimesh.Trimesh,
    vert_indices: np.ndarray | None = None,
) -> trimesh.Trimesh:
    """Split non-manifold vertices by duplicating them per face-fan component.

    A vertex is non-manifold when its incident faces form >1 connected component.
    Each additional component gets a copy of the vertex.

    Args:
        mesh: Input mesh (not mutated).
        vert_indices: Pre-computed NM vertex indices from Defects.non_manifold_vertices.
            If None, detected automatically.

    Returns:
        New Trimesh with NM vertices resolved.
    """
    faces = mesh.faces
    vertices = mesh.vertices
    n_faces = faces.shape[0]

    if n_faces == 0:
        return trimesh.Trimesh(
            vertices=vertices.copy(), faces=faces.copy(), process=False
        )

    # Build edge→face map
    efm = build_edge_face_map(faces)
    _, adj_fa, adj_fb, a_starts, a_ends, a_verts = _build_vertex_adjacency(faces, efm)

    # Detect NM vertices if not provided
    if vert_indices is None:
        from meshbench.manifold import non_manifold_vertices as detect_nm_verts
        vert_indices = np.array(detect_nm_verts(mesh, _edge_face_map=efm), dtype=np.intp)

    if len(vert_indices) == 0:
        return trimesh.Trimesh(
            vertices=vertices.copy(), faces=faces.copy(), process=False
        )

    # Build vertex→incident faces mapping
    vert_idx_flat = faces.ravel()
    face_idx_flat = np.repeat(np.arange(n_faces), 3)
    v_order = np.argsort(vert_idx_flat)
    sv = vert_idx_flat[v_order]
    sf = face_idx_flat[v_order]
    v_changes = np.where(np.diff(sv) != 0)[0] + 1
    v_starts = np.concatenate([[0], v_changes])
    v_ends = np.concatenate([v_changes, [len(sv)]])
    v_group_verts = sv[v_starts]

    # Build lookup: vertex → (start, end) in sorted incident faces
    v_lookup: dict[int, tuple[int, int]] = {}
    for i in range(len(v_group_verts)):
        v_lookup[int(v_group_verts[i])] = (int(v_starts[i]), int(v_ends[i]))

    # Build adjacency lookup: vertex → (start, end) in adj arrays
    a_lookup: dict[int, tuple[int, int]] = {}
    for i in range(len(a_verts)):
        a_lookup[int(a_verts[i])] = (int(a_starts[i]), int(a_ends[i]))

    new_faces = faces.copy()
    extra_verts: list[np.ndarray] = []
    next_vert_id = vertices.shape[0]

    for v in vert_indices:
        v = int(v)
        bounds = v_lookup.get(v)
        if bounds is None:
            continue
        incident = sf[bounds[0]:bounds[1]]
        if len(incident) <= 1:
            continue

        # Get adjacency triples for this vertex
        a_bounds = a_lookup.get(v)
        if a_bounds is not None:
            s, e = a_bounds
            local_fa = adj_fa[s:e]
            local_fb = adj_fb[s:e]
        else:
            local_fa = np.array([], dtype=np.intp)
            local_fb = np.array([], dtype=np.intp)

        components = _fan_components(v, incident, local_fa, local_fb)

        if len(components) <= 1:
            continue

        # First component keeps original vertex
        for comp in components[1:]:
            new_v = next_vert_id
            extra_verts.append(vertices[v:v+1])
            next_vert_id += 1

            for fi in comp:
                f = new_faces[fi]
                for vi in range(3):
                    if f[vi] == v:
                        new_faces[fi, vi] = new_v

    if extra_verts:
        all_verts = np.vstack([vertices, *extra_verts])
    else:
        all_verts = vertices.copy()

    return trimesh.Trimesh(vertices=all_verts, faces=new_faces, process=False)
