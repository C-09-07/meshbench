"""Manifold analysis: boundary edges, non-manifold geometry, watertightness."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import trimesh

from meshbench.types import ManifoldReport


def _edge_face_counts(mesh: trimesh.Trimesh) -> dict[tuple[int, int], int]:
    """Build a map from sorted edge tuples to face count."""
    edge_counts: dict[tuple[int, int], int] = defaultdict(int)
    for face in mesh.faces:
        for i in range(3):
            v0, v1 = int(face[i]), int(face[(i + 1) % 3])
            edge = (min(v0, v1), max(v0, v1))
            edge_counts[edge] += 1
    return edge_counts


def boundary_edges(mesh: trimesh.Trimesh) -> list[tuple[int, int]]:
    """Return edges shared by exactly 1 face (boundary/open edges)."""
    counts = _edge_face_counts(mesh)
    return [edge for edge, count in counts.items() if count == 1]


def boundary_loops(mesh: trimesh.Trimesh) -> int:
    """Count closed loops formed by boundary edges (= number of holes)."""
    edges = boundary_edges(mesh)
    if not edges:
        return 0

    # Build adjacency from boundary edges
    adj: dict[int, list[int]] = defaultdict(list)
    for v0, v1 in edges:
        adj[v0].append(v1)
        adj[v1].append(v0)

    # Count connected components via BFS
    visited: set[int] = set()
    loop_count = 0
    for start in adj:
        if start in visited:
            continue
        loop_count += 1
        queue = [start]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    queue.append(neighbor)

    return loop_count


def non_manifold_edges(mesh: trimesh.Trimesh) -> list[tuple[int, int]]:
    """Return edges shared by more than 2 faces."""
    counts = _edge_face_counts(mesh)
    return [edge for edge, count in counts.items() if count > 2]


def non_manifold_vertices(mesh: trimesh.Trimesh) -> list[int]:
    """Return vertices where the face fan doesn't form a single disc/cone.

    A vertex is non-manifold if the faces around it form more than one
    connected component when considering only face-adjacency through that vertex.
    """
    if mesh.faces.shape[0] == 0:
        return []

    # Build vertex → face index map
    vert_faces: dict[int, set[int]] = defaultdict(set)
    for fi, face in enumerate(mesh.faces):
        for v in face:
            vert_faces[int(v)].add(fi)

    # Build face adjacency restricted to shared vertex
    non_manifold = []
    for v, face_set in vert_faces.items():
        if len(face_set) <= 1:
            continue

        # Two faces are connected through v if they share v AND another vertex
        face_list = list(face_set)
        face_verts = {fi: set(int(x) for x in mesh.faces[fi]) for fi in face_list}

        adj: dict[int, set[int]] = defaultdict(set)
        for i, fi in enumerate(face_list):
            for j in range(i + 1, len(face_list)):
                fj = face_list[j]
                # Shared verts besides v
                shared = face_verts[fi] & face_verts[fj] - {v}
                if shared:
                    adj[fi].add(fj)
                    adj[fj].add(fi)

        # BFS to check connectivity
        visited: set[int] = set()
        queue = [face_list[0]]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    queue.append(neighbor)

        if len(visited) < len(face_set):
            non_manifold.append(v)

    return non_manifold


def compute_manifold_report(mesh: trimesh.Trimesh) -> ManifoldReport:
    """Compute a full ManifoldReport for a mesh."""
    b_edges = boundary_edges(mesh)
    b_loops = boundary_loops(mesh)
    nm_edges = non_manifold_edges(mesh)
    nm_verts = non_manifold_vertices(mesh)

    return ManifoldReport(
        is_watertight=len(b_edges) == 0,
        boundary_edge_count=len(b_edges),
        boundary_loop_count=b_loops,
        non_manifold_edge_count=len(nm_edges),
        non_manifold_vertex_count=len(nm_verts),
    )
