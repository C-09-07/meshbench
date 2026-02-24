"""Topological analysis: genus, valence, degenerates, floating vertices, components."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import trimesh

from meshbench.types import TopologyReport


def genus(mesh: trimesh.Trimesh) -> int:
    """Compute topological genus from the Euler characteristic.

    V - E + F = 2(1 - g) for a single connected closed surface.
    For open or multi-component meshes this is a generalized genus.
    """
    v = mesh.vertices.shape[0]
    # Count unique edges
    edge_set: set[tuple[int, int]] = set()
    for face in mesh.faces:
        for i in range(3):
            v0, v1 = int(face[i]), int(face[(i + 1) % 3])
            edge_set.add((min(v0, v1), max(v0, v1)))
    e = len(edge_set)
    f = mesh.faces.shape[0]

    # V - E + F = 2(1 - g)  =>  g = 1 - (V - E + F) / 2
    euler = v - e + f
    g = 1 - euler // 2
    return max(0, g)


def valence_histogram(mesh: trimesh.Trimesh) -> dict[int, int]:
    """Compute vertex valence histogram.

    Returns dict mapping valence → count of vertices with that valence.
    """
    counter: Counter[int] = Counter()
    vert_edges: dict[int, set[int]] = defaultdict(set)

    for face in mesh.faces:
        for i in range(3):
            v0, v1 = int(face[i]), int(face[(i + 1) % 3])
            vert_edges[v0].add(v1)
            vert_edges[v1].add(v0)

    for v in range(mesh.vertices.shape[0]):
        counter[len(vert_edges[v])] += 1

    return dict(sorted(counter.items()))


def valence_stats(
    mesh: trimesh.Trimesh,
) -> tuple[float, float, float, int]:
    """Compute valence statistics.

    Returns:
        (mean, std, quad_compatible_ratio, outlier_count)
        where quad_compatible_ratio is fraction of vertices at valence 4
        and outlier_count is vertices with valence >= 10.
    """
    hist = valence_histogram(mesh)
    if not hist:
        return 0.0, 0.0, 0.0, 0

    total_verts = sum(hist.values())
    valences = []
    for val, count in hist.items():
        valences.extend([val] * count)

    arr = np.array(valences, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr))

    quad_count = hist.get(4, 0)
    quad_ratio = quad_count / total_verts if total_verts > 0 else 0.0

    outlier_count = sum(count for val, count in hist.items() if val >= 10)

    return mean, std, quad_ratio, outlier_count


def degenerate_faces(mesh: trimesh.Trimesh, area_threshold: float = 1e-10) -> list[int]:
    """Return indices of zero/near-zero area triangles."""
    areas = mesh.area_faces
    return [int(i) for i in np.where(areas < area_threshold)[0]]


def floating_vertices(mesh: trimesh.Trimesh) -> list[int]:
    """Return vertex indices not referenced by any face."""
    if mesh.faces.shape[0] == 0:
        return list(range(mesh.vertices.shape[0]))

    referenced = set(mesh.faces.flatten())
    return [i for i in range(mesh.vertices.shape[0]) if i not in referenced]


def component_sizes(mesh: trimesh.Trimesh) -> list[tuple[int, int]]:
    """Return (face_count, vertex_count) per connected component, sorted descending."""
    try:
        components = mesh.split()
        if not components:
            return [(mesh.faces.shape[0], mesh.vertices.shape[0])]
    except Exception:
        return [(mesh.faces.shape[0], mesh.vertices.shape[0])]

    sizes = [(c.faces.shape[0], c.vertices.shape[0]) for c in components]
    sizes.sort(reverse=True)
    return sizes


def compute_topology_report(mesh: trimesh.Trimesh) -> TopologyReport:
    """Compute a full TopologyReport for a mesh."""
    g = genus(mesh)
    hist = valence_histogram(mesh)
    mean, std, quad_ratio, outlier_count = valence_stats(mesh)
    degenerates = degenerate_faces(mesh)
    floating = floating_vertices(mesh)
    comp_sizes = component_sizes(mesh)

    return TopologyReport(
        genus=g,
        valence_mean=round(mean, 2),
        valence_std=round(std, 2),
        valence_histogram=hist,
        quad_compatible_ratio=round(quad_ratio, 4),
        valence_outlier_count=outlier_count,
        degenerate_face_count=len(degenerates),
        floating_vertex_count=len(floating),
        component_count=len(comp_sizes),
        component_sizes=comp_sizes,
    )
