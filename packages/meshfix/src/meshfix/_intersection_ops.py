"""Self-intersection resolution.

manifold3d backend (lossless): self-union via Manifold constructor.
numpy backend (lossy): remove all faces involved in self-intersections.
"""

from __future__ import annotations

import numpy as np
import trimesh

from meshbench._intersections import detect_self_intersections
from meshfix._numpy_ops import _compact


# ---------------------------------------------------------------------------
# manifold3d backend
# ---------------------------------------------------------------------------

_HAS_MANIFOLD3D = False
try:
    import manifold3d  # noqa: F401

    _HAS_MANIFOLD3D = True
except ImportError:
    pass


def _resolve_manifold3d(mesh: trimesh.Trimesh) -> trimesh.Trimesh | None:
    """Try to resolve self-intersections via manifold3d self-union.

    Returns None if manifold3d is not available or the mesh can't be imported.
    """
    if not _HAS_MANIFOLD3D:
        return None

    try:
        from manifold3d import Manifold, Mesh as M3Mesh

        m3_mesh = M3Mesh(
            vert_properties=np.array(mesh.vertices, dtype=np.float32),
            tri_verts=np.array(mesh.faces, dtype=np.uint32),
        )
        m = Manifold(m3_mesh)
        out = m.to_mesh()
        return trimesh.Trimesh(
            vertices=np.array(out.vert_properties[:, :3], dtype=np.float64),
            faces=np.array(out.tri_verts, dtype=np.intp),
            process=False,
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# numpy backend (lossy)
# ---------------------------------------------------------------------------


def _resolve_numpy(
    mesh: trimesh.Trimesh,
    intersection_pairs: np.ndarray | None = None,
) -> trimesh.Trimesh:
    """Remove all faces involved in self-intersections.

    This is lossy — it creates holes where intersecting faces were.
    The pipeline order means fill_holes ran before this step, and the
    caller can re-run fix() if needed.
    """
    if intersection_pairs is None:
        _, intersection_pairs = detect_self_intersections(mesh)

    if len(intersection_pairs) == 0:
        return trimesh.Trimesh(
            vertices=mesh.vertices.copy(),
            faces=mesh.faces.copy(),
            process=False,
        )

    # Collect all unique face indices involved
    bad_faces = np.unique(intersection_pairs.ravel())

    keep_mask = np.ones(mesh.faces.shape[0], dtype=bool)
    keep_mask[bad_faces] = False
    new_faces = mesh.faces[keep_mask]

    return _compact(mesh.vertices, new_faces)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_self_intersections(
    mesh: trimesh.Trimesh,
    intersection_pairs: np.ndarray | None = None,
) -> trimesh.Trimesh:
    """Resolve self-intersecting faces.

    Tries manifold3d (lossless self-union) first, falls back to numpy
    (lossy face removal).

    Args:
        mesh: Input mesh (not mutated).
        intersection_pairs: Pre-computed intersection face pairs (K,2) from
            Defects.self_intersection_pairs. If None, detected automatically.

    Returns:
        New Trimesh with self-intersections resolved.
    """
    faces = mesh.faces
    vertices = mesh.vertices

    if faces.shape[0] == 0:
        return trimesh.Trimesh(
            vertices=vertices.copy(), faces=faces.copy(), process=False
        )

    # Detect if there are actually self-intersections
    if intersection_pairs is None:
        count, intersection_pairs = detect_self_intersections(mesh)
        if count == 0:
            return trimesh.Trimesh(
                vertices=vertices.copy(), faces=faces.copy(), process=False
            )
    elif len(intersection_pairs) == 0:
        return trimesh.Trimesh(
            vertices=vertices.copy(), faces=faces.copy(), process=False
        )

    # Try manifold3d first (lossless)
    result = _resolve_manifold3d(mesh)
    if result is not None:
        return result

    # Fall back to numpy (lossy)
    return _resolve_numpy(mesh, intersection_pairs)
