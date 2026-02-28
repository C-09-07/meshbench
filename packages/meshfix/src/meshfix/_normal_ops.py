"""Normal and winding repair operations (trimesh wrappers)."""

from __future__ import annotations

import trimesh


def fix_winding(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Fix inconsistent face winding.

    Args:
        mesh: Input mesh (not mutated).

    Returns:
        New Trimesh with consistent winding.
    """
    new = mesh.copy()
    trimesh.repair.fix_winding(new)
    return new


def fix_normals(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Fix face normals to point outward.

    Args:
        mesh: Input mesh (not mutated).

    Returns:
        New Trimesh with outward-pointing normals.
    """
    new = mesh.copy()
    trimesh.repair.fix_normals(new)
    return new
