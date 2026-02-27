"""Mesh loading utilities."""

from __future__ import annotations

from pathlib import Path

import trimesh


def load(path: str | Path, force_geometry: bool = False) -> trimesh.Trimesh:
    """Load a mesh file, flattening scenes to a single Trimesh.

    Args:
        path: Path to mesh file (STL, OBJ, GLB, PLY, etc.).
        force_geometry: If True, strip textures/materials and rebuild
            a bare Trimesh from vertices and faces only.

    Returns:
        A trimesh.Trimesh instance.

    Raises:
        ValueError: If the file cannot be loaded or contains no geometry.
    """
    path = Path(path)
    try:
        result = trimesh.load(str(path))
    except Exception as exc:
        raise ValueError(f"Failed to load mesh: {path}") from exc

    mesh: trimesh.Trimesh | None = None

    if isinstance(result, trimesh.Scene):
        geometries = list(result.geometry.values())
        if not geometries:
            raise ValueError(f"Empty scene: {path}")
        mesh = trimesh.util.concatenate(geometries)
    elif isinstance(result, trimesh.Trimesh):
        mesh = result
    else:
        raise ValueError(
            f"Unexpected type {type(result).__name__} from {path}"
        )

    if force_geometry:
        mesh = trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=mesh.faces,
            process=False,
        )

    return mesh
