"""Shared test fixtures — synthetic geometry for meshbench tests."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh


@pytest.fixture
def unit_cube() -> trimesh.Trimesh:
    """A 1x1x1 cube centered at origin."""
    return trimesh.creation.box(extents=[1, 1, 1])


@pytest.fixture
def icosphere() -> trimesh.Trimesh:
    """A unit icosphere (subdivisions=3)."""
    return trimesh.creation.icosphere(subdivisions=3)


@pytest.fixture
def tall_box() -> trimesh.Trimesh:
    """An elongated box (0.4 x 1.8 x 0.3)."""
    return trimesh.creation.box(extents=[0.4, 1.8, 0.3])


@pytest.fixture
def cylinder() -> trimesh.Trimesh:
    """A cylinder along Y axis, radius=0.2, height=2.0."""
    return trimesh.creation.cylinder(radius=0.2, height=2.0)


@pytest.fixture
def flat_plane() -> trimesh.Trimesh:
    """A flat quad (two triangles) in the XZ plane."""
    verts = np.array([
        [-1, 0, -1],
        [1, 0, -1],
        [1, 0, 1],
        [-1, 0, 1],
    ], dtype=float)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    return trimesh.Trimesh(vertices=verts, faces=faces)


@pytest.fixture
def open_box() -> trimesh.Trimesh:
    """A box with one face removed (not watertight, has boundary edges)."""
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    # Remove the last 2 triangles (= 1 quad face of the box)
    mesh.faces = mesh.faces[:-2]
    mesh.face_normals  # force cache rebuild
    return mesh


@pytest.fixture
def two_component_mesh() -> trimesh.Trimesh:
    """Two separated cubes as a single mesh (process=False to preserve both)."""
    a = trimesh.creation.box(extents=[1, 1, 1])
    b = trimesh.creation.box(extents=[1, 1, 1])
    b.apply_translation([5, 0, 0])
    combined = trimesh.util.concatenate([a, b])
    # Re-wrap with process=False to prevent vertex merging
    return trimesh.Trimesh(
        vertices=combined.vertices,
        faces=combined.faces,
        process=False,
    )


@pytest.fixture
def tapered_mesh() -> trimesh.Trimesh:
    """A tapered surface of revolution: radius 1.0 at y=0 to 0.1 at y=5."""
    n_rings = 20
    n_sectors = 32
    verts = []
    faces = []
    for i in range(n_rings):
        t = i / (n_rings - 1)
        y = t * 5.0
        r = 1.0 - 0.9 * t
        for j in range(n_sectors):
            theta = 2 * np.pi * j / n_sectors
            verts.append([r * np.cos(theta), y, r * np.sin(theta)])

    for i in range(n_rings - 1):
        for j in range(n_sectors):
            j_next = (j + 1) % n_sectors
            v0 = i * n_sectors + j
            v1 = i * n_sectors + j_next
            v2 = (i + 1) * n_sectors + j
            v3 = (i + 1) * n_sectors + j_next
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])

    return trimesh.Trimesh(
        vertices=np.array(verts, dtype=float),
        faces=np.array(faces),
        process=False,
    )
