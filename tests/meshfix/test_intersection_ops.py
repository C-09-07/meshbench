"""Tests for self-intersection resolution."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from meshbench._intersections import detect_self_intersections
from meshfix._intersection_ops import resolve_self_intersections


class TestResolveSelfIntersections:
    def test_removes_intersecting_faces(self, self_intersecting_mesh):
        """Lossy: intersecting faces removed."""
        count_before, _ = detect_self_intersections(self_intersecting_mesh)
        assert count_before > 0

        result = resolve_self_intersections(self_intersecting_mesh)

        # Fewer faces (both removed)
        assert result.faces.shape[0] < self_intersecting_mesh.faces.shape[0]

        # No self-intersections remain
        count_after, _ = detect_self_intersections(result)
        assert count_after == 0

    def test_noop_on_clean_mesh(self, unit_cube):
        """Manifold mesh with no SI -> copy unchanged."""
        result = resolve_self_intersections(unit_cube)

        assert result.faces.shape[0] == unit_cube.faces.shape[0]
        assert result.vertices.shape[0] == unit_cube.vertices.shape[0]

    def test_with_precomputed_pairs(self, self_intersecting_mesh):
        """Pass pre-computed intersection pairs."""
        _, pairs = detect_self_intersections(self_intersecting_mesh)
        result = resolve_self_intersections(
            self_intersecting_mesh, intersection_pairs=pairs
        )

        count_after, _ = detect_self_intersections(result)
        assert count_after == 0

    def test_does_not_mutate_input(self, self_intersecting_mesh):
        """Input mesh not modified."""
        orig_verts = self_intersecting_mesh.vertices.copy()
        orig_faces = self_intersecting_mesh.faces.copy()

        resolve_self_intersections(self_intersecting_mesh)

        np.testing.assert_array_equal(self_intersecting_mesh.vertices, orig_verts)
        np.testing.assert_array_equal(self_intersecting_mesh.faces, orig_faces)

    def test_empty_mesh(self):
        """Empty mesh -> no-op."""
        mesh = trimesh.Trimesh(
            vertices=np.empty((0, 3)),
            faces=np.empty((0, 3), dtype=int),
            process=False,
        )
        result = resolve_self_intersections(mesh)
        assert result.faces.shape[0] == 0
