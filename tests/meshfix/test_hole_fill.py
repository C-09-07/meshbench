"""Tests for hole filling: boundary loop extraction and ear-clip triangulation."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from meshbench._edges import edge_face_counts
from meshfix._hole_fill import (
    ear_clip_triangulate,
    extract_boundary_loops,
    fill_holes,
)


class TestExtractBoundaryLoops:
    def test_single_loop_from_open_box(self, open_box):
        """Open box has 1 boundary loop."""
        loops = extract_boundary_loops(open_box)

        assert len(loops) == 1
        assert len(loops[0]) >= 3

    def test_no_boundary_on_watertight(self, unit_cube):
        """Watertight mesh -> no loops."""
        loops = extract_boundary_loops(unit_cube)
        assert len(loops) == 0

    def test_loops_are_closed(self, open_box):
        """Each loop is a closed chain (start != end but chain wraps)."""
        loops = extract_boundary_loops(open_box)
        for loop in loops:
            # Verify all vertices in the loop are unique
            assert len(loop) == len(set(loop.tolist()))


class TestEarClipTriangulate:
    def test_triangle_loop(self):
        """3-vertex loop -> 1 triangle."""
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        loop = np.array([0, 1, 2], dtype=np.intp)

        result = ear_clip_triangulate(loop, verts)

        assert result.shape == (1, 3)
        assert set(result[0].tolist()) == {0, 1, 2}

    def test_quad_loop(self):
        """4-vertex loop -> 2 triangles."""
        verts = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]
        ], dtype=float)
        loop = np.array([0, 1, 2, 3], dtype=np.intp)

        result = ear_clip_triangulate(loop, verts)

        assert result.shape[0] == 2
        # All triangles use only loop vertices
        assert set(result.ravel().tolist()).issubset({0, 1, 2, 3})

    def test_convex_hexagon(self):
        """6-vertex convex hexagon -> 4 triangles."""
        angles = np.linspace(0, 2 * np.pi, 7)[:-1]
        verts = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(6)])
        loop = np.arange(6, dtype=np.intp)

        result = ear_clip_triangulate(loop, verts)

        assert result.shape[0] == 4

    def test_empty_loop(self):
        """Empty loop -> no triangles."""
        verts = np.array([[0, 0, 0]], dtype=float)
        loop = np.array([], dtype=np.intp)

        result = ear_clip_triangulate(loop, verts)
        assert result.shape[0] == 0


class TestFillHoles:
    def test_fills_open_box(self, open_box):
        """After filling, boundary edges should disappear."""
        result = fill_holes(open_box)

        # More faces than before
        assert result.faces.shape[0] > open_box.faces.shape[0]

        # Check boundary edges gone
        edges, counts = edge_face_counts(result)
        boundary = edges[counts == 1]
        assert len(boundary) == 0

    def test_noop_on_watertight(self, unit_cube):
        """Watertight mesh -> no change."""
        result = fill_holes(unit_cube)

        assert result.faces.shape[0] == unit_cube.faces.shape[0]
        assert result.vertices.shape[0] == unit_cube.vertices.shape[0]

    def test_respects_max_hole_edges(self, open_box):
        """Large holes skipped if over max_hole_edges."""
        # Set max_hole_edges=2, which is smaller than any real loop
        result = fill_holes(open_box, max_hole_edges=2)

        # Should be unchanged since the hole was too big
        assert result.faces.shape[0] == open_box.faces.shape[0]

    def test_does_not_mutate_input(self, open_box):
        """Input mesh not modified."""
        orig_verts = open_box.vertices.copy()
        orig_faces = open_box.faces.copy()

        fill_holes(open_box)

        np.testing.assert_array_equal(open_box.vertices, orig_verts)
        np.testing.assert_array_equal(open_box.faces, orig_faces)

    def test_no_new_vertices(self, open_box):
        """Hole filling uses existing boundary vertices only."""
        result = fill_holes(open_box)

        assert result.vertices.shape[0] == open_box.vertices.shape[0]
