"""Tests for meshbench._intersections (self-intersection detection)."""

from __future__ import annotations

import numpy as np
import trimesh

from meshbench._intersections import detect_self_intersections


class TestNonIntersecting:
    """Clean meshes should report zero self-intersections."""

    def test_cube_no_intersections(self, unit_cube: trimesh.Trimesh) -> None:
        count, pairs = detect_self_intersections(unit_cube)
        assert count == 0
        assert pairs.shape == (0, 2)

    def test_icosphere_no_intersections(self, icosphere: trimesh.Trimesh) -> None:
        count, pairs = detect_self_intersections(icosphere)
        assert count == 0
        assert pairs.shape == (0, 2)


class TestOverlappingTriangles:
    """Two crossing triangles that share no vertex should be detected."""

    def test_two_crossing_triangles(self) -> None:
        # Triangle A in XY plane
        # Triangle B crosses through A along the Z axis
        verts = np.array(
            [
                # Triangle A (XY plane, z=0)
                [-1.0, -1.0, 0.0],  # 0
                [1.0, -1.0, 0.0],  # 1
                [0.0, 1.0, 0.0],  # 2
                # Triangle B (XZ plane, y=0) — crosses through A
                [0.0, 0.0, -1.0],  # 3
                [0.0, 0.0, 1.0],  # 4
                [0.0, 2.0, 0.0],  # 5
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        count, pairs = detect_self_intersections(mesh)
        assert count == 1
        assert pairs.shape == (1, 2)
        pair = tuple(sorted(pairs[0]))
        assert pair == (0, 1)


class TestBowtie:
    """Two triangles sharing a single vertex but NOT intersecting in 3D."""

    def test_bowtie_no_intersection(self) -> None:
        verts = np.array(
            [
                [0.0, 0.0, 0.0],  # 0 — shared vertex
                [1.0, 0.0, 0.0],  # 1
                [0.0, 1.0, 0.0],  # 2
                [-1.0, 0.0, 0.0],  # 3
                [0.0, -1.0, 0.0],  # 4
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2], [0, 3, 4]], dtype=np.int64)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        count, pairs = detect_self_intersections(mesh)
        # Adjacent (share vertex 0) so filtered out, and also not intersecting
        assert count == 0


class TestSelfIntersectingMesh:
    """Mesh where one face is pushed through another."""

    def test_pushed_face(self) -> None:
        # Start with a flat 2x2 grid of 4 triangles, then push the center
        # vertex through the plane so that opposite triangles cross.
        #
        #  3---2---5
        #  | / | / |
        #  0---1---4
        #
        # Push vertex 1 far below the plane, causing face [0,1,2] and [1,4,5]
        # (which do NOT share a vertex with each other via 3 or via 2-5 path
        # depending on connectivity) to potentially intersect.
        #
        # Simpler: build two non-adjacent triangles that cross.
        verts = np.array(
            [
                [0.0, 0.0, 0.0],  # 0
                [2.0, 0.0, 0.0],  # 1
                [1.0, 2.0, 0.0],  # 2
                [1.0, 0.5, -1.0],  # 3
                [1.0, 0.5, 1.0],  # 4
                [1.0, 1.5, 0.0],  # 5 — this triangle pokes through triangle 012
                # Add a separator triangle (adjacent to neither) that does not
                # interfere.
                [5.0, 0.0, 0.0],  # 6
                [6.0, 0.0, 0.0],  # 7
                [5.5, 1.0, 0.0],  # 8
            ],
            dtype=np.float64,
        )
        faces = np.array(
            [
                [0, 1, 2],  # face 0 — in XY plane
                [3, 4, 5],  # face 1 — crosses face 0
                [6, 7, 8],  # face 2 — far away, no intersection
            ],
            dtype=np.int64,
        )
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        count, pairs = detect_self_intersections(mesh)
        assert count > 0
        # Face pair (0, 1) should be in the results
        pair_set = {tuple(sorted(p)) for p in pairs}
        assert (0, 1) in pair_set


class TestEdgeCases:
    """Empty and single-face meshes."""

    def test_empty_mesh(self) -> None:
        mesh = trimesh.Trimesh(
            vertices=np.empty((0, 3)),
            faces=np.empty((0, 3), dtype=np.int64),
            process=False,
        )
        count, pairs = detect_self_intersections(mesh)
        assert count == 0
        assert pairs.shape == (0, 2)

    def test_single_face(self) -> None:
        verts = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        count, pairs = detect_self_intersections(mesh)
        assert count == 0
        assert pairs.shape == (0, 2)


class TestManifoldReportIntegration:
    """Self-intersection fields in ManifoldReport via compute_manifold_report."""

    def test_cube_report_zero(self, unit_cube: trimesh.Trimesh) -> None:
        from meshbench.manifold import compute_manifold_report

        report = compute_manifold_report(unit_cube)
        assert report.self_intersection_count == 0
        assert report.self_intersection_ratio == 0.0

    def test_crossing_triangles_report(self) -> None:
        from meshbench.manifold import compute_manifold_report

        verts = np.array(
            [
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0],
                [0.0, 2.0, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        report = compute_manifold_report(mesh)
        assert report.self_intersection_count == 1
        assert report.self_intersection_ratio > 0.0
