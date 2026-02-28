"""Tests for the fix pipeline and planner."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

import meshbench
from meshfix import fix, plan, FixConfig, FixResult, FixStepName


class TestPlan:
    def test_clean_mesh_only_normals(self, unit_cube):
        """Clean cube → only FIX_NORMALS planned."""
        report = meshbench.audit(unit_cube, include_defects=True)
        steps = plan(report)

        assert steps == [FixStepName.FIX_NORMALS]

    def test_open_box_has_winding(self, open_box):
        """Open box with flipped normals → FIX_WINDING + FIX_NORMALS."""
        report = meshbench.audit(open_box, include_defects=True)
        steps = plan(report)

        # Should at least have FIX_NORMALS
        assert FixStepName.FIX_NORMALS in steps

    def test_respects_disabled_config(self, unit_cube):
        """Disabled steps not planned."""
        report = meshbench.audit(unit_cube, include_defects=True)
        config = FixConfig(fix_normals=False)
        steps = plan(report, config)

        assert FixStepName.FIX_NORMALS not in steps

    def test_plans_degenerate_removal(self):
        """Mesh with degenerates → REMOVE_DEGENERATES planned."""
        verts = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0.5, 0, 0],  # collinear
        ], dtype=float)
        faces = np.array([[0, 1, 2], [0, 3, 1]])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        report = meshbench.audit(mesh, include_defects=True)
        steps = plan(report)

        assert FixStepName.REMOVE_DEGENERATES in steps

    def test_plans_floating_removal(self):
        """Mesh with floating vertex → REMOVE_FLOATING_VERTICES planned."""
        verts = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [5, 5, 5],
        ], dtype=float)
        faces = np.array([[0, 1, 2]])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        report = meshbench.audit(mesh, include_defects=True)
        steps = plan(report)

        assert FixStepName.REMOVE_FLOATING_VERTICES in steps

    def test_plans_shell_removal(self, two_component_mesh):
        """Multi-component mesh → REMOVE_SMALL_SHELLS planned."""
        report = meshbench.audit(two_component_mesh, include_defects=True)
        steps = plan(report)

        assert FixStepName.REMOVE_SMALL_SHELLS in steps


class TestFix:
    def test_end_to_end_degenerates(self):
        """Mesh with degenerates → fix() → re-audit shows 0 degenerates."""
        # Start with a real 3D box, then add a degenerate face
        box = trimesh.creation.box(extents=[1, 1, 1])
        # Add a degenerate (collinear) triangle
        degen_verts = np.array([[0, 0, 0], [0.5, 0, 0]], dtype=float)
        n = box.vertices.shape[0]
        degen_face = np.array([[n, n + 1, 0]])  # collinear with existing v0
        verts = np.vstack([box.vertices, degen_verts])
        faces = np.vstack([box.faces, degen_face])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        fixed, result = fix(mesh)

        assert result.after is not None
        assert result.after.topology.degenerate_face_count == 0
        assert fixed.faces.shape[0] == box.faces.shape[0]

    def test_end_to_end_floating(self):
        """fix() removes floating verts."""
        # Start with a real 3D box, then add a floating vertex
        box = trimesh.creation.box(extents=[1, 1, 1])
        floating_vert = np.array([[5, 5, 5]], dtype=float)
        verts = np.vstack([box.vertices, floating_vert])
        mesh = trimesh.Trimesh(vertices=verts, faces=box.faces.copy(), process=False)

        fixed, result = fix(mesh)

        assert result.after is not None
        assert result.after.topology.floating_vertex_count == 0
        assert fixed.vertices.shape[0] == box.vertices.shape[0]

    def test_clean_mesh_noop(self, unit_cube):
        """Clean mesh → no-op, before ≈ after."""
        fixed, result = fix(unit_cube)

        assert fixed.faces.shape[0] == unit_cube.faces.shape[0]
        assert fixed.vertices.shape[0] == unit_cube.vertices.shape[0]
        assert result.faces_before == result.faces_after
        assert result.vertices_before == result.vertices_after

    def test_auto_audits(self):
        """fix(mesh) without report auto-calls audit()."""
        mesh = trimesh.creation.box(extents=[1, 1, 1])

        fixed, result = fix(mesh)

        assert result.before is not None
        assert result.after is not None

    def test_result_has_before_after(self, unit_cube):
        """FixResult contains both reports."""
        report = meshbench.audit(unit_cube, include_defects=True)
        _, result = fix(unit_cube, report=report)

        assert result.before is report
        assert result.after is not None
        assert isinstance(result.after, meshbench.MeshReport)

    def test_steps_record_deltas(self):
        """FixStep shows faces_removed etc."""
        # Start with a real 3D box, add a degenerate face (3 collinear points)
        box = trimesh.creation.box(extents=[1, 1, 1])
        degen_verts = np.array([[10, 0, 0], [11, 0, 0], [10.5, 0, 0]], dtype=float)
        n = box.vertices.shape[0]
        degen_face = np.array([[n, n + 1, n + 2]])
        verts = np.vstack([box.vertices, degen_verts])
        faces = np.vstack([box.faces, degen_face])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        _, result = fix(mesh)

        degen_steps = [s for s in result.steps if s.name == FixStepName.REMOVE_DEGENERATES]
        assert len(degen_steps) == 1
        assert degen_steps[0].applied is True
        assert degen_steps[0].faces_removed == 1

    def test_does_not_mutate_input(self, unit_cube):
        """Input mesh not modified by fix()."""
        orig_verts = unit_cube.vertices.copy()
        orig_faces = unit_cube.faces.copy()

        fix(unit_cube)

        np.testing.assert_array_equal(unit_cube.vertices, orig_verts)
        np.testing.assert_array_equal(unit_cube.faces, orig_faces)

    def test_total_duration_positive(self, unit_cube):
        """Pipeline records non-zero duration."""
        _, result = fix(unit_cube)

        assert result.total_duration_ms > 0
