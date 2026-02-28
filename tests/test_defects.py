"""Tests for the Defects dataclass and include_defects flag."""

from __future__ import annotations

import json

import numpy as np
import trimesh

import meshbench
from meshbench.types import Defects


class TestDefectsOptIn:
    def test_default_off(self, unit_cube: trimesh.Trimesh) -> None:
        """Defects is None by default."""
        report = meshbench.audit(unit_cube)
        assert report.defects is None

    def test_opt_in(self, unit_cube: trimesh.Trimesh) -> None:
        """include_defects=True populates defects."""
        report = meshbench.audit(unit_cube, include_defects=True)
        assert report.defects is not None
        assert isinstance(report.defects, Defects)


class TestCellArrayShapes:
    def test_cube_arrays(self, unit_cube: trimesh.Trimesh) -> None:
        """Per-face arrays have shape (F,)."""
        report = meshbench.audit(unit_cube, include_defects=True)
        d = report.defects
        assert d.aspect_ratio.shape == (report.face_count,)
        assert d.min_angle.shape == (report.face_count,)
        assert d.max_angle.shape == (report.face_count,)
        assert d.skewness.shape == (report.face_count,)

    def test_icosphere_arrays(self, icosphere: trimesh.Trimesh) -> None:
        """Icosphere also gets correct shapes."""
        report = meshbench.audit(icosphere, include_defects=True)
        d = report.defects
        n = report.face_count
        assert d.aspect_ratio.shape == (n,)
        assert d.min_angle.shape == (n,)

    def test_cube_values_consistent_with_cell_report(
        self, unit_cube: trimesh.Trimesh,
    ) -> None:
        """Defect arrays should match cell report stats."""
        report = meshbench.audit(unit_cube, include_defects=True)
        ar = report.defects.aspect_ratio
        ar_finite = ar[np.isfinite(ar)]
        np.testing.assert_allclose(
            float(np.mean(ar_finite)),
            report.cell.aspect_ratio_mean,
            rtol=1e-6,
        )


class TestDefectIndices:
    def test_watertight_cube_no_boundary(self, unit_cube: trimesh.Trimesh) -> None:
        """Watertight cube has no boundary edges."""
        report = meshbench.audit(unit_cube, include_defects=True)
        assert report.defects.boundary_edges is None

    def test_watertight_cube_no_nm_edges(self, unit_cube: trimesh.Trimesh) -> None:
        """Watertight cube has no non-manifold edges."""
        report = meshbench.audit(unit_cube, include_defects=True)
        assert report.defects.non_manifold_edges is None

    def test_open_box_has_boundary(self, open_box: trimesh.Trimesh) -> None:
        """Open box should have boundary edges."""
        report = meshbench.audit(open_box, include_defects=True)
        assert report.defects.boundary_edges is not None
        assert report.defects.boundary_edges.ndim == 2
        assert report.defects.boundary_edges.shape[1] == 2

    def test_cube_no_degenerates(self, unit_cube: trimesh.Trimesh) -> None:
        """Cube should have no degenerate faces."""
        report = meshbench.audit(unit_cube, include_defects=True)
        assert report.defects.degenerate_faces is None

    def test_degenerate_face_detected(self) -> None:
        """Collinear vertices produce a degenerate face."""
        verts = np.array(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0]], dtype=float,
        )
        faces = np.array([[0, 1, 2], [0, 1, 3]])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        report = meshbench.audit(mesh, include_defects=True)
        assert report.defects.degenerate_faces is not None
        assert 0 in report.defects.degenerate_faces

    def test_floating_vertices_detected(self) -> None:
        """Unreferenced vertex shows up in floating_vertices."""
        verts = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [99, 99, 99]], dtype=float,
        )
        faces = np.array([[0, 1, 2]])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        report = meshbench.audit(mesh, include_defects=True)
        assert report.defects.floating_vertices is not None
        assert 3 in report.defects.floating_vertices


class TestDefectsSerialization:
    def test_to_dict_round_trip(self, unit_cube: trimesh.Trimesh) -> None:
        """Defects serializes to JSON-safe dict."""
        report = meshbench.audit(unit_cube, include_defects=True)
        d = report.to_dict()
        text = json.dumps(d)
        parsed = json.loads(text)
        assert "defects" in parsed
        defects = parsed["defects"]
        assert isinstance(defects["aspect_ratio"], list)
        assert len(defects["aspect_ratio"]) == report.face_count

    def test_none_defects_serializes(self, unit_cube: trimesh.Trimesh) -> None:
        """When defects is None, to_dict produces None."""
        report = meshbench.audit(unit_cube)
        d = report.to_dict()
        assert d["defects"] is None

    def test_open_box_boundary_serializes(self, open_box: trimesh.Trimesh) -> None:
        """Boundary edges serialize as list of lists."""
        report = meshbench.audit(open_box, include_defects=True)
        d = report.to_dict()
        be = d["defects"]["boundary_edges"]
        assert isinstance(be, list)
        assert isinstance(be[0], list)
        assert len(be[0]) == 2
