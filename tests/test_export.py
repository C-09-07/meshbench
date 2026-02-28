"""Tests for meshbench.export (JSON sidecar + viewer file generation)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh

import meshbench
from meshbench.export import build_defects_json, export_viewer_data


class TestBuildDefectsJson:
    def test_requires_defects(self, unit_cube: trimesh.Trimesh) -> None:
        """Raises ValueError if report.defects is None."""
        report = meshbench.audit(unit_cube)
        try:
            build_defects_json(report)
            assert False, "Expected ValueError"
        except ValueError:
            pass

    def test_structure(self, unit_cube: trimesh.Trimesh) -> None:
        """JSON has required top-level keys."""
        report = meshbench.audit(unit_cube, include_defects=True)
        data = build_defects_json(report)
        assert "face_count" in data
        assert "vertex_count" in data
        assert "metrics" in data
        assert "defects" in data
        assert data["face_count"] == report.face_count

    def test_metrics_keys(self, unit_cube: trimesh.Trimesh) -> None:
        """Metrics section has all four metric types."""
        report = meshbench.audit(unit_cube, include_defects=True)
        data = build_defects_json(report)
        metrics = data["metrics"]
        for key in ("aspect_ratio", "min_angle", "max_angle", "skewness"):
            assert key in metrics
            assert "values" in metrics[key]
            assert "stats" in metrics[key]
            assert len(metrics[key]["values"]) == report.face_count

    def test_defects_keys(self, unit_cube: trimesh.Trimesh) -> None:
        """Defects section has all index fields."""
        report = meshbench.audit(unit_cube, include_defects=True)
        data = build_defects_json(report)
        defects = data["defects"]
        for key in (
            "degenerate_faces", "non_manifold_vertices",
            "non_manifold_edges", "boundary_edges",
            "self_intersection_pairs", "floating_vertices",
        ):
            assert key in defects

    def test_with_score(self, unit_cube: trimesh.Trimesh) -> None:
        """Score section included when ScoreCard provided."""
        report = meshbench.audit(unit_cube, include_defects=True)
        score = meshbench.score(report, "print")
        data = build_defects_json(report, score)
        assert "score" in data
        assert data["score"]["profile"] == "print"
        assert "grade" in data["score"]
        assert "checks" in data["score"]

    def test_json_serializable(self, unit_cube: trimesh.Trimesh) -> None:
        """Entire output is JSON-serializable."""
        report = meshbench.audit(unit_cube, include_defects=True)
        score = meshbench.score(report, "print")
        data = build_defects_json(report, score)
        text = json.dumps(data)
        assert isinstance(json.loads(text), dict)


class TestExportViewerData:
    def test_creates_three_files(
        self, unit_cube: trimesh.Trimesh, tmp_path: Path,
    ) -> None:
        """Exports mesh.glb, defects.json, and index.html."""
        report = meshbench.audit(unit_cube, include_defects=True)
        out = export_viewer_data(unit_cube, report, tmp_path / "viewer")
        assert (out / "mesh.glb").exists()
        assert (out / "defects.json").exists()
        assert (out / "index.html").exists()

    def test_json_valid(
        self, unit_cube: trimesh.Trimesh, tmp_path: Path,
    ) -> None:
        """defects.json is valid JSON with expected structure."""
        report = meshbench.audit(unit_cube, include_defects=True)
        out = export_viewer_data(unit_cube, report, tmp_path / "viewer")
        with open(out / "defects.json") as f:
            data = json.load(f)
        assert data["face_count"] == report.face_count
        assert "metrics" in data
        assert "score" in data

    def test_html_contains_threejs(
        self, unit_cube: trimesh.Trimesh, tmp_path: Path,
    ) -> None:
        """index.html contains three.js import."""
        report = meshbench.audit(unit_cube, include_defects=True)
        out = export_viewer_data(unit_cube, report, tmp_path / "viewer")
        html = (out / "index.html").read_text()
        assert "three" in html
        assert "OrbitControls" in html

    def test_stl_format(
        self, unit_cube: trimesh.Trimesh, tmp_path: Path,
    ) -> None:
        """Can export as STL instead of GLB."""
        report = meshbench.audit(unit_cube, include_defects=True)
        out = export_viewer_data(
            unit_cube, report, tmp_path / "viewer", mesh_format="stl",
        )
        assert (out / "mesh.stl").exists()
        assert not (out / "mesh.glb").exists()

    def test_custom_profile(
        self, unit_cube: trimesh.Trimesh, tmp_path: Path,
    ) -> None:
        """Can specify a different scoring profile."""
        report = meshbench.audit(unit_cube, include_defects=True)
        out = export_viewer_data(
            unit_cube, report, tmp_path / "viewer", profile="simulation",
        )
        with open(out / "defects.json") as f:
            data = json.load(f)
        assert data["score"]["profile"] == "simulation"

    def test_raises_without_defects(
        self, unit_cube: trimesh.Trimesh, tmp_path: Path,
    ) -> None:
        """Raises ValueError if report has no defects."""
        report = meshbench.audit(unit_cube)
        try:
            export_viewer_data(unit_cube, report, tmp_path / "viewer")
            assert False, "Expected ValueError"
        except ValueError:
            pass
