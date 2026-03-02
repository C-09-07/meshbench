"""Tests for meshview comparison export (2-up viewer)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh

import meshbench
from meshview.export import (
    _compute_component_labels,
    build_defects_json,
    export_comparison,
)


class TestComputeComponentLabels:
    def test_single_component(self, unit_cube: trimesh.Trimesh) -> None:
        """A single cube has all faces in component 0."""
        labels = _compute_component_labels(unit_cube)
        assert len(labels) == unit_cube.faces.shape[0]
        assert np.unique(labels).tolist() == [0]

    def test_two_components(self, two_component_mesh: trimesh.Trimesh) -> None:
        """Two separated cubes produce 2 component IDs."""
        labels = _compute_component_labels(two_component_mesh)
        assert len(labels) == two_component_mesh.faces.shape[0]
        assert len(np.unique(labels)) == 2

    def test_matches_topology_count(self, unit_cube: trimesh.Trimesh) -> None:
        """Component count matches meshbench topology report."""
        report = meshbench.audit(unit_cube)
        labels = _compute_component_labels(unit_cube)
        n_comp = int(labels.max() + 1) if len(labels) > 0 else 0
        assert n_comp == report.topology.component_count

    def test_empty_mesh(self) -> None:
        """Empty mesh returns empty array."""
        mesh = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=int))
        labels = _compute_component_labels(mesh)
        assert len(labels) == 0


class TestBuildDefectsJsonComponentId:
    def test_component_id_in_metrics(self, unit_cube: trimesh.Trimesh) -> None:
        """component_id added to metrics when labels provided."""
        report = meshbench.audit(unit_cube, include_defects=True)
        labels = _compute_component_labels(unit_cube)
        data = build_defects_json(report, component_labels=labels)
        assert "component_id" in data["metrics"]
        assert len(data["metrics"]["component_id"]["values"]) == report.face_count
        assert data["metrics"]["component_id"]["stats"]["count"] == 1

    def test_no_component_id_without_labels(self, unit_cube: trimesh.Trimesh) -> None:
        """component_id absent when labels not provided (backwards compat)."""
        report = meshbench.audit(unit_cube, include_defects=True)
        data = build_defects_json(report)
        assert "component_id" not in data["metrics"]

    def test_two_component_count(self, two_component_mesh: trimesh.Trimesh) -> None:
        """component_id count matches for multi-component mesh."""
        report = meshbench.audit(two_component_mesh, include_defects=True)
        labels = _compute_component_labels(two_component_mesh)
        data = build_defects_json(report, component_labels=labels)
        assert data["metrics"]["component_id"]["stats"]["count"] == 2


class TestExportComparison:
    def test_creates_directory_structure(
        self, unit_cube: trimesh.Trimesh, tmp_path: Path,
    ) -> None:
        """Creates before/, after/, and compare.html."""
        report = meshbench.audit(unit_cube, include_defects=True)
        out = export_comparison(unit_cube, report, unit_cube, report, tmp_path / "cmp")
        assert (out / "before").is_dir()
        assert (out / "after").is_dir()
        assert (out / "compare.html").exists()

    def test_both_meshes_exported(
        self, unit_cube: trimesh.Trimesh, tmp_path: Path,
    ) -> None:
        """mesh.glb exists in each subdir."""
        report = meshbench.audit(unit_cube, include_defects=True)
        out = export_comparison(unit_cube, report, unit_cube, report, tmp_path / "cmp")
        assert (out / "before" / "mesh.glb").exists()
        assert (out / "after" / "mesh.glb").exists()

    def test_defects_json_has_component_id(
        self, unit_cube: trimesh.Trimesh, tmp_path: Path,
    ) -> None:
        """defects.json in both subdirs has component_id."""
        report = meshbench.audit(unit_cube, include_defects=True)
        out = export_comparison(unit_cube, report, unit_cube, report, tmp_path / "cmp")
        for sub in ("before", "after"):
            with open(out / sub / "defects.json") as f:
                data = json.load(f)
            assert "component_id" in data["metrics"]

    def test_score_included(
        self, unit_cube: trimesh.Trimesh, tmp_path: Path,
    ) -> None:
        """Score section present in both defects.json files."""
        report = meshbench.audit(unit_cube, include_defects=True)
        out = export_comparison(unit_cube, report, unit_cube, report, tmp_path / "cmp")
        for sub in ("before", "after"):
            with open(out / sub / "defects.json") as f:
                data = json.load(f)
            assert "score" in data
            assert "grade" in data["score"]

    def test_component_label_count(
        self, two_component_mesh: trimesh.Trimesh, tmp_path: Path,
    ) -> None:
        """Component count in defects.json matches topology."""
        report = meshbench.audit(two_component_mesh, include_defects=True)
        out = export_comparison(
            two_component_mesh, report,
            two_component_mesh, report,
            tmp_path / "cmp",
        )
        with open(out / "before" / "defects.json") as f:
            data = json.load(f)
        assert data["metrics"]["component_id"]["stats"]["count"] == report.topology.component_count

    def test_html_contains_compare_elements(
        self, unit_cube: trimesh.Trimesh, tmp_path: Path,
    ) -> None:
        """compare.html has the expected viewer elements."""
        report = meshbench.audit(unit_cube, include_defects=True)
        out = export_comparison(unit_cube, report, unit_cube, report, tmp_path / "cmp")
        html = (out / "compare.html").read_text()
        assert "canvas-before" in html
        assert "canvas-after" in html
        assert "OrbitControls" in html
        assert "mode-select" in html

    def test_different_meshes(
        self, unit_cube: trimesh.Trimesh, icosphere: trimesh.Trimesh, tmp_path: Path,
    ) -> None:
        """Works with different before/after meshes."""
        report_a = meshbench.audit(unit_cube, include_defects=True)
        report_b = meshbench.audit(icosphere, include_defects=True)
        out = export_comparison(unit_cube, report_a, icosphere, report_b, tmp_path / "cmp")
        with open(out / "before" / "defects.json") as f:
            before = json.load(f)
        with open(out / "after" / "defects.json") as f:
            after = json.load(f)
        assert before["face_count"] != after["face_count"]
