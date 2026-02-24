"""Tests for the benchmark runner modules."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import trimesh

from benchmarks.aggregate import AggregateStats, compute_aggregate, compute_delta
from benchmarks.corpus import audit_directory, discover_meshes, load_corpus, load_mesh, stem_name
from benchmarks.failmodes import FailureFingerprint, classify_failures, source_failure_summary
from benchmarks.report import NumpyEncoder, build_full_report, format_table, write_json


# -- Fixtures --


@pytest.fixture
def tmp_mesh_dir(tmp_path: Path) -> Path:
    """Create a temp dir with a couple of STL meshes."""
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    p1 = tmp_path / "cube.stl"
    mesh.export(str(p1))
    p2 = tmp_path / "cube2.stl"
    mesh.export(str(p2))
    return tmp_path


@pytest.fixture
def tmp_corpus(tmp_path: Path) -> Path:
    """Create a full corpus layout: reference/ + generated/source_a/."""
    ref = tmp_path / "reference"
    ref.mkdir()
    gen_a = tmp_path / "generated" / "source_a"
    gen_a.mkdir(parents=True)

    cube = trimesh.creation.box(extents=[1, 1, 1])
    sphere = trimesh.creation.icosphere(subdivisions=2)

    cube.export(str(ref / "cube.stl"))
    sphere.export(str(gen_a / "sphere.stl"))
    cube.export(str(gen_a / "cube_gen.stl"))
    return tmp_path


@pytest.fixture
def tmp_multitier(tmp_path: Path) -> Path:
    """Create a flat multi-tier layout: abc/, thingi10k/, meshy/."""
    cube = trimesh.creation.box(extents=[1, 1, 1])
    sphere = trimesh.creation.icosphere(subdivisions=2)

    for name in ("abc", "thingi10k", "meshy"):
        d = tmp_path / name
        d.mkdir()
        cube.export(str(d / f"{name}_cube.stl"))
    sphere.export(str(tmp_path / "meshy" / "meshy_sphere.stl"))
    return tmp_path


@pytest.fixture
def sample_reports() -> dict[str, dict]:
    """Two minimal report dicts with the fields aggregate needs."""
    return {
        "mesh_a": {
            "face_count": 100,
            "manifold": {
                "is_watertight": True,
                "boundary_edge_count": 0,
                "boundary_loop_count": 0,
                "non_manifold_edge_count": 0,
                "non_manifold_vertex_count": 0,
            },
            "normals": {
                "winding_consistency": 1.0,
                "flipped_count": 0,
                "backface_divergence": 0.1,
            },
            "topology": {
                "genus": 0,
                "valence_mean": 4.0,
                "valence_std": 0.5,
                "degenerate_face_count": 0,
                "floating_vertex_count": 0,
                "component_count": 1,
            },
            "density": {
                "vertices_per_area": 100.0,
                "face_density_variance": 0.01,
            },
            "fingerprint": {
                "hollowness": 0.8,
                "thinness": 1.2,
                "normal_entropy": 3.5,
            },
        },
        "mesh_b": {
            "face_count": 500,
            "manifold": {
                "is_watertight": False,
                "boundary_edge_count": 10,
                "boundary_loop_count": 2,
                "non_manifold_edge_count": 4,
                "non_manifold_vertex_count": 3,
            },
            "normals": {
                "winding_consistency": 0.9,
                "flipped_count": 20,
                "backface_divergence": 0.3,
            },
            "topology": {
                "genus": 2,
                "valence_mean": 5.0,
                "valence_std": 1.5,
                "degenerate_face_count": 10,
                "floating_vertex_count": 3,
                "component_count": 3,
            },
            "density": {
                "vertices_per_area": 200.0,
                "face_density_variance": 0.05,
            },
            "fingerprint": {
                "hollowness": 0.6,
                "thinness": 2.0,
                "normal_entropy": 4.0,
            },
        },
    }


# -- report.py tests --


class TestNumpyEncoder:
    def test_ndarray(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = json.dumps(arr, cls=NumpyEncoder)
        assert json.loads(result) == [1.0, 2.0, 3.0]

    def test_int64(self):
        val = np.int64(42)
        assert json.dumps(val, cls=NumpyEncoder) == "42"

    def test_float32(self):
        val = np.float32(3.14)
        result = json.loads(json.dumps(val, cls=NumpyEncoder))
        assert abs(result - 3.14) < 0.01

    def test_bool_(self):
        val = np.bool_(True)
        assert json.dumps(val, cls=NumpyEncoder) == "true"

    def test_nested(self):
        data = {"vec": np.array([1, 2]), "n": np.int64(5)}
        result = json.loads(json.dumps(data, cls=NumpyEncoder))
        assert result == {"vec": [1, 2], "n": 5}


class TestWriteJson:
    def test_writes_file(self, tmp_path: Path):
        report = {"timestamp": "2026-01-01", "mesh_count": 1, "sources": {}}
        path = write_json(report, tmp_path)
        assert path.exists()
        assert path.suffix == ".json"
        loaded = json.loads(path.read_text())
        assert loaded["mesh_count"] == 1

    def test_creates_output_dir(self, tmp_path: Path):
        out = tmp_path / "sub" / "dir"
        write_json({"a": 1}, out)
        assert out.is_dir()


class TestBuildFullReport:
    def test_structure(self):
        sources = {
            "reference": {"mesh_count": 2, "meshes": {}, "aggregate": {}},
        }
        report = build_full_report(sources)
        assert "timestamp" in report
        assert report["meshbench_version"] == "0.1.0"
        assert report["mesh_count"] == 2
        assert "reference" in report["sources"]

    def test_reference_name_recorded(self):
        sources = {"abc": {"mesh_count": 1, "meshes": {}, "aggregate": {}}}
        report = build_full_report(sources, reference_name="abc")
        assert report["reference_source"] == "abc"

    def test_no_reference_name(self):
        sources = {"abc": {"mesh_count": 1, "meshes": {}, "aggregate": {}}}
        report = build_full_report(sources)
        assert "reference_source" not in report


class TestFormatTable:
    def test_header_and_rows(self, sample_reports):
        agg = compute_aggregate(sample_reports)
        sources = {
            "reference": {"mesh_count": 2, "aggregate": asdict(agg), "failure_summary": {}},
        }
        table = format_table(sources)
        lines = table.strip().split("\n")
        assert len(lines) == 3  # header, separator, 1 row
        assert "reference" in lines[2]
        assert "Source" in lines[0]
        assert "Errs" in lines[0]

    def test_ordering_reference_first(self, sample_reports):
        agg = compute_aggregate(sample_reports)
        agg_dict = asdict(agg)
        sources = {
            "zebra": {"mesh_count": 1, "aggregate": agg_dict, "failure_summary": {}},
            "reference": {"mesh_count": 2, "aggregate": agg_dict, "failure_summary": {}},
            "alpha": {"mesh_count": 1, "aggregate": agg_dict, "failure_summary": {}},
        }
        table = format_table(sources)
        lines = table.strip().split("\n")
        data_lines = lines[2:]
        assert "reference" in data_lines[0]
        assert "alpha" in data_lines[1]
        assert "zebra" in data_lines[2]

    def test_custom_reference_name(self, sample_reports):
        agg_dict = asdict(compute_aggregate(sample_reports))
        sources = {
            "abc": {"mesh_count": 2, "aggregate": agg_dict, "failure_summary": {}},
            "meshy": {"mesh_count": 1, "aggregate": agg_dict, "failure_summary": {}},
        }
        table = format_table(sources, reference_name="abc")
        lines = table.strip().split("\n")
        data_lines = lines[2:]
        assert "abc" in data_lines[0]
        assert "meshy" in data_lines[1]


# -- corpus.py tests --


class TestDiscoverMeshes:
    def test_finds_stl(self, tmp_mesh_dir: Path):
        paths = discover_meshes(tmp_mesh_dir)
        assert len(paths) == 2
        assert all(p.suffix == ".stl" for p in paths)

    def test_empty_dir(self, tmp_path: Path):
        assert discover_meshes(tmp_path) == []


class TestLoadMesh:
    def test_loads_stl(self, tmp_mesh_dir: Path):
        stl = next(tmp_mesh_dir.glob("*.stl"))
        mesh = load_mesh(stl)
        assert isinstance(mesh, trimesh.Trimesh)
        assert mesh.vertices.shape[0] > 0

    def test_bad_file_returns_none(self, tmp_path: Path):
        bad = tmp_path / "bad.stl"
        bad.write_text("not a mesh")
        assert load_mesh(bad) is None


class TestStemName:
    def test_simple(self):
        assert stem_name(Path("/foo/bar/model.glb")) == "model"

    def test_dots(self):
        assert stem_name(Path("/foo/my.model.stl")) == "my.model"


class TestAuditDirectory:
    def test_audits_all(self, tmp_mesh_dir: Path):
        results = audit_directory(tmp_mesh_dir)
        assert len(results) == 2
        for name, report in results.items():
            assert "vertex_count" in report
            assert "fingerprint" in report
            assert "_audit_seconds" in report
            assert report["_audit_seconds"] > 0


class TestLoadCorpusMultiTier:
    def test_flat_multitier(self, tmp_multitier: Path):
        """Each subdir becomes a source."""
        sources = load_corpus(meshes_dir=tmp_multitier)
        assert set(sources.keys()) == {"abc", "thingi10k", "meshy"}
        assert len(sources["meshy"]) == 2  # cube + sphere
        assert len(sources["abc"]) == 1

    def test_legacy_layout(self, tmp_corpus: Path):
        """reference/ + generated/ still works."""
        sources = load_corpus(meshes_dir=tmp_corpus)
        assert "reference" in sources
        assert "source_a" in sources

    def test_single_flat_dir(self, tmp_mesh_dir: Path):
        """Dir with only files (no subdirs) becomes one source."""
        sources = load_corpus(meshes_dir=tmp_mesh_dir)
        assert len(sources) == 1


# -- failmodes.py tests --


class TestClassifyFailures:
    def test_clean_mesh(self, sample_reports):
        fp = classify_failures(sample_reports["mesh_a"])
        # Clean mesh: should have no errors or warnings
        assert fp.error_count == 0
        assert fp.warning_count == 0

    def test_non_manifold(self, sample_reports):
        fp = classify_failures(sample_reports["mesh_b"])
        codes = [m.code for m in fp.modes]
        assert "non_manifold_geometry" in codes
        assert "open_boundary" in codes

    def test_disconnected_components(self, sample_reports):
        fp = classify_failures(sample_reports["mesh_b"])
        codes = [m.code for m in fp.modes]
        assert "disconnected_components" in codes

    def test_degenerate_faces(self, sample_reports):
        fp = classify_failures(sample_reports["mesh_b"])
        codes = [m.code for m in fp.modes]
        assert "degenerate_faces" in codes

    def test_severe_winding(self):
        report = {
            "face_count": 100,
            "manifold": {"is_watertight": True, "boundary_edge_count": 0,
                         "boundary_loop_count": 0, "non_manifold_edge_count": 0,
                         "non_manifold_vertex_count": 0},
            "normals": {"winding_consistency": 0.7, "flipped_count": 50,
                        "backface_divergence": 0.1},
            "topology": {"genus": 0, "valence_mean": 4.0, "valence_std": 0.5,
                         "degenerate_face_count": 0, "floating_vertex_count": 0,
                         "component_count": 1},
            "density": {"vertices_per_area": 100.0, "face_density_variance": 0.01},
            "fingerprint": {"hollowness": 0.5, "thinness": 1.0, "normal_entropy": 3.0},
        }
        fp = classify_failures(report)
        winding_modes = [m for m in fp.modes if m.code == "inconsistent_winding"]
        assert len(winding_modes) == 1
        assert winding_modes[0].severity == "error"

    def test_extreme_density_variance(self):
        report = {
            "face_count": 100,
            "manifold": {"is_watertight": True, "boundary_edge_count": 0,
                         "boundary_loop_count": 0, "non_manifold_edge_count": 0,
                         "non_manifold_vertex_count": 0},
            "normals": {"winding_consistency": 1.0, "flipped_count": 0,
                        "backface_divergence": 0.1},
            "topology": {"genus": 0, "valence_mean": 4.0, "valence_std": 0.5,
                         "degenerate_face_count": 0, "floating_vertex_count": 0,
                         "component_count": 1},
            "density": {"vertices_per_area": 100.0, "face_density_variance": 2.5},
            "fingerprint": {"hollowness": 0.5, "thinness": 1.0, "normal_entropy": 3.0},
        }
        fp = classify_failures(report)
        codes = [m.code for m in fp.modes]
        assert "extreme_density_variance" in codes

    def test_dominant_category(self, sample_reports):
        fp = classify_failures(sample_reports["mesh_b"])
        # mesh_b has multiple manifold + topology issues
        assert fp.dominant_category in ("manifold", "topology")

    def test_to_dict_roundtrip(self, sample_reports):
        fp = classify_failures(sample_reports["mesh_b"])
        d = fp.to_dict()
        assert isinstance(d["modes"], list)
        assert isinstance(d["error_count"], int)
        assert isinstance(d["dominant_category"], str)


class TestSourceFailureSummary:
    def test_frequency_table(self, sample_reports):
        summary = source_failure_summary(sample_reports)
        # mesh_a is clean, mesh_b has issues — so each issue appears in 1/2 meshes
        assert "non_manifold_geometry" in summary
        assert summary["non_manifold_geometry"]["count"] == 1
        assert summary["non_manifold_geometry"]["ratio"] == 0.5

    def test_empty(self):
        assert source_failure_summary({}) == {}


# -- aggregate.py tests --


class TestComputeAggregate:
    def test_basic_stats(self, sample_reports):
        agg = compute_aggregate(sample_reports)
        assert agg.mesh_count == 2
        assert agg.watertight_ratio == 0.5  # 1 of 2
        assert agg.mean_winding_consistency == pytest.approx(0.95)
        assert agg.mean_genus == pytest.approx(1.0)
        assert agg.mean_degenerate_faces == pytest.approx(5.0)

    def test_empty(self):
        agg = compute_aggregate({})
        assert agg.mesh_count == 0
        assert agg.watertight_ratio == 0.0


class TestComputeDelta:
    def test_delta_values(self, sample_reports):
        ref_agg = compute_aggregate(sample_reports)
        gen = AggregateStats(
            mesh_count=3,
            watertight_ratio=0.3,
            mean_boundary_loops=5.0,
            mean_non_manifold_edges=10.0,
            mean_winding_consistency=0.8,
            std_winding_consistency=0.2,
            mean_backface_divergence=0.5,
            mean_genus=3.0,
            mean_valence_mean=5.5,
            mean_valence_std=2.0,
            mean_degenerate_faces=20.0,
            mean_floating_vertices=8.0,
            mean_vertices_per_area=300.0,
            mean_face_density_variance=0.1,
            mean_hollowness=0.5,
            mean_thinness=3.0,
            mean_normal_entropy=4.5,
        )
        delta = compute_delta(gen, ref_agg)
        assert "mesh_count" not in delta
        assert delta["watertight_ratio"] == pytest.approx(-0.2)
        assert isinstance(delta["mean_genus"], float)


# -- Integration: JSON roundtrip --


class TestJsonRoundtrip:
    def test_roundtrip(self, tmp_mesh_dir: Path, tmp_path: Path):
        results = audit_directory(tmp_mesh_dir)
        agg = compute_aggregate(results)
        fail_summary = source_failure_summary(results)
        # Inject per-mesh fingerprints
        for name, report in results.items():
            report["_failure_fingerprint"] = classify_failures(report).to_dict()
        sources = {
            "test": {
                "mesh_count": agg.mesh_count,
                "meshes": results,
                "aggregate": asdict(agg),
                "failure_summary": fail_summary,
            },
        }
        report = build_full_report(sources)
        path = write_json(report, tmp_path)

        loaded = json.loads(path.read_text())
        assert loaded["mesh_count"] == 2
        assert loaded["sources"]["test"]["mesh_count"] == 2
        assert isinstance(loaded["sources"]["test"]["aggregate"]["mean_hollowness"], float)
        # Failure data roundtrips
        assert "failure_summary" in loaded["sources"]["test"]
        first_mesh = list(loaded["sources"]["test"]["meshes"].values())[0]
        assert "_failure_fingerprint" in first_mesh
        assert isinstance(first_mesh["_failure_fingerprint"]["error_count"], int)
