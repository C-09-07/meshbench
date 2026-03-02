"""Export mesh + defects JSON + viewer HTML for interactive visualization."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import trimesh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as sparse_cc

from meshbench.types import Defects, MeshReport, ScoreCard

_VIEWER_DIR = Path(__file__).parent / "viewer"


def _sanitize_float(v: float) -> float | None:
    """Replace inf/nan with None for JSON safety."""
    if np.isfinite(v):
        return float(v)
    return None


def _sanitize_array(arr: np.ndarray) -> list:
    """Convert ndarray to JSON-safe list, replacing inf/nan with None."""
    result = arr.tolist()
    if arr.dtype.kind == 'f':  # float array — check for non-finite
        return [None if not np.isfinite(v) else v for v in result]
    return result


def _serialize_array(arr: np.ndarray | None) -> list | None:
    """Convert ndarray to JSON-safe list, or None."""
    if arr is None:
        return None
    return _sanitize_array(arr)


def _compute_component_labels(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return per-face component ID array (length F).

    Uses scipy.sparse connected_components on face adjacency (same pattern
    as meshbench/topology.py:component_sizes).
    """
    n_faces = mesh.faces.shape[0]
    if n_faces == 0:
        return np.array([], dtype=np.intp)

    adj_pairs = mesh.face_adjacency
    if len(adj_pairs) == 0:
        # Triangle soup — each face is its own component
        return np.arange(n_faces, dtype=np.intp)

    rows = np.concatenate([adj_pairs[:, 0], adj_pairs[:, 1]])
    cols = np.concatenate([adj_pairs[:, 1], adj_pairs[:, 0]])
    data = np.ones(len(rows), dtype=np.int8)
    graph = csr_matrix((data, (rows, cols)), shape=(n_faces, n_faces))
    _n_comp, labels = sparse_cc(graph, directed=False)
    return labels


def _build_metrics_section(defects: Defects) -> dict:
    """Build the metrics section of the JSON sidecar."""
    metrics = {}

    if defects.aspect_ratio is not None:
        ar = defects.aspect_ratio
        ar_finite = ar[np.isfinite(ar)]
        if len(ar_finite) == 0:
            ar_finite = np.array([0.0])
        metrics["aspect_ratio"] = {
            "values": _sanitize_array(ar),
            "stats": {
                "mean": float(np.mean(ar_finite)),
                "max": _sanitize_float(float(np.max(ar))),
            },
        }

    if defects.min_angle is not None:
        metrics["min_angle"] = {
            "values": _sanitize_array(defects.min_angle),
            "stats": {
                "mean": float(np.mean(defects.min_angle)),
                "min": float(np.min(defects.min_angle)),
            },
        }

    if defects.max_angle is not None:
        metrics["max_angle"] = {
            "values": _sanitize_array(defects.max_angle),
            "stats": {
                "mean": float(np.mean(defects.max_angle)),
                "max": float(np.max(defects.max_angle)),
            },
        }

    if defects.skewness is not None:
        metrics["skewness"] = {
            "values": _sanitize_array(defects.skewness),
            "stats": {
                "mean": float(np.mean(defects.skewness)),
                "max": float(np.max(defects.skewness)),
            },
        }

    return metrics


def _build_defects_section(defects: Defects) -> dict:
    """Build the defects section of the JSON sidecar."""
    return {
        "degenerate_faces": _serialize_array(defects.degenerate_faces),
        "non_manifold_vertices": _serialize_array(defects.non_manifold_vertices),
        "non_manifold_edges": _serialize_array(defects.non_manifold_edges),
        "boundary_edges": _serialize_array(defects.boundary_edges),
        "self_intersection_pairs": _serialize_array(defects.self_intersection_pairs),
        "floating_vertices": _serialize_array(defects.floating_vertices),
    }


def _build_score_section(score_card: ScoreCard) -> dict:
    """Build the score section of the JSON sidecar."""
    return {
        "profile": score_card.profile,
        "grade": score_card.grade,
        "score": score_card.score,
        "checks": [c.to_dict() for c in score_card.checks],
    }


def build_defects_json(
    report: MeshReport,
    score_card: ScoreCard | None = None,
    component_labels: np.ndarray | None = None,
) -> dict:
    """Build the complete defects.json structure from a report with defects.

    Args:
        report: MeshReport with defects populated (include_defects=True).
        score_card: Optional ScoreCard to include in output.
        component_labels: Optional per-face component IDs (length F).

    Returns:
        JSON-serializable dict.

    Raises:
        ValueError: If report.defects is None.
    """
    if report.defects is None:
        raise ValueError(
            "report.defects is None — call audit(mesh, include_defects=True)"
        )

    metrics = _build_metrics_section(report.defects)

    if component_labels is not None:
        n_comp = int(component_labels.max() + 1) if len(component_labels) > 0 else 0
        metrics["component_id"] = {
            "values": component_labels.tolist(),
            "stats": {"count": n_comp},
        }

    result: dict = {
        "face_count": report.face_count,
        "vertex_count": report.vertex_count,
        "metrics": metrics,
        "defects": _build_defects_section(report.defects),
    }

    if score_card is not None:
        result["score"] = _build_score_section(score_card)

    return result


def export_viewer_data(
    mesh: trimesh.Trimesh,
    report: MeshReport,
    output_dir: str | Path,
    mesh_format: str = "glb",
    profile: str = "print",
) -> Path:
    """Write mesh + defects JSON + viewer HTML to output_dir.

    Args:
        mesh: The mesh to export.
        report: MeshReport with defects populated (include_defects=True).
        output_dir: Directory to write files into (created if needed).
        mesh_format: Export format for mesh file ("glb" or "stl").
        profile: Scoring profile for the ScoreCard.

    Returns:
        Path to the output directory.

    Raises:
        ValueError: If report.defects is None.
    """
    from meshbench.scoring import compute_score

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Export mesh geometry
    mesh_file = out / f"mesh.{mesh_format}"
    mesh.export(str(mesh_file))

    # 2. Build and write defects JSON
    score_card = compute_score(report, profile)
    data = build_defects_json(report, score_card)

    json_file = out / "defects.json"
    with open(json_file, "w") as f:
        json.dump(data, f)

    # 3. Copy viewer HTML
    viewer_src = _VIEWER_DIR / "index.html"
    viewer_dst = out / "index.html"
    shutil.copy2(str(viewer_src), str(viewer_dst))

    return out


def export_comparison(
    mesh_before: trimesh.Trimesh,
    report_before: MeshReport,
    mesh_after: trimesh.Trimesh,
    report_after: MeshReport,
    output_dir: str | Path,
    mesh_format: str = "glb",
    profile: str = "print",
) -> Path:
    """Write before/after mesh comparison data + 2-up viewer HTML.

    Args:
        mesh_before: Original mesh.
        report_before: MeshReport for original (include_defects=True).
        mesh_after: Repaired mesh.
        report_after: MeshReport for repaired (include_defects=True).
        output_dir: Directory to write files into (created if needed).
        mesh_format: Export format for mesh files ("glb" or "stl").
        profile: Scoring profile for the ScoreCards.

    Returns:
        Path to the output directory.

    Raises:
        ValueError: If either report.defects is None.
    """
    from meshbench.scoring import compute_score

    out = Path(output_dir)

    for label, mesh, report in [
        ("before", mesh_before, report_before),
        ("after", mesh_after, report_after),
    ]:
        sub = out / label
        sub.mkdir(parents=True, exist_ok=True)

        # Export mesh geometry
        mesh.export(str(sub / f"mesh.{mesh_format}"))

        # Build defects JSON with component labels
        score_card = compute_score(report, profile)
        labels = _compute_component_labels(mesh)
        data = build_defects_json(report, score_card, component_labels=labels)

        with open(sub / "defects.json", "w") as f:
            json.dump(data, f)

    # Copy comparison viewer HTML
    viewer_src = _VIEWER_DIR / "compare.html"
    shutil.copy2(str(viewer_src), str(out / "compare.html"))

    return out
