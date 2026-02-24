"""JSON serialization, NumpyEncoder, and summary table output."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return super().default(o)


def mesh_report_to_dict(report: Any) -> dict:
    """Convert a MeshReport dataclass to a JSON-safe dict."""
    return asdict(report)


def build_full_report(
    sources: dict[str, dict],
    *,
    reference_name: str | None = None,
) -> dict:
    """Build the top-level benchmark report dict.

    sources: {source_name: {"meshes": {name: report_dict}, "aggregate": stats_dict, ...}}
    """
    total = sum(s.get("mesh_count", 0) for s in sources.values())
    report: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "meshbench_version": "0.1.0",
        "mesh_count": total,
        "sources": sources,
    }
    if reference_name:
        report["reference_source"] = reference_name
    return report


def write_json(report: dict, output_dir: Path) -> Path:
    """Write report dict to timestamped JSON file. Returns path written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"benchmark_{ts}.json"
    with open(path, "w") as f:
        json.dump(report, f, cls=NumpyEncoder, indent=2)
    return path


# -- Summary table --

# (label, format_spec, width)
_TABLE_COLUMNS = [
    ("Source", 16),
    ("N", 4),
    ("WT%", 5),
    ("Wind", 6),
    ("BfDiv", 6),
    ("Genus", 6),
    ("ValStd", 7),
    ("Degen", 6),
    ("Float", 6),
    ("FDV", 6),
    ("Errs", 5),
    ("Warn", 5),
]


def _format_row(name: str, n: int, agg: dict, fail: dict) -> str:
    # Count total errors/warnings across all failure codes
    errs = sum(e["count"] for e in fail.values() if e.get("severity_max") == "error")
    warns = sum(e["count"] for e in fail.values() if e.get("severity_max") == "warning")

    cells = [
        f"{name[:16]:<16s}",
        f"{n:>4d}",
        f"{agg.get('watertight_ratio', 0):>5.1%}",
        f"{agg.get('mean_winding_consistency', 0):>6.3f}",
        f"{agg.get('mean_backface_divergence', 0):>6.3f}",
        f"{agg.get('mean_genus', 0):>6.1f}",
        f"{agg.get('mean_valence_std', 0):>7.2f}",
        f"{agg.get('mean_degenerate_faces', 0):>6.0f}",
        f"{agg.get('mean_floating_vertices', 0):>6.0f}",
        f"{agg.get('mean_face_density_variance', 0):>6.3f}",
        f"{errs:>5d}",
        f"{warns:>5d}",
    ]
    return "  ".join(cells)


def format_table(
    sources: dict[str, dict],
    *,
    reference_name: str | None = None,
) -> str:
    """Format sources dict into a summary comparison table string."""
    # Build header
    parts = []
    for label, width in _TABLE_COLUMNS:
        if label == "Source":
            parts.append(f"{label:<{width}s}")
        else:
            parts.append(f"{label:>{width}s}")
    header = "  ".join(parts)

    lines = [header, "-" * len(header)]

    # Reference first, then alphabetical
    ref = reference_name or "reference"
    ordered = []
    if ref in sources:
        ordered.append(ref)
    ordered.extend(k for k in sorted(sources) if k != ref)

    for name in ordered:
        src = sources[name]
        agg = src.get("aggregate", {})
        fail = src.get("failure_summary", {})
        lines.append(_format_row(name, src.get("mesh_count", 0), agg, fail))

    return "\n".join(lines)
