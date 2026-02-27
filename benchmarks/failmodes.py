"""Mesh observation classifier: defects vs characteristics.

Splits mesh report observations into two categories:
- **defects**: Structural breakage (non-manifold geometry, degenerate faces,
  inconsistent winding). These are objectively broken regardless of context.
- **characteristics**: Measurable properties that are context-dependent
  (open boundaries, density variance, genus, component count). Whether these
  are "bad" depends on the use case (3D printing vs display vs game asset).

This avoids the trap of baking quality judgements into what should be
a measurement tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Observation:
    """A single mesh observation."""

    kind: str  # "defect" or "characteristic"
    category: str  # "manifold", "winding", "topology", "density"
    code: str  # machine-readable key
    detail: str  # human-readable description with numbers
    severity: str = "info"  # "info", "warning", "error" (defects only)


@dataclass
class MeshFingerprint:
    """Classified observations for a single mesh."""

    defects: list[Observation] = field(default_factory=list)
    characteristics: list[Observation] = field(default_factory=list)

    @property
    def defect_count(self) -> int:
        return len(self.defects)

    @property
    def has_defects(self) -> bool:
        return len(self.defects) > 0

    def to_dict(self) -> dict:
        return {
            "defects": [_obs_to_dict(o) for o in self.defects],
            "characteristics": [_obs_to_dict(o) for o in self.characteristics],
            "defect_count": self.defect_count,
        }


def _obs_to_dict(o: Observation) -> dict:
    d: dict = {"kind": o.kind, "category": o.category, "code": o.code, "detail": o.detail}
    if o.kind == "defect":
        d["severity"] = o.severity
    return d


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify(report: dict) -> MeshFingerprint:
    """Classify observations from a mesh report dict."""
    defects: list[Observation] = []
    chars: list[Observation] = []

    mf = report.get("manifold", {})
    nr = report.get("normals", {})
    tp = report.get("topology", {})
    dn = report.get("density", {})
    face_count = report.get("face_count", 1)

    # ── DEFECTS: objectively broken geometry ──

    # Non-manifold geometry
    nm_edges = mf.get("non_manifold_edge_count", 0)
    nm_verts = mf.get("non_manifold_vertex_count", 0)
    if nm_edges > 0 or nm_verts > 0:
        defects.append(Observation(
            kind="defect", category="manifold",
            code="non_manifold_geometry",
            detail=f"non-manifold geometry ({nm_edges} edges, {nm_verts} vertices)",
            severity="error",
        ))

    # Degenerate faces (zero-area triangles)
    degen = tp.get("degenerate_face_count", 0)
    if degen > 0:
        pct = degen / max(face_count, 1)
        sev = "error" if pct > 0.01 else "warning"
        defects.append(Observation(
            kind="defect", category="topology",
            code="degenerate_faces",
            detail=f"{degen} degenerate faces ({pct:.2%} of {face_count})",
            severity=sev,
        ))

    # Inconsistent winding
    winding = nr.get("winding_consistency", 1.0)
    flipped = nr.get("flipped_count", 0)
    if winding < 0.95:
        sev = "error" if winding < 0.80 else "warning"
        defects.append(Observation(
            kind="defect", category="winding",
            code="inconsistent_winding",
            detail=f"winding {winding:.1%} consistent ({flipped} flipped pairs)",
            severity=sev,
        ))

    # Floating vertices (referenced by nothing)
    floats = tp.get("floating_vertex_count", 0)
    if floats > 0:
        defects.append(Observation(
            kind="defect", category="topology",
            code="floating_vertices",
            detail=f"{floats} floating vertices",
            severity="warning" if floats > 10 else "info",
        ))

    # ── CHARACTERISTICS: context-dependent, not inherently bad ──

    # Open boundary (expected for CAD, display models; only matters for printing)
    boundary = mf.get("boundary_edge_count", 0)
    loops = mf.get("boundary_loop_count", 0)
    if boundary > 0:
        chars.append(Observation(
            kind="characteristic", category="manifold",
            code="open_boundary",
            detail=f"open boundary ({boundary} edges, {loops} loops)",
        ))
    else:
        chars.append(Observation(
            kind="characteristic", category="manifold",
            code="watertight",
            detail="watertight (no boundary edges)",
        ))

    # Genus
    genus = tp.get("genus", 0)
    if genus > 0:
        chars.append(Observation(
            kind="characteristic", category="topology",
            code="genus",
            detail=f"genus {genus}",
        ))

    # Component count
    components = tp.get("component_count", 1)
    if components > 1:
        chars.append(Observation(
            kind="characteristic", category="topology",
            code="multi_component",
            detail=f"{components} disconnected components",
        ))

    # Face density variance
    fdv = dn.get("face_density_variance", 0.0)
    chars.append(Observation(
        kind="characteristic", category="density",
        code="density_variance",
        detail=f"face density variance {fdv:.3f}",
    ))

    # Backface divergence
    bf_div = nr.get("backface_divergence", 0.0)
    chars.append(Observation(
        kind="characteristic", category="winding",
        code="backface_divergence",
        detail=f"backface divergence {bf_div:.3f}",
    ))

    # Valence regularity
    valence_std = tp.get("valence_std", 0.0)
    chars.append(Observation(
        kind="characteristic", category="topology",
        code="valence_std",
        detail=f"valence std {valence_std:.2f}",
    ))

    return MeshFingerprint(defects=defects, characteristics=chars)


# ---------------------------------------------------------------------------
# Source-level summaries
# ---------------------------------------------------------------------------

def source_defect_summary(reports: dict[str, dict]) -> dict:
    """Compute per-source defect frequency table.

    Returns {code: {"count": N, "ratio": float, "severity_max": str, "example": str}}.
    """
    n = len(reports)
    if n == 0:
        return {}

    counts: dict[str, dict] = {}
    _SEV_RANK = {"info": 0, "warning": 1, "error": 2}

    for name, report in reports.items():
        fp = classify(report)
        seen: set[str] = set()
        for obs in fp.defects:
            if obs.code in seen:
                continue
            seen.add(obs.code)
            if obs.code not in counts:
                counts[obs.code] = {
                    "count": 0,
                    "category": obs.category,
                    "severity_max": obs.severity,
                    "example": "",
                }
            entry = counts[obs.code]
            entry["count"] += 1
            if _SEV_RANK.get(obs.severity, 0) > _SEV_RANK.get(entry["severity_max"], 0):
                entry["severity_max"] = obs.severity
            if not entry["example"]:
                entry["example"] = obs.detail

    for entry in counts.values():
        entry["ratio"] = round(entry["count"] / n, 3)

    return dict(sorted(counts.items(), key=lambda x: -x[1]["count"]))


def source_characteristic_summary(reports: dict[str, dict]) -> dict:
    """Compute per-source characteristic frequency table.

    Returns {code: {"count": N, "ratio": float, "example": str}}.
    """
    n = len(reports)
    if n == 0:
        return {}

    counts: dict[str, dict] = {}
    for name, report in reports.items():
        fp = classify(report)
        seen: set[str] = set()
        for obs in fp.characteristics:
            if obs.code in seen:
                continue
            seen.add(obs.code)
            if obs.code not in counts:
                counts[obs.code] = {
                    "count": 0,
                    "category": obs.category,
                    "example": "",
                }
            entry = counts[obs.code]
            entry["count"] += 1
            if not entry["example"]:
                entry["example"] = obs.detail

    for entry in counts.values():
        entry["ratio"] = round(entry["count"] / n, 3)

    return dict(sorted(counts.items(), key=lambda x: -x[1]["count"]))


# ---------------------------------------------------------------------------
# Backward compat shims (used by run.py / report.py)
# ---------------------------------------------------------------------------

# Old names that external code might reference
FailureMode = Observation
FailureFingerprint = MeshFingerprint


def classify_failures(report: dict) -> MeshFingerprint:
    """Backward-compatible alias for classify()."""
    return classify(report)


def source_failure_summary(reports: dict[str, dict]) -> dict:
    """Backward-compatible alias — returns defect summary only."""
    return source_defect_summary(reports)
