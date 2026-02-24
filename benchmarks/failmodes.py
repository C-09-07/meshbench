"""Failure mode classification from mesh report dicts.

Classifies dominant structural failure modes per mesh, producing
actionable diagnostics rather than just scores. Taxonomy follows
geometric/topological categories that meshbench can measure.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class FailureMode:
    """A single detected failure mode."""

    category: str  # "manifold", "winding", "topology", "density"
    severity: str  # "info", "warning", "error"
    code: str  # machine-readable key, e.g. "non_manifold_edges"
    detail: str  # human-readable description with numbers


@dataclass
class FailureFingerprint:
    """Classified failure modes for a single mesh."""

    modes: list[FailureMode] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for m in self.modes if m.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for m in self.modes if m.severity == "warning")

    @property
    def dominant_category(self) -> str | None:
        """Most frequent failure category, or None if clean."""
        if not self.modes:
            return None
        cats = [m.category for m in self.modes]
        return max(set(cats), key=cats.count)

    def to_dict(self) -> dict:
        return {
            "modes": [asdict(m) for m in self.modes],
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "dominant_category": self.dominant_category,
        }


def classify_failures(report: dict) -> FailureFingerprint:
    """Classify failure modes from a mesh report dict."""
    modes: list[FailureMode] = []
    mf = report.get("manifold", {})
    nr = report.get("normals", {})
    tp = report.get("topology", {})
    dn = report.get("density", {})
    fp = report.get("fingerprint", {})

    # -- Manifold failures --

    nm_edges = mf.get("non_manifold_edge_count", 0)
    nm_verts = mf.get("non_manifold_vertex_count", 0)
    if nm_edges > 0 or nm_verts > 0:
        modes.append(FailureMode(
            category="manifold",
            severity="error",
            code="non_manifold_geometry",
            detail=f"non-manifold geometry ({nm_edges} edges, {nm_verts} vertices)",
        ))

    boundary = mf.get("boundary_edge_count", 0)
    loops = mf.get("boundary_loop_count", 0)
    if boundary > 0:
        modes.append(FailureMode(
            category="manifold",
            severity="error" if loops > 2 else "warning",
            code="open_boundary",
            detail=f"open boundaries ({boundary} edges, {loops} loops)",
        ))

    # -- Winding / normal failures --

    winding = nr.get("winding_consistency", 1.0)
    flipped = nr.get("flipped_count", 0)
    if winding < 0.80:
        modes.append(FailureMode(
            category="winding",
            severity="error",
            code="inconsistent_winding",
            detail=f"severely inconsistent winding ({winding:.1%} consistent, {flipped} flipped pairs)",
        ))
    elif winding < 0.95:
        modes.append(FailureMode(
            category="winding",
            severity="warning",
            code="inconsistent_winding",
            detail=f"inconsistent winding ({winding:.1%} consistent, {flipped} flipped pairs)",
        ))

    bf_div = nr.get("backface_divergence", 0.0)
    if bf_div > 0.5:
        modes.append(FailureMode(
            category="winding",
            severity="warning",
            code="high_backface_divergence",
            detail=f"high backface divergence ({bf_div:.3f})",
        ))

    # -- Topology failures --

    genus = tp.get("genus", 0)
    if genus > 5:
        modes.append(FailureMode(
            category="topology",
            severity="error",
            code="high_genus",
            detail=f"high genus ({genus}) — likely topological noise",
        ))
    elif genus > 0:
        modes.append(FailureMode(
            category="topology",
            severity="info",
            code="non_zero_genus",
            detail=f"genus {genus}",
        ))

    components = tp.get("component_count", 1)
    if components > 4:
        modes.append(FailureMode(
            category="topology",
            severity="error",
            code="many_disconnected_components",
            detail=f"{components} disconnected components",
        ))
    elif components > 1:
        modes.append(FailureMode(
            category="topology",
            severity="warning",
            code="disconnected_components",
            detail=f"{components} disconnected components",
        ))

    degen = tp.get("degenerate_face_count", 0)
    face_count = report.get("face_count", 1)
    if degen > 0:
        pct = degen / max(face_count, 1)
        sev = "error" if pct > 0.01 else "warning" if degen > 10 else "info"
        modes.append(FailureMode(
            category="topology",
            severity=sev,
            code="degenerate_faces",
            detail=f"{degen} degenerate faces ({pct:.2%} of {face_count})",
        ))

    floats = tp.get("floating_vertex_count", 0)
    if floats > 0:
        modes.append(FailureMode(
            category="topology",
            severity="warning" if floats > 10 else "info",
            code="floating_vertices",
            detail=f"{floats} floating vertices",
        ))

    valence_std = tp.get("valence_std", 0.0)
    if valence_std > 3.0:
        modes.append(FailureMode(
            category="topology",
            severity="warning",
            code="irregular_valence",
            detail=f"irregular valence distribution (std={valence_std:.2f})",
        ))

    # -- Density failures --

    fdv = dn.get("face_density_variance", 0.0)
    if fdv > 1.0:
        modes.append(FailureMode(
            category="density",
            severity="error",
            code="extreme_density_variance",
            detail=f"extreme face density variance ({fdv:.3f})",
        ))
    elif fdv > 0.3:
        modes.append(FailureMode(
            category="density",
            severity="warning",
            code="high_density_variance",
            detail=f"high face density variance ({fdv:.3f})",
        ))

    return FailureFingerprint(modes=modes)


def source_failure_summary(reports: dict[str, dict]) -> dict:
    """Compute per-source failure mode frequency table.

    Returns {code: {"count": N, "ratio": float, "severity_max": str, "example": str}}.
    """
    n = len(reports)
    if n == 0:
        return {}

    counts: dict[str, dict] = {}
    for name, report in reports.items():
        fp = classify_failures(report)
        seen_codes: set[str] = set()
        for mode in fp.modes:
            if mode.code in seen_codes:
                continue
            seen_codes.add(mode.code)
            if mode.code not in counts:
                counts[mode.code] = {
                    "count": 0,
                    "category": mode.category,
                    "severity_max": mode.severity,
                    "example": "",
                }
            entry = counts[mode.code]
            entry["count"] += 1
            # Escalate severity
            _SEV_RANK = {"info": 0, "warning": 1, "error": 2}
            if _SEV_RANK.get(mode.severity, 0) > _SEV_RANK.get(entry["severity_max"], 0):
                entry["severity_max"] = mode.severity
            if not entry["example"]:
                entry["example"] = mode.detail

    # Add ratio and sort by frequency
    for code, entry in counts.items():
        entry["ratio"] = round(entry["count"] / n, 3)

    return dict(sorted(counts.items(), key=lambda x: -x[1]["count"]))
