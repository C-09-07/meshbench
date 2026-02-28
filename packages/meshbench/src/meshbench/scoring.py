"""Application-specific scoring: profiles, checks, and grade cards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from meshbench.types import Check, MeshReport, ScoreCard


# ---------------------------------------------------------------------------
# Profile definitions
# ---------------------------------------------------------------------------
# Each check: (metric_path, operator, warn_threshold, fail_threshold, message_template)
# Operators:
#   "eq"  — pass if value == warn (bool checks, warn is the expected value)
#   "lt"  — pass if value < warn, fail if value >= fail
#   "gt"  — pass if value > warn, fail if value <= fail
#   "le"  — pass if value <= warn, fail if value > fail

@dataclass(frozen=True)
class _ThresholdDef:
    metric: str
    op: str
    warn: float | None  # threshold between pass and warn
    fail: float | None  # threshold between warn and fail
    template: str  # message format (may use {value})


_PROFILES: dict[str, list[_ThresholdDef]] = {
    "print": [
        _ThresholdDef("manifold.is_watertight", "eq", True, None, "watertight: {value}"),
        _ThresholdDef("manifold.non_manifold_edge_count", "le", 0, 0, "{value} non-manifold edges"),
        _ThresholdDef("manifold.non_manifold_vertex_count", "le", 0, 0, "{value} non-manifold vertices"),
        _ThresholdDef("manifold.self_intersection_count", "le", 0, None, "{value} self-intersections"),
        _ThresholdDef("manifold.self_intersection_ratio", "lt", 0.001, 0.01, "self-intersection ratio {value:.4f}"),
        _ThresholdDef("topology.degenerate_face_ratio", "lt", 0.001, 0.05, "degenerate face ratio {value:.4f}"),
        _ThresholdDef("manifold.boundary_loop_count", "le", 0, 0, "{value} boundary loops"),
        _ThresholdDef("topology.component_count", "le", 1, 3, "{value} components"),
    ],
    "animation": [
        _ThresholdDef("topology.valence_outlier_count", "le", 0, None, "{value} valence outliers"),
        _ThresholdDef("topology.quad_compatible_ratio", "gt", 0.5, 0.5, "quad-compatible ratio {value:.3f}"),
        _ThresholdDef("topology.degenerate_face_ratio", "lt", 0.001, 0.01, "degenerate face ratio {value:.4f}"),
        _ThresholdDef("topology.floating_vertex_count", "le", 0, 0, "{value} floating vertices"),
        _ThresholdDef("topology.component_count", "le", 1, 2, "{value} components"),
        _ThresholdDef("normals.winding_consistency", "gt", 0.95, 0.95, "winding consistency {value:.3f}"),
    ],
    "simulation": [
        _ThresholdDef("cell.aspect_ratio_max", "lt", 5.0, 10.0, "max aspect ratio {value:.2f}"),
        _ThresholdDef("cell.skewness_max", "lt", 0.5, 0.8, "max skewness {value:.3f}"),
        _ThresholdDef("cell.min_angle_min", "gt", 20.0, 10.0, "min angle {value:.1f}°"),
        _ThresholdDef("cell.max_angle_max", "lt", 150.0, 170.0, "max angle {value:.1f}°"),
        _ThresholdDef("manifold.is_watertight", "eq", True, None, "watertight: {value}"),
        _ThresholdDef("topology.degenerate_face_ratio", "lt", 0.0, 0.001, "degenerate face ratio {value:.4f}"),
    ],
    "ai_output": [
        _ThresholdDef("manifold.is_watertight", "eq", True, None, "watertight: {value}"),
        _ThresholdDef("manifold.non_manifold_vertex_ratio", "lt", 0.001, 0.05, "non-manifold vertex ratio {value:.4f}"),
        _ThresholdDef("manifold.self_intersection_ratio", "lt", 0.001, 0.05, "self-intersection ratio {value:.4f}"),
        _ThresholdDef("topology.degenerate_face_ratio", "lt", 0.01, 0.1, "degenerate face ratio {value:.4f}"),
        _ThresholdDef("topology.floating_vertex_ratio", "lt", 0.001, 0.01, "floating vertex ratio {value:.4f}"),
        _ThresholdDef("topology.component_count", "le", 1, 5, "{value} components"),
        _ThresholdDef("topology.genus", "le", 0, 10, "genus {value}"),
        _ThresholdDef("normals.winding_consistency", "gt", 0.95, 0.95, "winding consistency {value:.3f}"),
    ],
}


def _extract_metric(report: MeshReport, path: str) -> Any:
    """Navigate dotted path to extract a metric value from a MeshReport."""
    obj: Any = report
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _evaluate_check(defn: _ThresholdDef, value: Any) -> Check:
    """Evaluate a single threshold definition against a value."""
    try:
        msg = defn.template.format(value=value)
    except (ValueError, TypeError):
        msg = defn.template.replace("{value}", str(value))

    op = defn.op
    warn = defn.warn
    fail = defn.fail

    if op == "eq":
        # Boolean equality check — pass or fail, no warn
        status = "pass" if value == warn else "fail"
        return Check(
            metric=defn.metric,
            status=status,
            value=value,
            threshold_warn=None,
            threshold_fail=None,
            message=msg,
        )

    if op == "lt":
        if fail is not None and value >= fail:
            status = "fail"
        elif warn is not None and value >= warn:
            status = "warn"
        else:
            status = "pass"
    elif op == "gt":
        if fail is not None and value <= fail:
            status = "fail"
        elif warn is not None and value <= warn:
            status = "warn"
        else:
            status = "pass"
    elif op == "le":
        if fail is not None and value > fail:
            status = "fail"
        elif warn is not None and value > warn:
            status = "warn"
        else:
            status = "pass"
    else:
        status = "pass"

    return Check(
        metric=defn.metric,
        status=status,
        value=value,
        threshold_warn=float(warn) if warn is not None else None,
        threshold_fail=float(fail) if fail is not None else None,
        message=msg,
    )


def _score_to_grade(score: float) -> str:
    """Convert numeric score (0-100) to letter grade."""
    if score >= 90:
        return "A"
    if score >= 75:
        return "B"
    if score >= 60:
        return "C"
    if score >= 40:
        return "D"
    return "F"


def compute_score(report: MeshReport, profile: str) -> ScoreCard:
    """Score a MeshReport against a named profile.

    Args:
        report: Full mesh analysis report from audit().
        profile: One of "print", "animation", "simulation", "ai_output".

    Returns:
        ScoreCard with grade, score, and individual check results.

    Raises:
        ValueError: If profile name is not recognized.
    """
    if profile not in _PROFILES:
        raise ValueError(
            f"Unknown profile {profile!r}. "
            f"Available: {', '.join(sorted(_PROFILES))}"
        )

    checks: list[Check] = []
    for defn in _PROFILES[profile]:
        value = _extract_metric(report, defn.metric)
        checks.append(_evaluate_check(defn, value))

    pass_count = sum(1 for c in checks if c.status == "pass")
    warn_count = sum(1 for c in checks if c.status == "warn")
    fail_count = sum(1 for c in checks if c.status == "fail")

    # Score: pass=100, warn=50, fail=0, weighted equally
    total = len(checks)
    if total == 0:
        score = 100.0
    else:
        score = (pass_count * 100.0 + warn_count * 50.0) / total

    return ScoreCard(
        profile=profile,
        grade=_score_to_grade(score),
        score=round(score, 1),
        pass_count=pass_count,
        warn_count=warn_count,
        fail_count=fail_count,
        checks=checks,
    )


def available_profiles() -> list[str]:
    """Return list of available profile names."""
    return sorted(_PROFILES)
