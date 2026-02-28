"""Tests for meshbench.scoring (application-specific quality grades)."""

from __future__ import annotations

import trimesh
import numpy as np

import meshbench
from meshbench.scoring import compute_score, available_profiles, _PROFILES


class TestAvailableProfiles:
    def test_four_profiles(self) -> None:
        profiles = available_profiles()
        assert set(profiles) == {"print", "animation", "simulation", "ai_output"}


class TestScoreCardStructure:
    def test_fields_present(self, unit_cube: trimesh.Trimesh) -> None:
        report = meshbench.audit(unit_cube)
        card = meshbench.score(report, "print")
        assert hasattr(card, "profile")
        assert hasattr(card, "grade")
        assert hasattr(card, "score")
        assert hasattr(card, "pass_count")
        assert hasattr(card, "warn_count")
        assert hasattr(card, "fail_count")
        assert hasattr(card, "checks")

    def test_check_count_matches_profile(self, unit_cube: trimesh.Trimesh) -> None:
        report = meshbench.audit(unit_cube)
        for profile_name, defs in _PROFILES.items():
            card = meshbench.score(report, profile_name)
            assert len(card.checks) == len(defs), f"{profile_name}: expected {len(defs)} checks"
            assert card.pass_count + card.warn_count + card.fail_count == len(defs)

    def test_serialization(self, unit_cube: trimesh.Trimesh) -> None:
        report = meshbench.audit(unit_cube)
        card = meshbench.score(report, "print")
        d = card.to_dict()
        assert isinstance(d, dict)
        assert d["profile"] == "print"
        assert isinstance(d["checks"], list)
        assert all(isinstance(c, dict) for c in d["checks"])
        # Verify JSON-safe (no numpy types)
        import json
        json.dumps(d)

    def test_unknown_profile_raises(self, unit_cube: trimesh.Trimesh) -> None:
        report = meshbench.audit(unit_cube)
        try:
            meshbench.score(report, "nonexistent")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "nonexistent" in str(e)


class TestPrintProfile:
    def test_clean_cube_passes(self, unit_cube: trimesh.Trimesh) -> None:
        """A watertight unit cube should score well on print profile."""
        report = meshbench.audit(unit_cube)
        card = meshbench.score(report, "print")
        assert card.grade in ("A", "B")
        assert card.score >= 75.0
        assert card.fail_count == 0

    def test_open_box_fails_watertight(self, open_box: trimesh.Trimesh) -> None:
        """An open box is not watertight → print profile should fail."""
        report = meshbench.audit(open_box)
        card = meshbench.score(report, "print")
        # Must have at least one fail (watertight check)
        assert card.fail_count >= 1
        watertight_checks = [c for c in card.checks if "watertight" in c.metric]
        assert len(watertight_checks) == 1
        assert watertight_checks[0].status == "fail"

    def test_open_box_penalized(self, open_box: trimesh.Trimesh) -> None:
        """Open box should score below A due to watertight + boundary failures."""
        report = meshbench.audit(open_box)
        card = meshbench.score(report, "print")
        assert card.grade != "A"
        assert card.fail_count >= 2


class TestSimulationProfile:
    def test_cube_simulation(self, unit_cube: trimesh.Trimesh) -> None:
        """Unit cube has right-angle triangles: max_angle=90, min_angle=45."""
        report = meshbench.audit(unit_cube)
        card = meshbench.score(report, "simulation")
        # Cube has right triangles — min_angle=45 (pass), max_angle=90 (pass)
        # but aspect ratio and skewness should be reasonable
        assert card.score > 0

    def test_equilateral_mesh_scores_well(self) -> None:
        """An icosphere should score well on simulation profile."""
        mesh = trimesh.creation.icosphere(subdivisions=2)
        report = meshbench.audit(mesh)
        card = meshbench.score(report, "simulation")
        assert card.grade in ("A", "B")


class TestAnimationProfile:
    def test_cube_animation(self, unit_cube: trimesh.Trimesh) -> None:
        report = meshbench.audit(unit_cube)
        card = meshbench.score(report, "animation")
        assert card.score >= 0
        assert len(card.checks) == len(_PROFILES["animation"])


class TestAIOutputProfile:
    def test_cube_ai_output(self, unit_cube: trimesh.Trimesh) -> None:
        report = meshbench.audit(unit_cube)
        card = meshbench.score(report, "ai_output")
        assert card.grade in ("A", "B")
        assert card.fail_count == 0


class TestGrading:
    def test_grade_boundaries(self, unit_cube: trimesh.Trimesh) -> None:
        """Verify grade assignment from score."""
        from meshbench.scoring import _score_to_grade
        assert _score_to_grade(100.0) == "A"
        assert _score_to_grade(90.0) == "A"
        assert _score_to_grade(89.9) == "B"
        assert _score_to_grade(75.0) == "B"
        assert _score_to_grade(74.9) == "C"
        assert _score_to_grade(60.0) == "C"
        assert _score_to_grade(59.9) == "D"
        assert _score_to_grade(40.0) == "D"
        assert _score_to_grade(39.9) == "F"
        assert _score_to_grade(0.0) == "F"


class TestPublicAPI:
    def test_score_in_module(self) -> None:
        assert hasattr(meshbench, "score")
        assert hasattr(meshbench, "ScoreCard")
        assert hasattr(meshbench, "Check")

    def test_default_profile_is_print(self, unit_cube: trimesh.Trimesh) -> None:
        report = meshbench.audit(unit_cube)
        card = meshbench.score(report)
        assert card.profile == "print"
