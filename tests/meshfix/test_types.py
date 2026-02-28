"""Tests for meshfix type serialization."""

from __future__ import annotations

import json

from meshfix.types import (
    Backend,
    FixConfig,
    FixResult,
    FixStep,
    FixStepName,
)


class TestFixStep:
    def test_to_dict(self):
        """FixStep serializes correctly."""
        step = FixStep(
            name=FixStepName.REMOVE_DEGENERATES,
            applied=True,
            backend=Backend.NUMPY,
            faces_removed=5,
            duration_ms=1.23,
        )
        d = step.to_dict()

        assert d["applied"] is True
        assert d["faces_removed"] == 5
        assert d["duration_ms"] == 1.23
        # Enum values should be serializable
        json_str = json.dumps(d, default=str)
        assert "REMOVE_DEGENERATES" in json_str

    def test_skipped_step(self):
        """Skipped step has reason."""
        step = FixStep(
            name=FixStepName.FIX_WINDING,
            applied=False,
            backend=Backend.TRIMESH,
            skipped_reason="No flipped faces detected",
        )
        d = step.to_dict()

        assert d["applied"] is False
        assert d["skipped_reason"] == "No flipped faces detected"


class TestFixResult:
    def test_to_dict(self):
        """FixResult JSON-serializable round-trip."""
        result = FixResult(
            steps=[
                FixStep(
                    name=FixStepName.REMOVE_DEGENERATES,
                    applied=True,
                    backend=Backend.NUMPY,
                    faces_removed=3,
                ),
                FixStep(
                    name=FixStepName.FIX_NORMALS,
                    applied=True,
                    backend=Backend.TRIMESH,
                ),
            ],
            total_duration_ms=5.67,
            vertices_before=100,
            vertices_after=97,
            faces_before=200,
            faces_after=197,
        )
        d = result.to_dict()

        assert len(d["steps"]) == 2
        assert d["total_duration_ms"] == 5.67
        assert d["vertices_before"] == 100
        assert d["faces_after"] == 197
        assert d["before"] is None
        assert d["after"] is None

        # Full round-trip through JSON
        json_str = json.dumps(d, default=str)
        parsed = json.loads(json_str)
        assert parsed["faces_before"] == 200

    def test_empty_result(self):
        """Empty result serializes cleanly."""
        result = FixResult()
        d = result.to_dict()

        assert d["steps"] == []
        assert d["total_duration_ms"] == 0.0


class TestFixConfig:
    def test_defaults(self):
        """Default config has expected values."""
        config = FixConfig()

        assert config.remove_degenerates is True
        assert config.remove_floating_vertices is True
        assert config.remove_small_shells is True
        assert config.min_shell_face_count == 10
        assert config.min_shell_face_ratio == 0.01
        assert config.fix_winding is True
        assert config.fix_normals is True

    def test_custom_values(self):
        """Custom config overrides."""
        config = FixConfig(
            remove_degenerates=False,
            min_shell_face_count=50,
        )

        assert config.remove_degenerates is False
        assert config.min_shell_face_count == 50
        # Others still default
        assert config.fix_normals is True
