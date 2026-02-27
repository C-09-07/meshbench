"""Tests for report serialization (to_dict / JSON round-trip)."""

from __future__ import annotations

import json

import numpy as np
import trimesh

import meshbench


class TestToDict:
    def test_json_round_trip(self, unit_cube: trimesh.Trimesh) -> None:
        report = meshbench.audit(unit_cube)
        d = report.to_dict()
        text = json.dumps(d)
        parsed = json.loads(text)
        assert isinstance(parsed, dict)
        assert parsed["vertex_count"] == report.vertex_count

    def test_ndarray_becomes_list(self) -> None:
        fp = meshbench.Fingerprint(
            hollowness=0.5,
            thinness=1.0,
            aspect_ratio=2.0,
            symmetry_score=0.9,
            normal_entropy=3.0,
            component_count=1,
            taper_ratio=0.1,
            taper_direction=np.array([1.0, 0.0, 0.0]),
        )
        d = fp.to_dict()
        assert isinstance(d["taper_direction"], list)
        assert d["taper_direction"] == [1.0, 0.0, 0.0]

    def test_none_preserved(self) -> None:
        fp = meshbench.Fingerprint(
            hollowness=0.5,
            thinness=1.0,
            aspect_ratio=2.0,
            symmetry_score=0.9,
            normal_entropy=3.0,
            component_count=1,
            taper_ratio=None,
            taper_direction=None,
        )
        d = fp.to_dict()
        assert d["taper_ratio"] is None
        assert d["taper_direction"] is None

    def test_nested_dataclass_becomes_dict(self, unit_cube: trimesh.Trimesh) -> None:
        report = meshbench.audit(unit_cube)
        d = report.to_dict()
        assert isinstance(d["fingerprint"], dict)
        assert isinstance(d["manifold"], dict)
        assert isinstance(d["cell"], dict)

    def test_np_integer_becomes_int(self) -> None:
        from meshbench.types import _to_dict_value

        assert isinstance(_to_dict_value(np.int64(42)), int)
        assert _to_dict_value(np.int64(42)) == 42

    def test_np_floating_becomes_float(self) -> None:
        from meshbench.types import _to_dict_value

        assert isinstance(_to_dict_value(np.float32(3.14)), float)

    def test_dict_values_converted(self) -> None:
        from meshbench.types import _to_dict_value

        result = _to_dict_value({"a": np.int64(1), "b": np.array([1, 2])})
        assert result == {"a": 1, "b": [1, 2]}

    def test_list_values_converted(self) -> None:
        from meshbench.types import _to_dict_value

        result = _to_dict_value([np.int64(1), np.float64(2.5)])
        assert result == [1, 2.5]
