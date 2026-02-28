"""Tests for meshbench.load()."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import trimesh

import meshbench


class TestLoad:
    def test_stl_round_trip(self) -> None:
        mesh = trimesh.creation.box(extents=[1, 1, 1])
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
            mesh.export(f.name)
            loaded = meshbench.load(f.name)
        assert isinstance(loaded, trimesh.Trimesh)
        assert loaded.faces.shape[0] == mesh.faces.shape[0]

    def test_str_path(self) -> None:
        mesh = trimesh.creation.box(extents=[1, 1, 1])
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
            mesh.export(f.name)
            loaded = meshbench.load(str(f.name))
        assert isinstance(loaded, trimesh.Trimesh)

    def test_path_object(self) -> None:
        mesh = trimesh.creation.box(extents=[1, 1, 1])
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
            mesh.export(f.name)
            loaded = meshbench.load(Path(f.name))
        assert isinstance(loaded, trimesh.Trimesh)

    def test_force_geometry(self) -> None:
        mesh = trimesh.creation.box(extents=[1, 1, 1])
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
            mesh.export(f.name)
            loaded = meshbench.load(f.name, force_geometry=True)
        assert isinstance(loaded, trimesh.Trimesh)
        np.testing.assert_array_equal(loaded.vertices.shape, mesh.vertices.shape)

    def test_nonexistent_raises(self) -> None:
        with pytest.raises(ValueError, match="Failed to load"):
            meshbench.load("/tmp/nonexistent_mesh_12345.stl")
