"""Tests for meshbench.fingerprint."""

from __future__ import annotations

import numpy as np
import trimesh

from meshbench.fingerprint import compute_fingerprint, compute_pca, compute_taper_vector


class TestComputePCA:
    def test_axes_shape(self, unit_cube: trimesh.Trimesh) -> None:
        axes, extents = compute_pca(unit_cube)
        assert axes.shape == (3, 3)
        assert extents.shape == (3,)

    def test_extents_decreasing(self, tall_box: trimesh.Trimesh) -> None:
        _, extents = compute_pca(tall_box)
        assert extents[0] >= extents[1] >= extents[2]

    def test_tall_box_primary_axis(self, tall_box: trimesh.Trimesh) -> None:
        axes, extents = compute_pca(tall_box)
        # Primary axis should be along Y (the tall dimension)
        y_component = abs(axes[0] @ np.array([0, 1, 0]))
        assert y_component > 0.9


class TestComputeFingerprint:
    def test_cube_symmetry(self, unit_cube: trimesh.Trimesh) -> None:
        fp = compute_fingerprint(unit_cube)
        # Cube should have high symmetry (PC2 ≈ PC3)
        assert fp.symmetry_score > 0.8

    def test_cube_hollowness(self, unit_cube: trimesh.Trimesh) -> None:
        fp = compute_fingerprint(unit_cube)
        # Solid cube: mesh volume ≈ hull volume
        assert fp.hollowness > 0.9

    def test_cube_component_count(self, unit_cube: trimesh.Trimesh) -> None:
        fp = compute_fingerprint(unit_cube)
        assert fp.component_count == 1

    def test_two_components(self, two_component_mesh: trimesh.Trimesh) -> None:
        fp = compute_fingerprint(two_component_mesh)
        assert fp.component_count == 2

    def test_tall_box_aspect_ratio(self, tall_box: trimesh.Trimesh) -> None:
        fp = compute_fingerprint(tall_box)
        # 1.8 / 0.4 → aspect ratio significantly > 1
        assert fp.aspect_ratio > 2.0

    def test_sphere_entropy(self, icosphere: trimesh.Trimesh) -> None:
        fp = compute_fingerprint(icosphere)
        # Sphere has diffuse normals → high entropy
        assert fp.normal_entropy > 3.0

    def test_no_taper_on_cube(self, unit_cube: trimesh.Trimesh) -> None:
        fp = compute_fingerprint(unit_cube)
        # Cube is symmetric — no taper
        assert fp.taper_ratio is None
        assert fp.taper_direction is None

    def test_fields_are_rounded(self, unit_cube: trimesh.Trimesh) -> None:
        fp = compute_fingerprint(unit_cube)
        # Check that float fields have limited precision
        assert fp.hollowness == round(fp.hollowness, 4)
        assert fp.thinness == round(fp.thinness, 2)
        assert fp.aspect_ratio == round(fp.aspect_ratio, 2)


class TestTaper:
    def test_tapered_mesh_has_taper(self, tapered_mesh: trimesh.Trimesh) -> None:
        """A tapered surface of revolution should have significant taper."""
        axes, _ = compute_pca(tapered_mesh)
        result = compute_taper_vector(tapered_mesh, axes)
        assert result is not None
        direction, ratio = result
        assert ratio >= 2.0
        assert np.linalg.norm(direction) > 0.99

    def test_fingerprint_includes_taper(self, tapered_mesh: trimesh.Trimesh) -> None:
        fp = compute_fingerprint(tapered_mesh)
        assert fp.taper_ratio is not None
        assert fp.taper_ratio >= 2.0
        assert fp.taper_direction is not None

    def test_small_mesh_returns_none(self) -> None:
        """Mesh with <20 vertices should return None."""
        verts = np.random.randn(10, 3)
        faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        axes, _ = compute_pca(mesh)
        assert compute_taper_vector(mesh, axes) is None
