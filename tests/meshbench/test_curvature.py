"""Tests for discrete curvature estimation."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from meshbench.curvature import (
    compute_curvature_report,
    discrete_gaussian_curvature,
    discrete_mean_curvature,
    per_face_curvature,
)


class TestMeanCurvature:
    def test_sphere_uniform_H(self, icosphere: trimesh.Trimesh):
        """Icosphere |H| should be near-uniform (all vertices on sphere)."""
        H = discrete_mean_curvature(icosphere)
        assert H.shape[0] == icosphere.vertices.shape[0]
        # All values should be similar (low std relative to mean)
        H_std = np.std(H)
        H_mean = np.mean(H)
        assert H_mean > 0
        assert H_std / H_mean < 0.5  # coefficient of variation < 50%

    def test_flat_plane_zero_H(self, flat_plane: trimesh.Trimesh):
        """Flat plane should have H ≈ 0 at interior vertices."""
        H = discrete_mean_curvature(flat_plane)
        # Boundary vertices may have nonzero curvature due to boundary effects.
        # But overall the values should be very small for a flat surface.
        # The flat_plane is just 2 triangles so all vertices are boundary.
        # Use a larger flat plane for a better test.
        n = 10
        verts = []
        faces = []
        for i in range(n):
            for j in range(n):
                verts.append([float(i), 0.0, float(j)])
        for i in range(n - 1):
            for j in range(n - 1):
                v0 = i * n + j
                v1 = v0 + 1
                v2 = v0 + n
                v3 = v2 + 1
                faces.append([v0, v1, v2])
                faces.append([v1, v3, v2])
        plane = trimesh.Trimesh(
            vertices=np.array(verts), faces=np.array(faces), process=False,
        )
        H = discrete_mean_curvature(plane)
        # Interior vertices should have H ≈ 0
        interior = np.zeros(len(verts), dtype=bool)
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                interior[i * n + j] = True
        H_interior = H[interior]
        np.testing.assert_allclose(H_interior, 0.0, atol=1e-6)


class TestGaussianCurvature:
    def test_sphere_positive_K(self, icosphere: trimesh.Trimesh):
        """Icosphere should have K > 0 everywhere."""
        K = discrete_gaussian_curvature(icosphere)
        assert np.all(K > 0)

    def test_flat_plane_zero_K(self):
        """Interior vertices on a flat grid should have K ≈ 0."""
        n = 10
        verts = []
        faces = []
        for i in range(n):
            for j in range(n):
                verts.append([float(i), 0.0, float(j)])
        for i in range(n - 1):
            for j in range(n - 1):
                v0 = i * n + j
                v1 = v0 + 1
                v2 = v0 + n
                v3 = v2 + 1
                faces.append([v0, v1, v2])
                faces.append([v1, v3, v2])
        plane = trimesh.Trimesh(
            vertices=np.array(verts), faces=np.array(faces), process=False,
        )
        K = discrete_gaussian_curvature(plane)
        interior = np.zeros(len(verts), dtype=bool)
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                interior[i * n + j] = True
        K_interior = K[interior]
        np.testing.assert_allclose(K_interior, 0.0, atol=1e-6)

    def test_gauss_bonnet_sphere(self, icosphere: trimesh.Trimesh):
        """Gauss-Bonnet: sum(K * A) ≈ 4π for a closed genus-0 surface."""
        K = discrete_gaussian_curvature(icosphere)
        # Use vertex areas (face_area / 3 accumulated)
        n_verts = icosphere.vertices.shape[0]
        vertex_areas = np.zeros(n_verts, dtype=np.float64)
        face_areas = icosphere.area_faces
        area_third = face_areas / 3.0
        np.add.at(vertex_areas, icosphere.faces[:, 0], area_third)
        np.add.at(vertex_areas, icosphere.faces[:, 1], area_third)
        np.add.at(vertex_areas, icosphere.faces[:, 2], area_third)
        total = np.sum(K * vertex_areas)
        np.testing.assert_allclose(total, 4 * np.pi, rtol=0.01)


class TestPerFaceCurvature:
    def test_per_face_shape(self, icosphere: trimesh.Trimesh):
        output = per_face_curvature(icosphere)
        assert output.shape[0] == icosphere.faces.shape[0]


class TestCurvatureReport:
    def test_curvature_report_serializable(self, icosphere: trimesh.Trimesh):
        report = compute_curvature_report(icosphere)
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "mean_curvature_abs_avg" in d
        assert "flat_vertex_ratio" in d

    def test_empty_mesh(self):
        mesh = trimesh.Trimesh()
        report = compute_curvature_report(mesh)
        assert report.mean_curvature_abs_avg == 0.0
        assert report.flat_vertex_ratio == 1.0
