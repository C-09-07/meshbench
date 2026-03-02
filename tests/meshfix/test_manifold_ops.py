"""Tests for non-manifold edge/vertex splitting."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from meshbench._edges import edge_face_counts
from meshbench.manifold import non_manifold_edges, non_manifold_vertices
from meshfix._manifold_ops import split_non_manifold_edges, split_non_manifold_vertices


class TestSplitNonManifoldEdges:
    def test_splits_triple_edge(self, nm_edge_mesh):
        """3 faces on 1 edge -> vertex duplication resolves NM edge."""
        result = split_non_manifold_edges(nm_edge_mesh)

        # Should have more vertices (duplicates of the shared edge endpoints)
        assert result.vertices.shape[0] > nm_edge_mesh.vertices.shape[0]

        # Face count preserved
        assert result.faces.shape[0] == nm_edge_mesh.faces.shape[0]

        # No NM edges remain
        nm = non_manifold_edges(result)
        assert len(nm) == 0

    def test_noop_on_clean_mesh(self, unit_cube):
        """Manifold mesh -> copy unchanged."""
        result = split_non_manifold_edges(unit_cube)

        assert result.vertices.shape[0] == unit_cube.vertices.shape[0]
        assert result.faces.shape[0] == unit_cube.faces.shape[0]

    def test_with_precomputed_indices(self, nm_edge_mesh):
        """Pass Defects.non_manifold_edges as edge_pairs."""
        nm_pairs = non_manifold_edges(nm_edge_mesh)
        result = split_non_manifold_edges(nm_edge_mesh, edge_pairs=nm_pairs)

        assert result.faces.shape[0] == nm_edge_mesh.faces.shape[0]

        nm_after = non_manifold_edges(result)
        assert len(nm_after) == 0

    def test_does_not_mutate_input(self, nm_edge_mesh):
        """Input mesh not modified."""
        orig_verts = nm_edge_mesh.vertices.copy()
        orig_faces = nm_edge_mesh.faces.copy()

        split_non_manifold_edges(nm_edge_mesh)

        np.testing.assert_array_equal(nm_edge_mesh.vertices, orig_verts)
        np.testing.assert_array_equal(nm_edge_mesh.faces, orig_faces)

    def test_face_count_preserved(self, nm_edge_mesh):
        """No faces removed during edge splitting."""
        result = split_non_manifold_edges(nm_edge_mesh)
        assert result.faces.shape[0] == nm_edge_mesh.faces.shape[0]

    def test_vertex_count_increases(self, nm_edge_mesh):
        """Vertices added for each extra sheet."""
        result = split_non_manifold_edges(nm_edge_mesh)
        # 3 faces on 1 edge: at least 2 sheets, so at least 2 new vertices
        assert result.vertices.shape[0] > nm_edge_mesh.vertices.shape[0]

    def test_empty_edge_pairs_noop(self, unit_cube):
        """Empty edge_pairs array -> no-op."""
        result = split_non_manifold_edges(
            unit_cube, edge_pairs=np.empty((0, 2), dtype=np.int64)
        )
        assert result.vertices.shape[0] == unit_cube.vertices.shape[0]


class TestSplitNonManifoldVertices:
    def test_splits_bowtie(self, bowtie_mesh):
        """Bowtie -> vertex duplicated, fans separated."""
        result = split_non_manifold_vertices(bowtie_mesh)

        # Should have 1 more vertex (bowtie vertex split into 2)
        assert result.vertices.shape[0] == bowtie_mesh.vertices.shape[0] + 1

        # Face count preserved
        assert result.faces.shape[0] == bowtie_mesh.faces.shape[0]

        # No NM vertices remain
        nm = non_manifold_vertices(result)
        assert len(nm) == 0

    def test_noop_on_clean_mesh(self, unit_cube):
        """Manifold mesh -> copy unchanged."""
        result = split_non_manifold_vertices(unit_cube)

        assert result.vertices.shape[0] == unit_cube.vertices.shape[0]
        assert result.faces.shape[0] == unit_cube.faces.shape[0]

    def test_with_precomputed_indices(self, bowtie_mesh):
        """Pass Defects.non_manifold_vertices as vert_indices."""
        nm_verts = np.array(non_manifold_vertices(bowtie_mesh), dtype=np.intp)
        result = split_non_manifold_vertices(bowtie_mesh, vert_indices=nm_verts)

        assert result.vertices.shape[0] > bowtie_mesh.vertices.shape[0]

        nm_after = non_manifold_vertices(result)
        assert len(nm_after) == 0

    def test_does_not_mutate_input(self, bowtie_mesh):
        """Input mesh not modified."""
        orig_verts = bowtie_mesh.vertices.copy()
        orig_faces = bowtie_mesh.faces.copy()

        split_non_manifold_vertices(bowtie_mesh)

        np.testing.assert_array_equal(bowtie_mesh.vertices, orig_verts)
        np.testing.assert_array_equal(bowtie_mesh.faces, orig_faces)

    def test_face_count_preserved(self, bowtie_mesh):
        """No faces removed during vertex splitting."""
        result = split_non_manifold_vertices(bowtie_mesh)
        assert result.faces.shape[0] == bowtie_mesh.faces.shape[0]

    def test_each_fan_has_own_vertex(self, bowtie_mesh):
        """After split, each face fan uses a distinct vertex."""
        result = split_non_manifold_vertices(bowtie_mesh)

        # The two faces should no longer share any vertex
        f0_verts = set(result.faces[0].tolist())
        f1_verts = set(result.faces[1].tolist())
        assert len(f0_verts & f1_verts) == 0
