"""Optional numba-accelerated kernels for meshbench.

Used when numba is installed to replace Python inner loops with
JIT-compiled machine code. Pure-Python fallbacks in calling modules
handle the case when numba is not installed.
"""

from __future__ import annotations

import numpy as np
import numba


@numba.njit(cache=True)
def _check_vertices_numba(
    check_indices,
    v_group_verts,
    v_starts,
    v_ends,
    sf,
    a_verts,
    a_starts,
    a_ends,
    adj_fa,
    adj_fb,
):
    """Check face fan connectivity per vertex using union-find.

    For each vertex in check_indices, determines whether its incident
    faces form a single connected component (manifold) or multiple
    components (non-manifold).

    All inputs are 1D int64 numpy arrays. Uses binary search on sorted
    a_verts instead of Python dict lookups.

    Returns int64 array of non-manifold vertex IDs.
    """
    result = np.empty(len(check_indices), dtype=np.int64)
    count = 0

    for idx_i in range(len(check_indices)):
        idx = check_indices[idx_i]
        v = v_group_verts[idx]
        s = v_starts[idx]
        e = v_ends[idx]
        n = e - s
        if n <= 1:
            continue

        incident = sf[s:e]

        # Binary search for vertex in sorted adjacency vertex array
        pos = np.searchsorted(a_verts, v)
        if pos >= len(a_verts) or a_verts[pos] != v:
            # No adjacency data → faces are disconnected
            result[count] = v
            count += 1
            continue

        a_s = a_starts[pos]
        a_e = a_ends[pos]

        # Sort incident faces for binary search face mapping
        incident_sorted = np.sort(incident)

        # Union-find on local indices 0..n-1
        parent = np.arange(n, dtype=np.int64)

        for k in range(a_s, a_e):
            fa = adj_fa[k]
            fb = adj_fb[k]

            # Map global face IDs to local indices via binary search
            la = np.searchsorted(incident_sorted, fa)
            if la >= n or incident_sorted[la] != fa:
                continue
            lb = np.searchsorted(incident_sorted, fb)
            if lb >= n or incident_sorted[lb] != fb:
                continue

            # find(la) with path compression
            x = la
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            ra = x

            # find(lb) with path compression
            x = lb
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            rb = x

            if ra != rb:
                parent[ra] = rb

        # Count unique roots (early exit at 2)
        seen = np.empty(n, dtype=np.int64)
        n_seen = 0
        for i in range(n):
            x = i
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            root = x
            found = False
            for j in range(n_seen):
                if seen[j] == root:
                    found = True
                    break
            if not found:
                seen[n_seen] = root
                n_seen += 1
                if n_seen > 1:
                    break  # Non-manifold confirmed

        if n_seen > 1:
            result[count] = v
            count += 1

    return result[:count]


@numba.njit(cache=True)
def _sat_batch_numba(triangles, pair_a, pair_b):
    """Batch SAT triangle-triangle intersection test.

    Parameters
    ----------
    triangles : float64 array of shape (F, 3, 3)
        Vertex positions for every face.
    pair_a, pair_b : int64 arrays of shape (N,)
        Face-index arrays for the pairs to test.

    Returns
    -------
    bool array of length N — True where the pair intersects.
    """
    n = len(pair_a)
    result = np.empty(n, dtype=numba.boolean)

    for k in range(n):
        tri_a = triangles[pair_a[k]]
        tri_b = triangles[pair_b[k]]

        # Edges of each triangle
        ea0 = tri_a[1] - tri_a[0]
        ea1 = tri_a[2] - tri_a[1]
        ea2 = tri_a[0] - tri_a[2]
        eb0 = tri_b[1] - tri_b[0]
        eb1 = tri_b[2] - tri_b[1]
        eb2 = tri_b[0] - tri_b[2]

        # Face normals
        na = np.empty(3, dtype=np.float64)
        na[0] = ea0[1] * ea1[2] - ea0[2] * ea1[1]
        na[1] = ea0[2] * ea1[0] - ea0[0] * ea1[2]
        na[2] = ea0[0] * ea1[1] - ea0[1] * ea1[0]

        nb = np.empty(3, dtype=np.float64)
        nb[0] = eb0[1] * eb1[2] - eb0[2] * eb1[1]
        nb[1] = eb0[2] * eb1[0] - eb0[0] * eb1[2]
        nb[2] = eb0[0] * eb1[1] - eb0[1] * eb1[0]

        # Build 11 candidate axes: 2 normals + 9 edge cross products
        # Store in a flat (11, 3) array
        axes = np.empty((11, 3), dtype=np.float64)
        axes[0] = na
        axes[1] = nb

        # 9 edge cross products: ea_i x eb_j
        edge_a = np.empty((3, 3), dtype=np.float64)
        edge_a[0] = ea0
        edge_a[1] = ea1
        edge_a[2] = ea2
        edge_b = np.empty((3, 3), dtype=np.float64)
        edge_b[0] = eb0
        edge_b[1] = eb1
        edge_b[2] = eb2

        idx = 2
        for i in range(3):
            for j in range(3):
                ax = np.empty(3, dtype=np.float64)
                ax[0] = edge_a[i][1] * edge_b[j][2] - edge_a[i][2] * edge_b[j][1]
                ax[1] = edge_a[i][2] * edge_b[j][0] - edge_a[i][0] * edge_b[j][2]
                ax[2] = edge_a[i][0] * edge_b[j][1] - edge_a[i][1] * edge_b[j][0]
                axes[idx] = ax
                idx += 1

        separated = False
        for ai in range(11):
            ax = axes[ai]
            length_sq = ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2]
            if length_sq < 1e-30:
                continue

            # Project triangle A
            min_a = 1e30
            max_a = -1e30
            for vi in range(3):
                d = tri_a[vi][0] * ax[0] + tri_a[vi][1] * ax[1] + tri_a[vi][2] * ax[2]
                if d < min_a:
                    min_a = d
                if d > max_a:
                    max_a = d

            # Project triangle B
            min_b = 1e30
            max_b = -1e30
            for vi in range(3):
                d = tri_b[vi][0] * ax[0] + tri_b[vi][1] * ax[1] + tri_b[vi][2] * ax[2]
                if d < min_b:
                    min_b = d
                if d > max_b:
                    max_b = d

            if max_a < min_b or max_b < min_a:
                separated = True
                break

        result[k] = not separated

    return result
