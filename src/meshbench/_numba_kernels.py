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
