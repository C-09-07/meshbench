# meshbench

Geometric mesh quality analysis library. Extracts shape fingerprints, manifold integrity, normal consistency, topological metrics, and polycount efficiency from triangle meshes.

Built on [trimesh](https://trimesh.org/).

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python 3.11+.

## Quick Start

```python
import meshbench
import trimesh

mesh = trimesh.load("model.glb")
report = meshbench.audit(mesh)

print(f"Watertight: {report.manifold.is_watertight}")
print(f"Genus: {report.topology.genus}")
print(f"Winding consistency: {report.normals.winding_consistency}")
print(f"Hollowness: {report.fingerprint.hollowness}")
```

## API

### `meshbench.audit(mesh) -> MeshReport`

Full analysis. Returns a `MeshReport` with sub-reports:

- **`fingerprint`** — Hollowness, thinness, aspect ratio, symmetry, normal entropy, component count, taper
- **`manifold`** — Boundary edges, boundary loops (holes), non-manifold edges/vertices, watertight bool
- **`normals`** — Entropy, flipped face count, winding consistency, backface divergence, dominant normal
- **`topology`** — Genus, valence histogram/stats, degenerate faces, floating vertices, component sizes
- **`density`** — Vertices per area, face density variance

### `meshbench.fingerprint(mesh) -> Fingerprint`

Shape fingerprint only.

### `meshbench.manifold(mesh) -> ManifoldReport`

Manifold analysis only.

### `meshbench.compare(mesh_a, mesh_b)`

Placeholder — raises `NotImplementedError`.

## JSON Serialization

All reports are dataclasses:

```python
import dataclasses, json

d = dataclasses.asdict(report)
# Note: numpy arrays need a custom encoder for json.dumps()
```

## Running Tests

```bash
pytest tests/ -q
```

## Future: Curve-Aware Decomposition

Current taper detection uses PCA (a single straight axis) and measures cross-sectional spread at the two endpoints. This works for straight props (swords, guitars) but fails on curved geometry — e.g. a scorpion tail that tapers from base to tip along a semi-circular curve. PCA projects the extremes onto a straight line, so the endpoint slices don't correspond to "base" vs "tip" and the perpendicular spread gets contaminated by the curve itself.

The fix is SVD-based cross-section analysis along a fitted spine:

1. **Fit a centerline** — medial axis, curve skeleton, or spline through clustered centroids
2. **Sample cross-sections** perpendicular to the local tangent at N points along the spine
3. **SVD on each cross-section's vertices** — gives ellipse axes (semi-major/semi-minor) without assuming circular symmetry
4. **Measure taper as change in ellipse area** (or singular values) along arc length

This generalizes the current approach: PCA taper is the degenerate case where the spine is a straight line (PC1) with only 2 samples. The SVD ellipse sequence also captures twist (cross-section rotation), flattening (ribbon-like geometry), and branching.

**Known failure cases for current taper:** scorpion tails, ram horns, dragon tails, tentacles, S-curves — any prop where the "long axis" isn't straight.

Candidate for a `meshbench.decompose` module. The cross-section ellipse sequence would be a useful shape descriptor independent of taper.
