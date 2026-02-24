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
