# meshbench

Geometric mesh quality analysis library. Extracts shape fingerprints, manifold integrity, normal consistency, topological metrics, per-face cell quality, and polycount efficiency from triangle meshes.

Built on [trimesh](https://trimesh.org/), pure NumPy — no PyVista or VTK required.

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python 3.11+.

## Quick Start

```python
import meshbench

mesh = meshbench.load("model.glb")
report = meshbench.audit(mesh)

print(f"Watertight: {report.manifold.is_watertight}")
print(f"Genus: {report.topology.genus}")
print(f"Winding consistency: {report.normals.winding_consistency:.3f}")
print(f"Aspect ratio (mean): {report.cell.aspect_ratio_mean:.2f}")
print(f"Min angle (worst): {report.cell.min_angle_min:.1f}°")
```

## API

### `meshbench.load(path, force_geometry=False) -> Trimesh`

Load a mesh file (STL, OBJ, GLB, PLY, etc.). Handles `Scene` → `Trimesh` flattening automatically.

- `force_geometry=True` strips textures/materials, rebuilds bare geometry
- Raises `ValueError` on failure

### `meshbench.audit(mesh) -> MeshReport`

Full analysis. Returns a `MeshReport` with sub-reports:

| Sub-report | Key metrics |
|---|---|
| **`fingerprint`** | hollowness, thinness, aspect ratio, symmetry, normal entropy, component count, taper |
| **`manifold`** | boundary edges/loops, non-manifold edges/vertices, watertight bool, ratios |
| **`normals`** | entropy, flipped count, winding consistency, backface divergence, dominant normal |
| **`topology`** | genus, valence histogram/stats, degenerate faces, floating vertices, components, ratios |
| **`density`** | vertices per area, face density variance |
| **`cell`** | aspect ratio (mean/std/max/p95), min angle (mean/std/min/p05) |

### `meshbench.fingerprint(mesh) -> Fingerprint`

Shape fingerprint only.

### `meshbench.manifold(mesh) -> ManifoldReport`

Manifold analysis only.

## JSON Serialization

All reports have a `.to_dict()` method that returns a JSON-serializable dict:

```python
import json, meshbench

report = meshbench.audit(mesh)
d = report.to_dict()
print(json.dumps(d, indent=2))
```

Handles `numpy.ndarray` → `list`, `numpy.integer` → `int`, `numpy.floating` → `float`, nested dataclasses → `dict`, and `None` passthrough automatically.

## Benchmarking

Run the benchmark suite against your mesh corpus:

```bash
# Audit all meshes, compare AI sources against a reference
python -m benchmarks.run --meshes meshes/ --reference-name thingi10k --format both

# Generate charts from a JSON report
python -m benchmarks.charts output/benchmark_*.json --output output/charts/
```

### Corpus layout

```
meshes/
├── thingi10k/     # 20 STLs — real-world 3D prints (reference)
├── abc/           # 300 OBJs — CAD models (reference)
├── khronos/       # 20 GLBs — Khronos sample models (reference)
├── hunyuan3d/     # 28 GLBs — Tencent Hunyuan3D
├── triposg/       # 28 GLBs — TripoSG
├── sf3d/          # 28 GLBs — Stability AI SF3D
├── meshyai/       # 7 GLBs  — Meshy AI
└── tripo3d/       # 10 GLBs — Tripo3D
```

### Benchmark output

- **JSON report** with per-mesh audit data, per-source aggregates, failure classification, and delta vs reference
- **Summary table** comparing sources side-by-side
- **Charts**: radar quality profile, box plot distributions, failure mode frequency, winding vs density scatter, manifold status stacked bars

### Results: 445 meshes across 8 sources (v1.0.0)

Reference: Thingi10k (real-world 3D prints). Delta columns show difference from reference.

```
Source               N    WT%    Wind   BfDiv   Genus   ValStd   Degen   Float     FDV   Errs   Warn
----------------------------------------------------------------------------------------------------
thingi10k           20  85.0%   0.951   0.394     5.7     2.04      45       0  20.931     35     24
abc                300   1.0%   1.000   0.371     2.3     1.03       0       0   0.124    395    114
khronos             20   0.0%   0.995   0.378     0.0     1.30      45       0  20.037     48     17
hunyuan3d           27  77.8%   1.000   1.315     0.0     0.72  551895       0   8.178     88     33
triposg             27  81.5%   1.000   0.456    13.4     0.68      15       0   0.106     31     30
sf3d                27   0.0%   0.999   0.326     0.0     1.38      14       0   0.668     54     56
meshyai             14  100.0%  0.995   0.328   340.5     1.10     237       0   1.864     29     14
tripo3d             10  60.0%   0.998   0.246    11.6     1.02       2       0   0.459     12     13
```

**Key findings:**

| Metric | Pro meshes (thingi10k, abc, khronos) | AI-generated |
|--------|--------------------------------------|-------------|
| **Watertight** | 0-85% (abc is CAD = mostly open) | 0-100% (varies wildly by generator) |
| **Winding consistency** | 0.951-1.000 | 0.995-1.000 (AI is actually consistent) |
| **Degenerate faces** | 0-45 | 2-551K (Hunyuan3D: ~50% of faces degenerate) |
| **Genus** | 0-5.7 | 0-340 (MeshyAI: extreme topological noise) |
| **Non-manifold** | Common in all sources | SF3D: 100% have non-manifold vertices |
| **Face density variance** | 0.12-20.9 | 0.11-8.2 (AI is more uniform) |

**Per-generator failure signatures:**

- **Hunyuan3D** (Tencent): Massive face counts (up to 3.6M), 100% have non-manifold geometry and degenerate faces (mean 551K degenerates/mesh). Good winding but terrible cell quality.
- **TripoSG**: Cleanest AI generator — low degenerates, good density uniformity. Main issue is high genus (mean 13.4) and disconnected components.
- **SF3D** (Stability AI): Every mesh has non-manifold vertices and open boundaries. Small meshes (~25K faces) but universally broken topology.
- **MeshyAI**: 100% watertight but extreme genus (mean 340.5) — topological noise from self-intersections. 57% have >4 disconnected components.
- **Tripo3D**: Middle-of-the-road. 60% watertight, some genus issues, relatively clean otherwise.

## Running Tests

```bash
pytest tests/ -v
```

84 tests covering all modules.

## Architecture

```
src/meshbench/
├── __init__.py      # Public API: audit(), fingerprint(), manifold(), load()
├── types.py         # Dataclasses with _ReportMixin (to_dict)
├── loading.py       # Mesh file I/O with scene flattening
├── _edges.py        # Vectorized edge utilities (shared)
├── _entropy.py      # Shannon entropy helper (shared)
├── _cell.py         # Per-face aspect ratio + min angle
├── density.py       # Polycount efficiency metrics
├── fingerprint.py   # PCA-based shape descriptors
├── manifold.py      # Boundary/non-manifold analysis
├── normals.py       # Face normal consistency
└── topology.py      # Genus, valence, degenerates, components
```

`audit()` computes edges and PCA once, then threads shared data to all sub-reports.
