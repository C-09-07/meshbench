# meshx

Monorepo for geometric mesh analysis, repair, and visualization. Three packages:

| Package | Version | Purpose |
|---------|---------|---------|
| **meshbench** | 1.1.0 | Quality analysis — fingerprints, manifold integrity, normals, topology, cell quality, feature edges, curvature |
| **meshfix** | 0.1.0 | Targeted repair — degenerate removal, winding fixes, QEM decimation, adaptive remeshing |
| **meshview** | 0.1.0 | Interactive Three.js viewer — defect visualization, data export |

Built on [trimesh](https://trimesh.org/), pure NumPy/SciPy. Processes ~1M faces/sec on clean geometry.

## Installation

```bash
# uv workspace (recommended)
uv sync --all-packages

# Or install individual packages
pip install -e packages/meshbench
pip install -e packages/meshfix
pip install -e packages/meshview

# Optional: ~5x faster manifold analysis
pip install meshbench[fast]          # numba

# Optional: better decimation quality
pip install meshfix[pymeshlab]       # pymeshlab QEM with quality weighting
```

Requires Python 3.11+.

## Quick Start

### Analyze a mesh

```python
import meshbench

mesh = meshbench.load("model.glb")
report = meshbench.audit(mesh)

print(f"Watertight: {report.manifold.is_watertight}")
print(f"Genus: {report.topology.genus}")
print(f"Winding consistency: {report.normals.winding_consistency:.3f}")
print(f"Aspect ratio (mean): {report.cell.aspect_ratio_mean:.2f}")
print(f"Min angle (worst): {report.cell.min_angle_min:.1f}")
```

### Feature edges and curvature

```python
feat = meshbench.compute_feature_report(mesh)
print(f"Feature edges: {feat.feature_edge_count} ({feat.feature_edge_ratio:.1%})")

curv = meshbench.compute_curvature_report(mesh)
print(f"Mean |H|: {curv.mean_curvature_abs_avg:.4f}, flat: {curv.flat_vertex_ratio:.1%}")
```

### Repair a mesh

```python
import meshfix

fixed, result = meshfix.fix(mesh, report)
print(f"Steps: {[s.name for s in result.steps]}")
print(f"Faces: {result.faces_before} -> {result.faces_after}")
```

### Decimate

```python
# Uniform QEM decimation (with quality weighting via pymeshlab)
decimated, result = meshfix.decimate(mesh, target_ratio=0.5)
print(f"{result.faces_before} -> {result.faces_after} faces")
print(f"Feature edges: {result.feature_edges_before} -> {result.feature_edges_after}")

# Curvature-weighted: fewer faces on flat, more on curved
adaptive, result = meshfix.adaptive_remesh(mesh, target_ratio=0.25, curvature_weight=2.0)
```

### Score against a profile

```python
card = meshbench.score(report, profile="print")  # or "animation", "simulation", "ai_output"
for check in card.checks:
    print(f"  {check.status.value:4s}  {check.label}: {check.detail}")
```

## meshbench API

### `load(path, force_geometry=False) -> Trimesh`

Load a mesh file (STL, OBJ, GLB, PLY, etc.). Handles `Scene` -> `Trimesh` flattening.

### `audit(mesh, include_defects=False) -> MeshReport`

Full analysis. Returns sub-reports:

| Sub-report | Key metrics |
|---|---|
| **`fingerprint`** | hollowness, thinness, aspect ratio, symmetry, normal entropy, component count, taper |
| **`manifold`** | boundary edges/loops, non-manifold edges/vertices, watertight bool, ratios |
| **`normals`** | entropy, flipped count, winding consistency, backface divergence, dominant normal |
| **`topology`** | genus, valence histogram/stats, degenerate faces, floating vertices, components, ratios |
| **`density`** | vertices per area, face density variance |
| **`cell`** | aspect ratio (mean/std/max/p95), min angle (mean/std/min/p05) |

### Feature edges

```python
dihedral_angles(mesh) -> tuple[ndarray, ndarray]     # (edges, angles_deg)
feature_edges(mesh, feature_angle=30.0) -> ndarray    # (N, 2) vertex pairs
compute_feature_report(mesh) -> FeatureReport
```

### Curvature

```python
discrete_mean_curvature(mesh) -> ndarray              # per-vertex |H|
discrete_gaussian_curvature(mesh) -> ndarray           # per-vertex K
per_face_curvature(mesh) -> ndarray                    # averaged to faces
compute_curvature_report(mesh) -> CurvatureReport
```

### Other

```python
fingerprint(mesh) -> Fingerprint
manifold(mesh) -> ManifoldReport
score(report, profile="print") -> ScoreCard
```

## meshfix API

### `fix(mesh, report=None, config=None) -> (Trimesh, FixResult)`

Auto-repair driven by meshbench analysis. Runs only the steps the mesh needs.

### `plan(report, config=None) -> list[FixStepName]`

Preview which steps `fix()` would run without executing them.

### `decimate(mesh, target_faces=None, target_ratio=None, config=None) -> (Trimesh, DecimateResult)`

QEM decimation with per-vertex quality weighting. Uses pymeshlab when available (biases QEM to preserve high-curvature vertices and feature edges), falls back to fast-simplification.

### `adaptive_remesh(mesh, target_faces=None, target_ratio=None, curvature_weight=1.0, feature_angle=30.0) -> (Trimesh, DecimateResult)`

Curvature-weighted decimation. Same backend as `decimate()` but exposes `curvature_weight` to control how strongly curvature biases the importance map.

### Individual operations

All return new Trimesh, never mutate input:

```python
remove_degenerates(mesh, face_indices=None) -> Trimesh
remove_floating_vertices(mesh, vert_indices=None) -> Trimesh
remove_small_shells(mesh, min_face_count=None, min_face_ratio=None) -> Trimesh
fix_winding(mesh) -> Trimesh
fix_normals(mesh) -> Trimesh
```

### `available_backends() -> dict[Backend, bool]`

Check which optional backends are installed (pymeshlab, manifold3d, pynanoinstantmeshes).

## meshview API

### `export_viewer_data(mesh, report, dir) -> Path`

Export mesh + defect data for the interactive Three.js viewer.

### `build_defects_json(report) -> dict`

Build defect overlay data from a MeshReport.

## JSON Serialization

All reports have `.to_dict()` returning JSON-serializable dicts. Handles numpy types automatically.

```python
import json, meshbench

report = meshbench.audit(mesh)
print(json.dumps(report.to_dict(), indent=2))
```

## Benchmarking

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

### Results: 445 meshes across 8 sources

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

**Per-generator failure signatures:**

- **Hunyuan3D** (Tencent): Massive face counts (up to 3.6M), 100% have non-manifold geometry and degenerate faces (mean 551K degenerates/mesh). Good winding but terrible cell quality.
- **TripoSG**: Cleanest AI generator — low degenerates, good density uniformity. Main issue is high genus (mean 13.4) and disconnected components.
- **SF3D** (Stability AI): Every mesh has non-manifold vertices and open boundaries. Small meshes (~25K faces) but universally broken topology.
- **MeshyAI**: 100% watertight but extreme genus (mean 340.5) — topological noise from self-intersections. 57% have >4 disconnected components.
- **Tripo3D**: Middle-of-the-road. 60% watertight, some genus issues, relatively clean otherwise.

## Performance

`audit()` uses vectorized NumPy/SciPy throughout, with an optional [numba](https://numba.pydata.org/) JIT kernel for the non-manifold vertex check.

| Mesh | Faces | Time | Throughput |
|------|-------|------|-----------|
| Thingi10k STL (clean) | 22K | 0.024s | 950K f/s |
| SF3D GLB (non-manifold) | 29K | 0.028s | 1.05M f/s |
| TripoSG GLB (large) | 1.6M | 1.9s | 840K f/s |
| Hunyuan3D GLB (messy, 50% degenerate) | 1.3M | 2.3s | 570K f/s |
| TripoSG GLB (very large) | 4.6M | 6.2s | 750K f/s |

## Running Tests

```bash
uv run pytest tests/ -v
```

215 tests across all packages.

## Architecture

```
packages/
├── meshbench/src/meshbench/
│   ├── __init__.py        # Public API: audit(), fingerprint(), manifold(), score(), load()
│   ├── types.py           # Report dataclasses with _ReportMixin (to_dict)
│   ├── loading.py         # Mesh file I/O with scene flattening
│   ├── _edges.py          # Vectorized edge utilities + EdgeFaceMap (shared)
│   ├── _entropy.py        # Shannon entropy helper (shared)
│   ├── _numba_kernels.py  # Optional JIT-compiled kernels (numba)
│   ├── _cell.py           # Per-face aspect ratio + min angle
│   ├── _intersections.py  # Self-intersection detection (sweep-and-prune + SAT)
│   ├── curvature.py       # Cotangent Laplacian mean, angle deficit Gaussian
│   ├── density.py         # Polycount efficiency metrics
│   ├── features.py        # Dihedral angles, feature edge detection
│   ├── fingerprint.py     # PCA-based shape descriptors
│   ├── manifold.py        # Boundary/non-manifold analysis
│   ├── normals.py         # Face normal consistency
│   ├── scoring.py         # Profile-based scoring
│   └── topology.py        # Genus, valence, degenerates, components
│
├── meshfix/src/meshfix/
│   ├── __init__.py        # Public API: fix(), plan(), decimate(), adaptive_remesh()
│   ├── types.py           # FixConfig, DecimateConfig, Backend enum
│   ├── _backends.py       # Backend availability checking
│   ├── _decimate.py       # QEM decimation with quality weighting
│   ├── _remesh.py         # Curvature-weighted adaptive decimation
│   ├── _normal_ops.py     # fix_winding, fix_normals
│   ├── _numpy_ops.py      # remove_degenerates, remove_floating_vertices, remove_small_shells
│   ├── _pipeline.py       # Smart repair orchestration
│   └── _plan.py           # Plan what steps would fix
│
└── meshview/src/meshview/
    ├── __init__.py        # Public API: build_defects_json(), export_viewer_data()
    └── export.py          # Defect JSON + Three.js viewer data
```

`audit()` builds an `EdgeFaceMap` once, derives edge data and PCA, then threads shared structures to all sub-reports — no redundant computation across modules.
