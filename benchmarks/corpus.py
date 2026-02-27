"""Load meshes from directory tree, run audit on each."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import trimesh

import meshbench

logger = logging.getLogger(__name__)

MESH_EXTENSIONS = {".glb", ".gltf", ".obj", ".stl", ".ply", ".fbx"}


def discover_meshes(root: Path) -> list[Path]:
    """Recursively find all mesh files under root."""
    paths = []
    for ext in MESH_EXTENSIONS:
        paths.extend(root.rglob(f"*{ext}"))
    return sorted(set(paths))


def load_mesh(path: Path) -> trimesh.Trimesh | None:
    """Load a mesh file, flattening scenes to a single Trimesh.

    Returns None if loading fails.
    """
    try:
        return meshbench.load(path)
    except (ValueError, Exception):
        logger.warning("Failed to load %s", path, exc_info=True)
        return None


def audit_mesh(path: Path) -> dict | None:
    """Load and audit a single mesh. Returns report dict with timing, or None on failure."""
    mesh = load_mesh(path)
    if mesh is None:
        return None
    try:
        t0 = time.perf_counter()
        report = meshbench.audit(mesh)
        elapsed = time.perf_counter() - t0
        d = report.to_dict()
        d["_audit_seconds"] = round(elapsed, 4)
        d["_file"] = str(path)
        return d
    except Exception:
        logger.warning("Audit failed for %s", path, exc_info=True)
        return None


def stem_name(path: Path) -> str:
    """Derive a clean mesh name from path (stem, no extension)."""
    return path.stem


def audit_directory(root: Path) -> dict[str, dict]:
    """Audit all meshes in a directory. Returns {name: report_dict}."""
    paths = discover_meshes(root)
    results: dict[str, dict] = {}
    for path in paths:
        name = stem_name(path)
        # Avoid name collisions by appending parent dir
        if name in results:
            name = f"{path.parent.name}_{name}"
        report = audit_mesh(path)
        if report is not None:
            results[name] = report
            logger.info("  %s — %.3fs", name, report["_audit_seconds"])
        else:
            logger.warning("  %s — SKIPPED", name)
    return results


def load_corpus(
    meshes_dir: Path | None = None,
    reference_dir: Path | None = None,
    generated_dir: Path | None = None,
) -> dict[str, dict[str, dict]]:
    """Load and audit a full corpus. Returns {source: {name: report_dict}}.

    Three modes:
    - reference_dir + generated_dir: explicit two-way split
    - meshes_dir with reference/ + generated/: legacy two-tier auto-detect
    - meshes_dir with flat subdirs: each subdir is a source (three-tier+)

    The flat-subdir mode supports layouts like:
        meshes/abc/          → source "abc"
        meshes/thingi10k/    → source "thingi10k"
        meshes/meshy/        → source "meshy"
    """
    sources: dict[str, dict[str, dict]] = {}

    if reference_dir and generated_dir:
        # Explicit mode
        if reference_dir.is_dir():
            logger.info("Auditing reference: %s", reference_dir)
            sources["reference"] = audit_directory(reference_dir)
        for subdir in sorted(generated_dir.iterdir()):
            if subdir.is_dir():
                logger.info("Auditing %s: %s", subdir.name, subdir)
                sources[subdir.name] = audit_directory(subdir)
    elif meshes_dir:
        ref = meshes_dir / "reference"
        gen = meshes_dir / "generated"

        if ref.is_dir() and gen.is_dir():
            # Legacy two-tier: reference/ + generated/<source>/
            logger.info("Auditing reference: %s", ref)
            sources["reference"] = audit_directory(ref)
            for subdir in sorted(gen.iterdir()):
                if subdir.is_dir():
                    logger.info("Auditing %s: %s", subdir.name, subdir)
                    sources[subdir.name] = audit_directory(subdir)
        else:
            # Flat multi-tier: each subdir is a source
            subdirs = sorted(d for d in meshes_dir.iterdir() if d.is_dir())
            if subdirs:
                for subdir in subdirs:
                    logger.info("Auditing %s: %s", subdir.name, subdir)
                    results = audit_directory(subdir)
                    if results:
                        sources[subdir.name] = results
            # If no subdirs or none had meshes, treat whole dir as one source
            if not sources:
                dir_name = meshes_dir.name or "meshes"
                logger.info("Auditing flat directory as '%s': %s", dir_name, meshes_dir)
                sources[dir_name] = audit_directory(meshes_dir)
    else:
        raise ValueError("Provide either --meshes or --reference + --generated")

    return sources
