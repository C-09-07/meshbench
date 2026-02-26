"""Generate benchmark comparison charts from JSON report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Consistent source colors
SOURCE_COLORS = {
    "abc": "#2196F3",
    "khronos": "#4CAF50",
    "thingi10k": "#FF9800",
    # AI generators
    "meshy": "#E91E63",
    "tripo": "#9C27B0",
    "luma": "#00BCD4",
    "rodin": "#F44336",
    "instantmesh": "#795548",
    "sf3d": "#8BC34A",
    "trellis2": "#3F51B5",
    "hunyuan3d": "#FF5722",
    "triposg": "#9C27B0",
    "customuse": "#CDDC39",
}

def _color(name: str) -> str:
    return SOURCE_COLORS.get(name, "#607D8B")


def load_report(path: Path) -> dict:
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# 1. Radar chart — aggregate metrics normalized to [0, 1]
# ---------------------------------------------------------------------------

_RADAR_METRICS = [
    ("watertight_ratio", "Watertight", False),
    ("mean_winding_consistency", "Winding", False),
    ("mean_backface_divergence", "Backface Div", True),   # lower is better → invert
    ("mean_genus", "Genus", True),
    ("mean_valence_std", "Valence Std", True),
    ("mean_degenerate_faces", "Degen Faces", True),
    ("mean_face_density_variance", "FDV", True),
    ("mean_non_manifold_edges", "NM Edges", True),
]


def plot_radar(sources: dict, ax: plt.Axes) -> None:
    labels = [m[1] for m in _RADAR_METRICS]
    n_metrics = len(labels)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    # Collect raw values per source
    raw: dict[str, list[float]] = {}
    for name, src in sources.items():
        agg = src["aggregate"]
        raw[name] = [agg.get(m[0], 0.0) for m in _RADAR_METRICS]

    # Normalize each metric to [0, 1] across all sources
    all_vals = np.array(list(raw.values()))  # (n_sources, n_metrics)
    mins = all_vals.min(axis=0)
    maxs = all_vals.max(axis=0)
    ranges = np.where(maxs - mins > 1e-12, maxs - mins, 1.0)

    for name, vals in raw.items():
        normed = (np.array(vals) - mins) / ranges
        # Invert "lower is better" metrics so outward = better
        for i, (_, _, invert) in enumerate(_RADAR_METRICS):
            if invert:
                normed[i] = 1.0 - normed[i]
        normed = normed.tolist() + normed[:1].tolist()
        ax.plot(angles, normed, "o-", label=name, color=_color(name), linewidth=2, markersize=4)
        ax.fill(angles, normed, alpha=0.08, color=_color(name))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=8)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "", "", ""], size=7)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    ax.set_title("Quality Profile (outward = better)", size=11, pad=20)


# ---------------------------------------------------------------------------
# 2. Box plots — per-mesh metric distributions
# ---------------------------------------------------------------------------

_BOX_METRICS = [
    ("normals", "winding_consistency", "Winding Consistency"),
    ("normals", "backface_divergence", "Backface Divergence"),
    ("density", "face_density_variance", "Face Density Variance"),
    ("topology", "genus", "Genus"),
    ("topology", "valence_std", "Valence Std Dev"),
    ("topology", "degenerate_face_count", "Degenerate Faces"),
]


def _extract_per_mesh(sources: dict, section: str, field: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for name, src in sources.items():
        vals = []
        for mesh in src["meshes"].values():
            try:
                vals.append(float(mesh[section][field]))
            except (KeyError, TypeError):
                pass
        out[name] = vals
    return out


def plot_boxes(sources: dict, axes: list[plt.Axes]) -> None:
    source_names = list(sources.keys())
    colors = [_color(n) for n in source_names]

    for ax, (section, field, title) in zip(axes, _BOX_METRICS):
        data = _extract_per_mesh(sources, section, field)
        box_data = [data.get(n, []) for n in source_names]

        bp = ax.boxplot(
            box_data,
            tick_labels=source_names,
            patch_artist=True,
            widths=0.6,
            medianprops=dict(color="black", linewidth=1.5),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_title(title, size=9)
        ax.tick_params(axis="x", labelsize=7, rotation=30)
        ax.tick_params(axis="y", labelsize=7)


# ---------------------------------------------------------------------------
# 3. Failure mode frequency — grouped horizontal bars
# ---------------------------------------------------------------------------

def plot_failure_bars(sources: dict, ax: plt.Axes) -> None:
    # Collect all failure codes across sources
    all_codes: set[str] = set()
    for src in sources.values():
        all_codes.update(src.get("failure_summary", {}).keys())

    # Sort by total frequency
    code_totals = {}
    for code in all_codes:
        total = sum(
            src.get("failure_summary", {}).get(code, {}).get("ratio", 0)
            for src in sources.values()
        )
        code_totals[code] = total
    codes = sorted(code_totals, key=code_totals.get, reverse=True)[:12]

    source_names = list(sources.keys())
    n_sources = len(source_names)
    y = np.arange(len(codes))
    bar_height = 0.8 / n_sources

    for i, name in enumerate(source_names):
        fail = sources[name].get("failure_summary", {})
        ratios = [fail.get(code, {}).get("ratio", 0) for code in codes]
        ax.barh(
            y + i * bar_height,
            ratios,
            height=bar_height,
            label=name,
            color=_color(name),
            alpha=0.8,
        )

    ax.set_yticks(y + bar_height * (n_sources - 1) / 2)
    ax.set_yticklabels([c.replace("_", " ") for c in codes], size=7)
    ax.set_xlabel("Ratio of meshes affected", size=9)
    ax.set_xlim(0, 1.05)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Failure Mode Frequency by Source", size=11)
    ax.invert_yaxis()


# ---------------------------------------------------------------------------
# 4. Scatter: winding vs FDV colored by source
# ---------------------------------------------------------------------------

def plot_scatter(sources: dict, ax: plt.Axes) -> None:
    for name, src in sources.items():
        winding = []
        fdv = []
        for mesh in src["meshes"].values():
            try:
                w = mesh["normals"]["winding_consistency"]
                f = mesh["density"]["face_density_variance"]
                winding.append(w)
                fdv.append(f)
            except (KeyError, TypeError):
                pass
        ax.scatter(
            winding, fdv,
            label=f"{name} ({len(winding)})",
            color=_color(name),
            alpha=0.5,
            s=15,
            edgecolors="none",
        )

    ax.set_xlabel("Winding Consistency", size=9)
    ax.set_ylabel("Face Density Variance (log)", size=9)
    ax.set_yscale("log")
    ax.legend(fontsize=7, markerscale=2)
    ax.set_title("Winding Consistency vs Density Variance", size=11)
    ax.tick_params(labelsize=7)


# ---------------------------------------------------------------------------
# 5. Watertight / manifold status stacked bar
# ---------------------------------------------------------------------------

def plot_manifold_status(sources: dict, ax: plt.Axes) -> None:
    source_names = list(sources.keys())
    x = np.arange(len(source_names))

    watertight = []
    open_only = []  # has boundary but manifold edges
    non_manifold = []

    for name in source_names:
        meshes = list(sources[name]["meshes"].values())
        n = len(meshes)
        wt = sum(1 for m in meshes if m["manifold"]["is_watertight"])
        nm = sum(1 for m in meshes if m["manifold"]["non_manifold_edge_count"] > 0)
        op = n - wt - nm
        if op < 0:
            op = 0
        watertight.append(wt / n if n else 0)
        non_manifold.append(nm / n if n else 0)
        open_only.append(op / n if n else 0)

    ax.bar(x, watertight, label="Watertight", color="#4CAF50", alpha=0.8)
    ax.bar(x, open_only, bottom=watertight, label="Open boundary", color="#FF9800", alpha=0.8)
    bottoms = [w + o for w, o in zip(watertight, open_only)]
    ax.bar(x, non_manifold, bottom=bottoms, label="Non-manifold", color="#F44336", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(source_names, size=8)
    ax.set_ylabel("Fraction", size=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("Manifold Status by Source", size=11)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_charts(report_path: Path, output_dir: Path) -> list[Path]:
    data = load_report(report_path)
    sources = data["sources"]
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFAFA",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.family": "sans-serif",
    })

    # Fig 1: Radar
    fig, ax = plt.subplots(1, 1, figsize=(7, 7), subplot_kw=dict(polar=True))
    plot_radar(sources, ax)
    fig.tight_layout()
    p = output_dir / "radar_quality_profile.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    saved.append(p)
    plt.close(fig)

    # Fig 2: Box plots
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    plot_boxes(sources, axes.flatten().tolist())
    fig.suptitle("Per-Mesh Metric Distributions", size=13, y=1.02)
    fig.tight_layout()
    p = output_dir / "boxplots_distributions.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    saved.append(p)
    plt.close(fig)

    # Fig 3: Failure bars
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    plot_failure_bars(sources, ax)
    fig.tight_layout()
    p = output_dir / "failure_mode_frequency.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    saved.append(p)
    plt.close(fig)

    # Fig 4: Scatter
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    plot_scatter(sources, ax)
    fig.tight_layout()
    p = output_dir / "scatter_winding_vs_fdv.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    saved.append(p)
    plt.close(fig)

    # Fig 5: Manifold status
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    plot_manifold_status(sources, ax)
    fig.tight_layout()
    p = output_dir / "manifold_status.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    saved.append(p)
    plt.close(fig)

    return saved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark charts from JSON report")
    parser.add_argument("report", type=Path, help="Path to benchmark JSON report")
    parser.add_argument("--output", type=Path, default=Path("./output/charts"), help="Output directory for PNGs")
    args = parser.parse_args(argv)

    saved = generate_charts(args.report, args.output)
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()
