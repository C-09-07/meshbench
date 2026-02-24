"""CLI entry point: python benchmarks/run.py ..."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from benchmarks.aggregate import AggregateStats, compute_aggregate, compute_delta
from benchmarks.corpus import load_corpus
from benchmarks.failmodes import classify_failures, source_failure_summary
from benchmarks.report import build_full_report, format_table, write_json

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)


def build_source_entry(
    reports: dict[str, dict],
    reference_agg: AggregateStats | None = None,
) -> dict:
    """Build a source dict with meshes, aggregate, failure summary, and optional delta."""
    agg = compute_aggregate(reports)

    # Inject per-mesh failure fingerprints
    for name, report in reports.items():
        report["_failure_fingerprint"] = classify_failures(report).to_dict()

    entry: dict = {
        "mesh_count": agg.mesh_count,
        "meshes": reports,
        "aggregate": asdict(agg),
        "failure_summary": source_failure_summary(reports),
    }
    if reference_agg is not None:
        entry["delta"] = compute_delta(agg, reference_agg)
    return entry


def _detect_reference(
    sources: dict[str, dict[str, dict]],
    reference_name: str | None,
) -> str | None:
    """Determine which source is the reference for delta computation.

    Priority: explicit --reference-name > "reference" if present > None.
    """
    if reference_name:
        if reference_name in sources:
            return reference_name
        logger.warning("--reference-name '%s' not found in sources, skipping deltas", reference_name)
        return None
    if "reference" in sources:
        return "reference"
    return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="meshbench benchmark runner — audit mesh directories and compare quality",
    )
    parser.add_argument(
        "--meshes",
        type=Path,
        help="Mesh directory: each subdir becomes a source (e.g. abc/, thingi10k/, meshy/)",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="Reference mesh directory (explicit two-way mode)",
    )
    parser.add_argument(
        "--generated",
        type=Path,
        help="Generated mesh directory (subdirs become source tags)",
    )
    parser.add_argument(
        "--reference-name",
        type=str,
        default=None,
        help="Which source to use as reference for delta computation (default: 'reference' if present)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./output"),
        help="Output directory for JSON reports (default: ./output)",
    )
    parser.add_argument(
        "--format",
        dest="fmt",
        choices=["json", "table", "both"],
        default="json",
        help="Output format (default: json)",
    )

    args = parser.parse_args(argv)

    if not args.meshes and not (args.reference and args.generated):
        parser.error("Provide either --meshes or both --reference and --generated")

    # Load and audit
    raw_sources = load_corpus(
        meshes_dir=args.meshes,
        reference_dir=args.reference,
        generated_dir=args.generated,
    )

    if not raw_sources:
        logger.error("No meshes found.")
        sys.exit(1)

    total = sum(len(v) for v in raw_sources.values())
    logger.info("Audited %d meshes across %d sources", total, len(raw_sources))

    # Determine reference source for deltas
    ref_name = _detect_reference(raw_sources, args.reference_name)
    reference_agg = None
    if ref_name:
        reference_agg = compute_aggregate(raw_sources[ref_name])
        logger.info("Using '%s' as reference for delta computation", ref_name)

    sources: dict[str, dict] = {}
    for name, reports in raw_sources.items():
        ref = None if name == ref_name else reference_agg
        sources[name] = build_source_entry(reports, reference_agg=ref)

    report = build_full_report(sources, reference_name=ref_name)

    # Output
    if args.fmt in ("json", "both"):
        path = write_json(report, args.output)
        logger.info("Report written to %s", path)

    if args.fmt in ("table", "both"):
        print()
        print(format_table(sources, reference_name=ref_name))
        print()


if __name__ == "__main__":
    main()
