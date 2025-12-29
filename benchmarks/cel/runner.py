#!/usr/bin/env python3
"""
CEL Benchmark Runner

Run CEL benchmarks from the command line.

Usage:
    python -m benchmarks.cel.runner --all
    python -m benchmarks.cel.runner --category throughput
    python -m benchmarks.cel.runner --benchmark event_append
    python -m benchmarks.cel.runner --list
    python -m benchmarks.cel.runner --all --output results.json
    python -m benchmarks.cel.runner --all --compare baseline.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from benchmarks.woven_mind.base import (
    BenchmarkCategory,
    BenchmarkResult,
    BenchmarkSuite,
    BenchmarkStatus,
)

from .benchmarks import (
    EventAppendBenchmark,
    MaterializationBenchmark,
    SemanticIndexBenchmark,
    TimeTravelBenchmark,
    DAGTraversalBenchmark,
    ContentAddressingBenchmark,
    CompactionBenchmark,
)

from .sanity_benchmarks import (
    ALL_SANITY_BENCHMARKS,
    SANITY_BENCHMARK_MAP,
    SANITY_BENCHMARKS_BY_CATEGORY,
    CompactionScalabilityBenchmark,
    SemanticClusteringBenchmark,
    CausalChainBenchmark,
    HealthMonitorLatencyBenchmark,
    HealthCheckAccuracyBenchmark,
    MigrationThroughputBenchmark,
    CompactionSavingsEstimationBenchmark,
)


# =============================================================================
# BENCHMARK REGISTRY
# =============================================================================

# Map category names to BenchmarkCategory enum
CATEGORY_MAP = {
    "throughput": BenchmarkCategory.SCALE,
    "memory": BenchmarkCategory.REGRESSION,  # Using REGRESSION for memory
    "query": BenchmarkCategory.QUALITY,       # Using QUALITY for query
    "correctness": BenchmarkCategory.STABILITY,  # Using STABILITY for correctness
}

# All available benchmarks (core + sanity)
ALL_BENCHMARKS = [
    # Core CEL benchmarks
    EventAppendBenchmark,
    MaterializationBenchmark,
    SemanticIndexBenchmark,
    TimeTravelBenchmark,
    DAGTraversalBenchmark,
    ContentAddressingBenchmark,
    CompactionBenchmark,
    # Sanity module benchmarks
    CompactionScalabilityBenchmark,
    SemanticClusteringBenchmark,
    CausalChainBenchmark,
    HealthMonitorLatencyBenchmark,
    HealthCheckAccuracyBenchmark,
    MigrationThroughputBenchmark,
    CompactionSavingsEstimationBenchmark,
]

# Benchmark name to class mapping
BENCHMARK_MAP = {b.name: b for b in ALL_BENCHMARKS}

# Category to benchmarks mapping
BENCHMARKS_BY_CATEGORY = {
    "throughput": [EventAppendBenchmark, MaterializationBenchmark],
    "memory": [CompactionBenchmark],
    "query": [SemanticIndexBenchmark, TimeTravelBenchmark, DAGTraversalBenchmark],
    "correctness": [ContentAddressingBenchmark],
    # Sanity module categories
    "compaction": [
        CompactionScalabilityBenchmark,
        SemanticClusteringBenchmark,
        CausalChainBenchmark,
        CompactionSavingsEstimationBenchmark,
    ],
    "health": [
        HealthMonitorLatencyBenchmark,
        HealthCheckAccuracyBenchmark,
    ],
    "migration": [
        MigrationThroughputBenchmark,
    ],
}


# =============================================================================
# RUNNER
# =============================================================================

def create_suite(
    benchmarks: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    config: Optional[Dict] = None,
) -> BenchmarkSuite:
    """Create a benchmark suite with specified benchmarks."""
    suite = BenchmarkSuite(name="cel_benchmarks")

    if benchmarks:
        # Specific benchmarks requested
        for name in benchmarks:
            if name in BENCHMARK_MAP:
                suite.add(BENCHMARK_MAP[name](config))
            else:
                print(f"Warning: Unknown benchmark '{name}'", file=sys.stderr)
    elif categories:
        # Specific categories requested
        for category in categories:
            if category in BENCHMARKS_BY_CATEGORY:
                for benchmark_cls in BENCHMARKS_BY_CATEGORY[category]:
                    suite.add(benchmark_cls(config))
            else:
                print(f"Warning: Unknown category '{category}'", file=sys.stderr)
    else:
        # All benchmarks
        for benchmark_cls in ALL_BENCHMARKS:
            suite.add(benchmark_cls(config))

    return suite


def progress_callback(name: str, current: int, total: int) -> None:
    """Print progress during benchmark execution."""
    percent = (current / total) * 100
    bar_width = 30
    filled = int(bar_width * current / total)
    bar = "=" * filled + "-" * (bar_width - filled)
    print(f"\r[{bar}] {percent:5.1f}% | Running: {name[:40]:<40}", end="", flush=True)


def compare_results(
    current: List[BenchmarkResult],
    baseline_path: Path,
) -> str:
    """Compare current results against a baseline."""
    baseline_data = json.loads(baseline_path.read_text())
    baseline_results = {r["benchmark_name"]: r for r in baseline_data.get("results", [])}

    lines = [
        "\n" + "=" * 60,
        "COMPARISON WITH BASELINE",
        f"Baseline: {baseline_path}",
        "=" * 60,
    ]

    for result in current:
        baseline = baseline_results.get(result.benchmark_name)
        if not baseline:
            lines.append(f"\n{result.benchmark_name}: NEW (no baseline)")
            continue

        lines.append(f"\n{result.benchmark_name}:")

        baseline_metrics = {m["name"]: m["value"] for m in baseline.get("metrics", [])}

        for metric in result.metrics:
            baseline_value = baseline_metrics.get(metric.name)
            if baseline_value is None:
                lines.append(f"  {metric.name}: {metric.value:.3f}{metric.unit} (NEW)")
            else:
                diff = metric.value - baseline_value
                diff_pct = (diff / baseline_value * 100) if baseline_value != 0 else float('inf')

                # Determine if change is good or bad based on thresholds
                if metric.threshold_max is not None and diff > 0:
                    status = "SLOWER" if "time" in metric.name or "latency" in metric.name else "WORSE"
                elif metric.threshold_min is not None and diff < 0:
                    status = "WORSE"
                elif diff == 0:
                    status = "SAME"
                else:
                    status = "BETTER" if diff < 0 else "CHANGED"

                symbol = "+" if diff > 0 else ""
                lines.append(
                    f"  {metric.name}: {metric.value:.3f}{metric.unit} "
                    f"({symbol}{diff_pct:.1f}% {status})"
                )

    return "\n".join(lines)


def list_benchmarks() -> None:
    """Print available benchmarks."""
    print("\nAvailable CEL Benchmarks:")
    print("=" * 60)

    for category, benchmarks in BENCHMARKS_BY_CATEGORY.items():
        print(f"\n{category.upper()}:")
        for benchmark_cls in benchmarks:
            print(f"  - {benchmark_cls.name}: {benchmark_cls.description}")

    print("\n\nUsage examples:")
    print("  python -m benchmarks.cel.runner --all")
    print("  python -m benchmarks.cel.runner --category throughput")
    print("  python -m benchmarks.cel.runner --benchmark event_append")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run CEL benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m benchmarks.cel.runner --all
  python -m benchmarks.cel.runner --category throughput
  python -m benchmarks.cel.runner --benchmark event_append
  python -m benchmarks.cel.runner --all --quick
  python -m benchmarks.cel.runner --all --output results.json
  python -m benchmarks.cel.runner --all --compare baseline.json
        """,
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmarks",
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=list(BENCHMARKS_BY_CATEGORY.keys()),
        help="Run benchmarks in a specific category",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        help="Run a specific benchmark by name",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available benchmarks",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run with reduced iterations for quick feedback",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--compare",
        type=str,
        help="Compare against baseline JSON file",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output",
    )

    args = parser.parse_args()

    if args.list:
        list_benchmarks()
        return 0

    if not (args.all or args.category or args.benchmark):
        parser.print_help()
        return 1

    # Build configuration
    config = {}
    if args.quick:
        config = {
            "n_events": 100,
            "n_warmup": 10,
            "n_entities": 10,
            "n_queries": 50,
            "n_concepts": 50,
            "events_per_entity": 10,
        }

    # Create suite
    benchmarks = [args.benchmark] if args.benchmark else None
    categories = [args.category] if args.category else None
    suite = create_suite(benchmarks=benchmarks, categories=categories, config=config)

    if not suite.benchmarks:
        print("No benchmarks to run.", file=sys.stderr)
        return 1

    # Run benchmarks
    print(f"\nRunning {len(suite.benchmarks)} CEL benchmark(s)...")
    print("=" * 60)

    results = suite.run_all(progress_callback=progress_callback)
    print()  # Newline after progress bar

    # Print summary
    print(suite.summary())

    # Compare if baseline provided
    if args.compare:
        baseline_path = Path(args.compare)
        if baseline_path.exists():
            print(compare_results(results, baseline_path))
        else:
            print(f"\nWarning: Baseline file not found: {args.compare}", file=sys.stderr)

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        suite.save_results(output_path)
        print(f"\nResults saved to: {output_path}")

    # Return exit code based on results
    all_passed = all(r.status == BenchmarkStatus.PASSED for r in results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
