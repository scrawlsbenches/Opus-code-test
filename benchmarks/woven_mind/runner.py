#!/usr/bin/env python3
"""
Woven Mind Benchmark Runner

Usage:
    python -m benchmarks.woven_mind.runner --all
    python -m benchmarks.woven_mind.runner --category stability
    python -m benchmarks.woven_mind.runner --benchmark parameter_sensitivity
    python -m benchmarks.woven_mind.runner --list
    python -m benchmarks.woven_mind.runner --compare results/baseline.json

Options:
    --all               Run all benchmarks
    --category CAT      Run benchmarks in category (stability, quality, scale, cognitive)
    --benchmark NAME    Run specific benchmark by name
    --list              List all available benchmarks
    --output FILE       Save results to JSON file
    --compare FILE      Compare against previous results
    --verbose           Show detailed progress
    --quick             Run quick versions (smaller corpora, fewer iterations)
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Add project root to path for imports
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.woven_mind.base import (
    BaseBenchmark,
    BenchmarkCategory,
    BenchmarkResult,
    BenchmarkSuite,
    BenchmarkStatus,
)
from benchmarks.woven_mind.stability import (
    ParameterSensitivityBenchmark,
    BaselineDriftBenchmark,
    HomeostasisStabilityBenchmark,
)
from benchmarks.woven_mind.quality import (
    AbstractionQualityBenchmark,
    ModeSwitchingBenchmark,
    RetrievalRelevanceBenchmark,
)
from benchmarks.woven_mind.scale import (
    ScalabilityBenchmark,
    ColdStartBenchmark,
    MemoryUsageBenchmark,
)
from benchmarks.woven_mind.cognitive import (
    SurpriseCalibrationBenchmark,
    HomeostasisInteractionBenchmark,
    DualProcessCoherenceBenchmark,
)


# Registry of all benchmarks
ALL_BENCHMARKS = [
    # Stability
    ParameterSensitivityBenchmark,
    BaselineDriftBenchmark,
    HomeostasisStabilityBenchmark,
    # Quality
    AbstractionQualityBenchmark,
    ModeSwitchingBenchmark,
    RetrievalRelevanceBenchmark,
    # Scale
    ScalabilityBenchmark,
    ColdStartBenchmark,
    MemoryUsageBenchmark,
    # Cognitive
    SurpriseCalibrationBenchmark,
    HomeostasisInteractionBenchmark,
    DualProcessCoherenceBenchmark,
]


def create_suite(
    benchmarks: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    quick: bool = False,
) -> BenchmarkSuite:
    """Create a benchmark suite with specified filters."""
    suite = BenchmarkSuite(name="woven_mind_benchmarks")

    # Quick mode config
    quick_config = {
        "n_docs": 50,
        "doc_length": 30,
        "corpus_sizes": [50, 100, 200],
    } if quick else {}

    for benchmark_class in ALL_BENCHMARKS:
        # Filter by name
        if benchmarks and benchmark_class.name not in benchmarks:
            continue

        # Filter by category
        if categories:
            cat_values = [BenchmarkCategory[c.upper()] for c in categories]
            if benchmark_class.category not in cat_values:
                continue

        suite.add(benchmark_class(config=quick_config))

    return suite


def list_benchmarks() -> None:
    """List all available benchmarks."""
    print("\nAvailable Benchmarks:")
    print("=" * 60)

    by_category = {}
    for benchmark_class in ALL_BENCHMARKS:
        cat = benchmark_class.category.value
        by_category.setdefault(cat, []).append(benchmark_class)

    for category in BenchmarkCategory:
        if category.value not in by_category:
            continue

        print(f"\n{category.value.upper()}:")
        print("-" * 40)
        for benchmark_class in by_category[category.value]:
            print(f"  {benchmark_class.name}")
            print(f"    {benchmark_class.description}")


def compare_results(current: List[BenchmarkResult], baseline_path: Path) -> None:
    """Compare current results against baseline."""
    try:
        baseline_data = json.loads(baseline_path.read_text())
        baseline_results = {r["benchmark_name"]: r for r in baseline_data["results"]}
    except Exception as e:
        print(f"Error loading baseline: {e}")
        return

    print("\nComparison with Baseline:")
    print("=" * 70)

    for result in current:
        baseline = baseline_results.get(result.benchmark_name)
        if not baseline:
            print(f"\n{result.benchmark_name}: NEW (no baseline)")
            continue

        print(f"\n{result.benchmark_name}:")

        # Compare metrics
        baseline_metrics = {m["name"]: m for m in baseline["metrics"]}

        for metric in result.metrics:
            baseline_metric = baseline_metrics.get(metric.name)
            if not baseline_metric:
                print(f"  {metric.name}: {metric.value:.3f} (NEW)")
                continue

            old_value = baseline_metric["value"]
            new_value = metric.value
            delta = new_value - old_value
            pct_change = (delta / old_value * 100) if old_value != 0 else float('inf')

            # Determine if change is good or bad
            # For most metrics, check thresholds
            is_improvement = True
            if metric.threshold_min is not None and new_value < metric.threshold_min:
                is_improvement = False
            if metric.threshold_max is not None and new_value > metric.threshold_max:
                is_improvement = False

            if abs(pct_change) < 5:
                indicator = "~"  # No significant change
            elif is_improvement:
                indicator = "+" if delta > 0 else "-"
            else:
                indicator = "!" if abs(pct_change) > 20 else "?"

            print(f"  {metric.name}: {old_value:.3f} -> {new_value:.3f} ({indicator}{pct_change:+.1f}%)")


def progress_callback(name: str, current: int, total: int) -> None:
    """Print progress during benchmark execution."""
    pct = current / total * 100
    bar_width = 30
    filled = int(bar_width * current / total)
    bar = "=" * filled + "-" * (bar_width - filled)
    print(f"\r[{bar}] {pct:5.1f}% - {name}", end="", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run Woven Mind benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--category", type=str, nargs="+", help="Run benchmarks in category")
    parser.add_argument("--benchmark", type=str, nargs="+", help="Run specific benchmark(s)")
    parser.add_argument("--list", action="store_true", help="List available benchmarks")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    parser.add_argument("--compare", type=str, help="Compare against previous results")
    parser.add_argument("--verbose", action="store_true", help="Show detailed progress")
    parser.add_argument("--quick", action="store_true", help="Run quick versions")

    args = parser.parse_args()

    if args.list:
        list_benchmarks()
        return 0

    if not (args.all or args.category or args.benchmark):
        parser.print_help()
        print("\nError: Must specify --all, --category, or --benchmark")
        return 1

    # Create suite
    suite = create_suite(
        benchmarks=args.benchmark,
        categories=args.category,
        quick=args.quick,
    )

    if len(suite.benchmarks) == 0:
        print("No benchmarks matched the criteria")
        return 1

    print(f"\nRunning {len(suite.benchmarks)} benchmark(s)...")
    print("=" * 60)

    # Run benchmarks
    callback = progress_callback if args.verbose else None
    results = suite.run_all(progress_callback=callback)

    if args.verbose:
        print()  # Clear progress line

    # Print summary
    print(suite.summary())

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        suite.save_results(output_path)
        print(f"\nResults saved to: {output_path}")

    # Compare if requested
    if args.compare:
        compare_results(results, Path(args.compare))

    # Return exit code based on results
    failed = sum(1 for r in results if r.status in (BenchmarkStatus.FAILED, BenchmarkStatus.ERROR))
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
