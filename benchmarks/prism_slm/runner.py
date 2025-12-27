#!/usr/bin/env python3
"""
PRISM-SLM Benchmark Runner

Usage:
    python -m benchmarks.prism_slm.runner --all
    python -m benchmarks.prism_slm.runner --category generation
    python -m benchmarks.prism_slm.runner --benchmark hebbian_strengthening
    python -m benchmarks.prism_slm.runner --list
    python -m benchmarks.prism_slm.runner --quick

Options:
    --all               Run all benchmarks
    --category CAT      Run benchmarks in category (generation, learning, integration)
    --benchmark NAME    Run specific benchmark by name
    --list              List all available benchmarks
    --output FILE       Save results to JSON file
    --compare FILE      Compare against previous results
    --verbose           Show detailed progress
    --quick             Run quick versions (fewer iterations)
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Type

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.woven_mind.base import (
    BaseBenchmark,
    BenchmarkCategory,
    BenchmarkResult,
    BenchmarkStatus,
    BenchmarkSuite,
)
from benchmarks.prism_slm.generation import (
    GenerationCoherenceBenchmark,
    PerplexityCalibrationBenchmark,
    TemperatureDiversityBenchmark,
    FullCorpusPerplexityBenchmark,
    VariedCorpusDiversityBenchmark,
)
from benchmarks.prism_slm.learning import (
    HebbianStrengtheningBenchmark,
    DecayStabilityBenchmark,
    RewardLearningBenchmark,
)
from benchmarks.prism_slm.integration import (
    SpreadingActivationBenchmark,
    LateralInhibitionBenchmark,
    SparsecodingEfficiencyBenchmark,
)


# Registry of all benchmarks
ALL_BENCHMARKS: List[Type[BaseBenchmark]] = [
    # Generation (small corpus)
    GenerationCoherenceBenchmark,
    PerplexityCalibrationBenchmark,
    TemperatureDiversityBenchmark,
    # Generation (full corpus - Option A)
    FullCorpusPerplexityBenchmark,
    # Generation (varied corpus - Option B)
    VariedCorpusDiversityBenchmark,
    # Learning
    HebbianStrengtheningBenchmark,
    DecayStabilityBenchmark,
    RewardLearningBenchmark,
    # Integration
    SpreadingActivationBenchmark,
    LateralInhibitionBenchmark,
    SparsecodingEfficiencyBenchmark,
]

# Category mapping
CATEGORY_MAP = {
    "generation": BenchmarkCategory.QUALITY,
    "learning": BenchmarkCategory.STABILITY,
    "integration": BenchmarkCategory.COGNITIVE,
}


def list_benchmarks() -> None:
    """Print list of available benchmarks."""
    print("\nAvailable PRISM-SLM Benchmarks:")
    print("=" * 60)

    by_category = {}
    for bench_class in ALL_BENCHMARKS:
        cat = bench_class.category.value
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(bench_class)

    for cat, benchmarks in sorted(by_category.items()):
        print(f"\n{cat.upper()}:")
        for bench_class in benchmarks:
            print(f"  - {bench_class.name}: {bench_class.description}")

    print()


def run_benchmarks(
    benchmarks: List[Type[BaseBenchmark]],
    quick: bool = False,
    verbose: bool = False,
) -> List[BenchmarkResult]:
    """Run a list of benchmarks."""
    results = []

    print("\n" + "=" * 60)
    print("PRISM-SLM Benchmark Suite")
    print("=" * 60)
    print(f"Running {len(benchmarks)} benchmark(s)")
    print(f"Mode: {'quick' if quick else 'full'}")
    print()

    for bench_class in benchmarks:
        bench = bench_class(quick=quick)

        if verbose:
            print(f"Setting up {bench.name}...")

        bench.setup()

        print(f"Running: {bench.name}...", end=" ", flush=True)

        try:
            result = bench.run()
            results.append(result)

            status_symbol = "PASS" if result.status == BenchmarkStatus.PASSED else "FAIL"
            print(f"[{status_symbol}] ({result.duration_ms:.1f}ms)")

            if verbose:
                for metric in result.metrics:
                    status = "OK" if metric.is_passing else "FAIL"
                    print(f"    {metric.name}: {metric.value:.4f} {metric.unit} [{status}]")

        except Exception as e:
            print(f"[ERROR] {e}")
            results.append(BenchmarkResult(
                benchmark_name=bench.name,
                category=bench.category,
                status=BenchmarkStatus.ERROR,
                metrics=[],
                duration_ms=0,
                error_message=str(e),
            ))

        finally:
            bench.teardown()

    return results


def print_summary(results: List[BenchmarkResult]) -> None:
    """Print benchmark summary."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results if r.status == BenchmarkStatus.PASSED)
    failed = sum(1 for r in results if r.status == BenchmarkStatus.FAILED)
    errors = sum(1 for r in results if r.status == BenchmarkStatus.ERROR)

    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Errors: {errors}")
    print(f"  Total:  {len(results)}")

    total_duration = sum(r.duration_ms for r in results)
    print(f"\n  Total Duration: {total_duration:.1f}ms")

    if failed > 0:
        print("\nFailed Benchmarks:")
        for r in results:
            if r.status == BenchmarkStatus.FAILED:
                print(f"  - {r.benchmark_name}")
                for m in r.metrics:
                    if not m.is_passing:
                        print(f"      {m.name}: {m.value:.4f} (threshold: {m.threshold_min}-{m.threshold_max})")

    print()


def save_results(results: List[BenchmarkResult], output_path: str) -> None:
    """Save results to JSON file."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "benchmarks": [r.to_dict() for r in results],
        "summary": {
            "passed": sum(1 for r in results if r.status == BenchmarkStatus.PASSED),
            "failed": sum(1 for r in results if r.status == BenchmarkStatus.FAILED),
            "errors": sum(1 for r in results if r.status == BenchmarkStatus.ERROR),
            "total_duration_ms": sum(r.duration_ms for r in results),
        },
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"Results saved to: {output_path}")


def compare_results(results: List[BenchmarkResult], compare_path: str) -> None:
    """Compare current results against previous results."""
    with open(compare_path) as f:
        previous = json.load(f)

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    prev_by_name = {b["benchmark_name"]: b for b in previous.get("benchmarks", [])}

    for result in results:
        prev = prev_by_name.get(result.benchmark_name)
        if not prev:
            print(f"{result.benchmark_name}: NEW")
            continue

        # Compare status
        prev_status = prev.get("status", "unknown")
        curr_status = result.status.value

        if curr_status != prev_status:
            print(f"{result.benchmark_name}: {prev_status} -> {curr_status}")
        else:
            # Compare metrics
            prev_metrics = {m["name"]: m["value"] for m in prev.get("metrics", [])}
            for metric in result.metrics:
                prev_val = prev_metrics.get(metric.name)
                if prev_val is not None:
                    change = ((metric.value - prev_val) / prev_val * 100) if prev_val != 0 else 0
                    if abs(change) > 5:  # >5% change
                        direction = "+" if change > 0 else ""
                        print(f"  {result.benchmark_name}.{metric.name}: {direction}{change:.1f}%")

    print()


def main():
    parser = argparse.ArgumentParser(description="PRISM-SLM Benchmark Runner")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--category", type=str, help="Run benchmarks in category")
    parser.add_argument("--benchmark", type=str, help="Run specific benchmark")
    parser.add_argument("--list", action="store_true", help="List all benchmarks")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    parser.add_argument("--compare", type=str, help="Compare against previous results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quick", "-q", action="store_true", help="Quick mode")

    args = parser.parse_args()

    if args.list:
        list_benchmarks()
        return

    # Determine which benchmarks to run
    benchmarks_to_run = []

    if args.all:
        benchmarks_to_run = ALL_BENCHMARKS
    elif args.category:
        target_cat = CATEGORY_MAP.get(args.category.lower())
        if target_cat is None:
            print(f"Unknown category: {args.category}")
            print(f"Available: {list(CATEGORY_MAP.keys())}")
            sys.exit(1)
        benchmarks_to_run = [b for b in ALL_BENCHMARKS if b.category == target_cat]
    elif args.benchmark:
        for b in ALL_BENCHMARKS:
            if b.name == args.benchmark:
                benchmarks_to_run = [b]
                break
        if not benchmarks_to_run:
            print(f"Unknown benchmark: {args.benchmark}")
            list_benchmarks()
            sys.exit(1)
    else:
        # Default: run all
        benchmarks_to_run = ALL_BENCHMARKS

    # Run benchmarks
    results = run_benchmarks(
        benchmarks_to_run,
        quick=args.quick,
        verbose=args.verbose,
    )

    # Print summary
    print_summary(results)

    # Save results if requested
    if args.output:
        save_results(results, args.output)

    # Compare if requested
    if args.compare:
        compare_results(results, args.compare)

    # Exit with error if any failed
    if any(r.status in (BenchmarkStatus.FAILED, BenchmarkStatus.ERROR) for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
