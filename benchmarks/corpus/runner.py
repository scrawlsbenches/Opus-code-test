#!/usr/bin/env python3
"""
Corpus Benchmark Runner

Run CorticalTextProcessor benchmarks from the command line.

Usage:
    python -m benchmarks.corpus.runner --all
    python -m benchmarks.corpus.runner --category indexing
    python -m benchmarks.corpus.runner --benchmark indexing_throughput
    python -m benchmarks.corpus.runner --list
    python -m benchmarks.corpus.runner --all --quick
    python -m benchmarks.corpus.runner --all --output results.json
    python -m benchmarks.corpus.runner --all --compare baseline.json

Categories:
    indexing    - Document processing throughput
    query       - Search latency and relevance
    passage     - RAG/passage retrieval
    analysis    - PageRank, TF-IDF, clustering
    code_search - Code-specific search features
    fingerprint - Semantic fingerprinting
    persistence - Save/load operations
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Type

# Add project root to path for imports
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.woven_mind.base import (
    BaseBenchmark,
    BenchmarkResult,
    BenchmarkSuite,
    BenchmarkStatus,
)

from benchmarks.corpus.base import (
    CorpusBenchmark,
    CorpusBenchmarkCategory,
    CorpusCache,
)


# =============================================================================
# BENCHMARK REGISTRY
# =============================================================================

# All benchmark classes will be registered here
# (populated as benchmark modules are created)
ALL_BENCHMARKS: List[Type[CorpusBenchmark]] = []

# Benchmark name to class mapping
BENCHMARK_MAP: Dict[str, Type[CorpusBenchmark]] = {}

# Category to benchmarks mapping
BENCHMARKS_BY_CATEGORY: Dict[str, List[Type[CorpusBenchmark]]] = {
    "indexing": [],
    "query": [],
    "passage": [],
    "analysis": [],
    "code_search": [],
    "fingerprint": [],
    "persistence": [],
}


def register_benchmark(cls: Type[CorpusBenchmark]) -> Type[CorpusBenchmark]:
    """Decorator to register a benchmark class."""
    ALL_BENCHMARKS.append(cls)
    BENCHMARK_MAP[cls.name] = cls

    # Add to category
    category_name = cls.corpus_category.value
    if category_name in BENCHMARKS_BY_CATEGORY:
        BENCHMARKS_BY_CATEGORY[category_name].append(cls)

    return cls


# =============================================================================
# PLACEHOLDER BENCHMARKS (to be replaced with real implementations)
# =============================================================================

@register_benchmark
class IndexingThroughputBenchmark(CorpusBenchmark):
    """Placeholder for indexing throughput benchmark."""

    name = "indexing_throughput"
    description = "Measure document indexing throughput (docs/sec)"
    corpus_category = CorpusBenchmarkCategory.INDEXING

    def run(self) -> BenchmarkResult:
        result = self.create_result()
        # Note: Real implementation will measure actual throughput
        # For now, skip without adding metrics that have thresholds
        result.status = BenchmarkStatus.SKIPPED
        result.error_message = "Placeholder - run T-20251229-101547-acbfaa78 to implement"
        return result


# =============================================================================
# RUNNER
# =============================================================================

def create_suite(
    benchmarks: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    config: Optional[Dict] = None,
) -> BenchmarkSuite:
    """Create a benchmark suite with specified benchmarks."""
    suite = BenchmarkSuite(name="corpus_benchmarks")

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
    try:
        baseline_data = json.loads(baseline_path.read_text())
        baseline_results = {r["benchmark_name"]: r for r in baseline_data.get("results", [])}
    except Exception as e:
        return f"Error loading baseline: {e}"

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

                # Determine if change is good or bad
                if "latency" in metric.name or "time" in metric.name or "_ms" in metric.name:
                    # Lower is better for latency
                    status = "FASTER" if diff < 0 else "SLOWER"
                elif "throughput" in metric.name or "_per_" in metric.name:
                    # Higher is better for throughput
                    status = "BETTER" if diff > 0 else "WORSE"
                else:
                    status = "CHANGED"

                symbol = "+" if diff > 0 else ""
                lines.append(
                    f"  {metric.name}: {metric.value:.3f}{metric.unit} "
                    f"({symbol}{diff_pct:.1f}% {status})"
                )

    # Add cache stats
    cache = CorpusCache()
    stats = cache.stats()
    lines.extend([
        "\n" + "-" * 40,
        f"Cache: {stats['hits']} hits, {stats['misses']} misses",
    ])

    return "\n".join(lines)


def list_benchmarks() -> None:
    """Print available benchmarks."""
    print("\nAvailable Corpus Benchmarks:")
    print("=" * 60)

    for category in CorpusBenchmarkCategory:
        category_name = category.value
        benchmarks = BENCHMARKS_BY_CATEGORY.get(category_name, [])
        if not benchmarks:
            print(f"\n{category_name.upper()}: (no benchmarks yet)")
            continue

        print(f"\n{category_name.upper()}:")
        for benchmark_cls in benchmarks:
            print(f"  - {benchmark_cls.name}: {benchmark_cls.description}")

    print("\n\nUsage examples:")
    print("  python -m benchmarks.corpus.runner --all")
    print("  python -m benchmarks.corpus.runner --category indexing")
    print("  python -m benchmarks.corpus.runner --benchmark indexing_throughput")
    print("  python -m benchmarks.corpus.runner --all --quick")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Corpus benchmarks for CorticalTextProcessor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m benchmarks.corpus.runner --all
  python -m benchmarks.corpus.runner --category indexing
  python -m benchmarks.corpus.runner --benchmark indexing_throughput
  python -m benchmarks.corpus.runner --all --quick
  python -m benchmarks.corpus.runner --all --output results.json
  python -m benchmarks.corpus.runner --all --compare baseline.json

Categories:
  indexing    - Document processing throughput
  query       - Search latency and relevance
  passage     - RAG/passage retrieval
  analysis    - PageRank, TF-IDF, clustering
  code_search - Code-specific search features
  fingerprint - Semantic fingerprinting
  persistence - Save/load operations
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
        help="Run with smaller corpus for quick feedback",
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
    parser.add_argument(
        "--corpus-size",
        type=int,
        default=None,
        help="Override corpus size (n_docs)",
    )

    args = parser.parse_args()

    if args.list:
        list_benchmarks()
        return 0

    if not (args.all or args.category or args.benchmark):
        parser.print_help()
        return 1

    # Build configuration
    config = {"quick": args.quick}
    if args.corpus_size:
        config["n_docs"] = args.corpus_size

    # Create suite
    benchmarks = [args.benchmark] if args.benchmark else None
    categories = [args.category] if args.category else None
    suite = create_suite(benchmarks=benchmarks, categories=categories, config=config)

    if not suite.benchmarks:
        print("No benchmarks to run.", file=sys.stderr)
        return 1

    # Run benchmarks
    mode = "quick" if args.quick else "full"
    print(f"\nRunning {len(suite.benchmarks)} corpus benchmark(s) [{mode} mode]...")
    print("=" * 60)

    callback = progress_callback if args.verbose else None
    results = suite.run_all(progress_callback=callback)

    if args.verbose:
        print()  # Newline after progress bar

    # Print summary
    print(suite.summary())

    # Print cache stats
    cache = CorpusCache()
    stats = cache.stats()
    print(f"\nCache: {stats['hits']} hits, {stats['misses']} misses")
    print(f"       {stats['corpus_count']} corpora, {stats['processor_count']} processors cached")

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
        output_path.parent.mkdir(parents=True, exist_ok=True)
        suite.save_results(output_path)
        print(f"\nResults saved to: {output_path}")

    # Return exit code based on results
    failed = sum(
        1 for r in results
        if r.status in (BenchmarkStatus.FAILED, BenchmarkStatus.ERROR)
    )
    skipped = sum(1 for r in results if r.status == BenchmarkStatus.SKIPPED)

    if skipped == len(results):
        print("\nNote: All benchmarks were skipped (not yet implemented)")
        return 0  # Don't fail for skipped benchmarks

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
