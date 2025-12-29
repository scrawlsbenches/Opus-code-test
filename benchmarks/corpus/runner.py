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
    SyntheticCorpusConfig,
    SyntheticCorpusGenerator,
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
# INDEXING BENCHMARKS
# =============================================================================

import time
import statistics

from cortical.processor import CorticalTextProcessor


@register_benchmark
class IndexingThroughputBenchmark(CorpusBenchmark):
    """
    Measure document indexing throughput (docs/sec) at various corpus sizes.

    Tests process_document() performance across different corpus scales.
    """

    name = "indexing_throughput"
    description = "Measure document indexing throughput (docs/sec)"
    corpus_category = CorpusBenchmarkCategory.INDEXING

    # Test sizes for full and quick modes
    FULL_SIZES = [50, 100, 500, 1000]
    QUICK_SIZES = [25, 50, 100]

    def run(self) -> BenchmarkResult:
        result = self.create_result()

        # Determine which sizes to test
        is_quick = self.config.get("quick", False)
        sizes = self.QUICK_SIZES if is_quick else self.FULL_SIZES

        # Generate synthetic documents for testing
        generator = SyntheticCorpusGenerator(SyntheticCorpusConfig(
            n_docs=max(sizes),
            doc_length=self._corpus_config.doc_length,
            vocab_size=self._corpus_config.vocab_size,
            seed=42,  # Reproducible
        ))
        all_docs = generator.generate()
        doc_items = list(all_docs.items())

        for size in sizes:
            # Create fresh processor for each size
            processor = CorticalTextProcessor()

            # Measure indexing time
            start = time.perf_counter()
            for i in range(size):
                doc_id, text = doc_items[i]
                processor.process_document(doc_id, text)
            elapsed = time.perf_counter() - start

            docs_per_sec = size / elapsed if elapsed > 0 else float('inf')

            result.add_metric(
                name=f"throughput_{size}_docs",
                value=docs_per_sec,
                unit="docs/sec",
                threshold_min=10.0,  # At least 10 docs/sec
            )

            result.metadata[f"time_{size}_docs_ms"] = elapsed * 1000

        return result


@register_benchmark
class IncrementalIndexingBenchmark(CorpusBenchmark):
    """
    Compare add_document_incremental() vs full recompute.

    Measures the speedup factor when adding documents incrementally
    versus reprocessing the entire corpus.
    """

    name = "incremental_indexing"
    description = "Compare incremental vs full indexing performance"
    corpus_category = CorpusBenchmarkCategory.INDEXING

    def run(self) -> BenchmarkResult:
        result = self.create_result()

        # Get base corpus
        is_quick = self.config.get("quick", False)
        base_size = 25 if is_quick else 100
        add_count = 5 if is_quick else 20

        # Generate documents
        generator = SyntheticCorpusGenerator(SyntheticCorpusConfig(
            n_docs=base_size + add_count,
            doc_length=self._corpus_config.doc_length,
            vocab_size=self._corpus_config.vocab_size,
            seed=42,
        ))
        all_docs = generator.generate()
        doc_items = list(all_docs.items())

        base_docs = doc_items[:base_size]
        new_docs = doc_items[base_size:base_size + add_count]

        # Method 1: Full recompute after each add
        processor_full = CorticalTextProcessor()
        for doc_id, text in base_docs:
            processor_full.process_document(doc_id, text)
        processor_full.compute_all()

        full_times = []
        for doc_id, text in new_docs:
            start = time.perf_counter()
            processor_full.process_document(doc_id, text)
            processor_full.compute_all()
            elapsed = time.perf_counter() - start
            full_times.append(elapsed * 1000)  # ms

        # Method 2: Incremental add
        processor_incr = CorticalTextProcessor()
        for doc_id, text in base_docs:
            processor_incr.process_document(doc_id, text)
        processor_incr.compute_all()

        incr_times = []
        for doc_id, text in new_docs:
            start = time.perf_counter()
            processor_incr.add_document_incremental(doc_id, text)
            elapsed = time.perf_counter() - start
            incr_times.append(elapsed * 1000)  # ms

        # Calculate metrics
        avg_full = statistics.mean(full_times) if full_times else 0
        avg_incr = statistics.mean(incr_times) if incr_times else 0
        speedup = avg_full / avg_incr if avg_incr > 0 else float('inf')

        result.add_metric(
            name="avg_full_recompute_ms",
            value=avg_full,
            unit="ms",
        )
        result.add_metric(
            name="avg_incremental_ms",
            value=avg_incr,
            unit="ms",
        )
        result.add_metric(
            name="speedup_factor",
            value=speedup,
            unit="x",
            threshold_min=1.5,  # Incremental should be at least 1.5x faster
        )

        result.metadata.update({
            "base_corpus_size": base_size,
            "documents_added": add_count,
            "full_times_ms": full_times,
            "incr_times_ms": incr_times,
        })

        return result


@register_benchmark
class ComputeAllBenchmark(CorpusBenchmark):
    """
    Measure compute_all() phase-by-phase timing.

    Reports time for each major phase:
    - TF-IDF computation
    - PageRank computation
    - Clustering (Louvain)
    - Bigram connections
    - Semantics extraction
    """

    name = "compute_all_phases"
    description = "Measure compute_all() phase timing breakdown"
    corpus_category = CorpusBenchmarkCategory.ANALYSIS

    def run(self) -> BenchmarkResult:
        result = self.create_result()

        # Create a fresh processor with corpus
        generator = SyntheticCorpusGenerator(self._corpus_config)
        corpus = generator.generate()

        processor = CorticalTextProcessor()
        for doc_id, text in corpus.items():
            processor.process_document(doc_id, text)

        # Time each phase individually
        phases = {
            "tfidf": lambda: processor.compute_tfidf(),
            "pagerank": lambda: processor.compute_importance(),
            "bigram_connections": lambda: processor.compute_bigram_connections(),
            "concepts": lambda: processor.build_concept_clusters(),
        }

        total_time = 0
        phase_times = {}

        for phase_name, phase_func in phases.items():
            start = time.perf_counter()
            try:
                phase_func()
            except Exception as e:
                result.metadata[f"{phase_name}_error"] = str(e)
                continue
            elapsed_ms = (time.perf_counter() - start) * 1000
            phase_times[phase_name] = elapsed_ms
            total_time += elapsed_ms

            result.add_metric(
                name=f"{phase_name}_ms",
                value=elapsed_ms,
                unit="ms",
            )

        # Time semantics extraction (more expensive, optional)
        start = time.perf_counter()
        try:
            processor.extract_corpus_semantics()
            semantics_time = (time.perf_counter() - start) * 1000
            phase_times["semantics"] = semantics_time
            total_time += semantics_time

            result.add_metric(
                name="semantics_ms",
                value=semantics_time,
                unit="ms",
            )
        except Exception as e:
            result.metadata["semantics_error"] = str(e)

        # Add total and breakdown percentages
        result.add_metric(
            name="total_compute_ms",
            value=total_time,
            unit="ms",
        )

        result.metadata.update({
            "corpus_size": self._corpus_config.n_docs,
            "phase_times_ms": phase_times,
            "phase_percentages": {
                k: (v / total_time * 100) if total_time > 0 else 0
                for k, v in phase_times.items()
            },
        })

        return result


@register_benchmark
class BatchIndexingBenchmark(CorpusBenchmark):
    """
    Measure add_documents_batch() performance.

    Compares batch insertion vs sequential process_document() calls.
    """

    name = "batch_indexing"
    description = "Measure batch vs sequential document indexing"
    corpus_category = CorpusBenchmarkCategory.INDEXING

    def run(self) -> BenchmarkResult:
        result = self.create_result()

        is_quick = self.config.get("quick", False)
        batch_size = 25 if is_quick else 100

        # Generate documents
        generator = SyntheticCorpusGenerator(SyntheticCorpusConfig(
            n_docs=batch_size,
            doc_length=self._corpus_config.doc_length,
            vocab_size=self._corpus_config.vocab_size,
            seed=42,
        ))
        corpus = generator.generate()
        doc_items = list(corpus.items())

        # Convert to batch format: List[Tuple[str, str, Optional[Dict]]]
        batch_docs = [(doc_id, text, None) for doc_id, text in doc_items]

        # Method 1: Sequential processing
        processor_seq = CorticalTextProcessor()
        start = time.perf_counter()
        for doc_id, text in doc_items:
            processor_seq.process_document(doc_id, text)
        sequential_time = (time.perf_counter() - start) * 1000

        # Method 2: Batch processing
        processor_batch = CorticalTextProcessor()
        start = time.perf_counter()
        processor_batch.add_documents_batch(batch_docs, recompute='none', verbose=False)
        batch_time = (time.perf_counter() - start) * 1000

        speedup = sequential_time / batch_time if batch_time > 0 else float('inf')

        result.add_metric(
            name="sequential_ms",
            value=sequential_time,
            unit="ms",
        )
        result.add_metric(
            name="batch_ms",
            value=batch_time,
            unit="ms",
        )
        # Note: batch API has overhead that makes it slower for small batches.
        # This is informational - no threshold. For large batches, measure separately.
        result.add_metric(
            name="batch_speedup",
            value=speedup,
            unit="x",
            # No threshold - this is informational. Batch overhead makes it
            # slower for small batches but may be faster for very large batches.
        )

        result.metadata.update({
            "batch_size": batch_size,
            "sequential_docs_per_sec": batch_size / (sequential_time / 1000) if sequential_time > 0 else 0,
            "batch_docs_per_sec": batch_size / (batch_time / 1000) if batch_time > 0 else 0,
        })

        return result


@register_benchmark
class LargeDocumentBenchmark(CorpusBenchmark):
    """
    Measure indexing performance for large documents.

    Tests with documents of various sizes: 10KB, 100KB, 1MB equivalent.
    """

    name = "large_document_indexing"
    description = "Measure indexing performance for large documents"
    corpus_category = CorpusBenchmarkCategory.INDEXING

    # Approximate word counts for target sizes (assuming ~5 chars/word + space)
    SIZE_CONFIGS = {
        "10KB": 1700,    # ~10,000 chars / 6 ≈ 1700 words
        "100KB": 17000,  # ~100,000 chars / 6 ≈ 17000 words
        "1MB": 170000,   # ~1,000,000 chars / 6 ≈ 170000 words (quick mode skips)
    }

    QUICK_SIZE_CONFIGS = {
        "10KB": 1700,
        "50KB": 8500,
    }

    def run(self) -> BenchmarkResult:
        result = self.create_result()

        is_quick = self.config.get("quick", False)
        size_configs = self.QUICK_SIZE_CONFIGS if is_quick else self.SIZE_CONFIGS

        for size_name, word_count in size_configs.items():
            # Generate a large document
            generator = SyntheticCorpusGenerator(SyntheticCorpusConfig(
                n_docs=1,
                doc_length=word_count,
                vocab_size=5000,  # Larger vocab for big docs
                seed=42,
            ))
            corpus = generator.generate()
            doc_text = list(corpus.values())[0]

            # Time indexing
            processor = CorticalTextProcessor()
            start = time.perf_counter()
            processor.process_document(f"large_doc_{size_name}", doc_text)
            elapsed_ms = (time.perf_counter() - start) * 1000

            # Calculate throughput in KB/sec
            actual_size_kb = len(doc_text) / 1024
            kb_per_sec = actual_size_kb / (elapsed_ms / 1000) if elapsed_ms > 0 else 0

            result.add_metric(
                name=f"time_{size_name.lower()}_ms",
                value=elapsed_ms,
                unit="ms",
            )
            result.add_metric(
                name=f"throughput_{size_name.lower()}_kbps",
                value=kb_per_sec,
                unit="KB/sec",
                threshold_min=100.0,  # At least 100 KB/sec
            )

            result.metadata[f"actual_size_{size_name.lower()}_kb"] = actual_size_kb
            result.metadata[f"word_count_{size_name.lower()}"] = word_count

        return result


# =============================================================================
# QUERY BENCHMARKS (Stage 1)
# =============================================================================

from benchmarks.corpus.base import measure_latency_percentiles


@register_benchmark
class SearchLatencyBenchmark(CorpusBenchmark):
    """
    Measure search latency percentiles (p50/p90/p99).

    Tests find_documents_for_query() performance with:
    - Cold cache (first query)
    - Warm cache (repeated queries)
    - Various corpus sizes
    """

    name = "search_latency"
    description = "Measure search latency percentiles (p50/p90/p99)"
    corpus_category = CorpusBenchmarkCategory.QUERY

    # Test queries (concept-based to match synthetic corpus)
    TEST_QUERIES = [
        "concept0",
        "concept1 concept2",
        "word0 word1 word2",
        "concept5 important",
    ]

    def run(self) -> BenchmarkResult:
        result = self.create_result()

        # self._processor is pre-loaded and computed via setup()
        processor = self._processor

        # Cold cache test - first query
        cold_query = self.TEST_QUERIES[0]
        cold_start = time.perf_counter()
        processor.find_documents_for_query(cold_query, top_n=5)
        cold_latency_ms = (time.perf_counter() - cold_start) * 1000

        result.add_metric(
            name="cold_cache_latency_ms",
            value=cold_latency_ms,
            unit="ms",
        )

        # Warm cache test - measure percentiles over repeated queries
        is_quick = self.config.get("quick", False)
        n_iterations = 50 if is_quick else 200

        latencies_ms = []
        for i in range(n_iterations):
            query = self.TEST_QUERIES[i % len(self.TEST_QUERIES)]
            start = time.perf_counter()
            processor.find_documents_for_query(query, top_n=5)
            elapsed = (time.perf_counter() - start) * 1000
            latencies_ms.append(elapsed)

        latencies_ms.sort()
        n = len(latencies_ms)

        def percentile(p: float) -> float:
            idx = int(n * p / 100)
            return latencies_ms[min(idx, n - 1)]

        p50 = percentile(50)
        p90 = percentile(90)
        p99 = percentile(99)
        mean_latency = sum(latencies_ms) / n

        result.add_metric(
            name="p50_latency_ms",
            value=p50,
            unit="ms",
            threshold_max=100.0,  # Should be under 100ms
        )
        result.add_metric(
            name="p90_latency_ms",
            value=p90,
            unit="ms",
            threshold_max=200.0,  # p90 under 200ms
        )
        result.add_metric(
            name="p99_latency_ms",
            value=p99,
            unit="ms",
        )
        result.add_metric(
            name="mean_latency_ms",
            value=mean_latency,
            unit="ms",
        )

        # Calculate queries per second
        total_time_sec = sum(latencies_ms) / 1000
        qps = n_iterations / total_time_sec if total_time_sec > 0 else 0

        result.add_metric(
            name="queries_per_second",
            value=qps,
            unit="qps",
            threshold_min=10.0,  # At least 10 queries/sec
        )

        result.metadata.update({
            "corpus_size": self._corpus_config.n_docs,
            "iterations": n_iterations,
            "queries_tested": self.TEST_QUERIES,
            "latency_distribution": {
                "min_ms": min(latencies_ms),
                "max_ms": max(latencies_ms),
                "std_ms": (sum((x - mean_latency) ** 2 for x in latencies_ms) / n) ** 0.5,
            },
        })

        return result


@register_benchmark
class ColdWarmCacheBenchmark(CorpusBenchmark):
    """
    Compare cold vs warm cache search performance.

    Measures the speedup from query caching and pre-computed indices.
    """

    name = "cold_warm_cache"
    description = "Compare cold vs warm cache search performance"
    corpus_category = CorpusBenchmarkCategory.QUERY

    def run(self) -> BenchmarkResult:
        result = self.create_result()

        # Create fresh processor for cold cache test
        generator = SyntheticCorpusGenerator(self._corpus_config)
        corpus = generator.generate()

        processor = CorticalTextProcessor()
        for doc_id, text in corpus.items():
            processor.process_document(doc_id, text)
        processor.compute_all()

        test_query = "concept0 concept1"

        # Cold cache - first query on fresh processor
        cold_times = []
        for _ in range(5):
            # Create new processor each time for true cold cache
            fresh_processor = CorticalTextProcessor()
            for doc_id, text in corpus.items():
                fresh_processor.process_document(doc_id, text)
            fresh_processor.compute_all()

            start = time.perf_counter()
            fresh_processor.find_documents_for_query(test_query, top_n=5)
            cold_times.append((time.perf_counter() - start) * 1000)

        # Warm cache - repeated queries on same processor
        warm_times = []
        for _ in range(20):
            start = time.perf_counter()
            processor.find_documents_for_query(test_query, top_n=5)
            warm_times.append((time.perf_counter() - start) * 1000)

        avg_cold = statistics.mean(cold_times)
        avg_warm = statistics.mean(warm_times)
        speedup = avg_cold / avg_warm if avg_warm > 0 else float('inf')

        result.add_metric(
            name="avg_cold_cache_ms",
            value=avg_cold,
            unit="ms",
        )
        result.add_metric(
            name="avg_warm_cache_ms",
            value=avg_warm,
            unit="ms",
        )
        result.add_metric(
            name="cache_speedup",
            value=speedup,
            unit="x",
        )

        result.metadata.update({
            "cold_samples": len(cold_times),
            "warm_samples": len(warm_times),
            "test_query": test_query,
        })

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
