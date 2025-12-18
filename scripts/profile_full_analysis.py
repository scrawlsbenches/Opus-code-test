#!/usr/bin/env python3
"""
Profile full-analysis to identify bottlenecks.

This script profiles each phase of compute_all() separately to identify
where time is being spent and why full-analysis hangs.

Usage:
    python scripts/profile_full_analysis.py
    python scripts/profile_full_analysis.py --phase louvain
    python scripts/profile_full_analysis.py --phase semantics
    python scripts/profile_full_analysis.py --timeout 30
"""

import argparse
import cProfile
import pstats
import io
import time
import signal
import sys
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Callable, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cortical.processor import CorticalTextProcessor
from cortical.analysis import cluster_by_louvain, cluster_by_label_propagation
from cortical.layers import CorticalLayer


class TimeoutError(Exception):
    """Raised when operation times out."""
    pass


@contextmanager
def timeout(seconds: int, operation: str = "operation"):
    """Context manager for timing out long operations."""
    def handler(signum, frame):
        raise TimeoutError(f"{operation} timed out after {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def profile_function(func: Callable, *args, timeout_sec: int = 60, **kwargs) -> tuple:
    """
    Profile a function and return stats.

    Returns:
        (result, elapsed_time, profile_stats, timed_out)
    """
    profiler = cProfile.Profile()
    start = time.time()
    result = None
    timed_out = False

    try:
        with timeout(timeout_sec, func.__name__):
            profiler.enable()
            result = func(*args, **kwargs)
            profiler.disable()
    except TimeoutError as e:
        profiler.disable()
        timed_out = True
        print(f"\n⚠️  {e}")

    elapsed = time.time() - start

    # Get stats
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('cumulative')
    stats.print_stats(20)

    return result, elapsed, stream.getvalue(), timed_out


def load_corpus(corpus_path: str) -> CorticalTextProcessor:
    """Load the corpus for profiling."""
    print(f"Loading corpus from {corpus_path}...")
    processor = CorticalTextProcessor.load(corpus_path)

    # Print corpus stats
    layer0 = processor.layers.get(CorticalLayer.TOKENS)
    layer1 = processor.layers.get(CorticalLayer.BIGRAMS)

    print(f"  Documents: {len(processor.documents)}")
    print(f"  Tokens (Layer 0): {layer0.column_count() if layer0 else 0}")
    print(f"  Bigrams (Layer 1): {layer1.column_count() if layer1 else 0}")
    print()

    return processor


def profile_louvain(processor: CorticalTextProcessor, timeout_sec: int = 60) -> dict:
    """Profile Louvain clustering."""
    print("=" * 60)
    print("PROFILING: Louvain Clustering")
    print("=" * 60)

    layer0 = processor.layers.get(CorticalLayer.TOKENS)
    if not layer0:
        print("No token layer found!")
        return {}

    print(f"Layer 0 has {layer0.column_count()} minicolumns")
    print(f"Total connections: {layer0.total_connections()}")
    print()

    result, elapsed, stats, timed_out = profile_function(
        cluster_by_louvain,
        layer0,
        min_cluster_size=3,
        resolution=1.0,
        timeout_sec=timeout_sec
    )

    print(f"\nElapsed: {elapsed:.2f}s {'(TIMED OUT)' if timed_out else ''}")
    if result:
        print(f"Clusters found: {len(result)}")
    print("\nTop 20 functions by cumulative time:")
    print(stats)

    return {
        'phase': 'louvain',
        'elapsed': elapsed,
        'timed_out': timed_out,
        'clusters': len(result) if result else 0
    }


def profile_label_propagation(processor: CorticalTextProcessor, timeout_sec: int = 60) -> dict:
    """Profile label propagation clustering (legacy)."""
    print("=" * 60)
    print("PROFILING: Label Propagation Clustering")
    print("=" * 60)

    layer0 = processor.layers.get(CorticalLayer.TOKENS)
    if not layer0:
        print("No token layer found!")
        return {}

    print(f"Layer 0 has {layer0.column_count()} minicolumns")
    print()

    result, elapsed, stats, timed_out = profile_function(
        cluster_by_label_propagation,
        layer0,
        min_cluster_size=3,
        timeout_sec=timeout_sec
    )

    print(f"\nElapsed: {elapsed:.2f}s {'(TIMED OUT)' if timed_out else ''}")
    if result:
        print(f"Clusters found: {len(result)}")
    print("\nTop 20 functions by cumulative time:")
    print(stats)

    return {
        'phase': 'label_propagation',
        'elapsed': elapsed,
        'timed_out': timed_out,
        'clusters': len(result) if result else 0
    }


def profile_semantics(processor: CorticalTextProcessor, timeout_sec: int = 60) -> dict:
    """Profile semantic relation extraction."""
    print("=" * 60)
    print("PROFILING: Semantic Relation Extraction")
    print("=" * 60)

    print(f"Documents: {len(processor.documents)}")
    total_chars = sum(len(doc) for doc in processor.documents.values())
    print(f"Total characters: {total_chars:,}")
    print()

    result, elapsed, stats, timed_out = profile_function(
        processor.extract_corpus_semantics,
        use_pattern_extraction=True,
        verbose=False,
        timeout_sec=timeout_sec
    )

    print(f"\nElapsed: {elapsed:.2f}s {'(TIMED OUT)' if timed_out else ''}")
    print(f"Relations extracted: {len(processor.semantic_relations)}")
    print("\nTop 20 functions by cumulative time:")
    print(stats)

    return {
        'phase': 'semantics',
        'elapsed': elapsed,
        'timed_out': timed_out,
        'relations': len(processor.semantic_relations)
    }


def profile_bigram_connections(processor: CorticalTextProcessor, timeout_sec: int = 60) -> dict:
    """Profile bigram connection computation."""
    print("=" * 60)
    print("PROFILING: Bigram Connections")
    print("=" * 60)

    layer1 = processor.layers.get(CorticalLayer.BIGRAMS)
    if layer1:
        print(f"Bigrams: {layer1.column_count()}")
    print()

    result, elapsed, stats, timed_out = profile_function(
        processor.compute_bigram_connections,
        verbose=False,
        timeout_sec=timeout_sec
    )

    print(f"\nElapsed: {elapsed:.2f}s {'(TIMED OUT)' if timed_out else ''}")
    print("\nTop 20 functions by cumulative time:")
    print(stats)

    return {
        'phase': 'bigram_connections',
        'elapsed': elapsed,
        'timed_out': timed_out
    }


def profile_all_phases(processor: CorticalTextProcessor, timeout_sec: int = 30) -> list:
    """Profile all phases of full-analysis."""
    results = []

    # Fast phases first
    print("\n" + "=" * 60)
    print("PHASE 1: Activation Propagation")
    print("=" * 60)
    start = time.time()
    processor.propagate_activation(verbose=False)
    elapsed = time.time() - start
    print(f"Elapsed: {elapsed:.2f}s")
    results.append({'phase': 'activation', 'elapsed': elapsed, 'timed_out': False})

    print("\n" + "=" * 60)
    print("PHASE 2: PageRank")
    print("=" * 60)
    start = time.time()
    processor.compute_importance(verbose=False)
    elapsed = time.time() - start
    print(f"Elapsed: {elapsed:.2f}s")
    results.append({'phase': 'pagerank', 'elapsed': elapsed, 'timed_out': False})

    print("\n" + "=" * 60)
    print("PHASE 3: TF-IDF")
    print("=" * 60)
    start = time.time()
    processor.compute_tfidf(verbose=False)
    elapsed = time.time() - start
    print(f"Elapsed: {elapsed:.2f}s")
    results.append({'phase': 'tfidf', 'elapsed': elapsed, 'timed_out': False})

    print("\n" + "=" * 60)
    print("PHASE 4: Document Connections")
    print("=" * 60)
    start = time.time()
    processor.compute_document_connections(verbose=False)
    elapsed = time.time() - start
    print(f"Elapsed: {elapsed:.2f}s")
    results.append({'phase': 'doc_connections', 'elapsed': elapsed, 'timed_out': False})

    # Potentially slow phases
    results.append(profile_bigram_connections(processor, timeout_sec))
    results.append(profile_louvain(processor, timeout_sec))
    results.append(profile_semantics(processor, timeout_sec))

    return results


def print_summary(results: list):
    """Print profiling summary."""
    print("\n" + "=" * 60)
    print("PROFILING SUMMARY")
    print("=" * 60)

    total_time = sum(r['elapsed'] for r in results)

    print(f"\n{'Phase':<25} {'Time':>10} {'Status':>15}")
    print("-" * 50)

    for r in results:
        status = "TIMED OUT" if r.get('timed_out') else "OK"
        print(f"{r['phase']:<25} {r['elapsed']:>9.2f}s {status:>15}")

    print("-" * 50)
    print(f"{'TOTAL':<25} {total_time:>9.2f}s")

    # Identify bottleneck
    slowest = max(results, key=lambda x: x['elapsed'])
    print(f"\n🔍 BOTTLENECK: {slowest['phase']} ({slowest['elapsed']:.2f}s)")

    timed_out = [r for r in results if r.get('timed_out')]
    if timed_out:
        print(f"\n⚠️  TIMED OUT PHASES: {', '.join(r['phase'] for r in timed_out)}")


def main():
    parser = argparse.ArgumentParser(description="Profile full-analysis bottlenecks")
    parser.add_argument('--corpus', default='corpus_dev',
                       help='Path to corpus directory (JSON format)')
    parser.add_argument('--phase', choices=['all', 'louvain', 'label_propagation',
                                            'semantics', 'bigram'],
                       default='all', help='Phase to profile')
    parser.add_argument('--timeout', type=int, default=30,
                       help='Timeout per phase in seconds')

    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"Corpus not found: {corpus_path}")
        print("Run: python scripts/index_codebase.py first")
        sys.exit(1)

    processor = load_corpus(str(corpus_path))

    if args.phase == 'all':
        results = profile_all_phases(processor, args.timeout)
        print_summary(results)
    elif args.phase == 'louvain':
        profile_louvain(processor, args.timeout)
    elif args.phase == 'label_propagation':
        profile_label_propagation(processor, args.timeout)
    elif args.phase == 'semantics':
        profile_semantics(processor, args.timeout)
    elif args.phase == 'bigram':
        profile_bigram_connections(processor, args.timeout)


if __name__ == '__main__':
    main()
