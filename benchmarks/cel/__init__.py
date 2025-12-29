"""
Cognitive Event Lattice (CEL) Benchmark Suite

Comprehensive benchmarks for validating the event-sourced cognitive substrate.

Benchmark Categories:
- Throughput: Event append rates, materialization speed
- Memory: Storage efficiency, compaction effectiveness
- Query: Semantic search, temporal queries, DAG traversal
- Correctness: Content-addressing, causality preservation

Usage:
    python -m benchmarks.cel.runner --all
    python -m benchmarks.cel.runner --category throughput
    python -m benchmarks.cel.runner --benchmark event_append
"""

from benchmarks.woven_mind.base import (
    BenchmarkResult,
    BenchmarkSuite,
    BenchmarkStatus,
    BenchmarkCategory,
    BenchmarkMetric,
    BaseBenchmark,
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

__all__ = [
    # Base classes (re-exported)
    'BenchmarkResult',
    'BenchmarkSuite',
    'BenchmarkStatus',
    'BenchmarkCategory',
    'BenchmarkMetric',
    'BaseBenchmark',
    # CEL benchmarks
    'EventAppendBenchmark',
    'MaterializationBenchmark',
    'SemanticIndexBenchmark',
    'TimeTravelBenchmark',
    'DAGTraversalBenchmark',
    'ContentAddressingBenchmark',
    'CompactionBenchmark',
]
