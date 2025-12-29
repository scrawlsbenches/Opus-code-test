"""
Corpus Benchmarks for CorticalTextProcessor.

This package provides benchmarks for measuring CorticalTextProcessor performance:
- Indexing throughput at various corpus sizes
- Query latency and relevance
- Passage retrieval performance
- Analysis algorithm performance
- Code search capabilities
- Semantic fingerprinting

Usage:
    python -m benchmarks.corpus.runner --all
    python -m benchmarks.corpus.runner --category indexing
    python -m benchmarks.corpus.runner --list
    python -m benchmarks.corpus.runner --all --quick
"""

from benchmarks.corpus.base import (
    CorpusBenchmark,
    CorpusBenchmarkCategory,
    CorpusCache,
    SyntheticCorpusGenerator,
)

__all__ = [
    "CorpusBenchmark",
    "CorpusBenchmarkCategory",
    "CorpusCache",
    "SyntheticCorpusGenerator",
]
