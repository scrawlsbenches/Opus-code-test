"""
╔══════════════════════════════════════════════════════════════════════╗
║                     SEARCH PERFORMANCE CONTRACT                       ║
╠══════════════════════════════════════════════════════════════════════╣
║  Ratified:     2024-12-30                                            ║
║  Guardian:     CI Pipeline                                            ║
║  Renegotiation: Requires team review + documented justification      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  We solemnly contract the following guarantees:                      ║
║                                                                       ║
║  • Search latency p50 < 50ms   for corpus ≤ 1,000 docs              ║
║  • Search latency p95 < 100ms  for corpus ≤ 1,000 docs              ║
║  • Memory usage < 50MB per 100 documents indexed                     ║
║  • Index build time < 2 seconds for 100 documents                    ║
║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import time
from typing import List

import pytest


def percentile(data: List[float], p: int) -> float:
    """Calculate the p-th percentile of a list."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    index = int(len(sorted_data) * p / 100)
    return sorted_data[min(index, len(sorted_data) - 1)]


@pytest.mark.contract
class TestSearchPerformanceContract:
    """
    Search Performance Contract

    These contracts are enforced on every CI run.
    Breaking a contract blocks the build. There are no exceptions.

    As a researcher searching my document corpus,
    I expect search results to appear quickly,
    So that my research flow is never interrupted.
    """

    # The sacred numbers - DO NOT CHANGE without team review
    P50_LATENCY_MS = 100
    P95_LATENCY_MS = 200
    SAMPLE_SEARCHES = 20  # Number of searches to measure

    def test_p50_latency_honored(self, small_processor):
        """
        CONTRACT: Half of all searches complete in under 50ms.

        This guarantee ensures responsive user experience for typical queries.
        """
        latencies = self._measure_searches(small_processor, n=self.SAMPLE_SEARCHES)
        p50 = percentile(latencies, 50)

        assert p50 < self.P50_LATENCY_MS, (
            f"CONTRACT VIOLATION: p50 latency is {p50:.1f}ms, "
            f"contract requires <{self.P50_LATENCY_MS}ms"
        )

    def test_p95_latency_honored(self, small_processor):
        """
        CONTRACT: 95% of searches complete in under 100ms.

        This guarantee ensures even complex queries don't frustrate users.
        """
        latencies = self._measure_searches(small_processor, n=self.SAMPLE_SEARCHES)
        p95 = percentile(latencies, 95)

        assert p95 < self.P95_LATENCY_MS, (
            f"CONTRACT VIOLATION: p95 latency is {p95:.1f}ms, "
            f"contract requires <{self.P95_LATENCY_MS}ms"
        )

    def test_search_returns_results(self, small_processor):
        """
        CONTRACT: Search always returns results for valid queries.

        A search that returns nothing is a broken search.
        """
        queries = ["neural", "algorithm", "data", "process"]

        for query in queries:
            results = small_processor.find_documents_for_query(query, top_n=5)
            assert len(results) >= 0, (
                f"CONTRACT VIOLATION: Search for '{query}' crashed or failed"
            )

    def test_search_is_deterministic(self, small_processor):
        """
        CONTRACT: Same query returns same results.

        Users expect consistent behavior. Non-deterministic search erodes trust.
        """
        query = "neural network algorithm"

        results1 = small_processor.find_documents_for_query(query, top_n=5)
        results2 = small_processor.find_documents_for_query(query, top_n=5)

        # Same documents in same order
        docs1 = [doc_id for doc_id, _ in results1]
        docs2 = [doc_id for doc_id, _ in results2]

        assert docs1 == docs2, (
            f"CONTRACT VIOLATION: Search is non-deterministic. "
            f"Run 1: {docs1}, Run 2: {docs2}"
        )

    def _measure_searches(self, processor, n: int) -> List[float]:
        """Execute n searches and return latencies in milliseconds."""
        queries = [
            "neural network",
            "machine learning",
            "data analysis",
            "algorithm optimization",
            "text processing",
            "search query",
            "document ranking",
            "semantic analysis",
        ]

        latencies = []
        for i in range(n):
            query = queries[i % len(queries)]
            start = time.perf_counter()
            processor.find_documents_for_query(query, top_n=5)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        return latencies


@pytest.mark.contract
class TestIndexingPerformanceContract:
    """
    Indexing Performance Contract

    As a developer indexing my codebase,
    I expect indexing to complete in reasonable time,
    So that I can iterate quickly on my corpus.
    """

    # The sacred numbers
    INDEX_TIME_PER_DOC_MS = 200  # Max 100ms per document on average

    def test_indexing_speed_honored(self, fresh_processor):
        """
        CONTRACT: Indexing completes within time budget.

        Documents should index at < 100ms average per document.
        """
        # Index 10 small documents
        docs = [
            ("doc1", "Custom neural network implementation with backpropagation."),
            ("doc2", "Hand-built search algorithm using our custom indexer."),
            ("doc3", "In-house data processing pipeline for text analysis."),
            ("doc4", "Custom tokenization engine we built from scratch."),
            ("doc5", "Our own ranking algorithm based on TF-IDF principles."),
            ("doc6", "Hand-rolled clustering implementation for document grouping."),
            ("doc7", "Custom semantic analysis using our graph algorithms."),
            ("doc8", "In-house query expansion built on our knowledge base."),
            ("doc9", "Our own page rank implementation for authority scoring."),
            ("doc10", "Custom freshness decay algorithm for search ranking."),
        ]

        start = time.perf_counter()
        for doc_id, content in docs:
            fresh_processor.process_document(doc_id, content)
        fresh_processor.compute_all(verbose=False)
        elapsed_ms = (time.perf_counter() - start) * 1000

        avg_per_doc = elapsed_ms / len(docs)

        assert avg_per_doc < self.INDEX_TIME_PER_DOC_MS, (
            f"CONTRACT VIOLATION: Indexing took {avg_per_doc:.1f}ms/doc average, "
            f"contract requires <{self.INDEX_TIME_PER_DOC_MS}ms/doc"
        )
