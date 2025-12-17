"""
Performance benchmark tests for the Cortical Text Processor.

These tests measure execution time of key operations to detect
performance regressions. They log results but don't fail on timing
(since hardware varies).

Run with: python -m pytest tests/performance/test_benchmarks.py -v -s
"""

import time
import unittest
import logging

from cortical import CorticalTextProcessor, CorticalConfig

logger = logging.getLogger(__name__)


def create_synthetic_corpus(num_docs: int = 50, words_per_doc: int = 200) -> list:
    """
    Create a synthetic corpus for benchmarking.

    Returns:
        List of (doc_id, text) tuples
    """
    # Use deterministic content for reproducibility
    topics = [
        "neural networks machine learning deep learning",
        "natural language processing text analysis",
        "information retrieval search ranking",
        "graph algorithms pagerank clustering",
        "data structures algorithms optimization",
    ]

    docs = []
    for i in range(num_docs):
        topic = topics[i % len(topics)]
        words = []
        for j in range(words_per_doc // 5):
            words.extend(topic.split())
            words.append(f"doc{i}")
            words.append(f"word{j}")

        text = " ".join(words[:words_per_doc])
        docs.append((f"doc_{i:04d}", text))

    return docs


class BenchmarkResult:
    """Container for benchmark results."""

    def __init__(self, operation: str, duration_ms: float, iterations: int = 1):
        self.operation = operation
        self.duration_ms = duration_ms
        self.iterations = iterations
        self.avg_ms = duration_ms / iterations

    def __str__(self):
        if self.iterations > 1:
            return f"{self.operation}: {self.duration_ms:.2f}ms total, {self.avg_ms:.2f}ms avg ({self.iterations} iterations)"
        return f"{self.operation}: {self.duration_ms:.2f}ms"


class TestProcessDocumentBenchmark(unittest.TestCase):
    """Benchmarks for document processing."""

    def test_benchmark_process_single_document(self):
        """Benchmark processing a single document."""
        processor = CorticalTextProcessor()
        text = " ".join(["word"] * 500)  # 500-word document

        start = time.perf_counter()
        processor.process_document("bench_doc", text)
        duration_ms = (time.perf_counter() - start) * 1000

        result = BenchmarkResult("process_document (500 words)", duration_ms)
        logger.info(result)
        print(f"\n  {result}")

        # Sanity check - should be under 1 second
        self.assertLess(duration_ms, 1000, "process_document too slow")

    def test_benchmark_process_batch(self):
        """Benchmark processing multiple documents."""
        processor = CorticalTextProcessor()
        corpus = create_synthetic_corpus(num_docs=50, words_per_doc=100)

        start = time.perf_counter()
        for doc_id, text in corpus:
            processor.process_document(doc_id, text)
        duration_ms = (time.perf_counter() - start) * 1000

        result = BenchmarkResult("process_document batch", duration_ms, len(corpus))
        logger.info(result)
        print(f"\n  {result}")

        # Should process 50 docs in under 5 seconds
        self.assertLess(duration_ms, 5000, "Batch processing too slow")


class TestComputeAllBenchmark(unittest.TestCase):
    """Benchmarks for compute_all operation."""

    @classmethod
    def setUpClass(cls):
        """Create shared processor with corpus."""
        cls.processor = CorticalTextProcessor()
        corpus = create_synthetic_corpus(num_docs=50, words_per_doc=150)
        for doc_id, text in corpus:
            cls.processor.process_document(doc_id, text)

    def test_benchmark_compute_all(self):
        """Benchmark full computation pipeline."""
        # Create fresh processor for this test
        processor = CorticalTextProcessor()
        corpus = create_synthetic_corpus(num_docs=30, words_per_doc=100)
        for doc_id, text in corpus:
            processor.process_document(doc_id, text)

        start = time.perf_counter()
        processor.compute_all()
        duration_ms = (time.perf_counter() - start) * 1000

        result = BenchmarkResult("compute_all (30 docs)", duration_ms)
        logger.info(result)
        print(f"\n  {result}")

        # Should complete in under 30 seconds
        self.assertLess(duration_ms, 30000, "compute_all too slow")

    def test_benchmark_compute_tfidf_only(self):
        """Benchmark TF-IDF computation only."""
        processor = CorticalTextProcessor()
        corpus = create_synthetic_corpus(num_docs=50, words_per_doc=100)
        for doc_id, text in corpus:
            processor.process_document(doc_id, text)

        start = time.perf_counter()
        processor.compute_tfidf()
        duration_ms = (time.perf_counter() - start) * 1000

        result = BenchmarkResult("compute_tfidf (50 docs)", duration_ms)
        logger.info(result)
        print(f"\n  {result}")

        # TF-IDF should be fast
        self.assertLess(duration_ms, 5000, "compute_tfidf too slow")


class TestQueryBenchmark(unittest.TestCase):
    """Benchmarks for query operations."""

    @classmethod
    def setUpClass(cls):
        """Create shared processor with computed corpus."""
        cls.processor = CorticalTextProcessor()
        corpus = create_synthetic_corpus(num_docs=50, words_per_doc=150)
        for doc_id, text in corpus:
            cls.processor.process_document(doc_id, text)
        cls.processor.compute_all()

    def test_benchmark_find_documents_single(self):
        """Benchmark single query."""
        start = time.perf_counter()
        results = self.processor.find_documents_for_query("neural networks")
        duration_ms = (time.perf_counter() - start) * 1000

        result = BenchmarkResult("find_documents_for_query", duration_ms)
        logger.info(result)
        print(f"\n  {result}")

        # Single query should be very fast
        self.assertLess(duration_ms, 500, "Query too slow")
        self.assertGreater(len(results), 0, "Should find results")

    def test_benchmark_find_documents_batch(self):
        """Benchmark multiple queries."""
        queries = [
            "neural networks",
            "machine learning",
            "text analysis",
            "graph algorithms",
            "data structures",
        ]

        start = time.perf_counter()
        for query in queries:
            self.processor.find_documents_for_query(query)
        duration_ms = (time.perf_counter() - start) * 1000

        result = BenchmarkResult("find_documents batch", duration_ms, len(queries))
        logger.info(result)
        print(f"\n  {result}")

        # 5 queries should complete quickly
        self.assertLess(duration_ms, 2500, "Batch queries too slow")

    def test_benchmark_fast_find_documents(self):
        """Benchmark fast search variant."""
        start = time.perf_counter()
        results = self.processor.fast_find_documents("neural networks machine learning")
        duration_ms = (time.perf_counter() - start) * 1000

        result = BenchmarkResult("fast_find_documents", duration_ms)
        logger.info(result)
        print(f"\n  {result}")

        # Fast variant should be quick
        self.assertLess(duration_ms, 200, "fast_find_documents too slow")

    def test_benchmark_expand_query(self):
        """Benchmark query expansion."""
        iterations = 10

        start = time.perf_counter()
        for _ in range(iterations):
            self.processor.expand_query("neural networks")
        duration_ms = (time.perf_counter() - start) * 1000

        result = BenchmarkResult("expand_query", duration_ms, iterations)
        logger.info(result)
        print(f"\n  {result}")

        # Query expansion should be fast
        self.assertLess(result.avg_ms, 100, "expand_query too slow")


class TestCacheBenchmark(unittest.TestCase):
    """Benchmarks for cache performance."""

    @classmethod
    def setUpClass(cls):
        cls.processor = CorticalTextProcessor()
        corpus = create_synthetic_corpus(num_docs=30, words_per_doc=100)
        for doc_id, text in corpus:
            cls.processor.process_document(doc_id, text)
        cls.processor.compute_all()

    def test_benchmark_cache_hit_vs_miss(self):
        """Compare cache hit vs miss performance."""
        query = "neural networks machine"

        # First call - cache miss
        self.processor.clear_query_cache()
        start = time.perf_counter()
        self.processor.expand_query_cached(query)
        miss_ms = (time.perf_counter() - start) * 1000

        # Second call - cache hit
        start = time.perf_counter()
        self.processor.expand_query_cached(query)
        hit_ms = (time.perf_counter() - start) * 1000

        print(f"\n  Cache miss: {miss_ms:.2f}ms")
        print(f"  Cache hit:  {hit_ms:.2f}ms")
        print(f"  Speedup:    {miss_ms/hit_ms:.1f}x" if hit_ms > 0 else "  Speedup: N/A")

        # Cache hit should be faster than miss
        self.assertLess(hit_ms, miss_ms, "Cache hit should be faster than miss")


if __name__ == "__main__":
    # Configure logging for benchmark output
    logging.basicConfig(level=logging.INFO)
    unittest.main(verbosity=2)
