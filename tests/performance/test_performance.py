"""
Performance Tests
=================

Timing-based tests that catch performance regressions.

IMPORTANT: These tests should NOT run under coverage, which adds 10x+ overhead
and makes timing measurements meaningless. The conftest.py automatically skips
performance tests when coverage is detected.

Run with: pytest tests/performance/ -v --no-cov

Test Design:
- Uses small synthetic corpus (25 docs) for fast, repeatable timing
- Thresholds are set with generous margins for CI variability
- Each test documents expected baseline and threshold rationale

Performance Baselines (on typical hardware):
- Small corpus compute_all(): ~1-2s
- Single search query: ~10-50ms
- Query expansion: ~5-20ms
- Passage retrieval: ~50-100ms
"""

import time
import pytest


# Skip all tests in this module if running under coverage
pytestmark = pytest.mark.performance


class TestComputeAllPerformance:
    """Test compute_all() performance on small corpus."""

    def test_compute_all_small_corpus(self):
        """
        compute_all() on 25-doc corpus should complete quickly.

        Baseline: ~1-2s on typical hardware
        Threshold: 5s (generous for CI variability)

        This catches O(n^2) regressions that would blow up on larger corpora.
        """
        from cortical import CorticalTextProcessor
        from cortical.tokenizer import Tokenizer
        from tests.fixtures.small_corpus import SMALL_CORPUS_DOCS

        # Create fresh processor (don't use fixture - we're timing creation)
        tokenizer = Tokenizer(filter_code_noise=True)
        processor = CorticalTextProcessor(tokenizer=tokenizer)

        # Load documents
        for doc_id, content in SMALL_CORPUS_DOCS.items():
            processor.process_document(doc_id, content)

        # Time compute_all
        start = time.perf_counter()
        processor.compute_all(verbose=False)
        elapsed = time.perf_counter() - start

        # Threshold: 5 seconds (generous for CI)
        assert elapsed < 5.0, (
            f"compute_all() took {elapsed:.2f}s for {len(SMALL_CORPUS_DOCS)} docs. "
            f"Expected < 5s. Check for performance regression."
        )

    def test_individual_compute_phases(self):
        """
        Individual compute phases should complete within bounds.

        This helps identify which phase regressed if compute_all() slows down.
        """
        from cortical import CorticalTextProcessor
        from cortical.tokenizer import Tokenizer
        from tests.fixtures.small_corpus import SMALL_CORPUS_DOCS

        tokenizer = Tokenizer(filter_code_noise=True)
        processor = CorticalTextProcessor(tokenizer=tokenizer)

        for doc_id, content in SMALL_CORPUS_DOCS.items():
            processor.process_document(doc_id, content)

        # Phase thresholds (seconds) - generous for CI
        phase_thresholds = {
            'propagate_activation': 1.0,
            'compute_importance': 1.0,
            'compute_tfidf': 1.0,
            'compute_bigram_connections': 2.0,
            'build_concept_clusters': 2.0,
            'compute_graph_embeddings': 2.0,
        }

        timings = {}

        # Time each phase
        start = time.perf_counter()
        processor.propagate_activation(iterations=5, verbose=False)
        timings['propagate_activation'] = time.perf_counter() - start

        start = time.perf_counter()
        processor.compute_importance(verbose=False)
        timings['compute_importance'] = time.perf_counter() - start

        start = time.perf_counter()
        processor.compute_tfidf(verbose=False)
        timings['compute_tfidf'] = time.perf_counter() - start

        start = time.perf_counter()
        processor.compute_bigram_connections(verbose=False)
        timings['compute_bigram_connections'] = time.perf_counter() - start

        start = time.perf_counter()
        processor.build_concept_clusters(verbose=False)
        timings['build_concept_clusters'] = time.perf_counter() - start

        start = time.perf_counter()
        processor.compute_graph_embeddings(verbose=False)
        timings['compute_graph_embeddings'] = time.perf_counter() - start

        # Check each phase
        failures = []
        for phase, elapsed in timings.items():
            threshold = phase_thresholds[phase]
            if elapsed > threshold:
                failures.append(f"{phase}: {elapsed:.2f}s > {threshold}s")

        assert not failures, (
            f"Phase timing exceeded thresholds:\n" +
            "\n".join(failures) +
            f"\n\nAll timings: {timings}"
        )


class TestSearchPerformance:
    """Test search operation performance."""

    def test_single_query_latency(self, small_processor):
        """
        Single search query should be fast for interactive use.

        Baseline: ~10-50ms
        Threshold: 200ms (generous for CI)
        """
        queries = [
            "machine learning",
            "database indexing",
            "distributed consensus",
            "sorting algorithms",
            "test driven development",
        ]

        for query in queries:
            start = time.perf_counter()
            results = small_processor.find_documents_for_query(query, top_n=5)
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert elapsed_ms < 200, (
                f"Query '{query}' took {elapsed_ms:.1f}ms. "
                f"Expected < 200ms for interactive use."
            )

    def test_fast_search_performance(self, small_processor):
        """
        fast_find_documents() should be faster than standard search.

        This tests the optimized search path.
        """
        query = "neural network optimization"

        # Time standard search
        start = time.perf_counter()
        for _ in range(10):
            small_processor.find_documents_for_query(query, top_n=5)
        standard_elapsed = time.perf_counter() - start

        # Time fast search
        start = time.perf_counter()
        for _ in range(10):
            small_processor.fast_find_documents(query, top_n=5)
        fast_elapsed = time.perf_counter() - start

        # Fast search should not be slower than standard
        # (It may be similar on small corpus, but shouldn't be worse)
        assert fast_elapsed <= standard_elapsed * 1.5, (
            f"fast_find_documents ({fast_elapsed:.3f}s) should not be much slower "
            f"than find_documents_for_query ({standard_elapsed:.3f}s)"
        )

    def test_query_expansion_performance(self, small_processor):
        """
        Query expansion should be fast.

        Baseline: ~5-20ms
        Threshold: 100ms
        """
        queries = ["learning", "database", "algorithm", "testing", "network"]

        for query in queries:
            start = time.perf_counter()
            expanded = small_processor.expand_query(query, max_expansions=20)
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert elapsed_ms < 100, (
                f"expand_query('{query}') took {elapsed_ms:.1f}ms. "
                f"Expected < 100ms."
            )


class TestPassageRetrievalPerformance:
    """Test passage retrieval performance."""

    def test_passage_retrieval_latency(self, small_processor):
        """
        Passage retrieval should complete in reasonable time.

        Baseline: ~50-100ms
        Threshold: 500ms (includes chunking overhead)
        """
        queries = [
            "machine learning models",
            "database transactions",
            "graph algorithms",
        ]

        for query in queries:
            start = time.perf_counter()
            passages = small_processor.find_passages_for_query(
                query,
                top_n=5,
                chunk_size=200,
                overlap=50
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert elapsed_ms < 500, (
                f"find_passages_for_query('{query}') took {elapsed_ms:.1f}ms. "
                f"Expected < 500ms."
            )


class TestScalabilityIndicators:
    """Tests that help identify scaling issues."""

    def test_document_processing_scales_linearly(self):
        """
        Processing time should scale roughly linearly with document count.

        This catches O(n^2) issues in document processing.
        """
        from cortical import CorticalTextProcessor
        from tests.fixtures.small_corpus import SMALL_CORPUS_DOCS

        docs = list(SMALL_CORPUS_DOCS.items())

        # Warmup run to trigger any lazy imports/JIT compilation
        warmup = CorticalTextProcessor()
        for doc_id, content in docs[:3]:
            warmup.process_document(doc_id, content)
        del warmup

        # Time processing 5 docs
        processor1 = CorticalTextProcessor()
        start = time.perf_counter()
        for doc_id, content in docs[:5]:
            processor1.process_document(doc_id, content)
        time_5_docs = time.perf_counter() - start

        # Time processing 15 docs
        processor2 = CorticalTextProcessor()
        start = time.perf_counter()
        for doc_id, content in docs[:15]:
            processor2.process_document(doc_id, content)
        time_15_docs = time.perf_counter() - start

        # 15 docs should take roughly 3x time of 5 docs (linear scaling)
        # Allow 10x to account for CI variability, cold start overhead, and GC
        # Use a minimum floor for baseline to avoid issues when 5 docs is extremely fast
        baseline = max(time_5_docs, 0.02)  # At least 20ms floor
        expected_max = baseline * 10

        assert time_15_docs < expected_max, (
            f"Processing 15 docs took {time_15_docs:.3f}s, "
            f"but 5 docs took {time_5_docs:.3f}s (baseline: {baseline:.3f}s). "
            f"Expected roughly linear scaling (< {expected_max:.3f}s). "
            f"Possible O(n^2) issue in document processing."
        )

    def test_search_time_stable_across_queries(self, small_processor):
        """
        Search time should be stable regardless of query complexity.

        Large variance might indicate pathological cases.
        """
        queries = [
            "a",  # Very short
            "machine learning neural networks",  # Multiple terms
            "xyzzy_unknown_term",  # Unknown term
            "database indexing optimization performance",  # Many terms
        ]

        times = []
        for query in queries:
            start = time.perf_counter()
            small_processor.find_documents_for_query(query, top_n=5)
            times.append(time.perf_counter() - start)

        # Check variance isn't too high (no query should be 10x slower)
        min_time = min(times)
        max_time = max(times)

        assert max_time < min_time * 10 or max_time < 0.5, (
            f"Search time variance too high: min={min_time:.3f}s, max={max_time:.3f}s. "
            f"Query times: {[f'{t:.3f}s' for t in times]}"
        )
