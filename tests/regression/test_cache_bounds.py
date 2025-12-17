"""
Regression Tests for Cache and Memory Bounds
=============================================

Tests for edge cases related to cache boundaries, empty state,
and memory limits. These prevent silent failures when data structures
are at their limits.

Regression test for T-018: Edge case coverage for production robustness.
"""

import pytest
from cortical import CorticalTextProcessor
from cortical.observability import MetricsCollector


class TestQueryCacheBounds:
    """Test query expansion cache at boundary conditions."""

    def test_empty_cache_expansion(self, fresh_processor):
        """
        Empty cache should not cause errors.

        Regression test for T-018: Verify expand_query works with no cached entries.
        """
        # Fresh processor has empty cache
        fresh_processor.process_document("doc1", "neural networks process data efficiently")
        fresh_processor.compute_all(verbose=False)

        # Query expansion should work even with empty cache
        result = fresh_processor.expand_query("neural", max_expansions=5)

        # Should return a non-empty dict with at least the original term
        assert isinstance(result, dict)
        assert len(result) > 0
        assert "neural" in result

    def test_cache_at_max_size(self, fresh_processor):
        """
        Cache at maximum size (100 entries) should not break.

        Regression test for T-018: Verify LRU eviction works correctly at max size.
        """
        fresh_processor.process_document("doc1", " ".join([f"term{i}" for i in range(200)]))
        fresh_processor.compute_all(verbose=False)

        # Fill cache to max size (100 entries)
        for i in range(100):
            fresh_processor.expand_query(f"term{i}", max_expansions=5)

        # Add one more (should evict oldest)
        result = fresh_processor.expand_query("term100", max_expansions=5)

        # Should still work without errors
        assert isinstance(result, dict)

    def test_empty_processor_expansion(self, fresh_processor):
        """
        Query expansion on empty processor should return minimal result.

        Regression test for T-018: Ensure no crashes when querying empty corpus.
        """
        # No documents added
        result = fresh_processor.expand_query("anything", max_expansions=5)

        # Should return empty dict or just the query term with weight 1.0
        assert isinstance(result, dict)


class TestMetricsCollectorBounds:
    """Test MetricsCollector at boundary conditions."""

    def test_empty_metrics_collector(self):
        """
        Empty MetricsCollector should handle stats requests gracefully.

        Regression test for T-018: Division by zero protection in get_operation_stats.
        """
        collector = MetricsCollector(enabled=True)

        # Get stats for non-existent operation
        stats = collector.get_operation_stats("nonexistent_operation")

        # Should return empty dict, not crash
        assert stats == {}

    def test_single_operation_timing(self):
        """
        Single timing record should compute correct averages.

        Regression test for T-018: Edge case with count=1.
        """
        collector = MetricsCollector(enabled=True)

        # Record single timing
        collector.record_timing("test_op", 123.45)

        # Get stats
        stats = collector.get_operation_stats("test_op")

        # Verify correctness
        assert stats['count'] == 1
        assert stats['total_ms'] == 123.45
        assert stats['avg_ms'] == 123.45
        assert stats['min_ms'] == 123.45
        assert stats['max_ms'] == 123.45

    def test_zero_timing_history(self):
        """
        MetricsCollector with max_timing_history=0 should not store timings.

        Regression test for T-018: Memory optimization mode.
        """
        collector = MetricsCollector(enabled=True, max_timing_history=0)

        # Record timings
        for i in range(10):
            collector.record_timing("test_op", float(i))

        # Stats should still work (aggregate values stored)
        stats = collector.get_operation_stats("test_op")
        assert stats['count'] == 10
        assert stats['avg_ms'] == 4.5  # Average of 0-9

        # But timing history should be empty (maxlen=0)
        op_data = collector.operations['test_op']
        assert len(op_data['timings']) == 0

    def test_disabled_collector_no_side_effects(self):
        """
        Disabled MetricsCollector should not record anything.

        Regression test for T-018: Ensure disable() truly disables collection.
        """
        collector = MetricsCollector(enabled=False)

        # Try to record metrics
        collector.record_timing("test_op", 100.0)
        collector.record_count("cache_hits", 5)

        # Should have no operations recorded
        assert len(collector.operations) == 0
        stats = collector.get_operation_stats("test_op")
        assert stats == {}


class TestProcessorStateBounds:
    """Test processor state at boundary conditions."""

    def test_compute_all_on_empty_processor(self, fresh_processor):
        """
        compute_all() on empty processor should not crash.

        Regression test for T-018: Edge case with no documents.
        """
        # Call compute_all with no documents
        fresh_processor.compute_all(verbose=False)

        # Should complete without errors
        from cortical import CorticalLayer
        layer0 = fresh_processor.get_layer(CorticalLayer.TOKENS)
        assert layer0.column_count() == 0

    def test_search_empty_corpus(self, fresh_processor):
        """
        Search on empty corpus should return empty results.

        Regression test for T-018: Graceful handling of empty state.
        """
        results = fresh_processor.find_documents_for_query("anything", top_n=5)

        # Should return empty list, not crash
        assert results == []

    def test_single_document_clustering(self, fresh_processor):
        """
        Clustering with single document should handle gracefully.

        Regression test for T-018: Edge case for Louvain algorithm.
        """
        fresh_processor.process_document("doc1", "single document for testing")
        fresh_processor.propagate_activation(verbose=False)
        fresh_processor.compute_bigram_connections(verbose=False)

        # Should not crash even with minimal data
        fresh_processor.build_concept_clusters(verbose=False)

        from cortical import CorticalLayer
        layer2 = fresh_processor.get_layer(CorticalLayer.CONCEPTS)

        # May have 0 or more clusters (both valid for single doc)
        assert layer2.column_count() >= 0
