"""
Performance Regression Tests for Session 2025-12-29 Optimizations

These tests specifically guard against reintroducing performance bugs
that were fixed in the performance tuning session.

Optimizations protected:
1. Fast search O(n) removal - fast_find_documents must be faster than standard
2. Lazy Edge loading - Edge objects not created during from_dict
3. In-place Edge updates - weight updates don't create new objects

Reference:
- Baseline: benchmarks/results/baseline-real-corpus.json
- Knowledge transfer: samples/memories/2025-12-29-session-knowledge-transfer-performance-tuning-complete.md
- Commits: e9c183a1 (fast search), a9256211 (lazy Edge), be5c40f0 (in-place Edge)
"""

import time
import pytest


class TestFastSearchRegression:
    """
    Guard against O(n) document scan regression in fast_find_documents.

    Bug history:
    - Before fix: fast_speedup=0.139 (7x SLOWER than standard)
    - After fix: fast_speedup=34.08 (34x FASTER than standard)
    - Root cause: O(n) loop at lines 211-224 in search.py iterating ALL documents

    This test ensures fast_find_documents is actually faster, not slower.
    """

    def test_fast_search_must_be_faster_than_standard(self, small_processor):
        """
        fast_find_documents MUST be faster than find_documents_for_query.

        The entire purpose of fast search is to be faster by avoiding
        expensive operations. If it's slower, something is wrong.

        Note: This is stricter than the existing test which allowed 1.5x slower.
        """
        queries = [
            "machine learning",
            "database indexing",
            "neural networks",
            "sorting algorithms",
        ]

        # Warmup
        for query in queries:
            small_processor.find_documents_for_query(query, top_n=5)
            small_processor.fast_find_documents(query, top_n=5)

        standard_times = []
        fast_times = []

        # Run multiple queries to get stable measurements
        for query in queries:
            # Time standard search
            start = time.perf_counter()
            for _ in range(5):
                small_processor.find_documents_for_query(query, top_n=5)
            standard_times.append(time.perf_counter() - start)

            # Time fast search
            start = time.perf_counter()
            for _ in range(5):
                small_processor.fast_find_documents(query, top_n=5)
            fast_times.append(time.perf_counter() - start)

        total_standard = sum(standard_times)
        total_fast = sum(fast_times)

        # Fast search must not be slower than standard
        # We use 1.1x (10% margin) for CI variability, not 1.5x
        assert total_fast <= total_standard * 1.1, (
            f"REGRESSION: fast_find_documents ({total_fast:.4f}s) is slower than "
            f"find_documents_for_query ({total_standard:.4f}s). "
            f"This indicates the O(n) document scan may have been reintroduced. "
            f"Check cortical/query/search.py for O(n) loops in fast_find_documents."
        )


class TestLazyEdgeLoadingRegression:
    """
    Guard against eager Edge object creation during corpus load.

    Bug history:
    - Before fix: 17.5s spent creating 4.1M Edge objects during load
    - After fix: Edge objects created lazily on first property access
    - Result: 22% faster load time

    This test ensures Edge objects are not created during from_dict.
    """

    def test_minicolumn_from_dict_defers_edge_creation(self):
        """
        Minicolumn.from_dict should NOT create Edge objects immediately.

        The typed_connections should be stored as raw dict until first access.
        """
        from cortical.minicolumn import Minicolumn, Edge
        from cortical.layers import CorticalLayer

        # Create dict with typed_connections (simulating JSON load)
        # Note: Edge.to_dict() includes target_id redundantly
        data = {
            "id": "test_col",
            "content": "test",
            "layer": CorticalLayer.TOKENS.value,
            "activation": 0.0,
            "occurrence_count": 1,
            "document_ids": [],
            "lateral_connections": {},
            "typed_connections": {
                "target1": {"target_id": "target1", "weight": 1.0, "relation_type": "cooccurs", "confidence": 1.0, "source": "test"},
                "target2": {"target_id": "target2", "weight": 2.0, "relation_type": "cooccurs", "confidence": 1.0, "source": "test"},
                "target3": {"target_id": "target3", "weight": 3.0, "relation_type": "cooccurs", "confidence": 1.0, "source": "test"},
            },
            "feedforward_sources": {},
            "feedforward_connections": {},
            "feedback_connections": {},
            "tfidf": 0.0,
            "tfidf_per_doc": {},
            "pagerank": 0.0,
            "cluster_id": None,
            "doc_occurrence_counts": {},
            "name_tokens": None,
        }

        # Load from dict
        col = Minicolumn.from_dict(data)

        # The raw data should be stored, not converted to Edge objects yet
        assert col._typed_connections_raw is not None, (
            "REGRESSION: typed_connections were converted to Edge objects during from_dict. "
            "This defeats lazy loading optimization. Edge creation should be deferred "
            "until first property access."
        )

        # Now access the property - this should trigger conversion
        _ = col.typed_connections

        # After access, raw should be cleared
        assert col._typed_connections_raw is None, (
            "Raw typed_connections should be cleared after lazy conversion"
        )


class TestInPlaceEdgeUpdateRegression:
    """
    Guard against creating new Edge objects on weight updates.

    Bug history:
    - Before fix: Every weight update created a new Edge object
    - After fix: Edge.weight modified in place
    - Result: 51% faster compute_all

    This test ensures weight updates modify in place, not create new objects.
    """

    def test_add_lateral_connection_updates_in_place(self):
        """
        add_lateral_connection should modify existing Edge weight in place,
        not create a new Edge object.
        """
        from cortical.minicolumn import Minicolumn
        from cortical.layers import CorticalLayer

        col = Minicolumn(
            id="test",
            content="test",
            layer=CorticalLayer.TOKENS
        )

        # Add initial connection
        col.add_lateral_connection("target", weight=1.0)

        # Get reference to the Edge object
        original_edge = col.typed_connections["target"]
        original_id = id(original_edge)

        # Update the connection
        col.add_lateral_connection("target", weight=2.0)

        # Should be same object with updated weight
        updated_edge = col.typed_connections["target"]

        assert id(updated_edge) == original_id, (
            "REGRESSION: add_lateral_connection created a new Edge object instead of "
            "updating in place. This defeats the in-place update optimization. "
            f"Original Edge id: {original_id}, Updated Edge id: {id(updated_edge)}"
        )

        assert updated_edge.weight == 3.0, (
            f"Weight should be 1.0 + 2.0 = 3.0, got {updated_edge.weight}"
        )

    def test_add_lateral_connections_batch_updates_in_place(self):
        """
        add_lateral_connections_batch should modify existing Edge weights in place.
        """
        from cortical.minicolumn import Minicolumn
        from cortical.layers import CorticalLayer

        col = Minicolumn(
            id="test",
            content="test",
            layer=CorticalLayer.TOKENS
        )

        # Add initial connections (API takes Dict[str, float])
        col.add_lateral_connections_batch({
            "target1": 1.0,
            "target2": 1.0,
        })

        # Get references
        original_edge1 = col.typed_connections["target1"]
        original_id1 = id(original_edge1)

        # Batch update
        col.add_lateral_connections_batch({
            "target1": 2.0,  # Update existing
            "target3": 1.0,  # New connection
        })

        updated_edge1 = col.typed_connections["target1"]

        assert id(updated_edge1) == original_id1, (
            "REGRESSION: add_lateral_connections_batch created new Edge objects "
            "instead of updating in place."
        )

        assert updated_edge1.weight == 3.0, (
            f"Weight should be 1.0 + 2.0 = 3.0, got {updated_edge1.weight}"
        )


class TestNoOnDocumentScan:
    """
    Guard against O(n) document scans in fast_find_documents.

    This is a code-level check to ensure the O(n) pattern doesn't return.
    """

    def test_no_all_documents_iteration_in_fast_search(self):
        """
        fast_find_documents should NOT iterate over all documents.

        The previous bug iterated over ALL documents (layer3.minicolumns.values())
        to check for name matches, negating the performance benefit.
        """
        import inspect
        from cortical.query.search import fast_find_documents

        source = inspect.getsource(fast_find_documents)

        # These patterns indicate O(n) document iteration
        dangerous_patterns = [
            "for doc_col in layer3.minicolumns.values()",
            "for doc in layer3.minicolumns.values()",
            "for col in layer3.minicolumns.values()",
        ]

        for pattern in dangerous_patterns:
            assert pattern not in source, (
                f"REGRESSION: Found O(n) document iteration pattern in fast_find_documents:\n"
                f"  Pattern: {pattern}\n"
                f"This was the root cause of the 7x slowdown. "
                f"See commit e9c183a1 for the original fix."
            )
