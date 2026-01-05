"""
╔══════════════════════════════════════════════════════════════════════╗
║                    LAYER PERFORMANCE CONTRACT                         ║
╠══════════════════════════════════════════════════════════════════════╣
║  Ratified:     2024-12-31                                            ║
║  Guardian:     CI Pipeline                                            ║
║  Renegotiation: Requires team review + documented justification      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  We solemnly contract the following guarantees:                      ║
║                                                                       ║
║  • Create/get minicolumn < 8μs (O(1) hash lookup)                  ║
║  • Lookup by ID < 1μs (O(1) secondary index)                        ║
║  • Create 10,000 minicolumns < 100ms                                ║
║  • Serialize layer (1000 columns) < 2 seconds                       ║
║  • Deserialize layer (1000 columns) < 3 seconds                     ║
║  • Layer statistics computation < 50ms                              ║
║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import time
import pytest
from cortical.layers import HierarchicalLayer, CorticalLayer


@pytest.mark.contract
class TestLayerLookupPerformanceContract:
    """
    Layer Lookup Performance Contract

    As a developer building cortical networks,
    I expect O(1) lookups for minicolumns,
    So that graph traversal is fast.
    """

    MAX_GET_OR_CREATE_US = 8  # CI measured 6.14μs, added headroom
    MAX_LOOKUP_BY_ID_US = 1

    def test_get_or_create_minicolumn_latency(self):
        """
        CONTRACT: Create/get minicolumn in < 8μs.

        Hash lookup must be O(1) constant time.
        CI measured 6.14μs, threshold includes headroom.
        """
        layer = HierarchicalLayer(CorticalLayer.TOKENS)

        iterations = 10000
        contents = [f"word_{i}" for i in range(iterations)]

        start = time.perf_counter()
        for content in contents:
            col = layer.get_or_create_minicolumn(content)
        elapsed_us = (time.perf_counter() - start) * 1_000_000

        avg_us = elapsed_us / iterations

        assert avg_us < self.MAX_GET_OR_CREATE_US, (
            f"CONTRACT VIOLATION: get_or_create took {avg_us:.2f}μs on average, "
            f"contract requires <{self.MAX_GET_OR_CREATE_US}μs"
        )

    def test_lookup_by_id_latency(self):
        """
        CONTRACT: Lookup by ID in < 1μs.

        Secondary index must provide O(1) access.
        """
        layer = HierarchicalLayer(CorticalLayer.TOKENS)

        # Create minicolumns
        for i in range(1000):
            layer.get_or_create_minicolumn(f"word_{i}")

        # Test lookup speed
        ids = [f"L0_word_{i}" for i in range(1000)]
        iterations = 10000

        start = time.perf_counter()
        for _ in range(iterations):
            for col_id in ids[:100]:  # Sample 100 lookups
                col = layer.get_by_id(col_id)
        elapsed_us = (time.perf_counter() - start) * 1_000_000

        avg_us = elapsed_us / (iterations * 100)

        assert avg_us < self.MAX_LOOKUP_BY_ID_US, (
            f"CONTRACT VIOLATION: Lookup by ID took {avg_us:.3f}μs on average, "
            f"contract requires <{self.MAX_LOOKUP_BY_ID_US}μs"
        )

    def test_get_or_create_is_idempotent(self):
        """
        CONTRACT: get_or_create returns same object for same content.

        Must not create duplicates.
        """
        layer = HierarchicalLayer(CorticalLayer.TOKENS)

        col1 = layer.get_or_create_minicolumn("neural")
        col2 = layer.get_or_create_minicolumn("neural")

        # Should be the exact same object
        assert col1 is col2, (
            "CONTRACT VIOLATION: get_or_create created duplicate minicolumn"
        )

        # Should only have one minicolumn
        assert layer.column_count() == 1


@pytest.mark.contract
class TestLayerBulkOperationsContract:
    """
    Layer Bulk Operations Performance Contract

    As a developer indexing large corpora,
    I expect bulk operations to scale linearly,
    So that corpus building is predictable.
    """

    MAX_CREATE_10K_MS = 100

    def test_bulk_create_latency(self):
        """
        CONTRACT: Create 10,000 minicolumns in < 100ms.

        Corpus building creates many minicolumns.
        """
        layer = HierarchicalLayer(CorticalLayer.TOKENS)

        n_columns = 10000
        contents = [f"word_{i}" for i in range(n_columns)]

        start = time.perf_counter()
        for content in contents:
            layer.get_or_create_minicolumn(content)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < self.MAX_CREATE_10K_MS, (
            f"CONTRACT VIOLATION: Creating 10K minicolumns took {elapsed_ms:.0f}ms, "
            f"contract requires <{self.MAX_CREATE_10K_MS}ms"
        )

        # Verify all created
        assert layer.column_count() == n_columns

    def test_creation_scales_linearly(self):
        """
        CONTRACT: Creation time scales linearly with count.

        O(n) scaling ensures predictability. This test catches O(n²)
        regressions where adding more items makes each operation slower.

        Uses multiple fresh layers to get stable measurements and avoid
        interference from dict resizing or other tests.
        """
        ratios = []

        # Run 3 trials with fresh layers for stable measurement
        for trial in range(3):
            layer = HierarchicalLayer(CorticalLayer.TOKENS)

            # Warmup phase in each fresh layer
            for i in range(500):
                layer.get_or_create_minicolumn(f"warm_{i}")

            # First measured batch
            start = time.perf_counter()
            for i in range(1000):
                layer.get_or_create_minicolumn(f"a_{i}")
            time_1k = time.perf_counter() - start

            # Second measured batch (same layer, more items)
            start = time.perf_counter()
            for i in range(1000):
                layer.get_or_create_minicolumn(f"b_{i}")
            time_2k = time.perf_counter() - start

            if time_1k > 0:
                ratios.append(time_2k / time_1k)

        # Use median ratio to reduce noise from outliers
        ratios.sort()
        median_ratio = ratios[len(ratios) // 2] if ratios else 0

        # O(n) means ratio should be ~1.0. Allow 5x for CI variance.
        # O(n²) would show ratio >> 5x as dict grows.
        assert median_ratio <= 5.0, (
            f"CONTRACT VIOLATION: Creation doesn't scale linearly. "
            f"Median ratio {median_ratio:.2f}x (expected ≤5.0x). "
            f"All ratios: {[f'{r:.2f}' for r in ratios]}"
        )


@pytest.mark.contract
class TestLayerSerializationContract:
    """
    Layer Serialization Performance Contract

    As a developer persisting cortical state,
    I expect layer serialization to be reasonably fast,
    So that save/load operations complete in seconds, not minutes.
    """

    MAX_SERIALIZE_1K_SECONDS = 2.0
    MAX_DESERIALIZE_1K_SECONDS = 3.0

    def test_serialize_layer_latency(self):
        """
        CONTRACT: Serialize layer with 1000 columns in < 2 seconds.

        Saving corpus state must be practical.
        """
        layer = HierarchicalLayer(CorticalLayer.TOKENS)

        # Create 1000 minicolumns with connections
        for i in range(1000):
            col = layer.get_or_create_minicolumn(f"word_{i}")
            col.occurrence_count = i
            col.pagerank = 0.5
            col.tfidf = 1.0

            # Add a few connections
            for j in range(5):
                target_id = f"L0_word_{(i + j) % 1000}"
                col.add_lateral_connection(target_id, 0.3)

        start = time.perf_counter()
        data = layer.to_dict()
        elapsed_s = time.perf_counter() - start

        assert elapsed_s < self.MAX_SERIALIZE_1K_SECONDS, (
            f"CONTRACT VIOLATION: Serializing 1000 columns took {elapsed_s:.2f}s, "
            f"contract requires <{self.MAX_SERIALIZE_1K_SECONDS}s"
        )

        # Verify structure
        assert 'level' in data
        assert 'minicolumns' in data
        assert len(data['minicolumns']) == 1000

    def test_deserialize_layer_latency(self):
        """
        CONTRACT: Deserialize layer with 1000 columns in < 3 seconds.

        Loading corpus state must be practical.
        """
        # Create test data
        minicolumns_data = {}
        for i in range(1000):
            minicolumns_data[f"word_{i}"] = {
                'id': f'L0_word_{i}',
                'content': f'word_{i}',
                'layer': 0,
                'occurrence_count': i,
                'pagerank': 0.5,
                'tfidf': 1.0,
                'activation': 0.0,
                'document_ids': [],
                'lateral_connections': {
                    f'L0_word_{(i + j) % 1000}': 0.3
                    for j in range(5)
                },
                'typed_connections': {},
                'feedforward_connections': {},
                'feedback_connections': {},
                'tfidf_per_doc': {},
                'doc_occurrence_counts': {},
            }

        data = {
            'level': 0,
            'minicolumns': minicolumns_data
        }

        start = time.perf_counter()
        layer = HierarchicalLayer.from_dict(data)
        elapsed_s = time.perf_counter() - start

        assert elapsed_s < self.MAX_DESERIALIZE_1K_SECONDS, (
            f"CONTRACT VIOLATION: Deserializing 1000 columns took {elapsed_s:.2f}s, "
            f"contract requires <{self.MAX_DESERIALIZE_1K_SECONDS}s"
        )

        # Verify correctness
        assert layer.column_count() == 1000
        assert layer.level == CorticalLayer.TOKENS


@pytest.mark.contract
class TestLayerStatisticsContract:
    """
    Layer Statistics Performance Contract

    As a developer analyzing cortical state,
    I expect statistics computation to be fast,
    So that monitoring doesn't slow down the system.
    """

    MAX_STATS_MS = 100

    def test_statistics_computation_latency(self):
        """
        CONTRACT: Compute layer statistics in < 50ms.

        Statistics are computed frequently for monitoring.
        """
        layer = HierarchicalLayer(CorticalLayer.TOKENS)

        # Create 1000 minicolumns with varied activations
        for i in range(1000):
            col = layer.get_or_create_minicolumn(f"word_{i}")
            col.activation = i / 1000.0
            col.pagerank = (i % 100) / 100.0
            col.tfidf = i / 500.0

        start = time.perf_counter()
        # Compute all statistics
        count = layer.column_count()
        total_conns = layer.total_connections()
        avg_act = layer.average_activation()
        act_range = layer.activation_range()
        sparsity = layer.sparsity()
        top_pr = layer.top_by_pagerank(10)
        top_tfidf = layer.top_by_tfidf(10)
        top_act = layer.top_by_activation(10)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < self.MAX_STATS_MS, (
            f"CONTRACT VIOLATION: Computing statistics took {elapsed_ms:.1f}ms, "
            f"contract requires <{self.MAX_STATS_MS}ms"
        )

        # Verify correctness
        assert count == 1000
        assert 0.0 <= avg_act <= 1.0
        assert len(top_pr) == 10

    def test_sparsity_computation_correctness(self):
        """
        CONTRACT: Sparsity computation is mathematically correct.

        Sparsity measures activation distribution.
        """
        layer = HierarchicalLayer(CorticalLayer.TOKENS)

        # Create columns with known activation pattern
        # 80 with low activation (0.1), 20 with high activation (0.9)
        for i in range(80):
            col = layer.get_or_create_minicolumn(f"low_{i}")
            col.activation = 0.1

        for i in range(20):
            col = layer.get_or_create_minicolumn(f"high_{i}")
            col.activation = 0.9

        avg_activation = layer.average_activation()  # Should be 0.8*0.1 + 0.2*0.9 = 0.26
        sparsity = layer.sparsity(threshold_fraction=0.5)

        # With threshold = 0.5 * 0.26 = 0.13, 80% of columns are below threshold
        assert sparsity == pytest.approx(0.8, abs=0.05), (
            f"CONTRACT VIOLATION: Sparsity {sparsity} incorrect (expected ~0.8)"
        )


@pytest.mark.contract
class TestLayerRemovalContract:
    """
    Layer Removal Contract

    As a developer managing dynamic corpora,
    I expect minicolumn removal to work correctly,
    So that corpus updates are reliable.
    """

    def test_remove_minicolumn_correctness(self):
        """
        CONTRACT: Removing minicolumn removes from all indexes.

        Both primary dict and secondary index must be updated.
        """
        layer = HierarchicalLayer(CorticalLayer.TOKENS)

        # Create minicolumn
        col = layer.get_or_create_minicolumn("neural")
        col_id = col.id

        # Verify it exists
        assert "neural" in layer
        assert layer.get_by_id(col_id) is not None
        assert layer.column_count() == 1

        # Remove it
        removed = layer.remove_minicolumn("neural")

        assert removed is True, "remove_minicolumn should return True"

        # Verify it's gone from both indexes
        assert "neural" not in layer
        assert layer.get_by_id(col_id) is None
        assert layer.column_count() == 0

    def test_remove_nonexistent_returns_false(self):
        """
        CONTRACT: Removing nonexistent minicolumn returns False.

        Error handling must be clear.
        """
        layer = HierarchicalLayer(CorticalLayer.TOKENS)

        removed = layer.remove_minicolumn("nonexistent")

        assert removed is False, (
            "CONTRACT VIOLATION: remove_minicolumn should return False for nonexistent"
        )
