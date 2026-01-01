"""
╔══════════════════════════════════════════════════════════════════════╗
║                   MINICOLUMN PERFORMANCE CONTRACT                     ║
╠══════════════════════════════════════════════════════════════════════╣
║  Ratified:     2024-12-31                                            ║
║  Guardian:     CI Pipeline                                            ║
║  Renegotiation: Requires team review + documented justification      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  We solemnly contract the following guarantees:                      ║
║                                                                       ║
║  • Add single lateral connection < 10μs                             ║
║  • Add 1000 connections (batch) < 1ms                               ║
║  • Serialize minicolumn < 1ms                                       ║
║  • Deserialize minicolumn < 2ms                                     ║
║  • Lazy loading of typed connections is correct                    ║
║  • Connection weights are preserved in serialization                ║
║  • Memory usage scales linearly with connections                    ║
║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import time
import pytest
from cortical.minicolumn import Minicolumn, Edge


@pytest.mark.contract
class TestMinicolumnConnectionPerformanceContract:
    """
    Minicolumn Connection Performance Contract

    As a developer building cortical networks,
    I expect connection operations to be fast,
    So that graph construction scales.
    """

    MAX_SINGLE_CONNECTION_US = 10
    MAX_BATCH_1K_CONNECTIONS_MS = 1.0

    def test_add_lateral_connection_latency(self):
        """
        CONTRACT: Add lateral connection in < 10μs.

        Connection building creates millions of edges.
        """
        col = Minicolumn("L0_test", "test", 0)

        iterations = 10000

        start = time.perf_counter()
        for i in range(iterations):
            col.add_lateral_connection(f"L0_target_{i}", 0.5)
        elapsed_us = (time.perf_counter() - start) * 1_000_000

        avg_us = elapsed_us / iterations

        assert avg_us < self.MAX_SINGLE_CONNECTION_US, (
            f"CONTRACT VIOLATION: Adding connection took {avg_us:.2f}μs on average, "
            f"contract requires <{self.MAX_SINGLE_CONNECTION_US}μs"
        )

    def test_add_batch_connections_latency(self):
        """
        CONTRACT: Add 1000 connections (batch) in < 1ms.

        Batch API must be significantly faster than individual adds.
        """
        col = Minicolumn("L0_test", "test", 0)

        # Create batch
        connections = {f"L0_target_{i}": 0.5 for i in range(1000)}

        start = time.perf_counter()
        col.add_lateral_connections_batch(connections)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < self.MAX_BATCH_1K_CONNECTIONS_MS, (
            f"CONTRACT VIOLATION: Batch adding 1000 connections took {elapsed_ms:.2f}ms, "
            f"contract requires <{self.MAX_BATCH_1K_CONNECTIONS_MS}ms"
        )

    def test_connection_weights_accumulate(self):
        """
        CONTRACT: Adding to existing connection accumulates weight.

        Connection weights represent accumulated evidence.
        """
        col = Minicolumn("L0_test", "test", 0)

        target = "L0_target"
        col.add_lateral_connection(target, 0.5)
        col.add_lateral_connection(target, 0.3)

        weight = col.lateral_connections[target]

        assert weight == pytest.approx(0.8, rel=1e-6), (
            f"CONTRACT VIOLATION: Weight {weight} ≠ 0.8 (expected accumulation)"
        )


@pytest.mark.contract
class TestMinicolumnSerializationContract:
    """
    Minicolumn Serialization Performance Contract

    As a developer persisting cortical state,
    I expect serialization to be fast and correct,
    So that save/load operations scale.
    """

    MAX_SERIALIZE_MS = 1.0
    MAX_DESERIALIZE_MS = 2.0

    def test_serialize_minicolumn_latency(self):
        """
        CONTRACT: Serialize minicolumn in < 1ms.

        Serializing thousands of minicolumns must be fast.
        """
        # Create minicolumn with connections
        col = Minicolumn("L0_neural", "neural", 0)
        col.occurrence_count = 100
        col.pagerank = 0.5
        col.tfidf = 1.2
        col.activation = 0.8

        # Add connections
        for i in range(100):
            col.add_lateral_connection(f"L0_word_{i}", 0.5)

        # Add typed connections
        col.add_typed_connection("L0_network", 0.8, relation_type='RelatedTo')

        start = time.perf_counter()
        data = col.to_dict()
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < self.MAX_SERIALIZE_MS, (
            f"CONTRACT VIOLATION: Serializing minicolumn took {elapsed_ms:.2f}ms, "
            f"contract requires <{self.MAX_SERIALIZE_MS}ms"
        )

        # Verify structure
        assert 'id' in data
        assert 'content' in data
        assert 'lateral_connections' in data

    def test_deserialize_minicolumn_latency(self):
        """
        CONTRACT: Deserialize minicolumn in < 2ms.

        Loading corpus state must be fast.
        """
        # Create test data
        data = {
            'id': 'L0_neural',
            'content': 'neural',
            'layer': 0,
            'occurrence_count': 100,
            'pagerank': 0.5,
            'tfidf': 1.2,
            'activation': 0.8,
            'document_ids': ['doc1', 'doc2'],
            'lateral_connections': {f'L0_word_{i}': 0.5 for i in range(100)},
            'typed_connections': {
                'L0_network': {
                    'target_id': 'L0_network',
                    'weight': 0.8,
                    'relation_type': 'RelatedTo',
                    'confidence': 1.0,
                    'source': 'corpus'
                }
            },
            'feedforward_connections': {},
            'feedback_connections': {},
            'tfidf_per_doc': {},
            'doc_occurrence_counts': {},
        }

        start = time.perf_counter()
        col = Minicolumn.from_dict(data)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < self.MAX_DESERIALIZE_MS, (
            f"CONTRACT VIOLATION: Deserializing minicolumn took {elapsed_ms:.2f}ms, "
            f"contract requires <{self.MAX_DESERIALIZE_MS}ms"
        )

        # Verify correctness
        assert col.id == 'L0_neural'
        assert col.content == 'neural'
        assert col.occurrence_count == 100

    def test_serialization_roundtrip_preserves_data(self):
        """
        CONTRACT: Serialization roundtrip preserves all data.

        No data loss in save/load cycle.
        """
        # Create original
        col = Minicolumn("L0_neural", "neural", 0)
        col.occurrence_count = 42
        col.pagerank = 0.75
        col.tfidf = 2.5
        col.activation = 0.9
        col.cluster_id = 3
        col.document_ids = {'doc1', 'doc2', 'doc3'}

        col.add_lateral_connection("L0_network", 0.8)
        col.add_typed_connection("L0_brain", 0.9, relation_type='IsA', confidence=0.95)
        col.add_feedforward_connection("L1_neural_network", 1.0)

        # Roundtrip
        data = col.to_dict()
        restored = Minicolumn.from_dict(data)

        # Verify all fields
        assert restored.id == col.id
        assert restored.content == col.content
        assert restored.occurrence_count == col.occurrence_count
        assert restored.pagerank == pytest.approx(col.pagerank)
        assert restored.tfidf == pytest.approx(col.tfidf)
        assert restored.activation == pytest.approx(col.activation)
        assert restored.cluster_id == col.cluster_id
        assert restored.document_ids == col.document_ids

        # Verify connections
        assert "L0_network" in restored.lateral_connections
        assert restored.lateral_connections["L0_network"] == pytest.approx(0.8)


@pytest.mark.contract
class TestMinicolumnLazyLoadingContract:
    """
    Minicolumn Lazy Loading Contract

    As a developer loading large corpora,
    I expect lazy loading to improve load times,
    So that startup is fast.
    """

    def test_lazy_loading_defers_edge_creation(self):
        """
        CONTRACT: Lazy loading defers Edge object creation.

        Edge objects are only created when accessed.
        """
        # Create data with typed connections
        data = {
            'id': 'L0_neural',
            'content': 'neural',
            'layer': 0,
            'occurrence_count': 10,
            'document_ids': [],
            'lateral_connections': {},
            'typed_connections': {
                f'L0_word_{i}': {
                    'target_id': f'L0_word_{i}',
                    'weight': 0.5,
                    'relation_type': 'co_occurrence',
                    'confidence': 1.0,
                    'source': 'corpus'
                }
                for i in range(100)
            },
            'feedforward_connections': {},
            'feedback_connections': {},
            'tfidf_per_doc': {},
            'doc_occurrence_counts': {},
        }

        # Deserialize (should be fast - no Edge objects created yet)
        start = time.perf_counter()
        col = Minicolumn.from_dict(data)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Should be faster than creating 100 Edge objects
        assert elapsed_ms < 1.0, (
            f"CONTRACT VIOLATION: Lazy loading took {elapsed_ms:.2f}ms, "
            f"should be <1ms (Edge objects should not be created yet)"
        )

        # Now access typed_connections (triggers lazy load)
        typed = col.typed_connections

        # Should now have Edge objects
        assert len(typed) == 100
        assert isinstance(typed['L0_word_0'], Edge)

    def test_lazy_loading_correctness(self):
        """
        CONTRACT: Lazy loading produces correct Edge objects.

        Lazy-loaded edges must match eagerly-created edges.
        """
        data = {
            'id': 'L0_test',
            'content': 'test',
            'layer': 0,
            'occurrence_count': 1,
            'document_ids': [],
            'lateral_connections': {},
            'typed_connections': {
                'L0_target': {
                    'target_id': 'L0_target',
                    'weight': 0.8,
                    'relation_type': 'RelatedTo',
                    'confidence': 0.9,
                    'source': 'semantic'
                }
            },
            'feedforward_connections': {},
            'feedback_connections': {},
            'tfidf_per_doc': {},
            'doc_occurrence_counts': {},
        }

        col = Minicolumn.from_dict(data)
        edge = col.typed_connections['L0_target']

        assert edge.target_id == 'L0_target'
        assert edge.weight == pytest.approx(0.8)
        assert edge.relation_type == 'RelatedTo'
        assert edge.confidence == pytest.approx(0.9)
        assert edge.source == 'semantic'


@pytest.mark.contract
class TestMinicolumnMemoryContract:
    """
    Minicolumn Memory Usage Contract

    As a developer working with large graphs,
    I expect memory usage to scale linearly,
    So that memory doesn't explode with connections.
    """

    def test_connection_memory_scales_linearly(self):
        """
        CONTRACT: Memory usage scales linearly with connections.

        Doubling connections should roughly double memory.
        """
        import sys

        # Create with 100 connections
        col1 = Minicolumn("L0_test1", "test1", 0)
        for i in range(100):
            col1.add_lateral_connection(f"L0_target_{i}", 0.5)
        size1 = sys.getsizeof(col1.lateral_connections)

        # Create with 200 connections
        col2 = Minicolumn("L0_test2", "test2", 0)
        for i in range(200):
            col2.add_lateral_connection(f"L0_target_{i}", 0.5)
        size2 = sys.getsizeof(col2.lateral_connections)

        # Ratio should be close to 2.0 (within 50% tolerance)
        ratio = size2 / size1 if size1 > 0 else 0

        assert 1.5 <= ratio <= 2.5, (
            f"CONTRACT VIOLATION: Memory doesn't scale linearly. "
            f"Ratio {ratio:.2f} (expected ~2.0)"
        )
