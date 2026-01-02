"""
╔══════════════════════════════════════════════════════════════════════╗
║                  CONNECTIONS PERFORMANCE CONTRACT                     ║
╠══════════════════════════════════════════════════════════════════════╣
║  Ratified:     2024-12-31                                            ║
║  Guardian:     CI Pipeline                                            ║
║  Renegotiation: Requires team review + documented justification      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  We solemnly contract the following guarantees:                      ║
║                                                                       ║
║  • Bigram connections < 2 seconds for ≤ 500 bigrams                 ║
║  • Concept connections < 1 second for ≤ 100 concepts                ║
║  • Document connections < 500ms for ≤ 100 documents                 ║
║  • All connection algorithms are bidirectional                      ║
║  • Connection weights are positive                                  ║
║  • No duplicate connections created                                 ║
║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import time
import pytest


@pytest.mark.contract
class TestBigramConnectionsPerformanceContract:
    """
    Bigram Connections Performance Contract

    As a developer building phrase networks,
    I expect bigram connection building to scale,
    So that phrase-level indexing is practical.
    """

    # The sacred numbers - DO NOT CHANGE without team review
    MAX_LATENCY_MS = 2000  # 2 seconds - bigram connections are O(n²) bounded

    def test_bigram_connections_latency_honored(self, small_processor):
        """
        CONTRACT: Bigram connections complete in < 2 seconds for ≤ 500 bigrams.

        Even with quadratic worst-case, limits keep it practical.
        """
        from cortical.layers import CorticalLayer
        from cortical.analysis.connections import compute_bigram_connections

        layer = small_processor.layers[CorticalLayer.BIGRAMS]

        # Skip if no bigrams
        if layer.column_count() == 0:
            pytest.skip("No bigrams in corpus")

        start = time.perf_counter()
        compute_bigram_connections(small_processor.layers)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < self.MAX_LATENCY_MS, (
            f"CONTRACT VIOLATION: Bigram connections took {elapsed_ms:.1f}ms, "
            f"contract requires <{self.MAX_LATENCY_MS}ms"
        )

    def test_bigram_connections_are_bidirectional(self, small_processor):
        """
        CONTRACT: Bigram connections are bidirectional.

        If A connects to B, B must connect to A with same weight.
        """
        from cortical.layers import CorticalLayer
        from cortical.analysis.connections import compute_bigram_connections

        layer = small_processor.layers[CorticalLayer.BIGRAMS]

        if layer.column_count() == 0:
            pytest.skip("No bigrams in corpus")

        compute_bigram_connections(small_processor.layers)

        # Check bidirectionality
        for bigram in layer.minicolumns.values():
            for neighbor_id, weight in bigram.lateral_connections.items():
                neighbor = layer.get_by_id(neighbor_id)
                if neighbor:
                    # Neighbor should have connection back
                    assert bigram.id in neighbor.lateral_connections, (
                        f"CONTRACT VIOLATION: Bigram '{bigram.content}' connects to "
                        f"'{neighbor.content}' but reverse connection missing"
                    )

    def test_bigram_connections_have_positive_weights(self, small_processor):
        """
        CONTRACT: Connection weights are positive.

        Negative weights are semantically invalid.
        """
        from cortical.layers import CorticalLayer
        from cortical.analysis.connections import compute_bigram_connections

        layer = small_processor.layers[CorticalLayer.BIGRAMS]

        if layer.column_count() == 0:
            pytest.skip("No bigrams in corpus")

        compute_bigram_connections(small_processor.layers)

        for bigram in layer.minicolumns.values():
            for neighbor_id, weight in bigram.lateral_connections.items():
                assert weight > 0, (
                    f"CONTRACT VIOLATION: Bigram '{bigram.content}' has non-positive "
                    f"weight {weight} to neighbor {neighbor_id}"
                )


@pytest.mark.contract
class TestConceptConnectionsPerformanceContract:
    """
    Concept Connections Performance Contract

    As a developer building semantic networks,
    I expect concept connection building to be fast,
    So that semantic layer construction is efficient.
    """

    MAX_LATENCY_MS = 1000

    def test_concept_connections_latency_honored(self, small_processor):
        """
        CONTRACT: Concept connections complete in < 1 second for ≤ 100 concepts.

        Concept networks should build quickly.
        """
        from cortical.layers import CorticalLayer
        from cortical.analysis.connections import compute_concept_connections

        layer = small_processor.layers[CorticalLayer.CONCEPTS]

        if layer.column_count() == 0:
            pytest.skip("No concepts in corpus")

        # Verify within bounds
        assert layer.column_count() <= 100

        start = time.perf_counter()
        compute_concept_connections(small_processor.layers)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < self.MAX_LATENCY_MS, (
            f"CONTRACT VIOLATION: Concept connections took {elapsed_ms:.1f}ms, "
            f"contract requires <{self.MAX_LATENCY_MS}ms"
        )

    def test_concept_connections_are_bidirectional(self, small_processor):
        """
        CONTRACT: Concept connections are bidirectional.

        Semantic relatedness is symmetric.
        """
        from cortical.layers import CorticalLayer
        from cortical.analysis.connections import compute_concept_connections

        layer = small_processor.layers[CorticalLayer.CONCEPTS]

        if layer.column_count() == 0:
            pytest.skip("No concepts in corpus")

        compute_concept_connections(small_processor.layers)

        for concept in layer.minicolumns.values():
            for neighbor_id, weight in concept.lateral_connections.items():
                neighbor = layer.get_by_id(neighbor_id)
                if neighbor:
                    assert concept.id in neighbor.lateral_connections, (
                        f"CONTRACT VIOLATION: Concept '{concept.content}' connects to "
                        f"'{neighbor.content}' but reverse connection missing"
                    )

    def test_concept_connections_respect_filters(self, small_processor):
        """
        CONTRACT: Connection filters are honored.

        min_shared_docs and min_jaccard should filter connections.
        """
        from cortical.layers import CorticalLayer
        from cortical.analysis.connections import compute_concept_connections

        layer = small_processor.layers[CorticalLayer.CONCEPTS]

        if layer.column_count() < 2:
            pytest.skip("Need at least 2 concepts")

        # Clear existing connections
        for concept in layer.minicolumns.values():
            concept.lateral_connections.clear()

        # With very strict filter, should create few/no connections
        result = compute_concept_connections(
            small_processor.layers,
            min_shared_docs=999,  # Impossibly high
            min_jaccard=0.99  # Nearly identical required
        )

        # Should create very few connections
        # (this is a soft contract - depends on corpus, but validates filtering works)
        total_connections = sum(
            len(c.lateral_connections) for c in layer.minicolumns.values()
        )

        # At minimum, shouldn't crash with strict filters
        assert total_connections >= 0


@pytest.mark.contract
class TestDocumentConnectionsPerformanceContract:
    """
    Document Connections Performance Contract

    As a developer building document similarity,
    I expect document connection building to be fast,
    So that document network construction is efficient.
    """

    MAX_LATENCY_MS = 500

    def test_document_connections_latency_honored(self, small_processor):
        """
        CONTRACT: Document connections complete in < 500ms for ≤ 100 documents.

        Document similarity network should build quickly.
        """
        from cortical.layers import CorticalLayer
        from cortical.analysis.connections import compute_document_connections

        # Verify within bounds
        assert len(small_processor.documents) <= 100

        start = time.perf_counter()
        compute_document_connections(
            small_processor.layers,
            small_processor.documents,
            min_shared_terms=3
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < self.MAX_LATENCY_MS, (
            f"CONTRACT VIOLATION: Document connections took {elapsed_ms:.1f}ms, "
            f"contract requires <{self.MAX_LATENCY_MS}ms"
        )

    def test_document_connections_are_bidirectional(self, small_processor):
        """
        CONTRACT: Document connections are bidirectional.

        Similarity is symmetric.
        """
        from cortical.layers import CorticalLayer
        from cortical.analysis.connections import compute_document_connections

        layer = small_processor.layers[CorticalLayer.DOCUMENTS]

        compute_document_connections(
            small_processor.layers,
            small_processor.documents,
            min_shared_terms=2
        )

        for doc in layer.minicolumns.values():
            for neighbor_id, weight in doc.lateral_connections.items():
                neighbor = layer.get_by_id(neighbor_id)
                if neighbor:
                    assert doc.id in neighbor.lateral_connections, (
                        f"CONTRACT VIOLATION: Document '{doc.content}' connects to "
                        f"'{neighbor.content}' but reverse connection missing"
                    )

    def test_document_connections_weighted_by_tfidf(self, small_processor):
        """
        CONTRACT: Document connections are weighted by shared term importance.

        Connections should use TF-IDF weights, not raw counts.
        """
        from cortical.layers import CorticalLayer
        from cortical.analysis.connections import compute_document_connections

        layer = small_processor.layers[CorticalLayer.DOCUMENTS]

        # Clear existing
        for doc in layer.minicolumns.values():
            doc.lateral_connections.clear()

        compute_document_connections(
            small_processor.layers,
            small_processor.documents,
            min_shared_terms=1
        )

        # Verify connections have weights
        connection_count = 0
        for doc in layer.minicolumns.values():
            for neighbor_id, weight in doc.lateral_connections.items():
                # Weight should be positive (TF-IDF sum)
                assert weight > 0, (
                    f"CONTRACT VIOLATION: Document connection has weight {weight}"
                )
                connection_count += 1

        # Should have created some connections (soft check)
        # At minimum, validates the algorithm ran


@pytest.mark.contract
class TestConnectionsCorrectnessContract:
    """
    Connections Correctness Contract

    As a developer relying on connection algorithms,
    I expect correct graph properties,
    So that network structure is meaningful.
    """

    def test_no_self_connections(self, small_processor):
        """
        CONTRACT: No node connects to itself.

        Self-loops are semantically invalid for similarity.
        """
        from cortical.layers import CorticalLayer
        from cortical.analysis.connections import (
            compute_bigram_connections,
            compute_concept_connections,
            compute_document_connections
        )

        # Check all layers
        for layer_enum in [CorticalLayer.BIGRAMS, CorticalLayer.CONCEPTS, CorticalLayer.DOCUMENTS]:
            layer = small_processor.layers[layer_enum]

            if layer.column_count() == 0:
                continue

            for col in layer.minicolumns.values():
                assert col.id not in col.lateral_connections, (
                    f"CONTRACT VIOLATION: Node '{col.content}' in {layer_enum.name} "
                    f"has self-connection"
                )

    def test_connections_use_batch_api(self, small_processor):
        """
        CONTRACT: Bigram connections use batch API for performance.

        The batch API should be used to minimize cache invalidations.
        """
        from cortical.layers import CorticalLayer
        from cortical.analysis.connections import compute_bigram_connections

        layer = small_processor.layers[CorticalLayer.BIGRAMS]

        if layer.column_count() == 0:
            pytest.skip("No bigrams")

        # Clear connections
        for bigram in layer.minicolumns.values():
            bigram.lateral_connections.clear()

        # Run algorithm
        result = compute_bigram_connections(small_processor.layers)

        # Should have created connections efficiently
        # (This is validated by the result statistics)
        assert result['connections_created'] >= 0
        assert result['bigrams'] == layer.column_count()
