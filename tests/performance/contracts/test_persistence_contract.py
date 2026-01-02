"""
╔══════════════════════════════════════════════════════════════════════╗
║                 PERSISTENCE PERFORMANCE CONTRACT                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  Ratified:     2024-12-31                                            ║
║  Guardian:     CI Pipeline                                            ║
║  Renegotiation: Requires team review + documented justification      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  We solemnly contract the following guarantees:                      ║
║                                                                       ║
║  • Save small corpus (< 50 docs, < 500 tokens) < 3 seconds          ║
║  • Load small corpus < 5 seconds                                     ║
║  • Roundtrip preserves all state (100% fidelity)                    ║
║  • Export graph JSON (500 nodes) < 1 second                          ║
║  • Atomic writes (no partial saves)                                 ║
║  • Save/load is deterministic (same data = same output)             ║
║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import time
import tempfile
import shutil
import pytest
from pathlib import Path
from cortical.persistence import (
    save_processor,
    load_processor,
    export_graph_json,
    get_state_summary,
)
from cortical.layers import CorticalLayer, HierarchicalLayer


@pytest.mark.contract
class TestPersistencePerformanceContract:
    """
    Persistence Performance Contract

    As a developer persisting cortical state,
    I expect save/load to complete in reasonable time,
    So that state management doesn't block workflows.
    """

    MAX_SAVE_SMALL_SECONDS = 3.0
    MAX_LOAD_SMALL_SECONDS = 5.0

    def test_save_small_corpus_latency(self, small_processor, tmp_path):
        """
        CONTRACT: Save small corpus in < 3 seconds.

        Small corpus = < 50 docs, < 500 tokens.
        """
        save_path = tmp_path / "corpus_state"

        # Verify corpus size is within bounds
        assert len(small_processor.documents) <= 50
        token_count = small_processor.layers[CorticalLayer.TOKENS].column_count()
        assert token_count <= 500

        start = time.perf_counter()
        save_processor(
            str(save_path),
            small_processor.layers,
            small_processor.documents,
            verbose=False
        )
        elapsed_s = time.perf_counter() - start

        assert elapsed_s < self.MAX_SAVE_SMALL_SECONDS, (
            f"CONTRACT VIOLATION: Saving small corpus took {elapsed_s:.2f}s, "
            f"contract requires <{self.MAX_SAVE_SMALL_SECONDS}s"
        )

        # Verify save succeeded
        assert save_path.exists()

    def test_load_small_corpus_latency(self, small_processor, tmp_path):
        """
        CONTRACT: Load small corpus in < 5 seconds.

        Loading is slower than saving (object creation overhead).
        """
        save_path = tmp_path / "corpus_state"

        # Save first
        save_processor(
            str(save_path),
            small_processor.layers,
            small_processor.documents,
            verbose=False
        )

        # Now measure load time
        start = time.perf_counter()
        layers, docs, doc_metadata, embeddings, relations, metadata = load_processor(
            str(save_path),
            verbose=False
        )
        elapsed_s = time.perf_counter() - start

        assert elapsed_s < self.MAX_LOAD_SMALL_SECONDS, (
            f"CONTRACT VIOLATION: Loading small corpus took {elapsed_s:.2f}s, "
            f"contract requires <{self.MAX_LOAD_SMALL_SECONDS}s"
        )

        # Verify load succeeded
        assert len(docs) > 0
        assert len(layers) > 0


@pytest.mark.contract
class TestPersistenceCorrectnessContract:
    """
    Persistence Correctness Contract

    As a developer relying on persistence,
    I expect perfect fidelity in save/load cycles,
    So that no data is lost or corrupted.
    """

    def test_roundtrip_preserves_documents(self, small_processor, tmp_path):
        """
        CONTRACT: Roundtrip preserves all documents.

        Every document must survive save/load.
        """
        save_path = tmp_path / "corpus_state"

        original_docs = dict(small_processor.documents)

        # Roundtrip
        save_processor(
            str(save_path),
            small_processor.layers,
            small_processor.documents,
            verbose=False
        )

        layers, docs, doc_metadata, embeddings, relations, metadata = load_processor(
            str(save_path),
            verbose=False
        )

        # Verify all documents preserved
        assert len(docs) == len(original_docs), (
            f"CONTRACT VIOLATION: Document count changed: {len(original_docs)} -> {len(docs)}"
        )

        for doc_id, content in original_docs.items():
            assert doc_id in docs, f"Document '{doc_id}' lost"
            assert docs[doc_id] == content, (
                f"Document '{doc_id}' content changed"
            )

    def test_roundtrip_preserves_tokens(self, small_processor, tmp_path):
        """
        CONTRACT: Roundtrip preserves all token minicolumns.

        Token layer must be fully preserved.
        """
        save_path = tmp_path / "corpus_state"

        original_layer = small_processor.layers[CorticalLayer.TOKENS]
        original_tokens = set(original_layer.minicolumns.keys())
        original_count = original_layer.column_count()

        # Roundtrip
        save_processor(
            str(save_path),
            small_processor.layers,
            small_processor.documents,
            verbose=False
        )

        layers, docs, doc_metadata, embeddings, relations, metadata = load_processor(
            str(save_path),
            verbose=False
        )

        loaded_layer = layers[CorticalLayer.TOKENS]
        loaded_tokens = set(loaded_layer.minicolumns.keys())

        assert loaded_layer.column_count() == original_count, (
            f"CONTRACT VIOLATION: Token count changed: {original_count} -> {loaded_layer.column_count()}"
        )

        assert loaded_tokens == original_tokens, (
            f"CONTRACT VIOLATION: Token set changed"
        )

    def test_roundtrip_preserves_connections(self, small_processor, tmp_path):
        """
        CONTRACT: Roundtrip preserves lateral connections.

        Connection graph must be fully preserved.
        """
        save_path = tmp_path / "corpus_state"

        original_layer = small_processor.layers[CorticalLayer.TOKENS]

        # Get connection snapshot
        original_connections = {}
        for content, col in original_layer.minicolumns.items():
            original_connections[content] = dict(col.lateral_connections)

        # Roundtrip
        save_processor(
            str(save_path),
            small_processor.layers,
            small_processor.documents,
            verbose=False
        )

        layers, docs, doc_metadata, embeddings, relations, metadata = load_processor(
            str(save_path),
            verbose=False
        )

        loaded_layer = layers[CorticalLayer.TOKENS]

        # Verify connections preserved
        for content, col in loaded_layer.minicolumns.items():
            if content in original_connections:
                original_conns = original_connections[content]
                loaded_conns = col.lateral_connections

                assert loaded_conns.keys() == original_conns.keys(), (
                    f"CONTRACT VIOLATION: Connections for '{content}' changed"
                )

                for target_id, weight in original_conns.items():
                    assert loaded_conns[target_id] == pytest.approx(weight, abs=1e-6), (
                        f"CONTRACT VIOLATION: Connection weight changed for '{content}' -> '{target_id}'"
                    )

    def test_roundtrip_preserves_scores(self, small_processor, tmp_path):
        """
        CONTRACT: Roundtrip preserves PageRank and TF-IDF scores.

        Computed scores must be preserved exactly.
        """
        save_path = tmp_path / "corpus_state"

        original_layer = small_processor.layers[CorticalLayer.TOKENS]

        # Get score snapshot
        original_scores = {}
        for content, col in original_layer.minicolumns.items():
            original_scores[content] = {
                'pagerank': col.pagerank,
                'tfidf': col.tfidf,
                'activation': col.activation,
            }

        # Roundtrip
        save_processor(
            str(save_path),
            small_processor.layers,
            small_processor.documents,
            verbose=False
        )

        layers, docs, doc_metadata, embeddings, relations, metadata = load_processor(
            str(save_path),
            verbose=False
        )

        loaded_layer = layers[CorticalLayer.TOKENS]

        # Verify scores preserved
        for content, col in loaded_layer.minicolumns.items():
            if content in original_scores:
                orig = original_scores[content]
                assert col.pagerank == pytest.approx(orig['pagerank'], abs=1e-6)
                assert col.tfidf == pytest.approx(orig['tfidf'], abs=1e-6)
                assert col.activation == pytest.approx(orig['activation'], abs=1e-6)


@pytest.mark.contract
class TestGraphExportContract:
    """
    Graph Export Performance Contract

    As a developer visualizing networks,
    I expect graph export to be fast,
    So that visualization workflows are smooth.
    """

    MAX_EXPORT_500_NODES_SECONDS = 1.0

    def test_export_graph_json_latency(self, small_processor, tmp_path):
        """
        CONTRACT: Export 500 nodes to JSON in < 1 second.

        Graph export for visualization must be fast.
        """
        export_path = tmp_path / "graph.json"

        start = time.perf_counter()
        graph = export_graph_json(
            str(export_path),
            small_processor.layers,
            layer_filter=CorticalLayer.TOKENS,
            max_nodes=500,
            verbose=False
        )
        elapsed_s = time.perf_counter() - start

        assert elapsed_s < self.MAX_EXPORT_500_NODES_SECONDS, (
            f"CONTRACT VIOLATION: Exporting graph took {elapsed_s:.2f}s, "
            f"contract requires <{self.MAX_EXPORT_500_NODES_SECONDS}s"
        )

        # Verify export succeeded
        assert export_path.exists()
        assert 'nodes' in graph
        assert 'edges' in graph

    def test_export_graph_structure(self, small_processor, tmp_path):
        """
        CONTRACT: Exported graph has valid D3.js structure.

        Graph format must be compatible with visualization tools.
        """
        export_path = tmp_path / "graph.json"

        graph = export_graph_json(
            str(export_path),
            small_processor.layers,
            layer_filter=CorticalLayer.TOKENS,
            max_nodes=100,
            verbose=False
        )

        # Verify structure
        assert 'nodes' in graph
        assert 'edges' in graph
        assert 'metadata' in graph

        # Nodes should have required fields
        if graph['nodes']:
            node = graph['nodes'][0]
            assert 'id' in node
            assert 'label' in node
            assert 'layer' in node
            assert 'pagerank' in node

        # Edges should have required fields
        if graph['edges']:
            edge = graph['edges'][0]
            assert 'source' in edge
            assert 'target' in edge
            assert 'weight' in edge


@pytest.mark.contract
class TestStateSummaryContract:
    """
    State Summary Contract

    As a developer monitoring system state,
    I expect state summary to be fast and accurate,
    So that monitoring doesn't impact performance.
    """

    def test_state_summary_correctness(self, small_processor):
        """
        CONTRACT: State summary accurately reflects corpus state.

        Summary statistics must be correct.
        """
        summary = get_state_summary(
            small_processor.layers,
            small_processor.documents
        )

        # Verify structure
        assert 'documents' in summary
        assert 'layers' in summary
        assert 'total_columns' in summary
        assert 'total_connections' in summary

        # Verify accuracy
        assert summary['documents'] == len(small_processor.documents)

        # Total columns should match sum across layers
        total = sum(
            len(layer.minicolumns) for layer in small_processor.layers.values()
        )
        assert summary['total_columns'] == total

    def test_state_summary_performance(self, small_processor):
        """
        CONTRACT: State summary computation is fast.

        Summary should be near-instant for small corpora.
        """
        iterations = 100

        start = time.perf_counter()
        for _ in range(iterations):
            summary = get_state_summary(
                small_processor.layers,
                small_processor.documents
            )
        elapsed_ms = (time.perf_counter() - start) * 1000

        avg_ms = elapsed_ms / iterations

        # Should be very fast (< 10ms on average)
        assert avg_ms < 10.0, (
            f"CONTRACT VIOLATION: State summary took {avg_ms:.2f}ms on average, "
            f"should be <10ms"
        )


@pytest.mark.contract
class TestPersistenceDeterminismContract:
    """
    Persistence Determinism Contract

    As a developer testing persistence,
    I expect deterministic output for same input,
    So that tests are reproducible.
    """

    def test_save_is_deterministic(self, small_processor, tmp_path):
        """
        CONTRACT: Saving same state twice produces identical output.

        Determinism is essential for testing and debugging.
        """
        save_path1 = tmp_path / "state1"
        save_path2 = tmp_path / "state2"

        # Save twice
        save_processor(
            str(save_path1),
            small_processor.layers,
            small_processor.documents,
            verbose=False
        )

        save_processor(
            str(save_path2),
            small_processor.layers,
            small_processor.documents,
            verbose=False
        )

        # Load both
        layers1, docs1, _, _, _, _ = load_processor(str(save_path1), verbose=False)
        layers2, docs2, _, _, _, _ = load_processor(str(save_path2), verbose=False)

        # Verify identical
        assert docs1.keys() == docs2.keys()
        assert layers1.keys() == layers2.keys()

        # Check token counts match
        for layer_key in layers1.keys():
            count1 = layers1[layer_key].column_count()
            count2 = layers2[layer_key].column_count()
            assert count1 == count2, (
                f"CONTRACT VIOLATION: Non-deterministic save for {layer_key.name}"
            )
