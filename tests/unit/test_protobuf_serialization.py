"""
Unit tests for Protocol Buffers serialization.

Tests the protobuf serialization and deserialization functionality
for cross-language corpus sharing.
"""

import unittest
import tempfile
import os
from pathlib import Path

import pytest

# Mark entire module as optional (requires protobuf package)
pytestmark = [pytest.mark.optional, pytest.mark.protobuf]

# Import core modules
from cortical.layers import CorticalLayer, HierarchicalLayer
from cortical.minicolumn import Minicolumn, Edge
from cortical.persistence import save_processor, load_processor

# Try to import protobuf modules
try:
    from cortical.proto.serialization import (
        to_proto, from_proto,
        edge_to_proto, edge_from_proto,
        minicolumn_to_proto, minicolumn_from_proto,
        layer_to_proto, layer_from_proto,
        _get_proto_classes
    )
    PROTOBUF_AVAILABLE = True
except ImportError:
    PROTOBUF_AVAILABLE = False

# Check if protoc compiler is actually available (not just the package)
PROTOC_AVAILABLE = False
if PROTOBUF_AVAILABLE:
    try:
        # This will attempt to compile protos and fail if protoc is missing
        _get_proto_classes()
        PROTOC_AVAILABLE = True
    except (RuntimeError, FileNotFoundError):
        # protoc not installed - proto compilation fails
        pass


@unittest.skipIf(not PROTOC_AVAILABLE, "protobuf package or protoc compiler not available")
class TestEdgeSerialization(unittest.TestCase):
    """Test Edge protobuf serialization."""

    def test_edge_round_trip(self):
        """Test Edge serialization and deserialization."""
        edge = Edge(
            target_id="L0_network",
            weight=0.8,
            relation_type="RelatedTo",
            confidence=0.9,
            source="semantic"
        )

        # Convert to protobuf and back
        proto = edge_to_proto(edge)
        restored = edge_from_proto(proto)

        # Verify all fields
        self.assertEqual(restored.target_id, edge.target_id)
        self.assertEqual(restored.weight, edge.weight)
        self.assertEqual(restored.relation_type, edge.relation_type)
        self.assertEqual(restored.confidence, edge.confidence)
        self.assertEqual(restored.source, edge.source)

    def test_edge_default_values(self):
        """Test Edge with default values."""
        edge = Edge(target_id="L0_test")

        proto = edge_to_proto(edge)
        restored = edge_from_proto(proto)

        self.assertEqual(restored.target_id, "L0_test")
        self.assertEqual(restored.weight, 1.0)
        self.assertEqual(restored.relation_type, "co_occurrence")
        self.assertEqual(restored.confidence, 1.0)
        self.assertEqual(restored.source, "corpus")


@unittest.skipIf(not PROTOC_AVAILABLE, "protobuf package or protoc compiler not available")
class TestMinicolumnSerialization(unittest.TestCase):
    """Test Minicolumn protobuf serialization."""

    def test_minicolumn_basic(self):
        """Test basic Minicolumn serialization."""
        col = Minicolumn("L0_neural", "neural", 0)
        col.occurrence_count = 15
        col.activation = 0.5
        col.pagerank = 0.01
        col.tfidf = 2.5
        col.document_ids.add("doc1")
        col.document_ids.add("doc2")

        # Add connections
        col.add_lateral_connection("L0_network", 3.0)
        col.add_typed_connection("L0_brain", 2.0, relation_type="RelatedTo", confidence=0.85)

        # Convert to protobuf and back
        proto = minicolumn_to_proto(col)
        restored = minicolumn_from_proto(proto)

        # Verify basic fields
        self.assertEqual(restored.id, col.id)
        self.assertEqual(restored.content, col.content)
        self.assertEqual(restored.layer, col.layer)
        self.assertEqual(restored.occurrence_count, col.occurrence_count)
        self.assertEqual(restored.activation, col.activation)
        self.assertEqual(restored.pagerank, col.pagerank)
        self.assertEqual(restored.tfidf, col.tfidf)
        self.assertEqual(restored.document_ids, col.document_ids)

        # Verify connections
        self.assertEqual(len(restored.lateral_connections), 2)
        self.assertEqual(restored.lateral_connections["L0_network"], 3.0)
        self.assertEqual(restored.lateral_connections["L0_brain"], 2.0)

        # Verify typed connections
        self.assertEqual(len(restored.typed_connections), 2)
        brain_edge = restored.typed_connections["L0_brain"]
        self.assertEqual(brain_edge.relation_type, "RelatedTo")
        self.assertEqual(brain_edge.confidence, 0.85)

    def test_minicolumn_with_feedforward_feedback(self):
        """Test Minicolumn with feedforward/feedback connections."""
        col = Minicolumn("L1_neural_network", "neural network", 1)
        col.add_feedforward_connection("L0_neural", 1.0)
        col.add_feedforward_connection("L0_network", 1.0)
        col.add_feedback_connection("L2_cluster_5", 0.5)

        proto = minicolumn_to_proto(col)
        restored = minicolumn_from_proto(proto)

        self.assertEqual(len(restored.feedforward_connections), 2)
        self.assertEqual(restored.feedforward_connections["L0_neural"], 1.0)
        self.assertEqual(len(restored.feedback_connections), 1)
        self.assertEqual(restored.feedback_connections["L2_cluster_5"], 0.5)

    def test_minicolumn_with_tfidf_per_doc(self):
        """Test Minicolumn with per-document TF-IDF scores."""
        col = Minicolumn("L0_test", "test", 0)
        col.tfidf_per_doc = {"doc1": 1.5, "doc2": 2.0, "doc3": 0.8}
        col.doc_occurrence_counts = {"doc1": 3, "doc2": 5, "doc3": 1}

        proto = minicolumn_to_proto(col)
        restored = minicolumn_from_proto(proto)

        self.assertEqual(restored.tfidf_per_doc, col.tfidf_per_doc)
        self.assertEqual(restored.doc_occurrence_counts, col.doc_occurrence_counts)

    def test_minicolumn_with_cluster(self):
        """Test Minicolumn with cluster ID."""
        col = Minicolumn("L0_test", "test", 0)
        col.cluster_id = 42

        proto = minicolumn_to_proto(col)
        restored = minicolumn_from_proto(proto)

        self.assertEqual(restored.cluster_id, 42)

    def test_minicolumn_with_name_tokens(self):
        """Test Minicolumn with name tokens (for documents)."""
        col = Minicolumn("L3_doc1", "doc1", 3)
        col.name_tokens = {"test", "document", "sample"}

        proto = minicolumn_to_proto(col)
        restored = minicolumn_from_proto(proto)

        self.assertEqual(restored.name_tokens, col.name_tokens)

    def test_empty_minicolumn(self):
        """Test Minicolumn with no connections or metadata."""
        col = Minicolumn("L0_empty", "empty", 0)

        proto = minicolumn_to_proto(col)
        restored = minicolumn_from_proto(proto)

        self.assertEqual(restored.id, col.id)
        self.assertEqual(len(restored.lateral_connections), 0)
        self.assertEqual(len(restored.typed_connections), 0)


@unittest.skipIf(not PROTOC_AVAILABLE, "protobuf package or protoc compiler not available")
class TestLayerSerialization(unittest.TestCase):
    """Test HierarchicalLayer protobuf serialization."""

    def test_layer_basic(self):
        """Test basic HierarchicalLayer serialization."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        col1 = layer.get_or_create_minicolumn("neural")
        col1.occurrence_count = 10
        col2 = layer.get_or_create_minicolumn("network")
        col2.occurrence_count = 8
        col1.add_lateral_connection("L0_network", 5.0)

        proto = layer_to_proto(layer)
        restored = layer_from_proto(proto)

        # Verify layer structure
        self.assertEqual(restored.level, CorticalLayer.TOKENS)
        self.assertEqual(len(restored.minicolumns), 2)
        self.assertIn("neural", restored.minicolumns)
        self.assertIn("network", restored.minicolumns)

        # Verify minicolumn data
        restored_col1 = restored.minicolumns["neural"]
        self.assertEqual(restored_col1.occurrence_count, 10)
        self.assertEqual(len(restored_col1.lateral_connections), 1)

        # Verify ID index was rebuilt
        self.assertEqual(restored.get_by_id("L0_neural"), restored_col1)

    def test_empty_layer(self):
        """Test empty HierarchicalLayer serialization."""
        layer = HierarchicalLayer(CorticalLayer.CONCEPTS)

        proto = layer_to_proto(layer)
        restored = layer_from_proto(proto)

        self.assertEqual(restored.level, CorticalLayer.CONCEPTS)
        self.assertEqual(len(restored.minicolumns), 0)


@unittest.skipIf(not PROTOC_AVAILABLE, "protobuf package or protoc compiler not available")
class TestProcessorStateSerialization(unittest.TestCase):
    """Test complete processor state protobuf serialization."""

    def setUp(self):
        """Create test processor state."""
        # Create layers
        self.layers = {
            CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS),
            CorticalLayer.BIGRAMS: HierarchicalLayer(CorticalLayer.BIGRAMS),
            CorticalLayer.CONCEPTS: HierarchicalLayer(CorticalLayer.CONCEPTS),
            CorticalLayer.DOCUMENTS: HierarchicalLayer(CorticalLayer.DOCUMENTS)
        }

        # Add some data to token layer
        layer0 = self.layers[CorticalLayer.TOKENS]
        col1 = layer0.get_or_create_minicolumn("neural")
        col1.occurrence_count = 10
        col1.pagerank = 0.05
        col1.tfidf = 2.5
        col1.document_ids.add("doc1")
        col1.add_lateral_connection("L0_network", 3.0)

        col2 = layer0.get_or_create_minicolumn("network")
        col2.occurrence_count = 8
        col2.pagerank = 0.04
        col2.tfidf = 2.0
        col2.document_ids.add("doc1")

        # Documents
        self.documents = {
            "doc1": "Neural networks are powerful.",
            "doc2": "Machine learning is fascinating."
        }

        # Document metadata
        self.document_metadata = {
            "doc1": {"source": "test", "timestamp": "2025-01-01"},
            "doc2": {"source": "test", "timestamp": "2025-01-02"}
        }

        # Embeddings
        self.embeddings = {
            "neural": [0.1, 0.2, 0.3],
            "network": [0.15, 0.25, 0.35]
        }

        # Semantic relations
        self.semantic_relations = [
            ("neural", "RelatedTo", "network", 0.8),
            ("neural", "UsedFor", "learning", 0.5)
        ]

        # Metadata
        self.metadata = {
            "created": "2025-01-01",
            "version": "1.0",
            "settings": {"alpha": 0.85, "enabled": True}
        }

    def test_full_state_round_trip(self):
        """Test full processor state serialization."""
        # Convert to protobuf
        proto = to_proto(
            self.layers, self.documents, self.document_metadata,
            self.embeddings, self.semantic_relations, self.metadata
        )

        # Convert back to Python
        (restored_layers, restored_docs, restored_meta,
         restored_embeddings, restored_relations, restored_metadata) = from_proto(proto)

        # Verify layers
        self.assertEqual(len(restored_layers), 4)
        layer0 = restored_layers[CorticalLayer.TOKENS]
        self.assertEqual(len(layer0.minicolumns), 2)

        neural = layer0.minicolumns["neural"]
        self.assertEqual(neural.occurrence_count, 10)
        self.assertEqual(neural.pagerank, 0.05)
        self.assertEqual(neural.tfidf, 2.5)

        # Verify documents
        self.assertEqual(restored_docs, self.documents)

        # Verify document metadata
        self.assertEqual(len(restored_meta), 2)
        self.assertEqual(restored_meta["doc1"]["source"], "test")

        # Verify embeddings
        self.assertEqual(len(restored_embeddings), 2)
        self.assertEqual(restored_embeddings["neural"], [0.1, 0.2, 0.3])

        # Verify semantic relations
        self.assertEqual(len(restored_relations), 2)
        self.assertEqual(restored_relations[0][0], "neural")
        self.assertEqual(restored_relations[0][1], "RelatedTo")
        self.assertEqual(restored_relations[0][2], "network")
        self.assertEqual(restored_relations[0][3], 0.8)

        # Verify metadata
        self.assertEqual(restored_metadata["created"], "2025-01-01")
        self.assertEqual(restored_metadata["settings"]["alpha"], 0.85)
        self.assertEqual(restored_metadata["settings"]["enabled"], True)

    def test_minimal_state(self):
        """Test minimal processor state (layers and documents only)."""
        proto = to_proto(self.layers, self.documents)
        (restored_layers, restored_docs, restored_meta,
         restored_embeddings, restored_relations, restored_metadata) = from_proto(proto)

        self.assertEqual(len(restored_layers), 4)
        self.assertEqual(restored_docs, self.documents)
        self.assertEqual(restored_meta, {})
        self.assertEqual(restored_embeddings, {})
        self.assertEqual(restored_relations, [])
        self.assertEqual(restored_metadata, {})


@unittest.skipIf(not PROTOC_AVAILABLE, "protobuf package or protoc compiler not available")
class TestPersistenceIntegration(unittest.TestCase):
    """Test protobuf integration with persistence module."""

    def setUp(self):
        """Create test data."""
        self.layers = {
            CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS),
            CorticalLayer.BIGRAMS: HierarchicalLayer(CorticalLayer.BIGRAMS),
            CorticalLayer.CONCEPTS: HierarchicalLayer(CorticalLayer.CONCEPTS),
            CorticalLayer.DOCUMENTS: HierarchicalLayer(CorticalLayer.DOCUMENTS)
        }

        layer0 = self.layers[CorticalLayer.TOKENS]
        col = layer0.get_or_create_minicolumn("test")
        col.occurrence_count = 5
        col.add_lateral_connection("L0_data", 2.0)

        self.documents = {"doc1": "Test data"}
        self.document_metadata = {"doc1": {"source": "test"}}
        self.embeddings = {"test": [0.1, 0.2]}
        self.semantic_relations = [("test", "RelatedTo", "data", 0.7)]
        self.metadata = {"version": "1.0"}

    def test_save_and_load_protobuf(self):
        """Test saving and loading with protobuf format."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pb') as f:
            filepath = f.name

        try:
            # Save with protobuf format
            save_processor(
                filepath, self.layers, self.documents,
                self.document_metadata, self.embeddings,
                self.semantic_relations, self.metadata,
                verbose=False, format='protobuf'
            )

            # Load with protobuf format
            (restored_layers, restored_docs, restored_meta,
             restored_embeddings, restored_relations, restored_metadata) = load_processor(
                filepath, verbose=False, format='protobuf'
            )

            # Verify data
            self.assertEqual(len(restored_layers), 4)
            layer0 = restored_layers[CorticalLayer.TOKENS]
            self.assertEqual(len(layer0.minicolumns), 1)  # Only "test" minicolumn

            test_col = layer0.minicolumns["test"]
            self.assertEqual(test_col.occurrence_count, 5)
            self.assertEqual(len(test_col.lateral_connections), 1)
            self.assertIn("L0_data", test_col.lateral_connections)

            self.assertEqual(restored_docs, self.documents)
            self.assertEqual(restored_meta, self.document_metadata)
            self.assertEqual(restored_embeddings, self.embeddings)
            self.assertEqual(len(restored_relations), 1)

        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_save_and_load_pickle(self):
        """Test saving and loading with pickle format (backward compatibility)."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            filepath = f.name

        try:
            # Save with pickle format (default)
            save_processor(
                filepath, self.layers, self.documents,
                self.document_metadata, self.embeddings,
                self.semantic_relations, self.metadata,
                verbose=False, format='pickle'
            )

            # Load with pickle format
            (restored_layers, restored_docs, restored_meta,
             restored_embeddings, restored_relations, restored_metadata) = load_processor(
                filepath, verbose=False, format='pickle'
            )

            # Verify data
            self.assertEqual(len(restored_layers), 4)
            self.assertEqual(restored_docs, self.documents)

        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_auto_detect_format(self):
        """Test automatic format detection."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            pickle_file = f.name
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pb') as f:
            proto_file = f.name

        try:
            # Save in both formats
            save_processor(
                pickle_file, self.layers, self.documents,
                verbose=False, format='pickle'
            )
            save_processor(
                proto_file, self.layers, self.documents,
                verbose=False, format='protobuf'
            )

            # Load without specifying format (auto-detect)
            (layers1, docs1, _, _, _, _) = load_processor(pickle_file, verbose=False)
            (layers2, docs2, _, _, _, _) = load_processor(proto_file, verbose=False)

            # Both should load successfully
            self.assertEqual(len(layers1), 4)
            self.assertEqual(len(layers2), 4)
            self.assertEqual(docs1, self.documents)
            self.assertEqual(docs2, self.documents)

        finally:
            if os.path.exists(pickle_file):
                os.unlink(pickle_file)
            if os.path.exists(proto_file):
                os.unlink(proto_file)

    def test_invalid_format_error(self):
        """Test error handling for invalid format."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            filepath = f.name

        try:
            with self.assertRaises(ValueError):
                save_processor(
                    filepath, self.layers, self.documents,
                    verbose=False, format='json'
                )
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)


@unittest.skipIf(not PROTOC_AVAILABLE, "protobuf package or protoc compiler not available")
class TestComplexDataStructures(unittest.TestCase):
    """Test protobuf serialization with complex data structures."""

    def test_nested_metadata(self):
        """Test nested dictionaries in metadata."""
        layers = {
            CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS)
        }
        documents = {"doc1": "test"}
        metadata = {
            "config": {
                "params": {
                    "alpha": 0.85,
                    "beta": 0.15,
                    "nested": {
                        "value": 42,
                        "enabled": True
                    }
                },
                "lists": [1, 2, 3, "four", 5.0]
            }
        }

        proto = to_proto(layers, documents, metadata=metadata)
        (_, _, _, _, _, restored_metadata) = from_proto(proto)

        self.assertEqual(restored_metadata["config"]["params"]["alpha"], 0.85)
        self.assertEqual(restored_metadata["config"]["params"]["nested"]["value"], 42)
        self.assertEqual(restored_metadata["config"]["params"]["nested"]["enabled"], True)
        self.assertEqual(restored_metadata["config"]["lists"], [1, 2, 3, "four", 5.0])

    def test_large_embeddings(self):
        """Test large embedding vectors."""
        layers = {
            CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS)
        }
        documents = {"doc1": "test"}
        embeddings = {
            "term1": [float(i) for i in range(300)],  # 300-dimensional embedding
            "term2": [float(i * 0.5) for i in range(300)]
        }

        proto = to_proto(layers, documents, embeddings=embeddings)
        (_, _, _, restored_embeddings, _, _) = from_proto(proto)

        self.assertEqual(len(restored_embeddings["term1"]), 300)
        self.assertEqual(restored_embeddings["term1"][0], 0.0)
        self.assertEqual(restored_embeddings["term1"][299], 299.0)

    def test_many_semantic_relations(self):
        """Test many semantic relations."""
        layers = {
            CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS)
        }
        documents = {"doc1": "test"}
        relations = [
            (f"term{i}", "RelatedTo", f"term{i+1}", float(i) / 100.0)
            for i in range(100)
        ]

        proto = to_proto(layers, documents, semantic_relations=relations)
        (_, _, _, _, restored_relations, _) = from_proto(proto)

        self.assertEqual(len(restored_relations), 100)
        self.assertEqual(restored_relations[50][0], "term50")
        self.assertEqual(restored_relations[50][2], "term51")
        self.assertEqual(restored_relations[50][3], 0.5)


if __name__ == '__main__':
    unittest.main()
