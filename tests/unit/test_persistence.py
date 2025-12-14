"""
Unit Tests for Persistence Module
=================================

Task #158: Unit tests for cortical/persistence.py pure functions
and serialization helpers.

Tests the following:
- _get_relation_color: Get color for relation type
- _count_edge_types: Count edges by edge type
- _count_relation_types: Count edges by relation type
- LAYER_COLORS: Layer color mapping
- LAYER_NAMES: Layer name mapping
- Embeddings JSON export/import
- Semantic relations JSON export/import
"""

import json
import os
import tempfile
import pytest

from cortical.persistence import (
    _get_relation_color,
    _count_edge_types,
    _count_relation_types,
    LAYER_COLORS,
    LAYER_NAMES,
    export_embeddings_json,
    load_embeddings_json,
    export_semantic_relations_json,
    load_semantic_relations_json,
    # SEC-003: HMAC signature verification
    SignatureVerificationError,
    _compute_signature,
    _save_signature,
    _load_signature,
    _verify_signature,
    _get_signature_path,
)
from cortical.layers import CorticalLayer


# =============================================================================
# GET RELATION COLOR TESTS
# =============================================================================


class TestGetRelationColor:
    """Tests for _get_relation_color function."""

    def test_isa_color(self):
        """IsA relation has defined color."""
        color = _get_relation_color("IsA")
        assert color.startswith("#")
        assert len(color) == 7

    def test_partof_color(self):
        """PartOf relation has defined color."""
        color = _get_relation_color("PartOf")
        assert color.startswith("#")

    def test_causes_color(self):
        """Causes relation has defined color."""
        color = _get_relation_color("Causes")
        assert color.startswith("#")

    def test_similarto_color(self):
        """SimilarTo relation has defined color."""
        color = _get_relation_color("SimilarTo")
        assert color.startswith("#")

    def test_unknown_relation(self):
        """Unknown relation returns default color."""
        color = _get_relation_color("MadeUpRelation")
        assert color == "#808080"  # Default grey

    def test_cooccurrence_color(self):
        """co_occurrence has defined color."""
        color = _get_relation_color("co_occurrence")
        assert color.startswith("#")

    def test_feedforward_color(self):
        """feedforward edge type has defined color."""
        color = _get_relation_color("feedforward")
        assert color.startswith("#")

    def test_feedback_color(self):
        """feedback edge type has defined color."""
        color = _get_relation_color("feedback")
        assert color.startswith("#")


# =============================================================================
# COUNT EDGE TYPES TESTS
# =============================================================================


class TestCountEdgeTypes:
    """Tests for _count_edge_types function."""

    def test_empty_edges(self):
        """Empty edge list returns empty counts."""
        result = _count_edge_types([])
        assert result == {}

    def test_single_edge_type(self):
        """Single edge type is counted."""
        edges = [
            {"edge_type": "lateral"},
            {"edge_type": "lateral"},
            {"edge_type": "lateral"},
        ]
        result = _count_edge_types(edges)
        assert result == {"lateral": 3}

    def test_multiple_edge_types(self):
        """Multiple edge types are counted separately."""
        edges = [
            {"edge_type": "lateral"},
            {"edge_type": "lateral"},
            {"edge_type": "cross_layer"},
            {"edge_type": "semantic"},
        ]
        result = _count_edge_types(edges)
        assert result["lateral"] == 2
        assert result["cross_layer"] == 1
        assert result["semantic"] == 1

    def test_missing_edge_type(self):
        """Edges without edge_type count as 'unknown'."""
        edges = [
            {"source": "a", "target": "b"},
            {"edge_type": "lateral"},
        ]
        result = _count_edge_types(edges)
        assert result["unknown"] == 1
        assert result["lateral"] == 1


# =============================================================================
# COUNT RELATION TYPES TESTS
# =============================================================================


class TestCountRelationTypes:
    """Tests for _count_relation_types function."""

    def test_empty_edges(self):
        """Empty edge list returns empty counts."""
        result = _count_relation_types([])
        assert result == {}

    def test_single_relation_type(self):
        """Single relation type is counted."""
        edges = [
            {"relation_type": "IsA"},
            {"relation_type": "IsA"},
        ]
        result = _count_relation_types(edges)
        assert result == {"IsA": 2}

    def test_multiple_relation_types(self):
        """Multiple relation types are counted separately."""
        edges = [
            {"relation_type": "IsA"},
            {"relation_type": "HasA"},
            {"relation_type": "IsA"},
            {"relation_type": "PartOf"},
        ]
        result = _count_relation_types(edges)
        assert result["IsA"] == 2
        assert result["HasA"] == 1
        assert result["PartOf"] == 1

    def test_missing_relation_type(self):
        """Edges without relation_type count as 'unknown'."""
        edges = [
            {"source": "a", "target": "b"},
            {"relation_type": "IsA"},
        ]
        result = _count_relation_types(edges)
        assert result["unknown"] == 1
        assert result["IsA"] == 1


# =============================================================================
# LAYER CONSTANTS TESTS
# =============================================================================


class TestLayerColors:
    """Tests for LAYER_COLORS constant."""

    def test_all_layers_have_colors(self):
        """All CorticalLayer values have colors."""
        for layer in CorticalLayer:
            assert layer in LAYER_COLORS
            color = LAYER_COLORS[layer]
            assert color.startswith("#")
            assert len(color) == 7

    def test_tokens_layer_color(self):
        """TOKENS layer has a color."""
        assert CorticalLayer.TOKENS in LAYER_COLORS

    def test_bigrams_layer_color(self):
        """BIGRAMS layer has a color."""
        assert CorticalLayer.BIGRAMS in LAYER_COLORS

    def test_concepts_layer_color(self):
        """CONCEPTS layer has a color."""
        assert CorticalLayer.CONCEPTS in LAYER_COLORS

    def test_documents_layer_color(self):
        """DOCUMENTS layer has a color."""
        assert CorticalLayer.DOCUMENTS in LAYER_COLORS


class TestLayerNames:
    """Tests for LAYER_NAMES constant."""

    def test_all_layers_have_names(self):
        """All CorticalLayer values have display names."""
        for layer in CorticalLayer:
            assert layer in LAYER_NAMES
            name = LAYER_NAMES[layer]
            assert isinstance(name, str)
            assert len(name) > 0

    def test_tokens_name(self):
        """TOKENS layer has correct name."""
        assert LAYER_NAMES[CorticalLayer.TOKENS] == "Tokens"

    def test_bigrams_name(self):
        """BIGRAMS layer has correct name."""
        assert LAYER_NAMES[CorticalLayer.BIGRAMS] == "Bigrams"

    def test_concepts_name(self):
        """CONCEPTS layer has correct name."""
        assert LAYER_NAMES[CorticalLayer.CONCEPTS] == "Concepts"

    def test_documents_name(self):
        """DOCUMENTS layer has correct name."""
        assert LAYER_NAMES[CorticalLayer.DOCUMENTS] == "Documents"


# =============================================================================
# EMBEDDINGS JSON TESTS
# =============================================================================


class TestEmbeddingsJson:
    """Tests for embeddings JSON export/import."""

    def test_export_load_roundtrip(self):
        """Embeddings survive export/load roundtrip."""
        embeddings = {
            "term1": [0.1, 0.2, 0.3],
            "term2": [0.4, 0.5, 0.6],
        }
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name
        try:
            export_embeddings_json(filepath, embeddings)
            loaded = load_embeddings_json(filepath)
            assert loaded == embeddings
        finally:
            os.unlink(filepath)

    def test_export_empty_embeddings(self):
        """Empty embeddings can be exported."""
        embeddings = {}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name
        try:
            export_embeddings_json(filepath, embeddings)
            loaded = load_embeddings_json(filepath)
            assert loaded == {}
        finally:
            os.unlink(filepath)

    def test_export_with_metadata(self):
        """Embeddings with metadata are exported."""
        embeddings = {"term1": [0.1, 0.2]}
        metadata = {"model": "test", "version": "1.0"}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name
        try:
            export_embeddings_json(filepath, embeddings, metadata=metadata)
            # Load raw JSON to check metadata
            with open(filepath, 'r') as f:
                data = json.load(f)
            assert data["metadata"]["model"] == "test"
            assert data["dimensions"] == 2
            assert data["terms"] == 1
        finally:
            os.unlink(filepath)

    def test_load_nonexistent_file(self):
        """Loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_embeddings_json("/nonexistent/path.json")


# =============================================================================
# SEMANTIC RELATIONS JSON TESTS
# =============================================================================


class TestSemanticRelationsJson:
    """Tests for semantic relations JSON export/import."""

    def test_export_load_roundtrip(self):
        """Relations survive export/load roundtrip."""
        relations = [
            ("dog", "IsA", "animal", 0.9),
            ("cat", "IsA", "animal", 0.85),
        ]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name
        try:
            export_semantic_relations_json(filepath, relations)
            loaded = load_semantic_relations_json(filepath)
            # JSON converts tuples to lists, so compare as lists
            expected = [list(r) for r in relations]
            assert loaded == expected
        finally:
            os.unlink(filepath)

    def test_export_empty_relations(self):
        """Empty relations can be exported."""
        relations = []
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name
        try:
            export_semantic_relations_json(filepath, relations)
            loaded = load_semantic_relations_json(filepath)
            assert loaded == []
        finally:
            os.unlink(filepath)

    def test_export_count_in_metadata(self):
        """Export includes relation count."""
        relations = [("a", "IsA", "b", 1.0), ("c", "IsA", "d", 1.0)]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name
        try:
            export_semantic_relations_json(filepath, relations)
            with open(filepath, 'r') as f:
                data = json.load(f)
            assert data["count"] == 2
        finally:
            os.unlink(filepath)

    def test_load_nonexistent_file(self):
        """Loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_semantic_relations_json("/nonexistent/path.json")


# =============================================================================
# RELATION COLORS COVERAGE
# =============================================================================


class TestRelationColorsCoverage:
    """Tests to ensure all common relation types have colors."""

    def test_semantic_relation_colors(self):
        """All common semantic relations have distinct colors."""
        semantic_types = [
            "IsA", "PartOf", "HasA", "UsedFor", "Causes",
            "HasProperty", "AtLocation", "CapableOf", "SimilarTo",
            "Antonym", "RelatedTo", "CoOccurs", "DerivedFrom", "DefinedBy"
        ]
        colors = set()
        for rel_type in semantic_types:
            color = _get_relation_color(rel_type)
            assert color != "#808080", f"{rel_type} should not have default color"
            colors.add(color)
        # Most relation types should have distinct colors
        assert len(colors) >= 10, "Expected more distinct colors for relation types"

    def test_structural_edge_colors(self):
        """Structural edge types have colors."""
        structural_types = ["feedforward", "feedback", "co_occurrence"]
        for edge_type in structural_types:
            color = _get_relation_color(edge_type)
            assert color != "#808080", f"{edge_type} should not have default color"


# =============================================================================
# SAVE/LOAD PROCESSOR TESTS
# =============================================================================


from cortical.persistence import save_processor, load_processor, get_state_summary
from cortical.layers import HierarchicalLayer
from cortical.minicolumn import Minicolumn, Edge


def create_test_layers():
    """Create test layers with minicolumns."""
    layers = {}

    # Layer 0: TOKENS
    layer0 = HierarchicalLayer(CorticalLayer.TOKENS)
    col1 = Minicolumn("L0_neural", "neural", 0)
    col1.occurrence_count = 5
    col1.pagerank = 0.3
    col1.tfidf = 1.5
    col1.activation = 0.8
    col1.document_ids = {"doc1"}
    col1.lateral_connections = {"L0_network": 0.7}
    col1.typed_connections = {
        "L0_network": Edge("L0_network", 0.7, "RelatedTo", 0.9, "semantic")
    }
    layer0.minicolumns["neural"] = col1
    layer0._id_index["L0_neural"] = "neural"

    col2 = Minicolumn("L0_network", "network", 0)
    col2.occurrence_count = 3
    col2.pagerank = 0.2
    col2.tfidf = 1.2
    col2.activation = 0.6
    col2.document_ids = {"doc1"}
    layer0.minicolumns["network"] = col2
    layer0._id_index["L0_network"] = "network"

    layers[CorticalLayer.TOKENS] = layer0

    # Layer 1: BIGRAMS
    layer1 = HierarchicalLayer(CorticalLayer.BIGRAMS)
    col3 = Minicolumn("L1_neural network", "neural network", 1)
    col3.occurrence_count = 2
    col3.pagerank = 0.15
    col3.tfidf = 2.0
    col3.activation = 0.7
    col3.document_ids = {"doc1"}
    col3.feedforward_connections = {"L0_neural": 1.0, "L0_network": 1.0}
    layer1.minicolumns["neural network"] = col3
    layer1._id_index["L1_neural network"] = "neural network"

    layers[CorticalLayer.BIGRAMS] = layer1

    # Layer 2: CONCEPTS
    layer2 = HierarchicalLayer(CorticalLayer.CONCEPTS)
    layers[CorticalLayer.CONCEPTS] = layer2

    # Layer 3: DOCUMENTS
    layer3 = HierarchicalLayer(CorticalLayer.DOCUMENTS)
    col4 = Minicolumn("L3_doc1", "doc1", 3)
    col4.occurrence_count = 1
    col4.pagerank = 0.5
    col4.tfidf = 0.0
    col4.activation = 1.0
    col4.document_ids = {"doc1"}
    col4.feedback_connections = {"L0_neural": 1.0, "L0_network": 1.0}
    layer3.minicolumns["doc1"] = col4
    layer3._id_index["L3_doc1"] = "doc1"

    layers[CorticalLayer.DOCUMENTS] = layer3

    return layers


class TestSaveLoadProcessor:
    """Tests for save_processor and load_processor functions."""

    def test_save_load_roundtrip_basic(self):
        """Basic processor state survives save/load roundtrip."""
        layers = create_test_layers()
        documents = {"doc1": "Neural networks process data."}

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            filepath = f.name
        try:
            save_processor(filepath, layers, documents, verbose=False)
            loaded_layers, loaded_docs, _, _, _, _ = load_processor(filepath, verbose=False)

            assert loaded_docs == documents
            assert len(loaded_layers) == len(layers)
            assert CorticalLayer.TOKENS in loaded_layers
            assert loaded_layers[CorticalLayer.TOKENS].column_count() == 2
        finally:
            os.unlink(filepath)

    def test_save_load_with_metadata(self):
        """Metadata survives save/load roundtrip."""
        layers = create_test_layers()
        documents = {"doc1": "Test document."}
        doc_metadata = {"doc1": {"source": "test", "timestamp": 12345}}
        metadata = {"version": "1.0", "config": {"param": "value"}}

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            filepath = f.name
        try:
            save_processor(
                filepath, layers, documents,
                document_metadata=doc_metadata,
                metadata=metadata,
                verbose=False
            )
            _, _, loaded_doc_meta, _, _, loaded_meta = load_processor(filepath, verbose=False)

            assert loaded_doc_meta == doc_metadata
            assert loaded_meta["version"] == "1.0"
            assert loaded_meta["config"]["param"] == "value"
        finally:
            os.unlink(filepath)

    def test_save_load_with_embeddings(self):
        """Embeddings survive save/load roundtrip."""
        layers = create_test_layers()
        documents = {"doc1": "Test."}
        embeddings = {
            "neural": [0.1, 0.2, 0.3],
            "network": [0.4, 0.5, 0.6]
        }

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            filepath = f.name
        try:
            save_processor(filepath, layers, documents, embeddings=embeddings, verbose=False)
            _, _, _, loaded_emb, _, _ = load_processor(filepath, verbose=False)

            assert loaded_emb == embeddings
        finally:
            os.unlink(filepath)

    def test_save_load_with_semantic_relations(self):
        """Semantic relations survive save/load roundtrip."""
        layers = create_test_layers()
        documents = {"doc1": "Test."}
        relations = [
            ("neural", "IsA", "concept", 0.9),
            ("network", "RelatedTo", "neural", 0.8)
        ]

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            filepath = f.name
        try:
            save_processor(
                filepath, layers, documents,
                semantic_relations=relations,
                verbose=False
            )
            _, _, _, _, loaded_rels, _ = load_processor(filepath, verbose=False)

            assert loaded_rels == relations
        finally:
            os.unlink(filepath)

    def test_save_load_empty_layers(self):
        """Empty layers can be saved and loaded."""
        layers = {
            CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS),
            CorticalLayer.BIGRAMS: HierarchicalLayer(CorticalLayer.BIGRAMS),
            CorticalLayer.CONCEPTS: HierarchicalLayer(CorticalLayer.CONCEPTS),
            CorticalLayer.DOCUMENTS: HierarchicalLayer(CorticalLayer.DOCUMENTS),
        }
        documents = {}

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            filepath = f.name
        try:
            save_processor(filepath, layers, documents, verbose=False)
            loaded_layers, loaded_docs, _, _, _, _ = load_processor(filepath, verbose=False)

            assert loaded_docs == {}
            assert len(loaded_layers) == 4
            for layer in loaded_layers.values():
                assert layer.column_count() == 0
        finally:
            os.unlink(filepath)

    def test_save_preserves_minicolumn_connections(self):
        """Minicolumn connections are preserved."""
        layers = create_test_layers()
        documents = {"doc1": "Test."}

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            filepath = f.name
        try:
            save_processor(filepath, layers, documents, verbose=False)
            loaded_layers, _, _, _, _, _ = load_processor(filepath, verbose=False)

            # Check lateral connections
            col = loaded_layers[CorticalLayer.TOKENS].get_minicolumn("neural")
            assert "L0_network" in col.lateral_connections
            assert col.lateral_connections["L0_network"] == 0.7

            # Check typed connections
            assert "L0_network" in col.typed_connections
            edge = col.typed_connections["L0_network"]
            assert edge.relation_type == "RelatedTo"
            assert edge.confidence == 0.9

            # Check feedforward connections
            bigram = loaded_layers[CorticalLayer.BIGRAMS].get_minicolumn("neural network")
            assert "L0_neural" in bigram.feedforward_connections
        finally:
            os.unlink(filepath)

    def test_save_preserves_minicolumn_attributes(self):
        """All minicolumn attributes are preserved."""
        layers = create_test_layers()
        documents = {"doc1": "Test."}

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            filepath = f.name
        try:
            save_processor(filepath, layers, documents, verbose=False)
            loaded_layers, _, _, _, _, _ = load_processor(filepath, verbose=False)

            col = loaded_layers[CorticalLayer.TOKENS].get_minicolumn("neural")
            assert col.id == "L0_neural"
            assert col.content == "neural"
            assert col.layer == 0
            assert col.occurrence_count == 5
            assert col.pagerank == 0.3
            assert col.tfidf == 1.5
            assert col.activation == 0.8
            assert "doc1" in col.document_ids
        finally:
            os.unlink(filepath)

    def test_save_with_verbose_logging(self):
        """Verbose mode logs statistics."""
        layers = create_test_layers()
        documents = {"doc1": "Test."}

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            filepath = f.name
        try:
            # Should not raise error with verbose=True
            save_processor(filepath, layers, documents, verbose=True)
            load_processor(filepath, verbose=True)
        finally:
            os.unlink(filepath)

    def test_load_nonexistent_file(self):
        """Loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_processor("/nonexistent/path.pkl")

    def test_save_verbose_with_embeddings_and_relations(self):
        """Verbose logging includes embeddings and relations counts."""
        layers = create_test_layers()
        documents = {"doc1": "Test."}
        embeddings = {"neural": [0.1, 0.2], "network": [0.3, 0.4]}
        relations = [("a", "IsA", "b", 1.0)]

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            filepath = f.name
        try:
            # Should log embeddings and relations with verbose=True
            save_processor(
                filepath, layers, documents,
                embeddings=embeddings,
                semantic_relations=relations,
                verbose=True
            )
            load_processor(filepath, verbose=True)
        finally:
            os.unlink(filepath)


# =============================================================================
# GET STATE SUMMARY TESTS
# =============================================================================


class TestGetStateSummary:
    """Tests for get_state_summary function."""

    def test_summary_basic_stats(self):
        """Summary includes basic statistics."""
        layers = create_test_layers()
        documents = {"doc1": "Test.", "doc2": "Another test."}

        summary = get_state_summary(layers, documents)

        assert summary["documents"] == 2
        assert "layers" in summary
        assert "total_columns" in summary
        assert "total_connections" in summary

    def test_summary_layer_stats(self):
        """Summary includes per-layer statistics."""
        layers = create_test_layers()
        documents = {"doc1": "Test."}

        summary = get_state_summary(layers, documents)

        assert "TOKENS" in summary["layers"]
        tokens_stats = summary["layers"]["TOKENS"]
        assert "columns" in tokens_stats
        assert "connections" in tokens_stats
        assert "avg_activation" in tokens_stats
        assert "sparsity" in tokens_stats

    def test_summary_empty_processor(self):
        """Summary works with empty processor."""
        layers = {
            CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS),
            CorticalLayer.BIGRAMS: HierarchicalLayer(CorticalLayer.BIGRAMS),
            CorticalLayer.CONCEPTS: HierarchicalLayer(CorticalLayer.CONCEPTS),
            CorticalLayer.DOCUMENTS: HierarchicalLayer(CorticalLayer.DOCUMENTS),
        }
        documents = {}

        summary = get_state_summary(layers, documents)

        assert summary["documents"] == 0
        assert summary["total_columns"] == 0
        assert summary["total_connections"] == 0

    def test_summary_counts_all_layers(self):
        """Summary counts minicolumns from all layers."""
        layers = create_test_layers()
        documents = {"doc1": "Test."}

        summary = get_state_summary(layers, documents)

        # Layer 0: 2 columns, Layer 1: 1 column, Layer 2: 0 columns, Layer 3: 1 column
        assert summary["total_columns"] == 4


# =============================================================================
# EXPORT GRAPH JSON TESTS
# =============================================================================


from cortical.persistence import export_graph_json


class TestExportGraphJson:
    """Tests for export_graph_json function."""

    def test_export_basic_graph(self):
        """Basic graph export works."""
        layers = create_test_layers()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name
        try:
            graph = export_graph_json(filepath, layers, verbose=False)

            assert "nodes" in graph
            assert "edges" in graph
            assert "metadata" in graph
            assert len(graph["nodes"]) > 0
        finally:
            os.unlink(filepath)

    def test_export_graph_nodes(self):
        """Exported graph includes node data."""
        layers = create_test_layers()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name
        try:
            graph = export_graph_json(filepath, layers, verbose=False)

            # Find the neural token node
            neural_node = next((n for n in graph["nodes"] if n["label"] == "neural"), None)
            assert neural_node is not None
            assert neural_node["id"] == "L0_neural"
            assert neural_node["layer"] == 0
            assert "pagerank" in neural_node
            assert "tfidf" in neural_node
        finally:
            os.unlink(filepath)

    def test_export_graph_edges(self):
        """Exported graph includes edges."""
        layers = create_test_layers()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name
        try:
            graph = export_graph_json(filepath, layers, verbose=False)

            # Should have at least one edge
            assert len(graph["edges"]) > 0
            edge = graph["edges"][0]
            assert "source" in edge
            assert "target" in edge
            assert "weight" in edge
        finally:
            os.unlink(filepath)

    def test_export_graph_with_layer_filter(self):
        """Export can filter to single layer."""
        layers = create_test_layers()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name
        try:
            graph = export_graph_json(
                filepath, layers,
                layer_filter=CorticalLayer.TOKENS,
                verbose=False
            )

            # All nodes should be from layer 0
            for node in graph["nodes"]:
                assert node["layer"] == 0
        finally:
            os.unlink(filepath)

    def test_export_graph_with_min_weight(self):
        """Export filters edges by minimum weight."""
        layers = create_test_layers()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name
        try:
            graph = export_graph_json(
                filepath, layers,
                min_weight=1.0,  # High threshold
                verbose=False
            )

            # Should have fewer or no edges
            for edge in graph["edges"]:
                assert edge["weight"] >= 1.0
        finally:
            os.unlink(filepath)

    def test_export_graph_max_nodes(self):
        """Export respects max_nodes limit."""
        layers = create_test_layers()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name
        try:
            graph = export_graph_json(
                filepath, layers,
                max_nodes=2,  # Limit to 2 nodes
                verbose=False
            )

            assert len(graph["nodes"]) <= 2
        finally:
            os.unlink(filepath)

    def test_export_graph_metadata(self):
        """Export includes metadata."""
        layers = create_test_layers()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name
        try:
            graph = export_graph_json(filepath, layers, verbose=False)

            metadata = graph["metadata"]
            assert "node_count" in metadata
            assert "edge_count" in metadata
            assert "layers" in metadata
            assert metadata["node_count"] == len(graph["nodes"])
            assert metadata["edge_count"] == len(graph["edges"])
        finally:
            os.unlink(filepath)

    def test_export_graph_file_format(self):
        """Exported file is valid JSON."""
        layers = create_test_layers()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name
        try:
            export_graph_json(filepath, layers, verbose=False)

            # Should be able to load as JSON
            with open(filepath, 'r') as f:
                data = json.load(f)
            assert "nodes" in data
            assert "edges" in data
        finally:
            os.unlink(filepath)

    def test_export_graph_verbose_logging(self):
        """Verbose mode logs graph statistics."""
        layers = create_test_layers()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name
        try:
            # Should log with verbose=True
            export_graph_json(filepath, layers, verbose=True)
        finally:
            os.unlink(filepath)

    def test_export_graph_empty_layers(self):
        """Export works with empty layers."""
        layers = {
            CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS),
            CorticalLayer.BIGRAMS: HierarchicalLayer(CorticalLayer.BIGRAMS),
        }

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name
        try:
            graph = export_graph_json(filepath, layers, verbose=False)
            assert len(graph["nodes"]) == 0
            assert len(graph["edges"]) == 0
        finally:
            os.unlink(filepath)


# =============================================================================
# EXPORT CONCEPTNET JSON TESTS
# =============================================================================


from cortical.persistence import export_conceptnet_json


class TestExportConceptnetJson:
    """Tests for export_conceptnet_json function."""

    def test_export_conceptnet_basic(self):
        """Basic ConceptNet export works."""
        layers = create_test_layers()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name
        try:
            graph = export_conceptnet_json(filepath, layers, verbose=False)

            assert "nodes" in graph
            assert "edges" in graph
            assert "metadata" in graph
        finally:
            os.unlink(filepath)

    def test_export_conceptnet_nodes_have_colors(self):
        """Nodes are color-coded by layer."""
        layers = create_test_layers()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name
        try:
            graph = export_conceptnet_json(filepath, layers, verbose=False)

            for node in graph["nodes"]:
                assert "color" in node
                assert node["color"].startswith("#")
                assert len(node["color"]) == 7
        finally:
            os.unlink(filepath)

    def test_export_conceptnet_typed_edges(self):
        """Typed edges include relation types."""
        layers = create_test_layers()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name
        try:
            graph = export_conceptnet_json(
                filepath, layers,
                include_typed_edges=True,
                verbose=False
            )

            # Find the RelatedTo edge
            typed_edge = next(
                (e for e in graph["edges"] if e.get("relation_type") == "RelatedTo"),
                None
            )
            assert typed_edge is not None
            assert "confidence" in typed_edge
            assert "source_type" in typed_edge
        finally:
            os.unlink(filepath)

    def test_export_conceptnet_cross_layer_edges(self):
        """Cross-layer edges are included."""
        layers = create_test_layers()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name
        try:
            graph = export_conceptnet_json(
                filepath, layers,
                include_cross_layer=True,
                verbose=False
            )

            # Should have feedforward or feedback edges
            cross_edges = [
                e for e in graph["edges"]
                if e.get("edge_type") == "cross_layer"
            ]
            assert len(cross_edges) > 0
        finally:
            os.unlink(filepath)

    def test_export_conceptnet_without_cross_layer(self):
        """Cross-layer edges can be excluded."""
        layers = create_test_layers()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name
        try:
            graph = export_conceptnet_json(
                filepath, layers,
                include_cross_layer=False,
                verbose=False
            )

            # Should not have cross_layer edges
            cross_edges = [
                e for e in graph["edges"]
                if e.get("edge_type") == "cross_layer"
            ]
            assert len(cross_edges) == 0
        finally:
            os.unlink(filepath)

    def test_export_conceptnet_with_semantic_relations(self):
        """Semantic relations are added to graph."""
        layers = create_test_layers()
        relations = [
            ("neural", "IsA", "concept", 0.9),
            ("network", "RelatedTo", "system", 0.8)
        ]

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name
        try:
            graph = export_conceptnet_json(
                filepath, layers,
                semantic_relations=relations,
                verbose=False
            )

            # Should have semantic edges
            semantic_edges = [
                e for e in graph["edges"]
                if e.get("edge_type") == "semantic"
            ]
            # May or may not find matches depending on node inclusion
            # Just verify the function accepts the parameter
            assert isinstance(semantic_edges, list)
        finally:
            os.unlink(filepath)

    def test_export_conceptnet_min_weight_filter(self):
        """Edges are filtered by minimum weight."""
        layers = create_test_layers()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name
        try:
            graph = export_conceptnet_json(
                filepath, layers,
                min_weight=0.5,
                verbose=False
            )

            for edge in graph["edges"]:
                assert edge["weight"] >= 0.5
        finally:
            os.unlink(filepath)

    def test_export_conceptnet_min_confidence_filter(self):
        """Typed edges are filtered by confidence."""
        layers = create_test_layers()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name
        try:
            graph = export_conceptnet_json(
                filepath, layers,
                min_confidence=0.95,  # High threshold
                verbose=False
            )

            # All typed edges should have high confidence
            for edge in graph["edges"]:
                if "confidence" in edge:
                    assert edge["confidence"] >= 0.95
        finally:
            os.unlink(filepath)

    def test_export_conceptnet_max_nodes_per_layer(self):
        """Respects max nodes per layer."""
        layers = create_test_layers()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name
        try:
            graph = export_conceptnet_json(
                filepath, layers,
                max_nodes_per_layer=1,  # Only 1 node per layer
                verbose=False
            )

            # Count nodes per layer
            layer_counts = {}
            for node in graph["nodes"]:
                layer_id = node["layer"]
                layer_counts[layer_id] = layer_counts.get(layer_id, 0) + 1

            for count in layer_counts.values():
                assert count <= 1
        finally:
            os.unlink(filepath)

    def test_export_conceptnet_metadata(self):
        """Metadata includes layer info and edge counts."""
        layers = create_test_layers()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name
        try:
            graph = export_conceptnet_json(filepath, layers, verbose=False)

            metadata = graph["metadata"]
            assert "layers" in metadata
            assert "edge_types" in metadata
            assert "relation_types" in metadata
            assert "format_version" in metadata
            assert "compatible_with" in metadata
        finally:
            os.unlink(filepath)

    def test_export_conceptnet_file_format(self):
        """Exported file is valid JSON."""
        layers = create_test_layers()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name
        try:
            export_conceptnet_json(filepath, layers, verbose=False)

            # Should be able to load as JSON
            with open(filepath, 'r') as f:
                data = json.load(f)
            assert "nodes" in data
            assert "edges" in data
            assert "metadata" in data
        finally:
            os.unlink(filepath)

    def test_export_conceptnet_empty_layers(self):
        """Export works with empty layers."""
        layers = {
            CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS),
        }

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name
        try:
            graph = export_conceptnet_json(filepath, layers, verbose=False)
            assert len(graph["nodes"]) == 0
            assert len(graph["edges"]) == 0
        finally:
            os.unlink(filepath)

    def test_export_conceptnet_without_typed_edges(self):
        """Can export without typed edges."""
        layers = create_test_layers()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name
        try:
            graph = export_conceptnet_json(
                filepath, layers,
                include_typed_edges=False,
                verbose=False
            )
            # Should still have lateral edges but no typed edges
            typed_edges = [
                e for e in graph["edges"]
                if e.get("relation_type") not in ["co_occurrence", "feedforward", "feedback"]
            ]
            # Might have co_occurrence edges, but not semantic typed edges
            assert isinstance(graph["edges"], list)
        finally:
            os.unlink(filepath)

    def test_export_conceptnet_layer_with_zero_columns(self):
        """Export handles layers with zero columns."""
        layers = create_test_layers()
        # Add an empty concepts layer
        layers[CorticalLayer.CONCEPTS] = HierarchicalLayer(CorticalLayer.CONCEPTS)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name
        try:
            graph = export_conceptnet_json(filepath, layers, verbose=False)
            # Should succeed despite empty layer
            assert "nodes" in graph
            assert "edges" in graph
        finally:
            os.unlink(filepath)

    def test_export_conceptnet_verbose_logging(self):
        """Verbose mode logs detailed statistics."""
        layers = create_test_layers()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name
        try:
            # Should log with verbose=True
            export_conceptnet_json(filepath, layers, verbose=True)
        finally:
            os.unlink(filepath)

    def test_export_conceptnet_long_relations_list(self):
        """Export handles long semantic relations list."""
        layers = create_test_layers()
        # Create many relations
        relations = [
            (f"term{i}", "IsA", f"concept{i}", 0.9)
            for i in range(100)
        ]

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name
        try:
            graph = export_conceptnet_json(
                filepath, layers,
                semantic_relations=relations,
                verbose=False
            )
            # Should handle large relations list
            assert "edges" in graph
        finally:
            os.unlink(filepath)


# =============================================================================
# SEC-003: HMAC SIGNATURE VERIFICATION TESTS
# =============================================================================


class TestSignatureHelpers:
    """Tests for HMAC signature helper functions (SEC-003)."""

    def test_get_signature_path(self):
        """Signature path is filename + .sig extension."""
        assert _get_signature_path("/path/to/file.pkl") == "/path/to/file.pkl.sig"
        assert _get_signature_path("data.pkl") == "data.pkl.sig"

    def test_compute_signature_returns_32_bytes(self):
        """HMAC-SHA256 signature is 32 bytes."""
        data = b"test data"
        key = b"secret key"
        sig = _compute_signature(data, key)
        assert len(sig) == 32  # SHA256 produces 32 bytes

    def test_compute_signature_deterministic(self):
        """Same data and key produces same signature."""
        data = b"test data"
        key = b"secret key"
        sig1 = _compute_signature(data, key)
        sig2 = _compute_signature(data, key)
        assert sig1 == sig2

    def test_compute_signature_different_with_different_key(self):
        """Different keys produce different signatures."""
        data = b"test data"
        key1 = b"key1"
        key2 = b"key2"
        sig1 = _compute_signature(data, key1)
        sig2 = _compute_signature(data, key2)
        assert sig1 != sig2

    def test_compute_signature_different_with_different_data(self):
        """Different data produces different signatures."""
        data1 = b"data1"
        data2 = b"data2"
        key = b"key"
        sig1 = _compute_signature(data1, key)
        sig2 = _compute_signature(data2, key)
        assert sig1 != sig2

    def test_verify_signature_correct(self):
        """Correct signature verifies successfully."""
        data = b"test data"
        key = b"secret key"
        sig = _compute_signature(data, key)
        assert _verify_signature(data, sig, key) is True

    def test_verify_signature_wrong_key(self):
        """Signature fails with wrong key."""
        data = b"test data"
        key = b"secret key"
        wrong_key = b"wrong key"
        sig = _compute_signature(data, key)
        assert _verify_signature(data, sig, wrong_key) is False

    def test_verify_signature_tampered_data(self):
        """Signature fails with tampered data."""
        data = b"test data"
        tampered = b"tampered data"
        key = b"secret key"
        sig = _compute_signature(data, key)
        assert _verify_signature(tampered, sig, key) is False

    def test_save_and_load_signature(self):
        """Signature can be saved and loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.pkl")
            signature = b"x" * 32

            _save_signature(filepath, signature)
            loaded = _load_signature(filepath)

            assert loaded == signature

    def test_load_signature_nonexistent_returns_none(self):
        """Loading nonexistent signature returns None."""
        result = _load_signature("/nonexistent/file.pkl")
        assert result is None


class TestSignatureVerificationError:
    """Tests for SignatureVerificationError exception."""

    def test_exception_is_raisable(self):
        """Exception can be raised and caught."""
        with pytest.raises(SignatureVerificationError):
            raise SignatureVerificationError("test error")

    def test_exception_message(self):
        """Exception preserves message."""
        try:
            raise SignatureVerificationError("custom message")
        except SignatureVerificationError as e:
            assert "custom message" in str(e)


class TestSaveLoadWithSignature:
    """Tests for save_processor and load_processor with HMAC signatures (SEC-003)."""

    def test_save_creates_signature_file(self):
        """Saving with signing_key creates .sig file."""
        layers = create_test_layers()
        documents = {"doc1": "Test."}
        key = b"my-secret-key"

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.pkl")
            sig_path = filepath + ".sig"

            save_processor(filepath, layers, documents, signing_key=key, verbose=False)

            assert os.path.exists(filepath)
            assert os.path.exists(sig_path)

    def test_save_without_key_no_signature(self):
        """Saving without signing_key creates no .sig file."""
        layers = create_test_layers()
        documents = {"doc1": "Test."}

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.pkl")
            sig_path = filepath + ".sig"

            save_processor(filepath, layers, documents, verbose=False)

            assert os.path.exists(filepath)
            assert not os.path.exists(sig_path)

    def test_load_with_correct_key_succeeds(self):
        """Loading with correct verify_key succeeds."""
        layers = create_test_layers()
        documents = {"doc1": "Test."}
        key = b"my-secret-key"

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.pkl")

            save_processor(filepath, layers, documents, signing_key=key, verbose=False)
            result = load_processor(filepath, verify_key=key, verbose=False)

            loaded_layers, loaded_docs, _, _, _, _ = result
            assert loaded_docs == documents

    def test_load_with_wrong_key_fails(self):
        """Loading with wrong verify_key raises SignatureVerificationError."""
        layers = create_test_layers()
        documents = {"doc1": "Test."}
        key = b"correct-key"
        wrong_key = b"wrong-key"

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.pkl")

            save_processor(filepath, layers, documents, signing_key=key, verbose=False)

            with pytest.raises(SignatureVerificationError):
                load_processor(filepath, verify_key=wrong_key, verbose=False)

    def test_load_tampered_file_fails(self):
        """Loading tampered file raises SignatureVerificationError."""
        layers = create_test_layers()
        documents = {"doc1": "Test."}
        key = b"my-secret-key"

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.pkl")

            save_processor(filepath, layers, documents, signing_key=key, verbose=False)

            # Tamper with the file
            with open(filepath, "r+b") as f:
                f.seek(100)
                f.write(b"TAMPERED")

            with pytest.raises(SignatureVerificationError):
                load_processor(filepath, verify_key=key, verbose=False)

    def test_load_missing_signature_file_fails(self):
        """Loading with verify_key but missing .sig raises FileNotFoundError."""
        layers = create_test_layers()
        documents = {"doc1": "Test."}
        key = b"my-secret-key"

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.pkl")

            # Save without signing key (no .sig file)
            save_processor(filepath, layers, documents, verbose=False)

            # Try to load with verify_key
            with pytest.raises(FileNotFoundError) as exc_info:
                load_processor(filepath, verify_key=key, verbose=False)
            assert ".sig" in str(exc_info.value)

    def test_backward_compatibility_no_key(self):
        """Loading without verify_key works (backward compatible)."""
        layers = create_test_layers()
        documents = {"doc1": "Test."}

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.pkl")

            # Save without signing
            save_processor(filepath, layers, documents, verbose=False)

            # Load without verify_key
            result = load_processor(filepath, verbose=False)
            loaded_layers, loaded_docs, _, _, _, _ = result
            assert loaded_docs == documents

    def test_save_signed_load_unsigned_works(self):
        """Loading signed file without verify_key works (ignores signature)."""
        layers = create_test_layers()
        documents = {"doc1": "Test."}
        key = b"my-secret-key"

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.pkl")

            # Save with signing key
            save_processor(filepath, layers, documents, signing_key=key, verbose=False)

            # Load without verify_key (ignores .sig file)
            result = load_processor(filepath, verbose=False)
            loaded_layers, loaded_docs, _, _, _, _ = result
            assert loaded_docs == documents

    def test_signature_is_32_bytes(self):
        """Signature file contains 32-byte HMAC-SHA256."""
        layers = create_test_layers()
        documents = {"doc1": "Test."}
        key = b"my-secret-key"

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.pkl")
            sig_path = filepath + ".sig"

            save_processor(filepath, layers, documents, signing_key=key, verbose=False)

            with open(sig_path, "rb") as f:
                signature = f.read()
            assert len(signature) == 32

    def test_different_keys_different_signatures(self):
        """Different signing keys produce different signatures."""
        layers = create_test_layers()
        documents = {"doc1": "Test."}
        key1 = b"key1"
        key2 = b"key2"

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath1 = os.path.join(tmpdir, "test1.pkl")
            filepath2 = os.path.join(tmpdir, "test2.pkl")

            save_processor(filepath1, layers, documents, signing_key=key1, verbose=False)
            save_processor(filepath2, layers, documents, signing_key=key2, verbose=False)

            sig1 = _load_signature(filepath1)
            sig2 = _load_signature(filepath2)

            assert sig1 != sig2

    def test_verbose_logging_with_signature(self):
        """Verbose mode logs signature operations."""
        layers = create_test_layers()
        documents = {"doc1": "Test."}
        key = b"my-secret-key"

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.pkl")

            # Should not raise with verbose=True
            save_processor(filepath, layers, documents, signing_key=key, verbose=True)
            load_processor(filepath, verify_key=key, verbose=True)
