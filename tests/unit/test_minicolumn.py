"""
Unit Tests for Minicolumn Module
=================================

Task #162: Unit tests for cortical/minicolumn.py core data structures.

Tests the Minicolumn and Edge classes that form the core data structures
of the cortical text processor. These classes store connections, metadata,
and support serialization.

Coverage goal: 90% (from 31%)
Test count goal: 35+
"""

import pytest

from cortical.minicolumn import Minicolumn, Edge


# =============================================================================
# EDGE CLASS TESTS
# =============================================================================


class TestEdgeClass:
    """Tests for the Edge dataclass."""

    def test_edge_creation_defaults(self):
        """Edge created with minimal parameters uses defaults."""
        edge = Edge(target_id="L0_test")
        assert edge.target_id == "L0_test"
        assert edge.weight == 1.0
        assert edge.relation_type == "co_occurrence"
        assert edge.confidence == 1.0
        assert edge.source == "corpus"

    def test_edge_creation_all_params(self):
        """Edge created with all parameters."""
        edge = Edge(
            target_id="L0_network",
            weight=0.8,
            relation_type="RelatedTo",
            confidence=0.9,
            source="semantic"
        )
        assert edge.target_id == "L0_network"
        assert edge.weight == 0.8
        assert edge.relation_type == "RelatedTo"
        assert edge.confidence == 0.9
        assert edge.source == "semantic"

    def test_edge_to_dict(self):
        """Edge serializes to dictionary."""
        edge = Edge(
            target_id="L0_test",
            weight=2.5,
            relation_type="IsA",
            confidence=0.7,
            source="inferred"
        )
        d = edge.to_dict()
        assert d["target_id"] == "L0_test"
        assert d["weight"] == 2.5
        assert d["relation_type"] == "IsA"
        assert d["confidence"] == 0.7
        assert d["source"] == "inferred"

    def test_edge_from_dict_minimal(self):
        """Edge deserializes from minimal dict."""
        d = {"target_id": "L0_test"}
        edge = Edge.from_dict(d)
        assert edge.target_id == "L0_test"
        assert edge.weight == 1.0
        assert edge.relation_type == "co_occurrence"
        assert edge.confidence == 1.0
        assert edge.source == "corpus"

    def test_edge_from_dict_complete(self):
        """Edge deserializes from complete dict."""
        d = {
            "target_id": "L0_network",
            "weight": 3.5,
            "relation_type": "PartOf",
            "confidence": 0.85,
            "source": "semantic"
        }
        edge = Edge.from_dict(d)
        assert edge.target_id == "L0_network"
        assert edge.weight == 3.5
        assert edge.relation_type == "PartOf"
        assert edge.confidence == 0.85
        assert edge.source == "semantic"

    def test_edge_round_trip_serialization(self):
        """Edge survives round-trip serialization."""
        original = Edge("L0_test", 1.5, "RelatedTo", 0.8, "corpus")
        d = original.to_dict()
        restored = Edge.from_dict(d)
        assert restored.target_id == original.target_id
        assert restored.weight == original.weight
        assert restored.relation_type == original.relation_type
        assert restored.confidence == original.confidence
        assert restored.source == original.source

    def test_edge_equality(self):
        """Two edges with same values are equal."""
        edge1 = Edge("L0_test", 1.0, "RelatedTo", 0.9, "corpus")
        edge2 = Edge("L0_test", 1.0, "RelatedTo", 0.9, "corpus")
        assert edge1 == edge2

    def test_edge_inequality_different_target(self):
        """Edges with different targets are not equal."""
        edge1 = Edge("L0_test1")
        edge2 = Edge("L0_test2")
        assert edge1 != edge2

    def test_edge_inequality_different_weight(self):
        """Edges with different weights are not equal."""
        edge1 = Edge("L0_test", weight=1.0)
        edge2 = Edge("L0_test", weight=2.0)
        assert edge1 != edge2


# =============================================================================
# MINICOLUMN INITIALIZATION TESTS
# =============================================================================


class TestMinicolumnInitialization:
    """Tests for Minicolumn initialization."""

    def test_basic_initialization(self):
        """Minicolumn initializes with required parameters."""
        col = Minicolumn("L0_test", "test", 0)
        assert col.id == "L0_test"
        assert col.content == "test"
        assert col.layer == 0

    def test_default_values(self):
        """Minicolumn has correct default values."""
        col = Minicolumn("L0_test", "test", 0)
        assert col.activation == 0.0
        assert col.occurrence_count == 0
        assert col.document_ids == set()
        assert col.lateral_connections == {}
        assert col.typed_connections == {}
        assert col.feedforward_sources == set()
        assert col.feedforward_connections == {}
        assert col.feedback_connections == {}
        assert col.tfidf == 0.0
        assert col.tfidf_per_doc == {}
        assert col.pagerank == 1.0
        assert col.cluster_id is None
        assert col.doc_occurrence_counts == {}

    def test_initialization_different_layers(self):
        """Minicolumns can be created for different layers."""
        col0 = Minicolumn("L0_token", "token", 0)
        col1 = Minicolumn("L1_bigram", "word pair", 1)
        col2 = Minicolumn("L2_concept", "concept", 2)
        col3 = Minicolumn("L3_doc", "doc1", 3)

        assert col0.layer == 0
        assert col1.layer == 1
        assert col2.layer == 2
        assert col3.layer == 3

    def test_repr(self):
        """Minicolumn has useful string representation."""
        col = Minicolumn("L0_neural", "neural", 0)
        repr_str = repr(col)
        assert "L0_neural" in repr_str
        assert "neural" in repr_str
        assert "layer=0" in repr_str or "0" in repr_str


# =============================================================================
# LATERAL CONNECTION MANAGEMENT TESTS
# =============================================================================


class TestLateralConnections:
    """Tests for lateral connection management."""

    def test_add_single_connection(self):
        """Add a single lateral connection."""
        col = Minicolumn("L0_test", "test", 0)
        col.add_lateral_connection("L0_target", 0.5)
        assert "L0_target" in col.lateral_connections
        assert col.lateral_connections["L0_target"] == 0.5

    def test_add_connection_default_weight(self):
        """Add connection with default weight of 1.0."""
        col = Minicolumn("L0_test", "test", 0)
        col.add_lateral_connection("L0_target")
        assert col.lateral_connections["L0_target"] == 1.0

    def test_add_connection_accumulates(self):
        """Adding to existing connection accumulates weight."""
        col = Minicolumn("L0_test", "test", 0)
        col.add_lateral_connection("L0_target", 0.5)
        col.add_lateral_connection("L0_target", 0.3)
        assert col.lateral_connections["L0_target"] == pytest.approx(0.8)

    def test_add_multiple_different_connections(self):
        """Add connections to multiple targets."""
        col = Minicolumn("L0_test", "test", 0)
        col.add_lateral_connection("L0_target1", 1.0)
        col.add_lateral_connection("L0_target2", 2.0)
        col.add_lateral_connection("L0_target3", 3.0)

        assert len(col.lateral_connections) == 3
        assert col.lateral_connections["L0_target1"] == 1.0
        assert col.lateral_connections["L0_target2"] == 2.0
        assert col.lateral_connections["L0_target3"] == 3.0

    def test_add_lateral_connections_batch(self):
        """Batch add multiple connections at once."""
        col = Minicolumn("L0_test", "test", 0)
        connections = {
            "L0_target1": 1.0,
            "L0_target2": 2.0,
            "L0_target3": 3.0
        }
        col.add_lateral_connections_batch(connections)

        assert len(col.lateral_connections) == 3
        assert col.lateral_connections["L0_target1"] == 1.0
        assert col.lateral_connections["L0_target2"] == 2.0
        assert col.lateral_connections["L0_target3"] == 3.0

    def test_batch_add_accumulates(self):
        """Batch add accumulates with existing connections."""
        col = Minicolumn("L0_test", "test", 0)
        col.add_lateral_connection("L0_target1", 1.0)

        connections = {
            "L0_target1": 2.0,  # Should accumulate
            "L0_target2": 3.0   # New connection
        }
        col.add_lateral_connections_batch(connections)

        assert col.lateral_connections["L0_target1"] == 3.0
        assert col.lateral_connections["L0_target2"] == 3.0

    def test_connection_count(self):
        """connection_count returns number of lateral connections."""
        col = Minicolumn("L0_test", "test", 0)
        assert col.connection_count() == 0

        col.add_lateral_connection("L0_target1", 1.0)
        assert col.connection_count() == 1

        col.add_lateral_connection("L0_target2", 2.0)
        assert col.connection_count() == 2

        # Adding to existing doesn't increase count
        col.add_lateral_connection("L0_target1", 1.0)
        assert col.connection_count() == 2

    def test_top_connections_empty(self):
        """top_connections returns empty list when no connections."""
        col = Minicolumn("L0_test", "test", 0)
        top = col.top_connections(5)
        assert top == []

    def test_top_connections_sorted(self):
        """top_connections returns connections sorted by weight."""
        col = Minicolumn("L0_test", "test", 0)
        col.add_lateral_connection("L0_weak", 0.1)
        col.add_lateral_connection("L0_strong", 5.0)
        col.add_lateral_connection("L0_medium", 2.0)

        top = col.top_connections(5)
        assert len(top) == 3
        assert top[0] == ("L0_strong", 5.0)
        assert top[1] == ("L0_medium", 2.0)
        assert top[2] == ("L0_weak", 0.1)

    def test_top_connections_limit(self):
        """top_connections respects the limit parameter."""
        col = Minicolumn("L0_test", "test", 0)
        col.add_lateral_connection("L0_1", 1.0)
        col.add_lateral_connection("L0_2", 2.0)
        col.add_lateral_connection("L0_3", 3.0)
        col.add_lateral_connection("L0_4", 4.0)
        col.add_lateral_connection("L0_5", 5.0)

        top3 = col.top_connections(3)
        assert len(top3) == 3
        assert top3[0][0] == "L0_5"
        assert top3[1][0] == "L0_4"
        assert top3[2][0] == "L0_3"


# =============================================================================
# TYPED CONNECTION MANAGEMENT TESTS
# =============================================================================


class TestTypedConnections:
    """Tests for typed connection management."""

    def test_add_typed_connection_defaults(self):
        """Add typed connection with default parameters."""
        col = Minicolumn("L0_test", "test", 0)
        col.add_typed_connection("L0_target")

        edge = col.typed_connections["L0_target"]
        assert edge.target_id == "L0_target"
        assert edge.weight == 1.0
        assert edge.relation_type == "co_occurrence"
        assert edge.confidence == 1.0
        assert edge.source == "corpus"

    def test_add_typed_connection_all_params(self):
        """Add typed connection with all parameters."""
        col = Minicolumn("L0_test", "test", 0)
        col.add_typed_connection(
            "L0_target",
            weight=2.5,
            relation_type="IsA",
            confidence=0.8,
            source="semantic"
        )

        edge = col.typed_connections["L0_target"]
        assert edge.target_id == "L0_target"
        assert edge.weight == 2.5
        assert edge.relation_type == "IsA"
        assert edge.confidence == 0.8
        assert edge.source == "semantic"

    def test_typed_connection_accumulates_weight(self):
        """Adding to existing typed connection accumulates weight."""
        col = Minicolumn("L0_test", "test", 0)
        col.add_typed_connection("L0_target", weight=1.0)
        col.add_typed_connection("L0_target", weight=2.0)

        edge = col.typed_connections["L0_target"]
        assert edge.weight == 3.0

    def test_typed_connection_weighted_confidence(self):
        """Confidence is weighted average when accumulating."""
        col = Minicolumn("L0_test", "test", 0)
        # First: weight=2.0, confidence=1.0
        col.add_typed_connection("L0_target", weight=2.0, confidence=1.0)
        # Second: weight=2.0, confidence=0.5
        col.add_typed_connection("L0_target", weight=2.0, confidence=0.5)

        edge = col.typed_connections["L0_target"]
        # Weighted average: (1.0*2.0 + 0.5*2.0) / 4.0 = 3.0/4.0 = 0.75
        assert edge.confidence == pytest.approx(0.75)

    def test_typed_connection_relation_priority(self):
        """Non-co_occurrence relation types are preferred."""
        col = Minicolumn("L0_test", "test", 0)
        col.add_typed_connection("L0_target", relation_type="IsA")
        col.add_typed_connection("L0_target", relation_type="co_occurrence")

        edge = col.typed_connections["L0_target"]
        # Should keep IsA, not replace with co_occurrence
        assert edge.relation_type == "IsA"

    def test_typed_connection_relation_priority_reverse(self):
        """Non-co_occurrence relation replaces co_occurrence."""
        col = Minicolumn("L0_test", "test", 0)
        col.add_typed_connection("L0_target", relation_type="co_occurrence")
        col.add_typed_connection("L0_target", relation_type="PartOf")

        edge = col.typed_connections["L0_target"]
        # Should upgrade to PartOf
        assert edge.relation_type == "PartOf"

    def test_typed_connection_source_priority(self):
        """Source priority: inferred > semantic > corpus."""
        col = Minicolumn("L0_test", "test", 0)

        # corpus -> semantic: should upgrade
        col.add_typed_connection("L0_target1", source="corpus")
        col.add_typed_connection("L0_target1", source="semantic")
        assert col.typed_connections["L0_target1"].source == "semantic"

        # semantic -> inferred: should upgrade
        col.add_typed_connection("L0_target2", source="semantic")
        col.add_typed_connection("L0_target2", source="inferred")
        assert col.typed_connections["L0_target2"].source == "inferred"

        # inferred -> corpus: should keep inferred
        col.add_typed_connection("L0_target3", source="inferred")
        col.add_typed_connection("L0_target3", source="corpus")
        assert col.typed_connections["L0_target3"].source == "inferred"

    def test_typed_connection_updates_lateral(self):
        """Adding typed connection also updates lateral_connections."""
        col = Minicolumn("L0_test", "test", 0)
        col.add_typed_connection("L0_target", weight=2.5)

        # Both typed and lateral should be updated
        assert "L0_target" in col.typed_connections
        assert "L0_target" in col.lateral_connections
        assert col.lateral_connections["L0_target"] == 2.5

    def test_get_typed_connection_exists(self):
        """get_typed_connection returns edge if exists."""
        col = Minicolumn("L0_test", "test", 0)
        col.add_typed_connection("L0_target", weight=1.5, relation_type="IsA")

        edge = col.get_typed_connection("L0_target")
        assert edge is not None
        assert edge.target_id == "L0_target"
        assert edge.weight == 1.5
        assert edge.relation_type == "IsA"

    def test_get_typed_connection_missing(self):
        """get_typed_connection returns None if not exists."""
        col = Minicolumn("L0_test", "test", 0)
        edge = col.get_typed_connection("L0_nonexistent")
        assert edge is None

    def test_get_connections_by_type(self):
        """get_connections_by_type filters by relation type."""
        col = Minicolumn("L0_test", "test", 0)
        col.add_typed_connection("L0_target1", relation_type="IsA")
        col.add_typed_connection("L0_target2", relation_type="PartOf")
        col.add_typed_connection("L0_target3", relation_type="IsA")
        col.add_typed_connection("L0_target4", relation_type="RelatedTo")

        isa_edges = col.get_connections_by_type("IsA")
        assert len(isa_edges) == 2
        target_ids = {e.target_id for e in isa_edges}
        assert target_ids == {"L0_target1", "L0_target3"}

    def test_get_connections_by_type_empty(self):
        """get_connections_by_type returns empty list if no matches."""
        col = Minicolumn("L0_test", "test", 0)
        col.add_typed_connection("L0_target", relation_type="IsA")

        edges = col.get_connections_by_type("NonExistent")
        assert edges == []

    def test_get_connections_by_source(self):
        """get_connections_by_source filters by source."""
        col = Minicolumn("L0_test", "test", 0)
        col.add_typed_connection("L0_target1", source="corpus")
        col.add_typed_connection("L0_target2", source="semantic")
        col.add_typed_connection("L0_target3", source="semantic")
        col.add_typed_connection("L0_target4", source="inferred")

        semantic_edges = col.get_connections_by_source("semantic")
        assert len(semantic_edges) == 2
        target_ids = {e.target_id for e in semantic_edges}
        assert target_ids == {"L0_target2", "L0_target3"}

    def test_get_connections_by_source_empty(self):
        """get_connections_by_source returns empty list if no matches."""
        col = Minicolumn("L0_test", "test", 0)
        col.add_typed_connection("L0_target", source="corpus")

        edges = col.get_connections_by_source("semantic")
        assert edges == []


# =============================================================================
# FEEDFORWARD/FEEDBACK CONNECTION TESTS
# =============================================================================


class TestFeedforwardFeedbackConnections:
    """Tests for feedforward and feedback connections."""

    def test_add_feedforward_connection(self):
        """Add feedforward connection to lower layer."""
        col = Minicolumn("L1_bigram", "word pair", 1)
        col.add_feedforward_connection("L0_word", 1.0)

        assert "L0_word" in col.feedforward_connections
        assert col.feedforward_connections["L0_word"] == 1.0

    def test_feedforward_accumulates(self):
        """Feedforward connections accumulate weight."""
        col = Minicolumn("L1_bigram", "word pair", 1)
        col.add_feedforward_connection("L0_word", 1.0)
        col.add_feedforward_connection("L0_word", 2.0)

        assert col.feedforward_connections["L0_word"] == 3.0

    def test_feedforward_updates_legacy_sources(self):
        """Adding feedforward also updates feedforward_sources (legacy)."""
        col = Minicolumn("L1_bigram", "word pair", 1)
        col.add_feedforward_connection("L0_word1", 1.0)
        col.add_feedforward_connection("L0_word2", 2.0)

        assert "L0_word1" in col.feedforward_sources
        assert "L0_word2" in col.feedforward_sources
        assert len(col.feedforward_sources) == 2

    def test_add_feedback_connection(self):
        """Add feedback connection to higher layer."""
        col = Minicolumn("L0_word", "word", 0)
        col.add_feedback_connection("L1_bigram", 1.0)

        assert "L1_bigram" in col.feedback_connections
        assert col.feedback_connections["L1_bigram"] == 1.0

    def test_feedback_accumulates(self):
        """Feedback connections accumulate weight."""
        col = Minicolumn("L0_word", "word", 0)
        col.add_feedback_connection("L1_bigram", 1.0)
        col.add_feedback_connection("L1_bigram", 2.0)

        assert col.feedback_connections["L1_bigram"] == 3.0

    def test_feedforward_and_feedback_independent(self):
        """Feedforward and feedback connections are independent."""
        col = Minicolumn("L1_bigram", "word pair", 1)
        col.add_feedforward_connection("L0_word", 1.0)
        col.add_feedback_connection("L2_concept", 2.0)

        assert len(col.feedforward_connections) == 1
        assert len(col.feedback_connections) == 1
        assert "L0_word" in col.feedforward_connections
        assert "L2_concept" in col.feedback_connections


# =============================================================================
# SERIALIZATION TESTS
# =============================================================================


class TestSerialization:
    """Tests for Minicolumn serialization and deserialization."""

    def test_to_dict_minimal(self):
        """Minimal minicolumn serializes correctly."""
        col = Minicolumn("L0_test", "test", 0)
        d = col.to_dict()

        assert d["id"] == "L0_test"
        assert d["content"] == "test"
        assert d["layer"] == 0
        assert d["activation"] == 0.0
        assert d["occurrence_count"] == 0
        assert d["document_ids"] == []
        assert d["lateral_connections"] == {}
        assert d["typed_connections"] == {}

    def test_from_dict_minimal(self):
        """Minimal dict deserializes to minicolumn."""
        d = {
            "id": "L0_test",
            "content": "test",
            "layer": 0
        }
        col = Minicolumn.from_dict(d)

        assert col.id == "L0_test"
        assert col.content == "test"
        assert col.layer == 0
        # Check defaults
        assert col.activation == 0.0
        assert col.occurrence_count == 0
        assert col.pagerank == 1.0

    def test_round_trip_basic(self):
        """Basic minicolumn survives round-trip."""
        original = Minicolumn("L0_neural", "neural", 0)
        original.activation = 5.0
        original.occurrence_count = 10
        original.pagerank = 0.5

        d = original.to_dict()
        restored = Minicolumn.from_dict(d)

        assert restored.id == original.id
        assert restored.content == original.content
        assert restored.layer == original.layer
        assert restored.activation == original.activation
        assert restored.occurrence_count == original.occurrence_count
        assert restored.pagerank == original.pagerank

    def test_round_trip_with_lateral_connections(self):
        """Minicolumn with lateral connections survives round-trip."""
        original = Minicolumn("L0_test", "test", 0)
        original.add_lateral_connection("L0_target1", 1.5)
        original.add_lateral_connection("L0_target2", 2.5)

        d = original.to_dict()
        restored = Minicolumn.from_dict(d)

        assert restored.lateral_connections == original.lateral_connections
        assert restored.lateral_connections["L0_target1"] == 1.5
        assert restored.lateral_connections["L0_target2"] == 2.5

    def test_round_trip_with_typed_connections(self):
        """Minicolumn with typed connections survives round-trip."""
        original = Minicolumn("L0_test", "test", 0)
        original.add_typed_connection("L0_target1", 1.5, "IsA", 0.9, "semantic")
        original.add_typed_connection("L0_target2", 2.5, "PartOf", 0.7, "inferred")

        d = original.to_dict()
        restored = Minicolumn.from_dict(d)

        assert len(restored.typed_connections) == 2

        edge1 = restored.typed_connections["L0_target1"]
        assert edge1.weight == 1.5
        assert edge1.relation_type == "IsA"
        assert edge1.confidence == 0.9
        assert edge1.source == "semantic"

        edge2 = restored.typed_connections["L0_target2"]
        assert edge2.weight == 2.5
        assert edge2.relation_type == "PartOf"
        assert edge2.confidence == 0.7
        assert edge2.source == "inferred"

    def test_round_trip_with_document_ids(self):
        """Minicolumn with document_ids survives round-trip."""
        original = Minicolumn("L0_test", "test", 0)
        original.document_ids = {"doc1", "doc2", "doc3"}

        d = original.to_dict()
        restored = Minicolumn.from_dict(d)

        assert restored.document_ids == {"doc1", "doc2", "doc3"}

    def test_round_trip_with_tfidf_per_doc(self):
        """Minicolumn with tfidf_per_doc survives round-trip."""
        original = Minicolumn("L0_test", "test", 0)
        original.tfidf = 2.5
        original.tfidf_per_doc = {"doc1": 1.5, "doc2": 3.5}

        d = original.to_dict()
        restored = Minicolumn.from_dict(d)

        assert restored.tfidf == 2.5
        assert restored.tfidf_per_doc == {"doc1": 1.5, "doc2": 3.5}

    def test_round_trip_with_doc_occurrence_counts(self):
        """Minicolumn with doc_occurrence_counts survives round-trip."""
        original = Minicolumn("L0_test", "test", 0)
        original.doc_occurrence_counts = {"doc1": 5, "doc2": 3}

        d = original.to_dict()
        restored = Minicolumn.from_dict(d)

        assert restored.doc_occurrence_counts == {"doc1": 5, "doc2": 3}

    def test_round_trip_with_feedforward_feedback(self):
        """Minicolumn with feedforward/feedback connections survives round-trip."""
        original = Minicolumn("L1_bigram", "word pair", 1)
        original.add_feedforward_connection("L0_word1", 1.0)
        original.add_feedforward_connection("L0_word2", 2.0)
        original.add_feedback_connection("L2_concept", 3.0)

        d = original.to_dict()
        restored = Minicolumn.from_dict(d)

        assert restored.feedforward_connections == original.feedforward_connections
        assert restored.feedback_connections == original.feedback_connections
        assert restored.feedforward_sources == original.feedforward_sources

    def test_round_trip_with_cluster_id(self):
        """Minicolumn with cluster_id survives round-trip."""
        original = Minicolumn("L0_test", "test", 0)
        original.cluster_id = 5

        d = original.to_dict()
        restored = Minicolumn.from_dict(d)

        assert restored.cluster_id == 5

    def test_round_trip_complete_minicolumn(self):
        """Fully populated minicolumn survives round-trip."""
        original = Minicolumn("L0_neural", "neural", 0)

        # Set all attributes
        original.activation = 3.5
        original.occurrence_count = 15
        original.document_ids = {"doc1", "doc2"}
        original.add_lateral_connection("L0_network", 2.0)
        original.add_typed_connection("L0_brain", 1.5, "IsA", 0.8, "semantic")
        original.add_feedforward_connection("L0_component", 1.0)
        original.add_feedback_connection("L1_bigram", 2.0)
        original.tfidf = 4.5
        original.tfidf_per_doc = {"doc1": 3.0, "doc2": 6.0}
        original.pagerank = 0.7
        original.cluster_id = 3
        original.doc_occurrence_counts = {"doc1": 10, "doc2": 5}

        # Round-trip
        d = original.to_dict()
        restored = Minicolumn.from_dict(d)

        # Verify all attributes
        assert restored.id == "L0_neural"
        assert restored.content == "neural"
        assert restored.layer == 0
        assert restored.activation == 3.5
        assert restored.occurrence_count == 15
        assert restored.document_ids == {"doc1", "doc2"}
        assert restored.lateral_connections["L0_network"] == 2.0
        assert "L0_brain" in restored.typed_connections
        assert restored.feedforward_connections["L0_component"] == 1.0
        assert restored.feedback_connections["L1_bigram"] == 2.0
        assert restored.tfidf == 4.5
        assert restored.tfidf_per_doc == {"doc1": 3.0, "doc2": 6.0}
        assert restored.pagerank == 0.7
        assert restored.cluster_id == 3
        assert restored.doc_occurrence_counts == {"doc1": 10, "doc2": 5}
