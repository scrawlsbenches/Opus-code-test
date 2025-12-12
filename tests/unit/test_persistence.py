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
