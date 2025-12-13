"""
Unit Tests for Layers Module
==============================

Task #161: Unit tests for cortical/layers.py.

Tests the HierarchicalLayer class and CorticalLayer enum:
- Layer initialization and structure
- CRUD operations (add, get, remove)
- O(1) ID index lookups and consistency
- Statistics (counts, connections, activations)
- Sparsity calculations
- Top-N queries (pagerank, tfidf, activation)
- Iteration and container protocols
- Serialization (to_dict/from_dict)
- CorticalLayer enum properties

Coverage target: 90%+
"""

import pytest

from cortical.layers import CorticalLayer, HierarchicalLayer
from cortical.minicolumn import Minicolumn


# =============================================================================
# CORTICAL LAYER ENUM TESTS
# =============================================================================


class TestCorticalLayerEnum:
    """Tests for CorticalLayer enumeration."""

    def test_layer_values(self):
        """Layer enum has correct integer values."""
        assert CorticalLayer.TOKENS == 0
        assert CorticalLayer.BIGRAMS == 1
        assert CorticalLayer.CONCEPTS == 2
        assert CorticalLayer.DOCUMENTS == 3

    def test_description_property(self):
        """Each layer has a description."""
        assert "Token layer" in CorticalLayer.TOKENS.description
        assert "Bigram layer" in CorticalLayer.BIGRAMS.description
        assert "Concept layer" in CorticalLayer.CONCEPTS.description
        assert "Document layer" in CorticalLayer.DOCUMENTS.description

    def test_analogy_property(self):
        """Each layer has a visual cortex analogy."""
        assert "V1" in CorticalLayer.TOKENS.analogy
        assert "V2" in CorticalLayer.BIGRAMS.analogy
        assert "V4" in CorticalLayer.CONCEPTS.analogy
        assert "IT" in CorticalLayer.DOCUMENTS.analogy

    def test_enum_equality(self):
        """Enum values can be compared."""
        assert CorticalLayer.TOKENS == CorticalLayer.TOKENS
        assert CorticalLayer.TOKENS != CorticalLayer.BIGRAMS


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestHierarchicalLayerInit:
    """Tests for HierarchicalLayer initialization."""

    def test_init_token_layer(self):
        """Initialize token layer (Layer 0)."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        assert layer.level == CorticalLayer.TOKENS
        assert layer.level == 0
        assert len(layer.minicolumns) == 0
        assert len(layer._id_index) == 0

    def test_init_bigram_layer(self):
        """Initialize bigram layer (Layer 1)."""
        layer = HierarchicalLayer(CorticalLayer.BIGRAMS)
        assert layer.level == CorticalLayer.BIGRAMS
        assert layer.level == 1
        assert len(layer.minicolumns) == 0

    def test_init_concept_layer(self):
        """Initialize concept layer (Layer 2)."""
        layer = HierarchicalLayer(CorticalLayer.CONCEPTS)
        assert layer.level == CorticalLayer.CONCEPTS
        assert layer.level == 2

    def test_init_document_layer(self):
        """Initialize document layer (Layer 3)."""
        layer = HierarchicalLayer(CorticalLayer.DOCUMENTS)
        assert layer.level == CorticalLayer.DOCUMENTS
        assert layer.level == 3

    def test_init_empty_state(self):
        """New layer has empty state."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        assert layer.column_count() == 0
        assert layer.total_connections() == 0
        assert layer.average_activation() == 0.0


# =============================================================================
# CRUD OPERATIONS TESTS
# =============================================================================


class TestCRUDOperations:
    """Tests for create, read, update, delete operations."""

    def test_get_or_create_new(self):
        """get_or_create_minicolumn creates new minicolumn."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        col = layer.get_or_create_minicolumn("neural")

        assert col.content == "neural"
        assert col.id == "L0_neural"
        assert col.layer == 0
        assert layer.column_count() == 1

    def test_get_or_create_existing(self):
        """get_or_create_minicolumn returns existing minicolumn."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        col1 = layer.get_or_create_minicolumn("neural")
        col1.occurrence_count = 5

        col2 = layer.get_or_create_minicolumn("neural")

        assert col2 is col1
        assert col2.occurrence_count == 5
        assert layer.column_count() == 1

    def test_get_or_create_multiple(self):
        """get_or_create_minicolumn handles multiple minicolumns."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        col1 = layer.get_or_create_minicolumn("neural")
        col2 = layer.get_or_create_minicolumn("network")
        col3 = layer.get_or_create_minicolumn("learning")

        assert layer.column_count() == 3
        assert col1.content == "neural"
        assert col2.content == "network"
        assert col3.content == "learning"

    def test_get_minicolumn_found(self):
        """get_minicolumn returns existing minicolumn."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        layer.get_or_create_minicolumn("neural")

        col = layer.get_minicolumn("neural")

        assert col is not None
        assert col.content == "neural"

    def test_get_minicolumn_not_found(self):
        """get_minicolumn returns None for non-existent content."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        layer.get_or_create_minicolumn("neural")

        col = layer.get_minicolumn("nonexistent")

        assert col is None

    def test_get_by_id_found(self):
        """get_by_id returns minicolumn via O(1) index lookup."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        layer.get_or_create_minicolumn("neural")

        col = layer.get_by_id("L0_neural")

        assert col is not None
        assert col.content == "neural"
        assert col.id == "L0_neural"

    def test_get_by_id_not_found(self):
        """get_by_id returns None for non-existent ID."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        layer.get_or_create_minicolumn("neural")

        col = layer.get_by_id("L0_nonexistent")

        assert col is None

    def test_get_by_id_different_layer(self):
        """get_by_id generates correct ID for different layers."""
        layer1 = HierarchicalLayer(CorticalLayer.BIGRAMS)
        layer1.get_or_create_minicolumn("neural network")

        col = layer1.get_by_id("L1_neural network")

        assert col is not None
        assert col.content == "neural network"
        assert col.layer == 1

    def test_remove_minicolumn_success(self):
        """remove_minicolumn removes existing minicolumn."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        layer.get_or_create_minicolumn("neural")

        removed = layer.remove_minicolumn("neural")

        assert removed is True
        assert layer.column_count() == 0
        assert layer.get_minicolumn("neural") is None

    def test_remove_minicolumn_not_found(self):
        """remove_minicolumn returns False for non-existent content."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        layer.get_or_create_minicolumn("neural")

        removed = layer.remove_minicolumn("nonexistent")

        assert removed is False
        assert layer.column_count() == 1

    def test_remove_minicolumn_cleans_id_index(self):
        """remove_minicolumn removes entry from _id_index."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        layer.get_or_create_minicolumn("neural")

        assert "L0_neural" in layer._id_index
        layer.remove_minicolumn("neural")

        assert "L0_neural" not in layer._id_index
        assert layer.get_by_id("L0_neural") is None

    def test_remove_one_of_many(self):
        """Removing one minicolumn doesn't affect others."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        layer.get_or_create_minicolumn("neural")
        layer.get_or_create_minicolumn("network")
        layer.get_or_create_minicolumn("learning")

        layer.remove_minicolumn("network")

        assert layer.column_count() == 2
        assert layer.get_minicolumn("neural") is not None
        assert layer.get_minicolumn("learning") is not None
        assert layer.get_minicolumn("network") is None


# =============================================================================
# ID INDEX CONSISTENCY TESTS
# =============================================================================


class TestIDIndexConsistency:
    """Tests for _id_index consistency and O(1) lookups."""

    def test_id_index_updated_on_create(self):
        """_id_index is updated when minicolumn is created."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)

        layer.get_or_create_minicolumn("neural")

        assert "L0_neural" in layer._id_index
        assert layer._id_index["L0_neural"] == "neural"

    def test_id_index_multiple_entries(self):
        """_id_index contains all minicolumn IDs."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        layer.get_or_create_minicolumn("neural")
        layer.get_or_create_minicolumn("network")
        layer.get_or_create_minicolumn("learning")

        assert len(layer._id_index) == 3
        assert "L0_neural" in layer._id_index
        assert "L0_network" in layer._id_index
        assert "L0_learning" in layer._id_index

    def test_id_index_consistent_with_minicolumns(self):
        """_id_index and minicolumns stay in sync."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        layer.get_or_create_minicolumn("neural")
        layer.get_or_create_minicolumn("network")

        # ID index size matches minicolumns size
        assert len(layer._id_index) == len(layer.minicolumns)

        # All minicolumns are in ID index
        for content, col in layer.minicolumns.items():
            assert col.id in layer._id_index
            assert layer._id_index[col.id] == content

    def test_get_by_id_vs_get_minicolumn(self):
        """get_by_id and get_minicolumn return same object."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        layer.get_or_create_minicolumn("neural")

        col_by_content = layer.get_minicolumn("neural")
        col_by_id = layer.get_by_id("L0_neural")

        assert col_by_id is col_by_content

    def test_id_index_after_removal(self):
        """_id_index stays consistent after removals."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        layer.get_or_create_minicolumn("neural")
        layer.get_or_create_minicolumn("network")
        layer.get_or_create_minicolumn("learning")

        layer.remove_minicolumn("network")

        assert len(layer._id_index) == 2
        assert "L0_neural" in layer._id_index
        assert "L0_learning" in layer._id_index
        assert "L0_network" not in layer._id_index

    def test_id_index_after_multiple_operations(self):
        """_id_index stays consistent after mixed operations."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)

        # Add
        layer.get_or_create_minicolumn("neural")
        layer.get_or_create_minicolumn("network")
        assert len(layer._id_index) == 2

        # Remove
        layer.remove_minicolumn("neural")
        assert len(layer._id_index) == 1

        # Add again (reuse content)
        layer.get_or_create_minicolumn("neural")
        assert len(layer._id_index) == 2

        # Verify consistency
        assert len(layer._id_index) == len(layer.minicolumns)

    def test_id_format_by_layer(self):
        """ID format is L{layer}_{content}."""
        token_layer = HierarchicalLayer(CorticalLayer.TOKENS)
        bigram_layer = HierarchicalLayer(CorticalLayer.BIGRAMS)
        concept_layer = HierarchicalLayer(CorticalLayer.CONCEPTS)

        token_col = token_layer.get_or_create_minicolumn("neural")
        bigram_col = bigram_layer.get_or_create_minicolumn("neural network")
        concept_col = concept_layer.get_or_create_minicolumn("ai")

        assert token_col.id == "L0_neural"
        assert bigram_col.id == "L1_neural network"
        assert concept_col.id == "L2_ai"


# =============================================================================
# STATISTICS & METRICS TESTS
# =============================================================================


class TestStatisticsAndMetrics:
    """Tests for layer statistics and metrics."""

    def test_column_count_empty(self):
        """column_count returns 0 for empty layer."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        assert layer.column_count() == 0

    def test_column_count_multiple(self):
        """column_count returns correct count."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        layer.get_or_create_minicolumn("neural")
        layer.get_or_create_minicolumn("network")
        layer.get_or_create_minicolumn("learning")

        assert layer.column_count() == 3

    def test_total_connections_empty(self):
        """total_connections returns 0 for empty layer."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        assert layer.total_connections() == 0

    def test_total_connections_no_connections(self):
        """total_connections returns 0 when no connections exist."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        layer.get_or_create_minicolumn("neural")
        layer.get_or_create_minicolumn("network")

        assert layer.total_connections() == 0

    def test_total_connections_with_connections(self):
        """total_connections sums all lateral connections."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        col1 = layer.get_or_create_minicolumn("neural")
        col2 = layer.get_or_create_minicolumn("network")

        col1.add_lateral_connection("L0_network", 1.0)
        col1.add_lateral_connection("L0_learning", 1.0)
        col2.add_lateral_connection("L0_neural", 1.0)

        assert layer.total_connections() == 3

    def test_average_activation_empty(self):
        """average_activation returns 0.0 for empty layer."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        assert layer.average_activation() == 0.0

    def test_average_activation_all_zero(self):
        """average_activation returns 0.0 when all activations are 0."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        layer.get_or_create_minicolumn("neural")
        layer.get_or_create_minicolumn("network")

        assert layer.average_activation() == 0.0

    def test_average_activation_calculation(self):
        """average_activation calculates correct average."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        col1 = layer.get_or_create_minicolumn("neural")
        col2 = layer.get_or_create_minicolumn("network")
        col3 = layer.get_or_create_minicolumn("learning")

        col1.activation = 1.0
        col2.activation = 2.0
        col3.activation = 3.0

        avg = layer.average_activation()
        assert avg == pytest.approx(2.0)

    def test_activation_range_empty(self):
        """activation_range returns (0.0, 0.0) for empty layer."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        min_act, max_act = layer.activation_range()

        assert min_act == 0.0
        assert max_act == 0.0

    def test_activation_range_single(self):
        """activation_range returns (value, value) for single column."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        col = layer.get_or_create_minicolumn("neural")
        col.activation = 5.0

        min_act, max_act = layer.activation_range()

        assert min_act == 5.0
        assert max_act == 5.0

    def test_activation_range_multiple(self):
        """activation_range returns correct (min, max)."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        col1 = layer.get_or_create_minicolumn("neural")
        col2 = layer.get_or_create_minicolumn("network")
        col3 = layer.get_or_create_minicolumn("learning")

        col1.activation = 1.0
        col2.activation = 5.0
        col3.activation = 3.0

        min_act, max_act = layer.activation_range()

        assert min_act == 1.0
        assert max_act == 5.0


# =============================================================================
# SPARSITY TESTS
# =============================================================================


class TestSparsity:
    """Tests for sparsity calculation."""

    def test_sparsity_empty_layer(self):
        """sparsity returns 0.0 for empty layer."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        assert layer.sparsity() == 0.0

    def test_sparsity_all_zero_activation(self):
        """sparsity returns 1.0 when all activations are 0."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        layer.get_or_create_minicolumn("neural")
        layer.get_or_create_minicolumn("network")
        layer.get_or_create_minicolumn("learning")

        # All activations default to 0.0
        sparsity = layer.sparsity()

        assert sparsity == 1.0

    def test_sparsity_all_equal_activation(self):
        """sparsity is 0.0 when all activations are equal and above threshold."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        col1 = layer.get_or_create_minicolumn("neural")
        col2 = layer.get_or_create_minicolumn("network")
        col3 = layer.get_or_create_minicolumn("learning")

        col1.activation = 5.0
        col2.activation = 5.0
        col3.activation = 5.0

        # Average = 5.0, threshold = 2.5 (50%), all are >= 2.5
        sparsity = layer.sparsity(threshold_fraction=0.5)

        assert sparsity == 0.0

    def test_sparsity_mixed_activation(self):
        """sparsity calculates fraction below threshold."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        col1 = layer.get_or_create_minicolumn("neural")
        col2 = layer.get_or_create_minicolumn("network")
        col3 = layer.get_or_create_minicolumn("learning")
        col4 = layer.get_or_create_minicolumn("deep")

        col1.activation = 10.0
        col2.activation = 1.0
        col3.activation = 1.0
        col4.activation = 0.0

        # Average = 3.0, threshold = 1.5 (50%)
        # Below threshold: col2 (1.0), col3 (1.0), col4 (0.0) = 3/4 = 0.75
        sparsity = layer.sparsity(threshold_fraction=0.5)

        assert sparsity == 0.75

    def test_sparsity_custom_threshold(self):
        """sparsity respects custom threshold_fraction."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        col1 = layer.get_or_create_minicolumn("neural")
        col2 = layer.get_or_create_minicolumn("network")

        col1.activation = 10.0
        col2.activation = 2.0

        # Average = 6.0
        # threshold_fraction=0.5 -> threshold = 3.0 -> col2 is below -> 0.5
        # threshold_fraction=0.1 -> threshold = 0.6 -> neither below -> 0.0

        high_threshold = layer.sparsity(threshold_fraction=0.5)
        low_threshold = layer.sparsity(threshold_fraction=0.1)

        assert high_threshold == 0.5
        assert low_threshold == 0.0

    def test_sparsity_half_and_half(self):
        """sparsity with half below, half above threshold."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        col1 = layer.get_or_create_minicolumn("a")
        col2 = layer.get_or_create_minicolumn("b")
        col3 = layer.get_or_create_minicolumn("c")
        col4 = layer.get_or_create_minicolumn("d")

        col1.activation = 10.0
        col2.activation = 10.0
        col3.activation = 0.0
        col4.activation = 0.0

        # Average = 5.0, threshold = 2.5
        # Below: c, d = 2/4 = 0.5
        sparsity = layer.sparsity(threshold_fraction=0.5)

        assert sparsity == 0.5


# =============================================================================
# TOP-N QUERIES TESTS
# =============================================================================


class TestTopNQueries:
    """Tests for top_by_pagerank, top_by_tfidf, top_by_activation."""

    def test_top_by_pagerank_empty(self):
        """top_by_pagerank returns empty list for empty layer."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        results = layer.top_by_pagerank(n=5)

        assert results == []

    def test_top_by_pagerank_sorted(self):
        """top_by_pagerank returns results sorted by pagerank."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        col1 = layer.get_or_create_minicolumn("neural")
        col2 = layer.get_or_create_minicolumn("network")
        col3 = layer.get_or_create_minicolumn("learning")

        col1.pagerank = 0.5
        col2.pagerank = 0.9
        col3.pagerank = 0.3

        results = layer.top_by_pagerank(n=3)

        assert len(results) == 3
        assert results[0] == ("network", 0.9)
        assert results[1] == ("neural", 0.5)
        assert results[2] == ("learning", 0.3)

    def test_top_by_pagerank_limit_n(self):
        """top_by_pagerank respects n limit."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        for i in range(10):
            col = layer.get_or_create_minicolumn(f"term{i}")
            col.pagerank = i * 0.1

        results = layer.top_by_pagerank(n=3)

        assert len(results) == 3
        # Should be top 3 by pagerank
        assert results[0][1] >= results[1][1]
        assert results[1][1] >= results[2][1]

    def test_top_by_tfidf_sorted(self):
        """top_by_tfidf returns results sorted by tfidf."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        col1 = layer.get_or_create_minicolumn("neural")
        col2 = layer.get_or_create_minicolumn("network")
        col3 = layer.get_or_create_minicolumn("learning")

        col1.tfidf = 1.5
        col2.tfidf = 3.0
        col3.tfidf = 0.5

        results = layer.top_by_tfidf(n=3)

        assert len(results) == 3
        assert results[0] == ("network", 3.0)
        assert results[1] == ("neural", 1.5)
        assert results[2] == ("learning", 0.5)

    def test_top_by_activation_sorted(self):
        """top_by_activation returns results sorted by activation."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        col1 = layer.get_or_create_minicolumn("neural")
        col2 = layer.get_or_create_minicolumn("network")
        col3 = layer.get_or_create_minicolumn("learning")

        col1.activation = 2.0
        col2.activation = 5.0
        col3.activation = 1.0

        results = layer.top_by_activation(n=3)

        assert len(results) == 3
        assert results[0] == ("network", 5.0)
        assert results[1] == ("neural", 2.0)
        assert results[2] == ("learning", 1.0)

    def test_top_by_pagerank_n_exceeds_count(self):
        """top_by_pagerank when n > column count."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        col1 = layer.get_or_create_minicolumn("neural")
        col2 = layer.get_or_create_minicolumn("network")

        col1.pagerank = 0.5
        col2.pagerank = 0.9

        results = layer.top_by_pagerank(n=10)

        # Should return only 2 items
        assert len(results) == 2

    def test_top_by_default_n(self):
        """top_by_* uses default n=10."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        for i in range(15):
            col = layer.get_or_create_minicolumn(f"term{i}")
            col.pagerank = i * 0.1

        results = layer.top_by_pagerank()  # No n parameter

        # Default is n=10
        assert len(results) == 10


# =============================================================================
# ITERATION & CONTAINER TESTS
# =============================================================================


class TestIterationAndContainer:
    """Tests for iteration and container protocol support."""

    def test_iter_empty(self):
        """Iterating over empty layer yields nothing."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        items = list(layer)

        assert items == []

    def test_iter_minicolumns(self):
        """Iterating yields Minicolumn objects."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        layer.get_or_create_minicolumn("neural")
        layer.get_or_create_minicolumn("network")
        layer.get_or_create_minicolumn("learning")

        items = list(layer)

        assert len(items) == 3
        assert all(isinstance(item, Minicolumn) for item in items)

    def test_iter_contents(self):
        """Iterating yields all minicolumns."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        layer.get_or_create_minicolumn("neural")
        layer.get_or_create_minicolumn("network")
        layer.get_or_create_minicolumn("learning")

        contents = {col.content for col in layer}

        assert contents == {"neural", "network", "learning"}

    def test_len_empty(self):
        """len() returns 0 for empty layer."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        assert len(layer) == 0

    def test_len_multiple(self):
        """len() returns correct count."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        layer.get_or_create_minicolumn("neural")
        layer.get_or_create_minicolumn("network")
        layer.get_or_create_minicolumn("learning")

        assert len(layer) == 3

    def test_contains_found(self):
        """'in' operator returns True for existing content."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        layer.get_or_create_minicolumn("neural")

        assert "neural" in layer

    def test_contains_not_found(self):
        """'in' operator returns False for non-existent content."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        layer.get_or_create_minicolumn("neural")

        assert "nonexistent" not in layer

    def test_multiple_iterations(self):
        """Layer can be iterated multiple times."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        layer.get_or_create_minicolumn("neural")
        layer.get_or_create_minicolumn("network")

        items1 = list(layer)
        items2 = list(layer)

        assert len(items1) == len(items2) == 2


# =============================================================================
# SERIALIZATION TESTS
# =============================================================================


class TestSerialization:
    """Tests for to_dict and from_dict serialization."""

    def test_to_dict_empty(self):
        """to_dict for empty layer."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        data = layer.to_dict()

        assert data['level'] == CorticalLayer.TOKENS
        assert data['minicolumns'] == {}

    def test_to_dict_structure(self):
        """to_dict creates correct structure."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        col = layer.get_or_create_minicolumn("neural")
        col.pagerank = 0.5
        col.tfidf = 1.2

        data = layer.to_dict()

        assert 'level' in data
        assert 'minicolumns' in data
        assert data['level'] == 0
        assert 'neural' in data['minicolumns']
        assert data['minicolumns']['neural']['content'] == 'neural'

    def test_from_dict_empty(self):
        """from_dict reconstructs empty layer."""
        original = HierarchicalLayer(CorticalLayer.TOKENS)
        data = original.to_dict()

        restored = HierarchicalLayer.from_dict(data)

        assert restored.level == CorticalLayer.TOKENS
        assert len(restored.minicolumns) == 0

    def test_from_dict_reconstruction(self):
        """from_dict reconstructs layer with minicolumns."""
        original = HierarchicalLayer(CorticalLayer.TOKENS)
        col1 = original.get_or_create_minicolumn("neural")
        col2 = original.get_or_create_minicolumn("network")
        col1.pagerank = 0.5
        col2.pagerank = 0.8

        data = original.to_dict()
        restored = HierarchicalLayer.from_dict(data)

        assert restored.level == original.level
        assert len(restored.minicolumns) == 2
        assert restored.get_minicolumn("neural").pagerank == 0.5
        assert restored.get_minicolumn("network").pagerank == 0.8

    def test_from_dict_rebuilds_id_index(self):
        """from_dict rebuilds _id_index correctly."""
        original = HierarchicalLayer(CorticalLayer.TOKENS)
        original.get_or_create_minicolumn("neural")
        original.get_or_create_minicolumn("network")

        data = original.to_dict()
        restored = HierarchicalLayer.from_dict(data)

        # ID index should be rebuilt
        assert len(restored._id_index) == 2
        assert "L0_neural" in restored._id_index
        assert "L0_network" in restored._id_index

        # get_by_id should work
        assert restored.get_by_id("L0_neural") is not None
        assert restored.get_by_id("L0_network") is not None

    def test_roundtrip_preserves_data(self):
        """to_dict -> from_dict preserves all data."""
        original = HierarchicalLayer(CorticalLayer.BIGRAMS)
        col1 = original.get_or_create_minicolumn("neural network")
        col2 = original.get_or_create_minicolumn("deep learning")

        col1.activation = 5.0
        col1.pagerank = 0.7
        col1.tfidf = 2.3
        col1.occurrence_count = 10
        col1.add_lateral_connection("L1_deep learning", 3.0)

        col2.activation = 3.0
        col2.pagerank = 0.5
        col2.tfidf = 1.8

        data = original.to_dict()
        restored = HierarchicalLayer.from_dict(data)

        # Check layer properties
        assert restored.level == CorticalLayer.BIGRAMS
        assert len(restored) == 2

        # Check minicolumn properties
        col1_restored = restored.get_minicolumn("neural network")
        assert col1_restored.activation == 5.0
        assert col1_restored.pagerank == 0.7
        assert col1_restored.tfidf == 2.3
        assert col1_restored.occurrence_count == 10
        assert col1_restored.lateral_connections["L1_deep learning"] == 3.0

    def test_from_dict_different_layers(self):
        """from_dict works for all layer types."""
        for layer_type in [CorticalLayer.TOKENS, CorticalLayer.BIGRAMS,
                          CorticalLayer.CONCEPTS, CorticalLayer.DOCUMENTS]:
            original = HierarchicalLayer(layer_type)
            original.get_or_create_minicolumn("test")

            data = original.to_dict()
            restored = HierarchicalLayer.from_dict(data)

            assert restored.level == layer_type
            assert len(restored) == 1


# =============================================================================
# REPR TESTS
# =============================================================================


class TestRepr:
    """Tests for __repr__ string representation."""

    def test_repr_format(self):
        """__repr__ returns expected format."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        layer.get_or_create_minicolumn("neural")
        layer.get_or_create_minicolumn("network")

        repr_str = repr(layer)

        assert "HierarchicalLayer" in repr_str
        assert "TOKENS" in repr_str
        assert "columns=2" in repr_str

    def test_repr_empty(self):
        """__repr__ for empty layer."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        repr_str = repr(layer)

        assert "columns=0" in repr_str

    def test_repr_different_layers(self):
        """__repr__ shows correct layer name."""
        token_layer = HierarchicalLayer(CorticalLayer.TOKENS)
        bigram_layer = HierarchicalLayer(CorticalLayer.BIGRAMS)

        assert "TOKENS" in repr(token_layer)
        assert "BIGRAMS" in repr(bigram_layer)
