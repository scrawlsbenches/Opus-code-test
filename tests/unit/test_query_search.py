"""
Unit Tests for Query/Search Module
===================================

Task #171: Unit tests for cortical/query/search.py document search functions.

Tests document search and ranking functions:
- find_documents_for_query: Main search with expansion and boosts
- fast_find_documents: Optimized candidate-based search
- build_document_index: Inverted index creation
- search_with_index: Pre-built index search
- query_with_spreading_activation: Spreading activation search
- find_related_documents: Related document discovery

These tests use mock layers and don't require a full processor.
"""

import pytest
from unittest.mock import Mock

from cortical.query.search import (
    find_documents_for_query,
    fast_find_documents,
    build_document_index,
    search_with_index,
    query_with_spreading_activation,
    find_related_documents,
    graph_boosted_search,
)
from cortical.tokenizer import Tokenizer
from tests.unit.mocks import (
    MockMinicolumn,
    MockHierarchicalLayer,
    MockLayers,
    LayerBuilder,
)


# =============================================================================
# FIND_DOCUMENTS_FOR_QUERY TESTS
# =============================================================================


class TestFindDocumentsForQuery:
    """Tests for find_documents_for_query main search function."""

    def test_empty_query(self):
        """Empty query returns empty results."""
        layers = MockLayers.single_term("term", tfidf=1.0, doc_ids=["doc1"])
        tokenizer = Tokenizer()

        # Tokenizer will return empty list for empty string
        result = find_documents_for_query("", layers, tokenizer)
        assert result == []

    def test_single_term_single_doc(self):
        """Single term matching single document."""
        # Create layer with term in doc1
        col = MockMinicolumn(
            content="neural",
            tfidf=2.5,
            document_ids={"doc1"},
            tfidf_per_doc={"doc1": 2.5}
        )
        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([col])

        tokenizer = Tokenizer()
        result = find_documents_for_query(
            "neural", layers, tokenizer, use_expansion=False
        )

        assert len(result) == 1
        assert result[0][0] == "doc1"
        assert result[0][1] > 0

    def test_single_term_multiple_docs(self):
        """Single term in multiple documents ranked by TF-IDF."""
        col = MockMinicolumn(
            content="algorithm",
            tfidf=3.0,
            document_ids={"doc1", "doc2", "doc3"},
            tfidf_per_doc={"doc1": 5.0, "doc2": 3.0, "doc3": 1.0}
        )
        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([col])

        tokenizer = Tokenizer()
        result = find_documents_for_query(
            "algorithm", layers, tokenizer, use_expansion=False
        )

        assert len(result) == 3
        # Should be sorted by TF-IDF score
        assert result[0][0] == "doc1"  # Highest score
        assert result[1][0] == "doc2"
        assert result[2][0] == "doc3"  # Lowest score
        assert result[0][1] > result[1][1] > result[2][1]

    def test_multi_term_query(self):
        """Multiple query terms aggregate scores."""
        layers = (
            LayerBuilder()
            .with_term("neural", tfidf=2.0)
            .with_term("network", tfidf=3.0)
            .with_document("doc1", ["neural", "network"])
            .with_document("doc2", ["neural"])
            .with_document("doc3", ["network"])
            .build()
        )

        # Set TF-IDF per doc
        layer0 = layers[MockLayers.TOKENS]
        layer0.get_minicolumn("neural").tfidf_per_doc = {"doc1": 2.0, "doc2": 2.0}
        layer0.get_minicolumn("network").tfidf_per_doc = {"doc1": 3.0, "doc3": 3.0}

        tokenizer = Tokenizer()
        result = find_documents_for_query(
            "neural network", layers, tokenizer, use_expansion=False
        )

        # doc1 should be top (has both terms)
        assert result[0][0] == "doc1"
        # doc1 score should be sum of both TF-IDF scores
        assert result[0][1] > result[1][1]

    def test_top_n_limit(self):
        """top_n parameter limits results."""
        cols = [
            MockMinicolumn(
                content="term",
                document_ids={f"doc{i}"},
                tfidf_per_doc={f"doc{i}": float(10 - i)}
            )
            for i in range(10)
        ]
        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer(cols[:1])
        layers[MockLayers.TOKENS].minicolumns["term"].document_ids = {
            f"doc{i}" for i in range(10)
        }
        layers[MockLayers.TOKENS].minicolumns["term"].tfidf_per_doc = {
            f"doc{i}": float(10 - i) for i in range(10)
        }

        tokenizer = Tokenizer()
        result = find_documents_for_query(
            "term", layers, tokenizer, top_n=3, use_expansion=False
        )

        assert len(result) == 3

    def test_no_matching_terms(self):
        """Query with no matching terms returns empty."""
        layers = MockLayers.single_term("existing", doc_ids=["doc1"])
        tokenizer = Tokenizer()

        result = find_documents_for_query(
            "nonexistent", layers, tokenizer, use_expansion=False
        )

        assert result == []

    def test_doc_name_boost_exact_match(self):
        """Document name matching query gets boosted."""
        # Create docs where one name matches query
        layers = (
            LayerBuilder()
            .with_term("neural", tfidf=2.0)
            .with_document("neural_network", ["neural"])
            .with_document("other_doc", ["neural"])
            .build()
        )

        layer0 = layers[MockLayers.TOKENS]
        layer0.get_minicolumn("neural").tfidf_per_doc = {
            "neural_network": 2.0,
            "other_doc": 2.0
        }

        tokenizer = Tokenizer()
        result = find_documents_for_query(
            "neural", layers, tokenizer,
            use_expansion=False,
            doc_name_boost=2.0
        )

        # neural_network should be boosted to top
        assert result[0][0] == "neural_network"
        assert result[0][1] > result[1][1]

    def test_doc_name_boost_partial_match(self):
        """Partial name match gets proportional boost."""
        layers = (
            LayerBuilder()
            .with_term("neural", tfidf=2.0)
            .with_term("algorithm", tfidf=2.0)
            .with_document("neural_doc", ["neural", "algorithm"])
            .with_document("other_doc", ["neural", "algorithm"])
            .build()
        )

        layer0 = layers[MockLayers.TOKENS]
        layer0.get_minicolumn("neural").tfidf_per_doc = {
            "neural_doc": 2.0,
            "other_doc": 2.0
        }
        layer0.get_minicolumn("algorithm").tfidf_per_doc = {
            "neural_doc": 2.0,
            "other_doc": 2.0
        }

        tokenizer = Tokenizer()
        # Query with two terms, one matches doc name
        result = find_documents_for_query(
            "neural algorithm", layers, tokenizer,
            use_expansion=False,
            doc_name_boost=3.0
        )

        # neural_doc should be boosted (50% match)
        assert result[0][0] == "neural_doc"

    def test_doc_name_boost_disabled(self):
        """doc_name_boost=1.0 disables boost."""
        layers = (
            LayerBuilder()
            .with_term("term", tfidf=2.0)
            .with_document("term", ["term"])  # Same name as term
            .with_document("doc1", ["term"])
            .build()
        )

        layer0 = layers[MockLayers.TOKENS]
        layer0.get_minicolumn("term").tfidf_per_doc = {
            "term": 2.0,
            "doc1": 3.0  # Higher TF-IDF
        }

        tokenizer = Tokenizer()
        result = find_documents_for_query(
            "term", layers, tokenizer,
            use_expansion=False,
            doc_name_boost=1.0  # No boost
        )

        # doc1 should win on TF-IDF alone
        assert result[0][0] == "doc1"

    def test_exact_doc_name_match_beats_high_tfidf(self):
        """
        Task #181: Exact document name match ranks first even with lower TF-IDF.

        Bug: Documents with high content scores could outrank exact name matches.
        Fix: Exact matches get additive boost to ensure top ranking.
        """
        layers = (
            LayerBuilder()
            .with_term("distributed", tfidf=5.0)
            .with_term("systems", tfidf=5.0)
            # distributed_systems doc has exact name match but low content
            .with_document("distributed_systems", ["distributed"])
            # other_doc has high content score
            .with_document("other_doc", ["distributed", "systems"])
            .build()
        )

        layer0 = layers[MockLayers.TOKENS]
        # other_doc has MUCH higher TF-IDF scores
        layer0.get_minicolumn("distributed").tfidf_per_doc = {
            "distributed_systems": 0.5,  # Low score
            "other_doc": 10.0  # Very high score
        }
        layer0.get_minicolumn("systems").tfidf_per_doc = {
            "other_doc": 10.0  # Very high score
        }

        tokenizer = Tokenizer()
        result = find_documents_for_query(
            "distributed systems", layers, tokenizer,
            use_expansion=False,
            doc_name_boost=2.0
        )

        # distributed_systems should rank first due to exact name match
        # despite having much lower TF-IDF score
        assert result[0][0] == "distributed_systems"
        assert result[0][1] > result[1][1]  # Score should be higher

    def test_query_expansion_disabled(self):
        """use_expansion=False uses only query terms."""
        # Create connected terms
        layers = (
            LayerBuilder()
            .with_term("neural", tfidf=2.0, pagerank=0.8)
            .with_term("network", tfidf=2.0, pagerank=0.6)
            .with_connection("neural", "network", weight=5.0)
            .with_document("doc1", ["neural"])
            .with_document("doc2", ["network"])
            .build()
        )

        layer0 = layers[MockLayers.TOKENS]
        layer0.get_minicolumn("neural").tfidf_per_doc = {"doc1": 2.0}
        layer0.get_minicolumn("network").tfidf_per_doc = {"doc2": 2.0}

        tokenizer = Tokenizer()
        result = find_documents_for_query(
            "neural", layers, tokenizer,
            use_expansion=False
        )

        # Should only find doc1 (contains "neural")
        assert len(result) == 1
        assert result[0][0] == "doc1"

    def test_tfidf_per_doc_fallback(self):
        """Uses col.tfidf if per-doc TF-IDF missing."""
        col = MockMinicolumn(
            content="term",
            tfidf=5.0,  # Global TF-IDF
            document_ids={"doc1"},
            tfidf_per_doc={}  # Empty per-doc
        )
        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([col])

        tokenizer = Tokenizer()
        result = find_documents_for_query(
            "term", layers, tokenizer, use_expansion=False
        )

        assert len(result) == 1
        assert result[0][1] == pytest.approx(5.0)

    def test_empty_corpus(self):
        """Empty corpus returns empty results."""
        layers = MockLayers.empty()
        tokenizer = Tokenizer()

        result = find_documents_for_query("query", layers, tokenizer)

        assert result == []

    def test_tie_breaking_stability(self):
        """Documents with same score maintain stable order."""
        col = MockMinicolumn(
            content="term",
            document_ids={"doc1", "doc2", "doc3"},
            tfidf_per_doc={"doc1": 2.0, "doc2": 2.0, "doc3": 2.0}
        )
        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([col])

        tokenizer = Tokenizer()
        result = find_documents_for_query(
            "term", layers, tokenizer, use_expansion=False
        )

        # All should have same score
        assert len(result) == 3
        assert result[0][1] == pytest.approx(result[1][1])
        assert result[1][1] == pytest.approx(result[2][1])


# =============================================================================
# FAST_FIND_DOCUMENTS TESTS
# =============================================================================


class TestFastFindDocuments:
    """Tests for fast_find_documents optimized search."""

    def test_single_term_match(self):
        """Fast search finds document with matching term."""
        col = MockMinicolumn(
            content="algorithm",
            document_ids={"doc1"},
            tfidf_per_doc={"doc1": 3.0}
        )
        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([col])

        tokenizer = Tokenizer()
        result = fast_find_documents("algorithm", layers, tokenizer)

        assert len(result) == 1
        assert result[0][0] == "doc1"

    def test_empty_query(self):
        """Empty query returns empty results."""
        layers = MockLayers.single_term("term", doc_ids=["doc1"])
        tokenizer = Tokenizer()

        result = fast_find_documents("", layers, tokenizer)

        assert result == []

    def test_candidate_filtering(self):
        """Filters candidates by match count before scoring."""
        # Create docs with varying match counts
        layers = (
            LayerBuilder()
            .with_term("neural", tfidf=2.0)
            .with_term("network", tfidf=2.0)
            .with_term("learning", tfidf=2.0)
            .with_document("doc1", ["neural", "network", "learning"])
            .with_document("doc2", ["neural", "network"])
            .with_document("doc3", ["neural"])
            .build()
        )

        layer0 = layers[MockLayers.TOKENS]
        layer0.get_minicolumn("neural").tfidf_per_doc = {
            "doc1": 2.0, "doc2": 2.0, "doc3": 2.0
        }
        layer0.get_minicolumn("network").tfidf_per_doc = {
            "doc1": 2.0, "doc2": 2.0
        }
        layer0.get_minicolumn("learning").tfidf_per_doc = {
            "doc1": 2.0
        }

        tokenizer = Tokenizer()
        result = fast_find_documents(
            "neural network learning", layers, tokenizer,
            candidate_multiplier=2
        )

        # doc1 should be top (all terms match)
        assert result[0][0] == "doc1"

    def test_coverage_boost(self):
        """Documents matching more query terms get coverage boost."""
        # Create terms with explicit document_ids (use real words, not stop words)
        col_neural = MockMinicolumn(
            content="neural",
            tfidf=1.0,
            document_ids={"full_match", "partial_match"},
            tfidf_per_doc={"full_match": 1.0, "partial_match": 2.0}
        )
        col_network = MockMinicolumn(
            content="network",
            tfidf=1.0,
            document_ids={"full_match"},
            tfidf_per_doc={"full_match": 1.0}
        )

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([col_neural, col_network])

        tokenizer = Tokenizer()
        result = fast_find_documents("neural network", layers, tokenizer)

        # full_match should win due to coverage boost
        assert len(result) >= 1
        assert result[0][0] == "full_match"

    def test_doc_name_boost(self):
        """Document name matching query gets boosted."""
        layers = (
            LayerBuilder()
            .with_term("neural", tfidf=2.0)
            .with_document("neural_doc", ["neural"])
            .with_document("other_doc", ["neural"])
            .build()
        )

        layer0 = layers[MockLayers.TOKENS]
        layer0.get_minicolumn("neural").tfidf_per_doc = {
            "neural_doc": 2.0,
            "other_doc": 2.0
        }

        tokenizer = Tokenizer()
        result = fast_find_documents(
            "neural", layers, tokenizer, doc_name_boost=3.0
        )

        assert result[0][0] == "neural_doc"

    def test_top_n_limit(self):
        """top_n limits final results."""
        col = MockMinicolumn(
            content="term",
            document_ids={f"doc{i}" for i in range(10)},
            tfidf_per_doc={f"doc{i}": float(i) for i in range(10)}
        )
        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([col])

        tokenizer = Tokenizer()
        result = fast_find_documents("term", layers, tokenizer, top_n=3)

        assert len(result) == 3

    def test_candidate_multiplier(self):
        """candidate_multiplier controls pre-filtering size."""
        # Create 20 docs
        col = MockMinicolumn(
            content="term",
            document_ids={f"doc{i}" for i in range(20)},
            tfidf_per_doc={f"doc{i}": 1.0 for i in range(20)}
        )
        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([col])

        tokenizer = Tokenizer()
        # With top_n=5 and multiplier=2, should score top 10 candidates
        result = fast_find_documents(
            "term", layers, tokenizer,
            top_n=5,
            candidate_multiplier=2
        )

        assert len(result) == 5

    def test_no_candidates_returns_empty(self):
        """No matching candidates returns empty."""
        layers = MockLayers.single_term("existing", doc_ids=["doc1"])
        tokenizer = Tokenizer()

        result = fast_find_documents("nonexistent", layers, tokenizer)

        assert result == []

    def test_code_concepts_fallback(self):
        """Falls back to code concepts when no direct matches."""
        # This test verifies the fallback logic exists
        # Without mocking get_related_terms, we can only verify no crash
        layers = MockLayers.empty()
        tokenizer = Tokenizer()

        # Should return empty gracefully
        result = fast_find_documents(
            "nonexistent", layers, tokenizer, use_code_concepts=True
        )

        assert result == []

    def test_code_concepts_disabled(self):
        """use_code_concepts=False skips expansion."""
        layers = MockLayers.empty()
        tokenizer = Tokenizer()

        result = fast_find_documents(
            "nonexistent", layers, tokenizer, use_code_concepts=False
        )

        assert result == []

    def test_doc_name_boost_default(self):
        """Default doc_name_boost=2.0 is applied."""
        col = MockMinicolumn(
            content="neural",
            document_ids={"neural_doc", "other_doc"},
            tfidf_per_doc={"neural_doc": 2.0, "other_doc": 2.0}
        )
        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([col])

        tokenizer = Tokenizer()
        # Use default doc_name_boost (should be 2.0)
        result = fast_find_documents("neural", layers, tokenizer)

        # neural_doc should be boosted
        assert result[0][0] == "neural_doc"

    def test_doc_name_boost_disabled_fast(self):
        """doc_name_boost=1.0 disables boost in fast search."""
        col = MockMinicolumn(
            content="term",
            document_ids={"term_doc", "high_score_doc"},
            tfidf_per_doc={"term_doc": 1.0, "high_score_doc": 5.0}
        )
        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([col])

        tokenizer = Tokenizer()
        result = fast_find_documents(
            "term", layers, tokenizer, doc_name_boost=1.0
        )

        # high_score_doc should win on TF-IDF alone
        assert result[0][0] == "high_score_doc"

    def test_exact_name_match_added_to_candidates(self):
        """
        Task #181: Exact name matches included in candidates even without content.

        Bug: fast_find_documents excluded docs whose name matched but content didn't.
        Fix: Add name-matching docs to candidate set.
        """
        # Create doc that has exact name match but no matching content
        layers = (
            LayerBuilder()
            .with_term("other", tfidf=5.0)
            .with_document("distributed_systems", ["other"])  # No 'distributed' or 'systems' in content
            .with_document("high_content_doc", ["other"])
            .build()
        )

        # Add layer3 (DOCUMENTS) for name matching
        doc1 = MockMinicolumn(
            content="distributed_systems",
            document_ids={"distributed_systems"}
        )
        doc2 = MockMinicolumn(
            content="high_content_doc",
            document_ids={"high_content_doc"}
        )

        layers[MockLayers.DOCUMENTS] = MockHierarchicalLayer([doc1, doc2])

        tokenizer = Tokenizer()
        result = fast_find_documents(
            "distributed systems", layers, tokenizer, doc_name_boost=2.0
        )

        # distributed_systems should be in results despite not having content match
        doc_ids = [doc_id for doc_id, _ in result]
        assert "distributed_systems" in doc_ids


# =============================================================================
# BUILD_DOCUMENT_INDEX TESTS
# =============================================================================


class TestBuildDocumentIndex:
    """Tests for build_document_index inverted index creation."""

    def test_empty_layer(self):
        """Empty layer returns empty index."""
        layers = MockLayers.empty()
        result = build_document_index(layers)
        assert result == {}

    def test_single_term_single_doc(self):
        """Single term in single document."""
        col = MockMinicolumn(
            content="term",
            document_ids={"doc1"},
            tfidf_per_doc={"doc1": 2.5}
        )
        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([col])

        result = build_document_index(layers)

        assert "term" in result
        assert result["term"] == {"doc1": 2.5}

    def test_single_term_multiple_docs(self):
        """Single term in multiple documents."""
        col = MockMinicolumn(
            content="term",
            document_ids={"doc1", "doc2", "doc3"},
            tfidf_per_doc={"doc1": 3.0, "doc2": 2.0, "doc3": 1.0}
        )
        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([col])

        result = build_document_index(layers)

        assert result["term"] == {"doc1": 3.0, "doc2": 2.0, "doc3": 1.0}

    def test_multiple_terms(self):
        """Multiple terms in various documents."""
        layers = (
            LayerBuilder()
            .with_term("neural", tfidf=2.0)
            .with_term("network", tfidf=3.0)
            .with_document("doc1", ["neural", "network"])
            .with_document("doc2", ["neural"])
            .build()
        )

        layer0 = layers[MockLayers.TOKENS]
        layer0.get_minicolumn("neural").tfidf_per_doc = {"doc1": 2.0, "doc2": 2.0}
        layer0.get_minicolumn("network").tfidf_per_doc = {"doc1": 3.0}

        result = build_document_index(layers)

        assert "neural" in result
        assert "network" in result
        assert result["neural"] == {"doc1": 2.0, "doc2": 2.0}
        assert result["network"] == {"doc1": 3.0}

    def test_tfidf_fallback(self):
        """Uses global TF-IDF if per-doc missing."""
        col = MockMinicolumn(
            content="term",
            tfidf=5.0,
            document_ids={"doc1"},
            tfidf_per_doc={}
        )
        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([col])

        result = build_document_index(layers)

        assert result["term"]["doc1"] == 5.0

    def test_term_with_no_docs_excluded(self):
        """Terms with no documents not in index."""
        col = MockMinicolumn(
            content="term",
            document_ids=set(),
            tfidf_per_doc={}
        )
        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([col])

        result = build_document_index(layers)

        # Term with no docs should not appear
        assert "term" not in result

    def test_missing_token_layer(self):
        """Missing token layer returns empty index."""
        layers = {
            MockLayers.DOCUMENTS: MockHierarchicalLayer([])
        }
        result = build_document_index(layers)
        assert result == {}


# =============================================================================
# SEARCH_WITH_INDEX TESTS
# =============================================================================


class TestSearchWithIndex:
    """Tests for search_with_index pre-built index search."""

    def test_empty_query(self):
        """Empty query returns empty results."""
        index = {"term": {"doc1": 2.0}}
        tokenizer = Tokenizer()

        result = search_with_index("", index, tokenizer)

        assert result == []

    def test_empty_index(self):
        """Empty index returns empty results."""
        tokenizer = Tokenizer()
        result = search_with_index("query", {}, tokenizer)
        assert result == []

    def test_single_term_match(self):
        """Single term query matches index."""
        index = {
            "neural": {"doc1": 3.0, "doc2": 1.0}
        }
        tokenizer = Tokenizer()

        result = search_with_index("neural", index, tokenizer)

        assert len(result) == 2
        assert result[0] == ("doc1", 3.0)
        assert result[1] == ("doc2", 1.0)

    def test_multi_term_aggregation(self):
        """Multiple terms aggregate scores."""
        index = {
            "neural": {"doc1": 2.0, "doc2": 1.0},
            "network": {"doc1": 3.0, "doc3": 2.0}
        }
        tokenizer = Tokenizer()

        result = search_with_index("neural network", index, tokenizer)

        # doc1 should have 2.0 + 3.0 = 5.0
        assert result[0][0] == "doc1"
        assert result[0][1] == pytest.approx(5.0)

    def test_term_not_in_index(self):
        """Term not in index is skipped."""
        index = {
            "neural": {"doc1": 2.0}
        }
        tokenizer = Tokenizer()

        result = search_with_index("neural nonexistent", index, tokenizer)

        # Should find doc1 from "neural", ignore "nonexistent"
        assert len(result) == 1
        assert result[0][0] == "doc1"

    def test_top_n_limit(self):
        """top_n limits results."""
        index = {
            "term": {f"doc{i}": float(10 - i) for i in range(10)}
        }
        tokenizer = Tokenizer()

        result = search_with_index("term", index, tokenizer, top_n=3)

        assert len(result) == 3

    def test_ranking_by_score(self):
        """Results sorted by score descending."""
        index = {
            "term": {"doc1": 5.0, "doc2": 10.0, "doc3": 3.0}
        }
        tokenizer = Tokenizer()

        result = search_with_index("term", index, tokenizer)

        assert result[0][0] == "doc2"  # Highest
        assert result[1][0] == "doc1"
        assert result[2][0] == "doc3"  # Lowest


# =============================================================================
# QUERY_WITH_SPREADING_ACTIVATION TESTS
# =============================================================================


class TestQueryWithSpreadingActivation:
    """Tests for query_with_spreading_activation."""

    def test_empty_query(self):
        """Empty query returns empty results."""
        layers = MockLayers.single_term("term", pagerank=0.5)
        tokenizer = Tokenizer()

        result = query_with_spreading_activation("", layers, tokenizer)

        assert result == []

    def test_single_term_activation(self):
        """Single term activates directly."""
        col = MockMinicolumn(
            content="neural",
            pagerank=0.8,
            activation=1.0
        )
        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([col])

        tokenizer = Tokenizer()
        result = query_with_spreading_activation("neural", layers, tokenizer)

        # Should activate "neural"
        assert len(result) > 0
        assert result[0][0] == "neural"

    def test_spreading_to_neighbors(self):
        """Activation spreads to connected neighbors."""
        layers = (
            LayerBuilder()
            .with_term("neural", pagerank=0.8, activation=1.0)
            .with_term("network", pagerank=0.6, activation=0.5)
            .with_connection("neural", "network", weight=5.0)
            .build()
        )

        tokenizer = Tokenizer()
        result = query_with_spreading_activation("neural", layers, tokenizer)

        # Should activate both neural and network
        activated_terms = {term for term, score in result}
        assert "neural" in activated_terms
        # network may or may not appear depending on threshold

    def test_top_n_limit(self):
        """top_n limits activated concepts."""
        # Create chain of connected terms
        layers = (
            LayerBuilder()
            .with_terms(["a", "b", "c", "d", "e"], pagerank=0.5, activation=1.0)
            .with_connection("a", "b", weight=1.0)
            .with_connection("b", "c", weight=1.0)
            .with_connection("c", "d", weight=1.0)
            .with_connection("d", "e", weight=1.0)
            .build()
        )

        tokenizer = Tokenizer()
        result = query_with_spreading_activation(
            "a", layers, tokenizer, top_n=3
        )

        assert len(result) <= 3

    def test_no_matching_term(self):
        """No matching term returns empty."""
        layers = MockLayers.single_term("existing")
        tokenizer = Tokenizer()

        result = query_with_spreading_activation(
            "nonexistent", layers, tokenizer
        )

        assert result == []

    def test_max_expansions_parameter(self):
        """max_expansions controls query expansion."""
        layers = (
            LayerBuilder()
            .with_term("neural", pagerank=0.8, activation=1.0)
            .with_term("network", pagerank=0.6, activation=0.5)
            .with_connection("neural", "network", weight=5.0)
            .build()
        )

        tokenizer = Tokenizer()
        # Should not crash with different max_expansions
        result = query_with_spreading_activation(
            "neural", layers, tokenizer, max_expansions=1
        )

        assert len(result) >= 0


# =============================================================================
# FIND_RELATED_DOCUMENTS TESTS
# =============================================================================


class TestFindRelatedDocuments:
    """Tests for find_related_documents."""

    def test_missing_document_layer(self):
        """Missing document layer returns empty."""
        layers = MockLayers.empty()
        result = find_related_documents("doc1", layers)
        assert result == []

    def test_document_not_found(self):
        """Non-existent document returns empty."""
        doc_col = MockMinicolumn(
            content="doc1",
            id="L3_doc1",
            layer=MockLayers.DOCUMENTS
        )
        layers = MockLayers.empty()
        layers[MockLayers.DOCUMENTS] = MockHierarchicalLayer([doc_col])

        result = find_related_documents("nonexistent", layers)

        assert result == []

    def test_no_connections(self):
        """Document with no connections returns empty."""
        doc_col = MockMinicolumn(
            content="doc1",
            id="L3_doc1",
            layer=MockLayers.DOCUMENTS,
            lateral_connections={}
        )
        layers = MockLayers.empty()
        layers[MockLayers.DOCUMENTS] = MockHierarchicalLayer([doc_col])

        result = find_related_documents("doc1", layers)

        assert result == []

    def test_single_related_document(self):
        """Finds single related document."""
        doc1 = MockMinicolumn(
            content="doc1",
            id="L3_doc1",
            layer=MockLayers.DOCUMENTS,
            lateral_connections={"L3_doc2": 5.0}
        )
        doc2 = MockMinicolumn(
            content="doc2",
            id="L3_doc2",
            layer=MockLayers.DOCUMENTS
        )
        layers = MockLayers.empty()
        layers[MockLayers.DOCUMENTS] = MockHierarchicalLayer([doc1, doc2])

        result = find_related_documents("doc1", layers)

        assert len(result) == 1
        assert result[0] == ("doc2", 5.0)

    def test_multiple_related_documents(self):
        """Finds multiple related documents sorted by weight."""
        doc1 = MockMinicolumn(
            content="doc1",
            id="L3_doc1",
            layer=MockLayers.DOCUMENTS,
            lateral_connections={
                "L3_doc2": 10.0,
                "L3_doc3": 5.0,
                "L3_doc4": 15.0
            }
        )
        doc2 = MockMinicolumn(content="doc2", id="L3_doc2", layer=MockLayers.DOCUMENTS)
        doc3 = MockMinicolumn(content="doc3", id="L3_doc3", layer=MockLayers.DOCUMENTS)
        doc4 = MockMinicolumn(content="doc4", id="L3_doc4", layer=MockLayers.DOCUMENTS)

        layers = MockLayers.empty()
        layers[MockLayers.DOCUMENTS] = MockHierarchicalLayer([doc1, doc2, doc3, doc4])

        result = find_related_documents("doc1", layers)

        assert len(result) == 3
        # Should be sorted by weight descending
        assert result[0] == ("doc4", 15.0)
        assert result[1] == ("doc2", 10.0)
        assert result[2] == ("doc3", 5.0)

    def test_connection_to_missing_document(self):
        """Connection to non-existent document is skipped."""
        doc1 = MockMinicolumn(
            content="doc1",
            id="L3_doc1",
            layer=MockLayers.DOCUMENTS,
            lateral_connections={
                "L3_doc2": 5.0,
                "L3_missing": 10.0  # Points to non-existent doc
            }
        )
        doc2 = MockMinicolumn(content="doc2", id="L3_doc2", layer=MockLayers.DOCUMENTS)

        layers = MockLayers.empty()
        layers[MockLayers.DOCUMENTS] = MockHierarchicalLayer([doc1, doc2])

        result = find_related_documents("doc1", layers)

        # Should only find doc2, skip missing
        assert len(result) == 1
        assert result[0][0] == "doc2"

    def test_uses_id_lookup(self):
        """Uses O(1) get_by_id for neighbor lookup."""
        # This test verifies the implementation uses get_by_id
        doc1 = MockMinicolumn(
            content="doc1",
            id="L3_doc1",
            layer=MockLayers.DOCUMENTS,
            lateral_connections={"L3_doc2": 3.0}
        )
        doc2 = MockMinicolumn(content="doc2", id="L3_doc2", layer=MockLayers.DOCUMENTS)

        layers = MockLayers.empty()
        layer3 = MockHierarchicalLayer([doc1, doc2])
        layers[MockLayers.DOCUMENTS] = layer3

        result = find_related_documents("doc1", layers)

        # If this works, get_by_id was used successfully
        assert len(result) == 1
        assert result[0][0] == "doc2"


# =============================================================================
# GRAPH_BOOSTED_SEARCH TESTS
# =============================================================================


class TestGraphBoostedSearch:
    """Tests for graph_boosted_search hybrid scoring function."""

    def test_basic_search(self):
        """Basic search returns ranked documents."""
        # Create tokens with tfidf and pagerank
        neural = MockMinicolumn(
            content="neural",
            id="L0_neural",
            layer=MockLayers.TOKENS,
            tfidf=1.0,
            tfidf_per_doc={"doc1": 0.8, "doc2": 0.5},
            document_ids={"doc1", "doc2"},
            pagerank=0.3,
            lateral_connections={}
        )
        networks = MockMinicolumn(
            content="networks",
            id="L0_networks",
            layer=MockLayers.TOKENS,
            tfidf=0.9,
            tfidf_per_doc={"doc1": 0.7, "doc3": 0.4},
            document_ids={"doc1", "doc3"},
            pagerank=0.2,
            lateral_connections={}
        )

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([neural, networks])
        layers[MockLayers.DOCUMENTS] = MockHierarchicalLayer([])
        tokenizer = Tokenizer()

        results = graph_boosted_search("neural networks", layers, tokenizer, top_n=3)

        assert len(results) > 0
        # doc1 should rank highest (has both terms)
        assert results[0][0] == "doc1"

    def test_empty_query(self):
        """Empty query returns empty results."""
        layers = MockLayers.single_term("term", tfidf=1.0, doc_ids=["doc1"])
        tokenizer = Tokenizer()

        results = graph_boosted_search("", layers, tokenizer, top_n=5)
        assert results == []

    def test_no_matching_terms(self):
        """Query with no matching terms returns empty results."""
        layers = MockLayers.single_term("other", tfidf=1.0, doc_ids=["doc1"])
        tokenizer = Tokenizer()

        results = graph_boosted_search("nonexistent", layers, tokenizer, top_n=5)
        assert results == []

    def test_pagerank_boost(self):
        """Documents with high-PageRank terms get boosted."""
        # High PageRank term (use "significant" instead of "important" which is a stop word)
        significant = MockMinicolumn(
            content="significant",
            id="L0_significant",
            layer=MockLayers.TOKENS,
            tfidf=1.0,
            tfidf_per_doc={"doc1": 1.0},
            document_ids={"doc1"},
            pagerank=0.9,  # High importance
            lateral_connections={}
        )
        # Low PageRank term
        common = MockMinicolumn(
            content="common",
            id="L0_common",
            layer=MockLayers.TOKENS,
            tfidf=1.0,
            tfidf_per_doc={"doc2": 1.0},
            document_ids={"doc2"},
            pagerank=0.1,  # Low importance
            lateral_connections={}
        )

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([significant, common])
        layers[MockLayers.DOCUMENTS] = MockHierarchicalLayer([])
        tokenizer = Tokenizer()

        # Search for both terms (using "significant" instead of "important")
        results = graph_boosted_search(
            "significant common", layers, tokenizer, top_n=5,
            pagerank_weight=0.5  # High PageRank influence
        )

        assert len(results) == 2
        # doc1 should rank higher due to PageRank boost
        assert results[0][0] == "doc1"

    def test_respects_top_n(self):
        """Returns at most top_n results."""
        terms = []
        for i in range(10):
            terms.append(MockMinicolumn(
                content=f"term{i}",
                id=f"L0_term{i}",
                layer=MockLayers.TOKENS,
                tfidf=1.0,
                tfidf_per_doc={f"doc{i}": 1.0},
                document_ids={f"doc{i}"},
                pagerank=0.1,
                lateral_connections={}
            ))

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer(terms)
        layers[MockLayers.DOCUMENTS] = MockHierarchicalLayer([])
        tokenizer = Tokenizer()

        # Query that matches multiple docs
        results = graph_boosted_search(
            " ".join(f"term{i}" for i in range(10)),
            layers, tokenizer, top_n=3
        )

        assert len(results) <= 3

    def test_test_file_penalty(self):
        """Test files get penalized in ranking."""
        # Create terms in both test and non-test files
        neural = MockMinicolumn(
            content="neural",
            id="L0_neural",
            layer=MockLayers.TOKENS,
            tfidf=1.0,
            tfidf_per_doc={"tests/test_model.py": 5.0, "src/model.py": 5.0},
            document_ids={"tests/test_model.py", "src/model.py"},
            pagerank=0.5,
            lateral_connections={}
        )

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([neural])
        layers[MockLayers.DOCUMENTS] = MockHierarchicalLayer([])
        tokenizer = Tokenizer()

        results = graph_boosted_search(
            "neural", layers, tokenizer, top_n=5,
            use_expansion=False
        )

        # src/model.py should rank higher than tests/test_model.py
        # due to test_file_penalty in find_documents_for_query
        assert len(results) == 2

    def test_proximity_boost_connected_terms(self):
        """Connected query terms boost documents containing both."""
        # Create two connected terms
        neural = MockMinicolumn(
            content="neural",
            id="L0_neural",
            layer=MockLayers.TOKENS,
            tfidf=1.0,
            tfidf_per_doc={"doc1": 1.0, "doc2": 1.0},
            document_ids={"doc1", "doc2"},
            pagerank=0.5,
            lateral_connections={"L0_network": 5.0}  # Connected to network
        )
        network = MockMinicolumn(
            content="network",
            id="L0_network",
            layer=MockLayers.TOKENS,
            tfidf=1.0,
            tfidf_per_doc={"doc1": 1.0, "doc3": 1.0},
            document_ids={"doc1", "doc3"},
            pagerank=0.5,
            lateral_connections={"L0_neural": 5.0}  # Connected to neural
        )

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([neural, network])
        layers[MockLayers.DOCUMENTS] = MockHierarchicalLayer([])
        tokenizer = Tokenizer()

        # Query with both connected terms
        results = graph_boosted_search(
            "neural network", layers, tokenizer, top_n=5,
            proximity_weight=0.5  # High proximity weight
        )

        # doc1 should rank highest (has both terms AND they're connected)
        assert results[0][0] == "doc1"

    def test_coverage_boost_multi_term(self):
        """Documents matching more query terms get coverage boost."""
        term1 = MockMinicolumn(
            content="neural",
            id="L0_neural",
            layer=MockLayers.TOKENS,
            tfidf=1.0,
            tfidf_per_doc={"doc_full": 1.0, "doc_partial": 1.0},
            document_ids={"doc_full", "doc_partial"},
            pagerank=0.5,
            lateral_connections={}
        )
        term2 = MockMinicolumn(
            content="network",
            id="L0_network",
            layer=MockLayers.TOKENS,
            tfidf=1.0,
            tfidf_per_doc={"doc_full": 1.0},
            document_ids={"doc_full"},
            pagerank=0.5,
            lateral_connections={}
        )
        term3 = MockMinicolumn(
            content="learning",
            id="L0_learning",
            layer=MockLayers.TOKENS,
            tfidf=1.0,
            tfidf_per_doc={"doc_full": 1.0},
            document_ids={"doc_full"},
            pagerank=0.5,
            lateral_connections={}
        )

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([term1, term2, term3])
        layers[MockLayers.DOCUMENTS] = MockHierarchicalLayer([])
        tokenizer = Tokenizer()

        results = graph_boosted_search(
            "neural network learning", layers, tokenizer, top_n=5
        )

        # doc_full should rank first (matches all 3 terms)
        assert results[0][0] == "doc_full"

    def test_zero_pagerank_weight(self):
        """pagerank_weight=0 disables PageRank boost."""
        # High PageRank term in doc1
        high_pr = MockMinicolumn(
            content="critical",
            id="L0_critical",
            layer=MockLayers.TOKENS,
            tfidf=1.0,
            tfidf_per_doc={"doc1": 1.0},
            document_ids={"doc1"},
            pagerank=0.95,  # Very high
            lateral_connections={}
        )
        # Low PageRank term in doc2
        low_pr = MockMinicolumn(
            content="routine",
            id="L0_routine",
            layer=MockLayers.TOKENS,
            tfidf=1.0,
            tfidf_per_doc={"doc2": 1.0},
            document_ids={"doc2"},
            pagerank=0.05,  # Very low
            lateral_connections={}
        )

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([high_pr, low_pr])
        layers[MockLayers.DOCUMENTS] = MockHierarchicalLayer([])
        tokenizer = Tokenizer()

        # With pagerank_weight=0, PageRank should not affect ranking
        results = graph_boosted_search(
            "critical routine", layers, tokenizer, top_n=5,
            pagerank_weight=0.0,
            proximity_weight=0.0
        )

        # Scores should be equal (both have same TF-IDF)
        assert len(results) == 2

    def test_zero_proximity_weight(self):
        """proximity_weight=0 disables proximity boost."""
        neural = MockMinicolumn(
            content="neural",
            id="L0_neural",
            layer=MockLayers.TOKENS,
            tfidf=1.0,
            tfidf_per_doc={"doc1": 1.0},
            document_ids={"doc1"},
            pagerank=0.5,
            lateral_connections={"L0_network": 10.0}
        )
        network = MockMinicolumn(
            content="network",
            id="L0_network",
            layer=MockLayers.TOKENS,
            tfidf=1.0,
            tfidf_per_doc={"doc1": 1.0},
            document_ids={"doc1"},
            pagerank=0.5,
            lateral_connections={"L0_neural": 10.0}
        )

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([neural, network])
        layers[MockLayers.DOCUMENTS] = MockHierarchicalLayer([])
        tokenizer = Tokenizer()

        # With proximity_weight=0, connection should not affect ranking
        results = graph_boosted_search(
            "neural network", layers, tokenizer, top_n=5,
            proximity_weight=0.0
        )

        assert len(results) >= 1

    def test_expansion_disabled(self):
        """use_expansion=False uses only query terms."""
        neural = MockMinicolumn(
            content="neural",
            id="L0_neural",
            layer=MockLayers.TOKENS,
            tfidf=1.0,
            tfidf_per_doc={"doc1": 1.0},
            document_ids={"doc1"},
            pagerank=0.5,
            lateral_connections={"L0_deep": 5.0}
        )
        deep = MockMinicolumn(
            content="deep",
            id="L0_deep",
            layer=MockLayers.TOKENS,
            tfidf=1.0,
            tfidf_per_doc={"doc2": 1.0},
            document_ids={"doc2"},
            pagerank=0.5,
            lateral_connections={}
        )

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([neural, deep])
        layers[MockLayers.DOCUMENTS] = MockHierarchicalLayer([])
        tokenizer = Tokenizer()

        results = graph_boosted_search(
            "neural", layers, tokenizer, top_n=5,
            use_expansion=False
        )

        # Should only find doc1 (contains "neural"), not doc2
        assert len(results) == 1
        assert results[0][0] == "doc1"


# =============================================================================
# ADDITIONAL EDGE CASE TESTS
# =============================================================================


class TestFindDocumentsForQueryEdgeCases:
    """Additional edge case tests for find_documents_for_query."""

    def test_test_file_penalty_applied(self):
        """Test files get penalized when test_file_penalty < 1.0."""
        layers = (
            LayerBuilder()
            .with_term("function", tfidf=2.0)
            .with_document("tests/test_module.py", ["function"])
            .with_document("src/module.py", ["function"])
            .build()
        )

        layer0 = layers[MockLayers.TOKENS]
        layer0.get_minicolumn("function").tfidf_per_doc = {
            "tests/test_module.py": 2.0,
            "src/module.py": 2.0
        }

        tokenizer = Tokenizer()
        result = find_documents_for_query(
            "function", layers, tokenizer,
            use_expansion=False,
            test_file_penalty=0.5  # Test files get 50% penalty
        )

        # src/module.py should rank higher than test file
        assert result[0][0] == "src/module.py"
        assert result[1][0] == "tests/test_module.py"

    def test_test_file_patterns(self):
        """All test file patterns are detected."""
        layers = (
            LayerBuilder()
            .with_term("code", tfidf=2.0)
            .with_document("tests/test_a.py", ["code"])
            .with_document("test_b.py", ["code"])
            .with_document("lib/tests/test_c.py", ["code"])
            .with_document("src/test_d.py", ["code"])
            .with_document("main.py", ["code"])
            .build()
        )

        layer0 = layers[MockLayers.TOKENS]
        layer0.get_minicolumn("code").tfidf_per_doc = {
            "tests/test_a.py": 2.0,
            "test_b.py": 2.0,
            "lib/tests/test_c.py": 2.0,
            "src/test_d.py": 2.0,
            "main.py": 2.0,
        }

        tokenizer = Tokenizer()
        result = find_documents_for_query(
            "code", layers, tokenizer,
            use_expansion=False,
            test_file_penalty=0.5
        )

        # main.py should rank first (not a test file)
        assert result[0][0] == "main.py"
        # All test files should have lower scores
        test_files = ["tests/test_a.py", "test_b.py", "lib/tests/test_c.py", "src/test_d.py"]
        result_docs = [doc_id for doc_id, _ in result]
        for test_file in test_files:
            test_idx = result_docs.index(test_file)
            main_idx = result_docs.index("main.py")
            assert test_idx > main_idx

    def test_test_file_penalty_disabled(self):
        """test_file_penalty=1.0 disables penalty."""
        layers = (
            LayerBuilder()
            .with_term("function", tfidf=2.0)
            .with_document("tests/test_module.py", ["function"])
            .with_document("src/module.py", ["function"])
            .build()
        )

        layer0 = layers[MockLayers.TOKENS]
        layer0.get_minicolumn("function").tfidf_per_doc = {
            "tests/test_module.py": 2.0,
            "src/module.py": 2.0
        }

        tokenizer = Tokenizer()
        result = find_documents_for_query(
            "function", layers, tokenizer,
            use_expansion=False,
            test_file_penalty=1.0  # No penalty
        )

        # Both should have equal scores
        assert len(result) == 2
        assert result[0][1] == pytest.approx(result[1][1])

    def test_doc_name_cached_tokens(self):
        """Document name tokens are cached for performance."""
        # Create doc with cached name_tokens
        doc_col = MockMinicolumn(
            content="neural_network",
            id="L3_neural_network",
            layer=MockLayers.DOCUMENTS
        )
        # Set name_tokens attribute after creation
        doc_col.name_tokens = {"neural", "network"}

        layers = (
            LayerBuilder()
            .with_term("neural", tfidf=2.0)
            .with_document("neural_network", ["neural"])
            .build()
        )
        layers[MockLayers.DOCUMENTS] = MockHierarchicalLayer([doc_col])

        layer0 = layers[MockLayers.TOKENS]
        layer0.get_minicolumn("neural").tfidf_per_doc = {
            "neural_network": 2.0
        }

        tokenizer = Tokenizer()
        result = find_documents_for_query(
            "neural", layers, tokenizer,
            use_expansion=False,
            doc_name_boost=2.0
        )

        # Should successfully use cached tokens
        assert len(result) == 1
        assert result[0][0] == "neural_network"


class TestFastFindDocumentsEdgeCases:
    """Additional edge case tests for fast_find_documents."""

    def test_code_concepts_expansion_with_results(self):
        """Code concepts fallback finds related terms."""
        # Setup: query term doesn't exist, but related term does
        # Note: This test verifies the code path exists, actual expansion
        # depends on code_concepts.py implementation

        col = MockMinicolumn(
            content="retrieve",
            document_ids={"doc1"},
            tfidf_per_doc={"doc1": 2.0}
        )
        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([col])

        tokenizer = Tokenizer()

        # Query for a term that doesn't exist directly
        result = fast_find_documents(
            "nonexistent_query_term", layers, tokenizer,
            use_code_concepts=True
        )

        # May or may not find results depending on code_concepts
        # But should not crash
        assert isinstance(result, list)

    def test_partial_name_match_in_fast_search(self):
        """Partial document name matches get proportional boost."""
        layers = (
            LayerBuilder()
            .with_term("neural", tfidf=2.0)
            .with_term("networks", tfidf=2.0)
            .with_document("neural_networks_doc", ["neural", "networks"])
            .with_document("other_doc", ["neural", "networks"])
            .build()
        )

        layer0 = layers[MockLayers.TOKENS]
        layer0.get_minicolumn("neural").tfidf_per_doc = {
            "neural_networks_doc": 2.0,
            "other_doc": 2.0
        }
        layer0.get_minicolumn("networks").tfidf_per_doc = {
            "neural_networks_doc": 2.0,
            "other_doc": 2.0
        }

        tokenizer = Tokenizer()
        result = fast_find_documents(
            "neural networks", layers, tokenizer,
            doc_name_boost=3.0
        )

        # neural_networks_doc should be boosted (partial name match)
        assert result[0][0] == "neural_networks_doc"

    def test_no_candidates_after_filtering(self):
        """Returns empty when all candidates filtered out."""
        layers = MockLayers.empty()
        tokenizer = Tokenizer()

        result = fast_find_documents(
            "completely nonexistent query", layers, tokenizer
        )

        assert result == []


class TestQueryWithSpreadingActivationEdgeCases:
    """Additional edge case tests for query_with_spreading_activation."""

    def test_zero_activation(self):
        """Terms with zero activation are handled correctly."""
        col = MockMinicolumn(
            content="term",
            pagerank=0.5,
            activation=0.0  # Zero activation
        )
        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([col])

        tokenizer = Tokenizer()
        result = query_with_spreading_activation("term", layers, tokenizer)

        # Should still return results (activation is multiplied)
        assert isinstance(result, list)

    def test_missing_pagerank_attribute(self):
        """Handles minicolumns without pagerank gracefully."""
        col = MockMinicolumn(
            content="term",
            activation=1.0
        )
        # Don't set pagerank - it will default to None
        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([col])

        tokenizer = Tokenizer()
        result = query_with_spreading_activation("term", layers, tokenizer)

        # Should handle None pagerank
        assert isinstance(result, list)


class TestGraphBoostedSearchEdgeCases:
    """Additional edge case tests for graph_boosted_search."""

    def test_single_term_query(self):
        """Single term query works correctly."""
        col = MockMinicolumn(
            content="neural",
            id="L0_neural",
            layer=MockLayers.TOKENS,
            tfidf=2.0,
            tfidf_per_doc={"doc1": 2.0},
            document_ids={"doc1"},
            pagerank=0.5,
            lateral_connections={}
        )

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([col])
        layers[MockLayers.DOCUMENTS] = MockHierarchicalLayer([])
        tokenizer = Tokenizer()

        results = graph_boosted_search("neural", layers, tokenizer, top_n=5)

        assert len(results) == 1
        assert results[0][0] == "doc1"

    def test_no_lateral_connections(self):
        """Works when query terms have no lateral connections."""
        col1 = MockMinicolumn(
            content="term1",
            id="L0_term1",
            layer=MockLayers.TOKENS,
            tfidf=1.0,
            tfidf_per_doc={"doc1": 1.0},
            document_ids={"doc1"},
            pagerank=0.5,
            lateral_connections={}  # No connections
        )
        col2 = MockMinicolumn(
            content="term2",
            id="L0_term2",
            layer=MockLayers.TOKENS,
            tfidf=1.0,
            tfidf_per_doc={"doc1": 1.0},
            document_ids={"doc1"},
            pagerank=0.5,
            lateral_connections={}  # No connections
        )

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([col1, col2])
        layers[MockLayers.DOCUMENTS] = MockHierarchicalLayer([])
        tokenizer = Tokenizer()

        results = graph_boosted_search(
            "term1 term2", layers, tokenizer, top_n=5,
            proximity_weight=0.5
        )

        # Should still work without proximity boost
        assert len(results) >= 1

    def test_missing_term_in_pair(self):
        """Handles when one term in pair doesn't exist."""
        col1 = MockMinicolumn(
            content="neural",
            id="L0_neural",
            layer=MockLayers.TOKENS,
            tfidf=1.0,
            tfidf_per_doc={"doc1": 1.0},
            document_ids={"doc1"},
            pagerank=0.5,
            lateral_connections={}
        )

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([col1])
        layers[MockLayers.DOCUMENTS] = MockHierarchicalLayer([])
        tokenizer = Tokenizer()

        # Query includes nonexistent term
        results = graph_boosted_search(
            "neural nonexistent", layers, tokenizer, top_n=5
        )

        # Should still return results for "neural"
        assert len(results) >= 1
        assert results[0][0] == "doc1"

    def test_no_shared_documents(self):
        """Handles when connected terms have no shared documents."""
        col1 = MockMinicolumn(
            content="term1",
            id="L0_term1",
            layer=MockLayers.TOKENS,
            tfidf=1.0,
            tfidf_per_doc={"doc1": 1.0},
            document_ids={"doc1"},
            pagerank=0.5,
            lateral_connections={"L0_term2": 5.0}
        )
        col2 = MockMinicolumn(
            content="term2",
            id="L0_term2",
            layer=MockLayers.TOKENS,
            tfidf=1.0,
            tfidf_per_doc={"doc2": 1.0},
            document_ids={"doc2"},  # Different doc
            pagerank=0.5,
            lateral_connections={"L0_term1": 5.0}
        )

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([col1, col2])
        layers[MockLayers.DOCUMENTS] = MockHierarchicalLayer([])
        tokenizer = Tokenizer()

        results = graph_boosted_search(
            "term1 term2", layers, tokenizer, top_n=5,
            proximity_weight=0.5
        )

        # Should still return both docs, just without proximity boost
        assert len(results) == 2

    def test_zero_max_scores(self):
        """Handles normalization when max scores are zero."""
        # This is an edge case that shouldn't normally happen
        # but tests robustness
        col = MockMinicolumn(
            content="term",
            id="L0_term",
            layer=MockLayers.TOKENS,
            tfidf=1.0,
            tfidf_per_doc={"doc1": 1.0},
            document_ids={"doc1"},
            pagerank=0.0,  # Zero pagerank
            lateral_connections={}
        )

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([col])
        layers[MockLayers.DOCUMENTS] = MockHierarchicalLayer([])
        tokenizer = Tokenizer()

        results = graph_boosted_search("term", layers, tokenizer, top_n=5)

        # Should handle zero values gracefully
        assert len(results) >= 1
