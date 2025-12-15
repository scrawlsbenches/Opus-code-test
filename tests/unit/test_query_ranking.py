"""
Unit Tests for Query Ranking Module
====================================

Task #175: Unit tests for cortical/query/ranking.py (25% → 90%).

Tests document type boosting, conceptual query detection, and multi-stage
ranking pipelines.

Coverage targets:
- is_conceptual_query(): Conceptual vs implementation detection
- get_doc_type_boost(): Document type boost calculation
- apply_doc_type_boost(): Boost application to results
- find_documents_with_boost(): Search with optional boosting
- find_relevant_concepts(): Stage 1 concept finding
- multi_stage_rank(): Full 4-stage pipeline with chunks
- multi_stage_rank_documents(): 2-stage document-only pipeline
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Tuple

from cortical.query.ranking import (
    is_conceptual_query,
    get_doc_type_boost,
    apply_doc_type_boost,
    find_documents_with_boost,
    find_relevant_concepts,
    multi_stage_rank,
    multi_stage_rank_documents,
)
from cortical.constants import DOC_TYPE_BOOSTS

from tests.unit.mocks import (
    MockMinicolumn,
    MockHierarchicalLayer,
    MockLayers,
    LayerBuilder,
)


# =============================================================================
# CONCEPTUAL QUERY DETECTION
# =============================================================================


class TestIsConceptualQuery:
    """Tests for is_conceptual_query() keyword detection."""

    def test_empty_query(self):
        """Empty query is not conceptual."""
        result = is_conceptual_query("")
        assert result is False

    def test_simple_implementation_query(self):
        """Implementation keywords return False."""
        assert is_conceptual_query("where is the function") is False
        assert is_conceptual_query("implement the feature") is False
        assert is_conceptual_query("fix the bug") is False

    def test_simple_conceptual_query(self):
        """Conceptual keywords return True."""
        assert is_conceptual_query("what is the algorithm") is True
        assert is_conceptual_query("explain the architecture") is True
        assert is_conceptual_query("describe the pattern") is True

    def test_what_is_prefix(self):
        """Queries starting with 'what is' get bonus points."""
        result = is_conceptual_query("what is this thing")
        assert result is True

    def test_what_are_prefix(self):
        """Queries starting with 'what are' get bonus points."""
        result = is_conceptual_query("what are these components")
        assert result is True

    def test_how_does_prefix(self):
        """Queries starting with 'how does' get bonus points."""
        result = is_conceptual_query("how does this work")
        assert result is True

    def test_explain_prefix(self):
        """Queries starting with 'explain' get bonus points."""
        result = is_conceptual_query("explain the design")
        assert result is True

    def test_mixed_keywords_conceptual_wins(self):
        """More conceptual keywords than implementation."""
        result = is_conceptual_query("what is the architecture and design pattern")
        assert result is True

    def test_mixed_keywords_implementation_wins(self):
        """More implementation keywords than conceptual."""
        result = is_conceptual_query("where is the function class method")
        assert result is False

    def test_case_insensitive(self):
        """Query detection is case insensitive."""
        assert is_conceptual_query("WHAT IS") is True
        assert is_conceptual_query("What Is") is True
        assert is_conceptual_query("WHERE IS") is False

    def test_pure_code_query(self):
        """Code-only query without keywords."""
        result = is_conceptual_query("getUserData function")
        assert result is False

    def test_documentation_keyword(self):
        """'documentation' is a conceptual keyword."""
        result = is_conceptual_query("find documentation")
        assert result is True

    def test_overview_keyword(self):
        """'overview' is a conceptual keyword."""
        result = is_conceptual_query("project overview")
        assert result is True

    def test_algorithm_keyword(self):
        """'algorithm' is a conceptual keyword."""
        result = is_conceptual_query("algorithm used here")
        assert result is True


# =============================================================================
# DOCUMENT TYPE BOOST CALCULATION
# =============================================================================


class TestGetDocTypeBoost:
    """Tests for get_doc_type_boost() boost factor calculation."""

    def test_no_metadata_code_file(self):
        """Code file without metadata gets default boost."""
        result = get_doc_type_boost("src/module.py")
        assert result == DOC_TYPE_BOOSTS['code']

    def test_no_metadata_docs_folder(self):
        """docs/ folder markdown gets docs boost."""
        result = get_doc_type_boost("docs/guide.md")
        assert result == DOC_TYPE_BOOSTS['docs']

    def test_no_metadata_root_markdown(self):
        """Root markdown gets root_docs boost."""
        result = get_doc_type_boost("README.md")
        assert result == DOC_TYPE_BOOSTS['root_docs']

    def test_no_metadata_test_file(self):
        """Test file gets test boost penalty."""
        result = get_doc_type_boost("tests/test_module.py")
        assert result == DOC_TYPE_BOOSTS['test']

    def test_with_metadata_docs_type(self):
        """Metadata doc_type overrides path inference."""
        metadata = {"doc1": {"doc_type": "docs"}}
        result = get_doc_type_boost("doc1", doc_metadata=metadata)
        assert result == DOC_TYPE_BOOSTS['docs']

    def test_with_metadata_code_type(self):
        """Metadata specifies code type."""
        metadata = {"doc1": {"doc_type": "code"}}
        result = get_doc_type_boost("doc1", doc_metadata=metadata)
        assert result == DOC_TYPE_BOOSTS['code']

    def test_with_metadata_test_type(self):
        """Metadata specifies test type."""
        metadata = {"doc1": {"doc_type": "test"}}
        result = get_doc_type_boost("doc1", doc_metadata=metadata)
        assert result == DOC_TYPE_BOOSTS['test']

    def test_custom_boosts(self):
        """Custom boost factors override defaults."""
        custom = {"docs": 2.0, "code": 1.5}
        result = get_doc_type_boost("docs/guide.md", custom_boosts=custom)
        assert result == 2.0

    def test_unknown_doc_type_in_metadata(self):
        """Unknown doc_type falls back to 1.0."""
        metadata = {"doc1": {"doc_type": "unknown"}}
        result = get_doc_type_boost("doc1", doc_metadata=metadata)
        assert result == 1.0

    def test_metadata_without_doc_type(self):
        """Metadata exists but no doc_type key defaults to 'code'."""
        metadata = {"doc1": {"author": "someone"}}
        result = get_doc_type_boost("doc1", doc_metadata=metadata)
        assert result == DOC_TYPE_BOOSTS['code']

    def test_doc_not_in_metadata(self):
        """Doc not in metadata falls back to path inference."""
        metadata = {"other_doc": {"doc_type": "docs"}}
        result = get_doc_type_boost("tests/test.py", doc_metadata=metadata)
        assert result == DOC_TYPE_BOOSTS['test']


# =============================================================================
# BOOST APPLICATION
# =============================================================================


class TestApplyDocTypeBoost:
    """Tests for apply_doc_type_boost() result re-ranking."""

    def test_empty_results(self):
        """Empty results return empty list."""
        result = apply_doc_type_boost([])
        assert result == []

    def test_boost_disabled(self):
        """boost_docs=False returns original results."""
        results = [("doc1", 10.0), ("doc2", 5.0)]
        boosted = apply_doc_type_boost(results, boost_docs=False)
        assert boosted == results

    def test_single_result(self):
        """Single result gets boosted."""
        results = [("docs/guide.md", 10.0)]
        boosted = apply_doc_type_boost(results)
        # docs/ gets 1.5x boost
        assert boosted[0][0] == "docs/guide.md"
        assert boosted[0][1] == 10.0 * DOC_TYPE_BOOSTS['docs']

    def test_boost_changes_order(self):
        """Lower-scored doc can beat higher-scored with boost."""
        results = [
            ("src/code.py", 10.0),     # code: 1.0x
            ("docs/guide.md", 7.0)      # docs: 1.5x -> 10.5
        ]
        boosted = apply_doc_type_boost(results)
        # After boost: guide.md (10.5) > code.py (10.0)
        assert boosted[0][0] == "docs/guide.md"
        assert boosted[1][0] == "src/code.py"

    def test_test_file_penalty(self):
        """Test file with penalty drops in ranking."""
        results = [
            ("src/code.py", 10.0),      # code: 1.0x
            ("tests/test.py", 10.0)     # test: 0.8x -> 8.0
        ]
        boosted = apply_doc_type_boost(results)
        assert boosted[0][0] == "src/code.py"
        assert boosted[1][0] == "tests/test.py"
        assert boosted[1][1] == 10.0 * DOC_TYPE_BOOSTS['test']

    def test_custom_boosts(self):
        """Custom boost factors applied correctly."""
        results = [("doc1", 10.0)]
        metadata = {"doc1": {"doc_type": "docs"}}
        custom = {"docs": 3.0}
        boosted = apply_doc_type_boost(
            results,
            doc_metadata=metadata,
            custom_boosts=custom
        )
        assert boosted[0][1] == 30.0

    def test_multiple_docs_same_type(self):
        """Multiple docs of same type get same boost."""
        results = [
            ("docs/guide1.md", 10.0),
            ("docs/guide2.md", 8.0)
        ]
        boosted = apply_doc_type_boost(results)
        assert boosted[0][1] == 10.0 * DOC_TYPE_BOOSTS['docs']
        assert boosted[1][1] == 8.0 * DOC_TYPE_BOOSTS['docs']

    def test_preserve_relative_order_within_type(self):
        """Relative order preserved for same doc type."""
        results = [
            ("docs/guide1.md", 10.0),
            ("docs/guide2.md", 5.0)
        ]
        boosted = apply_doc_type_boost(results)
        # guide1 should still be first
        assert boosted[0][0] == "docs/guide1.md"
        assert boosted[1][0] == "docs/guide2.md"


# =============================================================================
# FIND DOCUMENTS WITH BOOST
# =============================================================================


class TestFindDocumentsWithBoost:
    """Tests for find_documents_with_boost() search integration."""

    @patch('cortical.query.ranking.find_documents_for_query')
    def test_prefer_docs_true_always_boosts(self, mock_find):
        """prefer_docs=True always applies boosting."""
        mock_find.return_value = [("code.py", 10.0), ("guide.md", 8.0)]

        layers = MockLayers.empty()
        tokenizer = Mock()

        result = find_documents_with_boost(
            "test query",
            layers,
            tokenizer,
            top_n=5,
            prefer_docs=True,
            auto_detect_intent=False
        )

        # Should apply boost and re-rank
        mock_find.assert_called_once()
        # Result should be re-ranked by boost
        assert len(result) <= 5

    @patch('cortical.query.ranking.find_documents_for_query')
    def test_auto_detect_conceptual(self, mock_find):
        """auto_detect_intent=True detects conceptual query."""
        mock_find.return_value = [("code.py", 10.0), ("docs/guide.md", 8.0)]

        layers = MockLayers.empty()
        tokenizer = Mock()

        result = find_documents_with_boost(
            "what is the architecture",  # Conceptual query
            layers,
            tokenizer,
            top_n=5,
            auto_detect_intent=True
        )

        # Should detect conceptual and apply boost
        mock_find.assert_called_once()
        # docs/guide.md should be boosted
        assert len(result) <= 5

    @patch('cortical.query.ranking.find_documents_for_query')
    def test_auto_detect_implementation(self, mock_find):
        """auto_detect_intent=True with implementation query doesn't boost."""
        mock_find.return_value = [("code.py", 10.0), ("guide.md", 8.0)]

        layers = MockLayers.empty()
        tokenizer = Mock()

        result = find_documents_with_boost(
            "where is the function",  # Implementation query
            layers,
            tokenizer,
            top_n=5,
            auto_detect_intent=True
        )

        # Should not apply boost
        mock_find.assert_called_once()
        # Results unchanged
        assert result == [("code.py", 10.0), ("guide.md", 8.0)]

    @patch('cortical.query.ranking.find_documents_for_query')
    def test_fetches_more_candidates(self, mock_find):
        """Fetches 2x candidates for re-ranking."""
        mock_find.return_value = []

        layers = MockLayers.empty()
        tokenizer = Mock()

        find_documents_with_boost(
            "test",
            layers,
            tokenizer,
            top_n=5,
            prefer_docs=True
        )

        # Should request top_n * 2
        call_kwargs = mock_find.call_args[1]
        assert call_kwargs['top_n'] == 10

    @patch('cortical.query.ranking.find_documents_for_query')
    def test_returns_requested_top_n(self, mock_find):
        """Returns only top_n results after re-ranking."""
        mock_find.return_value = [(f"doc{i}", float(i)) for i in range(20)]

        layers = MockLayers.empty()
        tokenizer = Mock()

        result = find_documents_with_boost(
            "test",
            layers,
            tokenizer,
            top_n=3,
            prefer_docs=True
        )

        assert len(result) == 3

    @patch('cortical.query.ranking.find_documents_for_query')
    def test_passes_expansion_params(self, mock_find):
        """Query expansion parameters passed through."""
        mock_find.return_value = []

        layers = MockLayers.empty()
        tokenizer = Mock()
        semantic_rels = [("a", "SameAs", "b", 1.0)]

        find_documents_with_boost(
            "test",
            layers,
            tokenizer,
            use_expansion=False,
            semantic_relations=semantic_rels,
            use_semantic=False
        )

        call_kwargs = mock_find.call_args[1]
        assert call_kwargs['use_expansion'] is False
        assert call_kwargs['semantic_relations'] == semantic_rels
        assert call_kwargs['use_semantic'] is False


# =============================================================================
# FIND RELEVANT CONCEPTS
# =============================================================================


class TestFindRelevantConcepts:
    """Tests for find_relevant_concepts() Stage 1 concept finding."""

    def test_empty_query_terms(self):
        """Empty query terms return empty list."""
        layers = MockLayers.empty()
        result = find_relevant_concepts({}, layers)
        assert result == []

    def test_no_concepts_layer(self):
        """No concepts layer returns empty list."""
        layers = MockLayers.single_term("test")
        result = find_relevant_concepts({"test": 1.0}, layers)
        assert result == []

    def test_empty_concepts_layer(self):
        """Empty concepts layer returns empty list."""
        layers = MockLayers.empty()
        result = find_relevant_concepts({"test": 1.0}, layers)
        assert result == []

    def test_single_term_single_concept(self):
        """Single term matches single concept."""
        # Create term
        term = MockMinicolumn(
            content="neural",
            id="L0_neural",
            layer=0,
            document_ids={"doc1"}
        )

        # Create concept containing term
        concept = MockMinicolumn(
            content="ai_concept",
            id="L2_ai_concept",
            layer=2,
            pagerank=0.8,
            feedforward_sources={"L0_neural"},
            document_ids={"doc1"}
        )

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([term], level=0)
        layers[MockLayers.CONCEPTS] = MockHierarchicalLayer([concept], level=2)

        query_terms = {"neural": 1.0}
        result = find_relevant_concepts(query_terms, layers, top_n=5)

        assert len(result) == 1
        assert result[0][0] == "ai_concept"  # concept name
        assert result[0][1] > 0  # relevance score
        assert result[0][2] == {"doc1"}  # doc_ids

    def test_multiple_terms_same_concept(self):
        """Multiple query terms in same concept accumulate score."""
        # Create terms
        term1 = MockMinicolumn(content="neural", id="L0_neural", layer=0)
        term2 = MockMinicolumn(content="network", id="L0_network", layer=0)

        # Create concept containing both
        concept = MockMinicolumn(
            content="ai_concept",
            id="L2_ai_concept",
            layer=2,
            pagerank=0.8,
            feedforward_sources={"L0_neural", "L0_network"},
            document_ids={"doc1"}
        )

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([term1, term2], level=0)
        layers[MockLayers.CONCEPTS] = MockHierarchicalLayer([concept], level=2)

        query_terms = {"neural": 1.0, "network": 1.0}
        result = find_relevant_concepts(query_terms, layers, top_n=5)

        assert len(result) == 1
        # Score should be higher than single term
        assert result[0][1] > 0

    def test_term_in_multiple_concepts(self):
        """Term appearing in multiple concepts scores both."""
        term = MockMinicolumn(content="data", id="L0_data", layer=0)

        concept1 = MockMinicolumn(
            content="concept1",
            id="L2_concept1",
            layer=2,
            pagerank=0.9,
            feedforward_sources={"L0_data"},
            document_ids={"doc1"}
        )

        concept2 = MockMinicolumn(
            content="concept2",
            id="L2_concept2",
            layer=2,
            pagerank=0.7,
            feedforward_sources={"L0_data"},
            document_ids={"doc2"}
        )

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([term], level=0)
        layers[MockLayers.CONCEPTS] = MockHierarchicalLayer([concept1, concept2], level=2)

        query_terms = {"data": 1.0}
        result = find_relevant_concepts(query_terms, layers, top_n=5)

        assert len(result) == 2
        # Higher pagerank concept should score higher
        assert result[0][1] > result[1][1]

    def test_term_weight_affects_score(self):
        """Higher query term weight produces higher concept score."""
        term = MockMinicolumn(content="important", id="L0_important", layer=0)

        concept = MockMinicolumn(
            content="concept",
            id="L2_concept",
            layer=2,
            pagerank=0.8,
            feedforward_sources={"L0_important"}
        )

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([term], level=0)
        layers[MockLayers.CONCEPTS] = MockHierarchicalLayer([concept], level=2)

        # Low weight
        result_low = find_relevant_concepts({"important": 0.1}, layers, top_n=5)
        # High weight
        result_high = find_relevant_concepts({"important": 2.0}, layers, top_n=5)

        assert result_high[0][1] > result_low[0][1]

    def test_pagerank_affects_score(self):
        """Higher PageRank concept scores higher."""
        term = MockMinicolumn(content="term", id="L0_term", layer=0)

        concept_high = MockMinicolumn(
            content="important_concept",
            id="L2_important",
            layer=2,
            pagerank=0.95,
            feedforward_sources={"L0_term"}
        )

        concept_low = MockMinicolumn(
            content="minor_concept",
            id="L2_minor",
            layer=2,
            pagerank=0.05,
            feedforward_sources={"L0_term"}
        )

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([term], level=0)
        layers[MockLayers.CONCEPTS] = MockHierarchicalLayer([concept_high, concept_low], level=2)

        result = find_relevant_concepts({"term": 1.0}, layers, top_n=5)

        # important_concept should rank first
        assert result[0][0] == "important_concept"

    def test_top_n_limit(self):
        """Returns at most top_n concepts."""
        term = MockMinicolumn(content="common", id="L0_common", layer=0)

        concepts = [
            MockMinicolumn(
                content=f"concept{i}",
                id=f"L2_concept{i}",
                layer=2,
                pagerank=0.5 + i * 0.01,
                feedforward_sources={"L0_common"}
            )
            for i in range(20)
        ]

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([term], level=0)
        layers[MockLayers.CONCEPTS] = MockHierarchicalLayer(concepts, level=2)

        result = find_relevant_concepts({"common": 1.0}, layers, top_n=3)

        assert len(result) == 3

    def test_unknown_term(self):
        """Unknown term doesn't crash, returns empty."""
        layers = MockLayers.empty()
        concept = MockMinicolumn(
            content="concept",
            id="L2_concept",
            layer=2,
            feedforward_sources=set()
        )
        layers[MockLayers.CONCEPTS] = MockHierarchicalLayer([concept], level=2)

        result = find_relevant_concepts({"unknown": 1.0}, layers, top_n=5)
        assert result == []

    def test_concept_size_affects_score(self):
        """Concepts with more terms get slight boost."""
        term = MockMinicolumn(content="term", id="L0_term", layer=0)

        # Large concept (many terms)
        concept_large = MockMinicolumn(
            content="large",
            id="L2_large",
            layer=2,
            pagerank=0.8,
            feedforward_sources={f"L0_term{i}" for i in range(10)} | {"L0_term"}
        )

        # Small concept (few terms)
        concept_small = MockMinicolumn(
            content="small",
            id="L2_small",
            layer=2,
            pagerank=0.8,
            feedforward_sources={"L0_term"}
        )

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([term], level=0)
        layers[MockLayers.CONCEPTS] = MockHierarchicalLayer([concept_large, concept_small], level=2)

        result = find_relevant_concepts({"term": 1.0}, layers, top_n=5)

        # Larger concept should score slightly higher (same pagerank)
        assert result[0][0] == "large"


# =============================================================================
# MULTI-STAGE DOCUMENT RANKING
# =============================================================================


class TestMultiStageRankDocuments:
    """Tests for multi_stage_rank_documents() 2-stage pipeline."""

    @patch('cortical.query.ranking.get_expanded_query_terms')
    def test_empty_query_terms(self, mock_expand):
        """Empty query terms return empty list."""
        mock_expand.return_value = {}

        layers = MockLayers.empty()
        tokenizer = Mock()

        result = multi_stage_rank_documents("query", layers, tokenizer)
        assert result == []

    @patch('cortical.query.ranking.get_expanded_query_terms')
    def test_no_concepts_layer(self, mock_expand):
        """Works without concepts layer (TF-IDF only)."""
        mock_expand.return_value = {"term": 1.0}

        # Create term with TF-IDF
        term = MockMinicolumn(
            content="term",
            id="L0_term",
            layer=0,
            tfidf=2.5,
            document_ids={"doc1"},
            tfidf_per_doc={"doc1": 2.5}
        )

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([term], level=0)
        tokenizer = Mock()

        result = multi_stage_rank_documents("query", layers, tokenizer, top_n=5)

        assert len(result) == 1
        assert result[0][0] == "doc1"  # doc_id
        assert result[0][1] > 0  # combined score

    @patch('cortical.query.ranking.get_expanded_query_terms')
    def test_with_concepts(self, mock_expand):
        """Combines concept and TF-IDF scores."""
        mock_expand.return_value = {"term": 1.0}

        # Create term
        term = MockMinicolumn(
            content="term",
            id="L0_term",
            layer=0,
            tfidf=2.0,
            document_ids={"doc1"},
            tfidf_per_doc={"doc1": 2.0}
        )

        # Create concept containing term
        concept = MockMinicolumn(
            content="concept",
            id="L2_concept",
            layer=2,
            pagerank=0.8,
            feedforward_sources={"L0_term"},
            document_ids={"doc1"}
        )

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([term], level=0)
        layers[MockLayers.CONCEPTS] = MockHierarchicalLayer([concept], level=2)
        tokenizer = Mock()

        result = multi_stage_rank_documents("query", layers, tokenizer, top_n=5)

        assert len(result) == 1
        doc_id, score, stage_scores = result[0]
        assert doc_id == "doc1"
        assert 'concept_score' in stage_scores
        assert 'tfidf_score' in stage_scores
        assert 'combined_score' in stage_scores

    @patch('cortical.query.ranking.get_expanded_query_terms')
    def test_concept_boost_weight(self, mock_expand):
        """concept_boost parameter controls weighting."""
        mock_expand.return_value = {"term": 1.0}

        # Create two documents with different concept vs TF-IDF scores
        term = MockMinicolumn(
            content="term",
            id="L0_term",
            layer=0,
            tfidf=1.0,
            document_ids={"doc1", "doc2"},
            tfidf_per_doc={"doc1": 10.0, "doc2": 1.0}  # doc1 high TF-IDF
        )

        concept = MockMinicolumn(
            content="concept",
            id="L2_concept",
            layer=2,
            pagerank=1.0,
            feedforward_sources={"L0_term"},
            document_ids={"doc2"}  # Only doc2 in concept (high concept score)
        )

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([term], level=0)
        layers[MockLayers.CONCEPTS] = MockHierarchicalLayer([concept], level=2)
        tokenizer = Mock()

        # High concept boost should favor doc2 (in concept)
        result_high = multi_stage_rank_documents(
            "query", layers, tokenizer, concept_boost=0.9, top_n=2
        )

        # Low concept boost should favor doc1 (high TF-IDF)
        result_low = multi_stage_rank_documents(
            "query", layers, tokenizer, concept_boost=0.1, top_n=2
        )

        # Top document should differ based on weighting
        assert result_high[0][0] != result_low[0][0] or result_high[0][1] != result_low[0][1]

    @patch('cortical.query.ranking.get_expanded_query_terms')
    def test_top_n_limit(self, mock_expand):
        """Returns at most top_n documents."""
        mock_expand.return_value = {"common": 1.0}

        term = MockMinicolumn(
            content="common",
            id="L0_common",
            layer=0,
            tfidf=1.0,
            document_ids={f"doc{i}" for i in range(20)},
            tfidf_per_doc={f"doc{i}": 1.0 + i * 0.1 for i in range(20)}
        )

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([term], level=0)
        tokenizer = Mock()

        result = multi_stage_rank_documents("query", layers, tokenizer, top_n=3)

        assert len(result) == 3

    @patch('cortical.query.ranking.get_expanded_query_terms')
    def test_sorting_by_combined_score(self, mock_expand):
        """Results sorted by combined score descending."""
        mock_expand.return_value = {"term": 1.0}

        term = MockMinicolumn(
            content="term",
            id="L0_term",
            layer=0,
            tfidf=1.0,
            document_ids={"doc1", "doc2", "doc3"},
            tfidf_per_doc={"doc1": 3.0, "doc2": 1.0, "doc3": 2.0}
        )

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([term], level=0)
        tokenizer = Mock()

        result = multi_stage_rank_documents("query", layers, tokenizer, top_n=10)

        # Should be sorted by score
        assert len(result) == 3
        assert result[0][1] >= result[1][1] >= result[2][1]

    @patch('cortical.query.ranking.get_expanded_query_terms')
    def test_expansion_params_passed(self, mock_expand):
        """Query expansion parameters passed correctly."""
        mock_expand.return_value = {"term": 1.0}

        layers = MockLayers.empty()
        tokenizer = Mock()
        semantic_rels = [("a", "SameAs", "b", 1.0)]

        multi_stage_rank_documents(
            "query",
            layers,
            tokenizer,
            use_expansion=False,
            semantic_relations=semantic_rels,
            use_semantic=False
        )

        call_kwargs = mock_expand.call_args[1]
        assert call_kwargs['use_expansion'] is False
        assert call_kwargs['semantic_relations'] == semantic_rels
        assert call_kwargs['use_semantic'] is False


# =============================================================================
# MULTI-STAGE CHUNK RANKING
# =============================================================================


class TestMultiStageRank:
    """Tests for multi_stage_rank() 4-stage pipeline with chunks."""

    @patch('cortical.query.ranking.get_expanded_query_terms')
    def test_empty_query_terms(self, mock_expand):
        """Empty query terms return empty list."""
        mock_expand.return_value = {}

        layers = MockLayers.empty()
        tokenizer = Mock()
        documents = {"doc1": "Some text here"}

        result = multi_stage_rank("query", layers, tokenizer, documents)
        assert result == []

    @patch('cortical.query.passages.score_chunk')
    @patch('cortical.query.passages.create_chunks')
    @patch('cortical.query.ranking.get_expanded_query_terms')
    def test_basic_pipeline(self, mock_expand, mock_chunks, mock_score):
        """Basic 4-stage pipeline execution."""
        mock_expand.return_value = {"term": 1.0}
        mock_chunks.return_value = [("chunk text", 0, 10)]
        mock_score.return_value = 5.0

        term = MockMinicolumn(
            content="term",
            id="L0_term",
            layer=0,
            tfidf=2.0,
            document_ids={"doc1"},
            tfidf_per_doc={"doc1": 2.0}
        )

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([term], level=0)
        tokenizer = Mock()
        documents = {"doc1": "Some text with term"}

        result = multi_stage_rank("query", layers, tokenizer, documents, top_n=5)

        assert len(result) > 0
        passage_text, doc_id, start, end, final_score, stage_scores = result[0]
        assert passage_text == "chunk text"
        assert doc_id == "doc1"
        assert start == 0
        assert end == 10
        assert final_score > 0
        assert 'concept_score' in stage_scores
        assert 'doc_score' in stage_scores
        assert 'chunk_score' in stage_scores
        assert 'final_score' in stage_scores

    @patch('cortical.query.passages.score_chunk')
    @patch('cortical.query.passages.create_chunks')
    @patch('cortical.query.ranking.get_expanded_query_terms')
    def test_chunk_size_params(self, mock_expand, mock_chunks, mock_score):
        """Chunk size and overlap parameters passed correctly."""
        mock_expand.return_value = {"term": 1.0}
        mock_chunks.return_value = []

        term = MockMinicolumn(
            content="term",
            id="L0_term",
            layer=0,
            tfidf=1.0,
            document_ids={"doc1"},
            tfidf_per_doc={"doc1": 1.0}
        )

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([term], level=0)
        tokenizer = Mock()
        documents = {"doc1": "text"}

        multi_stage_rank(
            "query",
            layers,
            tokenizer,
            documents,
            chunk_size=256,
            overlap=64
        )

        # create_chunks should be called with custom params
        mock_chunks.assert_called_with("text", 256, 64)

    @patch('cortical.query.passages.score_chunk')
    @patch('cortical.query.passages.create_chunks')
    @patch('cortical.query.ranking.get_expanded_query_terms')
    def test_top_docs_filtering(self, mock_expand, mock_chunks, mock_score):
        """Only top documents are chunked and scored."""
        mock_expand.return_value = {"term": 1.0}
        mock_chunks.return_value = [("chunk", 0, 5)]
        mock_score.return_value = 1.0

        # Create term in many documents
        term = MockMinicolumn(
            content="term",
            id="L0_term",
            layer=0,
            tfidf=1.0,
            document_ids={f"doc{i}" for i in range(100)},
            tfidf_per_doc={f"doc{i}": 1.0 for i in range(100)}
        )

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([term], level=0)
        tokenizer = Mock()
        documents = {f"doc{i}": "text" for i in range(100)}

        multi_stage_rank("query", layers, tokenizer, documents, top_n=5)

        # Should only chunk top documents (top_n * 3 = 15)
        # Each call is for one document
        assert mock_chunks.call_count <= 15

    @patch('cortical.query.passages.score_chunk')
    @patch('cortical.query.passages.create_chunks')
    @patch('cortical.query.ranking.get_expanded_query_terms')
    def test_top_n_limit(self, mock_expand, mock_chunks, mock_score):
        """Returns at most top_n passages."""
        mock_expand.return_value = {"term": 1.0}

        # Create many chunks
        mock_chunks.return_value = [(f"chunk{i}", i*10, i*10+10) for i in range(50)]
        mock_score.return_value = 1.0

        term = MockMinicolumn(
            content="term",
            id="L0_term",
            layer=0,
            tfidf=1.0,
            document_ids={"doc1"},
            tfidf_per_doc={"doc1": 1.0}
        )

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([term], level=0)
        tokenizer = Mock()
        documents = {"doc1": "text " * 100}

        result = multi_stage_rank("query", layers, tokenizer, documents, top_n=3)

        assert len(result) == 3

    @patch('cortical.query.passages.score_chunk')
    @patch('cortical.query.passages.create_chunks')
    @patch('cortical.query.ranking.get_expanded_query_terms')
    def test_concept_boost_weight(self, mock_expand, mock_chunks, mock_score):
        """concept_boost parameter affects final score."""
        mock_expand.return_value = {"term": 1.0}
        mock_chunks.return_value = [("chunk", 0, 10)]
        mock_score.return_value = 5.0

        # Create two documents with different concept vs TF-IDF scores
        term = MockMinicolumn(
            content="term",
            id="L0_term",
            layer=0,
            tfidf=1.0,
            document_ids={"doc1", "doc2"},
            tfidf_per_doc={"doc1": 10.0, "doc2": 1.0}  # doc1 high TF-IDF
        )

        concept = MockMinicolumn(
            content="concept",
            id="L2_concept",
            layer=2,
            pagerank=1.0,
            feedforward_sources={"L0_term"},
            document_ids={"doc2"}  # Only doc2 in concept
        )

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([term], level=0)
        layers[MockLayers.CONCEPTS] = MockHierarchicalLayer([concept], level=2)
        tokenizer = Mock()
        documents = {"doc1": "text1", "doc2": "text2"}

        # High concept boost should favor doc2
        result_high = multi_stage_rank(
            "query", layers, tokenizer, documents, concept_boost=0.8, top_n=2
        )

        # Low concept boost should favor doc1
        result_low = multi_stage_rank(
            "query", layers, tokenizer, documents, concept_boost=0.1, top_n=2
        )

        # Scores should differ based on weighting
        assert result_high[0][1] != result_low[0][1] or result_high[0][4] != result_low[0][4]

    @patch('cortical.query.passages.score_chunk')
    @patch('cortical.query.passages.create_chunks')
    @patch('cortical.query.ranking.get_expanded_query_terms')
    def test_skips_missing_documents(self, mock_expand, mock_chunks, mock_score):
        """Skips documents not in documents dict."""
        mock_expand.return_value = {"term": 1.0}

        term = MockMinicolumn(
            content="term",
            id="L0_term",
            layer=0,
            tfidf=1.0,
            document_ids={"doc1", "doc2", "doc_missing"},
            tfidf_per_doc={"doc1": 1.0, "doc2": 1.0, "doc_missing": 1.0}
        )

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([term], level=0)
        tokenizer = Mock()
        documents = {"doc1": "text1", "doc2": "text2"}  # doc_missing not present

        mock_chunks.return_value = [("chunk", 0, 5)]
        mock_score.return_value = 1.0

        result = multi_stage_rank("query", layers, tokenizer, documents)

        # Should process doc1 and doc2, skip doc_missing
        assert all(r[1] in ["doc1", "doc2"] for r in result)

    @patch('cortical.query.passages.score_chunk')
    @patch('cortical.query.passages.create_chunks')
    @patch('cortical.query.ranking.get_expanded_query_terms')
    def test_final_score_composition(self, mock_expand, mock_chunks, mock_score):
        """Final score combines chunk, doc, and concept scores."""
        mock_expand.return_value = {"term": 1.0}
        mock_chunks.return_value = [("chunk", 0, 10)]
        mock_score.return_value = 10.0  # High chunk score

        term = MockMinicolumn(
            content="term",
            id="L0_term",
            layer=0,
            tfidf=5.0,  # High TF-IDF
            document_ids={"doc1"},
            tfidf_per_doc={"doc1": 5.0}
        )

        concept = MockMinicolumn(
            content="concept",
            id="L2_concept",
            layer=2,
            pagerank=0.9,  # High PageRank
            feedforward_sources={"L0_term"},
            document_ids={"doc1"}
        )

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([term], level=0)
        layers[MockLayers.CONCEPTS] = MockHierarchicalLayer([concept], level=2)
        tokenizer = Mock()
        documents = {"doc1": "text"}

        result = multi_stage_rank("query", layers, tokenizer, documents)

        _, _, _, _, final_score, stage_scores = result[0]

        # All scores should contribute
        assert stage_scores['chunk_score'] > 0
        assert stage_scores['doc_score'] > 0
        assert stage_scores['concept_score'] > 0
        assert final_score > 0


# =============================================================================
# ADDITIONAL EDGE CASES AND ROBUSTNESS TESTS
# =============================================================================


class TestGetDocTypeBoostEdgeCases:
    """Additional edge cases for get_doc_type_boost()."""

    def test_test_in_middle_of_filename(self):
        """Test file with 'test' in middle of name gets test boost."""
        result = get_doc_type_boost("src/my_test_utils.py")
        assert result == DOC_TYPE_BOOSTS['test']

    def test_test_in_uppercase(self):
        """Test detection is case-insensitive."""
        result = get_doc_type_boost("src/TestModule.py")
        assert result == DOC_TYPE_BOOSTS['test']

    def test_docs_folder_with_subdirectory(self):
        """docs/ folder with subdirectory gets docs boost."""
        result = get_doc_type_boost("docs/api/reference.md")
        assert result == DOC_TYPE_BOOSTS['docs']

    def test_markdown_in_subdirectory(self):
        """Markdown in non-docs subdirectory gets root_docs boost."""
        result = get_doc_type_boost("examples/tutorial.md")
        assert result == DOC_TYPE_BOOSTS['root_docs']

    def test_empty_doc_id(self):
        """Empty doc_id gets default code boost."""
        result = get_doc_type_boost("")
        assert result == DOC_TYPE_BOOSTS['code']

    def test_metadata_with_empty_dict(self):
        """Empty metadata dict falls back to path inference."""
        result = get_doc_type_boost("tests/test.py", doc_metadata={})
        assert result == DOC_TYPE_BOOSTS['test']

    def test_custom_boosts_with_missing_type(self):
        """Custom boosts without the inferred type falls back to 1.0."""
        custom = {"other_type": 2.0}  # Missing 'code' type
        result = get_doc_type_boost("src/module.py", custom_boosts=custom)
        assert result == 1.0


class TestIsConceptualQueryEdgeCases:
    """Additional edge cases for is_conceptual_query()."""

    def test_only_whitespace(self):
        """Whitespace-only query is not conceptual."""
        assert is_conceptual_query("   ") is False

    def test_multiple_prefixes(self):
        """Query with multiple conceptual prefixes gets extra boost."""
        # "what is" gives +2, "explain" gives +2, = 4 conceptual score
        result = is_conceptual_query("what is explain")
        assert result is True

    def test_prefix_in_middle(self):
        """Conceptual prefix in middle doesn't get bonus."""
        # Only counts if query starts with prefix
        result = is_conceptual_query("find what is this")
        # Should still be conceptual due to "what is" keyword
        assert result is True

    def test_tie_score(self):
        """Equal conceptual and implementation scores return False."""
        # "what" (conceptual) vs "where" (implementation) = 1-1
        result = is_conceptual_query("what where")
        assert result is False


class TestApplyDocTypeBoostEdgeCases:
    """Additional edge cases for apply_doc_type_boost()."""

    def test_zero_score_preserved(self):
        """Zero scores remain zero after boosting."""
        results = [("docs/guide.md", 0.0)]
        boosted = apply_doc_type_boost(results)
        assert boosted[0][1] == 0.0

    def test_negative_score_boosted(self):
        """Negative scores (if they occur) are boosted correctly."""
        results = [("docs/guide.md", -5.0)]
        boosted = apply_doc_type_boost(results)
        assert boosted[0][1] == -5.0 * DOC_TYPE_BOOSTS['docs']

    def test_very_large_score(self):
        """Very large scores don't cause overflow."""
        results = [("docs/guide.md", 1e10)]
        boosted = apply_doc_type_boost(results)
        assert boosted[0][1] == 1e10 * DOC_TYPE_BOOSTS['docs']


class TestFindRelevantConceptsEdgeCases:
    """Additional edge cases for find_relevant_concepts()."""

    def test_concept_with_no_documents(self):
        """Concept with no documents returns empty doc set."""
        term = MockMinicolumn(content="term", id="L0_term", layer=0)
        concept = MockMinicolumn(
            content="concept",
            id="L2_concept",
            layer=2,
            pagerank=0.8,
            feedforward_sources={"L0_term"},
            document_ids=set()  # Empty document set
        )

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([term], level=0)
        layers[MockLayers.CONCEPTS] = MockHierarchicalLayer([concept], level=2)

        result = find_relevant_concepts({"term": 1.0}, layers, top_n=5)

        assert len(result) == 1
        assert result[0][2] == set()  # Empty document set

    def test_concept_without_term_in_sources(self):
        """Concepts not containing query terms are not returned."""
        term = MockMinicolumn(content="term", id="L0_term", layer=0)
        concept = MockMinicolumn(
            content="unrelated",
            id="L2_unrelated",
            layer=2,
            pagerank=0.9,
            feedforward_sources={"L0_other"},  # Doesn't contain L0_term
            document_ids={"doc1"}
        )

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([term], level=0)
        layers[MockLayers.CONCEPTS] = MockHierarchicalLayer([concept], level=2)

        result = find_relevant_concepts({"term": 1.0}, layers, top_n=5)

        assert result == []  # No concepts match

    def test_zero_pagerank_concept(self):
        """Concept with zero PageRank gets zero score."""
        term = MockMinicolumn(content="term", id="L0_term", layer=0)
        concept = MockMinicolumn(
            content="concept",
            id="L2_concept",
            layer=2,
            pagerank=0.0,  # Zero PageRank
            feedforward_sources={"L0_term"}
        )

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([term], level=0)
        layers[MockLayers.CONCEPTS] = MockHierarchicalLayer([concept], level=2)

        result = find_relevant_concepts({"term": 1.0}, layers, top_n=5)

        assert len(result) == 1
        assert result[0][1] == 0.0  # Zero score


class TestMultiStageRankDocumentsEdgeCases:
    """Additional edge cases for multi_stage_rank_documents()."""

    @patch('cortical.query.ranking.get_expanded_query_terms')
    def test_zero_concept_boost(self, mock_expand):
        """concept_boost=0.0 uses only TF-IDF."""
        mock_expand.return_value = {"term": 1.0}

        term = MockMinicolumn(
            content="term",
            id="L0_term",
            layer=0,
            tfidf=5.0,
            document_ids={"doc1"},
            tfidf_per_doc={"doc1": 5.0}
        )

        concept = MockMinicolumn(
            content="concept",
            id="L2_concept",
            layer=2,
            pagerank=1.0,
            feedforward_sources={"L0_term"},
            document_ids={"doc1"}
        )

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([term], level=0)
        layers[MockLayers.CONCEPTS] = MockHierarchicalLayer([concept], level=2)
        tokenizer = Mock()

        result = multi_stage_rank_documents(
            "query", layers, tokenizer, concept_boost=0.0, top_n=5
        )

        assert len(result) == 1
        _, score, stage_scores = result[0]
        # With concept_boost=0.0, score should equal tfidf_score
        assert score == stage_scores['tfidf_score']

    @patch('cortical.query.ranking.get_expanded_query_terms')
    def test_full_concept_boost(self, mock_expand):
        """concept_boost=1.0 uses only concept score."""
        mock_expand.return_value = {"term": 1.0}

        term = MockMinicolumn(
            content="term",
            id="L0_term",
            layer=0,
            tfidf=5.0,
            document_ids={"doc1"},
            tfidf_per_doc={"doc1": 5.0}
        )

        concept = MockMinicolumn(
            content="concept",
            id="L2_concept",
            layer=2,
            pagerank=1.0,
            feedforward_sources={"L0_term"},
            document_ids={"doc1"}
        )

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([term], level=0)
        layers[MockLayers.CONCEPTS] = MockHierarchicalLayer([concept], level=2)
        tokenizer = Mock()

        result = multi_stage_rank_documents(
            "query", layers, tokenizer, concept_boost=1.0, top_n=5
        )

        assert len(result) == 1
        _, score, stage_scores = result[0]
        # With concept_boost=1.0, score should equal concept_score
        assert score == stage_scores['concept_score']

    @patch('cortical.query.ranking.get_expanded_query_terms')
    def test_doc_only_in_concepts_not_tfidf(self, mock_expand):
        """Document in concept but not matching TF-IDF terms."""
        mock_expand.return_value = {"term": 1.0}

        term = MockMinicolumn(
            content="term",
            id="L0_term",
            layer=0,
            tfidf=1.0,
            document_ids={"doc1"},  # Only doc1 has the term
            tfidf_per_doc={"doc1": 1.0}
        )

        concept = MockMinicolumn(
            content="concept",
            id="L2_concept",
            layer=2,
            pagerank=1.0,
            feedforward_sources={"L0_term"},
            document_ids={"doc1", "doc2"}  # doc2 in concept but not in term
        )

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([term], level=0)
        layers[MockLayers.CONCEPTS] = MockHierarchicalLayer([concept], level=2)
        tokenizer = Mock()

        result = multi_stage_rank_documents("query", layers, tokenizer, top_n=10)

        # Should return both documents
        assert len(result) == 2
        doc_ids = {r[0] for r in result}
        assert doc_ids == {"doc1", "doc2"}


class TestMultiStageRankEdgeCases:
    """Additional edge cases for multi_stage_rank()."""

    @patch('cortical.query.passages.score_chunk')
    @patch('cortical.query.passages.create_chunks')
    @patch('cortical.query.ranking.get_expanded_query_terms')
    def test_empty_documents_dict(self, mock_expand, mock_chunks, mock_score):
        """Empty documents dict returns empty results."""
        mock_expand.return_value = {"term": 1.0}

        term = MockMinicolumn(
            content="term",
            id="L0_term",
            layer=0,
            tfidf=1.0,
            document_ids={"doc1"},
            tfidf_per_doc={"doc1": 1.0}
        )

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([term], level=0)
        tokenizer = Mock()
        documents = {}  # Empty

        result = multi_stage_rank("query", layers, tokenizer, documents)

        assert result == []
        # create_chunks should never be called
        mock_chunks.assert_not_called()

    @patch('cortical.query.passages.score_chunk')
    @patch('cortical.query.passages.create_chunks')
    @patch('cortical.query.ranking.get_expanded_query_terms')
    def test_zero_chunk_score(self, mock_expand, mock_chunks, mock_score):
        """Chunk with zero score is still included."""
        mock_expand.return_value = {"term": 1.0}
        mock_chunks.return_value = [("irrelevant chunk", 0, 10)]
        mock_score.return_value = 0.0  # Zero chunk score

        term = MockMinicolumn(
            content="term",
            id="L0_term",
            layer=0,
            tfidf=1.0,
            document_ids={"doc1"},
            tfidf_per_doc={"doc1": 1.0}
        )

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([term], level=0)
        tokenizer = Mock()
        documents = {"doc1": "text"}

        result = multi_stage_rank("query", layers, tokenizer, documents)

        assert len(result) == 1
        _, _, _, _, final_score, stage_scores = result[0]
        assert stage_scores['chunk_score'] == 0.0
        # Final score should still be > 0 due to doc and concept scores
        assert final_score >= 0.0

    @patch('cortical.query.passages.score_chunk')
    @patch('cortical.query.passages.create_chunks')
    @patch('cortical.query.ranking.get_expanded_query_terms')
    def test_no_chunks_created(self, mock_expand, mock_chunks, mock_score):
        """Document with no chunks returns no results."""
        mock_expand.return_value = {"term": 1.0}
        mock_chunks.return_value = []  # No chunks

        term = MockMinicolumn(
            content="term",
            id="L0_term",
            layer=0,
            tfidf=1.0,
            document_ids={"doc1"},
            tfidf_per_doc={"doc1": 1.0}
        )

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([term], level=0)
        tokenizer = Mock()
        documents = {"doc1": ""}  # Empty document

        result = multi_stage_rank("query", layers, tokenizer, documents)

        assert result == []

    @patch('cortical.query.passages.score_chunk')
    @patch('cortical.query.passages.create_chunks')
    @patch('cortical.query.ranking.get_expanded_query_terms')
    def test_multiple_chunks_per_document(self, mock_expand, mock_chunks, mock_score):
        """Multiple chunks from same document all included."""
        mock_expand.return_value = {"term": 1.0}
        mock_chunks.return_value = [
            ("chunk1", 0, 10),
            ("chunk2", 10, 20),
            ("chunk3", 20, 30)
        ]
        mock_score.side_effect = [5.0, 3.0, 1.0]  # Different scores

        term = MockMinicolumn(
            content="term",
            id="L0_term",
            layer=0,
            tfidf=1.0,
            document_ids={"doc1"},
            tfidf_per_doc={"doc1": 1.0}
        )

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([term], level=0)
        tokenizer = Mock()
        documents = {"doc1": "text " * 50}

        result = multi_stage_rank("query", layers, tokenizer, documents, top_n=10)

        # All chunks should be included (up to top_n)
        assert len(result) == 3
        # Should be sorted by final score
        assert result[0][4] >= result[1][4] >= result[2][4]

    @patch('cortical.query.passages.score_chunk')
    @patch('cortical.query.passages.create_chunks')
    @patch('cortical.query.ranking.get_expanded_query_terms')
    def test_normalization_with_zero_max_concept_score(self, mock_expand, mock_chunks, mock_score):
        """Handles normalization when max concept score is zero."""
        mock_expand.return_value = {"term": 1.0}
        mock_chunks.return_value = [("chunk", 0, 10)]
        mock_score.return_value = 1.0

        # Create term with no concept layer
        term = MockMinicolumn(
            content="term",
            id="L0_term",
            layer=0,
            tfidf=1.0,
            document_ids={"doc1"},
            tfidf_per_doc={"doc1": 1.0}
        )

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([term], level=0)
        # No concepts layer - concept scores will be 0
        tokenizer = Mock()
        documents = {"doc1": "text"}

        result = multi_stage_rank("query", layers, tokenizer, documents)

        assert len(result) == 1
        # Should not crash with division by zero
        _, _, _, _, final_score, stage_scores = result[0]
        assert stage_scores['concept_score'] == 0.0
        assert final_score > 0  # Should still have score from chunks and TF-IDF

    @patch('cortical.query.passages.score_chunk')
    @patch('cortical.query.passages.create_chunks')
    @patch('cortical.query.ranking.get_expanded_query_terms')
    def test_expanded_term_not_in_layer(self, mock_expand, mock_chunks, mock_score):
        """Handles expanded query terms that don't exist in layer0."""
        # Expansion returns terms, but only some exist in layer0
        mock_expand.return_value = {"term": 1.0, "nonexistent": 0.5}
        mock_chunks.return_value = [("chunk", 0, 10)]
        mock_score.return_value = 1.0

        # Only create "term", not "nonexistent"
        term = MockMinicolumn(
            content="term",
            id="L0_term",
            layer=0,
            tfidf=1.0,
            document_ids={"doc1"},
            tfidf_per_doc={"doc1": 1.0}
        )

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([term], level=0)
        tokenizer = Mock()
        documents = {"doc1": "text"}

        # Should not crash when "nonexistent" term is not found
        result = multi_stage_rank("query", layers, tokenizer, documents)

        assert len(result) == 1  # Still returns results from "term"


class TestMultiStageRankDocumentsAdditional:
    """Additional edge cases for multi_stage_rank_documents()."""

    @patch('cortical.query.ranking.get_expanded_query_terms')
    def test_expanded_term_not_in_layer(self, mock_expand):
        """Handles expanded query terms that don't exist in layer0."""
        # Expansion returns terms that don't all exist
        mock_expand.return_value = {"term": 1.0, "nonexistent": 0.8}

        # Only create "term", not "nonexistent"
        term = MockMinicolumn(
            content="term",
            id="L0_term",
            layer=0,
            tfidf=2.0,
            document_ids={"doc1"},
            tfidf_per_doc={"doc1": 2.0}
        )

        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([term], level=0)
        tokenizer = Mock()

        # Should not crash when "nonexistent" term is not found
        result = multi_stage_rank_documents("query", layers, tokenizer, top_n=5)

        assert len(result) == 1  # Still returns results from "term"
        assert result[0][0] == "doc1"
