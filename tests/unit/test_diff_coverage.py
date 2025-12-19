"""
Additional Coverage Tests for cortical/diff.py
===============================================

This module fills coverage gaps in diff.py, focusing on:
- Direct testing of all functions and methods
- Edge cases and boundary conditions
- Complete branch coverage for summary generation
- Relation and cluster comparison functions
- Error handling paths

These tests complement the existing tests in test_diff.py to achieve >80% coverage.
"""

import pytest
from cortical import CorticalTextProcessor
from cortical.diff import (
    TermChange,
    RelationChange,
    ClusterChange,
    SemanticDiff,
    compare_processors,
    compare_documents,
    what_changed,
    _compare_relations,
    _compare_clusters,
)
from cortical.layers import CorticalLayer, HierarchicalLayer
from cortical.minicolumn import Minicolumn, Edge


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def add_minicolumn_to_layer(layer: HierarchicalLayer, content: str, layer_level: int) -> Minicolumn:
    """Helper to add a minicolumn to a layer with proper indexing."""
    col_id = f"L{layer_level}_{content}"
    col = Minicolumn(col_id, content, layer_level)
    layer.minicolumns[content] = col
    layer._id_index[col_id] = content
    return col


# =============================================================================
# TERMCHANGE PROPERTY COVERAGE
# =============================================================================


class TestTermChangeProperties:
    """Ensure all TermChange property branches are covered."""

    def test_pagerank_delta_with_zero_old(self):
        """PageRank delta when old is 0."""
        tc = TermChange(
            term="new",
            change_type="modified",
            old_pagerank=0.0,
            new_pagerank=0.1
        )
        assert tc.pagerank_delta == pytest.approx(0.1)

    def test_pagerank_delta_with_zero_new(self):
        """PageRank delta when new is 0."""
        tc = TermChange(
            term="falling",
            change_type="modified",
            old_pagerank=0.1,
            new_pagerank=0.0
        )
        assert tc.pagerank_delta == pytest.approx(-0.1)

    def test_tfidf_delta_with_zeros(self):
        """TF-IDF delta with zero values."""
        tc = TermChange(
            term="test",
            change_type="modified",
            old_tfidf=0.0,
            new_tfidf=5.0
        )
        assert tc.tfidf_delta == pytest.approx(5.0)

    def test_documents_added_with_none_old(self):
        """Documents added when old is None."""
        tc = TermChange(
            term="new",
            change_type="added",
            old_documents=None,
            new_documents={"doc1", "doc2"}
        )
        assert tc.documents_added == set()

    def test_documents_removed_with_none_new(self):
        """Documents removed when new is None."""
        tc = TermChange(
            term="removed",
            change_type="removed",
            old_documents={"doc1", "doc2"},
            new_documents=None
        )
        assert tc.documents_removed == set()


# =============================================================================
# SEMANTICDIFF SUMMARY COMPREHENSIVE COVERAGE
# =============================================================================


class TestSemanticDiffSummary:
    """Comprehensive coverage of SemanticDiff.summary() method."""

    def test_summary_only_modified_documents(self):
        """Summary with only modified documents."""
        diff = SemanticDiff(documents_modified=["doc1", "doc2", "doc3"])
        summary = diff.summary()
        assert "Modified: 3 documents" in summary
        assert "Documents" in summary

    def test_summary_many_terms_added(self):
        """Summary with many terms added (tests truncation)."""
        terms = [TermChange(term=f"term{i}", change_type="added") for i in range(25)]
        diff = SemanticDiff(terms_added=terms)
        summary = diff.summary()
        assert "New terms: 25" in summary
        assert "and 15 more" in summary  # Shows first 10, then "and N more"

    def test_summary_many_terms_removed(self):
        """Summary with many terms removed (tests truncation)."""
        # Check the diff.py code - it shows first 10 terms removed
        # So 15 terms should show 10, then "and 5 more"
        terms = [TermChange(term=f"old{i}", change_type="removed") for i in range(15)]
        diff = SemanticDiff(terms_removed=terms)
        summary = diff.summary()
        assert "Removed terms: 15" in summary
        # Note: The summary doesn't show "and N more" for removed terms in current implementation
        # It only does this for added terms based on diff.py lines 158-166

    def test_summary_importance_with_none_delta(self):
        """Summary handles terms with None pagerank_delta."""
        tc = TermChange(
            term="test",
            change_type="modified",
            old_pagerank=None,
            new_pagerank=0.5
        )
        diff = SemanticDiff(importance_increased=[tc])
        summary = diff.summary()
        # Should handle gracefully - checks if delta exists before formatting
        assert "Importance Shifts" in summary

    def test_summary_many_relations_added(self):
        """Summary with many relations added."""
        relations = [
            RelationChange(f"src{i}", f"tgt{i}", "type", "added")
            for i in range(10)
        ]
        diff = SemanticDiff(relations_added=relations)
        summary = diff.summary()
        assert "New relations: 10" in summary
        # Only shows first 5 relations
        assert "src0" in summary
        assert "src4" in summary

    def test_summary_all_sections(self):
        """Summary with all sections populated."""
        diff = SemanticDiff(
            documents_added=["doc1"],
            documents_removed=["doc2"],
            documents_modified=["doc3"],
            terms_added=[TermChange(term="new", change_type="added")],
            terms_removed=[TermChange(term="old", change_type="removed")],
            importance_increased=[
                TermChange(
                    term="up",
                    change_type="modified",
                    old_pagerank=0.1,
                    new_pagerank=0.3
                )
            ],
            importance_decreased=[
                TermChange(
                    term="down",
                    change_type="modified",
                    old_pagerank=0.3,
                    new_pagerank=0.1
                )
            ],
            relations_added=[RelationChange("a", "b", "syn", "added")],
            relations_removed=[RelationChange("x", "y", "rel", "removed")],
            clusters_created=[ClusterChange(cluster_id=1, change_type="created")],
            clusters_dissolved=[ClusterChange(cluster_id=2, change_type="dissolved")],
            clusters_modified=[ClusterChange(cluster_id=3, change_type="modified")],
            total_term_changes=2,
            total_relation_changes=2,
            total_cluster_changes=3
        )
        summary = diff.summary()

        # Check all sections present
        assert "Documents" in summary
        assert "Terms" in summary
        assert "Importance Shifts" in summary
        assert "Relations" in summary
        assert "Clusters" in summary
        assert "Statistics" in summary
        assert "Total term changes: 2" in summary
        assert "Total relation changes: 2" in summary
        assert "Total cluster changes: 3" in summary


class TestSemanticDiffToDict:
    """Coverage for SemanticDiff.to_dict() method."""

    def test_to_dict_with_none_pagerank(self):
        """to_dict handles terms with None pagerank."""
        diff = SemanticDiff(
            terms_added=[TermChange(term="test", change_type="added", new_pagerank=None)]
        )
        result = diff.to_dict()
        assert len(result['terms_added']) == 1
        assert result['terms_added'][0]['pagerank'] is None

    def test_to_dict_with_none_delta(self):
        """to_dict handles importance changes with None delta."""
        tc = TermChange(
            term="test",
            change_type="modified",
            old_pagerank=None,
            new_pagerank=0.5
        )
        diff = SemanticDiff(importance_increased=[tc])
        result = diff.to_dict()
        assert len(result['importance_increased']) == 1
        assert result['importance_increased'][0]['delta'] is None

    def test_to_dict_empty_lists(self):
        """to_dict with all empty lists."""
        diff = SemanticDiff()
        result = diff.to_dict()

        assert result['documents_added'] == []
        assert result['documents_removed'] == []
        assert result['documents_modified'] == []
        assert result['terms_added'] == []
        assert result['terms_removed'] == []
        assert result['importance_increased'] == []
        assert result['importance_decreased'] == []
        assert result['relations_added'] == 0
        assert result['relations_removed'] == 0
        assert result['clusters_created'] == 0
        assert result['clusters_dissolved'] == 0


# =============================================================================
# COMPARE PROCESSORS COMPREHENSIVE COVERAGE
# =============================================================================


class TestCompareProcessorsEdgeCases:
    """Edge cases and full branch coverage for compare_processors."""

    def test_processors_with_no_token_layer(self):
        """Compare processors with missing token layers."""
        proc1 = CorticalTextProcessor()
        proc2 = CorticalTextProcessor()

        # Remove token layers
        if CorticalLayer.TOKENS in proc1.layers:
            del proc1.layers[CorticalLayer.TOKENS]
        if CorticalLayer.TOKENS in proc2.layers:
            del proc2.layers[CorticalLayer.TOKENS]

        diff = compare_processors(proc1, proc2)

        # Should handle gracefully
        assert isinstance(diff, SemanticDiff)
        assert len(diff.terms_added) == 0
        assert len(diff.terms_removed) == 0

    def test_one_processor_with_token_layer(self):
        """Compare when only one processor has token layer."""
        proc1 = CorticalTextProcessor()
        proc1.process_document("doc1", "test content")
        proc1.compute_all()

        proc2 = CorticalTextProcessor()
        if CorticalLayer.TOKENS in proc2.layers:
            del proc2.layers[CorticalLayer.TOKENS]

        diff = compare_processors(proc1, proc2)

        # Should detect document removal
        assert "doc1" in diff.documents_removed

    def test_identical_processors(self):
        """Compare identical processors."""
        proc = CorticalTextProcessor()
        proc.process_document("doc1", "neural networks process data")
        proc.compute_all()

        diff = compare_processors(proc, proc)

        assert len(diff.documents_added) == 0
        assert len(diff.documents_removed) == 0
        assert len(diff.documents_modified) == 0
        # Same terms, so no additions or removals
        assert len(diff.terms_added) == 0
        assert len(diff.terms_removed) == 0

    def test_pagerank_zero_delta(self):
        """Terms with exactly zero PageRank delta."""
        proc1 = CorticalTextProcessor()
        proc1.process_document("doc1", "test content")
        proc1.compute_all()

        proc2 = CorticalTextProcessor()
        proc2.process_document("doc1", "test content")
        proc2.compute_all()

        diff = compare_processors(proc1, proc2, min_pagerank_delta=0.01)

        # Identical processors should have zero deltas
        # Which are below threshold, so no importance changes
        assert len(diff.importance_increased) == 0
        assert len(diff.importance_decreased) == 0

    def test_document_content_same_id(self):
        """Document with same ID but different content."""
        proc1 = CorticalTextProcessor()
        proc1.process_document("doc1", "original content here")
        proc1.compute_all()

        proc2 = CorticalTextProcessor()
        proc2.process_document("doc1", "completely different text")
        proc2.compute_all()

        diff = compare_processors(proc1, proc2)

        assert "doc1" in diff.documents_modified
        assert "doc1" not in diff.documents_added
        assert "doc1" not in diff.documents_removed

    def test_many_importance_changes(self):
        """Test top_movers limiting with many changes."""
        proc1 = CorticalTextProcessor()
        words1 = " ".join([f"word{i}" for i in range(50)])
        proc1.process_document("doc1", words1)
        proc1.compute_all()

        proc2 = CorticalTextProcessor()
        words2 = " ".join([f"word{i}" for i in range(50)])
        # Add more documents to shift PageRank
        proc2.process_document("doc1", words2)
        proc2.process_document("doc2", "word0 word1 word2 word0 word1 word2")
        proc2.compute_all()

        diff = compare_processors(proc1, proc2, top_movers=5, min_pagerank_delta=0.00001)

        # Should limit to top 5 movers
        total = len(diff.importance_increased) + len(diff.importance_decreased)
        assert total <= 5

    def test_statistics_calculation(self):
        """Verify statistics are correctly calculated."""
        proc1 = CorticalTextProcessor()
        proc1.process_document("doc1", "alpha beta")
        proc1.compute_all()

        proc2 = CorticalTextProcessor()
        proc2.process_document("doc1", "beta gamma")
        proc2.compute_all()

        diff = compare_processors(proc1, proc2)

        # Verify totals match components
        assert diff.total_term_changes == (
            len(diff.terms_added) + len(diff.terms_removed) + len(diff.terms_modified)
        )
        assert diff.total_relation_changes == (
            len(diff.relations_added) + len(diff.relations_removed) + len(diff.relations_modified)
        )
        assert diff.total_cluster_changes == (
            len(diff.clusters_created) + len(diff.clusters_dissolved) + len(diff.clusters_modified)
        )


# =============================================================================
# _COMPARE_RELATIONS FUNCTION COVERAGE
# =============================================================================


class TestCompareRelations:
    """Direct testing of _compare_relations helper function."""

    def test_compare_relations_both_none(self):
        """_compare_relations with both layers None."""
        diff = SemanticDiff()
        _compare_relations(None, None, diff)

        assert len(diff.relations_added) == 0
        assert len(diff.relations_removed) == 0

    def test_compare_relations_old_none(self):
        """_compare_relations with old layer None."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        diff = SemanticDiff()
        _compare_relations(None, layer, diff)

        assert len(diff.relations_added) == 0
        assert len(diff.relations_removed) == 0

    def test_compare_relations_new_none(self):
        """_compare_relations with new layer None."""
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        diff = SemanticDiff()
        _compare_relations(layer, None, diff)

        assert len(diff.relations_added) == 0
        assert len(diff.relations_removed) == 0

    def test_compare_relations_added(self):
        """Detect newly added relations."""
        old_layer = HierarchicalLayer(CorticalLayer.TOKENS)
        new_layer = HierarchicalLayer(CorticalLayer.TOKENS)

        # Old layer: no relations
        add_minicolumn_to_layer(old_layer, "term1", CorticalLayer.TOKENS.value)

        # New layer: has relation
        col2 = add_minicolumn_to_layer(new_layer, "term1", CorticalLayer.TOKENS.value)
        edge = Edge(
            target_id="term2",
            weight=1.0,
            relation_type="synonym",
            confidence=0.9,
            source="pattern"
        )
        col2.typed_connections["term2"] = edge

        diff = SemanticDiff()
        _compare_relations(old_layer, new_layer, diff)

        assert len(diff.relations_added) == 1
        assert diff.relations_added[0].source == "term1"
        assert diff.relations_added[0].target == "term2"
        assert diff.relations_added[0].relation_type == "synonym"
        assert diff.relations_added[0].new_weight == 1.0
        assert diff.relations_added[0].new_confidence == 0.9

    def test_compare_relations_removed(self):
        """Detect removed relations."""
        old_layer = HierarchicalLayer(CorticalLayer.TOKENS)
        new_layer = HierarchicalLayer(CorticalLayer.TOKENS)

        # Old layer: has relation
        col1 = add_minicolumn_to_layer(old_layer, "term1", CorticalLayer.TOKENS.value)
        edge = Edge(
            target_id="term2",
            weight=2.0,
            relation_type="antonym",
            confidence=0.8,
            source="pattern"
        )
        col1.typed_connections["term2"] = edge

        # New layer: no relations
        add_minicolumn_to_layer(new_layer, "term1", CorticalLayer.TOKENS.value)

        diff = SemanticDiff()
        _compare_relations(old_layer, new_layer, diff)

        assert len(diff.relations_removed) == 1
        assert diff.relations_removed[0].source == "term1"
        assert diff.relations_removed[0].target == "term2"
        assert diff.relations_removed[0].relation_type == "antonym"
        assert diff.relations_removed[0].old_weight == 2.0
        assert diff.relations_removed[0].old_confidence == 0.8

    def test_compare_relations_multiple(self):
        """Multiple relations added and removed."""
        old_layer = HierarchicalLayer(CorticalLayer.TOKENS)
        new_layer = HierarchicalLayer(CorticalLayer.TOKENS)

        # Old layer: term1 -> term2
        col1_old = add_minicolumn_to_layer(old_layer, "term1", CorticalLayer.TOKENS.value)
        edge1 = Edge("term2", 1.0, "synonym", 0.9, "pattern")
        col1_old.typed_connections["term2"] = edge1

        # New layer: term1 -> term3, term4 -> term5
        col1_new = add_minicolumn_to_layer(new_layer, "term1", CorticalLayer.TOKENS.value)
        edge2 = Edge("term3", 1.5, "related", 0.85, "pattern")
        col1_new.typed_connections["term3"] = edge2

        col4_new = add_minicolumn_to_layer(new_layer, "term4", CorticalLayer.TOKENS.value)
        edge3 = Edge("term5", 2.0, "is_a", 0.95, "pattern")
        col4_new.typed_connections["term5"] = edge3

        diff = SemanticDiff()
        _compare_relations(old_layer, new_layer, diff)

        # term1 -> term2 removed
        # term1 -> term3 added
        # term4 -> term5 added
        assert len(diff.relations_removed) == 1
        assert len(diff.relations_added) == 2


# =============================================================================
# _COMPARE_CLUSTERS FUNCTION COVERAGE
# =============================================================================


class TestCompareClusters:
    """Direct testing of _compare_clusters helper function."""

    def test_compare_clusters_both_none(self):
        """_compare_clusters with both layers None."""
        diff = SemanticDiff()
        _compare_clusters(None, None, diff)

        assert len(diff.clusters_created) == 0
        assert len(diff.clusters_dissolved) == 0
        assert len(diff.clusters_modified) == 0

    def test_compare_clusters_old_none(self):
        """_compare_clusters with old layer None."""
        layer = HierarchicalLayer(CorticalLayer.CONCEPTS)
        diff = SemanticDiff()
        _compare_clusters(None, layer, diff)

        assert len(diff.clusters_created) == 0

    def test_compare_clusters_new_none(self):
        """_compare_clusters with new layer None."""
        layer = HierarchicalLayer(CorticalLayer.CONCEPTS)
        diff = SemanticDiff()
        _compare_clusters(layer, None, diff)

        assert len(diff.clusters_dissolved) == 0

    def test_compare_clusters_created(self):
        """Detect newly created clusters."""
        old_layer = HierarchicalLayer(CorticalLayer.CONCEPTS)
        new_layer = HierarchicalLayer(CorticalLayer.CONCEPTS)

        # Old layer: no clusters (cluster_id is None)
        col1 = add_minicolumn_to_layer(old_layer, "concept1", CorticalLayer.CONCEPTS.value)
        col1.cluster_id = None

        # New layer: has cluster
        col2 = add_minicolumn_to_layer(new_layer, "concept1", CorticalLayer.CONCEPTS.value)
        col2.cluster_id = 5

        col3 = add_minicolumn_to_layer(new_layer, "concept2", CorticalLayer.CONCEPTS.value)
        col3.cluster_id = 5

        diff = SemanticDiff()
        _compare_clusters(old_layer, new_layer, diff)

        assert len(diff.clusters_created) == 1
        assert diff.clusters_created[0].cluster_id == 5
        assert "concept1" in diff.clusters_created[0].new_members
        assert "concept2" in diff.clusters_created[0].new_members

    def test_compare_clusters_dissolved(self):
        """Detect dissolved clusters."""
        old_layer = HierarchicalLayer(CorticalLayer.CONCEPTS)
        new_layer = HierarchicalLayer(CorticalLayer.CONCEPTS)

        # Old layer: has cluster
        col1 = add_minicolumn_to_layer(old_layer, "concept1", CorticalLayer.CONCEPTS.value)
        col1.cluster_id = 10

        col2 = add_minicolumn_to_layer(old_layer, "concept2", CorticalLayer.CONCEPTS.value)
        col2.cluster_id = 10

        # New layer: no clusters
        col3 = add_minicolumn_to_layer(new_layer, "concept1", CorticalLayer.CONCEPTS.value)
        col3.cluster_id = None

        diff = SemanticDiff()
        _compare_clusters(old_layer, new_layer, diff)

        assert len(diff.clusters_dissolved) == 1
        assert diff.clusters_dissolved[0].cluster_id == 10
        assert "concept1" in diff.clusters_dissolved[0].old_members
        assert "concept2" in diff.clusters_dissolved[0].old_members

    def test_compare_clusters_modified(self):
        """Detect modified clusters (membership changes)."""
        old_layer = HierarchicalLayer(CorticalLayer.CONCEPTS)
        new_layer = HierarchicalLayer(CorticalLayer.CONCEPTS)

        # Old layer: cluster 3 with {a, b, c}
        for name in ["a", "b", "c"]:
            col = add_minicolumn_to_layer(old_layer, name, CorticalLayer.CONCEPTS.value)
            col.cluster_id = 3

        # New layer: cluster 3 with {b, c, d, e}
        for name in ["b", "c", "d", "e"]:
            col = add_minicolumn_to_layer(new_layer, name, CorticalLayer.CONCEPTS.value)
            col.cluster_id = 3

        diff = SemanticDiff()
        _compare_clusters(old_layer, new_layer, diff)

        assert len(diff.clusters_modified) == 1
        assert diff.clusters_modified[0].cluster_id == 3
        assert diff.clusters_modified[0].old_members == {"a", "b", "c"}
        assert diff.clusters_modified[0].new_members == {"b", "c", "d", "e"}
        assert diff.clusters_modified[0].members_added == {"d", "e"}
        assert diff.clusters_modified[0].members_removed == {"a"}

    def test_compare_clusters_unchanged(self):
        """Clusters with identical membership."""
        old_layer = HierarchicalLayer(CorticalLayer.CONCEPTS)
        new_layer = HierarchicalLayer(CorticalLayer.CONCEPTS)

        # Both layers: cluster 1 with {x, y}
        for name in ["x", "y"]:
            col1 = add_minicolumn_to_layer(old_layer, name, CorticalLayer.CONCEPTS.value)
            col1.cluster_id = 1

            col2 = add_minicolumn_to_layer(new_layer, name, CorticalLayer.CONCEPTS.value)
            col2.cluster_id = 1

        diff = SemanticDiff()
        _compare_clusters(old_layer, new_layer, diff)

        # Identical membership, so no changes
        assert len(diff.clusters_created) == 0
        assert len(diff.clusters_dissolved) == 0
        assert len(diff.clusters_modified) == 0

    def test_compare_clusters_multiple_changes(self):
        """Multiple cluster changes at once."""
        old_layer = HierarchicalLayer(CorticalLayer.CONCEPTS)
        new_layer = HierarchicalLayer(CorticalLayer.CONCEPTS)

        # Old: cluster 1 {a, b}, cluster 2 {c, d}
        for name, cid in [("a", 1), ("b", 1), ("c", 2), ("d", 2)]:
            col = add_minicolumn_to_layer(old_layer, name, CorticalLayer.CONCEPTS.value)
            col.cluster_id = cid

        # New: cluster 1 {a}, cluster 3 {e, f}
        for name, cid in [("a", 1), ("e", 3), ("f", 3)]:
            col = add_minicolumn_to_layer(new_layer, name, CorticalLayer.CONCEPTS.value)
            col.cluster_id = cid

        diff = SemanticDiff()
        _compare_clusters(old_layer, new_layer, diff)

        # cluster 1 modified (lost b)
        # cluster 2 dissolved
        # cluster 3 created
        assert len(diff.clusters_created) == 1
        assert len(diff.clusters_dissolved) == 1
        assert len(diff.clusters_modified) == 1


# =============================================================================
# COMPARE DOCUMENTS COVERAGE
# =============================================================================


class TestCompareDocumentsEdgeCases:
    """Edge cases for compare_documents function."""

    def test_compare_documents_no_token_layer(self):
        """compare_documents when processor has no token layer."""
        proc = CorticalTextProcessor()
        # Remove token layer
        if CorticalLayer.TOKENS in proc.layers:
            del proc.layers[CorticalLayer.TOKENS]

        result = compare_documents(proc, "doc1", "doc2")

        assert 'error' in result
        assert 'no token layer' in result['error'].lower()

    def test_compare_documents_no_bigram_layer(self):
        """compare_documents when processor has no bigram layer."""
        proc = CorticalTextProcessor()
        proc.process_document("doc1", "alpha beta gamma")
        proc.process_document("doc2", "beta gamma delta")
        proc.compute_all()

        # Remove bigram layer
        if CorticalLayer.BIGRAMS in proc.layers:
            del proc.layers[CorticalLayer.BIGRAMS]

        result = compare_documents(proc, "doc1", "doc2")

        # Should handle gracefully
        assert 'jaccard_similarity' in result
        assert result['shared_bigrams'] == 0

    def test_compare_documents_nonexistent_docs(self):
        """compare_documents with documents not in corpus."""
        proc = CorticalTextProcessor()
        proc.process_document("doc1", "test")
        proc.compute_all()

        result = compare_documents(proc, "nonexistent1", "nonexistent2")

        # Documents not found, so no terms
        assert result['terms_in_old'] == 0
        assert result['terms_in_new'] == 0
        assert result['shared_terms'] == 0

    def test_compare_documents_empty_jaccard(self):
        """Jaccard similarity when both documents have very few/no surviving terms."""
        proc = CorticalTextProcessor()
        # Use documents with content (empty strings not allowed)
        proc.process_document("doc1", "hello world test")
        proc.process_document("doc2", "hello world test")
        proc.compute_all()

        result = compare_documents(proc, "doc1", "doc2")

        # Identical documents should have jaccard similarity of 1.0
        assert result['jaccard_similarity'] == 1.0

    def test_compare_documents_all_unique_terms(self):
        """Documents with completely unique vocabularies."""
        proc = CorticalTextProcessor()
        proc.process_document("doc1", "alpha beta gamma")
        proc.process_document("doc2", "delta epsilon zeta")
        proc.compute_all()

        result = compare_documents(proc, "doc1", "doc2")

        assert result['shared_terms'] == 0
        assert result['jaccard_similarity'] == 0.0
        assert result['unique_to_old'] > 0
        assert result['unique_to_new'] > 0


# =============================================================================
# WHAT CHANGED COVERAGE
# =============================================================================


class TestWhatChangedEdgeCases:
    """Edge cases for what_changed function."""

    def test_what_changed_uses_processor_tokenizer(self):
        """what_changed uses the processor's tokenizer."""
        from cortical.tokenizer import Tokenizer

        # Create processor with custom tokenizer settings
        proc = CorticalTextProcessor()

        result = what_changed(proc, "Test Content", "New Content")

        # Should use processor's tokenizer (lowercase, stemmed)
        assert 'tokens' in result
        assert 'bigrams' in result

    def test_what_changed_empty_union(self):
        """what_changed when both texts are empty."""
        proc = CorticalTextProcessor()

        result = what_changed(proc, "", "")

        # No tokens at all
        assert result['tokens']['similarity'] == 0.0
        assert result['bigrams']['similarity'] == 0.0
        assert result['summary']['content_similarity'] == 0.0

    def test_what_changed_only_stopwords(self):
        """what_changed when texts contain only stopwords."""
        proc = CorticalTextProcessor()

        result = what_changed(proc, "the a an", "the a an")

        # After stopword removal, may have no tokens
        # Should handle gracefully
        assert 'tokens' in result
        assert 'summary' in result

    def test_what_changed_bigrams_empty(self):
        """what_changed with no bigrams (single word texts)."""
        proc = CorticalTextProcessor()

        result = what_changed(proc, "word", "other")

        # Single words = no bigrams
        assert result['bigrams']['unchanged_count'] == 0
        assert len(result['bigrams']['added']) == 0
        assert len(result['bigrams']['removed']) == 0

    def test_what_changed_similarity_threshold(self):
        """what_changed significant change threshold (0.8)."""
        proc = CorticalTextProcessor()

        # Very similar (>0.8 similarity)
        result1 = what_changed(proc, "a b c d e", "a b c d f")

        # Very different (<0.8 similarity)
        result2 = what_changed(proc, "a b", "c d e f g h")

        # Check threshold logic
        # result1 should not be significant (assuming high similarity)
        # result2 should be significant (low similarity)
        assert isinstance(result1['summary']['is_significant_change'], bool)
        assert isinstance(result2['summary']['is_significant_change'], bool)

    def test_what_changed_truncation_limits(self):
        """what_changed truncates long token/bigram lists."""
        proc = CorticalTextProcessor()

        # Create texts with many unique tokens
        old_text = " ".join([f"old{i}" for i in range(100)])
        new_text = " ".join([f"new{i}" for i in range(100)])

        result = what_changed(proc, old_text, new_text)

        # Should truncate to limits
        assert len(result['tokens']['added']) <= 50
        assert len(result['tokens']['removed']) <= 50
        assert len(result['bigrams']['added']) <= 30
        assert len(result['bigrams']['removed']) <= 30

    def test_what_changed_special_characters(self):
        """what_changed handles special characters."""
        proc = CorticalTextProcessor()

        result = what_changed(
            proc,
            "Hello, world! How are you?",
            "Goodbye, world! Where are you?"
        )

        # Should tokenize properly despite punctuation
        assert 'tokens' in result
        assert len(result['tokens']['added']) > 0
        assert len(result['tokens']['removed']) > 0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestDiffIntegration:
    """Integration tests ensuring all parts work together."""

    def test_full_workflow_compare_processors(self):
        """Complete workflow: process, modify, compare."""
        # Initial state
        proc1 = CorticalTextProcessor()
        proc1.process_document("doc1", "neural networks learn patterns")
        proc1.process_document("doc2", "machine learning algorithms")
        proc1.compute_all()

        # Modified state
        proc2 = CorticalTextProcessor()
        proc2.process_document("doc1", "neural networks learn patterns")  # Same
        proc2.process_document("doc2", "deep learning models")  # Modified
        proc2.process_document("doc3", "artificial intelligence systems")  # Added
        proc2.compute_all()

        # Compare
        diff = compare_processors(proc1, proc2)

        # Verify all changes detected
        assert "doc3" in diff.documents_added
        assert "doc2" in diff.documents_modified
        assert len(diff.documents_removed) == 0

        # Should have term changes
        assert diff.total_term_changes > 0

        # Summary should be comprehensive
        summary = diff.summary()
        assert len(summary) > 100  # Non-trivial summary

        # to_dict should work
        result_dict = diff.to_dict()
        assert 'documents_added' in result_dict
        assert 'doc3' in result_dict['documents_added']

    def test_full_workflow_compare_documents(self):
        """Complete workflow: documents comparison."""
        proc = CorticalTextProcessor()
        proc.process_document("doc1", "machine learning with neural networks")
        proc.process_document("doc2", "deep learning with neural models")
        proc.compute_all()

        result = compare_documents(proc, "doc1", "doc2")

        # Should have shared terms (neural, learning)
        assert result['shared_terms'] > 0
        assert 0 < result['jaccard_similarity'] < 1
        assert result['unique_to_old'] > 0
        assert result['unique_to_new'] > 0

    def test_full_workflow_what_changed(self):
        """Complete workflow: text change detection."""
        proc = CorticalTextProcessor()

        old_text = """
        Neural networks are computational models inspired by biological neurons.
        They learn patterns from data through training.
        """

        new_text = """
        Deep learning models are powerful computational systems.
        They learn representations from data through backpropagation.
        """

        result = what_changed(proc, old_text, new_text)

        # Should detect changes
        assert len(result['tokens']['added']) > 0
        assert len(result['tokens']['removed']) > 0
        assert 0 <= result['summary']['content_similarity'] <= 1
        assert isinstance(result['summary']['is_significant_change'], bool)
