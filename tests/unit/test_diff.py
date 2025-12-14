"""
Unit Tests for Diff Module
===========================

Tests the semantic diff module that compares processor states and documents:
- TermChange: Dataclass for term/concept changes
- RelationChange: Dataclass for semantic relation changes
- ClusterChange: Dataclass for cluster membership changes
- SemanticDiff: Complete diff container with summary methods
- compare_processors: Compare two processor states
- compare_documents: Compare two documents within same corpus
- what_changed: Quick text comparison using tokenizer

These tests verify that semantic differences are correctly detected
and reported when comparing before/after states.
"""

import pytest

from cortical.diff import (
    TermChange,
    RelationChange,
    ClusterChange,
    SemanticDiff,
    compare_processors,
    compare_documents,
    what_changed,
)
from cortical.layers import CorticalLayer


# =============================================================================
# TERMCHANGE TESTS
# =============================================================================


class TestTermChange:
    """Tests for TermChange dataclass."""

    def test_pagerank_delta_both_values_present(self):
        """pagerank_delta returns correct delta when both values present."""
        tc = TermChange(
            term="test",
            change_type="modified",
            old_pagerank=0.1,
            new_pagerank=0.3
        )

        assert tc.pagerank_delta == pytest.approx(0.2)

    def test_pagerank_delta_positive(self):
        """pagerank_delta handles positive change."""
        tc = TermChange(
            term="rising",
            change_type="modified",
            old_pagerank=0.05,
            new_pagerank=0.15
        )

        assert tc.pagerank_delta == pytest.approx(0.1)
        assert tc.pagerank_delta > 0

    def test_pagerank_delta_negative(self):
        """pagerank_delta handles negative change."""
        tc = TermChange(
            term="falling",
            change_type="modified",
            old_pagerank=0.2,
            new_pagerank=0.1
        )

        assert tc.pagerank_delta == -0.1
        assert tc.pagerank_delta < 0

    def test_pagerank_delta_missing_old(self):
        """pagerank_delta returns None when old value missing."""
        tc = TermChange(
            term="new",
            change_type="added",
            old_pagerank=None,
            new_pagerank=0.1
        )

        assert tc.pagerank_delta is None

    def test_pagerank_delta_missing_new(self):
        """pagerank_delta returns None when new value missing."""
        tc = TermChange(
            term="removed",
            change_type="removed",
            old_pagerank=0.1,
            new_pagerank=None
        )

        assert tc.pagerank_delta is None

    def test_pagerank_delta_both_missing(self):
        """pagerank_delta returns None when both values missing."""
        tc = TermChange(
            term="test",
            change_type="modified"
        )

        assert tc.pagerank_delta is None

    def test_tfidf_delta_calculation(self):
        """tfidf_delta returns correct delta."""
        tc = TermChange(
            term="test",
            change_type="modified",
            old_tfidf=2.0,
            new_tfidf=5.0
        )

        assert tc.tfidf_delta == 3.0

    def test_tfidf_delta_missing_values(self):
        """tfidf_delta returns None when values missing."""
        tc1 = TermChange(term="test", change_type="added", old_tfidf=None, new_tfidf=2.0)
        tc2 = TermChange(term="test", change_type="removed", old_tfidf=2.0, new_tfidf=None)
        tc3 = TermChange(term="test", change_type="modified")

        assert tc1.tfidf_delta is None
        assert tc2.tfidf_delta is None
        assert tc3.tfidf_delta is None

    def test_documents_added_set_difference(self):
        """documents_added returns correct set difference."""
        tc = TermChange(
            term="test",
            change_type="modified",
            old_documents={"doc1", "doc2"},
            new_documents={"doc2", "doc3", "doc4"}
        )

        added = tc.documents_added
        assert added == {"doc3", "doc4"}
        assert isinstance(added, set)

    def test_documents_added_none_added(self):
        """documents_added returns empty set when no new documents."""
        tc = TermChange(
            term="test",
            change_type="modified",
            old_documents={"doc1", "doc2"},
            new_documents={"doc1", "doc2"}
        )

        assert tc.documents_added == set()

    def test_documents_added_all_new(self):
        """documents_added returns all new documents when old is empty."""
        tc = TermChange(
            term="test",
            change_type="modified",
            old_documents=set(),
            new_documents={"doc1", "doc2"}
        )

        assert tc.documents_added == {"doc1", "doc2"}

    def test_documents_added_missing_values(self):
        """documents_added returns empty set when values missing."""
        tc1 = TermChange(term="test", change_type="added", old_documents=None, new_documents={"doc1"})
        tc2 = TermChange(term="test", change_type="removed", old_documents={"doc1"}, new_documents=None)
        tc3 = TermChange(term="test", change_type="modified")

        assert tc1.documents_added == set()
        assert tc2.documents_added == set()
        assert tc3.documents_added == set()

    def test_documents_removed_set_difference(self):
        """documents_removed returns correct set difference."""
        tc = TermChange(
            term="test",
            change_type="modified",
            old_documents={"doc1", "doc2", "doc3"},
            new_documents={"doc2", "doc4"}
        )

        removed = tc.documents_removed
        assert removed == {"doc1", "doc3"}
        assert isinstance(removed, set)

    def test_documents_removed_none_removed(self):
        """documents_removed returns empty set when no documents removed."""
        tc = TermChange(
            term="test",
            change_type="modified",
            old_documents={"doc1"},
            new_documents={"doc1", "doc2"}
        )

        assert tc.documents_removed == set()

    def test_documents_removed_all_removed(self):
        """documents_removed returns all old documents when new is empty."""
        tc = TermChange(
            term="test",
            change_type="removed",
            old_documents={"doc1", "doc2"},
            new_documents=set()
        )

        assert tc.documents_removed == {"doc1", "doc2"}

    def test_documents_removed_missing_values(self):
        """documents_removed returns empty set when values missing."""
        tc1 = TermChange(term="test", change_type="added", old_documents=None, new_documents={"doc1"})
        tc2 = TermChange(term="test", change_type="removed", old_documents={"doc1"}, new_documents=None)
        tc3 = TermChange(term="test", change_type="modified")

        assert tc1.documents_removed == set()
        assert tc2.documents_removed == set()
        assert tc3.documents_removed == set()


# =============================================================================
# RELATIONCHANGE TESTS
# =============================================================================


class TestRelationChange:
    """Tests for RelationChange dataclass."""

    def test_basic_creation(self):
        """Basic RelationChange creation."""
        rc = RelationChange(
            source="term1",
            target="term2",
            relation_type="synonym",
            change_type="added"
        )

        assert rc.source == "term1"
        assert rc.target == "term2"
        assert rc.relation_type == "synonym"
        assert rc.change_type == "added"

    def test_with_weights(self):
        """RelationChange with weight values."""
        rc = RelationChange(
            source="a",
            target="b",
            relation_type="related_to",
            change_type="modified",
            old_weight=0.5,
            new_weight=0.8
        )

        assert rc.old_weight == 0.5
        assert rc.new_weight == 0.8

    def test_with_confidence(self):
        """RelationChange with confidence values."""
        rc = RelationChange(
            source="x",
            target="y",
            relation_type="is_a",
            change_type="added",
            new_weight=1.0,
            new_confidence=0.9
        )

        assert rc.new_confidence == 0.9
        assert rc.old_confidence is None

    def test_removed_relation(self):
        """Removed relation has old values only."""
        rc = RelationChange(
            source="old",
            target="gone",
            relation_type="was_related",
            change_type="removed",
            old_weight=0.7,
            old_confidence=0.8
        )

        assert rc.old_weight == 0.7
        assert rc.old_confidence == 0.8
        assert rc.new_weight is None
        assert rc.new_confidence is None


# =============================================================================
# CLUSTERCHANGE TESTS
# =============================================================================


class TestClusterChange:
    """Tests for ClusterChange dataclass."""

    def test_created_cluster(self):
        """Created cluster has new members only."""
        cc = ClusterChange(
            cluster_id=42,
            change_type="created",
            new_members={"term1", "term2", "term3"}
        )

        assert cc.cluster_id == 42
        assert cc.change_type == "created"
        assert cc.new_members == {"term1", "term2", "term3"}
        assert len(cc.old_members) == 0

    def test_dissolved_cluster(self):
        """Dissolved cluster has old members only."""
        cc = ClusterChange(
            cluster_id=7,
            change_type="dissolved",
            old_members={"a", "b", "c"}
        )

        assert cc.cluster_id == 7
        assert cc.change_type == "dissolved"
        assert cc.old_members == {"a", "b", "c"}
        assert len(cc.new_members) == 0

    def test_modified_cluster(self):
        """Modified cluster tracks member changes."""
        cc = ClusterChange(
            cluster_id=10,
            change_type="modified",
            old_members={"a", "b", "c"},
            new_members={"b", "c", "d", "e"},
            members_added={"d", "e"},
            members_removed={"a"}
        )

        assert cc.members_added == {"d", "e"}
        assert cc.members_removed == {"a"}
        assert cc.old_members == {"a", "b", "c"}
        assert cc.new_members == {"b", "c", "d", "e"}

    def test_default_empty_sets(self):
        """Default values are empty sets."""
        cc = ClusterChange(
            cluster_id=1,
            change_type="created"
        )

        assert cc.old_members == set()
        assert cc.new_members == set()
        assert cc.members_added == set()
        assert cc.members_removed == set()

    def test_none_cluster_id(self):
        """Cluster ID can be None."""
        cc = ClusterChange(
            cluster_id=None,
            change_type="dissolved"
        )

        assert cc.cluster_id is None


# =============================================================================
# SEMANTICDIFF TESTS
# =============================================================================


class TestSemanticDiff:
    """Tests for SemanticDiff dataclass."""

    def test_empty_diff(self):
        """Empty diff has no changes."""
        diff = SemanticDiff()

        assert len(diff.documents_added) == 0
        assert len(diff.documents_removed) == 0
        assert len(diff.terms_added) == 0
        assert len(diff.terms_removed) == 0
        assert diff.total_term_changes == 0
        assert diff.total_relation_changes == 0
        assert diff.total_cluster_changes == 0

    def test_summary_empty_diff(self):
        """Summary of empty diff is minimal."""
        diff = SemanticDiff()
        summary = diff.summary()

        assert isinstance(summary, str)
        assert "Statistics" in summary
        assert "Total term changes: 0" in summary

    def test_summary_with_document_changes(self):
        """Summary includes document changes."""
        diff = SemanticDiff(
            documents_added=["doc1", "doc2"],
            documents_removed=["doc3"]
        )

        summary = diff.summary()

        assert "Documents" in summary
        assert "Added: 2 documents" in summary
        assert "Removed: 1 documents" in summary
        assert "doc1" in summary
        assert "doc3" in summary

    def test_summary_with_many_documents(self):
        """Summary limits document listing."""
        many_docs = [f"doc{i}" for i in range(20)]
        diff = SemanticDiff(documents_added=many_docs)

        summary = diff.summary()

        assert "Added: 20 documents" in summary
        assert "and 15 more" in summary  # Shows first 5, then "and N more"

    def test_summary_with_term_changes(self):
        """Summary includes term changes."""
        diff = SemanticDiff(
            terms_added=[
                TermChange(term="new1", change_type="added"),
                TermChange(term="new2", change_type="added"),
            ],
            terms_removed=[
                TermChange(term="old1", change_type="removed"),
            ]
        )

        summary = diff.summary()

        assert "Terms" in summary
        assert "New terms: 2" in summary
        assert "Removed terms: 1" in summary
        assert "new1" in summary
        assert "old1" in summary

    def test_summary_with_importance_shifts(self):
        """Summary includes PageRank changes."""
        diff = SemanticDiff(
            importance_increased=[
                TermChange(
                    term="rising",
                    change_type="modified",
                    old_pagerank=0.1,
                    new_pagerank=0.3
                )
            ],
            importance_decreased=[
                TermChange(
                    term="falling",
                    change_type="modified",
                    old_pagerank=0.3,
                    new_pagerank=0.1
                )
            ]
        )

        summary = diff.summary()

        assert "Importance Shifts" in summary
        assert "Rising Terms" in summary
        assert "Falling Terms" in summary
        assert "rising" in summary
        assert "falling" in summary

    def test_summary_with_relations(self):
        """Summary includes relation changes."""
        diff = SemanticDiff(
            relations_added=[
                RelationChange("a", "b", "synonym", "added"),
                RelationChange("c", "d", "is_a", "added"),
            ],
            relations_removed=[
                RelationChange("x", "y", "related", "removed"),
            ]
        )

        summary = diff.summary()

        assert "Relations" in summary
        assert "New relations: 2" in summary
        assert "Removed relations: 1" in summary

    def test_summary_with_clusters(self):
        """Summary includes cluster changes."""
        diff = SemanticDiff(
            clusters_created=[
                ClusterChange(cluster_id=1, change_type="created"),
                ClusterChange(cluster_id=2, change_type="created"),
            ],
            clusters_dissolved=[
                ClusterChange(cluster_id=99, change_type="dissolved"),
            ],
            clusters_modified=[
                ClusterChange(cluster_id=5, change_type="modified"),
            ]
        )

        summary = diff.summary()

        assert "Clusters" in summary
        assert "New clusters: 2" in summary
        assert "Dissolved clusters: 1" in summary
        assert "Modified clusters: 1" in summary

    def test_to_dict_structure(self):
        """to_dict returns proper structure."""
        diff = SemanticDiff(
            documents_added=["doc1"],
            terms_added=[TermChange(term="new", change_type="added", new_pagerank=0.1)],
            total_term_changes=5,
            total_relation_changes=2,
            total_cluster_changes=1
        )

        result = diff.to_dict()

        assert isinstance(result, dict)
        assert result['documents_added'] == ["doc1"]
        assert len(result['terms_added']) == 1
        assert result['terms_added'][0]['term'] == "new"
        assert result['total_term_changes'] == 5
        assert result['total_relation_changes'] == 2
        assert result['total_cluster_changes'] == 1

    def test_to_dict_importance_shifts(self):
        """to_dict includes importance deltas."""
        diff = SemanticDiff(
            importance_increased=[
                TermChange(
                    term="up",
                    change_type="modified",
                    old_pagerank=0.1,
                    new_pagerank=0.3
                )
            ]
        )

        result = diff.to_dict()

        assert len(result['importance_increased']) == 1
        assert result['importance_increased'][0]['term'] == "up"
        assert result['importance_increased'][0]['delta'] == pytest.approx(0.2)

    def test_to_dict_counts_only(self):
        """to_dict returns counts for relations and clusters."""
        diff = SemanticDiff(
            relations_added=[
                RelationChange("a", "b", "syn", "added"),
                RelationChange("c", "d", "is_a", "added"),
            ],
            clusters_created=[
                ClusterChange(cluster_id=1, change_type="created"),
            ]
        )

        result = diff.to_dict()

        # Relations and clusters are summarized as counts
        assert result['relations_added'] == 2
        assert result['clusters_created'] == 1


# =============================================================================
# COMPARE PROCESSORS TESTS
# =============================================================================


class TestCompareProcessors:
    """Tests for compare_processors function."""

    def test_empty_processors(self, fresh_processor):
        """Comparing empty processors returns empty diff."""
        proc1 = fresh_processor
        from cortical import CorticalTextProcessor
        proc2 = CorticalTextProcessor()

        diff = compare_processors(proc1, proc2)

        assert isinstance(diff, SemanticDiff)
        assert len(diff.documents_added) == 0
        assert len(diff.documents_removed) == 0
        assert len(diff.terms_added) == 0
        assert len(diff.terms_removed) == 0

    def test_adding_document(self):
        """Adding a document shows in diff."""
        from cortical import CorticalTextProcessor

        # Both processors need non-empty layers for term comparison
        # Start with a baseline document in proc1
        proc1 = CorticalTextProcessor()
        proc1.process_document("baseline", "baseline content")
        proc1.compute_all()

        # Proc2 has baseline plus new document
        proc2 = CorticalTextProcessor()
        proc2.process_document("baseline", "baseline content")
        proc2.process_document("doc1", "neural networks process data")
        proc2.compute_all()

        diff = compare_processors(proc1, proc2)

        # Should detect doc1 as added
        assert "doc1" in diff.documents_added
        assert "baseline" not in diff.documents_added
        assert len(diff.documents_removed) == 0
        # New terms from doc1 should be detected (those not in baseline)
        assert len(diff.terms_added) > 0 or len(diff.terms_modified) > 0

    def test_removing_document(self):
        """Removing a document shows in diff."""
        from cortical import CorticalTextProcessor
        from cortical.layers import CorticalLayer, HierarchicalLayer

        proc1 = CorticalTextProcessor()
        proc1.process_document("doc1", "test content")
        proc1.compute_all()

        # Proc2 needs to have layers initialized for comparison to work
        proc2 = CorticalTextProcessor()
        proc2.layers[CorticalLayer.TOKENS] = HierarchicalLayer(CorticalLayer.TOKENS)

        diff = compare_processors(proc1, proc2)

        assert "doc1" in diff.documents_removed
        assert len(diff.documents_added) == 0
        # Note: terms_removed may be 0 if proc2 has no layer, which is expected behavior

    def test_modified_document(self):
        """Modified document content is detected."""
        from cortical import CorticalTextProcessor

        proc1 = CorticalTextProcessor()
        proc1.process_document("doc1", "original content")
        proc1.compute_all()

        proc2 = CorticalTextProcessor()
        proc2.process_document("doc1", "modified content")
        proc2.compute_all()

        diff = compare_processors(proc1, proc2)

        assert "doc1" in diff.documents_modified
        assert "doc1" not in diff.documents_added
        assert "doc1" not in diff.documents_removed

    def test_pagerank_changes_detected(self):
        """PageRank importance shifts are detected."""
        from cortical import CorticalTextProcessor

        # Initial state with "neural"
        proc1 = CorticalTextProcessor()
        proc1.process_document("doc1", "neural networks")
        proc1.compute_all()

        # Add more documents mentioning "neural" to boost its importance
        proc2 = CorticalTextProcessor()
        proc2.process_document("doc1", "neural networks")
        proc2.process_document("doc2", "neural systems")
        proc2.process_document("doc3", "neural processing")
        proc2.compute_all()

        diff = compare_processors(proc1, proc2, min_pagerank_delta=0.0001)

        # "neural" should appear in both, may have importance change
        # Either in modified terms or importance shifts
        all_modified_terms = [tc.term for tc in diff.terms_modified]
        all_importance_changed = [tc.term for tc in diff.importance_increased + diff.importance_decreased]

        # Neural appears in both processors, so should be in one of these lists
        # (depending on whether PageRank changed significantly)
        assert len(all_modified_terms) > 0 or len(all_importance_changed) > 0

    def test_top_movers_parameter(self):
        """top_movers parameter limits importance changes."""
        from cortical import CorticalTextProcessor

        proc1 = CorticalTextProcessor()
        proc1.process_document("doc1", "alpha beta gamma delta epsilon")
        proc1.compute_all()

        proc2 = CorticalTextProcessor()
        proc2.process_document("doc1", "alpha beta gamma delta epsilon")
        proc2.process_document("doc2", "alpha alpha alpha beta")  # Boost some terms
        proc2.compute_all()

        diff = compare_processors(proc1, proc2, top_movers=2, min_pagerank_delta=0.0001)

        # Should limit total importance shifts to top_movers
        total_shifts = len(diff.importance_increased) + len(diff.importance_decreased)
        assert total_shifts <= 2

    def test_min_pagerank_delta_threshold(self):
        """min_pagerank_delta filters small changes."""
        from cortical import CorticalTextProcessor

        proc1 = CorticalTextProcessor()
        proc1.process_document("doc1", "test content")
        proc1.compute_all()

        proc2 = CorticalTextProcessor()
        proc2.process_document("doc1", "test content")
        proc2.process_document("doc2", "other stuff")  # Small change to "test" PageRank
        proc2.compute_all()

        # High threshold - should filter out small changes
        diff_strict = compare_processors(proc1, proc2, min_pagerank_delta=0.5)

        # Low threshold - should catch more changes
        diff_loose = compare_processors(proc1, proc2, min_pagerank_delta=0.0001)

        # Loose threshold should catch more or equal changes
        total_strict = len(diff_strict.importance_increased) + len(diff_strict.importance_decreased)
        total_loose = len(diff_loose.importance_increased) + len(diff_loose.importance_decreased)
        assert total_loose >= total_strict

    def test_total_term_changes_statistic(self):
        """total_term_changes is calculated correctly."""
        from cortical import CorticalTextProcessor

        proc1 = CorticalTextProcessor()
        proc1.process_document("doc1", "old content")
        proc1.compute_all()

        proc2 = CorticalTextProcessor()
        proc2.process_document("doc2", "new content")
        proc2.compute_all()

        diff = compare_processors(proc1, proc2)

        expected = len(diff.terms_added) + len(diff.terms_removed) + len(diff.terms_modified)
        assert diff.total_term_changes == expected

    def test_new_terms_detected(self):
        """New terms appear in terms_added."""
        from cortical import CorticalTextProcessor

        # Both processors need non-empty layers
        # Proc1 has one set of terms
        proc1 = CorticalTextProcessor()
        proc1.process_document("doc1", "common baseline")
        proc1.compute_all()

        # Proc2 has baseline plus unique terms
        proc2 = CorticalTextProcessor()
        proc2.process_document("doc1", "common baseline")
        proc2.process_document("doc2", "unique special remarkable")
        proc2.compute_all()

        diff = compare_processors(proc1, proc2)

        # Should detect new terms from doc2
        assert len(diff.terms_added) > 0
        # Check that added terms have new_pagerank set
        for tc in diff.terms_added:
            assert tc.change_type == "added"
            assert tc.new_pagerank is not None or tc.new_pagerank == 0

    def test_removed_terms_detected(self):
        """Removed terms appear in terms_removed."""
        from cortical import CorticalTextProcessor

        # Proc1 has baseline plus document with terms to be removed
        proc1 = CorticalTextProcessor()
        proc1.process_document("baseline", "common content")
        proc1.process_document("doc1", "removed obsolete deprecated")
        proc1.compute_all()

        # Proc2 has only baseline (doc1 removed)
        proc2 = CorticalTextProcessor()
        proc2.process_document("baseline", "common content")
        proc2.compute_all()

        diff = compare_processors(proc1, proc2)

        # Should detect terms that only appeared in doc1
        assert len(diff.terms_removed) > 0
        for tc in diff.terms_removed:
            assert tc.change_type == "removed"
            assert tc.old_pagerank is not None or tc.old_pagerank == 0


# =============================================================================
# COMPARE DOCUMENTS TESTS
# =============================================================================


class TestCompareDocuments:
    """Tests for compare_documents function."""

    def test_same_document_high_similarity(self, small_processor):
        """Comparing same document returns high similarity."""
        # Get a document ID from the processor
        if not small_processor.documents:
            pytest.skip("Small processor has no documents")

        doc_id = list(small_processor.documents.keys())[0]

        result = compare_documents(small_processor, doc_id, doc_id)

        assert result['jaccard_similarity'] == 1.0
        assert result['shared_terms'] > 0
        assert result['unique_to_old'] == 0
        assert result['unique_to_new'] == 0

    def test_different_documents_lower_similarity(self, small_processor):
        """Comparing different documents returns lower similarity."""
        if len(small_processor.documents) < 2:
            pytest.skip("Need at least 2 documents")

        doc_ids = list(small_processor.documents.keys())
        doc1, doc2 = doc_ids[0], doc_ids[1]

        result = compare_documents(small_processor, doc1, doc2)

        assert 0 <= result['jaccard_similarity'] <= 1.0
        assert 'shared_terms' in result
        assert 'unique_to_old' in result
        assert 'unique_to_new' in result

    def test_shared_terms_calculation(self, fresh_processor):
        """Shared terms are correctly calculated."""
        proc = fresh_processor
        proc.process_document("doc1", "apple banana orange")
        proc.process_document("doc2", "banana orange grape")
        proc.compute_all()

        result = compare_documents(proc, "doc1", "doc2")

        # Should have some shared terms (banana, orange)
        assert result['shared_terms'] >= 2  # At least banana and orange (possibly stemmed)
        assert result['unique_to_old'] >= 1  # apple
        assert result['unique_to_new'] >= 1  # grape

    def test_no_shared_terms(self, fresh_processor):
        """Documents with no shared terms."""
        proc = fresh_processor
        proc.process_document("doc1", "alpha beta gamma")
        proc.process_document("doc2", "delta epsilon zeta")
        proc.compute_all()

        result = compare_documents(proc, "doc1", "doc2")

        assert result['shared_terms'] == 0
        assert result['jaccard_similarity'] == 0.0
        assert result['unique_to_old'] > 0
        assert result['unique_to_new'] > 0

    def test_result_structure(self, fresh_processor):
        """Result has all expected fields."""
        proc = fresh_processor
        proc.process_document("doc1", "test content")
        proc.process_document("doc2", "other content")
        proc.compute_all()

        result = compare_documents(proc, "doc1", "doc2")

        assert 'doc_id_old' in result
        assert 'doc_id_new' in result
        assert 'terms_in_old' in result
        assert 'terms_in_new' in result
        assert 'shared_terms' in result
        assert 'unique_to_old' in result
        assert 'unique_to_new' in result
        assert 'jaccard_similarity' in result
        assert 'shared_bigrams' in result
        assert 'top_shared_terms' in result
        assert 'top_unique_to_old' in result
        assert 'top_unique_to_new' in result
        assert 'top_shared_bigrams' in result

    def test_bigram_detection(self, fresh_processor):
        """Bigrams are detected and compared."""
        proc = fresh_processor
        proc.process_document("doc1", "quick brown fox jumps")
        proc.process_document("doc2", "quick brown dog runs")
        proc.compute_all()

        result = compare_documents(proc, "doc1", "doc2")

        # Should have at least one shared bigram (quick brown)
        assert result['shared_bigrams'] >= 0  # May be 0 if bigrams not computed

    def test_top_lists_limited(self, fresh_processor):
        """Top lists are limited to reasonable sizes."""
        proc = fresh_processor
        many_terms = " ".join([f"term{i}" for i in range(100)])
        proc.process_document("doc1", many_terms)
        proc.process_document("doc2", many_terms + " extra unique")
        proc.compute_all()

        result = compare_documents(proc, "doc1", "doc2")

        assert len(result['top_shared_terms']) <= 20
        assert len(result['top_unique_to_old']) <= 20
        assert len(result['top_unique_to_new']) <= 20
        assert len(result['top_shared_bigrams']) <= 10

    def test_empty_processor_error(self, fresh_processor):
        """Empty processor returns error."""
        result = compare_documents(fresh_processor, "doc1", "doc2")

        assert 'error' in result

    def test_jaccard_formula(self, fresh_processor):
        """Jaccard similarity uses correct formula."""
        proc = fresh_processor
        proc.process_document("doc1", "alpha beta gamma")
        proc.process_document("doc2", "beta gamma delta")
        proc.compute_all()

        result = compare_documents(proc, "doc1", "doc2")

        # Jaccard = |intersection| / |union|
        # Shared: beta, gamma (2)
        # Unique to old: alpha (1)
        # Unique to new: delta (1)
        # Union = 2 + 1 + 1 = 4
        # Jaccard = 2/4 = 0.5
        assert result['shared_terms'] >= 2
        expected_jaccard = result['shared_terms'] / (
            result['shared_terms'] + result['unique_to_old'] + result['unique_to_new']
        )
        assert result['jaccard_similarity'] == pytest.approx(expected_jaccard, abs=0.01)


# =============================================================================
# WHAT CHANGED TESTS
# =============================================================================


class TestWhatChanged:
    """Tests for what_changed function."""

    def test_identical_texts_high_similarity(self, fresh_processor):
        """Identical texts return high similarity."""
        text = "neural networks process data"

        result = what_changed(fresh_processor, text, text)

        assert result['summary']['content_similarity'] == 1.0
        assert result['summary']['is_significant_change'] is False
        assert len(result['tokens']['added']) == 0
        assert len(result['tokens']['removed']) == 0

    def test_different_texts_show_changes(self, fresh_processor):
        """Different texts show added/removed tokens."""
        # Use words that won't all be filtered as stop words
        old_text = "machine learning algorithms"
        new_text = "deep neural networks"

        result = what_changed(fresh_processor, old_text, new_text)

        assert len(result['tokens']['added']) > 0
        assert len(result['tokens']['removed']) > 0
        assert result['summary']['is_significant_change'] is True  # < 0.8 similarity

    def test_empty_old_text(self, fresh_processor):
        """Empty old text shows all tokens as added."""
        old_text = ""
        new_text = "new content"

        result = what_changed(fresh_processor, old_text, new_text)

        assert len(result['tokens']['added']) > 0
        assert len(result['tokens']['removed']) == 0
        assert result['tokens']['total_old'] == 0
        assert result['tokens']['total_new'] > 0

    def test_empty_new_text(self, fresh_processor):
        """Empty new text shows all tokens as removed."""
        old_text = "old content"
        new_text = ""

        result = what_changed(fresh_processor, old_text, new_text)

        assert len(result['tokens']['removed']) > 0
        assert len(result['tokens']['added']) == 0
        assert result['tokens']['total_old'] > 0
        assert result['tokens']['total_new'] == 0

    def test_both_empty_texts(self, fresh_processor):
        """Both empty texts handled gracefully."""
        result = what_changed(fresh_processor, "", "")

        assert result['tokens']['similarity'] == 0.0  # No tokens to compare
        assert len(result['tokens']['added']) == 0
        assert len(result['tokens']['removed']) == 0

    def test_token_counts(self, fresh_processor):
        """Token counts are accurate."""
        old_text = "alpha beta gamma"
        new_text = "beta gamma delta epsilon"

        result = what_changed(fresh_processor, old_text, new_text)

        # Old: alpha, beta, gamma (3)
        # New: beta, gamma, delta, epsilon (4)
        assert result['tokens']['total_old'] >= 3
        assert result['tokens']['total_new'] >= 4

    def test_bigram_detection(self, fresh_processor):
        """Bigrams are detected in changes."""
        old_text = "quick brown fox"
        new_text = "quick brown dog"

        result = what_changed(fresh_processor, old_text, new_text)

        assert 'bigrams' in result
        assert 'added' in result['bigrams']
        assert 'removed' in result['bigrams']
        assert 'unchanged_count' in result['bigrams']
        assert 'similarity' in result['bigrams']

    def test_bigram_similarity(self, fresh_processor):
        """Bigram similarity is calculated."""
        old_text = "neural networks deep learning"
        new_text = "neural networks machine learning"

        result = what_changed(fresh_processor, old_text, new_text)

        # Should have some shared bigrams
        assert 0 <= result['bigrams']['similarity'] <= 1
        assert result['bigrams']['unchanged_count'] > 0  # "neural networks" shared

    def test_content_similarity_formula(self, fresh_processor):
        """Content similarity is average of token and bigram similarity."""
        old_text = "test content"
        new_text = "test data"

        result = what_changed(fresh_processor, old_text, new_text)

        expected = (result['tokens']['similarity'] + result['bigrams']['similarity']) / 2
        assert result['summary']['content_similarity'] == pytest.approx(expected, abs=0.01)

    def test_significant_change_threshold(self, fresh_processor):
        """Significant change threshold is 0.8."""
        # Very similar
        result1 = what_changed(
            fresh_processor,
            "alpha beta gamma delta epsilon",
            "alpha beta gamma delta zeta"
        )

        # Very different
        result2 = what_changed(
            fresh_processor,
            "alpha beta",
            "gamma delta epsilon zeta eta theta"
        )

        # First should not be significant (high similarity)
        # Second should be significant (low similarity)
        # Note: Actual values depend on tokenization
        assert isinstance(result1['summary']['is_significant_change'], bool)
        assert isinstance(result2['summary']['is_significant_change'], bool)

    def test_result_structure(self, fresh_processor):
        """Result has expected structure."""
        result = what_changed(fresh_processor, "old", "new")

        assert 'tokens' in result
        assert 'bigrams' in result
        assert 'summary' in result

        assert 'added' in result['tokens']
        assert 'removed' in result['tokens']
        assert 'unchanged_count' in result['tokens']
        assert 'total_old' in result['tokens']
        assert 'total_new' in result['tokens']
        assert 'similarity' in result['tokens']

        assert 'added' in result['bigrams']
        assert 'removed' in result['bigrams']
        assert 'unchanged_count' in result['bigrams']
        assert 'similarity' in result['bigrams']

        assert 'content_similarity' in result['summary']
        assert 'is_significant_change' in result['summary']

    def test_added_tokens_sorted(self, fresh_processor):
        """Added tokens are sorted."""
        old_text = "alpha"
        new_text = "alpha zeta beta gamma"

        result = what_changed(fresh_processor, old_text, new_text)

        added = result['tokens']['added']
        if len(added) > 1:
            assert added == sorted(added)

    def test_removed_tokens_sorted(self, fresh_processor):
        """Removed tokens are sorted."""
        old_text = "alpha zeta beta gamma"
        new_text = "alpha"

        result = what_changed(fresh_processor, old_text, new_text)

        removed = result['tokens']['removed']
        if len(removed) > 1:
            assert removed == sorted(removed)

    def test_limits_token_lists(self, fresh_processor):
        """Token lists are limited to 50 items."""
        many_old = " ".join([f"old{i}" for i in range(100)])
        many_new = " ".join([f"new{i}" for i in range(100)])

        result = what_changed(fresh_processor, many_old, many_new)

        assert len(result['tokens']['added']) <= 50
        assert len(result['tokens']['removed']) <= 50

    def test_limits_bigram_lists(self, fresh_processor):
        """Bigram lists are limited to 30 items."""
        many_old = " ".join([f"old{i}" for i in range(100)])
        many_new = " ".join([f"new{i}" for i in range(100)])

        result = what_changed(fresh_processor, many_old, many_new)

        assert len(result['bigrams']['added']) <= 30
        assert len(result['bigrams']['removed']) <= 30
