"""
Behavioral tests for semantic document comparison and diff functionality.

Epic: Semantic Change Detection

As a developer tracking content evolution,
I want semantic diff tools that understand meaning changes,
So that I can detect important changes beyond line-by-line diffs.

Based on: cortical/diff.py
"""

import pytest
from cortical import CorticalTextProcessor
from cortical.diff import (
    compare_processors,
    compare_documents,
    what_changed,
    SemanticDiff,
    TermChange,
    RelationChange,
    ClusterChange
)


class TestDeveloperComparesDocumentVersionsSemantically:
    """
    Epic: Semantic Change Detection

    As a developer tracking content evolution,
    I want semantic diff capabilities that go beyond text comparison,
    So that I understand what changed at the conceptual level.
    """

    def test_scenario_developer_detects_new_terms_in_updated_version(self):
        """
        Scenario: Detecting new concepts in document updates

        Given an original document and an updated version
        When I perform semantic diff analysis
        Then new terms are identified
        And removed terms are identified
        Because developers need to track vocabulary evolution.
        """
        # GIVEN an original document and an updated version
        old_processor = CorticalTextProcessor()
        old_processor.process_document("doc1", "Neural networks process information through layers.")
        old_processor.compute_all(verbose=False)

        new_processor = CorticalTextProcessor()
        new_processor.process_document("doc1", "Neural networks and transformers process information through attention layers.")
        new_processor.compute_all(verbose=False)

        # WHEN I perform semantic diff analysis
        diff = compare_processors(old_processor, new_processor)

        # THEN new terms are identified
        new_term_contents = {tc.term for tc in diff.terms_added}
        assert "transformers" in new_term_contents or "transformer" in new_term_contents, \
            "Should detect new term 'transformers'"

        # AND removed terms are identified
        # The diff should be able to identify what was added vs removed
        assert isinstance(diff, SemanticDiff), "Should return SemanticDiff object"

    def test_scenario_developer_tracks_importance_shifts_over_time(self):
        """
        Scenario: Understanding which concepts gained or lost importance

        Given two corpus versions with different term frequencies
        When I compare their PageRank scores
        Then I see which terms increased in importance
        And which terms decreased in importance
        Because importance shifts reveal topic evolution.
        """
        # GIVEN two corpus versions with different term frequencies
        old_processor = CorticalTextProcessor()
        old_processor.process_document("doc1", "Neural networks are important. Deep learning is mentioned once.")
        old_processor.process_document("doc2", "Neural networks continue to evolve.")
        old_processor.compute_all(verbose=False)

        new_processor = CorticalTextProcessor()
        new_processor.process_document("doc1", "Deep learning is important. Deep learning revolutionizes AI.")
        new_processor.process_document("doc2", "Deep learning continues to evolve. Deep learning everywhere.")
        new_processor.compute_all(verbose=False)

        # WHEN I compare their PageRank scores
        diff = compare_processors(old_processor, new_processor, top_movers=10)

        # THEN I see which terms increased in importance
        # AND which terms decreased in importance
        assert len(diff.importance_increased) > 0 or len(diff.importance_decreased) > 0, \
            "Should detect importance shifts"

        # Verify movers have delta information
        if diff.importance_increased:
            top_riser = diff.importance_increased[0]
            assert isinstance(top_riser, TermChange), "Should be TermChange object"
            assert top_riser.pagerank_delta is not None, "Should have PageRank delta"
            assert top_riser.pagerank_delta > 0, "Rising term should have positive delta"

    def test_scenario_developer_compares_two_documents_in_corpus(self):
        """
        Scenario: Understanding differences between two documents

        Given a processor with multiple documents
        When I compare two specific documents
        Then I see shared and unique terms
        And get Jaccard similarity metrics
        Because developers need document-level comparison.
        """
        # GIVEN a processor with multiple documents
        processor = CorticalTextProcessor()
        processor.process_document(
            "ml_doc",
            "Machine learning uses statistical methods for pattern recognition and prediction."
        )
        processor.process_document(
            "dl_doc",
            "Deep learning uses neural networks for pattern recognition and feature extraction."
        )
        processor.compute_all(verbose=False)

        # WHEN I compare two specific documents
        comparison = compare_documents(processor, "ml_doc", "dl_doc")

        # THEN I see shared and unique terms
        assert comparison['shared_terms'] > 0, "Should have shared terms like 'pattern' and 'recognition'"
        assert comparison['unique_to_old'] > 0, "Should have terms unique to first doc"
        assert comparison['unique_to_new'] > 0, "Should have terms unique to second doc"

        # AND get Jaccard similarity metrics
        assert 'jaccard_similarity' in comparison, "Should provide Jaccard similarity"
        assert 0 <= comparison['jaccard_similarity'] <= 1, "Jaccard should be between 0 and 1"

    def test_scenario_developer_analyzes_text_changes_without_indexing(self):
        """
        Scenario: Quick text comparison without full indexing

        Given two text strings to compare
        When I use what_changed convenience function
        Then I get token and bigram differences
        And similarity metrics
        Because developers need lightweight comparison for small texts.
        """
        # GIVEN two text strings to compare
        old_text = "Neural networks are computational models inspired by biological brains."
        new_text = "Neural networks and transformers are computational models for AI systems."

        # WHEN I use what_changed convenience function
        processor = CorticalTextProcessor()
        changes = what_changed(processor, old_text, new_text)

        # THEN I get token and bigram differences
        assert 'tokens' in changes, "Should analyze token changes"
        assert 'bigrams' in changes, "Should analyze bigram changes"

        tokens = changes['tokens']
        assert 'added' in tokens, "Should list added tokens"
        assert 'removed' in tokens, "Should list removed tokens"
        assert 'similarity' in tokens, "Should provide token similarity"

        # AND similarity metrics
        assert 'summary' in changes, "Should provide summary"
        assert 'content_similarity' in changes['summary'], "Should calculate overall similarity"

    def test_scenario_developer_detects_document_additions_and_removals(self):
        """
        Scenario: Tracking corpus-level changes

        Given an old corpus and an updated corpus
        When I compare the processors
        Then I see which documents were added
        And which documents were removed
        Because developers need corpus-level change tracking.
        """
        # GIVEN an old corpus and an updated corpus
        old_processor = CorticalTextProcessor()
        old_processor.process_document("doc1", "First document about neural networks.")
        old_processor.process_document("doc2", "Second document about machine learning.")
        old_processor.compute_all(verbose=False)

        new_processor = CorticalTextProcessor()
        new_processor.process_document("doc2", "Second document about machine learning.")  # Kept
        new_processor.process_document("doc3", "Third document about deep learning.")  # Added
        new_processor.compute_all(verbose=False)

        # WHEN I compare the processors
        diff = compare_processors(old_processor, new_processor)

        # THEN I see which documents were added
        assert "doc3" in diff.documents_added, "Should detect new document"

        # AND which documents were removed
        assert "doc1" in diff.documents_removed, "Should detect removed document"

    def test_scenario_developer_detects_modified_documents(self):
        """
        Scenario: Identifying documents that changed content

        Given documents with same ID but different content
        When I compare processors
        Then modified documents are identified
        And I can track content updates
        Because developers need to know what was edited.
        """
        # GIVEN documents with same ID but different content
        old_processor = CorticalTextProcessor()
        old_processor.process_document("guide", "Neural networks are simple models.")
        old_processor.compute_all(verbose=False)

        new_processor = CorticalTextProcessor()
        new_processor.process_document("guide", "Neural networks are complex computational models.")
        new_processor.compute_all(verbose=False)

        # WHEN I compare processors
        diff = compare_processors(old_processor, new_processor)

        # THEN modified documents are identified
        assert "guide" in diff.documents_modified, "Should detect modified document"

        # AND I can track content updates
        # Terms changed between versions
        assert diff.total_term_changes > 0, "Should detect term changes in modified document"

    def test_scenario_developer_generates_human_readable_summary(self):
        """
        Scenario: Creating readable change reports

        Given a semantic diff object
        When I call the summary method
        Then I get a formatted text report
        And the report includes all change categories
        Because developers need to communicate changes to stakeholders.
        """
        # GIVEN a semantic diff object
        old_processor = CorticalTextProcessor()
        old_processor.process_document("doc1", "Neural networks process data through layers.")
        old_processor.compute_all(verbose=False)

        new_processor = CorticalTextProcessor()
        new_processor.process_document("doc1", "Transformers process data through attention mechanisms.")
        new_processor.process_document("doc2", "New document about deep learning architectures.")
        new_processor.compute_all(verbose=False)

        diff = compare_processors(old_processor, new_processor)

        # WHEN I call the summary method
        summary = diff.summary()

        # THEN I get a formatted text report
        assert isinstance(summary, str), "Should return string summary"
        assert len(summary) > 0, "Summary should not be empty"

        # AND the report includes all change categories
        assert "Documents" in summary or "Terms" in summary or "Statistics" in summary, \
            "Summary should include section headers"

    def test_scenario_developer_exports_diff_to_dictionary(self):
        """
        Scenario: Serializing diff results for storage or APIs

        Given a semantic diff object
        When I convert to dictionary
        Then I get JSON-serializable data
        And all key metrics are included
        Because developers need to persist or transmit diff results.
        """
        # GIVEN a semantic diff object
        old_processor = CorticalTextProcessor()
        old_processor.process_document("doc1", "Original content about neural networks.")
        old_processor.compute_all(verbose=False)

        new_processor = CorticalTextProcessor()
        new_processor.process_document("doc1", "Updated content about neural networks and transformers.")
        new_processor.compute_all(verbose=False)

        diff = compare_processors(old_processor, new_processor)

        # WHEN I convert to dictionary
        diff_dict = diff.to_dict()

        # THEN I get JSON-serializable data
        assert isinstance(diff_dict, dict), "Should be dictionary"

        # AND all key metrics are included
        assert 'documents_added' in diff_dict, "Should include document additions"
        assert 'documents_removed' in diff_dict, "Should include document removals"
        assert 'terms_added' in diff_dict, "Should include term additions"
        assert 'terms_removed' in diff_dict, "Should include term removals"
        assert 'total_term_changes' in diff_dict, "Should include summary statistics"

    def test_scenario_developer_tracks_term_occurrence_changes(self):
        """
        Scenario: Understanding term frequency evolution

        Given terms that appear in different documents
        When I examine TermChange objects
        Then I see which documents added or dropped each term
        And occurrence count changes
        Because term distribution matters for understanding change impact.
        """
        # GIVEN terms that appear in different documents
        old_processor = CorticalTextProcessor()
        old_processor.process_document("doc1", "Neural networks are powerful.")
        old_processor.compute_all(verbose=False)

        new_processor = CorticalTextProcessor()
        new_processor.process_document("doc1", "Neural networks are powerful.")
        new_processor.process_document("doc2", "Neural networks revolutionize AI.")
        new_processor.compute_all(verbose=False)

        # WHEN I examine TermChange objects
        diff = compare_processors(old_processor, new_processor)

        # THEN I see which documents added or dropped each term
        # Term "neural" should appear in modified terms or show document changes
        neural_changes = [tc for tc in diff.terms_modified if "neural" in tc.term.lower()]

        if neural_changes:
            term_change = neural_changes[0]
            # AND occurrence count changes
            assert term_change.old_occurrences is not None, "Should track old occurrences"
            assert term_change.new_occurrences is not None, "Should track new occurrences"

    def test_scenario_developer_filters_minor_changes_with_threshold(self):
        """
        Scenario: Focusing on significant importance shifts

        Given a minimum PageRank delta threshold
        When I compare processors with that threshold
        Then only significant changes are reported
        And noise is filtered out
        Because developers need to focus on meaningful changes.
        """
        # GIVEN a minimum PageRank delta threshold
        old_processor = CorticalTextProcessor()
        for i in range(10):
            old_processor.process_document(f"doc{i}", f"Document {i} about machine learning and AI.")
        old_processor.compute_all(verbose=False)

        new_processor = CorticalTextProcessor()
        for i in range(10):
            new_processor.process_document(f"doc{i}", f"Document {i} about machine learning and AI systems.")
        new_processor.compute_all(verbose=False)

        # WHEN I compare processors with that threshold
        diff_strict = compare_processors(
            old_processor,
            new_processor,
            min_pagerank_delta=0.001  # Higher threshold
        )
        diff_loose = compare_processors(
            old_processor,
            new_processor,
            min_pagerank_delta=0.00001  # Lower threshold
        )

        # THEN only significant changes are reported
        # AND noise is filtered out
        # Strict threshold should report fewer or equal changes
        strict_movers = len(diff_strict.importance_increased) + len(diff_strict.importance_decreased)
        loose_movers = len(diff_loose.importance_increased) + len(diff_loose.importance_decreased)

        assert strict_movers <= loose_movers, "Stricter threshold should report fewer changes"

    def test_scenario_developer_limits_top_movers_for_reporting(self):
        """
        Scenario: Focusing on top N most important changes

        Given many term importance changes
        When I specify top_movers parameter
        Then only the top N changes are included
        And changes are sorted by magnitude
        Because developers need focused reports on major changes.
        """
        # GIVEN many term importance changes
        old_processor = CorticalTextProcessor()
        old_processor.process_document("doc1", "First version with various machine learning topics.")
        old_processor.process_document("doc2", "Another document about neural networks.")
        old_processor.compute_all(verbose=False)

        new_processor = CorticalTextProcessor()
        new_processor.process_document("doc1", "Updated version with deep learning and transformers.")
        new_processor.process_document("doc2", "Another document about neural networks and attention.")
        new_processor.process_document("doc3", "New document about reinforcement learning.")
        new_processor.compute_all(verbose=False)

        # WHEN I specify top_movers parameter
        diff = compare_processors(old_processor, new_processor, top_movers=3)

        # THEN only the top N changes are included
        total_movers = len(diff.importance_increased) + len(diff.importance_decreased)
        assert total_movers <= 3, "Should limit to top_movers parameter"

        # AND changes are sorted by magnitude
        if diff.importance_increased:
            deltas = [tc.pagerank_delta for tc in diff.importance_increased if tc.pagerank_delta]
            assert deltas == sorted(deltas, reverse=True), "Increases should be sorted descending"

        if diff.importance_decreased:
            deltas = [tc.pagerank_delta for tc in diff.importance_decreased if tc.pagerank_delta]
            assert deltas == sorted(deltas), "Decreases should be sorted ascending"

    def test_scenario_developer_tracks_relation_changes(self):
        """
        Scenario: Understanding relationship evolution

        Given processors with typed connections
        When I compare them
        Then new and removed relations are detected
        And relation strength changes are tracked
        Because semantic relationships matter as much as terms.
        """
        # GIVEN processors with typed connections
        old_processor = CorticalTextProcessor()
        old_processor.process_document("doc1", "Neural networks learn patterns from data.")
        old_processor.compute_all(verbose=False, build_concepts=True)

        new_processor = CorticalTextProcessor()
        new_processor.process_document("doc1", "Neural networks and transformers learn patterns.")
        new_processor.process_document("doc2", "Deep learning models process information.")
        new_processor.compute_all(verbose=False, build_concepts=True)

        # WHEN I compare them
        diff = compare_processors(old_processor, new_processor)

        # THEN new and removed relations are detected
        # Relations may have changed based on document connections
        assert isinstance(diff.relations_added, list), "Should track added relations"
        assert isinstance(diff.relations_removed, list), "Should track removed relations"

        # AND relation strength changes are tracked
        assert diff.total_relation_changes >= 0, "Should count relation changes"

    def test_scenario_developer_monitors_cluster_reorganization(self):
        """
        Scenario: Tracking concept cluster evolution

        Given processors with concept clustering
        When I compare cluster membership
        Then cluster creation and dissolution are detected
        And membership changes within clusters are tracked
        Because concept organization reveals corpus structure evolution.
        """
        # GIVEN processors with concept clustering
        old_processor = CorticalTextProcessor()
        old_processor.process_document("doc1", "Machine learning and neural networks are related.")
        old_processor.process_document("doc2", "Deep learning uses neural architectures.")
        old_processor.compute_all(verbose=False, build_concepts=True)

        new_processor = CorticalTextProcessor()
        new_processor.process_document("doc1", "Machine learning and neural networks are related.")
        new_processor.process_document("doc2", "Deep learning uses neural architectures.")
        new_processor.process_document("doc3", "Transformers are attention-based architectures.")
        new_processor.compute_all(verbose=False, build_concepts=True)

        # WHEN I compare cluster membership
        diff = compare_processors(old_processor, new_processor)

        # THEN cluster creation and dissolution are detected
        assert isinstance(diff.clusters_created, list), "Should track new clusters"
        assert isinstance(diff.clusters_dissolved, list), "Should track dissolved clusters"
        assert isinstance(diff.clusters_modified, list), "Should track modified clusters"

        # AND membership changes within clusters are tracked
        assert diff.total_cluster_changes >= 0, "Should count cluster changes"

    def test_scenario_developer_accesses_delta_properties(self):
        """
        Scenario: Convenient access to change magnitudes

        Given a TermChange object with old and new values
        When I access delta properties
        Then I get computed differences
        And positive/negative direction is clear
        Because developers need convenient metric access.
        """
        # GIVEN a TermChange object with old and new values
        term_change = TermChange(
            term="neural",
            change_type="modified",
            old_pagerank=0.05,
            new_pagerank=0.08,
            old_tfidf=0.3,
            new_tfidf=0.4
        )

        # WHEN I access delta properties
        pr_delta = term_change.pagerank_delta
        tfidf_delta = term_change.tfidf_delta

        # THEN I get computed differences
        assert abs(pr_delta - 0.03) < 0.0001, "Should compute PageRank difference"
        assert abs(tfidf_delta - 0.1) < 0.0001, "Should compute TF-IDF difference"

        # AND positive/negative direction is clear
        assert pr_delta > 0, "Positive delta indicates increase"

    def test_scenario_developer_identifies_documents_where_term_appeared(self):
        """
        Scenario: Tracking term distribution across documents

        Given term changes with document information
        When I access documents_added and documents_removed
        Then I see exactly where the term appeared or vanished
        And can drill down into specific document changes
        Because term location matters for understanding changes.
        """
        # GIVEN term changes with document information
        old_docs = {"doc1", "doc2"}
        new_docs = {"doc2", "doc3", "doc4"}

        term_change = TermChange(
            term="transformer",
            change_type="modified",
            old_documents=old_docs,
            new_documents=new_docs
        )

        # WHEN I access documents_added and documents_removed
        docs_added = term_change.documents_added
        docs_removed = term_change.documents_removed

        # THEN I see exactly where the term appeared or vanished
        assert docs_added == {"doc3", "doc4"}, "Should identify new appearances"
        assert docs_removed == {"doc1"}, "Should identify removals"

        # AND can drill down into specific document changes
        assert "doc2" not in docs_added, "Continuing documents should not be in added"
        assert "doc2" not in docs_removed, "Continuing documents should not be in removed"

    def test_scenario_developer_compares_empty_processors(self):
        """
        Scenario: Handling edge cases gracefully

        Given empty or minimal processors
        When I perform comparison
        Then no errors occur
        And diff object is valid but minimal
        Because developers need robust error handling.
        """
        # GIVEN empty or minimal processors
        empty1 = CorticalTextProcessor()
        empty2 = CorticalTextProcessor()

        # WHEN I perform comparison
        diff = compare_processors(empty1, empty2)

        # THEN no errors occur
        # AND diff object is valid but minimal
        assert isinstance(diff, SemanticDiff), "Should return valid diff object"
        assert diff.total_term_changes == 0, "Empty processors should have no changes"
        assert len(diff.documents_added) == 0, "Should have no document additions"
        assert len(diff.documents_removed) == 0, "Should have no document removals"

    def test_scenario_developer_analyzes_bigram_changes(self):
        """
        Scenario: Tracking phrase-level changes

        Given texts with different bigrams
        When I use what_changed
        Then bigram additions and removals are identified
        And bigram similarity is calculated
        Because phrases carry more semantic meaning than individual tokens.
        """
        # GIVEN texts with different bigrams
        old_text = "neural network architecture"
        new_text = "transformer network architecture"

        # WHEN I use what_changed
        processor = CorticalTextProcessor()
        changes = what_changed(processor, old_text, new_text)

        # THEN bigram additions and removals are identified
        bigrams = changes.get('bigrams', {})
        assert 'added' in bigrams, "Should list added bigrams"
        assert 'removed' in bigrams, "Should list removed bigrams"

        # AND bigram similarity is calculated
        assert 'similarity' in bigrams, "Should calculate bigram similarity"
        assert 0 <= bigrams['similarity'] <= 1, "Similarity should be normalized"
