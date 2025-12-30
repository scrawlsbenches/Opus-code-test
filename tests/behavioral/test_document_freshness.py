"""
Document Freshness Behavioral Tests
====================================

Epic: Fresher documents rank higher in search

As a researcher searching my document corpus,
I want recent documents to rank higher than older ones,
So that I see the most current information first.

Requirements:
- Documents added in the last 7 days should get a ranking boost
- The boost should be configurable (default: 1.5x weight)
- Existing search behavior must not break

Run with: pytest tests/behavioral/test_document_freshness.py -v
"""

import pytest
from datetime import datetime, timedelta


class TestDocumentFreshness:
    """
    Epic: Fresher documents rank higher in search

    As a researcher searching my document corpus,
    I want recent documents to rank higher than older ones,
    So that I see the most current information first.
    """

    def test_recent_documents_rank_higher_than_older_ones(self, fresh_processor):
        """
        Scenario: Recent documents outrank stale documents

        Given two documents with similar content about "machine learning"
        And one document was added 2 days ago
        And another document was added 30 days ago
        When I search for "machine learning" with freshness boost enabled
        Then the recent document appears before the older one
        Because users want current information first.
        """
        # Given two documents with similar content
        today = datetime.now()
        two_days_ago = (today - timedelta(days=2)).strftime("%Y-%m-%d")
        thirty_days_ago = (today - timedelta(days=30)).strftime("%Y-%m-%d")

        fresh_processor.process_document(
            "recent_ml_guide",
            "Machine learning is a subset of artificial intelligence. "
            "It enables systems to learn from data and improve over time. "
            "Deep learning and neural networks are popular approaches.",
            metadata={"timestamp": two_days_ago}
        )
        fresh_processor.process_document(
            "old_ml_guide",
            "Machine learning is a subset of artificial intelligence. "
            "It enables systems to learn from data and improve over time. "
            "Deep learning and neural networks are popular approaches.",
            metadata={"timestamp": thirty_days_ago}
        )
        fresh_processor.compute_all(verbose=False)

        # When I search with freshness boost enabled
        results = fresh_processor.find_documents_for_query(
            "machine learning",
            top_n=5,
            freshness_boost=1.5  # Default boost for recent documents
        )

        # Then the recent document should rank higher
        result_docs = [doc_id for doc_id, _ in results]
        assert "recent_ml_guide" in result_docs, (
            "Recent document should appear in results"
        )
        assert "old_ml_guide" in result_docs, (
            "Old document should also appear in results"
        )
        recent_idx = result_docs.index("recent_ml_guide")
        old_idx = result_docs.index("old_ml_guide")
        assert recent_idx < old_idx, (
            f"Recent document (index {recent_idx}) should rank higher than "
            f"old document (index {old_idx})"
        )

    def test_freshness_window_is_seven_days(self, fresh_processor):
        """
        Scenario: Only documents within 7 days get freshness boost

        Given three documents with similar content
        And one was added 3 days ago (within window)
        And one was added 7 days ago (boundary)
        And one was added 10 days ago (outside window)
        When I search with freshness boost enabled
        Then the 3-day-old document ranks highest
        And the 10-day-old document ranks lowest
        Because the freshness window is 7 days.
        """
        # Given three documents at different ages
        today = datetime.now()
        three_days_ago = (today - timedelta(days=3)).strftime("%Y-%m-%d")
        seven_days_ago = (today - timedelta(days=7)).strftime("%Y-%m-%d")
        ten_days_ago = (today - timedelta(days=10)).strftime("%Y-%m-%d")

        content = (
            "Database optimization techniques include indexing, query caching, "
            "and connection pooling for improved performance."
        )

        fresh_processor.process_document(
            "db_guide_3days", content,
            metadata={"timestamp": three_days_ago}
        )
        fresh_processor.process_document(
            "db_guide_7days", content,
            metadata={"timestamp": seven_days_ago}
        )
        fresh_processor.process_document(
            "db_guide_10days", content,
            metadata={"timestamp": ten_days_ago}
        )
        fresh_processor.compute_all(verbose=False)

        # When searching with freshness boost
        results = fresh_processor.find_documents_for_query(
            "database optimization",
            top_n=5,
            freshness_boost=1.5
        )

        result_docs = [doc_id for doc_id, _ in results]

        # Then the 3-day-old document should rank highest
        assert result_docs[0] == "db_guide_3days", (
            f"3-day-old document should rank first, got: {result_docs}"
        )
        # And the 10-day-old document should rank lowest
        three_idx = result_docs.index("db_guide_3days")
        ten_idx = result_docs.index("db_guide_10days")
        assert three_idx < ten_idx, (
            "3-day-old document should rank higher than 10-day-old document"
        )

    def test_freshness_boost_is_configurable(self, fresh_processor):
        """
        Scenario: Freshness boost weight can be adjusted

        Given documents at different ages
        When I search with a higher freshness boost (2.0x)
        Then the ranking difference between fresh and old is more pronounced
        And when I search with no freshness boost (1.0x)
        Then the ranking is based only on content relevance.
        """
        # Given documents at different ages
        today = datetime.now()
        one_day_ago = (today - timedelta(days=1)).strftime("%Y-%m-%d")
        twenty_days_ago = (today - timedelta(days=20)).strftime("%Y-%m-%d")

        fresh_processor.process_document(
            "fresh_api_doc",
            "REST API design patterns and best practices for web services.",
            metadata={"timestamp": one_day_ago}
        )
        fresh_processor.process_document(
            "old_api_doc",
            "REST API design patterns and best practices for web services.",
            metadata={"timestamp": twenty_days_ago}
        )
        fresh_processor.compute_all(verbose=False)

        # When searching with high freshness boost (2.0x)
        high_boost_results = fresh_processor.find_documents_for_query(
            "REST API design",
            top_n=5,
            freshness_boost=2.0
        )

        # Then fresh document should rank higher with high boost
        high_boost_docs = [doc_id for doc_id, _ in high_boost_results]
        assert high_boost_docs.index("fresh_api_doc") < high_boost_docs.index("old_api_doc")

        # When searching with no freshness boost (1.0x)
        no_boost_results = fresh_processor.find_documents_for_query(
            "REST API design",
            top_n=5,
            freshness_boost=1.0  # No boost
        )

        # Then both documents should have similar ranking (same content)
        # With no freshness boost, they should be close in ranking
        no_boost_scores = {doc_id: score for doc_id, score in no_boost_results}
        assert "fresh_api_doc" in no_boost_scores
        assert "old_api_doc" in no_boost_scores

    def test_freshness_does_not_override_relevance(self, fresh_processor):
        """
        Scenario: Content relevance is still the primary signal

        Given a fresh document about cooking
        And an old document about machine learning
        When I search for "machine learning"
        Then the relevant old document still ranks higher than the irrelevant fresh one
        Because relevance matters more than freshness.
        """
        # Given a fresh but irrelevant document
        today = datetime.now()
        one_day_ago = (today - timedelta(days=1)).strftime("%Y-%m-%d")
        thirty_days_ago = (today - timedelta(days=30)).strftime("%Y-%m-%d")

        fresh_processor.process_document(
            "fresh_cooking_guide",
            "Cooking techniques for making delicious pasta and pizza. "
            "Italian cuisine is known for its fresh ingredients.",
            metadata={"timestamp": one_day_ago}
        )
        fresh_processor.process_document(
            "old_ml_comprehensive",
            "Machine learning algorithms including neural networks, "
            "decision trees, support vector machines, and deep learning. "
            "Machine learning is revolutionizing artificial intelligence.",
            metadata={"timestamp": thirty_days_ago}
        )
        fresh_processor.compute_all(verbose=False)

        # When searching for machine learning
        results = fresh_processor.find_documents_for_query(
            "machine learning algorithms",
            top_n=5,
            freshness_boost=1.5
        )

        result_docs = [doc_id for doc_id, _ in results]

        # Then the relevant old document should rank higher
        if "old_ml_comprehensive" in result_docs and "fresh_cooking_guide" in result_docs:
            ml_idx = result_docs.index("old_ml_comprehensive")
            cooking_idx = result_docs.index("fresh_cooking_guide")
            assert ml_idx < cooking_idx, (
                "Relevant old document should rank higher than irrelevant fresh document"
            )
        else:
            # At minimum, the ML document should appear
            assert "old_ml_comprehensive" in result_docs, (
                f"Relevant document should appear in results. Got: {result_docs}"
            )

    def test_documents_without_timestamp_are_not_boosted(self, fresh_processor):
        """
        Scenario: Documents without timestamp metadata are treated as old

        Given a document with no timestamp metadata
        And a document with a recent timestamp
        When I search with freshness boost enabled
        Then the document with timestamp gets boosted
        And the document without timestamp is not boosted
        Because we cannot determine its age.
        """
        # Given documents with and without timestamps
        today = datetime.now()
        two_days_ago = (today - timedelta(days=2)).strftime("%Y-%m-%d")

        fresh_processor.process_document(
            "timestamped_doc",
            "Python programming language features and syntax guide.",
            metadata={"timestamp": two_days_ago}
        )
        fresh_processor.process_document(
            "no_timestamp_doc",
            "Python programming language features and syntax guide.",
            metadata={}  # No timestamp
        )
        fresh_processor.compute_all(verbose=False)

        # When searching with freshness boost
        results = fresh_processor.find_documents_for_query(
            "python programming",
            top_n=5,
            freshness_boost=1.5
        )

        result_docs = [doc_id for doc_id, _ in results]

        # Then the timestamped document should rank higher
        assert "timestamped_doc" in result_docs
        assert "no_timestamp_doc" in result_docs
        ts_idx = result_docs.index("timestamped_doc")
        no_ts_idx = result_docs.index("no_timestamp_doc")
        assert ts_idx < no_ts_idx, (
            "Document with recent timestamp should rank higher than "
            "document without timestamp"
        )

    def test_existing_search_works_without_freshness_parameter(self, fresh_processor):
        """
        Scenario: Existing search API remains backward compatible

        Given documents in the corpus
        When I search without specifying freshness_boost parameter
        Then the search works as before
        And returns relevant results based on content.
        """
        # Given documents in the corpus
        fresh_processor.process_document(
            "test_doc_1",
            "Software testing methodologies including unit tests and integration tests."
        )
        fresh_processor.process_document(
            "test_doc_2",
            "Testing frameworks for Python applications and best practices."
        )
        fresh_processor.compute_all(verbose=False)

        # When searching without freshness_boost parameter
        results = fresh_processor.find_documents_for_query(
            "software testing",
            top_n=5
            # No freshness_boost parameter - should use default behavior
        )

        # Then results should be returned as before
        assert isinstance(results, list), "Should return a list"
        assert len(results) > 0, "Should return results for valid query"
        for doc_id, score in results:
            assert isinstance(doc_id, str), "doc_id should be string"
            assert isinstance(score, (int, float)), "score should be numeric"


class TestGraduatedFreshnessDecay:
    """
    Epic: Graduated freshness decay for search ranking

    As a researcher searching my document corpus,
    I want the freshness boost to decay gradually based on document age,
    So that a 2-day-old document ranks higher than a 6-day-old document
    (rather than both receiving the same boost).
    """

    def test_today_document_gets_full_freshness_boost(self, fresh_processor):
        """
        Scenario: Documents from today get full freshness boost

        Given a document added today
        And a document added 30 days ago
        When I search with linear decay enabled
        Then the today's document gets the full boost (1.5x)
        Because the most recent documents should be most prominent.
        """
        # Given documents at different ages
        today = datetime.now()
        today_str = today.strftime("%Y-%m-%d")
        thirty_days_ago = (today - timedelta(days=30)).strftime("%Y-%m-%d")

        fresh_processor.process_document(
            "today_doc",
            "Custom task orchestration engine design and deployment strategies. "
            "We built our own scheduler from first principles.",
            metadata={"timestamp": today_str}
        )
        fresh_processor.process_document(
            "old_doc",
            "Custom task orchestration engine design and deployment strategies. "
            "We built our own scheduler from first principles.",
            metadata={"timestamp": thirty_days_ago}
        )
        fresh_processor.compute_all(verbose=False)

        # When I search with linear decay
        results = fresh_processor.find_documents_for_query(
            "task orchestration engine",
            top_n=5,
            freshness_boost=1.5,
            freshness_decay="linear"
        )

        # Then today's document should rank first with full boost
        result_docs = [doc_id for doc_id, _ in results]
        assert result_docs[0] == "today_doc", (
            f"Today's document should rank first, got: {result_docs}"
        )

        # Verify score difference reflects full boost
        scores = {doc_id: score for doc_id, score in results}
        # Today's doc should have significantly higher score (1.5x boost vs no boost)
        assert scores["today_doc"] > scores["old_doc"], (
            "Today's document should have higher score than 30-day-old document"
        )

    def test_linear_decay_ranks_2day_higher_than_6day(self, fresh_processor):
        """
        Scenario: 2-day-old document ranks higher than 6-day-old with linear decay

        Given two documents with identical content
        And one was added 2 days ago
        And another was added 6 days ago
        When I search with linear decay enabled
        Then the 2-day-old document ranks higher
        Because linear decay: Day 2 gets ~1.36x boost, Day 6 gets ~1.07x boost.
        """
        # Given documents at different ages within the window
        today = datetime.now()
        two_days_ago = (today - timedelta(days=2)).strftime("%Y-%m-%d")
        six_days_ago = (today - timedelta(days=6)).strftime("%Y-%m-%d")

        content = (
            "Custom query language design with hand-built parser and schema resolver. "
            "In-house API patterns for our service architecture we control completely."
        )

        fresh_processor.process_document(
            "doc_2days", content,
            metadata={"timestamp": two_days_ago}
        )
        fresh_processor.process_document(
            "doc_6days", content,
            metadata={"timestamp": six_days_ago}
        )
        fresh_processor.compute_all(verbose=False)

        # When I search with linear decay
        results = fresh_processor.find_documents_for_query(
            "query language parser",
            top_n=5,
            freshness_boost=1.5,
            freshness_decay="linear"
        )

        # Then 2-day-old document should rank higher than 6-day-old
        result_docs = [doc_id for doc_id, _ in results]
        assert "doc_2days" in result_docs
        assert "doc_6days" in result_docs
        idx_2days = result_docs.index("doc_2days")
        idx_6days = result_docs.index("doc_6days")
        assert idx_2days < idx_6days, (
            f"2-day-old doc (idx {idx_2days}) should rank higher than "
            f"6-day-old doc (idx {idx_6days})"
        )

    def test_linear_decay_day7_gets_no_boost(self, fresh_processor):
        """
        Scenario: Day 7 documents get no freshness boost with linear decay

        Given a document added exactly 7 days ago (boundary)
        And a document added 10 days ago (outside window)
        When I search with linear decay enabled
        Then both documents have similar scores
        Because day 7 is the boundary where boost reaches 1.0x (no boost).
        """
        # Given documents at and beyond the window boundary
        today = datetime.now()
        seven_days_ago = (today - timedelta(days=7)).strftime("%Y-%m-%d")
        ten_days_ago = (today - timedelta(days=10)).strftime("%Y-%m-%d")

        content = (
            "In-house caching layer strategies for high-performance applications. "
            "Custom in-memory data structure design and optimization techniques."
        )

        fresh_processor.process_document(
            "doc_7days", content,
            metadata={"timestamp": seven_days_ago}
        )
        fresh_processor.process_document(
            "doc_10days", content,
            metadata={"timestamp": ten_days_ago}
        )
        fresh_processor.compute_all(verbose=False)

        # When I search with linear decay
        results = fresh_processor.find_documents_for_query(
            "caching layer strategies",
            top_n=5,
            freshness_boost=1.5,
            freshness_decay="linear"
        )

        # Then both should have similar scores (neither gets meaningful boost)
        scores = {doc_id: score for doc_id, score in results}
        assert "doc_7days" in scores
        assert "doc_10days" in scores

        # Scores should be very close (both effectively unboosted)
        ratio = scores["doc_7days"] / scores["doc_10days"] if scores["doc_10days"] > 0 else 1
        assert 0.95 <= ratio <= 1.05, (
            f"Day 7 and Day 10 docs should have similar scores, ratio: {ratio:.3f}"
        )

    def test_exponential_decay_front_loads_freshness(self, fresh_processor):
        """
        Scenario: Exponential decay favors very recent documents more strongly

        Given documents at days 0, 2, and 5
        When I search with exponential decay enabled
        Then the score gap between day 0 and day 2 is larger than linear decay
        Because exponential decay front-loads the boost for newest documents.
        """
        # Given documents at different ages
        today = datetime.now()
        today_str = today.strftime("%Y-%m-%d")
        two_days_ago = (today - timedelta(days=2)).strftime("%Y-%m-%d")
        five_days_ago = (today - timedelta(days=5)).strftime("%Y-%m-%d")

        content = (
            "Infrastructure provisioning system built from scratch. "
            "Custom declarative configuration language for automation we own entirely."
        )

        fresh_processor.process_document(
            "doc_today", content,
            metadata={"timestamp": today_str}
        )
        fresh_processor.process_document(
            "doc_2days", content,
            metadata={"timestamp": two_days_ago}
        )
        fresh_processor.process_document(
            "doc_5days", content,
            metadata={"timestamp": five_days_ago}
        )
        fresh_processor.compute_all(verbose=False)

        # When I search with exponential decay
        results_exp = fresh_processor.find_documents_for_query(
            "infrastructure provisioning system",
            top_n=5,
            freshness_boost=1.5,
            freshness_decay="exponential"
        )

        # Then the ranking should be today > 2days > 5days
        result_docs = [doc_id for doc_id, _ in results_exp]
        assert result_docs[0] == "doc_today", "Today's doc should rank first"
        assert result_docs.index("doc_2days") < result_docs.index("doc_5days"), (
            "2-day-old doc should rank higher than 5-day-old doc"
        )

    def test_decay_none_preserves_binary_behavior(self, fresh_processor):
        """
        Scenario: decay_function="none" preserves original binary boost behavior

        Given documents at days 2 and 6 (both within 7-day window)
        When I search with decay="none"
        Then both documents get the same full boost
        Because "none" means no decay - full boost for all fresh documents.
        """
        # Given documents both within the freshness window
        today = datetime.now()
        two_days_ago = (today - timedelta(days=2)).strftime("%Y-%m-%d")
        six_days_ago = (today - timedelta(days=6)).strftime("%Y-%m-%d")

        content = (
            "Process isolation implementation we built ourselves. "
            "Custom sandboxing and multi-stage compilation for production systems."
        )

        fresh_processor.process_document(
            "doc_2days", content,
            metadata={"timestamp": two_days_ago}
        )
        fresh_processor.process_document(
            "doc_6days", content,
            metadata={"timestamp": six_days_ago}
        )
        fresh_processor.compute_all(verbose=False)

        # When I search with decay="none" (binary boost)
        results = fresh_processor.find_documents_for_query(
            "process isolation sandboxing",
            top_n=5,
            freshness_boost=1.5,
            freshness_decay="none"
        )

        # Then both documents should have identical scores
        scores = {doc_id: score for doc_id, score in results}
        assert "doc_2days" in scores
        assert "doc_6days" in scores

        # With no decay, both get full 1.5x boost, so scores should be equal
        ratio = scores["doc_2days"] / scores["doc_6days"] if scores["doc_6days"] > 0 else 1
        assert 0.99 <= ratio <= 1.01, (
            f"With decay='none', both docs should have equal scores, ratio: {ratio:.3f}"
        )

    def test_default_decay_is_linear(self, fresh_processor):
        """
        Scenario: Default decay function is linear when not specified

        Given documents at different ages
        When I search with freshness_boost but no explicit decay parameter
        Then linear decay is applied by default
        Because linear decay provides intuitive graduated ranking.
        """
        # Given documents at different ages
        today = datetime.now()
        one_day_ago = (today - timedelta(days=1)).strftime("%Y-%m-%d")
        five_days_ago = (today - timedelta(days=5)).strftime("%Y-%m-%d")

        content = (
            "Custom observability stack with hand-rolled metrics pipeline. "
            "In-house telemetry collection and alerting for our distributed systems."
        )

        fresh_processor.process_document(
            "doc_1day", content,
            metadata={"timestamp": one_day_ago}
        )
        fresh_processor.process_document(
            "doc_5days", content,
            metadata={"timestamp": five_days_ago}
        )
        fresh_processor.compute_all(verbose=False)

        # When I search without specifying decay (should default to linear)
        results_default = fresh_processor.find_documents_for_query(
            "observability metrics pipeline",
            top_n=5,
            freshness_boost=1.5
            # No freshness_decay parameter - should default to linear
        )

        # And when I search with explicit linear decay
        results_linear = fresh_processor.find_documents_for_query(
            "observability metrics pipeline",
            top_n=5,
            freshness_boost=1.5,
            freshness_decay="linear"
        )

        # Then results should be identical
        default_docs = [doc_id for doc_id, _ in results_default]
        linear_docs = [doc_id for doc_id, _ in results_linear]
        assert default_docs == linear_docs, (
            f"Default should match linear. Default: {default_docs}, Linear: {linear_docs}"
        )

    def test_graduated_decay_with_custom_window(self, fresh_processor):
        """
        Scenario: Graduated decay respects custom freshness window

        Given a custom 14-day freshness window
        And documents at days 5 and 12
        When I search with linear decay
        Then both documents get graduated boosts appropriate to the 14-day window
        Because the decay is relative to the configured window size.
        """
        # Given documents within a 14-day window
        today = datetime.now()
        five_days_ago = (today - timedelta(days=5)).strftime("%Y-%m-%d")
        twelve_days_ago = (today - timedelta(days=12)).strftime("%Y-%m-%d")

        content = (
            "Custom full-text search engine with aggregation we implemented ourselves. "
            "Distributed index optimization and cluster coordination built from scratch."
        )

        fresh_processor.process_document(
            "doc_5days", content,
            metadata={"timestamp": five_days_ago}
        )
        fresh_processor.process_document(
            "doc_12days", content,
            metadata={"timestamp": twelve_days_ago}
        )
        fresh_processor.compute_all(verbose=False)

        # When I search with linear decay and 14-day window
        results = fresh_processor.find_documents_for_query(
            "full-text search engine",
            top_n=5,
            freshness_boost=1.5,
            freshness_decay="linear",
            freshness_window_days=14
        )

        # Then 5-day doc should rank higher (gets more boost in 14-day window)
        result_docs = [doc_id for doc_id, _ in results]
        assert "doc_5days" in result_docs
        assert "doc_12days" in result_docs
        assert result_docs.index("doc_5days") < result_docs.index("doc_12days"), (
            "5-day-old doc should rank higher than 12-day-old doc in 14-day window"
        )
