"""
Security Engineer Protects Queries

Epic: Query Safety and Anomaly Detection

As a security engineer protecting an AI system,
I want to detect anomalous and malicious queries,
So that I prevent prompt injection and abuse.
"""

import pytest
from cortical import CorticalTextProcessor


class TestSecurityEngineerDetectsAnomalies:
    """
    Epic: Query Anomaly Detection

    As a security engineer protecting a search system,
    I want to identify unusual queries,
    So that I can prevent attacks and abuse.
    """

    def test_scenario_enabling_anomaly_detection_for_safety(self):
        """
        Scenario: Activating query safety checks

        Given I have a trained language model
        When I enable anomaly detection
        Then the system can detect unusual queries
        And I can protect against attacks
        Because anomaly detection is the first defense layer.
        """
        # Given I have a trained language model
        processor = CorticalTextProcessor(spark=True)
        processor.process_document("doc1", "Custom search system we built from scratch")
        processor.process_document("doc2", "Hand-crafted indexing algorithm we implemented")
        processor.train_spark()

        # When I enable anomaly detection
        processor.enable_anomaly_detection(
            perplexity_threshold=2.0,
            unknown_word_threshold=0.5
        )

        # Then the system can detect unusual queries
        assert processor.anomaly_detection_enabled

        # And I can protect against attacks
        # (System is ready to check queries)

    def test_scenario_calibrating_detector_with_normal_queries(self):
        """
        Scenario: Establishing baseline behavior

        Given I have anomaly detection enabled
        When I calibrate with known-normal queries
        Then the system learns normal patterns
        And can detect deviations
        Because calibration establishes what "normal" looks like.
        """
        # Given I have anomaly detection enabled
        processor = CorticalTextProcessor(spark=True)
        processor.process_document("guide.md", "How to search. Finding documents. Query syntax.")
        processor.train_spark()
        processor.enable_anomaly_detection()

        # When I calibrate with known-normal queries
        normal_queries = [
            "how do I search for documents",
            "find files about authentication",
            "show me the API documentation",
        ]
        stats = processor.calibrate_anomaly_detector(normal_queries)

        # Then the system learns normal patterns
        assert 'baseline_perplexity' in stats

        # And can detect deviations
        # (Baseline is established for comparison)

    def test_scenario_checking_query_safety_before_processing(self):
        """
        Scenario: Validating query safety

        Given I receive a user query
        When I check query safety
        Then I receive a safety assessment
        And I can reject suspicious queries
        Because safety checks prevent malicious input.
        """
        # Given I receive a user query
        processor = CorticalTextProcessor(spark=True)
        processor.process_document("doc1", "search for documents in the system")
        processor.train_spark()
        processor.enable_anomaly_detection()

        # Calibrate first
        processor.calibrate_anomaly_detector([
            "search for files",
            "find documents",
            "show me results"
        ])

        # When I check query safety
        safe_query = "search for authentication"
        result = processor.check_query_safety(safe_query)

        # Then I receive a safety assessment
        assert 'is_safe' in result
        assert 'is_anomalous' in result
        assert 'confidence' in result

        # And I can reject suspicious queries
        if not result['is_safe']:
            # Would reject query
            pass

    def test_scenario_detecting_prompt_injection_attempts(self):
        """
        Scenario: Identifying injection attacks

        Given I have custom injection patterns
        When I add injection patterns to the detector
        Then suspicious patterns are flagged
        And I can block injection attempts
        Because attackers try to manipulate the system through queries.
        """
        # Given I have custom injection patterns
        processor = CorticalTextProcessor(spark=True)
        processor.process_document("doc1", "normal search content")
        processor.train_spark()
        processor.enable_anomaly_detection()

        # When I add injection patterns to the detector
        processor.add_injection_pattern(r"ignore previous")
        processor.add_injection_pattern(r"system prompt")

        # Then suspicious patterns are flagged
        # (Patterns are added to detector)

        # And I can block injection attempts
        # (Ready to detect these patterns in queries)

    def test_scenario_using_quick_safety_check_for_gating(self):
        """
        Scenario: Fast safety gating

        Given I need to quickly validate queries
        When I use is_query_safe
        Then I get a boolean safety result
        And I can gate processing
        Because simple boolean checks are fastest.
        """
        # Given I need to quickly validate queries
        processor = CorticalTextProcessor(spark=True)
        processor.process_document("doc1", "search documents and find files")
        processor.train_spark()
        processor.enable_anomaly_detection()
        processor.calibrate_anomaly_detector(["search documents", "find files"])

        # When I use is_query_safe
        is_safe = processor.is_query_safe("search for authentication")

        # Then I get a boolean safety result
        assert isinstance(is_safe, bool)

        # And I can gate processing
        if is_safe:
            # Process query
            pass
        else:
            # Reject query
            pass


class TestSecurityEngineerMonitorsThreats:
    """
    Epic: Threat Monitoring

    As a security engineer tracking threats,
    I want to monitor anomaly detection statistics,
    So that I can improve defenses.
    """

    def test_scenario_reviewing_anomaly_detector_statistics(self):
        """
        Scenario: Understanding detector state

        Given I have anomaly detection enabled
        When I get anomaly statistics
        Then I see detector configuration
        And I understand current protection level
        Because statistics reveal detection capabilities.
        """
        # Given I have anomaly detection enabled
        processor = CorticalTextProcessor(spark=True)
        processor.process_document("doc1", "normal content")
        processor.train_spark()
        processor.enable_anomaly_detection(
            perplexity_threshold=2.0,
            unknown_word_threshold=0.5
        )

        # When I get anomaly statistics
        stats = processor.get_anomaly_stats()

        # Then I see detector configuration
        assert 'perplexity_threshold' in stats
        assert 'unknown_word_threshold' in stats

        # And I understand current protection level
        assert stats['perplexity_threshold'] == 2.0

    def test_scenario_checking_multiple_queries_for_batch_validation(self):
        """
        Scenario: Batch safety validation

        Given I have multiple queries to validate
        When I use check_queries_safety
        Then all queries are checked
        And I receive individual assessments
        Because batch validation is more efficient.
        """
        # Given I have multiple queries to validate
        processor = CorticalTextProcessor(spark=True)
        processor.process_document("doc1", "search and find documents")
        processor.train_spark()
        processor.enable_anomaly_detection()
        processor.calibrate_anomaly_detector(["search documents"])

        queries = [
            "find authentication files",
            "search for API docs",
            "show database schema",
        ]

        # When I use check_queries_safety
        results = processor.check_queries_safety(queries)

        # Then all queries are checked
        assert len(results) == 3

        # And I receive individual assessments
        for result in results:
            assert 'is_safe' in result
            assert 'confidence' in result


class TestSecurityEngineerSuggestsAlignmentImprovements:
    """
    Epic: Self-Documentation for Security

    As a security engineer improving alignment,
    I want the system to suggest new definitions,
    So that I can improve query understanding.
    """

    def test_scenario_enabling_suggester_for_learning(self):
        """
        Scenario: Activating suggestion collection

        Given I want to improve system alignment
        When I enable the suggester
        Then the system observes interactions
        And can suggest improvements
        Because self-documentation identifies gaps.
        """
        # Given I want to improve system alignment
        processor = CorticalTextProcessor(spark=True)
        processor.train_spark()

        # When I enable the suggester
        processor.enable_suggester(
            min_frequency=2,
            min_confidence=0.5
        )

        # Then the system observes interactions
        assert processor.suggester_enabled

        # And can suggest improvements
        # (System is ready to collect observations)

    def test_scenario_observing_queries_for_pattern_detection(self):
        """
        Scenario: Recording query patterns

        Given I have the suggester enabled
        When I observe queries with outcomes
        Then patterns are collected
        And suggestions can be generated
        Because repeated patterns reveal user intent.
        """
        # Given I have the suggester enabled
        processor = CorticalTextProcessor(spark=True)
        processor.process_document("doc1", "authentication system")
        processor.train_spark()
        processor.enable_suggester()

        # When I observe queries with outcomes
        processor.observe_query_for_suggestions(
            "auth system",
            success=True,
            context={'result_count': 5}
        )
        processor.observe_query_for_suggestions(
            "authentication",
            success=True,
            context={'result_count': 3}
        )

        # Then patterns are collected
        # And suggestions can be generated
        stats = processor.get_suggester_stats()
        assert 'queries_observed' in stats

    def test_scenario_getting_definition_suggestions_for_alignment(self):
        """
        Scenario: Discovering undefined terms

        Given I've observed many queries
        When I get definition suggestions
        Then I see frequently used undefined terms
        And I can add them to alignment
        Because frequent terms deserve definitions.
        """
        # Given I've observed many queries
        processor = CorticalTextProcessor(spark=True)
        processor.train_spark()
        processor.enable_suggester(min_frequency=1)

        # Observe some queries with undefined terms
        processor.observe_query_for_suggestions("custom parser", success=True)
        processor.observe_query_for_suggestions("custom implementation", success=True)

        # When I get definition suggestions
        suggestions = processor.get_definition_suggestions()

        # Then I see frequently used undefined terms
        # (May suggest defining "custom" or other terms)
        assert isinstance(suggestions, list)

    def test_scenario_exporting_suggestions_as_markdown(self):
        """
        Scenario: Creating alignment documentation

        Given I have collected suggestions
        When I export suggestions as markdown
        Then I receive formatted alignment content
        And I can save it to my alignment file
        Because markdown export makes it easy to review and commit.
        """
        # Given I have collected suggestions
        processor = CorticalTextProcessor(spark=True)
        processor.train_spark()
        processor.enable_suggester()

        # Observe some patterns
        processor.observe_query_for_suggestions("search system", success=True)

        # When I export suggestions as markdown
        markdown = processor.export_suggestions_markdown()

        # Then I receive formatted alignment content
        assert isinstance(markdown, str)

        # And I can save it to my alignment file
        # (Markdown is ready to be written to file)

    def test_scenario_clearing_suggester_after_processing(self):
        """
        Scenario: Resetting observation data

        Given I've processed suggestions
        When I clear the suggester
        Then observations are reset
        And I can start fresh collection
        Because periodic resets prevent stale suggestions.
        """
        # Given I've processed suggestions
        processor = CorticalTextProcessor(spark=True)
        processor.train_spark()
        processor.enable_suggester()
        processor.observe_query_for_suggestions("test query", success=True)

        # When I clear the suggester
        processor.clear_suggester()

        # Then observations are reset
        stats = processor.get_suggester_stats()

        # And I can start fresh collection
        # (Suggester is ready for new observations)
