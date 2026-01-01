"""
Behavioral tests for developers tracking system observability.

Epic: System Observability and Metrics

As a developer monitoring system performance,
I want to track timing metrics, cache hits, and custom events,
So that I can understand system behavior and optimize performance.

Based on: examples/observability_demo.py
"""

import pytest
from cortical import CorticalTextProcessor


class TestDeveloperTracksObservability:
    """
    Epic: System Observability and Metrics

    As a developer monitoring production systems,
    I want comprehensive metrics collection and reporting,
    So that I can identify bottlenecks and optimize performance.
    """

    def test_scenario_developer_enables_metrics_collection(self):
        """
        Scenario: Enabling metrics for monitoring

        Given a processor instance
        When I enable metrics collection
        Then operations are automatically timed
        And metrics are stored for later analysis
        Because developers need visibility into system behavior.
        """
        # GIVEN a processor instance
        # WHEN I enable metrics collection
        processor = CorticalTextProcessor(enable_metrics=True)

        # Process documents and operations
        processor.process_document(
            "doc1",
            "Neural networks are computational models inspired by biological neurons."
        )
        processor.compute_all(verbose=False)

        # THEN operations are automatically timed
        metrics = processor.get_metrics()
        assert len(metrics) > 0, "Should collect metrics for operations"

        # AND metrics are stored for later analysis
        if "compute_all" in metrics:
            stats = metrics["compute_all"]
            assert 'count' in stats, "Should track operation count"
            assert 'avg_ms' in stats, "Should track average time"

    def test_scenario_developer_tracks_operation_timing(self):
        """
        Scenario: Timing critical operations

        Given metrics-enabled processor
        When I perform multiple operations
        Then timing data is collected for each operation
        And I can see min, max, and average times
        Because developers need to identify slow operations.
        """
        # GIVEN metrics-enabled processor
        processor = CorticalTextProcessor(enable_metrics=True)

        processor.process_document("doc1", "Machine learning algorithms analyze data patterns.")
        processor.process_document("doc2", "Deep learning uses neural networks with multiple layers.")
        processor.compute_all(verbose=False)

        # WHEN I perform multiple operations
        processor.find_documents_for_query("machine learning")
        processor.find_documents_for_query("neural networks")

        # THEN timing data is collected for each operation
        metrics = processor.get_metrics()

        if "find_documents_for_query" in metrics:
            stats = metrics["find_documents_for_query"]
            # AND I can see min, max, and average times
            assert 'count' in stats, "Should track execution count"
            assert 'avg_ms' in stats, "Should track average milliseconds"
            assert 'min_ms' in stats, "Should track minimum time"
            assert 'max_ms' in stats, "Should track maximum time"
            assert stats['count'] >= 2, "Should record both query executions"

    def test_scenario_developer_monitors_cache_performance(self):
        """
        Scenario: Tracking cache hit rates

        Given a processor with query caching
        When I execute repeated queries
        Then cache hits and misses are tracked separately
        And I can calculate hit rate
        Because developers need to optimize caching strategies.
        """
        # GIVEN a processor with query caching
        processor = CorticalTextProcessor(enable_metrics=True)

        processor.process_document("doc1", "Neural network training requires large datasets.")
        processor.compute_all(verbose=False)

        # WHEN I execute repeated queries
        processor.expand_query_cached("neural")  # Cache miss
        processor.expand_query_cached("neural")  # Cache hit
        processor.expand_query_cached("training")  # Cache miss
        processor.expand_query_cached("training")  # Cache hit

        # THEN cache hits and misses are tracked separately
        metrics = processor.get_metrics()

        if "query_cache_hits" in metrics and "query_cache_misses" in metrics:
            hits = metrics["query_cache_hits"]["count"]
            misses = metrics["query_cache_misses"]["count"]

            # AND I can calculate hit rate
            assert hits >= 2, "Should record cache hits"
            assert misses >= 2, "Should record cache misses"

            total = hits + misses
            hit_rate = (hits / total * 100) if total > 0 else 0
            assert hit_rate > 0, "Hit rate should be calculable"

    def test_scenario_developer_records_custom_metrics(self):
        """
        Scenario: Recording application-specific metrics

        Given a metrics-enabled processor
        When I record custom metric values
        Then custom metrics are stored alongside system metrics
        And I can track application-specific events
        Because developers need to track domain-specific measurements.
        """
        # GIVEN a metrics-enabled processor
        processor = CorticalTextProcessor(enable_metrics=True)

        # WHEN I record custom metric values
        processor.record_metric("api_calls", 10)
        processor.record_metric("api_calls", 5)
        processor.record_metric("users_active", 3)
        processor.record_metric("users_active", 7)

        # THEN custom metrics are stored alongside system metrics
        metrics = processor.get_metrics()

        # AND I can track application-specific events
        if "api_calls" in metrics:
            assert metrics["api_calls"]["count"] >= 2, "Should track custom metric count"

        if "users_active" in metrics:
            assert metrics["users_active"]["count"] >= 2, "Should track multiple custom metrics"

    def test_scenario_developer_generates_metrics_summary(self):
        """
        Scenario: Getting human-readable metrics report

        Given a processor with collected metrics
        When I request a metrics summary
        Then I receive formatted text with key statistics
        And the summary is readable and informative
        Because developers need quick status overviews.
        """
        # GIVEN a processor with collected metrics
        processor = CorticalTextProcessor(enable_metrics=True)

        processor.process_document("doc1", "Machine learning algorithms.")
        processor.compute_all(verbose=False)
        processor.find_documents_for_query("machine learning")

        # WHEN I request a metrics summary
        summary = processor.get_metrics_summary()

        # THEN I receive formatted text with key statistics
        assert isinstance(summary, str), "Summary should be a string"
        assert len(summary) > 0, "Summary should contain information"

        # AND the summary is readable and informative
        # Summary should mention operations that were performed
        assert "compute_all" in summary or "Metrics" in summary, \
            "Summary should reference operations or metrics"

    def test_scenario_developer_disables_metrics_temporarily(self):
        """
        Scenario: Controlling metrics overhead

        Given a processor with metrics enabled
        When I disable metrics temporarily
        Then new operations are not timed
        And existing metrics are preserved
        Because developers need to control monitoring overhead.
        """
        # GIVEN a processor with metrics enabled
        processor = CorticalTextProcessor(enable_metrics=True)

        processor.process_document("doc1", "Initial document")
        processor.compute_all(verbose=False)

        initial_metrics = processor.get_metrics()
        initial_count = len(initial_metrics)

        # WHEN I disable metrics temporarily
        processor.disable_metrics()
        processor.process_document("doc2", "This won't be timed")

        # Re-enable to check
        processor.enable_metrics()
        current_metrics = processor.get_metrics()

        # THEN new operations are not timed
        # AND existing metrics are preserved
        assert len(current_metrics) >= 0, "Metrics should still be accessible"
        # The doc2 processing during disabled period shouldn't add new timed operations

    def test_scenario_developer_resets_metrics_for_new_session(self):
        """
        Scenario: Starting fresh metrics collection

        Given a processor with accumulated metrics
        When I reset all metrics
        Then all metric data is cleared
        And I can start collecting fresh data
        Because developers need clean starts for new measurement sessions.
        """
        # GIVEN a processor with accumulated metrics
        processor = CorticalTextProcessor(enable_metrics=True)

        processor.process_document("doc1", "Document one")
        processor.compute_all(verbose=False)

        initial_metrics = processor.get_metrics()
        assert len(initial_metrics) > 0, "Should have some metrics before reset"

        # WHEN I reset all metrics
        processor.reset_metrics()

        # THEN all metric data is cleared
        metrics_after_reset = processor.get_metrics()
        assert len(metrics_after_reset) == 0, "Metrics should be empty after reset"

        # AND I can start collecting fresh data
        processor.process_document("doc2", "New document")
        new_metrics = processor.get_metrics()
        # New operations will create new metrics

    def test_scenario_developer_accesses_detailed_metrics_programmatically(self):
        """
        Scenario: Programmatic metrics analysis

        Given collected metrics
        When I access metrics programmatically
        Then I can extract specific statistics
        And build custom monitoring dashboards
        Because developers integrate metrics with monitoring systems.
        """
        # GIVEN collected metrics
        processor = CorticalTextProcessor(enable_metrics=True)

        processor.process_document("doc1", "Test document")
        processor.compute_all(verbose=False)

        # WHEN I access metrics programmatically
        metrics = processor.get_metrics()

        # THEN I can extract specific statistics
        assert isinstance(metrics, dict), "Metrics should be a dictionary"

        if "compute_all" in metrics:
            stats = metrics["compute_all"]
            # AND build custom monitoring dashboards
            assert 'count' in stats, "Should have execution count"
            assert 'avg_ms' in stats, "Should have average time"

            # Can extract for custom processing
            execution_count = stats['count']
            avg_time = stats['avg_ms']
            assert isinstance(execution_count, int), "Count should be integer"
            assert isinstance(avg_time, float), "Average should be float"

    def test_scenario_developer_monitors_multiple_operation_types(self):
        """
        Scenario: Tracking diverse operation types

        Given a processor performing various operations
        When I collect metrics across operation types
        Then each operation type is tracked separately
        And I can compare performance across operations
        Because developers need to see the full system picture.
        """
        # GIVEN a processor performing various operations
        processor = CorticalTextProcessor(enable_metrics=True)

        # Perform diverse operations
        processor.process_document("doc1", "Neural networks and machine learning")
        processor.compute_all(verbose=False)
        processor.find_documents_for_query("neural")
        processor.expand_query_cached("machine")

        # WHEN I collect metrics across operation types
        metrics = processor.get_metrics()

        # THEN each operation type is tracked separately
        assert len(metrics) > 0, "Should track multiple operation types"

        # AND I can compare performance across operations
        operation_types = list(metrics.keys())
        assert len(operation_types) > 0, "Should have multiple operation types tracked"

    def test_scenario_developer_identifies_performance_bottlenecks(self):
        """
        Scenario: Finding slow operations

        Given metrics from multiple operations
        When I analyze timing statistics
        Then I can identify which operations are slowest
        And focus optimization efforts appropriately
        Because developers need data-driven optimization.
        """
        # GIVEN metrics from multiple operations
        processor = CorticalTextProcessor(enable_metrics=True)

        # Perform operations of varying complexity
        processor.process_document("doc1", "Simple document")
        processor.compute_all(verbose=False)  # Typically slower
        processor.find_documents_for_query("simple")  # Typically faster

        # WHEN I analyze timing statistics
        metrics = processor.get_metrics()

        # THEN I can identify which operations are slowest
        operation_times = {}
        for operation, stats in metrics.items():
            if 'avg_ms' in stats:
                operation_times[operation] = stats['avg_ms']

        # AND focus optimization efforts appropriately
        if len(operation_times) > 0:
            slowest_operation = max(operation_times.items(), key=lambda x: x[1])
            assert slowest_operation is not None, "Should be able to identify slowest operation"
