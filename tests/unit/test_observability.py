"""
Tests for observability module.

Tests timing decorators, metrics collection, and trace context functionality.
"""

import unittest
import time
from cortical import CorticalTextProcessor
from cortical.observability import (
    MetricsCollector,
    TraceContext,
    timed,
    measure_time,
    get_global_metrics,
    enable_global_metrics,
    disable_global_metrics,
    reset_global_metrics
)


class TestMetricsCollector(unittest.TestCase):
    """Tests for MetricsCollector class."""

    def setUp(self):
        self.metrics = MetricsCollector()

    def test_initialization(self):
        """Test MetricsCollector initializes correctly."""
        self.assertTrue(self.metrics.enabled)
        self.assertEqual(len(self.metrics.operations), 0)
        self.assertEqual(len(self.metrics.traces), 0)

    def test_record_timing(self):
        """Test recording timing measurements."""
        self.metrics.record_timing("test_op", 100.0)
        stats = self.metrics.get_operation_stats("test_op")

        self.assertEqual(stats['count'], 1)
        self.assertEqual(stats['total_ms'], 100.0)
        self.assertEqual(stats['avg_ms'], 100.0)
        self.assertEqual(stats['min_ms'], 100.0)
        self.assertEqual(stats['max_ms'], 100.0)

    def test_record_multiple_timings(self):
        """Test recording multiple measurements for same operation."""
        self.metrics.record_timing("test_op", 50.0)
        self.metrics.record_timing("test_op", 150.0)
        self.metrics.record_timing("test_op", 100.0)

        stats = self.metrics.get_operation_stats("test_op")
        self.assertEqual(stats['count'], 3)
        self.assertEqual(stats['total_ms'], 300.0)
        self.assertEqual(stats['avg_ms'], 100.0)
        self.assertEqual(stats['min_ms'], 50.0)
        self.assertEqual(stats['max_ms'], 150.0)

    def test_record_count(self):
        """Test recording count metrics."""
        self.metrics.record_count("cache_hits", 5)
        self.metrics.record_count("cache_hits", 3)

        stats = self.metrics.get_operation_stats("cache_hits")
        self.assertEqual(stats['count'], 8)
        # Count-only metrics should not have timing stats
        self.assertNotIn('total_ms', stats)

    def test_enabled_disabled(self):
        """Test enabling and disabling metrics collection."""
        self.metrics.disable()
        self.assertFalse(self.metrics.enabled)

        # Recording while disabled should be no-op
        self.metrics.record_timing("disabled_op", 100.0)
        self.assertEqual(len(self.metrics.operations), 0)

        # Re-enable
        self.metrics.enable()
        self.assertTrue(self.metrics.enabled)
        self.metrics.record_timing("enabled_op", 50.0)
        self.assertEqual(len(self.metrics.operations), 1)

    def test_reset(self):
        """Test resetting metrics."""
        self.metrics.record_timing("op1", 100.0)
        self.metrics.record_count("counter", 5)

        self.assertEqual(len(self.metrics.operations), 2)

        self.metrics.reset()
        self.assertEqual(len(self.metrics.operations), 0)
        self.assertEqual(len(self.metrics.traces), 0)

    def test_trace_context(self):
        """Test trace context recording."""
        with self.metrics.trace_context("trace-123"):
            self.metrics.record_timing("op1", 50.0, context={'arg': 'value1'})
            self.metrics.record_timing("op2", 75.0, context={'arg': 'value2'})

        trace = self.metrics.get_trace("trace-123")
        self.assertEqual(len(trace), 2)
        self.assertEqual(trace[0][0], "op1")
        self.assertEqual(trace[0][1], 50.0)
        self.assertEqual(trace[0][2], {'arg': 'value1'})

    def test_get_all_stats(self):
        """Test getting all statistics."""
        self.metrics.record_timing("op1", 100.0)
        self.metrics.record_timing("op2", 50.0)
        self.metrics.record_count("counter", 3)

        all_stats = self.metrics.get_all_stats()
        self.assertEqual(len(all_stats), 3)
        self.assertIn("op1", all_stats)
        self.assertIn("op2", all_stats)
        self.assertIn("counter", all_stats)

    def test_get_summary(self):
        """Test getting human-readable summary."""
        self.metrics.record_timing("compute_all", 1234.5)
        self.metrics.record_timing("find_documents", 56.7)
        self.metrics.record_count("cache_hits", 42)

        summary = self.metrics.get_summary()
        self.assertIn("Metrics Summary", summary)
        self.assertIn("compute_all", summary)
        self.assertIn("find_documents", summary)
        self.assertIn("cache_hits", summary)
        self.assertIn("42", summary)  # Count should appear

    def test_empty_summary(self):
        """Test summary when no metrics collected."""
        summary = self.metrics.get_summary()
        self.assertEqual(summary, "No metrics collected.")


class TestTraceContext(unittest.TestCase):
    """Tests for TraceContext class."""

    def test_initialization(self):
        """Test TraceContext initializes correctly."""
        trace = TraceContext("trace-123", metadata={'user': 'test'})
        self.assertEqual(trace.trace_id, "trace-123")
        self.assertEqual(trace.metadata, {'user': 'test'})

    def test_elapsed_time(self):
        """Test elapsed time measurement."""
        trace = TraceContext("trace-123")
        time.sleep(0.01)  # Sleep 10ms
        elapsed = trace.elapsed_ms()
        self.assertGreater(elapsed, 5.0)  # Should be > 5ms


class TestTimedDecorator(unittest.TestCase):
    """Tests for @timed decorator."""

    def test_timed_decorator_with_metrics(self):
        """Test @timed decorator records metrics."""
        metrics = MetricsCollector()

        class MockProcessor:
            def __init__(self):
                self._metrics = metrics

            @timed("test_method")
            def test_method(self):
                time.sleep(0.01)
                return "done"

        processor = MockProcessor()
        result = processor.test_method()

        self.assertEqual(result, "done")
        stats = metrics.get_operation_stats("test_method")
        self.assertEqual(stats['count'], 1)
        self.assertGreater(stats['avg_ms'], 5.0)

    def test_timed_decorator_without_metrics(self):
        """Test @timed decorator when metrics disabled."""
        class MockProcessor:
            def __init__(self):
                self._metrics = MetricsCollector(enabled=False)

            @timed("test_method")
            def test_method(self):
                return "done"

        processor = MockProcessor()
        result = processor.test_method()

        self.assertEqual(result, "done")
        stats = processor._metrics.get_operation_stats("test_method")
        self.assertEqual(stats, {})

    def test_timed_decorator_no_metrics_object(self):
        """Test @timed decorator when no _metrics attribute."""
        class MockProcessor:
            @timed("test_method")
            def test_method(self):
                return "done"

        processor = MockProcessor()
        result = processor.test_method()
        self.assertEqual(result, "done")

    def test_timed_with_include_args(self):
        """Test @timed decorator with include_args."""
        metrics = MetricsCollector()

        class MockProcessor:
            def __init__(self):
                self._metrics = metrics

            @timed("test_method", include_args=True)
            def test_method(self, arg1, kwarg1="default"):
                return "done"

        processor = MockProcessor()
        processor.test_method("value1", kwarg1="value2")

        # Check that context was recorded
        stats = metrics.get_operation_stats("test_method")
        self.assertEqual(stats['count'], 1)


class TestProcessorIntegration(unittest.TestCase):
    """Integration tests with CorticalTextProcessor."""

    def test_processor_with_metrics_enabled(self):
        """Test processor with metrics enabled."""
        processor = CorticalTextProcessor(enable_metrics=True)
        processor.process_document("doc1", "Neural networks process data.")
        processor.compute_all(verbose=False)

        metrics = processor.get_metrics()

        # Check that key operations were timed
        self.assertIn("process_document", metrics)
        self.assertIn("compute_all", metrics)

        # Check process_document was called once
        self.assertEqual(metrics["process_document"]["count"], 1)
        self.assertGreater(metrics["process_document"]["avg_ms"], 0)

    def test_processor_with_metrics_disabled(self):
        """Test processor with metrics disabled (default)."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Neural networks process data.")
        processor.compute_all(verbose=False)

        metrics = processor.get_metrics()

        # No metrics should be collected
        self.assertEqual(metrics, {})

    def test_processor_metrics_summary(self):
        """Test getting metrics summary."""
        processor = CorticalTextProcessor(enable_metrics=True)
        processor.process_document("doc1", "Neural networks process data.")
        processor.process_document("doc2", "Machine learning algorithms.")
        processor.compute_all(verbose=False)

        summary = processor.get_metrics_summary()

        # Check summary contains expected operations
        self.assertIn("process_document", summary)
        self.assertIn("compute_all", summary)
        self.assertIn("Metrics Summary", summary)

    def test_processor_reset_metrics(self):
        """Test resetting processor metrics."""
        processor = CorticalTextProcessor(enable_metrics=True)
        processor.process_document("doc1", "Neural networks process data.")

        metrics_before = processor.get_metrics()
        self.assertGreater(len(metrics_before), 0)

        processor.reset_metrics()
        metrics_after = processor.get_metrics()
        self.assertEqual(len(metrics_after), 0)

    def test_processor_enable_disable_metrics(self):
        """Test enabling and disabling metrics on processor."""
        processor = CorticalTextProcessor(enable_metrics=False)
        processor.process_document("doc1", "Neural networks process data.")

        metrics = processor.get_metrics()
        self.assertEqual(len(metrics), 0)

        # Enable metrics
        processor.enable_metrics()
        processor.process_document("doc2", "Machine learning algorithms.")

        metrics = processor.get_metrics()
        self.assertGreater(len(metrics), 0)

        # Disable again
        processor.disable_metrics()
        processor.reset_metrics()
        processor.process_document("doc3", "Deep learning networks.")

        metrics = processor.get_metrics()
        self.assertEqual(len(metrics), 0)

    def test_processor_record_custom_metric(self):
        """Test recording custom metrics."""
        processor = CorticalTextProcessor(enable_metrics=True)

        processor.record_metric("custom_counter", 5)
        processor.record_metric("custom_counter", 3)

        metrics = processor.get_metrics()
        self.assertIn("custom_counter", metrics)
        self.assertEqual(metrics["custom_counter"]["count"], 8)

    def test_compute_all_timing(self):
        """Test that compute_all phases are timed."""
        processor = CorticalTextProcessor(enable_metrics=True)
        processor.process_document("doc1", "Neural networks process data.")
        processor.process_document("doc2", "Machine learning algorithms.")
        processor.compute_all(verbose=False)

        metrics = processor.get_metrics()

        # Check individual computation methods
        self.assertIn("compute_all", metrics)
        self.assertIn("propagate_activation", metrics)
        self.assertIn("compute_importance", metrics)
        self.assertIn("compute_tfidf", metrics)

    def test_query_timing(self):
        """Test that query operations are timed."""
        processor = CorticalTextProcessor(enable_metrics=True)
        processor.process_document("doc1", "Neural networks process data.")
        processor.compute_all(verbose=False)

        processor.find_documents_for_query("neural networks")

        metrics = processor.get_metrics()
        self.assertIn("find_documents_for_query", metrics)
        self.assertGreater(metrics["find_documents_for_query"]["avg_ms"], 0)

    def test_save_timing(self):
        """Test that save operation is timed."""
        import tempfile
        import os

        processor = CorticalTextProcessor(enable_metrics=True)
        processor.process_document("doc1", "Neural networks process data.")
        processor.compute_all(verbose=False)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            temp_path = f.name

        try:
            processor.save(temp_path, verbose=False)

            metrics = processor.get_metrics()
            self.assertIn("save", metrics)
            self.assertGreater(metrics["save"]["avg_ms"], 0)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_cache_hit_metrics(self):
        """Test that cache hits/misses are recorded."""
        processor = CorticalTextProcessor(enable_metrics=True)
        processor.process_document("doc1", "Neural networks process data.")
        processor.compute_all(verbose=False)

        # First call - cache miss
        processor.expand_query_cached("neural")

        # Second call - cache hit
        processor.expand_query_cached("neural")

        metrics = processor.get_metrics()
        self.assertIn("query_cache_hits", metrics)
        self.assertIn("query_cache_misses", metrics)
        self.assertEqual(metrics["query_cache_hits"]["count"], 1)
        self.assertEqual(metrics["query_cache_misses"]["count"], 1)


class TestGlobalMetrics(unittest.TestCase):
    """Tests for global metrics functions."""

    def setUp(self):
        reset_global_metrics()

    def tearDown(self):
        reset_global_metrics()

    def test_global_metrics_singleton(self):
        """Test global metrics collector is a singleton."""
        metrics1 = get_global_metrics()
        metrics2 = get_global_metrics()
        self.assertIs(metrics1, metrics2)

    def test_enable_disable_global(self):
        """Test enabling/disabling global metrics."""
        metrics = get_global_metrics()

        disable_global_metrics()
        self.assertFalse(metrics.enabled)

        enable_global_metrics()
        self.assertTrue(metrics.enabled)

    def test_reset_global(self):
        """Test resetting global metrics."""
        metrics = get_global_metrics()
        enable_global_metrics()

        metrics.record_timing("test_op", 100.0)
        self.assertEqual(len(metrics.operations), 1)

        reset_global_metrics()
        self.assertEqual(len(metrics.operations), 0)


if __name__ == '__main__':
    unittest.main()
