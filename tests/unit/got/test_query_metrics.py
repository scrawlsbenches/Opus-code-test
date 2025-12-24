"""
Tests for QueryMetrics in the GoT Query API.

Validates:
- Metrics are collected correctly
- Timing is accurate
- Entity counts are tracked
- Summary is human-readable
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from cortical.got import GoTManager
from cortical.got.query_builder import Query, QueryMetrics, enable_query_metrics, disable_query_metrics, get_query_metrics


class TestQueryMetrics:
    """Test QueryMetrics functionality."""

    @pytest.fixture
    def manager_with_tasks(self):
        """Create a manager with some tasks."""
        temp_dir = tempfile.mkdtemp()
        got_dir = Path(temp_dir) / ".got"
        manager = GoTManager(got_dir)

        # Create tasks
        for i in range(10):
            priority = ["high", "medium", "low"][i % 3]
            manager.create_task(f"Task {i}", priority=priority)

        yield manager
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_metrics_disabled_by_default(self, manager_with_tasks):
        """Module-level metrics should be disabled by default."""
        metrics = get_query_metrics()
        assert not metrics.enabled

    def test_metrics_collect_timing(self, manager_with_tasks):
        """Metrics should collect query timing."""
        metrics = QueryMetrics(enabled=True)

        # Run a query with metrics
        Query(manager_with_tasks, metrics=metrics).tasks().execute()

        stats = metrics.get_stats()
        assert stats['total_queries'] == 1
        assert stats['avg_time_ms'] > 0

    def test_metrics_collect_entity_count(self, manager_with_tasks):
        """Metrics should collect entity counts."""
        metrics = QueryMetrics(enabled=True)

        Query(manager_with_tasks, metrics=metrics).tasks().execute()

        stats = metrics.get_stats()
        assert stats['total_entities'] == 10
        assert stats['avg_entities_per_query'] == 10.0

    def test_metrics_track_query_types(self, manager_with_tasks):
        """Metrics should track queries by entity type."""
        metrics = QueryMetrics(enabled=True)

        Query(manager_with_tasks, metrics=metrics).tasks().execute()
        Query(manager_with_tasks, metrics=metrics).tasks().execute()

        stats = metrics.get_stats()
        assert stats['queries_by_type'] == {'TASK': 2}

    def test_metrics_multiple_queries(self, manager_with_tasks):
        """Metrics should aggregate across multiple queries."""
        metrics = QueryMetrics(enabled=True)

        # Run multiple queries
        Query(manager_with_tasks, metrics=metrics).tasks().execute()
        Query(manager_with_tasks, metrics=metrics).tasks().where(priority="high").execute()
        Query(manager_with_tasks, metrics=metrics).tasks().limit(5).execute()

        stats = metrics.get_stats()
        assert stats['total_queries'] == 3

    def test_metrics_timing_range(self, manager_with_tasks):
        """Metrics should track min/max timing."""
        metrics = QueryMetrics(enabled=True)

        # Run multiple queries
        for _ in range(5):
            Query(manager_with_tasks, metrics=metrics).tasks().execute()

        stats = metrics.get_stats()
        assert stats['min_time_ms'] <= stats['avg_time_ms'] <= stats['max_time_ms']

    def test_metrics_reset(self, manager_with_tasks):
        """Metrics should be resettable."""
        metrics = QueryMetrics(enabled=True)

        Query(manager_with_tasks, metrics=metrics).tasks().execute()
        assert metrics.get_stats()['total_queries'] == 1

        metrics.reset()
        assert metrics.get_stats()['total_queries'] == 0

    def test_metrics_summary(self, manager_with_tasks):
        """Summary should be human-readable."""
        metrics = QueryMetrics(enabled=True)

        Query(manager_with_tasks, metrics=metrics).tasks().execute()

        summary = metrics.summary()
        assert "GoT Query API Metrics" in summary
        assert "Total queries: 1" in summary
        assert "TASK: 1" in summary

    def test_metrics_disabled(self, manager_with_tasks):
        """Disabled metrics should not collect data."""
        metrics = QueryMetrics(enabled=False)

        Query(manager_with_tasks, metrics=metrics).tasks().execute()

        stats = metrics.get_stats()
        assert stats['total_queries'] == 0

    def test_enable_disable_module_metrics(self, manager_with_tasks):
        """Module-level metrics can be enabled/disabled."""
        disable_query_metrics()
        metrics = get_query_metrics()
        assert not metrics.enabled

        enable_query_metrics()
        assert metrics.enabled

        disable_query_metrics()
        assert not metrics.enabled


class TestQueryMetricsWithFilters:
    """Test metrics with filtered queries."""

    @pytest.fixture
    def manager_with_tasks(self):
        """Create a manager with tasks of different priorities."""
        temp_dir = tempfile.mkdtemp()
        got_dir = Path(temp_dir) / ".got"
        manager = GoTManager(got_dir)

        for i in range(20):
            priority = ["high", "medium", "low"][i % 3]
            manager.create_task(f"Task {i}", priority=priority)

        yield manager
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_filtered_query_counts_correctly(self, manager_with_tasks):
        """Filtered queries should count only matching entities."""
        metrics = QueryMetrics(enabled=True)

        # Query only high priority (7 out of 20)
        results = Query(manager_with_tasks, metrics=metrics).tasks().where(priority="high").execute()

        stats = metrics.get_stats()
        assert stats['total_entities'] == len(results)
        assert stats['total_entities'] < 20  # Should be filtered

    def test_limited_query_counts_correctly(self, manager_with_tasks):
        """Limited queries should count only returned entities."""
        metrics = QueryMetrics(enabled=True)

        Query(manager_with_tasks, metrics=metrics).tasks().limit(5).execute()

        stats = metrics.get_stats()
        assert stats['total_entities'] == 5
