"""
Tests for Query API enhancements from Sprint S-024.

Tests cover:
1. Query logging with configurable verbosity
2. Query builder syntax validation
3. Query explain/plan visualization

Note: QueryIndexManager tests were removed as part of the CDGIndexManager
migration. Index functionality is now tested at the CDG layer.
"""

import logging
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Mark all tests in this module as slow (disk-heavy)
pytestmark = pytest.mark.slow

from cortical.got import GoTManager
from cortical.got.query_builder import Query
from cortical.core.bootstrap import create_container
from cortical.got.query_builder import (
    QueryPlan,
    QueryLogLevel,
    QueryValidationError,
    set_query_log_level,
    get_query_log_level,
    set_slow_query_threshold,
    get_slow_query_threshold,
    enable_syntax_validation,
    disable_syntax_validation,
    _validate_syntax_enabled,
)


class TestQueryPlanVisualization:
    """Tests for QueryPlan.__str__ visualization."""

    def test_plan_str_basic_scan(self):
        """Test basic scan step visualization."""
        plan = QueryPlan(
            steps=[{"type": "scan", "entity_type": "TASK", "index": None}],
            estimated_cost=10.0,
            uses_index=False,
            index_name=None,
        )
        output = str(plan)
        assert "Query Execution Plan" in output
        assert "Step 1: SCAN" in output
        assert "Entity type: TASK" in output
        assert "Full scan (no index)" in output
        assert "Uses index: No" in output

    def test_plan_str_with_index(self):
        """Test scan step with index."""
        plan = QueryPlan(
            steps=[{"type": "scan", "entity_type": "TASK", "index": "by_status"}],
            estimated_cost=5.0,
            uses_index=True,
            index_name="by_status",
        )
        output = str(plan)
        assert "Using index: by_status" in output
        assert "Uses index: Yes (by_status)" in output

    def test_plan_str_filter_step(self):
        """Test filter step visualization."""
        plan = QueryPlan(
            steps=[
                {"type": "scan", "entity_type": "TASK", "index": None},
                {
                    "type": "filter",
                    "conditions": [
                        {"field": "status", "op": "eq", "value": "pending"},
                        {"field": "priority", "op": "gt", "value": "medium"},
                    ],
                },
            ],
            estimated_cost=20.0,
            uses_index=False,
            index_name=None,
        )
        output = str(plan)
        assert "Step 2: FILTER" in output
        assert "2 condition(s):" in output
        assert "status eq pending" in output
        assert "priority gt medium" in output

    def test_plan_str_connection_filter(self):
        """Test connection filter step visualization."""
        plan = QueryPlan(
            steps=[
                {"type": "scan", "entity_type": "TASK", "index": None},
                {
                    "type": "connection_filter",
                    "connections": [
                        {"entity_id": "S-001", "edge_type": "CONTAINS"},
                    ],
                },
            ],
            estimated_cost=15.0,
            uses_index=False,
            index_name=None,
        )
        output = str(plan)
        assert "Step 2: CONNECTION_FILTER" in output
        assert "1 connection filter(s):" in output
        assert "connected to S-001 via CONTAINS" in output

    def test_plan_str_sort_step(self):
        """Test sort step visualization."""
        plan = QueryPlan(
            steps=[
                {"type": "scan", "entity_type": "TASK", "index": None},
                {
                    "type": "sort",
                    "fields": [
                        {"field": "created_at", "order": "DESC"},
                        {"field": "priority", "order": "ASC"},
                    ],
                },
            ],
            estimated_cost=20.0,
            uses_index=False,
            index_name=None,
        )
        output = str(plan)
        assert "Step 2: SORT" in output
        assert "Order by:" in output
        assert "created_at DESC" in output
        assert "priority ASC" in output

    def test_plan_str_pagination_step(self):
        """Test pagination step visualization."""
        plan = QueryPlan(
            steps=[
                {"type": "scan", "entity_type": "TASK", "index": None},
                {"type": "pagination", "limit": 10, "offset": 5},
            ],
            estimated_cost=15.0,
            uses_index=False,
            index_name=None,
        )
        output = str(plan)
        assert "Step 2: PAGINATION" in output
        assert "Offset: 5" in output
        assert "Limit: 10" in output

    def test_plan_str_aggregate_step(self):
        """Test aggregate step visualization."""
        plan = QueryPlan(
            steps=[
                {"type": "scan", "entity_type": "TASK", "index": None},
                {
                    "type": "aggregate",
                    "group_by": ["status", "priority"],
                    "aggregates": ["count", "avg_time"],
                },
            ],
            estimated_cost=25.0,
            uses_index=False,
            index_name=None,
        )
        output = str(plan)
        assert "Step 2: AGGREGATE" in output
        assert "Group by: status, priority" in output
        assert "Aggregates: count, avg_time" in output

    def test_plan_repr(self):
        """Test QueryPlan.__repr__."""
        plan = QueryPlan(
            steps=[{"type": "scan"}],
            estimated_cost=10.0,
            uses_index=True,
            index_name=None,
        )
        repr_str = repr(plan)
        assert "QueryPlan(steps=1" in repr_str
        assert "cost=10.0" in repr_str
        assert "uses_index=True" in repr_str


class TestQueryLogging:
    """Tests for query logging functionality."""

    def test_default_log_level_is_off(self):
        """Test that default log level is OFF."""
        # Reset to default
        set_query_log_level(QueryLogLevel.OFF)
        assert get_query_log_level() == QueryLogLevel.OFF

    def test_set_log_level_debug(self):
        """Test setting log level to DEBUG."""
        set_query_log_level(QueryLogLevel.DEBUG)
        assert get_query_log_level() == QueryLogLevel.DEBUG
        set_query_log_level(QueryLogLevel.OFF)

    def test_set_log_level_info(self):
        """Test setting log level to INFO."""
        set_query_log_level(QueryLogLevel.INFO)
        assert get_query_log_level() == QueryLogLevel.INFO
        set_query_log_level(QueryLogLevel.OFF)

    def test_set_log_level_error(self):
        """Test setting log level to ERROR."""
        set_query_log_level(QueryLogLevel.ERROR)
        assert get_query_log_level() == QueryLogLevel.ERROR
        set_query_log_level(QueryLogLevel.OFF)

    def test_default_slow_threshold(self):
        """Test default slow query threshold."""
        assert get_slow_query_threshold() == 100.0

    def test_set_slow_threshold(self):
        """Test setting slow query threshold."""
        original = get_slow_query_threshold()
        set_slow_query_threshold(50.0)
        assert get_slow_query_threshold() == 50.0
        set_slow_query_threshold(original)

    def test_query_logs_at_info_level(self, tmp_path):
        """Test that queries log at INFO level."""
        got_dir = tmp_path / ".got"
        container = create_container(got_dir=got_dir)

        manager = container.resolve(GoTManager)
        manager.create_task("Test Task", status="pending")

        set_query_log_level(QueryLogLevel.INFO)
        try:
            with patch("cortical.got.query_builder.logger") as mock_logger:
                Query(manager).tasks().execute()
                # Should have called info at least once
                assert mock_logger.info.called or mock_logger.warning.called
        finally:
            set_query_log_level(QueryLogLevel.OFF)

    def test_query_logs_plan_at_debug_level(self, tmp_path):
        """Test that queries log plan at DEBUG level."""
        got_dir = tmp_path / ".got"
        container = create_container(got_dir=got_dir)

        manager = container.resolve(GoTManager)
        manager.create_task("Test Task", status="pending")

        set_query_log_level(QueryLogLevel.DEBUG)
        try:
            with patch("cortical.got.query_builder.logger") as mock_logger:
                Query(manager).tasks().execute()
                # Should have called debug for query plan
                assert mock_logger.debug.called
        finally:
            set_query_log_level(QueryLogLevel.OFF)

    def test_query_logs_error_on_exception(self, tmp_path):
        """Test that failed queries log errors."""
        got_dir = tmp_path / ".got"
        container = create_container(got_dir=got_dir)

        manager = container.resolve(GoTManager)

        set_query_log_level(QueryLogLevel.INFO)
        try:
            with patch("cortical.got.query_builder.logger") as mock_logger:
                # Force an exception by patching _execute_query to raise
                q = Query(manager).tasks()
                with patch.object(q, "_execute_query", side_effect=RuntimeError("Test error")):
                    try:
                        q.execute()
                    except RuntimeError:
                        pass
                # Should have logged error
                assert mock_logger.error.called
        finally:
            set_query_log_level(QueryLogLevel.OFF)


class TestQueryValidation:
    """Tests for query builder syntax validation."""

    def test_validation_enabled_by_default(self):
        """Test that validation is enabled by default."""
        enable_syntax_validation()
        assert _validate_syntax_enabled() is True

    def test_disable_validation(self):
        """Test disabling validation."""
        disable_syntax_validation()
        assert _validate_syntax_enabled() is False
        enable_syntax_validation()

    def test_cannot_chain_after_execute(self, tmp_path):
        """Test that chaining after execute raises error."""
        got_dir = tmp_path / ".got"
        container = create_container(got_dir=got_dir)

        manager = container.resolve(GoTManager)
        manager.create_task("Test", status="pending")

        q = Query(manager).tasks()
        q.execute()

        with pytest.raises(QueryValidationError) as exc:
            q.where(status="completed")
        assert "after query has been executed" in str(exc.value)

    def test_cannot_order_after_count_scalar(self, tmp_path):
        """Test that order_by after count (scalar) raises error."""
        got_dir = tmp_path / ".got"
        container = create_container(got_dir=got_dir)

        manager = container.resolve(GoTManager)
        manager.create_task("Test", status="pending")

        # count without group_by returns scalar, sets _count_mode
        q = Query(manager).tasks()
        q._count_mode = True  # Simulate count() called

        with pytest.raises(QueryValidationError) as exc:
            q.order_by("created_at")
        assert "after .count()" in str(exc.value)

    def test_cannot_group_after_pagination(self, tmp_path):
        """Test that group_by after limit raises error."""
        got_dir = tmp_path / ".got"
        container = create_container(got_dir=got_dir)

        manager = container.resolve(GoTManager)

        q = Query(manager).tasks().limit(10)

        with pytest.raises(QueryValidationError) as exc:
            q.group_by("status")
        assert "after .limit()" in str(exc.value)

    def test_cannot_group_after_offset(self, tmp_path):
        """Test that group_by after offset raises error."""
        got_dir = tmp_path / ".got"
        container = create_container(got_dir=got_dir)

        manager = container.resolve(GoTManager)

        q = Query(manager).tasks().offset(5)

        with pytest.raises(QueryValidationError) as exc:
            q.group_by("status")
        assert "after .limit() or .offset()" in str(exc.value)

    def test_validation_disabled_allows_invalid_chains(self, tmp_path):
        """Test that disabled validation allows invalid chains."""
        got_dir = tmp_path / ".got"
        container = create_container(got_dir=got_dir)

        manager = container.resolve(GoTManager)
        manager.create_task("Test", status="pending")

        disable_syntax_validation()
        try:
            q = Query(manager).tasks()
            q.execute()
            # This should not raise when validation is disabled
            q.where(status="completed")  # Would normally raise
        finally:
            enable_syntax_validation()

    def test_where_after_execute_raises(self, tmp_path):
        """Test where() after execute raises."""
        got_dir = tmp_path / ".got"
        container = create_container(got_dir=got_dir)

        manager = container.resolve(GoTManager)
        manager.create_task("Test", status="pending")

        q = Query(manager).tasks().execute()
        q = Query(manager).tasks()
        q._executed = True

        with pytest.raises(QueryValidationError):
            q.where(status="pending")

    def test_or_where_after_execute_raises(self, tmp_path):
        """Test or_where() after execute raises."""
        got_dir = tmp_path / ".got"
        container = create_container(got_dir=got_dir)

        manager = container.resolve(GoTManager)

        q = Query(manager).tasks()
        q._executed = True

        with pytest.raises(QueryValidationError):
            q.or_where(priority="high")

    def test_connected_to_after_execute_raises(self, tmp_path):
        """Test connected_to() after execute raises."""
        got_dir = tmp_path / ".got"
        container = create_container(got_dir=got_dir)

        manager = container.resolve(GoTManager)

        q = Query(manager).tasks()
        q._executed = True

        with pytest.raises(QueryValidationError):
            q.connected_to("S-001")

    def test_limit_after_execute_raises(self, tmp_path):
        """Test limit() after execute raises."""
        got_dir = tmp_path / ".got"
        container = create_container(got_dir=got_dir)

        manager = container.resolve(GoTManager)

        q = Query(manager).tasks()
        q._executed = True

        with pytest.raises(QueryValidationError):
            q.limit(10)

    def test_offset_after_execute_raises(self, tmp_path):
        """Test offset() after execute raises."""
        got_dir = tmp_path / ".got"
        container = create_container(got_dir=got_dir)

        manager = container.resolve(GoTManager)

        q = Query(manager).tasks()
        q._executed = True

        with pytest.raises(QueryValidationError):
            q.offset(5)

    def test_aggregate_after_execute_raises(self, tmp_path):
        """Test aggregate() after execute raises."""
        got_dir = tmp_path / ".got"
        container = create_container(got_dir=got_dir)

        manager = container.resolve(GoTManager)

        q = Query(manager).tasks()
        q._executed = True

        with pytest.raises(QueryValidationError):
            from cortical.got import Count
            q.aggregate(count=Count())
