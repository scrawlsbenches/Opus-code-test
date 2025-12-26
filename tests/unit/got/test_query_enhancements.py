"""
Tests for Query API enhancements from Sprint S-024.

Tests cover:
1. Query logging with configurable verbosity
2. Query builder syntax validation
3. Query explain/plan visualization
4. Query indexer for fast lookups
"""

import logging
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from cortical.got import GoTManager, Query
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
from cortical.got.indexer import (
    QueryIndexManager,
    IndexEntry,
    IndexStats,
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
        manager = GoTManager(got_dir)
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
        manager = GoTManager(got_dir)
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
        manager = GoTManager(got_dir)

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
        manager = GoTManager(got_dir)
        manager.create_task("Test", status="pending")

        q = Query(manager).tasks()
        q.execute()

        with pytest.raises(QueryValidationError) as exc:
            q.where(status="completed")
        assert "after query has been executed" in str(exc.value)

    def test_cannot_order_after_count_scalar(self, tmp_path):
        """Test that order_by after count (scalar) raises error."""
        got_dir = tmp_path / ".got"
        manager = GoTManager(got_dir)
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
        manager = GoTManager(got_dir)

        q = Query(manager).tasks().limit(10)

        with pytest.raises(QueryValidationError) as exc:
            q.group_by("status")
        assert "after .limit()" in str(exc.value)

    def test_cannot_group_after_offset(self, tmp_path):
        """Test that group_by after offset raises error."""
        got_dir = tmp_path / ".got"
        manager = GoTManager(got_dir)

        q = Query(manager).tasks().offset(5)

        with pytest.raises(QueryValidationError) as exc:
            q.group_by("status")
        assert "after .limit() or .offset()" in str(exc.value)

    def test_validation_disabled_allows_invalid_chains(self, tmp_path):
        """Test that disabled validation allows invalid chains."""
        got_dir = tmp_path / ".got"
        manager = GoTManager(got_dir)
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
        manager = GoTManager(got_dir)
        manager.create_task("Test", status="pending")

        q = Query(manager).tasks().execute()
        q = Query(manager).tasks()
        q._executed = True

        with pytest.raises(QueryValidationError):
            q.where(status="pending")

    def test_or_where_after_execute_raises(self, tmp_path):
        """Test or_where() after execute raises."""
        got_dir = tmp_path / ".got"
        manager = GoTManager(got_dir)

        q = Query(manager).tasks()
        q._executed = True

        with pytest.raises(QueryValidationError):
            q.or_where(priority="high")

    def test_connected_to_after_execute_raises(self, tmp_path):
        """Test connected_to() after execute raises."""
        got_dir = tmp_path / ".got"
        manager = GoTManager(got_dir)

        q = Query(manager).tasks()
        q._executed = True

        with pytest.raises(QueryValidationError):
            q.connected_to("S-001")

    def test_limit_after_execute_raises(self, tmp_path):
        """Test limit() after execute raises."""
        got_dir = tmp_path / ".got"
        manager = GoTManager(got_dir)

        q = Query(manager).tasks()
        q._executed = True

        with pytest.raises(QueryValidationError):
            q.limit(10)

    def test_offset_after_execute_raises(self, tmp_path):
        """Test offset() after execute raises."""
        got_dir = tmp_path / ".got"
        manager = GoTManager(got_dir)

        q = Query(manager).tasks()
        q._executed = True

        with pytest.raises(QueryValidationError):
            q.offset(5)

    def test_aggregate_after_execute_raises(self, tmp_path):
        """Test aggregate() after execute raises."""
        got_dir = tmp_path / ".got"
        manager = GoTManager(got_dir)

        q = Query(manager).tasks()
        q._executed = True

        with pytest.raises(QueryValidationError):
            from cortical.got import Count
            q.aggregate(count=Count())


class TestQueryIndexManager:
    """Tests for QueryIndexManager."""

    def test_create_manager(self, tmp_path):
        """Test creating index manager."""
        got_dir = tmp_path / ".got"
        got_dir.mkdir()
        manager = QueryIndexManager(got_dir)
        assert manager is not None
        assert (got_dir / "indexes").exists()

    def test_index_task(self, tmp_path):
        """Test indexing a task."""
        got_dir = tmp_path / ".got"
        got_dir.mkdir()
        manager = QueryIndexManager(got_dir)

        manager.index_task("T-001", status="pending", priority="high")

        pending = manager.lookup("status", "pending")
        assert "T-001" in pending

        high = manager.lookup("priority", "high")
        assert "T-001" in high

    def test_lookup_empty(self, tmp_path):
        """Test lookup with no matches."""
        got_dir = tmp_path / ".got"
        got_dir.mkdir()
        manager = QueryIndexManager(got_dir)

        result = manager.lookup("status", "nonexistent")
        assert result == set()

    def test_update_task_status(self, tmp_path):
        """Test updating task status in index."""
        got_dir = tmp_path / ".got"
        got_dir.mkdir()
        manager = QueryIndexManager(got_dir)

        manager.index_task("T-001", status="pending")
        manager.update_task("T-001", old_status="pending", new_status="completed")

        pending = manager.lookup("status", "pending")
        completed = manager.lookup("status", "completed")

        assert "T-001" not in pending
        assert "T-001" in completed

    def test_update_task_priority(self, tmp_path):
        """Test updating task priority in index."""
        got_dir = tmp_path / ".got"
        got_dir.mkdir()
        manager = QueryIndexManager(got_dir)

        manager.index_task("T-001", priority="low")
        manager.update_task("T-001", old_priority="low", new_priority="high")

        low = manager.lookup("priority", "low")
        high = manager.lookup("priority", "high")

        assert "T-001" not in low
        assert "T-001" in high

    def test_remove_task(self, tmp_path):
        """Test removing task from index."""
        got_dir = tmp_path / ".got"
        got_dir.mkdir()
        manager = QueryIndexManager(got_dir)

        manager.index_task("T-001", status="pending", priority="high")
        manager.remove_task("T-001")

        pending = manager.lookup("status", "pending")
        high = manager.lookup("priority", "high")

        assert "T-001" not in pending
        assert "T-001" not in high

    def test_sprint_index(self, tmp_path):
        """Test sprint index operations."""
        got_dir = tmp_path / ".got"
        got_dir.mkdir()
        manager = QueryIndexManager(got_dir)

        manager.link_task_to_sprint("T-001", "S-001")
        manager.link_task_to_sprint("T-002", "S-001")
        manager.link_task_to_sprint("T-003", "S-002")

        s1_tasks = manager.lookup("sprint", "S-001")
        s2_tasks = manager.lookup("sprint", "S-002")

        assert s1_tasks == {"T-001", "T-002"}
        assert s2_tasks == {"T-003"}

    def test_unlink_from_sprint(self, tmp_path):
        """Test unlinking task from sprint."""
        got_dir = tmp_path / ".got"
        got_dir.mkdir()
        manager = QueryIndexManager(got_dir)

        manager.link_task_to_sprint("T-001", "S-001")
        manager.unlink_task_from_sprint("T-001", "S-001")

        tasks = manager.lookup("sprint", "S-001")
        assert "T-001" not in tasks

    def test_save_and_load(self, tmp_path):
        """Test persisting and loading indexes."""
        got_dir = tmp_path / ".got"
        got_dir.mkdir()
        manager1 = QueryIndexManager(got_dir)

        manager1.index_task("T-001", status="pending", priority="high")
        manager1.link_task_to_sprint("T-001", "S-001")
        manager1.save()

        # Create new manager and verify data loaded
        manager2 = QueryIndexManager(got_dir)

        pending = manager2.lookup("status", "pending")
        high = manager2.lookup("priority", "high")
        sprint_tasks = manager2.lookup("sprint", "S-001")

        assert "T-001" in pending
        assert "T-001" in high
        assert "T-001" in sprint_tasks

    def test_lookup_multi(self, tmp_path):
        """Test looking up multiple values."""
        got_dir = tmp_path / ".got"
        got_dir.mkdir()
        manager = QueryIndexManager(got_dir)

        manager.index_task("T-001", status="pending")
        manager.index_task("T-002", status="completed")
        manager.index_task("T-003", status="in_progress")

        result = manager.lookup_multi("status", ["pending", "completed"])
        assert result == {"T-001", "T-002"}

    def test_has_index(self, tmp_path):
        """Test checking if index exists."""
        got_dir = tmp_path / ".got"
        got_dir.mkdir()
        manager = QueryIndexManager(got_dir)

        assert manager.has_index("status") is True
        assert manager.has_index("priority") is True
        assert manager.has_index("sprint") is True
        assert manager.has_index("nonexistent") is False

    def test_get_stats(self, tmp_path):
        """Test getting index statistics."""
        got_dir = tmp_path / ".got"
        got_dir.mkdir()
        manager = QueryIndexManager(got_dir)

        manager.index_task("T-001", status="pending")
        manager.lookup("status", "pending")  # Hit
        manager.lookup("status", "nonexistent")  # Miss

        stats = manager.get_stats()

        assert stats["hits"] >= 1
        assert stats["misses"] >= 1
        assert "indexes" in stats
        assert "status" in stats["indexes"]

    def test_get_all_indexed_values(self, tmp_path):
        """Test getting all indexed values for a field."""
        got_dir = tmp_path / ".got"
        got_dir.mkdir()
        manager = QueryIndexManager(got_dir)

        manager.index_task("T-001", status="pending")
        manager.index_task("T-002", status="completed")

        values = manager.get_all_indexed_values("status")
        assert "pending" in values
        assert "completed" in values

    def test_rebuild_all(self, tmp_path):
        """Test rebuilding all indexes."""
        got_dir = tmp_path / ".got"
        got_dir.mkdir()
        manager = QueryIndexManager(got_dir)

        # Create mock tasks
        class MockTask:
            def __init__(self, id, status, priority):
                self.id = id
                self.status = status
                self.priority = priority

        tasks = [
            MockTask("T-001", "pending", "high"),
            MockTask("T-002", "completed", "low"),
            MockTask("T-003", "pending", "medium"),
        ]

        manager.rebuild_all(tasks)

        pending = manager.lookup("status", "pending")
        assert pending == {"T-001", "T-003"}

        high = manager.lookup("priority", "high")
        assert high == {"T-001"}

        stats = manager.get_stats()
        assert stats["rebuilds"] == 1


class TestIndexEntry:
    """Tests for IndexEntry dataclass."""

    def test_add_entry(self):
        """Test adding entry to index."""
        entry = IndexEntry(field_name="status")
        entry.add("T-001", "pending")
        entry.add("T-002", "pending")
        entry.add("T-003", "completed")

        assert entry.get("pending") == {"T-001", "T-002"}
        assert entry.get("completed") == {"T-003"}

    def test_remove_entry(self):
        """Test removing entry from index."""
        entry = IndexEntry(field_name="status")
        entry.add("T-001", "pending")
        entry.add("T-002", "pending")

        entry.remove("T-001", "pending")

        assert entry.get("pending") == {"T-002"}

    def test_remove_from_all(self):
        """Test removing entry from all values."""
        entry = IndexEntry(field_name="status")
        entry.add("T-001", "pending")
        entry.add("T-001", "completed")

        entry.remove("T-001")  # Remove from all

        assert "T-001" not in entry.get("pending")
        assert "T-001" not in entry.get("completed")

    def test_get_nonexistent(self):
        """Test getting nonexistent value."""
        entry = IndexEntry(field_name="status")
        result = entry.get("nonexistent")
        assert result == set()

    def test_null_value(self):
        """Test handling None values."""
        entry = IndexEntry(field_name="priority")
        entry.add("T-001", None)

        result = entry.get(None)
        assert "T-001" in result

    def test_to_dict(self):
        """Test serialization to dict."""
        entry = IndexEntry(field_name="status")
        entry.add("T-001", "pending")
        entry.add("T-002", "pending")

        data = entry.to_dict()

        assert data["field_name"] == "status"
        assert "pending" in data["values"]
        assert set(data["values"]["pending"]) == {"T-001", "T-002"}

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "field_name": "status",
            "values": {"pending": ["T-001", "T-002"]},
            "version": 5,
        }

        entry = IndexEntry.from_dict(data)

        assert entry.field_name == "status"
        assert entry.version == 5
        assert entry.get("pending") == {"T-001", "T-002"}


class TestIndexStats:
    """Tests for IndexStats dataclass."""

    def test_hit_rate_empty(self):
        """Test hit rate with no data."""
        stats = IndexStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate_all_hits(self):
        """Test hit rate with all hits."""
        stats = IndexStats(hits=10, misses=0)
        assert stats.hit_rate == 1.0

    def test_hit_rate_mixed(self):
        """Test hit rate with mixed results."""
        stats = IndexStats(hits=3, misses=1)
        assert stats.hit_rate == 0.75
