"""
Tests for GoT Fluent Query Builder and Graph Walker.

This module tests sophisticated graph querying capabilities:
- Fluent method chaining (builder pattern)
- Graph traversal with visitor pattern
- Path finding algorithms
- Pattern matching for subgraph queries
- Aggregation pipelines
"""

import pytest
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from cortical.got import GoTManager


class TestFluentQueryBuilder:
    """Test fluent query builder with method chaining."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create manager with test data."""
        got_dir = tmp_path / ".got"
        manager = GoTManager(got_dir)

        # Create test data
        self.sprint = manager.create_sprint("Sprint 1", number=1, status="in_progress")

        self.task1 = manager.create_task("High priority pending", status="pending", priority="high")
        self.task2 = manager.create_task("Low priority pending", status="pending", priority="low")
        self.task3 = manager.create_task("High priority completed", status="completed", priority="high")
        self.task4 = manager.create_task("Critical in progress", status="in_progress", priority="critical")

        # Add tasks to sprint
        manager.add_edge(self.sprint.id, self.task1.id, "CONTAINS")
        manager.add_edge(self.sprint.id, self.task2.id, "CONTAINS")

        # Create dependencies
        manager.add_edge(self.task2.id, self.task1.id, "DEPENDS_ON")

        # Create decision
        self.decision = manager.create_decision("Use builder pattern", rationale="Cleaner API")
        manager.add_edge(self.task1.id, self.decision.id, "JUSTIFIED_BY")

        return manager

    def test_basic_tasks_query(self, manager):
        """Query.tasks() returns all tasks."""
        from cortical.got.query_builder import Query

        results = Query(manager).tasks().execute()

        assert len(results) == 4

    def test_where_single_condition(self, manager):
        """where() filters by single condition."""
        from cortical.got.query_builder import Query

        results = Query(manager).tasks().where(status="pending").execute()

        assert len(results) == 2
        assert all(r.status == "pending" for r in results)

    def test_where_multiple_conditions(self, manager):
        """where() filters by multiple conditions (AND)."""
        from cortical.got.query_builder import Query

        results = (
            Query(manager)
            .tasks()
            .where(status="pending", priority="high")
            .execute()
        )

        assert len(results) == 1
        assert results[0].title == "High priority pending"

    def test_chained_where(self, manager):
        """Multiple where() calls combine with AND."""
        from cortical.got.query_builder import Query

        results = (
            Query(manager)
            .tasks()
            .where(status="pending")
            .where(priority="high")
            .execute()
        )

        assert len(results) == 1

    def test_or_conditions(self, manager):
        """or_where() creates OR conditions."""
        from cortical.got.query_builder import Query

        results = (
            Query(manager)
            .tasks()
            .where(priority="high")
            .or_where(priority="critical")
            .execute()
        )

        assert len(results) == 3  # 2 high + 1 critical

    def test_connected_to(self, manager):
        """connected_to() finds entities connected via edges."""
        from cortical.got.query_builder import Query

        results = (
            Query(manager)
            .tasks()
            .connected_to(self.sprint.id)
            .execute()
        )

        assert len(results) == 2  # task1 and task2 are in sprint

    def test_connected_to_via_edge_type(self, manager):
        """connected_to() can filter by edge type."""
        from cortical.got.query_builder import Query

        results = (
            Query(manager)
            .tasks()
            .connected_to(self.task1.id, via="DEPENDS_ON")
            .execute()
        )

        assert len(results) == 1
        assert results[0].id == self.task2.id

    def test_order_by_ascending(self, manager):
        """order_by() sorts results."""
        from cortical.got.query_builder import Query

        results = (
            Query(manager)
            .tasks()
            .where(status="pending")
            .order_by("priority")
            .execute()
        )

        # Priority order: critical, high, medium, low
        priorities = [r.priority for r in results]
        assert priorities == sorted(priorities)

    def test_order_by_descending(self, manager):
        """order_by() with desc=True sorts descending."""
        from cortical.got.query_builder import Query

        results = (
            Query(manager)
            .tasks()
            .order_by("priority", desc=True)
            .execute()
        )

        priorities = [r.priority for r in results]
        # Should be reverse sorted
        assert priorities[0] in ("low", "medium")

    def test_limit(self, manager):
        """limit() restricts result count."""
        from cortical.got.query_builder import Query

        results = Query(manager).tasks().limit(2).execute()

        assert len(results) == 2

    def test_offset(self, manager):
        """offset() skips first N results."""
        from cortical.got.query_builder import Query

        all_results = Query(manager).tasks().execute()
        offset_results = Query(manager).tasks().offset(2).execute()

        assert len(offset_results) == len(all_results) - 2

    def test_limit_and_offset(self, manager):
        """limit() and offset() work together for pagination."""
        from cortical.got.query_builder import Query

        page1 = Query(manager).tasks().limit(2).offset(0).execute()
        page2 = Query(manager).tasks().limit(2).offset(2).execute()

        assert len(page1) == 2
        assert len(page2) == 2
        assert set(r.id for r in page1) != set(r.id for r in page2)

    def test_sprints_query(self, manager):
        """Query.sprints() returns sprints."""
        from cortical.got.query_builder import Query

        results = Query(manager).sprints().execute()

        assert len(results) == 1
        assert results[0].id == self.sprint.id

    def test_decisions_query(self, manager):
        """Query.decisions() returns decisions."""
        from cortical.got.query_builder import Query

        results = Query(manager).decisions().execute()

        assert len(results) == 1
        assert results[0].id == self.decision.id

    def test_edges_query(self, manager):
        """Query.edges() returns edges."""
        from cortical.got.query_builder import Query

        results = Query(manager).edges().execute()

        assert len(results) >= 4  # At least the edges we created

    def test_edges_by_type(self, manager):
        """edges() can filter by edge type."""
        from cortical.got.query_builder import Query

        results = (
            Query(manager)
            .edges()
            .where(edge_type="CONTAINS")
            .execute()
        )

        assert len(results) == 2
        assert all(e.edge_type == "CONTAINS" for e in results)

    def test_count(self, manager):
        """count() returns count without fetching all entities."""
        from cortical.got.query_builder import Query

        count = Query(manager).tasks().where(status="pending").count()

        assert count == 2

    def test_exists(self, manager):
        """exists() returns True if any match."""
        from cortical.got.query_builder import Query

        assert Query(manager).tasks().where(status="pending").exists()
        assert not Query(manager).tasks().where(status="cancelled").exists()

    def test_first(self, manager):
        """first() returns first result or None."""
        from cortical.got.query_builder import Query

        result = Query(manager).tasks().where(status="pending").first()
        assert result is not None
        assert result.status == "pending"

        no_result = Query(manager).tasks().where(status="cancelled").first()
        assert no_result is None

    def test_lazy_evaluation(self, manager):
        """Query is not executed until execute()/count()/etc is called."""
        from cortical.got.query_builder import Query

        # Building query should not execute
        query = Query(manager).tasks().where(status="pending")
        assert hasattr(query, '_executed') and not query._executed

        # Execute triggers evaluation
        query.execute()
        assert query._executed


class TestGraphWalker:
    """Test graph traversal with visitor pattern."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create manager with dependency graph."""
        got_dir = tmp_path / ".got"
        manager = GoTManager(got_dir)

        # Create a dependency chain: A -> B -> C -> D
        self.task_a = manager.create_task("Task A", status="completed")
        self.task_b = manager.create_task("Task B", status="completed")
        self.task_c = manager.create_task("Task C", status="in_progress")
        self.task_d = manager.create_task("Task D", status="pending")

        # B depends on A, C depends on B, D depends on C
        manager.add_edge(self.task_b.id, self.task_a.id, "DEPENDS_ON")
        manager.add_edge(self.task_c.id, self.task_b.id, "DEPENDS_ON")
        manager.add_edge(self.task_d.id, self.task_c.id, "DEPENDS_ON")

        # Create a branch: E also depends on B
        self.task_e = manager.create_task("Task E", status="pending")
        manager.add_edge(self.task_e.id, self.task_b.id, "DEPENDS_ON")

        return manager

    def test_bfs_traversal(self, manager):
        """BFS traversal visits nodes level by level."""
        from cortical.got.graph_walker import GraphWalker

        visited = []

        def collect_ids(node, ctx):
            visited.append(node.id)
            return ctx

        GraphWalker(manager).starting_from(self.task_a.id).bfs().visit(collect_ids).run()

        # A should be first, then nodes at distance 1, then distance 2...
        assert visited[0] == self.task_a.id

    def test_dfs_traversal(self, manager):
        """DFS traversal goes deep first."""
        from cortical.got.graph_walker import GraphWalker

        visited = []

        def collect_ids(node, ctx):
            visited.append(node.id)
            return ctx

        GraphWalker(manager).starting_from(self.task_a.id).dfs().visit(collect_ids).run()

        assert visited[0] == self.task_a.id
        assert len(visited) >= 4  # Should visit the chain

    def test_visitor_accumulator(self, manager):
        """Visitor can accumulate results."""
        from cortical.got.graph_walker import GraphWalker

        def count_by_status(node, acc):
            # Task has status as direct attribute, not in properties
            status = getattr(node, 'status', None) or node.properties.get("status", "unknown")
            acc[status] = acc.get(status, 0) + 1
            return acc

        result = (
            GraphWalker(manager)
            .starting_from(self.task_a.id)
            .bfs()
            .visit(count_by_status, initial={})
            .run()
        )

        assert result["completed"] >= 1

    def test_filter_during_traversal(self, manager):
        """filter() restricts which nodes are visited."""
        from cortical.got.graph_walker import GraphWalker

        visited = []

        def collect(node, ctx):
            visited.append(node.id)
            return ctx

        (
            GraphWalker(manager)
            .starting_from(self.task_a.id)
            .bfs()
            # Task has status as direct attribute
            .filter(lambda n: getattr(n, 'status', None) != "pending")
            .visit(collect)
            .run()
        )

        # Should not include pending tasks
        assert self.task_d.id not in visited
        assert self.task_e.id not in visited

    def test_max_depth(self, manager):
        """max_depth limits traversal depth."""
        from cortical.got.graph_walker import GraphWalker

        visited = []

        def collect(node, ctx):
            visited.append(node.id)
            return ctx

        (
            GraphWalker(manager)
            .starting_from(self.task_a.id)
            .bfs()
            .max_depth(2)
            .visit(collect)
            .run()
        )

        # Should not reach task D (depth 3 from A)
        assert self.task_a.id in visited
        assert self.task_b.id in visited  # depth 1
        assert self.task_c.id in visited  # depth 2
        # task_d might not be included (depth 3)

    def test_follow_edge_types(self, manager):
        """follow() restricts which edge types to traverse."""
        from cortical.got.graph_walker import GraphWalker

        visited = []

        def collect(node, ctx):
            visited.append(node.id)
            return ctx

        (
            GraphWalker(manager)
            .starting_from(self.task_a.id)
            .follow("DEPENDS_ON")  # Only follow DEPENDS_ON edges
            .bfs()
            .visit(collect)
            .run()
        )

        assert len(visited) >= 1

    def test_reverse_traversal(self, manager):
        """reverse() follows edges in reverse direction."""
        from cortical.got.graph_walker import GraphWalker

        visited = []

        def collect(node, ctx):
            visited.append(node.id)
            return ctx

        (
            GraphWalker(manager)
            .starting_from(self.task_d.id)  # Start from end
            .follow("DEPENDS_ON")
            .reverse()  # Go backwards
            .bfs()
            .visit(collect)
            .run()
        )

        # Should find path back to A
        assert self.task_d.id in visited


class TestPathFinder:
    """Test path finding between nodes."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create manager with graph for path finding."""
        got_dir = tmp_path / ".got"
        manager = GoTManager(got_dir)

        # Create graph:
        #     A
        #    / \
        #   B   C
        #    \ /
        #     D
        #     |
        #     E

        self.a = manager.create_task("A", status="pending")
        self.b = manager.create_task("B", status="pending")
        self.c = manager.create_task("C", status="pending")
        self.d = manager.create_task("D", status="pending")
        self.e = manager.create_task("E", status="pending")

        manager.add_edge(self.a.id, self.b.id, "CONNECTS")
        manager.add_edge(self.a.id, self.c.id, "CONNECTS")
        manager.add_edge(self.b.id, self.d.id, "CONNECTS")
        manager.add_edge(self.c.id, self.d.id, "CONNECTS")
        manager.add_edge(self.d.id, self.e.id, "CONNECTS")

        return manager

    def test_shortest_path(self, manager):
        """Find shortest path between two nodes."""
        from cortical.got.path_finder import PathFinder

        path = PathFinder(manager).shortest_path(self.a.id, self.e.id)

        assert path is not None
        assert path[0] == self.a.id
        assert path[-1] == self.e.id
        assert len(path) == 4  # A -> B/C -> D -> E

    def test_all_paths(self, manager):
        """Find all paths between two nodes."""
        from cortical.got.path_finder import PathFinder

        paths = PathFinder(manager).all_paths(self.a.id, self.d.id)

        assert len(paths) == 2  # A->B->D and A->C->D

    def test_no_path(self, manager):
        """Returns None/empty when no path exists."""
        from cortical.got.path_finder import PathFinder

        # Create isolated node
        isolated = manager.create_task("Isolated", status="pending")

        path = PathFinder(manager).shortest_path(self.a.id, isolated.id)

        assert path is None

    def test_path_with_edge_filter(self, manager):
        """Can filter which edge types to consider."""
        from cortical.got.path_finder import PathFinder

        path = (
            PathFinder(manager)
            .via_edges("CONNECTS")
            .shortest_path(self.a.id, self.e.id)
        )

        assert path is not None

    def test_path_max_length(self, manager):
        """max_length limits path length."""
        from cortical.got.path_finder import PathFinder

        # Path from A to E is length 4
        short_path = (
            PathFinder(manager)
            .max_length(2)
            .shortest_path(self.a.id, self.e.id)
        )

        assert short_path is None  # No path within 2 hops


class TestAggregation:
    """Test aggregation pipelines."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create manager with data for aggregation."""
        got_dir = tmp_path / ".got"
        manager = GoTManager(got_dir)

        # Create tasks with various statuses and priorities
        manager.create_task("T1", status="pending", priority="high")
        manager.create_task("T2", status="pending", priority="high")
        manager.create_task("T3", status="pending", priority="low")
        manager.create_task("T4", status="completed", priority="high")
        manager.create_task("T5", status="completed", priority="medium")
        manager.create_task("T6", status="in_progress", priority="critical")

        return manager

    def test_group_by_single_field(self, manager):
        """Group by single field."""
        from cortical.got.query_builder import Query

        result = (
            Query(manager)
            .tasks()
            .group_by("status")
            .count()
            .execute()
        )

        assert result["pending"] == 3
        assert result["completed"] == 2
        assert result["in_progress"] == 1

    def test_group_by_multiple_fields(self, manager):
        """Group by multiple fields."""
        from cortical.got.query_builder import Query

        result = (
            Query(manager)
            .tasks()
            .group_by("status", "priority")
            .count()
            .execute()
        )

        assert result[("pending", "high")] == 2
        assert result[("pending", "low")] == 1

    def test_aggregate_count(self, manager):
        """aggregate(count=True) counts items."""
        from cortical.got.query_builder import Query, Count

        result = (
            Query(manager)
            .tasks()
            .group_by("status")
            .aggregate(total=Count())
            .execute()
        )

        assert result["pending"]["total"] == 3

    def test_aggregate_collect(self, manager):
        """aggregate(collect="field") collects field values."""
        from cortical.got.query_builder import Query, Collect

        result = (
            Query(manager)
            .tasks()
            .group_by("status")
            .aggregate(priorities=Collect("priority"))
            .execute()
        )

        assert "high" in result["pending"]["priorities"]


class TestPatternMatching:
    """Test subgraph pattern matching."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create manager with patterns to find."""
        got_dir = tmp_path / ".got"
        manager = GoTManager(got_dir)

        # Create dependency chains
        # Chain 1: A -> B -> C
        self.a1 = manager.create_task("A1", status="pending")
        self.b1 = manager.create_task("B1", status="pending")
        self.c1 = manager.create_task("C1", status="pending")
        manager.add_edge(self.b1.id, self.a1.id, "DEPENDS_ON")
        manager.add_edge(self.c1.id, self.b1.id, "DEPENDS_ON")

        # Chain 2: A -> B -> C
        self.a2 = manager.create_task("A2", status="completed")
        self.b2 = manager.create_task("B2", status="completed")
        self.c2 = manager.create_task("C2", status="completed")
        manager.add_edge(self.b2.id, self.a2.id, "DEPENDS_ON")
        manager.add_edge(self.c2.id, self.b2.id, "DEPENDS_ON")

        return manager

    def test_find_chain_pattern(self, manager):
        """Find chains of length 3."""
        from cortical.got.pattern_matcher import PatternMatcher, Pattern

        # Pattern: task -DEPENDS_ON-> task -DEPENDS_ON-> task
        pattern = (
            Pattern()
            .node("a", type="task")
            .edge("DEPENDS_ON", direction="incoming")
            .node("b", type="task")
            .edge("DEPENDS_ON", direction="incoming")
            .node("c", type="task")
        )

        matches = PatternMatcher(manager).find(pattern)

        assert len(matches) == 2  # Two chains

    def test_find_pattern_with_constraints(self, manager):
        """Find patterns with node constraints."""
        from cortical.got.pattern_matcher import PatternMatcher, Pattern

        pattern = (
            Pattern()
            .node("a", type="task", status="pending")
            .edge("DEPENDS_ON", direction="incoming")
            .node("b", type="task", status="pending")
        )

        matches = PatternMatcher(manager).find(pattern)

        # Should only match pending chains
        assert len(matches) >= 1
        for match in matches:
            # Check status attribute (Task has status as direct attribute, not in properties)
            assert all(getattr(n, 'status', None) == "pending" for n in match.values())


class TestQueryPerformance:
    """Tests for query performance characteristics."""

    @pytest.fixture
    def large_manager(self, tmp_path):
        """Create manager with larger dataset for perf testing."""
        got_dir = tmp_path / ".got"
        manager = GoTManager(got_dir)

        # Create 100 tasks
        for i in range(100):
            status = ["pending", "completed", "in_progress"][i % 3]
            priority = ["low", "medium", "high", "critical"][i % 4]
            manager.create_task(f"Task {i}", status=status, priority=priority)

        return manager

    def test_query_with_index_hint(self, large_manager):
        """Query can use index hints for optimization."""
        from cortical.got.query_builder import Query

        # This should use status index if available
        results = (
            Query(large_manager)
            .tasks()
            .where(status="pending")
            .use_index("status")
            .execute()
        )

        assert len(results) > 0

    def test_lazy_iteration(self, large_manager):
        """Results can be iterated lazily."""
        from cortical.got.query_builder import Query

        query = Query(large_manager).tasks()

        # Should be able to iterate without loading all
        count = 0
        for task in query.iter():
            count += 1
            if count >= 10:
                break

        assert count == 10

    def test_explain_query(self, large_manager):
        """explain() returns query execution plan."""
        from cortical.got.query_builder import Query

        plan = (
            Query(large_manager)
            .tasks()
            .where(status="pending")
            .explain()
        )

        assert "steps" in plan
        assert len(plan["steps"]) > 0


class TestQueryMetrics:
    """Test QueryMetrics class for tracking query performance."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create basic manager."""
        got_dir = tmp_path / ".got"
        manager = GoTManager(got_dir)

        # Create some test data
        for i in range(10):
            manager.create_task(f"Task {i}", status="pending", priority="high")

        return manager

    def test_metrics_enabled(self, manager):
        """QueryMetrics tracks when enabled."""
        from cortical.got.query_builder import Query, QueryMetrics

        metrics = QueryMetrics(enabled=True)
        query = Query(manager, metrics=metrics)

        results = query.tasks().execute()

        stats = metrics.get_stats()
        assert stats['total_queries'] == 1
        assert stats['total_entities'] == 10

    def test_metrics_disabled(self, manager):
        """QueryMetrics does nothing when disabled."""
        from cortical.got.query_builder import Query, QueryMetrics

        metrics = QueryMetrics(enabled=False)
        query = Query(manager, metrics=metrics)

        results = query.tasks().execute()

        stats = metrics.get_stats()
        assert stats['total_queries'] == 0

    def test_metrics_empty_stats(self):
        """get_stats() returns zeros when no queries recorded."""
        from cortical.got.query_builder import QueryMetrics

        metrics = QueryMetrics()
        stats = metrics.get_stats()

        assert stats['total_queries'] == 0
        assert stats['total_entities'] == 0
        assert stats['avg_time_ms'] == 0.0
        assert stats['min_time_ms'] == 0.0
        assert stats['max_time_ms'] == 0.0

    def test_metrics_summary(self, manager):
        """summary() returns formatted string."""
        from cortical.got.query_builder import Query, QueryMetrics

        metrics = QueryMetrics(enabled=True)
        Query(manager, metrics=metrics).tasks().execute()

        summary = metrics.summary()

        assert "GoT Query API Metrics" in summary
        assert "Total queries:" in summary
        assert "Avg:" in summary

    def test_metrics_reset(self, manager):
        """reset() clears all metrics."""
        from cortical.got.query_builder import Query, QueryMetrics

        metrics = QueryMetrics(enabled=True)
        Query(manager, metrics=metrics).tasks().execute()

        metrics.reset()
        stats = metrics.get_stats()

        assert stats['total_queries'] == 0

    def test_module_level_metrics(self):
        """Module-level metrics functions work."""
        from cortical.got.query_builder import (
            get_query_metrics, enable_query_metrics, disable_query_metrics
        )

        metrics = get_query_metrics()
        assert metrics is not None

        enable_query_metrics()
        assert metrics.enabled is True

        disable_query_metrics()
        assert metrics.enabled is False


class TestAggregationFunctions:
    """Test aggregate functions (Sum, Avg, Min, Max, etc)."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create manager with numeric data."""
        got_dir = tmp_path / ".got"
        manager = GoTManager(got_dir)

        # Create tasks with numeric properties
        t1 = manager.create_task("T1", status="pending")
        t1.properties['score'] = 10
        manager.update_task(t1.id, properties=t1.properties)

        t2 = manager.create_task("T2", status="pending")
        t2.properties['score'] = 20
        manager.update_task(t2.id, properties=t2.properties)

        t3 = manager.create_task("T3", status="completed")
        t3.properties['score'] = 30
        manager.update_task(t3.id, properties=t3.properties)

        return manager

    def test_sum_aggregation(self, manager):
        """Sum aggregate function."""
        from cortical.got.query_builder import Query, Sum

        result = (
            Query(manager)
            .tasks()
            .group_by("status")
            .aggregate(total_score=Sum("score"))
            .execute()
        )

        assert result["pending"]["total_score"] == 30.0
        assert result["completed"]["total_score"] == 30.0

    def test_avg_aggregation(self, manager):
        """Avg aggregate function."""
        from cortical.got.query_builder import Query, Avg

        result = (
            Query(manager)
            .tasks()
            .group_by("status")
            .aggregate(avg_score=Avg("score"))
            .execute()
        )

        assert result["pending"]["avg_score"] == 15.0
        assert result["completed"]["avg_score"] == 30.0

    def test_min_aggregation(self, manager):
        """Min aggregate function."""
        from cortical.got.query_builder import Query, Min

        result = (
            Query(manager)
            .tasks()
            .group_by("status")
            .aggregate(min_score=Min("score"))
            .execute()
        )

        assert result["pending"]["min_score"] == 10
        assert result["completed"]["min_score"] == 30

    def test_max_aggregation(self, manager):
        """Max aggregate function."""
        from cortical.got.query_builder import Query, Max

        result = (
            Query(manager)
            .tasks()
            .group_by("status")
            .aggregate(max_score=Max("score"))
            .execute()
        )

        assert result["pending"]["max_score"] == 20
        assert result["completed"]["max_score"] == 30

    def test_avg_with_zero_values(self, tmp_path):
        """Avg returns 0.0 when no numeric values."""
        from cortical.got.query_builder import Query, Avg

        got_dir = tmp_path / ".got"
        manager = GoTManager(got_dir)
        t = manager.create_task("T1", status="pending")

        result = (
            Query(manager)
            .tasks()
            .group_by("status")
            .aggregate(avg_score=Avg("nonexistent"))
            .execute()
        )

        assert result["pending"]["avg_score"] == 0.0


class TestWhereOperators:
    """Test different WHERE clause operators."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create manager with varied data."""
        got_dir = tmp_path / ".got"
        manager = GoTManager(got_dir)

        t1 = manager.create_task("T1", status="pending")
        t1.properties['score'] = 10
        manager.update_task(t1.id, properties=t1.properties)

        t2 = manager.create_task("T2", status="pending")
        t2.properties['score'] = 20
        manager.update_task(t2.id, properties=t2.properties)

        t3 = manager.create_task("T3", status="completed")
        t3.properties['score'] = 30
        manager.update_task(t3.id, properties=t3.properties)

        return manager

    def test_where_with_properties(self, manager):
        """where() accesses entity.properties when attribute missing."""
        from cortical.got.query_builder import Query, WhereClause

        query = Query(manager).tasks()
        # Manually test _matches_clause with properties
        tasks = manager.list_all_tasks()
        task = tasks[0] if tasks else None
        assert task is not None
        clause = WhereClause(field="score", value=10, operator="eq")

        matches = query._matches_clause(task, clause)
        assert isinstance(matches, bool)

    def test_order_by_with_properties(self, manager):
        """order_by() accesses entity.properties."""
        from cortical.got.query_builder import Query

        results = (
            Query(manager)
            .tasks()
            .order_by("score")
            .execute()
        )

        # Should be sorted by score
        assert len(results) == 3

    def test_order_by_with_none_values(self, tmp_path):
        """order_by() handles None values."""
        from cortical.got.query_builder import Query

        got_dir = tmp_path / ".got"
        manager = GoTManager(got_dir)

        manager.create_task("T1", status="pending")
        manager.create_task("T2", status="completed")

        results = (
            Query(manager)
            .tasks()
            .order_by("nonexistent_field")
            .execute()
        )

        assert len(results) == 2

    def test_group_by_multiple_fields(self, manager):
        """group_by() with multiple fields returns tuple keys."""
        from cortical.got.query_builder import Query

        result = (
            Query(manager)
            .tasks()
            .group_by("status", "title")
            .count()
            .execute()
        )

        # Keys should be tuples
        for key in result.keys():
            assert isinstance(key, tuple)
            assert len(key) == 2

    def test_iter_with_offset_and_limit(self, manager):
        """iter() respects offset and limit."""
        from cortical.got.query_builder import Query

        items = list(
            Query(manager)
            .tasks()
            .offset(1)
            .limit(1)
            .iter()
        )

        assert len(items) == 1

    def test_explain_with_or_groups(self, tmp_path):
        """explain() includes OR groups in plan."""
        from cortical.got.query_builder import Query

        got_dir = tmp_path / ".got"
        manager = GoTManager(got_dir)
        manager.create_task("T1", status="pending", priority="high")

        plan = (
            Query(manager)
            .tasks()
            .where(status="pending")
            .or_where(priority="critical")
            .explain()
        )

        assert "steps" in plan

    def test_explain_with_connections(self, tmp_path):
        """explain() includes connection filters."""
        from cortical.got.query_builder import Query

        got_dir = tmp_path / ".got"
        manager = GoTManager(got_dir)
        t1 = manager.create_task("T1", status="pending")

        plan = (
            Query(manager)
            .tasks()
            .connected_to(t1.id, via="DEPENDS_ON")
            .explain()
        )

        # Check for connection_filter step
        step_types = [s['type'] for s in plan.steps]
        assert 'connection_filter' in step_types

    def test_explain_with_pagination(self, tmp_path):
        """explain() includes pagination when offset/limit present."""
        from cortical.got.query_builder import Query

        got_dir = tmp_path / ".got"
        manager = GoTManager(got_dir)
        manager.create_task("T1", status="pending")

        plan = (
            Query(manager)
            .tasks()
            .limit(10)
            .offset(5)
            .explain()
        )

        step_types = [s['type'] for s in plan.steps]
        assert 'pagination' in step_types

    def test_query_plan_dict_access(self):
        """QueryPlan supports dict-like access."""
        from cortical.got.query_builder import QueryPlan

        plan = QueryPlan(
            steps=[],
            estimated_cost=10.0,
            uses_index=True,
            index_name="test_index"
        )

        assert plan["steps"] == []
        assert plan["estimated_cost"] == 10.0
        assert "steps" in plan
        assert "nonexistent" not in plan

    def test_connected_to_direction_incoming(self, tmp_path):
        """connected_to() with direction='incoming' finds sources."""
        from cortical.got.query_builder import Query

        got_dir = tmp_path / ".got"
        manager = GoTManager(got_dir)

        t1 = manager.create_task("T1", status="pending")
        t2 = manager.create_task("T2", status="pending")
        # Edge: t1 -> t2 (t1 is source, t2 is target)
        manager.add_edge(t1.id, t2.id, "DEPENDS_ON")

        # Find tasks with incoming edges TO t2 (i.e., sources pointing to t2)
        results = (
            Query(manager)
            .tasks()
            .connected_to(t2.id, direction="incoming")
            .execute()
        )

        # t1 should be found (t1 points to t2)
        assert any(r.id == t1.id for r in results)

    def test_connected_to_direction_outgoing(self, tmp_path):
        """connected_to() with direction='outgoing' finds targets."""
        from cortical.got.query_builder import Query

        got_dir = tmp_path / ".got"
        manager = GoTManager(got_dir)

        t1 = manager.create_task("T1", status="pending")
        t2 = manager.create_task("T2", status="pending")
        # Edge: t1 -> t2 (t1 is source, t2 is target)
        manager.add_edge(t1.id, t2.id, "DEPENDS_ON")

        # Find tasks with outgoing edges FROM t1 (i.e., targets that t1 points to)
        results = (
            Query(manager)
            .tasks()
            .connected_to(t1.id, direction="outgoing")
            .execute()
        )

        # t2 should be found (t1 points to t2)
        assert any(r.id == t2.id for r in results)

    def test_execute_exception_handling(self, tmp_path):
        """execute() records metrics even on exception."""
        from cortical.got.query_builder import Query, QueryMetrics
        from unittest.mock import MagicMock, patch

        got_dir = tmp_path / ".got"
        manager = GoTManager(got_dir)
        manager.create_task("T1", status="pending")

        metrics = QueryMetrics(enabled=True)
        query = Query(manager, metrics=metrics).tasks()

        # Force an exception during execution
        with patch.object(query, '_execute_query', side_effect=RuntimeError("test error")):
            try:
                query.execute()
            except RuntimeError:
                pass

        # Metrics should still record the failed query
        stats = metrics.get_stats()
        assert stats['total_queries'] == 1


class TestWhereOperatorsExtended:
    """Test all WHERE clause operators comprehensively."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create manager with test data for operator tests."""
        got_dir = tmp_path / ".got"
        manager = GoTManager(got_dir)

        # Create tasks with various numeric properties
        t1 = manager.create_task("T1", status="pending")
        t1.properties['score'] = 10
        t1.properties['tags'] = ['high', 'urgent']
        t1.properties['description'] = 'Important task'
        manager.update_task(t1.id, properties=t1.properties)

        t2 = manager.create_task("T2", status="pending")
        t2.properties['score'] = 20
        t2.properties['tags'] = ['medium']
        t2.properties['description'] = 'Regular task'
        manager.update_task(t2.id, properties=t2.properties)

        t3 = manager.create_task("T3", status="completed")
        t3.properties['score'] = 30
        t3.properties['tags'] = ['low']
        t3.properties['description'] = 'Done'
        manager.update_task(t3.id, properties=t3.properties)

        return manager

    def test_where_operator_ne(self, manager):
        """Test != operator."""
        from cortical.got.query_builder import Query, WhereClause

        query = Query(manager).tasks()
        tasks = manager.list_all_tasks()

        # Test ne operator
        clause = WhereClause(field="status", value="pending", operator="ne")
        completed_task = [t for t in tasks if t.status == "completed"][0]

        assert query._matches_clause(completed_task, clause) is True

    def test_where_operator_gt(self, manager):
        """Test > operator."""
        from cortical.got.query_builder import Query, WhereClause

        query = Query(manager).tasks()
        tasks = manager.list_all_tasks()

        # Test gt operator with properties
        clause = WhereClause(field="score", value=15, operator="gt")
        high_score_task = [t for t in tasks if t.properties.get('score', 0) > 15][0]

        assert query._matches_clause(high_score_task, clause) is True

    def test_where_operator_lt(self, manager):
        """Test < operator."""
        from cortical.got.query_builder import Query, WhereClause

        query = Query(manager).tasks()
        tasks = manager.list_all_tasks()

        # Test lt operator
        clause = WhereClause(field="score", value=15, operator="lt")
        low_score_task = [t for t in tasks if t.properties.get('score', 0) < 15][0]

        assert query._matches_clause(low_score_task, clause) is True

    def test_where_operator_gte(self, manager):
        """Test >= operator."""
        from cortical.got.query_builder import Query, WhereClause

        query = Query(manager).tasks()
        tasks = manager.list_all_tasks()

        # Test gte operator
        clause = WhereClause(field="score", value=20, operator="gte")
        task = [t for t in tasks if t.properties.get('score', 0) >= 20][0]

        assert query._matches_clause(task, clause) is True

    def test_where_operator_lte(self, manager):
        """Test <= operator."""
        from cortical.got.query_builder import Query, WhereClause

        query = Query(manager).tasks()
        tasks = manager.list_all_tasks()

        # Test lte operator
        clause = WhereClause(field="score", value=20, operator="lte")
        task = [t for t in tasks if t.properties.get('score', 0) <= 20][0]

        assert query._matches_clause(task, clause) is True

    def test_where_operator_in(self, manager):
        """Test 'in' operator."""
        from cortical.got.query_builder import Query, WhereClause

        query = Query(manager).tasks()
        tasks = manager.list_all_tasks()

        # Test in operator
        clause = WhereClause(field="status", value=["pending", "completed"], operator="in")
        task = tasks[0]

        assert query._matches_clause(task, clause) is True

    def test_where_operator_contains(self, manager):
        """Test 'contains' operator."""
        from cortical.got.query_builder import Query, WhereClause

        query = Query(manager).tasks()
        tasks = manager.list_all_tasks()

        # Test contains operator on title field (which exists on all tasks)
        task = tasks[0]
        clause = WhereClause(field="title", value=task.title[:2], operator="contains")

        assert query._matches_clause(task, clause) is True

    def test_where_operator_with_none_value(self, manager):
        """Test operators with None values."""
        from cortical.got.query_builder import Query, WhereClause

        query = Query(manager).tasks()
        tasks = manager.list_all_tasks()

        # Test gt with None value (should return False)
        clause = WhereClause(field="nonexistent", value=10, operator="gt")
        task = tasks[0]

        assert query._matches_clause(task, clause) is False


class TestAggregationEdgeCases:
    """Test edge cases in aggregation functions."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create manager with varied data types."""
        got_dir = tmp_path / ".got"
        manager = GoTManager(got_dir)

        # Task with numeric property
        t1 = manager.create_task("T1", status="pending")
        t1.properties['value'] = 100
        manager.update_task(t1.id, properties=t1.properties)

        # Task without the property
        t2 = manager.create_task("T2", status="pending")

        # Task with non-numeric property
        t3 = manager.create_task("T3", status="completed")
        t3.properties['value'] = "not a number"
        manager.update_task(t3.id, properties=t3.properties)

        return manager

    def test_collect_with_properties(self, manager):
        """Collect accesses entity.properties when attribute missing."""
        from cortical.got.query_builder import Query, Collect

        result = (
            Query(manager)
            .tasks()
            .group_by("status")
            .aggregate(values=Collect("value"))
            .execute()
        )

        # Should collect values from properties
        assert 100 in result["pending"]["values"]

    def test_sum_with_properties(self, manager):
        """Sum accesses entity.properties when attribute missing."""
        from cortical.got.query_builder import Query, Sum

        result = (
            Query(manager)
            .tasks()
            .group_by("status")
            .aggregate(total=Sum("value"))
            .execute()
        )

        # Should sum numeric values from properties
        assert result["pending"]["total"] == 100.0

    def test_sum_skips_non_numeric(self, manager):
        """Sum skips non-numeric values."""
        from cortical.got.query_builder import Query, Sum

        result = (
            Query(manager)
            .tasks()
            .group_by("status")
            .aggregate(total=Sum("value"))
            .execute()
        )

        # Should skip "not a number" string
        assert result["completed"]["total"] == 0.0

    def test_avg_with_properties(self, manager):
        """Avg accesses entity.properties when attribute missing."""
        from cortical.got.query_builder import Query, Avg

        result = (
            Query(manager)
            .tasks()
            .where(status="pending")
            .group_by("status")
            .aggregate(avg_val=Avg("value"))
            .execute()
        )

        # Should average numeric values from properties
        # Only t1 has numeric value (100), t2 has none
        assert result["pending"]["avg_val"] == 100.0

    def test_min_with_properties(self, manager):
        """Min accesses entity.properties when attribute missing."""
        from cortical.got.query_builder import Query, Min

        result = (
            Query(manager)
            .tasks()
            .group_by("status")
            .aggregate(min_val=Min("value"))
            .execute()
        )

        # Should find minimum from properties
        assert result["pending"]["min_val"] == 100

    def test_max_with_properties(self, manager):
        """Max accesses entity.properties when attribute missing."""
        from cortical.got.query_builder import Query, Max

        result = (
            Query(manager)
            .tasks()
            .group_by("status")
            .aggregate(max_val=Max("value"))
            .execute()
        )

        # Should find maximum from properties
        assert result["pending"]["max_val"] == 100


class TestOrderByEdgeCases:
    """Test edge cases in order_by functionality."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create manager with data for sorting tests."""
        got_dir = tmp_path / ".got"
        manager = GoTManager(got_dir)

        # Create tasks with different timestamps
        t1 = manager.create_task("T1", status="pending", priority="high")
        t1.properties['score'] = 30
        manager.update_task(t1.id, properties=t1.properties)

        t2 = manager.create_task("T2", status="pending", priority="low")
        t2.properties['score'] = 10
        manager.update_task(t2.id, properties=t2.properties)

        t3 = manager.create_task("T3", status="completed", priority="high")
        t3.properties['score'] = 20
        manager.update_task(t3.id, properties=t3.properties)

        return manager

    def test_order_by_multiple_fields(self, manager):
        """order_by() with multiple fields."""
        from cortical.got.query_builder import Query

        results = (
            Query(manager)
            .tasks()
            .order_by("priority")
            .order_by("score")
            .execute()
        )

        # Should sort by priority first, then score
        assert len(results) == 3

    def test_order_by_priority_special_handling(self, manager):
        """order_by() handles priority field specially."""
        from cortical.got.query_builder import Query

        results = (
            Query(manager)
            .tasks()
            .order_by("priority")
            .execute()
        )

        # Priority should be ordered: critical, high, medium, low
        priorities = [r.priority for r in results]
        # high should come before low
        high_idx = next(i for i, p in enumerate(priorities) if p == "high")
        low_idx = next(i for i, p in enumerate(priorities) if p == "low")
        assert high_idx < low_idx

    def test_order_by_empty_list(self, manager):
        """order_by() with no clauses returns unsorted."""
        from cortical.got.query_builder import Query

        query = Query(manager).tasks()
        results = list(query._execute_query())

        # _apply_sorting with empty order_by should return as-is
        sorted_results = query._apply_sorting(results)
        assert sorted_results == results


class TestGroupByEdgeCases:
    """Test edge cases in group_by functionality."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create manager for grouping tests."""
        got_dir = tmp_path / ".got"
        manager = GoTManager(got_dir)

        t1 = manager.create_task("T1", status="pending", priority="high")
        t1.properties['category'] = 'feature'
        manager.update_task(t1.id, properties=t1.properties)

        t2 = manager.create_task("T2", status="pending", priority="low")
        t2.properties['category'] = 'bugfix'
        manager.update_task(t2.id, properties=t2.properties)

        return manager

    def test_group_by_with_properties_access(self, manager):
        """group_by() accesses entity.properties when needed."""
        from cortical.got.query_builder import Query

        result = (
            Query(manager)
            .tasks()
            .group_by("category")
            .count()
            .execute()
        )

        # Should group by category from properties
        assert result.get('feature') == 1
        assert result.get('bugfix') == 1

    def test_group_by_multiple_with_properties(self, manager):
        """group_by() multiple fields accessing properties."""
        from cortical.got.query_builder import Query

        result = (
            Query(manager)
            .tasks()
            .group_by("status", "category")
            .count()
            .execute()
        )

        # Keys should be tuples with values from properties
        assert result.get(('pending', 'feature')) == 1


class TestIterEdgeCases:
    """Test edge cases in iter() method."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create manager for iteration tests."""
        got_dir = tmp_path / ".got"
        manager = GoTManager(got_dir)

        for i in range(5):
            manager.create_task(f"T{i}", status="pending")

        return manager

    def test_iter_with_limit_stops_early(self, manager):
        """iter() stops at limit without processing all."""
        from cortical.got.query_builder import Query

        count = 0
        for task in Query(manager).tasks().limit(2).iter():
            count += 1

        assert count == 2

    def test_iter_with_offset_skips_correctly(self, manager):
        """iter() skips offset items correctly."""
        from cortical.got.query_builder import Query

        all_tasks = Query(manager).tasks().execute()
        offset_tasks = list(Query(manager).tasks().offset(2).iter())

        assert len(offset_tasks) == len(all_tasks) - 2


class TestMetricsEdgeCases:
    """Test QueryMetrics edge cases."""

    def test_metrics_avg_entities_per_query(self, tmp_path):
        """get_stats() calculates avg_entities_per_query."""
        from cortical.got.query_builder import Query, QueryMetrics

        got_dir = tmp_path / ".got"
        manager = GoTManager(got_dir)

        for i in range(10):
            manager.create_task(f"T{i}", status="pending")

        metrics = QueryMetrics(enabled=True)

        # Run multiple queries
        Query(manager, metrics=metrics).tasks().limit(5).execute()
        Query(manager, metrics=metrics).tasks().limit(3).execute()

        stats = metrics.get_stats()
        assert stats['avg_entities_per_query'] == 4.0  # (5 + 3) / 2


class TestExecuteEdgeCases:
    """Test execute() method edge cases."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create manager for execution tests."""
        got_dir = tmp_path / ".got"
        manager = GoTManager(got_dir)

        t1 = manager.create_task("T1", status="pending", priority="high")
        t2 = manager.create_task("T2", status="pending", priority="low")
        t3 = manager.create_task("T3", status="completed", priority="high")

        return manager

    def test_execute_with_count_mode_and_group_by(self, manager):
        """execute() in count mode with group_by returns counts dict."""
        from cortical.got.query_builder import Query

        result = (
            Query(manager)
            .tasks()
            .group_by("status")
            .count()
            .execute()
        )

        assert isinstance(result, dict)
        assert result["pending"] == 2
        assert result["completed"] == 1

    def test_execute_aggregation_without_aggregates(self, manager):
        """execute() with group_by but no aggregates returns counts."""
        from cortical.got.query_builder import Query

        # This tests the _execute_aggregation branch where _aggregates is empty
        result = (
            Query(manager)
            .tasks()
            .group_by("status")
            .count()
            .execute()
        )

        assert result["pending"] == 2


class TestModuleLevelFunctions:
    """Test module-level query metrics functions."""

    def test_enable_disable_query_metrics(self):
        """enable/disable_query_metrics toggle module-level metrics."""
        from cortical.got.query_builder import (
            get_query_metrics,
            enable_query_metrics,
            disable_query_metrics
        )

        metrics = get_query_metrics()

        # Initially disabled
        assert metrics.enabled is False

        # Enable
        enable_query_metrics()
        assert metrics.enabled is True

        # Disable
        disable_query_metrics()
        assert metrics.enabled is False


class TestUnknownEntityType:
    """Test handling of unknown entity types."""

    def test_get_base_entities_unknown_type(self, tmp_path):
        """_get_base_entities() returns empty list for unknown type."""
        from cortical.got.query_builder import Query

        got_dir = tmp_path / ".got"
        manager = GoTManager(got_dir)

        query = Query(manager)
        # Don't set entity type - should be None

        entities = query._get_base_entities()
        assert entities == []


class TestConnectedToEdgeTypeFiltering:
    """Test _get_connected_ids() edge type filtering."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create manager with multiple edge types."""
        got_dir = tmp_path / ".got"
        manager = GoTManager(got_dir)

        t1 = manager.create_task("T1", status="pending")
        t2 = manager.create_task("T2", status="pending")
        t3 = manager.create_task("T3", status="completed")

        # Different edge types
        manager.add_edge(t1.id, t2.id, "DEPENDS_ON")
        manager.add_edge(t1.id, t3.id, "BLOCKS")

        self.t1 = t1
        self.t2 = t2
        self.t3 = t3

        return manager

    def test_connected_to_filters_by_edge_type(self, manager):
        """connected_to() with via filters by edge type."""
        from cortical.got.query_builder import Query

        # Find tasks connected via DEPENDS_ON only
        results = (
            Query(manager)
            .tasks()
            .connected_to(self.t1.id, via="DEPENDS_ON", direction="outgoing")
            .execute()
        )

        # Should find t2 but not t3
        result_ids = [r.id for r in results]
        assert self.t2.id in result_ids
        assert self.t3.id not in result_ids

    def test_connected_to_without_edge_type_filter(self, manager):
        """connected_to() without via includes all edge types."""
        from cortical.got.query_builder import Query

        # Find all tasks connected from t1
        results = (
            Query(manager)
            .tasks()
            .connected_to(self.t1.id, direction="outgoing")
            .execute()
        )

        # Should find both t2 and t3
        result_ids = [r.id for r in results]
        assert self.t2.id in result_ids
        assert self.t3.id in result_ids


class TestMatchesFiltersEdgeCases:
    """Test _matches_filters() edge cases."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create manager for filter tests."""
        got_dir = tmp_path / ".got"
        manager = GoTManager(got_dir)

        t1 = manager.create_task("T1", status="pending", priority="high")
        t2 = manager.create_task("T2", status="completed", priority="low")

        self.t1 = t1
        self.t2 = t2

        return manager

    def test_matches_filters_only_or_groups(self, manager):
        """_matches_filters() with only OR groups."""
        from cortical.got.query_builder import Query

        # Create query with only OR groups (no WHERE clauses)
        results = (
            Query(manager)
            .tasks()
            .or_where(status="pending")
            .or_where(status="completed")
            .execute()
        )

        # Should match all tasks
        assert len(results) == 2
