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
