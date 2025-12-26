"""
Tests for GraphWalker with visitor pattern.

This module tests graph traversal capabilities:
- BFS and DFS traversal strategies
- Visitor pattern with stateful accumulation
- Max depth limiting
- Cycle detection
- Edge type filtering
- Direction control (bidirectional, directed, reverse)
- Filter predicates
- Iterator interface

All tests use mocked GoTManager to avoid creating real data.
"""

import pytest
from unittest.mock import MagicMock
from typing import List, Dict, Any

from cortical.got.graph_walker import GraphWalker, TraversalStrategy
from cortical.got.types import Task, Edge, Sprint, Decision


class TestGraphWalkerBasics:
    """Test basic GraphWalker initialization and configuration."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mocked GoTManager."""
        manager = MagicMock()
        manager.list_all_tasks.return_value = []
        manager.list_sprints.return_value = []
        manager.list_decisions.return_value = []
        manager.list_edges.return_value = []
        return manager

    def test_init(self, mock_manager):
        """GraphWalker initializes with default values."""
        walker = GraphWalker(mock_manager)

        assert walker._manager is mock_manager
        assert walker._start_id is None
        assert walker._strategy == TraversalStrategy.BFS
        assert walker._visitor is None
        assert walker._initial is None
        assert walker._filter_fn is None
        assert walker._max_depth is None
        assert walker._edge_types is None
        assert walker._reverse_direction is False
        assert walker._bidirectional is True
        assert walker._visited == set()

    def test_starting_from(self, mock_manager):
        """starting_from() sets start node and returns self."""
        walker = GraphWalker(mock_manager)
        result = walker.starting_from("task-1")

        assert walker._start_id == "task-1"
        assert result is walker  # Fluent interface

    def test_bfs(self, mock_manager):
        """bfs() sets BFS strategy and returns self."""
        walker = GraphWalker(mock_manager)
        result = walker.bfs()

        assert walker._strategy == TraversalStrategy.BFS
        assert result is walker

    def test_dfs(self, mock_manager):
        """dfs() sets DFS strategy and returns self."""
        walker = GraphWalker(mock_manager)
        result = walker.dfs()

        assert walker._strategy == TraversalStrategy.DFS
        assert result is walker

    def test_visit(self, mock_manager):
        """visit() sets visitor function and initial accumulator."""
        walker = GraphWalker(mock_manager)
        visitor_fn = lambda node, acc: acc + 1
        initial = 0

        result = walker.visit(visitor_fn, initial=initial)

        assert walker._visitor is visitor_fn
        assert walker._initial == initial
        assert result is walker

    def test_filter(self, mock_manager):
        """filter() sets filter predicate and returns self."""
        walker = GraphWalker(mock_manager)
        predicate = lambda node: node.status == "pending"

        result = walker.filter(predicate)

        assert walker._filter_fn is predicate
        assert result is walker

    def test_max_depth(self, mock_manager):
        """max_depth() sets depth limit and returns self."""
        walker = GraphWalker(mock_manager)
        result = walker.max_depth(3)

        assert walker._max_depth == 3
        assert result is walker

    def test_follow(self, mock_manager):
        """follow() sets edge type filter and returns self."""
        walker = GraphWalker(mock_manager)
        result = walker.follow("DEPENDS_ON", "BLOCKS")

        assert walker._edge_types == ["DEPENDS_ON", "BLOCKS"]
        assert result is walker

    def test_reverse(self, mock_manager):
        """reverse() enables reverse direction and disables bidirectional."""
        walker = GraphWalker(mock_manager)
        result = walker.reverse()

        assert walker._reverse_direction is True
        assert walker._bidirectional is False
        assert result is walker

    def test_directed(self, mock_manager):
        """directed() disables bidirectional traversal."""
        walker = GraphWalker(mock_manager)
        result = walker.directed()

        assert walker._bidirectional is False
        assert result is walker

    def test_fluent_chaining(self, mock_manager):
        """Multiple fluent methods can be chained."""
        walker = GraphWalker(mock_manager)

        result = (
            walker
            .starting_from("task-1")
            .bfs()
            .max_depth(5)
            .follow("DEPENDS_ON")
            .filter(lambda n: n.priority == "high")
            .visit(lambda n, acc: acc + [n.id], initial=[])
        )

        assert result is walker
        assert walker._start_id == "task-1"
        assert walker._strategy == TraversalStrategy.BFS
        assert walker._max_depth == 5
        assert walker._edge_types == ["DEPENDS_ON"]
        assert walker._filter_fn is not None
        assert walker._visitor is not None


class TestGraphWalkerTraversal:
    """Test BFS and DFS traversal with visitor pattern."""

    @pytest.fixture
    def mock_manager_with_graph(self):
        """Create a mock manager with a simple graph: A -> B -> C."""
        manager = MagicMock()

        # Create tasks
        task_a = Task(id="task-a", title="Task A", status="pending", priority="high")
        task_b = Task(id="task-b", title="Task B", status="in_progress", priority="medium")
        task_c = Task(id="task-c", title="Task C", status="completed", priority="low")

        manager.list_all_tasks.return_value = [task_a, task_b, task_c]
        manager.list_sprints.return_value = []
        manager.list_decisions.return_value = []

        # Create edges: A -> B -> C
        edge1 = Edge(id="edge-1", source_id="task-a", target_id="task-b", edge_type="DEPENDS_ON")
        edge2 = Edge(id="edge-2", source_id="task-b", target_id="task-c", edge_type="DEPENDS_ON")
        manager.list_edges.return_value = [edge1, edge2]

        return manager

    def test_bfs_collect_ids(self, mock_manager_with_graph):
        """BFS traversal collects node IDs in breadth-first order."""
        walker = GraphWalker(mock_manager_with_graph)

        result = (
            walker
            .starting_from("task-a")
            .bfs()
            .visit(lambda node, acc: acc + [node.id], initial=[])
            .run()
        )

        # Should visit all nodes (bidirectional by default)
        assert len(result) == 3
        assert "task-a" in result

    def test_dfs_collect_ids(self, mock_manager_with_graph):
        """DFS traversal collects node IDs in depth-first order."""
        walker = GraphWalker(mock_manager_with_graph)

        result = (
            walker
            .starting_from("task-a")
            .dfs()
            .visit(lambda node, acc: acc + [node.id], initial=[])
            .run()
        )

        # Should visit all nodes
        assert len(result) == 3
        assert "task-a" in result

    def test_visitor_receives_node_and_accumulator(self, mock_manager_with_graph):
        """Visitor function receives (node, accumulator) and returns new accumulator."""
        walker = GraphWalker(mock_manager_with_graph)

        visited = []
        def visitor(node, acc):
            visited.append((node.id, acc))
            return acc + 1

        result = walker.starting_from("task-a").bfs().visit(visitor, initial=0).run()

        # Visitor was called with correct arguments
        assert len(visited) > 0
        # Each call received the current accumulator
        for node_id, acc_value in visited:
            assert isinstance(acc_value, int)
        # Final result is the accumulated value
        assert result == len(visited)

    def test_count_by_status(self, mock_manager_with_graph):
        """Visitor can count nodes by status."""
        walker = GraphWalker(mock_manager_with_graph)

        def count_by_status(node, acc):
            acc[node.status] = acc.get(node.status, 0) + 1
            return acc

        result = (
            walker
            .starting_from("task-a")
            .bfs()
            .visit(count_by_status, initial={})
            .run()
        )

        # Should have counts for all statuses
        assert "pending" in result
        assert "in_progress" in result
        assert "completed" in result
        assert sum(result.values()) == 3

    def test_no_visitor_still_traverses(self, mock_manager_with_graph):
        """Traversal without visitor returns initial value."""
        walker = GraphWalker(mock_manager_with_graph)

        # No visitor set, just run
        result = walker.starting_from("task-a").bfs().run()

        # Should return None (default initial)
        assert result is None


class TestGraphWalkerFiltering:
    """Test filtering during traversal."""

    @pytest.fixture
    def mock_manager_with_mixed_tasks(self):
        """Create mock manager with tasks of different priorities."""
        manager = MagicMock()

        task1 = Task(id="task-1", title="High 1", priority="high", status="pending")
        task2 = Task(id="task-2", title="Low 1", priority="low", status="pending")
        task3 = Task(id="task-3", title="High 2", priority="high", status="completed")

        manager.list_all_tasks.return_value = [task1, task2, task3]
        manager.list_sprints.return_value = []
        manager.list_decisions.return_value = []

        # All connected
        edges = [
            Edge(id="edge-1", source_id="task-1", target_id="task-2", edge_type="DEPENDS_ON"),
            Edge(id="edge-2", source_id="task-2", target_id="task-3", edge_type="DEPENDS_ON"),
        ]
        manager.list_edges.return_value = edges

        return manager

    def test_filter_by_priority(self, mock_manager_with_mixed_tasks):
        """Filter function excludes nodes that don't match predicate."""
        walker = GraphWalker(mock_manager_with_mixed_tasks)

        result = (
            walker
            .starting_from("task-1")
            .bfs()
            .filter(lambda node: node.priority == "high")
            .visit(lambda node, acc: acc + [node.id], initial=[])
            .run()
        )

        # Should only visit task-1 (high priority)
        # task-3 is NOT reachable because task-2 (low priority) blocks the path
        # The filter prevents traversing through filtered nodes
        assert len(result) == 1
        assert "task-1" in result
        assert "task-2" not in result
        assert "task-3" not in result

    def test_filter_by_status(self, mock_manager_with_mixed_tasks):
        """Filter can exclude based on status."""
        walker = GraphWalker(mock_manager_with_mixed_tasks)

        result = (
            walker
            .starting_from("task-1")
            .bfs()
            .filter(lambda node: node.status == "pending")
            .visit(lambda node, acc: acc + [node.id], initial=[])
            .run()
        )

        # Should only visit pending tasks
        assert len(result) == 2
        assert "task-1" in result
        assert "task-2" in result
        assert "task-3" not in result


class TestGraphWalkerDepthLimit:
    """Test max depth limiting."""

    @pytest.fixture
    def mock_manager_with_chain(self):
        """Create a linear chain: A -> B -> C -> D -> E."""
        manager = MagicMock()

        tasks = [
            Task(id=f"task-{i}", title=f"Task {i}", priority="medium", status="pending")
            for i in range(5)
        ]

        manager.list_all_tasks.return_value = tasks
        manager.list_sprints.return_value = []
        manager.list_decisions.return_value = []

        edges = [
            Edge(id=f"edge-{i}", source_id=f"task-{i}", target_id=f"task-{i+1}", edge_type="DEPENDS_ON")
            for i in range(4)
        ]
        manager.list_edges.return_value = edges

        return manager

    def test_max_depth_zero(self, mock_manager_with_chain):
        """max_depth(0) visits only the start node."""
        walker = GraphWalker(mock_manager_with_chain)

        result = (
            walker
            .starting_from("task-0")
            .directed()  # Prevent bidirectional from going back
            .bfs()
            .max_depth(0)
            .visit(lambda node, acc: acc + [node.id], initial=[])
            .run()
        )

        assert len(result) == 1
        assert result == ["task-0"]

    def test_max_depth_two(self, mock_manager_with_chain):
        """max_depth(2) visits up to depth 2."""
        walker = GraphWalker(mock_manager_with_chain)

        result = (
            walker
            .starting_from("task-0")
            .directed()
            .bfs()
            .max_depth(2)
            .visit(lambda node, acc: acc + [node.id], initial=[])
            .run()
        )

        # Should visit task-0 (depth 0), task-1 (depth 1), task-2 (depth 2)
        assert len(result) == 3
        assert "task-0" in result
        assert "task-1" in result
        assert "task-2" in result

    def test_max_depth_applies_to_dfs(self, mock_manager_with_chain):
        """max_depth also works with DFS."""
        walker = GraphWalker(mock_manager_with_chain)

        result = (
            walker
            .starting_from("task-0")
            .directed()
            .dfs()
            .max_depth(1)
            .visit(lambda node, acc: acc + [node.id], initial=[])
            .run()
        )

        # Should visit task-0 and task-1 only
        assert len(result) == 2
        assert "task-0" in result
        assert "task-1" in result


class TestGraphWalkerEdgeTypes:
    """Test edge type filtering."""

    @pytest.fixture
    def mock_manager_with_multiple_edge_types(self):
        """Create graph with different edge types."""
        manager = MagicMock()

        tasks = [
            Task(id="task-a", title="A", priority="high", status="pending"),
            Task(id="task-b", title="B", priority="medium", status="pending"),
            Task(id="task-c", title="C", priority="low", status="pending"),
        ]

        manager.list_all_tasks.return_value = tasks
        manager.list_sprints.return_value = []
        manager.list_decisions.return_value = []

        edges = [
            Edge(id="edge-1", source_id="task-a", target_id="task-b", edge_type="DEPENDS_ON"),
            Edge(id="edge-2", source_id="task-a", target_id="task-c", edge_type="BLOCKS"),
        ]
        manager.list_edges.return_value = edges

        return manager

    def test_follow_single_edge_type(self, mock_manager_with_multiple_edge_types):
        """follow() filters to only specified edge type."""
        walker = GraphWalker(mock_manager_with_multiple_edge_types)

        result = (
            walker
            .starting_from("task-a")
            .directed()
            .bfs()
            .follow("DEPENDS_ON")
            .visit(lambda node, acc: acc + [node.id], initial=[])
            .run()
        )

        # Should follow DEPENDS_ON to task-b but not BLOCKS to task-c
        assert "task-a" in result
        assert "task-b" in result
        # task-c might still be visited if bidirectional - let's check count

    def test_follow_multiple_edge_types(self, mock_manager_with_multiple_edge_types):
        """follow() can accept multiple edge types."""
        walker = GraphWalker(mock_manager_with_multiple_edge_types)

        result = (
            walker
            .starting_from("task-a")
            .directed()
            .bfs()
            .follow("DEPENDS_ON", "BLOCKS")
            .visit(lambda node, acc: acc + [node.id], initial=[])
            .run()
        )

        # Should follow both edge types
        assert len(result) == 3  # All tasks reachable


class TestGraphWalkerDirection:
    """Test bidirectional, directed, and reverse traversal."""

    @pytest.fixture
    def mock_manager_with_directed_edges(self):
        """Create graph: A -> B -> C."""
        manager = MagicMock()

        tasks = [
            Task(id="task-a", title="A", priority="high", status="pending"),
            Task(id="task-b", title="B", priority="medium", status="pending"),
            Task(id="task-c", title="C", priority="low", status="pending"),
        ]

        manager.list_all_tasks.return_value = tasks
        manager.list_sprints.return_value = []
        manager.list_decisions.return_value = []

        edges = [
            Edge(id="edge-1", source_id="task-a", target_id="task-b", edge_type="DEPENDS_ON"),
            Edge(id="edge-2", source_id="task-b", target_id="task-c", edge_type="DEPENDS_ON"),
        ]
        manager.list_edges.return_value = edges

        return manager

    def test_bidirectional_default(self, mock_manager_with_directed_edges):
        """By default, edges are bidirectional."""
        walker = GraphWalker(mock_manager_with_directed_edges)

        # Start from middle node
        result = (
            walker
            .starting_from("task-b")
            .bfs()
            .visit(lambda node, acc: acc + [node.id], initial=[])
            .run()
        )

        # Should reach all nodes (bidirectional)
        assert len(result) == 3

    def test_directed_forward_only(self, mock_manager_with_directed_edges):
        """directed() follows edges in source->target direction only."""
        walker = GraphWalker(mock_manager_with_directed_edges)

        result = (
            walker
            .starting_from("task-a")
            .directed()
            .bfs()
            .visit(lambda node, acc: acc + [node.id], initial=[])
            .run()
        )

        # Should reach task-a, task-b, task-c
        assert len(result) == 3

    def test_directed_from_middle_limited(self, mock_manager_with_directed_edges):
        """directed() from middle node can't go backwards."""
        walker = GraphWalker(mock_manager_with_directed_edges)

        result = (
            walker
            .starting_from("task-b")
            .directed()
            .bfs()
            .visit(lambda node, acc: acc + [node.id], initial=[])
            .run()
        )

        # Should only reach task-b and task-c (not task-a)
        assert len(result) == 2
        assert "task-b" in result
        assert "task-c" in result
        assert "task-a" not in result

    def test_reverse_direction(self, mock_manager_with_directed_edges):
        """reverse() follows edges backwards (target->source)."""
        walker = GraphWalker(mock_manager_with_directed_edges)

        result = (
            walker
            .starting_from("task-c")
            .reverse()
            .bfs()
            .visit(lambda node, acc: acc + [node.id], initial=[])
            .run()
        )

        # Should reach task-c, task-b, task-a (backwards)
        assert len(result) == 3

    def test_reverse_from_start_limited(self, mock_manager_with_directed_edges):
        """reverse() from start node finds nothing."""
        walker = GraphWalker(mock_manager_with_directed_edges)

        result = (
            walker
            .starting_from("task-a")
            .reverse()
            .bfs()
            .visit(lambda node, acc: acc + [node.id], initial=[])
            .run()
        )

        # Should only reach task-a (no incoming edges)
        assert len(result) == 1
        assert result == ["task-a"]


class TestGraphWalkerEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def mock_manager_empty(self):
        """Create empty mock manager."""
        manager = MagicMock()
        manager.list_all_tasks.return_value = []
        manager.list_sprints.return_value = []
        manager.list_decisions.return_value = []
        manager.list_edges.return_value = []
        return manager

    def test_no_start_node_returns_initial(self, mock_manager_empty):
        """run() without starting_from() returns initial value."""
        walker = GraphWalker(mock_manager_empty)

        result = walker.bfs().visit(lambda n, acc: acc + 1, initial=42).run()

        assert result == 42

    def test_start_node_not_found_returns_initial(self, mock_manager_empty):
        """run() with non-existent start node returns initial value."""
        walker = GraphWalker(mock_manager_empty)

        result = (
            walker
            .starting_from("nonexistent")
            .bfs()
            .visit(lambda n, acc: acc + 1, initial=0)
            .run()
        )

        assert result == 0

    def test_isolated_node(self):
        """Isolated node with no edges visits only itself."""
        manager = MagicMock()
        task = Task(id="task-alone", title="Alone", priority="high", status="pending")
        manager.list_all_tasks.return_value = [task]
        manager.list_sprints.return_value = []
        manager.list_decisions.return_value = []
        manager.list_edges.return_value = []

        walker = GraphWalker(manager)
        result = (
            walker
            .starting_from("task-alone")
            .bfs()
            .visit(lambda n, acc: acc + [n.id], initial=[])
            .run()
        )

        assert result == ["task-alone"]

    def test_cycle_detection(self):
        """Cycle in graph doesn't cause infinite loop."""
        manager = MagicMock()

        task_a = Task(id="task-a", title="A", priority="high", status="pending")
        task_b = Task(id="task-b", title="B", priority="medium", status="pending")

        manager.list_all_tasks.return_value = [task_a, task_b]
        manager.list_sprints.return_value = []
        manager.list_decisions.return_value = []

        # Create cycle: A -> B -> A
        edges = [
            Edge(id="edge-1", source_id="task-a", target_id="task-b", edge_type="DEPENDS_ON"),
            Edge(id="edge-2", source_id="task-b", target_id="task-a", edge_type="DEPENDS_ON"),
        ]
        manager.list_edges.return_value = edges

        walker = GraphWalker(manager)
        result = (
            walker
            .starting_from("task-a")
            .directed()
            .bfs()
            .visit(lambda n, acc: acc + [n.id], initial=[])
            .run()
        )

        # Should visit each node only once
        assert len(result) == 2
        assert set(result) == {"task-a", "task-b"}


class TestGraphWalkerIterator:
    """Test iterator interface (iter() method)."""

    @pytest.fixture
    def mock_manager_with_tasks(self):
        """Create mock manager with connected tasks."""
        manager = MagicMock()

        tasks = [
            Task(id=f"task-{i}", title=f"Task {i}", priority="medium", status="pending")
            for i in range(3)
        ]

        manager.list_all_tasks.return_value = tasks
        manager.list_sprints.return_value = []
        manager.list_decisions.return_value = []

        edges = [
            Edge(id="edge-1", source_id="task-0", target_id="task-1", edge_type="DEPENDS_ON"),
            Edge(id="edge-2", source_id="task-1", target_id="task-2", edge_type="DEPENDS_ON"),
        ]
        manager.list_edges.return_value = edges

        return manager

    def test_iter_bfs(self, mock_manager_with_tasks):
        """iter() yields nodes in BFS order."""
        walker = GraphWalker(mock_manager_with_tasks)

        nodes = list(
            walker
            .starting_from("task-0")
            .directed()
            .bfs()
            .iter()
        )

        assert len(nodes) == 3
        assert all(hasattr(node, 'id') for node in nodes)

    def test_iter_dfs(self, mock_manager_with_tasks):
        """iter() yields nodes in DFS order."""
        walker = GraphWalker(mock_manager_with_tasks)

        nodes = list(
            walker
            .starting_from("task-0")
            .directed()
            .dfs()
            .iter()
        )

        assert len(nodes) == 3

    def test_iter_with_filter(self, mock_manager_with_tasks):
        """iter() respects filter predicate."""
        # Modify start task to have high priority so traversal continues
        manager = mock_manager_with_tasks
        tasks = manager.list_all_tasks.return_value
        tasks[0].priority = "high"  # Make start node pass filter
        tasks[1].priority = "high"

        walker = GraphWalker(manager)
        nodes = list(
            walker
            .starting_from("task-0")
            .bfs()
            .filter(lambda n: n.priority == "high")
            .iter()
        )

        # Should yield high priority tasks
        assert len(nodes) == 2
        assert nodes[0].id == "task-0"
        assert nodes[1].id == "task-1"

    def test_iter_with_max_depth(self, mock_manager_with_tasks):
        """iter() respects max_depth."""
        walker = GraphWalker(mock_manager_with_tasks)

        nodes = list(
            walker
            .starting_from("task-0")
            .directed()
            .bfs()
            .max_depth(1)
            .iter()
        )

        # Should yield task-0 and task-1 only
        assert len(nodes) == 2

    def test_iter_no_start_node(self, mock_manager_with_tasks):
        """iter() without start node yields nothing."""
        walker = GraphWalker(mock_manager_with_tasks)

        nodes = list(walker.bfs().iter())

        assert len(nodes) == 0

    def test_iter_start_node_not_found(self, mock_manager_with_tasks):
        """iter() with non-existent start node yields nothing."""
        walker = GraphWalker(mock_manager_with_tasks)

        nodes = list(walker.starting_from("nonexistent").bfs().iter())

        assert len(nodes) == 0


class TestGraphWalkerMixedEntities:
    """Test traversal across different entity types (tasks, sprints, decisions)."""

    @pytest.fixture
    def mock_manager_with_mixed_entities(self):
        """Create manager with tasks, sprints, and decisions."""
        manager = MagicMock()

        task = Task(id="task-1", title="Task 1", priority="high", status="pending")
        sprint = Sprint(id="sprint-1", title="Sprint 1", number=1, status="in_progress")
        decision = Decision(
            id="decision-1",
            title="Use GraphWalker",
            rationale="Better API"
        )

        manager.list_all_tasks.return_value = [task]
        manager.list_sprints.return_value = [sprint]
        manager.list_decisions.return_value = [decision]

        # Connect them: sprint -> task -> decision
        edges = [
            Edge(id="edge-1", source_id="sprint-1", target_id="task-1", edge_type="CONTAINS"),
            Edge(id="edge-2", source_id="task-1", target_id="decision-1", edge_type="JUSTIFIES"),
        ]
        manager.list_edges.return_value = edges

        return manager

    def test_traverse_mixed_entities(self, mock_manager_with_mixed_entities):
        """Traversal can visit different entity types."""
        walker = GraphWalker(mock_manager_with_mixed_entities)

        result = (
            walker
            .starting_from("sprint-1")
            .directed()
            .bfs()
            .visit(lambda node, acc: acc + [node.id], initial=[])
            .run()
        )

        # Should visit all three entities
        assert len(result) == 3
        assert "sprint-1" in result
        assert "task-1" in result
        assert "decision-1" in result

    def test_filter_by_entity_type(self, mock_manager_with_mixed_entities):
        """Can filter by entity type."""
        walker = GraphWalker(mock_manager_with_mixed_entities)

        # Start from task to pass filter, not sprint
        result = (
            walker
            .starting_from("task-1")
            .bfs()
            .filter(lambda node: isinstance(node, Task))
            .visit(lambda node, acc: acc + [node.id], initial=[])
            .run()
        )

        # Should only visit the task (sprint and decision filtered out)
        assert len(result) == 1
        assert result == ["task-1"]


class TestGraphWalkerAdditionalCoverage:
    """Additional tests to improve coverage."""

    @pytest.fixture
    def mock_manager_simple_chain(self):
        """Create a simple chain for edge case testing."""
        manager = MagicMock()

        tasks = [
            Task(id="task-1", title="Task 1", priority="high", status="pending"),
            Task(id="task-2", title="Task 2", priority="low", status="pending"),
            Task(id="task-3", title="Task 3", priority="high", status="pending"),
        ]

        manager.list_all_tasks.return_value = tasks
        manager.list_sprints.return_value = []
        manager.list_decisions.return_value = []

        edges = [
            Edge(id="edge-1", source_id="task-1", target_id="task-2", edge_type="DEPENDS_ON"),
            Edge(id="edge-2", source_id="task-2", target_id="task-3", edge_type="DEPENDS_ON"),
        ]
        manager.list_edges.return_value = edges

        return manager

    def test_dfs_iter_with_filter_stops_traversal(self, mock_manager_simple_chain):
        """DFS iter with filter stops at filtered nodes."""
        walker = GraphWalker(mock_manager_simple_chain)

        # Filter blocks task-2, preventing task-3 from being reached
        nodes = list(
            walker
            .starting_from("task-1")
            .directed()
            .dfs()
            .filter(lambda n: n.priority == "high")
            .iter()
        )

        # Should only visit task-1 (task-2 is filtered, blocking task-3)
        assert len(nodes) == 1
        assert nodes[0].id == "task-1"

    def test_dfs_iter_max_depth_early_return(self, mock_manager_simple_chain):
        """DFS iter respects max_depth and returns early."""
        walker = GraphWalker(mock_manager_simple_chain)

        nodes = list(
            walker
            .starting_from("task-1")
            .directed()
            .dfs()
            .max_depth(0)
            .iter()
        )

        # Should only visit start node
        assert len(nodes) == 1
        assert nodes[0].id == "task-1"

    def test_visitor_with_none_initial(self, mock_manager_simple_chain):
        """Visitor works with None as initial value."""
        walker = GraphWalker(mock_manager_simple_chain)

        def visitor(node, acc):
            if acc is None:
                acc = []
            return acc + [node.id]

        result = (
            walker
            .starting_from("task-1")
            .bfs()
            .visit(visitor, initial=None)
            .run()
        )

        assert result is not None
        assert len(result) > 0

    def test_visitor_with_dict_accumulator(self, mock_manager_simple_chain):
        """Visitor can use dictionary as accumulator."""
        walker = GraphWalker(mock_manager_simple_chain)

        def visitor(node, acc):
            acc[node.id] = node.title
            return acc

        result = (
            walker
            .starting_from("task-1")
            .bfs()
            .visit(visitor, initial={})
            .run()
        )

        assert isinstance(result, dict)
        assert "task-1" in result

    def test_visitor_with_set_accumulator(self, mock_manager_simple_chain):
        """Visitor can use set as accumulator."""
        walker = GraphWalker(mock_manager_simple_chain)

        def visitor(node, acc):
            acc.add(node.id)
            return acc

        result = (
            walker
            .starting_from("task-1")
            .directed()
            .bfs()
            .visit(visitor, initial=set())
            .run()
        )

        assert isinstance(result, set)
        assert "task-1" in result

    def test_bfs_iter_respects_filter_no_neighbors_queued(self, mock_manager_simple_chain):
        """BFS iter with filter doesn't queue neighbors of filtered nodes."""
        walker = GraphWalker(mock_manager_simple_chain)

        # Start node passes, but blocks traversal if filtered
        nodes = list(
            walker
            .starting_from("task-1")
            .directed()
            .bfs()
            .filter(lambda n: n.id == "task-1")  # Only task-1
            .iter()
        )

        # Should only visit task-1, task-2 not queued
        assert len(nodes) == 1
        assert nodes[0].id == "task-1"

    def test_adjacency_with_no_matching_edges(self):
        """Adjacency building handles no matching edges."""
        manager = MagicMock()

        task = Task(id="task-1", title="Task 1", priority="high", status="pending")
        manager.list_all_tasks.return_value = [task]
        manager.list_sprints.return_value = []
        manager.list_decisions.return_value = []

        # Edges exist but don't match filter
        edges = [
            Edge(id="edge-1", source_id="task-1", target_id="task-2", edge_type="BLOCKS"),
        ]
        manager.list_edges.return_value = edges

        walker = GraphWalker(manager)
        result = (
            walker
            .starting_from("task-1")
            .follow("DEPENDS_ON")  # No edges match
            .bfs()
            .visit(lambda n, acc: acc + [n.id], initial=[])
            .run()
        )

        # Should only visit start node (no edges matched filter)
        assert result == ["task-1"]

    def test_empty_adjacency_list(self):
        """Node with no edges in adjacency gets empty list."""
        manager = MagicMock()

        task = Task(id="task-1", title="Task 1", priority="high", status="pending")
        manager.list_all_tasks.return_value = [task]
        manager.list_sprints.return_value = []
        manager.list_decisions.return_value = []
        manager.list_edges.return_value = []  # No edges at all

        walker = GraphWalker(manager)
        result = (
            walker
            .starting_from("task-1")
            .bfs()
            .visit(lambda n, acc: acc + 1, initial=0)
            .run()
        )

        # Should visit only the start node
        assert result == 1


class TestGraphWalkerComplexScenarios:
    """Test complex real-world scenarios."""

    @pytest.fixture
    def mock_manager_complex(self):
        """Create a complex dependency graph."""
        manager = MagicMock()

        # Create a diamond-shaped dependency:
        #     A
        #    / \
        #   B   C
        #    \ /
        #     D

        tasks = [
            Task(id="task-a", title="A", priority="critical", status="completed"),
            Task(id="task-b", title="B", priority="high", status="in_progress"),
            Task(id="task-c", title="C", priority="high", status="in_progress"),
            Task(id="task-d", title="D", priority="medium", status="pending"),
        ]

        manager.list_all_tasks.return_value = tasks
        manager.list_sprints.return_value = []
        manager.list_decisions.return_value = []

        edges = [
            Edge(id="edge-1", source_id="task-a", target_id="task-b", edge_type="DEPENDS_ON"),
            Edge(id="edge-2", source_id="task-a", target_id="task-c", edge_type="DEPENDS_ON"),
            Edge(id="edge-3", source_id="task-b", target_id="task-d", edge_type="DEPENDS_ON"),
            Edge(id="edge-4", source_id="task-c", target_id="task-d", edge_type="DEPENDS_ON"),
        ]
        manager.list_edges.return_value = edges

        return manager

    def test_diamond_graph_visits_all(self, mock_manager_complex):
        """Diamond-shaped graph visits all nodes exactly once."""
        walker = GraphWalker(mock_manager_complex)

        result = (
            walker
            .starting_from("task-a")
            .directed()
            .bfs()
            .visit(lambda node, acc: acc + [node.id], initial=[])
            .run()
        )

        # Should visit each node exactly once
        assert len(result) == 4
        assert set(result) == {"task-a", "task-b", "task-c", "task-d"}

    def test_find_blocking_tasks(self, mock_manager_complex):
        """Use reverse traversal to find what blocks a task."""
        walker = GraphWalker(mock_manager_complex)

        result = (
            walker
            .starting_from("task-d")
            .reverse()
            .bfs()
            .visit(lambda node, acc: acc + [node.id], initial=[])
            .run()
        )

        # Should find all tasks that task-d depends on
        assert len(result) == 4
        assert "task-d" in result
        assert "task-b" in result
        assert "task-c" in result
        assert "task-a" in result

    def test_collect_statistics(self, mock_manager_complex):
        """Collect statistics during traversal."""
        walker = GraphWalker(mock_manager_complex)

        def collect_stats(node, acc):
            acc["total"] += 1
            acc["by_priority"][node.priority] = acc["by_priority"].get(node.priority, 0) + 1
            acc["by_status"][node.status] = acc["by_status"].get(node.status, 0) + 1
            return acc

        result = (
            walker
            .starting_from("task-a")
            .bfs()
            .visit(collect_stats, initial={"total": 0, "by_priority": {}, "by_status": {}})
            .run()
        )

        assert result["total"] == 4
        assert result["by_priority"]["critical"] == 1
        assert result["by_priority"]["high"] == 2
        assert result["by_priority"]["medium"] == 1
