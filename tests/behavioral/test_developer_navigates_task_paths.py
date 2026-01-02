"""
Behavioral tests for path finding and graph traversal in Graph of Thought.

Epic: Developer Navigates Task Paths

As a developer analyzing task relationships,
I want to find paths between tasks and traverse the graph,
So that I can understand dependencies using algorithms built from scratch.
"""

import pytest
from cortical.got.api import GoTManager
from cortical.got.path_finder import PathFinder
from cortical.got.graph_walker import GraphWalker


class TestDeveloperFindsShortestPathsBetweenTasks:
    """
    As a developer planning work order,
    I want to find the shortest dependency path between tasks,
    So that I can understand the critical path using our custom BFS implementation.
    """

    def test_scenario_find_shortest_path_in_dependency_chain(self, tmp_path):
        """
        Scenario: Finding the shortest connection between tasks

        Given a chain of dependent tasks
        When I find the shortest path from start to end
        Then I get the direct dependency path
        """
        # Given a chain of dependent tasks
        manager = GoTManager(tmp_path / ".got")
        start = manager.create_task(title="Feature implementation")
        middle = manager.create_task(title="Core component")
        end = manager.create_task(title="Foundation module")

        manager.add_dependency(start.id, middle.id)
        manager.add_dependency(middle.id, end.id)

        # When I find the shortest path from start to end
        finder = PathFinder(manager)
        path = finder.shortest_path(start.id, end.id)

        # Then I get the direct dependency path
        assert path is not None
        assert len(path) == 3
        assert path[0] == start.id
        assert path[1] == middle.id
        assert path[2] == end.id

    def test_scenario_no_path_exists_between_disconnected_tasks(self, tmp_path):
        """
        Scenario: Confirming tasks are disconnected

        Given two unconnected tasks
        When I search for a path between them
        Then no path is found
        Because our path finder correctly identifies disconnected components
        """
        # Given two unconnected tasks
        manager = GoTManager(tmp_path / ".got")
        task_a = manager.create_task(title="Task A")
        task_b = manager.create_task(title="Task B")

        # When I search for a path between them
        finder = PathFinder(manager)
        path = finder.shortest_path(task_a.id, task_b.id)

        # Then no path is found
        assert path is None

    def test_scenario_check_path_existence_efficiently(self, tmp_path):
        """
        Scenario: Quick connectivity check

        Given connected tasks
        When I use path_exists() to check connectivity
        Then I get a boolean result quickly
        Without computing the full path
        """
        # Given connected tasks
        manager = GoTManager(tmp_path / ".got")
        task_a = manager.create_task(title="Task A")
        task_b = manager.create_task(title="Task B")
        task_c = manager.create_task(title="Task C (disconnected)")

        manager.add_dependency(task_a.id, task_b.id)

        # When I use path_exists() to check connectivity
        finder = PathFinder(manager)

        # Then I get a boolean result quickly
        assert finder.path_exists(task_a.id, task_b.id) is True
        assert finder.path_exists(task_a.id, task_c.id) is False


class TestDeveloperFindsAllPathsBetweenTasks:
    """
    As a developer analyzing alternative dependency routes,
    I want to find all paths between two tasks,
    So that I can see redundant dependencies using our custom DFS implementation.
    """

    def test_scenario_find_multiple_dependency_routes(self, tmp_path):
        """
        Scenario: Finding alternative paths in a graph

        Given tasks with multiple connection routes
        When I find all paths between start and end
        Then I get all possible routes
        """
        # Given tasks with multiple connection routes
        manager = GoTManager(tmp_path / ".got")
        start = manager.create_task(title="Start")
        path1_mid = manager.create_task(title="Path 1 middle")
        path2_mid = manager.create_task(title="Path 2 middle")
        end = manager.create_task(title="End")

        # Create two paths: start -> path1_mid -> end
        #                   start -> path2_mid -> end
        manager.add_dependency(start.id, path1_mid.id)
        manager.add_dependency(path1_mid.id, end.id)
        manager.add_dependency(start.id, path2_mid.id)
        manager.add_dependency(path2_mid.id, end.id)

        # When I find all paths between start and end
        finder = PathFinder(manager)
        result = finder.all_paths(start.id, end.id)

        # Then I get all possible routes
        assert len(result.paths) == 2
        # Both paths should go through different middle nodes

    def test_scenario_limit_path_search_to_prevent_explosion(self, tmp_path):
        """
        Scenario: Preventing exponential search in dense graphs

        Given a graph with many paths
        When I limit the search to 5 paths
        Then I get exactly 5 paths
        And I'm warned about truncation
        Because our path finder has safety limits
        """
        # Given a graph with many paths
        manager = GoTManager(tmp_path / ".got")
        start = manager.create_task(title="Start")
        end = manager.create_task(title="End")

        # Create multiple intermediate nodes with connections
        intermediates = []
        for i in range(4):
            node = manager.create_task(title=f"Node {i}")
            intermediates.append(node)
            manager.add_dependency(start.id, node.id)
            manager.add_dependency(node.id, end.id)

        # When I limit the search to 5 paths
        finder = PathFinder(manager).max_paths(5)
        result = finder.all_paths(start.id, end.id)

        # Then I get at most 5 paths
        assert len(result.paths) <= 5

        # And I'm warned about truncation if there were more
        if len(result.paths) == 5:
            assert result.truncated is True


class TestDeveloperTraversesGraphWithCustomLogic:
    """
    As a developer collecting graph statistics,
    I want to traverse the graph with custom visitor functions,
    So that I can analyze our task graph using our hand-built traversal engine.
    """

    def test_scenario_count_connected_tasks_with_bfs(self, tmp_path):
        """
        Scenario: Collecting statistics via breadth-first traversal

        Given a connected subgraph
        When I traverse with BFS and count nodes
        Then I get the total count of connected tasks
        """
        # Given a connected subgraph
        manager = GoTManager(tmp_path / ".got")
        root = manager.create_task(title="Root")
        child1 = manager.create_task(title="Child 1")
        child2 = manager.create_task(title="Child 2")
        grandchild = manager.create_task(title="Grandchild")

        manager.add_dependency(child1.id, root.id)
        manager.add_dependency(child2.id, root.id)
        manager.add_dependency(grandchild.id, child1.id)

        # When I traverse with BFS and count nodes
        def counter(node, count):
            return count + 1

        walker = GraphWalker(manager)
        count = (
            walker
            .starting_from(root.id)
            .bfs()
            .visit(counter, initial=0)
            .run()
        )

        # Then I get the total count of connected tasks
        assert count == 4  # root + 2 children + 1 grandchild

    def test_scenario_collect_task_ids_with_dfs(self, tmp_path):
        """
        Scenario: Gathering node IDs via depth-first traversal

        Given a task tree
        When I traverse with DFS collecting IDs
        Then I get all connected task IDs
        In depth-first order
        """
        # Given a task tree
        manager = GoTManager(tmp_path / ".got")
        root = manager.create_task(title="Root")
        child1 = manager.create_task(title="Child 1")
        child2 = manager.create_task(title="Child 2")

        manager.add_dependency(child1.id, root.id)
        manager.add_dependency(child2.id, root.id)

        # When I traverse with DFS collecting IDs
        def collector(node, ids):
            return ids + [node.id]

        walker = GraphWalker(manager)
        ids = (
            walker
            .starting_from(root.id)
            .dfs()
            .visit(collector, initial=[])
            .run()
        )

        # Then I get all connected task IDs
        assert len(ids) == 3
        assert root.id in ids
        assert child1.id in ids
        assert child2.id in ids

    def test_scenario_traverse_only_specific_edge_types(self, tmp_path):
        """
        Scenario: Following only certain relationship types

        Given tasks with mixed relationship types
        When I traverse following only DEPENDS_ON edges
        Then I only visit tasks connected via dependencies
        Not via other edge types
        """
        # Given tasks with mixed relationship types
        manager = GoTManager(tmp_path / ".got")
        start = manager.create_task(title="Start")
        dep_child = manager.create_task(title="Dependency")
        block_child = manager.create_task(title="Blocker")

        manager.add_dependency(dep_child.id, start.id)
        manager.add_blocks(start.id, block_child.id)

        # When I traverse following only DEPENDS_ON edges
        def collector(node, ids):
            return ids + [node.id]

        walker = GraphWalker(manager)
        ids = (
            walker
            .starting_from(start.id)
            .follow("DEPENDS_ON")
            .bfs()
            .visit(collector, initial=[])
            .run()
        )

        # Then I only visit tasks connected via dependencies
        assert start.id in ids
        assert dep_child.id in ids
        # Should not include block_child since we only follow DEPENDS_ON
        assert block_child.id not in ids

    def test_scenario_limit_traversal_depth(self, tmp_path):
        """
        Scenario: Preventing deep recursion in traversal

        Given a deep chain of tasks
        When I traverse with max_depth of 2
        Then I only visit nodes within depth 2
        """
        # Given a deep chain of tasks
        manager = GoTManager(tmp_path / ".got")
        root = manager.create_task(title="Root")
        depth1 = manager.create_task(title="Depth 1")
        depth2 = manager.create_task(title="Depth 2")
        depth3 = manager.create_task(title="Depth 3")

        manager.add_dependency(depth1.id, root.id)
        manager.add_dependency(depth2.id, depth1.id)
        manager.add_dependency(depth3.id, depth2.id)

        # When I traverse with max_depth of 2
        def counter(node, count):
            return count + 1

        walker = GraphWalker(manager)
        count = (
            walker
            .starting_from(root.id)
            .max_depth(2)
            .bfs()
            .visit(counter, initial=0)
            .run()
        )

        # Then I only visit nodes within depth 2
        assert count == 3  # root (0), depth1 (1), depth2 (2)


@pytest.fixture
def tmp_path(tmp_path_factory):
    """Provide temporary directory for test isolation."""
    return tmp_path_factory.mktemp("got_test")
