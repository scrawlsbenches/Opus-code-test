"""
Tests for PathFinder path limits.

Validates:
- Default limits are applied (max_paths=100, max_length=10)
- Custom limits can be set via fluent API
- Limits can be removed with None
- Exponential blowup is prevented
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from cortical.got import GoTManager
from cortical.got.path_finder import PathFinder


class TestPathFinderDefaults:
    """Test default limit values."""

    def test_default_max_paths(self):
        """Default max_paths should be 100."""
        assert PathFinder.DEFAULT_MAX_PATHS == 100

    def test_default_max_length(self):
        """Default max_length should be 10."""
        assert PathFinder.DEFAULT_MAX_LENGTH == 10


class TestPathFinderLimits:
    """Test path limit functionality."""

    @pytest.fixture
    def manager_with_graph(self):
        """Create a manager with a test graph."""
        temp_dir = tempfile.mkdtemp()
        got_dir = Path(temp_dir) / ".got"
        manager = GoTManager(got_dir)

        # Create a linear chain: A -> B -> C -> D -> E
        tasks = []
        for i in range(5):
            task = manager.create_task(f"Task {i}", priority="medium")
            tasks.append(task)

        # Create edges forming a chain
        for i in range(len(tasks) - 1):
            manager.add_edge(tasks[i].id, tasks[i + 1].id, "DEPENDS_ON")

        yield manager, tasks
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def manager_with_dense_graph(self):
        """Create a manager with a densely connected graph (many paths)."""
        temp_dir = tempfile.mkdtemp()
        got_dir = Path(temp_dir) / ".got"
        manager = GoTManager(got_dir)

        # Create tasks in layers to generate many paths
        # Layer 0: 1 node (start)
        # Layer 1: 3 nodes
        # Layer 2: 3 nodes
        # Layer 3: 1 node (end)
        # Connect each node to all nodes in next layer
        # This creates 3 * 3 = 9 paths

        start = manager.create_task("Start", priority="high")
        layer1 = [manager.create_task(f"L1-{i}", priority="medium") for i in range(3)]
        layer2 = [manager.create_task(f"L2-{i}", priority="medium") for i in range(3)]
        end = manager.create_task("End", priority="low")

        # Connect start to all L1
        for t in layer1:
            manager.add_edge(start.id, t.id, "NEXT")

        # Connect all L1 to all L2
        for t1 in layer1:
            for t2 in layer2:
                manager.add_edge(t1.id, t2.id, "NEXT")

        # Connect all L2 to end
        for t in layer2:
            manager.add_edge(t.id, end.id, "NEXT")

        yield manager, start.id, end.id
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_all_paths_with_defaults(self, manager_with_graph):
        """all_paths should apply default limits."""
        manager, tasks = manager_with_graph
        finder = PathFinder(manager)

        # Should use defaults (max_paths=100, max_length=10)
        paths = finder.all_paths(tasks[0].id, tasks[-1].id)

        # Should find the path
        assert len(paths) >= 1

    def test_max_paths_limit(self, manager_with_dense_graph):
        """max_paths should limit number of paths returned."""
        manager, start_id, end_id = manager_with_dense_graph

        # Without limit, would find 9 paths (3 * 3)
        # With limit of 5, should only return 5
        finder = PathFinder(manager).max_paths(5)
        paths = finder.all_paths(start_id, end_id)

        assert len(paths) <= 5

    def test_max_paths_none_removes_limit(self, manager_with_dense_graph):
        """max_paths(None) should remove the limit."""
        manager, start_id, end_id = manager_with_dense_graph

        # Remove limits explicitly, use directed to get predictable path count
        finder = PathFinder(manager).max_paths(None).max_length(None).directed()
        paths = finder.all_paths(start_id, end_id)

        # Should find all 9 paths (3 * 3 from layered structure)
        assert len(paths) == 9

    def test_max_length_limit(self, manager_with_graph):
        """max_length should limit path length."""
        manager, tasks = manager_with_graph

        # Path from first to last is 5 nodes
        # With max_length=3, should not find any path
        finder = PathFinder(manager).max_length(3)
        paths = finder.all_paths(tasks[0].id, tasks[-1].id)

        assert len(paths) == 0

    def test_max_length_allows_shorter_paths(self, manager_with_graph):
        """max_length should allow paths shorter than limit."""
        manager, tasks = manager_with_graph

        # Path from first to third is 3 nodes (A -> B -> C)
        # With max_length=5, should find it
        finder = PathFinder(manager).max_length(5)
        paths = finder.all_paths(tasks[0].id, tasks[2].id)

        assert len(paths) >= 1

    def test_combined_limits(self, manager_with_dense_graph):
        """Both limits can be set together."""
        manager, start_id, end_id = manager_with_dense_graph

        finder = PathFinder(manager).max_paths(3).max_length(5)
        paths = finder.all_paths(start_id, end_id)

        assert len(paths) <= 3
        for path in paths:
            assert len(path) <= 5

    def test_fluent_chaining(self, manager_with_graph):
        """Fluent methods should be chainable."""
        manager, tasks = manager_with_graph

        # Should be chainable without errors
        paths = (
            PathFinder(manager)
            .max_paths(50)
            .max_length(20)
            .via_edges("DEPENDS_ON")
            .all_paths(tasks[0].id, tasks[-1].id)
        )

        assert isinstance(paths, list)


class TestPathFinderSafety:
    """Test that limits prevent exponential blowup."""

    @pytest.fixture
    def manager_with_explosive_graph(self):
        """Create a graph that would explode without limits."""
        temp_dir = tempfile.mkdtemp()
        got_dir = Path(temp_dir) / ".got"
        manager = GoTManager(got_dir)

        # Create a complete graph with 6 nodes
        # This would have O(n!) paths without limits
        tasks = [manager.create_task(f"Node {i}", priority="medium") for i in range(6)]

        # Connect every node to every other node (complete graph)
        for i in range(len(tasks)):
            for j in range(len(tasks)):
                if i != j:
                    manager.add_edge(tasks[i].id, tasks[j].id, "CONNECTED")

        yield manager, tasks[0].id, tasks[-1].id
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_defaults_prevent_explosion(self, manager_with_explosive_graph):
        """Default limits should prevent exponential path explosion."""
        manager, start_id, end_id = manager_with_explosive_graph

        # With defaults, this should complete quickly
        import time
        start = time.time()
        paths = PathFinder(manager).all_paths(start_id, end_id)
        elapsed = time.time() - start

        # Should complete in reasonable time (< 1 second)
        assert elapsed < 1.0

        # Should be limited by max_paths default (100)
        assert len(paths) <= PathFinder.DEFAULT_MAX_PATHS

    def test_explicit_limits_work(self, manager_with_explosive_graph):
        """Explicit limits should also prevent explosion."""
        manager, start_id, end_id = manager_with_explosive_graph

        import time
        start = time.time()
        paths = PathFinder(manager).max_paths(10).all_paths(start_id, end_id)
        elapsed = time.time() - start

        # Should complete quickly
        assert elapsed < 0.5

        # Should respect explicit limit
        assert len(paths) <= 10
