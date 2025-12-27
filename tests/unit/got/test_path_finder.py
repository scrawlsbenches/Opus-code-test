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
            manager.add_edge(start.id, t.id, "DEPENDS_ON")

        # Connect all L1 to all L2
        for t1 in layer1:
            for t2 in layer2:
                manager.add_edge(t1.id, t2.id, "DEPENDS_ON")

        # Connect all L2 to end
        for t in layer2:
            manager.add_edge(t.id, end.id, "DEPENDS_ON")

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

        # PathSearchResult is list-like (supports iteration, len, indexing)
        from cortical.got.path_finder import PathSearchResult
        assert isinstance(paths, PathSearchResult)
        assert len(paths) >= 0  # Can use len()
        for path in paths:  # Can iterate
            assert isinstance(path, list)

    def test_all_paths_same_node(self, manager_with_graph):
        """all_paths should return single-node path when from==to."""
        manager, tasks = manager_with_graph
        finder = PathFinder(manager)

        result = finder.all_paths(tasks[0].id, tasks[0].id)

        # PathSearchResult has .paths attribute
        assert result.paths == [[tasks[0].id]]
        # Also works via backwards-compatible iteration
        assert list(result) == [[tasks[0].id]]

    def test_max_paths_limit_during_dfs(self, manager_with_dense_graph):
        """Test that max_paths limit is respected during DFS traversal."""
        manager, start_id, end_id = manager_with_dense_graph

        # Use a very small limit to ensure we hit the limit during DFS
        finder = PathFinder(manager).max_paths(1)
        paths = finder.all_paths(start_id, end_id)

        # Should stop after finding just 1 path
        assert len(paths) == 1

    def test_truncation_metadata_on_limit_hit(self, manager_with_dense_graph):
        """Test that PathSearchResult reports truncation when limits are hit."""
        manager, start_id, end_id = manager_with_dense_graph

        # Use a very small limit to trigger truncation
        finder = PathFinder(manager).max_paths(1)
        result = finder.all_paths(start_id, end_id)

        # Should indicate truncation occurred
        assert result.truncated is True
        assert result.truncation_reason == "max_paths"
        assert result.limit_value == 1
        assert result.paths_found == 1

    def test_no_truncation_when_all_paths_found(self, manager_with_graph):
        """Test that PathSearchResult shows no truncation when all paths found."""
        manager, tasks = manager_with_graph

        # Linear chain should have exactly 1 path, well under default limits
        finder = PathFinder(manager)
        result = finder.all_paths(tasks[0].id, tasks[-1].id)

        # Should NOT indicate truncation
        assert result.truncated is False
        assert result.truncation_reason is None
        assert result.limit_value is None


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
                    manager.add_edge(tasks[i].id, tasks[j].id, "RELATES_TO")

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


class TestShortestPath:
    """Test shortest_path method."""

    @pytest.fixture
    def manager_with_simple_graph(self):
        """Create a simple graph for testing."""
        temp_dir = tempfile.mkdtemp()
        got_dir = Path(temp_dir) / ".got"
        manager = GoTManager(got_dir)

        # Create linear chain: A -> B -> C -> D
        tasks = []
        for i, name in enumerate(["A", "B", "C", "D"]):
            task = manager.create_task(f"Task {name}", priority="medium")
            tasks.append(task)

        # Create edges
        for i in range(len(tasks) - 1):
            manager.add_edge(tasks[i].id, tasks[i + 1].id, "DEPENDS_ON")

        yield manager, tasks
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_shortest_path_same_node(self, manager_with_simple_graph):
        """shortest_path should return single-node path when from==to."""
        manager, tasks = manager_with_simple_graph
        finder = PathFinder(manager)

        path = finder.shortest_path(tasks[0].id, tasks[0].id)

        assert path == [tasks[0].id]

    def test_shortest_path_exists(self, manager_with_simple_graph):
        """shortest_path should find path when one exists."""
        manager, tasks = manager_with_simple_graph
        finder = PathFinder(manager)

        path = finder.shortest_path(tasks[0].id, tasks[3].id)

        assert path is not None
        assert len(path) == 4
        assert path[0] == tasks[0].id
        assert path[-1] == tasks[3].id

    def test_shortest_path_no_path(self, manager_with_simple_graph):
        """shortest_path should return None when no path exists."""
        manager, tasks = manager_with_simple_graph

        # Create disconnected node
        isolated = manager.create_task("Isolated", priority="low")

        finder = PathFinder(manager)
        path = finder.shortest_path(tasks[0].id, isolated.id)

        assert path is None

    def test_shortest_path_respects_max_length(self, manager_with_simple_graph):
        """shortest_path should respect max_length limit."""
        manager, tasks = manager_with_simple_graph

        # Path from A to D is 4 nodes, set limit to 3
        finder = PathFinder(manager).max_length(3)
        path = finder.shortest_path(tasks[0].id, tasks[3].id)

        assert path is None

    def test_shortest_path_directed_mode(self, manager_with_simple_graph):
        """shortest_path should work in directed mode."""
        manager, tasks = manager_with_simple_graph
        finder = PathFinder(manager).directed()

        # Forward direction should work
        path_forward = finder.shortest_path(tasks[0].id, tasks[2].id)
        assert path_forward is not None

        # Reverse direction should NOT work in directed mode
        finder_reverse = PathFinder(manager).directed()
        path_reverse = finder_reverse.shortest_path(tasks[2].id, tasks[0].id)
        assert path_reverse is None


class TestPathExists:
    """Test path_exists method."""

    @pytest.fixture
    def manager_with_branches(self):
        """Create a graph with branches."""
        temp_dir = tempfile.mkdtemp()
        got_dir = Path(temp_dir) / ".got"
        manager = GoTManager(got_dir)

        # Create: A -> B -> C
        #         A -> D -> E
        tasks = {name: manager.create_task(f"Task {name}", priority="medium")
                 for name in ["A", "B", "C", "D", "E"]}

        manager.add_edge(tasks["A"].id, tasks["B"].id, "DEPENDS_ON")
        manager.add_edge(tasks["B"].id, tasks["C"].id, "DEPENDS_ON")
        manager.add_edge(tasks["A"].id, tasks["D"].id, "DEPENDS_ON")
        manager.add_edge(tasks["D"].id, tasks["E"].id, "DEPENDS_ON")

        yield manager, tasks
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_path_exists_true(self, manager_with_branches):
        """path_exists should return True when path exists."""
        manager, tasks = manager_with_branches
        finder = PathFinder(manager)

        assert finder.path_exists(tasks["A"].id, tasks["C"].id)
        assert finder.path_exists(tasks["A"].id, tasks["E"].id)

    def test_path_exists_false(self, manager_with_branches):
        """path_exists should return False when no path exists."""
        manager, tasks = manager_with_branches

        # Create isolated node
        isolated = manager.create_task("Isolated", priority="low")

        finder = PathFinder(manager)
        assert not finder.path_exists(tasks["A"].id, isolated.id)

    def test_path_exists_same_node(self, manager_with_branches):
        """path_exists should return True for same node."""
        manager, tasks = manager_with_branches
        finder = PathFinder(manager)

        assert finder.path_exists(tasks["A"].id, tasks["A"].id)


class TestReachableFrom:
    """Test reachable_from method."""

    @pytest.fixture
    def manager_with_components(self):
        """Create graph with multiple components."""
        temp_dir = tempfile.mkdtemp()
        got_dir = Path(temp_dir) / ".got"
        manager = GoTManager(got_dir)

        # Component 1: A -> B -> C
        comp1 = [manager.create_task(f"C1-{i}", priority="high") for i in range(3)]
        for i in range(len(comp1) - 1):
            manager.add_edge(comp1[i].id, comp1[i + 1].id, "DEPENDS_ON")

        # Component 2: D -> E (isolated)
        comp2 = [manager.create_task(f"C2-{i}", priority="low") for i in range(2)]
        manager.add_edge(comp2[0].id, comp2[1].id, "DEPENDS_ON")

        yield manager, comp1, comp2
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_reachable_from_basic(self, manager_with_components):
        """reachable_from should find all connected nodes."""
        manager, comp1, comp2 = manager_with_components
        finder = PathFinder(manager)

        reachable = finder.reachable_from(comp1[0].id)

        # Should reach all of component 1 (including start node)
        assert len(reachable) == 3
        assert all(task.id in reachable for task in comp1)

        # Should NOT reach component 2
        assert all(task.id not in reachable for task in comp2)

    def test_reachable_from_single_node(self, manager_with_components):
        """reachable_from on isolated node should return just itself."""
        manager, comp1, comp2 = manager_with_components

        # Create truly isolated node
        isolated = manager.create_task("Isolated", priority="medium")

        finder = PathFinder(manager)
        reachable = finder.reachable_from(isolated.id)

        assert reachable == {isolated.id}

    def test_reachable_from_with_edge_filter(self, manager_with_components):
        """reachable_from should respect edge type filters."""
        manager, comp1, comp2 = manager_with_components

        # Add different edge type
        special = manager.create_task("Special", priority="high")
        manager.add_edge(comp1[0].id, special.id, "REFERENCES")

        # Filter to only DEPENDS_ON edges
        finder = PathFinder(manager).via_edges("DEPENDS_ON")
        reachable = finder.reachable_from(comp1[0].id)

        # Should NOT include special node
        assert special.id not in reachable

        # Filter to only REFERENCES edges
        finder_special = PathFinder(manager).via_edges("REFERENCES")
        reachable_special = finder_special.reachable_from(comp1[0].id)

        # Should include special but not the rest of chain
        assert special.id in reachable_special
        assert comp1[1].id not in reachable_special


class TestConnectedComponents:
    """Test connected_components method."""

    @pytest.fixture
    def manager_with_multiple_components(self):
        """Create graph with distinct components."""
        temp_dir = tempfile.mkdtemp()
        got_dir = Path(temp_dir) / ".got"
        manager = GoTManager(got_dir)

        # Component 1: 3 tasks in a chain
        comp1 = [manager.create_task(f"A{i}", priority="high") for i in range(3)]
        manager.add_edge(comp1[0].id, comp1[1].id, "DEPENDS_ON")
        manager.add_edge(comp1[1].id, comp1[2].id, "DEPENDS_ON")

        # Component 2: 2 tasks
        comp2 = [manager.create_task(f"B{i}", priority="medium") for i in range(2)]
        manager.add_edge(comp2[0].id, comp2[1].id, "DEPENDS_ON")

        # Component 3: 1 isolated task
        comp3 = [manager.create_task("C0", priority="low")]

        yield manager, comp1, comp2, comp3
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_connected_components_count(self, manager_with_multiple_components):
        """connected_components should find all components."""
        manager, comp1, comp2, comp3 = manager_with_multiple_components
        finder = PathFinder(manager)

        components = finder.connected_components()

        assert len(components) == 3

    def test_connected_components_content(self, manager_with_multiple_components):
        """connected_components should group nodes correctly."""
        manager, comp1, comp2, comp3 = manager_with_multiple_components
        finder = PathFinder(manager)

        components = finder.connected_components()

        # Check sizes
        sizes = sorted([len(c) for c in components])
        assert sizes == [1, 2, 3]

        # Check that nodes are in correct components
        all_nodes = set()
        for component in components:
            all_nodes.update(component)

        assert all(task.id in all_nodes for task in comp1)
        assert all(task.id in all_nodes for task in comp2)
        assert all(task.id in all_nodes for task in comp3)


class TestEdgeTypeFiltering:
    """Test via_edges edge type filtering."""

    @pytest.fixture
    def manager_with_mixed_edges(self):
        """Create graph with different edge types."""
        temp_dir = tempfile.mkdtemp()
        got_dir = Path(temp_dir) / ".got"
        manager = GoTManager(got_dir)

        # Create tasks
        tasks = [manager.create_task(f"Task {i}", priority="medium") for i in range(4)]

        # Create different edge types
        manager.add_edge(tasks[0].id, tasks[1].id, "DEPENDS_ON")
        manager.add_edge(tasks[1].id, tasks[2].id, "BLOCKS")
        manager.add_edge(tasks[2].id, tasks[3].id, "DEPENDS_ON")

        yield manager, tasks
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_via_edges_filters_correctly(self, manager_with_mixed_edges):
        """via_edges should only follow specified edge types."""
        manager, tasks = manager_with_mixed_edges

        # Only follow DEPENDS_ON edges
        finder = PathFinder(manager).via_edges("DEPENDS_ON")
        path = finder.shortest_path(tasks[0].id, tasks[3].id)

        # Should NOT find path (blocked by BLOCKS edge in middle)
        assert path is None

    def test_via_edges_multiple_types(self, manager_with_mixed_edges):
        """via_edges should accept multiple edge types."""
        manager, tasks = manager_with_mixed_edges

        # Follow both DEPENDS_ON and BLOCKS
        finder = PathFinder(manager).via_edges("DEPENDS_ON", "BLOCKS")
        path = finder.shortest_path(tasks[0].id, tasks[3].id)

        # Should find path now
        assert path is not None
        assert len(path) == 4


class TestAllNodeTypes:
    """Test that all entity types are included in graph operations."""

    @pytest.fixture
    def manager_with_all_types(self):
        """Create manager with tasks, sprints, and decisions."""
        temp_dir = tempfile.mkdtemp()
        got_dir = Path(temp_dir) / ".got"
        manager = GoTManager(got_dir)

        # Create different entity types
        task = manager.create_task("Task", priority="high")
        sprint = manager.create_sprint("Sprint 1", number=1)
        decision = manager.log_decision("Use Python", rationale="Best fit")

        # Connect them
        manager.add_edge(task.id, sprint.id, "PART_OF")
        manager.add_edge(decision.id, task.id, "REFERENCES")

        yield manager, task, sprint, decision
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_connected_components_includes_all_types(self, manager_with_all_types):
        """connected_components should include all entity types."""
        manager, task, sprint, decision = manager_with_all_types
        finder = PathFinder(manager)

        components = finder.connected_components()

        # All should be in one component
        assert len(components) == 1
        component = components[0]

        assert task.id in component
        assert sprint.id in component
        assert decision.id in component

    def test_shortest_path_across_types(self, manager_with_all_types):
        """shortest_path should work across different entity types."""
        manager, task, sprint, decision = manager_with_all_types
        finder = PathFinder(manager)

        path = finder.shortest_path(decision.id, sprint.id)

        assert path is not None
        assert decision.id in path
        assert task.id in path
        assert sprint.id in path


class TestPathFinderExplain:
    """Test PathFinder.explain() method."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create manager for explain tests."""
        got_dir = tmp_path / ".got"
        manager = GoTManager(got_dir)
        manager.create_task("T1", status="pending")
        manager.create_task("T2", status="pending")
        return manager

    def test_explain_returns_path_plan(self, manager):
        """explain() returns PathPlan with current config."""
        from cortical.got.path_finder import PathFinder, PathPlan

        plan = PathFinder(manager).via_edges("DEPENDS_ON").max_paths(50).explain()

        assert isinstance(plan, PathPlan)
        assert plan.edge_types == ["DEPENDS_ON"]
        assert plan.max_paths == 50

    def test_explain_shows_defaults(self, manager):
        """explain() shows effective defaults."""
        from cortical.got.path_finder import PathFinder

        plan = PathFinder(manager).explain()

        # Should show default limits
        assert plan.max_paths == PathFinder.DEFAULT_MAX_PATHS
        assert plan.max_length == PathFinder.DEFAULT_MAX_LENGTH
        assert plan.bidirectional is True

    def test_explain_str_output(self, manager):
        """explain() produces readable string."""
        from cortical.got.path_finder import PathFinder

        plan = PathFinder(manager).via_edges("BLOCKS").directed().explain()
        output = str(plan)

        assert "Path Finding Plan" in output
        assert "BLOCKS" in output
        assert "Directed" in output
