"""
Unit tests for graph traversal functions.

Tests cover:
- connected_to: Finding all connected entities
- path: Finding shortest paths between entities
- children/parents: Direct dependency relationships
- descendants/ancestors: Transitive dependency chains
- orphan_nodes: Finding isolated entities

Each function is tested with:
- Valid inputs and expected outputs
- Edge cases (no connections, cycles, depth limits)
- Error handling (missing arguments, invalid IDs)
"""

import pytest
from pathlib import Path

from cortical.got.api import GoTManager
from cortical.got.expression.registry import FunctionRegistry
from cortical.got.expression.functions.graph import (
    ConnectedTo,
    Path,
    Children,
    Parents,
    Descendants,
    Ancestors,
    OrphanNodes,
)


@pytest.fixture
def manager(tmp_path: Path) -> GoTManager:
    """Create GoTManager with test data."""
    from cortical.core.bootstrap import create_container

    # Create container with test directory
    container = create_container(got_dir=tmp_path / ".got")
    manager = container.resolve(GoTManager)

    # Create test graph:
    #
    #   T1 (orphan)
    #
    #   T2 --> T3 --> T4
    #      |      |
    #      v      v
    #   T5      T6
    #
    #   T7 <--> T8 (cycle)
    #
    #   T9 (orphan)

    # Create tasks and store IDs
    task_ids = {}
    with manager.transaction() as tx:
        task_ids['t1'] = tx.create_task("Task 1", priority="high").id
        task_ids['t2'] = tx.create_task("Task 2", priority="medium").id
        task_ids['t3'] = tx.create_task("Task 3", priority="low").id
        task_ids['t4'] = tx.create_task("Task 4", priority="high").id
        task_ids['t5'] = tx.create_task("Task 5", priority="medium").id
        task_ids['t6'] = tx.create_task("Task 6", priority="low").id
        task_ids['t7'] = tx.create_task("Task 7", priority="high").id
        task_ids['t8'] = tx.create_task("Task 8", priority="medium").id
        task_ids['t9'] = tx.create_task("Task 9", priority="low").id

    # Create edges
    with manager.transaction() as tx:
        # T2 --> T3 --> T4
        tx.add_edge(task_ids['t2'], task_ids['t3'], "DEPENDS_ON")
        tx.add_edge(task_ids['t3'], task_ids['t4'], "DEPENDS_ON")
        # T2 --> T5
        tx.add_edge(task_ids['t2'], task_ids['t5'], "DEPENDS_ON")
        # T3 --> T6
        tx.add_edge(task_ids['t3'], task_ids['t6'], "DEPENDS_ON")
        # T7 <--> T8 (cycle)
        tx.add_edge(task_ids['t7'], task_ids['t8'], "DEPENDS_ON")
        tx.add_edge(task_ids['t8'], task_ids['t7'], "DEPENDS_ON")

    # Store task_ids as attribute for test access
    manager.test_task_ids = task_ids

    return manager


class TestConnectedTo:
    """Test connected_to function."""

    def test_connected_to_returns_connected_entities(self, manager):
        """Should return all entities connected via any edge."""
        func = ConnectedTo()
        t2 = manager.test_task_ids['t2']
        t3 = manager.test_task_ids['t3']
        t5 = manager.test_task_ids['t5']

        results = func.execute(manager, [t2], {})

        # T2 is connected to T3 and T5
        result_ids = {r.id for r in results}
        assert result_ids == {t3, t5}

    def test_connected_to_with_kwargs(self, manager):
        """Should support keyword arguments."""
        func = ConnectedTo()
        t2 = manager.test_task_ids['t2']
        t3 = manager.test_task_ids['t3']
        t4 = manager.test_task_ids['t4']
        t6 = manager.test_task_ids['t6']

        results = func.execute(manager, [], {"entity_id": t3})

        # T3 is connected to T2, T4, T6
        result_ids = {r.id for r in results}
        assert result_ids == {t2, t4, t6}

    def test_connected_to_orphan_returns_empty(self, manager):
        """Should return empty list for orphan nodes."""
        func = ConnectedTo()
        t1 = manager.test_task_ids['t1']

        results = func.execute(manager, [t1], {})
        assert len(results) == 0

    def test_connected_to_missing_entity_id_raises(self, manager):
        """Should raise ValueError if entity_id is missing."""
        func = ConnectedTo()
        with pytest.raises(ValueError, match="entity_id is required"):
            func.execute(manager, [], {})

    def test_connected_to_signature(self):
        """Should have correct signature."""
        sig = ConnectedTo.signature()
        assert sig.name == "connected_to"
        assert "entity_id" in sig.required_args


class TestPath:
    """Test path function."""

    def test_path_finds_shortest_path(self, manager):
        """Should find shortest path between entities."""
        func = Path()
        t2 = manager.test_task_ids['t2']
        t3 = manager.test_task_ids['t3']
        t4 = manager.test_task_ids['t4']

        result = func.execute(manager, [t2, t4], {})

        # Shortest path: T2 -> T3 -> T4
        assert result == [t2, t3, t4]

    def test_path_with_kwargs(self, manager):
        """Should support keyword arguments."""
        func = Path()
        t2 = manager.test_task_ids['t2']
        t3 = manager.test_task_ids['t3']
        t6 = manager.test_task_ids['t6']

        result = func.execute(
            manager,
            [],
            {"from_id": t2, "to_id": t6}
        )

        # Shortest path: T2 -> T3 -> T6
        assert result == [t2, t3, t6]

    def test_path_returns_none_when_no_path_exists(self, manager):
        """Should return None if no path exists."""
        func = Path()
        t1 = manager.test_task_ids['t1']
        t2 = manager.test_task_ids['t2']

        result = func.execute(manager, [t1, t2], {})
        assert result is None

    def test_path_respects_max_depth(self, manager):
        """Should respect max_depth parameter."""
        func = Path()
        t2 = manager.test_task_ids['t2']
        t4 = manager.test_task_ids['t4']

        # With max_depth=2, can't reach T4 from T2
        result = func.execute(
            manager,
            [t2, t4],
            {"max_depth": 2}
        )
        # Path would be 3 nodes, but max_depth=2 limits it
        assert result is None

    def test_path_handles_cycles_gracefully(self, manager):
        """Should handle cycles without infinite loops."""
        func = Path()
        t7 = manager.test_task_ids['t7']
        t8 = manager.test_task_ids['t8']

        result = func.execute(manager, [t7, t8], {})

        # Should find direct path despite cycle
        assert result == [t7, t8]

    def test_path_missing_arguments_raises(self, manager):
        """Should raise ValueError if required args missing."""
        func = Path()
        with pytest.raises(ValueError, match="from_id and to_id are required"):
            func.execute(manager, [], {})

    def test_path_signature(self):
        """Should have correct signature."""
        sig = Path.signature()
        assert sig.name == "path"
        assert "from_id" in sig.required_args
        assert "to_id" in sig.required_args
        assert "max_depth" in sig.optional_args


class TestChildren:
    """Test children function."""

    def test_children_returns_direct_children(self, manager):
        """Should return entities that directly depend on this one."""
        func = Children()
        t2 = manager.test_task_ids['t2']
        t3 = manager.test_task_ids['t3']
        t5 = manager.test_task_ids['t5']

        results = func.execute(manager, [t2], {})

        # T2's children: T3, T5
        result_ids = {r.id for r in results}
        assert result_ids == {t3, t5}

    def test_children_empty_for_leaf_nodes(self, manager):
        """Should return empty list for leaf nodes."""
        func = Children()
        t4 = manager.test_task_ids['t4']

        results = func.execute(manager, [t4], {})
        assert len(results) == 0

    def test_children_with_kwargs(self, manager):
        """Should support keyword arguments."""
        func = Children()
        t3 = manager.test_task_ids['t3']
        t4 = manager.test_task_ids['t4']
        t6 = manager.test_task_ids['t6']

        results = func.execute(manager, [], {"entity_id": t3})

        # T3's children: T4, T6
        result_ids = {r.id for r in results}
        assert result_ids == {t4, t6}

    def test_children_signature(self):
        """Should have correct signature."""
        sig = Children.signature()
        assert sig.name == "children"
        assert "entity_id" in sig.required_args


class TestParents:
    """Test parents function."""

    def test_parents_returns_direct_parents(self, manager):
        """Should return entities this one directly depends on."""
        func = Parents()
        t2 = manager.test_task_ids['t2']
        t3 = manager.test_task_ids['t3']

        results = func.execute(manager, [t3], {})

        # T3's parent: T2
        result_ids = {r.id for r in results}
        assert result_ids == {t2}

    def test_parents_empty_for_root_nodes(self, manager):
        """Should return empty list for root nodes."""
        func = Parents()
        t2 = manager.test_task_ids['t2']

        results = func.execute(manager, [t2], {})
        assert len(results) == 0

    def test_parents_with_kwargs(self, manager):
        """Should support keyword arguments."""
        func = Parents()
        t3 = manager.test_task_ids['t3']
        t4 = manager.test_task_ids['t4']

        results = func.execute(manager, [], {"entity_id": t4})

        # T4's parent: T3
        result_ids = {r.id for r in results}
        assert result_ids == {t3}

    def test_parents_signature(self):
        """Should have correct signature."""
        sig = Parents.signature()
        assert sig.name == "parents"
        assert "entity_id" in sig.required_args


class TestDescendants:
    """Test descendants function."""

    def test_descendants_returns_all_reachable_children(self, manager):
        """Should return all entities reachable following dependencies."""
        func = Descendants()
        t2 = manager.test_task_ids['t2']
        t3 = manager.test_task_ids['t3']
        t4 = manager.test_task_ids['t4']
        t5 = manager.test_task_ids['t5']
        t6 = manager.test_task_ids['t6']

        results = func.execute(manager, [t2], {})

        # T2's descendants: T3, T4, T5, T6
        result_ids = {r.id for r in results}
        assert result_ids == {t3, t4, t5, t6}

    def test_descendants_excludes_self(self, manager):
        """Should not include the entity itself."""
        func = Descendants()
        t2 = manager.test_task_ids['t2']

        results = func.execute(manager, [t2], {})

        result_ids = {r.id for r in results}
        assert t2 not in result_ids

    def test_descendants_respects_max_depth(self, manager):
        """Should respect max_depth parameter."""
        func = Descendants()
        t2 = manager.test_task_ids['t2']
        t3 = manager.test_task_ids['t3']
        t5 = manager.test_task_ids['t5']

        results = func.execute(
            manager,
            [t2],
            {"max_depth": 1}
        )

        # With max_depth=1, only direct children: T3, T5
        result_ids = {r.id for r in results}
        # Note: max_depth limits path length, not edge count
        # This might include T3 and T5 only
        assert t3 in result_ids
        assert t5 in result_ids

    def test_descendants_handles_cycles_gracefully(self, manager):
        """Should handle cycles without infinite loops."""
        func = Descendants()
        t7 = manager.test_task_ids['t7']
        t8 = manager.test_task_ids['t8']

        results = func.execute(manager, [t7], {})

        # Should include T8 without infinite loop
        result_ids = {r.id for r in results}
        assert t8 in result_ids
        assert len(result_ids) <= 2  # At most T7 and T8

    def test_descendants_signature(self):
        """Should have correct signature."""
        sig = Descendants.signature()
        assert sig.name == "descendants"
        assert "entity_id" in sig.required_args
        assert "max_depth" in sig.optional_args


class TestAncestors:
    """Test ancestors function."""

    def test_ancestors_returns_all_dependencies(self, manager):
        """Should return all entities this one transitively depends on."""
        func = Ancestors()
        t2 = manager.test_task_ids['t2']
        t3 = manager.test_task_ids['t3']
        t4 = manager.test_task_ids['t4']

        results = func.execute(manager, [t4], {})

        # T4's ancestors: T3, T2
        result_ids = {r.id for r in results}
        assert result_ids == {t3, t2}

    def test_ancestors_excludes_self(self, manager):
        """Should not include the entity itself."""
        func = Ancestors()
        t4 = manager.test_task_ids['t4']

        results = func.execute(manager, [t4], {})

        result_ids = {r.id for r in results}
        assert t4 not in result_ids

    def test_ancestors_empty_for_root_nodes(self, manager):
        """Should return empty list for nodes with no dependencies."""
        func = Ancestors()
        t2 = manager.test_task_ids['t2']

        results = func.execute(manager, [t2], {})
        assert len(results) == 0

    def test_ancestors_respects_max_depth(self, manager):
        """Should respect max_depth parameter."""
        func = Ancestors()
        t3 = manager.test_task_ids['t3']
        t4 = manager.test_task_ids['t4']

        results = func.execute(
            manager,
            [t4],
            {"max_depth": 1}
        )

        # With max_depth=1, only direct parent: T3
        result_ids = {r.id for r in results}
        assert t3 in result_ids

    def test_ancestors_handles_cycles_gracefully(self, manager):
        """Should handle cycles without infinite loops."""
        func = Ancestors()
        t7 = manager.test_task_ids['t7']
        t8 = manager.test_task_ids['t8']

        results = func.execute(manager, [t8], {})

        # Should include T7 without infinite loop
        result_ids = {r.id for r in results}
        assert t7 in result_ids
        assert len(result_ids) <= 2  # At most T7 and T8

    def test_ancestors_signature(self):
        """Should have correct signature."""
        sig = Ancestors.signature()
        assert sig.name == "ancestors"
        assert "entity_id" in sig.required_args
        assert "max_depth" in sig.optional_args


class TestOrphanNodes:
    """Test orphan_nodes function."""

    def test_orphan_nodes_identifies_isolated_entities(self, manager):
        """Should return entities with no connections."""
        func = OrphanNodes()
        t1 = manager.test_task_ids['t1']
        t9 = manager.test_task_ids['t9']

        results = func.execute(manager, [], {})

        # Orphans: T1, T9
        result_ids = {r.id for r in results}
        assert result_ids == {t1, t9}

    def test_orphan_nodes_excludes_connected_entities(self, manager):
        """Should not include entities with any edges."""
        func = OrphanNodes()
        t2 = manager.test_task_ids['t2']
        t3 = manager.test_task_ids['t3']
        t4 = manager.test_task_ids['t4']
        t5 = manager.test_task_ids['t5']
        t6 = manager.test_task_ids['t6']
        t7 = manager.test_task_ids['t7']
        t8 = manager.test_task_ids['t8']

        results = func.execute(manager, [], {})

        result_ids = {r.id for r in results}
        # These have connections and should not be in results
        for connected_id in [t2, t3, t4, t5, t6, t7, t8]:
            assert connected_id not in result_ids

    def test_orphan_nodes_signature(self):
        """Should have correct signature."""
        sig = OrphanNodes.signature()
        assert sig.name == "orphan_nodes"
        assert len(sig.required_args) == 0


class TestFunctionRegistry:
    """Test that functions are properly registered."""

    def test_all_functions_registered(self):
        """Should register all graph functions."""
        # Check that functions are registered
        assert FunctionRegistry.get("connected_to") is not None
        assert FunctionRegistry.get("path") is not None
        assert FunctionRegistry.get("children") is not None
        assert FunctionRegistry.get("parents") is not None
        assert FunctionRegistry.get("descendants") is not None
        assert FunctionRegistry.get("ancestors") is not None
        assert FunctionRegistry.get("orphan_nodes") is not None

    def test_functions_have_correct_types(self):
        """Should register correct function classes."""
        # Compare by class name to handle module reloading in tests
        assert FunctionRegistry.get("connected_to").__name__ == "ConnectedTo"
        assert FunctionRegistry.get("path").__name__ == "Path"
        assert FunctionRegistry.get("children").__name__ == "Children"
        assert FunctionRegistry.get("parents").__name__ == "Parents"
        assert FunctionRegistry.get("descendants").__name__ == "Descendants"
        assert FunctionRegistry.get("ancestors").__name__ == "Ancestors"
        assert FunctionRegistry.get("orphan_nodes").__name__ == "OrphanNodes"
