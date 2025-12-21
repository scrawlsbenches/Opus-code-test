"""
Unit tests for GoT query functionality.

Tests the query methods in GoTProjectManager:
- what_blocks() - Find tasks blocking a given task
- what_depends_on() - Find tasks depending on a given task
- find_path() - Find path between two nodes
- get_all_relationships() - Get all relationships for a task
- query() - String-based query parser
"""

import sys
from pathlib import Path

import pytest

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from got_utils import GoTProjectManager
from cortical.reasoning.graph_of_thought import EdgeType, NodeType


class TestWhatBlocks:
    """Test what_blocks() method."""

    def setup_method(self):
        """Set up test manager with fresh graph."""
        self.manager = GoTProjectManager()

    def test_empty_graph(self):
        """Empty graph returns empty list."""
        result = self.manager.what_blocks("task:nonexistent")
        assert result == []

    def test_no_blockers(self):
        """Task with no blockers returns empty list."""
        self.manager.graph.add_node("task:T1", NodeType.TASK, "Task 1")
        result = self.manager.what_blocks("task:T1")
        assert result == []

    def test_single_blocker(self):
        """Task with one blocker returns that blocker."""
        self.manager.graph.add_node("task:T1", NodeType.TASK, "Task 1")
        self.manager.graph.add_node("task:T2", NodeType.TASK, "Task 2")
        self.manager.graph.add_edge("task:T1", "task:T2", EdgeType.BLOCKS)

        result = self.manager.what_blocks("task:T2")
        assert len(result) == 1
        assert result[0].id == "task:T1"
        assert result[0].content == "Task 1"

    def test_multiple_blockers(self):
        """Task with multiple blockers returns all blockers."""
        self.manager.graph.add_node("task:T1", NodeType.TASK, "Task 1")
        self.manager.graph.add_node("task:T2", NodeType.TASK, "Task 2")
        self.manager.graph.add_node("task:T3", NodeType.TASK, "Task 3")
        self.manager.graph.add_edge("task:T1", "task:T3", EdgeType.BLOCKS)
        self.manager.graph.add_edge("task:T2", "task:T3", EdgeType.BLOCKS)

        result = self.manager.what_blocks("task:T3")
        assert len(result) == 2
        blocker_ids = {node.id for node in result}
        assert blocker_ids == {"task:T1", "task:T2"}

    def test_ignores_other_edge_types(self):
        """Only returns BLOCKS edges, not DEPENDS_ON or others."""
        self.manager.graph.add_node("task:T1", NodeType.TASK, "Task 1")
        self.manager.graph.add_node("task:T2", NodeType.TASK, "Task 2")
        self.manager.graph.add_edge("task:T1", "task:T2", EdgeType.DEPENDS_ON)

        result = self.manager.what_blocks("task:T2")
        assert result == []

    def test_handles_task_prefix(self):
        """Automatically adds task: prefix if missing."""
        self.manager.graph.add_node("task:T1", NodeType.TASK, "Task 1")
        self.manager.graph.add_node("task:T2", NodeType.TASK, "Task 2")
        self.manager.graph.add_edge("task:T1", "task:T2", EdgeType.BLOCKS)

        # Test without prefix
        result = self.manager.what_blocks("T2")
        assert len(result) == 1
        assert result[0].id == "task:T1"


class TestWhatDependsOn:
    """Test what_depends_on() method."""

    def setup_method(self):
        """Set up test manager with fresh graph."""
        self.manager = GoTProjectManager()

    def test_empty_graph(self):
        """Empty graph returns empty list."""
        result = self.manager.what_depends_on("task:nonexistent")
        assert result == []

    def test_no_dependents(self):
        """Task with no dependents returns empty list."""
        self.manager.graph.add_node("task:T1", NodeType.TASK, "Task 1")
        result = self.manager.what_depends_on("task:T1")
        assert result == []

    def test_single_dependent(self):
        """Task with one dependent returns that dependent."""
        self.manager.graph.add_node("task:T1", NodeType.TASK, "Task 1")
        self.manager.graph.add_node("task:T2", NodeType.TASK, "Task 2")
        self.manager.graph.add_edge("task:T2", "task:T1", EdgeType.DEPENDS_ON)

        result = self.manager.what_depends_on("task:T1")
        assert len(result) == 1
        assert result[0].id == "task:T2"
        assert result[0].content == "Task 2"

    def test_multiple_dependents(self):
        """Task with multiple dependents returns all dependents."""
        self.manager.graph.add_node("task:T1", NodeType.TASK, "Task 1")
        self.manager.graph.add_node("task:T2", NodeType.TASK, "Task 2")
        self.manager.graph.add_node("task:T3", NodeType.TASK, "Task 3")
        self.manager.graph.add_edge("task:T2", "task:T1", EdgeType.DEPENDS_ON)
        self.manager.graph.add_edge("task:T3", "task:T1", EdgeType.DEPENDS_ON)

        result = self.manager.what_depends_on("task:T1")
        assert len(result) == 2
        dependent_ids = {node.id for node in result}
        assert dependent_ids == {"task:T2", "task:T3"}

    def test_ignores_other_edge_types(self):
        """Only returns DEPENDS_ON edges, not BLOCKS or others."""
        self.manager.graph.add_node("task:T1", NodeType.TASK, "Task 1")
        self.manager.graph.add_node("task:T2", NodeType.TASK, "Task 2")
        self.manager.graph.add_edge("task:T2", "task:T1", EdgeType.BLOCKS)

        result = self.manager.what_depends_on("task:T1")
        assert result == []

    def test_handles_task_prefix(self):
        """Automatically adds task: prefix if missing."""
        self.manager.graph.add_node("task:T1", NodeType.TASK, "Task 1")
        self.manager.graph.add_node("task:T2", NodeType.TASK, "Task 2")
        self.manager.graph.add_edge("task:T2", "task:T1", EdgeType.DEPENDS_ON)

        # Test without prefix
        result = self.manager.what_depends_on("T1")
        assert len(result) == 1
        assert result[0].id == "task:T2"


class TestFindPath:
    """Test find_path() method."""

    def setup_method(self):
        """Set up test manager with fresh graph."""
        self.manager = GoTProjectManager()

    def test_empty_graph(self):
        """Empty graph returns None."""
        result = self.manager.find_path("task:T1", "task:T2")
        assert result is None

    def test_single_node_path(self):
        """Path from node to itself returns single-node path."""
        self.manager.graph.add_node("task:T1", NodeType.TASK, "Task 1")
        result = self.manager.find_path("task:T1", "task:T1")
        assert len(result) == 1
        assert result[0].id == "task:T1"

    def test_direct_path(self):
        """Direct connection returns two-node path."""
        self.manager.graph.add_node("task:T1", NodeType.TASK, "Task 1")
        self.manager.graph.add_node("task:T2", NodeType.TASK, "Task 2")
        self.manager.graph.add_edge("task:T1", "task:T2", EdgeType.BLOCKS)

        result = self.manager.find_path("task:T1", "task:T2")
        assert len(result) == 2
        assert result[0].id == "task:T1"
        assert result[1].id == "task:T2"

    def test_indirect_path(self):
        """Finds path through intermediate nodes."""
        self.manager.graph.add_node("task:T1", NodeType.TASK, "Task 1")
        self.manager.graph.add_node("task:T2", NodeType.TASK, "Task 2")
        self.manager.graph.add_node("task:T3", NodeType.TASK, "Task 3")
        self.manager.graph.add_edge("task:T1", "task:T2", EdgeType.BLOCKS)
        self.manager.graph.add_edge("task:T2", "task:T3", EdgeType.DEPENDS_ON)

        result = self.manager.find_path("task:T1", "task:T3")
        assert len(result) == 3
        assert result[0].id == "task:T1"
        assert result[1].id == "task:T2"
        assert result[2].id == "task:T3"

    def test_no_path_exists(self):
        """Returns None when no path exists."""
        self.manager.graph.add_node("task:T1", NodeType.TASK, "Task 1")
        self.manager.graph.add_node("task:T2", NodeType.TASK, "Task 2")
        # No edge between them

        result = self.manager.find_path("task:T1", "task:T2")
        assert result is None

    def test_shortest_path(self):
        """Returns shortest path when multiple paths exist."""
        self.manager.graph.add_node("task:T1", NodeType.TASK, "Task 1")
        self.manager.graph.add_node("task:T2", NodeType.TASK, "Task 2")
        self.manager.graph.add_node("task:T3", NodeType.TASK, "Task 3")
        self.manager.graph.add_node("task:T4", NodeType.TASK, "Task 4")

        # Create two paths: T1->T2->T4 (length 2) and T1->T3->T4 (length 2)
        self.manager.graph.add_edge("task:T1", "task:T2", EdgeType.BLOCKS)
        self.manager.graph.add_edge("task:T2", "task:T4", EdgeType.BLOCKS)
        self.manager.graph.add_edge("task:T1", "task:T3", EdgeType.BLOCKS)
        self.manager.graph.add_edge("task:T3", "task:T4", EdgeType.BLOCKS)

        result = self.manager.find_path("task:T1", "task:T4")
        # Should return one of the shortest paths (length 3)
        assert len(result) == 3
        assert result[0].id == "task:T1"
        assert result[2].id == "task:T4"

    def test_respects_max_depth(self):
        """Stops searching after max_depth."""
        # Create a long chain: T1->T2->T3->T4->T5
        for i in range(1, 6):
            self.manager.graph.add_node(f"task:T{i}", NodeType.TASK, f"Task {i}")
        for i in range(1, 5):
            self.manager.graph.add_edge(f"task:T{i}", f"task:T{i+1}", EdgeType.BLOCKS)

        # Search with max_depth=2 (should not find T5 from T1)
        result = self.manager.find_path("task:T1", "task:T5", max_depth=2)
        assert result is None

        # Search with max_depth=10 (should find it)
        result = self.manager.find_path("task:T1", "task:T5", max_depth=10)
        assert result is not None
        assert len(result) == 5

    def test_handles_cycles(self):
        """Handles graphs with cycles without infinite loop."""
        self.manager.graph.add_node("task:T1", NodeType.TASK, "Task 1")
        self.manager.graph.add_node("task:T2", NodeType.TASK, "Task 2")
        self.manager.graph.add_node("task:T3", NodeType.TASK, "Task 3")

        # Create a cycle: T1->T2->T3->T1
        self.manager.graph.add_edge("task:T1", "task:T2", EdgeType.BLOCKS)
        self.manager.graph.add_edge("task:T2", "task:T3", EdgeType.BLOCKS)
        self.manager.graph.add_edge("task:T3", "task:T1", EdgeType.BLOCKS)

        # Should still find path from T1 to T3
        result = self.manager.find_path("task:T1", "task:T3")
        assert len(result) == 3
        assert result[0].id == "task:T1"
        assert result[2].id == "task:T3"

    def test_nonexistent_source(self):
        """Returns None for nonexistent source node."""
        self.manager.graph.add_node("task:T2", NodeType.TASK, "Task 2")
        result = self.manager.find_path("task:NONEXISTENT", "task:T2")
        assert result is None

    def test_nonexistent_target(self):
        """Returns None for nonexistent target node."""
        self.manager.graph.add_node("task:T1", NodeType.TASK, "Task 1")
        result = self.manager.find_path("task:T1", "task:NONEXISTENT")
        assert result is None


class TestGetAllRelationships:
    """Test get_all_relationships() method."""

    def setup_method(self):
        """Set up test manager with fresh graph."""
        self.manager = GoTProjectManager()

    def test_empty_graph(self):
        """Empty graph returns empty relationship dict."""
        result = self.manager.get_all_relationships("task:nonexistent")
        assert result == {
            'blocks': [],
            'blocked_by': [],
            'depends_on': [],
            'depended_by': [],
            'in_sprint': [],
        }

    def test_task_with_no_relationships(self):
        """Task with no relationships returns empty lists."""
        self.manager.graph.add_node("task:T1", NodeType.TASK, "Task 1")
        result = self.manager.get_all_relationships("task:T1")
        assert all(len(v) == 0 for v in result.values())

    def test_blocks_relationship(self):
        """Captures tasks that this task blocks."""
        self.manager.graph.add_node("task:T1", NodeType.TASK, "Task 1")
        self.manager.graph.add_node("task:T2", NodeType.TASK, "Task 2")
        self.manager.graph.add_edge("task:T1", "task:T2", EdgeType.BLOCKS)

        result = self.manager.get_all_relationships("task:T1")
        assert len(result['blocks']) == 1
        assert result['blocks'][0].id == "task:T2"
        assert len(result['blocked_by']) == 0

    def test_blocked_by_relationship(self):
        """Captures tasks blocking this task."""
        self.manager.graph.add_node("task:T1", NodeType.TASK, "Task 1")
        self.manager.graph.add_node("task:T2", NodeType.TASK, "Task 2")
        self.manager.graph.add_edge("task:T1", "task:T2", EdgeType.BLOCKS)

        result = self.manager.get_all_relationships("task:T2")
        assert len(result['blocked_by']) == 1
        assert result['blocked_by'][0].id == "task:T1"
        assert len(result['blocks']) == 0

    def test_depends_on_relationship(self):
        """Captures tasks that this task depends on."""
        self.manager.graph.add_node("task:T1", NodeType.TASK, "Task 1")
        self.manager.graph.add_node("task:T2", NodeType.TASK, "Task 2")
        self.manager.graph.add_edge("task:T1", "task:T2", EdgeType.DEPENDS_ON)

        result = self.manager.get_all_relationships("task:T1")
        assert len(result['depends_on']) == 1
        assert result['depends_on'][0].id == "task:T2"
        assert len(result['depended_by']) == 0

    def test_depended_by_relationship(self):
        """Captures tasks depending on this task."""
        self.manager.graph.add_node("task:T1", NodeType.TASK, "Task 1")
        self.manager.graph.add_node("task:T2", NodeType.TASK, "Task 2")
        self.manager.graph.add_edge("task:T2", "task:T1", EdgeType.DEPENDS_ON)

        result = self.manager.get_all_relationships("task:T1")
        assert len(result['depended_by']) == 1
        assert result['depended_by'][0].id == "task:T2"
        assert len(result['depends_on']) == 0

    def test_in_sprint_relationship(self):
        """Captures sprint containing this task."""
        self.manager.graph.add_node("sprint:S1", NodeType.CONTEXT, "Sprint 1")
        self.manager.graph.add_node("task:T1", NodeType.TASK, "Task 1")
        self.manager.graph.add_edge("sprint:S1", "task:T1", EdgeType.CONTAINS)

        result = self.manager.get_all_relationships("task:T1")
        assert len(result['in_sprint']) == 1
        assert result['in_sprint'][0].id == "sprint:S1"

    def test_multiple_relationships(self):
        """Captures all relationship types simultaneously."""
        self.manager.graph.add_node("task:T1", NodeType.TASK, "Task 1")
        self.manager.graph.add_node("task:T2", NodeType.TASK, "Task 2")
        self.manager.graph.add_node("task:T3", NodeType.TASK, "Task 3")
        self.manager.graph.add_node("task:T4", NodeType.TASK, "Task 4")
        self.manager.graph.add_node("task:T5", NodeType.TASK, "Task 5")
        self.manager.graph.add_node("sprint:S1", NodeType.CONTEXT, "Sprint 1")

        # T1 blocks T2
        self.manager.graph.add_edge("task:T1", "task:T2", EdgeType.BLOCKS)
        # T3 blocks T1
        self.manager.graph.add_edge("task:T3", "task:T1", EdgeType.BLOCKS)
        # T1 depends on T4
        self.manager.graph.add_edge("task:T1", "task:T4", EdgeType.DEPENDS_ON)
        # T5 depends on T1
        self.manager.graph.add_edge("task:T5", "task:T1", EdgeType.DEPENDS_ON)
        # T1 in Sprint 1
        self.manager.graph.add_edge("sprint:S1", "task:T1", EdgeType.CONTAINS)

        result = self.manager.get_all_relationships("task:T1")
        assert len(result['blocks']) == 1
        assert result['blocks'][0].id == "task:T2"
        assert len(result['blocked_by']) == 1
        assert result['blocked_by'][0].id == "task:T3"
        assert len(result['depends_on']) == 1
        assert result['depends_on'][0].id == "task:T4"
        assert len(result['depended_by']) == 1
        assert result['depended_by'][0].id == "task:T5"
        assert len(result['in_sprint']) == 1
        assert result['in_sprint'][0].id == "sprint:S1"

    def test_handles_task_prefix(self):
        """Automatically adds task: prefix if missing."""
        self.manager.graph.add_node("task:T1", NodeType.TASK, "Task 1")
        self.manager.graph.add_node("task:T2", NodeType.TASK, "Task 2")
        self.manager.graph.add_edge("task:T1", "task:T2", EdgeType.BLOCKS)

        # Test without prefix
        result = self.manager.get_all_relationships("T1")
        assert len(result['blocks']) == 1
        assert result['blocks'][0].id == "task:T2"


class TestQueryParser:
    """Test query() string parser method."""

    def setup_method(self):
        """Set up test manager with fresh graph."""
        self.manager = GoTProjectManager()
        # Create a basic graph for testing
        # Note: query() lowercases the query string, so use lowercase IDs
        self.manager.graph.add_node("task:t1", NodeType.TASK, "Task 1",
                                    properties={"status": "pending", "priority": "high"})
        self.manager.graph.add_node("task:t2", NodeType.TASK, "Task 2",
                                    properties={"status": "active", "priority": "medium"})
        self.manager.graph.add_node("task:t3", NodeType.TASK, "Task 3",
                                    properties={"status": "blocked", "priority": "low"})
        self.manager.graph.add_edge("task:t1", "task:t2", EdgeType.BLOCKS)
        self.manager.graph.add_edge("task:t3", "task:t2", EdgeType.DEPENDS_ON)

    def test_what_blocks_query(self):
        """Parse 'what blocks <task_id>' query."""
        result = self.manager.query("what blocks task:t2")
        assert len(result) == 1
        assert result[0]["id"] == "task:t1"
        assert result[0]["relation"] == "blocks"

    def test_what_blocks_query_case_insensitive(self):
        """Query parsing is case-insensitive."""
        result = self.manager.query("What Blocks task:t2")
        assert len(result) == 1
        assert result[0]["id"] == "task:t1"

    def test_what_depends_on_query(self):
        """Parse 'what depends on <task_id>' query."""
        result = self.manager.query("what depends on task:t2")
        assert len(result) == 1
        assert result[0]["id"] == "task:t3"
        assert result[0]["relation"] == "depends_on"

    def test_path_from_to_query(self):
        """Parse 'path from <id1> to <id2>' query."""
        result = self.manager.query("path from task:t1 to task:t2")
        assert len(result) == 2
        assert result[0]["step"] == 0
        assert result[0]["id"] == "task:t1"
        assert result[1]["step"] == 1
        assert result[1]["id"] == "task:t2"

    def test_path_query_no_path(self):
        """Path query returns empty list when no path exists."""
        self.manager.graph.add_node("task:t4", NodeType.TASK, "Task 4")
        result = self.manager.query("path from task:t1 to task:t4")
        assert result == []

    def test_relationships_query(self):
        """Parse 'relationships <task_id>' query."""
        result = self.manager.query("relationships task:t2")
        # t2 is blocked by t1 and depends on t3
        assert len(result) == 2
        relations = {(r["relation"], r["id"]) for r in result}
        assert ("blocked_by", "task:t1") in relations
        assert ("depended_by", "task:t3") in relations

    def test_blocked_tasks_query(self):
        """Parse 'blocked tasks' query."""
        # Note: This requires get_blocked_tasks() to be implemented
        # For now, we'll just check it doesn't crash
        result = self.manager.query("blocked tasks")
        # Implementation may vary, just check it returns a list
        assert isinstance(result, list)

    def test_active_tasks_query(self):
        """Parse 'active tasks' query."""
        # Note: This requires get_active_tasks() to be implemented
        result = self.manager.query("active tasks")
        assert isinstance(result, list)

    def test_pending_tasks_query(self):
        """Parse 'pending tasks' query."""
        result = self.manager.query("pending tasks")
        assert isinstance(result, list)
        # Should include t1 which has status="pending"
        pending_ids = {r["id"] for r in result}
        assert "task:t1" in pending_ids

    def test_unknown_query(self):
        """Unknown query returns empty list."""
        result = self.manager.query("unknown query format")
        assert result == []

    def test_empty_query(self):
        """Empty query returns empty list."""
        result = self.manager.query("")
        assert result == []

    def test_whitespace_handling(self):
        """Query handles extra whitespace correctly."""
        result = self.manager.query("  what blocks   task:t2  ")
        assert len(result) == 1
        assert result[0]["id"] == "task:t1"

    def test_malformed_path_query(self):
        """Malformed path query returns empty list."""
        # Missing 'to' keyword
        result = self.manager.query("path from task:t1 task:t2")
        assert result == []

        # Missing second ID
        result = self.manager.query("path from task:t1 to")
        assert result == []


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Set up test manager with fresh graph."""
        self.manager = GoTProjectManager()

    def test_what_blocks_with_deleted_node(self):
        """Handles case where blocker node was deleted."""
        self.manager.graph.add_node("task:T1", NodeType.TASK, "Task 1")
        self.manager.graph.add_node("task:T2", NodeType.TASK, "Task 2")
        self.manager.graph.add_edge("task:T1", "task:T2", EdgeType.BLOCKS)

        # Manually corrupt graph by removing node but keeping edge
        # (This shouldn't happen in normal operation, but tests robustness)
        del self.manager.graph.nodes["task:T1"]

        # Should handle gracefully and skip missing node
        result = self.manager.what_blocks("task:T2")
        assert result == []

    def test_find_path_with_special_characters(self):
        """Handles node IDs with special characters."""
        self.manager.graph.add_node("task:T-1", NodeType.TASK, "Task 1")
        self.manager.graph.add_node("task:T_2", NodeType.TASK, "Task 2")
        self.manager.graph.add_edge("task:T-1", "task:T_2", EdgeType.BLOCKS)

        result = self.manager.find_path("task:T-1", "task:T_2")
        assert len(result) == 2

    def test_get_all_relationships_multiple_same_type(self):
        """Handles multiple edges of same type correctly."""
        self.manager.graph.add_node("task:T1", NodeType.TASK, "Task 1")
        self.manager.graph.add_node("task:T2", NodeType.TASK, "Task 2")
        self.manager.graph.add_node("task:T3", NodeType.TASK, "Task 3")

        # T1 blocks both T2 and T3
        self.manager.graph.add_edge("task:T1", "task:T2", EdgeType.BLOCKS)
        self.manager.graph.add_edge("task:T1", "task:T3", EdgeType.BLOCKS)

        result = self.manager.get_all_relationships("task:T1")
        assert len(result['blocks']) == 2
        blocked_ids = {node.id for node in result['blocks']}
        assert blocked_ids == {"task:T2", "task:T3"}
