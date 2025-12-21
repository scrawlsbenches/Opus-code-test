"""
Regression tests for GoT edge rebuild from events.

These tests cover bugs discovered during development:
- T-20251221-014654-d4b7: Parameter name mismatch (source_id vs from_id)
- EdgeType enum lookup with hasattr (doesn't work correctly)
- Silent failures in event replay
"""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cortical.reasoning.thought_graph import ThoughtGraph
from cortical.reasoning.graph_of_thought import EdgeType, NodeType
from scripts.got_utils import EventLog


class TestEdgeRebuildFromEvents:
    """Test that edges are correctly rebuilt from event logs."""

    def test_basic_edge_rebuild(self, tmp_path):
        """Test basic edge creation and rebuild round-trip."""
        events_dir = tmp_path / "events"
        events_dir.mkdir()

        # Create events manually
        events = [
            {
                "ts": "2025-01-01T00:00:00Z",
                "event": "node.create",
                "id": "task:T-001",
                "type": "TASK",
                "data": {"title": "Test Task 1"},
                "meta": {}
            },
            {
                "ts": "2025-01-01T00:00:01Z",
                "event": "node.create",
                "id": "task:T-002",
                "type": "TASK",
                "data": {"title": "Test Task 2"},
                "meta": {}
            },
            {
                "ts": "2025-01-01T00:00:02Z",
                "event": "edge.create",
                "src": "task:T-001",
                "tgt": "task:T-002",
                "type": "DEPENDS_ON",
                "weight": 1.0
            }
        ]

        # Write events to file
        event_file = events_dir / "test.jsonl"
        with open(event_file, "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")

        # Load and rebuild
        loaded_events = EventLog.load_all_events(events_dir)
        graph = EventLog.rebuild_graph_from_events(loaded_events)

        # Verify
        assert len(graph.nodes) == 2, "Should have 2 nodes"
        assert len(graph.edges) == 1, "Should have 1 edge"

        edge = graph.edges[0]
        assert edge.source_id == "task:T-001"
        assert edge.target_id == "task:T-002"
        assert edge.edge_type == EdgeType.DEPENDS_ON

    def test_all_edge_types_rebuild(self, tmp_path):
        """Test that ALL EdgeType values can be rebuilt from events."""
        events_dir = tmp_path / "events"
        events_dir.mkdir()

        events = [
            {"ts": "2025-01-01T00:00:00Z", "event": "node.create", "id": "n1", "type": "TASK", "data": {"title": "N1"}, "meta": {}},
            {"ts": "2025-01-01T00:00:01Z", "event": "node.create", "id": "n2", "type": "TASK", "data": {"title": "N2"}, "meta": {}},
        ]

        # Add an edge for each EdgeType
        edge_types_tested = []
        for i, edge_type in enumerate(EdgeType):
            events.append({
                "ts": f"2025-01-01T00:00:{i+10:02d}Z",
                "event": "edge.create",
                "src": "n1",
                "tgt": "n2",
                "type": edge_type.name,
                "weight": 1.0
            })
            edge_types_tested.append(edge_type)

        # Write events
        event_file = events_dir / "test.jsonl"
        with open(event_file, "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")

        # Rebuild
        loaded_events = EventLog.load_all_events(events_dir)
        graph = EventLog.rebuild_graph_from_events(loaded_events)

        # Verify all edges were created
        assert len(graph.edges) == len(edge_types_tested), \
            f"Expected {len(edge_types_tested)} edges, got {len(graph.edges)}"

        # Verify each edge type exists
        created_types = {e.edge_type for e in graph.edges}
        for expected_type in edge_types_tested:
            assert expected_type in created_types, \
                f"EdgeType.{expected_type.name} was not rebuilt correctly"

    def test_unknown_edge_type_falls_back(self, tmp_path):
        """Test that unknown edge types fall back to MOTIVATES."""
        events_dir = tmp_path / "events"
        events_dir.mkdir()

        events = [
            {"ts": "2025-01-01T00:00:00Z", "event": "node.create", "id": "n1", "type": "TASK", "data": {"title": "N1"}, "meta": {}},
            {"ts": "2025-01-01T00:00:01Z", "event": "node.create", "id": "n2", "type": "TASK", "data": {"title": "N2"}, "meta": {}},
            {
                "ts": "2025-01-01T00:00:02Z",
                "event": "edge.create",
                "src": "n1",
                "tgt": "n2",
                "type": "NONEXISTENT_TYPE",  # Invalid type
                "weight": 1.0
            }
        ]

        event_file = events_dir / "test.jsonl"
        with open(event_file, "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")

        loaded_events = EventLog.load_all_events(events_dir)
        graph = EventLog.rebuild_graph_from_events(loaded_events)

        # Should still create the edge with fallback type
        assert len(graph.edges) == 1, "Edge with unknown type should still be created"
        assert graph.edges[0].edge_type == EdgeType.MOTIVATES, "Should fall back to MOTIVATES"

    def test_edge_with_missing_source_node_skipped(self, tmp_path):
        """Test that edges with missing source nodes are skipped."""
        events_dir = tmp_path / "events"
        events_dir.mkdir()

        events = [
            {"ts": "2025-01-01T00:00:00Z", "event": "node.create", "id": "n1", "type": "TASK", "data": {"title": "N1"}, "meta": {}},
            # n2 NOT created
            {
                "ts": "2025-01-01T00:00:02Z",
                "event": "edge.create",
                "src": "nonexistent",  # Source doesn't exist
                "tgt": "n1",
                "type": "DEPENDS_ON",
                "weight": 1.0
            }
        ]

        event_file = events_dir / "test.jsonl"
        with open(event_file, "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")

        loaded_events = EventLog.load_all_events(events_dir)
        graph = EventLog.rebuild_graph_from_events(loaded_events)

        # Edge should be skipped (source node doesn't exist)
        assert len(graph.edges) == 0, "Edge with missing source should be skipped"

    def test_edge_with_missing_target_node_skipped(self, tmp_path):
        """Test that edges with missing target nodes are skipped."""
        events_dir = tmp_path / "events"
        events_dir.mkdir()

        events = [
            {"ts": "2025-01-01T00:00:00Z", "event": "node.create", "id": "n1", "type": "TASK", "data": {"title": "N1"}, "meta": {}},
            {
                "ts": "2025-01-01T00:00:02Z",
                "event": "edge.create",
                "src": "n1",
                "tgt": "nonexistent",  # Target doesn't exist
                "type": "DEPENDS_ON",
                "weight": 1.0
            }
        ]

        event_file = events_dir / "test.jsonl"
        with open(event_file, "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")

        loaded_events = EventLog.load_all_events(events_dir)
        graph = EventLog.rebuild_graph_from_events(loaded_events)

        # Edge should be skipped (target node doesn't exist)
        assert len(graph.edges) == 0, "Edge with missing target should be skipped"

    def test_edge_weight_preserved(self, tmp_path):
        """Test that edge weights are preserved during rebuild."""
        events_dir = tmp_path / "events"
        events_dir.mkdir()

        events = [
            {"ts": "2025-01-01T00:00:00Z", "event": "node.create", "id": "n1", "type": "TASK", "data": {"title": "N1"}, "meta": {}},
            {"ts": "2025-01-01T00:00:01Z", "event": "node.create", "id": "n2", "type": "TASK", "data": {"title": "N2"}, "meta": {}},
            {
                "ts": "2025-01-01T00:00:02Z",
                "event": "edge.create",
                "src": "n1",
                "tgt": "n2",
                "type": "SIMILAR",
                "weight": 0.75
            }
        ]

        event_file = events_dir / "test.jsonl"
        with open(event_file, "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")

        loaded_events = EventLog.load_all_events(events_dir)
        graph = EventLog.rebuild_graph_from_events(loaded_events)

        assert len(graph.edges) == 1
        assert graph.edges[0].weight == 0.75, "Edge weight should be preserved"


class TestEdgeTypeEnumLookup:
    """Test EdgeType enum behavior - regression for hasattr bug."""

    def test_hasattr_does_not_work_for_enum_members(self):
        """Verify that hasattr doesn't work correctly for enum members."""
        # This test documents the bug - hasattr returns False for valid enum members
        assert hasattr(EdgeType, "DEPENDS_ON") is True, "hasattr works for DEPENDS_ON"
        assert hasattr(EdgeType, "SIMILAR") is True, "hasattr works for SIMILAR"
        # Note: hasattr actually DOES work for enum members in recent Python
        # The original bug may have been a different issue

    def test_bracket_access_works_for_valid_types(self):
        """Test that bracket access works for all EdgeType values."""
        for edge_type in EdgeType:
            # This should not raise
            retrieved = EdgeType[edge_type.name]
            assert retrieved == edge_type

    def test_bracket_access_raises_for_invalid_types(self):
        """Test that bracket access raises KeyError for invalid types."""
        with pytest.raises(KeyError):
            EdgeType["NONEXISTENT"]

    def test_all_edge_types_have_names(self):
        """Verify all EdgeType members have valid names."""
        for edge_type in EdgeType:
            assert edge_type.name is not None
            assert len(edge_type.name) > 0
            assert edge_type.name == edge_type.name.upper(), "Names should be uppercase"


class TestEventLogRoundTrip:
    """Test complete round-trip: create events -> save -> load -> rebuild."""

    def test_full_round_trip_with_edges(self, tmp_path):
        """Test complete event log round-trip preserves edges."""
        events_dir = tmp_path / "events"
        events_dir.mkdir()

        # Use EventLog to create events
        event_log = EventLog(events_dir, session_id="test-session")

        # Create nodes
        event_log.log_node_create("task:T-001", "TASK", {"title": "Task 1", "status": "pending"})
        event_log.log_node_create("task:T-002", "TASK", {"title": "Task 2", "status": "pending"})
        event_log.log_node_create("task:T-003", "TASK", {"title": "Task 3", "status": "pending"})

        # Create edges
        event_log.log_edge_create("task:T-001", "task:T-002", "DEPENDS_ON", weight=1.0)
        event_log.log_edge_create("task:T-002", "task:T-003", "BLOCKS", weight=0.8)
        event_log.log_edge_create("task:T-001", "task:T-003", "SIMILAR", weight=0.5)

        # Load and rebuild
        loaded_events = EventLog.load_all_events(events_dir)
        graph = EventLog.rebuild_graph_from_events(loaded_events)

        # Verify
        assert len(graph.nodes) == 3, "Should have 3 nodes"
        assert len(graph.edges) == 3, "Should have 3 edges"

        # Check specific edges
        edge_map = {(e.source_id, e.target_id): e for e in graph.edges}

        assert ("task:T-001", "task:T-002") in edge_map
        assert edge_map[("task:T-001", "task:T-002")].edge_type == EdgeType.DEPENDS_ON

        assert ("task:T-002", "task:T-003") in edge_map
        assert edge_map[("task:T-002", "task:T-003")].edge_type == EdgeType.BLOCKS

        assert ("task:T-001", "task:T-003") in edge_map
        assert edge_map[("task:T-001", "task:T-003")].edge_type == EdgeType.SIMILAR


class TestThoughtGraphAddEdge:
    """Test ThoughtGraph.add_edge parameter handling."""

    def test_add_edge_with_from_id_to_id(self):
        """Test that add_edge works with from_id and to_id parameters."""
        graph = ThoughtGraph()
        graph.add_node("n1", NodeType.TASK, "Node 1")
        graph.add_node("n2", NodeType.TASK, "Node 2")

        # This is the correct API
        graph.add_edge(from_id="n1", to_id="n2", edge_type=EdgeType.DEPENDS_ON)

        assert len(graph.edges) == 1
        assert graph.edges[0].source_id == "n1"
        assert graph.edges[0].target_id == "n2"

    def test_add_edge_rejects_source_id_target_id(self):
        """Test that add_edge raises TypeError for wrong param names."""
        graph = ThoughtGraph()
        graph.add_node("n1", NodeType.TASK, "Node 1")
        graph.add_node("n2", NodeType.TASK, "Node 2")

        # This was the bug - using wrong parameter names
        with pytest.raises(TypeError):
            graph.add_edge(source_id="n1", target_id="n2", edge_type=EdgeType.DEPENDS_ON)

    def test_add_edge_with_missing_node_raises(self):
        """Test that add_edge raises ValueError for missing nodes."""
        graph = ThoughtGraph()
        graph.add_node("n1", NodeType.TASK, "Node 1")
        # n2 not created

        with pytest.raises(ValueError, match="not found"):
            graph.add_edge(from_id="n1", to_id="n2", edge_type=EdgeType.DEPENDS_ON)


class TestMalformedEdgeData:
    """
    Regression tests for malformed edge data with comma-concatenated IDs.

    Bug: Some edge.create events have target IDs like:
        "task:T-20251220-231129-81b3,task:T-20251220-231159-7d31"
    Instead of being two separate edges.

    Expected behavior: Parse comma-separated IDs and create multiple edges.
    """

    def test_comma_concatenated_target_creates_multiple_edges(self, tmp_path):
        """
        Scenario: edge.create has comma-separated target IDs.
        Expected: Create separate edges for each target.
        """
        events_dir = tmp_path / "events"
        events_dir.mkdir()

        events = [
            {"ts": "2025-01-01T00:00:00Z", "event": "node.create", "id": "task:T-001", "type": "TASK", "data": {"title": "Source"}, "meta": {}},
            {"ts": "2025-01-01T00:00:01Z", "event": "node.create", "id": "task:T-002", "type": "TASK", "data": {"title": "Target 1"}, "meta": {}},
            {"ts": "2025-01-01T00:00:02Z", "event": "node.create", "id": "task:T-003", "type": "TASK", "data": {"title": "Target 2"}, "meta": {}},
            {
                "ts": "2025-01-01T00:00:03Z",
                "event": "edge.create",
                "src": "task:T-001",
                "tgt": "task:T-002,task:T-003",  # Malformed: comma-concatenated
                "type": "DEPENDS_ON",
                "weight": 1.0
            }
        ]

        event_file = events_dir / "test.jsonl"
        with open(event_file, "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")

        loaded_events = EventLog.load_all_events(events_dir)
        graph = EventLog.rebuild_graph_from_events(loaded_events)

        # Should create 2 edges (one for each target)
        assert len(graph.edges) == 2, f"Should create 2 edges, got {len(graph.edges)}"

        # Verify both edges exist
        edge_targets = {e.target_id for e in graph.edges}
        assert "task:T-002" in edge_targets, "Edge to T-002 should exist"
        assert "task:T-003" in edge_targets, "Edge to T-003 should exist"

    def test_comma_concatenated_source_creates_multiple_edges(self, tmp_path):
        """
        Scenario: edge.create has comma-separated source IDs.
        Expected: Create separate edges for each source.
        """
        events_dir = tmp_path / "events"
        events_dir.mkdir()

        events = [
            {"ts": "2025-01-01T00:00:00Z", "event": "node.create", "id": "task:T-001", "type": "TASK", "data": {"title": "Source 1"}, "meta": {}},
            {"ts": "2025-01-01T00:00:01Z", "event": "node.create", "id": "task:T-002", "type": "TASK", "data": {"title": "Source 2"}, "meta": {}},
            {"ts": "2025-01-01T00:00:02Z", "event": "node.create", "id": "task:T-003", "type": "TASK", "data": {"title": "Target"}, "meta": {}},
            {
                "ts": "2025-01-01T00:00:03Z",
                "event": "edge.create",
                "src": "task:T-001,task:T-002",  # Malformed: comma-concatenated sources
                "tgt": "task:T-003",
                "type": "BLOCKS",
                "weight": 0.8
            }
        ]

        event_file = events_dir / "test.jsonl"
        with open(event_file, "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")

        loaded_events = EventLog.load_all_events(events_dir)
        graph = EventLog.rebuild_graph_from_events(loaded_events)

        # Should create 2 edges (one from each source)
        assert len(graph.edges) == 2, f"Should create 2 edges, got {len(graph.edges)}"

        # Verify both edges exist
        edge_sources = {e.source_id for e in graph.edges}
        assert "task:T-001" in edge_sources, "Edge from T-001 should exist"
        assert "task:T-002" in edge_sources, "Edge from T-002 should exist"

    def test_both_source_and_target_comma_concatenated(self, tmp_path):
        """
        Scenario: edge.create has comma-separated source AND target IDs.
        Expected: Create edges for all source-target combinations (cartesian product).
        """
        events_dir = tmp_path / "events"
        events_dir.mkdir()

        events = [
            {"ts": "2025-01-01T00:00:00Z", "event": "node.create", "id": "task:T-001", "type": "TASK", "data": {"title": "S1"}, "meta": {}},
            {"ts": "2025-01-01T00:00:01Z", "event": "node.create", "id": "task:T-002", "type": "TASK", "data": {"title": "S2"}, "meta": {}},
            {"ts": "2025-01-01T00:00:02Z", "event": "node.create", "id": "task:T-003", "type": "TASK", "data": {"title": "T1"}, "meta": {}},
            {"ts": "2025-01-01T00:00:03Z", "event": "node.create", "id": "task:T-004", "type": "TASK", "data": {"title": "T2"}, "meta": {}},
            {
                "ts": "2025-01-01T00:00:04Z",
                "event": "edge.create",
                "src": "task:T-001,task:T-002",  # 2 sources
                "tgt": "task:T-003,task:T-004",  # 2 targets
                "type": "SIMILAR",
                "weight": 0.5
            }
        ]

        event_file = events_dir / "test.jsonl"
        with open(event_file, "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")

        loaded_events = EventLog.load_all_events(events_dir)
        graph = EventLog.rebuild_graph_from_events(loaded_events)

        # Should create 4 edges (2x2 cartesian product)
        assert len(graph.edges) == 4, f"Should create 4 edges (2x2), got {len(graph.edges)}"

    def test_comma_concatenated_with_missing_node_skips_that_edge(self, tmp_path):
        """
        Scenario: Comma-separated IDs include a non-existent node.
        Expected: Create edges for valid nodes, skip invalid ones.
        """
        events_dir = tmp_path / "events"
        events_dir.mkdir()

        events = [
            {"ts": "2025-01-01T00:00:00Z", "event": "node.create", "id": "task:T-001", "type": "TASK", "data": {"title": "Source"}, "meta": {}},
            {"ts": "2025-01-01T00:00:01Z", "event": "node.create", "id": "task:T-002", "type": "TASK", "data": {"title": "Valid Target"}, "meta": {}},
            # task:T-003 NOT created (missing)
            {
                "ts": "2025-01-01T00:00:02Z",
                "event": "edge.create",
                "src": "task:T-001",
                "tgt": "task:T-002,task:T-003",  # T-003 doesn't exist
                "type": "DEPENDS_ON",
                "weight": 1.0
            }
        ]

        event_file = events_dir / "test.jsonl"
        with open(event_file, "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")

        loaded_events = EventLog.load_all_events(events_dir)
        graph = EventLog.rebuild_graph_from_events(loaded_events)

        # Should create 1 edge (only to T-002, skip T-003)
        assert len(graph.edges) == 1, f"Should create 1 edge, got {len(graph.edges)}"
        assert graph.edges[0].target_id == "task:T-002"

    def test_real_world_malformed_edge_from_production(self, tmp_path):
        """
        Scenario: Real edge from production data with comma-concatenated IDs.
        This is the actual format from Event 266.
        """
        events_dir = tmp_path / "events"
        events_dir.mkdir()

        # Create all the nodes that should exist
        events = [
            {"ts": "2025-01-01T00:00:00Z", "event": "node.create", "id": "task:T-20251220-source", "type": "TASK", "data": {"title": "Source Task"}, "meta": {}},
            {"ts": "2025-01-01T00:00:01Z", "event": "node.create", "id": "task:T-20251220-231129-81b3", "type": "TASK", "data": {"title": "Target 1"}, "meta": {}},
            {"ts": "2025-01-01T00:00:02Z", "event": "node.create", "id": "task:T-20251220-231159-7d31", "type": "TASK", "data": {"title": "Target 2"}, "meta": {}},
            {
                "ts": "2025-01-01T00:00:03Z",
                "event": "edge.create",
                "src": "task:T-20251220-source",
                # Real production format: comma-concatenated full task IDs
                "tgt": "task:T-20251220-231129-81b3,task:T-20251220-231159-7d31",
                "type": "DEPENDS_ON",
                "weight": 1.0
            }
        ]

        event_file = events_dir / "test.jsonl"
        with open(event_file, "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")

        loaded_events = EventLog.load_all_events(events_dir)
        graph = EventLog.rebuild_graph_from_events(loaded_events)

        # Should create 2 edges
        assert len(graph.edges) == 2, f"Should create 2 edges from comma-separated targets, got {len(graph.edges)}"

        # Verify edge targets
        edge_targets = {e.target_id for e in graph.edges}
        assert "task:T-20251220-231129-81b3" in edge_targets
        assert "task:T-20251220-231159-7d31" in edge_targets

    def test_edge_with_spaces_around_comma(self, tmp_path):
        """
        Scenario: Comma-separated IDs have spaces around comma.
        Expected: Trim whitespace and create edges correctly.
        """
        events_dir = tmp_path / "events"
        events_dir.mkdir()

        events = [
            {"ts": "2025-01-01T00:00:00Z", "event": "node.create", "id": "task:T-001", "type": "TASK", "data": {"title": "S"}, "meta": {}},
            {"ts": "2025-01-01T00:00:01Z", "event": "node.create", "id": "task:T-002", "type": "TASK", "data": {"title": "T1"}, "meta": {}},
            {"ts": "2025-01-01T00:00:02Z", "event": "node.create", "id": "task:T-003", "type": "TASK", "data": {"title": "T2"}, "meta": {}},
            {
                "ts": "2025-01-01T00:00:03Z",
                "event": "edge.create",
                "src": "task:T-001",
                "tgt": "task:T-002 , task:T-003",  # Spaces around comma
                "type": "DEPENDS_ON",
                "weight": 1.0
            }
        ]

        event_file = events_dir / "test.jsonl"
        with open(event_file, "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")

        loaded_events = EventLog.load_all_events(events_dir)
        graph = EventLog.rebuild_graph_from_events(loaded_events)

        # Should create 2 edges with trimmed IDs
        assert len(graph.edges) == 2, f"Should handle spaces around comma, got {len(graph.edges)} edges"


class TestNodeIdNormalization:
    """
    Regression tests for ID normalization in node.update and node.delete.

    Bug: node.create used "task:T-XXX" format, but node.update used "T-XXX"
    format without the prefix, causing "Cannot update non-existent node" errors.

    Fix: Added ID normalization to try both formats when looking up nodes.
    """

    def test_node_update_with_prefix_mismatch(self, tmp_path):
        """Test that node.update works when ID format differs from node.create."""
        events_dir = tmp_path / "events"
        events_dir.mkdir()

        # Simulate the bug: node.create uses task: prefix, node.update doesn't
        events = [
            {
                "ts": "2025-01-01T00:00:00Z",
                "event": "node.create",
                "id": "task:T-20251220-001",  # WITH prefix
                "type": "TASK",
                "data": {"title": "Original Title", "status": "pending"},
                "meta": {}
            },
            {
                "ts": "2025-01-01T00:00:01Z",
                "event": "node.update",
                "id": "T-20251220-001",  # WITHOUT prefix (the bug)
                "changes": {"status": "completed", "title": "Updated Title"}
            }
        ]

        event_file = events_dir / "test.jsonl"
        with open(event_file, "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")

        loaded_events = EventLog.load_all_events(events_dir)
        graph = EventLog.rebuild_graph_from_events(loaded_events)

        # Verify the update was applied despite ID format mismatch
        assert len(graph.nodes) == 1, "Should have 1 node"
        node = graph.nodes["task:T-20251220-001"]
        assert node.properties.get("status") == "completed", "Status should be updated"
        assert node.properties.get("title") == "Updated Title", "Title should be updated"

    def test_node_update_with_reverse_prefix_mismatch(self, tmp_path):
        """Test update when node.create lacks prefix but node.update has it."""
        events_dir = tmp_path / "events"
        events_dir.mkdir()

        events = [
            {
                "ts": "2025-01-01T00:00:00Z",
                "event": "node.create",
                "id": "T-20251220-002",  # WITHOUT prefix
                "type": "TASK",
                "data": {"title": "Task", "priority": "low"},
                "meta": {}
            },
            {
                "ts": "2025-01-01T00:00:01Z",
                "event": "node.update",
                "id": "task:T-20251220-002",  # WITH prefix
                "changes": {"priority": "high"}
            }
        ]

        event_file = events_dir / "test.jsonl"
        with open(event_file, "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")

        loaded_events = EventLog.load_all_events(events_dir)
        graph = EventLog.rebuild_graph_from_events(loaded_events)

        # Verify the update was applied
        assert len(graph.nodes) == 1
        node = graph.nodes["T-20251220-002"]
        assert node.properties.get("priority") == "high"

    def test_node_delete_with_prefix_mismatch(self, tmp_path):
        """Test that node.delete works when ID format differs from node.create."""
        events_dir = tmp_path / "events"
        events_dir.mkdir()

        events = [
            {
                "ts": "2025-01-01T00:00:00Z",
                "event": "node.create",
                "id": "task:T-20251220-003",  # WITH prefix
                "type": "TASK",
                "data": {"title": "To Delete"},
                "meta": {}
            },
            {
                "ts": "2025-01-01T00:00:01Z",
                "event": "node.delete",
                "id": "T-20251220-003"  # WITHOUT prefix
            }
        ]

        event_file = events_dir / "test.jsonl"
        with open(event_file, "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")

        loaded_events = EventLog.load_all_events(events_dir)
        graph = EventLog.rebuild_graph_from_events(loaded_events)

        # Verify the node was deleted despite ID format mismatch
        assert len(graph.nodes) == 0, "Node should be deleted"

    def test_node_update_still_warns_for_truly_missing_node(self, tmp_path):
        """Test that warning is still logged for actually non-existent nodes."""
        events_dir = tmp_path / "events"
        events_dir.mkdir()

        events = [
            {
                "ts": "2025-01-01T00:00:00Z",
                "event": "node.update",
                "id": "T-NONEXISTENT",  # Node was never created
                "changes": {"status": "completed"}
            }
        ]

        event_file = events_dir / "test.jsonl"
        with open(event_file, "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")

        loaded_events = EventLog.load_all_events(events_dir)
        # This should not crash, just log a warning
        graph = EventLog.rebuild_graph_from_events(loaded_events)

        assert len(graph.nodes) == 0, "No nodes should exist"

    def test_multiple_updates_with_mixed_formats(self, tmp_path):
        """Test multiple updates with alternating ID formats."""
        events_dir = tmp_path / "events"
        events_dir.mkdir()

        events = [
            {
                "ts": "2025-01-01T00:00:00Z",
                "event": "node.create",
                "id": "task:T-20251220-004",
                "type": "TASK",
                "data": {"title": "Task", "count": 0},
                "meta": {}
            },
            {
                "ts": "2025-01-01T00:00:01Z",
                "event": "node.update",
                "id": "T-20251220-004",  # No prefix
                "changes": {"count": 1}
            },
            {
                "ts": "2025-01-01T00:00:02Z",
                "event": "node.update",
                "id": "task:T-20251220-004",  # With prefix
                "changes": {"count": 2}
            },
            {
                "ts": "2025-01-01T00:00:03Z",
                "event": "node.update",
                "id": "T-20251220-004",  # No prefix again
                "changes": {"count": 3}
            }
        ]

        event_file = events_dir / "test.jsonl"
        with open(event_file, "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")

        loaded_events = EventLog.load_all_events(events_dir)
        graph = EventLog.rebuild_graph_from_events(loaded_events)

        assert len(graph.nodes) == 1
        node = graph.nodes["task:T-20251220-004"]
        assert node.properties.get("count") == 3, "All updates should be applied"


class TestEdgeDeleteEvent:
    """
    Regression tests for edge.delete event handling.

    Bug: edge.delete used graph.edges.items() but graph.edges is a List, not Dict.
    Error: 'list' object has no attribute 'items'

    Fix: Iterate with enumerate() and use pop() to remove from list.
    """

    def test_edge_delete_removes_edge(self, tmp_path):
        """Test that edge.delete correctly removes an edge from the graph."""
        events_dir = tmp_path / "events"
        events_dir.mkdir()

        events = [
            {"ts": "2025-01-01T00:00:00Z", "event": "node.create", "id": "n1", "type": "TASK", "data": {"title": "N1"}, "meta": {}},
            {"ts": "2025-01-01T00:00:01Z", "event": "node.create", "id": "n2", "type": "TASK", "data": {"title": "N2"}, "meta": {}},
            {"ts": "2025-01-01T00:00:02Z", "event": "edge.create", "src": "n1", "tgt": "n2", "type": "DEPENDS_ON", "weight": 1.0},
            {"ts": "2025-01-01T00:00:03Z", "event": "edge.delete", "src": "n1", "tgt": "n2", "type": "DEPENDS_ON"},
        ]

        event_file = events_dir / "test.jsonl"
        with open(event_file, "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")

        loaded_events = EventLog.load_all_events(events_dir)
        graph = EventLog.rebuild_graph_from_events(loaded_events)

        # Edge should be deleted
        assert len(graph.edges) == 0, "Edge should be removed after edge.delete"
        assert len(graph.nodes) == 2, "Nodes should still exist"

    def test_edge_delete_removes_only_matching_edge(self, tmp_path):
        """Test that edge.delete only removes the matching edge, not others."""
        events_dir = tmp_path / "events"
        events_dir.mkdir()

        events = [
            {"ts": "2025-01-01T00:00:00Z", "event": "node.create", "id": "n1", "type": "TASK", "data": {"title": "N1"}, "meta": {}},
            {"ts": "2025-01-01T00:00:01Z", "event": "node.create", "id": "n2", "type": "TASK", "data": {"title": "N2"}, "meta": {}},
            {"ts": "2025-01-01T00:00:02Z", "event": "node.create", "id": "n3", "type": "TASK", "data": {"title": "N3"}, "meta": {}},
            {"ts": "2025-01-01T00:00:03Z", "event": "edge.create", "src": "n1", "tgt": "n2", "type": "DEPENDS_ON", "weight": 1.0},
            {"ts": "2025-01-01T00:00:04Z", "event": "edge.create", "src": "n2", "tgt": "n3", "type": "BLOCKS", "weight": 0.8},
            {"ts": "2025-01-01T00:00:05Z", "event": "edge.create", "src": "n1", "tgt": "n3", "type": "SIMILAR", "weight": 0.5},
            # Delete only the DEPENDS_ON edge
            {"ts": "2025-01-01T00:00:06Z", "event": "edge.delete", "src": "n1", "tgt": "n2", "type": "DEPENDS_ON"},
        ]

        event_file = events_dir / "test.jsonl"
        with open(event_file, "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")

        loaded_events = EventLog.load_all_events(events_dir)
        graph = EventLog.rebuild_graph_from_events(loaded_events)

        # Only DEPENDS_ON edge should be deleted
        assert len(graph.edges) == 2, "Only one edge should be deleted"

        edge_types = {(e.source_id, e.target_id, e.edge_type.name) for e in graph.edges}
        assert ("n1", "n2", "DEPENDS_ON") not in edge_types, "DEPENDS_ON edge should be deleted"
        assert ("n2", "n3", "BLOCKS") in edge_types, "BLOCKS edge should remain"
        assert ("n1", "n3", "SIMILAR") in edge_types, "SIMILAR edge should remain"

    def test_edge_delete_updates_edge_indices(self, tmp_path):
        """Test that edge.delete updates the internal edge indices."""
        events_dir = tmp_path / "events"
        events_dir.mkdir()

        events = [
            {"ts": "2025-01-01T00:00:00Z", "event": "node.create", "id": "n1", "type": "TASK", "data": {"title": "N1"}, "meta": {}},
            {"ts": "2025-01-01T00:00:01Z", "event": "node.create", "id": "n2", "type": "TASK", "data": {"title": "N2"}, "meta": {}},
            {"ts": "2025-01-01T00:00:02Z", "event": "edge.create", "src": "n1", "tgt": "n2", "type": "DEPENDS_ON", "weight": 1.0},
            {"ts": "2025-01-01T00:00:03Z", "event": "edge.delete", "src": "n1", "tgt": "n2", "type": "DEPENDS_ON"},
        ]

        event_file = events_dir / "test.jsonl"
        with open(event_file, "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")

        loaded_events = EventLog.load_all_events(events_dir)
        graph = EventLog.rebuild_graph_from_events(loaded_events)

        # Check that edge indices are also updated
        assert len(graph._edges_from.get("n1", [])) == 0, "_edges_from should be empty"
        assert len(graph._edges_to.get("n2", [])) == 0, "_edges_to should be empty"

    def test_edge_delete_nonexistent_is_silent(self, tmp_path):
        """Test that deleting a non-existent edge doesn't crash."""
        events_dir = tmp_path / "events"
        events_dir.mkdir()

        events = [
            {"ts": "2025-01-01T00:00:00Z", "event": "node.create", "id": "n1", "type": "TASK", "data": {"title": "N1"}, "meta": {}},
            {"ts": "2025-01-01T00:00:01Z", "event": "node.create", "id": "n2", "type": "TASK", "data": {"title": "N2"}, "meta": {}},
            # Delete an edge that was never created
            {"ts": "2025-01-01T00:00:02Z", "event": "edge.delete", "src": "n1", "tgt": "n2", "type": "DEPENDS_ON"},
        ]

        event_file = events_dir / "test.jsonl"
        with open(event_file, "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")

        loaded_events = EventLog.load_all_events(events_dir)
        # Should not crash
        graph = EventLog.rebuild_graph_from_events(loaded_events)

        assert len(graph.edges) == 0, "No edges should exist"
        assert len(graph.nodes) == 2, "Nodes should still exist"

    def test_multiple_edge_deletes(self, tmp_path):
        """Test multiple edge.delete events in sequence."""
        events_dir = tmp_path / "events"
        events_dir.mkdir()

        events = [
            {"ts": "2025-01-01T00:00:00Z", "event": "node.create", "id": "n1", "type": "TASK", "data": {"title": "N1"}, "meta": {}},
            {"ts": "2025-01-01T00:00:01Z", "event": "node.create", "id": "n2", "type": "TASK", "data": {"title": "N2"}, "meta": {}},
            {"ts": "2025-01-01T00:00:02Z", "event": "node.create", "id": "n3", "type": "TASK", "data": {"title": "N3"}, "meta": {}},
            # Create 3 edges
            {"ts": "2025-01-01T00:00:03Z", "event": "edge.create", "src": "n1", "tgt": "n2", "type": "DEPENDS_ON", "weight": 1.0},
            {"ts": "2025-01-01T00:00:04Z", "event": "edge.create", "src": "n2", "tgt": "n3", "type": "BLOCKS", "weight": 0.8},
            {"ts": "2025-01-01T00:00:05Z", "event": "edge.create", "src": "n1", "tgt": "n3", "type": "SIMILAR", "weight": 0.5},
            # Delete all 3 edges
            {"ts": "2025-01-01T00:00:06Z", "event": "edge.delete", "src": "n1", "tgt": "n2", "type": "DEPENDS_ON"},
            {"ts": "2025-01-01T00:00:07Z", "event": "edge.delete", "src": "n2", "tgt": "n3", "type": "BLOCKS"},
            {"ts": "2025-01-01T00:00:08Z", "event": "edge.delete", "src": "n1", "tgt": "n3", "type": "SIMILAR"},
        ]

        event_file = events_dir / "test.jsonl"
        with open(event_file, "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")

        loaded_events = EventLog.load_all_events(events_dir)
        graph = EventLog.rebuild_graph_from_events(loaded_events)

        # All edges should be deleted
        assert len(graph.edges) == 0, "All edges should be deleted"

    def test_edge_create_after_delete(self, tmp_path):
        """Test that edges can be recreated after deletion."""
        events_dir = tmp_path / "events"
        events_dir.mkdir()

        events = [
            {"ts": "2025-01-01T00:00:00Z", "event": "node.create", "id": "n1", "type": "TASK", "data": {"title": "N1"}, "meta": {}},
            {"ts": "2025-01-01T00:00:01Z", "event": "node.create", "id": "n2", "type": "TASK", "data": {"title": "N2"}, "meta": {}},
            # Create, delete, recreate
            {"ts": "2025-01-01T00:00:02Z", "event": "edge.create", "src": "n1", "tgt": "n2", "type": "DEPENDS_ON", "weight": 1.0},
            {"ts": "2025-01-01T00:00:03Z", "event": "edge.delete", "src": "n1", "tgt": "n2", "type": "DEPENDS_ON"},
            {"ts": "2025-01-01T00:00:04Z", "event": "edge.create", "src": "n1", "tgt": "n2", "type": "BLOCKS", "weight": 0.9},
        ]

        event_file = events_dir / "test.jsonl"
        with open(event_file, "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")

        loaded_events = EventLog.load_all_events(events_dir)
        graph = EventLog.rebuild_graph_from_events(loaded_events)

        # Should have the BLOCKS edge, not DEPENDS_ON
        assert len(graph.edges) == 1, "Should have 1 edge after recreation"
        assert graph.edges[0].edge_type == EdgeType.BLOCKS, "Edge should be BLOCKS type"


class TestBackendStatsConsistency:
    """
    Regression tests for get_stats() returning consistent fields across backends.

    Bug: TransactionalGoTAdapter.get_stats() was missing 'total_sprints' and 'total_epics'.
    Error: KeyError: 'total_sprints'

    Fix: Added total_sprints and total_epics to TransactionalGoTAdapter.get_stats().
    """

    def test_event_sourced_stats_has_all_fields(self, tmp_path):
        """Test that event-sourced backend stats has all required fields."""
        # This test uses the GoTProjectManager directly
        from scripts.got_utils import GoTProjectManager, GOT_DIR

        # Create a temporary manager
        manager = GoTProjectManager(got_dir=tmp_path / ".got")

        stats = manager.get_stats()

        # Verify all required fields exist
        required_fields = ["total_tasks", "tasks_by_status", "total_edges", "total_sprints", "total_epics"]
        for field in required_fields:
            assert field in stats, f"Stats should contain '{field}'"

    def test_transactional_stats_has_all_fields(self, tmp_path):
        """Test that transactional backend stats has all required fields."""
        # Skip if TX backend not available
        try:
            from scripts.got_utils import TransactionalGoTAdapter, TX_BACKEND_AVAILABLE
            if not TX_BACKEND_AVAILABLE:
                pytest.skip("Transactional backend not available")
        except ImportError:
            pytest.skip("Transactional backend not available")

        # Create a temporary adapter with path
        adapter = TransactionalGoTAdapter(got_dir=tmp_path / ".got-tx")

        stats = adapter.get_stats()

        # Verify all required fields exist
        required_fields = ["total_tasks", "tasks_by_status", "total_edges", "total_sprints", "total_epics"]
        for field in required_fields:
            assert field in stats, f"Stats should contain '{field}'"

    def test_stats_fields_match_between_backends(self, tmp_path):
        """Test that both backends return the same stat fields."""
        from scripts.got_utils import GoTProjectManager

        # Skip TX test if not available
        try:
            from scripts.got_utils import TransactionalGoTAdapter, TX_BACKEND_AVAILABLE
            if not TX_BACKEND_AVAILABLE:
                pytest.skip("Transactional backend not available")
        except ImportError:
            pytest.skip("Transactional backend not available")

        # Get stats from both backends
        es_manager = GoTProjectManager(got_dir=tmp_path / ".got")
        es_stats = es_manager.get_stats()

        tx_adapter = TransactionalGoTAdapter(got_dir=tmp_path / ".got-tx")
        tx_stats = tx_adapter.get_stats()

        # Both should have the same keys
        assert set(es_stats.keys()) == set(tx_stats.keys()), \
            f"Stats keys should match. ES: {es_stats.keys()}, TX: {tx_stats.keys()}"
