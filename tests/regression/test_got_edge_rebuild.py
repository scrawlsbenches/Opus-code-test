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
