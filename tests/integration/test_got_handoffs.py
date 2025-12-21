"""
Integration tests for GoT handoff system.

Tests the full handoff lifecycle across components:
- HandoffManager API
- EventLog persistence
- Graph rebuild from events
- Multi-agent coordination
"""

import json
import tempfile
from pathlib import Path
from datetime import datetime

import pytest

# Import the HandoffManager and related classes from got_utils
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from got_utils import (
    EventLog,
    HandoffManager,
    GoTProjectManager,
    generate_handoff_id,
)

from cortical.reasoning import (
    ThoughtGraph,
    NodeType,
    EdgeType,
)


class TestHandoffLifecycle:
    """Test complete handoff lifecycle: initiate → accept → complete."""

    def test_basic_handoff_lifecycle(self, tmp_path):
        """Test basic handoff from initiation to completion."""
        # Create event log and handoff manager
        event_log = EventLog(tmp_path / "events")
        manager = HandoffManager(event_log)

        # 1. Initiate handoff
        handoff_id = manager.initiate_handoff(
            source_agent="agent-a",
            target_agent="agent-b",
            task_id="task:T-001",
            context={"priority": "high", "files": ["test.py"]},
            instructions="Write integration tests for handoffs",
        )

        assert handoff_id.startswith("handoff:H-")
        assert manager._active_handoffs[handoff_id]["status"] == "initiated"

        # 2. Accept handoff
        result = manager.accept_handoff(
            handoff_id=handoff_id,
            agent="agent-b",
            acknowledgment="Starting work on integration tests",
        )

        assert result is True
        assert manager._active_handoffs[handoff_id]["status"] == "accepted"

        # 3. Add context during work
        manager.add_context(
            handoff_id=handoff_id,
            agent="agent-b",
            context_type="progress",
            data={"tests_written": 3, "tests_passing": 3},
        )

        # 4. Complete handoff
        result = manager.complete_handoff(
            handoff_id=handoff_id,
            agent="agent-b",
            result={
                "status": "success",
                "tests_written": 8,
                "file": "tests/integration/test_got_handoffs.py",
            },
            artifacts=["tests/integration/test_got_handoffs.py"],
        )

        assert result is True
        assert manager._active_handoffs[handoff_id]["status"] == "completed"
        assert manager._active_handoffs[handoff_id]["result"]["tests_written"] == 8

    def test_handoff_with_rejection(self, tmp_path):
        """Test handoff rejection workflow."""
        event_log = EventLog(tmp_path / "events")
        manager = HandoffManager(event_log)

        # Initiate handoff
        handoff_id = manager.initiate_handoff(
            source_agent="agent-a",
            target_agent="agent-b",
            task_id="task:T-002",
            context={"unclear_requirements": True},
            instructions="Implement feature without specifications",
        )

        # Agent rejects due to unclear requirements
        event_log.log_handoff_reject(
            handoff_id=handoff_id,
            agent="agent-b",
            reason="Requirements are unclear",
            suggestion="Please provide detailed specifications first",
        )

        # Load handoffs and verify rejection
        events = EventLog.load_all_events(tmp_path / "events")
        handoffs = HandoffManager.load_handoffs_from_events(events)

        rejected = [h for h in handoffs if h["id"] == handoff_id]
        assert len(rejected) == 1
        assert rejected[0]["status"] == "rejected"
        assert "unclear" in rejected[0]["reject_reason"].lower()


class TestEventLogPersistence:
    """Test that all handoff events are persisted correctly."""

    def test_all_handoff_events_logged(self, tmp_path):
        """Test all handoff events are captured in event log."""
        event_log = EventLog(tmp_path / "events")
        manager = HandoffManager(event_log)

        # Create complete handoff lifecycle
        handoff_id = manager.initiate_handoff(
            source_agent="tester",
            target_agent="implementer",
            task_id="task:T-003",
            context={"test": "data"},
            instructions="Test instructions",
        )

        manager.accept_handoff(handoff_id, "implementer", "Acknowledged")
        manager.add_context(handoff_id, "implementer", "progress", {"step": 1})
        manager.complete_handoff(
            handoff_id,
            "implementer",
            {"done": True},
            ["file1.py", "file2.py"],
        )

        # Load events and verify all types present
        events = EventLog.load_all_events(tmp_path / "events")

        event_types = {e["event"] for e in events}
        assert "handoff.initiate" in event_types
        assert "handoff.accept" in event_types
        assert "handoff.context" in event_types
        assert "handoff.complete" in event_types

        # Verify event data integrity
        initiate_event = next(e for e in events if e["event"] == "handoff.initiate")
        assert initiate_event["handoff_id"] == handoff_id
        assert initiate_event["source_agent"] == "tester"
        assert initiate_event["target_agent"] == "implementer"
        assert initiate_event["task_id"] == "task:T-003"
        assert initiate_event["context"]["test"] == "data"

        complete_event = next(e for e in events if e["event"] == "handoff.complete")
        assert complete_event["handoff_id"] == handoff_id
        assert complete_event["result"]["done"] is True
        assert "file1.py" in complete_event["artifacts"]

    def test_events_survive_restart(self, tmp_path):
        """Test events persist across manager restarts."""
        events_dir = tmp_path / "events"

        # Session 1: Create handoff
        event_log1 = EventLog(events_dir)
        manager1 = HandoffManager(event_log1)
        handoff_id = manager1.initiate_handoff(
            source_agent="a",
            target_agent="b",
            task_id="task:T-004",
            context={},
            instructions="Persist me",
        )
        manager1.accept_handoff(handoff_id, "b")

        # Session 2: Load events in new manager
        event_log2 = EventLog(events_dir)
        events = EventLog.load_all_events(events_dir)

        # Verify events from session 1 are available
        handoffs = HandoffManager.load_handoffs_from_events(events)
        assert len(handoffs) == 1
        assert handoffs[0]["id"] == handoff_id
        assert handoffs[0]["status"] == "accepted"


class TestGraphRebuildFromEvents:
    """Test graph can be rebuilt from handoff events."""

    def test_rebuild_includes_handoff_state(self, tmp_path):
        """Test graph rebuilt from events includes handoff state."""
        # Create GoTProjectManager which uses EventLog
        manager = GoTProjectManager(got_dir=tmp_path / ".got")

        # Create task nodes in graph AND log them as events
        manager.event_log.log_node_create("task:T-005", "TASK", {"title": "Implement feature"})
        manager.event_log.log_node_create("task:T-006", "TASK", {"title": "Write tests"})
        manager.graph.add_node("task:T-005", NodeType.TASK, "Implement feature")
        manager.graph.add_node("task:T-006", NodeType.TASK, "Write tests")

        # Create handoff between tasks
        handoff_id = generate_handoff_id()
        manager.event_log.log_handoff_initiate(
            handoff_id=handoff_id,
            source_agent="developer",
            target_agent="tester",
            task_id="task:T-005",
            context={"files": ["feature.py"]},
            instructions="Please test this feature",
        )

        manager.event_log.log_handoff_accept(
            handoff_id=handoff_id,
            agent="tester",
            acknowledgment="Testing started",
        )

        # Rebuild graph from events
        events = EventLog.load_all_events(tmp_path / ".got" / "events")
        rebuilt_graph = EventLog.rebuild_graph_from_events(events)

        # Verify tasks are in rebuilt graph
        assert rebuilt_graph.get_node("task:T-005") is not None
        assert rebuilt_graph.get_node("task:T-006") is not None

        # Verify we can reconstruct handoff state from events
        handoffs = HandoffManager.load_handoffs_from_events(events)
        assert len(handoffs) == 1
        assert handoffs[0]["id"] == handoff_id
        assert handoffs[0]["status"] == "accepted"
        assert handoffs[0]["task_id"] == "task:T-005"

    def test_multiple_handoffs_in_events(self, tmp_path):
        """Test multiple handoffs can be tracked in events."""
        event_log = EventLog(tmp_path / "events")

        # Create multiple handoffs
        h1 = generate_handoff_id()
        h2 = generate_handoff_id()
        h3 = generate_handoff_id()

        event_log.log_handoff_initiate(h1, "a1", "a2", "task:T-007", {}, "Work 1")
        event_log.log_handoff_initiate(h2, "a2", "a3", "task:T-008", {}, "Work 2")
        event_log.log_handoff_initiate(h3, "a3", "a1", "task:T-009", {}, "Work 3")

        event_log.log_handoff_accept(h1, "a2")
        event_log.log_handoff_accept(h2, "a3")
        # h3 not yet accepted

        event_log.log_handoff_complete(h1, "a2", {"done": True})
        # h2 in progress, h3 pending

        # Load and verify states
        events = EventLog.load_all_events(tmp_path / "events")
        handoffs = HandoffManager.load_handoffs_from_events(events)

        assert len(handoffs) == 3

        h1_state = next(h for h in handoffs if h["id"] == h1)
        h2_state = next(h for h in handoffs if h["id"] == h2)
        h3_state = next(h for h in handoffs if h["id"] == h3)

        assert h1_state["status"] == "completed"
        assert h2_state["status"] == "accepted"
        assert h3_state["status"] == "initiated"


class TestSequentialHandoffs:
    """Test multiple handoffs in sequence."""

    def test_handoff_chain(self, tmp_path):
        """Test a chain of handoffs: A → B → C → A."""
        event_log = EventLog(tmp_path / "events")
        manager = HandoffManager(event_log)

        # A hands off to B
        h1 = manager.initiate_handoff(
            source_agent="agent-a",
            target_agent="agent-b",
            task_id="task:T-010",
            context={"step": 1},
            instructions="Implement core logic",
        )
        manager.accept_handoff(h1, "agent-b")
        manager.complete_handoff(h1, "agent-b", {"code_written": True})

        # B hands off to C
        h2 = manager.initiate_handoff(
            source_agent="agent-b",
            target_agent="agent-c",
            task_id="task:T-010",  # Same task
            context={"step": 2, "previous_handoff": h1},
            instructions="Write integration tests",
        )
        manager.accept_handoff(h2, "agent-c")
        manager.complete_handoff(h2, "agent-c", {"tests_written": 5})

        # C hands back to A
        h3 = manager.initiate_handoff(
            source_agent="agent-c",
            target_agent="agent-a",
            task_id="task:T-010",
            context={"step": 3, "previous_handoff": h2},
            instructions="Review and merge",
        )
        manager.accept_handoff(h3, "agent-a")
        manager.complete_handoff(h3, "agent-a", {"merged": True})

        # Verify complete chain
        events = EventLog.load_all_events(tmp_path / "events")
        handoffs = HandoffManager.load_handoffs_from_events(events)

        assert len(handoffs) == 3
        assert all(h["status"] == "completed" for h in handoffs)

        # Verify sequence
        h1_data = next(h for h in handoffs if h["id"] == h1)
        h2_data = next(h for h in handoffs if h["id"] == h2)
        h3_data = next(h for h in handoffs if h["id"] == h3)

        assert h1_data["source_agent"] == "agent-a"
        assert h1_data["target_agent"] == "agent-b"
        assert h2_data["source_agent"] == "agent-b"
        assert h2_data["target_agent"] == "agent-c"
        assert h3_data["source_agent"] == "agent-c"
        assert h3_data["target_agent"] == "agent-a"


class TestContextPassing:
    """Test context passing between agents during handoffs."""

    def test_rich_context_transfer(self, tmp_path):
        """Test complex context data survives handoff."""
        event_log = EventLog(tmp_path / "events")
        manager = HandoffManager(event_log)

        # Complex context with nested data
        context = {
            "priority": "critical",
            "deadline": "2025-12-31",
            "files": [
                {"path": "src/main.py", "lines": [10, 25, 42]},
                {"path": "tests/test_main.py", "lines": [5, 15]},
            ],
            "related_tasks": ["task:T-001", "task:T-002"],
            "metadata": {
                "estimated_hours": 8,
                "complexity": "high",
                "tags": ["refactor", "performance"],
            },
        }

        handoff_id = manager.initiate_handoff(
            source_agent="planner",
            target_agent="implementer",
            task_id="task:T-011",
            context=context,
            instructions="Refactor with performance focus",
        )

        # Load events and verify context integrity
        events = EventLog.load_all_events(tmp_path / "events")
        initiate_event = next(
            e for e in events if e["event"] == "handoff.initiate" and e["handoff_id"] == handoff_id
        )

        # Verify nested structure preserved
        assert initiate_event["context"]["priority"] == "critical"
        assert len(initiate_event["context"]["files"]) == 2
        assert initiate_event["context"]["files"][0]["path"] == "src/main.py"
        assert 42 in initiate_event["context"]["files"][0]["lines"]
        assert initiate_event["context"]["metadata"]["estimated_hours"] == 8
        assert "refactor" in initiate_event["context"]["metadata"]["tags"]

    def test_incremental_context_addition(self, tmp_path):
        """Test adding context incrementally during handoff."""
        event_log = EventLog(tmp_path / "events")
        manager = HandoffManager(event_log)

        handoff_id = manager.initiate_handoff(
            source_agent="a",
            target_agent="b",
            task_id="task:T-012",
            context={"initial": True},
            instructions="Start work",
        )

        manager.accept_handoff(handoff_id, "b")

        # Add context as work progresses
        manager.add_context(
            handoff_id, "b", "discovery", {"found_issue": "Missing error handling"}
        )
        manager.add_context(
            handoff_id, "b", "progress", {"files_modified": 3, "tests_added": 5}
        )
        manager.add_context(
            handoff_id, "b", "blocker", {"needs_review": "Security implications"}
        )

        # Load and verify all context events
        events = EventLog.load_all_events(tmp_path / "events")
        context_events = [
            e for e in events if e["event"] == "handoff.context" and e["handoff_id"] == handoff_id
        ]

        assert len(context_events) == 3

        discovery = next(e for e in context_events if e["context_type"] == "discovery")
        assert "Missing error handling" in discovery["data"]["found_issue"]

        progress = next(e for e in context_events if e["context_type"] == "progress")
        assert progress["data"]["files_modified"] == 3

        blocker = next(e for e in context_events if e["context_type"] == "blocker")
        assert "Security" in blocker["data"]["needs_review"]


class TestHandoffTimestamps:
    """Test timestamp tracking throughout handoff lifecycle."""

    def test_timestamps_recorded(self, tmp_path):
        """Test all lifecycle timestamps are recorded."""
        event_log = EventLog(tmp_path / "events")
        manager = HandoffManager(event_log)

        # Create handoff with delays to ensure distinct timestamps
        handoff_id = manager.initiate_handoff(
            source_agent="a",
            target_agent="b",
            task_id="task:T-013",
            context={},
            instructions="Test timestamps",
        )

        import time
        time.sleep(0.01)  # Small delay

        manager.accept_handoff(handoff_id, "b")
        time.sleep(0.01)

        manager.complete_handoff(handoff_id, "b", {"done": True})

        # Load handoffs and check timestamps
        events = EventLog.load_all_events(tmp_path / "events")
        handoffs = HandoffManager.load_handoffs_from_events(events)
        handoff = handoffs[0]

        assert "initiated_at" in handoff
        assert "accepted_at" in handoff
        assert "completed_at" in handoff

        # Verify temporal ordering
        initiated = datetime.fromisoformat(handoff["initiated_at"].replace("Z", ""))
        accepted = datetime.fromisoformat(handoff["accepted_at"].replace("Z", ""))
        completed = datetime.fromisoformat(handoff["completed_at"].replace("Z", ""))

        assert initiated <= accepted <= completed


class TestConcurrentHandoffs:
    """Test multiple concurrent handoffs."""

    def test_parallel_handoffs_independent(self, tmp_path):
        """Test multiple handoffs can run in parallel independently."""
        event_log = EventLog(tmp_path / "events")
        manager = HandoffManager(event_log)

        # Create 3 parallel handoffs for different tasks
        h1 = manager.initiate_handoff("a1", "b1", "task:T-014", {}, "Work 1")
        h2 = manager.initiate_handoff("a2", "b2", "task:T-015", {}, "Work 2")
        h3 = manager.initiate_handoff("a3", "b3", "task:T-016", {}, "Work 3")

        # Accept in different order
        manager.accept_handoff(h2, "b2")
        manager.accept_handoff(h1, "b1")
        manager.accept_handoff(h3, "b3")

        # Complete in different order
        manager.complete_handoff(h3, "b3", {"result": 3})
        manager.complete_handoff(h1, "b1", {"result": 1})
        manager.complete_handoff(h2, "b2", {"result": 2})

        # Verify all completed independently
        events = EventLog.load_all_events(tmp_path / "events")
        handoffs = HandoffManager.load_handoffs_from_events(events)

        assert len(handoffs) == 3
        assert all(h["status"] == "completed" for h in handoffs)

        # Verify results match
        results = {h["id"]: h["result"]["result"] for h in handoffs}
        assert results[h1] == 1
        assert results[h2] == 2
        assert results[h3] == 3


class TestErrorHandling:
    """Test error handling in handoff workflows."""

    def test_accept_nonexistent_handoff(self, tmp_path):
        """Test accepting a handoff that doesn't exist doesn't crash."""
        event_log = EventLog(tmp_path / "events")
        manager = HandoffManager(event_log)

        # Accept non-existent handoff (should not raise)
        result = manager.accept_handoff("handoff:H-nonexistent", "agent")
        assert result is True  # Returns True but no state change

    def test_multiple_accepts_same_handoff(self, tmp_path):
        """Test accepting same handoff multiple times."""
        event_log = EventLog(tmp_path / "events")
        manager = HandoffManager(event_log)

        handoff_id = manager.initiate_handoff("a", "b", "task:T-017", {}, "Test")

        # Accept multiple times
        manager.accept_handoff(handoff_id, "b", "First accept")
        manager.accept_handoff(handoff_id, "b", "Second accept")

        # Should still be in accepted state
        assert manager._active_handoffs[handoff_id]["status"] == "accepted"

        # Events should record both accepts
        events = EventLog.load_all_events(tmp_path / "events")
        accept_events = [
            e for e in events if e["event"] == "handoff.accept" and e["handoff_id"] == handoff_id
        ]
        assert len(accept_events) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
