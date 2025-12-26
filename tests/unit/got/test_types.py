"""
Unit tests for GoT types module.

Tests Entity, Task, Decision, Edge, Sprint, Epic, and Handoff classes for serialization,
validation, versioning, and checksum computation.
"""

import pytest
from datetime import datetime, timezone

from cortical.got.types import Entity, Task, Decision, Edge, Sprint, Epic, Handoff, VALID_EDGE_TYPES
from cortical.got.errors import ValidationError


class TestEntity:
    """Test base Entity class functionality."""

    def test_entity_creation(self):
        """Test basic entity creation."""
        entity = Entity(id="test-1", entity_type="test")
        assert entity.id == "test-1"
        assert entity.entity_type == "test"
        assert entity.version == 1
        assert entity.created_at
        assert entity.modified_at

    def test_entity_to_dict(self):
        """Test entity serialization to dictionary."""
        entity = Entity(id="test-1", entity_type="test")
        data = entity.to_dict()

        assert data["id"] == "test-1"
        assert data["entity_type"] == "test"
        assert data["version"] == 1
        assert "created_at" in data
        assert "modified_at" in data

    def test_entity_from_dict(self):
        """Test entity deserialization from dictionary."""
        data = {
            "id": "test-1",
            "entity_type": "test",
            "version": 2,
            "created_at": "2025-01-01T00:00:00Z",
            "modified_at": "2025-01-02T00:00:00Z",
        }
        entity = Entity.from_dict(data)

        assert entity.id == "test-1"
        assert entity.entity_type == "test"
        assert entity.version == 2
        assert entity.created_at == "2025-01-01T00:00:00Z"
        assert entity.modified_at == "2025-01-02T00:00:00Z"

    def test_entity_roundtrip(self):
        """Test to_dict/from_dict round-trip preserves data."""
        entity1 = Entity(id="test-1", entity_type="test", version=5)
        data = entity1.to_dict()
        entity2 = Entity.from_dict(data)

        assert entity2.id == entity1.id
        assert entity2.entity_type == entity1.entity_type
        assert entity2.version == entity1.version
        assert entity2.created_at == entity1.created_at
        assert entity2.modified_at == entity1.modified_at

    def test_entity_bump_version(self):
        """Test version bumping increments version and updates modified_at."""
        entity = Entity(id="test-1", entity_type="test")
        original_version = entity.version
        original_modified = entity.modified_at

        entity.bump_version()

        assert entity.version == original_version + 1
        assert entity.modified_at != original_modified

    def test_entity_compute_checksum(self):
        """Test checksum computation returns consistent hash."""
        entity = Entity(id="test-1", entity_type="test", version=1)
        checksum1 = entity.compute_checksum()
        checksum2 = entity.compute_checksum()

        assert checksum1 == checksum2
        assert isinstance(checksum1, str)
        assert len(checksum1) == 16  # First 16 chars of SHA256

    def test_entity_checksum_changes_with_data(self):
        """Test checksum changes when entity data changes."""
        entity = Entity(id="test-1", entity_type="test", version=1)
        checksum1 = entity.compute_checksum()

        entity.bump_version()
        checksum2 = entity.compute_checksum()

        assert checksum1 != checksum2


class TestTask:
    """Test Task entity functionality."""

    def test_task_creation(self):
        """Test basic task creation."""
        task = Task(id="T-001", title="Test task")
        assert task.id == "T-001"
        assert task.title == "Test task"
        assert task.entity_type == "task"
        assert task.status == "pending"
        assert task.priority == "medium"
        assert task.description == ""
        assert task.properties == {}
        assert task.metadata == {}

    def test_task_with_all_fields(self):
        """Test task creation with all fields."""
        task = Task(
            id="T-001",
            title="Test task",
            status="in_progress",
            priority="high",
            description="A test task",
            properties={"key": "value"},
            metadata={"tags": ["test"]},
        )
        assert task.status == "in_progress"
        assert task.priority == "high"
        assert task.description == "A test task"
        assert task.properties["key"] == "value"
        assert task.metadata["tags"] == ["test"]

    def test_task_to_dict(self):
        """Test task serialization."""
        task = Task(
            id="T-001",
            title="Test task",
            status="completed",
            priority="critical",
        )
        data = task.to_dict()

        assert data["id"] == "T-001"
        assert data["title"] == "Test task"
        assert data["status"] == "completed"
        assert data["priority"] == "critical"
        assert data["entity_type"] == "task"

    def test_task_from_dict(self):
        """Test task deserialization."""
        data = {
            "id": "T-001",
            "title": "Test task",
            "status": "blocked",
            "priority": "low",
            "description": "Description",
            "properties": {"key": "value"},
            "metadata": {"tags": ["test"]},
        }
        task = Task.from_dict(data)

        assert task.id == "T-001"
        assert task.title == "Test task"
        assert task.status == "blocked"
        assert task.priority == "low"
        assert task.description == "Description"
        assert task.properties == {"key": "value"}
        assert task.metadata == {"tags": ["test"]}

    def test_task_roundtrip(self):
        """Test task to_dict/from_dict round-trip."""
        task1 = Task(
            id="T-001",
            title="Test",
            status="in_progress",
            priority="high",
            properties={"nested": {"key": "value"}},
        )
        data = task1.to_dict()
        task2 = Task.from_dict(data)

        assert task2.id == task1.id
        assert task2.title == task1.title
        assert task2.status == task1.status
        assert task2.priority == task1.priority
        assert task2.properties == task1.properties

    def test_task_invalid_status(self):
        """Test that invalid status raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Task(id="T-001", title="Test", status="invalid_status")
        assert "Invalid status" in str(exc_info.value)

    def test_task_invalid_priority(self):
        """Test that invalid priority raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Task(id="T-001", title="Test", priority="invalid_priority")
        assert "Invalid priority" in str(exc_info.value)


class TestDecision:
    """Test Decision entity functionality."""

    def test_decision_creation(self):
        """Test basic decision creation."""
        decision = Decision(id="D-001", title="Test decision")
        assert decision.id == "D-001"
        assert decision.title == "Test decision"
        assert decision.entity_type == "decision"
        assert decision.rationale == ""
        assert decision.affects == []
        assert decision.properties == {}

    def test_decision_with_all_fields(self):
        """Test decision creation with all fields."""
        decision = Decision(
            id="D-001",
            title="Use PostgreSQL",
            rationale="Better performance",
            affects=["T-001", "T-002"],
            properties={"impact": "high"},
        )
        assert decision.title == "Use PostgreSQL"
        assert decision.rationale == "Better performance"
        assert decision.affects == ["T-001", "T-002"]
        assert decision.properties["impact"] == "high"

    def test_decision_to_dict(self):
        """Test decision serialization."""
        decision = Decision(
            id="D-001",
            title="Test",
            rationale="Because",
            affects=["T-001"],
        )
        data = decision.to_dict()

        assert data["id"] == "D-001"
        assert data["title"] == "Test"
        assert data["rationale"] == "Because"
        assert data["affects"] == ["T-001"]
        assert data["entity_type"] == "decision"

    def test_decision_from_dict(self):
        """Test decision deserialization."""
        data = {
            "id": "D-001",
            "title": "Test",
            "rationale": "Rationale",
            "affects": ["T-001", "T-002"],
            "properties": {"key": "value"},
        }
        decision = Decision.from_dict(data)

        assert decision.id == "D-001"
        assert decision.title == "Test"
        assert decision.rationale == "Rationale"
        assert decision.affects == ["T-001", "T-002"]
        assert decision.properties == {"key": "value"}

    def test_decision_roundtrip(self):
        """Test decision to_dict/from_dict round-trip."""
        decision1 = Decision(
            id="D-001",
            title="Test",
            rationale="Reason",
            affects=["T-001", "T-002", "T-003"],
        )
        data = decision1.to_dict()
        decision2 = Decision.from_dict(data)

        assert decision2.id == decision1.id
        assert decision2.title == decision1.title
        assert decision2.rationale == decision1.rationale
        assert decision2.affects == decision1.affects


class TestEdge:
    """Test Edge entity functionality."""

    def test_edge_auto_generated_id(self):
        """Test edge with auto-generated ID."""
        edge = Edge(
            id="",  # Empty triggers auto-generation
            source_id="T-001",
            target_id="T-002",
            edge_type="DEPENDS_ON",
        )
        assert edge.id == "E-T-001-T-002-DEPENDS_ON"
        assert edge.entity_type == "edge"

    def test_edge_explicit_id(self):
        """Test edge with explicit ID."""
        edge = Edge(
            id="CUSTOM-001",
            source_id="T-001",
            target_id="T-002",
            edge_type="BLOCKS",
        )
        assert edge.id == "CUSTOM-001"

    def test_edge_with_weights(self):
        """Test edge with custom weight and confidence."""
        edge = Edge(
            id="E-001",
            source_id="T-001",
            target_id="T-002",
            edge_type="RELATES_TO",
            weight=0.75,
            confidence=0.9,
        )
        assert edge.weight == 0.75
        assert edge.confidence == 0.9

    def test_edge_invalid_weight_too_low(self):
        """Test that weight < 0.0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Edge(
                id="E-001",
                source_id="T-001",
                target_id="T-002",
                edge_type="DEPENDS_ON",
                weight=-0.1,
            )
        assert "weight must be in [0.0, 1.0]" in str(exc_info.value)

    def test_edge_invalid_weight_too_high(self):
        """Test that weight > 1.0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Edge(
                id="E-001",
                source_id="T-001",
                target_id="T-002",
                edge_type="DEPENDS_ON",
                weight=1.5,
            )
        assert "weight must be in [0.0, 1.0]" in str(exc_info.value)

    def test_edge_invalid_confidence_too_low(self):
        """Test that confidence < 0.0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Edge(
                id="E-001",
                source_id="T-001",
                target_id="T-002",
                edge_type="DEPENDS_ON",
                confidence=-0.5,
            )
        assert "confidence must be in [0.0, 1.0]" in str(exc_info.value)

    def test_edge_invalid_confidence_too_high(self):
        """Test that confidence > 1.0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Edge(
                id="E-001",
                source_id="T-001",
                target_id="T-002",
                edge_type="DEPENDS_ON",
                confidence=2.0,
            )
        assert "confidence must be in [0.0, 1.0]" in str(exc_info.value)

    # Edge type validation tests
    def test_edge_valid_edge_types(self):
        """Test that all valid edge types are accepted."""
        for edge_type in VALID_EDGE_TYPES:
            edge = Edge(
                id="E-001",
                source_id="T-001",
                target_id="T-002",
                edge_type=edge_type,
            )
            assert edge.edge_type == edge_type

    def test_edge_invalid_edge_type(self):
        """Test that invalid edge types raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Edge(
                id="E-001",
                source_id="T-001",
                target_id="T-002",
                edge_type="INVALID_TYPE",
            )
        assert "Invalid edge_type" in str(exc_info.value)
        assert "INVALID_TYPE" in str(exc_info.value)

    def test_edge_empty_edge_type(self):
        """Test that empty edge type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Edge(
                id="E-001",
                source_id="T-001",
                target_id="T-002",
                edge_type="",
            )
        assert "Invalid edge_type" in str(exc_info.value)

    def test_edge_case_sensitive_edge_type(self):
        """Test that edge types are case-sensitive (lowercase is invalid)."""
        with pytest.raises(ValidationError) as exc_info:
            Edge(
                id="E-001",
                source_id="T-001",
                target_id="T-002",
                edge_type="depends_on",  # Should be DEPENDS_ON
            )
        assert "Invalid edge_type" in str(exc_info.value)

    def test_edge_to_dict(self):
        """Test edge serialization."""
        edge = Edge(
            id="E-001",
            source_id="T-001",
            target_id="T-002",
            edge_type="CONTAINS",
            weight=0.8,
            confidence=0.95,
        )
        data = edge.to_dict()

        assert data["id"] == "E-001"
        assert data["source_id"] == "T-001"
        assert data["target_id"] == "T-002"
        assert data["edge_type"] == "CONTAINS"
        assert data["weight"] == 0.8
        assert data["confidence"] == 0.95
        assert data["entity_type"] == "edge"

    def test_edge_from_dict_with_auto_id(self):
        """Test edge deserialization with auto-generated ID."""
        data = {
            "source_id": "T-001",
            "target_id": "T-002",
            "edge_type": "REQUIRES",
            "weight": 1.0,
            "confidence": 1.0,
        }
        edge = Edge.from_dict(data)

        assert edge.id == "E-T-001-T-002-REQUIRES"
        assert edge.source_id == "T-001"
        assert edge.target_id == "T-002"
        assert edge.edge_type == "REQUIRES"

    def test_edge_roundtrip(self):
        """Test edge to_dict/from_dict round-trip."""
        edge1 = Edge(
            id="",
            source_id="T-001",
            target_id="T-002",
            edge_type="BLOCKS",
            weight=0.5,
            confidence=0.7,
        )
        data = edge1.to_dict()
        edge2 = Edge.from_dict(data)

        assert edge2.id == edge1.id
        assert edge2.source_id == edge1.source_id
        assert edge2.target_id == edge1.target_id
        assert edge2.edge_type == edge1.edge_type
        assert edge2.weight == edge1.weight
        assert edge2.confidence == edge1.confidence


class TestSprint:
    """Test Sprint entity functionality."""

    def test_sprint_creation(self):
        """Test basic sprint creation."""
        sprint = Sprint(id="S-001", title="Sprint 1")
        assert sprint.id == "S-001"
        assert sprint.title == "Sprint 1"
        assert sprint.entity_type == "sprint"
        assert sprint.status == "available"
        assert sprint.epic_id == ""
        assert sprint.number == 0
        assert sprint.session_id == ""
        assert sprint.isolation == []
        assert sprint.goals == []
        assert sprint.notes == []
        assert sprint.properties == {}
        assert sprint.metadata == {}

    def test_sprint_with_all_fields(self):
        """Test sprint creation with all fields."""
        sprint = Sprint(
            id="S-001",
            title="Authentication Sprint",
            status="in_progress",
            epic_id="E-001",
            number=7,
            session_id="abc123",
            isolation=["cortical/auth/", "tests/test_auth.py"],
            goals=[
                {"text": "Implement JWT", "status": "completed"},
                {"text": "Add OAuth2", "status": "in_progress"},
            ],
            notes=["Started with JWT", "OAuth2 next"],
            properties={"priority": "high"},
            metadata={"team": "backend"},
        )
        assert sprint.status == "in_progress"
        assert sprint.epic_id == "E-001"
        assert sprint.number == 7
        assert sprint.session_id == "abc123"
        assert len(sprint.isolation) == 2
        assert len(sprint.goals) == 2
        assert len(sprint.notes) == 2
        assert sprint.properties["priority"] == "high"
        assert sprint.metadata["team"] == "backend"

    def test_sprint_to_dict(self):
        """Test sprint serialization."""
        sprint = Sprint(
            id="S-001",
            title="Test Sprint",
            status="completed",
            number=5,
        )
        data = sprint.to_dict()

        assert data["id"] == "S-001"
        assert data["title"] == "Test Sprint"
        assert data["status"] == "completed"
        assert data["number"] == 5
        assert data["entity_type"] == "sprint"

    def test_sprint_from_dict(self):
        """Test sprint deserialization."""
        data = {
            "id": "S-001",
            "title": "Test Sprint",
            "status": "blocked",
            "epic_id": "E-001",
            "number": 3,
            "session_id": "xyz789",
            "isolation": ["cortical/"],
            "goals": [{"text": "Goal 1", "status": "pending"}],
            "notes": ["Note 1"],
            "properties": {"key": "value"},
            "metadata": {"tag": "test"},
        }
        sprint = Sprint.from_dict(data)

        assert sprint.id == "S-001"
        assert sprint.title == "Test Sprint"
        assert sprint.status == "blocked"
        assert sprint.epic_id == "E-001"
        assert sprint.number == 3
        assert sprint.session_id == "xyz789"
        assert sprint.isolation == ["cortical/"]
        assert sprint.goals == [{"text": "Goal 1", "status": "pending"}]
        assert sprint.notes == ["Note 1"]
        assert sprint.properties == {"key": "value"}
        assert sprint.metadata == {"tag": "test"}

    def test_sprint_roundtrip(self):
        """Test sprint to_dict/from_dict round-trip."""
        sprint1 = Sprint(
            id="S-001",
            title="Test",
            status="in_progress",
            epic_id="E-001",
            number=5,
            goals=[{"text": "Goal", "status": "pending"}],
        )
        data = sprint1.to_dict()
        sprint2 = Sprint.from_dict(data)

        assert sprint2.id == sprint1.id
        assert sprint2.title == sprint1.title
        assert sprint2.status == sprint1.status
        assert sprint2.epic_id == sprint1.epic_id
        assert sprint2.number == sprint1.number
        assert sprint2.goals == sprint1.goals

    def test_sprint_invalid_status(self):
        """Test that invalid status raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Sprint(id="S-001", title="Test", status="invalid_status")
        assert "Invalid status" in str(exc_info.value)

    def test_sprint_all_valid_statuses(self):
        """Test all valid sprint statuses."""
        valid_statuses = ["available", "in_progress", "completed", "blocked"]
        for status in valid_statuses:
            sprint = Sprint(id=f"S-{status}", title="Test", status=status)
            assert sprint.status == status


class TestEpic:
    """Test Epic entity functionality."""

    def test_epic_creation(self):
        """Test basic epic creation."""
        epic = Epic(id="E-001", title="Big Initiative")
        assert epic.id == "E-001"
        assert epic.title == "Big Initiative"
        assert epic.entity_type == "epic"
        assert epic.status == "active"
        assert epic.phase == 1
        assert epic.phases == []
        assert epic.properties == {}
        assert epic.metadata == {}

    def test_epic_with_all_fields(self):
        """Test epic creation with all fields."""
        epic = Epic(
            id="E-001",
            title="Security Hardening",
            status="active",
            phase=2,
            phases=[
                {"number": 1, "title": "Authentication", "status": "completed"},
                {"number": 2, "title": "Authorization", "status": "in_progress"},
                {"number": 3, "title": "Audit Logging", "status": "planned"},
            ],
            properties={"priority": "critical"},
            metadata={"quarter": "Q1-2025"},
        )
        assert epic.status == "active"
        assert epic.phase == 2
        assert len(epic.phases) == 3
        assert epic.phases[0]["title"] == "Authentication"
        assert epic.properties["priority"] == "critical"
        assert epic.metadata["quarter"] == "Q1-2025"

    def test_epic_to_dict(self):
        """Test epic serialization."""
        epic = Epic(
            id="E-001",
            title="Test Epic",
            status="completed",
            phase=3,
        )
        data = epic.to_dict()

        assert data["id"] == "E-001"
        assert data["title"] == "Test Epic"
        assert data["status"] == "completed"
        assert data["phase"] == 3
        assert data["entity_type"] == "epic"

    def test_epic_from_dict(self):
        """Test epic deserialization."""
        data = {
            "id": "E-001",
            "title": "Test Epic",
            "status": "on_hold",
            "phase": 2,
            "phases": [
                {"number": 1, "title": "Phase 1", "status": "completed"},
                {"number": 2, "title": "Phase 2", "status": "active"},
            ],
            "properties": {"key": "value"},
            "metadata": {"tag": "test"},
        }
        epic = Epic.from_dict(data)

        assert epic.id == "E-001"
        assert epic.title == "Test Epic"
        assert epic.status == "on_hold"
        assert epic.phase == 2
        assert len(epic.phases) == 2
        assert epic.phases[0]["title"] == "Phase 1"
        assert epic.properties == {"key": "value"}
        assert epic.metadata == {"tag": "test"}

    def test_epic_roundtrip(self):
        """Test epic to_dict/from_dict round-trip."""
        epic1 = Epic(
            id="E-001",
            title="Test",
            status="active",
            phase=2,
            phases=[{"number": 1, "title": "Phase 1"}],
        )
        data = epic1.to_dict()
        epic2 = Epic.from_dict(data)

        assert epic2.id == epic1.id
        assert epic2.title == epic1.title
        assert epic2.status == epic1.status
        assert epic2.phase == epic1.phase
        assert epic2.phases == epic1.phases

    def test_epic_invalid_status(self):
        """Test that invalid status raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Epic(id="E-001", title="Test", status="invalid_status")
        assert "Invalid status" in str(exc_info.value)

    def test_epic_all_valid_statuses(self):
        """Test all valid epic statuses."""
        valid_statuses = ["active", "completed", "on_hold"]
        for status in valid_statuses:
            epic = Epic(id=f"E-{status}", title="Test", status=status)
            assert epic.status == status


class TestHandoff:
    """Test Handoff entity functionality."""

    def test_handoff_creation(self):
        """Test basic handoff creation."""
        handoff = Handoff(id="H-001", source_agent="agent-a", target_agent="agent-b")
        assert handoff.id == "H-001"
        assert handoff.source_agent == "agent-a"
        assert handoff.target_agent == "agent-b"
        assert handoff.status == "initiated"
        assert handoff.entity_type == "handoff"

    def test_handoff_with_all_fields(self):
        """Test handoff with all fields populated."""
        handoff = Handoff(
            id="H-002",
            source_agent="agent-a",
            target_agent="agent-b",
            task_id="T-001",
            status="accepted",
            instructions="Complete this task",
            context={"priority": "high"},
            result={"completed": True},
            artifacts=["file1.py", "file2.py"],
            initiated_at="2025-12-22T10:00:00Z",
            accepted_at="2025-12-22T10:05:00Z",
            properties={"custom": "value"},
        )
        assert handoff.source_agent == "agent-a"
        assert handoff.target_agent == "agent-b"
        assert handoff.task_id == "T-001"
        assert handoff.status == "accepted"
        assert handoff.instructions == "Complete this task"
        assert handoff.context == {"priority": "high"}
        assert handoff.result == {"completed": True}
        assert handoff.artifacts == ["file1.py", "file2.py"]
        assert handoff.accepted_at == "2025-12-22T10:05:00Z"
        assert handoff.properties == {"custom": "value"}

    def test_handoff_to_dict(self):
        """Test handoff serialization to dictionary."""
        handoff = Handoff(
            id="H-003",
            source_agent="agent-a",
            target_agent="agent-b",
            task_id="T-001",
            instructions="Do the thing",
        )
        data = handoff.to_dict()

        assert data["id"] == "H-003"
        assert data["source_agent"] == "agent-a"
        assert data["target_agent"] == "agent-b"
        assert data["task_id"] == "T-001"
        assert data["status"] == "initiated"
        assert data["instructions"] == "Do the thing"
        assert data["entity_type"] == "handoff"

    def test_handoff_from_dict(self):
        """Test handoff deserialization from dictionary."""
        data = {
            "id": "H-004",
            "entity_type": "handoff",
            "source_agent": "agent-a",
            "target_agent": "agent-b",
            "task_id": "T-002",
            "status": "completed",
            "instructions": "Finish it",
            "result": {"status": "success"},
            "artifacts": ["output.txt"],
            "completed_at": "2025-12-22T11:00:00Z",
        }

        handoff = Handoff.from_dict(data)
        assert handoff.id == "H-004"
        assert handoff.source_agent == "agent-a"
        assert handoff.target_agent == "agent-b"
        assert handoff.status == "completed"
        assert handoff.result == {"status": "success"}
        assert handoff.artifacts == ["output.txt"]
        assert handoff.completed_at == "2025-12-22T11:00:00Z"

    def test_handoff_roundtrip(self):
        """Test handoff serialization roundtrip."""
        handoff1 = Handoff(
            id="H-005",
            source_agent="agent-a",
            target_agent="agent-b",
            task_id="T-003",
            status="rejected",
            reject_reason="Not enough context",
        )
        data = handoff1.to_dict()
        handoff2 = Handoff.from_dict(data)

        assert handoff1.id == handoff2.id
        assert handoff1.source_agent == handoff2.source_agent
        assert handoff1.target_agent == handoff2.target_agent
        assert handoff1.status == handoff2.status
        assert handoff1.reject_reason == handoff2.reject_reason

    def test_handoff_invalid_status(self):
        """Test that invalid status raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Handoff(id="H-001", source_agent="a", target_agent="b", status="invalid")
        assert "Invalid status" in str(exc_info.value)

    def test_handoff_all_valid_statuses(self):
        """Test all valid handoff statuses."""
        valid_statuses = ["initiated", "accepted", "completed", "rejected"]
        for status in valid_statuses:
            handoff = Handoff(
                id=f"H-{status}",
                source_agent="a",
                target_agent="b",
                status=status
            )
            assert handoff.status == status

    def test_handoff_auto_initiated_at(self):
        """Test that initiated_at is auto-set if not provided."""
        handoff = Handoff(id="H-auto", source_agent="a", target_agent="b")
        assert handoff.initiated_at != ""
        # Should be a valid ISO timestamp
        assert "T" in handoff.initiated_at
