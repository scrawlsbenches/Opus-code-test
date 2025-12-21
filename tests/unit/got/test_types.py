"""
Unit tests for GoT types module.

Tests Entity, Task, Decision, and Edge classes for serialization,
validation, versioning, and checksum computation.
"""

import pytest
from datetime import datetime, timezone

from cortical.got.types import Entity, Task, Decision, Edge
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
            edge_type="SIMILAR",
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
                edge_type="TEST",
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
                edge_type="TEST",
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
                edge_type="TEST",
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
                edge_type="TEST",
                confidence=2.0,
            )
        assert "confidence must be in [0.0, 1.0]" in str(exc_info.value)

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
