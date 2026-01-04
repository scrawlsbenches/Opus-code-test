"""
Tests for entity validation module.

Tests comprehensive validation of:
1. ID format validation for all entity types
2. Legacy format rejection
3. Relationship rules
4. Self-reference prevention
"""

import pytest

from tests.conftest import _create_got_manager
from cortical.got.validation import (
    validate_entity_id,
    validate_edge_relationship,
    validate_sprint_id_current_format,
    infer_entity_type_from_id,
    EntityIdValidator,
    RelationshipRules,
    ID_PATTERNS,
    RELATIONSHIP_RULES,
    LEGACY_PATTERNS,
)


class TestInferEntityType:
    """Tests for entity type inference from ID prefix."""

    def test_task_id(self):
        assert infer_entity_type_from_id("T-20251228-093045-a1b2c3d4") == "task"

    def test_decision_id(self):
        assert infer_entity_type_from_id("D-20251228-093045-a1b2c3d4") == "decision"

    def test_sprint_id(self):
        assert infer_entity_type_from_id("S-20251228-093045-a1b2c3d4") == "sprint"

    def test_epic_id_named(self):
        assert infer_entity_type_from_id("EPIC-woven-mind") == "epic"

    def test_epic_id_timestamped(self):
        assert infer_entity_type_from_id("EPIC-20251228-093045-a1b2c3d4") == "epic"

    def test_handoff_id(self):
        assert infer_entity_type_from_id("H-20251228-093045-a1b2c3d4") == "handoff"

    def test_edge_id(self):
        assert infer_entity_type_from_id("E-src-tgt-DEPENDS_ON") == "edge"

    def test_document_id(self):
        assert infer_entity_type_from_id("DOC-docs-architecture-md") == "document"

    def test_claudemd_layer_id(self):
        assert infer_entity_type_from_id("CML2-architecture-20251228-093045-a1b2c3d4") == "claudemd_layer"

    def test_team_id(self):
        assert infer_entity_type_from_id("TEAM-20251228-093045-a1b2c3d4") == "team"

    def test_persona_profile_id(self):
        assert infer_entity_type_from_id("PP-20251228-093045-a1b2c3d4") == "persona_profile"

    def test_goal_id(self):
        assert infer_entity_type_from_id("G-20251228-a1b2c3d4") == "goal"

    def test_unknown_id(self):
        assert infer_entity_type_from_id("UNKNOWN-12345") is None


class TestValidateEntityId:
    """Tests for entity ID validation."""

    def test_valid_task_id(self):
        entity_type = validate_entity_id("T-20251228-093045-a1b2c3d4")
        assert entity_type == "task"

    def test_valid_task_id_with_expected_type(self):
        entity_type = validate_entity_id("T-20251228-093045-a1b2c3d4", expected_type="task")
        assert entity_type == "task"

    def test_wrong_expected_type(self):
        with pytest.raises(ValueError, match="has type 'task', expected 'decision'"):
            validate_entity_id("T-20251228-093045-a1b2c3d4", expected_type="decision")

    def test_empty_id_rejected(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_entity_id("")

    def test_unknown_prefix_rejected(self):
        with pytest.raises(ValueError, match="Unrecognized entity ID format"):
            validate_entity_id("UNKNOWN-12345")

    def test_valid_sprint_id(self):
        entity_type = validate_entity_id("S-20251228-093045-a1b2c3d4")
        assert entity_type == "sprint"

    def test_legacy_sprint_short_rejected(self):
        with pytest.raises(ValueError, match="Cannot use legacy ID"):
            validate_entity_id("S-025", strict=True)

    def test_legacy_sprint_verbose_rejected(self):
        with pytest.raises(ValueError, match="Cannot use legacy ID"):
            validate_entity_id("S-sprint-017-spark-slm", strict=True)

    def test_valid_epic_named(self):
        entity_type = validate_entity_id("EPIC-woven-mind")
        assert entity_type == "epic"

    def test_valid_handoff(self):
        entity_type = validate_entity_id("H-20251228-093045-a1b2c3d4")
        assert entity_type == "handoff"


class TestValidateSprintIdCurrentFormat:
    """Tests for strict sprint ID format validation."""

    def test_current_format_accepted(self):
        # Should not raise
        validate_sprint_id_current_format("S-20251228-093045-a1b2c3d4")

    def test_non_sprint_id_ignored(self):
        # Should not raise for non-sprint IDs
        validate_sprint_id_current_format("T-20251228-093045-a1b2c3d4")
        validate_sprint_id_current_format("EPIC-woven-mind")
        validate_sprint_id_current_format("D-20251228-093045-a1b2c3d4")

    def test_legacy_short_format_rejected(self):
        with pytest.raises(ValueError, match="Legacy short format"):
            validate_sprint_id_current_format("S-025")

    def test_legacy_short_format_single_digit_rejected(self):
        with pytest.raises(ValueError, match="Legacy short format"):
            validate_sprint_id_current_format("S-1")

    def test_legacy_verbose_format_rejected(self):
        with pytest.raises(ValueError, match="Legacy verbose format"):
            validate_sprint_id_current_format("S-sprint-017-spark-slm")

    def test_legacy_verbose_format_simple_rejected(self):
        with pytest.raises(ValueError, match="Legacy verbose format"):
            validate_sprint_id_current_format("S-sprint-25")

    def test_error_message_includes_guidance(self):
        try:
            validate_sprint_id_current_format("S-025")
            assert False, "Should have raised"
        except ValueError as e:
            assert "got_utils.py sprint create" in str(e)


class TestValidateEdgeRelationship:
    """Tests for edge relationship validation."""

    def test_valid_task_depends_on_task(self):
        # Should not raise
        validate_edge_relationship(
            "T-20251228-093045-a1b2c3d4",
            "T-20251228-093046-b2c3d4e5",
            "DEPENDS_ON"
        )

    def test_valid_sprint_contains_task(self):
        # Should not raise
        validate_edge_relationship(
            "S-20251228-093045-a1b2c3d4",
            "T-20251228-093046-b2c3d4e5",
            "CONTAINS"
        )

    def test_valid_task_part_of_sprint(self):
        # Should not raise
        validate_edge_relationship(
            "T-20251228-093045-a1b2c3d4",
            "S-20251228-093046-b2c3d4e5",
            "PART_OF"
        )

    def test_invalid_edge_type_rejected(self):
        with pytest.raises(ValueError, match="Invalid edge type"):
            validate_edge_relationship(
                "T-20251228-093045-a1b2c3d4",
                "T-20251228-093046-b2c3d4e5",
                "INVALID_EDGE_TYPE"
            )

    def test_invalid_relationship_rejected(self):
        # Task cannot CONTAINS sprint (only sprint can CONTAINS task)
        with pytest.raises(ValueError, match="Invalid relationship"):
            validate_edge_relationship(
                "T-20251228-093045-a1b2c3d4",
                "S-20251228-093046-b2c3d4e5",
                "CONTAINS"
            )

    def test_self_reference_rejected(self):
        with pytest.raises(ValueError, match="Self-reference not allowed"):
            validate_edge_relationship(
                "T-20251228-093045-a1b2c3d4",
                "T-20251228-093045-a1b2c3d4",  # Same ID
                "DEPENDS_ON"
            )

    def test_self_reference_allowed_when_explicit(self):
        # Should not raise when allow_self_reference=True
        validate_edge_relationship(
            "T-20251228-093045-a1b2c3d4",
            "T-20251228-093045-a1b2c3d4",
            "DEPENDS_ON",
            allow_self_reference=True
        )

    def test_legacy_sprint_source_rejected(self):
        with pytest.raises(ValueError, match="Cannot use legacy ID"):
            validate_edge_relationship(
                "S-025",
                "T-20251228-093046-b2c3d4e5",
                "CONTAINS"
            )

    def test_legacy_sprint_target_rejected(self):
        with pytest.raises(ValueError, match="Cannot use legacy ID"):
            validate_edge_relationship(
                "T-20251228-093045-a1b2c3d4",
                "S-sprint-017-spark-slm",
                "PART_OF"
            )


class TestEntityIdValidator:
    """Tests for EntityIdValidator class."""

    def test_validate_caches_result(self):
        validator = EntityIdValidator()
        result1 = validator.validate("T-20251228-093045-a1b2c3d4")
        result2 = validator.validate("T-20251228-093045-a1b2c3d4")
        assert result1 == result2 == "task"

    def test_validate_batch_success(self):
        validator = EntityIdValidator()
        results = validator.validate_batch([
            "T-20251228-093045-a1b2c3d4",
            "D-20251228-093046-b2c3d4e5",
            "S-20251228-093047-c3d4e5f6",
        ])
        assert results == {
            "T-20251228-093045-a1b2c3d4": "task",
            "D-20251228-093046-b2c3d4e5": "decision",
            "S-20251228-093047-c3d4e5f6": "sprint",
        }

    def test_validate_batch_with_errors(self):
        validator = EntityIdValidator()
        with pytest.raises(ValueError, match="Batch validation failed"):
            validator.validate_batch([
                "T-20251228-093045-a1b2c3d4",
                "S-025",  # Invalid legacy format
            ])

    def test_clear_cache(self):
        validator = EntityIdValidator()
        validator.validate("T-20251228-093045-a1b2c3d4")
        assert "T-20251228-093045-a1b2c3d4" in validator._cache
        validator.clear_cache()
        assert "T-20251228-093045-a1b2c3d4" not in validator._cache


class TestRelationshipRules:
    """Tests for RelationshipRules class."""

    def test_can_connect_valid(self):
        rules = RelationshipRules()
        assert rules.can_connect("task", "task", "DEPENDS_ON") is True
        assert rules.can_connect("sprint", "task", "CONTAINS") is True
        assert rules.can_connect("task", "sprint", "PART_OF") is True

    def test_can_connect_invalid(self):
        rules = RelationshipRules()
        # Task cannot CONTAINS sprint
        assert rules.can_connect("task", "sprint", "CONTAINS") is False

    def test_get_allowed_targets(self):
        rules = RelationshipRules()
        targets = rules.get_allowed_targets("sprint", "CONTAINS")
        assert "task" in targets

    def test_get_allowed_sources(self):
        rules = RelationshipRules()
        sources = rules.get_allowed_sources("task", "CONTAINS")
        assert "sprint" in sources

    def test_allows_self_reference(self):
        rules = RelationshipRules()
        # By default, no edge types allow self-reference
        assert rules.allows_self_reference("DEPENDS_ON") is False
        assert rules.allows_self_reference("CONTAINS") is False


class TestAllRelationshipRulesValid:
    """Tests that all defined relationship rules are internally consistent."""

    def test_all_rules_have_valid_edge_types(self):
        from cortical.got.types import VALID_EDGE_TYPES
        for edge_type in RELATIONSHIP_RULES.keys():
            assert edge_type in VALID_EDGE_TYPES, f"Unknown edge type in rules: {edge_type}"

    def test_all_rules_reference_valid_entity_types(self):
        from cortical.got.types import VALID_ENTITY_TYPES
        for edge_type, pairs in RELATIONSHIP_RULES.items():
            for source_type, target_type in pairs:
                assert source_type in VALID_ENTITY_TYPES, \
                    f"Unknown source type '{source_type}' in {edge_type}"
                assert target_type in VALID_ENTITY_TYPES, \
                    f"Unknown target type '{target_type}' in {edge_type}"


class TestAllIdPatternsValid:
    """Tests that all ID patterns are valid regex."""

    def test_all_patterns_compile(self):
        for entity_type, pattern in ID_PATTERNS.items():
            assert pattern is not None, f"Pattern for {entity_type} is None"
            # Pattern already compiled in module, just verify it's usable
            assert hasattr(pattern, "match"), f"Pattern for {entity_type} is not compiled"


class TestLegacyPatterns:
    """Tests for legacy pattern detection."""

    def test_legacy_short_sprint_detected(self):
        pattern, msg = LEGACY_PATTERNS["sprint_legacy_short"]
        assert pattern.match("S-1")
        assert pattern.match("S-25")
        assert pattern.match("S-999")
        assert not pattern.match("S-20251228-093045-a1b2c3d4")

    def test_legacy_verbose_sprint_detected(self):
        pattern, msg = LEGACY_PATTERNS["sprint_legacy_verbose"]
        assert pattern.match("S-sprint-1-test")
        assert pattern.match("S-sprint-017-spark-slm")
        assert pattern.match("S-sprint-25")  # Slug is optional
        assert not pattern.match("S-20251228-093045-a1b2c3d4")

    def test_legacy_task_prefix_detected(self):
        pattern, msg = LEGACY_PATTERNS["task_legacy_prefix"]
        assert pattern.match("task:T-something")
        assert not pattern.match("T-20251228-093045-a1b2c3d4")


class TestValidateEntity:
    """Tests for validate_entity() data structure validation."""

    def test_valid_task_entity(self):
        from cortical.got.validation import validate_entity
        data = {
            "id": "T-20251228-093045-a1b2c3d4",
            "entity_type": "task",
            "created_at": "2025-12-28T09:30:45+00:00",
            "title": "Test task",
            "status": "pending",
            "priority": "medium",
        }
        is_valid, error = validate_entity(data)
        assert is_valid, f"Should be valid: {error}"

    def test_entity_not_dict(self):
        from cortical.got.validation import validate_entity
        is_valid, error = validate_entity("not a dict")
        assert not is_valid
        assert "must be a dictionary" in error

    def test_missing_required_field_id(self):
        from cortical.got.validation import validate_entity
        data = {"entity_type": "task", "created_at": "2025-12-28T09:30:45+00:00"}
        is_valid, error = validate_entity(data)
        assert not is_valid
        assert "Missing required field: id" in error

    def test_missing_required_field_entity_type(self):
        from cortical.got.validation import validate_entity
        data = {"id": "T-123", "created_at": "2025-12-28T09:30:45+00:00"}
        is_valid, error = validate_entity(data)
        assert not is_valid
        assert "Missing required field: entity_type" in error

    def test_invalid_entity_type(self):
        from cortical.got.validation import validate_entity
        data = {
            "id": "T-123",
            "entity_type": "invalid_type",
            "created_at": "2025-12-28T09:30:45+00:00",
        }
        is_valid, error = validate_entity(data)
        assert not is_valid
        assert "Invalid entity_type" in error

    def test_empty_id_rejected(self):
        from cortical.got.validation import validate_entity
        data = {"id": "", "entity_type": "task", "created_at": "2025-12-28T09:30:45+00:00"}
        is_valid, error = validate_entity(data)
        assert not is_valid
        assert "non-empty string" in error

    def test_id_not_string(self):
        from cortical.got.validation import validate_entity
        data = {"id": 123, "entity_type": "task", "created_at": "2025-12-28T09:30:45+00:00"}
        is_valid, error = validate_entity(data)
        assert not is_valid
        assert "non-empty string" in error

    def test_invalid_created_at_timestamp(self):
        from cortical.got.validation import validate_entity
        data = {
            "id": "T-123",
            "entity_type": "task",
            "created_at": "not-a-timestamp",
            "title": "Test",
            "status": "pending",
            "priority": "medium",
        }
        is_valid, error = validate_entity(data)
        assert not is_valid
        assert "Invalid created_at timestamp" in error

    def test_invalid_modified_at_timestamp(self):
        from cortical.got.validation import validate_entity
        data = {
            "id": "T-123",
            "entity_type": "task",
            "created_at": "2025-12-28T09:30:45+00:00",
            "modified_at": "invalid-date",
            "title": "Test",
            "status": "pending",
            "priority": "medium",
        }
        is_valid, error = validate_entity(data)
        assert not is_valid
        assert "Invalid modified_at timestamp" in error

    def test_invalid_version_not_positive(self):
        from cortical.got.validation import validate_entity
        data = {
            "id": "T-123",
            "entity_type": "task",
            "created_at": "2025-12-28T09:30:45+00:00",
            "version": 0,  # Must be >= 1
            "title": "Test",
            "status": "pending",
            "priority": "medium",
        }
        is_valid, error = validate_entity(data)
        assert not is_valid
        assert "positive integer" in error


class TestValidateTaskSpecific:
    """Tests for task-specific validation."""

    def test_task_missing_title(self):
        from cortical.got.validation import validate_entity
        data = {
            "id": "T-123",
            "entity_type": "task",
            "created_at": "2025-12-28T09:30:45+00:00",
            "status": "pending",
            "priority": "medium",
        }
        is_valid, error = validate_entity(data)
        assert not is_valid
        assert "Task missing required field: title" in error

    def test_task_invalid_status(self):
        from cortical.got.validation import validate_entity
        data = {
            "id": "T-123",
            "entity_type": "task",
            "created_at": "2025-12-28T09:30:45+00:00",
            "title": "Test",
            "status": "invalid_status",
            "priority": "medium",
        }
        is_valid, error = validate_entity(data)
        assert not is_valid
        assert "Invalid task status" in error

    def test_task_invalid_priority(self):
        from cortical.got.validation import validate_entity
        data = {
            "id": "T-123",
            "entity_type": "task",
            "created_at": "2025-12-28T09:30:45+00:00",
            "title": "Test",
            "status": "pending",
            "priority": "invalid_priority",
        }
        is_valid, error = validate_entity(data)
        assert not is_valid
        assert "Invalid task priority" in error


class TestValidateDecisionSpecific:
    """Tests for decision-specific validation."""

    def test_decision_missing_title(self):
        from cortical.got.validation import validate_entity
        data = {
            "id": "D-123",
            "entity_type": "decision",
            "created_at": "2025-12-28T09:30:45+00:00",
            "rationale": "Because reasons",
        }
        is_valid, error = validate_entity(data)
        assert not is_valid
        assert "Decision missing required field: title" in error

    def test_decision_missing_rationale(self):
        from cortical.got.validation import validate_entity
        data = {
            "id": "D-123",
            "entity_type": "decision",
            "created_at": "2025-12-28T09:30:45+00:00",
            "title": "Test decision",
        }
        is_valid, error = validate_entity(data)
        assert not is_valid
        assert "Decision missing required field: rationale" in error

    def test_valid_decision(self):
        from cortical.got.validation import validate_entity
        data = {
            "id": "D-123",
            "entity_type": "decision",
            "created_at": "2025-12-28T09:30:45+00:00",
            "title": "Test decision",
            "rationale": "Because reasons",
        }
        is_valid, error = validate_entity(data)
        assert is_valid, f"Should be valid: {error}"


class TestValidateEdgeSpecific:
    """Tests for edge-specific validation."""

    def test_edge_missing_source_id(self):
        from cortical.got.validation import validate_entity
        data = {
            "id": "E-123",
            "entity_type": "edge",
            "created_at": "2025-12-28T09:30:45+00:00",
            "target_id": "T-456",
            "edge_type": "DEPENDS_ON",
        }
        is_valid, error = validate_entity(data)
        assert not is_valid
        assert "Edge missing required field: source_id" in error

    def test_edge_invalid_edge_type(self):
        from cortical.got.validation import validate_entity
        data = {
            "id": "E-123",
            "entity_type": "edge",
            "created_at": "2025-12-28T09:30:45+00:00",
            "source_id": "T-123",
            "target_id": "T-456",
            "edge_type": "INVALID_TYPE",
        }
        is_valid, error = validate_entity(data)
        assert not is_valid
        assert "Invalid edge_type" in error

    def test_edge_invalid_weight(self):
        from cortical.got.validation import validate_entity
        data = {
            "id": "E-123",
            "entity_type": "edge",
            "created_at": "2025-12-28T09:30:45+00:00",
            "source_id": "T-123",
            "target_id": "T-456",
            "edge_type": "DEPENDS_ON",
            "weight": -1.0,  # Negative weight invalid
        }
        is_valid, error = validate_entity(data)
        assert not is_valid
        assert "non-negative number" in error

    def test_valid_edge(self):
        from cortical.got.validation import validate_entity
        data = {
            "id": "E-123",
            "entity_type": "edge",
            "created_at": "2025-12-28T09:30:45+00:00",
            "source_id": "T-123",
            "target_id": "T-456",
            "edge_type": "DEPENDS_ON",
            "weight": 1.0,
        }
        is_valid, error = validate_entity(data)
        assert is_valid, f"Should be valid: {error}"


class TestValidateSprintSpecific:
    """Tests for sprint-specific validation."""

    def test_sprint_missing_title(self):
        from cortical.got.validation import validate_entity
        data = {
            "id": "S-123",
            "entity_type": "sprint",
            "created_at": "2025-12-28T09:30:45+00:00",
            "status": "available",
        }
        is_valid, error = validate_entity(data)
        assert not is_valid
        assert "Sprint missing required field: title" in error

    def test_sprint_invalid_status(self):
        from cortical.got.validation import validate_entity
        data = {
            "id": "S-123",
            "entity_type": "sprint",
            "created_at": "2025-12-28T09:30:45+00:00",
            "title": "Sprint 1",
            "status": "invalid_status",
        }
        is_valid, error = validate_entity(data)
        assert not is_valid
        assert "Invalid sprint status" in error

    def test_valid_sprint(self):
        from cortical.got.validation import validate_entity
        data = {
            "id": "S-123",
            "entity_type": "sprint",
            "created_at": "2025-12-28T09:30:45+00:00",
            "title": "Sprint 1",
            "status": "in_progress",
        }
        is_valid, error = validate_entity(data)
        assert is_valid, f"Should be valid: {error}"


class TestValidateEpicSpecific:
    """Tests for epic-specific validation."""

    def test_epic_missing_title(self):
        from cortical.got.validation import validate_entity
        data = {
            "id": "EPIC-123",
            "entity_type": "epic",
            "created_at": "2025-12-28T09:30:45+00:00",
            "status": "active",
        }
        is_valid, error = validate_entity(data)
        assert not is_valid
        assert "Epic missing required field: title" in error

    def test_epic_invalid_status(self):
        from cortical.got.validation import validate_entity
        data = {
            "id": "EPIC-123",
            "entity_type": "epic",
            "created_at": "2025-12-28T09:30:45+00:00",
            "title": "Epic 1",
            "status": "invalid_status",
        }
        is_valid, error = validate_entity(data)
        assert not is_valid
        assert "Invalid epic status" in error

    def test_valid_epic(self):
        from cortical.got.validation import validate_entity
        data = {
            "id": "EPIC-123",
            "entity_type": "epic",
            "created_at": "2025-12-28T09:30:45+00:00",
            "title": "Epic 1",
            "status": "active",
        }
        is_valid, error = validate_entity(data)
        assert is_valid, f"Should be valid: {error}"


class TestValidateHandoffSpecific:
    """Tests for handoff-specific validation."""

    def test_handoff_missing_source_agent(self):
        from cortical.got.validation import validate_entity
        data = {
            "id": "H-123",
            "entity_type": "handoff",
            "created_at": "2025-12-28T09:30:45+00:00",
            "target_agent": "agent-B",
            "task_id": "T-456",
            "status": "initiated",
        }
        is_valid, error = validate_entity(data)
        assert not is_valid
        assert "Handoff missing required field: source_agent" in error

    def test_handoff_invalid_status(self):
        from cortical.got.validation import validate_entity
        data = {
            "id": "H-123",
            "entity_type": "handoff",
            "created_at": "2025-12-28T09:30:45+00:00",
            "source_agent": "agent-A",
            "target_agent": "agent-B",
            "task_id": "T-456",
            "status": "invalid_status",
        }
        is_valid, error = validate_entity(data)
        assert not is_valid
        assert "Invalid handoff status" in error

    def test_valid_handoff(self):
        from cortical.got.validation import validate_entity
        data = {
            "id": "H-123",
            "entity_type": "handoff",
            "created_at": "2025-12-28T09:30:45+00:00",
            "source_agent": "agent-A",
            "target_agent": "agent-B",
            "task_id": "T-456",
            "status": "completed",
        }
        is_valid, error = validate_entity(data)
        assert is_valid, f"Should be valid: {error}"


class TestValidateDocumentSpecific:
    """Tests for document-specific validation."""

    def test_document_missing_path(self):
        from cortical.got.validation import validate_entity
        data = {
            "id": "DOC-123",
            "entity_type": "document",
            "created_at": "2025-12-28T09:30:45+00:00",
            "doc_type": "markdown",
        }
        is_valid, error = validate_entity(data)
        assert not is_valid
        assert "Document missing required field: path" in error

    def test_valid_document(self):
        from cortical.got.validation import validate_entity
        data = {
            "id": "DOC-123",
            "entity_type": "document",
            "created_at": "2025-12-28T09:30:45+00:00",
            "path": "/docs/readme.md",
            "doc_type": "markdown",
        }
        is_valid, error = validate_entity(data)
        assert is_valid, f"Should be valid: {error}"


class TestValidateClaudeMdLayerSpecific:
    """Tests for ClaudeMd layer-specific validation."""

    def test_claudemd_missing_layer_type(self):
        from cortical.got.validation import validate_entity
        data = {
            "id": "CML-123",
            "entity_type": "claudemd_layer",
            "created_at": "2025-12-28T09:30:45+00:00",
            "section_id": "section-1",
            "title": "Test",
            "content": "Content here",
        }
        is_valid, error = validate_entity(data)
        assert not is_valid
        assert "ClaudeMd layer missing required field: layer_type" in error

    def test_valid_claudemd_layer(self):
        from cortical.got.validation import validate_entity
        data = {
            "id": "CML-123",
            "entity_type": "claudemd_layer",
            "created_at": "2025-12-28T09:30:45+00:00",
            "layer_type": "L2",
            "section_id": "section-1",
            "title": "Test",
            "content": "Content here",
        }
        is_valid, error = validate_entity(data)
        assert is_valid, f"Should be valid: {error}"


class TestValidateEntityFile:
    """Tests for validate_entity_file() wrapper validation."""

    def test_file_missing_data_wrapper(self):
        from cortical.got.validation import validate_entity_file
        file_data = {"_checksum": "abc123def456"}
        is_valid, error = validate_entity_file(file_data)
        assert not is_valid
        assert "missing 'data' wrapper" in error

    def test_file_missing_checksum(self):
        from cortical.got.validation import validate_entity_file
        file_data = {
            "data": {
                "id": "T-123",
                "entity_type": "task",
                "created_at": "2025-12-28T09:30:45+00:00",
            }
        }
        is_valid, error = validate_entity_file(file_data)
        assert not is_valid
        assert "missing 'checksum' field" in error

    def test_file_checksum_not_string(self):
        from cortical.got.validation import validate_entity_file
        file_data = {
            "data": {
                "id": "T-123",
                "entity_type": "task",
                "created_at": "2025-12-28T09:30:45+00:00",
            },
            "_checksum": 123,  # Not a string
        }
        is_valid, error = validate_entity_file(file_data)
        assert not is_valid
        assert "non-empty string" in error

    def test_file_with_underscore_checksum(self):
        from cortical.got.validation import validate_entity_file
        file_data = {
            "data": {
                "id": "T-123",
                "entity_type": "task",
                "created_at": "2025-12-28T09:30:45+00:00",
                "title": "Test",
                "status": "pending",
                "priority": "medium",
            },
            "_checksum": "abc123def456abcd",
        }
        is_valid, error = validate_entity_file(file_data)
        assert is_valid, f"Should be valid: {error}"

    def test_file_with_legacy_checksum(self):
        from cortical.got.validation import validate_entity_file
        file_data = {
            "data": {
                "id": "T-123",
                "entity_type": "task",
                "created_at": "2025-12-28T09:30:45+00:00",
                "title": "Test",
                "status": "pending",
                "priority": "medium",
            },
            "checksum": "abc123def456abcd",  # Legacy field name
        }
        is_valid, error = validate_entity_file(file_data)
        assert is_valid, f"Should be valid: {error}"


class TestValidateChecksum:
    """Tests for validate_checksum() format validation."""

    def test_checksum_not_string(self):
        from cortical.got.validation import validate_checksum
        is_valid, error = validate_checksum({}, 12345)
        assert not is_valid
        assert "must be a string" in error

    def test_checksum_empty(self):
        from cortical.got.validation import validate_checksum
        is_valid, error = validate_checksum({}, "")
        assert not is_valid
        assert "cannot be empty" in error

    def test_checksum_not_hex(self):
        from cortical.got.validation import validate_checksum
        is_valid, error = validate_checksum({}, "not-a-hex-string!")
        assert not is_valid
        assert "hexadecimal" in error

    def test_valid_checksum(self):
        from cortical.got.validation import validate_checksum
        is_valid, error = validate_checksum({}, "abc123def456abcd")
        assert is_valid, f"Should be valid: {error}"

    def test_valid_checksum_uppercase(self):
        from cortical.got.validation import validate_checksum
        is_valid, error = validate_checksum({}, "ABC123DEF456ABCD")
        assert is_valid, f"Should be valid: {error}"


class TestIsoDatetimeValidation:
    """Tests for ISO datetime validation helper."""

    def test_timestamp_not_string(self):
        from cortical.got.validation import validate_entity
        data = {
            "id": "T-123",
            "entity_type": "task",
            "created_at": 12345,  # Not a string
            "title": "Test",
            "status": "pending",
            "priority": "medium",
        }
        is_valid, error = validate_entity(data)
        assert not is_valid
        assert "Invalid created_at timestamp" in error

    def test_timestamp_with_z_suffix(self):
        from cortical.got.validation import validate_entity
        data = {
            "id": "T-123",
            "entity_type": "task",
            "created_at": "2025-12-28T09:30:45Z",  # Z suffix
            "title": "Test",
            "status": "pending",
            "priority": "medium",
        }
        is_valid, error = validate_entity(data)
        assert is_valid, f"Should be valid: {error}"


class TestIntegrationWithGoTManager:
    """Integration tests ensuring validation works with GoTManager."""

    def test_add_edge_validates_ids(self, tmp_path):
        """Verify that add_edge in GoTManager validates entity IDs."""
        from cortical.got.api import GoTManager

        got_dir = tmp_path / ".got"
        got_dir.mkdir()
        (got_dir / "entities").mkdir()

        manager = _create_got_manager(got_dir)

        # Valid task IDs should work (with validate_refs=False since entities don't exist)
        # This test just verifies validation doesn't raise for valid IDs
        # Actual edge creation would fail without entities, but validation passes
        try:
            manager.add_edge(
                "T-20251228-093045-a1b2c3d4",
                "T-20251228-093046-b2c3d4e5",
                "DEPENDS_ON",
                validate_refs=False,
                validate_relationship=True
            )
        except ValueError as e:
            # The only acceptable ValueError is about missing entities
            if "not found" in str(e):
                pass  # Expected - entities don't exist
            else:
                raise

    def test_add_edge_rejects_legacy_sprint(self, tmp_path):
        """Verify that add_edge in GoTManager rejects legacy sprint IDs."""
        from cortical.got.api import GoTManager

        got_dir = tmp_path / ".got"
        got_dir.mkdir()
        (got_dir / "entities").mkdir()

        manager = _create_got_manager(got_dir)

        with pytest.raises(ValueError, match="Cannot use legacy ID|Legacy"):
            manager.add_edge(
                "T-20251228-093045-a1b2c3d4",
                "S-025",  # Legacy sprint
                "PART_OF",
                validate_refs=False
            )

    def test_add_edge_rejects_invalid_relationship(self, tmp_path):
        """Verify that add_edge in GoTManager rejects invalid relationships."""
        from cortical.got.api import GoTManager

        got_dir = tmp_path / ".got"
        got_dir.mkdir()
        (got_dir / "entities").mkdir()

        manager = _create_got_manager(got_dir)

        # Task cannot CONTAINS sprint
        with pytest.raises(ValueError, match="Invalid relationship"):
            manager.add_edge(
                "T-20251228-093045-a1b2c3d4",
                "S-20251228-093046-b2c3d4e5",
                "CONTAINS",
                validate_refs=False
            )

    def test_add_edge_rejects_self_reference(self, tmp_path):
        """Verify that add_edge in GoTManager rejects self-references."""
        from cortical.got.api import GoTManager

        got_dir = tmp_path / ".got"
        got_dir.mkdir()
        (got_dir / "entities").mkdir()

        manager = _create_got_manager(got_dir)

        with pytest.raises(ValueError, match="Self-reference"):
            manager.add_edge(
                "T-20251228-093045-a1b2c3d4",
                "T-20251228-093045-a1b2c3d4",  # Same ID
                "DEPENDS_ON",
                validate_refs=False
            )
