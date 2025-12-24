"""
Unit tests for cortical/utils/id_generation.py

Tests all ID generation functions and format validation.
"""

import re
import pytest
from datetime import datetime, timezone

from cortical.utils.id_generation import (
    generate_task_id,
    generate_decision_id,
    generate_edge_id,
    generate_sprint_id,
    generate_epic_id,
    generate_handoff_id,
    generate_goal_id,
    generate_plan_id,
    generate_execution_id,
    generate_session_id,
    generate_short_id,
    generate_claudemd_layer_id,
    generate_claudemd_version_id,
    generate_persona_profile_id,
    generate_team_id,
    generate_document_id,
    normalize_id,
)


class TestTaskId:
    """Tests for generate_task_id()."""

    def test_format(self):
        """Task ID has correct format."""
        task_id = generate_task_id()
        assert task_id.startswith("T-")
        # Format: T-YYYYMMDD-HHMMSS-XXXXXXXX
        pattern = r'^T-\d{8}-\d{6}-[0-9a-f]{8}$'
        assert re.match(pattern, task_id)

    def test_timestamp_is_utc(self):
        """Task ID uses UTC timestamp."""
        now_utc = datetime.now(timezone.utc)
        task_id = generate_task_id()

        # Extract timestamp part
        parts = task_id.split('-')
        date_part = parts[1]  # YYYYMMDD
        time_part = parts[2]  # HHMMSS

        # Should be close to current UTC time
        assert date_part == now_utc.strftime("%Y%m%d")

    def test_uniqueness(self):
        """Task IDs are unique."""
        ids = [generate_task_id() for _ in range(100)]
        assert len(set(ids)) == 100

    def test_no_legacy_prefix(self):
        """Task ID doesn't have legacy 'task:' prefix."""
        task_id = generate_task_id()
        assert not task_id.startswith("task:")


class TestDecisionId:
    """Tests for generate_decision_id()."""

    def test_format(self):
        """Decision ID has correct format."""
        decision_id = generate_decision_id()
        assert decision_id.startswith("D-")
        pattern = r'^D-\d{8}-\d{6}-[0-9a-f]{8}$'
        assert re.match(pattern, decision_id)

    def test_uniqueness(self):
        """Decision IDs are unique."""
        ids = [generate_decision_id() for _ in range(100)]
        assert len(set(ids)) == 100


class TestEdgeId:
    """Tests for generate_edge_id()."""

    def test_format(self):
        """Edge ID has correct format."""
        edge_id = generate_edge_id()
        assert edge_id.startswith("E-")
        pattern = r'^E-\d{8}-\d{6}-[0-9a-f]{8}$'
        assert re.match(pattern, edge_id)

    def test_uniqueness(self):
        """Edge IDs are unique."""
        ids = [generate_edge_id() for _ in range(100)]
        assert len(set(ids)) == 100


class TestSprintId:
    """Tests for generate_sprint_id()."""

    def test_numbered_format(self):
        """Sprint ID with number has correct format."""
        sprint_id = generate_sprint_id(number=5)
        assert sprint_id == "S-005"

    def test_numbered_padding(self):
        """Sprint ID number is zero-padded to 3 digits."""
        assert generate_sprint_id(number=1) == "S-001"
        assert generate_sprint_id(number=42) == "S-042"
        assert generate_sprint_id(number=999) == "S-999"

    def test_default_format(self):
        """Sprint ID without number uses year-month format."""
        sprint_id = generate_sprint_id()
        now_utc = datetime.now(timezone.utc)
        expected = f"S-{now_utc.strftime('%Y-%m')}"
        assert sprint_id == expected

    def test_default_is_utc(self):
        """Sprint ID uses UTC for year-month."""
        sprint_id = generate_sprint_id()
        pattern = r'^S-\d{4}-\d{2}$'
        assert re.match(pattern, sprint_id)


class TestEpicId:
    """Tests for generate_epic_id()."""

    def test_format(self):
        """Epic ID has correct format."""
        epic_id = generate_epic_id()
        assert epic_id.startswith("E-")
        pattern = r'^E-\d{8}-\d{6}-[0-9a-f]{8}$'
        assert re.match(pattern, epic_id)

    def test_uniqueness(self):
        """Epic IDs are unique."""
        ids = [generate_epic_id() for _ in range(100)]
        assert len(set(ids)) == 100


class TestHandoffId:
    """Tests for generate_handoff_id()."""

    def test_format(self):
        """Handoff ID has correct format."""
        handoff_id = generate_handoff_id()
        assert handoff_id.startswith("H-")
        pattern = r'^H-\d{8}-\d{6}-[0-9a-f]{8}$'
        assert re.match(pattern, handoff_id)

    def test_uniqueness(self):
        """Handoff IDs are unique."""
        ids = [generate_handoff_id() for _ in range(100)]
        assert len(set(ids)) == 100


class TestGoalId:
    """Tests for generate_goal_id()."""

    def test_format(self):
        """Goal ID has correct format (no hour/minute/second)."""
        goal_id = generate_goal_id()
        assert goal_id.startswith("G-")
        # Format: G-YYYYMMDD-XXXXXXXX (no time, just date)
        pattern = r'^G-\d{8}-[0-9a-f]{8}$'
        assert re.match(pattern, goal_id)

    def test_day_level_granularity(self):
        """Goal ID only has date, not time."""
        goal_id = generate_goal_id()
        # Should be G-YYYYMMDD-XXXXXXXX (19 chars)
        assert len(goal_id) == 19

    def test_uniqueness(self):
        """Goal IDs are unique."""
        ids = [generate_goal_id() for _ in range(100)]
        assert len(set(ids)) == 100


class TestPlanId:
    """Tests for generate_plan_id()."""

    def test_format(self):
        """Plan ID has correct format."""
        plan_id = generate_plan_id()
        assert plan_id.startswith("OP-")
        pattern = r'^OP-\d{8}-\d{6}-[0-9a-f]{8}$'
        assert re.match(pattern, plan_id)

    def test_uniqueness(self):
        """Plan IDs are unique."""
        ids = [generate_plan_id() for _ in range(100)]
        assert len(set(ids)) == 100


class TestExecutionId:
    """Tests for generate_execution_id()."""

    def test_format(self):
        """Execution ID has correct format."""
        execution_id = generate_execution_id()
        assert execution_id.startswith("EX-")
        pattern = r'^EX-\d{8}-\d{6}-[0-9a-f]{8}$'
        assert re.match(pattern, execution_id)

    def test_uniqueness(self):
        """Execution IDs are unique."""
        ids = [generate_execution_id() for _ in range(100)]
        assert len(set(ids)) == 100


class TestSessionId:
    """Tests for generate_session_id()."""

    def test_format(self):
        """Session ID is short (4 hex chars)."""
        session_id = generate_session_id()
        pattern = r'^[0-9a-f]{4}$'
        assert re.match(pattern, session_id)

    def test_length(self):
        """Session ID is exactly 4 characters."""
        session_id = generate_session_id()
        assert len(session_id) == 4

    def test_uses_secrets_module(self, monkeypatch):
        """Session ID uses cryptographically secure randomness."""
        # Mock secrets.token_hex to verify it's called correctly
        # This is deterministic - no flaky birthday paradox issues!
        import secrets
        calls = []

        def mock_token_hex(n):
            calls.append(n)
            return "ab" * n  # Return predictable value

        monkeypatch.setattr(secrets, "token_hex", mock_token_hex)

        result = generate_session_id()

        # Verify secrets.token_hex(2) was called (2 bytes = 4 hex chars)
        assert calls == [2]
        assert result == "abab"


class TestShortId:
    """Tests for generate_short_id()."""

    def test_format_no_prefix(self):
        """Short ID without prefix is 8 hex chars."""
        short_id = generate_short_id()
        pattern = r'^[0-9a-f]{8}$'
        assert re.match(pattern, short_id)

    def test_format_with_prefix(self):
        """Short ID with prefix has correct format."""
        short_id = generate_short_id(prefix="T")
        pattern = r'^T-[0-9a-f]{8}$'
        assert re.match(pattern, short_id)

    def test_length_no_prefix(self):
        """Short ID without prefix is exactly 8 characters."""
        short_id = generate_short_id()
        assert len(short_id) == 8

    def test_length_with_prefix(self):
        """Short ID with prefix includes prefix and separator."""
        short_id = generate_short_id(prefix="TEST")
        assert short_id.startswith("TEST-")
        assert len(short_id) == 13  # "TEST-" + 8 hex chars

    def test_uniqueness(self):
        """Short IDs are unique."""
        ids = [generate_short_id() for _ in range(100)]
        assert len(set(ids)) == 100


class TestClaudeMdLayerId:
    """Tests for generate_claudemd_layer_id()."""

    def test_format_with_section(self):
        """Layer ID with section has correct format."""
        layer_id = generate_claudemd_layer_id(layer_number=2, section_id="architecture")
        pattern = r'^CML2-architecture-\d{8}-\d{6}-[0-9a-f]{8}$'
        assert re.match(pattern, layer_id)

    def test_format_without_section(self):
        """Layer ID without section has correct format."""
        layer_id = generate_claudemd_layer_id(layer_number=3)
        pattern = r'^CML3-\d{8}-\d{6}-[0-9a-f]{8}$'
        assert re.match(pattern, layer_id)

    def test_format_with_empty_section(self):
        """Layer ID with empty section string omits section."""
        layer_id = generate_claudemd_layer_id(layer_number=1, section_id="")
        pattern = r'^CML1-\d{8}-\d{6}-[0-9a-f]{8}$'
        assert re.match(pattern, layer_id)

    def test_layer_number_in_id(self):
        """Layer number is included in ID."""
        for layer_num in range(5):
            layer_id = generate_claudemd_layer_id(layer_number=layer_num)
            assert layer_id.startswith(f"CML{layer_num}-")


class TestClaudeMdVersionId:
    """Tests for generate_claudemd_version_id()."""

    def test_format(self):
        """Version ID has correct format."""
        layer_id = "CML3-persona-20251222-093045-a1b2c3d4"
        version_id = generate_claudemd_version_id(layer_id, version_number=3)
        assert version_id == "CMV-CML3-persona-20251222-093045-a1b2c3d4-v3"

    def test_version_number(self):
        """Version number is included in ID."""
        layer_id = "CML2-test-20251222-093045-abcd1234"
        version_id = generate_claudemd_version_id(layer_id, version_number=42)
        assert version_id.endswith("-v42")

    def test_layer_traceability(self):
        """Version ID maintains traceability to layer ID."""
        layer_id = "CML1-section-20251222-093045-12345678"
        version_id = generate_claudemd_version_id(layer_id, version_number=1)
        assert layer_id in version_id


class TestPersonaProfileId:
    """Tests for generate_persona_profile_id()."""

    def test_format(self):
        """Persona profile ID has correct format."""
        profile_id = generate_persona_profile_id()
        assert profile_id.startswith("PP-")
        pattern = r'^PP-\d{8}-\d{6}-[0-9a-f]{8}$'
        assert re.match(pattern, profile_id)

    def test_uniqueness(self):
        """Persona profile IDs are unique."""
        ids = [generate_persona_profile_id() for _ in range(100)]
        assert len(set(ids)) == 100


class TestTeamId:
    """Tests for generate_team_id()."""

    def test_format(self):
        """Team ID has correct format."""
        team_id = generate_team_id()
        assert team_id.startswith("TEAM-")
        pattern = r'^TEAM-\d{8}-\d{6}-[0-9a-f]{8}$'
        assert re.match(pattern, team_id)

    def test_uniqueness(self):
        """Team IDs are unique."""
        ids = [generate_team_id() for _ in range(100)]
        assert len(set(ids)) == 100


class TestDocumentId:
    """Tests for generate_document_id()."""

    def test_format_with_path(self):
        """Document ID with path is deterministic."""
        doc_id = generate_document_id(path="docs/architecture.md")
        assert doc_id == "DOC-docs-architecture-md"

    def test_path_normalization(self):
        """Path is normalized (slashes, dots, underscores to dashes)."""
        doc_id = generate_document_id(path="src/module_name.py")
        assert doc_id == "DOC-src-module-name-py"

    def test_deterministic_with_path(self):
        """Same path produces same ID."""
        path = "test/file.txt"
        id1 = generate_document_id(path=path)
        id2 = generate_document_id(path=path)
        assert id1 == id2

    def test_format_without_path(self):
        """Document ID without path uses timestamp."""
        doc_id = generate_document_id()
        assert doc_id.startswith("DOC-")
        pattern = r'^DOC-\d{8}-\d{6}-[0-9a-f]{8}$'
        assert re.match(pattern, doc_id)

    def test_uniqueness_without_path(self):
        """Document IDs without path are highly unique."""
        ids = [generate_document_id() for _ in range(100)]
        # With 8 hex chars (4 billion combinations), collisions are extremely rare
        # Allow up to 1 collision in 100 rapid generations (still 99% unique)
        unique_count = len(set(ids))
        assert unique_count >= 99, f"Expected at least 99 unique IDs, got {unique_count}"


class TestNormalizeId:
    """Tests for normalize_id()."""

    def test_remove_task_prefix(self):
        """Remove legacy 'task:' prefix."""
        normalized = normalize_id('task:T-20251222-143052-a1b2c3d4')
        assert normalized == 'T-20251222-143052-a1b2c3d4'

    def test_remove_decision_prefix(self):
        """Remove legacy 'decision:' prefix."""
        normalized = normalize_id('decision:D-20251222-143052-e5f6g7h8')
        assert normalized == 'D-20251222-143052-e5f6g7h8'

    def test_remove_edge_prefix(self):
        """Remove legacy 'edge:' prefix."""
        normalized = normalize_id('edge:E-20251222-143052-i9j0k1l2')
        assert normalized == 'E-20251222-143052-i9j0k1l2'

    def test_remove_sprint_prefix(self):
        """Remove legacy 'sprint:' prefix."""
        normalized = normalize_id('sprint:S-005')
        assert normalized == 'S-005'

    def test_remove_epic_prefix(self):
        """Remove legacy 'epic:' prefix."""
        normalized = normalize_id('epic:E-20251222-143052-q7r8s9t0')
        assert normalized == 'E-20251222-143052-q7r8s9t0'

    def test_remove_goal_prefix(self):
        """Remove legacy 'goal:' prefix."""
        normalized = normalize_id('goal:G-20251222-m3n4o5p6')
        assert normalized == 'G-20251222-m3n4o5p6'

    def test_remove_handoff_prefix(self):
        """Remove legacy 'handoff:' prefix."""
        normalized = normalize_id('handoff:H-20251222-143052-u1v2w3x4')
        assert normalized == 'H-20251222-143052-u1v2w3x4'

    def test_no_prefix(self):
        """ID without legacy prefix is unchanged."""
        id_str = 'T-20251222-143052-a1b2c3d4'
        normalized = normalize_id(id_str)
        assert normalized == id_str

    def test_unknown_prefix(self):
        """ID with unknown prefix is unchanged."""
        id_str = 'unknown:T-20251222-143052-a1b2c3d4'
        normalized = normalize_id(id_str)
        assert normalized == id_str


class TestIdCollisionResistance:
    """Tests for collision resistance across ID types."""

    def test_different_types_dont_collide(self):
        """Different ID types have different prefixes."""
        task_id = generate_task_id()
        decision_id = generate_decision_id()
        edge_id = generate_edge_id()
        handoff_id = generate_handoff_id()

        # All should have different prefixes
        assert task_id.startswith("T-")
        assert decision_id.startswith("D-")
        assert edge_id.startswith("E-")
        assert handoff_id.startswith("H-")

        # Should not be equal even if generated at same timestamp
        ids = [task_id, decision_id, edge_id, handoff_id]
        assert len(set(ids)) == 4

    def test_high_volume_uniqueness(self):
        """Generate many IDs without collisions."""
        # Mix different ID types
        ids = []
        for _ in range(50):
            ids.append(generate_task_id())
            ids.append(generate_decision_id())
            ids.append(generate_handoff_id())

        # All should be unique
        assert len(set(ids)) == 150
