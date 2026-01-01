"""
Tests for ID generation utilities.
"""

import re
import pytest

from cortical.utils.id_generation import (
    generate_task_id,
    generate_decision_id,
    generate_edge_id,
    generate_sprint_id,
    generate_epic_id,
    generate_handoff_id,
    generate_goal_id,
    normalize_id,
    generate_plan_id,
    generate_execution_id,
    generate_persona_profile_id,
    generate_team_id,
)


class TestGenerateTaskId:
    """Test generate_task_id function."""

    def test_format(self):
        """Test task ID has correct format."""
        task_id = generate_task_id()

        assert task_id.startswith("T-")
        # Format: T-YYYYMMDD-HHMMSS-XXXXXXXX
        pattern = r"^T-\d{8}-\d{6}-[a-f0-9]{8}$"
        assert re.match(pattern, task_id)

    def test_unique(self):
        """Test that generated IDs are unique."""
        ids = [generate_task_id() for _ in range(100)]
        assert len(set(ids)) == 100


class TestGenerateDecisionId:
    """Test generate_decision_id function."""

    def test_format(self):
        """Test decision ID has correct format."""
        decision_id = generate_decision_id()

        assert decision_id.startswith("D-")
        pattern = r"^D-\d{8}-\d{6}-[a-f0-9]{8}$"
        assert re.match(pattern, decision_id)


class TestGenerateEdgeId:
    """Test generate_edge_id function."""

    def test_format(self):
        """Test edge ID has correct format."""
        edge_id = generate_edge_id()

        assert edge_id.startswith("E-")
        pattern = r"^E-\d{8}-\d{6}-[a-f0-9]{8}$"
        assert re.match(pattern, edge_id)


class TestGenerateSprintId:
    """Test generate_sprint_id function."""

    def test_format(self):
        """Test sprint ID has correct format."""
        sprint_id = generate_sprint_id()

        assert sprint_id.startswith("S-")
        pattern = r"^S-\d{8}-\d{6}-[a-f0-9]{8}$"
        assert re.match(pattern, sprint_id)


class TestGenerateEpicId:
    """Test generate_epic_id function."""

    def test_format_without_name(self):
        """Test epic ID format without name."""
        epic_id = generate_epic_id()

        assert epic_id.startswith("EPIC-")
        pattern = r"^EPIC-\d{8}-\d{6}-[a-f0-9]{8}$"
        assert re.match(pattern, epic_id)

    def test_format_with_name(self):
        """Test epic ID format with name."""
        epic_id = generate_epic_id("Test Epic")

        assert epic_id == "EPIC-test-epic"

    def test_name_normalization(self):
        """Test name is normalized properly."""
        epic_id = generate_epic_id("My_Epic Name")

        assert epic_id == "EPIC-my-epic-name"


class TestGenerateHandoffId:
    """Test generate_handoff_id function."""

    def test_format(self):
        """Test handoff ID has correct format."""
        handoff_id = generate_handoff_id()

        assert handoff_id.startswith("H-")
        pattern = r"^H-\d{8}-\d{6}-[a-f0-9]{8}$"
        assert re.match(pattern, handoff_id)


class TestGenerateGoalId:
    """Test generate_goal_id function."""

    def test_format(self):
        """Test goal ID has correct format."""
        goal_id = generate_goal_id()

        assert goal_id.startswith("G-")
        # Note: Goal uses day-level granularity (no time)
        pattern = r"^G-\d{8}-[a-f0-9]{8}$"
        assert re.match(pattern, goal_id)


class TestGeneratePlanId:
    """Test generate_plan_id function."""

    def test_format(self):
        """Test plan ID has correct format."""
        plan_id = generate_plan_id()

        assert plan_id.startswith("OP-")
        pattern = r"^OP-\d{8}-\d{6}-[a-f0-9]{8}$"
        assert re.match(pattern, plan_id)


class TestGenerateExecutionId:
    """Test generate_execution_id function."""

    def test_format(self):
        """Test execution ID has correct format."""
        exec_id = generate_execution_id()

        assert exec_id.startswith("EX-")
        pattern = r"^EX-\d{8}-\d{6}-[a-f0-9]{8}$"
        assert re.match(pattern, exec_id)


class TestGeneratePersonaProfileId:
    """Test generate_persona_profile_id function."""

    def test_format(self):
        """Test persona profile ID has correct format."""
        pp_id = generate_persona_profile_id()

        assert pp_id.startswith("PP-")
        pattern = r"^PP-\d{8}-\d{6}-[a-f0-9]{8}$"
        assert re.match(pattern, pp_id)


class TestGenerateTeamId:
    """Test generate_team_id function."""

    def test_format(self):
        """Test team ID has correct format."""
        team_id = generate_team_id()

        assert team_id.startswith("TEAM-")
        pattern = r"^TEAM-\d{8}-\d{6}-[a-f0-9]{8}$"
        assert re.match(pattern, team_id)


class TestNormalizeId:
    """Test normalize_id function."""

    def test_task_prefix(self):
        """Test removing task: prefix."""
        result = normalize_id("task:T-20251222-143052-a1b2c3d4")
        assert result == "T-20251222-143052-a1b2c3d4"

    def test_decision_prefix(self):
        """Test removing decision: prefix."""
        result = normalize_id("decision:D-20251222-143052-e5f6g7h8")
        assert result == "D-20251222-143052-e5f6g7h8"

    def test_edge_prefix(self):
        """Test removing edge: prefix."""
        result = normalize_id("edge:E-20251222-143052-i9j0k1l2")
        assert result == "E-20251222-143052-i9j0k1l2"

    def test_sprint_prefix(self):
        """Test removing sprint: prefix."""
        result = normalize_id("sprint:S-20251222-143052-m3n4o5p6")
        assert result == "S-20251222-143052-m3n4o5p6"

    def test_epic_prefix(self):
        """Test removing epic: prefix."""
        result = normalize_id("epic:EPIC-test")
        assert result == "EPIC-test"

    def test_goal_prefix(self):
        """Test removing goal: prefix."""
        result = normalize_id("goal:G-20251222-a1b2c3d4")
        assert result == "G-20251222-a1b2c3d4"

    def test_handoff_prefix(self):
        """Test removing handoff: prefix."""
        result = normalize_id("handoff:H-20251222-143052-u1v2w3x4")
        assert result == "H-20251222-143052-u1v2w3x4"

    def test_no_prefix(self):
        """Test ID without prefix passes through."""
        result = normalize_id("T-20251222-143052-a1b2c3d4")
        assert result == "T-20251222-143052-a1b2c3d4"

    def test_unknown_prefix(self):
        """Test unknown prefix passes through unchanged."""
        result = normalize_id("custom:T-20251222-143052-a1b2c3d4")
        assert result == "custom:T-20251222-143052-a1b2c3d4"
