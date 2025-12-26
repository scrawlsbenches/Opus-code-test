"""
Unit tests for cortical/utils/id_generation.py

Tests all ID generation functions for format, uniqueness, and edge cases.
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
    generate_plan_id,
    generate_execution_id,
    generate_session_id,
    generate_short_id,
    normalize_id,
)


class TestTaskIdGeneration:
    """Tests for generate_task_id()."""

    def test_format(self):
        """Task ID matches expected format."""
        task_id = generate_task_id()
        # Format: T-YYYYMMDD-HHMMSS-XXXXXXXX
        pattern = r"^T-\d{8}-\d{6}-[a-f0-9]{8}$"
        assert re.match(pattern, task_id), f"Invalid format: {task_id}"

    def test_prefix(self):
        """Task ID starts with T-."""
        task_id = generate_task_id()
        assert task_id.startswith("T-")

    def test_uniqueness_deterministic(self, monkeypatch):
        """IDs are unique when secrets returns different values (deterministic mock).

        Uses mocked secrets.token_hex to verify uniqueness without relying on
        probabilistic behavior. This replaces the flaky birthday paradox test.
        """
        import secrets
        counter = [0]

        def mock_token_hex(n):
            counter[0] += 1
            # Return unique values for each call
            return format(counter[0], f'0{n*2}x')

        monkeypatch.setattr(secrets, "token_hex", mock_token_hex)

        # Generate 10 IDs with mocked unique random parts
        ids = {generate_task_id() for _ in range(10)}
        assert len(ids) == 10


class TestDecisionIdGeneration:
    """Tests for generate_decision_id()."""

    def test_format(self):
        """Decision ID matches expected format."""
        decision_id = generate_decision_id()
        pattern = r"^D-\d{8}-\d{6}-[a-f0-9]{8}$"
        assert re.match(pattern, decision_id), f"Invalid format: {decision_id}"

    def test_prefix(self):
        """Decision ID starts with D-."""
        decision_id = generate_decision_id()
        assert decision_id.startswith("D-")

    def test_uniqueness_deterministic(self, monkeypatch):
        """IDs are unique when secrets returns different values (deterministic mock).

        Uses mocked secrets.token_hex to verify uniqueness without relying on
        probabilistic behavior. This replaces the flaky birthday paradox test.
        """
        import secrets
        counter = [0]

        def mock_token_hex(n):
            counter[0] += 1
            # Return unique values for each call
            return format(counter[0], f'0{n*2}x')

        monkeypatch.setattr(secrets, "token_hex", mock_token_hex)

        # Generate 10 IDs with mocked unique random parts
        ids = {generate_decision_id() for _ in range(10)}
        assert len(ids) == 10


class TestEdgeIdGeneration:
    """Tests for generate_edge_id()."""

    def test_format(self):
        """Edge ID matches expected format."""
        edge_id = generate_edge_id()
        pattern = r"^E-\d{8}-\d{6}-[a-f0-9]{8}$"
        assert re.match(pattern, edge_id), f"Invalid format: {edge_id}"

    def test_prefix(self):
        """Edge ID starts with E-."""
        edge_id = generate_edge_id()
        assert edge_id.startswith("E-")


class TestSprintIdGeneration:
    """Tests for generate_sprint_id()."""

    def test_with_number(self):
        """Sprint ID with number uses S-NNN format."""
        sprint_id = generate_sprint_id(number=5)
        assert sprint_id == "S-005"

    def test_with_large_number(self):
        """Sprint ID handles large numbers."""
        sprint_id = generate_sprint_id(number=123)
        assert sprint_id == "S-123"

    def test_without_number(self):
        """Sprint ID without number uses year-month format."""
        sprint_id = generate_sprint_id()
        pattern = r"^S-\d{4}-\d{2}$"
        assert re.match(pattern, sprint_id), f"Invalid format: {sprint_id}"

    def test_number_zero(self):
        """Sprint ID with number 0."""
        sprint_id = generate_sprint_id(number=0)
        assert sprint_id == "S-000"


class TestEpicIdGeneration:
    """Tests for generate_epic_id()."""

    def test_format_without_name(self):
        """Epic ID matches expected format when no name provided."""
        epic_id = generate_epic_id()
        pattern = r"^EPIC-\d{8}-\d{6}-[a-f0-9]{8}$"
        assert re.match(pattern, epic_id), f"Invalid format: {epic_id}"

    def test_prefix(self):
        """Epic ID starts with EPIC-."""
        epic_id = generate_epic_id()
        assert epic_id.startswith("EPIC-")

    def test_with_name(self):
        """Epic ID uses name when provided."""
        epic_id = generate_epic_id("test-feature")
        assert epic_id == "EPIC-test-feature"


class TestHandoffIdGeneration:
    """Tests for generate_handoff_id()."""

    def test_format(self):
        """Handoff ID matches expected format."""
        handoff_id = generate_handoff_id()
        pattern = r"^H-\d{8}-\d{6}-[a-f0-9]{8}$"
        assert re.match(pattern, handoff_id), f"Invalid format: {handoff_id}"

    def test_prefix(self):
        """Handoff ID starts with H-."""
        handoff_id = generate_handoff_id()
        assert handoff_id.startswith("H-")


class TestGoalIdGeneration:
    """Tests for generate_goal_id()."""

    def test_format(self):
        """Goal ID matches expected format (no time, just date)."""
        goal_id = generate_goal_id()
        # Format: G-YYYYMMDD-XXXXXXXX (no time component)
        pattern = r"^G-\d{8}-[a-f0-9]{8}$"
        assert re.match(pattern, goal_id), f"Invalid format: {goal_id}"

    def test_prefix(self):
        """Goal ID starts with G-."""
        goal_id = generate_goal_id()
        assert goal_id.startswith("G-")


class TestPlanIdGeneration:
    """Tests for generate_plan_id()."""

    def test_format(self):
        """Plan ID matches expected format."""
        plan_id = generate_plan_id()
        pattern = r"^OP-\d{8}-\d{6}-[a-f0-9]{8}$"
        assert re.match(pattern, plan_id), f"Invalid format: {plan_id}"

    def test_prefix(self):
        """Plan ID starts with OP-."""
        plan_id = generate_plan_id()
        assert plan_id.startswith("OP-")


class TestExecutionIdGeneration:
    """Tests for generate_execution_id()."""

    def test_format(self):
        """Execution ID matches expected format."""
        exec_id = generate_execution_id()
        pattern = r"^EX-\d{8}-\d{6}-[a-f0-9]{8}$"
        assert re.match(pattern, exec_id), f"Invalid format: {exec_id}"

    def test_prefix(self):
        """Execution ID starts with EX-."""
        exec_id = generate_execution_id()
        assert exec_id.startswith("EX-")


class TestSessionIdGeneration:
    """Tests for generate_session_id()."""

    def test_format(self):
        """Session ID is 4 hex characters."""
        session_id = generate_session_id()
        pattern = r"^[a-f0-9]{4}$"
        assert re.match(pattern, session_id), f"Invalid format: {session_id}"

    def test_length(self):
        """Session ID is exactly 4 characters."""
        session_id = generate_session_id()
        assert len(session_id) == 4


class TestShortIdGeneration:
    """Tests for generate_short_id()."""

    def test_without_prefix(self):
        """Short ID without prefix is 8 hex characters."""
        short_id = generate_short_id()
        pattern = r"^[a-f0-9]{8}$"
        assert re.match(pattern, short_id), f"Invalid format: {short_id}"

    def test_with_prefix(self):
        """Short ID with prefix includes prefix."""
        short_id = generate_short_id(prefix="T")
        pattern = r"^T-[a-f0-9]{8}$"
        assert re.match(pattern, short_id), f"Invalid format: {short_id}"

    def test_custom_prefix(self):
        """Short ID with custom prefix."""
        short_id = generate_short_id(prefix="CUSTOM")
        assert short_id.startswith("CUSTOM-")


class TestNormalizeId:
    """Tests for normalize_id()."""

    def test_task_prefix(self):
        """Removes 'task:' prefix."""
        normalized = normalize_id("task:T-20251222-143052-a1b2c3d4")
        assert normalized == "T-20251222-143052-a1b2c3d4"

    def test_decision_prefix(self):
        """Removes 'decision:' prefix."""
        normalized = normalize_id("decision:D-20251222-143052-e5f6g7h8")
        assert normalized == "D-20251222-143052-e5f6g7h8"

    def test_edge_prefix(self):
        """Removes 'edge:' prefix."""
        normalized = normalize_id("edge:E-123")
        assert normalized == "E-123"

    def test_sprint_prefix(self):
        """Removes 'sprint:' prefix."""
        normalized = normalize_id("sprint:S-005")
        assert normalized == "S-005"

    def test_epic_prefix(self):
        """Removes 'epic:' prefix."""
        normalized = normalize_id("epic:E-123")
        assert normalized == "E-123"

    def test_goal_prefix(self):
        """Removes 'goal:' prefix."""
        normalized = normalize_id("goal:G-123")
        assert normalized == "G-123"

    def test_handoff_prefix(self):
        """Removes 'handoff:' prefix."""
        normalized = normalize_id("handoff:H-123")
        assert normalized == "H-123"

    def test_no_prefix(self):
        """ID without prefix returns unchanged."""
        normalized = normalize_id("T-20251222-143052-a1b2c3d4")
        assert normalized == "T-20251222-143052-a1b2c3d4"

    def test_unknown_prefix(self):
        """Unknown prefix returns unchanged."""
        normalized = normalize_id("unknown:T-123")
        assert normalized == "unknown:T-123"
