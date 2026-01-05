"""
Integration tests for GoT CLI sprint and epic commands.

Tests the CLI commands in scripts/got_utils.py using subprocess.
Each test uses an isolated temporary .got directory to avoid conflicts.
"""

import subprocess
import pytest
import json
import os
from pathlib import Path


class TestGoTCLI:
    """Integration tests for GoT CLI commands."""

    @pytest.fixture
    def got_env(self, tmp_path):
        """
        Set up a temp .got directory for CLI tests.

        Returns:
            tuple: (env dict, cwd path)
        """
        got_dir = tmp_path / ".got"
        got_dir.mkdir()

        # Create required subdirectories for TX backend
        (got_dir / "entities").mkdir()
        (got_dir / "wal").mkdir()
        (got_dir / "snapshots").mkdir()

        env = os.environ.copy()
        env["GOT_DIR"] = str(got_dir)
        # Force TX backend (faster, more reliable)
        env["GOT_USE_LEGACY"] = "0"

        # Use project root as cwd so imports work
        project_root = Path(__file__).parent.parent.parent

        return env, project_root

    def run_cli(self, args, env, cwd):
        """
        Run got_utils.py with args.

        Args:
            args: List of command arguments
            env: Environment dict
            cwd: Working directory

        Returns:
            subprocess.CompletedProcess result
        """
        result = subprocess.run(
            ["python", "scripts/got_utils.py"] + args,
            capture_output=True,
            text=True,
            env=env,
            cwd=cwd,
        )
        return result


class TestSprintCommands(TestGoTCLI):
    """Tests for sprint CLI commands."""

    def test_sprint_list_empty(self, got_env):
        """Test listing sprints when none exist."""
        env, cwd = got_env
        result = self.run_cli(["sprint", "list"], env, cwd)

        assert result.returncode == 0
        assert "No sprints found" in result.stdout

    def test_sprint_create_basic(self, got_env):
        """Test creating a sprint with just a name."""
        env, cwd = got_env
        result = self.run_cli(["sprint", "create", "Test Sprint"], env, cwd)

        assert result.returncode == 0
        assert "Created:" in result.stdout
        assert "S-" in result.stdout  # Sprint ID should be in output

    def test_sprint_create_with_number(self, got_env):
        """Test creating a sprint with a number."""
        env, cwd = got_env
        result = self.run_cli(
            ["sprint", "create", "Sprint 1", "--number", "1"],
            env, cwd
        )

        assert result.returncode == 0
        assert "Created:" in result.stdout

        # Verify it appears in list
        result = self.run_cli(["sprint", "list"], env, cwd)
        assert "Sprint 1" in result.stdout

    def test_sprint_create_with_epic(self, got_env):
        """Test creating a sprint associated with an epic."""
        env, cwd = got_env

        # First create an epic
        epic_result = self.run_cli(
            ["epic", "create", "Test Epic"],
            env, cwd
        )
        assert epic_result.returncode == 0

        # Extract epic ID from output (format: "Created: epic:E-...")
        epic_id = epic_result.stdout.strip().replace("Created: ", "").strip()

        # Create sprint with epic
        result = self.run_cli(
            ["sprint", "create", "Test Sprint", "--epic", epic_id],
            env, cwd
        )

        assert result.returncode == 0
        assert "Created:" in result.stdout

    def test_sprint_start(self, got_env):
        """Test starting a sprint."""
        env, cwd = got_env

        # Create a sprint first
        create_result = self.run_cli(
            ["sprint", "create", "Test Sprint"],
            env, cwd
        )
        assert create_result.returncode == 0

        # Extract sprint ID
        sprint_id = create_result.stdout.strip().replace("Created: ", "").strip()

        # Start the sprint
        result = self.run_cli(["sprint", "start", sprint_id], env, cwd)

        assert result.returncode == 0
        assert "Started:" in result.stdout
        assert sprint_id in result.stdout

    def test_sprint_complete(self, got_env):
        """Test completing a sprint."""
        env, cwd = got_env

        # Create and start a sprint
        create_result = self.run_cli(
            ["sprint", "create", "Test Sprint"],
            env, cwd
        )
        sprint_id = create_result.stdout.strip().replace("Created: ", "").strip()

        self.run_cli(["sprint", "start", sprint_id], env, cwd)

        # Complete the sprint
        result = self.run_cli(["sprint", "complete", sprint_id], env, cwd)

        assert result.returncode == 0
        assert "Completed:" in result.stdout
        assert sprint_id in result.stdout

    def test_sprint_status_no_sprints(self, got_env):
        """Test sprint status when no sprints exist."""
        env, cwd = got_env
        result = self.run_cli(["sprint", "status"], env, cwd)

        assert result.returncode == 0
        # Should handle empty gracefully (no crash)

    def test_sprint_status_with_active_sprint(self, got_env):
        """Test sprint status showing an active sprint."""
        env, cwd = got_env

        # Create and start a sprint
        create_result = self.run_cli(
            ["sprint", "create", "Active Sprint"],
            env, cwd
        )
        sprint_id = create_result.stdout.strip().replace("Created: ", "").strip()

        self.run_cli(["sprint", "start", sprint_id], env, cwd)

        # Check status
        result = self.run_cli(["sprint", "status"], env, cwd)

        assert result.returncode == 0
        assert "Active Sprint" in result.stdout or sprint_id in result.stdout

    def test_sprint_list_with_status_filter(self, got_env):
        """Test listing sprints filtered by status."""
        env, cwd = got_env

        # Create two sprints with explicit numbers to avoid ID collision
        create1 = self.run_cli(["sprint", "create", "Sprint 1", "--number", "1"], env, cwd)
        sprint_id1 = create1.stdout.strip().replace("Created: ", "").strip()

        create2 = self.run_cli(["sprint", "create", "Sprint 2", "--number", "2"], env, cwd)

        self.run_cli(["sprint", "start", sprint_id1], env, cwd)

        # List in_progress sprints
        result = self.run_cli(
            ["sprint", "list", "--status", "in_progress"],
            env, cwd
        )

        assert result.returncode == 0
        assert "Sprint 1" in result.stdout


class TestEpicCommands(TestGoTCLI):
    """Tests for epic CLI commands."""

    def test_epic_list_empty(self, got_env):
        """Test listing epics when none exist."""
        env, cwd = got_env
        result = self.run_cli(["epic", "list"], env, cwd)

        assert result.returncode == 0
        assert "No epics found" in result.stdout

    def test_epic_create_basic(self, got_env):
        """Test creating an epic with just a name."""
        env, cwd = got_env
        result = self.run_cli(["epic", "create", "Test Epic"], env, cwd)

        assert result.returncode == 0
        assert "Created:" in result.stdout
        assert "EPIC-" in result.stdout  # Epic ID should be in output

    def test_epic_create_with_custom_id(self, got_env):
        """Test creating an epic with a custom ID."""
        env, cwd = got_env
        result = self.run_cli(
            ["epic", "create", "Custom Epic", "--id", "EPIC-custom-123"],
            env, cwd
        )

        assert result.returncode == 0
        assert "Created:" in result.stdout

        # Verify it appears in list
        result = self.run_cli(["epic", "list"], env, cwd)
        assert "EPIC-custom-123" in result.stdout or "Custom Epic" in result.stdout

    def test_epic_show(self, got_env):
        """Test showing epic details."""
        env, cwd = got_env

        # Create an epic
        create_result = self.run_cli(
            ["epic", "create", "Test Epic"],
            env, cwd
        )
        assert create_result.returncode == 0

        # Extract epic ID
        epic_id = create_result.stdout.strip().replace("Created: ", "").strip()

        # Show epic details
        result = self.run_cli(["epic", "show", epic_id], env, cwd)

        assert result.returncode == 0
        assert "Epic:" in result.stdout
        assert epic_id in result.stdout
        assert "Test Epic" in result.stdout

    def test_epic_show_not_found(self, got_env):
        """Test showing a non-existent epic."""
        env, cwd = got_env
        result = self.run_cli(["epic", "show", "E-nonexistent"], env, cwd)

        assert result.returncode == 1  # Should fail
        assert "not found" in result.stdout

    def test_epic_list_multiple(self, got_env):
        """Test listing multiple epics."""
        env, cwd = got_env

        # Create multiple epics
        self.run_cli(["epic", "create", "Epic 1"], env, cwd)
        self.run_cli(["epic", "create", "Epic 2"], env, cwd)
        self.run_cli(["epic", "create", "Epic 3"], env, cwd)

        # List all
        result = self.run_cli(["epic", "list"], env, cwd)

        assert result.returncode == 0
        assert "Epic 1" in result.stdout
        assert "Epic 2" in result.stdout
        assert "Epic 3" in result.stdout

    def test_epic_with_associated_sprints(self, got_env):
        """Test showing an epic with associated sprints."""
        env, cwd = got_env

        # Create an epic
        epic_result = self.run_cli(["epic", "create", "Main Epic"], env, cwd)
        epic_id = epic_result.stdout.strip().replace("Created: ", "").strip()

        # Create sprints associated with the epic
        self.run_cli(
            ["sprint", "create", "Sprint 1", "--epic", epic_id],
            env, cwd
        )
        self.run_cli(
            ["sprint", "create", "Sprint 2", "--epic", epic_id],
            env, cwd
        )

        # Show epic (should list sprints)
        result = self.run_cli(["epic", "show", epic_id], env, cwd)

        assert result.returncode == 0
        assert "Sprint 1" in result.stdout or "Sprints" in result.stdout


class TestSprintEpicIntegration(TestGoTCLI):
    """Tests for sprint-epic integration."""

    def test_create_epic_with_multiple_sprints(self, got_env):
        """Test creating an epic and associating multiple sprints."""
        env, cwd = got_env

        # Create epic
        epic_result = self.run_cli(
            ["epic", "create", "Q1 2025 Goals"],
            env, cwd
        )
        epic_id = epic_result.stdout.strip().replace("Created: ", "").strip()

        # Create sprints for the epic
        sprint_names = ["Sprint 1", "Sprint 2", "Sprint 3"]
        for name in sprint_names:
            result = self.run_cli(
                ["sprint", "create", name, "--epic", epic_id],
                env, cwd
            )
            assert result.returncode == 0

        # Verify epic shows all sprints
        show_result = self.run_cli(["epic", "show", epic_id], env, cwd)
        assert show_result.returncode == 0

        # Check that at least some sprint information is shown
        # (either count or names)
        stdout = show_result.stdout
        assert "Sprint" in stdout or "3" in stdout

    def test_sprint_lifecycle_in_epic(self, got_env):
        """Test complete sprint lifecycle within an epic."""
        env, cwd = got_env

        # Create epic
        epic_result = self.run_cli(["epic", "create", "Feature Epic"], env, cwd)
        epic_id = epic_result.stdout.strip().replace("Created: ", "").strip()

        # Create sprint
        sprint_result = self.run_cli(
            ["sprint", "create", "Implementation Sprint", "--epic", epic_id],
            env, cwd
        )
        sprint_id = sprint_result.stdout.strip().replace("Created: ", "").strip()

        # Start sprint
        start = self.run_cli(["sprint", "start", sprint_id], env, cwd)
        assert start.returncode == 0

        # Check status
        status = self.run_cli(["sprint", "status"], env, cwd)
        assert status.returncode == 0

        # Complete sprint
        complete = self.run_cli(["sprint", "complete", sprint_id], env, cwd)
        assert complete.returncode == 0

        # Verify epic still shows the sprint
        show = self.run_cli(["epic", "show", epic_id], env, cwd)
        assert show.returncode == 0


class TestExpressionQueryCommands(TestGoTCLI):
    """Integration tests for expression query CLI commands (got expr)."""

    def test_expr_help(self, got_env):
        """Test expression query help shows all options."""
        env, cwd = got_env
        result = self.run_cli(["expr", "--help"], env, cwd)

        assert result.returncode == 0
        assert "--type" in result.stdout
        assert "--format" in result.stdout
        assert "--count" in result.stdout
        assert "--list-fields" in result.stdout
        assert "--list-functions" in result.stdout
        assert "--explain" in result.stdout

    def test_expr_list_functions(self, got_env):
        """Test listing available query functions."""
        env, cwd = got_env
        result = self.run_cli(["expr", "--list-functions"], env, cwd)

        assert result.returncode == 0
        assert "Available query functions" in result.stdout
        # Check for some known functions
        assert "blocked" in result.stdout or "connected_to" in result.stdout

    def test_expr_list_fields_task(self, got_env):
        """Test listing fields for task entity type."""
        env, cwd = got_env
        result = self.run_cli(["expr", "--list-fields", "--type", "task"], env, cwd)

        assert result.returncode == 0
        # Should show task fields or fall back to common fields
        assert "status" in result.stdout or "title" in result.stdout

    def test_expr_simple_query(self, got_env):
        """Test simple equality expression query."""
        env, cwd = got_env

        # Create a task first
        create_result = self.run_cli(
            ["task", "create", "Test Task", "--priority", "high"],
            env, cwd
        )
        assert create_result.returncode == 0

        # Query for pending tasks (default status)
        result = self.run_cli(
            ["expr", "status = 'pending'"],
            env, cwd
        )

        assert result.returncode == 0
        assert "Results" in result.stdout
        assert "Test Task" in result.stdout

    def test_expr_and_query(self, got_env):
        """Test AND expression query."""
        env, cwd = got_env

        # Create a high priority task
        self.run_cli(
            ["task", "create", "High Priority Task", "--priority", "high"],
            env, cwd
        )
        # Create a low priority task
        self.run_cli(
            ["task", "create", "Low Priority Task", "--priority", "low"],
            env, cwd
        )

        # Query for pending AND high priority
        result = self.run_cli(
            ["expr", "status = 'pending' AND priority = 'high'"],
            env, cwd
        )

        assert result.returncode == 0
        assert "High Priority Task" in result.stdout
        assert "Low Priority Task" not in result.stdout

    def test_expr_not_query(self, got_env):
        """Test NOT expression query."""
        env, cwd = got_env

        # Create a task
        create_result = self.run_cli(
            ["task", "create", "Test Task"],
            env, cwd
        )
        task_id = create_result.stdout.strip().replace("Created: task:", "").strip()

        # Complete the task
        self.run_cli(["task", "complete", task_id, "--retrospective", "Done"], env, cwd)

        # Create another pending task
        self.run_cli(["task", "create", "Pending Task"], env, cwd)

        # Query for NOT completed
        result = self.run_cli(
            ["expr", "NOT status = 'completed'"],
            env, cwd
        )

        assert result.returncode == 0
        assert "Pending Task" in result.stdout
        # The completed task should not be in results

    def test_expr_count_format(self, got_env):
        """Test count-only output format."""
        env, cwd = got_env

        # Create some tasks
        self.run_cli(["task", "create", "Task 1"], env, cwd)
        self.run_cli(["task", "create", "Task 2"], env, cwd)
        self.run_cli(["task", "create", "Task 3"], env, cwd)

        # Query with count flag
        result = self.run_cli(
            ["expr", "status = 'pending'", "--count"],
            env, cwd
        )

        assert result.returncode == 0
        # Output should be just a number
        count = result.stdout.strip()
        assert count.isdigit()
        assert int(count) >= 3

    def test_expr_json_format(self, got_env):
        """Test JSON output format."""
        env, cwd = got_env

        # Create a task
        self.run_cli(
            ["task", "create", "JSON Test Task", "--priority", "medium"],
            env, cwd
        )

        # Query with JSON format
        result = self.run_cli(
            ["expr", "status = 'pending'", "--format", "json"],
            env, cwd
        )

        assert result.returncode == 0
        # Should contain JSON array
        assert "[" in result.stdout
        assert "]" in result.stdout
        # Parse the JSON to verify it's valid (skip the "Results (N):" line)
        lines = result.stdout.strip().split("\n")
        json_start = next(i for i, l in enumerate(lines) if l.strip().startswith("["))
        json_text = "\n".join(lines[json_start:])
        data = json.loads(json_text)
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_expr_ids_format(self, got_env):
        """Test IDs-only output format."""
        env, cwd = got_env

        # Create a task
        self.run_cli(["task", "create", "IDs Test Task"], env, cwd)

        # Query with IDs format
        result = self.run_cli(
            ["expr", "status = 'pending'", "--format", "ids"],
            env, cwd
        )

        assert result.returncode == 0
        # Should contain task IDs
        assert "T-" in result.stdout

    def test_expr_explain(self, got_env):
        """Test explain mode shows AST."""
        env, cwd = got_env

        result = self.run_cli(
            ["expr", "--explain", "status = 'pending' AND priority = 'high'"],
            env, cwd
        )

        assert result.returncode == 0
        assert "Expression:" in result.stdout
        assert "Parsed AST:" in result.stdout
        assert "AND:" in result.stdout
        assert "Comparison:" in result.stdout

    def test_expr_invalid_expression(self, got_env):
        """Test error handling for invalid expressions."""
        env, cwd = got_env

        result = self.run_cli(
            ["expr", "invalid syntax @#$%"],
            env, cwd
        )

        # Should fail gracefully with error message
        assert result.returncode != 0 or "Error" in result.stdout

    def test_expr_no_results(self, got_env):
        """Test query with no matching results."""
        env, cwd = got_env

        # Query for a status that doesn't exist
        result = self.run_cli(
            ["expr", "status = 'nonexistent_status'"],
            env, cwd
        )

        assert result.returncode == 0
        assert "No results found" in result.stdout

    def test_expr_different_entity_type(self, got_env):
        """Test querying different entity types."""
        env, cwd = got_env

        # Create a decision
        self.run_cli(
            ["decision", "log", "Test Decision", "--rationale", "Because testing"],
            env, cwd
        )

        # Query decisions
        result = self.run_cli(
            ["expr", "--type", "decision", "status = 'draft'"],
            env, cwd
        )

        # Should succeed (may or may not have results depending on default status)
        assert result.returncode == 0
