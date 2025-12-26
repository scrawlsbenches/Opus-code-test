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
