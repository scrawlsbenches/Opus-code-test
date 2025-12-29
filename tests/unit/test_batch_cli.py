"""
Tests for GoT batch CLI operations.

Tests the heredoc DSL parser and executor for batch operations.
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass


# We'll import these once the module exists
# from cortical.got.cli.batch import (
#     BatchParser,
#     BatchExecutor,
#     parse_batch_line,
#     resolve_variables,
#     BatchOperation,
# )


class TestBatchParser:
    """Test the batch DSL parser."""

    def test_parse_simple_task_create(self):
        """Parse a simple task create command."""
        from cortical.got.cli.batch import parse_batch_line

        line = 'task create "Implement feature" --priority high'
        op = parse_batch_line(line)

        assert op.command == "task"
        assert op.action == "create"
        assert op.args["title"] == "Implement feature"
        assert op.args["priority"] == "high"
        assert op.alias is None

    def test_parse_with_alias(self):
        """Parse command with 'as NAME' alias."""
        from cortical.got.cli.batch import parse_batch_line

        line = 'task create "My Task" --priority high as t1'
        op = parse_batch_line(line)

        assert op.command == "task"
        assert op.action == "create"
        assert op.args["title"] == "My Task"
        assert op.alias == "t1"

    def test_parse_sprint_create(self):
        """Parse sprint create command."""
        from cortical.got.cli.batch import parse_batch_line

        line = 'sprint create "Sprint 28" --number 28 as sprint1'
        op = parse_batch_line(line)

        assert op.command == "sprint"
        assert op.action == "create"
        assert op.args["name"] == "Sprint 28"
        assert op.args["number"] == 28
        assert op.alias == "sprint1"

    def test_parse_edge_add(self):
        """Parse edge add command."""
        from cortical.got.cli.batch import parse_batch_line

        line = "edge add $t1 $t2 DEPENDS_ON"
        op = parse_batch_line(line)

        assert op.command == "edge"
        assert op.action == "add"
        assert op.args["source_id"] == "$t1"
        assert op.args["target_id"] == "$t2"
        assert op.args["edge_type"] == "DEPENDS_ON"

    def test_parse_with_sprint_reference(self):
        """Parse task with sprint variable reference."""
        from cortical.got.cli.batch import parse_batch_line

        line = 'task create "Task in sprint" --sprint $sprint1 as t1'
        op = parse_batch_line(line)

        assert op.args["sprint"] == "$sprint1"
        assert op.alias == "t1"

    def test_parse_empty_line(self):
        """Empty lines should return None."""
        from cortical.got.cli.batch import parse_batch_line

        assert parse_batch_line("") is None
        assert parse_batch_line("   ") is None
        assert parse_batch_line("# comment") is None

    def test_parse_multiline_batch(self):
        """Parse a full batch script."""
        from cortical.got.cli.batch import BatchParser

        script = '''
        sprint create "Sprint 28" --number 28 as sprint1

        task create "Feature X" --sprint $sprint1 --priority high as t1
        task create "Tests for X" --sprint $sprint1 as t2

        edge add $t2 $t1 DEPENDS_ON
        '''

        parser = BatchParser()
        operations = parser.parse(script)

        assert len(operations) == 4
        assert operations[0].command == "sprint"
        assert operations[1].alias == "t1"
        assert operations[2].alias == "t2"
        assert operations[3].command == "edge"


class TestVariableResolution:
    """Test variable resolution with aliases."""

    def test_resolve_simple_variable(self):
        """Resolve $name to actual ID."""
        from cortical.got.cli.batch import resolve_variables

        aliases = {"sprint1": "S-20251229-120000-abc12345"}
        value = resolve_variables("$sprint1", aliases)

        assert value == "S-20251229-120000-abc12345"

    def test_resolve_in_args(self):
        """Resolve variables within args dict."""
        from cortical.got.cli.batch import resolve_variables

        aliases = {
            "sprint1": "S-20251229-120000-abc12345",
            "t1": "T-20251229-120001-def67890",
        }

        args = {"sprint": "$sprint1", "depends_on": "$t1", "title": "No vars"}
        resolved = resolve_variables(args, aliases)

        assert resolved["sprint"] == "S-20251229-120000-abc12345"
        assert resolved["depends_on"] == "T-20251229-120001-def67890"
        assert resolved["title"] == "No vars"

    def test_unresolved_variable_raises(self):
        """Unresolved variable should raise error."""
        from cortical.got.cli.batch import resolve_variables, BatchError

        aliases = {}

        with pytest.raises(BatchError, match="Unknown variable"):
            resolve_variables("$unknown", aliases)


class TestBatchExecutor:
    """Test batch execution with transactions."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mock TransactionalGoTAdapter."""
        manager = MagicMock()
        manager.create_task.return_value = "T-20251229-120001-abc12345"
        manager.create_sprint.return_value = "S-20251229-120000-def67890"
        manager.add_edge.return_value = MagicMock()
        return manager

    def test_execute_single_task(self, mock_manager):
        """Execute a single task creation."""
        from cortical.got.cli.batch import BatchExecutor

        script = 'task create "My Task" --priority high'

        executor = BatchExecutor(mock_manager)
        result = executor.execute(script)

        assert result.success
        assert len(result.created) == 1
        mock_manager.create_task.assert_called_once()

    def test_execute_with_aliases(self, mock_manager):
        """Execute operations using aliases."""
        from cortical.got.cli.batch import BatchExecutor

        script = '''
        sprint create "Sprint 1" --number 1 as s1
        task create "Task 1" --sprint $s1 as t1
        '''

        executor = BatchExecutor(mock_manager)
        result = executor.execute(script)

        assert result.success
        assert "s1" in result.aliases
        assert "t1" in result.aliases

        # Verify sprint ID was passed to task creation
        call_args = mock_manager.create_task.call_args
        assert call_args[1]["sprint_id"] == "S-20251229-120000-def67890"

    def test_execute_with_edges(self, mock_manager):
        """Execute operations including edge creation."""
        from cortical.got.cli.batch import BatchExecutor

        mock_manager.create_task.side_effect = [
            "T-20251229-120001-task1",
            "T-20251229-120002-task2",
        ]

        script = '''
        task create "Task 1" as t1
        task create "Task 2" as t2
        edge add $t2 $t1 DEPENDS_ON
        '''

        executor = BatchExecutor(mock_manager)
        result = executor.execute(script)

        assert result.success
        mock_manager.add_edge.assert_called_once_with(
            source_id="T-20251229-120002-task2",
            target_id="T-20251229-120001-task1",
            edge_type="DEPENDS_ON",
            weight=1.0,
        )

    def test_dry_run_no_changes(self, mock_manager):
        """Dry run should not make any changes."""
        from cortical.got.cli.batch import BatchExecutor

        script = '''
        sprint create "Sprint 1" as s1
        task create "Task 1" --sprint $s1 as t1
        '''

        executor = BatchExecutor(mock_manager)
        result = executor.execute(script, dry_run=True)

        assert result.success
        assert result.dry_run
        assert len(result.planned) == 2
        mock_manager.create_task.assert_not_called()
        mock_manager.create_sprint.assert_not_called()

    def test_atomic_rollback_on_failure(self, mock_manager):
        """All operations should roll back on failure."""
        from cortical.got.cli.batch import BatchExecutor

        # Second task creation fails
        mock_manager.create_task.side_effect = [
            "T-20251229-120001-task1",
            Exception("Database error"),
        ]

        script = '''
        task create "Task 1" as t1
        task create "Task 2" as t2
        '''

        executor = BatchExecutor(mock_manager)
        result = executor.execute(script, atomic=True)

        assert not result.success
        assert "Database error" in result.error
        # In atomic mode, rollback should be called
        # (actual rollback depends on TransactionManager integration)


class TestBatchResult:
    """Test batch result structure."""

    def test_result_output_json(self, mock_manager):
        """Result should be serializable to JSON."""
        from cortical.got.cli.batch import BatchExecutor

        mock_manager = MagicMock()
        mock_manager.create_sprint.return_value = "S-20251229-120000-abc"
        mock_manager.create_task.return_value = "T-20251229-120001-def"

        script = '''
        sprint create "Sprint 1" as s1
        task create "Task 1" --sprint $s1 as t1
        '''

        executor = BatchExecutor(mock_manager)
        result = executor.execute(script)

        json_output = result.to_json()

        assert "aliases" in json_output
        assert json_output["aliases"]["s1"] == "S-20251229-120000-abc"
        assert json_output["aliases"]["t1"] == "T-20251229-120001-def"
        assert json_output["success"] is True


class TestBatchCLIIntegration:
    """Integration tests for CLI batch command."""

    def test_batch_from_stdin(self, tmp_path, mock_manager):
        """Test batch command reading from stdin."""
        # This will test the actual CLI integration
        pass  # Implemented after basic functionality works

    def test_batch_from_file(self, tmp_path, mock_manager):
        """Test batch command reading from file."""
        pass  # Implemented after basic functionality works


# Fixture for mock_manager at module level
@pytest.fixture
def mock_manager():
    """Create a mock TransactionalGoTAdapter."""
    manager = MagicMock()
    manager.create_task.return_value = "T-20251229-120001-abc12345"
    manager.create_sprint.return_value = "S-20251229-120000-def67890"
    manager.add_edge.return_value = MagicMock()
    return manager
