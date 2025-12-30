"""
Tests for cortical.got.cli.batch module.

This module provides comprehensive tests for the GoT batch CLI
operations, including:
- BatchOperation and BatchResult dataclasses
- parse_batch_line function
- Command argument parsing functions
- resolve_variables function
- BatchParser class
- BatchExecutor class
"""

import json
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from io import StringIO


class TestBatchOperation:
    """Tests for BatchOperation dataclass."""

    def test_batch_operation_creation(self):
        """Test basic BatchOperation creation."""
        from cortical.got.cli.batch import BatchOperation

        op = BatchOperation(
            command="task",
            action="create",
            args={"title": "Test Task"},
            alias="t1",
            line_number=1,
            raw_line='task create "Test Task" as t1'
        )

        assert op.command == "task"
        assert op.action == "create"
        assert op.args == {"title": "Test Task"}
        assert op.alias == "t1"
        assert op.line_number == 1

    def test_batch_operation_defaults(self):
        """Test BatchOperation with default values."""
        from cortical.got.cli.batch import BatchOperation

        op = BatchOperation(
            command="sprint",
            action="create",
            args={"name": "Sprint 1"}
        )

        assert op.alias is None
        assert op.line_number == 0
        assert op.raw_line == ""


class TestBatchResult:
    """Tests for BatchResult dataclass."""

    def test_batch_result_success(self):
        """Test successful BatchResult."""
        from cortical.got.cli.batch import BatchResult

        result = BatchResult(
            success=True,
            created=["T-123", "S-456"],
            aliases={"t1": "T-123", "s1": "S-456"}
        )

        assert result.success is True
        assert len(result.created) == 2
        assert result.aliases["t1"] == "T-123"
        assert result.error is None

    def test_batch_result_failure(self):
        """Test failed BatchResult."""
        from cortical.got.cli.batch import BatchResult

        result = BatchResult(
            success=False,
            error="Something went wrong"
        )

        assert result.success is False
        assert result.error == "Something went wrong"

    def test_batch_result_to_json(self):
        """Test BatchResult JSON serialization."""
        from cortical.got.cli.batch import BatchResult

        result = BatchResult(
            success=True,
            created=["T-123"],
            aliases={"t1": "T-123"},
            dry_run=True
        )

        json_data = result.to_json()

        assert json_data["success"] is True
        assert json_data["created"] == ["T-123"]
        assert json_data["aliases"] == {"t1": "T-123"}
        assert json_data["dry_run"] is True
        assert json_data["error"] is None

    def test_batch_result_default_lists(self):
        """Test BatchResult default empty lists."""
        from cortical.got.cli.batch import BatchResult

        result = BatchResult(success=True)

        assert result.created == []
        assert result.aliases == {}
        assert result.planned == []


class TestParseBatchLine:
    """Tests for parse_batch_line function."""

    def test_parse_empty_line(self):
        """Test parsing empty line returns None."""
        from cortical.got.cli.batch import parse_batch_line

        result = parse_batch_line("")
        assert result is None

        result = parse_batch_line("   ")
        assert result is None

    def test_parse_comment_line(self):
        """Test parsing comment line returns None."""
        from cortical.got.cli.batch import parse_batch_line

        result = parse_batch_line("# This is a comment")
        assert result is None

    def test_parse_task_create(self):
        """Test parsing task create command."""
        from cortical.got.cli.batch import parse_batch_line

        result = parse_batch_line('task create "My Task Title" --priority high')

        assert result is not None
        assert result.command == "task"
        assert result.action == "create"
        assert result.args["title"] == "My Task Title"
        assert result.args["priority"] == "high"

    def test_parse_task_create_with_alias(self):
        """Test parsing task create with alias."""
        from cortical.got.cli.batch import parse_batch_line

        result = parse_batch_line('task create "Task 1" as t1')

        assert result.alias == "t1"
        assert result.args["title"] == "Task 1"

    def test_parse_sprint_create(self):
        """Test parsing sprint create command."""
        from cortical.got.cli.batch import parse_batch_line

        result = parse_batch_line('sprint create "Sprint 28" --number 28')

        assert result.command == "sprint"
        assert result.action == "create"
        assert result.args["name"] == "Sprint 28"
        assert result.args["number"] == 28

    def test_parse_edge_add(self):
        """Test parsing edge add command."""
        from cortical.got.cli.batch import parse_batch_line

        result = parse_batch_line('edge add $t1 $t2 DEPENDS_ON')

        assert result.command == "edge"
        assert result.action == "add"
        assert result.args["source_id"] == "$t1"
        assert result.args["target_id"] == "$t2"
        assert result.args["edge_type"] == "DEPENDS_ON"

    def test_parse_edge_add_with_weight(self):
        """Test parsing edge add with weight."""
        from cortical.got.cli.batch import parse_batch_line

        result = parse_batch_line('edge add $t1 $t2 DEPENDS_ON --weight 0.5')

        assert result.args["weight"] == 0.5

    def test_parse_decision_log(self):
        """Test parsing decision log command."""
        from cortical.got.cli.batch import parse_batch_line

        result = parse_batch_line('decision log "Use PostgreSQL" --rationale "Better performance"')

        assert result.command == "decision"
        assert result.action == "log"
        assert result.args["decision"] == "Use PostgreSQL"
        assert result.args["rationale"] == "Better performance"

    def test_parse_epic_create(self):
        """Test parsing epic create command."""
        from cortical.got.cli.batch import parse_batch_line

        result = parse_batch_line('epic create "My Epic" --description "Epic description"')

        assert result.command == "epic"
        assert result.action == "create"
        assert result.args["name"] == "My Epic"
        assert result.args["description"] == "Epic description"

    def test_parse_invalid_too_few_tokens(self):
        """Test parsing with too few tokens raises error."""
        from cortical.got.cli.batch import parse_batch_line, BatchError

        with pytest.raises(BatchError) as exc_info:
            parse_batch_line("task")

        assert "Expected 'command action [args]'" in str(exc_info.value)

    def test_parse_unknown_command(self):
        """Test parsing unknown command raises error."""
        from cortical.got.cli.batch import parse_batch_line, BatchError

        with pytest.raises(BatchError) as exc_info:
            parse_batch_line("unknown command arg1")

        assert "Unknown command" in str(exc_info.value)

    def test_parse_malformed_quotes(self):
        """Test parsing with malformed quotes raises error."""
        from cortical.got.cli.batch import parse_batch_line, BatchError

        with pytest.raises(BatchError) as exc_info:
            parse_batch_line('task create "unclosed quote')

        assert "Parse error" in str(exc_info.value)


class TestParseTaskCreateArgs:
    """Tests for _parse_task_create_args function."""

    def test_parse_all_task_flags(self):
        """Test parsing all task create flags."""
        from cortical.got.cli.batch import parse_batch_line

        result = parse_batch_line(
            'task create "Task" --priority high --category feature --sprint $s1 '
            '--depends-on $t1 --blocks $t2 --description "Desc"'
        )

        assert result.args["title"] == "Task"
        assert result.args["priority"] == "high"
        assert result.args["category"] == "feature"
        assert result.args["sprint"] == "$s1"
        assert result.args["depends_on"] == "$t1"
        assert result.args["blocks"] == "$t2"
        assert result.args["description"] == "Desc"

    def test_parse_task_no_title(self):
        """Test parsing task create without title raises error."""
        from cortical.got.cli.batch import parse_batch_line, BatchError

        with pytest.raises(BatchError) as exc_info:
            parse_batch_line('task create --priority high')

        assert "requires a title" in str(exc_info.value)

    def test_parse_task_boolean_flag(self):
        """Test parsing task with boolean flag."""
        from cortical.got.cli.batch import parse_batch_line

        result = parse_batch_line('task create "Task" --urgent')

        assert result.args["urgent"] is True


class TestParseSprintCreateArgs:
    """Tests for _parse_sprint_create_args function."""

    def test_parse_sprint_all_flags(self):
        """Test parsing sprint create with all flags."""
        from cortical.got.cli.batch import parse_batch_line

        result = parse_batch_line('sprint create "Sprint Name" --number 28 --epic $e1')

        assert result.args["name"] == "Sprint Name"
        assert result.args["number"] == 28
        assert result.args["epic"] == "$e1"

    def test_parse_sprint_no_name(self):
        """Test parsing sprint create without name raises error."""
        from cortical.got.cli.batch import parse_batch_line, BatchError

        with pytest.raises(BatchError) as exc_info:
            parse_batch_line('sprint create --number 28')

        assert "requires a name" in str(exc_info.value)


class TestParseEpicCreateArgs:
    """Tests for _parse_epic_create_args function."""

    def test_parse_epic_no_name(self):
        """Test parsing epic create without name raises error."""
        from cortical.got.cli.batch import parse_batch_line, BatchError

        with pytest.raises(BatchError) as exc_info:
            parse_batch_line('epic create --description "Some desc"')

        assert "requires a name" in str(exc_info.value)

    def test_parse_epic_boolean_flag(self):
        """Test parsing epic with boolean flag."""
        from cortical.got.cli.batch import parse_batch_line

        result = parse_batch_line('epic create "My Epic" --archived')

        assert result.args["archived"] is True


class TestParseEdgeAddArgs:
    """Tests for _parse_edge_add_args function."""

    def test_parse_edge_insufficient_args(self):
        """Test parsing edge add with insufficient arguments."""
        from cortical.got.cli.batch import parse_batch_line, BatchError

        with pytest.raises(BatchError) as exc_info:
            parse_batch_line('edge add $t1 $t2')  # Missing edge type

        assert "requires SOURCE TARGET EDGE_TYPE" in str(exc_info.value)


class TestParseDecisionLogArgs:
    """Tests for _parse_decision_log_args function."""

    def test_parse_decision_no_decision(self):
        """Test parsing decision log without decision raises error."""
        from cortical.got.cli.batch import parse_batch_line, BatchError

        with pytest.raises(BatchError) as exc_info:
            parse_batch_line('decision log --rationale "Some reason"')

        assert "requires a decision" in str(exc_info.value)


class TestResolveVariables:
    """Tests for resolve_variables function."""

    def test_resolve_string_variable(self):
        """Test resolving a string variable."""
        from cortical.got.cli.batch import resolve_variables

        result = resolve_variables("$t1", {"t1": "T-123"})
        assert result == "T-123"

    def test_resolve_non_variable_string(self):
        """Test non-variable string passes through."""
        from cortical.got.cli.batch import resolve_variables

        result = resolve_variables("plain_string", {})
        assert result == "plain_string"

    def test_resolve_unknown_variable(self):
        """Test resolving unknown variable raises error."""
        from cortical.got.cli.batch import resolve_variables, BatchError

        with pytest.raises(BatchError) as exc_info:
            resolve_variables("$unknown", {})

        assert "Unknown variable" in str(exc_info.value)

    def test_resolve_dict_values(self):
        """Test resolving variables in a dictionary."""
        from cortical.got.cli.batch import resolve_variables

        result = resolve_variables(
            {"key1": "$t1", "key2": "static"},
            {"t1": "T-123"}
        )

        assert result["key1"] == "T-123"
        assert result["key2"] == "static"

    def test_resolve_list_values(self):
        """Test resolving variables in a list."""
        from cortical.got.cli.batch import resolve_variables

        result = resolve_variables(
            ["$t1", "static", "$t2"],
            {"t1": "T-123", "t2": "T-456"}
        )

        assert result == ["T-123", "static", "T-456"]

    def test_resolve_non_string_passthrough(self):
        """Test non-string values pass through unchanged."""
        from cortical.got.cli.batch import resolve_variables

        result = resolve_variables(42, {})
        assert result == 42

        result = resolve_variables(True, {})
        assert result is True


class TestBatchParser:
    """Tests for BatchParser class."""

    def test_parse_empty_script(self):
        """Test parsing empty script."""
        from cortical.got.cli.batch import BatchParser

        parser = BatchParser()
        result = parser.parse("")

        assert result == []

    def test_parse_multiline_script(self):
        """Test parsing multiline script."""
        from cortical.got.cli.batch import BatchParser

        parser = BatchParser()
        script = '''
        sprint create "Sprint 1" as s1
        task create "Task 1" --sprint $s1 as t1
        task create "Task 2" --sprint $s1 as t2
        edge add $t2 $t1 DEPENDS_ON
        '''

        result = parser.parse(script)

        assert len(result) == 4
        assert result[0].command == "sprint"
        assert result[1].command == "task"
        assert result[2].command == "task"
        assert result[3].command == "edge"

    def test_parse_script_with_comments(self):
        """Test parsing script with comments."""
        from cortical.got.cli.batch import BatchParser

        parser = BatchParser()
        script = '''
        # This is a comment
        sprint create "Sprint 1" as s1
        # Another comment
        task create "Task 1" as t1
        '''

        result = parser.parse(script)

        assert len(result) == 2

    def test_parse_preserves_line_numbers(self):
        """Test that parser preserves line numbers for errors."""
        from cortical.got.cli.batch import BatchParser

        parser = BatchParser()
        # After strip(), the lines are indexed starting at 1
        script = '''sprint create "Sprint 1" as s1
task create "Task 1" as t1'''

        result = parser.parse(script)

        # Line numbers are 1-indexed for the stripped lines
        assert result[0].line_number == 1
        assert result[1].line_number == 2


class TestBatchExecutor:
    """Tests for BatchExecutor class."""

    def test_executor_dry_run(self):
        """Test executor in dry run mode."""
        from cortical.got.cli.batch import BatchExecutor

        mock_manager = MagicMock()
        executor = BatchExecutor(mock_manager)

        script = '''
        sprint create "Sprint 1" as s1
        task create "Task 1" as t1
        '''

        result = executor.execute(script, dry_run=True)

        assert result.success is True
        assert result.dry_run is True
        assert len(result.planned) == 2
        # Manager should not be called in dry run
        mock_manager.create_sprint.assert_not_called()

    def test_executor_success(self):
        """Test executor successful execution."""
        from cortical.got.cli.batch import BatchExecutor

        mock_manager = MagicMock()
        mock_manager.create_sprint.return_value = "S-123"
        mock_manager.create_task.return_value = "T-456"

        executor = BatchExecutor(mock_manager)

        script = '''
        sprint create "Sprint 1" as s1
        task create "Task 1" --sprint $s1 as t1
        '''

        result = executor.execute(script)

        assert result.success is True
        assert "S-123" in result.created
        assert "T-456" in result.created
        assert result.aliases["s1"] == "S-123"
        assert result.aliases["t1"] == "T-456"
        mock_manager.save.assert_called_once()

    def test_executor_parse_error(self):
        """Test executor handles parse errors."""
        from cortical.got.cli.batch import BatchExecutor

        mock_manager = MagicMock()
        executor = BatchExecutor(mock_manager)

        script = 'invalid command'

        result = executor.execute(script)

        assert result.success is False
        assert result.error is not None

    def test_executor_execution_error(self):
        """Test executor handles execution errors."""
        from cortical.got.cli.batch import BatchExecutor

        mock_manager = MagicMock()
        mock_manager.create_sprint.side_effect = Exception("DB error")

        executor = BatchExecutor(mock_manager)

        script = 'sprint create "Sprint 1"'

        result = executor.execute(script)

        assert result.success is False
        assert "DB error" in result.error
        # save should not be called on failure
        mock_manager.save.assert_not_called()

    def test_executor_edge_add(self):
        """Test executor edge add operation."""
        from cortical.got.cli.batch import BatchExecutor

        mock_manager = MagicMock()
        mock_manager.create_task.side_effect = ["T-123", "T-456"]
        mock_manager.add_edge.return_value = MagicMock()

        executor = BatchExecutor(mock_manager)

        script = '''
        task create "Task 1" as t1
        task create "Task 2" as t2
        edge add $t2 $t1 DEPENDS_ON
        '''

        result = executor.execute(script)

        assert result.success is True
        mock_manager.add_edge.assert_called_once_with(
            source_id="T-456",
            target_id="T-123",
            edge_type="DEPENDS_ON",
            weight=1.0
        )

    def test_executor_decision_log(self):
        """Test executor decision log operation."""
        from cortical.got.cli.batch import BatchExecutor

        mock_manager = MagicMock()
        mock_manager.log_decision.return_value = "D-123"

        executor = BatchExecutor(mock_manager)

        script = 'decision log "Use PostgreSQL" --rationale "Better performance"'

        result = executor.execute(script)

        assert result.success is True
        mock_manager.log_decision.assert_called_once()

    def test_executor_epic_create(self):
        """Test executor epic create operation."""
        from cortical.got.cli.batch import BatchExecutor

        mock_manager = MagicMock()
        mock_manager.create_epic.return_value = "E-123"

        executor = BatchExecutor(mock_manager)

        script = 'epic create "My Epic" --description "Epic desc"'

        result = executor.execute(script)

        assert result.success is True
        assert "E-123" in result.created
        mock_manager.create_epic.assert_called_once_with(
            name="My Epic",
            description="Epic desc"
        )


class TestSetupBatchParser:
    """Tests for setup_batch_parser function."""

    def test_setup_adds_batch_subcommand(self):
        """Test that setup_batch_parser adds batch subcommand."""
        from cortical.got.cli.batch import setup_batch_parser

        mock_subparsers = MagicMock()
        mock_parser = MagicMock()
        mock_subparsers.add_parser.return_value = mock_parser

        setup_batch_parser(mock_subparsers)

        mock_subparsers.add_parser.assert_called_once()
        call_args = mock_subparsers.add_parser.call_args
        assert call_args[0][0] == "batch"

        # Verify arguments were added
        assert mock_parser.add_argument.call_count >= 4


class TestHandleBatchCommand:
    """Tests for handle_batch_command function."""

    def test_handle_from_file(self, tmp_path):
        """Test handling batch from file."""
        from cortical.got.cli.batch import handle_batch_command

        # Create batch file
        batch_file = tmp_path / "batch.got"
        batch_file.write_text('sprint create "Sprint 1"')

        mock_manager = MagicMock()
        mock_manager.create_sprint.return_value = "S-123"

        args = MagicMock()
        args.file = str(batch_file)
        args.dry_run = False
        args.output_json = False
        args.no_atomic = False

        result = handle_batch_command(args, mock_manager)

        assert result == 0

    def test_handle_file_not_found(self):
        """Test handling non-existent file."""
        from cortical.got.cli.batch import handle_batch_command

        mock_manager = MagicMock()

        args = MagicMock()
        args.file = "/nonexistent/file.got"

        result = handle_batch_command(args, mock_manager)

        assert result == 1

    def test_handle_empty_script(self, tmp_path):
        """Test handling empty script."""
        from cortical.got.cli.batch import handle_batch_command

        batch_file = tmp_path / "empty.got"
        batch_file.write_text("")

        mock_manager = MagicMock()

        args = MagicMock()
        args.file = str(batch_file)

        result = handle_batch_command(args, mock_manager)

        assert result == 1

    def test_handle_json_output(self, tmp_path):
        """Test handling with JSON output."""
        from cortical.got.cli.batch import handle_batch_command

        batch_file = tmp_path / "batch.got"
        batch_file.write_text('sprint create "Sprint 1" as s1')

        mock_manager = MagicMock()
        mock_manager.create_sprint.return_value = "S-123"

        args = MagicMock()
        args.file = str(batch_file)
        args.dry_run = False
        args.output_json = True
        args.no_atomic = False

        # Capture stdout
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            result = handle_batch_command(args, mock_manager)

        assert result == 0
        # Verify JSON was printed
        output = mock_stdout.getvalue()
        parsed = json.loads(output)
        assert parsed["success"] is True

    def test_handle_dry_run_output(self, tmp_path):
        """Test handling with dry run output."""
        from cortical.got.cli.batch import handle_batch_command

        batch_file = tmp_path / "batch.got"
        batch_file.write_text('sprint create "Sprint 1" as s1')

        mock_manager = MagicMock()

        args = MagicMock()
        args.file = str(batch_file)
        args.dry_run = True
        args.output_json = False
        args.no_atomic = False

        result = handle_batch_command(args, mock_manager)

        assert result == 0
        mock_manager.create_sprint.assert_not_called()

    def test_handle_from_stdin(self):
        """Test handling batch from stdin."""
        from cortical.got.cli.batch import handle_batch_command

        mock_manager = MagicMock()
        mock_manager.create_sprint.return_value = "S-123"

        args = MagicMock()
        args.file = None
        args.dry_run = False
        args.output_json = False
        args.no_atomic = False

        with patch('sys.stdin', StringIO('sprint create "Sprint 1"')):
            with patch('sys.stdin.isatty', return_value=False):
                result = handle_batch_command(args, mock_manager)

        assert result == 0


class TestBatchError:
    """Tests for BatchError exception."""

    def test_batch_error_message(self):
        """Test BatchError message."""
        from cortical.got.cli.batch import BatchError

        error = BatchError("Test error message")

        assert str(error) == "Test error message"

    def test_batch_error_inheritance(self):
        """Test BatchError inherits from Exception."""
        from cortical.got.cli.batch import BatchError

        assert issubclass(BatchError, Exception)
