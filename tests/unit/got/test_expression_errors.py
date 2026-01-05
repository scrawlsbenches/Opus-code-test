"""
Unit tests for expression parser error classes.

Tests verify that all error types:
- Include position tracking
- Format context with caret indicators
- Provide helpful suggestions
- Include relevant metadata (expected tokens, available functions, etc.)
"""

import unittest
from cortical.got.expression.errors import (
    QueryError,
    LexerError,
    ParseError,
    ExecutionError,
    QueryValidationError,
)


class TestQueryError(unittest.TestCase):
    """Test the base QueryError class."""

    def test_basic_error_message(self):
        """QueryError can be created with just a message."""
        error = QueryError("Something went wrong")
        self.assertEqual(error.message, "Something went wrong")
        self.assertIsNone(error.position)
        self.assertIsNone(error.source)
        self.assertEqual(error.suggestions, [])

    def test_error_with_position(self):
        """QueryError includes position in message."""
        error = QueryError("Invalid syntax", position=15)
        message = str(error)
        self.assertIn("Invalid syntax", message)

    def test_error_with_source_and_position(self):
        """QueryError shows caret at error position."""
        error = QueryError(
            "Unexpected token",
            position=10,
            source="status = pending AND"
        )
        message = str(error)
        # Should contain the source line
        self.assertIn("status = pending AND", message)
        # Should contain caret indicator
        self.assertIn("^^^", message)

    def test_error_with_suggestions(self):
        """QueryError includes suggestions in formatted output."""
        error = QueryError(
            "Unknown field",
            suggestions=["status", "priority", "category"]
        )
        message = str(error)
        self.assertIn("Did you mean:", message)
        self.assertIn("status", message)
        self.assertIn("priority", message)

    def test_format_with_context_no_position(self):
        """format_with_context works without position."""
        error = QueryError("Simple error")
        formatted = error.format_with_context()
        self.assertIn("QueryError: Simple error", formatted)

    def test_format_with_context_with_caret(self):
        """format_with_context shows caret at correct position."""
        error = QueryError(
            "Error here",
            position=5,
            source="hello world"
        )
        formatted = error.format_with_context()
        lines = formatted.split('\n')

        # Find the source line
        source_line_idx = None
        for i, line in enumerate(lines):
            if "hello world" in line:
                source_line_idx = i
                break

        self.assertIsNotNone(source_line_idx, "Source line not found")

        # Check caret line is right after source line
        if source_line_idx is not None and source_line_idx + 1 < len(lines):
            caret_line = lines[source_line_idx + 1]
            # Caret should be at position 5 (plus indentation)
            # The indentation is 2 spaces ("  ")
            self.assertIn("^^^", caret_line)
            # Position of caret should be at index 7 (2 spaces + 5 position)
            caret_pos = caret_line.index("^^^")
            self.assertEqual(caret_pos, 7)  # 2 spaces indentation + 5 position


class TestLexerError(unittest.TestCase):
    """Test LexerError for tokenization errors."""

    def test_basic_lexer_error(self):
        """LexerError can be raised with message."""
        error = LexerError("Invalid character")
        self.assertEqual(error.message, "Invalid character")
        self.assertIsNone(error.invalid_character)

    def test_lexer_error_with_invalid_character(self):
        """LexerError tracks the invalid character."""
        error = LexerError(
            "Unexpected character '@'",
            position=10,
            source="status = @pending",
            invalid_character="@"
        )
        self.assertEqual(error.invalid_character, "@")
        message = str(error)
        self.assertIn("Unexpected character", message)
        self.assertIn("status = @pending", message)

    def test_lexer_error_shows_position(self):
        """LexerError shows caret at error position."""
        error = LexerError(
            "Invalid character '#'",
            position=9,
            source="priority # high",
            invalid_character="#"
        )
        formatted = error.format_with_context()
        self.assertIn("priority # high", formatted)
        self.assertIn("^^^", formatted)


class TestParseError(unittest.TestCase):
    """Test ParseError for parsing errors."""

    def test_basic_parse_error(self):
        """ParseError can be raised with message."""
        error = ParseError("Syntax error")
        self.assertEqual(error.message, "Syntax error")
        self.assertEqual(error.expected_tokens, [])
        self.assertIsNone(error.found_token)

    def test_parse_error_with_expected_tokens(self):
        """ParseError lists expected tokens."""
        error = ParseError(
            "Unexpected token 'AND'",
            expected_tokens=["value", "string", "number"],
            found_token="AND"
        )
        formatted = error.format_with_context()
        self.assertIn("Expected: value, string, number", formatted)

    def test_parse_error_with_position(self):
        """ParseError shows caret at error position."""
        error = ParseError(
            "Unexpected token 'AND' at position 15",
            position=15,
            source="status = pending AND",
            expected_tokens=["value", "string", "number"],
            found_token="AND"
        )
        formatted = error.format_with_context()
        self.assertIn("status = pending AND", formatted)
        self.assertIn("^^^", formatted)
        self.assertIn("Expected:", formatted)

    def test_parse_error_with_suggestions(self):
        """ParseError includes suggestions in output."""
        error = ParseError(
            "Missing quotes around value",
            position=9,
            source="status = pending",
            suggestions=["status = 'pending'", "status = \"pending\""]
        )
        formatted = error.format_with_context()
        self.assertIn("Did you mean:", formatted)
        self.assertIn("status = 'pending'", formatted)

    def test_parse_error_format_all_features(self):
        """ParseError format includes all features together."""
        error = ParseError(
            "Unexpected token",
            position=15,
            source="status = pending AND",
            expected_tokens=["value"],
            found_token="AND",
            suggestions=["status = 'pending' AND ..."]
        )
        formatted = error.format_with_context()

        # Should have all components
        self.assertIn("ParseError:", formatted)
        self.assertIn("status = pending AND", formatted)
        self.assertIn("^^^", formatted)
        self.assertIn("Expected: value", formatted)
        self.assertIn("Did you mean:", formatted)


class TestExecutionError(unittest.TestCase):
    """Test ExecutionError for query execution errors."""

    def test_basic_execution_error(self):
        """ExecutionError can be raised with message."""
        error = ExecutionError("Execution failed")
        self.assertEqual(error.message, "Execution failed")
        self.assertIsNone(error.function_name)
        self.assertEqual(error.available_functions, [])

    def test_execution_error_with_function_name(self):
        """ExecutionError tracks unknown function name."""
        error = ExecutionError(
            "Unknown function 'foo'",
            function_name="foo",
            available_functions=["count", "sum", "avg"]
        )
        self.assertEqual(error.function_name, "foo")
        formatted = error.format_with_context()
        self.assertIn("Available functions:", formatted)
        # Functions should be sorted
        self.assertIn("avg, count, sum", formatted)

    def test_execution_error_with_position(self):
        """ExecutionError shows caret at error position."""
        error = ExecutionError(
            "Unknown function",
            position=0,
            source="foo()",
            function_name="foo",
            available_functions=["count", "sum"]
        )
        formatted = error.format_with_context()
        self.assertIn("foo()", formatted)
        self.assertIn("^^^", formatted)

    def test_execution_error_with_suggestions(self):
        """ExecutionError provides function suggestions."""
        error = ExecutionError(
            "Unknown function 'cnt'",
            function_name="cnt",
            available_functions=["count", "sum", "avg"],
            suggestions=["count"]
        )
        formatted = error.format_with_context()
        self.assertIn("Did you mean:", formatted)
        self.assertIn("count", formatted)

    def test_execution_error_sorts_available_functions(self):
        """ExecutionError sorts available functions alphabetically."""
        error = ExecutionError(
            "Unknown function",
            available_functions=["sum", "count", "avg", "max", "min"]
        )
        formatted = error.format_with_context()
        # Should be sorted: avg, count, max, min, sum
        self.assertIn("avg, count, max, min, sum", formatted)


class TestQueryValidationError(unittest.TestCase):
    """Test QueryValidationError for schema validation errors."""

    def test_basic_validation_error(self):
        """QueryValidationError can be raised with message."""
        error = QueryValidationError("Invalid field")
        self.assertEqual(error.message, "Invalid field")
        self.assertIsNone(error.field_name)
        self.assertEqual(error.valid_fields, [])

    def test_validation_error_with_field_name(self):
        """QueryValidationError tracks unknown field name."""
        error = QueryValidationError(
            "Unknown field 'stat'",
            field_name="stat",
            valid_fields=["status", "priority", "category"]
        )
        self.assertEqual(error.field_name, "stat")
        formatted = error.format_with_context()
        self.assertIn("Valid fields:", formatted)

    def test_validation_error_auto_suggests_similar_fields(self):
        """QueryValidationError auto-suggests similar field names."""
        error = QueryValidationError(
            "Unknown field 'stat'",
            field_name="stat",
            valid_fields=["status", "priority", "category", "state"]
        )
        # Should auto-suggest fields starting with "stat"
        self.assertGreater(len(error.suggestions), 0)
        # Status and state should be suggested
        self.assertTrue(
            any("stat" in s.lower() for s in error.suggestions),
            f"Expected 'stat' substring in suggestions: {error.suggestions}"
        )

    def test_validation_error_prefix_match_suggestion(self):
        """QueryValidationError suggests fields with matching prefix."""
        error = QueryValidationError(
            "Unknown field 'pri'",
            field_name="pri",
            valid_fields=["status", "priority", "category", "private"]
        )
        # Should suggest priority and/or private
        formatted = error.format_with_context()
        self.assertIn("Did you mean:", formatted)

    def test_validation_error_substring_match_suggestion(self):
        """QueryValidationError suggests fields with substring match."""
        error = QueryValidationError(
            "Unknown field 'ego'",
            field_name="ego",
            valid_fields=["status", "priority", "category", "tags"]
        )
        # Should suggest category (contains "ego")
        suggestions = error.suggestions
        self.assertTrue(
            any("category" in s for s in suggestions),
            f"Expected 'category' in suggestions: {suggestions}"
        )

    def test_validation_error_limits_suggestions(self):
        """QueryValidationError limits suggestions to 3."""
        error = QueryValidationError(
            "Unknown field 't'",
            field_name="t",
            valid_fields=["title", "tags", "type", "timestamp", "task_id"]
        )
        # Should have at most 3 suggestions
        self.assertLessEqual(len(error.suggestions), 3)

    def test_validation_error_manual_suggestions_override(self):
        """QueryValidationError respects manually provided suggestions."""
        error = QueryValidationError(
            "Unknown field 'foo'",
            field_name="foo",
            valid_fields=["bar", "baz"],
            suggestions=["custom_suggestion"]
        )
        # Manual suggestions should be used
        self.assertEqual(error.suggestions, ["custom_suggestion"])

    def test_validation_error_with_position(self):
        """QueryValidationError shows caret at error position."""
        error = QueryValidationError(
            "Unknown field",
            position=0,
            source="stat = 'pending'",
            field_name="stat",
            valid_fields=["status", "priority"]
        )
        formatted = error.format_with_context()
        self.assertIn("stat = 'pending'", formatted)
        self.assertIn("^^^", formatted)

    def test_validation_error_sorts_valid_fields(self):
        """QueryValidationError sorts valid fields alphabetically."""
        error = QueryValidationError(
            "Unknown field",
            field_name="foo",
            valid_fields=["status", "category", "priority", "tags"],
            suggestions=[]  # Disable auto-suggestion
        )
        formatted = error.format_with_context()
        # Should be sorted: category, priority, status, tags
        self.assertIn("category, priority, status, tags", formatted)


class TestErrorInheritance(unittest.TestCase):
    """Test that all error types inherit from QueryError."""

    def test_lexer_error_inheritance(self):
        """LexerError inherits from QueryError."""
        self.assertTrue(issubclass(LexerError, QueryError))
        with self.assertRaises(QueryError):
            raise LexerError("Test")

    def test_parse_error_inheritance(self):
        """ParseError inherits from QueryError."""
        self.assertTrue(issubclass(ParseError, QueryError))
        with self.assertRaises(QueryError):
            raise ParseError("Test")

    def test_execution_error_inheritance(self):
        """ExecutionError inherits from QueryError."""
        self.assertTrue(issubclass(ExecutionError, QueryError))
        with self.assertRaises(QueryError):
            raise ExecutionError("Test")

    def test_validation_error_inheritance(self):
        """QueryValidationError inherits from QueryError."""
        self.assertTrue(issubclass(QueryValidationError, QueryError))
        with self.assertRaises(QueryError):
            raise QueryValidationError("Test")


class TestErrorMessageQuality(unittest.TestCase):
    """Test that error messages are helpful and readable."""

    def test_parse_error_readable_format(self):
        """ParseError produces readable error message."""
        error = ParseError(
            "Unexpected token 'AND' at position 15",
            position=15,
            source="status = pending AND",
            expected_tokens=["value (string, number, or identifier)"],
            suggestions=["status = 'pending' AND ..."]
        )
        message = str(error)

        # Message should be multi-line and readable
        lines = message.split('\n')
        self.assertGreater(len(lines), 3)  # Should have multiple lines

        # Should not be cryptic
        self.assertIn("status = pending AND", message)
        self.assertIn("Expected:", message)
        self.assertIn("Did you mean:", message)

    def test_validation_error_readable_format(self):
        """QueryValidationError produces readable error message."""
        error = QueryValidationError(
            "Unknown field 'stat'",
            position=0,
            source="stat = 'pending'",
            field_name="stat",
            valid_fields=["status", "priority", "category"]
        )
        message = str(error)

        # Should show the problematic field
        self.assertIn("stat = 'pending'", message)
        # Should show valid alternatives
        self.assertIn("Valid fields:", message)
        # Should suggest similar fields
        self.assertIn("Did you mean:", message)


if __name__ == '__main__':
    unittest.main()
