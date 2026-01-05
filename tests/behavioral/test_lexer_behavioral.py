"""
Behavioral tests for the Lexer.

Feature: Query Expression Tokenization

As a developer using the query system,
I want to tokenize query expressions into tokens,
So that the parser can build an AST from them.

These tests verify that the lexer correctly identifies and extracts
tokens from query expressions, handling all supported syntax.
"""

import pytest
from cortical.got.expression.lexer import Lexer, Token, TokenType, tokenize
from cortical.got.expression.errors import LexerError


class TestBasicTokenization:
    """Scenario: Tokenize simple query expressions."""

    def test_tokenize_simple_equality(self):
        """
        Given a simple equality expression,
        When I tokenize it,
        Then I get the correct tokens.
        """
        tokens = tokenize("status = 'pending'")

        assert len(tokens) == 4  # IDENTIFIER, EQ, STRING, EOF
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "status"
        assert tokens[1].type == TokenType.EQ
        assert tokens[1].value == "="
        assert tokens[2].type == TokenType.STRING
        assert tokens[2].value == "pending"
        assert tokens[3].type == TokenType.EOF

    def test_tokenize_numeric_comparison(self):
        """
        Given a numeric comparison,
        When I tokenize it,
        Then I get correct number and operator tokens.
        """
        tokens = tokenize("priority > 5")

        assert len(tokens) == 4  # IDENTIFIER, GT, NUMBER, EOF
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "priority"
        assert tokens[1].type == TokenType.GT
        assert tokens[1].value == ">"
        assert tokens[2].type == TokenType.NUMBER
        assert tokens[2].value == "5"
        assert tokens[3].type == TokenType.EOF

    def test_tokenize_with_whitespace(self):
        """
        Given an expression with various whitespace,
        When I tokenize it,
        Then whitespace is properly skipped.
        """
        tokens = tokenize("  status   =   'pending'  ")

        assert len(tokens) == 4
        assert tokens[0].value == "status"
        assert tokens[1].value == "="
        assert tokens[2].value == "pending"


class TestComplexExpressions:
    """Scenario: Tokenize complex query expressions with multiple operators."""

    def test_tokenize_and_expression(self):
        """
        Given an AND expression,
        When I tokenize it,
        Then I get the AND keyword token.
        """
        tokens = tokenize("status = 'pending' AND priority = 'high'")

        # IDENTIFIER, EQ, STRING, AND, IDENTIFIER, EQ, STRING, EOF
        assert len(tokens) == 8
        assert tokens[3].type == TokenType.AND
        assert tokens[3].value.upper() == "AND"

    def test_tokenize_or_expression(self):
        """
        Given an OR expression,
        When I tokenize it,
        Then I get the OR keyword token.
        """
        tokens = tokenize("status = 'pending' OR status = 'completed'")

        assert any(t.type == TokenType.OR for t in tokens)

    def test_tokenize_not_expression(self):
        """
        Given a NOT expression,
        When I tokenize it,
        Then I get the NOT keyword token.
        """
        tokens = tokenize("NOT status = 'pending'")

        assert tokens[0].type == TokenType.NOT
        assert tokens[0].value.upper() == "NOT"


class TestFunctionCalls:
    """Scenario: Tokenize function call expressions."""

    def test_tokenize_simple_function_call(self):
        """
        Given a function call expression,
        When I tokenize it,
        Then I get function name, parentheses, and arguments.
        """
        tokens = tokenize("connected_to('T-123')")

        # IDENTIFIER, LPAREN, STRING, RPAREN, EOF
        assert len(tokens) == 5
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "connected_to"
        assert tokens[1].type == TokenType.LPAREN
        assert tokens[2].type == TokenType.STRING
        assert tokens[2].value == "T-123"
        assert tokens[3].type == TokenType.RPAREN

    def test_tokenize_function_with_multiple_args(self):
        """
        Given a function call with multiple arguments,
        When I tokenize it,
        Then I get commas separating arguments.
        """
        tokens = tokenize("range(1, 10)")

        # IDENTIFIER, LPAREN, NUMBER, COMMA, NUMBER, RPAREN, EOF
        assert len(tokens) == 7
        assert tokens[3].type == TokenType.COMMA
        assert tokens[3].value == ","


class TestEntityIdentifiers:
    """Scenario: Tokenize entity IDs like T-123."""

    def test_tokenize_entity_id_in_string(self):
        """
        Given an entity ID in a string,
        When I tokenize it,
        Then it's treated as a STRING token.
        """
        tokens = tokenize("'T-123'")

        assert len(tokens) == 2  # STRING, EOF
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == "T-123"

    def test_tokenize_entity_id_as_identifier(self):
        """
        Given an entity ID without quotes (as identifier),
        When I tokenize it,
        Then it's a single IDENTIFIER token, not T minus 123.
        """
        tokens = tokenize("T-123")

        assert len(tokens) == 2  # IDENTIFIER, EOF
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "T-123"


class TestStringHandling:
    """Scenario: Tokenize string literals with various content."""

    def test_tokenize_single_quoted_string(self):
        """
        Given a single-quoted string,
        When I tokenize it,
        Then I get a STRING token without quotes.
        """
        tokens = tokenize("'hello world'")

        assert len(tokens) == 2
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == "hello world"

    def test_tokenize_double_quoted_string(self):
        """
        Given a double-quoted string,
        When I tokenize it,
        Then I get a STRING token without quotes.
        """
        tokens = tokenize('"hello world"')

        assert len(tokens) == 2
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == "hello world"

    def test_tokenize_string_with_escaped_quote(self):
        """
        Given a string with escaped quotes,
        When I tokenize it,
        Then escapes are properly handled.
        """
        tokens = tokenize(r"'it\'s working'")

        assert len(tokens) == 2
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == "it's working"

    def test_tokenize_string_with_backslash(self):
        """
        Given a string with escaped backslash,
        When I tokenize it,
        Then backslash is properly handled.
        """
        tokens = tokenize(r"'path\\to\\file'")

        assert len(tokens) == 2
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == r"path\to\file"


class TestNumberHandling:
    """Scenario: Tokenize numeric literals."""

    def test_tokenize_integer(self):
        """
        Given an integer,
        When I tokenize it,
        Then I get a NUMBER token.
        """
        tokens = tokenize("42")

        assert len(tokens) == 2
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == "42"

    def test_tokenize_float(self):
        """
        Given a floating-point number,
        When I tokenize it,
        Then I get a NUMBER token.
        """
        tokens = tokenize("3.14159")

        assert len(tokens) == 2
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == "3.14159"

    def test_tokenize_negative_number(self):
        """
        Given a negative number with minus sign,
        When I tokenize it,
        Then I get an error since minus is not a supported operator.

        Note: The specification only includes comparison operators (=, !=, >, <, >=, <=),
        not arithmetic operators like minus. If needed, this would be a parser concern.
        """
        # Minus is not in the supported operator set, so it raises an error
        with pytest.raises(LexerError) as exc_info:
            tokenize("-5")

        error = exc_info.value
        assert "-" in error.message or "Unexpected" in error.message


class TestOperators:
    """Scenario: Tokenize comparison operators."""

    def test_tokenize_equality_operators(self):
        """
        Given equality operators,
        When I tokenize them,
        Then I get correct operator tokens.
        """
        test_cases = [
            ("=", TokenType.EQ),
            ("!=", TokenType.NE),
        ]

        for op, expected_type in test_cases:
            tokens = tokenize(f"x {op} y")
            assert any(t.type == expected_type for t in tokens), f"Failed for {op}"

    def test_tokenize_comparison_operators(self):
        """
        Given comparison operators,
        When I tokenize them,
        Then I get correct operator tokens.
        """
        test_cases = [
            (">", TokenType.GT),
            ("<", TokenType.LT),
            (">=", TokenType.GTE),
            ("<=", TokenType.LTE),
        ]

        for op, expected_type in test_cases:
            tokens = tokenize(f"x {op} y")
            assert any(t.type == expected_type for t in tokens), f"Failed for {op}"


class TestKeywords:
    """Scenario: Tokenize keywords case-insensitively."""

    def test_keywords_are_case_insensitive(self):
        """
        Given keywords in different cases,
        When I tokenize them,
        Then they're all recognized as the same keyword type.
        """
        test_cases = [
            ("AND", TokenType.AND),
            ("and", TokenType.AND),
            ("And", TokenType.AND),
            ("OR", TokenType.OR),
            ("or", TokenType.OR),
            ("NOT", TokenType.NOT),
            ("not", TokenType.NOT),
        ]

        for keyword, expected_type in test_cases:
            tokens = tokenize(keyword)
            assert tokens[0].type == expected_type, f"Failed for {keyword}"

    def test_tokenize_order_by_clause(self):
        """
        Given an ORDER BY clause,
        When I tokenize it,
        Then I get ORDER and BY keywords.
        """
        tokens = tokenize("ORDER BY priority ASC")

        # ORDER, BY, IDENTIFIER, ASC, EOF
        assert len(tokens) == 5
        assert tokens[0].type == TokenType.ORDER
        assert tokens[1].type == TokenType.BY
        assert tokens[3].type == TokenType.ASC

    def test_tokenize_limit_offset(self):
        """
        Given LIMIT and OFFSET keywords,
        When I tokenize them,
        Then I get correct keyword tokens.
        """
        tokens = tokenize("LIMIT 10 OFFSET 5")

        # LIMIT, NUMBER, OFFSET, NUMBER, EOF
        assert len(tokens) == 5
        assert tokens[0].type == TokenType.LIMIT
        assert tokens[2].type == TokenType.OFFSET


class TestPositionTracking:
    """Scenario: Track token positions for error reporting."""

    def test_tokens_have_position_information(self):
        """
        Given any expression,
        When I tokenize it,
        Then each token has position information.
        """
        tokens = tokenize("status = 'pending'")

        for token in tokens:
            assert hasattr(token, 'position')
            assert isinstance(token.position, int)
            assert token.position >= 0

    def test_position_increases_through_source(self):
        """
        Given a tokenized expression,
        When I examine token positions,
        Then positions increase (or stay same for EOF).
        """
        tokens = tokenize("a = b")

        positions = [t.position for t in tokens[:-1]]  # Exclude EOF
        for i in range(len(positions) - 1):
            assert positions[i] <= positions[i + 1]


class TestErrorHandling:
    """Scenario: Handle invalid input gracefully."""

    def test_invalid_character_raises_error(self):
        """
        Given a query with invalid character,
        When I tokenize it,
        Then I get a LexerError with position.
        """
        with pytest.raises(LexerError) as exc_info:
            tokenize("status @ 'pending'")

        error = exc_info.value
        assert error.position is not None
        assert "@ " in str(error) or "@" in error.message

    def test_unterminated_string_raises_error(self):
        """
        Given a string without closing quote,
        When I tokenize it,
        Then I get a LexerError.
        """
        with pytest.raises(LexerError) as exc_info:
            tokenize("'unterminated")

        error = exc_info.value
        assert "unterminated" in error.message.lower() or "quote" in error.message.lower()

    def test_error_includes_source_context(self):
        """
        Given an error during tokenization,
        When the error is raised,
        Then it includes source context for debugging.
        """
        with pytest.raises(LexerError) as exc_info:
            tokenize("status @ 'pending'")

        error = exc_info.value
        # Error should have source for context display
        assert error.source is not None or error.position is not None


class TestBrackets:
    """Scenario: Tokenize bracket expressions."""

    def test_tokenize_array_brackets(self):
        """
        Given an expression with array brackets,
        When I tokenize it,
        Then I get LBRACKET and RBRACKET tokens.
        """
        tokens = tokenize("['a', 'b']")

        assert tokens[0].type == TokenType.LBRACKET
        assert tokens[-2].type == TokenType.RBRACKET  # -1 is EOF

    def test_tokenize_in_expression_with_array(self):
        """
        Given an IN expression with array,
        When I tokenize it,
        Then I get IN keyword and brackets.
        """
        tokens = tokenize("status IN ['pending', 'active']")

        # IDENTIFIER, IN, LBRACKET, STRING, COMMA, STRING, RBRACKET, EOF
        assert len(tokens) == 8
        assert tokens[1].type == TokenType.IN
        assert tokens[2].type == TokenType.LBRACKET
        assert tokens[6].type == TokenType.RBRACKET


class TestIdentifiers:
    """Scenario: Tokenize various identifier formats."""

    def test_tokenize_simple_identifier(self):
        """
        Given a simple identifier,
        When I tokenize it,
        Then I get an IDENTIFIER token.
        """
        tokens = tokenize("status")

        assert len(tokens) == 2  # IDENTIFIER, EOF
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "status"

    def test_tokenize_identifier_with_underscore(self):
        """
        Given an identifier with underscores,
        When I tokenize it,
        Then underscores are included.
        """
        tokens = tokenize("created_at")

        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "created_at"

    def test_tokenize_identifier_with_numbers(self):
        """
        Given an identifier with numbers,
        When I tokenize it,
        Then numbers are included.
        """
        tokens = tokenize("field123")

        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "field123"

    def test_keyword_not_part_of_identifier(self):
        """
        Given an identifier containing a keyword,
        When I tokenize it,
        Then the keyword is not extracted separately.

        For example, "android" should be IDENTIFIER, not "AND" + "roid".
        """
        tokens = tokenize("android")

        assert len(tokens) == 2  # IDENTIFIER, EOF
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "android"
