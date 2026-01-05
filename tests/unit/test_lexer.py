"""
Unit tests for the Lexer module.

Tests individual methods and edge cases for the lexer implementation.
"""

import pytest
from cortical.got.expression.lexer import Lexer, Token, TokenType, tokenize
from cortical.got.expression.errors import LexerError


class TestLexerInitialization:
    """Test Lexer initialization."""

    def test_lexer_initialization(self):
        """Lexer should initialize with source and position."""
        lexer = Lexer("test")
        assert lexer.source == "test"
        assert lexer.position == 0
        assert lexer.length == 4

    def test_lexer_empty_source(self):
        """Lexer should handle empty source."""
        lexer = Lexer("")
        assert lexer.source == ""
        assert lexer.position == 0
        assert lexer.length == 0


class TestPeekMethod:
    """Test the _peek method."""

    def test_peek_at_current_position(self):
        """Peek should return current character without advancing."""
        lexer = Lexer("abc")
        assert lexer._peek() == "a"
        assert lexer.position == 0  # Position unchanged

    def test_peek_with_offset(self):
        """Peek should return character at offset."""
        lexer = Lexer("abc")
        assert lexer._peek(0) == "a"
        assert lexer._peek(1) == "b"
        assert lexer._peek(2) == "c"

    def test_peek_beyond_end(self):
        """Peek beyond end should return None."""
        lexer = Lexer("a")
        assert lexer._peek(1) is None
        assert lexer._peek(10) is None


class TestAdvanceMethod:
    """Test the _advance method."""

    def test_advance_returns_and_consumes(self):
        """Advance should return character and move position."""
        lexer = Lexer("abc")
        assert lexer._advance() == "a"
        assert lexer.position == 1
        assert lexer._advance() == "b"
        assert lexer.position == 2

    def test_advance_at_end_raises(self):
        """Advance at end should raise IndexError."""
        lexer = Lexer("a")
        lexer._advance()
        with pytest.raises(IndexError):
            lexer._advance()


class TestSkipWhitespace:
    """Test the _skip_whitespace method."""

    def test_skip_spaces(self):
        """Skip whitespace should skip spaces."""
        lexer = Lexer("   abc")
        lexer._skip_whitespace()
        assert lexer.position == 3
        assert lexer._peek() == "a"

    def test_skip_tabs_and_newlines(self):
        """Skip whitespace should skip tabs and newlines."""
        lexer = Lexer("\t\n\r abc")
        lexer._skip_whitespace()
        assert lexer._peek() == "a"

    def test_no_whitespace_to_skip(self):
        """Skip whitespace should do nothing if no whitespace."""
        lexer = Lexer("abc")
        lexer._skip_whitespace()
        assert lexer.position == 0


class TestTokenizeMethod:
    """Test the tokenize method."""

    def test_tokenize_returns_iterator(self):
        """Tokenize should return an iterator."""
        lexer = Lexer("a")
        result = lexer.tokenize()
        assert hasattr(result, '__iter__')
        assert hasattr(result, '__next__')

    def test_tokenize_always_ends_with_eof(self):
        """Tokenize should always end with EOF token."""
        tokens = list(Lexer("").tokenize())
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF

        tokens = list(Lexer("abc").tokenize())
        assert tokens[-1].type == TokenType.EOF

    def test_tokenize_skips_leading_whitespace(self):
        """Tokenize should skip leading whitespace."""
        tokens = list(Lexer("   abc").tokenize())
        # Should have at least identifier and EOF
        assert len(tokens) >= 2


class TestStringTokenization:
    """Test string literal tokenization."""

    def test_single_quoted_string(self):
        """Should tokenize single-quoted strings."""
        tokens = tokenize("'hello'")
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == "hello"

    def test_double_quoted_string(self):
        """Should tokenize double-quoted strings."""
        tokens = tokenize('"hello"')
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == "hello"

    def test_empty_string(self):
        """Should tokenize empty strings."""
        tokens = tokenize("''")
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == ""

    def test_string_with_spaces(self):
        """Should preserve spaces in strings."""
        tokens = tokenize("'hello world'")
        assert tokens[0].value == "hello world"

    def test_escaped_single_quote(self):
        """Should handle escaped single quotes."""
        tokens = tokenize(r"'it\'s'")
        assert tokens[0].value == "it's"

    def test_escaped_double_quote(self):
        """Should handle escaped double quotes."""
        # Need to properly escape the backslashes in the test string
        tokens = tokenize('"say \\"hi\\""')
        assert tokens[0].value == 'say "hi"'

    def test_escaped_backslash(self):
        """Should handle escaped backslashes."""
        tokens = tokenize(r"'a\\b'")
        assert tokens[0].value == r"a\b"

    def test_unterminated_single_quote_error(self):
        """Should raise error for unterminated single-quoted string."""
        with pytest.raises(LexerError):
            tokenize("'unterminated")

    def test_unterminated_double_quote_error(self):
        """Should raise error for unterminated double-quoted string."""
        with pytest.raises(LexerError):
            tokenize('"unterminated')


class TestNumberTokenization:
    """Test number tokenization."""

    def test_integer(self):
        """Should tokenize integers."""
        tokens = tokenize("123")
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == "123"

    def test_zero(self):
        """Should tokenize zero."""
        tokens = tokenize("0")
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == "0"

    def test_float_with_decimal(self):
        """Should tokenize floating-point numbers."""
        tokens = tokenize("3.14")
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == "3.14"

    def test_float_starting_with_zero(self):
        """Should tokenize floats starting with zero."""
        tokens = tokenize("0.5")
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == "0.5"


class TestIdentifierTokenization:
    """Test identifier tokenization."""

    def test_simple_identifier(self):
        """Should tokenize simple identifiers."""
        tokens = tokenize("status")
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "status"

    def test_identifier_with_underscore(self):
        """Should tokenize identifiers with underscores."""
        tokens = tokenize("created_at")
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "created_at"

    def test_identifier_with_numbers(self):
        """Should tokenize identifiers with numbers."""
        tokens = tokenize("field123")
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "field123"

    def test_identifier_starting_with_underscore(self):
        """Should tokenize identifiers starting with underscore."""
        tokens = tokenize("_private")
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "_private"

    def test_entity_id_as_identifier(self):
        """Should tokenize entity IDs (T-123) as single identifier."""
        tokens = tokenize("T-123")
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "T-123"

    def test_complex_entity_id(self):
        """Should tokenize complex entity IDs."""
        test_cases = ["T-001", "KT-42", "D-999", "TASK-1234"]
        for entity_id in test_cases:
            tokens = tokenize(entity_id)
            assert tokens[0].type == TokenType.IDENTIFIER
            assert tokens[0].value == entity_id


class TestKeywordTokenization:
    """Test keyword tokenization."""

    def test_and_keyword(self):
        """Should recognize AND keyword."""
        for variant in ["AND", "and", "And"]:
            tokens = tokenize(variant)
            assert tokens[0].type == TokenType.AND

    def test_or_keyword(self):
        """Should recognize OR keyword."""
        for variant in ["OR", "or", "Or"]:
            tokens = tokenize(variant)
            assert tokens[0].type == TokenType.OR

    def test_not_keyword(self):
        """Should recognize NOT keyword."""
        for variant in ["NOT", "not", "Not"]:
            tokens = tokenize(variant)
            assert tokens[0].type == TokenType.NOT

    def test_in_keyword(self):
        """Should recognize IN keyword."""
        tokens = tokenize("IN")
        assert tokens[0].type == TokenType.IN

    def test_like_keyword(self):
        """Should recognize LIKE keyword."""
        tokens = tokenize("LIKE")
        assert tokens[0].type == TokenType.LIKE

    def test_order_keyword(self):
        """Should recognize ORDER keyword."""
        tokens = tokenize("ORDER")
        assert tokens[0].type == TokenType.ORDER

    def test_by_keyword(self):
        """Should recognize BY keyword."""
        tokens = tokenize("BY")
        assert tokens[0].type == TokenType.BY

    def test_asc_keyword(self):
        """Should recognize ASC keyword."""
        tokens = tokenize("ASC")
        assert tokens[0].type == TokenType.ASC

    def test_desc_keyword(self):
        """Should recognize DESC keyword."""
        tokens = tokenize("DESC")
        assert tokens[0].type == TokenType.DESC

    def test_limit_keyword(self):
        """Should recognize LIMIT keyword."""
        tokens = tokenize("LIMIT")
        assert tokens[0].type == TokenType.LIMIT

    def test_offset_keyword(self):
        """Should recognize OFFSET keyword."""
        tokens = tokenize("OFFSET")
        assert tokens[0].type == TokenType.OFFSET

    def test_keyword_not_part_of_identifier(self):
        """Keywords should not be extracted from middle of identifiers."""
        # "android" contains "and" but should be identifier
        tokens = tokenize("android")
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "android"


class TestOperatorTokenization:
    """Test operator tokenization."""

    def test_equals(self):
        """Should tokenize = operator."""
        tokens = tokenize("=")
        assert tokens[0].type == TokenType.EQ
        assert tokens[0].value == "="

    def test_not_equals(self):
        """Should tokenize != operator."""
        tokens = tokenize("!=")
        assert tokens[0].type == TokenType.NE
        assert tokens[0].value == "!="

    def test_greater_than(self):
        """Should tokenize > operator."""
        tokens = tokenize(">")
        assert tokens[0].type == TokenType.GT
        assert tokens[0].value == ">"

    def test_less_than(self):
        """Should tokenize < operator."""
        tokens = tokenize("<")
        assert tokens[0].type == TokenType.LT
        assert tokens[0].value == "<"

    def test_greater_than_or_equal(self):
        """Should tokenize >= operator."""
        tokens = tokenize(">=")
        assert tokens[0].type == TokenType.GTE
        assert tokens[0].value == ">="

    def test_less_than_or_equal(self):
        """Should tokenize <= operator."""
        tokens = tokenize("<=")
        assert tokens[0].type == TokenType.LTE
        assert tokens[0].value == "<="


class TestPunctuationTokenization:
    """Test punctuation tokenization."""

    def test_left_paren(self):
        """Should tokenize (."""
        tokens = tokenize("(")
        assert tokens[0].type == TokenType.LPAREN

    def test_right_paren(self):
        """Should tokenize )."""
        tokens = tokenize(")")
        assert tokens[0].type == TokenType.RPAREN

    def test_left_bracket(self):
        """Should tokenize [."""
        tokens = tokenize("[")
        assert tokens[0].type == TokenType.LBRACKET

    def test_right_bracket(self):
        """Should tokenize ]."""
        tokens = tokenize("]")
        assert tokens[0].type == TokenType.RBRACKET

    def test_comma(self):
        """Should tokenize ,."""
        tokens = tokenize(",")
        assert tokens[0].type == TokenType.COMMA


class TestPositionTracking:
    """Test position tracking in tokens."""

    def test_first_token_position_zero(self):
        """First token should have position 0."""
        tokens = tokenize("abc")
        assert tokens[0].position == 0

    def test_token_positions_increase(self):
        """Token positions should increase through source."""
        tokens = tokenize("a b c")
        positions = [t.position for t in tokens[:-1]]  # Exclude EOF
        for i in range(len(positions) - 1):
            assert positions[i] < positions[i + 1]

    def test_position_after_whitespace(self):
        """Position should account for skipped whitespace."""
        tokens = tokenize("   abc")
        # First token should be at position 3 (after 3 spaces)
        assert tokens[0].position == 3

    def test_string_position(self):
        """String token position should be at opening quote."""
        lexer = Lexer("  'hello'")
        tokens = list(lexer.tokenize())
        # Position should be where the quote starts (after 2 spaces)
        assert tokens[0].position == 2


class TestErrorMessages:
    """Test error message quality."""

    def test_error_includes_position(self):
        """LexerError should include position."""
        with pytest.raises(LexerError) as exc_info:
            tokenize("@")
        error = exc_info.value
        assert error.position is not None

    def test_error_includes_source(self):
        """LexerError should include source for context."""
        with pytest.raises(LexerError) as exc_info:
            tokenize("status @ pending")
        error = exc_info.value
        assert error.source is not None

    def test_error_message_is_informative(self):
        """Error messages should be informative."""
        with pytest.raises(LexerError) as exc_info:
            tokenize("'unterminated")
        error = exc_info.value
        # Message should mention the problem
        assert "unterminated" in str(error).lower() or "quote" in str(error).lower()


class TestTokenConvenienceFunction:
    """Test the tokenize convenience function."""

    def test_tokenize_function_returns_list(self):
        """tokenize() should return a list of tokens."""
        result = tokenize("a")
        assert isinstance(result, list)
        assert all(isinstance(t, Token) for t in result)

    def test_tokenize_empty_string(self):
        """tokenize('') should return only EOF."""
        tokens = tokenize("")
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_character_tokens(self):
        """Should handle single character tokens."""
        tokens = tokenize("()")
        assert len(tokens) == 3  # LPAREN, RPAREN, EOF
        assert tokens[0].type == TokenType.LPAREN
        assert tokens[1].type == TokenType.RPAREN

    def test_no_whitespace_between_tokens(self):
        """Should tokenize without whitespace."""
        tokens = tokenize("a=b")
        assert len(tokens) == 4  # IDENTIFIER, EQ, IDENTIFIER, EOF

    def test_multiple_operators_in_sequence(self):
        """Should handle multiple operators."""
        tokens = tokenize("!=")
        assert len(tokens) == 2  # NE, EOF
        assert tokens[0].type == TokenType.NE

    def test_very_long_identifier(self):
        """Should handle very long identifiers."""
        long_id = "a" * 1000
        tokens = tokenize(long_id)
        assert tokens[0].type == TokenType.IDENTIFIER
        assert len(tokens[0].value) == 1000

    def test_very_long_string(self):
        """Should handle very long strings."""
        long_str = "a" * 1000
        tokens = tokenize(f"'{long_str}'")
        assert tokens[0].type == TokenType.STRING
        assert len(tokens[0].value) == 1000

    def test_unicode_in_string(self):
        """Should handle unicode in strings."""
        tokens = tokenize("'hello 世界'")
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == "hello 世界"
