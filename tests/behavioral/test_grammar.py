"""
Behavioral tests for the GoT query expression grammar.

These tests validate that the grammar definition is complete and accurate
by testing the parser against all valid and invalid example expressions.
"""

import pytest

from cortical.got.expression.grammar import (
    get_valid_examples,
    get_invalid_examples,
    get_grammar,
)
from cortical.got.expression.parser import Parser
from cortical.got.expression.errors import ParseError, QueryError


class TestGrammarDefinition:
    """Tests for the grammar definition itself."""

    def test_grammar_is_non_empty(self):
        """The grammar definition should contain the EBNF rules."""
        grammar = get_grammar()
        assert grammar, "Grammar should not be empty"
        assert "query" in grammar, "Grammar should define 'query' production"
        assert "expression" in grammar, "Grammar should define 'expression' production"

    def test_valid_examples_exist(self):
        """Valid examples dictionary should be populated."""
        examples = get_valid_examples()
        assert len(examples) > 0, "Should have valid examples"
        assert "simple_comparison" in examples, "Should include basic examples"

    def test_invalid_examples_exist(self):
        """Invalid examples dictionary should be populated."""
        examples = get_invalid_examples()
        assert len(examples) > 0, "Should have invalid examples"
        assert "missing_value" in examples, "Should include error cases"


class TestValidGrammarExamples:
    """
    Test that all valid grammar examples parse successfully.

    Each test validates one valid expression from the grammar definition.
    """

    @pytest.fixture
    def parser_factory(self):
        """Factory to create parser instances."""
        def create_parser(source: str) -> Parser:
            return Parser(source)
        return create_parser

    def test_valid_examples_are_parseable(self, parser_factory):
        """All valid examples should parse without error."""
        valid_examples = get_valid_examples()

        for name, expression in valid_examples.items():
            parser = parser_factory(expression)
            try:
                query = parser.parse()
                assert query is not None, f"Example '{name}' should produce a Query AST"
            except Exception as e:
                pytest.fail(f"Valid example '{name}' failed to parse: {expression}\nError: {e}")

    def test_simple_comparison(self, parser_factory):
        """Should parse: status = 'pending'"""
        parser = parser_factory("status = 'pending'")
        query = parser.parse()
        assert query is not None

    def test_and_expression(self, parser_factory):
        """Should parse: status = 'pending' AND priority = 'high'"""
        parser = parser_factory("status = 'pending' AND priority = 'high'")
        query = parser.parse()
        assert query is not None

    def test_or_expression(self, parser_factory):
        """Should parse: a = 1 OR b = 2"""
        parser = parser_factory("a = 1 OR b = 2")
        query = parser.parse()
        assert query is not None

    def test_mixed_precedence(self, parser_factory):
        """Should parse: a = 1 OR b = 2 AND c = 3 (AND binds tighter)"""
        parser = parser_factory("a = 1 OR b = 2 AND c = 3")
        query = parser.parse()
        assert query is not None

    def test_parenthesized(self, parser_factory):
        """Should parse: (a = 1 OR b = 2) AND c = 3"""
        parser = parser_factory("(a = 1 OR b = 2) AND c = 3")
        query = parser.parse()
        assert query is not None

    def test_not_expression(self, parser_factory):
        """Should parse: NOT status = 'completed'"""
        parser = parser_factory("NOT status = 'completed'")
        query = parser.parse()
        assert query is not None

    def test_in_operator(self, parser_factory):
        """Should parse: status IN ['pending', 'active']"""
        parser = parser_factory("status IN ['pending', 'active']")
        query = parser.parse()
        assert query is not None

    def test_not_in_operator(self, parser_factory):
        """Should parse: status NOT IN ['deleted']"""
        parser = parser_factory("status NOT IN ['deleted']")
        query = parser.parse()
        assert query is not None

    def test_like_operator(self, parser_factory):
        """Should parse: title LIKE '%bug%'"""
        parser = parser_factory("title LIKE '%bug%'")
        query = parser.parse()
        assert query is not None

    def test_function_call(self, parser_factory):
        """Should parse: connected_to(T-123)"""
        parser = parser_factory("connected_to(T-123)")
        query = parser.parse()
        assert query is not None

    def test_function_with_kwargs(self, parser_factory):
        """Should parse: path(T-1, T-2, max_depth=5)"""
        parser = parser_factory("path(T-1, T-2, max_depth=5)")
        query = parser.parse()
        assert query is not None

    def test_order_by(self, parser_factory):
        """Should parse: status = 'pending' ORDER BY created_at DESC"""
        parser = parser_factory("status = 'pending' ORDER BY created_at DESC")
        query = parser.parse()
        assert query is not None

    def test_limit_offset(self, parser_factory):
        """Should parse: category = 'bug' LIMIT 10 OFFSET 20"""
        parser = parser_factory("category = 'bug' LIMIT 10 OFFSET 20")
        query = parser.parse()
        assert query is not None

    def test_entity_id(self, parser_factory):
        """Should parse: id = T-123"""
        parser = parser_factory("id = T-123")
        query = parser.parse()
        assert query is not None

    def test_complex_query(self, parser_factory):
        """Should parse complex query with multiple clauses."""
        parser = parser_factory(
            "status IN ['pending', 'active'] AND NOT priority = 'low' "
            "ORDER BY created_at DESC LIMIT 50"
        )
        query = parser.parse()
        assert query is not None


class TestInvalidGrammarExamples:
    """
    Test that all invalid grammar examples raise ParseError.

    Each test validates that a malformed expression is properly rejected.
    """

    @pytest.fixture
    def parser_factory(self):
        """Factory to create parser instances."""
        def create_parser(source: str) -> Parser:
            return Parser(source)
        return create_parser

    def test_invalid_examples_raise_query_error(self, parser_factory):
        """All invalid examples should raise QueryError (ParseError or LexerError)."""
        invalid_examples = get_invalid_examples()

        for name, expression in invalid_examples.items():
            parser = parser_factory(expression)
            with pytest.raises(QueryError):
                parser.parse()

    def test_missing_value(self, parser_factory):
        """Should reject: status ="""
        parser = parser_factory("status =")
        with pytest.raises(ParseError):
            parser.parse()

    def test_unclosed_paren(self, parser_factory):
        """Should reject: (a = 1"""
        parser = parser_factory("(a = 1")
        with pytest.raises(ParseError):
            parser.parse()

    def test_invalid_operator(self, parser_factory):
        """Should reject: status == 'pending' (double equals not supported)"""
        parser = parser_factory("status == 'pending'")
        with pytest.raises(ParseError):
            parser.parse()

    def test_unclosed_string(self, parser_factory):
        """Should reject: status = 'pending (raises LexerError)"""
        parser = parser_factory("status = 'pending")
        with pytest.raises(QueryError):  # LexerError for unterminated strings
            parser.parse()

    def test_missing_operator(self, parser_factory):
        """Should reject: status 'pending'"""
        parser = parser_factory("status 'pending'")
        with pytest.raises(ParseError):
            parser.parse()

    def test_unclosed_list(self, parser_factory):
        """Should reject: status IN ['pending'"""
        parser = parser_factory("status IN ['pending'")
        with pytest.raises(ParseError):
            parser.parse()

    def test_missing_field(self, parser_factory):
        """Should reject: = 'value'"""
        parser = parser_factory("= 'value'")
        with pytest.raises(ParseError):
            parser.parse()

    def test_empty_expression(self, parser_factory):
        """Should reject: empty string"""
        parser = parser_factory("")
        with pytest.raises(ParseError):
            parser.parse()

    def test_only_whitespace(self, parser_factory):
        """Should reject: whitespace only"""
        parser = parser_factory("   ")
        with pytest.raises(ParseError):
            parser.parse()

    def test_only_keyword(self, parser_factory):
        """Should reject: AND (keyword without expression)"""
        parser = parser_factory("AND")
        with pytest.raises(ParseError):
            parser.parse()


class TestGrammarEdgeCases:
    """Test edge cases and corner cases in the grammar."""

    @pytest.fixture
    def parser_factory(self):
        """Factory to create parser instances."""
        def create_parser(source: str) -> Parser:
            return Parser(source)
        return create_parser

    def test_case_insensitive_keywords(self, parser_factory):
        """Keywords should be case-insensitive."""
        # Lowercase
        parser1 = parser_factory("status = 'pending' and priority = 'high'")
        query1 = parser1.parse()
        assert query1 is not None

        # Uppercase
        parser2 = parser_factory("status = 'pending' AND priority = 'high'")
        query2 = parser2.parse()
        assert query2 is not None

        # Mixed case
        parser3 = parser_factory("status = 'pending' AnD priority = 'high'")
        query3 = parser3.parse()
        assert query3 is not None

    def test_whitespace_variations(self, parser_factory):
        """Parser should handle different whitespace patterns."""
        expressions = [
            "status='pending'",  # No spaces
            "status = 'pending'",  # Normal spaces
            "status  =  'pending'",  # Extra spaces
            "  status = 'pending'  ",  # Leading/trailing
            "status\t=\t'pending'",  # Tabs
        ]

        for expr in expressions:
            parser = parser_factory(expr)
            query = parser.parse()
            assert query is not None, f"Should parse: {repr(expr)}"

    def test_nested_parentheses(self, parser_factory):
        """Should handle multiple levels of parentheses."""
        parser = parser_factory("((a = 1))")
        query = parser.parse()
        assert query is not None

    def test_empty_list(self, parser_factory):
        """Should parse empty list in IN operator."""
        parser = parser_factory("status IN []")
        query = parser.parse()
        assert query is not None

    def test_single_item_list(self, parser_factory):
        """Should parse list with single item."""
        parser = parser_factory("status IN ['pending']")
        query = parser.parse()
        assert query is not None

    def test_numeric_list(self, parser_factory):
        """Should parse list of numbers."""
        parser = parser_factory("priority IN [1, 2, 3]")
        query = parser.parse()
        assert query is not None

    def test_double_not(self, parser_factory):
        """Should parse double negation."""
        parser = parser_factory("NOT NOT status = 'pending'")
        query = parser.parse()
        assert query is not None

    def test_function_no_args(self, parser_factory):
        """Should parse function call with no arguments."""
        parser = parser_factory("orphan_nodes()")
        query = parser.parse()
        assert query is not None

    def test_function_kwargs_only(self, parser_factory):
        """Should parse function with only keyword arguments."""
        parser = parser_factory("path(from=T-1, to=T-2)")
        query = parser.parse()
        assert query is not None

    def test_hyphenated_identifier(self, parser_factory):
        """Should parse identifiers with hyphens (like T-123)."""
        parser = parser_factory("id = T-123-sub")
        query = parser.parse()
        assert query is not None

    def test_underscored_field(self, parser_factory):
        """Should parse field names with underscores."""
        parser = parser_factory("field_name = 'value'")
        query = parser.parse()
        assert query is not None

    def test_all_comparison_operators(self, parser_factory):
        """Should parse all comparison operators."""
        operators = ['=', '!=', '>', '<', '>=', '<=']

        for op in operators:
            parser = parser_factory(f"priority {op} 5")
            query = parser.parse()
            assert query is not None, f"Should parse operator: {op}"

    def test_string_escape_sequences(self, parser_factory):
        """Should handle escape sequences in strings."""
        # Newline
        parser1 = parser_factory(r"text = 'line1\nline2'")
        query1 = parser1.parse()
        assert query1 is not None

        # Tab
        parser2 = parser_factory(r"text = 'col1\tcol2'")
        query2 = parser2.parse()
        assert query2 is not None

        # Escaped quote
        parser3 = parser_factory(r"text = 'it\'s'")
        query3 = parser3.parse()
        assert query3 is not None
