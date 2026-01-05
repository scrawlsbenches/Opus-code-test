"""
Unit tests for the expression parser.

Tests all grammar rules, operator precedence, and error cases.
"""

import pytest
from cortical.got.expression.parser import Parser, parse
from cortical.got.expression.ast import (
    Query, Literal, Field, Comparison, AndExpr, OrExpr, NotExpr,
    FunctionCall, Op
)
from cortical.got.expression.errors import ParseError


class TestSimpleComparisons:
    """Test basic comparison expressions."""

    def test_string_equality(self):
        """status = 'pending'"""
        query = parse("status = 'pending'")
        assert isinstance(query, Query)
        assert isinstance(query.expression, Comparison)
        assert query.expression.field == Field(name='status')
        assert query.expression.op == Op.EQ
        assert query.expression.value == Literal(value='pending')

    def test_number_equality(self):
        """priority = 5"""
        query = parse("priority = 5")
        assert isinstance(query.expression, Comparison)
        assert query.expression.field == Field(name='priority')
        assert query.expression.op == Op.EQ
        assert query.expression.value == Literal(value=5)

    def test_float_comparison(self):
        """score > 3.14"""
        query = parse("score > 3.14")
        assert isinstance(query.expression, Comparison)
        assert query.expression.field == Field(name='score')
        assert query.expression.op == Op.GT
        assert query.expression.value == Literal(value=3.14)

    def test_identifier_value(self):
        """owner = john"""
        query = parse("owner = john")
        assert isinstance(query.expression, Comparison)
        assert query.expression.value == Literal(value='john')

    def test_not_equal(self):
        """status != 'completed'"""
        query = parse("status != 'completed'")
        assert query.expression.op == Op.NE

    def test_greater_than(self):
        """count > 10"""
        query = parse("count > 10")
        assert query.expression.op == Op.GT

    def test_less_than(self):
        """count < 100"""
        query = parse("count < 100")
        assert query.expression.op == Op.LT

    def test_greater_equal(self):
        """count >= 5"""
        query = parse("count >= 5")
        assert query.expression.op == Op.GTE

    def test_less_equal(self):
        """count <= 50"""
        query = parse("count <= 50")
        assert query.expression.op == Op.LTE


class TestListOperators:
    """Test IN and NOT IN operators."""

    def test_in_operator_strings(self):
        """status IN ['pending', 'active']"""
        query = parse("status IN ['pending', 'active']")
        assert isinstance(query.expression, Comparison)
        assert query.expression.op == Op.IN
        assert query.expression.value == Literal(value=['pending', 'active'])

    def test_in_operator_numbers(self):
        """priority IN [1, 2, 3]"""
        query = parse("priority IN [1, 2, 3]")
        assert query.expression.op == Op.IN
        assert query.expression.value == Literal(value=[1, 2, 3])

    def test_in_operator_mixed(self):
        """id IN [T-1, T-2, T-3]"""
        query = parse("id IN [T-1, T-2, T-3]")
        assert query.expression.value == Literal(value=['T-1', 'T-2', 'T-3'])

    def test_not_in_operator(self):
        """status NOT IN ['completed', 'cancelled']"""
        query = parse("status NOT IN ['completed', 'cancelled']")
        assert query.expression.op == Op.NOT_IN
        assert query.expression.value == Literal(value=['completed', 'cancelled'])

    def test_empty_list(self):
        """status IN []"""
        query = parse("status IN []")
        assert query.expression.value == Literal(value=[])


class TestLikeOperator:
    """Test LIKE and NOT LIKE operators."""

    def test_like_operator(self):
        """title LIKE '%bug%'"""
        query = parse("title LIKE '%bug%'")
        assert isinstance(query.expression, Comparison)
        assert query.expression.op == Op.LIKE
        assert query.expression.value == Literal(value='%bug%')

    def test_like_prefix(self):
        """name LIKE 'test%'"""
        query = parse("name LIKE 'test%'")
        assert query.expression.value == Literal(value='test%')

    def test_like_suffix(self):
        """name LIKE '%test'"""
        query = parse("name LIKE '%test'")
        assert query.expression.value == Literal(value='%test')

    def test_not_like_operator(self):
        """title NOT LIKE '%draft%'"""
        query = parse("title NOT LIKE '%draft%'")
        assert query.expression.op == Op.NOT_LIKE


class TestAndExpressions:
    """Test AND logical operator."""

    def test_simple_and(self):
        """status = 'pending' AND priority = 'high'"""
        query = parse("status = 'pending' AND priority = 'high'")
        assert isinstance(query.expression, AndExpr)
        assert len(query.expression.children) == 2

        left = query.expression.children[0]
        assert isinstance(left, Comparison)
        assert left.field == Field(name='status')

        right = query.expression.children[1]
        assert isinstance(right, Comparison)
        assert right.field == Field(name='priority')

    def test_three_way_and(self):
        """a = 1 AND b = 2 AND c = 3"""
        query = parse("a = 1 AND b = 2 AND c = 3")
        assert isinstance(query.expression, AndExpr)
        assert len(query.expression.children) == 3

    def test_case_insensitive_and(self):
        """a = 1 and b = 2"""
        query = parse("a = 1 and b = 2")
        assert isinstance(query.expression, AndExpr)


class TestOrExpressions:
    """Test OR logical operator."""

    def test_simple_or(self):
        """status = 'pending' OR status = 'active'"""
        query = parse("status = 'pending' OR status = 'active'")
        assert isinstance(query.expression, OrExpr)
        assert len(query.expression.children) == 2

    def test_three_way_or(self):
        """a = 1 OR b = 2 OR c = 3"""
        query = parse("a = 1 OR b = 2 OR c = 3")
        assert isinstance(query.expression, OrExpr)
        assert len(query.expression.children) == 3

    def test_case_insensitive_or(self):
        """a = 1 or b = 2"""
        query = parse("a = 1 or b = 2")
        assert isinstance(query.expression, OrExpr)


class TestNotExpressions:
    """Test NOT logical operator."""

    def test_simple_not(self):
        """NOT status = 'completed'"""
        query = parse("NOT status = 'completed'")
        assert isinstance(query.expression, NotExpr)
        assert isinstance(query.expression.child, Comparison)

    def test_double_not(self):
        """NOT NOT status = 'pending'"""
        query = parse("NOT NOT status = 'pending'")
        assert isinstance(query.expression, NotExpr)
        assert isinstance(query.expression.child, NotExpr)
        assert isinstance(query.expression.child.child, Comparison)

    def test_case_insensitive_not(self):
        """not status = 'completed'"""
        query = parse("not status = 'completed'")
        assert isinstance(query.expression, NotExpr)


class TestParentheses:
    """Test parenthesized expressions and operator precedence."""

    def test_simple_parens(self):
        """(status = 'pending')"""
        query = parse("(status = 'pending')")
        # Parens don't create a wrapper, just ensure precedence
        assert isinstance(query.expression, Comparison)

    def test_parens_with_or_and_and(self):
        """(a = 1 OR b = 2) AND c = 3"""
        query = parse("(a = 1 OR b = 2) AND c = 3")
        assert isinstance(query.expression, AndExpr)
        assert len(query.expression.children) == 2

        # First child should be the OR expression
        left = query.expression.children[0]
        assert isinstance(left, OrExpr)
        assert len(left.children) == 2

        # Second child is the c = 3 comparison
        right = query.expression.children[1]
        assert isinstance(right, Comparison)
        assert right.field == Field(name='c')

    def test_and_binds_tighter_than_or(self):
        """a = 1 OR b = 2 AND c = 3 (parsed as: a = 1 OR (b = 2 AND c = 3))"""
        query = parse("a = 1 OR b = 2 AND c = 3")
        assert isinstance(query.expression, OrExpr)
        assert len(query.expression.children) == 2

        # Left is a = 1
        left = query.expression.children[0]
        assert isinstance(left, Comparison)

        # Right is (b = 2 AND c = 3)
        right = query.expression.children[1]
        assert isinstance(right, AndExpr)

    def test_nested_parens(self):
        """((a = 1))"""
        query = parse("((a = 1))")
        assert isinstance(query.expression, Comparison)

    def test_complex_precedence(self):
        """(a = 1 OR b = 2) AND (c = 3 OR d = 4)"""
        query = parse("(a = 1 OR b = 2) AND (c = 3 OR d = 4)")
        assert isinstance(query.expression, AndExpr)
        assert isinstance(query.expression.children[0], OrExpr)
        assert isinstance(query.expression.children[1], OrExpr)


class TestFunctionCalls:
    """Test function call parsing."""

    def test_function_no_args(self):
        """all()"""
        query = parse("all()")
        assert isinstance(query.expression, FunctionCall)
        assert query.expression.name == "all"
        assert query.expression.args == ()
        assert query.expression.kwargs == ()

    def test_function_one_arg(self):
        """connected_to(T-123)"""
        query = parse("connected_to(T-123)")
        assert isinstance(query.expression, FunctionCall)
        assert query.expression.name == "connected_to"
        assert len(query.expression.args) == 1
        assert query.expression.args[0] == Literal(value='T-123')

    def test_function_multiple_args(self):
        """distance(T-1, T-2, 5)"""
        query = parse("distance(T-1, T-2, 5)")
        assert query.expression.name == "distance"
        assert len(query.expression.args) == 3
        assert query.expression.args[0] == Literal(value='T-1')
        assert query.expression.args[1] == Literal(value='T-2')
        assert query.expression.args[2] == Literal(value=5)

    def test_function_with_string_arg(self):
        """has_tag('critical')"""
        query = parse("has_tag('critical')")
        assert query.expression.name == "has_tag"
        assert query.expression.args[0] == Literal(value='critical')

    def test_function_with_kwargs(self):
        """search(query='bug', max_depth=3)"""
        query = parse("search(query='bug', max_depth=3)")
        assert query.expression.name == "search"
        assert len(query.expression.kwargs) == 2
        assert query.expression.kwargs[0] == ('query', Literal(value='bug'))
        assert query.expression.kwargs[1] == ('max_depth', Literal(value=3))

    def test_function_mixed_args(self):
        """func(T-1, depth=5, mode='fast')"""
        query = parse("func(T-1, depth=5, mode='fast')")
        assert len(query.expression.args) == 1
        assert query.expression.args[0] == Literal(value='T-1')
        assert len(query.expression.kwargs) == 2

    def test_function_in_boolean_expression(self):
        """connected_to(T-123) AND status = 'pending'"""
        query = parse("connected_to(T-123) AND status = 'pending'")
        assert isinstance(query.expression, AndExpr)
        assert isinstance(query.expression.children[0], FunctionCall)
        assert isinstance(query.expression.children[1], Comparison)


class TestOrderByClause:
    """Test ORDER BY clause parsing."""

    def test_order_by_default(self):
        """status = 'pending' ORDER BY created"""
        query = parse("status = 'pending' ORDER BY created")
        assert query.order_by == ('created', False)  # ASC is default

    def test_order_by_asc(self):
        """status = 'pending' ORDER BY created ASC"""
        query = parse("status = 'pending' ORDER BY created ASC")
        assert query.order_by == ('created', False)

    def test_order_by_desc(self):
        """status = 'pending' ORDER BY priority DESC"""
        query = parse("status = 'pending' ORDER BY priority DESC")
        assert query.order_by == ('priority', True)

    def test_case_insensitive_order(self):
        """status = 'pending' order by created desc"""
        query = parse("status = 'pending' order by created desc")
        assert query.order_by == ('created', True)


class TestLimitClause:
    """Test LIMIT and OFFSET clause parsing."""

    def test_limit_only(self):
        """status = 'pending' LIMIT 10"""
        query = parse("status = 'pending' LIMIT 10")
        assert query.limit == 10
        assert query.offset is None

    def test_limit_with_offset(self):
        """status = 'pending' LIMIT 10 OFFSET 20"""
        query = parse("status = 'pending' LIMIT 10 OFFSET 20")
        assert query.limit == 10
        assert query.offset == 20

    def test_case_insensitive_limit(self):
        """status = 'pending' limit 5 offset 10"""
        query = parse("status = 'pending' limit 5 offset 10")
        assert query.limit == 5
        assert query.offset == 10


class TestComplexQueries:
    """Test complex queries combining multiple features."""

    def test_all_features(self):
        """(status = 'pending' OR status = 'active') AND priority > 3 ORDER BY created DESC LIMIT 10"""
        query = parse(
            "(status = 'pending' OR status = 'active') AND priority > 3 "
            "ORDER BY created DESC LIMIT 10"
        )
        assert isinstance(query.expression, AndExpr)
        assert query.order_by == ('created', True)
        assert query.limit == 10

    def test_function_with_clauses(self):
        """connected_to(T-123) ORDER BY distance LIMIT 5"""
        query = parse("connected_to(T-123) ORDER BY distance LIMIT 5")
        assert isinstance(query.expression, FunctionCall)
        assert query.order_by == ('distance', False)
        assert query.limit == 5

    def test_not_with_parens(self):
        """NOT (status = 'completed' OR status = 'cancelled')"""
        query = parse("NOT (status = 'completed' OR status = 'cancelled')")
        assert isinstance(query.expression, NotExpr)
        assert isinstance(query.expression.child, OrExpr)

    def test_deeply_nested(self):
        """(a = 1 AND (b = 2 OR (c = 3 AND d = 4)))"""
        query = parse("(a = 1 AND (b = 2 OR (c = 3 AND d = 4)))")
        assert isinstance(query.expression, AndExpr)

    def test_multiple_functions(self):
        """connected_to(T-1) AND has_tag('bug')"""
        query = parse("connected_to(T-1) AND has_tag('bug')")
        assert isinstance(query.expression, AndExpr)
        assert isinstance(query.expression.children[0], FunctionCall)
        assert isinstance(query.expression.children[1], FunctionCall)


class TestEntityIdentifiers:
    """Test parsing of entity IDs like T-123, D-456."""

    def test_task_id(self):
        """id = T-123"""
        query = parse("id = T-123")
        assert query.expression.value == Literal(value='T-123')

    def test_decision_id(self):
        """decision_id = D-456"""
        query = parse("decision_id = D-456")
        assert query.expression.value == Literal(value='D-456')

    def test_id_in_list(self):
        """id IN [T-1, T-2, T-3]"""
        query = parse("id IN [T-1, T-2, T-3]")
        assert query.expression.value == Literal(value=['T-1', 'T-2', 'T-3'])

    def test_id_in_function(self):
        """connected_to(T-123)"""
        query = parse("connected_to(T-123)")
        assert query.expression.args[0] == Literal(value='T-123')


class TestErrorHandling:
    """Test error cases with good error messages."""

    def test_missing_closing_paren(self):
        """(status = 'pending'"""
        with pytest.raises(ParseError) as exc_info:
            parse("(status = 'pending'")
        assert "Expected ')' after expression" in str(exc_info.value)

    def test_missing_operator(self):
        """status 'pending'"""
        with pytest.raises(ParseError) as exc_info:
            parse("status 'pending'")
        assert "Expected comparison operator" in str(exc_info.value)

    def test_missing_value(self):
        """status ="""
        with pytest.raises(ParseError) as exc_info:
            parse("status =")
        assert "Expected value" in str(exc_info.value)

    def test_unexpected_token_after_query(self):
        """status = 'pending' extra"""
        with pytest.raises(ParseError) as exc_info:
            parse("status = 'pending' extra")
        assert "Expected end of query" in str(exc_info.value)

    def test_invalid_not_usage(self):
        """NOT AND"""
        with pytest.raises(ParseError) as exc_info:
            parse("NOT AND")
        # Will fail trying to parse primary after NOT

    def test_missing_by_after_order(self):
        """status = 'pending' ORDER created"""
        with pytest.raises(ParseError) as exc_info:
            parse("status = 'pending' ORDER created")
        assert "Expected BY after ORDER" in str(exc_info.value)

    def test_missing_number_after_limit(self):
        """status = 'pending' LIMIT"""
        with pytest.raises(ParseError) as exc_info:
            parse("status = 'pending' LIMIT")
        assert "Expected number after LIMIT" in str(exc_info.value)

    def test_missing_number_after_offset(self):
        """status = 'pending' LIMIT 10 OFFSET"""
        with pytest.raises(ParseError) as exc_info:
            parse("status = 'pending' LIMIT 10 OFFSET")
        assert "Expected number after OFFSET" in str(exc_info.value)

    def test_positional_after_keyword_arg(self):
        """func(a=1, b)"""
        with pytest.raises(ParseError) as exc_info:
            parse("func(a=1, b)")
        assert "Positional argument after keyword argument" in str(exc_info.value)

    def test_unclosed_list(self):
        """status IN ['a', 'b'"""
        with pytest.raises(ParseError) as exc_info:
            parse("status IN ['a', 'b'")
        assert "Expected ']' after list values" in str(exc_info.value)

    def test_not_without_in_or_like(self):
        """status NOT 'pending'"""
        with pytest.raises(ParseError) as exc_info:
            parse("status NOT 'pending'")
        assert "Expected IN or LIKE after NOT" in str(exc_info.value)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_query_fails(self):
        """Empty input should fail"""
        with pytest.raises(ParseError):
            parse("")

    def test_whitespace_handling(self):
        """   status   =   'pending'   """
        query = parse("   status   =   'pending'   ")
        assert isinstance(query.expression, Comparison)

    def test_single_identifier_fails(self):
        """status"""
        with pytest.raises(ParseError):
            parse("status")

    def test_operator_alone_fails(self):
        """="""
        with pytest.raises(ParseError):
            parse("=")

    def test_underscore_in_field_name(self):
        """created_at = 123"""
        query = parse("created_at = 123")
        assert query.expression.field == Field(name='created_at')

    def test_hyphen_in_identifier(self):
        """entity-id = test-123"""
        query = parse("entity-id = test-123")
        assert query.expression.field == Field(name='entity-id')
        assert query.expression.value == Literal(value='test-123')

    def test_number_with_leading_zeros(self):
        """count = 007"""
        query = parse("count = 007")
        # Leading zeros are kept in tokenization, then converted to int
        # int('007') == 7
        assert query.expression.value == Literal(value=7)

    def test_very_long_query(self):
        """Test that we can parse a very long query"""
        parts = [f"field{i} = {i}" for i in range(100)]
        query_str = " AND ".join(parts)
        query = parse(query_str)
        assert isinstance(query.expression, AndExpr)
        assert len(query.expression.children) == 100


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_parse_function(self):
        """Test the module-level parse() function"""
        query = parse("status = 'pending'")
        assert isinstance(query, Query)

    def test_parser_class_direct(self):
        """Test using Parser class directly"""
        parser = Parser("status = 'pending'")
        query = parser.parse()
        assert isinstance(query, Query)


class TestOperatorPrecedence:
    """Detailed tests for operator precedence."""

    def test_and_before_or(self):
        """a = 1 OR b = 2 AND c = 3 -> a = 1 OR (b = 2 AND c = 3)"""
        query = parse("a = 1 OR b = 2 AND c = 3")
        # Should be parsed as: a = 1 OR (b = 2 AND c = 3)
        assert isinstance(query.expression, OrExpr)
        assert isinstance(query.expression.children[1], AndExpr)

    def test_not_binds_tightest(self):
        """NOT a = 1 AND b = 2 -> (NOT a = 1) AND b = 2"""
        query = parse("NOT a = 1 AND b = 2")
        assert isinstance(query.expression, AndExpr)
        assert isinstance(query.expression.children[0], NotExpr)

    def test_parens_override_precedence(self):
        """a = 1 AND (b = 2 OR c = 3) -> a = 1 AND (b = 2 OR c = 3)"""
        query = parse("a = 1 AND (b = 2 OR c = 3)")
        assert isinstance(query.expression, AndExpr)
        assert isinstance(query.expression.children[1], OrExpr)
