"""
Operator Precedence Tests for GoT Query Parser
==============================================

Tests operator precedence handling in the expression parser.

Precedence Rules (highest to lowest):
1. Parentheses () - highest, explicit grouping
2. NOT - unary, binds tightly to its operand
3. Comparison operators (=, !=, >, <, >=, <=, IN, LIKE)
4. AND - binds tighter than OR
5. OR - lowest precedence

The grammar naturally encodes precedence through its structure:
    expression  ::= and_expr ('OR' and_expr)*       # OR - lowest
    and_expr    ::= not_expr ('AND' not_expr)*      # AND - higher
    not_expr    ::= 'NOT' not_expr | primary        # NOT - higher still
    primary     ::= comparison | '(' expression ')' # comparisons and parens - highest
"""

import pytest

from cortical.got.expression.parser import parse
from cortical.got.expression.ast import (
    Query, AndExpr, OrExpr, NotExpr, Comparison, Field, Literal, Op
)


# All tests validate the parser's correct implementation of operator precedence


class TestANDORPrecedence:
    """Test that AND binds tighter than OR."""

    def test_and_binds_tighter_than_or(self):
        """
        a = 1 OR b = 2 AND c = 3
        Should parse as: a = 1 OR (b = 2 AND c = 3)

        Expected AST:
        OrExpr([
            Comparison(a = 1),
            AndExpr([
                Comparison(b = 2),
                Comparison(c = 3)
            ])
        ])
        """
        query = parse("a = 1 OR b = 2 AND c = 3")

        # Top level should be OR
        assert isinstance(query.expression, OrExpr)
        assert len(query.expression.children) == 2

        # First child is comparison a = 1
        first = query.expression.children[0]
        assert isinstance(first, Comparison)
        assert first.field.name == "a"
        assert first.op == Op.EQ
        assert first.value.value == 1

        # Second child is AND expression
        second = query.expression.children[1]
        assert isinstance(second, AndExpr)
        assert len(second.children) == 2

        # Verify AND children
        assert isinstance(second.children[0], Comparison)
        assert second.children[0].field.name == "b"
        assert isinstance(second.children[1], Comparison)
        assert second.children[1].field.name == "c"

    def test_multiple_ands_with_or(self):
        """
        a = 1 AND b = 2 OR c = 3 AND d = 4
        Should parse as: (a = 1 AND b = 2) OR (c = 3 AND d = 4)
        """
        query = parse("a = 1 AND b = 2 OR c = 3 AND d = 4")

        assert isinstance(query.expression, OrExpr)
        assert len(query.expression.children) == 2

        # Both children should be AND expressions
        assert isinstance(query.expression.children[0], AndExpr)
        assert isinstance(query.expression.children[1], AndExpr)

        # Verify first AND: a = 1 AND b = 2
        first_and = query.expression.children[0]
        assert len(first_and.children) == 2
        assert first_and.children[0].field.name == "a"
        assert first_and.children[1].field.name == "b"

        # Verify second AND: c = 3 AND d = 4
        second_and = query.expression.children[1]
        assert len(second_and.children) == 2
        assert second_and.children[0].field.name == "c"
        assert second_and.children[1].field.name == "d"

    def test_three_way_or_with_middle_and(self):
        """
        a = 1 OR b = 2 AND c = 3 OR d = 4
        Should parse as: a = 1 OR (b = 2 AND c = 3) OR d = 4
        """
        query = parse("a = 1 OR b = 2 AND c = 3 OR d = 4")

        assert isinstance(query.expression, OrExpr)
        assert len(query.expression.children) == 3

        # First and third are comparisons
        assert isinstance(query.expression.children[0], Comparison)
        assert query.expression.children[0].field.name == "a"

        assert isinstance(query.expression.children[2], Comparison)
        assert query.expression.children[2].field.name == "d"

        # Middle is AND
        assert isinstance(query.expression.children[1], AndExpr)
        assert len(query.expression.children[1].children) == 2


class TestNOTPrecedence:
    """Test that NOT binds tightly to its immediate operand."""

    def test_not_binds_to_single_expression(self):
        """
        NOT a = 1 AND b = 2
        Should parse as: (NOT a = 1) AND b = 2

        Expected AST:
        AndExpr([
            NotExpr(Comparison(a = 1)),
            Comparison(b = 2)
        ])
        """
        query = parse("NOT a = 1 AND b = 2")

        # Top level should be AND
        assert isinstance(query.expression, AndExpr)
        assert len(query.expression.children) == 2

        # First child is NOT expression
        first = query.expression.children[0]
        assert isinstance(first, NotExpr)
        assert isinstance(first.child, Comparison)
        assert first.child.field.name == "a"

        # Second child is plain comparison
        second = query.expression.children[1]
        assert isinstance(second, Comparison)
        assert second.field.name == "b"

    def test_not_with_or(self):
        """
        NOT a = 1 OR b = 2
        Should parse as: (NOT a = 1) OR b = 2
        """
        query = parse("NOT a = 1 OR b = 2")

        assert isinstance(query.expression, OrExpr)
        assert len(query.expression.children) == 2

        # First child is NOT
        assert isinstance(query.expression.children[0], NotExpr)
        assert isinstance(query.expression.children[0].child, Comparison)

        # Second child is comparison
        assert isinstance(query.expression.children[1], Comparison)

    def test_multiple_nots_with_and(self):
        """
        NOT a = 1 AND NOT b = 2
        Should parse as: (NOT a = 1) AND (NOT b = 2)
        """
        query = parse("NOT a = 1 AND NOT b = 2")

        assert isinstance(query.expression, AndExpr)
        assert len(query.expression.children) == 2

        # Both children should be NOT expressions
        assert isinstance(query.expression.children[0], NotExpr)
        assert isinstance(query.expression.children[1], NotExpr)

    def test_double_not(self):
        """
        NOT NOT a = 1
        Should parse as: NotExpr(NotExpr(Comparison(a = 1)))
        """
        query = parse("NOT NOT a = 1")

        # Outer NOT
        assert isinstance(query.expression, NotExpr)

        # Inner NOT
        assert isinstance(query.expression.child, NotExpr)

        # Innermost comparison
        assert isinstance(query.expression.child.child, Comparison)
        assert query.expression.child.child.field.name == "a"

    def test_triple_not(self):
        """
        NOT NOT NOT a = 1
        Should parse as: NotExpr(NotExpr(NotExpr(Comparison(a = 1))))
        """
        query = parse("NOT NOT NOT a = 1")

        assert isinstance(query.expression, NotExpr)
        assert isinstance(query.expression.child, NotExpr)
        assert isinstance(query.expression.child.child, NotExpr)
        assert isinstance(query.expression.child.child.child, Comparison)


class TestParenthesesOverride:
    """Test that parentheses override default precedence."""

    def test_parens_override_and_precedence(self):
        """
        (a = 1 OR b = 2) AND c = 3
        Parens force OR to be evaluated first despite lower precedence

        Expected AST:
        AndExpr([
            OrExpr([Comparison(a = 1), Comparison(b = 2)]),
            Comparison(c = 3)
        ])
        """
        query = parse("(a = 1 OR b = 2) AND c = 3")

        # Top level should be AND
        assert isinstance(query.expression, AndExpr)
        assert len(query.expression.children) == 2

        # First child is OR expression (due to parens)
        first = query.expression.children[0]
        assert isinstance(first, OrExpr)
        assert len(first.children) == 2

        # Second child is comparison
        second = query.expression.children[1]
        assert isinstance(second, Comparison)
        assert second.field.name == "c"

    def test_parens_with_not(self):
        """
        NOT (a = 1 OR b = 2)
        NOT applies to entire parenthesized expression
        """
        query = parse("NOT (a = 1 OR b = 2)")

        # Top level is NOT
        assert isinstance(query.expression, NotExpr)

        # Child is OR expression
        assert isinstance(query.expression.child, OrExpr)
        assert len(query.expression.child.children) == 2

    def test_nested_parens(self):
        """
        ((a = 1 OR b = 2) AND c = 3) OR d = 4
        Nested parentheses create nested groupings
        """
        query = parse("((a = 1 OR b = 2) AND c = 3) OR d = 4")

        # Top level is OR
        assert isinstance(query.expression, OrExpr)
        assert len(query.expression.children) == 2

        # First child is AND (from outer parens)
        first = query.expression.children[0]
        assert isinstance(first, AndExpr)

        # AND's first child is OR (from inner parens)
        assert isinstance(first.children[0], OrExpr)

    def test_multiple_paren_groups(self):
        """
        (a = 1 OR b = 2) AND (c = 3 OR d = 4)
        Multiple independent parenthesized groups
        """
        query = parse("(a = 1 OR b = 2) AND (c = 3 OR d = 4)")

        # Top level is AND
        assert isinstance(query.expression, AndExpr)
        assert len(query.expression.children) == 2

        # Both children are OR expressions
        assert isinstance(query.expression.children[0], OrExpr)
        assert isinstance(query.expression.children[1], OrExpr)


class TestComplexNesting:
    """Test complex combinations of operators."""

    def test_complex_nesting_with_all_operators(self):
        """
        a = 1 AND (b = 2 OR c = 3) AND NOT d = 4

        Expected AST:
        AndExpr([
            Comparison(a = 1),
            OrExpr([Comparison(b = 2), Comparison(c = 3)]),
            NotExpr(Comparison(d = 4))
        ])
        """
        query = parse("a = 1 AND (b = 2 OR c = 3) AND NOT d = 4")

        # Top level is AND
        assert isinstance(query.expression, AndExpr)
        assert len(query.expression.children) == 3

        # First child: a = 1
        assert isinstance(query.expression.children[0], Comparison)
        assert query.expression.children[0].field.name == "a"

        # Second child: OR expression
        assert isinstance(query.expression.children[1], OrExpr)
        assert len(query.expression.children[1].children) == 2

        # Third child: NOT expression
        assert isinstance(query.expression.children[2], NotExpr)
        assert isinstance(query.expression.children[2].child, Comparison)
        assert query.expression.children[2].child.field.name == "d"

    def test_complex_not_with_parens(self):
        """
        NOT (a = 1 AND b = 2) OR c = 3
        Should parse as: (NOT (a = 1 AND b = 2)) OR c = 3
        """
        query = parse("NOT (a = 1 AND b = 2) OR c = 3")

        # Top level is OR
        assert isinstance(query.expression, OrExpr)
        assert len(query.expression.children) == 2

        # First child is NOT
        first = query.expression.children[0]
        assert isinstance(first, NotExpr)

        # NOT's child is AND
        assert isinstance(first.child, AndExpr)
        assert len(first.child.children) == 2

    def test_alternating_operators(self):
        """
        a = 1 OR b = 2 AND c = 3 OR d = 4 AND e = 5
        Should group ANDs first: a = 1 OR (b = 2 AND c = 3) OR (d = 4 AND e = 5)
        """
        query = parse("a = 1 OR b = 2 AND c = 3 OR d = 4 AND e = 5")

        # Top level is OR
        assert isinstance(query.expression, OrExpr)
        assert len(query.expression.children) == 3

        # First and third children are direct comparisons
        assert isinstance(query.expression.children[0], Comparison)

        # Second child is AND
        assert isinstance(query.expression.children[1], AndExpr)

        # Third child is also AND
        assert isinstance(query.expression.children[2], AndExpr)


class TestAssociativity:
    """Test operator associativity (left-to-right for same precedence)."""

    def test_or_is_left_associative(self):
        """
        a = 1 OR b = 2 OR c = 3
        Should be flattened to: OrExpr([a=1, b=2, c=3])
        (not nested as OrExpr(OrExpr(a, b), c))
        """
        query = parse("a = 1 OR b = 2 OR c = 3")

        # Should be flat OR with 3 children
        assert isinstance(query.expression, OrExpr)
        assert len(query.expression.children) == 3

        # All children are comparisons
        for child in query.expression.children:
            assert isinstance(child, Comparison)

        # Verify field names
        assert query.expression.children[0].field.name == "a"
        assert query.expression.children[1].field.name == "b"
        assert query.expression.children[2].field.name == "c"

    def test_and_is_left_associative(self):
        """
        a = 1 AND b = 2 AND c = 3
        Should be flattened to: AndExpr([a=1, b=2, c=3])
        """
        query = parse("a = 1 AND b = 2 AND c = 3")

        # Should be flat AND with 3 children
        assert isinstance(query.expression, AndExpr)
        assert len(query.expression.children) == 3

        # All children are comparisons
        for child in query.expression.children:
            assert isinstance(child, Comparison)

    def test_four_way_or(self):
        """
        a = 1 OR b = 2 OR c = 3 OR d = 4
        Should flatten to single OrExpr with 4 children
        """
        query = parse("a = 1 OR b = 2 OR c = 3 OR d = 4")

        assert isinstance(query.expression, OrExpr)
        assert len(query.expression.children) == 4

        # Verify all field names
        field_names = [child.field.name for child in query.expression.children]
        assert field_names == ["a", "b", "c", "d"]

    def test_four_way_and(self):
        """
        a = 1 AND b = 2 AND c = 3 AND d = 4
        Should flatten to single AndExpr with 4 children
        """
        query = parse("a = 1 AND b = 2 AND c = 3 AND d = 4")

        assert isinstance(query.expression, AndExpr)
        assert len(query.expression.children) == 4


class TestPrecedenceEdgeCases:
    """Test edge cases and corner cases in precedence."""

    def test_not_before_and_before_or(self):
        """
        NOT a = 1 AND b = 2 OR c = 3
        Should parse as: ((NOT a = 1) AND b = 2) OR c = 3
        """
        query = parse("NOT a = 1 AND b = 2 OR c = 3")

        # Top level is OR
        assert isinstance(query.expression, OrExpr)
        assert len(query.expression.children) == 2

        # First child is AND
        and_expr = query.expression.children[0]
        assert isinstance(and_expr, AndExpr)
        assert len(and_expr.children) == 2

        # AND's first child is NOT
        assert isinstance(and_expr.children[0], NotExpr)

        # AND's second child is comparison
        assert isinstance(and_expr.children[1], Comparison)

    def test_parens_around_single_term(self):
        """
        (a = 1) AND b = 2
        Parens are redundant but should still work
        """
        query = parse("(a = 1) AND b = 2")

        # Should simplify to AND with two comparisons
        assert isinstance(query.expression, AndExpr)
        assert len(query.expression.children) == 2
        assert isinstance(query.expression.children[0], Comparison)
        assert isinstance(query.expression.children[1], Comparison)

    def test_not_with_parens_and_or(self):
        """
        NOT (a = 1) OR b = 2
        Parens are redundant around single comparison
        """
        query = parse("NOT (a = 1) OR b = 2")

        # Top level is OR
        assert isinstance(query.expression, OrExpr)

        # First child is NOT
        assert isinstance(query.expression.children[0], NotExpr)
        assert isinstance(query.expression.children[0].child, Comparison)

    def test_deeply_nested_parens(self):
        """
        (((a = 1)))
        Multiple layers of redundant parentheses
        """
        query = parse("(((a = 1)))")

        # Should simplify to single comparison
        assert isinstance(query.expression, Comparison)
        assert query.expression.field.name == "a"

    def test_all_operators_combined(self):
        """
        NOT (a = 1 OR b = 2) AND (c = 3 OR NOT d = 4) OR e = 5
        Kitchen sink test with all operator types
        """
        query = parse("NOT (a = 1 OR b = 2) AND (c = 3 OR NOT d = 4) OR e = 5")

        # Top level is OR
        assert isinstance(query.expression, OrExpr)
        assert len(query.expression.children) == 2

        # First child is AND
        and_expr = query.expression.children[0]
        assert isinstance(and_expr, AndExpr)
        assert len(and_expr.children) == 2

        # AND's first child is NOT(OR)
        assert isinstance(and_expr.children[0], NotExpr)
        assert isinstance(and_expr.children[0].child, OrExpr)

        # AND's second child is OR containing NOT
        second_or = and_expr.children[1]
        assert isinstance(second_or, OrExpr)
        # Should contain: c = 3 and NOT d = 4
        assert isinstance(second_or.children[1], NotExpr)

        # OR's second child is comparison e = 5
        assert isinstance(query.expression.children[1], Comparison)
        assert query.expression.children[1].field.name == "e"


class TestPrecedenceSummary:
    """Summary tests documenting expected precedence behavior."""

    def test_precedence_table(self):
        """
        Documents the precedence hierarchy with concrete examples.

        From highest to lowest precedence:
        1. Parentheses ()
        2. NOT
        3. Comparisons (=, !=, >, <, >=, <=, IN, LIKE)
        4. AND
        5. OR
        """
        # Level 1: Parentheses override everything
        q1 = parse("(a = 1 OR b = 2) AND c = 3")
        assert isinstance(q1.expression, AndExpr)
        assert isinstance(q1.expression.children[0], OrExpr)

        # Level 2: NOT binds tightly
        q2 = parse("NOT a = 1 OR b = 2")
        assert isinstance(q2.expression, OrExpr)
        assert isinstance(q2.expression.children[0], NotExpr)

        # Level 4: AND binds tighter than OR
        q3 = parse("a = 1 OR b = 2 AND c = 3")
        assert isinstance(q3.expression, OrExpr)
        assert isinstance(q3.expression.children[1], AndExpr)

    def test_precedence_vs_associativity(self):
        """
        Precedence: Which operators bind more tightly (AND > OR)
        Associativity: How same-level operators group (left-to-right)
        """
        # Precedence: AND before OR
        prec = parse("a = 1 OR b = 2 AND c = 3")
        assert isinstance(prec.expression, OrExpr)
        assert isinstance(prec.expression.children[1], AndExpr)

        # Associativity: multiple ORs flatten left-to-right
        assoc = parse("a = 1 OR b = 2 OR c = 3")
        assert isinstance(assoc.expression, OrExpr)
        assert len(assoc.expression.children) == 3  # Flat, not nested
