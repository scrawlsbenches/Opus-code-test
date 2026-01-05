"""
Behavioral tests for AST node types.

These tests verify user-facing behavior of AST nodes:
- Node creation with correct attributes
- Immutability (frozen dataclasses)
- Readable string representations
- Proper equality semantics
"""

import pytest
from cortical.got.expression.ast import (
    Expression, Literal, Field, Comparison, AndExpr, OrExpr, NotExpr,
    FunctionCall, Query, Op
)


class TestLiteralNodeBehavior:
    """Test literal value nodes."""

    def test_create_literal_with_string_value(self):
        """User creates a literal node with a string value."""
        node = Literal("pending")
        assert node.value == "pending"

    def test_create_literal_with_number_value(self):
        """User creates a literal node with a numeric value."""
        node = Literal(42)
        assert node.value == 42

    def test_create_literal_with_list_value(self):
        """User creates a literal node with a list value."""
        node = Literal(["pending", "in_progress"])
        assert node.value == ["pending", "in_progress"]

    def test_literal_has_readable_repr(self):
        """User inspects literal node and sees readable representation."""
        node = Literal("pending")
        repr_str = repr(node)
        assert "Literal" in repr_str
        assert "pending" in repr_str


class TestFieldNodeBehavior:
    """Test field reference nodes."""

    def test_create_field_node(self):
        """User creates a field reference node."""
        node = Field("status")
        assert node.name == "status"

    def test_field_has_readable_repr(self):
        """User inspects field node and sees readable representation."""
        node = Field("priority")
        repr_str = repr(node)
        assert "Field" in repr_str
        assert "priority" in repr_str


class TestComparisonNodeBehavior:
    """Test comparison expression nodes."""

    def test_create_simple_comparison(self):
        """User creates a comparison: status = 'pending'."""
        node = Comparison(
            field=Field("status"),
            op=Op.EQ,
            value=Literal("pending")
        )
        assert node.field.name == "status"
        assert node.op == Op.EQ
        assert node.value.value == "pending"

    def test_create_comparison_with_different_operators(self):
        """User creates comparisons with various operators."""
        operators = [
            (Op.EQ, "="),
            (Op.NE, "!="),
            (Op.GT, ">"),
            (Op.LT, "<"),
            (Op.GTE, ">="),
            (Op.LTE, "<="),
            (Op.IN, "IN"),
            (Op.NOT_IN, "NOT IN"),
            (Op.LIKE, "LIKE"),
            (Op.NOT_LIKE, "NOT LIKE"),
        ]
        for op, _ in operators:
            node = Comparison(
                field=Field("priority"),
                op=op,
                value=Literal("high")
            )
            assert node.op == op

    def test_comparison_has_readable_repr(self):
        """User inspects comparison and sees readable representation."""
        node = Comparison(
            field=Field("status"),
            op=Op.EQ,
            value=Literal("pending")
        )
        repr_str = repr(node)
        assert "Comparison" in repr_str


class TestLogicalExpressionBehavior:
    """Test logical AND/OR/NOT expressions."""

    def test_create_and_expression_with_two_children(self):
        """User creates an AND expression with two comparisons."""
        comp1 = Comparison(Field("status"), Op.EQ, Literal("pending"))
        comp2 = Comparison(Field("priority"), Op.EQ, Literal("high"))
        node = AndExpr(children=(comp1, comp2))

        assert len(node.children) == 2
        assert node.children[0] == comp1
        assert node.children[1] == comp2

    def test_create_or_expression_with_multiple_children(self):
        """User creates an OR expression with multiple comparisons."""
        comp1 = Comparison(Field("status"), Op.EQ, Literal("pending"))
        comp2 = Comparison(Field("status"), Op.EQ, Literal("in_progress"))
        comp3 = Comparison(Field("status"), Op.EQ, Literal("blocked"))
        node = OrExpr(children=(comp1, comp2, comp3))

        assert len(node.children) == 3

    def test_create_not_expression(self):
        """User creates a NOT expression."""
        comp = Comparison(Field("status"), Op.EQ, Literal("completed"))
        node = NotExpr(child=comp)

        assert node.child == comp

    def test_nested_logical_expressions(self):
        """User creates nested logical expressions."""
        # (status = 'pending' OR status = 'in_progress') AND priority = 'high'
        comp1 = Comparison(Field("status"), Op.EQ, Literal("pending"))
        comp2 = Comparison(Field("status"), Op.EQ, Literal("in_progress"))
        or_expr = OrExpr(children=(comp1, comp2))

        comp3 = Comparison(Field("priority"), Op.EQ, Literal("high"))
        and_expr = AndExpr(children=(or_expr, comp3))

        assert len(and_expr.children) == 2
        assert isinstance(and_expr.children[0], OrExpr)
        assert isinstance(and_expr.children[1], Comparison)

    def test_logical_expressions_have_readable_repr(self):
        """User inspects logical expressions and sees readable representation."""
        comp1 = Comparison(Field("status"), Op.EQ, Literal("pending"))
        comp2 = Comparison(Field("priority"), Op.EQ, Literal("high"))
        and_node = AndExpr(children=(comp1, comp2))

        repr_str = repr(and_node)
        assert "AndExpr" in repr_str


class TestFunctionCallBehavior:
    """Test function call nodes."""

    def test_create_function_call_with_positional_args(self):
        """User creates a function call with positional arguments."""
        node = FunctionCall(
            name="blocked",
            args=(Literal("T-001"),),
            kwargs=()
        )
        assert node.name == "blocked"
        assert len(node.args) == 1
        assert node.args[0].value == "T-001"

    def test_create_function_call_with_keyword_args(self):
        """User creates a function call with keyword arguments."""
        node = FunctionCall(
            name="path",
            args=(),
            kwargs=(
                ("from_id", Literal("T-001")),
                ("to_id", Literal("T-010"))
            )
        )
        assert node.name == "path"
        assert len(node.kwargs) == 2
        # Check kwargs as tuple of pairs
        kwargs_dict = dict(node.kwargs)
        assert "from_id" in kwargs_dict
        assert "to_id" in kwargs_dict

    def test_create_function_call_with_mixed_args(self):
        """User creates a function call with both positional and keyword arguments."""
        node = FunctionCall(
            name="edge",
            args=(Literal("T-001"),),
            kwargs=(("edge_type", Literal("DEPENDS_ON")),)
        )
        assert len(node.args) == 1
        assert len(node.kwargs) == 1

    def test_function_call_has_readable_repr(self):
        """User inspects function call and sees readable representation."""
        node = FunctionCall(
            name="blocked",
            args=(Literal("T-001"),),
            kwargs=()
        )
        repr_str = repr(node)
        assert "FunctionCall" in repr_str
        assert "blocked" in repr_str


class TestQueryBehavior:
    """Test complete query nodes."""

    def test_create_simple_query_with_expression(self):
        """User creates a query with just a filter expression."""
        expr = Comparison(Field("status"), Op.EQ, Literal("pending"))
        query = Query(expression=expr)

        assert query.expression == expr
        assert query.entity_type is None
        assert query.order_by is None
        assert query.limit is None
        assert query.offset is None

    def test_create_query_with_entity_type(self):
        """User creates a query specifying entity type."""
        expr = Comparison(Field("status"), Op.EQ, Literal("pending"))
        query = Query(expression=expr, entity_type="task")

        assert query.entity_type == "task"

    def test_create_query_with_order_by(self):
        """User creates a query with ordering."""
        expr = Comparison(Field("status"), Op.EQ, Literal("pending"))
        query = Query(
            expression=expr,
            order_by=("priority", True)  # descending
        )

        assert query.order_by == ("priority", True)

    def test_create_query_with_limit_and_offset(self):
        """User creates a query with pagination."""
        expr = Comparison(Field("status"), Op.EQ, Literal("pending"))
        query = Query(
            expression=expr,
            limit=10,
            offset=20
        )

        assert query.limit == 10
        assert query.offset == 20

    def test_create_complete_query_with_all_clauses(self):
        """User creates a full query with all optional clauses."""
        expr = Comparison(Field("status"), Op.EQ, Literal("pending"))
        query = Query(
            expression=expr,
            entity_type="task",
            order_by=("created_at", False),  # ascending
            limit=50,
            offset=0
        )

        assert query.expression is not None
        assert query.entity_type == "task"
        assert query.order_by is not None
        assert query.limit == 50
        assert query.offset == 0

    def test_query_has_readable_repr(self):
        """User inspects query and sees readable representation."""
        expr = Comparison(Field("status"), Op.EQ, Literal("pending"))
        query = Query(expression=expr)

        repr_str = repr(query)
        assert "Query" in repr_str


class TestNodeImmutability:
    """Test that all nodes are immutable (frozen dataclasses)."""

    def test_literal_is_immutable(self):
        """User cannot modify literal node after creation."""
        node = Literal("pending")
        with pytest.raises(AttributeError):
            node.value = "completed"

    def test_field_is_immutable(self):
        """User cannot modify field node after creation."""
        node = Field("status")
        with pytest.raises(AttributeError):
            node.name = "priority"

    def test_comparison_is_immutable(self):
        """User cannot modify comparison node after creation."""
        node = Comparison(Field("status"), Op.EQ, Literal("pending"))
        with pytest.raises(AttributeError):
            node.op = Op.NE

    def test_and_expr_is_immutable(self):
        """User cannot modify AND expression after creation."""
        comp = Comparison(Field("status"), Op.EQ, Literal("pending"))
        node = AndExpr(children=(comp,))
        with pytest.raises(AttributeError):
            node.children = ()

    def test_query_is_immutable(self):
        """User cannot modify query after creation."""
        expr = Comparison(Field("status"), Op.EQ, Literal("pending"))
        query = Query(expression=expr)
        with pytest.raises(AttributeError):
            query.limit = 100


class TestNodeEquality:
    """Test equality semantics for AST nodes."""

    def test_identical_literals_are_equal(self):
        """Two literal nodes with same value are equal."""
        node1 = Literal("pending")
        node2 = Literal("pending")
        assert node1 == node2

    def test_different_literals_are_not_equal(self):
        """Two literal nodes with different values are not equal."""
        node1 = Literal("pending")
        node2 = Literal("completed")
        assert node1 != node2

    def test_identical_comparisons_are_equal(self):
        """Two comparison nodes with same components are equal."""
        node1 = Comparison(Field("status"), Op.EQ, Literal("pending"))
        node2 = Comparison(Field("status"), Op.EQ, Literal("pending"))
        assert node1 == node2

    def test_identical_and_expressions_are_equal(self):
        """Two AND expressions with same children are equal."""
        comp1 = Comparison(Field("status"), Op.EQ, Literal("pending"))
        comp2 = Comparison(Field("priority"), Op.EQ, Literal("high"))

        node1 = AndExpr(children=(comp1, comp2))
        node2 = AndExpr(children=(comp1, comp2))
        assert node1 == node2

    def test_identical_queries_are_equal(self):
        """Two queries with same attributes are equal."""
        expr = Comparison(Field("status"), Op.EQ, Literal("pending"))
        query1 = Query(expression=expr, limit=10)
        query2 = Query(expression=expr, limit=10)
        assert query1 == query2


class TestNodeHashability:
    """Test that frozen nodes are hashable and can be used in sets/dicts."""

    def test_literal_nodes_are_hashable(self):
        """Literal nodes can be used in sets and as dict keys."""
        node1 = Literal("pending")
        node2 = Literal("completed")
        node_set = {node1, node2}
        assert len(node_set) == 2

        node_dict = {node1: "first", node2: "second"}
        assert node_dict[node1] == "first"

    def test_comparison_nodes_are_hashable(self):
        """Comparison nodes can be used in sets."""
        node1 = Comparison(Field("status"), Op.EQ, Literal("pending"))
        node2 = Comparison(Field("priority"), Op.EQ, Literal("high"))
        node_set = {node1, node2}
        assert len(node_set) == 2

    def test_complex_expressions_are_hashable(self):
        """Complex nested expressions can be used in sets."""
        comp1 = Comparison(Field("status"), Op.EQ, Literal("pending"))
        comp2 = Comparison(Field("priority"), Op.EQ, Literal("high"))
        and_expr = AndExpr(children=(comp1, comp2))

        expr_set = {and_expr}
        assert len(expr_set) == 1
