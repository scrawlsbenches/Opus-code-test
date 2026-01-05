"""
Unit tests for AST node implementation details.

These tests verify internal implementation details:
- Dataclass field validation
- Type checking
- Edge cases
- Error conditions
"""

import pytest
from cortical.got.expression.ast import (
    Expression, Literal, Field, Comparison, AndExpr, OrExpr, NotExpr,
    FunctionCall, Query, Op
)


class TestOpEnum:
    """Test the Op enum implementation."""

    def test_all_operators_defined(self):
        """Verify all expected operators exist."""
        expected_ops = [
            "EQ", "NE", "GT", "LT", "GTE", "LTE",
            "IN", "NOT_IN", "LIKE", "NOT_LIKE"
        ]
        for op_name in expected_ops:
            assert hasattr(Op, op_name)

    def test_operators_are_unique(self):
        """Verify each operator has a unique value."""
        op_values = [op.value for op in Op]
        assert len(op_values) == len(set(op_values))

    def test_operator_comparison(self):
        """Verify operators can be compared."""
        assert Op.EQ == Op.EQ
        assert Op.EQ != Op.NE


class TestExpressionBaseClass:
    """Test the Expression base class."""

    def test_expression_is_frozen(self):
        """Verify Expression is a frozen dataclass."""
        expr = Expression()
        # Frozen dataclasses don't allow attribute assignment
        with pytest.raises(AttributeError):
            expr.new_field = "value"

    def test_all_nodes_inherit_from_expression(self):
        """Verify all AST nodes are Expression subclasses."""
        nodes = [
            Literal("test"),
            Field("name"),
            Comparison(Field("x"), Op.EQ, Literal(1)),
            AndExpr(children=()),
            OrExpr(children=()),
            NotExpr(child=Literal(True)),
            FunctionCall(name="f", args=(), kwargs={})
        ]
        for node in nodes:
            assert isinstance(node, Expression)


class TestLiteralNode:
    """Test Literal node implementation details."""

    def test_literal_with_none_value(self):
        """Literal can hold None value."""
        node = Literal(None)
        assert node.value is None

    def test_literal_with_boolean_value(self):
        """Literal can hold boolean values."""
        node_true = Literal(True)
        node_false = Literal(False)
        assert node_true.value is True
        assert node_false.value is False

    def test_literal_with_float_value(self):
        """Literal can hold float values."""
        node = Literal(3.14)
        assert node.value == 3.14

    def test_literal_with_tuple_value(self):
        """Literal can hold tuple values (immutable sequences)."""
        node = Literal((1, 2, 3))
        assert node.value == (1, 2, 3)

    def test_literal_equality_with_different_types(self):
        """Literals with different value types are not equal."""
        node1 = Literal("42")
        node2 = Literal(42)
        assert node1 != node2

    def test_literal_repr_format(self):
        """Verify Literal repr contains value."""
        node = Literal("test")
        repr_str = repr(node)
        assert "Literal" in repr_str
        assert "value" in repr_str or "test" in repr_str


class TestFieldNode:
    """Test Field node implementation details."""

    def test_field_with_empty_string(self):
        """Field can have empty string name (though semantically invalid)."""
        node = Field("")
        assert node.name == ""

    def test_field_with_special_characters(self):
        """Field can contain special characters."""
        node = Field("field_name_123")
        assert node.name == "field_name_123"

    def test_field_equality_case_sensitive(self):
        """Field names are case-sensitive."""
        node1 = Field("Status")
        node2 = Field("status")
        assert node1 != node2

    def test_field_repr_format(self):
        """Verify Field repr contains name."""
        node = Field("status")
        repr_str = repr(node)
        assert "Field" in repr_str
        assert "name" in repr_str or "status" in repr_str


class TestComparisonNode:
    """Test Comparison node implementation details."""

    def test_comparison_with_nested_expression_as_value(self):
        """Comparison value can be a complex expression."""
        # This is a bit unusual but structurally valid
        inner = Comparison(Field("x"), Op.GT, Literal(5))
        outer = Comparison(Field("result"), Op.EQ, inner)
        assert isinstance(outer.value, Comparison)

    def test_comparison_field_must_be_field_type(self):
        """Comparison field should be a Field instance."""
        comp = Comparison(Field("status"), Op.EQ, Literal("pending"))
        assert isinstance(comp.field, Field)

    def test_comparison_equality_with_different_ops(self):
        """Comparisons with different operators are not equal."""
        comp1 = Comparison(Field("x"), Op.EQ, Literal(1))
        comp2 = Comparison(Field("x"), Op.NE, Literal(1))
        assert comp1 != comp2

    def test_comparison_repr_format(self):
        """Verify Comparison repr contains all components."""
        comp = Comparison(Field("status"), Op.EQ, Literal("pending"))
        repr_str = repr(comp)
        assert "Comparison" in repr_str


class TestAndExprNode:
    """Test AndExpr node implementation details."""

    def test_and_expr_with_empty_children(self):
        """AndExpr can be created with empty children tuple."""
        node = AndExpr(children=())
        assert len(node.children) == 0

    def test_and_expr_with_single_child(self):
        """AndExpr with single child (semantically odd but valid)."""
        comp = Comparison(Field("x"), Op.EQ, Literal(1))
        node = AndExpr(children=(comp,))
        assert len(node.children) == 1

    def test_and_expr_children_is_tuple(self):
        """AndExpr children is a tuple (immutable)."""
        comp1 = Comparison(Field("x"), Op.EQ, Literal(1))
        comp2 = Comparison(Field("y"), Op.EQ, Literal(2))
        node = AndExpr(children=(comp1, comp2))
        assert isinstance(node.children, tuple)

    def test_and_expr_equality_order_matters(self):
        """AndExpr equality considers child order."""
        comp1 = Comparison(Field("x"), Op.EQ, Literal(1))
        comp2 = Comparison(Field("y"), Op.EQ, Literal(2))
        node1 = AndExpr(children=(comp1, comp2))
        node2 = AndExpr(children=(comp2, comp1))
        # Tuples are order-sensitive
        assert node1 != node2

    def test_and_expr_repr_format(self):
        """Verify AndExpr repr is reasonable."""
        comp = Comparison(Field("x"), Op.EQ, Literal(1))
        node = AndExpr(children=(comp,))
        repr_str = repr(node)
        assert "AndExpr" in repr_str


class TestOrExprNode:
    """Test OrExpr node implementation details."""

    def test_or_expr_with_empty_children(self):
        """OrExpr can be created with empty children tuple."""
        node = OrExpr(children=())
        assert len(node.children) == 0

    def test_or_expr_with_many_children(self):
        """OrExpr can have many children."""
        children = tuple(
            Comparison(Field("status"), Op.EQ, Literal(f"status_{i}"))
            for i in range(10)
        )
        node = OrExpr(children=children)
        assert len(node.children) == 10

    def test_or_expr_children_is_tuple(self):
        """OrExpr children is a tuple (immutable)."""
        comp = Comparison(Field("x"), Op.EQ, Literal(1))
        node = OrExpr(children=(comp,))
        assert isinstance(node.children, tuple)

    def test_or_expr_repr_format(self):
        """Verify OrExpr repr is reasonable."""
        comp = Comparison(Field("x"), Op.EQ, Literal(1))
        node = OrExpr(children=(comp,))
        repr_str = repr(node)
        assert "OrExpr" in repr_str


class TestNotExprNode:
    """Test NotExpr node implementation details."""

    def test_not_expr_with_simple_comparison(self):
        """NotExpr can wrap a simple comparison."""
        comp = Comparison(Field("x"), Op.EQ, Literal(1))
        node = NotExpr(child=comp)
        assert node.child == comp

    def test_not_expr_with_complex_expression(self):
        """NotExpr can wrap a complex expression."""
        comp1 = Comparison(Field("x"), Op.EQ, Literal(1))
        comp2 = Comparison(Field("y"), Op.EQ, Literal(2))
        and_expr = AndExpr(children=(comp1, comp2))
        not_node = NotExpr(child=and_expr)
        assert isinstance(not_node.child, AndExpr)

    def test_double_negation(self):
        """NotExpr can be nested (double negation)."""
        comp = Comparison(Field("x"), Op.EQ, Literal(1))
        not1 = NotExpr(child=comp)
        not2 = NotExpr(child=not1)
        assert isinstance(not2.child, NotExpr)

    def test_not_expr_repr_format(self):
        """Verify NotExpr repr is reasonable."""
        comp = Comparison(Field("x"), Op.EQ, Literal(1))
        node = NotExpr(child=comp)
        repr_str = repr(node)
        assert "NotExpr" in repr_str


class TestFunctionCallNode:
    """Test FunctionCall node implementation details."""

    def test_function_call_with_no_args(self):
        """FunctionCall can have empty args and kwargs."""
        node = FunctionCall(name="f", args=(), kwargs=())
        assert len(node.args) == 0
        assert len(node.kwargs) == 0

    def test_function_call_args_is_tuple(self):
        """FunctionCall args is a tuple (immutable)."""
        node = FunctionCall(name="f", args=(Literal(1),), kwargs=())
        assert isinstance(node.args, tuple)

    def test_function_call_kwargs_is_tuple(self):
        """FunctionCall kwargs is a tuple of pairs (immutable)."""
        node = FunctionCall(
            name="f",
            args=(),
            kwargs=(("x", Literal(1)),)
        )
        assert isinstance(node.kwargs, tuple)

    def test_function_call_with_many_args(self):
        """FunctionCall can have multiple positional args."""
        args = tuple(Literal(i) for i in range(5))
        node = FunctionCall(name="f", args=args, kwargs=())
        assert len(node.args) == 5

    def test_function_call_kwargs_keys_are_strings(self):
        """FunctionCall kwargs keys are strings."""
        node = FunctionCall(
            name="f",
            args=(),
            kwargs=(("key1", Literal(1)), ("key2", Literal(2)))
        )
        for key, _ in node.kwargs:
            assert isinstance(key, str)

    def test_function_call_equality_considers_all_parts(self):
        """FunctionCall equality checks name, args, and kwargs."""
        node1 = FunctionCall(name="f", args=(Literal(1),), kwargs=())
        node2 = FunctionCall(name="f", args=(Literal(2),), kwargs=())
        node3 = FunctionCall(name="g", args=(Literal(1),), kwargs=())
        assert node1 != node2  # Different args
        assert node1 != node3  # Different name

    def test_function_call_repr_format(self):
        """Verify FunctionCall repr contains name."""
        node = FunctionCall(name="blocked", args=(), kwargs=())
        repr_str = repr(node)
        assert "FunctionCall" in repr_str
        assert "blocked" in repr_str


class TestQueryNode:
    """Test Query node implementation details."""

    def test_query_with_all_none_fields(self):
        """Query can be created with all optional fields as None."""
        query = Query()
        assert query.expression is None
        assert query.entity_type is None
        assert query.order_by is None
        assert query.limit is None
        assert query.offset is None

    def test_query_expression_can_be_complex(self):
        """Query expression can be any Expression subclass."""
        comp1 = Comparison(Field("x"), Op.EQ, Literal(1))
        comp2 = Comparison(Field("y"), Op.EQ, Literal(2))
        and_expr = AndExpr(children=(comp1, comp2))
        query = Query(expression=and_expr)
        assert isinstance(query.expression, AndExpr)

    def test_query_order_by_structure(self):
        """Query order_by is a tuple of (field, desc_bool)."""
        query = Query(order_by=("priority", True))
        field, desc = query.order_by
        assert field == "priority"
        assert desc is True

    def test_query_limit_can_be_zero(self):
        """Query limit can be 0 (though semantically unusual)."""
        query = Query(limit=0)
        assert query.limit == 0

    def test_query_offset_can_be_zero(self):
        """Query offset can be 0."""
        query = Query(offset=0)
        assert query.offset == 0

    def test_query_equality_considers_all_fields(self):
        """Query equality checks all fields."""
        expr = Comparison(Field("x"), Op.EQ, Literal(1))
        query1 = Query(expression=expr, limit=10)
        query2 = Query(expression=expr, limit=20)
        query3 = Query(expression=expr, limit=10)
        assert query1 != query2  # Different limit
        assert query1 == query3  # Same fields

    def test_query_repr_format(self):
        """Verify Query repr is reasonable."""
        query = Query(expression=None, limit=10)
        repr_str = repr(query)
        assert "Query" in repr_str


class TestDataclassFeatures:
    """Test that all nodes have expected dataclass features."""

    def test_nodes_have_auto_generated_init(self):
        """Dataclasses auto-generate __init__."""
        # Can create instances with keyword args
        literal = Literal(value="test")
        field = Field(name="status")
        assert literal.value == "test"
        assert field.name == "status"

    def test_nodes_have_auto_generated_repr(self):
        """Dataclasses auto-generate __repr__."""
        nodes = [
            Literal("test"),
            Field("status"),
            Comparison(Field("x"), Op.EQ, Literal(1)),
            AndExpr(children=()),
            OrExpr(children=()),
            NotExpr(child=Literal(True)),
            FunctionCall(name="f", args=(), kwargs=()),
            Query()
        ]
        for node in nodes:
            repr_str = repr(node)
            assert len(repr_str) > 0
            assert node.__class__.__name__ in repr_str

    def test_nodes_have_auto_generated_eq(self):
        """Dataclasses auto-generate __eq__."""
        node1 = Literal("test")
        node2 = Literal("test")
        node3 = Literal("other")
        assert node1 == node2
        assert node1 != node3

    def test_nodes_have_auto_generated_hash(self):
        """Frozen dataclasses auto-generate __hash__."""
        nodes = [
            Literal("test"),
            Field("status"),
            Comparison(Field("x"), Op.EQ, Literal(1)),
            AndExpr(children=()),
            NotExpr(child=Literal(True)),
            FunctionCall(name="f", args=(), kwargs=()),
            Query()
        ]
        for node in nodes:
            hash_value = hash(node)
            assert isinstance(hash_value, int)


class TestImmutabilityEnforcement:
    """Test that frozen dataclass immutability is enforced."""

    def test_cannot_add_new_attributes(self):
        """Cannot add new attributes to frozen dataclass instances."""
        node = Literal("test")
        with pytest.raises(AttributeError):
            node.new_attr = "value"

    def test_cannot_modify_existing_attributes(self):
        """Cannot modify existing attributes."""
        node = Field("status")
        with pytest.raises(AttributeError):
            node.name = "priority"

    def test_cannot_delete_attributes(self):
        """Cannot delete attributes from frozen instances."""
        node = Literal("test")
        with pytest.raises(AttributeError):
            del node.value
