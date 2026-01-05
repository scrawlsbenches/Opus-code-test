"""
Field validation for query expressions.

Validates field names against SchemaRegistry, providing helpful error
messages with suggestions for typos.

Usage:
    validator = FieldValidator(entity_type='task')
    validator.validate_expression(expr)  # Raises QueryValidationError if invalid
"""

from typing import Optional, List, Set

from ..schema import get_registry
from .ast import (
    Expression,
    Field,
    Comparison,
    AndExpr,
    OrExpr,
    NotExpr,
    FunctionCall,
)
from .errors import QueryValidationError


# Common fields valid for all entity types
COMMON_FIELDS = {'id', 'title', 'status', 'created_at', 'modified_at'}


class FieldValidator:
    """Validates field names against SchemaRegistry."""

    def __init__(self, entity_type: Optional[str] = None):
        """
        Initialize validator.

        Args:
            entity_type: If specified, validate against this entity's schema.
                        If None, validate against all common fields only.
        """
        self.entity_type = entity_type
        self.registry = get_registry()
        self._valid_fields: Optional[Set[str]] = None

    def _get_valid_fields(self) -> Set[str]:
        """
        Get the set of valid field names.

        Returns:
            Set of valid field names for the entity type
        """
        if self._valid_fields is not None:
            return self._valid_fields

        valid_fields = set(COMMON_FIELDS)

        if self.entity_type:
            schema = self.registry.get_schema(self.entity_type)
            if schema:
                # Add all fields from schema
                valid_fields.update(schema.fields.keys())

        self._valid_fields = valid_fields
        return valid_fields

    def validate_field(self, field_name: str) -> None:
        """
        Validate that field_name exists.

        Args:
            field_name: Field name to validate

        Raises:
            QueryValidationError: If field doesn't exist, with suggestions
        """
        valid_fields = self._get_valid_fields()

        if field_name not in valid_fields:
            raise QueryValidationError(
                f"Unknown field: '{field_name}'",
                field_name=field_name,
                valid_fields=list(valid_fields),
            )

    def validate_expression(self, expr: Expression) -> None:
        """
        Walk the AST and validate all field references.

        Args:
            expr: Expression tree to validate

        Raises:
            QueryValidationError: For first invalid field found
        """
        self._validate_node(expr)

    def _validate_node(self, node: Expression) -> None:
        """
        Recursively validate a node and its children.

        Args:
            node: AST node to validate

        Raises:
            QueryValidationError: If any field is invalid
        """
        if isinstance(node, Field):
            # Direct field reference - validate it
            self.validate_field(node.name)

        elif isinstance(node, Comparison):
            # Comparison has a field - validate it
            # The field is guaranteed to be a Field node by the parser
            self.validate_field(node.field.name)
            # Also validate the value expression (it might contain fields)
            self._validate_node(node.value)

        elif isinstance(node, AndExpr):
            # Validate all children
            for child in node.children:
                self._validate_node(child)

        elif isinstance(node, OrExpr):
            # Validate all children
            for child in node.children:
                self._validate_node(child)

        elif isinstance(node, NotExpr):
            # Validate the child
            self._validate_node(node.child)

        elif isinstance(node, FunctionCall):
            # Validate arguments but NOT the function name itself
            # (function validation is handled by the executor)
            for arg in node.args:
                self._validate_node(arg)
            for _, value in node.kwargs:
                self._validate_node(value)

        # Other node types (Literal, etc.) don't have fields to validate
