"""
AST (Abstract Syntax Tree) node types for CDG query expressions.

All nodes are frozen dataclasses for immutability and hashability.

See: docs/design/cdg-query-language.md
"""

from dataclasses import dataclass
from typing import Any, Optional
from enum import Enum, auto


class Op(Enum):
    """Comparison operators."""
    EQ = auto()       # =
    NE = auto()       # !=
    GT = auto()       # >
    LT = auto()       # <
    GTE = auto()      # >=
    LTE = auto()      # <=
    IN = auto()       # IN
    NOT_IN = auto()   # NOT IN
    LIKE = auto()     # LIKE
    NOT_LIKE = auto() # NOT LIKE
    IS_NULL = auto()      # IS NULL (future)
    IS_NOT_NULL = auto()  # IS NOT NULL (future)


@dataclass(frozen=True)
class Expression:
    """Base class for all expression nodes."""
    pass


@dataclass(frozen=True)
class Literal(Expression):
    """A literal value (string, number, boolean, list, None)."""
    value: Any


@dataclass(frozen=True)
class Field(Expression):
    """A field reference."""
    name: str


@dataclass(frozen=True)
class Comparison(Expression):
    """A comparison expression (field op value)."""
    field: Field
    op: Op
    value: Expression


@dataclass(frozen=True)
class AndExpr(Expression):
    """Logical AND of multiple expressions."""
    children: tuple  # Tuple[Expression, ...]


@dataclass(frozen=True)
class OrExpr(Expression):
    """Logical OR of multiple expressions."""
    children: tuple  # Tuple[Expression, ...]


@dataclass(frozen=True)
class NotExpr(Expression):
    """Logical NOT of an expression."""
    child: Expression


@dataclass(frozen=True)
class FunctionCall(Expression):
    """
    A function call with positional and keyword arguments.

    Note: kwargs is a tuple of (key, value) pairs to maintain immutability
    and hashability. Use dict(node.kwargs) to convert to dict if needed.
    """
    name: str
    args: tuple  # Tuple[Expression, ...]
    kwargs: tuple  # Tuple[Tuple[str, Expression], ...] - immutable key-value pairs


@dataclass(frozen=True)
class CDGQuery:
    """
    A complete CDG query with optional clauses.

    This is the root node of the AST, representing a full query like:
        FROM task WHERE status = 'pending' ORDER BY created_at DESC LIMIT 10

    Attributes:
        entity_type: The entity type being queried (FROM clause) - required for entity queries
        expression: The filter expression (WHERE clause) - optional
        order_by: Optional (field, desc) tuple for sorting
        limit: Optional maximum results
        offset: Optional skip count
    """
    entity_type: Optional[str] = None
    expression: Optional[Expression] = None
    order_by: Optional[tuple] = None  # (field: str, desc: bool)
    limit: Optional[int] = None
    offset: Optional[int] = None

    def is_function_query(self) -> bool:
        """Check if this is a standalone function call (no FROM clause)."""
        return (
            self.entity_type is None and
            self.expression is not None and
            isinstance(self.expression, FunctionCall)
        )

    def is_entity_query(self) -> bool:
        """Check if this is an entity query (has FROM clause)."""
        return self.entity_type is not None
