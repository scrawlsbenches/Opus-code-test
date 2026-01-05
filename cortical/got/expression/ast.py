"""
AST (Abstract Syntax Tree) node types for query expressions.

All nodes are frozen dataclasses for immutability and hashability.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from enum import Enum, auto


class Op(Enum):
    """Comparison operators."""
    EQ = auto()      # =
    NE = auto()      # !=
    GT = auto()      # >
    LT = auto()      # <
    GTE = auto()     # >=
    LTE = auto()     # <=
    IN = auto()      # IN
    NOT_IN = auto()  # NOT IN
    LIKE = auto()    # LIKE
    NOT_LIKE = auto()  # NOT LIKE


@dataclass(frozen=True)
class Expression:
    """Base class for all expression nodes."""
    pass


@dataclass(frozen=True)
class Literal(Expression):
    """A literal value (string, number, list)."""
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
class Query:
    """
    A complete query with optional clauses.

    Attributes:
        expression: The filter expression (WHERE clause)
        entity_type: The entity type being queried (FROM clause)
        order_by: Optional (field, direction) tuple
        limit: Optional maximum results
        offset: Optional skip count
    """
    expression: Optional[Expression] = None
    entity_type: Optional[str] = None
    order_by: Optional[tuple] = None  # (field: str, desc: bool)
    limit: Optional[int] = None
    offset: Optional[int] = None
