"""
Query Expression System for Graph of Thought.

NOTE: This module is being deprecated in favor of cortical.cdg.query.
Most functionality has been migrated to the CDG query system.

Remaining components:
- translator: Natural language to DSL translation
- validator: Field validation with COMMON_FIELDS
- ast: AST node definitions (dependency of validator)
- errors: Error types (dependency of validator)

For new code, use:
    from cortical.cdg.query import CDGQueryEngine
"""

from .validator import FieldValidator, COMMON_FIELDS
from . import translator

from .ast import (
    Expression,
    Query,
    Literal,
    Field,
    Comparison,
    AndExpr,
    OrExpr,
    NotExpr,
    FunctionCall,
    Op,
)

from .errors import (
    QueryError,
    LexerError,
    ParseError,
    ExecutionError,
    QueryValidationError,
)


__all__ = [
    # Validator
    'FieldValidator',
    'COMMON_FIELDS',

    # Translator
    'translator',

    # AST nodes (kept for validator compatibility)
    'Expression',
    'Query',
    'Literal',
    'Field',
    'Comparison',
    'AndExpr',
    'OrExpr',
    'NotExpr',
    'FunctionCall',
    'Op',

    # Errors
    'QueryError',
    'LexerError',
    'ParseError',
    'ExecutionError',
    'QueryValidationError',
]
