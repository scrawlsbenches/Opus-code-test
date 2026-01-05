"""
Query Expression System for Graph of Thought.

This module provides a DSL (Domain Specific Language) for querying the GoT
graph. Expressions are parsed into an AST and executed against the Query
builder infrastructure.

Public API:
    parse(query_str) -> Query
        Parse a query string into an AST.

    execute(manager, query) -> Any
        Execute a Query AST against a GoTManager.

    validate(expression, entity_type=None) -> None
        Validate field names in an expression against the schema.
        Raises QueryValidationError if invalid fields are found.

Example:
    from cortical.got.expression import parse, execute, validate
    from cortical.core.bootstrap import create_container
    from cortical.got.api import GoTManager

    container = create_container()
    manager = container.resolve(GoTManager)

    query = parse("status = 'pending' AND priority = 'high'")
    validate(query.expression, entity_type='task')  # Validate fields
    results = execute(manager, query)
"""

from .parser import parse
from .executor import execute
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

from .registry import (
    FunctionRegistry,
    FunctionSignature,
    QueryFunction,
)

from .lexer import (
    Lexer,
    Token,
    TokenType,
    tokenize,
)

def validate(expression, entity_type=None):
    """
    Validate field names in an expression against the schema.

    Args:
        expression: Expression or Query to validate
        entity_type: Optional entity type to validate against (e.g., 'task')
                    If None, only common fields are valid.

    Raises:
        QueryValidationError: If any field references are invalid

    Example:
        query = parse("status = 'pending' AND priority = 'high'")
        validate(query.expression, entity_type='task')
    """
    # Handle Query objects - extract the expression
    if isinstance(expression, Query):
        expression = expression.expression

    # If expression is None (empty query), nothing to validate
    if expression is None:
        return

    validator = FieldValidator(entity_type=entity_type)
    validator.validate_expression(expression)


__all__ = [
    # Main API
    'parse',
    'execute',
    'validate',

    # AST nodes
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

    # Registry
    'FunctionRegistry',
    'FunctionSignature',
    'QueryFunction',

    # Lexer
    'Lexer',
    'Token',
    'TokenType',
    'tokenize',

    # Validator
    'FieldValidator',
    'COMMON_FIELDS',

    # Translator
    'translator',
]
