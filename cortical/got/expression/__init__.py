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

Example:
    from cortical.got.expression import parse, execute
    from cortical.core.bootstrap import create_container
    from cortical.got.api import GoTManager

    container = create_container()
    manager = container.resolve(GoTManager)

    query = parse("status = 'pending' AND priority = 'high'")
    results = execute(manager, query)
"""

from .parser import parse
from .executor import execute

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

__all__ = [
    # Main API
    'parse',
    'execute',

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
]
