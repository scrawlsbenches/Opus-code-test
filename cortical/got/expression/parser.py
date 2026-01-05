"""
Recursive descent parser for query expressions.

Parses tokens into an AST following the grammar:
    query       ::= expression [order_clause] [limit_clause]
    expression  ::= and_expr ('OR' and_expr)*
    and_expr    ::= not_expr ('AND' not_expr)*
    not_expr    ::= 'NOT' not_expr | primary
    primary     ::= comparison | function_call | '(' expression ')'
"""

from typing import Optional, List

from .ast import (
    Expression, Query, Literal, Field, Comparison,
    AndExpr, OrExpr, NotExpr, FunctionCall, Op
)
from .lexer import Token, TokenType, Lexer
from .errors import ParseError


class Parser:
    """
    Recursive descent parser for query expressions.

    Usage:
        parser = Parser("status = 'pending' AND priority = 'high'")
        query = parser.parse()
    """

    def __init__(self, source: str):
        self.source = source
        self.lexer = Lexer(source)
        self.tokens: List[Token] = []
        self.position = 0

    def parse(self) -> Query:
        """Parse the source into a Query AST."""
        # Implementation will be completed in T-007
        raise NotImplementedError("Parser implementation pending (T-007)")

    def _current(self) -> Token:
        """Get the current token."""
        if self.position < len(self.tokens):
            return self.tokens[self.position]
        return Token(TokenType.EOF, '', len(self.source))

    def _advance(self) -> Token:
        """Consume and return the current token."""
        token = self._current()
        self.position += 1
        return token

    def _match(self, *types: TokenType) -> bool:
        """Check if current token matches any of the given types."""
        return self._current().type in types

    def _expect(self, token_type: TokenType, message: str) -> Token:
        """Consume a token of the expected type or raise an error."""
        if self._current().type != token_type:
            raise ParseError(
                message,
                position=self._current().position,
                source=self.source
            )
        return self._advance()


def parse(source: str) -> Query:
    """Convenience function to parse a query string."""
    return Parser(source).parse()
