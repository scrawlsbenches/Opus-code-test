"""
Recursive descent parser for CDG query expressions.

Parses tokens into an AST following the grammar:

    query       ::= [from_clause] [where_clause] [order_clause] [limit_clause]
                  | function_call [order_clause] [limit_clause]
    from_clause ::= 'FROM' IDENTIFIER
    where_clause::= 'WHERE' expression
    expression  ::= and_expr ('OR' and_expr)*
    and_expr    ::= not_expr ('AND' not_expr)*
    not_expr    ::= 'NOT' not_expr | primary
    primary     ::= comparison | function_call | '(' expression ')'

Examples:
    FROM task WHERE status = 'pending'
    FROM decision WHERE status = 'draft' ORDER BY created_at DESC
    blockers('T-123')
    status = 'pending' AND priority = 'high'

See: docs/design/cdg-query-language.md
"""

from typing import Optional, List

from .ast import (
    Expression, CDGQuery, Literal, Field, Comparison,
    AndExpr, OrExpr, NotExpr, FunctionCall, Op
)
from .lexer import Token, TokenType, Lexer
from .errors import QueryParseError


class Parser:
    """
    Recursive descent parser for CDG query expressions.

    Usage:
        parser = Parser("FROM task WHERE status = 'pending'")
        query = parser.parse()
    """

    def __init__(self, source: str):
        self.source = source
        self.lexer = Lexer(source)
        self.tokens: List[Token] = []
        self.position = 0

    def parse(self) -> CDGQuery:
        """Parse the source into a CDGQuery AST."""
        # Tokenize the source
        self.tokens = list(self.lexer.tokenize())
        self.position = 0

        # Parse the query
        query = self._parse_query()

        # Ensure we consumed all tokens
        if not self._match(TokenType.EOF):
            raise QueryParseError(
                f"Expected end of query, found '{self._current().value}'",
                position=self._current().position,
                source=self.source,
                expected_tokens=["EOF"]
            )

        return query

    def _parse_query(self) -> CDGQuery:
        """
        Parse: query ::= [from_clause] [where_clause] [order_clause] [limit_clause]
                       | function_call [order_clause] [limit_clause]
        """
        entity_type = None
        expression = None

        # Check for FROM clause
        if self._match(TokenType.FROM):
            entity_type = self._parse_from_clause()

            # Check for optional WHERE clause
            if self._match(TokenType.WHERE):
                self._advance()  # Consume WHERE
                expression = self._parse_expression()
        else:
            # No FROM clause - either a function call or legacy expression
            # Check if we're at EOF (empty query)
            if self._match(TokenType.EOF):
                return CDGQuery()

            # Parse expression (could be function call or filter expression)
            expression = self._parse_expression()

        # Parse optional ORDER BY clause
        order_by = self._parse_order_clause()

        # Parse optional LIMIT/OFFSET clause
        limit, offset = self._parse_limit_clause()

        return CDGQuery(
            entity_type=entity_type,
            expression=expression,
            order_by=order_by,
            limit=limit,
            offset=offset
        )

    def _parse_from_clause(self) -> str:
        """Parse: from_clause ::= 'FROM' IDENTIFIER"""
        self._advance()  # Consume FROM

        entity_token = self._expect(TokenType.IDENTIFIER, "Expected entity type after FROM")
        return entity_token.value.lower()  # Normalize to lowercase

    def _parse_expression(self) -> Expression:
        """Parse: expression ::= and_expr ('OR' and_expr)*"""
        left = self._parse_and_expr()

        # Collect all OR operands
        or_children = [left]
        while self._match(TokenType.OR):
            self._advance()  # Consume OR
            or_children.append(self._parse_and_expr())

        # If only one child, return it directly (no OrExpr wrapper)
        if len(or_children) == 1:
            return or_children[0]

        return OrExpr(children=tuple(or_children))

    def _parse_and_expr(self) -> Expression:
        """Parse: and_expr ::= not_expr ('AND' not_expr)*"""
        left = self._parse_not_expr()

        # Collect all AND operands
        and_children = [left]
        while self._match(TokenType.AND):
            self._advance()  # Consume AND
            and_children.append(self._parse_not_expr())

        # If only one child, return it directly (no AndExpr wrapper)
        if len(and_children) == 1:
            return and_children[0]

        return AndExpr(children=tuple(and_children))

    def _parse_not_expr(self) -> Expression:
        """Parse: not_expr ::= 'NOT' not_expr | primary"""
        if self._match(TokenType.NOT):
            self._advance()  # Consume NOT
            child = self._parse_not_expr()  # Recursive for multiple NOTs
            return NotExpr(child=child)

        return self._parse_primary()

    def _parse_primary(self) -> Expression:
        """Parse: primary ::= comparison | function_call | '(' expression ')'"""
        # Parenthesized expression
        if self._match(TokenType.LPAREN):
            self._advance()  # Consume (
            expr = self._parse_expression()
            self._expect(TokenType.RPAREN, "Expected ')' after expression")
            return expr

        # Must be comparison or function call
        # Both start with IDENTIFIER, so we peek ahead
        if not self._match(TokenType.IDENTIFIER):
            raise QueryParseError(
                f"Expected identifier or '(', found '{self._current().value}'",
                position=self._current().position,
                source=self.source,
                expected_tokens=["IDENTIFIER", "("]
            )

        # Peek ahead to distinguish function call from comparison
        saved_pos = self.position
        self._advance()  # Consume identifier

        if self._match(TokenType.LPAREN):
            # It's a function call - backtrack and parse
            self.position = saved_pos
            return self._parse_function_call()
        else:
            # It's a comparison - backtrack and parse
            self.position = saved_pos
            return self._parse_comparison()

    def _parse_comparison(self) -> Comparison:
        """Parse: comparison ::= IDENTIFIER op value"""
        # Parse field name
        field_token = self._expect(TokenType.IDENTIFIER, "Expected field name")
        field = Field(name=field_token.value)

        # Parse operator
        op = self._parse_operator()

        # Parse value
        value = self._parse_value()

        return Comparison(field=field, op=op, value=value)

    def _parse_operator(self) -> Op:
        """Parse comparison operator."""
        token = self._current()

        # Check for IS NULL / IS NOT NULL
        if self._match(TokenType.IS):
            self._advance()  # Consume IS
            if self._match(TokenType.NOT):
                self._advance()  # Consume NOT
                if self._match(TokenType.NULL):
                    self._advance()  # Consume NULL
                    return Op.IS_NOT_NULL
                raise QueryParseError(
                    f"Expected NULL after IS NOT, found '{self._current().value}'",
                    position=self._current().position,
                    source=self.source,
                    expected_tokens=["NULL"]
                )
            if self._match(TokenType.NULL):
                self._advance()  # Consume NULL
                return Op.IS_NULL
            raise QueryParseError(
                f"Expected NULL or NOT after IS, found '{self._current().value}'",
                position=self._current().position,
                source=self.source,
                expected_tokens=["NULL", "NOT"]
            )

        # Check for two-word operators (NOT IN, NOT LIKE)
        if self._match(TokenType.NOT):
            self._advance()  # Consume NOT
            if self._match(TokenType.IN):
                self._advance()  # Consume IN
                return Op.NOT_IN
            elif self._match(TokenType.LIKE):
                self._advance()  # Consume LIKE
                return Op.NOT_LIKE
            else:
                raise QueryParseError(
                    f"Expected IN or LIKE after NOT, found '{self._current().value}'",
                    position=self._current().position,
                    source=self.source,
                    expected_tokens=["IN", "LIKE"]
                )

        # Single-token operators
        op_map = {
            TokenType.EQ: Op.EQ,
            TokenType.NE: Op.NE,
            TokenType.GT: Op.GT,
            TokenType.LT: Op.LT,
            TokenType.GTE: Op.GTE,
            TokenType.LTE: Op.LTE,
            TokenType.IN: Op.IN,
            TokenType.LIKE: Op.LIKE,
        }

        for token_type, op in op_map.items():
            if self._match(token_type):
                self._advance()
                return op

        raise QueryParseError(
            f"Expected comparison operator, found '{token.value}'",
            position=token.position,
            source=self.source,
            expected_tokens=["=", "!=", ">", "<", ">=", "<=", "IN", "NOT IN", "LIKE", "NOT LIKE", "IS NULL", "IS NOT NULL"]
        )

    def _parse_function_call(self) -> FunctionCall:
        """Parse: function_call ::= IDENTIFIER '(' [args] ')'"""
        # Parse function name
        name_token = self._expect(TokenType.IDENTIFIER, "Expected function name")
        name = name_token.value

        # Consume opening parenthesis
        self._expect(TokenType.LPAREN, f"Expected '(' after function name '{name}'")

        # Parse arguments
        args = []
        kwargs = []

        # Check for empty argument list
        if not self._match(TokenType.RPAREN):
            self._parse_function_arguments(args, kwargs)

        # Consume closing parenthesis
        self._expect(TokenType.RPAREN, "Expected ')' after function arguments")

        return FunctionCall(name=name, args=tuple(args), kwargs=tuple(kwargs))

    def _parse_function_arguments(self, args: list, kwargs: list) -> None:
        """Parse function arguments into args and kwargs lists."""
        # Parse first argument
        arg_expr = self._parse_argument()

        # Check if it's a keyword argument
        if isinstance(arg_expr, tuple):
            kwargs.append(arg_expr)
            self._parse_remaining_kwargs(kwargs)
        else:
            args.append(arg_expr)
            self._parse_remaining_args(args, kwargs)

    def _parse_remaining_args(self, args: list, kwargs: list) -> None:
        """Parse remaining positional args, switch to kwargs when encountered."""
        while self._match(TokenType.COMMA):
            self._advance()  # Consume comma
            arg_expr = self._parse_argument()
            if isinstance(arg_expr, tuple):
                # Switched to kwargs
                kwargs.append(arg_expr)
                self._parse_remaining_kwargs(kwargs)
                return
            else:
                args.append(arg_expr)

    def _parse_remaining_kwargs(self, kwargs: list) -> None:
        """Parse remaining keyword arguments (no positional allowed)."""
        while self._match(TokenType.COMMA):
            self._advance()  # Consume comma
            kwarg = self._parse_argument()
            if not isinstance(kwarg, tuple):
                raise QueryParseError(
                    "Positional argument after keyword argument",
                    position=self._current().position,
                    source=self.source
                )
            kwargs.append(kwarg)

    def _parse_argument(self):
        """Parse a function argument (positional or keyword).

        Returns Expression for positional, or (str, Expression) tuple for keyword.
        """
        # Check if it's a keyword argument (IDENTIFIER =)
        if self._match(TokenType.IDENTIFIER):
            # Peek ahead for =
            saved_pos = self.position
            identifier = self._advance().value

            if self._match(TokenType.EQ):
                # It's a keyword argument
                self._advance()  # Consume =
                value = self._parse_value()
                return (identifier, value)
            else:
                # It's a positional argument that's an identifier
                # Backtrack and parse as value
                self.position = saved_pos
                return self._parse_value()
        else:
            # Regular positional argument
            return self._parse_value()

    def _parse_value(self) -> Expression:
        """Parse: value ::= STRING | NUMBER | IDENTIFIER | TRUE | FALSE | NULL | list"""
        token = self._current()

        if self._match(TokenType.STRING):
            self._advance()
            return Literal(value=token.value)

        elif self._match(TokenType.NUMBER):
            self._advance()
            # Convert to int or float
            value_str = token.value
            if '.' in value_str:
                return Literal(value=float(value_str))
            else:
                return Literal(value=int(value_str))

        elif self._match(TokenType.TRUE):
            self._advance()
            return Literal(value=True)

        elif self._match(TokenType.FALSE):
            self._advance()
            return Literal(value=False)

        elif self._match(TokenType.NULL):
            self._advance()
            return Literal(value=None)

        elif self._match(TokenType.IDENTIFIER):
            self._advance()
            return Literal(value=token.value)

        elif self._match(TokenType.LBRACKET):
            return self._parse_list()

        else:
            raise QueryParseError(
                f"Expected value, found '{token.value}'",
                position=token.position,
                source=self.source,
                expected_tokens=["STRING", "NUMBER", "IDENTIFIER", "TRUE", "FALSE", "NULL", "["]
            )

    def _parse_list(self) -> Literal:
        """Parse: list ::= '[' value (',' value)* ']'"""
        self._expect(TokenType.LBRACKET, "Expected '['")

        values = []

        # Check for empty list
        if not self._match(TokenType.RBRACKET):
            # Parse first value
            values.append(self._parse_list_value())

            # Parse remaining values
            while self._match(TokenType.COMMA):
                self._advance()  # Consume comma
                values.append(self._parse_list_value())

        self._expect(TokenType.RBRACKET, "Expected ']' after list values")

        return Literal(value=values)

    def _parse_list_value(self):
        """Parse a value within a list (extract raw value, not Expression)."""
        token = self._current()

        if self._match(TokenType.STRING):
            self._advance()
            return token.value

        elif self._match(TokenType.NUMBER):
            self._advance()
            value_str = token.value
            if '.' in value_str:
                return float(value_str)
            else:
                return int(value_str)

        elif self._match(TokenType.TRUE):
            self._advance()
            return True

        elif self._match(TokenType.FALSE):
            self._advance()
            return False

        elif self._match(TokenType.NULL):
            self._advance()
            return None

        elif self._match(TokenType.IDENTIFIER):
            self._advance()
            return token.value

        else:
            raise QueryParseError(
                f"Expected list value, found '{token.value}'",
                position=token.position,
                source=self.source,
                expected_tokens=["STRING", "NUMBER", "IDENTIFIER", "TRUE", "FALSE", "NULL"]
            )

    def _parse_order_clause(self) -> Optional[tuple]:
        """Parse: order_clause ::= 'ORDER' 'BY' IDENTIFIER ['ASC' | 'DESC']

        Returns (field: str, desc: bool) or None
        """
        if not self._match(TokenType.ORDER):
            return None

        self._advance()  # Consume ORDER
        self._expect(TokenType.BY, "Expected BY after ORDER")

        field_token = self._expect(TokenType.IDENTIFIER, "Expected field name after ORDER BY")
        field = field_token.value

        # Check for ASC/DESC
        desc = False
        if self._match(TokenType.DESC):
            self._advance()
            desc = True
        elif self._match(TokenType.ASC):
            self._advance()
            desc = False

        return (field, desc)

    def _parse_limit_clause(self) -> tuple:
        """Parse: limit_clause ::= 'LIMIT' NUMBER ['OFFSET' NUMBER]

        Returns (limit: int | None, offset: int | None)
        """
        if not self._match(TokenType.LIMIT):
            return (None, None)

        self._advance()  # Consume LIMIT

        limit_token = self._expect(TokenType.NUMBER, "Expected number after LIMIT")
        limit = int(limit_token.value)

        # Check for OFFSET
        offset = None
        if self._match(TokenType.OFFSET):
            self._advance()  # Consume OFFSET
            offset_token = self._expect(TokenType.NUMBER, "Expected number after OFFSET")
            offset = int(offset_token.value)

        return (limit, offset)

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
            raise QueryParseError(
                message,
                position=self._current().position,
                source=self.source
            )
        return self._advance()


def parse(source: str) -> CDGQuery:
    """Convenience function to parse a query string."""
    return Parser(source).parse()
