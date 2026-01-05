"""
Lexer (tokenizer) for query expressions.

Converts a query string into a stream of tokens for the parser.
Includes position tracking for error reporting.
"""

from dataclasses import dataclass
from typing import List, Optional, Iterator
from enum import Enum, auto

from .errors import LexerError


class TokenType(Enum):
    """Token types for the query language."""
    # Literals
    STRING = auto()
    NUMBER = auto()
    IDENTIFIER = auto()

    # Operators
    EQ = auto()       # =
    NE = auto()       # !=
    GT = auto()       # >
    LT = auto()       # <
    GTE = auto()      # >=
    LTE = auto()      # <=

    # Keywords
    AND = auto()
    OR = auto()
    NOT = auto()
    IN = auto()
    LIKE = auto()
    ORDER = auto()
    BY = auto()
    ASC = auto()
    DESC = auto()
    LIMIT = auto()
    OFFSET = auto()

    # Punctuation
    LPAREN = auto()   # (
    RPAREN = auto()   # )
    LBRACKET = auto() # [
    RBRACKET = auto() # ]
    COMMA = auto()    # ,

    # Special
    EOF = auto()


@dataclass
class Token:
    """A token with type, value, and position information."""
    type: TokenType
    value: str
    position: int

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, pos={self.position})"


class Lexer:
    """
    Tokenizer for query expressions.

    Usage:
        lexer = Lexer("status = 'pending'")
        tokens = list(lexer.tokenize())
    """

    KEYWORDS = {
        'and': TokenType.AND,
        'or': TokenType.OR,
        'not': TokenType.NOT,
        'in': TokenType.IN,
        'like': TokenType.LIKE,
        'order': TokenType.ORDER,
        'by': TokenType.BY,
        'asc': TokenType.ASC,
        'desc': TokenType.DESC,
        'limit': TokenType.LIMIT,
        'offset': TokenType.OFFSET,
    }

    def __init__(self, source: str):
        self.source = source
        self.position = 0
        self.length = len(source)

    def tokenize(self) -> Iterator[Token]:
        """Generate tokens from the source string."""
        while self.position < self.length:
            self._skip_whitespace()
            if self.position >= self.length:
                break

            token = self._next_token()
            if token:
                yield token

        yield Token(TokenType.EOF, '', self.position)

    def _skip_whitespace(self) -> None:
        """Skip whitespace characters."""
        while self.position < self.length and self.source[self.position].isspace():
            self.position += 1

    def _next_token(self) -> Optional[Token]:
        """Extract the next token."""
        start_pos = self.position
        char = self._peek()

        if char is None:
            return None

        # String literals
        if char in ("'", '"'):
            return self._read_string()

        # Numbers
        if char.isdigit():
            return self._read_number()

        # Identifiers and keywords
        if char.isalpha() or char == '_':
            return self._read_identifier_or_keyword()

        # Entity IDs like T-123 (identifier starting with letter, containing hyphen)
        if char.isalpha():
            return self._read_identifier_or_keyword()

        # Operators
        if char == '=':
            self._advance()
            return Token(TokenType.EQ, '=', start_pos)
        if char == '!':
            self._advance()
            if self._peek() == '=':
                self._advance()
                return Token(TokenType.NE, '!=', start_pos)
            raise LexerError(
                f"Unexpected character '!' (expected '!=')",
                position=start_pos,
                source=self.source
            )
        if char == '>':
            self._advance()
            if self._peek() == '=':
                self._advance()
                return Token(TokenType.GTE, '>=', start_pos)
            return Token(TokenType.GT, '>', start_pos)
        if char == '<':
            self._advance()
            if self._peek() == '=':
                self._advance()
                return Token(TokenType.LTE, '<=', start_pos)
            return Token(TokenType.LT, '<', start_pos)

        # Punctuation
        if char == '(':
            self._advance()
            return Token(TokenType.LPAREN, '(', start_pos)
        if char == ')':
            self._advance()
            return Token(TokenType.RPAREN, ')', start_pos)
        if char == '[':
            self._advance()
            return Token(TokenType.LBRACKET, '[', start_pos)
        if char == ']':
            self._advance()
            return Token(TokenType.RBRACKET, ']', start_pos)
        if char == ',':
            self._advance()
            return Token(TokenType.COMMA, ',', start_pos)

        # Invalid character
        raise LexerError(
            f"Unexpected character '{char}'",
            position=start_pos,
            source=self.source
        )

    def _read_string(self) -> Token:
        """Read a string literal (single or double quoted)."""
        start_pos = self.position
        quote_char = self._advance()  # Consume opening quote
        chars = []

        while True:
            char = self._peek()

            if char is None:
                raise LexerError(
                    f"Unterminated string literal",
                    position=start_pos,
                    source=self.source
                )

            if char == quote_char:
                self._advance()  # Consume closing quote
                break

            if char == '\\':
                # Handle escape sequences
                self._advance()  # Consume backslash
                next_char = self._peek()
                if next_char is None:
                    raise LexerError(
                        f"Unterminated string literal (escape at end)",
                        position=start_pos,
                        source=self.source
                    )
                if next_char == 'n':
                    chars.append('\n')
                    self._advance()
                elif next_char == 't':
                    chars.append('\t')
                    self._advance()
                elif next_char == 'r':
                    chars.append('\r')
                    self._advance()
                elif next_char == '\\':
                    chars.append('\\')
                    self._advance()
                elif next_char == quote_char:
                    chars.append(quote_char)
                    self._advance()
                else:
                    # Unknown escape, keep the character as-is
                    chars.append(next_char)
                    self._advance()
            else:
                chars.append(char)
                self._advance()

        return Token(TokenType.STRING, ''.join(chars), start_pos)

    def _read_number(self) -> Token:
        """Read a numeric literal (integer or float)."""
        start_pos = self.position
        chars = []

        # Read integer part
        while self._peek() is not None and self._peek().isdigit():
            chars.append(self._advance())

        # Check for decimal point
        if self._peek() == '.' and self._peek(1) is not None and self._peek(1).isdigit():
            chars.append(self._advance())  # Consume '.'
            # Read fractional part
            while self._peek() is not None and self._peek().isdigit():
                chars.append(self._advance())

        return Token(TokenType.NUMBER, ''.join(chars), start_pos)

    def _read_identifier_or_keyword(self) -> Token:
        """Read an identifier or keyword."""
        start_pos = self.position
        chars = []

        # First character (already validated as letter or underscore)
        chars.append(self._advance())

        # Read remaining characters (letters, digits, underscores, hyphens)
        while True:
            char = self._peek()
            if char is None:
                break
            # Allow alphanumeric, underscore, and hyphen in identifiers
            # This handles entity IDs like T-123
            if char.isalnum() or char in ('_', '-'):
                chars.append(self._advance())
            else:
                break

        value = ''.join(chars)

        # Check if it's a keyword (case-insensitive)
        keyword_type = self.KEYWORDS.get(value.lower())
        if keyword_type:
            return Token(keyword_type, value, start_pos)

        return Token(TokenType.IDENTIFIER, value, start_pos)

    def _peek(self, offset: int = 0) -> Optional[str]:
        """Peek at a character without consuming it."""
        pos = self.position + offset
        if pos < self.length:
            return self.source[pos]
        return None

    def _advance(self) -> str:
        """Consume and return the current character."""
        char = self.source[self.position]
        self.position += 1
        return char


def tokenize(source: str) -> List[Token]:
    """Convenience function to tokenize a query string."""
    return list(Lexer(source).tokenize())
