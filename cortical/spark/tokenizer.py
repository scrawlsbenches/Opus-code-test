"""
Code Tokenizer
==============

Code-aware tokenizer that preserves punctuation, operators, and structure.
Unlike natural language tokenizers, this is designed specifically for source code.

Features:
- Preserves punctuation: . ( ) [ ] { } : , ; @ #
- Preserves operators: == != >= <= += -= etc.
- Splits camelCase and snake_case identifiers
- Optionally includes string literal contents

Example:
    >>> tokenizer = CodeTokenizer(split_identifiers=True)
    >>> tokenizer.tokenize("def getUserName(self):")
    ['def', 'getusername', 'get', 'user', 'name', '(', 'self', ')', ':']
"""

import re
from typing import List


class CodeTokenizer:
    """
    Code-aware tokenizer that preserves punctuation and operators.

    Unlike natural language tokenizers, this:
    - Keeps . ( ) [ ] { } : , as separate tokens
    - Preserves operators: == != >= <= += -= etc.
    - Splits camelCase and snake_case
    - Keeps string literals as single tokens (optionally)
    """

    # Operators to preserve as tokens
    OPERATORS = frozenset([
        '==', '!=', '>=', '<=', '+=', '-=', '*=', '/=', '//=', '%=',
        '**', '//', '->', '::', '...', '&&', '||', '<<', '>>', '**=',
        '&=', '|=', '^=', '>>=', '<<=', '@=',
    ])

    # Single-char punctuation to preserve
    PUNCTUATION = frozenset('.()[]{}:,;@#=+-*/<>|&^~%!')

    def __init__(self,
                 split_identifiers: bool = True,
                 preserve_case: bool = False,
                 include_strings: bool = False):
        """
        Initialize code tokenizer.

        Args:
            split_identifiers: Split camelCase/snake_case into parts
            preserve_case: Keep original case (default: lowercase)
            include_strings: Include string literal contents
        """
        self.split_identifiers = split_identifiers
        self.preserve_case = preserve_case
        self.include_strings = include_strings

        # Build operator pattern (longest first for correct matching)
        sorted_ops = sorted(self.OPERATORS, key=len, reverse=True)
        escaped_ops = [re.escape(op) for op in sorted_ops]
        self._op_pattern = re.compile('|'.join(escaped_ops))

        # Pattern for identifiers (including underscores)
        self._ident_pattern = re.compile(r'[a-zA-Z_][a-zA-Z0-9_]*')

        # Pattern for numbers
        self._num_pattern = re.compile(r'\d+\.?\d*')

        # Pattern for strings
        self._string_pattern = re.compile(
            r'(""".*?"""|\'\'\'.*?\'\'\'|".*?"|\'.*?\')', re.DOTALL
        )

    def tokenize(self, code: str) -> List[str]:
        """
        Tokenize code preserving structure.

        Args:
            code: Source code string

        Returns:
            List of tokens
        """
        tokens = []

        # Handle strings first (replace with placeholder to avoid tokenizing contents)
        string_map = {}
        if not self.include_strings:
            def replace_string(match):
                placeholder = f"__STRING_{len(string_map)}__"
                string_map[placeholder] = match.group(0)
                return placeholder
            code = self._string_pattern.sub(replace_string, code)

        # Split by whitespace first
        parts = code.split()

        for part in parts:
            tokens.extend(self._tokenize_part(part))

        # Optionally restore string placeholders
        if self.include_strings:
            tokens = [string_map.get(t, t) for t in tokens]
        else:
            # Remove string placeholders
            tokens = [t for t in tokens if not t.startswith('__STRING_')]

        return tokens

    def _tokenize_part(self, part: str) -> List[str]:
        """Tokenize a single whitespace-separated part."""
        result = []
        i = 0

        while i < len(part):
            # Check for multi-char operators
            matched_op = None
            for op_len in [3, 2]:  # Check 3-char then 2-char operators
                candidate = part[i:i+op_len]
                if candidate in self.OPERATORS:
                    matched_op = candidate
                    break

            if matched_op:
                result.append(matched_op)
                i += len(matched_op)
                continue

            # Check for single punctuation
            if part[i] in self.PUNCTUATION:
                result.append(part[i])
                i += 1
                continue

            # Check for identifier
            ident_match = self._ident_pattern.match(part, i)
            if ident_match:
                ident = ident_match.group(0)
                if self.split_identifiers:
                    result.extend(self._split_identifier(ident))
                else:
                    result.append(ident if self.preserve_case else ident.lower())
                i = ident_match.end()
                continue

            # Check for number
            num_match = self._num_pattern.match(part, i)
            if num_match:
                result.append(num_match.group(0))
                i = num_match.end()
                continue

            # Skip unknown character
            i += 1

        return result

    def _split_identifier(self, ident: str) -> List[str]:
        """Split camelCase and snake_case identifiers."""
        result = []

        # First add the full identifier
        full = ident if self.preserve_case else ident.lower()
        result.append(full)

        # Split on underscores
        parts = ident.split('_')

        for part in parts:
            if not part:
                continue

            # Split camelCase
            camel_parts = re.findall(
                r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+', part
            )

            for cp in camel_parts:
                lower = cp if self.preserve_case else cp.lower()
                if lower != full and lower not in result:
                    result.append(lower)

        return result

    def __repr__(self) -> str:
        return (
            f"CodeTokenizer(split_identifiers={self.split_identifiers}, "
            f"preserve_case={self.preserve_case}, "
            f"include_strings={self.include_strings})"
        )
