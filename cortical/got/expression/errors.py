"""
Custom exception types for the expression parser.

All errors include position tracking, source context, and suggestions
for debugging-quality error messages.
"""

from typing import List, Optional


class QueryError(Exception):
    """Base class for all query expression errors."""

    def __init__(
        self,
        message: str,
        position: Optional[int] = None,
        source: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
    ):
        self.message = message
        self.position = position
        self.source = source
        self.suggestions = suggestions or []
        super().__init__(self.format_with_context())

    def format_with_context(self) -> str:
        """Format error with source context and suggestions."""
        lines = [f"{self.__class__.__name__}: {self.message}"]

        if self.source and self.position is not None:
            lines.append(f"\n  {self.source}")
            lines.append(f"  {' ' * self.position}^^^")

        if self.suggestions:
            lines.append("\n  Did you mean:")
            for s in self.suggestions:
                lines.append(f"    {s}")

        return "\n".join(lines)


class LexerError(QueryError):
    """Error during tokenization."""

    def __init__(
        self,
        message: str,
        position: Optional[int] = None,
        source: Optional[str] = None,
        invalid_character: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
    ):
        self.invalid_character = invalid_character
        super().__init__(message, position, source, suggestions)


class ParseError(QueryError):
    """Error during parsing."""

    def __init__(
        self,
        message: str,
        position: Optional[int] = None,
        source: Optional[str] = None,
        expected_tokens: Optional[List[str]] = None,
        found_token: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
    ):
        self.expected_tokens = expected_tokens or []
        self.found_token = found_token
        super().__init__(message, position, source, suggestions)

    def format_with_context(self) -> str:
        """Format error with expected tokens and source context."""
        lines = [f"{self.__class__.__name__}: {self.message}"]

        if self.source and self.position is not None:
            lines.append(f"\n  {self.source}")
            lines.append(f"  {' ' * self.position}^^^")

        if self.expected_tokens:
            lines.append(f"\n  Expected: {', '.join(self.expected_tokens)}")

        if self.suggestions:
            lines.append("\n  Did you mean:")
            for s in self.suggestions:
                lines.append(f"    {s}")

        return "\n".join(lines)


class ExecutionError(QueryError):
    """Error during query execution."""

    def __init__(
        self,
        message: str,
        position: Optional[int] = None,
        source: Optional[str] = None,
        function_name: Optional[str] = None,
        available_functions: Optional[List[str]] = None,
        suggestions: Optional[List[str]] = None,
    ):
        self.function_name = function_name
        self.available_functions = available_functions or []
        super().__init__(message, position, source, suggestions)

    def format_with_context(self) -> str:
        """Format error with available functions list."""
        lines = [f"{self.__class__.__name__}: {self.message}"]

        if self.source and self.position is not None:
            lines.append(f"\n  {self.source}")
            lines.append(f"  {' ' * self.position}^^^")

        if self.available_functions:
            lines.append(f"\n  Available functions: {', '.join(sorted(self.available_functions))}")

        if self.suggestions:
            lines.append("\n  Did you mean:")
            for s in self.suggestions:
                lines.append(f"    {s}")

        return "\n".join(lines)


class QueryValidationError(QueryError):
    """Error during query validation (schema, field names, etc.)."""

    def __init__(
        self,
        message: str,
        position: Optional[int] = None,
        source: Optional[str] = None,
        field_name: Optional[str] = None,
        valid_fields: Optional[List[str]] = None,
        suggestions: Optional[List[str]] = None,
    ):
        self.field_name = field_name
        self.valid_fields = valid_fields or []

        # Auto-generate suggestions based on field similarity if not provided
        if suggestions is None and field_name and valid_fields:
            suggestions = self._suggest_similar_fields(field_name, valid_fields)

        super().__init__(message, position, source, suggestions)

    def _suggest_similar_fields(self, field: str, valid: List[str]) -> List[str]:
        """Suggest similar field names using simple string similarity."""
        field_lower = field.lower()
        suggestions = []

        # Exact prefix matches first
        for v in valid:
            if v.lower().startswith(field_lower):
                suggestions.append(v)

        # Substring matches
        if not suggestions:
            for v in valid:
                if field_lower in v.lower() or v.lower() in field_lower:
                    suggestions.append(v)

        # Return up to 3 suggestions
        return suggestions[:3]

    def format_with_context(self) -> str:
        """Format error with valid fields list."""
        lines = [f"{self.__class__.__name__}: {self.message}"]

        if self.source and self.position is not None:
            lines.append(f"\n  {self.source}")
            lines.append(f"  {' ' * self.position}^^^")

        if self.valid_fields:
            lines.append(f"\n  Valid fields: {', '.join(sorted(self.valid_fields))}")

        if self.suggestions:
            lines.append("\n  Did you mean:")
            for s in self.suggestions:
                lines.append(f"    {s}")

        return "\n".join(lines)
