"""
CDG Query Language error hierarchy.

All errors include position tracking, source context, and suggestions
for debugging-quality error messages.

See: docs/design/cdg-query-language.md#exception-hierarchy
"""

from typing import List, Optional


class CDGQueryError(Exception):
    """
    Base class for all CDG query errors.

    All query errors support:
    - Position tracking for error location
    - Source context for showing where the error occurred
    - Suggestions for common fixes
    """

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


class QueryLexerError(CDGQueryError):
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


class QueryParseError(CDGQueryError):
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


class QueryValidationError(CDGQueryError):
    """Error during query validation (schema, field names, entity types)."""

    def __init__(
        self,
        message: str,
        position: Optional[int] = None,
        source: Optional[str] = None,
        field_name: Optional[str] = None,
        entity_type: Optional[str] = None,
        valid_fields: Optional[List[str]] = None,
        valid_entity_types: Optional[List[str]] = None,
        suggestions: Optional[List[str]] = None,
    ):
        self.field_name = field_name
        self.entity_type = entity_type
        self.valid_fields = valid_fields or []
        self.valid_entity_types = valid_entity_types or []

        # Auto-generate suggestions based on field/type similarity if not provided
        if suggestions is None:
            suggestions = []
            if field_name and valid_fields:
                suggestions.extend(self._suggest_similar(field_name, valid_fields))
            if entity_type and valid_entity_types:
                suggestions.extend(self._suggest_similar(entity_type, valid_entity_types))

        super().__init__(message, position, source, suggestions)

    def _suggest_similar(self, value: str, valid: List[str]) -> List[str]:
        """Suggest similar values using simple string similarity."""
        value_lower = value.lower()
        suggestions = []

        # Exact prefix matches first
        for v in valid:
            if v.lower().startswith(value_lower):
                suggestions.append(v)

        # Substring matches
        if not suggestions:
            for v in valid:
                if value_lower in v.lower() or v.lower() in value_lower:
                    suggestions.append(v)

        return suggestions[:3]

    def format_with_context(self) -> str:
        """Format error with valid fields/types list."""
        lines = [f"{self.__class__.__name__}: {self.message}"]

        if self.source and self.position is not None:
            lines.append(f"\n  {self.source}")
            lines.append(f"  {' ' * self.position}^^^")

        if self.valid_fields:
            lines.append(f"\n  Valid fields: {', '.join(sorted(self.valid_fields))}")

        if self.valid_entity_types:
            lines.append(f"\n  Valid entity types: {', '.join(sorted(self.valid_entity_types))}")

        if self.suggestions:
            lines.append("\n  Did you mean:")
            for s in self.suggestions:
                lines.append(f"    {s}")

        return "\n".join(lines)


class QueryPlanError(CDGQueryError):
    """Error during query planning (unsupported operators, etc.)."""

    def __init__(
        self,
        message: str,
        position: Optional[int] = None,
        source: Optional[str] = None,
        unsupported_feature: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
    ):
        self.unsupported_feature = unsupported_feature
        super().__init__(message, position, source, suggestions)


class QueryExecutionError(CDGQueryError):
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


class QueryNotImplementedError(CDGQueryError, NotImplementedError):
    """
    Feature is designed but not yet implemented.

    Use this for features described in the design doc that aren't ready yet.
    Always include a reference to the design doc section.

    Example:
        raise QueryNotImplementedError(
            "OR expressions require full scan optimization",
            doc_reference="docs/design/cdg-query-language.md#open-questions"
        )
    """

    def __init__(
        self,
        message: str,
        doc_reference: Optional[str] = None,
        position: Optional[int] = None,
        source: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
    ):
        self.doc_reference = doc_reference
        super().__init__(message, position, source, suggestions)

    def format_with_context(self) -> str:
        """Format error with doc reference."""
        lines = [f"{self.__class__.__name__}: {self.message}"]

        if self.doc_reference:
            lines.append(f"\n  See: {self.doc_reference}")

        if self.source and self.position is not None:
            lines.append(f"\n  {self.source}")
            lines.append(f"  {' ' * self.position}^^^")

        if self.suggestions:
            lines.append("\n  Workarounds:")
            for s in self.suggestions:
                lines.append(f"    {s}")

        return "\n".join(lines)
