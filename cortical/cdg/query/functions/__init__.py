"""
Core CDG query functions.

These functions are available for all CDG queries and provide
basic operations like counting, existence checks, and type introspection.

Extension modules (like GoT) can register additional functions
using the FunctionRegistry.

See: docs/design/cdg-query-language.md
"""

from .core import (
    CountFunction,
    ExistsFunction,
    TypeOfFunction,
    FieldsFunction,
    EntityTypesFunction,
)

__all__ = [
    'CountFunction',
    'ExistsFunction',
    'TypeOfFunction',
    'FieldsFunction',
    'EntityTypesFunction',
]
