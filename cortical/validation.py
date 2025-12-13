"""
Validation Module
=================

Input validation utilities and decorators for the Cortical Text Processor.

This module provides reusable validators and decorators to ensure
parameters are valid before processing, reducing boilerplate validation
code throughout the codebase.

Staleness Tracking Decorators
------------------------------

The `marks_stale` and `marks_fresh` decorators are available for automating
staleness tracking. Currently, processor.py uses manual staleness tracking
via _mark_fresh() and _mark_all_stale() calls, which provides explicit
control and conditional logic.

Future refactoring could use decorators like:

    @marks_fresh(COMP_TFIDF)
    def compute_tfidf(self, verbose=False):
        # TF-IDF computation
        pass

    @marks_stale(COMP_TFIDF, COMP_PAGERANK)
    def process_document(self, doc_id, text):
        # Document processing
        pass

However, note that some methods have conditional staleness marking based
on parameters (e.g., add_document_incremental with recompute='none' vs 'tfidf'),
so manual tracking may be more appropriate in those cases.
"""

from typing import Any, Callable, Optional, TypeVar, Union
from functools import wraps
import inspect


def validate_non_empty_string(value: Any, param_name: str) -> None:
    """
    Validate that a value is a non-empty string.

    Args:
        value: The value to validate
        param_name: Name of the parameter (for error messages)

    Raises:
        ValueError: If value is not a non-empty string
    """
    if not isinstance(value, str):
        raise ValueError(f"{param_name} must be a string, got {type(value).__name__}")
    if not value:
        raise ValueError(f"{param_name} must be a non-empty string")


def validate_positive_int(value: Any, param_name: str) -> None:
    """
    Validate that a value is a positive integer.

    Args:
        value: The value to validate
        param_name: Name of the parameter (for error messages)

    Raises:
        ValueError: If value is not a positive integer
    """
    if not isinstance(value, int):
        raise ValueError(f"{param_name} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{param_name} must be positive, got {value}")


def validate_non_negative_int(value: Any, param_name: str) -> None:
    """
    Validate that a value is a non-negative integer.

    Args:
        value: The value to validate
        param_name: Name of the parameter (for error messages)

    Raises:
        ValueError: If value is not a non-negative integer
    """
    if not isinstance(value, int):
        raise ValueError(f"{param_name} must be an integer, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{param_name} must be non-negative, got {value}")


def validate_range(value: Any, param_name: str, min_val: Optional[float] = None,
                   max_val: Optional[float] = None, inclusive: bool = True) -> None:
    """
    Validate that a numeric value is within a specified range.

    Args:
        value: The value to validate
        param_name: Name of the parameter (for error messages)
        min_val: Minimum allowed value (None for no minimum)
        max_val: Maximum allowed value (None for no maximum)
        inclusive: Whether endpoints are inclusive (default True)

    Raises:
        ValueError: If value is outside the specified range
    """
    if not isinstance(value, (int, float)):
        raise ValueError(f"{param_name} must be numeric, got {type(value).__name__}")

    if min_val is not None:
        if inclusive and value < min_val:
            raise ValueError(f"{param_name} must be >= {min_val}, got {value}")
        elif not inclusive and value <= min_val:
            raise ValueError(f"{param_name} must be > {min_val}, got {value}")

    if max_val is not None:
        if inclusive and value > max_val:
            raise ValueError(f"{param_name} must be <= {max_val}, got {value}")
        elif not inclusive and value >= max_val:
            raise ValueError(f"{param_name} must be < {max_val}, got {value}")


# Type variable for decorator return type
F = TypeVar('F', bound=Callable[..., Any])


def validate_params(**validators: Callable[[Any], None]) -> Callable[[F], F]:
    """
    Decorator to validate function parameters.

    Args:
        **validators: Mapping of parameter names to validation functions.
                     Each validator should take one argument and raise
                     ValueError if validation fails.

    Returns:
        Decorated function with parameter validation

    Example:
        >>> @validate_params(
        ...     query=lambda q: validate_non_empty_string(q, 'query'),
        ...     top_n=lambda n: validate_positive_int(n, 'top_n')
        ... )
        ... def search(query: str, top_n: int = 5):
        ...     return f"Searching for '{query}', top {top_n} results"
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            sig = inspect.signature(func)
            bound_args = sig.bind_partial(*args, **kwargs)
            bound_args.apply_defaults()

            # Validate specified parameters
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    # Skip None values for optional parameters
                    if value is not None:
                        validator(value)

            return func(*args, **kwargs)
        return wrapper  # type: ignore
    return decorator


def marks_stale(*computation_types: str) -> Callable[[F], F]:
    """
    Decorator to mark computations as stale after method execution.

    This is used on methods that modify data and invalidate cached
    computations. The decorated method must be an instance method
    of a class that has a _mark_all_stale() method.

    Args:
        *computation_types: Computation type constants to mark as stale.
                           If empty, marks all computations stale.

    Returns:
        Decorated function that marks specified computations stale

    Example:
        >>> class Processor:
        ...     def _mark_all_stale(self): pass
        ...
        ...     @marks_stale('tfidf', 'pagerank')
        ...     def add_document(self, doc_id, text):
        ...         # Document addition logic
        ...         pass
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)

            # Mark computations as stale
            if computation_types:
                # Mark specific computations stale
                # Use the private _stale_computations set directly
                if hasattr(self, '_stale_computations'):
                    for comp_type in computation_types:
                        self._stale_computations.add(comp_type)
            else:
                # Mark all computations stale
                if hasattr(self, '_mark_all_stale'):
                    self._mark_all_stale()

            return result
        return wrapper  # type: ignore
    return decorator


def marks_fresh(*computation_types: str) -> Callable[[F], F]:
    """
    Decorator to mark computations as fresh after method execution.

    This is used on methods that compute derived data. The decorated
    method must be an instance method of a class that has a _mark_fresh()
    method.

    Args:
        *computation_types: Computation type constants to mark as fresh

    Returns:
        Decorated function that marks specified computations fresh

    Example:
        >>> class Processor:
        ...     def _mark_fresh(self, *types): pass
        ...
        ...     @marks_fresh('tfidf')
        ...     def compute_tfidf(self):
        ...         # TF-IDF computation logic
        ...         pass
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)

            # Mark computations as fresh
            if hasattr(self, '_mark_fresh'):
                self._mark_fresh(*computation_types)

            return result
        return wrapper  # type: ignore
    return decorator
