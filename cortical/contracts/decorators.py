"""
Contract decorators for Design-by-Contract programming.

These decorators define executable specifications:
- @requires: Preconditions that must hold BEFORE method execution
- @ensures: Postconditions that must hold AFTER method execution
- @invariant: Class invariants that must hold before AND after

All decorators:
1. Check their conditions at runtime
2. Raise ContractViolation on failure
3. Support both lambda predicates and string expressions
4. Can emit CEL events when integrated with ContractEventEmitter
"""

from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TypeVar, Union

F = TypeVar('F', bound=Callable[..., Any])


class ContractViolation(Exception):
    """
    Raised when a contract is violated.

    This is distinct from ValueError/AssertionError because it indicates
    a fundamental misunderstanding of system invariants, not a bug.

    Attributes:
        contract_type: 'requires', 'ensures', or 'invariant'
        description: Human-readable contract description
        context: Additional context (method name, arguments, etc.)
    """

    def __init__(
        self,
        contract_type: str,
        description: str,
        context: Optional[dict] = None,
    ):
        self.contract_type = contract_type
        self.description = description
        self.context = context or {}

        method = self.context.get('method', '<unknown>')
        super().__init__(
            f"Contract violation ({contract_type}): {description} "
            f"[method: {method}]"
        )

    def to_dict(self) -> dict:
        """Serialize for CEL event emission."""
        return {
            'contract_type': self.contract_type,
            'description': self.description,
            'context': self.context,
        }


@dataclass
class ContractSpec:
    """
    Specification for a single contract.

    Used internally to track contract metadata for registry integration.
    """

    contract_type: str  # 'requires', 'ensures', 'invariant'
    predicate: Callable[..., bool]
    description: str
    method_name: Optional[str] = None
    class_name: Optional[str] = None
    source_file: Optional[str] = None
    source_line: Optional[int] = None

    # Event emitter for CEL integration (injected by registry)
    _emitter: Optional[Any] = field(default=None, repr=False)

    def check(self, *args, **kwargs) -> bool:
        """
        Evaluate the predicate with given arguments.

        Handles predicates with different arities - only passes
        as many positional arguments as the predicate expects.
        """
        # Inspect predicate to determine how many args it expects
        try:
            sig = inspect.signature(self.predicate)
            params = [p for p in sig.parameters.values()
                     if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                  inspect.Parameter.POSITIONAL_OR_KEYWORD)]
            # Filter out *args and **kwargs style params
            expected_count = len(params)
            # Only pass as many args as predicate expects
            trimmed_args = args[:expected_count]
            return self.predicate(*trimmed_args, **kwargs)
        except (ValueError, TypeError):
            # Fall back to passing all args
            return self.predicate(*args, **kwargs)

    def emit_check(self, passed: bool, context: dict) -> None:
        """Emit CEL event for this contract check."""
        if self._emitter is not None:
            self._emitter.emit_check(self, passed, context)


def requires(
    predicate: Union[Callable[..., bool], str],
    description: str = "",
) -> Callable[[F], F]:
    """
    Decorator for preconditions.

    The predicate is checked BEFORE the method executes.
    If it returns False, ContractViolation is raised.

    Args:
        predicate: Callable that returns bool, or string expression
        description: Human-readable description of the contract

    Example:
        @requires(lambda self: not self._closed, "Connection must be open")
        def send(self, data):
            ...

        @requires(lambda self, amount: amount > 0, "Amount must be positive")
        def withdraw(self, amount):
            ...
    """
    if isinstance(predicate, str):
        description = description or predicate
        predicate = _compile_expression(predicate)

    def decorator(func: F) -> F:
        spec = _create_spec('requires', predicate, description, func)
        existing = getattr(func, '_contract_specs', [])

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Build context for error reporting
            context = _build_context(func, args, kwargs)

            # Check precondition
            try:
                passed = spec.check(*args, **kwargs)
            except Exception as e:
                # Predicate evaluation failed
                raise ContractViolation(
                    'requires',
                    f"{spec.description} (evaluation error: {e})",
                    context,
                ) from e

            spec.emit_check(passed, context)

            if not passed:
                raise ContractViolation('requires', spec.description, context)

            return func(*args, **kwargs)

        wrapper._contract_specs = existing + [spec]
        return wrapper  # type: ignore

    return decorator


def ensures(
    predicate: Union[Callable[..., bool], str],
    description: str = "",
) -> Callable[[F], F]:
    """
    Decorator for postconditions.

    The predicate is checked AFTER the method executes.
    It receives (self, result, *args, **kwargs) where result is the return value.

    Args:
        predicate: Callable that returns bool (receives result as 2nd arg)
        description: Human-readable description of the contract

    Example:
        @ensures(lambda self, result: result is not None, "Must return a value")
        def get(self, key):
            ...

        @ensures(lambda self, result: len(self._items) > 0, "Items must not be empty after add")
        def add(self, item):
            ...
    """
    if isinstance(predicate, str):
        description = description or predicate
        predicate = _compile_expression(predicate, include_result=True)

    def decorator(func: F) -> F:
        spec = _create_spec('ensures', predicate, description, func)
        existing = getattr(func, '_contract_specs', [])

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Execute the method first
            result = func(*args, **kwargs)

            # Build context
            context = _build_context(func, args, kwargs)
            context['result'] = repr(result)[:100]  # Truncate for safety

            # Check postcondition with result
            try:
                # Postcondition receives (self, result, *rest_args)
                if args:
                    check_args = (args[0], result) + args[1:]
                else:
                    check_args = (result,)
                passed = spec.check(*check_args, **kwargs)
            except Exception as e:
                raise ContractViolation(
                    'ensures',
                    f"{spec.description} (evaluation error: {e})",
                    context,
                ) from e

            spec.emit_check(passed, context)

            if not passed:
                raise ContractViolation('ensures', spec.description, context)

            return result

        wrapper._contract_specs = existing + [spec]
        return wrapper  # type: ignore

    return decorator


def invariant(
    predicate: Union[Callable[[Any], bool], str],
    description: str = "",
) -> Callable[[F], F]:
    """
    Decorator for class invariants.

    The predicate is checked BEFORE and AFTER the method executes.
    It receives only (self,) - invariants apply to the object state.

    Args:
        predicate: Callable that takes self and returns bool
        description: Human-readable description of the invariant

    Example:
        @invariant(lambda self: self._count >= 0, "Count must be non-negative")
        def decrement(self):
            self._count -= 1

        @invariant(lambda self: len(self._wal) >= len(self._committed),
                   "WAL entries must be >= committed entities")
        def commit(self):
            ...
    """
    if isinstance(predicate, str):
        description = description or predicate
        predicate = _compile_expression(predicate, invariant_style=True)

    def decorator(func: F) -> F:
        spec = _create_spec('invariant', predicate, description, func)
        existing = getattr(func, '_contract_specs', [])

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            context = _build_context(func, args, kwargs)

            # Check invariant BEFORE
            if args:
                self_arg = args[0]
                try:
                    passed_before = spec.check(self_arg)
                except Exception as e:
                    raise ContractViolation(
                        'invariant',
                        f"{spec.description} (pre-check error: {e})",
                        {**context, 'phase': 'before'},
                    ) from e

                spec.emit_check(passed_before, {**context, 'phase': 'before'})

                if not passed_before:
                    raise ContractViolation(
                        'invariant',
                        f"{spec.description} (violated before method)",
                        {**context, 'phase': 'before'},
                    )

            # Execute method
            result = func(*args, **kwargs)

            # Check invariant AFTER
            if args:
                try:
                    passed_after = spec.check(self_arg)
                except Exception as e:
                    raise ContractViolation(
                        'invariant',
                        f"{spec.description} (post-check error: {e})",
                        {**context, 'phase': 'after'},
                    ) from e

                spec.emit_check(passed_after, {**context, 'phase': 'after'})

                if not passed_after:
                    raise ContractViolation(
                        'invariant',
                        f"{spec.description} (violated after method)",
                        {**context, 'phase': 'after'},
                    )

            return result

        wrapper._contract_specs = existing + [spec]
        return wrapper  # type: ignore

    return decorator


def _create_spec(
    contract_type: str,
    predicate: Callable[..., bool],
    description: str,
    func: Callable,
) -> ContractSpec:
    """Create a ContractSpec with source location info."""
    try:
        source_file = inspect.getsourcefile(func)
        source_lines = inspect.getsourcelines(func)
        source_line = source_lines[1] if source_lines else None
    except (TypeError, OSError):
        source_file = None
        source_line = None

    return ContractSpec(
        contract_type=contract_type,
        predicate=predicate,
        description=description,
        method_name=func.__name__,
        class_name=func.__qualname__.rsplit('.', 1)[0] if '.' in func.__qualname__ else None,
        source_file=source_file,
        source_line=source_line,
    )


def _build_context(func: Callable, args: tuple, kwargs: dict) -> dict:
    """Build context dictionary for error reporting."""
    context = {
        'method': func.__qualname__,
        'module': func.__module__,
    }

    # Add argument info (truncated for safety)
    if args:
        context['args_count'] = len(args)
        if hasattr(args[0], '__class__'):
            context['self_class'] = args[0].__class__.__name__

    if kwargs:
        context['kwargs_keys'] = list(kwargs.keys())

    return context


def _compile_expression(
    expr: str,
    include_result: bool = False,
    invariant_style: bool = False,
) -> Callable[..., bool]:
    """
    Compile a string expression into a callable predicate.

    This allows contracts to be specified as strings for readability:
        @requires("amount > 0")
        @ensures("result is not None")
        @invariant("self._count >= 0")

    Security note: This uses eval() with restricted globals.
    Only use with trusted input (source code contracts).
    """
    if invariant_style:
        # Invariant: receives (self,)
        def predicate(self):
            return eval(expr, {'self': self, '__builtins__': {}})
        return predicate
    elif include_result:
        # Postcondition: receives (self, result, ...)
        def predicate(self, result, *args, **kwargs):
            return eval(expr, {'self': self, 'result': result, '__builtins__': {}})
        return predicate
    else:
        # Precondition: receives (self, ...)
        def predicate(self, *args, **kwargs):
            local_vars = {'self': self, '__builtins__': {}}
            # Try to bind positional args to parameter names
            return eval(expr, local_vars)
        return predicate
