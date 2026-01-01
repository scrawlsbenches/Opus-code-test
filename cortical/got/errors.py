"""
Exception classes for GoT (Graph of Thought) system.

All exceptions are designed to be text-friendly for JSON error messages
and CLI output.

Note on Error Hierarchy:
    After CDG/GoT unification, GoT's CorruptionError inherits from CDG's
    CorruptionError. This ensures that code catching GoT's CorruptionError
    will also catch exceptions raised by CDGStore operations.
"""

from typing import Optional, Dict, Any

# Import CDG error for inheritance
from cortical.cdg.errors import CorruptionError as CDGCorruptionError


class GoTError(Exception):
    """Base exception for all GoT errors."""

    def __init__(self, message: str, **context):
        """
        Initialize GoT error with message and optional context.

        Args:
            message: Human-readable error message
            **context: Additional context for debugging (must be JSON-serializable)
        """
        super().__init__(message)
        self.message = message
        self.context = context

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to JSON-serializable dictionary."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "context": self.context
        }


class TransactionError(GoTError):
    """Transaction-related errors (not active, already committed, etc.)."""
    pass


class ConflictError(GoTError):
    """Optimistic locking conflicts (version mismatch during concurrent updates)."""
    pass


# CorruptionError is re-exported from CDG for unified error handling.
# After CDG/GoT unification, CDGStore raises CDGCorruptionError, so GoT code
# that catches CorruptionError needs to catch the same class.
CorruptionError = CDGCorruptionError


class SyncError(GoTError):
    """Git sync errors (push rejected, pull failed, merge conflict)."""
    pass


class NotFoundError(GoTError):
    """Entity not found (task, decision, node, etc.)."""
    pass


class ValidationError(GoTError):
    """Invalid data (missing required fields, invalid enum values, etc.)."""
    pass
