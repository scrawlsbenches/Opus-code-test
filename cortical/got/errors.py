"""
Exception classes for GoT (Graph of Thought) system.

All exceptions are designed to be text-friendly for JSON error messages
and CLI output.
"""

from typing import Optional, Dict, Any


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


class CorruptionError(GoTError):
    """Data corruption detected (checksum mismatch, invalid event log, etc.)."""
    pass


class SyncError(GoTError):
    """Git sync errors (push rejected, pull failed, merge conflict)."""
    pass


class NotFoundError(GoTError):
    """Entity not found (task, decision, node, etc.)."""
    pass


class ValidationError(GoTError):
    """Invalid data (missing required fields, invalid enum values, etc.)."""
    pass
