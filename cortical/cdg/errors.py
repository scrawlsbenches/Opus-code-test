"""
CDG-specific error types.

Provides a hierarchy of errors for the Cortical Distributed Graph system,
enabling precise error handling and informative error messages.

Error Hierarchy:
    CDGError (base)
    ├── ValidationError - Schema/constraint violations
    ├── CorruptionError - Data integrity failures
    ├── TransactionError - Transaction lifecycle errors
    │   └── ConflictError - Optimistic locking conflicts
    ├── PartitionError - Partition routing/management errors
    └── StorageError - Low-level storage failures
"""

from typing import Any, Dict, List, Optional


class CDGError(Exception):
    """
    Base exception for all CDG errors.

    All CDG-specific errors inherit from this class, enabling
    broad exception handling when needed.

    Example:
        try:
            store.write(entity)
        except CDGError as e:
            logger.error(f"CDG operation failed: {e}")
    """

    def __init__(self, message: str, **context: Any):
        """
        Initialize error with message and optional context.

        Args:
            message: Human-readable error description
            **context: Additional context key-value pairs
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


class ValidationError(CDGError):
    """
    Raised when entity data fails validation.

    This includes schema validation failures, constraint violations,
    and invalid field values.

    Attributes:
        entity_type: Type of entity being validated (if known)
        errors: List of specific validation errors

    Example:
        raise ValidationError(
            "Invalid edge_type 'INVALID'",
            edge_type="INVALID",
            valid_types=list(VALID_EDGE_TYPES)
        )
    """

    def __init__(
        self,
        message: str,
        entity_type: Optional[str] = None,
        errors: Optional[List[str]] = None,
        **context: Any
    ):
        super().__init__(message, **context)
        self.entity_type = entity_type
        self.errors = errors or []


class CorruptionError(CDGError):
    """
    Raised when data integrity verification fails.

    This indicates that stored data has been corrupted or tampered with,
    typically detected via checksum verification failures.

    Attributes:
        entity_id: ID of corrupted entity (if known)
        expected_checksum: Expected checksum value
        actual_checksum: Actual computed checksum

    Example:
        raise CorruptionError(
            "Checksum mismatch",
            entity_id="E-001",
            expected_checksum="abc123",
            actual_checksum="def456"
        )
    """

    def __init__(
        self,
        message: str,
        entity_id: Optional[str] = None,
        expected_checksum: Optional[str] = None,
        actual_checksum: Optional[str] = None,
        **context: Any
    ):
        super().__init__(message, **context)
        self.entity_id = entity_id
        self.expected_checksum = expected_checksum
        self.actual_checksum = actual_checksum


class TransactionError(CDGError):
    """
    Raised when a transaction operation fails.

    This covers transaction lifecycle errors such as attempting
    to commit an already-aborted transaction.

    Attributes:
        tx_id: Transaction ID
        state: Current transaction state

    Example:
        raise TransactionError(
            "Cannot commit aborted transaction",
            tx_id="TX-20251231-120000-abc123",
            state="aborted"
        )
    """

    def __init__(
        self,
        message: str,
        tx_id: Optional[str] = None,
        state: Optional[str] = None,
        **context: Any
    ):
        super().__init__(message, **context)
        self.tx_id = tx_id
        self.state = state


class ConflictError(TransactionError):
    """
    Raised when optimistic locking detects a conflict.

    This occurs when an entity was modified by another transaction
    between read and commit, violating snapshot isolation.

    Attributes:
        entity_id: ID of conflicting entity
        read_version: Version when entity was read
        current_version: Current version in store

    Example:
        raise ConflictError(
            "Entity modified by concurrent transaction",
            tx_id="TX-20251231-120000-abc123",
            entity_id="E-001",
            read_version=5,
            current_version=6
        )
    """

    def __init__(
        self,
        message: str,
        entity_id: Optional[str] = None,
        read_version: Optional[int] = None,
        current_version: Optional[int] = None,
        **context: Any
    ):
        super().__init__(message, **context)
        self.entity_id = entity_id
        self.read_version = read_version
        self.current_version = current_version


class PartitionError(CDGError):
    """
    Raised when partition operations fail.

    This includes routing failures, partition not found errors,
    and partition management issues.

    Attributes:
        partition_id: ID of affected partition
        operation: Operation that failed

    Example:
        raise PartitionError(
            "Partition not found",
            partition_id=5,
            operation="read"
        )
    """

    def __init__(
        self,
        message: str,
        partition_id: Optional[int] = None,
        operation: Optional[str] = None,
        **context: Any
    ):
        super().__init__(message, **context)
        self.partition_id = partition_id
        self.operation = operation


class StorageError(CDGError):
    """
    Raised when low-level storage operations fail.

    This covers I/O errors, file system issues, and other
    storage-layer failures.

    Attributes:
        path: File/directory path involved
        operation: Storage operation that failed

    Example:
        raise StorageError(
            "Failed to write entity file",
            path="/data/entities/E-001.json",
            operation="write"
        )
    """

    def __init__(
        self,
        message: str,
        path: Optional[str] = None,
        operation: Optional[str] = None,
        **context: Any
    ):
        super().__init__(message, **context)
        self.path = path
        self.operation = operation
