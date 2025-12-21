"""
Tests for GoT exception classes.
"""

import unittest
from cortical.got.errors import (
    GoTError,
    TransactionError,
    ConflictError,
    CorruptionError,
    SyncError,
    NotFoundError,
    ValidationError
)


class TestGoTErrors(unittest.TestCase):
    """Test GoT exception classes."""

    def test_base_error_can_be_raised(self):
        """GoTError can be raised and caught."""
        with self.assertRaises(GoTError) as ctx:
            raise GoTError("Test error")

        self.assertEqual(str(ctx.exception), "Test error")
        self.assertEqual(ctx.exception.message, "Test error")

    def test_base_error_with_context(self):
        """GoTError can include context information."""
        with self.assertRaises(GoTError) as ctx:
            raise GoTError("Test error", task_id="T-123", node_id="N-456")

        self.assertEqual(ctx.exception.context["task_id"], "T-123")
        self.assertEqual(ctx.exception.context["node_id"], "N-456")

    def test_base_error_to_dict(self):
        """GoTError can be serialized to dictionary."""
        error = GoTError("Test error", task_id="T-123")
        result = error.to_dict()

        self.assertEqual(result["error_type"], "GoTError")
        self.assertEqual(result["message"], "Test error")
        self.assertEqual(result["context"]["task_id"], "T-123")

    def test_transaction_error_inheritance(self):
        """TransactionError inherits from GoTError."""
        self.assertTrue(issubclass(TransactionError, GoTError))

        with self.assertRaises(GoTError):
            raise TransactionError("Transaction not active")

    def test_transaction_error_can_be_raised(self):
        """TransactionError can be raised and caught."""
        with self.assertRaises(TransactionError) as ctx:
            raise TransactionError("Transaction already committed")

        self.assertEqual(str(ctx.exception), "Transaction already committed")

    def test_conflict_error_inheritance(self):
        """ConflictError inherits from GoTError."""
        self.assertTrue(issubclass(ConflictError, GoTError))

        with self.assertRaises(GoTError):
            raise ConflictError("Version mismatch")

    def test_conflict_error_can_be_raised(self):
        """ConflictError can be raised and caught."""
        with self.assertRaises(ConflictError) as ctx:
            raise ConflictError("Concurrent update detected", expected_version=1, actual_version=2)

        self.assertEqual(str(ctx.exception), "Concurrent update detected")
        self.assertEqual(ctx.exception.context["expected_version"], 1)
        self.assertEqual(ctx.exception.context["actual_version"], 2)

    def test_corruption_error_inheritance(self):
        """CorruptionError inherits from GoTError."""
        self.assertTrue(issubclass(CorruptionError, GoTError))

        with self.assertRaises(GoTError):
            raise CorruptionError("Checksum mismatch")

    def test_corruption_error_can_be_raised(self):
        """CorruptionError can be raised and caught."""
        with self.assertRaises(CorruptionError) as ctx:
            raise CorruptionError("Invalid event log", file_path="/path/to/events.jsonl")

        self.assertEqual(str(ctx.exception), "Invalid event log")
        self.assertEqual(ctx.exception.context["file_path"], "/path/to/events.jsonl")

    def test_sync_error_inheritance(self):
        """SyncError inherits from GoTError."""
        self.assertTrue(issubclass(SyncError, GoTError))

        with self.assertRaises(GoTError):
            raise SyncError("Git push rejected")

    def test_sync_error_can_be_raised(self):
        """SyncError can be raised and caught."""
        with self.assertRaises(SyncError) as ctx:
            raise SyncError("Merge conflict detected", branch="main")

        self.assertEqual(str(ctx.exception), "Merge conflict detected")
        self.assertEqual(ctx.exception.context["branch"], "main")

    def test_not_found_error_inheritance(self):
        """NotFoundError inherits from GoTError."""
        self.assertTrue(issubclass(NotFoundError, GoTError))

        with self.assertRaises(GoTError):
            raise NotFoundError("Task not found")

    def test_not_found_error_can_be_raised(self):
        """NotFoundError can be raised and caught."""
        with self.assertRaises(NotFoundError) as ctx:
            raise NotFoundError("Task not found", task_id="T-123")

        self.assertEqual(str(ctx.exception), "Task not found")
        self.assertEqual(ctx.exception.context["task_id"], "T-123")

    def test_validation_error_inheritance(self):
        """ValidationError inherits from GoTError."""
        self.assertTrue(issubclass(ValidationError, GoTError))

        with self.assertRaises(GoTError):
            raise ValidationError("Invalid status")

    def test_validation_error_can_be_raised(self):
        """ValidationError can be raised and caught."""
        with self.assertRaises(ValidationError) as ctx:
            raise ValidationError("Missing required field", field="title")

        self.assertEqual(str(ctx.exception), "Missing required field")
        self.assertEqual(ctx.exception.context["field"], "title")

    def test_all_exceptions_have_to_dict(self):
        """All exception types support to_dict() serialization."""
        exceptions = [
            GoTError("Base error", key="value"),
            TransactionError("Transaction error", tx_id="TX-123"),
            ConflictError("Conflict", version=1),
            CorruptionError("Corruption", checksum="abc"),
            SyncError("Sync error", remote="origin"),
            NotFoundError("Not found", entity_id="E-123"),
            ValidationError("Invalid", field="status")
        ]

        for exc in exceptions:
            result = exc.to_dict()
            self.assertIn("error_type", result)
            self.assertIn("message", result)
            self.assertIn("context", result)
            self.assertIsInstance(result["context"], dict)

    def test_exception_string_representation(self):
        """All exceptions have proper string representation."""
        error = TransactionError("Transaction failed")
        self.assertEqual(str(error), "Transaction failed")

        # String representation should match message
        self.assertEqual(str(error), error.message)

    def test_exception_context_is_optional(self):
        """Exceptions work without context parameters."""
        exceptions = [
            GoTError("Error"),
            TransactionError("Error"),
            ConflictError("Error"),
            CorruptionError("Error"),
            SyncError("Error"),
            NotFoundError("Error"),
            ValidationError("Error")
        ]

        for exc in exceptions:
            self.assertEqual(exc.context, {})
            result = exc.to_dict()
            self.assertEqual(result["context"], {})


if __name__ == '__main__':
    unittest.main()
