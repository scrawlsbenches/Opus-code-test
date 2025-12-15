#!/usr/bin/env python3
"""
Unit tests for ML data collector feedback functionality.

Tests the add_chat_feedback() and list_chats_needing_feedback() functions,
as well as schema validation for feedback data.
"""

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

import ml_data_collector as ml


class TestAddChatFeedback(unittest.TestCase):
    """Test add_chat_feedback() function."""

    def setUp(self):
        """Set up test environment with temporary directories."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_path = Path(self.test_dir.name)

        # Temporarily override ML data directories
        self.original_ml_data_dir = ml.ML_DATA_DIR
        self.original_chats_dir = ml.CHATS_DIR

        ml.ML_DATA_DIR = self.test_path / ".git-ml"
        ml.CHATS_DIR = ml.ML_DATA_DIR / "chats"

        # Create test directories
        ml.ensure_dirs()

    def tearDown(self):
        """Clean up test environment."""
        # Restore original directories
        ml.ML_DATA_DIR = self.original_ml_data_dir
        ml.CHATS_DIR = self.original_chats_dir

        # Clean up temp directory
        self.test_dir.cleanup()

    def _create_test_chat(self, chat_id: str, user_feedback=None) -> Path:
        """Helper to create a test chat file."""
        # Create date directory
        date_str = datetime.now().strftime("%Y-%m-%d")
        date_dir = ml.CHATS_DIR / date_str
        date_dir.mkdir(parents=True, exist_ok=True)

        # Create chat file
        chat_data = {
            "id": chat_id,
            "timestamp": datetime.now().isoformat(),
            "session_id": "test-session",
            "query": "Test query",
            "response": "Test response",
            "files_referenced": [],
            "files_modified": [],
            "tools_used": ["Read"],
        }

        if user_feedback is not None:
            chat_data["user_feedback"] = user_feedback

        chat_file = date_dir / f"{chat_id}.json"
        with open(chat_file, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f)

        return chat_file

    def test_add_feedback_to_new_chat(self):
        """Test adding feedback to a chat without existing feedback."""
        chat_id = "chat-test-001"
        self._create_test_chat(chat_id)

        # Add feedback
        result = ml.add_chat_feedback(chat_id, "good", "Great response!")

        # Should succeed
        self.assertTrue(result)

        # Verify feedback was added
        chat_file = ml.find_chat_file(chat_id)
        with open(chat_file, 'r', encoding='utf-8') as f:
            chat_data = json.load(f)

        self.assertIn("user_feedback", chat_data)
        self.assertIsInstance(chat_data["user_feedback"], dict)
        self.assertEqual(chat_data["user_feedback"]["rating"], "good")
        self.assertEqual(chat_data["user_feedback"]["comment"], "Great response!")
        self.assertIn("timestamp", chat_data["user_feedback"])

    def test_add_feedback_minimal(self):
        """Test adding feedback without comment."""
        chat_id = "chat-test-002"
        self._create_test_chat(chat_id)

        # Add feedback without comment
        result = ml.add_chat_feedback(chat_id, "neutral")

        # Should succeed
        self.assertTrue(result)

        # Verify feedback
        chat_file = ml.find_chat_file(chat_id)
        with open(chat_file, 'r', encoding='utf-8') as f:
            chat_data = json.load(f)

        self.assertEqual(chat_data["user_feedback"]["rating"], "neutral")
        self.assertIsNone(chat_data["user_feedback"]["comment"])

    def test_add_feedback_all_ratings(self):
        """Test all valid rating values."""
        valid_ratings = ["good", "bad", "neutral"]

        for i, rating in enumerate(valid_ratings):
            chat_id = f"chat-test-rating-{i}"
            self._create_test_chat(chat_id)

            result = ml.add_chat_feedback(chat_id, rating)
            self.assertTrue(result)

            # Verify rating
            chat_file = ml.find_chat_file(chat_id)
            with open(chat_file, 'r', encoding='utf-8') as f:
                chat_data = json.load(f)

            self.assertEqual(chat_data["user_feedback"]["rating"], rating)

    def test_add_feedback_invalid_chat_id(self):
        """Test adding feedback to non-existent chat."""
        result = ml.add_chat_feedback("nonexistent-chat-id", "good")

        # Should return False (chat not found)
        self.assertFalse(result)

    def test_add_feedback_invalid_rating(self):
        """Test adding feedback with invalid rating value."""
        chat_id = "chat-test-003"
        self._create_test_chat(chat_id)

        # Should raise ValueError
        with self.assertRaises(ValueError) as ctx:
            ml.add_chat_feedback(chat_id, "excellent")

        self.assertIn("Invalid rating", str(ctx.exception))
        self.assertIn("excellent", str(ctx.exception))

    def test_add_feedback_overwrite_without_force(self):
        """Test that overwriting existing feedback without force fails."""
        chat_id = "chat-test-004"
        existing_feedback = {
            "rating": "good",
            "comment": "Original feedback",
            "timestamp": datetime.now().isoformat(),
        }
        self._create_test_chat(chat_id, user_feedback=existing_feedback)

        # Try to update without force
        result = ml.add_chat_feedback(chat_id, "bad", "New feedback")

        # Should fail
        self.assertFalse(result)

        # Verify original feedback unchanged
        chat_file = ml.find_chat_file(chat_id)
        with open(chat_file, 'r', encoding='utf-8') as f:
            chat_data = json.load(f)

        self.assertEqual(chat_data["user_feedback"]["rating"], "good")
        self.assertEqual(chat_data["user_feedback"]["comment"], "Original feedback")

    def test_add_feedback_overwrite_with_force(self):
        """Test overwriting existing feedback with force=True."""
        chat_id = "chat-test-005"
        existing_feedback = {
            "rating": "good",
            "comment": "Original feedback",
            "timestamp": datetime.now().isoformat(),
        }
        self._create_test_chat(chat_id, user_feedback=existing_feedback)

        # Update with force
        result = ml.add_chat_feedback(chat_id, "bad", "Updated feedback", force=True)

        # Should succeed
        self.assertTrue(result)

        # Verify feedback was updated
        chat_file = ml.find_chat_file(chat_id)
        with open(chat_file, 'r', encoding='utf-8') as f:
            chat_data = json.load(f)

        self.assertEqual(chat_data["user_feedback"]["rating"], "bad")
        self.assertEqual(chat_data["user_feedback"]["comment"], "Updated feedback")

    def test_add_feedback_upgrade_legacy_format(self):
        """Test upgrading from legacy string format to dict format."""
        chat_id = "chat-test-006"
        # Create chat with legacy string feedback
        self._create_test_chat(chat_id, user_feedback="good")

        # Try to add feedback (should allow upgrade from legacy format)
        result = ml.add_chat_feedback(chat_id, "neutral", "Upgraded feedback")

        # Should succeed (legacy format allows implicit upgrade)
        self.assertTrue(result)

        # Verify feedback is now in dict format
        chat_file = ml.find_chat_file(chat_id)
        with open(chat_file, 'r', encoding='utf-8') as f:
            chat_data = json.load(f)

        self.assertIsInstance(chat_data["user_feedback"], dict)
        self.assertEqual(chat_data["user_feedback"]["rating"], "neutral")
        self.assertEqual(chat_data["user_feedback"]["comment"], "Upgraded feedback")


class TestListChatsNeedingFeedback(unittest.TestCase):
    """Test list_chats_needing_feedback() function."""

    def setUp(self):
        """Set up test environment with temporary directories."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_path = Path(self.test_dir.name)

        # Temporarily override ML data directories
        self.original_ml_data_dir = ml.ML_DATA_DIR
        self.original_chats_dir = ml.CHATS_DIR

        ml.ML_DATA_DIR = self.test_path / ".git-ml"
        ml.CHATS_DIR = ml.ML_DATA_DIR / "chats"

        # Create test directories
        ml.ensure_dirs()

    def tearDown(self):
        """Clean up test environment."""
        # Restore original directories
        ml.ML_DATA_DIR = self.original_ml_data_dir
        ml.CHATS_DIR = self.original_chats_dir

        # Clean up temp directory
        self.test_dir.cleanup()

    def _create_test_chat(
        self,
        chat_id: str,
        query: str = "Test query",
        user_feedback=None,
        date_str: str = None
    ):
        """Helper to create a test chat file."""
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        date_dir = ml.CHATS_DIR / date_str
        date_dir.mkdir(parents=True, exist_ok=True)

        chat_data = {
            "id": chat_id,
            "timestamp": datetime.now().isoformat(),
            "session_id": "test-session",
            "query": query,
            "response": "Test response",
            "files_referenced": [],
            "files_modified": [],
            "tools_used": [],
        }

        if user_feedback is not None:
            chat_data["user_feedback"] = user_feedback

        chat_file = date_dir / f"{chat_id}.json"
        with open(chat_file, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f)

    def test_list_empty_directory(self):
        """Test listing when no chats exist."""
        result = ml.list_chats_needing_feedback()

        # Should return empty list
        self.assertEqual(result, [])

    def test_list_chats_all_without_feedback(self):
        """Test listing when all chats lack feedback."""
        # Create 3 chats without feedback
        for i in range(3):
            self._create_test_chat(f"chat-test-{i}", f"Query {i}")

        result = ml.list_chats_needing_feedback()

        # Should return all 3 chats
        self.assertEqual(len(result), 3)

        # All should not have feedback
        for chat in result:
            self.assertFalse(chat["has_feedback"])
            self.assertIsNone(chat["feedback_rating"])

    def test_list_chats_all_with_feedback(self):
        """Test listing when all chats have feedback."""
        # Create chats with dict-format feedback
        for i in range(3):
            feedback = {
                "rating": "good",
                "comment": "Test",
                "timestamp": datetime.now().isoformat(),
            }
            self._create_test_chat(f"chat-test-{i}", f"Query {i}", user_feedback=feedback)

        result = ml.list_chats_needing_feedback()

        # Should return all chats (function lists all chats, not just needing feedback)
        self.assertEqual(len(result), 3)

        # All should have feedback
        for chat in result:
            self.assertTrue(chat["has_feedback"])
            self.assertEqual(chat["feedback_rating"], "good")

    def test_list_chats_mixed_feedback(self):
        """Test listing with mix of chats with and without feedback."""
        # Create 2 chats without feedback
        self._create_test_chat("chat-no-fb-1", "No feedback 1")
        self._create_test_chat("chat-no-fb-2", "No feedback 2")

        # Create 2 chats with feedback
        feedback = {"rating": "bad", "comment": None, "timestamp": datetime.now().isoformat()}
        self._create_test_chat("chat-with-fb-1", "Has feedback 1", user_feedback=feedback)
        self._create_test_chat("chat-with-fb-2", "Has feedback 2", user_feedback=feedback)

        result = ml.list_chats_needing_feedback()

        # Should return all 4 chats
        self.assertEqual(len(result), 4)

        # Count feedback status
        with_feedback = sum(1 for c in result if c["has_feedback"])
        without_feedback = sum(1 for c in result if not c["has_feedback"])

        self.assertEqual(with_feedback, 2)
        self.assertEqual(without_feedback, 2)

    def test_list_chats_limit_parameter(self):
        """Test that limit parameter works correctly."""
        # Create 10 chats
        for i in range(10):
            self._create_test_chat(f"chat-test-{i:02d}", f"Query {i}")

        # Request only 5
        result = ml.list_chats_needing_feedback(limit=5)

        # Should return exactly 5
        self.assertEqual(len(result), 5)

        # Request 20 (more than exist)
        result = ml.list_chats_needing_feedback(limit=20)

        # Should return all 10
        self.assertEqual(len(result), 10)

    def test_list_chats_query_truncation(self):
        """Test that long queries are truncated."""
        long_query = "A" * 200  # 200 characters
        self._create_test_chat("chat-test-long", long_query)

        result = ml.list_chats_needing_feedback()

        # Query should be truncated to 100 chars
        self.assertEqual(len(result[0]["query"]), 100)
        self.assertEqual(result[0]["query"], "A" * 100)

    def test_list_chats_legacy_string_feedback(self):
        """Test that legacy string format feedback is recognized."""
        # Create chat with legacy string feedback
        self._create_test_chat("chat-legacy", "Legacy chat", user_feedback="good")

        result = ml.list_chats_needing_feedback()

        # Should recognize as having feedback
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0]["has_feedback"])
        self.assertEqual(result[0]["feedback_rating"], "good")

    def test_list_chats_reverse_chronological(self):
        """Test that chats are returned in reverse chronological order (most recent first)."""
        import time

        # Create chats with slight time gaps
        for i in range(3):
            self._create_test_chat(f"chat-test-{i}", f"Query {i}")
            time.sleep(0.01)  # Small delay to ensure different mtimes

        result = ml.list_chats_needing_feedback()

        # IDs should be in reverse order (most recent first)
        # Note: This depends on file mtime, which should be newest first
        self.assertEqual(len(result), 3)
        # The most recently created should be first
        self.assertEqual(result[0]["id"], "chat-test-2")

    def test_list_chats_multiple_dates(self):
        """Test listing chats across multiple date directories."""
        # Create chats on different dates
        self._create_test_chat("chat-2025-01", "Query 1", date_str="2025-01-15")
        self._create_test_chat("chat-2025-02", "Query 2", date_str="2025-02-15")
        self._create_test_chat("chat-2025-03", "Query 3", date_str="2025-03-15")

        result = ml.list_chats_needing_feedback()

        # Should find all chats across dates
        self.assertEqual(len(result), 3)

        # Should be sorted by date (most recent first)
        chat_ids = [c["id"] for c in result]
        self.assertIn("chat-2025-01", chat_ids)
        self.assertIn("chat-2025-02", chat_ids)
        self.assertIn("chat-2025-03", chat_ids)


class TestFeedbackSchemaValidation(unittest.TestCase):
    """Test schema validation for feedback data."""

    def test_validate_chat_with_dict_feedback(self):
        """Test that chat with dict-format feedback validates."""
        chat_data = {
            "id": "test-chat",
            "timestamp": datetime.now().isoformat(),
            "session_id": "test-session",
            "query": "Test query",
            "response": "Test response",
            "files_referenced": [],
            "files_modified": [],
            "tools_used": [],
            "user_feedback": {
                "rating": "good",
                "comment": "Great!",
                "timestamp": datetime.now().isoformat(),
            },
        }

        errors = ml.validate_schema(chat_data, ml.CHAT_SCHEMA, "chat")

        # Should have no validation errors
        self.assertEqual(errors, [])

    def test_validate_chat_with_string_feedback(self):
        """Test that chat with legacy string-format feedback validates."""
        chat_data = {
            "id": "test-chat",
            "timestamp": datetime.now().isoformat(),
            "session_id": "test-session",
            "query": "Test query",
            "response": "Test response",
            "files_referenced": [],
            "files_modified": [],
            "tools_used": [],
            "user_feedback": "good",  # Legacy string format
        }

        errors = ml.validate_schema(chat_data, ml.CHAT_SCHEMA, "chat")

        # Should have no validation errors
        self.assertEqual(errors, [])

    def test_validate_chat_without_feedback(self):
        """Test that chat without feedback validates."""
        chat_data = {
            "id": "test-chat",
            "timestamp": datetime.now().isoformat(),
            "session_id": "test-session",
            "query": "Test query",
            "response": "Test response",
            "files_referenced": [],
            "files_modified": [],
            "tools_used": [],
        }

        errors = ml.validate_schema(chat_data, ml.CHAT_SCHEMA, "chat")

        # Should have no validation errors
        self.assertEqual(errors, [])

    def test_validate_chat_with_none_feedback(self):
        """Test that chat with None feedback validates."""
        chat_data = {
            "id": "test-chat",
            "timestamp": datetime.now().isoformat(),
            "session_id": "test-session",
            "query": "Test query",
            "response": "Test response",
            "files_referenced": [],
            "files_modified": [],
            "tools_used": [],
            "user_feedback": None,
        }

        errors = ml.validate_schema(chat_data, ml.CHAT_SCHEMA, "chat")

        # Should have no validation errors
        self.assertEqual(errors, [])

    def test_validate_chat_with_invalid_feedback_type(self):
        """Test that chat with invalid feedback type fails validation."""
        chat_data = {
            "id": "test-chat",
            "timestamp": datetime.now().isoformat(),
            "session_id": "test-session",
            "query": "Test query",
            "response": "Test response",
            "files_referenced": [],
            "files_modified": [],
            "tools_used": [],
            "user_feedback": 123,  # Invalid: should be dict, str, or None
        }

        errors = ml.validate_schema(chat_data, ml.CHAT_SCHEMA, "chat")

        # Should have validation errors
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("user_feedback" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
