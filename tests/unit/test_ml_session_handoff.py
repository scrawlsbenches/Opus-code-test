#!/usr/bin/env python3
"""
Tests for ML data collector session handoff functionality.
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


class TestSessionHandoff(unittest.TestCase):
    """Test session handoff document generation."""

    def setUp(self):
        """Set up test environment with temporary directories."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_path = Path(self.test_dir.name)

        # Temporarily override ML data directories
        self.original_ml_data_dir = ml.ML_DATA_DIR
        self.original_commits_dir = ml.COMMITS_DIR
        self.original_sessions_dir = ml.SESSIONS_DIR
        self.original_chats_dir = ml.CHATS_DIR
        self.original_current_session_file = ml.CURRENT_SESSION_FILE

        ml.ML_DATA_DIR = self.test_path / ".git-ml"
        ml.COMMITS_DIR = ml.ML_DATA_DIR / "commits"
        ml.SESSIONS_DIR = ml.ML_DATA_DIR / "sessions"
        ml.CHATS_DIR = ml.ML_DATA_DIR / "chats"
        ml.CURRENT_SESSION_FILE = ml.ML_DATA_DIR / "current_session.json"

        # Create test directories
        ml.ensure_dirs()

    def tearDown(self):
        """Clean up test environment."""
        # Restore original directories
        ml.ML_DATA_DIR = self.original_ml_data_dir
        ml.COMMITS_DIR = self.original_commits_dir
        ml.SESSIONS_DIR = self.original_sessions_dir
        ml.CHATS_DIR = self.original_chats_dir
        ml.CURRENT_SESSION_FILE = self.original_current_session_file

        # Clean up temp directory
        self.test_dir.cleanup()

    def test_handoff_no_session(self):
        """Test handoff with no active session."""
        handoff = ml.generate_session_handoff()
        self.assertIn("No active session", handoff)

    def test_handoff_empty_session(self):
        """Test handoff with empty session (no chats)."""
        # Create empty session
        session_id = ml.start_session()

        handoff = ml.generate_session_handoff()

        # Verify structure
        self.assertIn("Session Handoff:", handoff)
        self.assertIn(session_id, handoff)
        self.assertIn("## Summary", handoff)
        self.assertIn("## Key Work Done", handoff)
        self.assertIn("## Files Touched", handoff)
        self.assertIn("## Related Commits", handoff)
        self.assertIn("## Suggested Next Steps", handoff)

        # Verify empty session indicators
        self.assertIn("Exchanges: 0", handoff)
        self.assertIn("No significant work recorded", handoff)
        self.assertIn("No files modified or referenced", handoff)
        self.assertIn("No commits made in this session", handoff)

    def test_handoff_with_chats(self):
        """Test handoff with chat entries."""
        # Create session
        session_id = ml.start_session()

        # Add some chat entries
        chat1 = ml.ChatEntry(
            id="chat-test-001",
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
            query="How do I implement feature X?",
            response="You can implement it by...",
            files_referenced=["/path/to/file1.py"],
            files_modified=[],
            tools_used=["Read", "Grep"],
        )
        ml.save_chat_entry(chat1, validate=False)
        ml.add_chat_to_session(chat1.id)

        chat2 = ml.ChatEntry(
            id="chat-test-002",
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
            query="Fix bug in authentication",
            response="The bug is in auth.py line 42",
            files_referenced=["/path/to/auth.py"],
            files_modified=["/path/to/auth.py"],
            tools_used=["Read", "Edit"],
        )
        ml.save_chat_entry(chat2, validate=False)
        ml.add_chat_to_session(chat2.id)

        handoff = ml.generate_session_handoff()

        # Verify chat data appears
        self.assertIn("Exchanges: 2", handoff)
        self.assertIn("How do I implement feature X", handoff)
        self.assertIn("Fix bug in authentication", handoff)
        self.assertIn("/path/to/auth.py", handoff)
        self.assertIn("Edit, Grep, Read", handoff)  # Sorted tools

    def test_handoff_with_commits(self):
        """Test handoff with related commits."""
        # Create session
        session_id = ml.start_session()

        # Create a mock commit linked to session
        commit_data = {
            "hash": "abc123def456",
            "message": "feat: Add session handoff feature",
            "author": "Test Author",
            "timestamp": datetime.now().isoformat(),
            "branch": "main",
            "files_changed": ["scripts/ml_data_collector.py"],
            "insertions": 100,
            "deletions": 10,
            "hunks": [],
            "hour_of_day": 10,
            "day_of_week": "Monday",
            "seconds_since_last_commit": None,
            "is_merge": False,
            "is_initial": False,
            "parent_count": 1,
            "session_id": session_id,
            "related_chats": [],
        }

        commit_file = ml.COMMITS_DIR / f"{commit_data['hash'][:8]}_test.json"
        with open(commit_file, 'w', encoding='utf-8') as f:
            json.dump(commit_data, f)

        handoff = ml.generate_session_handoff()

        # Verify commit appears
        self.assertIn("abc123de", handoff)  # First 8 chars of hash
        self.assertIn("feat: Add session handoff feature", handoff)

    def test_handoff_suggestions_modified_files_no_commit(self):
        """Test suggestions when files are modified but no commit."""
        session_id = ml.start_session()

        # Add chat with file modifications
        chat = ml.ChatEntry(
            id="chat-test-003",
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
            query="Update feature",
            response="Done",
            files_referenced=[],
            files_modified=["/path/to/feature.py"],
            tools_used=["Edit"],
        )
        ml.save_chat_entry(chat, validate=False)
        ml.add_chat_to_session(chat.id)

        handoff = ml.generate_session_handoff()

        # Should suggest committing changes
        self.assertIn("Review modified files", handoff)
        self.assertIn("test suite", handoff)

    def test_handoff_suggestions_recent_errors(self):
        """Test suggestions when recent responses mention errors."""
        session_id = ml.start_session()

        # Add chats with errors in responses
        for i in range(3):
            chat = ml.ChatEntry(
                id=f"chat-test-00{i}",
                timestamp=datetime.now().isoformat(),
                session_id=session_id,
                query=f"Task {i}",
                response="Error: something failed" if i == 2 else "OK",
                files_referenced=[],
                files_modified=[],
                tools_used=[],
            )
            ml.save_chat_entry(chat, validate=False)
            ml.add_chat_to_session(chat.id)

        handoff = ml.generate_session_handoff()

        # Should suggest debugging
        self.assertIn("errors", handoff.lower())
        self.assertIn("debugging", handoff.lower())

    def test_handoff_duration_calculation(self):
        """Test that duration is calculated correctly."""
        session_id = ml.start_session()

        handoff = ml.generate_session_handoff()

        # Duration should be in minutes (session just started)
        self.assertIn("0m", handoff)

    def test_handoff_tools_list(self):
        """Test that tools are listed and sorted correctly."""
        session_id = ml.start_session()

        # Add chat with multiple tools
        chat = ml.ChatEntry(
            id="chat-test-tools",
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
            query="Complex task",
            response="Done using multiple tools",
            files_referenced=[],
            files_modified=[],
            tools_used=["Write", "Bash", "Read", "Edit"],
        )
        ml.save_chat_entry(chat, validate=False)
        ml.add_chat_to_session(chat.id)

        handoff = ml.generate_session_handoff()

        # Tools should be sorted alphabetically
        tools_line = [line for line in handoff.split('\n') if 'Tools used:' in line][0]
        self.assertIn("Bash, Edit, Read, Write", tools_line)

    def test_handoff_file_sections(self):
        """Test that files are separated into modified and referenced sections."""
        session_id = ml.start_session()

        # Add chat with both modified and referenced files
        chat = ml.ChatEntry(
            id="chat-test-files",
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
            query="Work on files",
            response="Updated files",
            files_referenced=["/ref1.py", "/ref2.py"],
            files_modified=["/mod1.py", "/mod2.py"],
            tools_used=["Read", "Edit"],
        )
        ml.save_chat_entry(chat, validate=False)
        ml.add_chat_to_session(chat.id)

        handoff = ml.generate_session_handoff()

        # Check for proper sections
        self.assertIn("### Modified:", handoff)
        self.assertIn("### Referenced:", handoff)
        self.assertIn("/mod1.py", handoff)
        self.assertIn("/ref1.py", handoff)


if __name__ == "__main__":
    unittest.main()
