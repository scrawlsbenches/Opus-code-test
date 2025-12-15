#!/usr/bin/env python3
"""
Tests for ML data collector quality-report functionality.
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


class TestDataQualityAnalysis(unittest.TestCase):
    """Test data quality analysis and reporting."""

    def setUp(self):
        """Set up test environment with temporary directories."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_path = Path(self.test_dir.name)

        # Temporarily override ML data directories
        self.original_ml_data_dir = ml.ML_DATA_DIR
        self.original_commits_dir = ml.COMMITS_DIR
        self.original_sessions_dir = ml.SESSIONS_DIR
        self.original_chats_dir = ml.CHATS_DIR
        self.original_actions_dir = ml.ACTIONS_DIR

        ml.ML_DATA_DIR = self.test_path / ".git-ml"
        ml.COMMITS_DIR = ml.ML_DATA_DIR / "commits"
        ml.SESSIONS_DIR = ml.ML_DATA_DIR / "sessions"
        ml.CHATS_DIR = ml.ML_DATA_DIR / "chats"
        ml.ACTIONS_DIR = ml.ML_DATA_DIR / "actions"

        # Create test directories
        ml.ensure_dirs()

    def tearDown(self):
        """Clean up test environment."""
        # Restore original directories
        ml.ML_DATA_DIR = self.original_ml_data_dir
        ml.COMMITS_DIR = self.original_commits_dir
        ml.SESSIONS_DIR = self.original_sessions_dir
        ml.CHATS_DIR = self.original_chats_dir
        ml.ACTIONS_DIR = self.original_actions_dir

        # Clean up temp directory
        self.test_dir.cleanup()

    def test_analyze_empty_data(self):
        """Test quality analysis with no data."""
        result = ml.analyze_data_quality()

        # Should return structure with zero counts
        self.assertIn('completeness', result)
        self.assertIn('diversity', result)
        self.assertIn('anomalies', result)
        self.assertIn('quality_score', result)

        comp = result['completeness']
        self.assertEqual(comp['chats_total'], 0)
        self.assertEqual(comp['commits_total'], 0)
        self.assertEqual(comp['sessions_total'], 0)

        # Quality score should be low with no data
        self.assertIsInstance(result['quality_score'], int)
        self.assertLessEqual(result['quality_score'], 100)
        self.assertGreaterEqual(result['quality_score'], 0)

    def test_completeness_all_fields_present(self):
        """Test completeness calculation with all required fields."""
        # Create complete chat entry
        chat_data = {
            'id': 'chat-test-001',
            'timestamp': datetime.now().isoformat(),
            'session_id': 'session-001',
            'query': 'Test query',
            'response': 'Test response',
            'files_referenced': ['/path/to/file.py'],
            'files_modified': [],
            'tools_used': ['Read'],
            'query_tokens': 10,
            'response_tokens': 20,
            'user_feedback': None,
        }

        chat_file = ml.CHATS_DIR / "chat-test-001.json"
        with open(chat_file, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f)

        result = ml.analyze_data_quality()

        comp = result['completeness']
        self.assertEqual(comp['chats_total'], 1)
        self.assertEqual(comp['chats_complete'], 1)
        self.assertEqual(comp['chats_complete_pct'], 100.0)

    def test_completeness_missing_fields(self):
        """Test completeness calculation with missing fields."""
        # Create incomplete chat entry (missing required fields)
        incomplete_chat = {
            'id': 'chat-test-002',
            'timestamp': datetime.now().isoformat(),
            'query': 'Test query',
            # Missing: session_id, response, files_referenced, files_modified, tools_used
        }

        chat_file = ml.CHATS_DIR / "chat-test-002.json"
        with open(chat_file, 'w', encoding='utf-8') as f:
            json.dump(incomplete_chat, f)

        result = ml.analyze_data_quality()

        comp = result['completeness']
        self.assertEqual(comp['chats_total'], 1)
        self.assertEqual(comp['chats_complete'], 0)
        self.assertEqual(comp['chats_complete_pct'], 0.0)

    def test_commits_with_ci_results(self):
        """Test tracking commits with CI results."""
        # Commit with CI results
        commit1 = {
            'hash': 'abc123def456',
            'message': 'Test commit',
            'timestamp': datetime.now().isoformat(),
            'files_changed': ['/path/to/file.py'],
            'ci_result': {'status': 'pass', 'coverage': 89.5},
        }

        # Commit without CI results
        commit2 = {
            'hash': 'def456abc789',
            'message': 'Another commit',
            'timestamp': datetime.now().isoformat(),
            'files_changed': ['/path/to/other.py'],
        }

        with open(ml.COMMITS_DIR / "commit1.json", 'w', encoding='utf-8') as f:
            json.dump(commit1, f)
        with open(ml.COMMITS_DIR / "commit2.json", 'w', encoding='utf-8') as f:
            json.dump(commit2, f)

        result = ml.analyze_data_quality()

        comp = result['completeness']
        self.assertEqual(comp['commits_total'], 2)
        self.assertEqual(comp['commits_with_ci'], 1)
        self.assertEqual(comp['commits_with_ci_pct'], 50.0)

    def test_sessions_with_commits(self):
        """Test tracking sessions with linked commits."""
        # Create session
        session_data = {
            'id': 'session-001',
            'start_time': datetime.now().isoformat(),
            'chat_ids': ['chat-001'],
        }

        with open(ml.SESSIONS_DIR / "session-001.json", 'w', encoding='utf-8') as f:
            json.dump(session_data, f)

        # Create commit linked to session
        commit_data = {
            'hash': 'abc123',
            'message': 'Test',
            'timestamp': datetime.now().isoformat(),
            'files_changed': ['test.py'],
            'session_id': 'session-001',
        }

        with open(ml.COMMITS_DIR / "commit.json", 'w', encoding='utf-8') as f:
            json.dump(commit_data, f)

        result = ml.analyze_data_quality()

        comp = result['completeness']
        self.assertEqual(comp['sessions_total'], 1)
        self.assertEqual(comp['sessions_with_commits'], 1)
        self.assertEqual(comp['sessions_with_commits_pct'], 100.0)

    def test_chats_with_feedback(self):
        """Test tracking chats with user feedback."""
        # Chat with feedback
        chat1 = {
            'id': 'chat-001',
            'timestamp': datetime.now().isoformat(),
            'session_id': 'session-001',
            'query': 'Test',
            'response': 'Response',
            'files_referenced': [],
            'files_modified': [],
            'tools_used': [],
            'user_feedback': {'rating': 5, 'comment': 'Great!'},
        }

        # Chat without feedback
        chat2 = {
            'id': 'chat-002',
            'timestamp': datetime.now().isoformat(),
            'session_id': 'session-001',
            'query': 'Test',
            'response': 'Response',
            'files_referenced': [],
            'files_modified': [],
            'tools_used': [],
        }

        with open(ml.CHATS_DIR / "chat1.json", 'w', encoding='utf-8') as f:
            json.dump(chat1, f)
        with open(ml.CHATS_DIR / "chat2.json", 'w', encoding='utf-8') as f:
            json.dump(chat2, f)

        result = ml.analyze_data_quality()

        comp = result['completeness']
        self.assertEqual(comp['chats_total'], 2)
        self.assertEqual(comp['chats_with_feedback'], 1)
        self.assertEqual(comp['chats_with_feedback_pct'], 50.0)

    def test_diversity_unique_files(self):
        """Test tracking unique files across chats and commits."""
        # Chat with files
        chat = {
            'id': 'chat-001',
            'timestamp': datetime.now().isoformat(),
            'session_id': 'session-001',
            'query': 'Test',
            'response': 'Response',
            'files_referenced': ['/file1.py', '/file2.py'],
            'files_modified': ['/file3.py'],
            'tools_used': [],
        }

        # Commit with files (one overlapping)
        commit = {
            'hash': 'abc123',
            'message': 'Test',
            'timestamp': datetime.now().isoformat(),
            'files_changed': ['/file3.py', '/file4.py'],
        }

        with open(ml.CHATS_DIR / "chat.json", 'w', encoding='utf-8') as f:
            json.dump(chat, f)
        with open(ml.COMMITS_DIR / "commit.json", 'w', encoding='utf-8') as f:
            json.dump(commit, f)

        result = ml.analyze_data_quality()

        div = result['diversity']
        # Should have 4 unique files: file1, file2, file3, file4
        self.assertEqual(div['unique_files'], 4)

    def test_diversity_unique_tools(self):
        """Test tracking unique tools and their usage counts."""
        chats = [
            {
                'id': f'chat-{i}',
                'timestamp': datetime.now().isoformat(),
                'session_id': 'session-001',
                'query': 'Test',
                'response': 'Response',
                'files_referenced': [],
                'files_modified': [],
                'tools_used': ['Read', 'Edit'] if i % 2 == 0 else ['Bash', 'Read'],
            }
            for i in range(4)
        ]

        for chat in chats:
            chat_file = ml.CHATS_DIR / f"{chat['id']}.json"
            with open(chat_file, 'w', encoding='utf-8') as f:
                json.dump(chat, f)

        result = ml.analyze_data_quality()

        div = result['diversity']
        # Should have 3 unique tools: Read, Edit, Bash
        self.assertEqual(div['unique_tools'], 3)

        # Check tool distribution
        tool_dist = div['tool_distribution']
        self.assertEqual(tool_dist['Read'], 4)  # Used in all 4 chats
        self.assertEqual(tool_dist['Edit'], 2)  # Used in 2 chats
        self.assertEqual(tool_dist['Bash'], 2)  # Used in 2 chats

    def test_diversity_query_response_lengths(self):
        """Test tracking query and response length statistics."""
        chats = [
            {
                'id': f'chat-{i}',
                'timestamp': datetime.now().isoformat(),
                'session_id': 'session-001',
                'query': 'Q' * (10 * (i + 1)),  # 10, 20, 30, 40, 50
                'response': 'R' * (100 * (i + 1)),  # 100, 200, 300, 400, 500
                'files_referenced': [],
                'files_modified': [],
                'tools_used': [],
            }
            for i in range(5)
        ]

        for chat in chats:
            chat_file = ml.CHATS_DIR / f"{chat['id']}.json"
            with open(chat_file, 'w', encoding='utf-8') as f:
                json.dump(chat, f)

        result = ml.analyze_data_quality()

        div = result['diversity']
        self.assertEqual(div['query_length_min'], 10)
        self.assertEqual(div['query_length_avg'], 30)  # (10+20+30+40+50)/5
        self.assertEqual(div['query_length_max'], 50)
        self.assertEqual(div['response_length_min'], 100)
        self.assertEqual(div['response_length_avg'], 300)  # (100+200+300+400+500)/5
        self.assertEqual(div['response_length_max'], 500)

    def test_anomaly_empty_responses(self):
        """Test detection of empty responses."""
        chats = [
            {
                'id': 'chat-empty',
                'timestamp': datetime.now().isoformat(),
                'session_id': 'session-001',
                'query': 'Test',
                'response': '',  # Empty response
                'files_referenced': [],
                'files_modified': [],
                'tools_used': [],
            },
            {
                'id': 'chat-whitespace',
                'timestamp': datetime.now().isoformat(),
                'session_id': 'session-001',
                'query': 'Test',
                'response': '   ',  # Whitespace only
                'files_referenced': [],
                'files_modified': [],
                'tools_used': [],
            },
            {
                'id': 'chat-normal',
                'timestamp': datetime.now().isoformat(),
                'session_id': 'session-001',
                'query': 'Test',
                'response': 'Normal response',
                'files_referenced': [],
                'files_modified': [],
                'tools_used': [],
            },
        ]

        for chat in chats:
            chat_file = ml.CHATS_DIR / f"{chat['id']}.json"
            with open(chat_file, 'w', encoding='utf-8') as f:
                json.dump(chat, f)

        result = ml.analyze_data_quality()

        anom = result['anomalies']
        self.assertEqual(anom['empty_responses'], 2)

    def test_anomaly_zero_file_commits(self):
        """Test detection of commits with no files changed."""
        commits = [
            {
                'hash': 'abc123',
                'message': 'Empty commit',
                'timestamp': datetime.now().isoformat(),
                'files_changed': [],  # No files
            },
            {
                'hash': 'def456',
                'message': 'Normal commit',
                'timestamp': datetime.now().isoformat(),
                'files_changed': ['file.py'],
            },
        ]

        for i, commit in enumerate(commits):
            commit_file = ml.COMMITS_DIR / f"commit{i}.json"
            with open(commit_file, 'w', encoding='utf-8') as f:
                json.dump(commit, f)

        result = ml.analyze_data_quality()

        anom = result['anomalies']
        self.assertEqual(anom['zero_file_commits'], 1)

    def test_anomaly_empty_sessions(self):
        """Test detection of sessions with no chats."""
        sessions = [
            {
                'id': 'session-empty',
                'start_time': datetime.now().isoformat(),
                'chat_ids': [],  # No chats
            },
            {
                'id': 'session-normal',
                'start_time': datetime.now().isoformat(),
                'chat_ids': ['chat-001'],
            },
        ]

        for session in sessions:
            session_file = ml.SESSIONS_DIR / f"{session['id']}.json"
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session, f)

        result = ml.analyze_data_quality()

        anom = result['anomalies']
        self.assertEqual(anom['empty_sessions'], 1)

    def test_anomaly_potential_duplicates(self):
        """Test detection of potential duplicate entries."""
        timestamp = datetime.now().isoformat()

        # Two chats with same timestamp and query
        chats = [
            {
                'id': 'chat-001',
                'timestamp': timestamp,
                'session_id': 'session-001',
                'query': 'Identical query',
                'response': 'Response 1',
                'files_referenced': [],
                'files_modified': [],
                'tools_used': [],
            },
            {
                'id': 'chat-002',
                'timestamp': timestamp,
                'session_id': 'session-001',
                'query': 'Identical query',
                'response': 'Response 2',
                'files_referenced': [],
                'files_modified': [],
                'tools_used': [],
            },
        ]

        for chat in chats:
            chat_file = ml.CHATS_DIR / f"{chat['id']}.json"
            with open(chat_file, 'w', encoding='utf-8') as f:
                json.dump(chat, f)

        result = ml.analyze_data_quality()

        anom = result['anomalies']
        # Second entry should be flagged as duplicate
        self.assertEqual(anom['potential_duplicates'], 1)

    def test_quality_score_perfect_data(self):
        """Test quality score calculation with perfect data."""
        # Create perfect data: complete chats, diverse tools/files, no anomalies

        # Create 10 complete chats
        for i in range(10):
            chat = {
                'id': f'chat-{i:03d}',
                'timestamp': datetime.now().isoformat(),
                'session_id': 'session-001',
                'query': f'Query {i}',
                'response': f'Response {i}',
                'files_referenced': [f'/file{i}.py'],
                'files_modified': [f'/modified{i}.py'],
                'tools_used': ['Read', 'Edit', 'Grep', 'Bash', 'Write'][i % 5:i % 5 + 2],
                'user_feedback': {'rating': 5},
            }
            chat_file = ml.CHATS_DIR / f"{chat['id']}.json"
            with open(chat_file, 'w', encoding='utf-8') as f:
                json.dump(chat, f)

        # Create 5 commits with CI results
        for i in range(5):
            commit = {
                'hash': f'commit{i:03d}',
                'message': f'Commit {i}',
                'timestamp': datetime.now().isoformat(),
                'files_changed': [f'/commit_file{i}.py'] * (i + 1),
                'ci_result': {'status': 'pass', 'coverage': 90},
                'session_id': 'session-001',
            }
            commit_file = ml.COMMITS_DIR / f"commit{i}.json"
            with open(commit_file, 'w', encoding='utf-8') as f:
                json.dump(commit, f)

        # Create session
        session = {
            'id': 'session-001',
            'start_time': datetime.now().isoformat(),
            'chat_ids': [f'chat-{i:03d}' for i in range(10)],
        }
        with open(ml.SESSIONS_DIR / "session-001.json", 'w', encoding='utf-8') as f:
            json.dump(session, f)

        result = ml.analyze_data_quality()

        # With perfect data, quality score should be very high
        self.assertGreater(result['quality_score'], 80)
        self.assertLessEqual(result['quality_score'], 100)

    def test_quality_score_low_with_anomalies(self):
        """Test that anomalies reduce quality score."""
        # Create data with many anomalies

        # Empty responses
        for i in range(10):
            chat = {
                'id': f'chat-{i:03d}',
                'timestamp': datetime.now().isoformat(),
                'session_id': 'session-001',
                'query': f'Query {i}',
                'response': '',  # Empty!
                'files_referenced': [],
                'files_modified': [],
                'tools_used': [],
            }
            chat_file = ml.CHATS_DIR / f"{chat['id']}.json"
            with open(chat_file, 'w', encoding='utf-8') as f:
                json.dump(chat, f)

        # Zero-file commits
        for i in range(5):
            commit = {
                'hash': f'commit{i:03d}',
                'message': f'Empty commit {i}',
                'timestamp': datetime.now().isoformat(),
                'files_changed': [],  # No files!
            }
            commit_file = ml.COMMITS_DIR / f"commit{i}.json"
            with open(commit_file, 'w', encoding='utf-8') as f:
                json.dump(commit, f)

        # Empty sessions
        for i in range(3):
            session = {
                'id': f'session-{i:03d}',
                'start_time': datetime.now().isoformat(),
                'chat_ids': [],  # No chats!
            }
            session_file = ml.SESSIONS_DIR / f"session-{i:03d}.json"
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session, f)

        result = ml.analyze_data_quality()

        # With many anomalies, quality score should be lower
        # Not necessarily very low since some completeness metrics may still be ok
        self.assertLess(result['quality_score'], 100)

    def test_corrupted_json_files_skipped(self):
        """Test that corrupted JSON files are skipped gracefully."""
        # Create valid chat
        valid_chat = {
            'id': 'chat-valid',
            'timestamp': datetime.now().isoformat(),
            'session_id': 'session-001',
            'query': 'Test',
            'response': 'Response',
            'files_referenced': [],
            'files_modified': [],
            'tools_used': [],
        }
        with open(ml.CHATS_DIR / "valid.json", 'w', encoding='utf-8') as f:
            json.dump(valid_chat, f)

        # Create corrupted JSON file
        with open(ml.CHATS_DIR / "corrupted.json", 'w', encoding='utf-8') as f:
            f.write("{ invalid json content }")

        # Should not raise an error
        result = ml.analyze_data_quality()

        # Should count only the valid chat
        comp = result['completeness']
        self.assertEqual(comp['chats_total'], 1)

    def test_missing_directories_handled(self):
        """Test that missing directories are handled gracefully."""
        # Remove all data directories
        import shutil
        shutil.rmtree(ml.ML_DATA_DIR)

        # Should not raise an error, just return empty metrics
        result = ml.analyze_data_quality()

        self.assertEqual(result['completeness']['chats_total'], 0)
        self.assertEqual(result['completeness']['commits_total'], 0)
        self.assertEqual(result['completeness']['sessions_total'], 0)

    def test_mixed_valid_invalid_data(self):
        """Test analysis with mix of valid and invalid data."""
        # Valid complete chat
        chat1 = {
            'id': 'chat-001',
            'timestamp': datetime.now().isoformat(),
            'session_id': 'session-001',
            'query': 'Test',
            'response': 'Response',
            'files_referenced': ['/file.py'],
            'files_modified': [],
            'tools_used': ['Read'],
        }

        # Invalid incomplete chat
        chat2 = {
            'id': 'chat-002',
            'timestamp': datetime.now().isoformat(),
            'query': 'Test',
            # Missing required fields
        }

        # Valid commit with CI
        commit1 = {
            'hash': 'abc123',
            'message': 'Test',
            'timestamp': datetime.now().isoformat(),
            'files_changed': ['file.py'],
            'ci_result': {'status': 'pass'},
        }

        # Valid commit without CI
        commit2 = {
            'hash': 'def456',
            'message': 'Test',
            'timestamp': datetime.now().isoformat(),
            'files_changed': ['other.py'],
        }

        with open(ml.CHATS_DIR / "chat1.json", 'w', encoding='utf-8') as f:
            json.dump(chat1, f)
        with open(ml.CHATS_DIR / "chat2.json", 'w', encoding='utf-8') as f:
            json.dump(chat2, f)
        with open(ml.COMMITS_DIR / "commit1.json", 'w', encoding='utf-8') as f:
            json.dump(commit1, f)
        with open(ml.COMMITS_DIR / "commit2.json", 'w', encoding='utf-8') as f:
            json.dump(commit2, f)

        result = ml.analyze_data_quality()

        comp = result['completeness']
        self.assertEqual(comp['chats_total'], 2)
        self.assertEqual(comp['chats_complete'], 1)
        self.assertEqual(comp['commits_total'], 2)
        self.assertEqual(comp['commits_with_ci'], 1)

    def test_quality_score_boundaries(self):
        """Test that quality score is always between 0 and 100."""
        # Test with various data scenarios
        scenarios = [
            {},  # Empty
            # Will add data in loop
        ]

        # Scenario 1: Empty data
        result = ml.analyze_data_quality()
        self.assertGreaterEqual(result['quality_score'], 0)
        self.assertLessEqual(result['quality_score'], 100)

        # Scenario 2: Some data
        chat = {
            'id': 'chat-001',
            'timestamp': datetime.now().isoformat(),
            'session_id': 'session-001',
            'query': 'Test',
            'response': 'Response',
            'files_referenced': [],
            'files_modified': [],
            'tools_used': [],
        }
        with open(ml.CHATS_DIR / "chat.json", 'w', encoding='utf-8') as f:
            json.dump(chat, f)

        result = ml.analyze_data_quality()
        self.assertGreaterEqual(result['quality_score'], 0)
        self.assertLessEqual(result['quality_score'], 100)

    def test_zero_division_protection(self):
        """Test that analysis handles edge cases without division by zero."""
        # Empty data should not cause division by zero
        result = ml.analyze_data_quality()

        # All percentages should be valid numbers
        comp = result['completeness']
        for key in comp:
            if '_pct' in key:
                self.assertIsInstance(comp[key], (int, float))
                self.assertGreaterEqual(comp[key], 0)

        # Diversity stats should have valid averages
        div = result['diversity']
        self.assertGreaterEqual(div['query_length_avg'], 0)
        self.assertGreaterEqual(div['response_length_avg'], 0)

    def test_diversity_with_no_lengths(self):
        """Test diversity calculation when no query/response data exists."""
        # Create commits only (no chats)
        commit = {
            'hash': 'abc123',
            'message': 'Test',
            'timestamp': datetime.now().isoformat(),
            'files_changed': ['file.py'],
        }
        with open(ml.COMMITS_DIR / "commit.json", 'w', encoding='utf-8') as f:
            json.dump(commit, f)

        result = ml.analyze_data_quality()

        div = result['diversity']
        # Should handle empty length lists gracefully
        self.assertEqual(div['query_length_min'], 0)
        self.assertEqual(div['query_length_avg'], 0)
        self.assertEqual(div['query_length_max'], 0)
        self.assertEqual(div['response_length_min'], 0)
        self.assertEqual(div['response_length_avg'], 0)
        self.assertEqual(div['response_length_max'], 0)


if __name__ == "__main__":
    unittest.main()
