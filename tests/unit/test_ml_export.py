#!/usr/bin/env python3
"""
Unit tests for ML data collector export functionality.

Tests the export_data() function and related helpers for exporting
collected ML data in training-ready formats (JSONL, CSV, HuggingFace).
"""

import csv
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

import ml_data_collector as ml


class TestSummarizeDiff(unittest.TestCase):
    """Tests for _summarize_diff helper function."""

    def test_empty_hunks(self):
        """Should return empty string for empty hunks list."""
        result = ml._summarize_diff([])
        self.assertEqual(result, "")

    def test_single_file_add_hunk(self):
        """Should summarize single add hunk."""
        hunks = [
            {"file": "test.py", "change_type": "add"}
        ]
        result = ml._summarize_diff(hunks)
        self.assertEqual(result, "test.py: +1")

    def test_single_file_delete_hunk(self):
        """Should summarize single delete hunk."""
        hunks = [
            {"file": "test.py", "change_type": "delete"}
        ]
        result = ml._summarize_diff(hunks)
        self.assertEqual(result, "test.py: -1")

    def test_single_file_modify_hunk(self):
        """Should summarize single modify hunk."""
        hunks = [
            {"file": "test.py", "change_type": "modify"}
        ]
        result = ml._summarize_diff(hunks)
        self.assertEqual(result, "test.py: ~1")

    def test_single_file_multiple_changes(self):
        """Should summarize multiple changes to same file."""
        hunks = [
            {"file": "test.py", "change_type": "add"},
            {"file": "test.py", "change_type": "modify"},
            {"file": "test.py", "change_type": "modify"},
        ]
        result = ml._summarize_diff(hunks)
        self.assertEqual(result, "test.py: +1 ~2")

    def test_multiple_files(self):
        """Should summarize changes across multiple files."""
        hunks = [
            {"file": "file1.py", "change_type": "add"},
            {"file": "file2.py", "change_type": "delete"},
            {"file": "file3.py", "change_type": "modify"},
        ]
        result = ml._summarize_diff(hunks)
        self.assertIn("file1.py: +1", result)
        self.assertIn("file2.py: -1", result)
        self.assertIn("file3.py: ~1", result)
        # Files separated by semicolons
        self.assertEqual(result.count(";"), 2)

    def test_missing_file_field(self):
        """Should handle hunks missing file field."""
        hunks = [
            {"change_type": "add"},
        ]
        result = ml._summarize_diff(hunks)
        self.assertIn("unknown", result)

    def test_missing_change_type_field(self):
        """Should default to modify when change_type missing."""
        hunks = [
            {"file": "test.py"},
        ]
        result = ml._summarize_diff(hunks)
        self.assertEqual(result, "test.py: ~1")

    def test_limit_to_ten_files(self):
        """Should limit summary to first 10 files."""
        hunks = [
            {"file": f"file{i}.py", "change_type": "modify"}
            for i in range(20)
        ]
        result = ml._summarize_diff(hunks)
        # Should only include first 10 files (9 semicolons)
        self.assertEqual(result.count(";"), 9)


class TestExportData(unittest.TestCase):
    """Tests for export_data() function."""

    def setUp(self):
        """Set up test environment with temporary directories."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_path = Path(self.test_dir.name)

        # Temporarily override ML data directories
        self.original_ml_data_dir = ml.ML_DATA_DIR
        self.original_commits_dir = ml.COMMITS_DIR
        self.original_chats_dir = ml.CHATS_DIR

        ml.ML_DATA_DIR = self.test_path / ".git-ml"
        ml.COMMITS_DIR = ml.ML_DATA_DIR / "commits"
        ml.CHATS_DIR = ml.ML_DATA_DIR / "chats"

        # Create test directories
        ml.ensure_dirs()

    def tearDown(self):
        """Clean up test environment."""
        # Restore original directories
        ml.ML_DATA_DIR = self.original_ml_data_dir
        ml.COMMITS_DIR = self.original_commits_dir
        ml.CHATS_DIR = self.original_chats_dir

        # Clean up temp directory
        self.test_dir.cleanup()

    def test_export_empty_data_jsonl(self):
        """Should export empty JSONL file when no data exists."""
        output_path = self.test_path / "export.jsonl"
        stats = ml.export_data("jsonl", output_path)

        # Check stats
        self.assertEqual(stats["format"], "jsonl")
        self.assertEqual(stats["records"], 0)
        self.assertEqual(stats["commits"], 0)
        self.assertEqual(stats["chats"], 0)
        self.assertEqual(stats["output_path"], str(output_path))

        # Check file is empty
        self.assertTrue(output_path.exists())
        content = output_path.read_text()
        self.assertEqual(content, "")

    def test_export_empty_data_csv(self):
        """Should export CSV with headers only when no data exists."""
        output_path = self.test_path / "export.csv"
        stats = ml.export_data("csv", output_path)

        # Check stats
        self.assertEqual(stats["format"], "csv")
        self.assertEqual(stats["records"], 0)

        # Check file has headers only
        self.assertTrue(output_path.exists())
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            self.assertIn('type', headers)
            self.assertIn('timestamp', headers)
            self.assertIn('input', headers)
            self.assertIn('output', headers)
            # Should have no data rows
            rows = list(reader)
            self.assertEqual(len(rows), 0)

    def test_export_empty_data_huggingface(self):
        """Should export empty HuggingFace format when no data exists."""
        output_path = self.test_path / "export.json"
        stats = ml.export_data("huggingface", output_path)

        # Check stats
        self.assertEqual(stats["format"], "huggingface")
        self.assertEqual(stats["records"], 0)

        # Check file has correct structure
        self.assertTrue(output_path.exists())
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Should be dict of lists
        self.assertIsInstance(data, dict)
        self.assertIn('type', data)
        self.assertIn('timestamp', data)
        self.assertIn('input', data)
        self.assertIn('output', data)
        self.assertIn('session_id', data)
        self.assertIn('files', data)
        self.assertIn('tools_used', data)

        # All lists should be empty
        for key in data:
            self.assertIsInstance(data[key], list)
            self.assertEqual(len(data[key]), 0)

    def test_export_with_commit_data_jsonl(self):
        """Should export commit data in JSONL format."""
        # Create mock commit
        commit_data = {
            "hash": "abc123",
            "message": "feat: Add feature",
            "timestamp": "2025-12-15T10:00:00",
            "files_changed": ["file1.py", "file2.py"],
            "insertions": 50,
            "deletions": 10,
            "branch": "main",
            "session_id": "sess1",
            "hunks": [
                {"file": "file1.py", "change_type": "add"},
                {"file": "file2.py", "change_type": "modify"},
            ]
        }

        commit_file = ml.COMMITS_DIR / "abc123_test.json"
        with open(commit_file, 'w', encoding='utf-8') as f:
            json.dump(commit_data, f)

        # Export
        output_path = self.test_path / "export.jsonl"
        stats = ml.export_data("jsonl", output_path)

        # Check stats
        self.assertEqual(stats["records"], 1)
        self.assertEqual(stats["commits"], 1)
        self.assertEqual(stats["chats"], 0)

        # Check JSONL content
        with open(output_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        self.assertEqual(len(lines), 1)
        record = json.loads(lines[0])

        # Check record structure
        self.assertEqual(record["type"], "commit")
        self.assertEqual(record["timestamp"], "2025-12-15T10:00:00")
        self.assertEqual(record["input"], "feat: Add feature")
        self.assertIn("file1.py", record["output"])  # Diff summary
        self.assertEqual(record["context"]["session_id"], "sess1")
        self.assertEqual(record["context"]["files"], ["file1.py", "file2.py"])
        self.assertEqual(record["context"]["insertions"], 50)
        self.assertEqual(record["context"]["deletions"], 10)

    def test_export_with_chat_data_jsonl(self):
        """Should export chat data in JSONL format."""
        # Create mock chat
        chat_data = {
            "id": "chat-001",
            "timestamp": "2025-12-15T11:00:00",
            "session_id": "sess1",
            "query": "How do I fix the bug?",
            "response": "You need to update line 42",
            "files_referenced": ["bug.py"],
            "files_modified": ["bug.py", "test.py"],
            "tools_used": ["Read", "Edit"],
        }

        chat_file = ml.CHATS_DIR / "2025-12-15" / "chat-001.json"
        chat_file.parent.mkdir(parents=True, exist_ok=True)
        with open(chat_file, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f)

        # Export
        output_path = self.test_path / "export.jsonl"
        stats = ml.export_data("jsonl", output_path)

        # Check stats
        self.assertEqual(stats["records"], 1)
        self.assertEqual(stats["commits"], 0)
        self.assertEqual(stats["chats"], 1)

        # Check JSONL content
        with open(output_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        self.assertEqual(len(lines), 1)
        record = json.loads(lines[0])

        # Check record structure
        self.assertEqual(record["type"], "chat")
        self.assertEqual(record["timestamp"], "2025-12-15T11:00:00")
        self.assertEqual(record["input"], "How do I fix the bug?")
        self.assertEqual(record["output"], "You need to update line 42")
        self.assertEqual(record["context"]["session_id"], "sess1")
        self.assertIn("bug.py", record["context"]["files"])
        self.assertIn("test.py", record["context"]["files"])
        self.assertEqual(record["context"]["tools_used"], ["Read", "Edit"])

    def test_export_mixed_data_csv(self):
        """Should export both commits and chats in CSV format."""
        # Create commit
        commit_data = {
            "hash": "abc123",
            "message": "fix: Bug fix",
            "timestamp": "2025-12-15T09:00:00",
            "files_changed": ["fix.py"],
            "insertions": 5,
            "deletions": 2,
            "branch": "main",
            "session_id": "sess1",
            "hunks": [{"file": "fix.py", "change_type": "modify"}]
        }
        commit_file = ml.COMMITS_DIR / "abc123_test.json"
        with open(commit_file, 'w', encoding='utf-8') as f:
            json.dump(commit_data, f)

        # Create chat
        chat_data = {
            "timestamp": "2025-12-15T10:00:00",
            "session_id": "sess1",
            "query": "Test query",
            "response": "Test response",
            "files_referenced": ["test.py"],
            "files_modified": [],
            "tools_used": ["Read"],
        }
        chat_file = ml.CHATS_DIR / "chat-001.json"
        with open(chat_file, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f)

        # Export
        output_path = self.test_path / "export.csv"
        stats = ml.export_data("csv", output_path)

        # Check stats
        self.assertEqual(stats["records"], 2)
        self.assertEqual(stats["commits"], 1)
        self.assertEqual(stats["chats"], 1)

        # Check CSV content
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.assertEqual(len(rows), 2)

        # First row should be commit (sorted by timestamp)
        self.assertEqual(rows[0]["type"], "commit")
        self.assertEqual(rows[0]["timestamp"], "2025-12-15T09:00:00")
        self.assertEqual(rows[0]["input"], "fix: Bug fix")
        self.assertIn("fix.py", rows[0]["output"])

        # Second row should be chat
        self.assertEqual(rows[1]["type"], "chat")
        self.assertEqual(rows[1]["timestamp"], "2025-12-15T10:00:00")
        self.assertEqual(rows[1]["input"], "Test query")
        self.assertEqual(rows[1]["output"], "Test response")

    def test_export_huggingface_format(self):
        """Should export in HuggingFace Dataset dict-of-lists format."""
        # Create test data
        chat_data = {
            "timestamp": "2025-12-15T10:00:00",
            "session_id": "sess1",
            "query": "Query text",
            "response": "Response text",
            "files_referenced": ["file1.py"],
            "files_modified": ["file2.py"],
            "tools_used": ["Read", "Edit"],
        }
        chat_file = ml.CHATS_DIR / "chat-001.json"
        with open(chat_file, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f)

        # Export
        output_path = self.test_path / "export.json"
        stats = ml.export_data("huggingface", output_path)

        # Check stats
        self.assertEqual(stats["records"], 1)

        # Check HuggingFace format
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Should be dict of lists
        self.assertIsInstance(data, dict)

        # Check all required fields
        required_fields = ['type', 'timestamp', 'input', 'output',
                          'session_id', 'files', 'tools_used']
        for field in required_fields:
            self.assertIn(field, data)
            self.assertIsInstance(data[field], list)
            self.assertEqual(len(data[field]), 1)

        # Check values
        self.assertEqual(data['type'][0], 'chat')
        self.assertEqual(data['timestamp'][0], '2025-12-15T10:00:00')
        self.assertEqual(data['input'][0], 'Query text')
        self.assertEqual(data['output'][0], 'Response text')
        self.assertEqual(data['session_id'][0], 'sess1')
        self.assertIsInstance(data['files'][0], list)
        self.assertIn('file1.py', data['files'][0])
        self.assertIn('file2.py', data['files'][0])
        self.assertEqual(data['tools_used'][0], ['Read', 'Edit'])

    def test_export_huggingface_equal_lengths(self):
        """HuggingFace format should have equal-length lists."""
        # Create multiple records
        for i in range(3):
            chat_data = {
                "timestamp": f"2025-12-15T1{i}:00:00",
                "session_id": "sess1",
                "query": f"Query {i}",
                "response": f"Response {i}",
                "files_referenced": [],
                "files_modified": [],
                "tools_used": [],
            }
            chat_file = ml.CHATS_DIR / f"chat-00{i}.json"
            with open(chat_file, 'w', encoding='utf-8') as f:
                json.dump(chat_data, f)

        # Export
        output_path = self.test_path / "export.json"
        ml.export_data("huggingface", output_path)

        # Check all lists have same length
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        lengths = {key: len(value) for key, value in data.items()}
        unique_lengths = set(lengths.values())
        self.assertEqual(len(unique_lengths), 1)  # All same length
        self.assertEqual(list(unique_lengths)[0], 3)

    def test_export_invalid_format(self):
        """Should raise ValueError for invalid format."""
        output_path = self.test_path / "export.txt"
        with self.assertRaises(ValueError) as context:
            ml.export_data("invalid_format", output_path)

        self.assertIn("Unknown format", str(context.exception))

    def test_export_creates_parent_directories(self):
        """Should create parent directories if they don't exist."""
        output_path = self.test_path / "nested" / "path" / "export.jsonl"
        self.assertFalse(output_path.parent.exists())

        ml.export_data("jsonl", output_path)

        self.assertTrue(output_path.parent.exists())
        self.assertTrue(output_path.exists())

    def test_export_csv_truncates_long_fields(self):
        """CSV export should truncate long input/output fields."""
        # Create chat with very long text
        long_text = "x" * 2000
        chat_data = {
            "timestamp": "2025-12-15T10:00:00",
            "session_id": "sess1",
            "query": long_text,
            "response": long_text,
            "files_referenced": [],
            "files_modified": [],
            "tools_used": [],
        }
        chat_file = ml.CHATS_DIR / "chat-001.json"
        with open(chat_file, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f)

        # Export
        output_path = self.test_path / "export.csv"
        ml.export_data("csv", output_path)

        # Check truncation
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            row = next(reader)

        # Should be truncated to 1000 chars
        self.assertLessEqual(len(row['input']), 1000)
        self.assertLessEqual(len(row['output']), 1000)

    def test_export_csv_escapes_special_chars(self):
        """CSV export should properly escape special characters."""
        # Create chat with special chars
        chat_data = {
            "timestamp": "2025-12-15T10:00:00",
            "session_id": "sess1",
            "query": 'Text with "quotes" and, commas',
            "response": "Text with\nnewlines",
            "files_referenced": [],
            "files_modified": [],
            "tools_used": [],
        }
        chat_file = ml.CHATS_DIR / "chat-001.json"
        with open(chat_file, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f)

        # Export
        output_path = self.test_path / "export.csv"
        ml.export_data("csv", output_path)

        # Should be able to read back without errors
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            row = next(reader)

        self.assertIn('quotes', row['input'])
        self.assertIn('newlines', row['output'])

    def test_export_jsonl_valid_json_per_line(self):
        """JSONL export should have valid JSON on each line."""
        # Create multiple records
        for i in range(3):
            chat_data = {
                "timestamp": f"2025-12-15T1{i}:00:00",
                "session_id": "sess1",
                "query": f"Query {i}",
                "response": f"Response {i}",
                "files_referenced": [],
                "files_modified": [],
                "tools_used": [],
            }
            chat_file = ml.CHATS_DIR / f"chat-00{i}.json"
            with open(chat_file, 'w', encoding='utf-8') as f:
                json.dump(chat_data, f)

        # Export
        output_path = self.test_path / "export.jsonl"
        ml.export_data("jsonl", output_path)

        # Check each line is valid JSON
        with open(output_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        self.assertEqual(len(lines), 3)
        for line in lines:
            # Should not raise JSONDecodeError
            record = json.loads(line)
            self.assertIsInstance(record, dict)
            self.assertIn('type', record)
            self.assertIn('timestamp', record)

    def test_export_sorts_by_timestamp(self):
        """Export should sort records by timestamp."""
        # Create records with different timestamps (out of order)
        timestamps = ["2025-12-15T12:00:00", "2025-12-15T09:00:00", "2025-12-15T15:00:00"]

        for i, ts in enumerate(timestamps):
            chat_data = {
                "timestamp": ts,
                "session_id": "sess1",
                "query": f"Query {i}",
                "response": f"Response {i}",
                "files_referenced": [],
                "files_modified": [],
                "tools_used": [],
            }
            chat_file = ml.CHATS_DIR / f"chat-00{i}.json"
            with open(chat_file, 'w', encoding='utf-8') as f:
                json.dump(chat_data, f)

        # Export
        output_path = self.test_path / "export.jsonl"
        ml.export_data("jsonl", output_path)

        # Check order
        with open(output_path, 'r', encoding='utf-8') as f:
            records = [json.loads(line) for line in f.readlines()]

        # Should be sorted
        sorted_timestamps = sorted(timestamps)
        for i, record in enumerate(records):
            self.assertEqual(record['timestamp'], sorted_timestamps[i])


if __name__ == "__main__":
    unittest.main()
