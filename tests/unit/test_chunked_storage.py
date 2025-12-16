#!/usr/bin/env python3
"""
Unit tests for ML data collector chunked storage functionality.

Tests compression, chunking, reconstruction, and migration utilities.
"""

import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from ml_collector import chunked_storage as cs


class TestCompression(unittest.TestCase):
    """Tests for compression utilities."""

    def test_compress_small_content(self):
        """Small content should not be compressed."""
        small = "Hello, world!"
        result, compressed = cs.compress_content(small)
        self.assertEqual(result, small)
        self.assertFalse(compressed)

    def test_compress_large_content(self):
        """Large repetitive content should be compressed."""
        # Repetitive content compresses well
        large = "Hello, world! " * 1000
        result, compressed = cs.compress_content(large)
        self.assertTrue(compressed)
        self.assertLess(len(result), len(large))

    def test_compress_decompress_roundtrip(self):
        """Compression and decompression should be lossless."""
        original = "Test content with special chars: éàü 日本語 🎉" * 100
        compressed, was_compressed = cs.compress_content(original)

        if was_compressed:
            decompressed = cs.decompress_content(compressed, True)
            self.assertEqual(decompressed, original)
        else:
            # If not compressed, content should be unchanged
            self.assertEqual(compressed, original)

    def test_decompress_uncompressed(self):
        """Decompressing uncompressed content returns it unchanged."""
        content = "Not compressed"
        result = cs.decompress_content(content, False)
        self.assertEqual(result, content)

    def test_content_hash(self):
        """Content hash should be deterministic."""
        content = "Test content"
        hash1 = cs.content_hash(content)
        hash2 = cs.content_hash(content)
        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 16)  # Truncated to 16 chars

    def test_content_hash_different(self):
        """Different content should produce different hashes."""
        hash1 = cs.content_hash("Content A")
        hash2 = cs.content_hash("Content B")
        self.assertNotEqual(hash1, hash2)


class TestChunkRecord(unittest.TestCase):
    """Tests for ChunkRecord dataclass."""

    def test_to_dict(self):
        """ChunkRecord should serialize to dict."""
        record = cs.ChunkRecord(
            record_type='chat',
            record_id='chat-001',
            timestamp='2025-12-16T10:00:00',
            sequence=0,
            total_parts=1,
            compressed=False,
            content_hash='abc123',
            data={'query': 'test', 'response': 'answer'}
        )

        d = record.to_dict()
        self.assertEqual(d['record_type'], 'chat')
        self.assertEqual(d['record_id'], 'chat-001')
        self.assertEqual(d['data']['query'], 'test')

    def test_from_dict(self):
        """ChunkRecord should deserialize from dict."""
        d = {
            'record_type': 'commit',
            'record_id': 'abc123',
            'timestamp': '2025-12-16T10:00:00',
            'sequence': 0,
            'total_parts': 1,
            'compressed': True,
            'content_hash': 'def456',
            'data': {'message': 'test commit'}
        }

        record = cs.ChunkRecord.from_dict(d)
        self.assertEqual(record.record_type, 'commit')
        self.assertEqual(record.record_id, 'abc123')
        self.assertTrue(record.compressed)


class TestChunkedStorage(unittest.TestCase):
    """Tests for chunked storage operations."""

    def setUp(self):
        """Set up test environment with temporary directories."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_path = Path(self.test_dir.name)

        # Override chunked directory
        self.original_chunked_dir = cs.CHUNKED_DIR
        self.original_tracked_dir = cs.TRACKED_DIR
        cs.TRACKED_DIR = self.test_path / "tracked"
        cs.CHUNKED_DIR = cs.TRACKED_DIR / "chunked"

    def tearDown(self):
        """Clean up test environment."""
        cs.CHUNKED_DIR = self.original_chunked_dir
        cs.TRACKED_DIR = self.original_tracked_dir
        self.test_dir.cleanup()

    def test_ensure_chunked_dir(self):
        """Should create chunked directory if it doesn't exist."""
        self.assertFalse(cs.CHUNKED_DIR.exists())
        cs.ensure_chunked_dir()
        self.assertTrue(cs.CHUNKED_DIR.exists())

    def test_store_chunked_chat_small(self):
        """Should store small chat without compression."""
        chat_data = {
            'id': 'chat-001',
            'query': 'Hello',
            'response': 'Hi there!',
            'session_id': 'sess-001',
            'files_referenced': [],
            'files_modified': [],
            'tools_used': []
        }

        path = cs.store_chunked_chat(chat_data, 'test-session')
        self.assertIn('.jsonl', path)
        self.assertTrue(Path(path).exists())

        # Verify content
        with open(path, 'r') as f:
            line = f.readline()
            record = json.loads(line)

        self.assertEqual(record['record_type'], 'chat')
        self.assertEqual(record['record_id'], 'chat-001')
        self.assertFalse(record['compressed'])

    def test_store_chunked_chat_large(self):
        """Should compress large chat responses."""
        large_response = "This is a detailed response. " * 500

        chat_data = {
            'id': 'chat-002',
            'query': 'Tell me more',
            'response': large_response,
            'session_id': 'sess-001',
            'files_referenced': [],
            'files_modified': [],
            'tools_used': []
        }

        path = cs.store_chunked_chat(chat_data, 'test-session')

        with open(path, 'r') as f:
            line = f.readline()
            record = json.loads(line)

        self.assertTrue(record['compressed'])
        self.assertTrue(record['data'].get('_response_compressed'))

    def test_store_chunked_commit(self):
        """Should store commit with compressed hunks."""
        large_hunks = [{'file': f'file{i}.py', 'diff': 'x' * 100} for i in range(50)]

        commit_data = {
            'hash': 'abc123def456',
            'message': 'Test commit',
            'hunks': large_hunks
        }

        path = cs.store_chunked_commit(commit_data, 'test-session')
        self.assertTrue(Path(path).exists())

    def test_reconstruct_record(self):
        """Should reconstruct stored record."""
        chat_data = {
            'id': 'chat-reconstruct',
            'query': 'Test query',
            'response': 'Test response ' * 500,  # Large enough to compress
            'session_id': 'sess-001'
        }

        cs.store_chunked_chat(chat_data, 'test-session')

        # Reconstruct
        reconstructed = cs.reconstruct_record('chat-reconstruct')

        self.assertIsNotNone(reconstructed)
        self.assertEqual(reconstructed['id'], 'chat-reconstruct')
        self.assertEqual(reconstructed['query'], 'Test query')
        self.assertIn('Test response', reconstructed['response'])

    def test_reconstruct_nonexistent(self):
        """Should return None for nonexistent record."""
        result = cs.reconstruct_record('nonexistent-id')
        self.assertIsNone(result)

    def test_reconstruct_all(self):
        """Should reconstruct all records of a type."""
        # Store multiple chats
        for i in range(3):
            chat_data = {
                'id': f'chat-{i}',
                'query': f'Query {i}',
                'response': f'Response {i}',
                'session_id': 'sess-001'
            }
            cs.store_chunked_chat(chat_data, 'test-session')

        # Reconstruct all chats
        all_chats = cs.reconstruct_all('chat')
        self.assertEqual(len(all_chats), 3)

    def test_multiple_stores_same_file(self):
        """Multiple stores in same session should append to same file."""
        for i in range(3):
            cs.store_chunked_chat({
                'id': f'chat-{i}',
                'query': 'test',
                'response': 'test'
            }, 'same-session')

        # Should be one file with 3 lines
        files = list(cs.CHUNKED_DIR.glob("*.jsonl"))
        self.assertEqual(len(files), 1)

        with open(files[0], 'r') as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 3)


class TestDecompressRecord(unittest.TestCase):
    """Tests for record decompression."""

    def test_decompress_uncompressed_record(self):
        """Should pass through uncompressed fields."""
        data = {
            'id': 'test',
            'query': 'Hello',
            'response': 'World'
        }

        result = cs.decompress_record(data)
        self.assertEqual(result['query'], 'Hello')
        self.assertEqual(result['response'], 'World')

    def test_decompress_compressed_string(self):
        """Should decompress compressed string fields."""
        original = "Test content " * 500  # Make sure it's large enough
        compressed, was_compressed = cs.compress_content(original)

        # Only test if it was actually compressed
        if not was_compressed:
            self.skipTest("Content was not large enough to compress")

        data = {
            'id': 'test',
            'response': compressed,
            '_response_compressed': True
        }

        result = cs.decompress_record(data)
        self.assertEqual(result['response'], original)
        self.assertNotIn('_response_compressed', result)

    def test_decompress_compressed_list(self):
        """Should decompress compressed list fields."""
        original_list = [{'a': i, 'data': 'x' * 100} for i in range(100)]
        list_json = json.dumps(original_list)
        compressed, was_compressed = cs.compress_content(list_json)

        # Only test if it was actually compressed
        if not was_compressed:
            self.skipTest("Content was not large enough to compress")

        data = {
            'hunks': compressed,
            '_hunks_compressed': True,
            '_hunks_was_list': True
        }

        result = cs.decompress_record(data)
        self.assertEqual(result['hunks'], original_list)


class TestCompaction(unittest.TestCase):
    """Tests for chunk compaction."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_path = Path(self.test_dir.name)

        self.original_chunked_dir = cs.CHUNKED_DIR
        self.original_tracked_dir = cs.TRACKED_DIR
        cs.TRACKED_DIR = self.test_path / "tracked"
        cs.CHUNKED_DIR = cs.TRACKED_DIR / "chunked"
        cs.ensure_chunked_dir()

    def tearDown(self):
        """Clean up."""
        cs.CHUNKED_DIR = self.original_chunked_dir
        cs.TRACKED_DIR = self.original_tracked_dir
        self.test_dir.cleanup()

    def test_compact_no_files(self):
        """Compaction with no files should return zeros."""
        # Remove the directory
        import shutil
        shutil.rmtree(cs.CHUNKED_DIR)

        result = cs.compact_chunks()
        self.assertEqual(result['files_before'], 0)
        self.assertEqual(result['files_after'], 0)

    def test_compact_recent_files_only(self):
        """Recent files should not be compacted."""
        # Create a recent file
        cs.store_chunked_chat({
            'id': 'recent-chat',
            'query': 'test',
            'response': 'test'
        }, 'recent-session')

        result = cs.compact_chunks(keep_days=30)

        # Should not compact recent files
        self.assertEqual(result['files_before'], result['files_after'])


class TestStats(unittest.TestCase):
    """Tests for statistics."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_path = Path(self.test_dir.name)

        self.original_chunked_dir = cs.CHUNKED_DIR
        self.original_tracked_dir = cs.TRACKED_DIR
        cs.TRACKED_DIR = self.test_path / "tracked"
        cs.CHUNKED_DIR = cs.TRACKED_DIR / "chunked"

    def tearDown(self):
        """Clean up."""
        cs.CHUNKED_DIR = self.original_chunked_dir
        cs.TRACKED_DIR = self.original_tracked_dir
        self.test_dir.cleanup()

    def test_stats_empty(self):
        """Stats for empty storage."""
        stats = cs.get_chunked_stats()
        self.assertEqual(stats['total_files'], 0)
        self.assertEqual(stats['total_bytes'], 0)
        self.assertEqual(stats['total_records'], 0)

    def test_stats_with_data(self):
        """Stats after storing data."""
        cs.ensure_chunked_dir()

        for i in range(5):
            cs.store_chunked_chat({
                'id': f'chat-{i}',
                'query': 'test',
                'response': 'test'
            }, 'test-session')

        stats = cs.get_chunked_stats()
        self.assertEqual(stats['total_files'], 1)
        self.assertEqual(stats['total_records'], 5)
        self.assertGreater(stats['total_bytes'], 0)
        self.assertIn('chats', stats['by_type'])


class TestMigration(unittest.TestCase):
    """Tests for migration utilities."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_path = Path(self.test_dir.name)

        self.original_chunked_dir = cs.CHUNKED_DIR
        self.original_tracked_dir = cs.TRACKED_DIR
        cs.TRACKED_DIR = self.test_path / "tracked"
        cs.CHUNKED_DIR = cs.TRACKED_DIR / "chunked"

        # Create source directory with JSON files
        self.source_dir = self.test_path / "source"
        self.source_dir.mkdir()

    def tearDown(self):
        """Clean up."""
        cs.CHUNKED_DIR = self.original_chunked_dir
        cs.TRACKED_DIR = self.original_tracked_dir
        self.test_dir.cleanup()

    def test_migrate_empty_dir(self):
        """Migration of empty directory returns 0."""
        empty_dir = self.test_path / "empty"
        result = cs.migrate_to_chunked(empty_dir, 'chat', 'migrate-session')
        self.assertEqual(result, 0)

    def test_migrate_chat_files(self):
        """Should migrate existing chat JSON files."""
        # Create some chat files
        for i in range(3):
            chat_file = self.source_dir / f"chat-{i}.json"
            with open(chat_file, 'w') as f:
                json.dump({
                    'id': f'chat-{i}',
                    'query': f'Query {i}',
                    'response': f'Response {i}'
                }, f)

        result = cs.migrate_to_chunked(self.source_dir, 'chat', 'migrate-session')
        self.assertEqual(result, 3)

        # Verify they can be reconstructed
        all_chats = cs.reconstruct_all('chat')
        self.assertEqual(len(all_chats), 3)


if __name__ == "__main__":
    unittest.main()
