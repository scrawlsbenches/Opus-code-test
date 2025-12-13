"""
Unit Tests for cortical/chunk_index.py
========================================

Task: Comprehensive unit tests for chunk-based indexing.

Coverage goal: 90%+

Test Categories:
1. ChunkOperation: Serialization and edge cases
2. Chunk: Filename generation and serialization
3. ChunkWriter: Document operations and file writing
4. ChunkLoader: Loading, replaying, and cache validation
5. ChunkCompactor: Compaction logic
6. Utility Functions: Manifest comparison
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

from cortical.chunk_index import (
    ChunkOperation,
    Chunk,
    ChunkWriter,
    ChunkLoader,
    ChunkCompactor,
    get_changes_from_manifest,
    CHUNK_VERSION,
    DEFAULT_WARN_SIZE_KB,
)


class TestChunkOperation(unittest.TestCase):
    """Test ChunkOperation dataclass and serialization."""

    def test_to_dict_add_operation(self):
        """Test serialization of add operation with all fields."""
        op = ChunkOperation(
            op='add',
            doc_id='doc1',
            content='Test content',
            mtime=1234567890.0,
            metadata={'doc_type': 'test'}
        )
        d = op.to_dict()

        self.assertEqual(d['op'], 'add')
        self.assertEqual(d['doc_id'], 'doc1')
        self.assertEqual(d['content'], 'Test content')
        self.assertEqual(d['mtime'], 1234567890.0)
        self.assertEqual(d['metadata'], {'doc_type': 'test'})

    def test_to_dict_delete_operation(self):
        """Test serialization of delete operation (no content)."""
        op = ChunkOperation(op='delete', doc_id='doc2')
        d = op.to_dict()

        self.assertEqual(d['op'], 'delete')
        self.assertEqual(d['doc_id'], 'doc2')
        self.assertNotIn('content', d)
        self.assertNotIn('mtime', d)
        self.assertNotIn('metadata', d)

    def test_to_dict_modify_with_partial_fields(self):
        """Test modify operation with only content."""
        op = ChunkOperation(op='modify', doc_id='doc3', content='New content')
        d = op.to_dict()

        self.assertEqual(d['op'], 'modify')
        self.assertEqual(d['doc_id'], 'doc3')
        self.assertEqual(d['content'], 'New content')
        self.assertNotIn('mtime', d)
        self.assertNotIn('metadata', d)

    def test_from_dict_full(self):
        """Test deserialization with all fields."""
        d = {
            'op': 'add',
            'doc_id': 'doc1',
            'content': 'Test',
            'mtime': 123.456,
            'metadata': {'type': 'python'}
        }
        op = ChunkOperation.from_dict(d)

        self.assertEqual(op.op, 'add')
        self.assertEqual(op.doc_id, 'doc1')
        self.assertEqual(op.content, 'Test')
        self.assertEqual(op.mtime, 123.456)
        self.assertEqual(op.metadata, {'type': 'python'})

    def test_from_dict_minimal(self):
        """Test deserialization with only required fields."""
        d = {'op': 'delete', 'doc_id': 'doc2'}
        op = ChunkOperation.from_dict(d)

        self.assertEqual(op.op, 'delete')
        self.assertEqual(op.doc_id, 'doc2')
        self.assertIsNone(op.content)
        self.assertIsNone(op.mtime)
        self.assertIsNone(op.metadata)

    def test_roundtrip_serialization(self):
        """Test that to_dict -> from_dict preserves data."""
        original = ChunkOperation(
            op='modify',
            doc_id='doc3',
            content='Content',
            mtime=999.0,
            metadata={'key': 'value'}
        )
        d = original.to_dict()
        restored = ChunkOperation.from_dict(d)

        self.assertEqual(original.op, restored.op)
        self.assertEqual(original.doc_id, restored.doc_id)
        self.assertEqual(original.content, restored.content)
        self.assertEqual(original.mtime, restored.mtime)
        self.assertEqual(original.metadata, restored.metadata)


class TestChunk(unittest.TestCase):
    """Test Chunk dataclass and filename generation."""

    def test_to_dict_empty_operations(self):
        """Test chunk serialization with no operations."""
        chunk = Chunk(
            version=1,
            timestamp='2025-12-10T21:53:45',
            session_id='a1b2c3d4',
            branch='main'
        )
        d = chunk.to_dict()

        self.assertEqual(d['version'], 1)
        self.assertEqual(d['timestamp'], '2025-12-10T21:53:45')
        self.assertEqual(d['session_id'], 'a1b2c3d4')
        self.assertEqual(d['branch'], 'main')
        self.assertEqual(d['operations'], [])

    def test_to_dict_with_operations(self):
        """Test chunk serialization with operations."""
        ops = [
            ChunkOperation(op='add', doc_id='doc1', content='Content 1'),
            ChunkOperation(op='delete', doc_id='doc2')
        ]
        chunk = Chunk(
            version=1,
            timestamp='2025-12-10T22:00:00',
            session_id='test123',
            branch='feature',
            operations=ops
        )
        d = chunk.to_dict()

        self.assertEqual(len(d['operations']), 2)
        self.assertEqual(d['operations'][0]['op'], 'add')
        self.assertEqual(d['operations'][1]['op'], 'delete')

    def test_from_dict_with_version(self):
        """Test deserialization with explicit version."""
        d = {
            'version': 1,
            'timestamp': '2025-12-10T21:53:45',
            'session_id': 'abc123',
            'branch': 'main',
            'operations': []
        }
        chunk = Chunk.from_dict(d)

        self.assertEqual(chunk.version, 1)
        self.assertEqual(chunk.timestamp, '2025-12-10T21:53:45')
        self.assertEqual(chunk.session_id, 'abc123')
        self.assertEqual(chunk.branch, 'main')
        self.assertEqual(len(chunk.operations), 0)

    def test_from_dict_defaults(self):
        """Test deserialization with default values."""
        d = {
            'timestamp': '2025-12-10T21:53:45',
            'session_id': 'abc123',
            'operations': []
        }
        chunk = Chunk.from_dict(d)

        self.assertEqual(chunk.version, 1)  # Default
        self.assertEqual(chunk.branch, 'unknown')  # Default

    def test_from_dict_with_operations(self):
        """Test deserialization restores operations."""
        d = {
            'version': 1,
            'timestamp': '2025-12-10T22:00:00',
            'session_id': 'test',
            'branch': 'main',
            'operations': [
                {'op': 'add', 'doc_id': 'doc1', 'content': 'Test'},
                {'op': 'delete', 'doc_id': 'doc2'}
            ]
        }
        chunk = Chunk.from_dict(d)

        self.assertEqual(len(chunk.operations), 2)
        self.assertEqual(chunk.operations[0].op, 'add')
        self.assertEqual(chunk.operations[1].op, 'delete')

    def test_get_filename_format(self):
        """Test filename generation follows format."""
        chunk = Chunk(
            version=1,
            timestamp='2025-12-10T21:53:45',
            session_id='a1b2c3d4e5f6',
            branch='main'
        )
        filename = chunk.get_filename()

        # Format: YYYY-MM-DD_HH-MM-SS_sessionid.json
        self.assertEqual(filename, '2025-12-10_21-53-45_a1b2c3d4.json')

    def test_get_filename_short_session_id(self):
        """Test filename uses first 8 chars of session_id."""
        chunk = Chunk(
            version=1,
            timestamp='2025-01-15T09:30:15',
            session_id='short',
            branch='main'
        )
        filename = chunk.get_filename()

        # Should only use up to 8 chars
        self.assertTrue(filename.endswith('_short.json'))

    def test_roundtrip_serialization(self):
        """Test that to_dict -> from_dict preserves chunk."""
        original = Chunk(
            version=1,
            timestamp='2025-12-10T21:53:45',
            session_id='test123',
            branch='feature',
            operations=[
                ChunkOperation(op='add', doc_id='doc1', content='C1'),
                ChunkOperation(op='modify', doc_id='doc2', content='C2')
            ]
        )
        d = original.to_dict()
        restored = Chunk.from_dict(d)

        self.assertEqual(original.version, restored.version)
        self.assertEqual(original.timestamp, restored.timestamp)
        self.assertEqual(original.session_id, restored.session_id)
        self.assertEqual(original.branch, restored.branch)
        self.assertEqual(len(original.operations), len(restored.operations))


class TestChunkWriter(unittest.TestCase):
    """Test ChunkWriter class."""

    def setUp(self):
        """Create temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_creates_session_id(self):
        """Test initialization creates unique session ID."""
        writer1 = ChunkWriter(self.temp_dir)
        writer2 = ChunkWriter(self.temp_dir)

        self.assertIsNotNone(writer1.session_id)
        self.assertIsNotNone(writer2.session_id)
        self.assertNotEqual(writer1.session_id, writer2.session_id)
        self.assertEqual(len(writer1.session_id), 16)

    def test_init_sets_timestamp(self):
        """Test initialization sets ISO timestamp."""
        writer = ChunkWriter(self.temp_dir)

        # Should be valid ISO format
        datetime.fromisoformat(writer.timestamp)

    @patch('subprocess.run')
    def test_get_git_branch_success(self, mock_run):
        """Test git branch detection when git is available."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='feature-branch\n'
        )

        writer = ChunkWriter(self.temp_dir)
        self.assertEqual(writer.branch, 'feature-branch')

    @patch('subprocess.run')
    def test_get_git_branch_failure(self, mock_run):
        """Test git branch defaults to 'unknown' on failure."""
        mock_run.return_value = MagicMock(returncode=1)

        writer = ChunkWriter(self.temp_dir)
        self.assertEqual(writer.branch, 'unknown')

    @patch('subprocess.run')
    def test_get_git_branch_timeout(self, mock_run):
        """Test git branch handles timeout."""
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired('git', 5)

        writer = ChunkWriter(self.temp_dir)
        self.assertEqual(writer.branch, 'unknown')

    @patch('subprocess.run')
    def test_get_git_branch_not_found(self, mock_run):
        """Test git branch handles missing git."""
        mock_run.side_effect = FileNotFoundError()

        writer = ChunkWriter(self.temp_dir)
        self.assertEqual(writer.branch, 'unknown')

    def test_add_document(self):
        """Test adding document operation."""
        writer = ChunkWriter(self.temp_dir)
        writer.add_document('doc1', 'Content 1', mtime=123.0)

        self.assertEqual(len(writer.operations), 1)
        self.assertEqual(writer.operations[0].op, 'add')
        self.assertEqual(writer.operations[0].doc_id, 'doc1')
        self.assertEqual(writer.operations[0].content, 'Content 1')
        self.assertEqual(writer.operations[0].mtime, 123.0)

    def test_add_document_with_metadata(self):
        """Test adding document with metadata."""
        writer = ChunkWriter(self.temp_dir)
        metadata = {'doc_type': 'python', 'headings': ['Header 1']}
        writer.add_document('doc1', 'Content', metadata=metadata)

        self.assertEqual(writer.operations[0].metadata, metadata)

    def test_modify_document(self):
        """Test modifying document operation."""
        writer = ChunkWriter(self.temp_dir)
        writer.modify_document('doc2', 'New content', mtime=456.0)

        self.assertEqual(len(writer.operations), 1)
        self.assertEqual(writer.operations[0].op, 'modify')
        self.assertEqual(writer.operations[0].doc_id, 'doc2')
        self.assertEqual(writer.operations[0].content, 'New content')

    def test_delete_document(self):
        """Test deleting document operation."""
        writer = ChunkWriter(self.temp_dir)
        writer.delete_document('doc3')

        self.assertEqual(len(writer.operations), 1)
        self.assertEqual(writer.operations[0].op, 'delete')
        self.assertEqual(writer.operations[0].doc_id, 'doc3')
        self.assertIsNone(writer.operations[0].content)

    def test_has_operations_empty(self):
        """Test has_operations when no operations."""
        writer = ChunkWriter(self.temp_dir)
        self.assertFalse(writer.has_operations())

    def test_has_operations_with_operations(self):
        """Test has_operations when operations exist."""
        writer = ChunkWriter(self.temp_dir)
        writer.add_document('doc1', 'Content')
        self.assertTrue(writer.has_operations())

    def test_save_no_operations(self):
        """Test save returns None when no operations."""
        writer = ChunkWriter(self.temp_dir)
        result = writer.save()

        self.assertIsNone(result)
        # No file should be created
        self.assertEqual(len(list(Path(self.temp_dir).glob('*.json'))), 0)

    def test_save_creates_file(self):
        """Test save creates chunk file."""
        writer = ChunkWriter(self.temp_dir)
        writer.add_document('doc1', 'Content 1')
        writer.modify_document('doc2', 'Content 2')

        filepath = writer.save()

        self.assertIsNotNone(filepath)
        self.assertTrue(filepath.exists())
        self.assertTrue(filepath.name.endswith('.json'))

    def test_save_creates_directory(self):
        """Test save creates chunks directory if needed."""
        chunks_dir = Path(self.temp_dir) / 'new_chunks'
        writer = ChunkWriter(str(chunks_dir))
        writer.add_document('doc1', 'Content')

        filepath = writer.save()

        self.assertTrue(chunks_dir.exists())
        self.assertTrue(filepath.exists())

    def test_save_valid_json(self):
        """Test saved file contains valid JSON."""
        writer = ChunkWriter(self.temp_dir)
        writer.add_document('doc1', 'Test content')
        filepath = writer.save()

        with open(filepath, 'r') as f:
            data = json.load(f)

        self.assertEqual(data['version'], CHUNK_VERSION)
        self.assertEqual(len(data['operations']), 1)

    def test_save_preserves_operations(self):
        """Test saved file preserves all operations."""
        writer = ChunkWriter(self.temp_dir)
        writer.add_document('doc1', 'Content 1', mtime=100.0)
        writer.modify_document('doc2', 'Content 2', mtime=200.0)
        writer.delete_document('doc3')

        filepath = writer.save()

        with open(filepath, 'r') as f:
            data = json.load(f)

        self.assertEqual(len(data['operations']), 3)
        self.assertEqual(data['operations'][0]['op'], 'add')
        self.assertEqual(data['operations'][1]['op'], 'modify')
        self.assertEqual(data['operations'][2]['op'], 'delete')

    def test_save_size_warning(self):
        """Test save warns on large chunks."""
        writer = ChunkWriter(self.temp_dir)
        # Create large content to trigger warning
        large_content = 'x' * (2 * 1024 * 1024)  # 2MB
        writer.add_document('doc1', large_content)

        with self.assertWarns(UserWarning) as cm:
            writer.save(warn_size_kb=1024)

        self.assertIn('exceeds', str(cm.warning))

    def test_save_no_warning_small_chunk(self):
        """Test save doesn't warn on small chunks."""
        writer = ChunkWriter(self.temp_dir)
        writer.add_document('doc1', 'Small content')

        # Should not raise warning
        writer.save(warn_size_kb=1024)

    def test_save_warning_disabled(self):
        """Test warning can be disabled."""
        writer = ChunkWriter(self.temp_dir)
        large_content = 'x' * (2 * 1024 * 1024)
        writer.add_document('doc1', large_content)

        # No warning with warn_size_kb=0
        writer.save(warn_size_kb=0)


class TestChunkLoader(unittest.TestCase):
    """Test ChunkLoader class."""

    def setUp(self):
        """Create temporary directory and sample chunks."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_chunk_file(self, timestamp, session_id, operations):
        """Helper to create a chunk file."""
        chunk = Chunk(
            version=1,
            timestamp=timestamp,
            session_id=session_id,
            branch='main',
            operations=operations
        )
        filename = chunk.get_filename()
        filepath = Path(self.temp_dir) / filename

        with open(filepath, 'w') as f:
            json.dump(chunk.to_dict(), f)

        return filepath

    def test_get_chunk_files_empty(self):
        """Test get_chunk_files when no chunks exist."""
        loader = ChunkLoader(self.temp_dir)
        files = loader.get_chunk_files()

        self.assertEqual(len(files), 0)

    def test_get_chunk_files_nonexistent_dir(self):
        """Test get_chunk_files when directory doesn't exist."""
        loader = ChunkLoader(self.temp_dir + '_nonexistent')
        files = loader.get_chunk_files()

        self.assertEqual(len(files), 0)

    def test_get_chunk_files_sorted(self):
        """Test get_chunk_files returns files sorted by timestamp."""
        self._create_chunk_file('2025-12-10T22:00:00', 'b', [])
        self._create_chunk_file('2025-12-10T21:00:00', 'a', [])
        self._create_chunk_file('2025-12-10T23:00:00', 'c', [])

        loader = ChunkLoader(self.temp_dir)
        files = loader.get_chunk_files()

        self.assertEqual(len(files), 3)
        # Check sorted order by filename
        names = [f.name for f in files]
        self.assertEqual(names, sorted(names))

    def test_load_chunk(self):
        """Test loading a single chunk file."""
        filepath = self._create_chunk_file(
            '2025-12-10T21:00:00',
            'test',
            [ChunkOperation(op='add', doc_id='doc1', content='Test')]
        )

        loader = ChunkLoader(self.temp_dir)
        chunk = loader.load_chunk(filepath)

        self.assertEqual(chunk.session_id, 'test')
        self.assertEqual(len(chunk.operations), 1)
        self.assertEqual(chunk.operations[0].doc_id, 'doc1')

    def test_load_all_empty(self):
        """Test load_all with no chunks."""
        loader = ChunkLoader(self.temp_dir)
        docs = loader.load_all()

        self.assertEqual(len(docs), 0)

    def test_load_all_single_add(self):
        """Test load_all with single add operation."""
        self._create_chunk_file(
            '2025-12-10T21:00:00',
            'test',
            [ChunkOperation(op='add', doc_id='doc1', content='Content 1')]
        )

        loader = ChunkLoader(self.temp_dir)
        docs = loader.load_all()

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs['doc1'], 'Content 1')

    def test_load_all_multiple_operations(self):
        """Test load_all with multiple operations."""
        self._create_chunk_file(
            '2025-12-10T21:00:00',
            'test',
            [
                ChunkOperation(op='add', doc_id='doc1', content='Content 1'),
                ChunkOperation(op='add', doc_id='doc2', content='Content 2'),
                ChunkOperation(op='add', doc_id='doc3', content='Content 3')
            ]
        )

        loader = ChunkLoader(self.temp_dir)
        docs = loader.load_all()

        self.assertEqual(len(docs), 3)
        self.assertEqual(docs['doc1'], 'Content 1')
        self.assertEqual(docs['doc2'], 'Content 2')

    def test_load_all_modify_operation(self):
        """Test load_all handles modify operations."""
        self._create_chunk_file(
            '2025-12-10T21:00:00',
            'test1',
            [ChunkOperation(op='add', doc_id='doc1', content='Original')]
        )
        self._create_chunk_file(
            '2025-12-10T22:00:00',
            'test2',
            [ChunkOperation(op='modify', doc_id='doc1', content='Modified')]
        )

        loader = ChunkLoader(self.temp_dir)
        docs = loader.load_all()

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs['doc1'], 'Modified')

    def test_load_all_delete_operation(self):
        """Test load_all handles delete operations."""
        self._create_chunk_file(
            '2025-12-10T21:00:00',
            'test1',
            [
                ChunkOperation(op='add', doc_id='doc1', content='Content 1'),
                ChunkOperation(op='add', doc_id='doc2', content='Content 2')
            ]
        )
        self._create_chunk_file(
            '2025-12-10T22:00:00',
            'test2',
            [ChunkOperation(op='delete', doc_id='doc1')]
        )

        loader = ChunkLoader(self.temp_dir)
        docs = loader.load_all()

        self.assertEqual(len(docs), 1)
        self.assertNotIn('doc1', docs)
        self.assertEqual(docs['doc2'], 'Content 2')

    def test_load_all_preserves_mtimes(self):
        """Test load_all preserves modification times."""
        self._create_chunk_file(
            '2025-12-10T21:00:00',
            'test',
            [
                ChunkOperation(op='add', doc_id='doc1', content='C1', mtime=100.0),
                ChunkOperation(op='add', doc_id='doc2', content='C2', mtime=200.0)
            ]
        )

        loader = ChunkLoader(self.temp_dir)
        loader.load_all()
        mtimes = loader.get_mtimes()

        self.assertEqual(mtimes['doc1'], 100.0)
        self.assertEqual(mtimes['doc2'], 200.0)

    def test_load_all_preserves_metadata(self):
        """Test load_all preserves document metadata."""
        self._create_chunk_file(
            '2025-12-10T21:00:00',
            'test',
            [
                ChunkOperation(
                    op='add',
                    doc_id='doc1',
                    content='C1',
                    metadata={'doc_type': 'python'}
                )
            ]
        )

        loader = ChunkLoader(self.temp_dir)
        loader.load_all()
        metadata = loader.get_metadata()

        self.assertEqual(metadata['doc1'], {'doc_type': 'python'})

    def test_load_all_modify_with_mtime_and_metadata(self):
        """Test modify operation preserves mtime and metadata."""
        self._create_chunk_file(
            '2025-12-10T21:00:00',
            'test1',
            [ChunkOperation(op='add', doc_id='doc1', content='Original')]
        )
        self._create_chunk_file(
            '2025-12-10T22:00:00',
            'test2',
            [
                ChunkOperation(
                    op='modify',
                    doc_id='doc1',
                    content='Modified',
                    mtime=999.0,
                    metadata={'updated': True}
                )
            ]
        )

        loader = ChunkLoader(self.temp_dir)
        loader.load_all()
        mtimes = loader.get_mtimes()
        metadata = loader.get_metadata()

        self.assertEqual(mtimes['doc1'], 999.0)
        self.assertEqual(metadata['doc1'], {'updated': True})

    def test_load_all_idempotent(self):
        """Test load_all can be called multiple times."""
        self._create_chunk_file(
            '2025-12-10T21:00:00',
            'test',
            [ChunkOperation(op='add', doc_id='doc1', content='Content')]
        )

        loader = ChunkLoader(self.temp_dir)
        docs1 = loader.load_all()
        docs2 = loader.load_all()

        self.assertEqual(docs1, docs2)

    def test_get_documents_auto_loads(self):
        """Test get_documents calls load_all if needed."""
        self._create_chunk_file(
            '2025-12-10T21:00:00',
            'test',
            [ChunkOperation(op='add', doc_id='doc1', content='Content')]
        )

        loader = ChunkLoader(self.temp_dir)
        docs = loader.get_documents()

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs['doc1'], 'Content')

    def test_get_mtimes_auto_loads(self):
        """Test get_mtimes calls load_all if needed."""
        self._create_chunk_file(
            '2025-12-10T21:00:00',
            'test',
            [ChunkOperation(op='add', doc_id='doc1', content='C', mtime=123.0)]
        )

        loader = ChunkLoader(self.temp_dir)
        mtimes = loader.get_mtimes()

        self.assertEqual(mtimes['doc1'], 123.0)

    def test_get_metadata_auto_loads(self):
        """Test get_metadata calls load_all if needed."""
        self._create_chunk_file(
            '2025-12-10T21:00:00',
            'test',
            [
                ChunkOperation(
                    op='add',
                    doc_id='doc1',
                    content='C',
                    metadata={'type': 'test'}
                )
            ]
        )

        loader = ChunkLoader(self.temp_dir)
        metadata = loader.get_metadata()

        self.assertEqual(metadata['doc1'], {'type': 'test'})

    def test_get_chunks(self):
        """Test get_chunks returns loaded chunks."""
        self._create_chunk_file('2025-12-10T21:00:00', 'a', [])
        self._create_chunk_file('2025-12-10T22:00:00', 'b', [])

        loader = ChunkLoader(self.temp_dir)
        chunks = loader.get_chunks()

        self.assertEqual(len(chunks), 2)

    def test_compute_hash_empty(self):
        """Test compute_hash with no documents."""
        loader = ChunkLoader(self.temp_dir)
        h = loader.compute_hash()

        self.assertIsNotNone(h)
        self.assertEqual(len(h), 16)

    def test_compute_hash_deterministic(self):
        """Test compute_hash is deterministic."""
        self._create_chunk_file(
            '2025-12-10T21:00:00',
            'test',
            [
                ChunkOperation(op='add', doc_id='doc1', content='Content 1'),
                ChunkOperation(op='add', doc_id='doc2', content='Content 2')
            ]
        )

        loader1 = ChunkLoader(self.temp_dir)
        loader2 = ChunkLoader(self.temp_dir)

        h1 = loader1.compute_hash()
        h2 = loader2.compute_hash()

        self.assertEqual(h1, h2)

    def test_compute_hash_changes_with_content(self):
        """Test compute_hash changes when content changes."""
        # Create first version
        self._create_chunk_file(
            '2025-12-10T21:00:00',
            'test',
            [ChunkOperation(op='add', doc_id='doc1', content='Original')]
        )
        loader1 = ChunkLoader(self.temp_dir)
        h1 = loader1.compute_hash()

        # Create modified version
        self.tearDown()
        self.setUp()
        self._create_chunk_file(
            '2025-12-10T21:00:00',
            'test',
            [ChunkOperation(op='add', doc_id='doc1', content='Modified')]
        )
        loader2 = ChunkLoader(self.temp_dir)
        h2 = loader2.compute_hash()

        self.assertNotEqual(h1, h2)

    def test_is_cache_valid_no_cache(self):
        """Test is_cache_valid when cache doesn't exist."""
        loader = ChunkLoader(self.temp_dir)
        cache_path = Path(self.temp_dir) / 'cache.pkl'

        self.assertFalse(loader.is_cache_valid(str(cache_path)))

    def test_is_cache_valid_no_hash_file(self):
        """Test is_cache_valid when hash file doesn't exist."""
        loader = ChunkLoader(self.temp_dir)
        cache_path = Path(self.temp_dir) / 'cache.pkl'

        # Create cache file but no hash
        cache_path.touch()

        self.assertFalse(loader.is_cache_valid(str(cache_path)))

    def test_is_cache_valid_matching_hash(self):
        """Test is_cache_valid when hash matches."""
        self._create_chunk_file(
            '2025-12-10T21:00:00',
            'test',
            [ChunkOperation(op='add', doc_id='doc1', content='Content')]
        )

        loader = ChunkLoader(self.temp_dir)
        cache_path = Path(self.temp_dir) / 'cache.pkl'
        cache_path.touch()

        # Save hash
        loader.save_cache_hash(str(cache_path))

        # Verify valid
        self.assertTrue(loader.is_cache_valid(str(cache_path)))

    def test_is_cache_valid_mismatched_hash(self):
        """Test is_cache_valid when hash doesn't match."""
        self._create_chunk_file(
            '2025-12-10T21:00:00',
            'test',
            [ChunkOperation(op='add', doc_id='doc1', content='Original')]
        )

        loader = ChunkLoader(self.temp_dir)
        cache_path = Path(self.temp_dir) / 'cache.pkl'
        cache_path.touch()
        loader.save_cache_hash(str(cache_path))

        # Modify chunks
        self._create_chunk_file(
            '2025-12-10T22:00:00',
            'test2',
            [ChunkOperation(op='modify', doc_id='doc1', content='Modified')]
        )

        # New loader with different state
        loader2 = ChunkLoader(self.temp_dir)
        self.assertFalse(loader2.is_cache_valid(str(cache_path)))

    def test_is_cache_valid_custom_hash_path(self):
        """Test is_cache_valid with custom hash file path."""
        loader = ChunkLoader(self.temp_dir)
        cache_path = Path(self.temp_dir) / 'cache.pkl'
        hash_path = Path(self.temp_dir) / 'custom.hash'

        cache_path.touch()
        loader.save_cache_hash(str(cache_path), str(hash_path))

        self.assertTrue(loader.is_cache_valid(str(cache_path), str(hash_path)))

    def test_is_cache_valid_corrupted_hash_file(self):
        """Test is_cache_valid handles corrupted hash file."""
        loader = ChunkLoader(self.temp_dir)
        cache_path = Path(self.temp_dir) / 'cache.pkl'
        hash_path = Path(self.temp_dir) / 'cache.pkl.hash'

        cache_path.touch()
        hash_path.touch()

        # Make hash file unreadable by using invalid permissions mock
        with patch('builtins.open', side_effect=IOError('Cannot read')):
            self.assertFalse(loader.is_cache_valid(str(cache_path)))

    def test_save_cache_hash(self):
        """Test save_cache_hash creates hash file."""
        self._create_chunk_file(
            '2025-12-10T21:00:00',
            'test',
            [ChunkOperation(op='add', doc_id='doc1', content='Content')]
        )

        loader = ChunkLoader(self.temp_dir)
        cache_path = Path(self.temp_dir) / 'cache.pkl'
        loader.save_cache_hash(str(cache_path))

        hash_file = Path(str(cache_path) + '.hash')
        self.assertTrue(hash_file.exists())

    def test_get_stats_empty(self):
        """Test get_stats with no chunks."""
        loader = ChunkLoader(self.temp_dir)
        stats = loader.get_stats()

        self.assertEqual(stats['chunk_count'], 0)
        self.assertEqual(stats['document_count'], 0)
        self.assertEqual(stats['total_operations'], 0)
        self.assertEqual(stats['add_operations'], 0)
        self.assertEqual(stats['modify_operations'], 0)
        self.assertEqual(stats['delete_operations'], 0)

    def test_get_stats_with_operations(self):
        """Test get_stats counts operations correctly."""
        self._create_chunk_file(
            '2025-12-10T21:00:00',
            'test1',
            [
                ChunkOperation(op='add', doc_id='doc1', content='C1'),
                ChunkOperation(op='add', doc_id='doc2', content='C2')
            ]
        )
        self._create_chunk_file(
            '2025-12-10T22:00:00',
            'test2',
            [
                ChunkOperation(op='modify', doc_id='doc1', content='C1m'),
                ChunkOperation(op='delete', doc_id='doc2')
            ]
        )

        loader = ChunkLoader(self.temp_dir)
        stats = loader.get_stats()

        self.assertEqual(stats['chunk_count'], 2)
        self.assertEqual(stats['document_count'], 1)  # doc1 remains
        self.assertEqual(stats['total_operations'], 4)
        self.assertEqual(stats['add_operations'], 2)
        self.assertEqual(stats['modify_operations'], 1)
        self.assertEqual(stats['delete_operations'], 1)
        self.assertIn('hash', stats)


class TestChunkCompactor(unittest.TestCase):
    """Test ChunkCompactor class."""

    def setUp(self):
        """Create temporary directory and sample chunks."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_chunk_file(self, timestamp, session_id, operations):
        """Helper to create a chunk file."""
        chunk = Chunk(
            version=1,
            timestamp=timestamp,
            session_id=session_id,
            branch='main',
            operations=operations
        )
        filename = chunk.get_filename()
        filepath = Path(self.temp_dir) / filename

        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(chunk.to_dict(), f)

        return filepath

    def test_compact_no_chunks(self):
        """Test compact with no chunks."""
        compactor = ChunkCompactor(self.temp_dir)
        result = compactor.compact()

        self.assertEqual(result['status'], 'no_chunks')
        self.assertEqual(result['compacted'], 0)

    def test_compact_dry_run(self):
        """Test compact in dry-run mode."""
        self._create_chunk_file(
            '2025-12-10T21:00:00',
            'a',
            [ChunkOperation(op='add', doc_id='doc1', content='C1')]
        )
        self._create_chunk_file(
            '2025-12-10T22:00:00',
            'b',
            [ChunkOperation(op='add', doc_id='doc2', content='C2')]
        )

        compactor = ChunkCompactor(self.temp_dir)
        result = compactor.compact(dry_run=True)

        self.assertEqual(result['status'], 'dry_run')
        self.assertEqual(result['would_compact'], 2)
        self.assertEqual(result['would_keep'], 0)

        # Files should still exist
        self.assertEqual(len(list(Path(self.temp_dir).glob('*.json'))), 2)

    def test_compact_all(self):
        """Test compacting all chunks."""
        self._create_chunk_file(
            '2025-12-10T21:00:00',
            'a',
            [ChunkOperation(op='add', doc_id='doc1', content='C1')]
        )
        self._create_chunk_file(
            '2025-12-10T22:00:00',
            'b',
            [ChunkOperation(op='add', doc_id='doc2', content='C2')]
        )

        compactor = ChunkCompactor(self.temp_dir)
        result = compactor.compact()

        self.assertEqual(result['status'], 'compacted')
        self.assertEqual(result['compacted'], 2)
        self.assertEqual(result['documents'], 2)

        # Should have one compacted file
        files = list(Path(self.temp_dir).glob('*.json'))
        self.assertEqual(len(files), 1)

    def test_compact_before_date(self):
        """Test compacting only chunks before a date."""
        self._create_chunk_file(
            '2025-12-01T21:00:00',
            'a',
            [ChunkOperation(op='add', doc_id='doc1', content='C1')]
        )
        self._create_chunk_file(
            '2025-12-15T22:00:00',
            'b',
            [ChunkOperation(op='add', doc_id='doc2', content='C2')]
        )

        compactor = ChunkCompactor(self.temp_dir)
        result = compactor.compact(before='2025-12-10')

        self.assertEqual(result['status'], 'compacted')
        self.assertEqual(result['compacted'], 1)
        self.assertEqual(result['kept'], 1)

        # Should have original recent file + compacted file
        files = list(Path(self.temp_dir).glob('*.json'))
        self.assertEqual(len(files), 2)

    def test_compact_keep_recent(self):
        """Test keeping N recent chunks."""
        self._create_chunk_file(
            '2025-12-10T21:00:00',
            'a',
            [ChunkOperation(op='add', doc_id='doc1', content='C1')]
        )
        self._create_chunk_file(
            '2025-12-10T22:00:00',
            'b',
            [ChunkOperation(op='add', doc_id='doc2', content='C2')]
        )
        self._create_chunk_file(
            '2025-12-10T23:00:00',
            'c',
            [ChunkOperation(op='add', doc_id='doc3', content='C3')]
        )

        compactor = ChunkCompactor(self.temp_dir)
        result = compactor.compact(keep_recent=1)

        self.assertEqual(result['status'], 'compacted')
        self.assertEqual(result['compacted'], 2)
        self.assertEqual(result['kept'], 1)

        # Should have 1 kept + 1 compacted
        files = list(Path(self.temp_dir).glob('*.json'))
        self.assertEqual(len(files), 2)

    def test_compact_nothing_to_compact(self):
        """Test compact when filters exclude everything."""
        self._create_chunk_file(
            '2025-12-15T21:00:00',
            'a',
            [ChunkOperation(op='add', doc_id='doc1', content='C1')]
        )

        compactor = ChunkCompactor(self.temp_dir)
        result = compactor.compact(before='2025-12-10')

        self.assertEqual(result['status'], 'nothing_to_compact')
        self.assertEqual(result['compacted'], 0)

    def test_compact_handles_modify(self):
        """Test compact correctly merges modify operations."""
        self._create_chunk_file(
            '2025-12-10T21:00:00',
            'a',
            [ChunkOperation(op='add', doc_id='doc1', content='Original')]
        )
        self._create_chunk_file(
            '2025-12-10T22:00:00',
            'b',
            [ChunkOperation(op='modify', doc_id='doc1', content='Modified')]
        )

        compactor = ChunkCompactor(self.temp_dir)
        result = compactor.compact()

        # Load compacted file
        loader = ChunkLoader(self.temp_dir)
        docs = loader.load_all()

        self.assertEqual(docs['doc1'], 'Modified')

    def test_compact_handles_delete(self):
        """Test compact correctly handles delete operations."""
        self._create_chunk_file(
            '2025-12-10T21:00:00',
            'a',
            [
                ChunkOperation(op='add', doc_id='doc1', content='C1'),
                ChunkOperation(op='add', doc_id='doc2', content='C2')
            ]
        )
        self._create_chunk_file(
            '2025-12-10T22:00:00',
            'b',
            [ChunkOperation(op='delete', doc_id='doc1')]
        )

        compactor = ChunkCompactor(self.temp_dir)
        result = compactor.compact()

        # Deleted doc should not appear in compacted chunk
        loader = ChunkLoader(self.temp_dir)
        docs = loader.load_all()

        self.assertNotIn('doc1', docs)
        self.assertEqual(docs['doc2'], 'C2')

    def test_compact_preserves_mtimes(self):
        """Test compact preserves modification times."""
        self._create_chunk_file(
            '2025-12-10T21:00:00',
            'a',
            [ChunkOperation(op='add', doc_id='doc1', content='C1', mtime=123.0)]
        )

        compactor = ChunkCompactor(self.temp_dir)
        compactor.compact()

        loader = ChunkLoader(self.temp_dir)
        loader.load_all()
        mtimes = loader.get_mtimes()

        self.assertEqual(mtimes['doc1'], 123.0)

    def test_compact_preserves_metadata(self):
        """Test compact preserves document metadata."""
        self._create_chunk_file(
            '2025-12-10T21:00:00',
            'a',
            [
                ChunkOperation(
                    op='add',
                    doc_id='doc1',
                    content='C1',
                    metadata={'type': 'python'}
                )
            ]
        )

        compactor = ChunkCompactor(self.temp_dir)
        compactor.compact()

        loader = ChunkLoader(self.temp_dir)
        loader.load_all()
        metadata = loader.get_metadata()

        self.assertEqual(metadata['doc1'], {'type': 'python'})

    def test_compact_creates_sorted_operations(self):
        """Test compact sorts operations by doc_id."""
        self._create_chunk_file(
            '2025-12-10T21:00:00',
            'a',
            [
                ChunkOperation(op='add', doc_id='doc3', content='C3'),
                ChunkOperation(op='add', doc_id='doc1', content='C1'),
                ChunkOperation(op='add', doc_id='doc2', content='C2')
            ]
        )

        compactor = ChunkCompactor(self.temp_dir)
        compactor.compact()

        # Load and verify order
        loader = ChunkLoader(self.temp_dir)
        chunks = loader.get_chunks()

        self.assertEqual(len(chunks), 1)
        doc_ids = [op.doc_id for op in chunks[0].operations]
        self.assertEqual(doc_ids, ['doc1', 'doc2', 'doc3'])


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""

    def test_get_changes_from_manifest_empty(self):
        """Test with empty current and manifest."""
        added, modified, deleted = get_changes_from_manifest({}, {})

        self.assertEqual(len(added), 0)
        self.assertEqual(len(modified), 0)
        self.assertEqual(len(deleted), 0)

    def test_get_changes_from_manifest_all_added(self):
        """Test when all files are new."""
        current = {'file1.txt': 100.0, 'file2.txt': 200.0}
        manifest = {}

        added, modified, deleted = get_changes_from_manifest(current, manifest)

        self.assertEqual(set(added), {'file1.txt', 'file2.txt'})
        self.assertEqual(len(modified), 0)
        self.assertEqual(len(deleted), 0)

    def test_get_changes_from_manifest_all_deleted(self):
        """Test when all files are deleted."""
        current = {}
        manifest = {'file1.txt': 100.0, 'file2.txt': 200.0}

        added, modified, deleted = get_changes_from_manifest(current, manifest)

        self.assertEqual(len(added), 0)
        self.assertEqual(len(modified), 0)
        self.assertEqual(set(deleted), {'file1.txt', 'file2.txt'})

    def test_get_changes_from_manifest_modified(self):
        """Test when files are modified."""
        current = {'file1.txt': 150.0, 'file2.txt': 200.0}
        manifest = {'file1.txt': 100.0, 'file2.txt': 200.0}

        added, modified, deleted = get_changes_from_manifest(current, manifest)

        self.assertEqual(len(added), 0)
        self.assertEqual(modified, ['file1.txt'])
        self.assertEqual(len(deleted), 0)

    def test_get_changes_from_manifest_mixed(self):
        """Test with mix of added, modified, deleted."""
        current = {
            'file1.txt': 150.0,  # Modified
            'file2.txt': 200.0,  # Unchanged
            'file3.txt': 300.0   # Added
        }
        manifest = {
            'file1.txt': 100.0,
            'file2.txt': 200.0,
            'file4.txt': 400.0   # Deleted
        }

        added, modified, deleted = get_changes_from_manifest(current, manifest)

        self.assertEqual(added, ['file3.txt'])
        self.assertEqual(modified, ['file1.txt'])
        self.assertEqual(deleted, ['file4.txt'])

    def test_get_changes_from_manifest_no_changes(self):
        """Test when files haven't changed."""
        current = {'file1.txt': 100.0, 'file2.txt': 200.0}
        manifest = {'file1.txt': 100.0, 'file2.txt': 200.0}

        added, modified, deleted = get_changes_from_manifest(current, manifest)

        self.assertEqual(len(added), 0)
        self.assertEqual(len(modified), 0)
        self.assertEqual(len(deleted), 0)

    def test_get_changes_from_manifest_mtime_equal(self):
        """Test that equal mtime is not considered modified."""
        current = {'file1.txt': 100.0}
        manifest = {'file1.txt': 100.0}

        added, modified, deleted = get_changes_from_manifest(current, manifest)

        self.assertEqual(len(modified), 0)

    def test_get_changes_from_manifest_mtime_older(self):
        """Test that older mtime is not considered modified."""
        current = {'file1.txt': 100.0}
        manifest = {'file1.txt': 150.0}  # Newer in manifest

        added, modified, deleted = get_changes_from_manifest(current, manifest)

        # Current is older than manifest, not modified
        self.assertEqual(len(modified), 0)


if __name__ == '__main__':
    unittest.main()
