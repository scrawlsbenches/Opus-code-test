"""Tests for chunk-based indexing."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cortical.chunk_index import (
    Chunk,
    ChunkOperation,
    ChunkWriter,
    ChunkLoader,
    ChunkCompactor,
    CHUNK_VERSION,
    get_changes_from_manifest,
)


class TestChunkOperation(unittest.TestCase):
    """Test ChunkOperation dataclass."""

    def test_add_operation(self):
        """Test creating an add operation."""
        op = ChunkOperation(op='add', doc_id='doc1', content='hello', mtime=123.0)
        self.assertEqual(op.op, 'add')
        self.assertEqual(op.doc_id, 'doc1')
        self.assertEqual(op.content, 'hello')
        self.assertEqual(op.mtime, 123.0)

    def test_delete_operation(self):
        """Test creating a delete operation."""
        op = ChunkOperation(op='delete', doc_id='doc1')
        self.assertEqual(op.op, 'delete')
        self.assertEqual(op.doc_id, 'doc1')
        self.assertIsNone(op.content)
        self.assertIsNone(op.mtime)

    def test_to_dict_add(self):
        """Test converting add operation to dict."""
        op = ChunkOperation(op='add', doc_id='doc1', content='hello', mtime=123.0)
        d = op.to_dict()
        self.assertEqual(d['op'], 'add')
        self.assertEqual(d['doc_id'], 'doc1')
        self.assertEqual(d['content'], 'hello')
        self.assertEqual(d['mtime'], 123.0)

    def test_to_dict_delete(self):
        """Test converting delete operation to dict (no content/mtime)."""
        op = ChunkOperation(op='delete', doc_id='doc1')
        d = op.to_dict()
        self.assertEqual(d['op'], 'delete')
        self.assertEqual(d['doc_id'], 'doc1')
        self.assertNotIn('content', d)
        self.assertNotIn('mtime', d)

    def test_from_dict(self):
        """Test creating operation from dict."""
        d = {'op': 'add', 'doc_id': 'doc1', 'content': 'hello', 'mtime': 123.0}
        op = ChunkOperation.from_dict(d)
        self.assertEqual(op.op, 'add')
        self.assertEqual(op.doc_id, 'doc1')
        self.assertEqual(op.content, 'hello')
        self.assertEqual(op.mtime, 123.0)


class TestChunk(unittest.TestCase):
    """Test Chunk dataclass."""

    def test_chunk_creation(self):
        """Test creating a chunk."""
        chunk = Chunk(
            version=1,
            timestamp='2025-12-10T12:00:00',
            session_id='abc123',
            branch='main',
            operations=[]
        )
        self.assertEqual(chunk.version, 1)
        self.assertEqual(chunk.timestamp, '2025-12-10T12:00:00')
        self.assertEqual(chunk.session_id, 'abc123')
        self.assertEqual(chunk.branch, 'main')

    def test_chunk_with_operations(self):
        """Test chunk with operations."""
        ops = [
            ChunkOperation(op='add', doc_id='doc1', content='hello'),
            ChunkOperation(op='delete', doc_id='doc2')
        ]
        chunk = Chunk(
            version=1,
            timestamp='2025-12-10T12:00:00',
            session_id='abc123',
            branch='main',
            operations=ops
        )
        self.assertEqual(len(chunk.operations), 2)

    def test_to_dict(self):
        """Test converting chunk to dict."""
        ops = [ChunkOperation(op='add', doc_id='doc1', content='hello')]
        chunk = Chunk(
            version=1,
            timestamp='2025-12-10T12:00:00',
            session_id='abc123',
            branch='main',
            operations=ops
        )
        d = chunk.to_dict()
        self.assertEqual(d['version'], 1)
        self.assertEqual(d['timestamp'], '2025-12-10T12:00:00')
        self.assertEqual(len(d['operations']), 1)

    def test_from_dict(self):
        """Test creating chunk from dict."""
        d = {
            'version': 1,
            'timestamp': '2025-12-10T12:00:00',
            'session_id': 'abc123',
            'branch': 'main',
            'operations': [
                {'op': 'add', 'doc_id': 'doc1', 'content': 'hello'}
            ]
        }
        chunk = Chunk.from_dict(d)
        self.assertEqual(chunk.version, 1)
        self.assertEqual(len(chunk.operations), 1)
        self.assertEqual(chunk.operations[0].doc_id, 'doc1')

    def test_get_filename(self):
        """Test filename generation."""
        chunk = Chunk(
            version=1,
            timestamp='2025-12-10T12:00:00',
            session_id='abc12345xyz',
            branch='main',
            operations=[]
        )
        filename = chunk.get_filename()
        self.assertTrue(filename.endswith('.json'))
        self.assertIn('2025-12-10', filename)
        self.assertIn('abc12345', filename)


class TestChunkWriter(unittest.TestCase):
    """Test ChunkWriter class."""

    def test_writer_creation(self):
        """Test creating a chunk writer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ChunkWriter(tmpdir)
            self.assertEqual(len(writer.session_id), 16)
            self.assertIsNotNone(writer.timestamp)

    def test_add_document(self):
        """Test adding a document."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ChunkWriter(tmpdir)
            writer.add_document('doc1', 'content here', mtime=123.0)
            self.assertEqual(len(writer.operations), 1)
            self.assertEqual(writer.operations[0].op, 'add')

    def test_modify_document(self):
        """Test modifying a document."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ChunkWriter(tmpdir)
            writer.modify_document('doc1', 'new content', mtime=456.0)
            self.assertEqual(len(writer.operations), 1)
            self.assertEqual(writer.operations[0].op, 'modify')

    def test_delete_document(self):
        """Test deleting a document."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ChunkWriter(tmpdir)
            writer.delete_document('doc1')
            self.assertEqual(len(writer.operations), 1)
            self.assertEqual(writer.operations[0].op, 'delete')

    def test_has_operations(self):
        """Test checking for operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ChunkWriter(tmpdir)
            self.assertFalse(writer.has_operations())
            writer.add_document('doc1', 'content')
            self.assertTrue(writer.has_operations())

    def test_save_empty(self):
        """Test saving with no operations returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ChunkWriter(tmpdir)
            result = writer.save()
            self.assertIsNone(result)

    def test_save_creates_file(self):
        """Test saving creates a JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ChunkWriter(tmpdir)
            writer.add_document('doc1', 'content here')
            filepath = writer.save()

            self.assertIsNotNone(filepath)
            self.assertTrue(filepath.exists())
            self.assertTrue(filepath.name.endswith('.json'))

            # Verify contents
            with open(filepath) as f:
                data = json.load(f)
            self.assertEqual(data['version'], CHUNK_VERSION)
            self.assertEqual(len(data['operations']), 1)

    def test_save_creates_directory(self):
        """Test saving creates the chunks directory if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chunks_dir = os.path.join(tmpdir, 'new_chunks')
            writer = ChunkWriter(chunks_dir)
            writer.add_document('doc1', 'content')
            filepath = writer.save()

            self.assertTrue(os.path.exists(chunks_dir))
            self.assertTrue(filepath.exists())


class TestChunkLoader(unittest.TestCase):
    """Test ChunkLoader class."""

    def test_loader_empty_directory(self):
        """Test loading from empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = ChunkLoader(tmpdir)
            docs = loader.load_all()
            self.assertEqual(len(docs), 0)

    def test_loader_nonexistent_directory(self):
        """Test loading from nonexistent directory."""
        loader = ChunkLoader('/nonexistent/path')
        docs = loader.load_all()
        self.assertEqual(len(docs), 0)

    def test_load_single_chunk(self):
        """Test loading a single chunk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a chunk
            writer = ChunkWriter(tmpdir)
            writer.add_document('doc1', 'content1')
            writer.add_document('doc2', 'content2')
            writer.save()

            # Load it
            loader = ChunkLoader(tmpdir)
            docs = loader.load_all()

            self.assertEqual(len(docs), 2)
            self.assertEqual(docs['doc1'], 'content1')
            self.assertEqual(docs['doc2'], 'content2')

    def test_load_multiple_chunks(self):
        """Test loading multiple chunks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create first chunk
            writer1 = ChunkWriter(tmpdir)
            writer1.timestamp = '2025-12-10T10:00:00'
            writer1.add_document('doc1', 'content1')
            writer1.save()

            # Create second chunk
            writer2 = ChunkWriter(tmpdir)
            writer2.timestamp = '2025-12-10T11:00:00'
            writer2.add_document('doc2', 'content2')
            writer2.save()

            # Load both
            loader = ChunkLoader(tmpdir)
            docs = loader.load_all()

            self.assertEqual(len(docs), 2)
            self.assertIn('doc1', docs)
            self.assertIn('doc2', docs)

    def test_later_chunk_wins(self):
        """Test that later timestamps override earlier."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create first chunk
            writer1 = ChunkWriter(tmpdir)
            writer1.timestamp = '2025-12-10T10:00:00'
            writer1.session_id = 'aaaa0000'
            writer1.add_document('doc1', 'old content')
            writer1.save()

            # Create second chunk with modification
            writer2 = ChunkWriter(tmpdir)
            writer2.timestamp = '2025-12-10T11:00:00'
            writer2.session_id = 'bbbb1111'
            writer2.modify_document('doc1', 'new content')
            writer2.save()

            # Load - should have new content
            loader = ChunkLoader(tmpdir)
            docs = loader.load_all()

            self.assertEqual(docs['doc1'], 'new content')

    def test_delete_removes_document(self):
        """Test that delete operations remove documents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create first chunk
            writer1 = ChunkWriter(tmpdir)
            writer1.timestamp = '2025-12-10T10:00:00'
            writer1.session_id = 'aaaa0000'
            writer1.add_document('doc1', 'content1')
            writer1.add_document('doc2', 'content2')
            writer1.save()

            # Create second chunk with deletion
            writer2 = ChunkWriter(tmpdir)
            writer2.timestamp = '2025-12-10T11:00:00'
            writer2.session_id = 'bbbb1111'
            writer2.delete_document('doc1')
            writer2.save()

            # Load - doc1 should be gone
            loader = ChunkLoader(tmpdir)
            docs = loader.load_all()

            self.assertEqual(len(docs), 1)
            self.assertNotIn('doc1', docs)
            self.assertIn('doc2', docs)

    def test_get_mtimes(self):
        """Test getting modification times."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ChunkWriter(tmpdir)
            writer.add_document('doc1', 'content', mtime=123.0)
            writer.add_document('doc2', 'content', mtime=456.0)
            writer.save()

            loader = ChunkLoader(tmpdir)
            mtimes = loader.get_mtimes()

            self.assertEqual(mtimes['doc1'], 123.0)
            self.assertEqual(mtimes['doc2'], 456.0)

    def test_compute_hash(self):
        """Test computing content hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ChunkWriter(tmpdir)
            writer.add_document('doc1', 'content1')
            writer.add_document('doc2', 'content2')
            writer.save()

            loader = ChunkLoader(tmpdir)
            hash1 = loader.compute_hash()

            # Same content should have same hash
            loader2 = ChunkLoader(tmpdir)
            hash2 = loader2.compute_hash()

            self.assertEqual(hash1, hash2)
            self.assertEqual(len(hash1), 16)  # Truncated hash

    def test_get_stats(self):
        """Test getting chunk statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ChunkWriter(tmpdir)
            writer.add_document('doc1', 'content')
            writer.modify_document('doc2', 'content')
            writer.delete_document('doc3')
            writer.save()

            loader = ChunkLoader(tmpdir)
            stats = loader.get_stats()

            self.assertEqual(stats['chunk_count'], 1)
            self.assertEqual(stats['document_count'], 2)  # doc1 and doc2
            self.assertEqual(stats['total_operations'], 3)
            self.assertEqual(stats['add_operations'], 1)
            self.assertEqual(stats['modify_operations'], 1)
            self.assertEqual(stats['delete_operations'], 1)

    def test_cache_validation(self):
        """Test cache hash validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create chunk
            writer = ChunkWriter(tmpdir)
            writer.add_document('doc1', 'content')
            writer.save()

            # Create fake cache file
            cache_path = os.path.join(tmpdir, 'cache.pkl')
            with open(cache_path, 'w') as f:
                f.write('fake cache')

            # Load and save hash
            loader = ChunkLoader(tmpdir)
            loader.load_all()
            loader.save_cache_hash(cache_path)

            # Validate - should be valid
            self.assertTrue(loader.is_cache_valid(cache_path))

            # Add another chunk
            writer2 = ChunkWriter(tmpdir)
            writer2.add_document('doc2', 'content2')
            writer2.save()

            # Reload - hash should be different
            loader2 = ChunkLoader(tmpdir)
            self.assertFalse(loader2.is_cache_valid(cache_path))


class TestChunkCompactor(unittest.TestCase):
    """Test ChunkCompactor class."""

    def test_compact_empty(self):
        """Test compacting empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            compactor = ChunkCompactor(tmpdir)
            result = compactor.compact()
            self.assertEqual(result['status'], 'no_chunks')

    def test_compact_all_chunks(self):
        """Test compacting all chunks into one."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple chunks
            for i in range(3):
                writer = ChunkWriter(tmpdir)
                writer.timestamp = f'2025-12-0{i+1}T10:00:00'
                writer.session_id = f'session{i}'
                writer.add_document(f'doc{i}', f'content{i}')
                writer.save()

            # Verify 3 chunks exist
            self.assertEqual(len(list(Path(tmpdir).glob('*.json'))), 3)

            # Compact
            compactor = ChunkCompactor(tmpdir)
            result = compactor.compact()

            self.assertEqual(result['status'], 'compacted')
            self.assertEqual(result['compacted'], 3)
            self.assertEqual(result['documents'], 3)

            # Should have 1 chunk now
            self.assertEqual(len(list(Path(tmpdir).glob('*.json'))), 1)

            # Documents should still be loadable
            loader = ChunkLoader(tmpdir)
            docs = loader.load_all()
            self.assertEqual(len(docs), 3)

    def test_compact_before_date(self):
        """Test compacting only chunks before a date."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create chunks on different dates
            writer1 = ChunkWriter(tmpdir)
            writer1.timestamp = '2025-12-01T10:00:00'
            writer1.session_id = 'session1'
            writer1.add_document('doc1', 'content1')
            writer1.save()

            writer2 = ChunkWriter(tmpdir)
            writer2.timestamp = '2025-12-05T10:00:00'
            writer2.session_id = 'session2'
            writer2.add_document('doc2', 'content2')
            writer2.save()

            writer3 = ChunkWriter(tmpdir)
            writer3.timestamp = '2025-12-10T10:00:00'
            writer3.session_id = 'session3'
            writer3.add_document('doc3', 'content3')
            writer3.save()

            # Compact only before 2025-12-08
            compactor = ChunkCompactor(tmpdir)
            result = compactor.compact(before='2025-12-08')

            self.assertEqual(result['compacted'], 2)  # doc1 and doc2
            self.assertEqual(result['kept'], 1)  # doc3

    def test_compact_dry_run(self):
        """Test dry run doesn't modify files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create chunks
            for i in range(2):
                writer = ChunkWriter(tmpdir)
                writer.timestamp = f'2025-12-0{i+1}T10:00:00'
                writer.session_id = f'session{i}'
                writer.add_document(f'doc{i}', f'content{i}')
                writer.save()

            # Dry run
            compactor = ChunkCompactor(tmpdir)
            result = compactor.compact(dry_run=True)

            self.assertEqual(result['status'], 'dry_run')
            self.assertEqual(result['would_compact'], 2)

            # Should still have 2 chunks
            self.assertEqual(len(list(Path(tmpdir).glob('*.json'))), 2)


class TestGetChangesFromManifest(unittest.TestCase):
    """Test change detection from manifest."""

    def test_no_changes(self):
        """Test when nothing changed."""
        current = {'doc1': 100.0, 'doc2': 200.0}
        manifest = {'doc1': 100.0, 'doc2': 200.0}

        added, modified, deleted = get_changes_from_manifest(current, manifest)

        self.assertEqual(len(added), 0)
        self.assertEqual(len(modified), 0)
        self.assertEqual(len(deleted), 0)

    def test_added_files(self):
        """Test detecting added files."""
        current = {'doc1': 100.0, 'doc2': 200.0, 'doc3': 300.0}
        manifest = {'doc1': 100.0, 'doc2': 200.0}

        added, modified, deleted = get_changes_from_manifest(current, manifest)

        self.assertEqual(added, ['doc3'])
        self.assertEqual(len(modified), 0)
        self.assertEqual(len(deleted), 0)

    def test_deleted_files(self):
        """Test detecting deleted files."""
        current = {'doc1': 100.0}
        manifest = {'doc1': 100.0, 'doc2': 200.0}

        added, modified, deleted = get_changes_from_manifest(current, manifest)

        self.assertEqual(len(added), 0)
        self.assertEqual(len(modified), 0)
        self.assertEqual(deleted, ['doc2'])

    def test_modified_files(self):
        """Test detecting modified files."""
        current = {'doc1': 150.0, 'doc2': 200.0}  # doc1 has newer mtime
        manifest = {'doc1': 100.0, 'doc2': 200.0}

        added, modified, deleted = get_changes_from_manifest(current, manifest)

        self.assertEqual(len(added), 0)
        self.assertEqual(modified, ['doc1'])
        self.assertEqual(len(deleted), 0)

    def test_all_change_types(self):
        """Test mix of all change types."""
        current = {'doc1': 150.0, 'doc3': 300.0}  # doc1 modified, doc3 added
        manifest = {'doc1': 100.0, 'doc2': 200.0}  # doc2 deleted

        added, modified, deleted = get_changes_from_manifest(current, manifest)

        self.assertEqual(added, ['doc3'])
        self.assertEqual(modified, ['doc1'])
        self.assertEqual(deleted, ['doc2'])


class TestChunkMetadata(unittest.TestCase):
    """Test metadata support in chunk indexing."""

    def test_operation_with_metadata(self):
        """Test creating an operation with metadata."""
        metadata = {'doc_type': 'code', 'language': 'python'}
        op = ChunkOperation(
            op='add',
            doc_id='doc1',
            content='hello',
            mtime=123.0,
            metadata=metadata
        )
        self.assertEqual(op.metadata, metadata)

    def test_operation_to_dict_with_metadata(self):
        """Test converting operation with metadata to dict."""
        metadata = {'doc_type': 'docs', 'headings': ['Section 1', 'Section 2']}
        op = ChunkOperation(
            op='add',
            doc_id='doc1',
            content='# Doc\n\n## Section 1\n\n## Section 2',
            mtime=123.0,
            metadata=metadata
        )
        d = op.to_dict()
        self.assertIn('metadata', d)
        self.assertEqual(d['metadata']['doc_type'], 'docs')
        self.assertEqual(d['metadata']['headings'], ['Section 1', 'Section 2'])

    def test_operation_to_dict_without_metadata(self):
        """Test that metadata is omitted from dict when None."""
        op = ChunkOperation(op='add', doc_id='doc1', content='hello')
        d = op.to_dict()
        self.assertNotIn('metadata', d)

    def test_operation_from_dict_with_metadata(self):
        """Test creating operation from dict with metadata."""
        d = {
            'op': 'add',
            'doc_id': 'doc1',
            'content': 'hello',
            'mtime': 123.0,
            'metadata': {'doc_type': 'test', 'function_count': 5}
        }
        op = ChunkOperation.from_dict(d)
        self.assertIsNotNone(op.metadata)
        self.assertEqual(op.metadata['doc_type'], 'test')
        self.assertEqual(op.metadata['function_count'], 5)

    def test_operation_from_dict_without_metadata(self):
        """Test creating operation from dict without metadata (backward compat)."""
        d = {'op': 'add', 'doc_id': 'doc1', 'content': 'hello'}
        op = ChunkOperation.from_dict(d)
        self.assertIsNone(op.metadata)

    def test_writer_add_with_metadata(self):
        """Test writer add_document with metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ChunkWriter(tmpdir)
            metadata = {'doc_type': 'code', 'language': 'python'}
            writer.add_document('doc1', 'content', mtime=123.0, metadata=metadata)

            self.assertEqual(len(writer.operations), 1)
            self.assertEqual(writer.operations[0].metadata, metadata)

    def test_writer_modify_with_metadata(self):
        """Test writer modify_document with metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ChunkWriter(tmpdir)
            metadata = {'doc_type': 'docs', 'headings': ['Intro']}
            writer.modify_document('doc1', 'new content', mtime=456.0, metadata=metadata)

            self.assertEqual(len(writer.operations), 1)
            self.assertEqual(writer.operations[0].metadata, metadata)

    def test_loader_get_metadata(self):
        """Test loader returns metadata for documents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ChunkWriter(tmpdir)
            writer.add_document('doc1', 'content1', metadata={'doc_type': 'code'})
            writer.add_document('doc2', 'content2', metadata={'doc_type': 'docs'})
            writer.save()

            loader = ChunkLoader(tmpdir)
            loader.load_all()
            metadata = loader.get_metadata()

            self.assertEqual(len(metadata), 2)
            self.assertEqual(metadata['doc1']['doc_type'], 'code')
            self.assertEqual(metadata['doc2']['doc_type'], 'docs')

    def test_loader_metadata_updated_on_modify(self):
        """Test metadata is updated when document is modified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First chunk: add with initial metadata
            writer1 = ChunkWriter(tmpdir)
            writer1.timestamp = '2025-12-10T10:00:00'
            writer1.session_id = 'aaaa0000'
            writer1.add_document('doc1', 'old', metadata={'version': 1})
            writer1.save()

            # Second chunk: modify with new metadata
            writer2 = ChunkWriter(tmpdir)
            writer2.timestamp = '2025-12-10T11:00:00'
            writer2.session_id = 'bbbb1111'
            writer2.modify_document('doc1', 'new', metadata={'version': 2})
            writer2.save()

            loader = ChunkLoader(tmpdir)
            loader.load_all()
            metadata = loader.get_metadata()

            self.assertEqual(metadata['doc1']['version'], 2)

    def test_loader_metadata_removed_on_delete(self):
        """Test metadata is removed when document is deleted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First chunk: add document
            writer1 = ChunkWriter(tmpdir)
            writer1.timestamp = '2025-12-10T10:00:00'
            writer1.session_id = 'aaaa0000'
            writer1.add_document('doc1', 'content', metadata={'doc_type': 'code'})
            writer1.save()

            # Second chunk: delete document
            writer2 = ChunkWriter(tmpdir)
            writer2.timestamp = '2025-12-10T11:00:00'
            writer2.session_id = 'bbbb1111'
            writer2.delete_document('doc1')
            writer2.save()

            loader = ChunkLoader(tmpdir)
            loader.load_all()
            metadata = loader.get_metadata()

            self.assertNotIn('doc1', metadata)

    def test_compactor_preserves_metadata(self):
        """Test compactor preserves metadata during compaction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create chunk with metadata
            writer = ChunkWriter(tmpdir)
            writer.timestamp = '2025-01-01T10:00:00'
            writer.session_id = 'aaaa0000'
            writer.add_document(
                'doc1',
                'content1',
                mtime=100.0,
                metadata={'doc_type': 'code', 'language': 'python'}
            )
            writer.add_document(
                'doc2',
                'content2',
                mtime=200.0,
                metadata={'doc_type': 'docs', 'headings': ['H1', 'H2']}
            )
            writer.save()

            # Compact
            compactor = ChunkCompactor(tmpdir)
            result = compactor.compact()

            self.assertEqual(result['status'], 'compacted')

            # Load compacted chunk and check metadata
            loader = ChunkLoader(tmpdir)
            loader.load_all()
            metadata = loader.get_metadata()

            self.assertEqual(len(metadata), 2)
            self.assertEqual(metadata['doc1']['doc_type'], 'code')
            self.assertEqual(metadata['doc1']['language'], 'python')
            self.assertEqual(metadata['doc2']['doc_type'], 'docs')
            self.assertEqual(metadata['doc2']['headings'], ['H1', 'H2'])

    def test_chunk_serialization_roundtrip(self):
        """Test metadata survives JSON serialization roundtrip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = {
                'doc_type': 'docs',
                'headings': ['Introduction', 'Methods', 'Results'],
                'line_count': 150,
                'mtime': 1234567890.5
            }
            writer = ChunkWriter(tmpdir)
            writer.add_document('doc1', 'content', metadata=metadata)
            filepath = writer.save()

            # Load from file and verify
            with open(filepath) as f:
                data = json.load(f)

            op_data = data['operations'][0]
            self.assertEqual(op_data['metadata'], metadata)


if __name__ == '__main__':
    unittest.main(verbosity=2)
