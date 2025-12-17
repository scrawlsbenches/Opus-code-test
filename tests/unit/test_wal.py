"""
Unit tests for Write-Ahead Logging (WAL) system.

Tests cover:
- WALEntry creation and serialization
- WALWriter append and rotation
- SnapshotManager create and load
- WALRecovery crash recovery
"""

import json
import tempfile
import unittest
from pathlib import Path

from cortical.wal import (
    WALEntry,
    WALIndex,
    WALWriter,
    SnapshotManager,
    WALRecovery,
    RecoveryResult,
    log_add_document,
    log_remove_document,
    log_compute_phase,
    log_staleness_change,
)


class TestWALEntry(unittest.TestCase):
    """Tests for WALEntry dataclass."""

    def test_create_entry(self):
        """Test creating a basic WAL entry."""
        entry = WALEntry(
            operation='add_document',
            doc_id='doc1',
            payload={'content': 'Hello world'}
        )

        self.assertEqual(entry.operation, 'add_document')
        self.assertEqual(entry.doc_id, 'doc1')
        self.assertIn('content', entry.payload)
        self.assertTrue(entry.checksum)  # Auto-computed

    def test_checksum_computed(self):
        """Test that checksum is auto-computed."""
        entry = WALEntry(operation='test')
        self.assertEqual(len(entry.checksum), 16)

    def test_checksum_deterministic(self):
        """Test that same content produces same checksum."""
        entry1 = WALEntry(
            operation='add_document',
            doc_id='doc1',
            timestamp='2025-12-16T10:00:00',
        )
        entry2 = WALEntry(
            operation='add_document',
            doc_id='doc1',
            timestamp='2025-12-16T10:00:00',
        )
        # Checksums should match for identical content
        self.assertEqual(entry1._compute_checksum(), entry2._compute_checksum())

    def test_checksum_changes_with_content(self):
        """Test that different content produces different checksum."""
        entry1 = WALEntry(operation='add_document', doc_id='doc1')
        entry2 = WALEntry(operation='add_document', doc_id='doc2')
        self.assertNotEqual(entry1._compute_checksum(), entry2._compute_checksum())

    def test_to_json(self):
        """Test JSON serialization."""
        entry = WALEntry(
            operation='add_document',
            doc_id='doc1',
            payload={'content': 'test'}
        )
        json_str = entry.to_json()
        data = json.loads(json_str)

        self.assertEqual(data['operation'], 'add_document')
        self.assertEqual(data['doc_id'], 'doc1')

    def test_from_json(self):
        """Test JSON deserialization."""
        entry = WALEntry(
            operation='remove_document',
            doc_id='doc123',
        )
        json_str = entry.to_json()
        restored = WALEntry.from_json(json_str)

        self.assertEqual(restored.operation, 'remove_document')
        self.assertEqual(restored.doc_id, 'doc123')
        self.assertEqual(restored.checksum, entry.checksum)

    def test_verify_valid(self):
        """Test verification passes for valid entry."""
        entry = WALEntry(operation='test')
        self.assertTrue(entry.verify())

    def test_verify_detects_tampering(self):
        """Test verification fails when content is modified."""
        entry = WALEntry(operation='test', doc_id='original')
        entry.doc_id = 'tampered'  # Modify without updating checksum
        self.assertFalse(entry.verify())


class TestWALWriter(unittest.TestCase):
    """Tests for WALWriter."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.wal_dir = Path(self.temp_dir) / "wal"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_creates_directory_structure(self):
        """Test that writer creates required directories."""
        writer = WALWriter(str(self.wal_dir))

        self.assertTrue(self.wal_dir.exists())
        self.assertTrue((self.wal_dir / "logs").exists())

    def test_creates_index(self):
        """Test that writer creates index file."""
        writer = WALWriter(str(self.wal_dir))

        self.assertTrue((self.wal_dir / "wal_index.json").exists())

    def test_append_creates_wal_file(self):
        """Test that appending creates a WAL file."""
        writer = WALWriter(str(self.wal_dir))
        entry = WALEntry(operation='test')

        writer.append(entry)

        wal_files = list((self.wal_dir / "logs").glob("wal_*.jsonl"))
        self.assertEqual(len(wal_files), 1)

    def test_append_writes_entry(self):
        """Test that entry is written to WAL file."""
        writer = WALWriter(str(self.wal_dir))
        entry = WALEntry(operation='add_document', doc_id='doc1')

        writer.append(entry)

        wal_path = writer.get_current_wal_path()
        with open(wal_path, 'r') as f:
            content = f.read()

        self.assertIn('add_document', content)
        self.assertIn('doc1', content)

    def test_append_increments_count(self):
        """Test that entry count is incremented."""
        writer = WALWriter(str(self.wal_dir))

        writer.append(WALEntry(operation='test1'))
        writer.append(WALEntry(operation='test2'))
        writer.append(WALEntry(operation='test3'))

        self.assertEqual(writer.get_entry_count(), 3)

    def test_get_entries_since(self):
        """Test retrieving entries from WAL."""
        writer = WALWriter(str(self.wal_dir))

        for i in range(5):
            writer.append(WALEntry(operation='test', doc_id=f'doc{i}'))

        entries = list(writer.get_entries_since(writer.index.current_wal_file))

        self.assertEqual(len(entries), 5)
        self.assertEqual(entries[0].doc_id, 'doc0')
        self.assertEqual(entries[4].doc_id, 'doc4')

    def test_get_entries_with_offset(self):
        """Test retrieving entries with offset."""
        writer = WALWriter(str(self.wal_dir))

        for i in range(5):
            writer.append(WALEntry(operation='test', doc_id=f'doc{i}'))

        entries = list(writer.get_entries_since(
            writer.index.current_wal_file,
            offset=3
        ))

        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0].doc_id, 'doc3')


class TestSnapshotManager(unittest.TestCase):
    """Tests for SnapshotManager."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.wal_dir = Path(self.temp_dir) / "wal"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_snapshot(self):
        """Test creating a snapshot."""
        mgr = SnapshotManager(str(self.wal_dir))

        state = {
            'documents': {'doc1': 'Hello', 'doc2': 'World'},
            'layers': {},
        }

        snapshot_id = mgr.create_snapshot(state)

        self.assertTrue(snapshot_id.startswith('snap_'))
        self.assertTrue((self.wal_dir / "snapshots").exists())

    def test_load_snapshot(self):
        """Test loading a snapshot."""
        mgr = SnapshotManager(str(self.wal_dir))

        state = {
            'documents': {'doc1': 'Hello', 'doc2': 'World'},
        }

        snapshot_id = mgr.create_snapshot(state)
        loaded = mgr.load_snapshot(snapshot_id)

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded['state']['documents']['doc1'], 'Hello')

    def test_load_latest_snapshot(self):
        """Test loading latest snapshot when ID not specified."""
        mgr = SnapshotManager(str(self.wal_dir))

        mgr.create_snapshot({'documents': {'doc1': 'First'}})
        import time
        time.sleep(0.1)  # Ensure different timestamps
        mgr.create_snapshot({'documents': {'doc2': 'Second'}})

        loaded = mgr.load_snapshot()  # No ID = latest

        self.assertIn('doc2', loaded['state']['documents'])

    def test_list_snapshots(self):
        """Test listing snapshots."""
        mgr = SnapshotManager(str(self.wal_dir), max_snapshots=5)

        mgr.create_snapshot({'documents': {}})
        mgr.create_snapshot({'documents': {}})

        snapshots = mgr.list_snapshots()

        self.assertEqual(len(snapshots), 2)

    def test_prune_old_snapshots(self):
        """Test that old snapshots are pruned."""
        mgr = SnapshotManager(str(self.wal_dir), max_snapshots=2)

        for i in range(4):
            mgr.create_snapshot({'documents': {f'doc{i}': 'content'}})

        snapshots = mgr.list_snapshots()

        self.assertEqual(len(snapshots), 2)

    def test_compact_wal(self):
        """Test WAL compaction."""
        # Create WAL entries first
        writer = WALWriter(str(self.wal_dir))
        for i in range(5):
            writer.append(WALEntry(operation='test', doc_id=f'doc{i}'))

        mgr = SnapshotManager(str(self.wal_dir))
        state = {'documents': {'final': 'state'}}

        snapshot_id = mgr.compact_wal(state)

        # WAL files should be removed
        wal_files = list((self.wal_dir / "logs").glob("wal_*.jsonl"))
        self.assertEqual(len(wal_files), 0)

        # Snapshot should exist
        loaded = mgr.load_snapshot(snapshot_id)
        self.assertIsNotNone(loaded)


class TestWALRecovery(unittest.TestCase):
    """Tests for WALRecovery."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.wal_dir = Path(self.temp_dir) / "wal"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_needs_recovery_no_wal(self):
        """Test no recovery needed when no WAL exists."""
        recovery = WALRecovery(str(self.wal_dir))

        self.assertFalse(recovery.needs_recovery())

    def test_needs_recovery_with_entries(self):
        """Test recovery needed when WAL has entries."""
        writer = WALWriter(str(self.wal_dir))
        writer.append(WALEntry(operation='test'))

        recovery = WALRecovery(str(self.wal_dir))

        self.assertTrue(recovery.needs_recovery())

    def test_recover_from_snapshot_only(self):
        """Test recovery from snapshot when no WAL entries."""
        mgr = SnapshotManager(str(self.wal_dir))
        state = {'documents': {'doc1': 'Hello'}}
        mgr.create_snapshot(state)

        recovery = WALRecovery(str(self.wal_dir))
        result = recovery.recover()

        self.assertTrue(result.success)
        self.assertEqual(result.documents_recovered, 1)

    def test_recover_replays_add_document(self):
        """Test recovery replays add_document entries."""
        # Create WAL writer first to get the current WAL file
        writer = WALWriter(str(self.wal_dir))
        current_wal = writer.index.current_wal_file

        # Create snapshot that references the WAL file at offset 0
        mgr = SnapshotManager(str(self.wal_dir))
        state = {'documents': {'doc1': 'Original'}}
        mgr.create_snapshot(state, wal_file=current_wal, wal_offset=0)

        # Add WAL entry after snapshot
        writer.append(WALEntry(
            operation='add_document',
            doc_id='doc2',
            payload={'content': 'New document'}
        ))

        # Recover
        recovery = WALRecovery(str(self.wal_dir))
        result = recovery.recover()

        self.assertTrue(result.success)
        self.assertEqual(result.wal_entries_replayed, 1)
        self.assertIn('doc2', result.state['documents'])

    def test_recover_replays_remove_document(self):
        """Test recovery replays remove_document entries."""
        # Create WAL writer first to get the current WAL file
        writer = WALWriter(str(self.wal_dir))
        current_wal = writer.index.current_wal_file

        # Create snapshot with documents
        mgr = SnapshotManager(str(self.wal_dir))
        state = {'documents': {'doc1': 'To be removed', 'doc2': 'Keep'}}
        mgr.create_snapshot(state, wal_file=current_wal, wal_offset=0)

        # Add removal entry after snapshot
        writer.append(WALEntry(
            operation='remove_document',
            doc_id='doc1',
        ))

        # Recover
        recovery = WALRecovery(str(self.wal_dir))
        result = recovery.recover()

        self.assertTrue(result.success)
        self.assertNotIn('doc1', result.state['documents'])
        self.assertIn('doc2', result.state['documents'])

    def test_get_recovery_info(self):
        """Test getting recovery info."""
        writer = WALWriter(str(self.wal_dir))
        writer.append(WALEntry(operation='test'))

        mgr = SnapshotManager(str(self.wal_dir))
        mgr.create_snapshot({'documents': {}})

        recovery = WALRecovery(str(self.wal_dir))
        info = recovery.get_recovery_info()

        self.assertTrue(info['has_wal'])
        self.assertEqual(info['wal_entries'], 1)
        self.assertTrue(info['needs_recovery'])


class TestConvenienceFunctions(unittest.TestCase):
    """Tests for convenience logging functions."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.wal_dir = Path(self.temp_dir) / "wal"
        self.writer = WALWriter(str(self.wal_dir))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_log_add_document(self):
        """Test logging document addition."""
        log_add_document(self.writer, 'doc1', 'Hello world')

        entries = list(self.writer.get_entries_since(
            self.writer.index.current_wal_file
        ))

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].operation, 'add_document')
        self.assertEqual(entries[0].doc_id, 'doc1')

    def test_log_remove_document(self):
        """Test logging document removal."""
        log_remove_document(self.writer, 'doc1')

        entries = list(self.writer.get_entries_since(
            self.writer.index.current_wal_file
        ))

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].operation, 'remove_document')

    def test_log_compute_phase(self):
        """Test logging compute phase."""
        log_compute_phase(self.writer, 'pagerank', duration_ms=1500)

        entries = list(self.writer.get_entries_since(
            self.writer.index.current_wal_file
        ))

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].operation, 'compute_phase')
        self.assertEqual(entries[0].phase, 'pagerank')

    def test_log_staleness_change(self):
        """Test logging staleness changes."""
        log_staleness_change(
            self.writer,
            mark_fresh=True,
            computations=['tfidf', 'pagerank']
        )

        entries = list(self.writer.get_entries_since(
            self.writer.index.current_wal_file
        ))

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].operation, 'mark_fresh')
        self.assertIn('tfidf', entries[0].affected_computations)


if __name__ == '__main__':
    unittest.main()
