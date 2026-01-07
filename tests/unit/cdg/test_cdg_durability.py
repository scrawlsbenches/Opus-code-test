"""
Tests for CDG durability modes.

Tests that durability modes correctly control fsync behavior across
CDG WAL and storage components.

These tests were extracted from tests/unit/got/test_config.py during
the CDG layer separation refactoring. They test CDG infrastructure,
not GoT domain logic.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cortical.cdg import CDGWALManager, CDGStore
from cortical.cdg.config import CDGConfig, DurabilityMode


class TestParanoidMode(unittest.TestCase):
    """Test PARANOID durability mode."""

    def setUp(self):
        """Set up temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.wal_dir = Path(self.temp_dir) / "wal"

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('os.fsync')
    def test_paranoid_mode_fsyncs_on_log(self, mock_fsync):
        """Test that PARANOID mode calls fsync on every WAL log."""
        wal = CDGWALManager(self.wal_dir, CDGConfig(durability=DurabilityMode.PARANOID))

        # Log a transaction begin
        wal.log_tx_begin("tx1", snapshot_version=0)

        # Should have called fsync (at least once for the log entry)
        self.assertGreater(mock_fsync.call_count, 0)

    @patch('os.fsync')
    def test_paranoid_mode_fsyncs_on_commit(self, mock_fsync):
        """Test that PARANOID mode calls fsync when committing transaction."""
        wal = CDGWALManager(self.wal_dir, CDGConfig(durability=DurabilityMode.PARANOID))

        # Clear any fsync calls from __init__
        mock_fsync.reset_mock()

        # Full transaction: begin + commit (which persists sequence)
        wal.log_tx_begin("tx1", snapshot_version=0)
        wal.log_tx_commit("tx1", version=1)

        # Should have called fsync for WAL entries (PARANOID mode)
        self.assertGreater(mock_fsync.call_count, 0)


class TestBalancedMode(unittest.TestCase):
    """Test BALANCED durability mode."""

    def setUp(self):
        """Set up temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.wal_dir = Path(self.temp_dir) / "wal"

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('os.fsync')
    def test_balanced_mode_skips_per_op_fsync(self, mock_fsync):
        """Test that BALANCED mode does NOT fsync on individual operations."""
        wal = CDGWALManager(self.wal_dir, CDGConfig(durability=DurabilityMode.BALANCED))

        # Clear any fsync calls from __init__
        mock_fsync.reset_mock()

        # Log a transaction begin
        wal.log_tx_begin("tx1", snapshot_version=0)

        # Should NOT have called fsync
        self.assertEqual(mock_fsync.call_count, 0)

    @patch('os.fsync')
    def test_balanced_mode_fsync_now_works(self, mock_fsync):
        """Test that BALANCED mode can fsync explicitly via fsync_now()."""
        wal = CDGWALManager(self.wal_dir, CDGConfig(durability=DurabilityMode.BALANCED))

        # Log some operations
        wal.log_tx_begin("tx1", snapshot_version=0)

        # Clear mock
        mock_fsync.reset_mock()

        # Explicitly sync
        wal.fsync_now()

        # Should have called fsync now
        self.assertGreater(mock_fsync.call_count, 0)


class TestRelaxedMode(unittest.TestCase):
    """Test RELAXED durability mode."""

    def setUp(self):
        """Set up temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.wal_dir = Path(self.temp_dir) / "wal"
        self.store_dir = Path(self.temp_dir) / "entities"

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('os.fsync')
    def test_relaxed_mode_never_fsyncs_wal(self, mock_fsync):
        """Test that RELAXED mode never calls fsync on WAL."""
        wal = CDGWALManager(self.wal_dir, CDGConfig(durability=DurabilityMode.RELAXED))

        # Clear any fsync calls from __init__
        mock_fsync.reset_mock()

        # Log multiple operations
        wal.log_tx_begin("tx1", snapshot_version=0)
        wal.log_tx_commit("tx1", version=1)

        # Should NEVER call fsync
        self.assertEqual(mock_fsync.call_count, 0)

    @patch('os.fsync')
    def test_relaxed_mode_never_fsyncs_store(self, mock_fsync):
        """Test that RELAXED mode never calls fsync on entity store."""
        from cortical.got import Task

        store = CDGStore(self.store_dir, CDGConfig(durability=DurabilityMode.RELAXED))

        # Clear any fsync calls from __init__
        mock_fsync.reset_mock()

        # Write a task
        task = Task(
            id="T-20251221-000000-test",
            title="Test task",
            priority="medium",
            status="pending",
        )
        store.write(task)

        # Should NEVER call fsync
        self.assertEqual(mock_fsync.call_count, 0)
