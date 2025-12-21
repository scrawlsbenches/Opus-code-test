"""
Tests for GoT configuration module (durability modes).

Tests that durability modes correctly control fsync behavior across
the GoT transactional system.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from cortical.got import (
    DurabilityMode,
    GoTConfig,
    GoTManager,
    WALManager,
    VersionedStore,
)


class TestDurabilityMode(unittest.TestCase):
    """Test DurabilityMode enum."""

    def test_durability_mode_enum_values(self):
        """Test that DurabilityMode has correct values."""
        self.assertEqual(DurabilityMode.PARANOID.value, "paranoid")
        self.assertEqual(DurabilityMode.BALANCED.value, "balanced")
        self.assertEqual(DurabilityMode.RELAXED.value, "relaxed")

    def test_durability_mode_has_three_values(self):
        """Test that DurabilityMode has exactly 3 modes."""
        modes = list(DurabilityMode)
        self.assertEqual(len(modes), 3)
        self.assertIn(DurabilityMode.PARANOID, modes)
        self.assertIn(DurabilityMode.BALANCED, modes)
        self.assertIn(DurabilityMode.RELAXED, modes)


class TestGoTConfig(unittest.TestCase):
    """Test GoTConfig dataclass."""

    def test_default_config_is_balanced(self):
        """Test that default durability mode is BALANCED."""
        config = GoTConfig()
        self.assertEqual(config.durability, DurabilityMode.BALANCED)

    def test_config_accepts_durability_param(self):
        """Test that GoTConfig accepts durability parameter."""
        config = GoTConfig(durability=DurabilityMode.PARANOID)
        self.assertEqual(config.durability, DurabilityMode.PARANOID)

        config = GoTConfig(durability=DurabilityMode.RELAXED)
        self.assertEqual(config.durability, DurabilityMode.RELAXED)


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
        wal = WALManager(self.wal_dir, durability=DurabilityMode.PARANOID)

        # Log a transaction begin
        wal.log_tx_begin("tx1", snapshot_version=0)

        # Should have called fsync (at least once for the log entry)
        self.assertGreater(mock_fsync.call_count, 0)

    @patch('os.fsync')
    def test_paranoid_mode_fsyncs_on_sequence(self, mock_fsync):
        """Test that PARANOID mode calls fsync when saving sequence."""
        wal = WALManager(self.wal_dir, durability=DurabilityMode.PARANOID)

        # Clear any fsync calls from __init__
        mock_fsync.reset_mock()

        # Next sequence increments and saves
        seq = wal._next_seq()

        # Should have called fsync for sequence file
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
        wal = WALManager(self.wal_dir, durability=DurabilityMode.BALANCED)

        # Clear any fsync calls from __init__
        mock_fsync.reset_mock()

        # Log a transaction begin
        wal.log_tx_begin("tx1", snapshot_version=0)

        # Should NOT have called fsync
        self.assertEqual(mock_fsync.call_count, 0)

    @patch('os.fsync')
    def test_balanced_mode_fsync_now_works(self, mock_fsync):
        """Test that BALANCED mode can fsync explicitly via fsync_now()."""
        wal = WALManager(self.wal_dir, durability=DurabilityMode.BALANCED)

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
        wal = WALManager(self.wal_dir, durability=DurabilityMode.RELAXED)

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

        store = VersionedStore(self.store_dir, durability=DurabilityMode.RELAXED)

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


class TestGoTManagerDurability(unittest.TestCase):
    """Test that GoTManager accepts and uses durability parameter."""

    def setUp(self):
        """Set up temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.got_dir = Path(self.temp_dir) / ".got"

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_manager_accepts_durability_param(self):
        """Test that GoTManager accepts durability parameter."""
        # Test PARANOID
        manager = GoTManager(self.got_dir, durability=DurabilityMode.PARANOID)
        self.assertEqual(manager.durability, DurabilityMode.PARANOID)
        self.assertEqual(manager.tx_manager.durability, DurabilityMode.PARANOID)

        # Clean up
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

        # Test BALANCED (default)
        manager = GoTManager(self.got_dir)
        self.assertEqual(manager.durability, DurabilityMode.BALANCED)
        self.assertEqual(manager.tx_manager.durability, DurabilityMode.BALANCED)

        # Clean up
        shutil.rmtree(self.temp_dir, ignore_errors=True)

        # Test RELAXED
        manager = GoTManager(self.got_dir, durability=DurabilityMode.RELAXED)
        self.assertEqual(manager.durability, DurabilityMode.RELAXED)
        self.assertEqual(manager.tx_manager.durability, DurabilityMode.RELAXED)

    def test_manager_default_is_balanced(self):
        """Test that GoTManager defaults to BALANCED mode."""
        manager = GoTManager(self.got_dir)
        self.assertEqual(manager.durability, DurabilityMode.BALANCED)


if __name__ == '__main__':
    unittest.main()
