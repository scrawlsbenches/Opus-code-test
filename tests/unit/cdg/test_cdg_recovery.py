"""
Unit tests for cortical.cdg.recovery module.

Tests crash recovery functionality using mocked storage and WAL components.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, PropertyMock

from cortical.cdg.recovery import (
    RecoveryResult,
    RepairResult,
    CDGRecoveryManager,
)
from cortical.cdg.config import CDGConfig, RecoveryMode, OrphanStrategy
from cortical.cdg.errors import CorruptionError


class TestRecoveryResult(unittest.TestCase):
    """Test RecoveryResult dataclass."""

    def test_default_values(self):
        """Test RecoveryResult with default values."""
        result = RecoveryResult(success=True, recovered_transactions=0)

        self.assertTrue(result.success)
        self.assertEqual(result.recovered_transactions, 0)
        self.assertEqual(result.rolled_back, [])
        self.assertEqual(result.corrupted_entities, [])
        self.assertEqual(result.corrupted_wal_entries, 0)
        self.assertEqual(result.orphans_detected, [])
        self.assertEqual(result.orphans_repaired, 0)
        self.assertFalse(result.indexes_rebuilt)
        self.assertEqual(result.actions_taken, [])

    def test_add_action(self):
        """Test adding actions to recovery result."""
        result = RecoveryResult(success=True, recovered_transactions=0)

        result.add_action("First action")
        result.add_action("Second action")

        self.assertEqual(len(result.actions_taken), 2)
        self.assertEqual(result.actions_taken[0], "First action")
        self.assertEqual(result.actions_taken[1], "Second action")

    def test_full_result(self):
        """Test RecoveryResult with all fields populated."""
        result = RecoveryResult(
            success=False,
            recovered_transactions=2,
            rolled_back=["tx1", "tx2"],
            corrupted_entities=["ent1"],
            corrupted_wal_entries=3,
            orphans_detected=["orphan1", "orphan2"],
            orphans_repaired=1,
            indexes_rebuilt=True,
            actions_taken=["Action 1", "Action 2"]
        )

        self.assertFalse(result.success)
        self.assertEqual(result.recovered_transactions, 2)
        self.assertEqual(len(result.rolled_back), 2)
        self.assertEqual(len(result.corrupted_entities), 1)
        self.assertEqual(result.corrupted_wal_entries, 3)
        self.assertEqual(len(result.orphans_detected), 2)
        self.assertEqual(result.orphans_repaired, 1)
        self.assertTrue(result.indexes_rebuilt)
        self.assertEqual(len(result.actions_taken), 2)


class TestRepairResult(unittest.TestCase):
    """Test RepairResult dataclass."""

    def test_default_values(self):
        """Test RepairResult with default values."""
        result = RepairResult(success=True, repaired_count=0)

        self.assertTrue(result.success)
        self.assertEqual(result.repaired_count, 0)
        self.assertEqual(result.repaired_entities, [])
        self.assertEqual(result.errors, [])

    def test_with_errors(self):
        """Test RepairResult with errors."""
        result = RepairResult(
            success=False,
            repaired_count=1,
            repaired_entities=["ent1"],
            errors=["Error 1", "Error 2"]
        )

        self.assertFalse(result.success)
        self.assertEqual(result.repaired_count, 1)
        self.assertEqual(len(result.errors), 2)


class TestCDGRecoveryManager(unittest.TestCase):
    """Test CDGRecoveryManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.store_dir = Path(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_creates_store_dir(self):
        """Test that initialization creates the store directory."""
        new_dir = self.store_dir / "new_store"
        config = CDGConfig.for_simple_storage()

        manager = CDGRecoveryManager(new_dir, config)

        self.assertTrue(new_dir.exists())

    def test_init_with_wal_disabled(self):
        """Test initialization with WAL disabled."""
        config = CDGConfig.for_simple_storage()
        config.enable_wal = False

        manager = CDGRecoveryManager(self.store_dir, config)

        self.assertIsNone(manager.wal)

    def test_init_with_wal_enabled(self):
        """Test initialization with WAL enabled."""
        config = CDGConfig.for_got()  # Has WAL enabled

        manager = CDGRecoveryManager(self.store_dir, config)

        self.assertIsNotNone(manager.wal)


class TestNeedsRecovery(unittest.TestCase):
    """Test needs_recovery() method."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.store_dir = Path(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_needs_recovery_none_mode(self):
        """Test that NONE mode never needs recovery."""
        config = CDGConfig.for_simple_storage()
        config.recovery_mode = RecoveryMode.NONE

        manager = CDGRecoveryManager(self.store_dir, config)

        self.assertFalse(manager.needs_recovery())

    def test_needs_recovery_no_issues(self):
        """Test needs_recovery when system is clean."""
        config = CDGConfig.for_simple_storage()
        config.recovery_mode = RecoveryMode.FULL
        config.enable_wal = False  # No WAL to check

        manager = CDGRecoveryManager(self.store_dir, config)

        # No entities, no WAL = no recovery needed
        self.assertFalse(manager.needs_recovery())


class TestRecover(unittest.TestCase):
    """Test recover() method."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.store_dir = Path(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_recover_none_mode(self):
        """Test recovery with NONE mode skips all recovery."""
        config = CDGConfig.for_simple_storage()
        config.recovery_mode = RecoveryMode.NONE

        manager = CDGRecoveryManager(self.store_dir, config)
        result = manager.recover()

        self.assertTrue(result.success)
        self.assertIn("Recovery skipped (mode=NONE)", result.actions_taken)

    def test_recover_checksum_mode_clean(self):
        """Test CHECKSUM mode with no corruption."""
        config = CDGConfig.for_simple_storage()
        config.recovery_mode = RecoveryMode.CHECKSUM

        manager = CDGRecoveryManager(self.store_dir, config)
        result = manager.recover()

        self.assertTrue(result.success)
        self.assertIn("Store integrity verified (no corruption)", result.actions_taken)

    def test_recover_full_mode_clean(self):
        """Test FULL mode with no issues."""
        config = CDGConfig.for_simple_storage()
        config.recovery_mode = RecoveryMode.FULL
        config.enable_wal = False

        manager = CDGRecoveryManager(self.store_dir, config)
        result = manager.recover()

        self.assertTrue(result.success)
        self.assertEqual(result.recovered_transactions, 0)


class TestVerifyStoreIntegrity(unittest.TestCase):
    """Test verify_store_integrity() method."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.store_dir = Path(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_verify_empty_store(self):
        """Test verification of empty store."""
        config = CDGConfig.for_simple_storage()
        manager = CDGRecoveryManager(self.store_dir, config)

        corrupted = manager.verify_store_integrity()

        self.assertEqual(corrupted, [])

    def test_verify_skips_temp_files(self):
        """Test that temporary files are skipped."""
        config = CDGConfig.for_simple_storage()
        manager = CDGRecoveryManager(self.store_dir, config)

        # Create a temp file that should be ignored
        temp_file = manager.store.store_dir / "_temp.json"
        temp_file.write_text("{}")

        corrupted = manager.verify_store_integrity()

        self.assertEqual(corrupted, [])

    def test_verify_skips_underscore_files(self):
        """Test that files starting with underscore are skipped."""
        config = CDGConfig.for_simple_storage()
        manager = CDGRecoveryManager(self.store_dir, config)

        # Create a file starting with underscore
        special_file = manager.store.store_dir / "_special.json"
        special_file.write_text("{}")

        corrupted = manager.verify_store_integrity()

        self.assertEqual(corrupted, [])


class TestVerifyWalIntegrity(unittest.TestCase):
    """Test verify_wal_integrity() method."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.store_dir = Path(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_verify_wal_disabled(self):
        """Test WAL verification when WAL is disabled."""
        config = CDGConfig.for_simple_storage()
        config.enable_wal = False

        manager = CDGRecoveryManager(self.store_dir, config)
        corrupted_count = manager.verify_wal_integrity()

        self.assertEqual(corrupted_count, 0)

    def test_verify_wal_no_file(self):
        """Test WAL verification when WAL file doesn't exist."""
        config = CDGConfig.for_got()

        manager = CDGRecoveryManager(self.store_dir, config)

        # Don't create the WAL file
        corrupted_count = manager.verify_wal_integrity()

        self.assertEqual(corrupted_count, 0)


class TestRollbackIncompleteTransactions(unittest.TestCase):
    """Test rollback_incomplete_transactions() method."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.store_dir = Path(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_rollback_no_wal(self):
        """Test rollback when WAL is disabled."""
        config = CDGConfig.for_simple_storage()
        config.enable_wal = False

        manager = CDGRecoveryManager(self.store_dir, config)
        rolled_back = manager.rollback_incomplete_transactions()

        self.assertEqual(rolled_back, [])


class TestDetectOrphanedEntities(unittest.TestCase):
    """Test detect_orphaned_entities() method."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.store_dir = Path(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_detect_orphans_no_wal(self):
        """Test orphan detection when WAL is disabled."""
        config = CDGConfig.for_simple_storage()
        config.enable_wal = False

        manager = CDGRecoveryManager(self.store_dir, config)
        orphans = manager.detect_orphaned_entities()

        self.assertEqual(orphans, [])


class TestNeedsIndexRecovery(unittest.TestCase):
    """Test needs_index_recovery() method."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.store_dir = Path(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_needs_index_recovery_returns_false(self):
        """Test that needs_index_recovery always returns False (placeholder)."""
        config = CDGConfig.for_simple_storage()

        manager = CDGRecoveryManager(self.store_dir, config)

        # This is a placeholder that always returns False
        self.assertFalse(manager.needs_index_recovery())


class TestRecoverWithIndexCallback(unittest.TestCase):
    """Test recovery with index rebuild callback."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.store_dir = Path(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_recover_with_index_callback_success(self):
        """Test recovery calls index rebuild callback on success."""
        config = CDGConfig.for_simple_storage()
        config.recovery_mode = RecoveryMode.FULL

        # Mock callback that returns task count
        callback = Mock(return_value=5)
        config.index_rebuild_callback = callback

        manager = CDGRecoveryManager(self.store_dir, config)
        result = manager.recover()

        self.assertTrue(result.success)
        self.assertTrue(result.indexes_rebuilt)
        callback.assert_called_once_with(self.store_dir)
        self.assertTrue(any("Rebuilt indexes" in action for action in result.actions_taken))

    def test_recover_with_index_callback_failure(self):
        """Test recovery handles index rebuild callback failure."""
        config = CDGConfig.for_simple_storage()
        config.recovery_mode = RecoveryMode.FULL

        # Mock callback that raises exception
        callback = Mock(side_effect=Exception("Index rebuild failed"))
        config.index_rebuild_callback = callback

        manager = CDGRecoveryManager(self.store_dir, config)
        result = manager.recover()

        self.assertFalse(result.success)
        self.assertFalse(result.indexes_rebuilt)
        self.assertTrue(any("Index rebuild failed" in action for action in result.actions_taken))


if __name__ == '__main__':
    unittest.main()
