"""
Tests for GoT recovery module.

Tests crash recovery, integrity verification, and recovery reporting.
"""

import json
import pytest
from pathlib import Path

from cortical.got import (
    TransactionManager,
    Task,
    CorruptionError,
)
from cortical.got.recovery import RecoveryManager, RecoveryResult
from cortical.utils.checksums import compute_checksum


class TestRecovery:
    """Test recovery operations."""

    def test_startup_recovery_rolls_back_incomplete(self, tmp_path):
        """Test that incomplete transactions are rolled back on startup."""
        # Create a transaction manager
        tm = TransactionManager(tmp_path)

        # Start a transaction but don't commit it
        tx = tm.begin()
        task = Task(id="T-incomplete", title="Incomplete task")
        tm.write(tx, task)

        # Simulate crash - don't commit, use RecoveryManager directly
        recovery_mgr = RecoveryManager(tmp_path)
        result = recovery_mgr.recover()

        # Transaction should have been rolled back
        assert len(result.rolled_back) >= 1
        assert any("incomplete" in action.lower() or "rolled back" in action.lower()
                   for action in result.actions_taken)

    def test_corrupted_entity_detected(self, tmp_path):
        """Test that corrupted entities are detected during recovery."""
        # Create and commit a task
        tm = TransactionManager(tmp_path)
        tx = tm.begin()
        task = Task(id="T-corrupted", title="Will be corrupted")
        tm.write(tx, task)
        tm.commit(tx)

        # Corrupt the entity file by modifying it
        entity_file = tmp_path / "entities" / "T-corrupted.json"
        with open(entity_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Modify the data without updating checksum
        data["data"]["title"] = "CORRUPTED"

        with open(entity_file, 'w', encoding='utf-8') as f:
            json.dump(data, f)

        # Run recovery
        recovery_mgr = RecoveryManager(tmp_path)
        result = recovery_mgr.recover()

        # Should detect corruption
        assert not result.success
        assert "T-corrupted" in result.corrupted_entities
        assert any("corrupted entity" in action.lower() for action in result.actions_taken)

    def test_corrupted_wal_detected(self, tmp_path):
        """Test that corrupted WAL entries are detected."""
        # Create transaction manager to write some WAL entries
        tm = TransactionManager(tmp_path)
        tx = tm.begin()
        task = Task(id="T-test", title="Test task")
        tm.write(tx, task)
        tm.commit(tx)

        # Corrupt a WAL entry
        wal_file = tmp_path / "wal" / "current.wal"
        with open(wal_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Corrupt the first line by changing checksum
        if lines:
            first_entry = json.loads(lines[0])
            first_entry['checksum'] = "0000000000000000"  # Invalid checksum
            lines[0] = json.dumps(first_entry) + '\n'

            with open(wal_file, 'w', encoding='utf-8') as f:
                f.writelines(lines)

        # Run recovery
        recovery_mgr = RecoveryManager(tmp_path)
        result = recovery_mgr.recover()

        # Should detect corrupted WAL entry
        assert result.corrupted_wal_entries > 0
        assert any("corrupted wal" in action.lower() for action in result.actions_taken)

    def test_recovery_from_clean_state(self, tmp_path):
        """Test that recovery is no-op when system is clean."""
        # Create a clean system
        tm = TransactionManager(tmp_path)
        tx = tm.begin()
        task = Task(id="T-clean", title="Clean task")
        tm.write(tx, task)
        tm.commit(tx)

        # Run recovery
        recovery_mgr = RecoveryManager(tmp_path)
        result = recovery_mgr.recover()

        # Should be clean
        assert result.success
        assert len(result.rolled_back) == 0
        assert len(result.corrupted_entities) == 0
        assert result.corrupted_wal_entries == 0
        assert any("clean" in action.lower() or "no recovery" in action.lower()
                   for action in result.actions_taken)

    def test_needs_recovery_true_with_incomplete_tx(self, tmp_path):
        """Test that needs_recovery() detects incomplete transactions."""
        # Create incomplete transaction
        tm = TransactionManager(tmp_path)
        tx = tm.begin()
        task = Task(id="T-incomplete", title="Incomplete")
        tm.write(tx, task)
        # Don't commit - simulate crash

        # Check needs_recovery
        recovery_mgr = RecoveryManager(tmp_path)
        assert recovery_mgr.needs_recovery() is True

    def test_needs_recovery_false_when_clean(self, tmp_path):
        """Test that needs_recovery() returns False for clean state."""
        # Create clean system
        tm = TransactionManager(tmp_path)
        tx = tm.begin()
        task = Task(id="T-clean", title="Clean")
        tm.write(tx, task)
        tm.commit(tx)

        # Check needs_recovery (create new manager to avoid auto-recovery)
        recovery_mgr = RecoveryManager(tmp_path)
        assert recovery_mgr.needs_recovery() is False

    def test_verify_store_integrity_returns_corrupted(self, tmp_path):
        """Test that verify_store_integrity() finds corrupted entities."""
        # Create entities
        tm = TransactionManager(tmp_path)
        tx = tm.begin()
        task1 = Task(id="T-good", title="Good task")
        task2 = Task(id="T-bad", title="Will be corrupted")
        tm.write(tx, task1)
        tm.write(tx, task2)
        tm.commit(tx)

        # Corrupt one entity
        entity_file = tmp_path / "entities" / "T-bad.json"
        with open(entity_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        data["data"]["title"] = "CORRUPTED"

        with open(entity_file, 'w', encoding='utf-8') as f:
            json.dump(data, f)

        # Verify integrity
        recovery_mgr = RecoveryManager(tmp_path)
        corrupted = recovery_mgr.verify_store_integrity()

        # Should find only the corrupted one
        assert len(corrupted) == 1
        assert "T-bad" in corrupted
        assert "T-good" not in corrupted

    def test_recovery_result_contains_actions(self, tmp_path):
        """Test that RecoveryResult properly logs actions."""
        # Create incomplete transaction
        tm = TransactionManager(tmp_path)
        tx = tm.begin()
        task = Task(id="T-test", title="Test")
        tm.write(tx, task)
        # Don't commit

        # Run recovery
        recovery_mgr = RecoveryManager(tmp_path)
        result = recovery_mgr.recover()

        # Should have actions logged
        assert len(result.actions_taken) > 0
        assert result.recovered_transactions > 0

        # Should be able to add actions
        result.add_action("Custom action")
        assert "Custom action" in result.actions_taken


class TestOrphanRepair:
    """Test orphan entity detection and repair."""

    def _create_orphan_file(self, entities_dir, entity_id, title="Orphaned", corrupted=False):
        """Helper to create an orphan entity file with correct format."""
        from cortical.utils.checksums import compute_checksum
        from datetime import datetime, timezone

        orphan_file = entities_dir / f"{entity_id}.json"

        orphan_entity_data = {
            "id": entity_id,
            "title": title,
            "entity_type": "task",
            "status": "pending",
            "priority": "medium",
            "description": "",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "modified_at": datetime.now(timezone.utc).isoformat(),
            "version": 1,
            "metadata": {},
            "properties": {}
        }

        orphan_wrapper = {
            "_checksum": compute_checksum(orphan_entity_data) if not corrupted else "invalid_checksum",
            "_written_at": datetime.now(timezone.utc).isoformat(),
            "data": orphan_entity_data
        }

        with open(orphan_file, 'w', encoding='utf-8') as f:
            json.dump(orphan_wrapper, f)

        return orphan_file

    def test_detect_orphaned_entities(self, tmp_path):
        """Test detection of orphaned entities."""
        # Create transaction manager and write a task
        tm = TransactionManager(tmp_path)
        tx = tm.begin()
        task = Task(id="T-normal", title="Normal task")
        tm.write(tx, task)
        tm.commit(tx)

        # Create an orphaned entity by directly writing to disk
        entities_dir = tmp_path / "entities"
        self._create_orphan_file(entities_dir, "T-orphan", "Orphaned task")

        # Detect orphans
        recovery_mgr = RecoveryManager(tmp_path)
        orphaned = recovery_mgr.detect_orphaned_entities()

        # Should detect the orphan
        assert "T-orphan" in orphaned
        assert "T-normal" not in orphaned

    def test_repair_orphans_delete(self, tmp_path):
        """Test that orphan files are deleted with 'delete' strategy."""
        # Create an orphaned entity
        entities_dir = tmp_path / "entities"
        entities_dir.mkdir(parents=True, exist_ok=True)

        orphan_file = self._create_orphan_file(entities_dir, "T-orphan")

        # Create WAL directory (empty WAL)
        wal_dir = tmp_path / "wal"
        wal_dir.mkdir(parents=True, exist_ok=True)
        (wal_dir / "current.wal").touch()

        # Repair orphans
        recovery_mgr = RecoveryManager(tmp_path)
        result = recovery_mgr.repair_orphans(strategy='delete')

        # Should have deleted the orphan
        assert result.success is True
        assert result.repaired_count == 1
        assert "T-orphan" in result.repaired_entities
        assert not orphan_file.exists()

    def test_repair_orphans_adopt(self, tmp_path):
        """Test that orphan files are added to WAL with 'adopt' strategy."""
        # Create an orphaned entity
        entities_dir = tmp_path / "entities"
        entities_dir.mkdir(parents=True, exist_ok=True)

        orphan_file = self._create_orphan_file(entities_dir, "T-orphan")

        # Create WAL directory
        wal_dir = tmp_path / "wal"
        wal_dir.mkdir(parents=True, exist_ok=True)
        wal_file = wal_dir / "current.wal"
        wal_file.touch()

        # Repair orphans with adopt strategy
        recovery_mgr = RecoveryManager(tmp_path)
        result = recovery_mgr.repair_orphans(strategy='adopt')

        # Should have adopted the orphan
        assert result.success is True
        assert result.repaired_count == 1
        assert "T-orphan" in result.repaired_entities
        assert orphan_file.exists()  # File should still exist

        # Check that WAL entry was added
        with open(wal_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["op"] == "ADOPTED"
        assert entry["entity_id"] == "T-orphan"
        assert entry["reason"] == "orphan_recovery"

    def test_repair_corrupted_orphan(self, tmp_path):
        """Test that corrupted orphan is deleted with error logged."""
        # Create a corrupted orphaned entity
        entities_dir = tmp_path / "entities"
        entities_dir.mkdir(parents=True, exist_ok=True)

        orphan_file = self._create_orphan_file(entities_dir, "T-corrupted-orphan", corrupted=True)

        # Create WAL directory
        wal_dir = tmp_path / "wal"
        wal_dir.mkdir(parents=True, exist_ok=True)
        (wal_dir / "current.wal").touch()

        # Try to adopt the corrupted orphan
        recovery_mgr = RecoveryManager(tmp_path)
        result = recovery_mgr.repair_orphans(strategy='adopt')

        # Should have deleted the corrupted orphan and logged error
        assert result.repaired_count == 1
        assert "T-corrupted-orphan" in result.repaired_entities
        assert not orphan_file.exists()
        assert len(result.errors) == 1
        assert "corrupted" in result.errors[0].lower()

    def test_recovery_repairs_orphans(self, tmp_path):
        """Test that full recovery includes orphan repair."""
        # Create an orphaned entity
        entities_dir = tmp_path / "entities"
        entities_dir.mkdir(parents=True, exist_ok=True)

        orphan_file = self._create_orphan_file(entities_dir, "T-orphan")

        # Create WAL directory
        wal_dir = tmp_path / "wal"
        wal_dir.mkdir(parents=True, exist_ok=True)
        (wal_dir / "current.wal").touch()

        # Run full recovery
        recovery_mgr = RecoveryManager(tmp_path)
        result = recovery_mgr.recover()

        # Should have adopted the orphan (valid checksum = preserve, add to WAL)
        assert any("adopted" in action.lower() for action in result.actions_taken)
        assert any("T-orphan" in action for action in result.actions_taken)
        # Valid orphan files are preserved (adopted), not deleted
        assert orphan_file.exists()

        # WAL should now have an ADOPTED entry for this entity
        wal_file = wal_dir / "current.wal"
        wal_content = wal_file.read_text()
        assert "T-orphan" in wal_content
        assert "ADOPTED" in wal_content

    def test_repair_orphans_invalid_strategy(self, tmp_path):
        """Test that invalid strategy raises ValueError."""
        recovery_mgr = RecoveryManager(tmp_path)

        with pytest.raises(ValueError) as exc_info:
            recovery_mgr.repair_orphans(strategy='invalid')

        assert "Invalid strategy" in str(exc_info.value)

    def test_repair_orphans_no_orphans(self, tmp_path):
        """Test that repair with no orphans returns empty result."""
        # Create a normal committed task
        tm = TransactionManager(tmp_path)
        tx = tm.begin()
        task = Task(id="T-normal", title="Normal")
        tm.write(tx, task)
        tm.commit(tx)

        # Repair orphans
        recovery_mgr = RecoveryManager(tmp_path)
        result = recovery_mgr.repair_orphans(strategy='delete')

        # Should have nothing to repair
        assert result.success is True
        assert result.repaired_count == 0
        assert len(result.repaired_entities) == 0
        assert len(result.errors) == 0


class TestOrphanRepairEdgeCases:
    """
    Edge case tests for orphan repair, especially for git-tracked entities.

    These tests verify the 'adopt' strategy handles scenarios where entity
    files exist (e.g., from git checkout) but WAL is missing or empty.
    """

    def _create_orphan_file(self, entities_dir, entity_id, title="Orphaned"):
        """Helper to create a valid orphan entity file."""
        from cortical.utils.checksums import compute_checksum
        from datetime import datetime, timezone

        orphan_file = entities_dir / f"{entity_id}.json"

        orphan_entity_data = {
            "id": entity_id,
            "title": title,
            "entity_type": "task",
            "status": "pending",
            "priority": "medium",
            "description": "",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "modified_at": datetime.now(timezone.utc).isoformat(),
            "version": 1,
            "metadata": {},
            "properties": {}
        }

        orphan_wrapper = {
            "_checksum": compute_checksum(orphan_entity_data),
            "_written_at": datetime.now(timezone.utc).isoformat(),
            "data": orphan_entity_data
        }

        with open(orphan_file, 'w', encoding='utf-8') as f:
            json.dump(orphan_wrapper, f)

        return orphan_file

    def test_fresh_clone_no_wal_file(self, tmp_path):
        """
        Test: Fresh git clone scenario - entities exist but no WAL file.

        This simulates checking out a repo where .got/entities/ is tracked
        but .got/wal/ is gitignored and doesn't exist.
        """
        # Create entities directory with orphan files (simulating git checkout)
        entities_dir = tmp_path / "entities"
        entities_dir.mkdir(parents=True, exist_ok=True)

        orphan1 = self._create_orphan_file(entities_dir, "T-task-1", "Task 1")
        orphan2 = self._create_orphan_file(entities_dir, "T-task-2", "Task 2")

        # NO WAL directory - simulating fresh clone
        # RecoveryManager will create it

        # Run recovery
        recovery_mgr = RecoveryManager(tmp_path)
        result = recovery_mgr.recover()

        # Both files should be preserved (adopted)
        assert orphan1.exists(), "Valid orphan 1 should be preserved"
        assert orphan2.exists(), "Valid orphan 2 should be preserved"

        # Should have adopted both
        assert result.recovered_transactions == 0  # No incomplete TX
        assert any("adopted" in action.lower() for action in result.actions_taken)

        # WAL should now have ADOPTED entries
        wal_file = tmp_path / "wal" / "current.wal"
        assert wal_file.exists(), "WAL file should be created"
        wal_content = wal_file.read_text()
        assert "T-task-1" in wal_content
        assert "T-task-2" in wal_content
        assert wal_content.count("ADOPTED") == 2

    def test_multiple_recovery_runs_idempotent(self, tmp_path):
        """
        Test: Multiple recovery runs don't cause WAL bloat.

        Once entities are adopted, subsequent recovery runs should
        not re-adopt them (they're no longer orphans).
        """
        # Create orphan entity
        entities_dir = tmp_path / "entities"
        entities_dir.mkdir(parents=True, exist_ok=True)
        orphan = self._create_orphan_file(entities_dir, "T-orphan")

        # Create WAL directory
        wal_dir = tmp_path / "wal"
        wal_dir.mkdir(parents=True, exist_ok=True)
        wal_file = wal_dir / "current.wal"
        wal_file.touch()

        recovery_mgr = RecoveryManager(tmp_path)

        # First recovery - should adopt
        result1 = recovery_mgr.recover()
        assert any("adopted" in action.lower() for action in result1.actions_taken)
        wal_content_after_first = wal_file.read_text()
        first_adopted_count = wal_content_after_first.count("ADOPTED")
        assert first_adopted_count == 1, "First run should adopt 1 entity"

        # Second recovery - should NOT adopt again (already in WAL)
        # Need new RecoveryManager to reset state
        recovery_mgr2 = RecoveryManager(tmp_path)
        result2 = recovery_mgr2.recover()

        wal_content_after_second = wal_file.read_text()
        second_adopted_count = wal_content_after_second.count("ADOPTED")
        assert second_adopted_count == 1, "Second run should NOT add more ADOPTED entries"

        # No orphan repair actions on second run
        orphan_actions = [a for a in result2.actions_taken if "adopted" in a.lower() or "orphan" in a.lower()]
        assert len(orphan_actions) == 0, "Second run should not have orphan repair actions"

    def test_adopted_entries_pass_wal_integrity(self, tmp_path):
        """
        Test: ADOPTED entries pass WAL integrity verification.

        The checksum computation for ADOPTED entries should be consistent
        so they pass verify_wal_integrity().
        """
        # Create orphan and adopt it
        entities_dir = tmp_path / "entities"
        entities_dir.mkdir(parents=True, exist_ok=True)
        self._create_orphan_file(entities_dir, "T-orphan")

        wal_dir = tmp_path / "wal"
        wal_dir.mkdir(parents=True, exist_ok=True)
        (wal_dir / "current.wal").touch()

        recovery_mgr = RecoveryManager(tmp_path)
        recovery_mgr.recover()

        # Verify WAL integrity - ADOPTED entries should pass
        corrupted_count = recovery_mgr.verify_wal_integrity()
        assert corrupted_count == 0, "ADOPTED entries should pass integrity check"

    def test_wal_truncated_entities_readopted(self, tmp_path):
        """
        Test: After WAL truncation, entities are re-adopted.

        If WAL is truncated (e.g., for cleanup), orphan entities should
        be re-adopted on next recovery without data loss.
        """
        # First: Create a committed task through normal transaction
        tm = TransactionManager(tmp_path)
        tx = tm.begin()
        task = Task(id="T-committed", title="Committed Task")
        tm.write(tx, task)
        tm.commit(tx)

        # Verify entity exists
        entity_file = tmp_path / "entities" / "T-committed.json"
        assert entity_file.exists()

        # Truncate WAL (simulating cleanup)
        wal_file = tmp_path / "wal" / "current.wal"
        wal_file.write_text("")  # Empty the WAL

        # Entity is now an "orphan" (no WAL entry)
        recovery_mgr = RecoveryManager(tmp_path)
        orphans = recovery_mgr.detect_orphaned_entities()
        assert "T-committed" in orphans, "Entity should be detected as orphan after WAL truncation"

        # Recovery should re-adopt it
        result = recovery_mgr.recover()

        # Entity should still exist
        assert entity_file.exists(), "Entity should be preserved after re-adoption"

        # Should have been adopted
        assert any("T-committed" in action for action in result.actions_taken)

        # No longer an orphan
        orphans_after = recovery_mgr.detect_orphaned_entities()
        assert "T-committed" not in orphans_after
