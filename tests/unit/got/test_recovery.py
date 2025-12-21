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
from cortical.got.checksums import compute_checksum


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
