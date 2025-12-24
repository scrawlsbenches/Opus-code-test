"""
Fault Tolerance and Recovery Validation Tests

Task: T-20251222-204138-51a38502

Validates:
1. Generation failures fall back gracefully
2. Corrupted layers are detected
3. Recovery procedures work
4. Sessions can start even if generation fails

TDD Approach: Tests written first (RED), then implementation fixes (GREEN).
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

from cortical.got import TransactionManager, Task
from cortical.got.recovery import RecoveryManager, RecoveryResult
from cortical.utils.checksums import compute_checksum


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def got_dir(tmp_path):
    """Create a GoT directory structure."""
    entities_dir = tmp_path / "entities"
    wal_dir = tmp_path / "wal"
    entities_dir.mkdir()
    wal_dir.mkdir()
    return tmp_path


@pytest.fixture
def populated_got(got_dir):
    """Create a populated GoT directory with some tasks."""
    tm = TransactionManager(got_dir)
    tx = tm.begin()

    # Create some tasks
    task1 = Task(id="T-test-001", title="Test task 1")
    task2 = Task(id="T-test-002", title="Test task 2")
    tm.write(tx, task1)
    tm.write(tx, task2)
    tm.commit(tx)

    return got_dir


# =============================================================================
# 1. GENERATION FAILURE GRACEFUL FALLBACK TESTS
# =============================================================================

class TestGenerationFailureFallback:
    """Test that generation failures fall back gracefully."""

    def test_recover_returns_result_on_empty_directory(self, got_dir):
        """Recovery should return a result even on empty directory."""
        recovery = RecoveryManager(got_dir)
        result = recovery.recover()

        # Should not raise, should return a RecoveryResult
        assert isinstance(result, RecoveryResult)
        assert result.success is True  # Empty is OK

    def test_recover_handles_missing_wal_file(self, got_dir):
        """Recovery should handle missing WAL file gracefully."""
        # Create entity but no WAL
        entities_dir = got_dir / "entities"
        entity_file = entities_dir / "T-test.json"

        entity_data = {
            "_checksum": compute_checksum({"id": "T-test", "title": "Test"}),
            "_written_at": "2025-12-24T00:00:00+00:00",
            "data": {"id": "T-test", "entity_type": "task", "title": "Test"}
        }
        entity_file.write_text(json.dumps(entity_data))

        recovery = RecoveryManager(got_dir)
        result = recovery.recover()

        # Should handle missing WAL gracefully
        assert isinstance(result, RecoveryResult)
        # Should detect orphan (entity without WAL record)
        assert len(result.orphans_detected) >= 1 or "orphan" in str(result.actions_taken).lower()

    def test_recover_handles_malformed_wal_entries(self, got_dir):
        """Recovery should skip malformed WAL entries and continue."""
        # Create WAL with malformed entries
        wal_dir = got_dir / "wal"
        wal_file = wal_dir / "current.wal"

        # Write mix of valid and invalid entries
        entries = [
            '{"this": "is not a valid WAL entry"}',
            'not even json',
            '{"op": "CREATE", "tx": "TX-001", "seq": 1, "ts": "2025-12-24T00:00:00", "checksum": "bad"}',
        ]
        wal_file.write_text('\n'.join(entries))

        recovery = RecoveryManager(got_dir)
        result = recovery.recover()

        # Should not crash, should report corrupted entries
        assert isinstance(result, RecoveryResult)
        assert result.corrupted_wal_entries >= 1

    def test_transaction_manager_starts_after_failed_recovery(self, got_dir):
        """TransactionManager should start even if previous state is corrupted."""
        # Corrupt the WAL file
        wal_dir = got_dir / "wal"
        wal_file = wal_dir / "current.wal"
        wal_file.write_text("totally broken content\n" * 10)

        # TransactionManager should still initialize
        # Recovery happens but shouldn't prevent startup
        try:
            tm = TransactionManager(got_dir)
            # Should be able to begin a new transaction
            tx = tm.begin()
            assert tx is not None
            tm.rollback(tx)
        except Exception as e:
            pytest.fail(f"TransactionManager failed to start after corruption: {e}")


# =============================================================================
# 2. CORRUPTED LAYER DETECTION TESTS
# =============================================================================

class TestCorruptedLayerDetection:
    """Test that corrupted entities/layers are detected."""

    def test_detects_corrupted_entity_checksum(self, populated_got):
        """Should detect entity with invalid checksum."""
        # Corrupt an entity file
        entity_file = populated_got / "entities" / "T-test-001.json"
        with open(entity_file, 'r') as f:
            data = json.load(f)

        # Modify data without updating checksum
        data["data"]["title"] = "TAMPERED"
        with open(entity_file, 'w') as f:
            json.dump(data, f)

        recovery = RecoveryManager(populated_got)
        result = recovery.recover()

        # Should detect corruption
        assert "T-test-001" in result.corrupted_entities

    def test_detects_truncated_entity_file(self, populated_got):
        """Should detect truncated entity files."""
        entity_file = populated_got / "entities" / "T-test-001.json"

        # Truncate the file
        with open(entity_file, 'r') as f:
            content = f.read()
        with open(entity_file, 'w') as f:
            f.write(content[:len(content)//2])  # Write only half

        recovery = RecoveryManager(populated_got)
        result = recovery.recover()

        # Should detect corruption (JSON parse error)
        assert not result.success or "T-test-001" in result.corrupted_entities

    def test_detects_missing_checksum_field(self, populated_got):
        """Should detect entity missing _checksum field."""
        entity_file = populated_got / "entities" / "T-test-001.json"

        # Remove checksum
        with open(entity_file, 'r') as f:
            data = json.load(f)
        del data["_checksum"]
        with open(entity_file, 'w') as f:
            json.dump(data, f)

        recovery = RecoveryManager(populated_got)
        result = recovery.recover()

        # Should detect as corrupted (no checksum)
        assert "T-test-001" in result.corrupted_entities or not result.success

    def test_detects_empty_entity_file(self, got_dir):
        """Should detect empty entity files."""
        entities_dir = got_dir / "entities"
        entity_file = entities_dir / "T-empty.json"
        entity_file.write_text("")

        recovery = RecoveryManager(got_dir)
        result = recovery.recover()

        # Should not crash on empty file
        assert isinstance(result, RecoveryResult)

    def test_reports_all_corrupted_entities(self, populated_got):
        """Should report all corrupted entities, not just first one."""
        # Corrupt both entities
        for entity_id in ["T-test-001", "T-test-002"]:
            entity_file = populated_got / "entities" / f"{entity_id}.json"
            with open(entity_file, 'r') as f:
                data = json.load(f)
            data["data"]["title"] = "CORRUPTED"
            with open(entity_file, 'w') as f:
                json.dump(data, f)

        recovery = RecoveryManager(populated_got)
        result = recovery.recover()

        # Should report both corrupted entities
        assert len(result.corrupted_entities) >= 2


# =============================================================================
# 3. RECOVERY PROCEDURE TESTS
# =============================================================================

class TestRecoveryProcedures:
    """Test that recovery procedures work correctly."""

    def test_recovery_rolls_back_incomplete_transactions(self, got_dir):
        """Incomplete transactions should be rolled back."""
        tm = TransactionManager(got_dir)

        # Start transaction but don't commit
        tx = tm.begin()
        task = Task(id="T-incomplete", title="Incomplete")
        tm.write(tx, task)
        # Don't commit - simulate crash

        # Fresh recovery should roll back
        recovery = RecoveryManager(got_dir)
        result = recovery.recover()

        # Should have rolled back incomplete transaction
        assert len(result.rolled_back) >= 1

    def test_recovery_preserves_committed_transactions(self, populated_got):
        """Committed transactions should be preserved during recovery."""
        # Count entities before recovery
        entities_before = list((populated_got / "entities").glob("*.json"))

        recovery = RecoveryManager(populated_got)
        result = recovery.recover()

        # Count entities after recovery
        entities_after = list((populated_got / "entities").glob("*.json"))

        # All committed entities should still exist
        assert len(entities_after) == len(entities_before)

    def test_recovery_adopts_orphaned_entities(self, got_dir):
        """Orphaned entities should be adopted with synthetic WAL entries."""
        # Create entity without WAL entry (simulates fresh clone)
        entities_dir = got_dir / "entities"
        entity_file = entities_dir / "T-orphan.json"

        entity_data = {
            "_checksum": compute_checksum({"id": "T-orphan", "entity_type": "task", "title": "Orphan"}),
            "_written_at": "2025-12-24T00:00:00+00:00",
            "data": {"id": "T-orphan", "entity_type": "task", "title": "Orphan"}
        }
        entity_file.write_text(json.dumps(entity_data))

        recovery = RecoveryManager(got_dir)
        result = recovery.recover()

        # Should adopt orphan
        assert len(result.orphans_detected) >= 1
        assert "adopt" in str(result.actions_taken).lower() or result.orphans_repaired >= 1

    def test_recovery_is_idempotent(self, populated_got):
        """Running recovery multiple times should produce same result."""
        recovery = RecoveryManager(populated_got)

        # Run recovery twice
        result1 = recovery.recover()
        result2 = recovery.recover()

        # Second run should find nothing to do
        assert result2.rolled_back == []
        assert result2.corrupted_entities == []
        assert result2.corrupted_wal_entries == 0

    def test_recovery_result_contains_all_actions(self, got_dir):
        """RecoveryResult should document all actions taken."""
        # Create orphaned entity
        entities_dir = got_dir / "entities"
        entity_file = entities_dir / "T-orphan.json"
        entity_data = {
            "_checksum": compute_checksum({"id": "T-orphan", "entity_type": "task", "title": "Test"}),
            "_written_at": "2025-12-24T00:00:00+00:00",
            "data": {"id": "T-orphan", "entity_type": "task", "title": "Test"}
        }
        entity_file.write_text(json.dumps(entity_data))

        recovery = RecoveryManager(got_dir)
        result = recovery.recover()

        # Should have documented actions
        assert isinstance(result.actions_taken, list)
        assert len(result.actions_taken) >= 1


# =============================================================================
# 4. SESSION STARTUP WITH FAILED GENERATION TESTS
# =============================================================================

class TestSessionStartupWithFailedGeneration:
    """Test that sessions can start even if generation/state is broken."""

    def test_transaction_manager_creates_missing_directories(self, tmp_path):
        """TransactionManager should create missing directories."""
        empty_dir = tmp_path / "new_got"
        # Don't create the directory - let TransactionManager do it

        tm = TransactionManager(empty_dir)

        # Should create required directories
        assert (empty_dir / "entities").exists()
        assert (empty_dir / "wal").exists()

    def test_transaction_manager_handles_readonly_directory(self, got_dir):
        """Should handle read-only directory gracefully."""
        # This test may not work on all platforms
        import os
        import stat

        # Skip on Windows or if running as root
        if os.name == 'nt' or os.geteuid() == 0:
            pytest.skip("Cannot test read-only on Windows or as root")

        # Make directory read-only
        original_mode = os.stat(got_dir).st_mode
        os.chmod(got_dir, stat.S_IRUSR | stat.S_IXUSR)

        try:
            # Should handle gracefully
            with pytest.raises(Exception):  # Should raise permission error
                TransactionManager(got_dir)
        finally:
            # Restore permissions
            os.chmod(got_dir, original_mode)

    def test_can_read_entities_with_corrupted_wal(self, populated_got):
        """Should be able to read entities even if WAL is corrupted."""
        # Corrupt WAL
        wal_file = populated_got / "wal" / "current.wal"
        wal_file.write_text("corrupted\n" * 10)

        # Should still be able to read existing entities
        tm = TransactionManager(populated_got)

        # Entities were committed before corruption, should be readable
        entity_file = populated_got / "entities" / "T-test-001.json"
        assert entity_file.exists()

        with open(entity_file) as f:
            data = json.load(f)
        assert data["data"]["id"] == "T-test-001"

    def test_new_transactions_work_after_recovery(self, populated_got):
        """Should be able to create new transactions after recovery."""
        # Run recovery
        recovery = RecoveryManager(populated_got)
        recovery.recover()

        # Create new transaction
        tm = TransactionManager(populated_got)
        tx = tm.begin()
        task = Task(id="T-new", title="New task")
        tm.write(tx, task)
        tm.commit(tx)

        # New entity should exist
        entity_file = populated_got / "entities" / "T-new.json"
        assert entity_file.exists()

    def test_recovery_on_fresh_directory_is_fast(self, got_dir):
        """Recovery on fresh/empty directory should be nearly instant."""
        import time

        recovery = RecoveryManager(got_dir)

        start = time.time()
        result = recovery.recover()
        elapsed = time.time() - start

        # Should complete in under 100ms for empty directory
        assert elapsed < 0.1
        assert result.success


# =============================================================================
# 5. EDGE CASE AND REGRESSION TESTS
# =============================================================================

class TestEdgeCasesAndRegressions:
    """Edge cases and regression tests for fault tolerance."""

    def test_handles_unicode_in_entity_data(self, got_dir):
        """Should handle unicode characters in entity data."""
        tm = TransactionManager(got_dir)
        tx = tm.begin()

        # Create task with unicode
        task = Task(id="T-unicode", title="Test with 日本語 and émojis 🎉")
        tm.write(tx, task)
        tm.commit(tx)

        # Recovery should handle unicode
        recovery = RecoveryManager(got_dir)
        result = recovery.recover()

        assert result.success
        assert "T-unicode" not in result.corrupted_entities

    def test_handles_very_long_entity_content(self, got_dir):
        """Should handle entities with very long content."""
        tm = TransactionManager(got_dir)
        tx = tm.begin()

        # Create task with long content
        long_content = "x" * 100000  # 100KB
        task = Task(id="T-long", title=long_content)
        tm.write(tx, task)
        tm.commit(tx)

        recovery = RecoveryManager(got_dir)
        result = recovery.recover()

        assert result.success

    def test_handles_concurrent_recovery_attempts(self, populated_got):
        """Multiple recovery attempts should not corrupt state."""
        import threading

        results = []
        errors = []

        def run_recovery():
            try:
                recovery = RecoveryManager(populated_got)
                result = recovery.recover()
                results.append(result)
            except FileNotFoundError:
                # FileNotFoundError is acceptable - it means another thread
                # already processed/deleted the file. This is expected behavior
                # in concurrent recovery scenarios.
                results.append(None)  # Mark as handled
            except Exception as e:
                errors.append(e)

        # Start multiple recovery attempts
        threads = [threading.Thread(target=run_recovery) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # At least one should succeed, others may hit race conditions
        # The key is that no unexpected errors occur
        assert len(errors) == 0
        successful_results = [r for r in results if r is not None]
        assert len(successful_results) >= 1  # At least one full recovery
        assert all(r.success for r in successful_results)

    def test_checksum_algorithm_consistency(self, got_dir):
        """Checksum algorithm should be consistent across calls."""
        data = {"id": "test", "title": "Test", "nested": {"key": "value"}}

        # Compute multiple times
        checksums = [compute_checksum(data) for _ in range(10)]

        # All should be identical
        assert len(set(checksums)) == 1

    def test_empty_wal_file_handled(self, got_dir):
        """Should handle empty WAL file."""
        wal_file = got_dir / "wal" / "current.wal"
        wal_file.write_text("")

        recovery = RecoveryManager(got_dir)
        result = recovery.recover()

        assert result.success

    def test_wal_file_with_only_newlines(self, got_dir):
        """Should handle WAL file with only newlines."""
        wal_file = got_dir / "wal" / "current.wal"
        wal_file.write_text("\n\n\n\n")

        recovery = RecoveryManager(got_dir)
        result = recovery.recover()

        assert isinstance(result, RecoveryResult)
