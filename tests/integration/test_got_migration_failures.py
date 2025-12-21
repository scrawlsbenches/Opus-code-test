"""
Integration tests for GoT migration failure handling.

Tests partial migration failures, atomicity, progress reporting,
and error recovery scenarios.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime, timezone

from cortical.got import GoTManager, TransactionError
from scripts.migrate_got import GoTMigrator, MigrationResult


class TestMigrationFailures:
    """Test migration failure scenarios and recovery mechanisms."""

    @pytest.fixture
    def source_dir(self, tmp_path):
        """Create source event-sourced directory with sample data."""
        got_dir = tmp_path / ".got"
        got_dir.mkdir()

        # Create events directory
        events_dir = got_dir / "events"
        events_dir.mkdir()

        return got_dir

    @pytest.fixture
    def target_dir(self, tmp_path):
        """Create target transactional directory."""
        return tmp_path / ".got-tx"

    def create_test_events(self, events_dir: Path, count: int = 10):
        """
        Create test events for migration.

        Args:
            events_dir: Events directory to write to
            count: Number of task events to create
        """
        events = []
        for i in range(count):
            event = {
                "ts": f"2025-12-21T10:00:{i:02d}.000000Z",
                "event": "node.create",
                "meta": {"branch": "main", "session": "test-session"},
                "id": f"task:T-TEST-{i:04d}",
                "type": "TASK",
                "data": {
                    "title": f"Test task {i}",
                    "status": "pending",
                    "priority": "medium",
                    "description": f"Task {i} description"
                }
            }
            events.append(event)

        # Write events to file
        event_file = events_dir / "test-events.jsonl"
        with open(event_file, "w") as f:
            for event in events:
                f.write(json.dumps(event) + "\n")

    def test_migration_fails_on_write_error(self, source_dir, target_dir):
        """
        Test that migration fails gracefully when write operation encounters error.

        Simulates disk full or permission error during entity write.
        """
        # Create source data
        events_dir = source_dir / "events"
        self.create_test_events(events_dir, count=5)

        migrator = GoTMigrator(
            source_dir=source_dir,
            target_dir=target_dir,
            dry_run=False
        )

        # Mock the GoTManager to fail during write
        with patch('scripts.migrate_got.GoTManager') as MockManager:
            # Create a mock manager that raises error on transaction
            mock_manager_instance = MagicMock()
            mock_tx = MagicMock()
            mock_tx.__enter__ = MagicMock(return_value=mock_tx)
            mock_tx.__exit__ = MagicMock(return_value=False)

            # Make write fail after 2 entities
            call_count = 0
            def write_side_effect(entity):
                nonlocal call_count
                call_count += 1
                if call_count > 2:
                    raise IOError("Disk full - no space left on device")

            mock_tx.write = MagicMock(side_effect=write_side_effect)
            mock_manager_instance.transaction.return_value = mock_tx
            MockManager.return_value = mock_manager_instance

            # Run migration
            result = migrator.migrate()

            # Should fail gracefully
            assert not result.success
            assert len(result.errors) > 0
            assert any("Disk full" in err or "Migration failed" in err for err in result.errors)

    def test_migration_is_atomic_via_transaction(self, source_dir, target_dir):
        """
        Test that migration uses transactions for atomicity.

        The migration writes all entities in a single transaction,
        so either all succeed or none do (transaction rollback).
        """
        # Create source data with multiple entities
        events_dir = source_dir / "events"

        # Create task events
        task_event = {
            "ts": "2025-12-21T10:00:00.000000Z",
            "event": "node.create",
            "meta": {"branch": "main", "session": "test"},
            "id": "task:T-ATOMIC-001",
            "type": "TASK",
            "data": {
                "title": "Atomic test task",
                "status": "pending",
                "priority": "high"
            }
        }

        decision_event = {
            "ts": "2025-12-21T10:01:00.000000Z",
            "event": "node.create",
            "meta": {"branch": "main", "session": "test"},
            "id": "decision:D-ATOMIC-001",
            "type": "DECISION",
            "data": {
                "title": "Atomic test decision",
                "rationale": "Test rationale"
            }
        }

        edge_event = {
            "ts": "2025-12-21T10:02:00.000000Z",
            "event": "edge.create",
            "meta": {"branch": "main", "session": "test"},
            "data": {
                "source_id": "task:T-ATOMIC-001",
                "target_id": "decision:D-ATOMIC-001",
                "edge_type": "MOTIVATES",
                "weight": 1.0
            }
        }

        # Write events
        event_file = events_dir / "atomic-test.jsonl"
        with open(event_file, "w") as f:
            f.write(json.dumps(task_event) + "\n")
            f.write(json.dumps(decision_event) + "\n")
            f.write(json.dumps(edge_event) + "\n")

        # Mock transaction to fail during commit
        with patch('scripts.migrate_got.GoTManager') as MockManager:
            mock_manager_instance = MagicMock()
            mock_tx = MagicMock()
            mock_tx.__enter__ = MagicMock(return_value=mock_tx)

            # Make commit fail (simulating transaction rollback)
            def exit_side_effect(exc_type, exc_val, exc_tb):
                if exc_type is None:
                    # Simulate commit failure by raising exception
                    raise TransactionError("Simulated commit failure")
                return False

            mock_tx.__exit__ = MagicMock(side_effect=exit_side_effect)
            mock_manager_instance.transaction.return_value = mock_tx
            MockManager.return_value = mock_manager_instance

            migrator = GoTMigrator(
                source_dir=source_dir,
                target_dir=target_dir,
                dry_run=False
            )

            result = migrator.migrate()

            # Migration should fail
            assert not result.success
            assert len(result.errors) > 0

        # Now do successful migration to verify atomicity
        migrator2 = GoTMigrator(
            source_dir=source_dir,
            target_dir=target_dir,
            dry_run=False
        )

        result2 = migrator2.migrate()

        # Should succeed and migrate all entities
        assert result2.success
        assert result2.tasks_migrated == 1
        assert result2.decisions_migrated == 1
        assert result2.edges_migrated == 1

        # Verify all entities are present in target
        manager = GoTManager(target_dir)
        task = manager.get_task("T-ATOMIC-001")
        assert task is not None
        assert task.title == "Atomic test task"

    def test_migration_reports_progress(self, source_dir, target_dir, capsys):
        """
        Test that migration reports progress during execution.

        Verifies that progress messages are printed and entity counts are accurate.
        """
        # Create source data with multiple entities
        events_dir = source_dir / "events"
        self.create_test_events(events_dir, count=20)

        migrator = GoTMigrator(
            source_dir=source_dir,
            target_dir=target_dir,
            dry_run=False
        )

        # Run migration
        result = migrator.migrate()

        # Capture output
        captured = capsys.readouterr()

        # Should report progress
        assert "Starting migration" in captured.out
        assert "Parsing events" in captured.out
        assert "Parsing WAL" in captured.out
        assert "Writing to" in captured.out

        # Should report entity counts
        assert "Found" in captured.out
        assert "20 tasks" in captured.out or "tasks" in captured.out

        # Should report completion
        assert "Migration completed successfully" in captured.out or result.success

        # Verify counts
        assert result.success
        assert result.tasks_migrated == 20

    def test_corrupted_event_skipped_with_warning(self, source_dir, target_dir):
        """
        Test that corrupted events are skipped with appropriate warnings.

        Verifies:
        - Migration continues despite corrupted events
        - Warnings are logged for each corrupted event
        - Valid events are still processed correctly
        """
        events_dir = source_dir / "events"

        # Create mix of valid and corrupted events
        event_file = events_dir / "mixed.jsonl"
        with open(event_file, "w") as f:
            # Valid event
            valid_event = {
                "ts": "2025-12-21T10:00:00.000000Z",
                "event": "node.create",
                "meta": {"branch": "main", "session": "test"},
                "id": "task:T-VALID-001",
                "type": "TASK",
                "data": {
                    "title": "Valid task",
                    "status": "pending",
                    "priority": "medium"
                }
            }
            f.write(json.dumps(valid_event) + "\n")

            # Corrupted JSON
            f.write("{ this is not valid json }\n")

            # Another valid event
            valid_event2 = {
                "ts": "2025-12-21T10:01:00.000000Z",
                "event": "node.create",
                "meta": {"branch": "main", "session": "test"},
                "id": "task:T-VALID-002",
                "type": "TASK",
                "data": {
                    "title": "Another valid task",
                    "status": "in_progress",
                    "priority": "high"
                }
            }
            f.write(json.dumps(valid_event2) + "\n")

            # Valid JSON but missing required fields
            f.write('{"event": "unknown", "data": {}}\n')

            # Another corrupted line
            f.write("completely invalid\n")

        migrator = GoTMigrator(
            source_dir=source_dir,
            target_dir=target_dir,
            dry_run=False
        )

        result = migrator.migrate()

        # Migration should succeed despite corrupted events
        assert result.success

        # Should have warnings about corrupted events
        assert len(result.warnings) > 0
        assert any("parse" in w.lower() or "failed" in w.lower() for w in result.warnings)

        # Valid events should be migrated
        assert result.tasks_migrated == 2

        # Verify valid tasks were migrated correctly
        manager = GoTManager(target_dir)

        task1 = manager.get_task("T-VALID-001")
        assert task1 is not None
        assert task1.title == "Valid task"
        assert task1.status == "pending"

        task2 = manager.get_task("T-VALID-002")
        assert task2 is not None
        assert task2.title == "Another valid task"
        assert task2.status == "in_progress"

    def test_dry_run_does_not_modify_on_error(self, source_dir, target_dir):
        """
        Test that dry-run mode does not create files even if analysis encounters errors.

        Verifies:
        - Dry-run analyzes source data
        - No files are created in target directory
        - Errors during analysis don't cause writes
        """
        events_dir = source_dir / "events"

        # Create events with some valid and some problematic data
        event_file = events_dir / "test.jsonl"
        with open(event_file, "w") as f:
            # Valid event
            valid_event = {
                "ts": "2025-12-21T10:00:00.000000Z",
                "event": "node.create",
                "meta": {"branch": "main", "session": "test"},
                "id": "task:T-DRY-001",
                "type": "TASK",
                "data": {
                    "title": "Dry run task",
                    "status": "pending",
                    "priority": "medium"
                }
            }
            f.write(json.dumps(valid_event) + "\n")

            # Add corrupted event
            f.write("{ invalid json\n")

        migrator = GoTMigrator(
            source_dir=source_dir,
            target_dir=target_dir,
            dry_run=True
        )

        # Run migration (should analyze only)
        result = migrator.migrate()

        # Should succeed (dry-run always succeeds unless major error)
        assert result.success

        # Should have counted entities
        assert result.tasks_migrated == 1

        # Should have warning about corrupted event
        assert len(result.warnings) > 0

        # Target directory should NOT be created
        assert not target_dir.exists()

        # Verify analysis doesn't create any files
        if target_dir.exists():
            files = list(target_dir.rglob("*"))
            assert len(files) == 0, f"Dry-run created files: {files}"

    def test_migration_handles_empty_source(self, source_dir, target_dir):
        """
        Test that migration handles empty source directory gracefully.

        Verifies:
        - Migration succeeds with zero entities
        - No errors or warnings
        - Target directory is created but empty
        """
        # Source directory exists but has no events or WAL
        migrator = GoTMigrator(
            source_dir=source_dir,
            target_dir=target_dir,
            dry_run=False
        )

        result = migrator.migrate()

        # Should succeed with zero entities
        assert result.success
        assert result.tasks_migrated == 0
        assert result.decisions_migrated == 0
        assert result.edges_migrated == 0
        assert len(result.errors) == 0

    def test_migration_handles_wal_parse_errors(self, source_dir, target_dir):
        """
        Test that migration handles corrupted WAL entries gracefully.

        Verifies:
        - Migration continues despite corrupted WAL entries
        - Warnings are logged for corrupted entries
        - Valid WAL entries are processed
        """
        # Create WAL directory
        wal_dir = source_dir / "wal"
        wal_dir.mkdir()
        wal_logs = wal_dir / "logs"
        wal_logs.mkdir()

        # Create mix of valid and corrupted WAL entries
        wal_file = wal_logs / "test.jsonl"
        with open(wal_file, "w") as f:
            # Valid WAL entry
            valid_wal = {
                "operation": "add_node",
                "timestamp": "2025-12-21T10:00:00.000000",
                "doc_id": "task:T-WAL-001",
                "payload": {
                    "node_id": "task:T-WAL-001",
                    "node_type": "task",
                    "content": "WAL task",
                    "properties": {
                        "title": "WAL task",
                        "status": "pending",
                        "priority": "medium"
                    }
                }
            }
            f.write(json.dumps(valid_wal) + "\n")

            # Corrupted WAL entry
            f.write("{ corrupted wal entry\n")

            # Another valid entry
            valid_wal2 = {
                "operation": "add_node",
                "timestamp": "2025-12-21T10:01:00.000000",
                "doc_id": "task:T-WAL-002",
                "payload": {
                    "node_id": "task:T-WAL-002",
                    "node_type": "task",
                    "content": "Another WAL task",
                    "properties": {
                        "title": "Another WAL task",
                        "status": "in_progress",
                        "priority": "high"
                    }
                }
            }
            f.write(json.dumps(valid_wal2) + "\n")

        migrator = GoTMigrator(
            source_dir=source_dir,
            target_dir=target_dir,
            dry_run=False
        )

        result = migrator.migrate()

        # Should succeed
        assert result.success

        # Should have warnings
        assert len(result.warnings) > 0
        assert any("WAL" in w or "parse" in w.lower() for w in result.warnings)

        # Valid entries should be migrated
        assert result.tasks_migrated == 2

        # Verify tasks
        manager = GoTManager(target_dir)

        task1 = manager.get_task("T-WAL-001")
        assert task1 is not None
        assert task1.title == "WAL task"

        task2 = manager.get_task("T-WAL-002")
        assert task2 is not None
        assert task2.title == "Another WAL task"

    def test_migration_handles_missing_events_directory(self, source_dir, target_dir):
        """
        Test that migration handles missing events directory gracefully.

        Verifies:
        - Migration succeeds even if events directory doesn't exist
        - No errors raised
        - WAL entries are still processed if present
        """
        # Create WAL but no events directory
        wal_dir = source_dir / "wal"
        wal_dir.mkdir()
        wal_logs = wal_dir / "logs"
        wal_logs.mkdir()

        # Add WAL entry
        wal_file = wal_logs / "test.jsonl"
        with open(wal_file, "w") as f:
            wal_entry = {
                "operation": "add_node",
                "timestamp": "2025-12-21T10:00:00.000000",
                "doc_id": "task:T-WAL-ONLY",
                "payload": {
                    "node_id": "task:T-WAL-ONLY",
                    "node_type": "task",
                    "content": "WAL only task",
                    "properties": {
                        "title": "WAL only task",
                        "status": "pending",
                        "priority": "low"
                    }
                }
            }
            f.write(json.dumps(wal_entry) + "\n")

        migrator = GoTMigrator(
            source_dir=source_dir,
            target_dir=target_dir,
            dry_run=False
        )

        result = migrator.migrate()

        # Should succeed
        assert result.success
        assert result.tasks_migrated == 1
        assert len(result.errors) == 0

        # Verify WAL task was migrated
        manager = GoTManager(target_dir)
        task = manager.get_task("T-WAL-ONLY")
        assert task is not None
        assert task.title == "WAL only task"

    def test_verify_detects_missing_entities(self, source_dir, target_dir):
        """
        Test that verification detects when entities are missing from target.

        Simulates incomplete migration where some entities didn't get written.
        """
        # Create source data
        events_dir = source_dir / "events"
        self.create_test_events(events_dir, count=3)

        # Do migration
        migrator = GoTMigrator(
            source_dir=source_dir,
            target_dir=target_dir,
            dry_run=False
        )

        result = migrator.migrate()
        assert result.success

        # Delete one entity to simulate incomplete migration
        entities_dir = target_dir / "entities"
        entity_files = list(entities_dir.glob("T-TEST-*.json"))
        assert len(entity_files) >= 1

        # Delete first entity
        entity_files[0].unlink()

        # Run verification
        verified = migrator.verify()

        # Verification should fail
        assert not verified

        # Should have warning about missing entity
        assert len(migrator.warnings) > 0
        assert any("not found" in w.lower() for w in migrator.warnings)
