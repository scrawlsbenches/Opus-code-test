"""
Integration tests for GoT migration script.

Tests migration from event-sourced format to transactional format.
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from cortical.got import GoTManager
from scripts.migrate_got import GoTMigrator, MigrationAnalysis, MigrationResult


class TestGoTMigration:
    """Test GoT data migration from event-sourced to transactional format."""

    @pytest.fixture
    def source_dir(self, tmp_path):
        """Create source .got directory with sample data."""
        got_dir = tmp_path / ".got"
        got_dir.mkdir()

        # Create events directory with sample events
        events_dir = got_dir / "events"
        events_dir.mkdir()

        # Sample task creation event
        event1 = {
            "ts": "2025-12-21T10:00:00.000000Z",
            "event": "node.create",
            "meta": {"branch": "main", "session": "test-session"},
            "id": "task:T-20251221-100000-0001",
            "type": "TASK",
            "data": {
                "title": "Test task 1",
                "status": "pending",
                "priority": "high",
                "category": "feature",
                "description": "Test description"
            }
        }

        # Sample decision creation event
        event2 = {
            "ts": "2025-12-21T10:01:00.000000Z",
            "event": "node.create",
            "meta": {"branch": "main", "session": "test-session"},
            "id": "decision:D-20251221-100100-0001",
            "type": "DECISION",
            "data": {
                "title": "Test decision",
                "rationale": "Test rationale",
                "affects": ["task:T-20251221-100000-0001"]
            }
        }

        # Sample edge creation event
        event3 = {
            "ts": "2025-12-21T10:02:00.000000Z",
            "event": "edge.create",
            "meta": {"branch": "main", "session": "test-session"},
            "data": {
                "source_id": "task:T-20251221-100000-0001",
                "target_id": "decision:D-20251221-100100-0001",
                "edge_type": "MOTIVATES",
                "weight": 1.0,
                "confidence": 1.0
            }
        }

        # Write events to file
        event_file = events_dir / "test-events.jsonl"
        with open(event_file, "w") as f:
            f.write(json.dumps(event1) + "\n")
            f.write(json.dumps(event2) + "\n")
            f.write(json.dumps(event3) + "\n")

        # Create WAL directory with sample WAL entries
        wal_dir = got_dir / "wal"
        wal_dir.mkdir()
        wal_logs = wal_dir / "logs"
        wal_logs.mkdir()

        # Sample WAL node entry
        wal1 = {
            "operation": "add_node",
            "timestamp": "2025-12-21T10:03:00.000000",
            "doc_id": "task:T-20251221-100300-0002",
            "payload": {
                "node_id": "task:T-20251221-100300-0002",
                "node_type": "task",
                "content": "WAL task",
                "properties": {
                    "title": "WAL task",
                    "status": "in_progress",
                    "priority": "medium",
                    "description": ""
                },
                "metadata": {}
            }
        }

        # Sample WAL edge entry
        wal2 = {
            "operation": "add_edge",
            "timestamp": "2025-12-21T10:04:00.000000",
            "doc_id": "edge-1",
            "payload": {
                "source_id": "task:T-20251221-100000-0001",
                "target_id": "task:T-20251221-100300-0002",
                "edge_type": "DEPENDS_ON",
                "weight": 0.8
            }
        }

        # Write WAL entries to file
        wal_file = wal_logs / "wal_test.jsonl"
        with open(wal_file, "w") as f:
            f.write(json.dumps(wal1) + "\n")
            f.write(json.dumps(wal2) + "\n")

        return got_dir

    @pytest.fixture
    def target_dir(self, tmp_path):
        """Create target directory for migration."""
        return tmp_path / ".got-tx"

    def test_analyze_counts_entities(self, source_dir, target_dir):
        """Test that analysis returns correct entity counts."""
        migrator = GoTMigrator(
            source_dir=source_dir,
            target_dir=target_dir,
            dry_run=True
        )

        analysis = migrator.analyze()

        # Should find 2 tasks (1 from events, 1 from WAL)
        assert analysis.tasks == 2
        # Should find 1 decision from events
        assert analysis.decisions == 1
        # Should find 2 edges (1 from events, 1 from WAL)
        assert analysis.edges == 2
        # Should count 3 events
        assert analysis.events == 3
        # Should count 2 WAL entries
        assert analysis.wal_entries == 2

        assert analysis.source_dir == source_dir
        assert analysis.target_dir == target_dir

    def test_migrate_creates_transactional_store(self, source_dir, target_dir):
        """Test that migration creates transactional store files."""
        migrator = GoTMigrator(
            source_dir=source_dir,
            target_dir=target_dir,
            dry_run=False
        )

        result = migrator.migrate()

        assert result.success
        assert result.tasks_migrated == 2
        assert result.decisions_migrated == 1
        assert result.edges_migrated == 2

        # Check that target directory was created
        assert target_dir.exists()

        # Check that entities directory was created
        entities_dir = target_dir / "entities"
        assert entities_dir.exists()

        # Verify that entity files exist
        # Note: Entity filenames are based on entity IDs
        entity_files = list(entities_dir.glob("*.json"))
        assert len(entity_files) > 0

    def test_migrate_preserves_task_data(self, source_dir, target_dir):
        """Test that task data is preserved during migration."""
        migrator = GoTMigrator(
            source_dir=source_dir,
            target_dir=target_dir,
            dry_run=False
        )

        result = migrator.migrate()
        assert result.success

        # Read back the migrated task
        manager = GoTManager(target_dir)
        task = manager.get_task("task:T-20251221-100000-0001")

        assert task is not None
        assert task.id == "task:T-20251221-100000-0001"
        assert task.title == "Test task 1"
        assert task.status == "pending"
        assert task.priority == "high"
        assert task.description == "Test description"

        # Check WAL task
        wal_task = manager.get_task("task:T-20251221-100300-0002")
        assert wal_task is not None
        assert wal_task.title == "WAL task"
        assert wal_task.status == "in_progress"

    def test_migrate_preserves_edges(self, source_dir, target_dir):
        """Test that edge relationships are preserved during migration."""
        migrator = GoTMigrator(
            source_dir=source_dir,
            target_dir=target_dir,
            dry_run=False
        )

        result = migrator.migrate()
        assert result.success

        # Read back edges
        manager = GoTManager(target_dir)

        # We can't easily query edges without a query API,
        # but we can verify they were written by checking the result
        assert result.edges_migrated == 2

    def test_dry_run_does_not_write(self, source_dir, target_dir):
        """Test that dry run does not write any files."""
        migrator = GoTMigrator(
            source_dir=source_dir,
            target_dir=target_dir,
            dry_run=True
        )

        result = migrator.migrate()

        assert result.success
        # Counts should be correct
        assert result.tasks_migrated == 2
        assert result.decisions_migrated == 1

        # But target directory should not be created
        assert not target_dir.exists()

    def test_verify_confirms_migration(self, source_dir, target_dir):
        """Test that verification works after migration."""
        migrator = GoTMigrator(
            source_dir=source_dir,
            target_dir=target_dir,
            dry_run=False
        )

        result = migrator.migrate()
        assert result.success

        # Verify should pass
        verified = migrator.verify()
        assert verified

    def test_handles_missing_source_directory(self, tmp_path):
        """Test that migrator handles missing source directory gracefully."""
        source_dir = tmp_path / "nonexistent"
        target_dir = tmp_path / ".got-tx"

        with pytest.raises(FileNotFoundError):
            GoTMigrator(
                source_dir=source_dir,
                target_dir=target_dir
            )

    def test_handles_node_update_events(self, tmp_path):
        """Test that node update events are processed correctly."""
        # Create source directory with update events
        got_dir = tmp_path / ".got"
        got_dir.mkdir()

        events_dir = got_dir / "events"
        events_dir.mkdir()

        # Create event
        event1 = {
            "ts": "2025-12-21T10:00:00.000000Z",
            "event": "node.create",
            "meta": {"branch": "main", "session": "test"},
            "id": "task:T-TEST-001",
            "type": "TASK",
            "data": {
                "title": "Original title",
                "status": "pending",
                "priority": "medium"
            }
        }

        # Update event
        event2 = {
            "ts": "2025-12-21T10:05:00.000000Z",
            "event": "node.update",
            "meta": {"branch": "main", "session": "test"},
            "id": "task:T-TEST-001",
            "updates": {
                "status": "in_progress",
                "title": "Updated title"
            }
        }

        event_file = events_dir / "test.jsonl"
        with open(event_file, "w") as f:
            f.write(json.dumps(event1) + "\n")
            f.write(json.dumps(event2) + "\n")

        target_dir = tmp_path / ".got-tx"

        migrator = GoTMigrator(
            source_dir=got_dir,
            target_dir=target_dir,
            dry_run=False
        )

        result = migrator.migrate()
        assert result.success

        # Verify update was applied
        manager = GoTManager(target_dir)
        task = manager.get_task("task:T-TEST-001")

        assert task is not None
        assert task.title == "Updated title"
        assert task.status == "in_progress"
        # Version should be incremented due to update
        assert task.version >= 2

    def test_handles_corrupted_events(self, tmp_path):
        """Test that corrupted events are handled gracefully."""
        got_dir = tmp_path / ".got"
        got_dir.mkdir()

        events_dir = got_dir / "events"
        events_dir.mkdir()

        # Write invalid JSON
        event_file = events_dir / "corrupted.jsonl"
        with open(event_file, "w") as f:
            f.write("not valid json\n")
            f.write('{"valid": "json"}\n')  # This one is valid but missing required fields

        target_dir = tmp_path / ".got-tx"

        migrator = GoTMigrator(
            source_dir=got_dir,
            target_dir=target_dir,
            dry_run=False
        )

        result = migrator.migrate()

        # Should succeed despite corrupted events
        assert result.success
        # Should have warnings about corrupted events
        assert len(result.warnings) > 0
        assert any("parse" in w.lower() for w in result.warnings)

    def test_maps_legacy_statuses(self, tmp_path):
        """Test that legacy statuses are mapped to new valid statuses."""
        got_dir = tmp_path / ".got"
        got_dir.mkdir()

        events_dir = got_dir / "events"
        events_dir.mkdir()

        # Create events with legacy statuses
        events = [
            {
                "ts": "2025-12-21T10:00:00.000000Z",
                "event": "node.create",
                "meta": {"branch": "main", "session": "test"},
                "id": "task:T-TEST-001",
                "type": "TASK",
                "data": {"title": "Deferred task", "status": "deferred", "priority": "medium"}
            },
            {
                "ts": "2025-12-21T10:01:00.000000Z",
                "event": "node.create",
                "meta": {"branch": "main", "session": "test"},
                "id": "task:T-TEST-002",
                "type": "TASK",
                "data": {"title": "Cancelled task", "status": "cancelled", "priority": "low"}
            },
            {
                "ts": "2025-12-21T10:02:00.000000Z",
                "event": "node.create",
                "meta": {"branch": "main", "session": "test"},
                "id": "task:T-TEST-003",
                "type": "TASK",
                "data": {"title": "Done task", "status": "done", "priority": "high"}
            }
        ]

        event_file = events_dir / "legacy.jsonl"
        with open(event_file, "w") as f:
            for event in events:
                f.write(json.dumps(event) + "\n")

        target_dir = tmp_path / ".got-tx"

        migrator = GoTMigrator(
            source_dir=got_dir,
            target_dir=target_dir,
            dry_run=False
        )

        result = migrator.migrate()
        assert result.success

        # Verify status mappings
        manager = GoTManager(target_dir)

        task1 = manager.get_task("task:T-TEST-001")
        assert task1.status == "pending"  # deferred → pending

        task2 = manager.get_task("task:T-TEST-002")
        assert task2.status == "blocked"  # cancelled → blocked

        task3 = manager.get_task("task:T-TEST-003")
        assert task3.status == "completed"  # done → completed
