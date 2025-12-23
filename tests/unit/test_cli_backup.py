"""
Unit tests for cortical.got.cli.backup module.

Tests all backup-related CLI commands with mocked manager.
"""

import pytest
import json
import gzip
from unittest.mock import MagicMock, patch, mock_open, call
from argparse import Namespace
from pathlib import Path

from cortical.got.cli.backup import (
    cmd_backup_create,
    cmd_backup_list,
    cmd_backup_verify,
    cmd_backup_restore,
    cmd_sync,
    cmd_migrate,
    cmd_migrate_events,
    handle_backup_command,
    handle_sync_migrate_commands,
)


class TestCmdBackupCreate:
    """Tests for cmd_backup_create function."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mock manager with WAL."""
        manager = MagicMock()
        manager.got_dir = Path("/fake/got")
        manager.wal = MagicMock()
        manager.graph = MagicMock()
        return manager

    def test_create_success(self, mock_manager, capsys):
        """Test backup create success."""
        mock_manager.wal.create_snapshot.return_value = "snap_20251223_120000_abc123"

        # Mock snapshot file
        with patch('cortical.got.cli.backup.Path.glob') as mock_glob:
            mock_file = MagicMock()
            mock_file.stat.return_value.st_size = 10240
            mock_glob.return_value = [mock_file]

            args = Namespace(compress=True)
            result = cmd_backup_create(args, mock_manager)

        assert result == 0
        mock_manager.wal.create_snapshot.assert_called_once_with(
            mock_manager.graph, compress=True
        )
        captured = capsys.readouterr()
        assert "Snapshot created" in captured.out
        assert "10.0 KB" in captured.out

    def test_create_uncompressed(self, mock_manager, capsys):
        """Test backup create without compression."""
        mock_manager.wal.create_snapshot.return_value = "snap_20251223_120000_abc123"

        with patch('cortical.got.cli.backup.Path.glob') as mock_glob:
            mock_file = MagicMock()
            mock_file.stat.return_value.st_size = 20480
            mock_glob.return_value = [mock_file]

            args = Namespace(compress=False)
            result = cmd_backup_create(args, mock_manager)

        assert result == 0
        mock_manager.wal.create_snapshot.assert_called_once_with(
            mock_manager.graph, compress=False
        )

    def test_create_error(self, mock_manager, capsys):
        """Test backup create with error."""
        mock_manager.wal.create_snapshot.side_effect = Exception("Snapshot failed")
        args = Namespace(compress=True)

        result = cmd_backup_create(args, mock_manager)

        assert result == 1
        captured = capsys.readouterr()
        assert "Error creating snapshot" in captured.out


class TestCmdBackupList:
    """Tests for cmd_backup_list function."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mock manager."""
        manager = MagicMock()
        manager.got_dir = Path("/fake/got")
        return manager

    def test_list_no_snapshots(self, mock_manager, capsys):
        """Test list with no snapshots."""
        with patch('cortical.got.cli.backup.Path.exists', return_value=False):
            args = Namespace(limit=10)
            result = cmd_backup_list(args, mock_manager)

        assert result == 0
        captured = capsys.readouterr()
        assert "No snapshots found" in captured.out

    def test_list_with_snapshots(self, mock_manager, capsys):
        """Test list with snapshots."""
        # Create mock snapshot files with sortable names
        mock_file1 = MagicMock()
        mock_file1.name = "snap_20251223_120000_abc123.json.gz"
        mock_file1.stem = "snap_20251223_120000_abc123.json"
        mock_file1.suffix = ".gz"
        mock_file1.stat.return_value.st_size = 10240
        mock_file1.__lt__ = lambda self, other: self.name < other.name
        mock_file1.__gt__ = lambda self, other: self.name > other.name

        mock_file2 = MagicMock()
        mock_file2.name = "snap_20251222_100000_def456.json"
        mock_file2.stem = "snap_20251222_100000_def456"
        mock_file2.suffix = ".json"
        mock_file2.stat.return_value.st_size = 20480
        mock_file2.__lt__ = lambda self, other: self.name < other.name
        mock_file2.__gt__ = lambda self, other: self.name > other.name

        snapshot_data1 = {"state": {"nodes": {"T-001": {}}, "edges": {}}}
        snapshot_data2 = {"state": {"nodes": {"T-001": {}, "T-002": {}}, "edges": {}}}

        with patch('cortical.got.cli.backup.Path.exists', return_value=True):
            with patch('cortical.got.cli.backup.Path.glob') as mock_glob:
                mock_glob.return_value = [mock_file1, mock_file2]

                # Mock json.load to return our data
                with patch('json.load') as mock_json_load:
                    mock_json_load.side_effect = [snapshot_data1, snapshot_data2]
                    with patch('cortical.got.cli.backup.gzip.open', mock_open()):
                        with patch('builtins.open', mock_open()):
                            args = Namespace(limit=10)
                            result = cmd_backup_list(args, mock_manager)

        assert result == 0
        captured = capsys.readouterr()
        assert "Available Snapshots" in captured.out

    def test_list_with_limit(self, mock_manager, capsys):
        """Test list with limit."""
        # Create many mock snapshot files with sortable names
        mock_files = []
        for i in range(15):
            mock_file = MagicMock()
            mock_file.name = f"snap_2025122{i % 10}_120000_abc{i:03d}.json"
            mock_file.stem = f"snap_2025122{i % 10}_120000_abc{i:03d}"
            mock_file.suffix = ".json"
            mock_file.stat.return_value.st_size = 10240
            # Make files sortable
            mock_file.__lt__ = lambda self, other: self.name < other.name
            mock_file.__gt__ = lambda self, other: self.name > other.name
            mock_files.append(mock_file)

        snapshot_data = {"state": {"nodes": {}, "edges": {}}}

        with patch('cortical.got.cli.backup.Path.exists', return_value=True):
            with patch('cortical.got.cli.backup.Path.glob') as mock_glob:
                mock_glob.return_value = mock_files

                with patch('json.load', return_value=snapshot_data):
                    with patch('builtins.open', mock_open()):
                        args = Namespace(limit=5)
                        result = cmd_backup_list(args, mock_manager)

        assert result == 0
        captured = capsys.readouterr()
        assert "... and 10 more" in captured.out


class TestCmdBackupVerify:
    """Tests for cmd_backup_verify function."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mock manager."""
        manager = MagicMock()
        manager.got_dir = Path("/fake/got")
        return manager

    def test_verify_no_snapshots(self, mock_manager, capsys):
        """Test verify with no snapshots directory."""
        with patch('cortical.got.cli.backup.Path.exists', return_value=False):
            args = Namespace(snapshot_id=None)
            result = cmd_backup_verify(args, mock_manager)

        assert result == 1
        captured = capsys.readouterr()
        assert "No snapshots found" in captured.out

    def test_verify_snapshot_not_found(self, mock_manager, capsys):
        """Test verify with snapshot not found."""
        with patch('cortical.got.cli.backup.Path.exists', return_value=True):
            with patch('cortical.got.cli.backup.Path.glob', return_value=[]):
                args = Namespace(snapshot_id="abc123")
                result = cmd_backup_verify(args, mock_manager)

        assert result == 1
        captured = capsys.readouterr()
        assert "Snapshot not found" in captured.out

    def test_verify_valid_snapshot(self, mock_manager, capsys):
        """Test verify with valid snapshot."""
        mock_file = MagicMock()
        mock_file.name = "snap_20251223_120000_abc123.json"
        mock_file.suffix = ".json"

        snapshot_data = {
            "snapshot_id": "snap_20251223_120000_abc123",
            "timestamp": "2025-12-23T12:00:00Z",
            "state": {
                "nodes": {
                    "T-001": {"node_type": "TASK", "content": "Task 1"},
                    "T-002": {"node_type": "TASK", "content": "Task 2"},
                },
                "edges": {},
            },
        }

        with patch('cortical.got.cli.backup.Path.exists', return_value=True):
            with patch('cortical.got.cli.backup.Path.glob', return_value=[mock_file]):
                with patch('builtins.open', mock_open(read_data=json.dumps(snapshot_data))):
                    args = Namespace(snapshot_id="abc123")
                    result = cmd_backup_verify(args, mock_manager)

        assert result == 0
        captured = capsys.readouterr()
        assert "Snapshot verification: PASSED" in captured.out
        assert "Nodes: 2" in captured.out

    def test_verify_compressed_snapshot(self, mock_manager, capsys):
        """Test verify with compressed snapshot."""
        mock_file = MagicMock()
        mock_file.name = "snap_20251223_120000_abc123.json.gz"
        mock_file.suffix = ".gz"

        snapshot_data = {
            "snapshot_id": "snap_20251223_120000_abc123",
            "timestamp": "2025-12-23T12:00:00Z",
            "state": {"nodes": {}, "edges": {}},
        }

        with patch('cortical.got.cli.backup.Path.exists', return_value=True):
            with patch('cortical.got.cli.backup.Path.glob', return_value=[mock_file]):
                with patch('cortical.got.cli.backup.gzip.open', mock_open(read_data=json.dumps(snapshot_data))):
                    args = Namespace(snapshot_id="abc123")
                    result = cmd_backup_verify(args, mock_manager)

        assert result == 0
        captured = capsys.readouterr()
        assert "PASSED" in captured.out

    def test_verify_invalid_json(self, mock_manager, capsys):
        """Test verify with invalid JSON."""
        mock_file = MagicMock()
        mock_file.name = "snap_20251223_120000_abc123.json"
        mock_file.suffix = ".json"

        with patch('cortical.got.cli.backup.Path.exists', return_value=True):
            with patch('cortical.got.cli.backup.Path.glob', return_value=[mock_file]):
                with patch('builtins.open', mock_open(read_data="invalid json")):
                    args = Namespace(snapshot_id="abc123")
                    result = cmd_backup_verify(args, mock_manager)

        assert result == 1
        captured = capsys.readouterr()
        assert "Invalid JSON" in captured.out

    def test_verify_missing_fields(self, mock_manager, capsys):
        """Test verify with missing required fields."""
        mock_file = MagicMock()
        mock_file.name = "snap_20251223_120000_abc123.json"
        mock_file.suffix = ".json"

        snapshot_data = {"snapshot_id": "abc123"}  # Missing timestamp and state

        with patch('cortical.got.cli.backup.Path.exists', return_value=True):
            with patch('cortical.got.cli.backup.Path.glob', return_value=[mock_file]):
                with patch('builtins.open', mock_open(read_data=json.dumps(snapshot_data))):
                    args = Namespace(snapshot_id="abc123")
                    result = cmd_backup_verify(args, mock_manager)

        assert result == 1
        captured = capsys.readouterr()
        assert "Missing fields" in captured.out


class TestCmdBackupRestore:
    """Tests for cmd_backup_restore function."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mock manager."""
        manager = MagicMock()
        manager.got_dir = Path("/fake/got")
        manager.graph = MagicMock()
        return manager

    def test_restore_snapshot_not_found(self, mock_manager, capsys):
        """Test restore with snapshot not found."""
        with patch('cortical.got.cli.backup.Path.exists', return_value=False):
            args = Namespace(snapshot_id="abc123", force=True)
            result = cmd_backup_restore(args, mock_manager)

        assert result == 1
        captured = capsys.readouterr()
        assert "No snapshots found" in captured.out

    def test_restore_success_forced(self, mock_manager, capsys):
        """Test restore success with force flag."""
        mock_file = MagicMock()
        mock_file.name = "snap_20251223_120000_abc123.json"
        mock_file.suffix = ".json"

        snapshot_data = {
            "state": {
                "nodes": {
                    "T-001": {
                        "node_type": "TASK",
                        "content": "Task 1",
                        "properties": {"status": "pending"},
                        "metadata": {"created_at": "2025-12-23"},
                    },
                },
                "edges": {
                    "E-001": {
                        "source_id": "T-001",
                        "target_id": "T-002",
                        "edge_type": "DEPENDS_ON",
                        "weight": 1.0,
                        "metadata": {},
                    },
                },
            },
        }

        mock_graph_instance = MagicMock()

        with patch('cortical.got.cli.backup.Path.exists', return_value=True):
            with patch('cortical.got.cli.backup.Path.glob', return_value=[mock_file]):
                with patch('cortical.reasoning.thought_graph.ThoughtGraph', return_value=mock_graph_instance):
                    with patch('cortical.reasoning.graph_of_thought.NodeType') as mock_node_type:
                        with patch('cortical.reasoning.graph_of_thought.EdgeType') as mock_edge_type:
                            with patch('json.load', return_value=snapshot_data):
                                with patch('builtins.open', mock_open()):
                                    args = Namespace(snapshot_id="abc123", force=True)
                                    result = cmd_backup_restore(args, mock_manager)

        assert result == 0
        mock_graph_instance.add_node.assert_called_once()
        captured = capsys.readouterr()
        assert "Restored from" in captured.out
        assert "Nodes: 1" in captured.out

    @patch('builtins.input', return_value='n')
    def test_restore_cancelled(self, mock_input, mock_manager, capsys):
        """Test restore cancelled by user."""
        mock_file = MagicMock()
        mock_file.name = "snap_20251223_120000_abc123.json"
        mock_file.suffix = ".json"

        with patch('cortical.got.cli.backup.Path.exists', return_value=True):
            with patch('cortical.got.cli.backup.Path.glob', return_value=[mock_file]):
                args = Namespace(snapshot_id="abc123", force=False)
                result = cmd_backup_restore(args, mock_manager)

        assert result == 0
        captured = capsys.readouterr()
        assert "Restore cancelled" in captured.out

    def test_restore_error(self, mock_manager, capsys):
        """Test restore with error."""
        mock_file = MagicMock()
        mock_file.name = "snap_20251223_120000_abc123.json"
        mock_file.suffix = ".json"

        with patch('cortical.got.cli.backup.Path.exists', return_value=True):
            with patch('cortical.got.cli.backup.Path.glob', return_value=[mock_file]):
                with patch('builtins.open', side_effect=Exception("Read failed")):
                    args = Namespace(snapshot_id="abc123", force=True)
                    result = cmd_backup_restore(args, mock_manager)

        assert result == 1
        captured = capsys.readouterr()
        assert "Error restoring" in captured.out


class TestCmdSync:
    """Tests for cmd_sync function."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mock manager."""
        manager = MagicMock()
        manager.snapshots_dir = Path("/fake/got/snapshots")
        return manager

    def test_sync_success(self, mock_manager, capsys):
        """Test sync command success."""
        mock_manager.sync_to_git.return_value = "snapshot_20251223_120000.json"
        mock_manager.get_stats.return_value = {
            "total_tasks": 10,
            "total_sprints": 3,
        }
        args = Namespace(message=None)

        result = cmd_sync(args, mock_manager)

        assert result == 0
        captured = capsys.readouterr()
        assert "Synced to git-tracked snapshot" in captured.out
        assert "Tasks: 10" in captured.out
        assert "Sprints: 3" in captured.out

    @patch('cortical.got.cli.backup.subprocess.run')
    def test_sync_with_commit(self, mock_run, mock_manager, capsys):
        """Test sync with auto-commit."""
        mock_manager.sync_to_git.return_value = "snapshot_20251223_120000.json"
        mock_manager.get_stats.return_value = {"total_tasks": 10, "total_sprints": 3}
        args = Namespace(message="Update GoT state")

        result = cmd_sync(args, mock_manager)

        assert result == 0
        assert mock_run.call_count == 2  # git add + git commit
        captured = capsys.readouterr()
        assert "Committed" in captured.out

    @patch('cortical.got.cli.backup.subprocess.run')
    def test_sync_commit_failed(self, mock_run, mock_manager, capsys):
        """Test sync with commit failure."""
        import subprocess

        mock_manager.sync_to_git.return_value = "snapshot_20251223_120000.json"
        mock_manager.get_stats.return_value = {"total_tasks": 10, "total_sprints": 3}
        mock_run.side_effect = subprocess.CalledProcessError(1, "git")
        args = Namespace(message="Update GoT state")

        result = cmd_sync(args, mock_manager)

        assert result == 0  # Sync still succeeds even if commit fails
        captured = capsys.readouterr()
        assert "Warning: Git commit failed" in captured.out

    def test_sync_error(self, mock_manager, capsys):
        """Test sync with error."""
        mock_manager.sync_to_git.side_effect = Exception("Sync failed")
        args = Namespace(message=None)

        result = cmd_sync(args, mock_manager)

        assert result == 1
        captured = capsys.readouterr()
        assert "Error syncing" in captured.out


class TestCmdMigrate:
    """Tests for cmd_migrate function."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mock manager."""
        manager = MagicMock()
        return manager

    def test_migrate_success(self, mock_manager, capsys):
        """Test migrate command success."""
        args = Namespace(dry_run=False)

        with patch('scripts.got_utils.TaskMigrator') as mock_migrator_class:
            mock_migrator = MagicMock()
            mock_migrator.migrate_all.return_value = {
                "sessions_processed": 5,
                "tasks_migrated": 10,
                "tasks_skipped": 2,
                "errors": [],
            }
            mock_migrator_class.return_value = mock_migrator

            result = cmd_migrate(args, mock_manager)

        assert result == 0
        captured = capsys.readouterr()
        assert "Sessions processed: 5" in captured.out
        assert "Tasks migrated: 10" in captured.out
        assert "Tasks skipped: 2" in captured.out

    def test_migrate_with_errors(self, mock_manager, capsys):
        """Test migrate with errors."""
        args = Namespace(dry_run=False)

        with patch('scripts.got_utils.TaskMigrator') as mock_migrator_class:
            mock_migrator = MagicMock()
            mock_migrator.migrate_all.return_value = {
                "sessions_processed": 5,
                "tasks_migrated": 8,
                "tasks_skipped": 4,
                "errors": ["Error 1", "Error 2", "Error 3"],
            }
            mock_migrator_class.return_value = mock_migrator

            result = cmd_migrate(args, mock_manager)

        assert result == 0
        captured = capsys.readouterr()
        assert "Errors:" in captured.out


class TestCmdMigrateEvents:
    """Tests for cmd_migrate_events function."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mock manager."""
        manager = MagicMock()
        manager.events_dir = Path("/fake/events")
        manager.graph = MagicMock()
        manager.graph.nodes = {
            "T-001": MagicMock(node_type=MagicMock(name="TASK"), content="Task 1", properties={"status": "pending"}),
        }
        manager.graph.edges = []
        return manager

    def test_migrate_events_dry_run(self, mock_manager, capsys):
        """Test migrate events in dry run mode."""
        args = Namespace(dry_run=True, force=False)

        with patch('scripts.got_utils.EventLog') as mock_event_log_class:
            mock_event_log_class.load_all_events.return_value = []
            result = cmd_migrate_events(args, mock_manager)

        assert result == 0
        captured = capsys.readouterr()
        assert "Dry run complete" in captured.out
        assert "Nodes: 1" in captured.out

    def test_migrate_events_success(self, mock_manager, capsys):
        """Test migrate events success."""
        args = Namespace(dry_run=False, force=False)

        with patch('scripts.got_utils.EventLog') as mock_event_log_class:
            mock_event_log_class.load_all_events.return_value = []
            mock_event_log_instance = MagicMock()
            mock_event_log_instance.event_file = "/fake/events/migration.jsonl"
            mock_event_log_class.return_value = mock_event_log_instance

            # No need to mock datetime - it's only used for session ID
            result = cmd_migrate_events(args, mock_manager)

        assert result == 0
        mock_event_log_instance.log_node_create.assert_called()
        captured = capsys.readouterr()
        assert "Migration complete" in captured.out

    def test_migrate_events_already_exist(self, mock_manager, capsys):
        """Test migrate events when events already exist."""
        args = Namespace(dry_run=False, force=False)

        with patch('scripts.got_utils.EventLog') as mock_event_log_class:
            mock_event_log_class.load_all_events.return_value = [{"event": "node.create"}]
            result = cmd_migrate_events(args, mock_manager)

        assert result == 1
        captured = capsys.readouterr()
        assert "Events already exist" in captured.out

    def test_migrate_events_force(self, mock_manager, capsys):
        """Test migrate events with force flag."""
        args = Namespace(dry_run=False, force=True)

        with patch('scripts.got_utils.EventLog') as mock_event_log_class:
            mock_event_log_class.load_all_events.return_value = [{"event": "node.create"}]
            mock_event_log_instance = MagicMock()
            mock_event_log_instance.event_file = "/fake/events/migration.jsonl"
            mock_event_log_class.return_value = mock_event_log_instance

            # No need to mock datetime - it's only used for session ID
            result = cmd_migrate_events(args, mock_manager)

        assert result == 0
        captured = capsys.readouterr()
        assert "Migration complete" in captured.out


class TestHandleBackupCommand:
    """Tests for handle_backup_command function."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mock manager."""
        return MagicMock()

    def test_handle_no_subcommand(self, mock_manager, capsys):
        """Test handle with no backup subcommand."""
        args = Namespace(command="backup")
        result = handle_backup_command(args, mock_manager)

        assert result == 1
        captured = capsys.readouterr()
        assert "No backup subcommand specified" in captured.out

    @patch('cortical.got.cli.backup.cmd_backup_create')
    def test_handle_create_command(self, mock_cmd, mock_manager):
        """Test handle create subcommand."""
        mock_cmd.return_value = 0
        args = Namespace(command="backup", backup_command="create")

        result = handle_backup_command(args, mock_manager)

        assert result == 0
        mock_cmd.assert_called_once_with(args, mock_manager)

    def test_handle_unknown_subcommand(self, mock_manager, capsys):
        """Test handle unknown subcommand."""
        args = Namespace(command="backup", backup_command="unknown")
        result = handle_backup_command(args, mock_manager)

        assert result == 1
        captured = capsys.readouterr()
        assert "Unknown backup subcommand" in captured.out


class TestHandleSyncMigrateCommands:
    """Tests for handle_sync_migrate_commands function."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mock manager."""
        return MagicMock()

    @patch('cortical.got.cli.backup.cmd_sync')
    def test_handle_sync_command(self, mock_cmd, mock_manager):
        """Test handle sync command."""
        mock_cmd.return_value = 0
        args = Namespace(command="sync")

        result = handle_sync_migrate_commands(args, mock_manager)

        assert result == 0
        mock_cmd.assert_called_once_with(args, mock_manager)

    @patch('cortical.got.cli.backup.cmd_migrate')
    def test_handle_migrate_command(self, mock_cmd, mock_manager):
        """Test handle migrate command."""
        mock_cmd.return_value = 0
        args = Namespace(command="migrate")

        result = handle_sync_migrate_commands(args, mock_manager)

        assert result == 0
        mock_cmd.assert_called_once_with(args, mock_manager)

    @patch('cortical.got.cli.backup.cmd_migrate_events')
    def test_handle_migrate_events_command(self, mock_cmd, mock_manager):
        """Test handle migrate-events command."""
        mock_cmd.return_value = 0
        args = Namespace(command="migrate-events")

        result = handle_sync_migrate_commands(args, mock_manager)

        assert result == 0
        mock_cmd.assert_called_once_with(args, mock_manager)

    def test_handle_unknown_command(self, mock_manager):
        """Test handle unknown command."""
        args = Namespace(command="unknown")

        result = handle_sync_migrate_commands(args, mock_manager)

        assert result is None  # Not handled
