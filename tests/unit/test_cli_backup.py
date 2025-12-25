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


# TestCmdMigrate removed - tests deprecated cmd_migrate function

# TestCmdMigrateEvents removed - tests deprecated cmd_migrate_events function
# The TX backend doesn't use event logs; see TransactionalGoTAdapter


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

    # test_handle_migrate_command removed - tests deprecated cmd_migrate function
    # test_handle_migrate_events_command removed - tests deprecated cmd_migrate_events function

    def test_handle_unknown_command(self, mock_manager):
        """Test handle unknown command."""
        args = Namespace(command="unknown")

        result = handle_sync_migrate_commands(args, mock_manager)

        assert result is None  # Not handled


class TestCmdBackupCreateRealExecution:
    """Tests that actually execute cmd_backup_create code."""

    @pytest.fixture
    def mock_manager_with_real_paths(self, tmp_path):
        """Create a mock manager with real paths."""
        manager = MagicMock()
        manager.got_dir = tmp_path
        manager.wal = MagicMock()
        manager.graph = MagicMock()

        # Create snapshots directory
        snapshots_dir = tmp_path / "wal" / "snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)

        return manager

    def test_create_with_file_not_found(self, mock_manager_with_real_paths, capsys):
        """Test when snapshot file cannot be found after creation."""
        mock_manager_with_real_paths.wal.create_snapshot.return_value = "snap_20251223_120000_notfound"

        args = Namespace(compress=True)
        result = cmd_backup_create(args, mock_manager_with_real_paths)

        # Should succeed even if file not found (snapshot was created)
        assert result == 0
        captured = capsys.readouterr()
        assert "Snapshot created" in captured.out

    def test_create_with_actual_snapshot_file(self, mock_manager_with_real_paths, capsys):
        """Test with actual snapshot file on disk."""
        snapshot_id = "snap_20251223_120000_abc123"
        mock_manager_with_real_paths.wal.create_snapshot.return_value = snapshot_id

        # Create actual snapshot file
        snapshots_dir = mock_manager_with_real_paths.got_dir / "wal" / "snapshots"
        snap_file = snapshots_dir / f"{snapshot_id}.json.gz"
        snap_file.write_text("test data")

        args = Namespace(compress=True)
        result = cmd_backup_create(args, mock_manager_with_real_paths)

        assert result == 0
        captured = capsys.readouterr()
        assert "Snapshot created" in captured.out
        assert "KB" in captured.out  # Size should be shown
        assert "Compressed: True" in captured.out


class TestCmdBackupListRealExecution:
    """Tests that actually execute cmd_backup_list code."""

    @pytest.fixture
    def manager_with_snapshots(self, tmp_path):
        """Create a manager with real snapshot files."""
        manager = MagicMock()
        manager.got_dir = tmp_path

        snapshots_dir = tmp_path / "wal" / "snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)

        # Create test snapshot files
        snap1 = snapshots_dir / "snap_20251223_120000_abc123.json.gz"
        snap2 = snapshots_dir / "snap_20251222_100000_def456.json"
        snap3 = snapshots_dir / "snap_20251221_080000_ghi789.json"

        # Write valid JSON to each
        import gzip
        data1 = {"state": {"nodes": {"T-001": {}}, "edges": {}}}
        with gzip.open(snap1, 'wt') as f:
            json.dump(data1, f)

        data2 = {"state": {"nodes": {"T-001": {}, "T-002": {}}, "edges": {}}}
        snap2.write_text(json.dumps(data2))

        data3 = {"state": {"nodes": {}, "edges": {}}}
        snap3.write_text(json.dumps(data3))

        return manager

    def test_list_with_real_files(self, manager_with_snapshots, capsys):
        """Test list with real snapshot files."""
        args = Namespace(limit=10)
        result = cmd_backup_list(args, manager_with_snapshots)

        assert result == 0
        captured = capsys.readouterr()
        assert "Available Snapshots" in captured.out
        assert "2025-12-23" in captured.out  # Timestamp parsing
        assert "2025-12-22" in captured.out

    def test_list_with_limit_real_files(self, manager_with_snapshots, capsys):
        """Test list with limit on real files."""
        args = Namespace(limit=2)
        result = cmd_backup_list(args, manager_with_snapshots)

        assert result == 0
        captured = capsys.readouterr()
        assert "... and 1 more" in captured.out

    def test_list_invalid_snapshot_skipped(self, tmp_path, capsys):
        """Test that invalid snapshots are skipped."""
        manager = MagicMock()
        manager.got_dir = tmp_path

        snapshots_dir = tmp_path / "wal" / "snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)

        # Create invalid snapshot (bad JSON)
        snap1 = snapshots_dir / "snap_20251223_120000_bad.json"
        snap1.write_text("not valid json")

        # Create valid snapshot
        snap2 = snapshots_dir / "snap_20251222_100000_good.json"
        snap2.write_text(json.dumps({"state": {"nodes": {}, "edges": {}}}))

        args = Namespace(limit=10)
        result = cmd_backup_list(args, manager)

        assert result == 0
        captured = capsys.readouterr()
        # Should show only the valid snapshot
        assert "2025-12-22" in captured.out

    def test_list_malformed_filename(self, tmp_path, capsys):
        """Test handling of malformed snapshot filenames."""
        manager = MagicMock()
        manager.got_dir = tmp_path

        snapshots_dir = tmp_path / "wal" / "snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)

        # Create snapshot with malformed name
        snap1 = snapshots_dir / "snap_bad.json"
        snap1.write_text(json.dumps({"state": {"nodes": {}, "edges": {}}}))

        args = Namespace(limit=10)
        result = cmd_backup_list(args, manager)

        assert result == 0
        captured = capsys.readouterr()
        # Should still list it with "unknown" timestamp
        assert "unknown" in captured.out or "snap_bad.json" in captured.out


class TestCmdBackupVerifyRealExecution:
    """Tests that actually execute cmd_backup_verify code."""

    @pytest.fixture
    def manager_with_snapshot(self, tmp_path):
        """Create manager with a real snapshot."""
        manager = MagicMock()
        manager.got_dir = tmp_path

        snapshots_dir = tmp_path / "wal" / "snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)

        # Create valid snapshot
        snap_file = snapshots_dir / "snap_20251223_120000_abc123.json"
        data = {
            "snapshot_id": "snap_20251223_120000_abc123",
            "timestamp": "2025-12-23T12:00:00Z",
            "state": {
                "nodes": {
                    "T-001": {"node_type": "TASK", "content": "Task 1"},
                },
                "edges": {},
            },
        }
        snap_file.write_text(json.dumps(data))

        return manager

    def test_verify_with_real_file(self, manager_with_snapshot, capsys):
        """Test verify with real snapshot file."""
        args = Namespace(snapshot_id="abc123")
        result = cmd_backup_verify(args, manager_with_snapshot)

        assert result == 0
        captured = capsys.readouterr()
        assert "PASSED" in captured.out
        assert "Nodes: 1" in captured.out

    def test_verify_latest_without_id(self, manager_with_snapshot, capsys):
        """Test verify without snapshot_id (uses latest)."""
        args = Namespace(snapshot_id=None)
        result = cmd_backup_verify(args, manager_with_snapshot)

        assert result == 0
        captured = capsys.readouterr()
        assert "PASSED" in captured.out

    def test_verify_invalid_nodes(self, tmp_path, capsys):
        """Test verify with invalid node structures."""
        manager = MagicMock()
        manager.got_dir = tmp_path

        snapshots_dir = tmp_path / "wal" / "snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)

        snap_file = snapshots_dir / "snap_20251223_120000_bad.json"
        data = {
            "snapshot_id": "snap_20251223_120000_bad",
            "timestamp": "2025-12-23T12:00:00Z",
            "state": {
                "nodes": {
                    "T-001": {"node_type": "TASK"},  # Valid
                    "T-002": "invalid",  # Invalid - not a dict
                    "T-003": {},  # Invalid - missing node_type
                },
                "edges": {},
            },
        }
        snap_file.write_text(json.dumps(data))

        args = Namespace(snapshot_id="bad")
        result = cmd_backup_verify(args, manager)

        assert result == 0  # Still passes, but warns
        captured = capsys.readouterr()
        assert "Invalid nodes: 2" in captured.out

    def test_verify_compressed_real_file(self, tmp_path, capsys):
        """Test verify with real compressed file."""
        manager = MagicMock()
        manager.got_dir = tmp_path

        snapshots_dir = tmp_path / "wal" / "snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)

        snap_file = snapshots_dir / "snap_20251223_120000_comp.json.gz"
        data = {
            "snapshot_id": "snap_20251223_120000_comp",
            "timestamp": "2025-12-23T12:00:00Z",
            "state": {"nodes": {}, "edges": {}},
        }

        with gzip.open(snap_file, 'wt') as f:
            json.dump(data, f)

        args = Namespace(snapshot_id="comp")
        result = cmd_backup_verify(args, manager)

        assert result == 0
        captured = capsys.readouterr()
        assert "PASSED" in captured.out


class TestCmdBackupRestoreRealExecution:
    """Tests that actually execute cmd_backup_restore code."""

    @pytest.fixture
    def manager_with_restore_snapshot(self, tmp_path):
        """Create manager with snapshot for restoration."""
        manager = MagicMock()
        manager.got_dir = tmp_path
        manager.graph = MagicMock()

        snapshots_dir = tmp_path / "wal" / "snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)

        snap_file = snapshots_dir / "snap_20251223_120000_restore.json"
        data = {
            "state": {
                "nodes": {
                    "T-001": {
                        "node_type": "TASK",
                        "content": "Task 1",
                        "properties": {"status": "pending"},
                        "metadata": {},
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
        snap_file.write_text(json.dumps(data))

        return manager

    @patch('builtins.input', return_value='y')
    @patch('cortical.reasoning.thought_graph.ThoughtGraph')
    @patch('cortical.reasoning.graph_of_thought.NodeType')
    @patch('cortical.reasoning.graph_of_thought.EdgeType')
    def test_restore_confirmed(self, mock_edge_type, mock_node_type, mock_graph_class,
                                mock_input, manager_with_restore_snapshot, capsys):
        """Test restore with user confirmation."""
        mock_graph_instance = MagicMock()
        mock_graph_class.return_value = mock_graph_instance

        # Mock NodeType and EdgeType
        mock_node_type.__getitem__ = MagicMock(return_value="TASK")
        mock_edge_type.__getitem__ = MagicMock(return_value="DEPENDS_ON")

        args = Namespace(snapshot_id="restore", force=False)
        result = cmd_backup_restore(args, manager_with_restore_snapshot)

        assert result == 0
        mock_graph_instance.add_node.assert_called_once()
        captured = capsys.readouterr()
        assert "Restored from" in captured.out

    @patch('cortical.reasoning.thought_graph.ThoughtGraph')
    @patch('cortical.reasoning.graph_of_thought.NodeType')
    @patch('cortical.reasoning.graph_of_thought.EdgeType')
    def test_restore_skip_invalid_edges(self, mock_edge_type, mock_node_type,
                                         mock_graph_class, tmp_path, capsys):
        """Test restore skips invalid edges."""
        manager = MagicMock()
        manager.got_dir = tmp_path
        manager.graph = MagicMock()

        snapshots_dir = tmp_path / "wal" / "snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)

        snap_file = snapshots_dir / "snap_20251223_120000_edges.json"
        data = {
            "state": {
                "nodes": {},
                "edges": {
                    "E-001": {
                        "source_id": "T-001",
                        "target_id": "T-002",
                        "edge_type": "INVALID_TYPE",  # Will cause error
                        "weight": 1.0,
                    },
                },
            },
        }
        snap_file.write_text(json.dumps(data))

        mock_graph_instance = MagicMock()
        mock_graph_instance.add_edge.side_effect = Exception("Invalid edge type")
        mock_graph_class.return_value = mock_graph_instance

        mock_node_type.__getitem__ = MagicMock(return_value="TASK")
        mock_edge_type.__getitem__ = MagicMock(side_effect=KeyError("INVALID_TYPE"))

        args = Namespace(snapshot_id="edges", force=True)
        result = cmd_backup_restore(args, manager)

        # Should still succeed (invalid edges are skipped)
        assert result == 0
        captured = capsys.readouterr()
        assert "Restored from" in captured.out

    @patch('cortical.reasoning.thought_graph.ThoughtGraph')
    @patch('cortical.reasoning.graph_of_thought.NodeType')
    def test_restore_compressed_snapshot(self, mock_node_type, mock_graph_class,
                                          tmp_path, capsys):
        """Test restore from compressed snapshot."""
        manager = MagicMock()
        manager.got_dir = tmp_path
        manager.graph = MagicMock()

        snapshots_dir = tmp_path / "wal" / "snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)

        snap_file = snapshots_dir / "snap_20251223_120000_comp.json.gz"
        data = {
            "state": {
                "nodes": {
                    "T-001": {
                        "node_type": "TASK",
                        "content": "Task 1",
                        "properties": {},
                        "metadata": {},
                    },
                },
                "edges": {},
            },
        }

        with gzip.open(snap_file, 'wt') as f:
            json.dump(data, f)

        mock_graph_instance = MagicMock()
        mock_graph_class.return_value = mock_graph_instance
        mock_node_type.__getitem__ = MagicMock(return_value="TASK")

        args = Namespace(snapshot_id="comp", force=True)
        result = cmd_backup_restore(args, manager)

        assert result == 0
        captured = capsys.readouterr()
        assert "Restored from" in captured.out


class TestCmdSyncRealExecution:
    """Tests that actually execute cmd_sync code."""

    @pytest.fixture
    def manager_with_snapshots_dir(self, tmp_path):
        """Create manager with snapshots directory."""
        manager = MagicMock()
        manager.snapshots_dir = tmp_path / "snapshots"
        manager.snapshots_dir.mkdir(parents=True, exist_ok=True)
        manager.sync_to_git.return_value = "snapshot_20251223_120000.json"
        manager.get_stats.return_value = {
            "total_tasks": 10,
            "total_sprints": 3,
        }
        return manager

    def test_sync_without_message(self, manager_with_snapshots_dir, capsys):
        """Test sync without commit message."""
        args = Namespace(message=None)
        result = cmd_sync(args, manager_with_snapshots_dir)

        assert result == 0
        captured = capsys.readouterr()
        assert "Synced to git-tracked snapshot" in captured.out
        assert "To persist across environments" in captured.out

    @patch('cortical.got.cli.backup.subprocess.run')
    def test_sync_subprocess_error_details(self, mock_run, manager_with_snapshots_dir, capsys):
        """Test sync with subprocess error showing details."""
        import subprocess

        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd="git",
            stderr=b"Permission denied"
        )

        args = Namespace(message="Test commit")
        result = cmd_sync(args, manager_with_snapshots_dir)

        assert result == 0  # Sync still succeeds
        captured = capsys.readouterr()
        assert "Warning: Git commit failed" in captured.out


# TestCmdMigrateEventsRealExecution removed - tests deprecated cmd_migrate_events function
# The TX backend doesn't use event logs; see TransactionalGoTAdapter


# TestCmdMigrateEdgeCases removed - tests deprecated cmd_migrate function
# The TX backend uses direct entity storage; see TransactionalGoTAdapter
