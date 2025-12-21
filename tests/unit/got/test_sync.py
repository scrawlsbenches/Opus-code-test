"""
Tests for GoT sync manager.
"""

import subprocess
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from cortical.got.sync import (
    SyncManager,
    SyncResult,
    SyncStatus,
)
from cortical.got.errors import SyncError


@pytest.fixture
def temp_git_repo(tmp_path):
    """Create a temporary git repository for testing."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()

    got_dir = repo_dir / ".got"
    got_dir.mkdir()

    # Initialize git repo
    subprocess.run(
        ["git", "init"],
        cwd=repo_dir,
        capture_output=True,
        check=True
    )

    # Configure git
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_dir,
        capture_output=True,
        check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_dir,
        capture_output=True,
        check=True
    )

    # Initial commit (disable hooks and signing for tests)
    (repo_dir / "README.md").write_text("# Test Repo")
    subprocess.run(
        ["git", "add", "README.md"],
        cwd=repo_dir,
        capture_output=True,
        check=True
    )
    subprocess.run(
        ["git", "commit", "-m", "Initial commit", "--no-gpg-sign", "--no-verify"],
        cwd=repo_dir,
        capture_output=True,
        check=True
    )

    return repo_dir, got_dir


class TestSyncResult:
    """Test SyncResult dataclass."""

    def test_sync_result_success(self):
        """Test successful sync result."""
        result = SyncResult(
            success=True,
            action="push",
            version="abc123"
        )

        assert result.success is True
        assert result.action == "push"
        assert result.version == "abc123"
        assert result.conflicts == []
        assert result.error is None

    def test_sync_result_failure(self):
        """Test failed sync result."""
        result = SyncResult(
            success=False,
            action="pull",
            error="Merge conflict"
        )

        assert result.success is False
        assert result.action == "pull"
        assert result.error == "Merge conflict"


class TestSyncStatus:
    """Test SyncStatus dataclass."""

    def test_sync_status_clean(self):
        """Test clean sync status."""
        status = SyncStatus(
            ahead=0,
            behind=0,
            dirty=False,
            has_active_tx=False
        )

        assert status.ahead == 0
        assert status.behind == 0
        assert status.dirty is False
        assert status.has_active_tx is False

    def test_sync_status_dirty(self):
        """Test dirty sync status."""
        status = SyncStatus(
            ahead=2,
            behind=1,
            dirty=True,
            has_active_tx=True
        )

        assert status.ahead == 2
        assert status.behind == 1
        assert status.dirty is True
        assert status.has_active_tx is True


class TestSyncManager:
    """Test SyncManager class."""

    def test_init_finds_git_root(self, temp_git_repo):
        """Test sync manager finds git root."""
        repo_dir, got_dir = temp_git_repo

        manager = SyncManager(got_dir)

        assert manager.got_dir == got_dir
        assert manager.git_dir == repo_dir

    def test_init_explicit_git_dir(self, temp_git_repo):
        """Test sync manager with explicit git dir."""
        repo_dir, got_dir = temp_git_repo

        manager = SyncManager(got_dir, git_dir=repo_dir)

        assert manager.git_dir == repo_dir

    def test_init_not_git_repo(self, tmp_path):
        """Test sync manager fails when not in git repo."""
        not_a_repo = tmp_path / "not_a_repo"
        not_a_repo.mkdir()

        got_dir = not_a_repo / ".got"
        got_dir.mkdir()

        with pytest.raises(SyncError) as exc_info:
            SyncManager(got_dir)

        assert "Not in a git repository" in str(exc_info.value)

    def test_can_sync_true_when_no_transactions(self, temp_git_repo):
        """Test can_sync returns True when no active transactions."""
        repo_dir, got_dir = temp_git_repo

        manager = SyncManager(got_dir)

        # No WAL directory means no transactions
        assert manager.can_sync() is True

    def test_can_sync_true_with_empty_wal(self, temp_git_repo):
        """Test can_sync returns True with empty WAL directory."""
        repo_dir, got_dir = temp_git_repo

        # Create empty WAL directory
        wal_dir = got_dir / "wal"
        wal_dir.mkdir()

        manager = SyncManager(got_dir)

        assert manager.can_sync() is True

    @patch('subprocess.run')
    def test_push_fails_with_active_transaction(self, mock_run, temp_git_repo):
        """Test push fails when active transactions exist."""
        repo_dir, got_dir = temp_git_repo

        manager = SyncManager(got_dir)

        # Mock can_sync to return False (active tx)
        with patch.object(manager, 'can_sync', return_value=False):
            result = manager.push()

        assert result.success is False
        assert result.action == "push"
        assert "Active transactions" in result.error

    @patch('subprocess.run')
    def test_pull_fails_with_active_transaction(self, mock_run, temp_git_repo):
        """Test pull fails when active transactions exist."""
        repo_dir, got_dir = temp_git_repo

        manager = SyncManager(got_dir)

        # Mock can_sync to return False (active tx)
        with patch.object(manager, 'can_sync', return_value=False):
            result = manager.pull()

        assert result.success is False
        assert result.action == "pull"
        assert "Active transactions" in result.error

    @patch('subprocess.run')
    def test_push_fails_when_behind(self, mock_run, temp_git_repo):
        """Test push fails when behind remote."""
        repo_dir, got_dir = temp_git_repo

        manager = SyncManager(got_dir)

        # Mock successful fetch
        mock_run.return_value = Mock(stdout="", returncode=0)

        # Mock status showing we're behind
        mock_status = SyncStatus(ahead=1, behind=2, dirty=False, has_active_tx=False)
        with patch.object(manager, 'get_status', return_value=mock_status):
            result = manager.push()

        assert result.success is False
        assert result.action == "push"
        assert "pull first" in result.error

    @patch('subprocess.run')
    def test_push_returns_none_when_nothing_to_push(self, mock_run, temp_git_repo):
        """Test push returns 'none' action when up to date."""
        repo_dir, got_dir = temp_git_repo

        manager = SyncManager(got_dir)

        # Mock successful fetch
        mock_run.return_value = Mock(stdout="", returncode=0)

        # Mock status showing nothing to push
        mock_status = SyncStatus(ahead=0, behind=0, dirty=False, has_active_tx=False)
        with patch.object(manager, 'get_status', return_value=mock_status):
            with patch.object(manager, '_get_current_commit', return_value="abc123"):
                result = manager.push()

        assert result.success is True
        assert result.action == "none"
        assert result.version == "abc123"

    @patch('subprocess.run')
    def test_pull_returns_none_when_up_to_date(self, mock_run, temp_git_repo):
        """Test pull returns 'none' action when up to date."""
        repo_dir, got_dir = temp_git_repo

        manager = SyncManager(got_dir)

        # Mock successful fetch
        mock_run.return_value = Mock(stdout="", returncode=0)

        # Mock status showing nothing to pull
        mock_status = SyncStatus(ahead=0, behind=0, dirty=False, has_active_tx=False)
        with patch.object(manager, 'get_status', return_value=mock_status):
            with patch.object(manager, '_get_current_commit', return_value="abc123"):
                result = manager.pull()

        assert result.success is True
        assert result.action == "none"
        assert result.version == "abc123"

    def test_get_status_returns_correct_state(self, temp_git_repo):
        """Test get_status returns accurate status."""
        repo_dir, got_dir = temp_git_repo

        manager = SyncManager(got_dir)

        status = manager.get_status()

        # Should be clean in a new repo
        assert isinstance(status, SyncStatus)
        assert status.dirty is False
        assert status.has_active_tx is False

    def test_get_status_detects_dirty_state(self, temp_git_repo):
        """Test get_status detects uncommitted changes."""
        repo_dir, got_dir = temp_git_repo

        # Create uncommitted file
        (repo_dir / "dirty.txt").write_text("uncommitted")

        manager = SyncManager(got_dir)

        status = manager.get_status()

        assert status.dirty is True

    def test_find_git_root_walks_up_tree(self, tmp_path):
        """Test git root detection walks up directory tree."""
        # Create nested structure
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        git_dir = repo_dir / ".git"
        git_dir.mkdir()

        nested = repo_dir / "nested" / "deep"
        nested.mkdir(parents=True)

        got_dir = nested / ".got"
        got_dir.mkdir()

        manager = SyncManager(got_dir)

        assert manager.git_dir == repo_dir

    def test_get_current_commit(self, temp_git_repo):
        """Test getting current commit hash."""
        repo_dir, got_dir = temp_git_repo

        manager = SyncManager(got_dir)

        commit = manager._get_current_commit()

        # Should be a 7-char hash
        assert len(commit) == 7
        assert commit.isalnum()

    @patch('subprocess.run')
    def test_run_git_timeout_raises_error(self, mock_run, temp_git_repo):
        """Test git command timeout raises SyncError."""
        repo_dir, got_dir = temp_git_repo

        manager = SyncManager(got_dir)

        # Mock timeout
        mock_run.side_effect = subprocess.TimeoutExpired("git", 30)

        with pytest.raises(SyncError) as exc_info:
            manager._run_git(["status"])

        assert "timed out" in str(exc_info.value)

    @patch('subprocess.run')
    def test_run_git_failure_raises_error(self, mock_run, temp_git_repo):
        """Test git command failure raises SyncError."""
        repo_dir, got_dir = temp_git_repo

        manager = SyncManager(got_dir)

        # Mock command failure
        mock_run.side_effect = subprocess.CalledProcessError(
            1, "git", stderr="fatal: error"
        )

        with pytest.raises(SyncError) as exc_info:
            manager._run_git(["invalid-command"])

        assert "Git command failed" in str(exc_info.value)
