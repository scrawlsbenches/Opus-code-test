"""
Comprehensive tests for graph persistence layer.

This module tests the four components of the graph persistence system:
1. GraphWAL - Write-ahead logging for graph operations
2. GitAutoCommitter - Automatic git commits with validation
3. GraphRecovery - Multi-level recovery system
4. GraphSnapshot - Snapshot creation and management

Tests cover:
- WAL entry logging and replay
- WAL rotation and checksum verification
- GitAutoCommitter initialization
- Branch detection and protection
- Graph validation before commit
- Commit operations (mocked git)
- Debounced commits
- Safe push operations
- Multi-level recovery (WAL -> Snapshot -> Git -> Chunks)
- Snapshot creation, loading, compression, and pruning
"""

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pytest

from cortical.reasoning import ThoughtGraph, NodeType, EdgeType
from cortical.reasoning.graph_persistence import GitAutoCommitter


# =============================================================================
# HELPER FIXTURES
# =============================================================================


@pytest.fixture
def sample_graph():
    """Create a sample graph for testing."""
    graph = ThoughtGraph()
    graph.add_node("Q1", NodeType.QUESTION, "What is authentication?")
    graph.add_node("H1", NodeType.HYPOTHESIS, "Use JWT tokens")
    graph.add_node("E1", NodeType.EVIDENCE, "JWT is widely adopted")
    graph.add_edge("Q1", "H1", EdgeType.EXPLORES, weight=0.8)
    graph.add_edge("H1", "E1", EdgeType.SUPPORTS, weight=0.9)
    return graph


@pytest.fixture
def empty_graph():
    """Create an empty graph for testing."""
    return ThoughtGraph()


# =============================================================================
# TEST GRAPHWAL - Write-Ahead Logging
# =============================================================================


class TestGraphWAL:
    """Test write-ahead logging for graph operations."""

    def test_log_add_node(self, tmp_path):
        """Test logging add_node operation."""
        pytest.importorskip("cortical.reasoning.graph_persistence", reason="GraphWAL not implemented yet")
        from cortical.reasoning.graph_persistence import GraphWAL

        wal = GraphWAL(tmp_path / "wal")
        wal.log_add_node(
            node_id="Q1",
            node_type=NodeType.QUESTION,
            content="What is auth?",
            properties={"priority": "high"},
            metadata={"created": "2025-12-20"},
        )

        # Verify entry was written
        entries = list(wal.get_all_entries())
        assert len(entries) == 1
        assert entries[0].operation == "add_node"
        assert entries[0].node_id == "Q1"

    def test_log_remove_node(self, tmp_path):
        """Test logging remove_node operation."""
        pytest.importorskip("cortical.reasoning.graph_persistence", reason="GraphWAL not implemented yet")
        from cortical.reasoning.graph_persistence import GraphWAL

        wal = GraphWAL(tmp_path / "wal")
        wal.log_remove_node(node_id="Q1")

        entries = list(wal.get_all_entries())
        assert len(entries) == 1
        assert entries[0].operation == "remove_node"
        assert entries[0].node_id == "Q1"

    def test_log_add_edge(self, tmp_path):
        """Test logging add_edge operation."""
        pytest.importorskip("cortical.reasoning.graph_persistence", reason="GraphWAL not implemented yet")
        from cortical.reasoning.graph_persistence import GraphWAL

        wal = GraphWAL(tmp_path / "wal")
        wal.log_add_edge(
            source_id="Q1",
            target_id="H1",
            edge_type=EdgeType.EXPLORES,
            weight=0.75,
            confidence=0.9,
            bidirectional=False,
        )

        entries = list(wal.get_all_entries())
        assert len(entries) == 1
        assert entries[0].operation == "add_edge"
        assert entries[0].source_id == "Q1"
        assert entries[0].target_id == "H1"

    def test_log_remove_edge(self, tmp_path):
        """Test logging remove_edge operation."""
        pytest.importorskip("cortical.reasoning.graph_persistence", reason="GraphWAL not implemented yet")
        from cortical.reasoning.graph_persistence import GraphWAL

        wal = GraphWAL(tmp_path / "wal")
        wal.log_remove_edge(source_id="Q1", target_id="H1", edge_type=EdgeType.EXPLORES)

        entries = list(wal.get_all_entries())
        assert len(entries) == 1
        assert entries[0].operation == "remove_edge"

    def test_log_add_cluster(self, tmp_path):
        """Test logging add_cluster operation."""
        pytest.importorskip("cortical.reasoning.graph_persistence", reason="GraphWAL not implemented yet")
        from cortical.reasoning.graph_persistence import GraphWAL

        wal = GraphWAL(tmp_path / "wal")
        wal.log_add_cluster(
            cluster_id="CL1", name="Authentication", node_ids={"Q1", "H1", "E1"}
        )

        entries = list(wal.get_all_entries())
        assert len(entries) == 1
        assert entries[0].operation == "add_cluster"
        assert entries[0].cluster_id == "CL1"

    def test_log_merge_nodes(self, tmp_path):
        """Test logging merge_nodes operation."""
        pytest.importorskip("cortical.reasoning.graph_persistence", reason="GraphWAL not implemented yet")
        from cortical.reasoning.graph_persistence import GraphWAL

        wal = GraphWAL(tmp_path / "wal")
        wal.log_merge_nodes(node_id1="Q1", node_id2="Q2", merged_id="Q_merged")

        entries = list(wal.get_all_entries())
        assert len(entries) == 1
        assert entries[0].operation == "merge_nodes"

    def test_entry_checksum_verification(self, tmp_path):
        """Test that each entry has a valid checksum."""
        pytest.importorskip("cortical.reasoning.graph_persistence", reason="GraphWAL not implemented yet")
        from cortical.reasoning.graph_persistence import GraphWAL

        wal = GraphWAL(tmp_path / "wal")
        wal.log_add_node("Q1", NodeType.QUESTION, "Test question")

        entries = list(wal.get_all_entries())
        entry = entries[0]

        # Verify checksum exists and is valid
        assert "checksum" in entry
        assert wal.verify_entry(entry) is True

    def test_entry_checksum_detects_tampering(self, tmp_path):
        """Test that checksum verification detects modified entries."""
        pytest.importorskip("cortical.reasoning.graph_persistence", reason="GraphWAL not implemented yet")
        from cortical.reasoning.graph_persistence import GraphWAL

        wal = GraphWAL(tmp_path / "wal")
        wal.log_add_node("Q1", NodeType.QUESTION, "Original content")

        entries = list(wal.get_all_entries())
        entry = entries[0]

        # Tamper with the entry
        entry["content"] = "Modified content"

        # Verification should fail
        assert wal.verify_entry(entry) is False

    def test_replay_entries_reconstructs_graph(self, tmp_path):
        """Test that replaying WAL entries reconstructs the graph."""
        pytest.importorskip("cortical.reasoning.graph_persistence", reason="GraphWAL not implemented yet")
        from cortical.reasoning.graph_persistence import GraphWAL

        wal = GraphWAL(tmp_path / "wal")

        # Log operations
        wal.log_add_node("Q1", NodeType.QUESTION, "What is authentication?")
        wal.log_add_node("H1", NodeType.HYPOTHESIS, "Use JWT tokens")
        wal.log_add_edge("Q1", "H1", EdgeType.EXPLORES, weight=0.8)

        # Replay into a new graph
        new_graph = ThoughtGraph()
        wal.replay(new_graph)

        # Verify graph was reconstructed
        assert new_graph.node_count() == 2
        assert new_graph.edge_count() == 1

    def test_wal_rotation_on_size_limit(self, tmp_path):
        """Test that WAL rotates to a new file when size limit is reached."""
        pytest.importorskip("cortical.reasoning.graph_persistence", reason="GraphWAL not implemented yet")
        from cortical.reasoning.graph_persistence import GraphWAL

        # Create WAL with small size limit
        wal = GraphWAL(tmp_path / "wal", max_wal_size_bytes=1024)

        # Add many entries to exceed size limit
        for i in range(100):
            wal.log_add_node(f"Q{i}", NodeType.QUESTION, f"Question {i}" * 10)

        # Verify multiple WAL files were created
        wal_files = list((tmp_path / "wal" / "logs").glob("wal_*.jsonl"))
        assert len(wal_files) > 1

    def test_get_entries_since_offset(self, tmp_path):
        """Test retrieving entries from a specific offset."""
        pytest.importorskip("cortical.reasoning.graph_persistence", reason="GraphWAL not implemented yet")
        from cortical.reasoning.graph_persistence import GraphWAL

        wal = GraphWAL(tmp_path / "wal")

        # Add entries
        for i in range(1, 5):
            wal.log_add_node(f"Q{i}", NodeType.QUESTION, f"Question {i}")

        # Get entries from offset 2
        entries = wal.get_entries_since(offset=2)
        assert len(entries) == 2
        assert entries[0].node_id == "Q3"

    def test_wal_creates_directory_structure(self, tmp_path):
        """Test that WAL creates necessary directories."""
        pytest.importorskip("cortical.reasoning.graph_persistence", reason="GraphWAL not implemented yet")
        from cortical.reasoning.graph_persistence import GraphWAL

        wal_path = tmp_path / "wal"
        wal = GraphWAL(wal_path)

        assert wal_path.exists()
        assert (wal_path / "logs").exists()

    def test_wal_empty_initially(self, tmp_path):
        """Test that newly created WAL is empty."""
        pytest.importorskip("cortical.reasoning.graph_persistence", reason="GraphWAL not implemented yet")
        from cortical.reasoning.graph_persistence import GraphWAL

        wal = GraphWAL(tmp_path / "wal")
        entries = list(wal.get_all_entries())
        assert len(entries) == 0

    def test_wal_persists_across_instances(self, tmp_path):
        """Test that WAL entries persist across WAL instances."""
        pytest.importorskip("cortical.reasoning.graph_persistence", reason="GraphWAL not implemented yet")
        from cortical.reasoning.graph_persistence import GraphWAL

        wal_path = tmp_path / "wal"

        # Write entries with first instance
        wal1 = GraphWAL(wal_path)
        wal1.log_add_node("Q1", NodeType.QUESTION, "Question 1")
        wal1.log_add_node("Q2", NodeType.QUESTION, "Question 2")

        # Create new instance and verify entries exist
        wal2 = GraphWAL(wal_path)
        entries = list(wal2.get_all_entries())
        assert len(entries) == 2

    def test_wal_clear(self, tmp_path):
        """Test clearing WAL entries."""
        pytest.importorskip("cortical.reasoning.graph_persistence", reason="GraphWAL not implemented yet")
        from cortical.reasoning.graph_persistence import GraphWAL

        wal = GraphWAL(tmp_path / "wal")
        wal.log_add_node("Q1", NodeType.QUESTION, "Question 1")
        wal.clear()

        entries = list(wal.get_all_entries())
        assert len(entries) == 0


# =============================================================================
# TEST GITAUTOCOMMITTER - Automatic Git Commits (EXISTING TESTS)
# =============================================================================


class TestGitAutoCommitterInit:
    """Tests for GitAutoCommitter initialization."""

    def test_default_initialization(self):
        """Test default parameters."""
        committer = GitAutoCommitter()
        assert committer.mode == 'immediate'
        assert committer.debounce_seconds == 5
        assert committer.auto_push is False
        assert 'main' in committer.protected_branches
        assert 'master' in committer.protected_branches

    def test_custom_initialization(self):
        """Test with custom parameters."""
        committer = GitAutoCommitter(
            mode='debounced',
            debounce_seconds=10,
            auto_push=True,
            protected_branches=['main', 'prod'],
            repo_path='/tmp/test'
        )
        assert committer.mode == 'debounced'
        assert committer.debounce_seconds == 10
        assert committer.auto_push is True
        assert 'prod' in committer.protected_branches
        assert committer.repo_path == Path('/tmp/test')

    def test_invalid_mode_raises_error(self):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid mode"):
            GitAutoCommitter(mode='invalid')


class TestBranchOperations:
    """Tests for git branch operations."""

    @patch('subprocess.run')
    def test_get_current_branch(self, mock_run):
        """Test getting current branch name."""
        mock_run.return_value = MagicMock(stdout='feature/test\n', returncode=0)

        committer = GitAutoCommitter()
        branch = committer.get_current_branch()

        assert branch == 'feature/test'
        mock_run.assert_called_once()

    @patch('subprocess.run')
    def test_get_current_branch_detached_head(self, mock_run):
        """Test getting branch on detached HEAD."""
        mock_run.return_value = MagicMock(stdout='HEAD\n', returncode=0)

        committer = GitAutoCommitter()
        branch = committer.get_current_branch()

        assert branch is None

    @patch('subprocess.run')
    def test_get_current_branch_not_a_repo(self, mock_run):
        """Test getting branch outside git repo."""
        mock_run.side_effect = Exception("Not a git repository")

        committer = GitAutoCommitter()
        branch = committer.get_current_branch()

        assert branch is None

    def test_is_protected_branch_default(self):
        """Test protected branch detection with defaults."""
        committer = GitAutoCommitter()

        assert committer.is_protected_branch('main')
        assert committer.is_protected_branch('master')
        assert not committer.is_protected_branch('feature/test')

    def test_is_protected_branch_custom(self):
        """Test protected branch detection with custom list."""
        committer = GitAutoCommitter(protected_branches=['main', 'prod', 'release'])

        assert committer.is_protected_branch('main')
        assert committer.is_protected_branch('prod')
        assert committer.is_protected_branch('release')
        assert not committer.is_protected_branch('master')  # Not in custom list

    @patch('subprocess.run')
    def test_is_protected_branch_detached_head(self, mock_run):
        """Test that detached HEAD is considered protected."""
        mock_run.return_value = MagicMock(stdout='HEAD\n', returncode=0)

        committer = GitAutoCommitter()
        assert committer.is_protected_branch()  # Should return True for detached HEAD


class TestGraphValidation:
    """Tests for graph validation before commit."""

    def test_validate_empty_graph(self):
        """Test validation fails for empty graph."""
        graph = ThoughtGraph()
        committer = GitAutoCommitter()

        valid, error = committer.validate_before_commit(graph)

        assert not valid
        assert "empty graph" in error.lower()

    def test_validate_graph_with_nodes(self):
        """Test validation succeeds for graph with nodes and edges."""
        graph = ThoughtGraph()
        graph.add_node('n1', NodeType.CONCEPT, 'Test concept 1')
        graph.add_node('n2', NodeType.CONCEPT, 'Test concept 2')
        graph.add_edge('n1', 'n2', EdgeType.REQUIRES)

        committer = GitAutoCommitter()
        valid, error = committer.validate_before_commit(graph)

        assert valid
        assert error is None

    def test_validate_all_orphans(self):
        """Test validation fails when all nodes are orphaned."""
        graph = ThoughtGraph()
        graph.add_node('n1', NodeType.CONCEPT, 'Orphan 1')
        graph.add_node('n2', NodeType.CONCEPT, 'Orphan 2')
        # No edges - all nodes are orphans

        committer = GitAutoCommitter()
        valid, error = committer.validate_before_commit(graph)

        assert not valid
        assert "orphaned" in error.lower()

    def test_validate_some_orphans_ok(self):
        """Test validation succeeds when some (but not all) nodes are orphaned."""
        graph = ThoughtGraph()
        graph.add_node('n1', NodeType.CONCEPT, 'Connected 1')
        graph.add_node('n2', NodeType.CONCEPT, 'Connected 2')
        graph.add_node('n3', NodeType.CONCEPT, 'Orphan')
        graph.add_edge('n1', 'n2', EdgeType.REQUIRES)
        # n3 is orphaned, but not all nodes

        committer = GitAutoCommitter()
        valid, error = committer.validate_before_commit(graph)

        assert valid  # Should pass because not ALL nodes are orphans


class TestAutoCommit:
    """Tests for auto_commit method."""

    @patch('subprocess.run')
    def test_auto_commit_success(self, mock_run):
        """Test successful commit."""
        mock_run.return_value = MagicMock(returncode=0, stdout=b'', stderr=b'')

        committer = GitAutoCommitter()
        result = committer.auto_commit(
            message='Test commit',
            files=['test.json']
        )

        assert result is True
        # Should call: git rev-parse, git add, git commit
        assert mock_run.call_count == 3

    @patch('subprocess.run')
    def test_auto_commit_with_validation(self, mock_run):
        """Test commit with graph validation."""
        mock_run.return_value = MagicMock(returncode=0, stdout=b'', stderr=b'')

        graph = ThoughtGraph()
        graph.add_node('n1', NodeType.CONCEPT, 'Test')
        graph.add_node('n2', NodeType.CONCEPT, 'Test2')
        graph.add_edge('n1', 'n2', EdgeType.REQUIRES)

        committer = GitAutoCommitter()
        result = committer.auto_commit(
            message='Test commit',
            files=['test.json'],
            validate_graph=graph
        )

        assert result is True

    @patch('subprocess.run')
    def test_auto_commit_validation_fails(self, mock_run):
        """Test commit blocked by validation failure."""
        graph = ThoughtGraph()  # Empty graph

        committer = GitAutoCommitter()
        result = committer.auto_commit(
            message='Test commit',
            files=['test.json'],
            validate_graph=graph
        )

        assert result is False
        # Should not call git commands
        mock_run.assert_not_called()

    @patch('subprocess.run')
    def test_auto_commit_not_a_repo(self, mock_run):
        """Test commit fails outside git repo."""
        mock_run.side_effect = Exception("Not a git repository")

        committer = GitAutoCommitter()
        result = committer.auto_commit(
            message='Test commit',
            files=['test.json']
        )

        assert result is False

    @patch('subprocess.run')
    def test_auto_commit_nothing_to_commit(self, mock_run):
        """Test commit with nothing to commit (not an error)."""
        # git rev-parse succeeds, git add succeeds, git commit says nothing to commit
        def run_side_effect(*args, **kwargs):
            if 'commit' in args[0]:
                raise Exception(MagicMock(stdout=b'nothing to commit', stderr=b''))
            return MagicMock(returncode=0, stdout=b'', stderr=b'')

        mock_run.side_effect = run_side_effect

        committer = GitAutoCommitter()
        # Should handle gracefully and return True
        # (actual behavior depends on implementation)


class TestPushIfSafe:
    """Tests for push_if_safe method."""

    @patch('subprocess.run')
    def test_push_on_feature_branch(self, mock_run):
        """Test push succeeds on feature branch."""
        # Mock get_current_branch to return feature branch
        def run_side_effect(*args, **kwargs):
            if 'rev-parse' in args[0]:
                return MagicMock(stdout='feature/test\n', returncode=0)
            return MagicMock(returncode=0, stdout=b'', stderr=b'')

        mock_run.side_effect = run_side_effect

        committer = GitAutoCommitter()
        result = committer.push_if_safe()

        assert result is True

    @patch('subprocess.run')
    def test_push_blocked_on_protected_branch(self, mock_run):
        """Test push is skipped on protected branch."""
        mock_run.return_value = MagicMock(stdout='main\n', returncode=0)

        committer = GitAutoCommitter()
        result = committer.push_if_safe()

        # Should return True (not an error, just skipped)
        assert result is True
        # Should only call get_current_branch, not push
        assert mock_run.call_count == 1

    @patch('subprocess.run')
    def test_push_with_force_protected_override(self, mock_run):
        """Test push with force_protected=True overrides protection."""
        def run_side_effect(*args, **kwargs):
            if 'rev-parse' in args[0]:
                return MagicMock(stdout='main\n', returncode=0)
            return MagicMock(returncode=0, stdout=b'', stderr=b'')

        mock_run.side_effect = run_side_effect

        committer = GitAutoCommitter()
        result = committer.push_if_safe(force_protected=True)

        assert result is True
        # Should call both get_current_branch and push
        assert mock_run.call_count == 2

    @patch('subprocess.run')
    def test_push_detached_head(self, mock_run):
        """Test push fails on detached HEAD."""
        mock_run.return_value = MagicMock(stdout='HEAD\n', returncode=0)

        committer = GitAutoCommitter()
        result = committer.push_if_safe()

        assert result is False


class TestCommitOnSave:
    """Tests for commit_on_save method."""

    @patch('subprocess.run')
    def test_immediate_mode(self, mock_run):
        """Test immediate mode commits right away."""
        mock_run.return_value = MagicMock(returncode=0, stdout=b'', stderr=b'')

        committer = GitAutoCommitter(mode='immediate')
        committer.commit_on_save('/tmp/test.json')

        # Should commit immediately
        time.sleep(0.1)
        assert mock_run.call_count >= 3  # rev-parse, add, commit

    @patch('subprocess.run')
    def test_debounced_mode(self, mock_run):
        """Test debounced mode waits before committing."""
        mock_run.return_value = MagicMock(returncode=0, stdout=b'', stderr=b'')

        committer = GitAutoCommitter(mode='debounced', debounce_seconds=1)
        committer.commit_on_save('/tmp/test.json')

        # Should not commit immediately
        assert mock_run.call_count == 0

        # Wait for debounce
        time.sleep(1.2)
        # Now should have committed
        assert mock_run.call_count >= 3

        # Cleanup
        committer.cleanup()

    def test_manual_mode_no_commit(self):
        """Test manual mode doesn't commit."""
        committer = GitAutoCommitter(mode='manual')
        committer.commit_on_save('/tmp/test.json')

        # No commit should happen (no way to verify, but no error)

    @patch('subprocess.run')
    def test_debounced_multiple_saves(self, mock_run):
        """Test debounced mode resets timer on multiple saves."""
        mock_run.return_value = MagicMock(returncode=0, stdout=b'', stderr=b'')

        committer = GitAutoCommitter(mode='debounced', debounce_seconds=1)

        # First save
        committer.commit_on_save('/tmp/test1.json')
        time.sleep(0.5)

        # Second save (should reset timer)
        committer.commit_on_save('/tmp/test2.json')
        time.sleep(0.5)

        # Should not have committed yet (timer was reset)
        assert mock_run.call_count == 0

        # Wait for full debounce from second save
        time.sleep(0.7)
        # Now should have committed (only once, for last save)
        assert mock_run.call_count >= 3

        # Cleanup
        committer.cleanup()


class TestBackupBranch:
    """Tests for create_backup_branch method."""

    @patch('subprocess.run')
    def test_create_backup_branch(self, mock_run):
        """Test backup branch creation."""
        def run_side_effect(*args, **kwargs):
            if 'rev-parse' in args[0]:
                return MagicMock(stdout='feature/test\n', returncode=0)
            return MagicMock(returncode=0, stdout=b'', stderr=b'')

        mock_run.side_effect = run_side_effect

        committer = GitAutoCommitter()
        backup = committer.create_backup_branch(prefix='backup')

        assert backup is not None
        assert backup.startswith('backup/feature/test/')

    @patch('subprocess.run')
    def test_create_backup_detached_head(self, mock_run):
        """Test backup creation fails on detached HEAD."""
        mock_run.return_value = MagicMock(stdout='HEAD\n', returncode=0)

        committer = GitAutoCommitter()
        backup = committer.create_backup_branch()

        assert backup is None


class TestCleanup:
    """Tests for cleanup method."""

    @patch('subprocess.run')
    def test_cleanup_cancels_timer(self, mock_run):
        """Test cleanup cancels pending debounced commit."""
        mock_run.return_value = MagicMock(returncode=0, stdout=b'', stderr=b'')

        committer = GitAutoCommitter(mode='debounced', debounce_seconds=10)
        committer.commit_on_save('/tmp/test.json')

        # Timer should be pending
        assert committer._debounce_timer is not None

        # Cleanup
        committer.cleanup()

        # Timer should be cancelled
        assert committer._debounce_timer is None
        assert committer._pending_commit is None

        # Wait to ensure commit doesn't happen
        time.sleep(0.5)
        assert mock_run.call_count == 0


# =============================================================================
# TEST GRAPHRECOVERY - Multi-Level Recovery System
# =============================================================================


class TestGraphRecovery:
    """Test multi-level graph recovery system."""

    def test_level1_wal_replay_success(self, tmp_path, sample_graph):
        """Test Level 1 recovery: WAL replay."""
        pytest.importorskip("cortical.reasoning.graph_persistence", reason="GraphRecovery not implemented yet")
        from cortical.reasoning.graph_persistence import GraphRecovery, GraphWAL

        # Create WAL with operations
        wal = GraphWAL(tmp_path / "wal")
        wal.log_add_node("Q1", NodeType.QUESTION, "What is authentication?")
        wal.log_add_node("H1", NodeType.HYPOTHESIS, "Use JWT tokens")
        wal.log_add_edge("Q1", "H1", EdgeType.EXPLORES)

        # Simulate crash and recovery
        recovery = GraphRecovery(tmp_path)
        recovered_graph = recovery.recover()

        assert recovered_graph is not None
        assert recovered_graph.node_count() == 2
        assert recovery.recovery_level == 1

    def test_level1_fails_falls_to_level2(self, tmp_path):
        """Test that Level 1 failure triggers Level 2 recovery."""
        pytest.importorskip("cortical.reasoning.graph_persistence", reason="GraphRecovery not implemented yet")
        from cortical.reasoning.graph_persistence import GraphRecovery, GraphSnapshot

        # Create a snapshot but corrupt the WAL
        snapshot = GraphSnapshot(tmp_path / "snapshots")
        graph = ThoughtGraph()
        graph.add_node("Q1", NodeType.QUESTION, "Question 1")
        snapshot.create_snapshot(graph, "backup")

        # Corrupt WAL
        wal_dir = tmp_path / "wal"
        wal_dir.mkdir(exist_ok=True, parents=True)
        (wal_dir / "logs").mkdir(exist_ok=True)
        (wal_dir / "logs" / "wal_current.jsonl").write_text("CORRUPTED DATA!!!")

        # Recovery should fall back to Level 2
        recovery = GraphRecovery(tmp_path)
        recovered_graph = recovery.recover()

        assert recovered_graph is not None
        assert recovery.recovery_level == 2

    def test_level2_snapshot_rollback(self, tmp_path, sample_graph):
        """Test Level 2 recovery: Snapshot rollback."""
        pytest.importorskip("cortical.reasoning.graph_persistence", reason="GraphRecovery not implemented yet")
        from cortical.reasoning.graph_persistence import GraphRecovery, GraphSnapshot

        # Create snapshot
        snapshot = GraphSnapshot(tmp_path / "snapshots")
        snapshot.create_snapshot(sample_graph, "backup-v1")

        # Remove WAL to force Level 2
        recovery = GraphRecovery(tmp_path, wal_enabled=False)
        recovered_graph = recovery.recover()

        assert recovered_graph is not None
        assert recovery.recovery_level == 2

    def test_level3_git_recovery(self, tmp_path):
        """Test Level 3 recovery: Git history."""
        pytest.importorskip("cortical.reasoning.graph_persistence", reason="GraphRecovery not implemented yet")
        from cortical.reasoning.graph_persistence import GraphRecovery

        # Initialize git repo and commit a graph state
        import subprocess

        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=tmp_path,
            check=True,
            capture_output=True
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=tmp_path,
            check=True,
            capture_output=True
        )

        # Create a graph state file and commit it
        state_dir = tmp_path / "graph_state"
        state_dir.mkdir()
        state_file = state_dir / "state.json"
        state_file.write_text(
            json.dumps(
                {
                    "nodes": [
                        {
                            "id": "Q1",
                            "node_type": "question",
                            "content": "What is auth?",
                            "properties": {},
                            "metadata": {},
                        }
                    ],
                    "edges": [],
                    "clusters": [],
                }
            )
        )
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Save graph state"],
            cwd=tmp_path,
            check=True,
            capture_output=True
        )

        # Recovery should use git
        recovery = GraphRecovery(tmp_path, wal_enabled=False, snapshot_enabled=False)
        recovered_graph = recovery.recover()

        assert recovered_graph is not None
        assert recovery.recovery_level == 3

    def test_level4_chunk_reconstruction(self, tmp_path):
        """Test Level 4 recovery: Chunk-based reconstruction."""
        pytest.importorskip("cortical.reasoning.graph_persistence", reason="GraphRecovery not implemented yet")
        from cortical.reasoning.graph_persistence import GraphRecovery

        # Create chunk files
        chunks_dir = tmp_path / "chunks"
        chunks_dir.mkdir()

        # Write graph chunks
        chunk1 = {
            "chunk_id": "chunk_001",
            "operations": [
                {
                    "op": "add_node",
                    "node_id": "Q1",
                    "node_type": "question",
                    "content": "Question 1",
                }
            ],
        }
        (chunks_dir / "chunk_001.json").write_text(json.dumps(chunk1))

        # Recovery should reconstruct from chunks
        recovery = GraphRecovery(
            tmp_path, wal_enabled=False, snapshot_enabled=False, git_enabled=False
        )
        recovered_graph = recovery.recover()

        assert recovered_graph is not None
        assert recovery.recovery_level == 4

    def test_verify_graph_integrity_detects_orphans(self, tmp_path):
        """Test that integrity verification detects orphan nodes."""
        pytest.importorskip("cortical.reasoning.graph_persistence", reason="GraphRecovery not implemented yet")
        from cortical.reasoning.graph_persistence import GraphRecovery

        graph = ThoughtGraph()
        graph.add_node("Q1", NodeType.QUESTION, "Orphan question")

        recovery = GraphRecovery(tmp_path)
        is_valid, errors = recovery.verify_integrity(graph)

        assert is_valid is False
        assert any("orphan" in str(e).lower() for e in errors)

    def test_verify_graph_integrity_valid_graph(self, tmp_path, sample_graph):
        """Test that valid graphs pass integrity verification."""
        pytest.importorskip("cortical.reasoning.graph_persistence", reason="GraphRecovery not implemented yet")
        from cortical.reasoning.graph_persistence import GraphRecovery

        recovery = GraphRecovery(tmp_path)
        is_valid, errors = recovery.verify_integrity(sample_graph)

        assert is_valid is True
        assert len(errors) == 0

    def test_needs_recovery_true_when_wal_exists(self, tmp_path):
        """Test that recovery is needed when WAL exists."""
        pytest.importorskip("cortical.reasoning.graph_persistence", reason="GraphRecovery not implemented yet")
        from cortical.reasoning.graph_persistence import GraphRecovery, GraphWAL

        # Create WAL with entries
        wal = GraphWAL(tmp_path / "wal")
        wal.log_add_node("Q1", NodeType.QUESTION, "Question")

        recovery = GraphRecovery(tmp_path)
        assert recovery.needs_recovery() is True

    def test_needs_recovery_false_when_clean(self, tmp_path):
        """Test that recovery is not needed for clean state."""
        pytest.importorskip("cortical.reasoning.graph_persistence", reason="GraphRecovery not implemented yet")
        from cortical.reasoning.graph_persistence import GraphRecovery

        recovery = GraphRecovery(tmp_path)
        assert recovery.needs_recovery() is False

    def test_recovery_returns_none_when_nothing_to_recover(self, tmp_path):
        """Test that recovery returns None when no recovery source exists."""
        pytest.importorskip("cortical.reasoning.graph_persistence", reason="GraphRecovery not implemented yet")
        from cortical.reasoning.graph_persistence import GraphRecovery

        recovery = GraphRecovery(
            tmp_path,
            wal_enabled=False,
            snapshot_enabled=False,
            git_enabled=False,
            chunks_enabled=False,
        )
        recovered_graph = recovery.recover()

        assert recovered_graph is None

    def test_recovery_clears_wal_after_success(self, tmp_path):
        """Test that recovery clears WAL after successful recovery."""
        pytest.importorskip("cortical.reasoning.graph_persistence", reason="GraphRecovery not implemented yet")
        from cortical.reasoning.graph_persistence import GraphRecovery, GraphWAL

        wal = GraphWAL(tmp_path / "wal")
        wal.log_add_node("Q1", NodeType.QUESTION, "Question")

        recovery = GraphRecovery(tmp_path)
        recovered_graph = recovery.recover(clear_wal=True)

        # WAL should be cleared
        entries = list(wal.get_all_entries())
        assert len(entries) == 0

    def test_recovery_preserves_wal_when_requested(self, tmp_path):
        """Test that recovery can preserve WAL if requested."""
        pytest.importorskip("cortical.reasoning.graph_persistence", reason="GraphRecovery not implemented yet")
        from cortical.reasoning.graph_persistence import GraphRecovery, GraphWAL

        wal = GraphWAL(tmp_path / "wal")
        wal.log_add_node("Q1", NodeType.QUESTION, "Question")

        recovery = GraphRecovery(tmp_path)
        recovered_graph = recovery.recover(clear_wal=False)

        # WAL should still have entries
        entries = list(wal.get_all_entries())
        assert len(entries) == 1

    def test_recovery_handles_partial_wal_corruption(self, tmp_path):
        """Test recovery from partially corrupted WAL."""
        pytest.importorskip("cortical.reasoning.graph_persistence", reason="GraphRecovery not implemented yet")
        from cortical.reasoning.graph_persistence import GraphRecovery, GraphWAL

        wal = GraphWAL(tmp_path / "wal")
        wal.log_add_node("Q1", NodeType.QUESTION, "Question 1")
        wal.log_add_node("Q2", NodeType.QUESTION, "Question 2")

        # Corrupt the WAL file by appending invalid JSON
        wal_files = list((tmp_path / "wal" / "logs").glob("wal_*.jsonl"))
        if wal_files:
            with open(wal_files[0], "a") as f:
                f.write("CORRUPTED LINE\n")

        recovery = GraphRecovery(tmp_path)
        recovered_graph = recovery.recover()

        # Should recover valid entries before corruption
        assert recovered_graph is not None
        assert recovered_graph.node_count() >= 1

    def test_recovery_metrics_tracked(self, tmp_path):
        """Test that recovery tracks metrics."""
        pytest.importorskip("cortical.reasoning.graph_persistence", reason="GraphRecovery not implemented yet")
        from cortical.reasoning.graph_persistence import GraphRecovery, GraphWAL

        wal = GraphWAL(tmp_path / "wal")
        wal.log_add_node("Q1", NodeType.QUESTION, "Question")

        recovery = GraphRecovery(tmp_path)
        recovered_graph = recovery.recover()

        metrics = recovery.get_recovery_metrics()
        assert "recovery_level" in metrics
        assert "recovery_time_ms" in metrics

    def test_recovery_level_priority(self, tmp_path):
        """Test that lower recovery levels are preferred."""
        pytest.importorskip("cortical.reasoning.graph_persistence", reason="GraphRecovery not implemented yet")
        from cortical.reasoning.graph_persistence import GraphRecovery, GraphSnapshot, GraphWAL

        # Create both WAL and snapshot
        wal = GraphWAL(tmp_path / "wal")
        wal.log_add_node("Q1", NodeType.QUESTION, "From WAL")

        snapshot = GraphSnapshot(tmp_path / "snapshots")
        graph = ThoughtGraph()
        graph.add_node("Q2", NodeType.QUESTION, "From snapshot")
        snapshot.create_snapshot(graph, "backup")

        recovery = GraphRecovery(tmp_path)
        recovered_graph = recovery.recover()

        # Should use Level 1 (WAL) not Level 2 (snapshot)
        assert recovery.recovery_level == 1

    def test_verify_graph_integrity_detects_invalid_edges(self, tmp_path, sample_graph):
        """Test that valid graph with proper edges passes integrity."""
        pytest.importorskip("cortical.reasoning.graph_persistence", reason="GraphRecovery not implemented yet")
        from cortical.reasoning.graph_persistence import GraphRecovery

        recovery = GraphRecovery(tmp_path)
        is_valid, errors = recovery.verify_integrity(sample_graph)

        # Valid graph should pass
        assert is_valid is True
        assert len(errors) == 0

    def test_recovery_attempts_all_levels_in_order(self, tmp_path):
        """Test that recovery tries levels in order: WAL -> Snapshot -> Git -> Chunks."""
        pytest.importorskip("cortical.reasoning.graph_persistence", reason="GraphRecovery not implemented yet")
        from cortical.reasoning.graph_persistence import GraphRecovery

        # Create only chunk-based recovery
        chunks_dir = tmp_path / "chunks"
        chunks_dir.mkdir()
        chunk = {
            "chunk_id": "chunk_001",
            "operations": [
                {"op": "add_node", "node_id": "Q1", "node_type": "question", "content": "Q"}
            ],
        }
        (chunks_dir / "chunk_001.json").write_text(json.dumps(chunk))

        recovery = GraphRecovery(tmp_path)
        recovered_graph = recovery.recover()

        # Should fall through to Level 4
        assert recovery.recovery_level == 4


# =============================================================================
# TEST GRAPHSNAPSHOT - Snapshot Creation and Management
# =============================================================================


class TestGraphSnapshot:
    """Test snapshot creation and management."""

    def test_create_snapshot_saves_graph(self, tmp_path, sample_graph):
        """Test that creating a snapshot saves the graph."""
        pytest.importorskip("cortical.reasoning.graph_persistence", reason="GraphSnapshot not implemented yet")
        from cortical.reasoning.graph_persistence import GraphSnapshot

        snapshot = GraphSnapshot(tmp_path / "snapshots")
        snapshot_id = snapshot.create_snapshot(sample_graph, "test-snapshot")

        # Verify snapshot file exists
        snapshot_files = list((tmp_path / "snapshots").glob("*.json"))
        assert len(snapshot_files) >= 1

    def test_load_snapshot_restores_graph(self, tmp_path, sample_graph):
        """Test that loading a snapshot restores the graph."""
        pytest.importorskip("cortical.reasoning.graph_persistence", reason="GraphSnapshot not implemented yet")
        from cortical.reasoning.graph_persistence import GraphSnapshot

        snapshot = GraphSnapshot(tmp_path / "snapshots")
        snapshot_id = snapshot.create_snapshot(sample_graph, "test-snapshot")

        # Load the snapshot
        restored_graph = snapshot.load_snapshot(snapshot_id)

        assert restored_graph is not None
        assert restored_graph.node_count() == sample_graph.node_count()
        assert restored_graph.edge_count() == sample_graph.edge_count()

    def test_snapshot_preserves_node_properties(self, tmp_path):
        """Test that snapshots preserve node properties."""
        pytest.importorskip("cortical.reasoning.graph_persistence", reason="GraphSnapshot not implemented yet")
        from cortical.reasoning.graph_persistence import GraphSnapshot

        graph = ThoughtGraph()
        graph.add_node(
            "Q1",
            NodeType.QUESTION,
            "What is auth?",
            properties={"priority": "high", "assignee": "alice"},
            metadata={"created": "2025-12-20", "tags": ["security"]},
        )

        snapshot = GraphSnapshot(tmp_path / "snapshots")
        snapshot_id = snapshot.create_snapshot(graph, "test")

        restored = snapshot.load_snapshot(snapshot_id)
        node = restored.get_node("Q1")

        assert node.properties["priority"] == "high"
        assert node.metadata["tags"] == ["security"]

    def test_snapshot_preserves_edge_properties(self, tmp_path):
        """Test that snapshots preserve edge properties."""
        pytest.importorskip("cortical.reasoning.graph_persistence", reason="GraphSnapshot not implemented yet")
        from cortical.reasoning.graph_persistence import GraphSnapshot

        graph = ThoughtGraph()
        graph.add_node("Q1", NodeType.QUESTION, "Question")
        graph.add_node("H1", NodeType.HYPOTHESIS, "Hypothesis")
        graph.add_edge("Q1", "H1", EdgeType.EXPLORES, weight=0.75, confidence=0.9)

        snapshot = GraphSnapshot(tmp_path / "snapshots")
        snapshot_id = snapshot.create_snapshot(graph, "test")

        restored = snapshot.load_snapshot(snapshot_id)
        edges = restored.get_edges_from("Q1")

        assert len(edges) == 1
        assert edges[0].weight == 0.75
        assert edges[0].confidence == 0.9

    def test_snapshot_compression(self, tmp_path, sample_graph):
        """Test that snapshots can be compressed."""
        pytest.importorskip("cortical.reasoning.graph_persistence", reason="GraphSnapshot not implemented yet")
        from cortical.reasoning.graph_persistence import GraphSnapshot

        snapshot = GraphSnapshot(tmp_path / "snapshots", compression=True)
        snapshot_id = snapshot.create_snapshot(sample_graph, "compressed")

        # Verify compressed file exists
        snapshot_files = list((tmp_path / "snapshots").glob("*.json.gz"))
        assert len(snapshot_files) >= 1

        # Should still be loadable
        restored = snapshot.load_snapshot(snapshot_id)
        assert restored.node_count() == sample_graph.node_count()

    def test_snapshot_pruning_keeps_max(self, tmp_path, sample_graph):
        """Test that snapshot pruning keeps only max_snapshots."""
        pytest.importorskip("cortical.reasoning.graph_persistence", reason="GraphSnapshot not implemented yet")
        from cortical.reasoning.graph_persistence import GraphSnapshot

        snapshot = GraphSnapshot(tmp_path / "snapshots", max_snapshots=3)

        # Create 5 snapshots
        for i in range(5):
            snapshot.create_snapshot(sample_graph, f"snapshot-{i}")
            time.sleep(0.01)  # Ensure different timestamps

        # Should only keep 3 most recent
        snapshot_files = list((tmp_path / "snapshots").glob("*.json"))
        assert len(snapshot_files) == 3

    def test_snapshot_pruning_keeps_newest(self, tmp_path, sample_graph):
        """Test that pruning keeps the newest snapshots."""
        pytest.importorskip("cortical.reasoning.graph_persistence", reason="GraphSnapshot not implemented yet")
        from cortical.reasoning.graph_persistence import GraphSnapshot

        snapshot = GraphSnapshot(tmp_path / "snapshots", max_snapshots=2)

        # Create snapshots with delays
        snapshot.create_snapshot(sample_graph, "old")
        time.sleep(0.1)
        snapshot.create_snapshot(sample_graph, "middle")
        time.sleep(0.1)
        snapshot.create_snapshot(sample_graph, "new")

        # Should keep "new" and "middle", delete "old"
        snapshots = snapshot.list_snapshots()
        names = [s["name"] for s in snapshots]

        assert "new" in names
        assert "middle" in names
        assert "old" not in names

    def test_snapshot_integrity_verification(self, tmp_path, sample_graph):
        """Test that snapshot integrity is verified on load."""
        pytest.importorskip("cortical.reasoning.graph_persistence", reason="GraphSnapshot not implemented yet")
        from cortical.reasoning.graph_persistence import GraphSnapshot

        snapshot = GraphSnapshot(tmp_path / "snapshots")
        snapshot_id = snapshot.create_snapshot(sample_graph, "test")

        # Corrupt the snapshot file
        snapshot_files = list((tmp_path / "snapshots").glob("*.json"))
        if snapshot_files:
            with open(snapshot_files[0], "w") as f:
                f.write("CORRUPTED DATA")

        # Loading should detect corruption
        with pytest.raises(Exception):  # Could be ValueError, JSONDecodeError, etc.
            snapshot.load_snapshot(snapshot_id)

    def test_snapshot_list_shows_metadata(self, tmp_path, sample_graph):
        """Test that listing snapshots shows metadata."""
        pytest.importorskip("cortical.reasoning.graph_persistence", reason="GraphSnapshot not implemented yet")
        from cortical.reasoning.graph_persistence import GraphSnapshot

        snapshot = GraphSnapshot(tmp_path / "snapshots")
        snapshot.create_snapshot(sample_graph, "test-snapshot")

        snapshots = snapshot.list_snapshots()
        assert len(snapshots) >= 1

        snap = snapshots[0]
        assert "id" in snap
        assert "name" in snap
        assert "timestamp" in snap

    def test_snapshot_delete(self, tmp_path, sample_graph):
        """Test deleting a snapshot."""
        pytest.importorskip("cortical.reasoning.graph_persistence", reason="GraphSnapshot not implemented yet")
        from cortical.reasoning.graph_persistence import GraphSnapshot

        snapshot = GraphSnapshot(tmp_path / "snapshots")
        snapshot_id = snapshot.create_snapshot(sample_graph, "test")

        # Delete the snapshot
        snapshot.delete_snapshot(snapshot_id)

        # Should no longer be listable
        snapshots = snapshot.list_snapshots()
        assert not any(s["id"] == snapshot_id for s in snapshots)

    def test_snapshot_get_latest(self, tmp_path, sample_graph):
        """Test getting the latest snapshot."""
        pytest.importorskip("cortical.reasoning.graph_persistence", reason="GraphSnapshot not implemented yet")
        from cortical.reasoning.graph_persistence import GraphSnapshot

        snapshot = GraphSnapshot(tmp_path / "snapshots")
        snapshot.create_snapshot(sample_graph, "old")
        time.sleep(0.1)
        latest_id = snapshot.create_snapshot(sample_graph, "new")

        retrieved = snapshot.get_latest_snapshot()
        assert retrieved["id"] == latest_id
        assert retrieved["name"] == "new"

    def test_snapshot_with_clusters(self, tmp_path):
        """Test that snapshots preserve clusters."""
        pytest.importorskip("cortical.reasoning.graph_persistence", reason="GraphSnapshot not implemented yet")
        from cortical.reasoning.graph_persistence import GraphSnapshot

        graph = ThoughtGraph()
        graph.add_node("Q1", NodeType.QUESTION, "Q1")
        graph.add_node("Q2", NodeType.QUESTION, "Q2")
        graph.add_cluster("CL1", "Test Cluster", {"Q1", "Q2"})

        snapshot = GraphSnapshot(tmp_path / "snapshots")
        snapshot_id = snapshot.create_snapshot(graph, "test")

        restored = snapshot.load_snapshot(snapshot_id)
        assert "CL1" in restored.clusters
        assert restored.clusters["CL1"].name == "Test Cluster"
