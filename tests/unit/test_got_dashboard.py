#!/usr/bin/env python3
"""
Tests for GoT Dashboard functionality.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.got_dashboard import DashboardMetrics


class TestGetCommitsBehindOrigin(unittest.TestCase):
    """Test the get_commits_behind_origin() method."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock manager with minimal structure
        self.manager = Mock()
        self.manager.graph = Mock()
        self.manager.graph.nodes = {}
        self.manager.graph.edges = []
        self.metrics = DashboardMetrics(self.manager)

    @patch('subprocess.run')
    @patch('scripts.got_dashboard.PROJECT_ROOT', Path('/fake/repo'))
    def test_up_to_date(self, mock_run):
        """Test when branch is up-to-date with origin."""
        # Mock git commands
        mock_run.side_effect = [
            # git fetch --quiet
            Mock(returncode=0, stdout="", stderr=""),
            # git rev-parse --abbrev-ref HEAD
            Mock(returncode=0, stdout="main\n", stderr=""),
            # git rev-parse --abbrev-ref @{upstream}
            Mock(returncode=0, stdout="origin/main\n", stderr=""),
            # git rev-list --left-right --count HEAD...@{upstream}
            Mock(returncode=0, stdout="0\t0\n", stderr=""),
        ]

        result = self.metrics.get_commits_behind_origin()

        self.assertEqual(result['status'], 'up-to-date')
        self.assertEqual(result['behind_count'], 0)
        self.assertEqual(result['ahead_count'], 0)
        self.assertIn('Up-to-date', result['message'])

    @patch('subprocess.run')
    @patch('scripts.got_dashboard.PROJECT_ROOT', Path('/fake/repo'))
    def test_behind_origin(self, mock_run):
        """Test when branch is behind origin."""
        # Mock git commands
        mock_run.side_effect = [
            # git fetch --quiet
            Mock(returncode=0, stdout="", stderr=""),
            # git rev-parse --abbrev-ref HEAD
            Mock(returncode=0, stdout="feature-branch\n", stderr=""),
            # git rev-parse --abbrev-ref @{upstream}
            Mock(returncode=0, stdout="origin/feature-branch\n", stderr=""),
            # git rev-list --left-right --count HEAD...@{upstream}
            Mock(returncode=0, stdout="0\t3\n", stderr=""),
        ]

        result = self.metrics.get_commits_behind_origin()

        self.assertEqual(result['status'], 'behind')
        self.assertEqual(result['behind_count'], 3)
        self.assertEqual(result['ahead_count'], 0)
        self.assertIn('Behind', result['message'])
        self.assertIn('3 commits', result['message'])

    @patch('subprocess.run')
    @patch('scripts.got_dashboard.PROJECT_ROOT', Path('/fake/repo'))
    def test_ahead_of_origin(self, mock_run):
        """Test when branch is ahead of origin."""
        # Mock git commands
        mock_run.side_effect = [
            # git fetch --quiet
            Mock(returncode=0, stdout="", stderr=""),
            # git rev-parse --abbrev-ref HEAD
            Mock(returncode=0, stdout="feature-branch\n", stderr=""),
            # git rev-parse --abbrev-ref @{upstream}
            Mock(returncode=0, stdout="origin/feature-branch\n", stderr=""),
            # git rev-list --left-right --count HEAD...@{upstream}
            Mock(returncode=0, stdout="2\t0\n", stderr=""),
        ]

        result = self.metrics.get_commits_behind_origin()

        self.assertEqual(result['status'], 'ahead')
        self.assertEqual(result['behind_count'], 0)
        self.assertEqual(result['ahead_count'], 2)
        self.assertIn('Ahead', result['message'])
        self.assertIn('2 commits', result['message'])

    @patch('subprocess.run')
    @patch('scripts.got_dashboard.PROJECT_ROOT', Path('/fake/repo'))
    def test_diverged(self, mock_run):
        """Test when branch has diverged from origin."""
        # Mock git commands
        mock_run.side_effect = [
            # git fetch --quiet
            Mock(returncode=0, stdout="", stderr=""),
            # git rev-parse --abbrev-ref HEAD
            Mock(returncode=0, stdout="feature-branch\n", stderr=""),
            # git rev-parse --abbrev-ref @{upstream}
            Mock(returncode=0, stdout="origin/feature-branch\n", stderr=""),
            # git rev-list --left-right --count HEAD...@{upstream}
            Mock(returncode=0, stdout="5\t3\n", stderr=""),
        ]

        result = self.metrics.get_commits_behind_origin()

        self.assertEqual(result['status'], 'diverged')
        self.assertEqual(result['behind_count'], 3)
        self.assertEqual(result['ahead_count'], 5)
        self.assertIn('Diverged', result['message'])
        self.assertIn('+5', result['message'])
        self.assertIn('-3', result['message'])

    @patch('subprocess.run')
    @patch('scripts.got_dashboard.PROJECT_ROOT', Path('/fake/repo'))
    def test_no_upstream_configured(self, mock_run):
        """Test when branch has no upstream configured."""
        # Mock git commands
        mock_run.side_effect = [
            # git fetch --quiet
            Mock(returncode=0, stdout="", stderr=""),
            # git rev-parse --abbrev-ref HEAD
            Mock(returncode=0, stdout="local-branch\n", stderr=""),
            # git rev-parse --abbrev-ref @{upstream} - fails
            subprocess.CalledProcessError(128, 'git', stderr="no upstream configured"),
        ]

        result = self.metrics.get_commits_behind_origin()

        self.assertEqual(result['status'], 'no-upstream')
        self.assertEqual(result['behind_count'], 0)
        self.assertEqual(result['ahead_count'], 0)
        self.assertIn('no upstream configured', result['message'])

    @patch('subprocess.run')
    @patch('scripts.got_dashboard.PROJECT_ROOT', Path('/fake/repo'))
    def test_network_timeout(self, mock_run):
        """Test network timeout during fetch."""
        # Mock git fetch timeout
        mock_run.side_effect = subprocess.TimeoutExpired('git fetch', 10)

        result = self.metrics.get_commits_behind_origin()

        self.assertEqual(result['status'], 'error')
        self.assertIn('timeout', result['message'].lower())

    @patch('subprocess.run')
    @patch('scripts.got_dashboard.PROJECT_ROOT', Path('/fake/repo'))
    def test_git_error(self, mock_run):
        """Test git error handling."""
        # Mock git commands
        mock_run.side_effect = [
            # git fetch --quiet
            Mock(returncode=0, stdout="", stderr=""),
            # git rev-parse --abbrev-ref HEAD
            Mock(returncode=0, stdout="main\n", stderr=""),
            # git rev-parse --abbrev-ref @{upstream}
            Mock(returncode=0, stdout="origin/main\n", stderr=""),
            # git rev-list fails
            subprocess.CalledProcessError(1, 'git', stderr="fatal: bad revision"),
        ]

        result = self.metrics.get_commits_behind_origin()

        self.assertEqual(result['status'], 'error')
        self.assertIn('Git error', result['message'])

    @patch('subprocess.run')
    @patch('scripts.got_dashboard.PROJECT_ROOT')
    def test_last_fetch_time(self, mock_root, mock_run):
        """Test last fetch time calculation."""
        # Create a mock FETCH_HEAD with recent mtime
        fetch_head = Mock()
        fetch_head.exists.return_value = True
        recent_time = datetime.now() - timedelta(minutes=5)
        fetch_head.stat.return_value = Mock(st_mtime=recent_time.timestamp())

        # Mock path to return our fetch_head
        mock_root_path = Mock()
        mock_root_path.__truediv__ = lambda self, x: mock_root_path
        mock_root_path.exists.return_value = True
        mock_root_path.stat.return_value = fetch_head.stat.return_value
        mock_root.__truediv__.return_value = mock_root_path

        # Set PROJECT_ROOT to return our mock path
        mock_root.return_value = mock_root_path

        # Mock git commands
        mock_run.side_effect = [
            # git fetch --quiet
            Mock(returncode=0, stdout="", stderr=""),
            # git rev-parse --abbrev-ref HEAD
            Mock(returncode=0, stdout="main\n", stderr=""),
            # git rev-parse --abbrev-ref @{upstream}
            Mock(returncode=0, stdout="origin/main\n", stderr=""),
            # git rev-list --left-right --count HEAD...@{upstream}
            Mock(returncode=0, stdout="0\t0\n", stderr=""),
        ]

        # Patch Path to use our mock
        with patch('scripts.got_dashboard.Path') as mock_path_class:
            mock_path_instance = Mock()
            mock_path_instance.__truediv__ = lambda self, x: mock_path_instance
            mock_path_instance.exists.return_value = True
            mock_path_instance.stat.return_value = fetch_head.stat.return_value
            mock_path_class.return_value = mock_path_instance

            result = self.metrics.get_commits_behind_origin()

            # Should have last_fetch time
            self.assertIsNotNone(result.get('last_fetch'))
            # Should be in minutes
            self.assertIn('m ago', result.get('last_fetch', ''))

    @patch('subprocess.run')
    @patch('scripts.got_dashboard.PROJECT_ROOT', Path('/fake/repo'))
    def test_single_commit_grammar(self, mock_run):
        """Test singular 'commit' grammar for 1 commit."""
        # Mock git commands for 1 commit behind
        mock_run.side_effect = [
            # git fetch --quiet
            Mock(returncode=0, stdout="", stderr=""),
            # git rev-parse --abbrev-ref HEAD
            Mock(returncode=0, stdout="main\n", stderr=""),
            # git rev-parse --abbrev-ref @{upstream}
            Mock(returncode=0, stdout="origin/main\n", stderr=""),
            # git rev-list --left-right --count HEAD...@{upstream}
            Mock(returncode=0, stdout="0\t1\n", stderr=""),
        ]

        result = self.metrics.get_commits_behind_origin()

        # Should say "1 commit" not "1 commits"
        self.assertIn('1 commit', result['message'])
        self.assertNotIn('1 commits', result['message'])


class TestGitIntegrationStatus(unittest.TestCase):
    """Test the get_git_integration_status() method."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = Mock()
        self.manager.graph = Mock()
        self.manager.graph.nodes = {}
        self.manager.graph.edges = []
        self.metrics = DashboardMetrics(self.manager)

    @patch.object(DashboardMetrics, 'get_commits_behind_origin')
    @patch('subprocess.run')
    @patch('scripts.got_dashboard.PROJECT_ROOT', Path('/fake/repo'))
    def test_integration_includes_origin_status(self, mock_run, mock_behind):
        """Test that get_git_integration_status includes origin status."""
        # Mock origin status
        mock_behind.return_value = {
            'status': 'behind',
            'behind_count': 3,
            'ahead_count': 0,
            'message': 'Behind origin/main by 3 commits',
            'last_fetch': '5m ago',
        }

        # Mock git commands
        mock_run.side_effect = [
            # git rev-parse --abbrev-ref HEAD
            Mock(returncode=0, stdout="feature-branch\n", stderr=""),
            # git rev-parse --verify main
            Mock(returncode=0),
            # git rev-list --count main..HEAD
            Mock(returncode=0, stdout="2\n", stderr=""),
            # git rev-list --count HEAD..main
            Mock(returncode=0, stdout="5\n", stderr=""),
            # git status --porcelain
            Mock(returncode=0, stdout="M file.py\n", stderr=""),
            # git log --oneline -20
            Mock(returncode=0, stdout="abc123 task: T-123 something\n", stderr=""),
        ]

        result = self.metrics.get_git_integration_status()

        self.assertIn('origin_status', result)
        self.assertEqual(result['origin_status']['status'], 'behind')
        self.assertEqual(result['origin_status']['behind_count'], 3)


class TestDashboardRendering(unittest.TestCase):
    """Test dashboard rendering with origin status."""

    @patch('scripts.got_dashboard.DashboardMetrics')
    def test_render_shows_origin_warning(self, mock_metrics_class):
        """Test that rendering shows warning when significantly behind."""
        from scripts.got_dashboard import render_git_integration_section

        # Create mock git stats with significant behind count
        stats = {
            'branch': 'feature-branch',
            'is_main': False,
            'drift': {'ahead': 2, 'behind': 1},
            'uncommitted_files': 1,
            'recent_task_commits': [],
            'origin_status': {
                'status': 'behind',
                'behind_count': 10,  # Significantly behind
                'ahead_count': 0,
                'message': 'Behind origin/feature-branch by 10 commits',
                'last_fetch': '1h ago',
            }
        }

        lines = render_git_integration_section(stats)
        rendered = '\n'.join(lines)

        # Should show warning
        self.assertIn('⚠️', rendered)
        self.assertIn('tip:', rendered.lower())
        self.assertIn('git pull', rendered.lower())

    @patch('scripts.got_dashboard.DashboardMetrics')
    def test_render_shows_up_to_date(self, mock_metrics_class):
        """Test that rendering shows up-to-date status."""
        from scripts.got_dashboard import render_git_integration_section

        stats = {
            'branch': 'main',
            'is_main': True,
            'drift': None,
            'uncommitted_files': 0,
            'recent_task_commits': [],
            'origin_status': {
                'status': 'up-to-date',
                'behind_count': 0,
                'ahead_count': 0,
                'message': 'Up-to-date with origin/main',
                'last_fetch': '2m ago',
            }
        }

        lines = render_git_integration_section(stats)
        rendered = '\n'.join(lines)

        # Should show up-to-date indicator
        self.assertIn('✓', rendered)
        self.assertIn('Up-to-date', rendered)

    @patch('scripts.got_dashboard.DashboardMetrics')
    def test_render_handles_no_upstream(self, mock_metrics_class):
        """Test that rendering handles no upstream gracefully."""
        from scripts.got_dashboard import render_git_integration_section

        stats = {
            'branch': 'local-only',
            'is_main': False,
            'drift': None,
            'uncommitted_files': 0,
            'recent_task_commits': [],
            'origin_status': {
                'status': 'no-upstream',
                'behind_count': 0,
                'ahead_count': 0,
                'message': "Branch 'local-only' has no upstream configured",
                'last_fetch': None,
            }
        }

        lines = render_git_integration_section(stats)
        rendered = '\n'.join(lines)

        # Should show info indicator
        self.assertIn('ⓘ', rendered)
        self.assertIn('no upstream configured', rendered)


class TestDashboardHeaderStats(unittest.TestCase):
    """
    Unit tests for dashboard header displaying edge count and orphan count.

    Task T-20251221-020101-6913: Show edge count and orphan count in header.
    Tests the render_header_summary function directly.
    """

    def test_header_summary_contains_edge_count(self):
        """Header summary should include edge count."""
        from scripts.got_dashboard import render_header_summary

        overview = {'total_nodes': 100, 'total_edges': 250}
        health = {'orphan_count': 15}

        summary = render_header_summary(overview, health)

        self.assertIn('250', summary, "Summary should show edge count")
        self.assertIn('edge', summary.lower(), "Summary should mention edges")

    def test_header_summary_contains_orphan_count(self):
        """Header summary should include orphan count."""
        from scripts.got_dashboard import render_header_summary

        overview = {'total_nodes': 100, 'total_edges': 250}
        health = {'orphan_count': 35}

        summary = render_header_summary(overview, health)

        self.assertIn('35', summary, "Summary should show orphan count")
        self.assertIn('orphan', summary.lower(), "Summary should mention orphans")

    def test_header_summary_format(self):
        """Header summary should be concise and readable."""
        from scripts.got_dashboard import render_header_summary

        overview = {'total_nodes': 50, 'total_edges': 100}
        health = {'orphan_count': 5}

        summary = render_header_summary(overview, health)

        # Should be a single line
        self.assertEqual(summary.count('\n'), 0, "Summary should be a single line")
        # Should contain key numbers
        self.assertIn('50', summary, "Should show node count")
        self.assertIn('100', summary, "Should show edge count")
        self.assertIn('5', summary, "Should show orphan count")


if __name__ == '__main__':
    unittest.main()
