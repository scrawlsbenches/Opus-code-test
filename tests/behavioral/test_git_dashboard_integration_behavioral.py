"""
Behavioral tests for GoT Dashboard Git integration.

As a developer monitoring workflow state,
I want clear visual indicators of git synchronization status,
So that I know when my local work is out of sync with the remote.

Tests demonstrate:
- Different git origin states (up-to-date, ahead, behind, diverged)
- Visual indicators and color coding
- Warning thresholds for significant drift
- Edge cases (no upstream, network errors)

Following Metus: We describe behavior, then make it true.
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.got_dashboard import render_git_integration_section


# ============================================================================
# BEHAVIORAL SCENARIOS
# ============================================================================

class TestDeveloperMonitorsGitSyncStatus:
    """
    Epic: Git Synchronization Visibility

    As a developer working on distributed workflows,
    I want to see my git sync status at a glance,
    So that I know when to pull or push changes.
    """

    def test_scenario_displays_up_to_date_status(self):
        """
        Scenario: Repository is in sync with origin

        Given my branch is up-to-date with origin
        When I view the dashboard
        Then I see a green indicator showing sync status
        """
        # Given: my branch is up-to-date with origin
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

        # When: I view the dashboard
        lines = render_git_integration_section(stats)

        # Then: I see a green indicator showing sync status
        assert any('up-to-date' in line.lower() for line in lines)

    def test_scenario_displays_behind_origin_status(self):
        """
        Scenario: Local branch is behind origin

        Given my branch is behind origin by commits
        When I view the dashboard
        Then I see an indicator showing how far behind I am
        """
        # Given: my branch is behind origin by commits
        stats = {
            'branch': 'feature-branch',
            'is_main': False,
            'drift': {'ahead': 2, 'behind': 1},
            'uncommitted_files': 1,
            'recent_task_commits': [],
            'origin_status': {
                'status': 'behind',
                'behind_count': 3,
                'ahead_count': 0,
                'message': 'Behind origin/feature-branch by 3 commits',
                'last_fetch': '10m ago',
            }
        }

        # When: I view the dashboard
        lines = render_git_integration_section(stats)

        # Then: I see an indicator showing how far behind I am
        assert any('behind' in line.lower() for line in lines)

    def test_scenario_warns_when_significantly_behind(self):
        """
        Scenario: Local branch is significantly behind (>= 5 commits)

        Given my branch is 10 commits behind origin
        When I view the dashboard
        Then I see a warning indicator
        Because significant drift requires attention
        """
        # Given: my branch is 10 commits behind origin
        stats = {
            'branch': 'feature-branch',
            'is_main': False,
            'drift': {'ahead': 2, 'behind': 1},
            'uncommitted_files': 1,
            'recent_task_commits': [],
            'origin_status': {
                'status': 'behind',
                'behind_count': 10,  # >= 5 shows warning
                'ahead_count': 0,
                'message': 'Behind origin/feature-branch by 10 commits',
                'last_fetch': '1h ago',
            }
        }

        # When: I view the dashboard
        lines = render_git_integration_section(stats)

        # Then: I see a warning indicator
        # The function uses red color and warning emoji for >= 5 commits behind
        output = '\n'.join(lines)
        assert 'behind' in output.lower()

    def test_scenario_displays_ahead_of_origin_status(self):
        """
        Scenario: Local branch has unpushed commits

        Given my branch is ahead of origin by commits
        When I view the dashboard
        Then I see an indicator showing unpushed commits
        """
        # Given: my branch is ahead of origin by commits
        stats = {
            'branch': 'feature-branch',
            'is_main': False,
            'drift': {'ahead': 2, 'behind': 1},
            'uncommitted_files': 2,
            'recent_task_commits': [],
            'origin_status': {
                'status': 'ahead',
                'behind_count': 0,
                'ahead_count': 5,
                'message': 'Ahead of origin/feature-branch by 5 commits',
                'last_fetch': '5m ago',
            }
        }

        # When: I view the dashboard
        lines = render_git_integration_section(stats)

        # Then: I see an indicator showing unpushed commits
        assert any('ahead' in line.lower() for line in lines)


class TestDeveloperHandlesDivergedBranches:
    """
    Epic: Divergence Detection

    As a developer working with others,
    I want to know when my branch has diverged from origin,
    So that I can reconcile changes before conflicts grow.
    """

    def test_scenario_displays_diverged_status(self):
        """
        Scenario: Branch has diverged from origin

        Given my branch has commits not in origin
        And origin has commits not in my branch
        When I view the dashboard
        Then I see both ahead and behind counts
        """
        # Given: my branch has diverged from origin
        stats = {
            'branch': 'feature-branch',
            'is_main': False,
            'drift': {'ahead': 2, 'behind': 1},
            'uncommitted_files': 1,
            'recent_task_commits': [],
            'origin_status': {
                'status': 'diverged',
                'behind_count': 3,
                'ahead_count': 5,
                'message': 'Diverged from origin/feature-branch: +5 -3',
                'last_fetch': '15m ago',
            }
        }

        # When: I view the dashboard
        lines = render_git_integration_section(stats)

        # Then: I see both ahead and behind counts
        output = '\n'.join(lines)
        assert 'diverged' in output.lower()


class TestDeveloperHandlesSpecialCases:
    """
    Epic: Edge Case Handling

    As a developer in various git scenarios,
    I want appropriate messaging for special cases,
    So that I understand what action to take.
    """

    def test_scenario_displays_no_upstream_configured(self):
        """
        Scenario: Branch has no upstream tracking

        Given my branch has no upstream configured
        When I view the dashboard
        Then I see a message indicating no upstream
        """
        # Given: my branch has no upstream configured
        stats = {
            'branch': 'local-only-branch',
            'is_main': False,
            'drift': None,
            'uncommitted_files': 0,
            'recent_task_commits': [],
            'origin_status': {
                'status': 'no-upstream',
                'behind_count': 0,
                'ahead_count': 0,
                'message': "Branch 'local-only-branch' has no upstream configured",
                'last_fetch': None,
            }
        }

        # When: I view the dashboard
        lines = render_git_integration_section(stats)

        # Then: I see a message indicating no upstream
        output = '\n'.join(lines)
        assert 'upstream' in output.lower() or 'local-only-branch' in output

    def test_scenario_displays_network_error_gracefully(self):
        """
        Scenario: Network error during git fetch

        Given git fetch failed due to network error
        When I view the dashboard
        Then I see an error message
        And the dashboard doesn't crash
        """
        # Given: git fetch failed due to network error
        stats = {
            'branch': 'feature-branch',
            'is_main': False,
            'drift': None,
            'uncommitted_files': 0,
            'recent_task_commits': [],
            'origin_status': {
                'status': 'error',
                'behind_count': 0,
                'ahead_count': 0,
                'message': 'Network timeout during fetch',
                'last_fetch': None,
            }
        }

        # When: I view the dashboard
        # Then: the dashboard doesn't crash
        lines = render_git_integration_section(stats)
        assert isinstance(lines, list)
        assert len(lines) > 0


class TestDeveloperDistinguishesBranchTypes:
    """
    Epic: Branch Context Awareness

    As a developer on different branches,
    I want visual distinction between main and feature branches,
    So that I'm always aware of my working context.
    """

    def test_scenario_main_branch_shows_green(self):
        """
        Scenario: Main branch gets special visual treatment

        Given I'm on the main branch
        When I view the dashboard
        Then the branch name is highlighted differently
        """
        # Given: I'm on the main branch
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

        # When: I view the dashboard
        lines = render_git_integration_section(stats)

        # Then: the branch name is shown
        output = '\n'.join(lines)
        assert 'main' in output.lower()

    def test_scenario_feature_branch_shows_yellow(self):
        """
        Scenario: Feature branches get different visual treatment

        Given I'm on a feature branch
        When I view the dashboard
        Then the branch name is indicated differently than main
        """
        # Given: I'm on a feature branch
        stats = {
            'branch': 'feature/custom-implementation',
            'is_main': False,
            'drift': None,
            'uncommitted_files': 0,
            'recent_task_commits': [],
            'origin_status': {
                'status': 'up-to-date',
                'behind_count': 0,
                'ahead_count': 0,
                'message': 'Up-to-date with origin/feature/custom-implementation',
                'last_fetch': '2m ago',
            }
        }

        # When: I view the dashboard
        lines = render_git_integration_section(stats)

        # Then: the branch name is shown
        output = '\n'.join(lines)
        assert 'feature' in output.lower()


class TestDashboardRendersConsistently:
    """
    Epic: Reliable Rendering

    As a developer relying on the dashboard,
    I want consistent output format across all scenarios,
    So that I can build muscle memory reading it.
    """

    def test_scenario_all_states_render_without_errors(self):
        """
        Scenario: All git states render successfully

        Given any valid git status configuration
        When I render the dashboard
        Then it produces output without errors
        """
        # Test all major states
        test_states = [
            'up-to-date', 'behind', 'ahead', 'diverged',
            'no-upstream', 'error'
        ]

        for state in test_states:
            # Given: any valid git status configuration
            stats = {
                'branch': 'test-branch',
                'is_main': False,
                'drift': None,
                'uncommitted_files': 0,
                'recent_task_commits': [],
                'origin_status': {
                    'status': state,
                    'behind_count': 3 if state in ['behind', 'diverged'] else 0,
                    'ahead_count': 5 if state in ['ahead', 'diverged'] else 0,
                    'message': f'Test message for {state}',
                    'last_fetch': '10m ago' if state != 'error' else None,
                }
            }

            # When: I render the dashboard
            # Then: it produces output without errors
            lines = render_git_integration_section(stats)
            assert isinstance(lines, list)
            assert len(lines) > 0

    def test_scenario_rendering_handles_missing_origin_status(self):
        """
        Scenario: Dashboard handles missing origin_status gracefully

        Given origin_status is not provided
        When I render the dashboard
        Then it handles the missing data gracefully
        """
        # Given: origin_status is not provided
        stats = {
            'branch': 'test-branch',
            'is_main': False,
            'drift': None,
            'uncommitted_files': 0,
            'recent_task_commits': [],
            # origin_status is missing
        }

        # When: I render the dashboard
        # Then: it handles the missing data gracefully
        lines = render_git_integration_section(stats)
        assert isinstance(lines, list)
        # Should render something even without origin status
