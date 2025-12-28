"""
Tests for GitHistoryTrainer - weighted training from git history.

Following TDD: These tests define expected behavior before implementation.
"""

import pytest
from datetime import datetime, timezone, timedelta
from cortical.spark.git_trainer import GitHistoryTrainer, WeightedCommit


class TestBranchWeights:
    """Test branch weight computation."""

    def test_main_branch_weight(self):
        """Main branch should have weight 1.0."""
        trainer = GitHistoryTrainer()
        assert trainer.get_branch_weight('main') == 1.0
        assert trainer.get_branch_weight('master') == 1.0

    def test_develop_branch_weight(self):
        """Develop branch should have weight 0.8."""
        trainer = GitHistoryTrainer()
        assert trainer.get_branch_weight('develop') == 0.8

    def test_release_hotfix_weights(self):
        """Release and hotfix branches should have weight 0.9."""
        trainer = GitHistoryTrainer()
        assert trainer.get_branch_weight('release') == 0.9
        assert trainer.get_branch_weight('hotfix') == 0.9
        assert trainer.get_branch_weight('release/v1.0') == 0.9
        assert trainer.get_branch_weight('hotfix/bug-123') == 0.9

    def test_feature_branch_weight(self):
        """Feature branches should have weight 0.6."""
        trainer = GitHistoryTrainer()
        assert trainer.get_branch_weight('feature') == 0.6
        assert trainer.get_branch_weight('feature/auth') == 0.6

    def test_claude_branch_weight(self):
        """Claude branches should have weight 0.4."""
        trainer = GitHistoryTrainer()
        assert trainer.get_branch_weight('claude') == 0.4
        assert trainer.get_branch_weight('claude/fix-bug') == 0.4
        assert trainer.get_branch_weight('claude/feature-x') == 0.4

    def test_unknown_branch_weight(self):
        """Unknown branches should have medium weight 0.5."""
        trainer = GitHistoryTrainer()
        assert trainer.get_branch_weight('random-branch') == 0.5
        assert trainer.get_branch_weight('experimental') == 0.5

    def test_branch_name_case_insensitive(self):
        """Branch names should be case insensitive."""
        trainer = GitHistoryTrainer()
        assert trainer.get_branch_weight('Main') == 1.0
        assert trainer.get_branch_weight('FEATURE/test') == 0.6
        assert trainer.get_branch_weight('Claude/fix') == 0.4


class TestTemporalDecay:
    """Test temporal decay computation."""

    def test_no_decay_for_current_time(self):
        """Commits at reference time should have no decay (weight 1.0)."""
        trainer = GitHistoryTrainer(temporal_half_life_days=30.0)
        now = datetime.now(timezone.utc)
        decay = trainer.compute_temporal_decay(now, now)
        assert abs(decay - 1.0) < 0.001

    def test_half_life_decay(self):
        """Commits at half-life age should have decay ~0.5."""
        trainer = GitHistoryTrainer(temporal_half_life_days=30.0)
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(days=30)
        decay = trainer.compute_temporal_decay(old_time, now)
        assert abs(decay - 0.5) < 0.01  # ~0.5

    def test_double_half_life_decay(self):
        """Commits at 2x half-life should have decay ~0.25."""
        trainer = GitHistoryTrainer(temporal_half_life_days=30.0)
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(days=60)
        decay = trainer.compute_temporal_decay(old_time, now)
        assert abs(decay - 0.25) < 0.01  # ~0.25

    def test_min_weight_floor(self):
        """Very old commits should not decay below min_weight."""
        trainer = GitHistoryTrainer(temporal_half_life_days=30.0, min_weight=0.1)
        now = datetime.now(timezone.utc)
        ancient_time = now - timedelta(days=1000)  # Very old
        decay = trainer.compute_temporal_decay(ancient_time, now)
        assert decay == 0.1

    def test_custom_half_life(self):
        """Custom half-life values should work correctly."""
        trainer = GitHistoryTrainer(temporal_half_life_days=7.0)
        now = datetime.now(timezone.utc)
        week_ago = now - timedelta(days=7)
        decay = trainer.compute_temporal_decay(week_ago, now)
        assert abs(decay - 0.5) < 0.01


class TestQualitySignals:
    """Test quality signal detection."""

    def test_detect_test_files(self):
        """Should detect commits that modify test files."""
        trainer = GitHistoryTrainer()
        commit = WeightedCommit(
            sha='abc123',
            message='Add tests',
            author='dev',
            timestamp=datetime.now(timezone.utc),
            branch='main',
            files_changed=['tests/test_auth.py', 'src/auth.py'],
            diff_content='',
        )
        trainer.detect_quality_signals(commit)
        assert commit.has_tests is True

    def test_detect_test_files_various_patterns(self):
        """Should detect various test file patterns."""
        trainer = GitHistoryTrainer()

        test_patterns = [
            'test_foo.py',
            'tests/test_bar.py',
            'spec_baz.py',
            'foo_test.py',
            'foo_spec.py',
        ]

        for pattern in test_patterns:
            commit = WeightedCommit(
                sha='abc123',
                message='Update',
                author='dev',
                timestamp=datetime.now(timezone.utc),
                branch='main',
                files_changed=[pattern],
                diff_content='',
            )
            trainer.detect_quality_signals(commit)
            assert commit.has_tests is True, f"Should detect {pattern} as test file"

    def test_no_tests_in_commit(self):
        """Should not flag non-test files."""
        trainer = GitHistoryTrainer()
        commit = WeightedCommit(
            sha='abc123',
            message='Update code',
            author='dev',
            timestamp=datetime.now(timezone.utc),
            branch='main',
            files_changed=['src/auth.py', 'src/user.py'],
            diff_content='',
        )
        trainer.detect_quality_signals(commit)
        assert commit.has_tests is False

    def test_detect_revert_commit(self):
        """Should detect revert commits."""
        trainer = GitHistoryTrainer()
        commit = WeightedCommit(
            sha='abc123',
            message='Revert "Add broken feature"\n\nThis reverts commit def456.',
            author='dev',
            timestamp=datetime.now(timezone.utc),
            branch='main',
            files_changed=['src/feature.py'],
            diff_content='',
        )
        trainer.detect_quality_signals(commit)
        assert commit.is_reverted is True

    def test_detect_revert_alternative_format(self):
        """Should detect revert commits with alternative wording."""
        trainer = GitHistoryTrainer()
        commit = WeightedCommit(
            sha='abc123',
            message='Reverts commit abc123 due to bug',
            author='dev',
            timestamp=datetime.now(timezone.utc),
            branch='main',
            files_changed=['src/feature.py'],
            diff_content='',
        )
        trainer.detect_quality_signals(commit)
        assert commit.is_reverted is True

    def test_normal_commit_not_revert(self):
        """Normal commits should not be flagged as reverts."""
        trainer = GitHistoryTrainer()
        commit = WeightedCommit(
            sha='abc123',
            message='Add new feature',
            author='dev',
            timestamp=datetime.now(timezone.utc),
            branch='main',
            files_changed=['src/feature.py'],
            diff_content='',
        )
        trainer.detect_quality_signals(commit)
        assert commit.is_reverted is False

    def test_detect_merge_commit(self):
        """Should detect merge commits."""
        trainer = GitHistoryTrainer()
        commit = WeightedCommit(
            sha='abc123',
            message='Merge branch "feature/auth" into main',
            author='dev',
            timestamp=datetime.now(timezone.utc),
            branch='main',
            files_changed=['src/auth.py'],
            diff_content='',
        )
        trainer.detect_quality_signals(commit)
        assert commit.is_merged is True

    def test_detect_pull_request_merge(self):
        """Should detect PR merge commits."""
        trainer = GitHistoryTrainer()
        commit = WeightedCommit(
            sha='abc123',
            message='Merge pull request #42 from user/feature',
            author='dev',
            timestamp=datetime.now(timezone.utc),
            branch='main',
            files_changed=['src/feature.py'],
            diff_content='',
        )
        trainer.detect_quality_signals(commit)
        assert commit.is_merged is True


class TestDeduplication:
    """Test commit deduplication."""

    def test_duplicate_detection_enabled(self):
        """Should detect duplicate commits when enabled."""
        trainer = GitHistoryTrainer(deduplicate=True)
        assert trainer.is_duplicate('abc123') is False
        assert trainer.is_duplicate('abc123') is True  # Second time

    def test_duplicate_detection_disabled(self):
        """Should not track duplicates when disabled."""
        trainer = GitHistoryTrainer(deduplicate=False)
        assert trainer.is_duplicate('abc123') is False
        assert trainer.is_duplicate('abc123') is False  # Still False

    def test_different_commits_not_duplicates(self):
        """Different SHAs should not be flagged as duplicates."""
        trainer = GitHistoryTrainer(deduplicate=True)
        assert trainer.is_duplicate('abc123') is False
        assert trainer.is_duplicate('def456') is False
        assert trainer.is_duplicate('ghi789') is False


class TestWeightComputation:
    """Test final weight computation with all signals."""

    def test_base_weight_main_branch(self):
        """Main branch commit should start with weight 1.0."""
        trainer = GitHistoryTrainer(temporal_half_life_days=30.0)
        now = datetime.now(timezone.utc)
        commit = WeightedCommit(
            sha='abc123',
            message='Add feature',
            author='dev',
            timestamp=now,
            branch='main',
            files_changed=['src/feature.py'],
            diff_content='',
        )
        weight = trainer.compute_weight(commit, now)
        # Branch weight 1.0 * no multipliers * no decay (current time)
        assert abs(weight - 1.0) < 0.001

    def test_merged_commit_multiplier(self):
        """Merged commits should get 1.2x multiplier."""
        trainer = GitHistoryTrainer(temporal_half_life_days=30.0)
        now = datetime.now(timezone.utc)
        commit = WeightedCommit(
            sha='abc123',
            message='Merge feature',
            author='dev',
            timestamp=now,
            branch='main',
            files_changed=['src/feature.py'],
            diff_content='',
            is_merged=True,
        )
        weight = trainer.compute_weight(commit, now)
        # 1.0 (branch) * 1.2 (merged) * 1.0 (no decay)
        assert abs(weight - 1.2) < 0.001

    def test_tested_commit_multiplier(self):
        """Commits with tests should get 1.1x multiplier."""
        trainer = GitHistoryTrainer(temporal_half_life_days=30.0)
        now = datetime.now(timezone.utc)
        commit = WeightedCommit(
            sha='abc123',
            message='Add feature with tests',
            author='dev',
            timestamp=now,
            branch='main',
            files_changed=['src/feature.py', 'tests/test_feature.py'],
            diff_content='',
            has_tests=True,
        )
        weight = trainer.compute_weight(commit, now)
        # 1.0 (branch) * 1.1 (tests) * 1.0 (no decay)
        assert abs(weight - 1.1) < 0.001

    def test_ci_passed_multiplier(self):
        """Commits with CI passing should get 1.1x multiplier."""
        trainer = GitHistoryTrainer(temporal_half_life_days=30.0)
        now = datetime.now(timezone.utc)
        commit = WeightedCommit(
            sha='abc123',
            message='Add feature',
            author='dev',
            timestamp=now,
            branch='main',
            files_changed=['src/feature.py'],
            diff_content='',
            ci_passed=True,
        )
        weight = trainer.compute_weight(commit, now)
        # 1.0 (branch) * 1.1 (ci) * 1.0 (no decay)
        assert abs(weight - 1.1) < 0.001

    def test_reverted_commit_penalty(self):
        """Reverted commits should get 0.1x multiplier (heavy penalty)."""
        trainer = GitHistoryTrainer(temporal_half_life_days=30.0)
        now = datetime.now(timezone.utc)
        commit = WeightedCommit(
            sha='abc123',
            message='Revert bad commit',
            author='dev',
            timestamp=now,
            branch='main',
            files_changed=['src/feature.py'],
            diff_content='',
            is_reverted=True,
        )
        weight = trainer.compute_weight(commit, now)
        # 1.0 (branch) * 0.1 (reverted) * 1.0 (no decay)
        assert abs(weight - 0.1) < 0.001

    def test_combined_multipliers(self):
        """All positive multipliers should combine multiplicatively."""
        trainer = GitHistoryTrainer(temporal_half_life_days=30.0)
        now = datetime.now(timezone.utc)
        commit = WeightedCommit(
            sha='abc123',
            message='Merge PR with tests',
            author='dev',
            timestamp=now,
            branch='main',
            files_changed=['src/feature.py', 'tests/test_feature.py'],
            diff_content='',
            is_merged=True,
            has_tests=True,
            ci_passed=True,
        )
        weight = trainer.compute_weight(commit, now)
        # 1.0 (branch) * 1.2 (merged) * 1.1 (tests) * 1.1 (ci) * 1.0 (no decay)
        expected = 1.0 * 1.2 * 1.1 * 1.1
        assert abs(weight - expected) < 0.001

    def test_temporal_decay_applied(self):
        """Temporal decay should be applied to final weight."""
        trainer = GitHistoryTrainer(temporal_half_life_days=30.0)
        now = datetime.now(timezone.utc)
        month_ago = now - timedelta(days=30)
        commit = WeightedCommit(
            sha='abc123',
            message='Add feature',
            author='dev',
            timestamp=month_ago,
            branch='main',
            files_changed=['src/feature.py'],
            diff_content='',
        )
        weight = trainer.compute_weight(commit, now)
        # 1.0 (branch) * no multipliers * 0.5 (half-life decay)
        assert abs(weight - 0.5) < 0.01

    def test_feature_branch_with_decay(self):
        """Feature branch weight should combine with decay."""
        trainer = GitHistoryTrainer(temporal_half_life_days=30.0)
        now = datetime.now(timezone.utc)
        month_ago = now - timedelta(days=30)
        commit = WeightedCommit(
            sha='abc123',
            message='Add feature',
            author='dev',
            timestamp=month_ago,
            branch='feature/auth',
            files_changed=['src/auth.py'],
            diff_content='',
        )
        weight = trainer.compute_weight(commit, now)
        # 0.6 (feature branch) * no multipliers * 0.5 (half-life decay)
        expected = 0.6 * 0.5
        assert abs(weight - expected) < 0.01

    def test_min_weight_enforced(self):
        """Weight should never go below min_weight."""
        trainer = GitHistoryTrainer(temporal_half_life_days=30.0, min_weight=0.1)
        now = datetime.now(timezone.utc)
        ancient = now - timedelta(days=1000)
        commit = WeightedCommit(
            sha='abc123',
            message='Ancient commit',
            author='dev',
            timestamp=ancient,
            branch='feature/old',
            files_changed=['src/old.py'],
            diff_content='',
            is_reverted=True,  # 0.1x penalty
        )
        weight = trainer.compute_weight(commit, now)
        # Even with 0.6 * 0.1 * tiny_decay, should not go below 0.1
        assert weight == 0.1

    def test_weight_breakdown_recorded(self):
        """Weight computation should record breakdown."""
        trainer = GitHistoryTrainer(temporal_half_life_days=30.0)
        now = datetime.now(timezone.utc)
        commit = WeightedCommit(
            sha='abc123',
            message='Merge with tests',
            author='dev',
            timestamp=now,
            branch='main',
            files_changed=['src/feature.py', 'tests/test_feature.py'],
            diff_content='',
            is_merged=True,
            has_tests=True,
        )
        trainer.compute_weight(commit, now)

        assert 'branch' in commit.weight_breakdown
        assert commit.weight_breakdown['branch'] == 1.0
        assert 'merged' in commit.weight_breakdown
        assert commit.weight_breakdown['merged'] == 1.2
        assert 'tested' in commit.weight_breakdown
        assert commit.weight_breakdown['tested'] == 1.1
        assert 'temporal' in commit.weight_breakdown


class TestPrepareTrainingData:
    """Test preparation of training data from commits."""

    def test_prepare_single_commit(self):
        """Should prepare single commit for training."""
        trainer = GitHistoryTrainer()
        now = datetime.now(timezone.utc)
        commits = [
            WeightedCommit(
                sha='abc123',
                message='Add feature',
                author='dev',
                timestamp=now,
                branch='main',
                files_changed=['src/feature.py'],
                diff_content='def foo():\n    pass',
            )
        ]

        documents, weights = trainer.prepare_training_data(commits)

        assert len(documents) == 1
        assert len(weights) == 1
        assert 'Add feature' in documents[0]
        assert 'def foo():' in documents[0]
        assert weights[0] > 0

    def test_prepare_multiple_commits(self):
        """Should prepare multiple commits."""
        trainer = GitHistoryTrainer()
        now = datetime.now(timezone.utc)
        commits = [
            WeightedCommit(
                sha='abc123',
                message='Add feature A',
                author='dev',
                timestamp=now,
                branch='main',
                files_changed=['a.py'],
                diff_content='code A',
            ),
            WeightedCommit(
                sha='def456',
                message='Add feature B',
                author='dev',
                timestamp=now,
                branch='main',
                files_changed=['b.py'],
                diff_content='code B',
            ),
        ]

        documents, weights = trainer.prepare_training_data(commits)

        assert len(documents) == 2
        assert len(weights) == 2
        assert 'Add feature A' in documents[0]
        assert 'Add feature B' in documents[1]

    def test_skip_duplicate_commits(self):
        """Should skip duplicate commits when deduplication enabled."""
        trainer = GitHistoryTrainer(deduplicate=True)
        now = datetime.now(timezone.utc)
        commits = [
            WeightedCommit(
                sha='abc123',
                message='Add feature',
                author='dev',
                timestamp=now,
                branch='main',
                files_changed=['a.py'],
                diff_content='code',
            ),
            WeightedCommit(
                sha='abc123',  # Duplicate SHA
                message='Add feature',
                author='dev',
                timestamp=now,
                branch='main',
                files_changed=['a.py'],
                diff_content='code',
            ),
        ]

        documents, weights = trainer.prepare_training_data(commits)

        # Should only include first occurrence
        assert len(documents) == 1
        assert len(weights) == 1

    def test_quality_signals_auto_detected(self):
        """Should auto-detect quality signals during preparation."""
        trainer = GitHistoryTrainer()
        now = datetime.now(timezone.utc)
        commits = [
            WeightedCommit(
                sha='abc123',
                message='Add tests',
                author='dev',
                timestamp=now,
                branch='main',
                files_changed=['tests/test_foo.py'],
                diff_content='test code',
            )
        ]

        documents, weights = trainer.prepare_training_data(commits)

        # Weight should be boosted by test file detection
        assert weights[0] > 1.0  # Base weight * test multiplier

    def test_weights_vary_by_quality(self):
        """Commits with different quality should have different weights."""
        trainer = GitHistoryTrainer(temporal_half_life_days=30.0)
        now = datetime.now(timezone.utc)
        commits = [
            WeightedCommit(
                sha='low',
                message='Quick fix',
                author='dev',
                timestamp=now - timedelta(days=60),  # Old
                branch='feature/temp',  # Low weight branch
                files_changed=['temp.py'],
                diff_content='',
            ),
            WeightedCommit(
                sha='high',
                message='Merge PR with tests',
                author='dev',
                timestamp=now,  # Recent
                branch='main',  # High weight branch
                files_changed=['tests/test_feature.py'],
                diff_content='',
                is_merged=True,
            ),
        ]

        documents, weights = trainer.prepare_training_data(commits)

        # High-quality commit should have higher weight
        assert weights[1] > weights[0]


class TestIterCommits:
    """Test commit iteration (stub for now)."""

    def test_iter_commits_returns_empty(self):
        """iter_commits should return empty iterator (stub implementation)."""
        trainer = GitHistoryTrainer()
        commits = list(trainer.iter_commits())
        assert commits == []

    def test_iter_commits_accepts_parameters(self):
        """iter_commits should accept filtering parameters without error."""
        trainer = GitHistoryTrainer()
        now = datetime.now(timezone.utc)
        commits = list(trainer.iter_commits(
            branches=['main'],
            since=now - timedelta(days=7),
            until=now,
            max_commits=100
        ))
        assert commits == []
