"""
GitHistoryTrainer: Train SparkSLM models from git commit history.

Assigns weights to commits based on:
- Branch type (main=1.0, feature=0.6, claude/*=0.4)
- Quality signals (merged=1.2x, tested=1.1x, reverted=0.1x)
- Temporal decay (recent commits weighted higher)

Philosophy:
    Not all commits are created equal. Production code on main branch
    with passing tests should influence the model more than experimental
    commits on feature branches. This module quantifies that intuition.

Usage:
    trainer = GitHistoryTrainer(temporal_half_life_days=30.0)
    commits = list(trainer.iter_commits(branches=['main', 'develop']))
    documents, weights = trainer.prepare_training_data(commits)

    model = NGramModel(n=3)
    model.train_weighted(documents, weights)

Weight Formula:
    final_weight = branch_weight × quality_multipliers × temporal_decay

    Where:
    - branch_weight: 0.4-1.0 based on branch type
    - quality_multipliers: Product of signal multipliers (1.1-1.2 each)
    - temporal_decay: exp(-log(2) × age / half_life)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Iterator, Tuple
from datetime import datetime, timezone
import math


@dataclass
class WeightedCommit:
    """
    A commit with computed training weight.

    Attributes:
        sha: Commit hash
        message: Commit message
        author: Commit author
        timestamp: Commit timestamp (timezone-aware)
        branch: Branch name
        files_changed: List of changed file paths
        diff_content: Diff content for training
        is_merged: True if this is a merge commit
        has_tests: True if commit modifies test files
        is_reverted: True if this is a revert commit
        ci_passed: True/False/None for CI status
        weight: Computed training weight (0.1-infinity)
        weight_breakdown: Components of weight computation
    """
    sha: str
    message: str
    author: str
    timestamp: datetime
    branch: str
    files_changed: List[str]
    diff_content: str

    # Quality signals
    is_merged: bool = False
    has_tests: bool = False
    is_reverted: bool = False
    ci_passed: Optional[bool] = None

    # Computed weight
    weight: float = 1.0
    weight_breakdown: Dict[str, float] = field(default_factory=dict)


class GitHistoryTrainer:
    """
    Trains SparkSLM models from git commit history with quality weighting.

    The trainer assigns weights to commits based on branch type, quality signals,
    and temporal decay. This allows the statistical model to learn more from
    high-quality, recent commits while still incorporating historical patterns.

    Branch Weights:
        main/master: 1.0 (production code)
        release/hotfix: 0.9 (stable branches)
        develop: 0.8 (integration branch)
        feature: 0.6 (work in progress)
        claude/*: 0.4 (AI-generated, needs review)
        unknown: 0.5 (default for unrecognized branches)

    Quality Multipliers:
        merged: 1.2× (code review passed)
        tested: 1.1× (includes test changes)
        ci_passed: 1.1× (automated checks passed)
        reverted: 0.1× (code was problematic)

    Temporal Decay:
        Uses exponential decay: exp(-log(2) × age / half_life)
        Default half-life: 30 days (commits decay to 50% weight after 1 month)

    Example:
        # Create trainer with 30-day half-life
        trainer = GitHistoryTrainer(temporal_half_life_days=30.0)

        # Get commits from git (stub for now)
        commits = list(trainer.iter_commits(branches=['main']))

        # Prepare weighted training data
        documents, weights = trainer.prepare_training_data(commits)

        # Train model with weights
        model = NGramModel(n=3)
        model.train_weighted(documents, weights)
    """

    # Branch weight tiers
    BRANCH_WEIGHTS = {
        'main': 1.0,
        'master': 1.0,
        'develop': 0.8,
        'release': 0.9,
        'hotfix': 0.9,
        'feature': 0.6,
        'claude': 0.4,  # claude/* branches
    }

    # Quality signal multipliers
    QUALITY_MULTIPLIERS = {
        'merged': 1.2,
        'tested': 1.1,
        'ci_passed': 1.1,
        'reverted': 0.1,
    }

    def __init__(
        self,
        repo_path: str = '.',
        temporal_half_life_days: float = 30.0,
        min_weight: float = 0.1,
        deduplicate: bool = True
    ):
        """
        Initialize GitHistoryTrainer.

        Args:
            repo_path: Path to git repository (default: current directory)
            temporal_half_life_days: Days for weight to decay to 50% (default: 30)
            min_weight: Minimum weight floor (default: 0.1)
            deduplicate: Skip duplicate commits by SHA (default: True)
        """
        self.repo_path = repo_path
        self.half_life = temporal_half_life_days
        self.min_weight = min_weight
        self._seen_shas: Set[str] = set() if deduplicate else None

    def get_branch_weight(self, branch: str) -> float:
        """
        Compute weight based on branch name.

        Branch names are matched case-insensitively. Supports both exact
        matches (e.g., "main") and prefix matches (e.g., "feature/auth").

        Args:
            branch: Branch name

        Returns:
            Weight between 0.4 and 1.0

        Example:
            >>> trainer.get_branch_weight('main')
            1.0
            >>> trainer.get_branch_weight('feature/auth')
            0.6
            >>> trainer.get_branch_weight('claude/fix-bug')
            0.4
        """
        branch_lower = branch.lower()

        # Check for exact matches
        for key, weight in self.BRANCH_WEIGHTS.items():
            if branch_lower == key or branch_lower.startswith(f'{key}/'):
                return weight

        # Check for prefix matches
        if branch_lower.startswith('claude/'):
            return self.BRANCH_WEIGHTS['claude']
        if branch_lower.startswith('feature/'):
            return self.BRANCH_WEIGHTS['feature']

        return 0.5  # Unknown branches get medium weight

    def compute_temporal_decay(
        self,
        commit_time: datetime,
        reference_time: datetime = None
    ) -> float:
        """
        Compute temporal decay multiplier using exponential decay.

        Uses exponential decay formula:
            decay = exp(-log(2) × age_days / half_life)

        This ensures that commits at the half-life age have exactly 50% weight,
        and decay is smooth and continuous.

        Args:
            commit_time: When the commit was made (timezone-aware)
            reference_time: Reference time for age calculation (default: now)

        Returns:
            Decay multiplier between min_weight and 1.0

        Example:
            >>> trainer = GitHistoryTrainer(temporal_half_life_days=30.0)
            >>> now = datetime.now(timezone.utc)
            >>> month_ago = now - timedelta(days=30)
            >>> trainer.compute_temporal_decay(month_ago, now)
            0.5  # Half-life decay
        """
        if reference_time is None:
            reference_time = datetime.now(timezone.utc)

        age_days = (reference_time - commit_time).total_seconds() / 86400
        decay = math.exp(-math.log(2) * age_days / self.half_life)
        return max(decay, self.min_weight)

    def compute_weight(
        self,
        commit: WeightedCommit,
        reference_time: datetime = None
    ) -> float:
        """
        Compute final weight for a commit.

        The final weight is computed as:
            weight = branch_weight × quality_multipliers × temporal_decay

        The weight_breakdown dictionary is updated with individual components.

        Args:
            commit: Commit to weight (modified in-place)
            reference_time: Reference time for temporal decay (default: now)

        Returns:
            Final computed weight (stored in commit.weight)

        Example:
            >>> commit = WeightedCommit(
            ...     sha='abc123',
            ...     message='Merge PR #42',
            ...     author='dev',
            ...     timestamp=datetime.now(timezone.utc),
            ...     branch='main',
            ...     files_changed=['src/auth.py', 'tests/test_auth.py'],
            ...     diff_content='...',
            ...     is_merged=True,
            ...     has_tests=True,
            ... )
            >>> trainer.compute_weight(commit)
            1.32  # 1.0 × 1.2 × 1.1 × 1.0 (no decay)
        """
        breakdown = {}

        # Base weight from branch
        base = self.get_branch_weight(commit.branch)
        breakdown['branch'] = base

        # Quality multipliers
        multiplier = 1.0
        if commit.is_merged:
            multiplier *= self.QUALITY_MULTIPLIERS['merged']
            breakdown['merged'] = self.QUALITY_MULTIPLIERS['merged']
        if commit.has_tests:
            multiplier *= self.QUALITY_MULTIPLIERS['tested']
            breakdown['tested'] = self.QUALITY_MULTIPLIERS['tested']
        if commit.ci_passed:
            multiplier *= self.QUALITY_MULTIPLIERS['ci_passed']
            breakdown['ci_passed'] = self.QUALITY_MULTIPLIERS['ci_passed']
        if commit.is_reverted:
            multiplier *= self.QUALITY_MULTIPLIERS['reverted']
            breakdown['reverted'] = self.QUALITY_MULTIPLIERS['reverted']

        # Temporal decay
        decay = self.compute_temporal_decay(commit.timestamp, reference_time)
        breakdown['temporal'] = decay

        final_weight = base * multiplier * decay
        commit.weight = max(final_weight, self.min_weight)
        commit.weight_breakdown = breakdown

        return commit.weight

    def is_duplicate(self, sha: str) -> bool:
        """
        Check if we've already seen this commit.

        Deduplication is only active if deduplicate=True was passed to __init__.
        Otherwise, always returns False.

        Args:
            sha: Commit hash

        Returns:
            True if this is a duplicate commit, False otherwise
        """
        if self._seen_shas is None:
            return False
        if sha in self._seen_shas:
            return True
        self._seen_shas.add(sha)
        return False

    def detect_quality_signals(self, commit: WeightedCommit) -> None:
        """
        Detect quality signals from commit metadata.

        Updates the commit object in-place with detected signals:
        - has_tests: True if any file path contains test patterns
        - is_reverted: True if message indicates a revert
        - is_merged: True if message indicates a merge

        Args:
            commit: Commit to analyze (modified in-place)

        Test Detection:
            Looks for these patterns in file paths:
            - test_*.py, *_test.py
            - tests/
            - spec_*.py, *_spec.py

        Revert Detection:
            Looks for "revert" + ("this reverts" or "reverts commit")

        Merge Detection:
            Looks for message starting with "Merge " or containing
            "merge pull request"
        """
        msg_lower = commit.message.lower()

        # Detect if commit adds/modifies tests
        test_indicators = ['test_', '_test.py', 'tests/', 'spec_', '_spec.']
        commit.has_tests = any(
            ind in f.lower()
            for f in commit.files_changed
            for ind in test_indicators
        )

        # Detect reverted commits
        commit.is_reverted = (
            'revert' in msg_lower and
            ('this reverts' in msg_lower or 'reverts commit' in msg_lower)
        )

        # Merged detection (from merge commits or message)
        commit.is_merged = (
            commit.message.startswith('Merge ') or
            'merge pull request' in msg_lower
        )

    def iter_commits(
        self,
        branches: List[str] = None,
        since: datetime = None,
        until: datetime = None,
        max_commits: int = None
    ) -> Iterator[WeightedCommit]:
        """
        Iterate over commits from git history.

        This is a stub implementation for now. The actual git integration
        will be added in a separate task using subprocess to call git log.

        Args:
            branches: Branch names to include (None = all branches)
            since: Only commits after this time
            until: Only commits before this time
            max_commits: Maximum number of commits to return

        Returns:
            Iterator of WeightedCommit objects

        Future Implementation:
            Will use subprocess to run:
            git log --all --format='%H|%s|%an|%at|%D' --numstat
        """
        # Stub implementation - returns empty iterator
        # Actual git integration will be added in separate task
        return iter([])

    def prepare_training_data(
        self,
        commits: List[WeightedCommit]
    ) -> Tuple[List[str], List[float]]:
        """
        Prepare commits for weighted training.

        For each commit:
        1. Check for duplicates (skip if seen before)
        2. Detect quality signals (tests, merges, reverts)
        3. Compute weight
        4. Combine message + diff into training document

        Args:
            commits: List of commits to prepare

        Returns:
            Tuple of (documents, weights) for NGramModel.train_weighted()
            - documents: List of strings (message + diff)
            - weights: List of floats (one per document)

        Example:
            >>> trainer = GitHistoryTrainer()
            >>> commits = [...]
            >>> documents, weights = trainer.prepare_training_data(commits)
            >>> model = NGramModel(n=3)
            >>> model.train_weighted(documents, weights)
        """
        documents = []
        weights = []

        for commit in commits:
            if self.is_duplicate(commit.sha):
                continue

            # Detect quality signals
            self.detect_quality_signals(commit)

            # Compute weight
            self.compute_weight(commit)

            # Combine message and diff for training
            doc = f"{commit.message}\n{commit.diff_content}"
            documents.append(doc)
            weights.append(commit.weight)

        return documents, weights
