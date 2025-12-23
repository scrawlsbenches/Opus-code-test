"""
Co-Change Model
===============

Learn file co-occurrence patterns from git history.
Predicts which files are likely to change together based on historical patterns.

Uses temporal weighting with exponential decay to prioritize recent changes.
Half-life of ~69 days means older changes contribute less.

Example:
    >>> model = CoChangeModel()
    >>> model.add_commit('abc123', ['auth.py', 'login.py', 'tests/test_auth.py'])
    >>> predictions = model.predict(['auth.py'], top_n=5)
    >>> predictions[0]
    ('login.py', 0.85)  # (file, confidence)
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional


@dataclass
class CoChangeEdge:
    """
    Represents a co-change relationship between two files.

    Attributes:
        source_file: Source file path
        target_file: Target file path
        co_change_count: Number of times files changed together
        weighted_score: Score with temporal decay applied
        confidence: Normalized probability (0-1)
        last_co_change: Timestamp of most recent co-change
        commits: List of commit SHAs where co-change occurred
    """
    source_file: str
    target_file: str
    co_change_count: int = 0
    weighted_score: float = 0.0
    confidence: float = 0.0
    last_co_change: datetime = field(default_factory=datetime.now)
    commits: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            'source_file': self.source_file,
            'target_file': self.target_file,
            'co_change_count': self.co_change_count,
            'weighted_score': self.weighted_score,
            'confidence': self.confidence,
            'last_co_change': self.last_co_change.isoformat(),
            'commits': self.commits
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'CoChangeEdge':
        """Create from JSON dict."""
        data = data.copy()
        data['last_co_change'] = datetime.fromisoformat(data['last_co_change'])
        return cls(**data)


@dataclass
class Commit:
    """
    Represents a git commit for co-change analysis.

    Attributes:
        sha: Commit SHA hash
        timestamp: When the commit was made
        files: List of file paths changed in this commit
        message: Optional commit message
    """
    sha: str
    timestamp: datetime
    files: List[str]
    message: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            'sha': self.sha,
            'timestamp': self.timestamp.isoformat(),
            'files': self.files,
            'message': self.message
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Commit':
        """Create from JSON dict."""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class CoChangeModel:
    """
    Co-change model for predicting related file changes.

    Learns from git commit history to predict which files are likely to change
    together. Uses temporal weighting with exponential decay (half-life ~69 days).

    Example:
        >>> model = CoChangeModel()
        >>> model.add_commit('abc123', ['auth.py', 'login.py'])
        >>> model.add_commit('def456', ['auth.py', 'tests/test_auth.py'])
        >>> predictions = model.predict(['auth.py'], top_n=5)
        >>> print(predictions)
        [('login.py', 0.52), ('tests/test_auth.py', 0.48)]
    """

    def __init__(self, decay_lambda: float = 0.01):
        """
        Initialize co-change model.

        Args:
            decay_lambda: Temporal decay parameter (default: 0.01).
                         Higher values decay faster. 0.01 gives ~69 day half-life.
        """
        self._edges: Dict[Tuple[str, str], CoChangeEdge] = {}
        self._file_index: Dict[str, Set[str]] = {}
        self._commits: Dict[str, Commit] = {}
        self._decay_lambda = decay_lambda
        self._dirty = False  # Track if normalization needed

    def add_commit(
        self,
        sha: str,
        files: List[str],
        timestamp: Optional[datetime] = None,
        message: Optional[str] = None
    ) -> None:
        """
        Add a commit to the co-change model.

        Args:
            sha: Commit SHA hash
            files: List of file paths changed in this commit
            timestamp: When the commit was made (default: now)
            message: Optional commit message
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Store commit (idempotent - ignore if already exists)
        if sha in self._commits:
            return

        commit = Commit(sha=sha, timestamp=timestamp, files=files, message=message)
        self._commits[sha] = commit

        # Update co-change edges for all file pairs
        for i, file_a in enumerate(files):
            for file_b in files[i+1:]:
                # Ensure consistent ordering (alphabetical) - don't modify loop vars
                key_a, key_b = (file_b, file_a) if file_a > file_b else (file_a, file_b)
                self._update_edge(key_a, key_b, sha, timestamp)

        # Mark as needing normalization (deferred to prediction time)
        self._dirty = True

    def add_commits_batch(self, commits: List[Commit]) -> None:
        """
        Add multiple commits in batch (optimized - normalizes once at end).

        Args:
            commits: List of Commit objects to add
        """
        for commit in commits:
            # Inline the add logic without triggering normalize each time
            if commit.sha in self._commits:
                continue

            self._commits[commit.sha] = commit

            for i, file_a in enumerate(commit.files):
                for file_b in commit.files[i+1:]:
                    key_a, key_b = (file_b, file_a) if file_a > file_b else (file_a, file_b)
                    self._update_edge(key_a, key_b, commit.sha, commit.timestamp)

        # Mark dirty once at the end
        if commits:
            self._dirty = True

    def _ensure_normalized(self) -> None:
        """Normalize confidence scores if needed (lazy evaluation)."""
        if self._dirty:
            self._normalize_confidence()
            self._dirty = False

    def predict(
        self,
        seed_files: List[str],
        top_n: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Predict which files are likely to change given seed files.

        Args:
            seed_files: Files that have changed or will change
            top_n: Number of predictions to return

        Returns:
            List of (file_path, confidence) tuples sorted by confidence
        """
        if not seed_files:
            return []

        # Ensure scores are normalized before prediction
        self._ensure_normalized()

        # Aggregate scores from all seed files
        candidates: Dict[str, float] = {}

        for seed_file in seed_files:
            related = self._file_index.get(seed_file, set())
            for related_file in related:
                score = self.get_co_change_score(seed_file, related_file)
                candidates[related_file] = candidates.get(related_file, 0.0) + score

        # Remove seed files from candidates
        for seed_file in seed_files:
            candidates.pop(seed_file, None)

        # Normalize aggregated scores to sum to <= 1
        total = sum(candidates.values())
        if total > 0:
            candidates = {f: score / total for f, score in candidates.items()}

        # Return top N sorted by confidence
        sorted_candidates = sorted(
            candidates.items(),
            key=lambda x: -x[1]
        )
        return sorted_candidates[:top_n]

    def get_co_change_score(self, file_a: str, file_b: str) -> float:
        """
        Get co-change score between two files.

        Args:
            file_a: First file path
            file_b: Second file path

        Returns:
            Confidence score (0-1), or 0.0 if no relationship
        """
        # Ensure scores are normalized
        self._ensure_normalized()

        # Ensure consistent ordering
        if file_a > file_b:
            file_a, file_b = file_b, file_a

        edge = self._edges.get((file_a, file_b))
        return edge.confidence if edge else 0.0

    def get_edges_for_file(self, file: str) -> List[CoChangeEdge]:
        """
        Get all co-change edges for a file.

        Args:
            file: File path

        Returns:
            List of CoChangeEdge objects
        """
        # Ensure scores are normalized
        self._ensure_normalized()

        edges = []
        related = self._file_index.get(file, set())

        for related_file in related:
            # Ensure consistent ordering
            file_a, file_b = (file, related_file) if file < related_file else (related_file, file)
            edge = self._edges.get((file_a, file_b))
            if edge:
                edges.append(edge)

        return edges

    def prune_old_edges(self, min_score: float = 0.01) -> int:
        """
        Remove weak edges with low weighted scores.

        Args:
            min_score: Minimum weighted score to keep

        Returns:
            Number of edges removed
        """
        to_remove = [
            key for key, edge in self._edges.items()
            if edge.weighted_score < min_score
        ]

        for key in to_remove:
            edge = self._edges.pop(key)
            # Update file index
            if edge.source_file in self._file_index:
                self._file_index[edge.source_file].discard(edge.target_file)
            if edge.target_file in self._file_index:
                self._file_index[edge.target_file].discard(edge.source_file)

        # Mark as needing re-normalization after pruning
        if to_remove:
            self._dirty = True

        return len(to_remove)

    def to_dict(self) -> dict:
        """
        Convert model to JSON-serializable dict.

        Returns:
            Dictionary with all model state
        """
        return {
            'decay_lambda': self._decay_lambda,
            'edges': {
                f"{k[0]}::{k[1]}": v.to_dict()
                for k, v in self._edges.items()
            },
            'commits': {
                sha: commit.to_dict()
                for sha, commit in self._commits.items()
            }
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'CoChangeModel':
        """
        Create model from JSON dict.

        Args:
            data: Dictionary from to_dict()

        Returns:
            CoChangeModel instance
        """
        model = cls(decay_lambda=data['decay_lambda'])

        # Restore edges
        for key_str, edge_data in data['edges'].items():
            file_a, file_b = key_str.split('::', 1)
            edge = CoChangeEdge.from_dict(edge_data)
            model._edges[(file_a, file_b)] = edge

            # Update file index
            if file_a not in model._file_index:
                model._file_index[file_a] = set()
            if file_b not in model._file_index:
                model._file_index[file_b] = set()
            model._file_index[file_a].add(file_b)
            model._file_index[file_b].add(file_a)

        # Restore commits
        for sha, commit_data in data['commits'].items():
            model._commits[sha] = Commit.from_dict(commit_data)

        # Loaded data is already normalized
        model._dirty = False

        return model

    def _compute_temporal_weight(self, timestamp: datetime) -> float:
        """
        Compute weight using exponential decay.

        Half-life of ~69 days with default lambda=0.01.

        Args:
            timestamp: When the commit occurred

        Returns:
            Weight in range (0, 1]
        """
        now = datetime.now()
        age_days = (now - timestamp).total_seconds() / 86400
        return math.exp(-self._decay_lambda * age_days)

    def _update_edge(
        self,
        file_a: str,
        file_b: str,
        sha: str,
        timestamp: datetime
    ) -> None:
        """
        Update or create co-change edge between two files.

        Args:
            file_a: First file (must be <= file_b alphabetically)
            file_b: Second file
            sha: Commit SHA
            timestamp: Commit timestamp
        """
        key = (file_a, file_b)
        weight = self._compute_temporal_weight(timestamp)

        if key not in self._edges:
            # Create new edge
            self._edges[key] = CoChangeEdge(
                source_file=file_a,
                target_file=file_b,
                co_change_count=1,
                weighted_score=weight,
                confidence=0.0,
                last_co_change=timestamp,
                commits=[sha]
            )

            # Update file index
            if file_a not in self._file_index:
                self._file_index[file_a] = set()
            if file_b not in self._file_index:
                self._file_index[file_b] = set()
            self._file_index[file_a].add(file_b)
            self._file_index[file_b].add(file_a)
        else:
            # Update existing edge
            edge = self._edges[key]
            edge.co_change_count += 1
            edge.weighted_score += weight
            edge.last_co_change = max(edge.last_co_change, timestamp)
            if sha not in edge.commits:
                edge.commits.append(sha)

    def _normalize_confidence(self) -> None:
        """
        Normalize weighted scores to confidence probabilities.

        For each file, normalizes all its edge weights so they sum to 1.
        This converts raw weighted scores into conditional probabilities.
        """
        # Group edges by source file
        file_totals: Dict[str, float] = {}

        for (file_a, file_b), edge in self._edges.items():
            file_totals[file_a] = file_totals.get(file_a, 0.0) + edge.weighted_score
            file_totals[file_b] = file_totals.get(file_b, 0.0) + edge.weighted_score

        # Normalize each edge
        for (file_a, file_b), edge in self._edges.items():
            total_a = file_totals.get(file_a, 0.0)
            total_b = file_totals.get(file_b, 0.0)

            # Average of both directions (symmetric)
            if total_a > 0 and total_b > 0:
                conf_a = edge.weighted_score / total_a
                conf_b = edge.weighted_score / total_b
                edge.confidence = (conf_a + conf_b) / 2.0
            elif total_a > 0:
                edge.confidence = edge.weighted_score / total_a
            elif total_b > 0:
                edge.confidence = edge.weighted_score / total_b
            else:
                edge.confidence = 0.0

    def __repr__(self) -> str:
        return (
            f"CoChangeModel(edges={len(self._edges)}, "
            f"files={len(self._file_index)}, "
            f"commits={len(self._commits)}, "
            f"lambda={self._decay_lambda})"
        )
