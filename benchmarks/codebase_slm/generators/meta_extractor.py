"""
Meta Extractor - Extract patterns from GoT entities and commit history.

Extracts:
- Tasks with status, priority, descriptions
- Decisions with rationale
- Sprints and epics
- Commit messages and file associations
- Edge relationships

Design:
- Lightweight extraction (JSON/JSONL parsing)
- No external dependencies
- Batch processing support
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Any, Iterator
from datetime import datetime


@dataclass
class TaskPattern:
    """Extracted task information."""
    id: str
    title: str
    status: str
    priority: str
    description: str
    category: str


@dataclass
class DecisionPattern:
    """Extracted decision information."""
    id: str
    title: str
    rationale: str
    status: str


@dataclass
class CommitPattern:
    """Extracted commit information."""
    hash: str
    message: str
    files_changed: List[str]
    commit_type: str  # feat, fix, docs, etc.


@dataclass
class EdgePattern:
    """Extracted relationship between entities."""
    source_id: str
    target_id: str
    edge_type: str


@dataclass
class MetaPattern:
    """Container for all metadata patterns."""
    tasks: List[TaskPattern] = field(default_factory=list)
    decisions: List[DecisionPattern] = field(default_factory=list)
    commits: List[CommitPattern] = field(default_factory=list)
    edges: List[EdgePattern] = field(default_factory=list)
    extracted_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'tasks': [asdict(t) for t in self.tasks],
            'decisions': [asdict(d) for d in self.decisions],
            'commits': [asdict(c) for c in self.commits],
            'edges': [asdict(e) for e in self.edges],
            'extracted_at': self.extracted_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetaPattern':
        """Create from dictionary."""
        return cls(
            tasks=[TaskPattern(**t) for t in data.get('tasks', [])],
            decisions=[DecisionPattern(**d) for d in data.get('decisions', [])],
            commits=[CommitPattern(**c) for c in data.get('commits', [])],
            edges=[EdgePattern(**e) for e in data.get('edges', [])],
            extracted_at=data.get('extracted_at', ''),
        )


class MetaExtractor:
    """
    Extract patterns from GoT entities and commit history.

    Usage:
        extractor = MetaExtractor(
            got_dir='.got/entities',
            commits_file='.git-ml/tracked/commits.jsonl'
        )
        patterns = extractor.extract_all()
    """

    # Commit type patterns
    COMMIT_TYPES = {
        'feat': ['feat', 'feature', 'add'],
        'fix': ['fix', 'bugfix', 'bug'],
        'docs': ['docs', 'doc', 'documentation'],
        'refactor': ['refactor', 'restructure', 'reorganize'],
        'test': ['test', 'tests', 'testing'],
        'chore': ['chore', 'update', 'upgrade'],
        'perf': ['perf', 'performance', 'optimize'],
        'style': ['style', 'format', 'formatting'],
    }

    def __init__(
        self,
        got_dir: Optional[Path] = None,
        commits_file: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize the extractor.

        Args:
            got_dir: Directory containing GoT entity JSON files
            commits_file: JSONL file with commit data
            cache_dir: Directory for caching
        """
        self.got_dir = Path(got_dir) if got_dir else Path('.got/entities')
        self.commits_file = Path(commits_file) if commits_file else Path('.git-ml/tracked/commits.jsonl')
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._pattern: Optional[MetaPattern] = None

    def _parse_commit_type(self, message: str) -> str:
        """Extract commit type from message."""
        message_lower = message.lower()

        # Check for conventional commit format (type: or type(scope):)
        if ':' in message:
            prefix = message.split(':')[0].lower()
            # Remove scope if present
            if '(' in prefix:
                prefix = prefix.split('(')[0]

            for commit_type, keywords in self.COMMIT_TYPES.items():
                if prefix in keywords:
                    return commit_type

        # Fallback: check for keywords
        for commit_type, keywords in self.COMMIT_TYPES.items():
            for keyword in keywords:
                if message_lower.startswith(keyword):
                    return commit_type

        return 'other'

    def _extract_tasks(self) -> List[TaskPattern]:
        """Extract task patterns from GoT entities."""
        tasks = []

        if not self.got_dir.exists():
            return tasks

        for entity_file in self.got_dir.glob('T-*.json'):
            try:
                with open(entity_file, 'r') as f:
                    data = json.load(f)

                entity_data = data.get('data', data)

                task = TaskPattern(
                    id=entity_data.get('id', entity_file.stem),
                    title=entity_data.get('title', ''),
                    status=entity_data.get('status', 'unknown'),
                    priority=entity_data.get('priority', 'medium'),
                    description=entity_data.get('description', ''),
                    category=entity_data.get('properties', {}).get('category', 'general'),
                )
                tasks.append(task)
            except (json.JSONDecodeError, KeyError):
                continue

        return tasks

    def _extract_decisions(self) -> List[DecisionPattern]:
        """Extract decision patterns from GoT entities."""
        decisions = []

        if not self.got_dir.exists():
            return decisions

        for entity_file in self.got_dir.glob('D-*.json'):
            try:
                with open(entity_file, 'r') as f:
                    data = json.load(f)

                entity_data = data.get('data', data)

                decision = DecisionPattern(
                    id=entity_data.get('id', entity_file.stem),
                    title=entity_data.get('title', ''),
                    rationale=entity_data.get('rationale', ''),
                    status=entity_data.get('status', 'active'),
                )
                decisions.append(decision)
            except (json.JSONDecodeError, KeyError):
                continue

        return decisions

    def _extract_commits(self, limit: int = 1000) -> List[CommitPattern]:
        """Extract commit patterns from ML commit data."""
        commits = []

        if not self.commits_file.exists():
            return commits

        try:
            with open(self.commits_file, 'r') as f:
                for i, line in enumerate(f):
                    if i >= limit:
                        break

                    try:
                        data = json.loads(line.strip())

                        # Skip merge commits
                        if data.get('is_merge', False):
                            continue

                        commit = CommitPattern(
                            hash=data.get('hash', '')[:8],
                            message=data.get('message', ''),
                            files_changed=data.get('files_changed', []),
                            commit_type=self._parse_commit_type(data.get('message', '')),
                        )
                        commits.append(commit)
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass

        return commits

    def _extract_edges(self) -> List[EdgePattern]:
        """Extract edge patterns from GoT entities."""
        edges = []

        if not self.got_dir.exists():
            return edges

        for entity_file in self.got_dir.glob('E-*.json'):
            try:
                with open(entity_file, 'r') as f:
                    data = json.load(f)

                entity_data = data.get('data', data)

                edge = EdgePattern(
                    source_id=entity_data.get('from_id', ''),
                    target_id=entity_data.get('to_id', ''),
                    edge_type=entity_data.get('edge_type', 'unknown'),
                )
                edges.append(edge)
            except (json.JSONDecodeError, KeyError):
                continue

        return edges

    def extract_all(self, commit_limit: int = 1000) -> MetaPattern:
        """
        Extract all metadata patterns.

        Args:
            commit_limit: Maximum number of commits to process

        Returns:
            MetaPattern containing all extracted patterns
        """
        pattern = MetaPattern(
            tasks=self._extract_tasks(),
            decisions=self._extract_decisions(),
            commits=self._extract_commits(limit=commit_limit),
            edges=self._extract_edges(),
            extracted_at=datetime.utcnow().isoformat(),
        )

        self._pattern = pattern

        # Cache if configured
        if self.cache_dir:
            self._save_cache()

        return pattern

    def _save_cache(self) -> None:
        """Save patterns to cache."""
        if not self.cache_dir or not self._pattern:
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / 'meta_patterns.json'

        with open(cache_file, 'w') as f:
            json.dump(self._pattern.to_dict(), f, indent=2)

    def get_statistics(self) -> Dict[str, int]:
        """Get extraction statistics."""
        if not self._pattern:
            return {'tasks': 0, 'decisions': 0, 'commits': 0, 'edges': 0}

        return {
            'tasks': len(self._pattern.tasks),
            'decisions': len(self._pattern.decisions),
            'commits': len(self._pattern.commits),
            'edges': len(self._pattern.edges),
        }


if __name__ == '__main__':
    cache = Path('benchmarks/codebase_slm/corpus')
    extractor = MetaExtractor(cache_dir=cache)

    print("Extracting metadata patterns...")
    patterns = extractor.extract_all()

    print(f"\nStatistics: {extractor.get_statistics()}")

    if patterns.tasks:
        print(f"\nSample task: {patterns.tasks[0].title}")
    if patterns.commits:
        print(f"Sample commit: {patterns.commits[0].message[:60]}...")
