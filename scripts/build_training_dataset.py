#!/usr/bin/env python3
"""
Build Training Dataset from Git History

Combines the commit data collection with weighted scoring to create
a preprocessed training dataset for nanoGPT or other language models.

Features:
- Loads commits from .git-ml/tracked/commits.jsonl
- Applies weighted scoring (branch, quality signals, temporal decay)
- Fetches diffs for code context
- Creates train/val/test splits
- Exports as JSONL for easy loading

Usage:
    # Build full dataset with defaults
    python scripts/build_training_dataset.py

    # Limit to recent commits
    python scripts/build_training_dataset.py --max-commits 1000

    # Include full diffs (larger but richer)
    python scripts/build_training_dataset.py --include-diffs

    # Custom output path
    python scripts/build_training_dataset.py --output datasets/training.jsonl

    # Show stats only (no export)
    python scripts/build_training_dataset.py --stats-only
"""

import argparse
import json
import logging
import math
import random
import subprocess
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
COMMITS_FILE = PROJECT_ROOT / ".git-ml" / "tracked" / "commits.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets" / "git_training_data.jsonl"


@dataclass
class WeightedCommit:
    """A commit with computed training weight."""
    hash: str
    message: str
    author: str
    timestamp: str
    branch: str
    files_changed: List[str]
    insertions: int
    deletions: int
    hour_of_day: int
    day_of_week: str
    is_merge: bool = False
    is_initial: bool = False

    # Quality signals (detected)
    has_tests: bool = False
    is_reverted: bool = False

    # Diff content (optional)
    diff_content: str = ""

    # Computed weight
    weight: float = 1.0
    weight_breakdown: Dict[str, float] = field(default_factory=dict)


class DatasetBuilder:
    """
    Builds weighted training datasets from git history.

    Weight Formula:
        final_weight = branch_weight × quality_multipliers × temporal_decay

    Branch Weights:
        main/master: 1.0 (production code)
        release/hotfix: 0.9 (stable)
        develop: 0.8 (integration)
        feature: 0.6 (WIP)
        claude/*: 0.4 (AI-generated)

    Quality Multipliers:
        has_tests: 1.1× (includes test changes)
        is_merge: 1.2× (code reviewed)
        is_reverted: 0.1× (problematic)
    """

    BRANCH_WEIGHTS = {
        'main': 1.0,
        'master': 1.0,
        'develop': 0.8,
        'release': 0.9,
        'hotfix': 0.9,
        'feature': 0.6,
        'claude': 0.4,
    }

    QUALITY_MULTIPLIERS = {
        'merged': 1.2,
        'tested': 1.1,
        'reverted': 0.1,
    }

    def __init__(
        self,
        temporal_half_life_days: float = 30.0,
        min_weight: float = 0.1,
        include_diffs: bool = False,
        max_diff_lines: int = 500,
    ):
        self.half_life = temporal_half_life_days
        self.min_weight = min_weight
        self.include_diffs = include_diffs
        self.max_diff_lines = max_diff_lines
        self.reference_time = datetime.now(timezone.utc)

    def get_branch_weight(self, branch: str) -> float:
        """Compute weight based on branch name."""
        branch_lower = branch.lower()

        for key, weight in self.BRANCH_WEIGHTS.items():
            if branch_lower == key or branch_lower.startswith(f'{key}/'):
                return weight

        if branch_lower.startswith('claude/'):
            return self.BRANCH_WEIGHTS['claude']
        if branch_lower.startswith('feature/'):
            return self.BRANCH_WEIGHTS['feature']

        return 0.5  # Unknown branches

    def compute_temporal_decay(self, timestamp_str: str) -> float:
        """Compute temporal decay multiplier."""
        try:
            # Parse timestamp (format: "2025-12-27 02:08:07 +0000")
            dt = datetime.strptime(timestamp_str[:19], "%Y-%m-%d %H:%M:%S")
            dt = dt.replace(tzinfo=timezone.utc)
        except (ValueError, IndexError):
            return 1.0  # Can't parse, no decay

        age_days = (self.reference_time - dt).total_seconds() / 86400
        decay = math.exp(-math.log(2) * age_days / self.half_life)
        return max(decay, self.min_weight)

    def detect_quality_signals(self, commit: WeightedCommit) -> None:
        """Detect quality signals from commit metadata."""
        msg_lower = commit.message.lower()

        # Test detection
        test_indicators = ['test_', '_test.py', 'tests/', 'spec_', '_spec.']
        commit.has_tests = any(
            ind in f.lower()
            for f in commit.files_changed
            for ind in test_indicators
        )

        # Revert detection
        commit.is_reverted = (
            'revert' in msg_lower and
            ('this reverts' in msg_lower or 'reverts commit' in msg_lower)
        )

    def compute_weight(self, commit: WeightedCommit) -> float:
        """Compute final weight for a commit."""
        breakdown = {}

        # Base weight from branch
        base = self.get_branch_weight(commit.branch)
        breakdown['branch'] = base

        # Quality multipliers
        multiplier = 1.0
        if commit.is_merge:
            multiplier *= self.QUALITY_MULTIPLIERS['merged']
            breakdown['merged'] = self.QUALITY_MULTIPLIERS['merged']
        if commit.has_tests:
            multiplier *= self.QUALITY_MULTIPLIERS['tested']
            breakdown['tested'] = self.QUALITY_MULTIPLIERS['tested']
        if commit.is_reverted:
            multiplier *= self.QUALITY_MULTIPLIERS['reverted']
            breakdown['reverted'] = self.QUALITY_MULTIPLIERS['reverted']

        # Temporal decay
        decay = self.compute_temporal_decay(commit.timestamp)
        breakdown['temporal'] = round(decay, 4)

        final_weight = base * multiplier * decay
        commit.weight = max(final_weight, self.min_weight)
        commit.weight_breakdown = breakdown

        return commit.weight

    def fetch_diff(self, commit_hash: str) -> str:
        """Fetch diff content for a commit."""
        try:
            result = subprocess.run(
                ["git", "show", "--format=", "-U3", commit_hash],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(PROJECT_ROOT)
            )
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                if len(lines) > self.max_diff_lines:
                    lines = lines[:self.max_diff_lines] + ['... (truncated)']
                return '\n'.join(lines)
        except Exception as e:
            logger.debug(f"Could not fetch diff for {commit_hash[:8]}: {e}")
        return ""

    def load_commits(self, max_commits: Optional[int] = None) -> List[WeightedCommit]:
        """Load commits from the JSONL file."""
        if not COMMITS_FILE.exists():
            logger.error(f"Commits file not found: {COMMITS_FILE}")
            return []

        commits = []
        with open(COMMITS_FILE, 'r') as f:
            for i, line in enumerate(f):
                if max_commits and i >= max_commits:
                    break
                try:
                    data = json.loads(line.strip())
                    commit = WeightedCommit(
                        hash=data['hash'],
                        message=data['message'],
                        author=data['author'],
                        timestamp=data['timestamp'],
                        branch=data['branch'],
                        files_changed=data['files_changed'],
                        insertions=data.get('insertions', 0),
                        deletions=data.get('deletions', 0),
                        hour_of_day=data.get('hour_of_day', 12),
                        day_of_week=data.get('day_of_week', 'Unknown'),
                        is_merge=data.get('is_merge', False),
                        is_initial=data.get('is_initial', False),
                    )
                    commits.append(commit)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Skipping malformed line {i}: {e}")

        logger.info(f"Loaded {len(commits)} commits from {COMMITS_FILE}")
        return commits

    def process_commits(
        self,
        commits: List[WeightedCommit],
        progress_interval: int = 100
    ) -> List[WeightedCommit]:
        """Process commits: detect signals, compute weights, optionally fetch diffs."""
        for i, commit in enumerate(commits):
            # Detect quality signals
            self.detect_quality_signals(commit)

            # Compute weight
            self.compute_weight(commit)

            # Optionally fetch diff
            if self.include_diffs:
                commit.diff_content = self.fetch_diff(commit.hash)

            if (i + 1) % progress_interval == 0:
                logger.info(f"Processed {i + 1}/{len(commits)} commits...")

        return commits

    def create_training_example(self, commit: WeightedCommit) -> Dict:
        """Create a training example from a commit."""
        # Format: instruction-style for fine-tuning
        example = {
            "text": f"### Commit Message\n{commit.message}\n\n### Files Changed\n" +
                    "\n".join(f"- {f}" for f in commit.files_changed[:20]),
            "weight": round(commit.weight, 4),
            "metadata": {
                "hash": commit.hash[:8],
                "branch": commit.branch,
                "author": commit.author,
                "timestamp": commit.timestamp,
                "insertions": commit.insertions,
                "deletions": commit.deletions,
                "is_merge": commit.is_merge,
                "has_tests": commit.has_tests,
                "weight_breakdown": commit.weight_breakdown,
            }
        }

        # Add diff if available
        if commit.diff_content:
            example["text"] += f"\n\n### Diff\n```\n{commit.diff_content}\n```"

        return example

    def split_dataset(
        self,
        examples: List[Dict],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split dataset into train/val/test sets."""
        random.seed(seed)
        shuffled = examples.copy()
        random.shuffle(shuffled)

        n = len(shuffled)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train = shuffled[:train_end]
        val = shuffled[train_end:val_end]
        test = shuffled[val_end:]

        return train, val, test

    def compute_stats(self, commits: List[WeightedCommit]) -> Dict:
        """Compute dataset statistics."""
        if not commits:
            return {}

        weights = [c.weight for c in commits]
        branches = {}
        for c in commits:
            branch_type = c.branch.split('/')[0] if '/' in c.branch else c.branch
            branches[branch_type] = branches.get(branch_type, 0) + 1

        return {
            "total_commits": len(commits),
            "weight_stats": {
                "min": round(min(weights), 4),
                "max": round(max(weights), 4),
                "mean": round(sum(weights) / len(weights), 4),
                "total": round(sum(weights), 2),
            },
            "quality_signals": {
                "with_tests": sum(1 for c in commits if c.has_tests),
                "merges": sum(1 for c in commits if c.is_merge),
                "reverts": sum(1 for c in commits if c.is_reverted),
            },
            "branch_distribution": dict(sorted(
                branches.items(), key=lambda x: -x[1]
            )[:10]),
        }

    def export(
        self,
        examples: List[Dict],
        output_path: Path,
        split: bool = True
    ) -> Dict[str, int]:
        """Export examples to JSONL files."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if split:
            train, val, test = self.split_dataset(examples)
            counts = {"train": len(train), "val": len(val), "test": len(test)}

            # Write split files
            for name, data in [("train", train), ("val", val), ("test", test)]:
                split_path = output_path.parent / f"{output_path.stem}_{name}.jsonl"
                with open(split_path, 'w') as f:
                    for ex in data:
                        f.write(json.dumps(ex) + '\n')
                logger.info(f"Wrote {len(data)} examples to {split_path}")
        else:
            with open(output_path, 'w') as f:
                for ex in examples:
                    f.write(json.dumps(ex) + '\n')
            counts = {"total": len(examples)}
            logger.info(f"Wrote {len(examples)} examples to {output_path}")

        return counts


def main():
    parser = argparse.ArgumentParser(
        description="Build weighted training dataset from git history"
    )
    parser.add_argument(
        "--max-commits", type=int, default=None,
        help="Maximum number of commits to process (default: all)"
    )
    parser.add_argument(
        "--include-diffs", action="store_true",
        help="Include diff content (slower but richer data)"
    )
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_OUTPUT),
        help=f"Output path (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--half-life", type=float, default=30.0,
        help="Temporal decay half-life in days (default: 30)"
    )
    parser.add_argument(
        "--no-split", action="store_true",
        help="Don't split into train/val/test"
    )
    parser.add_argument(
        "--stats-only", action="store_true",
        help="Only show statistics, don't export"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("=" * 60)
    print("Building Training Dataset from Git History")
    print("=" * 60)

    # Initialize builder
    builder = DatasetBuilder(
        temporal_half_life_days=args.half_life,
        include_diffs=args.include_diffs,
    )

    # Load commits
    commits = builder.load_commits(max_commits=args.max_commits)
    if not commits:
        print("No commits found!")
        return 1

    # Process commits
    print(f"\nProcessing {len(commits)} commits...")
    commits = builder.process_commits(commits)

    # Compute and show stats
    stats = builder.compute_stats(commits)
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total commits: {stats['total_commits']}")
    print(f"   Weight range: {stats['weight_stats']['min']} - {stats['weight_stats']['max']}")
    print(f"   Mean weight: {stats['weight_stats']['mean']}")
    print(f"   Total weight: {stats['weight_stats']['total']}")
    print(f"\n   Quality signals:")
    print(f"     With tests: {stats['quality_signals']['with_tests']}")
    print(f"     Merges: {stats['quality_signals']['merges']}")
    print(f"     Reverts: {stats['quality_signals']['reverts']}")
    print(f"\n   Top branches:")
    for branch, count in list(stats['branch_distribution'].items())[:5]:
        print(f"     {branch}: {count}")

    if args.stats_only:
        return 0

    # Create training examples
    print(f"\nCreating training examples...")
    examples = [builder.create_training_example(c) for c in commits]

    # Export
    output_path = Path(args.output)
    counts = builder.export(examples, output_path, split=not args.no_split)

    print(f"\n✅ Dataset created!")
    for name, count in counts.items():
        print(f"   {name}: {count} examples")

    return 0


if __name__ == "__main__":
    sys.exit(main())
