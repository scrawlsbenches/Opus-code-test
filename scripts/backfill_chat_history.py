#!/usr/bin/env python3
"""
Backfill Chat History from Git History.

This script safely extracts chat history from different points in git history
without affecting your current branch. The .git-ml/ data is gitignored,
so it accumulates locally regardless of which branch you're on.

SAFE WORKFLOW:
1. Save current branch name
2. Create a temp worktree (doesn't affect your branch)
3. For each commit, checkout and extract chat data
4. Merge extracted data into .git-ml/
5. Return to original branch (already there - worktrees don't change HEAD)

Usage:
    python scripts/backfill_chat_history.py --since "2024-12-01"
    python scripts/backfill_chat_history.py --commits 50
    python scripts/backfill_chat_history.py --branch main --commits 100
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

PROJECT_ROOT = Path(__file__).parent.parent


def run_git(args: List[str], cwd: Optional[Path] = None) -> str:
    """Run a git command and return output."""
    result = subprocess.run(
        ['git'] + args,
        cwd=cwd or PROJECT_ROOT,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Git command failed: {result.stderr}")
    return result.stdout.strip()


def get_current_branch() -> str:
    """Get current branch name."""
    return run_git(['branch', '--show-current'])


def get_commits(branch: str = 'main', since: Optional[str] = None,
                limit: int = 100) -> List[str]:
    """Get list of commit hashes to process."""
    args = ['log', branch, '--format=%H', f'-n{limit}']
    if since:
        args.append(f'--since={since}')
    output = run_git(args)
    return output.split('\n') if output else []


def extract_chat_from_commit(commit_hash: str, worktree_path: Path) -> List[Dict]:
    """Extract chat history from a specific commit using worktree."""
    patterns = []

    try:
        # Checkout the commit in the worktree
        run_git(['checkout', commit_hash], cwd=worktree_path)

        # Look for chat data in various locations
        chat_locations = [
            worktree_path / '.git-ml' / 'tracked' / 'chunked',
            worktree_path / '.git-ml' / 'chats',
        ]

        for location in chat_locations:
            if location.exists():
                for chat_file in location.glob('*.jsonl'):
                    try:
                        with open(chat_file) as f:
                            for line in f:
                                entry = json.loads(line)
                                if entry.get('record_type') == 'chat':
                                    data = entry.get('data', {})
                                    query = data.get('query', '')
                                    response = data.get('response', '')

                                    if query and response and not query.startswith('eJz'):
                                        patterns.append({
                                            'query': query[:500],
                                            'response': response[:2000],
                                            'commit': commit_hash[:8],
                                            'timestamp': data.get('timestamp', ''),
                                            'session_id': data.get('session_id', ''),
                                        })
                    except Exception as e:
                        print(f"  Warning: Could not parse {chat_file}: {e}")

    except Exception as e:
        print(f"  Warning: Could not checkout {commit_hash[:8]}: {e}")

    return patterns


def merge_patterns(existing: List[Dict], new: List[Dict]) -> List[Dict]:
    """Merge new patterns with existing, avoiding duplicates."""
    seen = set()

    # Index existing by query
    for p in existing:
        seen.add(p['query'][:100])

    merged = list(existing)
    added = 0

    for p in new:
        key = p['query'][:100]
        if key not in seen:
            merged.append(p)
            seen.add(key)
            added += 1

    return merged, added


def save_backfilled_data(patterns: List[Dict], output_path: Path):
    """Save backfilled patterns."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for p in patterns:
            f.write(json.dumps(p) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Backfill chat history from git')
    parser.add_argument('--branch', default='main', help='Branch to extract from')
    parser.add_argument('--since', help='Only commits since date (YYYY-MM-DD)')
    parser.add_argument('--commits', type=int, default=50, help='Number of commits')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done')
    args = parser.parse_args()

    print("=" * 60)
    print("CHAT HISTORY BACKFILL")
    print("=" * 60)

    current_branch = get_current_branch()
    print(f"\nCurrent branch: {current_branch}")
    print(f"Target branch: {args.branch}")
    print(f"Commits to process: {args.commits}")
    if args.since:
        print(f"Since: {args.since}")

    # Get commits to process
    commits = get_commits(args.branch, args.since, args.commits)
    print(f"\nFound {len(commits)} commits to process")

    if args.dry_run:
        print("\n[DRY RUN] Would process these commits:")
        for c in commits[:10]:
            print(f"  {c[:8]}")
        if len(commits) > 10:
            print(f"  ... and {len(commits) - 10} more")
        return

    if not commits:
        print("No commits to process")
        return

    # Create temporary worktree (SAFE - doesn't affect current branch)
    worktree_path = PROJECT_ROOT / '.git' / 'worktrees' / 'backfill-temp'

    print("\nCreating temporary worktree...")
    try:
        # Clean up any existing worktree
        if worktree_path.exists():
            run_git(['worktree', 'remove', '--force', str(worktree_path)])

        # Create new worktree
        run_git(['worktree', 'add', '--detach', str(worktree_path), commits[0]])
    except Exception as e:
        print(f"Could not create worktree: {e}")
        print("Falling back to direct checkout method (SAFE - .git-ml is gitignored)")

        # Alternative: just read files directly from git objects
        # This is safer but more complex - for now we'll note it
        print("\nNote: For full safety, consider using git show to read files directly")
        return

    # Process each commit
    all_patterns = []
    print("\nProcessing commits...")

    for i, commit in enumerate(commits):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(commits)}")

        patterns = extract_chat_from_commit(commit, worktree_path)
        all_patterns.extend(patterns)

    # Cleanup worktree
    print("\nCleaning up worktree...")
    try:
        run_git(['worktree', 'remove', '--force', str(worktree_path)])
    except:
        pass

    # Load existing backfilled data
    output_path = PROJECT_ROOT / '.git-ml' / 'backfill' / 'chat_history.jsonl'
    existing = []
    if output_path.exists():
        with open(output_path) as f:
            existing = [json.loads(line) for line in f]

    # Merge and save
    merged, added = merge_patterns(existing, all_patterns)
    save_backfilled_data(merged, output_path)

    print(f"\nResults:")
    print(f"  Existing patterns: {len(existing)}")
    print(f"  New patterns found: {len(all_patterns)}")
    print(f"  Unique new added: {added}")
    print(f"  Total after merge: {len(merged)}")
    print(f"\nSaved to: {output_path}")

    print("\n" + "=" * 60)
    print("BACKFILL COMPLETE")
    print("=" * 60)
    print(f"""
Your current branch ({current_branch}) is unchanged.
The .git-ml/ directory is gitignored, so this data:
- Won't cause merge conflicts
- Persists across branch switches
- Can be regenerated anytime

To use this data in training:
    python -m benchmarks.codebase_slm.data_augmentation

The augmentation pipeline will automatically find .git-ml/backfill/
""")


if __name__ == "__main__":
    main()
