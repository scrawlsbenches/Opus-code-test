"""
Commit collection module for ML Data Collector

Handles git operations, diff parsing, and commit data collection.
"""

import json
import re
import subprocess
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .config import COMMITS_DIR
from .core import GitCommandError
from .data_classes import CommitContext


logger = logging.getLogger(__name__)


# ============================================================================
# GIT OPERATIONS
# ============================================================================

def run_git(args: List[str], check: bool = True) -> str:
    """Run a git command and return output.

    Args:
        args: Git command arguments (without 'git' prefix)
        check: If True, raise GitCommandError on non-zero exit

    Returns:
        Command stdout stripped of whitespace

    Raises:
        GitCommandError: If check=True and command fails
    """
    result = subprocess.run(
        ["git"] + args,
        capture_output=True,
        text=True,
        cwd=str(Path.cwd())
    )
    if check and result.returncode != 0:
        raise GitCommandError(
            f"git {args[0]} failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout.strip()


def get_last_commit_time() -> Optional[datetime]:
    """Get timestamp of the previous commit."""
    output = run_git(["log", "-2", "--format=%ct"])
    lines = output.strip().split("\n")
    if len(lines) >= 2:
        return datetime.fromtimestamp(int(lines[1]))
    return None


# ============================================================================
# DIFF PARSING
# ============================================================================

def parse_diff_hunks(commit_hash: str, is_merge: bool = False) -> List[Dict]:
    """Parse diff hunks from a commit into structured data.

    Args:
        commit_hash: Git commit hash
        is_merge: If True, use --first-parent to get meaningful diff

    Returns:
        List of hunk dictionaries with file, function, lines, etc.
    """
    # Use -U10 for more context (better for ML training)
    # For merge commits, use --first-parent to get the actual changes
    args = ["show", "--format=", "-U10", commit_hash]
    if is_merge:
        args.insert(2, "--first-parent")

    diff_output = run_git(args, check=False)  # Don't fail on empty diffs

    hunks = []
    current_file = None
    current_hunk = None

    for line in diff_output.split("\n"):
        # New file
        if line.startswith("diff --git"):
            if current_hunk:
                hunks.append(current_hunk)
            match = re.search(r"b/(.+)$", line)
            current_file = match.group(1) if match else "unknown"
            current_hunk = None

        # Hunk header
        elif line.startswith("@@"):
            if current_hunk:
                hunks.append(current_hunk)

            # Parse line numbers
            match = re.search(r"@@ -(\d+)", line)
            start_line = int(match.group(1)) if match else 0

            # Extract function context if present
            func_match = re.search(r"@@ .+ @@ (.+)$", line)
            function = func_match.group(1).strip() if func_match else None

            current_hunk = {
                "file": current_file,
                "function": function,
                "start_line": start_line,
                "lines_added": [],
                "lines_removed": [],
                "context_before": [],
                "context_after": [],
            }

        # Diff content
        elif current_hunk is not None:
            if line.startswith("+") and not line.startswith("+++"):
                current_hunk["lines_added"].append(line[1:])
            elif line.startswith("-") and not line.startswith("---"):
                current_hunk["lines_removed"].append(line[1:])
            elif line.startswith(" "):
                # Context line
                if not current_hunk["lines_added"] and not current_hunk["lines_removed"]:
                    current_hunk["context_before"].append(line[1:])
                else:
                    current_hunk["context_after"].append(line[1:])

    if current_hunk:
        hunks.append(current_hunk)

    # Determine change type for each hunk
    for hunk in hunks:
        if hunk["lines_added"] and not hunk["lines_removed"]:
            hunk["change_type"] = "add"
        elif hunk["lines_removed"] and not hunk["lines_added"]:
            hunk["change_type"] = "delete"
        else:
            hunk["change_type"] = "modify"

    return hunks


# ============================================================================
# COMMIT DATA COLLECTION
# ============================================================================

def collect_commit_data(commit_hash: Optional[str] = None) -> CommitContext:
    """Collect rich context for a commit."""
    if commit_hash is None:
        commit_hash = run_git(["rev-parse", "HEAD"])

    # Basic metadata
    message = run_git(["log", "-1", "--format=%s", commit_hash])
    author = run_git(["log", "-1", "--format=%an", commit_hash])
    timestamp = run_git(["log", "-1", "--format=%ci", commit_hash])
    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"], check=False) or "HEAD"

    # Detect merge/initial commit by counting parents
    parents_output = run_git(["rev-list", "--parents", "-n1", commit_hash])
    parents = parents_output.split()
    parent_count = len(parents) - 1  # First element is the commit itself
    is_merge = parent_count > 1
    is_initial = parent_count == 0

    # Files and stats - for merges, use --first-parent for meaningful diff
    if is_merge:
        files_output = run_git(
            ["diff", "--name-only", f"{commit_hash}^1", commit_hash],
            check=False
        )
        stats = run_git(
            ["diff", "--stat", f"{commit_hash}^1", commit_hash],
            check=False
        )
    else:
        files_output = run_git(["show", "--name-only", "--format=", commit_hash])
        stats = run_git(["show", "--stat", "--format=", commit_hash])

    files_changed = [f for f in files_output.split("\n") if f]

    insertions = 0
    deletions = 0
    if stats:
        match = re.search(r"(\d+) insertion", stats)
        if match:
            insertions = int(match.group(1))
        match = re.search(r"(\d+) deletion", stats)
        if match:
            deletions = int(match.group(1))

    # Temporal context - use commit timestamp, not current time for backfill
    commit_time = run_git(["log", "-1", "--format=%ct", commit_hash])
    try:
        commit_dt = datetime.fromtimestamp(int(commit_time))
    except (ValueError, OSError):
        commit_dt = datetime.now()

    last_commit = get_last_commit_time()
    seconds_since = None
    if last_commit:
        seconds_since = int((commit_dt - last_commit).total_seconds())

    # Parse diff hunks (pass is_merge flag for proper handling)
    hunks = parse_diff_hunks(commit_hash, is_merge=is_merge)

    return CommitContext(
        hash=commit_hash,
        message=message,
        author=author,
        timestamp=timestamp,
        branch=branch,
        files_changed=files_changed,
        insertions=insertions,
        deletions=deletions,
        hunks=hunks,
        hour_of_day=commit_dt.hour,
        day_of_week=commit_dt.strftime("%A"),
        seconds_since_last_commit=seconds_since,
        is_merge=is_merge,
        is_initial=is_initial,
        parent_count=parent_count,
    )


# ============================================================================
# COMMIT FILE OPERATIONS
# ============================================================================

def find_commit_file(commit_hash: str) -> Optional[Path]:
    """Find the data file for a commit by its hash (full or prefix)."""
    if not COMMITS_DIR.exists():
        return None

    # Search for files starting with the commit hash prefix
    for f in COMMITS_DIR.glob(f"{commit_hash[:8]}_*.json"):
        return f

    # Try full hash search if prefix didn't match
    for f in COMMITS_DIR.glob("*.json"):
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
            if data.get('hash', '').startswith(commit_hash):
                return f
        except (json.JSONDecodeError, IOError):
            continue

    return None


def update_commit_ci_result(
    commit_hash: str,
    result: str,
    details: Optional[Dict] = None
) -> bool:
    """Update a commit's CI result.

    Args:
        commit_hash: Full or partial commit hash.
        result: CI result (e.g., "pass", "fail", "error", "pending").
        details: Optional dict with additional CI info (test_count, failures, etc.)

    Returns:
        True if commit was found and updated, False otherwise.
    """
    from .persistence import atomic_write_json

    commit_file = find_commit_file(commit_hash)
    if not commit_file:
        return False

    try:
        with open(commit_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Update CI fields
        data['ci_result'] = result
        if details:
            data['ci_details'] = details
        data['ci_updated_at'] = datetime.now().isoformat()

        atomic_write_json(commit_file, data)
        return True

    except (json.JSONDecodeError, IOError):
        return False


def mark_commit_reverted(commit_hash: str, reverting_commit: Optional[str] = None) -> bool:
    """Mark a commit as reverted.

    Args:
        commit_hash: The commit that was reverted.
        reverting_commit: The commit that performed the revert.

    Returns:
        True if commit was found and updated.
    """
    from .persistence import atomic_write_json

    commit_file = find_commit_file(commit_hash)
    if not commit_file:
        return False

    try:
        with open(commit_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        data['reverted'] = True
        if reverting_commit:
            data['reverted_by'] = reverting_commit
        data['reverted_at'] = datetime.now().isoformat()

        atomic_write_json(commit_file, data)
        return True

    except (json.JSONDecodeError, IOError):
        return False
