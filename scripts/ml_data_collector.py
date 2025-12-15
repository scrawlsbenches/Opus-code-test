#!/usr/bin/env python3
"""
ML Data Collector for Project-Specific Language Model Training

This module collects enriched data from git commits, chat sessions, and developer
actions to train a micro-model specific to this project.

Usage:
    # Collect commit data (call from git hook)
    python scripts/ml_data_collector.py commit

    # Log a chat session
    python scripts/ml_data_collector.py chat --query "..." --response "..."

    # Show statistics
    python scripts/ml_data_collector.py stats

    # Estimate when training is viable
    python scripts/ml_data_collector.py estimate
"""

import json
import os
import subprocess
import hashlib
import re
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field


class GitCommandError(Exception):
    """Raised when a git command fails."""
    pass

# ============================================================================
# CONFIGURATION
# ============================================================================

ML_DATA_DIR = Path(".git-ml")
COMMITS_DIR = ML_DATA_DIR / "commits"
SESSIONS_DIR = ML_DATA_DIR / "sessions"
CHATS_DIR = ML_DATA_DIR / "chats"
ACTIONS_DIR = ML_DATA_DIR / "actions"

# Training milestones
MILESTONES = {
    "file_prediction": {"commits": 500, "sessions": 100, "chats": 200},
    "commit_messages": {"commits": 2000, "sessions": 500, "chats": 1000},
    "code_suggestions": {"commits": 5000, "sessions": 2000, "chats": 5000},
}

# ============================================================================
# DATA SCHEMAS
# ============================================================================

@dataclass
class DiffHunk:
    """A single diff hunk from a commit."""
    file: str
    function: Optional[str]  # Function/class containing the change
    change_type: str  # add, modify, delete, rename
    start_line: int
    lines_added: List[str]
    lines_removed: List[str]
    context_before: List[str]
    context_after: List[str]


@dataclass
class CommitContext:
    """Rich context captured at commit time."""
    # Git metadata
    hash: str
    message: str
    author: str
    timestamp: str
    branch: str

    # Files changed
    files_changed: List[str]
    insertions: int
    deletions: int

    # Diff structure
    hunks: List[Dict]

    # Temporal context
    hour_of_day: int
    day_of_week: str
    seconds_since_last_commit: Optional[int]

    # Commit type detection
    is_merge: bool = False
    is_initial: bool = False
    parent_count: int = 1

    # Session context (if available)
    session_id: Optional[str] = None
    related_chats: List[str] = field(default_factory=list)

    # Outcome tracking (filled in later)
    ci_result: Optional[str] = None
    reverted: bool = False
    amended: bool = False


@dataclass
class ChatEntry:
    """A query/response pair from a chat session."""
    id: str
    timestamp: str
    session_id: str

    # The conversation
    query: str
    response: str

    # Context
    files_referenced: List[str]
    files_modified: List[str]
    tools_used: List[str]

    # Outcome
    user_feedback: Optional[str] = None  # positive, negative, neutral
    resulted_in_commit: bool = False
    related_commit: Optional[str] = None

    # Metadata
    query_tokens: int = 0
    response_tokens: int = 0
    duration_seconds: Optional[float] = None


@dataclass
class ActionEntry:
    """A discrete action taken during development."""
    id: str
    timestamp: str
    session_id: str

    action_type: str  # search, read, edit, test, commit, etc.
    target: str  # file path, query string, etc.

    # Context
    context: Dict[str, Any] = field(default_factory=dict)

    # Outcome
    success: bool = True
    result_summary: Optional[str] = None


# ============================================================================
# DATA COLLECTION FUNCTIONS
# ============================================================================

def ensure_dirs():
    """Create data directories if they don't exist."""
    for dir_path in [COMMITS_DIR, SESSIONS_DIR, CHATS_DIR, ACTIONS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)


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


def atomic_write_json(filepath: Path, data: dict):
    """Write JSON atomically using temp file + rename.

    This prevents data corruption if the process is interrupted.
    """
    # Write to temp file in same directory (for same-filesystem rename)
    temp_fd, temp_path = tempfile.mkstemp(
        suffix=".tmp",
        prefix=filepath.stem + "_",
        dir=filepath.parent
    )
    try:
        with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        # Atomic rename (on POSIX systems)
        os.replace(temp_path, filepath)
    except Exception:
        # Clean up temp file on failure
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def save_commit_data(context: CommitContext):
    """Save commit context to disk atomically."""
    ensure_dirs()

    # Use full hash + UUID suffix to prevent collisions
    unique_id = uuid.uuid4().hex[:8]
    filename = f"{context.hash[:8]}_{context.timestamp[:10]}_{unique_id}.json"
    filepath = COMMITS_DIR / filename

    atomic_write_json(filepath, asdict(context))
    print(f"Saved commit data to {filepath}")


def generate_chat_id() -> str:
    """Generate unique chat entry ID."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = hashlib.sha256(str(datetime.now().timestamp()).encode()).hexdigest()[:6]
    return f"chat-{timestamp}-{suffix}"


def generate_session_id() -> str:
    """Generate unique session ID."""
    return hashlib.sha256(str(datetime.now().timestamp()).encode()).hexdigest()[:8]


def save_chat_entry(entry: ChatEntry):
    """Save a chat entry to disk atomically."""
    ensure_dirs()

    # Organize by date
    date_dir = CHATS_DIR / entry.timestamp[:10]
    date_dir.mkdir(exist_ok=True)

    filename = f"{entry.id}.json"
    filepath = date_dir / filename

    atomic_write_json(filepath, asdict(entry))
    print(f"Saved chat entry to {filepath}")


def log_chat(
    query: str,
    response: str,
    session_id: Optional[str] = None,
    files_referenced: Optional[List[str]] = None,
    files_modified: Optional[List[str]] = None,
    tools_used: Optional[List[str]] = None,
    user_feedback: Optional[str] = None,
) -> ChatEntry:
    """Log a chat query/response pair."""

    entry = ChatEntry(
        id=generate_chat_id(),
        timestamp=datetime.now().isoformat(),
        session_id=session_id or generate_session_id(),
        query=query,
        response=response,
        files_referenced=files_referenced or [],
        files_modified=files_modified or [],
        tools_used=tools_used or [],
        user_feedback=user_feedback,
        query_tokens=len(query.split()),  # Rough estimate
        response_tokens=len(response.split()),
    )

    save_chat_entry(entry)
    return entry


def save_action(entry: ActionEntry):
    """Save an action entry to disk atomically."""
    ensure_dirs()

    date_dir = ACTIONS_DIR / entry.timestamp[:10]
    date_dir.mkdir(exist_ok=True)

    filename = f"{entry.id}.json"
    filepath = date_dir / filename

    atomic_write_json(filepath, asdict(entry))


def log_action(
    action_type: str,
    target: str,
    session_id: Optional[str] = None,
    context: Optional[Dict] = None,
    success: bool = True,
    result_summary: Optional[str] = None,
) -> ActionEntry:
    """Log a discrete action."""

    timestamp = datetime.now()
    action_id = f"act-{timestamp.strftime('%Y%m%d-%H%M%S')}-{hashlib.sha256(str(timestamp.timestamp()).encode()).hexdigest()[:4]}"

    entry = ActionEntry(
        id=action_id,
        timestamp=timestamp.isoformat(),
        session_id=session_id or generate_session_id(),
        action_type=action_type,
        target=target,
        context=context or {},
        success=success,
        result_summary=result_summary,
    )

    save_action(entry)
    return entry


# ============================================================================
# STATISTICS AND ESTIMATION
# ============================================================================

def count_data() -> Dict[str, int]:
    """Count collected data entries."""
    ensure_dirs()

    counts = {
        "commits": 0,
        "chats": 0,
        "actions": 0,
        "sessions": 0,
    }

    # Count commits
    if COMMITS_DIR.exists():
        counts["commits"] = len(list(COMMITS_DIR.glob("*.json")))

    # Count chats
    if CHATS_DIR.exists():
        counts["chats"] = len(list(CHATS_DIR.glob("**/*.json")))

    # Count actions
    if ACTIONS_DIR.exists():
        counts["actions"] = len(list(ACTIONS_DIR.glob("**/*.json")))

    # Count sessions
    if SESSIONS_DIR.exists():
        counts["sessions"] = len(list(SESSIONS_DIR.glob("*.json")))

    return counts


def calculate_data_size() -> Dict[str, int]:
    """Calculate total size of collected data."""
    ensure_dirs()

    sizes = {}
    for name, dir_path in [
        ("commits", COMMITS_DIR),
        ("chats", CHATS_DIR),
        ("actions", ACTIONS_DIR),
        ("sessions", SESSIONS_DIR),
    ]:
        total = 0
        if dir_path.exists():
            for f in dir_path.glob("**/*.json"):
                total += f.stat().st_size
        sizes[name] = total

    sizes["total"] = sum(sizes.values())
    return sizes


def estimate_progress() -> Dict[str, Dict]:
    """Estimate progress toward training milestones."""
    counts = count_data()

    progress = {}
    for milestone, requirements in MILESTONES.items():
        milestone_progress = {}
        for data_type, required in requirements.items():
            current = counts.get(data_type, 0)
            milestone_progress[data_type] = {
                "current": current,
                "required": required,
                "percent": min(100, int(100 * current / required)),
            }

        # Overall milestone progress (minimum of all types)
        overall = min(p["percent"] for p in milestone_progress.values())
        milestone_progress["overall"] = overall
        progress[milestone] = milestone_progress

    return progress


def print_stats():
    """Print collection statistics."""
    counts = count_data()
    sizes = calculate_data_size()
    progress = estimate_progress()

    print("\n" + "=" * 60)
    print("ML DATA COLLECTION STATISTICS")
    print("=" * 60)

    print("\n📊 Data Counts:")
    print(f"   Commits:  {counts['commits']:,}")
    print(f"   Chats:    {counts['chats']:,}")
    print(f"   Actions:  {counts['actions']:,}")
    print(f"   Sessions: {counts['sessions']:,}")

    print("\n💾 Data Sizes:")
    for name, size in sizes.items():
        if size > 1024 * 1024:
            print(f"   {name.capitalize():10s}: {size / 1024 / 1024:.2f} MB")
        elif size > 1024:
            print(f"   {name.capitalize():10s}: {size / 1024:.2f} KB")
        else:
            print(f"   {name.capitalize():10s}: {size} bytes")

    print("\n🎯 Training Milestones:")
    for milestone, data in progress.items():
        overall = data.pop("overall")
        bar = "█" * (overall // 5) + "░" * (20 - overall // 5)
        print(f"\n   {milestone.replace('_', ' ').title()}: [{bar}] {overall}%")
        for data_type, info in data.items():
            print(f"      {data_type}: {info['current']}/{info['required']}")

    print("\n" + "=" * 60)


def estimate_project_size():
    """Estimate final project size when all milestones are reached."""
    # Current averages
    sizes = calculate_data_size()
    counts = count_data()

    # Calculate average sizes per entry type
    avg_commit_size = sizes["commits"] / max(1, counts["commits"])
    avg_chat_size = sizes["chats"] / max(1, counts["chats"]) if counts["chats"] > 0 else 2000  # estimate 2KB per chat
    avg_action_size = sizes["actions"] / max(1, counts["actions"]) if counts["actions"] > 0 else 500  # estimate 500B per action

    # Target for "code_suggestions" milestone (the highest)
    target_commits = MILESTONES["code_suggestions"]["commits"]
    target_chats = MILESTONES["code_suggestions"]["chats"]
    target_actions = target_chats * 10  # Estimate 10 actions per chat

    estimated_total = (
        target_commits * max(avg_commit_size, 5000) +  # ~5KB per commit if no data
        target_chats * max(avg_chat_size, 2000) +
        target_actions * max(avg_action_size, 500)
    )

    print("\n" + "=" * 60)
    print("PROJECT SIZE ESTIMATE (Full Collection)")
    print("=" * 60)

    print("\n📈 Target Data Points:")
    print(f"   Commits:  {target_commits:,}")
    print(f"   Chats:    {target_chats:,}")
    print(f"   Actions:  {target_actions:,} (estimated)")

    print("\n💾 Estimated Sizes:")
    print(f"   Commits data:  {target_commits * max(avg_commit_size, 5000) / 1024 / 1024:.1f} MB")
    print(f"   Chats data:    {target_chats * max(avg_chat_size, 2000) / 1024 / 1024:.1f} MB")
    print(f"   Actions data:  {target_actions * max(avg_action_size, 500) / 1024 / 1024:.1f} MB")
    print(f"   ─────────────────────────")
    print(f"   TOTAL:         {estimated_total / 1024 / 1024:.1f} MB")

    print("\n🧠 Model Training Estimates:")
    print(f"   Vocabulary size:     ~15,000 tokens (this project)")
    print(f"   Training examples:   ~{target_commits + target_chats:,}")
    print(f"   Micro-model size:    1-10 MB (1-10M parameters)")
    print(f"   Training time:       ~1-4 hours (single GPU)")
    print(f"   Inference:           <100ms on CPU")

    print("\n⏱️  Time to Collection Complete:")
    counts = count_data()
    commits_per_day = 20  # Based on current rate
    chats_per_day = 15  # Estimated

    days_for_commits = (target_commits - counts["commits"]) / commits_per_day
    days_for_chats = (target_chats - counts["chats"]) / chats_per_day

    days_needed = max(days_for_commits, days_for_chats)
    print(f"   At current rate:     ~{int(days_needed)} days ({int(days_needed/30)} months)")
    print(f"   With active use:     ~{int(days_needed * 0.5)} days (more chatting)")

    print("\n" + "=" * 60)


# ============================================================================
# GIT HOOKS
# ============================================================================

ML_HOOK_MARKER = "# ML-DATA-COLLECTOR-HOOK"

POST_COMMIT_SNIPPET = '''
# ML-DATA-COLLECTOR-HOOK
# ML Data Collection - Post-Commit Hook
# Automatically collects enriched commit data for model training
python scripts/ml_data_collector.py commit 2>/dev/null || true
# END-ML-DATA-COLLECTOR-HOOK
'''

PRE_PUSH_SNIPPET = '''
# ML-DATA-COLLECTOR-HOOK
# ML Data Collection - Pre-Push Hook
# Validates data collection is working before push
if [ -d ".git-ml/commits" ]; then
    count=$(ls -1 .git-ml/commits/*.json 2>/dev/null | wc -l)
    echo "📊 ML Data: $count commits collected"
fi
# END-ML-DATA-COLLECTOR-HOOK
'''


def install_hooks():
    """Install git hooks for data collection, merging with existing hooks."""
    hooks_dir = Path(".git/hooks")

    for hook_name, snippet in [("post-commit", POST_COMMIT_SNIPPET), ("pre-push", PRE_PUSH_SNIPPET)]:
        hook_path = hooks_dir / hook_name

        if hook_path.exists():
            existing = hook_path.read_text(encoding="utf-8")

            # Check if our hook is already installed
            if ML_HOOK_MARKER in existing:
                print(f"✓ {hook_name}: ML hook already installed")
                continue

            # Append to existing hook
            with open(hook_path, "a", encoding="utf-8") as f:
                f.write(snippet)
            print(f"✓ {hook_name}: Added ML hook to existing hook")

        else:
            # Create new hook with shebang
            with open(hook_path, "w", encoding="utf-8") as f:
                f.write("#!/bin/bash\n")
                f.write(snippet)
                f.write("\nexit 0\n")
            hook_path.chmod(0o755)
            print(f"✓ {hook_name}: Created new hook")

    print("\nML hooks installed! Commit data will be collected automatically.")


# ============================================================================
# CLI
# ============================================================================

def main():
    import sys

    if len(sys.argv) < 2:
        print(__doc__)
        return

    command = sys.argv[1]

    if command == "commit":
        # Collect data for current or specified commit
        commit_hash = sys.argv[2] if len(sys.argv) > 2 else None
        context = collect_commit_data(commit_hash)
        save_commit_data(context)

    elif command == "backfill":
        # Backfill historical commits
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("-n", "--num", type=int, default=100,
                            help="Number of commits to backfill")
        args = parser.parse_args(sys.argv[2:])

        hashes = run_git(["log", f"-{args.num}", "--format=%H"]).split("\n")
        hashes = [h for h in hashes if h]
        print(f"Backfilling {len(hashes)} commits...")

        for i, h in enumerate(hashes):
            try:
                context = collect_commit_data(h)
                save_commit_data(context)
                if (i + 1) % 10 == 0:
                    print(f"  Progress: {i + 1}/{len(hashes)}")
            except Exception as e:
                print(f"  Error on {h[:8]}: {e}")

        print(f"Backfill complete: {len(hashes)} commits")

    elif command == "chat":
        # Log a chat entry
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--query", required=True)
        parser.add_argument("--response", required=True)
        parser.add_argument("--session", default=None)
        parser.add_argument("--files-ref", nargs="*", default=[])
        parser.add_argument("--files-mod", nargs="*", default=[])
        parser.add_argument("--tools", nargs="*", default=[])
        parser.add_argument("--feedback", choices=["positive", "negative", "neutral"])

        args = parser.parse_args(sys.argv[2:])

        entry = log_chat(
            query=args.query,
            response=args.response,
            session_id=args.session,
            files_referenced=args.files_ref,
            files_modified=args.files_mod,
            tools_used=args.tools,
            user_feedback=args.feedback,
        )
        print(f"Logged chat: {entry.id}")

    elif command == "action":
        # Log an action
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--type", required=True)
        parser.add_argument("--target", required=True)
        parser.add_argument("--session", default=None)

        args = parser.parse_args(sys.argv[2:])

        entry = log_action(
            action_type=args.type,
            target=args.target,
            session_id=args.session,
        )
        print(f"Logged action: {entry.id}")

    elif command == "stats":
        print_stats()

    elif command == "estimate":
        estimate_project_size()

    elif command == "install-hooks":
        install_hooks()

    else:
        print(f"Unknown command: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()
