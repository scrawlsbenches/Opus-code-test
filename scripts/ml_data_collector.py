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

    # Generate session handoff document
    python scripts/ml_data_collector.py handoff
"""

import json
import os
import subprocess
import hashlib
import re
import tempfile
import uuid
import fcntl
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from contextlib import contextmanager

# Setup logging
logger = logging.getLogger(__name__)

# Environment variable to disable collection
ML_COLLECTION_ENABLED = os.getenv("ML_COLLECTION_ENABLED", "1") != "0"


class GitCommandError(Exception):
    """Raised when a git command fails."""
    pass


class SchemaValidationError(Exception):
    """Raised when data fails schema validation."""
    pass


# ============================================================================
# CONFIGURATION
# ============================================================================

ML_DATA_DIR = Path(".git-ml")
COMMITS_DIR = ML_DATA_DIR / "commits"
SESSIONS_DIR = ML_DATA_DIR / "sessions"
CHATS_DIR = ML_DATA_DIR / "chats"
ACTIONS_DIR = ML_DATA_DIR / "actions"
CURRENT_SESSION_FILE = ML_DATA_DIR / "current_session.json"

# Training milestones
MILESTONES = {
    "file_prediction": {"commits": 500, "sessions": 100, "chats": 200},
    "commit_messages": {"commits": 2000, "sessions": 500, "chats": 1000},
    "code_suggestions": {"commits": 5000, "sessions": 2000, "chats": 5000},
}

# Schema validation definitions
COMMIT_SCHEMA = {
    "required": ["hash", "message", "author", "timestamp", "branch", "files_changed",
                 "insertions", "deletions", "hunks", "hour_of_day", "day_of_week"],
    "types": {
        "hash": str, "message": str, "author": str, "timestamp": str, "branch": str,
        "files_changed": list, "insertions": int, "deletions": int, "hunks": list,
        "hour_of_day": int, "day_of_week": str, "is_merge": bool, "is_initial": bool,
        "parent_count": int, "session_id": (str, type(None)), "related_chats": list,
    }
}

CHAT_SCHEMA = {
    "required": ["id", "timestamp", "session_id", "query", "response",
                 "files_referenced", "files_modified", "tools_used"],
    "types": {
        "id": str, "timestamp": str, "session_id": str, "query": str, "response": str,
        "files_referenced": list, "files_modified": list, "tools_used": list,
        "query_tokens": int, "response_tokens": int,
    }
}

ACTION_SCHEMA = {
    "required": ["id", "timestamp", "session_id", "action_type", "target"],
    "types": {
        "id": str, "timestamp": str, "session_id": str,
        "action_type": str, "target": str, "success": bool,
    }
}


def validate_schema(data: dict, schema: dict, data_type: str) -> List[str]:
    """Validate data against a schema. Returns list of errors (empty if valid)."""
    errors = []
    for field in schema["required"]:
        if field not in data:
            errors.append(f"{data_type}: missing required field '{field}'")
    for field, expected in schema["types"].items():
        if field in data and data[field] is not None:
            if isinstance(expected, tuple):
                if not isinstance(data[field], expected):
                    errors.append(f"{data_type}: field '{field}' has wrong type")
            elif not isinstance(data[field], expected):
                errors.append(f"{data_type}: field '{field}' has type {type(data[field]).__name__}, expected {expected.__name__}")
    return errors


# ============================================================================
# FILE LOCKING (for concurrent access safety)
# ============================================================================

@contextmanager
def file_lock(filepath: Path, exclusive: bool = True):
    """Context manager for file locking to prevent race conditions.

    Args:
        filepath: Path to lock file
        exclusive: True for write lock, False for read lock
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    lock_file = filepath.with_suffix(filepath.suffix + '.lock')

    with open(lock_file, 'w') as f:
        lock_type = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        try:
            fcntl.flock(f.fileno(), lock_type)
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


# ============================================================================
# SESSION MANAGEMENT (for commit-chat linking)
# ============================================================================

def get_current_session() -> Optional[Dict]:
    """Get the current active session info.

    Returns dict with 'id', 'started_at', 'chat_ids' or None if no session.
    """
    if CURRENT_SESSION_FILE.exists():
        try:
            with open(CURRENT_SESSION_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(f"Corrupted session file: {e}")
            return None
        except IOError as e:
            logger.error(f"Cannot read session file: {e}")
            return None
    return None


def start_session(session_id: Optional[str] = None) -> str:
    """Start a new session for commit-chat linking.

    Args:
        session_id: Optional session ID. Generated if not provided.

    Returns:
        The session ID.
    """
    ensure_dirs()

    session_id = session_id or generate_session_id()
    session_data = {
        'id': session_id,
        'started_at': datetime.now().isoformat(),
        'chat_ids': [],
        'action_ids': [],
    }

    atomic_write_json(CURRENT_SESSION_FILE, session_data)
    return session_id


def get_or_create_session() -> str:
    """Get current session ID or create a new one.

    Returns:
        The current session ID.
    """
    session = get_current_session()
    if session:
        return session['id']
    return start_session()


def add_chat_to_session(chat_id: str):
    """Record a chat ID in the current session for later commit linking.

    Uses file locking to prevent race conditions with concurrent access.
    """
    ensure_dirs()

    # Use file lock to prevent race conditions
    with file_lock(CURRENT_SESSION_FILE):
        session = get_current_session()
        if not session:
            # Auto-start session if needed
            session = {
                'id': generate_session_id(),
                'started_at': datetime.now().isoformat(),
                'chat_ids': [],
                'action_ids': [],
            }

        if chat_id not in session['chat_ids']:
            session['chat_ids'].append(chat_id)
            atomic_write_json(CURRENT_SESSION_FILE, session)


def link_commit_to_session_chats(commit_hash: str) -> List[str]:
    """Link a commit to all chats from the current session.

    Updates the chat entries to record that they resulted in this commit.
    Also updates the commit's related_chats field.

    Args:
        commit_hash: The commit hash to link.

    Returns:
        List of chat IDs that were linked.
    """
    session = get_current_session()
    if not session or not session.get('chat_ids'):
        return []

    linked_chats = []

    # Update each chat entry
    for chat_id in session['chat_ids']:
        chat_file = find_chat_file(chat_id)
        if chat_file and chat_file.exists():
            try:
                with open(chat_file, 'r', encoding='utf-8') as f:
                    chat_data = json.load(f)

                # Mark chat as resulting in commit
                chat_data['resulted_in_commit'] = True
                chat_data['related_commit'] = commit_hash

                atomic_write_json(chat_file, chat_data)
                linked_chats.append(chat_id)
            except (json.JSONDecodeError, IOError):
                continue

    return linked_chats


def find_chat_file(chat_id: str) -> Optional[Path]:
    """Find the file path for a chat ID."""
    if not CHATS_DIR.exists():
        return None

    # Chat files are organized by date
    for date_dir in CHATS_DIR.iterdir():
        if date_dir.is_dir():
            chat_file = date_dir / f"{chat_id}.json"
            if chat_file.exists():
                return chat_file

    return None


def end_session(summary: Optional[str] = None) -> Optional[Dict]:
    """End the current session and archive it.

    Args:
        summary: Optional summary of what was accomplished.

    Returns:
        The ended session data or None if no session.
    """
    session = get_current_session()
    if not session:
        return None

    session['ended_at'] = datetime.now().isoformat()
    session['summary'] = summary

    # Save to sessions archive
    ensure_dirs()
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    archive_file = SESSIONS_DIR / f"{session['started_at'][:10]}_{session['id']}.json"
    atomic_write_json(archive_file, session)

    # Remove current session file
    if CURRENT_SESSION_FILE.exists():
        CURRENT_SESSION_FILE.unlink()

    return session


def generate_session_handoff() -> str:
    """Generate a session handoff document with summary of work done.

    Returns:
        Markdown-formatted handoff document, or error message if no session.
    """
    session = get_current_session()
    if not session:
        return "No active session found. Use 'session start' to begin tracking."

    # Parse session info
    session_id = session['id']
    started_at = session['started_at']
    chat_ids = session.get('chat_ids', [])

    # Calculate duration
    start_time = datetime.fromisoformat(started_at)
    duration = datetime.now() - start_time
    hours = int(duration.total_seconds() // 3600)
    minutes = int((duration.total_seconds() % 3600) // 60)
    duration_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"

    # Load chat entries to analyze work done
    chats = []
    all_files_referenced = set()
    all_files_modified = set()
    all_tools_used = set()

    for chat_id in chat_ids:
        chat_file = find_chat_file(chat_id)
        if chat_file and chat_file.exists():
            try:
                with open(chat_file, 'r', encoding='utf-8') as f:
                    chat_data = json.load(f)
                    chats.append(chat_data)
                    all_files_referenced.update(chat_data.get('files_referenced', []))
                    all_files_modified.update(chat_data.get('files_modified', []))
                    all_tools_used.update(chat_data.get('tools_used', []))
            except (json.JSONDecodeError, IOError):
                continue

    # Summarize key tasks from queries
    key_tasks = []
    for chat in chats[-10:]:  # Last 10 chats
        query = chat.get('query', '')
        # Extract first sentence or first 80 chars as task summary
        task_summary = query.split('.')[0][:80]
        if task_summary and task_summary not in key_tasks:
            key_tasks.append(task_summary)

    # Find related commits
    related_commits = []
    if COMMITS_DIR.exists():
        for commit_file in COMMITS_DIR.glob("*.json"):
            try:
                with open(commit_file, 'r', encoding='utf-8') as f:
                    commit_data = json.load(f)
                    if commit_data.get('session_id') == session_id:
                        related_commits.append({
                            'hash': commit_data['hash'][:8],
                            'message': commit_data['message'],
                            'timestamp': commit_data['timestamp']
                        })
            except (json.JSONDecodeError, IOError):
                continue

    # Sort commits by timestamp
    related_commits.sort(key=lambda c: c['timestamp'])

    # Generate suggested next steps based on patterns
    suggestions = []

    # Check for incomplete work patterns
    if any('test' in tool.lower() for tool in all_tools_used):
        if not any('pass' in c['message'].lower() for c in related_commits):
            suggestions.append("Run and verify tests pass before committing")

    if all_files_modified and not related_commits:
        suggestions.append("Review modified files and commit changes if ready")

    if 'Edit' in all_tools_used or 'Write' in all_tools_used:
        suggestions.append("Consider running the test suite to verify changes")

    if len(chats) > 5 and not related_commits:
        suggestions.append("Session has multiple exchanges but no commits - consider saving work")

    # Check for unresolved errors in recent responses
    recent_errors = any('error' in chat.get('response', '').lower() or
                       'failed' in chat.get('response', '').lower()
                       for chat in chats[-3:])
    if recent_errors:
        suggestions.append("Recent responses mention errors - may need debugging or fixes")

    if not suggestions:
        suggestions.append("Continue with planned work")

    # Build markdown document
    md = []
    md.append(f"# Session Handoff: {session_id}")
    md.append("")
    md.append("## Summary")
    md.append(f"- Started: {started_at}")
    md.append(f"- Duration: {duration_str}")
    md.append(f"- Exchanges: {len(chats)}")
    md.append(f"- Tools used: {', '.join(sorted(all_tools_used)) if all_tools_used else 'none'}")
    md.append("")

    md.append("## Key Work Done")
    if key_tasks:
        for task in key_tasks[:5]:  # Top 5 tasks
            md.append(f"- {task}")
    else:
        md.append("- No significant work recorded")
    md.append("")

    md.append("## Files Touched")
    if all_files_modified:
        md.append("### Modified:")
        for f in sorted(all_files_modified)[:10]:  # Top 10 files
            md.append(f"- {f}")
    if all_files_referenced and all_files_referenced - all_files_modified:
        md.append("### Referenced:")
        for f in sorted(all_files_referenced - all_files_modified)[:10]:
            md.append(f"- {f}")
    if not all_files_modified and not all_files_referenced:
        md.append("- No files modified or referenced")
    md.append("")

    md.append("## Related Commits")
    if related_commits:
        for commit in related_commits:
            md.append(f"- `{commit['hash']}`: {commit['message']}")
    else:
        md.append("- No commits made in this session")
    md.append("")

    md.append("## Suggested Next Steps")
    for suggestion in suggestions:
        md.append(f"- {suggestion}")
    md.append("")

    return "\n".join(md)



# ============================================================================
# CI STATUS INTEGRATION
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


# ============================================================================
# TRANSCRIPT PARSING (for automatic session capture via Stop hook)
# ============================================================================

@dataclass
class TranscriptExchange:
    """A single query/response exchange extracted from a transcript."""
    query: str
    response: str
    tools_used: List[str]
    tool_inputs: List[Dict]
    timestamp: str
    thinking: Optional[str] = None


def parse_transcript_jsonl(filepath: Path) -> List[TranscriptExchange]:
    """Parse a Claude Code transcript JSONL file into exchanges.

    The JSONL format has entries with:
    - type: "user" or "assistant"
    - message.content: string (user) or array of content blocks (assistant)
    - timestamp: ISO timestamp

    Returns list of TranscriptExchange objects.
    """
    if not filepath.exists():
        logger.warning(f"Transcript file not found: {filepath}")
        return []

    exchanges = []
    current_query = None
    current_response_parts = []
    current_tools = []
    current_tool_inputs = []
    current_thinking = None
    current_timestamp = None

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                entry_type = entry.get('type')
                message = entry.get('message', {})
                timestamp = entry.get('timestamp', '')

                if entry_type == 'user':
                    # Save previous exchange if we have one
                    if current_query and current_response_parts:
                        exchanges.append(TranscriptExchange(
                            query=current_query,
                            response=' '.join(current_response_parts),
                            tools_used=current_tools,
                            tool_inputs=current_tool_inputs,
                            timestamp=current_timestamp or timestamp,
                            thinking=current_thinking,
                        ))

                    # Start new exchange
                    content = message.get('content', '')
                    if isinstance(content, str):
                        current_query = content
                    elif isinstance(content, list):
                        # Extract text from content blocks
                        current_query = ' '.join(
                            c.get('text', '') for c in content
                            if c.get('type') == 'text'
                        )
                    current_response_parts = []
                    current_tools = []
                    current_tool_inputs = []
                    current_thinking = None
                    current_timestamp = timestamp

                elif entry_type == 'assistant':
                    content = message.get('content', [])
                    if isinstance(content, list):
                        for block in content:
                            block_type = block.get('type')

                            if block_type == 'text':
                                text = block.get('text', '')
                                if text:
                                    current_response_parts.append(text)

                            elif block_type == 'thinking':
                                current_thinking = block.get('thinking', '')

                            elif block_type == 'tool_use':
                                tool_name = block.get('name', '')
                                tool_input = block.get('input', {})
                                if tool_name and tool_name not in current_tools:
                                    current_tools.append(tool_name)
                                current_tool_inputs.append({
                                    'tool': tool_name,
                                    'input': tool_input,
                                })

        # Don't forget the last exchange
        if current_query and current_response_parts:
            exchanges.append(TranscriptExchange(
                query=current_query,
                response=' '.join(current_response_parts),
                tools_used=current_tools,
                tool_inputs=current_tool_inputs,
                timestamp=current_timestamp or '',
                thinking=current_thinking,
            ))

    except IOError as e:
        logger.error(f"Error reading transcript: {e}")
        return []

    return exchanges


def extract_files_from_tool_inputs(tool_inputs: List[Dict]) -> tuple:
    """Extract file references and modifications from tool inputs.

    Returns (files_referenced, files_modified) tuple.
    """
    files_referenced = set()
    files_modified = set()

    for ti in tool_inputs:
        tool = ti.get('tool', '')
        inp = ti.get('input', {})

        if tool == 'Read':
            path = inp.get('file_path', '')
            if path:
                files_referenced.add(path)

        elif tool in ('Edit', 'Write', 'MultiEdit'):
            path = inp.get('file_path', '')
            if path:
                files_modified.add(path)

        elif tool == 'Bash':
            # Try to extract file paths from command
            cmd = inp.get('command', '')
            # Simple heuristic: look for paths
            for word in cmd.split():
                if '/' in word and not word.startswith('-'):
                    if word.endswith('.py') or word.endswith('.md') or word.endswith('.json'):
                        files_referenced.add(word)

        elif tool == 'Glob':
            path = inp.get('path', '')
            if path:
                files_referenced.add(path)

        elif tool == 'Grep':
            path = inp.get('path', '')
            if path:
                files_referenced.add(path)

    return list(files_referenced), list(files_modified)


def process_transcript(
    filepath: Path,
    session_id: Optional[str] = None,
    save_exchanges: bool = True
) -> Dict[str, Any]:
    """Process a transcript file and optionally save exchanges.

    Args:
        filepath: Path to the JSONL transcript
        session_id: Optional session ID to use (extracted from transcript if not provided)
        save_exchanges: Whether to save exchanges to .git-ml/chats/

    Returns:
        Summary dict with counts and extracted data.
    """
    exchanges = parse_transcript_jsonl(filepath)

    if not exchanges:
        return {'status': 'empty', 'exchanges': 0}

    # Use provided session_id or generate one
    if not session_id:
        session_id = generate_session_id()

    saved_count = 0
    total_tools = set()
    all_files_ref = set()
    all_files_mod = set()

    for ex in exchanges:
        files_ref, files_mod = extract_files_from_tool_inputs(ex.tool_inputs)
        all_files_ref.update(files_ref)
        all_files_mod.update(files_mod)
        total_tools.update(ex.tools_used)

        if save_exchanges:
            try:
                entry = ChatEntry(
                    id=generate_chat_id(),
                    timestamp=ex.timestamp or datetime.now().isoformat(),
                    session_id=session_id,
                    query=ex.query[:10000],  # Limit query length
                    response=ex.response[:50000],  # Limit response length
                    files_referenced=files_ref,
                    files_modified=files_mod,
                    tools_used=ex.tools_used,
                    query_tokens=len(ex.query.split()),
                    response_tokens=len(ex.response.split()),
                )
                save_chat_entry(entry, validate=True)
                add_chat_to_session(entry.id)
                saved_count += 1
            except Exception as e:
                logger.error(f"Error saving exchange: {e}")

    return {
        'status': 'success',
        'exchanges': len(exchanges),
        'saved': saved_count,
        'session_id': session_id,
        'tools_used': list(total_tools),
        'files_referenced': list(all_files_ref),
        'files_modified': list(all_files_mod),
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


def save_commit_data(context: CommitContext, validate: bool = True, link_session: bool = True):
    """Save commit context to disk atomically with validation.

    Args:
        context: The commit context to save.
        validate: Whether to validate against schema.
        link_session: Whether to link with current session chats.
    """
    ensure_dirs()

    # Link to current session if available
    if link_session:
        session = get_current_session()
        if session:
            context.session_id = session['id']
            context.related_chats = session.get('chat_ids', [])

    data = asdict(context)

    # Validate before writing
    if validate:
        errors = validate_schema(data, COMMIT_SCHEMA, "commit")
        if errors:
            raise SchemaValidationError(f"Commit validation failed: {errors}")

    # Use full hash + UUID suffix to prevent collisions
    unique_id = uuid.uuid4().hex[:8]
    filename = f"{context.hash[:8]}_{context.timestamp[:10]}_{unique_id}.json"
    filepath = COMMITS_DIR / filename

    atomic_write_json(filepath, data)
    print(f"Saved commit data to {filepath}")

    # Update chat entries to link back to this commit
    if link_session and context.related_chats:
        linked = link_commit_to_session_chats(context.hash)
        if linked:
            print(f"Linked {len(linked)} chat(s) to commit {context.hash[:8]}")


def generate_chat_id() -> str:
    """Generate unique chat entry ID."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = hashlib.sha256(str(datetime.now().timestamp()).encode()).hexdigest()[:6]
    return f"chat-{timestamp}-{suffix}"


def generate_session_id() -> str:
    """Generate unique session ID."""
    return hashlib.sha256(str(datetime.now().timestamp()).encode()).hexdigest()[:8]


def save_chat_entry(entry: ChatEntry, validate: bool = True):
    """Save a chat entry to disk atomically with validation."""
    ensure_dirs()

    data = asdict(entry)

    # Validate before writing
    if validate:
        errors = validate_schema(data, CHAT_SCHEMA, "chat")
        if errors:
            raise SchemaValidationError(f"Chat validation failed: {errors}")

    # Organize by date
    date_dir = CHATS_DIR / entry.timestamp[:10]
    date_dir.mkdir(exist_ok=True)

    filename = f"{entry.id}.json"
    filepath = date_dir / filename

    atomic_write_json(filepath, data)
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
    """Log a chat query/response pair.

    If no session_id is provided, uses the current session (creating one if needed).
    The chat is automatically registered with the session for commit linking.
    """
    # Use current session or create one
    if session_id is None:
        session_id = get_or_create_session()

    entry = ChatEntry(
        id=generate_chat_id(),
        timestamp=datetime.now().isoformat(),
        session_id=session_id,
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

    # Register with session for commit linking
    add_chat_to_session(entry.id)

    return entry


def save_action(entry: ActionEntry, validate: bool = True):
    """Save an action entry to disk atomically with validation."""
    ensure_dirs()

    data = asdict(entry)

    # Validate before writing
    if validate:
        errors = validate_schema(data, ACTION_SCHEMA, "action")
        if errors:
            raise SchemaValidationError(f"Action validation failed: {errors}")

    date_dir = ACTIONS_DIR / entry.timestamp[:10]
    date_dir.mkdir(exist_ok=True)

    filename = f"{entry.id}.json"
    filepath = date_dir / filename

    atomic_write_json(filepath, data)


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

    # Allow stats/estimate/validate even when collection is disabled
    read_only_commands = {"stats", "estimate", "validate", "session"}

    # Check if collection is disabled (via ML_COLLECTION_ENABLED=0)
    if not ML_COLLECTION_ENABLED and command not in read_only_commands:
        # Silently exit for collection commands when disabled
        return

    if command == "commit":
        # Collect data for current or specified commit
        commit_hash = sys.argv[2] if len(sys.argv) > 2 else None
        context = collect_commit_data(commit_hash)
        save_commit_data(context)

    elif command == "backfill":
        # Backfill historical commits (no session linking for historical data)
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
                # Disable session linking for historical backfill
                save_commit_data(context, link_session=False)
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

    elif command == "validate":
        # Validate existing data against schemas
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--fix", action="store_true",
                            help="Attempt to fix invalid entries")
        parser.add_argument("--verbose", "-v", action="store_true",
                            help="Show all validation errors")
        args = parser.parse_args(sys.argv[2:])

        print("\n" + "=" * 60)
        print("VALIDATING ML DATA AGAINST SCHEMAS")
        print("=" * 60)

        all_errors = []

        # Validate commits
        if COMMITS_DIR.exists():
            print(f"\n📁 Validating commits...")
            commit_files = list(COMMITS_DIR.glob("*.json"))
            commit_errors = 0
            for f in commit_files:
                try:
                    with open(f, "r", encoding="utf-8") as fp:
                        data = json.load(fp)
                    errors = validate_schema(data, COMMIT_SCHEMA, f"commit:{f.name}")
                    if errors:
                        commit_errors += 1
                        all_errors.extend(errors)
                        if args.verbose:
                            for err in errors:
                                print(f"   ❌ {err}")
                except json.JSONDecodeError as e:
                    commit_errors += 1
                    all_errors.append(f"commit:{f.name}: invalid JSON: {e}")
                    if args.verbose:
                        print(f"   ❌ {f.name}: invalid JSON")
            print(f"   ✓ {len(commit_files) - commit_errors}/{len(commit_files)} valid")

        # Validate chats
        if CHATS_DIR.exists():
            print(f"\n📁 Validating chats...")
            chat_files = list(CHATS_DIR.glob("**/*.json"))
            chat_errors = 0
            for f in chat_files:
                try:
                    with open(f, "r", encoding="utf-8") as fp:
                        data = json.load(fp)
                    errors = validate_schema(data, CHAT_SCHEMA, f"chat:{f.name}")
                    if errors:
                        chat_errors += 1
                        all_errors.extend(errors)
                        if args.verbose:
                            for err in errors:
                                print(f"   ❌ {err}")
                except json.JSONDecodeError as e:
                    chat_errors += 1
                    all_errors.append(f"chat:{f.name}: invalid JSON: {e}")
            print(f"   ✓ {len(chat_files) - chat_errors}/{len(chat_files)} valid")

        # Validate actions
        if ACTIONS_DIR.exists():
            print(f"\n📁 Validating actions...")
            action_files = list(ACTIONS_DIR.glob("**/*.json"))
            action_errors = 0
            for f in action_files:
                try:
                    with open(f, "r", encoding="utf-8") as fp:
                        data = json.load(fp)
                    errors = validate_schema(data, ACTION_SCHEMA, f"action:{f.name}")
                    if errors:
                        action_errors += 1
                        all_errors.extend(errors)
                        if args.verbose:
                            for err in errors:
                                print(f"   ❌ {err}")
                except json.JSONDecodeError as e:
                    action_errors += 1
                    all_errors.append(f"action:{f.name}: invalid JSON")
            print(f"   ✓ {len(action_files) - action_errors}/{len(action_files)} valid")

        # Summary
        print("\n" + "-" * 60)
        if all_errors:
            print(f"⚠️  Found {len(all_errors)} validation errors")
            if not args.verbose:
                print("   Run with --verbose to see details")
        else:
            print("✅ All data validated successfully!")
        print("=" * 60 + "\n")

    elif command == "handoff":
        # Generate session handoff document
        handoff_doc = generate_session_handoff()
        print(handoff_doc)

    elif command == "session":
        # Session management for commit-chat linking
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("action", choices=["start", "end", "status"],
                            help="Session action")
        parser.add_argument("--summary", help="Summary for end action")
        args = parser.parse_args(sys.argv[2:])

        if args.action == "start":
            session_id = start_session()
            print(f"Started session: {session_id}")
            print("Chats will be linked to commits made in this session.")

        elif args.action == "end":
            session = end_session(args.summary)
            if session:
                print(f"Ended session: {session['id']}")
                print(f"  Chats logged: {len(session.get('chat_ids', []))}")
                print(f"  Duration: {session.get('started_at', '?')} → {session.get('ended_at', '?')}")
            else:
                print("No active session to end.")

        elif args.action == "status":
            session = get_current_session()
            if session:
                print("\n" + "=" * 50)
                print("CURRENT SESSION")
                print("=" * 50)
                print(f"  ID:         {session['id']}")
                print(f"  Started:    {session['started_at']}")
                print(f"  Chats:      {len(session.get('chat_ids', []))}")
                if session.get('chat_ids'):
                    print("  Chat IDs:")
                    for cid in session['chat_ids'][:5]:  # Show first 5
                        print(f"    - {cid}")
                    if len(session['chat_ids']) > 5:
                        print(f"    ... and {len(session['chat_ids']) - 5} more")
                print("=" * 50 + "\n")
            else:
                print("No active session. Use 'session start' to begin.")

    elif command == "ci":
        # CI status integration for recording test results
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("action", choices=["set", "get"],
                            help="CI action")
        parser.add_argument("--commit", required=True,
                            help="Commit hash (full or prefix)")
        parser.add_argument("--result", choices=["pass", "fail", "error", "pending"],
                            help="CI result (for set action)")
        parser.add_argument("--tests-passed", type=int,
                            help="Number of tests passed")
        parser.add_argument("--tests-failed", type=int,
                            help="Number of tests failed")
        parser.add_argument("--coverage", type=float,
                            help="Code coverage percentage")
        parser.add_argument("--duration", type=float,
                            help="CI duration in seconds")
        parser.add_argument("--message", help="CI message or failure details")
        args = parser.parse_args(sys.argv[2:])

        if args.action == "set":
            if not args.result:
                print("Error: --result is required for 'set' action")
                sys.exit(1)

            # Build details dict from optional args
            details = {}
            if args.tests_passed is not None:
                details['tests_passed'] = args.tests_passed
            if args.tests_failed is not None:
                details['tests_failed'] = args.tests_failed
            if args.coverage is not None:
                details['coverage'] = args.coverage
            if args.duration is not None:
                details['duration_seconds'] = args.duration
            if args.message:
                details['message'] = args.message

            success = update_commit_ci_result(
                args.commit,
                args.result,
                details if details else None
            )

            if success:
                print(f"✓ Updated CI result for {args.commit[:8]}: {args.result}")
                if details:
                    for k, v in details.items():
                        print(f"  {k}: {v}")
            else:
                print(f"✗ Commit not found: {args.commit}")
                sys.exit(1)

        elif args.action == "get":
            commit_file = find_commit_file(args.commit)
            if commit_file:
                with open(commit_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"\nCI Status for {data['hash'][:12]}:")
                print(f"  Result: {data.get('ci_result', 'not set')}")
                if data.get('ci_details'):
                    print("  Details:")
                    for k, v in data['ci_details'].items():
                        print(f"    {k}: {v}")
                if data.get('ci_updated_at'):
                    print(f"  Updated: {data['ci_updated_at']}")
            else:
                print(f"✗ Commit not found: {args.commit}")
                sys.exit(1)

    elif command == "revert":
        # Mark a commit as reverted
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--commit", required=True,
                            help="Commit that was reverted")
        parser.add_argument("--by",
                            help="Commit that performed the revert")
        args = parser.parse_args(sys.argv[2:])

        success = mark_commit_reverted(args.commit, args.by)
        if success:
            print(f"✓ Marked {args.commit[:8]} as reverted")
        else:
            print(f"✗ Commit not found: {args.commit}")
            sys.exit(1)

    elif command == "transcript":
        # Process a Claude Code transcript file (called by Stop hook)
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--file", "-f", required=True,
                            help="Path to transcript JSONL file")
        parser.add_argument("--session-id",
                            help="Session ID to use (auto-generated if not provided)")
        parser.add_argument("--dry-run", action="store_true",
                            help="Parse and show stats without saving")
        parser.add_argument("--verbose", "-v", action="store_true",
                            help="Show detailed output")
        args = parser.parse_args(sys.argv[2:])

        filepath = Path(args.file)
        if not filepath.exists():
            print(f"✗ Transcript file not found: {filepath}")
            sys.exit(1)

        if args.verbose:
            print(f"\n{'='*60}")
            print("PROCESSING CLAUDE CODE TRANSCRIPT")
            print(f"{'='*60}")
            print(f"File: {filepath}")
            print(f"Size: {filepath.stat().st_size / 1024:.1f} KB")

        # Process the transcript
        result = process_transcript(
            filepath,
            session_id=args.session_id,
            save_exchanges=not args.dry_run
        )

        if args.verbose or args.dry_run:
            print(f"\n📊 Transcript Analysis:")
            print(f"   Exchanges found: {result.get('exchanges', 0)}")
            if not args.dry_run:
                print(f"   Exchanges saved: {result.get('saved', 0)}")
            print(f"   Session ID: {result.get('session_id', 'N/A')}")
            print(f"   Tools used: {', '.join(result.get('tools_used', [])) or 'none'}")
            print(f"   Files referenced: {len(result.get('files_referenced', []))}")
            print(f"   Files modified: {len(result.get('files_modified', []))}")
            print(f"{'='*60}\n")
        else:
            # Minimal output for hook usage
            saved = result.get('saved', 0)
            if saved > 0:
                print(f"📝 ML: Captured {saved} exchange(s) from session")

    else:
        print(f"Unknown command: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()
