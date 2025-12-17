#!/usr/bin/env python3
"""
ML Data Collector for Project-Specific Language Model Training

This module collects enriched data from git commits, chat sessions, and developer
actions to train a micro-model specific to this project.

PRIVACY: Sensitive data (API keys, passwords, tokens) is automatically redacted
before storage. See REDACTION_PATTERNS for the full list.

Usage:
    # Collect commit data (call from git hook)
    python scripts/ml_data_collector.py commit

    # Log a chat session
    python scripts/ml_data_collector.py chat --query "..." --response "..."

    # Show statistics
    python scripts/ml_data_collector.py stats

    # Estimate when training is viable
    python scripts/ml_data_collector.py estimate

    # Analyze data quality
    python scripts/ml_data_collector.py quality-report

    # Generate session handoff document
    python scripts/ml_data_collector.py handoff

    # Add feedback to a chat
    python scripts/ml_data_collector.py feedback --chat-id <id> --rating good [--comment "text"]

    # List recent chats and their feedback status
    python scripts/ml_data_collector.py feedback --list [--limit 20]

    # Export data for training
    python scripts/ml_data_collector.py export --format jsonl --output training_data.jsonl

    # Clean up old data (default: 730 days retention)
    python scripts/ml_data_collector.py cleanup [--days 730] [--dry-run]

    # Manage contribution consent
    python scripts/ml_data_collector.py contribute status     # Check consent status
    python scripts/ml_data_collector.py contribute enable     # Opt-in to share data
    python scripts/ml_data_collector.py contribute disable    # Opt-out of sharing
    python scripts/ml_data_collector.py contribute preview    # Preview what would be shared

    # Generate shared patterns (safe to commit)
    python scripts/ml_data_collector.py generate-patterns     # Creates .git-ml/shared/

    # Collect GitHub PR/Issue data (requires gh CLI)
    python scripts/ml_data_collector.py github collect        # Collect recent PRs and issues
    python scripts/ml_data_collector.py github stats          # Show GitHub data counts
    python scripts/ml_data_collector.py github fetch-pr --number 42   # Fetch specific PR
    python scripts/ml_data_collector.py github fetch-issue --number 10  # Fetch specific issue

    # Auto-capture CI results (called from GitHub Actions)
    python scripts/ml_data_collector.py ci-autocapture        # Read from CI environment vars

    # Backfill lightweight commit data (small files, tracked in git)
    python scripts/ml_data_collector.py backfill-lite -n 100  # Last 100 commits
    python scripts/ml_data_collector.py backfill-lite --all   # All history

    # Test redaction patterns
    python scripts/ml_data_collector.py redact-test --text "api_key=secret123"

    # Extract director orchestration data (sub-agent patterns)
    python scripts/ml_data_collector.py orchestration extract       # Extract from current project
    python scripts/ml_data_collector.py orchestration extract --save  # Extract and save
    python scripts/ml_data_collector.py orchestration summary       # Show summary only
    python scripts/ml_data_collector.py orchestration list          # List saved extractions

    # Chunked storage for git-friendly large file storage
    python scripts/ml_data_collector.py chunked migrate             # Migrate chats/commits to chunks
    python scripts/ml_data_collector.py chunked compact             # Compact old chunks
    python scripts/ml_data_collector.py chunked stats               # Show chunked storage stats
    python scripts/ml_data_collector.py chunked reconstruct -o data.jsonl  # Reconstruct from chunks
"""

import json
import os
import subprocess
import hashlib
import re
import shlex
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
COMMITS_DIR = ML_DATA_DIR / "commits"           # Full commit data with diffs (large, local only)
COMMITS_LITE_DIR = ML_DATA_DIR / "commits-lite" # Legacy: individual JSON files (being phased out)
SESSIONS_DIR = ML_DATA_DIR / "sessions"         # Full session data (local only)
CHATS_DIR = ML_DATA_DIR / "chats"
ACTIONS_DIR = ML_DATA_DIR / "actions"
SHARED_DIR = ML_DATA_DIR / "shared"  # Aggregated patterns - safe to commit
GITHUB_DIR = ML_DATA_DIR / "github"  # PR/Issue data - local (contains discussions)
CURRENT_SESSION_FILE = ML_DATA_DIR / "current_session.json"

# Git-tracked JSONL files (single file, append-only, git-friendly)
TRACKED_DIR = ML_DATA_DIR / "tracked"           # Directory for git-tracked data
COMMITS_LITE_FILE = TRACKED_DIR / "commits.jsonl"   # Commit metadata (one per line)
SESSIONS_LITE_FILE = TRACKED_DIR / "sessions.jsonl" # Session summaries (one per line)

# CALI storage (high-performance, git-friendly replacement for JSONL)
CALI_DIR = ML_DATA_DIR / "cali"
ML_USE_CALI = os.getenv("ML_USE_CALI", "1") == "1"  # Enable CALI by default

# Lazy-loaded CALI store instance
_cali_store = None

def get_cali_store():
    """Get or create the CALI store instance."""
    global _cali_store
    if _cali_store is None and ML_USE_CALI:
        try:
            from cortical.ml_storage import MLStore
            session_id = os.getenv("CLAUDE_SESSION_ID", uuid.uuid4().hex[:8])
            _cali_store = MLStore(CALI_DIR, session_id=session_id)
            logger.debug(f"CALI store initialized at {CALI_DIR}")
        except ImportError:
            logger.warning("CALI storage not available (cortical.ml_storage not found)")
        except Exception as e:
            logger.warning(f"Failed to initialize CALI store: {e}")
    return _cali_store

def cali_put(record_type: str, record_id: str, data: Dict[str, Any]) -> bool:
    """Write to CALI store if enabled. Returns True if written."""
    store = get_cali_store()
    if store:
        try:
            store.put(record_type, record_id, data)
            return True
        except Exception as e:
            logger.warning(f"CALI write failed for {record_type}/{record_id}: {e}")
    return False

def cali_exists(record_type: str, record_id: str) -> bool:
    """Check if record exists in CALI (O(1) bloom filter check)."""
    store = get_cali_store()
    if store:
        try:
            return store.exists(record_type, record_id)
        except Exception as e:
            logger.warning(f"CALI exists check failed: {e}")
    return False

# CSV export truncation defaults
CSV_DEFAULT_TRUNCATE_LENGTH = 1000  # Default max length for generic fields
CSV_DEFAULT_TRUNCATE_QUERY = 500    # Default max length for query/input fields
CSV_DEFAULT_TRUNCATE_RESPONSE = 2000  # Default max length for response/output fields

# Training milestones
MILESTONES = {
    "file_prediction": {"commits": 500, "sessions": 100, "chats": 200},
    "commit_messages": {"commits": 2000, "sessions": 500, "chats": 1000},
    "code_suggestions": {"commits": 5000, "sessions": 2000, "chats": 5000},
}

# Schema validation definitions
COMMIT_SCHEMA = {
    # Core required fields (present in both full and lite commits)
    "required": ["hash", "message", "author", "timestamp", "branch", "files_changed",
                 "insertions", "deletions", "hour_of_day", "day_of_week"],
    "types": {
        "hash": str, "message": str, "author": str, "timestamp": str, "branch": str,
        "files_changed": list, "insertions": int, "deletions": int,
        # Optional fields (may be omitted in lite commits)
        "hunks": list,  # Excluded from lite commits to reduce storage
        "related_chats": list,  # Only present when commit is linked to session
        # Commit metadata
        "hour_of_day": int, "day_of_week": str, "is_merge": bool, "is_initial": bool,
        "parent_count": int, "session_id": (str, type(None)),
        # Optional tracking fields
        "seconds_since_last_commit": (int, type(None)), "ci_result": (str, type(None)),
        "reverted": bool, "amended": bool,
    }
}

CHAT_SCHEMA = {
    "required": ["id", "timestamp", "session_id", "query", "response",
                 "files_referenced", "files_modified", "tools_used"],
    "types": {
        "id": str, "timestamp": str, "session_id": str, "query": str, "response": str,
        "files_referenced": list, "files_modified": list, "tools_used": list,
        "tool_outputs": list,  # Optional: tool outputs with success status
        "query_tokens": int, "response_tokens": int,
        "user_feedback": (dict, str, type(None)),  # Can be dict, legacy string, or None
    }
}

ACTION_SCHEMA = {
    "required": ["id", "timestamp", "session_id", "action_type", "target"],
    "types": {
        "id": str, "timestamp": str, "session_id": str,
        "action_type": str, "target": str, "success": bool,
    }
}

# Data retention configuration (days)
# 2 years - enough time to hit training milestones at typical dev pace
# (~66 commits/active day observed, need 5000 for code suggestions)
DEFAULT_RETENTION_DAYS = 730
CONSENT_FILE = ML_DATA_DIR / "contribution_consent.json"

# Sensitive data patterns to redact before storage
# These patterns are applied to query/response text before saving
REDACTION_PATTERNS = [
    # API keys and tokens
    (r'(?i)(api[_-]?key|apikey|api_secret|secret[_-]?key)\s*[=:]\s*["\']?[\w\-]{20,}["\']?', r'\1=<REDACTED>'),
    (r'(?i)(token|bearer|authorization)\s*[=:]\s*["\']?[\w\-\.]{20,}["\']?', r'\1=<REDACTED>'),
    # Passwords and secrets
    (r'(?i)(password|passwd|pwd|secret)\s*[=:]\s*["\']?[^\s"\']{8,}["\']?', r'\1=<REDACTED>'),
    # AWS credentials
    (r'(?i)(aws[_-]?access[_-]?key[_-]?id|aws[_-]?secret[_-]?access[_-]?key)\s*[=:]\s*["\']?[\w\+/]{16,}["\']?', r'\1=<REDACTED>'),
    (r'AKIA[0-9A-Z]{16}', '<AWS_KEY_REDACTED>'),
    # Private keys
    (r'-----BEGIN\s+(RSA\s+)?PRIVATE KEY-----[\s\S]*?-----END\s+(RSA\s+)?PRIVATE KEY-----', '<PRIVATE_KEY_REDACTED>'),
    (r'-----BEGIN\s+OPENSSH\s+PRIVATE KEY-----[\s\S]*?-----END\s+OPENSSH\s+PRIVATE KEY-----', '<SSH_KEY_REDACTED>'),
    # Database connection strings
    (r'(?i)(mongodb|postgresql|mysql|redis)://[^\s"\']+', r'\1://<CONNECTION_REDACTED>'),
    # Generic credentials in URLs
    (r'://[^:]+:[^@]+@', '://<CREDENTIALS>@'),
    # GitHub tokens
    (r'ghp_[a-zA-Z0-9]{36}', '<GITHUB_TOKEN_REDACTED>'),
    (r'gho_[a-zA-Z0-9]{36}', '<GITHUB_OAUTH_REDACTED>'),
    # Slack tokens
    (r'xox[baprs]-[0-9a-zA-Z\-]+', '<SLACK_TOKEN_REDACTED>'),
    # JWT tokens (basic pattern)
    (r'eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*', '<JWT_REDACTED>'),
]


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
# SENSITIVE DATA REDACTION
# ============================================================================

def redact_sensitive_data(text: str) -> str:
    """Redact sensitive data patterns from text before storage.

    Args:
        text: The text to redact

    Returns:
        Text with sensitive patterns replaced with redaction markers
    """
    if not text:
        return text

    redacted = text
    for pattern, replacement in REDACTION_PATTERNS:
        redacted = re.sub(pattern, replacement, redacted)
    return redacted


def count_redactions(original: str, redacted: str) -> int:
    """Count how many redactions were made."""
    if not original or not redacted:
        return 0
    # Count occurrences of redaction markers
    markers = ['<REDACTED>', '<AWS_KEY_REDACTED>', '<PRIVATE_KEY_REDACTED>',
               '<SSH_KEY_REDACTED>', '<CONNECTION_REDACTED>', '<CREDENTIALS>',
               '<GITHUB_TOKEN_REDACTED>', '<GITHUB_OAUTH_REDACTED>',
               '<SLACK_TOKEN_REDACTED>', '<JWT_REDACTED>']
    return sum(redacted.count(m) for m in markers)


# ============================================================================
# DATA RETENTION & CLEANUP
# ============================================================================

def cleanup_old_data(
    retention_days: int = DEFAULT_RETENTION_DAYS,
    dry_run: bool = False,
    verbose: bool = False
) -> Dict[str, Any]:
    """Remove data older than retention_days.

    Args:
        retention_days: Days to keep data (default: 90)
        dry_run: If True, only report what would be deleted
        verbose: Show detailed output

    Returns:
        Dict with counts of files removed/would be removed
    """
    from datetime import timedelta

    cutoff_date = datetime.now() - timedelta(days=retention_days)
    cutoff_str = cutoff_date.strftime('%Y-%m-%d')

    results = {
        'retention_days': retention_days,
        'cutoff_date': cutoff_str,
        'dry_run': dry_run,
        'commits_removed': 0,
        'chats_removed': 0,
        'actions_removed': 0,
        'sessions_removed': 0,
        'bytes_freed': 0,
    }

    # Cleanup commits
    if COMMITS_DIR.exists():
        for f in COMMITS_DIR.glob("*.json"):
            try:
                with open(f, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                timestamp = data.get('timestamp', '')[:10]  # YYYY-MM-DD
                if timestamp < cutoff_str:
                    file_size = f.stat().st_size
                    if verbose:
                        print(f"  {'Would remove' if dry_run else 'Removing'}: {f.name} ({timestamp})")
                    if not dry_run:
                        f.unlink()
                    results['commits_removed'] += 1
                    results['bytes_freed'] += file_size
            except (json.JSONDecodeError, IOError):
                continue

    # Cleanup chats (organized by date directories)
    if CHATS_DIR.exists():
        for date_dir in CHATS_DIR.iterdir():
            if date_dir.is_dir():
                dir_date = date_dir.name  # YYYY-MM-DD
                if dir_date < cutoff_str:
                    for f in date_dir.glob("*.json"):
                        file_size = f.stat().st_size
                        if verbose:
                            print(f"  {'Would remove' if dry_run else 'Removing'}: chats/{dir_date}/{f.name}")
                        if not dry_run:
                            f.unlink()
                        results['chats_removed'] += 1
                        results['bytes_freed'] += file_size
                    # Remove empty directory
                    if not dry_run and not any(date_dir.iterdir()):
                        date_dir.rmdir()

    # Cleanup actions (organized by date directories)
    if ACTIONS_DIR.exists():
        for date_dir in ACTIONS_DIR.iterdir():
            if date_dir.is_dir():
                dir_date = date_dir.name
                if dir_date < cutoff_str:
                    for f in date_dir.glob("*.json"):
                        file_size = f.stat().st_size
                        if verbose:
                            print(f"  {'Would remove' if dry_run else 'Removing'}: actions/{dir_date}/{f.name}")
                        if not dry_run:
                            f.unlink()
                        results['actions_removed'] += 1
                        results['bytes_freed'] += file_size
                    if not dry_run and not any(date_dir.iterdir()):
                        date_dir.rmdir()

    # Cleanup old sessions
    if SESSIONS_DIR.exists():
        for f in SESSIONS_DIR.glob("*.json"):
            try:
                with open(f, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                ended_at = data.get('ended_at', data.get('started_at', ''))[:10]
                if ended_at and ended_at < cutoff_str:
                    file_size = f.stat().st_size
                    if verbose:
                        print(f"  {'Would remove' if dry_run else 'Removing'}: sessions/{f.name}")
                    if not dry_run:
                        f.unlink()
                    results['sessions_removed'] += 1
                    results['bytes_freed'] += file_size
            except (json.JSONDecodeError, IOError):
                continue

    return results


# ============================================================================
# CONTRIBUTION CONSENT MANAGEMENT
# ============================================================================

def get_contribution_consent() -> Optional[Dict[str, Any]]:
    """Get the current contribution consent status.

    Returns:
        Consent data dict or None if no consent recorded
    """
    if not CONSENT_FILE.exists():
        return None
    try:
        with open(CONSENT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def set_contribution_consent(
    consented: bool,
    contributor_name: Optional[str] = None,
    contributor_email: Optional[str] = None,
    notes: Optional[str] = None
) -> Dict[str, Any]:
    """Record contribution consent decision.

    Args:
        consented: Whether user consents to contribute data
        contributor_name: Optional name for attribution
        contributor_email: Optional email for contact
        notes: Optional notes about the consent

    Returns:
        The consent record
    """
    ensure_dirs()

    consent_data = {
        'consented': consented,
        'timestamp': datetime.now().isoformat(),
        'contributor_name': contributor_name,
        'contributor_email': contributor_email,
        'notes': notes,
        'version': '1.0',
    }

    atomic_write_json(CONSENT_FILE, consent_data)
    return consent_data


def preview_contribution_data(
    max_samples: int = 5,
    include_full_text: bool = False
) -> Dict[str, Any]:
    """Preview what data would be shared if contributing.

    Args:
        max_samples: Maximum samples to show per category
        include_full_text: Whether to include full query/response text

    Returns:
        Preview of contribution data with statistics
    """
    ensure_dirs()

    preview = {
        'summary': {
            'total_commits': 0,
            'total_chats': 0,
            'total_sessions': 0,
            'date_range': {'earliest': None, 'latest': None},
            'unique_files': set(),
            'unique_tools': set(),
        },
        'sample_commits': [],
        'sample_chats': [],
        'redaction_stats': {
            'total_redactions': 0,
            'chats_with_redactions': 0,
        }
    }

    all_timestamps = []

    # Sample commits
    if COMMITS_DIR.exists():
        commit_files = sorted(COMMITS_DIR.glob("*.json"), reverse=True)
        preview['summary']['total_commits'] = len(commit_files)

        for f in commit_files[:max_samples]:
            try:
                with open(f, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                all_timestamps.append(data.get('timestamp', ''))
                preview['summary']['unique_files'].update(data.get('files_changed', []))

                sample = {
                    'hash': data.get('hash', '')[:12],
                    'message': data.get('message', '')[:100] + ('...' if len(data.get('message', '')) > 100 else ''),
                    'files_count': len(data.get('files_changed', [])),
                    'timestamp': data.get('timestamp', '')[:10],
                }
                preview['sample_commits'].append(sample)
            except (json.JSONDecodeError, IOError):
                continue

    # Sample chats
    if CHATS_DIR.exists():
        chat_files = sorted(CHATS_DIR.glob("**/*.json"), reverse=True)
        preview['summary']['total_chats'] = len(chat_files)

        for f in chat_files[:max_samples]:
            try:
                with open(f, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                all_timestamps.append(data.get('timestamp', ''))
                preview['summary']['unique_tools'].update(data.get('tools_used', []))

                query = data.get('query', '')
                response = data.get('response', '')

                # Check for redactions
                redacted_query = redact_sensitive_data(query)
                redacted_response = redact_sensitive_data(response)
                redactions = count_redactions(query, redacted_query) + count_redactions(response, redacted_response)

                if redactions > 0:
                    preview['redaction_stats']['total_redactions'] += redactions
                    preview['redaction_stats']['chats_with_redactions'] += 1

                sample = {
                    'id': data.get('id', ''),
                    'timestamp': data.get('timestamp', '')[:10],
                    'tools_used': data.get('tools_used', []),
                    'files_count': len(data.get('files_referenced', [])) + len(data.get('files_modified', [])),
                    'redactions_applied': redactions,
                }

                if include_full_text:
                    sample['query'] = redacted_query[:500] + ('...' if len(redacted_query) > 500 else '')
                    sample['response_preview'] = redacted_response[:200] + ('...' if len(redacted_response) > 200 else '')
                else:
                    sample['query_preview'] = redacted_query[:100] + ('...' if len(redacted_query) > 100 else '')

                preview['sample_chats'].append(sample)
            except (json.JSONDecodeError, IOError):
                continue

    # Count sessions
    if SESSIONS_DIR.exists():
        preview['summary']['total_sessions'] = len(list(SESSIONS_DIR.glob("*.json")))

    # Calculate date range
    if all_timestamps:
        sorted_ts = sorted([t for t in all_timestamps if t])
        if sorted_ts:
            preview['summary']['date_range']['earliest'] = sorted_ts[0][:10]
            preview['summary']['date_range']['latest'] = sorted_ts[-1][:10]

    # Convert sets to counts for JSON serialization
    preview['summary']['unique_files'] = len(preview['summary']['unique_files'])
    preview['summary']['unique_tools'] = len(preview['summary']['unique_tools'])

    return preview


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


def add_chat_feedback(
    chat_id: str,
    rating: str,
    comment: Optional[str] = None,
    force: bool = False
) -> bool:
    """Add or update user feedback for a chat entry.

    Args:
        chat_id: The chat ID to add feedback to.
        rating: Rating value (good, bad, neutral).
        comment: Optional feedback comment.
        force: If True, overwrite existing feedback.

    Returns:
        True if feedback was added/updated, False if chat not found or already has feedback.
    """
    # Validate rating
    valid_ratings = {"good", "bad", "neutral"}
    if rating not in valid_ratings:
        raise ValueError(f"Invalid rating '{rating}'. Must be one of: {', '.join(valid_ratings)}")

    # Find the chat file
    chat_file = find_chat_file(chat_id)
    if not chat_file:
        return False

    try:
        # Load existing chat data
        with open(chat_file, 'r', encoding='utf-8') as f:
            chat_data = json.load(f)

        # Check if feedback already exists
        existing_feedback = chat_data.get('user_feedback')
        if existing_feedback and not force:
            # Check if it's a dict (new format) or string (legacy)
            if isinstance(existing_feedback, dict):
                return False
            # Legacy string format - allow upgrade to dict format

        # Add feedback
        chat_data['user_feedback'] = {
            'rating': rating,
            'comment': comment,
            'timestamp': datetime.now().isoformat(),
        }

        # Save atomically
        atomic_write_json(chat_file, chat_data)
        return True

    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error updating chat feedback: {e}")
        return False


def list_chats_needing_feedback(limit: int = 10) -> List[Dict[str, Any]]:
    """List recent chats that don't have feedback yet.

    Args:
        limit: Maximum number of chats to return.

    Returns:
        List of chat info dicts with id, timestamp, query preview, and has_feedback status.
    """
    if not CHATS_DIR.exists():
        return []

    chats = []

    # Iterate through date directories in reverse order (most recent first)
    date_dirs = sorted(CHATS_DIR.iterdir(), reverse=True)
    for date_dir in date_dirs:
        if not date_dir.is_dir():
            continue

        # Get all chat files in this date directory
        chat_files = sorted(date_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)

        for chat_file in chat_files:
            if len(chats) >= limit:
                break

            try:
                with open(chat_file, 'r', encoding='utf-8') as f:
                    chat_data = json.load(f)

                # Check if chat has feedback
                feedback = chat_data.get('user_feedback')
                has_feedback = False
                feedback_rating = None

                if feedback:
                    if isinstance(feedback, dict):
                        has_feedback = True
                        feedback_rating = feedback.get('rating')
                    elif isinstance(feedback, str):
                        # Legacy string format
                        has_feedback = True
                        feedback_rating = feedback

                chat_info = {
                    'id': chat_data.get('id', 'unknown'),
                    'timestamp': chat_data.get('timestamp', ''),
                    'query': chat_data.get('query', '')[:100],  # First 100 chars
                    'has_feedback': has_feedback,
                    'feedback_rating': feedback_rating,
                    'session_id': chat_data.get('session_id', ''),
                }
                chats.append(chat_info)

            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error reading chat {chat_file}: {e}")
                continue

        if len(chats) >= limit:
            break

    return chats


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
    tool_outputs: List[Dict]  # Tool results with output, success status
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
    current_tool_outputs = []
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
                            tool_outputs=current_tool_outputs,
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
                    current_tool_outputs = []
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

                            elif block_type == 'tool_result':
                                # Capture tool outputs with truncation
                                output_content = block.get('content', '')
                                is_error = block.get('is_error', False)

                                # Truncate large outputs (keep first 500 chars)
                                MAX_OUTPUT_LENGTH = 500
                                if isinstance(output_content, str):
                                    truncated_output = output_content[:MAX_OUTPUT_LENGTH]
                                    if len(output_content) > MAX_OUTPUT_LENGTH:
                                        truncated_output += '... [truncated]'
                                else:
                                    # Handle non-string outputs (convert to string)
                                    truncated_output = str(output_content)[:MAX_OUTPUT_LENGTH]

                                current_tool_outputs.append({
                                    'output': truncated_output,
                                    'success': not is_error,
                                    'is_error': is_error,
                                })

        # Don't forget the last exchange
        if current_query and current_response_parts:
            exchanges.append(TranscriptExchange(
                query=current_query,
                response=' '.join(current_response_parts),
                tools_used=current_tools,
                tool_inputs=current_tool_inputs,
                tool_outputs=current_tool_outputs,
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

        elif tool == 'NotebookEdit':
            # NotebookEdit modifies notebooks
            path = inp.get('notebook_path', '')
            if path:
                files_modified.add(path)

        elif tool == 'Bash':
            # Try to extract file paths from command
            cmd = inp.get('command', '')

            # Common file extensions to track
            FILE_EXTENSIONS = (
                '.py', '.md', '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg',
                '.txt', '.rst', '.sh', '.bash', '.zsh',
                '.js', '.ts', '.jsx', '.tsx', '.mjs', '.cjs',
                '.html', '.css', '.scss', '.less',
                '.c', '.cpp', '.h', '.hpp', '.cc',
                '.java', '.kt', '.scala',
                '.go', '.rs', '.rb', '.php', '.pl',
                '.sql', '.graphql',
                '.xml', '.csv', '.env',
                '.dockerfile', 'Dockerfile', 'Makefile', 'Jenkinsfile'
            )

            # Use shlex.split() for safer parsing (handles quoted paths)
            try:
                words = shlex.split(cmd)
            except ValueError:
                # Fallback to simple split if shlex fails
                words = cmd.split()

            for word in words:
                # Strip quotes that might remain
                word = word.strip('\'"')

                # Skip flags/options
                if word.startswith('-'):
                    # But check if it's a flag with a value like --cov="file.py"
                    if '=' in word:
                        # Extract the value part after =
                        _, value = word.split('=', 1)
                        value = value.strip('\'"')
                        if any(value.endswith(ext) for ext in FILE_EXTENSIONS):
                            files_referenced.add(value)
                    continue

                # Check if it ends with a tracked extension
                if any(word.endswith(ext) for ext in FILE_EXTENSIONS):
                    files_referenced.add(word)
                # Also catch special files without extensions (case-insensitive)
                elif any(word.lower().endswith(name.lower()) for name in ('Dockerfile', 'Makefile', 'Jenkinsfile')):
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
    queries = []
    tool_usage_count = {}  # Track tool usage frequency

    actions_saved = 0
    for ex in exchanges:
        files_ref, files_mod = extract_files_from_tool_inputs(ex.tool_inputs)
        all_files_ref.update(files_ref)
        all_files_mod.update(files_mod)
        total_tools.update(ex.tools_used)

        # Count tool usage
        for tool in ex.tools_used:
            tool_usage_count[tool] = tool_usage_count.get(tool, 0) + 1

        # Collect queries for session summary (truncate for readability)
        query_preview = ex.query[:100].strip()
        if query_preview and query_preview not in queries:
            queries.append(query_preview)

        # Save individual actions for each tool use
        if save_exchanges:
            for i, (tool_name, tool_input) in enumerate(zip(ex.tools_used, ex.tool_inputs)):
                try:
                    # Determine action type from tool name
                    action_type_map = {
                        'Read': 'read', 'Glob': 'search', 'Grep': 'search',
                        'Edit': 'edit', 'Write': 'edit', 'MultiEdit': 'edit',
                        'Bash': 'command', 'Task': 'delegate',
                        'WebFetch': 'fetch', 'WebSearch': 'search',
                    }
                    action_type = action_type_map.get(tool_name, 'other')

                    # Extract target from tool input
                    target = ''
                    if isinstance(tool_input, dict):
                        target = tool_input.get('file_path') or tool_input.get('path') or \
                                tool_input.get('pattern') or tool_input.get('command') or \
                                tool_input.get('query') or tool_input.get('url') or ''

                    # Get output if available
                    tool_output = ex.tool_outputs[i] if i < len(ex.tool_outputs) else {}
                    success = tool_output.get('success', True) if isinstance(tool_output, dict) else True

                    action = ActionEntry(
                        id=f"A-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{session_id[:4]}-{actions_saved:03d}",
                        timestamp=ex.timestamp or datetime.now().isoformat(),
                        session_id=session_id,
                        action_type=action_type,
                        target=str(target)[:500],  # Truncate long targets
                        context={'tool': tool_name, 'input': tool_input},
                        success=success,
                        result_summary=str(tool_output)[:200] if tool_output else None,
                    )
                    save_action(action, validate=True)
                    actions_saved += 1
                except Exception as e:
                    logger.debug(f"Error saving action for {tool_name}: {e}")

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
                    tool_outputs=ex.tool_outputs,  # Include tool outputs
                    query_tokens=len(ex.query.split()),
                    response_tokens=len(ex.response.split()),
                )
                save_chat_entry(entry, validate=True)
                add_chat_to_session(entry.id)
                saved_count += 1
            except Exception as e:
                logger.error(f"Error saving exchange: {e}")

    # Save lightweight session summary for git tracking
    if save_exchanges and exchanges:
        try:
            # Calculate session duration from first to last exchange
            first_timestamp = exchanges[0].timestamp
            last_timestamp = exchanges[-1].timestamp

            duration_seconds = 0
            if first_timestamp and last_timestamp:
                try:
                    start = datetime.fromisoformat(first_timestamp.replace('Z', '+00:00'))
                    end = datetime.fromisoformat(last_timestamp.replace('Z', '+00:00'))
                    duration_seconds = int((end - start).total_seconds())
                except (ValueError, AttributeError):
                    pass

            # Build session summary
            session_summary = {
                'session_id': session_id,
                'timestamp': first_timestamp or datetime.now().isoformat(),
                'duration_seconds': duration_seconds,
                'exchanges': len(exchanges),
                'queries': queries[:10],  # Limit to first 10 queries
                'tools_used': tool_usage_count,
                'files_referenced': list(all_files_ref)[:50],  # Limit for size
                'files_modified': list(all_files_mod)[:50],  # Limit for size
            }

            # Save to tracked directory (git-committable)
            save_session_lite(session_summary)

        except Exception as e:
            # Don't fail the whole operation if session lite save fails
            logger.warning(f"Failed to save session lite: {e}")

    return {
        'status': 'success',
        'exchanges': len(exchanges),
        'saved': saved_count,
        'actions_saved': actions_saved,
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
    tool_outputs: List[Dict] = field(default_factory=list)  # Tool results with output, success status

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
    for dir_path in [COMMITS_DIR, COMMITS_LITE_DIR, SESSIONS_DIR, CHATS_DIR, ACTIONS_DIR]:
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

    # Also write to CALI for O(1) lookups and git-friendly storage
    if cali_put('commit', context.hash, data):
        logger.debug(f"CALI: stored commit {context.hash[:8]}")

    # Update chat entries to link back to this commit
    if link_session and context.related_chats:
        linked = link_commit_to_session_chats(context.hash)
        if linked:
            print(f"Linked {len(linked)} chat(s) to commit {context.hash[:8]}")


def save_commit_lite(context: CommitContext):
    """Save lightweight commit metadata (no diff hunks) for git tracking.

    Appends to a single JSONL file (.git-ml/tracked/commits.jsonl) for easy
    git tracking. Each commit is one line, making diffs clean and merge-friendly.

    Returns the path to the JSONL file.
    """
    TRACKED_DIR.mkdir(parents=True, exist_ok=True)

    # Create lightweight data without hunks
    lite_data = {
        "hash": context.hash,
        "message": context.message,
        "author": context.author,
        "timestamp": context.timestamp,
        "branch": context.branch,
        "files_changed": context.files_changed,
        "insertions": context.insertions,
        "deletions": context.deletions,
        "hour_of_day": context.hour_of_day,
        "day_of_week": context.day_of_week,
        "is_merge": context.is_merge,
        "is_initial": context.is_initial,
        "parent_count": context.parent_count,
        # Note: hunks excluded - that's what makes it "lite"
    }

    # Add optional fields if present
    if context.seconds_since_last_commit is not None:
        lite_data["seconds_since_last_commit"] = context.seconds_since_last_commit
    if context.ci_result:
        lite_data["ci_result"] = context.ci_result
    if context.session_id:
        lite_data["session_id"] = context.session_id

    # Check if this commit hash already exists in the file (idempotent)
    if COMMITS_LITE_FILE.exists():
        with open(COMMITS_LITE_FILE, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        existing = json.loads(line)
                        if existing.get("hash") == context.hash:
                            return COMMITS_LITE_FILE  # Already recorded
                    except json.JSONDecodeError:
                        continue

    # Append to JSONL file (one JSON object per line)
    with open(COMMITS_LITE_FILE, 'a') as f:
        f.write(json.dumps(lite_data, separators=(',', ':')) + '\n')

    return COMMITS_LITE_FILE


def save_session_lite(session_summary: Dict[str, Any]):
    """Save lightweight session summary for git tracking.

    Appends to .git-ml/tracked/sessions.jsonl with essential session data:
    - session_id, timestamp, duration
    - files_read, files_edited
    - queries (list of user queries)
    - tools_used (count by tool type)
    - commits_made (list of commit hashes)

    This captures the "why" and "how" that commit data alone doesn't provide.
    """
    TRACKED_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure required fields
    required = ["session_id", "timestamp"]
    for field in required:
        if field not in session_summary:
            raise ValueError(f"Session summary missing required field: {field}")

    session_id = session_summary.get("session_id")

    # CALI: O(1) existence check (fast path)
    if cali_exists('session', session_id):
        logger.debug(f"CALI: session {session_id} already exists")
        return SESSIONS_LITE_FILE

    # Fallback: Check JSONL file (O(n) scan)
    if SESSIONS_LITE_FILE.exists():
        with open(SESSIONS_LITE_FILE, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        existing = json.loads(line)
                        if existing.get("session_id") == session_id:
                            return SESSIONS_LITE_FILE  # Already recorded
                    except json.JSONDecodeError:
                        continue

    # Append to JSONL file (legacy format)
    with open(SESSIONS_LITE_FILE, 'a') as f:
        f.write(json.dumps(session_summary, separators=(',', ':')) + '\n')

    # Also write to CALI for O(1) lookups and git-friendly storage
    if cali_put('session', session_id, session_summary):
        logger.debug(f"CALI: stored session {session_id}")

    return SESSIONS_LITE_FILE


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

    # Also write to CALI for O(1) lookups and git-friendly storage
    if cali_put('chat', entry.id, data):
        logger.debug(f"CALI: stored chat {entry.id}")


def log_chat(
    query: str,
    response: str,
    session_id: Optional[str] = None,
    files_referenced: Optional[List[str]] = None,
    files_modified: Optional[List[str]] = None,
    tools_used: Optional[List[str]] = None,
    tool_outputs: Optional[List[Dict]] = None,
    user_feedback: Optional[str] = None,
    skip_redaction: bool = False,
) -> ChatEntry:
    """Log a chat query/response pair.

    If no session_id is provided, uses the current session (creating one if needed).
    The chat is automatically registered with the session for commit linking.

    Sensitive data (API keys, passwords, tokens) is automatically redacted before storage
    unless skip_redaction=True.
    """
    # Use current session or create one
    if session_id is None:
        session_id = get_or_create_session()

    # Redact sensitive data before storage
    if not skip_redaction:
        query = redact_sensitive_data(query)
        response = redact_sensitive_data(response)

    entry = ChatEntry(
        id=generate_chat_id(),
        timestamp=datetime.now().isoformat(),
        session_id=session_id,
        query=query,
        response=response,
        files_referenced=files_referenced or [],
        files_modified=files_modified or [],
        tools_used=tools_used or [],
        tool_outputs=tool_outputs or [],
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

    # Also write to CALI for O(1) lookups and git-friendly storage
    if cali_put('action', entry.id, data):
        logger.debug(f"CALI: stored action {entry.id}")


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
# DATA EXPORT FOR TRAINING
# ============================================================================

def _summarize_diff(hunks: List[Dict]) -> str:
    """Summarize diff hunks into a concise description for training."""
    if not hunks:
        return ""

    # Group by file
    files = {}
    for hunk in hunks:
        file = hunk.get('file', 'unknown')
        if file not in files:
            files[file] = {'add': 0, 'delete': 0, 'modify': 0}
        change_type = hunk.get('change_type', 'modify')
        files[file][change_type] = files[file].get(change_type, 0) + 1

    # Create summary
    parts = []
    for file, changes in files.items():
        change_desc = []
        if changes['add'] > 0:
            change_desc.append(f"+{changes['add']}")
        if changes['delete'] > 0:
            change_desc.append(f"-{changes['delete']}")
        if changes['modify'] > 0:
            change_desc.append(f"~{changes['modify']}")
        parts.append(f"{file}: {' '.join(change_desc)}")

    return '; '.join(parts[:10])  # Limit to first 10 files


def _export_jsonl(records: List[Dict], output_path: Path):
    """Export records as JSONL (one JSON per line)."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def _export_csv(records: List[Dict], output_path: Path,
                truncate_input: int = CSV_DEFAULT_TRUNCATE_QUERY,
                truncate_output: int = CSV_DEFAULT_TRUNCATE_RESPONSE,
                truncate_files: int = CSV_DEFAULT_TRUNCATE_QUERY):
    """
    Export records as CSV with configurable truncation.

    Args:
        records: List of records to export
        output_path: Path to output CSV file
        truncate_input: Max length for input/query fields (0 = no truncation)
        truncate_output: Max length for output/response fields (0 = no truncation)
        truncate_files: Max length for files field (0 = no truncation)
    """
    import csv

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'type', 'timestamp', 'input', 'output',
            'session_id', 'files', 'tools_used'
        ])
        writer.writeheader()

        for record in records:
            context = record.get('context', {})

            # Apply truncation (0 means no truncation)
            input_text = record.get('input', '')
            output_text = record.get('output', '')
            files_text = '; '.join(context.get('files', []))

            if truncate_input > 0:
                input_text = input_text[:truncate_input]
            if truncate_output > 0:
                output_text = output_text[:truncate_output]
            if truncate_files > 0:
                files_text = files_text[:truncate_files]

            row = {
                'type': record.get('type', ''),
                'timestamp': record.get('timestamp', ''),
                'input': input_text,
                'output': output_text,
                'session_id': context.get('session_id', ''),
                'files': files_text,
                'tools_used': '; '.join(context.get('tools_used', [])),
            }
            writer.writerow(row)


def _export_huggingface(records: List[Dict], output_path: Path):
    """Export records in HuggingFace Dataset dict format."""
    # HuggingFace datasets format: dict of lists
    dataset = {
        'type': [],
        'timestamp': [],
        'input': [],
        'output': [],
        'session_id': [],
        'files': [],
        'tools_used': [],
    }

    for record in records:
        context = record.get('context', {})
        dataset['type'].append(record.get('type', ''))
        dataset['timestamp'].append(record.get('timestamp', ''))
        dataset['input'].append(record.get('input', ''))
        dataset['output'].append(record.get('output', ''))
        dataset['session_id'].append(context.get('session_id', ''))
        dataset['files'].append(context.get('files', []))
        dataset['tools_used'].append(context.get('tools_used', []))

    # Save as JSON in HuggingFace format
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)


def export_data(format: str, output_path: Path,
                truncate_input: int = CSV_DEFAULT_TRUNCATE_QUERY,
                truncate_output: int = CSV_DEFAULT_TRUNCATE_RESPONSE,
                truncate_files: int = CSV_DEFAULT_TRUNCATE_QUERY) -> Dict[str, Any]:
    """Export collected ML data in training-ready formats.

    Args:
        format: Output format (jsonl, csv, huggingface)
        output_path: Path to write the exported data
        truncate_input: Max length for input fields in CSV (0 = no truncation)
        truncate_output: Max length for output fields in CSV (0 = no truncation)
        truncate_files: Max length for files field in CSV (0 = no truncation)

    Returns:
        Stats dict with counts and file paths

    Raises:
        ValueError: If format is invalid
    """
    ensure_dirs()

    # Collect all data
    all_records = []

    # Load commits
    if COMMITS_DIR.exists():
        for commit_file in COMMITS_DIR.glob("*.json"):
            try:
                with open(commit_file, 'r', encoding='utf-8') as f:
                    commit_data = json.load(f)

                # Transform commit to training format
                record = {
                    "type": "commit",
                    "timestamp": commit_data.get('timestamp', ''),
                    "input": commit_data.get('message', ''),
                    "output": _summarize_diff(commit_data.get('hunks', [])),
                    "context": {
                        "files": commit_data.get('files_changed', []),
                        "session_id": commit_data.get('session_id', ''),
                        "tools_used": [],
                        "insertions": commit_data.get('insertions', 0),
                        "deletions": commit_data.get('deletions', 0),
                        "branch": commit_data.get('branch', ''),
                    }
                }
                all_records.append(record)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error reading commit {commit_file}: {e}")

    # Load chats
    if CHATS_DIR.exists():
        for chat_file in CHATS_DIR.glob("**/*.json"):
            try:
                with open(chat_file, 'r', encoding='utf-8') as f:
                    chat_data = json.load(f)

                record = {
                    "type": "chat",
                    "timestamp": chat_data.get('timestamp', ''),
                    "input": chat_data.get('query', ''),
                    "output": chat_data.get('response', ''),
                    "context": {
                        "files": chat_data.get('files_referenced', []) + chat_data.get('files_modified', []),
                        "session_id": chat_data.get('session_id', ''),
                        "tools_used": chat_data.get('tools_used', []),
                    }
                }
                all_records.append(record)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error reading chat {chat_file}: {e}")

    # Sort by timestamp
    all_records.sort(key=lambda r: r['timestamp'])

    # Export based on format
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "jsonl":
        _export_jsonl(all_records, output_path)
    elif format == "csv":
        _export_csv(all_records, output_path,
                   truncate_input=truncate_input,
                   truncate_output=truncate_output,
                   truncate_files=truncate_files)
    elif format == "huggingface":
        _export_huggingface(all_records, output_path)
    else:
        raise ValueError(f"Unknown format: {format}")

    return {
        "format": format,
        "output_path": str(output_path),
        "records": len(all_records),
        "commits": sum(1 for r in all_records if r['type'] == 'commit'),
        "chats": sum(1 for r in all_records if r['type'] == 'chat'),
    }


# ============================================================================
# STATISTICS AND ESTIMATION
# ============================================================================

def count_jsonl_lines(filepath: Path) -> int:
    """Count non-empty lines in a JSONL file."""
    if not filepath.exists():
        return 0
    count = 0
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def count_data() -> Dict[str, int]:
    """Count collected data entries."""
    ensure_dirs()

    counts = {
        "commits": 0,
        "commits_lite": 0,
        "sessions_lite": 0,
        "chats": 0,
        "actions": 0,
        "sessions": 0,
        "prs": 0,
        "issues": 0,
    }

    # Count commits (full)
    if COMMITS_DIR.exists():
        counts["commits"] = len(list(COMMITS_DIR.glob("*.json")))

    # Count commits (lightweight - from JSONL)
    counts["commits_lite"] = count_jsonl_lines(COMMITS_LITE_FILE)
    # Also count legacy individual files if they exist
    if COMMITS_LITE_DIR.exists():
        counts["commits_lite"] += len(list(COMMITS_LITE_DIR.glob("*.json")))

    # Count sessions (lightweight - from JSONL)
    counts["sessions_lite"] = count_jsonl_lines(SESSIONS_LITE_FILE)

    # Count chats
    if CHATS_DIR.exists():
        counts["chats"] = len(list(CHATS_DIR.glob("**/*.json")))

    # Count actions
    if ACTIONS_DIR.exists():
        counts["actions"] = len(list(ACTIONS_DIR.glob("**/*.json")))

    # Count sessions
    if SESSIONS_DIR.exists():
        counts["sessions"] = len(list(SESSIONS_DIR.glob("*.json")))

    # Count GitHub data
    if GITHUB_DIR.exists():
        counts["prs"] = len(list(GITHUB_DIR.glob("pr_*.json")))
        counts["issues"] = len(list(GITHUB_DIR.glob("issue_*.json")))

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

    # Map milestone keys to actual count keys
    # MILESTONES uses "commits" but we want to count "commits_lite" (tracked in git)
    # MILESTONES uses "sessions" but we want to count "sessions_lite" (tracked in git)
    count_key_mapping = {
        "commits": "commits_lite",  # Use lite commits for milestone progress
        "sessions": "sessions_lite",  # Use lite sessions for milestone progress
    }

    progress = {}
    for milestone, requirements in MILESTONES.items():
        milestone_progress = {}
        for data_type, required in requirements.items():
            # Use mapped key if available, otherwise use data_type directly
            count_key = count_key_mapping.get(data_type, data_type)
            current = counts.get(count_key, 0)
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
    print(f"   Commits (full):  {counts['commits']:,}")
    print(f"   Commits (lite):  {counts.get('commits_lite', 0):,}  ← tracked in git")
    print(f"   Sessions (lite): {counts.get('sessions_lite', 0):,}  ← tracked in git")
    print(f"   Chats:           {counts['chats']:,}")
    print(f"   Actions:         {counts['actions']:,}")
    print(f"   Sessions (full): {counts['sessions']:,}")
    if counts.get('prs', 0) > 0 or counts.get('issues', 0) > 0:
        print(f"   PRs:             {counts.get('prs', 0):,}")
        print(f"   Issues:          {counts.get('issues', 0):,}")

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
# DATA QUALITY ANALYSIS
# ============================================================================

def analyze_data_quality() -> Dict[str, Any]:
    """Analyze data quality across all collected ML data.

    Returns:
        Dictionary with completeness, diversity, anomalies, and quality score.
    """
    ensure_dirs()

    # Initialize metrics containers
    completeness = {
        'chats_complete': 0,
        'chats_total': 0,
        'commits_with_ci': 0,
        'commits_total': 0,
        'sessions_with_commits': 0,
        'sessions_total': 0,
        'chats_with_feedback': 0,
    }

    diversity = {
        'unique_files': set(),
        'unique_tools': {},
        'query_lengths': [],
        'response_lengths': [],
    }

    anomalies = {
        'empty_responses': 0,
        'zero_file_commits': 0,
        'empty_sessions': 0,
        'potential_duplicates': 0,
    }

    # Track duplicates (timestamp + content hash)
    seen_entries = set()

    # Analyze commits
    if COMMITS_DIR.exists():
        for commit_file in COMMITS_DIR.glob("*.json"):
            try:
                with open(commit_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                completeness['commits_total'] += 1

                # Check CI results
                if data.get('ci_result'):
                    completeness['commits_with_ci'] += 1

                # Track files
                diversity['unique_files'].update(data.get('files_changed', []))

                # Check anomalies
                if not data.get('files_changed'):
                    anomalies['zero_file_commits'] += 1

                # Check duplicates (timestamp + message hash)
                entry_key = (data.get('timestamp', ''),
                           hashlib.md5(data.get('message', '').encode()).hexdigest()[:8])
                if entry_key in seen_entries:
                    anomalies['potential_duplicates'] += 1
                else:
                    seen_entries.add(entry_key)

            except (json.JSONDecodeError, IOError):
                continue

    # Analyze chats
    if CHATS_DIR.exists():
        for chat_file in CHATS_DIR.glob("**/*.json"):
            try:
                with open(chat_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                completeness['chats_total'] += 1

                # Check completeness (all required fields from CHAT_SCHEMA)
                errors = validate_schema(data, CHAT_SCHEMA, "chat")
                if not errors:
                    completeness['chats_complete'] += 1

                # Check feedback
                if data.get('user_feedback'):
                    completeness['chats_with_feedback'] += 1

                # Track diversity
                diversity['unique_files'].update(data.get('files_referenced', []))
                diversity['unique_files'].update(data.get('files_modified', []))

                for tool in data.get('tools_used', []):
                    diversity['unique_tools'][tool] = diversity['unique_tools'].get(tool, 0) + 1

                query = data.get('query', '')
                response = data.get('response', '')

                diversity['query_lengths'].append(len(query))
                diversity['response_lengths'].append(len(response))

                # Check anomalies
                if not response or len(response.strip()) == 0:
                    anomalies['empty_responses'] += 1

                # Check duplicates (timestamp + query hash)
                entry_key = (data.get('timestamp', ''),
                           hashlib.md5(query.encode()).hexdigest()[:8])
                if entry_key in seen_entries:
                    anomalies['potential_duplicates'] += 1
                else:
                    seen_entries.add(entry_key)

            except (json.JSONDecodeError, IOError):
                continue

    # Analyze sessions
    if SESSIONS_DIR.exists():
        for session_file in SESSIONS_DIR.glob("*.json"):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                completeness['sessions_total'] += 1

                # Check if session has chats
                if not data.get('chat_ids'):
                    anomalies['empty_sessions'] += 1

            except (json.JSONDecodeError, IOError):
                continue

    # Count sessions with commits by checking commits with session_id
    session_ids_with_commits = set()
    if COMMITS_DIR.exists():
        for commit_file in COMMITS_DIR.glob("*.json"):
            try:
                with open(commit_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    session_id = data.get('session_id')
                    if session_id:
                        session_ids_with_commits.add(session_id)
            except (json.JSONDecodeError, IOError):
                continue

    completeness['sessions_with_commits'] = len(session_ids_with_commits)

    # Calculate percentages for completeness
    completeness_metrics = {
        'chats_complete_pct': (completeness['chats_complete'] / max(1, completeness['chats_total'])) * 100,
        'commits_with_ci_pct': (completeness['commits_with_ci'] / max(1, completeness['commits_total'])) * 100,
        'sessions_with_commits_pct': (completeness['sessions_with_commits'] / max(1, completeness['sessions_total'])) * 100,
        'chats_with_feedback_pct': (completeness['chats_with_feedback'] / max(1, completeness['chats_total'])) * 100,
    }

    # Calculate diversity statistics
    diversity_stats = {
        'unique_files': len(diversity['unique_files']),
        'unique_tools': len(diversity['unique_tools']),
        'tool_distribution': diversity['unique_tools'],
        'query_length_min': min(diversity['query_lengths']) if diversity['query_lengths'] else 0,
        'query_length_avg': sum(diversity['query_lengths']) / max(1, len(diversity['query_lengths'])) if diversity['query_lengths'] else 0,
        'query_length_max': max(diversity['query_lengths']) if diversity['query_lengths'] else 0,
        'response_length_min': min(diversity['response_lengths']) if diversity['response_lengths'] else 0,
        'response_length_avg': sum(diversity['response_lengths']) / max(1, len(diversity['response_lengths'])) if diversity['response_lengths'] else 0,
        'response_length_max': max(diversity['response_lengths']) if diversity['response_lengths'] else 0,
    }

    # Calculate quality score (0-100)
    # Weighted components:
    # - Completeness: 40%
    # - Low anomalies: 30%
    # - Diversity: 30%

    # Completeness score (average of all completeness metrics)
    completeness_score = (
        completeness_metrics['chats_complete_pct'] * 0.4 +
        completeness_metrics['commits_with_ci_pct'] * 0.2 +
        completeness_metrics['sessions_with_commits_pct'] * 0.3 +
        completeness_metrics['chats_with_feedback_pct'] * 0.1
    )

    # Anomaly score (penalize based on anomaly percentage)
    total_entries = completeness['chats_total'] + completeness['commits_total'] + completeness['sessions_total']
    total_anomalies = (anomalies['empty_responses'] + anomalies['zero_file_commits'] +
                      anomalies['empty_sessions'] + anomalies['potential_duplicates'])
    anomaly_rate = total_anomalies / max(1, total_entries)
    anomaly_score = max(0, 100 - (anomaly_rate * 200))  # Cap at 0, scale anomalies harshly

    # Diversity score (based on having diverse tools and files)
    # Good diversity: >5 tools, >50 files = 100%, scale down from there
    tool_score = min(100, (diversity_stats['unique_tools'] / 5.0) * 100)
    file_score = min(100, (diversity_stats['unique_files'] / 50.0) * 100)
    diversity_score = (tool_score + file_score) / 2

    # Overall quality score
    quality_score = int(
        completeness_score * 0.4 +
        anomaly_score * 0.3 +
        diversity_score * 0.3
    )

    return {
        'completeness': {
            'chats_complete': completeness['chats_complete'],
            'chats_total': completeness['chats_total'],
            'chats_complete_pct': completeness_metrics['chats_complete_pct'],
            'commits_with_ci': completeness['commits_with_ci'],
            'commits_total': completeness['commits_total'],
            'commits_with_ci_pct': completeness_metrics['commits_with_ci_pct'],
            'sessions_with_commits': completeness['sessions_with_commits'],
            'sessions_total': completeness['sessions_total'],
            'sessions_with_commits_pct': completeness_metrics['sessions_with_commits_pct'],
            'chats_with_feedback': completeness['chats_with_feedback'],
            'chats_with_feedback_pct': completeness_metrics['chats_with_feedback_pct'],
        },
        'diversity': diversity_stats,
        'anomalies': anomalies,
        'quality_score': quality_score,
    }


def print_quality_report():
    """Print a comprehensive data quality report."""
    result = analyze_data_quality()

    comp = result['completeness']
    div = result['diversity']
    anom = result['anomalies']
    score = result['quality_score']

    print("\n" + "=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)

    print("\n📊 Completeness:")
    print(f"   Chats with all fields:    {comp['chats_complete_pct']:>3.0f}% ({comp['chats_complete']}/{comp['chats_total']})")
    print(f"   Commits with CI results:  {comp['commits_with_ci_pct']:>3.0f}% ({comp['commits_with_ci']}/{comp['commits_total']})")
    print(f"   Sessions with commits:    {comp['sessions_with_commits_pct']:>3.0f}% ({comp['sessions_with_commits']}/{comp['sessions_total']})")
    print(f"   Chats with feedback:      {comp['chats_with_feedback_pct']:>3.0f}% ({comp['chats_with_feedback']}/{comp['chats_total']})")

    print("\n📈 Diversity:")
    print(f"   Unique files:             {div['unique_files']}")
    print(f"   Unique tools:             {div['unique_tools']}")
    if div['tool_distribution']:
        print("   Tool usage:")
        for tool, count in sorted(div['tool_distribution'].items(), key=lambda x: -x[1])[:8]:
            print(f"      {tool}: {count}")
        if len(div['tool_distribution']) > 8:
            print(f"      ... and {len(div['tool_distribution']) - 8} more")
    print(f"   Query length:             min={div['query_length_min']}, avg={div['query_length_avg']:.0f}, max={div['query_length_max']} chars")
    print(f"   Response length:          min={div['response_length_min']}, avg={div['response_length_avg']:.0f}, max={div['response_length_max']} chars")

    print("\n⚠️  Anomalies:")
    print(f"   Empty responses:          {anom['empty_responses']}")
    print(f"   Zero-file commits:        {anom['zero_file_commits']}")
    print(f"   Empty sessions:           {anom['empty_sessions']}")
    print(f"   Potential duplicates:     {anom['potential_duplicates']}")

    print(f"\n🎯 Quality Score: {score}/100")
    print("=" * 60 + "\n")


# ============================================================================
# SHARED PATTERN AGGREGATION (safe to commit)
# ============================================================================

def generate_file_correlations() -> Dict[str, Any]:
    """Generate file correlation patterns from commit history.

    Identifies which files frequently change together - useful for
    predicting likely files to edit based on task description.

    Returns:
        Dict with file pairs and their co-occurrence counts
    """
    correlations: Dict[tuple, int] = {}
    file_counts: Dict[str, int] = {}

    if not COMMITS_DIR.exists():
        return {'correlations': [], 'file_counts': {}, 'total_commits': 0}

    total_commits = 0
    for commit_file in COMMITS_DIR.glob("*.json"):
        try:
            with open(commit_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            files = data.get('files_changed', [])
            total_commits += 1

            # Count individual file occurrences
            for file in files:
                file_counts[file] = file_counts.get(file, 0) + 1

            # Count file pairs (co-occurrence)
            for i, f1 in enumerate(files):
                for f2 in files[i+1:]:
                    pair = tuple(sorted([f1, f2]))
                    correlations[pair] = correlations.get(pair, 0) + 1

        except (json.JSONDecodeError, IOError):
            continue

    # Convert to list and sort by frequency
    corr_list = [
        {'files': list(pair), 'count': count, 'pct': round(count / total_commits * 100, 1)}
        for pair, count in correlations.items()
        if count >= 2  # Only include pairs that co-occur at least twice
    ]
    corr_list.sort(key=lambda x: -x['count'])

    return {
        'correlations': corr_list[:100],  # Top 100 pairs
        'file_counts': dict(sorted(file_counts.items(), key=lambda x: -x[1])[:50]),
        'total_commits': total_commits,
        'generated_at': datetime.now().isoformat(),
    }


def generate_commit_patterns() -> Dict[str, Any]:
    """Generate commit message patterns and statistics.

    Analyzes commit message styles, lengths, prefixes (feat/fix/etc),
    and temporal patterns.

    Returns:
        Dict with pattern statistics
    """
    patterns = {
        'prefixes': {},  # feat:, fix:, etc.
        'lengths': [],
        'hour_distribution': {str(h): 0 for h in range(24)},
        'day_distribution': {},
        'total_commits': 0,
    }

    if not COMMITS_DIR.exists():
        return patterns

    for commit_file in COMMITS_DIR.glob("*.json"):
        try:
            with open(commit_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            message = data.get('message', '')
            patterns['total_commits'] += 1
            patterns['lengths'].append(len(message))

            # Extract prefix (feat:, fix:, docs:, etc.)
            if ':' in message[:20]:
                prefix = message.split(':')[0].lower().strip()
                if len(prefix) <= 15:  # Reasonable prefix length
                    patterns['prefixes'][prefix] = patterns['prefixes'].get(prefix, 0) + 1

            # Temporal patterns
            hour = data.get('hour_of_day', 0)
            patterns['hour_distribution'][str(hour)] = patterns['hour_distribution'].get(str(hour), 0) + 1

            day = data.get('day_of_week', 'Unknown')
            patterns['day_distribution'][day] = patterns['day_distribution'].get(day, 0) + 1

        except (json.JSONDecodeError, IOError):
            continue

    # Calculate length statistics
    if patterns['lengths']:
        lengths = patterns['lengths']
        patterns['length_stats'] = {
            'min': min(lengths),
            'max': max(lengths),
            'avg': round(sum(lengths) / len(lengths), 1),
            'median': sorted(lengths)[len(lengths) // 2],
        }
    patterns.pop('lengths')  # Don't include raw list

    patterns['generated_at'] = datetime.now().isoformat()
    return patterns


def generate_tool_effectiveness() -> Dict[str, Any]:
    """Generate tool usage effectiveness patterns.

    Analyzes which tools are used most, success rates, and
    tool sequences that lead to commits.

    Returns:
        Dict with tool usage statistics
    """
    tool_stats = {
        'usage_counts': {},
        'tools_per_session': [],
        'total_actions': 0,
        'total_sessions': 0,
    }

    # Analyze actions
    if ACTIONS_DIR.exists():
        for action_file in ACTIONS_DIR.glob("**/*.json"):
            try:
                with open(action_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                tool = data.get('action_type', 'unknown')
                tool_stats['usage_counts'][tool] = tool_stats['usage_counts'].get(tool, 0) + 1
                tool_stats['total_actions'] += 1

            except (json.JSONDecodeError, IOError):
                continue

    # Analyze sessions for tool diversity
    if SESSIONS_DIR.exists():
        for session_file in SESSIONS_DIR.glob("*.json"):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                tool_stats['total_sessions'] += 1

            except (json.JSONDecodeError, IOError):
                continue

    tool_stats['generated_at'] = datetime.now().isoformat()
    return tool_stats


def generate_shared_patterns():
    """Generate all shared pattern files.

    Creates aggregated, anonymized pattern files in .git-ml/shared/
    that are safe to commit and share.
    """
    SHARED_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("GENERATING SHARED PATTERNS")
    print("=" * 60)

    # File correlations
    print("\n📁 Generating file correlations...")
    correlations = generate_file_correlations()
    corr_path = SHARED_DIR / "file_correlations.json"
    with open(corr_path, 'w', encoding='utf-8') as f:
        json.dump(correlations, f, indent=2)
    print(f"   Saved {len(correlations['correlations'])} file pairs to {corr_path}")

    # Commit patterns
    print("\n📝 Generating commit patterns...")
    patterns = generate_commit_patterns()
    patterns_path = SHARED_DIR / "commit_patterns.json"
    with open(patterns_path, 'w', encoding='utf-8') as f:
        json.dump(patterns, f, indent=2)
    print(f"   Analyzed {patterns['total_commits']} commits, saved to {patterns_path}")

    # Tool effectiveness
    print("\n🔧 Generating tool effectiveness...")
    tools = generate_tool_effectiveness()
    tools_path = SHARED_DIR / "tool_effectiveness.json"
    with open(tools_path, 'w', encoding='utf-8') as f:
        json.dump(tools, f, indent=2)
    print(f"   Analyzed {tools['total_actions']} actions, saved to {tools_path}")

    # Summary file
    summary = {
        'generated_at': datetime.now().isoformat(),
        'data_counts': count_data(),
        'files': [
            'file_correlations.json',
            'commit_patterns.json',
            'tool_effectiveness.json',
        ],
        'note': 'These patterns are aggregated and anonymized. Safe to commit.',
    }
    summary_path = SHARED_DIR / "summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Shared patterns generated in {SHARED_DIR}/")
    print("   These files are safe to commit and share.")
    print("=" * 60 + "\n")


# ============================================================================
# GITHUB DATA COLLECTION (PR/Issue context)
# ============================================================================

def fetch_pr_data(pr_number: int) -> Optional[Dict]:
    """Fetch PR data from GitHub using gh CLI.

    Args:
        pr_number: PR number to fetch.

    Returns:
        Dict with PR data or None if not found/gh not available.
    """
    try:
        # Use gh CLI to fetch PR data
        result = subprocess.run(
            ["gh", "pr", "view", str(pr_number), "--json",
             "number,title,body,state,author,createdAt,mergedAt,closedAt,"
             "headRefName,baseRefName,commits,files,comments,reviews"],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            return None

        pr_data = json.loads(result.stdout)

        # Add metadata
        pr_data['fetched_at'] = datetime.now().isoformat()
        pr_data['type'] = 'pull_request'

        # Redact sensitive content from body and comments
        if pr_data.get('body'):
            pr_data['body'] = redact_sensitive_data(pr_data['body'])
        if pr_data.get('comments'):
            for comment in pr_data['comments']:
                if comment.get('body'):
                    comment['body'] = redact_sensitive_data(comment['body'])
        if pr_data.get('reviews'):
            for review in pr_data['reviews']:
                if review.get('body'):
                    review['body'] = redact_sensitive_data(review['body'])

        return pr_data

    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        return None


def fetch_issue_data(issue_number: int) -> Optional[Dict]:
    """Fetch issue data from GitHub using gh CLI.

    Args:
        issue_number: Issue number to fetch.

    Returns:
        Dict with issue data or None if not found/gh not available.
    """
    try:
        result = subprocess.run(
            ["gh", "issue", "view", str(issue_number), "--json",
             "number,title,body,state,author,createdAt,closedAt,labels,comments"],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            return None

        issue_data = json.loads(result.stdout)

        # Add metadata
        issue_data['fetched_at'] = datetime.now().isoformat()
        issue_data['type'] = 'issue'

        # Redact sensitive content
        if issue_data.get('body'):
            issue_data['body'] = redact_sensitive_data(issue_data['body'])
        if issue_data.get('comments'):
            for comment in issue_data['comments']:
                if comment.get('body'):
                    comment['body'] = redact_sensitive_data(comment['body'])

        return issue_data

    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        return None


def save_github_data(data: Dict, data_type: str) -> Path:
    """Save GitHub data (PR or issue) to file.

    Args:
        data: PR or issue data dict.
        data_type: 'pr' or 'issue'.

    Returns:
        Path to saved file.
    """
    GITHUB_DIR.mkdir(parents=True, exist_ok=True)

    number = data.get('number', 'unknown')
    filename = f"{data_type}_{number}.json"
    filepath = GITHUB_DIR / filename

    atomic_write_json(filepath, data)
    return filepath


def collect_recent_prs(limit: int = 20) -> List[Dict]:
    """Collect recent PRs from the repository.

    Args:
        limit: Maximum number of PRs to collect.

    Returns:
        List of collected PR data dicts.
    """
    try:
        # Get list of recent PRs
        result = subprocess.run(
            ["gh", "pr", "list", "--limit", str(limit), "--state", "all",
             "--json", "number"],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            return []

        pr_list = json.loads(result.stdout)
        collected = []

        for pr in pr_list:
            pr_number = pr['number']
            # Check if already collected
            existing = GITHUB_DIR / f"pr_{pr_number}.json"
            if existing.exists():
                continue

            pr_data = fetch_pr_data(pr_number)
            if pr_data:
                save_github_data(pr_data, 'pr')
                collected.append(pr_data)

        return collected

    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        return []


def collect_recent_issues(limit: int = 20) -> List[Dict]:
    """Collect recent issues from the repository.

    Args:
        limit: Maximum number of issues to collect.

    Returns:
        List of collected issue data dicts.
    """
    try:
        result = subprocess.run(
            ["gh", "issue", "list", "--limit", str(limit), "--state", "all",
             "--json", "number"],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            return []

        issue_list = json.loads(result.stdout)
        collected = []

        for issue in issue_list:
            issue_number = issue['number']
            existing = GITHUB_DIR / f"issue_{issue_number}.json"
            if existing.exists():
                continue

            issue_data = fetch_issue_data(issue_number)
            if issue_data:
                save_github_data(issue_data, 'issue')
                collected.append(issue_data)

        return collected

    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        return []


def link_commit_to_pr(commit_hash: str) -> Optional[int]:
    """Find the PR that merged a commit, if any.

    Args:
        commit_hash: The commit hash to look up.

    Returns:
        PR number if found, None otherwise.
    """
    try:
        # Use gh to find associated PR
        result = subprocess.run(
            ["gh", "pr", "list", "--search", commit_hash, "--state", "merged",
             "--json", "number", "--limit", "1"],
            capture_output=True,
            text=True,
            timeout=15
        )

        if result.returncode == 0:
            prs = json.loads(result.stdout)
            if prs:
                return prs[0]['number']

    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        pass

    return None


def get_github_stats() -> Dict[str, int]:
    """Get counts of collected GitHub data.

    Returns:
        Dict with 'prs' and 'issues' counts.
    """
    if not GITHUB_DIR.exists():
        return {'prs': 0, 'issues': 0}

    prs = len(list(GITHUB_DIR.glob("pr_*.json")))
    issues = len(list(GITHUB_DIR.glob("issue_*.json")))

    return {'prs': prs, 'issues': issues}


# ============================================================================
# CI AUTO-CAPTURE (for GitHub Actions integration)
# ============================================================================

def ci_autocapture() -> bool:
    """Auto-capture CI results from GitHub Actions environment.

    Reads environment variables set by GitHub Actions and records
    CI results for the current commit.

    Environment variables used:
        GITHUB_SHA: Commit hash
        CI_RESULT: pass/fail/error (set by workflow)
        CI_TESTS_PASSED: Number of tests passed
        CI_TESTS_FAILED: Number of tests failed
        CI_COVERAGE: Coverage percentage
        CI_DURATION: Duration in seconds

    Returns:
        True if CI result was recorded, False otherwise.
    """
    commit_hash = os.getenv('GITHUB_SHA')
    if not commit_hash:
        # Try to get from git if not in Actions
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                commit_hash = result.stdout.strip()
        except Exception:
            pass

    if not commit_hash:
        print("❌ No commit hash available")
        return False

    # Get CI result from environment
    result = os.getenv('CI_RESULT', 'unknown')
    if result not in ('pass', 'fail', 'error', 'pending', 'unknown'):
        result = 'unknown'

    # Collect details from environment
    details = {}

    tests_passed = os.getenv('CI_TESTS_PASSED')
    if tests_passed:
        try:
            details['tests_passed'] = int(tests_passed)
        except ValueError:
            pass

    tests_failed = os.getenv('CI_TESTS_FAILED')
    if tests_failed:
        try:
            details['tests_failed'] = int(tests_failed)
        except ValueError:
            pass

    coverage = os.getenv('CI_COVERAGE')
    if coverage:
        try:
            details['coverage'] = float(coverage)
        except ValueError:
            pass

    duration = os.getenv('CI_DURATION')
    if duration:
        try:
            details['duration_seconds'] = float(duration)
        except ValueError:
            pass

    # Get job name and workflow
    details['workflow'] = os.getenv('GITHUB_WORKFLOW', 'unknown')
    details['job'] = os.getenv('GITHUB_JOB', 'unknown')
    details['run_id'] = os.getenv('GITHUB_RUN_ID', '')
    details['run_number'] = os.getenv('GITHUB_RUN_NUMBER', '')

    # Update the commit
    success = update_commit_ci_result(commit_hash, result, details if details else None)

    if success:
        print(f"✅ CI result recorded for {commit_hash[:8]}: {result}")
        if details.get('coverage'):
            print(f"   Coverage: {details['coverage']}%")
        if details.get('tests_passed') is not None:
            print(f"   Tests: {details.get('tests_passed', 0)} passed, {details.get('tests_failed', 0)} failed")
    else:
        # Commit might not be in our data yet - that's OK
        print(f"⚠️  Commit {commit_hash[:8]} not found in ML data (may not be collected yet)")

    return success


# ============================================================================
# GIT HOOKS
# ============================================================================

ML_HOOK_MARKER = "# ML-DATA-COLLECTOR-HOOK"

POST_COMMIT_SNIPPET = '''
# ML-DATA-COLLECTOR-HOOK
# ML Data Collection - Post-Commit Hook
# Automatically collects enriched commit data for model training

# Skip ML-only commits to prevent infinite loop
COMMIT_MSG=$(git log -1 --format=%s HEAD 2>/dev/null)
if [[ "$COMMIT_MSG" == "data: ML tracking data"* ]] || [[ "$COMMIT_MSG" == "data: ML"* ]]; then
    exit 0
fi

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

PREPARE_COMMIT_MSG_SNIPPET = '''
# ML-DATA-COLLECTOR-HOOK
# ML File Prediction Suggestion Hook
# Suggests potentially missing files based on commit message
bash scripts/ml-precommit-suggest.sh "$@"
# END-ML-DATA-COLLECTOR-HOOK
'''


def install_hooks():
    """Install git hooks for data collection, merging with existing hooks."""
    hooks_dir = Path(".git/hooks")

    for hook_name, snippet in [
        ("post-commit", POST_COMMIT_SNIPPET),
        ("pre-push", PRE_PUSH_SNIPPET),
        ("prepare-commit-msg", PREPARE_COMMIT_MSG_SNIPPET)
    ]:
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

    # Allow stats/estimate/validate/export/feedback/quality-report/orchestration even when collection is disabled
    read_only_commands = {"stats", "estimate", "validate", "session", "export", "feedback", "quality-report", "orchestration"}

    # Check if collection is disabled (via ML_COLLECTION_ENABLED=0)
    if not ML_COLLECTION_ENABLED and command not in read_only_commands:
        # Silently exit for collection commands when disabled
        return

    if command == "commit":
        # Collect data for current or specified commit
        commit_hash = sys.argv[2] if len(sys.argv) > 2 else None
        context = collect_commit_data(commit_hash)
        save_commit_data(context)
        # Also save lightweight version (trackable in git)
        lite_path = save_commit_lite(context)
        print(f"Saved lightweight commit to {lite_path}")

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

        lite_count = 0
        for i, h in enumerate(hashes):
            try:
                context = collect_commit_data(h)
                # Disable session linking for historical backfill
                save_commit_data(context, link_session=False)
                # Also save lightweight version
                save_commit_lite(context)
                lite_count += 1
                if (i + 1) % 10 == 0:
                    print(f"  Progress: {i + 1}/{len(hashes)}")
            except Exception as e:
                print(f"  Error on {h[:8]}: {e}")

        print(f"Backfill complete: {len(hashes)} commits ({lite_count} lightweight)")

    elif command == "backfill-lite":
        # Backfill ONLY lightweight commit data (small, trackable in git)
        # Faster than full backfill, suitable for ephemeral environments
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("-n", "--num", type=int, default=100,
                            help="Number of commits to backfill")
        parser.add_argument("--all", action="store_true",
                            help="Backfill all commits in history")
        args = parser.parse_args(sys.argv[2:])

        if args.all:
            hashes = run_git(["log", "--format=%H"]).split("\n")
        else:
            hashes = run_git(["log", f"-{args.num}", "--format=%H"]).split("\n")
        hashes = [h for h in hashes if h]

        # Check existing in JSONL file
        existing = set()
        if COMMITS_LITE_FILE.exists():
            with open(COMMITS_LITE_FILE, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            existing.add(data.get("hash", "")[:12])
                        except json.JSONDecodeError:
                            continue
        # Also check legacy directory
        if COMMITS_LITE_DIR.exists():
            for f in COMMITS_LITE_DIR.glob("*.json"):
                existing.add(f.stem.split("_")[0])  # Extract hash prefix

        new_hashes = [h for h in hashes if h[:12] not in existing]
        print(f"Found {len(hashes)} commits, {len(new_hashes)} new (skipping {len(hashes) - len(new_hashes)} existing)")

        if not new_hashes:
            print("All commits already have lightweight data.")
        else:
            TRACKED_DIR.mkdir(parents=True, exist_ok=True)
            for i, h in enumerate(new_hashes):
                try:
                    context = collect_commit_data(h)
                    save_commit_lite(context)
                    if (i + 1) % 50 == 0:
                        print(f"  Progress: {i + 1}/{len(new_hashes)}")
                except Exception as e:
                    print(f"  Error on {h[:8]}: {e}")

            print(f"Lightweight backfill complete: {len(new_hashes)} commits")

    elif command == "migrate":
        # Migrate legacy commits-lite/*.json files to tracked/commits.jsonl
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--dry-run", action="store_true",
                            help="Show what would be migrated without doing it")
        parser.add_argument("--delete-old", action="store_true",
                            help="Delete old files after migration")
        args = parser.parse_args(sys.argv[2:])

        if not COMMITS_LITE_DIR.exists():
            print("No legacy commits-lite directory found.")
            sys.exit(0)

        legacy_files = list(COMMITS_LITE_DIR.glob("*.json"))
        if not legacy_files:
            print("No legacy files to migrate.")
            sys.exit(0)

        print(f"Found {len(legacy_files)} legacy files to migrate")

        # Get existing hashes in JSONL
        existing_hashes = set()
        if COMMITS_LITE_FILE.exists():
            with open(COMMITS_LITE_FILE, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            existing_hashes.add(data.get("hash", ""))
                        except json.JSONDecodeError:
                            continue

        migrated = 0
        skipped = 0
        for f in legacy_files:
            try:
                with open(f, 'r') as fp:
                    data = json.load(fp)
                commit_hash = data.get("hash", "")
                if commit_hash in existing_hashes:
                    skipped += 1
                    continue
                if args.dry_run:
                    print(f"  Would migrate: {f.name}")
                    migrated += 1
                else:
                    TRACKED_DIR.mkdir(parents=True, exist_ok=True)
                    with open(COMMITS_LITE_FILE, 'a') as out:
                        out.write(json.dumps(data, separators=(',', ':')) + '\n')
                    existing_hashes.add(commit_hash)
                    migrated += 1
            except Exception as e:
                print(f"  Error migrating {f.name}: {e}")

        if args.dry_run:
            print(f"\nDry run: would migrate {migrated}, skip {skipped} (already in JSONL)")
        else:
            print(f"\nMigrated {migrated} commits to {COMMITS_LITE_FILE}")
            print(f"Skipped {skipped} (already in JSONL)")

            if args.delete_old and migrated > 0:
                print(f"\nDeleting {len(legacy_files)} legacy files...")
                for f in legacy_files:
                    f.unlink()
                # Try to remove the directory if empty
                try:
                    COMMITS_LITE_DIR.rmdir()
                    print(f"Removed empty directory: {COMMITS_LITE_DIR}")
                except OSError:
                    print(f"Note: {COMMITS_LITE_DIR} not empty, keeping it")

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

    elif command == "quality-report":
        print_quality_report()

    elif command == "install-hooks":
        install_hooks()

    elif command == "validate":
        # Validate existing data against schemas
        import argparse
        from ml_collector.config import (
            COMMITS_DIR, CHATS_DIR, ACTIONS_DIR,
            COMMIT_SCHEMA, CHAT_SCHEMA, ACTION_SCHEMA,
            validate_schema
        )
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

        # Archive the session after processing transcript (fixes session capture gap)
        if not args.dry_run and result.get('saved', 0) > 0:
            try:
                archived_session = end_session(
                    summary=f"Transcript: {result.get('saved', 0)} exchanges, "
                           f"{len(result.get('tools_used', []))} tools"
                )
                if archived_session and args.verbose:
                    print(f"📁 Archived session: {archived_session['id']}")
            except Exception as e:
                logger.warning(f"Failed to archive session: {e}")

    elif command == "export":
        # Export data for training
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--format", required=True,
                            choices=["jsonl", "csv", "huggingface"],
                            help="Output format")
        parser.add_argument("--output", required=True,
                            help="Output file path")
        parser.add_argument("--truncate-input", type=int,
                            default=CSV_DEFAULT_TRUNCATE_QUERY,
                            help=f"Max length for input/query fields in CSV (default: {CSV_DEFAULT_TRUNCATE_QUERY}, 0=no truncation)")
        parser.add_argument("--truncate-output", type=int,
                            default=CSV_DEFAULT_TRUNCATE_RESPONSE,
                            help=f"Max length for output/response fields in CSV (default: {CSV_DEFAULT_TRUNCATE_RESPONSE}, 0=no truncation)")
        parser.add_argument("--truncate-files", type=int,
                            default=CSV_DEFAULT_TRUNCATE_QUERY,
                            help=f"Max length for files field in CSV (default: {CSV_DEFAULT_TRUNCATE_QUERY}, 0=no truncation)")
        parser.add_argument("--no-truncate", action="store_true",
                            help="Disable all truncation (overrides other truncate options)")
        args = parser.parse_args(sys.argv[2:])

        output_path = Path(args.output)

        # Validate output path
        if output_path.exists():
            response = input(f"⚠️  {output_path} already exists. Overwrite? [y/N] ")
            if response.lower() != 'y':
                print("Export cancelled.")
                sys.exit(0)

        # Check if we have data to export
        counts = count_data()
        if counts['commits'] == 0 and counts['chats'] == 0:
            print("⚠️  No data to export. Collect some commits and chats first.")
            sys.exit(1)

        print(f"\n{'='*60}")
        print("EXPORTING ML DATA")
        print(f"{'='*60}")
        print(f"Format: {args.format}")
        print(f"Output: {output_path}")
        print(f"Data: {counts['commits']} commits, {counts['chats']} chats")

        # Handle --no-truncate flag (overrides individual truncate options)
        if args.no_truncate:
            truncate_input = 0
            truncate_output = 0
            truncate_files = 0
            print("Truncation: Disabled")
        else:
            truncate_input = args.truncate_input
            truncate_output = args.truncate_output
            truncate_files = args.truncate_files
            if args.format == "csv":
                print(f"Truncation: input={truncate_input}, output={truncate_output}, files={truncate_files}")

        print()

        try:
            stats = export_data(args.format, output_path,
                              truncate_input=truncate_input,
                              truncate_output=truncate_output,
                              truncate_files=truncate_files)
            print(f"✅ Export complete!")
            print(f"   Records: {stats['records']}")
            print(f"   Commits: {stats['commits']}")
            print(f"   Chats: {stats['chats']}")
            print(f"   File: {stats['output_path']}")
            print(f"{'='*60}\n")
        except Exception as e:
            print(f"✗ Export failed: {e}")
            sys.exit(1)

    elif command == "feedback":
        # Add or view feedback for chat entries
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--chat-id",
                            help="Chat ID to add feedback to")
        parser.add_argument("--rating", choices=["good", "bad", "neutral"],
                            help="Feedback rating")
        parser.add_argument("--comment",
                            help="Optional feedback comment")
        parser.add_argument("--force", action="store_true",
                            help="Overwrite existing feedback")
        parser.add_argument("--list", action="store_true",
                            help="List recent chats (showing feedback status)")
        parser.add_argument("--limit", type=int, default=10,
                            help="Number of chats to show (default: 10)")
        args = parser.parse_args(sys.argv[2:])

        if args.list:
            # List recent chats and their feedback status
            chats = list_chats_needing_feedback(limit=args.limit)

            if not chats:
                print("No chat entries found.")
                return

            print(f"\n{'='*60}")
            print(f"RECENT CHATS (last {args.limit})")
            print(f"{'='*60}\n")

            for chat in chats:
                feedback_status = "✓" if chat['has_feedback'] else "○"
                rating_display = f" [{chat['feedback_rating']}]" if chat['feedback_rating'] else ""

                print(f"{feedback_status} {chat['id']}")
                print(f"   Time: {chat['timestamp']}")
                print(f"   Query: {chat['query']}")
                if chat['has_feedback']:
                    print(f"   Feedback: {chat['feedback_rating']}")
                print()

            # Show summary
            with_feedback = sum(1 for c in chats if c['has_feedback'])
            without_feedback = len(chats) - with_feedback
            print(f"{'='*60}")
            print(f"Summary: {with_feedback} with feedback, {without_feedback} without")
            print(f"{'='*60}\n")

        else:
            # Add feedback to a specific chat
            if not args.chat_id:
                print("Error: --chat-id is required when not using --list")
                sys.exit(1)

            if not args.rating:
                print("Error: --rating is required when adding feedback")
                sys.exit(1)

            try:
                success = add_chat_feedback(
                    chat_id=args.chat_id,
                    rating=args.rating,
                    comment=args.comment,
                    force=args.force
                )

                if success:
                    print(f"✓ Added feedback to chat {args.chat_id}")
                    print(f"   Rating: {args.rating}")
                    if args.comment:
                        print(f"   Comment: {args.comment}")
                else:
                    # Check if chat exists or already has feedback
                    chat_file = find_chat_file(args.chat_id)
                    if not chat_file:
                        print(f"✗ Chat not found: {args.chat_id}")
                        sys.exit(1)
                    else:
                        print(f"✗ Chat {args.chat_id} already has feedback.")
                        print("   Use --force to overwrite.")
                        sys.exit(1)

            except ValueError as e:
                print(f"✗ {e}")
                sys.exit(1)

    elif command == "cleanup":
        # Remove data older than retention period
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--days", type=int, default=DEFAULT_RETENTION_DAYS,
                            help=f"Retention period in days (default: {DEFAULT_RETENTION_DAYS})")
        parser.add_argument("--dry-run", action="store_true",
                            help="Show what would be deleted without deleting")
        parser.add_argument("--verbose", "-v", action="store_true",
                            help="Show each file being removed")
        args = parser.parse_args(sys.argv[2:])

        print(f"\n{'='*60}")
        print("DATA CLEANUP")
        print(f"{'='*60}")
        print(f"Retention period: {args.days} days")
        if args.dry_run:
            print("Mode: DRY RUN (no files will be deleted)")
        print()

        results = cleanup_old_data(
            retention_days=args.days,
            dry_run=args.dry_run,
            verbose=args.verbose
        )

        total_removed = (results['commits_removed'] + results['chats_removed'] +
                        results['actions_removed'] + results['sessions_removed'])

        if total_removed == 0:
            print("✓ No data older than cutoff date found.")
        else:
            action = "Would remove" if args.dry_run else "Removed"
            print(f"\n{action}:")
            print(f"   Commits:  {results['commits_removed']}")
            print(f"   Chats:    {results['chats_removed']}")
            print(f"   Actions:  {results['actions_removed']}")
            print(f"   Sessions: {results['sessions_removed']}")

            if results['bytes_freed'] > 1024 * 1024:
                print(f"   Space:    {results['bytes_freed'] / 1024 / 1024:.2f} MB")
            elif results['bytes_freed'] > 1024:
                print(f"   Space:    {results['bytes_freed'] / 1024:.2f} KB")
            else:
                print(f"   Space:    {results['bytes_freed']} bytes")

        print(f"{'='*60}\n")

    elif command == "contribute":
        # Manage contribution consent and preview data
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("action", nargs="?", choices=["status", "enable", "disable", "preview"],
                            default="status", help="Contribution action")
        parser.add_argument("--name", help="Your name for attribution")
        parser.add_argument("--email", help="Contact email")
        parser.add_argument("--notes", help="Additional notes")
        parser.add_argument("--samples", type=int, default=5,
                            help="Number of samples to show in preview")
        parser.add_argument("--full-text", action="store_true",
                            help="Show full query/response text in preview")
        args = parser.parse_args(sys.argv[2:])

        if args.action == "status":
            consent = get_contribution_consent()
            print(f"\n{'='*60}")
            print("CONTRIBUTION STATUS")
            print(f"{'='*60}")

            if consent is None:
                print("\n📋 Status: NOT CONFIGURED")
                print("\n   You haven't decided whether to contribute data yet.")
                print("   Run 'contribute enable' to opt-in or 'contribute disable' to opt-out.")
            else:
                status = "✅ ENABLED" if consent.get('consented') else "❌ DISABLED"
                print(f"\n📋 Status: {status}")
                print(f"   Recorded: {consent.get('timestamp', 'unknown')[:10]}")
                if consent.get('contributor_name'):
                    print(f"   Name: {consent['contributor_name']}")
                if consent.get('contributor_email'):
                    print(f"   Email: {consent['contributor_email']}")
                if consent.get('notes'):
                    print(f"   Notes: {consent['notes']}")

            print(f"\n{'='*60}\n")

        elif args.action == "enable":
            print(f"\n{'='*60}")
            print("ENABLE DATA CONTRIBUTION")
            print(f"{'='*60}")
            print("""
By enabling contribution, you agree to share your collected data
(commits, chats, sessions) with the project maintainers for training
a project-specific micro-model.

WHAT GETS SHARED:
- Commit messages and file change patterns
- Query/response pairs (with sensitive data automatically redacted)
- Tool usage patterns and session metadata

WHAT DOESN'T GET SHARED:
- Raw file contents
- Actual code diffs (only patterns)
- Any data matching redaction patterns (API keys, passwords, etc.)

Your name and email (if provided) will be used for:
- Attribution in the trained model's credits
- Contact if we have questions about your contributions
""")

            # Get confirmation
            response = input("Do you consent to contributing your data? [y/N] ")
            if response.lower() != 'y':
                print("\nContribution not enabled. Your data stays local.")
                sys.exit(0)

            consent = set_contribution_consent(
                consented=True,
                contributor_name=args.name,
                contributor_email=args.email,
                notes=args.notes
            )

            print(f"\n✅ Contribution ENABLED")
            print(f"   Recorded at: {consent['timestamp'][:19]}")
            if args.name:
                print(f"   Name: {args.name}")
            print("\nYour data will be included in the next collection round.")
            print("Run 'contribute preview' to see what will be shared.")
            print(f"{'='*60}\n")

        elif args.action == "disable":
            consent = set_contribution_consent(
                consented=False,
                notes=args.notes or "User opted out"
            )

            print(f"\n{'='*60}")
            print("CONTRIBUTION DISABLED")
            print(f"{'='*60}")
            print("\n❌ Your data will NOT be shared with the project.")
            print("   Local collection continues (for your own use).")
            print("   You can re-enable anytime with 'contribute enable'.")
            print(f"{'='*60}\n")

        elif args.action == "preview":
            print(f"\n{'='*60}")
            print("CONTRIBUTION DATA PREVIEW")
            print(f"{'='*60}")
            print("\nThis shows what data would be shared if you contribute:\n")

            preview = preview_contribution_data(
                max_samples=args.samples,
                include_full_text=args.full_text
            )

            summary = preview['summary']
            print("📊 Summary:")
            print(f"   Total commits:  {summary['total_commits']}")
            print(f"   Total chats:    {summary['total_chats']}")
            print(f"   Total sessions: {summary['total_sessions']}")
            print(f"   Unique files:   {summary['unique_files']}")
            print(f"   Unique tools:   {summary['unique_tools']}")
            if summary['date_range']['earliest']:
                print(f"   Date range:     {summary['date_range']['earliest']} to {summary['date_range']['latest']}")

            redaction = preview['redaction_stats']
            if redaction['total_redactions'] > 0:
                print(f"\n🔒 Redaction Stats:")
                print(f"   Chats with redactions: {redaction['chats_with_redactions']}")
                print(f"   Total redactions made: {redaction['total_redactions']}")

            if preview['sample_commits']:
                print(f"\n📝 Sample Commits (latest {len(preview['sample_commits'])}):")
                for c in preview['sample_commits']:
                    print(f"   [{c['timestamp']}] {c['hash']} - {c['message']}")
                    print(f"            Files: {c['files_count']}")

            if preview['sample_chats']:
                print(f"\n💬 Sample Chats (latest {len(preview['sample_chats'])}):")
                for c in preview['sample_chats']:
                    print(f"   [{c['timestamp']}] {c['id']}")
                    if 'query' in c:
                        print(f"      Query: {c['query']}")
                    elif 'query_preview' in c:
                        print(f"      Query: {c['query_preview']}")
                    print(f"      Tools: {', '.join(c['tools_used']) or 'none'}")
                    if c['redactions_applied'] > 0:
                        print(f"      🔒 {c['redactions_applied']} sensitive item(s) redacted")

            print(f"\n{'='*60}\n")

    elif command == "generate-patterns":
        # Generate shared pattern files (safe to commit)
        generate_shared_patterns()

    elif command == "github":
        # Collect GitHub PR/Issue data
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("action", choices=["collect", "stats", "fetch-pr", "fetch-issue"],
                            help="GitHub data action")
        parser.add_argument("--limit", type=int, default=20,
                            help="Max items to collect (default: 20)")
        parser.add_argument("--number", type=int,
                            help="PR or issue number (for fetch-pr/fetch-issue)")
        args = parser.parse_args(sys.argv[2:])

        if args.action == "collect":
            print("\n" + "=" * 60)
            print("COLLECTING GITHUB DATA")
            print("=" * 60)

            # Collect PRs
            print(f"\n🔀 Collecting PRs (limit: {args.limit})...")
            prs = collect_recent_prs(limit=args.limit)
            print(f"   Collected {len(prs)} new PRs")

            # Collect Issues
            print(f"\n📋 Collecting Issues (limit: {args.limit})...")
            issues = collect_recent_issues(limit=args.limit)
            print(f"   Collected {len(issues)} new issues")

            # Show totals
            stats = get_github_stats()
            print(f"\n📊 Total GitHub data:")
            print(f"   PRs: {stats['prs']}")
            print(f"   Issues: {stats['issues']}")
            print("=" * 60 + "\n")

        elif args.action == "stats":
            stats = get_github_stats()
            print(f"\n📊 GitHub Data Statistics:")
            print(f"   PRs collected: {stats['prs']}")
            print(f"   Issues collected: {stats['issues']}")

        elif args.action == "fetch-pr":
            if not args.number:
                print("Error: --number is required for fetch-pr")
                sys.exit(1)
            pr_data = fetch_pr_data(args.number)
            if pr_data:
                filepath = save_github_data(pr_data, 'pr')
                print(f"✓ Fetched PR #{args.number}: {pr_data.get('title', 'untitled')}")
                print(f"  Saved to: {filepath}")
            else:
                print(f"✗ Could not fetch PR #{args.number} (not found or gh not available)")
                sys.exit(1)

        elif args.action == "fetch-issue":
            if not args.number:
                print("Error: --number is required for fetch-issue")
                sys.exit(1)
            issue_data = fetch_issue_data(args.number)
            if issue_data:
                filepath = save_github_data(issue_data, 'issue')
                print(f"✓ Fetched Issue #{args.number}: {issue_data.get('title', 'untitled')}")
                print(f"  Saved to: {filepath}")
            else:
                print(f"✗ Could not fetch Issue #{args.number} (not found or gh not available)")
                sys.exit(1)

    elif command == "ci-autocapture":
        # Auto-capture CI results from GitHub Actions environment
        success = ci_autocapture()
        sys.exit(0 if success else 1)

    elif command == "redact-test":
        # Test redaction patterns on sample text
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--text", "-t", help="Text to test redaction on")
        parser.add_argument("--file", "-f", help="File containing text to test")
        args = parser.parse_args(sys.argv[2:])

        if args.file:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
        elif args.text:
            text = args.text
        else:
            print("Provide --text or --file to test redaction patterns")
            sys.exit(1)

        print(f"\n{'='*60}")
        print("REDACTION TEST")
        print(f"{'='*60}")
        print(f"\nOriginal ({len(text)} chars):")
        print(text[:500] + ('...' if len(text) > 500 else ''))

        redacted = redact_sensitive_data(text)
        redaction_count = count_redactions(text, redacted)

        print(f"\nRedacted ({redaction_count} patterns matched):")
        print(redacted[:500] + ('...' if len(redacted) > 500 else ''))
        print(f"{'='*60}\n")

    elif command == "orchestration":
        # Extract orchestration patterns from sub-agent transcripts
        import argparse
        from ml_collector.orchestration import (
            extract_orchestration_from_directory,
            extract_and_save,
            print_orchestration_summary,
            ORCHESTRATION_DIR
        )

        parser = argparse.ArgumentParser(description="Extract director orchestration data")
        parser.add_argument("action", choices=["extract", "summary", "list"],
                           help="Action to perform")
        parser.add_argument("--project-dir", "-d", type=Path,
                           help="Claude project transcript directory")
        parser.add_argument("--session-id", "-s",
                           help="Filter by parent session ID")
        parser.add_argument("--save", action="store_true",
                           help="Save extracted data to .git-ml/orchestration/")
        args = parser.parse_args(sys.argv[2:])

        # Default project directory
        if args.project_dir is None:
            # Try to find the Claude project directory for this repo
            # Claude uses path with leading dash: /home/user/foo -> -home-user-foo
            cwd_safe = os.getcwd().replace('/', '-').replace('\\', '-')
            project_dir = Path.home() / '.claude' / 'projects' / cwd_safe
            if not project_dir.exists():
                print(f"Could not find Claude project directory at {project_dir}")
                print("Use --project-dir to specify the transcript location")
                sys.exit(1)
        else:
            project_dir = args.project_dir

        if args.action == "extract":
            if args.save:
                extraction, saved_path = extract_and_save(
                    project_dir,
                    parent_session_id=args.session_id
                )
                if saved_path:
                    print(f"Saved orchestration data to: {saved_path}")
                print_orchestration_summary(extraction)
            else:
                extraction = extract_orchestration_from_directory(
                    project_dir,
                    parent_session_id=args.session_id
                )
                print_orchestration_summary(extraction)

        elif args.action == "summary":
            extraction = extract_orchestration_from_directory(
                project_dir,
                parent_session_id=args.session_id
            )
            print_orchestration_summary(extraction)

        elif args.action == "list":
            # List saved orchestration files
            if not ORCHESTRATION_DIR.exists():
                print("No orchestration data saved yet.")
                print(f"Directory: {ORCHESTRATION_DIR}")
            else:
                files = sorted(ORCHESTRATION_DIR.glob("*.json"))
                if not files:
                    print("No orchestration files found.")
                else:
                    print(f"Saved orchestration extractions ({len(files)} files):")
                    for f in files:
                        print(f"  {f.name}")

    elif command == "chunked":
        # Chunked storage operations for git-friendly large file storage
        import argparse
        from ml_collector.chunked_storage import (
            migrate_to_chunked, compact_chunks, get_chunked_stats,
            reconstruct_all, CHUNKED_DIR
        )
        from ml_collector.config import CHATS_DIR, COMMITS_DIR

        parser = argparse.ArgumentParser(
            description="Chunked storage for git-friendly large file storage"
        )
        parser.add_argument("action", choices=["migrate", "compact", "stats", "reconstruct"],
                           help="Action to perform")
        parser.add_argument("--type", "-t", choices=["chat", "commit", "all"],
                           default="all", help="Record type to process")
        parser.add_argument("--keep-days", "-k", type=int, default=30,
                           help="Days to keep separate before compacting (default: 30)")
        parser.add_argument("--output", "-o", type=Path,
                           help="Output file for reconstruction")
        parser.add_argument("--session-id", "-s", default="migration",
                           help="Session ID for migration")
        args = parser.parse_args(sys.argv[2:])

        if args.action == "migrate":
            print("Migrating existing data to chunked storage...")
            total = 0

            if args.type in ("chat", "all"):
                count = migrate_to_chunked(CHATS_DIR, "chat", args.session_id)
                print(f"  Chats migrated: {count}")
                total += count

            if args.type in ("commit", "all"):
                count = migrate_to_chunked(COMMITS_DIR, "commit", args.session_id)
                print(f"  Commits migrated: {count}")
                total += count

            print(f"\nTotal records migrated: {total}")
            print(f"Chunked storage: {CHUNKED_DIR}")

        elif args.action == "compact":
            print(f"Compacting chunks older than {args.keep_days} days...")
            result = compact_chunks(keep_days=args.keep_days)
            print(f"  Files before: {result['files_before']}")
            print(f"  Files after:  {result['files_after']}")
            if result['bytes_saved'] > 0:
                print(f"  Bytes saved:  {result['bytes_saved']:,}")

        elif args.action == "stats":
            stats = get_chunked_stats()
            print("\n" + "=" * 50)
            print("CHUNKED STORAGE STATISTICS")
            print("=" * 50)
            print(f"\nTotal files:   {stats['total_files']}")
            print(f"Total records: {stats['total_records']}")
            if stats['total_bytes'] > 1024:
                print(f"Total size:    {stats['total_bytes'] / 1024:.1f} KB")
            else:
                print(f"Total size:    {stats['total_bytes']} bytes")

            if stats['by_type']:
                print("\nBy type:")
                for record_type, type_stats in stats['by_type'].items():
                    print(f"  {record_type}:")
                    print(f"    Files:   {type_stats['files']}")
                    print(f"    Records: {type_stats['records']}")
                    if type_stats['bytes'] > 1024:
                        print(f"    Size:    {type_stats['bytes'] / 1024:.1f} KB")
                    else:
                        print(f"    Size:    {type_stats['bytes']} bytes")
            print("=" * 50)

        elif args.action == "reconstruct":
            record_type = None if args.type == "all" else args.type
            records = reconstruct_all(record_type)
            print(f"Reconstructed {len(records)} records")

            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    for record in records:
                        f.write(json.dumps(record) + '\n')
                print(f"Written to: {args.output}")
            else:
                print("Use --output to save reconstructed data")

    else:
        print(f"Unknown command: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()
