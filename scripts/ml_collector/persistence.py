"""
Persistence module for ML Data Collector

Handles file I/O, atomic writes, and data storage operations.
"""

import json
import os
import sys
import tempfile
import fcntl
import hashlib
import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import asdict

# Add project root to path for cortical imports
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cortical.utils.id_generation import generate_short_id

from .config import (
    COMMITS_DIR, COMMITS_LITE_DIR, COMMITS_LITE_FILE, SESSIONS_LITE_FILE,
    SESSIONS_DIR, CHATS_DIR, ACTIONS_DIR, TRACKED_DIR, CALI_DIR,
    COMMIT_SCHEMA, CHAT_SCHEMA, ACTION_SCHEMA,
    validate_schema, redact_sensitive_data
)
from .core import SchemaValidationError
from .data_classes import CommitContext, ChatEntry, ActionEntry
from .chunked_storage import store_chunked_chat


logger = logging.getLogger(__name__)


# ============================================================================
# CALI STORAGE (high-performance, git-friendly)
# ============================================================================

# Environment variable to enable/disable CALI (default: enabled)
ML_USE_CALI = os.getenv("ML_USE_CALI", "1") == "1"

# Lazy-loaded CALI store instance and exception types
_cali_store = None
_cali_validation_error = None
_cali_serialization_error = None


def _get_cali_exceptions():
    """Get CALI exception types for proper error handling."""
    global _cali_validation_error, _cali_serialization_error
    if _cali_validation_error is None:
        try:
            from cortical.ml_storage import CALIValidationError, CALISerializationError
            _cali_validation_error = CALIValidationError
            _cali_serialization_error = CALISerializationError
        except ImportError:
            # Fallback to base Exception if not available
            _cali_validation_error = ValueError
            _cali_serialization_error = TypeError
    return _cali_validation_error, _cali_serialization_error


def get_cali_store():
    """Get or create the CALI store instance."""
    global _cali_store
    if _cali_store is None and ML_USE_CALI:
        try:
            from cortical.ml_storage import MLStore
            session_id = os.getenv("CLAUDE_SESSION_ID", generate_short_id())
            _cali_store = MLStore(CALI_DIR, session_id=session_id)
            logger.debug(f"CALI store initialized at {CALI_DIR}")
        except ImportError:
            logger.debug("CALI storage not available (cortical.ml_storage not found)")
        except Exception as e:
            logger.warning(f"Failed to initialize CALI store: {e}")
    return _cali_store


def cali_put(record_type: str, record_id: str, data: Dict[str, Any]) -> bool:
    """
    Write to CALI store if enabled. Returns True if written.

    Args:
        record_type: Type of record (e.g., 'commit', 'chat', 'session')
        record_id: Unique identifier for the record
        data: Dictionary data to store (must be JSON serializable)

    Returns:
        True if successfully written, False if CALI is disabled or unavailable

    Raises:
        ValueError: If record_type, record_id, or data is invalid
        TypeError: If data cannot be serialized to JSON
    """
    # Pre-validate inputs before calling CALI
    if not record_type or not isinstance(record_type, str):
        raise ValueError(f"record_type must be a non-empty string, got: {type(record_type)}")
    if not record_id or not isinstance(record_id, str):
        raise ValueError(f"record_id must be a non-empty string, got: {type(record_id)}")
    if data is None:
        raise ValueError("data cannot be None")
    if not isinstance(data, dict):
        raise TypeError(f"data must be a dictionary, got: {type(data)}")
    if len(data) == 0:
        raise ValueError("data cannot be an empty dictionary")

    store = get_cali_store()
    if store:
        CALIValidationError, CALISerializationError = _get_cali_exceptions()
        try:
            store.put(record_type, record_id, data)
            return True
        except CALIValidationError as e:
            # Re-raise validation errors - caller should fix their data
            raise ValueError(f"CALI validation failed: {e}") from e
        except CALISerializationError as e:
            # Re-raise serialization errors - caller should fix their data
            raise TypeError(f"CALI serialization failed: {e}") from e
        except Exception as e:
            # Log other errors but don't fail (graceful degradation)
            logger.warning(f"CALI write failed for {record_type}/{record_id}: {e}")
    return False


def cali_exists(record_type: str, record_id: str) -> bool:
    """
    Check if record exists in CALI (O(1) bloom filter check).

    Args:
        record_type: Type of record (e.g., 'commit', 'chat', 'session')
        record_id: Unique identifier for the record

    Returns:
        True if record exists, False if not or if CALI is disabled

    Raises:
        ValueError: If record_type or record_id is invalid
    """
    # Pre-validate inputs
    if not record_type or not isinstance(record_type, str):
        raise ValueError(f"record_type must be a non-empty string, got: {type(record_type)}")
    if not record_id or not isinstance(record_id, str):
        raise ValueError(f"record_id must be a non-empty string, got: {type(record_id)}")

    store = get_cali_store()
    if store:
        CALIValidationError, _ = _get_cali_exceptions()
        try:
            return store.exists(record_type, record_id)
        except CALIValidationError as e:
            # Re-raise validation errors
            raise ValueError(f"CALI validation failed: {e}") from e
        except Exception as e:
            logger.warning(f"CALI exists check failed: {e}")
    return False


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
# BASIC FILE OPERATIONS
# ============================================================================

def ensure_dirs():
    """Create data directories if they don't exist."""
    for dir_path in [COMMITS_DIR, COMMITS_LITE_DIR, SESSIONS_DIR, CHATS_DIR, ACTIONS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)


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


# ============================================================================
# ID GENERATION
# ============================================================================

def generate_chat_id() -> str:
    """Generate unique chat entry ID."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = hashlib.sha256(str(datetime.now().timestamp()).encode()).hexdigest()[:6]
    return f"chat-{timestamp}-{suffix}"


def generate_session_id() -> str:
    """Generate unique session ID."""
    return hashlib.sha256(str(datetime.now().timestamp()).encode()).hexdigest()[:8]


# ============================================================================
# COMMIT PERSISTENCE
# ============================================================================

def save_commit_data(context: CommitContext, validate: bool = True, link_session: bool = True):
    """Save commit context to disk atomically with validation.

    Args:
        context: The commit context to save.
        validate: Whether to validate against schema.
        link_session: Whether to link with current session chats.
    """
    # Import here to avoid circular dependency
    from .session import get_current_session, link_commit_to_session_chats

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

    # Use full hash + unique suffix to prevent collisions
    unique_id = generate_short_id()
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

    # CALI: O(1) existence check (fast path)
    if cali_exists('commit', context.hash):
        logger.debug(f"CALI: commit {context.hash[:8]} already exists")
        return COMMITS_LITE_FILE

    # Lock to prevent check-then-act race condition
    with file_lock(COMMITS_LITE_FILE):
        # Legacy: Check if this commit hash already exists in the file (idempotent)
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

        # Also write to CALI for O(1) lookups and git-friendly storage
        if cali_put('commit', context.hash, lite_data):
            logger.debug(f"CALI: stored commit {context.hash[:8]}")

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

    # Lock to prevent check-then-act race condition
    with file_lock(SESSIONS_LITE_FILE):
        # Legacy: Check if this session already exists (idempotent)
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

        # Append to JSONL file
        with open(SESSIONS_LITE_FILE, 'a') as f:
            f.write(json.dumps(session_summary, separators=(',', ':')) + '\n')

        # Also write to CALI for O(1) lookups and git-friendly storage
        if cali_put('session', session_id, session_summary):
            logger.debug(f"CALI: stored session {session_id}")

    return SESSIONS_LITE_FILE


# ============================================================================
# CHAT PERSISTENCE
# ============================================================================

def save_chat_entry(entry: ChatEntry, validate: bool = True):
    """Save a chat entry to disk atomically with validation.

    Saves to THREE locations for different purposes:
    1. .git-ml/chats/ - Full data, organized by date (gitignored - local only)
    2. .git-ml/tracked/chunked/ - Compressed, git-tracked (for sharing via git)
    3. CALI storage - O(1) lookups (local cache)
    """
    ensure_dirs()

    data = asdict(entry)

    # Validate before writing
    if validate:
        errors = validate_schema(data, CHAT_SCHEMA, "chat")
        if errors:
            raise SchemaValidationError(f"Chat validation failed: {errors}")

    # Organize by date (local storage - gitignored)
    date_dir = CHATS_DIR / entry.timestamp[:10]
    date_dir.mkdir(exist_ok=True)

    filename = f"{entry.id}.json"
    filepath = date_dir / filename

    atomic_write_json(filepath, data)
    print(f"Saved chat entry to {filepath}")

    # Also write to CALI for O(1) lookups and git-friendly storage
    if cali_put('chat', entry.id, data):
        logger.debug(f"CALI: stored chat {entry.id}")

    # Also write to chunked storage for git sharing
    # This enables backfilling chat history across branches
    try:
        chunk_file = store_chunked_chat(data, entry.session_id)
        logger.debug(f"Chunked storage: stored chat to {chunk_file}")
    except Exception as e:
        logger.warning(f"Failed to store chat in chunked storage: {e}")


def log_chat(
    query: str,
    response: str,
    session_id: Optional[str] = None,
    files_referenced: Optional[List[str]] = None,
    files_modified: Optional[List[str]] = None,
    tools_used: Optional[List[str]] = None,
    user_feedback: Optional[str] = None,
    skip_redaction: bool = False,
) -> ChatEntry:
    """Log a chat query/response pair.

    If no session_id is provided, uses the current session (creating one if needed).
    The chat is automatically registered with the session for commit linking.

    Sensitive data (API keys, passwords, tokens) is automatically redacted before storage
    unless skip_redaction=True.
    """
    # Import here to avoid circular dependency
    from .session import get_or_create_session, add_chat_to_session

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
        user_feedback=user_feedback,
        query_tokens=len(query.split()),  # Rough estimate
        response_tokens=len(response.split()),
    )

    save_chat_entry(entry)

    # Register with session for commit linking
    add_chat_to_session(entry.id)

    return entry


# ============================================================================
# ACTION PERSISTENCE
# ============================================================================

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
