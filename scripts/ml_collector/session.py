"""
Session management module for ML Data Collector

Handles session tracking, commit-chat linking, and session handoff generation.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from .config import CURRENT_SESSION_FILE, CHATS_DIR, SESSIONS_DIR
from .persistence import file_lock, generate_session_id, atomic_write_json, ensure_dirs


logger = logging.getLogger(__name__)


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
        summary: Optional summary of what was accomplished in the session.

    Returns:
        Session data dict or None if no active session.
    """
    session = get_current_session()
    if not session:
        return None

    # Add end metadata
    session['ended_at'] = datetime.now().isoformat()
    if summary:
        session['summary'] = summary

    # Archive to sessions directory
    ensure_dirs()
    session_file = SESSIONS_DIR / f"{session['id']}.json"
    atomic_write_json(session_file, session)

    # Remove current session file
    if CURRENT_SESSION_FILE.exists():
        CURRENT_SESSION_FILE.unlink()

    return session


def generate_session_handoff() -> str:
    """Generate a handoff document for the current session.

    Returns:
        Markdown formatted handoff document.
    """
    session = get_current_session()
    if not session:
        return "No active session."

    # Build handoff document
    doc = []
    doc.append("# Session Handoff\n")
    doc.append(f"**Session ID:** {session['id']}\n")
    doc.append(f"**Started:** {session['started_at']}\n")

    # Session stats
    doc.append("\n## Session Activity\n")
    doc.append(f"- Chats logged: {len(session.get('chat_ids', []))}\n")
    doc.append(f"- Actions logged: {len(session.get('action_ids', []))}\n")

    # Chat summaries
    chat_ids = session.get('chat_ids', [])
    if chat_ids:
        doc.append("\n## Recent Queries\n")
        for chat_id in chat_ids[-10:]:  # Last 10 chats
            chat_file = find_chat_file(chat_id)
            if chat_file and chat_file.exists():
                try:
                    with open(chat_file, 'r', encoding='utf-8') as f:
                        chat_data = json.load(f)
                    query = chat_data.get('query', '')[:100]
                    doc.append(f"- {query}\n")
                except (json.JSONDecodeError, IOError):
                    continue

    # Recommendation
    doc.append("\n## Next Steps\n")
    doc.append("- Review chat history for context\n")
    doc.append("- Check for any open issues or TODOs\n")
    doc.append("- Sync with team on progress\n")

    return ''.join(doc)
