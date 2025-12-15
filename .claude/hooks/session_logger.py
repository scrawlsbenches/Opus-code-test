#!/usr/bin/env python3
"""
Claude Session Logger Hook

This script logs Claude Code chat sessions for ML training data collection.
It can be called manually or integrated with Claude Code hooks.

Usage:
    # Log a complete exchange
    python .claude/hooks/session_logger.py \
        --query "How do I fix the timeout bug?" \
        --response "Let me look at the code..." \
        --files-read cortical/processor.py \
        --files-modified cortical/processor.py \
        --tools Read,Edit,Bash

    # Start a session
    python .claude/hooks/session_logger.py --start-session

    # End a session with summary
    python .claude/hooks/session_logger.py --end-session --summary "Fixed timeout bugs"
"""

import json
import os
import sys
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

try:
    from ml_data_collector import (
        log_chat, log_action, ensure_dirs,
        SESSIONS_DIR, generate_session_id
    )
    ML_COLLECTOR_AVAILABLE = True
except ImportError:
    ML_COLLECTOR_AVAILABLE = False


# Session state file
SESSION_STATE_FILE = Path(".git-ml/current_session.json")


def get_current_session() -> Optional[dict]:
    """Get the current active session."""
    if SESSION_STATE_FILE.exists():
        with open(SESSION_STATE_FILE) as f:
            return json.load(f)
    return None


def start_session() -> str:
    """Start a new logging session."""
    ensure_dirs()

    session = {
        "id": generate_session_id(),
        "started_at": datetime.now().isoformat(),
        "exchanges": 0,
        "files_touched": [],
        "tools_used": [],
    }

    SESSION_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SESSION_STATE_FILE, "w") as f:
        json.dump(session, f, indent=2)

    print(f"Started session: {session['id']}")
    return session["id"]


def end_session(summary: Optional[str] = None):
    """End the current session and save summary."""
    session = get_current_session()
    if not session:
        print("No active session")
        return

    session["ended_at"] = datetime.now().isoformat()
    session["summary"] = summary

    # Save to sessions directory
    ensure_dirs()
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    filename = f"{session['started_at'][:10]}_{session['id']}.json"
    filepath = SESSIONS_DIR / filename

    with open(filepath, "w") as f:
        json.dump(session, f, indent=2)

    # Remove current session file
    SESSION_STATE_FILE.unlink()

    print(f"Ended session: {session['id']}")
    print(f"  Exchanges: {session['exchanges']}")
    print(f"  Files: {len(session['files_touched'])}")
    print(f"  Saved to: {filepath}")


def log_exchange(
    query: str,
    response: str,
    files_read: Optional[List[str]] = None,
    files_modified: Optional[List[str]] = None,
    tools: Optional[List[str]] = None,
    feedback: Optional[str] = None,
):
    """Log a query/response exchange."""
    if not ML_COLLECTOR_AVAILABLE:
        print("Warning: ml_data_collector not available")
        return

    # Get or create session
    session = get_current_session()
    if not session:
        session_id = start_session()
        session = get_current_session()
    else:
        session_id = session["id"]

    # Log the chat
    entry = log_chat(
        query=query,
        response=response,
        session_id=session_id,
        files_referenced=files_read or [],
        files_modified=files_modified or [],
        tools_used=tools or [],
        user_feedback=feedback,
    )

    # Update session state
    session["exchanges"] += 1
    session["files_touched"] = list(set(
        session["files_touched"] +
        (files_read or []) +
        (files_modified or [])
    ))
    session["tools_used"] = list(set(
        session["tools_used"] +
        (tools or [])
    ))

    with open(SESSION_STATE_FILE, "w") as f:
        json.dump(session, f, indent=2)

    print(f"Logged exchange: {entry.id}")


def log_tool_use(tool_name: str, target: str, success: bool = True):
    """Log a tool use action."""
    if not ML_COLLECTOR_AVAILABLE:
        return

    session = get_current_session()
    session_id = session["id"] if session else None

    log_action(
        action_type=f"tool:{tool_name}",
        target=target,
        session_id=session_id,
        success=success,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Claude Session Logger")
    parser.add_argument("--start-session", action="store_true",
                        help="Start a new session")
    parser.add_argument("--end-session", action="store_true",
                        help="End the current session")
    parser.add_argument("--summary", help="Session summary (with --end-session)")

    parser.add_argument("--query", help="User query text")
    parser.add_argument("--response", help="Assistant response text")
    parser.add_argument("--files-read", nargs="*", default=[],
                        help="Files that were read")
    parser.add_argument("--files-modified", nargs="*", default=[],
                        help="Files that were modified")
    parser.add_argument("--tools", help="Comma-separated list of tools used")
    parser.add_argument("--feedback", choices=["positive", "negative", "neutral"],
                        help="User feedback on the exchange")

    parser.add_argument("--log-tool", help="Log a tool use (format: tool:target)")

    args = parser.parse_args()

    if args.start_session:
        start_session()

    elif args.end_session:
        end_session(args.summary)

    elif args.query and args.response:
        tools = args.tools.split(",") if args.tools else []
        log_exchange(
            query=args.query,
            response=args.response,
            files_read=args.files_read,
            files_modified=args.files_modified,
            tools=tools,
            feedback=args.feedback,
        )

    elif args.log_tool:
        parts = args.log_tool.split(":", 1)
        if len(parts) == 2:
            log_tool_use(parts[0], parts[1])

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
