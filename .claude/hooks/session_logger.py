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
from pathlib import Path
from typing import List, Optional

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

try:
    from ml_data_collector import log_chat, log_action, ensure_dirs
    # Import session management from authoritative source
    from ml_collector.session import (
        get_current_session,
        start_session,
        end_session as ml_end_session,
        find_chat_file
    )
    from ml_collector.config import CHATS_DIR
    ML_COLLECTOR_AVAILABLE = True
except ImportError:
    ML_COLLECTOR_AVAILABLE = False


def end_session(summary: Optional[str] = None):
    """End the current session and save summary.

    CLI-compatible wrapper around ml_collector.session.end_session
    that provides the same output format as the legacy implementation.
    """
    if not ML_COLLECTOR_AVAILABLE:
        print("Warning: ml_data_collector not available")
        return

    session = get_current_session()
    if not session:
        print("No active session")
        return

    # Calculate stats from chat_ids before ending
    session_id = session['id']
    chat_ids = session.get('chat_ids', [])

    # Collect files from all chats in session
    all_files = set()
    for chat_id in chat_ids:
        chat_file = find_chat_file(chat_id)
        if chat_file and chat_file.exists():
            try:
                with open(chat_file, 'r', encoding='utf-8') as f:
                    chat_data = json.load(f)
                all_files.update(chat_data.get('files_referenced', []))
                all_files.update(chat_data.get('files_modified', []))
            except (json.JSONDecodeError, IOError):
                continue

    # End the session (saves to SESSIONS_DIR)
    ended_session = ml_end_session(summary)

    if ended_session:
        print(f"Ended session: {session_id}")
        print(f"  Exchanges: {len(chat_ids)}")
        print(f"  Files: {len(all_files)}")
        # The ml_end_session already saved to SESSIONS_DIR


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
    else:
        session_id = session["id"]

    # Log the chat (this automatically adds to session via add_chat_to_session)
    entry = log_chat(
        query=query,
        response=response,
        session_id=session_id,
        files_referenced=files_read or [],
        files_modified=files_modified or [],
        tools_used=tools or [],
        user_feedback=feedback,
    )

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
