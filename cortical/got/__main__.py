#!/usr/bin/env python3
"""
GoT CLI entry point.

Usage:
    python -m cortical.got task list
    python -m cortical.got sprint status
    python -m cortical.got --help

This replaces scripts/got_utils.py as the primary CLI interface.
"""

import argparse
import os
import signal
import sys
from pathlib import Path

# CLI module imports
from cortical.got.cli.doc import setup_doc_parser, handle_doc_command
from cortical.got.cli.task import setup_task_parser, handle_task_command
from cortical.got.cli.sprint import (
    setup_sprint_parser,
    setup_epic_parser,
    handle_sprint_command,
    handle_epic_command,
)
from cortical.got.cli.handoff import setup_handoff_parser, handle_handoff_command
from cortical.got.cli.decision import setup_decision_parser, handle_decision_command
from cortical.got.cli.query import setup_query_parser, handle_query_commands
from cortical.got.cli.backup import setup_backup_parser, handle_backup_command
from cortical.got.cli.orphan import setup_orphan_parser, handle_orphan_command
from cortical.got.cli.backlog import setup_backlog_parser, handle_backlog_command
from cortical.got.cli.analyze import setup_analyze_parser, handle_analyze_command
from cortical.got.cli.edge import setup_edge_parser, handle_edge_command
from cortical.got.cli.batch import setup_batch_parser, handle_batch_command
from cortical.got.cli.knowledge_transfer import (
    setup_knowledge_transfer_parser,
    handle_knowledge_transfer_command,
)
from cortical.got.cli.failure import setup_failure_parser, handle_failure_command

# Factory and adapter
from cortical.got.factory import GoTBackendFactory

# All valid commands for suggestion
VALID_COMMANDS = [
    "task", "sprint", "epic", "handoff", "decision", "doc", "query", "expr",
    "blocked", "active", "stats", "dashboard", "validate", "infer",
    "export", "backup", "sync", "orphan", "backlog", "analyze", "edge",
    "batch", "knowledge", "kt", "failure",
]


def suggest_command(invalid_cmd: str, valid_commands: list = VALID_COMMANDS) -> list:
    """Suggest similar commands when user types an invalid one."""
    import difflib
    matches = difflib.get_close_matches(
        invalid_cmd.lower(),
        valid_commands,
        n=3,
        cutoff=0.4
    )
    return matches


def print_command_suggestion(invalid_cmd: str) -> None:
    """Print helpful suggestions when an invalid command is used."""
    suggestions = suggest_command(invalid_cmd)

    print(f"\nError: '{invalid_cmd}' is not a valid command.", file=sys.stderr)

    if suggestions:
        print("\nDid you mean:", file=sys.stderr)
        for suggestion in suggestions:
            print(f"  - {suggestion}", file=sys.stderr)

    print(f"\nRun 'python -m cortical.got --help' for available commands.", file=sys.stderr)


def main():
    """
    Main CLI entry point.

    This is a thin dispatcher that delegates to the modular CLI handlers
    in cortical/got/cli/. See the individual modules for command implementations.
    """
    parser = argparse.ArgumentParser(
        description="Graph of Thought Project Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Global options
    parser.add_argument(
        "--backend",
        choices=["transactional", "event-sourced"],
        help="Override backend selection (default: auto-detect)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Set up CLI parsers from modular CLI modules
    setup_task_parser(subparsers)
    setup_sprint_parser(subparsers)
    setup_epic_parser(subparsers)
    setup_handoff_parser(subparsers)
    setup_decision_parser(subparsers)
    setup_doc_parser(subparsers)
    setup_query_parser(subparsers)
    setup_backup_parser(subparsers)
    setup_orphan_parser(subparsers)
    setup_backlog_parser(subparsers)
    setup_analyze_parser(subparsers)
    setup_edge_parser(subparsers)
    setup_batch_parser(subparsers)
    setup_knowledge_transfer_parser(subparsers)
    setup_failure_parser(subparsers)

    # Pre-check for invalid commands to provide better error messages
    if len(sys.argv) > 1:
        potential_cmd = sys.argv[1]
        if not potential_cmd.startswith('-') and potential_cmd not in VALID_COMMANDS:
            print_command_suggestion(potential_cmd)
            return 2

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Initialize manager using factory
    try:
        backend = getattr(args, 'backend', None)
        manager = GoTBackendFactory.create(backend=backend)
        if os.environ.get("GOT_DEBUG"):
            print(f"[DEBUG] Using transactional backend at {manager.got_dir}", file=sys.stderr)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Route commands to modular handlers
    if args.command == "task":
        return handle_task_command(args, manager)

    elif args.command == "sprint":
        return handle_sprint_command(args, manager)

    elif args.command == "epic":
        return handle_epic_command(args, manager)

    elif args.command == "handoff":
        return handle_handoff_command(args, manager)

    elif args.command == "decision":
        return handle_decision_command(args, manager)

    elif args.command == "doc":
        return handle_doc_command(args, manager)

    elif args.command == "backup":
        return handle_backup_command(args, manager)

    elif args.command == "orphan":
        return handle_orphan_command(args, manager)

    elif args.command == "backlog":
        return handle_backlog_command(args, manager)

    elif args.command == "analyze":
        return handle_analyze_command(args, manager)

    elif args.command == "edge":
        return handle_edge_command(args, manager)

    elif args.command == "batch":
        return handle_batch_command(args, manager)

    elif args.command in ("knowledge", "kt"):
        return handle_knowledge_transfer_command(args, manager)

    elif args.command == "failure":
        return handle_failure_command(args, manager)

    # Query-related commands (query, blocked, active, stats, etc.)
    result = handle_query_commands(args, manager)
    if result is not None:
        return result

    # Fallback
    parser.print_help()
    return 1


if __name__ == "__main__":
    # Handle SIGPIPE gracefully (e.g., when piping to `head`)
    try:
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except AttributeError:
        pass  # SIGPIPE not available on Windows

    try:
        sys.exit(main())
    except BrokenPipeError:
        # Python flushes stdout on exit, which can raise BrokenPipeError
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        sys.exit(0)
