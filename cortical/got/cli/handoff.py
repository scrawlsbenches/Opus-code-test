# MERGE_CONFLICT_RESOLVED: From branch claude/engineering-session-T73QD on 20260108-215836
"""
Handoff CLI commands for GoT system.

Provides commands for agent-to-agent work transfers:
- Initiating handoffs
- Accepting handoffs
- Completing handoffs
- Rejecting handoffs
- Listing handoff status

This module can be integrated into got_utils.py CLI or used standalone.
"""

import json
import subprocess
import sys
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from cortical.got.api import GoTManager


def _get_git_branch() -> str:
    """Get current git branch name."""
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _get_git_modified_files() -> List[str]:
    """Get list of modified files from git status."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True
        )
        files = []
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                # Format: XY filename (where XY is status)
                files.append(line[3:].strip())
        return files
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def _get_recent_commits(count: int = 5) -> List[str]:
    """Get recent commit messages."""
    try:
        result = subprocess.run(
            ["git", "log", f"-{count}", "--oneline"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip().split("\n") if result.stdout.strip() else []
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


# 
# CLI COMMAND HANDLERS
# 

def cmd_handoff_initiate(args, manager: "GoTManager") -> int:
    """Handle 'got handoff initiate' command."""
    task = manager.get_task(args.task_id)
    if not task:
        print(f"Task not found: {args.task_id}")
        return 1

    # Read instructions from stdin if '-' is specified
    instructions = args.instructions
    if instructions == "-":
        instructions = sys.stdin.read().strip()

    # Use manager's handoff method (works with TX backend)
    handoff_id = manager.initiate_handoff(
        source_agent=args.source,
        target_agent=args.target,
        task_id=args.task_id,
        context={
            "task_title": task.content,
            "task_status": task.properties.get("status"),
            "task_priority": task.properties.get("priority"),
        },
        instructions=instructions,
    )

    print(f"Handoff initiated: {handoff_id}")
    print(f"  Task: {task.content}")
    print(f"  From: {args.source} → To: {args.target}")
    if instructions:
        # Truncate for display
        display_instructions = instructions[:100] + "..." if len(instructions) > 100 else instructions
        print(f"  Instructions: {display_instructions}")
    return 0


def cmd_handoff_accept(args, manager: "GoTManager") -> int:
    """Handle 'got handoff accept' command."""
    # Use manager's handoff method (works with TX backend)
    success = manager.accept_handoff(
        handoff_id=args.handoff_id,
        agent=args.agent,
        acknowledgment=args.message,
    )

    if not success:
        print(f"Failed to accept handoff: {args.handoff_id}")
        return 1

    print(f"Handoff accepted: {args.handoff_id}")
    print(f"  Agent: {args.agent}")
    return 0


def cmd_handoff_complete(args, manager: "GoTManager") -> int:
    """Handle 'got handoff complete' command."""
    try:
        result = json.loads(args.result)
    except json.JSONDecodeError:
        result = {"message": args.result}

    # Use manager's handoff method (works with TX backend)
    success = manager.complete_handoff(
        handoff_id=args.handoff_id,
        agent=args.agent,
        result=result,
        artifacts=args.artifacts or [],
    )

    if not success:
        print(f"Failed to complete handoff: {args.handoff_id}")
        return 1

    print(f"Handoff completed: {args.handoff_id}")
    print(f"  Agent: {args.agent}")
    print(f"  Result: {json.dumps(result, indent=2)}")
    return 0


def cmd_handoff_reject(args, manager: "GoTManager") -> int:
    """Handle 'got handoff reject' command."""
    # Read reason from stdin if '-' is specified
    reason = args.reason
    if reason == "-":
        reason = sys.stdin.read().strip()

    success = manager.reject_handoff(
        handoff_id=args.handoff_id,
        agent=args.agent,
        reason=reason,
    )

    if not success:
        print(f"Failed to reject handoff: {args.handoff_id}")
        return 1

    print(f"Handoff rejected: {args.handoff_id}")
    print(f"  Agent: {args.agent}")
    print(f"  Reason: {reason}")
    return 0


def cmd_handoff_show(args, manager: "GoTManager") -> int:
    """Handle 'got handoff show' command."""
    handoff_id = args.handoff_id

    # Get all handoffs and find the one we want
    handoffs = manager.list_handoffs(status=None)
    handoff = None
    for h in handoffs:
        if h.get("id") == handoff_id:
            handoff = h
            break

    if not handoff:
        print(f"Handoff not found: {handoff_id}")
        return 1

    # Display full handoff details
    status = handoff.get("status", "?")
    status_icon = {
        "initiated": "→",
        "accepted": "✓",
        "completed": "✓✓",
        "rejected": "✗",
    }.get(status, "?")

    print("=" * 60)
    print(f"HANDOFF: {handoff_id}")
    print("=" * 60)
    print(f"Status:      {status_icon} {status}")
    print(f"From:        {handoff.get('source_agent', '?')}")
    print(f"To:          {handoff.get('target_agent', '?')}")
    print(f"Task:        {handoff.get('task_id', '?')}")

    if handoff.get("created_at"):
        print(f"Created:     {handoff['created_at']}")
    if handoff.get("accepted_at"):
        print(f"Accepted:    {handoff['accepted_at']}")
    if handoff.get("completed_at"):
        print(f"Completed:   {handoff['completed_at']}")

    # Show context
    context = handoff.get("context", {})
    if context:
        print(f"\nContext:")
        for key, value in context.items():
            print(f"  {key}: {value}")

    # Show full instructions (not truncated)
    if handoff.get("instructions"):
        print(f"\nInstructions:")
        print("-" * 40)
        print(handoff["instructions"])
        print("-" * 40)

    # Show result if completed
    if handoff.get("result"):
        print(f"\nResult:")
        print(json.dumps(handoff["result"], indent=2))

    # Show artifacts if any
    if handoff.get("artifacts"):
        print(f"\nArtifacts:")
        for artifact in handoff["artifacts"]:
            print(f"  - {artifact}")

    print("=" * 60)
    return 0


def cmd_handoff_list(args, manager: "GoTManager") -> int:
    """Handle 'got handoff list' command."""
    # Normalize status: 'in_progress' is an alias for 'accepted'
    # This provides UX consistency with task terminology
    status = args.status
    if status == "in_progress":
        status = "accepted"

    # Use manager's handoff method (works with TX backend)
    handoffs = manager.list_handoffs(status=status)

    if not handoffs:
        print("No handoffs found.")
        return 0

    # Apply limit if specified
    limit = getattr(args, 'limit', None)
    if limit is not None and limit > 0:
        handoffs = handoffs[:limit]

    print(f"Handoffs ({len(handoffs)}):\n")
    for h in handoffs:
        status = h.get("status", "?")
        status_icon = {
            "initiated": "→",
            "accepted": "✓",
            "completed": "✓✓",
            "rejected": "✗",
        }.get(status, "?")

        print(f"  {status_icon} {h['id']}")
        print(f"      {h.get('source_agent', '?')} → {h.get('target_agent', '?')}")
        print(f"      Task: {h.get('task_id', '?')}")
        print(f"      Status: {status}")
        if h.get("instructions"):
            print(f"      Instructions: {h['instructions'][:50]}...")
        print()

    return 0


def cmd_handoff_session(args, manager: "GoTManager") -> int:
    """
    Handle 'got handoff session' command.

    Creates a session-level handoff that captures current git state,
    modified files, and session context for the next agent.
    """
    # Read summary from stdin if '-' is specified
    summary = getattr(args, 'summary', '') or ''
    if summary == "-":
        summary = sys.stdin.read().strip()

    # Auto-capture git context
    branch = getattr(args, 'branch', None) or _get_git_branch()
    files_modified = getattr(args, 'files', None) or _get_git_modified_files()
    recent_commits = _get_recent_commits(5)

    # Build context dict following HandoffContext structure
    context = {
        "current_branch": branch,
        "files_modified": files_modified if isinstance(files_modified, list) else [files_modified] if files_modified else [],
        "recent_commits": recent_commits,
        "session_id": getattr(args, 'session_id', '') or '',
        "blockers": getattr(args, 'blockers', []) or [],
        "notes": getattr(args, 'notes', '') or '',
    }

    # Add KT reference if provided
    kt_id = getattr(args, 'kt', None)
    if kt_id:
        context["kt_id"] = kt_id

    # Build instructions from summary and notes
    instructions = summary
    if context.get("notes"):
        instructions += f"\n\nNotes: {context['notes']}"

    # Create the handoff (no task_id required)
    handoff_id = manager.initiate_handoff(
        source_agent=args.source,
        target_agent=args.target,
        task_id="",  # Session handoff has no specific task
        context=context,
        instructions=instructions,
    )

    # Display summary
    print(f"Session handoff created: {handoff_id}")
    print(f"  From: {args.source} → To: {args.target}")
    print(f"  Branch: {branch}")
    if files_modified:
        print(f"  Modified files: {len(files_modified)}")
    if recent_commits:
        print(f"  Recent commits: {len(recent_commits)}")
    if kt_id:
        print(f"  Knowledge Transfer: {kt_id}")
    if summary:
        display_summary = summary[:100] + "..." if len(summary) > 100 else summary
        print(f"  Summary: {display_summary}")

    print(f"\nNext agent should run:")
    print(f"  python -m cortical.got handoff show {handoff_id}")
    print(f"  python -m cortical.got handoff accept {handoff_id} --agent <agent-id>")

    return 0


#
# CLI INTEGRATION
#

def setup_handoff_parser(subparsers) -> None:
    """
    Set up argparse subparsers for handoff commands.

    Args:
        subparsers: The subparsers object from argparse
    """
    # Create handoff subparser
    handoff_parser = subparsers.add_parser("handoff", help="Agent handoff operations")
    handoff_subparsers = handoff_parser.add_subparsers(
        dest="handoff_command",
        help="Handoff subcommands"
    )

    # handoff initiate
    handoff_init = handoff_subparsers.add_parser(
        "initiate",
        help="Initiate a handoff to another agent"
    )
    handoff_init.add_argument("task_id", help="Task to hand off")
    handoff_init.add_argument(
        "--target", "-t",
        required=True,
        help="Target agent (e.g., 'sub-agent-1')"
    )
    handoff_init.add_argument(
        "--source", "-s",
        default="main",
        help="Source agent (default: main)"
    )
    handoff_init.add_argument(
        "--instructions", "-i",
        default="",
        help="Instructions for target agent (use '-' to read from stdin)"
    )

    # handoff accept
    handoff_accept = handoff_subparsers.add_parser("accept", help="Accept a handoff")
    handoff_accept.add_argument("handoff_id", help="Handoff ID to accept")
    handoff_accept.add_argument("--agent", "-a", required=True, help="Agent accepting")
    handoff_accept.add_argument("--message", "-m", default="", help="Acknowledgment message")

    # handoff complete
    handoff_complete = handoff_subparsers.add_parser("complete", help="Complete a handoff")
    handoff_complete.add_argument("handoff_id", help="Handoff ID to complete")
    handoff_complete.add_argument("--agent", "-a", required=True, help="Agent completing")
    handoff_complete.add_argument("--result", "-r", default="{}", help="Result as JSON")
    handoff_complete.add_argument(
        "--artifacts",
        nargs="*",
        help="Artifacts created (files, commits)"
    )

    # handoff reject
    handoff_reject = handoff_subparsers.add_parser("reject", help="Reject a handoff")
    handoff_reject.add_argument("handoff_id", help="Handoff ID to reject")
    handoff_reject.add_argument("--agent", "-a", required=True, help="Agent rejecting")

    handoff_reject.add_argument(
        "--reason", "-r",
        required=True,
        help="Reason for rejection (use '-' to read from stdin)"
    )

    # handoff show
    handoff_show = handoff_subparsers.add_parser("show", help="Show handoff details")
    handoff_show.add_argument("handoff_id", help="Handoff ID to display")

    # handoff list
    handoff_list = handoff_subparsers.add_parser("list", help="List handoffs")
    handoff_list.add_argument(
        "--status",
        choices=["initiated", "in_progress", "accepted", "completed", "rejected"],
        help="Filter by status (in_progress is alias for accepted)"
    )
    handoff_list.add_argument(
        "--limit", "-n",
        type=int,
        help="Limit number of results"
    )

    # handoff session (session-level handoff without specific task)
    handoff_session = handoff_subparsers.add_parser(
        "session",
        help="Create a session handoff (no task required)"
    )
    handoff_session.add_argument(
        "--target", "-t",
        required=True,
        help="Target agent (e.g., 'next-agent')"
    )
    handoff_session.add_argument(
        "--source", "-s",
        default="cli",
        help="Source agent (default: cli)"
    )
    handoff_session.add_argument(
        "--summary",
        default="",
        help="Session summary (use '-' to read from stdin)"
    )
    handoff_session.add_argument(
        "--notes",
        default="",
        help="Additional notes for next agent"
    )
    handoff_session.add_argument(
        "--branch",
        help="Git branch (auto-detected if not specified)"
    )
    handoff_session.add_argument(
        "--files",
        nargs="*",
        help="Modified files (auto-detected if not specified)"
    )
    handoff_session.add_argument(
        "--blockers",
        nargs="*",
        help="Current blockers"
    )
    handoff_session.add_argument(
        "--kt",
        help="Knowledge Transfer ID to link"
    )
    handoff_session.add_argument(
        "--session-id",
        dest="session_id",
        default="",
        help="Session identifier"
    )


def handle_handoff_command(args, manager: "GoTManager") -> int:
    """
    Route handoff subcommand to appropriate handler.

    Args:
        args: Parsed command-line arguments
        manager: GoTProjectManager instance

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    if not hasattr(args, 'handoff_command') or args.handoff_command is None:
        print("Error: No handoff subcommand specified. Use 'got handoff --help' for usage.")
        return 1

    command_handlers = {
        "initiate": cmd_handoff_initiate,
        "session": cmd_handoff_session,
        "accept": cmd_handoff_accept,
        "complete": cmd_handoff_complete,
        "reject": cmd_handoff_reject,
        "show": cmd_handoff_show,
        "list": cmd_handoff_list,
    }

    handler = command_handlers.get(args.handoff_command)
    if handler:
        return handler(args, manager)

    print(f"Error: Unknown handoff subcommand: {args.handoff_command}")
    return 1
