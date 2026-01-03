"""
Failure CLI commands for GoT system.

Provides commands for capturing failed approaches and lessons learned:
- Logging failed attempts with errors and lessons
- Listing recent failures
- Showing failures for a specific task

This helps track what didn't work to avoid repeating mistakes.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, List, Dict, Any, Optional

if TYPE_CHECKING:
    from scripts.got_utils import TransactionalGoTAdapter


# =============================================================================
# FAILURE STORAGE
# =============================================================================

def _get_failures_dir(got_dir: Path) -> Path:
    """Get the failures directory, creating it if needed."""
    failures_dir = got_dir / "failures"
    failures_dir.mkdir(parents=True, exist_ok=True)
    return failures_dir


def _generate_failure_id() -> str:
    """Generate a unique failure ID."""
    import secrets
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    random_hex = secrets.token_hex(4)  # 8 hex chars
    return f"F-{timestamp}-{random_hex}"


def _save_failure(
    got_dir: Path,
    task_id: str,
    attempt: str,
    error: str,
    lesson: Optional[str] = None,
    files_affected: Optional[List[str]] = None,
) -> str:
    """
    Save a failure record to disk.

    Args:
        got_dir: GoT directory path
        task_id: The task being worked on
        attempt: Description of what was tried
        error: What went wrong
        lesson: What was learned (optional)
        files_affected: List of files involved (optional)

    Returns:
        Failure ID
    """
    failures_dir = _get_failures_dir(got_dir)
    failure_id = _generate_failure_id()

    failure_data = {
        "id": failure_id,
        "task_id": task_id,
        "attempt": attempt,
        "error": error,
        "lesson": lesson,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "files_affected": files_affected or [],
    }

    failure_file = failures_dir / f"{failure_id}.json"
    with open(failure_file, 'w') as f:
        json.dump(failure_data, f, indent=2)

    return failure_id


def _load_failure(got_dir: Path, failure_id: str) -> Optional[Dict[str, Any]]:
    """Load a failure record from disk."""
    failures_dir = _get_failures_dir(got_dir)
    failure_file = failures_dir / f"{failure_id}.json"

    if not failure_file.exists():
        return None

    with open(failure_file, 'r') as f:
        return json.load(f)


def _list_all_failures(got_dir: Path) -> List[Dict[str, Any]]:
    """Load all failure records, sorted by timestamp (newest first)."""
    failures_dir = _get_failures_dir(got_dir)
    failures = []

    for failure_file in failures_dir.glob("F-*.json"):
        try:
            with open(failure_file, 'r') as f:
                failure_data = json.load(f)
                failures.append(failure_data)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load {failure_file}: {e}")
            continue

    # Sort by timestamp, newest first
    failures.sort(key=lambda f: f.get("timestamp", ""), reverse=True)
    return failures


def _get_failures_for_task(got_dir: Path, task_id: str) -> List[Dict[str, Any]]:
    """Get all failures for a specific task."""
    all_failures = _list_all_failures(got_dir)
    return [f for f in all_failures if f.get("task_id") == task_id]


# =============================================================================
# CLI COMMAND HANDLERS
# =============================================================================

def cmd_failure_log(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got failure log' command."""
    task_id = args.task_id
    attempt = args.attempt
    error = args.error
    lesson = getattr(args, 'lesson', None)
    files = getattr(args, 'files', None)

    # Verify task exists
    try:
        task = manager.get_task(task_id)
        if not task:
            print(f"Error: Task not found: {task_id}")
            return 1
    except Exception as e:
        print(f"Error: Could not verify task: {e}")
        return 1

    # Save failure record
    try:
        failure_id = _save_failure(
            got_dir=manager.got_dir,
            task_id=task_id,
            attempt=attempt,
            error=error,
            lesson=lesson,
            files_affected=files,
        )

        print(f"Failure logged: {failure_id}")
        print(f"  Task:    {task_id}")
        print(f"  Attempt: {attempt}")
        print(f"  Error:   {error}")
        if lesson:
            print(f"  Lesson:  {lesson}")
        if files:
            print(f"  Files:   {', '.join(files)}")

        # Create edge from failure to task (FAILED_ATTEMPT)
        # Note: Failures are stored as JSON files, not entities, so we skip validation
        try:
            # Check if manager has validate_refs parameter
            import inspect
            sig = inspect.signature(manager.add_edge)
            if 'validate_refs' in sig.parameters:
                manager.add_edge(
                    source_id=failure_id,
                    target_id=task_id,
                    edge_type="FAILED_ATTEMPT",
                    reason=f"Failed attempt: {attempt[:50]}...",
                    validate_refs=False,
                )
            else:
                # Fallback for older API without validate_refs
                manager.add_edge(
                    source_id=failure_id,
                    target_id=task_id,
                    edge_type="FAILED_ATTEMPT",
                    reason=f"Failed attempt: {attempt[:50]}...",
                )
            print(f"\n✓ Created edge: {failure_id} -> {task_id} (FAILED_ATTEMPT)")
        except Exception as e:
            print(f"\n⚠ Warning: Could not create edge: {e}")
            print("  (Failure still logged successfully)")

        # TODO: Feed into LearningCycle if available
        # This would require checking for the learning cycle module
        # and calling an appropriate method to record the lesson

        return 0

    except Exception as e:
        print(f"Error: Failed to log failure: {e}")
        return 1


def cmd_failure_list(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got failure list' command."""
    limit = getattr(args, 'limit', 10)
    task_filter = getattr(args, 'task', None)

    # Get failures
    if task_filter:
        failures = _get_failures_for_task(manager.got_dir, task_filter)
        print(f"Failures for task {task_filter}:\n")
    else:
        failures = _list_all_failures(manager.got_dir)
        print(f"Recent failures:\n")

    if not failures:
        print("No failures logged yet.")
        return 0

    # Apply limit
    if limit and limit > 0:
        failures = failures[:limit]

    # Display failures
    for i, failure in enumerate(failures, 1):
        failure_id = failure.get("id", "Unknown")
        task_id = failure.get("task_id", "Unknown")
        attempt = failure.get("attempt", "")
        error = failure.get("error", "")
        timestamp = failure.get("timestamp", "")

        # Truncate long text
        if len(attempt) > 60:
            attempt = attempt[:57] + "..."
        if len(error) > 60:
            error = error[:57] + "..."

        # Format timestamp
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            time_str = dt.strftime("%Y-%m-%d %H:%M")
        except (ValueError, AttributeError):
            time_str = timestamp[:16] if timestamp else "Unknown"

        print(f"{i}. {failure_id} [{time_str}]")
        print(f"   Task:    {task_id}")
        print(f"   Attempt: {attempt}")
        print(f"   Error:   {error}")

        # Show lesson if present
        lesson = failure.get("lesson")
        if lesson:
            if len(lesson) > 60:
                lesson = lesson[:57] + "..."
            print(f"   Lesson:  {lesson}")

        print()

    total = len(_list_all_failures(manager.got_dir))
    if task_filter:
        print(f"Showing {len(failures)} of {len(_get_failures_for_task(manager.got_dir, task_filter))} failures for this task")
    else:
        print(f"Showing {len(failures)} of {total} total failures")

    return 0


def cmd_failure_show(args, manager: "TransactionalGoTAdapter") -> int:
    """Handle 'got failure show' command."""
    target = args.target

    # Determine if target is a failure ID or task ID
    if target.startswith('F-'):
        # Show single failure
        failure = _load_failure(manager.got_dir, target)
        if not failure:
            print(f"Failure not found: {target}")
            return 1

        failures = [failure]
        print(f"Failure Details: {target}\n")
        print("=" * 70)
    else:
        # Show all failures for task
        failures = _get_failures_for_task(manager.got_dir, target)
        if not failures:
            print(f"No failures found for task: {target}")
            return 0

        print(f"Failures for Task: {target}\n")
        print("=" * 70)

    # Display each failure in detail
    for failure in failures:
        failure_id = failure.get("id", "Unknown")
        task_id = failure.get("task_id", "Unknown")
        attempt = failure.get("attempt", "")
        error = failure.get("error", "")
        lesson = failure.get("lesson", "")
        timestamp = failure.get("timestamp", "")
        files = failure.get("files_affected", [])

        # Format timestamp
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            time_str = dt.strftime("%Y-%m-%d %H:%M:%S %Z")
        except (ValueError, AttributeError):
            time_str = timestamp

        print(f"\nFailure ID:  {failure_id}")
        print(f"Task:        {task_id}")
        print(f"Timestamp:   {time_str}")
        print(f"\nAttempt:")
        print(f"  {attempt}")
        print(f"\nError:")
        print(f"  {error}")

        if lesson:
            print(f"\nLesson Learned:")
            print(f"  {lesson}")

        if files:
            print(f"\nFiles Affected:")
            for file in files:
                print(f"  - {file}")

        print("\n" + "-" * 70)

    print()
    return 0


# =============================================================================
# CLI INTEGRATION
# =============================================================================

def setup_failure_parser(subparsers) -> None:
    """
    Set up argparse subparsers for failure commands.

    Args:
        subparsers: The subparsers object from argparse
    """
    # Create failure subparser
    failure_parser = subparsers.add_parser(
        "failure",
        help="Log and track failed approaches"
    )
    failure_subparsers = failure_parser.add_subparsers(
        dest="failure_command",
        help="Failure subcommands"
    )

    # failure log
    failure_log = failure_subparsers.add_parser(
        "log",
        help="Log a failed attempt"
    )
    failure_log.add_argument(
        "task_id",
        help="Task ID being worked on (e.g., T-XXXX)"
    )
    failure_log.add_argument(
        "--attempt", "-a",
        required=True,
        help="What was tried (brief description)"
    )
    failure_log.add_argument(
        "--error", "-e",
        required=True,
        help="What went wrong"
    )
    failure_log.add_argument(
        "--lesson", "-l",
        help="What was learned (optional)"
    )
    failure_log.add_argument(
        "--files", "-f",
        nargs="+",
        help="Files affected by this attempt (optional)"
    )

    # failure list
    failure_list = failure_subparsers.add_parser(
        "list",
        help="List recent failures"
    )
    failure_list.add_argument(
        "--limit", "-n",
        type=int,
        default=10,
        help="Number of failures to show (default: 10)"
    )
    failure_list.add_argument(
        "--task", "-t",
        help="Filter by task ID"
    )

    # failure show
    failure_show = failure_subparsers.add_parser(
        "show",
        help="Show failure details"
    )
    failure_show.add_argument(
        "target",
        help="Failure ID (F-XXXX) or Task ID (T-XXXX) to show failures for"
    )


def handle_failure_command(args, manager: "TransactionalGoTAdapter") -> int:
    """
    Route failure subcommand to appropriate handler.

    Args:
        args: Parsed command-line arguments
        manager: TransactionalGoTAdapter instance

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    if not hasattr(args, 'failure_command') or args.failure_command is None:
        print("Error: No failure subcommand specified. Use 'got failure --help' for usage.")
        return 1

    command_handlers = {
        "log": cmd_failure_log,
        "list": cmd_failure_list,
        "show": cmd_failure_show,
    }

    handler = command_handlers.get(args.failure_command)
    if handler:
        return handler(args, manager)

    print(f"Error: Unknown failure subcommand: {args.failure_command}")
    return 1
