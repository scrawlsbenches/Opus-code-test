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
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, List, Dict, Any, Optional

from cortical.utils.persistence import atomic_write_json
from cortical.utils.id_generation import generate_short_id

if TYPE_CHECKING:
    from scripts.got_utils import TransactionalGoTAdapter


# =============================================================================
# FAILURE STORAGE
# =============================================================================

def _validate_got_path(got_dir: Path, target_path: Path) -> None:
    """
    Validate that target path is within the .got directory.

    Prevents directory traversal attacks.

    Args:
        got_dir: GoT directory path
        target_path: Path to validate

    Raises:
        ValueError: If path is outside .got directory
    """
    try:
        # Resolve both paths to absolute, canonical forms
        got_dir_resolved = got_dir.resolve()
        target_resolved = target_path.resolve()

        # Check if target is within got_dir
        target_resolved.relative_to(got_dir_resolved)
    except (ValueError, RuntimeError) as e:
        raise ValueError(f"Path {target_path} is outside .got directory") from e


def _get_failures_dir(got_dir: Path) -> Path:
    """Get the failures directory, creating it if needed."""
    failures_dir = got_dir / "failures"

    # Validate path is within .got directory
    _validate_got_path(got_dir, failures_dir)

    failures_dir.mkdir(parents=True, exist_ok=True)
    return failures_dir


def _generate_failure_id() -> str:
    """
    Generate a unique failure ID following project standards.

    Format: F-YYYYMMDD-HHMMSS-XXXXXXXX
    - F-: Failure prefix
    - YYYYMMDD-HHMMSS: UTC timestamp
    - XXXXXXXX: 8 hex characters (random)

    Returns:
        Failure ID string (e.g., 'F-20260103-143052-a1b2c3d4')
    """
    import secrets
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    random_hex = secrets.token_hex(4)  # 8 hex chars
    return f"F-{timestamp}-{random_hex}"


def _sanitize_path_in_text(text: str, replacement: str = "<path>") -> str:
    """
    Sanitize file paths in error messages and stacktraces.

    Removes sensitive system paths that might leak user information.

    Args:
        text: Text to sanitize
        replacement: What to replace paths with (default: "<path>")

    Returns:
        Sanitized text
    """
    if not text:
        return text

    # Replace common path patterns
    # Unix absolute paths: /home/user/..., /usr/..., etc.
    text = re.sub(r'/(?:home|usr|opt|var|tmp)/[^\s,;:"\']+', replacement, text)

    # Windows paths: C:\Users\..., D:\..., etc.
    text = re.sub(r'[A-Z]:\\(?:Users|Program Files|Windows)[^\s,;:"\']+', replacement, text, flags=re.IGNORECASE)

    # Python module paths in tracebacks: File "path/to/file.py"
    text = re.sub(r'File "([^"]+)"', lambda m: f'File "{_sanitize_single_path(m.group(1))}"', text)

    return text


def _sanitize_single_path(path: str) -> str:
    """
    Sanitize a single file path, keeping only relative portion from project root.

    Args:
        path: File path to sanitize

    Returns:
        Sanitized path (relative from project or generic placeholder)
    """
    # Keep only the relative portion if it contains common project indicators
    for indicator in ['cortical/', 'tests/', 'scripts/', '.got/']:
        if indicator in path:
            idx = path.find(indicator)
            return path[idx:]

    # Otherwise return generic placeholder
    return "<path>"


def _validate_task_id(task_id: str) -> None:
    """
    Validate task ID format.

    Args:
        task_id: Task ID to validate

    Raises:
        ValueError: If task ID format is invalid
    """
    if not task_id:
        raise ValueError("Task ID cannot be empty")

    if not task_id.startswith('T-'):
        raise ValueError(f"Invalid task ID format: {task_id} (must start with 'T-')")

    # Basic format check: T-YYYYMMDD-HHMMSS-XXXXXXXX
    pattern = r'^T-\d{8}-\d{6}-[a-f0-9]{8}$'
    if not re.match(pattern, task_id):
        raise ValueError(f"Invalid task ID format: {task_id} (expected: T-YYYYMMDD-HHMMSS-XXXXXXXX)")


def _validate_and_sanitize_error(error_message: str, max_length: int = 10000) -> str:
    """
    Validate and sanitize error message.

    Args:
        error_message: Error message to validate
        max_length: Maximum allowed length (default: 10000)

    Returns:
        Sanitized error message

    Raises:
        ValueError: If error message is empty
    """
    if not error_message or not error_message.strip():
        raise ValueError("Error message cannot be empty")

    # Sanitize paths
    sanitized = _sanitize_path_in_text(error_message)

    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length - 20] + "\n...[truncated]"

    return sanitized


def _save_failure(
    got_dir: Path,
    task_id: str,
    attempt: str,
    error: str,
    lesson: Optional[str] = None,
    files_affected: Optional[List[str]] = None,
) -> str:
    """
    Save a failure record to disk using atomic writes.

    Args:
        got_dir: GoT directory path
        task_id: The task being worked on
        attempt: Description of what was tried
        error: What went wrong
        lesson: What was learned (optional)
        files_affected: List of files involved (optional)

    Returns:
        Failure ID

    Raises:
        ValueError: If validation fails (invalid task_id, empty error, etc.)
        OSError: If file write fails
    """
    # Validate inputs
    _validate_task_id(task_id)
    error_sanitized = _validate_and_sanitize_error(error)

    # Sanitize attempt and lesson
    attempt_sanitized = _sanitize_path_in_text(attempt) if attempt else ""
    lesson_sanitized = _sanitize_path_in_text(lesson) if lesson else None

    # Sanitize file paths (keep only relative paths from project root)
    files_sanitized = []
    if files_affected:
        for file_path in files_affected:
            files_sanitized.append(_sanitize_single_path(file_path))

    # Get failures directory and generate ID
    failures_dir = _get_failures_dir(got_dir)
    failure_id = _generate_failure_id()

    failure_data = {
        "id": failure_id,
        "task_id": task_id,
        "attempt": attempt_sanitized,
        "error": error_sanitized,
        "lesson": lesson_sanitized,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "files_affected": files_sanitized,
    }

    failure_file = failures_dir / f"{failure_id}.json"

    # Validate that failure file path is within .got directory
    _validate_got_path(got_dir, failure_file)

    # Use atomic write to ensure data integrity
    atomic_write_json(failure_file, failure_data, indent=2, mode=0o600)

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
# EDGE CREATION
# =============================================================================

def _create_failure_edge(
    manager: "TransactionalGoTAdapter",
    failure_id: str,
    task_id: str,
    attempt: str,
) -> None:
    """
    Create edge from failure to task with defensive API detection.

    This function safely attempts to create an edge, handling API variations
    and failures gracefully.

    Args:
        manager: TransactionalGoTAdapter instance
        failure_id: Failure ID
        task_id: Task ID
        attempt: Brief description of what was attempted

    Note:
        Failures are stored as JSON files, not entities, so we skip validation.
        If edge creation fails, a warning is printed but the failure is still logged.
    """
    try:
        # Defensive check: Does manager have add_edge method?
        if not hasattr(manager, 'add_edge'):
            print(f"\n⚠ Warning: Manager does not support edges")
            print("  (Failure still logged successfully)")
            return

        # Defensive check: Inspect add_edge signature
        import inspect

        try:
            sig = inspect.signature(manager.add_edge)
        except (ValueError, TypeError) as e:
            print(f"\n⚠ Warning: Could not inspect edge API: {e}")
            print("  (Failure still logged successfully)")
            return

        # Prepare edge arguments based on API signature
        edge_kwargs = {
            "source_id": failure_id,
            "target_id": task_id,
            "edge_type": "FAILED_ATTEMPT",
        }

        # Add reason if supported
        if 'reason' in sig.parameters:
            # Truncate attempt to reasonable length for edge metadata
            reason = f"Failed attempt: {attempt[:50]}"
            if len(attempt) > 50:
                reason += "..."
            edge_kwargs["reason"] = reason

        # Add validate_refs=False if supported (failures are JSON files, not entities)
        if 'validate_refs' in sig.parameters:
            edge_kwargs["validate_refs"] = False

        # Add validate_relationship=False if supported (to allow failure->task edges)
        if 'validate_relationship' in sig.parameters:
            edge_kwargs["validate_relationship"] = False

        # Attempt edge creation
        manager.add_edge(**edge_kwargs)
        print(f"\n✓ Created edge: {failure_id} -> {task_id} (FAILED_ATTEMPT)")

    except Exception as e:
        # Edge creation failed, but that's okay - failure is still logged
        print(f"\n⚠ Warning: Could not create edge: {e}")
        print("  (Failure still logged successfully)")


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

    # Validate inputs early
    try:
        _validate_task_id(task_id)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

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
        print(f"  Attempt: {attempt[:100]}")
        if len(attempt) > 100:
            print(f"           ... (truncated)")
        print(f"  Error:   {error[:100]}")
        if len(error) > 100:
            print(f"           ... (truncated)")
        if lesson:
            print(f"  Lesson:  {lesson[:100]}")
            if len(lesson) > 100:
                print(f"           ... (truncated)")
        if files:
            print(f"  Files:   {', '.join(files[:5])}")
            if len(files) > 5:
                print(f"           ... and {len(files) - 5} more")

        # Create edge from failure to task (FAILED_ATTEMPT)
        # Note: Failures are stored as JSON files, not entities, so we skip validation
        _create_failure_edge(manager, failure_id, task_id, attempt)

        # TODO: Feed into LearningCycle if available
        # This would require checking for the learning cycle module
        # and calling an appropriate method to record the lesson

        return 0

    except ValueError as e:
        # Validation error - user-friendly message
        print(f"Error: {e}")
        return 1
    except OSError as e:
        # File system error
        print(f"Error: Failed to save failure record: {e}")
        return 1
    except Exception as e:
        # Unexpected error
        print(f"Error: Unexpected failure while logging: {e}")
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
