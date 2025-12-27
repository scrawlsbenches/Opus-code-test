"""
Shared utilities and formatters for GoT CLI commands.

Provides common constants, formatting functions, and helpers
used across multiple CLI modules.
"""

from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from cortical.reasoning.graph_of_thought import ThoughtNode


# =============================================================================
# STATUS AND PRIORITY CONSTANTS
# =============================================================================

STATUS_PENDING = "pending"
STATUS_IN_PROGRESS = "in_progress"
STATUS_COMPLETED = "completed"
STATUS_BLOCKED = "blocked"
STATUS_DEFERRED = "deferred"

VALID_STATUSES = [
    STATUS_PENDING,
    STATUS_IN_PROGRESS,
    STATUS_COMPLETED,
    STATUS_BLOCKED,
    STATUS_DEFERRED,
]

PRIORITY_CRITICAL = "critical"
PRIORITY_HIGH = "high"
PRIORITY_MEDIUM = "medium"
PRIORITY_LOW = "low"

VALID_PRIORITIES = [
    PRIORITY_CRITICAL,
    PRIORITY_HIGH,
    PRIORITY_MEDIUM,
    PRIORITY_LOW,
]

# Note: 'research' is for exploration/investigation tasks (read-only, no code changes)
#       'exploration' is for discovery tasks that may lead to proposals
VALID_CATEGORIES = [
    "arch",
    "feature",
    "bugfix",
    "test",
    "docs",
    "refactor",
    "debt",
    "devex",
    "security",
    "performance",
    "optimization",
    "research",
    "exploration",
]

# Priority scoring for suggestion algorithms
PRIORITY_SCORES = {
    PRIORITY_CRITICAL: 100,
    PRIORITY_HIGH: 75,
    PRIORITY_MEDIUM: 50,
    PRIORITY_LOW: 25,
}


# =============================================================================
# FORMATTING UTILITIES
# =============================================================================

def format_task_table(tasks: List["ThoughtNode"]) -> str:
    """
    Format tasks as a text table.

    Args:
        tasks: List of ThoughtNode task objects

    Returns:
        Formatted ASCII table string
    """
    if not tasks:
        return "No tasks found."

    # Header
    lines = [
        "┌" + "─" * 28 + "┬" + "─" * 35 + "┬" + "─" * 12 + "┬" + "─" * 10 + "┐",
        "│ {:26} │ {:33} │ {:10} │ {:8} │".format("ID", "Title", "Status", "Priority"),
        "├" + "─" * 28 + "┼" + "─" * 35 + "┼" + "─" * 12 + "┼" + "─" * 10 + "┤",
    ]

    for task in tasks:
        task_id = task.id.replace("task:", "")[:26]
        title = task.content[:33]
        status = task.properties.get("status", "?")[:10]
        priority = task.properties.get("priority", "?")[:8]

        lines.append("│ {:26} │ {:33} │ {:10} │ {:8} │".format(
            task_id, title, status, priority
        ))

    lines.append("└" + "─" * 28 + "┴" + "─" * 35 + "┴" + "─" * 12 + "┴" + "─" * 10 + "┘")

    return "\n".join(lines)


def format_sprint_status(sprint: "ThoughtNode", progress: Dict[str, Any]) -> str:
    """
    Format sprint status for display.

    Args:
        sprint: Sprint ThoughtNode object
        progress: Dictionary with progress info (completed, total_tasks, progress_percent, by_status)

    Returns:
        Formatted sprint status string
    """
    lines = [
        f"Sprint: {sprint.content}",
        f"ID: {sprint.id}",
        f"Status: {sprint.properties.get('status', 'unknown')}",
    ]

    # Show claimed status if present
    claimed_by = sprint.properties.get('claimed_by')
    if claimed_by:
        lines.append(f"Claimed by: {claimed_by}")
        claimed_at = sprint.properties.get('claimed_at')
        if claimed_at:
            lines.append(f"Claimed at: {claimed_at}")

    lines.extend([
        "",
        f"Progress: {progress['completed']}/{progress['total_tasks']} tasks ({progress['progress_percent']:.1f}%)",
        "",
        "By Status:",
    ])

    for status, count in progress.get("by_status", {}).items():
        lines.append(f"  {status}: {count}")

    return "\n".join(lines)


def format_task_details(task: "ThoughtNode") -> str:
    """
    Format detailed task information.

    Args:
        task: Task ThoughtNode object

    Returns:
        Formatted task details string
    """
    lines = [
        "=" * 60,
        f"TASK: {task.id}",
        "=" * 60,
        f"Title:    {task.content}",
        f"Status:   {task.properties.get('status', 'unknown')}",
        f"Priority: {task.properties.get('priority', 'unknown')}",
        f"Category: {task.properties.get('category', 'unknown')}",
    ]

    if task.properties.get('description'):
        lines.append(f"\nDescription:\n  {task.properties['description']}")

    if task.properties.get('retrospective'):
        lines.append(f"\nRetrospective:\n  {task.properties['retrospective']}")

    if task.properties.get('blocked_reason'):
        lines.append(f"\nBlocked Reason:\n  {task.properties['blocked_reason']}")

    # Show timestamps
    lines.append("\nTimestamps:")
    if task.metadata.get('created_at'):
        lines.append(f"  Created:   {task.metadata['created_at']}")
    if task.metadata.get('updated_at'):
        lines.append(f"  Updated:   {task.metadata['updated_at']}")
    if task.metadata.get('completed_at'):
        lines.append(f"  Completed: {task.metadata['completed_at']}")

    lines.append("=" * 60)

    return "\n".join(lines)


def truncate(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to max length with suffix.

    Args:
        text: Text to truncate
        max_length: Maximum length including suffix
        suffix: Suffix to add when truncating

    Returns:
        Truncated text or original if shorter than max_length
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix
