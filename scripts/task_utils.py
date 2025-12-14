#!/usr/bin/env python3
"""
Merge-friendly task ID management utilities.

This module provides utilities for generating unique task IDs that won't
conflict when multiple agents work in parallel. Follows the same pattern
as cortical/chunk_index.py for append-only, git-friendly storage.

Task ID Format:
    Standalone: T-YYYYMMDD-HHMMSS-XXXX
    Session:    T-YYYYMMDD-HHMMSS-XXXX-NN

    Where:
    - T = Task prefix
    - YYYYMMDD = Date created
    - HHMMSS = Time created
    - XXXX = 4-char random suffix (from session UUID)
    - NN = Task number within session (01, 02, etc.)

Example:
    T-20251213-143052-a1b2       # Standalone
    T-20251213-143052-a1b2-01    # Session task 1
    T-20251213-143052-a1b2-02    # Session task 2

Usage:
    from scripts.task_utils import generate_task_id, TaskSession

    # Simple ID generation (standalone)
    task_id = generate_task_id()  # T-20251213-143052-a1b2

    # Session-based (guaranteed unique within session)
    session = TaskSession()
    task1 = session.new_task_id()  # T-20251213-143052-a1b2-01
    task2 = session.new_task_id()  # T-20251213-143052-a1b2-02
"""

import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


# Directory for per-session task files
DEFAULT_TASKS_DIR = "tasks"


def generate_session_id() -> str:
    """Generate a short session ID (4 hex chars)."""
    return uuid.uuid4().hex[:4]


def generate_task_id(session_id: Optional[str] = None) -> str:
    """
    Generate a unique, merge-friendly task ID.

    Args:
        session_id: Optional session suffix. If None, generates random suffix.

    Returns:
        Task ID in format T-YYYYMMDD-HHMMSSffffff-XXXX (with microseconds)

    Example:
        >>> generate_task_id()
        'T-20251213-143052123456-a1b2'
        >>> generate_task_id("test")
        'T-20251213-143052123456-test'
    """
    now = datetime.now()
    date_str = now.strftime("%Y%m%d")
    # Include microseconds to avoid collisions in tight loops
    time_str = now.strftime("%H%M%S%f")
    suffix = session_id or generate_session_id()
    return f"T-{date_str}-{time_str}-{suffix}"


def generate_short_task_id() -> str:
    """
    Generate a shorter unique task ID (8 hex chars).

    Returns:
        Task ID in format T-XXXXXXXX

    Example:
        >>> generate_short_task_id()
        'T-a1b2c3d4'
    """
    return f"T-{uuid.uuid4().hex[:8]}"


@dataclass
class Task:
    """A single task with merge-friendly ID."""
    id: str
    title: str
    status: str = "pending"  # pending, in_progress, completed, deferred
    priority: str = "medium"  # high, medium, low
    category: str = "general"
    description: str = ""
    depends_on: List[str] = field(default_factory=list)
    effort: str = "medium"  # small, medium, large
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: Optional[str] = None
    completed_at: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    retrospective: Optional[Dict[str, Any]] = None  # Captured on completion

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Task':
        """Create Task from dictionary."""
        return cls(**d)

    def mark_complete(self) -> None:
        """Mark task as completed."""
        self.status = "completed"
        self.completed_at = datetime.now().isoformat()
        self.updated_at = self.completed_at

    def mark_in_progress(self) -> None:
        """Mark task as in progress."""
        self.status = "in_progress"
        self.updated_at = datetime.now().isoformat()


@dataclass
class TaskSession:
    """
    A session for creating tasks with consistent session suffix.

    All tasks created in a session share the same suffix, making it
    easy to identify which tasks were created together.

    Example:
        session = TaskSession()
        task1 = session.create_task("Implement feature X")
        task2 = session.create_task("Add tests for feature X")
        session.save()  # Writes to tasks/2025-12-13_14-30-52_a1b2.json
    """
    session_id: str = field(default_factory=generate_session_id)
    tasks: List[Task] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    tasks_dir: str = DEFAULT_TASKS_DIR
    _task_counter: int = field(default=0, repr=False)

    def new_task_id(self) -> str:
        """Generate a new task ID with this session's suffix and counter.

        The counter ensures unique IDs even when multiple tasks are created
        within the same second.
        """
        self._task_counter += 1
        now = datetime.now()
        date_str = now.strftime("%Y%m%d")
        time_str = now.strftime("%H%M%S")
        # Format: T-YYYYMMDD-HHMMSS-SSSS-NNN where NNN is task number (supports 999 tasks)
        return f"T-{date_str}-{time_str}-{self.session_id}-{self._task_counter:03d}"

    def create_task(
        self,
        title: str,
        priority: str = "medium",
        category: str = "general",
        description: str = "",
        depends_on: Optional[List[str]] = None,
        effort: str = "medium",
        context: Optional[Dict[str, Any]] = None
    ) -> Task:
        """
        Create a new task in this session.

        Args:
            title: Task title/summary
            priority: high, medium, low
            category: Task category (arch, devex, codequal, etc.)
            description: Detailed description
            depends_on: List of task IDs this depends on
            effort: small, medium, large
            context: Quick context dict (files, methods, etc.)

        Returns:
            The created Task object
        """
        task = Task(
            id=self.new_task_id(),
            title=title,
            priority=priority,
            category=category,
            description=description,
            depends_on=depends_on or [],
            effort=effort,
            context=context or {}
        )
        self.tasks.append(task)
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get a task by ID from this session.

        Args:
            task_id: The task ID to find

        Returns:
            The Task object, or None if not found
        """
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def _calculate_duration(self, start_time: str) -> int:
        """
        Calculate duration in minutes from start time to now.

        Args:
            start_time: ISO format datetime string

        Returns:
            Duration in minutes (rounded)
        """
        start = datetime.fromisoformat(start_time)
        now = datetime.now()
        delta = now - start
        return round(delta.total_seconds() / 60)

    def capture_retrospective(
        self,
        task_id: str,
        files_touched: Optional[List[str]] = None,
        tests_added: int = 0,
        commits: Optional[List[str]] = None,
        notes: Optional[str] = None
    ) -> None:
        """
        Capture retrospective data for a completed task.

        This records metadata about what actually happened during task completion:
        - Which files were modified
        - How long it took
        - How many tests were added
        - Which commits were made
        - Any completion notes

        Args:
            task_id: The task ID to add retrospective to
            files_touched: List of file paths that were modified
            tests_added: Number of test cases/functions added
            commits: List of commit SHAs related to this task
            notes: Optional free-form notes about completion

        Raises:
            ValueError: If task_id is not found in this session
        """
        task = self.get_task(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")

        # Calculate duration from creation to now
        duration = self._calculate_duration(task.created_at)

        task.retrospective = {
            'files_touched': files_touched or [],
            'duration_minutes': duration,
            'tests_added': tests_added,
            'commits': commits or [],
            'notes': notes,
            'captured_at': datetime.now().isoformat()
        }

    def get_retrospective_summary(self) -> Dict[str, Any]:
        """
        Get aggregate statistics from all completed tasks with retrospective data.

        Returns:
            Dictionary with aggregate stats:
            - total_completed: Number of tasks with retrospective data
            - avg_duration_minutes: Average task duration
            - total_duration_minutes: Total time spent on all tasks
            - total_tests_added: Sum of all tests added
            - most_touched_files: List of (file, count) tuples for top 10 files
            - tasks_with_retrospective: List of task IDs that have retrospective data
        """
        from collections import Counter

        completed = [
            t for t in self.tasks
            if t.status == 'completed' and t.retrospective
        ]

        if not completed:
            return {
                'total_completed': 0,
                'avg_duration_minutes': 0,
                'total_duration_minutes': 0,
                'total_tests_added': 0,
                'most_touched_files': [],
                'tasks_with_retrospective': []
            }

        # Calculate aggregates
        durations = [t.retrospective['duration_minutes'] for t in completed]
        total_duration = sum(durations)
        avg_duration = total_duration / len(completed)

        total_tests = sum(t.retrospective['tests_added'] for t in completed)

        # Count file touches
        all_files = []
        for t in completed:
            all_files.extend(t.retrospective['files_touched'])
        file_counts = Counter(all_files)

        return {
            'total_completed': len(completed),
            'avg_duration_minutes': round(avg_duration, 1),
            'total_duration_minutes': total_duration,
            'total_tests_added': total_tests,
            'most_touched_files': file_counts.most_common(10),
            'tasks_with_retrospective': [t.id for t in completed]
        }

    def get_filename(self) -> str:
        """Get the session filename."""
        dt = datetime.fromisoformat(self.started_at)
        timestamp = dt.strftime("%Y-%m-%d_%H-%M-%S")
        return f"{timestamp}_{self.session_id}.json"

    def save(self, tasks_dir: Optional[str] = None) -> Path:
        """
        Save session tasks to a JSON file atomically.

        Uses write-to-temp-then-rename pattern to prevent data loss
        if the process crashes during write.

        Args:
            tasks_dir: Directory for task files (default: tasks/)

        Returns:
            Path to the saved file

        Raises:
            OSError: If write or rename fails
        """
        dir_path = Path(tasks_dir or self.tasks_dir)
        dir_path.mkdir(parents=True, exist_ok=True)

        filepath = dir_path / self.get_filename()
        temp_filepath = filepath.with_suffix('.json.tmp')

        data = {
            "version": 1,
            "session_id": self.session_id,
            "started_at": self.started_at,
            "saved_at": datetime.now().isoformat(),
            "tasks": [t.to_dict() for t in self.tasks]
        }

        try:
            # Write to temp file first
            with open(temp_filepath, 'w') as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())  # Ensure data is on disk

            # Atomic rename (on POSIX systems)
            temp_filepath.rename(filepath)
        except Exception:
            # Clean up temp file on failure
            if temp_filepath.exists():
                temp_filepath.unlink()
            raise

        return filepath

    @classmethod
    def load(cls, filepath: Path) -> 'TaskSession':
        """Load a session from file."""
        with open(filepath) as f:
            data = json.load(f)

        session = cls(
            session_id=data['session_id'],
            started_at=data['started_at']
        )
        session.tasks = [Task.from_dict(t) for t in data['tasks']]
        return session


def load_all_tasks(tasks_dir: str = DEFAULT_TASKS_DIR) -> List[Task]:
    """
    Load all tasks from all session files.

    Args:
        tasks_dir: Directory containing task session files

    Returns:
        List of all tasks, sorted by creation time
    """
    dir_path = Path(tasks_dir)
    if not dir_path.exists():
        return []

    all_tasks = []
    for filepath in sorted(dir_path.glob("*.json")):
        try:
            session = TaskSession.load(filepath)
            all_tasks.extend(session.tasks)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load {filepath}: {e}")

    # Sort by creation time
    all_tasks.sort(key=lambda t: t.created_at)
    return all_tasks


def get_task_by_id(task_id: str, tasks_dir: str = DEFAULT_TASKS_DIR) -> Optional[Task]:
    """Find a task by its ID across all session files."""
    for task in load_all_tasks(tasks_dir):
        if task.id == task_id:
            return task
    return None


def consolidate_tasks(
    tasks_dir: str = DEFAULT_TASKS_DIR,
    output_file: Optional[str] = None
) -> Dict[str, List[Task]]:
    """
    Consolidate all tasks from session files into a summary.

    Args:
        tasks_dir: Directory containing task session files
        output_file: Optional path to write consolidated markdown

    Returns:
        Dict of tasks grouped by status
    """
    all_tasks = load_all_tasks(tasks_dir)

    # Group by status
    grouped = {
        "in_progress": [],
        "pending": [],
        "completed": [],
        "deferred": []
    }

    for task in all_tasks:
        status = task.status if task.status in grouped else "pending"
        grouped[status].append(task)

    # Sort within groups by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    for status in grouped:
        grouped[status].sort(key=lambda t: priority_order.get(t.priority, 1))

    if output_file:
        _write_consolidated_markdown(grouped, output_file)

    return grouped


def _write_consolidated_markdown(
    grouped: Dict[str, List[Task]],
    output_file: str
) -> None:
    """Write consolidated tasks to markdown file."""
    lines = [
        "# Consolidated Task List",
        "",
        f"**Generated:** {datetime.now().isoformat()}",
        "",
        "---",
        ""
    ]

    status_headers = {
        "in_progress": "## 🔄 In Progress",
        "pending": "## 📋 Pending",
        "completed": "## ✅ Completed",
        "deferred": "## ⏸️ Deferred"
    }

    for status, tasks in grouped.items():
        if not tasks:
            continue

        lines.append(status_headers.get(status, f"## {status.title()}"))
        lines.append("")
        lines.append("| ID | Title | Priority | Category | Effort |")
        lines.append("|---|------|----------|----------|--------|")

        for task in tasks:
            lines.append(
                f"| {task.id} | {task.title} | {task.priority} | "
                f"{task.category} | {task.effort} |"
            )
        lines.append("")

    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge-friendly task management utilities"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # generate command
    gen_parser = subparsers.add_parser("generate", help="Generate a task ID")
    gen_parser.add_argument("--short", action="store_true", help="Short format (T-XXXXXXXX)")

    # consolidate command
    cons_parser = subparsers.add_parser("consolidate", help="Consolidate task files")
    cons_parser.add_argument("--dir", default=DEFAULT_TASKS_DIR, help="Tasks directory")
    cons_parser.add_argument("--output", help="Output markdown file")

    # list command
    list_parser = subparsers.add_parser("list", help="List all tasks")
    list_parser.add_argument("--dir", default=DEFAULT_TASKS_DIR, help="Tasks directory")
    list_parser.add_argument("--status", help="Filter by status")

    args = parser.parse_args()

    if args.command == "generate":
        if args.short:
            print(generate_short_task_id())
        else:
            print(generate_task_id())

    elif args.command == "consolidate":
        grouped = consolidate_tasks(args.dir, args.output)
        for status, tasks in grouped.items():
            print(f"{status}: {len(tasks)} tasks")
        if args.output:
            print(f"\nWritten to {args.output}")

    elif args.command == "list":
        tasks = load_all_tasks(args.dir)
        if args.status:
            tasks = [t for t in tasks if t.status == args.status]

        for task in tasks:
            print(f"[{task.status}] {task.id}: {task.title}")

    else:
        parser.print_help()
