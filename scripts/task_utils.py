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
import sys
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Import canonical ID generation (use as internal implementation)
from cortical.utils.id_generation import generate_task_id as _generate_task_id
from cortical.utils.text import slugify
from cortical.utils.persistence import atomic_write_json


# Directory for per-session task files
DEFAULT_TASKS_DIR = "tasks"


def generate_session_id() -> str:
    """Generate a short session ID (4 hex chars)."""
    return uuid.uuid4().hex[:4]


def generate_task_id(session_id: Optional[str] = None) -> str:
    """
    Generate a unique, merge-friendly task ID.

    This is a backward-compatibility wrapper that adds session_id support
    to the canonical ID generation function. New code should prefer the
    canonical function from cortical.utils.id_generation.

    Args:
        session_id: Optional session suffix. If None, uses canonical generation.

    Returns:
        Task ID in format T-YYYYMMDD-HHMMSS-XXXXXXXX

    Example:
        >>> generate_task_id()
        'T-20251213-143052-a1b2c3d4'
        >>> generate_task_id("test")
        'T-20251213-143052123456-test'

    Note:
        When session_id is provided, uses legacy format with microseconds
        for backward compatibility with existing session files.
    """
    if session_id:
        # Legacy format for backward compatibility with session files
        now = datetime.now()
        date_str = now.strftime("%Y%m%d")
        # Include microseconds to avoid collisions in tight loops
        time_str = now.strftime("%H%M%S%f")
        return f"T-{date_str}-{time_str}-{session_id}"

    # Use canonical ID generation
    return _generate_task_id()


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




def generate_memory_from_task(task: dict) -> str:
    """
    Generate memory entry markdown from a completed task.

    Args:
        task: Task dictionary with id, title, description, retrospective, etc.

    Returns:
        Markdown content for the memory entry
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")

    # Extract task fields
    task_id = task.get('id', 'Unknown')
    title = task.get('title', 'Untitled Task')
    category = task.get('category', 'general')
    description = task.get('description', 'No description provided')

    # Extract retrospective notes
    retrospective = task.get('retrospective', {})
    if isinstance(retrospective, dict):
        notes = retrospective.get('notes', 'No retrospective notes provided')
    else:
        notes = str(retrospective) if retrospective else 'No retrospective notes provided'

    # Extract related files from context or retrospective
    related_files = []
    if 'context' in task and isinstance(task['context'], dict):
        if 'files' in task['context']:
            related_files.extend(task['context']['files'])
    if isinstance(retrospective, dict) and 'files_touched' in retrospective:
        related_files.extend(retrospective['files_touched'])

    # Remove duplicates and format
    related_files = list(set(related_files))
    files_section = "\n".join(f"- `{f}`" for f in related_files) if related_files else "No files recorded"

    # Generate template
    template = f"""# Task Learning: {title}

**Task ID:** {task_id}
**Completed:** {date_str}
**Category:** {category}
**Tags:** `{category}`, `task-learning`

---

## Task Context

{description}

## What Was Learned

{notes}

## Related Files

{files_section}

---

*Auto-generated from task completion*
"""

    return template


def create_memory_for_task(task: dict, output_dir: str = "samples/memories") -> str:
    """
    Create a memory entry file from a completed task.

    Args:
        task: Task dictionary
        output_dir: Directory to write memory file to

    Returns:
        Path to created memory file
    """
    # Generate memory content
    content = generate_memory_from_task(task)

    # Generate merge-safe filename
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")
    session_id = generate_session_id()

    # Use task title for slug
    title = task.get('title', 'task-completion')
    slug = slugify(title)

    filename = f"{date_str}_{time_str}_{session_id}-task-{slug}.md"

    # Create directory if needed
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Write file
    filepath = output_path / filename
    with open(filepath, 'w') as f:
        f.write(content)

    return str(filepath)


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

    def complete_task(
        self,
        task_id: str,
        retrospective: Optional[str] = None,
        create_memory: bool = False
    ) -> Optional[str]:
        """
        Mark a task as completed and optionally create a memory entry.

        Args:
            task_id: The task ID to complete
            retrospective: Optional completion notes/learnings
            create_memory: If True, create a memory entry from the task

        Returns:
            Path to created memory file if create_memory=True, else None

        Raises:
            ValueError: If task_id is not found in this session
        """
        task = self.get_task(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")

        # Mark task as complete
        task.mark_complete()

        # Capture retrospective if provided
        if retrospective:
            duration = self._calculate_duration(task.created_at)
            task.retrospective = {
                'notes': retrospective,
                'duration_minutes': duration,
                'files_touched': [],
                'tests_added': 0,
                'commits': [],
                'captured_at': datetime.now().isoformat()
            }

        # Create memory if requested
        memory_path = None
        if create_memory and task.retrospective:
            memory_path = create_memory_for_task(task.to_dict())

        return memory_path

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

        data = {
            "version": 1,
            "session_id": self.session_id,
            "started_at": self.started_at,
            "saved_at": datetime.now().isoformat(),
            "tasks": [t.to_dict() for t in self.tasks]
        }

        atomic_write_json(filepath, data)
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


# Orchestration integration functions
def create_orchestration_tasks(
    plan: 'OrchestrationPlan',
    session: TaskSession
) -> List[Task]:
    """
    Create tasks from an orchestration plan.

    Creates:
    - One task per batch in the plan
    - Sets depends_on based on batch dependencies
    - Returns list of created tasks

    Args:
        plan: OrchestrationPlan with batches
        session: TaskSession to create tasks in

    Returns:
        List of created Task objects
    """
    # Import at function level to avoid circular imports
    from scripts.orchestration_utils import OrchestrationPlan

    created_tasks = []
    batch_to_task_map = {}  # Map batch_id -> task_id

    for batch in plan.batches:
        # Map batch depends_on to task depends_on
        task_depends_on = []
        for dep_batch_id in batch.depends_on:
            if dep_batch_id in batch_to_task_map:
                task_depends_on.append(batch_to_task_map[dep_batch_id])

        # Create task for this batch
        task = session.create_task(
            title=f"{plan.title} - {batch.name}",
            priority="medium",
            category="orchestration",
            description=f"Batch {batch.batch_id}: {batch.name} ({batch.batch_type})\n"
                       f"{len(batch.agents)} agent(s)",
            depends_on=task_depends_on,
            effort="medium",
            context={
                'plan_id': plan.plan_id,
                'batch_id': batch.batch_id,
                'batch_type': batch.batch_type,
                'agent_count': len(batch.agents)
            }
        )

        # Track mapping for dependency resolution
        batch_to_task_map[batch.batch_id] = task.id
        created_tasks.append(task)

    return created_tasks


def link_plan_to_task(
    plan_id: str,
    task_id: str,
    tasks_dir: str = DEFAULT_TASKS_DIR
) -> None:
    """
    Link an orchestration plan to an existing task.

    Updates the task's context with plan_id reference.

    Args:
        plan_id: The orchestration plan ID
        task_id: The task ID to link
        tasks_dir: Directory containing task files
    """
    # Find the session file containing this task
    dir_path = Path(tasks_dir)
    if not dir_path.exists():
        raise ValueError(f"Tasks directory not found: {tasks_dir}")

    target_session = None
    session_file = None

    for filepath in sorted(dir_path.glob("*.json")):
        try:
            session = TaskSession.load(filepath)
            if session.get_task(task_id):
                target_session = session
                session_file = filepath
                break
        except (json.JSONDecodeError, KeyError):
            continue

    if not target_session:
        raise ValueError(f"Task not found: {task_id}")

    # Update task context with plan_id
    task = target_session.get_task(task_id)
    if task:
        task.context['plan_id'] = plan_id
        task.updated_at = datetime.now().isoformat()

        # Save the session
        target_session.save(tasks_dir)


def get_tasks_for_plan(
    plan_id: str,
    tasks_dir: str = DEFAULT_TASKS_DIR
) -> List[Task]:
    """
    Find all tasks linked to an orchestration plan.

    Args:
        plan_id: The orchestration plan ID
        tasks_dir: Directory containing task files

    Returns:
        List of Task objects linked to the plan
    """
    all_tasks = load_all_tasks(tasks_dir)

    # Filter for tasks with matching plan_id in context
    linked_tasks = [
        task for task in all_tasks
        if isinstance(task.context, dict) and task.context.get('plan_id') == plan_id
    ]

    return linked_tasks


# Sprint tracking utilities
SPRINT_FILE = os.path.join(DEFAULT_TASKS_DIR, "CURRENT_SPRINT.md")


def read_sprint_status(sprint_file: str = SPRINT_FILE) -> Dict[str, Any]:
    """
    Read the current sprint status from CURRENT_SPRINT.md.

    Args:
        sprint_file: Path to the sprint tracking file

    Returns:
        Dictionary with sprint information:
        - sprint_id: Current sprint ID
        - epic: Epic name
        - started: Start date
        - status: Sprint status
        - goals: List of goal dictionaries with 'completed' and 'text'
        - notes: List of notes
        - blocked: List of blocked items

    Raises:
        FileNotFoundError: If sprint file doesn't exist
    """
    if not os.path.exists(sprint_file):
        raise FileNotFoundError(f"Sprint file not found: {sprint_file}")

    with open(sprint_file, 'r') as f:
        content = f.read()

    # Parse the markdown
    sprint_info = {
        'sprint_id': None,
        'epic': None,
        'started': None,
        'status': None,
        'goals': [],
        'completed': [],
        'blocked': [],
        'notes': []
    }

    lines = content.split('\n')
    current_section = None
    in_main_section = True  # Track if we're still in the main sprint section

    for line in lines:
        line_stripped = line.strip()

        # Stop parsing at the first separator (end of current sprint section)
        if line_stripped.startswith('---'):
            break

        # Parse header metadata (only in main section)
        if in_main_section and line_stripped.startswith('**Sprint ID:**'):
            sprint_info['sprint_id'] = line_stripped.split('**Sprint ID:**')[1].strip()
        elif in_main_section and line_stripped.startswith('**Epic:**'):
            sprint_info['epic'] = line_stripped.split('**Epic:**')[1].strip()
        elif in_main_section and line_stripped.startswith('**Started:**'):
            sprint_info['started'] = line_stripped.split('**Started:**')[1].strip()
        elif in_main_section and line_stripped.startswith('**Status:**'):
            sprint_info['status'] = line_stripped.split('**Status:**')[1].strip()

        # Track sections
        elif line_stripped == '## Goals':
            current_section = 'goals'
        elif line_stripped == '## Completed This Sprint':
            current_section = 'completed'
        elif line_stripped == '## Blocked':
            current_section = 'blocked'
        elif line_stripped == '## Notes':
            current_section = 'notes'

        # Parse section content
        elif current_section == 'goals' and line_stripped.startswith('- [ ]'):
            goal_text = line_stripped[5:].strip()
            sprint_info['goals'].append({'completed': False, 'text': goal_text})
        elif current_section == 'goals' and line_stripped.startswith('- [x]'):
            goal_text = line_stripped[5:].strip()
            sprint_info['goals'].append({'completed': True, 'text': goal_text})
        elif current_section == 'completed' and line_stripped.startswith('- [x]'):
            completed_text = line_stripped[5:].strip()
            sprint_info['completed'].append(completed_text)
        elif current_section == 'blocked' and line_stripped.startswith('-'):
            blocked_text = line_stripped[1:].strip()
            if blocked_text and blocked_text != '(None currently)':
                sprint_info['blocked'].append(blocked_text)
        elif current_section == 'notes' and line_stripped.startswith('-'):
            note_text = line_stripped[1:].strip()
            sprint_info['notes'].append(note_text)

    return sprint_info


def update_sprint_goal(
    goal_text: str,
    completed: bool = True,
    sprint_file: str = SPRINT_FILE
) -> None:
    """
    Mark a sprint goal as completed or incomplete.

    Args:
        goal_text: Text of the goal to update
        completed: Whether the goal is completed
        sprint_file: Path to the sprint tracking file

    Raises:
        FileNotFoundError: If sprint file doesn't exist
        ValueError: If goal text is not found
    """
    if not os.path.exists(sprint_file):
        raise FileNotFoundError(f"Sprint file not found: {sprint_file}")

    with open(sprint_file, 'r') as f:
        content = f.read()

    # Find and update the goal
    checkbox_old = '- [ ]' if not completed else '- [x]'
    checkbox_new = '- [x]' if completed else '- [ ]'

    # Try to find the goal with either checkbox state
    goal_patterns = [
        f'- [ ] {goal_text}',
        f'- [x] {goal_text}'
    ]

    found = False
    for pattern in goal_patterns:
        if pattern in content:
            content = content.replace(pattern, f'{checkbox_new} {goal_text}', 1)
            found = True
            break

    if not found:
        raise ValueError(f"Goal not found: {goal_text}")

    with open(sprint_file, 'w') as f:
        f.write(content)


def add_sprint_note(
    note: str,
    sprint_file: str = SPRINT_FILE
) -> None:
    """
    Add a note to the current sprint.

    Args:
        note: Note text to add
        sprint_file: Path to the sprint tracking file

    Raises:
        FileNotFoundError: If sprint file doesn't exist
    """
    if not os.path.exists(sprint_file):
        raise FileNotFoundError(f"Sprint file not found: {sprint_file}")

    with open(sprint_file, 'r') as f:
        content = f.read()

    # Find the Notes section and append the note at the end of the section
    notes_marker = '## Notes'
    if notes_marker not in content:
        raise ValueError("Notes section not found in sprint file")

    lines = content.split('\n')
    new_lines = []
    in_notes_section = False
    note_inserted = False

    for i, line in enumerate(lines):
        if line.strip() == notes_marker:
            in_notes_section = True
            new_lines.append(line)
        elif in_notes_section and (line.strip().startswith('---') or line.strip().startswith('##')):
            # End of notes section - insert note before the next section
            new_lines.append(f'- {note}')
            new_lines.append('')  # Add blank line before next section
            new_lines.append(line)
            in_notes_section = False
            note_inserted = True
        else:
            new_lines.append(line)

    # If we're still in notes section at end of file, append there
    if in_notes_section and not note_inserted:
        new_lines.append(f'- {note}')

    with open(sprint_file, 'w') as f:
        f.write('\n'.join(new_lines))


def get_sprint_summary(sprint_file: str = SPRINT_FILE) -> str:
    """
    Get a human-readable summary of the current sprint.

    Args:
        sprint_file: Path to the sprint tracking file

    Returns:
        Formatted string with sprint summary
    """
    try:
        info = read_sprint_status(sprint_file)
    except FileNotFoundError:
        return "No sprint file found. Create one at tasks/CURRENT_SPRINT.md"

    lines = [
        f"Sprint: {info['sprint_id']}",
        f"Epic: {info['epic']}",
        f"Status: {info['status']}",
        f"Started: {info['started']}",
        "",
        "Goals:"
    ]

    total_goals = len(info['goals'])
    completed_goals = sum(1 for g in info['goals'] if g['completed'])

    for goal in info['goals']:
        checkbox = '[x]' if goal['completed'] else '[ ]'
        lines.append(f"  {checkbox} {goal['text']}")

    lines.append(f"\nProgress: {completed_goals}/{total_goals} goals completed")

    if info['blocked']:
        lines.append("\nBlocked:")
        for blocked in info['blocked']:
            lines.append(f"  - {blocked}")

    if info['notes']:
        lines.append("\nRecent Notes:")
        for note in info['notes'][-3:]:  # Show last 3 notes
            lines.append(f"  - {note}")

    return '\n'.join(lines)


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

    # complete command
    complete_parser = subparsers.add_parser("complete", help="Complete a task")
    complete_parser.add_argument("task_id", help="Task ID to complete")
    complete_parser.add_argument("--retrospective", help="Completion notes/learnings")
    complete_parser.add_argument("--create-memory", action="store_true", help="Create memory entry")
    complete_parser.add_argument("--dir", default=DEFAULT_TASKS_DIR, help="Tasks directory")

    # sprint command
    sprint_parser = subparsers.add_parser("sprint", help="Sprint tracking commands")
    sprint_subparsers = sprint_parser.add_subparsers(dest="sprint_command", help="Sprint commands")

    # sprint status
    sprint_status_parser = sprint_subparsers.add_parser("status", help="Show current sprint status")

    # sprint complete
    sprint_complete_parser = sprint_subparsers.add_parser("complete", help="Mark a sprint goal complete")
    sprint_complete_parser.add_argument("goal", help="Goal text to mark complete")

    # sprint note
    sprint_note_parser = sprint_subparsers.add_parser("note", help="Add a note to current sprint")
    sprint_note_parser.add_argument("note", help="Note text to add")

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

    elif args.command == "complete":
        # Find the session file containing this task
        dir_path = Path(args.dir)
        if not dir_path.exists():
            print(f"Error: Tasks directory not found: {args.dir}")
            sys.exit(1)

        session_file = None
        target_session = None

        for filepath in sorted(dir_path.glob("*.json")):
            try:
                session = TaskSession.load(filepath)
                if session.get_task(args.task_id):
                    session_file = filepath
                    target_session = session
                    break
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load {filepath}: {e}")
                continue

        if not target_session:
            print(f"Error: Could not find session containing task {args.task_id}")
            sys.exit(1)

        # Complete the task
        memory_path = target_session.complete_task(
            args.task_id,
            retrospective=args.retrospective,
            create_memory=args.create_memory
        )

        # Save the session
        target_session.save(args.dir)

        print(f"✓ Task {args.task_id} marked as completed")
        if memory_path:
            print(f"✓ Memory entry created: {memory_path}")

    elif args.command == "sprint":
        if args.sprint_command == "status":
            try:
                summary = get_sprint_summary()
                print(summary)
            except FileNotFoundError as e:
                print(f"Error: {e}")
                sys.exit(1)

        elif args.sprint_command == "complete":
            try:
                update_sprint_goal(args.goal, completed=True)
                print(f"✓ Marked goal as complete: {args.goal}")
            except (FileNotFoundError, ValueError) as e:
                print(f"Error: {e}")
                sys.exit(1)

        elif args.sprint_command == "note":
            try:
                add_sprint_note(args.note)
                print(f"✓ Added note to sprint")
            except (FileNotFoundError, ValueError) as e:
                print(f"Error: {e}")
                sys.exit(1)

        else:
            sprint_parser.print_help()

    else:
        parser.print_help()
