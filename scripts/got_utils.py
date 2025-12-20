#!/usr/bin/env python3
"""
Graph of Thought Project Management CLI

Manages tasks, sprints, and epics using the Graph of Thought framework.
Replaces file-based task management with graph-native operations.

Usage:
    python scripts/got_utils.py task create "Fix bug" --priority high
    python scripts/got_utils.py task list --status pending
    python scripts/got_utils.py sprint status
    python scripts/got_utils.py migrate --from-files

See docs/got-cli-spec.md for complete command reference.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cortical.reasoning.thought_graph import ThoughtGraph
from cortical.reasoning.graph_of_thought import NodeType, EdgeType, ThoughtNode
from cortical.reasoning.graph_persistence import GraphWAL, GraphRecovery


# =============================================================================
# CONFIGURATION
# =============================================================================

GOT_DIR = PROJECT_ROOT / ".got"
WAL_DIR = GOT_DIR / "wal"
SNAPSHOTS_DIR = GOT_DIR / "snapshots"
TASKS_DIR = PROJECT_ROOT / "tasks"

# Status values
STATUS_PENDING = "pending"
STATUS_IN_PROGRESS = "in_progress"
STATUS_COMPLETED = "completed"
STATUS_BLOCKED = "blocked"
STATUS_DEFERRED = "deferred"

VALID_STATUSES = [STATUS_PENDING, STATUS_IN_PROGRESS, STATUS_COMPLETED,
                  STATUS_BLOCKED, STATUS_DEFERRED]

# Priority values
PRIORITY_CRITICAL = "critical"
PRIORITY_HIGH = "high"
PRIORITY_MEDIUM = "medium"
PRIORITY_LOW = "low"

VALID_PRIORITIES = [PRIORITY_CRITICAL, PRIORITY_HIGH, PRIORITY_MEDIUM, PRIORITY_LOW]

# Category values
VALID_CATEGORIES = ["arch", "feature", "bugfix", "test", "docs", "refactor",
                    "debt", "devex", "security", "performance", "optimization"]


# =============================================================================
# ID GENERATION
# =============================================================================

def generate_task_id() -> str:
    """Generate unique task ID: task:T-YYYYMMDD-HHMMSS-XXXX"""
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    suffix = os.urandom(2).hex()
    return f"task:T-{timestamp}-{suffix}"


def generate_sprint_id(number: Optional[int] = None) -> str:
    """Generate sprint ID: sprint:S-NNN or sprint:YYYY-MM"""
    if number:
        return f"sprint:S-{number:03d}"
    return f"sprint:{datetime.now().strftime('%Y-%m')}"


def generate_epic_id(name: str) -> str:
    """Generate epic ID: epic:E-XXXX"""
    suffix = os.urandom(2).hex()
    return f"epic:E-{suffix}"


def generate_goal_id() -> str:
    """Generate goal ID: goal:G-YYYYMMDD-XXXX"""
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d")
    suffix = os.urandom(2).hex()
    return f"goal:G-{timestamp}-{suffix}"


# =============================================================================
# GRAPH MANAGER
# =============================================================================

class GoTProjectManager:
    """
    Manages project artifacts (tasks, sprints, epics) in a ThoughtGraph.

    Provides CRUD operations with persistence via GraphWAL.
    """

    def __init__(self, got_dir: Path = GOT_DIR):
        self.got_dir = Path(got_dir)
        self.wal_dir = self.got_dir / "wal"
        self.snapshots_dir = self.got_dir / "snapshots"

        # Ensure directories exist
        self.got_dir.mkdir(parents=True, exist_ok=True)
        self.wal_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        # Initialize graph and WAL
        self.graph = ThoughtGraph()
        self.wal = GraphWAL(str(self.wal_dir))

        # Load existing state
        self._load_state()

    def _load_state(self) -> None:
        """Load graph state from WAL/snapshots."""
        # First, try loading from snapshot directly (most common case)
        try:
            snapshot = self.wal.load_snapshot()  # None = load latest
            if snapshot:
                self.graph = snapshot
                return
        except Exception:
            pass

        # If no snapshot, try recovery
        try:
            recovery = GraphRecovery(
                wal_dir=str(self.wal_dir),
                chunks_dir=str(self.got_dir / "chunks"),
            )

            if recovery.needs_recovery():
                result = recovery.recover()
                if result.success and result.graph:
                    self.graph = result.graph
        except Exception:
            # Start with empty graph if all else fails
            pass

    def save(self) -> None:
        """Save current graph state."""
        self.wal.create_snapshot(self.graph, compress=True)

    # =========================================================================
    # TASK OPERATIONS
    # =========================================================================

    def create_task(
        self,
        title: str,
        priority: str = PRIORITY_MEDIUM,
        category: str = "feature",
        description: str = "",
        sprint_id: Optional[str] = None,
        depends_on: Optional[List[str]] = None,
    ) -> str:
        """
        Create a new task.

        Returns:
            Task ID
        """
        task_id = generate_task_id()

        properties = {
            "title": title,
            "status": STATUS_PENDING,
            "priority": priority,
            "category": category,
            "description": description,
        }

        metadata = {
            "created_at": datetime.now().isoformat(),
            "updated_at": None,
            "session_id": os.environ.get("CLAUDE_SESSION_ID", "unknown"),
            "branch": self._get_current_branch(),
        }

        # Create task node
        self.graph.add_node(
            node_id=task_id,
            node_type=NodeType.TASK,
            content=title,
            properties=properties,
            metadata=metadata,
        )

        # Log to WAL
        self.wal.log_add_node(task_id, NodeType.TASK, title, properties, metadata)

        # Add to sprint if specified
        if sprint_id:
            self._add_task_to_sprint(task_id, sprint_id)

        # Add dependencies
        if depends_on:
            for dep_id in depends_on:
                self.add_dependency(task_id, dep_id)

        return task_id

    def get_task(self, task_id: str) -> Optional[ThoughtNode]:
        """Get a task by ID."""
        if not task_id.startswith("task:"):
            task_id = f"task:{task_id}"
        return self.graph.nodes.get(task_id)

    def list_tasks(
        self,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        category: Optional[str] = None,
        sprint_id: Optional[str] = None,
        blocked_only: bool = False,
    ) -> List[ThoughtNode]:
        """
        List tasks with optional filters.
        """
        tasks = []

        for node_id, node in self.graph.nodes.items():
            if node.node_type != NodeType.TASK:
                continue

            props = node.properties

            # Apply filters
            if status and props.get("status") != status:
                continue
            if priority and props.get("priority") != priority:
                continue
            if category and props.get("category") != category:
                continue
            if blocked_only and props.get("status") != STATUS_BLOCKED:
                continue

            # Sprint filter
            if sprint_id:
                if not self._task_in_sprint(node_id, sprint_id):
                    continue

            tasks.append(node)

        # Sort by priority, then by creation date
        priority_order = {PRIORITY_CRITICAL: 0, PRIORITY_HIGH: 1,
                         PRIORITY_MEDIUM: 2, PRIORITY_LOW: 3}
        tasks.sort(key=lambda t: (
            priority_order.get(t.properties.get("priority", PRIORITY_MEDIUM), 2),
            t.metadata.get("created_at", "")
        ))

        return tasks

    def update_task_status(self, task_id: str, status: str) -> bool:
        """Update task status."""
        task = self.get_task(task_id)
        if not task:
            return False

        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status: {status}")

        task.properties["status"] = status
        task.metadata["updated_at"] = datetime.now().isoformat()

        if status == STATUS_COMPLETED:
            task.metadata["completed_at"] = datetime.now().isoformat()

        # Log update
        self.wal.log_update_node(task_id, task.properties, task.metadata)

        return True

    def start_task(self, task_id: str) -> bool:
        """Mark task as in progress."""
        return self.update_task_status(task_id, STATUS_IN_PROGRESS)

    def complete_task(
        self,
        task_id: str,
        retrospective: Optional[str] = None,
    ) -> bool:
        """Complete a task with optional retrospective."""
        task = self.get_task(task_id)
        if not task:
            return False

        task.properties["status"] = STATUS_COMPLETED
        task.metadata["updated_at"] = datetime.now().isoformat()
        task.metadata["completed_at"] = datetime.now().isoformat()

        if retrospective:
            task.properties["retrospective"] = retrospective

        self.wal.log_update_node(task_id, task.properties, task.metadata)

        return True

    def block_task(
        self,
        task_id: str,
        reason: str,
        blocker_id: Optional[str] = None,
    ) -> bool:
        """Block a task with reason."""
        task = self.get_task(task_id)
        if not task:
            return False

        task.properties["status"] = STATUS_BLOCKED
        task.properties["blocked_reason"] = reason
        task.metadata["updated_at"] = datetime.now().isoformat()

        self.wal.log_update_node(task_id, task.properties, task.metadata)

        # Add blocking edge if blocker specified
        if blocker_id:
            if not blocker_id.startswith("task:"):
                blocker_id = f"task:{blocker_id}"
            if blocker_id in self.graph.nodes:
                self.graph.add_edge(
                    blocker_id, task_id, EdgeType.BLOCKS,
                    weight=1.0, confidence=1.0
                )
                self.wal.log_add_edge(blocker_id, task_id, EdgeType.BLOCKS)

        return True

    def add_dependency(self, task_id: str, depends_on_id: str) -> bool:
        """Add dependency between tasks."""
        if not task_id.startswith("task:"):
            task_id = f"task:{task_id}"
        if not depends_on_id.startswith("task:"):
            depends_on_id = f"task:{depends_on_id}"

        if task_id not in self.graph.nodes or depends_on_id not in self.graph.nodes:
            return False

        self.graph.add_edge(
            task_id, depends_on_id, EdgeType.DEPENDS_ON,
            weight=1.0, confidence=1.0
        )
        self.wal.log_add_edge(task_id, depends_on_id, EdgeType.DEPENDS_ON)

        return True

    def get_task_dependencies(self, task_id: str) -> List[ThoughtNode]:
        """Get all tasks this task depends on."""
        if not task_id.startswith("task:"):
            task_id = f"task:{task_id}"

        deps = []
        edges = self.graph._edges_from.get(task_id, [])
        for edge in edges:
            if edge.edge_type == EdgeType.DEPENDS_ON:
                dep_node = self.graph.nodes.get(edge.target_id)
                if dep_node:
                    deps.append(dep_node)

        return deps

    # =========================================================================
    # SPRINT OPERATIONS
    # =========================================================================

    def create_sprint(
        self,
        name: str,
        number: Optional[int] = None,
        epic_id: Optional[str] = None,
    ) -> str:
        """Create a new sprint."""
        sprint_id = generate_sprint_id(number)

        properties = {
            "name": name,
            "status": "available",
            "number": number,
        }

        metadata = {
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
        }

        self.graph.add_node(
            node_id=sprint_id,
            node_type=NodeType.GOAL,  # Sprints are GOAL nodes
            content=name,
            properties=properties,
            metadata=metadata,
        )

        self.wal.log_add_node(sprint_id, NodeType.GOAL, name, properties, metadata)

        # Add to epic if specified
        if epic_id:
            self._add_sprint_to_epic(sprint_id, epic_id)

        return sprint_id

    def get_sprint(self, sprint_id: str) -> Optional[ThoughtNode]:
        """Get sprint by ID."""
        if not sprint_id.startswith("sprint:"):
            sprint_id = f"sprint:{sprint_id}"
        return self.graph.nodes.get(sprint_id)

    def list_sprints(
        self,
        status: Optional[str] = None,
        epic_id: Optional[str] = None,
    ) -> List[ThoughtNode]:
        """List sprints."""
        sprints = []

        for node_id, node in self.graph.nodes.items():
            if not node_id.startswith("sprint:"):
                continue

            if status and node.properties.get("status") != status:
                continue

            sprints.append(node)

        # Sort by number or creation date
        sprints.sort(key=lambda s: (
            s.properties.get("number", 999),
            s.metadata.get("created_at", "")
        ))

        return sprints

    def get_sprint_tasks(self, sprint_id: str) -> List[ThoughtNode]:
        """Get all tasks in a sprint."""
        if not sprint_id.startswith("sprint:"):
            sprint_id = f"sprint:{sprint_id}"

        tasks = []
        edges = self.graph._edges_from.get(sprint_id, [])
        for edge in edges:
            if edge.edge_type == EdgeType.CONTAINS:
                task = self.graph.nodes.get(edge.target_id)
                if task and task.node_type == NodeType.TASK:
                    tasks.append(task)

        return tasks

    def get_sprint_progress(self, sprint_id: str) -> Dict[str, Any]:
        """Get sprint progress statistics."""
        tasks = self.get_sprint_tasks(sprint_id)

        total = len(tasks)
        by_status = {}
        for task in tasks:
            status = task.properties.get("status", STATUS_PENDING)
            by_status[status] = by_status.get(status, 0) + 1

        completed = by_status.get(STATUS_COMPLETED, 0)
        progress_pct = (completed / total * 100) if total > 0 else 0

        return {
            "total_tasks": total,
            "by_status": by_status,
            "completed": completed,
            "progress_percent": progress_pct,
        }

    def _add_task_to_sprint(self, task_id: str, sprint_id: str) -> bool:
        """Add task to sprint via CONTAINS edge."""
        if not sprint_id.startswith("sprint:"):
            sprint_id = f"sprint:{sprint_id}"

        if sprint_id not in self.graph.nodes:
            return False

        self.graph.add_edge(
            sprint_id, task_id, EdgeType.CONTAINS,
            weight=1.0, confidence=1.0
        )
        self.wal.log_add_edge(sprint_id, task_id, EdgeType.CONTAINS)

        return True

    def _task_in_sprint(self, task_id: str, sprint_id: str) -> bool:
        """Check if task is in sprint."""
        if not sprint_id.startswith("sprint:"):
            sprint_id = f"sprint:{sprint_id}"

        edges = self.graph._edges_from.get(sprint_id, [])
        for edge in edges:
            if edge.edge_type == EdgeType.CONTAINS and edge.target_id == task_id:
                return True
        return False

    # =========================================================================
    # EPIC OPERATIONS
    # =========================================================================

    def create_epic(self, name: str, epic_id: Optional[str] = None) -> str:
        """Create a new epic."""
        if not epic_id:
            epic_id = generate_epic_id(name)
        elif not epic_id.startswith("epic:"):
            epic_id = f"epic:{epic_id}"

        properties = {
            "name": name,
            "status": "active",
            "phase": 1,
        }

        metadata = {
            "created_at": datetime.now().isoformat(),
        }

        self.graph.add_node(
            node_id=epic_id,
            node_type=NodeType.GOAL,
            content=name,
            properties=properties,
            metadata=metadata,
        )

        self.wal.log_add_node(epic_id, NodeType.GOAL, name, properties, metadata)

        return epic_id

    def get_epic(self, epic_id: str) -> Optional[ThoughtNode]:
        """Get epic by ID."""
        if not epic_id.startswith("epic:"):
            epic_id = f"epic:{epic_id}"
        return self.graph.nodes.get(epic_id)

    def list_epics(self, status: Optional[str] = None) -> List[ThoughtNode]:
        """List epics."""
        epics = []

        for node_id, node in self.graph.nodes.items():
            if not node_id.startswith("epic:"):
                continue

            if status and node.properties.get("status") != status:
                continue

            epics.append(node)

        return epics

    def _add_sprint_to_epic(self, sprint_id: str, epic_id: str) -> bool:
        """Add sprint to epic."""
        if not epic_id.startswith("epic:"):
            epic_id = f"epic:{epic_id}"

        if epic_id not in self.graph.nodes:
            return False

        self.graph.add_edge(
            epic_id, sprint_id, EdgeType.CONTAINS,
            weight=1.0, confidence=1.0
        )
        self.wal.log_add_edge(epic_id, sprint_id, EdgeType.CONTAINS)

        return True

    # =========================================================================
    # QUERY OPERATIONS
    # =========================================================================

    def get_blocked_tasks(self) -> List[Tuple[ThoughtNode, Optional[str]]]:
        """Get all blocked tasks with their blocking reasons."""
        blocked = []

        for node_id, node in self.graph.nodes.items():
            if node.node_type != NodeType.TASK:
                continue
            if node.properties.get("status") != STATUS_BLOCKED:
                continue

            reason = node.properties.get("blocked_reason", "No reason given")
            blocked.append((node, reason))

        return blocked

    def get_active_tasks(self) -> List[ThoughtNode]:
        """Get all in-progress tasks."""
        return self.list_tasks(status=STATUS_IN_PROGRESS)

    def get_dependency_chain(
        self,
        task_id: str,
        max_depth: int = 10,
    ) -> List[List[ThoughtNode]]:
        """Get full dependency chain for a task."""
        if not task_id.startswith("task:"):
            task_id = f"task:{task_id}"

        chains = []
        visited = set()

        def traverse(node_id: str, chain: List[ThoughtNode], depth: int):
            if depth > max_depth or node_id in visited:
                return

            visited.add(node_id)
            node = self.graph.nodes.get(node_id)
            if not node:
                return

            new_chain = chain + [node]

            deps = self.get_task_dependencies(node_id)
            if not deps:
                chains.append(new_chain)
            else:
                for dep in deps:
                    traverse(dep.id, new_chain, depth + 1)

        traverse(task_id, [], 0)
        return chains

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _get_current_branch(self) -> str:
        """Get current git branch."""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True, text=True, timeout=5
            )
            return result.stdout.strip() or "unknown"
        except Exception:
            return "unknown"

    def export_graph(self, output_path: Optional[Path] = None) -> Dict[str, Any]:
        """Export graph to JSON."""
        nodes = []
        for node_id, node in self.graph.nodes.items():
            nodes.append({
                "id": node_id,
                "type": node.node_type.value,
                "content": node.content,
                "properties": node.properties,
                "metadata": node.metadata,
            })

        edges = []
        for edge in self.graph.edges:
            edges.append({
                "source": edge.source_id,
                "target": edge.target_id,
                "type": edge.edge_type.value,
                "weight": edge.weight,
            })

        data = {
            "exported_at": datetime.now().isoformat(),
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "node_count": len(nodes),
                "edge_count": len(edges),
            }
        }

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)

        return data

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        tasks = self.list_tasks()
        sprints = self.list_sprints()
        epics = self.list_epics()

        by_status = {}
        for task in tasks:
            status = task.properties.get("status", STATUS_PENDING)
            by_status[status] = by_status.get(status, 0) + 1

        return {
            "total_tasks": len(tasks),
            "tasks_by_status": by_status,
            "total_sprints": len(sprints),
            "total_epics": len(epics),
            "total_edges": len(self.graph.edges),
        }


# =============================================================================
# MIGRATION
# =============================================================================

class TaskMigrator:
    """Migrate from file-based task system to GoT."""

    def __init__(self, manager: GoTProjectManager, tasks_dir: Path = TASKS_DIR):
        self.manager = manager
        self.tasks_dir = tasks_dir

    def migrate_all(self, dry_run: bool = False) -> Dict[str, Any]:
        """Migrate all tasks from JSON files."""
        results = {
            "tasks_migrated": 0,
            "tasks_skipped": 0,
            "errors": [],
            "sessions_processed": 0,
        }

        # Find all task session files
        task_files = list(self.tasks_dir.glob("*.json"))

        for task_file in task_files:
            if task_file.name == "legacy_migration.json":
                continue

            try:
                with open(task_file) as f:
                    session_data = json.load(f)

                tasks = session_data.get("tasks", [])
                for task_data in tasks:
                    if dry_run:
                        results["tasks_migrated"] += 1
                        continue

                    try:
                        self._migrate_task(task_data)
                        results["tasks_migrated"] += 1
                    except Exception as e:
                        results["errors"].append(f"Task {task_data.get('id')}: {e}")
                        results["tasks_skipped"] += 1

                results["sessions_processed"] += 1

            except Exception as e:
                results["errors"].append(f"File {task_file.name}: {e}")

        if not dry_run:
            self.manager.save()

        return results

    def _migrate_task(self, task_data: Dict[str, Any]) -> str:
        """Migrate a single task."""
        old_id = task_data.get("id", "")
        title = task_data.get("title", "Untitled")

        # Map old status to new
        status = task_data.get("status", STATUS_PENDING)
        if status not in VALID_STATUSES:
            status = STATUS_PENDING

        # Create task in graph
        task_id = self.manager.create_task(
            title=title,
            priority=task_data.get("priority", PRIORITY_MEDIUM),
            category=task_data.get("category", "feature"),
            description=task_data.get("description", ""),
        )

        # Update status
        task = self.manager.get_task(task_id)
        if task:
            task.properties["status"] = status
            task.properties["legacy_id"] = old_id
            task.metadata["migrated_from"] = old_id
            task.metadata["migrated_at"] = datetime.now().isoformat()

            # Preserve retrospective if present
            retro = task_data.get("retrospective")
            if retro:
                if isinstance(retro, str):
                    task.properties["retrospective"] = retro
                elif isinstance(retro, dict):
                    task.properties["retrospective"] = retro.get("notes", "")

        return task_id


# =============================================================================
# CLI FORMATTING
# =============================================================================

def format_task_table(tasks: List[ThoughtNode]) -> str:
    """Format tasks as table."""
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


def format_sprint_status(sprint: ThoughtNode, progress: Dict[str, Any]) -> str:
    """Format sprint status."""
    lines = [
        f"Sprint: {sprint.content}",
        f"ID: {sprint.id}",
        f"Status: {sprint.properties.get('status', 'unknown')}",
        "",
        f"Progress: {progress['completed']}/{progress['total_tasks']} tasks ({progress['progress_percent']:.1f}%)",
        "",
        "By Status:",
    ]

    for status, count in progress.get("by_status", {}).items():
        lines.append(f"  {status}: {count}")

    return "\n".join(lines)


# =============================================================================
# CLI COMMANDS
# =============================================================================

def cmd_task_create(args, manager: GoTProjectManager) -> int:
    """Create a task."""
    task_id = manager.create_task(
        title=args.title,
        priority=getattr(args, 'priority', PRIORITY_MEDIUM),
        category=getattr(args, 'category', 'feature'),
        description=getattr(args, 'description', ''),
        sprint_id=getattr(args, 'sprint', None),
        depends_on=getattr(args, 'depends', None),
    )

    manager.save()
    print(f"Created: {task_id}")
    return 0


def cmd_task_list(args, manager: GoTProjectManager) -> int:
    """List tasks."""
    tasks = manager.list_tasks(
        status=getattr(args, 'status', None),
        priority=getattr(args, 'priority', None),
        category=getattr(args, 'category', None),
        sprint_id=getattr(args, 'sprint', None),
        blocked_only=getattr(args, 'blocked', False),
    )

    if getattr(args, 'json', False):
        data = [{"id": t.id, "title": t.content, **t.properties} for t in tasks]
        print(json.dumps(data, indent=2))
    else:
        print(format_task_table(tasks))

    return 0


def cmd_task_start(args, manager: GoTProjectManager) -> int:
    """Start a task."""
    if manager.start_task(args.task_id):
        manager.save()
        print(f"Started: {args.task_id}")
        return 0
    else:
        print(f"Task not found: {args.task_id}")
        return 1


def cmd_task_complete(args, manager: GoTProjectManager) -> int:
    """Complete a task."""
    if manager.complete_task(args.task_id, getattr(args, 'retrospective', None)):
        manager.save()
        print(f"Completed: {args.task_id}")
        return 0
    else:
        print(f"Task not found: {args.task_id}")
        return 1


def cmd_task_block(args, manager: GoTProjectManager) -> int:
    """Block a task."""
    if manager.block_task(args.task_id, args.reason, getattr(args, 'blocker', None)):
        manager.save()
        print(f"Blocked: {args.task_id}")
        return 0
    else:
        print(f"Task not found: {args.task_id}")
        return 1


def cmd_sprint_create(args, manager: GoTProjectManager) -> int:
    """Create a sprint."""
    sprint_id = manager.create_sprint(
        name=args.name,
        number=getattr(args, 'number', None),
        epic_id=getattr(args, 'epic', None),
    )

    manager.save()
    print(f"Created: {sprint_id}")
    return 0


def cmd_sprint_list(args, manager: GoTProjectManager) -> int:
    """List sprints."""
    sprints = manager.list_sprints(
        status=getattr(args, 'status', None),
    )

    if not sprints:
        print("No sprints found.")
        return 0

    for sprint in sprints:
        progress = manager.get_sprint_progress(sprint.id)
        status = sprint.properties.get("status", "?")
        print(f"{sprint.id}: {sprint.content} [{status}] - {progress['progress_percent']:.0f}% complete")

    return 0


def cmd_sprint_status(args, manager: GoTProjectManager) -> int:
    """Show sprint status."""
    sprint_id = getattr(args, 'sprint_id', None)

    if sprint_id:
        sprint = manager.get_sprint(sprint_id)
        if not sprint:
            print(f"Sprint not found: {sprint_id}")
            return 1
        sprints = [sprint]
    else:
        # Show all active sprints
        sprints = manager.list_sprints(status="in_progress")
        if not sprints:
            sprints = manager.list_sprints(status="available")

    for sprint in sprints:
        progress = manager.get_sprint_progress(sprint.id)
        print(format_sprint_status(sprint, progress))
        print()

    return 0


def cmd_blocked(args, manager: GoTProjectManager) -> int:
    """Show blocked tasks."""
    blocked = manager.get_blocked_tasks()

    if not blocked:
        print("No blocked tasks.")
        return 0

    print(f"Blocked Tasks ({len(blocked)}):")
    print()

    for task, reason in blocked:
        print(f"  {task.id}")
        print(f"    Title: {task.content}")
        print(f"    Reason: {reason}")
        print()

    return 0


def cmd_active(args, manager: GoTProjectManager) -> int:
    """Show active tasks."""
    active = manager.get_active_tasks()
    print(format_task_table(active))
    return 0


def cmd_stats(args, manager: GoTProjectManager) -> int:
    """Show statistics."""
    stats = manager.get_stats()

    print("GoT Project Statistics:")
    print(f"  Total tasks: {stats['total_tasks']}")
    print(f"  Total sprints: {stats['total_sprints']}")
    print(f"  Total epics: {stats['total_epics']}")
    print(f"  Total edges: {stats['total_edges']}")
    print()
    print("Tasks by status:")
    for status, count in stats.get("tasks_by_status", {}).items():
        print(f"  {status}: {count}")

    return 0


def cmd_migrate(args, manager: GoTProjectManager) -> int:
    """Migrate from file-based system."""
    migrator = TaskMigrator(manager)

    results = migrator.migrate_all(dry_run=getattr(args, 'dry_run', False))

    print("Migration Results:")
    print(f"  Sessions processed: {results['sessions_processed']}")
    print(f"  Tasks migrated: {results['tasks_migrated']}")
    print(f"  Tasks skipped: {results['tasks_skipped']}")

    if results['errors']:
        print()
        print("Errors:")
        for error in results['errors'][:10]:
            print(f"  - {error}")
        if len(results['errors']) > 10:
            print(f"  ... and {len(results['errors']) - 10} more")

    return 0


def cmd_export(args, manager: GoTProjectManager) -> int:
    """Export graph."""
    output = getattr(args, 'output', None)
    if output:
        output = Path(output)

    data = manager.export_graph(output)

    if output:
        print(f"Exported to: {output}")
    else:
        print(json.dumps(data, indent=2))

    return 0


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Graph of Thought Project Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Task commands
    task_parser = subparsers.add_parser("task", help="Task operations")
    task_subparsers = task_parser.add_subparsers(dest="task_command")

    # task create
    create_parser = task_subparsers.add_parser("create", help="Create a task")
    create_parser.add_argument("title", help="Task title")
    create_parser.add_argument("--priority", "-p", choices=VALID_PRIORITIES, default=PRIORITY_MEDIUM)
    create_parser.add_argument("--category", "-c", choices=VALID_CATEGORIES, default="feature")
    create_parser.add_argument("--description", "-d", default="")
    create_parser.add_argument("--sprint", "-s", help="Sprint ID")
    create_parser.add_argument("--depends", nargs="+", help="Dependency task IDs")

    # task list
    list_parser = task_subparsers.add_parser("list", help="List tasks")
    list_parser.add_argument("--status", choices=VALID_STATUSES)
    list_parser.add_argument("--priority", choices=VALID_PRIORITIES)
    list_parser.add_argument("--category", choices=VALID_CATEGORIES)
    list_parser.add_argument("--sprint", help="Filter by sprint")
    list_parser.add_argument("--blocked", action="store_true", help="Show only blocked")
    list_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # task start
    start_parser = task_subparsers.add_parser("start", help="Start a task")
    start_parser.add_argument("task_id", help="Task ID")

    # task complete
    complete_parser = task_subparsers.add_parser("complete", help="Complete a task")
    complete_parser.add_argument("task_id", help="Task ID")
    complete_parser.add_argument("--retrospective", "-r", help="Retrospective notes")

    # task block
    block_parser = task_subparsers.add_parser("block", help="Block a task")
    block_parser.add_argument("task_id", help="Task ID")
    block_parser.add_argument("--reason", "-r", required=True, help="Block reason")
    block_parser.add_argument("--blocker", "-b", help="Blocking task ID")

    # Sprint commands
    sprint_parser = subparsers.add_parser("sprint", help="Sprint operations")
    sprint_subparsers = sprint_parser.add_subparsers(dest="sprint_command")

    # sprint create
    sprint_create = sprint_subparsers.add_parser("create", help="Create a sprint")
    sprint_create.add_argument("name", help="Sprint name")
    sprint_create.add_argument("--number", "-n", type=int, help="Sprint number")
    sprint_create.add_argument("--epic", "-e", help="Epic ID")

    # sprint list
    sprint_list = sprint_subparsers.add_parser("list", help="List sprints")
    sprint_list.add_argument("--status", help="Filter by status")

    # sprint status
    sprint_status = sprint_subparsers.add_parser("status", help="Show sprint status")
    sprint_status.add_argument("sprint_id", nargs="?", help="Sprint ID (optional)")

    # Query commands
    subparsers.add_parser("blocked", help="Show blocked tasks")
    subparsers.add_parser("active", help="Show active tasks")
    subparsers.add_parser("stats", help="Show statistics")

    # Migration commands
    migrate_parser = subparsers.add_parser("migrate", help="Migrate from files")
    migrate_parser.add_argument("--dry-run", action="store_true", help="Don't actually migrate")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export graph")
    export_parser.add_argument("--output", "-o", help="Output file")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Initialize manager
    manager = GoTProjectManager()

    # Route commands
    if args.command == "task":
        if args.task_command == "create":
            return cmd_task_create(args, manager)
        elif args.task_command == "list":
            return cmd_task_list(args, manager)
        elif args.task_command == "start":
            return cmd_task_start(args, manager)
        elif args.task_command == "complete":
            return cmd_task_complete(args, manager)
        elif args.task_command == "block":
            return cmd_task_block(args, manager)
        else:
            task_parser.print_help()
            return 1

    elif args.command == "sprint":
        if args.sprint_command == "create":
            return cmd_sprint_create(args, manager)
        elif args.sprint_command == "list":
            return cmd_sprint_list(args, manager)
        elif args.sprint_command == "status":
            return cmd_sprint_status(args, manager)
        else:
            sprint_parser.print_help()
            return 1

    elif args.command == "blocked":
        return cmd_blocked(args, manager)

    elif args.command == "active":
        return cmd_active(args, manager)

    elif args.command == "stats":
        return cmd_stats(args, manager)

    elif args.command == "migrate":
        return cmd_migrate(args, manager)

    elif args.command == "export":
        return cmd_export(args, manager)

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
