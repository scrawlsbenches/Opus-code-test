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
import logging
import os
import re
import sys
import tempfile
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cortical.utils.id_generation import (
    generate_task_id,
    generate_decision_id,
    generate_sprint_id,
    generate_epic_id,
    generate_goal_id,
    normalize_id,
)
from cortical.utils.locking import ProcessLock
from cortical.reasoning.thought_graph import ThoughtGraph
from cortical.reasoning.graph_of_thought import NodeType, EdgeType, ThoughtNode, ThoughtEdge
from cortical.reasoning.graph_persistence import GraphWAL, GraphRecovery, GitAutoCommitter
from cortical.got.cli.doc import setup_doc_parser, handle_doc_command
from cortical.got.cli.task import setup_task_parser, handle_task_command
from cortical.got.cli.sprint import setup_sprint_parser, setup_epic_parser, handle_sprint_command, handle_epic_command
from cortical.got.cli.handoff import setup_handoff_parser, handle_handoff_command
from cortical.got.cli.decision import setup_decision_parser, handle_decision_command
from cortical.got.cli.query import setup_query_parser, handle_query_commands
from cortical.got.cli.backup import setup_backup_parser, handle_backup_command, handle_sync_migrate_commands
from cortical.got.cli.orphan import setup_orphan_parser, handle_orphan_command
from cortical.got.cli.backlog import setup_backlog_parser, handle_backlog_command
from cortical.got.cli.analyze import setup_analyze_parser, handle_analyze_command
from cortical.got.cli.edge import setup_edge_parser, handle_edge_command

# Import transactional backend (new)
try:
    from cortical.got.api import GoTManager as TxGoTManager
    from cortical.got.types import Task as TxTask, Decision as TxDecision, Edge as TxEdge
    from cortical.got.config import DurabilityMode
    TX_BACKEND_AVAILABLE = True
except ImportError:
    TX_BACKEND_AVAILABLE = False
    TxGoTManager = None
    TxTask = None
    TxDecision = None
    TxEdge = None
    DurabilityMode = None

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Allow GOT_DIR to be overridden via environment variable (for testing)
GOT_DIR = Path(os.environ.get("GOT_DIR", PROJECT_ROOT / ".got"))
WAL_DIR = GOT_DIR / "wal"
SNAPSHOTS_DIR = GOT_DIR / "snapshots"
EVENTS_DIR = GOT_DIR / "events"  # Git-tracked event logs (legacy, still read)
TASKS_DIR = PROJECT_ROOT / "tasks"

# Backend selection: TX backend is now the DEFAULT when available
# Set GOT_USE_LEGACY=1 to force event-sourced backend (for debugging only)
USE_TX_BACKEND = TX_BACKEND_AVAILABLE and os.environ.get("GOT_USE_LEGACY", "").lower() not in ("1", "true", "yes")

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

# Auto-commit configuration (DEFAULT: ON)
# GoT state is always safe to commit - it's just task/decision tracking data.
# Set GOT_AUTO_COMMIT=0 to disable automatic commits after GoT mutations.
GOT_AUTO_COMMIT_ENABLED = os.environ.get("GOT_AUTO_COMMIT", "1").lower() not in ("0", "false", "no")

# Auto-push configuration (DEFAULT: ON for environment resilience)
# SAFETY: Only pushes to claude/* branches (never main/master/prod)
# Set GOT_AUTO_PUSH=0 to disable automatic push after auto-commit.
GOT_AUTO_PUSH_ENABLED = os.environ.get("GOT_AUTO_PUSH", "1").lower() not in ("0", "false", "no")

# Protected branches that should NEVER be auto-pushed (even if GOT_AUTO_PUSH=1)
PROTECTED_BRANCHES = {"main", "master", "prod", "production", "release"}

# Commands that mutate GoT state (should trigger auto-commit)
MUTATING_COMMANDS = {
    "task": {"create", "start", "complete", "block", "delete", "depends"},
    "sprint": {"create", "start", "complete", "claim", "release", "link", "unlink", "goal"},
    "epic": {"create"},
    "decision": {"log"},
    "handoff": {"initiate", "accept", "complete"},
    "compact": True,  # Always mutating
    "migrate": True,
    "migrate-events": True,
}

# Global auto-committer instance (initialized lazily)
_got_auto_committer: Optional[GitAutoCommitter] = None


def _get_auto_committer() -> Optional[GitAutoCommitter]:
    """Get or create the auto-committer instance."""
    global _got_auto_committer
    if not GOT_AUTO_COMMIT_ENABLED:
        return None
    if _got_auto_committer is None:
        _got_auto_committer = GitAutoCommitter(
            mode='debounced',
            debounce_seconds=2,  # Wait 2s for batch operations
            auto_push=False,  # Don't auto-push, just commit
            repo_path=str(PROJECT_ROOT),
        )
    return _got_auto_committer


def got_auto_commit(command: str, subcommand: Optional[str] = None) -> bool:
    """
    Auto-commit .got/ changes if enabled and command was mutating.

    Args:
        command: Main command (e.g., "task", "sprint")
        subcommand: Subcommand (e.g., "create", "complete")

    Returns:
        True if commit was triggered, False otherwise
    """
    if not GOT_AUTO_COMMIT_ENABLED:
        return False

    # Check if this command mutates state
    cmd_config = MUTATING_COMMANDS.get(command)
    if cmd_config is None:
        return False
    if isinstance(cmd_config, set) and subcommand not in cmd_config:
        return False

    try:
        # Build commit message
        if subcommand:
            msg = f"chore(got): Auto-save after {command} {subcommand}"
        else:
            msg = f"chore(got): Auto-save after {command}"

        # Use direct git commands for .got/ directory
        import subprocess

        # Add all .got/ changes
        subprocess.run(
            ['git', 'add', str(GOT_DIR)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            timeout=10
        )

        # Check if there are staged changes
        result = subprocess.run(
            ['git', 'diff', '--cached', '--quiet'],
            cwd=str(PROJECT_ROOT),
            capture_output=True
        )

        if result.returncode == 0:
            # No changes to commit
            return False

        # Commit
        subprocess.run(
            ['git', 'commit', '-m', msg],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            check=True,
            timeout=10
        )

        logger.info(f"[GoT Auto-commit] {msg}")

        # Auto-push if enabled and on a safe branch
        if GOT_AUTO_PUSH_ENABLED:
            _got_auto_push()

        return True
    except subprocess.CalledProcessError as e:
        logger.debug(f"Auto-commit failed: {e}")
        return False
    except Exception as e:
        logger.debug(f"Auto-commit error: {e}")
        return False


def _got_auto_push() -> bool:
    """
    Auto-push to remote if on a safe branch (claude/*).

    Safety rules:
    - NEVER push to protected branches (main, master, prod, etc.)
    - Only push to claude/* branches (per-session unique, safe)
    - Try once, don't block on failures
    - Use exponential backoff for network errors (up to 3 retries)

    Returns:
        True if push succeeded, False otherwise
    """
    import subprocess
    import time

    try:
        # Get current branch
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=5
        )
        branch = result.stdout.strip()

        # Safety checks
        if branch in PROTECTED_BRANCHES:
            logger.debug(f"[GoT Auto-push] Skipped: {branch} is protected")
            return False

        if not branch.startswith("claude/"):
            logger.debug(f"[GoT Auto-push] Skipped: {branch} is not a claude/* branch")
            return False

        # Push with retries for network errors
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = subprocess.run(
                    ['git', 'push', '-u', 'origin', branch],
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode == 0:
                    logger.info(f"[GoT Auto-push] Pushed to origin/{branch}")
                    return True
                else:
                    # Check if it's a network error worth retrying
                    stderr = result.stderr.lower()
                    if any(err in stderr for err in ['network', 'timeout', 'connection', 'unable to access']):
                        if attempt < max_retries - 1:
                            wait_time = 2 ** (attempt + 1)  # 2, 4 seconds
                            logger.debug(f"[GoT Auto-push] Network error, retry in {wait_time}s")
                            time.sleep(wait_time)
                            continue
                    # Non-network error or final retry failed
                    logger.debug(f"[GoT Auto-push] Failed: {result.stderr}")
                    return False
            except subprocess.TimeoutExpired:
                if attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1)
                    logger.debug(f"[GoT Auto-push] Timeout, retry in {wait_time}s")
                    time.sleep(wait_time)
                    continue
                return False

        return False
    except Exception as e:
        logger.debug(f"[GoT Auto-push] Error: {e}")
        return False


# =============================================================================
# ID GENERATION
# =============================================================================

# ID generation functions now imported from cortical.utils.id_generation
# (canonical source for all ID generation across the codebase)


def get_current_branch() -> str:
    """Get current git branch name."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def generate_session_id() -> str:
    """Generate a unique session ID."""
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    suffix = os.urandom(2).hex()
    return f"{timestamp}-{suffix}"


# =============================================================================
# AUTO-TASK HOOK UTILITIES
# =============================================================================

# Pattern for GoT task IDs: T-YYYYMMDD-HHMMSS-XXXX
TASK_ID_PATTERN = re.compile(r'T-\d{8}-\d{6}-[a-f0-9]{4}', re.IGNORECASE)

# Conventional commit type prefixes
COMMIT_TYPE_PATTERN = re.compile(r'^(\w+):\s*(.+)$')

# Map commit types to GoT categories
COMMIT_TYPE_TO_CATEGORY = {
    'fix': 'bugfix',
    'feat': 'feature',
    'docs': 'docs',
    'refactor': 'refactor',
    'test': 'testing',
    'chore': 'chore',
    'style': 'chore',
    'perf': 'performance',
    'ci': 'chore',
    'build': 'chore',
}


def has_task_reference(commit_message: str) -> bool:
    """
    Check if a commit message contains a GoT task reference.

    Args:
        commit_message: The git commit message

    Returns:
        True if a valid task ID pattern (T-YYYYMMDD-HHMMSS-XXXX) is found
    """
    return bool(TASK_ID_PATTERN.search(commit_message))


def extract_commit_type(commit_message: str) -> Optional[str]:
    """
    Extract the conventional commit type prefix from a commit message.

    Args:
        commit_message: The git commit message

    Returns:
        The commit type (fix, feat, docs, etc.) or None if not found
    """
    match = COMMIT_TYPE_PATTERN.match(commit_message.strip())
    if match:
        return match.group(1).lower()
    return None


def suggest_task_category(commit_type: Optional[str]) -> str:
    """
    Suggest a GoT task category based on the commit type.

    Args:
        commit_type: The conventional commit type (fix, feat, etc.)

    Returns:
        The suggested category for a GoT task
    """
    if commit_type is None:
        return 'general'
    return COMMIT_TYPE_TO_CATEGORY.get(commit_type.lower(), 'general')


def generate_task_title_from_commit(commit_message: str) -> str:
    """
    Generate a task title from a commit message.

    Strips the conventional commit prefix if present.

    Args:
        commit_message: The git commit message

    Returns:
        A clean title suitable for a GoT task
    """
    message = commit_message.strip()
    match = COMMIT_TYPE_PATTERN.match(message)
    if match:
        return match.group(2).strip()
    return message


# =============================================================================
# BACKEND FACTORY
# =============================================================================

class GoTBackendFactory:
    """Factory for creating GoT backend instances (transactional only)."""

    @staticmethod
    def create(
        backend: Optional[str] = None,
        got_dir: Optional[Path] = None,
    ) -> "TransactionalGoTAdapter":
        """
        Create transactional GoT backend.

        Args:
            backend: Ignored (kept for compatibility), always uses transactional
            got_dir: Override default directory

        Returns:
            TransactionalGoTAdapter instance

        Raises:
            RuntimeError: If transactional backend not available
        """
        if not TX_BACKEND_AVAILABLE:
            raise RuntimeError("Transactional backend not available")
        return TransactionalGoTAdapter(got_dir or GOT_DIR)

    @staticmethod
    def get_available_backends() -> List[str]:
        """Get list of available backends (transactional only)."""
        if TX_BACKEND_AVAILABLE:
            return ["transactional"]
        return []


# =============================================================================
# TRANSACTIONAL ADAPTER (New Backend)
# =============================================================================

class TransactionalGoTAdapter:
    """
    Adapter that wraps the transactional GoTManager to provide
    the same interface as GoTProjectManager.

    This enables seamless switching between event-sourced and
    transactional backends without changing command handlers.
    """

    def __init__(self, got_dir: Path = GOT_DIR):
        if not TX_BACKEND_AVAILABLE:
            raise RuntimeError("Transactional backend not available")

        self.got_dir = Path(got_dir)
        self._manager = TxGoTManager(self.got_dir, durability=DurabilityMode.BALANCED)

        # Compatibility attributes (some commands access these directly)
        self._graph = None  # Lazy-loaded graph for compatibility
        self.events_dir = self.got_dir / "events"  # Not used but needed for compat
        self.wal_dir = self.got_dir / "wal"
        self.snapshots_dir = self.got_dir / "snapshots"

        # Ensure directories exist
        self.events_dir.mkdir(parents=True, exist_ok=True)

    @property
    def graph(self) -> ThoughtGraph:
        """Lazy-load graph from transactional store for compatibility."""
        if self._graph is None:
            self._graph = self._build_graph_from_store()
        return self._graph

    def _build_graph_from_store(self) -> ThoughtGraph:
        """Build ThoughtGraph from transactional store entities."""
        graph = ThoughtGraph()
        try:
            # Add all tasks as nodes
            for task in self._manager.list_all_tasks():
                node = self._tx_task_to_node(task)
                graph.nodes[node.id] = node

            # Add all decisions and edges from entity files
            entities_dir = self.got_dir / "entities"
            if entities_dir.exists():
                # Load decisions (D-*.json)
                for decision_file in entities_dir.glob("D-*.json"):
                    try:
                        with open(decision_file, 'r') as f:
                            wrapper = json.load(f)
                        data = wrapper.get("data", {})
                        if data.get("entity_type") == "decision":
                            node = ThoughtNode(
                                id=data.get("id", ""),
                                node_type=NodeType.DECISION,
                                content=data.get("title", ""),
                                properties={
                                    "rationale": data.get("rationale", ""),
                                    "affects": data.get("affects", []),
                                    **data.get("properties", {}),
                                },
                                metadata={
                                    "created_at": data.get("created_at", ""),
                                    "modified_at": data.get("modified_at", ""),
                                },
                            )
                            graph.nodes[node.id] = node
                    except Exception as e:
                        logger.debug(f"Skipping decision file {decision_file}: {e}")

                # Load edges (E-*.json)
                for edge_file in entities_dir.glob("E-*.json"):
                    try:
                        with open(edge_file, 'r') as f:
                            wrapper = json.load(f)
                        data = wrapper.get("data", {})
                        if data.get("entity_type") == "edge":
                            # Edge types are stored lowercase but EdgeType enum uses uppercase
                            edge_type_str = data.get("edge_type", "related_to").upper()
                            edge = ThoughtEdge(
                                source_id=data.get("source_id", ""),
                                target_id=data.get("target_id", ""),
                                edge_type=EdgeType[edge_type_str],
                                weight=data.get("weight", 1.0),
                            )
                            graph.edges.append(edge)
                    except Exception as e:
                        logger.debug(f"Skipping edge file {edge_file}: {e}")
        except Exception as e:
            logger.error(f"Failed to build graph from store: {e}")
        return graph

    def _strip_prefix(self, node_id: str) -> str:
        """Strip task:/decision: prefix from ID (legacy - maintains compatibility with old prefixed IDs)."""
        if node_id.startswith("task:"):
            return node_id[5:]
        if node_id.startswith("decision:"):
            return node_id[9:]
        return node_id

    def _add_prefix(self, node_id: str, prefix: str = "task:") -> str:
        """Add prefix to ID (legacy - now returns ID unchanged)."""
        return node_id  # No longer adding prefixes

    def _tx_task_to_node(self, task: "TxTask") -> ThoughtNode:
        """Convert TxTask to ThoughtNode for compatibility."""
        return ThoughtNode(
            id=task.id,
            node_type=NodeType.TASK,
            content=task.title,
            properties={
                "title": task.title,
                "status": task.status,
                "priority": task.priority,
                "category": task.properties.get("category", ""),
                "description": task.description,
                "retrospective": task.properties.get("retrospective", ""),
                **task.properties,
            },
            metadata={
                "created_at": task.created_at,
                "updated_at": task.modified_at,
                **task.metadata,
            },
        )

    def create_task(
        self,
        title: str,
        priority: str = "medium",
        category: str = "feature",
        description: str = "",
        sprint_id: Optional[str] = None,
        depends_on: Optional[List[str]] = None,
        blocks: Optional[List[str]] = None,
    ) -> str:
        """Create a new task."""
        task = self._manager.create_task(
            title=title,
            priority=priority,
            description=description,
            properties={"category": category},
            metadata={
                "session_id": os.environ.get("CLAUDE_SESSION_ID", "unknown"),
                "branch": self._get_current_branch(),
            },
        )

        # Add dependencies
        if depends_on:
            for dep_id in depends_on:
                clean_dep = self._strip_prefix(dep_id)
                try:
                    self._manager.add_dependency(task.id, clean_dep)
                    print(f"  Added dependency: {task.id} depends on {clean_dep}")
                except Exception as e:
                    logger.warning(f"Could not add dependency from {task.id} to {clean_dep}: {e}")
                    print(f"  Warning: Could not add dependency to {clean_dep}: {e}")

        # Add blocks
        if blocks:
            for blocked_id in blocks:
                clean_blocked = self._strip_prefix(blocked_id)
                try:
                    self._manager.add_blocks(task.id, clean_blocked)
                    print(f"  Added blocks: {task.id} blocks {clean_blocked}")
                except Exception as e:
                    logger.warning(f"Could not add blocks edge from {task.id} to {clean_blocked}: {e}")
                    print(f"  Warning: Could not add blocks to {clean_blocked}: {e}")

        # Add to sprint if specified
        if sprint_id:
            clean_sprint = self._strip_prefix(sprint_id)
            try:
                self._manager.add_edge(clean_sprint, task.id, "CONTAINS")
                print(f"  Added to sprint: {clean_sprint} contains {task.id}")
            except Exception as e:
                logger.warning(f"Could not add task {task.id} to sprint {clean_sprint}: {e}")
                print(f"  Warning: Could not add to sprint {clean_sprint}: {e}")

        return task.id

    def get_task(self, task_id: str) -> Optional[ThoughtNode]:
        """Get a task by ID."""
        clean_id = self._strip_prefix(task_id)
        with self._manager.transaction(read_only=True) as tx:
            task = tx.get_task(clean_id)
            if task:
                return self._tx_task_to_node(task)
        return None

    def list_tasks(
        self,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        category: Optional[str] = None,
        sprint_id: Optional[str] = None,
        blocked_only: bool = False,
    ) -> List[ThoughtNode]:
        """List tasks with optional filters."""
        tasks = self._manager.find_tasks(status=status, priority=priority)

        # Apply additional filters
        result = []
        for task in tasks:
            # Filter by category if specified
            if category and task.properties.get("category") != category:
                continue

            result.append(self._tx_task_to_node(task))

        return result

    def update_task(self, task_id: str, **updates) -> bool:
        """Update a task."""
        clean_id = self._strip_prefix(task_id)
        try:
            self._manager.update_task(clean_id, **updates)
            return True
        except Exception as e:
            logger.error(f"Failed to update task {clean_id}: {e}")
            return False

    def start_task(self, task_id: str) -> bool:
        """Start a task (set status to in_progress)."""
        clean_id = self._strip_prefix(task_id)
        try:
            task = self._manager.get_task(clean_id)
            if not task:
                return False
            # Update metadata with started_at timestamp
            task.metadata["started_at"] = datetime.now(timezone.utc).isoformat()
            task.metadata["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._manager.update_task(clean_id, status="in_progress", metadata=task.metadata)
            return True
        except Exception as e:
            logger.error(f"Failed to start task {clean_id}: {e}")
            return False

    def complete_task(self, task_id: str, retrospective: str = "") -> bool:
        """Complete a task."""
        clean_id = self._strip_prefix(task_id)
        try:
            task = self._manager.get_task(clean_id)
            if not task:
                return False
            # Update metadata with completed_at timestamp
            task.metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
            task.metadata["updated_at"] = datetime.now(timezone.utc).isoformat()
            updates = {"status": "completed", "metadata": task.metadata}
            if retrospective:
                # Copy existing properties and add/update retrospective
                merged_properties = dict(task.properties) if task.properties else {}
                # Filter out status to prevent conflicts
                merged_properties = {k: v for k, v in merged_properties.items() if k != "status"}
                merged_properties["retrospective"] = retrospective
                updates["properties"] = merged_properties
            self._manager.update_task(clean_id, **updates)
            return True
        except Exception as e:
            logger.error(f"Failed to complete task {clean_id}: {e}")
            return False

    def block_task(self, task_id: str, reason: str = "", blocked_by: Optional[str] = None) -> bool:
        """Block a task.

        Args:
            task_id: The task to block
            reason: Why the task is blocked
            blocked_by: Optional task ID that is blocking this task

        Returns:
            True if successful
        """
        clean_id = self._strip_prefix(task_id)
        try:
            # Get task and update properties
            task = self._manager.get_task(clean_id)
            if not task:
                return False

            # Set blocked_reason in properties (where tests expect it)
            props = dict(task.properties)
            props["blocked_reason"] = reason if reason else "No reason given"

            # Update task status and properties
            self._manager.update_task(clean_id, status="blocked", properties=props)

            # Create BLOCKS edge if blocker is specified
            if blocked_by:
                self.add_blocks(blocked_by, task_id)

            return True
        except Exception as e:
            logger.error(f"Failed to block task {clean_id}: {e}")
            return False

    def delete_task(self, task_id: str, force: bool = False) -> Tuple[bool, str]:
        """Delete a task."""
        clean_id = self._strip_prefix(task_id)
        try:
            self._manager.delete_task(clean_id, force=force)
            return True, f"Task {task_id} deleted"
        except Exception as e:
            return False, str(e)

    def add_dependency(self, task_id: str, depends_on_id: str) -> bool:
        """Add a dependency edge."""
        clean_task = self._strip_prefix(task_id)
        clean_dep = self._strip_prefix(depends_on_id)
        try:
            self._manager.add_dependency(clean_task, clean_dep)
            return True
        except AttributeError as e:
            logger.error(f"Method not implemented: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to add dependency from {clean_task} to {clean_dep}: {e}")
            return False

    def add_blocks(self, task_id: str, blocks_id: str) -> bool:
        """Add a blocks edge."""
        clean_task = self._strip_prefix(task_id)
        clean_blocked = self._strip_prefix(blocks_id)
        try:
            self._manager.add_blocks(clean_task, clean_blocked)
            return True
        except AttributeError as e:
            logger.error(f"Method not implemented: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to add blocks edge from {clean_task} to {clean_blocked}: {e}")
            return False

    def add_edge(self, source_id: str, target_id: str, edge_type: str, weight: float = 1.0):
        """Add a generic edge between two entities.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            edge_type: Type of edge (e.g., DEPENDS_ON, BLOCKS, CAUSED_BY)
            weight: Edge weight (default: 1.0)

        Returns:
            Edge object if successful, None otherwise
        """
        clean_source = self._strip_prefix(source_id)
        clean_target = self._strip_prefix(target_id)
        try:
            edge = self._manager.add_edge(clean_source, clean_target, edge_type, weight=weight)
            return edge
        except AttributeError as e:
            logger.error(f"Method not implemented: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to add edge from {clean_source} to {clean_target}: {e}")
            return None

    def list_edges(self) -> List:
        """List all edges in the graph.

        Returns:
            List of Edge objects
        """
        try:
            return self._manager.list_edges()
        except AttributeError as e:
            logger.error(f"Method not implemented: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to list edges: {e}")
            return []

    def get_edges_for_task(self, task_id: str) -> Tuple[List, List]:
        """Get all edges connected to a task.

        Args:
            task_id: Task ID to get edges for

        Returns:
            Tuple of (outgoing_edges, incoming_edges)
        """
        clean_id = self._strip_prefix(task_id)
        try:
            return self._manager.get_edges_for_task(clean_id)
        except AttributeError as e:
            logger.error(f"Method not implemented: {e}")
            return [], []
        except Exception as e:
            logger.error(f"Failed to get edges for {task_id}: {e}")
            return [], []

    def get_blockers(self, task_id: str) -> List[ThoughtNode]:
        """Get tasks that block this task."""
        clean_id = self._strip_prefix(task_id)
        blockers = self._manager.get_blockers(clean_id)
        return [self._tx_task_to_node(t) for t in blockers]

    def get_dependents(self, task_id: str) -> List[ThoughtNode]:
        """Get tasks that depend on this task."""
        clean_id = self._strip_prefix(task_id)
        dependents = self._manager.get_dependents(clean_id)
        return [self._tx_task_to_node(t) for t in dependents]

    def get_task_dependencies(self, task_id: str) -> List[ThoughtNode]:
        """Get all tasks this task depends on.

        Returns tasks that are the target of DEPENDS_ON edges from this task.
        """
        clean_id = self._strip_prefix(task_id)
        try:
            # Get outgoing edges from this task
            outgoing, _ = self._manager.get_edges_for_task(clean_id)
            deps = []
            for edge in outgoing:
                if edge.edge_type == "DEPENDS_ON":
                    dep_task = self._manager.get_task(edge.target_id)
                    if dep_task:
                        deps.append(self._tx_task_to_node(dep_task))
            return deps
        except Exception as e:
            logger.error(f"Failed to get dependencies for {task_id}: {e}")
            return []

    def get_active_tasks(self) -> List[ThoughtNode]:
        """Get all in-progress tasks."""
        try:
            tasks = self._manager.find_tasks(status="in_progress")
            return [self._tx_task_to_node(t) for t in tasks]
        except Exception as e:
            logger.error(f"Failed to get active tasks: {e}")
            return []

    def get_blocked_tasks(self) -> List[Tuple[ThoughtNode, Optional[str]]]:
        """Get all blocked tasks with their blocking reasons."""
        try:
            tasks = self._manager.find_tasks(status="blocked")
            result = []
            for task in tasks:
                node = self._tx_task_to_node(task)
                reason = task.properties.get("blocked_reason", "No reason given")
                result.append((node, reason))
            return result
        except Exception as e:
            logger.error(f"Failed to get blocked tasks: {e}")
            return []

    def get_orphan_tasks(self) -> List[ThoughtNode]:
        """Get all tasks that have no edges (not connected to sprints, decisions, or other tasks).

        Returns:
            List of ThoughtNode objects representing orphan tasks
        """
        try:
            # Get all task IDs
            all_tasks = self._manager.list_all_tasks()
            all_task_ids = {t.id for t in all_tasks}

            # Get all edges and find which tasks are connected
            edges = self._manager.list_edges()
            connected_ids = set()
            for edge in edges:
                if edge.source_id in all_task_ids:
                    connected_ids.add(edge.source_id)
                if edge.target_id in all_task_ids:
                    connected_ids.add(edge.target_id)

            # Find orphan tasks (those with no edges)
            orphan_ids = all_task_ids - connected_ids
            orphan_tasks = [t for t in all_tasks if t.id in orphan_ids]

            return [self._tx_task_to_node(t) for t in orphan_tasks]
        except Exception as e:
            logger.error(f"Failed to get orphan tasks: {e}")
            return []

    def what_blocks(self, task_id: str) -> List[ThoughtNode]:
        """Get tasks blocking this task.

        Follows BLOCKS edges pointing to this task.
        """
        clean_id = self._strip_prefix(task_id)
        try:
            blockers = self._manager.get_blockers(clean_id)
            return [self._tx_task_to_node(t) for t in blockers if t]
        except Exception as e:
            logger.error(f"Failed to get blockers for {task_id}: {e}")
            return []

    def what_depends_on(self, task_id: str) -> List[ThoughtNode]:
        """Get tasks that depend on this task.

        Follows DEPENDS_ON edges pointing to this task.
        """
        clean_id = self._strip_prefix(task_id)
        try:
            dependents = self._manager.get_dependents(clean_id)
            return [self._tx_task_to_node(t) for t in dependents if t]
        except Exception as e:
            logger.error(f"Failed to get dependents for {task_id}: {e}")
            return []

    def list_all_tasks(self) -> List[ThoughtNode]:
        """List all tasks."""
        return self.list_tasks()

    def validate(self) -> List[str]:
        """Validate the GoT state."""
        issues = []
        try:
            # Basic validation
            tasks = self._manager.list_all_tasks()
            if not tasks:
                issues.append("No tasks found")
        except Exception as e:
            issues.append(f"Validation error: {e}")
        return issues

    def _get_current_branch(self) -> str:
        """Get current git branch name."""
        import subprocess
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except Exception as e:
            logger.debug(f"Could not determine git branch: {e}")
            return "unknown"

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        try:
            all_tasks = self._manager.list_all_tasks()

            # Count tasks by status
            by_status = {}
            for task in all_tasks:
                status = task.status
                by_status[status] = by_status.get(status, 0) + 1

            # Count edges
            entities_dir = self._manager.got_dir / "entities"
            edge_count = 0
            if entities_dir.exists():
                edge_count = len(list(entities_dir.glob("E-*.json")))

            # Count sprints and epics (not yet fully supported in TX backend)
            sprints = self.list_sprints()
            # list_epics doesn't exist on adapter yet, so default to empty
            epics = getattr(self, 'list_epics', lambda: [])()

            return {
                "total_tasks": len(all_tasks),
                "tasks_by_status": by_status,
                "total_edges": edge_count,
                "total_sprints": len(sprints),
                "total_epics": len(epics),
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {
                "total_tasks": 0,
                "tasks_by_status": {},
                "total_edges": 0,
                "total_sprints": 0,
                "total_epics": 0,
            }

    def get_all_relationships(self, task_id: str) -> Dict[str, List[ThoughtNode]]:
        """Get all relationships for a task.

        Returns dict with keys:
        - 'blocks': Tasks this task blocks
        - 'blocked_by': Tasks blocking this task
        - 'depends_on': Tasks this task depends on
        - 'depended_by': Tasks depending on this task
        """
        clean_id = self._strip_prefix(task_id)

        result = {
            'blocks': [],
            'blocked_by': [],
            'depends_on': [],
            'depended_by': [],
        }

        try:
            # Get edges for this task
            outgoing, incoming = self._manager.get_edges_for_task(clean_id)

            # Process outgoing edges
            for edge in outgoing:
                target_task = self._manager.get_task(edge.target_id)
                if target_task:
                    target_node = self._tx_task_to_node(target_task)
                    if edge.edge_type == "BLOCKS":
                        result['blocks'].append(target_node)
                    elif edge.edge_type == "DEPENDS_ON":
                        result['depends_on'].append(target_node)

            # Process incoming edges
            for edge in incoming:
                source_task = self._manager.get_task(edge.source_id)
                if source_task:
                    source_node = self._tx_task_to_node(source_task)
                    if edge.edge_type == "BLOCKS":
                        result['blocked_by'].append(source_node)
                    elif edge.edge_type == "DEPENDS_ON":
                        result['depended_by'].append(source_node)

        except Exception as e:
            logger.error(f"Failed to get relationships for {task_id}: {e}")

        return result

    def get_dependency_chain(
        self,
        task_id: str,
        max_depth: int = 10,
    ) -> List[List[ThoughtNode]]:
        """Get full dependency chain for a task.

        Returns list of dependency chains (each chain is a path from task to leaf).
        Uses recursive traversal following DEPENDS_ON edges.
        """
        clean_id = self._strip_prefix(task_id)

        chains = []
        visited = set()

        def traverse(node_id: str, chain: List[ThoughtNode], depth: int):
            if depth > max_depth or node_id in visited:
                return

            visited.add(node_id)
            task = self._manager.get_task(node_id)
            if not task:
                return

            node = self._tx_task_to_node(task)
            new_chain = chain + [node]

            # Get dependencies
            try:
                outgoing, _ = self._manager.get_edges_for_task(node_id)
                deps = []
                for edge in outgoing:
                    if edge.edge_type == "depends_on":
                        dep_task = self._manager.get_task(edge.target_id)
                        if dep_task:
                            deps.append(self._tx_task_to_node(dep_task))

                if not deps:
                    chains.append(new_chain)
                else:
                    for dep in deps:
                        dep_id = self._strip_prefix(dep.id)
                        traverse(dep_id, new_chain, depth + 1)
            except Exception as e:
                logger.error(f"Error traversing dependencies for {node_id}: {e}")
                chains.append(new_chain)

        try:
            traverse(clean_id, [], 0)
        except Exception as e:
            logger.error(f"Failed to get dependency chain for {task_id}: {e}")

        return chains

    def find_path(
        self,
        from_id: str,
        to_id: str,
        max_depth: int = 10,
    ) -> Optional[List[ThoughtNode]]:
        """Find shortest path between two nodes using BFS.

        Follows any edge type to find a path.
        Returns None if no path exists.
        """
        from collections import deque

        clean_from = self._strip_prefix(from_id)
        clean_to = self._strip_prefix(to_id)

        # Check if both nodes exist
        from_task = self._manager.get_task(clean_from)
        to_task = self._manager.get_task(clean_to)

        if not from_task or not to_task:
            return None

        if clean_from == clean_to:
            return [self._tx_task_to_node(from_task)]

        try:
            # BFS
            queue = deque([(clean_from, [clean_from])])
            visited = {clean_from}

            while queue:
                current_id, path = queue.popleft()

                if len(path) > max_depth:
                    continue

                # Get outgoing edges
                outgoing, _ = self._manager.get_edges_for_task(current_id)
                for edge in outgoing:
                    next_id = edge.target_id
                    if next_id == clean_to:
                        # Found the target, construct node path
                        result_path = []
                        for task_id in path + [next_id]:
                            task = self._manager.get_task(task_id)
                            if task:
                                result_path.append(self._tx_task_to_node(task))
                        return result_path

                    if next_id not in visited:
                        visited.add(next_id)
                        queue.append((next_id, path + [next_id]))

        except Exception as e:
            logger.error(f"Failed to find path from {from_id} to {to_id}: {e}")

        return None

    def export_graph(self, output_path: Optional[Path] = None) -> Dict[str, Any]:
        """Export graph to JSON format.

        Args:
            output_path: Optional path to write JSON file

        Returns:
            Dict with 'nodes', 'edges', 'stats', and 'exported_at'
        """
        try:
            # Get all tasks
            all_tasks = self._manager.list_all_tasks()

            nodes = []
            for task in all_tasks:
                nodes.append({
                    "id": task.id,
                    "type": "task",
                    "content": task.title,
                    "properties": {
                        "title": task.title,
                        "status": task.status,
                        "priority": task.priority,
                        "description": task.description,
                        **task.properties,
                    },
                    "metadata": {
                        "created_at": task.created_at,
                        "updated_at": task.modified_at,
                        **task.metadata,
                    },
                })

            # Get all edges
            edges = []
            entities_dir = self._manager.got_dir / "entities"
            if entities_dir.exists():
                for edge_file in entities_dir.glob("E-*.json"):
                    try:
                        with open(edge_file, 'r', encoding='utf-8') as f:
                            wrapper = json.load(f)
                            edge_data = wrapper.get("data", {})
                            edges.append({
                                "source": edge_data.get('source_id', ''),
                                "target": edge_data.get('target_id', ''),
                                "type": edge_data.get("edge_type", ""),
                                "weight": edge_data.get("weight", 1.0),
                            })
                    except Exception as e:
                        logger.warning(f"Skipping corrupted edge file {edge_file}: {e}")
                        continue

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

        except Exception as e:
            logger.error(f"Failed to export graph: {e}")
            return {
                "exported_at": datetime.now().isoformat(),
                "nodes": [],
                "edges": [],
                "stats": {
                    "node_count": 0,
                    "edge_count": 0,
                },
                "error": str(e),
            }

    # Decision methods
    def create_decision(
        self,
        content: str,
        rationale: str = "",
        task_id: Optional[str] = None,
        alternatives: Optional[List[str]] = None,
    ) -> str:
        """Create a decision using TX backend."""
        affects = [task_id] if task_id else []
        decision = self._manager.create_decision(
            title=content,
            rationale=rationale,
            affects=affects,
            alternatives=alternatives or [],
        )
        return decision.id

    def list_decisions(self) -> List[ThoughtNode]:
        """List all decisions from TX backend."""
        from cortical.got.types import Decision
        entities_dir = self.got_dir / "entities"
        if not entities_dir.exists():
            return []

        decisions = []
        for entity_file in entities_dir.glob("D-*.json"):
            try:
                with open(entity_file, 'r') as f:
                    wrapper = json.load(f)
                data = wrapper.get("data", wrapper)
                if data.get("entity_type") == "decision":
                    decision = Decision.from_dict(data)
                    node = ThoughtNode(
                        id=decision.id,
                        node_type=NodeType.DECISION,
                        content=decision.title,
                        properties={
                            "rationale": decision.rationale,
                            "affects": decision.affects,
                            "alternatives": decision.properties.get("alternatives", []),
                        },
                        metadata={
                            "created_at": decision.created_at,
                            "modified_at": decision.modified_at,
                        },
                    )
                    decisions.append(node)
            except Exception:
                continue
        return decisions

    def get_decisions_for_task(self, task_id: str) -> List[ThoughtNode]:
        """Get decisions affecting a specific task."""
        all_decisions = self.list_decisions()
        return [d for d in all_decisions if task_id in d.properties.get("affects", [])]

    def why(self, task_id: str) -> List[Dict[str, Any]]:
        """Query: Why was this task created/modified this way?

        Returns all decisions that affect this task with their rationale.
        """
        decisions = self.get_decisions_for_task(task_id)
        return [
            {
                "decision_id": d.id,
                "decision": d.content,
                "rationale": d.properties.get("rationale", ""),
                "alternatives": d.properties.get("alternatives", []),
                "created_at": d.metadata.get("created_at", ""),
            }
            for d in decisions
        ]

    def log_decision(
        self,
        decision: str,
        rationale: str,
        affects: Optional[List[str]] = None,
        alternatives: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Log a decision with its rationale.

        Creates a decision node and JUSTIFIES edges to affected nodes.
        Future agents can query: "Why was this built this way?"

        Args:
            decision: What was decided
            rationale: Why this choice was made
            affects: List of node IDs affected (tasks, sprints, etc.)
            alternatives: Alternatives that were considered
            context: Additional context (file, line, function)

        Returns:
            Decision ID
        """
        # Build properties dict with alternatives and context
        props: Dict[str, Any] = {}
        if alternatives:
            props["alternatives"] = alternatives
        if context:
            props["context"] = context

        # Create decision via TX backend
        decision_entity = self._manager.create_decision(
            title=decision,
            rationale=rationale,
            affects=affects or [],
            properties=props,
        )

        # Create JUSTIFIES edges to affected nodes
        if affects:
            for affected_id in affects:
                try:
                    self._manager.add_edge(
                        source_id=decision_entity.id,
                        target_id=affected_id,
                        edge_type="JUSTIFIES",
                    )
                except Exception:
                    # Skip if target doesn't exist
                    pass

        return decision_entity.id

    def create_sprint(
        self,
        name: str,
        number: Optional[int] = None,
        epic_id: Optional[str] = None,
    ) -> str:
        """Create a new sprint using TX backend."""
        sprint = self._manager.create_sprint(
            title=name,
            number=number,
            epic_id=epic_id or "",
        )
        return sprint.id

    def get_current_sprint(self) -> Optional[ThoughtNode]:
        """Get the currently active sprint."""
        sprint = self._manager.get_current_sprint()
        if sprint is None:
            return None
        # Convert to ThoughtNode for compatibility
        return ThoughtNode(
            id=sprint.id,
            node_type=NodeType.GOAL,
            content=sprint.title,
            properties={
                "name": sprint.title,
                "status": sprint.status,
                "number": sprint.number,
                "epic_id": sprint.epic_id,
            },
            metadata={
                "created_at": sprint.created_at,
                "modified_at": sprint.modified_at,
            },
        )

    def get_sprint(self, sprint_id: str) -> Optional[ThoughtNode]:
        """Get a sprint by ID."""
        sprint = self._manager.get_sprint(sprint_id)
        if sprint is None:
            return None
        # Merge sprint.properties into the node properties
        props = {
            "name": sprint.title,
            "status": sprint.status,
            "number": sprint.number,
            "epic_id": sprint.epic_id,
            "session_id": sprint.session_id,
            "isolation": sprint.isolation,
            "goals": sprint.goals,
            "notes": sprint.notes,
        }
        # Include custom properties (like claimed_by, claimed_at)
        props.update(sprint.properties)

        return ThoughtNode(
            id=sprint.id,
            node_type=NodeType.GOAL,
            content=sprint.title,
            properties=props,
            metadata={
                "created_at": sprint.created_at,
                "modified_at": sprint.modified_at,
            },
        )

    def list_sprints(self, status: Optional[str] = None, epic_id: Optional[str] = None) -> List[ThoughtNode]:
        """List sprints from TX backend."""
        sprints = self._manager.list_sprints(status=status, epic_id=epic_id)
        result = []
        for sprint in sprints:
            # Merge sprint.properties into the node properties
            props = {
                "name": sprint.title,
                "status": sprint.status,
                "number": sprint.number,
                "epic_id": sprint.epic_id,
            }
            # Include custom properties (like claimed_by, claimed_at)
            props.update(sprint.properties)

            node = ThoughtNode(
                id=sprint.id,
                node_type=NodeType.GOAL,
                content=sprint.title,
                properties=props,
                metadata={
                    "created_at": sprint.created_at,
                    "modified_at": sprint.modified_at,
                },
            )
            result.append(node)
        return result

    def update_sprint(self, sprint_id: str, **updates) -> ThoughtNode:
        """Update a sprint."""
        sprint = self._manager.update_sprint(sprint_id, **updates)
        # Convert to ThoughtNode
        props = {
            "name": sprint.title,
            "status": sprint.status,
            "number": sprint.number,
            "epic_id": sprint.epic_id,
        }
        props.update(sprint.properties)

        return ThoughtNode(
            id=sprint.id,
            node_type=NodeType.GOAL,
            content=sprint.title,
            properties=props,
            metadata={
                "created_at": sprint.created_at,
                "modified_at": sprint.modified_at,
            },
        )

    def claim_sprint(self, sprint_id: str, agent: str) -> ThoughtNode:
        """Claim a sprint for an agent."""
        sprint = self._manager.get_sprint(sprint_id)
        if not sprint:
            raise ValueError(f"Sprint not found: {sprint_id}")

        # Check if already claimed by different agent
        current_owner = sprint.properties.get("claimed_by")
        if current_owner and current_owner != agent:
            raise ValueError(f"Sprint already claimed by {current_owner}")

        # Update sprint with claim
        return self.update_sprint(
            sprint_id,
            properties={
                **sprint.properties,
                "claimed_by": agent,
                "claimed_at": datetime.now(timezone.utc).isoformat()
            }
        )

    def release_sprint(self, sprint_id: str, agent: str) -> ThoughtNode:
        """Release a sprint claim."""
        sprint = self._manager.get_sprint(sprint_id)
        if not sprint:
            raise ValueError(f"Sprint not found: {sprint_id}")

        # Verify the agent owns the claim
        current_owner = sprint.properties.get("claimed_by")
        if current_owner != agent:
            raise ValueError(f"Sprint not claimed by {agent}")

        # Clear claim
        new_props = dict(sprint.properties)
        new_props.pop("claimed_by", None)
        new_props.pop("claimed_at", None)

        return self.update_sprint(
            sprint_id,
            properties=new_props
        )

    def add_sprint_goal(self, sprint_id: str, description: str) -> bool:
        """Add a goal to a sprint."""
        sprint = self._manager.get_sprint(sprint_id)
        if not sprint:
            return False

        goals = list(sprint.goals)  # Copy existing goals
        goals.append({"description": description, "completed": False})

        self._manager.update_sprint(sprint_id, goals=goals)
        return True

    def list_sprint_goals(self, sprint_id: str) -> List[Dict]:
        """List goals for a sprint."""
        sprint = self._manager.get_sprint(sprint_id)
        if not sprint:
            return []
        return sprint.goals

    def complete_sprint_goal(self, sprint_id: str, goal_index: int) -> bool:
        """Mark a goal as complete by index."""
        sprint = self._manager.get_sprint(sprint_id)
        if not sprint:
            return False

        goals = list(sprint.goals)
        if goal_index < 0 or goal_index >= len(goals):
            return False

        goals[goal_index]["completed"] = True
        self._manager.update_sprint(sprint_id, goals=goals)
        return True

    def link_task_to_sprint(self, sprint_id: str, task_id: str) -> bool:
        """Link a task to a sprint via CONTAINS edge."""
        # Verify both exist
        sprint = self._manager.get_sprint(sprint_id)
        task = self._manager.get_task(task_id)
        if not sprint or not task:
            return False

        # Create CONTAINS edge from sprint to task
        self._manager.add_task_to_sprint(task_id, sprint_id)
        return True

    def unlink_task_from_sprint(self, sprint_id: str, task_id: str) -> bool:
        """Remove task from sprint by deleting the CONTAINS edge."""
        # Find the CONTAINS edge
        entities_dir = self._manager.got_dir / "entities"
        if not entities_dir.exists():
            return False

        for edge_file in entities_dir.glob("E-*.json"):
            try:
                with open(edge_file, 'r', encoding='utf-8') as f:
                    wrapper = json.load(f)
                data = wrapper.get("data", {})

                if (data.get("entity_type") == "edge" and
                    data.get("source_id") == sprint_id and
                    data.get("target_id") == task_id and
                    data.get("edge_type") == "CONTAINS"):
                    # Delete the edge file
                    edge_file.unlink()
                    return True
            except (json.JSONDecodeError, KeyError, OSError):
                continue

        return False

    def list_epics(self, status: Optional[str] = None) -> List[ThoughtNode]:
        """List epics from TX backend."""
        epics = self._manager.list_epics(status=status)
        result = []
        for epic in epics:
            node = ThoughtNode(
                id=epic.id,
                node_type=NodeType.GOAL,
                content=epic.title,
                properties={
                    "name": epic.title,
                    "status": epic.status,
                    "phase": epic.phase,
                },
                metadata={
                    "created_at": epic.created_at,
                    "modified_at": epic.modified_at,
                },
            )
            result.append(node)
        return result

    def create_epic(self, name: str, epic_id: Optional[str] = None) -> str:
        """Create a new epic using TX backend."""
        epic = self._manager.create_epic(title=name, epic_id=epic_id)
        return epic.id

    def get_epic(self, epic_id: str) -> Optional[ThoughtNode]:
        """Get an epic by ID."""
        epic = self._manager.get_epic(epic_id)
        if epic is None:
            return None
        return ThoughtNode(
            id=epic.id,
            node_type=NodeType.GOAL,
            content=epic.title,
            properties={
                "name": epic.title,
                "status": epic.status,
                "phase": epic.phase,
                "phases": epic.phases,
            },
            metadata={
                "created_at": epic.created_at,
                "modified_at": epic.modified_at,
            },
        )

    def initiate_handoff(
        self,
        source_agent: str,
        target_agent: str,
        task_id: str,
        context: Dict[str, Any],
        instructions: str = "",
    ) -> str:
        """Initiate a handoff using TX backend."""
        handoff = self._manager.initiate_handoff(
            source_agent=source_agent,
            target_agent=target_agent,
            task_id=task_id,
            instructions=instructions,
            context=context,
        )
        return handoff.id

    def accept_handoff(
        self,
        handoff_id: str,
        agent: str,
        acknowledgment: str = "",
    ) -> bool:
        """Accept a handoff using TX backend."""
        try:
            self._manager.accept_handoff(handoff_id, agent, acknowledgment)
            return True
        except Exception:
            return False

    def complete_handoff(
        self,
        handoff_id: str,
        agent: str,
        result: Dict[str, Any],
        artifacts: Optional[List[str]] = None,
    ) -> bool:
        """Complete a handoff using TX backend."""
        try:
            self._manager.complete_handoff(
                handoff_id, agent, result, artifacts or []
            )
            return True
        except Exception:
            return False

    def reject_handoff(
        self,
        handoff_id: str,
        agent: str,
        reason: str = "",
    ) -> bool:
        """Reject a handoff using TX backend."""
        try:
            self._manager.reject_handoff(handoff_id, agent, reason)
            return True
        except Exception:
            return False

    def get_handoff(self, handoff_id: str) -> Optional[Dict[str, Any]]:
        """Get a handoff by ID using TX backend."""
        handoff = self._manager.get_handoff(handoff_id)
        if handoff is None:
            return None
        return {
            "id": handoff.id,
            "source_agent": handoff.source_agent,
            "target_agent": handoff.target_agent,
            "task_id": handoff.task_id,
            "status": handoff.status,
            "instructions": handoff.instructions,
            "context": handoff.context,
            "result": handoff.result,
            "artifacts": handoff.artifacts,
            "initiated_at": handoff.initiated_at,
            "accepted_at": handoff.accepted_at,
            "completed_at": handoff.completed_at,
            "rejected_at": handoff.rejected_at,
            "reject_reason": handoff.reject_reason,
        }

    def list_handoffs(
        self,
        status: Optional[str] = None,
        target_agent: Optional[str] = None,
        source_agent: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List handoffs using TX backend."""
        handoffs = self._manager.list_handoffs(
            status=status,
            target_agent=target_agent,
            source_agent=source_agent,
        )
        return [
            {
                "id": h.id,
                "source_agent": h.source_agent,
                "target_agent": h.target_agent,
                "task_id": h.task_id,
                "status": h.status,
                "instructions": h.instructions,
                "initiated_at": h.initiated_at,
            }
            for h in handoffs
        ]

    def save(self) -> None:
        """No-op for TX backend - transactions auto-commit."""
        pass  # TX backend auto-saves on transaction commit

    def get_sprint_tasks(self, sprint_id: str) -> List[ThoughtNode]:
        """Get all tasks in a sprint using TX backend."""
        tasks = self._manager.get_sprint_tasks(sprint_id)
        result = []
        for task in tasks:
            node = ThoughtNode(
                id=task.id,
                node_type=NodeType.TASK,
                content=task.title,
                properties={
                    "title": task.title,
                    "status": task.status,
                    "priority": task.priority,
                    "description": task.description,
                    **task.properties,
                },
                metadata={
                    "created_at": task.created_at,
                    "modified_at": task.modified_at,
                    **task.metadata,
                },
            )
            result.append(node)
        return result

    def get_sprint_progress(self, sprint_id: str) -> Dict[str, Any]:
        """Get sprint progress statistics using TX backend."""
        progress = self._manager.get_sprint_progress(sprint_id)
        # Normalize keys to match expected format
        return {
            "total_tasks": progress.get("total", 0),
            "by_status": {
                "completed": progress.get("completed", 0),
                "in_progress": progress.get("in_progress", 0),
                "pending": progress.get("pending", 0),
                "blocked": progress.get("blocked", 0),
            },
            "completed": progress.get("completed", 0),
            "progress_percent": progress.get("completion_rate", 0.0) * 100,
        }

    def get_next_task(self) -> Optional[Dict[str, Any]]:
        """Get the next task to work on.

        Selection criteria:
        1. Status must be 'pending' (not in_progress, completed, or blocked)
        2. Highest priority first (critical > high > medium > low)
        3. Oldest task within same priority

        Returns:
            Dict with 'id', 'title', 'priority', 'category' or None if no tasks available
        """
        # Get pending tasks
        pending_tasks = self.list_tasks(status=STATUS_PENDING)

        if not pending_tasks:
            return None

        # Sort by priority (critical > high > medium > low)
        priority_order = {
            PRIORITY_CRITICAL: 0,
            PRIORITY_HIGH: 1,
            PRIORITY_MEDIUM: 2,
            PRIORITY_LOW: 3,
        }

        def sort_key(task: ThoughtNode) -> Tuple[int, str]:
            priority = task.properties.get("priority", PRIORITY_MEDIUM)
            created_at = task.metadata.get("created_at", "")
            return (priority_order.get(priority, 99), created_at)

        sorted_tasks = sorted(pending_tasks, key=sort_key)

        if sorted_tasks:
            next_task = sorted_tasks[0]
            return {
                "id": next_task.id,
                "title": next_task.content,
                "priority": next_task.properties.get("priority", PRIORITY_MEDIUM),
                "category": next_task.properties.get("category", "general"),
            }

        return None

    def query(self, query_str: str) -> List[Dict[str, Any]]:
        """Query language for the graph.

        Supported queries:

        Relationship queries:
        - "what blocks <task_id>" - tasks blocking this task
        - "what depends on <task_id>" - tasks depending on this task
        - "relationships <task_id>" - all relationships for a task

        Status queries:
        - "blocked tasks" - tasks with blocked status
        - "active tasks" - tasks with in_progress status
        - "pending tasks" - tasks with pending status
        - "completed tasks" - tasks with completed status
        - "in_progress tasks" - tasks with in_progress status
        - "all tasks" - all tasks regardless of status

        Priority queries:
        - "high priority tasks" - tasks with high priority
        - "critical tasks" - tasks with critical priority

        Orphan queries:
        - "orphan tasks" / "orphan nodes" / "orphans" - tasks with no edges

        Sprint queries:
        - "tasks in sprint <sprint_id>" - tasks contained in a sprint
        - "current sprint" / "active sprint" - sprint with in_progress status
        - "sprints" / "all sprints" - all sprints

        Entity queries:
        - "decisions" / "all decisions" - all decisions

        Time-based queries:
        - "recent tasks" / "tasks today" - tasks created in last 24h
        - "stale tasks" - non-completed tasks not updated in 7+ days

        Returns:
            List of result dicts with id, title, and relevant fields.
        """
        # Preserve original for extracting IDs (case-sensitive)
        original_query = query_str.strip()
        query_str = original_query.lower()
        results = []

        if query_str.startswith("what blocks "):
            # Extract ID from original (case-sensitive)
            task_id = original_query[12:].strip()
            for node in self.what_blocks(task_id):
                results.append({
                    "id": node.id,
                    "title": node.content,
                    "status": node.properties.get("status"),
                    "relation": "blocks",
                })

        elif query_str.startswith("what depends on "):
            # Extract ID from original (case-sensitive)
            task_id = original_query[16:].strip()
            for node in self.what_depends_on(task_id):
                results.append({
                    "id": node.id,
                    "title": node.content,
                    "status": node.properties.get("status"),
                    "relation": "depends_on",
                })

        elif query_str == "blocked tasks":
            for node, reason in self.get_blocked_tasks():
                results.append({
                    "id": node.id,
                    "title": node.content,
                    "reason": reason,
                })

        elif query_str == "active tasks":
            for node in self.get_active_tasks():
                results.append({
                    "id": node.id,
                    "title": node.content,
                    "priority": node.properties.get("priority"),
                })

        elif query_str == "pending tasks":
            for node in self.list_tasks(status=STATUS_PENDING):
                results.append({
                    "id": node.id,
                    "title": node.content,
                    "priority": node.properties.get("priority"),
                })

        elif query_str in ("orphan tasks", "orphan nodes", "orphans"):
            for node in self.get_orphan_tasks():
                results.append({
                    "id": node.id,
                    "title": node.content,
                    "status": node.properties.get("status"),
                    "priority": node.properties.get("priority"),
                })

        elif query_str.startswith("relationships "):
            # Extract ID from original (case-sensitive)
            task_id = original_query[14:].strip()
            rels = self.get_all_relationships(task_id)
            for rel_type, nodes in rels.items():
                for node in nodes:
                    results.append({
                        "relation": rel_type,
                        "id": node.id,
                        "title": node.content,
                    })

        # Status-based queries
        elif query_str == "completed tasks":
            for node in self.list_tasks(status=STATUS_COMPLETED):
                results.append({
                    "id": node.id,
                    "title": node.content,
                    "status": "completed",
                    "priority": node.properties.get("priority"),
                })

        elif query_str in ("in_progress tasks", "in progress tasks"):
            for node in self.list_tasks(status=STATUS_IN_PROGRESS):
                results.append({
                    "id": node.id,
                    "title": node.content,
                    "status": "in_progress",
                    "priority": node.properties.get("priority"),
                })

        elif query_str == "all tasks":
            for node in self.list_all_tasks():
                results.append({
                    "id": node.id,
                    "title": node.content,
                    "status": node.properties.get("status"),
                    "priority": node.properties.get("priority"),
                })

        # Priority-based queries
        elif query_str == "high priority tasks":
            for node in self.list_all_tasks():
                if node.properties.get("priority") == "high":
                    results.append({
                        "id": node.id,
                        "title": node.content,
                        "status": node.properties.get("status"),
                        "priority": "high",
                    })

        elif query_str == "critical tasks":
            for node in self.list_all_tasks():
                if node.properties.get("priority") == "critical":
                    results.append({
                        "id": node.id,
                        "title": node.content,
                        "status": node.properties.get("status"),
                        "priority": "critical",
                    })

        # Sprint-based queries
        elif query_str.startswith("tasks in sprint "):
            # Extract ID from original (case-sensitive)
            sprint_id = original_query[16:].strip()
            # Find all tasks contained in this sprint via CONTAINS edges
            for edge in self._manager.list_edges():
                if edge.source_id == sprint_id and edge.edge_type == "CONTAINS":
                    task = self._manager.get_task(edge.target_id)
                    if task:
                        results.append({
                            "id": task.id,
                            "title": task.title,
                            "status": task.status,
                            "priority": task.priority,
                        })

        elif query_str in ("current sprint", "active sprint"):
            for sprint in self._manager.list_sprints():
                if sprint.status == "in_progress":
                    results.append({
                        "id": sprint.id,
                        "title": sprint.title,
                        "status": sprint.status,
                        "number": sprint.number,
                    })

        elif query_str in ("sprints", "all sprints"):
            for sprint in self._manager.list_sprints():
                results.append({
                    "id": sprint.id,
                    "title": sprint.title,
                    "status": sprint.status,
                    "number": sprint.number,
                })

        # Entity listing queries
        elif query_str in ("decisions", "all decisions"):
            for decision in self._manager.list_decisions():
                results.append({
                    "id": decision.id,
                    "title": decision.title,
                    "rationale": decision.rationale,
                })

        # Time-based queries
        elif query_str in ("recent tasks", "tasks today"):
            from datetime import datetime, timedelta
            cutoff = datetime.now() - timedelta(days=1)
            for node in self.list_all_tasks():
                created = node.metadata.get("created_at", "")
                if created:
                    try:
                        # Parse ISO format timestamp
                        created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                        if created_dt.replace(tzinfo=None) > cutoff:
                            results.append({
                                "id": node.id,
                                "title": node.content,
                                "status": node.properties.get("status"),
                                "created_at": created,
                            })
                    except (ValueError, TypeError):
                        pass

        elif query_str == "stale tasks":
            from datetime import datetime, timedelta
            cutoff = datetime.now() - timedelta(days=7)
            for node in self.list_all_tasks():
                # Check both updated_at and created_at
                updated = node.metadata.get("updated_at") or node.metadata.get("created_at", "")
                if updated:
                    try:
                        updated_dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                        if updated_dt.replace(tzinfo=None) < cutoff:
                            # Only include non-completed tasks
                            if node.properties.get("status") != "completed":
                                results.append({
                                    "id": node.id,
                                    "title": node.content,
                                    "status": node.properties.get("status"),
                                    "last_updated": updated,
                                })
                    except (ValueError, TypeError):
                        pass

        return results

    def sync_to_git(self) -> str:
        """Sync to git (no-op for TX backend, state is already persistent)."""
        return ""

    def infer_edges_from_commit(self, commit_message: str, files_changed: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Infer edges from a commit message.

        Parses commit messages for task references and creates edges:
        - "task:T-..." references → noted as IMPLEMENTS
        - "depends on task:T-..." → DEPENDS_ON edge
        - "blocks task:T-..." → BLOCKS edge
        - "closes task:T-..." → COMPLETES edge (marks task complete)

        Args:
            commit_message: The commit message to parse
            files_changed: Optional list of files changed in commit (for context)

        Returns:
            List of edges/actions performed
        """
        edges_created = []

        # Find all task references
        task_refs = re.findall(r'(?:task:)?(T-[\w-]+)', commit_message, re.IGNORECASE)

        # Find specific relationship patterns
        depends_pattern = re.findall(r'depends on (?:task:)?(T-[\w-]+)', commit_message, re.IGNORECASE)
        blocks_pattern = re.findall(r'blocks (?:task:)?(T-[\w-]+)', commit_message, re.IGNORECASE)
        closes_pattern = re.findall(r'(?:closes?|fixes?|resolves?) (?:task:)?(T-[\w-]+)', commit_message, re.IGNORECASE)

        # Get all known task IDs for matching
        all_tasks = {t.id.upper(): t.id for t in self.list_all_tasks()}

        # Track which tasks were referenced
        referenced_tasks = []
        for ref in task_refs:
            ref_upper = ref.upper()
            if ref_upper in all_tasks:
                referenced_tasks.append(all_tasks[ref_upper])
                edges_created.append({
                    "type": "REFERENCES",
                    "task": all_tasks[ref_upper],
                    "commit_message": commit_message[:50],
                })

        # Handle dependencies
        for dep_ref in depends_pattern:
            dep_upper = dep_ref.upper()
            if dep_upper in all_tasks and referenced_tasks:
                # First referenced task depends on this one
                first_task = referenced_tasks[0]
                target_task = all_tasks[dep_upper]
                if first_task != target_task:
                    self.add_dependency(first_task, target_task)
                    edges_created.append({
                        "type": "DEPENDS_ON",
                        "from": first_task,
                        "to": target_task,
                    })

        # Handle blocks
        for block_ref in blocks_pattern:
            block_upper = block_ref.upper()
            if block_upper in all_tasks and referenced_tasks:
                first_task = referenced_tasks[0]
                target_task = all_tasks[block_upper]
                if first_task != target_task:
                    self.add_blocks(first_task, target_task)
                    edges_created.append({
                        "type": "BLOCKS",
                        "from": first_task,
                        "to": target_task,
                    })

        # Handle closes/fixes (mark tasks complete)
        for close_ref in closes_pattern:
            close_upper = close_ref.upper()
            if close_upper in all_tasks:
                task_id = all_tasks[close_upper]
                self.complete_task(task_id, retrospective=f"Closed via commit: {commit_message[:50]}")
                edges_created.append({
                    "type": "CLOSES",
                    "task": task_id,
                })

        return edges_created

    def infer_edges_from_recent_commits(self, count: int = 10) -> List[Dict[str, Any]]:
        """Infer edges from recent git commits.

        Reads the last N commits and creates edges for any task references.

        Args:
            count: Number of recent commits to analyze

        Returns:
            List of all edges/actions created
        """
        import subprocess

        try:
            result = subprocess.run(
                ["git", "log", f"-{count}", "--pretty=format:%H|%s"],
                capture_output=True, text=True, check=True,
                cwd=str(self.got_dir.parent)  # Run from project root
            )
        except Exception as e:
            logger.warning(f"Failed to read git log: {e}")
            return []

        all_edges = []
        for line in result.stdout.strip().split("\n"):
            if "|" in line:
                commit_hash, message = line.split("|", 1)
                edges = self.infer_edges_from_commit(message)
                for edge in edges:
                    edge["commit_hash"] = commit_hash[:8]
                all_edges.extend(edges)

        return all_edges


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


# =============================================================================
# CLI COMMANDS
# =============================================================================

def cmd_task_create(args, manager: "TransactionalGoTAdapter") -> int:
    """Create a task."""
    task_id = manager.create_task(
        title=args.title,
        priority=getattr(args, 'priority', PRIORITY_MEDIUM),
        category=getattr(args, 'category', 'feature'),
        description=getattr(args, 'description', ''),
        sprint_id=getattr(args, 'sprint', None),
        depends_on=getattr(args, 'depends', None),
        blocks=getattr(args, 'blocks', None),
    )

    manager.save()
    print(f"Created: {task_id}")
    return 0


def cmd_task_list(args, manager: "TransactionalGoTAdapter") -> int:
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


def cmd_task_next(args, manager: "TransactionalGoTAdapter") -> int:
    """Get the next task to work on."""
    result = manager.get_next_task()

    if result is None:
        print("No pending tasks available.")
        return 0

    # Format output
    print(f"Next task: {result['id']}")
    print(f"  Title:    {result['title']}")
    print(f"  Priority: {result['priority']}")
    print(f"  Category: {result['category']}")

    # If --start flag, also start the task
    if getattr(args, 'start', False):
        task_id = result['id']
        if task_id.startswith("task:"):
            task_id = task_id[5:]
        success = manager.start_task(task_id)
        if success:
            print(f"\nStarted: {result['id']}")

    return 0


def cmd_task_show(args, manager: "TransactionalGoTAdapter") -> int:
    """Show details of a specific task."""
    task_id = args.task_id

    # Try to get task (with ID normalization)
    task = manager.get_task(task_id)

    # If not found, try with/without task: prefix
    if task is None:
        if task_id.startswith("task:"):
            task = manager.get_task(task_id[5:])
        else:
            task = manager.get_task(f"task:{task_id}")

    if task is None:
        print(f"Task not found: {task_id}")
        return 1

    # Display task details
    print("=" * 60)
    print(f"TASK: {task.id}")
    print("=" * 60)
    print(f"Title:    {task.content}")
    print(f"Status:   {task.properties.get('status', 'unknown')}")
    print(f"Priority: {task.properties.get('priority', 'unknown')}")
    print(f"Category: {task.properties.get('category', 'unknown')}")

    if task.properties.get('description'):
        print(f"\nDescription:\n  {task.properties['description']}")

    if task.properties.get('retrospective'):
        print(f"\nRetrospective:\n  {task.properties['retrospective']}")

    if task.properties.get('blocked_reason'):
        print(f"\nBlocked Reason:\n  {task.properties['blocked_reason']}")

    # Show timestamps
    print("\nTimestamps:")
    if task.metadata.get('created_at'):
        print(f"  Created:   {task.metadata['created_at']}")
    if task.metadata.get('updated_at'):
        print(f"  Updated:   {task.metadata['updated_at']}")
    if task.metadata.get('completed_at'):
        print(f"  Completed: {task.metadata['completed_at']}")

    # Show dependencies
    deps = manager.get_task_dependencies(task.id)
    if deps:
        print(f"\nDepends On ({len(deps)}):")
        for dep in deps:
            print(f"  - {dep.id}: {dep.content}")

    # Show what depends on this task
    dependents = manager.what_depends_on(task.id)
    if dependents:
        print(f"\nBlocks ({len(dependents)}):")
        for dep in dependents:
            print(f"  - {dep.id}: {dep.content}")

    print("=" * 60)
    return 0


def cmd_task_start(args, manager: "TransactionalGoTAdapter") -> int:
    """Start a task."""
    if manager.start_task(args.task_id):
        manager.save()
        print(f"Started: {args.task_id}")
        return 0
    else:
        print(f"Task not found: {args.task_id}")
        return 1


def cmd_task_complete(args, manager: "TransactionalGoTAdapter") -> int:
    """Complete a task."""
    if manager.complete_task(args.task_id, getattr(args, 'retrospective', None)):
        manager.save()
        print(f"Completed: {args.task_id}")
        return 0
    else:
        print(f"Task not found: {args.task_id}")
        return 1


def cmd_task_block(args, manager: "TransactionalGoTAdapter") -> int:
    """Block a task."""
    if manager.block_task(args.task_id, args.reason, getattr(args, 'blocker', None)):
        manager.save()
        print(f"Blocked: {args.task_id}")
        return 0
    else:
        print(f"Task not found: {args.task_id}")
        return 1


def cmd_task_depends(args, manager: "TransactionalGoTAdapter") -> int:
    """Create a dependency between tasks."""
    try:
        # Use add_dependency method
        if manager.add_dependency(args.task_id, args.depends_on_id):
            manager.save()
            print(f"Created dependency: {args.task_id} depends on {args.depends_on_id}")
            return 0
        else:
            print(f"Failed to create dependency - check that both task IDs exist")
            return 1
    except Exception as e:
        print(f"Error creating dependency: {e}")
        return 1


def cmd_task_delete(args, manager: "TransactionalGoTAdapter") -> int:
    """Delete a task with transactional safety checks.

    TRANSACTIONAL: Verifies pre-conditions before deletion.
    - Task must exist
    - Without --force: fails if task has dependents, blocks others, or is in progress
    - With --force: removes edges and deletes the task
    """
    task_id = args.task_id
    force = getattr(args, 'force', False)

    # Get task info before deletion for display
    task = manager.get_task(task_id)
    if not task:
        print(f"Task not found: {task_id}")
        return 1

    # Show what we're about to do
    task_title = task.content
    task_status = task.properties.get("status", "unknown")

    if not force:
        # Show warnings about what might block deletion
        dependents = manager.what_depends_on(task_id if task_id.startswith("task:") else f"task:{task_id}")
        if dependents:
            print(f"⚠️  Cannot delete: {len(dependents)} task(s) depend on this task:")
            for d in dependents[:5]:
                print(f"    - {d.id}: {d.content}")
            if len(dependents) > 5:
                print(f"    ... and {len(dependents) - 5} more")
            print("\nUse --force to delete anyway (will orphan dependent tasks)")
            return 1

        if task_status == STATUS_IN_PROGRESS:
            print(f"⚠️  Cannot delete: task is in progress")
            print("Use --force to delete anyway")
            return 1

    # Attempt deletion
    if manager.delete_task(task_id, force=force):
        manager.save()
        print(f"🗑️  Deleted: {task_id}")
        print(f"   Title: {task_title}")
        if force:
            print("   (forced deletion)")
        return 0
    else:
        print(f"Failed to delete: {task_id}")
        return 1


def cmd_sprint_create(args, manager: "TransactionalGoTAdapter") -> int:
    """Create a sprint."""
    sprint_id = manager.create_sprint(
        name=args.name,
        number=getattr(args, 'number', None),
        epic_id=getattr(args, 'epic', None),
    )

    manager.save()
    print(f"Created: {sprint_id}")
    return 0


def cmd_sprint_list(args, manager: "TransactionalGoTAdapter") -> int:
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
        claimed_by = sprint.properties.get("claimed_by", "")

        # Build status line
        status_line = f"{sprint.id}: {sprint.content} [{status}] - {progress['progress_percent']:.0f}% complete"
        if claimed_by:
            status_line += f" (claimed by {claimed_by})"

        print(status_line)

    return 0


def cmd_sprint_status(args, manager: "TransactionalGoTAdapter") -> int:
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


def cmd_sprint_start(args, manager: "TransactionalGoTAdapter") -> int:
    """Start a sprint."""
    sprint = manager.update_sprint(args.sprint_id, status="in_progress")
    manager.save()
    print(f"Started: {sprint.id}")
    print(f"  Title: {sprint.content}")
    return 0


def cmd_sprint_complete(args, manager: "TransactionalGoTAdapter") -> int:
    """Complete a sprint."""
    sprint = manager.update_sprint(args.sprint_id, status="completed")
    manager.save()
    print(f"Completed: {sprint.id}")
    print(f"  Title: {sprint.content}")
    return 0


def cmd_sprint_claim(args, manager: "TransactionalGoTAdapter") -> int:
    """Claim a sprint."""
    try:
        sprint = manager.claim_sprint(args.sprint_id, args.agent)
        manager.save()
        print(f"Claimed: {sprint.id}")
        print(f"  Agent: {args.agent}")
        return 0
    except ValueError as e:
        print(f"Error: {e}")
        return 1


def cmd_sprint_release(args, manager: "TransactionalGoTAdapter") -> int:
    """Release a sprint claim."""
    try:
        sprint = manager.release_sprint(args.sprint_id, args.agent)
        manager.save()
        print(f"Released: {sprint.id}")
        return 0
    except ValueError as e:
        print(f"Error: {e}")
        return 1


def cmd_sprint_goal_add(args, manager: "TransactionalGoTAdapter") -> int:
    """Add a goal to sprint."""
    if manager.add_sprint_goal(args.sprint_id, args.description):
        manager.save()
        print(f"Added goal to {args.sprint_id}: {args.description}")
        return 0
    else:
        print(f"Sprint not found: {args.sprint_id}")
        return 1


def cmd_sprint_goal_list(args, manager: "TransactionalGoTAdapter") -> int:
    """List sprint goals."""
    goals = manager.list_sprint_goals(args.sprint_id)
    if not goals:
        print(f"No goals for sprint {args.sprint_id}")
        return 0
    print(f"Goals for {args.sprint_id}:")
    for i, goal in enumerate(goals):
        status = "✓" if goal.get("completed") else " "
        print(f"  [{i}] [{status}] {goal.get('description', '')}")
    return 0


def cmd_sprint_goal_complete(args, manager: "TransactionalGoTAdapter") -> int:
    """Mark a goal as complete."""
    if manager.complete_sprint_goal(args.sprint_id, args.index):
        manager.save()
        print(f"Completed goal {args.index} in {args.sprint_id}")
        return 0
    else:
        print(f"Failed - check sprint ID and goal index")
        return 1


def cmd_sprint_link(args, manager: "TransactionalGoTAdapter") -> int:
    """Link a task to a sprint."""
    if manager.link_task_to_sprint(args.sprint_id, args.task_id):
        manager.save()
        print(f"Linked task {args.task_id} to sprint {args.sprint_id}")
        return 0
    else:
        print(f"Failed to link - check that both IDs exist")
        return 1


def cmd_sprint_unlink(args, manager: "TransactionalGoTAdapter") -> int:
    """Unlink a task from a sprint."""
    if manager.unlink_task_from_sprint(args.sprint_id, args.task_id):
        manager.save()
        print(f"Unlinked task {args.task_id} from sprint {args.sprint_id}")
        return 0
    else:
        print(f"No link found between {args.sprint_id} and {args.task_id}")
        return 1


def cmd_sprint_tasks(args, manager: "TransactionalGoTAdapter") -> int:
    """List tasks in a sprint."""
    tasks = manager.get_sprint_tasks(args.sprint_id)
    if not tasks:
        print(f"No tasks in sprint {args.sprint_id}")
        return 0
    print(f"Tasks in {args.sprint_id}:")
    for task in tasks:
        status = task.properties.get("status", "unknown")
        priority = task.properties.get("priority", "medium")
        print(f"  {task.id}: {task.content} [status={status}, priority={priority}]")
    return 0


def cmd_sprint_suggest(args, manager: "TransactionalGoTAdapter") -> int:
    """Suggest tasks for next sprint based on priority and dependencies."""
    try:
        # Get pending tasks
        if hasattr(manager, 'list_tasks'):
            pending_tasks = manager.list_tasks(status="pending")
        else:
            pending_tasks = [t for t in manager.tasks.values() if t.properties.get("status") == "pending"]

        if not pending_tasks:
            print("No pending tasks to suggest.")
            return 0

        # Priority scoring
        priority_scores = {"critical": 100, "high": 75, "medium": 50, "low": 25}

        # Score and sort tasks
        scored_tasks = []
        for task in pending_tasks:
            priority = task.properties.get("priority", "medium")
            score = priority_scores.get(priority, 50)

            # Check if blocked
            if hasattr(manager, 'what_blocks'):
                blockers = manager.what_blocks(task.id)
                if blockers:
                    score -= 30  # Penalty for blocked tasks

            scored_tasks.append((score, task))

        # Sort by score descending
        scored_tasks.sort(key=lambda x: -x[0])

        # Limit results
        limit = getattr(args, 'limit', 10)
        suggestions = scored_tasks[:limit]

        # Display suggestions
        print(f"\n{'='*60}")
        print(f"SPRINT SUGGESTIONS ({len(suggestions)} tasks)")
        print(f"{'='*60}\n")

        for i, (score, task) in enumerate(suggestions, 1):
            priority = task.properties.get("priority", "medium")
            category = task.properties.get("category", "feature")
            title = task.content[:50] + "..." if len(task.content) > 50 else task.content
            print(f"{i:2}. [{priority.upper():8}] {task.id}")
            print(f"    {title}")
            print(f"    Category: {category}, Score: {score}")
            print()

        return 0
    except Exception as e:
        print(f"Error generating suggestions: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_epic_create(args, manager: "TransactionalGoTAdapter") -> int:
    """Create an epic."""
    epic_id = manager.create_epic(
        name=args.name,
        epic_id=getattr(args, 'epic_id', None),
    )

    manager.save()
    print(f"Created: {epic_id}")
    return 0


def cmd_epic_list(args, manager: "TransactionalGoTAdapter") -> int:
    """List epics."""
    epics = manager.list_epics(
        status=getattr(args, 'status', None),
    )

    if not epics:
        print("No epics found.")
        return 0

    for epic in epics:
        status = epic.properties.get("status", "?")
        phase = epic.properties.get("phase", "?")
        print(f"{epic.id}: {epic.content} [{status}] - Phase: {phase}")

    return 0


def cmd_epic_show(args, manager: "TransactionalGoTAdapter") -> int:
    """Show epic details."""
    epic = manager.get_epic(args.epic_id)

    if not epic:
        print(f"Epic not found: {args.epic_id}")
        return 1

    print(f"Epic: {epic.id}")
    print(f"  Name: {epic.content}")
    print(f"  Status: {epic.properties.get('status', '?')}")
    print(f"  Phase: {epic.properties.get('phase', '?')}")

    # Show associated sprints
    sprints = manager.list_sprints(epic_id=epic.id)
    if sprints:
        print(f"  Sprints ({len(sprints)}):")
        for sprint in sprints:
            print(f"    - {sprint.id}: {sprint.content}")

    return 0


def cmd_blocked(args, manager: "TransactionalGoTAdapter") -> int:
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


def cmd_active(args, manager: "TransactionalGoTAdapter") -> int:
    """Show active tasks."""
    active = manager.get_active_tasks()
    print(format_task_table(active))
    return 0


def cmd_stats(args, manager: "TransactionalGoTAdapter") -> int:
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


def cmd_dashboard(args, manager: "TransactionalGoTAdapter") -> int:
    """Show comprehensive metrics dashboard."""
    # Import dashboard module
    try:
        from scripts.got_dashboard import render_dashboard
        dashboard = render_dashboard(manager)
        print(dashboard)
        return 0
    except ImportError as e:
        print(f"Error: Could not import dashboard module: {e}")
        return 1
    except Exception as e:
        print(f"Error rendering dashboard: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_export(args, manager: "TransactionalGoTAdapter") -> int:
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
# BACKUP COMMANDS
# =============================================================================


def cmd_backup_create(args, manager: "TransactionalGoTAdapter") -> int:
    """Create a backup snapshot."""
    compress = getattr(args, 'compress', True)

    try:
        snapshot_id = manager.wal.create_snapshot(manager.graph, compress=compress)
        print(f"Snapshot created: {snapshot_id}")

        # Show snapshot info
        snapshots_dir = manager.got_dir / "wal" / "snapshots"
        snapshot_files = list(snapshots_dir.glob(f"*{snapshot_id}*"))
        if snapshot_files:
            size = snapshot_files[0].stat().st_size
            print(f"  Size: {size / 1024:.1f} KB")
            print(f"  Compressed: {compress}")
        return 0
    except Exception as e:
        print(f"Error creating snapshot: {e}")
        return 1


def cmd_backup_list(args, manager: "TransactionalGoTAdapter") -> int:
    """List available snapshots."""
    limit = getattr(args, 'limit', 10)

    snapshots_dir = manager.got_dir / "wal" / "snapshots"
    if not snapshots_dir.exists():
        print("No snapshots found.")
        return 0

    # Find all snapshot files
    import gzip
    snapshots = []
    for snap_file in sorted(snapshots_dir.glob("snap_*.json*"), reverse=True):
        try:
            size = snap_file.stat().st_size
            is_compressed = snap_file.suffix == ".gz"

            # Extract timestamp from filename
            name = snap_file.stem
            if name.endswith(".json"):
                name = name[:-5]
            parts = name.split("_")
            if len(parts) >= 3:
                timestamp = f"{parts[1][:4]}-{parts[1][4:6]}-{parts[1][6:8]} "
                timestamp += f"{parts[2][:2]}:{parts[2][2:4]}:{parts[2][4:6]}"
            else:
                timestamp = "unknown"

            # Try to get node count
            node_count = "?"
            try:
                if is_compressed:
                    with gzip.open(snap_file, 'rt') as f:
                        data = json.load(f)
                else:
                    with open(snap_file) as f:
                        data = json.load(f)
                state = data.get("state", data)
                nodes = state.get("nodes", {})
                node_count = len(nodes)
            except Exception:
                pass

            snapshots.append({
                "file": snap_file.name,
                "timestamp": timestamp,
                "size": size,
                "compressed": is_compressed,
                "nodes": node_count
            })
        except Exception:
            continue

    if not snapshots:
        print("No snapshots found.")
        return 0

    print(f"Available Snapshots ({len(snapshots)} total):\n")
    print(f"{'Timestamp':<20} {'Nodes':<8} {'Size':<10} {'File'}")
    print("-" * 70)

    for snap in snapshots[:limit]:
        size_str = f"{snap['size'] / 1024:.1f} KB"
        print(f"{snap['timestamp']:<20} {str(snap['nodes']):<8} {size_str:<10} {snap['file']}")

    if len(snapshots) > limit:
        print(f"\n... and {len(snapshots) - limit} more")

    return 0


def cmd_backup_verify(args, manager: "TransactionalGoTAdapter") -> int:
    """Verify snapshot integrity."""
    snapshot_id = getattr(args, 'snapshot_id', None)

    snapshots_dir = manager.got_dir / "wal" / "snapshots"
    if not snapshots_dir.exists():
        print("No snapshots found.")
        return 1

    # Find the snapshot to verify
    import gzip
    if snapshot_id:
        files = list(snapshots_dir.glob(f"*{snapshot_id}*"))
    else:
        files = sorted(snapshots_dir.glob("snap_*.json*"), reverse=True)

    if not files:
        print(f"Snapshot not found: {snapshot_id or '(latest)'}")
        return 1

    snap_file = files[0]
    print(f"Verifying: {snap_file.name}")

    try:
        # Load and parse
        if snap_file.suffix == ".gz":
            with gzip.open(snap_file, 'rt') as f:
                data = json.load(f)
        else:
            with open(snap_file) as f:
                data = json.load(f)

        # Check required fields
        required = ["snapshot_id", "timestamp", "state"]
        missing = [r for r in required if r not in data]
        if missing:
            print(f"  ✗ Missing fields: {missing}")
            return 1

        # Check state structure
        state = data.get("state", {})
        nodes = state.get("nodes", {})
        edges = state.get("edges", {})

        print(f"  ✓ Valid JSON structure")
        print(f"  ✓ Snapshot ID: {data.get('snapshot_id', 'missing')}")
        print(f"  ✓ Timestamp: {data.get('timestamp', 'missing')}")
        print(f"  ✓ Nodes: {len(nodes)}")
        print(f"  ✓ Edges: {len(edges)}")

        # Verify node structure
        invalid_nodes = 0
        for node_id, node in nodes.items():
            if not isinstance(node, dict) or "node_type" not in node:
                invalid_nodes += 1
        if invalid_nodes:
            print(f"  ⚠ Invalid nodes: {invalid_nodes}")
        else:
            print(f"  ✓ All nodes valid")

        print("\nSnapshot verification: PASSED")
        return 0

    except json.JSONDecodeError as e:
        print(f"  ✗ Invalid JSON: {e}")
        return 1
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return 1


def cmd_backup_restore(args, manager: "TransactionalGoTAdapter") -> int:
    """Restore from a snapshot."""
    snapshot_id = args.snapshot_id
    force = getattr(args, 'force', False)

    snapshots_dir = manager.got_dir / "wal" / "snapshots"
    if not snapshots_dir.exists():
        print("No snapshots found.")
        return 1

    # Find the snapshot
    files = list(snapshots_dir.glob(f"*{snapshot_id}*"))
    if not files:
        print(f"Snapshot not found: {snapshot_id}")
        return 1

    snap_file = files[0]

    # Confirm unless forced
    if not force:
        print(f"About to restore from: {snap_file.name}")
        print("This will overwrite the current graph state.")
        response = input("Continue? [y/N]: ").strip().lower()
        if response != 'y':
            print("Restore cancelled.")
            return 0

    try:
        # Load snapshot
        import gzip
        if snap_file.suffix == ".gz":
            with gzip.open(snap_file, 'rt') as f:
                data = json.load(f)
        else:
            with open(snap_file) as f:
                data = json.load(f)

        state = data.get("state", {})
        nodes = state.get("nodes", {})

        # Rebuild graph
        manager.graph = ThoughtGraph()
        for node_id, node in nodes.items():
            manager.graph.add_node(
                node_id=node_id,
                node_type=NodeType[node.get("node_type", "TASK").upper()],
                content=node.get("content", ""),
                properties=node.get("properties", {}),
                metadata=node.get("metadata", {})
            )

        # Restore edges
        edges = state.get("edges", {})
        for edge_id, edge in edges.items():
            try:
                manager.graph.add_edge(
                    source_id=edge.get("source_id"),
                    target_id=edge.get("target_id"),
                    edge_type=EdgeType[edge.get("edge_type", "RELATES_TO").upper()],
                    weight=edge.get("weight", 1.0),
                    metadata=edge.get("metadata", {})
                )
            except Exception:
                pass  # Skip invalid edges

        # Save the restored state
        manager._save_state()

        print(f"Restored from: {snap_file.name}")
        print(f"  Nodes: {len(nodes)}")
        print(f"  Edges: {len(edges)}")
        return 0

    except Exception as e:
        print(f"Error restoring: {e}")
        return 1


def cmd_handoff_initiate(args, manager: "TransactionalGoTAdapter") -> int:
    """Initiate a handoff to another agent."""
    task = manager.get_task(args.task_id)
    if not task:
        print(f"Task not found: {args.task_id}")
        return 1

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
        instructions=args.instructions,
    )

    print(f"Handoff initiated: {handoff_id}")
    print(f"  Task: {task.content}")
    print(f"  From: {args.source} → To: {args.target}")
    if args.instructions:
        print(f"  Instructions: {args.instructions}")
    return 0


def cmd_handoff_accept(args, manager: "TransactionalGoTAdapter") -> int:
    """Accept a handoff."""
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


def cmd_handoff_complete(args, manager: "TransactionalGoTAdapter") -> int:
    """Complete a handoff."""
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


def cmd_handoff_list(args, manager: "TransactionalGoTAdapter") -> int:
    """List handoffs."""
    # Use manager's handoff method (works with TX backend)
    handoffs = manager.list_handoffs(status=args.status)

    if not handoffs:
        print("No handoffs found.")
        return 0

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


def cmd_decision_log(args, manager: "TransactionalGoTAdapter") -> int:
    """Log a decision with rationale."""
    context = {}
    if args.file:
        context["file"] = args.file

    decision_id = manager.log_decision(
        decision=args.decision,
        rationale=args.rationale,
        affects=args.affects,
        alternatives=args.alternatives,
        context=context if context else None,
    )

    print(f"Decision logged: {decision_id}")
    print(f"  Decision: {args.decision}")
    print(f"  Rationale: {args.rationale}")
    if args.affects:
        print(f"  Affects: {', '.join(args.affects)}")
    if args.alternatives:
        print(f"  Alternatives considered: {', '.join(args.alternatives)}")
    return 0


def cmd_decision_list(args, manager: "TransactionalGoTAdapter") -> int:
    """List all decisions."""
    decisions = manager.get_decisions()

    if not decisions:
        print("No decisions logged yet.")
        return 0

    print(f"Decisions ({len(decisions)}):\n")
    for d in decisions:
        print(f"  {d.id}")
        print(f"    Decision: {d.content}")
        print(f"    Rationale: {d.properties.get('rationale', 'N/A')}")
        if d.properties.get("alternatives"):
            print(f"    Alternatives: {', '.join(d.properties['alternatives'])}")
        print()

    return 0


def cmd_decision_why(args, manager: "TransactionalGoTAdapter") -> int:
    """Query why a task was created/modified."""
    reasons = manager.why(args.task_id)

    if not reasons:
        print(f"No decisions found affecting {args.task_id}")
        return 0

    print(f"Why {args.task_id}?\n")
    for r in reasons:
        print(f"  {r['decision_id']}")
        print(f"    Decision: {r['decision']}")
        print(f"    Rationale: {r['rationale']}")
        if r["alternatives"]:
            print(f"    Alternatives: {', '.join(r['alternatives'])}")
        print()

    return 0


def cmd_infer(args, manager: "TransactionalGoTAdapter") -> int:
    """Infer edges from git commits."""
    if args.message:
        # Analyze a specific message
        edges = manager.infer_edges_from_commit(args.message)
        print(f"Analyzing message: {args.message[:50]}...")
    else:
        # Analyze recent commits
        edges = manager.infer_edges_from_recent_commits(args.commits)
        print(f"Analyzed last {args.commits} commits")

    if not edges:
        print("\nNo task references found in commits.")
        return 0

    print(f"\nEdges inferred ({len(edges)}):\n")
    for edge in edges:
        if "commit_hash" in edge:
            print(f"  [{edge['commit_hash']}] {edge['type']}: {edge.get('from', edge.get('commit', ''))} → {edge.get('to', edge.get('task', ''))}")
        else:
            print(f"  {edge['type']}: {edge.get('from', '')} → {edge.get('to', '')}")

    return 0


def cmd_validate(args, manager: "TransactionalGoTAdapter") -> int:
    """Validate graph health and report issues."""
    print("=" * 60)
    print("GoT VALIDATION REPORT")
    print("=" * 60)

    issues = []
    warnings = []

    # Count nodes and edges from TX backend entities
    total_nodes = len(manager.graph.nodes)
    total_edges = len(manager.graph.edges)

    # Count tasks by status
    tasks = [n for n in manager.graph.nodes.values() if n.node_type == NodeType.TASK]
    task_count = len(tasks)

    # Count by status
    by_status = {}
    for task in tasks:
        status = task.properties.get("status", "unknown")
        by_status[status] = by_status.get(status, 0) + 1

    # Check for orphan nodes (no edges)
    # Only count edge references that point to existing nodes
    all_node_ids = set(manager.graph.nodes.keys())
    nodes_with_edges = set()
    for edge in manager.graph.edges:
        if edge.source_id in all_node_ids:
            nodes_with_edges.add(edge.source_id)
        if edge.target_id in all_node_ids:
            nodes_with_edges.add(edge.target_id)

    orphan_count = len(all_node_ids - nodes_with_edges)
    orphan_rate = orphan_count / max(total_nodes, 1) * 100

    # Check orphan rate (warning if high, but not critical)
    if orphan_rate > 50:
        warnings.append(f"High orphan rate: {orphan_rate:.1f}% of nodes have no edges")
    elif orphan_rate > 25:
        warnings.append(f"Moderate orphan rate: {orphan_rate:.1f}%")

    # Check edge density
    edge_density = total_edges / max(total_nodes, 1)
    if edge_density < 0.1 and total_nodes > 10:
        warnings.append(f"Low edge density: {edge_density:.2f} edges/node")

    # Count entity files for accurate statistics
    entities_dir = manager.got_dir / "entities"
    task_files = len(list(entities_dir.glob("T-*.json"))) if entities_dir.exists() else 0
    edge_files = len(list(entities_dir.glob("E-*.json"))) if entities_dir.exists() else 0
    decision_files = len(list(entities_dir.glob("D-*.json"))) if entities_dir.exists() else 0
    handoff_files = len(list(entities_dir.glob("H-*.json"))) if entities_dir.exists() else 0

    # Print stats
    print(f"\n📊 STATISTICS")
    print(f"   Tasks: {task_count}")
    print(f"   Edges: {total_edges}")
    print(f"   Edge density: {edge_density:.2f} edges/node")
    print(f"   Orphan nodes: {orphan_count} ({orphan_rate:.1f}%)")

    print(f"\n📁 ENTITY FILES")
    print(f"   Task files: {task_files}")
    print(f"   Edge files: {edge_files}")
    print(f"   Decision files: {decision_files}")
    print(f"   Handoff files: {handoff_files}")

    print(f"\n📈 TASKS BY STATUS")
    for status, count in sorted(by_status.items()):
        print(f"   {status}: {count}")

    # Print issues
    if issues:
        print(f"\n❌ ISSUES ({len(issues)})")
        for issue in issues:
            print(f"   • {issue}")

    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)})")
        for warning in warnings:
            print(f"   • {warning}")

    if not issues and not warnings:
        print(f"\n✅ HEALTHY - No issues detected")

    print()

    # Return non-zero if critical issues
    return 1 if issues else 0


def cmd_query(args, manager: "TransactionalGoTAdapter") -> int:
    """Run a query against the graph."""
    query_str = " ".join(args.query_string)

    print(f"Query: {query_str}\n")

    results = manager.query(query_str)

    if not results:
        print("No results found.")
        return 0

    print(f"Results ({len(results)}):\n")
    for r in results:
        if "step" in r:
            # Path query
            print(f"  [{r['step']}] {r['id']}: {r['title']}")
        elif "relation" in r:
            # Relationship query
            print(f"  {r['relation']}: {r['id']}")
            if r.get('title'):
                print(f"      {r['title']}")
        elif "reason" in r:
            # Blocked tasks
            print(f"  {r['id']}: {r['title']}")
            print(f"      Reason: {r['reason']}")
        else:
            # Generic result
            print(f"  {r['id']}: {r.get('title', '')}")
            if r.get('priority'):
                print(f"      Priority: {r['priority']}")
            if r.get('status'):
                print(f"      Status: {r['status']}")
        print()

    return 0


def cmd_sync(args, manager: "TransactionalGoTAdapter") -> int:
    """Sync GoT state to git-tracked snapshot.

    This is CRITICAL for environment resilience:
    - Ensures state survives fresh git clone
    - Enables cross-branch/cross-agent coordination
    - Should be run before committing
    """
    import subprocess

    try:
        # Sync to git-tracked location
        snapshot_name = manager.sync_to_git()
        print(f"Synced to git-tracked snapshot: {snapshot_name}")

        # Show stats
        stats = manager.get_stats()
        print(f"  Tasks: {stats['total_tasks']}")
        print(f"  Sprints: {stats['total_sprints']}")

        # Auto-commit if message provided
        message = getattr(args, 'message', None)
        if message:
            snapshot_path = manager.snapshots_dir / snapshot_name
            try:
                subprocess.run(
                    ["git", "add", str(snapshot_path)],
                    check=True, capture_output=True
                )
                subprocess.run(
                    ["git", "commit", "-m", f"got: {message}"],
                    check=True, capture_output=True
                )
                print(f"  Committed: got: {message}")
            except subprocess.CalledProcessError as e:
                print(f"  Warning: Git commit failed: {e}")

        print("\nTo persist across environments, commit .got/snapshots/")
        return 0

    except Exception as e:
        print(f"Error syncing: {e}")
        return 1


# =============================================================================
# COMMAND SUGGESTION HELPER
# =============================================================================

# All valid commands for suggestion
VALID_COMMANDS = [
    "task", "sprint", "epic", "handoff", "decision", "doc", "query",
    "blocked", "active", "stats", "dashboard", "validate", "infer",
    "export", "backup", "sync", "orphan", "backlog", "analyze", "edge",
]


def suggest_command(invalid_cmd: str, valid_commands: list = VALID_COMMANDS) -> list:
    """
    Suggest similar commands when user types an invalid one.

    Uses difflib to find close matches, making the CLI more user-friendly.

    Args:
        invalid_cmd: The invalid command the user typed
        valid_commands: List of valid commands to match against

    Returns:
        List of up to 3 similar command suggestions
    """
    import difflib
    matches = difflib.get_close_matches(
        invalid_cmd.lower(),
        valid_commands,
        n=3,
        cutoff=0.4  # Lower cutoff to catch more typos
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

    print(f"\nRun 'python scripts/got_utils.py --help' for available commands.", file=sys.stderr)


# =============================================================================
# MAIN (Thin Dispatcher)
# =============================================================================

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
    setup_analyze_parser(subparsers)  # Graph analysis using fluent Query API
    setup_edge_parser(subparsers)  # Direct edge management

    # Pre-check for invalid commands to provide better error messages
    # This runs before argparse's default error handling
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
            backend_type = "transactional" if isinstance(manager, TransactionalGoTAdapter) else "event-sourced"
            backend_dir = GOT_DIR if backend_type == "transactional" else GOT_DIR
            print(f"[DEBUG] Using {backend_type} backend at {backend_dir}", file=sys.stderr)
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

    # Query-related commands (query, blocked, active, stats, etc.)
    result = handle_query_commands(args, manager)
    if result is not None:
        return result

    # Sync and migrate commands
    result = handle_sync_migrate_commands(args, manager)
    if result is not None:
        return result

    # Fallback
    parser.print_help()
    return 1


def _run_with_auto_commit():
    """Run main() and trigger auto-commit on success."""
    # Parse args early to know the command
    import sys
    args_copy = sys.argv[1:]

    # Extract command and subcommand for auto-commit
    command = None
    subcommand = None
    for i, arg in enumerate(args_copy):
        if not arg.startswith('-'):
            if command is None:
                command = arg
            elif subcommand is None:
                subcommand = arg
                break

    # Run main
    result = main()

    # Trigger auto-commit on success
    if result == 0 and command:
        got_auto_commit(command, subcommand)

    # Cleanup auto-committer
    if _got_auto_committer is not None:
        _got_auto_committer.cleanup()

    return result


if __name__ == "__main__":
    sys.exit(_run_with_auto_commit())
