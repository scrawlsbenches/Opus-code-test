#!/usr/bin/env python3
"""
Graph of Thought Project Management CLI

Manages tasks, sprints, and epics using the Graph of Thought framework.
Replaces file-based task management with graph-native operations.

Usage:
    python scripts/got_utils.py task create "Fix bug" --priority high
    python scripts/got_utils.py task list --status pending
    python scripts/got_utils.py sprint status

See CLAUDE.md for complete command reference.
"""

import argparse
import json
import logging
import os
import re
import signal
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
from cortical.got.cli.task import (
    setup_task_parser,
    handle_task_command,
    # Re-export individual handlers for tests
    cmd_task_create,
    cmd_task_list,
    cmd_task_next,
    cmd_task_show,
    cmd_task_start,
    cmd_task_complete,
    cmd_task_block,
    cmd_task_update,
    cmd_task_depends,
    cmd_task_delete,
    cmd_task_import,
)
from cortical.got.cli.sprint import (
    setup_sprint_parser,
    setup_epic_parser,
    handle_sprint_command,
    handle_epic_command,
    # Re-export individual handlers for tests
    cmd_sprint_create,
    cmd_sprint_list,
    cmd_sprint_status,
    cmd_sprint_start,
    cmd_sprint_complete,
    cmd_sprint_claim,
    cmd_sprint_release,
    cmd_sprint_goal_add,
    cmd_sprint_goal_list,
    cmd_sprint_goal_complete,
    cmd_sprint_link,
    cmd_sprint_unlink,
    cmd_sprint_tasks,
    cmd_sprint_suggest,
    cmd_epic_create,
    cmd_epic_list,
    cmd_epic_show,
)
from cortical.got.cli.handoff import (
    setup_handoff_parser,
    handle_handoff_command,
    # Re-export individual handlers for tests
    cmd_handoff_initiate,
    cmd_handoff_accept,
    cmd_handoff_complete,
    cmd_handoff_reject,
    cmd_handoff_list,
)
from cortical.got.cli.decision import (
    setup_decision_parser,
    handle_decision_command,
    # Re-export individual handlers for tests
    cmd_decision_log,
    cmd_decision_list,
    cmd_decision_why,
)
from cortical.got.cli.query import (
    setup_query_parser,
    handle_query_commands,
    # Re-export individual handlers for tests
    cmd_query,
    cmd_expr,
    cmd_infer,
    cmd_blocked,
    cmd_active,
    cmd_stats,
    cmd_dashboard,
    cmd_validate,
    cmd_export,
)
from cortical.got.cli.backup import (
    setup_backup_parser,
    handle_backup_command,
    handle_sync_migrate_commands,
    # Re-export individual handlers for tests
    cmd_backup_create,
    cmd_backup_list,
    cmd_backup_verify,
    cmd_backup_restore,
    cmd_sync,
)
from cortical.got.cli.orphan import setup_orphan_parser, handle_orphan_command
from cortical.got.cli.backlog import setup_backlog_parser, handle_backlog_command
from cortical.got.cli.analyze import setup_analyze_parser, handle_analyze_command
from cortical.got.cli.edge import setup_edge_parser, handle_edge_command
from cortical.got.cli.batch import setup_batch_parser, handle_batch_command
from cortical.got.cli.knowledge_transfer import (
    setup_knowledge_transfer_parser,
    handle_knowledge_transfer_command,
)
from cortical.got.cli.failure import (
    setup_failure_parser,
    handle_failure_command,
)

# Import shared constants from canonical source (single source of truth)
from cortical.got.cli.shared import (
    STATUS_PENDING,
    STATUS_IN_PROGRESS,
    STATUS_COMPLETED,
    STATUS_BLOCKED,
    STATUS_DEFERRED,
    VALID_STATUSES,
    PRIORITY_CRITICAL,
    PRIORITY_HIGH,
    PRIORITY_MEDIUM,
    PRIORITY_LOW,
    VALID_PRIORITIES,
    VALID_CATEGORIES,
)

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

# NOTE: Status, Priority, and Category constants are imported from
# cortical.got.cli.shared (single source of truth). See imports above.

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
    "knowledge": {"create", "append", "link", "import", "finalize"},
    "kt": {"create", "append", "link", "import", "finalize"},
    "batch": True,  # Always mutating (creates multiple entities)
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


def _build_descriptive_commit_message(command: str, subcommand: Optional[str]) -> str:
    """
    Build a descriptive commit message by examining staged .got/ changes.

    Examines the staged entity files to extract entity IDs and titles
    for a more informative commit message.

    Args:
        command: Main command (e.g., "task", "sprint")
        subcommand: Subcommand (e.g., "create", "complete")

    Returns:
        Descriptive commit message
    """
    import subprocess

    try:
        # Get list of staged entity files
        result = subprocess.run(
            ['git', 'diff', '--cached', '--name-only', str(GOT_DIR / 'entities')],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode != 0 or not result.stdout.strip():
            # Fall back to generic message
            return _generic_commit_message(command, subcommand)

        changed_files = result.stdout.strip().split('\n')

        # Parse entity details from the first changed file
        for filepath in changed_files:
            full_path = PROJECT_ROOT / filepath
            if not full_path.exists():
                continue

            try:
                with open(full_path, 'r') as f:
                    wrapper = json.load(f)

                data = wrapper.get("data", {})
                entity_type = data.get("entity_type", "")
                entity_id = data.get("id", "")

                # Build message based on entity type and action
                if entity_type == "task":
                    title = data.get("title", "")
                    if subcommand == "create":
                        if title:
                            return f'chore(got): Create task "{title}" ({entity_id})'
                        return f"chore(got): Create task {entity_id}"
                    elif subcommand == "complete":
                        return f"chore(got): Complete task {entity_id}"
                    elif subcommand == "start":
                        return f"chore(got): Start task {entity_id}"
                    elif subcommand == "block":
                        return f"chore(got): Block task {entity_id}"
                    elif subcommand == "delete":
                        return f"chore(got): Delete task {entity_id}"

                elif entity_type == "decision":
                    title = data.get("title", "")
                    if title:
                        return f'chore(got): Log decision "{title}"'
                    return f"chore(got): Log decision {entity_id}"

                elif entity_type == "sprint":
                    title = data.get("title", "")
                    if subcommand == "create":
                        if title:
                            return f'chore(got): Create sprint "{title}" ({entity_id})'
                        return f"chore(got): Create sprint {entity_id}"
                    elif subcommand in ("start", "complete", "claim", "release"):
                        return f"chore(got): {subcommand.capitalize()} sprint {entity_id}"

                elif entity_type == "edge":
                    edge_type = data.get("edge_type", "")
                    source = data.get("source_id", "")
                    target = data.get("target_id", "")
                    if edge_type and source and target:
                        return f"chore(got): Add edge {edge_type} {source} -> {target}"

                elif entity_type == "handoff":
                    if subcommand == "initiate":
                        target = data.get("target_agent", "")
                        task_id = data.get("task_id", "")
                        return f"chore(got): Initiate handoff to {target} for {task_id}"
                    elif subcommand in ("accept", "complete"):
                        return f"chore(got): {subcommand.capitalize()} handoff {entity_id}"

            except (json.JSONDecodeError, KeyError, IOError):
                continue

    except (subprocess.TimeoutExpired, subprocess.SubprocessError):
        pass

    # Fall back to generic message
    return _generic_commit_message(command, subcommand)


def _generic_commit_message(command: str, subcommand: Optional[str]) -> str:
    """Build generic commit message when entity details unavailable."""
    if subcommand:
        return f"chore(got): Auto-save after {command} {subcommand}"
    return f"chore(got): Auto-save after {command}"


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

        # Build descriptive commit message from staged changes
        msg = _build_descriptive_commit_message(command, subcommand)

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
        # Use container to get properly configured GoTManager
        from cortical.core.bootstrap import create_container
        container = create_container(got_dir=self.got_dir)
        self._manager = container.resolve(TxGoTManager)

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

    def _tx_sprint_to_node(self, sprint) -> ThoughtNode:
        """Convert Sprint to ThoughtNode for compatibility."""
        return ThoughtNode(
            id=sprint.id,
            node_type=NodeType.CONTEXT,  # Sprints are contexts
            content=sprint.title,
            properties={
                "title": sprint.title,
                "status": sprint.status,
                "number": getattr(sprint, 'number', None),
                "epic_id": getattr(sprint, 'epic_id', None),
                "goals": getattr(sprint, 'goals', []),
                "notes": getattr(sprint, 'notes', ''),
                **getattr(sprint, 'properties', {}),
            },
            metadata={
                "created_at": getattr(sprint, 'created_at', ''),
                "modified_at": getattr(sprint, 'modified_at', ''),
            },
        )

    def _tx_decision_to_node(self, decision) -> ThoughtNode:
        """Convert Decision to ThoughtNode for compatibility."""
        return ThoughtNode(
            id=decision.id,
            node_type=NodeType.DECISION,
            content=decision.title,
            properties={
                "title": decision.title,
                "status": getattr(decision, 'status', 'accepted'),
                "rationale": getattr(decision, 'rationale', ''),
                **getattr(decision, 'properties', {}),
            },
            metadata={
                "created_at": getattr(decision, 'created_at', ''),
                "modified_at": getattr(decision, 'modified_at', ''),
            },
        )

    def _tx_epic_to_node(self, epic) -> ThoughtNode:
        """Convert Epic to ThoughtNode for compatibility."""
        return ThoughtNode(
            id=epic.id,
            node_type=NodeType.CONTEXT,  # Epics are contexts
            content=getattr(epic, 'title', epic.id),
            properties={
                "title": getattr(epic, 'title', epic.id),
                "status": getattr(epic, 'status', 'active'),
                "description": getattr(epic, 'description', ''),
                **getattr(epic, 'properties', {}),
            },
            metadata={
                "created_at": getattr(epic, 'created_at', ''),
                "modified_at": getattr(epic, 'modified_at', ''),
            },
        )

    def _tx_handoff_to_node(self, handoff) -> ThoughtNode:
        """Convert Handoff to ThoughtNode for compatibility."""
        return ThoughtNode(
            id=handoff.id,
            node_type=NodeType.TASK,  # Handoffs are task-like
            content=f"Handoff: {handoff.task_id}",
            properties={
                "source_agent": handoff.source_agent,
                "target_agent": handoff.target_agent,
                "task_id": handoff.task_id,
                "status": handoff.status,
                "instructions": getattr(handoff, 'instructions', ''),
            },
            metadata={
                "initiated_at": getattr(handoff, 'initiated_at', ''),
                "completed_at": getattr(handoff, 'completed_at', ''),
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
        # If sprint_id specified, get tasks from that sprint first
        if sprint_id:
            sprint_task_ids = set()
            try:
                # Get all edges and filter to CONTAINS from this sprint
                all_edges = self._manager.list_edges()
                for edge in all_edges:
                    if (edge.source_id == sprint_id and
                        edge.edge_type == "CONTAINS" and
                        edge.target_id.startswith("T-")):
                        sprint_task_ids.add(edge.target_id)
            except Exception as e:
                logger.debug(f"Could not get sprint tasks: {e}")

            if not sprint_task_ids:
                return []

            # Get all tasks and filter to sprint members
            all_tasks = self._manager.find_tasks(status=status, priority=priority)
            tasks = [t for t in all_tasks if t.id in sprint_task_ids]
        else:
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

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        weight: float = 1.0,
        reason: str = "",
        validate_refs: bool = True,
    ):
        """Add a generic edge between two entities.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            edge_type: Type of edge (e.g., DEPENDS_ON, BLOCKS, CAUSED_BY)
            weight: Edge weight (default: 1.0)
            reason: Why this relationship exists (context capture)
            validate_refs: Whether to validate that entities exist (default: True)

        Returns:
            Edge object if successful, None otherwise
        """
        clean_source = self._strip_prefix(source_id)
        clean_target = self._strip_prefix(target_id)
        try:
            edge = self._manager.add_edge(
                clean_source, clean_target, edge_type, weight=weight, reason=reason,
                validate_refs=validate_refs
            )
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

    def get_task_sprint(self, task_id: str) -> Optional[Dict[str, str]]:
        """Get the sprint that contains this task.

        Args:
            task_id: Task ID to find sprint for

        Returns:
            Dict with 'id' and 'name' keys, or None if not in a sprint
        """
        try:
            _, incoming = self.get_edges_for_task(task_id)
            for edge in incoming:
                # CONTAINS edges from sprints to tasks
                if edge.edge_type == "CONTAINS" and edge.source_id.startswith("S-"):
                    sprint = self.get_sprint(edge.source_id)
                    if sprint:
                        # sprint is a ThoughtNode, access attributes directly
                        return {
                            'id': sprint.id,
                            'name': sprint.content or 'Unknown'
                        }
            return None
        except Exception as e:
            logger.debug(f"Could not find sprint for {task_id}: {e}")
            return None

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

    def _get_entity_node(self, entity_id: str) -> Optional[ThoughtNode]:
        """Get any entity (task, sprint, decision, epic, handoff) as a ThoughtNode.

        Args:
            entity_id: ID of the entity (T-..., S-..., D-..., EPIC-..., H-...)

        Returns:
            ThoughtNode or None if not found
        """
        # Try task first (most common)
        if entity_id.startswith("T-"):
            task = self._manager.get_task(entity_id)
            if task:
                return self._tx_task_to_node(task)

        # Try sprint
        if entity_id.startswith("S-"):
            sprint = self._manager.get_sprint(entity_id)
            if sprint:
                return self._tx_sprint_to_node(sprint)

        # Try decision
        if entity_id.startswith("D-"):
            decision = self._manager.get_decision(entity_id)
            if decision:
                return self._tx_decision_to_node(decision)

        # Try epic
        if entity_id.startswith("EPIC-"):
            epic = self._manager.get_epic(entity_id)
            if epic:
                return self._tx_epic_to_node(epic)

        # Try handoff
        if entity_id.startswith("H-"):
            handoff = self._manager.get_handoff(entity_id)
            if handoff:
                return self._tx_handoff_to_node(handoff)

        # Fallback: try all types
        for getter, converter in [
            (self._manager.get_task, self._tx_task_to_node),
            (self._manager.get_sprint, self._tx_sprint_to_node),
            (self._manager.get_decision, self._tx_decision_to_node),
            (self._manager.get_epic, self._tx_epic_to_node),
            (self._manager.get_handoff, self._tx_handoff_to_node),
        ]:
            try:
                entity = getter(entity_id)
                if entity:
                    return converter(entity)
            except Exception:
                continue

        return None

    def get_all_relationships(self, entity_id: str) -> Dict[str, List[ThoughtNode]]:
        """Get all relationships for any entity (task, sprint, epic, decision, handoff).

        Returns dict with keys for each edge type found:
        - Outgoing edges: lowercase edge type (e.g., 'blocks', 'depends_on', 'transfers')
        - Incoming edges: edge type + '_by' (e.g., 'blocked_by', 'depended_by', 'transferred_by')

        Common keys (for backward compatibility):
        - 'blocks' / 'blocked_by': BLOCKS edges
        - 'depends_on' / 'depended_by': DEPENDS_ON edges
        - 'contains' / 'contained_by': CONTAINS edges

        All other edge types are handled dynamically.
        """
        clean_id = self._strip_prefix(entity_id)

        # Initialize with common keys for backward compatibility
        result: Dict[str, List[ThoughtNode]] = {
            'blocks': [],
            'blocked_by': [],
            'depends_on': [],
            'depended_by': [],
            'contains': [],
            'contained_by': [],
        }

        def get_outgoing_key(edge_type: str) -> str:
            """Convert edge type to outgoing relationship key."""
            return edge_type.lower()

        def get_incoming_key(edge_type: str) -> str:
            """Convert edge type to incoming relationship key.

            Special cases for grammatically correct names:
            - BLOCKS -> blocked_by
            - CONTAINS -> contained_by
            - TRANSFERS -> transferred_by
            Others just get _by suffix.
            """
            et = edge_type.lower()
            # Handle special grammatical cases
            if et == 'blocks':
                return 'blocked_by'
            elif et == 'contains':
                return 'contained_by'
            elif et == 'transfers':
                return 'transferred_by'
            elif et == 'triggers':
                return 'triggered_by'
            elif et == 'enables':
                return 'enabled_by'
            elif et == 'requires':
                return 'required_by'
            elif et == 'supports':
                return 'supported_by'
            elif et == 'refutes':
                return 'refuted_by'
            elif et == 'precedes':
                return 'preceded_by'
            elif et == 'answers':
                return 'answered_by'
            elif et == 'raises':
                return 'raised_by'
            elif et == 'explores':
                return 'explored_by'
            elif et == 'observes':
                return 'observed_by'
            elif et == 'suggests':
                return 'suggested_by'
            elif et == 'implements':
                return 'implemented_by'
            elif et == 'tests':
                return 'tested_by'
            elif et == 'refines':
                return 'refined_by'
            elif et == 'motivates':
                return 'motivated_by'
            elif et == 'justifies':
                return 'justified_by'
            else:
                # Default: just add _by
                return f"{et}_by"

        try:
            # Get edges for this entity (API method works for any entity ID)
            outgoing, incoming = self._manager.get_edges_for_task(clean_id)

            # Process outgoing edges - dynamically handle all edge types
            for edge in outgoing:
                target_node = self._get_entity_node(edge.target_id)
                if target_node:
                    key = get_outgoing_key(edge.edge_type)
                    if key not in result:
                        result[key] = []
                    result[key].append(target_node)

            # Process incoming edges - dynamically handle all edge types
            for edge in incoming:
                source_node = self._get_entity_node(edge.source_id)
                if source_node:
                    key = get_incoming_key(edge.edge_type)
                    if key not in result:
                        result[key] = []
                    result[key].append(source_node)

        except Exception as e:
            logger.error(f"Failed to get relationships for {entity_id}: {e}")

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

    def get_decision(self, decision_id: str) -> Optional[ThoughtNode]:
        """Get a single decision by ID from TX backend."""
        from cortical.got.types import Decision
        entities_dir = self.got_dir / "entities"
        decision_file = entities_dir / f"{decision_id}.json"

        if not decision_file.exists():
            return None

        try:
            with open(decision_file, 'r') as f:
                wrapper = json.load(f)
            data = wrapper.get("data", wrapper)
            if data.get("entity_type") == "decision":
                decision = Decision.from_dict(data)
                return ThoughtNode(
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
        except Exception:
            return None
        return None

    def delete_decision(self, decision_id: str, force: bool = False) -> None:
        """Delete a decision and its connected edges via TX backend."""
        self._manager.delete_decision(decision_id, force=force)
        # Invalidate cached graph
        self._graph = None

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
        description: Optional[str] = None,
    ) -> str:
        """Create a new sprint using TX backend.

        Args:
            name: Sprint name/title
            number: Optional sprint number (display metadata)
            epic_id: Optional epic ID this sprint belongs to
            description: Optional description/notes explaining the sprint context

        Returns:
            Created sprint ID
        """
        # Build notes list from description if provided
        notes = [description] if description else []

        sprint = self._manager.create_sprint(
            title=name,
            number=number,
            epic_id=epic_id or "",
            notes=notes,
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

    def delete_sprint(self, sprint_id: str, force: bool = False) -> None:
        """Delete a sprint and all its connected edges."""
        self._manager.delete_sprint(sprint_id, force=force)

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

    def create_epic(
        self,
        name: str,
        epic_id: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new epic using TX backend."""
        epic = self._manager.create_epic(
            title=name,
            epic_id=epic_id,
            properties=properties or {}
        )
        return epic.id

    def get_epic(self, epic_id: str) -> Optional[ThoughtNode]:
        """Get an epic by ID."""
        epic = self._manager.get_epic(epic_id)
        if epic is None:
            return None
        # Merge base properties with epic's custom properties
        props = {
            "name": epic.title,
            "status": epic.status,
            "phase": epic.phase,
            "phases": epic.phases,
        }
        props.update(epic.properties)  # Include description and other custom properties
        return ThoughtNode(
            id=epic.id,
            node_type=NodeType.GOAL,
            content=epic.title,
            properties=props,
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
        """Query the graph using natural language patterns.

        Translates natural language to expression DSL, then executes.
        See get_supported_patterns() for available patterns.
        """
        from cortical.got.expression.translator import translate
        from cortical.got.expression import parse, execute

        # Translate natural language to DSL expression
        expr_str = translate(query_str)

        # Handle empty expression (e.g., "all tasks" returns everything)
        if not expr_str:
            # Return all tasks
            return [self._task_to_dict(t) for t in self.list_all_tasks()]

        # Parse and execute
        try:
            ast = parse(expr_str)
            results = execute(self._manager, ast)
            return [self._entity_to_dict(query_str, e) for e in results]
        except Exception:
            # Unknown pattern - return empty list
            return []

    def _task_to_dict(self, task) -> Dict[str, Any]:
        """Convert a task to dict format."""
        return {
            "id": task.id,
            "title": task.content,
            "status": task.properties.get("status"),
            "priority": task.properties.get("priority"),
        }

    def _entity_to_dict(self, query_str: str, entity) -> Dict[str, Any]:
        """Convert entity to dict format for query results."""
        query_lower = query_str.lower()

        # Build base dict
        result = {
            "id": entity.id,
            "title": getattr(entity, 'title', getattr(entity, 'content', '')),
        }

        # Add context-specific fields based on query
        if hasattr(entity, 'status'):
            result["status"] = entity.status
        if hasattr(entity, 'priority'):
            result["priority"] = entity.priority
        if hasattr(entity, 'properties'):
            if "priority" in entity.properties:
                result["priority"] = entity.properties["priority"]
            if "status" in entity.properties:
                result["status"] = entity.properties["status"]
            # Include blocked_reason as "reason" for blocked task queries
            if "blocked_reason" in entity.properties:
                result["reason"] = entity.properties["blocked_reason"]

        # Add relation field for relationship queries
        if "blocks" in query_lower or "depends" in query_lower:
            result["relation"] = "blocks" if "blocks" in query_lower else "depends_on"

        return result

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

    # =========================================================================
    # KNOWLEDGE TRANSFER METHODS
    # =========================================================================

    def create_knowledge_transfer(
        self,
        title: str,
        session_id: str = "",
        session_date: str = "",
        summary: str = "",
        sections: Optional[Dict[str, str]] = None,
        code_refs: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        status: str = "draft",  # Draft by default - finalize to publish
        source_file: Optional[str] = None,
    ) -> str:
        """
        Create a knowledge transfer document.

        Args:
            title: Document title
            session_id: Session identifier
            session_date: Session date (ISO 8601)
            summary: Executive summary
            sections: Dictionary mapping section headings to content
            code_refs: List of file:line references
            tags: Classification tags
            status: Publication status (draft, published, archived)
            source_file: Original source file path

        Returns:
            Knowledge transfer entity ID
        """
        # Build properties dict for extra fields
        properties = {}
        if source_file:
            properties["source_file"] = source_file

        # Delegate to TransactionManager (proper transactional storage with checksums)
        kt = self._manager.tx_manager.create_knowledge_transfer(
            title=title,
            summary=summary,
            session_id=session_id,
            session_date=session_date,
            sections=sections,
            code_refs=code_refs,
            tags=tags,
            properties=properties,
        )

        # Update status if not the default (TransactionManager creates with status="published")
        if status != "published":
            tx = self._manager.tx_manager.begin()
            try:
                kt.status = status
                if source_file:
                    kt.source_file = source_file
                self._manager.tx_manager.write(tx, kt)
                self._manager.tx_manager.commit(tx)
            except Exception:
                self._manager.tx_manager.rollback(tx, reason="update_kt_status_failed")
                raise

        self.save()
        return kt.id

    def append_kt_section(
        self,
        kt_id: str,
        section_heading: str,
        content: str,
    ) -> bool:
        """
        Append or update a section in a knowledge transfer document.

        Args:
            kt_id: Knowledge transfer entity ID
            section_heading: Section heading
            content: Section content

        Returns:
            True if successful, False otherwise
        """
        kt = self.get_knowledge_transfer(kt_id)
        if not kt:
            return False

        # Update sections
        sections = kt.get('sections', {})
        sections[section_heading] = content

        # Update entity
        return self._update_kt_entity(kt_id, {'sections': sections})

    def link_kt_handoff(self, kt_id: str, handoff_id: str) -> bool:
        """Link a knowledge transfer to a handoff entity."""
        kt = self.get_knowledge_transfer(kt_id)
        if not kt:
            return False

        related_handoffs = kt.get('related_handoffs', [])
        if handoff_id not in related_handoffs:
            related_handoffs.append(handoff_id)

        return self._update_kt_entity(kt_id, {'related_handoffs': related_handoffs})

    def link_kt_task(self, kt_id: str, task_id: str) -> bool:
        """Link a knowledge transfer to a task entity."""
        kt = self.get_knowledge_transfer(kt_id)
        if not kt:
            return False

        related_tasks = kt.get('related_tasks', [])
        if task_id not in related_tasks:
            related_tasks.append(task_id)

        return self._update_kt_entity(kt_id, {'related_tasks': related_tasks})

    def link_kt_decision(self, kt_id: str, decision_id: str) -> bool:
        """Link a knowledge transfer to a decision entity."""
        kt = self.get_knowledge_transfer(kt_id)
        if not kt:
            return False

        related_decisions = kt.get('related_decisions', [])
        if decision_id not in related_decisions:
            related_decisions.append(decision_id)

        return self._update_kt_entity(kt_id, {'related_decisions': related_decisions})

    def list_knowledge_transfers(
        self,
        status: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        List knowledge transfer documents with optional filters.

        Args:
            status: Filter by status (draft, published, archived)
            tags: Filter by tags (matches if any tag matches)

        Returns:
            List of knowledge transfer dictionaries
        """
        entities_dir = self.got_dir / "entities"
        if not entities_dir.exists():
            return []

        kts = []
        for kt_file in entities_dir.glob("KT-*.json"):
            try:
                with open(kt_file, 'r') as f:
                    wrapper = json.load(f)
                data = wrapper.get("data", wrapper)

                if data.get("entity_type") == "knowledge_transfer":
                    # Apply filters
                    if status and data.get("status") != status:
                        continue

                    if tags:
                        kt_tags = data.get("tags", [])
                        if not any(tag in kt_tags for tag in tags):
                            continue

                    kts.append(data)
            except Exception as e:
                logger.debug(f"Skipping KT file {kt_file}: {e}")

        # Sort by creation date (newest first)
        kts.sort(key=lambda x: x.get('created_at', ''), reverse=True)

        return kts

    def get_knowledge_transfer(self, kt_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a knowledge transfer document by ID.

        Args:
            kt_id: Knowledge transfer entity ID

        Returns:
            Knowledge transfer dictionary or None if not found
        """
        entities_dir = self.got_dir / "entities"
        kt_file = entities_dir / f"{kt_id}.json"

        if not kt_file.exists():
            return None

        try:
            with open(kt_file, 'r') as f:
                wrapper = json.load(f)
            data = wrapper.get("data", wrapper)

            if data.get("entity_type") == "knowledge_transfer":
                return data
        except Exception as e:
            logger.error(f"Failed to load KT {kt_id}: {e}")

        return None

    def _update_kt_entity(self, kt_id: str, updates: Dict[str, Any]) -> bool:
        """
        Internal method to update a knowledge transfer entity.

        Uses TransactionManager for proper transactional storage with checksums.

        Args:
            kt_id: Knowledge transfer entity ID
            updates: Dictionary of fields to update

        Returns:
            True if successful, False otherwise
        """
        tx = self._manager.tx_manager.begin()
        try:
            # Read current entity via transaction
            entity = self._manager.tx_manager.read(tx, kt_id)
            if entity is None:
                self._manager.tx_manager.rollback(tx, reason="kt_not_found")
                return False

            # Apply updates to entity attributes
            for key, value in updates.items():
                if hasattr(entity, key):
                    setattr(entity, key, value)
                elif hasattr(entity, 'properties') and isinstance(entity.properties, dict):
                    entity.properties[key] = value

            # Bump version and write
            entity.bump_version()
            self._manager.tx_manager.write(tx, entity)
            result = self._manager.tx_manager.commit(tx)

            if not result.success:
                logger.error(f"Failed to update KT {kt_id}: {result.reason}")
                return False

            self.save()
            return True
        except Exception as e:
            self._manager.tx_manager.rollback(tx, reason="update_kt_failed")
            logger.error(f"Failed to update KT {kt_id}: {e}")
            return False

    def finalize_knowledge_transfer(
        self,
        kt_id: str,
        handoff_to: Optional[str] = None,
        instructions: str = "",
    ) -> bool:
        """
        Finalize a knowledge transfer and optionally create handoff for continuation.

        Args:
            kt_id: Knowledge transfer entity ID
            handoff_to: Optional agent to hand off continuation work to
            instructions: Instructions for continuation handoff

        Returns:
            True if successful, False otherwise
        """
        from datetime import datetime

        # Get the KT
        kt = self.get_knowledge_transfer(kt_id)
        if not kt:
            logger.error(f"Knowledge transfer not found: {kt_id}")
            return False

        # Verify it's in draft status (or already published = idempotent success)
        if kt.get('status') != 'draft':
            if kt.get('status') == 'published':
                logger.info(f"Knowledge transfer {kt_id} is already published")
                return True  # Idempotent - already in desired state
            logger.error(f"Knowledge transfer {kt_id} cannot be finalized (current status: {kt.get('status')})")
            return False

        # Change status to published
        if not self._update_kt_entity(kt_id, {'status': 'published'}):
            return False

        # If handoff requested, create it
        if handoff_to:
            try:
                # Create a handoff entity for continuation
                handoff_id = self.initiate_handoff(
                    source_agent="main",
                    target_agent=handoff_to,
                    task_id=kt_id,  # Link to KT instead of task
                    context={
                        "kt_title": kt.get('title', ''),
                        "kt_summary": kt.get('summary', ''),
                        "session_id": kt.get('session_id', ''),
                        "type": "knowledge_transfer_continuation",
                    },
                    instructions=instructions,
                )

                # Add CONTINUES edge from KT to Handoff
                edge = self.add_edge(kt_id, handoff_id, "CONTINUES", weight=1.0)
                if edge is None:
                    logger.warning(f"Failed to create CONTINUES edge from {kt_id} to {handoff_id}")

                logger.info(f"Created continuation handoff {handoff_id} for KT {kt_id}")

            except Exception as e:
                logger.error(f"Failed to create handoff for KT {kt_id}: {e}")
                return False

        return True

    def get_kt_history(self, kt_id: str) -> List[tuple]:
        """
        Get the history chain for a knowledge transfer.

        Traces CONTINUES edges to show evolution:
        KT1 → Handoff1 → KT2 → Handoff2 → KT3 (current)

        Args:
            kt_id: Knowledge transfer entity ID

        Returns:
            List of (entity_type, entity_id, title) tuples representing chain
        """
        # Get all edges using the query API
        try:
            all_edges = self._manager.query_api.list_edges()
        except Exception as e:
            logger.error(f"Failed to list edges: {e}")
            return []

        # Build a mapping of CONTINUES edges
        continues_edges = {}  # source_id -> target_id
        reverse_continues = {}  # target_id -> source_id

        for edge in all_edges:
            if edge.edge_type.upper() == "CONTINUES":
                continues_edges[edge.source_id] = edge.target_id
                reverse_continues[edge.target_id] = edge.source_id

        # Walk backward to find origin
        current_id = kt_id
        while current_id in reverse_continues:
            current_id = reverse_continues[current_id]

        # Now walk forward to build full chain
        chain = []
        visited = set()  # Prevent infinite loops

        while current_id and current_id not in visited:
            visited.add(current_id)

            # Determine entity type and get title
            entity_type = self._infer_entity_type(current_id)
            title = self._get_entity_title(current_id, entity_type)

            chain.append((entity_type, current_id, title))

            # Move to next in chain
            current_id = continues_edges.get(current_id)

        return chain

    def _infer_entity_type(self, entity_id: str) -> str:
        """Infer entity type from ID prefix."""
        if entity_id.startswith("KT-"):
            return "knowledge_transfer"
        elif entity_id.startswith("H-"):
            return "handoff"
        elif entity_id.startswith("T-"):
            return "task"
        elif entity_id.startswith("D-"):
            return "decision"
        elif entity_id.startswith("S-"):
            return "sprint"
        elif entity_id.startswith("E-"):
            if "-" in entity_id[2:]:  # E-xxx-yyy format
                return "edge"
            return "epic"
        else:
            return "unknown"

    def _get_entity_title(self, entity_id: str, entity_type: str) -> str:
        """Get title/name for an entity."""
        try:
            if entity_type == "knowledge_transfer":
                kt = self.get_knowledge_transfer(entity_id)
                return kt.get('title', 'Untitled') if kt else '?'
            elif entity_type == "handoff":
                handoff = self.get_handoff(entity_id)
                return f"{handoff.get('source_agent', '?')} → {handoff.get('target_agent', '?')}" if handoff else '?'
            elif entity_type == "task":
                task = self.get_task(entity_id)
                return task.content if task else '?'
            elif entity_type == "decision":
                # Read decision file
                entities_dir = self.got_dir / "entities"
                decision_file = entities_dir / f"{entity_id}.json"
                if decision_file.exists():
                    with open(decision_file, 'r') as f:
                        wrapper = json.load(f)
                    data = wrapper.get("data", wrapper)
                    return data.get('title', '?')
            return '?'
        except Exception as e:
            logger.debug(f"Failed to get title for {entity_id}: {e}")
            return '?'


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
# NOTE: All cmd_* command handlers are now imported from cortical/got/cli/ modules.
# See imports at top of file. This eliminates ~1100 lines of duplicate code.
#
# Handlers imported from:
#   - cortical/got/cli/task.py: cmd_task_*
#   - cortical/got/cli/sprint.py: cmd_sprint_*, cmd_epic_*
#   - cortical/got/cli/handoff.py: cmd_handoff_*
#   - cortical/got/cli/decision.py: cmd_decision_*
#   - cortical/got/cli/query.py: cmd_query, cmd_infer, cmd_blocked, cmd_active,
#                                 cmd_stats, cmd_dashboard, cmd_validate, cmd_export
#   - cortical/got/cli/backup.py: cmd_backup_*, cmd_sync
# =============================================================================


# COMMAND SUGGESTION HELPER
# =============================================================================

# All valid commands for suggestion
VALID_COMMANDS = [
    "task", "sprint", "epic", "handoff", "decision", "doc", "query", "expr",
    "blocked", "active", "stats", "dashboard", "validate", "infer",
    "export", "backup", "sync", "orphan", "backlog", "analyze", "edge",
    "batch", "knowledge", "kt", "failure",
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
    setup_batch_parser(subparsers)  # Batch operations with heredoc DSL
    setup_knowledge_transfer_parser(subparsers)  # Knowledge transfer documents
    setup_failure_parser(subparsers)  # Failure tracking and lesson learning

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
    # Handle SIGPIPE gracefully (e.g., when piping to `head`)
    # This prevents BrokenPipeError when output is piped to commands that close early
    try:
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except AttributeError:
        pass  # SIGPIPE not available on Windows

    try:
        sys.exit(_run_with_auto_commit())
    except BrokenPipeError:
        # Python flushes stdout on exit, which can raise BrokenPipeError
        # Quietly close stdout and exit
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        sys.exit(0)
