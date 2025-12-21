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

GOT_DIR = PROJECT_ROOT / ".got"
GOT_TX_DIR = PROJECT_ROOT / ".got-tx"  # Transactional backend directory
WAL_DIR = GOT_DIR / "wal"
SNAPSHOTS_DIR = GOT_DIR / "snapshots"
EVENTS_DIR = GOT_DIR / "events"  # Git-tracked event logs (source of truth)
TASKS_DIR = PROJECT_ROOT / "tasks"

# Backend selection (environment variable or auto-detect)
# Set GOT_USE_TX=1 to force transactional backend
# Set GOT_USE_TX=0 to force event-sourced backend
USE_TX_BACKEND = os.environ.get("GOT_USE_TX", "").lower() in ("1", "true", "yes")
if not USE_TX_BACKEND and TX_BACKEND_AVAILABLE:
    # Auto-detect: if .got-tx exists and has entities, use it
    USE_TX_BACKEND = (GOT_TX_DIR / "entities").exists()

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


def generate_decision_id() -> str:
    """Generate decision ID: decision:D-YYYYMMDD-HHMMSS-XXXX"""
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    suffix = os.urandom(2).hex()
    return f"decision:D-{timestamp}-{suffix}"


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
# PROCESS-SAFE LOCKING
# =============================================================================

class ProcessLock:
    """
    Process-safe file-based lock with stale lock detection.

    Features:
    - Works across processes (not just threads)
    - Detects and recovers from stale locks (dead processes)
    - Timeout support to prevent deadlocks
    - Context manager support
    - Reentrant option for same-process re-acquisition

    Usage:
        lock = ProcessLock("/path/to/.lock")
        with lock:
            # Critical section
            pass

        # Or explicit:
        if lock.acquire(timeout=5.0):
            try:
                # Critical section
            finally:
                lock.release()
    """

    def __init__(
        self,
        lock_path: Path,
        stale_timeout: float = 3600.0,  # 1 hour default
        reentrant: bool = False,
    ):
        """
        Initialize process lock.

        Args:
            lock_path: Path to the lock file
            stale_timeout: Seconds after which a lock is considered stale
            reentrant: If True, same process can acquire multiple times
        """
        self.lock_path = Path(lock_path)
        self.stale_timeout = stale_timeout
        self.reentrant = reentrant
        self._held = False
        self._acquire_count = 0
        self._fd = None

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire the lock.

        Args:
            timeout: Max seconds to wait (None=block forever, 0=non-blocking)

        Returns:
            True if lock acquired, False if timeout
        """
        import fcntl

        # Handle reentrant case
        if self._held and self.reentrant:
            self._acquire_count += 1
            return True

        start_time = time.time()
        poll_interval = 0.05  # 50ms

        while True:
            # Try to acquire
            if self._try_acquire():
                self._held = True
                self._acquire_count = 1
                return True

            # Check for stale lock
            if self._is_stale_lock():
                self._break_stale_lock()
                continue

            # Check timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return False

            # Wait before retry
            if timeout == 0:
                return False

            remaining = None
            if timeout is not None:
                remaining = timeout - (time.time() - start_time)
                if remaining <= 0:
                    return False

            time.sleep(min(poll_interval, remaining) if remaining else poll_interval)

    def _try_acquire(self) -> bool:
        """Attempt to acquire the lock file."""
        import fcntl

        try:
            # Create parent directory if needed
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)

            # Open or create lock file
            self._fd = os.open(
                str(self.lock_path),
                os.O_CREAT | os.O_RDWR,
                0o644
            )

            # Try exclusive lock (non-blocking)
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (IOError, OSError):
                os.close(self._fd)
                self._fd = None
                return False

            # Write PID and timestamp
            os.ftruncate(self._fd, 0)
            os.lseek(self._fd, 0, os.SEEK_SET)
            lock_info = f"{os.getpid()}\n{time.time()}\n"
            os.write(self._fd, lock_info.encode())
            os.fsync(self._fd)

            return True

        except Exception as e:
            if self._fd is not None:
                try:
                    os.close(self._fd)
                except:
                    pass
                self._fd = None
            return False

    def _is_stale_lock(self) -> bool:
        """Check if the existing lock is stale."""
        if not self.lock_path.exists():
            return False

        try:
            content = self.lock_path.read_text().strip()
            lines = content.split('\n')

            if len(lines) < 2:
                return True  # Corrupted, treat as stale

            pid = int(lines[0])
            timestamp = float(lines[1])

            # Check if process is dead
            if not self._process_exists(pid):
                return True

            # Check if lock is too old
            if time.time() - timestamp > self.stale_timeout:
                return True

            return False

        except (ValueError, IndexError, OSError):
            # Corrupted lock file, treat as stale
            return True

    def _process_exists(self, pid: int) -> bool:
        """Check if a process with given PID exists."""
        try:
            os.kill(pid, 0)  # Signal 0 doesn't kill, just checks
            return True
        except OSError:
            return False

    def _break_stale_lock(self) -> None:
        """Remove a stale lock file."""
        try:
            self.lock_path.unlink(missing_ok=True)
        except OSError:
            pass

    def release(self) -> None:
        """Release the lock."""
        import fcntl

        if not self._held:
            return

        if self.reentrant and self._acquire_count > 1:
            self._acquire_count -= 1
            return

        self._held = False
        self._acquire_count = 0

        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                os.close(self._fd)
            except:
                pass
            self._fd = None

        # Clean up lock file
        try:
            self.lock_path.unlink(missing_ok=True)
        except OSError:
            pass

    def is_locked(self) -> bool:
        """Check if lock is currently held by this instance."""
        return self._held

    def __enter__(self):
        """Context manager entry."""
        acquired = self.acquire()
        if not acquired:
            raise TimeoutError(f"Could not acquire lock: {self.lock_path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False

    def __del__(self):
        """Ensure lock is released on garbage collection."""
        if self._held:
            self.release()


# =============================================================================
# EVENT LOGGING (Source of Truth for Cross-Branch Coordination)
# =============================================================================

def atomic_append(filepath: str, content: str) -> None:
    """
    Append content to file atomically to prevent partial writes.

    Writes to a temporary file first, then appends atomically to the target.
    This prevents corruption if the process is interrupted during write.

    Args:
        filepath: Path to the file to append to
        content: Content to append (should include newline if needed)
    """
    filepath = Path(filepath)
    dir_path = filepath.parent

    # Write content to temp file in same directory (for atomic operations)
    fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix='.tmp', text=True)
    try:
        # Write and flush to temp file
        with os.fdopen(fd, 'w') as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())

        # Append from temp file to target file
        with open(tmp_path, 'r') as src:
            data = src.read()
            with open(filepath, 'a') as dst:
                dst.write(data)
                dst.flush()
                os.fsync(dst.fileno())

        # Clean up temp file
        os.unlink(tmp_path)
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise


class EventLog:
    """
    Append-only event log for merge-friendly persistence.

    Each session writes to a unique file, enabling conflict-free merges.
    The graph state is rebuilt from all event files on startup.
    """

    def __init__(self, events_dir: Path, session_id: Optional[str] = None):
        self.events_dir = Path(events_dir)
        self.events_dir.mkdir(parents=True, exist_ok=True)

        # Each session gets a unique event file
        self.session_id = session_id or generate_session_id()
        self.branch = get_current_branch()
        self.event_file = self.events_dir / f"{self.session_id}.jsonl"

    def log(self, event_type: str, **data) -> Dict[str, Any]:
        """Append an event to the session log."""
        event = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "event": event_type,
            "meta": {
                "branch": self.branch,
                "session": self.session_id,
            },
            **data
        }
        # Use atomic append to prevent partial writes
        atomic_append(self.event_file, json.dumps(event, default=str) + "\n")
        return event

    def log_node_create(self, node_id: str, node_type: str, data: Dict) -> Dict:
        """Log a node creation event."""
        return self.log("node.create", id=node_id, type=node_type, data=data)

    def log_node_update(self, node_id: str, changes: Dict) -> Dict:
        """Log a node update event."""
        return self.log("node.update", id=node_id, changes=changes)

    def log_node_delete(self, node_id: str) -> Dict:
        """Log a node deletion event."""
        return self.log("node.delete", id=node_id)

    def log_edge_create(self, src: str, tgt: str, edge_type: str, weight: float = 1.0) -> Dict:
        """Log an edge creation event."""
        return self.log("edge.create", src=src, tgt=tgt, type=edge_type, weight=weight)

    def log_edge_delete(self, src: str, tgt: str, edge_type: str) -> Dict:
        """Log an edge deletion event."""
        return self.log("edge.delete", src=src, tgt=tgt, type=edge_type)

    # =========================================================================
    # AGENT HANDOFF EVENTS
    # =========================================================================

    def log_handoff_initiate(
        self,
        handoff_id: str,
        source_agent: str,
        target_agent: str,
        task_id: str,
        context: Dict[str, Any],
        instructions: str = "",
    ) -> Dict:
        """Log a handoff initiation from one agent to another.

        Args:
            handoff_id: Unique identifier for this handoff
            source_agent: Agent initiating the handoff (e.g., "director", "main")
            target_agent: Agent receiving the work (e.g., "sub-agent-1", "reviewer")
            task_id: The task being handed off
            context: Context data for the receiving agent
            instructions: Specific instructions for the receiving agent
        """
        return self.log(
            "handoff.initiate",
            handoff_id=handoff_id,
            source_agent=source_agent,
            target_agent=target_agent,
            task_id=task_id,
            context=context,
            instructions=instructions,
        )

    def log_handoff_accept(
        self,
        handoff_id: str,
        agent: str,
        acknowledgment: str = "",
    ) -> Dict:
        """Log an agent accepting a handoff.

        Args:
            handoff_id: The handoff being accepted
            agent: Agent accepting the handoff
            acknowledgment: Optional acknowledgment message
        """
        return self.log(
            "handoff.accept",
            handoff_id=handoff_id,
            agent=agent,
            acknowledgment=acknowledgment,
        )

    def log_handoff_complete(
        self,
        handoff_id: str,
        agent: str,
        result: Dict[str, Any],
        artifacts: Optional[List[str]] = None,
    ) -> Dict:
        """Log completion of a handed-off task.

        Args:
            handoff_id: The handoff being completed
            agent: Agent completing the work
            result: Results of the work (success, findings, etc.)
            artifacts: List of artifacts created (files, commits, etc.)
        """
        return self.log(
            "handoff.complete",
            handoff_id=handoff_id,
            agent=agent,
            result=result,
            artifacts=artifacts or [],
        )

    def log_handoff_reject(
        self,
        handoff_id: str,
        agent: str,
        reason: str,
        suggestion: str = "",
    ) -> Dict:
        """Log an agent rejecting a handoff.

        Args:
            handoff_id: The handoff being rejected
            agent: Agent rejecting the handoff
            reason: Why the handoff was rejected
            suggestion: Suggested alternative approach
        """
        return self.log(
            "handoff.reject",
            handoff_id=handoff_id,
            agent=agent,
            reason=reason,
            suggestion=suggestion,
        )

    def log_handoff_context(
        self,
        handoff_id: str,
        agent: str,
        context_type: str,
        data: Dict[str, Any],
    ) -> Dict:
        """Log context being passed during a handoff.

        Args:
            handoff_id: The associated handoff
            agent: Agent providing the context
            context_type: Type of context (e.g., "files", "decisions", "blockers")
            data: The context data
        """
        return self.log(
            "handoff.context",
            handoff_id=handoff_id,
            agent=agent,
            context_type=context_type,
            data=data,
        )

    # =========================================================================
    # REASONING TRACE EVENTS
    # =========================================================================

    def log_decision(
        self,
        decision_id: str,
        decision: str,
        rationale: str,
        affects: Optional[List[str]] = None,
        alternatives: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """Log a decision with its rationale.

        This creates a trace of WHY something was done, not just WHAT.
        Future agents can query: "Why was this built this way?"

        Args:
            decision_id: Unique identifier for this decision
            decision: What was decided
            rationale: Why this choice was made
            affects: List of node IDs affected by this decision (creates JUSTIFIES edges)
            alternatives: Alternatives that were considered but rejected
            context: Additional context (file, line, function, etc.)
        """
        return self.log(
            "decision.create",
            id=decision_id,
            decision=decision,
            rationale=rationale,
            affects=affects or [],
            alternatives=alternatives or [],
            context=context or {},
        )

    def log_decision_supersede(
        self,
        new_decision_id: str,
        old_decision_id: str,
        reason: str,
    ) -> Dict:
        """Log that a new decision supersedes an old one.

        Creates a SUPERSEDES edge for tracking decision evolution.
        """
        return self.log(
            "decision.supersede",
            new_id=new_decision_id,
            old_id=old_decision_id,
            reason=reason,
        )

    def log_reasoning_step(
        self,
        step_type: str,
        content: str,
        parent_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """Log a reasoning step in the thought process.

        Step types: QUESTION, HYPOTHESIS, EVIDENCE, CONCLUSION, ACTION

        Args:
            step_type: Type of reasoning step
            content: The content of this step
            parent_id: ID of parent step (for chained reasoning)
            context: Additional context
        """
        step_id = f"step:{datetime.now().strftime('%Y%m%d-%H%M%S')}-{os.urandom(2).hex()}"
        return self.log(
            "reasoning.step",
            id=step_id,
            type=step_type,
            content=content,
            parent_id=parent_id,
            context=context or {},
        )

    @classmethod
    def load_all_events(cls, events_dir: Path) -> List[Dict]:
        """Load and sort all events from all session files."""
        events_dir = Path(events_dir)
        if not events_dir.exists():
            return []

        all_events = []
        for event_file in events_dir.glob("*.jsonl"):
            try:
                with open(event_file) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                all_events.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
            except Exception:
                continue

        # Sort by timestamp for deterministic replay
        all_events.sort(key=lambda e: e.get("ts", ""))
        return all_events

    @classmethod
    def rebuild_graph_from_events(cls, events: List[Dict], with_telemetry: bool = False):
        """Rebuild a ThoughtGraph from events (event sourcing).

        Args:
            events: List of event dictionaries to replay
            with_telemetry: If True, return dict with 'graph' and 'telemetry' keys

        Returns:
            ThoughtGraph if with_telemetry=False (default, backward compatible)
            Dict with 'graph' and 'telemetry' if with_telemetry=True
        """
        graph = ThoughtGraph()
        errors = []
        event_num = 0

        # Telemetry counters
        telemetry = {
            "node_create_events": 0,
            "nodes_created": 0,
            "edge_create_events": 0,
            "edges_created": 0,
            "edges_skipped": 0,
            "errors": 0,
            "validation_passed": True,
            "validation_errors": [],
            "summary": "",
        }

        for event in events:
            event_num += 1
            event_type = event.get("event", "")

            try:
                if event_type == "node.create":
                    telemetry["node_create_events"] += 1
                    try:
                        node_type_str = event.get("type", "TASK").upper()
                        node_type = NodeType[node_type_str] if hasattr(NodeType, node_type_str) else NodeType.TASK
                        graph.add_node(
                            node_id=event["id"],
                            node_type=node_type,
                            content=event.get("data", {}).get("title", ""),
                            properties=event.get("data", {}),
                            metadata=event.get("meta", {})
                        )
                        telemetry["nodes_created"] += 1
                    except KeyError as e:
                        error_msg = f"Event {event_num}: Missing required field for node.create: {e}"
                        logger.error(error_msg)
                        errors.append(error_msg)
                        telemetry["errors"] += 1
                    except Exception as e:
                        error_msg = f"Event {event_num}: Failed to create node: {e}"
                        logger.error(error_msg)
                        errors.append(error_msg)
                        telemetry["errors"] += 1

                elif event_type == "node.update":
                    node_id = event["id"]
                    changes = event.get("changes", {})

                    # Normalize ID - try with and without task: prefix
                    actual_id = node_id
                    if node_id not in graph.nodes:
                        # Try with task: prefix if it looks like a task ID
                        if node_id.startswith("T-") and f"task:{node_id}" in graph.nodes:
                            actual_id = f"task:{node_id}"
                        # Try without task: prefix
                        elif node_id.startswith("task:") and node_id[5:] in graph.nodes:
                            actual_id = node_id[5:]

                    if actual_id in graph.nodes:
                        for key, value in changes.items():
                            graph.nodes[actual_id].properties[key] = value
                    else:
                        logger.warning(f"Event {event_num}: Cannot update non-existent node {node_id}")

                elif event_type == "node.delete":
                    node_id = event["id"]

                    # Normalize ID - same as node.update
                    actual_id = node_id
                    if node_id not in graph.nodes:
                        if node_id.startswith("T-") and f"task:{node_id}" in graph.nodes:
                            actual_id = f"task:{node_id}"
                        elif node_id.startswith("task:") and node_id[5:] in graph.nodes:
                            actual_id = node_id[5:]

                    if actual_id in graph.nodes:
                        del graph.nodes[actual_id]
                    else:
                        logger.warning(f"Event {event_num}: Cannot delete non-existent node {node_id}")

                elif event_type == "edge.create":
                    telemetry["edge_create_events"] += 1
                    try:
                        edge_type_str = event.get("type", "RELATES_TO").upper()
                        # Use try/except for EdgeType lookup since hasattr doesn't work correctly with enums
                        try:
                            edge_type = EdgeType[edge_type_str]
                        except KeyError:
                            edge_type = EdgeType.MOTIVATES

                        src_raw = event["src"]
                        tgt_raw = event["tgt"]
                        weight = event.get("weight", 1.0)

                        # Handle comma-concatenated IDs (malformed data fix)
                        # Split on comma and trim whitespace from each ID
                        src_ids = [s.strip() for s in src_raw.split(",")]
                        tgt_ids = [t.strip() for t in tgt_raw.split(",")]

                        # Create edges for each source-target combination
                        edges_created = 0
                        edges_skipped_this_event = 0
                        for src_id in src_ids:
                            # Try ID normalization for source
                            actual_src = src_id
                            if src_id not in graph.nodes:
                                if src_id.startswith("T-") and f"task:{src_id}" in graph.nodes:
                                    actual_src = f"task:{src_id}"
                                elif src_id.startswith("task:") and src_id[5:] in graph.nodes:
                                    actual_src = src_id[5:]

                            if actual_src not in graph.nodes:
                                logger.warning(f"Event {event_num}: Skipping edge - source node {src_id} does not exist")
                                edges_skipped_this_event += len(tgt_ids)
                                continue

                            for tgt_id in tgt_ids:
                                # Try ID normalization for target
                                actual_tgt = tgt_id
                                if tgt_id not in graph.nodes:
                                    if tgt_id.startswith("T-") and f"task:{tgt_id}" in graph.nodes:
                                        actual_tgt = f"task:{tgt_id}"
                                    elif tgt_id.startswith("task:") and tgt_id[5:] in graph.nodes:
                                        actual_tgt = tgt_id[5:]

                                if actual_tgt not in graph.nodes:
                                    logger.warning(f"Event {event_num}: Skipping edge - target node {tgt_id} does not exist")
                                    edges_skipped_this_event += 1
                                    continue

                                graph.add_edge(
                                    from_id=actual_src,
                                    to_id=actual_tgt,
                                    edge_type=edge_type,
                                    weight=weight
                                )
                                edges_created += 1

                        telemetry["edges_created"] += edges_created
                        telemetry["edges_skipped"] += edges_skipped_this_event

                        if edges_created == 0 and len(src_ids) == 1 and len(tgt_ids) == 1:
                            # Log error only if it was a simple edge that failed (not comma-split)
                            error_msg = f"Event {event_num}: Cannot create edge - nodes not found"
                            logger.error(error_msg)
                            errors.append(error_msg)
                            telemetry["errors"] += 1

                    except KeyError as e:
                        error_msg = f"Event {event_num}: Missing required field for edge.create: {e}"
                        logger.error(error_msg)
                        errors.append(error_msg)
                        telemetry["errors"] += 1
                    except Exception as e:
                        error_msg = f"Event {event_num}: Failed to create edge: {e}"
                        logger.error(error_msg)
                        errors.append(error_msg)
                        telemetry["errors"] += 1

                elif event_type == "edge.delete":
                    # Find and remove the edge
                    edge_key = (event["src"], event["tgt"], event.get("type", ""))
                    for eid, edge in list(graph.edges.items()):
                        if (edge.source_id, edge.target_id, edge.edge_type.name) == edge_key:
                            del graph.edges[eid]
                            break

                elif event_type == "decision.create":
                    # Create decision node and JUSTIFIES edges
                    decision_id = event["id"]
                    graph.add_node(
                        node_id=decision_id,
                        node_type=NodeType.DECISION,  # Fixed: was incorrectly CONTEXT
                        content=event.get("decision", ""),
                        properties={
                            "type": "decision",
                            "rationale": event.get("rationale", ""),
                            "alternatives": event.get("alternatives", []),
                        },
                        metadata=event.get("context", {})
                    )
                    # Create JUSTIFIES edges to affected nodes
                    for affected_raw in event.get("affects", []):
                        # Handle comma-concatenated IDs (malformed data fix)
                        affected_ids = [a.strip() for a in affected_raw.split(",")]
                        for affected_id in affected_ids:
                            # Try ID normalization
                            actual_id = affected_id
                            if affected_id not in graph.nodes:
                                if affected_id.startswith("T-") and f"task:{affected_id}" in graph.nodes:
                                    actual_id = f"task:{affected_id}"
                                elif affected_id.startswith("task:") and affected_id[5:] in graph.nodes:
                                    actual_id = affected_id[5:]

                            if actual_id in graph.nodes:
                                try:
                                    graph.add_edge(
                                        decision_id, actual_id,
                                        EdgeType.MOTIVATES,  # JUSTIFIES conceptually
                                        weight=1.0, confidence=1.0
                                    )
                                except Exception as e:
                                    logger.warning(f"Event {event_num}: Failed to create JUSTIFIES edge to {affected_id}: {e}")
                            else:
                                logger.warning(f"Event {event_num}: Cannot create edge to non-existent node {affected_id}")

                elif event_type == "decision.supersede":
                    # Create SUPERSEDES edge (new decision supersedes old)
                    new_id = event.get("new_id")
                    old_id = event.get("old_id")
                    if new_id in graph.nodes and old_id in graph.nodes:
                        try:
                            graph.add_edge(
                                new_id, old_id,
                                EdgeType.MOTIVATES,  # SUPERSEDES conceptually
                                weight=1.0, confidence=1.0
                            )
                        except Exception as e:
                            logger.warning(f"Event {event_num}: Failed to create SUPERSEDES edge: {e}")
                    else:
                        missing_nodes = []
                        if new_id not in graph.nodes:
                            missing_nodes.append(f"new_id={new_id}")
                        if old_id not in graph.nodes:
                            missing_nodes.append(f"old_id={old_id}")
                        logger.warning(f"Event {event_num}: Cannot create SUPERSEDES edge - missing nodes: {', '.join(missing_nodes)}")

                elif event_type == "handoff.initiate":
                    # Handoff events don't create graph nodes - they're just logged
                    # But we should acknowledge them to prevent "unknown event type" warnings
                    handoff_id = event.get("handoff_id") or event.get("id")
                    if handoff_id:
                        logger.debug(f"Handoff initiated: {handoff_id}")

                elif event_type == "handoff.accept":
                    handoff_id = event.get("handoff_id") or event.get("id")
                    if handoff_id:
                        logger.debug(f"Handoff accepted: {handoff_id}")

                elif event_type == "handoff.complete":
                    handoff_id = event.get("handoff_id") or event.get("id")
                    if handoff_id:
                        logger.debug(f"Handoff completed: {handoff_id}")

                elif event_type == "handoff.reject":
                    handoff_id = event.get("handoff_id") or event.get("id")
                    if handoff_id:
                        logger.debug(f"Handoff rejected: {handoff_id}")

                elif event_type == "handoff.context":
                    # Context additions during handoff - just acknowledge
                    pass

                elif event_type == "":
                    logger.warning(f"Event {event_num}: Empty event type")
                else:
                    logger.warning(f"Event {event_num}: Unknown event type '{event_type}'")

            except Exception as e:
                error_msg = f"Event {event_num}: Unexpected error processing event type '{event_type}': {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                continue

        if errors:
            logger.warning(f"Graph rebuild completed with {len(errors)} error(s)")

        # Finalize telemetry
        if telemetry["edges_skipped"] > 0:
            telemetry["validation_passed"] = False
            telemetry["validation_errors"].append(
                f"Skipped {telemetry['edges_skipped']} edge(s) due to missing nodes"
            )

        if telemetry["errors"] > 0:
            telemetry["validation_passed"] = False
            telemetry["validation_errors"].append(
                f"Encountered {telemetry['errors']} error(s) during rebuild"
            )

        # Generate summary
        telemetry["summary"] = (
            f"Rebuilt graph: {telemetry['nodes_created']} nodes created "
            f"({telemetry['node_create_events']} events), "
            f"{telemetry['edges_created']} edges created "
            f"({telemetry['edge_create_events']} events), "
            f"{telemetry['edges_skipped']} edges skipped"
        )

        if with_telemetry:
            return {"graph": graph, "telemetry": telemetry}

        return graph

    @classmethod
    def compact_events(
        cls,
        events_dir: Path,
        preserve_handoffs: bool = True,
        preserve_days: int = 7,
    ) -> Dict[str, Any]:
        """Compact old events into a single consolidated event file.

        Event compaction works like git gc - it:
        1. Replays all events to get final state
        2. Creates node.create events for all current nodes
        3. Creates edge.create events for all current edges
        4. Optionally preserves handoff events (for audit trail)
        5. Preserves recent events (within preserve_days)
        6. Removes old session files

        Args:
            events_dir: Directory containing event files
            preserve_handoffs: Keep handoff events for audit trail
            preserve_days: Keep events from last N days unchanged

        Returns:
            Dict with compaction stats
        """
        events_dir = Path(events_dir)
        if not events_dir.exists():
            return {"error": "Events directory does not exist"}

        # Calculate cutoff timestamp
        cutoff = datetime.utcnow() - __import__('datetime').timedelta(days=preserve_days)
        cutoff_str = cutoff.isoformat() + "Z"

        # Load all events
        all_events = cls.load_all_events(events_dir)
        if not all_events:
            return {"status": "nothing_to_compact", "event_count": 0}

        # Separate events into categories
        old_events = []
        recent_events = []
        handoff_events = []

        for event in all_events:
            ts = event.get("ts", "")
            event_type = event.get("event", "")

            if event_type.startswith("handoff.") and preserve_handoffs:
                handoff_events.append(event)
            elif ts < cutoff_str:
                old_events.append(event)
            else:
                recent_events.append(event)

        if not old_events:
            return {
                "status": "nothing_to_compact",
                "recent_events": len(recent_events),
                "handoff_events": len(handoff_events),
            }

        # Rebuild graph from all events (including recent, for accurate final state)
        final_graph = cls.rebuild_graph_from_events(all_events)

        # Create compaction event log
        compact_log = cls(
            events_dir,
            session_id=f"compact-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )

        # Write final state as creation events
        nodes_written = 0
        edges_written = 0

        for node_id, node in final_graph.nodes.items():
            compact_log.log_node_create(
                node_id=node_id,
                node_type=node.node_type.name if hasattr(node.node_type, 'name') else str(node.node_type),
                data={
                    **node.properties,
                    "title": node.content,
                }
            )
            nodes_written += 1

        # Write edges
        edges = final_graph.edges
        if isinstance(edges, dict):
            edge_list = edges.values()
        else:
            edge_list = edges

        for edge in edge_list:
            src = getattr(edge, 'source_id', edge.get('source_id') if isinstance(edge, dict) else None)
            tgt = getattr(edge, 'target_id', edge.get('target_id') if isinstance(edge, dict) else None)
            etype = getattr(edge, 'edge_type', edge.get('edge_type') if isinstance(edge, dict) else None)
            weight = getattr(edge, 'weight', edge.get('weight', 1.0) if isinstance(edge, dict) else 1.0)

            if src and tgt:
                compact_log.log_edge_create(
                    src=src,
                    tgt=tgt,
                    edge_type=etype.name if hasattr(etype, 'name') else str(etype),
                    weight=weight
                )
                edges_written += 1

        # Write preserved handoff events to the compact file
        if handoff_events:
            for event in handoff_events:
                # Use atomic append to prevent partial writes
                atomic_append(compact_log.event_file, json.dumps(event, default=str) + "\n")

        # Find and remove old event files (but not the compact file or recent files)
        files_removed = []
        compact_filename = compact_log.event_file.name

        for event_file in events_dir.glob("*.jsonl"):
            if event_file.name == compact_filename:
                continue

            # Check if file contains only old events
            try:
                file_events = []
                with open(event_file) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            file_events.append(json.loads(line))

                # If all events in file are old (pre-cutoff), remove it
                all_old = all(e.get("ts", "9999") < cutoff_str for e in file_events)
                has_handoffs = any(e.get("event", "").startswith("handoff.") for e in file_events)

                if all_old and not (has_handoffs and preserve_handoffs):
                    event_file.unlink()
                    files_removed.append(event_file.name)

            except Exception:
                continue

        return {
            "status": "compacted",
            "nodes_written": nodes_written,
            "edges_written": edges_written,
            "handoffs_preserved": len(handoff_events),
            "files_removed": len(files_removed),
            "compact_file": compact_filename,
            "original_event_count": len(all_events),
            "old_events_consolidated": len(old_events),
            "recent_events_kept": len(recent_events),
        }


# =============================================================================
# HANDOFF MANAGER
# =============================================================================

def generate_handoff_id() -> str:
    """Generate a unique handoff ID: handoff:H-YYYYMMDD-HHMMSS-XXXX"""
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    suffix = os.urandom(2).hex()
    return f"handoff:H-{timestamp}-{suffix}"


class HandoffManager:
    """Manages agent handoffs using the event log.

    Provides high-level operations for initiating, accepting,
    completing, and tracking handoffs between agents.
    """

    def __init__(self, event_log: EventLog):
        self.event_log = event_log
        self._active_handoffs: Dict[str, Dict] = {}

    def initiate_handoff(
        self,
        source_agent: str,
        target_agent: str,
        task_id: str,
        context: Dict[str, Any],
        instructions: str = "",
    ) -> str:
        """Initiate a handoff to another agent.

        Returns:
            handoff_id for tracking
        """
        handoff_id = generate_handoff_id()

        self.event_log.log_handoff_initiate(
            handoff_id=handoff_id,
            source_agent=source_agent,
            target_agent=target_agent,
            task_id=task_id,
            context=context,
            instructions=instructions,
        )

        self._active_handoffs[handoff_id] = {
            "status": "initiated",
            "source_agent": source_agent,
            "target_agent": target_agent,
            "task_id": task_id,
            "initiated_at": datetime.now().isoformat(),
        }

        return handoff_id

    def accept_handoff(self, handoff_id: str, agent: str, acknowledgment: str = "") -> bool:
        """Accept a handoff."""
        self.event_log.log_handoff_accept(
            handoff_id=handoff_id,
            agent=agent,
            acknowledgment=acknowledgment,
        )

        if handoff_id in self._active_handoffs:
            self._active_handoffs[handoff_id]["status"] = "accepted"
            self._active_handoffs[handoff_id]["accepted_at"] = datetime.now().isoformat()

        return True

    def complete_handoff(
        self,
        handoff_id: str,
        agent: str,
        result: Dict[str, Any],
        artifacts: Optional[List[str]] = None,
    ) -> bool:
        """Complete a handoff with results."""
        self.event_log.log_handoff_complete(
            handoff_id=handoff_id,
            agent=agent,
            result=result,
            artifacts=artifacts,
        )

        if handoff_id in self._active_handoffs:
            self._active_handoffs[handoff_id]["status"] = "completed"
            self._active_handoffs[handoff_id]["completed_at"] = datetime.now().isoformat()
            self._active_handoffs[handoff_id]["result"] = result

        return True

    def add_context(
        self,
        handoff_id: str,
        agent: str,
        context_type: str,
        data: Dict[str, Any],
    ) -> bool:
        """Add context to a handoff."""
        self.event_log.log_handoff_context(
            handoff_id=handoff_id,
            agent=agent,
            context_type=context_type,
            data=data,
        )
        return True

    @classmethod
    def load_handoffs_from_events(cls, events: List[Dict]) -> List[Dict[str, Any]]:
        """Load handoff state from event log.

        Returns list of handoffs with their current state.
        """
        handoffs = {}

        for event in events:
            event_type = event.get("event", "")
            if not event_type.startswith("handoff."):
                continue

            handoff_id = event.get("handoff_id")
            if not handoff_id:
                continue

            if handoff_id not in handoffs:
                handoffs[handoff_id] = {"id": handoff_id, "events": []}

            handoffs[handoff_id]["events"].append(event)

            if event_type == "handoff.initiate":
                handoffs[handoff_id].update({
                    "status": "initiated",
                    "source_agent": event.get("source_agent"),
                    "target_agent": event.get("target_agent"),
                    "task_id": event.get("task_id"),
                    "instructions": event.get("instructions"),
                    "initiated_at": event.get("ts"),
                })
            elif event_type == "handoff.accept":
                handoffs[handoff_id]["status"] = "accepted"
                handoffs[handoff_id]["accepted_at"] = event.get("ts")
            elif event_type == "handoff.complete":
                handoffs[handoff_id]["status"] = "completed"
                handoffs[handoff_id]["completed_at"] = event.get("ts")
                handoffs[handoff_id]["result"] = event.get("result")
            elif event_type == "handoff.reject":
                handoffs[handoff_id]["status"] = "rejected"
                handoffs[handoff_id]["rejected_at"] = event.get("ts")
                handoffs[handoff_id]["reject_reason"] = event.get("reason")

        return list(handoffs.values())


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

    def __init__(self, got_dir: Path = GOT_TX_DIR):
        if not TX_BACKEND_AVAILABLE:
            raise RuntimeError("Transactional backend not available")

        self.got_dir = Path(got_dir)
        self._manager = TxGoTManager(self.got_dir, durability=DurabilityMode.BALANCED)

        # Compatibility attributes (some commands access these directly)
        self.graph = ThoughtGraph()
        self.events_dir = self.got_dir / "events"  # Not used but needed for compat
        self.wal_dir = self.got_dir / "wal"
        self.snapshots_dir = self.got_dir / "snapshots"

        # Ensure directories exist
        self.events_dir.mkdir(parents=True, exist_ok=True)

    def _strip_prefix(self, node_id: str) -> str:
        """Strip task:/decision: prefix from ID."""
        if node_id.startswith("task:"):
            return node_id[5:]
        if node_id.startswith("decision:"):
            return node_id[9:]
        return node_id

    def _add_prefix(self, node_id: str, prefix: str = "task:") -> str:
        """Add prefix to ID if not present."""
        if not node_id.startswith(prefix):
            return f"{prefix}{node_id}"
        return node_id

    def _tx_task_to_node(self, task: "TxTask") -> ThoughtNode:
        """Convert TxTask to ThoughtNode for compatibility."""
        return ThoughtNode(
            id=f"task:{task.id}",
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

        return f"task:{task.id}"

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
        return self.update_task(task_id, status="in_progress")

    def complete_task(self, task_id: str, retrospective: str = "") -> bool:
        """Complete a task."""
        updates = {"status": "completed"}
        if retrospective:
            # Get current task to merge properties (don't overwrite!)
            task = self.get_task(task_id)
            if not task:
                return False
            # Copy existing properties and add/update retrospective
            merged_properties = dict(task.properties) if task.properties else {}
            merged_properties["retrospective"] = retrospective
            updates["properties"] = merged_properties
        return self.update_task(task_id, **updates)

    def block_task(self, task_id: str, reason: str = "", blocked_by: Optional[str] = None) -> bool:
        """Block a task."""
        return self.update_task(task_id, status="blocked")

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
        """Get all tasks this task depends on."""
        clean_id = self._strip_prefix(task_id)
        try:
            # Tasks that block this task are tasks this task depends on
            deps = self._manager.get_blockers(clean_id)
            return [self._tx_task_to_node(t) for t in deps if t]
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

            return {
                "total_tasks": len(all_tasks),
                "tasks_by_status": by_status,
                "total_edges": edge_count,
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {
                "total_tasks": 0,
                "tasks_by_status": {},
                "total_edges": 0,
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
                    if edge.edge_type == "blocks":
                        result['blocks'].append(target_node)
                    elif edge.edge_type == "depends_on":
                        result['depends_on'].append(target_node)

            # Process incoming edges
            for edge in incoming:
                source_task = self._manager.get_task(edge.source_id)
                if source_task:
                    source_node = self._tx_task_to_node(source_task)
                    if edge.edge_type == "blocks":
                        result['blocked_by'].append(source_node)
                    elif edge.edge_type == "depends_on":
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
                    "id": f"task:{task.id}",
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
                                "source": f"task:{edge_data.get('source_id', '')}",
                                "target": f"task:{edge_data.get('target_id', '')}",
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

    # Stub methods for compatibility (not implemented in TX backend yet)
    def create_decision(self, *args, **kwargs) -> str:
        raise NotImplementedError("Decisions not yet implemented in TX backend")

    def list_decisions(self, *args, **kwargs) -> List:
        return []

    def get_decisions_for_task(self, *args, **kwargs) -> List:
        return []

    def create_sprint(self, *args, **kwargs) -> str:
        raise NotImplementedError("Sprints not yet implemented in TX backend")

    def get_current_sprint(self, *args, **kwargs):
        return None

    def list_sprints(self, *args, **kwargs) -> List:
        return []

    def initiate_handoff(self, *args, **kwargs) -> str:
        raise NotImplementedError("Handoffs not yet implemented in TX backend")

    def accept_handoff(self, *args, **kwargs) -> bool:
        raise NotImplementedError("Handoffs not yet implemented in TX backend")

    def complete_handoff(self, *args, **kwargs) -> bool:
        raise NotImplementedError("Handoffs not yet implemented in TX backend")

    def list_handoffs(self, *args, **kwargs) -> List:
        return []

    def get_sprint_tasks(self, sprint_id: str) -> List[ThoughtNode]:
        """Get all tasks in a sprint.

        Note: Sprint support is limited in TX backend.
        Returns empty list with warning.
        """
        logger.warning("Sprint support limited in TX backend - returning empty list")
        return []

    def get_sprint_progress(self, sprint_id: str) -> Dict[str, Any]:
        """Get sprint progress statistics.

        Note: Sprint support is limited in TX backend.
        Returns empty progress dict.
        """
        logger.warning("Sprint support limited in TX backend - returning empty progress")
        return {
            "total_tasks": 0,
            "by_status": {},
            "completed": 0,
            "progress_percent": 0.0,
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
        """Simple query language for the graph.

        Supported queries:
        - "what blocks <task_id>"
        - "what depends on <task_id>"
        - "blocked tasks"
        - "active tasks"
        - "pending tasks"
        - "relationships <task_id>"

        Returns list of result dicts.
        """
        query_str = query_str.strip().lower()
        results = []

        if query_str.startswith("what blocks "):
            task_id = query_str[12:].strip()
            for node in self.what_blocks(task_id):
                results.append({
                    "id": node.id,
                    "title": node.content,
                    "status": node.properties.get("status"),
                    "relation": "blocks",
                })

        elif query_str.startswith("what depends on "):
            task_id = query_str[16:].strip()
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

        elif query_str.startswith("relationships "):
            task_id = query_str[14:].strip()
            rels = self.get_all_relationships(task_id)
            for rel_type, nodes in rels.items():
                for node in nodes:
                    results.append({
                        "relation": rel_type,
                        "id": node.id,
                        "title": node.content,
                    })

        return results

    def sync_to_git(self) -> str:
        """Sync to git (no-op for TX backend, state is already persistent)."""
        return ""


# =============================================================================
# GRAPH MANAGER (Event-Sourced Backend)
# =============================================================================

class GoTProjectManager:
    """
    Manages project artifacts (tasks, sprints, epics) in a ThoughtGraph.

    Uses event-sourced persistence for merge-friendly cross-branch coordination.
    Each session writes to a unique event file, enabling conflict-free merges.
    """

    def __init__(self, got_dir: Path = GOT_DIR):
        self.got_dir = Path(got_dir)
        self.wal_dir = self.got_dir / "wal"
        self.snapshots_dir = self.got_dir / "snapshots"
        self.events_dir = self.got_dir / "events"

        # Ensure directories exist
        self.got_dir.mkdir(parents=True, exist_ok=True)
        self.wal_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.events_dir.mkdir(parents=True, exist_ok=True)

        # Process-safe lock for graph mutations
        self._lock = ProcessLock(
            self.got_dir / ".got.lock",
            stale_timeout=3600.0,  # 1 hour
            reentrant=True,  # Allow nested operations
        )

        # Initialize event log (each session gets unique file)
        self.event_log = EventLog(self.events_dir)

        # Initialize graph and WAL (WAL for crash recovery)
        self.graph = ThoughtGraph()
        self.wal = GraphWAL(str(self.wal_dir))

        # Load existing state
        self._load_state()

    def _load_state(self) -> None:
        """Load graph state with event-sourced priority.

        Recovery order (for environment resilience):
        1. Events (git-tracked, source of truth) - .got/events/*.jsonl
        2. WAL snapshots (local cache) - .got/wal/snapshots/
        3. Git-tracked snapshots (backup) - .got/snapshots/
        4. WAL recovery (crash recovery)
        5. Empty graph (fresh start)
        """
        # 1. Try loading from events (source of truth)
        events = EventLog.load_all_events(self.events_dir)
        if events:
            self.graph = EventLog.rebuild_graph_from_events(events)
            return

        # 2. Try loading from WAL snapshot (local cache)
        try:
            snapshot = self.wal.load_snapshot()
            if snapshot:
                self.graph = snapshot
                return
        except Exception:
            pass

        # 3. Try git-tracked snapshots (survives clone)
        try:
            self._load_from_git_tracked_snapshot()
            if len(self.graph.nodes) > 0:
                return
        except Exception:
            pass

        # 4. Try WAL recovery (crash recovery)
        try:
            recovery = GraphRecovery(
                wal_dir=str(self.wal_dir),
                chunks_dir=str(self.got_dir / "chunks"),
            )

            if recovery.needs_recovery():
                result = recovery.recover()
                if result.success and result.graph:
                    self.graph = result.graph
                    return
        except Exception:
            pass

        # 5. Start with empty graph (fresh start)

    def _load_from_git_tracked_snapshot(self) -> None:
        """Load from git-tracked snapshot (survives fresh clone)."""
        import gzip

        # Find latest snapshot in git-tracked directory
        snapshots = sorted(self.snapshots_dir.glob("*.json.gz"), reverse=True)
        if not snapshots:
            snapshots = sorted(self.snapshots_dir.glob("*.json"), reverse=True)
        if not snapshots:
            return

        snap_file = snapshots[0]
        try:
            if snap_file.suffix == ".gz":
                with gzip.open(snap_file, 'rt') as f:
                    data = json.load(f)
            else:
                with open(snap_file) as f:
                    data = json.load(f)

            state = data.get("state", data)
            nodes = state.get("nodes", {})

            # Rebuild graph
            self.graph = ThoughtGraph()
            for node_id, node in nodes.items():
                self.graph.add_node(
                    node_id=node_id,
                    node_type=NodeType[node.get("node_type", "task").upper()],
                    content=node.get("content", ""),
                    properties=node.get("properties", {}),
                    metadata=node.get("metadata", {})
                )

            # Restore edges
            edges = state.get("edges", {})
            for edge_id, edge in edges.items():
                try:
                    self.graph.add_edge(
                        source_id=edge.get("source_id"),
                        target_id=edge.get("target_id"),
                        edge_type=EdgeType[edge.get("edge_type", "RELATES_TO").upper()],
                        weight=edge.get("weight", 1.0),
                        metadata=edge.get("metadata", {})
                    )
                except Exception:
                    pass
        except Exception:
            pass

    def sync_to_git(self) -> str:
        """Sync current state to git-tracked snapshot.

        Call this before committing to ensure state survives clone.
        Returns the snapshot filename.
        """
        import gzip
        import shutil

        # First, create a fresh WAL snapshot
        snapshot_id = self.wal.create_snapshot(self.graph, compress=True)

        # Find and copy to git-tracked directory
        wal_snapshots = self.wal_dir / "snapshots"
        source_files = list(wal_snapshots.glob(f"*{snapshot_id}*"))

        if source_files:
            source = source_files[0]
            dest = self.snapshots_dir / source.name
            shutil.copy2(source, dest)
            return source.name

        return snapshot_id

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
        blocks: Optional[List[str]] = None,
    ) -> str:
        """
        Create a new task.

        Args:
            title: Task title
            priority: Task priority (low, medium, high, critical)
            category: Task category (feature, bugfix, etc.)
            description: Detailed task description
            sprint_id: Sprint ID to add task to
            depends_on: List of task IDs this task depends on
            blocks: List of task IDs this task blocks

        Returns:
            Task ID
        """
        with self._lock:
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

            # Log to event log (source of truth for cross-branch coordination)
            self.event_log.log_node_create(task_id, "TASK", {**properties, "title": title})

            # Also log to WAL for crash recovery
            self.wal.log_add_node(task_id, NodeType.TASK, title, properties, metadata)

            # Add to sprint if specified
            if sprint_id:
                self._add_task_to_sprint(task_id, sprint_id)

            # Add dependencies (edges from this task TO dependency tasks)
            if depends_on:
                for dep_id in depends_on:
                    if self.add_dependency(task_id, dep_id):
                        print(f"  Added dependency: {task_id} depends on {dep_id}")
                    else:
                        print(f"  Warning: Could not add dependency to {dep_id} (task not found)")

            # Add blocks (edges from this task TO blocked tasks)
            if blocks:
                for blocked_id in blocks:
                    if self.add_blocks(task_id, blocked_id):
                        print(f"  Added blocks: {task_id} blocks {blocked_id}")
                    else:
                        print(f"  Warning: Could not add blocks to {blocked_id} (task not found)")

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

    def get_next_task(self) -> Optional[Dict[str, Any]]:
        """
        Get the next task to work on.

        Selection criteria:
        1. Status must be 'pending' (not in_progress, completed, or blocked)
        2. Highest priority first (critical > high > medium > low)
        3. Oldest task within same priority

        Returns:
            Dict with 'id', 'title', 'priority', 'category' or None if no tasks available
        """
        # Get pending tasks sorted by priority and age
        pending_tasks = self.list_tasks(status=STATUS_PENDING)

        # Filter out blocked tasks (status=pending but have BLOCKS edges pointing to them)
        unblocked_tasks = []
        for task in pending_tasks:
            is_blocked = False
            for edge in self.graph.edges:
                if edge.target_id == task.id and edge.edge_type == EdgeType.BLOCKS:
                    # Check if the blocker is still pending/in_progress
                    blocker = self.graph.nodes.get(edge.source_id)
                    if blocker and blocker.properties.get("status") not in [STATUS_COMPLETED]:
                        is_blocked = True
                        break
            if not is_blocked:
                unblocked_tasks.append(task)

        if not unblocked_tasks:
            return None

        # Return the first (highest priority, oldest) task
        next_task = unblocked_tasks[0]
        return {
            "id": next_task.id,
            "title": next_task.content,
            "priority": next_task.properties.get("priority", PRIORITY_MEDIUM),
            "category": next_task.properties.get("category", "general"),
        }

    def update_task_status(self, task_id: str, status: str) -> bool:
        """Update task status."""
        with self._lock:
            task = self.get_task(task_id)
            if not task:
                return False

            if status not in VALID_STATUSES:
                raise ValueError(f"Invalid status: {status}")

            task.properties["status"] = status
            task.metadata["updated_at"] = datetime.now().isoformat()

            changes = {"status": status, "updated_at": task.metadata["updated_at"]}

            if status == STATUS_COMPLETED:
                task.metadata["completed_at"] = datetime.now().isoformat()
                changes["completed_at"] = task.metadata["completed_at"]

            # Log to event log (source of truth)
            self.event_log.log_node_update(task_id, changes)

            # Log to WAL for crash recovery
            self.wal.log_update_node(task_id, {**task.properties, "_meta": task.metadata})

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
        with self._lock:
            task = self.get_task(task_id)
            if not task:
                return False

            task.properties["status"] = STATUS_COMPLETED
            task.metadata["updated_at"] = datetime.now().isoformat()
            task.metadata["completed_at"] = datetime.now().isoformat()

            changes = {
                "status": STATUS_COMPLETED,
                "updated_at": task.metadata["updated_at"],
                "completed_at": task.metadata["completed_at"],
            }

            if retrospective:
                task.properties["retrospective"] = retrospective
                changes["retrospective"] = retrospective

            # Log to event log (source of truth)
            self.event_log.log_node_update(task_id, changes)

            # Log to WAL for crash recovery
            self.wal.log_update_node(task_id, {**task.properties, "_meta": task.metadata})

            return True

    def block_task(
        self,
        task_id: str,
        reason: str,
        blocker_id: Optional[str] = None,
    ) -> bool:
        """Block a task with reason."""
        with self._lock:
            task = self.get_task(task_id)
            if not task:
                return False

            task.properties["status"] = STATUS_BLOCKED
            task.properties["blocked_reason"] = reason
            task.metadata["updated_at"] = datetime.now().isoformat()

            changes = {
                "status": STATUS_BLOCKED,
                "blocked_reason": reason,
                "updated_at": task.metadata["updated_at"],
            }

            # Log to event log (source of truth)
            self.event_log.log_node_update(task_id, changes)

            # Log to WAL for crash recovery
            self.wal.log_update_node(task_id, {**task.properties, "_meta": task.metadata})

            # Add blocking edge if blocker specified
            if blocker_id:
                if not blocker_id.startswith("task:"):
                    blocker_id = f"task:{blocker_id}"
                if blocker_id in self.graph.nodes:
                    self.graph.add_edge(
                        blocker_id, task_id, EdgeType.BLOCKS,
                        weight=1.0, confidence=1.0
                    )
                    self.event_log.log_edge_create(blocker_id, task_id, "BLOCKS")
                    self.wal.log_add_edge(blocker_id, task_id, EdgeType.BLOCKS)

            return True

    def delete_task(
        self,
        task_id: str,
        force: bool = False,
    ) -> bool:
        """Delete a task with transactional safety checks.

        TRANSACTIONAL: This method verifies pre-conditions before deletion:
        - Task must exist
        - Without --force: fails if task has dependents, blocks others, or is in progress
        - With --force: removes edges and deletes the task

        Args:
            task_id: The task ID to delete
            force: If True, bypass safety checks and force deletion

        Returns:
            True if deleted, False if deletion blocked or task not found
        """
        with self._lock:
            # Normalize task ID
            if not task_id.startswith("task:"):
                task_id = f"task:{task_id}"

            # Transactional check 1: Task must exist
            task = self.get_task(task_id)
            if not task:
                logger.warning(f"Cannot delete non-existent task: {task_id}")
                return False

            # Get task status
            status = task.properties.get("status", STATUS_PENDING)

            if not force:
                # Transactional check 2: Task should not have dependents
                dependents = self.what_depends_on(task_id)
                if dependents:
                    dependent_ids = [d.id for d in dependents]
                    logger.warning(
                        f"Cannot delete task {task_id}: {len(dependents)} task(s) depend on it: "
                        f"{', '.join(dependent_ids[:3])}{'...' if len(dependents) > 3 else ''}"
                    )
                    return False

                # Transactional check 3: Task should not block others
                # Find tasks that this task blocks
                blocked_tasks = []
                for edge in self.graph.edges:
                    if edge.source_id == task_id and edge.edge_type == EdgeType.BLOCKS:
                        blocked_tasks.append(edge.target_id)
                if blocked_tasks:
                    logger.warning(
                        f"Cannot delete task {task_id}: it blocks {len(blocked_tasks)} task(s): "
                        f"{', '.join(blocked_tasks[:3])}{'...' if len(blocked_tasks) > 3 else ''}"
                    )
                    return False

                # Transactional check 4: Task should not be in progress
                if status == STATUS_IN_PROGRESS:
                    logger.warning(
                        f"Cannot delete in-progress task {task_id}. Use --force to override."
                    )
                    return False

            # Remove edges to/from this task
            edges_to_remove = []
            for edge in self.graph.edges:
                if edge.source_id == task_id or edge.target_id == task_id:
                    edges_to_remove.append(edge)

            for edge in edges_to_remove:
                # Log edge deletion
                self.event_log.log_edge_delete(
                    edge.source_id, edge.target_id, edge.edge_type.name
                )
                # Remove from graph
                self.graph.edges.remove(edge)

            # Remove the task from graph
            if task_id in self.graph.nodes:
                del self.graph.nodes[task_id]

            # Log the deletion (source of truth)
            self.event_log.log_node_delete(task_id)

            # Log to WAL for crash recovery
            self.wal.log_remove_node(task_id)

            logger.info(f"Deleted task: {task_id}")
            return True

    def add_dependency(self, task_id: str, depends_on_id: str) -> bool:
        """Add dependency between tasks.

        Args:
            task_id: The task that depends on another task
            depends_on_id: The task that task_id depends on

        Returns:
            True if edge was created, False if either task not found
        """
        with self._lock:
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
            self.event_log.log_edge_create(task_id, depends_on_id, "DEPENDS_ON")
            self.wal.log_add_edge(task_id, depends_on_id, EdgeType.DEPENDS_ON)

            return True

    def add_blocks(self, task_id: str, blocked_id: str) -> bool:
        """Add blocking relationship between tasks.

        Args:
            task_id: The task that blocks another task
            blocked_id: The task that is blocked by task_id

        Returns:
            True if edge was created, False if either task not found
        """
        with self._lock:
            if not task_id.startswith("task:"):
                task_id = f"task:{task_id}"
            if not blocked_id.startswith("task:"):
                blocked_id = f"task:{blocked_id}"

            if task_id not in self.graph.nodes or blocked_id not in self.graph.nodes:
                return False

            self.graph.add_edge(
                task_id, blocked_id, EdgeType.BLOCKS,
                weight=1.0, confidence=1.0
            )
            self.event_log.log_edge_create(task_id, blocked_id, "BLOCKS")
            self.wal.log_add_edge(task_id, blocked_id, EdgeType.BLOCKS)

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

        # Log to event log (source of truth)
        self.event_log.log_node_create(sprint_id, "GOAL", {**properties, "name": name})

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
    # DECISION & REASONING OPERATIONS
    # =========================================================================

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
        with self._lock:
            decision_id = generate_decision_id()

            # Create the decision node
            self.graph.add_node(
                node_id=decision_id,
                node_type=NodeType.CONTEXT,
                content=decision,
                properties={
                    "type": "decision",
                    "rationale": rationale,
                    "alternatives": alternatives or [],
                },
                metadata={
                    "created_at": datetime.now().isoformat(),
                    "branch": self._get_current_branch(),
                    **(context or {}),
                }
            )

            # Log to event log
            self.event_log.log_decision(
                decision_id=decision_id,
                decision=decision,
                rationale=rationale,
                affects=affects,
                alternatives=alternatives,
                context=context,
            )

            # Create edges to affected nodes
            for affected_id in (affects or []):
                if affected_id in self.graph.nodes:
                    self.graph.add_edge(
                        decision_id, affected_id,
                        EdgeType.MOTIVATES,
                        weight=1.0, confidence=1.0
                    )
                    self.event_log.log_edge_create(decision_id, affected_id, "RELATES_TO")

            return decision_id

    def get_decisions(self) -> List[ThoughtNode]:
        """Get all decision nodes."""
        decisions = []
        for node_id, node in self.graph.nodes.items():
            if node_id.startswith("decision:"):
                decisions.append(node)
        return decisions

    def get_decisions_for_task(self, task_id: str) -> List[ThoughtNode]:
        """Get all decisions that affect a task."""
        if not task_id.startswith("task:"):
            task_id = f"task:{task_id}"

        decisions = []
        edges_to = getattr(self.graph, '_edges_to', {})
        for edge in edges_to.get(task_id, []):
            source = self.graph.nodes.get(edge.source_id)
            if source and edge.source_id.startswith("decision:"):
                decisions.append(source)

        return decisions

    def why(self, node_id: str) -> List[Dict[str, Any]]:
        """Query: Why was this node created/modified this way?

        Returns all decisions that affect this node with their rationale.
        """
        decisions = self.get_decisions_for_task(node_id)
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

    # =========================================================================
    # AUTO-EDGE INFERENCE
    # =========================================================================

    def infer_edges_from_commit(self, commit_message: str, files_changed: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Infer edges from a commit message.

        Parses commit messages for task references and creates edges:
        - "task:T-..." references → IMPLEMENTS edge
        - "depends on task:T-..." → DEPENDS_ON edge
        - "blocks task:T-..." → BLOCKS edge
        - "closes task:T-..." → COMPLETES edge (marks task complete)

        Args:
            commit_message: The commit message to parse
            files_changed: Optional list of files changed in commit

        Returns:
            List of edges created
        """
        import re

        edges_created = []

        # Find all task references
        task_refs = re.findall(r'task:T-[\w-]+', commit_message, re.IGNORECASE)

        # Find specific relationship patterns
        depends_pattern = re.findall(r'depends on (task:T-[\w-]+)', commit_message, re.IGNORECASE)
        blocks_pattern = re.findall(r'blocks (task:T-[\w-]+)', commit_message, re.IGNORECASE)
        closes_pattern = re.findall(r'(?:closes?|fixes?|resolves?) (task:T-[\w-]+)', commit_message, re.IGNORECASE)

        # Create a commit node for context
        commit_id = f"commit:{datetime.now().strftime('%Y%m%d-%H%M%S')}-{os.urandom(2).hex()}"
        self.graph.add_node(
            node_id=commit_id,
            node_type=NodeType.CONTEXT,
            content=commit_message[:100],
            properties={
                "type": "commit",
                "message": commit_message,
                "files": files_changed or [],
            },
            metadata={"created_at": datetime.now().isoformat()}
        )
        self.event_log.log_node_create(commit_id, "CONTEXT", {
            "type": "commit",
            "message": commit_message,
        })

        # Create IMPLEMENTS edges for all referenced tasks
        for task_id in task_refs:
            task_id_lower = task_id.lower()
            # Find the actual task ID (case-insensitive match)
            actual_id = None
            for node_id in self.graph.nodes:
                if node_id.lower() == task_id_lower:
                    actual_id = node_id
                    break

            if actual_id:
                self.graph.add_edge(
                    commit_id, actual_id,
                    EdgeType.MOTIVATES,
                    weight=1.0, confidence=1.0
                )
                self.event_log.log_edge_create(commit_id, actual_id, "RELATES_TO")
                edges_created.append({
                    "type": "IMPLEMENTS",
                    "from": commit_id,
                    "to": actual_id,
                })

        # Handle dependencies
        for dep_ref in depends_pattern:
            dep_id = dep_ref.lower()
            for node_id in self.graph.nodes:
                if node_id.lower() == dep_id:
                    # The first task mentioned depends on this one
                    if task_refs:
                        first_task = task_refs[0]
                        for n_id in self.graph.nodes:
                            if n_id.lower() == first_task.lower():
                                self.add_dependency(n_id, node_id)
                                edges_created.append({
                                    "type": "DEPENDS_ON",
                                    "from": n_id,
                                    "to": node_id,
                                })
                                break
                    break

        # Handle closes/fixes (mark tasks complete)
        for close_ref in closes_pattern:
            for node_id in self.graph.nodes:
                if node_id.lower() == close_ref.lower():
                    self.complete_task(node_id, retrospective=f"Closed via commit: {commit_message[:50]}")
                    edges_created.append({
                        "type": "CLOSES",
                        "commit": commit_id,
                        "task": node_id,
                    })
                    break

        return edges_created

    def infer_edges_from_recent_commits(self, count: int = 10) -> List[Dict[str, Any]]:
        """Infer edges from recent git commits.

        Reads the last N commits and creates edges for any task references.
        """
        import subprocess

        try:
            result = subprocess.run(
                ["git", "log", f"-{count}", "--pretty=format:%H|%s"],
                capture_output=True, text=True, check=True
            )
        except Exception:
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

    def what_blocks(self, task_id: str) -> List[ThoughtNode]:
        """Query: What tasks are blocking this task?

        Follows BLOCKS edges pointing TO this task.
        """
        if not task_id.startswith("task:"):
            task_id = f"task:{task_id}"

        blockers = []
        # Check _edges_to for edges pointing to this task
        edges_to = getattr(self.graph, '_edges_to', {})
        for edge in edges_to.get(task_id, []):
            if edge.edge_type == EdgeType.BLOCKS:
                blocker = self.graph.nodes.get(edge.source_id)
                if blocker:
                    blockers.append(blocker)

        return blockers

    def what_depends_on(self, task_id: str) -> List[ThoughtNode]:
        """Query: What tasks depend on this task?

        Follows DEPENDS_ON edges pointing TO this task
        (i.e., other tasks that have this task as a dependency).
        """
        if not task_id.startswith("task:"):
            task_id = f"task:{task_id}"

        dependents = []
        # Check _edges_to for edges pointing to this task
        edges_to = getattr(self.graph, '_edges_to', {})
        for edge in edges_to.get(task_id, []):
            if edge.edge_type == EdgeType.DEPENDS_ON:
                dependent = self.graph.nodes.get(edge.source_id)
                if dependent:
                    dependents.append(dependent)

        return dependents

    def find_path(
        self,
        from_id: str,
        to_id: str,
        max_depth: int = 10,
    ) -> Optional[List[ThoughtNode]]:
        """Query: Find a path between two nodes.

        Uses BFS to find shortest path following any edge type.
        Returns None if no path exists.
        """
        from collections import deque

        if from_id not in self.graph.nodes or to_id not in self.graph.nodes:
            return None

        if from_id == to_id:
            return [self.graph.nodes[from_id]]

        # BFS
        queue = deque([(from_id, [from_id])])
        visited = {from_id}

        while queue:
            current_id, path = queue.popleft()

            if len(path) > max_depth:
                continue

            # Get outgoing edges
            edges_from = self.graph._edges_from.get(current_id, [])
            for edge in edges_from:
                next_id = edge.target_id
                if next_id == to_id:
                    return [self.graph.nodes[nid] for nid in path + [next_id]]
                if next_id not in visited:
                    visited.add(next_id)
                    queue.append((next_id, path + [next_id]))

        return None

    def get_all_relationships(self, task_id: str) -> Dict[str, List[ThoughtNode]]:
        """Query: Get all relationships for a task.

        Returns dict with keys:
        - 'blocks': Tasks this task blocks
        - 'blocked_by': Tasks blocking this task
        - 'depends_on': Tasks this task depends on
        - 'depended_by': Tasks depending on this task
        - 'in_sprint': Sprint containing this task
        """
        if not task_id.startswith("task:"):
            task_id = f"task:{task_id}"

        result = {
            'blocks': [],
            'blocked_by': [],
            'depends_on': [],
            'depended_by': [],
            'in_sprint': [],
        }

        # Outgoing edges (from this task)
        edges_from = self.graph._edges_from.get(task_id, [])
        for edge in edges_from:
            target = self.graph.nodes.get(edge.target_id)
            if not target:
                continue
            if edge.edge_type == EdgeType.BLOCKS:
                result['blocks'].append(target)
            elif edge.edge_type == EdgeType.DEPENDS_ON:
                result['depends_on'].append(target)

        # Incoming edges (to this task)
        edges_to = getattr(self.graph, '_edges_to', {})
        for edge in edges_to.get(task_id, []):
            source = self.graph.nodes.get(edge.source_id)
            if not source:
                continue
            if edge.edge_type == EdgeType.BLOCKS:
                result['blocked_by'].append(source)
            elif edge.edge_type == EdgeType.DEPENDS_ON:
                result['depended_by'].append(source)
            elif edge.edge_type == EdgeType.CONTAINS:
                result['in_sprint'].append(source)

        return result

    def query(self, query_str: str) -> List[Dict[str, Any]]:
        """Simple query language for the graph.

        Supported queries:
        - "what blocks <task_id>"
        - "what depends on <task_id>"
        - "path from <id1> to <id2>"
        - "relationships <task_id>"
        - "blocked tasks"
        - "active tasks"
        - "pending tasks"

        Returns list of result dicts.
        """
        query_str = query_str.strip().lower()
        results = []

        if query_str.startswith("what blocks "):
            task_id = query_str[12:].strip()
            for node in self.what_blocks(task_id):
                results.append({
                    "id": node.id,
                    "title": node.content,
                    "status": node.properties.get("status"),
                    "relation": "blocks",
                })

        elif query_str.startswith("what depends on "):
            task_id = query_str[16:].strip()
            for node in self.what_depends_on(task_id):
                results.append({
                    "id": node.id,
                    "title": node.content,
                    "status": node.properties.get("status"),
                    "relation": "depends_on",
                })

        elif query_str.startswith("path from "):
            # Parse "path from X to Y"
            parts = query_str[10:].split(" to ")
            if len(parts) == 2:
                from_id, to_id = parts[0].strip(), parts[1].strip()
                path = self.find_path(from_id, to_id)
                if path:
                    for i, node in enumerate(path):
                        results.append({
                            "step": i,
                            "id": node.id,
                            "title": node.content,
                        })

        elif query_str.startswith("relationships "):
            task_id = query_str[14:].strip()
            rels = self.get_all_relationships(task_id)
            for rel_type, nodes in rels.items():
                for node in nodes:
                    results.append({
                        "relation": rel_type,
                        "id": node.id,
                        "title": node.content,
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

        return results

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
        blocks=getattr(args, 'blocks', None),
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


def cmd_task_next(args, manager: GoTProjectManager) -> int:
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


def cmd_task_show(args, manager: GoTProjectManager) -> int:
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


def cmd_task_delete(args, manager: GoTProjectManager) -> int:
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


def cmd_dashboard(args, manager: GoTProjectManager) -> int:
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


def cmd_migrate_events(args, manager: GoTProjectManager) -> int:
    """Migrate existing snapshot to event-sourced format.

    This creates event log entries from the current graph state,
    enabling merge-friendly cross-branch coordination.
    """
    import gzip

    dry_run = getattr(args, 'dry_run', False)

    # Check if events already exist
    existing_events = EventLog.load_all_events(manager.events_dir)
    if existing_events and not getattr(args, 'force', False):
        print(f"Events already exist ({len(existing_events)} events).")
        print("Use --force to migrate anyway (will append, not replace).")
        return 1

    # Create migration event log
    migration_log = EventLog(
        manager.events_dir,
        session_id=f"migration-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )

    nodes_migrated = 0
    edges_migrated = 0

    # Migrate all nodes
    for node_id, node in manager.graph.nodes.items():
        if dry_run:
            print(f"  Would migrate node: {node_id}")
        else:
            migration_log.log_node_create(
                node_id=node_id,
                node_type=node.node_type.name if hasattr(node.node_type, 'name') else str(node.node_type),
                data={
                    **node.properties,
                    "title": node.content,
                }
            )
        nodes_migrated += 1

    # Migrate all edges
    edges = manager.graph.edges
    if isinstance(edges, dict):
        edge_list = edges.values()
    else:
        edge_list = edges

    for edge in edge_list:
        if dry_run:
            src = getattr(edge, 'source_id', edge.get('source_id', '?') if isinstance(edge, dict) else '?')
            tgt = getattr(edge, 'target_id', edge.get('target_id', '?') if isinstance(edge, dict) else '?')
            print(f"  Would migrate edge: {src} -> {tgt}")
        else:
            src = getattr(edge, 'source_id', edge.get('source_id') if isinstance(edge, dict) else None)
            tgt = getattr(edge, 'target_id', edge.get('target_id') if isinstance(edge, dict) else None)
            etype = getattr(edge, 'edge_type', edge.get('edge_type') if isinstance(edge, dict) else None)
            weight = getattr(edge, 'weight', edge.get('weight', 1.0) if isinstance(edge, dict) else 1.0)

            if src and tgt:
                migration_log.log_edge_create(
                    src=src,
                    tgt=tgt,
                    edge_type=etype.name if hasattr(etype, 'name') else str(etype),
                    weight=weight
                )
        edges_migrated += 1

    if dry_run:
        print(f"\nDry run complete:")
    else:
        print(f"\nMigration complete:")

    print(f"  Nodes: {nodes_migrated}")
    print(f"  Edges: {edges_migrated}")
    print(f"  Event file: {migration_log.event_file}")
    print(f"\nEvents are now the source of truth.")
    print("Commit .got/events/ to make this survive across environments.")

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
# BACKUP COMMANDS
# =============================================================================


def cmd_backup_create(args, manager: GoTProjectManager) -> int:
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


def cmd_backup_list(args, manager: GoTProjectManager) -> int:
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


def cmd_backup_verify(args, manager: GoTProjectManager) -> int:
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


def cmd_backup_restore(args, manager: GoTProjectManager) -> int:
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


def cmd_handoff_initiate(args, manager: GoTProjectManager) -> int:
    """Initiate a handoff to another agent."""
    task = manager.get_task(args.task_id)
    if not task:
        print(f"Task not found: {args.task_id}")
        return 1

    handoff_mgr = HandoffManager(manager.event_log)
    handoff_id = handoff_mgr.initiate_handoff(
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


def cmd_handoff_accept(args, manager: GoTProjectManager) -> int:
    """Accept a handoff."""
    handoff_mgr = HandoffManager(manager.event_log)
    handoff_mgr.accept_handoff(
        handoff_id=args.handoff_id,
        agent=args.agent,
        acknowledgment=args.message,
    )

    print(f"Handoff accepted: {args.handoff_id}")
    print(f"  Agent: {args.agent}")
    return 0


def cmd_handoff_complete(args, manager: GoTProjectManager) -> int:
    """Complete a handoff."""
    try:
        result = json.loads(args.result)
    except json.JSONDecodeError:
        result = {"message": args.result}

    handoff_mgr = HandoffManager(manager.event_log)
    handoff_mgr.complete_handoff(
        handoff_id=args.handoff_id,
        agent=args.agent,
        result=result,
        artifacts=args.artifacts or [],
    )

    print(f"Handoff completed: {args.handoff_id}")
    print(f"  Agent: {args.agent}")
    print(f"  Result: {json.dumps(result, indent=2)}")
    return 0


def cmd_handoff_list(args, manager: GoTProjectManager) -> int:
    """List handoffs."""
    events = EventLog.load_all_events(manager.events_dir)
    handoffs = HandoffManager.load_handoffs_from_events(events)

    if args.status:
        handoffs = [h for h in handoffs if h.get("status") == args.status]

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


def cmd_decision_log(args, manager: GoTProjectManager) -> int:
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


def cmd_decision_list(args, manager: GoTProjectManager) -> int:
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


def cmd_decision_why(args, manager: GoTProjectManager) -> int:
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


def cmd_infer(args, manager: GoTProjectManager) -> int:
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


def cmd_validate(args, manager: GoTProjectManager) -> int:
    """Validate graph health and report issues."""
    print("=" * 60)
    print("GoT VALIDATION REPORT")
    print("=" * 60)

    issues = []
    warnings = []

    # Count nodes and edges
    total_nodes = len(manager.graph.nodes)
    total_edges = len(manager.graph.edges)

    # Count tasks by status
    tasks = [n for n in manager.graph.nodes.values() if n.node_type == NodeType.TASK]
    task_count = len(tasks)

    # Check for orphan nodes (no edges)
    nodes_with_edges = set()
    for edge in manager.graph.edges:
        nodes_with_edges.add(edge.source_id)
        nodes_with_edges.add(edge.target_id)

    orphan_count = total_nodes - len(nodes_with_edges)
    orphan_rate = orphan_count / max(total_nodes, 1) * 100

    # Load events and compare
    events = EventLog.load_all_events(manager.events_dir)
    event_edge_count = sum(1 for e in events if e.get('event') == 'edge.create')
    event_node_count = sum(1 for e in events if e.get('event') == 'node.create')

    # Check for edge loss (the bug we fixed)
    edge_loss_rate = 0
    if event_edge_count > 0:
        edge_loss_rate = (1 - total_edges / event_edge_count) * 100
        if edge_loss_rate > 10:
            issues.append(f"EDGE LOSS: {edge_loss_rate:.1f}% of edges from events not in graph ({total_edges}/{event_edge_count})")
        elif edge_loss_rate > 0:
            warnings.append(f"Minor edge loss: {edge_loss_rate:.1f}% ({total_edges}/{event_edge_count})")

    # Check orphan rate
    if orphan_rate > 50:
        issues.append(f"HIGH ORPHAN RATE: {orphan_rate:.1f}% of nodes have no edges")
    elif orphan_rate > 25:
        warnings.append(f"Moderate orphan rate: {orphan_rate:.1f}%")

    # Check edge density
    edge_density = total_edges / max(total_nodes, 1)
    if edge_density < 0.1:
        warnings.append(f"Low edge density: {edge_density:.2f} edges/node")

    # Print stats
    print(f"\n📊 STATISTICS")
    print(f"   Nodes: {total_nodes}")
    print(f"   Tasks: {task_count}")
    print(f"   Edges: {total_edges}")
    print(f"   Edge density: {edge_density:.2f} edges/node")
    print(f"   Orphan nodes: {orphan_count} ({orphan_rate:.1f}%)")

    print(f"\n📁 EVENT LOG")
    print(f"   Node events: {event_node_count}")
    print(f"   Edge events: {event_edge_count}")
    if edge_loss_rate > 0:
        print(f"   Edge rebuild rate: {100 - edge_loss_rate:.1f}%")
    else:
        print(f"   Edge rebuild rate: 100%")

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


def cmd_query(args, manager: GoTProjectManager) -> int:
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


def cmd_compact(args, manager: GoTProjectManager) -> int:
    """Compact old events."""
    preserve_handoffs = not getattr(args, 'no_preserve_handoffs', False)
    preserve_days = getattr(args, 'preserve_days', 7)
    dry_run = getattr(args, 'dry_run', False)

    if dry_run:
        print(f"Dry run - would compact events older than {preserve_days} days")
        print(f"  Preserve handoffs: {preserve_handoffs}")

        # Load and analyze events
        events = EventLog.load_all_events(manager.events_dir)
        cutoff = datetime.utcnow() - __import__('datetime').timedelta(days=preserve_days)
        cutoff_str = cutoff.isoformat() + "Z"

        old_events = [e for e in events if e.get("ts", "") < cutoff_str]
        handoff_events = [e for e in events if e.get("event", "").startswith("handoff.")]
        recent_events = [e for e in events if e.get("ts", "") >= cutoff_str]

        print(f"\nAnalysis:")
        print(f"  Total events: {len(events)}")
        print(f"  Old events (would compact): {len(old_events)}")
        print(f"  Recent events (would keep): {len(recent_events)}")
        print(f"  Handoff events: {len(handoff_events)}")
        return 0

    result = EventLog.compact_events(
        manager.events_dir,
        preserve_handoffs=preserve_handoffs,
        preserve_days=preserve_days,
    )

    if result.get("error"):
        print(f"Error: {result['error']}")
        return 1

    if result.get("status") == "nothing_to_compact":
        print("Nothing to compact - all events are recent.")
        return 0

    print("Event compaction complete:")
    print(f"  Nodes written: {result.get('nodes_written', 0)}")
    print(f"  Edges written: {result.get('edges_written', 0)}")
    print(f"  Handoffs preserved: {result.get('handoffs_preserved', 0)}")
    print(f"  Files removed: {result.get('files_removed', 0)}")
    print(f"  Compact file: {result.get('compact_file', '?')}")
    print(f"\n  Original events: {result.get('original_event_count', 0)}")
    print(f"  Old events consolidated: {result.get('old_events_consolidated', 0)}")
    print(f"  Recent events kept: {result.get('recent_events_kept', 0)}")

    return 0


def cmd_sync(args, manager: GoTProjectManager) -> int:
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
    create_parser.add_argument("--depends-on", "--depends", nargs="+", dest="depends", help="Task IDs this task depends on")
    create_parser.add_argument("--blocks", nargs="+", help="Task IDs this task blocks")

    # task list
    list_parser = task_subparsers.add_parser("list", help="List tasks")
    list_parser.add_argument("--status", choices=VALID_STATUSES)
    list_parser.add_argument("--priority", choices=VALID_PRIORITIES)
    list_parser.add_argument("--category", choices=VALID_CATEGORIES)
    list_parser.add_argument("--sprint", help="Filter by sprint")
    list_parser.add_argument("--blocked", action="store_true", help="Show only blocked")
    list_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # task show
    show_parser = task_subparsers.add_parser("show", help="Show task details")
    show_parser.add_argument("task_id", help="Task ID to display")

    # task next
    next_parser = task_subparsers.add_parser("next", help="Get the next task to work on")
    next_parser.add_argument("--start", "-s", action="store_true",
                             help="Also start the task after selecting it")

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

    # task delete
    delete_parser = task_subparsers.add_parser("delete", help="Delete a task (transactional)")
    delete_parser.add_argument("task_id", help="Task ID to delete")
    delete_parser.add_argument("--force", "-f", action="store_true",
                               help="Force delete even if task has dependencies or is in progress")

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
    subparsers.add_parser("dashboard", help="Show comprehensive metrics dashboard")

    # Migration commands
    migrate_parser = subparsers.add_parser("migrate", help="Migrate from files")
    migrate_parser.add_argument("--dry-run", action="store_true", help="Don't actually migrate")

    # Migrate to events command (convert snapshot to event-sourced format)
    migrate_events_parser = subparsers.add_parser("migrate-events",
        help="Convert snapshot to event-sourced format for cross-branch coordination")
    migrate_events_parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated")
    migrate_events_parser.add_argument("--force", "-f", action="store_true", help="Migrate even if events exist")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export graph")
    export_parser.add_argument("--output", "-o", help="Output file")

    # Backup commands
    backup_parser = subparsers.add_parser("backup", help="Backup and recovery")
    backup_subparsers = backup_parser.add_subparsers(dest="backup_command")

    # backup create
    backup_create = backup_subparsers.add_parser("create", help="Create a snapshot")
    backup_create.add_argument("--compress", "-c", action="store_true", default=True,
                               help="Compress snapshot (default: true)")

    # backup list
    backup_list = backup_subparsers.add_parser("list", help="List available snapshots")
    backup_list.add_argument("--limit", "-n", type=int, default=10, help="Number to show")

    # backup verify
    backup_verify = backup_subparsers.add_parser("verify", help="Verify snapshot integrity")
    backup_verify.add_argument("snapshot_id", nargs="?", help="Snapshot ID (default: latest)")

    # backup restore
    backup_restore = backup_subparsers.add_parser("restore", help="Restore from snapshot")
    backup_restore.add_argument("snapshot_id", help="Snapshot ID to restore")
    backup_restore.add_argument("--force", "-f", action="store_true",
                                help="Force restore without confirmation")

    # Sync command (critical for environment resilience)
    sync_parser = subparsers.add_parser("sync", help="Sync state to git-tracked snapshot")
    sync_parser.add_argument("--message", "-m", help="Commit message (auto-commits if provided)")

    # Handoff commands (for agent coordination)
    handoff_parser = subparsers.add_parser("handoff", help="Agent handoff operations")
    handoff_subparsers = handoff_parser.add_subparsers(dest="handoff_command")

    # handoff initiate
    handoff_init = handoff_subparsers.add_parser("initiate", help="Initiate a handoff to another agent")
    handoff_init.add_argument("task_id", help="Task to hand off")
    handoff_init.add_argument("--target", "-t", required=True, help="Target agent (e.g., 'sub-agent-1')")
    handoff_init.add_argument("--source", "-s", default="main", help="Source agent (default: main)")
    handoff_init.add_argument("--instructions", "-i", default="", help="Instructions for target agent")

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
    handoff_complete.add_argument("--artifacts", nargs="*", help="Artifacts created (files, commits)")

    # handoff list
    handoff_list = handoff_subparsers.add_parser("list", help="List handoffs")
    handoff_list.add_argument("--status", choices=["initiated", "accepted", "completed", "rejected"])

    # Compaction command
    compact_parser = subparsers.add_parser("compact", help="Compact old events into consolidated file")
    compact_parser.add_argument("--preserve-days", "-d", type=int, default=7,
                                help="Preserve events from last N days (default: 7)")
    compact_parser.add_argument("--no-preserve-handoffs", action="store_true",
                                help="Don't preserve handoff events")
    compact_parser.add_argument("--dry-run", action="store_true",
                                help="Show what would be compacted")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the graph")
    query_parser.add_argument("query_string", nargs="+", help="Query (e.g., 'what blocks task:T-...')")

    # Decision commands (Reasoning Trace Logger)
    decision_parser = subparsers.add_parser("decision", help="Log decisions with rationale")
    decision_subparsers = decision_parser.add_subparsers(dest="decision_command")

    # decision log
    decision_log = decision_subparsers.add_parser("log", help="Log a decision")
    decision_log.add_argument("decision", help="What was decided")
    decision_log.add_argument("--rationale", "-r", required=True, help="Why this choice was made")
    decision_log.add_argument("--affects", "-a", nargs="+", help="Task IDs affected by this decision")
    decision_log.add_argument("--alternatives", nargs="+", help="Alternatives considered")
    decision_log.add_argument("--file", "-f", help="File this decision relates to")

    # decision list
    decision_list = decision_subparsers.add_parser("list", help="List all decisions")

    # decision why
    decision_why = decision_subparsers.add_parser("why", help="Ask why a task exists")
    decision_why.add_argument("task_id", help="Task ID to query")

    # Validation command
    validate_parser = subparsers.add_parser("validate", help="Validate graph health")

    # Edge inference commands
    infer_parser = subparsers.add_parser("infer", help="Infer edges from git history")
    infer_parser.add_argument("--commits", "-n", type=int, default=10,
                              help="Number of recent commits to analyze")
    infer_parser.add_argument("--message", "-m", help="Analyze a specific commit message")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Initialize manager (use transactional backend if available)
    if USE_TX_BACKEND and TX_BACKEND_AVAILABLE:
        try:
            manager = TransactionalGoTAdapter(GOT_TX_DIR)
            if os.environ.get("GOT_DEBUG"):
                print(f"[DEBUG] Using transactional backend at {GOT_TX_DIR}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Failed to initialize transactional backend: {e}", file=sys.stderr)
            print("Falling back to event-sourced backend", file=sys.stderr)
            manager = GoTProjectManager()
    else:
        manager = GoTProjectManager()

    # Route commands
    if args.command == "task":
        if args.task_command == "create":
            return cmd_task_create(args, manager)
        elif args.task_command == "list":
            return cmd_task_list(args, manager)
        elif args.task_command == "show":
            return cmd_task_show(args, manager)
        elif args.task_command == "next":
            return cmd_task_next(args, manager)
        elif args.task_command == "start":
            return cmd_task_start(args, manager)
        elif args.task_command == "complete":
            return cmd_task_complete(args, manager)
        elif args.task_command == "block":
            return cmd_task_block(args, manager)
        elif args.task_command == "delete":
            return cmd_task_delete(args, manager)
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

    elif args.command == "dashboard":
        return cmd_dashboard(args, manager)

    elif args.command == "migrate":
        return cmd_migrate(args, manager)

    elif args.command == "migrate-events":
        return cmd_migrate_events(args, manager)

    elif args.command == "export":
        return cmd_export(args, manager)

    elif args.command == "backup":
        if args.backup_command == "create":
            return cmd_backup_create(args, manager)
        elif args.backup_command == "list":
            return cmd_backup_list(args, manager)
        elif args.backup_command == "verify":
            return cmd_backup_verify(args, manager)
        elif args.backup_command == "restore":
            return cmd_backup_restore(args, manager)
        else:
            backup_parser.print_help()
            return 1

    elif args.command == "sync":
        return cmd_sync(args, manager)

    elif args.command == "handoff":
        if args.handoff_command == "initiate":
            return cmd_handoff_initiate(args, manager)
        elif args.handoff_command == "accept":
            return cmd_handoff_accept(args, manager)
        elif args.handoff_command == "complete":
            return cmd_handoff_complete(args, manager)
        elif args.handoff_command == "list":
            return cmd_handoff_list(args, manager)
        else:
            handoff_parser.print_help()
            return 1

    elif args.command == "compact":
        return cmd_compact(args, manager)

    elif args.command == "query":
        return cmd_query(args, manager)

    elif args.command == "decision":
        if args.decision_command == "log":
            return cmd_decision_log(args, manager)
        elif args.decision_command == "list":
            return cmd_decision_list(args, manager)
        elif args.decision_command == "why":
            return cmd_decision_why(args, manager)
        else:
            decision_parser.print_help()
            return 1

    elif args.command == "infer":
        return cmd_infer(args, manager)

    elif args.command == "validate":
        return cmd_validate(args, manager)

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
