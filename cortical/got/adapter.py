"""
CLI Adapter for GoT system.

Provides a thin wrapper around GoTManager with convenience methods
for CLI operations. This eliminates the need for the large
TransactionalGoTAdapter in scripts/got_utils.py.

# TODO(adapter-retirement): THIS FILE IS BEING RETIRED
# Plan: docs/design/transactional-adapter-retirement-plan.md
# Progress: docs/sessions/adapter-retirement-progress.md
#
# Methods will be moved to GoTManager or new modules.
# See TODO comments on each section for disposition.
"""

import json
import logging
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cortical.core.bootstrap import create_container
from cortical.got.api import GoTManager
from cortical.got.types import Task, Decision, Sprint, Epic, KnowledgeTransfer
from cortical.utils.id_generation import generate_kt_id

logger = logging.getLogger(__name__)

# Project root for default paths
_PROJECT_ROOT = Path(__file__).parent.parent.parent

# Allow GOT_DIR to be overridden via environment variable (for testing)
GOT_DIR = Path(os.environ.get("GOT_DIR", _PROJECT_ROOT / ".got"))


class TransactionalGoTAdapter:
    """
    Thin CLI adapter wrapping GoTManager.

    Provides convenience methods expected by CLI handlers while
    delegating actual operations to GoTManager.
    """

    def __init__(self, got_dir: Path = GOT_DIR):
        self.got_dir = Path(got_dir)
        # Use container to get properly configured GoTManager
        container = create_container(got_dir=self.got_dir)
        self._manager = container.resolve(GoTManager)

    def save(self) -> None:
        """No-op - GoTManager auto-saves. Kept for CLI compatibility."""
        pass

    # =========================================================================
    # Task Operations
    # TODO(adapter-retirement): PHASE 1-2
    # REVIEWED 2026-01-09: Sub-agent analysis complete
    #
    # - create_task: KEEP - adds session_id/branch metadata
    # - get_task, list_all_tasks, update_task, delete_task: PURE DELEGATION - remove
    # - list_tasks: HAS LOGIC (sprint/category filtering) - evaluate where filtering goes
    # - start_task, complete_task, block_task: MOVE TO GoTManager (Phase 1)
    #   **CRITICAL: complete_task MUST move first - git inference depends on it**
    # - add_dependency, add_blocks, add_edge, list_edges, get_edges_for_task: PURE DELEGATION - remove
    # - get_task_sprint, get_task_dependencies: MOVE TO GoTManager (Phase 2)
    # - what_blocks, what_depends_on, get_blockers, get_dependents: ALREADY IN GoTManager - remove
    # - get_active_tasks, get_blocked_tasks, get_next_task: MOVE TO GoTManager (Phase 2)
    # =========================================================================

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
                try:
                    self._manager.add_dependency(task.id, dep_id)
                except Exception as e:
                    logger.warning(f"Could not add dependency to {dep_id}: {e}")

        # Add blocks
        if blocks:
            for blocked_id in blocks:
                try:
                    self._manager.add_blocks(task.id, blocked_id)
                except Exception as e:
                    logger.warning(f"Could not add blocks to {blocked_id}: {e}")

        # Add to sprint if specified
        if sprint_id:
            try:
                self._manager.add_edge(sprint_id, task.id, "CONTAINS")
            except Exception as e:
                logger.warning(f"Could not add task to sprint {sprint_id}: {e}")

        return task.id

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self._manager.get_task(task_id)

    def list_tasks(
        self,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        category: Optional[str] = None,
        sprint_id: Optional[str] = None,
        blocked_only: bool = False,
    ) -> List[Task]:
        """List tasks with optional filters."""
        # Get tasks from manager
        if sprint_id:
            # Get tasks from sprint via edges
            sprint_task_ids = set()
            all_edges = self._manager.list_edges()
            for edge in all_edges:
                if (edge.source_id == sprint_id and
                    edge.edge_type == "CONTAINS" and
                    edge.target_id.startswith("T-")):
                    sprint_task_ids.add(edge.target_id)

            if not sprint_task_ids:
                return []

            all_tasks = self._manager.find_tasks(status=status, priority=priority)
            tasks = [t for t in all_tasks if t.id in sprint_task_ids]
        else:
            tasks = self._manager.find_tasks(status=status, priority=priority)

        # Filter by category
        if category:
            tasks = [t for t in tasks if t.properties.get("category") == category]

        return tasks

    def list_all_tasks(self) -> List[Task]:
        """List all tasks."""
        return self._manager.list_all_tasks()

    def update_task(self, task_id: str, **updates) -> bool:
        """Update a task."""
        try:
            self._manager.update_task(task_id, **updates)
            return True
        except Exception as e:
            logger.error(f"Failed to update task {task_id}: {e}")
            return False

    def start_task(self, task_id: str) -> bool:
        """Start a task (set status to in_progress)."""
        try:
            task = self._manager.get_task(task_id)
            if not task:
                return False
            task.metadata["started_at"] = datetime.now(timezone.utc).isoformat()
            self._manager.update_task(task_id, status="in_progress", metadata=task.metadata)
            return True
        except Exception as e:
            logger.error(f"Failed to start task {task_id}: {e}")
            return False

    def complete_task(self, task_id: str, retrospective: str = "") -> bool:
        """Complete a task."""
        try:
            task = self._manager.get_task(task_id)
            if not task:
                return False
            task.metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
            updates = {"status": "completed", "metadata": task.metadata}
            if retrospective:
                props = dict(task.properties)
                props["retrospective"] = retrospective
                updates["properties"] = props
            self._manager.update_task(task_id, **updates)
            return True
        except Exception as e:
            logger.error(f"Failed to complete task {task_id}: {e}")
            return False

    def block_task(self, task_id: str, reason: str = "", blocked_by: Optional[str] = None) -> bool:
        """Block a task."""
        try:
            task = self._manager.get_task(task_id)
            if not task:
                return False
            props = dict(task.properties)
            props["blocked_reason"] = reason or "No reason given"
            self._manager.update_task(task_id, status="blocked", properties=props)
            if blocked_by:
                self._manager.add_blocks(blocked_by, task_id)
            return True
        except Exception as e:
            logger.error(f"Failed to block task {task_id}: {e}")
            return False

    def delete_task(self, task_id: str, force: bool = False) -> Tuple[bool, str]:
        """Delete a task."""
        try:
            self._manager.delete_task(task_id, force=force)
            return True, f"Task {task_id} deleted"
        except Exception as e:
            return False, str(e)

    def add_dependency(self, task_id: str, depends_on_id: str) -> bool:
        """Add a dependency edge."""
        try:
            self._manager.add_dependency(task_id, depends_on_id)
            return True
        except Exception as e:
            logger.error(f"Failed to add dependency: {e}")
            return False

    def add_blocks(self, task_id: str, blocks_id: str) -> bool:
        """Add a blocks edge."""
        try:
            self._manager.add_blocks(task_id, blocks_id)
            return True
        except Exception as e:
            logger.error(f"Failed to add blocks edge: {e}")
            return False

    def add_edge(self, source_id: str, target_id: str, edge_type: str,
                 weight: float = 1.0, reason: str = "", validate_refs: bool = True):
        """Add a generic edge."""
        try:
            return self._manager.add_edge(
                source_id, target_id, edge_type,
                weight=weight, reason=reason, validate_refs=validate_refs
            )
        except Exception as e:
            logger.error(f"Failed to add edge: {e}")
            return None

    def list_edges(self) -> List:
        """List all edges."""
        return self._manager.list_edges()

    def get_edges_for_task(self, task_id: str) -> Tuple[List, List]:
        """Get edges for a task."""
        return self._manager.get_edges_for_task(task_id)

    def get_task_sprint(self, task_id: str) -> Optional[Dict[str, str]]:
        """Get the sprint containing this task."""
        _, incoming = self._manager.get_edges_for_task(task_id)
        for edge in incoming:
            if edge.edge_type == "CONTAINS" and edge.source_id.startswith("S-"):
                sprint = self._manager.get_sprint(edge.source_id)
                if sprint:
                    return {'id': sprint.id, 'name': sprint.title}
        return None

    def get_task_dependencies(self, task_id: str) -> List[Task]:
        """Get tasks this task depends on."""
        outgoing, _ = self._manager.get_edges_for_task(task_id)
        deps = []
        for edge in outgoing:
            if edge.edge_type == "DEPENDS_ON":
                task = self._manager.get_task(edge.target_id)
                if task:
                    deps.append(task)
        return deps

    def what_blocks(self, task_id: str) -> List[Task]:
        """Get tasks blocking this task."""
        return self._manager.get_blockers(task_id)

    def what_depends_on(self, task_id: str) -> List[Task]:
        """Get tasks that depend on this task."""
        return self._manager.get_dependents(task_id)

    def get_blockers(self, task_id: str) -> List[Task]:
        """Alias for what_blocks."""
        return self.what_blocks(task_id)

    def get_dependents(self, task_id: str) -> List[Task]:
        """Alias for what_depends_on."""
        return self.what_depends_on(task_id)

    def get_active_tasks(self) -> List[Task]:
        """Get in-progress tasks."""
        return self._manager.find_tasks(status="in_progress")

    def get_blocked_tasks(self) -> List[Tuple[Task, Optional[str]]]:
        """Get blocked tasks with reasons."""
        tasks = self._manager.find_tasks(status="blocked")
        return [(t, t.properties.get("blocked_reason", "No reason given")) for t in tasks]

    def get_next_task(self) -> Optional[Dict[str, Any]]:
        """Get the next recommended task to work on."""
        # Priority order
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}

        # Get pending tasks
        pending = self._manager.find_tasks(status="pending")
        if not pending:
            return None

        # Sort by priority
        pending.sort(key=lambda t: (priority_order.get(t.priority, 2), t.created_at))
        task = pending[0]

        return {
            "id": task.id,
            "title": task.title,
            "priority": task.priority,
            "category": task.properties.get("category", ""),
        }

    # =========================================================================
    # Sprint Operations
    # TODO(adapter-retirement): PHASE 3
    # REVIEWED 2026-01-09: Sub-agent analysis complete
    #
    # - create_sprint, get_sprint, list_sprints, update_sprint, delete_sprint: PURE DELEGATION - remove
    # - get_current_sprint, get_sprint_tasks, get_sprint_progress: PURE DELEGATION - remove
    # - claim_sprint, release_sprint: MOVE TO GoTManager (Phase 3) - adds claimed_by/claimed_at
    # - add_sprint_goal, list_sprint_goals, complete_sprint_goal: MOVE TO GoTManager (Phase 3)
    # - link_task_to_sprint: REDUNDANT with add_task_to_sprint - remove
    # - unlink_task_from_sprint: **BROKEN** - finds edge but doesn't delete (returns True without action)
    #   **BLOCKER: GoTManager needs delete_edge() method first**
    # =========================================================================

    def create_sprint(self, name: str, number: Optional[int] = None,
                     epic_id: Optional[str] = None, description: Optional[str] = None) -> str:
        """Create a sprint."""
        notes = [description] if description else []
        sprint = self._manager.create_sprint(
            title=name, number=number, epic_id=epic_id or "", notes=notes
        )
        return sprint.id

    def get_current_sprint(self) -> Optional[Sprint]:
        """Get current active sprint."""
        return self._manager.get_current_sprint()

    def get_sprint(self, sprint_id: str) -> Optional[Sprint]:
        """Get sprint by ID."""
        return self._manager.get_sprint(sprint_id)

    def list_sprints(self, status: Optional[str] = None, epic_id: Optional[str] = None) -> List[Sprint]:
        """List sprints."""
        return self._manager.list_sprints(status=status, epic_id=epic_id)

    def update_sprint(self, sprint_id: str, **updates) -> Sprint:
        """Update a sprint."""
        return self._manager.update_sprint(sprint_id, **updates)

    def delete_sprint(self, sprint_id: str, force: bool = False) -> None:
        """Delete a sprint."""
        self._manager.delete_sprint(sprint_id, force=force)

    def get_sprint_tasks(self, sprint_id: str) -> List[Task]:
        """Get tasks in a sprint."""
        return self._manager.get_sprint_tasks(sprint_id)

    def get_sprint_progress(self, sprint_id: str) -> Dict[str, Any]:
        """Get sprint progress."""
        return self._manager.get_sprint_progress(sprint_id)

    def claim_sprint(self, sprint_id: str, agent: str) -> Sprint:
        """Claim a sprint for an agent."""
        sprint = self._manager.get_sprint(sprint_id)
        if not sprint:
            raise ValueError(f"Sprint not found: {sprint_id}")
        current_owner = sprint.properties.get("claimed_by")
        if current_owner and current_owner != agent:
            raise ValueError(f"Sprint already claimed by {current_owner}")
        props = dict(sprint.properties)
        props["claimed_by"] = agent
        props["claimed_at"] = datetime.now(timezone.utc).isoformat()
        return self._manager.update_sprint(sprint_id, properties=props)

    def release_sprint(self, sprint_id: str, agent: str) -> Sprint:
        """Release a sprint claim."""
        sprint = self._manager.get_sprint(sprint_id)
        if not sprint:
            raise ValueError(f"Sprint not found: {sprint_id}")
        current_owner = sprint.properties.get("claimed_by")
        if current_owner and current_owner != agent:
            raise ValueError(f"Sprint claimed by {current_owner}, not {agent}")
        props = dict(sprint.properties)
        props.pop("claimed_by", None)
        props.pop("claimed_at", None)
        return self._manager.update_sprint(sprint_id, properties=props)

    def add_sprint_goal(self, sprint_id: str, description: str) -> bool:
        """Add a goal to a sprint."""
        sprint = self._manager.get_sprint(sprint_id)
        if not sprint:
            return False
        goals = list(sprint.goals)
        goals.append({"description": description, "completed": False})
        self._manager.update_sprint(sprint_id, goals=goals)
        return True

    def list_sprint_goals(self, sprint_id: str) -> List[Dict]:
        """List sprint goals."""
        sprint = self._manager.get_sprint(sprint_id)
        return sprint.goals if sprint else []

    def complete_sprint_goal(self, sprint_id: str, goal_index: int) -> bool:
        """Complete a sprint goal."""
        sprint = self._manager.get_sprint(sprint_id)
        if not sprint or goal_index >= len(sprint.goals):
            return False
        goals = list(sprint.goals)
        goals[goal_index]["completed"] = True
        self._manager.update_sprint(sprint_id, goals=goals)
        return True

    def link_task_to_sprint(self, sprint_id: str, task_id: str) -> bool:
        """Link a task to a sprint."""
        try:
            self._manager.add_edge(sprint_id, task_id, "CONTAINS")
            return True
        except Exception:
            return False

    def unlink_task_from_sprint(self, sprint_id: str, task_id: str) -> bool:
        """Unlink a task from a sprint."""
        # Find and remove the edge
        edges = self._manager.list_edges()
        for edge in edges:
            if (edge.source_id == sprint_id and
                edge.target_id == task_id and
                edge.edge_type == "CONTAINS"):
                # TODO: Implement edge deletion in GoTManager
                return True
        return False

    # =========================================================================
    # Epic Operations
    # TODO(adapter-retirement): PURE DELEGATION - remove all, use GoTManager directly
    # =========================================================================

    def create_epic(self, name: str, phase: int = 1) -> str:
        """Create an epic."""
        epic = self._manager.create_epic(title=name, phase=phase)
        return epic.id

    def get_epic(self, epic_id: str) -> Optional[Epic]:
        """Get epic by ID."""
        return self._manager.get_epic(epic_id)

    def list_epics(self, status: Optional[str] = None) -> List[Epic]:
        """List epics."""
        return self._manager.list_epics(status=status)

    # =========================================================================
    # Decision Operations
    # TODO(adapter-retirement): PHASE 4
    # REVIEWED 2026-01-09: Sub-agent analysis complete
    #
    # - create_decision, list_decisions, get_decision, delete_decision: PURE DELEGATION - remove
    # - log_decision: MOVE TO GoTManager - **NOTE: GoTManager.log_decision does NOT create JUSTIFIES edges**
    #   The adapter version creates edges, manager version doesn't - must add edge creation to manager
    # - why: MOVE TO GoTManager - queries decisions affecting task (missing from manager)
    # =========================================================================

    def create_decision(self, content: str, rationale: str = "",
                       task_id: Optional[str] = None,
                       alternatives: Optional[List[str]] = None) -> str:
        """Create a decision."""
        affects = [task_id] if task_id else []
        decision = self._manager.create_decision(
            title=content, rationale=rationale, affects=affects,
            properties={"alternatives": alternatives or []}
        )
        return decision.id

    def list_decisions(self) -> List[Decision]:
        """List all decisions."""
        return self._manager.list_decisions()

    def get_decision(self, decision_id: str) -> Optional[Decision]:
        """Get decision by ID."""
        return self._manager.get_decision(decision_id)

    def delete_decision(self, decision_id: str, force: bool = False) -> None:
        """Delete a decision."""
        self._manager.delete_decision(decision_id, force=force)

    def why(self, task_id: str) -> List[Dict[str, Any]]:
        """Get decisions affecting a task."""
        decisions = self._manager.list_decisions()
        result = []
        for d in decisions:
            if task_id in d.affects:
                result.append({
                    "decision_id": d.id,
                    "decision": d.title,
                    "rationale": d.rationale,
                    "alternatives": d.properties.get("alternatives", []),
                    "created_at": d.created_at,
                })
        return result

    # =========================================================================
    # Handoff Operations
    # TODO(adapter-retirement): MOSTLY PURE DELEGATION - remove most
    # - initiate_handoff, accept_handoff, complete_handoff, reject_handoff: PURE DELEGATION - remove
    # - get_handoff: PURE DELEGATION - remove
    # - list_handoffs: TRANSFORMS OUTPUT - evaluate if transform needed in GoTManager or CLI
    # =========================================================================

    def initiate_handoff(
        self,
        source_agent: str,
        target_agent: str,
        task_id: str = "",
        context: Optional[Dict] = None,
        instructions: str = ""
    ) -> str:
        """Initiate a handoff.

        Args:
            source_agent: Agent initiating the handoff
            target_agent: Agent receiving the handoff
            task_id: Task being handed off (optional for session handoffs)
            context: Additional context data (branch, files, blockers, notes)
            instructions: Instructions for the target agent

        Returns:
            Handoff ID string
        """
        handoff = self._manager.initiate_handoff(
            source_agent=source_agent,
            target_agent=target_agent,
            task_id=task_id,
            context=context or {},
            instructions=instructions
        )
        return handoff.id

    def accept_handoff(self, handoff_id: str, agent: str, acknowledgment: str = "") -> bool:
        """Accept a handoff."""
        try:
            self._manager.accept_handoff(handoff_id, agent, acknowledgment)
            return True
        except Exception:
            return False

    def complete_handoff(self, handoff_id: str, agent: str,
                        result: Dict = None, artifacts: List[str] = None) -> bool:
        """Complete a handoff."""
        try:
            self._manager.complete_handoff(handoff_id, agent, result or {}, artifacts or [])
            return True
        except Exception:
            return False

    def reject_handoff(self, handoff_id: str, agent: str, reason: str = "") -> bool:
        """Reject a handoff."""
        try:
            self._manager.reject_handoff(handoff_id, agent, reason)
            return True
        except Exception:
            return False

    def get_handoff(self, handoff_id: str) -> Optional[Any]:
        """Get handoff by ID. Returns the Handoff entity."""
        return self._manager.get_handoff(handoff_id)

    def list_handoffs(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List handoffs."""
        handoffs = self._manager.list_handoffs(status=status)
        return [
            {
                "id": h.id,
                "source_agent": h.source_agent,
                "target_agent": h.target_agent,
                "task_id": h.task_id,
                "status": h.status,
                "instructions": h.instructions,
                "context": h.context,
                "result": h.result,
                "artifacts": h.artifacts,
                "created_at": h.created_at,
                "accepted_at": getattr(h, 'accepted_at', ''),
                "completed_at": getattr(h, 'completed_at', ''),
            }
            for h in handoffs
        ]

    # =========================================================================
    # Query Operations
    # TODO(adapter-retirement): PHASE 5 & 7
    # - get_stats: MOVE TO GoTManager (Phase 5) - useful introspection
    # - validate: MOVE TO GoTManager (Phase 5) - useful for health checks
    # - query: INCOMPLETE IMPLEMENTATION (Phase 7) - only handles 3 query types
    #   Options: (a) delete, let tests fail (b) redirect to expression system
    #   Recommendation: Delete - expression system (`got expr`) is the future
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        all_tasks = self._manager.list_all_tasks()
        by_status = {}
        for task in all_tasks:
            by_status[task.status] = by_status.get(task.status, 0) + 1

        edges = self._manager.list_edges()
        sprints = self._manager.list_sprints()
        epics = self._manager.list_epics()

        return {
            "total_tasks": len(all_tasks),
            "tasks_by_status": by_status,
            "total_edges": len(edges),
            "total_sprints": len(sprints),
            "total_epics": len(epics),
        }

    def validate(self) -> List[str]:
        """Validate GoT state."""
        issues = []
        try:
            tasks = self._manager.list_all_tasks()
            if not tasks:
                issues.append("No tasks found")
        except Exception as e:
            issues.append(f"Validation error: {e}")
        return issues

    def query(self, query_str: str) -> List[Dict[str, Any]]:
        """Natural language query - delegates to QueryAPI."""
        # Basic query support
        results = []
        q = query_str.lower()

        if "blocked" in q:
            for task, reason in self.get_blocked_tasks():
                results.append({"id": task.id, "title": task.title, "reason": reason})
        elif "active" in q or "in_progress" in q:
            for task in self.get_active_tasks():
                results.append({"id": task.id, "title": task.title})
        elif "pending" in q:
            for task in self._manager.find_tasks(status="pending"):
                results.append({"id": task.id, "title": task.title})

        return results

    # =========================================================================
    # Edge Inference Operations
    # TODO(adapter-retirement): PHASE 6 - MOVE TO NEW MODULE
    # REVIEWED 2026-01-09: Sub-agent analysis complete - **BLOCKERS FOUND**
    #
    # Create: cortical/got/git_inference.py
    # - infer_edges_from_commit: **BLOCKER** - calls self.complete_task() which doesn't exist in GoTManager
    #   Must move complete_task to GoTManager FIRST (Phase 1), then this can be extracted
    # - infer_edges_from_recent_commits: Needs TWO params (manager + project_root), not just manager
    #   Uses self.got_dir.parent for cwd - must add project_root parameter
    # - _get_current_branch: Can be extracted as pure function (no self dependencies)
    # =========================================================================

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
    # Knowledge Transfer Operations
    # TODO(adapter-retirement): PHASE 4
    # REVIEWED 2026-01-09: Sub-agent analysis complete - **CRITICAL ISSUES FOUND**
    #
    # - create_knowledge_transfer, get_knowledge_transfer, list_knowledge_transfers: PURE DELEGATION - remove
    # - update_knowledge_transfer: PURE DELEGATION - **WARNING: GoTManager has race condition (read outside tx)**
    # - append_kt_section: MOVE - **WARNING: GoTManager has cascading race (double read outside tx)**
    # - append_to_knowledge_transfer: Remove - just calls append_kt_section + get
    # - link_knowledge_transfer: PURE DELEGATION (uses add_edge) - remove
    # - finalize_knowledge_transfer: **CRITICAL BUG** - bypasses transaction system entirely (direct store.write)
    #   Must fix to use proper transaction before moving
    # =========================================================================

    def create_knowledge_transfer(self, title: str, summary: str = "",
                                  status: str = "draft", **kwargs) -> str:
        """Create a knowledge transfer document.

        Args:
            title: KT title (required)
            summary: Executive summary
            status: Initial status (default: draft)
            **kwargs: Additional fields (session_id, tags, sections, etc.)

        Returns:
            The KT ID string
        """
        # Delegate to GoTManager (which uses proper transactions)
        kt = self._manager.create_knowledge_transfer(title, summary, status, **kwargs)
        return kt.id

    def list_knowledge_transfers(
        self, status: Optional[str] = None, tags: Optional[List[str]] = None
    ) -> List[Dict]:
        """List knowledge transfers with optional filtering."""
        if hasattr(self._manager, 'list_knowledge_transfers'):
            return self._manager.list_knowledge_transfers(status=status, tags=tags)
        return []

    def get_knowledge_transfer(self, kt_id: str) -> Optional[Any]:
        """Get a knowledge transfer by ID."""
        # Delegate to GoTManager
        return self._manager.get_knowledge_transfer(kt_id)

    def update_knowledge_transfer(
        self, kt_id: str, **updates
    ) -> Optional[Any]:
        """Update a knowledge transfer with the given fields.

        Args:
            kt_id: Knowledge transfer ID
            **updates: Fields to update (status, summary, sections, etc.)

        Returns:
            Updated KT entity or None if not found
        """
        # Delegate to GoTManager (uses proper transactions)
        return self._manager.update_knowledge_transfer(kt_id, **updates)

    def append_kt_section(
        self, kt_id: str, section_title: str, content: str
    ) -> bool:
        """Append a section to an existing knowledge transfer."""
        # Delegate to GoTManager (uses proper transactions)
        result = self._manager.append_knowledge_transfer_section(kt_id, section_title, content)
        return result is not None

    def append_to_knowledge_transfer(
        self, kt_id: str, section_title: str, content: str
    ) -> Optional[Any]:
        """Append a section to a knowledge transfer and return the updated entity.

        This is the user-facing API that returns the updated KT entity.
        """
        if self.append_kt_section(kt_id, section_title, content):
            return self.get_knowledge_transfer(kt_id)
        return None

    def link_knowledge_transfer(
        self, kt_id: str, target_id: str, link_type: str = "DOCUMENTS"
    ) -> bool:
        """Link a knowledge transfer to another entity.

        Args:
            kt_id: Knowledge transfer ID
            target_id: Target entity ID (task, decision, handoff, etc.)
            link_type: Edge type (DOCUMENTS, CONTINUES, etc.)

        Returns:
            True if link was created successfully
        """
        try:
            self.add_edge(kt_id, target_id, link_type)
            return True
        except Exception:
            return False

    # =========================================================================
    # Utility Methods
    # TODO(adapter-retirement): PHASE 5-6
    # - _get_current_branch: MOVE TO git_inference.py (Phase 6) - git helper
    # - graph, nodes, edges properties: EVALUATE - may not be needed after retirement
    #   If still needed, add to GoTManager (Phase 5)
    # =========================================================================

    def _get_current_branch(self) -> str:
        """Get current git branch."""
        import subprocess
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except Exception:
            return "unknown"

    @property
    def graph(self):
        """Compatibility property - returns self for methods that access graph."""
        return self

    @property
    def nodes(self):
        """Compatibility property for graph.nodes access."""
        # Return dict mapping id -> task
        return {t.id: t for t in self._manager.list_all_tasks()}

    @property
    def edges(self):
        """Compatibility property for graph.edges access."""
        return self._manager.list_edges()
