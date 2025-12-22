#!/usr/bin/env python3
"""
Director orchestration tracking utilities.

This module provides data structures and utilities for tracking multi-agent
orchestration plans, batch execution, and delegation. Designed for the
/director slash command to coordinate parallel agent workflows.

Orchestration Plan Format:
    Plan ID: OP-YYYYMMDD-HHMMSS-XXXX
    Execution ID: EX-YYYYMMDD-HHMMSS-XXXX

    Where:
    - OP/EX = Orchestration Plan / Execution prefix
    - YYYYMMDD = Date created
    - HHMMSS = Time created
    - XXXX = 4-char random suffix (from session UUID)

Example:
    OP-20251215-143052-a1b2  # Orchestration plan
    EX-20251215-143100-b2c3  # Execution tracking

Usage:
    from scripts.orchestration_utils import (
        OrchestrationPlan, Batch, Agent,
        ExecutionTracker, AgentResult, BatchVerification
    )

    # Create a plan
    plan = OrchestrationPlan.create(
        title="Implement feature X",
        goal={"summary": "...", "success_criteria": [...]}
    )

    # Add a batch with agents
    batch = plan.add_batch(
        name="Research phase",
        batch_type="parallel",
        agents=[
            Agent(
                agent_id="A1",
                task_type="research",
                description="Explore existing implementations",
                scope={"files_read": ["src/**/*.py"]}
            )
        ]
    )

    # Save plan
    plan.save()

    # Track execution
    tracker = ExecutionTracker.create(plan)
    tracker.start_batch("B1")
    # ... execute agents ...
    tracker.record_agent_result("A1", AgentResult(...))
    tracker.complete_batch("B1", BatchVerification(...))
    tracker.save()
"""

import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from cortical.utils.persistence import atomic_write_json


# Directory structure for orchestration data
ORCHESTRATION_DIR = Path(".claude/orchestration")
PLANS_DIR = ORCHESTRATION_DIR / "plans"
EXECUTIONS_DIR = ORCHESTRATION_DIR / "executions"
METRICS_FILE = ORCHESTRATION_DIR / "metrics.jsonl"


def generate_plan_id() -> str:
    """
    Generate unique orchestration plan ID.

    Returns:
        Plan ID in format OP-YYYYMMDD-HHMMSS-XXXXXXXX

    Example:
        >>> generate_plan_id()
        'OP-20251215-143052-a1b2c3d4'
    """
    now = datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S")
    suffix = uuid.uuid4().hex[:8]  # 8 chars = 4 billion possibilities
    return f"OP-{date_str}-{time_str}-{suffix}"


def generate_execution_id() -> str:
    """
    Generate unique execution ID.

    Returns:
        Execution ID in format EX-YYYYMMDD-HHMMSS-XXXXXXXX

    Example:
        >>> generate_execution_id()
        'EX-20251215-143100-b2c3d4e5'
    """
    now = datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S")
    suffix = uuid.uuid4().hex[:8]  # 8 chars = 4 billion possibilities
    return f"EX-{date_str}-{time_str}-{suffix}"


@dataclass
class Agent:
    """
    A single agent task within a batch.

    Attributes:
        agent_id: Unique identifier (A1, A2, etc.)
        task_type: Type of task (research | implement | test | verify)
        description: What this agent should accomplish
        scope: Dict with files_read, files_write, constraints, etc.
        status: Current status (pending | in_progress | completed | failed)
        result: Optional result data after completion
    """
    agent_id: str
    task_type: str  # research | implement | test | verify
    description: str
    scope: Dict[str, Any]  # files_read, files_write, constraints
    status: str = "pending"  # pending | in_progress | completed | failed
    result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Agent':
        """Create Agent from dictionary."""
        return cls(**d)

    def mark_in_progress(self) -> None:
        """Mark agent task as in progress."""
        self.status = "in_progress"

    def mark_completed(self, result: Optional[Dict[str, Any]] = None) -> None:
        """Mark agent task as completed with optional result."""
        self.status = "completed"
        self.result = result

    def mark_failed(self, error: Optional[str] = None) -> None:
        """Mark agent task as failed with optional error message."""
        self.status = "failed"
        if error:
            self.result = {"error": error}


@dataclass
class Batch:
    """
    A batch of agents to execute together.

    Batches can be parallel (agents run concurrently) or sequential
    (agents run one after another). Batches can depend on other batches
    completing first.

    Attributes:
        batch_id: Unique identifier (B1, B2, etc.)
        name: Human-readable name for this batch
        batch_type: Execution mode (parallel | sequential)
        agents: List of Agent objects in this batch
        depends_on: List of batch_ids that must complete first
        status: Current status (pending | in_progress | completed | failed)
    """
    batch_id: str  # B1, B2, etc.
    name: str
    batch_type: str  # parallel | sequential
    agents: List[Agent]
    depends_on: List[str] = field(default_factory=list)
    status: str = "pending"  # pending | in_progress | completed | failed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Ensure agents are serialized properly
        data['agents'] = [a if isinstance(a, dict) else a.to_dict() for a in self.agents]
        return data

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Batch':
        """Create Batch from dictionary."""
        # Convert agent dicts to Agent objects
        agents = [Agent.from_dict(a) if isinstance(a, dict) else a for a in d.get('agents', [])]
        d_copy = d.copy()
        d_copy['agents'] = agents
        return cls(**d_copy)

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get an agent by ID from this batch."""
        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent
        return None

    def all_agents_completed(self) -> bool:
        """Check if all agents in this batch have completed."""
        return all(agent.status == "completed" for agent in self.agents)

    def any_agent_failed(self) -> bool:
        """Check if any agent in this batch has failed."""
        return any(agent.status == "failed" for agent in self.agents)

    def mark_in_progress(self) -> None:
        """Mark batch as in progress."""
        self.status = "in_progress"

    def mark_completed(self) -> None:
        """Mark batch as completed."""
        self.status = "completed"

    def mark_failed(self) -> None:
        """Mark batch as failed."""
        self.status = "failed"


@dataclass
class OrchestrationPlan:
    """
    A complete orchestration plan with batches and agents.

    An orchestration plan represents a high-level goal decomposed into
    batches of agent tasks. Batches can run in parallel or sequence,
    with dependency tracking between batches.

    Attributes:
        plan_id: Unique plan identifier (OP-YYYYMMDD-HHMMSS-XXXX)
        title: Human-readable plan title
        goal: Dict with summary and success_criteria
        batches: List of Batch objects
        created_at: ISO timestamp of plan creation
        task_links: Dict linking to parent/child task IDs
    """
    plan_id: str
    title: str
    goal: Dict[str, Any]  # summary, success_criteria
    batches: List[Batch]
    created_at: str
    task_links: Dict[str, Any] = field(default_factory=dict)  # parent_task_id, child_task_ids

    def add_batch(
        self,
        name: str,
        batch_type: str,
        agents: List[Agent],
        depends_on: Optional[List[str]] = None
    ) -> Batch:
        """
        Add a new batch to this plan.

        Args:
            name: Human-readable batch name
            batch_type: "parallel" or "sequential"
            agents: List of Agent objects
            depends_on: List of batch_ids this batch depends on

        Returns:
            The created Batch object
        """
        batch_id = f"B{len(self.batches) + 1}"
        batch = Batch(
            batch_id=batch_id,
            name=name,
            batch_type=batch_type,
            agents=agents,
            depends_on=depends_on or []
        )
        self.batches.append(batch)
        return batch

    def get_batch(self, batch_id: str) -> Optional[Batch]:
        """
        Get a batch by ID from this plan.

        Args:
            batch_id: The batch ID to find (e.g., "B1", "B2")

        Returns:
            The Batch object, or None if not found
        """
        for batch in self.batches:
            if batch.batch_id == batch_id:
                return batch
        return None

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """
        Get an agent by ID from any batch in this plan.

        Args:
            agent_id: The agent ID to find (e.g., "A1", "A2")

        Returns:
            The Agent object, or None if not found
        """
        for batch in self.batches:
            agent = batch.get_agent(agent_id)
            if agent:
                return agent
        return None

    def get_ready_batches(self) -> List[Batch]:
        """
        Get batches that are ready to execute.

        A batch is ready if:
        1. Its status is "pending"
        2. All batches it depends on have completed

        Returns:
            List of Batch objects ready for execution
        """
        ready = []
        for batch in self.batches:
            if batch.status != "pending":
                continue

            # Check if all dependencies are completed
            deps_completed = all(
                self.get_batch(dep_id).status == "completed"
                for dep_id in batch.depends_on
                if self.get_batch(dep_id) is not None
            )

            if deps_completed:
                ready.append(batch)

        return ready

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "plan_id": self.plan_id,
            "title": self.title,
            "goal": self.goal,
            "batches": [b.to_dict() for b in self.batches],
            "created_at": self.created_at,
            "task_links": self.task_links
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'OrchestrationPlan':
        """Create OrchestrationPlan from dictionary."""
        batches = [Batch.from_dict(b) for b in d.get('batches', [])]
        return cls(
            plan_id=d['plan_id'],
            title=d['title'],
            goal=d['goal'],
            batches=batches,
            created_at=d['created_at'],
            task_links=d.get('task_links', {})
        )

    def save(self, path: Optional[Path] = None) -> Path:
        """
        Save plan to a JSON file atomically.

        Uses write-to-temp-then-rename pattern to prevent data loss
        if the process crashes during write.

        Args:
            path: Optional custom path. If None, uses default location
                  in PLANS_DIR with filename based on plan_id

        Returns:
            Path to the saved file

        Raises:
            OSError: If write or rename fails
        """
        if path is None:
            PLANS_DIR.mkdir(parents=True, exist_ok=True)
            path = PLANS_DIR / f"{self.plan_id}.json"

        data = {
            "version": 1,
            "schema": "orchestration_plan",
            "saved_at": datetime.now().isoformat(),
            "plan": self.to_dict()
        }

        atomic_write_json(path, data)
        return path

    @classmethod
    def load(cls, path: Path) -> 'OrchestrationPlan':
        """
        Load a plan from file.

        Args:
            path: Path to the plan JSON file

        Returns:
            OrchestrationPlan object

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file is not valid JSON
            KeyError: If required fields are missing
        """
        with open(path) as f:
            data = json.load(f)

        return cls.from_dict(data['plan'])

    @classmethod
    def create(
        cls,
        title: str,
        goal: Dict[str, Any],
        parent_task_id: Optional[str] = None
    ) -> 'OrchestrationPlan':
        """
        Create a new orchestration plan.

        Args:
            title: Human-readable plan title
            goal: Dict with 'summary' and 'success_criteria' keys
            parent_task_id: Optional parent task ID from task_utils

        Returns:
            New OrchestrationPlan object

        Example:
            >>> plan = OrchestrationPlan.create(
            ...     title="Implement feature X",
            ...     goal={
            ...         "summary": "Add new search feature",
            ...         "success_criteria": ["Tests pass", "Coverage >90%"]
            ...     }
            ... )
        """
        task_links = {}
        if parent_task_id:
            task_links['parent_task_id'] = parent_task_id
            task_links['child_task_ids'] = []

        return cls(
            plan_id=generate_plan_id(),
            title=title,
            goal=goal,
            batches=[],
            created_at=datetime.now().isoformat(),
            task_links=task_links
        )


@dataclass
class AgentResult:
    """
    Result of a completed agent execution.

    Captures the outcome of an agent task including status, timing,
    output summary, files modified, and any errors encountered.

    Attributes:
        status: Execution outcome (completed | failed)
        started_at: ISO timestamp when agent started
        completed_at: ISO timestamp when agent finished
        duration_ms: Execution duration in milliseconds
        output_summary: Brief summary of agent output
        files_modified: List of file paths modified by agent
        errors: List of error messages if any
    """
    status: str  # completed | failed
    started_at: str
    completed_at: str
    duration_ms: int
    output_summary: str
    files_modified: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'AgentResult':
        """Create AgentResult from dictionary."""
        return cls(**d)


@dataclass
class BatchVerification:
    """
    Verification result for a completed batch.

    Records verification checks performed after a batch completes,
    including tests, conflicts, and git status.

    Attributes:
        batch_id: ID of the verified batch
        verified_at: ISO timestamp of verification
        checks: Dict of check results (tests_pass, no_conflicts, git_clean)
        verdict: Overall verification verdict (pass | pass_with_warnings | fail)
        notes: Optional notes or details about the verification
    """
    batch_id: str
    verified_at: str
    checks: Dict[str, bool]  # tests_pass, no_conflicts, git_clean
    verdict: str  # pass | pass_with_warnings | fail
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'BatchVerification':
        """Create BatchVerification from dictionary."""
        return cls(**d)


@dataclass
class ReplanEvent:
    """
    Record of a replanning event during execution.

    Captures when and why a plan was modified during execution,
    including the trigger, reason, and comparison of old vs new plan.

    Attributes:
        at: ISO timestamp of replanning event
        trigger: What triggered replanning (agent_blocker | verification_fail | scope_change)
        reason: Detailed explanation of why replanning occurred
        old_plan_summary: Summary of plan before replanning
        new_plan_summary: Summary of plan after replanning
    """
    at: str  # timestamp
    trigger: str  # agent_blocker | verification_fail | scope_change
    reason: str
    old_plan_summary: str
    new_plan_summary: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ReplanEvent':
        """Create ReplanEvent from dictionary."""
        return cls(**d)


@dataclass
class ExecutionTracker:
    """
    Track execution of an orchestration plan.

    Monitors plan execution state, batch progress, agent results,
    verification outcomes, and replanning events. Provides methods
    for updating execution state and calculating batch durations.

    Attributes:
        plan_id: ID of the plan being executed
        execution_id: Unique execution ID (EX-YYYYMMDD-HHMMSS-XXXX)
        started_at: ISO timestamp when execution started
        status: Current execution status (in_progress | completed | failed | paused)
        current_batch: ID of currently executing batch, or None
        batches_completed: List of completed batch IDs
        batches_remaining: List of pending batch IDs
        agent_results: Dict mapping agent_id to AgentResult
        verifications: List of BatchVerification results
        replanning_events: List of ReplanEvent records
    """
    plan_id: str
    execution_id: str
    started_at: str
    status: str  # in_progress | completed | failed | paused
    current_batch: Optional[str]
    batches_completed: List[str]
    batches_remaining: List[str]
    agent_results: Dict[str, AgentResult]
    verifications: List[BatchVerification]
    replanning_events: List[ReplanEvent]

    @classmethod
    def create(cls, plan: OrchestrationPlan) -> 'ExecutionTracker':
        """
        Create a new execution tracker for a plan.

        Args:
            plan: OrchestrationPlan to track

        Returns:
            New ExecutionTracker initialized with plan's batches
        """
        return cls(
            plan_id=plan.plan_id,
            execution_id=generate_execution_id(),
            started_at=datetime.now().isoformat(),
            status="in_progress",
            current_batch=None,
            batches_completed=[],
            batches_remaining=[b.batch_id for b in plan.batches],
            agent_results={},
            verifications=[],
            replanning_events=[]
        )

    def start_batch(self, batch_id: str) -> None:
        """
        Mark a batch as currently executing.

        Args:
            batch_id: ID of the batch to start
        """
        self.current_batch = batch_id

    def record_agent_result(self, agent_id: str, result: AgentResult) -> None:
        """
        Record the result of an agent execution.

        Args:
            agent_id: ID of the agent that completed
            result: AgentResult object with execution details
        """
        self.agent_results[agent_id] = result

    def complete_batch(self, batch_id: str, verification: BatchVerification) -> None:
        """
        Mark a batch as completed with verification results.

        Moves batch from remaining to completed and records verification.

        Args:
            batch_id: ID of the batch that completed
            verification: BatchVerification result
        """
        if batch_id in self.batches_remaining:
            self.batches_remaining.remove(batch_id)
        if batch_id not in self.batches_completed:
            self.batches_completed.append(batch_id)
        self.verifications.append(verification)
        if self.current_batch == batch_id:
            self.current_batch = None

    def record_replan(
        self,
        trigger: str,
        reason: str,
        old_summary: str,
        new_summary: str
    ) -> None:
        """
        Record a replanning event.

        Args:
            trigger: What triggered replanning (agent_blocker | verification_fail | scope_change)
            reason: Detailed explanation
            old_summary: Summary of plan before replanning
            new_summary: Summary of plan after replanning
        """
        event = ReplanEvent(
            at=datetime.now().isoformat(),
            trigger=trigger,
            reason=reason,
            old_plan_summary=old_summary,
            new_plan_summary=new_summary
        )
        self.replanning_events.append(event)

    def get_batch_duration(self, batch_id: str) -> Optional[int]:
        """
        Calculate duration of a batch in milliseconds.

        Finds all agent results for agents in the specified batch and
        calculates the total duration from earliest start to latest completion.

        Args:
            batch_id: ID of the batch to calculate duration for

        Returns:
            Duration in milliseconds, or None if batch not found or incomplete
        """
        # Find all agents in this batch
        batch_agents = [
            agent_id for agent_id in self.agent_results.keys()
            if agent_id.startswith('A') and batch_id in self.batches_completed
        ]

        if not batch_agents:
            return None

        # Get earliest start and latest completion
        results = [self.agent_results[aid] for aid in batch_agents if aid in self.agent_results]
        if not results:
            return None

        # Parse timestamps and calculate duration
        try:
            start_times = [datetime.fromisoformat(r.started_at) for r in results]
            end_times = [datetime.fromisoformat(r.completed_at) for r in results]
            earliest_start = min(start_times)
            latest_end = max(end_times)
            duration_ms = int((latest_end - earliest_start).total_seconds() * 1000)
            return duration_ms
        except (ValueError, AttributeError):
            return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "plan_id": self.plan_id,
            "execution_id": self.execution_id,
            "started_at": self.started_at,
            "status": self.status,
            "current_batch": self.current_batch,
            "batches_completed": self.batches_completed,
            "batches_remaining": self.batches_remaining,
            "agent_results": {k: v.to_dict() for k, v in self.agent_results.items()},
            "verifications": [v.to_dict() for v in self.verifications],
            "replanning_events": [e.to_dict() for e in self.replanning_events]
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ExecutionTracker':
        """Create ExecutionTracker from dictionary."""
        agent_results = {
            k: AgentResult.from_dict(v)
            for k, v in d.get('agent_results', {}).items()
        }
        verifications = [
            BatchVerification.from_dict(v)
            for v in d.get('verifications', [])
        ]
        replanning_events = [
            ReplanEvent.from_dict(e)
            for e in d.get('replanning_events', [])
        ]

        return cls(
            plan_id=d['plan_id'],
            execution_id=d['execution_id'],
            started_at=d['started_at'],
            status=d['status'],
            current_batch=d.get('current_batch'),
            batches_completed=d.get('batches_completed', []),
            batches_remaining=d.get('batches_remaining', []),
            agent_results=agent_results,
            verifications=verifications,
            replanning_events=replanning_events
        )

    def save(self, path: Optional[Path] = None) -> Path:
        """
        Save execution tracker to a JSON file atomically.

        Uses write-to-temp-then-rename pattern to prevent data loss
        if the process crashes during write.

        Args:
            path: Optional custom path. If None, uses default location
                  in EXECUTIONS_DIR with filename based on plan_id

        Returns:
            Path to the saved file

        Raises:
            OSError: If write or rename fails
        """
        if path is None:
            EXECUTIONS_DIR.mkdir(parents=True, exist_ok=True)
            path = EXECUTIONS_DIR / f"{self.plan_id}_execution.json"

        data = {
            "version": 1,
            "schema": "execution_tracker",
            "saved_at": datetime.now().isoformat(),
            "execution": self.to_dict()
        }

        atomic_write_json(path, data)
        return path

    @classmethod
    def load(cls, path: Path) -> 'ExecutionTracker':
        """
        Load an execution tracker from file.

        Args:
            path: Path to the execution tracker JSON file

        Returns:
            ExecutionTracker object

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file is not valid JSON
            KeyError: If required fields are missing
        """
        with open(path) as f:
            data = json.load(f)

        return cls.from_dict(data['execution'])


def load_all_plans(plans_dir: Optional[Path] = None) -> List[OrchestrationPlan]:
    """
    Load all orchestration plans from directory.

    Args:
        plans_dir: Directory containing plan files (default: PLANS_DIR)

    Returns:
        List of OrchestrationPlan objects, sorted by creation time
    """
    dir_path = plans_dir or PLANS_DIR
    if not dir_path.exists():
        return []

    all_plans = []
    for filepath in sorted(dir_path.glob("*.json")):
        try:
            plan = OrchestrationPlan.load(filepath)
            all_plans.append(plan)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load {filepath}: {e}")

    # Sort by creation time
    all_plans.sort(key=lambda p: p.created_at)
    return all_plans


def get_plan_by_id(
    plan_id: str,
    plans_dir: Optional[Path] = None
) -> Optional[OrchestrationPlan]:
    """
    Find a plan by its ID.

    Args:
        plan_id: The plan ID to find
        plans_dir: Directory containing plan files (default: PLANS_DIR)

    Returns:
        OrchestrationPlan object, or None if not found
    """
    for plan in load_all_plans(plans_dir):
        if plan.plan_id == plan_id:
            return plan
    return None


@dataclass
class MetricsEvent:
    """
    A single metrics event for orchestration tracking.

    Events are stored in JSONL format (one JSON object per line) for
    append-only tracking and easy analysis.

    Attributes:
        timestamp: ISO timestamp of event
        plan_id: Orchestration plan ID
        event_type: Type of event (batch_start | batch_complete | agent_complete | verification | replan)
        batch_id: Optional batch identifier
        agent_id: Optional agent identifier
        duration_ms: Optional duration in milliseconds
        success: Optional success indicator
        metadata: Additional event-specific data
    """
    timestamp: str
    plan_id: str
    event_type: str  # batch_start | batch_complete | agent_complete | verification | replan
    batch_id: Optional[str] = None
    agent_id: Optional[str] = None
    duration_ms: Optional[int] = None
    success: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json_line(self) -> str:
        """
        Serialize event to a single JSON line.

        Returns:
            JSON string without newline (caller adds newline)
        """
        data = asdict(self)
        return json.dumps(data, separators=(',', ':'))

    @classmethod
    def from_json_line(cls, line: str) -> 'MetricsEvent':
        """
        Deserialize event from JSON line.

        Args:
            line: JSON string (with or without trailing newline)

        Returns:
            MetricsEvent object

        Raises:
            json.JSONDecodeError: If line is not valid JSON
        """
        data = json.loads(line.strip())
        return cls(**data)


class OrchestrationMetrics:
    """
    Collects and aggregates orchestration metrics.

    Metrics are stored in append-only JSONL format for durability and
    easy analysis. Each event is one line, making the file git-friendly
    and merge-resistant.

    Usage:
        metrics = OrchestrationMetrics()

        # Record events
        metrics.record_batch_start("OP-123", "B1")
        metrics.record_agent_complete("OP-123", "B1", "A1", duration_ms=5000, success=True)
        metrics.record_batch_complete("OP-123", "B1", duration_ms=300000, success=True)

        # Analyze
        summary = metrics.get_summary()
        failures = metrics.get_failure_patterns()
    """

    def __init__(self, metrics_file: Optional[Path] = None):
        """
        Initialize metrics collector.

        Args:
            metrics_file: Path to metrics JSONL file (default: METRICS_FILE)
        """
        self.metrics_file = metrics_file or METRICS_FILE

    def record(
        self,
        event_type: str,
        plan_id: str,
        batch_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        duration_ms: Optional[int] = None,
        success: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Append a metrics event to the JSONL file.

        Args:
            event_type: Type of event (batch_start | batch_complete | agent_complete | verification | replan)
            plan_id: Orchestration plan ID
            batch_id: Optional batch identifier
            agent_id: Optional agent identifier
            duration_ms: Optional duration in milliseconds
            success: Optional success indicator
            metadata: Additional event-specific data
        """
        event = MetricsEvent(
            timestamp=datetime.now().isoformat(),
            plan_id=plan_id,
            event_type=event_type,
            batch_id=batch_id,
            agent_id=agent_id,
            duration_ms=duration_ms,
            success=success,
            metadata=metadata or {}
        )

        # Ensure parent directory exists
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)

        # Append to JSONL file
        with open(self.metrics_file, 'a') as f:
            f.write(event.to_json_line() + '\n')

    def record_batch_start(self, plan_id: str, batch_id: str) -> None:
        """
        Record batch starting.

        Args:
            plan_id: Orchestration plan ID
            batch_id: Batch identifier
        """
        self.record('batch_start', plan_id, batch_id=batch_id)

    def record_batch_complete(
        self,
        plan_id: str,
        batch_id: str,
        duration_ms: int,
        success: bool
    ) -> None:
        """
        Record batch completion.

        Args:
            plan_id: Orchestration plan ID
            batch_id: Batch identifier
            duration_ms: Duration in milliseconds
            success: Whether batch completed successfully
        """
        self.record(
            'batch_complete',
            plan_id,
            batch_id=batch_id,
            duration_ms=duration_ms,
            success=success
        )

    def record_agent_complete(
        self,
        plan_id: str,
        batch_id: str,
        agent_id: str,
        duration_ms: int,
        success: bool
    ) -> None:
        """
        Record agent completion.

        Args:
            plan_id: Orchestration plan ID
            batch_id: Batch identifier
            agent_id: Agent identifier
            duration_ms: Duration in milliseconds
            success: Whether agent completed successfully
        """
        self.record(
            'agent_complete',
            plan_id,
            batch_id=batch_id,
            agent_id=agent_id,
            duration_ms=duration_ms,
            success=success
        )

    def record_verification(
        self,
        plan_id: str,
        batch_id: str,
        passed: bool,
        checks: Dict[str, bool]
    ) -> None:
        """
        Record verification result.

        Args:
            plan_id: Orchestration plan ID
            batch_id: Batch identifier
            passed: Whether verification passed overall
            checks: Dict mapping check names to pass/fail
        """
        self.record(
            'verification',
            plan_id,
            batch_id=batch_id,
            success=passed,
            metadata={'checks': checks}
        )

    def record_replan(self, plan_id: str, trigger: str, reason: str) -> None:
        """
        Record replanning event.

        Args:
            plan_id: Orchestration plan ID
            trigger: What triggered the replan (e.g., 'verification_failed', 'new_requirements')
            reason: Human-readable reason for replanning
        """
        self.record(
            'replan',
            plan_id,
            metadata={'trigger': trigger, 'reason': reason}
        )

    def get_events(self, plan_id: Optional[str] = None) -> List[MetricsEvent]:
        """
        Load events from file, optionally filtered by plan_id.

        Args:
            plan_id: Optional plan ID to filter by

        Returns:
            List of MetricsEvent objects
        """
        if not self.metrics_file.exists():
            return []

        events = []
        with open(self.metrics_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = MetricsEvent.from_json_line(line)
                    if plan_id is None or event.plan_id == plan_id:
                        events.append(event)
                except (json.JSONDecodeError, TypeError) as e:
                    # Skip malformed lines
                    continue

        return events

    def get_summary(self) -> Dict[str, Any]:
        """
        Get aggregate statistics.

        Returns:
            Dict with summary statistics including:
            - total_plans: Number of unique plans
            - total_batches: Number of batch completions
            - total_agents: Number of agent completions
            - batch_success_rate: Percentage of successful batches
            - agent_success_rate: Percentage of successful agents
            - avg_batch_duration_ms: Average batch duration
            - avg_agent_duration_ms: Average agent duration
            - total_replans: Number of replan events
        """
        events = self.get_events()

        if not events:
            return {
                'total_plans': 0,
                'total_batches': 0,
                'total_agents': 0,
                'batch_success_rate': 0.0,
                'agent_success_rate': 0.0,
                'avg_batch_duration_ms': 0,
                'avg_agent_duration_ms': 0,
                'total_replans': 0
            }

        # Count unique plans
        plan_ids = set(e.plan_id for e in events)

        # Analyze batch completions
        batch_completions = [e for e in events if e.event_type == 'batch_complete']
        batch_successes = sum(1 for e in batch_completions if e.success)
        batch_durations = [e.duration_ms for e in batch_completions if e.duration_ms is not None]

        # Analyze agent completions
        agent_completions = [e for e in events if e.event_type == 'agent_complete']
        agent_successes = sum(1 for e in agent_completions if e.success)
        agent_durations = [e.duration_ms for e in agent_completions if e.duration_ms is not None]

        # Count replans
        replans = [e for e in events if e.event_type == 'replan']

        return {
            'total_plans': len(plan_ids),
            'total_batches': len(batch_completions),
            'total_agents': len(agent_completions),
            'batch_success_rate': (batch_successes / len(batch_completions) * 100) if batch_completions else 0.0,
            'agent_success_rate': (agent_successes / len(agent_completions) * 100) if agent_completions else 0.0,
            'avg_batch_duration_ms': int(sum(batch_durations) / len(batch_durations)) if batch_durations else 0,
            'avg_agent_duration_ms': int(sum(agent_durations) / len(agent_durations)) if agent_durations else 0,
            'total_replans': len(replans)
        }

    def get_failure_patterns(self) -> List[Dict[str, Any]]:
        """
        Analyze common failure modes.

        Returns:
            List of failure patterns sorted by frequency, each containing:
            - failure_type: Type of failure (batch_complete, agent_complete, verification, replan:trigger)
            - count: Number of occurrences
            - example_plan_ids: List of up to 5 unique example plan IDs
        """
        events = self.get_events()

        # Group failures by type
        failures: Dict[str, List[str]] = {}
        for event in events:
            if event.success is False:
                key = event.event_type
                if key not in failures:
                    failures[key] = []
                failures[key].append(event.plan_id)

        # Also include replan triggers
        for event in events:
            if event.event_type == 'replan':
                trigger = event.metadata.get('trigger', 'unknown')
                key = f"replan:{trigger}"
                if key not in failures:
                    failures[key] = []
                failures[key].append(event.plan_id)

        # Convert to sorted list
        patterns = []
        for failure_type, plan_ids in failures.items():
            patterns.append({
                'failure_type': failure_type,
                'count': len(plan_ids),
                'example_plan_ids': list(set(plan_ids))[:5]  # Up to 5 unique examples
            })

        # Sort by frequency (descending)
        patterns.sort(key=lambda x: x['count'], reverse=True)

        return patterns


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Director orchestration utilities"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # generate command
    gen_parser = subparsers.add_parser("generate", help="Generate an ID")
    gen_parser.add_argument(
        "--type",
        choices=["plan", "execution"],
        default="plan",
        help="Type of ID to generate"
    )

    # list command
    list_parser = subparsers.add_parser("list", help="List all plans")
    list_parser.add_argument("--dir", type=Path, help="Plans directory")

    # show command
    show_parser = subparsers.add_parser("show", help="Show plan details")
    show_parser.add_argument("plan_id", help="Plan ID to show")
    show_parser.add_argument("--dir", type=Path, help="Plans directory")

    args = parser.parse_args()

    if args.command == "generate":
        if args.type == "plan":
            print(generate_plan_id())
        else:
            print(generate_execution_id())

    elif args.command == "list":
        plans = load_all_plans(args.dir)
        for plan in plans:
            status = "active" if any(b.status == "in_progress" for b in plan.batches) else "pending"
            print(f"[{status}] {plan.plan_id}: {plan.title} ({len(plan.batches)} batches)")

    elif args.command == "show":
        plan = get_plan_by_id(args.plan_id, args.dir)
        if not plan:
            print(f"Error: Plan not found: {args.plan_id}")
        else:
            print(f"\nPlan: {plan.title}")
            print(f"ID: {plan.plan_id}")
            print(f"Created: {plan.created_at}")
            print(f"\nGoal: {plan.goal.get('summary', 'N/A')}")
            print(f"\nBatches:")
            for batch in plan.batches:
                print(f"  {batch.batch_id}: {batch.name} [{batch.status}]")
                print(f"    Type: {batch.batch_type}")
                print(f"    Agents: {len(batch.agents)}")
                if batch.depends_on:
                    print(f"    Depends on: {', '.join(batch.depends_on)}")

    else:
        parser.print_help()
