"""
Agent implementations for the LLM Orchestration Framework.

Agents are the execution units in the hierarchy:
- Directors: Orchestrate workers, manage phases
- Workers: Execute focused tasks within sprints
- HybridDirector: Bridge between kanban (above) and agile (below)

Each agent type has:
- A context that defines its scope and resources
- An event loop for execution
- Communication channels for coordination
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Literal

from .types import (
    AgentRole,
    Blocked,
    Channel,
    Checkpoint,
    Constraint,
    Delegation,
    DirectorContext,
    Event,
    EventBus,
    Goal,
    Impediment,
    Increment,
    Result,
    Retrospective,
    Scope,
    SprintTask,
    Task,
    TaskStatus,
    WorkerContext,
)


# =============================================================================
# BASE AGENT
# =============================================================================


class Agent(ABC):
    """Base class for all agents."""

    def __init__(self, agent_id: str, role: AgentRole):
        self.agent_id = agent_id
        self.role = role
        self.status = TaskStatus.PENDING
        self.spawned_at = datetime.now()

    @abstractmethod
    async def run(self) -> Result:
        """Execute the agent's main loop."""
        pass

    @abstractmethod
    async def checkpoint(self) -> Checkpoint:
        """Create a resumable checkpoint."""
        pass

    @abstractmethod
    async def resume(self, checkpoint: Checkpoint) -> None:
        """Resume from a checkpoint."""
        pass


# =============================================================================
# WORKER
# =============================================================================


@dataclass
class WorkerResult:
    """Result from a worker execution."""

    status: Literal["complete", "blocked", "failed"]
    output: Any = None
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


class Worker(Agent):
    """
    Worker agent - executes focused tasks.

    Workers are leaf nodes in the hierarchy. They:
    - Receive focused tasks with clear scope
    - Execute using available tools
    - Publish progress events
    - Return structured results
    """

    def __init__(
        self,
        agent_id: str,
        context: WorkerContext,
    ):
        super().__init__(agent_id, AgentRole.WORKER)
        self.context = context
        self.current_task: Task | None = None
        self.progress: float = 0.0

    async def run(self) -> Result:
        """Execute the worker's task."""
        try:
            # Publish start
            if self.context.event_bus:
                await self.context.event_bus.publish(Event(
                    type="worker.started",
                    payload={"task": self.context.task},
                    source_agent_id=self.agent_id,
                ))

            # Execute task
            self.status = TaskStatus.IN_PROGRESS
            result = await self.execute_task()

            # Publish completion
            if self.context.event_bus:
                await self.context.event_bus.publish(Event(
                    type="worker.completed",
                    payload={"result": result},
                    source_agent_id=self.agent_id,
                ))

            self.status = TaskStatus.COMPLETED
            return Result(success=True, output=result)

        except Blocked as b:
            self.status = TaskStatus.BLOCKED
            if self.context.event_bus:
                await self.context.event_bus.publish(Event(
                    type="worker.blocked",
                    payload={
                        "reason": b.reason,
                        "need": b.what_i_need,
                    },
                    source_agent_id=self.agent_id,
                ))
            return Result(success=False, error=b.reason)

        except Exception as e:
            self.status = TaskStatus.FAILED
            if self.context.event_bus:
                await self.context.event_bus.publish(Event(
                    type="worker.failed",
                    payload={"error": str(e)},
                    source_agent_id=self.agent_id,
                ))
            return Result(success=False, error=str(e))

    async def execute_task(self) -> Any:
        """Execute the task using available tools."""
        # Subclasses would implement specific task execution
        return {"status": "completed", "task": self.context.task}

    async def report_progress(self, progress: float, step: str) -> None:
        """Report progress to director."""
        self.progress = progress
        if self.context.event_bus:
            await self.context.event_bus.publish(Event(
                type="worker.progress",
                payload={
                    "progress": progress,
                    "step": step,
                },
                source_agent_id=self.agent_id,
            ))

    async def checkpoint(self) -> Checkpoint:
        """Create a resumable checkpoint."""
        return Checkpoint(
            agent_id=self.agent_id,
            role="worker",
            current_step=self.context.task,
            draft_outputs={"progress": self.progress},
        )

    async def resume(self, checkpoint: Checkpoint) -> None:
        """Resume from checkpoint."""
        self.progress = checkpoint.draft_outputs.get("progress", 0.0)


# =============================================================================
# DIRECTOR
# =============================================================================


@dataclass
class DirectorResult:
    """Result from a director execution."""

    status: Literal["complete", "partial", "failed"]
    output: Any = None
    workers_spawned: int = 0
    events_handled: int = 0
    decisions_made: list[str] = field(default_factory=list)


@dataclass
class WorkerHandle:
    """Handle to a spawned worker."""

    worker_id: str
    task: Task
    worker: Worker
    status: TaskStatus = TaskStatus.PENDING


class Director(Agent):
    """
    Director agent - orchestrates workers.

    Directors are intermediate nodes that:
    - Decompose goals into worker tasks
    - Spawn and manage workers
    - Handle events and impediments
    - Synthesize results
    """

    def __init__(
        self,
        agent_id: str,
        context: DirectorContext,
    ):
        super().__init__(agent_id, AgentRole.DIRECTOR)
        self.context = context
        self.workers: dict[str, WorkerHandle] = {}
        self.completed_outputs: dict[str, Any] = {}
        self.decisions: list[str] = []
        self.event_count = 0

    async def run(self) -> Result:
        """Execute the director's orchestration loop."""
        try:
            # 1. Decompose goal into tasks
            plan = await self.decompose_goal()

            # Publish plan
            if self.context.event_bus:
                await self.context.event_bus.publish(Event(
                    type="director.plan_created",
                    payload={
                        "subtask_count": len(plan),
                    },
                    source_agent_id=self.agent_id,
                ))

            # 2. Spawn workers for each task
            for task in plan:
                await self.spawn_worker(task)

            # 3. Orchestrate: manage execution
            await self.orchestrate()

            # 4. Synthesize results
            synthesis = await self.synthesize_results()

            # 5. Report completion
            self.status = TaskStatus.COMPLETED
            return Result(
                success=True,
                output=synthesis,
                metadata={
                    "workers_spawned": len(self.workers),
                    "events_handled": self.event_count,
                    "decisions_made": self.decisions,
                },
            )

        except Exception as e:
            self.status = TaskStatus.FAILED
            return Result(success=False, error=str(e))

    async def decompose_goal(self) -> list[Task]:
        """Decompose the goal into worker tasks."""
        # Subclasses would implement specific decomposition logic
        # This is a placeholder that creates a single task
        return [
            Task(
                id=f"{self.agent_id}-task-1",
                description=self.context.goal,
            )
        ]

    async def spawn_worker(self, task: Task) -> WorkerHandle:
        """Spawn a worker for a task."""
        worker_id = f"{self.agent_id}-worker-{len(self.workers)}"

        worker_context = WorkerContext(
            task=task.description,
            tools=self.context.tools_available,
            event_bus=self.context.event_bus,
            constraints=self.context.constraints,
        )

        worker = Worker(worker_id, worker_context)

        handle = WorkerHandle(
            worker_id=worker_id,
            task=task,
            worker=worker,
        )

        self.workers[worker_id] = handle

        # Subscribe to worker events
        if self.context.event_bus:
            self.context.event_bus.subscribe(
                f"worker.{worker_id}.*",
                self.handle_worker_event,
            )

        return handle

    def handle_worker_event(self, event: Event) -> None:
        """Handle an event from a worker."""
        self.event_count += 1

        # Extract worker ID from event type
        if event.source_agent_id in self.workers:
            handle = self.workers[event.source_agent_id]

            if "completed" in event.type:
                handle.status = TaskStatus.COMPLETED
                self.completed_outputs[handle.worker_id] = event.payload.get(
                    "result"
                )

            elif "blocked" in event.type:
                handle.status = TaskStatus.BLOCKED

            elif "failed" in event.type:
                handle.status = TaskStatus.FAILED

    async def orchestrate(self) -> None:
        """Manage worker execution."""
        pending = set(self.workers.keys())
        max_concurrent = 3  # WIP limit

        while pending:
            # Start workers up to WIP limit
            in_progress = sum(
                1 for w in self.workers.values()
                if w.status == TaskStatus.IN_PROGRESS
            )

            ready_to_start = [
                wid for wid in pending
                if self.workers[wid].status == TaskStatus.PENDING
                and self.dependencies_met(wid)
            ]

            for wid in ready_to_start[:max_concurrent - in_progress]:
                handle = self.workers[wid]
                handle.status = TaskStatus.IN_PROGRESS
                asyncio.create_task(handle.worker.run())

            # Wait briefly for events
            await asyncio.sleep(0.1)

            # Update pending set
            pending = {
                wid for wid in pending
                if self.workers[wid].status not in {
                    TaskStatus.COMPLETED,
                    TaskStatus.FAILED,
                }
            }

    def dependencies_met(self, worker_id: str) -> bool:
        """Check if a worker's dependencies are met."""
        # Subclasses would implement dependency checking
        return True

    async def synthesize_results(self) -> Any:
        """Synthesize worker outputs into final result."""
        return {
            "workers": len(self.workers),
            "completed": len(self.completed_outputs),
            "outputs": self.completed_outputs,
        }

    async def escalate(self, issue: str) -> None:
        """Escalate an issue to the orchestrator."""
        if self.context.event_bus:
            await self.context.event_bus.publish(Event(
                type="director.escalation",
                payload={
                    "issue": issue,
                    "director_id": self.agent_id,
                },
                source_agent_id=self.agent_id,
            ))

    async def checkpoint(self) -> Checkpoint:
        """Create a resumable checkpoint."""
        return Checkpoint(
            agent_id=self.agent_id,
            role="director",
            completed_steps=[
                wid for wid, h in self.workers.items()
                if h.status == TaskStatus.COMPLETED
            ],
            pending_steps=[
                wid for wid, h in self.workers.items()
                if h.status == TaskStatus.PENDING
            ],
            decisions=self.decisions,
            draft_outputs=self.completed_outputs,
        )

    async def resume(self, checkpoint: Checkpoint) -> None:
        """Resume from checkpoint."""
        self.completed_outputs = checkpoint.draft_outputs
        self.decisions = checkpoint.decisions


# =============================================================================
# AGILE WORKER
# =============================================================================


@dataclass
class WorkerSprint:
    """A time-boxed sprint for a worker."""

    sprint_id: str
    goal: str
    timebox: timedelta
    tasks: list[SprintTask] = field(default_factory=list)

    # Tracking
    estimated_points: int = 0
    completed_points: int = 0
    impediments: list[Impediment] = field(default_factory=list)

    # Outcome
    increment: Increment | None = None
    retrospective: Retrospective | None = None


class AgileWorker(Worker):
    """
    Worker that operates in agile sprints.

    Extends basic Worker with:
    - Time-boxed execution
    - Velocity tracking
    - Retrospectives
    - Increment delivery
    """

    def __init__(
        self,
        agent_id: str,
        context: WorkerContext,
    ):
        super().__init__(agent_id, context)
        self.current_sprint: WorkerSprint | None = None
        self.velocity_history: list[int] = []

    async def run(self) -> Result:
        """Execute as a sprint."""
        # Plan sprint
        sprint = await self.plan_sprint(
            self.context.task,
            self.context.timebox,
        )

        # Execute sprint
        sprint_result = await self.execute_sprint()

        # Retrospective
        retro = await self.retrospective()

        # Deliver increment
        increment = await self.deliver_increment()

        return Result(
            success=sprint_result.completed_points > 0,
            output=increment,
            metadata={
                "sprint_id": sprint.sprint_id,
                "velocity": sprint_result.completed_points,
                "retrospective": retro,
            },
        )

    async def plan_sprint(
        self,
        goal: str,
        timebox: timedelta,
    ) -> WorkerSprint:
        """Plan a sprint with estimated tasks."""
        tasks = await self.decompose_to_tasks(goal)

        # Estimate tasks
        for task in tasks:
            task.estimate_points = await self.estimate_task(task)

        # Commit based on velocity
        velocity = self.get_velocity()
        capacity = self._timebox_to_points(timebox, velocity)

        committed = []
        points = 0
        for task in tasks:
            if points + task.estimate_points <= capacity:
                committed.append(task)
                points += task.estimate_points

        sprint = WorkerSprint(
            sprint_id=f"sprint-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            goal=goal,
            timebox=timebox,
            tasks=committed,
            estimated_points=points,
        )

        self.current_sprint = sprint
        return sprint

    def get_velocity(self) -> float:
        """Get average velocity from history."""
        if not self.velocity_history:
            return 5.0  # Default
        return sum(self.velocity_history[-5:]) / min(len(self.velocity_history), 5)

    async def execute_sprint(self) -> WorkerSprint:
        """Execute the sprint tasks."""
        if not self.current_sprint:
            raise ValueError("No sprint planned")

        sprint = self.current_sprint
        start_time = datetime.now()

        for task in sprint.tasks:
            # Check timebox
            elapsed = datetime.now() - start_time
            if elapsed > sprint.timebox:
                break

            # Execute task
            task.status = TaskStatus.IN_PROGRESS
            await self.report_progress(
                sprint.completed_points / max(sprint.estimated_points, 1),
                task.description,
            )

            try:
                await self.work_on_task(task)
                task.status = TaskStatus.COMPLETED
                task.actual_points = task.estimate_points
                sprint.completed_points += task.actual_points

            except Blocked as b:
                task.status = TaskStatus.BLOCKED
                sprint.impediments.append(Impediment(
                    task_id=task.id,
                    description=b.reason,
                ))

        return sprint

    async def retrospective(self) -> Retrospective:
        """Conduct a retrospective."""
        if not self.current_sprint:
            raise ValueError("No sprint to retrospect on")

        sprint = self.current_sprint

        retro = Retrospective(
            sprint_id=sprint.sprint_id,
            went_well=await self.analyze_successes(),
            improvements=await self.analyze_improvements(),
            action_items=await self.generate_actions(),
            velocity_actual=sprint.completed_points,
            velocity_planned=sprint.estimated_points,
            estimation_accuracy=(
                sprint.completed_points / max(sprint.estimated_points, 1)
            ),
            impediment_count=len(sprint.impediments),
        )

        # Update velocity history
        self.velocity_history.append(sprint.completed_points)

        # Publish for evolution
        if self.context.event_bus:
            await self.context.event_bus.publish(Event(
                type="worker.retrospective",
                payload=retro.__dict__,
                source_agent_id=self.agent_id,
            ))

        sprint.retrospective = retro
        return retro

    async def deliver_increment(self) -> Increment:
        """Package sprint output as deliverable."""
        if not self.current_sprint:
            raise ValueError("No sprint to deliver")

        sprint = self.current_sprint

        increment = Increment(
            sprint_id=sprint.sprint_id,
            goal=sprint.goal,
            outputs=await self.collect_outputs(),
            acceptance_met=await self.verify_acceptance(),
            metrics={
                "planned_points": sprint.estimated_points,
                "completed_points": sprint.completed_points,
            },
        )

        sprint.increment = increment

        # Publish increment ready
        if self.context.event_bus:
            await self.context.event_bus.publish(Event(
                type="worker.increment_ready",
                payload=increment.__dict__,
                source_agent_id=self.agent_id,
            ))

        return increment

    # Helper methods - subclasses would implement
    async def decompose_to_tasks(self, goal: str) -> list[SprintTask]:
        """Decompose goal into sprint tasks."""
        return [
            SprintTask(
                id=f"task-{i}",
                description=f"Task {i} for: {goal}",
                estimate_points=1,
            )
            for i in range(3)
        ]

    async def estimate_task(self, task: SprintTask) -> int:
        """Estimate a task in points."""
        return 1

    async def work_on_task(self, task: SprintTask) -> None:
        """Execute a single task."""
        pass

    async def analyze_successes(self) -> list[str]:
        """Analyze what went well."""
        return []

    async def analyze_improvements(self) -> list[str]:
        """Analyze what could improve."""
        return []

    async def generate_actions(self) -> list[str]:
        """Generate action items."""
        return []

    async def collect_outputs(self) -> dict[str, Any]:
        """Collect sprint outputs."""
        return {}

    async def verify_acceptance(self) -> bool:
        """Verify acceptance criteria."""
        return True

    def _timebox_to_points(self, timebox: timedelta, velocity: float) -> int:
        """Convert timebox to capacity in points."""
        # Assume 1 point = 5 minutes of work
        minutes = timebox.total_seconds() / 60
        return int(minutes / 5 * (velocity / 5))


# =============================================================================
# HYBRID DIRECTOR
# =============================================================================


@dataclass
class Phase:
    """A phase of work (like an epic)."""

    id: str
    description: str
    tasks: list[Task] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING


@dataclass
class WorkerKanbanItem:
    """An item on the worker kanban board."""

    id: str
    task: Task
    column: Literal["backlog", "ready", "in_progress", "done"]
    worker_id: str | None = None


class WorkerKanbanBoard:
    """Mini kanban board for managing workers."""

    def __init__(self):
        self.items: dict[str, WorkerKanbanItem] = {}
        self.wip_limit = 3

    def add_to_backlog(self, task: Task) -> None:
        """Add a task to backlog."""
        item = WorkerKanbanItem(
            id=task.id,
            task=task,
            column="backlog",
        )
        self.items[task.id] = item

    def move_to_ready(self, task_id: str) -> None:
        """Move task to ready column."""
        if task_id in self.items:
            self.items[task_id].column = "ready"

    def move_to_in_progress(self, task_id: str, worker_id: str) -> bool:
        """Move task to in_progress if WIP allows."""
        current_wip = sum(
            1 for item in self.items.values()
            if item.column == "in_progress"
        )

        if current_wip >= self.wip_limit:
            return False

        if task_id in self.items:
            self.items[task_id].column = "in_progress"
            self.items[task_id].worker_id = worker_id
            return True

        return False

    def move_to_done(self, task_id: str) -> None:
        """Move task to done column."""
        if task_id in self.items:
            self.items[task_id].column = "done"

    def get_wip(self) -> int:
        """Get current WIP count."""
        return sum(
            1 for item in self.items.values()
            if item.column == "in_progress"
        )

    def get_blocked_count(self) -> int:
        """Get count of blocked items."""
        return 0  # Would track blocked state


class HybridDirector(Director):
    """
    Director that bridges kanban (above) and agile (below).

    Receives work via kanban pull from orchestrator.
    Manages workers using agile sprints.
    """

    def __init__(
        self,
        agent_id: str,
        context: DirectorContext,
    ):
        super().__init__(agent_id, context)
        self.worker_board = WorkerKanbanBoard()
        self.current_phase: Phase | None = None
        self.phases: list[Phase] = []
        self.retrospectives: list[Retrospective] = []

    async def run(self) -> Result:
        """Execute as hybrid: receive via pull, execute as sprints."""
        try:
            # Plan phases (like epics → stories)
            self.phases = await self.plan_phases()

            # Execute each phase
            for phase in self.phases:
                self.current_phase = phase
                phase.status = TaskStatus.IN_PROGRESS

                # Add phase tasks to kanban board
                for task in phase.tasks:
                    self.worker_board.add_to_backlog(task)
                    self.worker_board.move_to_ready(task.id)

                # Execute phase
                phase_result = await self.execute_phase(phase)

                phase.status = TaskStatus.COMPLETED

            # Synthesize
            synthesis = await self.synthesize_phases()

            return Result(
                success=True,
                output=synthesis,
                metadata={
                    "phases_completed": len(self.phases),
                    "retrospectives": len(self.retrospectives),
                },
            )

        except Exception as e:
            return Result(success=False, error=str(e))

    async def plan_phases(self) -> list[Phase]:
        """Plan phases from goal."""
        # Subclasses would implement decomposition
        return [
            Phase(
                id="phase-1",
                description=self.context.goal,
                tasks=[
                    Task(id="task-1", description="First task"),
                    Task(id="task-2", description="Second task"),
                ],
            )
        ]

    async def execute_phase(self, phase: Phase) -> dict[str, Any]:
        """Execute a phase using agile workers."""
        results = {}

        for task in phase.tasks:
            # Pull from board (respects WIP)
            worker_id = f"{self.agent_id}-worker-{task.id}"

            if not self.worker_board.move_to_in_progress(task.id, worker_id):
                # WIP limit reached, wait
                await self.wait_for_capacity()
                self.worker_board.move_to_in_progress(task.id, worker_id)

            # Spawn agile worker
            worker_context = WorkerContext(
                task=task.description,
                tools=self.context.tools_available,
                event_bus=self.context.event_bus,
                timebox=timedelta(minutes=10),
            )

            worker = AgileWorker(worker_id, worker_context)

            # Execute
            result = await worker.run()
            results[task.id] = result

            # Update board
            self.worker_board.move_to_done(task.id)

            # Collect retrospective
            if worker.current_sprint and worker.current_sprint.retrospective:
                self.retrospectives.append(
                    worker.current_sprint.retrospective
                )

        return results

    async def wait_for_capacity(self) -> None:
        """Wait for WIP capacity."""
        while self.worker_board.get_wip() >= self.worker_board.wip_limit:
            await asyncio.sleep(0.1)

    async def synthesize_phases(self) -> dict[str, Any]:
        """Synthesize all phase results."""
        return {
            "phases": len(self.phases),
            "retrospectives": [r.__dict__ for r in self.retrospectives],
        }

    def report_up_flow(self) -> dict[str, Any]:
        """Report to orchestrator in kanban terms."""
        return {
            "wip": self.worker_board.get_wip(),
            "throughput": len([
                item for item in self.worker_board.items.values()
                if item.column == "done"
            ]),
            "blocked": self.worker_board.get_blocked_count(),
        }

    async def swarm_on_blocked(self, blocked_task_id: str) -> None:
        """Swarm available workers to help unblock."""
        # Classic agile practice
        available = [
            wid for wid, handle in self.workers.items()
            if handle.status == TaskStatus.COMPLETED
        ]

        if not available:
            await self.escalate(f"Cannot unblock {blocked_task_id}, no workers available")
            return

        # Assign helpers
        for helper_id in available[:2]:
            if self.context.event_bus:
                await self.context.event_bus.publish(Event(
                    type="director.swarm_requested",
                    payload={
                        "blocked_task": blocked_task_id,
                        "helper": helper_id,
                    },
                    source_agent_id=self.agent_id,
                ))
