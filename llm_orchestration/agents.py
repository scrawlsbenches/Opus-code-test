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

        # Validate inputs
        if not agent_id or not isinstance(agent_id, str):
            raise ValueError("agent_id must be a non-empty string")
        if context is None:
            raise ValueError("context cannot be None")
        if not isinstance(context, WorkerContext):
            raise TypeError(f"context must be WorkerContext, got {type(context)}")

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
        """
        Execute the task using available tools.

        This method:
        - Validates task context
        - Tracks execution time
        - Records the execution as an experience in the LearningCycle
        - Executes using available tools
        - Returns structured results

        Subclasses can override this to implement specific task execution logic.

        Returns:
            dict: Task execution results with status, output, and timing
        """
        import logging
        from datetime import datetime
        from pathlib import Path

        logger = logging.getLogger(__name__)
        start_time = datetime.now()

        # Validate execution context
        if not self.context or not self.context.task:
            raise ValueError("Cannot execute task: context or task is not set")

        try:
            # Initialize learning cycle if available
            learning_cycle = None
            experience = None
            try:
                from .learning import (
                    LearningCycle, Context, Action, Outcome, OutcomeType,
                    ExperienceType
                )
                storage_dir = Path.home() / ".llm_orchestration" / "learning"
                learning_cycle = LearningCycle(storage_dir)

                # Create experience context
                context = Context(
                    goal_type="task_execution",
                    goal_complexity="moderate",
                    available_tools=self.context.tools,
                    domain="worker_task"
                )
                experience = learning_cycle.start_experience(
                    context=context,
                    intent=self.context.task,
                    experience_type=ExperienceType.TASK_EXECUTION
                )
            except Exception as e:
                logger.debug(f"Learning cycle not available: {e}")

            # Execute the task
            logger.info(f"Worker {self.agent_id} executing task: {self.context.task}")

            # Record action if learning
            if learning_cycle and experience:
                action = Action(
                    action_type="execute_task",
                    description=f"Execute task: {self.context.task}",
                    target=self.context.task,
                    parameters={"tools": self.context.tools}
                )
                experience.add_action(action)

            # Execute task logic - try to use available tools
            tool_outputs = []
            if self.context.tools:
                logger.debug(f"Available tools: {self.context.tools}")
                for tool_name in self.context.tools:
                    try:
                        # Placeholder for tool execution
                        # Subclasses should implement actual tool invocation
                        tool_result = f"Tool '{tool_name}' would be invoked for: {self.context.task}"
                        tool_outputs.append({
                            "tool": tool_name,
                            "result": tool_result,
                            "status": "simulated"
                        })
                        logger.debug(f"Simulated tool '{tool_name}' execution")
                    except Exception as tool_error:
                        logger.warning(f"Tool '{tool_name}' execution failed: {tool_error}")
                        tool_outputs.append({
                            "tool": tool_name,
                            "error": str(tool_error),
                            "status": "failed"
                        })

            # Build result with actual execution details
            result = {
                "status": "completed",
                "task": self.context.task,
                "agent_id": self.agent_id,
                "output": {
                    "description": f"Executed task: {self.context.task}",
                    "tools_used": len(tool_outputs),
                    "tool_results": tool_outputs if tool_outputs else None,
                },
                "timestamp": datetime.now().isoformat(),
                "execution_metadata": {
                    "worker_id": self.agent_id,
                    "tools_available": self.context.tools or [],
                    "constraints": len(self.context.constraints) if self.context.constraints else 0,
                }
            }

            # Calculate duration
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            result["duration_ms"] = duration_ms

            # Record outcome if learning
            if learning_cycle and experience:
                outcome = Outcome(
                    outcome_type=OutcomeType.SUCCESS,
                    description="Task completed successfully",
                    achieved=["task_execution"],
                    quality_score=0.8,
                    efficiency_score=0.7
                )

                # Complete experience
                learning_cycle.complete_experience(
                    experience,
                    outcome,
                    reflection={
                        "worked": [
                            "Task execution completed",
                            f"Used {len(tool_outputs)} tools" if tool_outputs else "No tools used"
                        ],
                        "didnt_work": [],
                        "different": []
                    }
                )

            logger.info(
                f"Task completed in {duration_ms:.2f}ms "
                f"(tools: {len(tool_outputs) if tool_outputs else 0})"
            )
            return result

        except Exception as e:
            logger.error(f"Task execution failed: {e}", exc_info=True)

            # Record failure if learning
            if learning_cycle and experience:
                outcome = Outcome(
                    outcome_type=OutcomeType.FAILURE,
                    description=f"Task failed: {str(e)}",
                    error_type=type(e).__name__,
                    error_message=str(e)
                )
                learning_cycle.complete_experience(experience, outcome)

            raise

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
        """
        Decompose the goal into worker tasks.

        This method breaks down a complex goal into smaller, manageable tasks
        based on:
        - Goal complexity (inferred from description length and keywords)
        - Available resources
        - Scope constraints

        The decomposition follows these heuristics:
        - Simple goals (< 50 chars): 1-2 tasks
        - Moderate goals (50-150 chars): 2-4 tasks
        - Complex goals (> 150 chars): 4-8 tasks

        Returns:
            list[Task]: List of decomposed tasks
        """
        import logging
        import re

        logger = logging.getLogger(__name__)

        goal = self.context.goal
        tasks = []

        # Analyze goal complexity
        goal_length = len(goal)
        goal_lower = goal.lower()

        # Check for complexity indicators
        complex_keywords = [
            "implement", "build", "create", "design", "refactor",
            "migrate", "integrate", "analyze", "optimize"
        ]
        simple_keywords = [
            "fix", "update", "change", "modify", "add", "remove"
        ]

        complexity_score = 1.0

        # Adjust based on length
        if goal_length > 150:
            complexity_score += 2.0
        elif goal_length > 100:
            complexity_score += 1.5
        elif goal_length > 50:
            complexity_score += 1.0

        # Adjust based on keywords
        for keyword in complex_keywords:
            if keyword in goal_lower:
                complexity_score += 0.5

        for keyword in simple_keywords:
            if keyword in goal_lower:
                complexity_score -= 0.5

        # Check for multiple components (indicated by "and", commas, etc.)
        if " and " in goal_lower:
            complexity_score += 1.0
        component_count = len(re.split(r'[,;]', goal)) - 1
        if component_count > 0:
            complexity_score += component_count * 0.5

        # Determine number of tasks based on complexity
        num_tasks = min(max(int(complexity_score), 1), 8)

        logger.info(
            f"Decomposing goal (complexity: {complexity_score:.1f}) "
            f"into {num_tasks} tasks"
        )

        # Create tasks based on decomposition strategy
        if num_tasks == 1:
            # Simple goal - single task
            tasks.append(
                Task(
                    id=f"{self.agent_id}-task-1",
                    description=goal,
                    acceptance_criteria=["Task completed successfully"]
                )
            )
        elif num_tasks <= 3:
            # Moderate goal - break into planning, execution, verification
            tasks.extend([
                Task(
                    id=f"{self.agent_id}-task-1",
                    description=f"Plan approach for: {goal}",
                    acceptance_criteria=["Approach documented and reviewed"]
                ),
                Task(
                    id=f"{self.agent_id}-task-2",
                    description=f"Execute: {goal}",
                    acceptance_criteria=["Implementation completed"]
                ),
                Task(
                    id=f"{self.agent_id}-task-3",
                    description=f"Verify completion of: {goal}",
                    acceptance_criteria=["Tests pass", "Quality checks pass"]
                )
            ])
        else:
            # Complex goal - break into multiple phases
            # Try to split by conjunctions or components
            components = re.split(r'\s+and\s+|,\s*|;\s*', goal)

            if len(components) > 1 and len(components) <= num_tasks:
                # Use natural language components
                for i, component in enumerate(components, 1):
                    if component.strip():
                        tasks.append(
                            Task(
                                id=f"{self.agent_id}-task-{i}",
                                description=component.strip(),
                                acceptance_criteria=[f"Component {i} completed"]
                            )
                        )
            else:
                # Generic decomposition
                phases = [
                    ("Research and design", "Design documented"),
                    ("Implement core functionality", "Core implementation complete"),
                    ("Add supporting features", "Features implemented"),
                    ("Testing and validation", "Tests pass"),
                    ("Documentation", "Documentation complete"),
                    ("Review and refinement", "Quality standards met")
                ]

                for i in range(min(num_tasks, len(phases))):
                    phase_name, acceptance = phases[i]
                    tasks.append(
                        Task(
                            id=f"{self.agent_id}-task-{i+1}",
                            description=f"{phase_name}: {goal}",
                            acceptance_criteria=[acceptance]
                        )
                    )

        logger.info(f"Created {len(tasks)} tasks from goal decomposition")

        # Add task estimates (simple heuristic)
        for task in tasks:
            task.estimate_points = max(1, 8 // len(tasks))

        return tasks

    async def spawn_worker(self, task: Task) -> WorkerHandle:
        """Spawn a worker for a task."""
        # Validate task
        if task is None:
            raise ValueError("task cannot be None")
        if not isinstance(task, Task):
            raise TypeError(f"task must be Task, got {type(task)}")
        if not task.id or not isinstance(task.id, str):
            raise ValueError("task.id must be a non-empty string")
        if not task.description:
            raise ValueError("task.description cannot be empty")

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

    @property
    def completion_rate(self) -> float:
        """Calculate completion rate (completed/estimated points)."""
        if self.estimated_points == 0:
            return 0.0
        return self.completed_points / self.estimated_points

    @property
    def velocity(self) -> float:
        """Get velocity (completed points per day)."""
        if not self.timebox or self.timebox.total_seconds() == 0:
            return 0.0
        days = self.timebox.total_seconds() / (24 * 3600)
        if days == 0:
            return float(self.completed_points)
        return self.completed_points / days


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
        # Validate inputs
        if not goal or not isinstance(goal, str):
            raise ValueError("goal must be a non-empty string")
        if timebox is None:
            raise ValueError("timebox cannot be None")
        if not isinstance(timebox, timedelta):
            raise TypeError(f"timebox must be timedelta, got {type(timebox)}")
        if timebox.total_seconds() <= 0:
            raise ValueError(f"timebox must be positive, got {timebox.total_seconds()}s")

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
        """
        Estimate a task in points using historical data.

        This method uses the Estimator class from agile.py to estimate
        task complexity based on:
        - Historical estimation accuracy
        - Task description analysis
        - Complexity indicators

        Args:
            task: The sprint task to estimate

        Returns:
            int: Estimated points (1-8)
        """
        import logging
        from .agile import Estimator

        logger = logging.getLogger(__name__)

        try:
            # Initialize or get estimator
            if not hasattr(self, '_estimator'):
                self._estimator = Estimator()

            # Estimate using historical data
            estimate = self._estimator.estimate(
                task=task,
                task_type="worker_task"
            )

            logger.debug(
                f"Estimated task '{task.description}' at {estimate} points"
            )

            return estimate

        except Exception as e:
            logger.warning(f"Estimation failed, using default: {e}")
            # Fallback to simple heuristic
            return 1

    async def work_on_task(self, task: SprintTask) -> None:
        """
        Execute a single task within sprint context.

        This method:
        - Executes the task using the parent Worker's execute_task logic
        - Tracks progress within the sprint
        - Records actual points for estimation learning
        - Updates task timestamps

        Args:
            task: The sprint task to execute

        Raises:
            Blocked: If the task cannot proceed due to dependencies
            Exception: If task execution fails
        """
        import logging
        from datetime import datetime

        logger = logging.getLogger(__name__)

        # Preserve original context for restoration
        original_task = self.context.task

        try:
            # Mark task as started
            task.started_at = datetime.now()
            logger.info(f"Starting sprint task: {task.description}")

            # Update worker context to current task
            self.context.task = task.description

            # Execute the task using parent Worker logic
            result = await self.execute_task()

            # Mark task as completed
            task.completed_at = datetime.now()

            # Record actual points for learning
            if hasattr(self, '_estimator'):
                self._estimator.record(
                    task=task,
                    task_type="worker_task",
                    actual_points=task.estimate_points  # Actual equals estimate for now
                )

            logger.info(
                f"Completed sprint task '{task.description}' "
                f"({task.estimate_points} points)"
            )

        except Blocked as b:
            # Let blocked exceptions propagate
            logger.warning(f"Task blocked: {b.reason}")
            raise

        except Exception as e:
            logger.error(f"Task execution failed: {e}", exc_info=True)
            raise

        finally:
            # Always restore original context
            self.context.task = original_task

    async def analyze_successes(self) -> list[str]:
        """
        Analyze what went well in the sprint.

        This method examines the current sprint and identifies positive outcomes:
        - High completion rates
        - Tasks completed without impediments
        - Good estimation accuracy
        - Efficient velocity

        Returns:
            list[str]: List of success observations
        """
        import logging

        logger = logging.getLogger(__name__)
        successes = []

        if not self.current_sprint:
            logger.warning("No sprint to analyze")
            return successes

        sprint = self.current_sprint

        # Analyze completion rate
        if sprint.completion_rate >= 0.9:
            successes.append(
                f"Excellent completion rate: {sprint.completion_rate:.0%} of planned work"
            )
        elif sprint.completion_rate >= 0.7:
            successes.append(
                f"Good completion rate: {sprint.completion_rate:.0%} of planned work"
            )

        # Analyze tasks without blocks
        unblocked_tasks = [
            t for t in sprint.tasks
            if t.status == TaskStatus.COMPLETED
        ]
        if unblocked_tasks and not sprint.impediments:
            successes.append(
                f"All {len(unblocked_tasks)} tasks completed without impediments"
            )

        # Analyze velocity
        if sprint.velocity > 0:
            avg_velocity = self.get_velocity()
            if sprint.velocity >= avg_velocity * 1.1:
                successes.append(
                    f"Velocity improved: {sprint.velocity:.1f} vs avg {avg_velocity:.1f}"
                )
            elif sprint.velocity >= avg_velocity * 0.9:
                successes.append(
                    f"Maintained steady velocity: {sprint.velocity:.1f}"
                )

        # Analyze estimation accuracy
        if 0.8 <= sprint.completion_rate <= 1.2:
            successes.append("Estimation accuracy was good")

        logger.debug(f"Identified {len(successes)} successes")
        return successes

    async def analyze_improvements(self) -> list[str]:
        """
        Analyze what could be improved in the sprint.

        This method identifies areas for improvement:
        - Low completion rates
        - Impediments encountered
        - Poor estimation accuracy
        - Declining velocity

        Returns:
            list[str]: List of improvement observations
        """
        import logging

        logger = logging.getLogger(__name__)
        improvements = []

        if not self.current_sprint:
            logger.warning("No sprint to analyze")
            return improvements

        sprint = self.current_sprint

        # Analyze completion rate
        if sprint.completion_rate < 0.5:
            improvements.append(
                f"Low completion rate: only {sprint.completion_rate:.0%} of planned work"
            )
        elif sprint.completion_rate < 0.7:
            improvements.append(
                f"Moderate completion rate: {sprint.completion_rate:.0%} of planned work"
            )

        # Analyze impediments
        if sprint.impediments:
            improvements.append(
                f"Encountered {len(sprint.impediments)} impediment(s)"
            )
            # List specific impediments
            for imp in sprint.impediments[:3]:  # Limit to first 3
                improvements.append(f"  - {imp.description}")

        # Analyze velocity trend
        if len(self.velocity_history) >= 2:
            recent_velocity = sprint.velocity
            avg_velocity = self.get_velocity()
            if recent_velocity < avg_velocity * 0.8:
                improvements.append(
                    f"Velocity declined: {recent_velocity:.1f} vs avg {avg_velocity:.1f}"
                )

        # Analyze estimation
        if sprint.completion_rate < 0.6:
            improvements.append(
                "Overestimated capacity - consider smaller commitments"
            )
        elif sprint.completion_rate > 1.5:
            improvements.append(
                "Underestimated capacity - could commit to more work"
            )

        # Analyze incomplete tasks
        incomplete_tasks = [
            t for t in sprint.tasks
            if t.status != TaskStatus.COMPLETED
        ]
        if incomplete_tasks:
            improvements.append(
                f"{len(incomplete_tasks)} task(s) not completed"
            )

        logger.debug(f"Identified {len(improvements)} improvements")
        return improvements

    async def generate_actions(self) -> list[str]:
        """
        Generate actionable items from sprint analysis.

        This method creates specific, actionable recommendations for future sprints:
        - Adjustment to estimation
        - Changes to planning
        - Process improvements
        - Impediment resolution

        Returns:
            list[str]: List of action items
        """
        import logging

        logger = logging.getLogger(__name__)
        actions = []

        if not self.current_sprint:
            logger.warning("No sprint to generate actions from")
            return actions

        sprint = self.current_sprint

        # Actions based on completion rate
        if sprint.completion_rate < 0.6:
            actions.append(
                "Action: Reduce sprint commitment by 30% in next sprint"
            )
            actions.append(
                "Action: Break down large tasks into smaller chunks"
            )
        elif sprint.completion_rate > 1.5:
            actions.append(
                "Action: Increase sprint commitment by 20% in next sprint"
            )

        # Actions based on impediments
        if sprint.impediments:
            for imp in sprint.impediments:
                actions.append(
                    f"Action: Resolve impediment - {imp.description}"
                )
            actions.append(
                "Action: Identify and mitigate common blockers proactively"
            )

        # Actions based on velocity trend
        if len(self.velocity_history) >= 3:
            recent_avg = sum(self.velocity_history[-3:]) / 3
            older_avg = sum(self.velocity_history[-6:-3]) / 3 if len(self.velocity_history) >= 6 else recent_avg

            if recent_avg < older_avg * 0.85:
                actions.append(
                    "Action: Investigate causes of declining velocity"
                )
                actions.append(
                    "Action: Review and eliminate process bottlenecks"
                )

        # Actions based on estimation accuracy
        if sprint.completion_rate < 0.8 or sprint.completion_rate > 1.2:
            actions.append(
                "Action: Review estimation process for accuracy"
            )
            actions.append(
                "Action: Compare estimates vs actuals for pattern identification"
            )

        # Default action if sprint was successful
        if not actions and sprint.completion_rate >= 0.8:
            actions.append(
                "Action: Continue current approach - showing good results"
            )

        logger.debug(f"Generated {len(actions)} action items")
        return actions

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
