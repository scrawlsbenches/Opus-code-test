# Agent Rejection Protocol

## Overview

The Agent Rejection Protocol enables agents to reject tasks with structured, actionable information while preventing lazy rejections. This protocol integrates with the existing GoT handoff system and enables the Director to make informed decisions about replanning.

**Design Principles:**
1. **Rejection must be earned** - Agents must demonstrate they tried before rejecting
2. **Rejection must help** - Provide actionable information for replanning
3. **Rejection is learning** - All rejections are logged to GoT for pattern detection
4. **Rejection is not failure** - It's honest communication about blockers

---

## Core Data Structures

### RejectionReason Enum

```python
from enum import Enum, auto

class RejectionReason(Enum):
    """
    Valid reasons for rejecting a task.

    Each reason has specific validation requirements to prevent abuse.
    """

    BLOCKER = auto()
    """External blocker prevents progress.

    Examples:
    - API unavailable
    - Required credentials missing
    - External dependency down

    Validation:
    - Must specify what_blocked (concrete blocker)
    - Must provide evidence (error logs, status checks)
    - Must suggest how to unblock
    """

    SCOPE_CREEP = auto()
    """Task evolved beyond original scope during investigation.

    Examples:
    - "Fix login bug" revealed auth system redesign needed
    - "Add button" requires refactoring entire component hierarchy
    - Simple change requires breaking API changes

    Validation:
    - Must cite original scope
    - Must document how scope grew
    - Must quantify complexity increase (time estimate, files affected)
    - Must suggest decomposition into smaller tasks
    """

    MISSING_DEPENDENCY = auto()
    """Internal dependency not yet met.

    Examples:
    - Task X requires Task Y to be completed first
    - Needs design decision not yet made
    - Depends on infrastructure not yet built

    Validation:
    - Must identify specific dependency (task ID or decision ID)
    - Must explain why dependency is hard requirement
    - Must suggest dependency resolution path
    - Cannot reject if dependency is trivial to implement yourself
    """

    INFEASIBLE = auto()
    """Task fundamentally cannot be done as stated.

    Examples:
    - "Make it backwards compatible" conflicts with "Use new API only"
    - "Fix without changing behavior" for bug in behavior
    - Physical impossibility (e.g., achieve O(1) for inherently O(n) operation)

    Validation:
    - Must demonstrate logical impossibility
    - Must cite conflicting constraints
    - Must suggest reformulation that IS feasible
    - Highest burden of proof - must be truly impossible
    """

    UNCLEAR_REQUIREMENTS = auto()
    """Task requirements are ambiguous after attempting clarification.

    Examples:
    - "Improve performance" without metrics or targets
    - "Fix the bug" without reproduction steps
    - Multiple valid interpretations of acceptance criteria

    Validation:
    - Must list specific questions that need answers
    - Must show attempted interpretation(s)
    - Must cite concrete ambiguities in requirements
    - Should not reject without asking clarifying questions first
    """
```

### TaskRejection Dataclass

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any

@dataclass
class TaskRejection:
    """
    Structured rejection with evidence and alternatives.

    This is the core data structure passed during rejection.
    It must contain enough information for the Director to:
    1. Validate the rejection is legitimate
    2. Understand what went wrong
    3. Decide how to replan
    """

    # Identity
    task_id: str
    handoff_id: str
    agent_id: str
    rejected_at: datetime = field(default_factory=datetime.now)

    # Rejection reason
    reason_type: RejectionReason
    reason_summary: str  # One-line explanation
    reason_detail: str   # Multi-paragraph explanation

    # What was attempted (REQUIRED - demonstrates effort)
    what_attempted: List[str] = field(default_factory=list)
    """
    Concrete steps taken before rejecting.

    Examples:
    - "Analyzed codebase structure in auth/ directory"
    - "Attempted to run existing tests - 12 failures"
    - "Searched for configuration file - not found"
    - "Asked user for clarification via blocker"

    Validation: Must have at least 2 concrete attempts
    """

    # Blocking factor (specific, concrete)
    blocking_factor: str = ""
    """
    The specific thing preventing completion.

    Must be concrete and verifiable:
    - ❌ "Too complex" (vague)
    - ✅ "Requires changes to 47 files across 8 modules" (concrete)

    - ❌ "Missing information" (vague)
    - ✅ "Need API key for external service X" (concrete)
    """

    # Evidence (supports the rejection)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    """
    Supporting evidence for the rejection.

    Each evidence item should have:
    - type: "error_log" | "file_analysis" | "complexity_metric" | "conflict"
    - data: Relevant data (error message, file count, etc.)
    - source: Where this evidence came from

    Examples:
    {
        "type": "error_log",
        "data": "ConnectionError: API endpoint returned 503",
        "source": "attempt to call auth API"
    },
    {
        "type": "complexity_metric",
        "data": {"files_requiring_changes": 47, "estimated_hours": 16},
        "source": "static analysis of task scope"
    }
    """

    # Suggested alternative (REQUIRED - must help, not just complain)
    suggested_alternative: str = ""
    """
    Actionable suggestion for how to proceed.

    Must be specific and implementable:
    - ❌ "Get more context" (vague)
    - ✅ "Ask user for production API credentials" (specific)

    - ❌ "Break into smaller tasks" (vague)
    - ✅ "Split into: (1) refactor auth module, (2) add login feature" (specific)
    """

    # Alternative tasks (optional decomposition)
    alternative_tasks: List[Dict[str, str]] = field(default_factory=list)
    """
    If suggesting task decomposition, provide the breakdown.

    Each alternative task should have:
    - title: Short description
    - scope: What it covers
    - dependencies: What it depends on
    - estimate: Rough size estimate

    Example:
    [
        {
            "title": "Refactor auth module for extensibility",
            "scope": "Isolate auth logic, add plugin interface",
            "dependencies": "none",
            "estimate": "4-6 hours"
        },
        {
            "title": "Add OAuth login provider",
            "scope": "Implement OAuth using new plugin interface",
            "dependencies": "Requires refactor task above",
            "estimate": "2-3 hours"
        }
    ]
    """

    # Metadata
    task_original_scope: str = ""  # For SCOPE_CREEP validation
    scope_growth_factor: Optional[float] = None  # Quantified scope increase

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for event logging."""
        return {
            "task_id": self.task_id,
            "handoff_id": self.handoff_id,
            "agent_id": self.agent_id,
            "rejected_at": self.rejected_at.isoformat(),
            "reason_type": self.reason_type.name,
            "reason_summary": self.reason_summary,
            "reason_detail": self.reason_detail,
            "what_attempted": self.what_attempted,
            "blocking_factor": self.blocking_factor,
            "evidence": self.evidence,
            "suggested_alternative": self.suggested_alternative,
            "alternative_tasks": self.alternative_tasks,
            "task_original_scope": self.task_original_scope,
            "scope_growth_factor": self.scope_growth_factor,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskRejection":
        """Reconstruct from dict."""
        data["reason_type"] = RejectionReason[data["reason_type"]]
        data["rejected_at"] = datetime.fromisoformat(data["rejected_at"])
        return cls(**data)
```

---

## Rejection Validation

### RejectionValidator Class

```python
from typing import Tuple, List

class RejectionValidator:
    """
    Validates that rejections are legitimate and not lazy.

    This is the gatekeeper that prevents agents from rejecting
    tasks they should be able to handle.
    """

    def validate(self, rejection: TaskRejection, task_context: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a rejection.

        Args:
            rejection: The rejection to validate
            task_context: Original task context (scope, requirements, etc.)

        Returns:
            (is_valid, issues) where issues lists why rejection is invalid
        """
        issues = []

        # Universal validations
        issues.extend(self._validate_effort(rejection))
        issues.extend(self._validate_alternative(rejection))
        issues.extend(self._validate_blocking_factor(rejection))

        # Reason-specific validations
        if rejection.reason_type == RejectionReason.BLOCKER:
            issues.extend(self._validate_blocker(rejection))
        elif rejection.reason_type == RejectionReason.SCOPE_CREEP:
            issues.extend(self._validate_scope_creep(rejection, task_context))
        elif rejection.reason_type == RejectionReason.MISSING_DEPENDENCY:
            issues.extend(self._validate_missing_dependency(rejection))
        elif rejection.reason_type == RejectionReason.INFEASIBLE:
            issues.extend(self._validate_infeasible(rejection))
        elif rejection.reason_type == RejectionReason.UNCLEAR_REQUIREMENTS:
            issues.extend(self._validate_unclear_requirements(rejection))

        return (len(issues) == 0, issues)

    def _validate_effort(self, rejection: TaskRejection) -> List[str]:
        """Agent must demonstrate they tried."""
        issues = []

        if len(rejection.what_attempted) < 2:
            issues.append(
                "Insufficient effort demonstrated. Must document at least 2 concrete attempts."
            )

        # Check for lazy attempts
        lazy_phrases = ["tried to", "looked at", "checked", "considered"]
        concrete_attempts = [
            attempt for attempt in rejection.what_attempted
            if not any(phrase in attempt.lower() for phrase in lazy_phrases)
        ]

        if len(concrete_attempts) < 1:
            issues.append(
                "Attempts are too vague. At least 1 must be concrete and specific."
            )

        return issues

    def _validate_alternative(self, rejection: TaskRejection) -> List[str]:
        """Agent must provide actionable alternative."""
        issues = []

        if not rejection.suggested_alternative:
            issues.append("Must provide suggested_alternative.")
        elif len(rejection.suggested_alternative) < 20:
            issues.append("Suggested alternative is too vague. Provide specific actionable steps.")

        # Check for lazy suggestions
        lazy_suggestions = [
            "ask the user",
            "get more context",
            "clarify requirements",
            "break into smaller tasks",
        ]

        if any(phrase in rejection.suggested_alternative.lower() for phrase in lazy_suggestions):
            if len(rejection.alternative_tasks) == 0 and "break into" not in rejection.suggested_alternative.lower():
                issues.append(
                    "Suggestion is too generic. Provide concrete actionable steps or task breakdown."
                )

        return issues

    def _validate_blocking_factor(self, rejection: TaskRejection) -> List[str]:
        """Blocking factor must be specific and concrete."""
        issues = []

        if not rejection.blocking_factor:
            issues.append("Must specify blocking_factor.")
        elif len(rejection.blocking_factor) < 15:
            issues.append("Blocking factor is too vague. Be specific.")

        # Check for vague blockers
        vague_blockers = ["too complex", "too hard", "not sure", "unclear", "confusing"]
        if any(phrase in rejection.blocking_factor.lower() for phrase in vague_blockers):
            issues.append(
                f"Blocking factor contains vague language. Must be concrete and measurable."
            )

        return issues

    def _validate_blocker(self, rejection: TaskRejection) -> List[str]:
        """Validate BLOCKER rejection."""
        issues = []

        # Must have evidence
        if len(rejection.evidence) == 0:
            issues.append("BLOCKER rejection requires evidence (error logs, status checks, etc.)")

        # Evidence should include error or status data
        has_concrete_evidence = any(
            ev.get("type") in ["error_log", "status_check", "api_response"]
            for ev in rejection.evidence
        )

        if not has_concrete_evidence:
            issues.append("BLOCKER evidence must include error logs or status checks.")

        return issues

    def _validate_scope_creep(self, rejection: TaskRejection, task_context: Dict[str, Any]) -> List[str]:
        """Validate SCOPE_CREEP rejection."""
        issues = []

        # Must document original scope
        if not rejection.task_original_scope:
            issues.append("SCOPE_CREEP requires documenting original scope.")

        # Must quantify growth
        if rejection.scope_growth_factor is None or rejection.scope_growth_factor < 2.0:
            issues.append(
                "SCOPE_CREEP requires demonstrating at least 2x complexity increase. "
                "Provide scope_growth_factor with evidence."
            )

        # Must suggest decomposition
        if len(rejection.alternative_tasks) < 2:
            issues.append(
                "SCOPE_CREEP requires suggesting task decomposition (at least 2 sub-tasks)."
            )

        return issues

    def _validate_missing_dependency(self, rejection: TaskRejection) -> List[str]:
        """Validate MISSING_DEPENDENCY rejection."""
        issues = []

        # Must identify specific dependency
        if not rejection.blocking_factor or "depends on" not in rejection.blocking_factor.lower():
            issues.append(
                "MISSING_DEPENDENCY must identify specific dependency (task ID or decision ID)."
            )

        # Should explain why it's a hard requirement
        if "required because" not in rejection.reason_detail.lower():
            issues.append(
                "MISSING_DEPENDENCY should explain why dependency is a hard requirement."
            )

        return issues

    def _validate_infeasible(self, rejection: TaskRejection) -> List[str]:
        """Validate INFEASIBLE rejection (highest burden of proof)."""
        issues = []

        # Must demonstrate logical impossibility
        if len(rejection.evidence) < 2:
            issues.append(
                "INFEASIBLE has highest burden of proof. Must provide at least 2 pieces of evidence "
                "demonstrating logical impossibility."
            )

        # Must cite conflicting constraints
        if "conflict" not in rejection.reason_detail.lower():
            issues.append(
                "INFEASIBLE must cite conflicting constraints that make task impossible."
            )

        # Must suggest feasible reformulation
        if not rejection.suggested_alternative or len(rejection.suggested_alternative) < 50:
            issues.append(
                "INFEASIBLE must suggest detailed reformulation that IS feasible."
            )

        return issues

    def _validate_unclear_requirements(self, rejection: TaskRejection) -> List[str]:
        """Validate UNCLEAR_REQUIREMENTS rejection."""
        issues = []

        # Must list specific questions
        if "?" not in rejection.reason_detail:
            issues.append(
                "UNCLEAR_REQUIREMENTS must list specific questions that need answers."
            )

        # Should show attempted interpretation
        if "interpreted as" not in rejection.reason_detail.lower():
            issues.append(
                "UNCLEAR_REQUIREMENTS should document attempted interpretations."
            )

        return issues
```

---

## Director Response Protocol

### DirectorRejectionHandler Class

```python
class DirectorRejectionHandler:
    """
    Handles rejected tasks at the Director level.

    Provides three response paths:
    1. Accept rejection and replan
    2. Override rejection with additional context
    3. Decompose and reassign
    """

    def __init__(self, manager: GoTProjectManager):
        self.manager = manager
        self.validator = RejectionValidator()

    def handle_rejection(
        self,
        rejection: TaskRejection,
        task_context: Dict[str, Any],
    ) -> "RejectionDecision":
        """
        Process a task rejection and decide how to respond.

        Args:
            rejection: The rejection from the agent
            task_context: Original task context

        Returns:
            RejectionDecision with chosen response path
        """
        # Step 1: Validate the rejection
        is_valid, issues = self.validator.validate(rejection, task_context)

        if not is_valid:
            # Rejection is invalid - override with explanation
            return self._override_rejection(rejection, issues)

        # Step 2: Analyze rejection reason and decide response
        if rejection.reason_type == RejectionReason.BLOCKER:
            return self._handle_blocker(rejection, task_context)

        elif rejection.reason_type == RejectionReason.SCOPE_CREEP:
            return self._handle_scope_creep(rejection, task_context)

        elif rejection.reason_type == RejectionReason.MISSING_DEPENDENCY:
            return self._handle_missing_dependency(rejection, task_context)

        elif rejection.reason_type == RejectionReason.INFEASIBLE:
            return self._handle_infeasible(rejection, task_context)

        elif rejection.reason_type == RejectionReason.UNCLEAR_REQUIREMENTS:
            return self._handle_unclear_requirements(rejection, task_context)

        # Default: accept and replan
        return self._accept_rejection(rejection)

    def _override_rejection(
        self,
        rejection: TaskRejection,
        validation_issues: List[str],
    ) -> "RejectionDecision":
        """
        Override an invalid rejection.

        Provides additional context and reassigns to same or different agent.
        """
        override_message = self._generate_override_message(validation_issues)

        return RejectionDecision(
            decision_type=DecisionType.OVERRIDE,
            rejection=rejection,
            rationale=f"Rejection validation failed: {', '.join(validation_issues)}",
            override_message=override_message,
            reassign_to=rejection.agent_id,  # Same agent with more context
        )

    def _handle_blocker(
        self,
        rejection: TaskRejection,
        task_context: Dict[str, Any],
    ) -> "RejectionDecision":
        """
        Handle BLOCKER rejection.

        Strategy: Extract blocker as separate task, defer original task.
        """
        # Create blocker resolution task
        blocker_task_id = self.manager.create_task(
            title=f"Resolve blocker: {rejection.blocking_factor[:50]}",
            description=rejection.reason_detail,
            priority="high",
            category="blocker",
        )

        # Link original task to blocker
        self.manager.graph.add_edge(
            rejection.task_id,
            blocker_task_id,
            EdgeType.BLOCKS,
            weight=1.0,
        )

        return RejectionDecision(
            decision_type=DecisionType.ACCEPT_AND_DEFER,
            rejection=rejection,
            rationale=f"Valid blocker. Created blocker resolution task: {blocker_task_id}",
            created_tasks=[blocker_task_id],
            deferred_task=rejection.task_id,
        )

    def _handle_scope_creep(
        self,
        rejection: TaskRejection,
        task_context: Dict[str, Any],
    ) -> "RejectionDecision":
        """
        Handle SCOPE_CREEP rejection.

        Strategy: Accept decomposition, create sub-tasks, link with dependencies.
        """
        # Validate we have alternative tasks
        if len(rejection.alternative_tasks) < 2:
            return self._override_rejection(
                rejection,
                ["SCOPE_CREEP requires at least 2 alternative sub-tasks"]
            )

        # Create sub-tasks from alternatives
        created_task_ids = []
        for alt_task in rejection.alternative_tasks:
            task_id = self.manager.create_task(
                title=alt_task["title"],
                description=alt_task["scope"],
                priority=task_context.get("priority", "medium"),
                category=task_context.get("category", "feature"),
            )
            created_task_ids.append(task_id)

            # Link to original task
            self.manager.graph.add_edge(
                task_id,
                rejection.task_id,
                EdgeType.REFINES,
                weight=0.8,
            )

        # Link dependencies between sub-tasks
        for i, task_id in enumerate(created_task_ids[1:], start=1):
            prev_task_id = created_task_ids[i - 1]
            alt_task = rejection.alternative_tasks[i]

            if "depends" in alt_task.get("dependencies", "").lower():
                self.manager.graph.add_edge(
                    task_id,
                    prev_task_id,
                    EdgeType.DEPENDS_ON,
                    weight=1.0,
                )

        # Mark original task as decomposed
        self.manager.update_task(
            rejection.task_id,
            status="decomposed",
            properties={"decomposed_into": created_task_ids},
        )

        return RejectionDecision(
            decision_type=DecisionType.ACCEPT_AND_DECOMPOSE,
            rejection=rejection,
            rationale=f"Valid scope creep. Decomposed into {len(created_task_ids)} sub-tasks.",
            created_tasks=created_task_ids,
        )

    def _handle_missing_dependency(
        self,
        rejection: TaskRejection,
        task_context: Dict[str, Any],
    ) -> "RejectionDecision":
        """
        Handle MISSING_DEPENDENCY rejection.

        Strategy: Identify dependency, create if needed, defer original task.
        """
        # Try to extract dependency ID from blocking factor
        dependency_id = self._extract_dependency_id(rejection.blocking_factor)

        if not dependency_id:
            # Create dependency as new task
            dependency_id = self.manager.create_task(
                title=f"Dependency for {rejection.task_id}: {rejection.blocking_factor[:50]}",
                description=rejection.reason_detail,
                priority="high",
                category="dependency",
            )

        # Link dependency
        self.manager.graph.add_edge(
            rejection.task_id,
            dependency_id,
            EdgeType.DEPENDS_ON,
            weight=1.0,
        )

        return RejectionDecision(
            decision_type=DecisionType.ACCEPT_AND_DEFER,
            rejection=rejection,
            rationale=f"Valid dependency. Task depends on {dependency_id}.",
            created_tasks=[dependency_id] if not self._task_exists(dependency_id) else [],
            deferred_task=rejection.task_id,
        )

    def _handle_infeasible(
        self,
        rejection: TaskRejection,
        task_context: Dict[str, Any],
    ) -> "RejectionDecision":
        """
        Handle INFEASIBLE rejection.

        Strategy: Accept reformulation, create new task with feasible scope.
        """
        # Create new task with reformulated scope
        new_task_id = self.manager.create_task(
            title=f"Reformulated: {task_context.get('title', 'Task')}",
            description=rejection.suggested_alternative,
            priority=task_context.get("priority", "medium"),
            category=task_context.get("category", "feature"),
        )

        # Link to original (marks it as replaced)
        self.manager.graph.add_edge(
            new_task_id,
            rejection.task_id,
            EdgeType.REFINES,
            weight=1.0,
        )

        # Mark original as infeasible
        self.manager.update_task(
            rejection.task_id,
            status="infeasible",
            properties={"reformulated_as": new_task_id},
        )

        return RejectionDecision(
            decision_type=DecisionType.ACCEPT_AND_REFORMULATE,
            rejection=rejection,
            rationale="Valid infeasibility. Created reformulated task.",
            created_tasks=[new_task_id],
        )

    def _handle_unclear_requirements(
        self,
        rejection: TaskRejection,
        task_context: Dict[str, Any],
    ) -> "RejectionDecision":
        """
        Handle UNCLEAR_REQUIREMENTS rejection.

        Strategy: Extract questions, create clarification blocker, defer.
        """
        # Create clarification task
        clarification_task_id = self.manager.create_task(
            title=f"Clarify requirements for {rejection.task_id[:20]}",
            description=rejection.reason_detail,
            priority="high",
            category="clarification",
        )

        # Link as blocker
        self.manager.graph.add_edge(
            rejection.task_id,
            clarification_task_id,
            EdgeType.BLOCKS,
            weight=1.0,
        )

        return RejectionDecision(
            decision_type=DecisionType.ACCEPT_AND_DEFER,
            rejection=rejection,
            rationale="Valid unclear requirements. Created clarification task.",
            created_tasks=[clarification_task_id],
            deferred_task=rejection.task_id,
        )

    def _accept_rejection(self, rejection: TaskRejection) -> "RejectionDecision":
        """Accept rejection and mark task as blocked."""
        self.manager.update_task(
            rejection.task_id,
            status="blocked",
            properties={"rejection": rejection.to_dict()},
        )

        return RejectionDecision(
            decision_type=DecisionType.ACCEPT,
            rejection=rejection,
            rationale="Rejection accepted.",
        )

    def _generate_override_message(self, issues: List[str]) -> str:
        """Generate helpful override message."""
        return f"""
Your rejection was not accepted due to the following issues:

{chr(10).join(f'- {issue}' for issue in issues)}

Please retry the task and ensure you:
1. Document at least 2 concrete attempts
2. Provide specific, measurable blocking factors
3. Suggest actionable alternatives
4. Include evidence supporting your rejection

Remember: Rejection must be earned through demonstrated effort.
"""

    def _extract_dependency_id(self, blocking_factor: str) -> Optional[str]:
        """Try to extract task/decision ID from blocking factor."""
        import re
        # Look for patterns like "task:T-..." or "decision:D-..."
        match = re.search(r'(task:T-\w+|decision:D-\w+)', blocking_factor)
        return match.group(1) if match else None

    def _task_exists(self, task_id: str) -> bool:
        """Check if task already exists in graph."""
        return self.manager.graph.get_node(task_id) is not None


class DecisionType(Enum):
    """Director's decision on how to handle rejection."""
    OVERRIDE = auto()              # Rejection invalid, provide more context
    ACCEPT = auto()                # Accept rejection, mark blocked
    ACCEPT_AND_DEFER = auto()      # Accept, defer until blocker resolved
    ACCEPT_AND_DECOMPOSE = auto()  # Accept, create sub-tasks
    ACCEPT_AND_REFORMULATE = auto() # Accept, create new task with feasible scope


@dataclass
class RejectionDecision:
    """
    Director's decision on how to handle a rejection.

    This is logged to GoT for learning patterns.
    """
    decision_type: DecisionType
    rejection: TaskRejection
    rationale: str

    # If OVERRIDE
    override_message: str = ""
    reassign_to: str = ""

    # If creating new tasks
    created_tasks: List[str] = field(default_factory=list)

    # If deferring
    deferred_task: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for event logging."""
        return {
            "decision_type": self.decision_type.name,
            "rejection": self.rejection.to_dict(),
            "rationale": self.rationale,
            "override_message": self.override_message,
            "reassign_to": self.reassign_to,
            "created_tasks": self.created_tasks,
            "deferred_task": self.deferred_task,
        }
```

---

## GoT Integration

### Logging Rejections as ThoughtNodes

```python
def log_rejection_to_got(
    graph: ThoughtGraph,
    rejection: TaskRejection,
    decision: RejectionDecision,
) -> str:
    """
    Log rejection to GoT for pattern learning.

    Creates nodes and edges capturing:
    - The rejection event
    - The blocking factors
    - The director's response
    - Pattern for future analysis

    Returns:
        rejection_node_id for reference
    """
    # Create rejection node
    rejection_node_id = f"rejection:{rejection.task_id}:{int(rejection.rejected_at.timestamp())}"

    graph.add_node(
        rejection_node_id,
        NodeType.OBSERVATION,
        f"Task {rejection.task_id} rejected: {rejection.reason_summary}",
        properties={
            "reason_type": rejection.reason_type.name,
            "agent_id": rejection.agent_id,
            "blocking_factor": rejection.blocking_factor,
            "what_attempted": rejection.what_attempted,
            "evidence": rejection.evidence,
        },
    )

    # Link to task
    graph.add_edge(
        rejection_node_id,
        rejection.task_id,
        EdgeType.OBSERVES,
        weight=1.0,
    )

    # Create blocking factor node if concrete blocker
    if rejection.reason_type == RejectionReason.BLOCKER:
        blocker_node_id = f"blocker:{rejection.rejected_at.timestamp()}"
        graph.add_node(
            blocker_node_id,
            NodeType.CONSTRAINT,
            rejection.blocking_factor,
            properties={"evidence": rejection.evidence},
        )
        graph.add_edge(
            blocker_node_id,
            rejection.task_id,
            EdgeType.BLOCKS,
            weight=1.0,
        )

    # Create decision node
    decision_node_id = f"decision:rejection-response:{int(rejection.rejected_at.timestamp())}"
    graph.add_node(
        decision_node_id,
        NodeType.DECISION,
        f"Rejection decision: {decision.decision_type.name}",
        properties={
            "rationale": decision.rationale,
            "created_tasks": decision.created_tasks,
            "deferred_task": decision.deferred_task,
        },
    )

    # Link decision to rejection
    graph.add_edge(
        decision_node_id,
        rejection_node_id,
        EdgeType.ANSWERS,
        weight=1.0,
    )

    # Link created tasks
    for task_id in decision.created_tasks:
        graph.add_edge(
            task_id,
            decision_node_id,
            EdgeType.IMPLEMENTS,
            weight=1.0,
        )

    return rejection_node_id
```

### Pattern Learning from Rejections

```python
def analyze_rejection_patterns(graph: ThoughtGraph) -> Dict[str, Any]:
    """
    Analyze rejection patterns in the graph.

    Identifies:
    - Which tasks get rejected most
    - Which agents reject most
    - Which blockers are most common
    - Success rate of override vs accept

    Returns:
        Pattern analysis for Director learning
    """
    rejection_nodes = [
        node for node in graph.nodes.values()
        if node.id.startswith("rejection:")
    ]

    # Group by reason type
    by_reason = {}
    for node in rejection_nodes:
        reason = node.properties.get("reason_type", "UNKNOWN")
        by_reason[reason] = by_reason.get(reason, 0) + 1

    # Group by agent
    by_agent = {}
    for node in rejection_nodes:
        agent = node.properties.get("agent_id", "unknown")
        by_agent[agent] = by_agent.get(agent, 0) + 1

    # Extract common blockers
    common_blockers = []
    for node in rejection_nodes:
        if node.properties.get("reason_type") == "BLOCKER":
            blocker = node.properties.get("blocking_factor")
            if blocker:
                common_blockers.append(blocker)

    # Analyze decision outcomes
    decision_nodes = [
        node for node in graph.nodes.values()
        if node.id.startswith("decision:rejection-response:")
    ]

    decision_types = {}
    for node in decision_nodes:
        # Extract decision type from properties if stored
        dec_type = "UNKNOWN"
        if "rationale" in node.properties:
            if "override" in node.properties["rationale"].lower():
                dec_type = "OVERRIDE"
            elif "decompose" in node.properties["rationale"].lower():
                dec_type = "DECOMPOSE"
            elif "defer" in node.properties["rationale"].lower():
                dec_type = "DEFER"
            else:
                dec_type = "ACCEPT"

        decision_types[dec_type] = decision_types.get(dec_type, 0) + 1

    return {
        "total_rejections": len(rejection_nodes),
        "by_reason": by_reason,
        "by_agent": by_agent,
        "common_blockers": common_blockers[:10],  # Top 10
        "decision_outcomes": decision_types,
        "override_rate": decision_types.get("OVERRIDE", 0) / max(len(decision_nodes), 1),
    }
```

---

## Integration with Handoff System

### Enhanced HandoffManager

```python
# Extension to existing HandoffManager in got_utils.py

class HandoffManager:
    """Extended with rejection protocol support."""

    # ... existing methods ...

    def reject_handoff(
        self,
        handoff_id: str,
        agent: str,
        rejection: TaskRejection,
    ) -> bool:
        """
        Reject a handoff with structured rejection data.

        This replaces the simple log_handoff_reject with full protocol.
        """
        # Validate rejection
        validator = RejectionValidator()
        task_context = self._get_task_context(rejection.task_id)
        is_valid, issues = validator.validate(rejection, task_context)

        if not is_valid:
            # Log validation failure
            self.event_log.log(
                "handoff.reject.invalid",
                handoff_id=handoff_id,
                agent=agent,
                rejection=rejection.to_dict(),
                validation_issues=issues,
            )
            return False

        # Log valid rejection
        self.event_log.log(
            "handoff.reject",
            handoff_id=handoff_id,
            agent=agent,
            rejection=rejection.to_dict(),
        )

        if handoff_id in self._active_handoffs:
            self._active_handoffs[handoff_id]["status"] = "rejected"
            self._active_handoffs[handoff_id]["rejected_at"] = datetime.now().isoformat()
            self._active_handoffs[handoff_id]["rejection"] = rejection.to_dict()

        return True

    def handle_rejection_response(
        self,
        handoff_id: str,
        decision: RejectionDecision,
    ) -> bool:
        """
        Log Director's response to rejection.

        This creates the full rejection → response → outcome chain.
        """
        self.event_log.log(
            "handoff.reject.response",
            handoff_id=handoff_id,
            decision=decision.to_dict(),
        )

        # Update handoff status based on decision
        if handoff_id in self._active_handoffs:
            if decision.decision_type == DecisionType.OVERRIDE:
                self._active_handoffs[handoff_id]["status"] = "overridden"
                # Re-initiate with override message
                self._active_handoffs[handoff_id]["override_message"] = decision.override_message
            else:
                self._active_handoffs[handoff_id]["status"] = "rejection_accepted"

        return True

    def _get_task_context(self, task_id: str) -> Dict[str, Any]:
        """Get task context for validation."""
        # This would query the graph for task details
        # Simplified for example
        return {
            "title": "Example task",
            "scope": "Original scope",
            "priority": "medium",
            "category": "feature",
        }
```

---

## Example Usage Flow

### Scenario: Agent Rejects Task Due to Scope Creep

```python
# 1. Agent attempts task
agent_id = "sub-agent-1"
task_id = "task:T-20251220-153045-a1b2"
handoff_id = "handoff:H-20251220-153100-c3d4"

# Agent discovers scope is much larger than expected
attempts = [
    "Analyzed auth/ module structure - 47 files, 12,000 LOC",
    "Reviewed existing tests - 89 test files to update",
    "Checked API documentation - 23 endpoints need changes",
    "Estimated effort - 16-20 hours for complete implementation",
]

# 2. Agent creates structured rejection
rejection = TaskRejection(
    task_id=task_id,
    handoff_id=handoff_id,
    agent_id=agent_id,
    reason_type=RejectionReason.SCOPE_CREEP,
    reason_summary="Add OAuth login requires auth system redesign",
    reason_detail="""
Original task: "Add OAuth login button to login page"

Investigation revealed this requires:
1. Refactoring entire auth module for plugin architecture
2. Updating 23 API endpoints to support multiple auth methods
3. Migrating existing user sessions to new auth system
4. Updating 89 test files

Original scope: Add UI button and OAuth flow (~2 hours)
Actual scope: Auth system redesign (~16-20 hours)

Scope growth factor: 8-10x
""",
    what_attempted=attempts,
    blocking_factor="Cannot add OAuth without refactoring auth module for extensibility",
    evidence=[
        {
            "type": "file_analysis",
            "data": {"files_requiring_changes": 47, "lines_of_code": 12000},
            "source": "static analysis of auth/ module",
        },
        {
            "type": "complexity_metric",
            "data": {"test_files_to_update": 89, "api_endpoints_affected": 23},
            "source": "impact analysis",
        },
        {
            "type": "time_estimate",
            "data": {"original_estimate_hours": 2, "revised_estimate_hours": 18},
            "source": "engineering judgment",
        },
    ],
    suggested_alternative="""
Decompose into 3 sequential tasks:

1. Refactor auth module for plugin architecture (8 hours)
   - Extract interface for auth providers
   - Migrate existing password auth to plugin
   - Update tests for new architecture

2. Add OAuth provider plugin (4 hours)
   - Implement OAuth plugin using new interface
   - Add OAuth-specific tests
   - Document OAuth configuration

3. Add OAuth UI and flows (2 hours)
   - Add OAuth button to login page
   - Implement OAuth redirect flows
   - Update user documentation
""",
    alternative_tasks=[
        {
            "title": "Refactor auth module for plugin architecture",
            "scope": "Extract auth provider interface, migrate existing auth",
            "dependencies": "none",
            "estimate": "8 hours",
        },
        {
            "title": "Implement OAuth provider plugin",
            "scope": "Add OAuth using new plugin interface",
            "dependencies": "Requires auth refactor task",
            "estimate": "4 hours",
        },
        {
            "title": "Add OAuth UI and user flows",
            "scope": "Add login button, redirect flows, documentation",
            "dependencies": "Requires OAuth plugin task",
            "estimate": "2 hours",
        },
    ],
    task_original_scope="Add OAuth login button to login page (~2 hours)",
    scope_growth_factor=9.0,
)

# 3. Agent submits rejection through HandoffManager
manager = HandoffManager(event_log)
rejection_accepted = manager.reject_handoff(
    handoff_id=handoff_id,
    agent=agent_id,
    rejection=rejection,
)

if not rejection_accepted:
    print("Rejection validation failed - retry with more evidence")
    sys.exit(1)

# 4. Director processes rejection
director_handler = DirectorRejectionHandler(project_manager)
decision = director_handler.handle_rejection(
    rejection=rejection,
    task_context={
        "title": "Add OAuth login",
        "scope": "Add OAuth login button to login page",
        "priority": "medium",
        "category": "feature",
    },
)

# 5. Director logs decision
manager.handle_rejection_response(handoff_id, decision)

# 6. Log to GoT for learning
rejection_node_id = log_rejection_to_got(
    project_manager.graph,
    rejection,
    decision,
)

# 7. Output result
print(f"Rejection decision: {decision.decision_type.name}")
print(f"Rationale: {decision.rationale}")
print(f"Created tasks: {decision.created_tasks}")
print(f"GoT node: {rejection_node_id}")

# Result:
# Rejection decision: ACCEPT_AND_DECOMPOSE
# Rationale: Valid scope creep. Decomposed into 3 sub-tasks.
# Created tasks: ['task:T-20251220-160001-e5f6', 'task:T-20251220-160002-g7h8', 'task:T-20251220-160003-i9j0']
# GoT node: rejection:task:T-20251220-153045-a1b2:1734710000
```

### Scenario: Agent Attempts Lazy Rejection (Gets Overridden)

```python
# Agent tries to reject with insufficient effort
lazy_rejection = TaskRejection(
    task_id=task_id,
    handoff_id=handoff_id,
    agent_id=agent_id,
    reason_type=RejectionReason.BLOCKER,
    reason_summary="Too complex",
    reason_detail="This task is too complex for me to handle.",
    what_attempted=[
        "Looked at the code",
        "Tried to understand it",
    ],
    blocking_factor="It's confusing",
    evidence=[],
    suggested_alternative="Ask someone else to do it",
)

# Validation fails
rejection_accepted = manager.reject_handoff(
    handoff_id=handoff_id,
    agent=agent_id,
    rejection=lazy_rejection,
)

# rejection_accepted = False
# Validation issues logged:
# - Attempts are too vague. At least 1 must be concrete and specific.
# - BLOCKER rejection requires evidence (error logs, status checks, etc.)
# - Blocking factor contains vague language. Must be concrete and measurable.
# - Suggestion is too generic. Provide concrete actionable steps or task breakdown.
```

---

## CLI Integration

### got_utils.py Commands

```bash
# Reject a handoff with structured data
python scripts/got_utils.py handoff reject HANDOFF_ID \
  --agent sub-agent-1 \
  --reason SCOPE_CREEP \
  --summary "Task scope grew 5x during investigation" \
  --detail @rejection_detail.md \
  --attempted "Analyzed codebase" "Ran tests" "Checked docs" \
  --blocker "Requires auth system redesign" \
  --alternative @alternative_plan.md \
  --scope-growth 5.0

# View rejection details
python scripts/got_utils.py handoff status HANDOFF_ID

# Analyze rejection patterns
python scripts/got_utils.py analyze rejections --since 2025-12-01

# Director responds to rejection
python scripts/got_utils.py handoff rejection-response HANDOFF_ID \
  --decision ACCEPT_AND_DECOMPOSE \
  --created-tasks task:T-xxx task:T-yyy task:T-zzz
```

---

## Summary

The Agent Rejection Protocol provides:

1. **Structured rejection reasons** - 5 valid types with specific validation
2. **Evidence requirements** - Agents must demonstrate effort and provide concrete blockers
3. **Actionable alternatives** - Rejections must help, not just complain
4. **Director response paths** - Accept, override, decompose, or reformulate
5. **GoT integration** - All rejections logged for pattern learning
6. **Validation layer** - Prevents lazy rejections that waste time

**Key constraints enforced:**
- Minimum 2 concrete attempts before rejection
- Specific, measurable blocking factors (no vague language)
- Actionable alternatives with concrete steps
- Evidence supporting the rejection claim
- Higher burden of proof for INFEASIBLE rejections

**Flows through GitHub PRs:**
- Rejection events logged to `.got/events/*.jsonl` (git-tracked)
- Director responses create new tasks visible in graph
- Pattern analysis available across branches
- Each PR can see rejection history for context

This enables honest, productive communication about blockers while preventing agents from giving up too easily.
