"""
Agent Rejection Protocol

Enables agents to reject tasks with structured, validated reasons while
preventing lazy rejections. Integrates with GoT handoff system for
auto-replanning workflows.

See docs/agent-rejection-protocol.md for complete design documentation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
import re

from .graph_of_thought import NodeType, EdgeType
from .thought_graph import ThoughtGraph


# =============================================================================
# REJECTION REASONS
# =============================================================================


class RejectionReason(Enum):
    """
    Valid reasons for rejecting a task.

    Each reason has specific validation requirements to prevent abuse.
    See docs/agent-rejection-protocol.md for detailed validation rules.
    """

    BLOCKER = auto()
    """External blocker prevents progress (requires evidence)."""

    SCOPE_CREEP = auto()
    """Task evolved beyond original scope (requires 2x growth factor)."""

    MISSING_DEPENDENCY = auto()
    """Internal dependency not yet met (must identify specific dependency)."""

    INFEASIBLE = auto()
    """Task fundamentally cannot be done (highest burden of proof)."""

    UNCLEAR_REQUIREMENTS = auto()
    """Task requirements ambiguous (must list specific questions)."""


# =============================================================================
# TASK REJECTION
# =============================================================================


@dataclass
class TaskRejection:
    """
    Structured rejection with evidence and alternatives.

    Agents must provide:
    - What they attempted (demonstrates effort)
    - Specific blocking factor (concrete, not vague)
    - Evidence supporting the rejection
    - Actionable alternative (helps, doesn't just complain)
    """

    # Identity
    task_id: str
    handoff_id: str
    agent_id: str
    rejected_at: datetime = field(default_factory=datetime.now)

    # Rejection reason
    reason_type: RejectionReason = RejectionReason.BLOCKER
    reason_summary: str = ""
    reason_detail: str = ""

    # What was attempted (REQUIRED - demonstrates effort)
    what_attempted: List[str] = field(default_factory=list)

    # Blocking factor (specific, concrete)
    blocking_factor: str = ""

    # Evidence (supports the rejection)
    evidence: List[Dict[str, Any]] = field(default_factory=list)

    # Suggested alternative (REQUIRED - must help)
    suggested_alternative: str = ""

    # Alternative tasks (optional decomposition)
    alternative_tasks: List[Dict[str, str]] = field(default_factory=list)

    # Metadata
    task_original_scope: str = ""
    scope_growth_factor: Optional[float] = None

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
        data = data.copy()
        data["reason_type"] = RejectionReason[data["reason_type"]]
        data["rejected_at"] = datetime.fromisoformat(data["rejected_at"])
        return cls(**data)


# =============================================================================
# REJECTION VALIDATION
# =============================================================================


class RejectionValidator:
    """
    Validates that rejections are legitimate and not lazy.

    Enforces:
    - Minimum effort demonstrated (at least 2 concrete attempts)
    - Specific blocking factors (no vague language)
    - Actionable alternatives (concrete steps)
    - Reason-specific validation rules
    """

    def validate(
        self,
        rejection: TaskRejection,
        task_context: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
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
                "Blocking factor contains vague language. Must be concrete and measurable."
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

    def _validate_scope_creep(
        self,
        rejection: TaskRejection,
        task_context: Dict[str, Any]
    ) -> List[str]:
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


# =============================================================================
# DIRECTOR DECISION
# =============================================================================


class DecisionType(Enum):
    """Director's decision on how to handle rejection."""
    OVERRIDE = auto()              # Rejection invalid, provide more context
    ACCEPT = auto()                # Accept rejection, mark blocked
    ACCEPT_AND_DEFER = auto()      # Accept, defer until blocker resolved
    ACCEPT_AND_DECOMPOSE = auto()  # Accept, create sub-tasks
    ACCEPT_AND_REFORMULATE = auto()  # Accept, create new task with feasible scope


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


# =============================================================================
# GOT INTEGRATION
# =============================================================================


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
    - Patterns for future analysis

    Args:
        graph: ThoughtGraph to log into
        rejection: The task rejection
        decision: Director's decision on the rejection

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
        blocker_node_id = f"blocker:{int(rejection.rejected_at.timestamp())}"
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


def analyze_rejection_patterns(graph: ThoughtGraph) -> Dict[str, Any]:
    """
    Analyze rejection patterns in the graph.

    Identifies:
    - Which tasks get rejected most
    - Which agents reject most
    - Which blockers are most common
    - Success rate of override vs accept

    Args:
        graph: ThoughtGraph to analyze

    Returns:
        Pattern analysis for Director learning
    """
    rejection_nodes = [
        node for node in graph.nodes.values()
        if node.id.startswith("rejection:")
    ]

    # Group by reason type
    by_reason: Dict[str, int] = {}
    for node in rejection_nodes:
        reason = node.properties.get("reason_type", "UNKNOWN")
        by_reason[reason] = by_reason.get(reason, 0) + 1

    # Group by agent
    by_agent: Dict[str, int] = {}
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

    decision_types: Dict[str, int] = {}
    for node in decision_nodes:
        # Extract decision type from properties if stored
        dec_type = "UNKNOWN"
        if "rationale" in node.properties:
            rationale_lower = node.properties["rationale"].lower()
            if "override" in rationale_lower:
                dec_type = "OVERRIDE"
            elif "decompose" in rationale_lower:
                dec_type = "DECOMPOSE"
            elif "defer" in rationale_lower:
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
