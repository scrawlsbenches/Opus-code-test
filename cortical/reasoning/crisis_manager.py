"""
Crisis Management: Handling Failures and Recovery.

This module implements the crisis management framework from Part 13 of
docs/complex-reasoning-workflow.md. It provides:

- Crisis severity classification (HICCUP → OBSTACLE → WALL → CRISIS)
- Repeated failure detection and escalation
- Recovery procedures (rollback, partial recovery, knowledge preservation)
- Scope creep detection
- Blocked dependency management

Design Philosophy:
    How you handle failure determines long-term success. Systems that can't
    recover gracefully become fragile and untrusted. This module provides
    structured approaches to common failure modes.

Key Insight:
    "The temptation is to keep trying. But insanity is doing the same thing
    expecting different results."
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid


class CrisisLevel(Enum):
    """
    Crisis severity levels from docs/complex-reasoning-workflow.md Part 13.1.

    HICCUP: Self-recoverable
        - Test failure with obvious fix
        - Minor misunderstanding, easily clarified
        - Time estimate off by <50%
        Response: Fix it, note it, continue

    OBSTACLE: Needs adaptation
        - Verification repeatedly failing
        - Blocked by external dependency
        - Approach hitting diminishing returns
        Response: Pause, analyze, adjust approach

    WALL: Needs human intervention
        - Fundamental assumption proven false
        - Multiple approaches have failed
        - Scope has grown beyond original boundaries
        Response: Stop, document, escalate to human

    CRISIS: Immediate stop required
        - Work is causing damage (breaking other systems)
        - Security or data integrity issue discovered
        - Work contradicts explicit user values
        Response: STOP NOW, preserve state, alert human
    """
    HICCUP = 1  # Self-recoverable
    OBSTACLE = 2  # Needs adaptation
    WALL = 3  # Needs human intervention
    CRISIS = 4  # Immediate stop required


class RecoveryAction(Enum):
    """Actions that can be taken in response to a crisis."""
    CONTINUE = auto()  # Fix and continue
    ADAPT = auto()  # Adjust approach
    ROLLBACK = auto()  # Return to known good state
    PARTIAL_RECOVER = auto()  # Salvage what works
    ESCALATE = auto()  # Hand off to human
    STOP = auto()  # Immediate halt


@dataclass
class CrisisEvent:
    """
    Record of a crisis event.

    Captures what happened, how severe it was, and what action was taken.
    This forms an audit trail and training data for future crisis prevention.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    level: CrisisLevel = CrisisLevel.HICCUP
    description: str = ""
    context: Dict[str, Any] = field(default_factory=dict)  # What was happening
    timestamp: datetime = field(default_factory=datetime.now)
    action_taken: Optional[RecoveryAction] = None
    resolution: Optional[str] = None
    resolved_at: Optional[datetime] = None
    lessons_learned: List[str] = field(default_factory=list)

    def resolve(self, action: RecoveryAction, resolution: str) -> None:
        """Mark crisis as resolved."""
        self.action_taken = action
        self.resolution = resolution
        self.resolved_at = datetime.now()

    def add_lesson(self, lesson: str) -> None:
        """Record a lesson learned from this crisis."""
        self.lessons_learned.append(lesson)


@dataclass
class FailureAttempt:
    """
    Record of a single attempt that failed.

    Used in the repeated failure detection system (Part 13.2):
    "Attempt 1 failed → Normal: Investigate, hypothesize, fix
     Attempt 2 failed → Concern: Was hypothesis wrong?
     Attempt 3 failed → WARNING: Pattern suggests deeper issue
     Attempt 4+ → ESCALATE: Human intervention required"
    """
    attempt_number: int
    hypothesis: str  # What we thought was wrong
    action_taken: str  # What we tried
    result: str  # What happened
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RepeatedFailureTracker:
    """
    Tracker for repeated failures on the same issue.

    Implements the escalation ladder from docs/complex-reasoning-workflow.md Part 13.2.
    After 3 failures, we stop trying random fixes and escalate.
    """
    issue_description: str
    attempts: List[FailureAttempt] = field(default_factory=list)
    escalated: bool = False
    escalation_report: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

    def record_attempt(self, hypothesis: str, action: str, result: str) -> int:
        """
        Record a failure attempt.

        Returns:
            Current attempt count
        """
        attempt = FailureAttempt(
            attempt_number=len(self.attempts) + 1,
            hypothesis=hypothesis,
            action_taken=action,
            result=result,
        )
        self.attempts.append(attempt)
        return len(self.attempts)

    def should_escalate(self) -> bool:
        """Check if we've exceeded the failure threshold (3+ attempts)."""
        return len(self.attempts) >= 3 and not self.escalated

    def get_escalation_report(self) -> str:
        """
        Generate an escalation report for human review.

        Format from docs/complex-reasoning-workflow.md Part 13.2.
        """
        lines = [
            "## Escalation: Repeated Verification Failure",
            "",
            f"**What I'm trying to do:**",
            f"{self.issue_description}",
            "",
            "**Attempts made:**",
        ]

        for attempt in self.attempts:
            lines.append(
                f"{attempt.attempt_number}. [{attempt.hypothesis}] → {attempt.result}"
            )

        # Analyze patterns
        hypotheses = [a.hypothesis for a in self.attempts]
        unique_hypotheses = set(hypotheses)

        lines.extend([
            "",
            "**What I've learned:**",
            f"- Tried {len(unique_hypotheses)} unique hypotheses",
            f"- All {len(self.attempts)} attempts failed",
            "",
            "**Current hypotheses:**",
        ])

        # List remaining hypotheses with evidence
        for hyp in unique_hypotheses:
            count = hypotheses.count(hyp)
            lines.append(f"- {hyp}: Tried {count} time(s)")

        lines.extend([
            "",
            "**Help needed:**",
            "- Review assumptions - am I solving the right problem?",
            "- Suggest alternative approaches I haven't considered",
            "- Provide additional context or constraints I may be missing",
        ])

        return "\n".join(lines)


@dataclass
class ScopeCreepDetector:
    """
    Detector for scope creep during work.

    From docs/complex-reasoning-workflow.md Part 13.4:
    "Warning signs of scope creep:
    - 'Just one more thing' has been said 3+ times
    - Original time estimate is exceeded by 2x
    - You're modifying files you didn't expect to touch
    - The solution now requires learning new concepts
    - You can't remember the original goal clearly"
    """
    original_scope: str
    original_files: List[str] = field(default_factory=list)
    original_estimate_minutes: int = 30
    started_at: datetime = field(default_factory=datetime.now)

    # Tracking
    additions: List[str] = field(default_factory=list)  # "Just one more thing"
    unexpected_files: List[str] = field(default_factory=list)
    new_concepts: List[str] = field(default_factory=list)

    def record_addition(self, addition: str) -> None:
        """Record a scope addition ('just one more thing')."""
        self.additions.append(addition)

    def record_unexpected_file(self, file_path: str) -> None:
        """Record a file modification that wasn't expected."""
        if file_path not in self.original_files and file_path not in self.unexpected_files:
            self.unexpected_files.append(file_path)

    def record_new_concept(self, concept: str) -> None:
        """Record a new concept that needs to be learned."""
        if concept not in self.new_concepts:
            self.new_concepts.append(concept)

    def elapsed_minutes(self) -> float:
        """Get elapsed time in minutes."""
        return (datetime.now() - self.started_at).total_seconds() / 60

    def detect_creep(self) -> Tuple[bool, List[str]]:
        """
        Detect if scope creep has occurred.

        Returns:
            (is_creeping, list of warning signs)
        """
        warnings = []

        # Check for "just one more thing" pattern
        if len(self.additions) >= 3:
            warnings.append(f"'Just one more thing' said {len(self.additions)} times")

        # Check time estimate
        if self.elapsed_minutes() > self.original_estimate_minutes * 2:
            warnings.append(
                f"Time exceeded by {self.elapsed_minutes() / self.original_estimate_minutes:.1f}x"
            )

        # Check unexpected files
        if len(self.unexpected_files) >= 3:
            warnings.append(f"Modifying {len(self.unexpected_files)} unexpected files")

        # Check new concepts
        if len(self.new_concepts) >= 2:
            warnings.append(f"Had to learn {len(self.new_concepts)} new concepts")

        return len(warnings) > 0, warnings

    def generate_alert(self) -> str:
        """
        Generate a scope creep alert document.

        Format from docs/complex-reasoning-workflow.md Part 13.4.
        """
        is_creeping, warnings = self.detect_creep()

        lines = [
            "## Scope Creep Alert",
            "",
            "**Original scope:**",
            self.original_scope,
            "",
            "**Current scope:**",
            f"{self.original_scope} + {len(self.additions)} additions",
            "",
            "**Warning signs:**",
        ]
        for warning in warnings:
            lines.append(f"- {warning}")

        lines.extend([
            "",
            "**Additions made:**",
        ])
        for addition in self.additions:
            lines.append(f"- {addition}")

        lines.extend([
            "",
            "**Options:**",
            "1. Finish expanded scope (estimate: ? more time)",
            "2. Return to original scope (cut additions)",
            "3. Pause and reframe (new planning session)",
        ])

        return "\n".join(lines)


@dataclass
class BlockedDependency:
    """
    Record of a blocked dependency.

    From docs/complex-reasoning-workflow.md Part 13.3:
    External or internal dependencies that prevent progress.
    """
    description: str
    dependency_type: str  # 'external' or 'internal'
    blocking_since: datetime = field(default_factory=datetime.now)
    workaround_possible: bool = False
    workaround_description: Optional[str] = None
    alternative_work: Optional[str] = None  # What to do while blocked
    expected_resolution: Optional[str] = None
    resolved: bool = False
    resolution_notes: Optional[str] = None


class CrisisManager:
    """
    Central manager for crisis detection, classification, and recovery.

    This class orchestrates the crisis management system, providing:
    - Crisis event recording and tracking
    - Severity classification
    - Recovery procedure guidance
    - Escalation management
    - Knowledge preservation from failures

    Example:
        >>> manager = CrisisManager()
        >>> event = manager.record_crisis(
        ...     CrisisLevel.OBSTACLE,
        ...     "Tests keep failing after each fix",
        ...     context={'test_file': 'test_auth.py'}
        ... )
        >>> action = manager.recommend_action(event)
        >>> print(action)  # RecoveryAction.ADAPT
    """

    def __init__(self):
        """Initialize the crisis manager."""
        self._events: List[CrisisEvent] = []
        self._failure_trackers: Dict[str, RepeatedFailureTracker] = {}
        self._scope_detectors: Dict[str, ScopeCreepDetector] = {}
        self._blocked_dependencies: List[BlockedDependency] = []

        # Callbacks
        self._on_crisis: List[Callable[[CrisisEvent], None]] = []
        self._on_escalation: List[Callable[[CrisisEvent], None]] = []

    def record_crisis(
        self,
        level: CrisisLevel,
        description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> CrisisEvent:
        """
        Record a new crisis event.

        Args:
            level: Severity level
            description: What happened
            context: Additional context

        Returns:
            CrisisEvent instance
        """
        event = CrisisEvent(
            level=level,
            description=description,
            context=context or {},
        )
        self._events.append(event)

        # Notify handlers
        for handler in self._on_crisis:
            try:
                handler(event)
            except Exception:
                pass

        # Auto-escalate CRISIS level
        if level == CrisisLevel.CRISIS:
            for handler in self._on_escalation:
                try:
                    handler(event)
                except Exception:
                    pass

        return event

    def recommend_action(self, event: CrisisEvent) -> RecoveryAction:
        """
        Recommend a recovery action based on crisis level.

        Args:
            event: The crisis event

        Returns:
            Recommended RecoveryAction
        """
        recommendations = {
            CrisisLevel.HICCUP: RecoveryAction.CONTINUE,
            CrisisLevel.OBSTACLE: RecoveryAction.ADAPT,
            CrisisLevel.WALL: RecoveryAction.ESCALATE,
            CrisisLevel.CRISIS: RecoveryAction.STOP,
        }
        return recommendations.get(event.level, RecoveryAction.ESCALATE)

    def create_failure_tracker(self, issue_description: str) -> RepeatedFailureTracker:
        """
        Create a tracker for repeated failures on an issue.

        Args:
            issue_description: What we're trying to fix

        Returns:
            RepeatedFailureTracker instance
        """
        tracker = RepeatedFailureTracker(issue_description=issue_description)
        self._failure_trackers[issue_description] = tracker
        return tracker

    def create_scope_detector(
        self,
        original_scope: str,
        original_files: List[str],
        estimate_minutes: int = 30
    ) -> ScopeCreepDetector:
        """
        Create a scope creep detector for a task.

        Args:
            original_scope: Description of original scope
            original_files: Files expected to be modified
            estimate_minutes: Original time estimate

        Returns:
            ScopeCreepDetector instance
        """
        detector = ScopeCreepDetector(
            original_scope=original_scope,
            original_files=original_files,
            original_estimate_minutes=estimate_minutes,
        )
        self._scope_detectors[original_scope] = detector
        return detector

    def record_blocked(
        self,
        description: str,
        dependency_type: str = "external"
    ) -> BlockedDependency:
        """
        Record a blocked dependency.

        Args:
            description: What's blocked
            dependency_type: 'external' or 'internal'

        Returns:
            BlockedDependency instance
        """
        blocked = BlockedDependency(
            description=description,
            dependency_type=dependency_type,
        )
        self._blocked_dependencies.append(blocked)
        return blocked

    def get_active_blockers(self) -> List[BlockedDependency]:
        """Get all unresolved blockers."""
        return [b for b in self._blocked_dependencies if not b.resolved]

    def get_unresolved_crises(self) -> List[CrisisEvent]:
        """Get all unresolved crisis events."""
        return [e for e in self._events if e.resolved_at is None]

    def get_crises_by_level(self, level: CrisisLevel) -> List[CrisisEvent]:
        """Get all crises of a specific level."""
        return [e for e in self._events if e.level == level]

    def register_crisis_handler(self, handler: Callable[[CrisisEvent], None]) -> None:
        """Register a handler called when any crisis is recorded."""
        self._on_crisis.append(handler)

    def register_escalation_handler(self, handler: Callable[[CrisisEvent], None]) -> None:
        """Register a handler called when a crisis is escalated."""
        self._on_escalation.append(handler)

    def get_lessons_learned(self) -> List[str]:
        """Get all lessons learned from resolved crises."""
        lessons = []
        for event in self._events:
            if event.resolved_at and event.lessons_learned:
                lessons.extend(event.lessons_learned)
        return lessons

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of crisis management state."""
        by_level = {}
        for level in CrisisLevel:
            by_level[level.name] = len(self.get_crises_by_level(level))

        return {
            'total_events': len(self._events),
            'by_level': by_level,
            'unresolved': len(self.get_unresolved_crises()),
            'active_blockers': len(self.get_active_blockers()),
            'failure_trackers': len(self._failure_trackers),
            'scope_detectors': len(self._scope_detectors),
            'lessons_learned': len(self.get_lessons_learned()),
        }


# =============================================================================
# STUB CLASSES FOR COMPLEX IMPLEMENTATIONS
# =============================================================================


class RecoveryProcedures:
    """
    STUB: Automated recovery procedures for different crisis types.

    Full Implementation Would:
    --------------------------
    From docs/complex-reasoning-workflow.md Part 13.5:

    1. Full Rollback:
       - Automatically stash current state
       - Identify last known good commit
       - Restore to that state
       - Generate memory document about what was learned

    2. Partial Recovery:
       - Identify which files/changes are salvageable
       - Commit working pieces separately
       - Stash broken pieces for analysis
       - Create follow-up tasks for remaining work

    3. Knowledge Preservation:
       - Extract learnings even from failed approaches
       - Document what was tried and why it failed
       - Identify conditions that would make approach work
       - Record red flags for future reference

    Integration Points:
    -------------------
    - Git operations for rollback
    - Task system for follow-up creation
    - Memory system for knowledge preservation
    - Metrics for failure analysis
    """

    def full_rollback(self, checkpoint: str) -> Dict[str, Any]:
        """
        STUB: Perform full rollback to a checkpoint.

        Full Implementation Would:
        - git stash save "failed-attempt-{timestamp}"
        - git checkout {checkpoint}
        - Generate memory document

        Returns:
            {'success': bool, 'stash_ref': str, 'restored_to': str}
        """
        return {
            'success': True,
            'stash_ref': 'stash@{0}',
            'restored_to': checkpoint,
            'note': 'STUB: Would perform actual git operations',
        }

    def partial_recovery(self, working_files: List[str], broken_files: List[str]) -> Dict[str, Any]:
        """
        STUB: Salvage working parts, preserve broken parts for analysis.

        Returns:
            {'committed_files': list, 'stashed_files': list, 'task_created': str}
        """
        return {
            'committed_files': working_files,
            'stashed_files': broken_files,
            'task_created': 'T-STUB-001',
            'note': 'STUB: Would perform actual git operations',
        }

    def generate_failure_analysis(self, event: CrisisEvent) -> str:
        """
        STUB: Generate a failure analysis document.

        Full Implementation Would:
        - Analyze event context and timeline
        - Extract patterns from failure attempts
        - Identify root causes
        - Suggest preventive measures

        Returns:
            Markdown-formatted analysis document
        """
        return f"""## Failed Approach Analysis: {event.description}

**Goal:** [Would extract from context]

**Approach:** [Would analyze attempts]

**Why it failed:**
[Would identify root cause from patterns]

**What we learned:**
- [Would extract from event history]

**Red flags we should have noticed:**
- [Would identify early warning signs]

**Conditions that would make this work:**
- [Would suggest when to revisit]
"""


class CrisisPredictor:
    """
    STUB: ML-based crisis prediction to prevent issues before they occur.

    Full Implementation Would:
    --------------------------
    1. Feature extraction from ongoing work:
       - Time spent in each phase
       - Number of iterations
       - File modification patterns
       - Comment marker density

    2. Pattern matching against historical crises:
       - Similar time patterns
       - Similar scope patterns
       - Similar failure sequences

    3. Risk scoring:
       - Probability of escalation
       - Expected time to crisis
       - Recommended preventive action

    4. Proactive alerts:
       - "You're approaching the pattern that led to crisis X"
       - "Consider: [preventive action]"

    Training Data:
    --------------
    - Historical crisis events with context
    - Successful recovery patterns
    - Scope creep progression sequences
    - Failure escalation timelines
    """

    def predict_risk(self, current_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        STUB: Predict crisis risk from current context.

        Returns:
            {'risk_level': float, 'likely_type': CrisisLevel, 'suggestions': list}
        """
        return {
            'risk_level': 0.2,  # Low risk placeholder
            'likely_type': CrisisLevel.HICCUP,
            'suggestions': [
                'Continue monitoring',
                'Consider time-boxing current phase',
            ],
            'note': 'STUB: Would use ML model for prediction',
        }

    def get_similar_past_crises(self, context: Dict[str, Any]) -> List[CrisisEvent]:
        """
        STUB: Find historically similar crises.

        Returns:
            List of similar past CrisisEvent instances
        """
        return []  # Would query historical data
