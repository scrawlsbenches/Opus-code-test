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
import os
import re
import shutil
import subprocess
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
    Automated recovery procedures for different crisis types.

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

    def __init__(self, memory_path: Optional[str] = None):
        """
        Initialize recovery procedures.

        Args:
            memory_path: Directory for memory documents (default: samples/memories/)
        """
        self._git_available = shutil.which('git') is not None
        self._recovery_log: List[Dict[str, Any]] = []
        self._memory_path = memory_path or 'samples/memories/'
        self._outcome_history: List[Dict[str, Any]] = []

    def suggest_recovery(self, crisis: CrisisEvent) -> List[RecoveryAction]:
        """Suggest recovery actions based on crisis level."""
        suggestions = []
        if crisis.level == CrisisLevel.HICCUP:
            suggestions.append(RecoveryAction.CONTINUE)
        elif crisis.level == CrisisLevel.OBSTACLE:
            suggestions.append(RecoveryAction.ADAPT)
            suggestions.append(RecoveryAction.ROLLBACK)
        elif crisis.level == CrisisLevel.WALL:
            suggestions.append(RecoveryAction.ESCALATE)
            suggestions.append(RecoveryAction.PARTIAL_RECOVER)
        elif crisis.level == CrisisLevel.CRISIS:
            suggestions.append(RecoveryAction.STOP)
            suggestions.append(RecoveryAction.ESCALATE)
        return suggestions

    def execute_recovery(self, action: RecoveryAction, context: Dict[str, Any]) -> bool:
        """Execute a recovery action."""
        try:
            if action in (RecoveryAction.CONTINUE, RecoveryAction.STOP,
                         RecoveryAction.ESCALATE, RecoveryAction.ADAPT):
                return True
            elif action == RecoveryAction.ROLLBACK:
                checkpoint = context.get('checkpoint', '')
                if not checkpoint:
                    checkpoint = self.get_last_good_commit()
                if checkpoint:
                    result = self.full_rollback(checkpoint, context.get('description', ''))
                    return result.get('success', False)
                return False
            elif action == RecoveryAction.PARTIAL_RECOVER:
                working = context.get('working_files', [])
                broken = context.get('broken_files', [])
                if working or broken:
                    result = self.partial_recovery(working, broken)
                    return result.get('success', False)
                return False
            return False
        except Exception:
            return False

    def escalate(self, crisis: CrisisEvent) -> CrisisLevel:
        """Escalate crisis to next level."""
        escalation_map = {
            CrisisLevel.HICCUP: CrisisLevel.OBSTACLE,
            CrisisLevel.OBSTACLE: CrisisLevel.WALL,
            CrisisLevel.WALL: CrisisLevel.CRISIS,
            CrisisLevel.CRISIS: CrisisLevel.CRISIS,
        }
        return escalation_map[crisis.level]

    def record_outcome(self, action: RecoveryAction, success: bool, context: Optional[Dict[str, Any]] = None) -> None:
        """Record recovery outcome."""
        self._outcome_history.append({
            'action': action.name,
            'success': success,
            'timestamp': datetime.now(),
            'context': context or {},
        })

    def get_outcome_statistics(self) -> Dict[str, Any]:
        """Get outcome statistics."""
        if not self._outcome_history:
            return {}
        stats = {}
        for action in RecoveryAction:
            outcomes = [o for o in self._outcome_history if o['action'] == action.name]
            if outcomes:
                successes = sum(1 for o in outcomes if o['success'])
                stats[action.name] = {
                    'total': len(outcomes),
                    'successes': successes,
                    'failures': len(outcomes) - successes,
                    'success_rate': successes / len(outcomes),
                }
        return stats

    def _run_git(self, *args: str, **kwargs) -> subprocess.CompletedProcess:
        """
        Run a git command safely.

        Args:
            *args: Git command arguments
            **kwargs: Additional subprocess.run arguments

        Returns:
            CompletedProcess result

        Raises:
            RuntimeError: If git is not available
            subprocess.CalledProcessError: If git command fails
        """
        if not self._git_available:
            raise RuntimeError("Git is not available")

        cmd = ['git'] + list(args)
        defaults = {
            'capture_output': True,
            'text': True,
            'check': True,
        }
        defaults.update(kwargs)
        return subprocess.run(cmd, **defaults)

    def full_rollback(self, checkpoint: str, crisis_description: str = "") -> Dict[str, Any]:
        """
        Perform full rollback to a checkpoint.

        Steps:
        1. Stash current state with recovery label
        2. Checkout to the checkpoint commit
        3. Generate memory document about the failure
        4. Log the recovery action

        Args:
            checkpoint: Git commit hash or reference to restore to
            crisis_description: Description of what went wrong

        Returns:
            Dict with:
            - success: bool - Whether rollback succeeded
            - stash_ref: str - Reference to stashed changes
            - restored_to: str - Commit hash restored to
            - memory_file: str - Path to generated memory document
            - error: str - Error message if failed (optional)
        """
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        result = {
            'success': False,
            'stash_ref': '',
            'restored_to': checkpoint,
            'memory_file': '',
        }

        try:
            if not self._git_available:
                result['error'] = 'Git is not available'
                return result

            # Stash current state
            stash_label = f"recovery-{timestamp}"
            try:
                self._run_git('stash', 'save', stash_label)
                # Get the stash ref (should be stash@{0})
                stash_list = self._run_git('stash', 'list')
                stash_ref = 'stash@{0}'  # Most recent stash
                result['stash_ref'] = stash_ref
            except subprocess.CalledProcessError as e:
                # Might fail if there are no changes to stash
                result['stash_ref'] = 'none (no changes)'

            # Checkout to checkpoint
            self._run_git('checkout', checkpoint)

            # Generate memory document
            memory_file = self.create_recovery_memory(
                crisis=crisis_description or f"Rollback to {checkpoint}",
                actions=[
                    f"Stashed current state as: {result['stash_ref']}",
                    f"Restored to commit: {checkpoint}",
                ],
                lessons=[
                    "Full rollback was necessary",
                    "Previous approach was not viable",
                ],
            )
            result['memory_file'] = memory_file
            result['success'] = True

            # Log the recovery
            self._recovery_log.append({
                'type': 'full_rollback',
                'timestamp': timestamp,
                'checkpoint': checkpoint,
                'stash_ref': result['stash_ref'],
                'memory_file': memory_file,
            })

        except subprocess.CalledProcessError as e:
            result['error'] = f"Git command failed: {e.stderr}"
        except Exception as e:
            result['error'] = str(e)

        return result

    def partial_recovery(
        self,
        working_files: List[str],
        broken_files: List[str],
        commit_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Salvage working parts, preserve broken parts for analysis.

        Steps:
        1. Add working files to staging
        2. Commit working changes
        3. Stash broken files separately
        4. Generate follow-up task description

        Args:
            working_files: Files that are working and should be committed
            broken_files: Files that are broken and should be stashed
            commit_message: Custom commit message (optional)

        Returns:
            Dict with:
            - success: bool - Whether recovery succeeded
            - committed_files: list - Files successfully committed
            - commit_hash: str - Hash of the salvage commit
            - stashed_files: list - Files stashed for later analysis
            - stash_ref: str - Reference to stashed broken files
            - task_description: str - Follow-up task description
            - error: str - Error message if failed (optional)
        """
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        result = {
            'success': False,
            'committed_files': [],
            'commit_hash': '',
            'stashed_files': [],
            'stash_ref': '',
            'task_description': '',
        }

        try:
            if not self._git_available:
                result['error'] = 'Git is not available'
                return result

            # Add working files and commit
            if working_files:
                for file_path in working_files:
                    self._run_git('add', file_path)
                result['committed_files'] = working_files

                msg = commit_message or f"partial: salvaged working changes ({timestamp})"
                self._run_git('commit', '-m', msg)

                # Get the commit hash
                commit_hash_result = self._run_git('rev-parse', 'HEAD')
                result['commit_hash'] = commit_hash_result.stdout.strip()

            # Stash broken files
            if broken_files:
                for file_path in broken_files:
                    self._run_git('add', file_path)

                stash_label = f"broken-{timestamp}"
                self._run_git('stash', 'save', stash_label)
                result['stashed_files'] = broken_files
                result['stash_ref'] = 'stash@{0}'

            # Generate follow-up task description
            result['task_description'] = f"""## Follow-up: Fix Broken Files

**Working files committed:** {len(working_files)}
- {chr(10).join(f"  - {f}" for f in working_files)}

**Broken files stashed:** {len(broken_files)}
- {chr(10).join(f"  - {f}" for f in broken_files)}

**Stash reference:** {result['stash_ref']}

**Next steps:**
1. Analyze why broken files failed
2. Fix issues individually
3. Restore from stash: `git stash apply {result['stash_ref']}`
4. Re-test and commit fixed files
"""

            result['success'] = True

            # Log the recovery
            self._recovery_log.append({
                'type': 'partial_recovery',
                'timestamp': timestamp,
                'working_files': working_files,
                'broken_files': broken_files,
                'commit_hash': result['commit_hash'],
                'stash_ref': result['stash_ref'],
            })

        except subprocess.CalledProcessError as e:
            result['error'] = f"Git command failed: {e.stderr}"
        except Exception as e:
            result['error'] = str(e)

        return result

    def create_recovery_memory(
        self,
        crisis: str,
        actions: List[str],
        lessons: List[str]
    ) -> str:
        """
        Generate and save a recovery memory document.

        Args:
            crisis: Description of the crisis
            actions: List of actions taken during recovery
            lessons: List of lessons learned

        Returns:
            Path to the created memory document
        """
        date_str = datetime.now().strftime("%Y-%m-%d")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Create memory directory if it doesn't exist
        os.makedirs(self._memory_path, exist_ok=True)

        # Generate filename
        filename = f"[DRAFT]-recovery-{date_str}-{timestamp}.md"
        file_path = os.path.join(self._memory_path, filename)

        # Generate content
        content = f"""# Recovery Memory: {crisis}

**Date:** {date_str}
**Type:** Crisis Recovery
**Tags:** `recovery`, `crisis`, `lessons-learned`

## Crisis Description

{crisis}

## Recovery Actions Taken

{chr(10).join(f"{i+1}. {action}" for i, action in enumerate(actions))}

## Lessons Learned

{chr(10).join(f"- {lesson}" for lesson in lessons)}

## Preventive Measures

Consider the following to prevent similar crises:
- Review the warning signs that led to this crisis
- Update documentation or process guidelines
- Add automated checks if applicable
- Share learnings with the team

## Related

- [[crisis-management.md]]
- [[recovery-procedures.md]]

---
*This is a draft memory document. Review, edit, and remove [DRAFT] prefix when finalized.*
"""

        # Write the file
        with open(file_path, 'w') as f:
            f.write(content)

        return file_path

    def get_last_good_commit(self, max_commits: int = 50) -> str:
        """
        Find the most recent commit where tests likely passed.

        Heuristics:
        1. Look for commits with "test" or "fix" in message
        2. Avoid commits with "WIP", "broken", "failing"
        3. Look for merge commits (usually tested)
        4. Check for CI markers in commit messages

        Args:
            max_commits: Maximum number of commits to search

        Returns:
            Commit hash of the last likely good commit, or empty string if not found
        """
        if not self._git_available:
            return ""

        try:
            # Get recent commit log
            result = self._run_git('log', f'-{max_commits}', '--pretty=format:%H|%s')
            lines = result.stdout.strip().split('\n')

            # Score each commit
            for line in lines:
                if '|' not in line:
                    continue

                commit_hash, message = line.split('|', 1)
                message_lower = message.lower()

                # Bad indicators
                if any(bad in message_lower for bad in ['wip', 'broken', 'failing', 'todo', 'fixme']):
                    continue

                # Good indicators
                if any(good in message_lower for good in ['test', 'fix', 'merge', 'ci pass', 'coverage']):
                    return commit_hash

            # If no obviously good commit found, return the 5th most recent
            # (assuming recent commits might be problematic)
            if len(lines) >= 5:
                return lines[4].split('|')[0]

            # Fallback to HEAD~1
            result = self._run_git('rev-parse', 'HEAD~1')
            return result.stdout.strip()

        except (subprocess.CalledProcessError, IndexError):
            return ""

    def stash_current_state(self, label: str) -> str:
        """
        Stash the current working state with a label.

        Args:
            label: Label for the stash

        Returns:
            Stash reference (e.g., "stash@{0}"), or empty string if failed
        """
        if not self._git_available:
            return ""

        try:
            self._run_git('stash', 'save', label)
            return 'stash@{0}'  # Most recent stash
        except subprocess.CalledProcessError:
            return ""

    def list_stashes(self) -> List[Dict[str, str]]:
        """
        List all stashes in the repository.

        Returns:
            List of dicts with: {ref, message, date}
        """
        if not self._git_available:
            return []

        try:
            # Format: stash@{0}: On branch: message
            result = self._run_git('stash', 'list')
            stashes = []

            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue

                # Parse stash entry
                # Format: stash@{N}: WIP on branch: message
                match = re.match(r'(stash@\{(\d+)\}):\s*(.+)', line)
                if match:
                    stash_ref = match.group(1)
                    stash_num = match.group(2)
                    message = match.group(3)

                    stashes.append({
                        'ref': stash_ref,
                        'message': message,
                        'index': int(stash_num),
                    })

            return stashes

        except subprocess.CalledProcessError:
            return []

    def restore_from_stash(self, stash_ref: str) -> bool:
        """
        Restore (apply) a stash without dropping it.

        Args:
            stash_ref: Stash reference (e.g., "stash@{0}")

        Returns:
            True if successful, False otherwise
        """
        if not self._git_available:
            return False

        try:
            self._run_git('stash', 'apply', stash_ref)
            return True
        except subprocess.CalledProcessError:
            return False

    def generate_failure_analysis(self, event: CrisisEvent) -> str:
        """
        Generate a failure analysis document from a crisis event.

        Args:
            event: The crisis event to analyze

        Returns:
            Markdown-formatted analysis document
        """
        # Extract context
        context = event.context
        goal = context.get('goal', '[Unknown goal]')
        approach = context.get('approach', '[Unknown approach]')
        attempts = context.get('attempts', [])

        # Build analysis
        lines = [
            f"## Failed Approach Analysis: {event.description}",
            "",
            f"**Goal:** {goal}",
            "",
            f"**Approach:** {approach}",
            "",
            "**Why it failed:**",
        ]

        # Analyze attempts if available
        if attempts:
            lines.append("Multiple attempts were made:")
            for i, attempt in enumerate(attempts, 1):
                lines.append(f"{i}. {attempt}")
            lines.append("")
            lines.append("Pattern suggests a fundamental issue with the approach.")
        else:
            lines.append("[Analyze from event context]")

        lines.extend([
            "",
            "**What we learned:**",
        ])

        if event.lessons_learned:
            for lesson in event.lessons_learned:
                lines.append(f"- {lesson}")
        else:
            lines.append("- [Extract lessons from failure]")

        lines.extend([
            "",
            "**Red flags we should have noticed:**",
            "- Repeated failures with same approach",
            "- Escalating complexity without progress",
            "- Divergence from original scope",
            "",
            "**Conditions that would make this work:**",
            "- [Identify prerequisites for approach to succeed]",
            "- [List assumptions that need to be true]",
            "- [Suggest when to revisit this approach]",
        ])

        return "\n".join(lines)


class CrisisPredictor:
    """Crisis prediction using heuristic-based risk scoring."""

    def __init__(self, crisis_history: Optional[List[CrisisEvent]] = None):
        """Initialize crisis predictor."""
        self._history = crisis_history or []

    def analyze_patterns(self, events: List[CrisisEvent]) -> List[str]:
        """Find patterns in crisis history."""
        patterns = []
        if not events:
            return patterns
        from collections import Counter
        descriptions = [e.description.lower() for e in events]
        desc_counts = Counter(descriptions)
        for desc, count in desc_counts.items():
            if count >= 2:
                patterns.append(f"Repeated: {desc} ({count} times)")
        if len(events) >= 3:
            levels = [e.level.value for e in events[-3:]]
            if levels == sorted(levels):
                patterns.append("Escalating severity detected")
        unresolved = sum(1 for e in events if e.resolved_at is None)
        if unresolved >= 3:
            patterns.append(f"Accumulating unresolved crises ({unresolved})")
        return patterns

    def predict_risk(self, context: Dict[str, Any]) -> float:
        """Predict crisis risk level."""
        risk = 0.0
        repeated_failures = context.get('repeated_failures', 0)
        if repeated_failures >= 3:
            risk += 0.4
        elif repeated_failures == 2:
            risk += 0.2
        elif repeated_failures == 1:
            risk += 0.1
        if context.get('time_pressure', False):
            risk += 0.15
        complexity = context.get('complexity', 'low')
        if complexity == 'high':
            risk += 0.25
        elif complexity == 'medium':
            risk += 0.1
        scope_additions = context.get('scope_additions', 0)
        if scope_additions >= 3:
            risk += 0.2
        elif scope_additions >= 1:
            risk += 0.1
        if context.get('unexpected_files', 0) >= 3:
            risk += 0.15
        if context.get('new_concepts', 0) >= 2:
            risk += 0.15
        time_overrun = context.get('time_overrun_factor', 1.0)
        if time_overrun >= 2.0:
            risk += 0.2
        elif time_overrun >= 1.5:
            risk += 0.1
        active_blockers = context.get('active_blockers', 0)
        if active_blockers >= 2:
            risk += 0.2
        elif active_blockers == 1:
            risk += 0.1
        return min(risk, 1.0)

    def suggest_prevention(self, risk_factors: List[str]) -> List[str]:
        """Suggest preventive measures."""
        suggestions = []
        prevention_map = {
            'repeated_failures': ['Stop and analyze root cause', 'Consider alternative approach'],
            'time_pressure': ['Reduce scope to essentials', 'Time-box remaining work'],
            'high_complexity': ['Break into smaller steps', 'Prototype complex parts first'],
            'scope_creep': ['Return to original scope', 'Create follow-up tasks'],
            'unexpected_files': ['Review why files are needed', 'Document rationale'],
            'new_concepts': ['Allocate learning time', 'Find working examples'],
            'time_overrun': ['Re-estimate work', 'Consider partial delivery'],
            'active_blockers': ['Find workarounds', 'Escalate blocker resolution'],
        }
        for factor in risk_factors:
            if factor in prevention_map:
                suggestions.extend(prevention_map[factor])
        if len(risk_factors) >= 3:
            suggestions.append('Consider pausing for planning session')
        seen = set()
        unique = []
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        return unique

    def get_similar_past_crises(self, context: Dict[str, Any], max_results: int = 5) -> List[CrisisEvent]:
        """Find similar past crises."""
        if not self._history:
            return []
        scored = [(self._calculate_similarity(context, e.context), e) for e in self._history]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for score, e in scored[:max_results] if score > 0]

    def _calculate_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """Calculate similarity between contexts."""
        if not context1 or not context2:
            return 0.0
        keys1, keys2 = set(context1.keys()), set(context2.keys())
        common = keys1 & keys2
        if not common:
            return 0.0
        key_sim = len(common) / len(keys1 | keys2)
        value_matches = sum(1 for k in common if context1[k] == context2[k])
        value_sim = value_matches / len(common) if common else 0.0
        return 0.4 * key_sim + 0.6 * value_sim
