"""
Cognitive State: Externalized Thinking

This module provides the structures for externalizing my cognitive state—
the things I'm thinking about, questions I'm exploring, decisions I've made.

Why This Matters:
    I forget everything between sessions. Complex work spans sessions.
    This module lets me persist my thinking so I can resume where I left off.

Core Concepts:
    - Focus: What am I currently working on?
    - Questions: What don't I know yet?
    - Hypotheses: What am I considering?
    - Decisions: What have I committed to (and why)?
    - Observations: What have I noticed?
    - Checkpoints: Full snapshots for recovery

The cognitive state is my "working memory" externalized to disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Iterator
import hashlib


# =============================================================================
# ENUMS
# =============================================================================


class QuestionStatus(Enum):
    """Status of a question."""
    OPEN = auto()       # Still investigating
    ANSWERED = auto()   # Found an answer
    DEFERRED = auto()   # Postponed for later
    ABANDONED = auto()  # No longer relevant


class DecisionStatus(Enum):
    """Status of a decision."""
    TENTATIVE = auto()  # Made but could change
    COMMITTED = auto()  # Firm, proceeding with this
    SUPERSEDED = auto() # Replaced by another decision
    REVERTED = auto()   # Undone


class HypothesisStatus(Enum):
    """Status of a hypothesis."""
    EXPLORING = auto()  # Actively investigating
    SUPPORTED = auto()  # Evidence supports it
    REFUTED = auto()    # Evidence contradicts it
    ABANDONED = auto()  # Stopped investigating


# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================


@dataclass
class Question:
    """
    An open question I'm trying to answer.

    Questions drive investigation. They can:
    - Spawn sub-questions
    - Lead to hypotheses
    - Result in decisions
    """

    id: str
    text: str
    context: str = ""  # Why am I asking this?

    status: QuestionStatus = QuestionStatus.OPEN
    created_at: datetime = field(default_factory=datetime.now)
    answered_at: datetime | None = None

    answer: str | None = None
    sub_questions: list[str] = field(default_factory=list)  # Question IDs
    related_hypotheses: list[str] = field(default_factory=list)
    led_to_decision: str | None = None  # Decision ID

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "context": self.context,
            "status": self.status.name,
            "created_at": self.created_at.isoformat(),
            "answered_at": self.answered_at.isoformat() if self.answered_at else None,
            "answer": self.answer,
            "sub_questions": self.sub_questions,
            "related_hypotheses": self.related_hypotheses,
            "led_to_decision": self.led_to_decision,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Question:
        return cls(
            id=data["id"],
            text=data["text"],
            context=data.get("context", ""),
            status=QuestionStatus[data["status"]],
            created_at=datetime.fromisoformat(data["created_at"]),
            answered_at=datetime.fromisoformat(data["answered_at"]) if data.get("answered_at") else None,
            answer=data.get("answer"),
            sub_questions=data.get("sub_questions", []),
            related_hypotheses=data.get("related_hypotheses", []),
            led_to_decision=data.get("led_to_decision"),
        )


@dataclass
class Hypothesis:
    """
    A possible answer or approach I'm considering.

    Hypotheses are tested against evidence. They can be:
    - Supported (evidence for)
    - Refuted (evidence against)
    - Transformed into decisions
    """

    id: str
    statement: str
    rationale: str = ""  # Why do I think this might be true?

    status: HypothesisStatus = HypothesisStatus.EXPLORING
    confidence: float = 0.5  # 0-1, how confident am I?
    created_at: datetime = field(default_factory=datetime.now)

    supporting_evidence: list[str] = field(default_factory=list)
    contradicting_evidence: list[str] = field(default_factory=list)
    related_question: str | None = None  # Question this addresses
    became_decision: str | None = None  # If promoted to decision

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "statement": self.statement,
            "rationale": self.rationale,
            "status": self.status.name,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "supporting_evidence": self.supporting_evidence,
            "contradicting_evidence": self.contradicting_evidence,
            "related_question": self.related_question,
            "became_decision": self.became_decision,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Hypothesis:
        return cls(
            id=data["id"],
            statement=data["statement"],
            rationale=data.get("rationale", ""),
            status=HypothesisStatus[data["status"]],
            confidence=data.get("confidence", 0.5),
            created_at=datetime.fromisoformat(data["created_at"]),
            supporting_evidence=data.get("supporting_evidence", []),
            contradicting_evidence=data.get("contradicting_evidence", []),
            related_question=data.get("related_question"),
            became_decision=data.get("became_decision"),
        )


@dataclass
class Decision:
    """
    A choice I've made.

    Decisions are the outputs of reasoning. They include:
    - The decision itself
    - The rationale (why this choice)
    - Alternatives considered (what I didn't choose)
    - The context (what led to this)

    Rationale is crucial—it lets future-me understand
    why past-me made this choice.
    """

    id: str
    decision: str
    rationale: str

    status: DecisionStatus = DecisionStatus.TENTATIVE
    created_at: datetime = field(default_factory=datetime.now)
    committed_at: datetime | None = None

    alternatives: list[str] = field(default_factory=list)
    context: str = ""  # What situation led to this?
    from_hypothesis: str | None = None  # Which hypothesis became this?
    from_question: str | None = None  # Which question this answers
    superseded_by: str | None = None  # If replaced, what replaced it?

    # Tracking impact
    actions_taken: list[str] = field(default_factory=list)  # What I did because of this
    outcomes_observed: list[str] = field(default_factory=list)  # What happened

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "decision": self.decision,
            "rationale": self.rationale,
            "status": self.status.name,
            "created_at": self.created_at.isoformat(),
            "committed_at": self.committed_at.isoformat() if self.committed_at else None,
            "alternatives": self.alternatives,
            "context": self.context,
            "from_hypothesis": self.from_hypothesis,
            "from_question": self.from_question,
            "superseded_by": self.superseded_by,
            "actions_taken": self.actions_taken,
            "outcomes_observed": self.outcomes_observed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Decision:
        return cls(
            id=data["id"],
            decision=data["decision"],
            rationale=data["rationale"],
            status=DecisionStatus[data["status"]],
            created_at=datetime.fromisoformat(data["created_at"]),
            committed_at=datetime.fromisoformat(data["committed_at"]) if data.get("committed_at") else None,
            alternatives=data.get("alternatives", []),
            context=data.get("context", ""),
            from_hypothesis=data.get("from_hypothesis"),
            from_question=data.get("from_question"),
            superseded_by=data.get("superseded_by"),
            actions_taken=data.get("actions_taken", []),
            outcomes_observed=data.get("outcomes_observed", []),
        )


@dataclass
class Observation:
    """
    Something I noticed during execution.

    Observations are raw data—what I saw happen. They can:
    - Support or refute hypotheses
    - Lead to new questions
    - Inform future decisions
    """

    id: str
    observation: str
    context: str = ""  # What was I doing when I noticed this?

    created_at: datetime = field(default_factory=datetime.now)
    source: str = ""  # Where did this come from? (file, tool, user)

    related_action: str | None = None  # What action led to this?
    informs_hypothesis: list[str] = field(default_factory=list)
    led_to_question: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "observation": self.observation,
            "context": self.context,
            "created_at": self.created_at.isoformat(),
            "source": self.source,
            "related_action": self.related_action,
            "informs_hypothesis": self.informs_hypothesis,
            "led_to_question": self.led_to_question,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Observation:
        return cls(
            id=data["id"],
            observation=data["observation"],
            context=data.get("context", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
            source=data.get("source", ""),
            related_action=data.get("related_action"),
            informs_hypothesis=data.get("informs_hypothesis", []),
            led_to_question=data.get("led_to_question"),
        )


@dataclass
class Focus:
    """
    What I'm currently focused on.

    Focus provides context for all other cognitive activity.
    It answers: "What am I trying to do right now?"
    """

    description: str
    started_at: datetime = field(default_factory=datetime.now)
    goal_id: str | None = None  # Link to goal if applicable

    # What I'm tracking while focused
    active_questions: list[str] = field(default_factory=list)
    active_hypotheses: list[str] = field(default_factory=list)
    recent_decisions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "started_at": self.started_at.isoformat(),
            "goal_id": self.goal_id,
            "active_questions": self.active_questions,
            "active_hypotheses": self.active_hypotheses,
            "recent_decisions": self.recent_decisions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Focus:
        return cls(
            description=data["description"],
            started_at=datetime.fromisoformat(data["started_at"]),
            goal_id=data.get("goal_id"),
            active_questions=data.get("active_questions", []),
            active_hypotheses=data.get("active_hypotheses", []),
            recent_decisions=data.get("recent_decisions", []),
        )


# =============================================================================
# COGNITIVE STATE MANAGER
# =============================================================================


class CognitiveStateManager:
    """
    Manages my externalized cognitive state.

    This is my "working memory" persisted to disk. It tracks:
    - Current focus
    - Open questions
    - Active hypotheses
    - Decisions made
    - Observations recorded

    All state is persisted after each mutation, so if I crash,
    I can recover my last known state.
    """

    def __init__(self, state_dir: str | Path):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # In-memory state
        self.focus: Focus | None = None
        self.questions: dict[str, Question] = {}
        self.hypotheses: dict[str, Hypothesis] = {}
        self.decisions: dict[str, Decision] = {}
        self.observations: dict[str, Observation] = {}

        # ID generation counter
        self._id_counter = 0

        # Load existing state
        self._load_state()

    # =========================================================================
    # ID GENERATION
    # =========================================================================

    def _generate_id(self, prefix: str) -> str:
        """Generate a unique ID with prefix."""
        self._id_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        hash_part = hashlib.sha256(
            f"{timestamp}{self._id_counter}".encode()
        ).hexdigest()[:8]
        return f"{prefix}-{timestamp}-{hash_part}"

    # =========================================================================
    # FOCUS MANAGEMENT
    # =========================================================================

    def set_focus(self, description: str, goal_id: str | None = None) -> Focus:
        """
        Set what I'm currently focused on.

        This is the first thing to do when starting work—establish focus.
        """
        self.focus = Focus(
            description=description,
            goal_id=goal_id,
        )
        self._save_state()
        return self.focus

    def get_focus(self) -> Focus | None:
        """Get current focus."""
        return self.focus

    def clear_focus(self) -> None:
        """Clear current focus (when done with a task)."""
        self.focus = None
        self._save_state()

    # =========================================================================
    # QUESTION MANAGEMENT
    # =========================================================================

    def ask_question(
        self,
        text: str,
        context: str = "",
        parent_question_id: str | None = None,
    ) -> Question:
        """
        Record a question I'm trying to answer.

        Questions drive investigation. Every inquiry starts with a question.
        """
        question = Question(
            id=self._generate_id("Q"),
            text=text,
            context=context,
        )

        self.questions[question.id] = question

        # Link to parent question if this is a sub-question
        if parent_question_id and parent_question_id in self.questions:
            self.questions[parent_question_id].sub_questions.append(question.id)

        # Add to focus if we have one
        if self.focus:
            self.focus.active_questions.append(question.id)

        self._save_state()
        return question

    def answer_question(
        self,
        question_id: str,
        answer: str,
        led_to_decision_id: str | None = None,
    ) -> Question:
        """
        Mark a question as answered.

        The answer is recorded for future reference.
        """
        if question_id not in self.questions:
            raise ValueError(f"Unknown question: {question_id}")

        question = self.questions[question_id]
        question.status = QuestionStatus.ANSWERED
        question.answer = answer
        question.answered_at = datetime.now()
        question.led_to_decision = led_to_decision_id

        # Remove from active in focus
        if self.focus and question_id in self.focus.active_questions:
            self.focus.active_questions.remove(question_id)

        self._save_state()
        return question

    def defer_question(self, question_id: str, reason: str = "") -> Question:
        """Defer a question for later."""
        if question_id not in self.questions:
            raise ValueError(f"Unknown question: {question_id}")

        question = self.questions[question_id]
        question.status = QuestionStatus.DEFERRED
        if reason:
            question.context += f"\n[Deferred: {reason}]"

        self._save_state()
        return question

    def get_open_questions(self) -> list[Question]:
        """Get all open questions."""
        return [
            q for q in self.questions.values()
            if q.status == QuestionStatus.OPEN
        ]

    # =========================================================================
    # HYPOTHESIS MANAGEMENT
    # =========================================================================

    def form_hypothesis(
        self,
        statement: str,
        rationale: str = "",
        for_question_id: str | None = None,
    ) -> Hypothesis:
        """
        Form a hypothesis to investigate.

        Hypotheses are possible answers that need testing.
        """
        hypothesis = Hypothesis(
            id=self._generate_id("H"),
            statement=statement,
            rationale=rationale,
            related_question=for_question_id,
        )

        self.hypotheses[hypothesis.id] = hypothesis

        # Link to question
        if for_question_id and for_question_id in self.questions:
            self.questions[for_question_id].related_hypotheses.append(
                hypothesis.id
            )

        # Add to focus
        if self.focus:
            self.focus.active_hypotheses.append(hypothesis.id)

        self._save_state()
        return hypothesis

    def update_hypothesis_confidence(
        self,
        hypothesis_id: str,
        confidence: float,
        reason: str = "",
    ) -> Hypothesis:
        """
        Update confidence in a hypothesis.

        Confidence changes as evidence accumulates.
        """
        if hypothesis_id not in self.hypotheses:
            raise ValueError(f"Unknown hypothesis: {hypothesis_id}")

        hypothesis = self.hypotheses[hypothesis_id]
        old_confidence = hypothesis.confidence
        hypothesis.confidence = max(0.0, min(1.0, confidence))

        # Update status based on confidence
        if hypothesis.confidence >= 0.8:
            hypothesis.status = HypothesisStatus.SUPPORTED
        elif hypothesis.confidence <= 0.2:
            hypothesis.status = HypothesisStatus.REFUTED

        if reason:
            hypothesis.rationale += f"\n[Confidence {old_confidence:.2f} → {confidence:.2f}: {reason}]"

        self._save_state()
        return hypothesis

    def add_evidence(
        self,
        hypothesis_id: str,
        evidence: str,
        supports: bool,
    ) -> Hypothesis:
        """Add evidence for or against a hypothesis."""
        if hypothesis_id not in self.hypotheses:
            raise ValueError(f"Unknown hypothesis: {hypothesis_id}")

        hypothesis = self.hypotheses[hypothesis_id]

        if supports:
            hypothesis.supporting_evidence.append(evidence)
        else:
            hypothesis.contradicting_evidence.append(evidence)

        self._save_state()
        return hypothesis

    def get_active_hypotheses(self) -> list[Hypothesis]:
        """Get hypotheses still being explored."""
        return [
            h for h in self.hypotheses.values()
            if h.status == HypothesisStatus.EXPLORING
        ]

    # =========================================================================
    # DECISION MANAGEMENT
    # =========================================================================

    def make_decision(
        self,
        decision: str,
        rationale: str,
        alternatives: list[str] | None = None,
        context: str = "",
        from_hypothesis_id: str | None = None,
        from_question_id: str | None = None,
    ) -> Decision:
        """
        Record a decision.

        Decisions include rationale—why I made this choice.
        This is crucial for future understanding.
        """
        decision_obj = Decision(
            id=self._generate_id("D"),
            decision=decision,
            rationale=rationale,
            alternatives=alternatives or [],
            context=context,
            from_hypothesis=from_hypothesis_id,
            from_question=from_question_id,
        )

        self.decisions[decision_obj.id] = decision_obj

        # Link to hypothesis
        if from_hypothesis_id and from_hypothesis_id in self.hypotheses:
            self.hypotheses[from_hypothesis_id].became_decision = decision_obj.id
            self.hypotheses[from_hypothesis_id].status = HypothesisStatus.SUPPORTED

        # Link to question
        if from_question_id and from_question_id in self.questions:
            self.questions[from_question_id].led_to_decision = decision_obj.id

        # Add to focus
        if self.focus:
            self.focus.recent_decisions.append(decision_obj.id)

        self._save_state()
        return decision_obj

    def commit_decision(self, decision_id: str) -> Decision:
        """Commit to a decision (no longer tentative)."""
        if decision_id not in self.decisions:
            raise ValueError(f"Unknown decision: {decision_id}")

        decision = self.decisions[decision_id]
        decision.status = DecisionStatus.COMMITTED
        decision.committed_at = datetime.now()

        self._save_state()
        return decision

    def supersede_decision(
        self,
        old_decision_id: str,
        new_decision: str,
        rationale: str,
    ) -> Decision:
        """Replace an old decision with a new one."""
        if old_decision_id not in self.decisions:
            raise ValueError(f"Unknown decision: {old_decision_id}")

        # Create new decision
        new = self.make_decision(
            decision=new_decision,
            rationale=rationale,
            context=f"Supersedes {old_decision_id}",
        )

        # Update old decision
        old = self.decisions[old_decision_id]
        old.status = DecisionStatus.SUPERSEDED
        old.superseded_by = new.id

        self._save_state()
        return new

    def record_decision_outcome(
        self,
        decision_id: str,
        action: str | None = None,
        outcome: str | None = None,
    ) -> Decision:
        """Record what happened because of a decision."""
        if decision_id not in self.decisions:
            raise ValueError(f"Unknown decision: {decision_id}")

        decision = self.decisions[decision_id]

        if action:
            decision.actions_taken.append(action)
        if outcome:
            decision.outcomes_observed.append(outcome)

        self._save_state()
        return decision

    def get_recent_decisions(self, limit: int = 10) -> list[Decision]:
        """Get recent decisions."""
        sorted_decisions = sorted(
            self.decisions.values(),
            key=lambda d: d.created_at,
            reverse=True,
        )
        return sorted_decisions[:limit]

    # =========================================================================
    # OBSERVATION MANAGEMENT
    # =========================================================================

    def record_observation(
        self,
        observation: str,
        context: str = "",
        source: str = "",
        related_action: str | None = None,
    ) -> Observation:
        """
        Record something I observed.

        Observations are raw data that can inform hypotheses.
        """
        obs = Observation(
            id=self._generate_id("O"),
            observation=observation,
            context=context,
            source=source,
            related_action=related_action,
        )

        self.observations[obs.id] = obs
        self._save_state()
        return obs

    # =========================================================================
    # CHECKPOINTING
    # =========================================================================

    def checkpoint(self) -> dict[str, Any]:
        """
        Create a full checkpoint of cognitive state.

        Checkpoints are used for recovery when I get confused.
        """
        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "focus": self.focus.to_dict() if self.focus else None,
            "questions": {qid: q.to_dict() for qid, q in self.questions.items()},
            "hypotheses": {hid: h.to_dict() for hid, h in self.hypotheses.items()},
            "decisions": {did: d.to_dict() for did, d in self.decisions.items()},
            "observations": {oid: o.to_dict() for oid, o in self.observations.items()},
            "id_counter": self._id_counter,
        }

        # Save checkpoint
        checkpoint_dir = self.state_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint_file = checkpoint_dir / f"checkpoint-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint, f, indent=2)

        return checkpoint

    def restore_from_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Restore state from a checkpoint."""
        self.focus = Focus.from_dict(checkpoint["focus"]) if checkpoint.get("focus") else None
        self.questions = {
            qid: Question.from_dict(q)
            for qid, q in checkpoint.get("questions", {}).items()
        }
        self.hypotheses = {
            hid: Hypothesis.from_dict(h)
            for hid, h in checkpoint.get("hypotheses", {}).items()
        }
        self.decisions = {
            did: Decision.from_dict(d)
            for did, d in checkpoint.get("decisions", {}).items()
        }
        self.observations = {
            oid: Observation.from_dict(o)
            for oid, o in checkpoint.get("observations", {}).items()
        }
        self._id_counter = checkpoint.get("id_counter", 0)

        self._save_state()

    def list_checkpoints(self) -> list[Path]:
        """List available checkpoints."""
        checkpoint_dir = self.state_dir / "checkpoints"
        if not checkpoint_dir.exists():
            return []

        return sorted(
            checkpoint_dir.glob("checkpoint-*.json"),
            reverse=True,
        )

    def load_checkpoint(self, checkpoint_path: Path) -> dict[str, Any]:
        """Load a specific checkpoint."""
        with open(checkpoint_path) as f:
            return json.load(f)

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def _save_state(self) -> None:
        """Save current state to disk."""
        state = {
            "focus": self.focus.to_dict() if self.focus else None,
            "questions": {qid: q.to_dict() for qid, q in self.questions.items()},
            "hypotheses": {hid: h.to_dict() for hid, h in self.hypotheses.items()},
            "decisions": {did: d.to_dict() for did, d in self.decisions.items()},
            "observations": {oid: o.to_dict() for oid, o in self.observations.items()},
            "id_counter": self._id_counter,
        }

        state_file = self.state_dir / "current_state.json"
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self) -> None:
        """Load state from disk."""
        state_file = self.state_dir / "current_state.json"
        if not state_file.exists():
            return

        try:
            with open(state_file) as f:
                state = json.load(f)

            self.focus = Focus.from_dict(state["focus"]) if state.get("focus") else None
            self.questions = {
                qid: Question.from_dict(q)
                for qid, q in state.get("questions", {}).items()
            }
            self.hypotheses = {
                hid: Hypothesis.from_dict(h)
                for hid, h in state.get("hypotheses", {}).items()
            }
            self.decisions = {
                did: Decision.from_dict(d)
                for did, d in state.get("decisions", {}).items()
            }
            self.observations = {
                oid: Observation.from_dict(o)
                for oid, o in state.get("observations", {}).items()
            }
            self._id_counter = state.get("id_counter", 0)

        except (json.JSONDecodeError, KeyError) as e:
            # Corrupted state file—start fresh but log
            print(f"Warning: Could not load state: {e}")

    # =========================================================================
    # SUMMARY AND CONTEXT
    # =========================================================================

    def get_summary(self) -> str:
        """
        Get a summary of current cognitive state.

        Useful for context when resuming work.
        """
        lines = []

        if self.focus:
            lines.append(f"## Current Focus")
            lines.append(f"{self.focus.description}")
            lines.append("")

        open_questions = self.get_open_questions()
        if open_questions:
            lines.append(f"## Open Questions ({len(open_questions)})")
            for q in open_questions[:5]:
                lines.append(f"- {q.text}")
            if len(open_questions) > 5:
                lines.append(f"  ... and {len(open_questions) - 5} more")
            lines.append("")

        active_hypotheses = self.get_active_hypotheses()
        if active_hypotheses:
            lines.append(f"## Active Hypotheses ({len(active_hypotheses)})")
            for h in active_hypotheses[:3]:
                lines.append(f"- {h.statement} (confidence: {h.confidence:.0%})")
            lines.append("")

        recent_decisions = self.get_recent_decisions(5)
        if recent_decisions:
            lines.append(f"## Recent Decisions")
            for d in recent_decisions:
                status = "✓" if d.status == DecisionStatus.COMMITTED else "?"
                lines.append(f"- [{status}] {d.decision}")
            lines.append("")

        return "\n".join(lines)

    def get_context_for_question(self, question_id: str) -> dict[str, Any]:
        """Get all context related to a question."""
        if question_id not in self.questions:
            return {}

        question = self.questions[question_id]

        return {
            "question": question.to_dict(),
            "sub_questions": [
                self.questions[qid].to_dict()
                for qid in question.sub_questions
                if qid in self.questions
            ],
            "hypotheses": [
                self.hypotheses[hid].to_dict()
                for hid in question.related_hypotheses
                if hid in self.hypotheses
            ],
            "decision": (
                self.decisions[question.led_to_decision].to_dict()
                if question.led_to_decision and question.led_to_decision in self.decisions
                else None
            ),
        }
