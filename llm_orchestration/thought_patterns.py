"""
Thought Patterns: Structured Approaches to Reasoning

This module provides structured patterns for thinking through problems.
These patterns are not just descriptions—they're executable guides
that structure my cognitive process.

Core Patterns:
    - QAPV: Question → Answer → Produce → Verify
    - Hypothesis Testing: Form → Test → Refine/Reject
    - Decision Matrix: Options → Criteria → Score → Decide
    - Root Cause Analysis: Symptom → Causes → Test → Confirm
    - Exploration: Breadth-first → Depth on promising → Synthesize

Each pattern creates nodes in the Graph of Thought and manages
transitions through cognitive states.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Iterator

from .cognitive_state import (
    CognitiveStateManager,
    Decision,
    Hypothesis,
    Question,
)


# =============================================================================
# PATTERN PHASES
# =============================================================================


class QAPVPhase(Enum):
    """Phases of the QAPV cycle."""
    QUESTION = auto()   # What am I trying to do?
    ANSWER = auto()     # Research, explore, decide
    PRODUCE = auto()    # Create artifact
    VERIFY = auto()     # Check quality
    COMPLETE = auto()   # Done


class HypothesisPhase(Enum):
    """Phases of hypothesis testing."""
    FORM = auto()       # State the hypothesis
    DESIGN = auto()     # Design the test
    EXECUTE = auto()    # Run the test
    EVALUATE = auto()   # Assess results
    CONCLUDE = auto()   # Accept, reject, or refine


class ExplorationPhase(Enum):
    """Phases of exploration."""
    SURVEY = auto()     # Breadth-first scan
    IDENTIFY = auto()   # Find promising areas
    DIVE = auto()       # Depth-first on selected
    SYNTHESIZE = auto() # Combine findings


# =============================================================================
# BASE PATTERN
# =============================================================================


@dataclass
class PatternStep:
    """A single step in a thought pattern."""
    name: str
    description: str
    required_inputs: list[str]
    expected_outputs: list[str]
    validation: Callable[[dict[str, Any]], bool] | None = None


class ThoughtPattern(ABC):
    """
    Base class for structured thinking patterns.

    Patterns provide:
    1. A sequence of phases/steps
    2. Validation at each transition
    3. Integration with cognitive state
    4. Recovery if stuck
    """

    def __init__(self, cognitive_state: CognitiveStateManager):
        self.state = cognitive_state
        self.current_phase: Enum | None = None
        self.context: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []

    @abstractmethod
    def get_phases(self) -> list[Enum]:
        """Get the ordered phases of this pattern."""
        ...

    @abstractmethod
    def get_current_guidance(self) -> str:
        """Get guidance for current phase."""
        ...

    @abstractmethod
    def can_advance(self) -> tuple[bool, str]:
        """Check if ready to advance. Returns (can_advance, reason)."""
        ...

    @abstractmethod
    def advance(self) -> Enum:
        """Advance to next phase."""
        ...

    def record_step(self, action: str, result: Any) -> None:
        """Record an action taken in the current phase."""
        self.history.append({
            "phase": self.current_phase.name if self.current_phase else None,
            "action": action,
            "result": result,
            "timestamp": datetime.now().isoformat(),
        })

    def get_progress(self) -> dict[str, Any]:
        """Get pattern progress summary."""
        phases = self.get_phases()
        current_index = (
            phases.index(self.current_phase)
            if self.current_phase in phases
            else -1
        )

        return {
            "pattern": self.__class__.__name__,
            "current_phase": self.current_phase.name if self.current_phase else None,
            "progress": f"{current_index + 1}/{len(phases)}",
            "steps_taken": len(self.history),
        }


# =============================================================================
# QAPV PATTERN
# =============================================================================


class QAPVPattern(ThoughtPattern):
    """
    Question → Answer → Produce → Verify

    The core reasoning loop. Use this for:
    - Implementing features
    - Fixing bugs
    - Answering complex questions
    - Any task that produces an artifact

    Flow:
        QUESTION: What am I trying to do? What's the success criteria?
        ANSWER: Research, explore options, make decisions
        PRODUCE: Create the artifact (code, doc, etc.)
        VERIFY: Test, review, validate quality
        (if verify fails, loop back to QUESTION with new understanding)
    """

    def __init__(
        self,
        cognitive_state: CognitiveStateManager,
        goal: str,
    ):
        super().__init__(cognitive_state)
        self.goal = goal
        self.current_phase = QAPVPhase.QUESTION

        # Phase-specific state
        self.question: Question | None = None
        self.decisions: list[Decision] = []
        self.artifact: Any = None
        self.verification_result: dict[str, Any] | None = None

        # Set focus
        self.state.set_focus(f"QAPV: {goal}")

    def get_phases(self) -> list[Enum]:
        return list(QAPVPhase)

    def get_current_guidance(self) -> str:
        """Get guidance for what to do in current phase."""
        guidance = {
            QAPVPhase.QUESTION: """
## QUESTION Phase

Your task: Clearly define what you're trying to accomplish.

Actions:
1. State the core question/goal
2. Identify what "success" looks like
3. List what you need to know before proceeding
4. Record any assumptions

When done, you should have:
- A clear question recorded
- Success criteria defined
- Open sub-questions identified

Advance when: You can clearly state what you're doing and how you'll know it's done.
""",
            QAPVPhase.ANSWER: """
## ANSWER Phase

Your task: Research, explore, and make decisions.

Actions:
1. Answer the questions from QUESTION phase
2. Explore options/approaches
3. Make and record decisions (with rationale!)
4. Resolve blocking questions

When done, you should have:
- All blocking questions answered
- Key decisions made with rationale
- A clear plan for PRODUCE phase

Advance when: You have enough information to start producing.
""",
            QAPVPhase.PRODUCE: """
## PRODUCE Phase

Your task: Create the artifact (code, document, etc.)

Actions:
1. Execute your plan from ANSWER phase
2. Create the artifact incrementally
3. Record observations as you work
4. Note any surprises or deviations

When done, you should have:
- The artifact created
- Notes on what was actually done
- Any new questions discovered

Advance when: The artifact exists and is ready for verification.
""",
            QAPVPhase.VERIFY: """
## VERIFY Phase

Your task: Validate the artifact meets requirements.

Actions:
1. Test against success criteria from QUESTION phase
2. Run any automated tests
3. Review for quality
4. Document verification results

When done, you should have:
- Verification results recorded
- Pass/fail determination
- If failed: understanding of what needs to change

Advance when: Verification passes, OR you understand what to fix.

If verification fails, you'll return to QUESTION with new understanding.
""",
            QAPVPhase.COMPLETE: """
## COMPLETE

The QAPV cycle is complete.

The artifact has been verified and meets the success criteria.

Consider:
- What did you learn?
- What would you do differently?
- Any follow-up tasks?
""",
        }

        return guidance.get(self.current_phase, "Unknown phase")

    def can_advance(self) -> tuple[bool, str]:
        """Check if ready to advance to next phase."""
        if self.current_phase == QAPVPhase.QUESTION:
            if not self.question:
                return False, "Need to record the core question first"
            return True, "Question phase complete"

        elif self.current_phase == QAPVPhase.ANSWER:
            open_questions = self.state.get_open_questions()
            blocking = [q for q in open_questions if "blocking" in q.context.lower()]
            if blocking:
                return False, f"Still have {len(blocking)} blocking questions"
            if not self.decisions:
                return False, "Need at least one decision to proceed"
            return True, "Answer phase complete"

        elif self.current_phase == QAPVPhase.PRODUCE:
            if self.artifact is None:
                return False, "Need to produce the artifact"
            return True, "Artifact produced"

        elif self.current_phase == QAPVPhase.VERIFY:
            if self.verification_result is None:
                return False, "Need to run verification"
            return True, "Verification complete"

        return False, "Cannot advance from current phase"

    def advance(self) -> QAPVPhase:
        """Advance to next phase."""
        can, reason = self.can_advance()
        if not can:
            raise ValueError(f"Cannot advance: {reason}")

        phases = list(QAPVPhase)
        current_index = phases.index(self.current_phase)

        # Special handling for VERIFY
        if self.current_phase == QAPVPhase.VERIFY:
            if self.verification_result and self.verification_result.get("passed"):
                self.current_phase = QAPVPhase.COMPLETE
            else:
                # Loop back to QUESTION with new understanding
                self.current_phase = QAPVPhase.QUESTION
                self.record_step("verification_failed", self.verification_result)

        elif current_index < len(phases) - 1:
            self.current_phase = phases[current_index + 1]

        return self.current_phase

    # =========================================================================
    # PHASE-SPECIFIC ACTIONS
    # =========================================================================

    def set_question(self, question_text: str, success_criteria: str) -> Question:
        """Set the core question for this QAPV cycle."""
        self.question = self.state.ask_question(
            text=question_text,
            context=f"Success criteria: {success_criteria}",
        )
        self.record_step("set_question", question_text)
        return self.question

    def record_decision(
        self,
        decision: str,
        rationale: str,
        alternatives: list[str] | None = None,
    ) -> Decision:
        """Record a decision made during ANSWER phase."""
        d = self.state.make_decision(
            decision=decision,
            rationale=rationale,
            alternatives=alternatives,
            from_question_id=self.question.id if self.question else None,
        )
        self.decisions.append(d)
        self.record_step("decision", decision)
        return d

    def set_artifact(self, artifact: Any, description: str) -> None:
        """Set the artifact produced during PRODUCE phase."""
        self.artifact = artifact
        self.record_step("artifact_produced", description)

    def record_verification(
        self,
        passed: bool,
        details: dict[str, Any],
    ) -> dict[str, Any]:
        """Record verification results."""
        self.verification_result = {
            "passed": passed,
            "details": details,
            "timestamp": datetime.now().isoformat(),
        }
        self.record_step("verification", self.verification_result)
        return self.verification_result


# =============================================================================
# HYPOTHESIS TESTING PATTERN
# =============================================================================


class HypothesisTestingPattern(ThoughtPattern):
    """
    Form → Design → Execute → Evaluate → Conclude

    Use this for:
    - Debugging (hypothesis about cause)
    - Exploring approaches (hypothesis about best option)
    - Validating assumptions

    Flow:
        FORM: State the hypothesis clearly
        DESIGN: Design a test that would prove/disprove it
        EXECUTE: Run the test
        EVALUATE: Assess results against hypothesis
        CONCLUDE: Accept, reject, or refine the hypothesis
    """

    def __init__(
        self,
        cognitive_state: CognitiveStateManager,
        initial_observation: str,
    ):
        super().__init__(cognitive_state)
        self.initial_observation = initial_observation
        self.current_phase = HypothesisPhase.FORM

        self.hypothesis: Hypothesis | None = None
        self.test_design: str | None = None
        self.test_results: dict[str, Any] | None = None
        self.conclusion: str | None = None

    def get_phases(self) -> list[Enum]:
        return list(HypothesisPhase)

    def get_current_guidance(self) -> str:
        guidance = {
            HypothesisPhase.FORM: f"""
## FORM Phase

Observation: {self.initial_observation}

Your task: Form a testable hypothesis.

A good hypothesis:
- Is specific and falsifiable
- Explains the observation
- Suggests what you'd expect to see if true

Template: "If [hypothesis], then when I [test], I should see [expected result]"

Actions:
1. State the hypothesis clearly
2. Identify what evidence would support it
3. Identify what evidence would refute it

Advance when: Hypothesis is formed and recorded.
""",
            HypothesisPhase.DESIGN: """
## DESIGN Phase

Your task: Design a test for the hypothesis.

A good test:
- Has a clear expected outcome if hypothesis is true
- Has a clear expected outcome if hypothesis is false
- Is practical to execute
- Controls for confounding factors

Actions:
1. Describe the test procedure
2. Define expected results for true/false cases
3. Identify any prerequisites

Advance when: Test is designed and documented.
""",
            HypothesisPhase.EXECUTE: """
## EXECUTE Phase

Your task: Run the test.

Actions:
1. Execute the test procedure
2. Record actual results
3. Note any unexpected observations
4. Document any deviations from plan

Advance when: Test is complete and results recorded.
""",
            HypothesisPhase.EVALUATE: """
## EVALUATE Phase

Your task: Compare results to expectations.

Actions:
1. Compare actual results to expected (if true)
2. Compare actual results to expected (if false)
3. Assess which matches better
4. Consider alternative explanations

Questions to answer:
- Do results support the hypothesis?
- Do results refute the hypothesis?
- Are results inconclusive?
- Did anything unexpected happen?

Advance when: Evaluation is complete.
""",
            HypothesisPhase.CONCLUDE: """
## CONCLUDE Phase

Your task: Draw conclusions and decide next steps.

Options:
- ACCEPT: Evidence supports hypothesis, proceed as if true
- REJECT: Evidence refutes hypothesis, consider alternatives
- REFINE: Partially supported, form refined hypothesis

Actions:
1. State your conclusion
2. Update hypothesis confidence
3. Decide next action
4. Document rationale
""",
        }

        return guidance.get(self.current_phase, "Unknown phase")

    def can_advance(self) -> tuple[bool, str]:
        if self.current_phase == HypothesisPhase.FORM:
            if not self.hypothesis:
                return False, "Need to form hypothesis first"
            return True, "Hypothesis formed"

        elif self.current_phase == HypothesisPhase.DESIGN:
            if not self.test_design:
                return False, "Need to design test"
            return True, "Test designed"

        elif self.current_phase == HypothesisPhase.EXECUTE:
            if not self.test_results:
                return False, "Need to record test results"
            return True, "Test executed"

        elif self.current_phase == HypothesisPhase.EVALUATE:
            # Can always advance after execution
            return True, "Ready to conclude"

        return False, "Cannot advance"

    def advance(self) -> HypothesisPhase:
        can, reason = self.can_advance()
        if not can:
            raise ValueError(f"Cannot advance: {reason}")

        phases = list(HypothesisPhase)
        current_index = phases.index(self.current_phase)

        if current_index < len(phases) - 1:
            self.current_phase = phases[current_index + 1]

        return self.current_phase

    def form_hypothesis(self, statement: str, rationale: str) -> Hypothesis:
        """Form the hypothesis."""
        self.hypothesis = self.state.form_hypothesis(
            statement=statement,
            rationale=rationale,
        )
        self.record_step("formed_hypothesis", statement)
        return self.hypothesis

    def design_test(self, test_procedure: str, expected_if_true: str, expected_if_false: str) -> None:
        """Design the test."""
        self.test_design = {
            "procedure": test_procedure,
            "expected_if_true": expected_if_true,
            "expected_if_false": expected_if_false,
        }
        self.record_step("designed_test", test_procedure)

    def record_results(self, results: dict[str, Any]) -> None:
        """Record test results."""
        self.test_results = results
        self.record_step("test_results", results)

    def conclude(
        self,
        conclusion: str,
        new_confidence: float,
        next_action: str,
    ) -> None:
        """Record conclusion and update hypothesis."""
        self.conclusion = conclusion

        if self.hypothesis:
            self.state.update_hypothesis_confidence(
                self.hypothesis.id,
                new_confidence,
                reason=conclusion,
            )

        self.record_step("conclusion", {
            "conclusion": conclusion,
            "confidence": new_confidence,
            "next_action": next_action,
        })


# =============================================================================
# DECISION MATRIX PATTERN
# =============================================================================


@dataclass
class Option:
    """An option in a decision."""
    name: str
    description: str
    scores: dict[str, float] = field(default_factory=dict)  # criterion → score


@dataclass
class Criterion:
    """A criterion for evaluating options."""
    name: str
    weight: float = 1.0
    description: str = ""


class DecisionMatrixPattern(ThoughtPattern):
    """
    Options → Criteria → Score → Decide

    Use this for:
    - Choosing between approaches
    - Evaluating trade-offs
    - Making transparent, justified decisions

    Flow:
        1. List all viable options
        2. Define evaluation criteria with weights
        3. Score each option on each criterion
        4. Calculate weighted totals
        5. Make and document decision
    """

    def __init__(
        self,
        cognitive_state: CognitiveStateManager,
        decision_question: str,
    ):
        super().__init__(cognitive_state)
        self.decision_question = decision_question

        self.options: list[Option] = []
        self.criteria: list[Criterion] = []
        self.final_decision: Decision | None = None

    def get_phases(self) -> list[Enum]:
        # Simple progression
        return []

    def get_current_guidance(self) -> str:
        return f"""
## Decision Matrix: {self.decision_question}

### Options ({len(self.options)})
{self._format_options()}

### Criteria ({len(self.criteria)})
{self._format_criteria()}

### Matrix
{self._format_matrix()}

### Recommendation
{self._get_recommendation()}
"""

    def _format_options(self) -> str:
        if not self.options:
            return "(no options added yet)"
        return "\n".join(f"- {o.name}: {o.description}" for o in self.options)

    def _format_criteria(self) -> str:
        if not self.criteria:
            return "(no criteria added yet)"
        return "\n".join(
            f"- {c.name} (weight: {c.weight}): {c.description}"
            for c in self.criteria
        )

    def _format_matrix(self) -> str:
        if not self.options or not self.criteria:
            return "(incomplete matrix)"

        # Header
        header = "| Option | " + " | ".join(c.name for c in self.criteria) + " | Total |"
        separator = "|" + "|".join("-" * 10 for _ in range(len(self.criteria) + 2)) + "|"

        rows = []
        for option in self.options:
            scores = [str(option.scores.get(c.name, "-")) for c in self.criteria]
            total = self._calculate_total(option)
            rows.append(f"| {option.name} | " + " | ".join(scores) + f" | {total:.2f} |")

        return "\n".join([header, separator] + rows)

    def _calculate_total(self, option: Option) -> float:
        total = 0.0
        for criterion in self.criteria:
            score = option.scores.get(criterion.name, 0)
            total += score * criterion.weight
        return total

    def _get_recommendation(self) -> str:
        if not self.options or not self.criteria:
            return "(need options and criteria to recommend)"

        # Check if all options are scored
        for option in self.options:
            for criterion in self.criteria:
                if criterion.name not in option.scores:
                    return f"(need to score {option.name} on {criterion.name})"

        # Find winner
        scored = [(o, self._calculate_total(o)) for o in self.options]
        scored.sort(key=lambda x: x[1], reverse=True)

        winner, score = scored[0]
        runner_up = scored[1] if len(scored) > 1 else None

        result = f"**Recommended: {winner.name}** (score: {score:.2f})"
        if runner_up:
            result += f"\nRunner-up: {runner_up[0].name} (score: {runner_up[1]:.2f})"

        return result

    def can_advance(self) -> tuple[bool, str]:
        if len(self.options) < 2:
            return False, "Need at least 2 options"
        if not self.criteria:
            return False, "Need at least 1 criterion"

        # Check all scored
        for option in self.options:
            for criterion in self.criteria:
                if criterion.name not in option.scores:
                    return False, f"Need to score {option.name} on {criterion.name}"

        return True, "Ready to decide"

    def advance(self) -> None:
        """Make the decision."""
        can, reason = self.can_advance()
        if not can:
            raise ValueError(f"Cannot decide: {reason}")

        scored = [(o, self._calculate_total(o)) for o in self.options]
        scored.sort(key=lambda x: x[1], reverse=True)
        winner = scored[0][0]

        # Create decision
        self.final_decision = self.state.make_decision(
            decision=f"Choose {winner.name}",
            rationale=self._get_recommendation(),
            alternatives=[o.name for o in self.options if o != winner],
            context=self.decision_question,
        )

    # =========================================================================
    # BUILDER METHODS
    # =========================================================================

    def add_option(self, name: str, description: str) -> Option:
        """Add an option to consider."""
        option = Option(name=name, description=description)
        self.options.append(option)
        return option

    def add_criterion(
        self,
        name: str,
        weight: float = 1.0,
        description: str = "",
    ) -> Criterion:
        """Add an evaluation criterion."""
        criterion = Criterion(name=name, weight=weight, description=description)
        self.criteria.append(criterion)
        return criterion

    def score(self, option_name: str, criterion_name: str, score: float) -> None:
        """Score an option on a criterion (0-10 scale recommended)."""
        option = next((o for o in self.options if o.name == option_name), None)
        if not option:
            raise ValueError(f"Unknown option: {option_name}")

        criterion = next((c for c in self.criteria if c.name == criterion_name), None)
        if not criterion:
            raise ValueError(f"Unknown criterion: {criterion_name}")

        option.scores[criterion_name] = score


# =============================================================================
# EXPLORATION PATTERN
# =============================================================================


class ExplorationPattern(ThoughtPattern):
    """
    Survey → Identify → Dive → Synthesize

    Use this for:
    - Understanding a new codebase
    - Researching a topic
    - Finding relevant information

    Flow:
        SURVEY: Breadth-first scan of the space
        IDENTIFY: Find promising areas to explore deeper
        DIVE: Depth-first exploration of selected areas
        SYNTHESIZE: Combine findings into understanding
    """

    def __init__(
        self,
        cognitive_state: CognitiveStateManager,
        exploration_goal: str,
    ):
        super().__init__(cognitive_state)
        self.exploration_goal = exploration_goal
        self.current_phase = ExplorationPhase.SURVEY

        self.survey_findings: list[str] = []
        self.promising_areas: list[str] = []
        self.deep_findings: dict[str, list[str]] = {}
        self.synthesis: str | None = None

    def get_phases(self) -> list[Enum]:
        return list(ExplorationPhase)

    def get_current_guidance(self) -> str:
        guidance = {
            ExplorationPhase.SURVEY: f"""
## SURVEY Phase

Goal: {self.exploration_goal}

Your task: Broad scan of the space.

Actions:
1. List major areas/components/topics
2. Note what exists without going deep
3. Identify potential relevance to goal
4. Record initial impressions

Tips:
- Don't go deep yet—just survey
- Note questions that arise
- Mark what looks promising

Advance when: You have a map of the territory.
""",
            ExplorationPhase.IDENTIFY: """
## IDENTIFY Phase

Your task: Choose where to dive deep.

Actions:
1. Review survey findings
2. Rank by relevance to goal
3. Select 2-3 most promising areas
4. Note why each is promising

Criteria for selection:
- Relevance to goal
- Likely information density
- Accessibility

Advance when: Promising areas are selected.
""",
            ExplorationPhase.DIVE: f"""
## DIVE Phase

Selected areas: {', '.join(self.promising_areas) or '(none selected)'}

Your task: Deep exploration of each selected area.

For each area:
1. Read/explore thoroughly
2. Extract key information
3. Note relationships to other areas
4. Record specific findings

Tips:
- Go deep, not broad
- Follow interesting threads
- Record as you go

Advance when: Each selected area has been explored.
""",
            ExplorationPhase.SYNTHESIZE: """
## SYNTHESIZE Phase

Your task: Combine findings into understanding.

Actions:
1. Review all findings
2. Identify patterns and themes
3. Answer the original exploration goal
4. Note what remains unknown

Output:
- Summary addressing the goal
- Key insights
- Remaining questions

Advance when: Synthesis is complete.
""",
        }

        return guidance.get(self.current_phase, "Unknown phase")

    def can_advance(self) -> tuple[bool, str]:
        if self.current_phase == ExplorationPhase.SURVEY:
            if not self.survey_findings:
                return False, "Need survey findings"
            return True, "Survey complete"

        elif self.current_phase == ExplorationPhase.IDENTIFY:
            if not self.promising_areas:
                return False, "Need to identify promising areas"
            return True, "Areas identified"

        elif self.current_phase == ExplorationPhase.DIVE:
            for area in self.promising_areas:
                if area not in self.deep_findings:
                    return False, f"Need to dive into {area}"
            return True, "All areas explored"

        elif self.current_phase == ExplorationPhase.SYNTHESIZE:
            if not self.synthesis:
                return False, "Need to synthesize findings"
            return True, "Synthesis complete"

        return False, "Cannot advance"

    def advance(self) -> ExplorationPhase:
        can, reason = self.can_advance()
        if not can:
            raise ValueError(f"Cannot advance: {reason}")

        phases = list(ExplorationPhase)
        current_index = phases.index(self.current_phase)

        if current_index < len(phases) - 1:
            self.current_phase = phases[current_index + 1]

        return self.current_phase

    # =========================================================================
    # PHASE ACTIONS
    # =========================================================================

    def add_survey_finding(self, finding: str) -> None:
        """Add a finding from the survey phase."""
        self.survey_findings.append(finding)
        self.record_step("survey_finding", finding)

    def mark_promising(self, area: str, reason: str) -> None:
        """Mark an area as promising for deep exploration."""
        self.promising_areas.append(area)
        self.record_step("marked_promising", {"area": area, "reason": reason})

    def add_deep_finding(self, area: str, finding: str) -> None:
        """Add a finding from deep exploration of an area."""
        if area not in self.deep_findings:
            self.deep_findings[area] = []
        self.deep_findings[area].append(finding)
        self.record_step("deep_finding", {"area": area, "finding": finding})

    def set_synthesis(self, synthesis: str) -> None:
        """Set the final synthesis."""
        self.synthesis = synthesis
        self.record_step("synthesis", synthesis)


# =============================================================================
# PATTERN FACTORY
# =============================================================================


def create_pattern(
    pattern_type: str,
    cognitive_state: CognitiveStateManager,
    **kwargs: Any,
) -> ThoughtPattern:
    """
    Factory for creating thought patterns.

    Args:
        pattern_type: One of "qapv", "hypothesis", "decision", "exploration"
        cognitive_state: The cognitive state manager
        **kwargs: Pattern-specific arguments

    Returns:
        The appropriate thought pattern
    """
    patterns = {
        "qapv": lambda: QAPVPattern(
            cognitive_state,
            goal=kwargs.get("goal", "undefined goal"),
        ),
        "hypothesis": lambda: HypothesisTestingPattern(
            cognitive_state,
            initial_observation=kwargs.get("observation", ""),
        ),
        "decision": lambda: DecisionMatrixPattern(
            cognitive_state,
            decision_question=kwargs.get("question", ""),
        ),
        "exploration": lambda: ExplorationPattern(
            cognitive_state,
            exploration_goal=kwargs.get("goal", ""),
        ),
    }

    if pattern_type not in patterns:
        raise ValueError(f"Unknown pattern type: {pattern_type}")

    return patterns[pattern_type]()
