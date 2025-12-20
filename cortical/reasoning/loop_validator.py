"""
QAPV Cycle Behavioral Validation.

This module provides validation logic for ensuring CognitiveLoop instances
follow QAPV (Question-Answer-Produce-Verify) best practices. It can audit
a loop's history and flag violations of the cognitive workflow rules.

The validator checks for:
- Proper phase entry (must start with QUESTION)
- Evidence in ANSWER phase (decisions with rationales)
- Artifact tracking in PRODUCE phase
- Verification checks in VERIFY phase
- Proper phase transitions (no improper backward skips)

Usage:
    >>> from cortical.reasoning import CognitiveLoop, LoopPhase
    >>> from cortical.reasoning.loop_validator import LoopValidator
    >>>
    >>> loop = CognitiveLoop(goal="Implement feature")
    >>> loop.start(LoopPhase.QUESTION)
    >>> loop.transition(LoopPhase.ANSWER, "Clarified requirements")
    >>>
    >>> validator = LoopValidator()
    >>> results = validator.validate(loop)
    >>> summary = validator.get_summary(results)
    >>> print(f"Passed: {summary['passed']}/{summary['total_checks']}")
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .cognitive_loop import CognitiveLoop, LoopPhase


@dataclass
class ValidationResult:
    """
    Result of validating a single QAPV rule.

    Attributes:
        rule_name: Identifier for the rule being validated
        passed: Whether the validation passed
        message: Human-readable description of the result
        severity: Severity level ("error", "warning", "info")
        phase: The LoopPhase associated with this validation (if applicable)
    """
    rule_name: str
    passed: bool
    message: str
    severity: str  # "error", "warning", "info"
    phase: Optional[LoopPhase] = None


class LoopValidator:
    """
    Validates CognitiveLoop instances follow QAPV best practices.

    This validator performs behavioral validation on cognitive loops to ensure
    they follow the Question-Answer-Produce-Verify workflow correctly. It checks
    both structural requirements (phase ordering) and content requirements
    (evidence, artifacts, verification notes).

    Validation Rules:
        1. question_phase_required: Loop must start in QUESTION phase
        2. answer_evidence_required: ANSWER phase must have decisions with rationales
        3. produce_artifacts_required: PRODUCE phase must create artifacts
        4. verify_checks_required: VERIFY phase must have verification notes
        5. no_backward_skips: No improper backward phase transitions

    Example:
        >>> validator = LoopValidator()
        >>> loop = create_valid_loop()
        >>> results = validator.validate(loop)
        >>>
        >>> for result in results:
        ...     if not result.passed:
        ...         print(f"FAILED: {result.message}")
        >>>
        >>> summary = validator.get_summary(results)
        >>> if summary['failed'] == 0:
        ...     print("All validations passed!")
    """

    def validate(self, loop: CognitiveLoop) -> List[ValidationResult]:
        """
        Run all validations on a cognitive loop.

        Args:
            loop: The CognitiveLoop to validate

        Returns:
            List of ValidationResult objects, one per rule checked
        """
        results = []
        results.append(self.validate_question_phase(loop))
        results.append(self.validate_answer_evidence(loop))
        results.append(self.validate_produce_artifacts(loop))
        results.append(self.validate_verify_checks(loop))
        results.extend(self.validate_no_backward_skips(loop))
        return results

    def validate_question_phase(self, loop: CognitiveLoop) -> ValidationResult:
        """
        Check loop started in QUESTION phase.

        QAPV requires starting with understanding before acting. Skipping the
        QUESTION phase means acting without proper context.

        Args:
            loop: The CognitiveLoop to validate

        Returns:
            ValidationResult indicating whether loop started in QUESTION
        """
        if not loop.transitions:
            return ValidationResult(
                rule_name="question_phase_required",
                passed=False,
                message="Loop has no transitions",
                severity="error"
            )

        first_transition = loop.transitions[0]
        if first_transition.to_phase == LoopPhase.QUESTION:
            return ValidationResult(
                rule_name="question_phase_required",
                passed=True,
                message="Loop correctly started in QUESTION phase",
                severity="info",
                phase=LoopPhase.QUESTION
            )
        else:
            return ValidationResult(
                rule_name="question_phase_required",
                passed=False,
                message=f"Loop started in {first_transition.to_phase.value} instead of QUESTION",
                severity="error",
                phase=first_transition.to_phase
            )

    def validate_answer_evidence(self, loop: CognitiveLoop) -> ValidationResult:
        """
        Check ANSWER phase has decisions with rationales.

        The ANSWER phase should produce decisions backed by reasoning. Empty
        decisions or missing rationales suggest incomplete analysis.

        Args:
            loop: The CognitiveLoop to validate

        Returns:
            ValidationResult indicating whether ANSWER phase has proper evidence
        """
        # Find ANSWER phase contexts
        answer_contexts = [ctx for ctx in loop.phase_contexts if ctx.phase == LoopPhase.ANSWER]

        if not answer_contexts:
            return ValidationResult(
                rule_name="answer_evidence_required",
                passed=True,
                message="No ANSWER phase found (not applicable)",
                severity="info"
            )

        # Check if any ANSWER phase has decisions
        has_decisions = any(len(ctx.decisions_made) > 0 for ctx in answer_contexts)

        if has_decisions:
            # Check if decisions have rationales
            total_decisions = sum(len(ctx.decisions_made) for ctx in answer_contexts)
            decisions_with_rationale = sum(
                1 for ctx in answer_contexts
                for decision in ctx.decisions_made
                if decision.get('rationale')
            )

            if decisions_with_rationale == total_decisions:
                return ValidationResult(
                    rule_name="answer_evidence_required",
                    passed=True,
                    message=f"All {total_decisions} decisions have rationales",
                    severity="info",
                    phase=LoopPhase.ANSWER
                )
            else:
                missing = total_decisions - decisions_with_rationale
                return ValidationResult(
                    rule_name="answer_evidence_required",
                    passed=False,
                    message=f"{missing} decision(s) lack rationales",
                    severity="warning",
                    phase=LoopPhase.ANSWER
                )
        else:
            return ValidationResult(
                rule_name="answer_evidence_required",
                passed=False,
                message="ANSWER phase has no decisions recorded",
                severity="warning",
                phase=LoopPhase.ANSWER
            )

    def validate_produce_artifacts(self, loop: CognitiveLoop) -> ValidationResult:
        """
        Check PRODUCE phase created artifacts.

        The PRODUCE phase should create tangible artifacts (code, docs, tests).
        Missing artifacts suggest the phase was skipped or incomplete.

        Args:
            loop: The CognitiveLoop to validate

        Returns:
            ValidationResult indicating whether PRODUCE phase tracked artifacts
        """
        # Find PRODUCE phase contexts
        produce_contexts = [ctx for ctx in loop.phase_contexts if ctx.phase == LoopPhase.PRODUCE]

        if not produce_contexts:
            return ValidationResult(
                rule_name="produce_artifacts_required",
                passed=True,
                message="No PRODUCE phase found (not applicable)",
                severity="info"
            )

        # Check if any PRODUCE phase has artifacts
        has_artifacts = any(len(ctx.artifacts_produced) > 0 for ctx in produce_contexts)

        if has_artifacts:
            total_artifacts = sum(len(ctx.artifacts_produced) for ctx in produce_contexts)
            return ValidationResult(
                rule_name="produce_artifacts_required",
                passed=True,
                message=f"PRODUCE phase created {total_artifacts} artifact(s)",
                severity="info",
                phase=LoopPhase.PRODUCE
            )
        else:
            return ValidationResult(
                rule_name="produce_artifacts_required",
                passed=False,
                message="PRODUCE phase has no artifacts recorded",
                severity="warning",
                phase=LoopPhase.PRODUCE
            )

    def validate_verify_checks(self, loop: CognitiveLoop) -> ValidationResult:
        """
        Check VERIFY phase has verification notes.

        The VERIFY phase should perform checks and document results. Missing
        notes suggest verification was skipped or not documented.

        Args:
            loop: The CognitiveLoop to validate

        Returns:
            ValidationResult indicating whether VERIFY phase has verification notes
        """
        # Find VERIFY phase contexts
        verify_contexts = [ctx for ctx in loop.phase_contexts if ctx.phase == LoopPhase.VERIFY]

        if not verify_contexts:
            return ValidationResult(
                rule_name="verify_checks_required",
                passed=True,
                message="No VERIFY phase found (not applicable)",
                severity="info"
            )

        # Check if any VERIFY phase has notes
        has_notes = any(len(ctx.notes) > 0 for ctx in verify_contexts)

        if has_notes:
            total_notes = sum(len(ctx.notes) for ctx in verify_contexts)
            return ValidationResult(
                rule_name="verify_checks_required",
                passed=True,
                message=f"VERIFY phase has {total_notes} verification note(s)",
                severity="info",
                phase=LoopPhase.VERIFY
            )
        else:
            return ValidationResult(
                rule_name="verify_checks_required",
                passed=False,
                message="VERIFY phase has no verification notes",
                severity="warning",
                phase=LoopPhase.VERIFY
            )

    def validate_no_backward_skips(self, loop: CognitiveLoop) -> List[ValidationResult]:
        """
        Check transitions don't skip phases backward improperly.

        While loops can cycle back (VERIFY → QUESTION for iteration), jumping
        backward by more than one phase (e.g., PRODUCE → QUESTION) suggests
        skipping important steps.

        Args:
            loop: The CognitiveLoop to validate

        Returns:
            List of ValidationResult objects for backward skip violations
        """
        results = []

        # Define phase order (for detecting backward skips)
        phase_order = {
            LoopPhase.QUESTION: 0,
            LoopPhase.ANSWER: 1,
            LoopPhase.PRODUCE: 2,
            LoopPhase.VERIFY: 3
        }

        for i in range(1, len(loop.transitions)):
            prev_transition = loop.transitions[i - 1]
            curr_transition = loop.transitions[i]

            from_phase = prev_transition.to_phase
            to_phase = curr_transition.to_phase

            # Check for backward skip (e.g., PRODUCE -> QUESTION without going through VERIFY)
            if phase_order[to_phase] < phase_order[from_phase]:
                # It's a backward transition
                # Check if it's jumping more than one phase back
                phase_diff = phase_order[from_phase] - phase_order[to_phase]

                if phase_diff > 1:
                    # This is a skip (e.g., PRODUCE -> QUESTION skips VERIFY and ANSWER)
                    results.append(ValidationResult(
                        rule_name="no_backward_skips",
                        passed=False,
                        message=f"Backward skip detected: {from_phase.value} -> {to_phase.value} (skipped {phase_diff - 1} phase(s))",
                        severity="warning",
                        phase=to_phase
                    ))

        if not results:
            # No backward skips detected
            results.append(ValidationResult(
                rule_name="no_backward_skips",
                passed=True,
                message="No improper backward phase skips detected",
                severity="info"
            ))

        return results

    def get_summary(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """
        Summarize validation results.

        Args:
            results: List of ValidationResult objects from validate()

        Returns:
            Dictionary with summary statistics:
                - total_checks: Total number of validations run
                - passed: Number of validations that passed
                - failed: Number of validations that failed
                - by_severity: Count of results by severity level
                - failures: List of failed validation details
        """
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed

        by_severity = {}
        for severity in ["error", "warning", "info"]:
            by_severity[severity] = sum(1 for r in results if r.severity == severity)

        failures = [r for r in results if not r.passed]

        return {
            'total_checks': total,
            'passed': passed,
            'failed': failed,
            'by_severity': by_severity,
            'failures': [
                {
                    'rule': r.rule_name,
                    'message': r.message,
                    'severity': r.severity,
                    'phase': r.phase.value if r.phase else None
                }
                for r in failures
            ]
        }
