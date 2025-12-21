"""
Tests for QAPV Cycle Behavioral Validation.

This module tests the LoopValidator class to ensure it correctly identifies
violations of QAPV best practices in CognitiveLoop instances.
"""

import unittest
from cortical.reasoning import CognitiveLoop, LoopPhase, LoopStatus
from cortical.reasoning.loop_validator import LoopValidator, ValidationResult


class TestLoopValidator(unittest.TestCase):
    """Tests for LoopValidator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = LoopValidator()

    def test_valid_loop_passes_all_checks(self):
        """Valid loop following QAPV should pass all validations."""
        # Create a valid loop that follows QAPV
        loop = CognitiveLoop(goal="Implement authentication")

        # Start in QUESTION phase
        ctx = loop.start(LoopPhase.QUESTION)
        ctx.record_question("What auth method should we use?")

        # Move to ANSWER with decisions
        ctx = loop.transition(LoopPhase.ANSWER, "User clarified: use OAuth2")
        ctx.record_decision("Use OAuth2", "User requires third-party login support")

        # Move to PRODUCE with artifacts
        ctx = loop.transition(LoopPhase.PRODUCE, "Solution approved, begin implementation")
        ctx.artifacts_produced.append("cortical/auth.py")
        ctx.artifacts_produced.append("tests/test_auth.py")

        # Move to VERIFY with verification notes
        ctx = loop.transition(LoopPhase.VERIFY, "Implementation complete, running tests")
        ctx.add_note("Tests passed: 5/5")
        ctx.add_note("Coverage: 95%")

        # Validate
        results = self.validator.validate(loop)
        summary = self.validator.get_summary(results)

        # All checks should pass
        self.assertEqual(summary['failed'], 0, f"Expected no failures, got: {summary['failures']}")
        self.assertGreater(summary['passed'], 0)

        # Check individual results
        for result in results:
            if result.severity == "error":
                self.assertTrue(result.passed, f"Error-level check failed: {result.message}")

    def test_loop_skipping_question_phase_fails_validation(self):
        """Loop starting in ANSWER instead of QUESTION should fail."""
        # Create loop starting in wrong phase
        loop = CognitiveLoop(goal="Quick fix")
        loop.start(LoopPhase.ANSWER)  # Skipped QUESTION!

        # Validate
        results = self.validator.validate(loop)

        # Should have at least one failed validation
        failed_results = [r for r in results if not r.passed]
        self.assertGreater(len(failed_results), 0)

        # Specifically check question_phase_required failed
        question_phase_result = next(
            (r for r in results if r.rule_name == "question_phase_required"),
            None
        )
        self.assertIsNotNone(question_phase_result)
        self.assertFalse(question_phase_result.passed)
        self.assertEqual(question_phase_result.severity, "error")
        self.assertIn("started in answer instead of question", question_phase_result.message.lower())

    def test_answer_without_decisions_triggers_warning(self):
        """ANSWER phase with no decisions should trigger a warning."""
        # Create loop with empty ANSWER phase
        loop = CognitiveLoop(goal="Research task")
        loop.start(LoopPhase.QUESTION)
        loop.transition(LoopPhase.ANSWER, "Moving to research")
        # No decisions recorded!

        # Validate
        results = self.validator.validate(loop)

        # Check answer_evidence_required
        answer_result = next(
            (r for r in results if r.rule_name == "answer_evidence_required"),
            None
        )
        self.assertIsNotNone(answer_result)
        self.assertFalse(answer_result.passed)
        self.assertEqual(answer_result.severity, "warning")
        self.assertIn("no decisions", answer_result.message.lower())

    def test_answer_with_decisions_lacking_rationales_triggers_warning(self):
        """ANSWER phase with decisions but missing rationales should warn."""
        # Create loop with decisions but no rationales
        loop = CognitiveLoop(goal="Decision task")
        loop.start(LoopPhase.QUESTION)
        ctx = loop.transition(LoopPhase.ANSWER, "Making decisions")

        # Add decision without rationale
        ctx.decisions_made.append({
            'decision': 'Use Redis for caching',
            'rationale': '',  # Empty rationale!
            'timestamp': '2025-12-19T10:00:00'
        })

        # Validate
        results = self.validator.validate(loop)

        # Check answer_evidence_required
        answer_result = next(
            (r for r in results if r.rule_name == "answer_evidence_required"),
            None
        )
        self.assertIsNotNone(answer_result)
        self.assertFalse(answer_result.passed)
        self.assertEqual(answer_result.severity, "warning")
        self.assertIn("lack rationales", answer_result.message.lower())

    def test_produce_without_artifacts_triggers_warning(self):
        """PRODUCE phase with no artifacts should trigger a warning."""
        # Create loop with empty PRODUCE phase
        loop = CognitiveLoop(goal="Build feature")
        loop.start(LoopPhase.QUESTION)
        loop.transition(LoopPhase.ANSWER, "Decided on approach")
        loop.transition(LoopPhase.PRODUCE, "Start implementation")
        # No artifacts recorded!

        # Validate
        results = self.validator.validate(loop)

        # Check produce_artifacts_required
        produce_result = next(
            (r for r in results if r.rule_name == "produce_artifacts_required"),
            None
        )
        self.assertIsNotNone(produce_result)
        self.assertFalse(produce_result.passed)
        self.assertEqual(produce_result.severity, "warning")
        self.assertIn("no artifacts", produce_result.message.lower())

    def test_verify_without_notes_triggers_warning(self):
        """VERIFY phase with no notes should trigger a warning."""
        # Create loop with empty VERIFY phase
        loop = CognitiveLoop(goal="Test feature")
        loop.start(LoopPhase.QUESTION)
        loop.transition(LoopPhase.ANSWER, "Decided on tests")
        loop.transition(LoopPhase.PRODUCE, "Wrote tests")
        loop.transition(LoopPhase.VERIFY, "Running verification")
        # No notes added!

        # Validate
        results = self.validator.validate(loop)

        # Check verify_checks_required
        verify_result = next(
            (r for r in results if r.rule_name == "verify_checks_required"),
            None
        )
        self.assertIsNotNone(verify_result)
        self.assertFalse(verify_result.passed)
        self.assertEqual(verify_result.severity, "warning")
        self.assertIn("no verification notes", verify_result.message.lower())

    def test_backward_skip_detected(self):
        """Jumping backward by more than one phase should be detected."""
        # Create loop with backward skip
        loop = CognitiveLoop(goal="Complex task")
        loop.start(LoopPhase.QUESTION)
        loop.transition(LoopPhase.ANSWER, "Answered questions")
        loop.transition(LoopPhase.PRODUCE, "Building solution")
        # Skip back from PRODUCE to QUESTION (skipping VERIFY and ANSWER)
        loop.transition(LoopPhase.QUESTION, "Found new questions")

        # Validate
        results = self.validator.validate(loop)

        # Check no_backward_skips
        backward_results = [r for r in results if r.rule_name == "no_backward_skips"]
        self.assertGreater(len(backward_results), 0)

        # Should have at least one failed backward skip check
        failed_backward = [r for r in backward_results if not r.passed]
        self.assertGreater(len(failed_backward), 0)

        # Check the message
        skip_result = failed_backward[0]
        self.assertEqual(skip_result.severity, "warning")
        self.assertIn("backward skip", skip_result.message.lower())
        self.assertIn("produce", skip_result.message.lower())
        self.assertIn("question", skip_result.message.lower())

    def test_normal_backward_transition_allowed(self):
        """Going back one phase (e.g., VERIFY -> PRODUCE) is allowed."""
        # Create loop with normal backward transition
        loop = CognitiveLoop(goal="Iterative task")
        loop.start(LoopPhase.QUESTION)
        loop.transition(LoopPhase.ANSWER, "Initial answer")
        loop.transition(LoopPhase.PRODUCE, "Initial implementation")
        loop.transition(LoopPhase.VERIFY, "Testing")
        # Go back one phase (normal iteration)
        loop.transition(LoopPhase.PRODUCE, "Fix found issues")

        # Validate
        results = self.validator.validate(loop)

        # Check no_backward_skips
        backward_results = [r for r in results if r.rule_name == "no_backward_skips"]

        # Should all pass (no skips, just normal backward flow)
        for result in backward_results:
            self.assertTrue(result.passed)

    def test_loop_without_phases_handled_gracefully(self):
        """Loop with missing phases should not crash validator."""
        # Create loop with only QUESTION and ANSWER
        loop = CognitiveLoop(goal="Partial task")
        loop.start(LoopPhase.QUESTION)
        loop.transition(LoopPhase.ANSWER, "Answered")
        # No PRODUCE or VERIFY

        # Validate - should not crash
        results = self.validator.validate(loop)
        summary = self.validator.get_summary(results)

        # Should have some results
        self.assertGreater(len(results), 0)

        # PRODUCE and VERIFY checks should pass (not applicable)
        produce_result = next(
            (r for r in results if r.rule_name == "produce_artifacts_required"),
            None
        )
        verify_result = next(
            (r for r in results if r.rule_name == "verify_checks_required"),
            None
        )

        self.assertIsNotNone(produce_result)
        self.assertIsNotNone(verify_result)
        self.assertTrue(produce_result.passed)
        self.assertTrue(verify_result.passed)

    def test_get_summary_structure(self):
        """get_summary should return properly structured data."""
        # Create a loop with some violations
        loop = CognitiveLoop(goal="Test summary")
        loop.start(LoopPhase.ANSWER)  # Wrong starting phase

        # Validate
        results = self.validator.validate(loop)
        summary = self.validator.get_summary(results)

        # Check summary structure
        self.assertIn('total_checks', summary)
        self.assertIn('passed', summary)
        self.assertIn('failed', summary)
        self.assertIn('by_severity', summary)
        self.assertIn('failures', summary)

        # Check types
        self.assertIsInstance(summary['total_checks'], int)
        self.assertIsInstance(summary['passed'], int)
        self.assertIsInstance(summary['failed'], int)
        self.assertIsInstance(summary['by_severity'], dict)
        self.assertIsInstance(summary['failures'], list)

        # Check severity breakdown
        self.assertIn('error', summary['by_severity'])
        self.assertIn('warning', summary['by_severity'])
        self.assertIn('info', summary['by_severity'])

        # Check failures format
        if summary['failures']:
            failure = summary['failures'][0]
            self.assertIn('rule', failure)
            self.assertIn('message', failure)
            self.assertIn('severity', failure)
            self.assertIn('phase', failure)

    def test_multiple_answer_phases_aggregated(self):
        """Multiple ANSWER phases should be checked together."""
        # Create loop with multiple ANSWER iterations
        loop = CognitiveLoop(goal="Complex decision")
        loop.start(LoopPhase.QUESTION)

        # First ANSWER with decision
        ctx1 = loop.transition(LoopPhase.ANSWER, "Initial analysis")
        ctx1.record_decision("Use approach A", "Best for performance")

        # Second ANSWER with decision
        loop.transition(LoopPhase.PRODUCE, "Try implementation")
        loop.transition(LoopPhase.VERIFY, "Test approach A")
        loop.transition(LoopPhase.QUESTION, "Found issues")
        ctx2 = loop.transition(LoopPhase.ANSWER, "Re-analyze")
        ctx2.record_decision("Switch to approach B", "Approach A had edge cases")

        # Validate
        results = self.validator.validate(loop)

        # Check answer_evidence_required
        answer_result = next(
            (r for r in results if r.rule_name == "answer_evidence_required"),
            None
        )
        self.assertIsNotNone(answer_result)
        self.assertTrue(answer_result.passed)
        self.assertIn("2 decisions", answer_result.message)

    def test_validation_result_attributes(self):
        """ValidationResult should have all expected attributes."""
        # Create a simple result
        result = ValidationResult(
            rule_name="test_rule",
            passed=True,
            message="Test passed",
            severity="info",
            phase=LoopPhase.QUESTION
        )

        # Check attributes
        self.assertEqual(result.rule_name, "test_rule")
        self.assertTrue(result.passed)
        self.assertEqual(result.message, "Test passed")
        self.assertEqual(result.severity, "info")
        self.assertEqual(result.phase, LoopPhase.QUESTION)

    def test_empty_loop_handled(self):
        """Loop with no transitions should be handled gracefully."""
        # Create empty loop (not started)
        loop = CognitiveLoop(goal="Not started")

        # Validate
        results = self.validator.validate(loop)

        # Should have results (even if checking empty state)
        self.assertGreater(len(results), 0)

        # Question phase check should fail (no transitions)
        question_result = next(
            (r for r in results if r.rule_name == "question_phase_required"),
            None
        )
        self.assertIsNotNone(question_result)
        self.assertFalse(question_result.passed)

    def test_all_validation_rules_executed(self):
        """validate() should execute all validation rules."""
        # Create a complete loop
        loop = CognitiveLoop(goal="Complete workflow")
        loop.start(LoopPhase.QUESTION)
        ctx = loop.transition(LoopPhase.ANSWER, "Analyzed")
        ctx.record_decision("Decision", "Rationale")
        ctx = loop.transition(LoopPhase.PRODUCE, "Building")
        ctx.artifacts_produced.append("file.py")
        ctx = loop.transition(LoopPhase.VERIFY, "Testing")
        ctx.add_note("Tests passed")

        # Validate
        results = self.validator.validate(loop)

        # Check all rules are present
        rule_names = {r.rule_name for r in results}
        expected_rules = {
            "question_phase_required",
            "answer_evidence_required",
            "produce_artifacts_required",
            "verify_checks_required",
            "no_backward_skips"
        }

        self.assertTrue(expected_rules.issubset(rule_names))


if __name__ == '__main__':
    unittest.main()
