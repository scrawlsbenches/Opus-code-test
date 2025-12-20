"""
Unit tests for LoopStateSerializer.

Tests cover:
- Serialization/deserialization roundtrip for all classes
- Edge cases (empty lists, None values, nested data)
- Integration: serialize active loop with full state
- Behavioral: serialize manager with multiple loops
"""

import unittest
from datetime import datetime, timedelta
from cortical.reasoning.cognitive_loop import (
    CognitiveLoop,
    CognitiveLoopManager,
    LoopPhase,
    LoopStatus,
    TerminationReason,
    PhaseContext,
    LoopTransition,
    LoopStateSerializer,
)


class TestPhaseContextSerialization(unittest.TestCase):
    """Test serialization of PhaseContext objects."""

    def setUp(self):
        """Set up test fixtures."""
        self.serializer = LoopStateSerializer()

    def test_serialize_deserialize_basic_context(self):
        """Test basic PhaseContext roundtrip."""
        context = PhaseContext(
            phase=LoopPhase.QUESTION,
            started_at=datetime.now(),
            iteration=1,
        )

        # Serialize and deserialize
        data = self.serializer.serialize_phase_context(context)
        restored = self.serializer.deserialize_phase_context(data)

        # Verify
        self.assertEqual(restored.phase, context.phase)
        self.assertEqual(restored.iteration, context.iteration)
        self.assertEqual(restored.time_box_minutes, context.time_box_minutes)
        # Datetime comparison with tolerance
        self.assertAlmostEqual(
            restored.started_at.timestamp(),
            context.started_at.timestamp(),
            delta=1
        )

    def test_serialize_deserialize_full_context(self):
        """Test PhaseContext with all fields populated."""
        started = datetime.now()
        ended = started + timedelta(minutes=10)

        context = PhaseContext(
            phase=LoopPhase.PRODUCE,
            started_at=started,
            iteration=2,
            notes=["Note 1", "Note 2"],
            questions_raised=["Question 1?", "Question 2?"],
            decisions_made=[
                {'decision': 'Use approach A', 'rationale': 'Better performance'},
                {'decision': 'Skip feature B', 'rationale': 'Out of scope'},
            ],
            artifacts_produced=["auth.py", "test_auth.py"],
            time_box_minutes=45,
            ended_at=ended,
            duration_seconds=600.0,
        )

        # Roundtrip
        data = self.serializer.serialize_phase_context(context)
        restored = self.serializer.deserialize_phase_context(data)

        # Verify all fields
        self.assertEqual(restored.phase, context.phase)
        self.assertEqual(restored.iteration, context.iteration)
        self.assertEqual(restored.notes, context.notes)
        self.assertEqual(restored.questions_raised, context.questions_raised)
        self.assertEqual(restored.decisions_made, context.decisions_made)
        self.assertEqual(restored.artifacts_produced, context.artifacts_produced)
        self.assertEqual(restored.time_box_minutes, context.time_box_minutes)
        self.assertEqual(restored.duration_seconds, context.duration_seconds)
        self.assertIsNotNone(restored.ended_at)

    def test_serialize_empty_lists(self):
        """Test PhaseContext with empty lists."""
        context = PhaseContext(
            phase=LoopPhase.VERIFY,
            started_at=datetime.now(),
            iteration=1,
            notes=[],
            questions_raised=[],
            decisions_made=[],
            artifacts_produced=[],
        )

        data = self.serializer.serialize_phase_context(context)
        restored = self.serializer.deserialize_phase_context(data)

        self.assertEqual(restored.notes, [])
        self.assertEqual(restored.questions_raised, [])
        self.assertEqual(restored.decisions_made, [])
        self.assertEqual(restored.artifacts_produced, [])

    def test_serialize_none_ended_at(self):
        """Test PhaseContext with None ended_at (ongoing phase)."""
        context = PhaseContext(
            phase=LoopPhase.ANSWER,
            started_at=datetime.now(),
            iteration=1,
            ended_at=None,
            duration_seconds=None,
        )

        data = self.serializer.serialize_phase_context(context)
        restored = self.serializer.deserialize_phase_context(data)

        self.assertIsNone(restored.ended_at)
        self.assertIsNone(restored.duration_seconds)


class TestLoopTransitionSerialization(unittest.TestCase):
    """Test serialization of LoopTransition objects."""

    def setUp(self):
        """Set up test fixtures."""
        self.serializer = LoopStateSerializer()

    def test_serialize_deserialize_basic_transition(self):
        """Test basic LoopTransition roundtrip."""
        transition = LoopTransition(
            from_phase=LoopPhase.QUESTION,
            to_phase=LoopPhase.ANSWER,
            timestamp=datetime.now(),
            reason="Questions clarified",
        )

        # Roundtrip
        data = self.serializer.serialize_loop_transition(transition)
        restored = self.serializer.deserialize_loop_transition(data)

        # Verify
        self.assertEqual(restored.from_phase, transition.from_phase)
        self.assertEqual(restored.to_phase, transition.to_phase)
        self.assertEqual(restored.reason, transition.reason)
        self.assertAlmostEqual(
            restored.timestamp.timestamp(),
            transition.timestamp.timestamp(),
            delta=1
        )

    def test_serialize_transition_with_context_snapshot(self):
        """Test LoopTransition with context snapshot."""
        transition = LoopTransition(
            from_phase=LoopPhase.PRODUCE,
            to_phase=LoopPhase.VERIFY,
            timestamp=datetime.now(),
            reason="Implementation complete",
            context_snapshot={
                'phase': 'produce',
                'iteration': 2,
                'elapsed_minutes': 15.5,
                'notes_count': 5,
                'questions_raised': 2,
                'decisions_made': 3,
                'artifacts_produced': 4,
            }
        )

        data = self.serializer.serialize_loop_transition(transition)
        restored = self.serializer.deserialize_loop_transition(data)

        self.assertEqual(restored.context_snapshot, transition.context_snapshot)
        self.assertEqual(restored.context_snapshot['iteration'], 2)
        self.assertEqual(restored.context_snapshot['artifacts_produced'], 4)

    def test_serialize_none_from_phase(self):
        """Test transition with None from_phase (loop start)."""
        transition = LoopTransition(
            from_phase=None,
            to_phase=LoopPhase.QUESTION,
            timestamp=datetime.now(),
            reason="Loop started",
        )

        data = self.serializer.serialize_loop_transition(transition)
        restored = self.serializer.deserialize_loop_transition(data)

        self.assertIsNone(restored.from_phase)
        self.assertEqual(restored.to_phase, LoopPhase.QUESTION)

    def test_serialize_empty_context_snapshot(self):
        """Test transition with empty context snapshot."""
        transition = LoopTransition(
            from_phase=LoopPhase.ANSWER,
            to_phase=LoopPhase.PRODUCE,
            timestamp=datetime.now(),
            reason="Ready to implement",
            context_snapshot={},
        )

        data = self.serializer.serialize_loop_transition(transition)
        restored = self.serializer.deserialize_loop_transition(data)

        self.assertEqual(restored.context_snapshot, {})


class TestCognitiveLoopSerialization(unittest.TestCase):
    """Test serialization of CognitiveLoop objects."""

    def setUp(self):
        """Set up test fixtures."""
        self.serializer = LoopStateSerializer()

    def test_serialize_deserialize_basic_loop(self):
        """Test basic CognitiveLoop roundtrip."""
        loop = CognitiveLoop(
            id="test-123",
            goal="Test goal",
            status=LoopStatus.NOT_STARTED,
        )

        # Roundtrip
        json_str = self.serializer.serialize(loop)
        restored = self.serializer.deserialize(json_str)

        # Verify basic attributes
        self.assertEqual(restored.id, loop.id)
        self.assertEqual(restored.goal, loop.goal)
        self.assertEqual(restored.status, loop.status)
        self.assertIsNone(restored.current_phase)
        self.assertIsNone(restored.parent_id)
        self.assertEqual(restored.child_ids, [])

    def test_serialize_active_loop_with_state(self):
        """Test active loop with phase contexts and transitions."""
        loop = CognitiveLoop(goal="Implement authentication")

        # Start and transition through phases
        loop.start(LoopPhase.QUESTION)
        loop.add_note("Clarifying OAuth requirements")
        loop.current_context().record_question("Which OAuth provider?")
        loop.current_context().record_decision("Use Google OAuth", "Most common")

        loop.transition(LoopPhase.ANSWER, reason="Questions clarified")
        loop.add_note("Researching OAuth libraries")
        loop.current_context().record_question("Use authlib or requests-oauthlib?")

        loop.transition(LoopPhase.PRODUCE, reason="Solution designed")
        loop.current_context().artifacts_produced.append("auth.py")
        loop.current_context().artifacts_produced.append("test_auth.py")

        # Roundtrip
        json_str = self.serializer.serialize(loop)
        restored = self.serializer.deserialize(json_str)

        # Verify loop state
        self.assertEqual(restored.id, loop.id)
        self.assertEqual(restored.goal, loop.goal)
        self.assertEqual(restored.status, LoopStatus.ACTIVE)
        self.assertEqual(restored.current_phase, LoopPhase.PRODUCE)

        # Verify phase contexts
        self.assertEqual(len(restored.phase_contexts), 3)
        self.assertEqual(restored.phase_contexts[0].phase, LoopPhase.QUESTION)
        self.assertEqual(restored.phase_contexts[1].phase, LoopPhase.ANSWER)
        self.assertEqual(restored.phase_contexts[2].phase, LoopPhase.PRODUCE)

        # Verify notes and questions were preserved
        self.assertGreater(len(restored.phase_contexts[0].notes), 0)
        self.assertEqual(len(restored.phase_contexts[0].questions_raised), 1)
        self.assertEqual(len(restored.phase_contexts[0].decisions_made), 1)

        # Verify artifacts
        self.assertEqual(len(restored.phase_contexts[2].artifacts_produced), 2)
        self.assertIn("auth.py", restored.phase_contexts[2].artifacts_produced)

        # Verify transitions
        self.assertEqual(len(restored.transitions), 3)
        self.assertIsNone(restored.transitions[0].from_phase)  # Loop start
        self.assertEqual(restored.transitions[0].to_phase, LoopPhase.QUESTION)
        self.assertEqual(restored.transitions[1].from_phase, LoopPhase.QUESTION)
        self.assertEqual(restored.transitions[1].to_phase, LoopPhase.ANSWER)

    def test_serialize_completed_loop(self):
        """Test completed loop with termination reason."""
        loop = CognitiveLoop(goal="Fix bug")
        loop.start(LoopPhase.VERIFY)
        loop.complete(TerminationReason.SUCCESS)

        # Roundtrip
        json_str = self.serializer.serialize(loop)
        restored = self.serializer.deserialize(json_str)

        # Verify completion state
        self.assertEqual(restored.status, LoopStatus.COMPLETED)
        self.assertEqual(restored.termination_reason, TerminationReason.SUCCESS)
        self.assertIsNotNone(restored.completed_at)

    def test_serialize_blocked_loop(self):
        """Test blocked loop with block reason."""
        loop = CognitiveLoop(goal="Deploy to production")
        loop.start(LoopPhase.VERIFY)
        loop.block("Waiting for CI to pass")

        # Roundtrip
        json_str = self.serializer.serialize(loop)
        restored = self.serializer.deserialize(json_str)

        # Verify blocked state
        self.assertEqual(restored.status, LoopStatus.BLOCKED)
        self.assertEqual(restored.block_reason, "Waiting for CI to pass")

    def test_serialize_nested_loops(self):
        """Test parent-child loop relationships."""
        parent = CognitiveLoop(id="parent-1", goal="Main feature")
        child1 = CognitiveLoop(id="child-1", goal="Sub-task 1", parent_id="parent-1")
        child2 = CognitiveLoop(id="child-2", goal="Sub-task 2", parent_id="parent-1")
        parent.child_ids = ["child-1", "child-2"]

        # Roundtrip parent
        json_str = self.serializer.serialize(parent)
        restored_parent = self.serializer.deserialize(json_str)

        self.assertEqual(restored_parent.child_ids, ["child-1", "child-2"])
        self.assertIsNone(restored_parent.parent_id)

        # Roundtrip child
        json_str = self.serializer.serialize(child1)
        restored_child = self.serializer.deserialize(json_str)

        self.assertEqual(restored_child.parent_id, "parent-1")
        self.assertEqual(restored_child.child_ids, [])

    def test_serialize_empty_phase_contexts(self):
        """Test loop with no phase contexts (not started)."""
        loop = CognitiveLoop(goal="Future task")

        json_str = self.serializer.serialize(loop)
        restored = self.serializer.deserialize(json_str)

        self.assertEqual(restored.phase_contexts, [])
        self.assertEqual(restored.transitions, [])

    def test_serialize_multiple_iterations(self):
        """Test loop with multiple iterations of same phase."""
        loop = CognitiveLoop(goal="Debug issue")
        loop.start(LoopPhase.VERIFY)
        loop.transition(LoopPhase.QUESTION, reason="Test failed, investigating")
        loop.transition(LoopPhase.ANSWER, reason="Found root cause")
        loop.transition(LoopPhase.PRODUCE, reason="Implementing fix")
        loop.transition(LoopPhase.VERIFY, reason="Re-running tests")  # 2nd iteration

        # Roundtrip
        json_str = self.serializer.serialize(loop)
        restored = self.serializer.deserialize(json_str)

        # Verify iterations
        verify_contexts = [ctx for ctx in restored.phase_contexts if ctx.phase == LoopPhase.VERIFY]
        self.assertEqual(len(verify_contexts), 2)
        self.assertEqual(verify_contexts[0].iteration, 1)
        self.assertEqual(verify_contexts[1].iteration, 2)


class TestCognitiveLoopManagerSerialization(unittest.TestCase):
    """Test serialization of CognitiveLoopManager."""

    def setUp(self):
        """Set up test fixtures."""
        self.serializer = LoopStateSerializer()

    def test_serialize_deserialize_empty_manager(self):
        """Test empty manager roundtrip."""
        manager = CognitiveLoopManager()

        json_str = self.serializer.serialize_manager(manager)
        restored = self.serializer.deserialize_manager(json_str)

        self.assertEqual(len(restored._loops), 0)

    def test_serialize_manager_with_multiple_loops(self):
        """Test manager with multiple loops."""
        manager = CognitiveLoopManager()

        # Create several loops
        loop1 = manager.create_loop(goal="Feature A")
        loop1.start(LoopPhase.QUESTION)
        loop1.add_note("Working on feature A")

        loop2 = manager.create_loop(goal="Feature B")
        loop2.start(LoopPhase.ANSWER)
        loop2.add_note("Working on feature B")

        loop3 = manager.create_loop(goal="Bug fix C")
        loop3.start(LoopPhase.VERIFY)
        loop3.complete(TerminationReason.SUCCESS)

        # Roundtrip
        json_str = self.serializer.serialize_manager(manager)
        restored = self.serializer.deserialize_manager(json_str)

        # Verify all loops restored
        self.assertEqual(len(restored._loops), 3)
        self.assertIn(loop1.id, restored._loops)
        self.assertIn(loop2.id, restored._loops)
        self.assertIn(loop3.id, restored._loops)

        # Verify loop states
        restored_loop1 = restored.get_loop(loop1.id)
        self.assertEqual(restored_loop1.goal, "Feature A")
        self.assertEqual(restored_loop1.status, LoopStatus.ACTIVE)
        self.assertEqual(restored_loop1.current_phase, LoopPhase.QUESTION)

        restored_loop3 = restored.get_loop(loop3.id)
        self.assertEqual(restored_loop3.status, LoopStatus.COMPLETED)
        self.assertEqual(restored_loop3.termination_reason, TerminationReason.SUCCESS)

    def test_serialize_manager_with_nested_loops(self):
        """Test manager with parent-child loop hierarchies."""
        manager = CognitiveLoopManager()

        # Create parent loop
        parent = manager.create_loop(goal="Main project")
        parent.start(LoopPhase.QUESTION)

        # Create child loops
        child1 = manager.create_loop(goal="Sub-task 1", parent_id=parent.id)
        child1.start(LoopPhase.PRODUCE)
        parent.child_ids.append(child1.id)

        child2 = manager.create_loop(goal="Sub-task 2", parent_id=parent.id)
        child2.start(LoopPhase.ANSWER)
        parent.child_ids.append(child2.id)

        # Roundtrip
        json_str = self.serializer.serialize_manager(manager)
        restored = self.serializer.deserialize_manager(json_str)

        # Verify hierarchy
        restored_parent = restored.get_loop(parent.id)
        self.assertEqual(len(restored_parent.child_ids), 2)
        self.assertIn(child1.id, restored_parent.child_ids)

        restored_child1 = restored.get_loop(child1.id)
        self.assertEqual(restored_child1.parent_id, parent.id)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def setUp(self):
        """Set up test fixtures."""
        self.serializer = LoopStateSerializer()

    def test_serialize_loop_with_all_statuses(self):
        """Test loops in all possible statuses."""
        statuses = [
            LoopStatus.NOT_STARTED,
            LoopStatus.ACTIVE,
            LoopStatus.PAUSED,
            LoopStatus.BLOCKED,
            LoopStatus.COMPLETED,
            LoopStatus.ABANDONED,
        ]

        for status in statuses:
            loop = CognitiveLoop(goal=f"Test {status.name}")
            loop.status = status

            json_str = self.serializer.serialize(loop)
            restored = self.serializer.deserialize(json_str)

            self.assertEqual(restored.status, status)

    def test_serialize_loop_with_all_termination_reasons(self):
        """Test loops with all possible termination reasons."""
        reasons = [
            TerminationReason.SUCCESS,
            TerminationReason.USER_APPROVED,
            TerminationReason.BUDGET_EXHAUSTED,
            TerminationReason.QUESTION_INVALID,
            TerminationReason.ESCALATED,
            TerminationReason.CRISIS,
        ]

        for reason in reasons:
            loop = CognitiveLoop(goal=f"Test {reason.value}")
            loop.status = LoopStatus.COMPLETED
            loop.termination_reason = reason

            json_str = self.serializer.serialize(loop)
            restored = self.serializer.deserialize(json_str)

            self.assertEqual(restored.termination_reason, reason)

    def test_serialize_complex_decisions(self):
        """Test complex decision structures."""
        context = PhaseContext(
            phase=LoopPhase.ANSWER,
            started_at=datetime.now(),
            iteration=1,
        )

        # Add complex decisions with nested data
        context.decisions_made.append({
            'decision': 'Use microservices architecture',
            'rationale': 'Better scalability and independent deployment',
            'timestamp': datetime.now().isoformat(),
            'alternatives_considered': ['monolith', 'SOA'],
            'trade_offs': {
                'pros': ['scalability', 'flexibility'],
                'cons': ['complexity', 'overhead']
            }
        })

        data = self.serializer.serialize_phase_context(context)
        restored = self.serializer.deserialize_phase_context(data)

        self.assertEqual(len(restored.decisions_made), 1)
        self.assertEqual(restored.decisions_made[0]['decision'], 'Use microservices architecture')
        self.assertIn('alternatives_considered', restored.decisions_made[0])
        self.assertIn('trade_offs', restored.decisions_made[0])


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for realistic workflows."""

    def setUp(self):
        """Set up test fixtures."""
        self.serializer = LoopStateSerializer()

    def test_full_qapv_cycle_roundtrip(self):
        """Test complete QAPV cycle serialization."""
        loop = CognitiveLoop(goal="Implement user registration")

        # QUESTION phase
        loop.start(LoopPhase.QUESTION)
        loop.add_note("Clarifying requirements")
        loop.current_context().record_question("What authentication method?")
        loop.current_context().record_question("Password complexity rules?")
        loop.current_context().record_decision("Use email + password", "Standard approach")

        # ANSWER phase
        loop.transition(LoopPhase.ANSWER, reason="Requirements clarified")
        loop.add_note("Researching libraries")
        loop.current_context().record_question("Use bcrypt or argon2?")
        loop.current_context().record_decision("Use argon2", "Better security")

        # PRODUCE phase
        loop.transition(LoopPhase.PRODUCE, reason="Design complete")
        loop.add_note("Implementing registration endpoint")
        loop.current_context().artifacts_produced.append("auth/register.py")
        loop.current_context().artifacts_produced.append("tests/test_register.py")
        loop.current_context().artifacts_produced.append("docs/api.md")

        # VERIFY phase
        loop.transition(LoopPhase.VERIFY, reason="Implementation complete")
        loop.add_note("Running tests")
        loop.add_note("All tests passed")

        # Complete
        loop.complete(TerminationReason.SUCCESS)

        # Roundtrip
        json_str = self.serializer.serialize(loop)
        restored = self.serializer.deserialize(json_str)

        # Verify full cycle
        self.assertEqual(len(restored.phase_contexts), 4)
        self.assertEqual(restored.phase_contexts[0].phase, LoopPhase.QUESTION)
        self.assertEqual(restored.phase_contexts[1].phase, LoopPhase.ANSWER)
        self.assertEqual(restored.phase_contexts[2].phase, LoopPhase.PRODUCE)
        self.assertEqual(restored.phase_contexts[3].phase, LoopPhase.VERIFY)

        # Verify questions from QUESTION phase
        self.assertEqual(len(restored.phase_contexts[0].questions_raised), 2)
        self.assertIn("authentication method", restored.phase_contexts[0].questions_raised[0])

        # Verify artifacts from PRODUCE phase
        self.assertEqual(len(restored.phase_contexts[2].artifacts_produced), 3)
        self.assertIn("auth/register.py", restored.phase_contexts[2].artifacts_produced)

        # Verify completion
        self.assertEqual(restored.status, LoopStatus.COMPLETED)
        self.assertEqual(restored.termination_reason, TerminationReason.SUCCESS)

    def test_manager_state_persistence(self):
        """Test manager state can be persisted and restored."""
        # Create manager with complex state
        manager = CognitiveLoopManager()

        # Active development work
        active_loop = manager.create_loop(goal="Feature development")
        active_loop.start(LoopPhase.PRODUCE)
        active_loop.add_note("In progress")

        # Blocked on dependency
        blocked_loop = manager.create_loop(goal="Integration")
        blocked_loop.start(LoopPhase.VERIFY)
        blocked_loop.block("Waiting for API deployment")

        # Completed work
        completed_loop = manager.create_loop(goal="Bug fix")
        completed_loop.start(LoopPhase.VERIFY)
        completed_loop.complete(TerminationReason.SUCCESS)

        # Nested work
        parent = manager.create_loop(goal="Epic feature")
        parent.start(LoopPhase.QUESTION)
        child = manager.create_loop(goal="Sub-feature", parent_id=parent.id)
        child.start(LoopPhase.ANSWER)
        parent.child_ids.append(child.id)

        # Serialize and restore
        json_str = self.serializer.serialize_manager(manager)
        restored_manager = self.serializer.deserialize_manager(json_str)

        # Verify manager methods work
        active_loops = restored_manager.get_active_loops()
        blocked_loops = restored_manager.get_blocked_loops()

        self.assertEqual(len(active_loops), 3)  # active_loop, parent, child
        self.assertEqual(len(blocked_loops), 1)

        # Verify specific loop retrieval
        restored_blocked = restored_manager.get_loop(blocked_loop.id)
        self.assertEqual(restored_blocked.status, LoopStatus.BLOCKED)
        self.assertEqual(restored_blocked.block_reason, "Waiting for API deployment")

        # Verify hierarchy preserved
        restored_parent = restored_manager.get_loop(parent.id)
        restored_child = restored_manager.get_loop(child.id)
        self.assertIn(child.id, restored_parent.child_ids)
        self.assertEqual(restored_child.parent_id, parent.id)


if __name__ == '__main__':
    unittest.main()
