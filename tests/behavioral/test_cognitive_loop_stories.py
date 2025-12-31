"""
Behavioral Tests for QAPV Cognitive Loop.

This module tests the Question-Answer-Produce-Verify loop implementation,
the core reasoning cycle we built ourselves for complex problem solving.

Epic: Developer orchestrates complex reasoning workflows
Story: As a developer building custom reasoning systems,
       I want a structured QAPV cognitive loop we built from scratch,
       So that I can manage complex multi-step reasoning we control completely.
"""

import pytest
from cortical.reasoning.cognitive_loop import (
    CognitiveLoop,
    CognitiveLoopManager,
    LoopPhase,
    LoopStatus,
    TerminationReason,
)


class DeveloperOrchestratesComplexReasoning:
    """
    Epic: Developer Orchestrates Complex Reasoning Workflows

    As a developer building custom reasoning orchestration,
    I want structured cognitive loops we implemented ourselves,
    So that I control the reasoning process completely.
    """

    def test_scenario_loop_starts_at_question_phase(self):
        """
        Scenario: Reasoning begins with clarifying questions

        Given a new reasoning task we're managing
        When I start the cognitive loop
        Then it begins in QUESTION phase for requirement clarification
        Because we built our loop to start with understanding
        """
        # Given a new reasoning task
        loop = CognitiveLoop(goal="Implement custom search algorithm")

        # When starting the loop
        context = loop.start(LoopPhase.QUESTION)

        # Then question phase is active
        assert loop.status == LoopStatus.ACTIVE
        assert loop.current_phase == LoopPhase.QUESTION
        assert context.phase == LoopPhase.QUESTION

    def test_scenario_loop_transitions_through_qapv_phases(self):
        """
        Scenario: Reasoning progresses through structured phases

        Given an active cognitive loop we built
        When I transition through QAPV phases
        Then each transition is recorded and tracked
        Because we need to observe our reasoning process
        """
        # Given active loop
        loop = CognitiveLoop(goal="Build custom reasoning engine")
        loop.start(LoopPhase.QUESTION)

        # When transitioning through phases
        loop.transition(LoopPhase.ANSWER, reason="Requirements clarified")
        loop.transition(LoopPhase.PRODUCE, reason="Approach approved")
        loop.transition(LoopPhase.VERIFY, reason="Implementation complete")

        # Then transitions are recorded
        assert loop.current_phase == LoopPhase.VERIFY
        assert len(loop.transitions) == 4  # start + 3 transitions
        assert all(t.reason for t in loop.transitions)

    def test_scenario_loop_captures_decisions_and_notes(self):
        """
        Scenario: Loop records decisions and observations

        Given a running loop managing our reasoning
        When I record decisions and notes
        Then context preserves the reasoning trail
        Because we built traceability into our system
        """
        # Given running loop
        loop = CognitiveLoop(goal="Design custom indexing strategy")
        loop.start(LoopPhase.ANSWER)

        # When recording decisions
        loop.current_context().record_decision(
            "Use inverted index we built ourselves",
            "Full control over implementation"
        )
        loop.add_note("Considered B-tree but chose simpler approach")

        # Then context preserves information
        context = loop.current_context()
        assert len(context.decisions_made) == 1
        assert len(context.notes) == 1
        assert "inverted index" in context.decisions_made[0]['decision']

    def test_scenario_loop_detects_iteration_without_progress(self):
        """
        Scenario: System detects stuck loops needing intervention

        Given a loop cycling through phases we're managing
        When the same phase repeats multiple times
        Then iteration count tracks potential stuck state
        Because we built crisis detection into our system
        """
        # Given cycling loop
        loop = CognitiveLoop(goal="Optimize custom query performance")
        loop.start(LoopPhase.PRODUCE)

        # When repeating phase
        for i in range(3):
            loop.transition(LoopPhase.VERIFY, reason=f"Attempt {i+1}")
            loop.transition(LoopPhase.PRODUCE, reason="Rework needed")

        # Then iteration count reflects repetition
        produce_iterations = loop.get_iteration_count(LoopPhase.PRODUCE)
        assert produce_iterations >= 3
        assert len(loop.transitions) > 6

    def test_scenario_nested_loops_decompose_complex_work(self):
        """
        Scenario: Complex tasks spawn nested cognitive loops

        Given a complex task managed by our loop system
        When I spawn child loops for sub-tasks
        Then parent-child relationships are tracked
        Because we built hierarchical decomposition ourselves
        """
        # Given complex parent task
        parent = CognitiveLoop(goal="Build complete custom search system")
        parent.start()

        # When spawning child loops
        indexer_loop = parent.spawn_child("Implement custom indexer")
        ranker_loop = parent.spawn_child("Build ranking algorithm ourselves")

        # Then relationships are tracked
        assert indexer_loop.parent_id == parent.id
        assert ranker_loop.parent_id == parent.id
        assert len(parent.child_ids) == 2
        assert indexer_loop.id in parent.child_ids

    def test_scenario_loop_can_pause_and_resume(self):
        """
        Scenario: Loops pause for external input and resume

        Given an active loop we're controlling
        When I pause for external input
        Then loop suspends and can resume
        Because we need to coordinate with external systems we built
        """
        # Given active loop
        loop = CognitiveLoop(goal="Integrate custom module")
        loop.start()

        # When pausing
        loop.pause("Waiting for user approval")

        # Then loop is paused
        assert loop.status == LoopStatus.PAUSED

        # When resuming
        loop.resume()

        # Then loop is active again
        assert loop.status == LoopStatus.ACTIVE

    def test_scenario_loop_completes_with_termination_reason(self):
        """
        Scenario: Completed loops record why they finished

        Given a loop reaching its conclusion
        When I mark it complete
        Then termination reason and timestamp are captured
        Because we track outcomes in our system
        """
        # Given concluding loop
        loop = CognitiveLoop(goal="Verify custom implementation")
        loop.start()
        loop.transition(LoopPhase.VERIFY, "Ready to verify")

        # When completing
        loop.complete(TerminationReason.SUCCESS)

        # Then completion is recorded
        assert loop.status == LoopStatus.COMPLETED
        assert loop.termination_reason == TerminationReason.SUCCESS
        assert loop.completed_at is not None


class ReasoningSystemManagesMultipleLoops:
    """
    Epic: System Manages Multiple Concurrent Reasoning Loops

    As a system managing parallel reasoning built ourselves,
    I want to track and coordinate multiple loops,
    So that I can orchestrate complex workflows we control.
    """

    def test_scenario_manager_creates_and_tracks_loops(self):
        """
        Scenario: Loop manager coordinates multiple reasoning threads

        Given a loop manager we built for orchestration
        When I create multiple concurrent loops
        Then all loops are tracked and accessible
        Because we manage parallel work ourselves
        """
        # Given loop manager
        manager = CognitiveLoopManager()

        # When creating multiple loops
        loop1 = manager.create_loop("Build custom indexer")
        loop2 = manager.create_loop("Implement ranking ourselves")
        loop3 = manager.create_loop("Design query parser")

        # Then all tracked
        assert len(manager._loops) == 3
        assert manager.get_loop(loop1.id) == loop1
        assert manager.get_loop(loop2.id) == loop2

    def test_scenario_manager_identifies_stuck_loops(self):
        """
        Scenario: System detects loops stuck without progress

        Given loops managed by our orchestration system
        When a loop iterates excessively without progress
        Then manager identifies it as potentially stuck
        Because we built automated monitoring ourselves
        """
        # Given managed loops
        manager = CognitiveLoopManager()
        loop = manager.create_loop("Optimize custom algorithm")
        loop.start(LoopPhase.PRODUCE)

        # When iterating excessively
        for _ in range(4):
            loop.transition(LoopPhase.VERIFY, "Check")
            loop.transition(LoopPhase.PRODUCE, "Rework")

        # Then manager identifies stuck loops
        stuck = manager.get_stuck_loops(iteration_threshold=3)
        assert len(stuck) > 0
        assert loop in stuck

    def test_scenario_manager_tracks_blocked_dependencies(self):
        """
        Scenario: System tracks loops blocked on dependencies

        Given loops with dependencies we're managing
        When a loop becomes blocked
        Then manager identifies blockers needing resolution
        Because we coordinate dependencies ourselves
        """
        # Given loops with dependencies
        manager = CognitiveLoopManager()
        loop = manager.create_loop("Implement feature requiring custom library")
        loop.start()

        # When blocking occurs
        loop.block("Waiting for custom library implementation")

        # Then blockers are tracked
        blocked = manager.get_blocked_loops()
        assert len(blocked) == 1
        assert blocked[0].block_reason == "Waiting for custom library implementation"

    def test_scenario_manager_provides_aggregate_statistics(self):
        """
        Scenario: Manager reports on overall reasoning health

        Given multiple loops in our orchestration system
        When I query aggregate statistics
        Then summary shows distribution and health
        Because we monitor our system ourselves
        """
        # Given multiple loops in various states
        manager = CognitiveLoopManager()
        loop1 = manager.create_loop("Task 1")
        loop1.start()
        loop2 = manager.create_loop("Task 2")
        loop2.start()
        loop2.complete(TerminationReason.SUCCESS)

        # When querying statistics
        summary = manager.get_summary()

        # Then distribution is shown
        assert summary['total_loops'] == 2
        assert 'by_status' in summary
        assert summary['by_status']['ACTIVE'] >= 1
        assert summary['by_status']['COMPLETED'] >= 1

    def test_scenario_manager_handles_event_notifications(self):
        """
        Scenario: Manager broadcasts loop events for monitoring

        Given a manager with registered handlers we built
        When loop transitions occur
        Then handlers receive notifications
        Because we implement our own event system
        """
        # Given manager with handler
        manager = CognitiveLoopManager()
        events_received = []

        def transition_handler(loop, transition):
            events_received.append(('transition', loop.id, transition.to_phase))

        manager.register_transition_handler(transition_handler)

        # When loop transitions
        loop = manager.create_loop("Test events")
        loop.start(LoopPhase.QUESTION)
        loop.transition(LoopPhase.ANSWER, "Move to answer")

        # Then handler receives events
        assert len(events_received) >= 2  # start + transition
        assert any(e[2] == LoopPhase.QUESTION for e in events_received)
        assert any(e[2] == LoopPhase.ANSWER for e in events_received)


class DeveloperBuildsKnowledgeTransferSystem:
    """
    Epic: Developer Builds Knowledge Transfer System

    As a developer building handoff mechanisms,
    I want loop state to be serializable,
    So that I can persist and transfer reasoning context we control.
    """

    def test_scenario_loop_state_serializes_completely(self):
        """
        Scenario: Loop state can be serialized for persistence

        Given a loop with rich context we've built
        When I serialize the loop state
        Then all context is preserved in our format
        Because we implement serialization ourselves
        """
        # Given loop with rich context
        loop = CognitiveLoop(goal="Implement custom persistence layer")
        loop.start(LoopPhase.PRODUCE)
        loop.add_note("Using hand-built serialization")
        loop.current_context().record_decision("JSON format", "Simple and readable")

        # When serializing (using our built-in serializer)
        from cortical.reasoning.cognitive_loop import LoopStateSerializer
        serializer = LoopStateSerializer()
        serialized = serializer.serialize(loop)

        # Then state is preserved
        assert loop.id in serialized
        assert loop.goal in serialized
        assert "PRODUCE" in serialized

    def test_scenario_serialized_loops_can_be_restored(self):
        """
        Scenario: Persisted loops can be restored completely

        Given a serialized loop from our system
        When I deserialize it
        Then restored loop maintains full state
        Because we built complete state restoration
        """
        # Given serialized loop
        from cortical.reasoning.cognitive_loop import LoopStateSerializer

        original = CognitiveLoop(goal="Test serialization we built")
        original.start()
        original.add_note("Test note")

        serializer = LoopStateSerializer()
        serialized = serializer.serialize(original)

        # When deserializing
        restored = serializer.deserialize(serialized)

        # Then state is maintained
        assert restored.id == original.id
        assert restored.goal == original.goal
        assert restored.status == original.status
        assert len(restored.phase_contexts) == len(original.phase_contexts)
