"""
Unit tests for the CognitiveMemory class.

Tests the enhanced memory system including:
- Episodic memory (observations)
- Working memory (intentions with pending tracking)
- Meta-cognition (learnings, reflections)
- Concept indexing for fast lookups
- Associative recall
- Temporal queries
- Context window generation
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from examples.cognitive_memory_demo import CognitiveMemory


class TestCognitiveMemoryBasics:
    """Test basic memory operations."""

    def test_init_creates_session(self):
        """Memory should create a unique session ID."""
        memory = CognitiveMemory()
        assert memory._session_id.startswith("session-")

    def test_init_with_custom_session(self):
        """Memory should accept custom session ID."""
        memory = CognitiveMemory(session_id="test-session")
        assert memory._session_id == "test-session"

    def test_current_horizon_initially_none(self):
        """Current horizon should be None before any events."""
        memory = CognitiveMemory()
        assert memory.current_horizon() is None

    def test_current_horizon_after_event(self):
        """Current horizon should return the last event ID."""
        memory = CognitiveMemory()
        event_id = memory.observe("test observation")
        assert memory.current_horizon() == event_id


class TestEpisodicMemory:
    """Test observation (episodic memory) operations."""

    def test_observe_returns_event_id(self):
        """Observe should return an event ID."""
        memory = CognitiveMemory()
        event_id = memory.observe("something happened")
        assert event_id is not None
        assert len(event_id) > 0

    def test_observe_with_details(self):
        """Observe should include additional details."""
        memory = CognitiveMemory()
        memory.observe("file read", {"path": "/test.py", "lines": 100})
        observations = memory.recall_observations()
        assert len(observations) == 1
        assert observations[0]['content']['path'] == "/test.py"

    def test_observe_user_request(self):
        """User request should be recorded as observation."""
        memory = CognitiveMemory()
        memory.observe_user_request("fix the bug")
        observations = memory.recall_observations()
        assert len(observations) == 1
        assert observations[0]['content']['request'] == "fix the bug"

    def test_observe_error(self):
        """Errors should be recorded and recallable."""
        memory = CognitiveMemory()
        memory.observe_error("TypeError: NoneType", context="test.py:42")
        errors = memory.recall_errors()
        assert len(errors) == 1
        assert "TypeError" in errors[0]['error']
        assert errors[0]['context'] == "test.py:42"

    def test_observe_file_change(self):
        """File changes should be recorded."""
        memory = CognitiveMemory()
        memory.observe_file_change("/path/to/file.py", "modified")
        observations = memory.recall_observations()
        assert len(observations) == 1
        assert observations[0]['content']['path'] == "/path/to/file.py"


class TestWorkingMemory:
    """Test intention (working memory) operations."""

    def test_intend_creates_pending_intention(self):
        """Intend should create a pending intention."""
        memory = CognitiveMemory()
        task_id = memory.intend("fix bug")
        pending = memory.pending_intentions()
        assert len(pending) == 1
        assert pending[0]['id'] == task_id
        assert pending[0]['goal'] == "fix bug"

    def test_complete_intention_removes_from_pending(self):
        """Completing an intention should remove it from pending."""
        memory = CognitiveMemory()
        task_id = memory.intend("fix bug")
        assert len(memory.pending_intentions()) == 1

        memory.complete_intention(task_id, "bug fixed")
        assert len(memory.pending_intentions()) == 0

    def test_abandon_intention_removes_from_pending(self):
        """Abandoning an intention should remove it from pending."""
        memory = CognitiveMemory()
        task_id = memory.intend("fix bug")
        assert len(memory.pending_intentions()) == 1

        memory.abandon_intention(task_id, "no longer needed")
        assert len(memory.pending_intentions()) == 0

    def test_multiple_intentions_tracked(self):
        """Multiple intentions should be tracked separately."""
        memory = CognitiveMemory()
        id1 = memory.intend("task 1", priority="high")
        id2 = memory.intend("task 2", priority="low")
        id3 = memory.intend("task 3", priority="medium")

        pending = memory.pending_intentions()
        assert len(pending) == 3

        memory.complete_intention(id2, "done")
        pending = memory.pending_intentions()
        assert len(pending) == 2
        assert all(p['id'] != id2 for p in pending)


class TestMetaCognition:
    """Test meta-cognition operations."""

    def test_reflect_creates_metacognition_event(self):
        """Reflect should create a metacognition event."""
        memory = CognitiveMemory()
        memory.reflect("I understand the problem now", category="insight")
        stats = memory.stats
        assert stats['by_type'].get('METACOGNITION', 0) == 1

    def test_learn_records_problem_solution_pair(self):
        """Learn should record problem-solution pairs."""
        memory = CognitiveMemory()
        memory.learn("null pointer error", "add null check")
        learnings = memory.recall_learnings()
        assert len(learnings) == 1
        assert learnings[0]['problem'] == "null pointer error"
        assert learnings[0]['solution'] == "add null check"

    def test_note_confusion_records_confusion(self):
        """Note confusion should record areas of confusion."""
        memory = CognitiveMemory()
        memory.note_confusion("how does authentication work?")
        stats = memory.stats
        assert stats['by_type'].get('METACOGNITION', 0) == 1


class TestConceptIndexing:
    """Test concept indexing for fast lookups."""

    def test_concepts_extracted_from_observation(self):
        """Concepts should be extracted from observation text."""
        memory = CognitiveMemory()
        memory.observe("examining authentication module")
        # Should have indexed 'examining', 'authentication', 'module'
        assert len(memory._concept_index) >= 2

    def test_concept_index_enables_fast_recall(self):
        """Concept index should enable fast concept-based recall."""
        memory = CognitiveMemory()
        memory.observe("authentication bug found")
        memory.observe("database connection issue")
        memory.observe("authentication fix applied")

        # Recall by concept should use index
        auth_memories = memory.recall_observations(concept="authentication")
        assert len(auth_memories) == 2  # Only auth-related

    def test_recall_by_multiple_concepts(self):
        """Should find memories matching any of multiple concepts."""
        memory = CognitiveMemory()
        memory.observe("authentication bug")
        memory.observe("database error")
        memory.observe("authentication fixed")

        memories = memory.recall_by_concepts(['authentication', 'database'])
        assert len(memories) == 3


class TestAssociativeRecall:
    """Test associative recall by shared concepts."""

    def test_find_related_returns_related_memories(self):
        """Should find memories related by shared concepts."""
        memory = CognitiveMemory()
        id1 = memory.observe("authentication module has a bug")
        memory.observe("database connection works")
        memory.observe("authentication fix needed")

        related = memory.find_related(id1)
        # Should find the other authentication memory
        assert len(related) >= 1
        assert any('authentication' in r.get('shared_concepts', []) for r in related)

    def test_find_related_excludes_source_event(self):
        """Source event should not be in related results."""
        memory = CognitiveMemory()
        id1 = memory.observe("test observation")
        related = memory.find_related(id1)
        assert all(r['id'] != id1[:12] for r in related)

    def test_find_related_respects_limit(self):
        """Should respect the limit parameter."""
        memory = CognitiveMemory()
        id1 = memory.observe("authentication module")
        for i in range(10):
            memory.observe(f"authentication issue {i}")

        related = memory.find_related(id1, limit=3)
        assert len(related) <= 3


class TestTemporalQueries:
    """Test temporal (time-based) queries."""

    def test_state_at_returns_counts_up_to_horizon(self):
        """State at horizon should only count events up to that point."""
        memory = CognitiveMemory()

        memory.observe("observation 1")
        memory.observe("observation 2")
        horizon = memory.current_horizon()
        memory.observe("observation 3")
        memory.intend("task 1")

        state = memory.state_at(horizon)
        assert state['observations'] == 2  # Only first two
        assert state['intentions'] == 0  # Intention came after


class TestContextWindow:
    """Test context window generation."""

    def test_context_window_returns_limited_results(self):
        """Context window should respect limit."""
        memory = CognitiveMemory()
        for i in range(20):
            memory.observe(f"observation {i}")

        context = memory.context_window(limit=5)
        assert len(context) <= 5

    def test_context_window_filters_by_concepts(self):
        """Context window should filter by concepts when provided."""
        memory = CognitiveMemory()
        memory.observe("authentication bug")
        memory.observe("database issue")
        memory.observe("authentication fix")

        context = memory.context_window(concepts=['authentication'], limit=10)
        assert len(context) == 2


class TestImportanceScoring:
    """Test importance scoring for memories."""

    def test_high_priority_intentions_have_higher_importance(self):
        """High priority intentions should have higher importance scores."""
        memory = CognitiveMemory()
        id_high = memory.intend("critical task", priority="high")
        id_low = memory.intend("minor task", priority="low")

        assert memory._importance_scores[id_high] > memory._importance_scores[id_low]

    def test_completions_have_elevated_importance(self):
        """Fulfillment events should have elevated importance."""
        memory = CognitiveMemory()
        task_id = memory.intend("task")
        completion_id = memory.complete_intention(task_id, "done")

        assert memory._importance_scores[completion_id] >= 1.5


class TestStats:
    """Test statistics reporting."""

    def test_stats_includes_event_counts(self):
        """Stats should include counts by event type."""
        memory = CognitiveMemory()
        memory.observe("obs 1")
        memory.observe("obs 2")
        memory.intend("task 1")
        memory.reflect("insight")

        stats = memory.stats
        assert stats['total_events'] == 4
        assert stats['by_type']['OBSERVATION'] == 2
        assert stats['by_type']['INTENTION'] == 1
        assert stats['by_type']['METACOGNITION'] == 1

    def test_stats_includes_pending_count(self):
        """Stats should include pending intentions count."""
        memory = CognitiveMemory()
        memory.intend("task 1")
        memory.intend("task 2")

        stats = memory.stats
        assert stats['pending_intentions'] == 2

    def test_stats_includes_concept_count(self):
        """Stats should include indexed concept count."""
        memory = CognitiveMemory()
        memory.observe("authentication bug")
        memory.observe("database issue")

        stats = memory.stats
        assert stats['indexed_concepts'] >= 2


class TestSessionSummarization:
    """Test session summarization (compaction)."""

    def test_summarize_session_creates_compaction_event(self):
        """Summarize should create a compaction event."""
        memory = CognitiveMemory()
        memory.observe("did some work")
        memory.intend("task")

        memory.summarize_session("completed task successfully")
        stats = memory.stats
        assert stats['by_type'].get('COMPACTION', 0) == 1


class TestCausalChaining:
    """Test that events are properly chained causally."""

    def test_events_linked_causally(self):
        """Each event should be causally linked to the previous one."""
        memory = CognitiveMemory()
        id1 = memory.observe("first")
        id2 = memory.observe("second")
        id3 = memory.observe("third")

        # Verify causal chain through the store
        event2 = memory._store.get(id2)
        event3 = memory._store.get(id3)

        assert id1 in event2.causal_parents
        assert id2 in event3.causal_parents
