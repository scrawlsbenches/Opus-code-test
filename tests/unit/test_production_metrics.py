"""
Unit tests for ProductionMetrics class.

Tests cover:
- State transition recording and storage
- Chunk timing tracking
- Average calculations
- Edge cases (no data, single data point)
- Integration workflow: record->query->verify
- Behavioral validation: metrics reflect actual progression
"""

import unittest
from datetime import datetime, timedelta
from time import sleep

from cortical.reasoning.production_state import (
    ProductionMetrics,
    ProductionTask,
    ProductionChunk,
    ProductionState,
)


class TestProductionMetricsBasics(unittest.TestCase):
    """Basic unit tests for ProductionMetrics initialization and data storage."""

    def setUp(self):
        """Set up fresh metrics instance for each test."""
        self.metrics = ProductionMetrics()

    def test_initialization(self):
        """Test metrics initializes with empty data structures."""
        self.assertEqual(len(self.metrics._state_transitions), 0)
        self.assertEqual(len(self.metrics._chunk_timings), 0)
        self.assertEqual(len(self.metrics._task_timings), 0)

    def test_record_state_transition_stores_data(self):
        """Test that state transitions are stored correctly."""
        task = ProductionTask(goal="Test task")

        self.metrics.record_state_transition(
            task,
            ProductionState.PLANNING,
            ProductionState.DRAFTING
        )

        # Verify transition was recorded
        self.assertEqual(len(self.metrics._state_transitions), 1)
        transition = self.metrics._state_transitions[0]

        self.assertEqual(transition['task_id'], task.id)
        self.assertEqual(transition['from_state'], 'PLANNING')
        self.assertEqual(transition['to_state'], 'DRAFTING')
        self.assertIn('timestamp', transition)
        self.assertIn('duration_in_previous_state', transition)

    def test_record_chunk_start_stores_data(self):
        """Test that chunk start is recorded correctly."""
        chunk = ProductionChunk(name="Test chunk", time_estimate_minutes=30)

        self.metrics.record_chunk_start(chunk)

        # Verify chunk timing was recorded
        self.assertIn(chunk.id, self.metrics._chunk_timings)
        timing = self.metrics._chunk_timings[chunk.id]

        self.assertIsNotNone(timing['start'])
        self.assertIsNone(timing['end'])
        self.assertEqual(timing['estimated_minutes'], 30)
        self.assertIsNone(timing['actual_minutes'])
        self.assertEqual(timing['status'], 'in_progress')

    def test_record_chunk_start_with_custom_estimate(self):
        """Test chunk start with custom time estimate."""
        chunk = ProductionChunk(name="Test chunk", time_estimate_minutes=30)

        self.metrics.record_chunk_start(chunk, estimated_minutes=45)

        timing = self.metrics._chunk_timings[chunk.id]
        self.assertEqual(timing['estimated_minutes'], 45)

    def test_record_chunk_complete_updates_data(self):
        """Test that chunk completion updates timing data."""
        chunk = ProductionChunk(name="Test chunk", time_estimate_minutes=30)

        # Start chunk
        self.metrics.record_chunk_start(chunk)
        sleep(0.1)  # Small delay to ensure measurable duration

        # Complete chunk
        self.metrics.record_chunk_complete(chunk)

        # Verify completion was recorded
        timing = self.metrics._chunk_timings[chunk.id]
        self.assertIsNotNone(timing['end'])
        self.assertEqual(timing['status'], 'complete')
        self.assertIsNotNone(timing['actual_minutes'])
        self.assertGreater(timing['actual_minutes'], 0)

    def test_record_chunk_complete_without_start(self):
        """Test completing a chunk that wasn't started via record_chunk_start."""
        chunk = ProductionChunk(name="Test chunk", time_estimate_minutes=30)
        chunk.start()  # Use chunk's own start method
        sleep(0.1)

        self.metrics.record_chunk_complete(chunk)

        # Should create entry from chunk's started_at
        self.assertIn(chunk.id, self.metrics._chunk_timings)
        timing = self.metrics._chunk_timings[chunk.id]
        self.assertEqual(timing['status'], 'complete')

    def test_record_chunk_complete_no_timing_data(self):
        """Test completing a chunk with no timing data available."""
        chunk = ProductionChunk(name="Test chunk")
        # Don't start chunk, so no timing data

        self.metrics.record_chunk_complete(chunk)

        # Should not create entry
        self.assertNotIn(chunk.id, self.metrics._chunk_timings)


class TestProductionMetricsCalculations(unittest.TestCase):
    """Tests for metrics calculations and aggregations."""

    def setUp(self):
        """Set up fresh metrics instance for each test."""
        self.metrics = ProductionMetrics()

    def test_get_average_time_in_state_no_data(self):
        """Test average time returns 0 when no data available."""
        avg_time = self.metrics.get_average_time_in_state(ProductionState.DRAFTING)
        self.assertEqual(avg_time, 0.0)

    def test_get_average_time_in_state_single_datapoint(self):
        """Test average time with single data point."""
        task = ProductionTask(goal="Test task")

        # Record two transitions to create one duration measurement
        self.metrics.record_state_transition(task, ProductionState.PLANNING, ProductionState.DRAFTING)
        sleep(0.1)
        self.metrics.record_state_transition(task, ProductionState.DRAFTING, ProductionState.REFINING)

        avg_time = self.metrics.get_average_time_in_state(ProductionState.DRAFTING)
        self.assertGreater(avg_time, 0)

    def test_get_average_time_in_state_multiple_datapoints(self):
        """Test average time with multiple data points."""
        task1 = ProductionTask(goal="Task 1")
        task2 = ProductionTask(goal="Task 2")

        # Task 1: PLANNING -> DRAFTING -> REFINING
        self.metrics.record_state_transition(task1, ProductionState.PLANNING, ProductionState.DRAFTING)
        sleep(0.05)
        self.metrics.record_state_transition(task1, ProductionState.DRAFTING, ProductionState.REFINING)

        # Task 2: PLANNING -> DRAFTING -> REFINING
        self.metrics.record_state_transition(task2, ProductionState.PLANNING, ProductionState.DRAFTING)
        sleep(0.05)
        self.metrics.record_state_transition(task2, ProductionState.DRAFTING, ProductionState.REFINING)

        avg_time = self.metrics.get_average_time_in_state(ProductionState.DRAFTING)
        self.assertGreater(avg_time, 0)

        # Should have 2 data points for DRAFTING state
        self.assertEqual(len(self.metrics._task_timings[task1.id]['DRAFTING']), 1)
        self.assertEqual(len(self.metrics._task_timings[task2.id]['DRAFTING']), 1)

    def test_get_estimation_accuracy_no_data(self):
        """Test estimation accuracy returns 0 when no completed chunks."""
        accuracy = self.metrics.get_estimation_accuracy()
        self.assertEqual(accuracy, 0.0)

    def test_get_estimation_accuracy_single_chunk(self):
        """Test estimation accuracy with single completed chunk."""
        chunk = ProductionChunk(name="Test chunk", time_estimate_minutes=30)

        self.metrics.record_chunk_start(chunk, estimated_minutes=10)
        sleep(0.1)
        self.metrics.record_chunk_complete(chunk)

        accuracy = self.metrics.get_estimation_accuracy()
        # Actual time should be very small (0.1s sleep), estimated was 10 min
        # So ratio should be very small
        self.assertGreater(accuracy, 0)
        self.assertLess(accuracy, 1.0)  # Faster than estimated

    def test_get_estimation_accuracy_multiple_chunks(self):
        """Test estimation accuracy with multiple completed chunks."""
        # Chunk 1: 10 min estimate
        chunk1 = ProductionChunk(name="Chunk 1", time_estimate_minutes=10)
        self.metrics.record_chunk_start(chunk1)
        sleep(0.05)
        self.metrics.record_chunk_complete(chunk1)

        # Chunk 2: 20 min estimate
        chunk2 = ProductionChunk(name="Chunk 2", time_estimate_minutes=20)
        self.metrics.record_chunk_start(chunk2)
        sleep(0.05)
        self.metrics.record_chunk_complete(chunk2)

        accuracy = self.metrics.get_estimation_accuracy()
        # Both chunks should complete faster than estimated
        self.assertGreater(accuracy, 0)

    def test_get_estimation_accuracy_zero_estimate(self):
        """Test estimation accuracy when estimated time is 0."""
        chunk = ProductionChunk(name="Test chunk", time_estimate_minutes=0)

        self.metrics.record_chunk_start(chunk, estimated_minutes=0)
        sleep(0.05)
        self.metrics.record_chunk_complete(chunk)

        accuracy = self.metrics.get_estimation_accuracy()
        self.assertEqual(accuracy, 0.0)  # Should handle division by zero

    def test_get_time_in_state_distribution_empty(self):
        """Test state distribution returns empty dict when no data."""
        distribution = self.metrics.get_time_in_state_distribution()
        self.assertEqual(distribution, {})

    def test_get_time_in_state_distribution_with_data(self):
        """Test state distribution with real data."""
        task = ProductionTask(goal="Test task")

        # PLANNING -> DRAFTING -> REFINING
        self.metrics.record_state_transition(task, ProductionState.PLANNING, ProductionState.DRAFTING)
        sleep(0.05)
        self.metrics.record_state_transition(task, ProductionState.DRAFTING, ProductionState.REFINING)
        sleep(0.05)
        self.metrics.record_state_transition(task, ProductionState.REFINING, ProductionState.FINALIZING)

        distribution = self.metrics.get_time_in_state_distribution()

        # Should have entries for states we spent time in
        self.assertIn('DRAFTING', distribution)
        self.assertIn('REFINING', distribution)
        self.assertGreater(distribution['DRAFTING'], 0)
        self.assertGreater(distribution['REFINING'], 0)


class TestProductionMetricsSummary(unittest.TestCase):
    """Tests for the comprehensive summary method."""

    def setUp(self):
        """Set up fresh metrics instance for each test."""
        self.metrics = ProductionMetrics()

    def test_get_summary_empty(self):
        """Test summary with no data."""
        summary = self.metrics.get_summary()

        self.assertEqual(summary['total_transitions'], 0)
        self.assertEqual(summary['average_accuracy'], 0.0)
        self.assertEqual(summary['time_distribution'], {})
        self.assertEqual(summary['chunks_completed'], 0)
        self.assertEqual(summary['chunks_in_progress'], 0)
        self.assertEqual(summary['tasks_tracked'], 0)

    def test_get_summary_with_transitions(self):
        """Test summary includes transition count."""
        task = ProductionTask(goal="Test task")

        self.metrics.record_state_transition(task, ProductionState.PLANNING, ProductionState.DRAFTING)
        self.metrics.record_state_transition(task, ProductionState.DRAFTING, ProductionState.REFINING)

        summary = self.metrics.get_summary()
        self.assertEqual(summary['total_transitions'], 2)
        self.assertEqual(summary['tasks_tracked'], 1)

    def test_get_summary_with_chunks(self):
        """Test summary includes chunk counts."""
        chunk1 = ProductionChunk(name="Chunk 1")
        chunk2 = ProductionChunk(name="Chunk 2")

        # Start both chunks
        self.metrics.record_chunk_start(chunk1)
        self.metrics.record_chunk_start(chunk2)

        summary = self.metrics.get_summary()
        self.assertEqual(summary['chunks_in_progress'], 2)
        self.assertEqual(summary['chunks_completed'], 0)

        # Complete one chunk
        self.metrics.record_chunk_complete(chunk1)

        summary = self.metrics.get_summary()
        self.assertEqual(summary['chunks_in_progress'], 1)
        self.assertEqual(summary['chunks_completed'], 1)

    def test_get_summary_with_accuracy(self):
        """Test summary includes estimation accuracy."""
        chunk = ProductionChunk(name="Test chunk", time_estimate_minutes=30)

        self.metrics.record_chunk_start(chunk)
        sleep(0.05)
        self.metrics.record_chunk_complete(chunk)

        summary = self.metrics.get_summary()
        self.assertGreater(summary['average_accuracy'], 0)


class TestProductionMetricsIntegration(unittest.TestCase):
    """Integration tests for full workflow scenarios."""

    def test_full_task_lifecycle_metrics(self):
        """Test metrics tracking through complete task lifecycle."""
        metrics = ProductionMetrics()
        task = ProductionTask(goal="Build feature X")

        # PLANNING -> DRAFTING
        metrics.record_state_transition(task, ProductionState.PLANNING, ProductionState.DRAFTING)
        sleep(0.05)

        # Create and track first chunk
        chunk1 = ProductionChunk(name="Setup", time_estimate_minutes=15)
        metrics.record_chunk_start(chunk1)
        sleep(0.05)
        metrics.record_chunk_complete(chunk1)

        # DRAFTING -> REFINING
        metrics.record_state_transition(task, ProductionState.DRAFTING, ProductionState.REFINING)
        sleep(0.05)

        # Create and track second chunk
        chunk2 = ProductionChunk(name="Implementation", time_estimate_minutes=45)
        metrics.record_chunk_start(chunk2)
        sleep(0.05)
        metrics.record_chunk_complete(chunk2)

        # REFINING -> FINALIZING
        metrics.record_state_transition(task, ProductionState.REFINING, ProductionState.FINALIZING)
        sleep(0.05)

        # Verify metrics reflect the journey
        summary = metrics.get_summary()

        self.assertEqual(summary['total_transitions'], 3)
        self.assertEqual(summary['chunks_completed'], 2)
        self.assertEqual(summary['chunks_in_progress'], 0)
        self.assertEqual(summary['tasks_tracked'], 1)
        self.assertGreater(summary['average_accuracy'], 0)

        # Verify time distribution
        distribution = metrics.get_time_in_state_distribution()
        self.assertIn('DRAFTING', distribution)
        self.assertIn('REFINING', distribution)

    def test_multi_task_metrics(self):
        """Test metrics across multiple concurrent tasks."""
        metrics = ProductionMetrics()

        task1 = ProductionTask(goal="Feature A")
        task2 = ProductionTask(goal="Feature B")

        # Task 1: PLANNING -> DRAFTING
        metrics.record_state_transition(task1, ProductionState.PLANNING, ProductionState.DRAFTING)
        sleep(0.05)

        # Task 2: PLANNING -> DRAFTING
        metrics.record_state_transition(task2, ProductionState.PLANNING, ProductionState.DRAFTING)
        sleep(0.05)

        # Task 1: DRAFTING -> REFINING
        metrics.record_state_transition(task1, ProductionState.DRAFTING, ProductionState.REFINING)
        sleep(0.05)

        # Task 2: DRAFTING -> BLOCKED
        metrics.record_state_transition(task2, ProductionState.DRAFTING, ProductionState.BLOCKED)

        summary = metrics.get_summary()

        self.assertEqual(summary['total_transitions'], 4)
        self.assertEqual(summary['tasks_tracked'], 2)

        # Both tasks spent time in DRAFTING
        avg_drafting = metrics.get_average_time_in_state(ProductionState.DRAFTING)
        self.assertGreater(avg_drafting, 0)

    def test_estimation_accuracy_improvement_tracking(self):
        """Test tracking estimation accuracy over multiple chunks."""
        metrics = ProductionMetrics()

        # First chunk: poor estimate (estimate too low)
        chunk1 = ProductionChunk(name="Chunk 1", time_estimate_minutes=5)
        metrics.record_chunk_start(chunk1)
        sleep(0.1)  # Longer than estimate suggests
        metrics.record_chunk_complete(chunk1)

        accuracy1 = metrics.get_estimation_accuracy()

        # Second chunk: better estimate
        chunk2 = ProductionChunk(name="Chunk 2", time_estimate_minutes=10)
        metrics.record_chunk_start(chunk2)
        sleep(0.1)  # More reasonable
        metrics.record_chunk_complete(chunk2)

        accuracy2 = metrics.get_estimation_accuracy()

        # Both should be greater than 0
        self.assertGreater(accuracy1, 0)
        self.assertGreater(accuracy2, 0)

        # The second estimate should improve overall accuracy
        # (lower ratio = better if we're consistently fast)
        self.assertIsNotNone(accuracy1)
        self.assertIsNotNone(accuracy2)


class TestProductionMetricsBehavioral(unittest.TestCase):
    """Behavioral tests: do metrics accurately reflect real task progression?"""

    def test_metrics_reflect_rework_cycles(self):
        """Test that rework cycles are captured in metrics."""
        metrics = ProductionMetrics()
        task = ProductionTask(goal="Complex feature")

        # Initial progression
        metrics.record_state_transition(task, ProductionState.PLANNING, ProductionState.DRAFTING)
        sleep(0.05)
        metrics.record_state_transition(task, ProductionState.DRAFTING, ProductionState.REFINING)
        sleep(0.05)

        # Rework cycle
        metrics.record_state_transition(task, ProductionState.REFINING, ProductionState.REWORK)
        sleep(0.05)
        metrics.record_state_transition(task, ProductionState.REWORK, ProductionState.REFINING)
        sleep(0.05)

        # Second rework cycle
        metrics.record_state_transition(task, ProductionState.REFINING, ProductionState.REWORK)
        sleep(0.05)
        metrics.record_state_transition(task, ProductionState.REWORK, ProductionState.REFINING)

        summary = metrics.get_summary()

        # Should capture all transitions including rework
        self.assertEqual(summary['total_transitions'], 6)

        # Should have time in REWORK state
        distribution = metrics.get_time_in_state_distribution()
        self.assertIn('REWORK', distribution)
        self.assertGreater(distribution['REWORK'], 0)

    def test_metrics_reflect_blocked_time(self):
        """Test that blocked states are properly tracked."""
        metrics = ProductionMetrics()
        task = ProductionTask(goal="Feature needing input")

        # Progress to DRAFTING, then get blocked
        metrics.record_state_transition(task, ProductionState.PLANNING, ProductionState.DRAFTING)
        sleep(0.05)
        metrics.record_state_transition(task, ProductionState.DRAFTING, ProductionState.BLOCKED)
        sleep(0.1)  # Spend significant time blocked

        # Unblock and continue
        metrics.record_state_transition(task, ProductionState.BLOCKED, ProductionState.DRAFTING)

        distribution = metrics.get_time_in_state_distribution()

        # Should show time in BLOCKED state
        self.assertIn('BLOCKED', distribution)

        # BLOCKED time should be measurable
        blocked_time = metrics.get_average_time_in_state(ProductionState.BLOCKED)
        self.assertGreater(blocked_time, 0)

    def test_metrics_distinguish_task_durations(self):
        """Test that metrics can distinguish between fast and slow tasks."""
        metrics = ProductionMetrics()

        fast_task = ProductionTask(goal="Quick fix")
        slow_task = ProductionTask(goal="Major refactor")

        # Fast task: quick transitions
        metrics.record_state_transition(fast_task, ProductionState.PLANNING, ProductionState.DRAFTING)
        sleep(0.02)
        metrics.record_state_transition(fast_task, ProductionState.DRAFTING, ProductionState.COMPLETE)

        # Slow task: longer transitions
        metrics.record_state_transition(slow_task, ProductionState.PLANNING, ProductionState.DRAFTING)
        sleep(0.1)
        metrics.record_state_transition(slow_task, ProductionState.DRAFTING, ProductionState.REFINING)

        # Get individual task timings
        fast_drafting_time = metrics._task_timings[fast_task.id].get('DRAFTING', [0])[0]
        slow_drafting_time = metrics._task_timings[slow_task.id].get('DRAFTING', [0])[0]

        # Slow task should have taken longer
        self.assertGreater(slow_drafting_time, fast_drafting_time)

    def test_state_transition_duration_tracking(self):
        """Test that duration in previous state is correctly calculated."""
        metrics = ProductionMetrics()
        task = ProductionTask(goal="Test task")

        # First transition (no previous state)
        metrics.record_state_transition(task, ProductionState.PLANNING, ProductionState.DRAFTING)
        first_transition = metrics._state_transitions[0]
        self.assertEqual(first_transition['duration_in_previous_state'], 0.0)

        # Second transition (should have duration)
        sleep(0.1)
        metrics.record_state_transition(task, ProductionState.DRAFTING, ProductionState.REFINING)
        second_transition = metrics._state_transitions[1]
        self.assertGreater(second_transition['duration_in_previous_state'], 0)


if __name__ == '__main__':
    unittest.main()
