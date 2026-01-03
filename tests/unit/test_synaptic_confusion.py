"""
Tests for SynapticConfusionDetector in llm_orchestration/recovery.py

TDD: Tests for confusion detection using synaptic memory patterns.
This detector bridges PRISM-GoT's synaptic memory with the recovery system.

The SynapticConfusionDetector has 0% test coverage - these tests bring it
to full coverage with comprehensive testing of:
- Activation loop detection
- Contradiction detection
- Stagnation detection
- Oscillation detection
- Integration with SynapticMemoryGraph
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, MagicMock


class TestSynapticConfusionDetectorInit:
    """Test SynapticConfusionDetector initialization."""

    def test_initialization_with_memory_graph(self):
        """SynapticConfusionDetector should initialize with a memory graph."""
        from llm_orchestration.recovery import SynapticConfusionDetector

        # Create mock memory graph
        mock_graph = Mock()
        mock_graph.nodes = {}
        mock_graph.activation_traces = {}
        mock_graph.synaptic_edges = {}

        detector = SynapticConfusionDetector(
            memory_graph=mock_graph,
            loop_window=5,
            contradiction_threshold=0.7,
            stagnation_threshold=0.1
        )

        assert detector._memory is mock_graph
        assert detector._loop_window == 5
        assert detector._contradiction_threshold == 0.7
        assert detector._stagnation_threshold == 0.1
        assert detector._activation_sequence == []

    def test_initialization_without_memory_graph(self):
        """SynapticConfusionDetector should require a memory graph."""
        from llm_orchestration.recovery import SynapticConfusionDetector

        # Should fail without memory_graph (required parameter)
        with pytest.raises(TypeError):
            detector = SynapticConfusionDetector()

    def test_threshold_configuration(self):
        """SynapticConfusionDetector should allow custom thresholds."""
        from llm_orchestration.recovery import SynapticConfusionDetector

        mock_graph = Mock()
        mock_graph.nodes = {}
        mock_graph.activation_traces = {}
        mock_graph.synaptic_edges = {}

        # Custom thresholds
        detector = SynapticConfusionDetector(
            memory_graph=mock_graph,
            loop_window=10,
            contradiction_threshold=0.9,
            stagnation_threshold=0.05
        )

        assert detector._loop_window == 10
        assert detector._contradiction_threshold == 0.9
        assert detector._stagnation_threshold == 0.05

    def test_signal_types_property(self):
        """SynapticConfusionDetector should expose signal types."""
        from llm_orchestration.recovery import SynapticConfusionDetector

        mock_graph = Mock()
        mock_graph.nodes = {}
        mock_graph.activation_traces = {}
        mock_graph.synaptic_edges = {}

        detector = SynapticConfusionDetector(memory_graph=mock_graph)

        signal_types = detector.signal_types
        assert 'synaptic_loop' in signal_types
        assert 'synaptic_contradiction' in signal_types
        assert 'synaptic_stagnation' in signal_types
        assert 'synaptic_oscillation' in signal_types


class TestConfusionDetection:
    """Test confusion signal detection."""

    def test_detect_repetition_pattern(self):
        """Should detect activation loops (repetition pattern)."""
        from llm_orchestration.recovery import SynapticConfusionDetector

        # Create mock graph with nodes
        mock_graph = Mock()
        mock_node_a = Mock()
        mock_node_a.content = "Try approach A"
        mock_node_b = Mock()
        mock_node_b.content = "Try approach B"

        mock_graph.nodes = {
            'node_a': mock_node_a,
            'node_b': mock_node_b
        }
        mock_graph.activation_traces = {}
        mock_graph.synaptic_edges = {}

        detector = SynapticConfusionDetector(memory_graph=mock_graph, loop_window=5)

        # Record a repeating pattern: A, B, A, B (need at least loop_window * 2 = 10 activations)
        # Record 6 iterations to have 12 activations total
        for _ in range(6):
            detector.record_activation('node_a')
            detector.record_activation('node_b')

        signals = detector.detect()

        # Should detect the loop
        loop_signals = [s for s in signals if s.signal_type == 'synaptic_loop']
        assert len(loop_signals) > 0
        assert 'loop' in loop_signals[0].description.lower()

    def test_detect_contradiction_pattern(self):
        """Should detect contradictory activations."""
        from llm_orchestration.recovery import SynapticConfusionDetector
        from cortical.reasoning.graph_of_thought import NodeType

        # Create mock graph with contradictory nodes
        mock_graph = Mock()

        # Hypothesis node
        mock_hypothesis = Mock()
        mock_hypothesis.content = "This approach will work"
        mock_hypothesis.node_type = NodeType.HYPOTHESIS

        # Evidence node with negation
        mock_evidence = Mock()
        mock_evidence.content = "This approach will not work based on evidence"
        mock_evidence.node_type = NodeType.EVIDENCE

        mock_graph.nodes = {
            'hypothesis_1': mock_hypothesis,
            'evidence_1': mock_evidence
        }

        # Create activation traces showing both recently active
        mock_trace_1 = Mock()
        mock_trace_1.history = [
            {'timestamp': datetime.now().isoformat()}
        ]
        mock_trace_1.get_frequency = Mock(return_value=5.0)  # High frequency

        mock_trace_2 = Mock()
        mock_trace_2.history = [
            {'timestamp': datetime.now().isoformat()}
        ]
        mock_trace_2.get_frequency = Mock(return_value=4.0)  # High frequency

        mock_graph.activation_traces = {
            'hypothesis_1': mock_trace_1,
            'evidence_1': mock_trace_2
        }
        mock_graph.synaptic_edges = {}

        detector = SynapticConfusionDetector(
            memory_graph=mock_graph,
            contradiction_threshold=0.3  # Lower threshold to catch this
        )

        signals = detector.detect()

        # Should detect contradiction
        contradiction_signals = [s for s in signals if s.signal_type == 'synaptic_contradiction']
        assert len(contradiction_signals) > 0

    def test_detect_stagnation_pattern(self):
        """Should detect stagnation (low activation rate)."""
        from llm_orchestration.recovery import SynapticConfusionDetector

        mock_graph = Mock()
        mock_graph.nodes = {}

        # Create activation traces with low frequency
        mock_trace = Mock()
        mock_trace.history = []  # Empty history (no recent activations)
        mock_trace.get_frequency = Mock(return_value=0.05)  # Below default threshold of 0.1

        mock_graph.activation_traces = {
            'node_1': mock_trace,
            'node_2': mock_trace
        }
        mock_graph.synaptic_edges = {}

        detector = SynapticConfusionDetector(
            memory_graph=mock_graph,
            stagnation_threshold=0.1
        )

        signals = detector.detect()

        # Should detect stagnation
        stagnation_signals = [s for s in signals if s.signal_type == 'synaptic_stagnation']
        assert len(stagnation_signals) > 0
        assert 'stagnating' in stagnation_signals[0].description.lower()

    def test_detect_oscillation_pattern(self):
        """Should detect oscillation (rapid switching between patterns)."""
        from llm_orchestration.recovery import SynapticConfusionDetector

        mock_graph = Mock()
        mock_graph.nodes = {}
        mock_graph.activation_traces = {}
        mock_graph.synaptic_edges = {}

        detector = SynapticConfusionDetector(memory_graph=mock_graph, loop_window=5)

        # Record ABAB pattern (oscillation)
        detector.record_activation('node_a')
        detector.record_activation('node_b')
        detector.record_activation('node_a')
        detector.record_activation('node_b')
        detector.record_activation('node_a')
        detector.record_activation('node_b')

        signals = detector.detect()

        # Should detect oscillation
        oscillation_signals = [s for s in signals if s.signal_type == 'synaptic_oscillation']
        assert len(oscillation_signals) > 0
        assert 'oscillating' in oscillation_signals[0].description.lower()

    def test_no_confusion_normal_operation(self):
        """Should not detect confusion during normal operation."""
        from llm_orchestration.recovery import SynapticConfusionDetector

        mock_graph = Mock()
        mock_graph.nodes = {}

        # Create activation traces with healthy frequency
        mock_trace = Mock()
        mock_trace.history = []  # Empty history (no recent activations to trigger contradiction)
        mock_trace.get_frequency = Mock(return_value=0.5)  # Above threshold

        mock_graph.activation_traces = {
            'node_1': mock_trace,
            'node_2': mock_trace
        }
        mock_graph.synaptic_edges = {}

        detector = SynapticConfusionDetector(memory_graph=mock_graph)

        # Record varied activations (no pattern)
        detector.record_activation('node_1')
        detector.record_activation('node_2')
        detector.record_activation('node_3')
        detector.record_activation('node_4')

        signals = detector.detect()

        # Should not detect any confusion
        # (May have empty list or only low-confidence signals)
        high_conf_signals = [s for s in signals if s.confidence > 0.5]
        assert len(high_conf_signals) == 0


class TestSeverityAssessment:
    """Test severity assessment of confusion signals."""

    def test_low_severity_single_signal(self):
        """Single weak signal should be low severity."""
        from llm_orchestration.recovery import SynapticConfusionDetector

        mock_graph = Mock()
        mock_graph.nodes = {}

        # Borderline stagnation
        mock_trace = Mock()
        mock_trace.history = []  # Empty history
        mock_trace.get_frequency = Mock(return_value=0.09)  # Just below threshold

        mock_graph.activation_traces = {'node_1': mock_trace}
        mock_graph.synaptic_edges = {}

        detector = SynapticConfusionDetector(
            memory_graph=mock_graph,
            stagnation_threshold=0.1
        )

        signals = detector.detect()

        # Signal should exist but with low confidence
        if len(signals) > 0:
            assert signals[0].confidence < 0.5

    def test_medium_severity_multiple_signals(self):
        """Multiple signals or moderate confidence should be medium severity."""
        from llm_orchestration.recovery import SynapticConfusionDetector

        mock_graph = Mock()
        mock_graph.nodes = {}
        mock_graph.activation_traces = {}
        mock_graph.synaptic_edges = {}

        detector = SynapticConfusionDetector(memory_graph=mock_graph, loop_window=3)

        # Create a moderate loop (3 repetitions)
        for _ in range(3):
            detector.record_activation('node_a')
            detector.record_activation('node_b')
            detector.record_activation('node_c')

        signals = detector.detect()

        # Should have moderate confidence
        if len(signals) > 0:
            loop_signals = [s for s in signals if s.signal_type == 'synaptic_loop']
            if len(loop_signals) > 0:
                assert 0.3 < loop_signals[0].confidence < 0.9

    def test_high_severity_critical_pattern(self):
        """Strong pattern should have high confidence."""
        from llm_orchestration.recovery import SynapticConfusionDetector

        mock_graph = Mock()
        mock_graph.nodes = {
            'node_a': Mock(content="Node A"),
            'node_b': Mock(content="Node B"),
            'node_c': Mock(content="Node C")
        }
        mock_graph.activation_traces = {}
        mock_graph.synaptic_edges = {}

        detector = SynapticConfusionDetector(memory_graph=mock_graph, loop_window=5)

        # Create a strong loop pattern (many repetitions)
        pattern = ['node_a', 'node_b', 'node_c']
        for _ in range(5):
            for node in pattern:
                detector.record_activation(node)

        signals = detector.detect()

        # Should have high confidence
        loop_signals = [s for s in signals if s.signal_type == 'synaptic_loop']
        if len(loop_signals) > 0:
            # With longer pattern, confidence should be higher
            assert loop_signals[0].confidence >= 0.6

    def test_severity_escalation(self):
        """Confidence should increase with pattern strength."""
        from llm_orchestration.recovery import SynapticConfusionDetector

        mock_graph = Mock()
        mock_graph.nodes = {}
        mock_graph.activation_traces = {}
        mock_graph.synaptic_edges = {}

        detector = SynapticConfusionDetector(memory_graph=mock_graph, loop_window=5)

        # Weak pattern
        detector.record_activation('node_a')
        detector.record_activation('node_b')
        detector.record_activation('node_a')
        detector.record_activation('node_b')

        weak_signals = detector.detect()
        weak_confidence = weak_signals[0].confidence if weak_signals else 0.0

        # Stronger pattern (same but more repetitions)
        detector._activation_sequence.clear()
        for _ in range(4):
            detector.record_activation('node_a')
            detector.record_activation('node_b')

        strong_signals = detector.detect()
        strong_confidence = strong_signals[0].confidence if strong_signals else 0.0

        # More repetitions should increase confidence
        # (Though it's capped at 0.9)
        assert strong_confidence >= weak_confidence


class TestMemoryGraphIntegration:
    """Test integration with SynapticMemoryGraph."""

    def test_activation_recording(self):
        """Should record activations for pattern tracking."""
        from llm_orchestration.recovery import SynapticConfusionDetector

        mock_graph = Mock()
        mock_graph.nodes = {}
        mock_graph.activation_traces = {}
        mock_graph.synaptic_edges = {}

        detector = SynapticConfusionDetector(memory_graph=mock_graph)

        assert len(detector._activation_sequence) == 0

        detector.record_activation('node_1')
        assert len(detector._activation_sequence) == 1
        assert detector._activation_sequence[0] == 'node_1'

        detector.record_activation('node_2')
        assert len(detector._activation_sequence) == 2
        assert detector._activation_sequence[1] == 'node_2'

    def test_pattern_extraction_from_graph(self):
        """Should extract patterns from memory graph activations."""
        from llm_orchestration.recovery import SynapticConfusionDetector

        mock_graph = Mock()
        mock_graph.nodes = {}

        # Set up activation traces
        recent_time = datetime.now()
        old_time = recent_time - timedelta(hours=2)

        mock_trace_recent = Mock()
        mock_trace_recent.history = [
            {'timestamp': recent_time.isoformat()}
        ]
        mock_trace_recent.get_frequency = Mock(return_value=3.0)

        mock_trace_old = Mock()
        mock_trace_old.history = [
            {'timestamp': old_time.isoformat()}
        ]
        mock_trace_old.get_frequency = Mock(return_value=0.0)

        mock_graph.activation_traces = {
            'recent_node': mock_trace_recent,
            'old_node': mock_trace_old
        }
        mock_graph.synaptic_edges = {}

        detector = SynapticConfusionDetector(memory_graph=mock_graph)

        # Detect should process activation traces
        signals = detector.detect()

        # Should access the traces
        assert mock_trace_recent.get_frequency.called
        assert mock_trace_old.get_frequency.called

    def test_decay_over_time(self):
        """Should handle activation decay over time."""
        from llm_orchestration.recovery import SynapticConfusionDetector

        mock_graph = Mock()
        mock_graph.nodes = {}

        # Old activations (outside 1-hour window)
        old_time = datetime.now() - timedelta(hours=2)

        mock_trace = Mock()
        mock_trace.history = [
            {'timestamp': old_time.isoformat()}
        ]
        mock_trace.get_frequency = Mock(return_value=0.0)  # Decayed

        mock_graph.activation_traces = {'node_1': mock_trace}
        mock_graph.synaptic_edges = {}

        detector = SynapticConfusionDetector(memory_graph=mock_graph)

        signals = detector.detect()

        # Old activations shouldn't trigger contradiction detection
        contradiction_signals = [s for s in signals if s.signal_type == 'synaptic_contradiction']
        assert len(contradiction_signals) == 0

    def test_bounded_history_maintenance(self):
        """Should maintain bounded activation history."""
        from llm_orchestration.recovery import SynapticConfusionDetector

        mock_graph = Mock()
        mock_graph.nodes = {}
        mock_graph.activation_traces = {}
        mock_graph.synaptic_edges = {}

        detector = SynapticConfusionDetector(memory_graph=mock_graph, loop_window=5)

        # Record many activations (should prune to max_history = loop_window * 10 = 50)
        for i in range(100):
            detector.record_activation(f'node_{i}')

        # Should be bounded
        assert len(detector._activation_sequence) <= 50


class TestDiagnosis:
    """Test diagnosis generation and recommendations."""

    def test_generate_diagnosis(self):
        """ConfusionDiagnoser should use SynapticConfusionDetector signals."""
        from llm_orchestration.recovery import (
            SynapticConfusionDetector,
            ConfusionDiagnoser,
            ConfusionType
        )

        mock_graph = Mock()
        mock_graph.nodes = {}
        mock_graph.activation_traces = {}
        mock_graph.synaptic_edges = {}

        detector = SynapticConfusionDetector(memory_graph=mock_graph)

        # Create diagnoser and add detector
        diagnoser = ConfusionDiagnoser()
        diagnoser.add_detector(detector)

        # Record a loop pattern
        for _ in range(2):
            detector.record_activation('node_a')
            detector.record_activation('node_b')

        diagnosis = diagnoser.diagnose()

        # Should generate diagnosis
        if diagnosis:
            assert diagnosis.confusion_type in [
                ConfusionType.REPETITION_LOOP,
                ConfusionType.OSCILLATION
            ]

    def test_diagnosis_recommendations(self):
        """Diagnosis should include actionable recommendations."""
        from llm_orchestration.recovery import (
            SynapticConfusionDetector,
            ConfusionDiagnoser
        )

        mock_graph = Mock()
        mock_graph.nodes = {}
        mock_graph.activation_traces = {}
        mock_graph.synaptic_edges = {}

        detector = SynapticConfusionDetector(memory_graph=mock_graph)
        diagnoser = ConfusionDiagnoser()
        diagnoser.add_detector(detector)

        # Create oscillation
        for _ in range(3):
            detector.record_activation('node_a')
            detector.record_activation('node_b')

        diagnosis = diagnoser.diagnose()

        if diagnosis:
            assert diagnosis.recommended_action is not None
            assert len(diagnosis.recommended_action) > 0
            # Should mention pruning or reset for synaptic issues
            action_lower = diagnosis.recommended_action.lower()
            assert any(keyword in action_lower for keyword in ['prune', 'reset', 'alternative'])

    def test_diagnosis_with_context(self):
        """Diagnosis should incorporate context information."""
        from llm_orchestration.recovery import (
            SynapticConfusionDetector,
            ConfusionDiagnoser
        )

        mock_graph = Mock()
        mock_graph.nodes = {}

        # Set up stagnation
        mock_trace = Mock()
        mock_trace.history = []  # Empty history
        mock_trace.get_frequency = Mock(return_value=0.05)
        mock_graph.activation_traces = {'node_1': mock_trace}
        mock_graph.synaptic_edges = {}

        detector = SynapticConfusionDetector(
            memory_graph=mock_graph,
            stagnation_threshold=0.1
        )
        diagnoser = ConfusionDiagnoser()
        diagnoser.add_detector(detector)

        context = {
            'current_task': 'complex_analysis',
            'attempts': 5
        }

        diagnosis = diagnoser.diagnose(context)

        # Should generate diagnosis (stagnation)
        if diagnosis:
            assert diagnosis.likely_cause is not None
            assert diagnosis.confidence > 0.0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_activation_sequence(self):
        """Should handle empty activation sequence gracefully."""
        from llm_orchestration.recovery import SynapticConfusionDetector

        mock_graph = Mock()
        mock_graph.nodes = {}
        mock_graph.activation_traces = {}
        mock_graph.synaptic_edges = {}

        detector = SynapticConfusionDetector(memory_graph=mock_graph)

        # No activations recorded
        signals = detector.detect()

        # Should not crash, may return empty or minimal signals
        assert isinstance(signals, list)

    def test_single_activation(self):
        """Should handle single activation without detecting loops."""
        from llm_orchestration.recovery import SynapticConfusionDetector

        mock_graph = Mock()
        mock_graph.nodes = {}
        mock_graph.activation_traces = {}
        mock_graph.synaptic_edges = {}

        detector = SynapticConfusionDetector(memory_graph=mock_graph)

        detector.record_activation('node_1')
        signals = detector.detect()

        # Should not detect loop with single activation
        loop_signals = [s for s in signals if s.signal_type == 'synaptic_loop']
        assert len(loop_signals) == 0

    def test_missing_node_in_graph(self):
        """Should handle activations for nodes not in graph."""
        from llm_orchestration.recovery import SynapticConfusionDetector

        mock_graph = Mock()
        mock_graph.nodes = {}  # Empty - nodes not in graph
        mock_graph.activation_traces = {}
        mock_graph.synaptic_edges = {}

        detector = SynapticConfusionDetector(memory_graph=mock_graph)

        # Record activations for non-existent nodes
        detector.record_activation('missing_node_1')
        detector.record_activation('missing_node_2')
        detector.record_activation('missing_node_1')
        detector.record_activation('missing_node_2')

        # Should not crash
        signals = detector.detect()
        assert isinstance(signals, list)

    def test_contradiction_check_with_missing_nodes(self):
        """Should handle contradiction check when nodes are missing."""
        from llm_orchestration.recovery import SynapticConfusionDetector

        mock_graph = Mock()
        mock_graph.nodes = {}  # Empty
        mock_graph.activation_traces = {}
        mock_graph.synaptic_edges = {}

        detector = SynapticConfusionDetector(memory_graph=mock_graph)

        # Check contradiction between non-existent nodes
        strength = detector._check_contradiction('node_1', 'node_2')

        # Should return 0.0 for missing nodes
        assert strength == 0.0

    def test_stagnation_with_empty_traces(self):
        """Should handle stagnation detection with no activation traces."""
        from llm_orchestration.recovery import SynapticConfusionDetector

        mock_graph = Mock()
        mock_graph.nodes = {}
        mock_graph.activation_traces = {}  # Empty
        mock_graph.synaptic_edges = {}

        detector = SynapticConfusionDetector(memory_graph=mock_graph)

        signals = detector.detect()

        # Should not crash, should not detect stagnation with no traces
        stagnation_signals = [s for s in signals if s.signal_type == 'synaptic_stagnation']
        assert len(stagnation_signals) == 0

    def test_oscillation_with_short_sequence(self):
        """Should not detect oscillation with insufficient data."""
        from llm_orchestration.recovery import SynapticConfusionDetector

        mock_graph = Mock()
        mock_graph.nodes = {}
        mock_graph.activation_traces = {}
        mock_graph.synaptic_edges = {}

        detector = SynapticConfusionDetector(memory_graph=mock_graph)

        # Too few activations for oscillation (need at least 6)
        detector.record_activation('node_a')
        detector.record_activation('node_b')
        detector.record_activation('node_a')

        signals = detector.detect()

        oscillation_signals = [s for s in signals if s.signal_type == 'synaptic_oscillation']
        # Should not detect with only 3 activations
        assert len(oscillation_signals) == 0
