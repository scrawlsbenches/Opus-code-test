"""
Unit tests for the Loom dual-process integration layer.

Tests cover:
- SurpriseSignal and LoomConfig dataclasses
- SurpriseDetector prediction error computation
- ModeController mode selection logic
- Loom full integration
- Observable mode transitions
- Edge cases and error handling
"""

import pytest
from collections import deque
from datetime import datetime
from typing import List
from unittest.mock import MagicMock

from cortical.reasoning.loom import (
    # Enums
    ThinkingMode,
    TransitionTrigger,
    # Data structures
    SurpriseSignal,
    LoomConfig,
    ModeTransition,
    # Protocols
    LoomObserverProtocol,
    # Implementations
    SurpriseDetector,
    ModeController,
    Loom,
)


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def default_config():
    """Default LoomConfig for testing."""
    return LoomConfig()


@pytest.fixture
def strict_config():
    """Config with lower thresholds for stricter mode switching."""
    return LoomConfig(
        surprise_threshold=0.2,
        confidence_threshold=0.7,
    )


@pytest.fixture
def surprise_detector(default_config):
    """Fresh SurpriseDetector for testing."""
    return SurpriseDetector(default_config)


@pytest.fixture
def mode_controller(default_config):
    """Fresh ModeController for testing."""
    return ModeController(default_config)


@pytest.fixture
def loom(default_config):
    """Fresh Loom instance for testing."""
    return Loom(default_config)


class MockObserver:
    """Mock observer for testing observability."""

    def __init__(self):
        self.surprises: List[SurpriseSignal] = []
        self.transitions: List[ModeTransition] = []
        self.mode_selections: List[tuple] = []

    def on_surprise(self, signal: SurpriseSignal) -> None:
        self.surprises.append(signal)

    def on_transition(self, transition: ModeTransition) -> None:
        self.transitions.append(transition)

    def on_mode_selected(self, mode: ThinkingMode, reason: str) -> None:
        self.mode_selections.append((mode, reason))


# ==============================================================================
# DATA STRUCTURE TESTS
# ==============================================================================


class TestSurpriseSignal:
    """Tests for SurpriseSignal dataclass."""

    def test_creation_with_defaults(self):
        """Test creating SurpriseSignal with minimal args."""
        signal = SurpriseSignal(magnitude=0.5, source="test")
        assert signal.magnitude == 0.5
        assert signal.source == "test"
        assert signal.context == {}
        assert isinstance(signal.timestamp, datetime)

    def test_creation_with_context(self):
        """Test creating SurpriseSignal with context."""
        ctx = {"missed": ["a", "b"]}
        signal = SurpriseSignal(magnitude=0.7, source="detector", context=ctx)
        assert signal.context == ctx

    def test_magnitude_validation_low(self):
        """Test that magnitude below 0 raises ValueError."""
        with pytest.raises(ValueError, match="Magnitude must be in"):
            SurpriseSignal(magnitude=-0.1, source="test")

    def test_magnitude_validation_high(self):
        """Test that magnitude above 1 raises ValueError."""
        with pytest.raises(ValueError, match="Magnitude must be in"):
            SurpriseSignal(magnitude=1.1, source="test")

    def test_is_significant_above_threshold(self):
        """Test is_significant returns True above threshold."""
        signal = SurpriseSignal(magnitude=0.5, source="test")
        assert signal.is_significant(threshold=0.3) is True

    def test_is_significant_below_threshold(self):
        """Test is_significant returns False below threshold."""
        signal = SurpriseSignal(magnitude=0.2, source="test")
        assert signal.is_significant(threshold=0.3) is False

    def test_is_significant_at_threshold(self):
        """Test is_significant returns False at threshold."""
        signal = SurpriseSignal(magnitude=0.3, source="test")
        assert signal.is_significant(threshold=0.3) is False


class TestLoomConfig:
    """Tests for LoomConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LoomConfig()
        assert config.surprise_threshold == 0.3
        assert config.confidence_threshold == 0.6
        assert config.history_window == 100
        assert config.adaptation_rate == 0.1
        assert config.timeout_ms == 100
        assert config.enable_observability is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = LoomConfig(
            surprise_threshold=0.5,
            confidence_threshold=0.8,
            history_window=50,
        )
        assert config.surprise_threshold == 0.5
        assert config.confidence_threshold == 0.8
        assert config.history_window == 50

    def test_surprise_threshold_validation(self):
        """Test surprise_threshold must be in [0, 1]."""
        with pytest.raises(ValueError):
            LoomConfig(surprise_threshold=1.5)
        with pytest.raises(ValueError):
            LoomConfig(surprise_threshold=-0.1)

    def test_confidence_threshold_validation(self):
        """Test confidence_threshold must be in [0, 1]."""
        with pytest.raises(ValueError):
            LoomConfig(confidence_threshold=1.5)

    def test_history_window_validation(self):
        """Test history_window must be >= 1."""
        with pytest.raises(ValueError):
            LoomConfig(history_window=0)

    def test_adaptation_rate_validation(self):
        """Test adaptation_rate must be in (0, 1]."""
        with pytest.raises(ValueError):
            LoomConfig(adaptation_rate=0.0)
        with pytest.raises(ValueError):
            LoomConfig(adaptation_rate=1.5)

    def test_timeout_validation(self):
        """Test timeout_ms must be >= 0."""
        with pytest.raises(ValueError):
            LoomConfig(timeout_ms=-1)


class TestModeTransition:
    """Tests for ModeTransition dataclass."""

    def test_creation(self):
        """Test creating ModeTransition."""
        transition = ModeTransition(
            from_mode=ThinkingMode.FAST,
            to_mode=ThinkingMode.SLOW,
            trigger=TransitionTrigger.SURPRISE,
        )
        assert transition.from_mode == ThinkingMode.FAST
        assert transition.to_mode == ThinkingMode.SLOW
        assert transition.trigger == TransitionTrigger.SURPRISE
        assert transition.surprise_signal is None
        assert isinstance(transition.timestamp, datetime)

    def test_with_surprise_signal(self):
        """Test ModeTransition with surprise signal."""
        signal = SurpriseSignal(magnitude=0.8, source="test")
        transition = ModeTransition(
            from_mode=ThinkingMode.FAST,
            to_mode=ThinkingMode.SLOW,
            trigger=TransitionTrigger.SURPRISE,
            surprise_signal=signal,
        )
        assert transition.surprise_signal is signal


# ==============================================================================
# SURPRISE DETECTOR TESTS
# ==============================================================================


class TestSurpriseDetector:
    """Tests for SurpriseDetector implementation."""

    def test_perfect_prediction_zero_surprise(self, surprise_detector):
        """Perfect prediction should yield zero surprise."""
        signal = surprise_detector.compute_surprise(
            predicted={"a": 1.0, "b": 0.0},
            actual={"a"},
        )
        assert signal.magnitude == 0.0

    def test_complete_miss_high_surprise(self, surprise_detector):
        """Complete miss should yield high surprise."""
        signal = surprise_detector.compute_surprise(
            predicted={"a": 0.9, "b": 0.1},
            actual={"c", "d"},
        )
        # 0.9 error for a (predicted 0.9, actual 0)
        # 0.1 error for b (predicted 0.1, actual 0)
        # 1.0 error for c (unpredicted activation)
        # 1.0 error for d (unpredicted activation)
        # Total: (0.9 + 0.1 + 1.0 + 1.0) / 4 = 0.75
        assert signal.magnitude == 0.75

    def test_partial_match_medium_surprise(self, surprise_detector):
        """Partial match should yield medium surprise."""
        signal = surprise_detector.compute_surprise(
            predicted={"a": 0.8, "b": 0.2},
            actual={"a", "c"},  # a correct, b wrong, c unpredicted
        )
        # a: |0.8 - 1.0| = 0.2
        # b: |0.2 - 0.0| = 0.2
        # c: 1.0 (unpredicted)
        # Total: (0.2 + 0.2 + 1.0) / 3 = 0.467
        assert 0.4 < signal.magnitude < 0.5

    def test_empty_inputs_zero_surprise(self, surprise_detector):
        """Empty predictions and actuals should yield zero surprise."""
        signal = surprise_detector.compute_surprise(
            predicted={},
            actual=set(),
        )
        assert signal.magnitude == 0.0

    def test_only_predictions_no_actuals(self, surprise_detector):
        """All predictions, no actuals = high surprise."""
        signal = surprise_detector.compute_surprise(
            predicted={"a": 0.9, "b": 0.8},
            actual=set(),
        )
        # a: |0.9 - 0| = 0.9
        # b: |0.8 - 0| = 0.8
        # Average: 0.85
        assert abs(signal.magnitude - 0.85) < 0.001

    def test_only_actuals_no_predictions(self, surprise_detector):
        """All actuals, no predictions = high surprise."""
        signal = surprise_detector.compute_surprise(
            predicted={},
            actual={"a", "b"},
        )
        # Both unpredicted = 1.0 each
        assert signal.magnitude == 1.0

    def test_baseline_adaptation(self, surprise_detector):
        """Baseline should adapt over time."""
        initial_baseline = surprise_detector.get_baseline()
        assert initial_baseline == 0.0

        # First observation sets baseline
        surprise_detector.compute_surprise(
            predicted={"a": 0.5},
            actual={"b"},
        )
        first_baseline = surprise_detector.get_baseline()
        assert first_baseline > 0

        # Multiple low-surprise observations should decrease baseline
        for _ in range(50):
            surprise_detector.compute_surprise(
                predicted={"a": 1.0},  # Perfect prediction
                actual={"a"},
            )

        # Baseline should have moved toward lower values
        final_baseline = surprise_detector.get_baseline()
        assert final_baseline < first_baseline  # Adapted toward low surprise

    def test_should_engage_slow_above_threshold(self, surprise_detector):
        """Should engage slow when surprise exceeds threshold."""
        signal = SurpriseSignal(magnitude=0.5, source="test")
        assert surprise_detector.should_engage_slow(signal) is True

    def test_should_engage_slow_below_threshold(self, surprise_detector):
        """Should not engage slow when surprise below threshold."""
        signal = SurpriseSignal(magnitude=0.2, source="test")
        assert surprise_detector.should_engage_slow(signal) is False

    def test_reset_baseline(self, surprise_detector):
        """Reset should clear baseline and history."""
        # Build up some history
        for _ in range(5):
            surprise_detector.compute_surprise({"a": 0.5}, {"b"})

        assert surprise_detector.get_baseline() > 0
        assert len(surprise_detector.history) > 0

        surprise_detector.reset_baseline()

        assert surprise_detector.get_baseline() == 0.0
        assert len(surprise_detector.history) == 0

    def test_context_contains_diagnostics(self, surprise_detector):
        """Signal context should contain diagnostic info."""
        signal = surprise_detector.compute_surprise(
            predicted={"a": 0.9, "b": 0.1},
            actual={"b", "c"},
        )
        assert "raw_surprise" in signal.context
        assert "baseline" in signal.context
        assert "predicted_count" in signal.context
        assert "actual_count" in signal.context
        assert "unpredicted_count" in signal.context
        assert signal.context["predicted_count"] == 2
        assert signal.context["actual_count"] == 2
        assert signal.context["unpredicted_count"] == 1


# ==============================================================================
# MODE CONTROLLER TESTS
# ==============================================================================


class TestModeController:
    """Tests for ModeController implementation."""

    def test_default_mode_is_fast(self, mode_controller):
        """Default mode should be FAST."""
        assert mode_controller.get_current_mode() == ThinkingMode.FAST

    def test_low_surprise_stays_fast(self, mode_controller):
        """Low surprise should keep FAST mode."""
        signal = SurpriseSignal(magnitude=0.1, source="test")
        mode = mode_controller.select_mode(surprise=signal)
        assert mode == ThinkingMode.FAST

    def test_high_surprise_switches_to_slow(self, mode_controller):
        """High surprise should switch to SLOW mode."""
        signal = SurpriseSignal(magnitude=0.8, source="test")
        mode = mode_controller.select_mode(surprise=signal)
        assert mode == ThinkingMode.SLOW

    def test_low_confidence_switches_to_slow(self, mode_controller):
        """Low confidence should switch to SLOW mode."""
        mode = mode_controller.select_mode(confidence=0.3)
        assert mode == ThinkingMode.SLOW

    def test_high_confidence_stays_fast(self, mode_controller):
        """High confidence should keep FAST mode."""
        mode = mode_controller.select_mode(confidence=0.9)
        assert mode == ThinkingMode.FAST

    def test_high_complexity_switches_to_slow(self, mode_controller):
        """High complexity should switch to SLOW mode."""
        mode = mode_controller.select_mode(complexity=0.9)
        assert mode == ThinkingMode.SLOW

    def test_complexity_takes_priority(self, mode_controller):
        """Complexity should be checked before surprise."""
        signal = SurpriseSignal(magnitude=0.1, source="test")  # Low surprise
        mode = mode_controller.select_mode(
            surprise=signal,
            complexity=0.9,  # But high complexity
        )
        assert mode == ThinkingMode.SLOW

    def test_no_signals_stays_fast(self, mode_controller):
        """No signals should keep FAST mode."""
        mode = mode_controller.select_mode()
        assert mode == ThinkingMode.FAST

    def test_transition_recorded(self, mode_controller):
        """Mode transitions should be recorded."""
        # Initial state
        assert len(mode_controller.get_transition_history()) == 0

        # Trigger transition
        signal = SurpriseSignal(magnitude=0.8, source="test")
        mode_controller.select_mode(surprise=signal)

        history = mode_controller.get_transition_history()
        assert len(history) == 1
        assert history[0].from_mode == ThinkingMode.FAST
        assert history[0].to_mode == ThinkingMode.SLOW
        assert history[0].trigger == TransitionTrigger.SURPRISE

    def test_no_duplicate_transitions(self, mode_controller):
        """Same mode selection should not record duplicate transitions."""
        # Switch to SLOW
        signal = SurpriseSignal(magnitude=0.8, source="test")
        mode_controller.select_mode(surprise=signal)

        # Stay in SLOW (should not record new transition)
        mode_controller.select_mode(surprise=signal)

        history = mode_controller.get_transition_history()
        assert len(history) == 1

    def test_observer_notified_on_transition(self, mode_controller):
        """Observer should be notified on mode transitions."""
        observer = MockObserver()
        mode_controller.register_observer(observer)

        signal = SurpriseSignal(magnitude=0.8, source="test")
        mode_controller.select_mode(surprise=signal)

        assert len(observer.transitions) == 1
        assert observer.transitions[0].to_mode == ThinkingMode.SLOW

    def test_unregister_observer(self, mode_controller):
        """Unregistered observer should not receive notifications."""
        observer = MockObserver()
        mode_controller.register_observer(observer)
        mode_controller.unregister_observer(observer)

        signal = SurpriseSignal(magnitude=0.8, source="test")
        mode_controller.select_mode(surprise=signal)

        assert len(observer.transitions) == 0

    def test_force_mode(self, mode_controller):
        """force_mode should override normal selection."""
        mode_controller.force_mode(ThinkingMode.SLOW, reason="testing")

        assert mode_controller.get_current_mode() == ThinkingMode.SLOW

        history = mode_controller.get_transition_history()
        assert len(history) == 1
        assert history[0].trigger == TransitionTrigger.EXPLICIT_REQUEST


# ==============================================================================
# LOOM INTEGRATION TESTS
# ==============================================================================


class TestLoom:
    """Tests for Loom integration."""

    def test_detect_surprise_delegates(self, loom):
        """detect_surprise should delegate to SurpriseDetector."""
        signal = loom.detect_surprise(
            predicted={"a": 0.9},
            actual={"b"},
        )
        assert signal.magnitude > 0.5
        assert signal.source == "surprise_detector"

    def test_select_mode_delegates(self, loom):
        """select_mode should delegate to ModeController."""
        signal = SurpriseSignal(magnitude=0.8, source="test")
        mode = loom.select_mode(signal)
        assert mode == ThinkingMode.SLOW

    def test_full_workflow(self, loom):
        """Test complete detect -> select workflow."""
        # Detect surprise
        signal = loom.detect_surprise(
            predicted={"a": 0.9, "b": 0.1},
            actual={"c"},  # Complete miss
        )

        # Select mode based on surprise
        mode = loom.select_mode(signal)

        assert mode == ThinkingMode.SLOW
        assert loom.get_current_mode() == ThinkingMode.SLOW

    def test_observer_receives_all_events(self, loom):
        """Observer should receive surprise and mode events."""
        observer = MockObserver()
        loom.register_observer(observer)

        signal = loom.detect_surprise({"a": 0.9}, {"b"})
        loom.select_mode(signal)

        assert len(observer.surprises) == 1
        assert len(observer.mode_selections) == 1

    def test_get_config(self, loom, default_config):
        """get_config should return the configuration."""
        config = loom.get_config()
        assert config.surprise_threshold == default_config.surprise_threshold

    def test_get_surprise_baseline(self, loom):
        """get_surprise_baseline should return current baseline."""
        assert loom.get_surprise_baseline() == 0.0

        loom.detect_surprise({"a": 0.5}, {"b"})
        assert loom.get_surprise_baseline() > 0

    def test_get_transition_history(self, loom):
        """get_transition_history should return mode transitions."""
        signal = SurpriseSignal(magnitude=0.8, source="test")
        loom.select_mode(signal)

        history = loom.get_transition_history()
        assert len(history) == 1

    def test_reset(self, loom):
        """reset should restore initial state."""
        # Build up state
        loom.detect_surprise({"a": 0.5}, {"b"})
        loom.select_mode(SurpriseSignal(magnitude=0.8, source="test"))

        assert loom.get_current_mode() == ThinkingMode.SLOW
        assert loom.get_surprise_baseline() > 0

        # Reset
        loom.reset()

        assert loom.get_current_mode() == ThinkingMode.FAST
        assert loom.get_surprise_baseline() == 0.0
        assert len(loom.get_transition_history()) == 0

    def test_with_custom_config(self):
        """Loom should respect custom configuration."""
        config = LoomConfig(surprise_threshold=0.9)
        loom = Loom(config)

        # 0.5 surprise should NOT trigger slow with high threshold
        signal = SurpriseSignal(magnitude=0.5, source="test")
        mode = loom.select_mode(signal)
        assert mode == ThinkingMode.FAST

        # But 0.95 should
        signal = SurpriseSignal(magnitude=0.95, source="test")
        mode = loom.select_mode(signal)
        assert mode == ThinkingMode.SLOW


# ==============================================================================
# ENUM TESTS
# ==============================================================================


class TestThinkingMode:
    """Tests for ThinkingMode enum."""

    def test_fast_value(self):
        """FAST should have expected value."""
        assert ThinkingMode.FAST.name == "FAST"

    def test_slow_value(self):
        """SLOW should have expected value."""
        assert ThinkingMode.SLOW.name == "SLOW"

    def test_str_representation(self):
        """__str__ should return name."""
        assert str(ThinkingMode.FAST) == "FAST"
        assert str(ThinkingMode.SLOW) == "SLOW"


class TestTransitionTrigger:
    """Tests for TransitionTrigger enum."""

    def test_all_triggers_exist(self):
        """All expected triggers should exist."""
        assert hasattr(TransitionTrigger, "SURPRISE")
        assert hasattr(TransitionTrigger, "EXPLICIT_REQUEST")
        assert hasattr(TransitionTrigger, "TIMEOUT")
        assert hasattr(TransitionTrigger, "CONFIDENCE_LOW")
        assert hasattr(TransitionTrigger, "COMPLEXITY")
        assert hasattr(TransitionTrigger, "NOVELTY")


# ==============================================================================
# EDGE CASE TESTS
# ==============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_observer_error_does_not_break_controller(self, mode_controller):
        """Observer errors should not break the controller."""

        class BrokenObserver:
            def on_transition(self, transition):
                raise RuntimeError("Observer exploded!")

            def on_surprise(self, signal):
                raise RuntimeError("Observer exploded!")

            def on_mode_selected(self, mode, reason):
                raise RuntimeError("Observer exploded!")

        mode_controller.register_observer(BrokenObserver())

        # Should not raise
        signal = SurpriseSignal(magnitude=0.8, source="test")
        mode = mode_controller.select_mode(surprise=signal)
        assert mode == ThinkingMode.SLOW

    def test_loom_observer_error_does_not_break(self, loom):
        """Observer errors should not break the Loom."""

        class BrokenObserver:
            def on_transition(self, transition):
                raise RuntimeError("Boom!")

            def on_surprise(self, signal):
                raise RuntimeError("Boom!")

            def on_mode_selected(self, mode, reason):
                raise RuntimeError("Boom!")

        loom.register_observer(BrokenObserver())

        # Should not raise
        signal = loom.detect_surprise({"a": 0.9}, {"b"})
        loom.select_mode(signal)

    def test_boundary_surprise_values(self, surprise_detector):
        """Test surprise at exact boundaries."""
        # Exactly 0.0
        signal = SurpriseSignal(magnitude=0.0, source="test")
        assert surprise_detector.should_engage_slow(signal) is False

        # Exactly 1.0
        signal = SurpriseSignal(magnitude=1.0, source="test")
        assert surprise_detector.should_engage_slow(signal) is True

    def test_very_small_predictions(self, surprise_detector):
        """Test with very small prediction probabilities."""
        signal = surprise_detector.compute_surprise(
            predicted={"a": 0.001, "b": 0.001},
            actual={"a", "b"},
        )
        # Both nearly perfect: |0.001 - 1.0| = 0.999 each... wait that's wrong
        # Actually these are nearly wrong predictions that happen to match
        # Error for a: |0.001 - 1.0| = 0.999
        # Error for b: |0.001 - 1.0| = 0.999
        # High surprise because we didn't expect them to activate
        assert signal.magnitude > 0.9

    def test_many_predictions(self, surprise_detector):
        """Test with many predictions."""
        predictions = {f"node_{i}": 0.1 for i in range(100)}
        actual = {f"node_{i}" for i in range(50)}  # Half match

        signal = surprise_detector.compute_surprise(predictions, actual)
        # Some matched, some didn't
        assert 0.3 < signal.magnitude < 0.7


# ==============================================================================
# LOOM ENHANCED ATTENTION TESTS
# ==============================================================================


class TestLoomAttentionResult:
    """Tests for LoomAttentionResult dataclass."""

    def test_creation_with_defaults(self):
        """Test creating LoomAttentionResult with minimal args."""
        from cortical.reasoning.loom import LoomAttentionResult

        result = LoomAttentionResult(
            top_nodes=["a", "b", "c"],
            weights={"a": 0.8, "b": 0.5, "c": 0.3},
            mode_used=ThinkingMode.FAST,
        )
        assert result.top_nodes == ["a", "b", "c"]
        assert result.weights == {"a": 0.8, "b": 0.5, "c": 0.3}
        assert result.mode_used == ThinkingMode.FAST
        assert result.surprise is None
        assert result.pln_support == 0.0
        assert result.slm_fluency == 0.0
        assert result.fast_result is None
        assert result.slow_result is None

    def test_creation_with_all_args(self):
        """Test creating LoomAttentionResult with all arguments."""
        from cortical.reasoning.loom import LoomAttentionResult

        signal = SurpriseSignal(magnitude=0.5, source="test")
        result = LoomAttentionResult(
            top_nodes=["x"],
            weights={"x": 1.0},
            mode_used=ThinkingMode.SLOW,
            surprise=signal,
            pln_support=0.8,
            slm_fluency=0.7,
            fast_result={"fast": "data"},
            slow_result={"slow": "data"},
        )
        assert result.surprise is signal
        assert result.pln_support == 0.8
        assert result.slm_fluency == 0.7
        assert result.fast_result == {"fast": "data"}
        assert result.slow_result == {"slow": "data"}


class TestLoomEnhancedAttention:
    """Tests for LoomEnhancedAttention class."""

    @pytest.fixture
    def mock_graph(self):
        """Create a mock SynapticMemoryGraph."""
        graph = MagicMock()
        graph.nodes = {
            "node_a": MagicMock(content="neural network learning"),
            "node_b": MagicMock(content="machine learning model"),
            "node_c": MagicMock(content="deep learning algorithm"),
        }
        return graph

    @pytest.fixture
    def mock_attention_weights(self):
        """Standard attention weights for testing."""
        return {"node_a": 0.8, "node_b": 0.5, "node_c": 0.3}

    @pytest.fixture
    def mock_attention_layer(self, mock_attention_weights):
        """Create a mock AttentionLayer."""
        mock_layer = MagicMock()
        mock_layer.attend.return_value = mock_attention_weights
        return mock_layer

    def test_init_without_slm_pln(self, mock_graph, mock_attention_layer):
        """Test initialization without SLM or PLN."""
        from unittest.mock import patch
        from cortical.reasoning.loom import LoomEnhancedAttention

        with patch(
            "cortical.reasoning.prism_attention.AttentionLayer",
            return_value=mock_attention_layer,
        ):
            enhanced = LoomEnhancedAttention(mock_graph)

            assert enhanced._graph is mock_graph
            assert enhanced._slm is None
            assert enhanced._pln is None
            assert enhanced._unified_attention is None
            assert isinstance(enhanced._config, LoomConfig)

    def test_init_with_slm(self, mock_graph, mock_attention_layer):
        """Test initialization with SLM."""
        from unittest.mock import patch
        from cortical.reasoning.loom import LoomEnhancedAttention

        mock_slm = MagicMock()
        mock_unified = MagicMock()

        with patch(
            "cortical.reasoning.prism_attention.AttentionLayer",
            return_value=mock_attention_layer,
        ), patch(
            "cortical.reasoning.prism_attention.UnifiedAttention",
            return_value=mock_unified,
        ):
            enhanced = LoomEnhancedAttention(mock_graph, slm=mock_slm)

            assert enhanced._slm is mock_slm
            assert enhanced._unified_attention is mock_unified

    def test_init_with_pln(self, mock_graph, mock_attention_layer):
        """Test initialization with PLN."""
        from unittest.mock import patch
        from cortical.reasoning.loom import LoomEnhancedAttention

        mock_pln = MagicMock()
        mock_unified = MagicMock()

        with patch(
            "cortical.reasoning.prism_attention.AttentionLayer",
            return_value=mock_attention_layer,
        ), patch(
            "cortical.reasoning.prism_attention.UnifiedAttention",
            return_value=mock_unified,
        ):
            enhanced = LoomEnhancedAttention(mock_graph, pln=mock_pln)

            assert enhanced._pln is mock_pln
            assert enhanced._unified_attention is mock_unified

    def test_init_with_custom_config(self, mock_graph, mock_attention_layer):
        """Test initialization with custom config."""
        from unittest.mock import patch
        from cortical.reasoning.loom import LoomEnhancedAttention

        config = LoomConfig(surprise_threshold=0.5)

        with patch(
            "cortical.reasoning.prism_attention.AttentionLayer",
            return_value=mock_attention_layer,
        ):
            enhanced = LoomEnhancedAttention(mock_graph, config=config)

            assert enhanced._config.surprise_threshold == 0.5

    def test_attend_first_call_no_surprise(
        self, mock_graph, mock_attention_layer, mock_attention_weights
    ):
        """Test attend on first call (no previous predictions)."""
        from unittest.mock import patch
        from cortical.reasoning.loom import LoomEnhancedAttention

        with patch(
            "cortical.reasoning.prism_attention.AttentionLayer",
            return_value=mock_attention_layer,
        ):
            enhanced = LoomEnhancedAttention(mock_graph)
            result = enhanced.attend("test query")

            assert result.surprise is None
            assert result.mode_used == ThinkingMode.FAST
            assert "node_a" in result.top_nodes
            assert result.weights == mock_attention_weights

    def test_attend_second_call_with_surprise(
        self, mock_graph, mock_attention_layer, mock_attention_weights
    ):
        """Test attend on second call (with previous predictions)."""
        from unittest.mock import patch
        from cortical.reasoning.loom import LoomEnhancedAttention

        with patch(
            "cortical.reasoning.prism_attention.AttentionLayer",
            return_value=mock_attention_layer,
        ):
            enhanced = LoomEnhancedAttention(mock_graph)

            # First call to set up predictions
            enhanced.attend("query 1")

            # Second call should compute surprise
            result = enhanced.attend("query 2")

            # Should have surprise computed (exact value depends on matching)
            assert result.surprise is not None

    def test_attend_with_force_mode_slow(
        self, mock_graph, mock_attention_layer, mock_attention_weights
    ):
        """Test attend with forced SLOW mode."""
        from unittest.mock import patch
        from cortical.reasoning.loom import LoomEnhancedAttention

        with patch(
            "cortical.reasoning.prism_attention.AttentionLayer",
            return_value=mock_attention_layer,
        ):
            enhanced = LoomEnhancedAttention(mock_graph)
            result = enhanced.attend("test", force_mode=ThinkingMode.SLOW)

            assert result.mode_used == ThinkingMode.SLOW

    def test_attend_with_force_mode_fast(
        self, mock_graph, mock_attention_layer, mock_attention_weights
    ):
        """Test attend with forced FAST mode."""
        from unittest.mock import patch
        from cortical.reasoning.loom import LoomEnhancedAttention

        with patch(
            "cortical.reasoning.prism_attention.AttentionLayer",
            return_value=mock_attention_layer,
        ):
            enhanced = LoomEnhancedAttention(mock_graph)
            result = enhanced.attend("test", force_mode=ThinkingMode.FAST)

            assert result.mode_used == ThinkingMode.FAST

    def test_attend_slow_mode_with_unified_attention(self, mock_graph):
        """Test SLOW mode uses unified attention when available."""
        from unittest.mock import patch
        from cortical.reasoning.loom import LoomEnhancedAttention

        mock_slm = MagicMock()

        mock_attention_layer = MagicMock()
        mock_attention_layer.attend.return_value = {"node_a": 0.5}

        mock_unified = MagicMock()
        mock_unified_result = MagicMock()
        mock_unified_result.weights = {"unified_a": 0.9}
        mock_unified_result.top_nodes = ["unified_a"]
        mock_unified_result.pln_support = 0.8
        mock_unified_result.slm_fluency = 0.7
        mock_unified.attend.return_value = mock_unified_result

        with patch(
            "cortical.reasoning.prism_attention.AttentionLayer",
            return_value=mock_attention_layer,
        ), patch(
            "cortical.reasoning.prism_attention.UnifiedAttention",
            return_value=mock_unified,
        ):
            enhanced = LoomEnhancedAttention(mock_graph, slm=mock_slm)
            result = enhanced.attend("test", force_mode=ThinkingMode.SLOW)

            assert result.mode_used == ThinkingMode.SLOW
            assert result.pln_support == 0.8
            assert result.slm_fluency == 0.7
            assert result.top_nodes == ["unified_a"]
            assert result.slow_result is mock_unified_result

    def test_get_loom(self, mock_graph, mock_attention_layer):
        """Test get_loom returns the Loom instance."""
        from unittest.mock import patch
        from cortical.reasoning.loom import LoomEnhancedAttention

        with patch(
            "cortical.reasoning.prism_attention.AttentionLayer",
            return_value=mock_attention_layer,
        ):
            enhanced = LoomEnhancedAttention(mock_graph)
            loom = enhanced.get_loom()

            assert isinstance(loom, Loom)

    def test_get_transition_history(self, mock_graph, mock_attention_layer):
        """Test get_transition_history delegates to Loom."""
        from unittest.mock import patch
        from cortical.reasoning.loom import LoomEnhancedAttention

        with patch(
            "cortical.reasoning.prism_attention.AttentionLayer",
            return_value=mock_attention_layer,
        ):
            enhanced = LoomEnhancedAttention(mock_graph)
            history = enhanced.get_transition_history()

            assert history == []

    def test_reset(self, mock_graph, mock_attention_layer, mock_attention_weights):
        """Test reset clears state."""
        from unittest.mock import patch
        from cortical.reasoning.loom import LoomEnhancedAttention

        with patch(
            "cortical.reasoning.prism_attention.AttentionLayer",
            return_value=mock_attention_layer,
        ):
            enhanced = LoomEnhancedAttention(mock_graph)

            # Build up some state
            enhanced.attend("query 1")
            assert enhanced._last_predictions  # Should have predictions

            # Reset
            enhanced.reset()

            assert enhanced._last_predictions == {}
            assert enhanced.get_loom().get_current_mode() == ThinkingMode.FAST

    def test_attend_with_empty_weights(self, mock_graph):
        """Test attend handles empty attention weights gracefully."""
        from unittest.mock import patch
        from cortical.reasoning.loom import LoomEnhancedAttention

        empty_attention_layer = MagicMock()
        empty_attention_layer.attend.return_value = {}

        with patch(
            "cortical.reasoning.prism_attention.AttentionLayer",
            return_value=empty_attention_layer,
        ):
            enhanced = LoomEnhancedAttention(mock_graph)
            result = enhanced.attend("empty query")

            assert result.top_nodes == []
            assert result.weights == {}


# ==============================================================================
# PROTOCOL CONFORMANCE TESTS
# ==============================================================================


class TestProtocolConformance:
    """Tests to verify protocol implementations are correct."""

    def test_loom_implements_interface(self):
        """Verify Loom implements LoomInterface correctly."""
        from cortical.reasoning.loom import LoomInterface

        loom = Loom()

        # Verify all abstract methods are implemented
        assert hasattr(loom, 'detect_surprise')
        assert hasattr(loom, 'select_mode')
        assert hasattr(loom, 'register_observer')
        assert hasattr(loom, 'get_current_mode')
        assert hasattr(loom, 'get_config')

        # Verify it's a proper subclass
        assert isinstance(loom, LoomInterface)

    def test_surprise_detector_matches_protocol(self, surprise_detector):
        """Verify SurpriseDetector matches SurpriseDetectorProtocol."""
        # Check required methods exist and work
        signal = surprise_detector.compute_surprise({"a": 0.5}, {"a"})
        assert isinstance(signal, SurpriseSignal)

        result = surprise_detector.should_engage_slow(signal)
        assert isinstance(result, bool)

    def test_mode_controller_matches_protocol(self, mode_controller):
        """Verify ModeController matches ModeControllerProtocol."""
        # Check required methods exist and work
        mode = mode_controller.select_mode()
        assert isinstance(mode, ThinkingMode)

        history = mode_controller.get_transition_history()
        assert isinstance(history, list)

        # Record transition works
        transition = ModeTransition(
            from_mode=ThinkingMode.FAST,
            to_mode=ThinkingMode.SLOW,
            trigger=TransitionTrigger.EXPLICIT_REQUEST,
        )
        mode_controller.record_transition(transition)
        assert len(mode_controller.get_transition_history()) == 1


# ==============================================================================
# OBSERVABILITY DISABLED TESTS
# ==============================================================================


class TestObservabilityDisabled:
    """Tests for behavior when observability is disabled."""

    def test_loom_no_observer_calls_when_disabled(self):
        """Loom should not call observers when observability disabled."""
        config = LoomConfig(enable_observability=False)
        loom = Loom(config)

        observer = MockObserver()
        loom.register_observer(observer)

        # Actions should work but not notify observers
        signal = loom.detect_surprise({"a": 0.9}, {"b"})
        loom.select_mode(signal)

        # Observer should NOT have been notified
        assert len(observer.surprises) == 0
        assert len(observer.mode_selections) == 0

    def test_mode_controller_no_notifications_when_no_observers(self, mode_controller):
        """ModeController works fine with no observers registered."""
        # This should not raise even with transition
        signal = SurpriseSignal(magnitude=0.8, source="test")
        mode = mode_controller.select_mode(surprise=signal)
        assert mode == ThinkingMode.SLOW

    def test_loom_select_mode_without_signal(self):
        """Loom select_mode works without surprise signal."""
        loom = Loom()
        observer = MockObserver()
        loom.register_observer(observer)

        mode = loom.select_mode()

        assert mode == ThinkingMode.FAST
        # Observer should get mode selection notification
        assert len(observer.mode_selections) == 1
        assert "default" in observer.mode_selections[0][1]

    def test_loom_select_mode_with_context(self):
        """Loom select_mode with context kwargs."""
        loom = Loom()

        # Low confidence should trigger SLOW
        mode = loom.select_mode(confidence=0.3)
        assert mode == ThinkingMode.SLOW

        # High complexity should trigger SLOW
        loom.reset()
        mode = loom.select_mode(complexity=0.9)
        assert mode == ThinkingMode.SLOW


# ==============================================================================
# ADDITIONAL EDGE CASES
# ==============================================================================


class TestAdditionalEdgeCases:
    """Additional edge case tests for full coverage."""

    def test_mode_controller_double_registration(self):
        """Registering same observer twice should not duplicate."""
        controller = ModeController()
        observer = MockObserver()

        controller.register_observer(observer)
        controller.register_observer(observer)  # Second registration

        assert len(controller._observers) == 1

    def test_mode_controller_unregister_nonexistent(self):
        """Unregistering non-existent observer should not error."""
        controller = ModeController()
        observer = MockObserver()

        # Should not raise
        controller.unregister_observer(observer)

    def test_force_mode_no_change(self):
        """force_mode with same mode should not create transition."""
        controller = ModeController()

        # Already in FAST mode
        controller.force_mode(ThinkingMode.FAST, reason="no change")

        # No transition should be recorded
        assert len(controller.get_transition_history()) == 0

    def test_loom_observability_with_none_signal(self):
        """Loom observability handles None signal in select_mode."""
        config = LoomConfig(enable_observability=True)
        loom = Loom(config)

        observer = MockObserver()
        loom.register_observer(observer)

        # Select mode with no signal
        mode = loom.select_mode(None)

        assert mode == ThinkingMode.FAST
        # Reason should mention "default"
        assert "default" in observer.mode_selections[0][1]
