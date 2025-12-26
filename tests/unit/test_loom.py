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
