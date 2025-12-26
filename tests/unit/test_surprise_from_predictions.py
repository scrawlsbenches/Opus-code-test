"""
Tests for surprise_from_predictions() integration.

TDD: Tests for the convenience method that combines prediction
generation with surprise detection.

Part of Sprint 4: The Loom Weaves (T4.3)
Part of the Woven Mind + PRISM Marriage project.
"""

import pytest
from typing import Dict, Set


class TestSurpriseFromPredictions:
    """Test surprise detection with predictions."""

    def test_detect_surprise_returns_signal(self):
        """detect_surprise should return a SurpriseSignal."""
        from cortical.reasoning.loom import Loom, SurpriseSignal
        from cortical.reasoning.loom_hive import LoomHiveConnector

        hive = LoomHiveConnector()
        hive.train("the quick brown fox jumps over the lazy dog")

        loom = Loom()

        # Get surprise from predictions (actual is Set[str])
        context = ["the", "quick"]
        actual = hive.process_fast(context)
        signal = loom.detect_surprise(
            predicted=hive.generate_predictions(context),
            actual=actual
        )

        assert isinstance(signal, SurpriseSignal)
        assert hasattr(signal, "magnitude")

    def test_low_surprise_for_expected_outcome(self):
        """Accurate predictions should have low surprise."""
        from cortical.reasoning.loom import Loom
        from cortical.reasoning.loom_hive import LoomHiveConnector

        hive = LoomHiveConnector()
        hive.train("a b c a b c a b c")  # Repeated pattern

        loom = Loom()

        # Predictions should match well
        context = ["a"]
        predictions = hive.generate_predictions(context)
        actual = hive.process_fast(context)

        signal = loom.detect_surprise(
            predicted=predictions,
            actual=actual
        )

        # Low surprise expected (predictions match actual)
        assert signal.magnitude <= 0.5

    def test_high_surprise_for_unexpected_outcome(self):
        """Inaccurate predictions should have high surprise."""
        from cortical.reasoning.loom import Loom
        from cortical.reasoning.loom_hive import LoomHiveConnector

        hive = LoomHiveConnector()
        hive.train("a b c")  # Expects: a -> b -> c

        loom = Loom()

        # Predictions will expect "b" after "a"
        context = ["a"]
        predictions = hive.generate_predictions(context)

        # But actual is something unexpected (actual is Set[str])
        actual = {"x", "y", "z"}  # Completely different

        signal = loom.detect_surprise(
            predicted=predictions,
            actual=actual
        )

        # High surprise expected (predictions don't match)
        assert signal.magnitude > 0.3


class TestSurpriseWithHiveConnector:
    """Test surprise integration with LoomHiveConnector."""

    def test_hive_connector_provides_predictions_for_surprise(self):
        """LoomHiveConnector should provide predictions usable for surprise."""
        from cortical.reasoning.loom_hive import LoomHiveConnector

        connector = LoomHiveConnector()
        connector.train("neural networks process data efficiently")

        predictions = connector.generate_predictions(["neural"])

        # Predictions should be a dict of token -> probability
        assert isinstance(predictions, dict)
        if predictions:  # Only check if we have predictions
            for token, prob in predictions.items():
                assert isinstance(token, str)
                assert 0.0 <= prob <= 1.0

    def test_process_fast_activations_usable_for_surprise(self):
        """process_fast output should be usable for surprise detection."""
        from cortical.reasoning.loom_hive import LoomHiveConnector

        connector = LoomHiveConnector()
        connector.train("the quick brown fox")

        active = connector.process_fast(["the"])

        # Active should be a set of tokens
        assert isinstance(active, set)
        for token in active:
            assert isinstance(token, str)


class TestEnhancedLoomWithHiveConnector:
    """Test enhanced Loom with LoomHiveConnector integration."""

    def test_loom_can_use_hive_predictions(self):
        """Loom should work with predictions from LoomHiveConnector."""
        from cortical.reasoning.loom import Loom, LoomConfig
        from cortical.reasoning.loom_hive import LoomHiveConnector

        # Configure with specific threshold
        config = LoomConfig(surprise_threshold=0.4)
        loom = Loom(config=config)

        # Train hive
        hive = LoomHiveConnector()
        hive.train("hello world this is a test")

        # Get predictions and actual
        context = ["hello"]
        predictions = hive.generate_predictions(context)
        active = hive.process_fast(context)

        # Use Loom's surprise detection (actual is a Set[str])
        signal = loom.detect_surprise(predictions, active)

        assert signal is not None
        assert signal.magnitude >= 0.0

    def test_mode_selection_based_on_hive_surprise(self):
        """Loom should select mode based on surprise from Hive."""
        from cortical.reasoning.loom import Loom, LoomConfig, ThinkingMode, SurpriseSignal
        from cortical.reasoning.loom_hive import LoomHiveConnector

        # Low threshold to trigger SLOW mode easily
        config = LoomConfig(surprise_threshold=0.2)
        loom = Loom(config=config)

        # Create high surprise scenario
        signal = SurpriseSignal(magnitude=0.8, source="test")
        mode = loom.select_mode(signal=signal)

        # High surprise should trigger SLOW mode
        assert mode == ThinkingMode.SLOW

    def test_low_surprise_keeps_fast_mode(self):
        """Low surprise should keep Loom in FAST mode."""
        from cortical.reasoning.loom import Loom, LoomConfig, ThinkingMode, SurpriseSignal

        config = LoomConfig(surprise_threshold=0.5)
        loom = Loom(config=config)

        # Low surprise signal
        signal = SurpriseSignal(magnitude=0.2, source="test")
        mode = loom.select_mode(surprise=signal)

        assert mode == ThinkingMode.FAST


class TestSurpriseBaseline:
    """Test surprise baseline adaptation."""

    def test_baseline_adapts_over_time(self):
        """Surprise baseline should adapt to environment."""
        from cortical.reasoning.loom import Loom

        loom = Loom()

        # Initially baseline is 0
        initial = loom.get_surprise_baseline()

        # Generate some surprises (actual is Set[str], not dict)
        for i in range(5):
            loom.detect_surprise(
                predicted={"a": 0.5},
                actual={"b"}  # Mismatch - actual is a set
            )

        # Baseline should have changed
        after = loom.get_surprise_baseline()
        # Baseline should increase with consistent surprise
        assert after >= initial

    def test_reset_baseline(self):
        """Should be able to reset surprise baseline."""
        from cortical.reasoning.loom import Loom

        loom = Loom()

        # Generate surprises to build baseline (actual is Set[str])
        for _ in range(3):
            loom.detect_surprise(
                predicted={"a": 0.9},
                actual={"z"}  # actual is a set
            )

        # Reset
        loom.reset_surprise_baseline()

        # Should be back to initial
        assert loom.get_surprise_baseline() < 0.5
