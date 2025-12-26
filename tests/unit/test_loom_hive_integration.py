"""
Tests for Loom-Hive integration (T4.1: Connect Loom to enhanced Hive).

The integration should:
1. Wrap PRISM-SLM and HomeostasisRegulator
2. Provide predictions for surprise detection
3. Process input in FAST mode through the Hive
4. Support k-winners-take-all and lateral inhibition
"""

import pytest
from typing import Dict, Set

from cortical.reasoning.loom import Loom, LoomConfig, ThinkingMode, SurpriseSignal
from cortical.reasoning.prism_slm import PRISMLanguageModel, HiveNode
from cortical.reasoning.homeostasis import HomeostasisRegulator, HomeostasisConfig


class TestLoomHiveConnectorCreation:
    """Test LoomHiveConnector initialization."""

    def test_connector_with_defaults(self):
        """Connector should initialize with default PRISM-SLM."""
        from cortical.reasoning.loom_hive import LoomHiveConnector

        connector = LoomHiveConnector()
        assert connector.model is not None
        assert connector.regulator is not None

    def test_connector_with_custom_model(self):
        """Connector should accept custom PRISM-SLM instance."""
        from cortical.reasoning.loom_hive import LoomHiveConnector

        model = PRISMLanguageModel(context_size=5)
        connector = LoomHiveConnector(model=model)
        assert connector.model is model

    def test_connector_with_homeostasis(self):
        """Connector should use HomeostasisRegulator."""
        from cortical.reasoning.loom_hive import LoomHiveConnector

        config = HomeostasisConfig(target_activation=0.10)
        regulator = HomeostasisRegulator(config)
        connector = LoomHiveConnector(regulator=regulator)
        assert connector.regulator is regulator


class TestHiveProcessing:
    """Test FAST mode processing through the Hive."""

    def test_process_fast_returns_activations(self):
        """process_fast should return set of active nodes."""
        from cortical.reasoning.loom_hive import LoomHiveConnector

        connector = LoomHiveConnector()

        # Train with some data
        connector.train("neural networks learn patterns")

        # Process and get activations
        activations = connector.process_fast(["neural"])
        assert isinstance(activations, set)
        assert len(activations) > 0

    def test_process_fast_applies_lateral_inhibition(self):
        """process_fast should apply lateral inhibition for sparse output."""
        from cortical.reasoning.loom_hive import LoomHiveConnector

        connector = LoomHiveConnector(k_winners=3)
        connector.train("the quick brown fox jumps over the lazy dog")

        # Process - should get at most k winners
        activations = connector.process_fast(["the"])
        assert len(activations) <= 3

    def test_process_fast_updates_homeostasis(self):
        """process_fast should update homeostatic regulation."""
        from cortical.reasoning.loom_hive import LoomHiveConnector

        connector = LoomHiveConnector()
        connector.train("neural networks learn patterns")

        # Process multiple times
        for _ in range(10):
            connector.process_fast(["neural"])

        # Check that homeostasis tracked this
        stats = connector.get_homeostasis_stats()
        assert stats["total_nodes_tracked"] > 0


class TestPredictionGeneration:
    """Test prediction generation for surprise detection."""

    def test_generate_predictions(self):
        """Connector should generate predictions from context."""
        from cortical.reasoning.loom_hive import LoomHiveConnector

        connector = LoomHiveConnector()
        connector.train("neural networks process information efficiently")

        predictions = connector.generate_predictions(["neural", "networks"])
        assert isinstance(predictions, dict)
        # Should have some non-zero predictions
        assert any(p > 0 for p in predictions.values())

    def test_predictions_are_probabilities(self):
        """Predictions should be probabilities (0-1, sum to ~1)."""
        from cortical.reasoning.loom_hive import LoomHiveConnector

        connector = LoomHiveConnector()
        connector.train("the quick brown fox jumps")

        predictions = connector.generate_predictions(["the", "quick"])
        for prob in predictions.values():
            assert 0.0 <= prob <= 1.0

    def test_predictions_match_model_transitions(self):
        """Predictions should match PRISM-SLM transition probabilities."""
        from cortical.reasoning.loom_hive import LoomHiveConnector

        connector = LoomHiveConnector()
        connector.train("a b c a b c a b c")  # Strong transition pattern

        predictions = connector.generate_predictions(["a", "b"])
        # c should be predicted after a b
        assert "c" in predictions
        assert predictions["c"] > 0.5


class TestLoomIntegration:
    """Test integration with Loom for surprise-based mode switching."""

    def test_connector_works_with_loom(self):
        """Connector predictions should work with Loom.detect_surprise()."""
        from cortical.reasoning.loom_hive import LoomHiveConnector

        connector = LoomHiveConnector()
        loom = Loom()

        connector.train("the quick brown fox")

        # Generate predictions
        predictions = connector.generate_predictions(["the", "quick"])

        # Simulate actual activations
        actual = connector.process_fast(["quick"])

        # Use Loom to detect surprise
        signal = loom.detect_surprise(predictions, actual)
        assert isinstance(signal, SurpriseSignal)

    def test_accurate_prediction_low_surprise(self):
        """Accurate predictions should result in low surprise."""
        from cortical.reasoning.loom_hive import LoomHiveConnector

        connector = LoomHiveConnector()
        loom = Loom()

        # Train with very clear pattern
        for _ in range(10):
            connector.train("a b c a b c a b c")

        # Predict after "a b" - should predict "c"
        predictions = connector.generate_predictions(["a", "b"])

        # Actual is what model predicts
        actual = {"c"}

        signal = loom.detect_surprise(predictions, actual)
        assert signal.magnitude < 0.5  # Low surprise

    def test_inaccurate_prediction_high_surprise(self):
        """Inaccurate predictions should result in high surprise."""
        from cortical.reasoning.loom_hive import LoomHiveConnector

        connector = LoomHiveConnector()
        loom = Loom()

        connector.train("a b c d e f")

        # Predict after "a b" - expects "c"
        predictions = connector.generate_predictions(["a", "b"])

        # Actual is completely unexpected
        actual = {"xyz_unexpected"}

        signal = loom.detect_surprise(predictions, actual)
        assert signal.magnitude > 0.5  # High surprise


class TestHiveNodeIntegration:
    """Test HiveNode integration with connector."""

    def test_get_hive_node(self):
        """Connector should provide access to HiveNodes."""
        from cortical.reasoning.loom_hive import LoomHiveConnector

        connector = LoomHiveConnector()
        connector.train("neural networks")

        node = connector.get_hive_node("neural")
        assert node is not None
        assert isinstance(node, HiveNode)

    def test_hive_nodes_track_activations(self):
        """HiveNodes should track activation history."""
        from cortical.reasoning.loom_hive import LoomHiveConnector

        connector = LoomHiveConnector()
        connector.train("neural networks")

        # Process multiple times
        for _ in range(5):
            connector.process_fast(["neural"])

        node = connector.get_hive_node("neural")
        assert node.activation_count >= 5


class TestSpreadingActivation:
    """Test spreading activation through the Hive."""

    def test_spread_activation(self):
        """Connector should support spreading activation."""
        from cortical.reasoning.loom_hive import LoomHiveConnector

        connector = LoomHiveConnector()
        connector.train("neural networks process data efficiently")

        # Activate "neural" and spread
        activations = connector.spread_activation(
            seeds=["neural"],
            steps=2,
            decay=0.5,
        )

        # Should spread to connected nodes
        assert "neural" in activations
        assert len(activations) > 1


class TestLoomHiveConnectorSerialization:
    """Test serialization of connector state."""

    def test_to_dict(self):
        """Connector should serialize to dictionary."""
        from cortical.reasoning.loom_hive import LoomHiveConnector

        connector = LoomHiveConnector(k_winners=5)
        connector.train("test data here")

        data = connector.to_dict()
        assert "k_winners" in data
        assert data["k_winners"] == 5
        assert "model_graph_state" in data

    def test_from_dict(self):
        """Connector should deserialize from dictionary."""
        from cortical.reasoning.loom_hive import LoomHiveConnector

        original = LoomHiveConnector(k_winners=5)
        original.train("test data for serialization")

        data = original.to_dict()
        restored = LoomHiveConnector.from_dict(data)

        assert restored.k_winners == 5
        # Model should have same vocabulary
        assert restored.model.vocab_size > 0
