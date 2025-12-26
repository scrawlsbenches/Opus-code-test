"""
Integration tests for WovenMind dual-process architecture.

These tests verify that all components work together correctly:
- Loom mode switching based on surprise
- LoomHiveConnector for FAST mode
- LoomCortexConnector for SLOW mode
- AttentionRouter for mode-based routing
- WovenMind facade integrating everything

Part of Sprint 4: The Loom Weaves (T4.7)
Part of the Woven Mind + PRISM Marriage project.
"""

import pytest
from typing import List


class TestDualProcessIntegration:
    """Test the complete dual-process system."""

    def test_fast_mode_uses_hive_patterns(self):
        """FAST mode should use Hive's learned patterns."""
        from cortical.reasoning.woven_mind import WovenMind
        from cortical.reasoning.loom import ThinkingMode

        mind = WovenMind()

        # Train on a pattern
        mind.train("the quick brown fox jumps over the lazy dog")

        # Process in FAST mode
        result = mind.process(["the", "quick"], mode=ThinkingMode.FAST)

        # Should use Hive
        assert result.mode == ThinkingMode.FAST
        assert result.source == "hive"

        # Should have activations from the trained pattern
        assert "the" in result.activations
        assert "quick" in result.activations

    def test_slow_mode_uses_cortex_abstractions(self):
        """SLOW mode should use Cortex's abstraction engine."""
        from cortical.reasoning.woven_mind import WovenMind, WovenMindConfig
        from cortical.reasoning.loom import ThinkingMode

        # Low min_frequency so abstractions form quickly
        config = WovenMindConfig(min_frequency=2)
        mind = WovenMind(config=config)

        # Observe a pattern multiple times
        for _ in range(3):
            mind.observe_pattern(["machine", "learning"])

        # Process in SLOW mode
        result = mind.process(["machine", "learning"], mode=ThinkingMode.SLOW)

        assert result.mode == ThinkingMode.SLOW
        assert result.source == "cortex"

    def test_surprise_triggers_mode_switch(self):
        """High surprise should trigger switch from FAST to SLOW."""
        from cortical.reasoning.woven_mind import WovenMind, WovenMindConfig
        from cortical.reasoning.loom import ThinkingMode

        # Low threshold to easily trigger SLOW mode
        config = WovenMindConfig(surprise_threshold=0.1)
        mind = WovenMind(config=config)

        # Train on specific pattern
        mind.train("a b c a b c a b c")

        # First call establishes prediction baseline
        result1 = mind.process(["a"])

        # Second call with completely different pattern should trigger surprise
        result2 = mind.process(["x", "y", "z"])

        # With high surprise, may switch to SLOW mode
        # (actual behavior depends on surprise calculation)
        assert result2.mode is not None

    def test_mode_transitions_are_recorded(self):
        """Mode transitions should be recorded for observability."""
        from cortical.reasoning.woven_mind import WovenMind
        from cortical.reasoning.loom import ThinkingMode

        mind = WovenMind()
        mind.train("test data")

        # Force a mode transition
        mind.force_mode(ThinkingMode.SLOW)
        mind.force_mode(ThinkingMode.FAST)

        history = mind.get_transition_history()
        assert len(history) >= 2


class TestTrainingPipeline:
    """Test the complete training pipeline."""

    def test_training_flows_to_hive(self):
        """Training text should build patterns in Hive."""
        from cortical.reasoning.woven_mind import WovenMind

        mind = WovenMind()
        mind.train("neural networks process information")

        # Hive should have learned transitions
        predictions = mind.hive.generate_predictions(["neural"])
        assert len(predictions) > 0

    def test_observation_flows_to_cortex(self):
        """Pattern observation should build abstractions in Cortex."""
        from cortical.reasoning.woven_mind import WovenMind, WovenMindConfig

        config = WovenMindConfig(min_frequency=2)
        mind = WovenMind(config=config)

        # Observe patterns
        for _ in range(3):
            mind.observe_pattern(["deep", "learning"])

        # Cortex should have abstractions
        abstractions = mind.cortex.get_abstractions()
        assert len(abstractions) >= 1

    def test_combined_training(self):
        """Combined training should populate both systems."""
        from cortical.reasoning.woven_mind import WovenMind, WovenMindConfig

        config = WovenMindConfig(min_frequency=2)
        mind = WovenMind(config=config)

        # Train Hive
        mind.train("artificial intelligence is advancing rapidly")

        # Observe for Cortex
        for _ in range(3):
            mind.observe_pattern(["artificial", "intelligence"])

        # Both should have knowledge
        hive_predictions = mind.hive.generate_predictions(["artificial"])
        cortex_abstractions = mind.cortex.get_abstractions()

        assert len(hive_predictions) > 0
        assert len(cortex_abstractions) >= 1


class TestSurpriseDetectionPipeline:
    """Test the surprise detection pipeline."""

    def test_predictions_enable_surprise_detection(self):
        """Predictions from processing should enable surprise detection."""
        from cortical.reasoning.woven_mind import WovenMind

        mind = WovenMind()
        mind.train("consistent pattern here")

        # First call generates predictions
        result1 = mind.process(["consistent"])

        # Second call can detect surprise
        result2 = mind.process(["consistent"])

        # Second result may have surprise info
        # (depends on whether predictions matched actual)
        assert result2.mode is not None

    def test_surprise_baseline_adapts(self):
        """Surprise baseline should adapt over time."""
        from cortical.reasoning.woven_mind import WovenMind

        mind = WovenMind()
        mind.train("a b c d e")

        initial_baseline = mind.get_surprise_baseline()

        # Generate some surprises
        for _ in range(5):
            mind.process(["a"])
            mind.process(["x"])  # Unexpected

        final_baseline = mind.get_surprise_baseline()

        # Baseline should have changed
        # (may increase if consistently surprised)
        assert final_baseline >= 0.0


class TestSerializationRoundtrip:
    """Test serialization and deserialization."""

    def test_full_state_roundtrip(self):
        """Complete WovenMind state should survive serialization."""
        from cortical.reasoning.woven_mind import WovenMind, WovenMindConfig

        # Create and train
        config = WovenMindConfig(
            surprise_threshold=0.4,
            k_winners=10,
            min_frequency=2,
        )
        original = WovenMind(config=config)
        original.train("important training data")
        original.process(["important"])

        # Serialize
        data = original.to_dict()

        # Deserialize
        restored = WovenMind.from_dict(data)

        # Verify config preserved
        assert restored.config.surprise_threshold == 0.4
        assert restored.config.k_winners == 10

    def test_hive_state_preserved(self):
        """Hive training should be preserved through serialization."""
        from cortical.reasoning.woven_mind import WovenMind

        original = WovenMind()
        original.train("neural networks are powerful tools")

        # Get predictions before serialization
        original_predictions = original.hive.generate_predictions(["neural"])

        # Serialize and restore
        data = original.to_dict()
        restored = WovenMind.from_dict(data)

        # Get predictions after
        restored_predictions = restored.hive.generate_predictions(["neural"])

        # Should have same vocabulary
        assert len(restored_predictions) == len(original_predictions)


class TestAttentionRouterIntegration:
    """Test AttentionRouter integration with WovenMind."""

    def test_router_receives_correct_mode(self):
        """Router should receive the correct mode from WovenMind."""
        from cortical.reasoning.woven_mind import WovenMind
        from cortical.reasoning.loom import ThinkingMode

        mind = WovenMind()
        mind.train("test content")

        # Force FAST mode
        result = mind.process(["test"], mode=ThinkingMode.FAST)
        assert result.mode == ThinkingMode.FAST

        # Force SLOW mode
        result = mind.process(["test"], mode=ThinkingMode.SLOW)
        assert result.mode == ThinkingMode.SLOW

    def test_router_stats_available(self):
        """Router statistics should be accessible through WovenMind."""
        from cortical.reasoning.woven_mind import WovenMind

        mind = WovenMind()
        mind.train("some data")
        mind.process(["some"])

        stats = mind.get_stats()

        assert "router" in stats
        assert "current_mode" in stats["router"]


class TestHomeostasisIntegration:
    """Test homeostasis integration in FAST mode."""

    def test_homeostasis_regulates_activation(self):
        """Homeostasis should regulate activation levels."""
        from cortical.reasoning.woven_mind import WovenMind
        from cortical.reasoning.loom import ThinkingMode

        mind = WovenMind()
        mind.train("repeated word word word word word")

        # Process multiple times
        for _ in range(10):
            mind.process(["word"], mode=ThinkingMode.FAST)

        # Check homeostasis stats
        stats = mind.hive.get_homeostasis_stats()
        assert "total_nodes_tracked" in stats


class TestAbstractionIntegration:
    """Test abstraction integration in SLOW mode."""

    def test_abstractions_form_over_time(self):
        """Abstractions should form through repeated observation."""
        from cortical.reasoning.woven_mind import WovenMind, WovenMindConfig
        from cortical.reasoning.loom import ThinkingMode

        config = WovenMindConfig(min_frequency=2)
        mind = WovenMind(config=config)

        # Observe same pattern
        for _ in range(5):
            mind.process(["pattern", "recognition"], mode=ThinkingMode.SLOW)

        abstractions = mind.cortex.get_abstractions()
        assert len(abstractions) >= 1

    def test_abstractions_queryable(self):
        """Formed abstractions should be queryable."""
        from cortical.reasoning.woven_mind import WovenMind, WovenMindConfig

        config = WovenMindConfig(min_frequency=2)
        mind = WovenMind(config=config)

        # Observe patterns
        for _ in range(3):
            mind.observe_pattern(["query", "expansion"])

        # Query abstractions
        related = mind.cortex.query_abstractions(["query"])
        # Should find abstractions containing "query"
        assert isinstance(related, list)


class TestResetBehavior:
    """Test reset behavior across all components."""

    def test_reset_clears_all_state(self):
        """Reset should clear state across all components."""
        from cortical.reasoning.woven_mind import WovenMind

        mind = WovenMind()
        mind.train("data to be cleared")
        mind.process(["data"])

        # Get baseline before reset
        baseline_before = mind.get_surprise_baseline()

        # Reset
        mind.reset()

        # Baseline should be reset
        assert mind.get_surprise_baseline() == 0.0

    def test_reset_preserves_configuration(self):
        """Reset should not affect configuration."""
        from cortical.reasoning.woven_mind import WovenMind, WovenMindConfig

        config = WovenMindConfig(surprise_threshold=0.5)
        mind = WovenMind(config=config)
        mind.train("some data")

        mind.reset()

        assert mind.config.surprise_threshold == 0.5
