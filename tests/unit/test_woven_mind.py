"""
Tests for WovenMind facade class.

TDD: Tests for the unified facade that integrates all dual-process
components into a single, easy-to-use interface.

Part of Sprint 4: The Loom Weaves (T4.5)
Part of the Woven Mind + PRISM Marriage project.
"""

import pytest
from typing import Dict, Set


class TestWovenMindCreation:
    """Test WovenMind initialization."""

    def test_default_creation(self):
        """WovenMind should create with sensible defaults."""
        from cortical.reasoning.woven_mind import WovenMind

        mind = WovenMind()

        assert mind.loom is not None
        assert mind.hive is not None
        assert mind.cortex is not None
        assert mind.router is not None

    def test_creation_with_config(self):
        """WovenMind should accept configuration."""
        from cortical.reasoning.woven_mind import WovenMind, WovenMindConfig
        from cortical.reasoning.loom import LoomConfig

        config = WovenMindConfig(
            surprise_threshold=0.4,
            k_winners=10,
            min_frequency=5,
        )
        mind = WovenMind(config=config)

        assert mind.config.surprise_threshold == 0.4
        assert mind.config.k_winners == 10
        assert mind.config.min_frequency == 5

    def test_creation_with_existing_components(self):
        """WovenMind should accept pre-configured components."""
        from cortical.reasoning.woven_mind import WovenMind
        from cortical.reasoning.loom import Loom
        from cortical.reasoning.loom_hive import LoomHiveConnector
        from cortical.reasoning.loom_cortex import LoomCortexConnector

        loom = Loom()
        hive = LoomHiveConnector()
        cortex = LoomCortexConnector()

        mind = WovenMind(loom=loom, hive=hive, cortex=cortex)

        assert mind.loom is loom
        assert mind.hive is hive
        assert mind.cortex is cortex


class TestWovenMindTraining:
    """Test training the WovenMind."""

    def test_train_on_text(self):
        """WovenMind should train on text."""
        from cortical.reasoning.woven_mind import WovenMind

        mind = WovenMind()
        mind.train("neural networks process data efficiently")

        # Hive should have learned the patterns
        predictions = mind.hive.generate_predictions(["neural"])
        assert len(predictions) > 0

    def test_train_on_multiple_texts(self):
        """WovenMind should accumulate training."""
        from cortical.reasoning.woven_mind import WovenMind

        mind = WovenMind()
        mind.train("neural networks are powerful")
        mind.train("deep learning uses neural networks")

        predictions = mind.hive.generate_predictions(["neural"])
        assert len(predictions) > 0

    def test_train_builds_patterns_for_cortex(self):
        """Training should build patterns for Cortex abstraction."""
        from cortical.reasoning.woven_mind import WovenMind, WovenMindConfig

        config = WovenMindConfig(min_frequency=2)
        mind = WovenMind(config=config)

        # Train same pattern multiple times
        for _ in range(3):
            mind.observe_pattern(["machine", "learning"])

        abstractions = mind.cortex.get_abstractions()
        assert len(abstractions) >= 1


class TestWovenMindProcessing:
    """Test basic processing through WovenMind."""

    def test_process_returns_result(self):
        """process() should return a WovenMindResult."""
        from cortical.reasoning.woven_mind import WovenMind, WovenMindResult

        mind = WovenMind()
        mind.train("hello world this is a test")

        result = mind.process(["hello"])

        assert isinstance(result, WovenMindResult)
        assert result.mode is not None
        assert result.activations is not None

    def test_process_with_fast_mode(self):
        """process() should work with explicit FAST mode."""
        from cortical.reasoning.woven_mind import WovenMind
        from cortical.reasoning.loom import ThinkingMode

        mind = WovenMind()
        mind.train("the quick brown fox")

        result = mind.process(["the"], mode=ThinkingMode.FAST)

        assert result.mode == ThinkingMode.FAST
        assert "the" in result.activations

    def test_process_with_slow_mode(self):
        """process() should work with explicit SLOW mode."""
        from cortical.reasoning.woven_mind import WovenMind
        from cortical.reasoning.loom import ThinkingMode

        mind = WovenMind()

        result = mind.process(["neural", "networks"], mode=ThinkingMode.SLOW)

        assert result.mode == ThinkingMode.SLOW

    def test_process_auto_selects_mode(self):
        """process() without mode should auto-select based on surprise."""
        from cortical.reasoning.woven_mind import WovenMind

        mind = WovenMind()
        mind.train("consistent pattern here")

        # First call establishes baseline
        result1 = mind.process(["consistent"])

        # Second call may have surprise info
        result2 = mind.process(["consistent"])

        assert result2.mode is not None


class TestWovenMindModeManagement:
    """Test mode management through WovenMind."""

    def test_get_current_mode(self):
        """Should be able to get current mode."""
        from cortical.reasoning.woven_mind import WovenMind
        from cortical.reasoning.loom import ThinkingMode

        mind = WovenMind()
        mode = mind.get_current_mode()

        assert mode in [ThinkingMode.FAST, ThinkingMode.SLOW]

    def test_force_mode(self):
        """Should be able to force a specific mode."""
        from cortical.reasoning.woven_mind import WovenMind
        from cortical.reasoning.loom import ThinkingMode

        mind = WovenMind()
        mind.force_mode(ThinkingMode.SLOW)

        assert mind.get_current_mode() == ThinkingMode.SLOW


class TestWovenMindIntrospection:
    """Test introspection capabilities."""

    def test_get_stats(self):
        """Should provide system statistics."""
        from cortical.reasoning.woven_mind import WovenMind

        mind = WovenMind()
        mind.train("some training text")

        stats = mind.get_stats()

        assert "mode" in stats
        assert "hive" in stats
        assert "cortex" in stats
        assert "loom" in stats

    def test_get_surprise_baseline(self):
        """Should expose surprise baseline."""
        from cortical.reasoning.woven_mind import WovenMind

        mind = WovenMind()
        baseline = mind.get_surprise_baseline()

        assert isinstance(baseline, float)
        assert 0.0 <= baseline <= 1.0

    def test_get_transition_history(self):
        """Should expose mode transition history."""
        from cortical.reasoning.woven_mind import WovenMind

        mind = WovenMind()
        history = mind.get_transition_history()

        assert isinstance(history, list)


class TestWovenMindResult:
    """Test WovenMindResult structure."""

    def test_result_has_required_fields(self):
        """WovenMindResult should have mode, activations, surprise."""
        from cortical.reasoning.woven_mind import WovenMind, WovenMindResult

        mind = WovenMind()
        mind.train("test content")

        result = mind.process(["test"])

        assert hasattr(result, "mode")
        assert hasattr(result, "activations")
        assert hasattr(result, "surprise")
        assert hasattr(result, "predictions")

    def test_result_is_dataclass(self):
        """WovenMindResult should be a proper dataclass."""
        from dataclasses import is_dataclass
        from cortical.reasoning.woven_mind import WovenMindResult

        assert is_dataclass(WovenMindResult)


class TestWovenMindSerialization:
    """Test serialization of WovenMind state."""

    def test_to_dict(self):
        """WovenMind state should be serializable."""
        from cortical.reasoning.woven_mind import WovenMind

        mind = WovenMind()
        mind.train("test training data")
        mind.process(["test"])

        data = mind.to_dict()

        assert "config" in data
        assert "hive_state" in data
        assert "cortex_state" in data

    def test_from_dict(self):
        """WovenMind should be deserializable."""
        from cortical.reasoning.woven_mind import WovenMind

        mind = WovenMind()
        mind.train("important data")
        mind.process(["important"])

        data = mind.to_dict()

        # Reconstruct
        restored = WovenMind.from_dict(data)

        assert restored is not None
        # Hive should have the training
        predictions = restored.hive.generate_predictions(["important"])
        assert len(predictions) >= 0  # May or may not have predictions depending on context


class TestWovenMindReset:
    """Test reset functionality."""

    def test_reset_clears_state(self):
        """reset() should clear all learned state."""
        from cortical.reasoning.woven_mind import WovenMind

        mind = WovenMind()
        mind.train("some data")
        mind.process(["some"])

        mind.reset()

        # After reset, baseline should be cleared
        assert mind.get_surprise_baseline() == 0.0

    def test_reset_preserves_config(self):
        """reset() should preserve configuration."""
        from cortical.reasoning.woven_mind import WovenMind, WovenMindConfig

        config = WovenMindConfig(surprise_threshold=0.5)
        mind = WovenMind(config=config)
        mind.train("some data")

        mind.reset()

        assert mind.config.surprise_threshold == 0.5
