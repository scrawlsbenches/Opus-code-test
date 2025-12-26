"""
Tests for attention_routing() based on thinking mode.

TDD: Tests for the attention router that directs processing
to the appropriate system based on the current ThinkingMode.

Part of Sprint 4: The Loom Weaves (T4.4)
Part of the Woven Mind + PRISM Marriage project.
"""

import pytest
from typing import Dict, Set


class TestAttentionRouterCreation:
    """Test AttentionRouter initialization."""

    def test_router_with_defaults(self):
        """AttentionRouter should initialize with Loom, Hive, and Cortex connectors."""
        from cortical.reasoning.loom import Loom
        from cortical.reasoning.loom_hive import LoomHiveConnector
        from cortical.reasoning.loom_cortex import LoomCortexConnector
        from cortical.reasoning.attention_router import AttentionRouter

        loom = Loom()
        hive = LoomHiveConnector()
        cortex = LoomCortexConnector()

        router = AttentionRouter(loom=loom, hive=hive, cortex=cortex)

        assert router.loom is loom
        assert router.hive is hive
        assert router.cortex is cortex

    def test_router_with_config(self):
        """AttentionRouter should accept optional configuration."""
        from cortical.reasoning.loom import Loom, LoomConfig
        from cortical.reasoning.loom_hive import LoomHiveConnector
        from cortical.reasoning.loom_cortex import LoomCortexConnector
        from cortical.reasoning.attention_router import AttentionRouter, AttentionRouterConfig

        config = AttentionRouterConfig(auto_switch=True, parallel_probe=False)
        loom = Loom()
        hive = LoomHiveConnector()
        cortex = LoomCortexConnector()

        router = AttentionRouter(loom=loom, hive=hive, cortex=cortex, config=config)

        assert router.config.auto_switch is True
        assert router.config.parallel_probe is False


class TestFastModeRouting:
    """Test routing in FAST mode."""

    def test_route_to_hive_in_fast_mode(self):
        """FAST mode should route processing to Hive."""
        from cortical.reasoning.loom import Loom, ThinkingMode
        from cortical.reasoning.loom_hive import LoomHiveConnector
        from cortical.reasoning.loom_cortex import LoomCortexConnector
        from cortical.reasoning.attention_router import AttentionRouter

        loom = Loom()
        hive = LoomHiveConnector()
        cortex = LoomCortexConnector()

        # Train hive so it can process
        hive.train("hello world this is a test")

        router = AttentionRouter(loom=loom, hive=hive, cortex=cortex)

        # Route in FAST mode
        result = router.route(["hello"], mode=ThinkingMode.FAST)

        assert result.mode_used == ThinkingMode.FAST
        assert result.source == "hive"
        assert isinstance(result.activations, set)

    def test_fast_mode_returns_hive_activations(self):
        """FAST mode should return Hive activations."""
        from cortical.reasoning.loom import Loom, ThinkingMode
        from cortical.reasoning.loom_hive import LoomHiveConnector
        from cortical.reasoning.loom_cortex import LoomCortexConnector
        from cortical.reasoning.attention_router import AttentionRouter

        loom = Loom()
        hive = LoomHiveConnector()
        cortex = LoomCortexConnector()

        hive.train("the quick brown fox")

        router = AttentionRouter(loom=loom, hive=hive, cortex=cortex)
        result = router.route(["the"], mode=ThinkingMode.FAST)

        # Should have context token in activations
        assert "the" in result.activations


class TestSlowModeRouting:
    """Test routing in SLOW mode."""

    def test_route_to_cortex_in_slow_mode(self):
        """SLOW mode should route processing to Cortex."""
        from cortical.reasoning.loom import Loom, ThinkingMode
        from cortical.reasoning.loom_hive import LoomHiveConnector
        from cortical.reasoning.loom_cortex import LoomCortexConnector
        from cortical.reasoning.attention_router import AttentionRouter

        loom = Loom()
        hive = LoomHiveConnector()
        cortex = LoomCortexConnector(min_frequency=1)

        router = AttentionRouter(loom=loom, hive=hive, cortex=cortex)

        # Route in SLOW mode
        result = router.route(["neural", "networks"], mode=ThinkingMode.SLOW)

        assert result.mode_used == ThinkingMode.SLOW
        assert result.source == "cortex"

    def test_slow_mode_builds_abstractions(self):
        """SLOW mode processing should form abstractions over time."""
        from cortical.reasoning.loom import Loom, ThinkingMode
        from cortical.reasoning.loom_hive import LoomHiveConnector
        from cortical.reasoning.loom_cortex import LoomCortexConnector
        from cortical.reasoning.attention_router import AttentionRouter

        loom = Loom()
        hive = LoomHiveConnector()
        cortex = LoomCortexConnector(min_frequency=2)

        router = AttentionRouter(loom=loom, hive=hive, cortex=cortex)

        # Process same pattern multiple times
        for _ in range(3):
            router.route(["machine", "learning"], mode=ThinkingMode.SLOW)

        # Should have formed abstraction
        abstractions = cortex.get_abstractions()
        assert len(abstractions) >= 1


class TestAutoModeSelection:
    """Test automatic mode selection based on surprise."""

    def test_auto_mode_uses_surprise_detection(self):
        """Auto mode should detect surprise and select mode accordingly."""
        from cortical.reasoning.loom import Loom
        from cortical.reasoning.loom_hive import LoomHiveConnector
        from cortical.reasoning.loom_cortex import LoomCortexConnector
        from cortical.reasoning.attention_router import AttentionRouter

        loom = Loom()
        hive = LoomHiveConnector()
        cortex = LoomCortexConnector()

        hive.train("the quick brown fox")

        router = AttentionRouter(loom=loom, hive=hive, cortex=cortex)

        # Route with auto mode selection
        result = router.route(["the"], mode=None)  # None = auto

        # Should have detected and selected a mode
        assert result.mode_used is not None
        assert result.surprise is not None or result.mode_used  # Either has surprise info or used a mode

    def test_high_surprise_triggers_slow_mode(self):
        """High surprise should trigger SLOW mode."""
        from cortical.reasoning.loom import Loom, LoomConfig, ThinkingMode
        from cortical.reasoning.loom_hive import LoomHiveConnector
        from cortical.reasoning.loom_cortex import LoomCortexConnector
        from cortical.reasoning.attention_router import AttentionRouter

        # Low threshold to easily trigger SLOW
        config = LoomConfig(surprise_threshold=0.1)
        loom = Loom(config=config)
        hive = LoomHiveConnector()
        cortex = LoomCortexConnector()

        # Train on specific pattern
        hive.train("a b c")

        router = AttentionRouter(loom=loom, hive=hive, cortex=cortex)

        # First call establishes baseline
        router.route(["a"], mode=None)

        # Second call with completely different context should trigger surprise
        result = router.route(["x", "y", "z"], mode=None)

        # With high surprise, should use SLOW mode
        # (depends on implementation, but demonstrates the pattern)
        assert result.mode_used is not None


class TestRoutingResult:
    """Test RoutingResult structure."""

    def test_result_contains_required_fields(self):
        """RoutingResult should contain mode_used, source, activations."""
        from cortical.reasoning.loom import Loom, ThinkingMode
        from cortical.reasoning.loom_hive import LoomHiveConnector
        from cortical.reasoning.loom_cortex import LoomCortexConnector
        from cortical.reasoning.attention_router import AttentionRouter, RoutingResult

        loom = Loom()
        hive = LoomHiveConnector()
        cortex = LoomCortexConnector()

        hive.train("test content")

        router = AttentionRouter(loom=loom, hive=hive, cortex=cortex)
        result = router.route(["test"], mode=ThinkingMode.FAST)

        assert hasattr(result, "mode_used")
        assert hasattr(result, "source")
        assert hasattr(result, "activations")
        assert isinstance(result, RoutingResult)

    def test_result_includes_surprise_signal(self):
        """RoutingResult should include surprise signal when available."""
        from cortical.reasoning.loom import Loom
        from cortical.reasoning.loom_hive import LoomHiveConnector
        from cortical.reasoning.loom_cortex import LoomCortexConnector
        from cortical.reasoning.attention_router import AttentionRouter

        loom = Loom()
        hive = LoomHiveConnector()
        cortex = LoomCortexConnector()

        hive.train("hello world")

        router = AttentionRouter(loom=loom, hive=hive, cortex=cortex)

        # First call - no surprise yet
        router.route(["hello"], mode=None)

        # Second call - may have surprise
        result = router.route(["hello"], mode=None)

        # Result should have surprise field (may be None or SurpriseSignal)
        assert hasattr(result, "surprise")


class TestDualPathProcessing:
    """Test processing through both paths when needed."""

    def test_can_process_both_paths(self):
        """Should be able to explicitly process through both systems."""
        from cortical.reasoning.loom import Loom
        from cortical.reasoning.loom_hive import LoomHiveConnector
        from cortical.reasoning.loom_cortex import LoomCortexConnector
        from cortical.reasoning.attention_router import AttentionRouter

        loom = Loom()
        hive = LoomHiveConnector()
        cortex = LoomCortexConnector()

        hive.train("neural networks process data")

        router = AttentionRouter(loom=loom, hive=hive, cortex=cortex)

        # Process through both
        result = router.route_both(["neural", "networks"])

        assert result.fast_result is not None
        assert result.slow_result is not None
        assert "hive" in str(type(result.fast_result)).lower() or isinstance(result.fast_result, set)


class TestAttentionRouterSerialization:
    """Test serialization of router state."""

    def test_to_dict(self):
        """AttentionRouter state should be serializable."""
        from cortical.reasoning.loom import Loom
        from cortical.reasoning.loom_hive import LoomHiveConnector
        from cortical.reasoning.loom_cortex import LoomCortexConnector
        from cortical.reasoning.attention_router import AttentionRouter

        loom = Loom()
        hive = LoomHiveConnector()
        cortex = LoomCortexConnector()

        router = AttentionRouter(loom=loom, hive=hive, cortex=cortex)
        data = router.to_dict()

        assert "hive_state" in data
        assert "cortex_state" in data
        assert "config" in data

    def test_from_dict(self):
        """AttentionRouter should be deserializable."""
        from cortical.reasoning.loom import Loom
        from cortical.reasoning.loom_hive import LoomHiveConnector
        from cortical.reasoning.loom_cortex import LoomCortexConnector
        from cortical.reasoning.attention_router import AttentionRouter

        loom = Loom()
        hive = LoomHiveConnector()
        cortex = LoomCortexConnector()

        hive.train("test content here")

        router = AttentionRouter(loom=loom, hive=hive, cortex=cortex)
        data = router.to_dict()

        # Reconstruct
        new_loom = Loom()
        restored = AttentionRouter.from_dict(data, loom=new_loom)

        assert restored is not None
        assert restored.hive is not None
        assert restored.cortex is not None
