"""
WovenMind: Unified Facade for Dual-Process Cognitive Architecture.

The WovenMind class provides a simple, unified interface to the
dual-process architecture integrating:
- The Loom (mode switching based on surprise)
- LoomHiveConnector (FAST mode - Hebbian pattern matching)
- LoomCortexConnector (SLOW mode - deliberate abstraction)
- AttentionRouter (mode-based routing)

Part of Sprint 4: The Loom Weaves (T4.5)
Part of the Woven Mind + PRISM Marriage project.

Example:
    >>> from cortical.reasoning.woven_mind import WovenMind
    >>>
    >>> # Create with defaults
    >>> mind = WovenMind()
    >>>
    >>> # Train on text
    >>> mind.train("neural networks process data efficiently")
    >>>
    >>> # Process input
    >>> result = mind.process(["neural", "networks"])
    >>> print(f"Mode: {result.mode}, Activations: {result.activations}")
    >>>
    >>> # Force slow thinking
    >>> from cortical.reasoning.loom import ThinkingMode
    >>> result = mind.process(["complex", "problem"], mode=ThinkingMode.SLOW)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .loom import (
    Loom,
    LoomConfig,
    ModeTransition,
    SurpriseSignal,
    ThinkingMode,
)
from .loom_hive import LoomHiveConnector, LoomHiveConfig
from .loom_cortex import LoomCortexConnector, LoomCortexConfig
from .attention_router import AttentionRouter, AttentionRouterConfig


@dataclass
class WovenMindConfig:
    """Configuration for WovenMind.

    Unified configuration that controls all subsystems.

    Attributes:
        surprise_threshold: Surprise level to trigger SLOW mode (0-1).
        k_winners: Number of winners in lateral inhibition.
        min_frequency: Minimum observations for abstraction formation.
        auto_switch: Whether to auto-switch modes based on surprise.
        enable_observability: Whether to emit events for monitoring.
    """
    surprise_threshold: float = 0.3
    k_winners: int = 5
    min_frequency: int = 3
    auto_switch: bool = True
    enable_observability: bool = True


@dataclass
class WovenMindResult:
    """Result of processing through WovenMind.

    Attributes:
        mode: The thinking mode used for processing.
        activations: Set of activated node IDs.
        surprise: Surprise signal if detected.
        predictions: Predictions generated (if any).
        source: Which system produced the result ("hive" or "cortex").
        metadata: Additional processing metadata.
    """
    mode: ThinkingMode
    activations: Set[str]
    surprise: Optional[SurpriseSignal] = None
    predictions: Optional[Dict[str, float]] = None
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


class WovenMind:
    """
    Unified facade for the dual-process cognitive architecture.

    WovenMind integrates all components of the Woven Mind + PRISM marriage:
    - Loom: Mode switching based on surprise detection
    - LoomHiveConnector: FAST mode (Hebbian Hive - pattern matching)
    - LoomCortexConnector: SLOW mode (Cultured Cortex - abstraction)
    - AttentionRouter: Intelligent routing between systems

    The facade provides a simple interface for:
    - Training on text data
    - Processing input with automatic or explicit mode selection
    - Observing patterns for abstraction formation
    - Introspection and state management

    Example:
        >>> mind = WovenMind()
        >>> mind.train("the quick brown fox jumps over the lazy dog")
        >>> result = mind.process(["quick", "brown"])
        >>> print(f"Mode: {result.mode}")
    """

    def __init__(
        self,
        config: Optional[WovenMindConfig] = None,
        loom: Optional[Loom] = None,
        hive: Optional[LoomHiveConnector] = None,
        cortex: Optional[LoomCortexConnector] = None,
    ) -> None:
        """
        Initialize WovenMind.

        Args:
            config: Configuration for all subsystems.
            loom: Pre-configured Loom (creates default if not provided).
            hive: Pre-configured LoomHiveConnector (creates default if not provided).
            cortex: Pre-configured LoomCortexConnector (creates default if not provided).
        """
        self.config = config or WovenMindConfig()

        # Create component configurations
        loom_config = LoomConfig(
            surprise_threshold=self.config.surprise_threshold,
            enable_observability=self.config.enable_observability,
        )
        hive_config = LoomHiveConfig(
            k_winners=self.config.k_winners,
        )
        cortex_config = LoomCortexConfig(
            min_frequency=self.config.min_frequency,
        )

        # Initialize components
        self.loom = loom or Loom(config=loom_config)
        self.hive = hive or LoomHiveConnector(config=hive_config)
        self.cortex = cortex or LoomCortexConnector(config=cortex_config)

        # Create router configuration
        router_config = AttentionRouterConfig(
            auto_switch=self.config.auto_switch,
        )

        # Initialize attention router
        self.router = AttentionRouter(
            loom=self.loom,
            hive=self.hive,
            cortex=self.cortex,
            config=router_config,
        )

    def train(self, text: str) -> None:
        """
        Train the WovenMind on text data.

        This trains the Hive (PRISM-SLM) for pattern matching.
        For deliberate abstraction, use observe_pattern().

        Args:
            text: Text to train on.
        """
        self.hive.train(text)

    def observe_pattern(self, tokens: List[str]) -> Set[str]:
        """
        Observe a pattern for abstraction formation.

        This feeds the Cortex for building abstractions through
        repeated observation.

        Args:
            tokens: List of tokens representing the pattern.

        Returns:
            Set of abstraction IDs that became active.
        """
        return self.cortex.process_slow(tokens)

    def process(
        self,
        context: List[str],
        mode: Optional[ThinkingMode] = None,
    ) -> WovenMindResult:
        """
        Process input through the WovenMind.

        This is the main entry point for processing. It routes the
        input to the appropriate system based on the current mode
        or surprise detection.

        Args:
            context: List of context tokens.
            mode: Explicit mode to use. If None, auto-selects.

        Returns:
            WovenMindResult with activations and metadata.
        """
        # Route through the attention router
        routing_result = self.router.route(context, mode=mode)

        return WovenMindResult(
            mode=routing_result.mode_used,
            activations=routing_result.activations,
            surprise=routing_result.surprise,
            predictions=routing_result.predictions,
            source=routing_result.source,
            metadata=routing_result.metadata,
        )

    def get_current_mode(self) -> ThinkingMode:
        """Get the current thinking mode."""
        return self.loom.get_current_mode()

    def force_mode(self, mode: ThinkingMode, reason: str = "explicit") -> None:
        """
        Force a specific thinking mode.

        Args:
            mode: The mode to force.
            reason: Reason for forcing (for debugging).
        """
        self.loom._mode_controller.force_mode(mode, reason)

    def get_surprise_baseline(self) -> float:
        """Get the current surprise baseline."""
        return self.loom.get_surprise_baseline()

    def get_transition_history(self) -> List[ModeTransition]:
        """Get the mode transition history."""
        return self.loom.get_transition_history()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics.

        Returns:
            Dictionary with stats from all subsystems.
        """
        return {
            "mode": self.get_current_mode().name,
            "loom": {
                "surprise_baseline": self.get_surprise_baseline(),
                "transition_count": len(self.get_transition_history()),
            },
            "hive": self.hive.get_homeostasis_stats(),
            "cortex": self.cortex.get_pattern_stats(),
            "router": self.router.get_routing_stats(),
        }

    def reset(self) -> None:
        """
        Reset the WovenMind to initial state.

        Clears learned patterns and baselines while preserving configuration.
        """
        self.loom.reset()
        self.router._last_predictions.clear()
        self.router._last_context = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize WovenMind state.

        Returns:
            Dictionary representation.
        """
        return {
            "config": {
                "surprise_threshold": self.config.surprise_threshold,
                "k_winners": self.config.k_winners,
                "min_frequency": self.config.min_frequency,
                "auto_switch": self.config.auto_switch,
                "enable_observability": self.config.enable_observability,
            },
            "hive_state": self.hive.to_dict(),
            "cortex_state": self.cortex.to_dict(),
            "router_state": {
                "last_predictions": self.router._last_predictions,
                "last_context": self.router._last_context,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WovenMind":
        """
        Deserialize WovenMind from dictionary.

        Args:
            data: Serialized WovenMind data.

        Returns:
            Reconstructed WovenMind.
        """
        # Reconstruct config
        config_data = data.get("config", {})
        config = WovenMindConfig(
            surprise_threshold=config_data.get("surprise_threshold", 0.3),
            k_winners=config_data.get("k_winners", 5),
            min_frequency=config_data.get("min_frequency", 3),
            auto_switch=config_data.get("auto_switch", True),
            enable_observability=config_data.get("enable_observability", True),
        )

        # Reconstruct components
        hive = LoomHiveConnector.from_dict(data.get("hive_state", {}))
        cortex = LoomCortexConnector.from_dict(data.get("cortex_state", {}))

        # Create mind with restored components
        mind = cls(config=config, hive=hive, cortex=cortex)

        # Restore router state
        router_state = data.get("router_state", {})
        mind.router._last_predictions = router_state.get("last_predictions", {})
        mind.router._last_context = router_state.get("last_context")

        return mind


__all__ = [
    "WovenMindConfig",
    "WovenMindResult",
    "WovenMind",
]
