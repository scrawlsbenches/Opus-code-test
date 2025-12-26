"""
Attention Router: Mode-Based Routing for Dual-Process Architecture.

Routes processing to the appropriate system based on the current ThinkingMode:
- FAST mode → LoomHiveConnector (Hebbian Hive - pattern matching)
- SLOW mode → LoomCortexConnector (Cultured Cortex - deliberate analysis)

Part of Sprint 4: The Loom Weaves (T4.4)
Part of the Woven Mind + PRISM Marriage project.

Key functionality:
- route(): Process input through the appropriate system based on mode
- route_both(): Process through both systems for comparison
- Auto mode selection based on surprise detection

Example:
    >>> from cortical.reasoning.attention_router import AttentionRouter
    >>> from cortical.reasoning.loom import Loom, ThinkingMode
    >>> from cortical.reasoning.loom_hive import LoomHiveConnector
    >>> from cortical.reasoning.loom_cortex import LoomCortexConnector
    >>>
    >>> loom = Loom()
    >>> hive = LoomHiveConnector()
    >>> cortex = LoomCortexConnector()
    >>>
    >>> router = AttentionRouter(loom=loom, hive=hive, cortex=cortex)
    >>>
    >>> # Route with explicit mode
    >>> result = router.route(["neural", "networks"], mode=ThinkingMode.FAST)
    >>> print(f"Mode: {result.mode_used}, Source: {result.source}")
    >>>
    >>> # Auto mode selection (detects surprise)
    >>> result = router.route(["neural", "networks"], mode=None)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .loom import Loom, SurpriseSignal, ThinkingMode
from .loom_hive import LoomHiveConnector
from .loom_cortex import LoomCortexConnector


@dataclass
class AttentionRouterConfig:
    """Configuration for AttentionRouter.

    Attributes:
        auto_switch: Whether to automatically switch modes based on surprise.
        parallel_probe: Whether to probe both systems in parallel for comparison.
        fast_timeout_ms: Timeout for fast mode processing.
        cache_predictions: Whether to cache predictions for surprise detection.
    """
    auto_switch: bool = True
    parallel_probe: bool = False
    fast_timeout_ms: int = 100
    cache_predictions: bool = True


@dataclass
class RoutingResult:
    """Result of routing through the attention router.

    Attributes:
        mode_used: Which thinking mode was used for processing.
        source: Which system processed the input ("hive" or "cortex").
        activations: Set of activated node IDs.
        surprise: Surprise signal if detected during auto mode selection.
        predictions: Predictions generated (if any).
        metadata: Additional routing metadata.
    """
    mode_used: ThinkingMode
    source: str
    activations: Set[str]
    surprise: Optional[SurpriseSignal] = None
    predictions: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DualRoutingResult:
    """Result of routing through both systems.

    Attributes:
        fast_result: Result from FAST mode (Hive).
        slow_result: Result from SLOW mode (Cortex).
        surprise: Surprise between fast prediction and slow result.
        recommended_mode: Which mode the router recommends.
    """
    fast_result: Set[str]
    slow_result: Set[str]
    surprise: Optional[SurpriseSignal] = None
    recommended_mode: Optional[ThinkingMode] = None


class AttentionRouter:
    """
    Routes attention to the appropriate system based on thinking mode.

    The router integrates Loom (mode selection), LoomHiveConnector (FAST),
    and LoomCortexConnector (SLOW) to provide unified attention routing.

    Attributes:
        loom: The Loom for mode selection and surprise detection.
        hive: LoomHiveConnector for FAST mode processing.
        cortex: LoomCortexConnector for SLOW mode processing.
        config: Router configuration.
    """

    def __init__(
        self,
        loom: Loom,
        hive: LoomHiveConnector,
        cortex: LoomCortexConnector,
        config: Optional[AttentionRouterConfig] = None,
    ) -> None:
        """
        Initialize the attention router.

        Args:
            loom: Loom instance for mode selection.
            hive: LoomHiveConnector for FAST mode.
            cortex: LoomCortexConnector for SLOW mode.
            config: Optional configuration.
        """
        self.loom = loom
        self.hive = hive
        self.cortex = cortex
        self.config = config or AttentionRouterConfig()

        # Cache for predictions (for surprise detection)
        self._last_predictions: Dict[str, float] = {}
        self._last_context: Optional[List[str]] = None

    def route(
        self,
        context: List[str],
        mode: Optional[ThinkingMode] = None,
    ) -> RoutingResult:
        """
        Route input through the appropriate system.

        Args:
            context: List of context tokens.
            mode: Explicit mode to use. If None, auto-selects based on surprise.

        Returns:
            RoutingResult with activations and metadata.
        """
        surprise: Optional[SurpriseSignal] = None
        predictions: Optional[Dict[str, float]] = None

        # Auto mode selection if not specified
        if mode is None:
            mode, surprise, predictions = self._auto_select_mode(context)

        # Route to appropriate system
        if mode == ThinkingMode.FAST:
            activations = self.hive.process_fast(context)
            source = "hive"

            # Generate predictions for next call
            if self.config.cache_predictions:
                self._last_predictions = self.hive.generate_predictions(context)
                self._last_context = context
        else:
            activations = self.cortex.process_slow(context)
            source = "cortex"

        return RoutingResult(
            mode_used=mode,
            source=source,
            activations=activations,
            surprise=surprise,
            predictions=predictions,
            metadata={
                "context_length": len(context),
                "auto_selected": mode is None,
            },
        )

    def _auto_select_mode(
        self,
        context: List[str],
    ) -> tuple[ThinkingMode, Optional[SurpriseSignal], Optional[Dict[str, float]]]:
        """
        Automatically select mode based on surprise detection.

        Args:
            context: List of context tokens.

        Returns:
            Tuple of (selected_mode, surprise_signal, predictions).
        """
        surprise: Optional[SurpriseSignal] = None
        predictions: Optional[Dict[str, float]] = None

        # Get current predictions
        predictions = self.hive.generate_predictions(context)

        # Detect surprise if we have previous predictions
        if self._last_predictions and self._last_context:
            # Get what actually activated with current context
            actual = self.hive.process_fast(context)

            # Compute surprise
            surprise = self.loom.detect_surprise(
                predicted=self._last_predictions,
                actual=actual,
            )

            # Select mode based on surprise
            mode = self.loom.select_mode(signal=surprise)
        else:
            # First call - default to FAST
            mode = ThinkingMode.FAST

        return mode, surprise, predictions

    def route_both(
        self,
        context: List[str],
    ) -> DualRoutingResult:
        """
        Route through both systems for comparison.

        Useful for understanding the difference between fast and slow
        processing, and for training/calibration.

        Args:
            context: List of context tokens.

        Returns:
            DualRoutingResult with both system outputs.
        """
        # Process through FAST (Hive)
        fast_result = self.hive.process_fast(context)

        # Generate predictions before slow processing
        predictions = self.hive.generate_predictions(context)

        # Process through SLOW (Cortex)
        slow_result = self.cortex.process_slow(context)

        # Compute surprise between fast prediction and slow result
        surprise = self.loom.detect_surprise(
            predicted=predictions,
            actual=slow_result,
        )

        # Recommend mode based on surprise
        recommended = self.loom.select_mode(signal=surprise)

        return DualRoutingResult(
            fast_result=fast_result,
            slow_result=slow_result,
            surprise=surprise,
            recommended_mode=recommended,
        )

    def get_current_mode(self) -> ThinkingMode:
        """Get the current thinking mode from the Loom."""
        return self.loom.get_current_mode()

    def get_routing_stats(self) -> Dict[str, Any]:
        """
        Get statistics about routing behavior.

        Returns:
            Dictionary with routing statistics.
        """
        return {
            "current_mode": self.get_current_mode().name,
            "hive_stats": self.hive.get_homeostasis_stats(),
            "cortex_stats": self.cortex.get_pattern_stats(),
            "surprise_baseline": self.loom.get_surprise_baseline(),
            "transition_count": len(self.loom.get_transition_history()),
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize router state.

        Returns:
            Dictionary representation.
        """
        return {
            "config": {
                "auto_switch": self.config.auto_switch,
                "parallel_probe": self.config.parallel_probe,
                "fast_timeout_ms": self.config.fast_timeout_ms,
                "cache_predictions": self.config.cache_predictions,
            },
            "hive_state": self.hive.to_dict(),
            "cortex_state": self.cortex.to_dict(),
            "last_predictions": self._last_predictions,
            "last_context": self._last_context,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        loom: Loom,
    ) -> "AttentionRouter":
        """
        Deserialize router from dictionary.

        Args:
            data: Serialized router data.
            loom: Loom instance to use.

        Returns:
            Reconstructed AttentionRouter.
        """
        # Reconstruct config
        config_data = data.get("config", {})
        config = AttentionRouterConfig(
            auto_switch=config_data.get("auto_switch", True),
            parallel_probe=config_data.get("parallel_probe", False),
            fast_timeout_ms=config_data.get("fast_timeout_ms", 100),
            cache_predictions=config_data.get("cache_predictions", True),
        )

        # Reconstruct Hive and Cortex
        hive = LoomHiveConnector.from_dict(data.get("hive_state", {}))
        cortex = LoomCortexConnector.from_dict(data.get("cortex_state", {}))

        # Create router
        router = cls(
            loom=loom,
            hive=hive,
            cortex=cortex,
            config=config,
        )

        # Restore cached state
        router._last_predictions = data.get("last_predictions", {})
        router._last_context = data.get("last_context")

        return router


__all__ = [
    "AttentionRouterConfig",
    "RoutingResult",
    "DualRoutingResult",
    "AttentionRouter",
]
