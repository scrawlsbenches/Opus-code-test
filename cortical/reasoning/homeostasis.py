"""
Homeostatic Regulation for PRISM-SLM Hebbian Hive.

Implements biological-inspired homeostatic plasticity that maintains
stable activity levels in the synaptic memory network. This prevents:
- Runaway activation (nodes that fire too frequently)
- Dead nodes (nodes that never fire)

Part of Sprint 2: Hebbian Hive Enhancement (Woven Mind + PRISM Marriage)

Key concepts:
- Target activation: Each node aims for ~5% average activation
- Excitability: Multiplicative factor that modulates node responsiveness
- History tracking: Rolling window of activation history for averaging

Example:
    >>> from cortical.reasoning.homeostasis import HomeostasisRegulator
    >>> regulator = HomeostasisRegulator(target_activation=0.05)
    >>> regulator.record_activation("token_a", 0.8)
    >>> regulator.record_activation("token_a", 0.7)
    >>> excitability = regulator.get_excitability("token_a")
    >>> # excitability < 1.0 because token_a fires above target
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, List, Tuple


@dataclass
class HomeostasisConfig:
    """Configuration for homeostatic regulation.

    Attributes:
        target_activation: Target average activation level (0.0-1.0).
            Default 0.05 (5%) produces sparse representations.
        min_excitability: Lower bound for excitability adjustment.
        max_excitability: Upper bound for excitability adjustment.
        adjustment_rate: How quickly excitability adjusts (0.0-1.0).
            Higher = faster adaptation, lower = more stable.
        history_size: Number of activations to track per node.
        min_history_for_adjustment: Minimum history entries needed
            before excitability adjustments begin.
    """
    target_activation: float = 0.05
    min_excitability: float = 0.1
    max_excitability: float = 10.0
    adjustment_rate: float = 0.01
    history_size: int = 100
    min_history_for_adjustment: int = 10

    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0.0 <= self.target_activation <= 1.0:
            raise ValueError(
                f"target_activation must be in [0.0, 1.0], got {self.target_activation}"
            )
        if self.min_excitability <= 0.0:
            raise ValueError(
                f"min_excitability must be > 0, got {self.min_excitability}"
            )
        if self.max_excitability < self.min_excitability:
            raise ValueError(
                f"max_excitability ({self.max_excitability}) must be >= "
                f"min_excitability ({self.min_excitability})"
            )
        if not 0.0 < self.adjustment_rate <= 1.0:
            raise ValueError(
                f"adjustment_rate must be in (0.0, 1.0], got {self.adjustment_rate}"
            )


@dataclass
class NodeState:
    """State tracking for a single node.

    Attributes:
        excitability: Current excitability multiplier.
        activation_history: Recent activation values.
        total_activations: Count of times this node has been activated.
    """
    excitability: float = 1.0
    activation_history: deque = field(default_factory=lambda: deque(maxlen=100))
    total_activations: int = 0

    def record(self, activation: float) -> None:
        """Record an activation event.

        Args:
            activation: Activation value (0.0-1.0).
        """
        self.activation_history.append(activation)
        if activation > 0.0:
            self.total_activations += 1

    def average_activation(self) -> float:
        """Calculate average activation from history.

        Returns:
            Average activation, or 0.0 if no history.
        """
        if not self.activation_history:
            return 0.0
        return sum(self.activation_history) / len(self.activation_history)

    def recent_activation_variance(self) -> float:
        """Calculate variance in recent activations.

        Returns:
            Variance of activation history, or 0.0 if insufficient data.
        """
        if len(self.activation_history) < 2:
            return 0.0
        avg = self.average_activation()
        variance = sum((a - avg) ** 2 for a in self.activation_history)
        return variance / len(self.activation_history)


class HomeostasisRegulator:
    """Regulates neural activity through homeostatic plasticity.

    This class implements biological-inspired homeostatic regulation
    that maintains stable activity levels across the network. Each
    node's excitability is adjusted based on its firing history
    relative to the target activation level.

    Attributes:
        config: Configuration parameters.
        nodes: Dictionary mapping node IDs to their states.

    Example:
        >>> regulator = HomeostasisRegulator()
        >>>
        >>> # Simulate a node that fires frequently
        >>> for _ in range(20):
        ...     regulator.record_activation("busy_node", 0.9)
        >>>
        >>> # Simulate a node that rarely fires
        >>> for _ in range(20):
        ...     regulator.record_activation("quiet_node", 0.01)
        >>>
        >>> # Check excitabilities
        >>> busy_exc = regulator.get_excitability("busy_node")
        >>> quiet_exc = regulator.get_excitability("quiet_node")
        >>> busy_exc < 1.0  # Suppressed
        True
        >>> quiet_exc > 1.0  # Boosted
        True
    """

    def __init__(self, config: Optional[HomeostasisConfig] = None, **kwargs):
        """Initialize the regulator.

        Args:
            config: Configuration object. If None, uses defaults.
            **kwargs: Override config parameters (convenience).
                Supports: target_activation, min_excitability,
                max_excitability, adjustment_rate, history_size,
                min_history_for_adjustment.
        """
        if config is None:
            config = HomeostasisConfig(**kwargs)
        self.config = config
        self.nodes: Dict[str, NodeState] = {}

        # Update history size for new nodes
        self._history_maxlen = config.history_size

    def _ensure_node(self, node_id: str) -> NodeState:
        """Ensure a node exists and return its state.

        Args:
            node_id: The node identifier.

        Returns:
            The NodeState for this node.
        """
        if node_id not in self.nodes:
            self.nodes[node_id] = NodeState(
                activation_history=deque(maxlen=self._history_maxlen)
            )
        return self.nodes[node_id]

    def record_activation(self, node_id: str, activation: float) -> None:
        """Record an activation event for a node.

        Args:
            node_id: The node that was activated.
            activation: The activation value (0.0-1.0).
        """
        state = self._ensure_node(node_id)
        state.record(activation)

    def record_activations(self, activations: Dict[str, float]) -> None:
        """Record multiple activations at once.

        Args:
            activations: Dictionary mapping node IDs to activation values.
        """
        for node_id, activation in activations.items():
            self.record_activation(node_id, activation)

    def get_excitability(self, node_id: str) -> float:
        """Get the current excitability for a node.

        Args:
            node_id: The node identifier.

        Returns:
            Excitability multiplier (1.0 = neutral).
        """
        if node_id not in self.nodes:
            return 1.0  # Default excitability for unknown nodes
        return self.nodes[node_id].excitability

    def get_all_excitabilities(self) -> Dict[str, float]:
        """Get excitabilities for all tracked nodes.

        Returns:
            Dictionary mapping node IDs to their excitability values.
        """
        return {
            node_id: state.excitability
            for node_id, state in self.nodes.items()
        }

    def regulate(self) -> Dict[str, float]:
        """Perform one step of homeostatic regulation.

        Adjusts excitability for all nodes based on their activation
        history relative to the target activation level.

        Returns:
            Dictionary of nodes that were adjusted and their new values.
        """
        adjusted = {}

        for node_id, state in self.nodes.items():
            # Skip if not enough history
            if len(state.activation_history) < self.config.min_history_for_adjustment:
                continue

            avg_activation = state.average_activation()

            # Calculate adjustment direction and magnitude
            error = avg_activation - self.config.target_activation

            if error > 0:
                # Firing too much -> decrease excitability
                adjustment = 1.0 - self.config.adjustment_rate
            else:
                # Firing too little -> increase excitability
                adjustment = 1.0 + self.config.adjustment_rate

            # Apply adjustment
            new_excitability = state.excitability * adjustment

            # Bound excitability
            new_excitability = max(
                self.config.min_excitability,
                min(self.config.max_excitability, new_excitability)
            )

            # Record if changed
            if new_excitability != state.excitability:
                adjusted[node_id] = new_excitability
                state.excitability = new_excitability

        return adjusted

    def apply_excitability(
        self,
        activations: Dict[str, float]
    ) -> Dict[str, float]:
        """Apply excitability modulation to activations.

        This scales each activation by its node's excitability,
        implementing the homeostatic influence on neural responses.

        Args:
            activations: Raw activation values per node.

        Returns:
            Modulated activations (activation * excitability).
        """
        return {
            node_id: activation * self.get_excitability(node_id)
            for node_id, activation in activations.items()
        }

    def get_underactive_nodes(
        self,
        threshold_ratio: float = 0.5
    ) -> List[Tuple[str, float]]:
        """Find nodes that are firing below target.

        Args:
            threshold_ratio: What fraction of target to consider
                "underactive". Default 0.5 means nodes firing at
                less than 50% of target.

        Returns:
            List of (node_id, avg_activation) sorted by activity.
        """
        underactive = []
        threshold = self.config.target_activation * threshold_ratio

        for node_id, state in self.nodes.items():
            if len(state.activation_history) < self.config.min_history_for_adjustment:
                continue

            avg = state.average_activation()
            if avg < threshold:
                underactive.append((node_id, avg))

        return sorted(underactive, key=lambda x: x[1])

    def get_overactive_nodes(
        self,
        threshold_ratio: float = 2.0
    ) -> List[Tuple[str, float]]:
        """Find nodes that are firing above target.

        Args:
            threshold_ratio: What multiple of target to consider
                "overactive". Default 2.0 means nodes firing at
                more than 200% of target.

        Returns:
            List of (node_id, avg_activation) sorted by activity (desc).
        """
        overactive = []
        threshold = self.config.target_activation * threshold_ratio

        for node_id, state in self.nodes.items():
            if len(state.activation_history) < self.config.min_history_for_adjustment:
                continue

            avg = state.average_activation()
            if avg > threshold:
                overactive.append((node_id, avg))

        return sorted(overactive, key=lambda x: -x[1])

    def get_health_metrics(self) -> Dict[str, float]:
        """Get overall health metrics for the network.

        Returns:
            Dictionary with metrics including:
            - avg_excitability: Mean excitability across nodes
            - excitability_std: Standard deviation of excitability
            - pct_underactive: Percentage of underactive nodes
            - pct_overactive: Percentage of overactive nodes
            - avg_activation: Mean activation across network
        """
        if not self.nodes:
            return {
                "avg_excitability": 1.0,
                "excitability_std": 0.0,
                "pct_underactive": 0.0,
                "pct_overactive": 0.0,
                "avg_activation": 0.0,
                "node_count": 0,
            }

        excitabilities = [s.excitability for s in self.nodes.values()]
        avg_exc = sum(excitabilities) / len(excitabilities)
        exc_variance = sum((e - avg_exc) ** 2 for e in excitabilities) / len(excitabilities)
        exc_std = exc_variance ** 0.5

        nodes_with_history = [
            s for s in self.nodes.values()
            if len(s.activation_history) >= self.config.min_history_for_adjustment
        ]

        if nodes_with_history:
            avg_activations = [s.average_activation() for s in nodes_with_history]
            network_avg = sum(avg_activations) / len(avg_activations)

            underactive = sum(
                1 for a in avg_activations
                if a < self.config.target_activation * 0.5
            )
            overactive = sum(
                1 for a in avg_activations
                if a > self.config.target_activation * 2.0
            )

            pct_under = 100.0 * underactive / len(nodes_with_history)
            pct_over = 100.0 * overactive / len(nodes_with_history)
        else:
            network_avg = 0.0
            pct_under = 0.0
            pct_over = 0.0

        return {
            "avg_excitability": avg_exc,
            "excitability_std": exc_std,
            "pct_underactive": pct_under,
            "pct_overactive": pct_over,
            "avg_activation": network_avg,
            "node_count": len(self.nodes),
        }

    def reset_node(self, node_id: str) -> None:
        """Reset a node's state to defaults.

        Args:
            node_id: The node to reset.
        """
        if node_id in self.nodes:
            del self.nodes[node_id]

    def reset_all(self) -> None:
        """Reset all node states."""
        self.nodes.clear()

    def apply_decay(self, decay_factor: float = 0.9) -> int:
        """Apply decay to activation history and excitability.

        Part of the consolidation cycle - decays old activation history
        to allow forgetting of stale patterns.

        Args:
            decay_factor: Factor to multiply excitability by (0.0-1.0).
                Values closer to 1.0 mean slower decay.

        Returns:
            Number of nodes that were decayed.
        """
        decayed_count = 0

        for node_id, state in list(self.nodes.items()):
            # Decay excitability toward neutral (1.0)
            if state.excitability != 1.0:
                if state.excitability > 1.0:
                    # Reduce boosted excitability
                    state.excitability = 1.0 + (state.excitability - 1.0) * decay_factor
                else:
                    # Raise suppressed excitability
                    state.excitability = 1.0 - (1.0 - state.excitability) * decay_factor
                decayed_count += 1

            # Decay activation history by removing oldest entries
            if len(state.activation_history) > 0:
                # Keep only recent 75% of history
                history_keep = int(len(state.activation_history) * 0.75)
                if history_keep > 0:
                    state.activation_history = deque(
                        list(state.activation_history)[-history_keep:],
                        maxlen=self._history_maxlen
                    )

        return decayed_count

    def to_dict(self) -> Dict:
        """Serialize regulator state to dictionary.

        Returns:
            Dictionary representation for persistence.
        """
        return {
            "config": {
                "target_activation": self.config.target_activation,
                "min_excitability": self.config.min_excitability,
                "max_excitability": self.config.max_excitability,
                "adjustment_rate": self.config.adjustment_rate,
                "history_size": self.config.history_size,
                "min_history_for_adjustment": self.config.min_history_for_adjustment,
            },
            "nodes": {
                node_id: {
                    "excitability": state.excitability,
                    "activation_history": list(state.activation_history),
                    "total_activations": state.total_activations,
                }
                for node_id, state in self.nodes.items()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "HomeostasisRegulator":
        """Deserialize regulator from dictionary.

        Args:
            data: Dictionary from to_dict().

        Returns:
            Reconstructed HomeostasisRegulator.
        """
        config = HomeostasisConfig(**data.get("config", {}))
        regulator = cls(config=config)

        for node_id, node_data in data.get("nodes", {}).items():
            state = NodeState(
                excitability=node_data.get("excitability", 1.0),
                activation_history=deque(
                    node_data.get("activation_history", []),
                    maxlen=config.history_size
                ),
                total_activations=node_data.get("total_activations", 0),
            )
            regulator.nodes[node_id] = state

        return regulator


class AdaptiveHomeostasisRegulator(HomeostasisRegulator):
    """Extended regulator with adaptive target activation.

    This variant adjusts the target activation based on network-wide
    statistics, allowing the system to find its own optimal sparsity
    level within configured bounds.

    Attributes:
        min_target: Minimum allowed target activation.
        max_target: Maximum allowed target activation.
        target_adjustment_rate: How fast target adapts.
    """

    def __init__(
        self,
        config: Optional[HomeostasisConfig] = None,
        min_target: float = 0.01,
        max_target: float = 0.20,
        target_adjustment_rate: float = 0.001,
        **kwargs
    ):
        """Initialize adaptive regulator.

        Args:
            config: Base configuration.
            min_target: Minimum target activation (default 1%).
            max_target: Maximum target activation (default 20%).
            target_adjustment_rate: How fast target adapts.
            **kwargs: Passed to parent.
        """
        super().__init__(config, **kwargs)
        self.min_target = min_target
        self.max_target = max_target
        self.target_adjustment_rate = target_adjustment_rate
        self._target_history: deque = deque(maxlen=100)

    def regulate(self) -> Dict[str, float]:
        """Perform regulation with adaptive target.

        First adjusts the target based on network health,
        then performs standard regulation.

        Returns:
            Dictionary of adjusted nodes.
        """
        # Adjust target based on network health
        metrics = self.get_health_metrics()

        if metrics["node_count"] > 0:
            # If too many nodes are underactive, lower target
            # If too many nodes are overactive, raise target
            if metrics["pct_underactive"] > 50:
                new_target = self.config.target_activation * (1 - self.target_adjustment_rate)
            elif metrics["pct_overactive"] > 50:
                new_target = self.config.target_activation * (1 + self.target_adjustment_rate)
            else:
                new_target = self.config.target_activation

            # Bound target
            new_target = max(self.min_target, min(self.max_target, new_target))

            # Only update if changed
            if new_target != self.config.target_activation:
                self._target_history.append(self.config.target_activation)
                self.config.target_activation = new_target

        # Perform standard regulation
        return super().regulate()

    def get_target_history(self) -> List[float]:
        """Get history of target activation adjustments.

        Returns:
            List of historical target values.
        """
        return list(self._target_history)
