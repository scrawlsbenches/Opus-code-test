"""
Loom-Hive Connector: Integrating the Loom with Enhanced Hebbian Hive.

Provides the connection between the Loom (mode switching) and the
enhanced Hebbian Hive (PRISM-SLM + homeostasis + lateral inhibition).

Part of Sprint 4: The Loom Weaves (T4.1)
Part of the Woven Mind + PRISM Marriage project.

Key functionality:
- process_fast(): Process input through the Hive for FAST mode
- generate_predictions(): Generate predictions for surprise detection
- spread_activation(): Spreading activation through the network
- Homeostatic regulation integration

Example:
    >>> from cortical.reasoning.loom_hive import LoomHiveConnector
    >>> from cortical.reasoning.loom import Loom
    >>>
    >>> connector = LoomHiveConnector(k_winners=5)
    >>> connector.train("neural networks process data")
    >>>
    >>> # Process in FAST mode
    >>> activations = connector.process_fast(["neural"])
    >>>
    >>> # Generate predictions for Loom
    >>> predictions = connector.generate_predictions(["neural"])
    >>> actual = connector.process_fast(["neural"])
    >>>
    >>> loom = Loom()
    >>> surprise = loom.detect_surprise(predictions, actual)
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .prism_slm import PRISMLanguageModel, HiveNode, HiveEdge, SynapticTransition
from .homeostasis import HomeostasisRegulator, HomeostasisConfig


@dataclass
class LoomHiveConfig:
    """Configuration for Loom-Hive connector.

    Attributes:
        k_winners: Number of winners in k-winners-take-all competition.
        lateral_inhibition_strength: How strongly active nodes inhibit others.
        spreading_decay: Decay factor for spreading activation.
        context_size: PRISM-SLM context window size.
    """
    k_winners: int = 5
    lateral_inhibition_strength: float = 0.3
    spreading_decay: float = 0.5
    context_size: int = 3


class LoomHiveConnector:
    """
    Connects the Loom to the enhanced Hebbian Hive.

    Wraps PRISM-SLM and HomeostasisRegulator to provide:
    - FAST mode processing through the Hive
    - Prediction generation for surprise detection
    - Lateral inhibition and k-winners-take-all
    - Spreading activation for associative retrieval

    Attributes:
        model: The PRISM-SLM language model
        regulator: The HomeostasisRegulator for stable activation
        k_winners: Number of winners to keep after competition
    """

    def __init__(
        self,
        model: Optional[PRISMLanguageModel] = None,
        regulator: Optional[HomeostasisRegulator] = None,
        k_winners: int = 5,
        config: Optional[LoomHiveConfig] = None,
    ):
        """
        Initialize the connector.

        Args:
            model: PRISM-SLM instance. Creates default if not provided.
            regulator: HomeostasisRegulator. Creates default if not provided.
            k_winners: Number of winners for lateral inhibition.
            config: Configuration object (overrides k_winners).
        """
        config = config or LoomHiveConfig(k_winners=k_winners)
        self.k_winners = config.k_winners
        self.config = config

        # Initialize model
        self.model = model or PRISMLanguageModel(
            context_size=config.context_size
        )

        # Initialize homeostasis regulator
        self.regulator = regulator or HomeostasisRegulator()

        # HiveNode cache for activation tracking
        self._hive_nodes: Dict[str, HiveNode] = {}

        # Activation step counter
        self._step = 0

    def train(self, text: str) -> None:
        """
        Train the Hive on text.

        Args:
            text: Text to train on.
        """
        self.model.train(text)

    def process_fast(self, context: List[str]) -> Set[str]:
        """
        Process input through the Hive (FAST mode).

        Uses the PRISM-SLM model to generate likely next tokens,
        then applies lateral inhibition (k-winners-take-all).

        Args:
            context: List of context tokens.

        Returns:
            Set of active node IDs after competition.
        """
        self._step += 1

        # Normalize context to lowercase
        normalized_context = [t.lower() for t in context]

        # Create HiveNodes for context tokens (input activation)
        for token in normalized_context:
            if token not in self._hive_nodes:
                self._hive_nodes[token] = HiveNode(id=token)
            # Activate context tokens
            self._hive_nodes[token].activate(1.0, self._step)
            self.regulator.record_activation(token, 1.0)

        # Get predictions from model
        predictions = self._get_raw_predictions(context)

        if not predictions:
            return set(normalized_context)

        # Create HiveNodes for each prediction
        for token, weight in predictions.items():
            if token not in self._hive_nodes:
                self._hive_nodes[token] = HiveNode(id=token)

        # Apply homeostatic modulation
        modulated = {}
        for token, weight in predictions.items():
            node = self._hive_nodes[token]
            excitability = self.regulator.get_excitability(token)
            modulated[token] = weight * excitability

        # Apply lateral inhibition (k-winners-take-all)
        sorted_tokens = sorted(modulated.items(), key=lambda x: -x[1])
        winners = sorted_tokens[:self.k_winners]

        # Activate winning nodes
        active = set(normalized_context)  # Context tokens are always active
        for token, activation in winners:
            if activation > 0:
                node = self._hive_nodes[token]
                actual_activation = node.activate(activation, self._step)

                # Record with homeostasis
                self.regulator.record_activation(token, actual_activation)

                active.add(token)

        # Decay traces for non-winners
        for token, node in self._hive_nodes.items():
            if token not in active:
                node.decay_trace()

        return active

    def _get_raw_predictions(self, context: List[str]) -> Dict[str, float]:
        """Get raw predictions from PRISM-SLM.

        Args:
            context: Context tokens.

        Returns:
            Token -> weight mapping.
        """
        if not context:
            return {}

        # Get transition probabilities from the model
        predictions: Dict[str, float] = {}

        # Normalize context to lowercase like the model does
        normalized_context = [t.lower() for t in context]

        # Try progressively shorter contexts until we find transitions
        for ctx_len in range(min(len(normalized_context), self.model.context_size), 0, -1):
            ctx_tuple = tuple(normalized_context[-ctx_len:])
            transitions = self.model.graph.get_transitions(ctx_tuple)

            if transitions:
                total_weight = sum(t.weight for t in transitions)
                for transition in transitions:
                    prob = transition.probability(total_weight)
                    predictions[transition.to_token] = prob
                break  # Found transitions, stop looking

        return predictions

    def generate_predictions(self, context: List[str]) -> Dict[str, float]:
        """
        Generate predictions for surprise detection.

        Uses the PRISM-SLM model to predict what nodes should activate
        given the current context.

        Args:
            context: List of context tokens.

        Returns:
            Mapping of node_id -> predicted probability.
        """
        predictions = self._get_raw_predictions(context)

        # Normalize to ensure sum <= 1
        total = sum(predictions.values())
        if total > 0:
            predictions = {k: v / total for k, v in predictions.items()}

        return predictions

    def spread_activation(
        self,
        seeds: List[str],
        steps: int = 2,
        decay: float = 0.5,
    ) -> Dict[str, float]:
        """
        Spread activation through the network.

        Starting from seed nodes, activation spreads to connected nodes
        with decay at each step.

        Args:
            seeds: Initial seed nodes to activate.
            steps: Number of spreading steps.
            decay: Decay factor per step.

        Returns:
            Node -> activation level mapping.
        """
        # Normalize seeds to lowercase
        normalized_seeds = [s.lower() for s in seeds]

        # Initialize activations
        activations: Dict[str, float] = {seed: 1.0 for seed in normalized_seeds}

        for step in range(steps):
            new_activations: Dict[str, float] = defaultdict(float)

            for node, activation in activations.items():
                # Add current activation
                new_activations[node] = max(new_activations[node], activation)

                # Spread to connected nodes by checking all contexts containing this node
                # Use the graph's spreading activation if available, or manual lookup
                for context, transitions in self.model.graph._transitions.items():
                    if node in context:
                        total_weight = sum(t.weight for t in transitions)
                        for transition in transitions:
                            prob = transition.probability(total_weight)
                            spread_amount = activation * prob * decay
                            new_activations[transition.to_token] = max(
                                new_activations[transition.to_token],
                                spread_amount
                            )

            activations = dict(new_activations)

        return activations

    def get_hive_node(self, token: str) -> Optional[HiveNode]:
        """
        Get HiveNode for a token.

        Creates a HiveNode lazily if the token exists in the model's vocabulary
        but hasn't been activated yet.

        Args:
            token: The token to look up.

        Returns:
            HiveNode or None if not in vocabulary.
        """
        # Normalize to lowercase
        token_lower = token.lower()

        # Return existing node
        if token_lower in self._hive_nodes:
            return self._hive_nodes[token_lower]

        # Check if token is in vocabulary - if so, create lazily
        if token_lower in self.model.graph._vocab:
            self._hive_nodes[token_lower] = HiveNode(id=token_lower)
            return self._hive_nodes[token_lower]

        return None

    def get_homeostasis_stats(self) -> Dict[str, Any]:
        """
        Get homeostasis statistics.

        Returns:
            Dictionary with homeostasis metrics.
        """
        # Use get_health_metrics() which provides all needed stats
        health = self.regulator.get_health_metrics()
        return {
            "total_nodes_tracked": health.get("node_count", 0),
            "mean_excitability": health.get("avg_excitability", 1.0),
            "pct_overactive": health.get("pct_overactive", 0.0),
            "pct_underactive": health.get("pct_underactive", 0.0),
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize connector state.

        Returns:
            Dictionary representation.
        """
        return {
            "k_winners": self.k_winners,
            "config": {
                "k_winners": self.config.k_winners,
                "lateral_inhibition_strength": self.config.lateral_inhibition_strength,
                "spreading_decay": self.config.spreading_decay,
                "context_size": self.config.context_size,
            },
            "model_graph_state": self.model.graph.to_dict(),
            "regulator_state": self.regulator.to_dict(),
            "hive_nodes": {
                token: node.to_dict()
                for token, node in self._hive_nodes.items()
            },
            "step": self._step,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LoomHiveConnector":
        """
        Deserialize connector from dictionary.

        Args:
            data: Serialized connector data.

        Returns:
            Reconstructed LoomHiveConnector.
        """
        from .prism_slm import TransitionGraph

        # Reconstruct config
        config_data = data.get("config", {})
        config = LoomHiveConfig(
            k_winners=config_data.get("k_winners", 5),
            lateral_inhibition_strength=config_data.get("lateral_inhibition_strength", 0.3),
            spreading_decay=config_data.get("spreading_decay", 0.5),
            context_size=config_data.get("context_size", 3),
        )

        # Reconstruct model - create new one and restore graph
        model = PRISMLanguageModel(context_size=config.context_size)
        graph_data = data.get("model_graph_state", {})
        if graph_data:
            model.graph = TransitionGraph.from_dict(graph_data)

        # Reconstruct regulator
        regulator = HomeostasisRegulator.from_dict(data.get("regulator_state", {}))

        # Create connector
        connector = cls(
            model=model,
            regulator=regulator,
            config=config,
        )

        # Restore hive nodes
        for token, node_data in data.get("hive_nodes", {}).items():
            connector._hive_nodes[token] = HiveNode.from_dict(node_data)

        # Restore step
        connector._step = data.get("step", 0)

        return connector
