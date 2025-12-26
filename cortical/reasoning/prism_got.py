"""
Predictive Reasoning through Incremental Synaptic Memory Graph of Thought (PRISM-GoT).

This module implements a biologically-inspired reasoning framework that combines:
1. Synaptic plasticity - connections that strengthen/weaken based on usage
2. Incremental learning - graph structure built through experience
3. Predictive reasoning - anticipating likely next thoughts based on patterns
4. Graph of Thought - network-based representation of reasoning

Key concepts:
- SynapticEdge: Edge with activation history, temporal decay, and prediction accuracy
- SynapticMemoryGraph: ThoughtGraph extended with plasticity rules
- PlasticityRules: Hebbian, Anti-Hebbian, and Reward-based learning
- IncrementalReasoner: Orchestrates incremental graph building with prediction

The algorithm is inspired by:
- Hebbian learning: "neurons that fire together wire together"
- Synaptic decay: unused connections weaken over time
- Reinforcement learning: successful paths are strengthened

Example:
    >>> graph = SynapticMemoryGraph()
    >>> reasoner = IncrementalReasoner(graph)
    >>>
    >>> # Process thoughts incrementally
    >>> q1 = reasoner.process_thought("What auth method?", NodeType.QUESTION)
    >>> h1 = reasoner.process_thought("Use JWT", NodeType.HYPOTHESIS, EdgeType.EXPLORES)
    >>>
    >>> # Get predictions
    >>> predictions = reasoner.predict_next(q1.id)
    >>> print(predictions[0].node_id)  # Most likely next thought
    >>>
    >>> # Mark successful outcome
    >>> reasoner.mark_outcome_success(path=[q1.id, h1.id])
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
import hashlib
import math

from cortical.utils.id_generation import generate_short_id
from .graph_of_thought import EdgeType, NodeType, ThoughtCluster, ThoughtEdge, ThoughtNode
from .thought_graph import ThoughtGraph


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================


@dataclass(slots=True)
class ActivationRecord:
    """A single activation event for a node or edge."""

    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActivationTrace:
    """
    Tracks the activation history of a node over time.

    Maintains a bounded history of activation events with context,
    enabling temporal pattern analysis and frequency calculations.

    Attributes:
        node_id: The node this trace belongs to
        max_history: Maximum number of events to retain
        history: List of activation records
        total_activations: Cumulative count of all activations
    """

    node_id: str
    max_history: int = 100
    history: List[Dict[str, Any]] = field(default_factory=list)
    total_activations: int = 0

    def record(
        self,
        timestamp: Optional[datetime] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record an activation event.

        Args:
            timestamp: When the activation occurred (defaults to now)
            context: Optional context about the activation
        """
        if timestamp is None:
            timestamp = datetime.now()

        event = {
            "timestamp": timestamp.isoformat(),
            "context": context or {},
        }

        self.history.append(event)
        self.total_activations += 1

        # Maintain bounded history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_frequency(self, window_minutes: int = 60) -> float:
        """
        Calculate activation frequency within a time window.

        Args:
            window_minutes: Time window to consider

        Returns:
            Activations per minute within the window
        """
        if not self.history:
            return 0.0

        now = datetime.now()
        cutoff = now - timedelta(minutes=window_minutes)

        count = 0
        for event in self.history:
            event_time = datetime.fromisoformat(event["timestamp"])
            if event_time >= cutoff:
                count += 1

        return count / window_minutes

    def get_recent(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get the n most recent activation events.

        Args:
            n: Number of events to return

        Returns:
            List of recent activation records
        """
        return list(reversed(self.history[-n:]))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "node_id": self.node_id,
            "max_history": self.max_history,
            "history": self.history,
            "total_activations": self.total_activations,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActivationTrace":
        """Deserialize from dictionary."""
        return cls(
            node_id=data["node_id"],
            max_history=data.get("max_history", 100),
            history=data.get("history", []),
            total_activations=data.get("total_activations", 0),
        )


@dataclass
class SynapticEdge:
    """
    An edge with synaptic plasticity properties.

    Extends the concept of a graph edge with:
    - Activation tracking: when and how often the edge is traversed
    - Weight decay: unused connections weaken over time
    - Prediction accuracy: tracks how often this edge leads to correct predictions
    - Learning: supports Hebbian strengthening and weakening

    Attributes:
        source_id: ID of the source node
        target_id: ID of the target node
        edge_type: Type of relationship
        weight: Current connection strength (0.0 to 1.0+)
        confidence: Confidence in the relationship
        bidirectional: Whether relationship goes both ways
        last_activation_time: When this edge was last activated
        activation_count: Total number of activations
        decay_factor: How quickly the weight decays (0.0 to 1.0)
        prediction_accuracy: Running accuracy of predictions through this edge
        prediction_correct: Count of correct predictions
        prediction_total: Total predictions made
    """

    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    confidence: float = 1.0
    bidirectional: bool = False
    last_activation_time: Optional[datetime] = None
    activation_count: int = 0
    decay_factor: float = 0.99
    prediction_accuracy: float = 0.5  # Prior: uncertain
    prediction_correct: int = 0
    prediction_total: int = 0

    def __post_init__(self) -> None:
        """Initialize derived state."""
        # Weight can exceed 1.0 through strengthening
        if self.weight < 0.0:
            raise ValueError(f"Weight must be >= 0.0, got {self.weight}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0.0, 1.0], got {self.confidence}")
        if not 0.0 <= self.decay_factor <= 1.0:
            raise ValueError(f"Decay factor must be in [0.0, 1.0], got {self.decay_factor}")

    def record_activation(self, timestamp: Optional[datetime] = None) -> None:
        """
        Record that this edge was activated/traversed.

        Args:
            timestamp: When the activation occurred (defaults to now)
        """
        self.last_activation_time = timestamp or datetime.now()
        self.activation_count += 1

    def apply_decay(self) -> None:
        """Apply temporal decay to the weight."""
        self.weight *= self.decay_factor

    def strengthen(self, amount: float = 0.1) -> None:
        """
        Strengthen the connection (Hebbian learning).

        Args:
            amount: How much to increase the weight
        """
        self.weight = min(self.weight + amount, 1.0)

    def weaken(self, amount: float = 0.1) -> None:
        """
        Weaken the connection (Anti-Hebbian).

        Args:
            amount: How much to decrease the weight
        """
        self.weight = max(self.weight - amount, 0.0)

    def record_prediction_outcome(self, correct: bool) -> None:
        """
        Record whether a prediction through this edge was correct.

        Uses Bayesian updating with a Beta prior for smooth accuracy estimation.

        Args:
            correct: Whether the prediction was correct
        """
        if correct:
            self.prediction_correct += 1
        self.prediction_total += 1

        # Beta prior smoothing: add 1 to both correct and total
        # This gives a more stable estimate especially with few samples
        alpha = self.prediction_correct + 1
        beta = (self.prediction_total - self.prediction_correct) + 1
        self.prediction_accuracy = alpha / (alpha + beta)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "weight": self.weight,
            "confidence": self.confidence,
            "bidirectional": self.bidirectional,
            "last_activation_time": (
                self.last_activation_time.isoformat()
                if self.last_activation_time
                else None
            ),
            "activation_count": self.activation_count,
            "decay_factor": self.decay_factor,
            "prediction_accuracy": self.prediction_accuracy,
            "prediction_correct": self.prediction_correct,
            "prediction_total": self.prediction_total,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SynapticEdge":
        """Deserialize from dictionary."""
        last_activation = None
        if data.get("last_activation_time"):
            last_activation = datetime.fromisoformat(data["last_activation_time"])

        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            edge_type=EdgeType(data["edge_type"]),
            weight=data.get("weight", 1.0),
            confidence=data.get("confidence", 1.0),
            bidirectional=data.get("bidirectional", False),
            last_activation_time=last_activation,
            activation_count=data.get("activation_count", 0),
            decay_factor=data.get("decay_factor", 0.99),
            prediction_accuracy=data.get("prediction_accuracy", 0.5),
            prediction_correct=data.get("prediction_correct", 0),
            prediction_total=data.get("prediction_total", 0),
        )

    def to_thought_edge(self) -> ThoughtEdge:
        """Convert to a standard ThoughtEdge."""
        return ThoughtEdge(
            source_id=self.source_id,
            target_id=self.target_id,
            edge_type=self.edge_type,
            weight=min(self.weight, 1.0),  # Clamp for standard edge
            confidence=self.confidence,
            bidirectional=self.bidirectional,
        )


@dataclass
class PredictionResult:
    """
    Result of predicting the next thought from current state.

    Attributes:
        node_id: Predicted next node ID
        node: The predicted ThoughtNode
        probability: Estimated probability of this being correct
        edge: The edge leading to this prediction
        reasoning: Explanation of why this was predicted
    """

    node_id: str
    node: ThoughtNode
    probability: float
    edge: SynapticEdge
    reasoning: str = ""

    def __repr__(self) -> str:
        return f"PredictionResult({self.node_id}, prob={self.probability:.2f})"


# ==============================================================================
# PLASTICITY RULES
# ==============================================================================


class PlasticityRules:
    """
    Implements synaptic plasticity learning rules.

    Three types of learning:
    1. Hebbian: "neurons that fire together wire together"
       - Co-activated nodes strengthen their connection
    2. Anti-Hebbian: "use it or lose it"
       - Connections weaken when not co-activated
    3. Reward-modulated: reinforce successful paths
       - Positive outcomes strengthen, negative weaken

    Attributes:
        hebbian_rate: Learning rate for Hebbian strengthening
        anti_hebbian_rate: Learning rate for Anti-Hebbian weakening
        reward_rate: Learning rate for reward modulation
        min_weight: Minimum allowed weight
        max_weight: Maximum allowed weight
    """

    def __init__(
        self,
        hebbian_rate: float = 0.1,
        anti_hebbian_rate: float = 0.05,
        reward_rate: float = 0.2,
        min_weight: float = 0.0,
        max_weight: float = 2.0,
    ):
        """
        Initialize plasticity rules.

        Args:
            hebbian_rate: Learning rate for co-activation strengthening
            anti_hebbian_rate: Learning rate for unused connection weakening
            reward_rate: Learning rate for reward-based adjustment
            min_weight: Minimum allowed weight
            max_weight: Maximum allowed weight
        """
        self.hebbian_rate = hebbian_rate
        self.anti_hebbian_rate = anti_hebbian_rate
        self.reward_rate = reward_rate
        self.min_weight = min_weight
        self.max_weight = max_weight

    def apply_hebbian(
        self,
        edge: SynapticEdge,
        source_active: bool,
        target_active: bool,
    ) -> float:
        """
        Apply Hebbian learning rule.

        Strengthens connection when both source and target are active.

        Args:
            edge: The edge to potentially strengthen
            source_active: Whether source node is active
            target_active: Whether target node is active

        Returns:
            New weight after applying rule
        """
        if source_active and target_active:
            # Classic Hebbian: Δw = η * x_source * x_target
            # Here we use a simplified additive version
            delta = self.hebbian_rate * edge.confidence
            new_weight = min(edge.weight + delta, self.max_weight)
            return new_weight

        return edge.weight

    def apply_anti_hebbian(
        self,
        edge: SynapticEdge,
        source_active: bool,
        target_active: bool,
    ) -> float:
        """
        Apply Anti-Hebbian learning rule.

        Weakens connection when source fires but target doesn't.

        Args:
            edge: The edge to potentially weaken
            source_active: Whether source node is active
            target_active: Whether target node is active

        Returns:
            New weight after applying rule
        """
        if source_active and not target_active:
            # Weaken when source fires but prediction wasn't followed
            delta = self.anti_hebbian_rate
            new_weight = max(edge.weight - delta, self.min_weight)
            return new_weight

        return edge.weight

    def apply_reward(self, edge: SynapticEdge, reward: float) -> float:
        """
        Apply reward-modulated learning.

        Adjusts weight based on outcome:
        - Positive reward (0 to 1): strengthen
        - Negative reward (-1 to 0): weaken

        Args:
            edge: The edge to adjust
            reward: Reward signal (-1 to 1)

        Returns:
            New weight after applying reward
        """
        delta = self.reward_rate * reward
        new_weight = edge.weight + delta
        new_weight = max(self.min_weight, min(new_weight, self.max_weight))
        return new_weight


# ==============================================================================
# SYNAPTIC MEMORY GRAPH
# ==============================================================================


class SynapticMemoryGraph(ThoughtGraph):
    """
    ThoughtGraph extended with synaptic memory and plasticity.

    Adds to the base ThoughtGraph:
    - Activation traces for all nodes
    - Synaptic edges with plasticity properties
    - Learning rules for connection adjustment
    - Prediction based on connection patterns

    This creates a graph that "learns" from usage patterns,
    strengthening frequently used connections and weakening
    unused ones, similar to how synapses work in the brain.
    """

    def __init__(
        self,
        plasticity_rules: Optional[PlasticityRules] = None,
    ):
        """
        Initialize a synaptic memory graph.

        Args:
            plasticity_rules: Rules for learning (uses defaults if None)
        """
        super().__init__()

        self.activation_traces: Dict[str, ActivationTrace] = {}
        self.synaptic_edges: Dict[Tuple[str, str, EdgeType], SynapticEdge] = {}
        self.plasticity = plasticity_rules or PlasticityRules()

        # Track recent activations for co-activation detection
        self._recent_activations: Dict[str, datetime] = {}

    def add_node(
        self,
        node_id: str,
        node_type: NodeType,
        content: str,
        properties: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ) -> ThoughtNode:
        """
        Add a node and initialize its activation trace.

        Args:
            node_id: Unique identifier for the node
            node_type: Type of thought node
            content: Main content/description
            properties: Optional type-specific properties
            metadata: Optional additional metadata

        Returns:
            The created ThoughtNode
        """
        node = super().add_node(node_id, node_type, content, properties, metadata)

        # Create activation trace for the new node
        self.activation_traces[node_id] = ActivationTrace(node_id=node_id)

        return node

    def remove_node(self, node_id: str) -> ThoughtNode:
        """
        Remove a node and its activation trace.

        Args:
            node_id: ID of node to remove

        Returns:
            The removed ThoughtNode
        """
        node = super().remove_node(node_id)

        # Clean up activation trace
        self.activation_traces.pop(node_id, None)
        self._recent_activations.pop(node_id, None)

        # Clean up synaptic edges
        edges_to_remove = [
            key for key in self.synaptic_edges
            if key[0] == node_id or key[1] == node_id
        ]
        for key in edges_to_remove:
            del self.synaptic_edges[key]

        return node

    def add_synaptic_edge(
        self,
        from_id: str,
        to_id: str,
        edge_type: EdgeType,
        weight: float = 1.0,
        confidence: float = 1.0,
        bidirectional: bool = False,
        decay_factor: float = 0.99,
    ) -> SynapticEdge:
        """
        Add a synaptic edge between two nodes.

        Args:
            from_id: Source node ID
            to_id: Target node ID
            edge_type: Type of relationship
            weight: Initial connection strength
            confidence: Confidence in relationship
            bidirectional: Whether relationship goes both ways
            decay_factor: How quickly weight decays

        Returns:
            The created SynapticEdge
        """
        # Create synaptic edge
        synaptic = SynapticEdge(
            source_id=from_id,
            target_id=to_id,
            edge_type=edge_type,
            weight=weight,
            confidence=confidence,
            bidirectional=bidirectional,
            decay_factor=decay_factor,
        )

        # Store in synaptic registry
        key = (from_id, to_id, edge_type)
        self.synaptic_edges[key] = synaptic

        # Also add as regular edge (uses clamped weight)
        super().add_edge(from_id, to_id, edge_type, min(weight, 1.0), confidence, bidirectional)

        return synaptic

    def get_synaptic_edges_from(self, node_id: str) -> List[SynapticEdge]:
        """
        Get all synaptic edges originating from a node.

        Args:
            node_id: Source node ID

        Returns:
            List of synaptic edges
        """
        return [
            edge for (src, _, _), edge in self.synaptic_edges.items()
            if src == node_id
        ]

    def get_synaptic_edges_to(self, node_id: str) -> List[SynapticEdge]:
        """
        Get all synaptic edges pointing to a node.

        Args:
            node_id: Target node ID

        Returns:
            List of synaptic edges
        """
        return [
            edge for (_, tgt, _), edge in self.synaptic_edges.items()
            if tgt == node_id
        ]

    def activate_node(
        self,
        node_id: str,
        context: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Record that a node was activated.

        Updates the node's activation trace and marks outgoing edges as activated.

        Args:
            node_id: ID of the activated node
            context: Optional context about the activation
            timestamp: When the activation occurred
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")

        timestamp = timestamp or datetime.now()

        # Update activation trace
        if node_id in self.activation_traces:
            self.activation_traces[node_id].record(timestamp, context)

        # Track for co-activation detection
        self._recent_activations[node_id] = timestamp

        # Activate outgoing edges
        for edge in self.get_synaptic_edges_from(node_id):
            edge.record_activation(timestamp)

    def apply_hebbian_learning(self, time_window_seconds: float = 60.0) -> int:
        """
        Apply Hebbian learning based on recent co-activations.

        Nodes activated within the time window are considered co-activated,
        and edges between them are strengthened.

        Args:
            time_window_seconds: Time window for co-activation

        Returns:
            Number of edges strengthened
        """
        now = datetime.now()
        cutoff = now - timedelta(seconds=time_window_seconds)

        # Find recently active nodes
        active_nodes = {
            node_id for node_id, timestamp in self._recent_activations.items()
            if timestamp >= cutoff
        }

        strengthened = 0

        # Strengthen edges between co-activated nodes
        for (src, tgt, _), edge in self.synaptic_edges.items():
            source_active = src in active_nodes
            target_active = tgt in active_nodes

            old_weight = edge.weight
            new_weight = self.plasticity.apply_hebbian(edge, source_active, target_active)

            if new_weight != old_weight:
                edge.weight = new_weight
                strengthened += 1

        return strengthened

    def apply_global_decay(self) -> int:
        """
        Apply decay to all edges.

        Returns:
            Number of edges that decayed
        """
        decayed = 0

        for edge in self.synaptic_edges.values():
            old_weight = edge.weight
            edge.apply_decay()
            if edge.weight != old_weight:
                decayed += 1

        return decayed

    def apply_reward(self, path: List[str], reward: float) -> None:
        """
        Apply reward-based learning to edges along a path.

        Args:
            path: List of node IDs representing the reasoning path
            reward: Reward signal (-1 to 1)
        """
        for i in range(len(path) - 1):
            src, tgt = path[i], path[i + 1]

            # Find edge between consecutive nodes
            for (edge_src, edge_tgt, _), edge in self.synaptic_edges.items():
                if edge_src == src and edge_tgt == tgt:
                    new_weight = self.plasticity.apply_reward(edge, reward)
                    edge.weight = new_weight
                    break

    def predict_next_thoughts(
        self,
        node_id: str,
        top_n: int = 5,
    ) -> List[PredictionResult]:
        """
        Predict likely next thoughts from current node.

        Uses edge weights and prediction accuracy to rank possibilities.

        Args:
            node_id: Current node ID
            top_n: Number of predictions to return

        Returns:
            List of PredictionResults sorted by probability
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")

        predictions = []

        for edge in self.get_synaptic_edges_from(node_id):
            target = self.nodes.get(edge.target_id)
            if not target:
                continue

            # Combine weight and prediction accuracy for probability
            # Higher weight = more likely, higher accuracy = more reliable
            probability = edge.weight * (0.5 + 0.5 * edge.prediction_accuracy)

            predictions.append(PredictionResult(
                node_id=edge.target_id,
                node=target,
                probability=probability,
                edge=edge,
                reasoning=f"weight={edge.weight:.2f}, accuracy={edge.prediction_accuracy:.2f}",
            ))

        # Sort by probability descending
        predictions.sort(key=lambda p: p.probability, reverse=True)

        return predictions[:top_n]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the entire graph to dictionary."""
        return {
            "nodes": {
                node_id: {
                    "id": node.id,
                    "node_type": node.node_type.value,
                    "content": node.content,
                    "properties": node.properties,
                    "metadata": node.metadata,
                }
                for node_id, node in self.nodes.items()
            },
            "synaptic_edges": [
                edge.to_dict() for edge in self.synaptic_edges.values()
            ],
            "activation_traces": {
                node_id: trace.to_dict()
                for node_id, trace in self.activation_traces.items()
            },
            "clusters": {
                cluster_id: {
                    "id": cluster.id,
                    "name": cluster.name,
                    "node_ids": list(cluster.node_ids),
                    "properties": cluster.properties,
                }
                for cluster_id, cluster in self.clusters.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SynapticMemoryGraph":
        """Deserialize a graph from dictionary."""
        graph = cls()

        # Restore nodes
        for node_data in data.get("nodes", {}).values():
            graph.add_node(
                node_id=node_data["id"],
                node_type=NodeType(node_data["node_type"]),
                content=node_data["content"],
                properties=node_data.get("properties"),
                metadata=node_data.get("metadata"),
            )

        # Restore synaptic edges
        for edge_data in data.get("synaptic_edges", []):
            edge = SynapticEdge.from_dict(edge_data)
            key = (edge.source_id, edge.target_id, edge.edge_type)
            graph.synaptic_edges[key] = edge

            # Also add as regular edge (but don't duplicate)
            try:
                graph.edges.append(edge.to_thought_edge())
                graph._edges_from[edge.source_id].append(edge.to_thought_edge())
                graph._edges_to[edge.target_id].append(edge.to_thought_edge())
            except ValueError:
                pass  # Edge exists

        # Restore activation traces
        for node_id, trace_data in data.get("activation_traces", {}).items():
            graph.activation_traces[node_id] = ActivationTrace.from_dict(trace_data)

        # Restore clusters
        for cluster_data in data.get("clusters", {}).values():
            graph.add_cluster(
                cluster_id=cluster_data["id"],
                name=cluster_data["name"],
                node_ids=set(cluster_data.get("node_ids", [])),
            )

        return graph


# ==============================================================================
# INCREMENTAL REASONER
# ==============================================================================


class IncrementalReasoner:
    """
    Orchestrates incremental graph building and predictive reasoning.

    The reasoner processes thoughts one at a time, building the graph
    incrementally while learning from the patterns that emerge.

    Features:
    - Process thoughts and automatically link to current focus
    - Predict next likely thoughts based on learned patterns
    - Verify predictions and update accuracy tracking
    - Mark successful outcomes to reinforce good paths
    - Auto-detect similar content and create links

    Example:
        >>> graph = SynapticMemoryGraph()
        >>> reasoner = IncrementalReasoner(graph)
        >>>
        >>> # Process a chain of reasoning
        >>> q1 = reasoner.process_thought("What auth?", NodeType.QUESTION)
        >>> h1 = reasoner.process_thought("Use JWT", NodeType.HYPOTHESIS,
        ...                                relation_to_focus=EdgeType.EXPLORES)
        >>>
        >>> # Get predictions for what might come next
        >>> predictions = reasoner.predict_next(q1.id)
        >>>
        >>> # Mark a successful outcome
        >>> reasoner.mark_outcome_success(path=[q1.id, h1.id])
    """

    def __init__(
        self,
        graph: SynapticMemoryGraph,
        auto_link_similar: bool = False,
        similarity_threshold: float = 0.7,
    ):
        """
        Initialize the incremental reasoner.

        Args:
            graph: The synaptic memory graph to build
            auto_link_similar: Whether to auto-create SIMILAR edges
            similarity_threshold: Threshold for similarity detection
        """
        self.graph = graph
        self.current_focus: Optional[str] = None
        self.auto_link_similar = auto_link_similar
        self.similarity_threshold = similarity_threshold

        # Track processed content for similarity detection
        self._content_hashes: Dict[str, str] = {}

    def _generate_node_id(self, node_type: NodeType) -> str:
        """Generate a unique node ID."""
        prefix = node_type.value[0].upper()
        return f"{prefix}-{generate_short_id()}"

    def _compute_content_hash(self, content: str) -> str:
        """Compute a hash of content for similarity matching."""
        # Simple word-based fingerprint
        words = set(content.lower().split())
        sorted_words = sorted(words)
        return hashlib.md5(" ".join(sorted_words).encode()).hexdigest()

    def _compute_similarity(self, content1: str, content2: str) -> float:
        """
        Compute Jaccard similarity between two pieces of content.

        Args:
            content1: First text
            content2: Second text

        Returns:
            Similarity score (0.0 to 1.0)
        """
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def process_thought(
        self,
        content: str,
        node_type: NodeType,
        relation_to_focus: Optional[EdgeType] = None,
        properties: Optional[Dict] = None,
        node_id: Optional[str] = None,
    ) -> ThoughtNode:
        """
        Process a single thought and add it to the graph.

        Args:
            content: The thought content
            node_type: Type of thought
            relation_to_focus: How this relates to current focus (creates edge)
            properties: Optional properties for the node
            node_id: Optional custom node ID

        Returns:
            The created ThoughtNode
        """
        # Generate or use provided ID
        if node_id is None:
            node_id = self._generate_node_id(node_type)

        # Add node to graph
        node = self.graph.add_node(
            node_id=node_id,
            node_type=node_type,
            content=content,
            properties=properties,
        )

        # Activate the new node
        self.graph.activate_node(node_id)

        # Link to current focus if relation specified
        if self.current_focus and relation_to_focus:
            self.graph.add_synaptic_edge(
                from_id=self.current_focus,
                to_id=node_id,
                edge_type=relation_to_focus,
            )

        # Auto-link similar content
        if self.auto_link_similar:
            self._link_similar_content(node_id, content)

        # Update focus
        self.current_focus = node_id

        # Store content hash for future similarity checks
        self._content_hashes[node_id] = self._compute_content_hash(content)

        return node

    def _link_similar_content(self, new_node_id: str, content: str) -> None:
        """Create SIMILAR edges to nodes with similar content."""
        for existing_id, existing_hash in self._content_hashes.items():
            if existing_id == new_node_id:
                continue

            existing_node = self.graph.nodes.get(existing_id)
            if not existing_node:
                continue

            similarity = self._compute_similarity(content, existing_node.content)

            if similarity >= self.similarity_threshold:
                # Create bidirectional SIMILAR edge
                try:
                    self.graph.add_synaptic_edge(
                        from_id=new_node_id,
                        to_id=existing_id,
                        edge_type=EdgeType.SIMILAR,
                        weight=similarity,
                        bidirectional=True,
                    )
                except ValueError:
                    pass  # Edge exists

    def predict_next(self, node_id: str, top_n: int = 5) -> List[PredictionResult]:
        """
        Predict the next likely thoughts from a node.

        Args:
            node_id: Current node ID
            top_n: Number of predictions to return

        Returns:
            List of predictions sorted by probability
        """
        return self.graph.predict_next_thoughts(node_id, top_n)

    def verify_prediction(
        self,
        predicted_node_id: str,
        actual_node_id: str,
    ) -> None:
        """
        Verify a prediction and update accuracy tracking.

        Args:
            predicted_node_id: What was predicted
            actual_node_id: What actually happened
        """
        # Find the edge that led to this prediction
        for edge in self.graph.synaptic_edges.values():
            if edge.target_id == predicted_node_id:
                correct = predicted_node_id == actual_node_id
                edge.record_prediction_outcome(correct)

    def mark_outcome_success(
        self,
        path: List[str],
        reward: float = 0.5,
    ) -> None:
        """
        Mark a reasoning path as successful.

        Applies reward learning to strengthen the path.

        Args:
            path: List of node IDs in the successful path
            reward: Reward amount (0 to 1)
        """
        self.graph.apply_reward(path, reward)

    def mark_outcome_failure(
        self,
        path: List[str],
        penalty: float = 0.2,
    ) -> None:
        """
        Mark a reasoning path as unsuccessful.

        Applies negative reward to weaken the path.

        Args:
            path: List of node IDs in the failed path
            penalty: Penalty amount (will be negated)
        """
        self.graph.apply_reward(path, -penalty)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current reasoning graph.

        Returns:
            Dictionary with graph statistics
        """
        nodes_by_type = defaultdict(int)
        for node in self.graph.nodes.values():
            nodes_by_type[node.node_type.value] += 1

        edges_by_type = defaultdict(int)
        for edge in self.graph.synaptic_edges.values():
            edges_by_type[edge.edge_type.value] += 1

        total_activations = sum(
            trace.total_activations
            for trace in self.graph.activation_traces.values()
        )

        avg_weight = 0.0
        if self.graph.synaptic_edges:
            avg_weight = sum(
                e.weight for e in self.graph.synaptic_edges.values()
            ) / len(self.graph.synaptic_edges)

        return {
            "total_nodes": len(self.graph.nodes),
            "total_edges": len(self.graph.synaptic_edges),
            "nodes_by_type": dict(nodes_by_type),
            "edges_by_type": dict(edges_by_type),
            "current_focus": self.current_focus,
            "total_activations": total_activations,
            "average_edge_weight": avg_weight,
        }

    def reset_focus(self) -> None:
        """Reset the current focus to None."""
        self.current_focus = None

    def set_focus(self, node_id: str) -> None:
        """
        Set the current focus to a specific node.

        Args:
            node_id: Node to focus on

        Raises:
            ValueError: If node doesn't exist
        """
        if node_id not in self.graph.nodes:
            raise ValueError(f"Node {node_id} not found")
        self.current_focus = node_id
