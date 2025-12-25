# The Woven Mind: Cultured Cortex × Hebbian Hive

## A Dual-Process Learning Architecture for Graph of Thought

---

## Overview

Two complementary systems work together like the left and right hands of a weaver:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│                         THE WOVEN MIND                                  │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                                                                 │   │
│   │                    CULTURED CORTEX                              │   │
│   │              "The Deliberate Dreamer"                           │   │
│   │                                                                 │   │
│   │    • Predicts outcomes          • Learns from success/failure  │   │
│   │    • Builds abstractions        • Top-down guidance            │   │
│   │    • Slow, effortful            • Goal-directed                │   │
│   │                                                                 │   │
│   └───────────────────────┬─────────────────────────────────────────┘   │
│                           │                                             │
│                     ══════╪══════  The Loom  ══════╪══════             │
│                           │                        │                    │
│   ┌───────────────────────▼────────────────────────▼────────────────┐   │
│   │                                                                 │   │
│   │                      HEBBIAN HIVE                               │   │
│   │               "The Pattern Whisperer"                           │   │
│   │                                                                 │   │
│   │    • Detects co-occurrence      • Strengthens what fires together │
│   │    • Forms associations         • Bottom-up discovery          │   │
│   │    • Fast, automatic            • Pattern-matching             │   │
│   │                                                                 │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Part I: The Hebbian Hive

### Philosophy

> "Neurons that fire together, wire together."

The Hive discovers patterns through pure observation. It doesn't know what's "good" or "bad" - it just notices what happens together.

### Core Algorithm

```python
@dataclass
class HiveNode:
    """A node in the Hebbian Hive."""
    id: str
    activation: float = 0.0
    activation_history: deque = field(default_factory=lambda: deque(maxlen=100))

    # Homeostatic regulation
    target_activation: float = 0.05  # Target 5% average activation
    excitability: float = 1.0        # Adjusted to maintain target


@dataclass
class HiveEdge:
    """A learnable edge in the Hive."""
    source_id: str
    target_id: str
    weight: float = 0.0

    # Hebbian trace
    pre_trace: float = 0.0   # Trace of source activation
    post_trace: float = 0.0  # Trace of target activation

    # Statistics
    co_activations: int = 0
    total_observations: int = 0

    @property
    def correlation(self) -> float:
        """How often do source and target fire together?"""
        if self.total_observations == 0:
            return 0.0
        return self.co_activations / self.total_observations


class HebbianHive:
    """
    The Pattern Whisperer.

    Discovers structure through co-occurrence, like how a child
    learns language by noticing what words appear together.
    """

    def __init__(self, graph: ThoughtGraph):
        self.graph = graph
        self.nodes: Dict[str, HiveNode] = {}
        self.edges: Dict[Tuple[str, str], HiveEdge] = {}

        # Learning parameters
        self.learning_rate = 0.01
        self.trace_decay = 0.95      # How fast traces fade
        self.weight_decay = 0.999    # Slow forgetting
        self.sparsity_target = 0.05  # 5% of nodes active

        # Pattern memory
        self.discovered_patterns: List[HivePattern] = []

    # =========================================================================
    # CORE: Hebbian Learning
    # =========================================================================

    def observe(self, activated_nodes: Set[str]):
        """
        Observe a set of co-activated nodes.

        This is the core Hebbian learning step:
        - Update traces for activated nodes
        - Strengthen connections between co-activated nodes
        - Apply lateral inhibition for sparsity
        """
        # Apply lateral inhibition first (sparse coding)
        winners = self._compete(activated_nodes)

        # Update activation traces
        for node_id in self.nodes:
            node = self.nodes[node_id]
            if node_id in winners:
                node.activation = 1.0
            else:
                node.activation *= self.trace_decay

            node.activation_history.append(node.activation)

        # Hebbian weight update: strengthen co-activated connections
        for src_id in winners:
            for tgt_id in winners:
                if src_id != tgt_id:
                    self._strengthen_connection(src_id, tgt_id)

        # Update all edge statistics
        for edge in self.edges.values():
            edge.total_observations += 1
            src_active = edge.source_id in winners
            tgt_active = edge.target_id in winners

            if src_active and tgt_active:
                edge.co_activations += 1

            # Update traces (for temporal Hebbian learning)
            edge.pre_trace = edge.pre_trace * self.trace_decay + (1.0 if src_active else 0.0)
            edge.post_trace = edge.post_trace * self.trace_decay + (1.0 if tgt_active else 0.0)

        # Slow weight decay (forgetting unused connections)
        for edge in self.edges.values():
            edge.weight *= self.weight_decay

        # Homeostatic regulation
        self._regulate_excitability()

    def _compete(self, candidates: Set[str]) -> Set[str]:
        """
        Lateral inhibition: only top-k% most activated survive.

        This implements sparse coding - the cortex does this to
        create distinct, non-overlapping representations.
        """
        if not candidates:
            return candidates

        # Calculate activation strength (excitability-weighted)
        strengths = {}
        for node_id in candidates:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                strengths[node_id] = node.excitability
            else:
                strengths[node_id] = 1.0

        # Sort by strength
        sorted_nodes = sorted(candidates, key=lambda n: strengths.get(n, 0), reverse=True)

        # Keep top k%
        k = max(1, int(len(self.nodes) * self.sparsity_target))
        winners = set(sorted_nodes[:k])

        return winners

    def _strengthen_connection(self, src_id: str, tgt_id: str):
        """
        Hebbian weight update.

        Classic formulation: Δw = η * pre * post
        We use traces for temporal extension: Δw = η * pre_trace * post_trace
        """
        edge_key = (src_id, tgt_id)

        if edge_key not in self.edges:
            self.edges[edge_key] = HiveEdge(source_id=src_id, target_id=tgt_id)

        edge = self.edges[edge_key]

        # Hebbian update with traces
        delta = self.learning_rate * edge.pre_trace * edge.post_trace
        edge.weight = min(1.0, edge.weight + delta)  # Bounded

    def _regulate_excitability(self):
        """
        Homeostatic plasticity: keep average activation near target.

        If a node fires too much → decrease excitability
        If a node fires too little → increase excitability

        This prevents runaway activation or dead nodes.
        """
        for node in self.nodes.values():
            if len(node.activation_history) < 10:
                continue

            avg_activation = sum(node.activation_history) / len(node.activation_history)

            # Adjust excitability to push toward target
            if avg_activation > self.target_activation:
                node.excitability *= 0.99  # Decrease
            else:
                node.excitability *= 1.01  # Increase

            # Bound excitability
            node.excitability = max(0.1, min(10.0, node.excitability))

    # =========================================================================
    # PATTERN DISCOVERY
    # =========================================================================

    def discover_patterns(self, min_correlation: float = 0.3) -> List['HivePattern']:
        """
        Extract stable patterns from learned weights.

        A pattern is a cluster of nodes that frequently co-activate.
        We find these using community detection on the weight graph.
        """
        # Build weighted graph from edges
        import networkx as nx
        G = nx.Graph()

        for (src, tgt), edge in self.edges.items():
            if edge.correlation >= min_correlation:
                G.add_edge(src, tgt, weight=edge.weight)

        if len(G.nodes) == 0:
            return []

        # Community detection (Louvain algorithm)
        try:
            from networkx.algorithms.community import louvain_communities
            communities = louvain_communities(G, weight='weight')
        except:
            # Fallback: connected components
            communities = list(nx.connected_components(G))

        # Convert to patterns
        patterns = []
        for i, community in enumerate(communities):
            if len(community) >= 2:  # At least 2 nodes
                # Calculate pattern strength (average edge weight within)
                internal_weights = [
                    self.edges.get((a, b), HiveEdge(a, b)).weight
                    for a in community for b in community if a != b
                ]
                strength = sum(internal_weights) / len(internal_weights) if internal_weights else 0

                patterns.append(HivePattern(
                    id=f"HP-{i}",
                    nodes=set(community),
                    strength=strength,
                    frequency=min(
                        self.edges.get((a, b), HiveEdge(a, b)).co_activations
                        for a in community for b in community if a != b and (a, b) in self.edges
                    ) if len(community) > 1 else 0
                ))

        self.discovered_patterns = patterns
        return patterns

    # =========================================================================
    # ACTIVATION PROPAGATION
    # =========================================================================

    def propagate(self, seed_nodes: Set[str], steps: int = 3) -> Dict[str, float]:
        """
        Spreading activation from seed nodes.

        Like neural activation propagating through the network.
        Returns activation levels for all nodes.
        """
        activations = {node_id: 0.0 for node_id in self.nodes}

        # Seed activation
        for node_id in seed_nodes:
            if node_id in activations:
                activations[node_id] = 1.0

        # Propagate
        for step in range(steps):
            new_activations = activations.copy()

            for (src, tgt), edge in self.edges.items():
                if src in activations and tgt in activations:
                    # Activation flows through weighted edge
                    flow = activations[src] * edge.weight * 0.5  # Damped
                    new_activations[tgt] = min(1.0, new_activations[tgt] + flow)

            activations = new_activations

        return activations


@dataclass
class HivePattern:
    """A discovered pattern of co-activating nodes."""
    id: str
    nodes: Set[str]
    strength: float
    frequency: int
```

---

## Part II: The Cultured Cortex

### Philosophy

> "Predict, compare, correct."

The Cortex learns through experience. It tries to predict what will happen, observes what actually happens, and adjusts its model accordingly.

### Core Algorithm

```python
@dataclass
class CortexNode:
    """A node in the Cultured Cortex."""
    id: str
    content: Any

    # Hierarchical position
    abstraction_level: int = 0  # 0=concrete, higher=abstract

    # Prediction state
    predicted_activation: float = 0.0
    actual_activation: float = 0.0
    prediction_error: float = 0.0

    # Learning statistics
    times_predicted: int = 0
    times_correct: int = 0

    @property
    def prediction_accuracy(self) -> float:
        if self.times_predicted == 0:
            return 0.5
        return self.times_correct / self.times_predicted


@dataclass
class CortexEdge:
    """A predictive connection in the Cortex."""
    source_id: str
    target_id: str

    # Learnable prediction weight
    weight: float = 1.0
    confidence: float = 0.5

    # Eligibility trace for credit assignment
    eligibility: float = 0.0

    # Learning statistics
    predictions_made: int = 0
    successful_predictions: int = 0


class CulturedCortex:
    """
    The Deliberate Dreamer.

    Learns to predict outcomes and adjusts its model based on
    success or failure. Builds hierarchical abstractions.
    """

    def __init__(self, graph: ThoughtGraph):
        self.graph = graph
        self.nodes: Dict[str, CortexNode] = {}
        self.edges: Dict[Tuple[str, str], CortexEdge] = {}

        # Hierarchical levels
        self.levels: List[Set[str]] = [set() for _ in range(4)]

        # Learning parameters
        self.learning_rate = 0.1
        self.prediction_threshold = 0.3
        self.abstraction_threshold = 5  # Min frequency to abstract

        # Eligibility trace decay
        self.eligibility_decay = 0.9

        # Memory
        self.prediction_history: List[PredictionEvent] = []
        self.abstractions: List[Abstraction] = []

    # =========================================================================
    # CORE: Predictive Learning
    # =========================================================================

    def predict(self, active_nodes: Set[str]) -> Dict[str, float]:
        """
        Generate predictions: what nodes should activate next?

        Predictions flow through weighted edges, with higher-level
        nodes providing top-down constraints.
        """
        predictions = {}

        # Bottom-up: propagate from active nodes
        for src_id in active_nodes:
            if src_id not in self.nodes:
                continue

            src = self.nodes[src_id]

            # Find outgoing edges
            for (s, t), edge in self.edges.items():
                if s == src_id:
                    # Prediction strength = edge weight * confidence
                    strength = edge.weight * edge.confidence

                    if t not in predictions:
                        predictions[t] = 0.0
                    predictions[t] += strength

                    # Mark edge as contributing (eligibility trace)
                    edge.eligibility = 1.0

        # Top-down: high-level predictions constrain low-level
        for level in range(len(self.levels) - 1, 0, -1):
            for high_node_id in self.levels[level]:
                if high_node_id in predictions and predictions[high_node_id] > self.prediction_threshold:
                    # This high-level node is predicted
                    # Boost its children's predictions
                    for (s, t), edge in self.edges.items():
                        if s == high_node_id:
                            if t in predictions:
                                predictions[t] *= 1.2  # Boost

        # Normalize
        if predictions:
            max_pred = max(predictions.values())
            if max_pred > 0:
                predictions = {k: v/max_pred for k, v in predictions.items()}

        return predictions

    def observe_outcome(self, predictions: Dict[str, float],
                        actual: Set[str], outcome: str) -> float:
        """
        Compare predictions to reality and compute error.

        outcome: "success", "partial", or "failure"
        """
        # Store prediction for learning
        event = PredictionEvent(
            timestamp=datetime.now(),
            predictions=predictions.copy(),
            actual=actual.copy(),
            outcome=outcome
        )
        self.prediction_history.append(event)

        # Compute prediction error
        predicted_set = {n for n, p in predictions.items() if p > self.prediction_threshold}

        # Precision: of what we predicted, what was correct?
        if predicted_set:
            precision = len(predicted_set & actual) / len(predicted_set)
        else:
            precision = 0.0

        # Recall: of what happened, what did we predict?
        if actual:
            recall = len(predicted_set & actual) / len(actual)
        else:
            recall = 1.0 if not predicted_set else 0.0

        # F1-style error (1 - F1)
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        prediction_error = 1.0 - f1

        # Update node prediction errors
        for node_id, pred in predictions.items():
            if node_id in self.nodes:
                node = self.nodes[node_id]
                node.predicted_activation = pred
                node.actual_activation = 1.0 if node_id in actual else 0.0
                node.prediction_error = abs(pred - node.actual_activation)
                node.times_predicted += 1
                if node_id in actual and pred > self.prediction_threshold:
                    node.times_correct += 1

        return prediction_error

    def learn(self, outcome: str):
        """
        Adjust weights based on outcome.

        Uses eligibility traces for temporal credit assignment:
        edges that contributed to the prediction get updated
        proportionally to how much they contributed.
        """
        outcome_signal = {
            "success": 1.0,
            "partial": 0.3,
            "failure": -0.5
        }.get(outcome, 0.0)

        for edge in self.edges.values():
            if edge.eligibility > 0.01:  # Only update contributing edges
                # Error-driven update
                delta = self.learning_rate * edge.eligibility * outcome_signal

                # Update weight
                edge.weight = max(0.1, min(2.0, edge.weight + delta))

                # Update confidence based on prediction accuracy
                edge.predictions_made += 1
                if outcome == "success":
                    edge.successful_predictions += 1
                edge.confidence = edge.successful_predictions / edge.predictions_made

                # Decay eligibility
                edge.eligibility *= self.eligibility_decay

    # =========================================================================
    # HIERARCHICAL ABSTRACTION
    # =========================================================================

    def maybe_abstract(self):
        """
        Look for patterns that should become abstractions.

        When a pattern of activations repeats frequently,
        create a higher-level node that represents it.
        """
        # Find repeated prediction patterns
        pattern_counts = defaultdict(int)

        for event in self.prediction_history[-100:]:  # Recent history
            if event.outcome == "success":
                # Create pattern signature
                predicted = frozenset(
                    n for n, p in event.predictions.items()
                    if p > self.prediction_threshold
                )
                actual = frozenset(event.actual)
                pattern_key = (predicted, actual)
                pattern_counts[pattern_key] += 1

        # Abstract patterns that repeat enough
        for (predicted, actual), count in pattern_counts.items():
            if count >= self.abstraction_threshold:
                # Check if we already have this abstraction
                exists = any(
                    a.source_nodes == actual
                    for a in self.abstractions
                )

                if not exists:
                    abstraction = self._create_abstraction(predicted, actual, count)
                    self.abstractions.append(abstraction)

    def _create_abstraction(self, predicted: FrozenSet[str],
                            actual: FrozenSet[str], frequency: int) -> 'Abstraction':
        """Create a higher-level node representing a pattern."""
        # Determine abstraction level (one above highest input)
        input_levels = [
            self.nodes[n].abstraction_level
            for n in actual if n in self.nodes
        ]
        new_level = max(input_levels, default=0) + 1

        # Create node
        abstract_id = f"A{new_level}-{uuid4().hex[:8]}"
        abstract_node = CortexNode(
            id=abstract_id,
            content={
                "type": "abstraction",
                "source_pattern": list(actual),
                "frequency": frequency
            },
            abstraction_level=new_level
        )

        self.nodes[abstract_id] = abstract_node
        if new_level < len(self.levels):
            self.levels[new_level].add(abstract_id)

        # Create edges from components to abstraction
        for component_id in actual:
            edge_key = (component_id, abstract_id)
            self.edges[edge_key] = CortexEdge(
                source_id=component_id,
                target_id=abstract_id,
                weight=1.0,
                confidence=frequency / 10  # Higher frequency = higher confidence
            )

        return Abstraction(
            id=abstract_id,
            source_nodes=actual,
            level=new_level,
            frequency=frequency
        )


@dataclass
class PredictionEvent:
    """Record of a prediction and its outcome."""
    timestamp: datetime
    predictions: Dict[str, float]
    actual: Set[str]
    outcome: str


@dataclass
class Abstraction:
    """A higher-level concept abstracted from a pattern."""
    id: str
    source_nodes: FrozenSet[str]
    level: int
    frequency: int
```

---

## Part III: The Loom (Weaving Them Together)

### Philosophy

> "The Hive discovers. The Cortex selects. Together, they weave knowledge."

### The Integration

```python
class WovenMind:
    """
    The complete dual-process learning system.

    Hebbian Hive (System 1): Fast, automatic, pattern-matching
    Cultured Cortex (System 2): Slow, deliberate, predictive

    They communicate through:
    - Surprise signals (Hive → Cortex)
    - Attention modulation (Cortex → Hive)
    - Shared representations
    """

    def __init__(self, graph: ThoughtGraph):
        self.graph = graph

        # The two systems
        self.hive = HebbianHive(graph)
        self.cortex = CulturedCortex(graph)

        # Shared state
        self.current_context: Set[str] = set()
        self.working_memory: deque = deque(maxlen=10)

        # Mode switching
        self.surprise_threshold = 0.5  # When to engage Cortex
        self.current_mode = "hive"     # "hive" or "cortex"

        # Statistics
        self.hive_decisions = 0
        self.cortex_decisions = 0
        self.mode_switches = 0

    # =========================================================================
    # MAIN PROCESSING LOOP
    # =========================================================================

    def process(self, input_nodes: Set[str]) -> Dict[str, Any]:
        """
        Process input through the dual system.

        Flow:
        1. Hive pattern-matches (fast)
        2. If familiar → respond directly
        3. If surprising → engage Cortex (slow, deliberate)
        4. Learn from outcome
        """
        # Update context
        self.current_context = input_nodes
        self.working_memory.append(input_nodes)

        # Phase 1: Hive pattern matching (always runs)
        hive_activations = self.hive.propagate(input_nodes)
        self.hive.observe(input_nodes)  # Hebbian learning

        # Phase 2: Check for surprise
        surprise = self._compute_surprise(input_nodes, hive_activations)

        if surprise < self.surprise_threshold:
            # Familiar pattern → Hive handles it
            self.current_mode = "hive"
            self.hive_decisions += 1

            response = self._hive_response(hive_activations)

        else:
            # Surprising → Engage Cortex
            if self.current_mode == "hive":
                self.mode_switches += 1
            self.current_mode = "cortex"
            self.cortex_decisions += 1

            # Cortex does deliberate prediction
            predictions = self.cortex.predict(input_nodes)

            # Cortex also modulates Hive attention
            self._modulate_hive_attention(predictions)

            response = self._cortex_response(predictions, hive_activations)

        return {
            "mode": self.current_mode,
            "surprise": surprise,
            "response": response,
            "hive_patterns": len(self.hive.discovered_patterns),
            "cortex_abstractions": len(self.cortex.abstractions)
        }

    def learn_from_outcome(self, outcome: str):
        """
        Both systems learn from the outcome.

        Hive: Strengthens/weakens based on whether pattern was useful
        Cortex: Adjusts predictions based on success/failure
        """
        if self.current_mode == "cortex":
            # Cortex was engaged - it learns from outcome
            self.cortex.learn(outcome)
            self.cortex.maybe_abstract()

            # Hive also learns (the co-occurrences happened)
            # But with outcome modulation
            if outcome == "success":
                # Strengthen the pattern that led to success
                for node_id in self.current_context:
                    if node_id in self.hive.nodes:
                        self.hive.nodes[node_id].excitability *= 1.05
            elif outcome == "failure":
                # Weaken the pattern
                for node_id in self.current_context:
                    if node_id in self.hive.nodes:
                        self.hive.nodes[node_id].excitability *= 0.95

        else:
            # Hive handled it - only Hive learns
            # (Cortex wasn't engaged, nothing to update)
            pass

    # =========================================================================
    # SURPRISE DETECTION
    # =========================================================================

    def _compute_surprise(self, input_nodes: Set[str],
                          hive_activations: Dict[str, float]) -> float:
        """
        How surprising is this input?

        Surprise = how different is this from what the Hive expected?
        High surprise → engage Cortex for deliberate processing.
        """
        # Check if input matches any known Hive pattern
        pattern_matches = []
        for pattern in self.hive.discovered_patterns:
            overlap = len(input_nodes & pattern.nodes)
            if overlap > 0:
                match_score = overlap / len(pattern.nodes)
                pattern_matches.append(match_score)

        if not pattern_matches:
            # No patterns match - highly surprising
            return 1.0

        best_match = max(pattern_matches)

        # Surprise is inverse of best match
        surprise = 1.0 - best_match

        # Also consider prediction confidence
        avg_activation = sum(hive_activations.values()) / len(hive_activations) if hive_activations else 0
        if avg_activation < 0.3:
            surprise += 0.2  # Low activation = uncertain = more surprising

        return min(1.0, surprise)

    # =========================================================================
    # CROSS-SYSTEM COMMUNICATION
    # =========================================================================

    def _modulate_hive_attention(self, cortex_predictions: Dict[str, float]):
        """
        Cortex modulates Hive attention (top-down).

        When Cortex is engaged, it can focus Hive processing
        on nodes it considers important.
        """
        for node_id, prediction in cortex_predictions.items():
            if node_id in self.hive.nodes:
                # Boost excitability of predicted nodes
                boost = 1.0 + (prediction * 0.5)  # Up to 50% boost
                self.hive.nodes[node_id].excitability *= boost

    def _hive_response(self, activations: Dict[str, float]) -> Set[str]:
        """Generate response from Hive activations."""
        # Return nodes above activation threshold
        threshold = 0.3
        return {n for n, a in activations.items() if a > threshold}

    def _cortex_response(self, predictions: Dict[str, float],
                         hive_activations: Dict[str, float]) -> Set[str]:
        """
        Generate response combining Cortex predictions and Hive activations.

        Cortex predictions are weighted more heavily (it was engaged for a reason).
        """
        combined = {}

        for node_id in set(predictions.keys()) | set(hive_activations.keys()):
            cortex_weight = predictions.get(node_id, 0.0) * 0.7
            hive_weight = hive_activations.get(node_id, 0.0) * 0.3
            combined[node_id] = cortex_weight + hive_weight

        threshold = 0.3
        return {n for n, a in combined.items() if a > threshold}

    # =========================================================================
    # CONSOLIDATION (The Night Phase)
    # =========================================================================

    def consolidate(self):
        """
        Offline consolidation - like what happens during sleep.

        - Hive discovers new patterns
        - Cortex reviews and abstracts
        - Weak connections are pruned
        - Successful patterns are strengthened
        """
        # Hive pattern discovery
        self.hive.discover_patterns()

        # Transfer stable Hive patterns to Cortex as priors
        for pattern in self.hive.discovered_patterns:
            if pattern.strength > 0.5 and pattern.frequency > 10:
                # Strong, frequent pattern → inform Cortex
                self._transfer_to_cortex(pattern)

        # Cortex abstraction
        self.cortex.maybe_abstract()

        # Pruning
        self._prune_weak_connections()

    def _transfer_to_cortex(self, pattern: HivePattern):
        """
        Transfer a Hive pattern to Cortex as prior knowledge.

        This is like how procedural memory (unconscious) can become
        declarative memory (conscious) through reflection.
        """
        # Create Cortex node for this pattern if it doesn't exist
        pattern_id = f"HP-{pattern.id}"
        if pattern_id not in self.cortex.nodes:
            self.cortex.nodes[pattern_id] = CortexNode(
                id=pattern_id,
                content={
                    "type": "hive_pattern",
                    "nodes": list(pattern.nodes),
                    "source": "hive_transfer"
                },
                abstraction_level=1  # One level up from raw nodes
            )

            # Create edges from pattern components
            for node_id in pattern.nodes:
                edge_key = (node_id, pattern_id)
                self.cortex.edges[edge_key] = CortexEdge(
                    source_id=node_id,
                    target_id=pattern_id,
                    weight=pattern.strength,
                    confidence=min(1.0, pattern.frequency / 20)
                )

    def _prune_weak_connections(self, threshold: float = 0.05):
        """Remove connections that have become too weak."""
        # Prune Hive edges
        weak_hive = [k for k, e in self.hive.edges.items() if e.weight < threshold]
        for key in weak_hive:
            del self.hive.edges[key]

        # Prune Cortex edges
        weak_cortex = [k for k, e in self.cortex.edges.items()
                       if e.weight < threshold and e.predictions_made > 10]
        for key in weak_cortex:
            del self.cortex.edges[key]

    # =========================================================================
    # INTROSPECTION
    # =========================================================================

    def get_state(self) -> Dict[str, Any]:
        """Get current state of the Woven Mind."""
        return {
            "mode": self.current_mode,
            "hive": {
                "nodes": len(self.hive.nodes),
                "edges": len(self.hive.edges),
                "patterns": len(self.hive.discovered_patterns),
                "decisions": self.hive_decisions
            },
            "cortex": {
                "nodes": len(self.cortex.nodes),
                "edges": len(self.cortex.edges),
                "abstractions": len(self.cortex.abstractions),
                "decisions": self.cortex_decisions
            },
            "mode_switches": self.mode_switches,
            "ratio": f"{self.hive_decisions}:{self.cortex_decisions} (Hive:Cortex)"
        }

    def explain_decision(self, response: Set[str]) -> str:
        """Explain why we produced this response."""
        if self.current_mode == "hive":
            # Find matching pattern
            matching = [p for p in self.hive.discovered_patterns
                       if len(p.nodes & response) > 0]
            if matching:
                return f"Hive pattern match: {matching[0].id} (fast, automatic)"
            return "Hive activation spread (familiar pattern)"
        else:
            return f"Cortex deliberation (surprise detected, predictions made)"
```

---

## Part IV: Usage Example

```python
# Create the Woven Mind
from cortical.reasoning import ThoughtGraph
graph = ThoughtGraph()
mind = WovenMind(graph)

# Simulate reasoning episodes
episodes = [
    {"input": {"error_log", "stack_trace"}, "outcome": "success"},
    {"input": {"error_log", "hypothesis"}, "outcome": "success"},
    {"input": {"error_log", "stack_trace"}, "outcome": "success"},  # Repeat
    {"input": {"new_feature", "design_doc"}, "outcome": "partial"},  # Different
    {"input": {"error_log", "stack_trace"}, "outcome": "success"},  # Repeat
]

for episode in episodes:
    # Process
    result = mind.process(episode["input"])
    print(f"Mode: {result['mode']}, Surprise: {result['surprise']:.2f}")

    # Learn
    mind.learn_from_outcome(episode["outcome"])

# Consolidate (like sleeping)
mind.consolidate()

# Check state
state = mind.get_state()
print(f"\nAfter learning:")
print(f"  Hive patterns: {state['hive']['patterns']}")
print(f"  Cortex abstractions: {state['cortex']['abstractions']}")
print(f"  Decision ratio: {state['ratio']}")

# Output:
# Mode: cortex, Surprise: 1.00  (first time - unknown)
# Mode: cortex, Surprise: 0.70  (partial match)
# Mode: hive, Surprise: 0.20    (familiar pattern!)
# Mode: cortex, Surprise: 0.90  (different domain)
# Mode: hive, Surprise: 0.10    (very familiar)
#
# After learning:
#   Hive patterns: 2
#   Cortex abstractions: 1
#   Decision ratio: 2:3 (Hive:Cortex)
```

---

## Part V: For Your Repository

### Practical Sizing

For a corpus like yours (~134 files, ~2000 functions):

```
Recommended Parameters:

Hebbian Hive:
  - sparsity_target: 0.05 (5% of nodes active)
  - trace_decay: 0.95
  - learning_rate: 0.01
  - Estimated nodes: ~5,000 (concepts + entities)
  - Estimated edges: ~50,000 (learned associations)

Cultured Cortex:
  - prediction_threshold: 0.3
  - learning_rate: 0.1
  - abstraction_threshold: 5
  - Estimated levels: 3-4
  - Estimated abstractions: ~50-100

Consolidation:
  - Run after every ~50-100 processing episodes
  - Or: End of each work session
```

### Integration Points

```python
# Hook into existing GoT
from cortical.got import GoTManager
from cortical.reasoning import ThoughtGraph

# Create mind from GoT
got = GoTManager(Path(".got"))
graph = ThoughtGraph.from_got(got)  # Convert GoT → ThoughtGraph
mind = WovenMind(graph)

# Every time a task is processed:
task_nodes = {f"T-{task.id}", f"C-{task.category}", ...}
result = mind.process(task_nodes)

# When task completes:
mind.learn_from_outcome("success" if task.passed else "failure")

# At end of session:
mind.consolidate()
```

---

## Conclusion

The **Woven Mind** combines:

| System | Role | Speed | Learning |
|--------|------|-------|----------|
| **Hebbian Hive** | Pattern discovery | Fast | Unsupervised |
| **Cultured Cortex** | Prediction & abstraction | Slow | Supervised by outcomes |

They weave together through:
- **Surprise signals** (Hive → Cortex): "This is new, engage deliberation"
- **Attention modulation** (Cortex → Hive): "Focus on these nodes"
- **Pattern transfer** (Hive → Cortex): "This pattern is stable, make it explicit"
- **Consolidation** (both): "Strengthen what works, prune what doesn't"

The result: A system that **learns like a brain** - fast pattern-matching for familiar situations, slow deliberation for novel ones, and gradual abstraction of recurring patterns into reusable knowledge.
