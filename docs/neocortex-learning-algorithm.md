# Neocortex-Inspired Learning on the Graph of Thought

## Vision

The biological neocortex learns by **predicting what comes next** and **strengthening connections that lead to accurate predictions**. We can apply these principles to the Graph of Thought, creating a system that learns from experience and improves its own reasoning.

---

## Part I: Neocortical Principles

### How the Neocortex Learns

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    NEOCORTICAL LEARNING PRINCIPLES                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. PREDICTION           "What will happen next?"                       │
│     The cortex constantly predicts future inputs                        │
│                                                                         │
│  2. PREDICTION ERROR     "Was I right?"                                 │
│     Learning happens when predictions don't match reality               │
│                                                                         │
│  3. HEBBIAN LEARNING     "Fire together, wire together"                 │
│     Connections strengthen when neurons activate together               │
│                                                                         │
│  4. SPARSE CODING        "Few neurons active at once"                   │
│     Information encoded across many neurons, few active (2-5%)          │
│                                                                         │
│  5. HIERARCHICAL         "Abstract as you go up"                        │
│     Lower layers: edges, textures | Higher layers: objects, concepts    │
│                                                                         │
│  6. LATERAL INHIBITION   "Winner takes most"                            │
│     Active neurons suppress neighbors, sharpening representations       │
│                                                                         │
│  7. TEMPORAL SEQUENCES   "Patterns unfold in time"                      │
│     The cortex learns sequences, not just static patterns               │
│                                                                         │
│  8. CONSOLIDATION        "Sleep on it"                                  │
│     Replay and strengthen important memories during rest                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Mapping to Graph of Thought

| Neocortex | Graph of Thought Equivalent |
|-----------|----------------------------|
| Neurons | Nodes (Tasks, Decisions, Concepts) |
| Synapses | Edges (DEPENDS_ON, SUPPORTS, etc.) |
| Minicolumns | Clusters of related nodes |
| Layers | Abstraction levels (Facts → Hypotheses → Decisions) |
| Activation | Node salience/importance |
| Prediction | "Given this context, what node should activate next?" |
| Learning | Edge weight adjustment based on outcomes |

---

## Part II: The Cortical GoT Learning Algorithm

### Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CORTICAL GOT LEARNING CYCLE                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│                         ┌─────────────┐                                 │
│                         │   SENSE     │ ← New input (task, question)    │
│                         └──────┬──────┘                                 │
│                                │                                        │
│                         ┌──────▼──────┐                                 │
│                         │   PREDICT   │ ← What nodes should activate?   │
│                         └──────┬──────┘                                 │
│                                │                                        │
│                         ┌──────▼──────┐                                 │
│                         │    ACT      │ ← Execute reasoning path        │
│                         └──────┬──────┘                                 │
│                                │                                        │
│                         ┌──────▼──────┐                                 │
│                         │   COMPARE   │ ← Actual vs predicted outcome   │
│                         └──────┬──────┘                                 │
│                                │                                        │
│                         ┌──────▼──────┐                                 │
│                         │   LEARN     │ ← Adjust edge weights           │
│                         └──────┬──────┘                                 │
│                                │                                        │
│                         ┌──────▼──────┐                                 │
│                         │ CONSOLIDATE │ ← Replay important paths        │
│                         └─────────────┘                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Core Data Structures

```python
@dataclass
class CorticalNode:
    """A node in the Cortical GoT with learning capabilities."""
    id: str
    content: Any
    node_type: NodeType

    # Activation state
    activation: float = 0.0          # Current activation (0-1)
    resting_potential: float = 0.1   # Baseline activation
    decay_rate: float = 0.1          # How fast activation decays

    # Learning state
    prediction_history: List[PredictionRecord] = field(default_factory=list)
    activation_count: int = 0        # How often this node fires
    last_activated: datetime = None

    # Hierarchical position
    abstraction_level: int = 0       # 0=concrete, higher=abstract

    def activate(self, strength: float = 1.0):
        """Activate this node, triggering downstream effects."""
        self.activation = min(1.0, self.activation + strength)
        self.activation_count += 1
        self.last_activated = datetime.now()

    def decay(self, dt: float):
        """Natural decay of activation over time."""
        self.activation = max(
            self.resting_potential,
            self.activation * (1 - self.decay_rate * dt)
        )


@dataclass
class CorticalEdge:
    """An edge with learnable weight and prediction tracking."""
    source_id: str
    target_id: str
    edge_type: EdgeType

    # Learnable parameters
    weight: float = 1.0              # Strength of connection
    confidence: float = 0.5          # How reliable is this edge?

    # Learning statistics
    times_traversed: int = 0         # How often used
    times_successful: int = 0        # How often led to good outcomes
    last_traversed: datetime = None

    # Hebbian trace
    eligibility_trace: float = 0.0   # For temporal credit assignment

    @property
    def success_rate(self) -> float:
        if self.times_traversed == 0:
            return 0.5  # Prior
        return self.times_successful / self.times_traversed


@dataclass
class PredictionRecord:
    """Record of a prediction for learning."""
    timestamp: datetime
    context_nodes: List[str]         # What was active when predicting
    predicted_nodes: List[str]       # What we predicted would activate
    actual_nodes: List[str]          # What actually activated
    outcome: str                     # "success", "partial", "failure"
    prediction_error: float          # Magnitude of error
```

### The Learning Algorithm

```python
class CorticalGoTLearner:
    """
    Neocortex-inspired learning on Graph of Thought.

    Core idea: Learn to predict which nodes should activate next
    given the current context, and adjust edge weights based on
    whether predictions lead to successful outcomes.
    """

    def __init__(self, graph: ThoughtGraph):
        self.graph = graph
        self.learning_rate = 0.1
        self.prediction_threshold = 0.3
        self.sparse_k = 0.05  # Only top 5% of nodes can be active

        # Learning memory
        self.episode_buffer: List[Episode] = []
        self.consolidated_patterns: List[Pattern] = []

    # =========================================================================
    # PHASE 1: SENSE - Encode new input into activation pattern
    # =========================================================================

    def sense(self, input_context: Dict[str, Any]) -> Set[str]:
        """
        Convert input into sparse activation pattern.

        Like the sensory cortex encoding raw input into neural activity.
        """
        activated_nodes = set()

        # Find nodes that match input concepts
        for concept in self._extract_concepts(input_context):
            matching_nodes = self._find_matching_nodes(concept)
            for node_id in matching_nodes:
                self.graph.nodes[node_id].activate(strength=1.0)
                activated_nodes.add(node_id)

        # Apply lateral inhibition (sparse coding)
        activated_nodes = self._apply_lateral_inhibition(activated_nodes)

        return activated_nodes

    def _apply_lateral_inhibition(self, candidates: Set[str]) -> Set[str]:
        """
        Sparse coding: Only keep top-k% most activated nodes.

        This mimics how cortical minicolumns compete - only the
        most strongly activated survive.
        """
        if not candidates:
            return candidates

        # Sort by activation
        sorted_nodes = sorted(
            candidates,
            key=lambda n: self.graph.nodes[n].activation,
            reverse=True
        )

        # Keep only top k%
        k = max(1, int(len(self.graph.nodes) * self.sparse_k))
        winners = set(sorted_nodes[:k])

        # Suppress losers
        for node_id in candidates - winners:
            self.graph.nodes[node_id].activation *= 0.1

        return winners

    # =========================================================================
    # PHASE 2: PREDICT - What should activate next?
    # =========================================================================

    def predict(self, active_nodes: Set[str]) -> Dict[str, float]:
        """
        Predict which nodes will activate next.

        Like the cortex predicting upcoming sensory input.
        Uses edge weights to propagate activation predictions.
        """
        predictions = {}

        for source_id in active_nodes:
            source_activation = self.graph.nodes[source_id].activation

            # Propagate through outgoing edges
            for edge in self.graph.get_outgoing_edges(source_id):
                target_id = edge.target_id

                # Prediction strength = source activation * edge weight * confidence
                pred_strength = (
                    source_activation *
                    edge.weight *
                    edge.confidence
                )

                if target_id not in predictions:
                    predictions[target_id] = 0.0
                predictions[target_id] += pred_strength

                # Mark edge as having contributed to prediction
                edge.eligibility_trace = 1.0

        # Normalize predictions
        if predictions:
            max_pred = max(predictions.values())
            if max_pred > 0:
                predictions = {k: v/max_pred for k, v in predictions.items()}

        return predictions

    # =========================================================================
    # PHASE 3: ACT - Execute reasoning and observe outcomes
    # =========================================================================

    def act(self, predictions: Dict[str, float]) -> Tuple[Set[str], str]:
        """
        Execute reasoning path and observe what actually happens.

        Returns:
            actually_activated: Nodes that actually fired
            outcome: "success", "partial", or "failure"
        """
        # Select predicted nodes above threshold
        selected = {
            node_id for node_id, strength in predictions.items()
            if strength > self.prediction_threshold
        }

        # Execute reasoning (this would integrate with actual GoT execution)
        actually_activated, outcome = self._execute_reasoning(selected)

        return actually_activated, outcome

    # =========================================================================
    # PHASE 4: COMPARE - Calculate prediction error
    # =========================================================================

    def compare(
        self,
        predictions: Dict[str, float],
        actual: Set[str],
        outcome: str
    ) -> PredictionRecord:
        """
        Compare predictions to reality.

        Like computing prediction error in predictive coding theory.
        """
        predicted_set = {
            n for n, s in predictions.items()
            if s > self.prediction_threshold
        }

        # Calculate error components
        false_positives = predicted_set - actual  # Predicted but didn't happen
        false_negatives = actual - predicted_set  # Happened but not predicted
        true_positives = predicted_set & actual   # Correctly predicted

        # Prediction error magnitude
        if len(predicted_set | actual) == 0:
            error = 0.0
        else:
            error = len(false_positives | false_negatives) / len(predicted_set | actual)

        record = PredictionRecord(
            timestamp=datetime.now(),
            context_nodes=list(self._get_active_nodes()),
            predicted_nodes=list(predicted_set),
            actual_nodes=list(actual),
            outcome=outcome,
            prediction_error=error
        )

        return record

    # =========================================================================
    # PHASE 5: LEARN - Adjust edge weights based on outcomes
    # =========================================================================

    def learn(self, record: PredictionRecord):
        """
        Hebbian learning with outcome modulation.

        "Neurons that fire together wire together" - but only
        when the outcome is good.
        """
        outcome_signal = {
            "success": 1.0,
            "partial": 0.5,
            "failure": -0.5
        }.get(record.outcome, 0.0)

        # Adjust edges based on eligibility traces
        for edge in self.graph.edges.values():
            if edge.eligibility_trace > 0:
                # Hebbian update modulated by outcome
                delta = (
                    self.learning_rate *
                    edge.eligibility_trace *
                    outcome_signal
                )

                # Update weight (bounded)
                edge.weight = max(0.01, min(2.0, edge.weight + delta))

                # Update confidence based on prediction accuracy
                if edge.target_id in record.actual_nodes:
                    edge.times_successful += 1
                edge.times_traversed += 1
                edge.confidence = edge.success_rate

                # Decay eligibility trace
                edge.eligibility_trace *= 0.9

        # Store episode for later consolidation
        self.episode_buffer.append(Episode(
            record=record,
            edges_updated=[e.id for e in self.graph.edges.values()
                          if e.eligibility_trace > 0]
        ))

    # =========================================================================
    # PHASE 6: CONSOLIDATE - Replay and strengthen important patterns
    # =========================================================================

    def consolidate(self):
        """
        Memory consolidation - like what happens during sleep.

        Replay successful episodes to strengthen good patterns,
        and let unsuccessful patterns decay.
        """
        if not self.episode_buffer:
            return

        # Sort episodes by outcome quality
        successful_episodes = [
            e for e in self.episode_buffer
            if e.record.outcome == "success"
        ]

        # Replay successful episodes (strengthening)
        for episode in successful_episodes:
            self._replay_episode(episode, strength_multiplier=1.5)

        # Extract recurring patterns
        patterns = self._extract_patterns(successful_episodes)
        for pattern in patterns:
            if pattern.frequency >= 3:  # Seen at least 3 times
                self.consolidated_patterns.append(pattern)

        # Prune weak edges (forgetting)
        self._prune_weak_edges()

        # Clear episode buffer
        self.episode_buffer = []

    def _replay_episode(self, episode: Episode, strength_multiplier: float):
        """Replay an episode to strengthen its edges."""
        for edge_id in episode.edges_updated:
            if edge_id in self.graph.edges:
                edge = self.graph.edges[edge_id]
                edge.weight *= strength_multiplier
                edge.weight = min(2.0, edge.weight)  # Cap

    def _prune_weak_edges(self, threshold: float = 0.1):
        """Remove edges that have become too weak (forgetting)."""
        to_remove = [
            edge_id for edge_id, edge in self.graph.edges.items()
            if edge.weight < threshold and edge.times_traversed > 10
        ]
        for edge_id in to_remove:
            del self.graph.edges[edge_id]

    def _extract_patterns(self, episodes: List[Episode]) -> List[Pattern]:
        """Find recurring activation patterns across episodes."""
        # Group by similar context
        pattern_counts = defaultdict(int)

        for episode in episodes:
            # Create pattern signature
            context_sig = frozenset(episode.record.context_nodes)
            result_sig = frozenset(episode.record.actual_nodes)
            pattern_key = (context_sig, result_sig)
            pattern_counts[pattern_key] += 1

        patterns = []
        for (context, result), count in pattern_counts.items():
            patterns.append(Pattern(
                context_nodes=set(context),
                result_nodes=set(result),
                frequency=count
            ))

        return patterns


@dataclass
class Episode:
    """A single learning episode."""
    record: PredictionRecord
    edges_updated: List[str]


@dataclass
class Pattern:
    """A learned pattern (context → result)."""
    context_nodes: Set[str]
    result_nodes: Set[str]
    frequency: int
```

---

## Part III: Hierarchical Abstraction Learning

### Learning Across Abstraction Levels

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    HIERARCHICAL LEARNING FLOW                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  LEVEL 3: STRATEGIES        "When investigating, use evidence graph"   │
│      ↑ abstracts from            │ predicts                            │
│      │                           ↓                                      │
│  LEVEL 2: PATTERNS          "Error + logs → hypothesis → test"         │
│      ↑ abstracts from            │ predicts                            │
│      │                           ↓                                      │
│  LEVEL 1: SEQUENCES         "First check logs, then form hypothesis"   │
│      ↑ abstracts from            │ predicts                            │
│      │                           ↓                                      │
│  LEVEL 0: ACTIONS           "Read file X, Run command Y"               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

```python
class HierarchicalLearner:
    """
    Learn abstractions at multiple levels.

    Like how the visual cortex learns:
    V1 (edges) → V2 (textures) → V4 (shapes) → IT (objects)

    We learn:
    Actions → Sequences → Patterns → Strategies
    """

    def __init__(self, num_levels: int = 4):
        self.levels = [CorticalGoTLearner() for _ in range(num_levels)]
        self.abstraction_threshold = 5  # Repetitions before abstracting

    def learn_and_abstract(self, episode: Episode):
        """Process episode, potentially creating higher abstractions."""

        # Level 0: Learn raw action sequences
        self.levels[0].learn(episode.record)

        # Check if we should create abstractions
        for level in range(len(self.levels) - 1):
            patterns = self.levels[level].consolidated_patterns

            # Find patterns that repeat enough to abstract
            frequent_patterns = [
                p for p in patterns
                if p.frequency >= self.abstraction_threshold
            ]

            for pattern in frequent_patterns:
                # Create abstracted node at next level
                abstract_node = self._create_abstraction(pattern, level + 1)

                # Add to higher level's graph
                self.levels[level + 1].graph.add_node(abstract_node)

                # Create predictive edge: abstract → concrete pattern
                self._create_predictive_edge(
                    abstract_node.id,
                    pattern,
                    level
                )

    def _create_abstraction(self, pattern: Pattern, level: int) -> CorticalNode:
        """Create abstract node representing a pattern."""
        # Name based on pattern characteristics
        name = self._generate_pattern_name(pattern)

        return CorticalNode(
            id=f"L{level}_{name}_{uuid4().hex[:8]}",
            content={
                "type": "abstraction",
                "source_pattern": pattern,
                "level": level
            },
            node_type=NodeType.CONCEPT,
            abstraction_level=level
        )

    def predict_with_hierarchy(self, context: Set[str]) -> Dict[str, float]:
        """
        Make predictions using all levels of abstraction.

        Higher levels provide "big picture" predictions that
        constrain lower-level predictions (top-down).
        """
        all_predictions = {}

        # Bottom-up: Activate matching patterns at each level
        for level in range(len(self.levels)):
            level_pred = self.levels[level].predict(context)

            # Weight by level (higher = more influence on strategy)
            level_weight = 1.0 + (level * 0.5)

            for node_id, strength in level_pred.items():
                if node_id not in all_predictions:
                    all_predictions[node_id] = 0.0
                all_predictions[node_id] += strength * level_weight

        # Top-down: High-level predictions constrain low-level
        for level in range(len(self.levels) - 1, 0, -1):
            high_level_active = self._get_active_at_level(level)

            for abstract_node in high_level_active:
                # Boost predictions consistent with this abstraction
                predicted_by_abstract = self._get_predicted_by(abstract_node)
                for node_id in predicted_by_abstract:
                    if node_id in all_predictions:
                        all_predictions[node_id] *= 1.5  # Boost

        return all_predictions
```

---

## Part IV: Temporal Sequence Learning

### Learning Sequences Over Time

```python
class TemporalSequenceLearner:
    """
    Learn temporal sequences of node activations.

    Inspired by Hierarchical Temporal Memory (HTM) theory:
    the cortex learns sequences and predicts what comes next.
    """

    def __init__(self, sequence_length: int = 5):
        self.sequence_length = sequence_length
        self.sequence_memory: Dict[Tuple, Counter] = defaultdict(Counter)
        self.current_sequence: List[str] = []

    def observe(self, activated_node: str):
        """Observe a node activation, updating sequence memory."""
        self.current_sequence.append(activated_node)

        # Keep fixed-length window
        if len(self.current_sequence) > self.sequence_length:
            self.current_sequence.pop(0)

        # Learn transition probabilities
        if len(self.current_sequence) >= 2:
            # For each prefix length, record what followed
            for i in range(1, len(self.current_sequence)):
                prefix = tuple(self.current_sequence[:i])
                following = self.current_sequence[i]
                self.sequence_memory[prefix][following] += 1

    def predict_next(self, recent_context: List[str]) -> Dict[str, float]:
        """
        Predict what node will activate next based on recent history.

        Uses variable-order Markov model - tries longest matching
        prefix first, falls back to shorter.
        """
        predictions = {}

        # Try longest prefix first, fall back to shorter
        for length in range(len(recent_context), 0, -1):
            prefix = tuple(recent_context[-length:])

            if prefix in self.sequence_memory:
                counts = self.sequence_memory[prefix]
                total = sum(counts.values())

                for node_id, count in counts.items():
                    prob = count / total
                    # Longer prefix = higher confidence
                    confidence = length / len(recent_context)

                    if node_id not in predictions:
                        predictions[node_id] = 0.0
                    predictions[node_id] = max(
                        predictions[node_id],
                        prob * confidence
                    )

        return predictions

    def detect_anomaly(self, activated_node: str) -> float:
        """
        Detect if activation is anomalous (unexpected given history).

        High anomaly score = surprising activation = learning opportunity.
        """
        if len(self.current_sequence) == 0:
            return 0.0

        predictions = self.predict_next(self.current_sequence)

        if activated_node in predictions:
            # Expected - low anomaly
            return 1.0 - predictions[activated_node]
        else:
            # Unexpected - high anomaly
            return 1.0
```

---

## Part V: Putting It Together

### The Complete Cortical Learning System

```python
class CorticalGoTSystem:
    """
    Complete neocortex-inspired learning system for Graph of Thought.

    Integrates:
    - Sparse distributed representations
    - Prediction and error-driven learning
    - Hierarchical abstraction
    - Temporal sequence learning
    - Memory consolidation
    """

    def __init__(self, graph: ThoughtGraph):
        self.graph = graph

        # Core learners
        self.cortical_learner = CorticalGoTLearner(graph)
        self.hierarchical_learner = HierarchicalLearner()
        self.temporal_learner = TemporalSequenceLearner()

        # State
        self.active_nodes: Set[str] = set()
        self.recent_activations: List[str] = []

        # Learning schedule
        self.consolidation_interval = 100  # Episodes between consolidation
        self.episode_count = 0

    def process_input(self, input_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process new input through the full cortical pipeline.

        Returns reasoning result and learning statistics.
        """
        # 1. SENSE: Encode input
        self.active_nodes = self.cortical_learner.sense(input_context)

        # 2. PREDICT: What should happen next?
        # Combine predictions from all systems
        cortical_pred = self.cortical_learner.predict(self.active_nodes)
        hierarchical_pred = self.hierarchical_learner.predict_with_hierarchy(
            self.active_nodes
        )
        temporal_pred = self.temporal_learner.predict_next(
            self.recent_activations
        )

        # Ensemble predictions
        combined_pred = self._combine_predictions(
            cortical_pred, hierarchical_pred, temporal_pred
        )

        # 3. ACT: Execute reasoning
        actual_nodes, outcome = self.cortical_learner.act(combined_pred)

        # Update temporal tracking
        for node_id in actual_nodes:
            self.temporal_learner.observe(node_id)
            self.recent_activations.append(node_id)

        # 4. COMPARE: Evaluate predictions
        record = self.cortical_learner.compare(
            combined_pred, actual_nodes, outcome
        )

        # Detect anomalies for focused learning
        anomalies = [
            (node_id, self.temporal_learner.detect_anomaly(node_id))
            for node_id in actual_nodes
        ]

        # 5. LEARN: Update weights
        self.cortical_learner.learn(record)
        self.hierarchical_learner.learn_and_abstract(Episode(
            record=record,
            edges_updated=[]
        ))

        # 6. CONSOLIDATE: Periodically strengthen patterns
        self.episode_count += 1
        if self.episode_count % self.consolidation_interval == 0:
            self.cortical_learner.consolidate()

        return {
            "activated_nodes": list(actual_nodes),
            "outcome": outcome,
            "prediction_error": record.prediction_error,
            "anomalies": [(n, a) for n, a in anomalies if a > 0.5],
            "patterns_learned": len(self.cortical_learner.consolidated_patterns)
        }

    def _combine_predictions(
        self,
        cortical: Dict[str, float],
        hierarchical: Dict[str, float],
        temporal: Dict[str, float]
    ) -> Dict[str, float]:
        """Combine predictions from different systems."""
        all_nodes = set(cortical) | set(hierarchical) | set(temporal)

        combined = {}
        for node_id in all_nodes:
            # Weighted combination
            c = cortical.get(node_id, 0.0) * 0.4
            h = hierarchical.get(node_id, 0.0) * 0.3
            t = temporal.get(node_id, 0.0) * 0.3
            combined[node_id] = c + h + t

        return combined

    def get_learned_patterns(self) -> List[Dict]:
        """Return human-readable description of learned patterns."""
        patterns = []

        for pattern in self.cortical_learner.consolidated_patterns:
            patterns.append({
                "context": list(pattern.context_nodes),
                "leads_to": list(pattern.result_nodes),
                "frequency": pattern.frequency,
                "description": self._describe_pattern(pattern)
            })

        return patterns

    def _describe_pattern(self, pattern: Pattern) -> str:
        """Generate human-readable description of a pattern."""
        context_types = [
            self.graph.nodes[n].node_type.value
            for n in pattern.context_nodes
            if n in self.graph.nodes
        ]
        result_types = [
            self.graph.nodes[n].node_type.value
            for n in pattern.result_nodes
            if n in self.graph.nodes
        ]

        return f"When {context_types} → then {result_types}"
```

---

## Part VI: Learning Outcomes

### What the System Learns

After processing many reasoning episodes, the system learns:

```
LEARNED PATTERNS (examples):

1. "Investigation Pattern"
   Context: [OBSERVATION, ERROR_LOG]
   → Activates: [HYPOTHESIS, EVIDENCE_SEARCH]
   Frequency: 47 occurrences

2. "Refactoring Pattern"
   Context: [CODE_SMELL, TEST_SUITE]
   → Activates: [INCREMENTAL_CHANGE, VERIFY_TESTS]
   Frequency: 23 occurrences

3. "Decision Under Uncertainty"
   Context: [MULTIPLE_OPTIONS, INCOMPLETE_INFO]
   → Activates: [GATHER_MORE_INFO, DEFER_DECISION]
   Frequency: 18 occurrences

4. "Crisis Escalation"
   Context: [REPEATED_FAILURE, BLOCKER]
   → Activates: [HUMAN_ESCALATION, PRESERVE_STATE]
   Frequency: 12 occurrences
```

### Edge Weight Evolution

Over time, edges that lead to successful outcomes strengthen:

```
EVOLVED EDGE WEIGHTS:

  HYPOTHESIS → EVIDENCE_SEARCH     weight: 1.87  (was: 1.0)
  EVIDENCE → CONCLUSION            weight: 1.65  (was: 1.0)
  ERROR → CHECK_LOGS               weight: 1.92  (was: 1.0)

  GUESS → IMPLEMENT_DIRECTLY       weight: 0.23  (was: 1.0)  # Weakened
  SKIP_TESTS → DEPLOY              weight: 0.08  (was: 1.0)  # Nearly pruned
```

---

## Conclusion

This neocortex-inspired algorithm brings **true learning** to the Graph of Thought:

1. **Prediction-Driven**: The system predicts what should happen next
2. **Error-Correcting**: Learning from prediction errors, not just successes
3. **Hierarchical**: Abstracts recurring patterns into reusable strategies
4. **Temporal**: Learns sequences, not just static associations
5. **Consolidating**: Strengthens important patterns, forgets unused ones

The result is a system that gets better at reasoning over time by learning which paths through the knowledge graph lead to successful outcomes.

**Key Insight**: The neocortex doesn't store memories - it learns *how to predict*. Similarly, this system doesn't memorize solutions - it learns *how to reason effectively*.
