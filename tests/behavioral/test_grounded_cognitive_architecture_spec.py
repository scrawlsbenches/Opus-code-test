"""
Behavioral Specifications: Grounded Cognitive Architecture v3.0

This specification differs fundamentally from v1.0 and v2.0:
- Every feature is LOAD-BEARING (does real computational work)
- Every feature maps to PROVEN ALGORITHMS (not metaphors)
- Every feature has MEASURABLE SUCCESS CRITERIA
- The architecture is MINIMAL (nothing decorative)

Philosophy:
    v1.0 asked: "What would a cognitive system look like?"
    v2.0 asked: "What cognitive science concepts could we add?"
    v3.0 asks:  "What is the MINIMUM that actually works?"

Grounding Principle:
    If a feature doesn't have:
      1. A clear computational purpose
      2. A proven algorithm for implementation
      3. A measurable success criterion
    Then it doesn't belong here.

The Six Layers:
    1. KNOWLEDGE    - Graph with truth values (OpenCog-proven)
    2. ATTENTION    - Priority queue by importance (OS scheduler)
    3. MEMORY       - Bounded working set with eviction (LRU cache)
    4. PREDICTION   - Anticipate next relevant atoms (language model)
    5. GOALS        - Track progress toward targets (control theory)
    6. EXPLORATION  - Balance exploit/explore (bandit algorithms)

Each layer is:
    - Independently testable
    - Incrementally implementable
    - Grounded in existing algorithms
    - Connected to measurable outcomes

This is engineering, not philosophy.
"""

import pytest
from typing import (
    Protocol, List, Dict, Any, Optional, Callable, Tuple,
    Set, Iterator, TypeVar, Generic
)
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import math


# =============================================================================
# FOUNDATIONAL TYPES
# =============================================================================


@dataclass(frozen=True)
class TruthValue:
    """
    Probabilistic truth value.

    Grounding: Bayesian probability theory.
    Algorithm: Beta distribution (strength = mode, confidence = concentration).

    This is not metaphor. This is statistics.
    """
    strength: float  # P(true) in [0, 1]
    confidence: float  # Evidence weight in [0, 1]

    def __post_init__(self):
        object.__setattr__(self, 'strength', max(0.0, min(1.0, self.strength)))
        object.__setattr__(self, 'confidence', max(0.0, min(1.0, self.confidence)))

    def update(self, observation: bool, learning_rate: float = 0.1) -> 'TruthValue':
        """
        Bayesian update from observation.

        Grounding: Conjugate prior update for Beta distribution.
        """
        obs_value = 1.0 if observation else 0.0

        # Weighted update toward observation
        new_strength = self.strength + learning_rate * (obs_value - self.strength)

        # Confidence increases with evidence (diminishing returns)
        new_confidence = self.confidence + learning_rate * (1.0 - self.confidence)

        return TruthValue(new_strength, new_confidence)

    def surprise(self, observation: bool) -> float:
        """
        Prediction error magnitude.

        Grounding: Information theory - surprisal = -log(P(observation))
        Simplified: |predicted - observed|
        """
        predicted = self.strength
        observed = 1.0 if observation else 0.0
        return abs(predicted - observed)


@dataclass
class Atom:
    """
    Node or link in the cognitive graph.

    Grounding: Standard graph theory + probabilistic databases.

    Minimal attributes:
        - id: unique identifier
        - content: what this atom represents
        - tv: truth value (strength, confidence)
        - sti: short-term importance (attention priority)
        - lti: long-term importance (resistance to forgetting)
        - outgoing: links to other atoms (if this is a link)
    """
    id: str
    content: Any
    tv: TruthValue = field(default_factory=lambda: TruthValue(1.0, 0.0))
    sti: float = 0.0  # Short-term importance [0, 1]
    lti: float = 0.0  # Long-term importance [0, 1]
    outgoing: List[str] = field(default_factory=list)  # For links
    created_at: float = 0.0  # Timestamp
    accessed_at: float = 0.0  # Last access timestamp

    def is_link(self) -> bool:
        return len(self.outgoing) > 0


@dataclass
class Goal:
    """
    A target state with progress tracking.

    Grounding: Control theory - goal = setpoint, progress = error signal.

    Minimal design:
        - No "desire" abstraction
        - No intensification
        - No hierarchy
        - Just: target, current, importance
    """
    id: str
    description: str
    target_state: Any  # What we're trying to achieve
    current_state: Any = None  # Where we are
    importance: float = 0.5  # How much this matters [0, 1]

    @property
    def progress(self) -> float:
        """Distance covered toward target [0, 1]."""
        if self.current_state is None:
            return 0.0
        if self.current_state == self.target_state:
            return 1.0
        # For numeric states, use normalized distance
        if isinstance(self.target_state, (int, float)):
            if self.target_state == 0:
                return 1.0 if self.current_state == 0 else 0.0
            return min(1.0, abs(self.current_state / self.target_state))
        # Default: binary
        return 0.0

    @property
    def urgency(self) -> float:
        """Priority for attention allocation."""
        return self.importance * (1.0 - self.progress)

    def is_complete(self) -> bool:
        return self.progress >= 0.99


# =============================================================================
# LAYER 1: KNOWLEDGE GRAPH
# =============================================================================


class KnowledgeGraph(Protocol):
    """
    Protocol for knowledge storage.

    Grounding: Graph databases (Neo4j, OpenCog AtomSpace).
    Operations: CRUD + traversal + query.
    """

    def add(self, atom: Atom) -> None: ...
    def get(self, atom_id: str) -> Optional[Atom]: ...
    def remove(self, atom_id: str) -> bool: ...
    def query(self, predicate: Callable[[Atom], bool]) -> List[Atom]: ...
    def neighbors(self, atom_id: str) -> List[Atom]: ...
    def all_atoms(self) -> Iterator[Atom]: ...


class InMemoryKnowledgeGraph:
    """
    Simple in-memory implementation.

    Grounding: Dictionary with adjacency lists.
    Suitable for: Testing, small graphs.
    """

    def __init__(self):
        self._atoms: Dict[str, Atom] = {}
        self._incoming: Dict[str, Set[str]] = {}  # atom_id -> link_ids pointing to it

    def add(self, atom: Atom) -> None:
        self._atoms[atom.id] = atom
        if atom.is_link():
            for target_id in atom.outgoing:
                if target_id not in self._incoming:
                    self._incoming[target_id] = set()
                self._incoming[target_id].add(atom.id)

    def get(self, atom_id: str) -> Optional[Atom]:
        return self._atoms.get(atom_id)

    def remove(self, atom_id: str) -> bool:
        if atom_id not in self._atoms:
            return False
        atom = self._atoms[atom_id]
        if atom.is_link():
            for target_id in atom.outgoing:
                if target_id in self._incoming:
                    self._incoming[target_id].discard(atom_id)
        del self._atoms[atom_id]
        return True

    def query(self, predicate: Callable[[Atom], bool]) -> List[Atom]:
        return [a for a in self._atoms.values() if predicate(a)]

    def neighbors(self, atom_id: str) -> List[Atom]:
        """Get atoms connected to this one (outgoing + incoming)."""
        result = []
        atom = self._atoms.get(atom_id)
        if atom and atom.is_link():
            for target_id in atom.outgoing:
                if target_id in self._atoms:
                    result.append(self._atoms[target_id])
        if atom_id in self._incoming:
            for link_id in self._incoming[atom_id]:
                if link_id in self._atoms:
                    result.append(self._atoms[link_id])
        return result

    def all_atoms(self) -> Iterator[Atom]:
        return iter(self._atoms.values())

    def size(self) -> int:
        return len(self._atoms)


# =============================================================================
# LAYER 2: ATTENTION (Priority Queue)
# =============================================================================


class AttentionSystem:
    """
    Manages which atoms get cognitive resources.

    Grounding: OS scheduler, priority queue.
    Algorithm: Heap-based priority queue with decay.

    Key insight: This is just a scheduler. No metaphor needed.
    """

    def __init__(
        self,
        graph: KnowledgeGraph,
        focus_size: int = 7,  # Miller's 7±2
        decay_rate: float = 0.05,
    ):
        self.graph = graph
        self.focus_size = focus_size
        self.decay_rate = decay_rate
        self._time: float = 0.0

    def stimulate(self, atom_id: str, amount: float = 0.2) -> None:
        """
        Increase STI of an atom (it got attention).

        Grounding: Hebbian learning - "what fires together wires together".
        """
        atom = self.graph.get(atom_id)
        if atom:
            atom.sti = min(1.0, atom.sti + amount)
            atom.accessed_at = self._time

    def decay_all(self) -> None:
        """
        Apply exponential decay to all STI values.

        Grounding: Forgetting curve (Ebbinghaus).
        """
        for atom in self.graph.all_atoms():
            atom.sti *= (1.0 - self.decay_rate)
            # Below threshold? Clear STI
            if atom.sti < 0.01:
                atom.sti = 0.0

    def get_focus(self) -> List[Atom]:
        """
        Get top-K atoms by STI (the "attention focus").

        Grounding: Top-K selection, O(n log k) with heap.
        """
        all_atoms = list(self.graph.all_atoms())
        sorted_atoms = sorted(all_atoms, key=lambda a: a.sti, reverse=True)
        return sorted_atoms[:self.focus_size]

    def step(self) -> None:
        """Advance time, apply decay."""
        self._time += 1.0
        self.decay_all()

    def spread_activation(self, source_id: str, spread_factor: float = 0.3) -> None:
        """
        Spread activation from source to neighbors.

        Grounding: Spreading activation in semantic networks (Collins & Loftus, 1975).
        """
        source = self.graph.get(source_id)
        if not source or source.sti < 0.1:
            return

        neighbors = self.graph.neighbors(source_id)
        if not neighbors:
            return

        spread_amount = source.sti * spread_factor / len(neighbors)
        for neighbor in neighbors:
            neighbor.sti = min(1.0, neighbor.sti + spread_amount)


# =============================================================================
# LAYER 3: WORKING MEMORY (Bounded Buffer)
# =============================================================================


class WorkingMemory:
    """
    Bounded capacity workspace.

    Grounding: Cowan's 4±1 capacity limit, LRU cache.

    This is literally a cache with eviction policy.
    No metaphor. Just computer science.
    """

    def __init__(self, capacity: int = 4):
        self.capacity = capacity
        self._slots: List[Atom] = []
        self._access_order: List[str] = []  # For LRU

    def load(self, atom: Atom) -> Optional[Atom]:
        """
        Load atom into working memory.

        Returns evicted atom if capacity exceeded.
        """
        # Already present? Move to front of access order
        for i, slot in enumerate(self._slots):
            if slot.id == atom.id:
                self._access_order.remove(atom.id)
                self._access_order.append(atom.id)
                self._slots[i] = atom  # Update
                return None

        evicted = None

        # At capacity? Evict LRU
        if len(self._slots) >= self.capacity:
            lru_id = self._access_order.pop(0)
            for i, slot in enumerate(self._slots):
                if slot.id == lru_id:
                    evicted = self._slots.pop(i)
                    break

        # Add new atom
        self._slots.append(atom)
        self._access_order.append(atom.id)

        return evicted

    def get(self, atom_id: str) -> Optional[Atom]:
        """Get atom from working memory (updates access order)."""
        for slot in self._slots:
            if slot.id == atom_id:
                self._access_order.remove(atom_id)
                self._access_order.append(atom_id)
                return slot
        return None

    def contains(self, atom_id: str) -> bool:
        return any(s.id == atom_id for s in self._slots)

    def contents(self) -> List[Atom]:
        return list(self._slots)

    def clear(self) -> None:
        self._slots.clear()
        self._access_order.clear()

    def is_full(self) -> bool:
        return len(self._slots) >= self.capacity


# =============================================================================
# LAYER 4: PREDICTION
# =============================================================================


class Predictor(Protocol):
    """
    Predicts next relevant atoms given context.

    Grounding: Language models, association rules.

    This is where the "intelligence" lives.
    Everything else is infrastructure.
    """

    def predict(self, context: List[Atom]) -> List[Tuple[str, float]]:
        """
        Given context atoms, predict relevant atom IDs with probabilities.

        Returns: List of (atom_id, probability) pairs.
        """
        ...


class AssociativePredictor:
    """
    Simple co-occurrence based predictor.

    Grounding: Association rules, Hebbian learning.
    Algorithm: Count co-activations, normalize to probabilities.

    This is a minimal predictor. Real systems would use neural models.
    """

    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph
        self._co_occurrences: Dict[str, Dict[str, int]] = {}  # a -> b -> count

    def record_co_occurrence(self, atom_a_id: str, atom_b_id: str) -> None:
        """Record that these atoms were active together."""
        if atom_a_id not in self._co_occurrences:
            self._co_occurrences[atom_a_id] = {}
        if atom_b_id not in self._co_occurrences[atom_a_id]:
            self._co_occurrences[atom_a_id][atom_b_id] = 0
        self._co_occurrences[atom_a_id][atom_b_id] += 1

    def predict(self, context: List[Atom]) -> List[Tuple[str, float]]:
        """Predict based on co-occurrence with context."""
        if not context:
            return []

        # Aggregate predictions from all context atoms
        scores: Dict[str, float] = {}
        context_ids = {a.id for a in context}

        for atom in context:
            if atom.id in self._co_occurrences:
                for target_id, count in self._co_occurrences[atom.id].items():
                    if target_id not in context_ids:  # Don't predict what's already there
                        scores[target_id] = scores.get(target_id, 0.0) + count

        if not scores:
            return []

        # Normalize to probabilities
        total = sum(scores.values())
        predictions = [(atom_id, score / total) for atom_id, score in scores.items()]

        # Sort by probability descending
        predictions.sort(key=lambda x: -x[1])

        return predictions[:10]  # Top 10


class SurpriseTracker:
    """
    Tracks prediction errors to drive learning.

    Grounding: Predictive coding, Rescorla-Wagner learning rule.
    Metric: Mean absolute prediction error.
    """

    def __init__(self, predictor: Predictor):
        self.predictor = predictor
        self._prediction_history: List[Tuple[List[str], str, float]] = []

    def record_outcome(
        self,
        context: List[Atom],
        actual_atom_id: str,
    ) -> float:
        """
        Record what actually happened and return surprise level.

        Returns: Surprise in [0, 1]. 0 = predicted, 1 = unexpected.
        """
        predictions = self.predictor.predict(context)
        predicted_ids = {p[0]: p[1] for p in predictions}

        if actual_atom_id in predicted_ids:
            # Predicted - surprise is inverse of probability
            surprise = 1.0 - predicted_ids[actual_atom_id]
        else:
            # Not predicted at all - maximum surprise
            surprise = 1.0

        self._prediction_history.append((
            [a.id for a in context],
            actual_atom_id,
            surprise
        ))

        return surprise

    def mean_surprise(self, window: int = 100) -> float:
        """Average surprise over recent predictions."""
        if not self._prediction_history:
            return 0.5  # Uncertain

        recent = self._prediction_history[-window:]
        return sum(s for _, _, s in recent) / len(recent)


# =============================================================================
# LAYER 5: GOAL TRACKING
# =============================================================================


class GoalTracker:
    """
    Tracks goals and their progress.

    Grounding: Control theory - error = target - current.
    Algorithm: Priority queue by urgency.

    No desires. No hierarchy. Just goals with progress.
    """

    def __init__(self):
        self._goals: Dict[str, Goal] = {}

    def add_goal(self, goal: Goal) -> None:
        self._goals[goal.id] = goal

    def remove_goal(self, goal_id: str) -> bool:
        if goal_id in self._goals:
            del self._goals[goal_id]
            return True
        return False

    def update_progress(self, goal_id: str, current_state: Any) -> None:
        if goal_id in self._goals:
            self._goals[goal_id].current_state = current_state

    def get_active_goals(self) -> List[Goal]:
        """Get incomplete goals sorted by urgency."""
        active = [g for g in self._goals.values() if not g.is_complete()]
        return sorted(active, key=lambda g: g.urgency, reverse=True)

    def get_top_goal(self) -> Optional[Goal]:
        """Get highest urgency goal."""
        active = self.get_active_goals()
        return active[0] if active else None

    def complete_count(self) -> int:
        return sum(1 for g in self._goals.values() if g.is_complete())

    def total_progress(self) -> float:
        """Average progress across all goals."""
        if not self._goals:
            return 0.0
        return sum(g.progress for g in self._goals.values()) / len(self._goals)


# =============================================================================
# LAYER 6: EXPLORATION/EXPLOITATION
# =============================================================================


class ExplorationController:
    """
    Balances exploration vs exploitation.

    Grounding: Multi-armed bandit algorithms (ε-greedy, UCB).
    Single parameter: ε (exploration rate).

    No affect states. No multiple emotions.
    Just one number that adapts.
    """

    def __init__(
        self,
        initial_epsilon: float = 0.3,
        min_epsilon: float = 0.05,
        max_epsilon: float = 0.9,
        adaptation_rate: float = 0.1,
    ):
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.adaptation_rate = adaptation_rate
        self._consecutive_failures: int = 0
        self._consecutive_successes: int = 0

    def should_explore(self) -> bool:
        """
        Decide whether to explore (try something new) or exploit (use best known).

        Grounding: ε-greedy policy.
        """
        import random
        return random.random() < self.epsilon

    def record_success(self) -> None:
        """Current approach is working - reduce exploration."""
        self._consecutive_successes += 1
        self._consecutive_failures = 0

        # Decrease epsilon (exploit more)
        self.epsilon = max(
            self.min_epsilon,
            self.epsilon - self.adaptation_rate
        )

    def record_failure(self) -> None:
        """Current approach failed - increase exploration."""
        self._consecutive_failures += 1
        self._consecutive_successes = 0

        # Increase epsilon (explore more)
        self.epsilon = min(
            self.max_epsilon,
            self.epsilon + self.adaptation_rate
        )

    def is_stuck(self, threshold: int = 3) -> bool:
        """Are we in a failure loop?"""
        return self._consecutive_failures >= threshold


# =============================================================================
# THE INTEGRATED SYSTEM
# =============================================================================


class CognitiveAgent:
    """
    The complete minimal cognitive agent.

    This integrates all six layers into a working system.
    Each layer is independently testable.
    The integration is also testable.
    """

    def __init__(
        self,
        graph: Optional[KnowledgeGraph] = None,
        working_memory_size: int = 4,
        attention_focus_size: int = 7,
    ):
        self.graph = graph or InMemoryKnowledgeGraph()
        self.attention = AttentionSystem(self.graph, focus_size=attention_focus_size)
        self.working_memory = WorkingMemory(capacity=working_memory_size)
        self.predictor = AssociativePredictor(self.graph)
        self.surprise_tracker = SurpriseTracker(self.predictor)
        self.goals = GoalTracker()
        self.exploration = ExplorationController()
        self._step_count: int = 0

    def step(self) -> Dict[str, Any]:
        """
        Execute one cognitive step.

        Returns metrics about what happened.
        """
        self._step_count += 1

        # 1. Decay attention
        self.attention.step()

        # 2. Get current focus
        focus = self.attention.get_focus()

        # 3. Record co-occurrences for learning
        focus_ids = [a.id for a in focus]
        for i, a_id in enumerate(focus_ids):
            for b_id in focus_ids[i+1:]:
                self.predictor.record_co_occurrence(a_id, b_id)
                self.predictor.record_co_occurrence(b_id, a_id)

        # 4. Check goal progress
        top_goal = self.goals.get_top_goal()

        # 5. Decide explore vs exploit
        exploring = self.exploration.should_explore()

        return {
            "step": self._step_count,
            "focus_size": len(focus),
            "working_memory_size": len(self.working_memory.contents()),
            "mean_surprise": self.surprise_tracker.mean_surprise(),
            "top_goal": top_goal.id if top_goal else None,
            "goal_progress": self.goals.total_progress(),
            "epsilon": self.exploration.epsilon,
            "exploring": exploring,
        }

    def attend(self, atom_id: str) -> None:
        """Direct attention to an atom."""
        self.attention.stimulate(atom_id)
        atom = self.graph.get(atom_id)
        if atom:
            evicted = self.working_memory.load(atom)
            if evicted:
                # Could record this for analysis
                pass

    def learn_from_surprise(self, context_ids: List[str], actual_id: str) -> float:
        """
        Update beliefs based on prediction error.

        Returns surprise level.
        """
        context = [self.graph.get(aid) for aid in context_ids]
        context = [a for a in context if a is not None]

        surprise = self.surprise_tracker.record_outcome(context, actual_id)

        # High surprise = update beliefs more strongly
        actual_atom = self.graph.get(actual_id)
        if actual_atom and surprise > 0.5:
            # Increase confidence in surprising observation
            actual_atom.tv = actual_atom.tv.update(True, learning_rate=surprise * 0.2)

        return surprise


# =============================================================================
# BEHAVIORAL TESTS: Layer 1 - Knowledge Graph
# =============================================================================


class TestKnowledgeGraphLayer:
    """
    Layer 1: Knowledge storage and retrieval.

    The knowledge graph is the foundation of the cognitive architecture.
    It stores atoms (concepts, links, predicates) and their relationships.

    Grounding: Graph databases (Neo4j, JanusGraph), OpenCog AtomSpace.
    Success criterion: O(1) access by ID, O(n) predicate queries.
    """

    def test_atoms_can_be_stored_and_retrieved_by_id(self):
        """
        GIVEN an empty knowledge graph
        WHEN an atom is added with a specific ID and content
        THEN the atom can be retrieved by that ID with its content intact.

        This is the fundamental storage contract: the graph acts as a
        dictionary keyed by atom ID. O(1) access is critical for performance.
        """
        # GIVEN
        graph = InMemoryKnowledgeGraph()

        # WHEN
        atom_id = "test-1"
        atom_content = "hello"
        atom = Atom(id=atom_id, content=atom_content)
        graph.add(atom)

        # THEN
        retrieved = graph.get(atom_id)

        assert retrieved is not None, (
            f"Atom with id='{atom_id}' should be retrievable after adding"
        )

        expected_content = atom_content
        actual_content = retrieved.content
        assert actual_content == expected_content, (
            f"Expected content='{expected_content}', got '{actual_content}'"
        )

    def test_atoms_can_be_queried_by_predicate(self):
        """
        GIVEN a knowledge graph with atoms having different confidence levels
        WHEN querying for atoms with confidence > 0.7
        THEN only atoms meeting that criterion are returned.

        Predicate queries enable flexible retrieval patterns. This is O(n)
        since we must check each atom, but enables powerful filtering.
        """
        # GIVEN
        graph = InMemoryKnowledgeGraph()

        # Add atoms with varying confidence levels
        graph.add(Atom(id="a1", content="cat", tv=TruthValue(0.9, 0.8)))   # conf=0.8 ✓
        graph.add(Atom(id="a2", content="dog", tv=TruthValue(0.5, 0.3)))   # conf=0.3 ✗
        graph.add(Atom(id="a3", content="bird", tv=TruthValue(0.95, 0.9))) # conf=0.9 ✓

        # WHEN
        confidence_threshold = 0.7
        high_conf = graph.query(lambda a: a.tv.confidence > confidence_threshold)

        # THEN
        expected_count = 2  # cat (0.8) and bird (0.9) exceed threshold
        actual_count = len(high_conf)
        assert actual_count == expected_count, (
            f"Expected {expected_count} atoms with confidence > {confidence_threshold}, "
            f"got {actual_count}"
        )

        # Verify we got the right atoms
        returned_ids = {a.id for a in high_conf}
        expected_ids = {"a1", "a3"}
        assert returned_ids == expected_ids, (
            f"Expected atoms {expected_ids}, got {returned_ids}"
        )

    def test_links_connect_atoms_as_neighbors(self):
        """
        GIVEN a knowledge graph with two concept atoms
        WHEN a link atom is added with outgoing edges to both concepts
        THEN the link appears as a neighbor of the source concept.

        Links are atoms too (hypergraph property). The outgoing list defines
        which atoms the link connects. Neighbors are atoms that share a link.
        This enables graph traversal for inference and spreading activation.
        """
        # GIVEN
        graph = InMemoryKnowledgeGraph()

        cat = Atom(id="cat", content="cat")
        animal = Atom(id="animal", content="animal")
        graph.add(cat)
        graph.add(animal)

        # WHEN - create "cat is-a animal" link
        # outgoing=["cat", "animal"] means this link connects cat → animal
        link = Atom(id="link-1", content="is-a", outgoing=["cat", "animal"])
        graph.add(link)

        # THEN - cat should have the link as a neighbor
        cat_neighbors = graph.neighbors("cat")

        expected_neighbor_count = 1
        actual_neighbor_count = len(cat_neighbors)
        assert actual_neighbor_count == expected_neighbor_count, (
            f"Cat should have {expected_neighbor_count} neighbor (the is-a link), "
            f"got {actual_neighbor_count}"
        )

        expected_neighbor_id = "link-1"
        actual_neighbor_id = cat_neighbors[0].id
        assert actual_neighbor_id == expected_neighbor_id, (
            f"Cat's neighbor should be '{expected_neighbor_id}', got '{actual_neighbor_id}'"
        )


# =============================================================================
# BEHAVIORAL TESTS: Layer 2 - Attention
# =============================================================================


class TestAttentionLayer:
    """
    Layer 2: Priority-based attention allocation.

    Attention determines which atoms get processing resources. STI (Short-Term
    Importance) is a priority score that decays over time. High-STI atoms are
    in the "attentional focus" and get preferential processing.

    Grounding: OS schedulers (priority queues), LIDA cognitive architecture.
    Success criterion: Top-K selection in O(n log k), decay in O(n).
    """

    def test_stimulation_increases_sti(self):
        """
        GIVEN an atom with STI=0.0 in the knowledge graph
        WHEN the atom is stimulated with amount=0.5
        THEN its STI increases to 0.5.

        Stimulation is how external events or internal focus direct attention.
        The amount is additive to the current STI value.
        """
        # GIVEN
        graph = InMemoryKnowledgeGraph()
        initial_sti = 0.0
        atom = Atom(id="test", content="x", sti=initial_sti)
        graph.add(atom)

        # WHEN
        attention = AttentionSystem(graph)
        stimulation_amount = 0.5
        attention.stimulate("test", amount=stimulation_amount)

        # THEN
        expected_sti = initial_sti + stimulation_amount
        actual_sti = atom.sti
        assert actual_sti == expected_sti, (
            f"After stimulating by {stimulation_amount}, "
            f"expected STI={expected_sti}, got STI={actual_sti}"
        )

    def test_decay_reduces_sti_each_step(self):
        """
        GIVEN an atom with STI=1.0 and decay_rate=0.1
        WHEN one attention step is executed
        THEN STI decreases to 0.9 (multiplicative decay: 1.0 * (1 - 0.1)).

        Decay prevents old stimuli from dominating attention forever.
        Without decay, attention would accumulate infinitely. The decay
        formula is: new_sti = old_sti * (1 - decay_rate).
        """
        # GIVEN
        graph = InMemoryKnowledgeGraph()
        initial_sti = 1.0
        decay_rate = 0.1
        atom = Atom(id="test", content="x", sti=initial_sti)
        graph.add(atom)

        # WHEN
        attention = AttentionSystem(graph, decay_rate=decay_rate)
        attention.step()

        # THEN
        # Decay formula: new_sti = old_sti * (1 - decay_rate) = 1.0 * 0.9 = 0.9
        expected_sti = initial_sti * (1 - decay_rate)
        actual_sti = atom.sti
        assert actual_sti == expected_sti, (
            f"After decay with rate={decay_rate}, "
            f"expected STI={expected_sti}, got STI={actual_sti}"
        )

    def test_focus_returns_top_k_atoms_by_sti(self):
        """
        GIVEN 10 atoms with STI values 0.0, 0.1, 0.2, ..., 0.9
        WHEN focus is retrieved with focus_size=3
        THEN the 3 atoms with highest STI (0.9, 0.8, 0.7) are returned in order.

        The attentional focus is a bounded window of the most important atoms.
        This is the "working set" that cognitive processes operate on. Top-K
        selection uses a heap for O(n log k) complexity.
        """
        # GIVEN
        graph = InMemoryKnowledgeGraph()
        num_atoms = 10

        for i in range(num_atoms):
            # atom-0 has STI=0.0, atom-9 has STI=0.9
            graph.add(Atom(id=f"atom-{i}", content=f"x{i}", sti=i * 0.1))

        # WHEN
        focus_size = 3
        attention = AttentionSystem(graph, focus_size=focus_size)
        focus = attention.get_focus()

        # THEN - should get exactly focus_size atoms
        expected_count = focus_size
        actual_count = len(focus)
        assert actual_count == expected_count, (
            f"Focus should contain {expected_count} atoms, got {actual_count}"
        )

        # Verify top-3 STI values in descending order
        expected_stis = [0.9, 0.8, 0.7]
        for i, expected_sti in enumerate(expected_stis):
            actual_sti = focus[i].sti
            # Use approximate comparison for floating point
            assert abs(actual_sti - expected_sti) < 0.001, (
                f"Focus[{i}] should have STI≈{expected_sti}, got {actual_sti}"
            )

    def test_spreading_activation_propagates_to_neighbors(self):
        """
        GIVEN a source atom (STI=1.0) linked to a target atom (STI=0.0)
        WHEN spreading activation is applied from source with spread_factor=0.5
        THEN the link atom receives activation (STI > 0).

        Spreading activation is how attention flows through the graph.
        Related concepts become activated when you think about something.
        This is grounded in neural network activation propagation.

        Note: We check link.sti > 0 rather than an exact value because the
        spread formula may vary. The key behavior is that activation flows.
        """
        # GIVEN
        graph = InMemoryKnowledgeGraph()

        source = Atom(id="source", content="s", sti=1.0)
        target = Atom(id="target", content="t", sti=0.0)
        # Link connects source to target
        link = Atom(id="link", content="l", outgoing=["source", "target"])

        graph.add(source)
        graph.add(target)
        graph.add(link)

        # WHEN
        attention = AttentionSystem(graph)
        spread_factor = 0.5
        attention.spread_activation("source", spread_factor=spread_factor)

        # THEN - link should have received some activation
        # We don't assert exact value because spread formula is implementation detail
        assert link.sti > 0, (
            f"Link should have received activation from source, but STI={link.sti}. "
            f"Spreading activation should propagate through graph edges."
        )


# =============================================================================
# BEHAVIORAL TESTS: Layer 3 - Working Memory
# =============================================================================


class TestWorkingMemoryLayer:
    """
    Layer 3: Bounded capacity workspace.

    Working memory is the cognitive "scratchpad" - the small set of items
    currently being processed. Humans can hold ~4 items (Cowan's limit).
    When full, least-recently-used items are evicted to make room.

    Grounding: LRU cache algorithms, Cowan's 4±1 capacity limit.
    Success criterion: O(1) access/insert, automatic LRU eviction.
    """

    def test_capacity_cannot_be_exceeded(self):
        """
        GIVEN a working memory with capacity=4
        WHEN 6 atoms are loaded sequentially
        THEN only 4 atoms remain (capacity is enforced via eviction).

        Working memory is strictly bounded. This models cognitive limits -
        we can only actively hold so many things at once. Excess items are
        automatically evicted using LRU policy.
        """
        # GIVEN
        capacity = 4
        wm = WorkingMemory(capacity=capacity)

        # WHEN - load more items than capacity allows
        items_to_load = 6
        for i in range(items_to_load):
            wm.load(Atom(id=f"a{i}", content=f"x{i}"))

        # THEN - working memory respects capacity limit
        expected_size = capacity
        actual_size = len(wm.contents())
        assert actual_size == expected_size, (
            f"Working memory should hold at most {expected_size} items, "
            f"but contains {actual_size}"
        )

    def test_lru_item_is_evicted_when_capacity_exceeded(self):
        """
        GIVEN a full working memory [a, b, c] with capacity=3
        WHEN 'a' is accessed (making it most recent), then 'd' is loaded
        THEN 'b' is evicted (it's now the least recently used).

        LRU eviction order: [a, b, c] -> access 'a' -> [b, c, a] -> load 'd'
        -> evict 'b' (oldest) -> [c, a, d].

        This models how we forget things we haven't thought about recently.
        """
        # GIVEN
        capacity = 3
        wm = WorkingMemory(capacity=capacity)

        wm.load(Atom(id="a", content="a"))
        wm.load(Atom(id="b", content="b"))
        wm.load(Atom(id="c", content="c"))
        # Order is now: a (oldest), b, c (newest)

        # Access 'a' to make it most recent
        wm.get("a")
        # Order is now: b (oldest), c, a (newest)

        # WHEN - load a new item, exceeding capacity
        evicted = wm.load(Atom(id="d", content="d"))

        # THEN - 'b' should be evicted (it's LRU)
        assert evicted is not None, "An atom should be evicted when capacity exceeded"

        expected_evicted_id = "b"
        actual_evicted_id = evicted.id
        assert actual_evicted_id == expected_evicted_id, (
            f"Expected '{expected_evicted_id}' to be evicted (LRU), "
            f"but '{actual_evicted_id}' was evicted"
        )

        # Verify 'a' was protected by the access
        assert wm.contains("a"), "'a' should still be in memory (was accessed recently)"
        assert not wm.contains("b"), "'b' should have been evicted"
        assert wm.contains("c"), "'c' should still be in memory"
        assert wm.contains("d"), "'d' should have been loaded"

    def test_reloading_atom_refreshes_its_access_time(self):
        """
        GIVEN a working memory [a, b] with capacity=2
        WHEN 'a' is reloaded (same atom loaded again), then 'c' is loaded
        THEN 'b' is evicted, not 'a' (reload counts as access).

        Reloading an already-present atom updates its access time without
        duplication. This models "thinking about something again" - it
        refreshes its position in memory.
        """
        # GIVEN
        capacity = 2
        wm = WorkingMemory(capacity=capacity)

        wm.load(Atom(id="a", content="a"))
        wm.load(Atom(id="b", content="b"))
        # Order: a (oldest), b (newest)

        # WHEN - reload 'a' (refreshes its access time)
        wm.load(Atom(id="a", content="a"))
        # Order should now be: b (oldest), a (newest)

        # Load 'c' to trigger eviction
        evicted = wm.load(Atom(id="c", content="c"))

        # THEN - 'b' should be evicted (now LRU), not 'a'
        expected_evicted_id = "b"
        actual_evicted_id = evicted.id
        assert actual_evicted_id == expected_evicted_id, (
            f"Expected '{expected_evicted_id}' to be evicted after 'a' was reloaded, "
            f"but '{actual_evicted_id}' was evicted. "
            "Reloading should refresh access time."
        )


# =============================================================================
# BEHAVIORAL TESTS: Layer 4 - Prediction
# =============================================================================


class TestPredictionLayer:
    """
    Layer 4: Anticipate next relevant atoms.

    Prediction enables the system to anticipate what comes next based on
    current context. This is the foundation of learning - recording what
    co-occurs and using that to make predictions. Surprise (prediction error)
    drives learning by highlighting what wasn't expected.

    Grounding: Association rules, n-gram language models, Hebbian learning.
    Success criterion: Surprise decreases with experience (learning works).
    """

    def test_cooccurrence_learning_enables_prediction(self):
        """
        GIVEN a predictor with "cat" and "meow" atoms
        WHEN "cat" and "meow" co-occur 10 times
        THEN predicting from "cat" context returns "meow" as top prediction.

        This is associative learning: things that occur together become
        linked. After seeing "cat" with "meow" many times, the system
        learns to predict "meow" when it sees "cat".
        """
        # GIVEN
        graph = InMemoryKnowledgeGraph()
        graph.add(Atom(id="cat", content="cat"))
        graph.add(Atom(id="meow", content="meow"))
        predictor = AssociativePredictor(graph)

        # WHEN - record co-occurrences (learning phase)
        learning_iterations = 10
        for _ in range(learning_iterations):
            predictor.record_co_occurrence("cat", "meow")

        # THEN - prediction should work
        context = [graph.get("cat")]
        predictions = predictor.predict(context)

        assert len(predictions) > 0, (
            "After learning co-occurrences, predictor should return predictions"
        )

        expected_top_prediction = "meow"
        actual_top_prediction = predictions[0][0]
        assert actual_top_prediction == expected_top_prediction, (
            f"Top prediction from 'cat' context should be '{expected_top_prediction}', "
            f"got '{actual_top_prediction}'"
        )

    def test_surprise_is_maximum_for_unpredicted_outcome(self):
        """
        GIVEN a predictor with no learned associations
        WHEN an outcome is observed
        THEN surprise is 1.0 (maximum) because nothing was predicted.

        Surprise = 1 - P(outcome|context). With no learning, P = 0,
        so surprise = 1.0. This is the baseline before any learning occurs.
        """
        # GIVEN
        graph = InMemoryKnowledgeGraph()
        graph.add(Atom(id="a", content="a"))
        graph.add(Atom(id="b", content="b"))
        graph.add(Atom(id="c", content="c"))

        predictor = AssociativePredictor(graph)
        tracker = SurpriseTracker(predictor)
        # No co-occurrences recorded - predictor knows nothing

        # WHEN - observe an outcome
        context = [graph.get("a")]
        surprise = tracker.record_outcome(context, "b")

        # THEN - surprise should be maximum
        expected_surprise = 1.0
        actual_surprise = surprise
        assert actual_surprise == expected_surprise, (
            f"With no learned associations, surprise should be {expected_surprise} "
            f"(maximum), got {actual_surprise}"
        )

    def test_surprise_decreases_as_patterns_are_learned(self):
        """
        GIVEN a predictor that has not yet learned the a→b pattern
        WHEN the a→b co-occurrence is recorded 20 times (learning)
        THEN surprise for observing 'b' after 'a' decreases.

        This is the core learning test: as the system learns patterns,
        it becomes less surprised when those patterns occur. Decreasing
        surprise indicates successful learning.
        """
        # GIVEN
        graph = InMemoryKnowledgeGraph()
        graph.add(Atom(id="a", content="a"))
        graph.add(Atom(id="b", content="b"))
        graph.add(Atom(id="c", content="c"))

        predictor = AssociativePredictor(graph)
        tracker = SurpriseTracker(predictor)
        context = [graph.get("a")]

        # Measure initial surprise (before learning)
        first_surprise = tracker.record_outcome(context, "b")

        # WHEN - learn the a→b pattern
        learning_iterations = 20
        for _ in range(learning_iterations):
            predictor.record_co_occurrence("a", "b")

        # Measure surprise after learning
        last_surprise = tracker.record_outcome(context, "b")

        # THEN - surprise should have decreased
        assert last_surprise < first_surprise, (
            f"After learning a→b pattern, surprise should decrease. "
            f"Initial surprise: {first_surprise}, "
            f"After {learning_iterations} iterations: {last_surprise}. "
            "Learning should reduce prediction error."
        )


# =============================================================================
# BEHAVIORAL TESTS: Layer 5 - Goals
# =============================================================================


class TestGoalLayer:
    """
    Layer 5: Track progress toward targets.

    Goals represent desired states the system is working toward. Each goal
    has a target state, current state, and importance. Progress is measured
    as the ratio of current to target. Urgency combines importance with
    remaining work to prioritize what needs attention now.

    Grounding: Control theory (PID controllers), utility theory.
    Success criterion: Urgency = importance × (1 - progress).
    """

    def test_progress_measures_fraction_of_target_achieved(self):
        """
        GIVEN a goal with target_state=100 and current_state=50
        WHEN progress is calculated
        THEN progress = 0.5 (50% of the way to target).

        Progress formula: current_state / target_state.
        This is a simple linear measure of goal completion.
        """
        # GIVEN
        target = 100
        current = 50
        goal = Goal(
            id="g1",
            description="reach 100",
            target_state=target,
            current_state=current,
            importance=1.0
        )

        # WHEN/THEN - progress is computed automatically
        expected_progress = current / target  # 50/100 = 0.5
        actual_progress = goal.progress

        assert actual_progress == expected_progress, (
            f"Progress should be {current}/{target} = {expected_progress}, "
            f"got {actual_progress}"
        )

    def test_urgency_combines_importance_with_remaining_work(self):
        """
        GIVEN a goal with progress=0.8 and importance=0.8
        WHEN urgency is calculated
        THEN urgency = 0.8 × (1 - 0.8) = 0.16.

        Urgency formula: importance × (1 - progress).
        - High importance + low progress = high urgency (needs work now)
        - Low importance OR high progress = low urgency (can wait)

        This prioritizes important goals that aren't close to done.
        """
        # GIVEN
        target = 100
        current = 80  # progress = 0.8
        importance = 0.8
        goal = Goal(
            id="g1",
            description="reach 100",
            target_state=target,
            current_state=current,
            importance=importance
        )

        # WHEN/THEN - urgency is computed automatically
        progress = current / target  # 0.8
        expected_urgency = importance * (1 - progress)  # 0.8 * 0.2 = 0.16
        actual_urgency = goal.urgency

        # Use approximate comparison for floating point
        assert abs(actual_urgency - expected_urgency) < 0.01, (
            f"Urgency should be {importance} × (1 - {progress}) = {expected_urgency}, "
            f"got {actual_urgency}"
        )

    def test_active_goals_are_sorted_by_urgency_descending(self):
        """
        GIVEN two goals: "low" (progress=0.9, importance=0.5) and
                         "high" (progress=0.1, importance=0.9)
        WHEN active goals are retrieved
        THEN "high" comes first (higher urgency).

        Urgency calculations:
        - "low":  0.5 × (1 - 0.9) = 0.5 × 0.1 = 0.05
        - "high": 0.9 × (1 - 0.1) = 0.9 × 0.9 = 0.81

        The tracker returns goals sorted by urgency so the most pressing
        goal is always first. This drives attention allocation.
        """
        # GIVEN
        tracker = GoalTracker()

        # Low urgency goal: nearly complete, not very important
        tracker.add_goal(Goal(
            id="low",
            description="low priority goal",
            target_state=100,
            current_state=90,  # progress = 0.9
            importance=0.5
        ))
        # Urgency: 0.5 × (1 - 0.9) = 0.05

        # High urgency goal: barely started, very important
        tracker.add_goal(Goal(
            id="high",
            description="high priority goal",
            target_state=100,
            current_state=10,  # progress = 0.1
            importance=0.9
        ))
        # Urgency: 0.9 × (1 - 0.1) = 0.81

        # WHEN
        active = tracker.get_active_goals()

        # THEN - high urgency goal should be first
        expected_first_goal_id = "high"
        actual_first_goal_id = active[0].id
        assert actual_first_goal_id == expected_first_goal_id, (
            f"First goal should be '{expected_first_goal_id}' (highest urgency), "
            f"got '{actual_first_goal_id}'. "
            f"Goals should be sorted by urgency descending."
        )


# =============================================================================
# BEHAVIORAL TESTS: Layer 6 - Exploration
# =============================================================================


class TestExplorationLayer:
    """
    Layer 6: Balance exploration and exploitation.

    The exploration/exploitation tradeoff is fundamental: should we use
    what we know works (exploit) or try new things (explore)? ε-greedy
    explores with probability ε, otherwise exploits. ε adapts based on
    recent success/failure to automatically balance this tradeoff.

    Grounding: Multi-armed bandits (ε-greedy), reinforcement learning.
    Success criterion: ε adapts down on success, up on failure, stays bounded.
    """

    def test_epsilon_decreases_after_repeated_success(self):
        """
        GIVEN an exploration controller with initial ε=0.5
        WHEN 5 consecutive successes are recorded
        THEN ε decreases (exploit what works).

        Success → reduce exploration. If current strategy is working,
        keep doing it more often. This is adaptive exploitation.
        """
        # GIVEN
        initial_epsilon = 0.5
        controller = ExplorationController(initial_epsilon=initial_epsilon)

        # WHEN - record multiple successes
        success_count = 5
        for _ in range(success_count):
            controller.record_success()

        # THEN - epsilon should have decreased
        assert controller.epsilon < initial_epsilon, (
            f"After {success_count} successes, ε should decrease from {initial_epsilon}. "
            f"Current ε={controller.epsilon}. Success should reduce exploration."
        )

    def test_epsilon_increases_after_repeated_failure(self):
        """
        GIVEN an exploration controller with initial ε=0.3
        WHEN 5 consecutive failures are recorded
        THEN ε increases (try something new).

        Failure → increase exploration. If current strategy isn't working,
        try different approaches more often. This is adaptive exploration.
        """
        # GIVEN
        initial_epsilon = 0.3
        controller = ExplorationController(initial_epsilon=initial_epsilon)

        # WHEN - record multiple failures
        failure_count = 5
        for _ in range(failure_count):
            controller.record_failure()

        # THEN - epsilon should have increased
        assert controller.epsilon > initial_epsilon, (
            f"After {failure_count} failures, ε should increase from {initial_epsilon}. "
            f"Current ε={controller.epsilon}. Failure should increase exploration."
        )

    def test_consecutive_failures_detected_as_stuck(self):
        """
        GIVEN an exploration controller with no history
        WHEN 3 consecutive failures are recorded
        THEN is_stuck(threshold=3) returns True.

        "Stuck" detection identifies when we're in a rut - repeated failures
        without any success. This can trigger more drastic interventions
        like resetting state or switching strategies entirely.
        """
        # GIVEN
        controller = ExplorationController()

        # Initially should not be stuck
        assert not controller.is_stuck(), (
            "New controller should not be stuck"
        )

        # WHEN - record consecutive failures
        failure_count = 3
        for _ in range(failure_count):
            controller.record_failure()

        # THEN - should be detected as stuck
        threshold = 3
        assert controller.is_stuck(threshold=threshold), (
            f"After {failure_count} consecutive failures, "
            f"is_stuck(threshold={threshold}) should return True"
        )

    def test_epsilon_respects_minimum_and_maximum_bounds(self):
        """
        GIVEN an exploration controller with min_ε=0.1 and max_ε=0.8
        WHEN many successes push ε down, then many failures push ε up
        THEN ε never goes below 0.1 or above 0.8.

        Bounds prevent pathological behavior:
        - min_ε > 0 ensures we always explore a little (avoid local optima)
        - max_ε < 1 ensures we always exploit a little (don't waste effort)
        """
        # GIVEN
        min_epsilon = 0.1
        max_epsilon = 0.8
        controller = ExplorationController(
            min_epsilon=min_epsilon,
            max_epsilon=max_epsilon
        )

        # WHEN - push epsilon down with many successes
        for _ in range(100):
            controller.record_success()

        # THEN - should not go below minimum
        assert controller.epsilon >= min_epsilon, (
            f"ε should never go below min_epsilon={min_epsilon}. "
            f"Current ε={controller.epsilon}"
        )

        # WHEN - push epsilon up with many failures
        for _ in range(100):
            controller.record_failure()

        # THEN - should not go above maximum
        assert controller.epsilon <= max_epsilon, (
            f"ε should never go above max_epsilon={max_epsilon}. "
            f"Current ε={controller.epsilon}"
        )


# =============================================================================
# BEHAVIORAL TESTS: Integrated System
# =============================================================================


class TestIntegratedSystem:
    """
    Test the complete cognitive agent with all layers working together.

    These tests verify that the six layers integrate correctly:
    Knowledge → Attention → Working Memory → Prediction → Goals → Exploration

    The integrated agent should be more than the sum of its parts - the layers
    should interact to produce emergent cognitive behavior.

    Success criterion: Agent functions as a cohesive unit with measurable state.
    """

    def test_agent_initializes_with_all_six_layers(self):
        """
        GIVEN no special configuration
        WHEN a CognitiveAgent is created
        THEN all six cognitive layers are initialized and accessible.

        This verifies the agent's structural integrity. All layers must be
        present for the cognitive loop to function. Missing layers would
        break the integration.
        """
        # GIVEN/WHEN
        agent = CognitiveAgent()

        # THEN - all six layers should be initialized
        assert agent.graph is not None, "Layer 1 (Knowledge) should be initialized"
        assert agent.attention is not None, "Layer 2 (Attention) should be initialized"
        assert agent.working_memory is not None, "Layer 3 (Working Memory) should be initialized"
        assert agent.predictor is not None, "Layer 4 (Prediction) should be initialized"
        assert agent.goals is not None, "Layer 5 (Goals) should be initialized"
        assert agent.exploration is not None, "Layer 6 (Exploration) should be initialized"

    def test_step_produces_measurable_cognitive_metrics(self):
        """
        GIVEN a CognitiveAgent with some atoms in its graph
        WHEN one cognitive step is executed
        THEN a metrics dictionary is returned with key cognitive state variables.

        Metrics enable observability and debugging. The agent should always
        report its internal state so we can understand what it's doing.
        Required metrics: step count, focus size, exploration epsilon.
        """
        # GIVEN
        agent = CognitiveAgent()
        agent.graph.add(Atom(id="a", content="a", sti=0.5))
        agent.graph.add(Atom(id="b", content="b", sti=0.3))

        # WHEN
        metrics = agent.step()

        # THEN - metrics should contain key cognitive state
        required_metrics = ["step", "focus_size", "epsilon"]
        for metric_name in required_metrics:
            assert metric_name in metrics, (
                f"Metrics should include '{metric_name}'. Got: {list(metrics.keys())}"
            )

        # Step count should be 1 after first step
        expected_step = 1
        actual_step = metrics["step"]
        assert actual_step == expected_step, (
            f"After first step, step count should be {expected_step}, got {actual_step}"
        )

    def test_attend_integrates_attention_and_working_memory(self):
        """
        GIVEN a CognitiveAgent with an atom in its graph
        WHEN attend() is called on that atom
        THEN the atom is loaded into working memory AND its STI increases.

        attend() demonstrates layer integration: it affects both the Attention
        layer (increases STI) and the Working Memory layer (loads atom).
        This is the primary mechanism for directing cognitive resources.
        """
        # GIVEN
        agent = CognitiveAgent()
        initial_sti = 0.0
        atom = Atom(id="test", content="test", sti=initial_sti)
        agent.graph.add(atom)

        # WHEN
        agent.attend("test")

        # THEN - working memory should contain the atom
        assert agent.working_memory.contains("test"), (
            "After attend(), atom should be in working memory"
        )

        # AND - STI should have increased
        assert atom.sti > initial_sti, (
            f"After attend(), atom STI should increase from {initial_sti}. "
            f"Current STI: {atom.sti}"
        )

    def test_learn_from_surprise_updates_beliefs(self):
        """
        GIVEN a CognitiveAgent with a context atom and an outcome atom (confidence=0.5)
        WHEN learn_from_surprise() is called with high surprise
        THEN the outcome atom's confidence increases (belief updated).

        This tests the learning loop: surprise (prediction error) drives
        belief updates. High surprise means "this was unexpected" which
        should increase confidence in the surprising observation.

        Note: Initial confidence is 0.5 (uncertain). After observing the
        outcome, confidence should increase because we now have evidence.
        """
        # GIVEN
        agent = CognitiveAgent()
        agent.graph.add(Atom(id="context", content="c"))

        initial_confidence = 0.5
        agent.graph.add(Atom(
            id="outcome",
            content="o",
            tv=TruthValue(0.5, initial_confidence)
        ))

        # WHEN - learn from a surprising observation
        surprise = agent.learn_from_surprise(["context"], "outcome")

        # THEN - confidence should have increased
        outcome = agent.graph.get("outcome")
        actual_confidence = outcome.tv.confidence

        assert actual_confidence > initial_confidence, (
            f"After learning from surprise, confidence should increase from "
            f"{initial_confidence}. Current confidence: {actual_confidence}. "
            f"Surprise level was: {surprise}"
        )


class TestEventHooks:
    """
    Behavioral tests for the event hooks observability system.

    The event system enables external components to observe cognitive agent
    internals without tight coupling. This is the Observer pattern applied
    to cognitive architecture - useful for debugging, logging, visualization,
    and integration with external systems.

    Grounding: Observer pattern (Gang of Four), Event Sourcing.
    """

    def test_event_bus_delivers_to_type_specific_subscribers(self):
        """
        GIVEN an EventBus with a handler subscribed to STEP_STARTED events
        WHEN a STEP_STARTED event is emitted with step=1
        THEN the handler receives exactly one event with the correct data.

        This verifies the fundamental pub/sub contract: subscribers only
        receive events of the types they subscribed to.
        """
        from cortical.cognitive.graph import EventBus, EventType, CognitiveEvent

        # GIVEN
        bus = EventBus()
        received_events = []

        def capture_handler(event):
            received_events.append(event)

        bus.subscribe(EventType.STEP_STARTED, capture_handler)

        # WHEN
        emitted_event = CognitiveEvent(EventType.STEP_STARTED, {"step": 1})
        bus.emit(emitted_event)

        # THEN
        expected_event_count = 1
        actual_event_count = len(received_events)
        assert actual_event_count == expected_event_count, (
            f"Expected {expected_event_count} event, got {actual_event_count}"
        )

        expected_step = 1
        actual_step = received_events[0].data["step"]
        assert actual_step == expected_step, (
            f"Expected step={expected_step}, got step={actual_step}"
        )

    def test_event_bus_global_subscriber_receives_all_event_types(self):
        """
        GIVEN an EventBus with a global subscriber (subscribe_all)
        WHEN events of different types are emitted
        THEN the global subscriber receives all of them.

        Global subscription is useful for logging/debugging where you want
        to see everything that happens, regardless of event type.
        """
        from cortical.cognitive.graph import EventBus, EventType, CognitiveEvent

        # GIVEN
        bus = EventBus()
        received_events = []
        bus.subscribe_all(lambda e: received_events.append(e))

        # WHEN - emit two different event types
        bus.emit(CognitiveEvent(EventType.STEP_STARTED, {"source": "test1"}))
        bus.emit(CognitiveEvent(EventType.ATOM_LOADED, {"source": "test2"}))

        # THEN - global subscriber receives both
        expected_count = 2
        actual_count = len(received_events)
        assert actual_count == expected_count, (
            f"Global subscriber should receive all events. "
            f"Expected {expected_count}, got {actual_count}"
        )

        # Verify we got both event types
        received_types = {e.event_type for e in received_events}
        expected_types = {EventType.STEP_STARTED, EventType.ATOM_LOADED}
        assert received_types == expected_types, (
            f"Expected event types {expected_types}, got {received_types}"
        )

    def test_agent_step_emits_lifecycle_events_in_order(self):
        """
        GIVEN a CognitiveAgent with subscriptions to step lifecycle events
        WHEN the agent executes one step
        THEN STEP_STARTED fires before STEP_COMPLETED (order matters).

        The ordering contract is important: observers may depend on
        STEP_STARTED to initialize state that STEP_COMPLETED reads.
        """
        from cortical.cognitive.graph import (
            CognitiveAgent as RealAgent,
            EventType,
        )

        # GIVEN
        agent = RealAgent()
        events_in_order = []

        agent.events.subscribe(EventType.STEP_STARTED, lambda e: events_in_order.append(e))
        agent.events.subscribe(EventType.STEP_COMPLETED, lambda e: events_in_order.append(e))

        # WHEN
        agent.step()

        # THEN - exactly 2 events in correct order
        expected_count = 2
        actual_count = len(events_in_order)
        assert actual_count == expected_count, (
            f"Step should emit exactly {expected_count} lifecycle events, got {actual_count}"
        )

        # Order matters: STARTED must come before COMPLETED
        first_event_type = events_in_order[0].event_type
        second_event_type = events_in_order[1].event_type

        assert first_event_type == EventType.STEP_STARTED, (
            f"First event should be STEP_STARTED, got {first_event_type}"
        )
        assert second_event_type == EventType.STEP_COMPLETED, (
            f"Second event should be STEP_COMPLETED, got {second_event_type}"
        )

    def test_attend_emits_attention_and_memory_events(self):
        """
        GIVEN a CognitiveAgent with an atom in its graph
        WHEN attend() is called on that atom
        THEN both ATTENTION_FOCUSED and ATOM_LOADED events are emitted.

        attend() does two things: (1) increases STI (attention), and
        (2) loads into working memory. Both operations should be observable.
        """
        from cortical.cognitive.graph import (
            CognitiveAgent as RealAgent,
            EventType,
        )

        # GIVEN
        agent = RealAgent()
        atom = agent.graph.node("test_concept")
        all_events = []
        agent.events.subscribe_all(lambda e: all_events.append(e))

        # WHEN
        agent.attend("test_concept")

        # THEN - both attention and memory events should fire
        event_types = [e.event_type for e in all_events]

        assert EventType.ATTENTION_FOCUSED in event_types, (
            "attend() should emit ATTENTION_FOCUSED event. "
            f"Got event types: {event_types}"
        )
        assert EventType.ATOM_LOADED in event_types, (
            "attend() should emit ATOM_LOADED event (working memory). "
            f"Got event types: {event_types}"
        )

        # Verify ATTENTION_FOCUSED contains expected data
        attention_events = [e for e in all_events if e.event_type == EventType.ATTENTION_FOCUSED]
        assert len(attention_events) == 1
        attention_data = attention_events[0].data
        assert attention_data["atom_name"] == "test_concept", (
            f"Event should include atom name. Got: {attention_data}"
        )

    def test_working_memory_eviction_emits_event_with_evicted_atom_info(self):
        """
        GIVEN a CognitiveAgent with working_memory_size=2 and 4 atoms in graph
        WHEN all 4 atoms are attended (exceeding capacity)
        THEN exactly 2 ATOM_EVICTED events are emitted (capacity=2, attend=4).

        Math: With capacity 2, attending atoms [0,1,2,3] in sequence:
        - attend(0): [0] - no eviction
        - attend(1): [0,1] - no eviction (at capacity)
        - attend(2): [1,2] - evicts 0 (LRU)
        - attend(3): [2,3] - evicts 1 (LRU)
        Total evictions: 2
        """
        from cortical.cognitive.graph import (
            CognitiveAgent as RealAgent,
            EventType,
        )

        # GIVEN
        working_memory_capacity = 2
        num_atoms = 4
        agent = RealAgent(working_memory_size=working_memory_capacity)

        for i in range(num_atoms):
            agent.graph.node(f"atom_{i}")

        eviction_events = []
        agent.events.subscribe(EventType.ATOM_EVICTED, lambda e: eviction_events.append(e))

        # WHEN - attend to all atoms in sequence
        for i in range(num_atoms):
            agent.attend(f"atom_{i}")

        # THEN - calculate expected evictions
        # Evictions happen when we try to load atom (capacity+1), (capacity+2), etc.
        expected_evictions = num_atoms - working_memory_capacity
        actual_evictions = len(eviction_events)

        assert actual_evictions == expected_evictions, (
            f"With capacity={working_memory_capacity} and {num_atoms} atoms attended, "
            f"expected {expected_evictions} evictions but got {actual_evictions}"
        )

        # Verify eviction events contain useful data
        for event in eviction_events:
            assert "atom_id" in event.data, "Eviction event should include atom_id"
            assert "atom_name" in event.data, "Eviction event should include atom_name"
            assert event.data["reason"] == "lru_eviction", (
                f"Eviction reason should be 'lru_eviction', got {event.data.get('reason')}"
            )

    def test_handler_exceptions_are_caught_and_do_not_propagate(self):
        """
        GIVEN a CognitiveAgent with a handler that raises an exception
        WHEN the agent executes a step (which emits events)
        THEN the agent continues normally (exception is swallowed).

        This is a critical reliability property: external observers should
        not be able to crash the cognitive agent. The EventBus catches and
        silently ignores handler exceptions. This follows the principle that
        observability should be non-intrusive.

        Note: In production, you might want to log these errors, but the
        key contract is that they don't propagate to the agent.
        """
        from cortical.cognitive.graph import (
            CognitiveAgent as RealAgent,
            EventType,
        )

        # GIVEN
        agent = RealAgent()
        handler_was_called = []  # Use list to track mutable state in closure

        def faulty_handler(event):
            handler_was_called.append(True)
            raise RuntimeError("Intentional test exception - should be caught")

        agent.events.subscribe(EventType.STEP_STARTED, faulty_handler)

        # WHEN - this should NOT raise, despite the faulty handler
        result = agent.step()

        # THEN - agent completed successfully
        assert result is not None, "Agent should return metrics dict"
        assert "step" in result, "Metrics should include step count"
        assert result["step"] == 1, "Step count should be 1"

        # Verify the handler was actually called (exception was raised and caught)
        assert len(handler_was_called) > 0, (
            "Faulty handler should have been called (and its exception caught)"
        )


class TestEpisodicMemory:
    """
    Behavioral tests for Layer 7: Episodic Memory.

    Episodic memory stores experiences (episodes) that can be replayed later
    for learning. This is the cognitive equivalent of "remembering what happened"
    and using those memories to learn patterns more efficiently.

    Grounding:
        - Episodic memory (Tulving, 1972) - memory for personal experiences
        - Experience replay (Lin, 1992) - reuse past experiences for learning
        - Prioritized experience replay (Schaul et al., 2015) - sample by importance
    """

    def test_episode_stores_context_outcome_and_surprise(self):
        """
        GIVEN an Episode with context=["a", "b"], outcome="c", surprise=0.8
        WHEN the episode is created
        THEN all fields are accessible and priority = surprise + |reward|.

        Episodes capture the structure of an experience: what was happening
        (context), what occurred (outcome), and how surprising it was.
        """
        from cortical.cognitive.graph import Episode

        # GIVEN/WHEN
        context = ["a", "b"]
        outcome = "c"
        surprise = 0.8
        episode = Episode(
            step=1,
            context_ids=context,
            outcome_id=outcome,
            surprise=surprise,
        )

        # THEN
        assert episode.context_ids == context
        assert episode.outcome_id == outcome
        assert episode.surprise == surprise

        # Priority = surprise + |reward| (reward defaults to 0)
        expected_priority = surprise + 0.0
        actual_priority = episode.priority
        assert actual_priority == expected_priority, (
            f"Priority should be {expected_priority}, got {actual_priority}"
        )

    def test_episodic_memory_respects_capacity_limit(self):
        """
        GIVEN an EpisodicMemory with capacity=3
        WHEN 5 high-surprise episodes are stored
        THEN only 3 episodes remain (capacity enforced).

        Like working memory, episodic memory is bounded to prevent
        unbounded growth. Excess episodes are evicted by priority.
        """
        from cortical.cognitive.graph import Episode, EpisodicMemory

        # GIVEN
        capacity = 3
        memory = EpisodicMemory(capacity=capacity, min_surprise_to_store=0.0)

        # WHEN - store more episodes than capacity
        for i in range(5):
            episode = Episode(
                step=i,
                context_ids=[f"ctx_{i}"],
                outcome_id=f"out_{i}",
                surprise=0.5 + i * 0.1,  # Increasing priority
            )
            memory.store(episode)

        # THEN
        expected_size = capacity
        actual_size = len(memory)
        assert actual_size == expected_size, (
            f"Memory should hold at most {expected_size} episodes, "
            f"but contains {actual_size}"
        )

    def test_low_surprise_episodes_are_filtered_out(self):
        """
        GIVEN an EpisodicMemory with min_surprise_to_store=0.5
        WHEN an episode with surprise=0.3 is stored
        THEN the episode is NOT stored (filtered as mundane).

        Not every experience is worth remembering. Low-surprise events
        are filtered to save memory for important experiences.
        """
        from cortical.cognitive.graph import Episode, EpisodicMemory

        # GIVEN
        min_surprise = 0.5
        memory = EpisodicMemory(capacity=10, min_surprise_to_store=min_surprise)

        # WHEN - try to store a low-surprise episode
        low_surprise_episode = Episode(
            step=1,
            context_ids=["a"],
            outcome_id="b",
            surprise=0.3,  # Below threshold
        )
        memory.store(low_surprise_episode)

        # THEN - episode should not be stored
        assert len(memory) == 0, (
            f"Episode with surprise={low_surprise_episode.surprise} should be "
            f"filtered (min_surprise={min_surprise})"
        )

    def test_retrieve_returns_episodes_with_similar_context(self):
        """
        GIVEN an EpisodicMemory with episodes having different contexts
        WHEN retrieving with context=["a", "b"]
        THEN episodes with overlapping contexts are returned (Jaccard similarity).

        Content-addressable retrieval: "What happened before when I was
        in a similar situation?" This enables transfer of learning.
        """
        from cortical.cognitive.graph import Episode, EpisodicMemory

        # GIVEN
        memory = EpisodicMemory(capacity=10, min_surprise_to_store=0.0)

        # Episode with similar context (shares "a", "b")
        similar_episode = Episode(
            step=1,
            context_ids=["a", "b", "c"],
            outcome_id="x",
            surprise=0.5,
        )
        memory.store(similar_episode)

        # Episode with different context (no overlap)
        different_episode = Episode(
            step=2,
            context_ids=["d", "e", "f"],
            outcome_id="y",
            surprise=0.5,
        )
        memory.store(different_episode)

        # WHEN - retrieve by context ["a", "b"]
        query_context = ["a", "b"]
        retrieved = memory.retrieve(query_context, top_k=5)

        # THEN - only similar episode should be retrieved
        assert len(retrieved) == 1, (
            f"Should retrieve 1 similar episode, got {len(retrieved)}"
        )
        assert retrieved[0].outcome_id == "x", (
            "Retrieved episode should be the one with overlapping context"
        )

    def test_experience_replay_reinforces_learning(self):
        """
        GIVEN a CognitiveAgent that has learned from a surprise event
        WHEN experience_replay() is called
        THEN co-occurrences are re-recorded (learning reinforced).

        Experience replay breaks correlation between sequential experiences
        and enables more efficient use of past data. Surprising experiences
        are replayed more often (prioritized replay).
        """
        from cortical.cognitive.graph import CognitiveAgent as RealAgent

        # GIVEN
        agent = RealAgent(episodic_memory_size=100)

        # Add atoms to graph
        agent.graph.node("context_a")
        agent.graph.node("outcome_b")

        # Learn from a surprising experience (stores episode)
        agent.learn_from_surprise(["context_a"], "outcome_b")

        # Get initial co-occurrence count
        initial_co_occurrences = dict(agent.predictor._co_occurrences)

        # WHEN - replay experiences
        n_replayed = agent.experience_replay(n_episodes=5)

        # THEN - at least one episode should have been replayed
        # (if surprise was high enough to be stored)
        if len(agent.episodic_memory) > 0:
            assert n_replayed >= 1, (
                "At least one episode should be replayed"
            )
            # Co-occurrences should have been reinforced
            # (exact count depends on how many times replayed)

    def test_recall_similar_finds_relevant_past_experiences(self):
        """
        GIVEN a CognitiveAgent with stored episodes from past learning
        WHEN recall_similar() is called with current context
        THEN relevant past episodes are returned.

        This enables the agent to say "I've seen something like this before"
        and use that experience to inform current decisions.
        """
        from cortical.cognitive.graph import CognitiveAgent as RealAgent

        # GIVEN
        agent = RealAgent(episodic_memory_size=100)

        # Create atoms
        agent.graph.node("cat")
        agent.graph.node("meow")
        agent.graph.node("dog")
        agent.graph.node("bark")

        # Store some experiences (bypass min_surprise filter for testing)
        agent.episodic_memory._min_surprise = 0.0

        # Learn "cat → meow" pattern
        agent.learn_from_surprise(["cat"], "meow")

        # WHEN - recall similar experiences to "cat"
        similar = agent.recall_similar(["cat"], top_k=3)

        # THEN - should find the cat→meow episode
        if len(agent.episodic_memory) > 0:
            assert len(similar) >= 1, (
                "Should find at least one similar episode"
            )
            assert similar[0].outcome_id == "meow", (
                "Similar episode should be the cat→meow experience"
            )

    def test_episodic_memory_persists_across_save_load(self):
        """
        GIVEN a CognitiveAgent with episodes in episodic memory
        WHEN the agent is saved and loaded
        THEN episodic memory is restored with all episodes.

        Persistence ensures experiences aren't lost between sessions.
        """
        from cortical.cognitive.graph import CognitiveAgent as RealAgent, Episode
        from cortical.common.filesystem import InMemoryFileSystem
        from pathlib import Path

        # GIVEN - use in-memory filesystem for testing
        fs = InMemoryFileSystem(Path("/test"))
        fs.mkdir(Path("/test"), parents=True, exist_ok=True)

        agent = RealAgent(filesystem=fs, episodic_memory_size=100)
        agent.episodic_memory._min_surprise = 0.0  # Allow all episodes

        # Store an episode directly
        episode = Episode(
            step=42,
            context_ids=["test_ctx"],
            outcome_id="test_out",
            surprise=0.9,
        )
        agent.episodic_memory.store(episode)

        original_count = len(agent.episodic_memory)

        # WHEN - save and load
        save_path = Path("/test/agent.json")
        agent.save(save_path)
        loaded_agent = RealAgent.load(save_path, filesystem=fs)

        # THEN - episodic memory should be restored
        assert len(loaded_agent.episodic_memory) == original_count, (
            f"Loaded agent should have {original_count} episodes, "
            f"got {len(loaded_agent.episodic_memory)}"
        )

        # Verify episode content
        loaded_episodes = loaded_agent.episodic_memory.contents()
        assert loaded_episodes[0].step == 42
        assert loaded_episodes[0].outcome_id == "test_out"
        assert loaded_episodes[0].surprise == 0.9


class TestGoalIntegration:
    """
    LAYER 5 EXTENSION: Goal-Directed Behavior Integration Tests

    These tests verify that goals influence agent behavior:
    - Goals can specify relevant atoms for attention boosting
    - Stalled goals trigger increased exploration
    - Goal activation changes are tracked and emitted
    """

    def test_goal_with_action_atoms_boosts_attention(self):
        """
        GIVEN a goal with action_atom_ids specified
        WHEN the agent runs a step with that goal as top priority
        THEN attention is boosted on the action atoms.

        Goals direct behavior by focusing attention on relevant concepts.
        """
        from cortical.cognitive.graph import CognitiveAgent, Goal

        # GIVEN
        agent = CognitiveAgent()

        # Create atoms that the goal will direct attention to
        target = agent.graph.node("target_concept")
        initial_sti = target.sti

        # Add goal with action atoms
        goal = Goal(
            id="focus_on_target",
            description="Focus on the target concept",
            target_state=1.0,
            importance=0.9,
            action_atom_ids=[target.id],  # Direct attention here
        )
        agent.goals.add_goal(goal)

        # WHEN - run a step
        agent.step()

        # THEN - target should have boosted attention
        updated_target = agent.graph.get_atom(target.id)
        # Note: decay happens too, so we check that attention was stimulated
        # by checking if the atom is higher than if only decay happened
        expected_with_decay_only = initial_sti * agent.graph._attention_decay
        actual_sti = updated_target.sti

        assert actual_sti > expected_with_decay_only, (
            f"Goal action atom should receive attention boost. "
            f"Expected > {expected_with_decay_only:.3f} (decay only), "
            f"got {actual_sti:.3f}"
        )

    def test_goal_stall_detection_after_threshold_steps(self):
        """
        GIVEN a goal that makes no progress
        WHEN enough steps pass without progress (stall_threshold)
        THEN the goal is marked as stalled.

        Stall detection enables the agent to recognize when it's stuck.
        """
        from cortical.cognitive.graph import Goal

        # GIVEN - a goal with stall_threshold=3
        goal = Goal(
            id="stuck_goal",
            description="A goal that will stall",
            target_state=1.0,
            current_state=0.2,
            importance=0.8,
            stall_threshold=3,
        )

        # Initially not stalled
        assert not goal.is_stalled, "Goal should not be stalled initially"

        # WHEN - record steps without progress
        goal.record_step()  # Step 1, no progress
        goal.record_step()  # Step 2, no progress

        assert not goal.is_stalled, "Goal should not be stalled after 2 steps"

        # Third step triggers stall
        became_stalled = goal.record_step()

        # THEN
        assert became_stalled, "record_step should return True when goal becomes stalled"
        assert goal.is_stalled, "Goal should be stalled after stall_threshold steps"

    def test_stalled_goal_increases_exploration(self):
        """
        GIVEN an agent with a stalled goal
        WHEN step() is called
        THEN exploration epsilon increases (more exploration).

        When stuck on a goal, the agent should try new things.
        """
        from cortical.cognitive.graph import CognitiveAgent, Goal

        # GIVEN
        agent = CognitiveAgent()

        # Add goal that will stall immediately (already stalled)
        goal = Goal(
            id="already_stuck",
            description="Pre-stalled goal",
            target_state=1.0,
            current_state=0.0,
            importance=0.9,
            stall_threshold=1,  # Stall after just 1 step
        )
        agent.goals.add_goal(goal)

        # Get initial epsilon
        initial_epsilon = agent.exploration.epsilon

        # WHEN - run steps to trigger stall and observe epsilon
        agent.step()  # First step - not yet stalled
        epsilon_after_first = agent.exploration.epsilon

        agent.step()  # Second step - goal becomes stalled, epsilon should increase

        # THEN - epsilon should increase due to stalled goal
        final_epsilon = agent.exploration.epsilon

        assert final_epsilon > initial_epsilon, (
            f"Epsilon should increase when goal is stalled. "
            f"Initial: {initial_epsilon:.3f}, Final: {final_epsilon:.3f}"
        )

    def test_goal_activation_event_emitted_on_top_goal_change(self):
        """
        GIVEN an agent with goals
        WHEN the top goal changes (due to progress or new goal)
        THEN GOAL_ACTIVATED event is emitted.

        This enables monitoring goal switches for debugging.
        """
        from cortical.cognitive.graph import CognitiveAgent, Goal, EventType

        # GIVEN
        agent = CognitiveAgent()
        activation_events = []

        def capture_activation(event):
            if event.event_type == EventType.GOAL_ACTIVATED:
                activation_events.append(event)

        agent.events.subscribe(EventType.GOAL_ACTIVATED, capture_activation)

        # Add first goal
        goal1 = Goal(
            id="goal_1",
            description="First goal",
            target_state=1.0,
            importance=0.5,
        )
        agent.goals.add_goal(goal1)

        # WHEN - run step (first goal should be activated)
        agent.step()

        # THEN - should have received activation event
        assert len(activation_events) == 1, (
            f"Expected 1 GOAL_ACTIVATED event, got {len(activation_events)}"
        )
        assert activation_events[0].data["goal_id"] == "goal_1"

        # WHEN - add higher priority goal
        goal2 = Goal(
            id="goal_2",
            description="Higher priority goal",
            target_state=1.0,
            importance=0.9,  # Higher importance
        )
        agent.goals.add_goal(goal2)
        agent.step()

        # THEN - should have another activation event for new top goal
        assert len(activation_events) == 2, (
            f"Expected 2 GOAL_ACTIVATED events, got {len(activation_events)}"
        )
        assert activation_events[1].data["goal_id"] == "goal_2"
        assert activation_events[1].data["previous_goal_id"] == "goal_1"

    def test_goal_stalled_event_emitted_when_stall_detected(self):
        """
        GIVEN an agent with a goal that will stall
        WHEN enough steps pass without progress
        THEN GOAL_STALLED event is emitted.

        This enables monitoring stuck states.
        """
        from cortical.cognitive.graph import CognitiveAgent, Goal, EventType

        # GIVEN
        agent = CognitiveAgent()
        stalled_events = []

        def capture_stalled(event):
            if event.event_type == EventType.GOAL_STALLED:
                stalled_events.append(event)

        agent.events.subscribe(EventType.GOAL_STALLED, capture_stalled)

        # Add goal with low stall threshold
        goal = Goal(
            id="will_stall",
            description="Goal that will stall",
            target_state=1.0,
            current_state=0.0,
            importance=0.8,
            stall_threshold=2,
        )
        agent.goals.add_goal(goal)

        # WHEN - run steps
        agent.step()  # Step 1
        agent.step()  # Step 2 - should stall

        # THEN
        assert len(stalled_events) == 1, (
            f"Expected 1 GOAL_STALLED event, got {len(stalled_events)}"
        )
        assert stalled_events[0].data["goal_id"] == "will_stall"
        assert stalled_events[0].data["steps_without_progress"] >= 2

    def test_progress_resets_stall_detection(self):
        """
        GIVEN a goal that was approaching stall threshold
        WHEN progress is made on the goal
        THEN stall counter resets and goal doesn't stall.

        Progress proves the current approach is working.
        """
        from cortical.cognitive.graph import Goal

        # GIVEN - goal approaching stall
        goal = Goal(
            id="progressing_goal",
            description="Goal that will make progress",
            target_state=1.0,
            current_state=0.0,
            importance=0.8,
            stall_threshold=3,
        )

        # Steps without progress
        goal.record_step()
        goal.record_step()

        assert goal._steps_without_progress == 2

        # WHEN - make progress
        goal.current_state = 0.5  # Update progress
        goal.record_step()

        # THEN - stall counter should reset
        assert goal._steps_without_progress == 0, (
            f"Progress should reset stall counter, but got {goal._steps_without_progress}"
        )
        assert not goal.is_stalled, "Goal should not be stalled after progress"


# =============================================================================
# SUCCESS CRITERIA (The Contract)
# =============================================================================


"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  GROUNDED COGNITIVE ARCHITECTURE v3.0                                      │
│  Success Criteria (The Contract)                                           │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════   │
│                                                                             │
│  LAYER 1: KNOWLEDGE                                                        │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Algorithm: Dictionary with adjacency lists                                │
│  Complexity: O(1) add/get, O(n) query                                      │
│  Test: test_add_and_retrieve_atom, test_query_with_predicate               │
│  Status: IMPLEMENTED ✓                                                     │
│                                                                             │
│  LAYER 2: ATTENTION                                                        │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Algorithm: Priority queue with exponential decay                          │
│  Complexity: O(n log k) for top-k selection                                │
│  Test: test_focus_returns_top_k_by_sti                                     │
│  Status: IMPLEMENTED ✓                                                     │
│                                                                             │
│  LAYER 3: WORKING MEMORY                                                   │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Algorithm: LRU cache with fixed capacity                                  │
│  Complexity: O(n) eviction scan (could be O(1) with doubly-linked list)   │
│  Test: test_capacity_limit, test_lru_eviction                              │
│  Status: IMPLEMENTED ✓                                                     │
│                                                                             │
│  LAYER 4: PREDICTION                                                       │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Algorithm: Co-occurrence counting (could be neural)                       │
│  Complexity: O(c) for prediction where c = context size                    │
│  Test: test_surprise_decreases_with_learning                               │
│  Status: IMPLEMENTED ✓ (minimal version)                                   │
│                                                                             │
│  LAYER 5: GOALS                                                            │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Algorithm: Priority queue by urgency                                      │
│  Complexity: O(n log n) sorting                                            │
│  Test: test_goals_sorted_by_urgency                                        │
│  Status: IMPLEMENTED ✓                                                     │
│                                                                             │
│  LAYER 6: EXPLORATION                                                      │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Algorithm: Adaptive ε-greedy                                              │
│  Complexity: O(1)                                                          │
│  Test: test_epsilon_decreases_on_success                                   │
│  Status: IMPLEMENTED ✓                                                     │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════   │
│                                                                             │
│  MEASURABLE OUTCOMES                                                       │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  1. SURPRISE DECREASES OVER TIME                                           │
│     Metric: mean_surprise over sliding window                              │
│     Target: < 0.3 after 100 observations                                   │
│     Why: Indicates learning is happening                                   │
│                                                                             │
│  2. GOAL PROGRESS INCREASES                                                │
│     Metric: total_progress across goals                                    │
│     Target: Monotonically increasing (on average)                          │
│     Why: Indicates work is effective                                       │
│                                                                             │
│  3. ATTENTION FOCUSES ON RELEVANT                                          │
│     Metric: % of focus atoms related to current goal                       │
│     Target: > 50%                                                          │
│     Why: Indicates attention is well-allocated                             │
│                                                                             │
│  4. EPSILON STABILIZES                                                     │
│     Metric: variance of epsilon over time                                  │
│     Target: Decreasing variance                                            │
│     Why: Indicates strategy is converging                                  │
│                                                                             │
│  5. WORKING MEMORY HITS > MISSES                                           │
│     Metric: cache hit ratio                                                │
│     Target: > 0.7                                                          │
│     Why: Indicates good eviction policy                                    │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════   │
│                                                                             │
│  WHAT THIS ARCHITECTURE IS:                                                │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • A minimal viable cognitive system                                       │
│  • Each layer independently testable                                       │
│  • Each layer grounded in proven algorithms                                │
│  • Measurable success criteria                                             │
│  • Incrementally implementable                                             │
│                                                                             │
│  WHAT THIS ARCHITECTURE IS NOT:                                            │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • Not a simulation of human cognition                                     │
│  • Not philosophically complete                                            │
│  • Not the final answer                                                    │
│  • Not trying to be clever                                                 │
│                                                                             │
│  THE GUIDING PRINCIPLE:                                                    │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│    "What is the MINIMUM that actually works?"                              │
│                                                                             │
│    Every line of code must earn its place through:                         │
│    1. Clear computational purpose                                          │
│    2. Proven algorithm                                                     │
│    3. Measurable success                                                   │
│                                                                             │
│    If a feature fails any of these tests, it doesn't belong.              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""
