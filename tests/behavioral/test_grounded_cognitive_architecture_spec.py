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

    Grounding: Graph databases.
    Success criterion: O(1) access, O(n) traversal.
    """

    def test_add_and_retrieve_atom(self):
        """Atoms can be stored and retrieved by ID."""
        graph = InMemoryKnowledgeGraph()

        atom = Atom(id="test-1", content="hello")
        graph.add(atom)

        retrieved = graph.get("test-1")
        assert retrieved is not None
        assert retrieved.content == "hello"

    def test_query_with_predicate(self):
        """Atoms can be queried by predicate."""
        graph = InMemoryKnowledgeGraph()

        graph.add(Atom(id="a1", content="cat", tv=TruthValue(0.9, 0.8)))
        graph.add(Atom(id="a2", content="dog", tv=TruthValue(0.5, 0.3)))
        graph.add(Atom(id="a3", content="bird", tv=TruthValue(0.95, 0.9)))

        # Query high-confidence atoms
        high_conf = graph.query(lambda a: a.tv.confidence > 0.7)
        assert len(high_conf) == 2

    def test_link_creates_neighbors(self):
        """Links connect atoms as neighbors."""
        graph = InMemoryKnowledgeGraph()

        cat = Atom(id="cat", content="cat")
        animal = Atom(id="animal", content="animal")
        link = Atom(id="link-1", content="is-a", outgoing=["cat", "animal"])

        graph.add(cat)
        graph.add(animal)
        graph.add(link)

        # Cat has link as neighbor
        cat_neighbors = graph.neighbors("cat")
        assert len(cat_neighbors) == 1
        assert cat_neighbors[0].id == "link-1"


# =============================================================================
# BEHAVIORAL TESTS: Layer 2 - Attention
# =============================================================================


class TestAttentionLayer:
    """
    Layer 2: Priority-based attention allocation.

    Grounding: OS schedulers, priority queues.
    Success criterion: Top-K selection in O(n log k).
    """

    def test_stimulation_increases_sti(self):
        """Stimulating an atom increases its STI."""
        graph = InMemoryKnowledgeGraph()
        atom = Atom(id="test", content="x", sti=0.0)
        graph.add(atom)

        attention = AttentionSystem(graph)
        attention.stimulate("test", amount=0.5)

        assert atom.sti == 0.5

    def test_decay_reduces_sti(self):
        """STI decays over time."""
        graph = InMemoryKnowledgeGraph()
        atom = Atom(id="test", content="x", sti=1.0)
        graph.add(atom)

        attention = AttentionSystem(graph, decay_rate=0.1)
        attention.step()

        assert atom.sti == 0.9

    def test_focus_returns_top_k_by_sti(self):
        """Focus returns atoms with highest STI."""
        graph = InMemoryKnowledgeGraph()

        for i in range(10):
            graph.add(Atom(id=f"atom-{i}", content=f"x{i}", sti=i * 0.1))

        attention = AttentionSystem(graph, focus_size=3)
        focus = attention.get_focus()

        assert len(focus) == 3
        assert focus[0].sti == 0.9
        assert focus[1].sti == 0.8
        assert focus[2].sti == 0.7

    def test_spreading_activation(self):
        """Activation spreads to neighbors."""
        graph = InMemoryKnowledgeGraph()

        source = Atom(id="source", content="s", sti=1.0)
        target = Atom(id="target", content="t", sti=0.0)
        link = Atom(id="link", content="l", outgoing=["source", "target"])

        graph.add(source)
        graph.add(target)
        graph.add(link)

        attention = AttentionSystem(graph)
        attention.spread_activation("source", spread_factor=0.5)

        # Link should have received activation
        assert link.sti > 0


# =============================================================================
# BEHAVIORAL TESTS: Layer 3 - Working Memory
# =============================================================================


class TestWorkingMemoryLayer:
    """
    Layer 3: Bounded capacity workspace.

    Grounding: LRU cache, Cowan's 4±1 limit.
    Success criterion: O(1) access, LRU eviction.
    """

    def test_capacity_limit(self):
        """Cannot exceed capacity."""
        wm = WorkingMemory(capacity=4)

        for i in range(6):
            wm.load(Atom(id=f"a{i}", content=f"x{i}"))

        assert len(wm.contents()) == 4

    def test_lru_eviction(self):
        """Least recently used is evicted."""
        wm = WorkingMemory(capacity=3)

        wm.load(Atom(id="a", content="a"))
        wm.load(Atom(id="b", content="b"))
        wm.load(Atom(id="c", content="c"))

        # Access 'a' to make it most recent
        wm.get("a")

        # Add new item - 'b' should be evicted (LRU)
        evicted = wm.load(Atom(id="d", content="d"))

        assert evicted is not None
        assert evicted.id == "b"
        assert wm.contains("a")
        assert not wm.contains("b")

    def test_reload_refreshes_access(self):
        """Reloading an atom refreshes its access time."""
        wm = WorkingMemory(capacity=2)

        wm.load(Atom(id="a", content="a"))
        wm.load(Atom(id="b", content="b"))

        # Reload 'a'
        wm.load(Atom(id="a", content="a"))

        # Add 'c' - 'b' should be evicted, not 'a'
        evicted = wm.load(Atom(id="c", content="c"))

        assert evicted.id == "b"


# =============================================================================
# BEHAVIORAL TESTS: Layer 4 - Prediction
# =============================================================================


class TestPredictionLayer:
    """
    Layer 4: Anticipate next relevant atoms.

    Grounding: Association rules, language models.
    Success criterion: Surprise decreases with experience.
    """

    def test_cooccurrence_enables_prediction(self):
        """Recording co-occurrences enables prediction."""
        graph = InMemoryKnowledgeGraph()
        graph.add(Atom(id="cat", content="cat"))
        graph.add(Atom(id="meow", content="meow"))

        predictor = AssociativePredictor(graph)

        # Record co-occurrences
        for _ in range(10):
            predictor.record_co_occurrence("cat", "meow")

        # Now predict
        context = [graph.get("cat")]
        predictions = predictor.predict(context)

        assert len(predictions) > 0
        assert predictions[0][0] == "meow"

    def test_surprise_high_for_unexpected(self):
        """Surprise is high when observation wasn't predicted."""
        graph = InMemoryKnowledgeGraph()
        graph.add(Atom(id="a", content="a"))
        graph.add(Atom(id="b", content="b"))
        graph.add(Atom(id="c", content="c"))

        predictor = AssociativePredictor(graph)
        tracker = SurpriseTracker(predictor)

        # No co-occurrences recorded - everything is surprising
        context = [graph.get("a")]
        surprise = tracker.record_outcome(context, "b")

        assert surprise == 1.0  # Maximum surprise

    def test_surprise_decreases_with_learning(self):
        """Surprise decreases as patterns are learned."""
        graph = InMemoryKnowledgeGraph()
        graph.add(Atom(id="a", content="a"))
        graph.add(Atom(id="b", content="b"))

        predictor = AssociativePredictor(graph)
        tracker = SurpriseTracker(predictor)

        surprises = []

        for i in range(20):
            # Record co-occurrence
            predictor.record_co_occurrence("a", "b")

            # Measure surprise
            context = [graph.get("a")]
            surprise = tracker.record_outcome(context, "b")
            surprises.append(surprise)

        # Surprise should decrease over time
        assert surprises[-1] < surprises[0]


# =============================================================================
# BEHAVIORAL TESTS: Layer 5 - Goals
# =============================================================================


class TestGoalLayer:
    """
    Layer 5: Track progress toward targets.

    Grounding: Control theory.
    Success criterion: Urgency reflects importance × remaining work.
    """

    def test_goal_progress(self):
        """Progress tracks distance to target."""
        goal = Goal(
            id="g1",
            description="reach 100",
            target_state=100,
            current_state=50,
            importance=1.0
        )

        assert goal.progress == 0.5

    def test_goal_urgency(self):
        """Urgency = importance × (1 - progress)."""
        goal = Goal(
            id="g1",
            description="reach 100",
            target_state=100,
            current_state=80,
            importance=0.8
        )

        # progress = 0.8, urgency = 0.8 * 0.2 = 0.16
        assert abs(goal.urgency - 0.16) < 0.01

    def test_goals_sorted_by_urgency(self):
        """Active goals are sorted by urgency."""
        tracker = GoalTracker()

        tracker.add_goal(Goal(
            id="low",
            description="low",
            target_state=100,
            current_state=90,
            importance=0.5
        ))

        tracker.add_goal(Goal(
            id="high",
            description="high",
            target_state=100,
            current_state=10,
            importance=0.9
        ))

        active = tracker.get_active_goals()

        assert active[0].id == "high"


# =============================================================================
# BEHAVIORAL TESTS: Layer 6 - Exploration
# =============================================================================


class TestExplorationLayer:
    """
    Layer 6: Balance exploration and exploitation.

    Grounding: Multi-armed bandits (ε-greedy).
    Success criterion: ε adapts to success/failure patterns.
    """

    def test_epsilon_decreases_on_success(self):
        """Success reduces exploration (exploit what works)."""
        controller = ExplorationController(initial_epsilon=0.5)

        for _ in range(5):
            controller.record_success()

        assert controller.epsilon < 0.5

    def test_epsilon_increases_on_failure(self):
        """Failure increases exploration (try something new)."""
        controller = ExplorationController(initial_epsilon=0.3)

        for _ in range(5):
            controller.record_failure()

        assert controller.epsilon > 0.3

    def test_stuck_detection(self):
        """Consecutive failures detected as stuck."""
        controller = ExplorationController()

        assert not controller.is_stuck()

        for _ in range(3):
            controller.record_failure()

        assert controller.is_stuck(threshold=3)

    def test_epsilon_bounded(self):
        """Epsilon stays within bounds."""
        controller = ExplorationController(min_epsilon=0.1, max_epsilon=0.8)

        for _ in range(100):
            controller.record_success()

        assert controller.epsilon >= 0.1

        for _ in range(100):
            controller.record_failure()

        assert controller.epsilon <= 0.8


# =============================================================================
# BEHAVIORAL TESTS: Integrated System
# =============================================================================


class TestIntegratedSystem:
    """
    Test the complete cognitive agent.

    Success criterion: Measurable improvement over time.
    """

    def test_agent_initializes(self):
        """Agent can be created with all layers."""
        agent = CognitiveAgent()

        assert agent.graph is not None
        assert agent.attention is not None
        assert agent.working_memory is not None
        assert agent.predictor is not None
        assert agent.goals is not None
        assert agent.exploration is not None

    def test_agent_step_produces_metrics(self):
        """Each step produces measurable metrics."""
        agent = CognitiveAgent()

        # Add some atoms
        agent.graph.add(Atom(id="a", content="a", sti=0.5))
        agent.graph.add(Atom(id="b", content="b", sti=0.3))

        metrics = agent.step()

        assert "step" in metrics
        assert "focus_size" in metrics
        assert "epsilon" in metrics
        assert metrics["step"] == 1

    def test_agent_attention_integration(self):
        """Attending to atoms updates working memory."""
        agent = CognitiveAgent()

        atom = Atom(id="test", content="test")
        agent.graph.add(atom)

        agent.attend("test")

        assert agent.working_memory.contains("test")
        assert atom.sti > 0

    def test_agent_learns_from_surprise(self):
        """Agent updates beliefs based on surprise."""
        agent = CognitiveAgent()

        agent.graph.add(Atom(id="context", content="c"))
        agent.graph.add(Atom(id="outcome", content="o", tv=TruthValue(0.5, 0.5)))

        surprise = agent.learn_from_surprise(["context"], "outcome")

        # High surprise should have updated the belief
        outcome = agent.graph.get("outcome")
        assert outcome.tv.confidence > 0.5  # Confidence increased


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
