"""
Cognitive Graph: Bio-Inspired Hypergraph for Knowledge Representation.

This module implements a hypergraph where links are first-class atoms,
enabling meta-reasoning (reasoning about relationships themselves).

Key Insight:
    In a regular graph: Node --edge--> Node
    In a hypergraph:    Atom --link--> Atom (where links ARE atoms)

    This allows statements ABOUT statements:
    - "John believes that cats are animals"
    - "Evidence A supports conclusion B"
    - "If X then Y" where X and Y are themselves relationships

Built on First Principles:
    1. Atoms are the universal unit (nodes and links)
    2. Truth is probabilistic (strength, confidence)
    3. Attention is finite and dynamic
    4. Dependencies are injected (IoC)
"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Union,
    runtime_checkable,
)

from cortical.common import Container, ContainerModule, Lifecycle


# =============================================================================
# Core Types
# =============================================================================


class AtomType(Enum):
    """
    Types of atoms in the cognitive graph.

    Nodes represent entities/concepts.
    Links represent relationships (and can point to other links).
    """

    # Node types
    CONCEPT = auto()      # General concept
    PERSON = auto()       # A person/agent
    PREDICATE = auto()    # A relation type
    VARIABLE = auto()     # For pattern matching
    NUMBER = auto()       # Numeric value
    WORD = auto()         # Lexical item

    # Link types (first-order: typically connect nodes)
    INHERITANCE = auto()  # IS-A relationship
    SIMILARITY = auto()   # Bidirectional similarity
    EVALUATION = auto()   # Predicate application
    MEMBER = auto()       # Set membership
    LIST = auto()         # Ordered collection

    # Link types (higher-order: can connect to links)
    BELIEVES = auto()     # Agent believes a statement
    DOUBTS = auto()       # Agent doubts a statement
    IMPLIES = auto()      # Logical implication
    EVIDENCE_FOR = auto() # Evidential support
    CONTEXT = auto()      # Contextual truth
    STRONGER_THAN = auto()  # Comparison between links


@dataclass
class TruthValue:
    """
    Probabilistic truth value.

    Unlike boolean logic:
    - strength: P(statement is true) in [0, 1]
    - confidence: How much evidence supports this estimate in [0, 1]

    Examples:
        TruthValue(0.99, 0.95) - Very likely true, high confidence
        TruthValue(0.5, 0.1)   - Uncertain, low evidence
        TruthValue(0.01, 0.99) - Almost certainly false, high confidence

    Grounding: Bayesian probability theory.
    Algorithm: Beta distribution (strength = mode, confidence = concentration).
    """

    strength: float = 1.0
    confidence: float = 0.0

    def __post_init__(self):
        """Clamp values to valid range."""
        self.strength = max(0.0, min(1.0, self.strength))
        self.confidence = max(0.0, min(1.0, self.confidence))

    def merge(self, other: TruthValue) -> TruthValue:
        """
        Combine evidence from two sources.

        Uses weighted average by confidence.
        Combined confidence increases (more evidence = more confident).
        """
        total_conf = self.confidence + other.confidence
        if total_conf == 0:
            return TruthValue(0.5, 0.0)

        # Weighted average of strength
        new_strength = (
            self.strength * self.confidence +
            other.strength * other.confidence
        ) / total_conf

        # Confidence increases with more evidence (diminishing returns)
        new_confidence = 1.0 - (1.0 - self.confidence) * (1.0 - other.confidence)

        return TruthValue(new_strength, new_confidence)

    def update(self, observation: bool, learning_rate: float = 0.1) -> 'TruthValue':
        """
        Bayesian update from observation.

        Grounding: Conjugate prior update for Beta distribution.

        Args:
            observation: True if observation confirmed, False if refuted
            learning_rate: How fast to update (default 0.1)

        Returns:
            New TruthValue with updated strength and confidence
        """
        obs_value = 1.0 if observation else 0.0

        # Weighted update toward observation
        new_strength = self.strength + learning_rate * (obs_value - self.strength)

        # Confidence increases with evidence (diminishing returns)
        new_confidence = self.confidence + learning_rate * (1.0 - self.confidence)

        return TruthValue(new_strength, new_confidence)

    def surprise(self, observation: bool) -> float:
        """
        Prediction error magnitude (surprisal).

        Grounding: Information theory - surprisal = -log(P(observation))
        Simplified: |predicted - observed|

        Args:
            observation: What actually happened

        Returns:
            Surprise level in [0, 1]. 0 = expected, 1 = completely unexpected
        """
        predicted = self.strength
        observed = 1.0 if observation else 0.0
        return abs(predicted - observed)

    def __repr__(self) -> str:
        return f"TV({self.strength:.2f}, {self.confidence:.2f})"


@dataclass
class Atom:
    """
    The universal unit of the cognitive graph.

    Both nodes and links are atoms. Links are distinguished by having
    a non-empty outgoing list.

    Attributes:
        id: Unique identifier
        atom_type: The type of this atom
        name: Human-readable name (for nodes)
        outgoing: List of atom IDs this atom points to (for links)
        tv: Probabilistic truth value
        sti: Short-term importance (attention) [0, 1]
        lti: Long-term importance (persistent significance) [0, 1]
        created_at: Timestamp of creation
        accessed_at: Last access timestamp (for LRU tracking)

    Grounding: Standard graph theory + probabilistic databases.
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    atom_type: AtomType = AtomType.CONCEPT
    name: str = ""
    outgoing: List[str] = field(default_factory=list)
    tv: TruthValue = field(default_factory=TruthValue)
    sti: float = 0.0  # Short-term importance
    lti: float = 0.0  # Long-term importance
    created_at: float = 0.0  # Timestamp of creation
    accessed_at: float = 0.0  # Last access timestamp

    def is_link(self) -> bool:
        """True if this atom is a link (has outgoing connections)."""
        return len(self.outgoing) > 0

    def is_node(self) -> bool:
        """True if this atom is a node (no outgoing connections)."""
        return len(self.outgoing) == 0

    def __repr__(self) -> str:
        if self.name:
            return f"Atom({self.name}, {self.atom_type.name})"
        return f"Atom({self.atom_type.name}[{self.id[:4]}], out={len(self.outgoing)})"


# =============================================================================
# Storage Backend (Protocol for DI)
# =============================================================================


@runtime_checkable
class StorageBackend(Protocol):
    """
    Protocol for atom storage.

    Implementations can be in-memory, file-based, or database-backed.
    The cognitive graph uses this via dependency injection.
    """

    def save(self, atom: Atom) -> None:
        """Persist an atom."""
        ...

    def load(self, atom_id: str) -> Optional[Atom]:
        """Load an atom by ID."""
        ...

    def delete(self, atom_id: str) -> bool:
        """Delete an atom by ID."""
        ...

    def find_by_name(self, name: str) -> Optional[Atom]:
        """Find a node by name."""
        ...

    def find_by_type(self, atom_type: AtomType) -> List[Atom]:
        """Find all atoms of a given type."""
        ...

    def all_atoms(self) -> List[Atom]:
        """Get all atoms."""
        ...


class InMemoryStorage:
    """
    In-memory storage backend for testing and lightweight use.

    Thread-safe for single-process use.
    """

    def __init__(self):
        self._atoms: Dict[str, Atom] = {}
        self._by_name: Dict[str, str] = {}  # name -> id
        self._incoming: Dict[str, Set[str]] = {}  # atom_id -> set of link_ids

    def save(self, atom: Atom) -> None:
        """Persist an atom."""
        self._atoms[atom.id] = atom

        if atom.name:
            self._by_name[atom.name] = atom.id

        # Update incoming index for links
        if atom.is_link():
            for target_id in atom.outgoing:
                if target_id not in self._incoming:
                    self._incoming[target_id] = set()
                self._incoming[target_id].add(atom.id)

    def load(self, atom_id: str) -> Optional[Atom]:
        """Load an atom by ID."""
        return self._atoms.get(atom_id)

    def delete(self, atom_id: str) -> bool:
        """Delete an atom by ID."""
        if atom_id not in self._atoms:
            return False

        atom = self._atoms[atom_id]

        # Remove from name index
        if atom.name and self._by_name.get(atom.name) == atom_id:
            del self._by_name[atom.name]

        # Remove from incoming index
        if atom.is_link():
            for target_id in atom.outgoing:
                if target_id in self._incoming:
                    self._incoming[target_id].discard(atom_id)

        del self._atoms[atom_id]
        return True

    def find_by_name(self, name: str) -> Optional[Atom]:
        """Find a node by name."""
        atom_id = self._by_name.get(name)
        if atom_id:
            return self._atoms.get(atom_id)
        return None

    def find_by_type(self, atom_type: AtomType) -> List[Atom]:
        """Find all atoms of a given type."""
        return [a for a in self._atoms.values() if a.atom_type == atom_type]

    def all_atoms(self) -> List[Atom]:
        """Get all atoms."""
        return list(self._atoms.values())

    def get_incoming(self, atom_id: str) -> List[Atom]:
        """Get all links pointing to this atom."""
        link_ids = self._incoming.get(atom_id, set())
        return [self._atoms[lid] for lid in link_ids if lid in self._atoms]


# =============================================================================
# Cognitive Graph
# =============================================================================


class CognitiveGraph:
    """
    Bio-inspired hypergraph for knowledge representation.

    Key Properties:
        - Links are atoms (can be linked to)
        - Truth is probabilistic (strength, confidence)
        - Attention is finite (STI decays, spreads)
        - Content-addressed (same content = same atom)

    Usage:
        graph = CognitiveGraph()

        # Create nodes
        cat = graph.node("cat")
        animal = graph.node("animal")

        # Create links (relationships)
        link = graph.link(AtomType.INHERITANCE, [cat, animal])

        # Create meta-links (links about links)
        john = graph.node("john", atom_type=AtomType.PERSON)
        belief = graph.link(AtomType.BELIEVES, [john, link])
    """

    def __init__(self, storage: StorageBackend = None):
        """
        Initialize the cognitive graph.

        Args:
            storage: Storage backend (defaults to InMemoryStorage if None)
        """
        # Support both DI (storage injected) and direct use (storage=None)
        if storage is None:
            storage = InMemoryStorage()
        self._storage: StorageBackend = storage
        self._attention_decay = 0.9
        self._attention_spread_factor = 0.1

    # =========================================================================
    # Atom Creation
    # =========================================================================

    def node(
        self,
        name: str,
        atom_type: AtomType = AtomType.CONCEPT,
        tv: Optional[TruthValue] = None,
    ) -> Atom:
        """
        Create or retrieve a node atom.

        Nodes are content-addressed by name: requesting the same name
        returns the same atom.

        Args:
            name: The node's name
            atom_type: Type of node (default: CONCEPT)
            tv: Truth value (default: strength=1.0, confidence=0.0)

        Returns:
            The node atom (new or existing)
        """
        # Check for existing
        existing = self._storage.find_by_name(name)
        if existing is not None:
            return existing

        # Create new
        atom = Atom(
            atom_type=atom_type,
            name=name,
            tv=tv or TruthValue(),
        )
        self._storage.save(atom)
        return atom

    def link(
        self,
        link_type: AtomType,
        targets: List[Union[str, Atom]],
        tv: Optional[TruthValue] = None,
    ) -> Atom:
        """
        Create or retrieve a link atom.

        Links are content-addressed by type + outgoing set.
        THE KEY INSIGHT: targets can be nodes OR other links.

        Args:
            link_type: Type of link (e.g., INHERITANCE, BELIEVES)
            targets: List of atoms or atom IDs to link
            tv: Truth value for this relationship

        Returns:
            The link atom (new or existing with merged TV)
        """
        # Normalize targets to IDs
        target_ids = []
        for t in targets:
            if isinstance(t, Atom):
                target_ids.append(t.id)
            elif isinstance(t, str):
                # Could be a name or an ID
                existing = self._storage.find_by_name(t)
                if existing:
                    target_ids.append(existing.id)
                else:
                    # Assume it's an ID
                    target_ids.append(t)
            else:
                raise TypeError(f"Invalid target type: {type(t)}")

        # Check for existing identical link
        for atom in self._storage.find_by_type(link_type):
            if atom.outgoing == target_ids:
                # Merge truth values if new evidence provided
                if tv and tv.confidence > 0:
                    atom.tv = atom.tv.merge(tv)
                    self._storage.save(atom)
                return atom

        # Create new link
        atom = Atom(
            atom_type=link_type,
            outgoing=target_ids,
            tv=tv or TruthValue(),
        )
        self._storage.save(atom)
        return atom

    # =========================================================================
    # Atom Retrieval
    # =========================================================================

    def get_atom(self, atom_id: str) -> Optional[Atom]:
        """Get an atom by ID."""
        return self._storage.load(atom_id)

    def get_node(self, name: str) -> Optional[Atom]:
        """Get a node by name."""
        return self._storage.find_by_name(name)

    def get_incoming(self, atom_id: str) -> List[Atom]:
        """Get all links pointing to this atom."""
        if isinstance(self._storage, InMemoryStorage):
            return self._storage.get_incoming(atom_id)

        # Fallback: scan all atoms
        result = []
        for atom in self._storage.all_atoms():
            if atom.is_link() and atom_id in atom.outgoing:
                result.append(atom)
        return result

    def find_by_type(self, atom_type: AtomType) -> List[Atom]:
        """Find all atoms of a given type."""
        return self._storage.find_by_type(atom_type)

    # =========================================================================
    # Attention Dynamics
    # =========================================================================

    def stimulate(self, name_or_id: str, amount: float) -> None:
        """
        Increase an atom's short-term importance (STI).

        Args:
            name_or_id: Atom name or ID
            amount: Amount to add to STI
        """
        atom = self._storage.find_by_name(name_or_id)
        if atom is None:
            atom = self._storage.load(name_or_id)

        if atom is not None:
            atom.sti += amount
            self._storage.save(atom)

    def step(self) -> None:
        """
        Execute one cognitive step.

        - Spread attention through links (BEFORE decay)
        - Decay all STI values
        """
        atoms = self._storage.all_atoms()

        # Collect attention to spread BEFORE decay
        spread_amounts: Dict[str, float] = {}

        for atom in atoms:
            # Calculate spread from high-STI atoms BEFORE decay
            if atom.sti > 1.0:
                # Get connected atoms (through links)
                neighbors = set()
                if atom.is_link():
                    neighbors.update(atom.outgoing)
                # Also include atoms connected via incoming links
                for link in self.get_incoming(atom.id):
                    # Add the other atoms in the link
                    for target_id in link.outgoing:
                        if target_id != atom.id:
                            neighbors.add(target_id)

                if neighbors:
                    spread_per_neighbor = atom.sti * self._attention_spread_factor / len(neighbors)
                    for neighbor_id in neighbors:
                        spread_amounts[neighbor_id] = spread_amounts.get(neighbor_id, 0) + spread_per_neighbor

        # Now decay all atoms
        for atom in atoms:
            atom.sti *= self._attention_decay
            self._storage.save(atom)

        # Apply spread
        for atom_id, amount in spread_amounts.items():
            atom = self._storage.load(atom_id)
            if atom:
                atom.sti += amount
                self._storage.save(atom)

    def get_attention_focus(self, top_k: int = 10) -> List[Atom]:
        """Get the top-K atoms by STI (attention focus)."""
        atoms = self._storage.all_atoms()
        sorted_atoms = sorted(atoms, key=lambda a: a.sti, reverse=True)
        return [a for a in sorted_atoms[:top_k] if a.sti > 0]

    # =========================================================================
    # Cognitive Processing
    # =========================================================================

    def process_experience(
        self,
        concepts: List[str],
        relations: List[Tuple[str, str, str]],
    ) -> Dict[str, Any]:
        """
        Process an experience and update the graph.

        Args:
            concepts: List of concept names encountered
            relations: List of (source, relation_type, target) tuples

        Returns:
            Statistics about what was created/updated
        """
        nodes_created = 0
        links_created = 0

        # Create/get nodes for concepts
        for concept in concepts:
            existing = self._storage.find_by_name(concept)
            if existing is None:
                self.node(concept)
                nodes_created += 1
            else:
                # Stimulate existing concepts (attention from experience)
                existing.sti += 1.0
                self._storage.save(existing)

        # Create relations
        relation_type_map = {
            "is_a": AtomType.INHERITANCE,
            "has": AtomType.EVALUATION,
            "similar_to": AtomType.SIMILARITY,
        }

        for source, rel_type, target in relations:
            source_node = self.node(source)
            target_node = self.node(target)

            link_type = relation_type_map.get(rel_type, AtomType.EVALUATION)

            # Check if this exact link exists
            existing_link = None
            for link in self._storage.find_by_type(link_type):
                if link.outgoing == [source_node.id, target_node.id]:
                    existing_link = link
                    break

            if existing_link is None:
                self.link(link_type, [source_node, target_node], TruthValue(1.0, 0.5))
                links_created += 1
            else:
                # Increase confidence with repeated observation
                existing_link.tv = existing_link.tv.merge(TruthValue(1.0, 0.3))
                self._storage.save(existing_link)

        return {
            "nodes_created": nodes_created,
            "links_created": links_created,
        }

    def observe_patterns(self) -> Dict[str, Any]:
        """
        Observe patterns in the graph (self-reflection).

        Returns statistics about:
        - Frequent concepts (high LTI or connection count)
        - Consistent relations (high confidence links)
        """
        atoms = self._storage.all_atoms()

        # Count concept frequencies (by incoming links)
        concept_scores: Dict[str, float] = {}
        for atom in atoms:
            if atom.is_node() and atom.name:
                incoming_count = len(self.get_incoming(atom.id))
                concept_scores[atom.name] = incoming_count + atom.lti

        # Find consistent relations
        relation_counts: Dict[str, int] = {}
        for atom in atoms:
            if atom.is_link() and len(atom.outgoing) == 2:
                source = self._storage.load(atom.outgoing[0])
                target = self._storage.load(atom.outgoing[1])
                if source and target and source.name and target.name:
                    key = f"{source.name}_is_a_{target.name}"
                    relation_counts[key] = relation_counts.get(key, 0) + 1

                    # Count based on truth value confidence
                    if atom.tv.confidence > 0.5:
                        relation_counts[key] += int(atom.tv.confidence * 10)

        # Sort by frequency
        sorted_concepts = sorted(concept_scores.items(), key=lambda x: -x[1])
        frequent_concepts = [name for name, _ in sorted_concepts[:10]]

        return {
            "frequent_concepts": frequent_concepts,
            "consistent_relations": relation_counts,
        }


# =============================================================================
# Container Module for DI
# =============================================================================


class CognitiveGraphModule(ContainerModule):
    """
    DI module for cognitive graph components.

    Registers:
        - StorageBackend (default: InMemoryStorage)
        - CognitiveGraph

    Usage:
        container = Container()
        container.apply_module(CognitiveGraphModule())
        graph = container.resolve(CognitiveGraph)
    """

    def __init__(
        self,
        lifecycle: Lifecycle = Lifecycle.SINGLETON,
        storage_class: type = InMemoryStorage,
    ):
        """
        Initialize the module.

        Args:
            lifecycle: Lifecycle for the graph (SINGLETON or TRANSIENT)
            storage_class: Storage backend implementation
        """
        self.lifecycle = lifecycle
        self.storage_class = storage_class

    def register(self, container: Container) -> None:
        """Register cognitive graph components."""
        container.register(
            StorageBackend,
            self.storage_class,
            lifecycle=self.lifecycle,
        )
        # Register CognitiveGraph with explicit dependency on StorageBackend
        container.register(
            CognitiveGraph,
            CognitiveGraph,
            lifecycle=self.lifecycle,
            storage=StorageBackend,  # Explicit dependency injection
        )


# =============================================================================
# Cognitive Layers: Grounded Architecture v3.0
# =============================================================================
#
# These layers extend the hypergraph foundation with proven cognitive algorithms:
#   - Working Memory (LRU cache with capacity limits)
#   - Prediction (co-occurrence based, swappable for neural models)
#   - Goals (control theory with urgency tracking)
#   - Exploration (ε-greedy bandit algorithm)
#
# Design Principle: Each layer is independently testable and grounded in
# proven algorithms. No metaphors, just computer science.
# =============================================================================


@dataclass
class Goal:
    """
    A target state with progress tracking.

    Grounding: Control theory - goal = setpoint, progress = error signal.

    Attributes:
        id: Unique identifier
        description: Human-readable description
        target_state: What we're trying to achieve
        current_state: Where we are now
        importance: How much this matters [0, 1]
    """
    id: str
    description: str
    target_state: Any
    current_state: Any = None
    importance: float = 0.5

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
        """Priority for attention allocation = importance × (1 - progress)."""
        return self.importance * (1.0 - self.progress)

    def is_complete(self) -> bool:
        """Check if goal has been achieved."""
        return self.progress >= 0.99


class WorkingMemory:
    """
    Bounded capacity workspace with LRU eviction.

    Grounding: Cowan's 4±1 capacity limit, LRU cache.

    This is literally a cache with eviction policy.
    No metaphor. Just computer science.

    TODO: Consider O(1) eviction with doubly-linked list + hashmap
    TODO: Add memory consolidation to long-term storage
    """

    def __init__(self, capacity: int = 4):
        """
        Initialize working memory.

        Args:
            capacity: Maximum number of atoms to hold (default 4, based on Cowan's research)
        """
        self.capacity = capacity
        self._slots: List[Atom] = []
        self._access_order: List[str] = []  # For LRU tracking

    def load(self, atom: Atom) -> Optional[Atom]:
        """
        Load atom into working memory.

        Returns evicted atom if capacity exceeded, None otherwise.
        """
        # Already present? Move to front of access order (most recent)
        for i, slot in enumerate(self._slots):
            if slot.id == atom.id:
                self._access_order.remove(atom.id)
                self._access_order.append(atom.id)
                self._slots[i] = atom  # Update with latest version
                return None

        evicted = None

        # At capacity? Evict LRU (least recently used)
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
        """Check if atom is in working memory."""
        return any(s.id == atom_id for s in self._slots)

    def contents(self) -> List[Atom]:
        """Get all atoms in working memory."""
        return list(self._slots)

    def clear(self) -> None:
        """Clear working memory."""
        self._slots.clear()
        self._access_order.clear()

    def is_full(self) -> bool:
        """Check if working memory is at capacity."""
        return len(self._slots) >= self.capacity


class AssociativePredictor:
    """
    Simple co-occurrence based predictor.

    Grounding: Association rules, Hebbian learning.
    Algorithm: Count co-activations, normalize to probabilities.

    This is a minimal predictor that predicts based on what we tell it.
    The predictor learns from recorded co-occurrences only.

    TODO: Swap for neural model (e.g., transformer, RNN)
    TODO: Add context window for temporal patterns
    TODO: Add decay for old co-occurrences
    """

    def __init__(self, graph: 'CognitiveGraph'):
        """
        Initialize predictor.

        Args:
            graph: The cognitive graph to make predictions about
        """
        self.graph = graph
        self._co_occurrences: Dict[str, Dict[str, int]] = {}  # a -> b -> count

    def record_co_occurrence(self, atom_a_id: str, atom_b_id: str) -> None:
        """
        Record that these atoms were active together.

        This is how the predictor learns: you tell it what co-occurs.
        """
        if atom_a_id not in self._co_occurrences:
            self._co_occurrences[atom_a_id] = {}
        if atom_b_id not in self._co_occurrences[atom_a_id]:
            self._co_occurrences[atom_a_id][atom_b_id] = 0
        self._co_occurrences[atom_a_id][atom_b_id] += 1

    def predict(self, context: List[Atom]) -> List[Tuple[str, float]]:
        """
        Predict next relevant atoms based on context.

        Args:
            context: List of atoms currently active

        Returns:
            List of (atom_id, probability) pairs, sorted by probability
        """
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

    TODO: Add exponential moving average for smoother tracking
    TODO: Add per-context surprise tracking
    """

    def __init__(self, predictor: AssociativePredictor):
        """
        Initialize surprise tracker.

        Args:
            predictor: The predictor whose errors we track
        """
        self.predictor = predictor
        self._prediction_history: List[Tuple[List[str], str, float]] = []

    def record_outcome(
        self,
        context: List[Atom],
        actual_atom_id: str,
    ) -> float:
        """
        Record what actually happened and return surprise level.

        Args:
            context: The atoms that were active before the outcome
            actual_atom_id: What actually happened

        Returns:
            Surprise in [0, 1]. 0 = perfectly predicted, 1 = completely unexpected
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
        """
        Average surprise over recent predictions.

        Args:
            window: Number of recent predictions to consider

        Returns:
            Mean surprise level [0, 1]
        """
        if not self._prediction_history:
            return 0.5  # Uncertain baseline

        recent = self._prediction_history[-window:]
        return sum(s for _, _, s in recent) / len(recent)


class GoalTracker:
    """
    Tracks goals and their progress.

    Grounding: Control theory - error = target - current.
    Algorithm: Priority queue by urgency.

    No desires. No hierarchy. Just goals with progress.

    TODO: Add goal dependencies (blocked_by relationships)
    TODO: Add goal decomposition (subgoals)
    TODO: Integrate with GoT task system
    """

    def __init__(self):
        """Initialize goal tracker."""
        self._goals: Dict[str, Goal] = {}

    def add_goal(self, goal: Goal) -> None:
        """Add a goal to track."""
        self._goals[goal.id] = goal

    def remove_goal(self, goal_id: str) -> bool:
        """Remove a goal. Returns True if found."""
        if goal_id in self._goals:
            del self._goals[goal_id]
            return True
        return False

    def update_progress(self, goal_id: str, current_state: Any) -> None:
        """Update a goal's current state."""
        if goal_id in self._goals:
            self._goals[goal_id].current_state = current_state

    def get_active_goals(self) -> List[Goal]:
        """Get incomplete goals sorted by urgency (highest first)."""
        active = [g for g in self._goals.values() if not g.is_complete()]
        return sorted(active, key=lambda g: g.urgency, reverse=True)

    def get_top_goal(self) -> Optional[Goal]:
        """Get highest urgency goal."""
        active = self.get_active_goals()
        return active[0] if active else None

    def complete_count(self) -> int:
        """Count completed goals."""
        return sum(1 for g in self._goals.values() if g.is_complete())

    def total_progress(self) -> float:
        """Average progress across all goals."""
        if not self._goals:
            return 0.0
        return sum(g.progress for g in self._goals.values()) / len(self._goals)


class ExplorationController:
    """
    Balances exploration vs exploitation.

    Grounding: Multi-armed bandit algorithms (ε-greedy).
    Single parameter: ε (exploration rate).

    No affect states. No multiple emotions.
    Just one number that adapts based on success/failure.

    TODO: Implement UCB (Upper Confidence Bound) as alternative
    TODO: Add Thompson sampling option
    TODO: Add contextual bandits for state-dependent exploration
    """

    def __init__(
        self,
        initial_epsilon: float = 0.3,
        min_epsilon: float = 0.05,
        max_epsilon: float = 0.9,
        adaptation_rate: float = 0.1,
    ):
        """
        Initialize exploration controller.

        Args:
            initial_epsilon: Starting exploration rate [0, 1]
            min_epsilon: Minimum exploration rate
            max_epsilon: Maximum exploration rate
            adaptation_rate: How fast to adapt
        """
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

        Returns:
            True if should explore, False if should exploit
        """
        import random
        return random.random() < self.epsilon

    def record_success(self) -> None:
        """Current approach is working - reduce exploration (exploit more)."""
        self._consecutive_successes += 1
        self._consecutive_failures = 0

        # Decrease epsilon
        self.epsilon = max(
            self.min_epsilon,
            self.epsilon - self.adaptation_rate
        )

    def record_failure(self) -> None:
        """Current approach failed - increase exploration (try new things)."""
        self._consecutive_failures += 1
        self._consecutive_successes = 0

        # Increase epsilon
        self.epsilon = min(
            self.max_epsilon,
            self.epsilon + self.adaptation_rate
        )

    def is_stuck(self, threshold: int = 3) -> bool:
        """
        Are we in a failure loop?

        Args:
            threshold: Number of consecutive failures to consider "stuck"

        Returns:
            True if stuck in failure pattern
        """
        return self._consecutive_failures >= threshold


class CognitiveAgent:
    """
    The complete minimal cognitive agent integrating all six layers.

    Layers:
        1. Knowledge (CognitiveGraph) - Hypergraph with truth values
        2. Attention (via CognitiveGraph) - STI/LTI with decay
        3. Working Memory - LRU bounded buffer
        4. Prediction - Co-occurrence based
        5. Goals - Control theory with urgency
        6. Exploration - ε-greedy adaptation

    Each layer is independently testable.
    The integration is also testable.

    TODO: Add persistence/serialization for agent state
    TODO: Add event hooks for layer interactions
    TODO: Connect to GoT for task/entity integration
    TODO: Add episodic memory for experience replay
    """

    def __init__(
        self,
        graph: Optional[CognitiveGraph] = None,
        working_memory_size: int = 4,
        attention_focus_size: int = 7,
    ):
        """
        Initialize cognitive agent.

        Args:
            graph: Knowledge graph (creates new if None)
            working_memory_size: Capacity of working memory (default 4)
            attention_focus_size: Size of attention focus (default 7, Miller's 7±2)
        """
        self.graph = graph or CognitiveGraph()
        self._attention_focus_size = attention_focus_size
        self.working_memory = WorkingMemory(capacity=working_memory_size)
        self.predictor = AssociativePredictor(self.graph)
        self.surprise_tracker = SurpriseTracker(self.predictor)
        self.goals = GoalTracker()
        self.exploration = ExplorationController()
        self._step_count: int = 0

    def step(self) -> Dict[str, Any]:
        """
        Execute one cognitive step.

        Returns:
            Metrics about what happened in this step
        """
        self._step_count += 1

        # 1. Apply attention decay (delegates to graph)
        self.graph.step()

        # 2. Get current focus
        focus = self.graph.get_attention_focus(top_k=self._attention_focus_size)

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

    def attend(self, name_or_id: str, amount: float = 0.2) -> None:
        """
        Direct attention to an atom and load it into working memory.

        Args:
            name_or_id: Atom name or ID to attend to
            amount: Amount to increase STI (default 0.2)
        """
        self.graph.stimulate(name_or_id, amount)
        atom = self.graph.get_node(name_or_id)
        if atom is None:
            atom = self.graph.get_atom(name_or_id)
        if atom:
            evicted = self.working_memory.load(atom)
            # TODO: Could log evicted atoms for analysis

    def learn_from_surprise(self, context_ids: List[str], actual_id: str) -> float:
        """
        Update beliefs based on prediction error.

        Args:
            context_ids: IDs of atoms that were active before outcome
            actual_id: ID of atom that actually appeared

        Returns:
            Surprise level [0, 1]
        """
        context = []
        for aid in context_ids:
            atom = self.graph.get_node(aid)
            if atom is None:
                atom = self.graph.get_atom(aid)
            if atom:
                context.append(atom)

        surprise = self.surprise_tracker.record_outcome(context, actual_id)

        # High surprise = update beliefs more strongly
        actual_atom = self.graph.get_atom(actual_id)
        if actual_atom and surprise > 0.5:
            # Increase confidence in surprising observation
            actual_atom.tv = actual_atom.tv.update(True, learning_rate=surprise * 0.2)
            self.graph._storage.save(actual_atom)

        return surprise

    # =========================================================================
    # Persistence (JSON-based for security and git-friendliness)
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize agent state to a dictionary.

        Returns:
            Dictionary containing all agent state
        """
        # Serialize atoms
        atoms_data = []
        for atom in self.graph._storage.all_atoms():
            atoms_data.append({
                "id": atom.id,
                "atom_type": atom.atom_type.name,
                "name": atom.name,
                "outgoing": atom.outgoing,
                "tv": {"strength": atom.tv.strength, "confidence": atom.tv.confidence},
                "sti": atom.sti,
                "lti": atom.lti,
                "created_at": atom.created_at,
                "accessed_at": atom.accessed_at,
            })

        # Serialize goals
        goals_data = []
        for goal in self.goals._goals.values():
            goals_data.append({
                "id": goal.id,
                "description": goal.description,
                "target_state": goal.target_state,
                "current_state": goal.current_state,
                "importance": goal.importance,
            })

        # Serialize working memory
        wm_data = {
            "capacity": self.working_memory.capacity,
            "slot_ids": [a.id for a in self.working_memory._slots],
            "access_order": self.working_memory._access_order,
        }

        # Serialize predictor co-occurrences
        co_occurrences = self.predictor._co_occurrences

        # Serialize exploration state
        exploration_data = {
            "epsilon": self.exploration.epsilon,
            "min_epsilon": self.exploration.min_epsilon,
            "max_epsilon": self.exploration.max_epsilon,
            "adaptation_rate": self.exploration.adaptation_rate,
            "consecutive_failures": self.exploration._consecutive_failures,
            "consecutive_successes": self.exploration._consecutive_successes,
        }

        return {
            "version": "3.0",
            "step_count": self._step_count,
            "attention_focus_size": self._attention_focus_size,
            "atoms": atoms_data,
            "goals": goals_data,
            "working_memory": wm_data,
            "co_occurrences": co_occurrences,
            "exploration": exploration_data,
            "surprise_history": [
                {"context": ctx, "actual": actual, "surprise": s}
                for ctx, actual, s in self.surprise_tracker._prediction_history[-1000:]
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CognitiveAgent':
        """
        Deserialize agent from dictionary.

        Args:
            data: Dictionary from to_dict()

        Returns:
            Reconstructed CognitiveAgent
        """
        # Create agent with correct sizes
        wm_capacity = data.get("working_memory", {}).get("capacity", 4)
        focus_size = data.get("attention_focus_size", 7)
        agent = cls(
            working_memory_size=wm_capacity,
            attention_focus_size=focus_size,
        )

        # Restore step count
        agent._step_count = data.get("step_count", 0)

        # Restore atoms
        atom_lookup: Dict[str, Atom] = {}
        for atom_data in data.get("atoms", []):
            atom = Atom(
                id=atom_data["id"],
                atom_type=AtomType[atom_data["atom_type"]],
                name=atom_data.get("name", ""),
                outgoing=atom_data.get("outgoing", []),
                tv=TruthValue(
                    atom_data["tv"]["strength"],
                    atom_data["tv"]["confidence"],
                ),
                sti=atom_data.get("sti", 0.0),
                lti=atom_data.get("lti", 0.0),
                created_at=atom_data.get("created_at", 0.0),
                accessed_at=atom_data.get("accessed_at", 0.0),
            )
            agent.graph._storage.save(atom)
            atom_lookup[atom.id] = atom

        # Restore goals
        for goal_data in data.get("goals", []):
            goal = Goal(
                id=goal_data["id"],
                description=goal_data["description"],
                target_state=goal_data["target_state"],
                current_state=goal_data.get("current_state"),
                importance=goal_data.get("importance", 0.5),
            )
            agent.goals.add_goal(goal)

        # Restore working memory
        wm_data = data.get("working_memory", {})
        for atom_id in wm_data.get("slot_ids", []):
            if atom_id in atom_lookup:
                agent.working_memory.load(atom_lookup[atom_id])

        # Restore co-occurrences
        agent.predictor._co_occurrences = data.get("co_occurrences", {})

        # Restore exploration state
        exp_data = data.get("exploration", {})
        agent.exploration.epsilon = exp_data.get("epsilon", 0.3)
        agent.exploration.min_epsilon = exp_data.get("min_epsilon", 0.05)
        agent.exploration.max_epsilon = exp_data.get("max_epsilon", 0.9)
        agent.exploration.adaptation_rate = exp_data.get("adaptation_rate", 0.1)
        agent.exploration._consecutive_failures = exp_data.get("consecutive_failures", 0)
        agent.exploration._consecutive_successes = exp_data.get("consecutive_successes", 0)

        # Restore surprise history
        for entry in data.get("surprise_history", []):
            agent.surprise_tracker._prediction_history.append((
                entry["context"],
                entry["actual"],
                entry["surprise"],
            ))

        return agent

    def save(self, path: Union[str, Path]) -> None:
        """
        Save agent state to JSON file.

        Args:
            path: File path (will add .json if not present)
        """
        path = Path(path)
        if path.suffix != ".json":
            path = path.with_suffix(".json")

        data = self.to_dict()
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'CognitiveAgent':
        """
        Load agent state from JSON file.

        Args:
            path: File path to load from

        Returns:
            Reconstructed CognitiveAgent
        """
        path = Path(path)
        if not path.exists() and path.with_suffix(".json").exists():
            path = path.with_suffix(".json")

        with open(path, "r") as f:
            data = json.load(f)

        return cls.from_dict(data)


# =============================================================================
# GoT Integration Bridge
# =============================================================================


class GoTBridge:
    """
    Bridge between CognitiveAgent and Graph of Thought (GoT) system.

    Enables bidirectional sync between:
    - CognitiveAgent Goals <-> GoT Tasks
    - CognitiveAgent Atoms <-> GoT Entities (future)

    This is a loose coupling - CognitiveAgent works standalone,
    but can optionally sync with GoT when available.

    Usage:
        from cortical.core.bootstrap import create_container
        from cortical.got.api import GoTManager

        container = create_container()
        got_manager = container.resolve(GoTManager)

        bridge = GoTBridge(agent, got_manager)
        bridge.sync_goals_to_tasks()
    """

    def __init__(
        self,
        agent: CognitiveAgent,
        got_manager: Any = None,  # GoTManager, but optional import
    ):
        """
        Initialize the bridge.

        Args:
            agent: The CognitiveAgent to sync
            got_manager: Optional GoTManager for GoT operations
        """
        self.agent = agent
        self.got_manager = got_manager
        self._goal_to_task_map: Dict[str, str] = {}  # goal_id -> task_id
        self._task_to_goal_map: Dict[str, str] = {}  # task_id -> goal_id

    def sync_goals_to_tasks(self) -> Dict[str, str]:
        """
        Export agent goals as GoT tasks.

        Creates new tasks for goals that don't have corresponding tasks,
        updates existing tasks if goal progress changed.

        Returns:
            Mapping of goal_id -> task_id
        """
        if self.got_manager is None:
            raise RuntimeError("GoT manager not available")

        results = {}

        for goal in self.agent.goals._goals.values():
            if goal.id in self._goal_to_task_map:
                # Update existing task
                task_id = self._goal_to_task_map[goal.id]
                status = "completed" if goal.is_complete() else (
                    "in_progress" if goal.progress > 0 else "pending"
                )
                try:
                    self.got_manager.update_task(
                        task_id,
                        status=status,
                        properties={"progress": goal.progress},
                    )
                except Exception:
                    pass  # Task may have been deleted
            else:
                # Create new task
                priority = "critical" if goal.importance > 0.8 else (
                    "high" if goal.importance > 0.5 else "medium"
                )
                task = self.got_manager.create_task(
                    title=goal.description,
                    priority=priority,
                    properties={
                        "cognitive_goal_id": goal.id,
                        "target_state": str(goal.target_state),
                        "progress": goal.progress,
                    },
                )
                self._goal_to_task_map[goal.id] = task.id
                self._task_to_goal_map[task.id] = goal.id

            results[goal.id] = self._goal_to_task_map.get(goal.id, "")

        return results

    def import_tasks_as_goals(self, status_filter: Optional[str] = None) -> int:
        """
        Import GoT tasks as agent goals.

        Args:
            status_filter: Optional status to filter tasks (e.g., "in_progress")

        Returns:
            Number of goals created
        """
        if self.got_manager is None:
            raise RuntimeError("GoT manager not available")

        count = 0

        # Query tasks
        tasks = self.got_manager.query_tasks(status=status_filter)

        for task in tasks:
            if task.id in self._task_to_goal_map:
                # Already mapped - update progress
                goal_id = self._task_to_goal_map[task.id]
                if goal_id in self.agent.goals._goals:
                    if task.status == "completed":
                        self.agent.goals._goals[goal_id].current_state = \
                            self.agent.goals._goals[goal_id].target_state
            else:
                # Create new goal from task
                importance = {"critical": 0.95, "high": 0.8, "medium": 0.5, "low": 0.3}.get(
                    task.priority, 0.5
                )
                goal = Goal(
                    id=f"got-{task.id}",
                    description=task.title,
                    target_state=1.0,  # Completion = 100%
                    current_state=1.0 if task.status == "completed" else 0.0,
                    importance=importance,
                )
                self.agent.goals.add_goal(goal)
                self._task_to_goal_map[task.id] = goal.id
                self._goal_to_task_map[goal.id] = task.id
                count += 1

        return count

    def get_task_for_goal(self, goal_id: str) -> Optional[str]:
        """Get the GoT task ID mapped to a goal."""
        return self._goal_to_task_map.get(goal_id)

    def get_goal_for_task(self, task_id: str) -> Optional[str]:
        """Get the goal ID mapped to a GoT task."""
        return self._task_to_goal_map.get(task_id)


# =============================================================================
# Container Module for Full Cognitive Agent
# =============================================================================


class CognitiveAgentModule(ContainerModule):
    """
    DI module for full cognitive agent stack.

    Registers:
        - StorageBackend
        - CognitiveGraph
        - CognitiveAgent

    Usage:
        container = Container()
        container.apply_module(CognitiveAgentModule())
        agent = container.resolve(CognitiveAgent)
    """

    def __init__(
        self,
        lifecycle: Lifecycle = Lifecycle.SINGLETON,
        storage_class: type = InMemoryStorage,
        working_memory_size: int = 4,
        attention_focus_size: int = 7,
    ):
        self.lifecycle = lifecycle
        self.storage_class = storage_class
        self.working_memory_size = working_memory_size
        self.attention_focus_size = attention_focus_size

    def register(self, container: Container) -> None:
        """Register cognitive agent components."""
        container.register(
            StorageBackend,
            self.storage_class,
            lifecycle=self.lifecycle,
        )
        container.register(
            CognitiveGraph,
            CognitiveGraph,
            lifecycle=self.lifecycle,
            storage=StorageBackend,
        )
        # Register factory for CognitiveAgent
        container.register_instance(
            CognitiveAgent,
            CognitiveAgent(
                working_memory_size=self.working_memory_size,
                attention_focus_size=self.attention_focus_size,
            )
        )
