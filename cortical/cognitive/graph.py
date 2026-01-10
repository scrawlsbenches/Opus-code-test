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

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
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
        sti: Short-term importance (attention)
        lti: Long-term importance (persistent significance)
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    atom_type: AtomType = AtomType.CONCEPT
    name: str = ""
    outgoing: List[str] = field(default_factory=list)
    tv: TruthValue = field(default_factory=TruthValue)
    sti: float = 0.0  # Short-term importance
    lti: float = 0.0  # Long-term importance

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
