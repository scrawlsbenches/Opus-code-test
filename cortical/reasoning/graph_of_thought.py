"""
Graph of Thought: Network-Based Reasoning Data Structures.

This module provides dataclasses and enums for representing thought processes
as graphs, enabling network-based reasoning for software development tasks.

The graph metaphor captures how ideas connect in networks rather than chains,
supporting multiple entry points, cross-cutting concerns, feedback loops, and
emergent clusters of related concepts.

Key components:
- NodeType: Categorizes units of thought (concepts, questions, decisions, etc.)
- EdgeType: Categorizes relationships between thoughts (requires, enables, etc.)
- ThoughtNode: Represents a discrete concept, decision, question, or fact
- ThoughtEdge: Represents a typed relationship between nodes
- ThoughtCluster: Represents groups of densely-connected nodes

See docs/graph-of-thought.md for detailed usage patterns and examples.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Set


class NodeType(Enum):
    """
    Types of nodes in a thought graph.

    Each node type represents a different kind of cognitive unit:

    Core types (from original design):
    - CONCEPT: An idea, pattern, or abstraction (e.g., "Dependency Injection")
    - QUESTION: Something unknown or uncertain (e.g., "How should we handle auth?")
    - DECISION: A choice point with options (e.g., "REST vs GraphQL")
    - FACT: Verified information (e.g., "Response time is 200ms")
    - TASK: Work to be done (e.g., "Implement caching")
    - ARTIFACT: Something created (e.g., "auth.py module")
    - INSIGHT: A learning or realization (e.g., "The bottleneck is in serialization")

    Extended types (for reasoning patterns):
    - HYPOTHESIS: A proposed explanation to be tested (e.g., "Auth fails due to token expiry")
    - OPTION: A choice within a decision (e.g., "Use OAuth" as option for auth decision)
    - EVIDENCE: Data supporting or refuting something (e.g., "Logs show 401 errors")
    - OBSERVATION: Something noticed during investigation (e.g., "Response times spike at 3pm")
    - GOAL: A desired outcome (e.g., "Reduce latency to <100ms")
    - CONTEXT: Background information (e.g., "User story: As a developer...")
    - CONSTRAINT: A limitation or requirement (e.g., "Must support IE11")
    - ACTION: A concrete step to take (e.g., "Add caching layer")
    """

    # Core types
    CONCEPT = "concept"
    QUESTION = "question"
    DECISION = "decision"
    FACT = "fact"
    TASK = "task"
    ARTIFACT = "artifact"
    INSIGHT = "insight"

    # Extended types for reasoning patterns
    HYPOTHESIS = "hypothesis"
    OPTION = "option"
    EVIDENCE = "evidence"
    OBSERVATION = "observation"
    GOAL = "goal"
    CONTEXT = "context"
    CONSTRAINT = "constraint"
    ACTION = "action"

    # Semantic types for attention mechanisms
    ENTITY = "entity"      # A named thing (person, character, organization)
    LOCATION = "location"  # A place or setting
    OBJECT = "object"      # A physical or abstract object

    def __repr__(self) -> str:
        """Return enum representation."""
        return f"NodeType.{self.name}"


class EdgeType(Enum):
    """
    Types of edges (relationships) in a thought graph.

    Edges represent HOW thoughts relate, not just THAT they relate.
    Categorized into semantic, temporal, epistemic, and practical relationships.

    Semantic edges (meaning relationships):
    - REQUIRES: A requires B to exist/function
    - ENABLES: A makes B possible
    - CONFLICTS: A and B cannot both be true/chosen
    - SUPPORTS: A provides evidence for B
    - REFUTES: A provides evidence against B
    - SIMILAR: A and B share significant properties
    - CONTAINS: A includes B as a component
    - CONTRADICTS: A contradicts B (stronger than REFUTES)

    Temporal edges (time relationships):
    - PRECEDES: A must happen before B
    - TRIGGERS: A causes B to happen
    - CAUSED_BY: A was caused by B (inverse of TRIGGERS, for root cause analysis)
    - BLOCKS: A prevents B until resolved

    Epistemic edges (knowledge relationships):
    - ANSWERS: A answers question B
    - RAISES: A raises question B
    - EXPLORES: A explores/investigates B
    - OBSERVES: A observes/notices B
    - SUGGESTS: A suggests B as possibility

    Practical edges (work relationships):
    - IMPLEMENTS: A implements concept/decision B
    - TESTS: A tests/verifies B
    - DEPENDS_ON: A needs B to be complete first
    - REFINES: A refines/details B
    - MOTIVATES: A motivates/justifies B
    - JUSTIFIES: A justifies/rationalizes B (decision → task)

    Structural edges (organization relationships):
    - HAS_OPTION: A (decision) has B as an option
    - HAS_ASPECT: A has B as an aspect/dimension
    - PART_OF: A is part of B (hierarchical containment, e.g., sprint → epic)
    """

    # Semantic edges
    REQUIRES = "requires"
    ENABLES = "enables"
    CONFLICTS = "conflicts"
    SUPPORTS = "supports"
    REFUTES = "refutes"
    SIMILAR = "similar"
    CONTAINS = "contains"
    CONTRADICTS = "contradicts"

    # Temporal edges
    PRECEDES = "precedes"
    TRIGGERS = "triggers"
    CAUSED_BY = "caused_by"  # Inverse of TRIGGERS: B was caused by A (for root cause analysis)
    BLOCKS = "blocks"

    # Epistemic edges
    ANSWERS = "answers"
    RAISES = "raises"
    EXPLORES = "explores"
    OBSERVES = "observes"
    SUGGESTS = "suggests"

    # Practical edges
    IMPLEMENTS = "implements"
    TESTS = "tests"
    DEPENDS_ON = "depends_on"
    REFINES = "refines"
    MOTIVATES = "motivates"
    JUSTIFIES = "justifies"  # Decision justifies task/entity

    # Structural edges
    HAS_OPTION = "has_option"
    HAS_ASPECT = "has_aspect"
    PART_OF = "part_of"  # Entity belongs to sprint/epic

    # Semantic role edges (for attention mechanisms)
    LOCATED_IN = "located_in"  # Entity is at location
    PERFORMS = "performs"      # Entity performs action
    USES = "uses"              # Action uses object

    def __repr__(self) -> str:
        """Return enum representation."""
        return f"EdgeType.{self.name}"


@dataclass(slots=True)
class ThoughtNode:
    """
    A node in a thought graph representing a discrete unit of thought.

    Attributes:
        id: Unique identifier for the node
        node_type: The type of thought this node represents
        content: The actual content (text, description, or value)
        properties: Type-specific properties stored as key-value pairs
                   Examples:
                   - CONCEPT: {'definition': str, 'examples': list}
                   - QUESTION: {'context': str, 'urgency': str}
                   - DECISION: {'options': list, 'chosen': str, 'rationale': str}
                   - FACT: {'claim': str, 'evidence': str, 'confidence': float}
                   - TASK: {'status': str, 'dependencies': list, 'assignee': str}
                   - ARTIFACT: {'path': str, 'type': str, 'version': str}
                   - INSIGHT: {'how_discovered': str, 'implications': list}
        metadata: Additional metadata (creation time, author, tags, etc.)

    Example:
        >>> node = ThoughtNode(
        ...     id="C1",
        ...     node_type=NodeType.CONCEPT,
        ...     content="Dependency Injection",
        ...     properties={'definition': 'Pattern for providing dependencies...'},
        ...     metadata={'created': '2025-12-19', 'tags': ['design-pattern']}
        ... )
    """

    id: str
    node_type: NodeType
    content: str
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"ThoughtNode(id={self.id!r}, type={self.node_type.name}, content={self.content[:50]!r}...)"

    def __hash__(self) -> int:
        """Return hash based on ID."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Check equality based on ID."""
        if not isinstance(other, ThoughtNode):
            return NotImplemented
        return self.id == other.id


@dataclass(slots=True)
class ThoughtEdge:
    """
    An edge in a thought graph representing a relationship between nodes.

    Attributes:
        source_id: ID of the source node
        target_id: ID of the target node
        edge_type: The type of relationship this edge represents
        weight: Strength or importance of the relationship (0.0 to 1.0)
        confidence: Confidence in this relationship (0.0 to 1.0)
        bidirectional: Whether the relationship goes both ways
                      True for symmetric relations like SIMILAR
                      False for directed relations like REQUIRES

    Example:
        >>> edge = ThoughtEdge(
        ...     source_id="T1",
        ...     target_id="A1",
        ...     edge_type=EdgeType.IMPLEMENTS,
        ...     weight=0.9,
        ...     confidence=1.0,
        ...     bidirectional=False
        ... )
    """

    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    confidence: float = 1.0
    bidirectional: bool = False

    def __post_init__(self) -> None:
        """Validate edge attributes."""
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"Weight must be in [0.0, 1.0], got {self.weight}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0.0, 1.0], got {self.confidence}")

    def __repr__(self) -> str:
        """Return string representation."""
        arrow = "<-->" if self.bidirectional else "-->"
        return f"ThoughtEdge({self.source_id} {arrow}[{self.edge_type.name}] {self.target_id})"

    def __hash__(self) -> int:
        """Return hash based on source, target, and edge type."""
        return hash((self.source_id, self.target_id, self.edge_type))

    def __eq__(self, other: object) -> bool:
        """Check equality based on source, target, and edge type."""
        if not isinstance(other, ThoughtEdge):
            return NotImplemented
        return (
            self.source_id == other.source_id
            and self.target_id == other.target_id
            and self.edge_type == other.edge_type
        )


@dataclass(slots=True)
class ThoughtCluster:
    """
    A cluster of densely-connected nodes in a thought graph.

    Clusters represent coherent sub-problems or topic areas that emerge when
    nodes are densely connected. They help manage complexity by grouping
    related concepts that can be discussed independently.

    A cluster typically forms when:
    - Multiple nodes share many edges
    - Nodes can be discussed independently of others
    - There's a unifying concept or purpose
    - Changes to one node likely affect others in the group

    Attributes:
        id: Unique identifier for the cluster
        name: Human-readable name for the cluster
        node_ids: Set of node IDs belonging to this cluster
        properties: Cluster-level properties
                   Examples:
                   - 'core_nodes': List of central node IDs
                   - 'internal_edges': Number of edges within cluster
                   - 'external_edges': Number of edges leaving cluster
                   - 'coherence': Measure of how interconnected the cluster is
                   - 'topic': Main topic or theme of the cluster

    Example:
        >>> cluster = ThoughtCluster(
        ...     id="CL1",
        ...     name="Authentication",
        ...     node_ids={"C1", "D1", "Q1", "T1", "A1"},
        ...     properties={
        ...         'internal_edges': 12,
        ...         'external_edges': 3,
        ...         'coherence': 0.8,
        ...         'topic': 'user authentication and authorization'
        ...     }
        ... )
    """

    id: str
    name: str
    node_ids: Set[str] = field(default_factory=set)
    properties: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"ThoughtCluster(id={self.id!r}, name={self.name!r}, nodes={len(self.node_ids)})"

    def __hash__(self) -> int:
        """Return hash based on ID."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Check equality based on ID."""
        if not isinstance(other, ThoughtCluster):
            return NotImplemented
        return self.id == other.id

    def add_node(self, node_id: str) -> None:
        """
        Add a node to this cluster.

        Args:
            node_id: ID of the node to add
        """
        self.node_ids.add(node_id)

    def remove_node(self, node_id: str) -> None:
        """
        Remove a node from this cluster.

        Args:
            node_id: ID of the node to remove

        Raises:
            KeyError: If node_id is not in the cluster
        """
        self.node_ids.remove(node_id)

    def contains_node(self, node_id: str) -> bool:
        """
        Check if a node is in this cluster.

        Args:
            node_id: ID of the node to check

        Returns:
            True if the node is in the cluster, False otherwise
        """
        return node_id in self.node_ids

    @property
    def size(self) -> int:
        """Return the number of nodes in this cluster."""
        return len(self.node_ids)
