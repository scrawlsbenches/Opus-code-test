"""
Minicolumn Module
=================

Core data structure representing a cortical minicolumn.

In the neocortex, minicolumns are vertical structures containing
~80-100 neurons that respond to similar features. This class models
that concept for text processing.
"""

from typing import Set, Dict, Optional, List
from dataclasses import dataclass, field, asdict


@dataclass
class Edge:
    """
    Typed edge with metadata for ConceptNet-style graph representation.

    Stores not just the connection weight, but also relation type,
    confidence, and source information.

    Attributes:
        target_id: ID of the target minicolumn
        weight: Connection strength (accumulated from multiple sources)
        relation_type: Semantic relation type ('co_occurrence', 'IsA', 'PartOf', etc.)
        confidence: Confidence score for this edge (0.0 to 1.0)
        source: Where this edge came from ('corpus', 'semantic', 'inferred')

    Example:
        edge = Edge("L0_network", 0.8, relation_type='RelatedTo', confidence=0.9)
    """
    target_id: str
    weight: float = 1.0
    relation_type: str = 'co_occurrence'
    confidence: float = 1.0
    source: str = 'corpus'

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'Edge':
        """Create an Edge from dictionary representation."""
        return cls(
            target_id=data['target_id'],
            weight=data.get('weight', 1.0),
            relation_type=data.get('relation_type', 'co_occurrence'),
            confidence=data.get('confidence', 1.0),
            source=data.get('source', 'corpus')
        )


class Minicolumn:
    """
    A minicolumn represents a single concept/feature at a given hierarchy level.
    
    In the biological neocortex, minicolumns are the fundamental processing
    units. Here, each minicolumn represents:
    - Layer 0: A single token/word
    - Layer 1: A bigram pattern
    - Layer 2: A concept cluster
    - Layer 3: A document
    
    Attributes:
        id: Unique identifier (e.g., "L0_neural")
        content: The actual content (word, bigram, doc_id)
        layer: Which layer this column belongs to
        activation: Current activation level (like neural firing rate)
        occurrence_count: How many times this has been observed
        document_ids: Which documents contain this content
        lateral_connections: Connections to other columns at same layer (simple weight dict)
        typed_connections: Typed edges with metadata (ConceptNet-style)
        feedforward_sources: IDs of columns that feed into this one (deprecated, use feedforward_connections)
        feedforward_connections: Weighted connections to lower layer columns
        feedback_connections: Weighted connections to higher layer columns
        tfidf: TF-IDF weight for this term
        tfidf_per_doc: Document-specific TF-IDF scores
        pagerank: Importance score from PageRank algorithm
        cluster_id: Which cluster this belongs to (for Layer 0)
        doc_occurrence_counts: Per-document occurrence counts for accurate TF-IDF

    Example:
        col = Minicolumn("L0_neural", "neural", 0)
        col.occurrence_count = 15
        col.add_lateral_connection("L0_network", 0.8)
        col.add_typed_connection("L0_network", 0.8, relation_type='RelatedTo')
    """

    __slots__ = [
        'id', 'content', 'layer', 'activation', 'occurrence_count',
        'document_ids', '_lateral_cache', '_lateral_cache_valid', 'typed_connections',
        'feedforward_sources', 'feedforward_connections', 'feedback_connections',
        'tfidf', 'tfidf_per_doc', 'pagerank', 'cluster_id',
        'doc_occurrence_counts', 'name_tokens'
    ]
    
    def __init__(self, id: str, content: str, layer: int):
        """
        Initialize a minicolumn.
        
        Args:
            id: Unique identifier for this column
            content: The content this column represents
            layer: Layer number (0-3)
        """
        self.id = id
        self.content = content
        self.layer = layer
        self.activation = 0.0
        self.occurrence_count = 0
        self.document_ids: Set[str] = set()
        self._lateral_cache: Dict[str, float] = {}  # Cached view of typed_connections weights
        self._lateral_cache_valid: bool = True  # Cache starts valid (empty matches empty)
        self.typed_connections: Dict[str, Edge] = {}  # Single source of truth for connections
        self.feedforward_sources: Set[str] = set()  # Deprecated: use feedforward_connections
        self.feedforward_connections: Dict[str, float] = {}  # Weighted links to lower layer
        self.feedback_connections: Dict[str, float] = {}  # Weighted links to higher layer
        self.tfidf = 0.0
        self.tfidf_per_doc: Dict[str, float] = {}
        self.pagerank = 1.0
        self.cluster_id: Optional[int] = None
        self.doc_occurrence_counts: Dict[str, int] = {}
        self.name_tokens: Optional[Set[str]] = None  # Cached tokenized name for document minicolumns

    @property
    def lateral_connections(self) -> Dict[str, float]:
        """
        Get lateral connections as a simple weight dictionary.

        This is a cached view derived from typed_connections. For backward
        compatibility, this returns a dict mapping target_id to weight.
        The cache is invalidated when connections are added/modified.

        Returns:
            Dictionary mapping target_id to connection weight

        Note:
            This property returns a reference to the internal cache. Modifying
            it directly is deprecated - use add_lateral_connection() or
            set_lateral_connection_weight() instead.
        """
        if not self._lateral_cache_valid:
            self._lateral_cache = {
                target_id: edge.weight
                for target_id, edge in self.typed_connections.items()
            }
            self._lateral_cache_valid = True
        return self._lateral_cache

    @lateral_connections.setter
    def lateral_connections(self, value: Dict[str, float]) -> None:
        """
        Set lateral connections from a dictionary (for deserialization).

        This converts simple weight entries to typed connections with
        default metadata (relation_type='co_occurrence', source='corpus').

        Args:
            value: Dictionary mapping target_id to weight
        """
        # Clear existing and rebuild from the provided dict
        self.typed_connections.clear()
        for target_id, weight in value.items():
            self.typed_connections[target_id] = Edge(
                target_id=target_id,
                weight=weight,
                relation_type='co_occurrence',
                confidence=1.0,
                source='corpus'
            )
        self._lateral_cache = dict(value)  # Copy to avoid external mutation
        self._lateral_cache_valid = True

    def _invalidate_lateral_cache(self) -> None:
        """Invalidate the lateral connections cache."""
        self._lateral_cache_valid = False

    def add_lateral_connection(self, target_id: str, weight: float = 1.0) -> None:
        """
        Add or strengthen a lateral connection to another column.

        Lateral connections represent associations learned through
        co-occurrence (like Hebbian learning: "neurons that fire together
        wire together").

        Args:
            target_id: ID of the target minicolumn
            weight: Connection strength to add
        """
        if target_id in self.typed_connections:
            existing = self.typed_connections[target_id]
            self.typed_connections[target_id] = Edge(
                target_id=target_id,
                weight=existing.weight + weight,
                relation_type=existing.relation_type,
                confidence=existing.confidence,
                source=existing.source
            )
        else:
            self.typed_connections[target_id] = Edge(
                target_id=target_id,
                weight=weight,
                relation_type='co_occurrence',
                confidence=1.0,
                source='corpus'
            )
        self._invalidate_lateral_cache()

    def add_lateral_connections_batch(self, connections: Dict[str, float]) -> None:
        """
        Add or strengthen multiple lateral connections at once.

        More efficient than calling add_lateral_connection() in a loop
        because it reduces function call overhead.

        Args:
            connections: Dictionary mapping target_id to weight to add
        """
        typed = self.typed_connections
        for target_id, weight in connections.items():
            if target_id in typed:
                existing = typed[target_id]
                typed[target_id] = Edge(
                    target_id=target_id,
                    weight=existing.weight + weight,
                    relation_type=existing.relation_type,
                    confidence=existing.confidence,
                    source=existing.source
                )
            else:
                typed[target_id] = Edge(
                    target_id=target_id,
                    weight=weight,
                    relation_type='co_occurrence',
                    confidence=1.0,
                    source='corpus'
                )
        self._invalidate_lateral_cache()

    def set_lateral_connection_weight(self, target_id: str, weight: float) -> None:
        """
        Set the weight of a lateral connection directly (not additive).

        Unlike add_lateral_connection() which adds to existing weight,
        this method sets the weight to an exact value. Used primarily
        by semantic retrofitting which needs to adjust weights directly.

        Args:
            target_id: ID of the target minicolumn
            weight: Exact weight to set (replaces existing weight)

        Note:
            If the connection doesn't exist, it will be created with
            default metadata (relation_type='co_occurrence', source='corpus').
        """
        if target_id in self.typed_connections:
            existing = self.typed_connections[target_id]
            self.typed_connections[target_id] = Edge(
                target_id=target_id,
                weight=weight,
                relation_type=existing.relation_type,
                confidence=existing.confidence,
                source=existing.source
            )
        else:
            self.typed_connections[target_id] = Edge(
                target_id=target_id,
                weight=weight,
                relation_type='co_occurrence',
                confidence=1.0,
                source='corpus'
            )
        self._invalidate_lateral_cache()

    def add_typed_connection(
        self,
        target_id: str,
        weight: float = 1.0,
        relation_type: str = 'co_occurrence',
        confidence: float = 1.0,
        source: str = 'corpus'
    ) -> None:
        """
        Add or update a typed connection with metadata.

        Typed connections store ConceptNet-style edge information including
        relation type, confidence, and source. If a connection to the target
        already exists, the weight is accumulated and metadata is updated.

        Args:
            target_id: ID of the target minicolumn
            weight: Connection strength to add (accumulates with existing)
            relation_type: Semantic relation type ('co_occurrence', 'IsA', etc.)
            confidence: Confidence score for this edge (0.0 to 1.0)
            source: Where this edge came from ('corpus', 'semantic', 'inferred')

        Example:
            col.add_typed_connection("L0_network", 0.8, relation_type='RelatedTo')
            col.add_typed_connection("L0_brain", 0.5, relation_type='IsA', source='semantic')
        """
        if target_id in self.typed_connections:
            # Accumulate weight, keep most informative metadata
            existing = self.typed_connections[target_id]
            new_weight = existing.weight + weight
            # Prefer more specific relation types over 'co_occurrence'
            new_relation = relation_type if relation_type != 'co_occurrence' else existing.relation_type
            # Weighted average of confidence (allows confidence to decrease with weaker evidence)
            new_confidence = (existing.confidence * existing.weight + confidence * weight) / new_weight
            # Prefer semantic/inferred over corpus
            source_priority = {'inferred': 3, 'semantic': 2, 'corpus': 1}
            new_source = source if source_priority.get(source, 0) > source_priority.get(existing.source, 0) else existing.source
            self.typed_connections[target_id] = Edge(
                target_id=target_id,
                weight=new_weight,
                relation_type=new_relation,
                confidence=new_confidence,
                source=new_source
            )
        else:
            self.typed_connections[target_id] = Edge(
                target_id=target_id,
                weight=weight,
                relation_type=relation_type,
                confidence=confidence,
                source=source
            )

        # Invalidate cache so lateral_connections property rebuilds on next access
        self._invalidate_lateral_cache()

    def get_typed_connection(self, target_id: str) -> Optional[Edge]:
        """
        Get a typed connection by target ID.

        Args:
            target_id: ID of the target minicolumn

        Returns:
            Edge object if exists, None otherwise
        """
        return self.typed_connections.get(target_id)

    def get_connections_by_type(self, relation_type: str) -> List[Edge]:
        """
        Get all typed connections with a specific relation type.

        Args:
            relation_type: Relation type to filter by (e.g., 'IsA', 'PartOf')

        Returns:
            List of Edge objects matching the relation type
        """
        return [
            edge for edge in self.typed_connections.values()
            if edge.relation_type == relation_type
        ]

    def get_connections_by_source(self, source: str) -> List[Edge]:
        """
        Get all typed connections from a specific source.

        Args:
            source: Source to filter by ('corpus', 'semantic', 'inferred')

        Returns:
            List of Edge objects from the specified source
        """
        return [
            edge for edge in self.typed_connections.values()
            if edge.source == source
        ]

    def add_feedforward_connection(self, target_id: str, weight: float = 1.0) -> None:
        """
        Add or strengthen a feedforward connection to a lower layer column.

        Feedforward connections link higher-level representations to their
        component parts (e.g., bigram → tokens, concept → tokens).

        Args:
            target_id: ID of the lower-layer minicolumn
            weight: Connection strength to add
        """
        self.feedforward_connections[target_id] = (
            self.feedforward_connections.get(target_id, 0) + weight
        )
        # Also maintain legacy feedforward_sources for backward compatibility
        self.feedforward_sources.add(target_id)

    def add_feedback_connection(self, target_id: str, weight: float = 1.0) -> None:
        """
        Add or strengthen a feedback connection to a higher layer column.

        Feedback connections link lower-level representations to the
        higher-level structures they participate in (e.g., token → bigrams).

        Args:
            target_id: ID of the higher-layer minicolumn
            weight: Connection strength to add
        """
        self.feedback_connections[target_id] = (
            self.feedback_connections.get(target_id, 0) + weight
        )
    
    def connection_count(self) -> int:
        """Return the number of lateral connections."""
        return len(self.lateral_connections)
    
    def top_connections(self, n: int = 5) -> list:
        """
        Get the strongest lateral connections.
        
        Args:
            n: Number of connections to return
            
        Returns:
            List of (target_id, weight) tuples, sorted by weight
        """
        sorted_conns = sorted(
            self.lateral_connections.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_conns[:n]
    
    def to_dict(self) -> Dict:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary representation of this minicolumn
        """
        d = {
            'id': self.id,
            'content': self.content,
            'layer': self.layer,
            'activation': self.activation,
            'occurrence_count': self.occurrence_count,
            'document_ids': list(self.document_ids),
            'lateral_connections': self.lateral_connections,
            'typed_connections': {
                target_id: edge.to_dict()
                for target_id, edge in self.typed_connections.items()
            },
            'feedforward_sources': list(self.feedforward_sources),
            'feedforward_connections': self.feedforward_connections,
            'feedback_connections': self.feedback_connections,
            'tfidf': self.tfidf,
            'tfidf_per_doc': self.tfidf_per_doc,
            'pagerank': self.pagerank,
            'cluster_id': self.cluster_id,
            'doc_occurrence_counts': self.doc_occurrence_counts
        }
        # Only include name_tokens if it's set (for document minicolumns)
        if self.name_tokens is not None:
            d['name_tokens'] = list(self.name_tokens)
        return d
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Minicolumn':
        """
        Create a minicolumn from dictionary representation.

        Handles backward compatibility: if typed_connections is present, use it.
        If only lateral_connections is present (old format), convert it.

        Args:
            data: Dictionary with minicolumn data

        Returns:
            New Minicolumn instance
        """
        col = cls(data['id'], data['content'], data['layer'])
        col.activation = data.get('activation', 0.0)
        col.occurrence_count = data.get('occurrence_count', 0)
        col.document_ids = set(data.get('document_ids', []))

        # Handle connection deserialization with backward compatibility
        typed_conn_data = data.get('typed_connections', {})
        lateral_conn_data = data.get('lateral_connections', {})

        if typed_conn_data:
            # New format: deserialize typed connections directly
            col.typed_connections = {
                target_id: Edge.from_dict(edge_data)
                for target_id, edge_data in typed_conn_data.items()
            }
            col._lateral_cache_valid = False  # Will rebuild on first access
        elif lateral_conn_data:
            # Old format: convert lateral_connections to typed_connections
            col.lateral_connections = lateral_conn_data  # Uses setter
        # else: both empty, nothing to do (already initialized empty)

        col.feedforward_sources = set(data.get('feedforward_sources', []))
        col.feedforward_connections = data.get('feedforward_connections', {})
        col.feedback_connections = data.get('feedback_connections', {})
        col.tfidf = data.get('tfidf', 0.0)
        col.tfidf_per_doc = data.get('tfidf_per_doc', {})
        col.pagerank = data.get('pagerank', 1.0)
        col.cluster_id = data.get('cluster_id')
        col.doc_occurrence_counts = data.get('doc_occurrence_counts', {})
        # Deserialize name_tokens if present (for document minicolumns)
        col.name_tokens = set(data.get('name_tokens', [])) if data.get('name_tokens') else None
        return col
    
    def __repr__(self) -> str:
        return f"Minicolumn(id={self.id}, content={self.content}, layer={self.layer})"
