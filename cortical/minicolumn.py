"""
Minicolumn Module
=================

Core data structure representing a cortical minicolumn.

In the neocortex, minicolumns are vertical structures containing
~80-100 neurons that respond to similar features. This class models
that concept for text processing.
"""

from typing import Set, Dict, Optional
from collections import defaultdict


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
        lateral_connections: Connections to other columns at same layer
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
    """
    
    __slots__ = [
        'id', 'content', 'layer', 'activation', 'occurrence_count',
        'document_ids', 'lateral_connections', 'feedforward_sources',
        'feedforward_connections', 'feedback_connections',
        'tfidf', 'tfidf_per_doc', 'pagerank', 'cluster_id',
        'doc_occurrence_counts'
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
        self.lateral_connections: Dict[str, float] = {}
        self.feedforward_sources: Set[str] = set()  # Deprecated: use feedforward_connections
        self.feedforward_connections: Dict[str, float] = {}  # Weighted links to lower layer
        self.feedback_connections: Dict[str, float] = {}  # Weighted links to higher layer
        self.tfidf = 0.0
        self.tfidf_per_doc: Dict[str, float] = {}
        self.pagerank = 1.0
        self.cluster_id: Optional[int] = None
        self.doc_occurrence_counts: Dict[str, int] = {}
    
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
        self.lateral_connections[target_id] = (
            self.lateral_connections.get(target_id, 0) + weight
        )

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
        return {
            'id': self.id,
            'content': self.content,
            'layer': self.layer,
            'activation': self.activation,
            'occurrence_count': self.occurrence_count,
            'document_ids': list(self.document_ids),
            'lateral_connections': self.lateral_connections,
            'feedforward_sources': list(self.feedforward_sources),
            'feedforward_connections': self.feedforward_connections,
            'feedback_connections': self.feedback_connections,
            'tfidf': self.tfidf,
            'tfidf_per_doc': self.tfidf_per_doc,
            'pagerank': self.pagerank,
            'cluster_id': self.cluster_id,
            'doc_occurrence_counts': self.doc_occurrence_counts
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Minicolumn':
        """
        Create a minicolumn from dictionary representation.

        Args:
            data: Dictionary with minicolumn data

        Returns:
            New Minicolumn instance
        """
        col = cls(data['id'], data['content'], data['layer'])
        col.activation = data.get('activation', 0.0)
        col.occurrence_count = data.get('occurrence_count', 0)
        col.document_ids = set(data.get('document_ids', []))
        col.lateral_connections = data.get('lateral_connections', {})
        col.feedforward_sources = set(data.get('feedforward_sources', []))
        col.feedforward_connections = data.get('feedforward_connections', {})
        col.feedback_connections = data.get('feedback_connections', {})
        col.tfidf = data.get('tfidf', 0.0)
        col.tfidf_per_doc = data.get('tfidf_per_doc', {})
        col.pagerank = data.get('pagerank', 1.0)
        col.cluster_id = data.get('cluster_id')
        col.doc_occurrence_counts = data.get('doc_occurrence_counts', {})
        return col
    
    def __repr__(self) -> str:
        return f"Minicolumn(id={self.id}, content={self.content}, layer={self.layer})"
