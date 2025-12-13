"""
Layers Module
=============

Defines the hierarchical layer structure inspired by the visual cortex.

The neocortex processes information through a hierarchy of layers,
each extracting progressively more abstract features:
- V1: Edge detection (→ tokens)
- V2: Simple patterns (→ bigrams)
- V4: Complex shapes (→ concepts)
- IT: Object recognition (→ documents)
"""

from enum import IntEnum
from typing import Dict, Optional, Iterator

from .minicolumn import Minicolumn


class CorticalLayer(IntEnum):
    """
    Enumeration of cortical processing layers.
    
    Maps visual cortex layers to text processing hierarchy:
        TOKENS (0): Like V1 - basic feature extraction (words)
        BIGRAMS (1): Like V2 - simple patterns (word pairs)
        CONCEPTS (2): Like V4 - higher-level features (clusters)
        DOCUMENTS (3): Like IT - holistic recognition (full docs)
    """
    TOKENS = 0      # Individual words (V1-like)
    BIGRAMS = 1     # Word pairs (V2-like)
    CONCEPTS = 2    # Concept clusters (V4-like)
    DOCUMENTS = 3   # Full documents (IT-like)
    
    @property
    def description(self) -> str:
        """Human-readable description of this layer."""
        descriptions = {
            0: "Token layer - individual words (V1-like)",
            1: "Bigram layer - word pairs (V2-like)",
            2: "Concept layer - semantic clusters (V4-like)",
            3: "Document layer - full documents (IT-like)"
        }
        return descriptions[self.value]
    
    @property
    def analogy(self) -> str:
        """Visual cortex analogy for this layer."""
        analogies = {
            0: "V1-like: Edge/token detection",
            1: "V2-like: Feature/pattern detection",
            2: "V4-like: Shape/concept detection",
            3: "IT-like: Object/document recognition"
        }
        return analogies[self.value]


class HierarchicalLayer:
    """
    A layer in the cortical hierarchy containing minicolumns.
    
    Each layer contains a collection of minicolumns and provides
    methods for managing them. Layers are organized hierarchically,
    with feedforward connections from lower to higher layers and
    lateral connections within each layer.
    
    Attributes:
        level: The layer number (0-3)
        minicolumns: Dictionary mapping content to Minicolumn objects
        _id_index: Secondary index mapping minicolumn IDs to content for O(1) lookups

    Example:
        layer = HierarchicalLayer(CorticalLayer.TOKENS)
        col = layer.get_or_create_minicolumn("neural")
        col.occurrence_count += 1
    """
    
    def __init__(self, level: CorticalLayer):
        """
        Initialize a hierarchical layer.
        
        Args:
            level: The CorticalLayer enum value for this layer
        """
        self.level = level
        self.minicolumns: Dict[str, Minicolumn] = {}
        self._id_index: Dict[str, str] = {}  # Maps minicolumn ID to content for O(1) lookup
    
    def get_or_create_minicolumn(self, content: str) -> Minicolumn:
        """
        Get existing minicolumn or create new one.
        
        This is the primary way to add content to a layer. If a
        minicolumn for this content already exists, return it.
        Otherwise, create a new one.
        
        Args:
            content: The content for this minicolumn
            
        Returns:
            The existing or newly created Minicolumn
        """
        if content not in self.minicolumns:
            col_id = f"L{self.level}_{content}"
            self.minicolumns[content] = Minicolumn(col_id, content, self.level)
            self._id_index[col_id] = content  # Maintain ID index for O(1) lookup
        return self.minicolumns[content]
    
    def get_minicolumn(self, content: str) -> Optional[Minicolumn]:
        """
        Get a minicolumn by content, or None if not found.

        Args:
            content: The content to look up

        Returns:
            The Minicolumn if found, None otherwise
        """
        return self.minicolumns.get(content)

    def get_by_id(self, col_id: str) -> Optional[Minicolumn]:
        """
        Get a minicolumn by its ID in O(1) time.

        This method uses a secondary index to avoid O(n) linear searches
        when looking up minicolumns by their ID rather than content.

        Args:
            col_id: The minicolumn ID (e.g., "L0_neural")

        Returns:
            The Minicolumn if found, None otherwise
        """
        content = self._id_index.get(col_id)
        return self.minicolumns.get(content) if content else None

    def remove_minicolumn(self, content: str) -> bool:
        """
        Remove a minicolumn from this layer.

        Args:
            content: The content key of the minicolumn to remove

        Returns:
            True if the minicolumn was found and removed, False otherwise
        """
        if content not in self.minicolumns:
            return False

        col = self.minicolumns[content]
        # Remove from ID index
        if col.id in self._id_index:
            del self._id_index[col.id]
        # Remove from minicolumns dict
        del self.minicolumns[content]
        return True

    def column_count(self) -> int:
        """Return the number of minicolumns in this layer."""
        return len(self.minicolumns)
    
    def total_connections(self) -> int:
        """Return total number of lateral connections in this layer."""
        return sum(col.connection_count() for col in self.minicolumns.values())
    
    def average_activation(self) -> float:
        """Calculate average activation across all minicolumns."""
        if not self.minicolumns:
            return 0.0
        return sum(col.activation for col in self.minicolumns.values()) / len(self.minicolumns)
    
    def activation_range(self) -> tuple:
        """Return (min, max) activation values."""
        if not self.minicolumns:
            return (0.0, 0.0)
        activations = [col.activation for col in self.minicolumns.values()]
        return (min(activations), max(activations))
    
    def sparsity(self, threshold_fraction: float = 0.5) -> float:
        """
        Calculate sparsity (fraction of columns with below-average activation).

        In biological neural networks, sparse representations are
        more efficient and allow for more distinct patterns. This measures
        the fraction of columns activated below a threshold relative to
        the average activation.

        Args:
            threshold_fraction: Fraction of average activation to use as threshold.
                Columns with activation < (average * threshold_fraction) count as sparse.
                Default 0.5 means columns below 50% of average activation.

        Returns:
            Fraction of columns with activation below threshold
        """
        if not self.minicolumns:
            return 0.0
        avg_activation = self.average_activation()
        if avg_activation == 0:
            return 1.0  # All columns are sparse if no activation
        threshold = avg_activation * threshold_fraction
        low_activation = sum(1 for col in self.minicolumns.values()
                            if col.activation < threshold)
        return low_activation / len(self.minicolumns)
    
    def top_by_pagerank(self, n: int = 10) -> list:
        """
        Get top minicolumns by PageRank score.
        
        Args:
            n: Number of results to return
            
        Returns:
            List of (content, pagerank) tuples
        """
        sorted_cols = sorted(
            self.minicolumns.values(),
            key=lambda c: c.pagerank,
            reverse=True
        )
        return [(col.content, col.pagerank) for col in sorted_cols[:n]]
    
    def top_by_tfidf(self, n: int = 10) -> list:
        """
        Get top minicolumns by TF-IDF score.
        
        Args:
            n: Number of results to return
            
        Returns:
            List of (content, tfidf) tuples
        """
        sorted_cols = sorted(
            self.minicolumns.values(),
            key=lambda c: c.tfidf,
            reverse=True
        )
        return [(col.content, col.tfidf) for col in sorted_cols[:n]]
    
    def top_by_activation(self, n: int = 10) -> list:
        """
        Get top minicolumns by activation level.
        
        Args:
            n: Number of results to return
            
        Returns:
            List of (content, activation) tuples
        """
        sorted_cols = sorted(
            self.minicolumns.values(),
            key=lambda c: c.activation,
            reverse=True
        )
        return [(col.content, col.activation) for col in sorted_cols[:n]]
    
    def __iter__(self) -> Iterator[Minicolumn]:
        """Iterate over minicolumns in this layer."""
        return iter(self.minicolumns.values())
    
    def __len__(self) -> int:
        """Return number of minicolumns."""
        return len(self.minicolumns)
    
    def __contains__(self, content: str) -> bool:
        """Check if content exists in this layer."""
        return content in self.minicolumns
    
    def to_dict(self) -> Dict:
        """
        Convert layer to dictionary for serialization.
        
        Returns:
            Dictionary representation of this layer
        """
        return {
            'level': self.level,
            'minicolumns': {
                content: col.to_dict() 
                for content, col in self.minicolumns.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'HierarchicalLayer':
        """
        Create a layer from dictionary representation.

        Args:
            data: Dictionary with layer data

        Returns:
            New HierarchicalLayer instance

        Raises:
            ValueError: If layer value is invalid (must be 0-3)
        """
        # Validate layer value before creating enum
        level_value = data['level']
        if level_value not in [0, 1, 2, 3]:
            raise ValueError(
                f"Invalid layer value {level_value} in layer data. "
                f"Layer values must be 0-3 (TOKENS=0, BIGRAMS=1, CONCEPTS=2, DOCUMENTS=3)."
            )
        layer = cls(CorticalLayer(level_value))
        for content, col_data in data.get('minicolumns', {}).items():
            col = Minicolumn.from_dict(col_data)
            layer.minicolumns[content] = col
            layer._id_index[col.id] = content  # Rebuild ID index
        return layer
    
    def __repr__(self) -> str:
        return f"HierarchicalLayer(level={self.level.name}, columns={len(self.minicolumns)})"
