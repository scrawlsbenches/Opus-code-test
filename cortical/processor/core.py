"""
Core processor functionality: initialization, staleness tracking, and layer management.

This module contains the base class definition and core infrastructure that all
other processor mixins depend on.
"""

import logging
from typing import Dict, Optional, Any

from ..tokenizer import Tokenizer
from ..minicolumn import Minicolumn
from ..layers import CorticalLayer, HierarchicalLayer
from ..config import CorticalConfig

logger = logging.getLogger(__name__)


class CoreMixin:
    """
    Core mixin providing initialization and staleness tracking.

    This mixin defines the fundamental attributes and methods that all other
    processor functionality depends on.
    """

    # Computation types for tracking staleness
    COMP_TFIDF = 'tfidf'
    COMP_PAGERANK = 'pagerank'
    COMP_ACTIVATION = 'activation'
    COMP_DOC_CONNECTIONS = 'doc_connections'
    COMP_BIGRAM_CONNECTIONS = 'bigram_connections'
    COMP_CONCEPTS = 'concepts'
    COMP_EMBEDDINGS = 'embeddings'
    COMP_SEMANTICS = 'semantics'

    def __init__(
        self,
        tokenizer: Optional[Tokenizer] = None,
        config: Optional[CorticalConfig] = None
    ):
        """
        Initialize the Cortical Text Processor.

        Args:
            tokenizer: Optional custom tokenizer. Defaults to standard Tokenizer.
            config: Optional configuration. Defaults to CorticalConfig with defaults.
        """
        self.tokenizer = tokenizer or Tokenizer()
        self.config = config or CorticalConfig()
        self.layers: Dict[CorticalLayer, HierarchicalLayer] = {
            CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS),
            CorticalLayer.BIGRAMS: HierarchicalLayer(CorticalLayer.BIGRAMS),
            CorticalLayer.CONCEPTS: HierarchicalLayer(CorticalLayer.CONCEPTS),
            CorticalLayer.DOCUMENTS: HierarchicalLayer(CorticalLayer.DOCUMENTS),
        }
        self.documents: Dict[str, str] = {}
        self.document_metadata: Dict[str, Dict[str, Any]] = {}
        self.embeddings: Dict[str, list] = {}
        self.semantic_relations: list = []
        # Track which computations are stale and need recomputation
        self._stale_computations: set = set()
        # LRU cache for query expansion results
        self._query_expansion_cache: Dict[str, Dict[str, float]] = {}
        self._query_cache_max_size: int = 100

    def _mark_all_stale(self) -> None:
        """Mark all computations as stale (needing recomputation)."""
        self._stale_computations = {
            self.COMP_TFIDF,
            self.COMP_PAGERANK,
            self.COMP_ACTIVATION,
            self.COMP_DOC_CONNECTIONS,
            self.COMP_BIGRAM_CONNECTIONS,
            self.COMP_CONCEPTS,
            self.COMP_EMBEDDINGS,
            self.COMP_SEMANTICS,
        }

    def _mark_fresh(self, *computation_types: str) -> None:
        """Mark specified computations as fresh (up-to-date)."""
        for comp in computation_types:
            self._stale_computations.discard(comp)

    def is_stale(self, computation_type: str) -> bool:
        """
        Check if a specific computation is stale.

        Args:
            computation_type: One of COMP_TFIDF, COMP_PAGERANK, etc.

        Returns:
            True if the computation needs to be run again
        """
        return computation_type in self._stale_computations

    def get_stale_computations(self) -> set:
        """
        Get the set of computations that are currently stale.

        Returns:
            Set of computation type strings that need recomputation
        """
        return self._stale_computations.copy()

    def get_layer(self, layer: CorticalLayer) -> HierarchicalLayer:
        """Get a specific layer by enum."""
        return self.layers[layer]
