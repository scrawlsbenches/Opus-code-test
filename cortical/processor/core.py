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
from ..observability import MetricsCollector

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
        config: Optional[CorticalConfig] = None,
        enable_metrics: bool = False
    ):
        """
        Initialize the Cortical Text Processor.

        Args:
            tokenizer: Optional custom tokenizer. Defaults to standard Tokenizer.
            config: Optional configuration. Defaults to CorticalConfig with defaults.
            enable_metrics: Enable timing and metrics collection for observability.
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
        # Document length tracking for BM25
        self.doc_lengths: Dict[str, int] = {}  # doc_id -> token count
        self.avg_doc_length: float = 0.0  # Average document length in tokens
        # Track which computations are stale and need recomputation
        self._stale_computations: set = set()
        # LRU cache for query expansion results
        self._query_expansion_cache: Dict[str, Dict[str, float]] = {}
        self._query_cache_max_size: int = 100
        # Observability: metrics collection
        self._metrics = MetricsCollector(enabled=enable_metrics)

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

    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all collected metrics.

        Returns:
            Dict mapping operation names to their statistics
            (count, total_ms, avg_ms, min_ms, max_ms)

        Example:
            >>> processor = CorticalTextProcessor(enable_metrics=True)
            >>> processor.compute_all()
            >>> metrics = processor.get_metrics()
            >>> print(f"compute_all: {metrics['compute_all']['avg_ms']:.2f}ms")
        """
        return self._metrics.get_all_stats()

    def get_metrics_summary(self) -> str:
        """
        Get a human-readable summary of all metrics.

        Returns:
            Formatted string with metrics table

        Example:
            >>> processor = CorticalTextProcessor(enable_metrics=True)
            >>> processor.compute_all()
            >>> print(processor.get_metrics_summary())
        """
        return self._metrics.get_summary()

    def reset_metrics(self) -> None:
        """Clear all collected metrics."""
        self._metrics.reset()

    def enable_metrics(self) -> None:
        """Enable metrics collection."""
        self._metrics.enable()

    def disable_metrics(self) -> None:
        """Disable metrics collection."""
        self._metrics.disable()

    def record_metric(self, metric_name: str, count: int = 1) -> None:
        """
        Record a custom count metric.

        Args:
            metric_name: Name of the metric
            count: Count to add (default 1)

        Example:
            >>> processor.record_metric("cache_hits")
            >>> processor.record_metric("documents_processed", count=10)
        """
        self._metrics.record_count(metric_name, count)
