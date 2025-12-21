"""
Cortical Text Processor - Main processor package.

This package splits the monolithic processor.py into focused modules:
- core.py: Initialization, staleness tracking, layer management
- documents.py: Document processing, adding, removing, metadata
- compute.py: Analysis computations, clustering, embeddings, semantics
- query_api.py: Search, expansion, retrieval methods
- introspection.py: State inspection, fingerprints, gaps, summaries
- persistence_api.py: Save, load, export methods
- spark_api.py: SparkSLM integration for first-blitz priming

The CorticalTextProcessor class is composed from mixins in each module,
maintaining full backwards compatibility with the original API.
"""

from .core import CoreMixin
from .spark_api import SparkMixin
from .documents import DocumentsMixin
from .compute import ComputeMixin
from .query_api import QueryMixin
from .introspection import IntrospectionMixin
from .persistence_api import PersistenceMixin


class CorticalTextProcessor(
    SparkMixin,  # Must come before CoreMixin for proper __init__ chaining
    CoreMixin,
    DocumentsMixin,
    ComputeMixin,
    QueryMixin,
    IntrospectionMixin,
    PersistenceMixin
):
    """
    Neocortex-inspired text processing system.

    This class provides a complete API for:
    - Document processing and management
    - TF-IDF, PageRank, and graph analysis
    - Semantic relation extraction
    - Query expansion and document search
    - Passage retrieval for RAG systems
    - State persistence and export
    - SparkSLM first-blitz priming (optional)

    Example:
        >>> from cortical import CorticalTextProcessor
        >>> processor = CorticalTextProcessor()
        >>> processor.process_document("doc1", "Neural networks process data.")
        >>> processor.compute_all()
        >>> results = processor.find_documents_for_query("neural")
        >>> processor.save("corpus.pkl")

    SparkSLM Example:
        >>> processor = CorticalTextProcessor(spark=True)
        >>> processor.process_document("doc1", "Neural networks...")
        >>> processor.train_spark()
        >>> hints = processor.prime_query("neural")

    The processor is composed from focused mixins:
    - SparkMixin: First-blitz priming and alignment (optional)
    - CoreMixin: Initialization, staleness tracking
    - DocumentsMixin: Document add/remove operations
    - ComputeMixin: Analysis and computation methods
    - QueryMixin: Search and retrieval methods
    - IntrospectionMixin: State inspection and comparison
    - PersistenceMixin: Save/load operations
    """
    pass


# Re-export for backwards compatibility
__all__ = ['CorticalTextProcessor']
