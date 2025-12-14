"""
Cortical Text Processor - Main processor package.

This package splits the monolithic processor.py into focused modules:
- core.py: Initialization, staleness tracking, layer management
- documents.py: Document processing, adding, removing, metadata
- compute.py: Analysis computations, clustering, embeddings, semantics
- query_api.py: Search, expansion, retrieval methods
- introspection.py: State inspection, fingerprints, gaps, summaries
- persistence_api.py: Save, load, export methods

The CorticalTextProcessor class is composed from mixins in each module,
maintaining full backwards compatibility with the original API.
"""

from .core import CoreMixin
from .documents import DocumentsMixin
from .compute import ComputeMixin
from .query_api import QueryMixin
from .introspection import IntrospectionMixin
from .persistence_api import PersistenceMixin


class CorticalTextProcessor(
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

    Example:
        >>> from cortical import CorticalTextProcessor
        >>> processor = CorticalTextProcessor()
        >>> processor.process_document("doc1", "Neural networks process data.")
        >>> processor.compute_all()
        >>> results = processor.find_documents_for_query("neural")
        >>> processor.save("corpus.pkl")

    The processor is composed from focused mixins:
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
