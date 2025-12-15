"""
Analysis Module
===============

Graph analysis algorithms for the cortical network.

Contains implementations of:
- PageRank for importance scoring
- TF-IDF for term weighting
- Louvain community detection for clustering (recommended)
- Label propagation for clustering (legacy, for backward compatibility)
- Activation propagation for information flow
- Connection building for all layers
- Clustering quality metrics
"""

# Import all public functions from submodules
from .pagerank import (
    compute_pagerank,
    compute_semantic_pagerank,
    compute_hierarchical_pagerank,
    _pagerank_core,
)

from .tfidf import (
    compute_tfidf,
    compute_bm25,
    _tfidf_core,
    _bm25_core,
)

from .clustering import (
    cluster_by_louvain,
    cluster_by_label_propagation,
    build_concept_clusters,
    _louvain_core,
)

from .connections import (
    compute_document_connections,
    compute_bigram_connections,
    compute_concept_connections,
)

from .activation import (
    propagate_activation,
)

from .quality import (
    compute_clustering_quality,
    _compute_modularity,
    _compute_silhouette,
    _compute_cluster_balance,
    _generate_quality_assessment,
    _modularity_core,
    _silhouette_core,
)

from .utils import (
    SparseMatrix,
    cosine_similarity,
    _doc_similarity,
    _vector_similarity,
)

# Define __all__ for explicit exports
__all__ = [
    # PageRank algorithms
    'compute_pagerank',
    'compute_semantic_pagerank',
    'compute_hierarchical_pagerank',
    '_pagerank_core',

    # TF-IDF and BM25
    'compute_tfidf',
    'compute_bm25',
    '_tfidf_core',
    '_bm25_core',

    # Clustering algorithms
    'cluster_by_louvain',
    'cluster_by_label_propagation',
    'build_concept_clusters',
    '_louvain_core',

    # Connection building
    'compute_document_connections',
    'compute_bigram_connections',
    'compute_concept_connections',

    # Activation propagation
    'propagate_activation',

    # Quality metrics
    'compute_clustering_quality',
    '_compute_modularity',
    '_compute_silhouette',
    '_compute_cluster_balance',
    '_generate_quality_assessment',
    '_modularity_core',
    '_silhouette_core',

    # Utilities
    'SparseMatrix',
    'cosine_similarity',
    '_doc_similarity',
    '_vector_similarity',
]
