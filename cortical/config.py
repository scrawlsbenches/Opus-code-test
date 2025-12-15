"""
Configuration Module
====================

Centralized configuration for the Cortical Text Processor.

This module provides a dataclass-based configuration system that allows
users to customize algorithm parameters, thresholds, and defaults without
modifying the source code.

Example:
    from cortical import CorticalTextProcessor, CorticalConfig

    # Use custom configuration
    config = CorticalConfig(
        pagerank_damping=0.9,
        min_cluster_size=5,
        isolation_threshold=0.03
    )
    processor = CorticalTextProcessor(config=config)

    # Or modify defaults
    config = CorticalConfig()
    config.pagerank_iterations = 50
    processor = CorticalTextProcessor(config=config)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple, FrozenSet


@dataclass
class CorticalConfig:
    """
    Configuration settings for the Cortical Text Processor.

    All values have sensible defaults that work well for typical text corpora.
    Adjust these based on your specific use case:
    - Smaller corpora may need lower thresholds
    - Specialized domains may need different relation weights
    - Performance-critical applications may want fewer iterations

    Attributes:
        pagerank_damping: Damping factor for PageRank (0-1). Higher values
            give more weight to link structure vs uniform distribution.
        pagerank_iterations: Maximum PageRank iterations before stopping.
        pagerank_tolerance: Convergence threshold for PageRank. Algorithm
            stops when max change between iterations is below this value.

        min_cluster_size: Minimum nodes required to form a concept cluster.
            Smaller values create more fine-grained concepts.
        cluster_strictness: Controls clustering aggressiveness (0.0-1.0).
            Lower values allow more cross-topic mixing.
        louvain_resolution: Resolution parameter for Louvain clustering (>0).
            Higher values produce more, smaller clusters. Lower values produce
            fewer, larger clusters. Default 2.0 produces ~50-100 clusters for
            medium corpora (50-200 docs). Typical range: 1.0-10.0.

        isolation_threshold: Documents below this average similarity are
            considered isolated from the corpus.
        well_connected_threshold: Documents above this average similarity
            are considered well-integrated.
        weak_topic_tfidf_threshold: Terms above this TF-IDF are considered
            significant topics.
        bridge_similarity_min: Minimum similarity for bridge opportunities.
        bridge_similarity_max: Maximum similarity for bridge opportunities.

        chunk_size: Default chunk size for passage retrieval (in characters).
        chunk_overlap: Default overlap between chunks (in characters).

        max_query_expansions: Maximum expansion terms to add to queries.
        semantic_expansion_discount: Weight discount for semantic expansions
            relative to lateral connection expansions.

        cross_layer_damping: Damping at layer boundaries for hierarchical
            PageRank propagation.

        relation_weights: Weights for semantic relation types. Higher weights
            increase influence of that relation type in algorithms.
    """

    # PageRank settings
    pagerank_damping: float = 0.85
    pagerank_iterations: int = 20
    pagerank_tolerance: float = 1e-6

    # Clustering settings
    min_cluster_size: int = 3
    cluster_strictness: float = 1.0
    louvain_resolution: float = 2.0  # Resolution for Louvain clustering (higher = more clusters)

    # Gap detection thresholds
    isolation_threshold: float = 0.02
    well_connected_threshold: float = 0.03
    weak_topic_tfidf_threshold: float = 0.005
    bridge_similarity_min: float = 0.005
    bridge_similarity_max: float = 0.03

    # Chunking settings for RAG
    chunk_size: int = 512
    chunk_overlap: int = 128

    # Query expansion settings
    max_query_expansions: int = 10
    semantic_expansion_discount: float = 0.7

    # Scoring algorithm settings
    scoring_algorithm: str = 'bm25'  # 'tfidf' or 'bm25'
    bm25_k1: float = 1.2  # Term frequency saturation parameter (0.0-3.0, typical 1.2-2.0)
    bm25_b: float = 0.75  # Length normalization parameter (0.0-1.0)

    # Cross-layer propagation
    cross_layer_damping: float = 0.7

    # Bigram connection weights
    bigram_component_weight: float = 0.5
    bigram_chain_weight: float = 0.7
    bigram_cooccurrence_weight: float = 0.3

    # Concept connection thresholds
    concept_min_shared_docs: int = 1
    concept_min_jaccard: float = 0.1
    concept_embedding_threshold: float = 0.3

    # Multi-hop expansion settings
    multihop_max_hops: int = 2
    multihop_decay_factor: float = 0.5
    multihop_min_path_score: float = 0.3

    # Property inheritance settings
    inheritance_decay_factor: float = 0.7
    inheritance_max_depth: int = 5
    inheritance_boost_factor: float = 0.3

    # Relation weights for semantic algorithms
    relation_weights: Dict[str, float] = field(default_factory=lambda: {
        'IsA': 1.5,
        'PartOf': 1.2,
        'HasA': 1.0,
        'UsedFor': 0.8,
        'CapableOf': 0.7,
        'HasProperty': 1.1,
        'SimilarTo': 1.3,
        'RelatedTo': 1.0,
        'Causes': 1.0,
        'Antonym': 0.3,
        'DerivedFrom': 1.1,
        'AtLocation': 0.9,
        'CoOccurs': 0.8,
    })

    def __post_init__(self):
        """Validate configuration values after initialization."""
        self._validate()

    def _validate(self):
        """
        Validate configuration values are within acceptable ranges.

        Raises:
            ValueError: If any configuration value is invalid.
        """
        # PageRank validation
        if not (0 < self.pagerank_damping < 1):
            raise ValueError(
                f"pagerank_damping must be between 0 and 1, got {self.pagerank_damping}"
            )
        if self.pagerank_iterations < 1:
            raise ValueError(
                f"pagerank_iterations must be at least 1, got {self.pagerank_iterations}"
            )
        if self.pagerank_tolerance <= 0:
            raise ValueError(
                f"pagerank_tolerance must be positive, got {self.pagerank_tolerance}"
            )

        # Clustering validation
        if self.min_cluster_size < 1:
            raise ValueError(
                f"min_cluster_size must be at least 1, got {self.min_cluster_size}"
            )
        if not (0 <= self.cluster_strictness <= 1):
            raise ValueError(
                f"cluster_strictness must be between 0 and 1, got {self.cluster_strictness}"
            )
        if math.isnan(self.louvain_resolution) or math.isinf(self.louvain_resolution):
            raise ValueError(
                f"louvain_resolution must be a finite number, got {self.louvain_resolution}"
            )
        if self.louvain_resolution <= 0:
            raise ValueError(
                f"louvain_resolution must be positive, got {self.louvain_resolution}"
            )
        if self.louvain_resolution > 20:
            import warnings
            warnings.warn(
                f"louvain_resolution={self.louvain_resolution} is very high. "
                f"This may produce hundreds of tiny clusters. "
                f"Typical range is 1.0-10.0."
            )

        # Threshold validation
        if self.isolation_threshold < 0:
            raise ValueError(
                f"isolation_threshold must be non-negative, got {self.isolation_threshold}"
            )
        if self.well_connected_threshold < 0:
            raise ValueError(
                f"well_connected_threshold must be non-negative, got {self.well_connected_threshold}"
            )
        if self.weak_topic_tfidf_threshold < 0:
            raise ValueError(
                f"weak_topic_tfidf_threshold must be non-negative, got {self.weak_topic_tfidf_threshold}"
            )

        # Chunking validation
        if self.chunk_size < 1:
            raise ValueError(
                f"chunk_size must be at least 1, got {self.chunk_size}"
            )
        if self.chunk_overlap < 0:
            raise ValueError(
                f"chunk_overlap must be non-negative, got {self.chunk_overlap}"
            )
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than chunk_size ({self.chunk_size})"
            )

        # Query expansion validation
        if self.max_query_expansions < 0:
            raise ValueError(
                f"max_query_expansions must be non-negative, got {self.max_query_expansions}"
            )
        if not (0 <= self.semantic_expansion_discount <= 1):
            raise ValueError(
                f"semantic_expansion_discount must be between 0 and 1, got {self.semantic_expansion_discount}"
            )

        # Cross-layer damping validation
        if not (0 < self.cross_layer_damping < 1):
            raise ValueError(
                f"cross_layer_damping must be between 0 and 1, got {self.cross_layer_damping}"
            )

        # BM25 validation
        if self.scoring_algorithm not in ('tfidf', 'bm25'):
            raise ValueError(
                f"scoring_algorithm must be 'tfidf' or 'bm25', got {self.scoring_algorithm}"
            )
        if not (0 <= self.bm25_k1 <= 3):
            raise ValueError(
                f"bm25_k1 must be between 0 and 3, got {self.bm25_k1}"
            )
        if not (0 <= self.bm25_b <= 1):
            raise ValueError(
                f"bm25_b must be between 0 and 1, got {self.bm25_b}"
            )

    def copy(self) -> 'CorticalConfig':
        """
        Create a copy of this configuration.

        Returns:
            A new CorticalConfig instance with the same values.
        """
        return CorticalConfig(
            pagerank_damping=self.pagerank_damping,
            pagerank_iterations=self.pagerank_iterations,
            pagerank_tolerance=self.pagerank_tolerance,
            min_cluster_size=self.min_cluster_size,
            cluster_strictness=self.cluster_strictness,
            louvain_resolution=self.louvain_resolution,
            isolation_threshold=self.isolation_threshold,
            well_connected_threshold=self.well_connected_threshold,
            weak_topic_tfidf_threshold=self.weak_topic_tfidf_threshold,
            bridge_similarity_min=self.bridge_similarity_min,
            bridge_similarity_max=self.bridge_similarity_max,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            max_query_expansions=self.max_query_expansions,
            semantic_expansion_discount=self.semantic_expansion_discount,
            scoring_algorithm=self.scoring_algorithm,
            bm25_k1=self.bm25_k1,
            bm25_b=self.bm25_b,
            cross_layer_damping=self.cross_layer_damping,
            bigram_component_weight=self.bigram_component_weight,
            bigram_chain_weight=self.bigram_chain_weight,
            bigram_cooccurrence_weight=self.bigram_cooccurrence_weight,
            concept_min_shared_docs=self.concept_min_shared_docs,
            concept_min_jaccard=self.concept_min_jaccard,
            concept_embedding_threshold=self.concept_embedding_threshold,
            multihop_max_hops=self.multihop_max_hops,
            multihop_decay_factor=self.multihop_decay_factor,
            multihop_min_path_score=self.multihop_min_path_score,
            inheritance_decay_factor=self.inheritance_decay_factor,
            inheritance_max_depth=self.inheritance_max_depth,
            inheritance_boost_factor=self.inheritance_boost_factor,
            relation_weights=dict(self.relation_weights),
        )

    def to_dict(self) -> Dict:
        """
        Convert configuration to a dictionary for serialization.

        Returns:
            Dictionary representation of the configuration.
        """
        return {
            'pagerank_damping': self.pagerank_damping,
            'pagerank_iterations': self.pagerank_iterations,
            'pagerank_tolerance': self.pagerank_tolerance,
            'min_cluster_size': self.min_cluster_size,
            'cluster_strictness': self.cluster_strictness,
            'louvain_resolution': self.louvain_resolution,
            'isolation_threshold': self.isolation_threshold,
            'well_connected_threshold': self.well_connected_threshold,
            'weak_topic_tfidf_threshold': self.weak_topic_tfidf_threshold,
            'bridge_similarity_min': self.bridge_similarity_min,
            'bridge_similarity_max': self.bridge_similarity_max,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'max_query_expansions': self.max_query_expansions,
            'semantic_expansion_discount': self.semantic_expansion_discount,
            'scoring_algorithm': self.scoring_algorithm,
            'bm25_k1': self.bm25_k1,
            'bm25_b': self.bm25_b,
            'cross_layer_damping': self.cross_layer_damping,
            'bigram_component_weight': self.bigram_component_weight,
            'bigram_chain_weight': self.bigram_chain_weight,
            'bigram_cooccurrence_weight': self.bigram_cooccurrence_weight,
            'concept_min_shared_docs': self.concept_min_shared_docs,
            'concept_min_jaccard': self.concept_min_jaccard,
            'concept_embedding_threshold': self.concept_embedding_threshold,
            'multihop_max_hops': self.multihop_max_hops,
            'multihop_decay_factor': self.multihop_decay_factor,
            'multihop_min_path_score': self.multihop_min_path_score,
            'inheritance_decay_factor': self.inheritance_decay_factor,
            'inheritance_max_depth': self.inheritance_max_depth,
            'inheritance_boost_factor': self.inheritance_boost_factor,
            'relation_weights': dict(self.relation_weights),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'CorticalConfig':
        """
        Create configuration from a dictionary.

        Args:
            data: Dictionary with configuration values.

        Returns:
            CorticalConfig instance.
        """
        return cls(**data)


# Valid relation chains for multi-hop inference
# Maps (relation1, relation2) -> validity score (0.0 to 1.0)
# Higher scores indicate more semantically valid inference chains
VALID_RELATION_CHAINS: Dict[Tuple[str, str], float] = {
    # Transitive hierarchies
    ('IsA', 'IsA'): 1.0,           # dog IsA animal IsA living_thing
    ('PartOf', 'PartOf'): 1.0,     # wheel PartOf car PartOf vehicle
    ('IsA', 'HasProperty'): 0.9,   # dog IsA animal HasProperty alive
    ('PartOf', 'HasProperty'): 0.8,  # wheel PartOf car HasProperty fast

    # Association chains
    ('RelatedTo', 'RelatedTo'): 0.6,
    ('SimilarTo', 'SimilarTo'): 0.7,
    ('CoOccurs', 'CoOccurs'): 0.5,
    ('RelatedTo', 'IsA'): 0.7,
    ('RelatedTo', 'SimilarTo'): 0.7,

    # Causal chains
    ('Causes', 'Causes'): 0.8,
    ('Causes', 'HasProperty'): 0.7,

    # Derivation chains
    ('DerivedFrom', 'DerivedFrom'): 0.8,
    ('DerivedFrom', 'IsA'): 0.7,

    # Invalid/contradictory chains (low scores)
    ('Antonym', 'IsA'): 0.1,       # Contradictory: opposite → type
    ('Antonym', 'Antonym'): 0.4,   # Double negation
}

# Default validity score for unlisted relation chain pairs
DEFAULT_CHAIN_VALIDITY: float = 0.4


def get_default_config() -> CorticalConfig:
    """
    Get a new instance of the default configuration.

    Returns:
        CorticalConfig with default values.
    """
    return CorticalConfig()
