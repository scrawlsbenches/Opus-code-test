"""
Audit Module - Codebase Quality Analysis Services.

Registers audit-related services in the container, making them injectable
for CLI commands and other components.

Services Provided:
    - SuspiciousCommentFilter (Bloom filter for pre-screening)
    - CommentClassifier (Naive Bayes classifier)
    - SimilarCommentFinder (LSH index for similarity)
    - AuditInvertedIndex (Full-text search)
    - CommentMarkerTrie (Marker lookup)
    - PatternFrequencySketch (Count-min sketch for frequencies)

Usage:
    from cortical.core.modules import AuditModule

    container = Container()
    container.apply_module(AuditModule())

    classifier = container.resolve(CommentClassifier)

Why This Module Exists:
    The audit tools need various algorithm implementations that can be
    expensive to initialize (LSH indexes, Bloom filters, trained models).

    By registering them in the container:
    1. Shared instances across commands (no re-initialization)
    2. Tests can inject mock implementations
    3. Configuration (thresholds, sizes) can be centralized
    4. Lazy initialization until first use

Configuration:
    The module accepts optional configuration for tuning algorithm parameters:
    - bloom_filter_size: Expected number of patterns
    - bloom_filter_fp_rate: False positive rate
    - lsh_num_hashes: Number of hash functions for LSH
    - lsh_num_bands: Number of bands for LSH

TODO(migration): Add model path configuration for loading trained models
"""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from cortical.common import Container, ContainerModule


@dataclass
class AuditConfig:
    """Configuration for audit services."""
    # Bloom filter settings
    bloom_filter_size: int = 100
    bloom_filter_fp_rate: float = 0.01

    # LSH settings
    lsh_num_hashes: int = 100
    lsh_num_bands: int = 20

    # Model paths
    model_dir: Path = Path(".audit_models")

    # Count-min sketch settings
    sketch_width: int = 1000
    sketch_depth: int = 5


class AuditModule(ContainerModule):
    """
    Container module for Audit services.

    Registers algorithm implementations and audit utilities
    in the container for dependency injection.
    """

    def __init__(self, config: Optional[AuditConfig] = None):
        """
        Initialize the audit module.

        Args:
            config: Optional configuration for audit services.
                    Uses defaults if not provided.
        """
        self.config = config or AuditConfig()

    def register(self, container: Container) -> None:
        """Register Audit services with the container."""
        from cortical.audits.algorithms import (
            SuspiciousCommentFilter,
            CommentClassifier,
            SimilarCommentFinder,
            AuditInvertedIndex,
            CommentMarkerTrie,
            PatternFrequencySketch,
            CommentPatternFinder,
        )
        from cortical.audits.patterns import SUSPICIOUS_PATTERNS, COMMENT_MARKERS

        # Register configuration
        container.register_instance(AuditConfig, self.config)

        # Register Bloom filter (initialized with suspicious patterns)
        bloom_filter = SuspiciousCommentFilter(
            expected_patterns=self.config.bloom_filter_size,
            fp_rate=self.config.bloom_filter_fp_rate,
        )
        for pattern in SUSPICIOUS_PATTERNS:
            bloom_filter.add(pattern.lower())
        container.register_instance(SuspiciousCommentFilter, bloom_filter)

        # Register Trie (initialized with comment markers)
        trie = CommentMarkerTrie()
        for marker in COMMENT_MARKERS:
            trie.insert(marker.lower())
        container.register_instance(CommentMarkerTrie, trie)

        # Register other services as factories (lazy initialization)
        # These are created on-demand since they may not always be needed

        def create_classifier():
            return CommentClassifier()

        def create_lsh_finder():
            return SimilarCommentFinder(
                num_hashes=self.config.lsh_num_hashes,
                num_bands=self.config.lsh_num_bands,
            )

        def create_inverted_index():
            return AuditInvertedIndex()

        def create_frequency_sketch():
            return PatternFrequencySketch(
                width=self.config.sketch_width,
                depth=self.config.sketch_depth,
            )

        # Register factories
        # TODO(migration): Consider using register_factory when available
        container.register(CommentClassifier, create_classifier)
        container.register(SimilarCommentFinder, create_lsh_finder)
        container.register(AuditInvertedIndex, create_inverted_index)
        container.register(PatternFrequencySketch, create_frequency_sketch)

        # Register reasoning services
        from cortical.audits.persistence import (
            PersistenceBackend,
            FilePersistenceBackend,
        )
        from cortical.audits.reasoning import AuditReasoner
        from cortical.audits.health import CodebaseAnalyzer
        from cortical.audits.discovery import (
            WovenMindDiscovery,
            DiscoveryConfig,
            FileDiscoveryPersistence,
        )
        from cortical.common.filesystem import RealFileSystem

        # Persistence backend (default to file-based)
        def create_persistence():
            return FilePersistenceBackend(RealFileSystem())

        container.register(PersistenceBackend, create_persistence)

        # Audit reasoner
        def create_reasoner():
            persistence = container.resolve(PersistenceBackend)
            return AuditReasoner(persistence=persistence)

        container.register(AuditReasoner, create_reasoner)

        # Health analyzer
        def create_health_analyzer():
            return CodebaseAnalyzer(RealFileSystem())

        container.register(CodebaseAnalyzer, create_health_analyzer)

        # Discovery system
        def create_discovery():
            return WovenMindDiscovery(
                config=DiscoveryConfig(),
                persistence=FileDiscoveryPersistence(),
            )

        container.register(WovenMindDiscovery, create_discovery)
