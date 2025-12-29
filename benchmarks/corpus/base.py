"""
Base classes for Corpus benchmarks.

Provides infrastructure for benchmarking CorticalTextProcessor performance:
- Corpus caching to avoid repeated loading
- Synthetic corpus generation with controllable parameters
- Corpus-specific benchmark categories
"""

from __future__ import annotations

import hashlib
import random
import sys
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path for imports
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.woven_mind.base import (
    BaseBenchmark,
    BenchmarkCategory,
    BenchmarkMetric,
    BenchmarkResult,
    BenchmarkStatus,
)


class CorpusBenchmarkCategory(Enum):
    """Benchmark categories specific to corpus operations."""

    INDEXING = "indexing"         # Document processing throughput
    QUERY = "query"               # Search latency and relevance
    PASSAGE = "passage"           # RAG/passage retrieval
    ANALYSIS = "analysis"         # PageRank, TF-IDF, clustering
    CODE_SEARCH = "code_search"   # Code-specific search features
    FINGERPRINT = "fingerprint"   # Semantic fingerprinting
    PERSISTENCE = "persistence"   # Save/load operations


# Map corpus categories to base BenchmarkCategory for compatibility
CATEGORY_TO_BASE = {
    CorpusBenchmarkCategory.INDEXING: BenchmarkCategory.SCALE,
    CorpusBenchmarkCategory.QUERY: BenchmarkCategory.QUALITY,
    CorpusBenchmarkCategory.PASSAGE: BenchmarkCategory.QUALITY,
    CorpusBenchmarkCategory.ANALYSIS: BenchmarkCategory.REGRESSION,
    CorpusBenchmarkCategory.CODE_SEARCH: BenchmarkCategory.QUALITY,
    CorpusBenchmarkCategory.FINGERPRINT: BenchmarkCategory.STABILITY,
    CorpusBenchmarkCategory.PERSISTENCE: BenchmarkCategory.SCALE,
}


@dataclass
class SyntheticCorpusConfig:
    """Configuration for synthetic corpus generation."""

    n_docs: int = 100
    doc_length: int = 100  # Average words per document
    vocab_size: int = 1000
    concept_count: int = 20  # Number of distinct concepts
    concept_frequency: float = 0.3  # How often to inject concepts
    code_style: bool = False  # Include code-like identifiers
    seed: Optional[int] = None  # For reproducibility

    def cache_key(self) -> str:
        """Generate cache key for this configuration."""
        key_parts = [
            f"n={self.n_docs}",
            f"len={self.doc_length}",
            f"vocab={self.vocab_size}",
            f"concepts={self.concept_count}",
            f"freq={self.concept_frequency:.2f}",
            f"code={self.code_style}",
        ]
        if self.seed is not None:
            key_parts.append(f"seed={self.seed}")
        key_str = ",".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()[:12]


class SyntheticCorpusGenerator:
    """
    Generate synthetic corpora for benchmarking.

    Features:
    - Controllable vocabulary size
    - Concept injection with configurable frequency
    - Code-style identifiers (camelCase, snake_case)
    - Reproducible with seed
    """

    # Common patterns for concept injection
    CONCEPT_TEMPLATES = [
        "{concept} is important",
        "the {concept} approach works",
        "using {concept} effectively",
        "{concept} and {related} together",
        "consider {concept} when",
    ]

    # Code-style identifiers
    CODE_PATTERNS = [
        "get{word}",
        "set{word}",
        "is{word}Valid",
        "{word}_handler",
        "{word}_config",
        "process_{word}",
        "{word}Manager",
    ]

    def __init__(self, config: Optional[SyntheticCorpusConfig] = None):
        self.config = config or SyntheticCorpusConfig()
        self._rng = random.Random(self.config.seed)
        self._vocab: List[str] = []
        self._concepts: List[str] = []
        self._build_vocab()

    def _build_vocab(self) -> None:
        """Build vocabulary and concept lists."""
        # Base vocabulary
        self._vocab = [f"word{i}" for i in range(self.config.vocab_size)]

        # Add code-style vocab if requested
        if self.config.code_style:
            code_words = []
            for i in range(min(100, self.config.vocab_size // 5)):
                base = f"item{i}"
                for pattern in self.CODE_PATTERNS:
                    code_words.append(pattern.format(word=base.title()))
            self._vocab.extend(code_words)

        # Generate concepts (higher-level terms)
        self._concepts = [f"concept{i}" for i in range(self.config.concept_count)]

    def _generate_document(self) -> str:
        """Generate a single document."""
        words = []
        remaining = self.config.doc_length

        # Maybe inject concept patterns
        if self._rng.random() < self.config.concept_frequency:
            concept = self._rng.choice(self._concepts)
            related = self._rng.choice(self._concepts)
            template = self._rng.choice(self.CONCEPT_TEMPLATES)
            pattern_words = template.format(concept=concept, related=related).split()
            words.extend(pattern_words)
            remaining -= len(pattern_words)

        # Fill with random vocabulary
        if remaining > 0:
            words.extend(self._rng.choices(self._vocab, k=remaining))

        # Shuffle to avoid obvious patterns
        self._rng.shuffle(words)
        return " ".join(words)

    def generate(self) -> Dict[str, str]:
        """
        Generate a synthetic corpus.

        Returns:
            Dictionary mapping doc_id to document text
        """
        corpus = {}
        for i in range(self.config.n_docs):
            doc_id = f"synthetic_doc_{i:04d}"
            corpus[doc_id] = self._generate_document()
        return corpus

    def generate_with_queries(
        self,
        n_queries: int = 10,
    ) -> Tuple[Dict[str, str], List[str]]:
        """
        Generate corpus with matching queries.

        Returns:
            Tuple of (corpus, queries) where queries should match documents
        """
        corpus = self.generate()

        # Generate queries from concepts (should find matches)
        queries = []
        for _ in range(n_queries):
            concept = self._rng.choice(self._concepts)
            template = self._rng.choice(["find {}", "{} usage", "about {}"])
            queries.append(template.format(concept))

        return corpus, queries


class CorpusCache:
    """
    Cache for loaded processors to avoid repeated computation.

    Caches both raw corpora and fully-computed processors.
    Uses configuration hash for cache key.
    """

    _instance: Optional["CorpusCache"] = None

    def __new__(cls) -> "CorpusCache":
        """Singleton pattern for global cache."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._corpus_cache = {}
            cls._instance._processor_cache = {}
            cls._instance._stats = {"hits": 0, "misses": 0}
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the cache (useful for testing)."""
        if cls._instance is not None:
            cls._instance._corpus_cache.clear()
            cls._instance._processor_cache.clear()
            cls._instance._stats = {"hits": 0, "misses": 0}

    def get_corpus(
        self,
        config: SyntheticCorpusConfig,
    ) -> Optional[Dict[str, str]]:
        """Get cached corpus or None."""
        key = config.cache_key()
        if key in self._corpus_cache:
            self._stats["hits"] += 1
            return self._corpus_cache[key]
        self._stats["misses"] += 1
        return None

    def set_corpus(
        self,
        config: SyntheticCorpusConfig,
        corpus: Dict[str, str],
    ) -> None:
        """Cache a corpus."""
        key = config.cache_key()
        self._corpus_cache[key] = corpus

    def get_processor(
        self,
        config: SyntheticCorpusConfig,
        computed: bool = True,
    ) -> Optional[Any]:
        """
        Get cached processor or None.

        Args:
            config: Corpus configuration
            computed: Whether processor should have compute_all() called
        """
        key = f"{config.cache_key()}_computed={computed}"
        if key in self._processor_cache:
            self._stats["hits"] += 1
            return self._processor_cache[key]
        self._stats["misses"] += 1
        return None

    def set_processor(
        self,
        config: SyntheticCorpusConfig,
        processor: Any,
        computed: bool = True,
    ) -> None:
        """Cache a processor."""
        key = f"{config.cache_key()}_computed={computed}"
        self._processor_cache[key] = processor

    def get_or_create_processor(
        self,
        config: SyntheticCorpusConfig,
        computed: bool = True,
    ) -> Any:
        """
        Get cached processor or create one.

        This is the main entry point for benchmarks.
        """
        from cortical.processor import CorticalTextProcessor

        # Check cache first
        cached = self.get_processor(config, computed)
        if cached is not None:
            return cached

        # Create processor
        processor = CorticalTextProcessor()

        # Get or generate corpus
        corpus = self.get_corpus(config)
        if corpus is None:
            generator = SyntheticCorpusGenerator(config)
            corpus = generator.generate()
            self.set_corpus(config, corpus)

        # Process documents
        for doc_id, text in corpus.items():
            processor.process_document(doc_id, text)

        # Compute if requested
        if computed:
            processor.compute_all()

        # Cache and return
        self.set_processor(config, processor, computed)
        return processor

    def stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            **self._stats,
            "corpus_count": len(self._corpus_cache),
            "processor_count": len(self._processor_cache),
        }


class CorpusBenchmark(BaseBenchmark):
    """
    Base class for corpus benchmarks.

    Extends BaseBenchmark with:
    - Corpus caching integration
    - Synthetic corpus generation
    - Quick mode support
    """

    name: str = "unnamed_corpus_benchmark"
    description: str = "No description provided"
    corpus_category: CorpusBenchmarkCategory = CorpusBenchmarkCategory.QUERY

    # Default corpus configuration
    default_config = SyntheticCorpusConfig(
        n_docs=100,
        doc_length=100,
        vocab_size=1000,
        concept_count=20,
    )

    # Quick mode configuration (smaller corpus)
    quick_config = SyntheticCorpusConfig(
        n_docs=25,
        doc_length=50,
        vocab_size=500,
        concept_count=10,
    )

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._cache = CorpusCache()
        self._processor = None
        self._corpus = None

        # Determine which corpus config to use
        is_quick = self.config.get("quick", False)
        if is_quick:
            self._corpus_config = self.quick_config
        else:
            # Allow config override
            self._corpus_config = SyntheticCorpusConfig(
                n_docs=self.config.get("n_docs", self.default_config.n_docs),
                doc_length=self.config.get("doc_length", self.default_config.doc_length),
                vocab_size=self.config.get("vocab_size", self.default_config.vocab_size),
                concept_count=self.config.get("concept_count", self.default_config.concept_count),
                concept_frequency=self.config.get("concept_frequency", self.default_config.concept_frequency),
                code_style=self.config.get("code_style", self.default_config.code_style),
                seed=self.config.get("seed", self.default_config.seed),
            )

    @property
    def category(self) -> BenchmarkCategory:
        """Map corpus category to base category."""
        return CATEGORY_TO_BASE.get(self.corpus_category, BenchmarkCategory.REGRESSION)

    def setup(self) -> None:
        """Load or create processor from cache."""
        # Check if a pre-loaded processor was provided (--use-corpus)
        loaded_processor = self.config.get("_loaded_processor")
        if loaded_processor is not None:
            self._processor = loaded_processor
        else:
            self._processor = self._cache.get_or_create_processor(
                self._corpus_config,
                computed=True,
            )

    @abstractmethod
    def run(self) -> BenchmarkResult:
        """Execute the benchmark."""
        pass

    def execute(self) -> BenchmarkResult:
        """
        Full execution lifecycle with SKIPPED status preservation.

        Overrides base execute() to not override SKIPPED status.
        """
        result = BenchmarkResult(
            benchmark_name=self.name,
            category=self.category,
            status=BenchmarkStatus.RUNNING,
        )

        start_time = time.perf_counter()

        try:
            self.setup()
            result = self.run()
            # Don't override SKIPPED status
            if result.status != BenchmarkStatus.SKIPPED:
                result.status = (
                    BenchmarkStatus.PASSED if result.is_passing
                    else BenchmarkStatus.FAILED
                )
        except Exception as e:
            result.status = BenchmarkStatus.ERROR
            result.error_message = str(e)
        finally:
            try:
                self.teardown()
            except Exception:
                pass

            result.duration_ms = (time.perf_counter() - start_time) * 1000

        self._result = result
        return result

    def create_result(self) -> BenchmarkResult:
        """Create a result object for this benchmark."""
        return BenchmarkResult(
            benchmark_name=self.name,
            category=self.category,
            status=BenchmarkStatus.RUNNING,
            metadata={
                "corpus_config": {
                    "n_docs": self._corpus_config.n_docs,
                    "doc_length": self._corpus_config.doc_length,
                    "vocab_size": self._corpus_config.vocab_size,
                },
            },
        )


# Utility functions

def measure_throughput(
    func,
    n_iterations: int = 10,
) -> Tuple[float, float]:
    """
    Measure throughput of a function.

    Returns:
        Tuple of (ops_per_second, std_dev)
    """
    import statistics

    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        func()
        elapsed = time.perf_counter() - start
        if elapsed > 0:
            times.append(1.0 / elapsed)
        else:
            times.append(float('inf'))

    # Filter out infinities
    times = [t for t in times if t != float('inf')]
    if not times:
        return 0.0, 0.0

    mean = statistics.mean(times)
    std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
    return mean, std_dev


def measure_latency_percentiles(
    func,
    n_iterations: int = 100,
) -> Dict[str, float]:
    """
    Measure latency percentiles.

    Returns:
        Dictionary with p50, p90, p95, p99, mean, max latencies in ms
    """
    times_ms = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        func()
        elapsed = (time.perf_counter() - start) * 1000
        times_ms.append(elapsed)

    times_ms.sort()
    n = len(times_ms)

    def percentile(p: float) -> float:
        idx = int(n * p / 100)
        return times_ms[min(idx, n - 1)]

    return {
        "p50_ms": percentile(50),
        "p90_ms": percentile(90),
        "p95_ms": percentile(95),
        "p99_ms": percentile(99),
        "mean_ms": sum(times_ms) / n,
        "max_ms": max(times_ms),
    }
