"""
Spark API: First-blitz priming and alignment integration.

This module integrates SparkSLM (statistical first-blitz predictor) with
the processor for fast query priming and human-AI alignment acceleration.

SparkSLM is optional - enable with enable_spark() or spark=True in constructor.
"""

import logging
import os
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


class SparkMixin:
    """
    Mixin providing SparkSLM integration for first-blitz priming.

    Requires CoreMixin to be present (provides tokenizer, documents).

    Example:
        >>> processor = CorticalTextProcessor(spark=True)
        >>> processor.train_spark()  # Train on corpus
        >>> processor.load_alignment("samples/alignment")  # Load definitions
        >>>
        >>> # Get first-blitz hints for a query
        >>> hints = processor.prime_query("neural network training")
        >>> print(hints['predictions'])  # Likely next tokens
        >>> print(hints['alignment'])    # Relevant definitions/patterns
    """

    # Staleness tracking for spark model
    COMP_SPARK = 'spark'

    def __init__(self, *args, spark: bool = False, **kwargs):
        """
        Initialize with optional Spark support.

        Args:
            spark: If True, initialize SparkPredictor for first-blitz priming.
                  Can also call enable_spark() later.
        """
        super().__init__(*args, **kwargs)
        self._spark = None
        self._spark_enabled = False
        self._alignment_loaded = False

        if spark:
            self.enable_spark()

    def enable_spark(self, ngram_order: int = 3) -> None:
        """
        Enable SparkSLM for first-blitz query priming.

        Creates a SparkPredictor instance that can be trained on the corpus
        and used to provide fast statistical predictions.

        Args:
            ngram_order: N-gram order (default 3 for trigrams)

        Example:
            >>> processor = CorticalTextProcessor()
            >>> processor.enable_spark(ngram_order=4)  # Use 4-grams
            >>> processor.train_spark()
        """
        from ..spark import SparkPredictor

        self._spark = SparkPredictor(ngram_order=ngram_order)
        self._spark_enabled = True
        logger.info(f"SparkSLM enabled with {ngram_order}-gram model")

    def disable_spark(self) -> None:
        """Disable SparkSLM and free memory."""
        self._spark = None
        self._spark_enabled = False
        self._alignment_loaded = False
        logger.info("SparkSLM disabled")

    @property
    def spark_enabled(self) -> bool:
        """Check if SparkSLM is enabled."""
        return self._spark_enabled and self._spark is not None

    def train_spark(self, min_doc_length: int = 10) -> Dict[str, Any]:
        """
        Train SparkSLM on the current corpus.

        Trains the n-gram model on all documents in the processor,
        allowing it to predict likely next tokens based on context.

        Args:
            min_doc_length: Minimum document length (chars) to include

        Returns:
            Training statistics including document count and token count

        Raises:
            RuntimeError: If SparkSLM is not enabled

        Example:
            >>> processor.process_document("doc1", "Neural networks are...")
            >>> processor.enable_spark()
            >>> stats = processor.train_spark()
            >>> print(f"Trained on {stats['documents']} documents")
        """
        if not self.spark_enabled:
            raise RuntimeError("SparkSLM not enabled. Call enable_spark() first.")

        # Filter documents by length - extract just the texts
        texts = [
            text
            for doc_id, text in self.documents.items()
            if len(text) >= min_doc_length
        ]

        if not texts:
            logger.warning("No documents to train on")
            return {'documents': 0, 'tokens': 0}

        # Train on documents
        self._spark.train_from_documents(texts)

        # Mark as fresh
        self._mark_fresh(self.COMP_SPARK)

        # Build stats
        stats = {
            'documents': len(texts),
            'tokens': len(self._spark.ngram.vocab),
        }

        logger.info(f"Spark trained on {stats['documents']} docs, "
                   f"{stats['tokens']} tokens")

        return stats

    def load_alignment(self, path: str) -> int:
        """
        Load alignment context from a directory or markdown file.

        Alignment context includes:
        - Definitions: "When I say X, I mean Y"
        - Patterns: "In this codebase, we do X this way"
        - Preferences: "I prefer X over Y"
        - Goals: Current objectives

        Args:
            path: Path to alignment directory or markdown file.
                  If directory, loads all .md files within.

        Returns:
            Number of entries loaded

        Raises:
            RuntimeError: If SparkSLM is not enabled

        Example:
            >>> processor.enable_spark()
            >>> count = processor.load_alignment("samples/alignment")
            >>> print(f"Loaded {count} alignment entries")
        """
        if not self.spark_enabled:
            raise RuntimeError("SparkSLM not enabled. Call enable_spark() first.")

        # load_alignment returns self for chaining, count entries afterward
        before_count = len(self._spark.alignment)
        self._spark.load_alignment(path)
        count = len(self._spark.alignment) - before_count
        self._alignment_loaded = len(self._spark.alignment) > 0

        logger.info(f"Loaded {count} alignment entries from {path}")
        return count

    def prime_query(self, query: str) -> Dict[str, Any]:
        """
        Get first-blitz hints for a query.

        Returns fast statistical predictions and relevant alignment context
        that can prime deeper semantic analysis.

        Args:
            query: The query text to prime

        Returns:
            Dict containing:
            - query: The original query
            - keywords: Key terms extracted from query
            - completions: List of (token, probability) tuples for predicted next words
            - alignment: List of relevant definitions/patterns
            - topics: Suggested topic classifications
            - is_trained: Whether the model has been trained

        Raises:
            RuntimeError: If SparkSLM is not enabled

        Example:
            >>> hints = processor.prime_query("neural network")
            >>> for token, prob in hints['completions']:
            ...     print(f"  {token}: {prob:.3f}")
        """
        if not self.spark_enabled:
            raise RuntimeError("SparkSLM not enabled. Call enable_spark() first.")

        return self._spark.prime(query)

    def complete_query(self, query: str, length: int = 3) -> str:
        """
        Generate possible query completions.

        Uses the n-gram model to predict likely continuations of the query.

        Args:
            query: Query prefix to complete
            length: Number of words to add (default 3)

        Returns:
            Completed query string

        Raises:
            RuntimeError: If SparkSLM is not enabled

        Example:
            >>> completed = processor.complete_query("how do I")
            >>> print(completed)  # "how do I process documents"
        """
        if not self.spark_enabled:
            raise RuntimeError("SparkSLM not enabled. Call enable_spark() first.")

        return self._spark.complete_sequence(query, length=length)

    def expand_query_with_spark(
        self,
        query: str,
        max_expansions: Optional[int] = None,
        use_code_concepts: bool = False,
        spark_boost: float = 0.3
    ) -> Dict[str, float]:
        """
        Expand query using both corpus statistics and spark priming.

        Combines traditional query expansion with spark's first-blitz
        predictions to surface statistically likely terms.

        Args:
            query: Query text to expand
            max_expansions: Maximum expansion terms (default from config)
            use_code_concepts: Include programming synonyms
            spark_boost: Weight for spark predictions (0.0-1.0)

        Returns:
            Dict mapping terms to weights, with spark predictions boosted

        Example:
            >>> expanded = processor.expand_query_with_spark("neural network")
            >>> # Includes both lateral expansions and spark predictions
        """
        # Get base expansion from standard method
        if max_expansions is None:
            max_expansions = self.config.max_query_expansions

        expanded = self.expand_query(
            query,
            max_expansions=max_expansions,
            use_code_concepts=use_code_concepts
        )

        # Add spark predictions if enabled
        if self.spark_enabled:
            try:
                hints = self.prime_query(query)

                # Use completions from prime result
                for token, prob in hints.get('completions', []):
                    # Boost if already in expansion, add if not
                    if token in expanded:
                        expanded[token] += prob * spark_boost
                    elif prob > 0.05:  # Only add confident predictions
                        expanded[token] = prob * spark_boost

                # Also add alignment-related terms
                for entry in hints.get('alignment', []):
                    key = entry.get('key', '')
                    if key and key not in expanded:
                        expanded[key] = 0.1 * spark_boost

            except Exception as e:
                logger.warning(f"Spark priming failed: {e}")

        return expanded

    def get_alignment_context(self, term: str) -> List[Dict[str, Any]]:
        """
        Look up alignment context for a term.

        Returns definitions, patterns, and preferences related to the term.

        Args:
            term: Term to look up

        Returns:
            List of alignment entries with type, key, and value

        Raises:
            RuntimeError: If SparkSLM is not enabled

        Example:
            >>> context = processor.get_alignment_context("spark")
            >>> for entry in context:
            ...     print(f"{entry['type']}: {entry['value']}")
        """
        if not self.spark_enabled:
            raise RuntimeError("SparkSLM not enabled. Call enable_spark() first.")

        # Get all entries from the alignment index (not just first one)
        entries = self._spark.alignment.lookup(term)
        return [
            {
                'key': e.key,
                'value': e.value,
                'type': e.entry_type,
                'source': e.source,
            }
            for e in entries
        ]

    def get_alignment_summary(self) -> str:
        """
        Get a summary of loaded alignment context.

        Useful for understanding what definitions/patterns are loaded.

        Returns:
            Human-readable summary of alignment entries

        Raises:
            RuntimeError: If SparkSLM is not enabled
        """
        if not self.spark_enabled:
            raise RuntimeError("SparkSLM not enabled. Call enable_spark() first.")

        return self._spark.alignment.get_context_summary()

    def save_spark(self, path: str) -> None:
        """
        Save SparkSLM state to disk.

        Saves both the n-gram model and alignment index.

        Args:
            path: Directory path to save to

        Raises:
            RuntimeError: If SparkSLM is not enabled
        """
        if not self.spark_enabled:
            raise RuntimeError("SparkSLM not enabled. Call enable_spark() first.")

        os.makedirs(path, exist_ok=True)
        self._spark.save(path)
        logger.info(f"Spark state saved to {path}")

    def load_spark(self, path: str) -> None:
        """
        Load SparkSLM state from disk.

        Loads both the n-gram model and alignment index.

        Args:
            path: Directory path to load from
        """
        from ..spark import SparkPredictor

        # Use classmethod to load saved state
        self._spark = SparkPredictor.load(path)
        self._spark_enabled = True
        self._alignment_loaded = len(self._spark.alignment) > 0
        self._mark_fresh(self.COMP_SPARK)
        logger.info(f"Spark state loaded from {path}")

    def get_spark_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the SparkSLM model.

        Returns:
            Dict with model statistics including vocabulary size,
            n-gram counts, and alignment entry counts

        Raises:
            RuntimeError: If SparkSLM is not enabled
        """
        if not self.spark_enabled:
            raise RuntimeError("SparkSLM not enabled. Call enable_spark() first.")

        stats = {
            'enabled': True,
            'ngram_order': self._spark.ngram.n,
            'vocabulary_size': len(self._spark.ngram.vocab),
            'alignment_entries': len(self._spark.alignment),
            'alignment_loaded': self._alignment_loaded,
        }

        # Add n-gram specific stats
        ngram = self._spark.ngram
        stats['context_count'] = len(ngram.counts)

        # Add alignment breakdown
        alignment = self._spark.alignment
        stats['alignment_breakdown'] = {
            'definitions': len(alignment.get_all_definitions()),
            'patterns': len(alignment.get_all_patterns()),
            'preferences': len(alignment.get_all_preferences()),
            'goals': len(alignment.get_current_goals()),
        }

        return stats
