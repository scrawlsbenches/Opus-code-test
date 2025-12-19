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
        self._anomaly_detector = None

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

    # =========================================================================
    # Anomaly Detection Methods
    # =========================================================================

    def enable_anomaly_detection(
        self,
        perplexity_threshold: float = 2.0,
        unknown_word_threshold: float = 0.5
    ) -> None:
        """
        Enable anomaly detection for query safety.

        Creates an AnomalyDetector that can identify:
        - Prompt injection attempts
        - Statistically unusual queries
        - Queries with high unknown word ratios

        Args:
            perplexity_threshold: Flag if perplexity > baseline * threshold
            unknown_word_threshold: Flag if unknown words > threshold

        Raises:
            RuntimeError: If SparkSLM not enabled (n-gram model required)

        Example:
            >>> processor = CorticalTextProcessor(spark=True)
            >>> processor.train_spark()
            >>> processor.enable_anomaly_detection()
            >>> processor.calibrate_anomaly_detector(["normal query 1", "normal query 2"])
        """
        if not self.spark_enabled:
            raise RuntimeError("SparkSLM must be enabled first. Call enable_spark().")

        from ..spark import AnomalyDetector

        self._anomaly_detector = AnomalyDetector(
            ngram_model=self._spark.ngram,
            perplexity_threshold=perplexity_threshold,
            unknown_word_threshold=unknown_word_threshold
        )
        logger.info("Anomaly detection enabled")

    def calibrate_anomaly_detector(self, normal_queries: List[str]) -> Dict[str, float]:
        """
        Calibrate anomaly detector with known-normal queries.

        Establishes baseline perplexity statistics for detecting unusual queries.

        Args:
            normal_queries: List of queries known to be normal/safe

        Returns:
            Calibration statistics including baseline perplexity

        Raises:
            RuntimeError: If anomaly detection not enabled

        Example:
            >>> stats = processor.calibrate_anomaly_detector([
            ...     "How do I search for documents?",
            ...     "Show me the PageRank algorithm",
            ...     "Find files related to authentication",
            ... ])
            >>> print(f"Baseline perplexity: {stats['baseline_perplexity']:.2f}")
        """
        if not self._anomaly_detector:
            raise RuntimeError("Anomaly detection not enabled. Call enable_anomaly_detection() first.")

        return self._anomaly_detector.calibrate(normal_queries)

    def check_query_safety(self, query: str) -> Dict[str, Any]:
        """
        Check if a query is safe/normal or potentially anomalous.

        Runs all enabled anomaly detection methods and returns results.

        Args:
            query: Query to check

        Returns:
            Dict with:
            - is_safe: True if query appears normal
            - is_anomalous: True if query appears suspicious
            - confidence: Anomaly confidence (0.0-1.0)
            - reasons: List of reasons if anomalous
            - metrics: Detailed metrics from each check

        Raises:
            RuntimeError: If anomaly detection not enabled

        Example:
            >>> result = processor.check_query_safety("normal search query")
            >>> if result['is_safe']:
            ...     # Process normally
            ...     pass
            >>> else:
            ...     print(f"Suspicious query: {result['reasons']}")
        """
        if not self._anomaly_detector:
            raise RuntimeError("Anomaly detection not enabled. Call enable_anomaly_detection() first.")

        result = self._anomaly_detector.check(query)

        return {
            'is_safe': not result.is_anomalous,
            'is_anomalous': result.is_anomalous,
            'confidence': result.confidence,
            'reasons': result.reasons,
            'metrics': result.metrics,
        }

    def check_queries_safety(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Check multiple queries for safety.

        Args:
            queries: List of queries to check

        Returns:
            List of safety check results

        Raises:
            RuntimeError: If anomaly detection not enabled
        """
        return [self.check_query_safety(q) for q in queries]

    def is_query_safe(self, query: str) -> bool:
        """
        Quick check if query is safe.

        Convenience method that returns just True/False.

        Args:
            query: Query to check

        Returns:
            True if query appears safe, False if anomalous

        Raises:
            RuntimeError: If anomaly detection not enabled

        Example:
            >>> if processor.is_query_safe(user_query):
            ...     results = processor.find_documents_for_query(user_query)
            ... else:
            ...     print("Query flagged as potentially unsafe")
        """
        result = self.check_query_safety(query)
        return result['is_safe']

    def get_anomaly_stats(self) -> Dict[str, Any]:
        """
        Get anomaly detector statistics.

        Returns:
            Dict with detector configuration and calibration state

        Raises:
            RuntimeError: If anomaly detection not enabled
        """
        if not self._anomaly_detector:
            raise RuntimeError("Anomaly detection not enabled.")

        return self._anomaly_detector.get_stats()

    def add_injection_pattern(self, pattern: str) -> None:
        """
        Add a custom injection pattern to detect.

        Args:
            pattern: Regex pattern to match against queries

        Raises:
            RuntimeError: If anomaly detection not enabled

        Example:
            >>> processor.add_injection_pattern(r'\\bmalicious_keyword\\b')
        """
        if not self._anomaly_detector:
            raise RuntimeError("Anomaly detection not enabled.")

        self._anomaly_detector.add_injection_pattern(pattern)
        logger.info(f"Added custom injection pattern")

    @property
    def anomaly_detection_enabled(self) -> bool:
        """Check if anomaly detection is enabled."""
        return self._anomaly_detector is not None

    # =========================================================================
    # Sample Suggestion Methods (Phase 4: Self-Documentation)
    # =========================================================================

    def enable_suggester(
        self,
        known_terms: Optional[set] = None,
        min_frequency: int = 3,
        min_confidence: float = 0.5
    ) -> None:
        """
        Enable sample suggestion for alignment improvement.

        Creates a SampleSuggester that observes interactions and suggests
        new alignment entries (definitions, patterns, preferences).

        Args:
            known_terms: Set of already-defined terms to skip
            min_frequency: Minimum occurrences before suggesting
            min_confidence: Minimum confidence for suggestions

        Example:
            >>> processor = CorticalTextProcessor(spark=True)
            >>> processor.enable_suggester()
            >>> # Now queries will be observed for pattern detection
        """
        from ..spark import SampleSuggester

        # Load known terms from alignment if available
        if known_terms is None and self.spark_enabled:
            known_terms = set()
            for entry in self._spark.alignment.get_all_definitions():
                known_terms.add(entry.key.lower())

        self._suggester = SampleSuggester(
            known_terms=known_terms,
            min_frequency=min_frequency,
            min_confidence=min_confidence
        )
        logger.info("Sample suggester enabled")

    def observe_query_for_suggestions(
        self,
        query: str,
        success: bool = True,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a query for suggestion analysis.

        Call this after each query to build up patterns for suggestions.

        Args:
            query: The query text
            success: Whether the query was successful
            context: Additional context (e.g., result count)

        Raises:
            RuntimeError: If suggester not enabled

        Example:
            >>> results = processor.find_documents_for_query("neural network")
            >>> processor.observe_query_for_suggestions(
            ...     "neural network",
            ...     success=len(results) > 0,
            ...     context={'result_count': len(results)}
            ... )
        """
        if not hasattr(self, '_suggester') or self._suggester is None:
            raise RuntimeError("Suggester not enabled. Call enable_suggester() first.")

        self._suggester.observe_query(query, success=success, context=context)

    def observe_choice_for_suggestions(
        self,
        choice_type: str,
        chosen: str,
        alternatives: List[str]
    ) -> None:
        """
        Record a choice for preference detection.

        Args:
            choice_type: Type of choice (e.g., "naming", "approach")
            chosen: What was chosen
            alternatives: What could have been chosen

        Raises:
            RuntimeError: If suggester not enabled

        Example:
            >>> processor.observe_choice_for_suggestions(
            ...     choice_type="naming_convention",
            ...     chosen="camelCase",
            ...     alternatives=["snake_case", "kebab-case"]
            ... )
        """
        if not hasattr(self, '_suggester') or self._suggester is None:
            raise RuntimeError("Suggester not enabled. Call enable_suggester() first.")

        self._suggester.observe_choice(choice_type, chosen, alternatives)

    def get_suggestions(self) -> Dict[str, List]:
        """
        Get all current suggestions organized by type.

        Returns:
            Dict with 'definitions', 'patterns', 'preferences' lists

        Raises:
            RuntimeError: If suggester not enabled

        Example:
            >>> suggestions = processor.get_suggestions()
            >>> for defn in suggestions['definitions']:
            ...     print(f"Define '{defn.term}': seen {defn.frequency} times")
        """
        if not hasattr(self, '_suggester') or self._suggester is None:
            raise RuntimeError("Suggester not enabled. Call enable_suggester() first.")

        return self._suggester.get_all_suggestions()

    def get_definition_suggestions(self) -> List:
        """
        Get suggested definitions for undefined terms.

        Returns:
            List of DefinitionSuggestion objects

        Raises:
            RuntimeError: If suggester not enabled
        """
        if not hasattr(self, '_suggester') or self._suggester is None:
            raise RuntimeError("Suggester not enabled. Call enable_suggester() first.")

        return self._suggester.suggest_definitions()

    def get_pattern_suggestions(self) -> List:
        """
        Get suggested patterns from repeated query structures.

        Returns:
            List of PatternSuggestion objects

        Raises:
            RuntimeError: If suggester not enabled
        """
        if not hasattr(self, '_suggester') or self._suggester is None:
            raise RuntimeError("Suggester not enabled. Call enable_suggester() first.")

        return self._suggester.suggest_patterns()

    def get_preference_suggestions(self) -> List:
        """
        Get suggested preferences from consistent choices.

        Returns:
            List of PreferenceSuggestion objects

        Raises:
            RuntimeError: If suggester not enabled
        """
        if not hasattr(self, '_suggester') or self._suggester is None:
            raise RuntimeError("Suggester not enabled. Call enable_suggester() first.")

        return self._suggester.suggest_preferences()

    def export_suggestions_markdown(self) -> str:
        """
        Export all suggestions as markdown for alignment file.

        Returns:
            Markdown string ready to save

        Raises:
            RuntimeError: If suggester not enabled

        Example:
            >>> md = processor.export_suggestions_markdown()
            >>> with open("suggested_alignment.md", "w") as f:
            ...     f.write(md)
        """
        if not hasattr(self, '_suggester') or self._suggester is None:
            raise RuntimeError("Suggester not enabled. Call enable_suggester() first.")

        return self._suggester.export_suggestions_markdown()

    def add_known_term_to_suggester(self, term: str) -> None:
        """
        Add a term to the known set (won't be suggested).

        Use this after adding a definition to the alignment index.

        Args:
            term: Term to mark as known

        Raises:
            RuntimeError: If suggester not enabled
        """
        if not hasattr(self, '_suggester') or self._suggester is None:
            raise RuntimeError("Suggester not enabled. Call enable_suggester() first.")

        self._suggester.add_known_term(term)

    def get_suggester_stats(self) -> Dict[str, Any]:
        """
        Get suggestion system statistics.

        Returns:
            Dict with observation counts, suggestion counts, etc.

        Raises:
            RuntimeError: If suggester not enabled
        """
        if not hasattr(self, '_suggester') or self._suggester is None:
            raise RuntimeError("Suggester not enabled. Call enable_suggester() first.")

        return self._suggester.get_stats()

    def clear_suggester(self) -> None:
        """
        Clear all observations from the suggester.

        Use this to start fresh after processing suggestions.

        Raises:
            RuntimeError: If suggester not enabled
        """
        if not hasattr(self, '_suggester') or self._suggester is None:
            raise RuntimeError("Suggester not enabled. Call enable_suggester() first.")

        self._suggester.clear()
        logger.info("Suggester observations cleared")

    @property
    def suggester_enabled(self) -> bool:
        """Check if sample suggester is enabled."""
        return hasattr(self, '_suggester') and self._suggester is not None

    # =========================================================================
    # Transfer Learning Methods (Phase 5: Cross-Project Transfer)
    # =========================================================================

    def analyze_vocabulary(self) -> Dict[str, Any]:
        """
        Analyze vocabulary composition for transfer learning.

        Separates programming language constructs (transferable) from
        project-specific terms.

        Returns:
            Dict with vocabulary analysis including:
            - total_terms: Total unique terms
            - programming_terms: Count of programming terms
            - project_specific_terms: Count of project-specific terms
            - programming_ratio: Ratio of programming terms
            - top_programming_terms: Most common programming terms
            - top_project_terms: Most common project-specific terms

        Raises:
            RuntimeError: If SparkSLM not enabled

        Example:
            >>> analysis = processor.analyze_vocabulary()
            >>> print(f"Programming: {analysis['programming_ratio']:.1%}")
        """
        if not self.spark_enabled:
            raise RuntimeError("SparkSLM must be enabled. Call enable_spark() first.")

        from ..spark import VocabularyAnalyzer

        analyzer = VocabularyAnalyzer()
        analysis = analyzer.analyze(self._spark.ngram)
        return analysis.to_dict()

    def export_portable_model(
        self,
        path: str,
        project_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Export a portable model for cross-project transfer.

        Creates a model containing only transferable patterns (programming
        constructs) that can be loaded into other projects.

        Args:
            path: Directory to save the portable model
            project_name: Optional name for this project

        Returns:
            Dict with export statistics

        Raises:
            RuntimeError: If SparkSLM not enabled

        Example:
            >>> stats = processor.export_portable_model("./portable_model")
            >>> print(f"Exported {stats['vocab_size']} terms")
        """
        if not self.spark_enabled:
            raise RuntimeError("SparkSLM must be enabled. Call enable_spark() first.")

        from ..spark import PortableModel

        portable = PortableModel.from_ngram_model(
            self._spark.ngram,
            source_project=project_name or ""
        )
        portable.save(path)

        logger.info(f"Portable model saved to {path}")
        return portable.get_stats()

    def import_base_model(
        self,
        path: str,
        blend_weight: float = 0.3
    ) -> Dict[str, Any]:
        """
        Import a portable model as a base for transfer learning.

        Blends transferred patterns with the current model's knowledge.

        Args:
            path: Directory containing the portable model
            blend_weight: How much to weight transferred knowledge (0-1).
                         0.3 means 30% transfer, 70% local.

        Returns:
            Dict with transfer metrics

        Raises:
            RuntimeError: If SparkSLM not enabled

        Example:
            >>> metrics = processor.import_base_model("./base_model")
            >>> print(f"Vocabulary overlap: {metrics['vocabulary_overlap']:.1%}")
        """
        if not self.spark_enabled:
            raise RuntimeError("SparkSLM must be enabled. Call enable_spark() first.")

        from ..spark import PortableModel, TransferAdapter

        portable = PortableModel.load(path)
        adapter = TransferAdapter(portable, blend_weight=blend_weight)

        # Adapt in-place to current model
        adapter.adapt(self._spark.ngram, in_place=True)

        # Measure effectiveness
        metrics = adapter.measure_effectiveness(self._spark.ngram)

        logger.info(f"Imported model from {path}, overlap: {metrics.vocabulary_overlap:.1%}")
        return metrics.to_dict()

    def measure_transfer_effectiveness(
        self,
        source_path: str,
        test_queries: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Measure how effective a transfer would be without applying it.

        Useful for deciding whether to import a base model.

        Args:
            source_path: Path to portable model to evaluate
            test_queries: Optional queries to measure perplexity on

        Returns:
            Dict with effectiveness metrics

        Raises:
            RuntimeError: If SparkSLM not enabled

        Example:
            >>> metrics = processor.measure_transfer_effectiveness("./model")
            >>> if metrics['vocabulary_overlap'] > 0.3:
            ...     processor.import_base_model("./model")
        """
        if not self.spark_enabled:
            raise RuntimeError("SparkSLM must be enabled. Call enable_spark() first.")

        from ..spark import PortableModel, TransferAdapter

        portable = PortableModel.load(source_path)
        adapter = TransferAdapter(portable)

        metrics = adapter.measure_effectiveness(
            self._spark.ngram,
            test_texts=test_queries
        )

        return {
            **metrics.to_dict(),
            'source_project': portable.source_project,
            'summary': adapter.get_transfer_summary(),
        }

    def get_transferable_vocabulary(self) -> List[str]:
        """
        Get list of terms that are good candidates for transfer.

        Returns programming terms that appear frequently enough
        to have meaningful patterns.

        Returns:
            List of transferable terms

        Raises:
            RuntimeError: If SparkSLM not enabled
        """
        if not self.spark_enabled:
            raise RuntimeError("SparkSLM must be enabled. Call enable_spark() first.")

        from ..spark import VocabularyAnalyzer

        analyzer = VocabularyAnalyzer()
        transferable = analyzer.get_transferable_terms(self._spark.ngram)
        return sorted(transferable)

    def calculate_vocabulary_overlap(self, other_model_path: str) -> float:
        """
        Calculate vocabulary overlap with another model.

        Useful for determining if transfer learning would be beneficial.

        Args:
            other_model_path: Path to another portable model

        Returns:
            Jaccard similarity (0.0 to 1.0)

        Raises:
            RuntimeError: If SparkSLM not enabled
        """
        if not self.spark_enabled:
            raise RuntimeError("SparkSLM must be enabled. Call enable_spark() first.")

        from ..spark import PortableModel

        other = PortableModel.load(other_model_path)

        # Calculate overlap between vocabularies
        my_vocab = set(self._spark.ngram.vocab)
        other_vocab = other.shared_vocab

        intersection = len(my_vocab & other_vocab)
        union = len(my_vocab | other_vocab)

        return intersection / max(union, 1)

    # =========================================================================
    # Quality Evaluation Methods (Phase 2: Quality Measurement)
    # =========================================================================

    def evaluate_prediction_quality(
        self,
        test_texts: Optional[List[str]] = None,
        test_ratio: float = 0.2,
        context_size: int = 2
    ) -> Dict[str, Any]:
        """
        Evaluate SparkSLM prediction quality.

        Measures accuracy@k, perplexity, and coverage on test data.

        Args:
            test_texts: Optional test texts. If not provided, uses
                       held-out split from documents.
            test_ratio: Fraction of documents for test (if test_texts not given)
            context_size: Number of context words for predictions

        Returns:
            Dict with:
            - accuracy_at_1: Top-1 prediction accuracy
            - accuracy_at_5: Top-5 prediction accuracy
            - accuracy_at_10: Top-10 prediction accuracy
            - mean_reciprocal_rank: MRR
            - perplexity: Average perplexity
            - coverage: Fraction with predictions

        Raises:
            RuntimeError: If SparkSLM not enabled

        Example:
            >>> metrics = processor.evaluate_prediction_quality()
            >>> print(f"Accuracy@5: {metrics['accuracy_at_5']:.1%}")
        """
        if not self.spark_enabled:
            raise RuntimeError("SparkSLM must be enabled. Call enable_spark() first.")

        from ..spark import QualityEvaluator

        evaluator = QualityEvaluator(self._spark.ngram)

        # Use provided test texts or create held-out split
        if test_texts is None:
            all_texts = list(self.documents.values())
            _, test_texts = evaluator.create_held_out_split(all_texts, test_ratio)

        metrics = evaluator.evaluate_predictions(test_texts, context_size)
        return metrics.to_dict()

    def cross_validate_predictions(
        self,
        folds: int = 5
    ) -> Dict[str, Any]:
        """
        Cross-validate prediction quality across folds.

        Args:
            folds: Number of cross-validation folds

        Returns:
            Dict with:
            - folds: List of metrics per fold
            - mean_accuracy_at_5: Average accuracy@5 across folds
            - std_accuracy_at_5: Std of accuracy@5
            - mean_perplexity: Average perplexity

        Raises:
            RuntimeError: If SparkSLM not enabled

        Example:
            >>> cv = processor.cross_validate_predictions(folds=5)
            >>> print(f"Mean accuracy@5: {cv['mean_accuracy_at_5']:.1%}")
        """
        if not self.spark_enabled:
            raise RuntimeError("SparkSLM must be enabled. Call enable_spark() first.")

        from ..spark import QualityEvaluator

        evaluator = QualityEvaluator(self._spark.ngram)
        all_texts = list(self.documents.values())

        results = evaluator.cross_validate_predictions(all_texts, folds)

        # Aggregate results
        accuracies = [r.accuracy_at_5 for r in results]
        perplexities = [r.perplexity for r in results if r.perplexity < float('inf')]

        mean_acc = sum(accuracies) / len(accuracies) if accuracies else 0
        std_acc = (
            (sum((a - mean_acc) ** 2 for a in accuracies) / len(accuracies)) ** 0.5
            if accuracies else 0
        )
        mean_perp = sum(perplexities) / len(perplexities) if perplexities else float('inf')

        return {
            'folds': [r.to_dict() for r in results],
            'mean_accuracy_at_5': mean_acc,
            'std_accuracy_at_5': std_acc,
            'mean_perplexity': mean_perp,
        }

    def measure_perplexity_stability(
        self,
        test_texts: Optional[List[str]] = None,
        runs: int = 5
    ) -> Dict[str, float]:
        """
        Measure perplexity stability across runs.

        Validates that the model produces consistent perplexity scores.

        Args:
            test_texts: Optional test texts. Uses all documents if not provided.
            runs: Number of runs to average

        Returns:
            Dict with mean, std, min, max, and is_stable flag

        Raises:
            RuntimeError: If SparkSLM not enabled
        """
        if not self.spark_enabled:
            raise RuntimeError("SparkSLM must be enabled. Call enable_spark() first.")

        from ..spark import QualityEvaluator

        evaluator = QualityEvaluator(self._spark.ngram)

        if test_texts is None:
            test_texts = list(self.documents.values())[:20]  # Sample for speed

        return evaluator.measure_perplexity_stability(test_texts, runs)

    def compare_search_quality(
        self,
        queries: List[str],
        relevance: Dict[str, set],
        k: int = 5
    ) -> Dict[str, Any]:
        """
        Compare search quality with and without Spark enhancement.

        Args:
            queries: List of test queries
            relevance: Dict mapping queries to sets of relevant doc_ids
            k: Number of results to evaluate

        Returns:
            Dict with baseline metrics, spark metrics, and improvements

        Raises:
            RuntimeError: If SparkSLM not enabled

        Example:
            >>> comparison = processor.compare_search_quality(
            ...     queries=["authentication", "login"],
            ...     relevance={
            ...         "authentication": {"auth.py", "login.py"},
            ...         "login": {"login.py", "session.py"},
            ...     }
            ... )
            >>> print(f"Precision improved by {comparison['precision_improvement']:.1%}")
        """
        if not self.spark_enabled:
            raise RuntimeError("SparkSLM must be enabled. Call enable_spark() first.")

        from ..spark import SearchQualityEvaluator

        def search_baseline(query: str) -> List[Tuple[str, float]]:
            # Temporarily disable spark for baseline
            return self.find_documents_for_query(query, top_n=k)

        def search_with_spark(query: str) -> List[Tuple[str, float]]:
            # Use spark-enhanced expansion
            expanded = self.expand_query_with_spark(query)
            # Find documents using expanded terms
            return self.find_documents_for_query(query, top_n=k)

        evaluator = SearchQualityEvaluator(search_baseline, search_with_spark)
        comparison = evaluator.compare_search(queries, relevance, k)

        return comparison.to_dict()

    def generate_quality_report(self) -> str:
        """
        Generate a comprehensive quality report.

        Evaluates prediction quality, perplexity stability, and provides
        recommendations.

        Returns:
            Markdown-formatted quality report

        Raises:
            RuntimeError: If SparkSLM not enabled

        Example:
            >>> report = processor.generate_quality_report()
            >>> print(report)
        """
        if not self.spark_enabled:
            raise RuntimeError("SparkSLM must be enabled. Call enable_spark() first.")

        lines = ["# SparkSLM Quality Report\n"]

        # Prediction quality
        try:
            pred_metrics = self.evaluate_prediction_quality()
            lines.append("## Prediction Quality\n")
            lines.append(f"- Accuracy@1: {pred_metrics['accuracy_at_1']:.1%}")
            lines.append(f"- Accuracy@5: {pred_metrics['accuracy_at_5']:.1%}")
            lines.append(f"- Accuracy@10: {pred_metrics['accuracy_at_10']:.1%}")
            lines.append(f"- MRR: {pred_metrics['mean_reciprocal_rank']:.3f}")
            lines.append(f"- Perplexity: {pred_metrics['perplexity']:.1f}")
            lines.append(f"- Coverage: {pred_metrics['coverage']:.1%}")
            lines.append("")

            # Validation against roadmap criteria
            lines.append("### Roadmap Validation\n")

            if pred_metrics['accuracy_at_5'] > 0.30:
                lines.append("- [x] Accuracy@5 > 30% (better than random) - PASSED")
            else:
                lines.append(f"- [ ] Accuracy@5 > 30% - FAILED ({pred_metrics['accuracy_at_5']:.1%})")

        except Exception as e:
            lines.append(f"## Prediction Quality\n\nError: {e}")

        # Perplexity stability
        try:
            stability = self.measure_perplexity_stability()
            lines.append("\n## Perplexity Stability\n")
            lines.append(f"- Mean: {stability['mean']:.1f}")
            lines.append(f"- Std: {stability['std']:.2f}")
            lines.append(f"- Stable: {'Yes' if stability['is_stable'] else 'No'}")
        except Exception as e:
            lines.append(f"\n## Perplexity Stability\n\nError: {e}")

        # Model stats
        try:
            stats = self.get_spark_stats()
            lines.append("\n## Model Statistics\n")
            lines.append(f"- Vocabulary size: {stats['vocabulary_size']}")
            lines.append(f"- N-gram order: {stats['ngram_order']}")
            lines.append(f"- Contexts: {stats['context_count']}")
        except Exception as e:
            lines.append(f"\n## Model Statistics\n\nError: {e}")

        return "\n".join(lines)
