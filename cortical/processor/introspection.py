"""
Introspection: state inspection, fingerprints, gaps, and summaries.

This module contains methods for examining the processor state and
comparing texts/documents.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING

from ..layers import CorticalLayer
from .. import gaps as gaps_module
from .. import fingerprint as fp_module
from .. import persistence
from .. import patterns as patterns_module

if TYPE_CHECKING:
    from . import CorticalTextProcessor

logger = logging.getLogger(__name__)


class IntrospectionMixin:
    """
    Mixin providing introspection functionality.

    Requires CoreMixin to be present (provides layers, documents, tokenizer).
    """

    def get_document_signature(self, doc_id: str, n: int = 10) -> List[Tuple[str, float]]:
        """Get the top-n TF-IDF terms for a document."""
        layer0 = self.layers[CorticalLayer.TOKENS]
        terms = [(col.content, col.tfidf_per_doc.get(doc_id, col.tfidf))
                 for col in layer0.minicolumns.values() if doc_id in col.document_ids]
        return sorted(terms, key=lambda x: x[1], reverse=True)[:n]

    def get_corpus_summary(self) -> Dict:
        """Get summary statistics about the corpus."""
        return persistence.get_state_summary(self.layers, self.documents)

    def analyze_knowledge_gaps(self) -> Dict:
        """Analyze the corpus for knowledge gaps."""
        return gaps_module.analyze_knowledge_gaps(self.layers, self.documents)

    def detect_anomalies(self, threshold: float = 0.3) -> List[Dict]:
        """Detect anomalous patterns in the corpus."""
        return gaps_module.detect_anomalies(self.layers, self.documents, threshold)

    # Fingerprint methods for semantic comparison
    def get_fingerprint(self, text: str, top_n: int = 20) -> Dict:
        """
        Compute the semantic fingerprint of a text.

        The fingerprint captures the semantic essence of the text including
        term weights, concept memberships, and bigrams.

        Args:
            text: Input text to fingerprint
            top_n: Number of top terms to include

        Returns:
            Dict with 'terms', 'concepts', 'bigrams', 'top_terms', 'term_count'
        """
        return fp_module.compute_fingerprint(text, self.tokenizer, self.layers, top_n)

    def compare_fingerprints(self, fp1: Dict, fp2: Dict) -> Dict:
        """
        Compare two fingerprints and compute similarity metrics.

        Args:
            fp1: First fingerprint from get_fingerprint()
            fp2: Second fingerprint from get_fingerprint()

        Returns:
            Dict with similarity scores and shared terms
        """
        return fp_module.compare_fingerprints(fp1, fp2)

    def explain_fingerprint(self, fp: Dict, top_n: int = 10) -> Dict:
        """
        Generate a human-readable explanation of a fingerprint.

        Args:
            fp: Fingerprint from get_fingerprint()
            top_n: Number of top items to include

        Returns:
            Dict with explanation components including summary
        """
        return fp_module.explain_fingerprint(fp, top_n)

    def explain_similarity(self, fp1: Dict, fp2: Dict) -> str:
        """
        Generate a human-readable explanation of fingerprint similarity.

        Args:
            fp1: First fingerprint
            fp2: Second fingerprint

        Returns:
            Human-readable explanation string
        """
        return fp_module.explain_similarity(fp1, fp2)

    def find_similar_texts(
        self,
        text: str,
        candidates: List[Tuple[str, str]],
        top_n: int = 5
    ) -> List[Tuple[str, float, Dict]]:
        """
        Find texts most similar to the given text.

        Args:
            text: Query text to compare
            candidates: List of (id, text) tuples to search
            top_n: Number of results to return

        Returns:
            List of (id, similarity_score, comparison) tuples sorted by similarity
        """
        query_fp = self.get_fingerprint(text)
        results = []

        for candidate_id, candidate_text in candidates:
            candidate_fp = self.get_fingerprint(candidate_text)
            comparison = self.compare_fingerprints(query_fp, candidate_fp)
            results.append((candidate_id, comparison['overall_similarity'], comparison))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]

    # Semantic Diff methods
    def compare_with(
        self,
        other: 'CorticalTextProcessor',
        top_movers: int = 20,
        min_pagerank_delta: float = 0.0001
    ) -> 'diff_module.SemanticDiff':
        """
        Compare this processor state with another to find semantic differences.

        Args:
            other: Another CorticalTextProcessor to compare with
            top_movers: Number of top importance changes to track
            min_pagerank_delta: Minimum PageRank change to consider significant

        Returns:
            SemanticDiff object with all detected changes
        """
        from .. import diff as diff_module
        return diff_module.compare_processors(
            old_processor=other,
            new_processor=self,
            top_movers=top_movers,
            min_pagerank_delta=min_pagerank_delta
        )

    def compare_documents(self, doc_id_1: str, doc_id_2: str) -> Dict:
        """
        Compare two documents within this corpus.

        Args:
            doc_id_1: ID of first document
            doc_id_2: ID of second document

        Returns:
            Dict with comparison results
        """
        from .. import diff as diff_module
        return diff_module.compare_documents(self, doc_id_1, doc_id_2)

    def what_changed(self, old_content: str, new_content: str) -> Dict:
        """
        Compare two text contents to show what changed semantically.

        Args:
            old_content: The "before" text
            new_content: The "after" text

        Returns:
            Dict with semantic diff results
        """
        from .. import diff as diff_module
        return diff_module.what_changed(self, old_content, new_content)

    def summarize_document(self, doc_id: str, num_sentences: int = 3) -> str:
        """
        Generate a summary of a document using extractive summarization.

        Args:
            doc_id: Document identifier
            num_sentences: Number of sentences to include

        Returns:
            Summary string (empty if document not found)
        """
        if doc_id not in self.documents:
            return ""
        content = self.documents[doc_id]
        sentences = re.split(r'(?<=[.!?])\s+', content)
        if len(sentences) <= num_sentences:
            return content

        layer0 = self.layers[CorticalLayer.TOKENS]
        scored = []
        for sent in sentences:
            tokens = self.tokenizer.tokenize(sent)
            score = sum(layer0.get_minicolumn(t).tfidf if layer0.get_minicolumn(t) else 0 for t in tokens)
            scored.append((sent, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = [s for s, _ in scored[:num_sentences]]
        return ' '.join([s for s in sentences if s in top])

    # Pattern detection methods
    def detect_patterns(
        self,
        doc_id: str,
        patterns: Optional[List[str]] = None
    ) -> Dict[str, List[int]]:
        """
        Detect programming patterns in a specific document.

        Args:
            doc_id: Document identifier
            patterns: Specific pattern names to search for (None = all patterns)

        Returns:
            Dict mapping pattern names to list of line numbers where found

        Example:
            >>> processor.process_document("code.py", "async def fetch(): await get()")
            >>> patterns = processor.detect_patterns("code.py")
            >>> 'async_await' in patterns
            True
        """
        if doc_id not in self.documents:
            return {}

        content = self.documents[doc_id]
        return patterns_module.detect_patterns_in_text(content, patterns)

    def detect_patterns_in_corpus(
        self,
        patterns: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, List[int]]]:
        """
        Detect patterns across all documents in the corpus.

        Args:
            patterns: Specific pattern names to search for (None = all patterns)

        Returns:
            Dict mapping doc_id to pattern detection results

        Example:
            >>> results = processor.detect_patterns_in_corpus()
            >>> for doc_id, patterns in results.items():
            ...     print(f"{doc_id}: {list(patterns.keys())}")
        """
        return patterns_module.detect_patterns_in_documents(self.documents, patterns)

    def get_pattern_summary(
        self,
        doc_id: str
    ) -> Dict[str, int]:
        """
        Get a summary of pattern occurrences in a document.

        Args:
            doc_id: Document identifier

        Returns:
            Dict mapping pattern names to occurrence counts

        Example:
            >>> summary = processor.get_pattern_summary("code.py")
            >>> summary['async_await']
            3
        """
        patterns = self.detect_patterns(doc_id)
        return patterns_module.get_pattern_summary(patterns)

    def get_corpus_pattern_statistics(self) -> Dict[str, Any]:
        """
        Get pattern statistics across the entire corpus.

        Returns:
            Dict with corpus-wide statistics including:
            - total_documents: Number of documents analyzed
            - patterns_found: Number of distinct patterns detected
            - pattern_document_counts: How many docs contain each pattern
            - pattern_occurrences: Total occurrences of each pattern
            - most_common_pattern: Most frequently occurring pattern

        Example:
            >>> stats = processor.get_corpus_pattern_statistics()
            >>> stats['most_common_pattern']
            'error_handling'
        """
        doc_patterns = self.detect_patterns_in_corpus()
        return patterns_module.get_corpus_pattern_statistics(doc_patterns)

    def format_pattern_report(
        self,
        doc_id: str,
        show_lines: bool = False
    ) -> str:
        """
        Format pattern detection results as a human-readable report.

        Args:
            doc_id: Document identifier
            show_lines: Whether to show line numbers in the report

        Returns:
            Formatted report string

        Example:
            >>> report = processor.format_pattern_report("code.py", show_lines=True)
            >>> print(report)
        """
        patterns = self.detect_patterns(doc_id)
        return patterns_module.format_pattern_report(patterns, show_lines)

    def list_available_patterns(self) -> List[str]:
        """
        List all available pattern names that can be detected.

        Returns:
            Sorted list of pattern names

        Example:
            >>> patterns = processor.list_available_patterns()
            >>> 'singleton' in patterns
            True
        """
        return patterns_module.list_all_patterns()

    def list_pattern_categories(self) -> List[str]:
        """
        List all pattern categories.

        Returns:
            Sorted list of category names

        Example:
            >>> categories = processor.list_pattern_categories()
            >>> 'creational' in categories
            True
        """
        return patterns_module.list_all_categories()

    def __repr__(self) -> str:
        stats = self.get_corpus_summary()
        return f"CorticalTextProcessor(documents={stats['documents']}, columns={stats['total_columns']})"
