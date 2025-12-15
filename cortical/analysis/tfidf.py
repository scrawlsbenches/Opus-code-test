"""
TF-IDF and BM25 scoring algorithms.

Contains:
- compute_tfidf: Traditional TF-IDF scoring
- compute_bm25: Okapi BM25 scoring with length normalization
- _tfidf_core: Pure TF-IDF algorithm for unit testing
- _bm25_core: Pure BM25 algorithm for unit testing
"""

import math
from typing import Dict, Tuple

from ..layers import CorticalLayer, HierarchicalLayer


def _tfidf_core(
    term_stats: Dict[str, Tuple[int, int, Dict[str, int]]],
    num_docs: int
) -> Dict[str, Tuple[float, Dict[str, float]]]:
    """
    Pure TF-IDF calculation.

    This core function takes primitive types and can be unit tested without
    needing HierarchicalLayer objects.

    Args:
        term_stats: Dictionary mapping term to (occurrence_count, doc_frequency, {doc_id: count})
                   - occurrence_count: total times term appears in corpus
                   - doc_frequency: number of documents containing term
                   - doc_counts: per-document occurrence counts
        num_docs: Total number of documents in corpus

    Returns:
        Dictionary mapping term to (global_tfidf, {doc_id: per_doc_tfidf})

    Example:
        >>> stats = {
        ...     "rare": (5, 1, {"doc1": 5}),      # Rare term in one doc
        ...     "common": (100, 10, {"doc1": 10, "doc2": 10, ...})  # Common term
        ... }
        >>> results = _tfidf_core(stats, num_docs=10)
        >>> assert results["rare"][0] > results["common"][0]  # Rare term has higher TF-IDF
    """
    # O(total_term_occurrences) = sum over all terms of their document frequencies
    if num_docs == 0:
        return {}

    results = {}

    for term, (occurrence_count, doc_frequency, doc_counts) in term_stats.items():
        if doc_frequency > 0:
            # Inverse document frequency
            idf = math.log(num_docs / doc_frequency)

            # Global TF-IDF (using total occurrence count)
            tf = math.log1p(occurrence_count)
            global_tfidf = tf * idf

            # Per-document TF-IDF
            per_doc_tfidf = {}
            for doc_id, count in doc_counts.items():
                doc_tf = math.log1p(count)
                per_doc_tfidf[doc_id] = doc_tf * idf

            results[term] = (global_tfidf, per_doc_tfidf)
        else:
            results[term] = (0.0, {})

    return results


def _bm25_core(
    term_stats: Dict[str, Tuple[int, int, Dict[str, int]]],
    num_docs: int,
    doc_lengths: Dict[str, int],
    avg_doc_length: float,
    k1: float = 1.2,
    b: float = 0.75
) -> Dict[str, Tuple[float, Dict[str, float]]]:
    """
    Pure BM25 calculation.

    BM25 (Best Match 25) is a ranking function that improves on TF-IDF by:
    - Term frequency saturation: diminishing returns for repeated terms
    - Document length normalization: fair comparison across different lengths

    This core function takes primitive types and can be unit tested without
    needing HierarchicalLayer objects.

    Args:
        term_stats: Dictionary mapping term to (occurrence_count, doc_frequency, {doc_id: count})
                   - occurrence_count: total times term appears in corpus
                   - doc_frequency: number of documents containing term
                   - doc_counts: per-document occurrence counts
        num_docs: Total number of documents in corpus
        doc_lengths: Dictionary mapping doc_id to document length (in tokens)
        avg_doc_length: Average document length across corpus
        k1: Term frequency saturation parameter (0.0-3.0, typical 1.2-2.0)
            - Higher k1 = more weight to term frequency
            - k1=0 = binary model (presence/absence only)
        b: Length normalization parameter (0.0-1.0)
            - b=0 = no length normalization
            - b=1 = full length normalization

    Returns:
        Dictionary mapping term to (global_bm25, {doc_id: per_doc_bm25})

    Example:
        >>> stats = {
        ...     "rare": (5, 1, {"doc1": 5}),
        ...     "common": (100, 10, {"doc1": 10, "doc2": 10, ...})
        ... }
        >>> doc_lengths = {"doc1": 100, "doc2": 150}
        >>> results = _bm25_core(stats, num_docs=10, doc_lengths=doc_lengths, avg_doc_length=125)
        >>> assert results["rare"][0] > results["common"][0]
    """
    if num_docs == 0 or avg_doc_length == 0:
        return {}

    results = {}

    for term, (occurrence_count, doc_frequency, doc_counts) in term_stats.items():
        if doc_frequency > 0:
            # IDF component (same as TF-IDF but can use BM25 variant)
            # Standard BM25 IDF: log((N - df + 0.5) / (df + 0.5))
            # This can go negative for very common terms, so we use a floor
            idf = math.log((num_docs - doc_frequency + 0.5) / (doc_frequency + 0.5) + 1)

            # Global BM25 (using total occurrence count and average length)
            # This is an approximation for global term importance
            tf_global = occurrence_count / num_docs  # Average TF across docs
            global_bm25 = idf * ((tf_global * (k1 + 1)) / (tf_global + k1))

            # Per-document BM25
            per_doc_bm25 = {}
            for doc_id, tf in doc_counts.items():
                doc_len = doc_lengths.get(doc_id, avg_doc_length)
                # Length normalization factor
                len_norm = 1 - b + b * (doc_len / avg_doc_length)
                # BM25 score for this document
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * len_norm
                per_doc_bm25[doc_id] = idf * (numerator / denominator)

            results[term] = (global_bm25, per_doc_bm25)
        else:
            results[term] = (0.0, {})

    return results


def compute_tfidf(
    layers: Dict[CorticalLayer, HierarchicalLayer],
    documents: Dict[str, str]
) -> None:
    """
    Compute TF-IDF scores for tokens.

    TF-IDF (Term Frequency - Inverse Document Frequency) measures
    how distinctive a term is to the corpus. High TF-IDF terms are
    both frequent in their documents and rare across the corpus.

    Args:
        layers: Dictionary of layers (needs TOKENS layer)
        documents: Dictionary mapping doc_id to content
    """
    # O(total_term_document_occurrences) = sum over all tokens of len(token.document_ids)
    layer0 = layers[CorticalLayer.TOKENS]
    num_docs = len(documents)

    if num_docs == 0:
        return

    for col in layer0.minicolumns.values():
        # Document frequency
        df = len(col.document_ids)

        if df > 0:
            # Inverse document frequency
            idf = math.log(num_docs / df)

            # Term frequency (normalized by occurrence count)
            tf = math.log1p(col.occurrence_count)

            # TF-IDF
            col.tfidf = tf * idf

            # Per-document TF-IDF using actual occurrence counts
            for doc_id in col.document_ids:
                # Get actual term frequency in this document
                doc_tf = col.doc_occurrence_counts.get(doc_id, 1)
                col.tfidf_per_doc[doc_id] = math.log1p(doc_tf) * idf


def compute_bm25(
    layers: Dict[CorticalLayer, HierarchicalLayer],
    documents: Dict[str, str],
    doc_lengths: Dict[str, int],
    avg_doc_length: float,
    k1: float = 1.2,
    b: float = 0.75
) -> None:
    """
    Compute BM25 scores for tokens.

    BM25 (Best Match 25) is a ranking function that addresses TF-IDF limitations:
    - Term frequency saturation: prevents domination by repeated terms
    - Document length normalization: fair comparison across different lengths

    This stores scores in the same fields as TF-IDF (tfidf, tfidf_per_doc)
    for backward compatibility with existing search functions.

    Args:
        layers: Dictionary of layers (needs TOKENS layer)
        documents: Dictionary mapping doc_id to content
        doc_lengths: Dictionary mapping doc_id to token count
        avg_doc_length: Average document length in tokens
        k1: Term frequency saturation (0-3, default 1.2)
        b: Length normalization (0-1, default 0.75)
    """
    layer0 = layers[CorticalLayer.TOKENS]
    num_docs = len(documents)

    if num_docs == 0 or avg_doc_length == 0:
        return

    for col in layer0.minicolumns.values():
        # Document frequency
        df = len(col.document_ids)

        if df > 0:
            # IDF component
            # BM25 IDF: log((N - df + 0.5) / (df + 0.5) + 1)
            # The +1 ensures non-negative values for common terms
            idf = math.log((num_docs - df + 0.5) / (df + 0.5) + 1)

            # Global BM25 (approximation using average TF)
            avg_tf = col.occurrence_count / num_docs
            col.tfidf = idf * ((avg_tf * (k1 + 1)) / (avg_tf + k1))

            # Per-document BM25
            for doc_id in col.document_ids:
                tf = col.doc_occurrence_counts.get(doc_id, 1)
                doc_len = doc_lengths.get(doc_id, avg_doc_length)
                # Length normalization factor
                len_norm = 1 - b + b * (doc_len / avg_doc_length)
                # BM25 score
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * len_norm
                col.tfidf_per_doc[doc_id] = idf * (numerator / denominator)
