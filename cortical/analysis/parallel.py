"""
Parallel processing for compute operations.

This module provides parallelized versions of expensive compute operations
using Python's ProcessPoolExecutor from the standard library.

Key features:
- Zero external dependencies (uses stdlib only)
- Automatic fallback to sequential for small corpora
- Same results as sequential (deterministic)
- Configurable chunk size and worker count

Design:
- Pure core functions (_tfidf_core, _bm25_core) are perfect for multiprocessing
- They take primitive types (dicts, lists) and return primitive types
- No side effects, no HierarchicalLayer objects passed across processes
- Results are merged and applied back to layers in the main process
"""

import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

from .tfidf import _tfidf_core, _bm25_core


@dataclass
class ParallelConfig:
    """
    Configuration for parallel processing.

    Attributes:
        num_workers: Number of worker processes (None = CPU count)
        chunk_size: Items per chunk (default 1000)
        min_items_for_parallel: Minimum items to use parallel processing
                                (below this, use sequential for efficiency)
    """
    num_workers: Optional[int] = None
    chunk_size: int = 1000
    min_items_for_parallel: int = 2000


def chunk_dict(data: dict, chunk_size: int) -> List[dict]:
    """
    Split a dictionary into chunks for parallel processing.

    Args:
        data: Dictionary to split
        chunk_size: Maximum items per chunk

    Returns:
        List of dictionary chunks

    Example:
        >>> data = {"a": 1, "b": 2, "c": 3}
        >>> chunks = chunk_dict(data, 2)
        >>> len(chunks)
        2
        >>> sum(len(c) for c in chunks)
        3
    """
    items = list(data.items())
    chunks = []
    for i in range(0, len(items), chunk_size):
        chunk = dict(items[i:i+chunk_size])
        chunks.append(chunk)
    return chunks


def extract_term_stats(layer) -> Dict[str, Tuple[int, int, Dict[str, int]]]:
    """
    Extract term statistics from a layer into picklable primitives.

    This extracts all data needed for TF-IDF/BM25 computation without
    passing HierarchicalLayer or Minicolumn objects across process boundaries.

    Args:
        layer: HierarchicalLayer (TOKENS layer)

    Returns:
        Dictionary mapping term to (occurrence_count, doc_frequency, doc_counts)

    Example:
        >>> layer = processor.layers[CorticalLayer.TOKENS]
        >>> stats = extract_term_stats(layer)
        >>> term, (occ, df, doc_counts) = next(iter(stats.items()))
        >>> isinstance(occ, int) and isinstance(df, int) and isinstance(doc_counts, dict)
        True
    """
    term_stats = {}
    for col in layer.minicolumns.values():
        term_stats[col.content] = (
            col.occurrence_count,
            len(col.document_ids),
            dict(col.doc_occurrence_counts)  # Copy to ensure it's a plain dict
        )
    return term_stats


def parallel_tfidf(
    term_stats: Dict[str, Tuple[int, int, Dict[str, int]]],
    num_docs: int,
    config: Optional[ParallelConfig] = None
) -> Dict[str, Tuple[float, Dict[str, float]]]:
    """
    Parallel TF-IDF computation using ProcessPoolExecutor.

    Splits term_stats into chunks, computes TF-IDF for each chunk in parallel,
    then merges results. Falls back to sequential for small corpora to avoid
    multiprocessing overhead.

    Args:
        term_stats: Dictionary mapping term to (occurrence_count, doc_frequency, doc_counts)
        num_docs: Total number of documents in corpus
        config: Optional ParallelConfig (uses defaults if None)

    Returns:
        Dictionary mapping term to (global_tfidf, per_doc_tfidf)

    Example:
        >>> from cortical.layers import CorticalLayer
        >>> layer = processor.layers[CorticalLayer.TOKENS]
        >>> stats = extract_term_stats(layer)
        >>> results = parallel_tfidf(stats, len(processor.documents))
        >>> # Results match sequential computation
        >>> len(results) == len(stats)
        True
    """
    if config is None:
        config = ParallelConfig()

    # Fall back to sequential for small corpora
    if len(term_stats) < config.min_items_for_parallel:
        return _tfidf_core(term_stats, num_docs)

    # Split into chunks
    chunks = chunk_dict(term_stats, config.chunk_size)

    # Process chunks in parallel
    results = {}
    with ProcessPoolExecutor(max_workers=config.num_workers) as executor:
        # Submit all chunks
        future_to_chunk = {
            executor.submit(_tfidf_core, chunk, num_docs): chunk
            for chunk in chunks
        }

        # Collect results as they complete
        for future in as_completed(future_to_chunk):
            chunk_result = future.result()
            results.update(chunk_result)

    return results


def parallel_bm25(
    term_stats: Dict[str, Tuple[int, int, Dict[str, int]]],
    num_docs: int,
    doc_lengths: Dict[str, int],
    avg_doc_length: float,
    k1: float = 1.2,
    b: float = 0.75,
    config: Optional[ParallelConfig] = None
) -> Dict[str, Tuple[float, Dict[str, float]]]:
    """
    Parallel BM25 computation using ProcessPoolExecutor.

    Splits term_stats into chunks, computes BM25 for each chunk in parallel,
    then merges results. Falls back to sequential for small corpora to avoid
    multiprocessing overhead.

    Args:
        term_stats: Dictionary mapping term to (occurrence_count, doc_frequency, doc_counts)
        num_docs: Total number of documents in corpus
        doc_lengths: Dictionary mapping doc_id to document length (in tokens)
        avg_doc_length: Average document length across corpus
        k1: Term frequency saturation parameter (0.0-3.0)
        b: Length normalization parameter (0.0-1.0)
        config: Optional ParallelConfig (uses defaults if None)

    Returns:
        Dictionary mapping term to (global_bm25, per_doc_bm25)

    Example:
        >>> from cortical.layers import CorticalLayer
        >>> layer = processor.layers[CorticalLayer.TOKENS]
        >>> stats = extract_term_stats(layer)
        >>> results = parallel_bm25(
        ...     stats,
        ...     len(processor.documents),
        ...     processor.doc_lengths,
        ...     processor.avg_doc_length
        ... )
        >>> # Results match sequential computation
        >>> len(results) == len(stats)
        True
    """
    if config is None:
        config = ParallelConfig()

    # Fall back to sequential for small corpora
    if len(term_stats) < config.min_items_for_parallel:
        return _bm25_core(term_stats, num_docs, doc_lengths, avg_doc_length, k1, b)

    # Split into chunks
    chunks = chunk_dict(term_stats, config.chunk_size)

    # Process chunks in parallel
    results = {}
    with ProcessPoolExecutor(max_workers=config.num_workers) as executor:
        # Submit all chunks
        future_to_chunk = {
            executor.submit(_bm25_core, chunk, num_docs, doc_lengths, avg_doc_length, k1, b): chunk
            for chunk in chunks
        }

        # Collect results as they complete
        for future in as_completed(future_to_chunk):
            chunk_result = future.result()
            results.update(chunk_result)

    return results
