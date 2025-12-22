"""
Chunking Module
==============

Functions for splitting documents into chunks for passage retrieval.

This module provides:
- Fixed-size text chunking with overlap
- Code-aware chunking aligned to semantic boundaries
- Chunk scoring against query terms
"""

from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
import re

from ..layers import HierarchicalLayer
from ..tokenizer import Tokenizer
from .utils import get_tfidf_score

if TYPE_CHECKING:
    from ..minicolumn import Minicolumn


# Pattern to detect code structure boundaries
CODE_BOUNDARY_PATTERN = re.compile(
    r'^(?:'
    r'class\s+\w+|'          # Class definitions
    r'def\s+\w+|'            # Function definitions
    r'async\s+def\s+\w+|'    # Async function definitions
    r'@\w+|'                 # Decorators
    r'#\s*[-=]{3,}|'         # Comment separators (# --- or # ===)
    r'"""[^"]*"""|'          # Module/class docstrings (simple)
    r"'''[^']*'''"           # Module/class docstrings (simple, single quotes)
    r')',
    re.MULTILINE
)


def create_chunks(
    text: str,
    chunk_size: int = 512,
    overlap: int = 128
) -> List[Tuple[str, int, int]]:
    """
    Split text into overlapping chunks.

    Args:
        text: Document text to chunk
        chunk_size: Target size of each chunk in characters
        overlap: Number of overlapping characters between chunks

    Returns:
        List of (chunk_text, start_char, end_char) tuples

    Raises:
        ValueError: If chunk_size <= 0 or overlap < 0 or overlap >= chunk_size
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(f"overlap must be less than chunk_size, got overlap={overlap}, chunk_size={chunk_size}")

    if not text:
        return []

    chunks = []
    stride = max(1, chunk_size - overlap)
    text_len = len(text)

    for start in range(0, text_len, stride):
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append((chunk, start, end))

        if end >= text_len:
            break

    return chunks


def find_code_boundaries(text: str) -> List[int]:
    """
    Find semantic boundaries in code (class/function definitions, decorators).

    Args:
        text: Source code text

    Returns:
        Sorted list of character positions where semantic units begin
    """
    boundaries = set([0])  # Always include start

    # Find class/def boundaries
    for match in CODE_BOUNDARY_PATTERN.finditer(text):
        # Find the start of the line containing this match
        line_start = text.rfind('\n', 0, match.start()) + 1
        boundaries.add(line_start)

    # Also add positions after blank line sequences (natural section breaks)
    blank_line_pattern = re.compile(r'\n\n+')
    for match in blank_line_pattern.finditer(text):
        boundaries.add(match.end())

    return sorted(boundaries)


def create_code_aware_chunks(
    text: str,
    target_size: int = 512,
    min_size: int = 100,
    max_size: int = 1024
) -> List[Tuple[str, int, int]]:
    """
    Create chunks aligned to code structure boundaries.

    Unlike fixed-size chunking, this function tries to split code at
    natural boundaries (class definitions, function definitions, blank lines)
    to preserve semantic context within each chunk.

    Args:
        text: Source code text to chunk
        target_size: Target chunk size in characters (default 512)
        min_size: Minimum chunk size - won't create chunks smaller than this (default 100)
        max_size: Maximum chunk size - will split even mid-code if exceeded (default 1024)

    Returns:
        List of (chunk_text, start_char, end_char) tuples

    Example:
        >>> text = '''
        ... class Foo:
        ...     def bar(self):
        ...         pass
        ...
        ... class Baz:
        ...     def qux(self):
        ...         pass
        ... '''
        >>> chunks = create_code_aware_chunks(text, target_size=100)
        >>> # Chunks will be aligned to class/function boundaries
    """
    if not text:
        return []

    if len(text) <= target_size:
        return [(text, 0, len(text))]

    boundaries = find_code_boundaries(text)
    boundaries.append(len(text))  # Add end of text

    chunks = []
    chunk_start = 0
    i = 1

    while chunk_start < len(text):
        # Find the next boundary that would exceed target_size
        best_end = chunk_start + max_size  # Default to max_size if no boundary found

        # Look for a boundary between target_size and max_size
        for j in range(i, len(boundaries)):
            boundary = boundaries[j]
            chunk_len = boundary - chunk_start

            if chunk_len >= target_size:
                if chunk_len <= max_size:
                    # Good boundary within range
                    best_end = boundary
                    i = j + 1
                    break
                else:
                    # Boundary too far, use previous one or force split
                    if j > i:
                        prev_boundary = boundaries[j - 1]
                        prev_len = prev_boundary - chunk_start
                        if prev_len >= min_size:
                            best_end = prev_boundary
                            i = j
                            break
                    # Force split at max_size
                    best_end = chunk_start + max_size
                    # Find the next boundary after our split point
                    for k in range(i, len(boundaries)):
                        if boundaries[k] > best_end:
                            i = k
                            break
                    break
        else:
            # Reached end of boundaries, use text end
            best_end = len(text)
            i = len(boundaries)

        # Ensure we don't exceed max_size
        if best_end - chunk_start > max_size:
            best_end = chunk_start + max_size

        # Create chunk if non-empty
        chunk_text = text[chunk_start:best_end]
        if chunk_text.strip():
            chunks.append((chunk_text, chunk_start, best_end))

        chunk_start = best_end

    return chunks


def is_code_file(doc_id: str) -> bool:
    """
    Determine if a document is a code file based on its path/extension.

    Args:
        doc_id: Document identifier (typically a file path)

    Returns:
        True if the document appears to be a code file
    """
    code_extensions = {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.h',
        '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala', '.cs'
    }
    for ext in code_extensions:
        if doc_id.endswith(ext):
            return True
    return False


def precompute_term_cols(
    query_terms: Dict[str, float],
    layer0: HierarchicalLayer
) -> Dict[str, 'Minicolumn']:
    """
    Pre-compute minicolumn lookups for query terms.

    This avoids repeated O(1) dictionary lookups for each chunk,
    enabling faster scoring when processing many chunks.

    Args:
        query_terms: Dict mapping query terms to weights
        layer0: Token layer for lookups

    Returns:
        Dict mapping term to Minicolumn (only for terms that exist in corpus)
    """
    term_cols = {}
    for term in query_terms:
        col = layer0.get_minicolumn(term)
        if col:
            term_cols[term] = col
    return term_cols


def score_chunk_fast(
    chunk_tokens: List[str],
    query_terms: Dict[str, float],
    term_cols: Dict[str, 'Minicolumn'],
    doc_id: Optional[str] = None
) -> float:
    """
    Fast chunk scoring using pre-computed minicolumn lookups.

    This is an optimized version of score_chunk that accepts pre-tokenized
    text and pre-computed minicolumn lookups. Use when scoring many chunks
    from the same document.

    Args:
        chunk_tokens: Pre-tokenized chunk tokens
        query_terms: Dict mapping query terms to weights
        term_cols: Pre-computed term->Minicolumn mapping from precompute_term_cols()
        doc_id: Optional document ID for per-document TF-IDF

    Returns:
        Relevance score for the chunk
    """
    if not chunk_tokens:
        return 0.0

    # Count token occurrences in chunk
    token_counts: Dict[str, int] = {}
    for token in chunk_tokens:
        token_counts[token] = token_counts.get(token, 0) + 1

    score = 0.0
    for term, term_weight in query_terms.items():
        if term in token_counts and term in term_cols:
            col = term_cols[term]
            # Use per-document TF-IDF if available, otherwise global
            tfidf = get_tfidf_score(col, doc_id)
            # Weight by occurrence in chunk and query weight
            score += tfidf * token_counts[term] * term_weight

    # Normalize by chunk length to avoid bias toward longer chunks
    return score / len(chunk_tokens)


def score_chunk(
    chunk_text: str,
    query_terms: Dict[str, float],
    layer0: HierarchicalLayer,
    tokenizer: Tokenizer,
    doc_id: Optional[str] = None
) -> float:
    """
    Score a chunk against query terms using TF-IDF.

    Args:
        chunk_text: Text of the chunk
        query_terms: Dict mapping query terms to weights
        layer0: Token layer for TF-IDF lookups
        tokenizer: Tokenizer instance
        doc_id: Optional document ID for per-document TF-IDF

    Returns:
        Relevance score for the chunk
    """
    chunk_tokens = tokenizer.tokenize(chunk_text)
    if not chunk_tokens:
        return 0.0

    # Count token occurrences in chunk
    token_counts: Dict[str, int] = {}
    for token in chunk_tokens:
        token_counts[token] = token_counts.get(token, 0) + 1

    score = 0.0
    for term, term_weight in query_terms.items():
        if term in token_counts:
            col = layer0.get_minicolumn(term)
            if col:
                # Use per-document TF-IDF if available, otherwise global
                tfidf = get_tfidf_score(col, doc_id)
                # Weight by occurrence in chunk and query weight
                score += tfidf * token_counts[term] * term_weight

    # Normalize by chunk length to avoid bias toward longer chunks
    return score / len(chunk_tokens) if chunk_tokens else 0.0
