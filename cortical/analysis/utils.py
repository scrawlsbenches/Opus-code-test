"""
Utility functions and classes for analysis algorithms.

Contains:
- SparseMatrix: Zero-dependency sparse matrix for bigram connections
- Similarity functions: cosine_similarity, _doc_similarity, _vector_similarity
"""

import math
from typing import Dict, List, Tuple
from collections import defaultdict


# =============================================================================
# SPARSE MATRIX UTILITIES (Zero-dependency sparse matrix for bigram connections)
# =============================================================================


class SparseMatrix:
    """
    Simple sparse matrix implementation using dictionary of keys (DOK) format.

    This is a zero-dependency alternative to scipy.sparse for the specific
    use case of computing bigram co-occurrence matrices.

    Attributes:
        rows: Number of rows
        cols: Number of columns
        data: Dictionary mapping (row, col) to value
    """

    def __init__(self, rows: int, cols: int):
        """
        Initialize sparse matrix.

        Args:
            rows: Number of rows
            cols: Number of columns
        """
        self.rows = rows
        self.cols = cols
        self.data: Dict[Tuple[int, int], float] = {}

    def set(self, row: int, col: int, value: float) -> None:
        """Set value at (row, col)."""
        if value != 0:
            self.data[(row, col)] = value
        elif (row, col) in self.data:
            del self.data[(row, col)]

    def get(self, row: int, col: int) -> float:
        """Get value at (row, col)."""
        return self.data.get((row, col), 0.0)

    def multiply_transpose(self) -> 'SparseMatrix':
        """
        Multiply this matrix by its transpose: M * M^T

        For a document-term matrix D (docs x terms), D * D^T gives
        a term-term co-occurrence matrix showing which terms appear
        in the same documents.

        Returns:
            SparseMatrix of shape (cols, cols)
        """
        result = SparseMatrix(self.cols, self.cols)

        # Group by column for efficient computation
        # col_entries[col] = [(row, value), ...]
        col_entries: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        for (row, col), value in self.data.items():
            col_entries[col].append((row, value))

        # For each pair of columns, compute dot product
        cols_list = sorted(col_entries.keys())
        for i, col1 in enumerate(cols_list):
            entries1 = col_entries[col1]

            # Diagonal element (col1 with itself)
            diagonal = sum(val * val for _, val in entries1)
            result.set(col1, col1, diagonal)

            # Off-diagonal elements (col1 with col2)
            for col2 in cols_list[i+1:]:
                entries2 = col_entries[col2]

                # Compute dot product of columns col1 and col2
                # Both columns must have non-zero entries in the same row
                dict1 = {row: val for row, val in entries1}
                dot_product = 0.0
                for row, val2 in entries2:
                    if row in dict1:
                        dot_product += dict1[row] * val2

                if dot_product != 0:
                    result.set(col1, col2, dot_product)
                    result.set(col2, col1, dot_product)  # Symmetric

        return result

    def get_nonzero(self) -> List[Tuple[int, int, float]]:
        """
        Get all non-zero entries.

        Returns:
            List of (row, col, value) tuples
        """
        return [(row, col, value) for (row, col), value in self.data.items()]


# =============================================================================
# SIMILARITY FUNCTIONS
# =============================================================================


def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """
    Compute cosine similarity between two sparse vectors.

    Args:
        vec1: First vector as dict of term -> weight
        vec2: Second vector as dict of term -> weight

    Returns:
        Cosine similarity between 0 and 1
    """
    # Find common keys
    common = set(vec1.keys()) & set(vec2.keys())

    if not common:
        return 0.0

    # Compute dot product
    dot = sum(vec1[k] * vec2[k] for k in common)

    # Compute magnitudes
    mag1 = math.sqrt(sum(v * v for v in vec1.values()))
    mag2 = math.sqrt(sum(v * v for v in vec2.values()))

    if mag1 == 0 or mag2 == 0:
        return 0.0

    return dot / (mag1 * mag2)


def _doc_similarity(docs1: frozenset, docs2: frozenset) -> float:
    """
    Compute Jaccard similarity between two document sets.

    Args:
        docs1: Frozenset of document IDs for first token
        docs2: Frozenset of document IDs for second token

    Returns:
        Jaccard similarity: |intersection| / |union|
    """
    if not docs1 or not docs2:
        return 0.0

    intersection = len(docs1 & docs2)
    union = len(docs1 | docs2)

    return intersection / union if union > 0 else 0.0


def _vector_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """
    Compute similarity between two connection vectors.

    Uses Jaccard-style similarity based on shared connections.
    """
    if not vec1 or not vec2:
        return 0.0

    keys1 = set(vec1.keys())
    keys2 = set(vec2.keys())

    intersection = keys1 & keys2
    union = keys1 | keys2

    if not union:
        return 0.0

    # Weighted Jaccard: sum of mins / sum of maxes
    min_sum = 0.0
    max_sum = 0.0

    for key in union:
        v1 = vec1.get(key, 0.0)
        v2 = vec2.get(key, 0.0)
        min_sum += min(v1, v2)
        max_sum += max(v1, v2)

    return min_sum / max_sum if max_sum > 0 else 0.0
