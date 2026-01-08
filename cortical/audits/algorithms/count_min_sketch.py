"""
Count-Min Sketch Implementation for Comment Pattern Frequency

Algorithm: Count-Min Sketch
Purpose: Sub-linear space frequency estimation with guaranteed no underestimation
Complexity: O(d) for add and query, where d = depth (number of hash functions)

Key Properties:
1. Never underestimates (query >= actual count)
2. May overestimate due to hash collisions
3. Space: O(d * w) independent of number of distinct patterns
4. Error bound: Overestimate by at most N/w with probability 1 - (1/2)^d

Why Take Minimum Across Rows?
-----------------------------
Each row uses a different hash function. When querying, we get d estimates,
one from each row. Hash collisions can cause overestimation (other patterns
hashing to the same bucket inflate the count). The actual count is AT MOST
the minimum across all rows, because that's the row with least collision.

Example:
  Row 0: pattern hashes to bucket 5, which has count 10 (from pattern + collisions)
  Row 1: pattern hashes to bucket 8, which has count 7 (less collision)
  Row 2: pattern hashes to bucket 2, which has count 6 (least collision)

  We return min(10, 7, 6) = 6, which is closest to actual count.

Double Hashing Technique:
------------------------
To generate d independent hash functions from a single hash (MD5):
  h_i(x) = (hash1(x) + i * hash2(x)) % width

This gives us different hash functions for each row without computing d separate hashes.

When Merging Is Useful:
----------------------
Distributed counting across modules:
  - Worker 1 processes cortical/got/ comments -> CMS1
  - Worker 2 processes cortical/cdg/ comments -> CMS2
  - Merge CMS1 + CMS2 to get global pattern frequencies
"""

from typing import List
import hashlib
import math


class PatternFrequencySketch:
    def __init__(self, width: int, depth: int):
        """
        Initialize Count-Min Sketch for pattern frequency.

        Args:
            width: Number of counters per row (w) - more = less collision
            depth: Number of hash functions/rows (d) - more = higher accuracy

        Error bounds:
        - Overestimate by at most N/w with probability 1 - (1/2)^d
        - where N = total count of all items

        Hash function generation:
        Use double hashing: h_i(x) = (hash1(x) + i * hash2(x)) % width
        where:
        - hash1(x) = int(md5(x).hexdigest()[:8], 16)
        - hash2(x) = int(md5(x).hexdigest()[8:16], 16)
        """
        self._width = width
        self._depth = depth
        # Initialize d rows of w counters each, all starting at 0
        self._counters: List[List[int]] = [[0] * width for _ in range(depth)]
        self._total_count = 0

    def _hash(self, item: str, row: int) -> int:
        """
        Generate hash for item at given row using double hashing.

        Args:
            item: The pattern string to hash
            row: Which row (hash function) to use (0 to depth-1)

        Returns:
            Column index in range [0, width-1]

        Formula: h_row(item) = (hash1(item) + row * hash2(item)) % width

        Use MD5 for consistent cross-platform behavior:
        - hash1 = first 8 hex digits of md5 (32 bits)
        - hash2 = next 8 hex digits of md5 (32 bits)

        This gives us d different hash functions from a single MD5 computation.
        """
        # Compute MD5 hash of the item (encoded as UTF-8)
        md5_hash = hashlib.md5(item.encode('utf-8')).hexdigest()

        # Extract two independent hash values from the hex digest
        # hexdigest() gives 32 hex chars (128 bits)
        # Take first 8 hex chars (32 bits) for hash1
        # Take next 8 hex chars (32 bits) for hash2
        hash1 = int(md5_hash[:8], 16)
        hash2 = int(md5_hash[8:16], 16)

        # Double hashing formula: linear combination modulo width
        # For row 0: (hash1 + 0 * hash2) % width = hash1 % width
        # For row 1: (hash1 + 1 * hash2) % width
        # For row 2: (hash1 + 2 * hash2) % width
        # Each row gets a different hash function
        return (hash1 + row * hash2) % self._width

    def add(self, pattern: str, count: int = 1) -> None:
        """
        Add count for pattern. Updates all d rows.

        Args:
            pattern: The pattern string to add (e.g., "FUTURE:", "will be")
            count: How many times to count this pattern (default 1)

        Algorithm:
            For each row i in [0, depth-1]:
                col = hash_i(pattern)
                counters[i][col] += count

        Time: O(d) where d = depth
        """
        for row in range(self._depth):
            col = self._hash(pattern, row)
            self._counters[row][col] += count

        # Track total count across all patterns
        self._total_count += count

    def query(self, pattern: str) -> int:
        """
        Estimate count for pattern.

        Args:
            pattern: The pattern string to query

        Returns:
            Estimated count (guaranteed >= actual count, may overestimate)

        Algorithm:
            Return min(counters[i][hash_i(pattern)] for i in range(depth))

        Why minimum?
        Each row may have collisions (multiple patterns hash to same bucket).
        The minimum across rows is the best estimate because it has least collision.
        This never underestimates because the actual count contributes to every row.

        Time: O(d) where d = depth
        """
        estimates = []
        for row in range(self._depth):
            col = self._hash(pattern, row)
            estimates.append(self._counters[row][col])

        return min(estimates)

    def merge(self, other: 'PatternFrequencySketch') -> 'PatternFrequencySketch':
        """
        Merge two sketches with same dimensions.

        Args:
            other: Another PatternFrequencySketch to merge with

        Returns:
            New sketch with combined counts (self + other)

        Raises:
            ValueError: If dimensions don't match

        Use case: Distributed counting
        - Process cortical/got/ comments -> CMS1
        - Process cortical/cdg/ comments -> CMS2
        - merged = CMS1.merge(CMS2) -> combined frequencies

        Algorithm:
            For each position (i, j):
                merged[i][j] = self[i][j] + other[i][j]

        Why this works:
        CMS is a linear sketch. If pattern P contributes count c1 to self and
        c2 to other, then merged will have c1+c2 for P in each row.
        The minimum property is preserved.
        """
        # Validate dimensions match
        if self._width != other._width or self._depth != other._depth:
            raise ValueError(
                f"Cannot merge sketches with different dimensions: "
                f"({self._width}, {self._depth}) vs ({other._width}, {other._depth})"
            )

        # Create new sketch with same dimensions
        merged = PatternFrequencySketch(self._width, self._depth)

        # Add counters element-wise
        for row in range(self._depth):
            for col in range(self._width):
                merged._counters[row][col] = (
                    self._counters[row][col] + other._counters[row][col]
                )

        # Merge total counts
        merged._total_count = self._total_count + other._total_count

        return merged

    @property
    def total_count(self) -> int:
        """Return total number of items added (sum of all counts)."""
        return self._total_count

    def heavy_hitters(self, threshold_fraction: float = 0.01) -> List[str]:
        """
        Note: CMS cannot enumerate heavy hitters by itself.
        This method is a placeholder - in practice you'd track
        candidates separately and verify with query().

        For this experiment, we'll skip this method.
        """
        raise NotImplementedError("CMS cannot enumerate - track candidates separately")
