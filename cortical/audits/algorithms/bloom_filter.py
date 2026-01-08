"""
Bloom Filter for Suspicious Comment Detection

This implementation provides a space-efficient probabilistic data structure
for testing whether an element is a member of a set.

Key Properties:
- False negatives are IMPOSSIBLE (if we added it, we'll always find it)
- False positives are POSSIBLE (may say it's there when it's not)
- Space-efficient (much smaller than storing all patterns)
- Fast O(k) operations where k = number of hash functions

Mathematical Foundation:
- Optimal bit array size: m = -n * ln(p) / (ln(2)^2)
  where n = expected items, p = target false positive rate
- Optimal hash count: k = (m/n) * ln(2)
- Actual FP rate: (1 - e^(-k*n/m))^k where n = items actually added
"""

from typing import List
import math


class SuspiciousCommentFilter:
    def __init__(self, expected_patterns: int, fp_rate: float = 0.01):
        """
        Initialize bloom filter for suspicious pattern detection.

        Args:
            expected_patterns: Expected number of suspicious patterns to add
            fp_rate: Target false positive rate (e.g., 0.01 = 1%)

        The constructor calculates optimal parameters:

        1. Bit Array Size (m):
           Formula: m = -n * ln(p) / (ln(2)^2)
           - Larger m = lower false positive rate but more memory
           - This formula minimizes space for a given target FP rate

        2. Hash Function Count (k):
           Formula: k = (m/n) * ln(2) ≈ 0.693 * (m/n)
           - More hash functions = lower FP rate up to a point
           - Too many hash functions = slower + more bit collisions
           - This formula finds the optimal trade-off
        """
        n = expected_patterns
        p = fp_rate

        # Calculate optimal bit array size
        # m = -n * ln(p) / (ln(2)^2)
        # Example: n=100, p=0.01 → m ≈ 959 bits
        self._size = int(-n * math.log(p) / (math.log(2) ** 2))

        # Ensure minimum size to avoid division by zero
        self._size = max(1, self._size)

        # Calculate optimal hash count
        # k = (m/n) * ln(2) ≈ 0.693 * (m/n)
        # Example: m=959, n=100 → k ≈ 6.65 → 7 hash functions
        self._hash_count = int((self._size / n) * math.log(2))

        # Ensure at least 3 hash functions as per requirements
        self._hash_count = max(3, self._hash_count)

        # Initialize bit array - all False initially
        self._bit_array: List[bool] = [False] * self._size

        # Track how many items have been added (for FP rate estimation)
        self._items_added: int = 0

    def add(self, pattern: str) -> None:
        """
        Add a suspicious pattern to the filter.

        Process:
        1. Generate k different hash positions for the pattern
        2. Set all k bits to True in the bit array

        Why false negatives are impossible:
        - Once we set k bits to True for a pattern, they stay True forever
        - When we query that pattern, we check the same k positions
        - Since we set them all to True, the check will always succeed

        Args:
            pattern: The suspicious pattern string to add
        """
        for i in range(self._hash_count):
            # Generate k different hash positions using double-hashing
            index = self._hash(pattern, i)
            self._bit_array[index] = True

        self._items_added += 1

    def probably_suspicious(self, pattern: str) -> bool:
        """
        Check if pattern is probably in the suspicious set.

        Returns:
            True: Pattern is PROBABLY suspicious (may be false positive)
            False: Pattern is DEFINITELY NOT suspicious (never false negative)

        Process:
        1. Generate the same k hash positions we would use in add()
        2. Check if ALL k bits are True
        3. If ANY bit is False → definitely not in the set
        4. If ALL bits are True → probably in the set (or hash collision)

        Why false positives can happen:
        - Other patterns may have set the same bits to True
        - If enough patterns overlap, all k bits might be True by coincidence
        - This is the trade-off for space efficiency

        Args:
            pattern: The pattern string to check

        Returns:
            True if all k bits are set, False if any bit is unset
        """
        for i in range(self._hash_count):
            index = self._hash(pattern, i)
            if not self._bit_array[index]:
                # Found at least one bit that's False
                # Therefore this pattern was definitely NOT added
                return False

        # All k bits are True
        # Either we added this pattern, or it's a false positive
        return True

    def false_positive_rate(self) -> float:
        """
        Estimate current false positive rate based on fill ratio.

        Formula: (1 - e^(-k*n/m))^k
        where:
        - k = number of hash functions
        - n = number of items actually added
        - m = bit array size

        Intuition:
        - As we add more items, more bits get set to True
        - The probability that k random bits are all True increases
        - This formula models that probability mathematically

        Returns:
            Estimated false positive rate (0.0 to 1.0)
        """
        if self._items_added == 0:
            # No items added = no false positives possible
            return 0.0

        k = self._hash_count
        n = self._items_added
        m = self._size

        # Calculate (1 - e^(-k*n/m))^k
        # This is the probability that k random bits are all True
        exponent = -k * n / m
        fp_rate = (1 - math.exp(exponent)) ** k

        return fp_rate

    @property
    def size(self) -> int:
        """Return bit array size (m)."""
        return self._size

    @property
    def hash_count(self) -> int:
        """Return number of hash functions (k)."""
        return self._hash_count

    def _hash(self, item: str, seed: int) -> int:
        """
        Generate hash for item with given seed using double-hashing.

        Double-hashing technique:
            h_i(x) = (hash1(x) + i * hash2(x)) % m

        where:
        - hash1(x): Primary hash function
        - hash2(x): Secondary hash function
        - i: The seed (0 to k-1)
        - m: Bit array size

        Why double-hashing?
        - Need k different hash functions for each pattern
        - Computing k completely independent hash functions is expensive
        - Double-hashing generates k hash values from just 2 hash functions
        - The values are well-distributed if hash1 and hash2 are independent

        Implementation:
        - hash1: Polynomial rolling hash with prime 31
        - hash2: Polynomial rolling hash with prime 37
        - Both are deterministic (same input always gives same output)
        - hash2 is made odd to avoid clustering issues

        Args:
            item: The string to hash
            seed: The hash function index (0 to k-1)

        Returns:
            A valid index into the bit array (0 to m-1)
        """
        # hash1: Simple polynomial rolling hash with prime 31
        # This is similar to Java's String.hashCode()
        hash1 = 0
        for c in item:
            hash1 = (hash1 * 31 + ord(c)) % (2**32)

        # hash2: Different polynomial rolling hash with prime 37
        # Using a different prime ensures independence from hash1
        hash2 = 0
        for c in item:
            hash2 = (hash2 * 37 + ord(c)) % (2**32)

        # Make hash2 odd (never zero) to ensure good distribution
        # If hash2 were 0, all h_i(x) would be the same (just hash1)
        hash2 = hash2 * 2 + 1

        # Double-hashing formula: h_i(x) = (hash1(x) + i * hash2(x)) % m
        # Each seed i gives a different position
        index = (hash1 + seed * hash2) % self._size

        return index
