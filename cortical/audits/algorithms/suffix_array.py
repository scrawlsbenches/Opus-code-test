"""
Suffix Array Implementation for Comment Pattern Mining

This implementation provides:
1. Suffix array construction - O(n² log n) time, O(n) space
2. Pattern search using binary search - O(m log n) time
3. LCP array using Kasai's algorithm - O(n) time
4. Repeated substring detection using LCP array
"""

from typing import List, Tuple


class CommentPatternFinder:
    def __init__(self, text: str):
        """
        Build suffix array for the given text.
        Store:
        - self._text: the original text
        - self._suffix_array: sorted indices of suffixes
        - self._lcp: LCP array (computed lazily or eagerly)
        """
        self._text: str = text
        self._suffix_array: List[int] = []
        self._lcp: List[int] = []
        self._build()

    def _build(self) -> None:
        """
        Build the suffix array by sorting suffix indices.
        Suffix at index i is text[i:].
        Sort by lexicographic order of the suffixes.

        Time complexity: O(n² log n) - comparing strings is O(n), sorting is O(n log n)
        Space complexity: O(n)

        How it works:
        - Create indices [0, 1, 2, ..., n-1]
        - Sort these indices by comparing text[i:] for each index i
        - The result is suffix_array where suffix_array[j] is the starting position
          of the j-th suffix in lexicographic order
        """
        n = len(self._text)
        if n == 0:
            return

        # Create list of all suffix indices and sort by suffix comparison
        self._suffix_array = list(range(n))
        self._suffix_array.sort(key=lambda i: self._text[i:])

    def search(self, pattern: str) -> List[int]:
        """
        Find all starting positions where pattern occurs in text.
        Use binary search on suffix array for O(m log n) search.
        Returns list of positions, sorted.

        How suffix array enables fast pattern search:
        - Suffixes are sorted lexicographically
        - All suffixes starting with pattern P form a contiguous range
        - Use binary search to find [left, right) bounds of this range
        - Return the original text positions (suffix_array values in range)

        Binary search bounds:
        - Left bound: first suffix >= pattern (or would come after pattern)
        - Right bound: first suffix > pattern (strictly greater)
        - Range [left, right) contains all matches
        """
        if not pattern or not self._text:
            return []

        n = len(self._suffix_array)
        m = len(pattern)

        # Binary search for leftmost position where suffix >= pattern
        left = 0
        right = n

        while left < right:
            mid = (left + right) // 2
            suffix_idx = self._suffix_array[mid]
            # Compare only the first m characters of the suffix
            suffix = self._text[suffix_idx:suffix_idx + m]

            if suffix < pattern:
                left = mid + 1
            else:
                right = mid

        first = left

        # Binary search for rightmost position where suffix <= pattern
        left = 0
        right = n

        while left < right:
            mid = (left + right) // 2
            suffix_idx = self._suffix_array[mid]
            suffix = self._text[suffix_idx:suffix_idx + m]

            if suffix <= pattern:
                left = mid + 1
            else:
                right = mid

        last = left

        # Extract all matching positions and verify full pattern match
        result = []
        for i in range(first, last):
            suffix_idx = self._suffix_array[i]
            # Verify the full pattern matches (not just lexicographic comparison)
            if self._text[suffix_idx:suffix_idx + m] == pattern:
                result.append(suffix_idx)

        return sorted(result)

    def lcp_array(self) -> List[int]:
        """
        Compute the Longest Common Prefix array.
        lcp[i] = length of longest common prefix between suffix[i] and suffix[i-1]
        lcp[0] = 0 by convention

        Use Kasai's algorithm for O(n) computation:

        Algorithm explanation:
        1. Build rank array: rank[i] = position of suffix i in sorted suffix array
           (This is the inverse of suffix_array)

        2. Process suffixes in TEXT order (0, 1, 2, ..., n-1), not sorted order
           For each suffix starting at position i:
           - Find its predecessor in sorted order (the suffix before it)
           - Compute LCP between current suffix and its predecessor
           - Use previous LCP value as starting point (KEY OPTIMIZATION)

        3. Key property: lcp[rank[i]] >= lcp[rank[i-1]] - 1
           When moving from suffix i-1 to suffix i (in text order),
           the LCP can decrease by at most 1.

           Why? If text[i-1:] and its predecessor share k characters,
           then text[i:] and its predecessor share at least k-1 characters
           (we're removing the first character from both suffixes)

        Time complexity: O(n) - h increases at most n times, decreases at most n times
        """
        if self._lcp:
            return self._lcp

        n = len(self._text)
        if n == 0:
            return []

        # Build rank array (inverse of suffix array)
        # rank[i] = position of suffix starting at i in the sorted suffix array
        rank = [0] * n
        for i in range(n):
            rank[self._suffix_array[i]] = i

        # Compute LCP array using Kasai's algorithm
        self._lcp = [0] * n
        h = 0  # Current LCP length (accumulated from previous iteration)

        # Process suffixes in TEXT order
        for i in range(n):
            if rank[i] > 0:
                # Get the previous suffix in sorted order
                j = self._suffix_array[rank[i] - 1]

                # Compute LCP between text[i:] and text[j:]
                # Start from h (reusing computation from previous iteration)
                while i + h < n and j + h < n and self._text[i + h] == self._text[j + h]:
                    h += 1

                self._lcp[rank[i]] = h

                # Decrease h by 1 for next iteration (key optimization)
                # The next suffix (i+1) will have LCP at least h-1 with its predecessor
                if h > 0:
                    h -= 1

        return self._lcp

    def repeated_substrings(self, min_length: int = 5) -> List[Tuple[str, int]]:
        """
        Find substrings that appear more than once.
        Returns list of (substring, count) tuples, sorted by length (longest first).

        Uses LCP array:
        - If lcp[i] >= min_length, then suffix[i-1] and suffix[i] share a common prefix
        - This common prefix appears at least twice in the text
        - Collect all such prefixes and count their total occurrences

        Algorithm:
        1. Scan LCP array for values >= min_length
        2. Extract the common prefix for each such position
        3. Deduplicate substrings (same substring may appear in multiple LCP positions)
        4. Count total occurrences of each unique substring in the text
        5. Return sorted by length (longest first), then by count
        """
        if not self._text:
            return []

        lcp = self.lcp_array()
        n = len(lcp)

        # Set to collect unique repeated substrings of length >= min_length
        unique_substrings = set()

        for i in range(1, n):
            if lcp[i] >= min_length:
                suffix_idx = self._suffix_array[i]
                # Extract the longest common prefix between suffix[i-1] and suffix[i]
                prefix = self._text[suffix_idx:suffix_idx + lcp[i]]
                unique_substrings.add(prefix)

        # Count occurrences of each unique substring in the original text
        result = []
        for substring in unique_substrings:
            count = self._text.count(substring)
            if count > 1:  # Only include substrings that actually repeat
                result.append((substring, count))

        # Sort by length (descending), then by count (descending)
        return sorted(result, key=lambda x: (-len(x[0]), -x[1]))

    @property
    def suffixes(self) -> List[int]:
        """Return the suffix array (list of starting indices in sorted order)."""
        return self._suffix_array


# ============================================================================
# TEST CASES
# ============================================================================

def run_all_tests():
    """Run all test cases from the experiment."""

    print("=" * 70)
    print("SUFFIX ARRAY IMPLEMENTATION - TEST SUITE")
    print("=" * 70)

    test_count = 0
    passed = 0

    # Test 1: Build suffix array for "banana" (classic example)
    print("\n[Test 1] Classic 'banana' example")
    test_count += 1
    try:
        finder = CommentPatternFinder("banana")
        # Suffixes sorted: "a", "ana", "anana", "banana", "na", "nana"
        expected = [5, 3, 1, 0, 4, 2]
        assert finder.suffixes == expected, f"Expected {expected}, got {finder.suffixes}"
        print("✓ PASS: Suffix array correctly sorted")
        passed += 1
    except AssertionError as e:
        print(f"✗ FAIL: {e}")

    # Test 2: Search for patterns in comment text
    print("\n[Test 2] Pattern search in repeated text")
    test_count += 1
    try:
        comment_text = "will be implemented and will be done and will be fixed"
        finder = CommentPatternFinder(comment_text)

        # Find all "will be"
        positions = finder.search("will be")
        assert len(positions) == 3, f"Expected 3 occurrences, found {len(positions)}"
        assert positions == sorted(positions), "Positions should be sorted"

        # Find "implemented"
        positions = finder.search("implemented")
        assert len(positions) == 1, f"Expected 1 occurrence, found {len(positions)}"
        assert positions[0] == 8, f"Expected position 8, got {positions[0]}"

        print("✓ PASS: Pattern search works correctly")
        passed += 1
    except AssertionError as e:
        print(f"✗ FAIL: {e}")

    # Test 3: Pattern not found
    print("\n[Test 3] Pattern not found edge case")
    test_count += 1
    try:
        finder = CommentPatternFinder("hello world")
        result = finder.search("missing")
        assert result == [], f"Expected empty list, got {result}"
        print("✓ PASS: Correctly returns empty list for missing pattern")
        passed += 1
    except AssertionError as e:
        print(f"✗ FAIL: {e}")

    # Test 4: LCP array correctness
    print("\n[Test 4] LCP array computation (Kasai's algorithm)")
    test_count += 1
    try:
        finder = CommentPatternFinder("banana")
        lcp = finder.lcp_array()
        assert lcp[0] == 0, f"lcp[0] should be 0, got {lcp[0]}"
        assert lcp[2] == 1, f"lcp[2] should be 1 (a and ana share 'a'), got {lcp[2]}"
        assert lcp[3] == 3, f"lcp[3] should be 3 (ana and anana share 'ana'), got {lcp[3]}"
        print("✓ PASS: LCP array computed correctly")
        print(f"  Full LCP array: {lcp}")
        passed += 1
    except AssertionError as e:
        print(f"✗ FAIL: {e}")

    # Test 5: Find repeated substrings in misleading comments
    print("\n[Test 5] Search in copy-pasted comment patterns")
    test_count += 1
    try:
        misleading_text = "FUTURE: When CDG index is implemented this will be handled. FUTURE: When CDG index is done this will be replaced."
        finder = CommentPatternFinder(misleading_text)

        # "FUTURE: When CDG index is" should appear twice
        positions = finder.search("FUTURE: When CDG index is")
        assert len(positions) == 2, f"Expected 2 occurrences of 'FUTURE: When CDG index is', found {len(positions)}"

        # "will be" should appear twice
        positions = finder.search("will be")
        assert len(positions) == 2, f"Expected 2 occurrences of 'will be', found {len(positions)}"

        print("✓ PASS: Found repeated misleading comment patterns")
        passed += 1
    except AssertionError as e:
        print(f"✗ FAIL: {e}")

    # Test 6: Repeated substrings via LCP
    print("\n[Test 6] Repeated substrings detection via LCP")
    test_count += 1
    try:
        misleading_text = "FUTURE: When CDG index is implemented this will be handled. FUTURE: When CDG index is done this will be replaced."
        finder = CommentPatternFinder(misleading_text)
        repeated = finder.repeated_substrings(min_length=10)

        # Should find "FUTURE: When CDG index is" (or longer)
        found_phrases = [phrase for phrase, count in repeated]
        assert any("FUTURE:" in phrase or "When CDG" in phrase for phrase in found_phrases), \
            f"Should find phrases with 'FUTURE:' or 'When CDG'. Found: {found_phrases[:5]}"

        print("✓ PASS: Detected repeated patterns using LCP array")
        print(f"  Top 3 repeated patterns:")
        for phrase, count in repeated[:3]:
            print(f"    '{phrase[:50]}...' (length={len(phrase)}, count={count})")
        passed += 1
    except AssertionError as e:
        print(f"✗ FAIL: {e}")

    # Test 7: Edge cases
    print("\n[Test 7] Edge cases (empty string, single character)")
    test_count += 1
    try:
        # Empty string
        finder = CommentPatternFinder("")
        assert finder.suffixes == [], "Empty string should have empty suffix array"
        assert finder.search("any") == [], "Search in empty string should return empty"
        assert finder.lcp_array() == [], "LCP array for empty string should be empty"

        # Single character
        finder = CommentPatternFinder("a")
        assert finder.suffixes == [0], f"Single char should have suffix array [0], got {finder.suffixes}"
        assert finder.search("a") == [0], "Should find single character"
        assert finder.search("b") == [], "Should not find non-existent character"

        print("✓ PASS: Edge cases handled correctly")
        passed += 1
    except AssertionError as e:
        print(f"✗ FAIL: {e}")

    # Test 8: Real audit scenario - find copy-pasted patterns
    print("\n[Test 8] Real audit scenario - copy-pasted documentation references")
    test_count += 1
    try:
        audit_comments = """
Comment 1: FUTURE: When CDG index is implemented this will be handled at storage layer
Comment 2: See: docs/design/cdg-transactional-indexing-design.md for details
Comment 3: FUTURE: When CDG index is implemented this will be replaced
Comment 4: TODO: Add error handling
Comment 5: See: docs/design/cdg-transactional-indexing-design.md
"""
        finder = CommentPatternFinder(audit_comments)

        # Find repeated doc references
        doc_positions = finder.search("docs/design/cdg-transactional-indexing-design.md")
        assert len(doc_positions) == 2, f"Expected 2 doc references, found {len(doc_positions)}"

        # Find repeated "FUTURE: When CDG" pattern
        future_positions = finder.search("FUTURE: When CDG index is implemented")
        assert len(future_positions) == 2, f"Expected 2 FUTURE patterns, found {len(future_positions)}"

        # Get all repeated patterns of length >= 15
        repeated = finder.repeated_substrings(min_length=15)
        assert len(repeated) > 0, "Should find repeated patterns"

        print("✓ PASS: Real audit scenario working correctly")
        print(f"  Found {len(repeated)} repeated patterns (length >= 15)")
        print(f"  Top patterns:")
        for phrase, count in repeated[:5]:
            preview = phrase[:60] + "..." if len(phrase) > 60 else phrase
            print(f"    '{preview}' (count={count})")
        passed += 1
    except AssertionError as e:
        print(f"✗ FAIL: {e}")

    # Summary
    print("\n" + "=" * 70)
    print(f"TEST SUMMARY: {passed}/{test_count} tests passed")
    print("=" * 70)

    if passed == test_count:
        print("🎉 ALL TESTS PASSED!")
        return True
    else:
        print(f"⚠️  {test_count - passed} test(s) failed")
        return False


# Demo/test cases - this module doesn't use DI so standalone execution is safe
if __name__ == "__main__":
    success = run_all_tests()

    # Additional demonstration
    print("\n" + "=" * 70)
    print("DEMONSTRATION: Suffix Array Details for 'banana'")
    print("=" * 70)

    finder = CommentPatternFinder("banana")
    print(f"\nText: '{finder._text}'")
    print(f"Suffix Array: {finder.suffixes}")
    print("\nSuffixes in sorted order:")
    for i, idx in enumerate(finder.suffixes):
        suffix = finder._text[idx:]
        print(f"  [{i}] index={idx} -> '{suffix}'")

    lcp = finder.lcp_array()
    print(f"\nLCP Array: {lcp}")
    print("LCP values (common prefix with previous suffix):")
    for i in range(len(lcp)):
        if i == 0:
            print(f"  lcp[{i}] = {lcp[i]} (first element, no previous suffix)")
        else:
            idx_curr = finder.suffixes[i]
            idx_prev = finder.suffixes[i-1]
            suffix_curr = finder._text[idx_curr:]
            suffix_prev = finder._text[idx_prev:]
            common = finder._text[idx_curr:idx_curr + lcp[i]] if lcp[i] > 0 else ""
            print(f"  lcp[{i}] = {lcp[i]}: '{suffix_prev}' & '{suffix_curr}' share '{common}'")
