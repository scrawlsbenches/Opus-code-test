"""
Comment Marker Trie Implementation
Algorithm Challenge: exp-20260107-200300-trie

A Trie (Prefix Tree) for indexing comment markers in the Cortical codebase.
Supports case-insensitive insert, search, prefix matching, and deletion with cleanup.
"""

from typing import Dict, List, Optional


class CommentMarkerNode:
    """Trie node for comment marker indexing."""
    def __init__(self):
        self.children: Dict[str, 'CommentMarkerNode'] = {}
        self.is_end_of_marker: bool = False
        self.count: int = 0  # How many times this marker was seen


class CommentMarkerTrie:
    """
    Trie for efficient comment marker storage and retrieval.

    Key Design Decisions:
    1. Case-insensitive: All markers stored in lowercase for uniform matching
    2. Count tracking: Each end node tracks occurrence count
    3. Delete cleanup: Removes empty paths to prevent memory bloat
    4. O(m) operations: All operations scale with marker length, not total markers
    """

    def __init__(self):
        """Initialize empty trie for comment markers."""
        self.root = CommentMarkerNode()

    def insert(self, marker: str, count: int = 1, accumulate: bool = False) -> None:
        """
        Add a comment marker to the trie.

        marker: The marker text (e.g., "FUTURE:", "TODO:")
        count: Count value for this marker
        accumulate: If True, add to existing count; if False (default), set count

        Note: Default behavior sets/replaces count for backwards compatibility.
        Use accumulate=True when counting multiple occurrences incrementally.

        Stores lowercase internally for case-insensitive matching.
        Complexity: O(m) where m = len(marker)
        """
        # Normalize to lowercase for case-insensitive storage
        marker_lower = marker.lower()

        current = self.root
        for char in marker_lower:
            # Create node if it doesn't exist
            if char not in current.children:
                current.children[char] = CommentMarkerNode()
            current = current.children[char]

        # Mark end of marker and update count
        current.is_end_of_marker = True
        if accumulate:
            current.count += count
        else:
            current.count = count

    def search(self, marker: str) -> bool:
        """
        Return True if exact marker exists in trie.

        Case-insensitive.
        Complexity: O(m) where m = len(marker)

        Note: Must be an exact match with is_end_of_marker=True.
        Prefix-only paths return False.
        """
        marker_lower = marker.lower()

        current = self.root
        for char in marker_lower:
            if char not in current.children:
                return False
            current = current.children[char]

        # Must be a complete marker, not just a prefix
        return current.is_end_of_marker

    def starts_with(self, prefix: str) -> bool:
        """
        Return True if any marker in trie starts with prefix.

        Case-insensitive.
        Complexity: O(m) where m = len(prefix)
        """
        prefix_lower = prefix.lower()

        current = self.root
        for char in prefix_lower:
            if char not in current.children:
                return False
            current = current.children[char]

        # If we successfully traversed the prefix, at least one marker exists
        return True

    def get_all_with_prefix(self, prefix: str) -> List[str]:
        """
        Return all markers that start with the given prefix.

        Case-insensitive. Results returned in lowercase.
        Complexity: O(m + k) where m = len(prefix), k = markers found

        Algorithm:
        1. Navigate to prefix node (O(m))
        2. DFS from that node to collect all complete markers (O(k))
        """
        prefix_lower = prefix.lower()
        results = []

        # Navigate to the prefix node
        current = self.root
        for char in prefix_lower:
            if char not in current.children:
                return []  # Prefix doesn't exist
            current = current.children[char]

        # DFS to collect all markers from this point
        self._collect_markers(current, prefix_lower, results)

        return results

    def _collect_markers(self, node: CommentMarkerNode, current_prefix: str, results: List[str]) -> None:
        """
        Helper for DFS collection of all markers from a given node.

        node: Current node in traversal
        current_prefix: Path taken to reach this node
        results: List to accumulate found markers
        """
        # If this node marks the end of a marker, add it
        if node.is_end_of_marker:
            results.append(current_prefix)

        # Recursively explore all children
        for char, child_node in node.children.items():
            self._collect_markers(child_node, current_prefix + char, results)

    def delete(self, marker: str) -> bool:
        """
        Remove marker from trie.

        Return True if marker existed and was removed.
        Must clean up unused nodes (no dangling empty paths).

        Complexity: O(m) where m = len(marker)

        Algorithm:
        1. Navigate to marker while tracking path
        2. Mark node as not end-of-marker
        3. Recursively clean up empty nodes from leaf to root

        Edge cases:
        - Marker doesn't exist: return False
        - Marker shares prefix with others: only clean unique path
        - Marker is prefix of others: only unmark end, keep children
        """
        marker_lower = marker.lower()

        # First check if marker exists
        if not self.search(marker):
            return False

        # Perform deletion with cleanup
        self._delete_recursive(self.root, marker_lower, 0)
        return True

    def _delete_recursive(self, node: CommentMarkerNode, marker: str, depth: int) -> bool:
        """
        Recursive helper for deletion with node cleanup.

        Returns True if the current node should be deleted by its parent.
        This is separate from whether the deletion was successful.

        A node should be deleted if:
        - It's not a marker endpoint AND
        - It has no children
        """
        # Base case: reached end of marker
        if depth == len(marker):
            # Unmark as end of marker
            node.is_end_of_marker = False
            node.count = 0

            # Return True if this node can be deleted (no children)
            return len(node.children) == 0

        # Recursive case: navigate to next character
        char = marker[depth]
        child = node.children[char]

        # Recursively delete and check if child should be removed
        should_delete_child = self._delete_recursive(child, marker, depth + 1)

        # If child should be deleted, remove it
        if should_delete_child:
            del node.children[char]

        # Return True if current node should be deleted:
        # - Not a marker endpoint
        # - Has no children
        return not node.is_end_of_marker and len(node.children) == 0

    def get_count(self, marker: str) -> int:
        """
        Return the count for a marker, 0 if not found.

        Complexity: O(m) where m = len(marker)
        """
        marker_lower = marker.lower()

        current = self.root
        for char in marker_lower:
            if char not in current.children:
                return 0
            current = current.children[char]

        # Return count only if it's a complete marker
        if current.is_end_of_marker:
            return current.count
        return 0

    def all_markers(self) -> List[str]:
        """
        Return all markers in the trie.

        Complexity: O(n) where n = total characters in all markers
        """
        results = []
        self._collect_markers(self.root, "", results)
        return results


# =============================================================================
# TEST CASES
# =============================================================================

def run_tests():
    """Run all 8 test cases from the experiment specification."""

    print("=" * 70)
    print("Running Comment Marker Trie Tests")
    print("=" * 70)

    # Test 1: Index comment markers from audit
    print("\n[Test 1] Index comment markers from audit")
    trie = CommentMarkerTrie()
    trie.insert("FUTURE:", 10)  # 10 "FUTURE:" comments found
    trie.insert("TODO:", 5)      # 5 "TODO:" comments
    trie.insert("See:", 8)       # 8 "See:" references
    trie.insert("FIXME:", 2)
    trie.insert("NOTE:", 3)
    trie.insert("File:", 1)

    assert trie.search("FUTURE:") == True
    assert trie.search("future:") == True  # Case-insensitive
    assert trie.search("NEVER:") == False
    print("✓ Basic insert and search works")
    print("✓ Case-insensitive search works")

    # Test 2: Prefix matching for grouping
    print("\n[Test 2] Prefix matching for grouping")
    trie = CommentMarkerTrie()
    trie.insert("FUTURE:")
    trie.insert("FIXME:")
    trie.insert("File:")
    trie.insert("TODO:")

    f_markers = trie.get_all_with_prefix("F")
    assert set(f_markers) == {"future:", "fixme:", "file:"}
    assert trie.starts_with("FI") == True
    assert trie.starts_with("XX") == False
    print("✓ Prefix matching works")
    print(f"✓ Found F* markers: {f_markers}")

    # Test 3: Count tracking
    print("\n[Test 3] Count tracking")
    trie = CommentMarkerTrie()
    trie.insert("FUTURE:", 10)
    trie.insert("TODO:", 5)
    assert trie.get_count("FUTURE:") == 10
    assert trie.get_count("todo:") == 5  # Case-insensitive
    assert trie.get_count("MISSING:") == 0
    print("✓ Count tracking works")
    print(f"✓ FUTURE: count = {trie.get_count('FUTURE:')}")

    # Test 4: Delete with shared prefix cleanup
    print("\n[Test 4] Delete with shared prefix cleanup")
    trie = CommentMarkerTrie()
    trie.insert("FUTURE:")
    trie.insert("FIXME:")
    assert trie.delete("FUTURE:") == True
    assert trie.search("FUTURE:") == False
    assert trie.search("FIXME:") == True  # Should still exist
    assert trie.starts_with("FI") == True  # Prefix still valid (FIXME)
    assert trie.delete("FUTURE:") == False  # Already deleted
    print("✓ Delete works with shared prefixes")
    print("✓ FIXME: still exists after deleting FUTURE:")

    # Test 5: Delete cleans up empty paths
    print("\n[Test 5] Delete cleans up empty paths")
    trie = CommentMarkerTrie()
    trie.insert("XYZ:")  # Unique path
    trie.delete("XYZ:")
    assert trie.starts_with("X") == False  # Path should be cleaned up
    print("✓ Delete cleans up empty paths")

    # Test 6: Empty string and edge cases
    print("\n[Test 6] Empty string and edge cases")
    trie = CommentMarkerTrie()
    trie.insert("")  # Empty marker
    assert trie.search("") == True
    assert trie.get_all_with_prefix("") != []  # Returns all markers
    print("✓ Empty string edge case handled")

    # Test 7: All markers retrieval
    print("\n[Test 7] All markers retrieval")
    trie = CommentMarkerTrie()
    trie.insert("FUTURE:")
    trie.insert("TODO:")
    trie.insert("See:")
    markers = trie.all_markers()
    assert set(markers) == {"future:", "todo:", "see:"}
    print(f"✓ All markers: {sorted(markers)}")

    # Test 8: Real audit analysis scenario
    print("\n[Test 8] Real audit analysis scenario")
    trie = CommentMarkerTrie()
    misleading_markers = [
        ("FUTURE:", 10),
        ("See:", 8),
        ("will be", 7),
    ]
    for marker, count in misleading_markers:
        trie.insert(marker, count)

    speculative = trie.get_all_with_prefix("f") + trie.get_all_with_prefix("will")
    assert "future:" in speculative
    assert trie.get_count("FUTURE:") > trie.get_count("See:")
    print("✓ Audit analysis scenario works")
    print(f"✓ Speculative markers: {speculative}")

    print("\n" + "=" * 70)
    print("ALL 8 TESTS PASSED ✓")
    print("=" * 70)

    # Additional edge case demonstrations
    print("\n" + "=" * 70)
    print("Additional Edge Cases Handled")
    print("=" * 70)

    print("\n[Edge Case 1] Marker as prefix of another")
    trie = CommentMarkerTrie()
    trie.insert("FIX")
    trie.insert("FIXME:")
    assert trie.search("FIX") == True
    assert trie.search("FIXME:") == True
    trie.delete("FIX")
    assert trie.search("FIX") == False
    assert trie.search("FIXME:") == True  # Longer marker preserved
    print("✓ Deleting prefix doesn't affect longer markers")

    print("\n[Edge Case 2] Multiple deletes on shared prefix")
    trie = CommentMarkerTrie()
    trie.insert("FIXME:")
    trie.insert("FIX:")
    trie.insert("FILE:")
    trie.delete("FIXME:")
    assert set(trie.get_all_with_prefix("FI")) == {"fix:", "file:"}
    trie.delete("FIX:")
    assert trie.get_all_with_prefix("FI") == ["file:"]
    trie.delete("FILE:")
    assert trie.starts_with("F") == False  # All F* markers gone
    print("✓ Progressive deletion cleans up correctly")

    print("\n[Edge Case 3] Single character markers")
    trie = CommentMarkerTrie()
    trie.insert(":")
    assert trie.search(":") == True
    assert trie.all_markers() == [":"]
    print("✓ Single character markers work")

    print("\n" + "=" * 70)


# Demo/test cases - this module doesn't use DI so standalone execution is safe
if __name__ == "__main__":
    run_tests()
