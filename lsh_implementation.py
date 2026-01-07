"""
Locality Sensitive Hashing (LSH) with MinHash for Similar Comment Detection

This implementation uses:
- MinHash for Jaccard similarity estimation
- Banding technique for candidate generation
- Deterministic hash function generation using linear congruential formula
"""

from typing import Dict, List, Set, Tuple
import hashlib
import math


class SimilarCommentFinder:
    def __init__(self, num_hashes: int = 100, num_bands: int = 20):
        """
        Initialize LSH index for similar comment detection.
        num_hashes: Number of hash functions for MinHash signature (must be divisible by num_bands)
        num_bands: Number of bands for LSH

        The relationship:
        - rows_per_band = num_hashes / num_bands
        - Approximate threshold ≈ (1/num_bands)^(1/rows_per_band)
        - For 100 hashes, 20 bands: rows=5, threshold ≈ 0.55

        Hash function generation:
        We use the formula: h_i(x) = (a_i * hash(x) + b_i) % PRIME
        where a_i and b_i are coefficients derived deterministically from seed i.
        """
        self._num_hashes = num_hashes
        self._num_bands = num_bands
        self._rows_per_band = num_hashes // num_bands
        self._hash_coeffs: List[Tuple[int, int]] = []  # (a, b) pairs for each hash
        self._buckets: Dict[int, Dict[int, Set[str]]] = {}  # band -> bucket_hash -> {doc_ids}
        self._signatures: Dict[str, Tuple[int, ...]] = {}  # doc_id -> signature
        self._documents: Dict[str, Set[str]] = {}  # doc_id -> original tokens
        self._initialize_hash_functions()

    def _initialize_hash_functions(self) -> None:
        """
        Generate num_hashes hash function coefficients.
        Use deterministic seeding for reproducibility.

        For each hash function i:
            a_i = (i * 0x5DEECE66D + 0xB) % PRIME  # Linear congruential
            b_i = ((i + 1) * 0x5DEECE66D + 0xB) % PRIME

        Where PRIME = 2^31 - 1 (Mersenne prime for good distribution)
        """
        PRIME = (1 << 31) - 1  # 2^31 - 1 = 2147483647

        for i in range(self._num_hashes):
            # Linear congruential formula for coefficients
            a_i = (i * 0x5DEECE66D + 0xB) % PRIME
            b_i = ((i + 1) * 0x5DEECE66D + 0xB) % PRIME
            self._hash_coeffs.append((a_i, b_i))

    def _token_hash(self, token: str) -> int:
        """
        Hash a token to an integer.
        Use hashlib.md5 for consistent cross-platform behavior.
        Return a positive integer.
        """
        # Use MD5 for consistent hashing
        hash_bytes = hashlib.md5(token.encode('utf-8')).digest()
        # Convert first 8 bytes to integer
        hash_int = int.from_bytes(hash_bytes[:8], byteorder='big')
        # Return positive integer
        return abs(hash_int)

    def minhash_signature(self, tokens: Set[str]) -> Tuple[int, ...]:
        """
        Compute MinHash signature for a set of tokens.
        Returns tuple of num_hashes minimum hash values.

        For each hash function h_i with coefficients (a_i, b_i):
            sig[i] = min((a_i * token_hash(t) + b_i) % PRIME for t in tokens)

        If tokens is empty, return tuple of MAX_INT values.
        """
        PRIME = (1 << 31) - 1
        MAX_INT = float('inf')

        # Handle empty set case
        if not tokens:
            return tuple([2**31 - 1] * self._num_hashes)

        signature = []

        # For each hash function
        for a_i, b_i in self._hash_coeffs:
            min_hash = MAX_INT

            # Compute hash for each token and find minimum
            for token in tokens:
                token_hash_val = self._token_hash(token)
                hash_val = (a_i * token_hash_val + b_i) % PRIME
                min_hash = min(min_hash, hash_val)

            signature.append(int(min_hash))

        return tuple(signature)

    def _band_hash(self, band_signature: Tuple[int, ...]) -> int:
        """
        Hash a band signature to a bucket ID.
        Use hashlib for consistent, deterministic hashing.
        """
        # Convert band signature to bytes and hash
        band_str = ','.join(str(x) for x in band_signature)
        hash_bytes = hashlib.md5(band_str.encode('utf-8')).digest()
        return int.from_bytes(hash_bytes[:8], byteorder='big')

    def add(self, comment_id: str, tokens: Set[str]) -> None:
        """
        Add comment to the LSH index.
        1. Compute MinHash signature
        2. For each band, hash the band's portion of signature
        3. Add comment_id to appropriate bucket
        """
        # Compute and store signature
        signature = self.minhash_signature(tokens)
        self._signatures[comment_id] = signature
        self._documents[comment_id] = tokens

        # Add to LSH buckets using banding technique
        for band_idx in range(self._num_bands):
            # Extract band portion of signature
            start_idx = band_idx * self._rows_per_band
            end_idx = start_idx + self._rows_per_band
            band_signature = signature[start_idx:end_idx]

            # Hash the band signature to get bucket using consistent hash function
            bucket_hash = self._band_hash(band_signature)

            # Initialize band bucket if needed
            if band_idx not in self._buckets:
                self._buckets[band_idx] = {}

            # Initialize bucket set if needed
            if bucket_hash not in self._buckets[band_idx]:
                self._buckets[band_idx][bucket_hash] = set()

            # Add comment to bucket
            self._buckets[band_idx][bucket_hash].add(comment_id)

    def query(self, tokens: Set[str], threshold: float = 0.5) -> List[Tuple[str, float]]:
        """
        Find comments with estimated Jaccard similarity >= threshold.
        1. Compute MinHash signature for query
        2. Find candidate comments from buckets (any band match)
        3. Compute actual signature similarity for candidates
        4. Filter by threshold and return sorted by similarity descending

        Returns list of (comment_id, estimated_similarity) pairs.
        """
        # Compute query signature
        query_signature = self.minhash_signature(tokens)

        # Find candidates from buckets
        candidates = set()

        for band_idx in range(self._num_bands):
            # Extract band portion of query signature
            start_idx = band_idx * self._rows_per_band
            end_idx = start_idx + self._rows_per_band
            band_signature = query_signature[start_idx:end_idx]

            # Hash the band signature using consistent hash function
            bucket_hash = self._band_hash(band_signature)

            # Find matching bucket
            if band_idx in self._buckets and bucket_hash in self._buckets[band_idx]:
                candidates.update(self._buckets[band_idx][bucket_hash])

        # Compute similarities for candidates
        results = []
        for candidate_id in candidates:
            candidate_signature = self._signatures[candidate_id]
            similarity = self.jaccard_similarity(query_signature, candidate_signature)

            if similarity >= threshold:
                results.append((candidate_id, similarity))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    def jaccard_similarity(self, sig1: Tuple[int, ...], sig2: Tuple[int, ...]) -> float:
        """
        Estimate Jaccard similarity from MinHash signatures.
        J ≈ (number of matching positions) / (signature length)
        """
        if len(sig1) != len(sig2):
            return 0.0

        matches = sum(1 for i in range(len(sig1)) if sig1[i] == sig2[i])
        return matches / len(sig1)

    def exact_jaccard(self, tokens1: Set[str], tokens2: Set[str]) -> float:
        """
        Compute exact Jaccard similarity between two token sets.
        J(A,B) = |A ∩ B| / |A ∪ B|
        Returns 0.0 if both sets are empty.
        """
        # Handle empty sets
        if not tokens1 and not tokens2:
            return 0.0

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        if union == 0:
            return 0.0

        return intersection / union


# Test cases
if __name__ == "__main__":
    print("Running test cases...")

    # Test 1: Identical comments have similarity 1.0
    print("\nTest 1: Identical comments")
    lsh = SimilarCommentFinder(num_hashes=100, num_bands=20)
    tokens = {"future", "when", "cdg", "index", "is", "implemented"}
    sig1 = lsh.minhash_signature(tokens)
    sig2 = lsh.minhash_signature(tokens)
    assert lsh.jaccard_similarity(sig1, sig2) == 1.0
    print("✓ PASSED: Identical comments have similarity 1.0")

    # Test 2: Disjoint comments have similarity ~0.0
    print("\nTest 2: Disjoint comments")
    lsh = SimilarCommentFinder(num_hashes=100, num_bands=20)
    misleading_tokens = {"future", "will", "be", "implemented"}
    accurate_tokens = {"todo", "fix", "bug", "now"}
    sig1 = lsh.minhash_signature(misleading_tokens)
    sig2 = lsh.minhash_signature(accurate_tokens)
    sim = lsh.jaccard_similarity(sig1, sig2)
    assert sim < 0.2, f"Similarity {sim} too high for disjoint sets"
    print(f"✓ PASSED: Disjoint comments have similarity {sim:.3f} < 0.2")

    # Test 3: Similar comments have intermediate similarity
    print("\nTest 3: Similar comments")
    lsh = SimilarCommentFinder(num_hashes=100, num_bands=20)
    # Comment 1: "FUTURE: When CDG index is implemented this will be handled"
    tokens1 = {"future", "when", "cdg", "index", "is", "implemented", "this", "will", "be", "handled"}
    # Comment 2: "FUTURE: When feature is implemented this will be done"
    tokens2 = {"future", "when", "feature", "is", "implemented", "this", "will", "be", "done"}
    # Overlap: future, when, is, implemented, this, will, be = 7
    # Union: future, when, cdg, index, feature, is, implemented, this, will, be, handled, done = 12
    # Exact J = 7/12 ≈ 0.58

    sig1 = lsh.minhash_signature(tokens1)
    sig2 = lsh.minhash_signature(tokens2)
    estimated_sim = lsh.jaccard_similarity(sig1, sig2)
    exact_sim = lsh.exact_jaccard(tokens1, tokens2)

    # Estimated should be close to exact (within 0.2 for 100 hashes)
    assert abs(estimated_sim - exact_sim) < 0.2, f"Estimate {estimated_sim} too far from exact {exact_sim}"
    print(f"✓ PASSED: Estimated sim {estimated_sim:.3f} close to exact {exact_sim:.3f}")

    # Test 4: Add and query for similar comments
    print("\nTest 4: Add and query")
    lsh = SimilarCommentFinder(num_hashes=100, num_bands=20)

    # Add known misleading comments from our audit
    lsh.add("F001", {"future", "when", "cdg", "index", "implemented", "will", "be", "handled"})
    lsh.add("F002", {"see", "docs", "design", "cdg", "transactional", "indexing", "md"})
    lsh.add("F003", {"todo", "add", "error", "handling", "edge", "case"})  # Accurate comment

    # Query with similar misleading pattern
    query_tokens = {"future", "when", "feature", "implemented", "will", "be", "done"}
    results = lsh.query(query_tokens, threshold=0.4)

    # F001 should be found (high similarity)
    result_ids = [doc_id for doc_id, sim in results]
    assert "F001" in result_ids, f"F001 not found in results: {results}"

    # F003 (accurate) should NOT be found (low similarity)
    if "F003" in result_ids:
        f003_sim = [sim for doc_id, sim in results if doc_id == "F003"][0]
        assert f003_sim < 0.4, f"F003 similarity {f003_sim} should be below threshold"

    print(f"✓ PASSED: Query found {len(results)} results, F001 in results")

    # Test 5: Query returns similarity scores sorted descending
    print("\nTest 5: Sorted results")
    lsh = SimilarCommentFinder(num_hashes=100, num_bands=20)
    lsh.add("exact", {"a", "b", "c", "d"})
    lsh.add("similar", {"a", "b", "c", "e"})
    lsh.add("different", {"x", "y", "z", "w"})

    results = lsh.query({"a", "b", "c", "d"}, threshold=0.3)
    assert len(results) >= 1
    # First result should be highest similarity
    if len(results) >= 2:
        assert results[0][1] >= results[1][1], "Results should be sorted by similarity descending"
    print(f"✓ PASSED: Results sorted ({len(results)} results)")

    # Test 6: Empty set handling
    print("\nTest 6: Empty set handling")
    lsh = SimilarCommentFinder(num_hashes=100, num_bands=20)
    sig = lsh.minhash_signature(set())
    assert len(sig) == 100  # Should return valid signature (all max values)

    # Empty vs empty should have similarity 1.0 (both map to same signature)
    # OR return 0.0 if treating empty as special case - either is acceptable
    empty_sim = lsh.jaccard_similarity(sig, sig)
    assert empty_sim in [0.0, 1.0]
    print(f"✓ PASSED: Empty set handled (similarity={empty_sim})")

    # Test 7: Hash function determinism
    print("\nTest 7: Hash function determinism")
    lsh = SimilarCommentFinder(num_hashes=100, num_bands=20)
    tokens = {"test", "tokens", "for", "hashing"}
    sig1 = lsh.minhash_signature(tokens)
    sig2 = lsh.minhash_signature(tokens)
    assert sig1 == sig2, "MinHash should be deterministic"
    print("✓ PASSED: MinHash is deterministic")

    # Test 8: Real audit scenario - find similar misleading comments
    print("\nTest 8: Real audit scenario")
    lsh = SimilarCommentFinder(num_hashes=100, num_bands=20)

    # Add all misleading comments from audit
    misleading_comments = {
        "F001": {"future", "when", "cdg", "index", "is", "implemented", "this", "will", "be", "handled", "at", "storage", "layer"},
        "F002": {"see", "docs", "design", "cdg", "transactional", "indexing", "design", "md"},
        "F003": {"future", "when", "cdg", "index", "is", "implemented", "this", "will", "be", "replaced"},
        "F004": {"will", "be", "done", "when", "feature", "is", "ready"},
    }

    for comment_id, tokens in misleading_comments.items():
        lsh.add(comment_id, tokens)

    # New potentially misleading comment
    new_comment = {"future", "when", "feature", "implemented", "this", "will", "be", "handled"}
    similar = lsh.query(new_comment, threshold=0.3)

    # Should find F001 and F003 (both are FUTURE patterns)
    found_ids = {doc_id for doc_id, sim in similar}
    assert "F001" in found_ids or "F003" in found_ids, f"Should find similar FUTURE patterns, got {found_ids}"

    # Report similarities for human review
    print("Similar misleading comments found:")
    for doc_id, sim in similar:
        print(f"  {doc_id}: {sim:.2%} similar")
    print(f"✓ PASSED: Found {len(similar)} similar patterns")

    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
