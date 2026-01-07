"""Debug LSH Test 8 failure"""

from lsh_implementation import SimilarCommentFinder

# Test 8: Real audit scenario
print("Test 8 Debug: Real audit scenario")
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

# Check exact similarities first
print("\nExact Jaccard similarities:")
for comment_id, tokens in misleading_comments.items():
    exact_sim = lsh.exact_jaccard(new_comment, tokens)
    print(f"  {comment_id}: {exact_sim:.3f}")

# Check MinHash estimated similarities
print("\nMinHash estimated similarities:")
query_sig = lsh.minhash_signature(new_comment)
for comment_id in misleading_comments:
    doc_sig = lsh._signatures[comment_id]
    est_sim = lsh.jaccard_similarity(query_sig, doc_sig)
    print(f"  {comment_id}: {est_sim:.3f}")

# Now query with LSH
print("\nLSH query results (threshold=0.3):")
similar = lsh.query(new_comment, threshold=0.3)
if similar:
    for doc_id, sim in similar:
        print(f"  {doc_id}: {sim:.3f}")
else:
    print("  No results found!")

# Debug: Check band matches
print("\nBand match debugging:")
for band_idx in range(lsh._num_bands):
    start_idx = band_idx * lsh._rows_per_band
    end_idx = start_idx + lsh._rows_per_band
    query_band = query_sig[start_idx:end_idx]
    query_bucket = hash(query_band)

    # Check if this bucket exists and has docs
    if band_idx in lsh._buckets and query_bucket in lsh._buckets[band_idx]:
        docs_in_bucket = lsh._buckets[band_idx][query_bucket]
        print(f"  Band {band_idx}: Found {len(docs_in_bucket)} docs in bucket: {docs_in_bucket}")

# Alternative: brute force check all docs
print("\nBrute force check (no LSH, all docs):")
results_brute = []
for comment_id in misleading_comments:
    doc_sig = lsh._signatures[comment_id]
    sim = lsh.jaccard_similarity(query_sig, doc_sig)
    if sim >= 0.3:
        results_brute.append((comment_id, sim))
results_brute.sort(key=lambda x: x[1], reverse=True)
for doc_id, sim in results_brute:
    print(f"  {doc_id}: {sim:.3f}")
