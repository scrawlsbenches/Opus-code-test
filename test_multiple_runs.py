"""Test if Test 8 passes on multiple runs (show it's probabilistic)"""

from lsh_implementation import SimilarCommentFinder

# Run Test 8 multiple times
num_trials = 20
successes = 0

for trial in range(num_trials):
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

    if "F001" in found_ids or "F003" in found_ids:
        successes += 1
        status = "PASS"
    else:
        status = "FAIL"

    print(f"Trial {trial+1:2d}: {status} - Found {found_ids}")

print(f"\nSuccess rate: {successes}/{num_trials} = {100*successes/num_trials:.1f}%")
print(f"Expected: ~60% (based on probability calculation)")
