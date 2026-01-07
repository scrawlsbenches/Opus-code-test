"""Deep debug of LSH banding"""

from lsh_implementation import SimilarCommentFinder

lsh = SimilarCommentFinder(num_hashes=100, num_bands=20)

# Just compare two similar documents
tokens1 = {"future", "when", "cdg", "index", "is", "implemented", "this", "will", "be", "handled", "at", "storage", "layer"}
tokens2 = {"future", "when", "feature", "implemented", "this", "will", "be", "handled"}

# Add first doc
lsh.add("F001", tokens1)

# Compute signatures
sig1 = lsh._signatures["F001"]
sig2 = lsh.minhash_signature(tokens2)

print(f"Exact Jaccard: {lsh.exact_jaccard(tokens1, tokens2):.3f}")
print(f"MinHash estimate: {lsh.jaccard_similarity(sig1, sig2):.3f}")
print(f"\nBand-by-band analysis (rows_per_band={lsh._rows_per_band}):")

band_matches = 0
for band_idx in range(lsh._num_bands):
    start_idx = band_idx * lsh._rows_per_band
    end_idx = start_idx + lsh._rows_per_band

    band1 = sig1[start_idx:end_idx]
    band2 = sig2[start_idx:end_idx]

    # Count matching positions in band
    matches_in_band = sum(1 for i in range(len(band1)) if band1[i] == band2[i])

    exact_match = (band1 == band2)
    if exact_match:
        band_matches += 1
        print(f"  Band {band_idx:2d}: EXACT MATCH ({matches_in_band}/5 values match)")
    elif matches_in_band >= 4:
        print(f"  Band {band_idx:2d}: {matches_in_band}/5 values match (not exact)")

print(f"\nTotal bands with exact match: {band_matches}/20")
print(f"Expected probability of at least 1 match: ~1-(1-0.54^5)^20 = {1-(1-0.54**5)**20:.3f}")
