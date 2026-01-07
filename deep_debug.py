"""Deep debug - compare actual signature values"""

from lsh_implementation import SimilarCommentFinder

lsh = SimilarCommentFinder(num_hashes=100, num_bands=20)

# Simple test case
tokens1 = {"future", "when", "cdg", "index"}
tokens2 = {"future", "when", "feature", "index"}

# These share 3 out of 4 tokens
# Exact Jaccard = 3/5 = 0.6

exact_sim = lsh.exact_jaccard(tokens1, tokens2)
print(f"Exact Jaccard: {exact_sim:.3f}")

sig1 = lsh.minhash_signature(tokens1)
sig2 = lsh.minhash_signature(tokens2)

est_sim = lsh.jaccard_similarity(sig1, sig2)
print(f"Estimated Jaccard: {est_sim:.3f}")

print(f"\nFirst 20 signature values:")
print("i  sig1         sig2         match?")
print("-" * 40)
matches = 0
for i in range(20):
    match = "✓" if sig1[i] == sig2[i] else "✗"
    if sig1[i] == sig2[i]:
        matches += 1
    print(f"{i:2d} {sig1[i]:12d} {sig2[i]:12d}   {match}")

print(f"\nMatches in first 20: {matches}/20 = {matches/20:.3f}")

# Now check what the minimum hash should be
print("\n" + "="*60)
print("Manual verification of first hash function:")
print("="*60)

a_0, b_0 = lsh._hash_coeffs[0]
PRIME = (1 << 31) - 1

print(f"Coefficients: a_0={a_0}, b_0={b_0}")
print(f"\nFor tokens1 = {tokens1}:")
min_hash_1 = float('inf')
for token in tokens1:
    token_hash = lsh._token_hash(token)
    hash_val = (a_0 * token_hash + b_0) % PRIME
    print(f"  {token:10s}: token_hash={token_hash:12d}, h_0={hash_val:12d}")
    min_hash_1 = min(min_hash_1, hash_val)
print(f"  Minimum: {min_hash_1}")
print(f"  sig1[0]: {sig1[0]}")
print(f"  Match: {min_hash_1 == sig1[0]}")

print(f"\nFor tokens2 = {tokens2}:")
min_hash_2 = float('inf')
for token in tokens2:
    token_hash = lsh._token_hash(token)
    hash_val = (a_0 * token_hash + b_0) % PRIME
    print(f"  {token:10s}: token_hash={token_hash:12d}, h_0={hash_val:12d}")
    min_hash_2 = min(min_hash_2, hash_val)
print(f"  Minimum: {min_hash_2}")
print(f"  sig2[0]: {sig2[0]}")
print(f"  Match: {min_hash_2 == sig2[0]}")

print(f"\nDo sig1[0] and sig2[0] match? {sig1[0] == sig2[0]}")
print("Should they match? Only if they pick the same minimum element")
print("from their respective sets.")
