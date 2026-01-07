"""Check if hash functions are diverse enough"""

from lsh_implementation import SimilarCommentFinder

lsh = SimilarCommentFinder(num_hashes=100, num_bands=20)

print("First 10 hash function coefficients:")
PRIME = (1 << 31) - 1
for i in range(10):
    a_i, b_i = lsh._hash_coeffs[i]
    print(f"  h_{i}: a={a_i:15d}, b={b_i:15d}")

# Check if coefficients are distinct
a_values = [a for a, b in lsh._hash_coeffs]
b_values = [b for a, b in lsh._hash_coeffs]

print(f"\nUnique a values: {len(set(a_values))}/100")
print(f"Unique b values: {len(set(b_values))}/100")

# Test on a simple example
tokens = {"a", "b", "c"}
sig = lsh.minhash_signature(tokens)

print(f"\nSignature for {tokens}:")
print(f"  First 10 values: {sig[:10]}")
print(f"  All values unique: {len(set(sig)) == len(sig)}")
print(f"  Number of unique values: {len(set(sig))}/100")
