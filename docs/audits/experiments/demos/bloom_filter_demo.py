"""
Bloom Filter Demonstration - Edge Cases and Properties

This script demonstrates various edge cases and interesting properties
of the Bloom Filter implementation.
"""

from bloom_filter_impl import SuspiciousCommentFilter


def demo_edge_cases():
    """Demonstrate various edge cases"""
    print("=" * 70)
    print("EDGE CASE DEMONSTRATIONS")
    print("=" * 70)

    # Edge Case 1: Very small filter
    print("\n1. Very Small Filter (1 expected pattern)")
    bf_small = SuspiciousCommentFilter(expected_patterns=1, fp_rate=0.01)
    print(f"   Size: {bf_small.size} bits")
    print(f"   Hash count: {bf_small.hash_count}")
    bf_small.add("single_pattern")
    print(f"   Can find added pattern: {bf_small.probably_suspicious('single_pattern')}")
    print(f"   Other pattern: {bf_small.probably_suspicious('other')}")

    # Edge Case 2: Very large filter
    print("\n2. Large Filter (10,000 expected patterns)")
    bf_large = SuspiciousCommentFilter(expected_patterns=10000, fp_rate=0.001)
    print(f"   Size: {bf_large.size} bits ({bf_large.size / 8:.1f} bytes)")
    print(f"   Hash count: {bf_large.hash_count}")

    # Edge Case 3: Very strict FP rate
    print("\n3. Very Strict FP Rate (0.001 = 0.1%)")
    bf_strict = SuspiciousCommentFilter(expected_patterns=100, fp_rate=0.001)
    print(f"   Size: {bf_strict.size} bits (vs 958 for fp_rate=0.01)")
    print(f"   Hash count: {bf_strict.hash_count}")

    # Edge Case 4: Special characters and unicode
    print("\n4. Special Characters and Unicode")
    bf_unicode = SuspiciousCommentFilter(expected_patterns=10, fp_rate=0.01)
    special_patterns = [
        "emoji: 🚀 rocket",
        "newline:\ntext",
        "tab:\ttext",
        "quote: \"quoted\"",
        "backslash: \\path\\to\\file",
        "中文字符",  # Chinese characters
        "العربية",  # Arabic
    ]
    for pattern in special_patterns:
        bf_unicode.add(pattern)
        found = bf_unicode.probably_suspicious(pattern)
        print(f"   '{pattern[:20]}...': {found}")

    # Edge Case 5: Very long strings
    print("\n5. Very Long Strings")
    bf_long = SuspiciousCommentFilter(expected_patterns=10, fp_rate=0.01)
    long_pattern = "a" * 10000  # 10,000 character string
    bf_long.add(long_pattern)
    print(f"   Added 10,000 char string")
    print(f"   Can find it: {bf_long.probably_suspicious(long_pattern)}")
    print(f"   Different long string: {bf_long.probably_suspicious('b' * 10000)}")

    # Edge Case 6: Fill ratio and FP rate relationship
    print("\n6. Fill Ratio and FP Rate Relationship")
    bf_fill = SuspiciousCommentFilter(expected_patterns=100, fp_rate=0.01)
    print(f"   Initial FP rate: {bf_fill.false_positive_rate():.4f}")

    for i in [10, 25, 50, 75, 100, 150, 200]:
        bf_test = SuspiciousCommentFilter(expected_patterns=100, fp_rate=0.01)
        for j in range(i):
            bf_test.add(f"pattern_{j}")
        print(f"   After adding {i:3d} items: FP rate = {bf_test.false_positive_rate():.4f}")

    # Edge Case 7: Identical patterns (idempotency)
    print("\n7. Idempotency (adding same pattern multiple times)")
    bf_idem = SuspiciousCommentFilter(expected_patterns=10, fp_rate=0.01)
    bf_idem.add("duplicate")
    fp_rate_1 = bf_idem.false_positive_rate()
    bf_idem.add("duplicate")
    fp_rate_2 = bf_idem.false_positive_rate()
    print(f"   FP rate after 1st add: {fp_rate_1:.4f}")
    print(f"   FP rate after 2nd add: {fp_rate_2:.4f}")
    print(f"   (Note: FP rate increases because _items_added counts duplicates)")


def demo_hash_distribution():
    """Demonstrate hash function distribution"""
    print("\n" + "=" * 70)
    print("HASH DISTRIBUTION ANALYSIS")
    print("=" * 70)

    bf = SuspiciousCommentFilter(expected_patterns=100, fp_rate=0.01)

    # Add 100 patterns and track which bits get set
    bit_set_count = [0] * bf.size

    for i in range(100):
        pattern = f"pattern_{i}"
        # Manually track which bits would be set
        for seed in range(bf.hash_count):
            index = bf._hash(pattern, seed)
            bit_set_count[index] += 1

    # Analyze distribution
    max_collisions = max(bit_set_count)
    min_collisions = min(bit_set_count)
    avg_collisions = sum(bit_set_count) / len(bit_set_count)
    bits_set = sum(1 for count in bit_set_count if count > 0)

    print(f"\nAfter adding 100 patterns with {bf.hash_count} hash functions:")
    print(f"  Total bits: {bf.size}")
    print(f"  Bits set: {bits_set} ({bits_set / bf.size * 100:.1f}%)")
    print(f"  Bits unset: {bf.size - bits_set} ({(bf.size - bits_set) / bf.size * 100:.1f}%)")
    print(f"  Max collisions on single bit: {max_collisions}")
    print(f"  Min collisions on single bit: {min_collisions}")
    print(f"  Avg collisions per bit: {avg_collisions:.2f}")

    # Show distribution histogram
    print("\n  Collision histogram:")
    histogram = {}
    for count in bit_set_count:
        histogram[count] = histogram.get(count, 0) + 1

    for collision_count in sorted(histogram.keys()):
        bar_length = histogram[collision_count] // 5
        bar = "█" * bar_length
        print(f"    {collision_count:2d} collisions: {histogram[collision_count]:3d} bits {bar}")


def demo_performance_characteristics():
    """Demonstrate space efficiency"""
    print("\n" + "=" * 70)
    print("SPACE EFFICIENCY COMPARISON")
    print("=" * 70)

    patterns_count = 1000
    avg_pattern_length = 30  # bytes

    # Naive storage: store all patterns
    naive_storage = patterns_count * avg_pattern_length
    print(f"\nNaive approach (store all {patterns_count} patterns):")
    print(f"  Storage: {naive_storage:,} bytes ({naive_storage / 1024:.1f} KB)")

    # Bloom filter storage
    bf = SuspiciousCommentFilter(expected_patterns=patterns_count, fp_rate=0.01)
    bloom_storage = bf.size / 8  # bits to bytes
    print(f"\nBloom filter (FP rate = 1%):")
    print(f"  Storage: {bloom_storage:.0f} bytes ({bloom_storage / 1024:.1f} KB)")
    print(f"  Space savings: {(1 - bloom_storage / naive_storage) * 100:.1f}%")

    # Different FP rates
    print("\nBloom filter space usage for different FP rates:")
    for fp_rate in [0.1, 0.05, 0.01, 0.005, 0.001]:
        bf_test = SuspiciousCommentFilter(expected_patterns=patterns_count, fp_rate=fp_rate)
        storage_bytes = bf_test.size / 8
        print(f"  FP rate {fp_rate:.3f} ({fp_rate*100:5.1f}%): {storage_bytes:6.0f} bytes "
              f"({storage_bytes / 1024:5.1f} KB) - {bf_test.hash_count} hash functions")


def main():
    """Run all demonstrations"""
    demo_edge_cases()
    demo_hash_distribution()
    demo_performance_characteristics()

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
