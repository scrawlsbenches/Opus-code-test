"""
Additional Edge Case Demonstrations for Count-Min Sketch

Beyond the 9 required tests, these demonstrate robust handling of edge cases.
"""

from pattern_frequency_sketch import PatternFrequencySketch


def edge_case_1_empty_string():
    """Edge Case 1: Empty string pattern"""
    print("\n=== Edge Case 1: Empty string pattern ===")
    cms = PatternFrequencySketch(width=100, depth=3)
    cms.add("", 5)
    result = cms.query("")
    print(f"  Empty string added with count=5")
    print(f"  Query result: {result} (expected >= 5)")
    assert result >= 5
    print("  ✓ Handles empty strings correctly")
    return True


def edge_case_2_unicode_patterns():
    """Edge Case 2: Unicode patterns"""
    print("\n=== Edge Case 2: Unicode patterns ===")
    cms = PatternFrequencySketch(width=100, depth=3)
    patterns = ["TODO:", "待辦:", "할일:", "TODO:📝"]
    for pattern in patterns:
        cms.add(pattern, 10)

    for pattern in patterns:
        result = cms.query(pattern)
        print(f"  '{pattern}' -> {result} (expected >= 10)")
        assert result >= 10

    print("  ✓ Handles Unicode patterns correctly")
    return True


def edge_case_3_zero_count_add():
    """Edge Case 3: Adding with count=0"""
    print("\n=== Edge Case 3: Zero count ===")
    cms = PatternFrequencySketch(width=100, depth=3)
    cms.add("test", 0)
    result = cms.query("test")
    print(f"  Added 'test' with count=0")
    print(f"  Query result: {result} (expected 0)")
    assert result == 0
    assert cms.total_count == 0
    print("  ✓ Zero count handled correctly")
    return True


def edge_case_4_very_large_counts():
    """Edge Case 4: Very large counts (billions)"""
    print("\n=== Edge Case 4: Very large counts ===")
    cms = PatternFrequencySketch(width=1000, depth=5)
    large_count = 1_000_000_000  # 1 billion
    cms.add("popular_pattern", large_count)
    result = cms.query("popular_pattern")
    print(f"  Added pattern with count={large_count:,}")
    print(f"  Query result: {result:,} (expected >= {large_count:,})")
    assert result >= large_count
    assert cms.total_count == large_count
    print("  ✓ Large counts handled correctly")
    return True


def edge_case_5_width_equals_1():
    """Edge Case 5: Minimal width (width=1)"""
    print("\n=== Edge Case 5: Minimal width (width=1) ===")
    cms = PatternFrequencySketch(width=1, depth=5)
    # All patterns hash to same bucket - extreme collision
    cms.add("a", 10)
    cms.add("b", 20)
    cms.add("c", 30)

    # With width=1, all patterns collide
    # Query should return total count in the single bucket
    result_a = cms.query("a")
    result_b = cms.query("b")
    result_c = cms.query("c")

    print(f"  All patterns hash to same bucket (width=1)")
    print(f"  query('a') -> {result_a} (actual: 10, but collides with b,c)")
    print(f"  query('b') -> {result_b} (actual: 20, but collides with a,c)")
    print(f"  query('c') -> {result_c} (actual: 30, but collides with a,b)")
    print(f"  All should return total count: {cms.total_count}")

    # With extreme collision, all queries return the sum
    # This is the worst-case scenario
    assert result_a >= 10
    assert result_b >= 20
    assert result_c >= 30

    print("  ✓ Extreme collision scenario handled (never underestimates)")
    return True


def edge_case_6_depth_equals_1():
    """Edge Case 6: Minimal depth (depth=1)"""
    print("\n=== Edge Case 6: Minimal depth (depth=1) ===")
    cms = PatternFrequencySketch(width=100, depth=1)
    cms.add("test", 10)
    result = cms.query("test")
    print(f"  Single hash function (depth=1)")
    print(f"  Added 'test' with count=10")
    print(f"  Query result: {result} (expected >= 10)")
    assert result >= 10
    print("  ✓ Single depth handled correctly")
    return True


def edge_case_7_long_pattern_strings():
    """Edge Case 7: Very long pattern strings"""
    print("\n=== Edge Case 7: Long pattern strings ===")
    cms = PatternFrequencySketch(width=1000, depth=5)
    long_pattern = "A" * 10000  # 10K character pattern
    cms.add(long_pattern, 5)
    result = cms.query(long_pattern)
    print(f"  Pattern length: {len(long_pattern)} characters")
    print(f"  Query result: {result} (expected >= 5)")
    assert result >= 5
    print("  ✓ Long patterns handled correctly")
    return True


def edge_case_8_special_characters():
    """Edge Case 8: Special characters and whitespace"""
    print("\n=== Edge Case 8: Special characters ===")
    cms = PatternFrequencySketch(width=100, depth=3)
    patterns = [
        "TODO:\t",        # Tab
        "TODO:\n",        # Newline
        "TODO:  ",        # Multiple spaces
        "TODO:\x00",      # Null byte
        "!@#$%^&*()",     # Special chars
        "path/to/file",   # Slashes
        "key=value",      # Equals
    ]

    for i, pattern in enumerate(patterns, 1):
        cms.add(pattern, i)

    for i, pattern in enumerate(patterns, 1):
        result = cms.query(pattern)
        print(f"  '{repr(pattern)}' -> {result} (expected >= {i})")
        assert result >= i

    print("  ✓ Special characters handled correctly")
    return True


def edge_case_9_merge_empty_sketches():
    """Edge Case 9: Merge with empty sketch"""
    print("\n=== Edge Case 9: Merge empty sketches ===")
    cms1 = PatternFrequencySketch(width=100, depth=3)
    cms2 = PatternFrequencySketch(width=100, depth=3)

    # cms1 has data, cms2 is empty
    cms1.add("test", 10)

    merged = cms1.merge(cms2)
    result = merged.query("test")

    print(f"  cms1 has 'test'=10, cms2 is empty")
    print(f"  merged.query('test') -> {result} (expected >= 10)")
    assert result >= 10
    assert merged.total_count == 10

    # Both empty
    cms3 = PatternFrequencySketch(width=100, depth=3)
    cms4 = PatternFrequencySketch(width=100, depth=3)
    merged_empty = cms3.merge(cms4)
    assert merged_empty.total_count == 0

    print("  ✓ Empty sketch merging handled correctly")
    return True


def edge_case_10_collision_analysis():
    """Edge Case 10: Analyze collision behavior"""
    print("\n=== Edge Case 10: Collision analysis ===")

    # Create sketch with moderate width
    cms = PatternFrequencySketch(width=100, depth=5)

    # Add 1000 distinct patterns
    num_patterns = 1000
    for i in range(num_patterns):
        cms.add(f"pattern_{i}", 1)

    # Query a few and see overestimation
    samples = [0, 100, 500, 999]
    overestimates = []

    for i in samples:
        estimate = cms.query(f"pattern_{i}")
        overestimate_factor = estimate / 1.0  # Actual is 1
        overestimates.append(overestimate_factor)
        print(f"  pattern_{i}: estimate={estimate}, overestimate={overestimate_factor:.2f}x")

    avg_overestimate = sum(overestimates) / len(overestimates)
    print(f"  Average overestimate: {avg_overestimate:.2f}x")
    print(f"  Expected overestimate: ~{num_patterns / 100:.0f}x (based on N/w)")

    # All should never underestimate
    for i in samples:
        assert cms.query(f"pattern_{i}") >= 1

    print("  ✓ Collision behavior as expected (never underestimates)")
    return True


def run_edge_case_tests():
    """Run all edge case tests"""
    print("=" * 70)
    print("Count-Min Sketch - Edge Case Demonstrations")
    print("=" * 70)

    tests = [
        edge_case_1_empty_string,
        edge_case_2_unicode_patterns,
        edge_case_3_zero_count_add,
        edge_case_4_very_large_counts,
        edge_case_5_width_equals_1,
        edge_case_6_depth_equals_1,
        edge_case_7_long_pattern_strings,
        edge_case_8_special_characters,
        edge_case_9_merge_empty_sketches,
        edge_case_10_collision_analysis,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"  ✗ FAIL - Exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test.__name__, False))

    print("\n" + "=" * 70)
    print("EDGE CASE SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test_name}: {status}")

    print(f"\nTotal: {passed}/{total} edge case tests passed")

    if passed == total:
        print("\n🎉 ALL EDGE CASES HANDLED CORRECTLY!")
        return True
    else:
        print(f"\n⚠️  {total - passed} edge case(s) failed")
        return False


if __name__ == "__main__":
    success = run_edge_case_tests()
    exit(0 if success else 1)
