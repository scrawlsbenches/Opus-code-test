"""
Test suite for Bloom Filter implementation

All test cases from exp-20260107-200600-bloom-filter.md
"""

from bloom_filter_impl import SuspiciousCommentFilter


def test_1_add_known_misleading_patterns():
    """Test 1: Add known misleading patterns from audit"""
    print("\n=== Test 1: Add known misleading patterns from audit ===")

    bf = SuspiciousCommentFilter(expected_patterns=20, fp_rate=0.01)

    # Real patterns from our 10 misleading comments
    suspicious_patterns = [
        "future when implemented",
        "will be handled",
        "will be replaced",
        "see docs design",
        "when cdg is done",
        "when feature implemented",
        "future this will",
        "storage layer recovery",
        "see missing file",
        "will be done later",
    ]

    for pattern in suspicious_patterns:
        bf.add(pattern)

    # All added patterns should be found (no false negatives!)
    all_found = True
    for pattern in suspicious_patterns:
        found = bf.probably_suspicious(pattern)
        if not found:
            print(f"  ❌ False negative for '{pattern}'!")
            all_found = False
        else:
            print(f"  ✓ Found '{pattern}'")

    assert all_found, "Some patterns were not found (false negatives)!"
    print("✅ Test 1 PASSED - All patterns found, no false negatives")


def test_2_no_false_negatives():
    """Test 2: No false negatives (CRITICAL property)"""
    print("\n=== Test 2: No false negatives with 100 patterns ===")

    bf = SuspiciousCommentFilter(expected_patterns=100, fp_rate=0.01)
    patterns = [f"pattern_{i}" for i in range(100)]

    for pattern in patterns:
        bf.add(pattern)

    # Every added pattern MUST be found
    false_negatives = []
    for pattern in patterns:
        if not bf.probably_suspicious(pattern):
            false_negatives.append(pattern)

    if false_negatives:
        print(f"  ❌ Found {len(false_negatives)} false negatives!")
        for pattern in false_negatives[:5]:  # Show first 5
            print(f"    - {pattern}")
        assert False, f"False negatives detected: {false_negatives[:5]}"

    print(f"  ✓ All 100 patterns found correctly")
    print("✅ Test 2 PASSED - No false negatives")


def test_3_false_positive_rate():
    """Test 3: False positive rate is reasonable"""
    print("\n=== Test 3: False positive rate is reasonable ===")

    bf = SuspiciousCommentFilter(expected_patterns=100, fp_rate=0.05)  # 5% target

    for i in range(100):
        bf.add(f"suspicious_{i}")

    # Test with patterns that were never added
    false_positives = 0
    test_count = 10000

    for i in range(test_count):
        if bf.probably_suspicious(f"innocent_{i}"):
            false_positives += 1

    actual_fp_rate = false_positives / test_count
    estimated_fp_rate = bf.false_positive_rate()

    print(f"  Target FP rate: 5.00%")
    print(f"  Estimated FP rate: {estimated_fp_rate:.2%}")
    print(f"  Actual FP rate: {actual_fp_rate:.2%}")
    print(f"  False positives: {false_positives}/{test_count}")

    # Should be roughly around target (allow 3x margin for randomness)
    assert actual_fp_rate < 0.15, f"FP rate {actual_fp_rate:.2%} too high (>15%)!"
    print(f"✅ Test 3 PASSED - FP rate {actual_fp_rate:.2%} is acceptable (<15%)")


def test_4_size_calculation():
    """Test 4: Size calculation is reasonable"""
    print("\n=== Test 4: Size calculation is reasonable ===")

    bf = SuspiciousCommentFilter(expected_patterns=100, fp_rate=0.01)

    # For n=100, p=0.01:
    # m = -100 * ln(0.01) / (ln(2)^2) ≈ 959 bits
    # k = (959/100) * ln(2) ≈ 6.65 → 7 hash functions

    print(f"  Bit array size: {bf.size}")
    print(f"  Hash count: {bf.hash_count}")

    assert bf.size > 500, f"Size {bf.size} too small (expected ~959)"
    assert bf.size < 2000, f"Size {bf.size} too large (expected ~959)"
    assert bf.hash_count >= 3, f"Hash count {bf.hash_count} too few (min 3)"
    assert bf.hash_count <= 15, f"Hash count {bf.hash_count} too many (expected ~7)"

    print(f"✅ Test 4 PASSED - Size and hash count in reasonable ranges")


def test_5_deterministic_behavior():
    """Test 5: Deterministic behavior (same input = same result)"""
    print("\n=== Test 5: Deterministic behavior ===")

    bf = SuspiciousCommentFilter(expected_patterns=10, fp_rate=0.01)
    bf.add("test_pattern")

    result1 = bf.probably_suspicious("test_pattern")
    result2 = bf.probably_suspicious("test_pattern")
    result3 = bf.probably_suspicious("other_pattern")

    print(f"  First query for 'test_pattern': {result1}")
    print(f"  Second query for 'test_pattern': {result2}")
    print(f"  Query for 'other_pattern': {result3}")

    assert result1 == True, "Added pattern not found"
    assert result2 == True, "Added pattern not found on second query"
    assert result1 == result2, "Non-deterministic behavior detected!"

    print("✅ Test 5 PASSED - Deterministic behavior confirmed")


def test_6_real_audit_scenario():
    """Test 6: Real audit scenario - screening comments"""
    print("\n=== Test 6: Real audit scenario - screening comments ===")

    bf = SuspiciousCommentFilter(expected_patterns=20, fp_rate=0.01)

    # Real patterns from our 10 misleading comments
    suspicious_patterns = [
        "future when implemented",
        "will be handled",
        "will be replaced",
        "see docs design",
        "when cdg is done",
        "when feature implemented",
        "future this will",
        "storage layer recovery",
        "see missing file",
        "will be done later",
    ]

    # Add our known misleading patterns
    for pattern in suspicious_patterns:
        bf.add(pattern)

    # Simulate screening new comments
    new_comments = [
        "future when feature is done",  # Similar to suspicious
        "todo fix this bug",  # Safe pattern
        "fixme handle edge case",  # Safe pattern
        "will be implemented soon",  # Similar to suspicious
        "returns the count of items",  # Safe pattern
    ]

    # Check which need review
    needs_review = []
    safe = []

    for comment in new_comments:
        if bf.probably_suspicious(comment):
            needs_review.append(comment)
            print(f"  🔍 Needs review: '{comment}'")
        else:
            safe.append(comment)
            print(f"  ✓ Safe: '{comment}'")

    print(f"\n  Total flagged for review: {len(needs_review)}/{len(new_comments)}")
    print(f"  Estimated FP rate: {bf.false_positive_rate():.2%}")

    # This is more of an integration test - we're checking the system works
    # We don't assert specific results since we don't know which will be FPs
    print("✅ Test 6 PASSED - Screening system operational")


def test_7_empty_filter():
    """Test 7: Empty filter behavior"""
    print("\n=== Test 7: Empty filter behavior ===")

    bf = SuspiciousCommentFilter(expected_patterns=10, fp_rate=0.01)

    # Nothing added yet
    result = bf.probably_suspicious("anything")
    fp_rate = bf.false_positive_rate()

    print(f"  Query on empty filter: {result}")
    print(f"  FP rate on empty filter: {fp_rate}")

    assert result == False, "Empty filter should return False for any query"
    assert fp_rate == 0.0, "Empty filter should have 0.0 FP rate"

    print("✅ Test 7 PASSED - Empty filter behaves correctly")


def test_8_empty_string():
    """Test 8: Edge case - empty string"""
    print("\n=== Test 8: Edge case - empty string ===")

    bf = SuspiciousCommentFilter(expected_patterns=10, fp_rate=0.01)

    bf.add("")  # Empty pattern

    result = bf.probably_suspicious("")

    print(f"  Added empty string: True")
    print(f"  Query for empty string: {result}")

    assert result == True, "Should find empty string after adding it"

    print("✅ Test 8 PASSED - Empty string handled correctly")


def run_all_tests():
    """Run all test cases"""
    print("=" * 70)
    print("BLOOM FILTER TEST SUITE")
    print("=" * 70)

    tests = [
        test_1_add_known_misleading_patterns,
        test_2_no_false_negatives,
        test_3_false_positive_rate,
        test_4_size_calculation,
        test_5_deterministic_behavior,
        test_6_real_audit_scenario,
        test_7_empty_filter,
        test_8_empty_string,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"❌ {test_func.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} ERROR: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    print(f"Success rate: {passed / len(tests) * 100:.1f}%")
    print("=" * 70)

    return passed, failed


if __name__ == "__main__":
    passed, failed = run_all_tests()
    exit(0 if failed == 0 else 1)
