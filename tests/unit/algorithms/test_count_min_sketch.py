"""
Test suite for PatternFrequencySketch (Count-Min Sketch)

All 9 test cases from the experiment file.
"""

from cortical.audits.algorithms.count_min_sketch import PatternFrequencySketch


def test_1_basic_add_and_query():
    """Test 1: Basic add and query with audit patterns"""
    print("\n=== Test 1: Basic add and query ===")
    cms = PatternFrequencySketch(width=1000, depth=5)

    # Real patterns from our audit
    cms.add("FUTURE:", 10)  # 10 misleading FUTURE comments
    cms.add("TODO:", 5)      # 5 accurate TODO comments
    cms.add("See:", 8)       # 8 reference comments
    cms.add("will be", 15)   # 15 speculation patterns

    assert cms.query("FUTURE:") >= 10, "Never underestimates"
    assert cms.query("TODO:") >= 5
    assert cms.query("See:") >= 8
    assert cms.query("will be") >= 15
    assert cms.query("missing") >= 0  # Unknown returns 0 or small overestimate

    print(f"  FUTURE: -> {cms.query('FUTURE:')} (expected >= 10)")
    print(f"  TODO: -> {cms.query('TODO:')} (expected >= 5)")
    print(f"  See: -> {cms.query('See:')} (expected >= 8)")
    print(f"  will be -> {cms.query('will be')} (expected >= 15)")
    print(f"  missing -> {cms.query('missing')} (expected >= 0)")
    print("  ✓ PASS")
    # Test passed


def test_2_multiple_adds_accumulate():
    """Test 2: Multiple adds accumulate"""
    print("\n=== Test 2: Multiple adds accumulate ===")
    cms = PatternFrequencySketch(width=1000, depth=5)
    cms.add("will be", 5)
    cms.add("will be", 3)
    cms.add("will be", 2)
    result = cms.query("will be")
    assert result >= 10, f"Expected >= 10, got {result}"
    print(f"  will be -> {result} (expected >= 10)")
    print("  ✓ PASS")
    # Test passed


def test_3_estimates_accurate_with_large_width():
    """Test 3: Estimates are reasonably accurate with large width"""
    print("\n=== Test 3: Estimates accurate with large width ===")
    cms = PatternFrequencySketch(width=10000, depth=7)
    patterns = {
        "FUTURE:": 100,
        "TODO:": 50,
        "FIXME:": 25,
        "NOTE:": 10,
    }
    for pattern, count in patterns.items():
        cms.add(pattern, count)

    # With large width, estimates should be close to actual
    for pattern, actual in patterns.items():
        estimate = cms.query(pattern)
        assert estimate >= actual, f"Underestimate for {pattern}: {estimate} < {actual}"
        assert estimate <= actual * 1.5, f"Overestimate for {pattern}: {estimate} > {actual * 1.5}"
        print(f"  {pattern} -> {estimate} (actual: {actual}, within 1.5x: ✓)")

    print("  ✓ PASS")
    # Test passed


def test_4_total_count_tracking():
    """Test 4: Total count tracking"""
    print("\n=== Test 4: Total count tracking ===")
    cms = PatternFrequencySketch(width=100, depth=3)
    cms.add("a", 10)
    cms.add("b", 20)
    cms.add("c", 30)
    assert cms.total_count == 60, f"Expected 60, got {cms.total_count}"
    print(f"  total_count -> {cms.total_count} (expected 60)")
    print("  ✓ PASS")
    # Test passed


def test_5_merge_sketches():
    """Test 5: Merge sketches from different modules"""
    print("\n=== Test 5: Merge sketches ===")
    cms1 = PatternFrequencySketch(width=100, depth=3)
    cms2 = PatternFrequencySketch(width=100, depth=3)

    # Module 1 has these patterns
    cms1.add("FUTURE:", 5)
    cms1.add("TODO:", 3)

    # Module 2 has these patterns
    cms2.add("FUTURE:", 3)
    cms2.add("See:", 4)

    merged = cms1.merge(cms2)
    future_count = merged.query("FUTURE:")
    todo_count = merged.query("TODO:")
    see_count = merged.query("See:")

    assert future_count >= 8, f"Expected >= 8, got {future_count}"  # 5 + 3
    assert todo_count >= 3, f"Expected >= 3, got {todo_count}"
    assert see_count >= 4, f"Expected >= 4, got {see_count}"
    assert merged.total_count == cms1.total_count + cms2.total_count

    print(f"  merged.query('FUTURE:') -> {future_count} (expected >= 8)")
    print(f"  merged.query('TODO:') -> {todo_count} (expected >= 3)")
    print(f"  merged.query('See:') -> {see_count} (expected >= 4)")
    print(f"  merged.total_count -> {merged.total_count} (cms1: {cms1.total_count} + cms2: {cms2.total_count})")
    print("  ✓ PASS")
    # Test passed


def test_6_merge_dimension_mismatch():
    """Test 6: Merge dimension mismatch"""
    print("\n=== Test 6: Merge dimension mismatch ===")
    cms1 = PatternFrequencySketch(width=100, depth=3)
    cms2 = PatternFrequencySketch(width=200, depth=3)  # Different width
    try:
        cms1.merge(cms2)
        print("  ✗ FAIL - Should raise ValueError for dimension mismatch")
        return False
    except ValueError as e:
        print(f"  ValueError raised as expected: {e}")
        print("  ✓ PASS")
        # Test passed


def test_7_high_collision_scenario():
    """Test 7: High collision scenario (small width)"""
    print("\n=== Test 7: High collision scenario ===")
    cms = PatternFrequencySketch(width=10, depth=3)
    # Add many different patterns - will have collisions
    for i in range(100):
        cms.add(f"pattern_{i}", 1)

    # Estimates will be inflated due to collisions
    # But minimum over depths helps reduce this
    estimate = cms.query("pattern_0")
    assert estimate >= 1, f"Never underestimates: {estimate} < 1"
    # With small width (10) and 100 items, expect overestimate
    # But depth=3 should help keep it reasonable

    print(f"  pattern_0 -> {estimate} (actual: 1, expected overestimate due to collisions)")
    print(f"  With width=10 and 100 patterns, avg collision per bucket: ~10")
    print(f"  Depth=3 helps reduce collision impact via minimum")
    print("  ✓ PASS")
    # Test passed


def test_8_real_audit_scenario():
    """Test 8: Real audit scenario - streaming comment analysis"""
    print("\n=== Test 8: Real audit scenario ===")
    cms = PatternFrequencySketch(width=1000, depth=5)

    # Simulate streaming through all comments in cortical/
    comment_patterns = [
        ("FUTURE:", 1),
        ("will be", 1),
        ("FUTURE:", 1),
        ("TODO:", 1),
        ("will be", 1),
        ("See:", 1),
        ("FUTURE:", 1),
        ("will be", 1),
        ("FIXME:", 1),
        ("will be", 1),
        ("FUTURE:", 1),
        ("See:", 1),
    ]

    for pattern, count in comment_patterns:
        cms.add(pattern, count)

    # Query frequencies for audit report
    future_count = cms.query("FUTURE:")
    will_be_count = cms.query("will be")
    todo_count = cms.query("TODO:")
    fixme_count = cms.query("FIXME:")

    assert future_count >= 4, f"Expected >= 4, got {future_count}"  # Should be at least 4
    assert will_be_count >= 4, f"Expected >= 4, got {will_be_count}"  # Should be at least 4
    assert todo_count >= 1, f"Expected >= 1, got {todo_count}"
    assert fixme_count >= 1, f"Expected >= 1, got {fixme_count}"

    # "will be" is the heavy hitter (speculation pattern)
    speculation_count = cms.query("will be")
    todo_count = cms.query("TODO:")
    assert speculation_count > todo_count, f"Speculation pattern should be more frequent: {speculation_count} <= {todo_count}"

    print(f"  FUTURE: -> {future_count} (actual: 4)")
    print(f"  will be -> {will_be_count} (actual: 4)")
    print(f"  TODO: -> {todo_count} (actual: 1)")
    print(f"  FIXME: -> {fixme_count} (actual: 1)")
    print(f"  Heavy hitter: 'will be' ({speculation_count}) > 'TODO:' ({todo_count})")
    print("  ✓ PASS")
    # Test passed


def test_9_deterministic_behavior():
    """Test 9: Deterministic behavior"""
    print("\n=== Test 9: Deterministic behavior ===")
    cms = PatternFrequencySketch(width=100, depth=3)
    cms.add("test", 5)
    result1 = cms.query("test")
    result2 = cms.query("test")
    assert result1 == result2, f"Query should be deterministic: {result1} != {result2}"
    print(f"  First query: {result1}")
    print(f"  Second query: {result2}")
    print(f"  Deterministic: {result1 == result2}")
    print("  ✓ PASS")
    # Test passed


def run_all_tests():
    """Run all 9 test cases"""
    print("=" * 70)
    print("Count-Min Sketch Test Suite")
    print("=" * 70)

    tests = [
        test_1_basic_add_and_query,
        test_2_multiple_adds_accumulate,
        test_3_estimates_accurate_with_large_width,
        test_4_total_count_tracking,
        test_5_merge_sketches,
        test_6_merge_dimension_mismatch,
        test_7_high_collision_scenario,
        test_8_real_audit_scenario,
        test_9_deterministic_behavior,
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
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test_name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        # Test passed
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
