"""
Test suite for FindingCluster Union-Find implementation
All test cases from exp-20260107-200500-union-find.md
"""

from cortical.audits.algorithms.union_find import FindingCluster


def test_1_basic_clustering():
    """Test 1: Basic clustering of related findings"""
    print("\n=== Test 1: Basic clustering ===")
    uf = FindingCluster()

    # Two findings reference the same missing design doc
    uf.make_set("F001")  # cortical/got/indexer.py:478 - refs cdg-transactional-indexing-design.md
    uf.make_set("F002")  # cortical/got/indexer.py:508 - refs same file

    assert uf.connected("F001", "F002") == False
    uf.union("F001", "F002")
    assert uf.connected("F001", "F002") == True
    assert uf.cluster_count() == 1
    print("✓ Test 1 passed")


def test_2_transitivity():
    """Test 2: Transitivity"""
    print("\n=== Test 2: Transitivity ===")
    uf = FindingCluster()
    for f in ["F001", "F002", "F003", "F004", "F005"]:
        uf.make_set(f)

    # F001, F002, F003 all reference missing docs
    uf.union("F001", "F002")
    uf.union("F002", "F003")
    # F004, F005 are "will be" speculation patterns
    uf.union("F004", "F005")

    assert uf.connected("F001", "F003") == True  # Transitive
    assert uf.connected("F001", "F004") == False  # Different clusters
    assert uf.cluster_count() == 2
    print("✓ Test 2 passed")


def test_3_get_cluster_members():
    """Test 3: Get cluster members"""
    print("\n=== Test 3: Get cluster members ===")
    uf = FindingCluster()
    for f in ["F001", "F002", "F003"]:
        uf.make_set(f)
    uf.union("F001", "F002")

    cluster = uf.get_cluster("F001")
    assert cluster == {"F001", "F002"}
    assert uf.get_cluster("F003") == {"F003"}
    print("✓ Test 3 passed")


def test_4_get_all_clusters():
    """Test 4: Get all clusters"""
    print("\n=== Test 4: Get all clusters ===")
    uf = FindingCluster()
    for f in ["F001", "F002", "F003"]:
        uf.make_set(f)
    uf.union("F001", "F002")

    all_clusters = uf.get_all_clusters()
    assert len(all_clusters) == 2
    cluster_sizes = sorted([len(c) for c in all_clusters])
    assert cluster_sizes == [1, 2]
    print("✓ Test 4 passed")


def test_5_path_compression():
    """Test 5: Path compression verification"""
    print("\n=== Test 5: Path compression ===")
    uf = FindingCluster()
    for f in ["F001", "F002", "F003", "F004", "F005", "F006"]:
        uf.make_set(f)

    # Create a chain: F001-F002-F003-F004-F005-F006
    uf.union("F001", "F002")
    uf.union("F002", "F003")
    uf.union("F003", "F004")
    uf.union("F004", "F005")
    uf.union("F005", "F006")

    # After find on F006, path should be compressed
    root = uf.find("F006")
    # All should now point directly to root
    assert uf._parent["F006"] == root  # Path compressed
    assert uf._parent["F005"] == root  # Path compressed
    print(f"  Root: {root}")
    print(f"  F006 parent: {uf._parent['F006']}")
    print(f"  F005 parent: {uf._parent['F005']}")
    print("✓ Test 5 passed")


def test_6_union_return_value():
    """Test 6: Union returns correct boolean"""
    print("\n=== Test 6: Union return value ===")
    uf = FindingCluster()
    uf.make_set("F001")
    uf.make_set("F002")
    assert uf.union("F001", "F002") == True   # Different clusters merged
    assert uf.union("F001", "F002") == False  # Same cluster, no merge
    print("✓ Test 6 passed")


def test_7_auto_create_on_union():
    """Test 7: Auto-create on union"""
    print("\n=== Test 7: Auto-create on union ===")
    uf = FindingCluster()
    uf.union("NEW1", "NEW2")  # Should auto-create both
    assert uf.connected("NEW1", "NEW2") == True
    assert uf.cluster_count() == 1
    print("✓ Test 7 passed")


def test_8_error_handling():
    """Test 8: Error handling"""
    print("\n=== Test 8: Error handling ===")
    uf = FindingCluster()
    try:
        uf.find("NONEXISTENT")
        assert False, "Should raise ValueError"
    except ValueError:
        pass
    print("✓ Test 8 passed")


def test_9_real_audit_scenario():
    """Test 9: Real audit scenario - cluster related findings"""
    print("\n=== Test 9: Real audit scenario ===")
    uf = FindingCluster()

    # From our actual audit:
    # - F001, F002: Both in cortical/got/indexer.py, reference same missing doc
    # - F003: cortical/got/orphan.py TODO (different issue)
    # - F004, F005: Both have "will be" pattern

    findings = {
        "F001": {"file": "indexer.py", "pattern": "See:", "ref": "cdg-design.md"},
        "F002": {"file": "indexer.py", "pattern": "See:", "ref": "cdg-design.md"},
        "F003": {"file": "orphan.py", "pattern": "TODO:", "ref": None},
        "F004": {"file": "storage.py", "pattern": "will be", "ref": None},
        "F005": {"file": "api.py", "pattern": "will be", "ref": None},
    }

    for f_id in findings:
        uf.make_set(f_id)

    # Cluster by same referenced file
    uf.union("F001", "F002")  # Same missing doc reference

    # Cluster by same pattern type
    uf.union("F004", "F005")  # Both "will be" patterns

    assert uf.cluster_count() == 3  # [F001,F002], [F003], [F004,F005]
    assert uf.get_cluster("F001") == {"F001", "F002"}
    assert uf.get_cluster("F004") == {"F004", "F005"}
    print(f"  Cluster count: {uf.cluster_count()}")
    print(f"  F001 cluster: {uf.get_cluster('F001')}")
    print(f"  F004 cluster: {uf.get_cluster('F004')}")
    print("✓ Test 9 passed")


def run_all_tests():
    """Run all test cases"""
    print("=" * 60)
    print("Running Union-Find Tests for Audit Finding Clustering")
    print("=" * 60)

    tests = [
        test_1_basic_clustering,
        test_2_transitivity,
        test_3_get_cluster_members,
        test_4_get_all_clusters,
        test_5_path_compression,
        test_6_union_return_value,
        test_7_auto_create_on_union,
        test_8_error_handling,
        test_9_real_audit_scenario,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} ERROR: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{len(tests)} tests passed")
    if failed == 0:
        print("🎉 ALL TESTS PASSED!")
    else:
        print(f"❌ {failed} test(s) failed")
    print("=" * 60)

    return passed, failed


if __name__ == "__main__":
    run_all_tests()
