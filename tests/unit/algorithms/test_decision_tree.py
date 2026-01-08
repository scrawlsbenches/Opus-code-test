#!/usr/bin/env python3
"""
Test cases for CommentDecisionTree implementation.
All tests from exp-20260107-200200-decision-tree.md
"""

from cortical.audits.algorithms.decision_tree import CommentDecisionTree, entropy


def test_1_learn_from_audit_data():
    """Test 1: Learn from audit-like data"""
    print("\n=== Test 1: Learn from audit-like data ===")

    X = [
        # Misleading: references file that doesn't exist
        {"has_see_ref": "yes", "ref_exists": "no", "has_future": "yes"},
        {"has_see_ref": "yes", "ref_exists": "no", "has_future": "yes"},
        {"has_see_ref": "yes", "ref_exists": "no", "has_future": "no"},
        # Accurate: TODO that correctly identifies missing feature
        {"has_see_ref": "no", "ref_exists": "na", "has_future": "no"},
        {"has_see_ref": "no", "ref_exists": "na", "has_future": "no"},
        # Accurate: references file that exists
        {"has_see_ref": "yes", "ref_exists": "yes", "has_future": "no"},
    ]
    y = ["misleading", "misleading", "misleading", "accurate", "accurate", "accurate"]

    tree = CommentDecisionTree()
    tree.fit(X, y)

    # Should classify new comments correctly
    result1 = tree.predict({"has_see_ref": "yes", "ref_exists": "no", "has_future": "yes"})
    assert result1 == "misleading", f"Expected 'misleading', got '{result1}'"
    print(f"✓ Prediction 1: {result1}")

    result2 = tree.predict({"has_see_ref": "no", "ref_exists": "na", "has_future": "no"})
    assert result2 == "accurate", f"Expected 'accurate', got '{result2}'"
    print(f"✓ Prediction 2: {result2}")

    result3 = tree.predict({"has_see_ref": "yes", "ref_exists": "yes", "has_future": "no"})
    assert result3 == "accurate", f"Expected 'accurate', got '{result3}'"
    print(f"✓ Prediction 3: {result3}")

    print("✓ Test 1 PASSED")


def test_2_tree_structure_interpretable():
    """Test 2: Tree structure is interpretable"""
    print("\n=== Test 2: Tree structure is interpretable ===")

    X = [
        {"has_see_ref": "yes", "ref_exists": "no", "has_future": "yes"},
        {"has_see_ref": "yes", "ref_exists": "no", "has_future": "yes"},
        {"has_see_ref": "yes", "ref_exists": "no", "has_future": "no"},
        {"has_see_ref": "no", "ref_exists": "na", "has_future": "no"},
        {"has_see_ref": "no", "ref_exists": "na", "has_future": "no"},
        {"has_see_ref": "yes", "ref_exists": "yes", "has_future": "no"},
    ]
    y = ["misleading", "misleading", "misleading", "accurate", "accurate", "accurate"]

    tree = CommentDecisionTree()
    tree.fit(X, y)
    tree_str = tree.print_tree()

    print("Tree structure:")
    print(tree_str)
    print()

    assert "has_see_ref" in tree_str.lower() or "ref_exists" in tree_str.lower(), \
        "Tree should contain feature names"
    assert "misleading" in tree_str.lower(), "Tree should contain 'misleading' class"
    assert "accurate" in tree_str.lower(), "Tree should contain 'accurate' class"
    assert "\n" in tree_str, "Tree should show hierarchy with newlines"

    print("✓ Test 2 PASSED")


def test_3_handles_pure_nodes():
    """Test 3: Handles pure nodes (all same class)"""
    print("\n=== Test 3: Handles pure nodes ===")

    X = [{"a": "1"}, {"a": "2"}, {"a": "3"}]
    y = ["same", "same", "same"]
    tree = CommentDecisionTree()
    tree.fit(X, y)

    result1 = tree.predict({"a": "1"})
    assert result1 == "same", f"Expected 'same', got '{result1}'"
    print(f"✓ Known value prediction: {result1}")

    result2 = tree.predict({"a": "999"})
    assert result2 == "same", f"Expected 'same', got '{result2}'"
    print(f"✓ Unknown value prediction: {result2}")

    print("✓ Test 3 PASSED")


def test_4_handles_unknown_feature_values():
    """Test 4: Handles unknown feature values at prediction"""
    print("\n=== Test 4: Handles unknown feature values ===")

    X = [{"color": "red"}, {"color": "blue"}]
    y = ["apple", "sky"]
    tree = CommentDecisionTree()
    tree.fit(X, y)

    result = tree.predict({"color": "green"})
    assert result in ["apple", "sky"], \
        f"Unknown value should return majority class, got '{result}'"
    print(f"✓ Unknown value handled: {result} (should be 'apple' or 'sky')")

    print("✓ Test 4 PASSED")


def test_5_entropy_edge_cases():
    """Test 5: Entropy calculation edge cases"""
    print("\n=== Test 5: Entropy calculation edge cases ===")

    # Pure set - entropy should be 0
    ent1 = entropy(["a", "a", "a"])
    assert ent1 == 0.0, f"Pure set should have entropy 0.0, got {ent1}"
    print(f"✓ Pure set entropy: {ent1}")

    # Balanced binary - entropy should be 1.0
    ent2 = entropy(["a", "b"])
    assert abs(ent2 - 1.0) < 0.001, f"Balanced binary should have entropy ~1.0, got {ent2}"
    print(f"✓ Balanced binary entropy: {ent2}")

    # Empty set
    ent3 = entropy([])
    assert ent3 == 0.0, f"Empty set should have entropy 0.0, got {ent3}"
    print(f"✓ Empty set entropy: {ent3}")

    print("✓ Test 5 PASSED")


def test_6_depth_limiting():
    """Test 6: Depth limiting"""
    print("\n=== Test 6: Depth limiting ===")

    tree = CommentDecisionTree(max_depth=1)
    X = [
        {"a": "1", "b": "x"},
        {"a": "1", "b": "y"},
        {"a": "2", "b": "x"},
        {"a": "2", "b": "y"},
    ]
    y = ["yes", "no", "no", "yes"]
    tree.fit(X, y)
    tree_str = tree.print_tree()

    print("Tree with max_depth=1:")
    print(tree_str)
    print()

    # With depth=1, should only split on one feature
    lines = [l for l in tree_str.split("\n") if l.strip()]

    # Count how deep the tree goes by checking indentation
    max_indent = 0
    for line in lines:
        indent = (len(line) - len(line.lstrip())) // 2
        max_indent = max(max_indent, indent)

    assert max_indent <= 2, \
        f"With max_depth=1, should have at most 2 indent levels, got {max_indent}"
    print(f"✓ Max indentation level: {max_indent} (should be ≤ 2)")

    print("✓ Test 6 PASSED")


def test_7_real_audit_scenario():
    """Test 7: Real audit scenario"""
    print("\n=== Test 7: Real audit scenario ===")

    # Simulate the actual finding that triggered our audit
    X_real = [
        # The original misleading comment (See: + file doesn't exist)
        {"has_see_ref": "yes", "ref_exists": "no", "has_will_be": "yes", "has_future": "yes"},
        # A TODO that correctly identifies unimplemented feature
        {"has_see_ref": "no", "ref_exists": "na", "has_will_be": "no", "has_future": "no"},
    ]
    y_real = ["misleading", "accurate"]

    tree = CommentDecisionTree()
    tree.fit(X_real, y_real)

    # The original comment features
    original_comment_features = {
        "has_see_ref": "yes",
        "ref_exists": "no",  # docs/design/cdg-transactional-indexing-design.md doesn't exist
        "has_will_be": "yes",  # "this will be handled"
        "has_future": "yes"   # "FUTURE:"
    }
    result = tree.predict(original_comment_features)
    assert result == "misleading", f"Expected 'misleading', got '{result}'"
    print(f"✓ Original misleading comment classified as: {result}")

    print("Tree for audit scenario:")
    print(tree.print_tree())

    print("✓ Test 7 PASSED")


def test_bonus_edge_cases():
    """Bonus: Additional edge cases"""
    print("\n=== Bonus: Additional edge cases ===")

    # Empty features dictionary
    tree = CommentDecisionTree()
    X = [{}, {}, {}]
    y = ["a", "b", "c"]
    tree.fit(X, y)
    result = tree.predict({})
    print(f"✓ Empty features prediction: {result}")

    # Single sample
    tree2 = CommentDecisionTree()
    X2 = [{"x": "1"}]
    y2 = ["lonely"]
    tree2.fit(X2, y2)
    result2 = tree2.predict({"x": "1"})
    assert result2 == "lonely", f"Single sample should predict correctly, got '{result2}'"
    print(f"✓ Single sample prediction: {result2}")

    # All features have same value
    tree3 = CommentDecisionTree()
    X3 = [{"x": "1"}, {"x": "1"}, {"x": "1"}]
    y3 = ["a", "b", "a"]
    tree3.fit(X3, y3)
    result3 = tree3.predict({"x": "1"})
    assert result3 in ["a", "b"], f"Should predict majority class, got '{result3}'"
    print(f"✓ Same feature values prediction: {result3}")

    print("✓ Bonus tests PASSED")


if __name__ == "__main__":
    print("="*60)
    print("Running Decision Tree Tests")
    print("="*60)

    try:
        test_1_learn_from_audit_data()
        test_2_tree_structure_interpretable()
        test_3_handles_pure_nodes()
        test_4_handles_unknown_feature_values()
        test_5_entropy_edge_cases()
        test_6_depth_limiting()
        test_7_real_audit_scenario()
        test_bonus_edge_cases()

        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)

        # Print summary
        print("\nImplementation Summary:")
        print("- ✓ fit(), predict(), print_tree() implemented")
        print("- ✓ Entropy handles p=0 (no crash, returns 0 contribution)")
        print("- ✓ Information gain selects best split")
        print("- ✓ All 7 test cases pass")
        print("- ✓ Tree output shows clear decision path")
        print("- ✓ Unknown feature values handled gracefully")
        print("- ✓ max_depth parameter works")
        print("- ✓ Bonus edge cases handled")

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
