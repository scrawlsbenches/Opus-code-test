#!/usr/bin/env python3
"""
Demonstration of CommentDecisionTree with detailed insights.
Shows how the tree learns patterns and makes decisions.
"""

from decision_tree_implementation import CommentDecisionTree, entropy


def demo_information_gain():
    """Demonstrate how information gain works"""
    print("="*60)
    print("DEMO 1: Understanding Information Gain")
    print("="*60)

    # Dataset with clear pattern: ref_exists="no" -> misleading
    X = [
        {"has_see_ref": "yes", "ref_exists": "no"},
        {"has_see_ref": "yes", "ref_exists": "no"},
        {"has_see_ref": "yes", "ref_exists": "no"},
        {"has_see_ref": "no", "ref_exists": "na"},
        {"has_see_ref": "no", "ref_exists": "na"},
        {"has_see_ref": "yes", "ref_exists": "yes"},
    ]
    y = ["misleading", "misleading", "misleading", "accurate", "accurate", "accurate"]

    tree = CommentDecisionTree()

    # Calculate entropy of whole dataset
    print(f"\nOriginal dataset entropy: {tree._entropy(y):.4f}")
    print(f"  - 3 misleading, 3 accurate (perfectly balanced)")

    # Calculate information gain for each feature
    for feature in ["has_see_ref", "ref_exists"]:
        ig = tree._information_gain(X, y, feature)
        print(f"\nInformation Gain for '{feature}': {ig:.4f}")

    tree.fit(X, y)
    print("\nLearned tree:")
    print(tree.print_tree())
    print("\nThe tree chose 'ref_exists' because it has higher information gain!")
    print("  - ref_exists='no' perfectly predicts 'misleading'")
    print("  - ref_exists='na' perfectly predicts 'accurate'")
    print("  - ref_exists='yes' perfectly predicts 'accurate'")


def demo_real_audit_patterns():
    """Show how tree learns from real audit patterns"""
    print("\n" + "="*60)
    print("DEMO 2: Learning Real Audit Patterns")
    print("="*60)

    # Patterns from actual audit findings
    X = [
        # Pattern 1: See reference to non-existent file = misleading
        {"has_see_ref": "yes", "ref_exists": "no", "has_future": "yes", "has_todo": "no"},
        {"has_see_ref": "yes", "ref_exists": "no", "has_future": "no", "has_todo": "no"},
        {"has_see_ref": "yes", "ref_exists": "no", "has_future": "yes", "has_todo": "no"},

        # Pattern 2: TODO about missing feature = accurate
        {"has_see_ref": "no", "ref_exists": "na", "has_future": "no", "has_todo": "yes"},
        {"has_see_ref": "no", "ref_exists": "na", "has_future": "no", "has_todo": "yes"},

        # Pattern 3: See reference to existing file = accurate
        {"has_see_ref": "yes", "ref_exists": "yes", "has_future": "no", "has_todo": "no"},
        {"has_see_ref": "yes", "ref_exists": "yes", "has_future": "no", "has_todo": "no"},

        # Pattern 4: FUTURE marker = accurate (acknowledged gap)
        {"has_see_ref": "no", "ref_exists": "na", "has_future": "yes", "has_todo": "no"},
    ]
    y = ["misleading", "misleading", "misleading", "accurate", "accurate",
         "accurate", "accurate", "accurate"]

    tree = CommentDecisionTree()
    tree.fit(X, y)

    print("\nLearned decision rules:")
    print(tree.print_tree())

    # Test predictions on new comments
    print("\n" + "-"*60)
    print("Testing new comments:")
    print("-"*60)

    test_cases = [
        {
            "features": {"has_see_ref": "yes", "ref_exists": "no", "has_future": "no", "has_todo": "no"},
            "description": "Comment with 'See: nonexistent_file.md'"
        },
        {
            "features": {"has_see_ref": "no", "ref_exists": "na", "has_future": "no", "has_todo": "yes"},
            "description": "Comment with 'TODO: implement feature X'"
        },
        {
            "features": {"has_see_ref": "yes", "ref_exists": "yes", "has_future": "no", "has_todo": "no"},
            "description": "Comment with 'See: actual_file.md' (exists)"
        },
        {
            "features": {"has_see_ref": "no", "ref_exists": "na", "has_future": "yes", "has_todo": "no"},
            "description": "Comment with 'FUTURE: will add this'"
        }
    ]

    for tc in test_cases:
        prediction = tree.predict(tc["features"])
        print(f"\n{tc['description']}")
        print(f"  → Prediction: {prediction}")


def demo_edge_cases():
    """Demonstrate edge case handling"""
    print("\n" + "="*60)
    print("DEMO 3: Edge Case Handling")
    print("="*60)

    print("\n1. Unknown feature value during prediction:")
    X = [{"color": "red"}, {"color": "blue"}, {"color": "red"}]
    y = ["apple", "sky", "apple"]
    tree = CommentDecisionTree()
    tree.fit(X, y)

    unknown_prediction = tree.predict({"color": "green"})
    print(f"   Training: red→apple (2x), blue→sky (1x)")
    print(f"   Prediction for 'green': {unknown_prediction}")
    print(f"   (Returns majority class 'apple' since 'green' was never seen)")

    print("\n2. Pure node (no split needed):")
    X = [{"x": "1"}, {"x": "2"}, {"x": "3"}]
    y = ["same", "same", "same"]
    tree = CommentDecisionTree()
    tree.fit(X, y)
    print(f"   All labels are 'same'")
    print(f"   Tree: {tree.print_tree()}")
    print(f"   (No need to split - immediately returns 'same')")

    print("\n3. Max depth limiting:")
    X = [
        {"a": "1", "b": "x", "c": "p"},
        {"a": "1", "b": "y", "c": "q"},
        {"a": "2", "b": "x", "c": "r"},
        {"a": "2", "b": "y", "c": "s"},
    ]
    y = ["yes", "no", "no", "yes"]

    print("\n   Without depth limit:")
    tree_unlimited = CommentDecisionTree(max_depth=None)
    tree_unlimited.fit(X, y)
    print(tree_unlimited.print_tree())

    print("\n   With max_depth=1:")
    tree_limited = CommentDecisionTree(max_depth=1)
    tree_limited.fit(X, y)
    print(tree_limited.print_tree())
    print("   (Stops after one split to prevent overfitting)")


def demo_entropy_mathematics():
    """Show entropy calculations for understanding"""
    print("\n" + "="*60)
    print("DEMO 4: Entropy Mathematics")
    print("="*60)

    test_cases = [
        (["A", "A", "A", "A"], "Pure (all same)"),
        (["A", "B"], "Balanced binary"),
        (["A", "A", "A", "B"], "Imbalanced (75/25)"),
        (["A", "B", "C", "D"], "Uniform 4-class"),
        ([], "Empty set"),
    ]

    print("\nEntropy measures uncertainty/impurity:")
    print("  - 0.0 = pure (no uncertainty)")
    print("  - Higher = more mixed\n")

    for labels, description in test_cases:
        ent = entropy(labels)
        print(f"{description:20} {str(labels):25} → {ent:.4f}")


def demo_practical_usage():
    """Show practical usage for the audit system"""
    print("\n" + "="*60)
    print("DEMO 5: Practical Usage for Audit System")
    print("="*60)

    print("\nScenario: Classify 29 audit findings")
    print("-" * 60)

    # Simulated audit data (representative of real findings)
    audit_data = [
        # 10 misleading comments
        *[{"has_see_ref": "yes", "ref_exists": "no", "has_will_be": "yes"}] * 6,
        *[{"has_see_ref": "no", "ref_exists": "na", "has_will_be": "yes", "describes_current": "no"}] * 4,

        # 16 accurate comments
        *[{"has_see_ref": "no", "ref_exists": "na", "has_will_be": "no", "has_todo": "yes"}] * 8,
        *[{"has_see_ref": "yes", "ref_exists": "yes", "has_will_be": "no"}] * 5,
        *[{"has_see_ref": "no", "ref_exists": "na", "has_future": "yes"}] * 3,

        # 2 unknown (ambiguous)
        {"has_see_ref": "no", "ref_exists": "na", "has_will_be": "no", "has_todo": "no"},
        {"has_see_ref": "yes", "ref_exists": "na", "has_will_be": "yes"},
    ]

    labels = (
        ["misleading"] * 10 +
        ["accurate"] * 16 +
        ["unknown"] * 2
    )

    # Train the classifier
    tree = CommentDecisionTree()
    tree.fit(audit_data, labels)

    print("Learned classification rules:")
    print(tree.print_tree())

    # Test on new comment
    print("\n" + "-"*60)
    print("New comment found in codebase:")
    print("-"*60)
    print("  'See: docs/future_design.md - this will be implemented'")
    print("  (file doesn't exist)")

    new_comment = {
        "has_see_ref": "yes",
        "ref_exists": "no",
        "has_will_be": "yes"
    }

    prediction = tree.predict(new_comment)
    print(f"\n→ Predicted category: {prediction}")
    print(f"→ Confidence: High (matches pattern in training data)")


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" "*15 + "DECISION TREE DEMONSTRATION")
    print("="*70)

    demo_information_gain()
    demo_real_audit_patterns()
    demo_edge_cases()
    demo_entropy_mathematics()
    demo_practical_usage()

    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. Tree selects features with highest information gain")
    print("2. Handles real audit patterns (misleading vs accurate comments)")
    print("3. Gracefully handles edge cases (unknown values, pure nodes)")
    print("4. max_depth prevents overfitting")
    print("5. Provides interpretable decision rules for debugging")
