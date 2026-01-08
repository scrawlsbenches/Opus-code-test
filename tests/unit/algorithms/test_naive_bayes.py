"""
Test suite for Naive Bayes Comment Classifier
All 8 test cases from exp-20260107-200400-naive-bayes.md
"""

from cortical.audits.algorithms.naive_bayes import CommentClassifier


def test_1_basic_classification():
    """Test 1: Train on actual misleading vs accurate patterns"""
    print("\n=== Test 1: Basic Classification ===")
    classifier = CommentClassifier()

    # Real patterns from our audit
    misleading_comments = [
        ["future", "when", "cdg", "index", "is", "implemented", "this", "will", "be", "handled"],
        ["see", "docs", "design", "cdg", "transactional", "indexing", "design", "md"],
        ["will", "be", "replaced", "when", "feature", "is", "done"],
        ["future", "this", "will", "be", "handled", "at", "storage", "layer"],
        ["see", "docs", "design", "missing", "file", "md"],
    ]

    accurate_comments = [
        ["todo", "add", "error", "handling", "for", "edge", "case"],
        ["fixme", "this", "loop", "is", "slow", "optimize", "later"],
        ["returns", "the", "count", "of", "processed", "items"],
        ["raises", "valueerror", "if", "input", "is", "invalid"],
        ["note", "this", "function", "is", "thread", "safe"],
    ]

    labels = ["misleading"] * 5 + ["accurate"] * 5
    classifier.fit(misleading_comments + accurate_comments, labels)

    # Test speculative comment
    result1 = classifier.predict(["will", "be", "implemented", "soon"])
    print(f"  Speculative comment prediction: {result1}")
    assert result1 == "misleading", f"Expected 'misleading', got '{result1}'"

    # Test actionable comment
    result2 = classifier.predict(["todo", "fix", "this", "bug"])
    print(f"  Actionable comment prediction: {result2}")
    assert result2 == "accurate", f"Expected 'accurate', got '{result2}'"

    print("  ✓ Test 1 PASSED")
    # Test passed


def test_2_unseen_words():
    """Test 2: Handles unseen words (Laplace smoothing)"""
    print("\n=== Test 2: Unseen Words (Laplace Smoothing) ===")
    classifier = CommentClassifier()

    misleading_comments = [
        ["future", "when", "cdg", "index", "is", "implemented", "this", "will", "be", "handled"],
        ["see", "docs", "design", "cdg", "transactional", "indexing", "design", "md"],
        ["will", "be", "replaced", "when", "feature", "is", "done"],
        ["future", "this", "will", "be", "handled", "at", "storage", "layer"],
        ["see", "docs", "design", "missing", "file", "md"],
    ]

    accurate_comments = [
        ["todo", "add", "error", "handling", "for", "edge", "case"],
        ["fixme", "this", "loop", "is", "slow", "optimize", "later"],
        ["returns", "the", "count", "of", "processed", "items"],
        ["raises", "valueerror", "if", "input", "is", "invalid"],
        ["note", "this", "function", "is", "thread", "safe"],
    ]

    labels = ["misleading"] * 5 + ["accurate"] * 5
    classifier.fit(misleading_comments + accurate_comments, labels)

    # "unknown" wasn't in training data, but shouldn't crash
    result = classifier.predict(["unknown", "never", "seen", "words"])
    print(f"  Unseen words prediction: {result}")
    assert result in ["misleading", "accurate"], f"Expected valid class, got '{result}'"

    print("  ✓ Test 2 PASSED")
    # Test passed


def test_3_probabilities_sum_to_one():
    """Test 3: Probabilities sum to 1"""
    print("\n=== Test 3: Probabilities Sum to 1.0 ===")
    classifier = CommentClassifier()

    misleading_comments = [
        ["future", "when", "cdg", "index", "is", "implemented", "this", "will", "be", "handled"],
        ["see", "docs", "design", "cdg", "transactional", "indexing", "design", "md"],
        ["will", "be", "replaced", "when", "feature", "is", "done"],
        ["future", "this", "will", "be", "handled", "at", "storage", "layer"],
        ["see", "docs", "design", "missing", "file", "md"],
    ]

    accurate_comments = [
        ["todo", "add", "error", "handling", "for", "edge", "case"],
        ["fixme", "this", "loop", "is", "slow", "optimize", "later"],
        ["returns", "the", "count", "of", "processed", "items"],
        ["raises", "valueerror", "if", "input", "is", "invalid"],
        ["note", "this", "function", "is", "thread", "safe"],
    ]

    labels = ["misleading"] * 5 + ["accurate"] * 5
    classifier.fit(misleading_comments + accurate_comments, labels)

    probs = classifier.predict_proba(["future", "will", "be", "done"])
    prob_sum = sum(probs.values())
    print(f"  Probabilities: {probs}")
    print(f"  Sum: {prob_sum}")

    assert abs(prob_sum - 1.0) < 0.0001, f"Expected sum=1.0, got {prob_sum}"
    assert "misleading" in probs, "Missing 'misleading' class"
    assert "accurate" in probs, "Missing 'accurate' class"

    print("  ✓ Test 3 PASSED")
    # Test passed


def test_4_prior_probability():
    """Test 4: Prior probability matters (class imbalance)"""
    print("\n=== Test 4: Prior Probability (Class Imbalance) ===")
    classifier = CommentClassifier()

    # 16 accurate vs 10 misleading (like our real audit)
    many_accurate = [["good", "comment"]] * 16
    few_misleading = [["bad", "comment"]] * 10
    classifier.fit(many_accurate + few_misleading, ["accurate"] * 16 + ["misleading"] * 10)

    # A "good comment" should lean strongly toward "accurate" due to both prior and likelihood
    probs = classifier.predict_proba(["good", "comment"])
    print(f"  Probabilities for 'good comment': {probs}")
    print(f"  accurate: {probs['accurate']:.4f}, misleading: {probs['misleading']:.4f}")

    assert probs["accurate"] > probs["misleading"], \
        f"Expected accurate > misleading for 'good comment', got accurate={probs['accurate']}, misleading={probs['misleading']}"

    print("  ✓ Test 4 PASSED")
    # Test passed


def test_5_most_indicative_words():
    """Test 5: Most indicative words"""
    print("\n=== Test 5: Most Indicative Words ===")
    classifier = CommentClassifier()

    misleading_comments = [
        ["future", "when", "cdg", "index", "is", "implemented", "this", "will", "be", "handled"],
        ["see", "docs", "design", "cdg", "transactional", "indexing", "design", "md"],
        ["will", "be", "replaced", "when", "feature", "is", "done"],
        ["future", "this", "will", "be", "handled", "at", "storage", "layer"],
        ["see", "docs", "design", "missing", "file", "md"],
    ]

    accurate_comments = [
        ["todo", "add", "error", "handling", "for", "edge", "case"],
        ["fixme", "this", "loop", "is", "slow", "optimize", "later"],
        ["returns", "the", "count", "of", "processed", "items"],
        ["raises", "valueerror", "if", "input", "is", "invalid"],
        ["note", "this", "function", "is", "thread", "safe"],
    ]

    labels = ["misleading"] * 5 + ["accurate"] * 5
    classifier.fit(misleading_comments + accurate_comments, labels)

    misleading_words = classifier.most_indicative_words("misleading", top_n=5)
    print(f"  Top 5 misleading words: {misleading_words}")

    # Words like "will", "future", "be", "see" should be high for misleading
    word_list = [w for w, p in misleading_words]
    found = any(w in word_list for w in ["will", "future", "be", "see"])
    assert found, f"Expected key misleading words in {word_list}"

    print("  ✓ Test 5 PASSED")
    # Test passed


def test_6_single_class():
    """Test 6: Single class handling"""
    print("\n=== Test 6: Single Class Handling ===")
    classifier = CommentClassifier()

    classifier.fit([["a"], ["b"], ["c"]], ["only", "only", "only"])
    result = classifier.predict(["anything"])
    print(f"  Single class prediction: {result}")

    assert result == "only", f"Expected 'only', got '{result}'"

    print("  ✓ Test 6 PASSED")
    # Test passed


def test_7_empty_input():
    """Test 7: Empty vocabulary edge case"""
    print("\n=== Test 7: Empty Input Edge Case ===")
    classifier = CommentClassifier()

    classifier.fit([["word"]], ["class"])
    # Predicting with empty comment
    result = classifier.predict([])
    print(f"  Empty comment prediction: {result}")

    assert result == "class", f"Expected 'class', got '{result}'"  # Falls back to prior

    print("  ✓ Test 7 PASSED")
    # Test passed


def test_8_real_misleading_detection():
    """Test 8: Real misleading comment detection scenario"""
    print("\n=== Test 8: Real Misleading Comment Detection ===")
    classifier = CommentClassifier()

    misleading_comments = [
        ["future", "when", "cdg", "index", "is", "implemented", "this", "will", "be", "handled"],
        ["see", "docs", "design", "cdg", "transactional", "indexing", "design", "md"],
        ["will", "be", "replaced", "when", "feature", "is", "done"],
        ["future", "this", "will", "be", "handled", "at", "storage", "layer"],
        ["see", "docs", "design", "missing", "file", "md"],
    ]

    accurate_comments = [
        ["todo", "add", "error", "handling", "for", "edge", "case"],
        ["fixme", "this", "loop", "is", "slow", "optimize", "later"],
        ["returns", "the", "count", "of", "processed", "items"],
        ["raises", "valueerror", "if", "input", "is", "invalid"],
        ["note", "this", "function", "is", "thread", "safe"],
    ]

    # Train with our actual audit data patterns
    train_data = misleading_comments + accurate_comments
    train_labels = ["misleading"] * len(misleading_comments) + ["accurate"] * len(accurate_comments)
    classifier.fit(train_data, train_labels)

    # Test with actual misleading comment from finding F-001
    real_misleading = "FUTURE: When CDG index is implemented, this will be handled at the storage layer with WAL-based recovery."
    tokens = real_misleading.lower().replace(":", "").replace(",", "").replace(".", "").split()
    print(f"  Tokenized comment: {tokens[:10]}...")

    prediction = classifier.predict(tokens)
    probs = classifier.predict_proba(tokens)

    print(f"  Prediction: {prediction}")
    print(f"  Probabilities: {probs}")
    print(f"  Confidence: {probs['misleading']:.4f}")

    # Should predict misleading with high confidence
    assert prediction == "misleading", f"Expected 'misleading', got '{prediction}'"
    assert probs["misleading"] > 0.6, f"Expected confidence > 0.6, got {probs['misleading']}"

    print("  ✓ Test 8 PASSED")
    # Test passed


def run_all_tests():
    """Run all test cases"""
    print("\n" + "="*60)
    print("NAIVE BAYES CLASSIFIER - COMPREHENSIVE TEST SUITE")
    print("="*60)

    tests = [
        ("Test 1: Basic Classification", test_1_basic_classification),
        ("Test 2: Unseen Words", test_2_unseen_words),
        ("Test 3: Probabilities Sum to 1", test_3_probabilities_sum_to_one),
        ("Test 4: Prior Probability", test_4_prior_probability),
        ("Test 5: Most Indicative Words", test_5_most_indicative_words),
        ("Test 6: Single Class", test_6_single_class),
        ("Test 7: Empty Input", test_7_empty_input),
        ("Test 8: Real Misleading Detection", test_8_real_misleading_detection),
    ]

    passed = 0
    failed = 0
    results = []

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
            results.append(f"✓ {name}")
        except AssertionError as e:
            failed += 1
            results.append(f"✗ {name}: {str(e)}")
        except Exception as e:
            failed += 1
            results.append(f"✗ {name}: EXCEPTION - {str(e)}")

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for result in results:
        print(result)

    print(f"\nTotal: {passed + failed} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("="*60)

    return passed, failed


if __name__ == "__main__":
    passed, failed = run_all_tests()
    exit(0 if failed == 0 else 1)
