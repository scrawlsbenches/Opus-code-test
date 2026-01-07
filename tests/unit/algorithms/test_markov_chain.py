#!/usr/bin/env python3
"""Test suite for CommentMarkovChain implementation."""

import random
from markov_chain_implementation import CommentMarkovChain

def test_1_train_on_misleading_patterns():
    """Test 1: Train on misleading comment patterns"""
    print("Test 1: Train on misleading comment patterns...")
    mc = CommentMarkovChain()

    # Patterns from our misleading comments
    misleading_patterns = [
        ["will", "be", "implemented"],
        ["will", "be", "handled"],
        ["will", "be", "fixed"],
        ["see:", "docs/design/"],
        ["future:", "when", "cdg", "is", "implemented"],
    ]
    mc.train(misleading_patterns)

    # "will" -> "be" should be 100%
    assert mc.probability("will", "be") == 1.0, f"Expected 1.0, got {mc.probability('will', 'be')}"
    # "be" -> can lead to multiple outcomes
    assert mc.probability("be", "implemented") > 0, "be -> implemented should have non-zero probability"
    assert mc.probability("be", "handled") > 0, "be -> handled should have non-zero probability"
    print("✓ Test 1 passed")

def test_2_probabilities_sum_to_1():
    """Test 2: Probabilities sum to 1"""
    print("Test 2: Probabilities sum to 1...")
    mc = CommentMarkovChain()
    mc.train([["a", "b"], ["a", "c"], ["a", "d"], ["a", "b"]])
    transitions = mc.transitions_from("a")
    total = sum(transitions.values())
    assert abs(total - 1.0) < 0.0001, f"Expected sum ~1.0, got {total}"
    print(f"  Sum of probabilities: {total}")
    print("✓ Test 2 passed")

def test_3_most_likely_with_tie_breaking():
    """Test 3: Most likely with tie-breaking"""
    print("Test 3: Most likely with tie-breaking...")
    mc = CommentMarkovChain()
    mc.train([["start", "alpha"], ["start", "beta"]])  # 50/50 tie
    # Should return "alpha" (lexicographically first)
    result = mc.most_likely_next("start")
    assert result == "alpha", f"Expected 'alpha', got '{result}'"
    print("✓ Test 3 passed")

def test_4_generate_sequence():
    """Test 4: Generate sequence"""
    print("Test 4: Generate sequence...")
    mc = CommentMarkovChain()
    mc.train([
        ["will", "be", "done"],
        ["will", "be", "implemented"],
    ])
    random.seed(42)
    seq = mc.generate("will", 3)
    assert seq[0] == "will", f"Expected first word 'will', got '{seq[0]}'"
    assert len(seq) <= 3, f"Expected length <= 3, got {len(seq)}"
    # Each transition should be valid
    for i in range(len(seq) - 1):
        prob = mc.probability(seq[i], seq[i+1])
        assert prob > 0, f"Transition {seq[i]} -> {seq[i+1]} should have non-zero probability"
    print(f"  Generated: {seq}")
    print("✓ Test 4 passed")

def test_5_dead_end_handling():
    """Test 5: Dead end handling"""
    print("Test 5: Dead end handling...")
    mc = CommentMarkovChain()
    mc.train([["a", "b"]])  # "b" is dead end
    seq = mc.generate("a", 10)
    assert seq == ["a", "b"], f"Expected ['a', 'b'], got {seq}"
    print("✓ Test 5 passed")

def test_6_unknown_start():
    """Test 6: Unknown start"""
    print("Test 6: Unknown start...")
    mc = CommentMarkovChain()
    mc.train([["a", "b"]])
    seq = mc.generate("unknown", 5)
    assert seq == ["unknown"], f"Expected ['unknown'], got {seq}"
    print("✓ Test 6 passed")

def test_7_additive_training():
    """Test 7: Additive training"""
    print("Test 7: Additive training...")
    mc = CommentMarkovChain()
    mc.train([["a", "b"]])
    mc.train([["a", "c"]])
    prob_b = mc.probability("a", "b")
    prob_c = mc.probability("a", "c")
    assert prob_b == 0.5, f"Expected 0.5, got {prob_b}"
    assert prob_c == 0.5, f"Expected 0.5, got {prob_c}"
    print("✓ Test 7 passed")

def test_8_pattern_scoring():
    """Test 8: Pattern scoring"""
    print("Test 8: Pattern scoring...")
    mc = CommentMarkovChain()
    mc.train([
        ["will", "be", "implemented"],
        ["will", "be", "implemented"],
        ["will", "be", "implemented"],
    ])
    # Common pattern should have high score
    high_score = mc.pattern_score(["will", "be", "implemented"])
    # Unseen pattern should have low score
    low_score = mc.pattern_score(["will", "not", "work"])
    assert high_score > low_score, f"Expected high_score ({high_score}) > low_score ({low_score})"
    print(f"  High score: {high_score}, Low score: {low_score}")
    print("✓ Test 8 passed")

def test_9_likely_patterns_sorted():
    """Test 9: likely_patterns returns sorted results"""
    print("Test 9: likely_patterns returns sorted results...")
    mc = CommentMarkovChain()
    mc.train([
        ["start", "common"],
        ["start", "common"],
        ["start", "rare"],
    ])
    patterns = mc.likely_patterns("start", top_n=2)
    assert patterns[0][0] == "common", f"Expected 'common' first, got '{patterns[0][0]}'"
    assert patterns[0][1] > patterns[1][1], f"Expected descending probabilities: {patterns[0][1]} should be > {patterns[1][1]}"
    print(f"  Patterns: {patterns}")
    print("✓ Test 9 passed")

def test_10_real_misleading_comment_analysis():
    """Test 10: Real misleading comment analysis"""
    print("Test 10: Real misleading comment analysis...")
    mc = CommentMarkovChain()

    # Train on misleading patterns from our audit
    misleading_comments = [
        "FUTURE: When CDG index is implemented this will be handled".lower().split(),
        "See: docs/design/cdg-transactional-indexing-design.md".lower().split(),
        "FUTURE: When CDG index is implemented this will be replaced".lower().split(),
    ]
    mc.train(misleading_comments)

    # Check learned patterns
    next_word = mc.most_likely_next("future:")
    assert next_word == "when", f"Expected 'when', got '{next_word}'"
    next_word = mc.most_likely_next("will")
    assert next_word == "be", f"Expected 'be', got '{next_word}'"

    # Score a new potentially misleading comment
    new_comment = "future: when feature is done it will be implemented".lower().split()
    score = mc.pattern_score(new_comment)
    assert score > 0, f"Expected score > 0, got {score}"

    # Compare to an accurate comment pattern
    accurate_comment = "TODO: add error handling for edge case".lower().split()
    accurate_score = mc.pattern_score(accurate_comment)

    print(f"  Misleading pattern score: {score}")
    print(f"  Accurate pattern score: {accurate_score}")
    print("✓ Test 10 passed")

def run_all_tests():
    """Run all test cases."""
    tests = [
        test_1_train_on_misleading_patterns,
        test_2_probabilities_sum_to_1,
        test_3_most_likely_with_tie_breaking,
        test_4_generate_sequence,
        test_5_dead_end_handling,
        test_6_unknown_start,
        test_7_additive_training,
        test_8_pattern_scoring,
        test_9_likely_patterns_sorted,
        test_10_real_misleading_comment_analysis,
    ]

    passed = 0
    failed = 0

    print("=" * 60)
    print("Running Markov Chain Test Suite")
    print("=" * 60)
    print()

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"✗ {test.__name__} FAILED: {e}")
        except Exception as e:
            failed += 1
            print(f"✗ {test.__name__} ERROR: {e}")
        print()

    print("=" * 60)
    print(f"Results: {passed}/{len(tests)} tests passed")
    if failed > 0:
        print(f"         {failed}/{len(tests)} tests failed")
    print("=" * 60)

    return passed, failed

if __name__ == "__main__":
    passed, failed = run_all_tests()
    exit(0 if failed == 0 else 1)
