#!/usr/bin/env python3
"""Additional edge case tests for CommentMarkovChain."""

from cortical.audits.algorithms.markov_chain import CommentMarkovChain

def test_edge_empty_sequence():
    """Test pattern_score with empty sequence"""
    print("Edge Case: Empty sequence...")
    mc = CommentMarkovChain()
    mc.train([["a", "b", "c"]])
    score = mc.pattern_score([])
    assert score == 0.0, f"Empty sequence should score 0.0, got {score}"
    print("✓ Empty sequence handled correctly")

def test_edge_single_word():
    """Test pattern_score with single word"""
    print("Edge Case: Single word sequence...")
    mc = CommentMarkovChain()
    mc.train([["a", "b", "c"]])
    score = mc.pattern_score(["a"])
    assert score == 0.0, f"Single word should score 0.0, got {score}"
    print("✓ Single word handled correctly")

def test_edge_zero_length_generate():
    """Test generate with length 0"""
    print("Edge Case: Generate with length 0...")
    mc = CommentMarkovChain()
    mc.train([["a", "b", "c"]])
    seq = mc.generate("a", 0)
    assert len(seq) == 0, f"Expected empty sequence for length 0, got {seq}"
    print("✓ Zero-length generation handled correctly")

def test_edge_unknown_word_probability():
    """Test probability with unknown words"""
    print("Edge Case: Unknown word probability...")
    mc = CommentMarkovChain()
    mc.train([["a", "b"]])

    # Unknown from_word
    prob = mc.probability("unknown", "b")
    assert prob == 0.0, f"Unknown from_word should give 0.0, got {prob}"

    # Unknown to_word
    prob = mc.probability("a", "unknown")
    assert prob == 0.0, f"Unknown to_word should give 0.0, got {prob}"

    print("✓ Unknown word probabilities handled correctly")

def test_edge_likely_patterns_unknown_word():
    """Test likely_patterns with unknown word"""
    print("Edge Case: likely_patterns with unknown word...")
    mc = CommentMarkovChain()
    mc.train([["a", "b"]])

    patterns = mc.likely_patterns("unknown")
    assert patterns == [], f"Unknown word should return empty list, got {patterns}"
    print("✓ Unknown word in likely_patterns handled correctly")

def test_edge_transitions_from_unknown():
    """Test transitions_from with unknown word"""
    print("Edge Case: transitions_from with unknown word...")
    mc = CommentMarkovChain()
    mc.train([["a", "b"]])

    transitions = mc.transitions_from("unknown")
    assert transitions == {}, f"Unknown word should return empty dict, got {transitions}"
    print("✓ Unknown word in transitions_from handled correctly")

def test_edge_most_likely_unknown():
    """Test most_likely_next with unknown word"""
    print("Edge Case: most_likely_next with unknown word...")
    mc = CommentMarkovChain()
    mc.train([["a", "b"]])

    result = mc.most_likely_next("unknown")
    assert result is None, f"Unknown word should return None, got {result}"
    print("✓ Unknown word in most_likely_next handled correctly")

def test_edge_multiple_tie_breaking():
    """Test tie-breaking with 3+ equal probabilities"""
    print("Edge Case: Multiple way tie-breaking...")
    mc = CommentMarkovChain()
    mc.train([
        ["start", "zebra"],
        ["start", "alpha"],
        ["start", "beta"],
    ])

    result = mc.most_likely_next("start")
    assert result == "alpha", f"Expected 'alpha' (lexicographically first), got '{result}'"
    print("✓ 3-way tie resolved correctly (lexicographic)")

def test_edge_long_sequence_scoring():
    """Test pattern_score with long sequence"""
    print("Edge Case: Long sequence scoring...")
    mc = CommentMarkovChain()
    mc.train([["a", "b", "c", "d", "e", "f"]])

    # All transitions seen
    full_score = mc.pattern_score(["a", "b", "c", "d", "e", "f"])
    assert full_score == 1.0, f"Fully trained sequence should score 1.0, got {full_score}"

    # Partial match
    partial_score = mc.pattern_score(["a", "b", "x", "y", "z"])
    assert 0 < partial_score < 1.0, f"Partially matched sequence should score between 0 and 1, got {partial_score}"
    print(f"  Full match score: {full_score}")
    print(f"  Partial match score: {partial_score}")
    print("✓ Long sequence scoring works correctly")

def test_edge_self_loop():
    """Test self-loops in Markov chain"""
    print("Edge Case: Self-loop transitions...")
    mc = CommentMarkovChain()
    mc.train([["a", "a", "a", "b"]])

    prob = mc.probability("a", "a")
    assert prob > 0, f"Self-loop should have non-zero probability, got {prob}"

    # Should be 2/3 (two a->a transitions, one a->b)
    expected = 2.0 / 3.0
    assert abs(prob - expected) < 0.0001, f"Expected {expected}, got {prob}"
    print(f"  Self-loop probability: {prob:.3f}")
    print("✓ Self-loops handled correctly")

def test_edge_empty_training_sequences():
    """Test training with empty sequences"""
    print("Edge Case: Empty training sequences...")
    mc = CommentMarkovChain()
    mc.train([[], ["a", "b"], []])

    # Should only learn from non-empty sequence
    prob = mc.probability("a", "b")
    assert prob == 1.0, f"Should learn from non-empty sequences, got {prob}"
    print("✓ Empty training sequences handled correctly")

def test_edge_single_word_sequences():
    """Test training with single-word sequences"""
    print("Edge Case: Single-word sequences...")
    mc = CommentMarkovChain()
    mc.train([["a"], ["b"], ["c"]])

    # Should have no transitions
    assert mc.most_likely_next("a") is None
    assert mc.most_likely_next("b") is None
    assert mc.most_likely_next("c") is None
    print("✓ Single-word sequences handled correctly (no transitions learned)")

def run_edge_case_tests():
    """Run all edge case tests."""
    tests = [
        test_edge_empty_sequence,
        test_edge_single_word,
        test_edge_zero_length_generate,
        test_edge_unknown_word_probability,
        test_edge_likely_patterns_unknown_word,
        test_edge_transitions_from_unknown,
        test_edge_most_likely_unknown,
        test_edge_multiple_tie_breaking,
        test_edge_long_sequence_scoring,
        test_edge_self_loop,
        test_edge_empty_training_sequences,
        test_edge_single_word_sequences,
    ]

    passed = 0
    failed = 0

    print("=" * 60)
    print("Running Edge Case Tests")
    print("=" * 60)
    print()

    for test in tests:
        try:
            test()
            passed += 1
            print()
        except AssertionError as e:
            failed += 1
            print(f"✗ FAILED: {e}")
            print()
        except Exception as e:
            failed += 1
            print(f"✗ ERROR: {e}")
            print()

    print("=" * 60)
    print(f"Edge Case Results: {passed}/{len(tests)} tests passed")
    if failed > 0:
        print(f"                   {failed}/{len(tests)} tests failed")
    print("=" * 60)

    return passed, failed

if __name__ == "__main__":
    passed, failed = run_edge_case_tests()
    exit(0 if failed == 0 else 1)
