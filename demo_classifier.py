"""
Interactive demonstration of Naive Bayes Comment Classifier
Shows predictions on various comment types
"""

from naive_bayes_classifier import CommentClassifier


def demo_classifier():
    """Demonstrate the classifier with various comment types"""

    print("="*70)
    print("NAIVE BAYES COMMENT CLASSIFIER - INTERACTIVE DEMO")
    print("="*70)

    # Train on audit patterns
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

    print("\n📚 Training completed on 10 audit comment patterns")
    print("   - 5 misleading (speculative, broken references)")
    print("   - 5 accurate (actionable, factual)")

    # Test various comment types
    test_cases = [
        ("FUTURE: This will be implemented when X is ready", "Speculative future promise"),
        ("TODO: Fix the memory leak in line 42", "Actionable TODO item"),
        ("See: docs/architecture/missing-file.md", "Broken reference"),
        ("Returns None if the input is empty", "Factual description"),
        ("will be refactored later", "Vague future statement"),
        ("Raises ValueError if validation fails", "Clear exception documentation"),
        ("when the new API is available this will change", "Uncertain future dependency"),
        ("FIXME: O(n²) complexity, needs optimization", "Specific technical debt"),
    ]

    print("\n" + "="*70)
    print("PREDICTIONS ON TEST COMMENTS")
    print("="*70)

    for comment, description in test_cases:
        tokens = comment.lower().replace(":", "").replace(",", "").replace(".", "").split()
        prediction = classifier.predict(tokens)
        probs = classifier.predict_proba(tokens)
        confidence = probs[prediction]

        # Color coding based on prediction
        symbol = "⚠️ " if prediction == "misleading" else "✅"

        print(f"\n{symbol} Comment: \"{comment}\"")
        print(f"   Type: {description}")
        print(f"   Prediction: {prediction.upper()} ({confidence:.1%} confidence)")
        print(f"   Probabilities: misleading={probs['misleading']:.1%}, accurate={probs['accurate']:.1%}")

    # Show most indicative words
    print("\n" + "="*70)
    print("MOST INDICATIVE WORDS FOR EACH CLASS")
    print("="*70)

    print("\n🔴 Top 10 words for 'misleading' class:")
    misleading_words = classifier.most_indicative_words("misleading", top_n=10)
    for i, (word, prob) in enumerate(misleading_words, 1):
        print(f"   {i:2d}. {word:15s} (P={prob:.4f})")

    print("\n🟢 Top 10 words for 'accurate' class:")
    accurate_words = classifier.most_indicative_words("accurate", top_n=10)
    for i, (word, prob) in enumerate(accurate_words, 1):
        print(f"   {i:2d}. {word:15s} (P={prob:.4f})")

    # Pattern analysis
    print("\n" + "="*70)
    print("PATTERN ANALYSIS")
    print("="*70)

    misleading_word_list = [w for w, _ in misleading_words[:5]]
    accurate_word_list = [w for w, _ in accurate_words[:5]]

    print("\n🔍 Key patterns detected:")
    print(f"   Misleading markers: {', '.join(misleading_word_list)}")
    print(f"   Accurate markers: {', '.join(accurate_word_list)}")

    print("\n💡 Insights:")
    print("   - Words like 'will', 'future', 'when' indicate speculation")
    print("   - Words like 'see:', 'docs/' suggest references (may be broken)")
    print("   - Words like 'todo', 'fixme', 'returns' indicate clear documentation")
    print("   - Actionable keywords correlate with accurate comments")

    print("\n" + "="*70)


if __name__ == "__main__":
    demo_classifier()
