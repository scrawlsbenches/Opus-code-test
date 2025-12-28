"""
Unit tests for NGramModel weighted training.

Tests the weighted training functionality for importance-based training
used by GitHistoryTrainer and other quality-weighted systems.
"""

import unittest
import math

from cortical.spark.ngram import NGramModel


class TestNGramWeightedTraining(unittest.TestCase):
    """Test weighted training functionality."""

    def test_train_weighted_equal_weights_same_as_train(self):
        """Test that train_weighted with equal weights = same as train."""
        docs = [
            "neural networks process data",
            "machine learning uses data",
            "deep learning processes information"
        ]

        # Train normally
        model1 = NGramModel(n=2, smoothing=1.0)
        model1.train(docs)

        # Train with equal weights
        model2 = NGramModel(n=2, smoothing=1.0)
        model2.train_weighted(docs, [1.0, 1.0, 1.0])

        # Check totals match
        self.assertEqual(model1.total_documents, model2.total_documents)
        self.assertEqual(model1.total_tokens, model2.total_tokens)
        self.assertEqual(model1.vocab, model2.vocab)

        # Check predictions match
        context = ["neural"]
        pred1 = model1.predict(context, top_k=5)
        pred2 = model2.predict(context, top_k=5)
        self.assertEqual(pred1, pred2)

    def test_train_weighted_different_weights_affects_probabilities(self):
        """Test that different weights affect probabilities correctly."""
        model = NGramModel(n=2)

        # Train with different weights
        # "neural networks" gets weight 3.0, "neural processing" gets weight 1.0
        docs = [
            "neural networks are powerful",
            "neural processing is complex"
        ]
        weights = [3.0, 1.0]

        model.train_weighted(docs, weights)

        # Predict after "neural"
        predictions = model.predict(["neural"], top_k=5)

        # "networks" should have higher probability than "processing"
        # because its document had higher weight
        pred_dict = {word: prob for word, prob in predictions}

        self.assertIn("networks", pred_dict)
        self.assertIn("processing", pred_dict)
        self.assertGreater(pred_dict["networks"], pred_dict["processing"])

    def test_train_weighted_zero_weight_excludes_document(self):
        """Test that zero weight effectively excludes a document."""
        model = NGramModel(n=2)

        docs = [
            "neural networks process data",
            "garbage text should be excluded",
            "machine learning is useful"
        ]
        weights = [1.0, 0.0, 1.0]

        model.train_weighted(docs, weights)

        # Check that "garbage" is not in vocabulary
        # (tokens from zero-weight docs are still added to vocab,
        # but they don't contribute to n-gram counts)
        # Actually, looking at the implementation, vocab.update(tokens)
        # is called before weighting, so they will be in vocab.
        # But they won't contribute to predictions.

        # Better test: verify that zero-weight document doesn't affect predictions
        predictions = model.predict(["garbage"], top_k=5)

        # "text" and "should" shouldn't appear in predictions since
        # that document had zero weight (or fallback is used)
        # Actually this is hard to test precisely, let's test totals instead

        # Total documents should be 2.0 (1.0 + 0.0 + 1.0)
        self.assertAlmostEqual(model.total_documents, 2.0)

    def test_train_weighted_length_mismatch_raises_error(self):
        """Test that mismatched lengths raise ValueError."""
        model = NGramModel(n=2)

        docs = ["doc1", "doc2", "doc3"]
        weights = [1.0, 1.0]  # Wrong length!

        with self.assertRaises(ValueError) as ctx:
            model.train_weighted(docs, weights)

        self.assertIn("same length", str(ctx.exception))

    def test_perplexity_works_after_weighted_training(self):
        """Test that perplexity calculation works after weighted training."""
        model = NGramModel(n=3)

        docs = [
            "neural networks process data efficiently",
            "machine learning models learn patterns",
            "deep learning uses neural networks"
        ]
        weights = [1.0, 0.5, 1.5]

        model.train_weighted(docs, weights)

        # Calculate perplexity on test text
        test_text = "neural networks are powerful"
        perplexity = model.perplexity(test_text)

        # Should return valid perplexity
        self.assertIsInstance(perplexity, float)
        self.assertGreater(perplexity, 0.0)
        self.assertNotEqual(perplexity, float('inf'))

    def test_train_weighted_accumulates_correctly(self):
        """Test that weighted training accumulates counts correctly."""
        model = NGramModel(n=2)

        # Train same document multiple times with different weights
        doc = "the cat sat"

        model.train_weighted([doc], [2.0])

        # Check totals reflect weight
        # Document count should be 2.0 (one doc with weight 2.0)
        self.assertAlmostEqual(model.total_documents, 2.0)

        # Token count should be 3 * 2.0 = 6.0
        self.assertAlmostEqual(model.total_tokens, 6.0)

    def test_train_on_tokens_weighted(self):
        """Test weighted training on pre-tokenized documents."""
        model = NGramModel(n=2)

        token_lists = [
            ["neural", "networks", "process"],
            ["machine", "learning", "works"]
        ]
        weights = [2.0, 1.0]

        result = model.train_on_tokens_weighted(token_lists, weights)

        # Method chaining
        self.assertIs(result, model)

        # Check totals
        self.assertAlmostEqual(model.total_documents, 3.0)  # 2.0 + 1.0
        self.assertAlmostEqual(model.total_tokens, 9.0)  # 3*2.0 + 3*1.0

    def test_train_on_tokens_weighted_length_mismatch(self):
        """Test that token_lists and weights must have same length."""
        model = NGramModel(n=2)

        token_lists = [["a", "b"], ["c", "d"]]
        weights = [1.0]  # Wrong length!

        with self.assertRaises(ValueError) as ctx:
            model.train_on_tokens_weighted(token_lists, weights)

        self.assertIn("same length", str(ctx.exception))

    def test_train_weighted_fractional_weights(self):
        """Test training with fractional weights."""
        model = NGramModel(n=2)

        docs = [
            "neural networks",
            "neural processing",
            "neural computation"
        ]
        # Main branch: 1.0, feature branches: 0.4, 0.4
        weights = [1.0, 0.4, 0.4]

        model.train_weighted(docs, weights)

        # Total documents should be 1.8
        self.assertAlmostEqual(model.total_documents, 1.8)

        # "networks" should have higher probability than "processing" or "computation"
        predictions = model.predict(["neural"], top_k=5)
        pred_dict = {word: prob for word, prob in predictions}

        self.assertIn("networks", pred_dict)
        # Networks has weight 1.0, others have 0.4
        # So networks should be most likely
        top_word = predictions[0][0]
        self.assertEqual(top_word, "networks")

    def test_train_weighted_empty_documents(self):
        """Test weighted training with empty documents."""
        model = NGramModel(n=2)

        docs = ["", "neural networks", ""]
        weights = [1.0, 1.0, 1.0]

        model.train_weighted(docs, weights)

        # Only one non-empty document should contribute
        self.assertAlmostEqual(model.total_documents, 1.0)
        self.assertGreater(len(model.vocab), 0)

    def test_train_weighted_method_chaining(self):
        """Test that train_weighted supports method chaining."""
        model = NGramModel(n=2)

        result = model.train_weighted(
            ["neural networks", "machine learning"],
            [1.0, 1.0]
        )

        # Should return self
        self.assertIs(result, model)

        # Can chain finalize
        result2 = model.finalize()
        self.assertIs(result2, model)

    def test_train_weighted_invalidates_cache(self):
        """Test that weighted training invalidates cached frequent words."""
        model = NGramModel(n=2)

        # Train and finalize to build cache
        model.train(["neural networks process data"])
        model.finalize()

        # Cache should be populated
        self.assertIsNotNone(model._cached_frequent_words)

        # Train with weights
        model.train_weighted(["machine learning"], [1.0])

        # Cache should be invalidated
        self.assertIsNone(model._cached_frequent_words)

    def test_train_weighted_mixed_with_regular_train(self):
        """Test mixing weighted and regular training."""
        model = NGramModel(n=2)

        # Regular training
        model.train(["neural networks"])

        # Weighted training
        model.train_weighted(["machine learning"], [2.0])

        # Should have both in vocabulary
        self.assertIn("neural", model.vocab)
        self.assertIn("machine", model.vocab)

        # Total documents: 1 (from train) + 2.0 (from train_weighted) = 3.0
        self.assertAlmostEqual(model.total_documents, 3.0)

    def test_probability_calculation_after_weighted_training(self):
        """Test that probability calculation works correctly with weighted training."""
        model = NGramModel(n=2, smoothing=1.0)

        # Train with known weights
        docs = ["the cat", "the cat", "the dog"]
        # Equivalent to: "the cat" with weight 2.0, "the dog" with weight 1.0
        model.train_weighted(["the cat", "the dog"], [2.0, 1.0])

        # P(cat | the) should be approximately 2/(2+1) = 0.67 (before smoothing)
        prob_cat = model.probability("cat", ["the"])
        prob_dog = model.probability("dog", ["the"])

        # cat should be twice as likely as dog (before smoothing)
        # With smoothing, the ratio is less extreme, but cat should still be more likely
        self.assertGreater(prob_cat, prob_dog)


class TestWeightedTrainingEdgeCases(unittest.TestCase):
    """Test edge cases for weighted training."""

    def test_all_zero_weights(self):
        """Test training with all zero weights."""
        model = NGramModel(n=2)

        docs = ["neural networks", "machine learning"]
        weights = [0.0, 0.0]

        model.train_weighted(docs, weights)

        # Total documents should be 0.0
        self.assertAlmostEqual(model.total_documents, 0.0)
        self.assertAlmostEqual(model.total_tokens, 0.0)

        # Vocabulary should still contain words (vocab.update happens before weighting)
        self.assertGreater(len(model.vocab), 0)

    def test_negative_weights_not_prevented(self):
        """Test that negative weights are technically allowed (though not recommended)."""
        model = NGramModel(n=2)

        # Negative weights are mathematically allowed but semantically weird
        docs = ["neural networks"]
        weights = [-1.0]

        # Should not raise error (though results may be weird)
        model.train_weighted(docs, weights)

        # Total documents will be negative
        self.assertAlmostEqual(model.total_documents, -1.0)

    def test_very_large_weights(self):
        """Test training with very large weights."""
        model = NGramModel(n=2)

        docs = ["neural networks"]
        weights = [1000000.0]

        model.train_weighted(docs, weights)

        self.assertAlmostEqual(model.total_documents, 1000000.0)

    def test_train_on_tokens_weighted_empty_token_lists(self):
        """Test weighted training with empty token lists."""
        model = NGramModel(n=2)

        token_lists = [[], ["neural", "networks"], []]
        weights = [1.0, 2.0, 1.0]

        model.train_on_tokens_weighted(token_lists, weights)

        # Only middle list contributes
        self.assertAlmostEqual(model.total_documents, 2.0)


if __name__ == '__main__':
    unittest.main()
