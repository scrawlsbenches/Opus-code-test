"""
Unit tests for SparkSLM package.

Tests the statistical language model components:
- NGramModel: n-gram language modeling
- AlignmentIndex: user definitions/patterns/preferences
- SparkPredictor: unified facade
"""

import unittest
import tempfile
import os
from pathlib import Path

from cortical.spark.ngram import NGramModel
from cortical.spark.alignment import AlignmentIndex, AlignmentEntry
from cortical.spark.predictor import SparkPredictor


class TestNGramModel(unittest.TestCase):
    """Test NGramModel class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        model = NGramModel()
        self.assertEqual(model.n, 3)
        self.assertEqual(model.smoothing, 1.0)
        self.assertEqual(len(model.vocab), 0)
        self.assertEqual(model.total_documents, 0)
        self.assertEqual(model.total_tokens, 0)

    def test_init_custom_order(self):
        """Test initialization with custom order."""
        model = NGramModel(n=4, smoothing=0.5)
        self.assertEqual(model.n, 4)
        self.assertEqual(model.smoothing, 0.5)

    def test_init_invalid_order(self):
        """Test that n < 2 raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            NGramModel(n=1)
        self.assertIn("at least 2", str(ctx.exception))

        with self.assertRaises(ValueError):
            NGramModel(n=0)

        with self.assertRaises(ValueError):
            NGramModel(n=-1)

    def test_train_basic(self):
        """Test basic training on documents."""
        model = NGramModel(n=2)
        docs = [
            "the cat sat on the mat",
            "the dog ran fast"
        ]
        result = model.train(docs)

        # Method chaining
        self.assertIs(result, model)

        # Training stats
        self.assertEqual(model.total_documents, 2)
        self.assertGreater(model.total_tokens, 0)
        self.assertGreater(len(model.vocab), 0)

    def test_train_builds_vocabulary(self):
        """Test that training builds vocabulary correctly."""
        model = NGramModel(n=3)
        docs = ["neural networks process data efficiently"]
        model.train(docs)

        # Check vocabulary
        self.assertIn("neural", model.vocab)
        self.assertIn("networks", model.vocab)
        self.assertIn("process", model.vocab)
        self.assertIn("data", model.vocab)
        self.assertIn("efficiently", model.vocab)

        # Check token count
        self.assertEqual(len(model.vocab), 5)
        self.assertEqual(model.total_tokens, 5)

    def test_predict_basic(self):
        """Test basic prediction."""
        model = NGramModel(n=2)
        docs = [
            "the cat sat",
            "the cat ran",
            "the dog barked"
        ]
        model.train(docs)

        # Predict after "the"
        predictions = model.predict(["the"], top_k=3)

        # Should return list of (word, probability) tuples
        self.assertIsInstance(predictions, list)
        self.assertGreater(len(predictions), 0)

        # Check format
        for word, prob in predictions:
            self.assertIsInstance(word, str)
            self.assertIsInstance(prob, float)
            self.assertGreater(prob, 0.0)
            self.assertLessEqual(prob, 1.0)

        # "cat" should be most likely after "the"
        top_word = predictions[0][0]
        self.assertEqual(top_word, "cat")

    def test_predict_returns_probabilities(self):
        """Test that predictions return valid probabilities."""
        model = NGramModel(n=3)
        docs = ["neural networks process data"] * 10
        model.train(docs)

        predictions = model.predict(["neural", "networks"], top_k=5)

        for word, prob in predictions:
            self.assertGreater(prob, 0.0)
            self.assertLessEqual(prob, 1.0)

    def test_predict_unknown_context(self):
        """Test prediction with unknown context."""
        model = NGramModel(n=3)
        docs = ["the cat sat on the mat"]
        model.train(docs)

        # Predict with completely unknown context
        predictions = model.predict(["unknown", "context"], top_k=5)

        # Should return fallback predictions (most frequent words)
        self.assertIsInstance(predictions, list)
        # Might be empty or contain fallback words

    def test_predict_sequence(self):
        """Test sequence prediction (greedy decoding)."""
        model = NGramModel(n=3)
        docs = [
            "neural networks process data efficiently",
            "neural networks learn patterns automatically",
            "neural networks are powerful models"
        ]
        model.train(docs)

        # Predict sequence starting with "neural networks"
        sequence = model.predict_sequence(["neural", "networks"], length=2)

        self.assertIsInstance(sequence, list)
        self.assertGreater(len(sequence), 0)
        self.assertLessEqual(len(sequence), 2)

        # All predictions should be strings
        for word in sequence:
            self.assertIsInstance(word, str)
            self.assertNotEqual(word, model.START)
            self.assertNotEqual(word, model.END)

    def test_probability(self):
        """Test probability calculation."""
        model = NGramModel(n=2)
        docs = [
            "the cat sat",
            "the cat ran",
            "the dog barked"
        ]
        model.train(docs)

        # P(cat | the) should be higher than P(dog | the)
        prob_cat = model.probability("cat", ["the"])
        prob_dog = model.probability("dog", ["the"])

        self.assertGreater(prob_cat, 0.0)
        self.assertGreater(prob_dog, 0.0)
        self.assertGreater(prob_cat, prob_dog)

    def test_perplexity(self):
        """Test perplexity calculation."""
        model = NGramModel(n=3)
        train_docs = [
            "neural networks process data efficiently",
            "machine learning models are powerful",
            "deep learning uses neural networks"
        ]
        model.train(train_docs)

        # Perplexity on training data should be reasonable
        test_text = "neural networks are powerful"
        perplexity = model.perplexity(test_text)

        self.assertIsInstance(perplexity, float)
        self.assertGreater(perplexity, 0.0)
        self.assertNotEqual(perplexity, float('inf'))

    def test_save_and_load(self):
        """Test saving and loading model."""
        model = NGramModel(n=3, smoothing=0.5)
        docs = [
            "neural networks process data",
            "machine learning is powerful"
        ]
        model.train(docs)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.json")

            # Save
            model.save(path)
            self.assertTrue(os.path.exists(path))

            # Load
            loaded = NGramModel.load(path)

            # Check parameters preserved
            self.assertEqual(loaded.n, model.n)
            self.assertEqual(loaded.smoothing, model.smoothing)

            # Check vocabulary preserved
            self.assertEqual(loaded.vocab, model.vocab)
            self.assertEqual(loaded.total_tokens, model.total_tokens)
            self.assertEqual(loaded.total_documents, model.total_documents)

            # Check predictions match
            context = ["neural", "networks"]
            orig_pred = model.predict(context, top_k=3)
            loaded_pred = loaded.predict(context, top_k=3)
            self.assertEqual(orig_pred, loaded_pred)


class TestAlignmentIndex(unittest.TestCase):
    """Test AlignmentIndex class."""

    def test_add_definition(self):
        """Test adding definitions."""
        index = AlignmentIndex()
        index.add_definition("spark", "fast statistical predictor", source="test")

        entries = index.lookup("spark")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].key, "spark")
        self.assertEqual(entries[0].value, "fast statistical predictor")
        self.assertEqual(entries[0].entry_type, "definition")
        self.assertEqual(entries[0].source, "test")

    def test_add_pattern(self):
        """Test adding patterns."""
        index = AlignmentIndex()
        index.add_pattern("error handling", "use Result types", source="test")

        entries = index.lookup("error handling")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].entry_type, "pattern")
        self.assertEqual(entries[0].value, "use Result types")

    def test_add_preference(self):
        """Test adding preferences."""
        index = AlignmentIndex()
        index.add_preference("naming", "snake_case for functions", source="test")

        entries = index.lookup("naming")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].entry_type, "preference")

    def test_add_goal(self):
        """Test adding goals."""
        index = AlignmentIndex()
        index.add_goal("build fast search", "optimize for speed", source="test")

        goals = index.get_current_goals()
        self.assertEqual(len(goals), 1)
        self.assertEqual(goals[0].entry_type, "goal")
        self.assertIn("build fast search", goals[0].value)
        self.assertIn("optimize for speed", goals[0].value)

    def test_lookup_case_insensitive(self):
        """Test that lookup is case-insensitive."""
        index = AlignmentIndex()
        index.add_definition("SparkSLM", "statistical language model")

        # Lookup with different cases
        entries1 = index.lookup("sparkslm")
        entries2 = index.lookup("SPARKSLM")
        entries3 = index.lookup("SparkSLM")

        self.assertEqual(len(entries1), 1)
        self.assertEqual(len(entries2), 1)
        self.assertEqual(len(entries3), 1)
        self.assertEqual(entries1[0].value, entries2[0].value)
        self.assertEqual(entries2[0].value, entries3[0].value)

    def test_search(self):
        """Test search functionality."""
        index = AlignmentIndex()
        index.add_definition("neural network", "artificial neural network")
        index.add_definition("deep learning", "multi-layer neural networks")
        index.add_pattern("network architecture", "use modular design")

        # Search for "neural"
        results = index.search("neural", top_k=5)

        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

        # Check result format
        for key, entry in results:
            self.assertIsInstance(key, str)
            self.assertIsInstance(entry, AlignmentEntry)

        # Should find entries related to "neural"
        keys = [key for key, _ in results]
        self.assertIn("neural network", keys)

    def test_get_all_definitions(self):
        """Test getting all definitions."""
        index = AlignmentIndex()
        index.add_definition("term1", "meaning1")
        index.add_definition("term2", "meaning2")
        index.add_pattern("pattern1", "desc1")
        index.add_preference("pref1", "value1")

        definitions = index.get_all_definitions()
        self.assertEqual(len(definitions), 2)
        for d in definitions:
            self.assertEqual(d.entry_type, "definition")

    def test_get_context_summary(self):
        """Test context summary generation."""
        index = AlignmentIndex()
        index.add_definition("spark", "fast predictor")
        index.add_pattern("testing", "use unittest")
        index.add_preference("style", "PEP8")
        index.add_goal("optimize", "improve speed")

        summary = index.get_context_summary()

        self.assertIsInstance(summary, str)
        self.assertIn("Alignment Context", summary)
        self.assertIn("spark", summary)
        self.assertIn("testing", summary)
        self.assertIn("style", summary)
        self.assertIn("optimize", summary)

    def test_load_from_markdown(self):
        """Test loading from markdown file."""
        index = AlignmentIndex()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create markdown file with definitions
            md_path = os.path.join(tmpdir, "definitions.md")
            with open(md_path, 'w') as f:
                f.write("""# My Definitions

## Definitions
- **spark**: fast statistical predictor
- **alignment**: human-AI context matching

## Patterns
- **testing**: use unittest framework
""")

            count = index.load_from_markdown(md_path)

            # Should load 3 entries
            self.assertEqual(count, 3)

            # Check definitions loaded
            spark_entries = index.lookup("spark")
            self.assertEqual(len(spark_entries), 1)
            self.assertEqual(spark_entries[0].value, "fast statistical predictor")

            # Check patterns loaded
            testing_entries = index.lookup("testing")
            self.assertEqual(len(testing_entries), 1)
            self.assertEqual(testing_entries[0].entry_type, "pattern")

    def test_save_and_load(self):
        """Test saving and loading index."""
        index = AlignmentIndex()
        index.add_definition("term1", "meaning1", tags=["tag1"])
        index.add_pattern("pattern1", "desc1")
        index.add_preference("pref1", "value1")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "alignment.json")

            # Save
            index.save(path)
            self.assertTrue(os.path.exists(path))

            # Load
            loaded = AlignmentIndex.load(path)

            # Check entries preserved
            self.assertEqual(len(loaded), len(index))

            # Check specific entry
            term1_entries = loaded.lookup("term1")
            self.assertEqual(len(term1_entries), 1)
            self.assertEqual(term1_entries[0].value, "meaning1")
            self.assertEqual(term1_entries[0].tags, ["tag1"])


class TestSparkPredictor(unittest.TestCase):
    """Test SparkPredictor facade class."""

    def test_init(self):
        """Test initialization."""
        spark = SparkPredictor()
        self.assertIsInstance(spark.ngram, NGramModel)
        self.assertIsInstance(spark.alignment, AlignmentIndex)
        self.assertEqual(spark.ngram.n, 3)
        self.assertFalse(spark._trained)
        self.assertFalse(spark._alignment_loaded)

    def test_init_custom_order(self):
        """Test initialization with custom n-gram order."""
        spark = SparkPredictor(ngram_order=4)
        self.assertEqual(spark.ngram.n, 4)

    def test_train_from_documents(self):
        """Test training from document list."""
        spark = SparkPredictor()
        docs = [
            "neural networks process data",
            "machine learning is powerful",
            "deep learning uses neural networks"
        ]

        result = spark.train_from_documents(docs)

        # Method chaining
        self.assertIs(result, spark)

        # Check trained flag
        self.assertTrue(spark._trained)

        # Check ngram model trained
        self.assertGreater(spark.ngram.total_documents, 0)
        self.assertGreater(len(spark.ngram.vocab), 0)

    def test_train_from_directory(self):
        """Test training from directory."""
        spark = SparkPredictor()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            with open(os.path.join(tmpdir, "file1.txt"), 'w') as f:
                f.write("neural networks are powerful")
            with open(os.path.join(tmpdir, "file2.md"), 'w') as f:
                f.write("machine learning uses data")
            with open(os.path.join(tmpdir, "ignored.xyz"), 'w') as f:
                f.write("this should be ignored")

            result = spark.train_from_directory(tmpdir)

            # Method chaining
            self.assertIs(result, spark)

            # Should be trained
            self.assertTrue(spark._trained)

            # Should have loaded 2 documents (txt and md, not xyz)
            self.assertGreater(spark.ngram.total_documents, 0)

    def test_prime(self):
        """Test query priming."""
        spark = SparkPredictor()
        docs = [
            "neural networks process data efficiently",
            "machine learning models are powerful"
        ]
        spark.train_from_documents(docs)

        result = spark.prime("neural networks")

        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertIn('query', result)
        self.assertIn('keywords', result)
        self.assertIn('completions', result)
        self.assertIn('alignment', result)
        self.assertIn('is_trained', result)

        # Check keywords extracted
        self.assertIn("neural", result['keywords'])
        self.assertIn("networks", result['keywords'])

        # Check completions present (model is trained)
        self.assertTrue(result['is_trained'])
        self.assertIsInstance(result['completions'], list)

    def test_complete(self):
        """Test word completion."""
        spark = SparkPredictor()
        docs = [
            "the cat sat on the mat",
            "the cat ran fast"
        ]
        spark.train_from_documents(docs)

        completions = spark.complete("the cat", top_k=3)

        self.assertIsInstance(completions, list)
        self.assertGreater(len(completions), 0)

        # Check format
        for word, prob in completions:
            self.assertIsInstance(word, str)
            self.assertIsInstance(prob, float)

    def test_complete_sequence(self):
        """Test sequence completion."""
        spark = SparkPredictor()
        docs = [
            "neural networks process data efficiently",
            "neural networks learn patterns quickly"
        ]
        spark.train_from_documents(docs)

        completed = spark.complete_sequence("neural networks", length=2)

        self.assertIsInstance(completed, str)
        self.assertTrue(completed.startswith("neural networks"))
        # Should have added words
        self.assertGreater(len(completed.split()), 2)

    def test_add_and_get_alignment(self):
        """Test adding and retrieving alignment context."""
        spark = SparkPredictor()

        # Add definition
        spark.add_definition("spark", "fast predictor")
        self.assertTrue(spark._alignment_loaded)

        # Retrieve it
        context = spark.get_alignment_context("spark")
        self.assertIsNotNone(context)
        self.assertEqual(context['key'], "spark")
        self.assertEqual(context['value'], "fast predictor")
        self.assertEqual(context['type'], "definition")

        # Add pattern
        spark.add_pattern("testing", "use unittest")

        # Add preference
        spark.add_preference("style", "PEP8")

        # Get summary
        summary = spark.get_context_summary()
        self.assertIn("spark", summary)
        self.assertIn("testing", summary)
        self.assertIn("style", summary)

    def test_save_and_load(self):
        """Test saving and loading predictor."""
        spark = SparkPredictor()
        docs = ["neural networks process data"]
        spark.train_from_documents(docs)
        spark.add_definition("spark", "fast predictor")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            spark.save(tmpdir)

            # Check files created
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "ngram.json")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "alignment.json")))

            # Load
            loaded = SparkPredictor.load(tmpdir)

            # Check state preserved
            self.assertTrue(loaded._trained)
            self.assertTrue(loaded._alignment_loaded)

            # Check ngram model
            self.assertEqual(loaded.ngram.vocab, spark.ngram.vocab)

            # Check alignment
            self.assertEqual(len(loaded.alignment), len(spark.alignment))

    def test_get_context_summary(self):
        """Test getting alignment context summary."""
        spark = SparkPredictor()
        spark.add_definition("term1", "meaning1")
        spark.add_pattern("pattern1", "desc1")

        summary = spark.get_context_summary()

        self.assertIsInstance(summary, str)
        self.assertIn("term1", summary)
        self.assertIn("pattern1", summary)


class TestBenchmarks(unittest.TestCase):
    """Benchmark tests for performance requirements."""

    def test_training_speed_small_corpus(self):
        """Test that training on 100 docs completes in <1s."""
        import time

        model = NGramModel(n=3)

        # Generate 100 simple documents
        docs = []
        for i in range(100):
            doc = f"document {i} contains neural networks and machine learning models"
            docs.append(doc)

        start = time.time()
        model.train(docs)
        elapsed = time.time() - start

        # Should complete in under 1 second
        self.assertLess(elapsed, 1.0, f"Training took {elapsed:.3f}s, expected <1s")

    def test_prediction_speed(self):
        """Test that 1000 predictions complete in <1s."""
        import time

        model = NGramModel(n=3)

        # Train on sample data
        docs = [
            "neural networks process data efficiently",
            "machine learning models learn patterns",
            "deep learning uses neural networks"
        ] * 10  # 30 docs total
        model.train(docs)

        # Time 1000 predictions
        start = time.time()
        for _ in range(1000):
            model.predict(["neural", "networks"], top_k=5)
        elapsed = time.time() - start

        # Should complete in under 1 second
        self.assertLess(elapsed, 1.0, f"1000 predictions took {elapsed:.3f}s, expected <1s")


# Additional edge case tests

class TestNGramModelEdgeCases(unittest.TestCase):
    """Test edge cases for NGramModel."""

    def test_empty_documents(self):
        """Test training with empty documents."""
        model = NGramModel(n=3)
        model.train(["", "", ""])

        self.assertEqual(model.total_documents, 0)
        self.assertEqual(model.total_tokens, 0)

    def test_single_word_document(self):
        """Test single word document."""
        model = NGramModel(n=2)
        model.train(["hello"])

        self.assertEqual(model.total_documents, 1)
        self.assertEqual(model.total_tokens, 1)
        self.assertIn("hello", model.vocab)

    def test_predict_empty_context(self):
        """Test prediction with empty context."""
        model = NGramModel(n=3)
        model.train(["the cat sat on the mat"])

        # Empty context should use START tokens
        predictions = model.predict([], top_k=5)
        self.assertIsInstance(predictions, list)

    def test_predict_long_context(self):
        """Test prediction with context longer than n-1."""
        model = NGramModel(n=3)
        model.train(["the quick brown fox jumps over the lazy dog"])

        # Should use only last n-1 words
        predictions = model.predict(["the", "quick", "brown", "fox", "jumps"], top_k=5)
        self.assertIsInstance(predictions, list)

    def test_perplexity_empty_text(self):
        """Test perplexity on empty text."""
        model = NGramModel(n=3)
        model.train(["some training text"])

        perplexity = model.perplexity("")
        self.assertEqual(perplexity, float('inf'))


class TestAlignmentIndexEdgeCases(unittest.TestCase):
    """Test edge cases for AlignmentIndex."""

    def test_lookup_nonexistent_term(self):
        """Test looking up nonexistent term."""
        index = AlignmentIndex()
        index.add_definition("term1", "meaning1")

        entries = index.lookup("nonexistent")
        self.assertEqual(len(entries), 0)

    def test_search_empty_query(self):
        """Test search with empty query."""
        index = AlignmentIndex()
        index.add_definition("term1", "meaning1")

        results = index.search("", top_k=5)
        # Should return empty or handle gracefully
        self.assertIsInstance(results, list)

    def test_multiple_entries_same_key(self):
        """Test multiple entries for same key."""
        index = AlignmentIndex()
        index.add_definition("term", "meaning1")
        index.add_definition("term", "meaning2")

        entries = index.lookup("term")
        self.assertEqual(len(entries), 2)

        # Most recent first
        self.assertEqual(entries[0].value, "meaning2")
        self.assertEqual(entries[1].value, "meaning1")

    def test_load_nonexistent_markdown(self):
        """Test loading from nonexistent markdown file."""
        index = AlignmentIndex()
        count = index.load_from_markdown("/nonexistent/path.md")
        self.assertEqual(count, 0)

    def test_load_empty_json(self):
        """Test loading from path that doesn't exist."""
        index = AlignmentIndex.load("/nonexistent/path.json")
        self.assertEqual(len(index), 0)


class TestSparkPredictorEdgeCases(unittest.TestCase):
    """Test edge cases for SparkPredictor."""

    def test_complete_without_training(self):
        """Test completion before training."""
        spark = SparkPredictor()

        # Should return empty list
        completions = spark.complete("some text", top_k=5)
        self.assertEqual(completions, [])

    def test_complete_sequence_without_training(self):
        """Test sequence completion before training."""
        spark = SparkPredictor()

        # Should return original prefix
        completed = spark.complete_sequence("some text", length=3)
        self.assertEqual(completed, "some text")

    def test_prime_without_training(self):
        """Test priming without training."""
        spark = SparkPredictor()

        result = spark.prime("neural networks")

        self.assertFalse(result['is_trained'])
        self.assertEqual(result['completions'], [])

    def test_get_alignment_context_missing(self):
        """Test getting context for nonexistent term."""
        spark = SparkPredictor()
        spark.add_definition("existing", "value")

        context = spark.get_alignment_context("nonexistent")
        self.assertIsNone(context)

    def test_load_from_nonexistent_directory(self):
        """Test loading from nonexistent directory."""
        spark = SparkPredictor.load("/nonexistent/path")

        # Should create empty predictor
        self.assertFalse(spark._trained)
        self.assertFalse(spark._alignment_loaded)

    def test_train_from_nonexistent_directory(self):
        """Test training from nonexistent directory."""
        spark = SparkPredictor()
        result = spark.train_from_directory("/nonexistent/path")

        # Should handle gracefully
        self.assertIs(result, spark)
        self.assertFalse(spark._trained)


if __name__ == '__main__':
    unittest.main()
