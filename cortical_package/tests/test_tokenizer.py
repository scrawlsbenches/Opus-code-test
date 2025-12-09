"""Tests for the Tokenizer class."""

import unittest
import sys
sys.path.insert(0, '..')

from cortical import Tokenizer


class TestTokenizer(unittest.TestCase):
    """Test the Tokenizer class."""
    
    def setUp(self):
        self.tokenizer = Tokenizer()
    
    def test_basic_tokenization(self):
        """Test basic word extraction."""
        tokens = self.tokenizer.tokenize("Hello world")
        self.assertEqual(tokens, ["hello", "world"])
    
    def test_stop_word_removal(self):
        """Test that stop words are removed."""
        tokens = self.tokenizer.tokenize("The quick brown fox")
        self.assertNotIn("the", tokens)
        self.assertIn("quick", tokens)
        self.assertIn("brown", tokens)
        self.assertIn("fox", tokens)
    
    def test_minimum_length(self):
        """Test minimum word length filtering."""
        tokens = self.tokenizer.tokenize("I am a test of tokenization")
        for token in tokens:
            self.assertGreaterEqual(len(token), 3)
    
    def test_lowercase(self):
        """Test that tokens are lowercased."""
        tokens = self.tokenizer.tokenize("NEURAL Networks PROCESSING")
        self.assertEqual(tokens, ["neural", "networks", "processing"])
    
    def test_alphanumeric(self):
        """Test handling of alphanumeric tokens."""
        tokens = self.tokenizer.tokenize("word2vec and bert3 models")
        self.assertIn("word2vec", tokens)
        self.assertIn("bert3", tokens)
    
    def test_extract_bigrams(self):
        """Test bigram extraction."""
        tokens = ["neural", "network", "processing"]
        bigrams = self.tokenizer.extract_ngrams(tokens, n=2)
        self.assertEqual(bigrams, ["neural network", "network processing"])
    
    def test_extract_trigrams(self):
        """Test trigram extraction."""
        tokens = ["neural", "network", "information", "processing"]
        trigrams = self.tokenizer.extract_ngrams(tokens, n=3)
        self.assertEqual(len(trigrams), 2)


class TestTokenizerStemming(unittest.TestCase):
    """Test tokenizer stemming and word variants."""
    
    def setUp(self):
        self.tokenizer = Tokenizer()
    
    def test_stem_basic(self):
        """Test basic stemming."""
        self.assertEqual(self.tokenizer.stem("running"), "runn")
        self.assertEqual(self.tokenizer.stem("processing"), "process")
    
    def test_stem_preserves_short_words(self):
        """Test that short words are not stemmed."""
        self.assertEqual(self.tokenizer.stem("run"), "run")
        self.assertEqual(self.tokenizer.stem("the"), "the")
    
    def test_get_word_variants_basic(self):
        """Test basic word variant generation."""
        variants = self.tokenizer.get_word_variants("bread")
        self.assertIn("bread", variants)
        self.assertIn("sourdough", variants)
    
    def test_get_word_variants_includes_plural(self):
        """Test that variants include plural forms."""
        variants = self.tokenizer.get_word_variants("network")
        self.assertIn("network", variants)
        self.assertIn("networks", variants)
    
    def test_word_mappings_brain(self):
        """Test brain-related word mappings."""
        variants = self.tokenizer.get_word_variants("brain")
        self.assertIn("neural", variants)
        self.assertIn("cortical", variants)


if __name__ == "__main__":
    unittest.main(verbosity=2)
