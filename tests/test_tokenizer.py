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


class TestSplitIdentifier(unittest.TestCase):
    """Test the split_identifier function."""

    def test_camel_case(self):
        """Test splitting camelCase identifiers."""
        from cortical.tokenizer import split_identifier
        self.assertEqual(split_identifier("getUserCredentials"), ["get", "user", "credentials"])
        self.assertEqual(split_identifier("processData"), ["process", "data"])

    def test_pascal_case(self):
        """Test splitting PascalCase identifiers."""
        from cortical.tokenizer import split_identifier
        self.assertEqual(split_identifier("UserCredentials"), ["user", "credentials"])
        self.assertEqual(split_identifier("DataProcessor"), ["data", "processor"])

    def test_snake_case(self):
        """Test splitting snake_case identifiers."""
        from cortical.tokenizer import split_identifier
        self.assertEqual(split_identifier("get_user_data"), ["get", "user", "data"])
        self.assertEqual(split_identifier("process_http_request"), ["process", "http", "request"])

    def test_screaming_snake_case(self):
        """Test splitting SCREAMING_SNAKE_CASE identifiers."""
        from cortical.tokenizer import split_identifier
        self.assertEqual(split_identifier("MAX_RETRY_COUNT"), ["max", "retry", "count"])

    def test_acronyms(self):
        """Test handling of acronyms in identifiers."""
        from cortical.tokenizer import split_identifier
        self.assertEqual(split_identifier("XMLParser"), ["xml", "parser"])
        self.assertEqual(split_identifier("parseHTTPResponse"), ["parse", "http", "response"])
        self.assertEqual(split_identifier("getURLString"), ["get", "url", "string"])

    def test_mixed_case_with_underscore(self):
        """Test mixed camelCase and snake_case."""
        from cortical.tokenizer import split_identifier
        result = split_identifier("get_UserData")
        self.assertIn("get", result)
        self.assertIn("user", result)
        self.assertIn("data", result)

    def test_single_word(self):
        """Test single word identifiers."""
        from cortical.tokenizer import split_identifier
        self.assertEqual(split_identifier("process"), ["process"])
        self.assertEqual(split_identifier("data"), ["data"])

    def test_empty_string(self):
        """Test empty string input."""
        from cortical.tokenizer import split_identifier
        self.assertEqual(split_identifier(""), [])


class TestCodeAwareTokenization(unittest.TestCase):
    """Test code-aware tokenization with identifier splitting."""

    def test_split_identifiers_disabled_by_default(self):
        """Test that identifier splitting is disabled by default."""
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("getUserCredentials")
        self.assertEqual(tokens, ["getusercredentials"])

    def test_split_identifiers_enabled(self):
        """Test tokenization with identifier splitting enabled."""
        tokenizer = Tokenizer(split_identifiers=True)
        tokens = tokenizer.tokenize("getUserCredentials")
        self.assertIn("getusercredentials", tokens)
        self.assertIn("get", tokens)
        self.assertIn("user", tokens)
        self.assertIn("credentials", tokens)

    def test_split_identifiers_snake_case(self):
        """Test splitting snake_case in tokenization."""
        tokenizer = Tokenizer(split_identifiers=True)
        tokens = tokenizer.tokenize("process_user_data")
        self.assertIn("process_user_data", tokens)
        self.assertIn("process", tokens)
        self.assertIn("user", tokens)
        self.assertIn("data", tokens)

    def test_split_identifiers_preserves_context(self):
        """Test that split tokens appear alongside regular tokens."""
        tokenizer = Tokenizer(split_identifiers=True)
        tokens = tokenizer.tokenize("The getUserCredentials function returns data")
        self.assertIn("getusercredentials", tokens)
        self.assertIn("credentials", tokens)
        self.assertIn("function", tokens)
        self.assertIn("returns", tokens)
        self.assertIn("data", tokens)

    def test_split_identifiers_override(self):
        """Test overriding split_identifiers at call time."""
        tokenizer = Tokenizer(split_identifiers=False)
        # Override to True
        tokens = tokenizer.tokenize("getUserData", split_identifiers=True)
        self.assertIn("get", tokens)
        self.assertIn("user", tokens)

    def test_no_duplicate_tokens(self):
        """Test that split tokens don't create duplicates."""
        tokenizer = Tokenizer(split_identifiers=True)
        tokens = tokenizer.tokenize("data process_data getData")
        # 'data' should appear only once
        self.assertEqual(tokens.count("data"), 1)

    def test_stop_words_filtered_from_splits(self):
        """Test that stop words in split parts are filtered."""
        tokenizer = Tokenizer(split_identifiers=True)
        # 'the' is a stop word
        tokens = tokenizer.tokenize("getTheData")
        self.assertNotIn("the", tokens)
        self.assertIn("data", tokens)

    def test_min_length_applied_to_splits(self):
        """Test that min_word_length applies to split parts."""
        tokenizer = Tokenizer(split_identifiers=True, min_word_length=4)
        tokens = tokenizer.tokenize("getUserID")
        # 'id' is too short (length 2)
        self.assertNotIn("id", tokens)
        self.assertIn("user", tokens)


if __name__ == "__main__":
    unittest.main(verbosity=2)
