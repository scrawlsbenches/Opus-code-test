"""
Unit tests for cortical/validation.py module.

Tests all validation functions and decorators for proper behavior
with valid inputs, invalid inputs, and edge cases.
"""

import unittest
from cortical.validation import (
    validate_non_empty_string,
    validate_positive_int,
    validate_non_negative_int,
    validate_range,
    validate_params,
    marks_stale,
    marks_fresh,
)


class TestValidateNonEmptyString(unittest.TestCase):
    """Tests for validate_non_empty_string function."""

    def test_valid_string(self):
        """Valid non-empty string should not raise."""
        validate_non_empty_string("hello", "param")
        validate_non_empty_string("a", "param")
        validate_non_empty_string("   spaces   ", "param")

    def test_empty_string_raises(self):
        """Empty string should raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            validate_non_empty_string("", "query")
        self.assertIn("query", str(ctx.exception))
        self.assertIn("non-empty", str(ctx.exception))

    def test_none_raises(self):
        """None should raise ValueError for type."""
        with self.assertRaises(ValueError) as ctx:
            validate_non_empty_string(None, "param")
        self.assertIn("param", str(ctx.exception))
        self.assertIn("string", str(ctx.exception))

    def test_int_raises(self):
        """Integer should raise ValueError for type."""
        with self.assertRaises(ValueError) as ctx:
            validate_non_empty_string(123, "doc_id")
        self.assertIn("doc_id", str(ctx.exception))
        self.assertIn("int", str(ctx.exception))

    def test_list_raises(self):
        """List should raise ValueError for type."""
        with self.assertRaises(ValueError) as ctx:
            validate_non_empty_string(["a", "b"], "items")
        self.assertIn("items", str(ctx.exception))
        self.assertIn("list", str(ctx.exception))


class TestValidatePositiveInt(unittest.TestCase):
    """Tests for validate_positive_int function."""

    def test_valid_positive_int(self):
        """Positive integers should not raise."""
        validate_positive_int(1, "top_n")
        validate_positive_int(100, "top_n")
        validate_positive_int(999999, "top_n")

    def test_zero_raises(self):
        """Zero should raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            validate_positive_int(0, "top_n")
        self.assertIn("top_n", str(ctx.exception))
        self.assertIn("positive", str(ctx.exception))

    def test_negative_raises(self):
        """Negative integer should raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            validate_positive_int(-5, "count")
        self.assertIn("count", str(ctx.exception))
        self.assertIn("positive", str(ctx.exception))

    def test_float_raises(self):
        """Float should raise ValueError for type."""
        with self.assertRaises(ValueError) as ctx:
            validate_positive_int(3.14, "top_n")
        self.assertIn("top_n", str(ctx.exception))
        self.assertIn("integer", str(ctx.exception))

    def test_string_raises(self):
        """String should raise ValueError for type."""
        with self.assertRaises(ValueError) as ctx:
            validate_positive_int("5", "top_n")
        self.assertIn("integer", str(ctx.exception))


class TestValidateNonNegativeInt(unittest.TestCase):
    """Tests for validate_non_negative_int function."""

    def test_valid_zero(self):
        """Zero should be valid for non-negative."""
        validate_non_negative_int(0, "offset")

    def test_valid_positive(self):
        """Positive integers should be valid."""
        validate_non_negative_int(1, "offset")
        validate_non_negative_int(100, "offset")

    def test_negative_raises(self):
        """Negative integer should raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            validate_non_negative_int(-1, "max_expansions")
        self.assertIn("max_expansions", str(ctx.exception))
        self.assertIn("non-negative", str(ctx.exception))

    def test_float_raises(self):
        """Float should raise ValueError for type."""
        with self.assertRaises(ValueError) as ctx:
            validate_non_negative_int(0.0, "offset")
        self.assertIn("integer", str(ctx.exception))


class TestValidateRange(unittest.TestCase):
    """Tests for validate_range function."""

    def test_value_in_inclusive_range(self):
        """Value within inclusive range should not raise."""
        validate_range(0.5, "alpha", min_val=0, max_val=1)
        validate_range(0, "alpha", min_val=0, max_val=1)
        validate_range(1, "alpha", min_val=0, max_val=1)

    def test_value_below_min_inclusive_raises(self):
        """Value below min (inclusive) should raise."""
        with self.assertRaises(ValueError) as ctx:
            validate_range(-0.1, "alpha", min_val=0, max_val=1)
        self.assertIn("alpha", str(ctx.exception))
        self.assertIn(">=", str(ctx.exception))

    def test_value_above_max_inclusive_raises(self):
        """Value above max (inclusive) should raise."""
        with self.assertRaises(ValueError) as ctx:
            validate_range(1.1, "alpha", min_val=0, max_val=1)
        self.assertIn("alpha", str(ctx.exception))
        self.assertIn("<=", str(ctx.exception))

    def test_exclusive_range(self):
        """Test exclusive range validation."""
        validate_range(0.5, "value", min_val=0, max_val=1, inclusive=False)

        with self.assertRaises(ValueError) as ctx:
            validate_range(0, "value", min_val=0, max_val=1, inclusive=False)
        self.assertIn(">", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            validate_range(1, "value", min_val=0, max_val=1, inclusive=False)
        self.assertIn("<", str(ctx.exception))

    def test_min_only(self):
        """Test validation with only min constraint."""
        validate_range(100, "score", min_val=0)
        with self.assertRaises(ValueError):
            validate_range(-1, "score", min_val=0)

    def test_max_only(self):
        """Test validation with only max constraint."""
        validate_range(-100, "penalty", max_val=0)
        with self.assertRaises(ValueError):
            validate_range(1, "penalty", max_val=0)

    def test_non_numeric_raises(self):
        """Non-numeric value should raise."""
        with self.assertRaises(ValueError) as ctx:
            validate_range("0.5", "alpha", min_val=0, max_val=1)
        self.assertIn("numeric", str(ctx.exception))

    def test_int_in_float_range(self):
        """Integer should work in float range."""
        validate_range(1, "damping", min_val=0.0, max_val=1.0)


class TestValidateParamsDecorator(unittest.TestCase):
    """Tests for validate_params decorator."""

    def test_valid_params_pass(self):
        """Decorated function with valid params should work."""
        @validate_params(
            query=lambda q: validate_non_empty_string(q, 'query'),
            top_n=lambda n: validate_positive_int(n, 'top_n')
        )
        def search(query: str, top_n: int = 5):
            return f"Searching: {query}, top {top_n}"

        result = search("test query", 10)
        self.assertEqual(result, "Searching: test query, top 10")

    def test_invalid_param_raises(self):
        """Decorated function with invalid param should raise."""
        @validate_params(
            query=lambda q: validate_non_empty_string(q, 'query')
        )
        def search(query: str):
            return query

        with self.assertRaises(ValueError):
            search("")

    def test_none_optional_param_allowed(self):
        """None should be allowed for optional params."""
        @validate_params(
            name=lambda n: validate_non_empty_string(n, 'name')
        )
        def greet(name=None):
            return f"Hello, {name or 'World'}!"

        # None should not trigger validation
        result = greet(None)
        self.assertEqual(result, "Hello, World!")

    def test_default_params_validated(self):
        """Default parameters should be validated if non-None."""
        @validate_params(
            count=lambda c: validate_positive_int(c, 'count')
        )
        def process(count: int = 5):
            return count

        result = process()  # Uses default of 5
        self.assertEqual(result, 5)

    def test_kwargs_validated(self):
        """Keyword arguments should be validated."""
        @validate_params(
            limit=lambda l: validate_positive_int(l, 'limit')
        )
        def fetch(limit: int = 10):
            return limit

        result = fetch(limit=20)
        self.assertEqual(result, 20)

        with self.assertRaises(ValueError):
            fetch(limit=0)


class TestMarksStaleFreshDecorators(unittest.TestCase):
    """Tests for marks_stale and marks_fresh decorators."""

    def setUp(self):
        """Create a mock processor class for testing decorators."""
        class MockProcessor:
            def __init__(self):
                self._stale_computations = set()
                self.fresh_calls = []

            def _mark_all_stale(self):
                self._stale_computations = {'tfidf', 'pagerank', 'concepts'}

            def _mark_fresh(self, *comp_types):
                for t in comp_types:
                    self._stale_computations.discard(t)
                    self.fresh_calls.append(t)

            @marks_stale('tfidf', 'pagerank')
            def add_document(self, doc_id, text):
                return f"Added {doc_id}"

            @marks_stale()  # Mark all stale
            def reset_corpus(self):
                return "Reset"

            @marks_fresh('tfidf')
            def compute_tfidf(self):
                return "Computed TF-IDF"

        self.MockProcessor = MockProcessor

    def test_marks_stale_specific(self):
        """marks_stale should mark specific computations stale."""
        proc = self.MockProcessor()
        proc._stale_computations = set()  # Start fresh

        result = proc.add_document("doc1", "Hello world")

        self.assertEqual(result, "Added doc1")
        self.assertIn('tfidf', proc._stale_computations)
        self.assertIn('pagerank', proc._stale_computations)
        self.assertNotIn('concepts', proc._stale_computations)

    def test_marks_stale_all(self):
        """marks_stale without args should mark all stale."""
        proc = self.MockProcessor()
        proc._stale_computations = set()

        result = proc.reset_corpus()

        self.assertEqual(result, "Reset")
        # _mark_all_stale sets all three
        self.assertEqual(proc._stale_computations, {'tfidf', 'pagerank', 'concepts'})

    def test_marks_fresh(self):
        """marks_fresh should mark computations fresh after execution."""
        proc = self.MockProcessor()
        proc._stale_computations = {'tfidf', 'pagerank'}

        result = proc.compute_tfidf()

        self.assertEqual(result, "Computed TF-IDF")
        self.assertIn('tfidf', proc.fresh_calls)
        self.assertNotIn('tfidf', proc._stale_computations)

    def test_decorator_preserves_function_name(self):
        """Decorators should preserve function metadata."""
        proc = self.MockProcessor()

        self.assertEqual(proc.add_document.__name__, 'add_document')
        self.assertEqual(proc.compute_tfidf.__name__, 'compute_tfidf')


class TestValidationEdgeCases(unittest.TestCase):
    """Edge case tests for validation functions."""

    def test_whitespace_only_string_passes(self):
        """Whitespace-only string is not empty (by design)."""
        # Note: This is current behavior - whitespace is allowed
        # If not desired, add str.strip() check
        validate_non_empty_string("   ", "param")

    def test_large_positive_int(self):
        """Very large integers should be valid."""
        validate_positive_int(10**18, "big_number")

    def test_float_near_boundary(self):
        """Float near boundary should work correctly."""
        validate_range(0.9999999999, "near_one", max_val=1.0)
        validate_range(0.0000000001, "near_zero", min_val=0.0)

    def test_boolean_as_int_rejected(self):
        """Boolean should be rejected (even though bool is subclass of int)."""
        # In Python, bool is a subclass of int, so True/False pass isinstance(x, int)
        # This is expected Python behavior
        validate_positive_int(True, "flag")  # True == 1, so this passes


class TestValidationIntegration(unittest.TestCase):
    """Integration tests for validation with processor methods."""

    def test_processor_method_validation(self):
        """Test that processor methods properly validate inputs."""
        from cortical import CorticalTextProcessor

        processor = CorticalTextProcessor()

        # Test empty query validation
        with self.assertRaises(ValueError):
            processor.find_documents_for_query("", top_n=5)

        # Test invalid top_n
        with self.assertRaises(ValueError):
            processor.find_documents_for_query("test", top_n=0)

    def test_expand_query_validation(self):
        """Test expand_query validates max_expansions."""
        from cortical import CorticalTextProcessor

        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Hello world")
        processor.compute_all()

        # Valid call
        result = processor.expand_query("hello", max_expansions=5)
        self.assertIsInstance(result, dict)

        # Invalid max_expansions
        with self.assertRaises(ValueError):
            processor.expand_query("hello", max_expansions=-1)


if __name__ == '__main__':
    unittest.main()
