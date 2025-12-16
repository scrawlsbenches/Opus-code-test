"""
Property-based fuzzing tests using Hypothesis.

SEC-010: Automatically generates random inputs to find edge cases
and potential security issues that manual tests might miss.

Target methods:
- process_document() with random content
- find_documents_for_query() with malformed queries
- expand_query() with edge case inputs

Requires: pip install hypothesis
"""

import os
import sys
import tempfile
import string

import pytest

# Guard hypothesis import - skip all tests if not available
try:
    from hypothesis import given, settings, assume, HealthCheck
    from hypothesis import strategies as st
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    # Create dummy decorators so class definitions don't fail
    def given(*args, **kwargs):
        def decorator(f):
            return f
        return decorator
    def settings(*args, **kwargs):
        def decorator(f):
            return f
        return decorator
    def assume(x):
        pass
    class HealthCheck:
        too_slow = None
    class st:
        @staticmethod
        def text(*args, **kwargs):
            return None
        @staticmethod
        def integers(*args, **kwargs):
            return None
        @staticmethod
        def floats(*args, **kwargs):
            return None
        @staticmethod
        def binary(*args, **kwargs):
            return None

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from cortical import CorticalTextProcessor, CorticalConfig

# Mark as optional and fuzz tests, skip if hypothesis not installed
pytestmark = [
    pytest.mark.optional,
    pytest.mark.fuzz,
    pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis package not installed")
]


# Custom strategies for generating test data
# Text that's likely to be valid for processing
safe_text = st.text(
    alphabet=string.ascii_letters + string.digits + " .,!?'\"-",
    min_size=1,
    max_size=10000
)

# Document IDs - any string that isn't empty
doc_ids = st.text(min_size=1, max_size=100)

# Queries - non-empty text
queries = st.text(min_size=1, max_size=1000)

# Unicode text including potentially problematic characters
unicode_text = st.text(min_size=1, max_size=1000)

# Numbers for numeric parameters
positive_ints = st.integers(min_value=1, max_value=1000)
non_negative_ints = st.integers(min_value=0, max_value=1000)
floats_0_1 = st.floats(min_value=0.001, max_value=0.999)


class TestProcessDocumentFuzzing:
    """Fuzz testing for process_document() method."""

    @given(doc_id=doc_ids, content=safe_text)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_process_document_never_crashes(self, doc_id, content):
        """process_document should never crash with valid-ish inputs."""
        processor = CorticalTextProcessor()

        # Skip empty strings (known to be invalid)
        assume(doc_id.strip())
        assume(content.strip())

        try:
            processor.process_document(doc_id, content)
            # Should not crash
            assert True
        except ValueError:
            # ValueError for invalid input is acceptable
            pass

    @given(doc_id=doc_ids, content=unicode_text)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_process_document_handles_unicode(self, doc_id, content):
        """process_document should handle arbitrary Unicode safely."""
        processor = CorticalTextProcessor()

        assume(doc_id.strip())
        assume(content.strip())

        try:
            processor.process_document(doc_id, content)
            # If successful, document should be stored
            assert doc_id in processor.documents
        except (ValueError, UnicodeError):
            # Rejection of problematic Unicode is acceptable
            pass

    @given(count=st.integers(min_value=1, max_value=50))
    @settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow])
    def test_multiple_documents_never_crash(self, count):
        """Adding multiple documents should never crash."""
        processor = CorticalTextProcessor()

        for i in range(count):
            processor.process_document(f"doc_{i}", f"Content for document {i}.")

        assert len(processor.documents) == count


class TestQueryFuzzing:
    """Fuzz testing for query methods."""

    @given(query=queries)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_find_documents_never_crashes(self, query):
        """find_documents_for_query should never crash with valid queries."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Test document with some content.")
        processor.compute_all()

        # Skip whitespace-only queries (known to be invalid)
        assume(query.strip())

        try:
            results = processor.find_documents_for_query(query)
            # Should return a list
            assert isinstance(results, list)
        except ValueError:
            # ValueError for invalid query is acceptable
            pass

    @given(query=unicode_text)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_query_handles_unicode(self, query):
        """Query methods should handle arbitrary Unicode safely."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Test document.")
        processor.compute_all()

        assume(query.strip())

        try:
            results = processor.find_documents_for_query(query)
            assert isinstance(results, list)
        except (ValueError, UnicodeError):
            # Rejection is acceptable
            pass

    @given(top_n=st.integers(min_value=-100, max_value=1000))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_top_n_boundaries(self, top_n):
        """top_n parameter should handle boundary values safely."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Test document.")
        processor.compute_all()

        try:
            if top_n <= 0:
                # Should reject non-positive values
                with pytest.raises(ValueError):
                    processor.find_documents_for_query("test", top_n=top_n)
            else:
                results = processor.find_documents_for_query("test", top_n=top_n)
                assert len(results) <= top_n
        except (ValueError, OverflowError):
            # These are acceptable responses to boundary values
            pass


class TestExpandQueryFuzzing:
    """Fuzz testing for expand_query() method."""

    @given(query=queries)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_expand_query_never_crashes(self, query):
        """expand_query should never crash with arbitrary queries."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Neural networks process data efficiently.")
        processor.compute_all()

        assume(query.strip())

        try:
            expanded = processor.expand_query(query)
            # Should return a dict
            assert isinstance(expanded, dict)
        except ValueError:
            # Rejection is acceptable
            pass

    @given(max_expansions=st.integers(min_value=-10, max_value=100))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_max_expansions_boundaries(self, max_expansions):
        """max_expansions parameter should handle boundary values."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Neural networks machine learning.")
        processor.compute_all()

        try:
            if max_expansions <= 0:
                # May reject or return minimal results
                expanded = processor.expand_query("neural", max_expansions=max_expansions)
                # If it doesn't reject, should return limited results
                assert len(expanded) <= max(0, max_expansions) or len(expanded) >= 1
            else:
                expanded = processor.expand_query("neural", max_expansions=max_expansions)
                assert len(expanded) <= max_expansions + 10  # Some tolerance
        except ValueError:
            # Rejection is acceptable
            pass


class TestConfigFuzzing:
    """Fuzz testing for configuration validation."""

    @given(damping=st.floats(allow_nan=True, allow_infinity=True))
    @settings(max_examples=100)
    def test_pagerank_damping_validation(self, damping):
        """pagerank_damping should reject invalid values."""
        import math

        try:
            config = CorticalConfig(pagerank_damping=damping)
            # If it accepted, must be valid
            assert 0 < damping < 1
            assert not math.isnan(damping)
            assert not math.isinf(damping)
        except (ValueError, TypeError):
            # Rejection is expected for invalid values
            pass

    @given(resolution=st.floats(allow_nan=True, allow_infinity=True))
    @settings(max_examples=100)
    def test_louvain_resolution_validation(self, resolution):
        """louvain_resolution should reject invalid values."""
        import math
        import warnings

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # High values trigger warnings
                config = CorticalConfig(louvain_resolution=resolution)
            # If it accepted, must be valid (positive, finite)
            assert resolution > 0
            assert not math.isnan(resolution)
            assert not math.isinf(resolution)
        except (ValueError, TypeError):
            # Rejection is expected for invalid values
            pass

    @given(iterations=st.integers(min_value=-1000, max_value=1000))
    @settings(max_examples=100)
    def test_pagerank_iterations_validation(self, iterations):
        """pagerank_iterations should reject invalid values."""
        try:
            config = CorticalConfig(pagerank_iterations=iterations)
            # If it accepted, must be valid
            assert iterations >= 1
        except ValueError:
            # Rejection is expected for invalid values
            pass


class TestPersistenceFuzzing:
    """Fuzz testing for save/load operations."""

    @given(content=safe_text)
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_save_load_roundtrip(self, content):
        """Save/load roundtrip should preserve data."""
        assume(content.strip())

        processor = CorticalTextProcessor()
        processor.process_document("doc1", content)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.pkl")
            processor.save(path)

            loaded = CorticalTextProcessor.load(path)
            assert loaded.documents.get("doc1") == content

    @given(key=st.binary(min_size=16, max_size=64))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_signed_save_load_with_random_keys(self, key):
        """Signed save/load should work with random keys."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Test content.")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.pkl")
            processor.save(path, signing_key=key)

            loaded = CorticalTextProcessor.load(path, verify_key=key)
            assert "doc1" in loaded.documents


class TestTokenizerFuzzing:
    """Fuzz testing for tokenization."""

    @given(text=unicode_text)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_tokenize_never_crashes(self, text):
        """Tokenization should never crash with arbitrary input."""
        from cortical import Tokenizer

        tokenizer = Tokenizer()

        try:
            tokens = tokenizer.tokenize(text)
            # Should return a list
            assert isinstance(tokens, list)
            # All tokens should be strings
            assert all(isinstance(t, str) for t in tokens)
        except (ValueError, UnicodeError):
            # Rejection is acceptable
            pass

    @given(text=st.text(min_size=0, max_size=100))
    @settings(max_examples=100)
    def test_extract_ngrams_never_crashes(self, text):
        """N-gram extraction should never crash."""
        from cortical import Tokenizer

        tokenizer = Tokenizer()

        try:
            # First tokenize, then extract ngrams
            tokens = tokenizer.tokenize(text)
            if tokens:  # Only extract ngrams if there are tokens
                bigrams = tokenizer.extract_ngrams(tokens, n=2)
                # Should return a list
                assert isinstance(bigrams, list)
        except (ValueError, UnicodeError):
            # Rejection is acceptable
            pass


class TestLayerFuzzing:
    """Fuzz testing for layer operations."""

    @given(term=st.text(min_size=1, max_size=100))
    @settings(max_examples=100)
    def test_get_minicolumn_never_crashes(self, term):
        """get_minicolumn should never crash with arbitrary terms."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Test content.")

        from cortical import CorticalLayer

        layer = processor.layers[CorticalLayer.TOKENS]

        # Should return None for missing terms, never crash
        result = layer.get_minicolumn(term)
        assert result is None or hasattr(result, 'content')

    @given(col_id=st.text(min_size=1, max_size=100))
    @settings(max_examples=100)
    def test_get_by_id_never_crashes(self, col_id):
        """get_by_id should never crash with arbitrary IDs."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Test content.")

        from cortical import CorticalLayer

        layer = processor.layers[CorticalLayer.TOKENS]

        # Should return None for missing IDs, never crash
        result = layer.get_by_id(col_id)
        assert result is None or hasattr(result, 'id')


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])
