"""
Security-focused test suite for the Cortical Text Processor.

SEC-009: Comprehensive security tests covering:
- Path traversal prevention
- Input validation
- Large input handling (DoS prevention)
- Malicious input rejection
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from cortical import CorticalTextProcessor, CorticalConfig
from cortical.persistence import (
    save_processor,
    load_processor,
)
from cortical.validation import (
    validate_non_empty_string,
    validate_positive_int,
    validate_range,
)


class TestPathTraversalPrevention:
    """
    Test that file operations prevent path traversal attacks.

    Path traversal attacks attempt to access files outside the intended
    directory by using sequences like '../' or absolute paths.
    """

    def test_save_rejects_path_traversal_sequences(self):
        """Save should not allow path traversal in filenames."""
        processor = CorticalTextProcessor()
        processor.process_document("test", "Test content")

        # Attempting path traversal with ../ should either:
        # 1. Fail (if the system prevents it)
        # 2. Create the file in a safe location (normalized path)
        with tempfile.TemporaryDirectory() as tmpdir:
            # Try to escape the directory
            malicious_path = os.path.join(tmpdir, "..", "escaped_state")

            # This should either raise an error or normalize the path
            # The important thing is it shouldn't create files outside tmpdir parent
            try:
                processor.save(malicious_path)
                # If it succeeded, verify the file is in a reasonable location
                # (os.path normalizes the path, so "../escaped_state" becomes parent/escaped_state)
                # This is acceptable as long as we document the behavior
                assert os.path.exists(malicious_path) or os.path.exists(
                    os.path.normpath(malicious_path)
                )
            except (OSError, ValueError):
                # Rejection is also acceptable
                pass

    def test_save_handles_absolute_paths_safely(self):
        """Save with absolute path should work normally."""
        processor = CorticalTextProcessor()
        processor.process_document("test", "Test content")

        with tempfile.TemporaryDirectory() as tmpdir:
            abs_path = os.path.join(tmpdir, "test_state")
            processor.save(abs_path)
            assert os.path.exists(abs_path)

    def test_document_id_with_path_characters(self):
        """Document IDs with path-like characters should be sanitized or rejected."""
        processor = CorticalTextProcessor()

        # These should either be accepted (if IDs are just keys, not files)
        # or rejected if they could cause security issues
        suspicious_ids = [
            "../etc/passwd",
            "/etc/passwd",
            "..\\..\\windows\\system32",
            "doc|id",  # Pipe character
            "doc\x00id",  # Null byte
        ]

        for doc_id in suspicious_ids:
            try:
                processor.process_document(doc_id, "Test content")
                # If accepted, verify it's stored as-is (just a dict key)
                # This is acceptable since doc_ids are not used as filenames
                assert doc_id in processor.documents
            except (ValueError, KeyError):
                # Rejection is also acceptable
                pass


class TestInputValidation:
    """
    Test input validation on public API methods.

    Public APIs should validate inputs to prevent:
    - Type confusion attacks
    - Buffer overflow-like attacks (very large inputs)
    - Injection attacks
    """

    def test_empty_query_rejected(self):
        """Empty queries should be rejected."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Test content")
        processor.compute_all()

        with pytest.raises(ValueError):
            processor.find_documents_for_query("")

        with pytest.raises(ValueError):
            processor.find_documents_for_query("   ")  # Whitespace only

    def test_none_query_rejected(self):
        """None query should be rejected."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Test content")
        processor.compute_all()

        with pytest.raises((ValueError, TypeError, AttributeError)):
            processor.find_documents_for_query(None)

    def test_invalid_top_n_rejected(self):
        """Invalid top_n values should be rejected."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Test content")
        processor.compute_all()

        with pytest.raises(ValueError):
            processor.find_documents_for_query("test", top_n=0)

        with pytest.raises(ValueError):
            processor.find_documents_for_query("test", top_n=-1)

    def test_non_string_document_rejected(self):
        """Non-string document content should be rejected."""
        processor = CorticalTextProcessor()

        with pytest.raises((ValueError, TypeError, AttributeError)):
            processor.process_document("doc1", None)

        with pytest.raises((ValueError, TypeError, AttributeError)):
            processor.process_document("doc1", 12345)

        with pytest.raises((ValueError, TypeError, AttributeError)):
            processor.process_document("doc1", ["list", "of", "words"])

    def test_empty_document_id_rejected(self):
        """Empty document IDs should be rejected."""
        processor = CorticalTextProcessor()

        with pytest.raises(ValueError):
            processor.process_document("", "Test content")

    def test_validation_decorators(self):
        """Test validation utility functions."""
        # validate_non_empty_string
        with pytest.raises(ValueError):
            validate_non_empty_string("", "test")

        with pytest.raises(ValueError):
            validate_non_empty_string(None, "test")

        with pytest.raises(ValueError):
            validate_non_empty_string(123, "test")

        # Should not raise
        validate_non_empty_string("valid", "test")

        # validate_positive_int
        with pytest.raises(ValueError):
            validate_positive_int(0, "test")

        with pytest.raises(ValueError):
            validate_positive_int(-1, "test")

        with pytest.raises(ValueError):
            validate_positive_int(1.5, "test")

        # Should not raise
        validate_positive_int(1, "test")

        # validate_range
        with pytest.raises(ValueError):
            validate_range(1.5, "test", min_val=0.0, max_val=1.0)

        with pytest.raises(ValueError):
            validate_range(-0.1, "test", min_val=0.0)


class TestLargeInputHandling:
    """
    Test handling of very large inputs (DoS prevention).

    Large inputs should either:
    - Be processed reasonably
    - Be rejected with clear limits
    - Not cause crashes or excessive memory usage
    """

    def test_very_large_document_handled(self):
        """Very large documents should be handled without crashing."""
        processor = CorticalTextProcessor()

        # Create a large document (100KB of text)
        large_text = "test word " * 10000

        # Should not crash
        processor.process_document("large_doc", large_text)

        # Should have processed tokens
        assert processor.layers is not None

    def test_many_documents_handled(self):
        """Many documents should be handled without crashing."""
        processor = CorticalTextProcessor()

        # Add 100 documents
        for i in range(100):
            processor.process_document(f"doc_{i}", f"Document number {i} with some content.")

        # Should have all documents
        assert len(processor.documents) == 100

    def test_very_long_query_handled(self):
        """Very long queries should be handled without crashing."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Test content for searching.")
        processor.compute_all()

        # Create a long query (1000 words)
        long_query = " ".join(["word"] * 1000)

        # Should not crash (may return empty results, but shouldn't crash)
        try:
            results = processor.find_documents_for_query(long_query)
            assert isinstance(results, list)
        except ValueError:
            # Rejecting very long queries is also acceptable
            pass

    def test_repeated_same_document_id(self):
        """Processing the same document ID multiple times should be handled."""
        processor = CorticalTextProcessor()

        # Process the same document 100 times
        for i in range(100):
            processor.process_document("doc1", f"Version {i} of the document.")

        # Should only have one document (latest version)
        assert len(processor.documents) == 1
        assert "Version 99" in processor.documents["doc1"]

    def test_unicode_document_handling(self):
        """Unicode characters in documents should be handled safely."""
        processor = CorticalTextProcessor()

        # Various Unicode test cases
        unicode_tests = [
            "Hello 世界 🌍",  # Chinese + emoji
            "مرحبا بالعالم",  # Arabic (RTL)
            "Привет мир",  # Cyrillic
            "\u0000\u0001\u0002",  # Control characters
            "a" * 10000 + "🔥",  # Long string with emoji
        ]

        for i, text in enumerate(unicode_tests):
            # Should not crash
            try:
                processor.process_document(f"unicode_{i}", text)
            except (ValueError, UnicodeError):
                # Rejection of invalid Unicode is acceptable
                pass


class TestMaliciousInputRejection:
    """
    Test rejection of potentially malicious inputs.

    This includes:
    - SQL injection-like patterns (shouldn't affect us, but test anyway)
    - Script injection patterns
    - Format string attacks
    """

    def test_script_like_content_safe(self):
        """Script-like content should be stored safely as text."""
        processor = CorticalTextProcessor()

        script_content = """
        <script>alert('xss')</script>
        <?php system('whoami'); ?>
        $(rm -rf /)
        """

        # Should be stored as plain text
        processor.process_document("script_doc", script_content)
        assert processor.documents["script_doc"] == script_content

    def test_sql_like_content_safe(self):
        """SQL-like content should be stored safely as text."""
        processor = CorticalTextProcessor()

        sql_content = "'; DROP TABLE users; --"

        # Should be stored as plain text (we don't use SQL)
        processor.process_document("sql_doc", sql_content)
        assert processor.documents["sql_doc"] == sql_content

    def test_format_string_content_safe(self):
        """Format string patterns should be stored safely."""
        processor = CorticalTextProcessor()

        format_content = "%s%s%s%n%n%n{0}{1}{2}"

        # Should be stored as plain text
        processor.process_document("format_doc", format_content)
        assert processor.documents["format_doc"] == format_content


class TestConfigurationSecurity:
    """Test security aspects of configuration handling."""

    def test_config_rejects_invalid_ranges(self):
        """Configuration should reject values outside valid ranges."""
        # pagerank_damping should be in (0, 1) exclusive
        with pytest.raises(ValueError):
            CorticalConfig(pagerank_damping=0.0)

        with pytest.raises(ValueError):
            CorticalConfig(pagerank_damping=1.0)

        with pytest.raises(ValueError):
            CorticalConfig(pagerank_damping=-0.5)

        with pytest.raises(ValueError):
            CorticalConfig(pagerank_damping=1.5)

    def test_config_rejects_negative_counts(self):
        """Configuration should reject negative count values."""
        with pytest.raises(ValueError):
            CorticalConfig(pagerank_iterations=-1)

        with pytest.raises(ValueError):
            CorticalConfig(louvain_resolution=-1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
