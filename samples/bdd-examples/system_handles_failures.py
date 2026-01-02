"""
System Reliability Behavioral Tests (Sample)
=============================================

Epic: System handles failures gracefully

As a system operator,
I want the system to recover gracefully from unexpected situations,
So that users experience minimal disruption.

Requirements:
- Invalid input should produce helpful error messages
- Resource exhaustion should be handled without crashes
- Concurrent operations should not corrupt data
- System should maintain consistency after failures

Run with: pytest samples/bdd-examples/system_handles_failures.py -v

NOTE: This is a SAMPLE file demonstrating BDD patterns.
      Real behavioral tests live in tests/behavioral/
"""

import pytest
import threading
from typing import Optional, Dict, Any


# ============================================================================
# SAMPLE INFRASTRUCTURE
# ============================================================================

class OperationError(Exception):
    """Raised when an operation fails."""
    pass


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


class MockIndexer:
    """
    Mock indexer for demonstration purposes.

    Simulates a document indexing system with failure handling.
    """

    def __init__(self):
        self._documents: Dict[str, str] = {}
        self._index_built = False
        self._lock = threading.Lock()

    def add_document(self, doc_id: str, content: str) -> None:
        """Add a document for indexing."""
        if not doc_id:
            raise ValidationError("Document ID cannot be empty")
        if not content:
            raise ValidationError("Document content cannot be empty")
        if len(doc_id) > 255:
            raise ValidationError("Document ID exceeds maximum length (255)")

        with self._lock:
            self._documents[doc_id] = content
            self._index_built = False

    def build_index(self) -> None:
        """Build the search index from documents."""
        with self._lock:
            if not self._documents:
                raise OperationError("Cannot build index: no documents")
            # Simulate index building
            self._index_built = True

    def search(self, query: str) -> list:
        """Search the index."""
        if not self._index_built:
            raise OperationError("Index not built. Call build_index() first.")
        if not query:
            return []

        results = []
        query_lower = query.lower()
        for doc_id, content in self._documents.items():
            if query_lower in content.lower():
                results.append({"doc_id": doc_id, "score": 1.0})
        return results

    @property
    def document_count(self) -> int:
        """Return the number of indexed documents."""
        return len(self._documents)


@pytest.fixture
def indexer():
    """Provide a fresh indexer for each test."""
    return MockIndexer()


# ============================================================================
# INPUT VALIDATION SCENARIOS
# ============================================================================

class TestInputValidation:
    """
    Epic: Robust Input Handling

    As a developer using the indexing API,
    I want clear validation errors for invalid input,
    So that I can fix problems quickly.
    """

    def test_empty_document_id_rejected(self, indexer):
        """
        Scenario: Empty document ID is rejected with clear error

        Given an indexer
        When I try to add a document with empty ID
        Then I receive a ValidationError
        And the error message explains the problem
        Because developers need actionable feedback.
        """
        # Given an indexer (from fixture)

        # When I try to add a document with empty ID
        with pytest.raises(ValidationError) as exc_info:
            indexer.add_document("", "Some content")

        # Then the error message explains the problem
        assert "empty" in str(exc_info.value).lower(), (
            f"Error should mention 'empty'. Got: {exc_info.value}"
        )

    def test_empty_content_rejected(self, indexer):
        """
        Scenario: Empty document content is rejected with clear error

        Given an indexer
        When I try to add a document with empty content
        Then I receive a ValidationError
        And the error message explains the problem.
        """
        # Given an indexer

        # When I try to add document with empty content
        with pytest.raises(ValidationError) as exc_info:
            indexer.add_document("doc_id", "")

        # Then the error explains the problem
        assert "content" in str(exc_info.value).lower() or "empty" in str(exc_info.value).lower(), (
            f"Error should mention 'content' or 'empty'. Got: {exc_info.value}"
        )

    def test_oversized_document_id_rejected(self, indexer):
        """
        Scenario: Excessively long document IDs are rejected

        Given an indexer
        When I try to add a document with a 500-character ID
        Then I receive a ValidationError
        Because unreasonable input should be rejected early.
        """
        # Given an indexer

        # When I try to add document with very long ID
        long_id = "x" * 500

        with pytest.raises(ValidationError) as exc_info:
            indexer.add_document(long_id, "Some content")

        # Then error mentions the length issue
        assert "length" in str(exc_info.value).lower() or "255" in str(exc_info.value), (
            f"Error should mention length limit. Got: {exc_info.value}"
        )


# ============================================================================
# OPERATION SEQUENCE SCENARIOS
# ============================================================================

class TestOperationSequence:
    """
    Epic: Predictable Operation Order

    As a developer using the indexing API,
    I want operations to fail predictably when used incorrectly,
    So that I can understand and fix my code.
    """

    def test_search_before_index_fails_with_clear_error(self, indexer):
        """
        Scenario: Search before indexing produces clear error

        Given an indexer with documents added but not indexed
        When I try to search
        Then I receive an OperationError
        And the error tells me to build the index first
        Because the API should guide correct usage.
        """
        # Given documents added but not indexed
        indexer.add_document("doc1.md", "Content about custom systems.")

        # When I try to search before building index
        with pytest.raises(OperationError) as exc_info:
            indexer.search("custom")

        # Then error explains what to do
        error_msg = str(exc_info.value).lower()
        assert "index" in error_msg and ("build" in error_msg or "not" in error_msg), (
            f"Error should mention building index. Got: {exc_info.value}"
        )

    def test_build_empty_index_fails(self, indexer):
        """
        Scenario: Building index with no documents fails clearly

        Given an empty indexer
        When I try to build the index
        Then I receive an OperationError
        And the error explains that documents are needed
        Because building nothing is likely a mistake.
        """
        # Given an empty indexer (no documents added)

        # When I try to build index
        with pytest.raises(OperationError) as exc_info:
            indexer.build_index()

        # Then error explains the problem
        assert "document" in str(exc_info.value).lower() or "no" in str(exc_info.value).lower(), (
            f"Error should mention no documents. Got: {exc_info.value}"
        )


# ============================================================================
# CONCURRENT OPERATION SCENARIOS
# ============================================================================

class TestConcurrentOperations:
    """
    Epic: Thread-Safe Operations

    As a developer running multi-threaded applications,
    I want the indexer to be thread-safe,
    So that concurrent operations don't corrupt data.
    """

    def test_concurrent_document_additions(self, indexer):
        """
        Scenario: Concurrent document additions don't lose data

        Given multiple threads adding documents simultaneously
        When all threads complete
        Then all documents are present in the indexer
        Because concurrent writes must be safe.
        """
        # Given setup for concurrent additions
        num_threads = 10
        docs_per_thread = 10
        errors = []

        def add_documents(thread_id):
            try:
                for i in range(docs_per_thread):
                    doc_id = f"thread_{thread_id}_doc_{i}.md"
                    content = f"Content from thread {thread_id} document {i}"
                    indexer.add_document(doc_id, content)
            except Exception as e:
                errors.append(e)

        # When multiple threads add documents simultaneously
        threads = [
            threading.Thread(target=add_documents, args=(t,))
            for t in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Then all documents are present
        assert not errors, f"Errors during concurrent adds: {errors}"

        expected_count = num_threads * docs_per_thread
        assert indexer.document_count == expected_count, (
            f"Expected {expected_count} documents, "
            f"got {indexer.document_count}. Data loss during concurrent writes!"
        )


# ============================================================================
# RECOVERY SCENARIOS
# ============================================================================

class TestRecoveryBehavior:
    """
    Epic: Graceful Recovery

    As a system operator,
    I want the system to remain usable after errors,
    So that one failure doesn't require a restart.
    """

    def test_system_usable_after_validation_error(self, indexer):
        """
        Scenario: System remains usable after validation error

        Given a validation error occurred
        When I retry with valid input
        Then the operation succeeds
        Because errors should not corrupt system state.
        """
        # Given a validation error occurred
        try:
            indexer.add_document("", "Invalid doc")
        except ValidationError:
            pass  # Expected

        # When I retry with valid input
        indexer.add_document("valid_doc.md", "Valid content")

        # Then the operation succeeds
        assert indexer.document_count == 1, "Valid document should be added"

    def test_system_usable_after_operation_error(self, indexer):
        """
        Scenario: System remains usable after operation error

        Given an operation error occurred (search before index)
        When I fix the issue and retry
        Then the operation succeeds
        Because transient errors shouldn't require restart.
        """
        # Given an operation error occurred
        indexer.add_document("doc.md", "Some content")
        try:
            indexer.search("content")  # Should fail - no index built
        except OperationError:
            pass  # Expected

        # When I fix the issue and retry
        indexer.build_index()
        results = indexer.search("content")

        # Then the operation succeeds
        assert len(results) > 0, "Search should work after building index"

    def test_multiple_errors_dont_accumulate_corruption(self, indexer):
        """
        Scenario: Multiple sequential errors don't corrupt state

        Given several errors occur in sequence
        When I eventually provide valid input
        Then the system works correctly
        Because error handling must not have side effects.
        """
        # Given several errors occur in sequence
        for _ in range(5):
            try:
                indexer.add_document("", "bad")
            except ValidationError:
                pass

            try:
                indexer.build_index()
            except OperationError:
                pass

        # When I eventually provide valid input
        indexer.add_document("good_doc.md", "Valid content here")
        indexer.build_index()
        results = indexer.search("content")

        # Then system works correctly
        assert len(results) == 1, "Should find the valid document"
        assert results[0]["doc_id"] == "good_doc.md"


# ============================================================================
# BOUNDARY CONDITION SCENARIOS
# ============================================================================

class TestBoundaryConditions:
    """
    Epic: Boundary Handling

    As a developer pushing system limits,
    I want predictable behavior at boundaries,
    So that I know what to expect in edge cases.
    """

    def test_document_id_at_max_length(self, indexer):
        """
        Scenario: Document ID at exactly maximum length is accepted

        Given a document ID of exactly 255 characters
        When I add the document
        Then it succeeds
        Because the limit should be inclusive.
        """
        # Given a document ID at exactly the limit
        max_length_id = "x" * 255

        # When I add the document
        indexer.add_document(max_length_id, "Content")

        # Then it succeeds
        assert indexer.document_count == 1, "Document with max-length ID should be accepted"

    def test_single_character_document_id(self, indexer):
        """
        Scenario: Single character document ID is valid

        Given a document ID of exactly one character
        When I add the document
        Then it succeeds
        Because minimal valid input should work.
        """
        # Given a single character ID
        single_char_id = "x"

        # When I add the document
        indexer.add_document(single_char_id, "Content")

        # Then it succeeds
        assert indexer.document_count == 1

    def test_single_character_content(self, indexer):
        """
        Scenario: Single character content is valid

        Given document content of exactly one character
        When I add the document
        Then it succeeds
        Because we shouldn't impose arbitrary minimums.
        """
        # Given single character content
        single_char_content = "x"

        # When I add the document
        indexer.add_document("doc.md", single_char_content)

        # Then it succeeds
        assert indexer.document_count == 1


# ============================================================================
# RUNNING INSTRUCTIONS
# ============================================================================

if __name__ == "__main__":
    # Allow running directly for demonstration
    pytest.main([__file__, "-v"])
