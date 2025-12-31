"""
Behavioral tests for developers using typed result containers.

Epic: Type-Safe Query Results

As a developer using modern Python tooling,
I want strongly-typed result containers with IDE autocomplete,
So that I can write safer code with better tooling support.

Based on: examples/examples_results_usage.py
"""

import pytest
from cortical import (
    CorticalTextProcessor,
    DocumentMatch,
    PassageMatch,
    QueryResult,
    convert_document_matches,
    convert_passage_matches
)


class TestDeveloperUsesTypedResults:
    """
    Epic: Type-Safe Query Results

    As a developer building maintainable applications,
    I want typed result dataclasses instead of tuples,
    So that I get IDE autocomplete and type checking.
    """

    def test_scenario_developer_creates_document_matches_with_type_safety(self):
        """
        Scenario: Creating strongly-typed document matches

        Given search results as tuples
        When I convert to DocumentMatch objects
        Then I get type-safe objects with attributes
        And IDE autocomplete works correctly
        Because typed objects prevent errors and improve developer experience.
        """
        # GIVEN search results as tuples
        raw_results = [
            ("neural_networks.md", 0.95),
            ("deep_learning.py", 0.87)
        ]

        # WHEN I convert to DocumentMatch objects
        matches = convert_document_matches(raw_results)

        # THEN I get type-safe objects with attributes
        assert len(matches) == 2, "Should convert all results"
        assert isinstance(matches[0], DocumentMatch), "Should be DocumentMatch instance"

        # AND IDE autocomplete works correctly
        first_match = matches[0]
        assert hasattr(first_match, 'doc_id'), "Should have doc_id attribute"
        assert hasattr(first_match, 'score'), "Should have score attribute"
        assert first_match.doc_id == "neural_networks.md", "Should preserve doc_id"
        assert first_match.score == 0.95, "Should preserve score"

    def test_scenario_developer_creates_passage_matches_with_context(self):
        """
        Scenario: Working with passage-level results

        Given passage search results
        When I create PassageMatch objects
        Then I get text, location, and score information
        And can easily cite passages
        Because RAG systems need rich passage metadata.
        """
        # GIVEN passage search results
        # WHEN I create PassageMatch objects
        passage = PassageMatch(
            doc_id="cortical/processor.py",
            text="def compute_pagerank(self):\n    \"\"\"Compute PageRank scores.\"\"\"",
            score=0.92,
            start=1500,
            end=1580
        )

        # THEN I get text, location, and score information
        assert passage.doc_id == "cortical/processor.py", "Should store document ID"
        assert passage.text.startswith("def compute_pagerank"), "Should store passage text"
        assert passage.score == 0.92, "Should store relevance score"
        assert passage.start == 1500, "Should store start position"
        assert passage.end == 1580, "Should store end position"

        # AND can easily cite passages
        location = passage.location
        assert "1500:1580" in location, "Location property should format position"
        assert passage.length == 80, "Length property should calculate correctly"

    def test_scenario_developer_integrates_with_cortical_processor(self):
        """
        Scenario: Using typed results with processor queries

        Given a processor with indexed documents
        When I search and convert results to typed objects
        Then I work with DocumentMatch objects
        And code is more readable and maintainable
        Because typed results improve code quality.
        """
        # GIVEN a processor with indexed documents
        processor = CorticalTextProcessor()
        processor.process_document(
            "neural_networks.md",
            "Neural networks are computational models inspired by biological neurons."
        )
        processor.process_document(
            "deep_learning.py",
            "Deep learning uses neural networks with multiple layers."
        )
        processor.compute_all(verbose=False)

        # WHEN I search and convert results to typed objects
        raw_results = processor.find_documents_for_query("neural networks", top_n=3)
        matches = convert_document_matches(raw_results)

        # THEN I work with DocumentMatch objects
        assert len(matches) > 0, "Should get results"
        assert isinstance(matches[0], DocumentMatch), "Should be typed object"

        # AND code is more readable and maintainable
        for match in matches:
            # Type-safe attribute access
            doc_name = match.doc_id
            relevance = match.score
            assert isinstance(doc_name, str), "doc_id should be string"
            assert isinstance(relevance, float), "score should be float"

    def test_scenario_developer_wraps_results_with_query_metadata(self):
        """
        Scenario: QueryResult wrapper with rich metadata

        Given search results with context
        When I create a QueryResult wrapper
        Then I get query metadata and statistics
        And can access expansion terms and timing
        Because developers need complete query context.
        """
        # GIVEN search results with context
        matches = [
            DocumentMatch("neural_networks.md", 0.95),
            DocumentMatch("deep_learning.py", 0.87),
            DocumentMatch("ai_overview.md", 0.72)
        ]

        # WHEN I create a QueryResult wrapper
        result = QueryResult(
            query="neural networks",
            matches=matches,
            expansion_terms={
                "neural": 1.0,
                "network": 0.95,
                "neuron": 0.7
            },
            timing_ms=15.3
        )

        # THEN I get query metadata and statistics
        assert result.query == "neural networks", "Should store original query"
        assert result.match_count == 3, "Should calculate match count"
        assert result.timing_ms == 15.3, "Should store timing information"

        # AND can access expansion terms and timing
        assert "neural" in result.expansion_terms, "Should store expansion terms"
        assert result.average_score > 0, "Should calculate average score"
        assert result.top_match.doc_id == "neural_networks.md", "Should identify top match"

    def test_scenario_developer_converts_batch_results_with_metadata(self):
        """
        Scenario: Batch conversion with document metadata

        Given raw results and metadata dictionary
        When I convert with metadata mapping
        Then each match includes its metadata
        And I can filter by metadata properties
        Because developers need to propagate document metadata.
        """
        # GIVEN raw results and metadata dictionary
        raw_results = [
            ("neural_networks.md", 0.95),
            ("deep_learning.py", 0.87),
            ("ai_overview.md", 0.72)
        ]

        metadata = {
            "neural_networks.md": {"type": "documentation", "size": 2048},
            "deep_learning.py": {"type": "code", "language": "python"},
            "ai_overview.md": {"type": "documentation", "size": 1024}
        }

        # WHEN I convert with metadata mapping
        matches = convert_document_matches(raw_results, metadata)

        # THEN each match includes its metadata
        assert len(matches) == 3, "Should convert all matches"
        assert matches[0].metadata is not None, "Should include metadata"
        assert matches[0].metadata["type"] == "documentation", "Should preserve metadata values"

        # AND I can filter by metadata properties
        code_matches = [m for m in matches if m.metadata and m.metadata.get("type") == "code"]
        assert len(code_matches) == 1, "Should be able to filter by metadata"
        assert code_matches[0].doc_id == "deep_learning.py", "Should find code file"

    def test_scenario_developer_benefits_from_immutability(self):
        """
        Scenario: Immutable results prevent accidental modification

        Given a DocumentMatch object
        When I try to modify its attributes
        Then modification is prevented
        And results remain consistent
        Because immutable results prevent bugs.
        """
        # GIVEN a DocumentMatch object
        match = DocumentMatch("test.txt", 0.8)

        # WHEN I try to modify its attributes
        # THEN modification is prevented
        with pytest.raises(AttributeError):
            match.score = 0.9

        # AND results remain consistent
        assert match.score == 0.8, "Score should remain unchanged"

    def test_scenario_developer_serializes_results_to_dict(self):
        """
        Scenario: Converting results to dictionaries for JSON

        Given typed result objects
        When I convert to dictionary format
        Then I get JSON-serializable data
        And can send results over APIs
        Because developers need to serialize results.
        """
        # GIVEN typed result objects
        match = DocumentMatch("neural_networks.md", 0.95, metadata={"type": "doc"})

        # WHEN I convert to dictionary format
        match_dict = match.to_dict()

        # THEN I get JSON-serializable data
        assert isinstance(match_dict, dict), "Should be dictionary"
        assert match_dict["doc_id"] == "neural_networks.md", "Should preserve doc_id"
        assert match_dict["score"] == 0.95, "Should preserve score"
        assert match_dict["metadata"]["type"] == "doc", "Should preserve metadata"

        # AND can send results over APIs
        # Dictionary can be JSON-serialized

    def test_scenario_developer_deserializes_from_dict(self):
        """
        Scenario: Reconstructing typed objects from dictionaries

        Given a dictionary representation
        When I deserialize to typed object
        Then I get back a proper typed instance
        And round-trip conversion works
        Because developers need bidirectional conversion.
        """
        # GIVEN a dictionary representation
        result_dict = {
            "query": "neural networks",
            "matches": [
                {"doc_id": "doc1.txt", "score": 0.95, "metadata": None},
                {"doc_id": "doc2.txt", "score": 0.87, "metadata": None}
            ],
            "expansion_terms": {"neural": 1.0, "network": 0.95},
            "timing_ms": 15.3
        }

        # WHEN I deserialize to typed object
        restored = QueryResult.from_dict(result_dict)

        # THEN I get back a proper typed instance
        assert isinstance(restored, QueryResult), "Should be QueryResult instance"
        assert restored.query == "neural networks", "Should restore query"
        assert restored.match_count == 2, "Should restore matches"

        # AND round-trip conversion works
        assert restored.timing_ms == 15.3, "Should restore all fields"

    def test_scenario_developer_converts_tuples_bidirectionally(self):
        """
        Scenario: Backward compatibility with tuple format

        Given legacy code using tuples
        When I convert between tuples and typed objects
        Then conversion works in both directions
        And I can gradually migrate codebase
        Because developers need backward compatibility.
        """
        # GIVEN legacy code using tuples
        original_tuple = ("neural_networks.md", 0.95)

        # WHEN I convert between tuples and typed objects
        match = DocumentMatch.from_tuple(*original_tuple)
        converted_back = match.to_tuple()

        # THEN conversion works in both directions
        assert match.doc_id == original_tuple[0], "Should convert from tuple"
        assert match.score == original_tuple[1], "Should preserve values"
        assert converted_back == original_tuple, "Should convert back to tuple"

        # AND I can gradually migrate codebase
        # Old code can continue using tuples while new code uses typed objects

    def test_scenario_developer_accesses_passage_properties(self):
        """
        Scenario: Using computed properties on passages

        Given a PassageMatch object
        When I access computed properties
        Then location and length are calculated correctly
        And I can easily format citations
        Because passage metadata is complex.
        """
        # GIVEN a PassageMatch object
        passage = PassageMatch(
            doc_id="document.py",
            text="Sample passage text here",
            score=0.88,
            start=100,
            end=124
        )

        # WHEN I access computed properties
        location = passage.location
        length = passage.length

        # THEN location and length are calculated correctly
        assert location == "document.py:100:124", "Location should format correctly"
        assert length == 24, "Length should be end - start"

        # AND I can easily format citations
        citation = f"[{passage.location}]"
        assert citation == "[document.py:100:124]", "Should create proper citation"

    def test_scenario_developer_works_with_passage_conversion(self):
        """
        Scenario: Converting passage tuples to typed objects

        Given raw passage results from processor
        When I convert to PassageMatch objects
        Then all passage information is preserved
        And I can work with rich passage data
        Because passage results are more complex than document results.
        """
        # GIVEN raw passage results from processor
        raw_passages = [
            ("Sample passage text", "doc1.txt", 0, 50, 0.92),
            ("Another passage", "doc2.txt", 100, 115, 0.85)
        ]

        # WHEN I convert to PassageMatch objects
        passages = convert_passage_matches(raw_passages)

        # THEN all passage information is preserved
        assert len(passages) == 2, "Should convert all passages"
        assert isinstance(passages[0], PassageMatch), "Should be PassageMatch instance"

        first = passages[0]
        assert first.text == "Sample passage text", "Should preserve text"
        assert first.doc_id == "doc1.txt", "Should preserve doc_id"
        assert first.start == 0, "Should preserve start position"
        assert first.end == 50, "Should preserve end position"
        assert first.score == 0.92, "Should preserve score"

        # AND I can work with rich passage data
        assert first.length == 50, "Should calculate length"
        assert "doc1.txt" in first.location, "Should format location"

    def test_scenario_developer_filters_results_with_type_safety(self):
        """
        Scenario: Type-safe filtering and manipulation

        Given a list of typed matches
        When I filter and process results
        Then type checking helps prevent errors
        And code is self-documenting
        Because typed objects make code intent clear.
        """
        # GIVEN a list of typed matches
        matches = [
            DocumentMatch("high_score.txt", 0.95, metadata={"category": "A"}),
            DocumentMatch("medium_score.txt", 0.75, metadata={"category": "B"}),
            DocumentMatch("low_score.txt", 0.55, metadata={"category": "A"})
        ]

        # WHEN I filter and process results
        # Filter by score threshold
        high_quality = [m for m in matches if m.score >= 0.8]
        assert len(high_quality) == 1, "Should filter by score"

        # Filter by metadata
        category_a = [m for m in matches if m.metadata and m.metadata.get("category") == "A"]
        assert len(category_a) == 2, "Should filter by metadata"

        # THEN type checking helps prevent errors
        # AND code is self-documenting
        # Type hints make it clear what we're working with
        for match in matches:
            # IDE knows match.score is float, match.doc_id is str
            assert isinstance(match.score, float), "Should have typed score"
            assert isinstance(match.doc_id, str), "Should have typed doc_id"
