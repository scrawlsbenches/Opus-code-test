#!/usr/bin/env python
"""
Example Usage: Result Dataclasses
==================================

Demonstrates the strongly-typed result containers for query operations
that provide IDE autocomplete and type checking support.

Task #185: Create result dataclasses for the Cortical Text Processor.
"""

from cortical import (
    CorticalTextProcessor,
    DocumentMatch,
    PassageMatch,
    QueryResult,
    convert_document_matches,
    convert_passage_matches
)


def example_basic_document_match():
    """Example 1: Basic DocumentMatch usage."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic DocumentMatch Usage")
    print("="*60)

    # Create matches manually
    match1 = DocumentMatch("neural_networks.md", 0.95)
    match2 = DocumentMatch("deep_learning.py", 0.87, metadata={"type": "code"})

    print(f"\nMatch 1: {match1.doc_id} - Score: {match1.score:.2f}")
    print(f"Match 2: {match2.doc_id} - Score: {match2.score:.2f}")
    print(f"         Metadata: {match2.metadata}")

    # Convert to/from tuple (for compatibility with legacy code)
    tuple_form = match1.to_tuple()
    print(f"\nTuple form: {tuple_form}")
    restored = DocumentMatch.from_tuple(*tuple_form)
    print(f"Restored: {restored}")

    # Convert to/from dict (for JSON serialization)
    dict_form = match1.to_dict()
    print(f"\nDict form: {dict_form}")


def example_basic_passage_match():
    """Example 2: Basic PassageMatch usage."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Basic PassageMatch Usage")
    print("="*60)

    # Create passage match
    passage = PassageMatch(
        doc_id="cortical/processor.py",
        text="def compute_pagerank(self):\n    \"\"\"Compute PageRank importance scores.\"\"\"",
        score=0.92,
        start=1500,
        end=1580
    )

    print(f"\nDocument: {passage.doc_id}")
    print(f"Location: {passage.location}")
    print(f"Length: {passage.length} characters")
    print(f"Score: {passage.score:.2f}")
    print(f"Text:\n{passage.text}")

    # Useful properties
    print(f"\nCitation: [{passage.location}]")


def example_with_processor():
    """Example 3: Using dataclasses with CorticalTextProcessor."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Integration with CorticalTextProcessor")
    print("="*60)

    # Set up processor with sample documents
    processor = CorticalTextProcessor()
    processor.process_document(
        "neural_networks.md",
        "Neural networks are computational models inspired by biological neurons. "
        "They consist of interconnected nodes that process information."
    )
    processor.process_document(
        "deep_learning.py",
        "class DeepNetwork:\n    def __init__(self):\n        self.layers = []\n    "
        "def forward(self, input):\n        # Process through layers\n        pass"
    )
    processor.process_document(
        "ai_overview.md",
        "Artificial intelligence encompasses machine learning, neural networks, "
        "and deep learning approaches to solving complex problems."
    )
    processor.compute_all()

    # Perform document search
    print("\n--- Document Search ---")
    raw_results = processor.find_documents_for_query("neural networks", top_n=3)
    matches = convert_document_matches(raw_results)

    for i, match in enumerate(matches, 1):
        print(f"{i}. {match.doc_id}: {match.score:.3f}")

    # Perform passage search
    print("\n--- Passage Search ---")
    raw_passages = processor.find_passages_for_query("neural", top_n=2)
    passages = convert_passage_matches(raw_passages)

    for i, passage in enumerate(passages, 1):
        text_preview = passage.text[:60] + "..." if len(passage.text) > 60 else passage.text
        print(f"{i}. [{passage.location}] {text_preview}")
        print(f"   Score: {passage.score:.3f}")


def example_query_result_wrapper():
    """Example 4: Using QueryResult wrapper with metadata."""
    print("\n" + "="*60)
    print("EXAMPLE 4: QueryResult Wrapper with Metadata")
    print("="*60)

    # Simulate search results
    matches = [
        DocumentMatch("neural_networks.md", 0.95),
        DocumentMatch("deep_learning.py", 0.87),
        DocumentMatch("ai_overview.md", 0.72)
    ]

    # Wrap in QueryResult with metadata
    result = QueryResult(
        query="neural networks",
        matches=matches,
        expansion_terms={
            "neural": 1.0,
            "network": 0.95,
            "neuron": 0.7,
            "artificial": 0.5
        },
        timing_ms=15.3
    )

    print(f"\nQuery: '{result.query}'")
    print(f"Match count: {result.match_count}")
    print(f"Average score: {result.average_score:.3f}")
    print(f"Query time: {result.timing_ms}ms")

    print(f"\nTop match: {result.top_match.doc_id} ({result.top_match.score:.3f})")

    print("\nExpansion terms:")
    for term, weight in sorted(result.expansion_terms.items(), key=lambda x: -x[1]):
        print(f"  {term}: {weight:.2f}")

    # Serialization
    print("\n--- Serialization Example ---")
    result_dict = result.to_dict()
    print(f"Serialized keys: {list(result_dict.keys())}")

    restored = QueryResult.from_dict(result_dict)
    print(f"Restored query: '{restored.query}'")
    print(f"Restored matches: {restored.match_count}")


def example_batch_conversion():
    """Example 5: Batch conversion with metadata."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Batch Conversion with Metadata")
    print("="*60)

    # Simulate raw results from search
    raw_results = [
        ("neural_networks.md", 0.95),
        ("deep_learning.py", 0.87),
        ("ai_overview.md", 0.72)
    ]

    # Metadata for each document
    metadata = {
        "neural_networks.md": {"type": "documentation", "size": 2048},
        "deep_learning.py": {"type": "code", "language": "python"},
        "ai_overview.md": {"type": "documentation", "size": 1024}
    }

    # Convert with metadata
    matches = convert_document_matches(raw_results, metadata)

    print("\nConverted matches with metadata:")
    for match in matches:
        print(f"  {match.doc_id}: {match.score:.2f}")
        if match.metadata:
            print(f"    {match.metadata}")


def example_type_safety():
    """Example 6: Type safety and IDE support."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Type Safety and IDE Support")
    print("="*60)

    # Create a match
    match = DocumentMatch("test.txt", 0.8)

    # IDE autocomplete works because match has known attributes
    print(f"\nAttributes available with autocomplete:")
    print(f"  match.doc_id = {match.doc_id}")
    print(f"  match.score = {match.score}")
    print(f"  match.metadata = {match.metadata}")

    # Type checking catches errors at development time
    print("\nDataclasses are immutable (frozen):")
    try:
        match.score = 0.9  # This will raise an error
    except AttributeError as e:
        print(f"  ✓ Cannot modify: {e}")

    # PassageMatch has additional useful properties
    passage = PassageMatch("doc.py", "code here", 0.9, 100, 110)
    print(f"\nPassageMatch properties:")
    print(f"  passage.location = {passage.location}")
    print(f"  passage.length = {passage.length}")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("CORTICAL TEXT PROCESSOR - Result Dataclasses Examples")
    print("Task #185: Strongly-typed query result containers")
    print("="*60)

    example_basic_document_match()
    example_basic_passage_match()
    example_with_processor()
    example_query_result_wrapper()
    example_batch_conversion()
    example_type_safety()

    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
