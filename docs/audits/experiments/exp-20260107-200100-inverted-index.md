# Experiment: exp-20260107-200100-inverted-index

## Algorithm
**Name:** Inverted Index
**Expected complexity:** O(n) build, O(1) term lookup, O(k) for k results
**Required operations:**
- `add(term: str, doc_id: str, position: int)` - Add term occurrence
- `search(term: str) -> List[Tuple[doc_id, positions]]` - Find documents containing term
- `search_phrase(terms: List[str]) -> List[doc_id]` - Find documents with consecutive terms
- `remove_document(doc_id: str)` - Remove all terms for a document
- `term_frequency(term: str, doc_id: str) -> int` - Count occurrences in document

## Hypothesis
**I expect:** The agent will correctly implement a basic inverted index
**Because:** This is a well-documented data structure with clear semantics. The main challenge is phrase search (position tracking).

## Task Prompt (Given to Agent)

```
Implement an Inverted Index from scratch in Python.

An inverted index maps terms to the documents (and positions) where they appear.
This is the foundation of search engines.

Requirements:
1. NO external libraries except typing (no collections.defaultdict, no nltk, etc.)
2. Must track positions for phrase search
3. Must handle these operations:

class InvertedIndex:
    def add(self, term: str, doc_id: str, position: int) -> None:
        """Add a term occurrence at a specific position in a document."""
        pass

    def search(self, term: str) -> List[Tuple[str, List[int]]]:
        """Return list of (doc_id, [positions]) for documents containing term."""
        pass

    def search_phrase(self, terms: List[str]) -> List[str]:
        """Return doc_ids where terms appear consecutively."""
        pass

    def remove_document(self, doc_id: str) -> None:
        """Remove all entries for a document."""
        pass

    def term_frequency(self, term: str, doc_id: str) -> int:
        """Return number of times term appears in document."""
        pass

Test cases that MUST pass:

# Test 1: Basic indexing and search
idx = InvertedIndex()
idx.add("hello", "doc1", 0)
idx.add("world", "doc1", 1)
idx.add("hello", "doc2", 0)
assert idx.search("hello") == [("doc1", [0]), ("doc2", [0])] or similar
assert idx.search("world") == [("doc1", [1])]
assert idx.search("missing") == []

# Test 2: Phrase search
idx = InvertedIndex()
idx.add("the", "doc1", 0)
idx.add("quick", "doc1", 1)
idx.add("brown", "doc1", 2)
idx.add("fox", "doc1", 3)
idx.add("quick", "doc2", 0)
idx.add("fox", "doc2", 1)  # Not consecutive with "quick brown"
assert "doc1" in idx.search_phrase(["quick", "brown"])
assert "doc2" not in idx.search_phrase(["quick", "brown"])

# Test 3: Term frequency
idx = InvertedIndex()
idx.add("the", "doc1", 0)
idx.add("the", "doc1", 5)
idx.add("the", "doc1", 10)
assert idx.term_frequency("the", "doc1") == 3
assert idx.term_frequency("the", "doc2") == 0

# Test 4: Document removal
idx = InvertedIndex()
idx.add("hello", "doc1", 0)
idx.add("hello", "doc2", 0)
idx.remove_document("doc1")
result = idx.search("hello")
assert len(result) == 1
assert result[0][0] == "doc2"

Write the complete implementation with all methods working.
Include brief comments explaining your approach.
```

## Success Criteria
- [ ] All 5 operations implemented
- [ ] All 4 test cases pass
- [ ] O(1) term lookup (uses dict)
- [ ] Positions tracked correctly for phrase search
- [ ] Document removal cleans up all term entries

## Failure Criteria
- [ ] Missing operations
- [ ] Phrase search doesn't check consecutive positions
- [ ] Uses defaultdict or other forbidden imports
- [ ] O(n) term lookup (linear scan)
- [ ] Memory leak on document removal

## Prediction
Before running: **PASS**
Confidence: **HIGH**
Reasoning: Inverted index is a standard structure. Position tracking for phrase search is the hardest part but well-documented.

## Actual Result
Status: [NOT YET RUN]
Operations implemented: [X/5]
Tests passed: [X/4]
Notes:

## Agent Output
```python
[Agent's code will be pasted here after running]
```

## Test Results
```
[Test execution output will be pasted here]
```

## Analysis
**Discrepancy:**
**Root cause:**
**Learning:**

## Recommendations

