# Experiment: exp-20260107-200100-inverted-index

## Algorithm
**Name:** Inverted Index for Audit Comment Search
**Expected complexity:** O(n) build, O(1) term lookup, O(k) for k results
**Required operations:**
- `add(term: str, finding_id: str, position: int)` - Add term occurrence
- `search(term: str) -> List[Tuple[finding_id, positions]]` - Find findings containing term
- `search_phrase(terms: List[str]) -> List[finding_id]` - Find findings with consecutive terms
- `remove_finding(finding_id: str)` - Remove all terms for a finding
- `term_frequency(term: str, finding_id: str) -> int` - Count occurrences in finding

## Codebase Application

**Problem:** We have 29 audit findings across `docs/audits/misleading-comments/outbox/`. We need fast search to find patterns like "will be", "See:", or "document" across findings.

**Use Case:** When a new misleading comment is found, quickly find similar existing findings.

**Reference:** The codebase has an existing `InvertedIndex` at `cortical/cel/wisdom/semantic.py:150`.

## Hypothesis
**I expect:** The agent will correctly implement an inverted index for audit findings
**Because:** The codebase already has a reference implementation the agent can study.

## Task Prompt (Given to Agent)

```
Implement an Inverted Index for searching audit findings in the Cortical codebase.

Context: We have 29 audit findings from a misleading comments audit stored in
docs/audits/misleading-comments/outbox/result-*.md. Each finding has comment text
that we want to search for patterns like "will be", "See:", "FUTURE:".

Reference: Study cortical/cel/wisdom/semantic.py:150 for the existing InvertedIndex
pattern. Implement your own version optimized for audit finding search.

Requirements:
1. NO external libraries except typing
2. Must track positions for phrase search
3. Case-insensitive search
4. Must handle these operations:

from typing import Dict, List, Set, Tuple

class AuditInvertedIndex:
    def __init__(self):
        """Initialize empty index."""
        self._term_to_findings: Dict[str, Dict[str, List[int]]] = {}
        self._finding_terms: Dict[str, Set[str]] = {}

    def add(self, term: str, finding_id: str, position: int) -> None:
        """Add a term occurrence at a specific position in a finding."""
        pass

    def search(self, term: str) -> List[Tuple[str, List[int]]]:
        """
        Return list of (finding_id, [positions]) for findings containing term.
        Search is case-insensitive. Results sorted by finding_id.
        """
        pass

    def search_phrase(self, terms: List[str]) -> List[str]:
        """
        Return finding_ids where terms appear consecutively.
        Empty terms list returns empty list.
        """
        pass

    def remove_finding(self, finding_id: str) -> None:
        """Remove all entries for a finding. No-op if finding doesn't exist."""
        pass

    def term_frequency(self, term: str, finding_id: str) -> int:
        """Return number of times term appears in finding. 0 if not found."""
        pass

    def index_text(self, finding_id: str, text: str) -> None:
        """Tokenize text and add all terms with positions."""
        words = text.lower().split()
        for pos, word in enumerate(words):
            self.add(word.lower(), finding_id, pos)

Test cases that MUST pass:

from typing import Dict, List, Set, Tuple

# Test 1: Basic indexing with real audit patterns
idx = AuditInvertedIndex()
idx.index_text("F001", "FUTURE: When CDG index is implemented this will be handled")
idx.index_text("F002", "TODO: Add decision tracking")
idx.index_text("F003", "See: docs/design/cdg-transactional-indexing-design.md")

result = idx.search("future:")
assert len(result) == 1
assert result[0][0] == "F001"

result = idx.search("todo:")
assert len(result) == 1
assert result[0][0] == "F002"

# Test 2: Phrase search for "will be" pattern (common in misleading comments)
idx = AuditInvertedIndex()
idx.index_text("F001", "this will be handled")      # "will be" consecutive
idx.index_text("F002", "will not be done")          # "will" and "be" NOT consecutive
idx.index_text("F003", "it will be replaced")       # "will be" consecutive

result = sorted(idx.search_phrase(["will", "be"]))
assert result == ["F001", "F003"]
assert "F002" not in result  # "not" is between "will" and "be"

# Test 3: Term frequency
idx = AuditInvertedIndex()
idx.index_text("F001", "the the the quick brown fox")
assert idx.term_frequency("the", "F001") == 3
assert idx.term_frequency("quick", "F001") == 1
assert idx.term_frequency("missing", "F001") == 0
assert idx.term_frequency("the", "F999") == 0

# Test 4: Finding removal
idx = AuditInvertedIndex()
idx.index_text("F001", "hello world")
idx.index_text("F002", "hello there")
idx.remove_finding("F001")
result = idx.search("hello")
assert len(result) == 1
assert result[0][0] == "F002"

# Test 5: Edge cases - empty and non-existent
idx = AuditInvertedIndex()
assert idx.search("nonexistent") == []
assert idx.search_phrase([]) == []
assert idx.search_phrase(["never", "indexed"]) == []
idx.remove_finding("nonexistent")  # Should not crash

# Test 6: Case insensitivity
idx = AuditInvertedIndex()
idx.index_text("F001", "FUTURE: This WILL be done")
assert len(idx.search("future:")) == 1
assert len(idx.search("FUTURE:")) == 1
assert len(idx.search("Future:")) == 1

# Test 7: Results sorted by finding_id
idx = AuditInvertedIndex()
idx.index_text("F003", "test word")
idx.index_text("F001", "test case")
idx.index_text("F002", "test data")
result = idx.search("test")
finding_ids = [r[0] for r in result]
assert finding_ids == ["F001", "F002", "F003"]  # Sorted

Write the complete implementation with comments.
After implementing, demonstrate it works on this real finding from our audit:

real_finding = '''FUTURE: When CDG index is implemented, this will be handled at the
storage layer with WAL-based recovery. See:
docs/design/cdg-transactional-indexing-design.md'''

idx.index_text("REAL001", real_finding)
# Should find "will be" pattern
assert "REAL001" in idx.search_phrase(["will", "be"])
# Should find "See:" pattern
assert len(idx.search("see:")) == 1
```

## Success Criteria
- [ ] All 7 test cases pass
- [ ] O(1) term lookup (uses dict)
- [ ] Phrase search correctly checks consecutive positions
- [ ] Finding removal cleans up all entries
- [ ] Case-insensitive search
- [ ] Results sorted by finding_id
- [ ] Works on real audit finding example

## Failure Criteria
- [ ] Phrase search doesn't check consecutive positions
- [ ] Uses forbidden imports (defaultdict, collections)
- [ ] O(n) term lookup (linear scan)
- [ ] Crashes on edge cases
- [ ] Case sensitivity causes search misses

## Prediction
Before running: **PASS**
Confidence: **MEDIUM**
Reasoning: The existing `cortical/cel/wisdom/semantic.py` implementation provides a reference. Main challenge is phrase search position checking.

## Actual Result
Status: [NOT YET RUN]
Operations implemented: [X/6]
Tests passed: [X/7]
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

## Integration Plan

After successful implementation:
1. Add to `docs/audits/audit_indexer.py` as alternative search
2. Use to find similar findings when categorizing new comments
3. Index all comments in `cortical/` for pattern detection
