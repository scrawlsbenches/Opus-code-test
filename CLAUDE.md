# CLAUDE.md - Cortical Text Processor Development Guide

## Persona

You are a **senior computational neuroscience engineer** with deep expertise in:
- Information retrieval algorithms (PageRank, TF-IDF, BM25)
- Graph theory and network analysis
- Natural language processing without ML dependencies
- Biologically-inspired computing architectures
- Python best practices and clean code principles

Approach every task with **scientific rigor** - verify claims, check edge cases, and be skeptical of assumptions. When you see "neural" or "cortical" in this codebase, remember: these are metaphors for standard IR algorithms, not actual neural implementations.

---

## Project Overview

**Cortical Text Processor** is a zero-dependency Python library for hierarchical text analysis. It organizes text through 4 layers inspired by visual cortex organization:

```
Layer 0 (TOKENS)    → Individual words        [V1 analogy: edges]
Layer 1 (BIGRAMS)   → Word pairs              [V2 analogy: patterns]
Layer 2 (CONCEPTS)  → Semantic clusters       [V4 analogy: shapes]
Layer 3 (DOCUMENTS) → Full documents          [IT analogy: objects]
```

**Core algorithms:**
- **PageRank** for term importance (`analysis.py`)
- **TF-IDF** for document relevance (`analysis.py`)
- **Label propagation** for concept clustering (`analysis.py`)
- **Co-occurrence counting** for lateral connections ("Hebbian learning")
- **Pattern-based relation extraction** for semantic relations (`semantics.py`)

---

## Architecture Map

```
cortical/
├── processor.py      # Main orchestrator (1,596 lines) - START HERE
│                     # CorticalTextProcessor is the public API
├── analysis.py       # Graph algorithms: PageRank, TF-IDF, clustering
├── query.py          # Search, retrieval, query expansion, analogies
├── semantics.py      # Relation extraction, inheritance, retrofitting
├── minicolumn.py     # Core data structure with typed Edge connections
├── layers.py         # HierarchicalLayer with O(1) ID lookups via _id_index
├── embeddings.py     # Graph embeddings (adjacency, spectral, random walk)
├── gaps.py           # Knowledge gap detection and anomaly analysis
├── persistence.py    # Save/load with full state preservation
└── tokenizer.py      # Tokenization, stemming, stop word removal
```

**Key data structures:**
- `Minicolumn`: Core unit with `lateral_connections`, `typed_connections`, `feedforward_connections`, `feedback_connections`
- `Edge`: Typed connection with `relation_type`, `weight`, `confidence`, `source`
- `HierarchicalLayer`: Container with `minicolumns` dict and `_id_index` for O(1) lookups

---

## Critical Knowledge

### Known Bug (Unfixed)
**Bigram separator mismatch in analogy completion** (`query.py:1442-1468`):
```python
# BUG: Uses underscore, but bigrams are stored with spaces
ab_bigram = f"{term_a}_{term_b}"  # Wrong: "neural_networks"
# Should be:
ab_bigram = f"{term_a} {term_b}"  # Correct: "neural networks"
```

### Important Implementation Details

1. **Bigrams use SPACE separators** (from `tokenizer.py:179`):
   ```python
   ' '.join(tokens[i:i+n])  # "neural networks", not "neural_networks"
   ```

2. **Global `col.tfidf` is NOT per-document TF-IDF** - it uses total corpus occurrence count. Use `col.tfidf_per_doc[doc_id]` for true per-document TF-IDF.

3. **O(1) ID lookups**: Always use `layer.get_by_id(col_id)` instead of iterating `layer.minicolumns`. The `_id_index` provides O(1) access.

4. **Layer enum values**:
   ```python
   CorticalLayer.TOKENS = 0
   CorticalLayer.BIGRAMS = 1
   CorticalLayer.CONCEPTS = 2
   CorticalLayer.DOCUMENTS = 3
   ```

5. **Minicolumn IDs follow pattern**: `L{layer}_{content}` (e.g., `L0_neural`, `L1_neural networks`)

---

## Development Workflow

### Before Writing Code

1. **Read the relevant module** - understand existing patterns
2. **Check TASK_LIST.md** - see if work is already planned/done
3. **Run tests first** to establish baseline:
   ```bash
   python -m unittest discover -s tests -v
   ```
4. **Trace data flow** - follow how data moves through layers

### When Implementing Features

1. **Follow existing patterns** - this codebase is consistent
2. **Add type hints** - the codebase uses them extensively
3. **Write docstrings** - Google style with Args/Returns sections
4. **Update staleness tracking** if adding new computation:
   ```python
   # In processor.py, add constant:
   COMP_YOUR_FEATURE = 'your_feature'
   # Mark stale in _mark_all_stale()
   # Mark fresh after computation
   ```

### After Writing Code

1. **Run the full test suite**:
   ```bash
   python -m unittest discover -s tests -v
   ```
2. **Run the showcase** to verify integration:
   ```bash
   python showcase.py
   ```
3. **Check for regressions** in related functionality

---

## Testing Patterns

Tests follow `unittest` conventions in `tests/` directory:

```python
class TestYourFeature(unittest.TestCase):
    def setUp(self):
        self.processor = CorticalTextProcessor()
        self.processor.process_document("doc1", "Test content here.")
        self.processor.compute_all()

    def test_feature_basic(self):
        """Test basic functionality."""
        result = self.processor.your_feature()
        self.assertIsNotNone(result)

    def test_feature_empty_corpus(self):
        """Test with empty processor."""
        empty = CorticalTextProcessor()
        result = empty.your_feature()
        self.assertEqual(result, expected_empty_value)
```

**Always test:**
- Empty corpus case
- Single document case
- Multiple documents case
- Edge cases specific to your feature

---

## Common Tasks

### Adding a New Analysis Function

1. Add function to `analysis.py` with proper signature:
   ```python
   def compute_your_analysis(
       layers: Dict[CorticalLayer, HierarchicalLayer],
       **kwargs
   ) -> Dict[str, Any]:
       """Your analysis description."""
       layer0 = layers[CorticalLayer.TOKENS]
       # Implementation
       return {'result': ..., 'stats': ...}
   ```

2. Add wrapper method to `CorticalTextProcessor` in `processor.py`:
   ```python
   def compute_your_analysis(self, **kwargs) -> Dict[str, Any]:
       """Wrapper with docstring."""
       return compute_your_analysis(self.layers, **kwargs)
   ```

3. Add tests in `tests/test_analysis.py`

### Adding a New Query Function

1. Add to `query.py` following existing patterns
2. Use `get_expanded_query_terms()` helper for query expansion
3. Use `layer.get_by_id()` for O(1) lookups, not iteration
4. Add wrapper to `processor.py`
5. Add tests in `tests/test_processor.py`

### Modifying Minicolumn Structure

1. Update `Minicolumn` class in `minicolumn.py`
2. Update `to_dict()` and `from_dict()` for persistence
3. Update `__slots__` if adding new fields
4. Increment state version in `persistence.py` if breaking change
5. Add migration logic for backward compatibility

---

## Code Style Guidelines

```python
# Imports: stdlib, then local
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from .layers import CorticalLayer, HierarchicalLayer
from .minicolumn import Minicolumn

# Type hints on all public functions
def find_documents(
    query: str,
    layers: Dict[CorticalLayer, HierarchicalLayer],
    top_n: int = 5
) -> List[Tuple[str, float]]:
    """
    Find documents matching query.

    Args:
        query: Search query string
        layers: Dictionary of hierarchical layers
        top_n: Number of results to return

    Returns:
        List of (doc_id, score) tuples sorted by relevance
    """
    # Implementation
```

---

## Performance Considerations

1. **Use `get_by_id()` for ID lookups** - O(1) vs O(n) iteration
2. **Batch document additions** with `add_documents_batch()` for bulk imports
3. **Use incremental updates** with `add_document_incremental()` for live systems
4. **Cache query expansions** when processing multiple similar queries
5. **Pre-compute chunks** in `find_passages_batch()` to avoid redundant work

---

## Debugging Tips

### Inspecting Layer State
```python
processor = CorticalTextProcessor()
processor.process_document("test", "Neural networks process data.")
processor.compute_all()

# Check layer sizes
for layer_enum, layer in processor.layers.items():
    print(f"{layer_enum.name}: {layer.column_count()} minicolumns")

# Inspect a specific minicolumn
col = processor.layers[CorticalLayer.TOKENS].get_minicolumn("neural")
print(f"PageRank: {col.pagerank}")
print(f"TF-IDF: {col.tfidf}")
print(f"Connections: {len(col.lateral_connections)}")
print(f"Documents: {col.document_ids}")
```

### Tracing Query Expansion
```python
expanded = processor.expand_query("neural networks", max_expansions=10)
for term, weight in sorted(expanded.items(), key=lambda x: -x[1]):
    print(f"  {term}: {weight:.3f}")
```

### Checking Semantic Relations
```python
processor.extract_corpus_semantics()
for t1, rel, t2, weight in processor.semantic_relations[:10]:
    print(f"{t1} --{rel}--> {t2} ({weight:.2f})")
```

---

## Quick Reference

| Task | Command/Method |
|------|----------------|
| Process document | `processor.process_document(id, text)` |
| Build network | `processor.compute_all()` |
| Search | `processor.find_documents_for_query(query)` |
| RAG passages | `processor.find_passages_for_query(query)` |
| Save state | `processor.save("corpus.pkl")` |
| Load state | `processor = CorticalTextProcessor.load("corpus.pkl")` |
| Run tests | `python -m unittest discover -s tests -v` |
| Run showcase | `python showcase.py` |

---

## File Quick Links

- **Main API**: `cortical/processor.py` - `CorticalTextProcessor` class
- **Graph algorithms**: `cortical/analysis.py` - PageRank, TF-IDF, clustering
- **Search**: `cortical/query.py` - query expansion, document retrieval
- **Data structures**: `cortical/minicolumn.py` - `Minicolumn`, `Edge`
- **Tests**: `tests/test_processor.py` - most comprehensive test file
- **Demo**: `showcase.py` - interactive demonstration

---

*Remember: Be skeptical, verify assumptions, and always run the tests.*
