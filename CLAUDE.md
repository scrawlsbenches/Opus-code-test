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

## AI Agent Onboarding

**New to this codebase?** Follow these steps to get oriented quickly:

### Step 1: Generate AI Metadata (if missing)

```bash
# Check if metadata exists
ls cortical/*.ai_meta

# If not present, generate it (~1s)
python scripts/generate_ai_metadata.py
```

### Step 2: Read Module Metadata First

Instead of reading entire source files, start with `.ai_meta` files:

```bash
# Get structured overview of any module
cat cortical/processor.py.ai_meta
```

**What metadata provides:**
- Module docstring and purpose
- Function signatures with `see_also` cross-references
- Class structures with inheritance
- Logical section groupings (Persistence, Query, Analysis, etc.)
- Complexity hints for expensive operations

### Step 3: Use the Full Toolchain

```bash
# Index codebase + generate metadata (recommended startup command)
python scripts/index_codebase.py --incremental && python scripts/generate_ai_metadata.py --incremental

# Then search semantically
python scripts/search_codebase.py "your query here"
```

### AI Navigation Tips

1. **Read `.ai_meta` before source code** - Get the map before exploring the territory
2. **Follow `see_also` references** - Functions are cross-linked to related functions
3. **Check `complexity_hints`** - Know which operations are expensive before calling them
4. **Use semantic search** - The codebase is indexed for meaning-based retrieval
5. **Trust the sections** - Functions are grouped by purpose in the metadata

### Example Workflow

```bash
# I need to understand how search works
cat cortical/query.py.ai_meta | head -100    # Get overview
python scripts/search_codebase.py "expand query"  # Find specific code
# Then read specific line ranges as needed
```

---

## Architecture Map

```
cortical/
├── processor.py      # Main orchestrator (2,301 lines) - START HERE
│                     # CorticalTextProcessor is the public API
├── query.py          # Search, retrieval, query expansion (2,719 lines)
├── analysis.py       # Graph algorithms: PageRank, TF-IDF, clustering (1,123 lines)
├── semantics.py      # Relation extraction, inheritance, retrofitting (915 lines)
├── persistence.py    # Save/load with full state preservation (606 lines)
├── chunk_index.py    # Git-friendly chunk-based storage (574 lines)
├── tokenizer.py      # Tokenization, stemming, stop word removal (398 lines)
├── minicolumn.py     # Core data structure with typed Edge connections (357 lines)
├── config.py         # CorticalConfig dataclass with validation (352 lines)
├── fingerprint.py    # Semantic fingerprinting and similarity (315 lines)
├── layers.py         # HierarchicalLayer with O(1) ID lookups via _id_index (294 lines)
├── code_concepts.py  # Programming concept synonyms for code search (249 lines)
├── gaps.py           # Knowledge gap detection and anomaly analysis (245 lines)
└── embeddings.py     # Graph embeddings (adjacency, spectral, random walk) (209 lines)
```

**Total:** ~10,700 lines of core library code

### Module Purpose Quick Reference

| If you need to... | Look in... |
|-------------------|------------|
| Add/modify public API | `processor.py` - wrapper methods call other modules |
| Implement search/retrieval | `query.py` - all search functions |
| Add graph algorithms | `analysis.py` - PageRank, TF-IDF, clustering |
| Add semantic relations | `semantics.py` - pattern extraction, retrofitting |
| Modify data structures | `minicolumn.py` - Minicolumn, Edge classes |
| Change layer behavior | `layers.py` - HierarchicalLayer class |
| Adjust tokenization | `tokenizer.py` - stemming, stop words, ngrams |
| Change configuration | `config.py` - CorticalConfig dataclass |
| Modify persistence | `persistence.py` - save/load, export formats |
| Add code search features | `code_concepts.py` - programming synonyms |
| Modify embeddings | `embeddings.py` - graph embedding methods |
| Change gap detection | `gaps.py` - knowledge gap analysis |
| Add fingerprinting | `fingerprint.py` - semantic fingerprints |
| Modify chunk storage | `chunk_index.py` - git-friendly indexing |

**Key data structures:**
- `Minicolumn`: Core unit with `lateral_connections`, `typed_connections`, `feedforward_connections`, `feedback_connections`
- `Edge`: Typed connection with `relation_type`, `weight`, `confidence`, `source`
- `HierarchicalLayer`: Container with `minicolumns` dict and `_id_index` for O(1) lookups

### Test File Locations

| When testing... | Add tests to... |
|-----------------|-----------------|
| Processor methods | `tests/test_processor.py` (most comprehensive) |
| Query functions | `tests/test_query.py` |
| Analysis algorithms | `tests/test_analysis.py` |
| Semantic extraction | `tests/test_semantics.py` |
| Persistence/save/load | `tests/test_persistence.py` |
| Tokenization | `tests/test_tokenizer.py` |
| Configuration | `tests/test_config.py` |
| Layers | `tests/test_layers.py` |
| Embeddings | `tests/test_embeddings.py` |
| Gap detection | `tests/test_gaps.py` |
| Fingerprinting | `tests/test_fingerprint.py` |
| Code concepts | `tests/test_code_concepts.py` |
| Chunk indexing | `tests/test_chunk_indexing.py` |
| Incremental updates | `tests/test_incremental_indexing.py` |
| Intent queries | `tests/test_intent_query.py` |

---

## Critical Knowledge

### Fixed Bugs (2025-12-10)
Historical note: Bigram separator mismatch bugs have been **fixed**. Bigrams now correctly use space separators throughout the codebase (see `tokenizer.py:extract_ngrams` and `analysis.py:compute_bigram_connections`).

### Important Implementation Details

1. **Bigrams use SPACE separators** (from `tokenizer.py:319-332`):
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

### Common Mistakes to Avoid

**❌ DON'T iterate to find by ID:**
```python
# WRONG - O(n) linear scan
for col in layer.minicolumns.values():
    if col.id == target_id:
        return col

# CORRECT - O(1) lookup
col = layer.get_by_id(target_id)
```

**❌ DON'T use underscores in bigrams:**
```python
# WRONG - bigrams use spaces
bigram = f"{term1}_{term2}"

# CORRECT
bigram = f"{term1} {term2}"
```

**❌ DON'T confuse global TF-IDF with per-document TF-IDF:**
```python
# WRONG - global TF-IDF (uses total corpus occurrence)
score = col.tfidf

# CORRECT - per-document TF-IDF
score = col.tfidf_per_doc.get(doc_id, 0.0)
```

**❌ DON'T assume compute_all() is always needed:**
```python
# WRONG - overkill for incremental updates
processor.add_document_incremental(doc_id, text)
processor.compute_all()  # Recomputes EVERYTHING

# CORRECT - let incremental handle it
processor.add_document_incremental(doc_id, text)
# TF-IDF and connections updated automatically
```

**❌ DON'T forget to check staleness before relying on computed values:**
```python
# WRONG - may be using stale data
if processor.is_stale(processor.COMP_PAGERANK):
    # PageRank values may be outdated!
    pass

# CORRECT - ensure freshness
if processor.is_stale(processor.COMP_PAGERANK):
    processor.compute_importance()
```

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
4. **Dog-food the feature** - test with real usage (see [dogfooding-checklist.md](docs/dogfooding-checklist.md))
5. **Document all findings** - add issues to TASK_LIST.md (see [code-of-ethics.md](docs/code-of-ethics.md))
6. **Verify completion** - use [definition-of-done.md](docs/definition-of-done.md) checklist

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
6. **Use `fast_find_documents()`** for ~2-3x faster search on large corpora
7. **Pre-build index** with `build_search_index()` for fastest repeated queries

---

## Code Search Capabilities

### Code-Aware Tokenization
```python
# Enable identifier splitting for code search
tokenizer = Tokenizer(split_identifiers=True)
tokens = tokenizer.tokenize("getUserCredentials")
# ['getusercredentials', 'get', 'user', 'credentials']
```

### Programming Concept Expansion
```python
# Expand queries with programming synonyms (get/fetch/load)
results = processor.expand_query("fetch data", use_code_concepts=True)
# Or use the convenience method
results = processor.expand_query_for_code("fetch data")
```

### Intent-Based Search
```python
# Parse natural language queries
parsed = processor.parse_intent_query("where do we handle authentication?")
# {'intent': 'location', 'action': 'handle', 'subject': 'authentication', ...}

# Search with intent understanding
results = processor.search_by_intent("how do we validate input?")
```

### Semantic Fingerprinting
```python
# Compare code similarity
fp1 = processor.get_fingerprint(code_block_1)
fp2 = processor.get_fingerprint(code_block_2)
comparison = processor.compare_fingerprints(fp1, fp2)
explanation = processor.explain_similarity(fp1, fp2)
```

### Fast Search
```python
# Fast document search (~2-3x faster)
results = processor.fast_find_documents("authentication")

# Pre-built index for fastest search
index = processor.build_search_index()
results = processor.search_with_index("query", index)
```

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
| Fast search | `processor.fast_find_documents(query)` |
| Code search | `processor.expand_query_for_code(query)` |
| Intent search | `processor.search_by_intent("where do we...")` |
| RAG passages | `processor.find_passages_for_query(query)` |
| Fingerprint | `processor.get_fingerprint(text)` |
| Compare | `processor.compare_fingerprints(fp1, fp2)` |
| Save state | `processor.save("corpus.pkl")` |
| Load state | `processor = CorticalTextProcessor.load("corpus.pkl")` |
| Run tests | `python -m unittest discover -s tests -v` |
| Run showcase | `python showcase.py` |

---

## Dog-Fooding: Search the Codebase

The Cortical Text Processor can index and search its own codebase, providing semantic search capabilities during development.

### Quick Start

```bash
# Index the codebase (creates corpus_dev.pkl, ~2s)
python scripts/index_codebase.py

# Incremental update (only changed files)
python scripts/index_codebase.py --incremental

# Search for code
python scripts/search_codebase.py "PageRank algorithm"
python scripts/search_codebase.py "bigram separator" --verbose
python scripts/search_codebase.py --interactive
```

### Claude Skills

Three skills are available in `.claude/skills/`:

1. **codebase-search**: Search the indexed codebase for code patterns and implementations
2. **corpus-indexer**: Re-index the codebase after making changes
3. **ai-metadata**: View pre-generated module metadata for rapid understanding

### Indexer Options

| Option | Description |
|--------|-------------|
| `--incremental`, `-i` | Only re-index changed files (fastest) |
| `--status`, `-s` | Show what would change without indexing |
| `--force`, `-f` | Force full rebuild |
| `--log FILE` | Write detailed log to file |
| `--verbose`, `-v` | Show per-file progress |
| `--use-chunks` | Use git-compatible chunk-based storage |
| `--compact` | Compact old chunk files (with `--use-chunks`) |

### Search Options

| Option | Description |
|--------|-------------|
| `--top N` | Number of results (default: 5) |
| `--verbose` | Show full passage text |
| `--expand` | Show query expansion terms |
| `--interactive` | Interactive search mode |

### Interactive Mode Commands

| Command | Description |
|---------|-------------|
| `/expand <query>` | Show query expansion |
| `/concepts` | List concept clusters |
| `/stats` | Show corpus statistics |
| `/quit` | Exit interactive mode |

### Example Queries

```bash
# Find how PageRank is implemented
python scripts/search_codebase.py "compute pagerank damping factor"

# Find test patterns
python scripts/search_codebase.py "unittest setUp processor"

# Explore query expansion code
python scripts/search_codebase.py "expand query semantic lateral"
```

### Git-Compatible Chunk-Based Indexing

For team collaboration, use chunk-based indexing which stores document changes as git-friendly JSON files:

```bash
# Index with chunk storage (creates corpus_chunks/*.json)
python scripts/index_codebase.py --incremental --use-chunks

# Check chunk status
python scripts/index_codebase.py --status --use-chunks

# Compact old chunks (reduces git history size)
python scripts/index_codebase.py --compact --before 2025-12-01
```

**Architecture:**
```
corpus_chunks/                        # Tracked in git (append-only)
├── 2025-12-10_21-53-45_a1b2.json    # Session 1 changes
├── 2025-12-10_22-15-30_c3d4.json    # Session 2 changes
└── 2025-12-10_23-00-00_e5f6.json    # Session 3 changes

corpus_dev.pkl                        # NOT tracked (local cache)
corpus_dev.pkl.hash                   # NOT tracked (cache validation)
```

**Benefits:**
- No merge conflicts (unique timestamp+session filenames)
- Shared indexed state across team/branches
- Fast startup when cache is valid
- Git-friendly (small JSON, append-only)
- Periodic compaction like `git gc`

### Chunk Compaction

Over time, chunk files accumulate. Use compaction to consolidate them, similar to `git gc`:

**When to compact:**
- After many indexing sessions (10+ chunk files)
- When you see size warnings during indexing
- Before merging branches with different chunk histories
- To clean up deleted/modified document history

**Compaction commands:**
```bash
# Compact all chunks into a single consolidated file
python scripts/index_codebase.py --compact --use-chunks

# Compact only chunks created before a specific date
python scripts/index_codebase.py --compact --before 2025-12-01 --use-chunks

# Check chunk status before compacting
python scripts/index_codebase.py --status --use-chunks
```

**How compaction works:**
1. Reads all chunk files (sorted by timestamp)
2. Replays operations in order (later timestamps override)
3. Creates a single compacted chunk with final state
4. Removes old chunk files
5. Preserves cache if still valid

**Recommended frequency:**
- Weekly for active development
- Monthly for maintenance repositories
- Before major releases

---

## File Quick Links

- **Main API**: `cortical/processor.py` - `CorticalTextProcessor` class
- **Graph algorithms**: `cortical/analysis.py` - PageRank, TF-IDF, clustering
- **Search**: `cortical/query.py` - query expansion, document retrieval
- **Data structures**: `cortical/minicolumn.py` - `Minicolumn`, `Edge`
- **Configuration**: `cortical/config.py` - `CorticalConfig` dataclass
- **Tests**: `tests/test_processor.py` - most comprehensive test file
- **Demo**: `showcase.py` - interactive demonstration

**Process Documentation:**
- **Getting Started**: `docs/quickstart.md` - 5-minute tutorial for newcomers
- **Contributing**: `CONTRIBUTING.md` - how to contribute (fork, test, PR workflow)
- **Ethics**: `docs/code-of-ethics.md` - documentation, testing, and completion standards
- **Dog-fooding**: `docs/dogfooding-checklist.md` - checklist for testing with real usage
- **Definition of Done**: `docs/definition-of-done.md` - when is a task truly complete?
- **Task Archive**: `TASK_ARCHIVE.md` - completed tasks history

---

*Remember: Be skeptical, verify assumptions, and always run the tests.*
