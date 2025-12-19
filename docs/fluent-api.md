# FluentProcessor API Reference

A chainable, builder-pattern interface for the Cortical Text Processor.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Initialization Methods](#initialization-methods)
4. [Builder Methods](#builder-methods)
5. [Analysis Methods](#analysis-methods)
6. [Query Methods](#query-methods)
7. [Persistence Methods](#persistence-methods)
8. [Chaining Patterns](#chaining-patterns)
9. [Comparison with CorticalTextProcessor](#comparison-with-corticaltextprocessor)
10. [API Reference](#api-reference)

---

## Overview

**FluentProcessor** is a chainable API wrapper around `CorticalTextProcessor` that provides a fluent, builder-pattern interface for constructing and querying text processors.

### Why Use FluentProcessor?

**Traditional API:**
```python
processor = CorticalTextProcessor()
processor.process_document("doc1", "Neural networks process information")
processor.process_document("doc2", "Deep learning uses neural architectures")
processor.compute_all()
results = processor.find_documents_for_query("neural processing")
```

**Fluent API:**
```python
results = (FluentProcessor()
    .add_document("doc1", "Neural networks process information")
    .add_document("doc2", "Deep learning uses neural architectures")
    .build()
    .search("neural processing"))
```

### Key Benefits

- **Method chaining** - Express complex workflows in a single expression
- **Readable pipelines** - Operations flow naturally from left to right
- **Concise code** - Reduce boilerplate and intermediate variables
- **Type safety** - Clear distinction between builder and terminal operations
- **Convenience methods** - Simplified initialization from files and directories

---

## Quick Start

### Simple Search Pipeline

```python
from cortical import FluentProcessor

# Build and search in one expression
results = (FluentProcessor()
    .add_document("doc1", "Neural networks learn patterns from data")
    .add_document("doc2", "Machine learning algorithms improve with experience")
    .build()
    .search("neural learning", top_n=5))

for doc_id, score in results:
    print(f"{doc_id}: {score:.2f}")
```

### From Files

```python
# Load documents from files
processor = (FluentProcessor
    .from_files(["doc1.txt", "doc2.txt", "doc3.txt"])
    .build(verbose=True))

# Query the processor
results = processor.search("your query here")
```

### From Directory

```python
# Load all markdown files from a directory
processor = (FluentProcessor
    .from_directory("./docs", pattern="*.md", recursive=True)
    .build()
    .save("docs_corpus.pkl"))
```

### Load and Query

```python
# Load a saved processor and query
results = (FluentProcessor
    .load("corpus.pkl")
    .search("neural networks", top_n=10))
```

---

## Initialization Methods

### `__init__(tokenizer=None, config=None)`

Create a new FluentProcessor instance.

**Parameters:**
- `tokenizer` (Optional[Tokenizer]): Custom tokenizer instance
- `config` (Optional[CorticalConfig]): Configuration object

**Returns:** FluentProcessor instance

**Example:**
```python
from cortical import FluentProcessor, Tokenizer, CorticalConfig

# Basic initialization
processor = FluentProcessor()

# With custom tokenizer
tokenizer = Tokenizer(split_identifiers=True)
processor = FluentProcessor(tokenizer=tokenizer)

# With configuration
config = CorticalConfig(min_token_length=2, scoring_algorithm='bm25')
processor = FluentProcessor(config=config)
```

---

### `from_files(file_paths, tokenizer=None, config=None)` (classmethod)

Create a processor from a list of files.

**Parameters:**
- `file_paths` (List[Union[str, Path]]): List of file paths to load
- `tokenizer` (Optional[Tokenizer]): Custom tokenizer
- `config` (Optional[CorticalConfig]): Configuration object

**Returns:** FluentProcessor with documents added

**Behavior:**
- Uses filename (without extension) as document ID
- Adds `source` metadata with full file path
- Raises `FileNotFoundError` if file doesn't exist
- Raises `ValueError` if path is not a file

**Example:**
```python
# Load multiple files
processor = FluentProcessor.from_files([
    "paper1.txt",
    "paper2.txt",
    "paper3.txt"
])

# With custom configuration
from cortical import CorticalConfig

config = CorticalConfig(min_token_length=3)
processor = FluentProcessor.from_files(
    ["doc1.txt", "doc2.txt"],
    config=config
)
```

---

### `from_directory(directory, pattern="*.txt", recursive=False, tokenizer=None, config=None)` (classmethod)

Create a processor from all files in a directory.

**Parameters:**
- `directory` (Union[str, Path]): Directory path to scan
- `pattern` (str): Glob pattern for file matching (default: `"*.txt"`)
- `recursive` (bool): Whether to search subdirectories (default: False)
- `tokenizer` (Optional[Tokenizer]): Custom tokenizer
- `config` (Optional[CorticalConfig]): Configuration object

**Returns:** FluentProcessor with documents added from directory

**Example:**
```python
# Load all text files in a directory
processor = FluentProcessor.from_directory("./documents")

# Load markdown files recursively
processor = FluentProcessor.from_directory(
    "./docs",
    pattern="*.md",
    recursive=True
)

# Load Python files with custom tokenizer
tokenizer = Tokenizer(split_identifiers=True)
processor = FluentProcessor.from_directory(
    "./src",
    pattern="*.py",
    recursive=True,
    tokenizer=tokenizer
)
```

---

### `from_existing(processor)` (classmethod)

Create a FluentProcessor from an existing CorticalTextProcessor.

**Parameters:**
- `processor` (CorticalTextProcessor): Existing processor instance

**Returns:** FluentProcessor wrapping the existing processor

**Example:**
```python
from cortical import CorticalTextProcessor, FluentProcessor

# Existing processor
standard_proc = CorticalTextProcessor()
standard_proc.process_document("doc1", "content")
standard_proc.compute_all()

# Wrap in fluent interface
fluent = FluentProcessor.from_existing(standard_proc)
results = fluent.search("query")
```

---

### `load(path)` (classmethod)

Load a processor from a saved file.

**Parameters:**
- `path` (Union[str, Path]): Path to saved processor file

**Returns:** FluentProcessor loaded from file (already built)

**Example:**
```python
# Load and query immediately
results = FluentProcessor.load("corpus.pkl").search("query")

# Load and continue building
processor = (FluentProcessor.load("corpus.pkl")
    .add_document("new_doc", "new content")
    .build())
```

---

## Builder Methods

Builder methods are chainable and return `self` to enable method chaining.

### `add_document(doc_id, content, metadata=None)`

Add a single document to the processor.

**Parameters:**
- `doc_id` (str): Unique document identifier
- `content` (str): Document text content
- `metadata` (Optional[Dict[str, Any]]): Optional metadata dictionary

**Returns:** Self (for chaining)

**Example:**
```python
processor = (FluentProcessor()
    .add_document("doc1", "First document")
    .add_document("doc2", "Second document", {"author": "Alice"})
    .add_document("doc3", "Third document"))
```

---

### `add_documents(documents)`

Add multiple documents at once.

**Parameters:**
- `documents` (Union[Dict, List]): Can be:
  - Dict mapping `doc_id -> content`
  - List of `(doc_id, content)` tuples
  - List of `(doc_id, content, metadata)` tuples

**Returns:** Self (for chaining)

**Example:**
```python
# From dictionary
processor = FluentProcessor().add_documents({
    "doc1": "First content",
    "doc2": "Second content",
    "doc3": "Third content"
})

# From list of tuples
processor = FluentProcessor().add_documents([
    ("doc1", "First content"),
    ("doc2", "Second content", {"source": "web"}),
    ("doc3", "Third content")
])
```

---

### `with_config(config)`

Set configuration for the processor.

**Parameters:**
- `config` (CorticalConfig): Configuration object

**Returns:** Self (for chaining)

**Example:**
```python
from cortical import CorticalConfig

config = CorticalConfig(
    min_token_length=3,
    scoring_algorithm='bm25',
    bm25_k1=1.5
)

processor = (FluentProcessor()
    .with_config(config)
    .add_document("doc1", "content")
    .build())
```

---

### `with_tokenizer(tokenizer)`

Set a custom tokenizer.

**Parameters:**
- `tokenizer` (Tokenizer): Custom Tokenizer instance

**Returns:** Self (for chaining)

**Example:**
```python
from cortical import Tokenizer

# Code-aware tokenizer
tokenizer = Tokenizer(split_identifiers=True)

processor = (FluentProcessor()
    .with_tokenizer(tokenizer)
    .add_document("code1", "getUserData() fetchRecords()")
    .build())
```

---

## Analysis Methods

### `build(verbose=True, build_concepts=True, **kwargs)`

Build the processor by computing all analysis phases. This is a chainable method that computes PageRank, TF-IDF, connections, and optionally concept clusters.

**Parameters:**
- `verbose` (bool): Print debug messages (default: True)
- `build_concepts` (bool): Build concept clusters (Layer 2) (default: True)
- `pagerank_method` (str): `'standard'`, `'semantic'`, or `'hierarchical'`
- `connection_strategy` (str): `'document_overlap'`, `'semantic'`, `'embedding'`, or `'hybrid'`
- `cluster_strictness` (float): Clustering aggressiveness (0.0-1.0, default: 1.0)
- `bridge_weight` (float): Inter-document token bridging (0.0-1.0, default: 0.0)
- `show_progress` (bool): Show progress bar (default: False)

**Returns:** Self (for chaining)

**Analysis Phases:**
1. TF-IDF computation
2. PageRank importance
3. Bigram lateral connections
4. Document-to-document connections
5. Concept clustering (if enabled)

**Example:**
```python
# Standard build
processor = (FluentProcessor()
    .add_documents({"doc1": "content1", "doc2": "content2"})
    .build())

# Custom build configuration
processor = (FluentProcessor()
    .add_documents(documents)
    .build(
        verbose=False,
        build_concepts=True,
        pagerank_method='semantic',
        cluster_strictness=0.8,
        show_progress=True
    ))

# Minimal build (no concepts)
processor = (FluentProcessor()
    .add_documents(documents)
    .build(build_concepts=False))
```

---

## Query Methods

Query methods are **terminal operations** that return results, not `self`. After a query method, you cannot continue chaining builder methods.

### `search(query, top_n=5, use_expansion=True, use_semantic=True)`

Search for documents matching the query.

**Parameters:**
- `query` (str): Search query string
- `top_n` (int): Number of results to return (default: 5)
- `use_expansion` (bool): Use query expansion (default: True)
- `use_semantic` (bool): Use semantic expansion (default: True)

**Returns:** `List[Tuple[str, float]]` - List of (doc_id, score) tuples sorted by relevance

**Example:**
```python
# Standard search
results = (FluentProcessor()
    .add_documents(documents)
    .build()
    .search("neural networks", top_n=10))

for doc_id, score in results:
    print(f"{doc_id}: {score:.2f}")

# Without expansion (exact match only)
results = processor.search("neural networks", use_expansion=False)
```

---

### `fast_search(query, top_n=5, candidate_multiplier=3, use_code_concepts=True)`

Fast document search with pre-filtering. Approximately 2-3x faster than standard search on large corpora.

**Parameters:**
- `query` (str): Search query string
- `top_n` (int): Number of results to return (default: 5)
- `candidate_multiplier` (int): Candidate pool size multiplier (default: 3)
- `use_code_concepts` (bool): Use code concept expansion (default: True)

**Returns:** `List[Tuple[str, float]]` - List of (doc_id, score) tuples

**Tuning:**
- `candidate_multiplier=1`: Aggressive (may miss relevant docs)
- `candidate_multiplier=3`: Balanced (recommended)
- `candidate_multiplier=5`: Conservative (slower but higher recall)

**Example:**
```python
# Fast search for large corpus
results = (FluentProcessor
    .load("large_corpus.pkl")
    .fast_search("authentication", top_n=10))

# Code search with fast mode
results = processor.fast_search(
    "getUserData",
    top_n=5,
    candidate_multiplier=4,
    use_code_concepts=True
)
```

---

### `search_passages(query, top_n=5, chunk_size=None, overlap=None, use_expansion=True)`

Search for passage chunks matching the query. Perfect for RAG systems.

**Parameters:**
- `query` (str): Search query string
- `top_n` (int): Number of passage results (default: 5)
- `chunk_size` (Optional[int]): Token count per chunk (default from config)
- `overlap` (Optional[int]): Token overlap between chunks (default from config)
- `use_expansion` (bool): Use query expansion (default: True)

**Returns:** `List[Tuple[str, str, int, int, float]]` - List of (doc_id, passage_text, start_pos, end_pos, score) tuples

**Example:**
```python
# RAG passage retrieval
passages = (FluentProcessor
    .from_directory("./docs", pattern="*.md", recursive=True)
    .build()
    .search_passages("how do neural networks learn", top_n=5))

for doc_id, text, start, end, score in passages:
    print(f"[{doc_id}:{start}-{end}] Score: {score:.2f}")
    print(f"  {text[:100]}...")

# Custom chunk size
passages = processor.search_passages(
    "query",
    chunk_size=512,
    overlap=128
)
```

---

### `expand(query, max_expansions=None, use_variants=True, use_code_concepts=False)`

Expand a query with related terms. Useful for understanding query interpretation.

**Parameters:**
- `query` (str): Query string to expand
- `max_expansions` (Optional[int]): Maximum number of expansion terms
- `use_variants` (bool): Include term variants (default: True)
- `use_code_concepts` (bool): Use code concept synonyms (default: False)

**Returns:** `Dict[str, float]` - Dictionary mapping terms to expansion weights

**Example:**
```python
# View query expansion
expansions = (FluentProcessor()
    .add_documents(documents)
    .build()
    .expand("neural networks", max_expansions=10))

for term, weight in sorted(expansions.items(), key=lambda x: -x[1]):
    print(f"  {term}: {weight:.3f}")
# Output:
#   neural: 1.000
#   networks: 1.000
#   learning: 0.450
#   data: 0.380
#   patterns: 0.320
```

---

## Persistence Methods

### `save(path)`

Save the processor to disk. This is a **chainable method** that returns `self`.

**Parameters:**
- `path` (Union[str, Path]): File path to save to

**Returns:** Self (for chaining)

**Example:**
```python
# Build and save in one chain
processor = (FluentProcessor()
    .add_documents(documents)
    .build()
    .save("corpus.pkl"))

# Continue after saving
results = (FluentProcessor()
    .add_documents(documents)
    .build()
    .save("corpus.pkl")
    .search("query"))  # Can still query after save
```

---

## Chaining Patterns

### Pattern 1: Simple Pipeline

```python
# Load, build, query in one expression
results = (FluentProcessor
    .from_files(["doc1.txt", "doc2.txt"])
    .build()
    .search("neural networks"))
```

### Pattern 2: Configuration Pipeline

```python
from cortical import CorticalConfig, Tokenizer

# Configure, load, build, save
processor = (FluentProcessor()
    .with_config(CorticalConfig(scoring_algorithm='bm25'))
    .with_tokenizer(Tokenizer(split_identifiers=True))
    .add_documents(documents)
    .build(verbose=False)
    .save("configured_corpus.pkl"))
```

### Pattern 3: Directory to Saved Corpus

```python
# Process entire directory and save
(FluentProcessor
    .from_directory("./docs", pattern="*.md", recursive=True)
    .build(show_progress=True)
    .save("docs_corpus.pkl"))
```

### Pattern 4: Load-Query-Save Pattern

```python
# Load, add documents, rebuild, save
updated = (FluentProcessor
    .load("corpus.pkl")
    .add_document("new_doc", "new content")
    .build()
    .save("corpus_updated.pkl"))
```

### Pattern 5: Multi-Query Pipeline

```python
processor = (FluentProcessor
    .from_directory("./papers")
    .build())

# Multiple queries on same processor
results1 = processor.search("neural networks")
results2 = processor.search("deep learning")
results3 = processor.expand("machine learning")
```

### Pattern 6: RAG Pipeline

```python
# Build RAG backend from knowledge base
def create_rag_backend(kb_dir: str) -> FluentProcessor:
    return (FluentProcessor
        .from_directory(kb_dir, pattern="*.txt", recursive=True)
        .build(verbose=False)
        .save("rag_corpus.pkl"))

# Use in RAG system
processor = create_rag_backend("./knowledge_base")
passages = processor.search_passages(user_query, top_n=5)
```

### Pattern 7: Conditional Building

```python
from cortical import CorticalConfig

def build_corpus(files, code_mode=False):
    processor = FluentProcessor()

    if code_mode:
        processor = processor.with_tokenizer(
            Tokenizer(split_identifiers=True)
        )

    return (processor
        .add_documents(files)
        .build(build_concepts=not code_mode)
        .save("corpus.pkl"))
```

---

## Comparison with CorticalTextProcessor

### When to Use FluentProcessor

- **Prototyping** - Quick experimentation with different configurations
- **Pipelines** - Express document processing as a single chain
- **Scripts** - Concise code for one-off tasks
- **Simple workflows** - Load, build, query patterns
- **File-based loading** - Need `from_files()` or `from_directory()` convenience

### When to Use CorticalTextProcessor

- **Fine-grained control** - Need access to individual computation phases
- **Advanced features** - Using specialized methods not exposed in fluent API
- **Performance optimization** - Managing staleness and selective recomputation
- **Library integration** - Building other abstractions on top of the processor
- **Complex workflows** - Multi-step processes with branching logic

### Feature Comparison

| Feature | FluentProcessor | CorticalTextProcessor |
|---------|-----------------|----------------------|
| Method chaining | Yes | No |
| File loading | Built-in (`from_files`, `from_directory`) | Manual |
| Builder pattern | Yes | No |
| All search methods | Yes (via delegation) | Yes |
| Staleness control | Automatic | Manual |
| Selective recomputation | Limited | Full control |
| Introspection | Via `.processor` property | Direct |
| Learning curve | Easier | Steeper |

### Converting Between APIs

**Fluent to Standard:**
```python
fluent = FluentProcessor().add_documents(docs).build()
standard = fluent.processor  # Access underlying processor
```

**Standard to Fluent:**
```python
standard = CorticalTextProcessor()
standard.process_document("doc1", "content")
fluent = FluentProcessor.from_existing(standard)
```

### Hybrid Usage

```python
# Use fluent for building
fluent = (FluentProcessor()
    .add_documents(documents)
    .build())

# Use standard API for advanced operations
processor = fluent.processor
processor.compute_graph_embeddings()
processor.extract_corpus_semantics()
embeddings = processor.get_embeddings()

# Back to fluent for querying
results = fluent.search("query")
```

---

## API Reference

### Properties

#### `processor`
Access the underlying `CorticalTextProcessor` instance.

**Type:** `CorticalTextProcessor`

**Example:**
```python
fluent = FluentProcessor().add_documents(docs).build()
raw_processor = fluent.processor
raw_processor.compute_importance()
```

#### `is_built`
Check if the processor has been built (i.e., `build()` was called).

**Type:** `bool`

**Example:**
```python
processor = FluentProcessor().add_documents(docs)
print(processor.is_built)  # False

processor.build()
print(processor.is_built)  # True
```

---

### Method Summary

**Initialization (Class Methods):**
- `FluentProcessor(tokenizer=None, config=None)` - Create new instance
- `from_files(file_paths, ...)` - Create from file list
- `from_directory(directory, pattern="*.txt", ...)` - Create from directory
- `from_existing(processor)` - Wrap existing processor
- `load(path)` - Load from saved file

**Builder Methods (Chainable):**
- `add_document(doc_id, content, metadata=None)` - Add single document
- `add_documents(documents)` - Add multiple documents
- `with_config(config)` - Set configuration
- `with_tokenizer(tokenizer)` - Set tokenizer
- `build(**kwargs)` - Compute all analysis phases
- `save(path)` - Save to disk

**Query Methods (Terminal):**
- `search(query, top_n=5, ...)` - Standard document search
- `fast_search(query, top_n=5, ...)` - Fast document search
- `search_passages(query, top_n=5, ...)` - Passage retrieval
- `expand(query, ...)` - Query expansion

---

## Advanced Examples

### Example 1: Code Search System

```python
from cortical import FluentProcessor, Tokenizer

def build_code_search(repo_dir: str):
    """Build a code search system from a repository."""
    tokenizer = Tokenizer(split_identifiers=True)

    return (FluentProcessor()
        .with_tokenizer(tokenizer)
        .from_directory(repo_dir, pattern="*.py", recursive=True)
        .build(verbose=False, build_concepts=False)
        .save("code_search.pkl"))

# Use it
processor = build_code_search("./my_project")
results = processor.fast_search("authentication handler", top_n=10)
```

### Example 2: Multi-Format Document Loader

```python
from pathlib import Path

def load_mixed_formats(directory: str):
    """Load documents from multiple file formats."""
    processor = FluentProcessor()

    # Load text files
    for ext in ['*.txt', '*.md', '*.rst']:
        try:
            processor.from_directory(directory, pattern=ext, recursive=True)
        except ValueError:
            pass  # No files found for this extension

    return processor.build()

processor = load_mixed_formats("./docs")
```

### Example 3: Incremental Corpus Builder

```python
def incremental_builder(base_corpus: str, new_docs: dict):
    """Add documents to existing corpus incrementally."""
    return (FluentProcessor
        .load(base_corpus)
        .add_documents(new_docs)
        .build()  # Rebuild with new documents
        .save(base_corpus))  # Overwrite

# Use it
updated = incremental_builder(
    "corpus.pkl",
    {"doc_new1": "content1", "doc_new2": "content2"}
)
```

### Example 4: Query Debugging Tool

```python
def debug_query(corpus_path: str, query: str):
    """Debug why a query returns certain results."""
    processor = FluentProcessor.load(corpus_path)

    # Show expansion
    print("Query Expansion:")
    expansions = processor.expand(query, max_expansions=10)
    for term, weight in sorted(expansions.items(), key=lambda x: -x[1]):
        print(f"  {term}: {weight:.3f}")

    # Show results
    print("\nSearch Results:")
    results = processor.search(query, top_n=5)
    for doc_id, score in results:
        print(f"  {doc_id}: {score:.2f}")

    return processor

# Use it
debug_query("corpus.pkl", "neural networks")
```

### Example 5: Batch Processing Pipeline

```python
from typing import List

def batch_process_files(file_list: List[str], output: str):
    """Process files in batches and save."""
    batch_size = 100

    for i in range(0, len(file_list), batch_size):
        batch = file_list[i:i+batch_size]
        batch_output = f"{output}_batch{i//batch_size}.pkl"

        (FluentProcessor()
            .from_files(batch)
            .build(show_progress=True)
            .save(batch_output))

        print(f"Saved batch to {batch_output}")

# Process 1000 files in batches of 100
import glob
files = glob.glob("./documents/**/*.txt", recursive=True)
batch_process_files(files, "corpus")
```

---

## Best Practices

### 1. Always Call `build()` Before Querying

```python
# WRONG - will fail
results = (FluentProcessor()
    .add_documents(docs)
    .search("query"))  # Error: not built!

# CORRECT
results = (FluentProcessor()
    .add_documents(docs)
    .build()
    .search("query"))
```

### 2. Use `from_directory()` for File Collections

```python
# Instead of manually listing files
processor = FluentProcessor().add_documents({
    "doc1": open("doc1.txt").read(),
    "doc2": open("doc2.txt").read(),
    # ... tedious
})

# Use from_directory
processor = FluentProcessor.from_directory("./docs")
```

### 3. Save After Building Large Corpora

```python
# Build once, save, reuse
(FluentProcessor()
    .from_directory("./large_corpus", recursive=True)
    .build(show_progress=True)
    .save("large_corpus.pkl"))

# Load instantly later
processor = FluentProcessor.load("large_corpus.pkl")
```

### 4. Use Fast Search for Large Corpora

```python
# Standard search is fine for <1000 documents
if num_docs < 1000:
    results = processor.search("query")
else:
    results = processor.fast_search("query")
```

### 5. Access Raw Processor for Advanced Features

```python
fluent = FluentProcessor().add_documents(docs).build()

# Use raw processor for advanced operations
raw = fluent.processor
raw.extract_corpus_semantics()
relations = raw.semantic_relations
```

---

## Troubleshooting

### Error: "Processor not built"

**Cause:** Calling a query method before `build()`

**Solution:** Always call `.build()` before query methods

```python
# Fix this
processor = FluentProcessor().add_documents(docs)
results = processor.search("query")  # Error!

# To this
processor = FluentProcessor().add_documents(docs).build()
results = processor.search("query")  # Works!
```

### Error: "Cannot chain after terminal operation"

**Cause:** Trying to chain builder methods after a query method

**Solution:** Query methods are terminal - they return results, not `self`

```python
# This won't work
results = (FluentProcessor()
    .add_documents(docs)
    .build()
    .search("query")
    .save("corpus.pkl"))  # Error: search() returns List, not FluentProcessor

# Do this instead
processor = (FluentProcessor()
    .add_documents(docs)
    .build()
    .save("corpus.pkl"))
results = processor.search("query")
```

### Slow Performance

**Solution:** Use `fast_search()` or build a search index via the raw processor

```python
fluent = FluentProcessor().from_directory("./large").build()

# Fast search
results = fluent.fast_search("query", candidate_multiplier=4)

# Or use raw processor for index
index = fluent.processor.build_search_index()
results = fluent.processor.search_with_index("query", index)
```

---

## See Also

- [Quickstart Guide](quickstart.md) - Get started with the Cortical Text Processor
- [Cookbook](cookbook.md) - Common patterns and recipes
- [Query Guide](query-guide.md) - Advanced query techniques
- [CLAUDE.md](../CLAUDE.md) - Full developer documentation
- [Architecture Guide](architecture.md) - How the processor works

---

*FluentProcessor provides a concise, chainable interface for common workflows. For advanced control, access the underlying processor via the `.processor` property.*
