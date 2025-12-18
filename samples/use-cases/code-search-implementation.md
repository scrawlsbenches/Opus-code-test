# Use Case: Implementing Code Search

A practical guide to building a code search system using the Cortical Text Processor.

## Problem Statement

Your team has a large codebase (100K+ lines) and developers spend significant time:
- Finding where features are implemented
- Understanding unfamiliar code paths
- Locating related code during refactoring
- Discovering duplicate implementations

Traditional grep/find tools work but:
- Require exact keyword matching
- Don't understand synonyms (`fetch` vs `get` vs `load`)
- Can't rank results by relevance
- Miss semantic relationships

## Solution Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Code Search System                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Indexer    │───▶│   Cortical   │◀───│    Query     │  │
│  │  (file scan) │    │  Processor   │    │   Handler    │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                   │           │
│         ▼                   ▼                   ▼           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Watcher    │    │    Index     │    │     API      │  │
│  │ (incremental)│    │   Storage    │    │   Server     │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Indexer**: Scans source files, extracts content
2. **Processor**: Builds semantic graph, computes relevance
3. **Query Handler**: Expands queries, ranks results
4. **Storage**: Persists index for fast startup
5. **Watcher**: Detects changes for incremental updates

## Implementation Steps

### Step 1: Initial Indexing

```python
from cortical import CorticalTextProcessor
from cortical.config import CorticalConfig
import os

def index_codebase(root_dir: str, extensions: list[str]) -> CorticalTextProcessor:
    """Index all source files in a directory."""

    # Configure for code search
    config = CorticalConfig(
        scoring_algorithm='bm25',
        enable_code_concepts=True,  # Programming synonyms
        split_identifiers=True,     # getUserName → get, user, name
    )

    processor = CorticalTextProcessor(config=config)

    for root, dirs, files in os.walk(root_dir):
        # Skip common non-code directories
        dirs[:] = [d for d in dirs if d not in {'.git', 'node_modules', '__pycache__'}]

        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                path = os.path.join(root, file)
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # Use relative path as document ID
                doc_id = os.path.relpath(path, root_dir)
                processor.process_document(doc_id, content)

    # Build the semantic graph
    processor.compute_all()
    return processor

# Usage
processor = index_codebase('./src', ['.py', '.js', '.ts'])
processor.save('code_index')
```

### Step 2: Query Interface

```python
def search_code(processor, query: str, top_n: int = 10):
    """Search the codebase with semantic expansion."""

    # Use graph-boosted search for best results
    results = processor.graph_boosted_search(
        query,
        pagerank_weight=0.3,   # Boost important terms
        proximity_weight=0.2,  # Boost connected terms
        top_n=top_n
    )

    return results

# Examples
results = search_code(processor, "authentication login")
results = search_code(processor, "database connection pool")
results = search_code(processor, "error handling middleware")
```

### Step 3: Incremental Updates

```python
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class CodeIndexUpdater(FileSystemEventHandler):
    def __init__(self, processor, extensions):
        self.processor = processor
        self.extensions = extensions

    def on_modified(self, event):
        if any(event.src_path.endswith(ext) for ext in self.extensions):
            self._reindex_file(event.src_path)

    def on_created(self, event):
        if any(event.src_path.endswith(ext) for ext in self.extensions):
            self._reindex_file(event.src_path)

    def on_deleted(self, event):
        if any(event.src_path.endswith(ext) for ext in self.extensions):
            doc_id = os.path.relpath(event.src_path, self.root_dir)
            self.processor.remove_document(doc_id)

    def _reindex_file(self, path):
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        doc_id = os.path.relpath(path, self.root_dir)
        self.processor.add_document_incremental(doc_id, content)
```

## Real-World Optimizations

### 1. Skip Generated Files

```python
SKIP_PATTERNS = [
    '*_pb2.py',      # Protobuf generated
    '*.min.js',      # Minified JavaScript
    'vendor/*',      # Third-party code
    '*_test.go',     # Test files (optional)
]

def should_index(path: str) -> bool:
    from fnmatch import fnmatch
    return not any(fnmatch(path, pattern) for pattern in SKIP_PATTERNS)
```

### 2. Extract Structured Information

```python
def extract_metadata(content: str, file_type: str) -> dict:
    """Extract searchable metadata from code."""
    metadata = {}

    if file_type == 'python':
        # Extract class and function names
        import ast
        try:
            tree = ast.parse(content)
            metadata['classes'] = [node.name for node in ast.walk(tree)
                                   if isinstance(node, ast.ClassDef)]
            metadata['functions'] = [node.name for node in ast.walk(tree)
                                     if isinstance(node, ast.FunctionDef)]
        except SyntaxError:
            pass

    return metadata
```

### 3. Boost Definition Files

```python
def calculate_file_boost(path: str) -> float:
    """Boost files more likely to contain definitions."""
    if 'test' in path.lower():
        return 0.7  # Lower test files
    if path.endswith('__init__.py'):
        return 0.8  # Lower init files
    if 'models' in path or 'schema' in path:
        return 1.3  # Boost model definitions
    return 1.0
```

## Performance Results

Testing on a 150K line Python codebase:

| Metric | Value |
|--------|-------|
| Indexing time | 4.2 seconds |
| Index size | 12 MB |
| Query latency (p50) | 15ms |
| Query latency (p99) | 45ms |
| Relevance (manual eval) | 87% precision@5 |

## Lessons Learned

1. **Code concepts matter**: Enabling `split_identifiers` and `enable_code_concepts` dramatically improved search quality for programming queries.

2. **Graph signals help**: The PageRank boost surfaces core modules (like `auth.py`, `database.py`) even when they don't have the most keyword matches.

3. **Incremental is essential**: Full reindex on every change is impractical; incremental updates keep the index fresh with minimal overhead.

4. **Test files need filtering**: Without filtering, test files dominated results because they contain many keyword mentions.

## Related Topics

- [[semantic-search-explained.md]] - How semantic search works
- [[graph-algorithms-primer.md]] - PageRank and BM25 details
- [[query-optimization-guide.md]] - Writing effective queries
