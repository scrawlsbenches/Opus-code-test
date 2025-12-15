# Quickstart Guide

Get up and running with the Cortical Text Processor in 5 minutes.

## Installation

```bash
# Clone the repository
git clone https://github.com/scrawlsbenches/Opus-code-test.git
cd Opus-code-test

# Install the package
pip install -e .
```

That's it! No external dependencies required.

## Your First 10 Lines

```python
from cortical import CorticalTextProcessor

# Create a processor
processor = CorticalTextProcessor()

# Add some documents
processor.process_document("ai_intro", "Neural networks learn patterns from data using layers of neurons.")
processor.process_document("ml_basics", "Machine learning algorithms improve through experience with data.")
processor.process_document("cooking", "Sourdough bread requires flour, water, salt, and a starter culture.")

# Build the network (computes PageRank, TF-IDF, connections)
processor.compute_all()

# Search!
results = processor.find_documents_for_query("neural learning")
for doc_id, score in results:
    print(f"{doc_id}: {score:.2f}")
```

**Output:**
```
ai_intro: 3.47
ml_basics: 2.15
```

The processor found relevant documents and ranked them by semantic similarity.

## Simplified Facade Methods

For common use cases, these simplified methods provide sensible defaults:

### Quick Search (Just Doc IDs)

```python
# One-call search - returns just document IDs
docs = processor.quick_search("neural learning")
print(docs)  # ['ai_intro', 'ml_basics']
```

### RAG Retrieve (For LLMs)

```python
# Get passages ready for LLM context injection
passages = processor.rag_retrieve("how do neural networks work", top_n=3)
for p in passages:
    print(f"[{p['doc_id']}] {p['text'][:60]}... (score: {p['score']:.2f})")
```

Each passage dict contains: `text`, `doc_id`, `start`, `end`, `score`.

### Explore (With Query Expansion)

```python
# See how your query was expanded
result = processor.explore("machine learning")
print(f"Original: {result['original_terms']}")
print(f"Expanded to: {list(result['expansion'].keys())[:5]}")
print(f"Top result: {result['results'][0][0]}")
```

## Understanding the Results

The Cortical Text Processor builds a graph of your documents:

1. **Tokens** (Layer 0): Individual words like "neural", "learning", "data"
2. **Bigrams** (Layer 1): Word pairs like "neural networks", "machine learning"
3. **Concepts** (Layer 2): Clusters of related terms
4. **Documents** (Layer 3): Your full documents

When you search, the processor:
- Tokenizes your query
- Expands it with related terms (query expansion)
- Scores documents using TF-IDF and PageRank
- Returns ranked results

## Query Expansion

See how the processor expands your queries:

```python
expanded = processor.expand_query("neural networks")
for term, weight in sorted(expanded.items(), key=lambda x: -x[1])[:5]:
    print(f"  {term}: {weight:.2f}")
```

**Output:**
```
  neural: 1.00
  networks: 1.00
  learning: 0.45
  data: 0.38
  patterns: 0.32
```

The processor automatically finds related terms to improve search coverage.

## Passage Retrieval (for RAG)

Get specific text passages, not just document IDs:

```python
passages = processor.find_passages_for_query("neural patterns", top_n=3)
for text, doc_id, start, end, score in passages:
    print(f"[{doc_id}] {text[:50]}... (score: {score:.2f})")
```

Perfect for feeding context to LLMs in RAG systems.

## Key Concepts

| Term | Meaning |
|------|---------|
| **Minicolumn** | A unit in the network (word, bigram, concept, or document) |
| **Lateral connections** | Links between co-occurring terms ("neurons that fire together, wire together") |
| **PageRank** | Importance score based on connection graph |
| **TF-IDF** | Term distinctiveness (rare but meaningful terms score higher) |
| **Query expansion** | Adding related terms to improve search recall |

## Saving and Loading

```python
# Save the processed corpus
processor.save("my_corpus.pkl")

# Load it later (instant startup, no reprocessing)
processor = CorticalTextProcessor.load("my_corpus.pkl")
```

## Next Steps

- Run `python showcase.py` to see all features in action
- Read [cookbook.md](cookbook.md) for common recipes
- See [CLAUDE.md](../CLAUDE.md) for the full developer guide
- Check [architecture.md](architecture.md) for how it works

### For AI Agents

If you're an AI coding assistant exploring this codebase:

1. **Use `.ai_meta` files** for rapid module understanding:
   ```bash
   cat cortical/processor/__init__.py.ai_meta  # Structured overview of main API
   ```

2. **Check Claude skills** in `.claude/skills/` for:
   - `codebase-search` - Semantic search
   - `corpus-indexer` - Index management
   - `ai-metadata` - Metadata viewer

3. **See AI Agent Onboarding** in [CLAUDE.md](../CLAUDE.md#ai-agent-onboarding) for detailed guidance

**Note:** The processor is now a package (`cortical/processor/`) with mixin-based composition for better organization.

## Common Patterns

### Batch Processing

```python
documents = [
    ("doc1", "First document content..."),
    ("doc2", "Second document content..."),
    ("doc3", "Third document content..."),
]

for doc_id, content in documents:
    processor.process_document(doc_id, content)

processor.compute_all()  # One-time computation for all docs
```

### Adding Documents Incrementally

```python
# Already have a computed processor
processor.add_document_incremental("new_doc", "New content here...", recompute='tfidf')
```

### Document Metadata

```python
processor.process_document(
    "article1",
    "Content here...",
    metadata={"author": "Jane Doe", "date": "2025-01-15", "source": "blog"}
)

# Retrieve later
meta = processor.get_document_metadata("article1")
print(meta["author"])  # Jane Doe
```

## Troubleshooting

**Import error?**
```bash
pip install -e .  # Make sure package is installed
```

**No results from search?**
```python
processor.compute_all()  # Make sure to compute after adding documents
```

**Slow performance?**
```python
# Use fast search for large corpora
results = processor.fast_find_documents("query")
```

---

*Ready for more? See the [full documentation index](README.md).*
