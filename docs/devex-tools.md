# Developer Experience Tools

The Cortical Text Processor includes four powerful CLI tools for exploring and understanding your codebase through semantic analysis.

## Quick Reference

| Tool | Purpose | Example |
|------|---------|---------|
| `corpus_health.py` | Corpus statistics and health metrics | `python scripts/corpus_health.py` |
| `find_similar.py` | Find similar code blocks | `python scripts/find_similar.py file.py` |
| `explain_code.py` | Explain what code is about | `python scripts/explain_code.py file.py` |
| `suggest_related.py` | Suggest related files | `python scripts/suggest_related.py file.py` |

---

## 1. Corpus Health Dashboard

**Purpose:** Monitor corpus statistics, staleness, and overall health.

### Basic Usage

```bash
# Quick health check
python scripts/corpus_health.py

# Detailed statistics
python scripts/corpus_health.py --verbose

# Include concept cluster analysis
python scripts/corpus_health.py --concepts

# Get recommendations
python scripts/corpus_health.py --recommendations
```

### What It Shows

- **Overall Health Score** (0-100) based on:
  - Document count and coverage
  - Layer statistics (tokens, bigrams, concepts)
  - Computation freshness
  - Semantic relations
  - Embeddings

- **Document Statistics:**
  - Total documents and size
  - Document type breakdown (code/test/docs)
  - Average document size

- **Layer Statistics:**
  - Minicolumn counts per layer
  - Average/max connections
  - Connection density

- **Staleness Status:**
  - Which computations need updating
  - Recommendations for improvement

### Example Output

```
Overall Health: Good (64/100)
████████████████████████████████░░░░░░░░░░░░░░░░░░

📚 Documents:
  Total documents: 96
  Total size: 2,162,585 characters
  Average doc size: 22527 chars

🧠 Layer Statistics:
  TOKENS      :   9312 minicolumns (avg 32.8 connections)
  BIGRAMS     :  65320 minicolumns
  CONCEPTS    :      0 minicolumns
  DOCUMENTS   :     96 minicolumns (avg 95.0 connections)
```

---

## 2. Find Similar Code

**Purpose:** Locate code blocks similar to a given file or text snippet using semantic fingerprinting.

### Basic Usage

```bash
# Find similar to a file
python scripts/find_similar.py cortical/processor/__init__.py

# More results
python scripts/find_similar.py cortical/processor/__init__.py --top 10

# Show full passages
python scripts/find_similar.py cortical/processor/__init__.py --verbose

# Explain why they're similar
python scripts/find_similar.py cortical/processor/__init__.py --explain

# Find similar to text snippet
python scripts/find_similar.py --text "def compute_pagerank(graph, damping=0.85):"

# Adjust sensitivity
python scripts/find_similar.py file.py --min-similarity 0.3
```

### How It Works

1. **Fingerprinting:** Computes semantic fingerprint (terms, concepts, bigrams)
2. **Chunking:** Splits documents into ~400 character chunks
3. **Comparison:** Compares fingerprints using multiple similarity metrics
4. **Ranking:** Sorts by overall similarity score

### Similarity Metrics

- **Term Similarity:** Shared vocabulary and TF-IDF weighted terms
- **Concept Similarity:** Shared programming concepts (from `code_concepts.py`)
- **Bigram Similarity:** Shared phrase patterns
- **Overall Similarity:** Weighted combination of above

### Example Output

```
[1] [TEST] tests/test_edge_cases.py:129
    Similarity: 56.7%
    Shared terms: compute, compute_tfidf, corpus, def, document

  Similarity breakdown:
    Term overlap: 59.0%
    Concept overlap: 88.5%
    Bigram overlap: 3.6%
```

### Use Cases

- **Code Review:** Find similar patterns to ensure consistency
- **Refactoring:** Identify duplicate or near-duplicate code
- **Learning:** See how similar concepts are implemented elsewhere
- **Bug Fixes:** Find related code that might have the same issue

---

## 3. Explain This Code

**Purpose:** Analyze and explain what a code file is about using semantic analysis.

### Basic Usage

```bash
# Analyze a file
python scripts/explain_code.py cortical/processor/compute.py

# Detailed analysis
python scripts/explain_code.py cortical/processor/compute.py --verbose

# Show semantic relations
python scripts/explain_code.py cortical/processor/compute.py --relations

# Analyze text directly
python scripts/explain_code.py --text "your code snippet here"
```

### What It Shows

- **Key Terms:** Most important terms by TF-IDF weight
- **Primary Concepts:** Programming concepts detected (e.g., iteration, storage, auth)
- **Concept Clusters:** Which concept clusters this file contributes to
- **Related Documents:** Files with similar content
- **Semantic Relations:** Relationships between terms (with `--relations`)
- **Key Phrases:** Important bigrams/phrases

### Example Output

```
📊 Overview:
  Unique terms: 1132
  Key terms identified: 15
  Concepts detected: 10
  Related documents: 5

🔑 Key Terms (by importance):
   1. self                 ████████ 0.274
   2. verbose              ███ 0.121
   3. str                  ███ 0.110

💡 Primary Concepts:
  • logging                        (0.255)
  • iteration                      (0.128)
  • config                         (0.063)

🔗 Related Documents:
  [TEST] tests/unit/test_processor_core.py     (score: 32.41)
  [CODE] cortical/persistence.py               (score: 19.74)
```

### Use Cases

- **Onboarding:** Quickly understand what a file does
- **Documentation:** Generate insights for documentation
- **Architecture:** Understand file relationships and responsibilities
- **Code Review:** Verify file purpose matches expectations

---

## 4. Suggest Related Files

**Purpose:** Find files related to a given file through imports, concepts, and semantic similarity.

### Basic Usage

```bash
# Find related files
python scripts/suggest_related.py cortical/processor/__init__.py

# More suggestions
python scripts/suggest_related.py cortical/processor/__init__.py --top 15

# Detailed information
python scripts/suggest_related.py cortical/processor/__init__.py --verbose

# Only import relationships
python scripts/suggest_related.py cortical/processor/__init__.py --imports-only
```

### Relationship Types

1. **Imports:** Files this file imports
2. **Imported By:** Files that import this file
3. **Shared Concepts:** Files that share concept clusters
4. **Semantically Similar:** Files with similar semantic fingerprints

### How It Works

- **Import Analysis:** Parses `import` and `from ... import` statements
- **Concept Matching:** Finds files in the same concept clusters
- **Semantic Similarity:** Compares full-file fingerprints
- **Combined Ranking:** Presents results by relationship type

### Example Output

```
📦 Imports (4 files):
  [CODE] cortical/analysis.py
  [CODE] cortical/semantics.py
  [CODE] cortical/query/

📥 Imported By (6 files):
  [TEST] tests/test_processor.py
  [TEST] tests/unit/test_processor_core.py

💡 Shared Concepts (5 files):
  [TEST] tests/test_coverage_gaps.py           (score: 8.2)
  [CODE] cortical/persistence.py               (score: 6.5)

🔍 Semantically Similar (5 files):
  [TEST] tests/test_edge_cases.py              50.2%
  [CODE] cortical/persistence.py               48.5%
```

### Use Cases

- **Navigation:** Quickly find related files while coding
- **Impact Analysis:** See what files depend on changes
- **Architecture:** Understand module dependencies
- **Code Review:** Find all affected files

---

## Workflow Examples

### Understanding a New Module

```bash
# 1. Get overview
python scripts/explain_code.py cortical/analysis.py

# 2. Find related files
python scripts/suggest_related.py cortical/analysis.py

# 3. Find similar implementations
python scripts/find_similar.py cortical/analysis.py --top 3
```

### Code Review Workflow

```bash
# 1. Check corpus health
python scripts/corpus_health.py --recommendations

# 2. Explain changes
python scripts/explain_code.py path/to/changed_file.py --verbose

# 3. Find similar code (for consistency check)
python scripts/find_similar.py path/to/changed_file.py --explain

# 4. Check impact (what imports this?)
python scripts/suggest_related.py path/to/changed_file.py
```

### Finding Duplication

```bash
# Find code similar to a specific function
python scripts/find_similar.py --text "def your_function():" --top 10 --explain

# High similarity threshold (potential duplicates)
python scripts/find_similar.py file.py --min-similarity 0.7
```

### Documentation Generation

```bash
# Get file overview
python scripts/explain_code.py file.py --verbose --relations > file_analysis.txt

# Find related files for cross-references
python scripts/suggest_related.py file.py --top 20
```

---

## Advanced Usage

### Custom Corpus

All tools support custom corpus files:

```bash
python scripts/corpus_health.py --corpus my_corpus.pkl
python scripts/find_similar.py file.py --corpus my_corpus.pkl
```

### Batch Analysis

```bash
# Analyze all Python files
for file in cortical/*.py; do
    echo "=== $file ==="
    python scripts/explain_code.py "$file"
done > analysis_report.txt
```

### Integration with CI/CD

```bash
# Check corpus health in CI
python scripts/corpus_health.py --recommendations
if [ $? -ne 0 ]; then
    echo "Corpus needs attention!"
    exit 1
fi
```

---

## Performance Tips

1. **Incremental Indexing:** Use `--incremental` when updating corpus
   ```bash
   python scripts/index_codebase.py --incremental
   ```

2. **Chunk Size:** Adjust for different code styles
   ```bash
   python scripts/find_similar.py file.py --chunk-size 600  # Larger chunks
   ```

3. **Top-N Tuning:** More results = slower
   ```bash
   python scripts/find_similar.py file.py --top 3  # Faster
   ```

4. **Similarity Threshold:** Higher threshold = faster
   ```bash
   python scripts/find_similar.py file.py --min-similarity 0.3  # Faster
   ```

---

## Troubleshooting

### "Corpus file not found"

```bash
# Index the codebase first
python scripts/index_codebase.py
```

### "File not found in corpus"

```bash
# Re-index with the file
python scripts/index_codebase.py --incremental

# Check what's indexed
python scripts/corpus_health.py --verbose
```

### Low similarity scores

- Try lower `--min-similarity` threshold
- Check if corpus is stale: `python scripts/corpus_health.py`
- Re-index with fresh data

### Slow performance

- Use smaller `--top` values
- Increase `--min-similarity` threshold
- Use `--imports-only` for faster related file search

---

## Script Details

### find_similar.py

**Parameters:**
- `file`: File path to analyze (or use `--text`)
- `--text`: Text snippet instead of file
- `--top N`: Number of results (default: 5)
- `--verbose`: Show full passages
- `--explain`: Show similarity breakdown
- `--min-similarity`: Threshold 0-1 (default: 0.1)
- `--chunk-size`: Chunk size in chars (default: 400)

**Returns:** List of similar code locations with similarity scores

### explain_code.py

**Parameters:**
- `file`: File path to analyze (or use `--text`)
- `--text`: Text snippet instead of file
- `--verbose`: Detailed information
- `--relations`: Show semantic relations

**Returns:** Semantic analysis including terms, concepts, related docs

### suggest_related.py

**Parameters:**
- `file`: File path to analyze
- `--top N`: Suggestions per category (default: 10)
- `--verbose`: Detailed information
- `--imports-only`: Only import relationships

**Returns:** Related files by import, concept, and semantic similarity

### corpus_health.py

**Parameters:**
- `--verbose`: Detailed statistics
- `--concepts`: Include concept cluster analysis
- `--recommendations`: Show improvement suggestions

**Returns:** Health score and comprehensive corpus statistics

---

## See Also

- [Dog-fooding Checklist](dogfooding-checklist.md) - Testing with real usage
- [Architecture Overview](architecture.md) - System design
- [Search Guide](../README.md) - Semantic search capabilities
