# Claude Usage Guide: Semantic Search System

This guide is written specifically for Claude (AI agents) to understand how to effectively use the Cortical Text Processor's semantic search system when working with this codebase.

## Overview

The Cortical Text Processor can index and semantically search its own codebase, providing meaning-based retrieval instead of simple keyword matching. This guide explains how to use this capability strategically during development tasks.

**Key principle:** The system finds code by understanding intent and concepts, not just exact keywords. "Fetch", "get", "load", and "retrieve" are treated as semantically similar.

---

## Table of Contents

1. [When to Use Codebase-Search](#when-to-use-codebase-search)
2. [When to Use Direct File Reading](#when-to-use-direct-file-reading)
3. [Formulating Effective Search Queries](#formulating-effective-search-queries)
4. [Understanding Search Results](#understanding-search-results)
5. [When to Re-Index](#when-to-re-index)
6. [Handling No Results](#handling-no-results)
7. [Iterative Search Strategy](#iterative-search-strategy)
8. [Query Expansion Leverage](#query-expansion-leverage)
9. [System Limitations and Workarounds](#system-limitations-and-workarounds)
10. [Common Code Query Patterns](#common-code-query-patterns)
11. [Performance Considerations](#performance-considerations)

---

## When to Use Codebase-Search

Use the **codebase-search** skill when you need to:

### 1. Find implementations of concepts
```
"How does PageRank algorithm work?"
"How is TF-IDF computed?"
"How are bigrams created?"
```

The system will find relevant code passages even if your exact words don't match the implementation. For example, searching for "importance scoring" will find PageRank code.

### 2. Locate functionality by intent
```
"Where do we handle errors?"
"Where do we validate input?"
"Where do we tokenize text?"
```

Intent-based queries parse the natural language structure and find code implementing that action.

### 3. Understand relationships between components
```
"What connects to the tokenizer?"
"How do layers interact?"
"What uses layer 2 concepts?"
```

The system understands component relationships through graph connections.

### 4. Explore semantic concepts across the codebase
```
"Neural network terminology"
"Graph algorithms"
"Performance optimization patterns"
```

Query expansion automatically includes related terms, finding all discussions of a concept.

### 5. When you need to understand code context
You want to see how something is actually implemented, not just read the file directly. The search system gives you relevant passages in context.

**Cost consideration:** Search is fast (~1 second for typical queries), so it's efficient for exploratory research.

---

## When to Use Direct File Reading

Use **direct file reading** (Read tool) when you:

### 1. Know the exact file location
If you already know the file path (e.g., `cortical/processor/compute.py`), reading directly is faster than searching.

### 2. Need the complete file context
When you need to see the entire file structure, imports, and all methods in a class, reading the file is more efficient than multiple targeted searches.

### 3. Are implementing a pattern you've already found
After a search tells you the file location, switch to direct reading to implement your changes.

### 4. Need accurate line numbers for edits
While search provides file:line references, reading the file confirms the exact content at those lines.

### 5. The concept is very common
If the concept appears frequently (like "process" or "handle"), search may return many results. Direct reading is faster when you know where to look.

**Workflow:** Search → Find file → Read file → Implement

---

## Formulating Effective Search Queries

### Query Structure

The system parses queries into three components:

1. **Question word** (optional): "where", "how", "what", "why" → affects intent
2. **Action verb** (optional): "handle", "process", "create", "validate" → narrows scope
3. **Subject**: The main concept you're searching for

Examples:
- "where do we validate input?" → Intent: location, Action: validate, Subject: input
- "how are bigrams created?" → Intent: implementation, Action: create, Subject: bigrams
- "PageRank algorithm" → Intent: general, Subject: PageRank algorithm

### Writing Effective Queries

**✓ DO:** Use natural language as you would ask a colleague
```
"How does query expansion find related terms?"
"Where do we compute document relevance?"
"What's the structure of a minicolumn?"
```

**✓ DO:** Include multiple related terms
```
"PageRank importance scoring algorithm" (better than just "PageRank")
"TF-IDF term weighting relevance" (better than just "TF-IDF")
```

**✓ DO:** Use intent words
```
"Find implementations of label propagation"
"Locate the tokenizer code"
"Show me how errors are handled"
```

**✗ DON'T:** Use only exact technical names without context
```
"L0" (too abstract - use "token layer" instead)
"col" (use "minicolumn" or "column")
```

**✗ DON'T:** Use implementation details you're not sure about
```
"Use lateral_connections" (search for the concept instead: "related terms")
"_id_index lookup" (search for: "ID lookup performance")
```

**✗ DON'T:** Search for very common words alone
```
"the" or "and" (these appear everywhere)
"layer" (almost every file mentions layers - add context: "layer connections")
```

### Query Length

- **Short queries (1-3 words):** Fast, but may return many results
  - Good for: "PageRank", "stemming", "TF-IDF"
  - Problem: High recall, may need filtering

- **Medium queries (4-6 words):** Optimal for most cases
  - Good for: "how bigrams are created", "Layer 0 token structure"
  - Sweet spot for precision and recall

- **Long queries (7+ words):** Very specific, low recall
  - Good for: Complete question phrases
  - Problem: May miss results if wording doesn't match docs

**Best practice:** Start with 4-5 word queries; adjust based on results.

---

## Understanding Search Results

### Result Format

Each result shows:

```
[N] cortical/processor.py:1265
    Score: 0.847
  - Passage text showing relevant code
  - Up to 5 lines displayed by default
```

### Score Interpretation

Scores range from 0.0 to 1.0:

| Score | Meaning | What to do |
|-------|---------|-----------|
| 0.9-1.0 | Excellent match | This is what you're looking for |
| 0.75-0.89 | Strong match | Very relevant, likely useful |
| 0.6-0.74 | Good match | Relevant but may need context |
| 0.45-0.59 | Weak match | May be tangentially related |
| <0.45 | Poor match | Likely noise, but sometimes useful |

**Note:** Scores depend on query quality and corpus structure. A 0.75 for a common topic may be more relevant than a 0.95 for a niche query.

### File:Line References

The format `filename:linenumber` tells you:
- Which file to examine
- Approximately where to look (line number may be off by ±10 lines due to chunking)

**Action:** When you get a file:line reference:
1. Use Read tool on that file
2. Look around the suggested line (±5 lines on each side)
3. If not found, search again with different terms

### Passage Text

The system shows relevant passages of code in context:

- **In brief mode** (default): First 5 lines of the passage
- **In verbose mode** (`--verbose` flag): Up to 10 lines

**Interpreting passages:**
- Look for function definitions, class declarations, and key logic
- Passages may be partial—read the full file for complete understanding
- Comments in passages are usually significant (the system ranks them highly)

---

## When to Re-Index

The semantic search uses a pre-built index (`corpus_dev.pkl`) created from your codebase. It's not real-time—it reflects the state when the index was last built.

### Use corpus-indexer After:

**1. You make code changes** (Most important)
```
- Add a new function
- Modify algorithm logic
- Change class structure
- Add new documentation
```

**When:** Use `--incremental` flag for speed (1-2 seconds vs 2-3 seconds for full rebuild)
```python
# In your task: "Use corpus-indexer with --incremental flag"
```

**2. You add new files**
The indexer automatically detects new files in `cortical/`, `tests/`, and `docs/`.

**When:** After adding `new_feature.py` or `test_new_feature.py`

**3. Major refactoring**
If you restructure multiple files, use `--force` flag to ensure clean rebuild.

### When Index Staleness Matters

Search results won't reflect changes until re-indexing. This is fine for:
- Reading old code
- Understanding historical implementation
- Learning the architecture

This is problematic for:
- Verifying your own changes are searchable
- Finding newly added functionality
- Debugging code you just wrote

### Index Staleness Detection

Before using search, check if the index is stale:

```bash
# Check what would change
python scripts/index_codebase.py --status
```

If files changed since last index, results may be out of date.

---

## Handling No Results

When a search returns no results, try these strategies in order:

### Strategy 1: Broaden Your Query

**Narrow query with no results:**
```
"compute_semantic_pagerank with damping factor"
```

**Broadened version:**
```
"PageRank algorithm"
```

**Action:** Remove specific implementation details and search for the concept.

### Strategy 2: Use Synonym/Related Terms

**Query with no results:**
```
"fetch documents from corpus"
```

**Synonym version:**
```
"retrieve documents relevance"
```

**Action:** Replace implementation-specific words with general synonyms.

### Strategy 3: Search Different Layers

**Technical terms not found:**
```
"minicolumn lateral connection weight"
```

**Higher-level concept:**
```
"related terms word associations"
```

**Action:** Describe the concept instead of the implementation.

### Strategy 4: Check if Index Exists

**Problem:** "Error: Corpus file not found"

**Solution:**
```bash
python scripts/index_codebase.py
```

This creates `corpus_dev.pkl` (~2-3 seconds).

### Strategy 5: Use Direct File Search

If semantic search fails, fall back to:

1. **Grep search** for exact keywords:
   ```
   grep -r "function_name" cortical/
   ```

2. **Direct file reading** if you know the likely file:
   ```
   Read cortical/analysis.py
   ```

### Strategy 6: Check Query Expansion

Use `--expand` flag to see what the system is actually searching for:

```bash
python scripts/search_codebase.py "your query" --expand
```

This shows the expanded terms. If expansion is incorrect, try a different query.

### Why No Results Happen

1. **Concept doesn't exist in codebase** - You're asking for something that isn't implemented
2. **Different terminology** - The codebase uses different words than you're using
3. **Index is stale** - Recent changes haven't been indexed
4. **Query too specific** - You're combining terms that don't co-occur
5. **Implementation detail** - You're searching for internal variable names instead of the concept

---

## Iterative Search Strategy

When researching a complex topic, use iterative searching:

### Iteration 1: Broad Exploration
```
Query: "PageRank"
Goal: Find where PageRank is implemented
Action: Choose the most relevant result file
Result: cortical/analysis.py:22
```

### Iteration 2: Find Related Components
```
Query: "how does PageRank use connections"
Goal: Understand what PageRank operates on
Action: Search results show "lateral connections" and "weighted edges"
Result: Learn that PageRank uses graph structure
```

### Iteration 3: Understand Integration
```
Query: "where is PageRank computed in processor"
Goal: Find where PageRank is called
Action: Results show processor.py lines that trigger compute_pagerank
Result: Understand when PageRank runs (after corpus changes)
```

### Iteration 4: Deep Dive
```
Query: "PageRank damping factor convergence"
Goal: Understand algorithm parameters
Action: Read the full analysis.py function
Result: Understand implementation details
```

**Pattern:** Start broad → narrow down → deepen understanding → read full files

---

## Query Expansion Leverage

The system automatically expands queries using:

1. **Lateral connections** - Terms frequently appearing together
2. **Concept clusters** - Semantic groupings
3. **Word variants** - Plurals, stems, related forms
4. **Code concepts** - Programming synonyms (get/fetch/load)

### How to Leverage Expansion

**1. Use umbrella terms**

Rather than searching for specific functions:
```
# Instead of: "expand_query"
# Search for: "query expansion"
```

The system will automatically find `expand_query`, `get_expanded_query_terms`, etc.

**2. Use related terminology**

Expansion finds connections:
```
"authentication" → also finds "login", "credential", "token", "session"
"fetch" → also finds "get", "load", "retrieve", "access"
```

**3. Check what's actually being searched**

Use `--expand` flag:
```bash
python scripts/search_codebase.py "PageRank" --expand
```

Output shows:
```
pagerank: 1.000
importance: 0.847
score: 0.812
rank: 0.791
...
```

These are the actual terms being searched.

**4. Add expansion hints to queries**

If expansion misses terms, add them explicitly:
```
# Instead of: "PageRank"
# Try: "PageRank importance scoring algorithm"
```

Now expansion includes more related terms.

### Expansion Limitations

Expansion works well for:
- Common terms (appear in many documents)
- Concepts with multiple discussions
- Well-connected terms in the knowledge graph

Expansion works poorly for:
- Rare specialized terms (appear in 1-2 documents)
- Very new features (not yet well-connected)
- Acronyms (expansion may not handle well)

---

## System Limitations and Workarounds

### Limitation 1: Exact Matches Don't Always Score Highest

**Problem:** When you search for a function name exactly, variations sometimes score higher.

```
Query: "find_documents_for_query"
Top result: "fast_find_documents" (unrelated function)
```

**Reason:** The system ranks by relevance semantically, not by exact match.

**Workaround:** Read the file you found or refine your query:
```
"find_documents relevance scoring"
```

### Limitation 2: Code Structure Queries May Miss Abstract Concepts

**Problem:** Searching for the structure of a data type:
```
"what fields does Minicolumn have"
```

May not find the class definition as well as you'd hope.

**Reason:** The definition doesn't discuss relationships; it just declares fields.

**Workaround:** Search for the concept instead:
```
"minicolumn structure representation"
```

Or use direct file reading for data structure definitions:
```
Read cortical/minicolumn.py
```

### Limitation 3: Semantic Similarity Can Be Too Broad

**Problem:** Searching for common concepts returns too many results:
```
Query: "connection"
Result: Returns all mentions of "connections" (hundreds)
```

**Reason:** "Connection" is a core concept mentioned everywhere.

**Workaround:** Be more specific:
```
"lateral connections co-occurrence"
"feedforward connections hierarchy"
```

### Limitation 4: Fast Mode Only Returns Documents, Not Passages

**Problem:** When using `--fast` flag, you only get file names, not specific passages.

```bash
python scripts/search_codebase.py "PageRank" --fast
# Returns: cortical/analysis.py:1 (without specific passage)
```

**Reason:** Fast mode skips passage extraction for speed (~2-3x faster).

**Workaround:** Use without `--fast` for specific passages, or read the file directly after getting the filename.

### Limitation 5: Index Doesn't Cover Git History

**Problem:** You can't search for how code looked before changes.

**Reason:** The index is built from current files only.

**Workaround:** Use git history for temporal queries:
```bash
git log -p cortical/query/ | grep "function_name"
```

### Limitation 6: Documentation May Be Outdated

**Problem:** Docs in the index reflect what was written, not necessarily what code actually does.

```
Query: "how layer computation works"
Result: May find outdated documentation
```

**Reason:** Docs and code can drift.

**Workaround:** Verify by reading the actual code after finding relevant documentation.

### Limitation 7: Very New Code May Not Be Discoverable

**Problem:** Code you just wrote won't be found until re-indexing.

**Workaround:** Re-index with `--incremental` after writing code:
```bash
python scripts/index_codebase.py --incremental
```

---

## Common Code Query Patterns

### Finding Algorithm Implementations

**Goal:** Understand how a specific algorithm works

```
"PageRank importance scoring"
"TF-IDF term weighting"
"label propagation clustering"
```

**What to expect:** Functions implementing the algorithm, parameter documentation

### Finding Bug Locations

**Goal:** Locate where a bug might be

```
"bigram separator space" (if debugging bigram issues)
"layer ID index lookup" (if debugging lookups)
"tokenizer stemming" (if debugging tokenization)
```

**What to expect:** Code that handles the buggy component

### Finding Integration Points

**Goal:** Understand how components connect

```
"where PageRank results used"
"TF-IDF score returned"
"minicolumn connected to layer"
```

**What to expect:** Code that calls or uses the component

### Finding Test Patterns

**Goal:** Understand how to test a feature

```
"test PageRank computation"
"unittest layer structure"
"assert results valid"
```

**What to expect:** Test files showing testing patterns

### Finding Performance Optimizations

**Goal:** Understand efficiency strategies

```
"fast search document only"
"incremental indexing changes"
"O(1) ID lookup cache"
```

**What to expect:** Code with performance-related comments/optimization

### Finding Data Structure Details

**Goal:** Understand internal representations

```
"minicolumn connections fields"
"layer minicolumns dictionary"
"document ID format"
```

**What to expect:** Class definitions, docstrings explaining structure

---

## Performance Considerations

### When to Use Each Search Method

| Method | Speed | Use Case |
|--------|-------|----------|
| Normal search | 1-2s | Default, accurate passage extraction |
| Fast search (`--fast`) | 0.2-0.5s | Need just documents, not passages |
| Direct file read | <0.1s | Know exact file location |
| Interactive mode | 0.5-1s per query | Exploratory research sessions |

### Batching Queries

If you have multiple searches, use interactive mode instead of multiple CLI calls:

```bash
python scripts/search_codebase.py --interactive
# Then issue multiple queries in one session
# More efficient than multiple command calls
```

### Caching Expansion

If you're searching for related terms repeatedly:

```python
# In code, use:
processor.expand_query_cached(query)
```

Instead of:
```python
processor.expand_query(query)
```

The cached version uses LRU cache for repeated queries.

### Index Size Trade-offs

**Fast mode (default):**
- Smaller index (~30MB)
- Faster indexing (2-3 seconds)
- Fast search (0.5-1s)
- No bigram connections, no concept analysis

**Full analysis mode:**
- Larger index (~100+MB)
- Slow indexing (10+ minutes)
- More comprehensive results
- Use only when you need deep exploration

For normal development: **Use fast mode**. Use `--full-analysis` only for research sessions.

---

## Decision Tree: How to Find Code

```
Do you know the exact file?
├─ YES: Use Read tool directly
└─ NO: Continue...

Do you know what to search for?
├─ YES: Use codebase-search with query
└─ NO: Continue...

Is it a well-known component?
├─ YES: Search for the component name
└─ NO: Continue...

Can you describe what it does?
├─ YES: Search for the concept/behavior
└─ NO: Use grep or browse manually

Is the search too slow?
├─ YES: Use --fast flag or break into narrower queries
└─ NO: Proceed normally

Did you get results?
├─ YES: Pick the best match, read full file
└─ NO: Go to "Handling No Results" section
```

---

## Summary for Claude

When working with this codebase:

1. **Start with search, not reading** - The semantic search is fast and gives you context
2. **Use natural language queries** - Write queries as you would ask a colleague
3. **Trust the expansion** - The system automatically finds related terms
4. **Check scores, but don't over-interpret** - High scores are good, but context matters more
5. **Re-index after changes** - Always use `--incremental` after making code changes
6. **Fall back to direct reading** - Once you have a file:line reference, switch to Read
7. **Broaden when stuck** - If search returns nothing, remove implementation details and try again
8. **Use iterative refinement** - Start broad, then narrow based on what you learn

The semantic search system is designed to accelerate your understanding of the codebase by making it searchable by meaning, not just keywords. Use it as your primary tool for exploration and learning.

---

*Last updated: 2025-12-10*
*For the Cortical Text Processor codebase*
