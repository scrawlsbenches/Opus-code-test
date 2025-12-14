# Text as Memories: A Knowledge Transfer Guide

**Date:** 2025-12-14
**Status:** Complete
**Related Tasks:** Search relevance investigation (T-20251214-171301-6aa8-001)

## Executive Summary

This document explores how text documents can be conceptualized as **memories**—discrete units of knowledge that, when stored in git, form a persistent, versioned, and interconnected knowledge base. The Cortical Text Processor provides the algorithmic foundation for this metaphor, treating documents as "episodes" and terms as "associations" that strengthen through co-occurrence.

---

## The Memory Metaphor

### Human Memory vs. Text Systems

| Human Memory | Text in Git | Cortical Text Processor |
|--------------|-------------|-------------------------|
| **Encoding** | Writing a document | `process_document()` |
| **Storage** | Committing to git | Minicolumns + connections |
| **Consolidation** | Merging branches | PageRank identifies hubs |
| **Retrieval** | Git search / blame | Query expansion + ranking |
| **Association** | Related memories activate | Lateral connections fire |
| **Forgetting** | Deleting files | `remove_document()` |

### Why This Matters

When you commit text to git, you're not just storing files—you're creating a **persistent memory system** that:

1. **Tracks evolution**: Git history shows how ideas develop over time
2. **Enables collaboration**: Multiple minds contribute to shared memory
3. **Supports retrieval**: Semantic search finds related concepts
4. **Preserves context**: Blame shows when and why knowledge was added

---

## The Cortical Processing Model

### Four Layers of Memory

The processor organizes text into hierarchical layers, inspired by how the visual cortex processes information:

```
Layer 3: DOCUMENTS   →  "Episodes" (complete experiences)
         ↑
Layer 2: CONCEPTS    →  "Schemas" (organized knowledge structures)
         ↑
Layer 1: BIGRAMS     →  "Associations" (paired ideas)
         ↑
Layer 0: TOKENS      →  "Features" (atomic elements)
```

**Example progression:**

```
Document: "Neural networks learn patterns from data"

Layer 0 (Features):     [neural, networks, learn, patterns, data]
Layer 1 (Associations): [neural networks, networks learn, learn patterns, patterns data]
Layer 2 (Schema):       [machine_learning_concepts]
Layer 3 (Episode):      [doc_ml_intro_001]
```

### Hebbian Learning: "Neurons That Fire Together, Wire Together"

The processor applies this neuroscience principle to text:

```python
# When "neural" and "networks" appear together repeatedly:
col_neural.lateral_connections["networks"] += 1

# After processing 100 documents:
# "neural" → "networks": weight 87
# "neural" → "learning": weight 45
# "neural" → "artificial": weight 23
```

**Result**: Terms that co-occur form stronger associations, enabling retrieval through related concepts.

---

## Git as Memory Infrastructure

### The Version Control Memory Model

```
Repository = Long-term Memory Store
├── main branch = Consolidated, verified knowledge
├── feature branches = Working memory (experimental ideas)
├── commits = Memory encoding events
├── merges = Memory consolidation
└── tags = Landmark memories (releases, milestones)
```

### Memory Operations Mapped to Git

| Memory Operation | Git Command | Effect |
|-----------------|-------------|--------|
| **Encode new memory** | `git add && git commit` | Store new knowledge |
| **Recall** | `git log --grep` or semantic search | Retrieve by content |
| **Update memory** | `git commit --amend` | Modify recent memory |
| **Consolidate** | `git merge` | Integrate new with existing |
| **Create association** | Cross-reference in docs | Link related memories |
| **Time travel** | `git checkout <hash>` | Access past states |
| **Attribution** | `git blame` | Who knew what when |

### Practical Pattern: Memory Journaling

Store thoughts, decisions, and learnings as versioned documents:

```bash
# Create a memory entry
echo "Today I learned that PageRank..." > memories/2025-12-14-pagerank.md
git add memories/
git commit -m "memory: PageRank insight from debugging session"

# Later, search your memories
git log --all --grep="PageRank" --oneline
python scripts/search_codebase.py "PageRank importance calculation"
```

---

## Building a Personal Knowledge Base

### Directory Structure for Memory Storage

```
knowledge-base/
├── memories/                    # Daily/episodic memories
│   ├── 2025-12-14-security.md
│   └── 2025-12-15-debugging.md
├── concepts/                    # Consolidated knowledge
│   ├── algorithms/
│   │   ├── pagerank.md
│   │   └── tfidf.md
│   └── patterns/
│       └── hebbian-learning.md
├── decisions/                   # Architectural decisions
│   └── adr-001-layer-structure.md
└── corpus_dev.pkl              # Indexed for semantic search
```

### Indexing Your Memories

```python
from cortical import CorticalTextProcessor
import glob

# Create processor
processor = CorticalTextProcessor()

# Index all memories
for filepath in glob.glob("knowledge-base/**/*.md", recursive=True):
    with open(filepath) as f:
        processor.process_document(filepath, f.read())

# Build connections
processor.compute_all()

# Save for future sessions
processor.save("knowledge-base/corpus_dev.pkl")
```

### Querying Your Knowledge

```python
# Load your memory index
processor = CorticalTextProcessor.load("knowledge-base/corpus_dev.pkl")

# Semantic search
results = processor.find_documents_for_query("debugging network issues")
for doc_id, score in results:
    print(f"{doc_id}: {score:.2f}")

# Find related concepts
expanded = processor.expand_query("network")
# Returns: {network: 1.0, connection: 0.67, layer: 0.45, ...}
```

---

## Memory Consolidation Patterns

### Pattern 1: Daily Memory Capture

```markdown
<!-- memories/2025-12-14.md -->
# Memory Entry: 2025-12-14

## What I Learned
- Fuzzing found a bug in config validation
- NaN and infinity weren't being rejected

## Connections
- Related to: [[concepts/validation.md]]
- Builds on: [[memories/2025-12-13.md]]

## Future Exploration
- [ ] Apply fuzzing to other validation code
```

### Pattern 2: Concept Consolidation

When a topic appears in multiple memories, consolidate:

```markdown
<!-- concepts/input-validation.md -->
# Input Validation

## Core Principle
Always validate at system boundaries.

## Learned From
- 2025-12-14: Fuzzing found NaN/inf bug
- 2025-12-10: Path traversal prevention added

## Implementation
See: cortical/validation.py

## Related Concepts
- [[concepts/security.md]]
- [[concepts/testing.md]]
```

### Pattern 3: Decision Records

Capture **why** decisions were made:

```markdown
<!-- decisions/adr-003-microseconds-in-task-id.md -->
# ADR-003: Add Microseconds to Task IDs

## Status
Accepted (2025-12-14)

## Context
Task IDs were colliding when generated in tight loops.
Format was: T-YYYYMMDD-HHMMSS-XXXX (seconds precision)

## Decision
Add microseconds: T-YYYYMMDD-HHMMSSffffff-XXXX

## Consequences
- Pro: No more collisions in tight loops
- Pro: 1,000,000x more unique IDs per second
- Con: Longer IDs (6 more characters)
```

---

## Semantic Search as Memory Recall

### How Query Expansion Models Association

When you search for "neural", the processor:

1. **Activates** the "neural" minicolumn
2. **Spreads** activation through lateral connections
3. **Includes** associated terms (networks, learning, patterns)
4. **Ranks** documents by cumulative relevance

```python
# Query: "neural"
# Expanded: {neural: 1.0, networks: 0.87, learning: 0.45, artificial: 0.23}
#
# This mimics how thinking of "neural" naturally brings
# "networks" and "learning" to mind
```

### Improving Recall with Metadata

Add tags and links to strengthen associations:

```markdown
---
tags: [machine-learning, neural-networks, debugging]
related: [memories/2025-12-13.md, concepts/pagerank.md]
---

# Today's Learning

The PageRank algorithm can be applied to...
```

---

## Integration with Development Workflow

### Memory-Aware Git Hooks

```bash
#!/bin/bash
# .git/hooks/post-commit

# Re-index after commits to docs/memories
if git diff --cached --name-only | grep -q "^docs/\|^memories/"; then
    python scripts/index_codebase.py --incremental
fi
```

### Semantic Commit Messages

Treat commits as memory encoding:

```bash
# Bad: "fix bug"
# Good: "memory: discovered that TF-IDF scores need normalization"

git commit -m "memory: fuzzing revealed NaN acceptance in config validation"
```

### Branch as Working Memory

```bash
# Create a working memory space
git checkout -b memory/exploring-pagerank

# Capture discoveries
echo "# PageRank Exploration..." > memories/pagerank-deep-dive.md
git add memories/
git commit -m "memory: initial pagerank exploration"

# When ready, consolidate to main
git checkout main
git merge memory/exploring-pagerank
```

---

## Benefits of Text-as-Memories

1. **Searchable**: Semantic search finds related knowledge
2. **Versioned**: See how understanding evolved
3. **Shareable**: Collaborate on shared knowledge
4. **Persistent**: Never lose insights
5. **Attributable**: Know when and why you learned something
6. **Interconnected**: Cross-references strengthen recall

---

## Getting Started

1. **Create a memories directory** in your repo
2. **Index with the processor**: `python scripts/index_codebase.py`
3. **Search semantically**: `python scripts/search_codebase.py "your query"`
4. **Link related concepts** with `[[wiki-style]]` references
5. **Review periodically** to consolidate into concept documents

---

## Related Documentation

- [Architecture Guide](architecture.md) - How layers work
- [Algorithms Guide](algorithms.md) - PageRank, TF-IDF explained
- [Dogfooding Guide](dogfooding.md) - Using the system on itself
- [Query Guide](query-guide.md) - Search techniques

---

*"The palest ink is better than the best memory." — Chinese proverb*

*But versioned, indexed ink is even better.*
