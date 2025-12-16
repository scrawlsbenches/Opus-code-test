# Our Story: Building Search That Searches Itself

*How we develop the Cortical Text Processor*

---

## The Premise

What if the best way to test a search library was to use it to search itself?

That's not a hypothetical - it's our actual development process.

---

## How It Started

The Cortical Text Processor began as an experiment: could we build information retrieval using only proven algorithms (PageRank, TF-IDF, Louvain) without neural networks or external dependencies?

The answer was yes. But more interesting was what happened next.

---

## The Dog-Fooding Loop

As the codebase grew, we needed to search it. So we indexed it with... the codebase.

```bash
# This is a real command we run daily
python scripts/index_codebase.py --incremental
python scripts/search_codebase.py "how does query expansion work"
```

When search didn't find what we needed, we had to fix it. But unlike using someone else's search library, we could:

1. **See** why results ranked poorly
2. **Trace** through the graph connections
3. **Identify** the algorithmic issue
4. **Fix** it immediately
5. **Test** on our own code

This created a virtuous cycle:

```
Use search → Find problem → Fix algorithm → Better search → Use search...
```

---

## Real Examples

### The Bigram Separator Bug

We searched for "neural networks" but results weren't matching bigrams.

Investigation:
```python
# We could inspect exactly what was happening
bigram_col = processor.layers[CorticalLayer.BIGRAMS].get_minicolumn("neural_networks")
# None! Why?

bigram_col = processor.layers[CorticalLayer.BIGRAMS].get_minicolumn("neural networks")
# Found it! Bigrams use spaces, not underscores
```

The fix was simple once we could see the problem. With a black box, we'd still be guessing.

### The Performance Bottleneck

`compute_all()` was taking 30+ seconds. Initial assumption: Louvain clustering (the most complex algorithm) was slow.

We profiled:
```
bigram_connections: 20.85s  ← Actual problem!
semantics:          30.05s  ← Also a problem!
louvain:            2.2s    ← Not the bottleneck
```

The "obvious" culprit wasn't the issue. Common terms like "self" were creating O(n²) pair explosions.

Fix: Add limits (`max_bigrams_per_term=100`). Time dropped to 10 seconds.

**Lesson:** Profile before optimizing. The obvious answer is often wrong.

### The Test File Ranking Problem

Searching for "PageRank implementation" returned test files before the actual implementation.

Why? Test files mentioned "PageRank" frequently and were well-connected in the graph.

Fix: Added `is_test_file()` detection with a scoring penalty.

We could only find this because we could inspect rankings and trace connections.

---

## What We Learned

### 1. Transparency Enables Debugging

Every bug we fixed required being able to see:
- What terms were extracted
- How queries expanded
- Which connections influenced ranking
- Why one result beat another

Black box systems don't give you this.

### 2. Real Usage Reveals Real Problems

Unit tests catch obvious bugs. Dog-fooding catches subtle ones:
- "This should rank higher" (relevance)
- "This is too slow" (performance)
- "This connection doesn't make sense" (quality)

### 3. The Best Documentation Is Usage

Our CLAUDE.md is detailed because we use it. The debugging section exists because we needed to debug. The performance lessons are real lessons from real problems.

---

## The Artifacts

Dog-fooding produced these project elements:

| Artifact | Origin |
|----------|--------|
| `scripts/index_codebase.py` | We needed to index ourselves |
| `scripts/search_codebase.py` | We needed to search ourselves |
| `.ai_meta` generator | AI agents needed module navigation |
| Performance limits | We hit O(n²) walls |
| Complexity hints | We needed to know what was slow |
| Staleness tracking | We needed to know what to recompute |

These aren't theoretical features - they're solutions to problems we actually had.

---

## Try It Yourself

```bash
# Clone the repository
git clone https://github.com/scrawlsbenches/Opus-code-test.git
cd Opus-code-test

# Index the codebase (uses the library on itself)
python scripts/index_codebase.py

# Search for anything
python scripts/search_codebase.py "how does TF-IDF work"
python scripts/search_codebase.py "PageRank implementation"
python scripts/search_codebase.py "query expansion lateral connections"

# Interactive mode
python scripts/search_codebase.py --interactive
```

You'll be using the same search we use to develop the search.

---

## The Philosophy

> "If you won't use your own software, why should anyone else?"

We're not building search for hypothetical users with hypothetical needs. We're building search for ourselves, on our own code, solving our own problems.

Every feature exists because we needed it.
Every bug fix came from us hitting it.
Every optimization solved a real slowdown.

This is software built by usage, not speculation.

---

*"The best test suite is production use on code you care about."*
