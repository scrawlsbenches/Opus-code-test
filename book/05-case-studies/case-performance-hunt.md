---
title: "The Great Performance Hunt"
generated: 2025-12-17T00:00:00Z
generator: "manual-synthesis"
source_files:
  - "CLAUDE.md"
  - "cortical/processor/compute.py"
  - "cortical/analysis/connections.py"
tags:
  - case-study
  - performance
  - debugging
  - profiling
  - O(n²)
---

# Case Study: The Great Performance Hunt

*A tale of assumptions, profiling, and unexpected bottlenecks.*

## The Problem

It started with a timeout. The `compute_all()` function was hanging on a corpus of just 125 documents—far smaller than our target of 10,000+. Something was fundamentally wrong.

The system would start processing, print "Computing PageRank...", then silence. No errors, no warnings, just an infinite wait. After 30 seconds, the timeout would trigger and the process would die.

This wasn't a minor performance issue. This was a showstopper.

## The Suspect

The obvious culprit was Louvain clustering. Think about it:

- **Most complex algorithm** - O(n log n) community detection with multiple passes
- **Graph manipulation** - Rewiring communities iteratively until modularity converges
- **Nested loops** - Pass after pass until stability

Every instinct, every pattern-matching neuron in our brains said: "Start there. It has to be Louvain."

But assumptions are dangerous.

## The Investigation

We started with profiling, not guessing. The `profile_full_analysis.py` script measured every phase of `compute_all()`:

```bash
python scripts/profile_full_analysis.py
```

The results were shocking:

| Phase | Before | After | Fix |
|-------|--------|-------|-----|
| `bigram_connections` | **20.85s timeout** | 10.79s | `max_bigrams_per_term=100`, `max_bigrams_per_doc=500` |
| `semantics` | **30.05s timeout** | 5.56s | `max_similarity_pairs=100000`, `min_context_keys=3` |
| `louvain` | **2.2s** | 2.2s | **Not the bottleneck!** |

Read that last line again: **Louvain was innocent.**

The algorithm everyone suspected—the complex graph clustering with multiple iterative passes—was responsible for just 2.2 seconds of a 50+ second hang.

99% of the execution time was hidden in `bigram_connections()` and `extract_corpus_semantics()`.

## The Discovery

Then we saw it. Looking at the code in `cortical/analysis/connections.py`:

```python
# Build indexes for efficient lookup
left_index: Dict[str, List[Minicolumn]] = defaultdict(list)
right_index: Dict[str, List[Minicolumn]] = defaultdict(list)

for bigram in bigrams:
    parts = bigram.content.split(' ')
    if len(parts) == 2:
        left_index[parts[0]].append(bigram)
        right_index[parts[1]].append(bigram)

# Connect bigrams sharing components
for component, bigram_list in left_index.items():
    # THIS is where it exploded
    for i, b1 in enumerate(bigram_list):
        for b2 in bigram_list[i+1:]:
            # Create connection between b1 and b2
```

The problem was hiding in plain sight. For every term that appears in bigrams, we were creating connections between **all pairs** of bigrams containing that term.

**Common terms like "self" appeared in hundreds of bigrams.**

If "self" appears in 300 bigrams (self_attention, self_healing, self_referential, etc.), the nested loop creates:

```
300 × 299 / 2 = 44,850 connections
```

For a single term.

Now imagine dozens of common terms ("return", "function", "value", "data", "process"). Each creating tens of thousands of pairwise connections.

**O(n²) complexity from common terms was creating millions of pairs.**

The complexity analysis confirmed it:

```python
# Without limits: O(n_bigrams²) worst case from common terms creating all-to-all connections
# With limits: O(n_terms * max_bigrams_per_term² + n_docs * max_bigrams_per_doc²)
# Typical with defaults (100, 500): O(n_terms * 10000 + n_docs * 250000) ≈ O(n_bigrams) linear
```

Without limits, the algorithm had **quadratic worst-case complexity**. With limits, it became **effectively linear**.

## The Solution

The fix was elegant: **skip overly common terms** to prevent the O(n²) explosion:

```python
def compute_bigram_connections(
    layers: Dict[CorticalLayer, HierarchicalLayer],
    component_weight: float = 0.5,
    chain_weight: float = 0.7,
    cooccurrence_weight: float = 0.3,
    max_bigrams_per_term: int = 100,      # NEW: Prevent O(n²) from common terms
    max_bigrams_per_doc: int = 500,       # NEW: Prevent O(n²) from large docs
    max_connections_per_bigram: int = 50  # NEW: Cap per-bigram connections
) -> Dict[str, Any]:
    """
    Compute lateral connections between bigrams.

    Args:
        max_bigrams_per_term: Skip terms appearing in more than this many bigrams
            to avoid O(n²) explosion from common terms like "self", "return"
        max_bigrams_per_doc: Skip documents with more than this many bigrams for
            co-occurrence connections to avoid O(n²) explosion
    """
```

With these limits in place:

```python
# Left component matches: "neural_networks" ↔ "neural_processing"
for component, bigram_list in left_index.items():
    # Skip overly common terms to avoid O(n²) explosion
    if len(bigram_list) > max_bigrams_per_term:
        skipped_common_terms += 1
        continue

    for i, b1 in enumerate(bigram_list):
        for b2 in bigram_list[i+1:]:
            # Safe now - bounded by max_bigrams_per_term²
            create_connection(b1, b2, weight=component_weight)
```

**Results:**
- `bigram_connections`: 20.85s timeout → **10.79s** (48% improvement)
- `semantics`: 30.05s timeout → **5.56s** (81% improvement)
- Total `compute_all()`: timeout → **~27s** (viable for production)

The same approach was applied to `extract_corpus_semantics()`:

```python
max_similarity_pairs: int = 100000  # Prevent similarity explosion
min_context_keys: int = 3           # Require meaningful context overlap
```

## The Lesson

**Profile before optimizing.** The obvious culprit is often innocent. The real bottleneck hides in unexpected places.

We suspected Louvain—the complex, iterative graph algorithm with nested loops and community rewiring. The actual problem was a simple nested loop over common terms, creating millions of unnecessary connections.

**Key takeaways:**

1. **Measure, don't assume** - Run the profiler before making changes
2. **Look for O(n²) patterns** - Nested loops over unbounded collections
3. **Common items are dangerous** - High-frequency terms/documents create all-to-all explosions
4. **Add limits early** - Prevent worst-case scenarios with sensible bounds
5. **Track what you skip** - Return stats on skipped items for monitoring

## The Aftermath

This investigation led to several improvements across the codebase:

1. **Profiling became standard practice** - `profile_full_analysis.py` is now run routinely
2. **O(n²) awareness** - Code reviews specifically check for quadratic patterns
3. **Limit parameters everywhere** - All connection-building functions have max limits
4. **Performance tests** - Regression tests verify compute times stay bounded
5. **Documentation** - CLAUDE.md now includes "Performance Lessons Learned" section

The fix enabled the system to scale from 125 documents (timeout) to **10,000+ documents** (27 seconds).

## Try It Yourself

Run the profiler on your own corpus to identify bottlenecks:

```bash
python scripts/profile_full_analysis.py
```

Watch for these warning signs:
- O(n²) patterns in loops over connections
- Common terms/documents creating explosions
- Phases taking >10x longer than expected
- Nested loops without bounds

Look for code like this:

```python
# DANGER: O(n²) if items list is unbounded
for i, item1 in enumerate(items):
    for item2 in items[i+1:]:
        # Creates n × (n-1) / 2 operations
```

And replace with:

```python
# SAFE: Bounded by max_items_per_group
for group, items in grouped_items.items():
    if len(items) > max_items_per_group:
        continue  # Skip overly common groups

    for i, item1 in enumerate(items):
        for item2 in items[i+1:]:
            # Now bounded by max_items_per_group²
```

---

**Remember:** The algorithm you suspect is often innocent. The real bottleneck is hiding in the code you didn't think to check.

**Profile first. Optimize second. Always.**
