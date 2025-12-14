# Concept: Hebbian Learning in Text Processing

**Tags:** `algorithms`, `neuroscience`, `information-retrieval`
**Consolidated From:** Multiple sessions exploring the Cortical Text Processor

---

## The Core Principle

> "Neurons that fire together, wire together." — Donald Hebb, 1949

In text processing, this translates to:

> **Words that appear together, link together.**

## How It Works

### In Biology

When two neurons are active simultaneously, their synaptic connection strengthens. Over time, activating one neuron makes the other more likely to fire.

### In Text Processing

When two words co-occur within a context window, their connection weight increases:

```python
# Processing: "neural networks process data efficiently"
# Context window: ±3 words

# Result:
connections["neural"]["networks"] += 1.0   # Adjacent
connections["neural"]["process"] += 1.0    # Within window
connections["neural"]["data"] += 1.0       # Within window
connections["networks"]["process"] += 1.0
# etc.
```

### The Accumulation Effect

After processing thousands of documents:

```
"neural" connections:
  → "networks":    weight 87  (very strong - always together)
  → "learning":    weight 45  (often together)
  → "artificial":  weight 23  (sometimes together)
  → "bread":       weight 0   (never together)
```

## Why This Matters

### 1. Query Expansion

When a user searches for "neural", the system can suggest:
- "networks" (strong association)
- "learning" (moderate association)

This improves recall without requiring exact matches.

### 2. Semantic Similarity

Documents with similar Hebbian connection patterns are semantically related, even without shared vocabulary.

### 3. Knowledge Discovery

Strong unexpected connections reveal insights:
- "yeast" → "bread" makes sense
- "yeast" → "genetic research" reveals scientific usage

## Implementation in the Cortical Processor

From `cortical/processor/documents.py`:

```python
def _build_lateral_connections(self, tokens, layer0):
    """Build Hebbian-style connections from co-occurrence."""
    window_size = 3

    for i, token in enumerate(tokens):
        col = layer0.get_minicolumn(token)
        if not col:
            continue

        # Connect to tokens within window
        for j in range(max(0, i - window_size),
                       min(len(tokens), i + window_size + 1)):
            if i != j:
                other = layer0.get_minicolumn(tokens[j])
                if other:
                    # Strengthen the connection
                    col.add_lateral_connection(other.id, weight=1.0)
```

## Limitations

### 1. Common Word Pollution

Frequent words ("the", "is", "self") connect to everything, creating noise.

**Solution:** Stop word filtering, TF-IDF weighting

### 2. O(n²) Scaling

Every pair within a window creates a connection.

**Solution:** Limits on connections per term (`max_bigrams_per_term`)

### 3. Context Blindness

"bank" (financial) and "bank" (river) create the same connections.

**Solution:** Bigram layer provides some disambiguation

## Related Concepts

- [[pagerank.md]] - Uses connection structure to rank importance
- [[tfidf.md]] - Weights terms by distinctiveness
- [[louvain-clustering.md]] - Groups connected terms into concepts

## Sources

- Hebb, D.O. (1949). *The Organization of Behavior*
- Cortical Text Processor source: `cortical/processor/documents.py:88-106`
- README.md visualization of connection weights

---

*Consolidated: 2025-12-14*
*Last updated: 2025-12-14*
