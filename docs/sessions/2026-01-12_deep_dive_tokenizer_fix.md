## Deep Dive Summary

### Are We Doing This Correctly?

**Partially.** The deep code review revealed several issues:

1. **Two separate tokenizer implementations** - `cortical/tokenizer.py` has a well-tested `split_identifier` function, but `BPETokenizer` in `text_bridge.py` had its own regex-based implementation. This violated DRY and the regex was inferior (couldn't handle acronyms).

2. **No documentation on train/reindex order** - Future sessions had no guidance on which to run first.

3. **The AST parser exists but is for structural analysis** - `cortical/spark/ast_index.py` provides class/function/call graph indexing, not for tokenization. The tokenizer properly handles identifier splitting without AST.

### What I Fixed

| Issue | Fix |
|-------|-----|
| BPETokenizer regex was inferior | Now uses `split_identifier` from `cortical.tokenizer` |
| XMLParser → ['parser'] (lost acronym) | XMLParser → ['xml', 'parser'] ✓ |
| No train/reindex documentation | Added "Training and Reindexing Order" section to CLAUDE.md |

### Train/Reindex Order

**Train first, then Reindex.**

```
Train → Creates word atoms and raw similarity links
Reindex → Applies IDF weighting based on document frequency
```

If you reindex before training, you're recalculating weights on stale data.

### Verification

After retraining with the fixed tokenizer:

```
Query "bridge" → Top result: "atoms" (1.7388 IDF weight)
Query "atoms" → Top result: "bridge" (1.7388 IDF weight)
```

The semantic links now connect class name components!

### Communication to Future Sessions

Added to CLAUDE.md section "Training and Reindexing Order (IMPORTANT)":
- Correct order with explanation
- Why it matters (staleness, IDF)
- When to reindex
- Quick reference commands

### Commits

| Commit | Description |
|--------|-------------|
| `0940bf2a` | fix(cognitive): Use split_identifier + add documentation |
| `0de22c4d` | chore(cognitive): Retrain model with improved identifier splitting |
