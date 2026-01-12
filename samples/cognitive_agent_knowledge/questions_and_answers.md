# Questions and Answers: Context Recovery Guide

## When You're Lost

These questions and answers help you recover context when starting fresh or feeling confused.

---

### Q: I just started a new session. What is this codebase about?

A: This is the Cortical Text Processor codebase - a cognitive computing system with several major components:

1. **CorticalTextProcessor** - NLP pipeline for document analysis
2. **Graph of Thought (GoT)** - Task and decision tracking
3. **Cognitive Agent** - Knowledge graph for learning and querying
4. **CDG** - Core Data Graph storage layer
5. **PRISM** - Hebbian learning and synaptic memory

The cognitive agent specifically helps agents like you understand and navigate the codebase through natural language queries.

---

### Q: What was the previous session working on?

A: Check these sources:
1. `git log --oneline -10` - Recent commits
2. `python -m cortical.got task list --status in_progress` - Active tasks
3. `docs/sessions/*.md` - Session handoff notes

Recent work included IDF-weighted similarity links, performance optimization for predict_next() and incremental saves.

---

### Q: How do I find code related to a concept?

A: Use the cognitive agent:
```bash
python -m cortical.cognitive query "concept_name"
python -m cortical.cognitive ask "How does concept_name work?"
```

Or programmatically:
```python
agent.get_associations("storage", top_k=10)
```

---

### Q: I see "staleness" warnings. What does that mean?

A: Staleness means the IDF weights are outdated. When new documents are trained, existing link weights become stale.

Fix it with:
```bash
python -m cortical.cognitive reindex
```

Staleness above 20% triggers warnings. Reindex recomputes IDF weights for all similarity links.

---

### Q: What's the difference between FOLLOWS and SIMILARITY links?

A:
- **SIMILARITY**: Bidirectional. "A and B co-occur" (semantic relatedness)
- **FOLLOWS**: Directional. "B follows A" (sequential prediction)

Use SIMILARITY for "what relates to X?" queries.
Use FOLLOWS for "what comes after X?" predictions.

---

### Q: Why is predict_next() fast now?

A: We added an `_outgoing` index. Previously, finding links from a word required scanning ALL 248k FOLLOWS links (O(n), 55-70ms). Now it's O(1) lookup (<1ms for rare words, ~20ms for common words with many links).

The fix is in `cortical/cognitive/graph.py`:
- `InMemoryStorage._outgoing` index
- `CognitiveGraph.get_outgoing()` method

---

### Q: Why does save take ~10 seconds?

A: Incremental saves only rewrite dirty shards. But if you touch FOLLOWS or SIMILARITY atoms (the largest types), their entire shard must be rewritten.

With 267k FOLLOWS atoms in 4 shards, each shard is ~16MB and takes ~3s to write.

No-change saves take 0.05s because nothing is dirty.

---

### Q: How do I train on new files?

A:
```bash
# Train on a directory
python -m cortical.cognitive train path/to/files --incremental

# Train on specific files
python -m cortical.cognitive train file1.py file2.py
```

The `--incremental` flag skips already-trained files.

---

### Q: What tests should I run to verify things work?

A:
```bash
# Quick sanity check
python -m pytest tests/smoke/ -v

# IDF-specific tests
python -m pytest tests/behavioral/test_idf_weighted_links_spec.py -v

# Cognitive agent integration tests
python -m pytest tests/integration/test_cognitive_agent_queries.py -v
```

---

### Q: Where is the cognitive agent code located?

A:
- `cortical/cognitive/graph.py` - CognitiveGraph, InMemoryStorage, Atom types
- `cortical/cognitive/text_bridge.py` - TextToAtomsBridge, BPETokenizer
- `cortical/cognitive/training.py` - IncrementalTrainer, TrainingManifest
- `cortical/cognitive/graph_storage.py` - ShardedGraphStorage
- `cortical/cognitive/__main__.py` - CLI commands

---

### Q: How do I check the model's current state?

A:
```bash
python -m cortical.cognitive status
```

This shows: documents trained, vocabulary size, last training time.

For more detail:
```python
trainer.manifest.total_documents
trainer.manifest.get_staleness()
len(agent.graph._storage._atoms)
```

---

### Q: What performance numbers should I expect?

A:
| Operation | Expected Time |
|-----------|---------------|
| get_associations() | <100ms |
| predict_next() | <50ms average |
| Save (no changes) | <1s |
| Save (incremental) | ~10s |
| Full save | ~54s (avoid) |
| Load model | ~10s |

---

### Q: Something broke. How do I debug?

A: Check in order:
1. Run smoke tests: `python -m pytest tests/smoke/ -v`
2. Check model status: `python -m cortical.cognitive status`
3. Validate GoT: `python -m cortical.got validate`
4. Look at recent commits: `git log --oneline -5`
5. Check session notes: `cat docs/sessions/*.md | head -100`

---

### Q: How do I know if my changes broke something?

A: Run the test suite:
```bash
# Minimum: smoke + IDF + integration
python -m pytest tests/smoke/ tests/behavioral/test_idf_weighted_links_spec.py tests/integration/test_cognitive_agent_queries.py -v
```

Integration tests verify:
- Associations return results
- Predictions work
- Performance contracts hold
- Dirty tracking functions
