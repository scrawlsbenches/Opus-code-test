# Improving Natural Language Queries: A Deep Analysis

*Session: 2026-01-12 | Topic: NL Query Enhancement Strategies*

---

## CRITICAL CAVEATS AND ASSUMPTIONS

> **READ THIS FIRST**: This document contains proposals that need hypothesis testing before implementation. Many suggestions assume runtime integration, but some should be applied during TRAINING instead. The CognitiveAgent graph is fast to query (~1-20ms) but slow to build (~45s for 200 files). Design decisions should preserve query speed.

### Key Assumptions to Verify

1. **CognitiveAgent already has atom-based prediction** via `predict_next()` using FOLLOWS links. Before integrating PRISM-SLM or NGram models, verify if CognitiveAgent's existing prediction is sufficient.

2. **cortical/query/expansion.py operates on CorticalTextProcessor**, not CognitiveAgent. Integration requires either:
   - Training expansion results INTO the CognitiveAgent graph (preferred)
   - Runtime bridge between processor and agent (adds latency)

3. **Model persistence is currently gitignored** (`models/cognitive_agent/` in .gitignore line 190). This breaks cold-start. **FIX REQUIRED**: Remove from .gitignore or provide bootstrap script.

4. **Some proposed modules may be redundant** with CognitiveAgent's existing capabilities. Audit before adding complexity.

---

## 1. What Happens When We Don't Use the Correct Word?

When a user queries with an inexact term, the current NLQuery pipeline faces several challenges:

### The Problem

```
User asks: "What is the brain memory?"
Intended: "What is the cognitive agent?"

Current behavior:
1. Extract concepts: ["brainmemory", "brain_memory", "brain", "memory"]
2. Query for associations of "brain" → may return neuroscience terms
3. Query for associations of "memory" → may return RAM, cache, storage
4. Neither captures the intended "cognitive agent" concept
```

The system fails gracefully (returns what it finds) but doesn't bridge the semantic gap.

### Why This Happens

1. **Literal Matching**: We look up exact tokens in the graph
2. **No Fuzzy Resolution**: Misspellings aren't corrected
3. **Missing Synonymy**: "brain" doesn't automatically expand to "cognitive"
4. **No Intent Disambiguation**: We don't ask "did you mean...?"

### How to Improve Natural Language Understanding

#### A. Query Expansion

> **CAVEAT**: `cortical/query/expansion.py` operates on CorticalTextProcessor, NOT CognitiveAgent. Direct integration would require bridging two different graph systems.

> **RECOMMENDATION**: Instead of runtime expansion, consider TRAINING the CognitiveAgent on expansion relationships. When we see "brain" in documents near "cognitive", the SIMILARITY links already capture this. The question is: are we training on the right documents?

**Training-Time Solution** (Preferred):
```python
# Train on documents that establish synonymy:
# "The cognitive agent (also called brain memory or neural system)..."
# This creates SIMILARITY links: brain ↔ cognitive automatically
```

**Runtime Solution** (If needed, adds latency):
```python
# Would need to verify this works with CognitiveAgent
from cortical.query.expansion import expand_query_multihop

expanded = expand_query_multihop(
    processor,  # NOTE: This is CorticalTextProcessor, not CognitiveAgent
    ["brain"],
    relation_types=["IsA", "HasA", "RelatedTo"],
    max_hops=2
)
```

#### B. Semantic Retrofitting

> **CAVEAT**: `cortical/semantics.py` also operates on CorticalTextProcessor. Same integration challenge.

> **RECOMMENDATION**: Apply retrofitting DURING training pipeline, not at query time. This improves the stored associations rather than computing them on every query.

```python
# Apply DURING training (one-time cost):
from cortical.semantics import retrofit_connections, extract_pattern_relations

relations = extract_pattern_relations(processor)
retrofit_connections(processor, relations, alpha=0.5)
# Then persist the improved connections
```

#### C. Spelling Correction via Edit Distance

> **PRIORITY: Low** - Only implement if user testing shows frequent typos.

```python
def find_closest_known_word(unknown_word: str, vocabulary: Set[str], max_distance: int = 2) -> List[str]:
    """Find vocabulary words within edit distance of unknown word."""
    # O(n) scan of vocabulary - acceptable if vocabulary < 50k terms
    candidates = []
    for known in vocabulary:
        if abs(len(known) - len(unknown_word)) <= max_distance:
            distance = levenshtein_distance(unknown_word, known)
            if distance <= max_distance:
                candidates.append((known, distance))
    return sorted(candidates, key=lambda x: x[1])
```

#### D. Intent Disambiguation

> **PRIORITY: Medium** - Good UX improvement, low implementation risk.

```python
def suggest_clarifications(concepts: List[str], knowledge: GatheredKnowledge) -> List[str]:
    """When results are weak, suggest alternative interpretations."""
    if len(knowledge.associations) < 3:
        return [
            f"Did you mean '{expand_concept(c)}'?"
            for c in concepts
        ]
    return []
```

---

## 2. Integrating Audit Functionality with Probabilistic Logic Rules

### What We Already Have

The `cortical/audits/reasoning.py` module implements **Probabilistic Logic Networks (PLN)**:

```python
class AuditQuery:
    """Translates natural language to structured audit queries."""

    def translate_audit_query(self, nl_query: str) -> StructuredQuery:
        # "risky files in reasoning/" becomes structured query
```

> **CAVEAT**: PLN reasoning is designed for audit use cases (risk scoring, code quality). Integration with NLQuery requires verifying the inference rules are applicable to general knowledge queries.

> **HYPOTHESIS TO TEST**: Can PLN inference chains improve "what is X?" answers, or is it overkill for simple queries?

### Integration Consideration

> **PRIORITY: Low for initial implementation** - PLN adds complexity. Start with simpler improvements first.

The audit algorithms (`cortical/audits/algorithms/`) provide:

| Algorithm | NLQuery Use Case | Integration Complexity |
|-----------|------------------|------------------------|
| Naive Bayes | Classify question intent | Low - stateless classifier |
| Decision Tree | Rule-based query routing | Low - can train on query logs |
| Markov Chain | Predict next concepts | **Redundant with predict_next()** |
| LSH | Fast fuzzy matching | Medium - needs vocabulary index |

> **NOTE**: Markov Chain is redundant with CognitiveAgent.predict_next() which already uses FOLLOWS links for sequential prediction. **Do not integrate** - use existing capability.

---

## 3. Using Our Statistical Model for Better Natural Language Generation

### Current State: CognitiveAgent vs PRISM-SLM

> **CRITICAL COMPARISON**:

| Feature | CognitiveAgent.predict_next() | PRISM-SLM.generate() |
|---------|-------------------------------|----------------------|
| Mechanism | FOLLOWS links (atom → atom) | SynapticTransition (token → token) |
| Training | Trained with IDF weighting | Separate Hebbian training |
| Storage | Part of unified graph | Separate TransitionGraph |
| Query Speed | ~1-20ms (indexed) | Unknown - needs benchmark |
| Already Integrated | Yes | No |

> **QUESTION TO ANSWER**: Does PRISM-SLM offer something CognitiveAgent's predict_next() doesn't?
>
> - **Hebbian decay**: PRISM-SLM has synaptic decay for unused transitions. CognitiveAgent has IDF weighting but no decay.
> - **Context window**: PRISM-SLM maintains sliding context. CognitiveAgent is single-word prediction.
>
> **RECOMMENDATION**: Before integrating PRISM-SLM, test if chaining CognitiveAgent.predict_next() calls achieves similar quality. If so, don't add complexity.

### Existing Generation via CognitiveAgent

The CLI already has a `generate` command that chains predict_next():

```bash
python -m cortical.cognitive generate --prompt "The cognitive agent" --max-tokens 20
```

> **TEST THIS FIRST** before adding PRISM-SLM integration.

### If PRISM-SLM Integration Is Needed

> **CAVEAT**: PRISM-SLM would need separate training. This adds to build time and storage.

```python
# Only if CognitiveAgent generation is insufficient:
def generate_response_with_model(
    self,
    intent: QueryIntent,
    knowledge: GatheredKnowledge,
    language_model: PRISMLanguageModel  # Requires separate training!
) -> str:
    # ... generation logic ...
```

### New Training Angles to Consider

> **QUESTION**: What new training approaches does our current architecture enable?

1. **Train on query-response pairs**: Log successful queries and their responses, train to predict good responses from concepts.

2. **Train on definition patterns**: "X is a Y that does Z" patterns could strengthen FOLLOWS links for definition generation.

3. **Fine-tune on results**: After generating responses, rate them (manual or automated), retrain on high-rated outputs.

4. **Self-referential training**: Train the model on its own documentation (like this file) to improve self-description.

---

## 4. Reducing Reliance on Grep: Using Our Framework

### CognitiveAgent as Trained Search

> **KEY INSIGHT**: CognitiveAgent IS a form of trained search. The question isn't "use CognitiveAgent instead of Grep" but "train CognitiveAgent to return what we need."

### Proposed: CognitiveAgent.semantic_search()

> **INSTEAD OF**: Wrapping grep, train on grep-like queries and their expected results.

```python
# Concept: Train on search patterns
# Input: "storage class definition"
# Expected: files containing storage classes

# Training document (create these):
"""
When searching for 'storage class definition', the relevant files are:
- cortical/cdg/storage.py (StorageBackend class)
- cortical/cognitive/graph_storage.py (ShardedStorage class)
- cortical/got/versioned_store.py (VersionedStore class)
"""

# After training, agent.get_associations("storage class definition")
# should return these files as associations
```

### Storing Grep Results in Graph

> **HYPOTHESIS**: If we store grep results as REFERS_TO links, future queries could bypass grep entirely.

```python
# When grep finds "class StorageBackend" in storage.py:
# Create: WORD("storagebackend") --REFERS_TO--> FILE("cortical/cdg/storage.py")
#
# Future query for "storage backend" finds the file through associations
# without running grep
```

> **BENEFIT**: Grep results become training data. The model learns what grep would return.
>
> **RISK**: Stale results if files change. Need staleness tracking (already have this).

### Cold-Start Problem: CRITICAL FIX NEEDED

> **CURRENT STATE**: `models/cognitive_agent/` is in `.gitignore` (line 190).
> This means the trained model is NOT committed, causing cold-start on every fresh clone.

**Options**:
1. **Remove from .gitignore** - Commit the model (increases repo size by ~150MB+)
2. **Bootstrap script** - Run training on first use, cache results
3. **Smaller committed model** - Commit only essential vocabulary, rebuild links on demand

> **RECOMMENDATION**: Option 2 (bootstrap script) balances repo size with usability:
```bash
# scripts/bootstrap_cognitive.sh
if [ ! -d "models/cognitive_agent" ]; then
    echo "Building cognitive model (first run only)..."
    python -m cortical.cognitive train cortical/ --pattern "*.py"
fi
```

---

## 5. Other Functionality That May Help the Ask Function

### Inventory with Integration Assessment

> **LEGEND**:
> - **Use**: Already applicable, low integration effort
> - **Train**: Apply during training, not runtime
> - **Redundant**: CognitiveAgent already has equivalent
> - **Complex**: High effort, uncertain benefit

| Module | Capability | Assessment | Notes |
|--------|------------|------------|-------|
| `query/expansion.py` | Query expansion | **Train** | Apply during training, not runtime |
| `spark/intent_parser.py` | Action/entity extraction | **Use** | Stateless, can integrate directly |
| `spark/predictor.py` | Query completion | **Redundant** | CognitiveAgent.predict_next() does this |
| `spark/ngram.py` | N-gram prediction | **Redundant** | predict_next() with FOLLOWS links |
| `embeddings.py` | Similarity by embedding | **Train** | Use to improve training, not runtime |
| `semantics.py` | Semantic relations | **Train** | Retrofit during training |
| `query/analogy.py` | Analogy reasoning | **Complex** | Novel capability but uncertain value |
| `reasoning/cognitive_loop.py` | QAPV cycle | **Complex** | Overkill for simple queries |
| `audits/reasoning.py` | PLN inference | **Complex** | Designed for audits, not NL queries |

### Should We Integrate or Retire?

> **OBSERVATION**: We have multiple prediction/generation systems:
> - CognitiveAgent.predict_next() (FOLLOWS links)
> - PRISM-SLM (synaptic transitions)
> - NGramModel (n-gram statistics)
> - SparkPredictor (ngram + alignment)

> **RECOMMENDATION**: **Consolidate around CognitiveAgent**. The others exist for historical reasons or specific use cases. For NLQuery, use CognitiveAgent exclusively unless testing shows it's insufficient.

---

## 6. First-Pass Generation for Training

> **QUESTION**: Do we need a first-pass text generator to create training data for better generation?

**Yes, this is valuable.** The idea:

1. Generate responses using current (imperfect) templates
2. Human reviews and corrects/approves responses
3. Train on the corrected responses
4. Model learns to generate better responses directly

```python
# First-pass generation pipeline
def generate_training_pairs():
    questions = load_sample_questions()
    for q in questions:
        response = nl_query.ask(q)  # Current imperfect response
        yield {
            "question": q,
            "draft_response": response,
            "human_corrected": None,  # To be filled by human
            "approved": False
        }

# After human review, train on approved pairs
```

---

## Prioritized Roadmap with Justifications

### Priority 1: CRITICAL (Do Immediately)

| Task | Justification | Effort |
|------|---------------|--------|
| Fix cold-start (bootstrap script) | Model currently unusable on fresh clone | Low |
| Verify CognitiveAgent.generate quality | Avoid adding PRISM-SLM if unnecessary | Low |
| Remove redundant Markov/NGram from integration plan | CognitiveAgent.predict_next() already does this | None |

### Priority 2: HIGH (Next Sprint)

| Task | Justification | Effort |
|------|---------------|--------|
| Integrate IntentParser | Stateless, improves question understanding | Low |
| Create training documents for self-description | "Teach" the model about itself | Low |
| Store grep results as training data | Self-improving search | Medium |

### Priority 3: MEDIUM (After Validation)

| Task | Justification | Effort |
|------|---------------|--------|
| Apply semantic retrofitting during training | Improves association quality at build time | Medium |
| First-pass generation for training pipeline | Creates feedback loop for improvement | Medium |
| Add intent disambiguation | Better UX for ambiguous queries | Medium |

### Priority 4: LOW (Needs Hypothesis Testing)

| Task | Justification | Effort |
|------|---------------|--------|
| PLN integration | May be overkill for simple queries | High |
| PRISM-SLM integration | Only if CognitiveAgent.generate insufficient | High |
| Fuzzy spelling correction | Only if user testing shows need | Medium |

---

## Latency Budget Considerations

### Current State
- CognitiveAgent.get_associations(): ~1-20ms
- CognitiveAgent.predict_next(): ~1-20ms
- Full ask() pipeline: ~50-100ms

### Proposed Tiered Response Strategy

| Query Type | Budget | Approach |
|------------|--------|----------|
| Quick lookup | <100ms | Direct association lookup |
| Standard query | <500ms | Expansion + inference |
| Deep analysis | <5s | Multi-hop reasoning, PLN |
| Comprehensive | <60s | Full corpus scan, report generation |

> **IMPLEMENTATION**: Add `--depth` flag to ask command:
> ```bash
> python -m cortical.cognitive ask "What is X?" --depth quick   # <100ms
> python -m cortical.cognitive ask "What is X?" --depth deep    # <5s
> ```

---

## Questions Resolved

1. **SparkSLM vs CognitiveAgent.predict_next()?** → Use predict_next() first. SparkSLM only if predict_next() chaining is insufficient.

2. **New training angles?** → Self-referential training, query-response pairs, grep result capture.

3. **Unified training pipeline?** → Yes, see Priority 2 tasks. Store everything in CognitiveAgent graph.

4. **Too many prediction models?** → Yes. Consolidate around CognitiveAgent. Others are redundant for NLQuery.

5. **First-pass generation?** → Yes, valuable for creating training feedback loop.

6. **CognitiveAgent.grep?** → Better: train on grep results so semantic search returns what grep would.

7. **Cold-start?** → **BROKEN**. Model in .gitignore. Need bootstrap script.

---

*This document serves as analysis, roadmap, and caveat guide for improving natural language query capabilities. Verify assumptions through testing before implementing.*
