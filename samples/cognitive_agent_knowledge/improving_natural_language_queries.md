# Improving Natural Language Queries: A Deep Analysis

*Session: 2026-01-12 | Topic: NL Query Enhancement Strategies*

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

#### A. Query Expansion (Already Exists!)

The `cortical/query/expansion.py` module provides multi-method expansion:

```python
# Current capability we're NOT using:
from cortical.query.expansion import expand_query_multihop, get_expanded_query_terms

# Multi-hop semantic inference
expanded = expand_query_multihop(
    processor,
    ["brain"],
    relation_types=["IsA", "HasA", "RelatedTo"],
    max_hops=2
)
# Could yield: brain → neural → cognitive → agent
```

**Integration Point**: NLQuery.gather_knowledge() should call `get_expanded_query_terms()` before looking up associations.

#### B. Semantic Retrofitting (Already Exists!)

The `cortical/semantics.py` module can align terms:

```python
from cortical.semantics import retrofit_connections, extract_pattern_relations

# Extract implicit relations from trained documents
relations = extract_pattern_relations(processor)
# Might find: "cognitive agent is a type of reasoning system"

# Retrofit connections to bring related terms closer
retrofit_connections(processor, relations, alpha=0.5)
```

**Integration Point**: Run retrofitting after training to improve association quality.

#### C. Spelling Correction via Edit Distance

We could add fuzzy matching:

```python
def find_closest_known_word(unknown_word: str, vocabulary: Set[str], max_distance: int = 2) -> List[str]:
    """Find vocabulary words within edit distance of unknown word."""
    candidates = []
    for known in vocabulary:
        if abs(len(known) - len(unknown_word)) <= max_distance:
            distance = levenshtein_distance(unknown_word, known)
            if distance <= max_distance:
                candidates.append((known, distance))
    return sorted(candidates, key=lambda x: x[1])
```

#### D. Intent Disambiguation

For ambiguous queries, we could return clarification:

```python
def suggest_clarifications(concepts: List[str], knowledge: GatheredKnowledge) -> List[str]:
    """When results are weak, suggest alternative interpretations."""
    if len(knowledge.associations) < 3:
        # Results are thin - maybe user meant something else
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
        # "risky files in reasoning/" becomes:
        # StructuredQuery(
        #     target="files",
        #     filters={"path": "reasoning/"},
        #     metric="risk_score",
        #     threshold=0.7
        # )
```

The PLN engine supports:
- **Multi-rule aggregation**: Combine evidence from multiple sources
- **Attention-based focus**: Prioritize relevant atoms (STI/LTI/VLTI weights)
- **Inference traces**: Explainable reasoning chains
- **WovenMind integration**: Discovery → PLN validation pipeline

### How to Integrate with NLQuery

```python
# In nl_query.py, add PLN-based inference:

from cortical.audits.reasoning import PLNReasoner, InferenceContext

class NLQuery:
    def __init__(self, agent, reasoner: Optional[PLNReasoner] = None):
        self.agent = agent
        self.reasoner = reasoner or PLNReasoner()

    def gather_knowledge_with_inference(self, intent: QueryIntent) -> GatheredKnowledge:
        """Gather knowledge using both graph lookup AND probabilistic inference."""
        knowledge = self.gather_knowledge(intent)

        # Add PLN-inferred facts
        for concept in intent.concepts:
            # Create inference context
            ctx = InferenceContext(
                focus_atoms=[concept],
                max_depth=3,
                confidence_threshold=0.6
            )

            # Run inference
            inferred = self.reasoner.infer(ctx)

            # Add high-confidence inferences to knowledge
            for fact in inferred:
                if fact.confidence > 0.7:
                    knowledge.associations.append(fact.conclusion)

        return knowledge
```

### Learnable Rules from Patterns

The audit algorithms (`cortical/audits/algorithms/`) provide:

| Algorithm | Use Case for NLQuery |
|-----------|---------------------|
| Naive Bayes | Classify question intent with learned priors |
| Decision Tree | Rule-based query routing |
| Markov Chain | Predict likely next concepts in a query |
| LSH | Fast approximate matching for fuzzy queries |

**Example: Learning Query Patterns**

```python
from cortical.audits.algorithms.naive_bayes import NaiveBayesClassifier

# Train on historical queries
classifier = NaiveBayesClassifier()
classifier.train([
    ("what is the cognitive agent", "identity"),
    ("how does training work", "mechanism"),
    ("where is storage defined", "location"),
    # ... more examples
])

# Classify new queries
intent_type = classifier.predict("what's the brain memory")
# Returns: "identity" (with confidence)
```

---

## 3. Using Our Statistical Model for Better Natural Language Generation

### Current Limitation

The `ask` function returns templated responses:

```python
# Current approach (rigid templates):
return f"The {subject_title} is a component that works with {tech_str}."
```

This produces grammatically correct but mechanical text.

### What We Have for Generation

#### PRISM-SLM (Synaptic Language Model)

```python
# cortical/reasoning/prism_slm.py
from cortical.reasoning.prism_slm import PRISMLanguageModel

slm = PRISMLanguageModel()
slm.train_on_corpus(documents)

# Generate text given a seed
generated = slm.generate(
    seed="The cognitive agent",
    max_tokens=50,
    temperature=0.7
)
# Could yield: "The cognitive agent maintains semantic associations
#              through hebbian learning and graph-based storage..."
```

#### N-gram Model for Fluency

```python
# cortical/spark/ngram.py
from cortical.spark.ngram import NGramModel

ngram = NGramModel(order=3)  # Trigram
ngram.train(corpus_text)

# Complete a partial response
completion = ngram.generate(
    context="The cognitive agent is",
    max_tokens=20
)
```

### Proposed Integration: Hybrid Response Generation

```python
def generate_response_with_model(
    self,
    intent: QueryIntent,
    knowledge: GatheredKnowledge,
    language_model: PRISMLanguageModel
) -> str:
    """Generate natural response using statistical language model."""

    # 1. Build seed from knowledge
    subject = self._extract_subject(intent)
    key_terms = knowledge.associations[:5]
    seed = f"The {subject} "

    # 2. Prime the model with relevant context
    context_text = " ".join([
        f"{subject} relates to {term}"
        for term in key_terms
    ])
    language_model.prime(context_text)

    # 3. Generate fluent continuation
    generated = language_model.generate(
        seed=seed,
        max_tokens=100,
        temperature=0.8,
        stop_tokens=[".", "?", "!"]
    )

    # 4. Validate generated text contains our knowledge
    if not any(term in generated.lower() for term in key_terms[:3]):
        # Fallback to template if generation missed key facts
        return self._generate_template_response(intent, knowledge)

    return generated
```

### Benefits of Model-Generated Responses

| Aspect | Template | Model-Generated |
|--------|----------|-----------------|
| Naturalness | Mechanical | Fluent |
| Variety | Repetitive | Diverse |
| Accuracy | Guaranteed | Needs validation |
| Speed | Fast | Slower |

**Recommendation**: Use model generation with template fallback for validation.

---

## 4. Reducing Reliance on Grep: Using Our Framework

### The Current Reality

When exploring code, we often reach for Grep:

```bash
grep -r "class CognitiveAgent" cortical/
```

But our framework already provides semantic search capabilities.

### What Our Framework Offers Instead

#### Semantic Code Search

```python
# Instead of grep, use trained associations:
agent = CognitiveAgent.load("model/")

# Find code related to "cognitive agent"
results = agent.query("code_for_word", "cognitiveagent")
# Returns: Atom objects with file_path, lineno, context

# Find callers
callers = agent.query("callers_of", "CognitiveAgent")

# Find by semantic similarity (not just text match)
similar = agent.get_associations("storage", top_k=20)
# May find: persistence, disk, file, cache, save, load
```

#### Multi-hop Semantic Traversal

```python
from cortical.query.expansion import expand_query_multihop

# Find concepts 2 hops away from "agent"
related = expand_query_multihop(
    processor,
    ["agent"],
    relation_types=["IsA", "UsedBy", "Contains"],
    max_hops=2
)
# Might find: agent → cognitive → memory → storage → file
```

### New Problem Angles This Creates

#### Challenge 1: Cold Start

Grep works immediately. Our framework requires training.

**Solution**: Incremental training with manifest tracking (already implemented).

```python
# Quick bootstrap for new codebase
trainer.train_directory("cortical/", pattern="*.py")
# ~45 seconds for 200 files, then queries work
```

#### Challenge 2: Vocabulary Mismatch

Grep finds exact text. Our model finds trained vocabulary.

**Solution**: Hybrid approach - try semantic first, fall back to text search.

```python
def smart_search(query: str, agent: CognitiveAgent, codebase_path: str) -> List[Result]:
    """Semantic-first search with text fallback."""

    # Try semantic search
    semantic_results = agent.query("code_for_word", query)

    if semantic_results:
        return semantic_results

    # Fallback: Use our Trie for prefix matching
    from cortical.audits.algorithms.trie import CommentMarkerTrie
    trie = CommentMarkerTrie()
    # ... populate and search

    # Last resort: shell out to grep (but log it for training)
    log_missed_query(query)  # So we can improve
    return grep_fallback(query, codebase_path)
```

#### Challenge 3: Freshness

Grep always sees current files. Our model sees trained state.

**Solution**: Staleness tracking (already implemented).

```python
staleness = trainer.manifest.get_staleness()
if staleness > 0.1:  # >10% new content
    print(f"Warning: Model is {staleness*100:.1f}% stale. Consider retraining.")
```

### Does This Give Pause for Imagination?

Yes. Consider these possibilities:

1. **Self-Improving Search**: Log queries that fall back to grep, automatically retrain on those patterns.

2. **Query Understanding**: Instead of searching for text, understand what the user wants:
   ```
   User: "find the storage bug"
   System: Understands "storage" + "bug" → searches for:
           - Files with "storage" in associations
           - Recent commits mentioning "fix" near storage code
           - Error handling patterns in storage modules
   ```

3. **Conversational Refinement**:
   ```
   User: "where's the agent?"
   System: "I found 3 agents: CognitiveAgent, TaskAgent, AuditAgent.
            Which are you looking for?"
   ```

4. **Predictive Assistance**: Based on recent queries, predict what the user might need next.

---

## 5. Other Functionality That May Help the Ask Function

### Already Available in Our Codebase

#### 1. Query Expansion Module (`cortical/query/expansion.py`)

```python
from cortical.query.expansion import (
    expand_query,           # Basic lateral expansion
    expand_query_semantic,  # Relation-based expansion
    expand_query_multihop,  # Multi-hop inference
    get_expanded_query_terms  # Consolidated expansion
)

# Use in NLQuery to expand concepts before lookup
expanded_concepts = get_expanded_query_terms(
    processor,
    intent.concepts,
    methods=["lateral", "clusters", "code_concepts"],
    max_terms=20
)
```

#### 2. Intent Parser (`cortical/spark/intent_parser.py`)

```python
from cortical.spark.intent_parser import IntentParser

parser = IntentParser()
parsed = parser.parse("fix the storage bug in cognitive module")
# Returns:
#   action: "fix"
#   entities: ["storage", "bug", "cognitive", "module"]
#   priority: "medium"
#   confidence: 0.85
```

**Integration**: Use IntentParser to better understand what the user wants to DO, not just what they're asking about.

#### 3. Spark Predictor (`cortical/spark/predictor.py`)

```python
from cortical.spark.predictor import SparkPredictor

predictor = SparkPredictor(ngram_model, alignment_index)
predictor.prime("cognitive agent")

# Suggest completions for partial queries
completions = predictor.complete("what is the cog")
# Returns: ["cognitive agent", "cognitive graph", "cognitive loop"]
```

**Integration**: Add auto-complete to the ask command for query assistance.

#### 4. Embeddings for Similarity (`cortical/embeddings.py`)

```python
from cortical.embeddings import (
    compute_graph_embeddings,
    find_similar_by_embedding
)

# Compute embeddings using TF-IDF method (best for semantic similarity)
embeddings = compute_graph_embeddings(processor, method="tfidf")

# Find terms similar to query even if not directly connected
similar = find_similar_by_embedding(embeddings, "agent", top_n=10)
```

**Integration**: When direct lookup fails, use embedding similarity as fallback.

#### 5. Semantic Relations (`cortical/semantics.py`)

```python
from cortical.semantics import (
    extract_pattern_relations,  # Extract IsA, HasA, etc.
    build_isa_hierarchy,        # Build type hierarchy
    inherit_properties          # Property inheritance
)

# Build semantic hierarchy from trained corpus
relations = extract_pattern_relations(processor)
hierarchy = build_isa_hierarchy(relations)

# Use hierarchy for inference
# "CognitiveAgent IsA Agent" → inherit Agent properties
```

**Integration**: Use semantic hierarchy to answer "What kind of X is Y?" questions.

#### 6. Analogy Engine (`cortical/query/analogy.py`)

```python
from cortical.query.analogy import find_analogies

# "cognitive is to agent as neural is to ?"
analogies = find_analogies(
    processor,
    a="cognitive", b="agent",
    c="neural"
)
# Might return: "network", "model", "system"
```

**Integration**: Support analogy-based queries in ask function.

#### 7. Cognitive Loop (`cortical/reasoning/cognitive_loop.py`)

```python
from cortical.reasoning.cognitive_loop import CognitiveLoop, Phase

loop = CognitiveLoop()

# QAPV cycle for complex queries
loop.question("What is the cognitive agent?")
answer = loop.answer()      # Gather initial knowledge
product = loop.produce()    # Synthesize response
verified = loop.verify()    # Check for consistency
```

**Integration**: Use QAPV cycle for complex, multi-part questions.

---

## Proposed Integration Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Enhanced NLQuery Pipeline                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. PARSE                                                           │
│     ├─ IntentParser.parse() → action, entities, priority            │
│     ├─ NLQuery.parse_intent() → question_type, concepts             │
│     └─ SparkPredictor.complete() → query suggestions                │
│                                                                      │
│  2. EXPAND                                                          │
│     ├─ get_expanded_query_terms() → lateral, semantic, code         │
│     ├─ find_similar_by_embedding() → embedding-based matches        │
│     └─ fuzzy_match() → spelling correction                          │
│                                                                      │
│  3. GATHER                                                          │
│     ├─ agent.query() → direct graph lookup                          │
│     ├─ PLNReasoner.infer() → probabilistic inference                │
│     ├─ expand_query_multihop() → relation chain traversal           │
│     └─ build_isa_hierarchy() → type-based inheritance               │
│                                                                      │
│  4. GENERATE                                                        │
│     ├─ PRISMLanguageModel.generate() → fluent response              │
│     ├─ template_fallback() → guaranteed accuracy                    │
│     └─ CognitiveLoop.verify() → consistency check                   │
│                                                                      │
│  5. REFINE                                                          │
│     ├─ suggest_clarifications() → "Did you mean...?"                │
│     └─ log_for_training() → improve future queries                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Summary: Key Improvements to Implement

| Priority | Improvement | Existing Module | Effort |
|----------|-------------|-----------------|--------|
| High | Query expansion before lookup | `cortical/query/expansion.py` | Low |
| High | Embedding fallback for unknown terms | `cortical/embeddings.py` | Medium |
| Medium | Intent parser integration | `cortical/spark/intent_parser.py` | Low |
| Medium | PLN inference for multi-hop reasoning | `cortical/audits/reasoning.py` | Medium |
| Medium | Language model response generation | `cortical/reasoning/prism_slm.py` | Medium |
| Low | Fuzzy matching for typos | New implementation | Medium |
| Low | Query auto-complete | `cortical/spark/predictor.py` | Low |

The foundation already exists. The opportunity is integration.

---

## Questions for Future Sessions

1. Should we implement a unified `EnhancedNLQuery` class that composes all these capabilities?
2. What's the acceptable latency budget for query response? (Currently ~50ms, could increase with inference)
3. Should we add a "confidence" score to responses so users know when to trust vs verify?
4. How do we handle the cold-start problem for new codebases gracefully?

---

*This document serves as both analysis and roadmap for improving natural language query capabilities in the Cognitive Agent system.*
