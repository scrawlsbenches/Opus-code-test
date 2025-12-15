# Usage Patterns Guide

Advanced usage patterns for the Cortical Text Processor, focusing on code-aware search, semantic fingerprinting, and intent-based querying.

---

## Table of Contents

1. [Code Search Patterns](#code-search-patterns)
2. [Fingerprint Comparison](#fingerprint-comparison)
3. [Intent-Based Querying](#intent-based-querying)
4. [Document Type Boosting](#document-type-boosting)
5. [Configuration Patterns](#configuration-patterns)

---

## Code Search Patterns

### Pattern 1: Code-Aware Tokenization

Enable identifier splitting to search for code patterns:

```python
from cortical import CorticalTextProcessor, Tokenizer

# Create processor with code-aware tokenizer
tokenizer = Tokenizer(split_identifiers=True)
processor = CorticalTextProcessor()

# Process code with identifier splitting
processor.process_document(
    "auth.py",
    """
    def getUserCredentials(userId):
        return fetchUserFromDB(userId).credentials
    """
)

# Now searches for "user" will find "getUserCredentials"
# because it was split into ["get", "user", "credentials"]
```

**What identifier splitting does:**
- `getUserCredentials` → `["getusercredentials", "get", "user", "credentials"]`
- `fetch_user_from_db` → `["fetch", "user", "from", "db"]`
- `HTTPResponseCode` → `["httpresponsecode", "http", "response", "code"]`

---

### Pattern 2: Programming Concept Expansion

Expand queries with programming synonyms:

```python
# Search with code concept expansion
results = processor.find_documents_for_query(
    "fetch user data",
    use_code_concepts=True
)

# Or use dedicated method
results = processor.expand_query_for_code("fetch user data")
# Expands "fetch" to include: get, retrieve, load, read, query
# Expands "user" to include: account, profile, member
```

**Built-in concept groups:**

| Concept Group | Terms |
|--------------|-------|
| retrieval | get, fetch, retrieve, load, read, query |
| storage | save, store, write, persist, cache, put |
| deletion | delete, remove, drop, clear, purge, erase |
| auth | auth, authenticate, authorize, login, signin |
| error | error, exception, fail, invalid, wrong |
| validation | validate, check, verify, assert, ensure |
| transform | convert, transform, parse, serialize, encode |
| async | async, await, promise, future, callback |

---

### Pattern 3: Intent-Based Code Search

Search by developer intent rather than exact keywords:

```python
# Parse natural language query into structured intent
parsed = processor.parse_intent_query("where do we handle authentication?")
print(parsed)
# {
#   'intent': 'location',        # What type of answer expected
#   'action': 'handle',          # The action verb
#   'subject': 'authentication', # What the action operates on
#   'question_word': 'where',
#   'expanded_terms': ['handle', 'authentication', 'auth', 'login', ...]
# }

# Search with intent understanding
results = processor.search_by_intent("how do we validate user input?")
# Returns documents relevant to validation, input checking, assertions
```

**Supported intent types:**

| Question Word | Intent Type | Typical Results |
|--------------|-------------|-----------------|
| where | location | File paths, module locations |
| how | implementation | Code implementation details |
| what | definition | Type definitions, interfaces |
| why | rationale | Comments, design decisions |
| when | lifecycle | Initialization, cleanup code |

---

### Pattern 4: Combined Code Search

Combine multiple code search features:

```python
# Full code search with all features
results = processor.find_documents_for_query(
    "authentication handler",
    use_expansion=True,           # Lateral connection expansion
    use_code_concepts=True,       # Programming synonyms
    use_semantic=True             # Semantic relations
)

# Or use intent search for natural language
results = processor.search_by_intent(
    "where is the password validation logic?"
)
```

---

## Fingerprint Comparison

Semantic fingerprinting enables comparing the meaning of code blocks without embedding models.

### Pattern 5: Basic Fingerprinting

```python
# Get semantic fingerprint of a code block
code1 = """
def authenticate_user(username, password):
    user = database.find_user(username)
    if user and verify_password(password, user.hash):
        return create_session(user)
    return None
"""

code2 = """
def login(name, pwd):
    account = db.get_account(name)
    if account and check_pwd(pwd, account.password_hash):
        return generate_token(account)
    return None
"""

fp1 = processor.get_fingerprint(code1)
fp2 = processor.get_fingerprint(code2)

# Compare fingerprints
comparison = processor.compare_fingerprints(fp1, fp2)
print(f"Similarity: {comparison['similarity']:.2%}")
# Output: Similarity: 78.5%
```

**Fingerprint contents:**
```python
{
    'terms': {'user': 0.8, 'password': 0.7, 'authenticate': 0.6, ...},
    'concepts': ['L2_authentication', 'L2_database_operations'],
    'bigrams': ['find user', 'verify password', ...],
    'top_terms': [('user', 0.8), ('password', 0.7), ...]
}
```

---

### Pattern 6: Explain Similarity

Get human-readable explanation of why two code blocks are similar:

```python
explanation = processor.explain_similarity(fp1, fp2)
print(explanation)
# Output:
# Shared terms: user (0.8), password (0.7), database (0.5)
# Shared concepts: authentication (2), database_operations (1)
# Both use patterns: user lookup, password verification

# Or explain a single fingerprint
fp_explanation = processor.explain_fingerprint(fp1)
print(fp_explanation)
# Output:
# Main concepts: authentication, database access
# Key terms: user (0.8), password (0.7), authenticate (0.6)
# Notable bigrams: verify password, create session
```

---

### Pattern 7: Find Similar Code Blocks

Search for similar code across the corpus:

```python
# Find code blocks similar to a reference
target_code = """
def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None
"""

similar = processor.find_similar_texts(
    target_code,
    top_n=5,
    min_similarity=0.3
)

for doc_id, similarity in similar:
    print(f"{doc_id}: {similarity:.2%} similar")
```

---

### Pattern 8: Code Deduplication

Use fingerprints to detect duplicate or near-duplicate code:

```python
def find_duplicates(processor, min_similarity=0.85):
    """Find potentially duplicate code blocks."""
    docs = processor.documents
    fingerprints = {
        doc_id: processor.get_fingerprint(content)
        for doc_id, content in docs.items()
    }

    duplicates = []
    doc_ids = list(fingerprints.keys())

    for i, doc_id1 in enumerate(doc_ids):
        for doc_id2 in doc_ids[i+1:]:
            result = processor.compare_fingerprints(
                fingerprints[doc_id1],
                fingerprints[doc_id2]
            )
            if result['similarity'] >= min_similarity:
                duplicates.append((doc_id1, doc_id2, result['similarity']))

    return sorted(duplicates, key=lambda x: -x[2])

# Find duplicates
dupes = find_duplicates(processor, min_similarity=0.9)
for doc1, doc2, sim in dupes:
    print(f"Possible duplicate: {doc1} <-> {doc2} ({sim:.1%})")
```

---

## Intent-Based Querying

### Pattern 9: Query Intent Detection

Let the system auto-detect query intent:

```python
# The system detects query type
queries = [
    "what is PageRank",           # Conceptual (wants definition)
    "where is PageRank computed", # Implementation (wants location)
    "how does PageRank work",     # Implementation (wants details)
]

for query in queries:
    is_conceptual = processor.is_conceptual_query(query)
    query_type = "conceptual" if is_conceptual else "implementation"
    print(f"{query} -> {query_type}")
```

---

### Pattern 10: Intent-Aware Search

Search with intent understanding:

```python
# Conceptual query: boost documentation
results = processor.search_by_intent("what is the 4-layer architecture?")
# Will prefer: docs/architecture.md over cortical/processor/__init__.py

# Implementation query: boost code
results = processor.search_by_intent("where is PageRank computed?")
# Will prefer: cortical/analysis.py over docs/algorithms.md
```

---

## Document Type Boosting

### Pattern 11: Boost Documentation

When searching for concepts, boost documentation files:

```python
# Auto-detect intent and boost accordingly
results = processor.find_documents_with_boost(
    "PageRank algorithm",
    auto_detect_intent=True,  # Auto-boost docs for conceptual queries
    top_n=5
)

# Always prefer documentation
results = processor.find_documents_with_boost(
    "PageRank algorithm",
    prefer_docs=True,         # Always boost docs
    top_n=5
)

# Custom boost factors
results = processor.find_documents_with_boost(
    "PageRank algorithm",
    custom_boosts={
        'docs': 2.0,    # Double weight for docs/ folder
        'root_docs': 1.5,  # 1.5x for root-level .md
        'code': 1.0,    # Normal weight for code
        'test': 0.5     # Half weight for tests
    }
)
```

---

### Pattern 12: Search with Type Filtering

Limit search to specific document types:

```python
# Search only in documentation
doc_results = [
    (doc_id, score)
    for doc_id, score in processor.find_documents_for_query("PageRank")
    if doc_id.endswith('.md') or doc_id.startswith('docs/')
]

# Search only in code
code_results = [
    (doc_id, score)
    for doc_id, score in processor.find_documents_for_query("PageRank")
    if doc_id.endswith('.py') and not doc_id.startswith('tests/')
]
```

---

## Configuration Patterns

### Pattern 13: Custom Configuration

Use custom configuration for specific use cases:

```python
from cortical import CorticalTextProcessor, CorticalConfig

# High-precision configuration (less expansion, stricter clustering)
precision_config = CorticalConfig(
    max_query_expansions=5,
    cluster_strictness=1.0,
    pagerank_damping=0.85
)

# High-recall configuration (more expansion, looser clustering)
recall_config = CorticalConfig(
    max_query_expansions=20,
    cluster_strictness=0.5,
    semantic_expansion_discount=0.8
)

# Create processor with configuration
processor = CorticalTextProcessor(config=precision_config)
```

---

### Pattern 14: Save and Restore Configuration

```python
# Save configuration with corpus
config = CorticalConfig(pagerank_damping=0.9, min_cluster_size=5)
processor = CorticalTextProcessor(config=config)
processor.add_documents_batch(docs, recompute='full')
processor.save("corpus.pkl")  # Config saved with corpus

# Load and check configuration
loaded = CorticalTextProcessor.load("corpus.pkl")
print(f"Loaded config: {loaded.config.pagerank_damping}")

# Or export/import config separately
config_dict = config.to_dict()
# Save to JSON
import json
with open("config.json", "w") as f:
    json.dump(config_dict, f)

# Restore
with open("config.json") as f:
    restored_config = CorticalConfig.from_dict(json.load(f))
```

---

### Pattern 15: Domain-Specific Configurations

```python
# Code search configuration
code_config = CorticalConfig(
    chunk_size=256,              # Smaller chunks for code
    chunk_overlap=64,
    max_query_expansions=15,     # More expansion for code synonyms
)

# Documentation search configuration
docs_config = CorticalConfig(
    chunk_size=1024,             # Larger chunks for prose
    chunk_overlap=256,
    max_query_expansions=8,
)

# RAG-optimized configuration
rag_config = CorticalConfig(
    chunk_size=512,
    chunk_overlap=128,
    pagerank_iterations=30,      # More iterations for stability
    cluster_strictness=0.7,      # Balanced clustering
)
```

---

## Quick Reference

| Pattern | Use Case | Key Method |
|---------|----------|------------|
| Code tokenization | Index code files | `Tokenizer(split_identifiers=True)` |
| Code concepts | Expand with synonyms | `expand_query_for_code()` |
| Intent search | Natural language | `search_by_intent()` |
| Fingerprinting | Compare code blocks | `get_fingerprint()`, `compare_fingerprints()` |
| Similarity search | Find duplicates | `find_similar_texts()` |
| Doc boosting | Prefer documentation | `find_documents_with_boost()` |
| Custom config | Tune behavior | `CorticalConfig()` |

---

*See also: [Cookbook](cookbook.md) for more recipes, [Query Guide](query-guide.md) for query details.*
