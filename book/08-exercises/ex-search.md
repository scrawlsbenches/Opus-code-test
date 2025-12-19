---
title: "Exercises: Search"
generated: "2025-12-17T00:26:22.688240Z"
generator: "exercises"
source_files:
  - "test_query_search.py"
  - "test_query_expansion.py"
  - "test_query_ranking.py"
  - "test_query_passages.py"
  - "test_query_definitions.py"
tags:
  - exercises
  - search
  - intermediate
---

# Search Exercises

*Hands-on coding exercises to master search concepts.*

**Difficulty Level:** Intermediate

---

## Introduction

Master the search and retrieval capabilities:

- Query expansion techniques
- Document ranking algorithms
- Passage retrieval
- Definition extraction

## Exercise: Empty Query

**Concept:** Empty query returns empty results

**Difficulty:** Intermediate

**Time:** ~20 minutes

**Source:** `test_query_search.py`

### Setup

```python
from cortical import CorticalTextProcessor
from cortical.tokenizer import Tokenizer
```

### Your Task

Empty query returns empty results.

### Hints

<details>
<summary>Hint 1</summary>

Start by considering the edge case of empty input.

</details>

<details>
<summary>Hint 2</summary>

You may need to create mock layers or minicolumns for testing

</details>

<details>
<summary>Hint 3</summary>

Check the expected value and comparison in the assertion

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_empty_query(self):
        """Empty query returns empty results."""
        layers = MockLayers.single_term("term", tfidf=1.0, doc_ids=["doc1"])
        tokenizer = Tokenizer()

        # Tokenizer will return empty list for empty string
        result = find_documents_for_query("", layers, tokenizer)
        assert result == []
```

</details>

---

## Exercise: Single Term Single Doc

**Concept:** Single term matching single document

**Difficulty:** Intermediate

**Time:** ~20 minutes

**Source:** `test_query_search.py`

### Setup

```python
from cortical import CorticalTextProcessor
from cortical.tokenizer import Tokenizer
```

### Your Task

Single term matching single document.

### Hints

<details>
<summary>Hint 1</summary>

Focus on the simplest case with just one element.

</details>

<details>
<summary>Hint 2</summary>

You may need to create mock layers or minicolumns for testing

</details>

<details>
<summary>Hint 3</summary>

Check the expected value and comparison in the assertion

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_single_term_single_doc(self):
        """Single term matching single document."""
        # Create layer with term in doc1
        col = MockMinicolumn(
            content="neural",
            tfidf=2.5,
            document_ids={"doc1"},
            tfidf_per_doc={"doc1": 2.5}
        )
        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([col])

        tokenizer = Tokenizer()
        result = find_documents_for_query(
            "neural", layers, tokenizer, use_expansion=False
        )

        assert len(result) == 1
        assert result[0][0] == "doc1"
        assert result[0][1] > 0
```

</details>

---

## Exercise: Single Term Multiple Docs

**Concept:** Single term in multiple documents ranked by TF-IDF

**Difficulty:** Intermediate

**Time:** ~20 minutes

**Source:** `test_query_search.py`

### Setup

```python
from cortical import CorticalTextProcessor
from cortical.tokenizer import Tokenizer
```

### Your Task

Single term in multiple documents ranked by TF-IDF.

### Hints

<details>
<summary>Hint 1</summary>

Focus on the simplest case with just one element.

</details>

<details>
<summary>Hint 2</summary>

You may need to create mock layers or minicolumns for testing

</details>

<details>
<summary>Hint 3</summary>

Check the expected value and comparison in the assertion

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_single_term_multiple_docs(self):
        """Single term in multiple documents ranked by TF-IDF."""
        col = MockMinicolumn(
            content="algorithm",
            tfidf=3.0,
            document_ids={"doc1", "doc2", "doc3"},
            tfidf_per_doc={"doc1": 5.0, "doc2": 3.0, "doc3": 1.0}
        )
        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([col])

        tokenizer = Tokenizer()
        result = find_documents_for_query(
            "algorithm", layers, tokenizer, use_expansion=False
        )

        assert len(result) == 3
        # Should be sorted by TF-IDF score
        assert result[0][0] == "doc1"  # Highest score
        assert result[1][0] == "doc2"
        assert result[2][0] == "doc3"  # Lowest score
        assert result[0][1] > result[1][1] > result[2][1]
```

</details>

---

## Exercise: Query Expansion Disabled

**Concept:** use_expansion=False uses only query terms

**Difficulty:** Intermediate

**Time:** ~20 minutes

**Source:** `test_query_search.py`

### Setup

```python
from cortical import CorticalTextProcessor
from cortical.tokenizer import Tokenizer
```

### Your Task

use_expansion=False uses only query terms.

### Hints

<details>
<summary>Hint 1</summary>

Break down the problem into smaller steps.

</details>

<details>
<summary>Hint 2</summary>

PageRank is computed with `compute_pagerank()` or `compute_importance()`

</details>

<details>
<summary>Hint 3</summary>

Check the expected value and comparison in the assertion

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_query_expansion_disabled(self):
        """use_expansion=False uses only query terms."""
        # Create connected terms
        layers = (
            LayerBuilder()
            .with_term("neural", tfidf=2.0, pagerank=0.8)
            .with_term("network", tfidf=2.0, pagerank=0.6)
            .with_connection("neural", "network", weight=5.0)
            .with_document("doc1", ["neural"])
            .with_document("doc2", ["network"])
            .build()
        )

        layer0 = layers[MockLayers.TOKENS]
        layer0.get_minicolumn("neural").tfidf_per_doc = {"doc1": 2.0}
        layer0.get_minicolumn("network").tfidf_per_doc = {"doc2": 2.0}

        tokenizer = Tokenizer()
        result = find_documents_for_query(
            "neural", layers, tokenizer,
            use_expansion=False
        )

        # Should only find doc1 (contains "neural")
        assert len(result) == 1
        assert result[0][0] == "doc1"
```

</details>

---

## Exercise: Empty Corpus

**Concept:** Empty corpus returns empty results

**Difficulty:** Intermediate

**Time:** ~20 minutes

**Source:** `test_query_search.py`

### Setup

```python
from cortical import CorticalTextProcessor
from cortical.tokenizer import Tokenizer
```

### Your Task

Empty corpus returns empty results.

### Hints

<details>
<summary>Hint 1</summary>

Start by considering the edge case of empty input.

</details>

<details>
<summary>Hint 2</summary>

You may need to create mock layers or minicolumns for testing

</details>

<details>
<summary>Hint 3</summary>

Check the expected value and comparison in the assertion

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_empty_corpus(self):
        """Empty corpus returns empty results."""
        layers = MockLayers.empty()
        tokenizer = Tokenizer()

        result = find_documents_for_query("query", layers, tokenizer)

        assert result == []
```

</details>

---

## Exercise: Single Term Match

**Concept:** Fast search finds document with matching term

**Difficulty:** Intermediate

**Time:** ~20 minutes

**Source:** `test_query_search.py`

### Setup

```python
from cortical import CorticalTextProcessor
from cortical.tokenizer import Tokenizer
```

### Your Task

Fast search finds document with matching term.

### Hints

<details>
<summary>Hint 1</summary>

Focus on the simplest case with just one element.

</details>

<details>
<summary>Hint 2</summary>

You may need to create mock layers or minicolumns for testing

</details>

<details>
<summary>Hint 3</summary>

Check the expected value and comparison in the assertion

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_single_term_match(self):
        """Fast search finds document with matching term."""
        col = MockMinicolumn(
            content="algorithm",
            document_ids={"doc1"},
            tfidf_per_doc={"doc1": 3.0}
        )
        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([col])

        tokenizer = Tokenizer()
        result = fast_find_documents("algorithm", layers, tokenizer)

        assert len(result) == 1
        assert result[0][0] == "doc1"
```

</details>

---

## Exercise: Empty Query

**Concept:** Empty query returns empty results

**Difficulty:** Intermediate

**Time:** ~20 minutes

**Source:** `test_query_search.py`

### Setup

```python
from cortical import CorticalTextProcessor
from cortical.tokenizer import Tokenizer
```

### Your Task

Empty query returns empty results.

### Hints

<details>
<summary>Hint 1</summary>

Start by considering the edge case of empty input.

</details>

<details>
<summary>Hint 2</summary>

You may need to create mock layers or minicolumns for testing

</details>

<details>
<summary>Hint 3</summary>

Check the expected value and comparison in the assertion

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_empty_query(self):
        """Empty query returns empty results."""
        layers = MockLayers.single_term("term", doc_ids=["doc1"])
        tokenizer = Tokenizer()

        result = fast_find_documents("", layers, tokenizer)

        assert result == []
```

</details>

---

## Exercise: No Candidates Returns Empty

**Concept:** No matching candidates returns empty

**Difficulty:** Intermediate

**Time:** ~20 minutes

**Source:** `test_query_search.py`

### Setup

```python
from cortical import CorticalTextProcessor
from cortical.tokenizer import Tokenizer
```

### Your Task

No matching candidates returns empty.

### Hints

<details>
<summary>Hint 1</summary>

Start by considering the edge case of empty input.

</details>

<details>
<summary>Hint 2</summary>

You may need to create mock layers or minicolumns for testing

</details>

<details>
<summary>Hint 3</summary>

Check the expected value and comparison in the assertion

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_no_candidates_returns_empty(self):
        """No matching candidates returns empty."""
        layers = MockLayers.single_term("existing", doc_ids=["doc1"])
        tokenizer = Tokenizer()

        result = fast_find_documents("nonexistent", layers, tokenizer)

        assert result == []
```

</details>

---

## Exercise: Empty Layer

**Concept:** Empty layer returns empty index

**Difficulty:** Intermediate

**Time:** ~10 minutes

**Source:** `test_query_search.py`

### Setup

```python
from cortical import CorticalTextProcessor
```

### Your Task

Empty layer returns empty index.

### Hints

<details>
<summary>Hint 1</summary>

Start by considering the edge case of empty input.

</details>

<details>
<summary>Hint 2</summary>

You may need to create mock layers or minicolumns for testing

</details>

<details>
<summary>Hint 3</summary>

Check the expected value and comparison in the assertion

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_empty_layer(self):
        """Empty layer returns empty index."""
        layers = MockLayers.empty()
        result = build_document_index(layers)
        assert result == {}
```

</details>

---

## Exercise: Single Term Single Doc

**Concept:** Single term in single document

**Difficulty:** Intermediate

**Time:** ~20 minutes

**Source:** `test_query_search.py`

### Setup

```python
from cortical import CorticalTextProcessor
```

### Your Task

Single term in single document.

### Hints

<details>
<summary>Hint 1</summary>

Focus on the simplest case with just one element.

</details>

<details>
<summary>Hint 2</summary>

You may need to create mock layers or minicolumns for testing

</details>

<details>
<summary>Hint 3</summary>

Check the expected value and comparison in the assertion

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_single_term_single_doc(self):
        """Single term in single document."""
        col = MockMinicolumn(
            content="term",
            document_ids={"doc1"},
            tfidf_per_doc={"doc1": 2.5}
        )
        layers = MockLayers.empty()
        layers[MockLayers.TOKENS] = MockHierarchicalLayer([col])

        result = build_document_index(layers)

        assert "term" in result
        assert result["term"] == {"doc1": 2.5}
```

</details>

---

---

*Completed 10 exercises? Check out the other topics for more challenges!*
