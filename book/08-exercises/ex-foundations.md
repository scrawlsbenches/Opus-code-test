---
title: "Exercises: Foundations"
generated: "2025-12-17T00:01:49.542513Z"
generator: "exercises"
source_files:
  - "test_analysis.py"
  - "test_layers.py"
  - "test_tokenizer.py"
  - "test_minicolumn.py"
tags:
  - exercises
  - foundations
  - beginner
---

# Foundations Exercises

*Hands-on coding exercises to master foundations concepts.*

**Difficulty Level:** Beginner

---

## Introduction

These exercises cover the fundamental algorithms and data structures of the Cortical Text Processor:

- PageRank for term importance
- TF-IDF for relevance scoring
- Graph structures and connections
- Tokenization and text processing

## Exercise: Empty Graph

**Concept:** Empty graph returns empty dict

**Difficulty:** Beginner

**Time:** ~10 minutes

**Source:** `test_analysis.py`

### Setup

```python
from cortical import CorticalTextProcessor
```

### Your Task

Empty graph returns empty dict.

### Hints

<details>
<summary>Hint 1</summary>

Start by considering the edge case of empty input.

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
def test_empty_graph(self):
        """Empty graph returns empty dict."""
        result = _pagerank_core({})
        assert result == {}
```

</details>

---

## Exercise: Single Node No Edges

**Concept:** Single node with no edges gets base rank from damping

**Difficulty:** Beginner

**Time:** ~20 minutes

**Source:** `test_analysis.py`

### Setup

```python
from cortical import CorticalTextProcessor
```

### Your Task

Single node with no edges gets base rank from damping.

### Hints

<details>
<summary>Hint 1</summary>

Focus on the simplest case with just one element.

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
def test_single_node_no_edges(self):
        """Single node with no edges gets base rank from damping."""
        graph = {"a": []}
        result = _pagerank_core(graph, damping=0.85)
        assert "a" in result
        # With no incoming edges, rank = (1-d)/n = 0.15/1 = 0.15
        assert result["a"] == pytest.approx(0.15)
```

</details>

---

## Exercise: Single Node Self Loop

**Concept:** Single node with self-loop still gets rank 1.0

**Difficulty:** Beginner

**Time:** ~20 minutes

**Source:** `test_analysis.py`

### Setup

```python
from cortical import CorticalTextProcessor
```

### Your Task

Single node with self-loop still gets rank 1.0.

### Hints

<details>
<summary>Hint 1</summary>

Focus on the simplest case with just one element.

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
def test_single_node_self_loop(self):
        """Single node with self-loop still gets rank 1.0."""
        graph = {"a": [("a", 1.0)]}
        result = _pagerank_core(graph)
        assert result["a"] == pytest.approx(1.0)
```

</details>

---

## Exercise: Three Node Chain

**Concept:** Chain: a -> b -> c. C should have highest rank

**Difficulty:** Beginner

**Time:** ~20 minutes

**Source:** `test_analysis.py`

### Setup

```python
from cortical import CorticalTextProcessor
```

### Your Task

Chain: a -> b -> c. C should have highest rank.

### Hints

<details>
<summary>Hint 1</summary>

Think about how elements connect in sequence.

</details>

<details>
<summary>Hint 2</summary>

PageRank is computed with `compute_pagerank()` or `compute_importance()`

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_three_node_chain(self):
        """Chain: a -> b -> c. C should have highest rank."""
        graph = {
            "a": [("b", 1.0)],
            "b": [("c", 1.0)],
            "c": []
        }
        result = _pagerank_core(graph)
        # c receives transitively, b receives from a
        assert result["c"] >= result["b"]
        assert result["b"] >= result["a"]
```

</details>

---

## Exercise: Cycle

**Concept:** Cycle: a -> b -> c -> a. All should have equal rank

**Difficulty:** Beginner

**Time:** ~20 minutes

**Source:** `test_analysis.py`

### Setup

```python
from cortical import CorticalTextProcessor
```

### Your Task

Cycle: a -> b -> c -> a. All should have equal rank.

### Hints

<details>
<summary>Hint 1</summary>

Think about how elements connect in sequence.

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
def test_cycle(self):
        """Cycle: a -> b -> c -> a. All should have equal rank."""
        graph = {
            "a": [("b", 1.0)],
            "b": [("c", 1.0)],
            "c": [("a", 1.0)]
        }
        result = _pagerank_core(graph)
        # All nodes in cycle should have equal rank
        assert result["a"] == pytest.approx(result["b"], rel=0.01)
        assert result["b"] == pytest.approx(result["c"], rel=0.01)
```

</details>

---

## Exercise: Empty Corpus

**Concept:** Empty corpus returns empty dict

**Difficulty:** Beginner

**Time:** ~10 minutes

**Source:** `test_analysis.py`

### Setup

```python
from cortical import CorticalTextProcessor
```

### Your Task

Empty corpus returns empty dict.

### Hints

<details>
<summary>Hint 1</summary>

Start by considering the edge case of empty input.

</details>

<details>
<summary>Hint 2</summary>

Check the expected value and comparison in the assertion

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_empty_corpus(self):
        """Empty corpus returns empty dict."""
        result = _tfidf_core({}, num_docs=0)
        assert result == {}
```

</details>

---

## Exercise: Single Term Single Doc

**Concept:** Single term in single doc has IDF of 0

**Difficulty:** Beginner

**Time:** ~20 minutes

**Source:** `test_analysis.py`

### Setup

```python
from cortical import CorticalTextProcessor
```

### Your Task

Single term in single doc has IDF of 0.

### Hints

<details>
<summary>Hint 1</summary>

Focus on the simplest case with just one element.

</details>

<details>
<summary>Hint 2</summary>

Check the expected value and comparison in the assertion

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_single_term_single_doc(self):
        """Single term in single doc has IDF of 0."""
        stats = {
            "term": (5, 1, {"doc1": 5})
        }
        result = _tfidf_core(stats, num_docs=1)
        # IDF = log(1/1) = 0, so TF-IDF = 0
        assert result["term"][0] == pytest.approx(0.0)
```

</details>

---

## Exercise: Empty Graph

**Concept:** Empty graph returns empty dict

**Difficulty:** Beginner

**Time:** ~10 minutes

**Source:** `test_analysis.py`

### Setup

```python
from cortical import CorticalTextProcessor
```

### Your Task

Empty graph returns empty dict.

### Hints

<details>
<summary>Hint 1</summary>

Start by considering the edge case of empty input.

</details>

<details>
<summary>Hint 2</summary>

Check the expected value and comparison in the assertion

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_empty_graph(self):
        """Empty graph returns empty dict."""
        result = _louvain_core({})
        assert result == {}
```

</details>

---

## Exercise: Single Node

**Concept:** Single node is its own community

**Difficulty:** Beginner

**Time:** ~10 minutes

**Source:** `test_analysis.py`

### Setup

```python
from cortical import CorticalTextProcessor
```

### Your Task

Single node is its own community.

### Hints

<details>
<summary>Hint 1</summary>

Focus on the simplest case with just one element.

</details>

<details>
<summary>Hint 2</summary>

Check the expected value and comparison in the assertion

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_single_node(self):
        """Single node is its own community."""
        result = _louvain_core({"a": {}})
        assert "a" in result
        assert result["a"] == 0
```

</details>

---

## Exercise: Empty Graph

**Concept:** Empty graph has zero modularity

**Difficulty:** Beginner

**Time:** ~10 minutes

**Source:** `test_analysis.py`

### Setup

```python
from cortical import CorticalTextProcessor
```

### Your Task

Empty graph has zero modularity.

### Hints

<details>
<summary>Hint 1</summary>

Start by considering the edge case of empty input.

</details>

<details>
<summary>Hint 2</summary>

Check the expected value and comparison in the assertion

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_empty_graph(self):
        """Empty graph has zero modularity."""
        result = _modularity_core({}, {})
        assert result == 0.0
```

</details>

---

---

*Completed 10 exercises? Check out the other topics for more challenges!*
