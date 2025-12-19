---
title: "Exercises: Advanced"
generated: "2025-12-17T00:26:24.308012Z"
generator: "exercises"
source_files:
  - "test_semantics.py"
  - "test_fingerprint.py"
  - "test_embeddings.py"
  - "test_gaps.py"
  - "test_patterns.py"
tags:
  - exercises
  - advanced
  - advanced
---

# Advanced Exercises

*Hands-on coding exercises to master advanced concepts.*

**Difficulty Level:** Advanced

---

## Introduction

Challenge yourself with advanced features:

- Semantic relation extraction
- Fingerprint-based similarity
- Graph embeddings
- Knowledge gap detection

## Exercise: Empty Documents

**Concept:** Empty documents return no relations

**Difficulty:** Advanced

**Time:** ~20 minutes

**Source:** `test_semantics.py`

### Setup

```python
from cortical import CorticalTextProcessor
```

### Your Task

Empty documents return no relations.

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
def test_empty_documents(self):
        """Empty documents return no relations."""
        result = extract_pattern_relations({}, {"term1", "term2"})
        assert result == []
```

</details>

---

## Exercise: Empty Valid Terms

**Concept:** No valid terms means no relations extracted

**Difficulty:** Advanced

**Time:** ~20 minutes

**Source:** `test_semantics.py`

### Setup

```python
from cortical import CorticalTextProcessor
```

### Your Task

No valid terms means no relations extracted.

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
def test_empty_valid_terms(self):
        """No valid terms means no relations extracted."""
        docs = {"doc1": "A dog is an animal."}
        result = extract_pattern_relations(docs, set())
        assert result == []
```

</details>

---

## Exercise: Empty Relations

**Concept:** Empty relations list

**Difficulty:** Advanced

**Time:** ~20 minutes

**Source:** `test_semantics.py`

### Setup

```python
from cortical import CorticalTextProcessor
```

### Your Task

Empty relations list.

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
def test_empty_relations(self):
        """Empty relations list."""
        result = get_pattern_statistics([])
        assert result["total_relations"] == 0
        assert result["relation_type_counts"] == {}
```

</details>

---

## Exercise: Single Relation

**Concept:** Single relation statistics

**Difficulty:** Advanced

**Time:** ~20 minutes

**Source:** `test_semantics.py`

### Setup

```python
from cortical import CorticalTextProcessor
```

### Your Task

Single relation statistics.

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
def test_single_relation(self):
        """Single relation statistics."""
        relations = [("dog", "IsA", "animal", 0.9)]
        result = get_pattern_statistics(relations)
        assert result["total_relations"] == 1
        assert result["relation_type_counts"]["IsA"] == 1
```

</details>

---

## Exercise: Empty Relations

**Concept:** Empty relations produce empty hierarchy

**Difficulty:** Advanced

**Time:** ~20 minutes

**Source:** `test_semantics.py`

### Setup

```python
from cortical import CorticalTextProcessor
```

### Your Task

Empty relations produce empty hierarchy.

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
def test_empty_relations(self):
        """Empty relations produce empty hierarchy."""
        parents, children = build_isa_hierarchy([])
        assert parents == {}
        assert children == {}
```

</details>

---

## Exercise: Single Isa

**Concept:** Single IsA relation creates parent-child

**Difficulty:** Advanced

**Time:** ~20 minutes

**Source:** `test_semantics.py`

### Setup

```python
from cortical import CorticalTextProcessor
```

### Your Task

Single IsA relation creates parent-child.

### Hints

<details>
<summary>Hint 1</summary>

Focus on the simplest case with just one element.

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_single_isa(self):
        """Single IsA relation creates parent-child."""
        relations = [("dog", "IsA", "animal", 0.9)]
        parents, children = build_isa_hierarchy(relations)
        assert "dog" in parents
        assert "animal" in parents["dog"]
        assert "animal" in children
        assert "dog" in children["animal"]
```

</details>

---

## Exercise: Hierarchy Chain

**Concept:** Chain: poodle IsA dog IsA animal

**Difficulty:** Advanced

**Time:** ~20 minutes

**Source:** `test_semantics.py`

### Setup

```python
from cortical import CorticalTextProcessor
```

### Your Task

Chain: poodle IsA dog IsA animal.

### Hints

<details>
<summary>Hint 1</summary>

Think about how elements connect in sequence.

</details>

### Solution

<details>
<summary>Click to reveal</summary>

```python
def test_hierarchy_chain(self):
        """Chain: poodle IsA dog IsA animal."""
        relations = [
            ("poodle", "IsA", "dog", 0.9),
            ("dog", "IsA", "animal", 0.9)
        ]
        parents, children = build_isa_hierarchy(relations)
        assert "poodle" in parents
        assert "dog" in parents["poodle"]
        assert "dog" in parents
        assert "animal" in parents["dog"]
```

</details>

---

## Exercise: Empty Hierarchy

**Concept:** Empty hierarchy returns empty ancestors

**Difficulty:** Advanced

**Time:** ~20 minutes

**Source:** `test_semantics.py`

### Setup

```python
from cortical import CorticalTextProcessor
```

### Your Task

Empty hierarchy returns empty ancestors.

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
def test_empty_hierarchy(self):
        """Empty hierarchy returns empty ancestors."""
        result = get_ancestors("dog", {})
        assert result == {}
```

</details>

---

## Exercise: Empty Hierarchy

**Concept:** Empty children dict returns empty descendants

**Difficulty:** Advanced

**Time:** ~20 minutes

**Source:** `test_semantics.py`

### Setup

```python
from cortical import CorticalTextProcessor
```

### Your Task

Empty children dict returns empty descendants.

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
def test_empty_hierarchy(self):
        """Empty children dict returns empty descendants."""
        result = get_descendants("animal", {})
        assert result == {}
```

</details>

---

## Exercise: Empty Corpus

**Concept:** Empty corpus returns no relations

**Difficulty:** Advanced

**Time:** ~20 minutes

**Source:** `test_semantics.py`

### Setup

```python
from cortical import CorticalTextProcessor
from cortical.tokenizer import Tokenizer
```

### Your Task

Empty corpus returns no relations.

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
        """Empty corpus returns no relations."""
        layers = {CorticalLayer.TOKENS: HierarchicalLayer(CorticalLayer.TOKENS)}
        tokenizer = Tokenizer()
        result = extract_corpus_semantics(layers, {}, tokenizer)
        assert result == []
```

</details>

---

---

*Completed 10 exercises? Check out the other topics for more challenges!*
