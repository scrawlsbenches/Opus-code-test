# Experiment: exp-20260107-200200-decision-tree

## Algorithm
**Name:** Decision Tree (ID3/C4.5 style)
**Expected complexity:** O(n × m × log n) build where n=samples, m=features; O(depth) classify
**Required operations:**
- `fit(X: List[Dict], y: List[str])` - Build tree from labeled data
- `predict(x: Dict) -> str` - Classify a single example
- `print_tree()` - Human-readable tree representation
- Information gain calculation for splits

## Hypothesis
**I expect:** The agent will implement a working decision tree but may struggle with the information gain calculation
**Because:** Decision trees require understanding entropy and information gain formulas. The recursive structure is straightforward but the math is error-prone.

## Task Prompt (Given to Agent)

```
Implement a Decision Tree classifier from scratch in Python.

A decision tree recursively splits data based on features that maximize information gain.
This is the foundation of interpretable machine learning.

Requirements:
1. NO external libraries except typing and math (no sklearn, no numpy, no pandas)
2. Must use information gain (entropy-based) for split selection
3. Must handle categorical features (not numeric ranges)
4. Must handle these operations:

class DecisionTree:
    def fit(self, X: List[Dict[str, str]], y: List[str]) -> None:
        """
        Build the decision tree from training data.
        X: List of feature dictionaries, e.g., [{"color": "red", "size": "big"}, ...]
        y: List of labels, e.g., ["apple", "apple", "orange", ...]
        """
        pass

    def predict(self, x: Dict[str, str]) -> str:
        """Classify a single example."""
        pass

    def print_tree(self, indent: int = 0) -> str:
        """Return human-readable tree representation."""
        pass

Helper functions you'll need:
- entropy(labels) -> float: Calculate entropy of a label distribution
- information_gain(data, labels, feature) -> float: Calculate IG for splitting on feature

Formulas:
- Entropy: H(S) = -Σ p(x) * log2(p(x)) for each class x
- Information Gain: IG(S, A) = H(S) - Σ (|Sv|/|S|) * H(Sv) for each value v of attribute A

Test cases that MUST pass:

# Test 1: Simple AND logic
X = [
    {"a": "T", "b": "T"},
    {"a": "T", "b": "F"},
    {"a": "F", "b": "T"},
    {"a": "F", "b": "F"},
]
y = ["T", "F", "F", "F"]  # a AND b
tree = DecisionTree()
tree.fit(X, y)
assert tree.predict({"a": "T", "b": "T"}) == "T"
assert tree.predict({"a": "T", "b": "F"}) == "F"
assert tree.predict({"a": "F", "b": "T"}) == "F"

# Test 2: Weather/Play tennis classic dataset
X = [
    {"outlook": "sunny", "humidity": "high"},
    {"outlook": "sunny", "humidity": "high"},
    {"outlook": "overcast", "humidity": "high"},
    {"outlook": "rain", "humidity": "high"},
    {"outlook": "rain", "humidity": "normal"},
    {"outlook": "overcast", "humidity": "normal"},
    {"outlook": "sunny", "humidity": "normal"},
]
y = ["no", "no", "yes", "yes", "yes", "yes", "yes"]
tree = DecisionTree()
tree.fit(X, y)
# Sunny + high humidity should be "no"
assert tree.predict({"outlook": "sunny", "humidity": "high"}) == "no"
# Overcast should always be "yes"
assert tree.predict({"outlook": "overcast", "humidity": "high"}) == "yes"

# Test 3: Print tree should show structure
tree_str = tree.print_tree()
assert "outlook" in tree_str.lower() or "humidity" in tree_str.lower()

# Test 4: Single class (pure node)
X = [{"a": "1"}, {"a": "2"}, {"a": "3"}]
y = ["same", "same", "same"]
tree = DecisionTree()
tree.fit(X, y)
assert tree.predict({"a": "1"}) == "same"

Write the complete implementation with entropy and information gain calculations.
Include comments explaining the algorithm at each step.
```

## Success Criteria
- [ ] fit(), predict(), print_tree() implemented
- [ ] Entropy calculation correct
- [ ] Information gain calculation correct
- [ ] All 4 test cases pass
- [ ] Tree structure is interpretable

## Failure Criteria
- [ ] Wrong entropy formula (e.g., missing log2, wrong sign)
- [ ] Information gain doesn't find best split
- [ ] Infinite recursion (no base case)
- [ ] Can't handle pure nodes
- [ ] Uses numpy/sklearn

## Prediction
Before running: **PARTIAL**
Confidence: **MEDIUM**
Reasoning: The recursive structure is standard, but entropy/IG math has many opportunities for errors (log of 0, wrong base, forgetting proportions).

## Actual Result
Status: [NOT YET RUN]
Operations implemented: [X/3]
Tests passed: [X/4]
Notes:

## Agent Output
```python
[Agent's code will be pasted here after running]
```

## Test Results
```
[Test execution output will be pasted here]
```

## Analysis
**Discrepancy:**
**Root cause:**
**Learning:**

## Recommendations

