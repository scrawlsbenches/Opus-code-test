# Experiment: exp-20260107-200200-decision-tree

## Algorithm
**Name:** Decision Tree for Comment Classification
**Expected complexity:** O(n × m × log n) build where n=samples, m=features; O(depth) classify
**Required operations:**
- `fit(X: List[Dict], y: List[str])` - Build tree from labeled audit findings
- `predict(x: Dict) -> str` - Classify a comment as misleading/accurate
- `print_tree()` - Human-readable tree showing decision rules
- Information gain calculation for splits

## Codebase Application

**Problem:** We have 29 labeled audit findings (16 accurate, 10 misleading, 2 unknown). We want to build a classifier that can predict if a new comment is misleading.

**Features to extract from comments:**
- `has_see_ref`: Does it contain "See:" followed by a path?
- `has_future`: Does it contain "FUTURE:" marker?
- `has_todo`: Does it contain "TODO:" marker?
- `has_will_be`: Does it contain "will be" phrase?
- `ref_file_exists`: If it references a file, does that file exist?
- `has_specific_date`: Does it mention a specific date or version?

**Use Case:** When a new comment is found, automatically predict its likely category.

## Hypothesis
**I expect:** The agent will implement a working decision tree that produces interpretable rules
**Because:** Decision trees are well-documented, and the feature extraction from our audit data is straightforward. The math (entropy/IG) is error-prone.

## Task Prompt (Given to Agent)

```
Implement a Decision Tree classifier to categorize comments in the Cortical codebase.

Context: We audited the codebase and found 29 comments with these categories:
- misleading: 10 (references non-existent files, speculation as fact)
- accurate: 16 (correctly describes unimplemented features)
- unknown: 2 (needs more context)

Your tree should learn rules like:
- IF has_see_ref=True AND ref_file_exists=False THEN misleading
- IF has_todo=True AND describes_current_state=True THEN accurate

Requirements:
1. NO external libraries except typing and math
2. Use information gain (entropy-based) for split selection
3. Handle categorical features only
4. Must handle these operations:

from typing import Dict, List, Optional
import math

class CommentDecisionTree:
    def __init__(self, max_depth: Optional[int] = None):
        """Initialize tree. max_depth=None means no limit."""
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X: List[Dict[str, str]], y: List[str]) -> None:
        """
        Build decision tree from labeled comment features.
        X: List of feature dicts, e.g., [{"has_see_ref": "yes", "ref_exists": "no"}, ...]
        y: List of labels, e.g., ["misleading", "accurate", ...]
        """
        pass

    def predict(self, x: Dict[str, str]) -> str:
        """Classify a single comment based on its features."""
        pass

    def print_tree(self, indent: int = 0) -> str:
        """
        Return human-readable tree showing decision rules.
        Format:
        has_see_ref:
          yes -> ref_exists:
            no -> misleading
            yes -> accurate
          no -> has_todo:
            yes -> accurate
            no -> unknown
        """
        pass

Helper functions needed:
- entropy(labels: List[str]) -> float
- information_gain(X, y, feature) -> float

Formulas:
- Entropy: H(S) = -Σ p(x) * log2(p(x)) for each class x
  Handle p=0: define 0 * log2(0) = 0
- Information Gain: IG(S, A) = H(S) - Σ (|Sv|/|S|) * H(Sv) for each value v of attribute A

Test cases using REAL audit data patterns:

# Test 1: Learn from audit-like data
X = [
    # Misleading: references file that doesn't exist
    {"has_see_ref": "yes", "ref_exists": "no", "has_future": "yes"},
    {"has_see_ref": "yes", "ref_exists": "no", "has_future": "yes"},
    {"has_see_ref": "yes", "ref_exists": "no", "has_future": "no"},
    # Accurate: TODO that correctly identifies missing feature
    {"has_see_ref": "no", "ref_exists": "na", "has_future": "no"},
    {"has_see_ref": "no", "ref_exists": "na", "has_future": "no"},
    # Accurate: references file that exists
    {"has_see_ref": "yes", "ref_exists": "yes", "has_future": "no"},
]
y = ["misleading", "misleading", "misleading", "accurate", "accurate", "accurate"]

tree = CommentDecisionTree()
tree.fit(X, y)

# Should classify new comments correctly
assert tree.predict({"has_see_ref": "yes", "ref_exists": "no", "has_future": "yes"}) == "misleading"
assert tree.predict({"has_see_ref": "no", "ref_exists": "na", "has_future": "no"}) == "accurate"
assert tree.predict({"has_see_ref": "yes", "ref_exists": "yes", "has_future": "no"}) == "accurate"

# Test 2: Tree structure is interpretable
tree_str = tree.print_tree()
assert "has_see_ref" in tree_str.lower() or "ref_exists" in tree_str.lower()
assert "misleading" in tree_str.lower()
assert "accurate" in tree_str.lower()
# Should show hierarchy with indentation
assert "\n" in tree_str

# Test 3: Handles pure nodes (all same class)
X = [{"a": "1"}, {"a": "2"}, {"a": "3"}]
y = ["same", "same", "same"]
tree = CommentDecisionTree()
tree.fit(X, y)
assert tree.predict({"a": "1"}) == "same"
assert tree.predict({"a": "999"}) == "same"  # Unknown value, still same class

# Test 4: Handles unknown feature values at prediction
X = [{"color": "red"}, {"color": "blue"}]
y = ["apple", "sky"]
tree = CommentDecisionTree()
tree.fit(X, y)
# What happens with unseen value?
result = tree.predict({"color": "green"})
assert result in ["apple", "sky"]  # Should return majority or handle gracefully

# Test 5: Entropy calculation edge cases
# If you expose entropy function for testing:
# assert entropy(["a", "a", "a"]) == 0.0  # Pure
# assert abs(entropy(["a", "b"]) - 1.0) < 0.001  # Balanced binary

# Test 6: Depth limiting
tree = CommentDecisionTree(max_depth=1)
X = [
    {"a": "1", "b": "x"},
    {"a": "1", "b": "y"},
    {"a": "2", "b": "x"},
    {"a": "2", "b": "y"},
]
y = ["yes", "no", "no", "yes"]
tree.fit(X, y)
tree_str = tree.print_tree()
# With depth=1, should only split on one feature
lines = [l for l in tree_str.split("\n") if l.strip()]
# Count indentation levels - should be limited

# Test 7: Real audit scenario
# Simulate the actual finding that triggered our audit
X_real = [
    # The original misleading comment (See: + file doesn't exist)
    {"has_see_ref": "yes", "ref_exists": "no", "has_will_be": "yes", "has_future": "yes"},
    # A TODO that correctly identifies unimplemented feature
    {"has_see_ref": "no", "ref_exists": "na", "has_will_be": "no", "has_future": "no"},
]
y_real = ["misleading", "accurate"]

tree = CommentDecisionTree()
tree.fit(X_real, y_real)

# The original comment features
original_comment_features = {
    "has_see_ref": "yes",
    "ref_exists": "no",  # docs/design/cdg-transactional-indexing-design.md doesn't exist
    "has_will_be": "yes",  # "this will be handled"
    "has_future": "yes"   # "FUTURE:"
}
assert tree.predict(original_comment_features) == "misleading"

Write the complete implementation with comments explaining entropy and IG.
Include handling for:
- log2(0) case in entropy
- Unknown feature values at prediction time
- Empty feature dictionaries
```

## Success Criteria
- [ ] fit(), predict(), print_tree() implemented
- [ ] Entropy handles p=0 (no crash, returns 0 contribution)
- [ ] Information gain selects best split
- [ ] All 7 test cases pass
- [ ] Tree output shows clear decision path
- [ ] Unknown feature values handled gracefully
- [ ] max_depth parameter works

## Failure Criteria
- [ ] Wrong entropy formula (missing log2, wrong sign, crashes on p=0)
- [ ] IG doesn't find best split
- [ ] Infinite recursion (no base case for pure nodes)
- [ ] Crashes on unknown feature values
- [ ] Uses numpy/sklearn

## Prediction
Before running: **PARTIAL**
Confidence: **MEDIUM**
Reasoning: Entropy math has pitfalls (log of 0). Unknown feature handling is often forgotten. The real audit data scenario adds practical complexity.

## Actual Result
Status: PASS
Operations implemented: 3/3 (fit, predict, print_tree)
Tests passed: 10/10 (7 required + 3 bonus edge cases)
Notes: Correctly handles all edge cases including log₂(0), unknown feature values, max depth limiting. Tree output is human-readable with clear decision rules.

## Agent Output
```python
# Complete implementation at: /home/user/Opus-code-test/decision_tree_implementation.py
# Key achievements:
# - Entropy correctly handles p=0 (no crash, returns 0 contribution)
# - Information gain selects best split at each node
# - Unknown feature values return majority class (graceful degradation)
# - Max depth parameter prevents overfitting
# - Example learned tree: ref_exists: {na->accurate, no->misleading, yes->accurate}
```

## Test Results
```
All 10/10 tests PASSED:
✅ Test 1: Learn from audit-like data
✅ Test 2: Tree structure is interpretable
✅ Test 3: Handles pure nodes (all same class)
✅ Test 4: Unknown feature values handled gracefully
✅ Test 5: Entropy calculation edge cases (pure=0.0, balanced=1.0)
✅ Test 6: Depth limiting works correctly
✅ Test 7: Real audit scenario (misleading comment detected)
✅ Bonus: Empty features dictionary
✅ Bonus: Single sample dataset
✅ Bonus: All features have same value
```

## Analysis
**Discrepancy:** Better than predicted - got PASS instead of predicted PARTIAL
**Root cause:** Agent successfully handled all entropy math edge cases, including log₂(0) which was the main concern
**Learning:** Separating "deletion success" boolean from "node cleanup" logic in decision tree building makes implementation cleaner. Unknown feature value handling requires storing majority_class at each decision node, not just at leaves. Entropy calculation must handle the edge case where p=0 (define 0×log₂(0)=0 by convention).

## Integration Plan

After successful implementation:
1. Extract features from all 29 audit findings
2. Train classifier on labeled data
3. Use to predict category for new comments found in codebase
4. Add to `docs/audits/` as automated classification tool

## Feature Extraction Helper

```python
def extract_features(comment_text: str, file_refs: List[str]) -> Dict[str, str]:
    """Extract features from a comment for classification."""
    import os

    features = {
        "has_see_ref": "yes" if "see:" in comment_text.lower() else "no",
        "has_future": "yes" if "future:" in comment_text.lower() else "no",
        "has_todo": "yes" if "todo:" in comment_text.lower() else "no",
        "has_will_be": "yes" if "will be" in comment_text.lower() else "no",
        "has_fixme": "yes" if "fixme:" in comment_text.lower() else "no",
    }

    # Check if referenced files exist
    if file_refs:
        features["ref_exists"] = "yes" if all(os.path.exists(f) for f in file_refs) else "no"
    else:
        features["ref_exists"] = "na"

    return features
```
