from typing import Dict, List, Optional, Any
import math


class CommentDecisionTree:
    def __init__(self, max_depth: Optional[int] = None):
        """Initialize tree. max_depth=None means no limit."""
        self.max_depth = max_depth
        self.tree = None

    def _entropy(self, labels: List[str]) -> float:
        """
        Calculate entropy of a label set.
        H(S) = -Σ p(x) * log2(p(x)) for each class x

        Entropy measures the impurity/uncertainty in a set of labels.
        - 0.0 means pure (all same class)
        - Higher values mean more mixed

        Edge case: 0 * log2(0) is mathematically undefined, but we define it as 0
        (no contribution to entropy from non-existent class)
        """
        if not labels:
            return 0.0

        # Count occurrences of each label
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1

        # Calculate entropy
        total = len(labels)
        entropy = 0.0
        for count in label_counts.values():
            if count == 0:
                continue  # 0 * log2(0) = 0 by definition
            p = count / total
            entropy -= p * math.log2(p)

        return entropy

    def _information_gain(self, X: List[Dict[str, str]], y: List[str], feature: str) -> float:
        """
        Calculate information gain for splitting on a feature.
        IG(S, A) = H(S) - Σ (|Sv|/|S|) * H(Sv) for each value v of attribute A

        Information gain measures how much splitting on this feature reduces uncertainty.
        - Higher IG means better split
        - We choose the feature with highest IG at each node
        """
        # Calculate entropy of parent (before split)
        parent_entropy = self._entropy(y)

        # Group samples by feature value
        feature_groups = {}
        for i, sample in enumerate(X):
            value = sample.get(feature, "unknown")
            if value not in feature_groups:
                feature_groups[value] = []
            feature_groups[value].append(y[i])

        # Calculate weighted average of child entropies (after split)
        total = len(y)
        weighted_entropy = 0.0
        for labels in feature_groups.values():
            weight = len(labels) / total
            weighted_entropy += weight * self._entropy(labels)

        # Information gain = reduction in entropy
        return parent_entropy - weighted_entropy

    def _get_majority_class(self, labels: List[str]) -> str:
        """Return the most common label. Used for leaf nodes and handling unknowns."""
        if not labels:
            return "unknown"

        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1

        return max(label_counts.items(), key=lambda x: x[1])[0]

    def _build_tree(self, X: List[Dict[str, str]], y: List[str],
                    features: List[str], depth: int = 0) -> Dict[str, Any]:
        """
        Recursively build decision tree using ID3 algorithm.

        Returns a tree node (dict) with structure:
        Leaf node: {"type": "leaf", "class": class_label}
        Decision node: {"type": "decision", "feature": feature_name,
                        "children": {value: subtree}, "majority_class": fallback}
        """
        # Base case 1: No samples (shouldn't happen but handle gracefully)
        if not X or not y:
            return {"type": "leaf", "class": "unknown"}

        # Base case 2: All labels are the same (pure node - no need to split)
        unique_labels = set(y)
        if len(unique_labels) == 1:
            return {"type": "leaf", "class": y[0]}

        # Base case 3: Max depth reached (prevent overfitting)
        if self.max_depth is not None and depth >= self.max_depth:
            return {"type": "leaf", "class": self._get_majority_class(y)}

        # Base case 4: No features left to split on
        if not features:
            return {"type": "leaf", "class": self._get_majority_class(y)}

        # Find best feature to split on (highest information gain)
        best_feature = None
        best_gain = -1
        for feature in features:
            gain = self._information_gain(X, y, feature)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature

        # If no information gain, create leaf (splitting won't help)
        if best_feature is None or best_gain == 0:
            return {"type": "leaf", "class": self._get_majority_class(y)}

        # Create decision node
        node = {
            "type": "decision",
            "feature": best_feature,
            "children": {},
            "majority_class": self._get_majority_class(y)  # Fallback for unknown values
        }

        # Split data by feature values
        feature_splits = {}
        for i, sample in enumerate(X):
            value = sample.get(best_feature, "unknown")
            if value not in feature_splits:
                feature_splits[value] = {"X": [], "y": []}
            feature_splits[value]["X"].append(sample)
            feature_splits[value]["y"].append(y[i])

        # Recursively build subtrees for each value
        remaining_features = [f for f in features if f != best_feature]
        for value, split_data in feature_splits.items():
            node["children"][value] = self._build_tree(
                split_data["X"],
                split_data["y"],
                remaining_features,
                depth + 1
            )

        return node

    def fit(self, X: List[Dict[str, str]], y: List[str]) -> None:
        """
        Build decision tree from labeled comment features.
        X: List of feature dicts, e.g., [{"has_see_ref": "yes", "ref_exists": "no"}, ...]
        y: List of labels, e.g., ["misleading", "accurate", ...]

        Algorithm: ID3 (Iterative Dichotomiser 3)
        - Recursively select best feature using information gain
        - Split on that feature
        - Repeat for each branch until pure or no features left
        """
        if not X or not y:
            self.tree = {"type": "leaf", "class": "unknown"}
            return

        # Extract all unique features from the dataset
        all_features = set()
        for sample in X:
            all_features.update(sample.keys())
        features = list(all_features)

        # Build the tree
        self.tree = self._build_tree(X, y, features)

    def predict(self, x: Dict[str, str]) -> str:
        """
        Classify a single comment based on its features.

        Edge case handling:
        - Unknown feature values: Use majority class of current node
        - Empty input: Return "unknown"
        - Missing features: Treated as unknown values
        """
        if self.tree is None:
            return "unknown"

        # Traverse the tree
        node = self.tree
        while node["type"] == "decision":
            feature = node["feature"]
            value = x.get(feature, "unknown")

            # If we've seen this value during training, follow that branch
            if value in node["children"]:
                node = node["children"][value]
            else:
                # Unknown value: return majority class of this node
                # This handles unseen feature values gracefully
                return node.get("majority_class", "unknown")

        return node["class"]

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
        if self.tree is None:
            return "Empty tree"

        return self._print_node(self.tree, 0)

    def _print_node(self, node: Dict[str, Any], indent: int) -> str:
        """Recursively print tree node with proper indentation."""
        if node["type"] == "leaf":
            return node["class"]

        # Decision node
        lines = [f"{node['feature']}:"]

        for value, child in sorted(node["children"].items()):
            child_str = self._print_node(child, indent + 1)

            if "\n" in child_str:
                # Multi-line child (nested decision node)
                first_line, *rest_lines = child_str.split("\n")
                lines.append(f"{'  ' * (indent + 1)}{value} -> {first_line}")
                for line in rest_lines:
                    lines.append(f"{'  ' * (indent + 1)}{line}")
            else:
                # Single line (leaf node)
                lines.append(f"{'  ' * (indent + 1)}{value} -> {child_str}")

        return "\n".join(lines)


# Helper functions for external testing
def entropy(labels: List[str]) -> float:
    """Standalone entropy function for testing."""
    tree = CommentDecisionTree()
    return tree._entropy(labels)
