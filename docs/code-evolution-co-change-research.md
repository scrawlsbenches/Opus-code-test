# Co-Change Analysis and Change Impact Prediction Research

**Date:** 2025-12-22
**Task:** T-20251222-234301-c60527ec
**Context:** SparkCodeIntelligence Code Evolution Model
**Tags:** `co-change`, `change-impact`, `MSR`, `temporal-weighting`, `git-mining`

---

## Executive Summary

This document synthesizes research on co-change analysis and change impact prediction for the Code Evolution Model. The goal is to augment SparkCodeIntelligence's static analysis (AST + N-gram) with temporal co-change patterns learned from git history.

**Key Insight:** Files that changed together in the past are likely to change together in the future. This "logical coupling" captures dependencies invisible to static analysis (e.g., configuration files, test files, documentation).

**Proposed Implementation:** A hybrid model combining:
1. **Static analysis** - AST call graphs, import dependencies (already in SparkCodeIntelligence)
2. **Co-change mining** - Association rules from git commit history
3. **Temporal weighting** - Recent commits are more predictive than old ones
4. **ML-based ranking** - Confidence scoring for change predictions

---

## 1. Co-Change Patterns

### Definition

**Co-change** (also called "change coupling" or "logical coupling") occurs when files frequently change together in commits. If file A and file B are modified in the same commit repeatedly, they are logically coupled.

### Types of Coupling

| Type | Definition | Detection Method | Example |
|------|------------|------------------|---------|
| **Structural Coupling** | Direct source code dependencies (imports, calls, inheritance) | Static analysis (AST parsing) | `import module` creates structural coupling |
| **Logical Coupling** | Files that change together historically, with or without structural dependency | Commit history mining | Test file always changes when implementation changes |
| **Change Coupling** | Temporal co-occurrence in commits | Git log analysis | Config file changes when feature is added |
| **Conceptual Coupling** | Files related by shared concepts (e.g., "authentication") | Text similarity, topic modeling | Multiple auth-related files |

**Critical Distinction:** Logical coupling captures relationships that static analysis misses:
- Test files don't import production code in Python
- Configuration files have no code dependencies
- Documentation files have semantic relationships
- Build scripts change with new features

### Mining Techniques from Git History

**Basic Algorithm:**
```python
# Pseudocode for co-change mining
commits = parse_git_log()
co_change_matrix = defaultdict(lambda: defaultdict(int))

for commit in commits:
    files = commit.changed_files
    for file_a in files:
        for file_b in files:
            if file_a != file_b:
                co_change_matrix[file_a][file_b] += 1

# Normalize by frequency
for file_a, neighbors in co_change_matrix.items():
    total = sum(neighbors.values())
    for file_b in neighbors:
        neighbors[file_b] /= total  # Convert to probability
```

**Advanced Considerations:**
- **Commit granularity:** Large refactoring commits vs small bug fixes
- **Author bias:** Different developers have different commit patterns
- **Time windows:** Recent commits may be more relevant
- **File churn:** High-churn files co-change with everything (need dampening)

---

## 2. Change Impact Analysis Literature

### Key Research Areas

#### 2.1 Association Rule Mining (APRIORI-based)

**Seminal Work:** Zimmermann et al. (2005) - "Mining Version Histories to Guide Software Changes"

**Approach:**
- Apply APRIORI algorithm (from market basket analysis) to commits
- Treat files as "items" and commits as "transactions"
- Extract association rules: `{file_a.py} → {file_b.py, file_c.py}` (confidence: 0.75)

**Metrics:**
- **Support:** How often does the rule occur? `support(A → B) = commits_with_both / total_commits`
- **Confidence:** When A changes, how often does B change? `confidence(A → B) = commits_with_both / commits_with_A`
- **Lift:** How much more likely is the co-change than random? `lift(A → B) = confidence / P(B)`

**Results from Literature:**
- 26% of further file changes correctly predicted
- Top 3 suggestions contain correct file with 64% likelihood
- Detects coupling invisible to program analysis

**Limitations:**
- High-churn files generate spurious rules
- Large commits can skew statistics
- Requires minimum support threshold tuning

#### 2.2 Graph-Based Approaches

**Change Propagation Graphs:**
- Nodes: Source files
- Edges: Co-change frequency (weighted by commit count)
- Queries: "If I change file X, what's the reachability?"

**Recent Advances (2023-2024):**
1. **Temporal Graph Neural Networks (TGNN)**
   - Model software as temporal graph where edges represent co-changeability
   - LSTM + Graph Attention Networks learn propagation patterns
   - 21% better accuracy than previous methods at package level (NDCG@K metric)

2. **Skip-gram Embeddings for File Co-change**
   - Learn file embeddings from commit sequences
   - Files that co-change have similar embeddings
   - Unsupervised nearest neighbors for prediction

3. **Graph-Based ML for JIT Defect Prediction**
   - Contribution graphs (developers + files)
   - F1 score up to 77.55% (152% improvement over baseline)

**Key Papers:**
- "To change or not to change? Modeling software system interactions using Temporal Graphs and Graph Neural Networks" (ScienceDirect, 2023)
- "Enhancing Change Impact Prediction by Integrating Evolutionary Coupling with Software Change Relationships" (ACM ESEM, 2024)

#### 2.3 ML Approaches for Change Prediction

**Features Used:**
1. **Structural:** Call graph, import graph, class hierarchy
2. **Historical:** Co-change frequency, recency, author patterns
3. **Textual:** Commit message intent, file path similarity
4. **Semantic:** Code clone detection, refactoring patterns

**Algorithms:**
- Decision trees (J48, LMT)
- Random forests
- Neural networks (LSTM, GNN)
- Ensemble methods

**Performance:**
- Precision: 30-50% typical
- Recall: 50-70% typical
- **Recall is more important than precision** for guiding developers (false negatives are costly)

---

## 3. Temporal Weighting

### Recency Bias: Are Recent Commits More Predictive?

**Research Consensus:** YES. Recent commits are significantly more predictive of future changes.

**Evidence:**
- Joshi et al. (2007) & Kim et al. (2007): Recency weighting improves bug prediction accuracy to 99% (AUC 0.9251)
- Temporal features outperform static features alone
- Older co-changes may reflect obsolete architecture

### Decay Functions

| Function | Formula | Characteristics | Use Case |
|----------|---------|-----------------|----------|
| **Exponential** | `weight = exp(-λ * age_days)` | Steep decay, favors recent | Fast-moving codebases |
| **Linear** | `weight = max(0, 1 - age_days / window)` | Gradual decline | Stable codebases |
| **Sliding Window** | `weight = 1 if age < window else 0` | Hard cutoff | Feature-specific windows |
| **Hyperbolic** | `weight = 1 / (1 + age_days)` | Gentle long-tail decay | Mixed change rates |

**Recommended:** Exponential decay with λ = 0.01 (50% weight at ~69 days)

**Pseudocode:**
```python
def compute_weighted_co_change(commits, decay_lambda=0.01):
    today = datetime.now()
    weighted_matrix = defaultdict(lambda: defaultdict(float))

    for commit in commits:
        age_days = (today - commit.date).days
        weight = math.exp(-decay_lambda * age_days)

        for file_a in commit.files:
            for file_b in commit.files:
                if file_a != file_b:
                    weighted_matrix[file_a][file_b] += weight

    return weighted_matrix
```

### Seasonal Patterns in Codebases

**Observed Patterns:**
- **Pre-release churn:** High co-change before major releases
- **Post-release stabilization:** Focused bug fixes (different co-change patterns)
- **Feature development cycles:** Different files during feature vs maintenance

**Implication:** May need multiple models for different development phases.

### Developer-Specific Patterns

**Findings:**
- Different developers have different commit granularities
- Some developers make atomic commits, others batch changes
- Author-weighted co-change can improve predictions

**Proposal:** Track per-developer commit patterns, use as signal (not filter).

---

## 4. Proposed Co-Change Model Design

### Data Structure

```python
from dataclasses import dataclass
from typing import Dict, List, Set
from datetime import datetime

@dataclass
class CoChangeEdge:
    """Represents a co-change relationship between two files."""
    source_file: str
    target_file: str
    co_change_count: int
    weighted_score: float  # Temporal decay applied
    confidence: float      # Normalized probability
    last_co_change: datetime
    commits: List[str]     # Commit SHAs for traceability

@dataclass
class CoChangeGraph:
    """Graph of file co-change relationships."""
    edges: Dict[str, Dict[str, CoChangeEdge]]  # {file_a: {file_b: edge}}
    file_metadata: Dict[str, FileMetadata]     # File stats
    decay_lambda: float = 0.01                 # Temporal decay parameter

    def get_neighbors(self, file: str, threshold: float = 0.1) -> List[CoChangeEdge]:
        """Get files that co-change with given file (filtered by confidence)."""
        return [edge for edge in self.edges.get(file, {}).values()
                if edge.confidence >= threshold]

    def predict_changes(self, changed_files: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        """Given changed files, predict what else will change."""
        candidates = defaultdict(float)

        for file in changed_files:
            for edge in self.get_neighbors(file):
                # Aggregate scores from multiple changed files
                candidates[edge.target_file] += edge.confidence

        # Remove already-changed files
        for file in changed_files:
            candidates.pop(file, None)

        # Return top K by aggregated confidence
        return sorted(candidates.items(), key=lambda x: -x[1])[:top_k]

@dataclass
class FileMetadata:
    """Metadata for a file in the codebase."""
    path: str
    total_commits: int
    last_modified: datetime
    primary_authors: List[str]
    avg_churn_per_commit: float  # Lines changed per commit
```

### Update Algorithm (Incremental)

```python
class CoChangeModelUpdater:
    """Incrementally update co-change model as new commits arrive."""

    def __init__(self, model: CoChangeGraph):
        self.model = model

    def process_new_commit(self, commit_sha: str, changed_files: List[str], commit_date: datetime):
        """Add a new commit to the co-change model."""
        # Compute temporal weight for this commit
        age_days = (datetime.now() - commit_date).days
        weight = math.exp(-self.model.decay_lambda * age_days)

        # Update co-change edges
        for file_a in changed_files:
            for file_b in changed_files:
                if file_a == file_b:
                    continue

                # Get or create edge
                if file_a not in self.model.edges:
                    self.model.edges[file_a] = {}

                if file_b not in self.model.edges[file_a]:
                    self.model.edges[file_a][file_b] = CoChangeEdge(
                        source_file=file_a,
                        target_file=file_b,
                        co_change_count=0,
                        weighted_score=0.0,
                        confidence=0.0,
                        last_co_change=commit_date,
                        commits=[]
                    )

                edge = self.model.edges[file_a][file_b]
                edge.co_change_count += 1
                edge.weighted_score += weight
                edge.last_co_change = max(edge.last_co_change, commit_date)
                edge.commits.append(commit_sha)

        # Recompute confidences (normalize by source file activity)
        self._recompute_confidences()

    def _recompute_confidences(self):
        """Normalize weighted scores to [0, 1] confidence range."""
        for file_a, neighbors in self.model.edges.items():
            total_weight = sum(edge.weighted_score for edge in neighbors.values())
            if total_weight > 0:
                for edge in neighbors.values():
                    edge.confidence = edge.weighted_score / total_weight

    def decay_old_scores(self):
        """Apply temporal decay to all existing edges (called periodically)."""
        today = datetime.now()
        for file_a, neighbors in self.model.edges.items():
            for edge in neighbors.values():
                age_days = (today - edge.last_co_change).days
                decay_factor = math.exp(-self.model.decay_lambda * age_days)
                edge.weighted_score *= decay_factor

        self._recompute_confidences()
```

### Query Interface

```python
class CoChangeQuery:
    """High-level query interface for co-change model."""

    def __init__(self, model: CoChangeGraph):
        self.model = model

    def what_else_changes(self, file: str, threshold: float = 0.2) -> List[str]:
        """Given file X changed, what else likely changes?"""
        return [edge.target_file for edge in self.model.get_neighbors(file, threshold)]

    def predict_missing_files(self, changed_files: List[str], top_k: int = 5) -> List[Tuple[str, float, str]]:
        """Predict files missing from a commit, with justification."""
        predictions = self.model.predict_changes(changed_files, top_k)

        # Add justification
        results = []
        for file, score in predictions:
            # Find which changed file most strongly suggests this
            strongest_source = max(
                changed_files,
                key=lambda src: self.model.edges.get(src, {}).get(file, CoChangeEdge(..., confidence=0.0)).confidence
            )
            results.append((file, score, f"Often changes with {strongest_source}"))

        return results

    def find_change_clusters(self, min_cluster_size: int = 3) -> List[Set[str]]:
        """Find groups of files that always change together (Louvain clustering)."""
        # Use existing Louvain algorithm from cortical.analysis
        # Build graph, run community detection
        pass  # TODO: Integration with CorticalTextProcessor
```

### Confidence Scoring

**Multi-signal confidence score:**
```python
def compute_confidence(file_a: str, file_b: str, model: CoChangeGraph) -> float:
    """Compute confidence that file_b will change given file_a changed."""
    edge = model.edges.get(file_a, {}).get(file_b)
    if not edge:
        return 0.0

    # Base signal: normalized co-change probability
    base_confidence = edge.confidence

    # Boost for recent co-changes (recency matters)
    age_days = (datetime.now() - edge.last_co_change).days
    recency_boost = math.exp(-0.01 * age_days)

    # Penalty for high-churn files (they co-change with everything)
    churn_penalty = 1.0 / (1.0 + model.file_metadata[file_b].total_commits / 100)

    # Boost if there's also structural coupling (AST call graph)
    structural_boost = 1.5 if has_structural_dependency(file_a, file_b) else 1.0

    # Combined score
    confidence = base_confidence * recency_boost * churn_penalty * structural_boost
    return min(1.0, confidence)  # Cap at 1.0
```

---

## 5. Evaluation Approach

### Metrics

#### 5.1 Precision & Recall

**Setup:** Split commit history into train (80%) and test (20%).

```python
def evaluate_precision_recall(model: CoChangeGraph, test_commits: List[Commit]):
    """Evaluate precision and recall on held-out commits."""
    total_precision = []
    total_recall = []

    for commit in test_commits:
        changed_files = commit.files

        # Use first file to predict rest
        seed_file = changed_files[0]
        remaining_files = set(changed_files[1:])

        # Predict top K
        predictions = model.predict_changes([seed_file], top_k=10)
        predicted_files = {pred[0] for pred in predictions}

        # Compute metrics
        true_positives = len(predicted_files & remaining_files)
        precision = true_positives / len(predicted_files) if predicted_files else 0
        recall = true_positives / len(remaining_files) if remaining_files else 0

        total_precision.append(precision)
        total_recall.append(recall)

    return {
        'mean_precision': np.mean(total_precision),
        'mean_recall': np.mean(total_recall),
        'f1': 2 * (mean_p * mean_r) / (mean_p + mean_r)
    }
```

**Expected Results (from literature):**
- Precision: 30-50%
- Recall: 50-70%
- **Recall is more important** (missing files is worse than false suggestions)

#### 5.2 Recall@K

**Definition:** What fraction of actual changed files appear in top K predictions?

```python
def recall_at_k(model: CoChangeGraph, test_commits: List[Commit], k_values=[1, 3, 5, 10]):
    """Compute Recall@K for different K values."""
    results = {k: [] for k in k_values}

    for commit in test_commits:
        seed = commit.files[0]
        actual = set(commit.files[1:])

        for k in k_values:
            top_k_preds = model.predict_changes([seed], top_k=k)
            top_k_files = {pred[0] for pred in top_k_preds}

            recall_k = len(top_k_files & actual) / len(actual) if actual else 0
            results[k].append(recall_k)

    return {k: np.mean(scores) for k, scores in results.items()}
```

**Interpretation:**
- Recall@1: First suggestion is correct
- Recall@5: One of top 5 suggestions is correct (most useful for developers)
- Recall@10: Broader safety net

#### 5.3 Temporal Precision

**Question:** Does recency weighting improve results?

**Experiment:**
1. Train model with no temporal weighting (uniform weights)
2. Train model with exponential decay (λ = 0.01)
3. Train model with sliding window (6 months)
4. Compare Recall@5 on same test set

**Hypothesis:** Temporal weighting should improve by 10-20% (based on literature).

#### 5.4 Comparison vs Static Call Graph

**Setup:**
1. **Baseline:** Predict changes using static call graph only
   - If `file_a.py` calls function in `file_b.py`, predict `file_b.py` changes
2. **Co-change model:** Predict using historical co-change only
3. **Hybrid model:** Combine static + co-change with weighted sum

**Metrics:** Precision, Recall, F1 on same test set

**Expected Outcome:**
- Static call graph: High precision, low recall (misses tests, configs, docs)
- Co-change only: Medium precision, high recall (captures everything)
- Hybrid: Best F1 (combines strengths)

### Cross-Validation Strategy

**Time-series cross-validation** (NOT random split, since time matters):
```
Train: [commits 1-100]    Test: [commits 101-120]
Train: [commits 1-120]    Test: [commits 121-140]
Train: [commits 1-140]    Test: [commits 141-160]
...
```

**Why:** Prevents data leakage (can't train on future to predict past).

### Real-World Validation

**Integration with ML Pre-commit Hook:**
- Current system suggests files based on commit message + static file co-occurrence
- Augment with co-change model predictions
- Compare suggestions before/after integration
- Track developer acceptance rate (did they add suggested files?)

**Metrics:**
- Suggestion acceptance rate (baseline vs co-change)
- False positive rate (annoying suggestions)
- Recall (did we catch missing files?)

---

## 6. Integration with SparkCodeIntelligence

### Proposed Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                SparkCodeIntelligence                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌───────────────┐  ┌─────────────────┐  ┌──────────────┐  │
│  │   ASTIndex    │  │   NGramModel    │  │  CoChangeGraph│  │
│  │ (Static)      │  │ (Statistical)   │  │  (Temporal)  │  │
│  └───────┬───────┘  └────────┬────────┘  └──────┬───────┘  │
│          │                   │                    │          │
│          └───────────────────┴────────────────────┘          │
│                              │                               │
│                    ┌─────────▼─────────┐                     │
│                    │  HybridPredictor  │                     │
│                    │  - Static deps    │                     │
│                    │  - N-gram context │                     │
│                    │  - Co-change hist │                     │
│                    └───────────────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

### API Extensions

```python
# In cortical/spark/intelligence.py

class SparkCodeIntelligence:
    def __init__(self, ...):
        self.ast_index = ASTIndex()
        self.ngram_model = NGramModel()
        self.co_change_model = CoChangeGraph()  # NEW

    def train_on_repository(self, repo_path: str):
        """Train all models on codebase."""
        # Existing: AST + N-gram
        self.ast_index.index_directory(repo_path)
        self.ngram_model.train(texts)

        # NEW: Co-change from git history
        commits = parse_git_log(repo_path)
        updater = CoChangeModelUpdater(self.co_change_model)
        for commit in commits:
            updater.process_new_commit(commit.sha, commit.files, commit.date)

    def predict_related_changes(self, changed_files: List[str]) -> List[Tuple[str, float, str]]:
        """Predict what else should change (hybrid model)."""
        # Static dependencies (from AST)
        static_deps = set()
        for file in changed_files:
            static_deps.update(self.ast_index.find_callers(file))
            static_deps.update(self.ast_index.find_imports(file))

        # Co-change predictions (from git history)
        co_change_preds = self.co_change_model.predict_changes(changed_files, top_k=20)

        # Combine with weighted scoring
        combined = {}
        for file, score in co_change_preds:
            boost = 1.5 if file in static_deps else 1.0
            combined[file] = score * boost

        # Rank and return
        ranked = sorted(combined.items(), key=lambda x: -x[1])[:10]

        # Add justifications
        return [(file, score, self._explain_prediction(file, changed_files))
                for file, score in ranked]

    def _explain_prediction(self, predicted_file: str, changed_files: List[str]) -> str:
        """Generate human-readable justification."""
        reasons = []

        # Check static deps
        for src in changed_files:
            if self.ast_index.has_dependency(src, predicted_file):
                reasons.append(f"called by {src}")

        # Check co-change
        for src in changed_files:
            edge = self.co_change_model.edges.get(src, {}).get(predicted_file)
            if edge and edge.confidence > 0.2:
                reasons.append(f"often changes with {src} ({edge.co_change_count} times)")

        return "; ".join(reasons) if reasons else "historical pattern"
```

### CLI Integration

```bash
# New commands for scripts/spark_code_intelligence.py

# Predict related files for current working changes
python scripts/spark_code_intelligence.py predict-changes --staged

# Output:
# Based on your changes to:
#   - cortical/processor/core.py
#   - cortical/processor/compute.py
#
# Suggested related files:
#   1. tests/unit/test_processor_core.py      (0.85 - often changes with core.py; 47 times)
#   2. cortical/processor/query_api.py        (0.72 - called by core.py; often changes with compute.py)
#   3. docs/architecture.md                   (0.45 - often changes with processor/* files)

# Analyze co-change patterns
python scripts/spark_code_intelligence.py co-change-analysis --file cortical/processor/core.py

# Train co-change model
python scripts/spark_code_intelligence.py train --include-git-history --since 2024-01-01
```

---

## 7. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
- [ ] Implement `CoChangeGraph` data structure
- [ ] Git log parser (extract files per commit)
- [ ] Basic co-change matrix builder
- [ ] Unit tests for core data structures

### Phase 2: Temporal Weighting (Week 3)
- [ ] Implement exponential decay
- [ ] Compare decay functions (exponential vs linear vs sliding window)
- [ ] Add incremental update algorithm
- [ ] Tests for temporal weighting

### Phase 3: Evaluation (Week 4)
- [ ] Split commit history (train/test)
- [ ] Implement Precision/Recall/F1 evaluation
- [ ] Implement Recall@K
- [ ] Baseline: static call graph only
- [ ] Generate evaluation report

### Phase 4: Integration (Week 5-6)
- [ ] Integrate with SparkCodeIntelligence
- [ ] Add CLI commands
- [ ] Hybrid model (static + co-change)
- [ ] Update ML pre-commit hook
- [ ] Documentation

### Phase 5: Production (Week 7)
- [ ] Performance optimization (graph indexing)
- [ ] Persistence (save/load co-change model as JSON)
- [ ] CI integration
- [ ] User acceptance testing

---

## 8. Open Research Questions

1. **Commit granularity:** Should we penalize large refactoring commits (they co-change everything)?
2. **File similarity:** Should semantically similar files (e.g., `auth.py` and `authentication.py`) be treated specially?
3. **Cross-repo learning:** Can we train on multiple repos and transfer knowledge?
4. **Directory-level predictions:** Is file-level too granular? Should we predict at directory level?
5. **Negative co-change:** Can we learn "these files NEVER change together" as a signal?

---

## 9. Expected Challenges

1. **Data sparsity:** Small repos may not have enough commits for statistical significance
2. **Overfitting to developers:** If one developer always batches changes, model may overfit
3. **Refactoring detection:** Large refactorings create spurious co-changes
4. **Merge commits:** Should they be included or excluded?
5. **Performance:** Large repos (100k+ commits) may require efficient indexing

**Mitigation strategies:**
- Minimum commit count threshold (e.g., 500 commits)
- Outlier detection (flag commits with >50 files changed)
- Incremental computation (don't reprocess all history on each update)
- Chunk-based persistence (like existing `corpus_chunks/`)

---

## 10. Success Criteria

**Quantitative:**
- Recall@5 > 60% on test set
- Precision > 35% on test set
- 15% improvement over static analysis baseline
- < 100ms prediction latency for typical queries

**Qualitative:**
- Developers find suggestions useful (acceptance rate > 40%)
- False positive rate tolerable (< 3 spurious suggestions per commit)
- Explanations are clear ("Often changes with X" vs opaque scores)

---

## References

### Academic Papers

1. Zimmermann et al. (2005) - "Mining Version Histories to Guide Software Changes" - Seminal APRIORI work
2. Joshi et al. (2007) & Kim et al. (2007) - "Local and Global Recency Weighting Approach to Bug Prediction" - Temporal weighting
3. Canfora & Cerulo (2005) - "Impact analysis by mining software and change request repositories" - MSR for impact analysis
4. "To change or not to change? Modeling software system interactions using Temporal Graphs and Graph Neural Networks" (2023)
5. "Enhancing Change Impact Prediction by Integrating Evolutionary Coupling with Software Change Relationships" (ACM ESEM, 2024)
6. "Graph-based machine learning improves just-in-time defect prediction" (PLOS One, 2023)

### Tools & Frameworks

- **git2net** - Fine-grained co-editing network extraction
- **RepoDriller** - MSR framework
- **CodeScene** - Temporal coupling visualization
- **merge-coupling** - Structural/logical/conceptual coupling measurement

### Links

- [Mining Version Histories to Guide Software Changes (ResearchGate)](https://www.researchgate.net/publication/4083485_Mining_Version_Histories_to_Guide_Software_Changes)
- [Change Coupling Between Software Artifacts (ResearchGate)](https://www.researchgate.net/publication/283802354_Change_Coupling_Between_Software_Artifacts_Learning_from_Past_Changes)
- [Local and Global Recency Weighting (ResearchGate)](https://www.researchgate.net/publication/4252748_Local_and_Global_Recency_Weighting_Approach_to_Bug_Prediction)
- [Temporal Graph Neural Networks (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0950584923002239)
- [Graph-Based ML for JIT Defect Prediction (PLOS One)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0284077)
- [Precision vs Recall in Defect Prediction (ACM)](https://dl.acm.org/doi/10.1145/3655022)
- [Logical Coupling Based on Fine-Grained Changes (ResearchGate)](https://www.researchgate.net/publication/221200077_Logical_Coupling_Based_on_Fine-Grained_Change_Information)
- [Understanding Logical and Structural Coupling (ResearchGate)](https://www.researchgate.net/publication/319285932_Understanding_the_Interplay_between_the_Logical_and_Structural_Coupling_of_Software_Classes)
- [Mining Software Repositories (Wikipedia)](https://en.wikipedia.org/wiki/Mining_software_repositories)
- [Apriori Algorithm (Wikipedia)](https://en.wikipedia.org/wiki/Apriori_algorithm)

---

## Next Steps

1. **Validate with stakeholders:** Review this research summary
2. **Prototype Phase 1:** Implement basic co-change matrix on this repository
3. **Evaluate feasibility:** Check if 403 commits (current repo) is enough for statistical significance
4. **Iterate:** Refine based on early results

---

*This research synthesis provides a foundation for implementing temporal co-change analysis in SparkCodeIntelligence. The hybrid approach (static AST + historical co-change) should significantly improve change impact prediction compared to static analysis alone.*
