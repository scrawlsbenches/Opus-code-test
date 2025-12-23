# Diff Tokenization Research for Language Model Training

**Date:** 2025-12-22
**Context:** Intelligent tokenization strategies for training SparkSLM on git history

---

## Executive Summary

This document synthesizes research on diff tokenization strategies for training language models on git commit history. Based on 2024-2025 academic research and industry practices, we propose a hybrid tokenization approach combining structural awareness (AST-based), semantic understanding, and efficient chunking for N-gram training.

**Key Recommendation:** Implement a multi-level tokenizer that preserves both syntactic structure (via special tokens) and semantic context (via surrounding code), with intelligent chunking to handle large diffs.

---

## 1. Diff Structure Analysis

### 1.1 Git Diff Anatomy

A git diff is hierarchically structured:

```
diff --git a/file.py b/file.py          ← File-level marker
index abc123..def456 100644              ← Blob metadata
--- a/file.py                            ← Old version
+++ b/file.py                            ← New version
@@ -10,7 +10,8 @@ def function_name():  ← Hunk header (line numbers + context)
 context_line_1                          ← Unchanged context
 context_line_2
-removed_line                            ← Deletion
+added_line                              ← Addition
 context_line_3
```

**Key structural elements:**
1. **File markers** (`diff --git`) - separate multi-file diffs
2. **Hunk headers** (`@@`) - provide line number ranges and function context
3. **Change prefixes** (`+`, `-`, ` `) - indicate addition, deletion, unchanged
4. **Context lines** - surrounding unchanged code for semantic grounding

### 1.2 Semantic Meaning of Change Types

| Change Pattern | Semantic Intent | ML Training Value |
|----------------|-----------------|-------------------|
| Pure additions (`+` only) | New functionality, feature addition | HIGH - learn patterns of "what gets added together" |
| Pure deletions (`-` only) | Cleanup, deprecation, removal | MEDIUM - learn what becomes obsolete |
| Paired `-`/`+` (modify) | Refactoring, bug fixes, evolution | HIGHEST - learn transformation patterns |
| Renames (file-level) | Structural refactoring | MEDIUM - learn naming conventions |

**Critical insight:** The *modification* pattern (paired deletions/additions) is the richest signal for learning code evolution. This is where bug fixes, refactoring, and improvements manifest.

### 1.3 Multi-File Diffs and Relationships

Commits often touch multiple files in coordinated ways:
- **Interface + implementation** - changing a function signature requires updating callers
- **Code + tests** - feature additions typically include test coverage
- **Code + docs** - documentation follows implementation

**ML implication:** Cross-file correlation is valuable but challenging for token-based models. Requires either:
1. Flattening all files into a single sequence (loses structure)
2. Training separate models for file-level patterns (complexity)
3. Using positional embeddings to distinguish files (our approach)

---

## 2. Tokenization Approaches

Based on research from [SemanticDiff](https://semanticdiff.com/blog/semanticdiff-vs-difftastic/), [GitHub Semantic](https://github.com/github/semantic), and [AST-aware transformers](https://dl.acm.org/doi/10.1145/3696002), we identify four primary approaches:

### 2.1 Token-Level Diff (Traditional)

**Approach:** Prefix each line with change type tokens.

```
[FILE] cortical/processor.py
[HUNK] @@ -10,7 +10,8 @@ def compute_all():
[CTX] def compute_all(self):
[CTX]     """Compute all layers."""
[DEL]     self.compute_tfidf()
[ADD]     if self.is_stale(self.COMP_TFIDF):
[ADD]         self.compute_tfidf()
[CTX]     self.compute_importance()
```

**Pros:**
- Simple to implement
- Preserves exact line-level changes
- Works for any file type (language-agnostic)

**Cons:**
- No understanding of code structure
- Treats whitespace changes same as logic changes
- Misses semantic equivalence (e.g., `1337` vs `0x539`)

**Best for:** Line-level prediction tasks, general-purpose diff models

### 2.2 AST-Level Diff (Structural)

**Approach:** Parse code into Abstract Syntax Trees, compute structural diff.

```
[FILE] cortical/processor.py [LANG:python]
[FUNC] compute_all
  [MOD_STMT] function_call → if_statement
    [DEL_NODE] FunctionCall(self.compute_tfidf)
    [ADD_NODE] IfStmt
      [ADD_NODE] Condition(self.is_stale(self.COMP_TFIDF))
      [ADD_NODE] Body(self.compute_tfidf())
```

**Pros:**
- Semantic equivalence detection (e.g., `x+1` vs `1+x`)
- Refactoring-aware (renames don't look like rewrites)
- Hierarchical structure preserved

**Cons:**
- Language-specific parsers required
- Fails on syntax errors (incomplete code)
- Complex to implement
- Higher computational cost

**Best for:** Refactoring detection, cross-language code models, semantic search

**Academic support:** [ACM TOSEM 2025 paper](https://dl.acm.org/doi/10.1145/3696002) shows AST-based diff tools significantly outperform line-based for detecting semantic-preserving changes.

### 2.3 Semantic Diff (Conceptual)

**Approach:** Abstract changes to their semantic intent, not syntax.

```
[CHANGE] Add staleness check before recomputation
  [PATTERN] guard_condition
  [BEFORE] unconditional_call(compute_tfidf)
  [AFTER] conditional_call(compute_tfidf, condition=is_stale)
  [IMPACT] performance_optimization
```

**Pros:**
- Captures "why" not just "what"
- Enables high-level learning (patterns like "add caching", "guard computation")
- Language-independent concepts

**Cons:**
- Requires sophisticated NLP/ML to extract intent
- Loses fine-grained details
- Subjective interpretation of "semantic meaning"

**Best for:** Commit message generation, automated code review, pattern mining

### 2.4 Hybrid Approach (Recommended)

**Approach:** Combine token-level precision with structural awareness and semantic hints.

```
[FILE:cortical/processor.py] [TYPE:python] [COMMIT_TYPE:refactor]
[HUNK:compute_all] [CHANGE:modify] [IMPACT:performance]
[CTX] def compute_all(self):
[CTX]     """Compute all layers."""
[DEL]     self.compute_tfidf()
[ADD]     if self.is_stale(self.COMP_TFIDF):
[ADD]         self.compute_tfidf()
[PATTERN:guard_addition]
```

**Pros:**
- Preserves exact changes (token-level)
- Provides structural context (AST hints)
- Annotates semantic intent (patterns)
- Extensible for different languages

**Cons:**
- Most complex to implement
- Requires multiple processing stages
- Larger token vocabulary

**Best for:** General-purpose code LLM training, our SparkSLM use case

---

## 3. Key Considerations

### 3.1 Preserving Change Context

Research on [chunking strategies](https://www.pinecone.io/learn/chunking-strategies/) shows that context preservation is critical for model quality.

**Problem:** How much surrounding code should we include?

**Options:**
1. **Fixed context window** - Always include N lines before/after (e.g., git's `-U10`)
2. **Syntactic boundaries** - Include complete function/class definitions
3. **Semantic boundaries** - Include all code referenced by the change

**Recommendation:** Hybrid approach:
- Start with syntactic boundary (complete function)
- If too large (&gt;512 tokens), fall back to fixed window (10 lines)
- Always include hunk header for function context

**Why 512 tokens?** [2024 RAG research](https://www.marktechpost.com/2025/08/30/chunking-vs-tokenization-key-differences-in-ai-text-processing/) shows 512-1024 tokens consistently outperforms smaller/larger chunks for QA tasks.

### 3.2 Handling Renames vs Modifications

Git detects renames but our current implementation may not distinguish them:

```python
# Current hunk structure (from scripts/ml_collector/data_classes.py)
@dataclass
class DiffHunk:
    file: str
    function: Optional[str]
    change_type: str  # add, modify, delete, rename ← "rename" exists!
    start_line: int
    lines_added: List[str]
    lines_removed: List[str]
    context_before: List[str]
    context_after: List[str]
```

**Enhancement needed:** Capture file renames explicitly:
```python
old_path: Optional[str] = None  # For renames/moves
similarity_index: Optional[int] = None  # Git's rename detection confidence
```

**ML benefit:** Renames shouldn't be weighted as heavily as true code changes. They're structural refactoring, not logic evolution.

### 3.3 Weighting Strategies

**Question:** Should recent commits be weighted more than older ones?

**Research insight:** Depends on training objective:

| Objective | Weighting Strategy | Rationale |
|-----------|-------------------|-----------|
| **File prediction** | Recency bias (exponential decay) | Recent patterns more relevant to current work |
| **Pattern learning** | Uniform (all commits equal) | Historical patterns still valid |
| **Bug fix prediction** | Type-based (weight bug fixes higher) | Rare but valuable signal |

**Recommendation for SparkSLM:** Hybrid weighting:
```python
weight = base_weight * recency_factor * type_factor
recency_factor = exp(-age_in_days / 180)  # Half-life of 180 days
type_factor = {
    'fix': 2.0,      # Bug fixes are valuable
    'feat': 1.5,     # Features show evolution
    'refactor': 1.0, # Refactorings are neutral
    'docs': 0.5,     # Docs less relevant for code prediction
}
```

### 3.4 Dealing with Large Diffs

**Problem:** Some commits touch hundreds of lines across dozens of files.

**Current approach** (from ml_data_collector.py:1579):
```python
# Use -U10 for more context (better for ML training)
args = ["diff", "-U10", f"{commit_hash}^", commit_hash]
```

**Challenge:** 10-line context can create massive token sequences for large changes.

**Solutions:**

1. **Hierarchical chunking** - Chunk by file, then by hunk within file
2. **Sliding window with overlap** - Create overlapping chunks to preserve cross-hunk context
3. **Adaptive context** - Reduce context for large diffs, increase for small focused changes
4. **Summarization** - For very large diffs, create summary tokens

**Proposed adaptive strategy:**
```python
if total_lines_changed < 50:
    context_lines = 10  # Rich context for focused changes
elif total_lines_changed < 200:
    context_lines = 5   # Moderate context
else:
    context_lines = 2   # Minimal context for mass changes
```

### 3.5 Handling Merge Commits

**Current handling** (from ml_data_collector.py:1574):
```python
if is_merge:
    args.insert(2, "--first-parent")  # Get meaningful diff
```

**Issue:** Merge commits show *all* changes from merged branch, which may be noisy.

**Recommendation:**
- **Option A:** Skip merge commits entirely (they duplicate individual commits)
- **Option B:** Tag merge commits with `[MERGE]` token, weight them lower
- **Option C:** Extract only *conflict resolution* hunks (requires parsing merge markers)

**Our choice:** Option B - include but downweight, since merges show integration patterns.

---

## 4. Proposed DiffTokenizer Design

### 4.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    DiffTokenizer                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Input:  CommitContext (with hunks)                     │
│                                                          │
│  Pipeline:                                              │
│  1. Structural Analysis    → Extract file/function ctx  │
│  2. Change Classification  → Identify patterns          │
│  3. Context Preservation   → Adaptive windowing         │
│  4. Token Generation       → Emit special tokens        │
│  5. Chunking              → Split for N-gram training   │
│                                                          │
│  Output: List[str] (token sequence)                     │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 4.2 Special Tokens Vocabulary

```python
SPECIAL_TOKENS = {
    # File-level markers
    '[FILE]',      # Start of file diff
    '[FILE_NEW]',  # New file created
    '[FILE_DEL]',  # File deleted
    '[FILE_REN]',  # File renamed/moved

    # Hunk-level markers
    '[HUNK]',      # Start of hunk
    '[FUNC]',      # Function context from @@ header
    '[CLASS]',     # Class context (if parseable)

    # Change type markers
    '[ADD]',       # Line added
    '[DEL]',       # Line removed
    '[MOD]',       # Line modified (paired del+add)
    '[CTX]',       # Context line (unchanged)

    # Semantic pattern markers (optional)
    '[PATTERN:guard]',        # Added conditional check
    '[PATTERN:cache]',        # Added caching
    '[PATTERN:refactor]',     # Code restructuring
    '[PATTERN:bugfix]',       # Bug fix pattern

    # Metadata markers
    '[LANG:python]',   # Programming language
    '[TYPE:feat]',     # Commit type (from conventional commits)
    '[IMPACT:high]',   # Change magnitude
}
```

### 4.3 Example Tokenized Output

**Input (raw diff):**
```diff
diff --git a/cortical/processor.py b/cortical/processor.py
@@ -145,7 +145,9 @@ def compute_all(self):
     def compute_all(self):
         """Compute all layers."""
-        self.compute_tfidf()
+        if self.is_stale(self.COMP_TFIDF):
+            self.compute_tfidf()
         self.compute_importance()
```

**Output (tokenized sequence):**
```
[FILE:cortical/processor.py] [LANG:python] [TYPE:refactor]
[HUNK:compute_all] [IMPACT:low]
[CTX] def compute_all ( self ) :
[CTX] """ Compute all layers . """
[DEL] self . compute_tfidf ( )
[ADD] if self . is_stale ( self . COMP_TFIDF ) :
[ADD] self . compute_tfidf ( )
[PATTERN:guard_addition]
[CTX] self . compute_importance ( )
```

**Key features:**
- Code is tokenized at word level (preserve identifiers)
- Special tokens provide structural context
- Pattern annotation enables high-level learning
- Lightweight enough for N-gram models

### 4.4 Implementation Pseudocode

```python
from typing import List, Dict, Optional
from dataclasses import dataclass
from scripts.ml_collector.data_classes import CommitContext, DiffHunk

@dataclass
class DiffTokenizerConfig:
    """Configuration for diff tokenization."""
    preserve_whitespace: bool = False
    include_patterns: bool = True
    include_language_tags: bool = True
    max_context_lines: int = 10
    adaptive_context: bool = True
    chunk_size: int = 512  # Max tokens per chunk
    chunk_overlap: int = 50  # Overlap between chunks

class DiffTokenizer:
    """
    Tokenize git diffs into sequences suitable for N-gram training.

    Combines structural awareness (special tokens for file/hunk boundaries)
    with semantic hints (pattern detection) and intelligent chunking.
    """

    def __init__(self, config: Optional[DiffTokenizerConfig] = None):
        self.config = config or DiffTokenizerConfig()
        self.code_tokenizer = CodeTokenizer()  # From cortical/spark/tokenizer.py

    def tokenize_commit(self, commit: CommitContext) -> List[str]:
        """
        Tokenize an entire commit into a flat token sequence.

        Args:
            commit: CommitContext with hunks and metadata

        Returns:
            List of tokens ready for N-gram training
        """
        tokens = []

        # Commit-level metadata
        commit_type = self._extract_commit_type(commit.message)
        tokens.append(f'[TYPE:{commit_type}]')

        # Determine context size (adaptive)
        if self.config.adaptive_context:
            total_changes = commit.insertions + commit.deletions
            context_lines = self._adaptive_context_size(total_changes)
        else:
            context_lines = self.config.max_context_lines

        # Process each file
        files_by_name = self._group_hunks_by_file(commit.hunks)

        for filename, hunks in files_by_name.items():
            # File-level marker
            file_type = self._detect_file_type(hunks[0])
            tokens.append(f'[FILE:{filename}]')

            if self.config.include_language_tags:
                lang = self._detect_language(filename)
                tokens.append(f'[LANG:{lang}]')

            # Process hunks
            for hunk in hunks:
                tokens.extend(self._tokenize_hunk(hunk, context_lines))

        return tokens

    def _tokenize_hunk(self, hunk: DiffHunk, max_context: int) -> List[str]:
        """Tokenize a single diff hunk."""
        tokens = []

        # Hunk header
        tokens.append('[HUNK]')
        if hunk.function:
            tokens.append(f'[FUNC:{hunk.function}]')

        # Context before (limited)
        ctx_before = hunk.context_before[-max_context:] if hunk.context_before else []
        for line in ctx_before:
            tokens.append('[CTX]')
            tokens.extend(self.code_tokenizer.tokenize(line))

        # Changes (core content)
        change_tokens = self._tokenize_changes(
            hunk.lines_removed,
            hunk.lines_added,
            hunk.change_type
        )
        tokens.extend(change_tokens)

        # Pattern detection (optional)
        if self.config.include_patterns:
            pattern = self._detect_pattern(hunk)
            if pattern:
                tokens.append(f'[PATTERN:{pattern}]')

        # Context after (limited)
        ctx_after = hunk.context_after[:max_context] if hunk.context_after else []
        for line in ctx_after:
            tokens.append('[CTX]')
            tokens.extend(self.code_tokenizer.tokenize(line))

        return tokens

    def _tokenize_changes(
        self,
        removed: List[str],
        added: List[str],
        change_type: str
    ) -> List[str]:
        """
        Tokenize the actual changed lines.

        For modifications, pair deletions with additions to show transformation.
        """
        tokens = []

        if change_type == 'modify':
            # Pair deletions with additions (transformation learning)
            max_pairs = max(len(removed), len(added))
            for i in range(max_pairs):
                if i < len(removed):
                    tokens.append('[DEL]')
                    tokens.extend(self.code_tokenizer.tokenize(removed[i]))
                if i < len(added):
                    tokens.append('[ADD]')
                    tokens.extend(self.code_tokenizer.tokenize(added[i]))
        else:
            # Pure additions or deletions
            for line in removed:
                tokens.append('[DEL]')
                tokens.extend(self.code_tokenizer.tokenize(line))
            for line in added:
                tokens.append('[ADD]')
                tokens.extend(self.code_tokenizer.tokenize(line))

        return tokens

    def chunk_tokens(
        self,
        tokens: List[str],
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None
    ) -> List[List[str]]:
        """
        Split token sequence into overlapping chunks for training.

        Uses sliding window to preserve context across chunk boundaries.
        """
        chunk_size = chunk_size or self.config.chunk_size
        overlap = overlap or self.config.chunk_overlap

        if len(tokens) <= chunk_size:
            return [tokens]

        chunks = []
        start = 0

        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk = tokens[start:end]
            chunks.append(chunk)

            # Move forward by (chunk_size - overlap)
            start += (chunk_size - overlap)

        return chunks

    # Helper methods

    def _extract_commit_type(self, message: str) -> str:
        """Extract commit type from conventional commit message."""
        # feat: add feature -> 'feat'
        # fix(auth): fix bug -> 'fix'
        import re
        match = re.match(r'^(\w+)(?:\([^)]+\))?:', message)
        return match.group(1) if match else 'other'

    def _group_hunks_by_file(self, hunks: List[DiffHunk]) -> Dict[str, List[DiffHunk]]:
        """Group hunks by filename."""
        from collections import defaultdict
        grouped = defaultdict(list)
        for hunk in hunks:
            grouped[hunk.file].append(hunk)
        return dict(grouped)

    def _detect_language(self, filename: str) -> str:
        """Detect programming language from file extension."""
        ext_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust',
            '.md': 'markdown',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
        }
        import os
        ext = os.path.splitext(filename)[1]
        return ext_map.get(ext, 'unknown')

    def _detect_file_type(self, hunk: DiffHunk) -> str:
        """Detect if file is new, deleted, or modified."""
        # This would need to be passed from git diff metadata
        # For now, infer from change_type
        return hunk.change_type

    def _adaptive_context_size(self, total_changes: int) -> int:
        """Determine context size based on change magnitude."""
        if total_changes < 50:
            return 10  # Rich context for focused changes
        elif total_changes < 200:
            return 5   # Moderate context
        else:
            return 2   # Minimal context for mass changes

    def _detect_pattern(self, hunk: DiffHunk) -> Optional[str]:
        """
        Detect high-level patterns in the change.

        This is heuristic-based pattern matching. Could be enhanced
        with ML-based pattern recognition in the future.
        """
        removed = ' '.join(hunk.lines_removed).lower()
        added = ' '.join(hunk.lines_added).lower()

        # Guard pattern: adding if/check before operation
        if 'if' in added and not ('if' in removed):
            if any(keyword in added for keyword in ['is_stale', 'is_valid', 'exists', 'check']):
                return 'guard_addition'

        # Caching pattern: storing result for reuse
        if any(keyword in added for keyword in ['cache', 'memo', 'store', 'save']):
            return 'cache_addition'

        # Refactoring: extracting to function/method
        if 'def ' in added and len(hunk.lines_added) > 5:
            return 'extract_method'

        # Error handling: adding try/except
        if 'try:' in added or 'except' in added:
            return 'error_handling'

        # Logging: adding debug/info statements
        if any(keyword in added for keyword in ['log.', 'logger.', 'print(']):
            return 'logging_addition'

        return None


class CodeTokenizer:
    """
    Tokenize code at word level, preserving identifiers.

    Reuses logic from cortical/tokenizer.py but adapted for code.
    """

    def tokenize(self, line: str) -> List[str]:
        """
        Tokenize a single line of code.

        Splits on whitespace and punctuation but keeps identifiers intact.
        """
        # Remove leading/trailing whitespace
        line = line.strip()
        if not line:
            return []

        # Split on whitespace and common punctuation
        # But keep identifiers like 'self.compute_tfidf' as multiple tokens
        import re
        # Split on whitespace and most punctuation, but preserve '.'
        tokens = re.findall(r'\b\w+\b|[^\s\w]', line)

        return [t for t in tokens if t.strip()]
```

### 4.5 Integration with SparkSLM Training

**Usage in training pipeline:**

```python
from cortical.spark.ngram import NGramModel
from scripts.ml_data_collector import load_commits
from diff_tokenizer import DiffTokenizer, DiffTokenizerConfig

# Configure tokenizer
config = DiffTokenizerConfig(
    include_patterns=True,
    adaptive_context=True,
    chunk_size=512,
    chunk_overlap=50
)
tokenizer = DiffTokenizer(config)

# Load commits
commits = load_commits(limit=1000)

# Tokenize all commits
all_tokens = []
for commit in commits:
    commit_tokens = tokenizer.tokenize_commit(commit)
    # Chunk if needed
    chunks = tokenizer.chunk_tokens(commit_tokens)
    for chunk in chunks:
        all_tokens.append(chunk)

# Train N-gram model
model = NGramModel(n=3)  # Trigram
for token_seq in all_tokens:
    model.add_sequence(token_seq)

# Model can now predict:
# - Next token given context (for autocompletion)
# - Likely changes given commit message
# - Common patterns in specific files
```

### 4.6 Evaluation Metrics

To assess tokenizer quality, measure:

1. **Compression ratio** - How much does tokenization reduce sequence length?
   - Target: 30-50% reduction vs. raw character-level

2. **Pattern coverage** - What % of changes get pattern labels?
   - Target: >60% for common patterns (guard, cache, error handling)

3. **N-gram model perplexity** - How predictable are token sequences?
   - Lower perplexity = better tokenization
   - Baseline: Character-level perplexity

4. **Downstream task performance** - Does it help file prediction?
   - Measure MRR, Recall@10 on test set
   - Compare against current implementation

---

## 5. Recommendations

### 5.1 Implementation Phases

**Phase 1: Basic Hybrid Tokenizer (1-2 weeks)**
- Implement core token generation with special tokens ([FILE], [HUNK], [ADD], [DEL], [CTX])
- Integrate with existing CodeTokenizer
- Add adaptive context sizing
- Unit tests for edge cases (empty diffs, large diffs, merges)

**Phase 2: Pattern Detection (1 week)**
- Implement heuristic pattern detection (guard, cache, error handling)
- Add commit type extraction from conventional commits
- Evaluate pattern coverage on historical commits

**Phase 3: Chunking & Integration (1 week)**
- Implement sliding window chunking
- Integrate with NGramModel training pipeline
- Compare N-gram perplexity vs. baseline

**Phase 4: Optimization (ongoing)**
- Profile tokenization performance (target: &lt;100ms per commit)
- Add language-specific enhancements (Python decorators, JS async/await)
- Experiment with different chunk sizes
- Evaluate downstream impact on file prediction

### 5.2 Alternative Approaches to Consider

1. **Full AST parsing** - If performance allows, use tree-sitter for true structural diff
   - **Pros:** Semantic equivalence, refactoring detection
   - **Cons:** Complexity, language-specific parsers, fails on syntax errors

2. **LLM-based semantic extraction** - Use small LLM to generate summaries
   - **Pros:** Rich semantic understanding
   - **Cons:** Slow, requires external model, adds dependency

3. **Graph-based representation** - Represent diffs as graph edits
   - **Pros:** Captures relationships, enables graph-based learning
   - **Cons:** Incompatible with N-gram models, requires graph NN

**Verdict:** Stick with hybrid token approach for Phase 1. Evaluate AST parsing in Phase 4 if needed.

---

## 6. References

### Academic Research
- [A Novel Refactoring and Semantic Aware Abstract Syntax Tree Differencing Tool](https://dl.acm.org/doi/10.1145/3696002) - ACM TOSEM 2025
- [Outline, Then Details: Syntactically Guided Coarse-To-Fine Code Generation](https://proceedings.mlr.press/v202/zheng23e/zheng23e.pdf) - ICML 2023

### Industry Tools & Practices
- [SemanticDiff vs. Difftastic: How do they differ?](https://semanticdiff.com/blog/semanticdiff-vs-difftastic/)
- [GitHub Semantic - Parsing source code across many languages](https://github.com/github/semantic)
- [CodeGen: Semantic's improved language support system](https://github.blog/2020-08-04-codegen-semantics-improved-language-support-system/)

### Chunking & Tokenization Research (2024-2025)
- [Chunking Strategies for LLM Applications | Pinecone](https://www.pinecone.io/learn/chunking-strategies/)
- [Tokenization vs Chunking: Choosing the Right Text-Splitting Strategy for AI](https://kiadev.net/news/2025-08-30-tokenization-vs-chunking-ai-text-processing)
- [Chunking vs. Tokenization: Key Differences in AI Text Processing - MarkTechPost](https://www.marktechpost.com/2025/08/30/chunking-vs-tokenization-key-differences-in-ai-text-processing/)
- [Mastering RAG: Advanced Chunking Techniques for LLM Applications](https://galileo.ai/blog/mastering-rag-advanced-chunking-techniques-for-llm-applications)
- [Mastering Chunking Strategies for RAG](https://medium.com/@asimadnan/mastering-chunking-strategies-for-rag-balancing-context-window-and-semantic-relevance-d21f57f6daed)

### Code LLM Resources
- [Awesome-Code-LLM: A curated list of language modeling researches for code](https://github.com/codefuse-ai/Awesome-Code-LLM)
- [Tokenization Changes Meaning in Large Language Models - MIT Press](https://direct.mit.edu/coli/article/51/3/785/128327/Tokenization-Changes-Meaning-in-Large-Language)

---

## Appendix: Current System Analysis

### Existing Diff Collection (scripts/ml_data_collector.py)

**Current implementation:**
```python
def parse_diff_hunks(commit_hash: str, is_merge: bool = False) -> List[Dict]:
    """Parse diff hunks from a commit into structured data."""
    # Use -U10 for more context (better for ML training)
    args = ["diff", "-U10", f"{commit_hash}^", commit_hash]
    if is_merge:
        args.insert(2, "--first-parent")

    diff_output = run_git(args, check=False)

    # Parses into:
    # - file: str
    # - function: Optional[str]  (from @@ header)
    # - change_type: str  (add, modify, delete)
    # - start_line: int
    # - lines_added: List[str]
    # - lines_removed: List[str]
    # - context_before: List[str]
    # - context_after: List[str]
```

**Strengths:**
- Already captures structured hunks with context
- Handles merge commits correctly
- Includes function context from diff headers
- Preserves before/after context (10 lines)

**Gaps:**
- No tokenization - stores raw line strings
- No pattern detection
- No language-specific handling
- No chunking for large diffs
- No special tokens for ML training

**Integration point:** DiffTokenizer should consume `CommitContext.hunks` and produce token sequences.

---

**End of Research Summary**
