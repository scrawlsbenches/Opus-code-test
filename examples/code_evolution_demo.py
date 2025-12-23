#!/usr/bin/env python3
"""
Code Evolution Model Demo
=========================

Demonstrates the three components of the Code Evolution Model:
1. IntentParser - Parse commit messages to extract structured intent
2. DiffTokenizer - Tokenize git diffs with semantic markers
3. CoChangeModel - Predict related files from git history

Usage:
    python examples/code_evolution_demo.py
"""

import sys
from pathlib import Path

# Add project root to path for direct execution
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from datetime import datetime, timedelta

from cortical.spark import (
    IntentParser, IntentResult,
    DiffTokenizer, DiffFile,
    CoChangeModel, CoChangeEdge,
)


def demo_intent_parser():
    """Demonstrate commit message parsing."""
    print("=" * 60)
    print("1. INTENT PARSER DEMO")
    print("=" * 60)
    print()

    parser = IntentParser()

    # Example commit messages
    commits = [
        "feat(auth): Add OAuth2 login with Google provider",
        "fix(query): Handle null values in search expansion",
        "refactor(processor): Split into modular package structure",
        "docs: Update API documentation with examples",
        "perf(analysis): Optimize PageRank computation",
        "feat!: Change authentication API (breaking change)",
        "Add user profile settings page",  # Free-form
        "Fixed the annoying bug in login flow",  # Free-form
    ]

    for msg in commits:
        result = parser.parse(msg)
        print(f"Message: {msg}")
        print(f"  Type: {result.type:10} | Action: {result.action:12} | Priority: {result.priority}")
        print(f"  Scope: {result.scope or 'N/A':10} | Breaking: {result.breaking} | Confidence: {result.confidence:.2f}")
        if result.entities:
            print(f"  Entities: {', '.join(result.entities[:5])}")
        print()

    # Demonstrate reference extraction
    print("Reference Extraction:")
    msg_with_refs = """fix(api): Resolve timeout issues

This fixes the connection timeout problem reported in #123.
Also addresses feedback from #456 and T-20251222-093045-abcd1234.

BREAKING CHANGE: Connection pool size now configurable.
"""
    result = parser.parse(msg_with_refs)
    print(f"  Message: fix(api): Resolve timeout issues...")
    print(f"  References found: {result.references}")
    print(f"  Breaking: {result.breaking}")
    print(f"  Priority: {result.priority}")
    print()


def demo_diff_tokenizer():
    """Demonstrate git diff tokenization."""
    print("=" * 60)
    print("2. DIFF TOKENIZER DEMO")
    print("=" * 60)
    print()

    tokenizer = DiffTokenizer(include_patterns=True)

    # Example git diff
    diff = """diff --git a/cortical/processor.py b/cortical/processor.py
--- a/cortical/processor.py
+++ b/cortical/processor.py
@@ -45,6 +45,12 @@ class CorticalTextProcessor:
     def __init__(self):
         self._cache = {}

+    def process_document(self, doc_id, text):
+        if not text:
+            return None
+        tokens = self._tokenize(text)
+        return self._build_graph(doc_id, tokens)
+
     def _tokenize(self, text):
         return text.lower().split()

diff --git a/tests/test_processor.py b/tests/test_processor.py
new file mode 100644
--- /dev/null
+++ b/tests/test_processor.py
@@ -0,0 +1,15 @@
+import unittest
+from cortical.processor import CorticalTextProcessor
+
+class TestProcessor(unittest.TestCase):
+    def test_process_document(self):
+        processor = CorticalTextProcessor()
+        result = processor.process_document("doc1", "hello world")
+        self.assertIsNotNone(result)
+
+    def test_empty_document(self):
+        processor = CorticalTextProcessor()
+        result = processor.process_document("doc2", "")
+        self.assertIsNone(result)
"""

    # Structured output
    print("Structured Diff Analysis:")
    files = tokenizer.tokenize_structured(diff)
    for f in files:
        print(f"  File: {f.new_path}")
        print(f"    Change type: {f.change_type}")
        print(f"    Language: {f.language}")
        print(f"    Hunks: {len(f.hunks)}")
        for i, hunk in enumerate(f.hunks):
            adds = sum(1 for t in hunk.lines if t.token_type == 'ADD')
            dels = sum(1 for t in hunk.lines if t.token_type == 'DEL')
            print(f"      Hunk {i+1}: +{adds} -{dels} lines")
        print()

    # Flat token output (for N-gram training)
    print("Flat Token Output (first 30 tokens):")
    tokens = tokenizer.tokenize(diff)
    for i, token in enumerate(tokens[:30]):
        if token.startswith('['):
            print(f"  {token}")
        else:
            print(f"    {token[:50]}{'...' if len(token) > 50 else ''}")
    print(f"  ... ({len(tokens)} total tokens)")
    print()

    # Pattern detection
    print("Pattern Detection Examples:")
    patterns = {
        'guard': """diff --git a/api.py b/api.py
@@ -1,3 +1,6 @@
+    if not user:
+        return None
+
     return process(user)""",

        'error': """diff --git a/api.py b/api.py
@@ -1,3 +1,8 @@
+    try:
+        result = api_call()
+    except APIError as e:
+        logger.error(e)
+        raise
     return result""",

        'cache': """diff --git a/api.py b/api.py
@@ -1,3 +1,6 @@
+    if key in self._cache:
+        return self._cache[key]
+    self._cache[key] = compute(key)
     return result""",
    }

    for pattern_name, pattern_diff in patterns.items():
        tokens = tokenizer.tokenize(pattern_diff)
        detected = [t for t in tokens if t.startswith('[PATTERN:')]
        print(f"  {pattern_name.upper()} pattern: {detected[0] if detected else 'Not detected'}")
    print()

    # Adaptive context sizing
    print("Adaptive Context Sizing:")
    print(f"  Small diff (<50 lines):  {DiffTokenizer.adaptive_context_size(30)} context lines")
    print(f"  Medium diff (100 lines): {DiffTokenizer.adaptive_context_size(100)} context lines")
    print(f"  Large diff (500 lines):  {DiffTokenizer.adaptive_context_size(500)} context lines")
    print()


def demo_co_change_model():
    """Demonstrate file co-change prediction."""
    print("=" * 60)
    print("3. CO-CHANGE MODEL DEMO")
    print("=" * 60)
    print()

    model = CoChangeModel(decay_lambda=0.01)
    now = datetime.now()

    # Simulate git history
    commits = [
        # Feature: Authentication
        ('abc001', ['auth/login.py', 'auth/oauth.py', 'tests/test_auth.py'], 60),
        ('abc002', ['auth/login.py', 'auth/session.py'], 55),
        ('abc003', ['auth/oauth.py', 'auth/session.py', 'config/auth.yaml'], 50),

        # Feature: Query system
        ('def001', ['query/search.py', 'query/expansion.py', 'tests/test_search.py'], 45),
        ('def002', ['query/search.py', 'query/ranking.py'], 40),
        ('def003', ['query/expansion.py', 'query/ranking.py', 'tests/test_search.py'], 35),

        # Cross-cutting: Processor depends on query
        ('ghi001', ['processor/core.py', 'query/search.py'], 30),
        ('ghi002', ['processor/core.py', 'query/expansion.py'], 25),

        # Recent changes (higher weight due to temporal decay)
        ('jkl001', ['auth/login.py', 'auth/oauth.py'], 5),
        ('jkl002', ['query/search.py', 'query/ranking.py', 'query/utils.py'], 3),
        ('jkl003', ['processor/core.py', 'processor/compute.py', 'tests/test_processor.py'], 1),
    ]

    print("Loading simulated git history:")
    for sha, files, days_ago in commits:
        timestamp = now - timedelta(days=days_ago)
        model.add_commit(sha, files, timestamp)
        print(f"  {sha}: {len(files)} files, {days_ago} days ago")

    print()
    print(f"Model statistics: {model}")
    print()

    # Predictions
    test_seeds = [
        ['auth/login.py'],
        ['query/search.py'],
        ['processor/core.py'],
        ['auth/login.py', 'auth/oauth.py'],  # Multiple seeds
    ]

    print("File Co-Change Predictions:")
    print("-" * 50)
    for seeds in test_seeds:
        predictions = model.predict(seeds, top_n=5)
        print(f"\nSeed files: {seeds}")
        if predictions:
            for file, confidence in predictions:
                bar = '█' * int(confidence * 20)
                print(f"  {file:40} {confidence:.3f} {bar}")
        else:
            print("  No predictions (unknown files)")

    print()

    # Temporal decay visualization
    print("Temporal Decay Effect:")
    print("-" * 50)
    print("Recent commits have higher influence on predictions.")
    print()

    # Show edge weights for auth files
    edges = model.get_edges_for_file('auth/login.py')
    print(f"Edges for auth/login.py:")
    for edge in sorted(edges, key=lambda e: -e.weighted_score)[:5]:
        other = edge.target_file if edge.source_file == 'auth/login.py' else edge.source_file
        print(f"  -> {other}")
        print(f"     Co-changes: {edge.co_change_count}, Weighted: {edge.weighted_score:.3f}, Confidence: {edge.confidence:.3f}")


def demo_end_to_end():
    """Demonstrate end-to-end workflow."""
    print()
    print("=" * 60)
    print("4. END-TO-END WORKFLOW")
    print("=" * 60)
    print()
    print("Scenario: Analyze a commit and predict related files")
    print("-" * 50)

    # Initialize components
    parser = IntentParser()
    tokenizer = DiffTokenizer()
    co_change = CoChangeModel()

    # Train co-change model with some history
    now = datetime.now()
    history = [
        ('h1', ['api/auth.py', 'api/users.py', 'tests/test_api.py'], 30),
        ('h2', ['api/auth.py', 'middleware/jwt.py'], 25),
        ('h3', ['api/users.py', 'models/user.py', 'tests/test_api.py'], 20),
        ('h4', ['api/auth.py', 'middleware/jwt.py', 'config/security.py'], 10),
    ]
    for sha, files, days in history:
        co_change.add_commit(sha, files, now - timedelta(days=days))

    # New commit to analyze
    commit_msg = "feat(auth): Add JWT refresh token support"
    diff = """diff --git a/api/auth.py b/api/auth.py
--- a/api/auth.py
+++ b/api/auth.py
@@ -25,6 +25,15 @@ class AuthHandler:
     def authenticate(self, token):
         return self._validate(token)

+    def refresh_token(self, refresh_token):
+        if not refresh_token:
+            return None
+        user = self._get_user_from_refresh(refresh_token)
+        if not user:
+            return None
+        return self._generate_tokens(user)
+
     def _validate(self, token):
         pass
"""

    # 1. Parse intent
    print("1. Parse Commit Intent:")
    intent = parser.parse(commit_msg)
    print(f"   Type: {intent.type}, Scope: {intent.scope}, Action: {intent.action}")
    print(f"   Entities: {intent.entities}")
    print()

    # 2. Tokenize diff
    print("2. Tokenize Diff:")
    tokens = tokenizer.tokenize(diff)
    special = [t for t in tokens if t.startswith('[')]
    print(f"   Special tokens: {special}")
    files = tokenizer.tokenize_structured(diff)
    print(f"   Files changed: {[f.new_path for f in files]}")
    print()

    # 3. Predict related files
    print("3. Predict Related Files:")
    changed_files = [f.new_path for f in files]
    predictions = co_change.predict(changed_files, top_n=5)
    print(f"   Based on changes to: {changed_files}")
    print(f"   You might also need to update:")
    for file, conf in predictions:
        print(f"     - {file} (confidence: {conf:.2f})")
    print()

    # 4. Add this commit to history
    print("4. Update Co-Change Model:")
    co_change.add_commit('new_commit', changed_files, now)
    print(f"   Added commit with {len(changed_files)} files")
    print(f"   Model now has {len(co_change._commits)} commits, {len(co_change._edges)} edges")


def main():
    """Run all demos."""
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " CODE EVOLUTION MODEL - DEMO ".center(58) + "║")
    print("║" + " SparkSLM Components for Git History Analysis ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    print()

    demo_intent_parser()
    demo_diff_tokenizer()
    demo_co_change_model()
    demo_end_to_end()

    print()
    print("=" * 60)
    print("Demo complete!")
    print()
    print("For more information, see:")
    print("  - docs/commit-intent-parsing-research.md")
    print("  - docs/diff-tokenization-research.md")
    print("  - docs/code-evolution-co-change-research.md")
    print("  - docs/code-evolution-model-architecture.md")
    print("=" * 60)


if __name__ == '__main__':
    main()
