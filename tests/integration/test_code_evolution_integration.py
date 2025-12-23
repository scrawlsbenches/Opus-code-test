"""
Integration tests for Code Evolution Model components.

Tests that IntentParser, DiffTokenizer, and CoChangeModel work together
in realistic end-to-end scenarios.
"""

import unittest
from datetime import datetime, timedelta

from cortical.spark import (
    IntentParser, IntentResult,
    DiffTokenizer, DiffFile, DiffHunk, DiffToken,
    CoChangeModel, CoChangeEdge, Commit,
)


class TestCodeEvolutionIntegration(unittest.TestCase):
    """Integration tests for the Code Evolution Model."""

    def setUp(self):
        """Set up test fixtures."""
        self.parser = IntentParser()
        self.tokenizer = DiffTokenizer()
        self.co_change = CoChangeModel()

    def test_full_commit_analysis_pipeline(self):
        """Test analyzing a complete commit with intent, diff, and co-change."""
        # Simulate a commit message
        commit_message = """feat(auth): Add OAuth2 login support

Implements OAuth2 authentication flow with Google provider.
Also adds refresh token handling and session management.

Fixes #123
Related: T-20251222-093045-abcd1234
"""

        # Simulate a git diff
        diff = """diff --git a/auth/oauth.py b/auth/oauth.py
new file mode 100644
--- /dev/null
+++ b/auth/oauth.py
@@ -0,0 +1,50 @@
+class OAuth2Handler:
+    def __init__(self, provider):
+        self.provider = provider
+
+    def authenticate(self, code):
+        if not code:
+            return None
+        token = self._exchange_code(code)
+        return token
+
+    def _exchange_code(self, code):
+        # Exchange authorization code for access token
+        pass

diff --git a/auth/session.py b/auth/session.py
--- a/auth/session.py
+++ b/auth/session.py
@@ -10,5 +10,15 @@ class SessionManager:
     def __init__(self):
         self._sessions = {}

+    def create_session(self, user_id, tokens):
+        session_id = self._generate_id()
+        self._sessions[session_id] = {
+            'user_id': user_id,
+            'tokens': tokens,
+            'created_at': datetime.now()
+        }
+        return session_id
+
     def get_session(self, session_id):
         return self._sessions.get(session_id)
"""

        # 1. Parse intent
        intent = self.parser.parse(commit_message)
        self.assertEqual(intent.type, 'feat')
        self.assertEqual(intent.scope, 'auth')
        self.assertIn('add', intent.action)
        self.assertIn('123', intent.references)
        self.assertTrue(intent.confidence > 0.9)

        # 2. Tokenize diff
        tokens = self.tokenizer.tokenize(diff)
        self.assertIn('[FILE_NEW]', tokens)
        self.assertIn('[FILE]', tokens)
        self.assertIn('[HUNK]', tokens)
        self.assertIn('[ADD]', tokens)
        self.assertIn('[LANG:python]', tokens)

        # 3. Extract files for co-change
        files = self.tokenizer.tokenize_structured(diff)
        file_paths = [f.new_path for f in files]
        self.assertEqual(len(file_paths), 2)
        self.assertIn('auth/oauth.py', file_paths)
        self.assertIn('auth/session.py', file_paths)

        # 4. Add to co-change model
        self.co_change.add_commit('abc123', file_paths, datetime.now())

        # Verify the edge was created
        edges = self.co_change.get_edges_for_file('auth/oauth.py')
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].target_file, 'auth/session.py')

    def test_multiple_commit_history_analysis(self):
        """Test analyzing a series of commits to build co-change patterns."""
        # Simulate a development session with multiple commits
        commits = [
            {
                'sha': 'commit1',
                'message': 'feat(processor): Add query expansion',
                'files': ['cortical/processor.py', 'cortical/query.py', 'tests/test_processor.py'],
                'age_days': 30
            },
            {
                'sha': 'commit2',
                'message': 'fix(query): Fix expansion edge case',
                'files': ['cortical/query.py', 'tests/test_query.py'],
                'age_days': 25
            },
            {
                'sha': 'commit3',
                'message': 'feat(processor): Add passage retrieval',
                'files': ['cortical/processor.py', 'cortical/query.py'],
                'age_days': 20
            },
            {
                'sha': 'commit4',
                'message': 'test(processor): Add integration tests',
                'files': ['cortical/processor.py', 'tests/test_processor.py'],
                'age_days': 10
            },
            {
                'sha': 'commit5',
                'message': 'refactor(query): Extract search utils',
                'files': ['cortical/query.py', 'cortical/utils.py'],
                'age_days': 5
            },
        ]

        now = datetime.now()

        # Add all commits
        for commit in commits:
            intent = self.parser.parse(commit['message'])
            timestamp = now - timedelta(days=commit['age_days'])
            self.co_change.add_commit(
                commit['sha'],
                commit['files'],
                timestamp,
                commit['message']
            )

            # Verify intent parsing worked
            self.assertIsNotNone(intent.type)
            self.assertIsNotNone(intent.action)

        # Now predict related files
        predictions = self.co_change.predict(['cortical/processor.py'], top_n=5)

        # query.py and test_processor.py should be top predictions
        predicted_files = [f for f, _ in predictions]
        self.assertIn('cortical/query.py', predicted_files)
        self.assertIn('tests/test_processor.py', predicted_files)

        # Recent commits should have higher weight (query.py -> utils.py is recent)
        if 'cortical/utils.py' in predicted_files:
            idx_utils = predicted_files.index('cortical/utils.py')
            # utils.py is only related through query.py, not processor.py directly
            # so it might not appear

    def test_diff_pattern_detection_with_intent(self):
        """Test that diff patterns align with commit intent."""
        # Guard clause addition
        guard_message = "fix(validation): Add early return for invalid input"
        guard_diff = """diff --git a/validate.py b/validate.py
--- a/validate.py
+++ b/validate.py
@@ -10,5 +10,8 @@ def validate(data):
+    if not data:
+        return None
+
     # Process the data
     result = process(data)
     return result
"""

        intent = self.parser.parse(guard_message)
        tokens = self.tokenizer.tokenize(guard_diff)

        # Intent should be 'fix' type
        self.assertEqual(intent.type, 'fix')
        self.assertIn('add', intent.action)

        # Diff should detect guard pattern
        self.assertIn('[PATTERN:guard]', tokens)

        # Error handling addition
        error_message = "fix(api): Add exception handling for API calls"
        error_diff = """diff --git a/api.py b/api.py
--- a/api.py
+++ b/api.py
@@ -5,3 +5,10 @@ def call_api(endpoint):
-    response = requests.get(endpoint)
-    return response.json()
+    try:
+        response = requests.get(endpoint)
+        return response.json()
+    except requests.RequestException as e:
+        logger.error(f"API call failed: {e}")
+        raise
"""

        intent = self.parser.parse(error_message)
        tokens = self.tokenizer.tokenize(error_diff)

        self.assertEqual(intent.type, 'fix')
        self.assertIn('[PATTERN:error]', tokens)

    def test_breaking_change_analysis(self):
        """Test detection of breaking changes across components."""
        breaking_message = """feat(api)!: Change authentication method

BREAKING CHANGE: The API now requires OAuth2 tokens instead of API keys.
Old API key authentication is no longer supported.

Migration guide: See docs/migration.md
Fixes #456
"""

        intent = self.parser.parse(breaking_message)

        self.assertEqual(intent.type, 'feat')
        self.assertTrue(intent.breaking)
        self.assertEqual(intent.priority, 'critical')
        self.assertIn('456', intent.references)

    def test_serialization_roundtrip(self):
        """Test that all components serialize and deserialize correctly."""
        # Build up some state
        intent = self.parser.parse("feat(core): Add new feature")

        diff = """diff --git a/core.py b/core.py
--- a/core.py
+++ b/core.py
@@ -1,3 +1,5 @@
+def new_feature():
+    pass
"""
        files = self.tokenizer.tokenize_structured(diff)

        self.co_change.add_commit('test1', ['core.py', 'utils.py'])
        self.co_change.add_commit('test2', ['core.py', 'helpers.py'])

        # Serialize everything
        intent_dict = intent.to_dict()
        diff_dict = self.tokenizer.to_dict(files)
        co_change_dict = self.co_change.to_dict()

        # Deserialize
        intent2 = IntentResult.from_dict(intent_dict)
        files2 = DiffTokenizer.from_dict(diff_dict)
        co_change2 = CoChangeModel.from_dict(co_change_dict)

        # Verify intent
        self.assertEqual(intent2.type, intent.type)
        self.assertEqual(intent2.scope, intent.scope)
        self.assertEqual(intent2.action, intent.action)

        # Verify diff
        self.assertEqual(len(files2), len(files))
        self.assertEqual(files2[0].new_path, files[0].new_path)

        # Verify co-change
        self.assertEqual(len(co_change2._commits), len(self.co_change._commits))
        predictions1 = self.co_change.predict(['core.py'])
        predictions2 = co_change2.predict(['core.py'])
        self.assertEqual(len(predictions1), len(predictions2))

    def test_empty_inputs(self):
        """Test handling of empty inputs across components."""
        # Empty commit message
        intent = self.parser.parse("")
        self.assertEqual(intent.type, 'unknown')
        self.assertEqual(intent.confidence, 0.0)

        # Empty diff
        files = self.tokenizer.tokenize_structured("")
        self.assertEqual(files, [])

        tokens = self.tokenizer.tokenize("")
        self.assertEqual(tokens, [])

        # Empty predictions
        predictions = self.co_change.predict([])
        self.assertEqual(predictions, [])

        # Unknown file
        predictions = self.co_change.predict(['unknown_file.py'])
        self.assertEqual(predictions, [])

    def test_language_detection_consistency(self):
        """Test that language detection works for various file types."""
        diff = """diff --git a/src/main.py b/src/main.py
--- a/src/main.py
+++ b/src/main.py
@@ -1,3 +1,4 @@
+import os

diff --git a/src/utils.js b/src/utils.js
--- a/src/utils.js
+++ b/src/utils.js
@@ -1,3 +1,4 @@
+const fs = require('fs');

diff --git a/src/handler.go b/src/handler.go
--- a/src/handler.go
+++ b/src/handler.go
@@ -1,3 +1,4 @@
+import "fmt"

diff --git a/src/lib.rs b/src/lib.rs
--- a/src/lib.rs
+++ b/src/lib.rs
@@ -1,3 +1,4 @@
+use std::io;
"""

        files = self.tokenizer.tokenize_structured(diff)
        languages = {f.new_path: f.language for f in files}

        self.assertEqual(languages['src/main.py'], 'python')
        self.assertEqual(languages['src/utils.js'], 'javascript')
        self.assertEqual(languages['src/handler.go'], 'go')
        self.assertEqual(languages['src/lib.rs'], 'rust')

    def test_commit_type_inference_from_keywords(self):
        """Test that free-form messages correctly infer commit types."""
        test_cases = [
            ("Add new authentication module", "feat"),
            ("Fix null pointer exception in parser", "fix"),
            ("Refactor database connection handling", "refactor"),
            ("Document the API endpoints", "docs"),  # 'document' verb maps to docs
            ("Implement caching for query results", "feat"),
            ("Remove deprecated legacy code", "refactor"),
            ("Optimize search algorithm performance", "perf"),
        ]

        for message, expected_type in test_cases:
            with self.subTest(message=message):
                intent = self.parser.parse(message)
                self.assertEqual(intent.type, expected_type,
                    f"Expected '{expected_type}' for '{message}', got '{intent.type}'")

    def test_temporal_weighting_affects_predictions(self):
        """Test that recent commits have higher influence."""
        now = datetime.now()

        # Add old commit
        self.co_change.add_commit(
            'old_commit',
            ['file_a.py', 'old_related.py'],
            now - timedelta(days=200),  # Old commit
        )

        # Add recent commit
        self.co_change.add_commit(
            'new_commit',
            ['file_a.py', 'new_related.py'],
            now - timedelta(days=5),  # Recent commit
        )

        predictions = self.co_change.predict(['file_a.py'])

        # new_related.py should have higher confidence than old_related.py
        self.assertEqual(len(predictions), 2)
        pred_dict = dict(predictions)

        self.assertGreater(
            pred_dict['new_related.py'],
            pred_dict['old_related.py'],
            "Recent file should have higher confidence"
        )


class TestCodeEvolutionEdgeCases(unittest.TestCase):
    """Edge case tests for Code Evolution Model integration."""

    def test_unicode_in_commit_and_diff(self):
        """Test handling of Unicode characters."""
        parser = IntentParser()
        tokenizer = DiffTokenizer()

        message = "feat(i18n): Add Japanese translations 日本語サポート"
        intent = parser.parse(message)
        self.assertEqual(intent.type, 'feat')
        self.assertEqual(intent.scope, 'i18n')

        diff = """diff --git a/i18n/ja.json b/i18n/ja.json
--- a/i18n/ja.json
+++ b/i18n/ja.json
@@ -1,3 +1,4 @@
+{"greeting": "こんにちは"}
"""
        tokens = tokenizer.tokenize(diff)
        self.assertIn('[FILE]', tokens)

    def test_very_large_commit(self):
        """Test handling of commits with many files."""
        co_change = CoChangeModel()

        # Simulate a large refactoring commit
        files = [f"module_{i}/file_{j}.py" for i in range(10) for j in range(5)]
        co_change.add_commit('large_commit', files)

        # Should create edges between all pairs
        # 50 files -> 50*49/2 = 1225 edges
        self.assertGreater(len(co_change._edges), 1000)

        # Predictions should still work
        predictions = co_change.predict(['module_0/file_0.py'], top_n=10)
        self.assertEqual(len(predictions), 10)

    def test_single_file_commits(self):
        """Test that single-file commits don't create edges."""
        co_change = CoChangeModel()

        co_change.add_commit('single1', ['file_a.py'])
        co_change.add_commit('single2', ['file_b.py'])

        # No edges should be created
        self.assertEqual(len(co_change._edges), 0)

        # Predictions should be empty
        predictions = co_change.predict(['file_a.py'])
        self.assertEqual(predictions, [])


if __name__ == '__main__':
    unittest.main()
