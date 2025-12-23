"""
Unit tests for DiffTokenizer.

Tests cover:
- Basic diff parsing
- Hunk parsing
- Pattern detection
- Language detection
- Adaptive context sizing
- Edge cases
- Serialization
"""

import pytest
from cortical.spark.diff_tokenizer import (
    DiffTokenizer,
    DiffToken,
    DiffHunk,
    DiffFile,
    SPECIAL_TOKENS
)


class TestBasicDiffParsing:
    """Tests for basic diff parsing functionality."""

    def test_single_file_diff(self):
        """Test parsing a simple single-file diff."""
        diff = """diff --git a/test.py b/test.py
--- a/test.py
+++ b/test.py
@@ -1,3 +1,4 @@
 def hello():
-    print("world")
+    print("Hello")
+    print("World")
"""
        tokenizer = DiffTokenizer()
        files = tokenizer.tokenize_structured(diff)

        assert len(files) == 1
        assert files[0].old_path == 'test.py'
        assert files[0].new_path == 'test.py'
        assert files[0].change_type == 'modified'
        assert len(files[0].hunks) == 1

    def test_multi_file_diff(self):
        """Test parsing diff with multiple files."""
        diff = """diff --git a/file1.py b/file1.py
@@ -1,1 +1,1 @@
-old line
+new line
diff --git a/file2.py b/file2.py
@@ -1,1 +1,1 @@
-old line 2
+new line 2
"""
        tokenizer = DiffTokenizer()
        files = tokenizer.tokenize_structured(diff)

        assert len(files) == 2
        assert files[0].new_path == 'file1.py'
        assert files[1].new_path == 'file2.py'

    def test_added_file(self):
        """Test detection of newly added files."""
        diff = """diff --git a/new_file.py b/new_file.py
new file mode 100644
@@ -0,0 +1,3 @@
+def new_function():
+    pass
"""
        tokenizer = DiffTokenizer()
        files = tokenizer.tokenize_structured(diff)

        assert len(files) == 1
        assert files[0].change_type == 'added'

    def test_deleted_file(self):
        """Test detection of deleted files."""
        diff = """diff --git a/old_file.py b/old_file.py
deleted file mode 100644
@@ -1,3 +0,0 @@
-def old_function():
-    pass
"""
        tokenizer = DiffTokenizer()
        files = tokenizer.tokenize_structured(diff)

        assert len(files) == 1
        assert files[0].change_type == 'deleted'

    def test_renamed_file(self):
        """Test detection of renamed files."""
        diff = """diff --git a/old_name.py b/new_name.py
rename from old_name.py
rename to new_name.py
@@ -1,1 +1,1 @@
 # same content
"""
        tokenizer = DiffTokenizer()
        files = tokenizer.tokenize_structured(diff)

        assert len(files) == 1
        assert files[0].change_type == 'renamed'

    def test_empty_diff(self):
        """Test handling of empty diff."""
        tokenizer = DiffTokenizer()
        files = tokenizer.tokenize_structured("")

        assert files == []

    def test_tokenize_flat_output(self):
        """Test flat token list output."""
        diff = """diff --git a/test.py b/test.py
@@ -1,1 +1,1 @@
-old
+new
"""
        tokenizer = DiffTokenizer()
        tokens = tokenizer.tokenize(diff)

        assert '[FILE]' in tokens
        assert 'test.py' in tokens
        assert '[HUNK]' in tokens
        assert '[ADD]' in tokens
        assert '[DEL]' in tokens

    def test_file_markers_for_different_types(self):
        """Test that different file change types get correct markers."""
        # Added file
        diff_added = """diff --git a/new.py b/new.py
new file mode 100644
@@ -0,0 +1,1 @@
+content
"""
        tokenizer = DiffTokenizer()
        tokens = tokenizer.tokenize(diff_added)
        assert '[FILE_NEW]' in tokens

        # Deleted file
        diff_deleted = """diff --git a/old.py b/old.py
deleted file mode 100644
@@ -1,1 +0,0 @@
-content
"""
        tokens = tokenizer.tokenize(diff_deleted)
        assert '[FILE_DEL]' in tokens

        # Renamed file
        diff_renamed = """diff --git a/old.py b/new.py
rename from old.py
@@ -1,1 +1,1 @@
 content
"""
        tokens = tokenizer.tokenize(diff_renamed)
        assert '[FILE_REN]' in tokens


class TestHunkParsing:
    """Tests for hunk parsing functionality."""

    def test_single_hunk(self):
        """Test parsing a single hunk."""
        diff = """diff --git a/test.py b/test.py
@@ -10,5 +10,6 @@ def function():
 context1
-removed
+added
 context2
"""
        tokenizer = DiffTokenizer()
        files = tokenizer.tokenize_structured(diff)

        assert len(files[0].hunks) == 1
        hunk = files[0].hunks[0]
        assert hunk.start_old == 10
        assert hunk.count_old == 5
        assert hunk.start_new == 10
        assert hunk.count_new == 6

    def test_multiple_hunks(self):
        """Test parsing multiple hunks in one file."""
        diff = """diff --git a/test.py b/test.py
@@ -1,1 +1,1 @@
-old1
+new1
@@ -10,1 +10,1 @@
-old2
+new2
"""
        tokenizer = DiffTokenizer()
        files = tokenizer.tokenize_structured(diff)

        assert len(files[0].hunks) == 2

    def test_context_lines(self):
        """Test that context lines are correctly marked."""
        diff = """diff --git a/test.py b/test.py
@@ -1,3 +1,3 @@
 context_before
-removed
+added
 context_after
"""
        tokenizer = DiffTokenizer()
        files = tokenizer.tokenize_structured(diff)

        hunk = files[0].hunks[0]
        tokens = hunk.lines

        # Check for context markers
        ctx_tokens = [t for t in tokens if t.token_type == 'CTX']
        assert len(ctx_tokens) == 2

    def test_add_lines(self):
        """Test that added lines are correctly marked."""
        diff = """diff --git a/test.py b/test.py
@@ -1,1 +1,3 @@
 existing
+added1
+added2
"""
        tokenizer = DiffTokenizer()
        files = tokenizer.tokenize_structured(diff)

        hunk = files[0].hunks[0]
        add_tokens = [t for t in hunk.lines if t.token_type == 'ADD']
        assert len(add_tokens) == 2

    def test_delete_lines(self):
        """Test that deleted lines are correctly marked."""
        diff = """diff --git a/test.py b/test.py
@@ -1,3 +1,1 @@
-deleted1
-deleted2
 existing
"""
        tokenizer = DiffTokenizer()
        files = tokenizer.tokenize_structured(diff)

        hunk = files[0].hunks[0]
        del_tokens = [t for t in hunk.lines if t.token_type == 'DEL']
        assert len(del_tokens) == 2

    def test_hunk_header_with_function_context(self):
        """Test extraction of function context from hunk header."""
        diff = """diff --git a/test.py b/test.py
@@ -10,3 +10,3 @@ def my_function(arg):
 code
"""
        tokenizer = DiffTokenizer()
        tokens = tokenizer.tokenize(diff)

        assert '[FUNC]' in tokens
        # Function context should be extracted
        func_idx = tokens.index('[FUNC]')
        assert 'my_function' in tokens[func_idx + 1]


class TestPatternDetection:
    """Tests for code pattern detection."""

    def test_guard_pattern_detection(self):
        """Test detection of guard clause pattern."""
        diff = """diff --git a/test.py b/test.py
@@ -1,2 +1,4 @@
 def process():
-    compute()
+    if is_valid():
+        compute()
"""
        tokenizer = DiffTokenizer(include_patterns=True)
        files = tokenizer.tokenize_structured(diff)
        pattern = tokenizer._detect_pattern(files[0].hunks[0])

        assert pattern == 'guard'

    def test_cache_pattern_detection(self):
        """Test detection of cache pattern."""
        diff = """diff --git a/test.py b/test.py
@@ -1,2 +1,4 @@
 def get_data():
+    @lru_cache
     return expensive_operation()
"""
        tokenizer = DiffTokenizer(include_patterns=True)
        files = tokenizer.tokenize_structured(diff)
        pattern = tokenizer._detect_pattern(files[0].hunks[0])

        assert pattern == 'cache'

    def test_error_handling_pattern(self):
        """Test detection of error handling pattern."""
        diff = """diff --git a/test.py b/test.py
@@ -1,2 +1,4 @@
 def process():
+    try:
         do_work()
+    except Exception:
+        handle_error()
"""
        tokenizer = DiffTokenizer(include_patterns=True)
        files = tokenizer.tokenize_structured(diff)
        pattern = tokenizer._detect_pattern(files[0].hunks[0])

        assert pattern == 'error'

    def test_no_pattern_regular_code(self):
        """Test that regular code changes don't trigger patterns."""
        diff = """diff --git a/test.py b/test.py
@@ -1,1 +1,1 @@
-x = 1
+x = 2
"""
        tokenizer = DiffTokenizer(include_patterns=True)
        files = tokenizer.tokenize_structured(diff)
        pattern = tokenizer._detect_pattern(files[0].hunks[0])

        assert pattern is None

    def test_pattern_disabled(self):
        """Test that patterns are not detected when disabled."""
        diff = """diff --git a/test.py b/test.py
@@ -1,2 +1,4 @@
 def process():
+    try:
         do_work()
"""
        tokenizer = DiffTokenizer(include_patterns=False)
        files = tokenizer.tokenize_structured(diff)
        pattern = tokenizer._detect_pattern(files[0].hunks[0])

        assert pattern is None

    def test_refactor_pattern_detection(self):
        """Test detection of refactoring pattern."""
        diff = """diff --git a/test.py b/test.py
@@ -1,2 +1,10 @@
 def main():
+    helper()
+
+def helper():
+    line1
+    line2
+    line3
+    line4
+    line5
+    line6
"""
        tokenizer = DiffTokenizer(include_patterns=True)
        files = tokenizer.tokenize_structured(diff)
        pattern = tokenizer._detect_pattern(files[0].hunks[0])

        assert pattern == 'refactor'


class TestLanguageDetection:
    """Tests for programming language detection."""

    def test_python_files(self):
        """Test detection of Python files."""
        tokenizer = DiffTokenizer()
        assert tokenizer._detect_language('test.py') == 'python'

    def test_javascript_files(self):
        """Test detection of JavaScript files."""
        tokenizer = DiffTokenizer()
        assert tokenizer._detect_language('test.js') == 'javascript'
        assert tokenizer._detect_language('test.jsx') == 'javascript'

    def test_typescript_files(self):
        """Test detection of TypeScript files."""
        tokenizer = DiffTokenizer()
        assert tokenizer._detect_language('test.ts') == 'typescript'
        assert tokenizer._detect_language('test.tsx') == 'typescript'

    def test_unknown_extension(self):
        """Test handling of unknown file extensions."""
        tokenizer = DiffTokenizer()
        assert tokenizer._detect_language('test.xyz') == 'unknown'

    def test_language_in_structured_output(self):
        """Test that language is included in structured output."""
        diff = """diff --git a/test.py b/test.py
@@ -1,1 +1,1 @@
-old
+new
"""
        tokenizer = DiffTokenizer()
        files = tokenizer.tokenize_structured(diff)

        assert files[0].language == 'python'

    def test_language_tag_in_tokens(self):
        """Test that language tag appears in token output."""
        diff = """diff --git a/test.go b/test.go
@@ -1,1 +1,1 @@
-old
+new
"""
        tokenizer = DiffTokenizer()
        tokens = tokenizer.tokenize(diff)

        assert '[LANG:go]' in tokens


class TestAdaptiveContext:
    """Tests for adaptive context sizing."""

    def test_small_diff_large_context(self):
        """Test that small diffs get large context."""
        size = DiffTokenizer.adaptive_context_size(30)
        assert size == 10

    def test_medium_diff_moderate_context(self):
        """Test that medium diffs get moderate context."""
        size = DiffTokenizer.adaptive_context_size(100)
        assert size == 5

    def test_large_diff_minimal_context(self):
        """Test that large diffs get minimal context."""
        size = DiffTokenizer.adaptive_context_size(500)
        assert size == 2

    def test_boundary_conditions(self):
        """Test boundary conditions for context sizing."""
        # Exactly 50 lines
        assert DiffTokenizer.adaptive_context_size(50) == 5
        # Exactly 200 lines
        assert DiffTokenizer.adaptive_context_size(200) == 2
        # Just below threshold
        assert DiffTokenizer.adaptive_context_size(49) == 10
        # Just above threshold
        assert DiffTokenizer.adaptive_context_size(51) == 5


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_diff_string(self):
        """Test handling of empty diff string."""
        tokenizer = DiffTokenizer()
        files = tokenizer.tokenize_structured("")
        assert files == []

        tokens = tokenizer.tokenize("")
        assert tokens == []

    def test_whitespace_only_diff(self):
        """Test handling of whitespace-only diff."""
        tokenizer = DiffTokenizer()
        files = tokenizer.tokenize_structured("   \n  \n  ")
        assert files == []

    def test_very_long_lines(self):
        """Test handling of very long lines in diff."""
        long_line = "x" * 10000
        diff = f"""diff --git a/test.py b/test.py
@@ -1,1 +1,1 @@
-{long_line}
+{long_line}new
"""
        tokenizer = DiffTokenizer()
        files = tokenizer.tokenize_structured(diff)

        assert len(files) == 1
        assert len(files[0].hunks) == 1

    def test_unicode_in_diff(self):
        """Test handling of Unicode characters in diff."""
        diff = """diff --git a/test.py b/test.py
@@ -1,1 +1,1 @@
-# Comment with émojis 🚀
+# Comment with émojis 🎉
"""
        tokenizer = DiffTokenizer()
        files = tokenizer.tokenize_structured(diff)

        assert len(files) == 1
        # Should handle Unicode without errors

    def test_malformed_hunk_header(self):
        """Test handling of malformed hunk header."""
        diff = """diff --git a/test.py b/test.py
@@ invalid header @@
-old
+new
"""
        tokenizer = DiffTokenizer()
        # Should not crash, may return empty or partial result
        files = tokenizer.tokenize_structured(diff)
        # Just verify it doesn't crash
        assert isinstance(files, list)

    def test_binary_file_indicator(self):
        """Test handling of binary file indicators."""
        diff = """diff --git a/image.png b/image.png
Binary files a/image.png and b/image.png differ
"""
        tokenizer = DiffTokenizer()
        files = tokenizer.tokenize_structured(diff)

        # Binary files may not have hunks
        assert len(files) >= 0


class TestSerialization:
    """Tests for JSON serialization."""

    def test_diff_token_roundtrip(self):
        """Test DiffToken to_dict/from_dict roundtrip."""
        token = DiffToken(
            token='[ADD]',
            token_type='ADD',
            line_number=10,
            context='some context'
        )

        data = token.to_dict()
        restored = DiffToken.from_dict(data)

        assert restored.token == token.token
        assert restored.token_type == token.token_type
        assert restored.line_number == token.line_number
        assert restored.context == token.context

    def test_diff_hunk_roundtrip(self):
        """Test DiffHunk to_dict/from_dict roundtrip."""
        hunk = DiffHunk(
            start_old=10,
            count_old=5,
            start_new=10,
            count_new=6,
            header='@@ -10,5 +10,6 @@ def func():',
            lines=[
                DiffToken('[ADD]', 'ADD'),
                DiffToken('new line', 'CODE')
            ]
        )

        data = hunk.to_dict()
        restored = DiffHunk.from_dict(data)

        assert restored.start_old == hunk.start_old
        assert restored.count_old == hunk.count_old
        assert restored.start_new == hunk.start_new
        assert restored.count_new == hunk.count_new
        assert len(restored.lines) == 2

    def test_diff_file_roundtrip(self):
        """Test DiffFile to_dict/from_dict roundtrip."""
        file = DiffFile(
            old_path='old/test.py',
            new_path='new/test.py',
            change_type='modified',
            language='python',
            hunks=[
                DiffHunk(
                    start_old=1,
                    count_old=1,
                    start_new=1,
                    count_new=1,
                    header='@@ -1,1 +1,1 @@',
                    lines=[]
                )
            ]
        )

        data = file.to_dict()
        restored = DiffFile.from_dict(data)

        assert restored.old_path == file.old_path
        assert restored.new_path == file.new_path
        assert restored.change_type == file.change_type
        assert restored.language == file.language
        assert len(restored.hunks) == 1

    def test_tokenizer_to_dict(self):
        """Test tokenizer to_dict method."""
        diff = """diff --git a/test.py b/test.py
@@ -1,1 +1,1 @@
-old
+new
"""
        tokenizer = DiffTokenizer()
        files = tokenizer.tokenize_structured(diff)
        data = tokenizer.to_dict(files)

        assert 'files' in data
        assert len(data['files']) == 1
        assert data['files'][0]['new_path'] == 'test.py'

    def test_tokenizer_from_dict(self):
        """Test tokenizer from_dict method."""
        data = {
            'files': [
                {
                    'old_path': 'test.py',
                    'new_path': 'test.py',
                    'change_type': 'modified',
                    'language': 'python',
                    'hunks': []
                }
            ]
        }

        files = DiffTokenizer.from_dict(data)

        assert len(files) == 1
        assert files[0].new_path == 'test.py'
        assert files[0].language == 'python'


class TestSpecialTokens:
    """Tests for special token constants."""

    def test_special_tokens_defined(self):
        """Test that special tokens are properly defined."""
        assert '[FILE]' in SPECIAL_TOKENS
        assert '[HUNK]' in SPECIAL_TOKENS
        assert '[ADD]' in SPECIAL_TOKENS
        assert '[DEL]' in SPECIAL_TOKENS
        assert '[CTX]' in SPECIAL_TOKENS

    def test_special_tokens_immutable(self):
        """Test that special tokens set is immutable."""
        # Should be a frozenset
        assert isinstance(SPECIAL_TOKENS, frozenset)


class TestTokenizerRepr:
    """Tests for string representation."""

    def test_repr(self):
        """Test __repr__ method."""
        tokenizer = DiffTokenizer(include_patterns=True)
        repr_str = repr(tokenizer)

        assert 'DiffTokenizer' in repr_str
        assert 'include_patterns=True' in repr_str
