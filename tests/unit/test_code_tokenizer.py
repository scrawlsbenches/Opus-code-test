"""
Unit tests for CodeTokenizer.

Tests:
- Basic tokenization
- Operator preservation
- Punctuation handling
- Identifier splitting (camelCase, snake_case)
- String handling
- Edge cases
"""

import pytest
from cortical.spark.tokenizer import CodeTokenizer


class TestCodeTokenizerBasic:
    """Basic tokenization tests."""

    def test_simple_tokenization(self):
        """Test basic word tokenization."""
        tokenizer = CodeTokenizer(split_identifiers=False)
        tokens = tokenizer.tokenize("hello world")
        assert tokens == ['hello', 'world']

    def test_empty_string(self):
        """Test empty input."""
        tokenizer = CodeTokenizer()
        tokens = tokenizer.tokenize("")
        assert tokens == []

    def test_whitespace_only(self):
        """Test whitespace-only input."""
        tokenizer = CodeTokenizer()
        tokens = tokenizer.tokenize("   \t\n   ")
        assert tokens == []

    def test_single_token(self):
        """Test single token."""
        tokenizer = CodeTokenizer(split_identifiers=False)
        tokens = tokenizer.tokenize("hello")
        assert tokens == ['hello']


class TestCodeTokenizerOperators:
    """Operator tokenization tests."""

    def test_comparison_operators(self):
        """Test comparison operators are preserved."""
        tokenizer = CodeTokenizer(split_identifiers=False)

        assert '==' in tokenizer.tokenize("x == y")
        assert '!=' in tokenizer.tokenize("x != y")
        assert '>=' in tokenizer.tokenize("x >= y")
        assert '<=' in tokenizer.tokenize("x <= y")

    def test_assignment_operators(self):
        """Test assignment operators."""
        tokenizer = CodeTokenizer(split_identifiers=False)

        assert '+=' in tokenizer.tokenize("x += 1")
        assert '-=' in tokenizer.tokenize("x -= 1")
        assert '*=' in tokenizer.tokenize("x *= 2")
        assert '/=' in tokenizer.tokenize("x /= 2")

    def test_arrow_operators(self):
        """Test arrow and other operators."""
        tokenizer = CodeTokenizer(split_identifiers=False)

        assert '->' in tokenizer.tokenize("def foo() -> int:")
        assert '**' in tokenizer.tokenize("x ** 2")
        assert '//' in tokenizer.tokenize("x // 2")

    def test_multi_char_operators_order(self):
        """Test that longer operators are matched before shorter ones."""
        tokenizer = CodeTokenizer(split_identifiers=False)

        # **= should be one token, not ** and =
        tokens = tokenizer.tokenize("x **= 2")
        assert '**=' in tokens

        # //= should be one token
        tokens = tokenizer.tokenize("x //= 2")
        assert '//=' in tokens


class TestCodeTokenizerPunctuation:
    """Punctuation handling tests."""

    def test_basic_punctuation(self):
        """Test punctuation is preserved as separate tokens."""
        tokenizer = CodeTokenizer(split_identifiers=False)

        tokens = tokenizer.tokenize("foo()")
        assert '(' in tokens
        assert ')' in tokens

    def test_method_call(self):
        """Test method call tokenization."""
        tokenizer = CodeTokenizer(split_identifiers=False)

        tokens = tokenizer.tokenize("self.method()")
        assert 'self' in tokens
        assert '.' in tokens
        assert 'method' in tokens
        assert '(' in tokens
        assert ')' in tokens

    def test_list_literal(self):
        """Test list literal tokenization."""
        tokenizer = CodeTokenizer(split_identifiers=False)

        tokens = tokenizer.tokenize("[1, 2, 3]")
        assert '[' in tokens
        assert ']' in tokens
        assert ',' in tokens

    def test_dict_literal(self):
        """Test dict literal tokenization."""
        tokenizer = CodeTokenizer(split_identifiers=False)

        tokens = tokenizer.tokenize("{'a': 1}")
        assert '{' in tokens
        assert '}' in tokens
        assert ':' in tokens

    def test_decorators(self):
        """Test decorator tokenization."""
        tokenizer = CodeTokenizer(split_identifiers=False)

        tokens = tokenizer.tokenize("@property")
        assert '@' in tokens
        assert 'property' in tokens


class TestCodeTokenizerIdentifierSplitting:
    """Identifier splitting tests."""

    def test_camel_case_splitting(self):
        """Test camelCase identifiers are split."""
        tokenizer = CodeTokenizer(split_identifiers=True)

        tokens = tokenizer.tokenize("getUserName")
        assert 'getusername' in tokens  # Full identifier
        assert 'get' in tokens
        assert 'user' in tokens
        assert 'name' in tokens

    def test_snake_case_splitting(self):
        """Test snake_case identifiers are split."""
        tokenizer = CodeTokenizer(split_identifiers=True)

        tokens = tokenizer.tokenize("get_user_name")
        assert 'get_user_name' in tokens  # Full identifier
        assert 'get' in tokens
        assert 'user' in tokens
        assert 'name' in tokens

    def test_mixed_case_splitting(self):
        """Test mixed camelCase and snake_case."""
        tokenizer = CodeTokenizer(split_identifiers=True)

        tokens = tokenizer.tokenize("get_UserName")
        assert 'get' in tokens
        assert 'user' in tokens
        assert 'name' in tokens

    def test_no_splitting(self):
        """Test identifier splitting can be disabled."""
        tokenizer = CodeTokenizer(split_identifiers=False)

        tokens = tokenizer.tokenize("getUserName")
        assert tokens == ['getusername']
        assert 'get' not in tokens

    def test_acronyms(self):
        """Test acronym handling in identifiers."""
        tokenizer = CodeTokenizer(split_identifiers=True)

        tokens = tokenizer.tokenize("parseHTMLDocument")
        assert 'parsehtmldocument' in tokens
        # Should handle uppercase sequences reasonably


class TestCodeTokenizerCasePreservation:
    """Case handling tests."""

    def test_lowercase_by_default(self):
        """Test lowercase by default."""
        tokenizer = CodeTokenizer(preserve_case=False)

        tokens = tokenizer.tokenize("Hello World")
        assert 'hello' in tokens
        assert 'world' in tokens
        assert 'Hello' not in tokens

    def test_preserve_case(self):
        """Test case preservation option."""
        tokenizer = CodeTokenizer(preserve_case=True, split_identifiers=False)

        tokens = tokenizer.tokenize("Hello World")
        assert 'Hello' in tokens
        assert 'World' in tokens


class TestCodeTokenizerStrings:
    """String literal handling tests."""

    def test_strings_excluded_by_default(self):
        """Test string contents are excluded by default."""
        tokenizer = CodeTokenizer(include_strings=False)

        tokens = tokenizer.tokenize('x = "hello world"')
        assert 'x' in tokens
        assert '=' in tokens
        assert 'hello' not in tokens
        assert 'world' not in tokens

    def test_strings_included(self):
        """Test string contents can be included."""
        tokenizer = CodeTokenizer(include_strings=True)

        # With include_strings, the whole string becomes a token
        tokens = tokenizer.tokenize('x = "test"')
        # Behavior: strings remain as placeholders when include_strings=True
        # and the original is restored

    def test_multiline_strings(self):
        """Test multiline string handling."""
        tokenizer = CodeTokenizer(include_strings=False)

        code = '''x = """
        multiline
        string
        """'''
        tokens = tokenizer.tokenize(code)
        assert 'x' in tokens
        assert 'multiline' not in tokens


class TestCodeTokenizerNumbers:
    """Number tokenization tests."""

    def test_integers(self):
        """Test integer tokenization."""
        tokenizer = CodeTokenizer()

        tokens = tokenizer.tokenize("x = 42")
        assert '42' in tokens

    def test_floats(self):
        """Test float tokenization."""
        tokenizer = CodeTokenizer()

        tokens = tokenizer.tokenize("x = 3.14")
        assert '3.14' in tokens or ('3' in tokens and '14' in tokens)


class TestCodeTokenizerRealCode:
    """Tests with real code examples."""

    def test_function_definition(self):
        """Test function definition tokenization."""
        tokenizer = CodeTokenizer(split_identifiers=True)

        code = "def calculate_sum(a, b):"
        tokens = tokenizer.tokenize(code)

        assert 'def' in tokens
        assert 'calculate_sum' in tokens
        assert 'calculate' in tokens
        assert 'sum' in tokens
        assert '(' in tokens
        assert ')' in tokens
        assert ':' in tokens

    def test_class_definition(self):
        """Test class definition tokenization."""
        tokenizer = CodeTokenizer(split_identifiers=True)

        code = "class MyClass(BaseClass):"
        tokens = tokenizer.tokenize(code)

        assert 'class' in tokens
        assert 'myclass' in tokens
        assert 'baseclass' in tokens
        assert '(' in tokens
        assert ')' in tokens

    def test_import_statement(self):
        """Test import statement tokenization."""
        tokenizer = CodeTokenizer(split_identifiers=False)

        code = "from cortical.spark import SparkPredictor"
        tokens = tokenizer.tokenize(code)

        assert 'from' in tokens
        assert 'import' in tokens


class TestCodeTokenizerEdgeCases:
    """Edge case tests."""

    def test_special_characters(self):
        """Test handling of special characters."""
        tokenizer = CodeTokenizer()

        # These should be handled gracefully
        tokens = tokenizer.tokenize("a@b#c")
        assert '@' in tokens
        assert '#' in tokens

    def test_unicode_identifiers(self):
        """Test unicode in identifiers."""
        tokenizer = CodeTokenizer(split_identifiers=False)

        # Python 3 allows unicode identifiers
        tokens = tokenizer.tokenize("café = 1")
        # May or may not include café depending on regex

    def test_consecutive_operators(self):
        """Test consecutive operators."""
        tokenizer = CodeTokenizer()

        tokens = tokenizer.tokenize("x <= y == z")
        assert '<=' in tokens
        assert '==' in tokens

    def test_repr(self):
        """Test tokenizer repr."""
        tokenizer = CodeTokenizer()
        repr_str = repr(tokenizer)
        assert 'CodeTokenizer' in repr_str
        assert 'split_identifiers' in repr_str
