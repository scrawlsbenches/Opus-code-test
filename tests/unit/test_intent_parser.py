"""
Unit tests for IntentParser.

Tests cover:
- Conventional commit parsing
- Free-form message parsing
- Reference extraction
- Priority inference
- Edge cases
- Serialization
"""

import unittest
from cortical.spark.intent_parser import IntentParser, IntentResult


class TestConventionalCommitParsing(unittest.TestCase):
    """Test conventional commit format parsing."""

    def test_basic_feat(self):
        """Test basic feat commit."""
        parser = IntentParser()
        result = parser.parse("feat: Add new feature")

        assert result.type == 'feat'
        assert result.scope is None
        assert result.action == 'add'
        assert 'feature' in result.entities
        assert result.confidence == 0.95
        assert result.method == 'conventional'

    def test_basic_fix(self):
        """Test basic fix commit."""
        parser = IntentParser()
        result = parser.parse("fix: Resolve authentication bug")

        assert result.type == 'fix'
        assert result.scope is None
        assert result.action == 'resolve'
        assert result.confidence == 0.95

    def test_basic_refactor(self):
        """Test basic refactor commit."""
        parser = IntentParser()
        result = parser.parse("refactor: Simplify code structure")

        assert result.type == 'refactor'
        assert result.action == 'simplify'

    def test_basic_docs(self):
        """Test basic docs commit."""
        parser = IntentParser()
        result = parser.parse("docs: Update README")

        assert result.type == 'docs'
        assert result.action == 'update'

    def test_basic_test(self):
        """Test basic test commit."""
        parser = IntentParser()
        result = parser.parse("test: Add unit tests")

        assert result.type == 'test'
        assert result.action == 'add'

    def test_basic_chore(self):
        """Test basic chore commit."""
        parser = IntentParser()
        result = parser.parse("chore: Update dependencies")

        assert result.type == 'chore'
        assert result.action == 'update'

    def test_feat_with_scope(self):
        """Test feat commit with scope."""
        parser = IntentParser()
        result = parser.parse("feat(auth): Add OAuth2 support")

        assert result.type == 'feat'
        assert result.scope == 'auth'
        assert result.action == 'add'
        assert 'oauth2' in result.entities or 'oauth' in result.entities
        assert result.confidence == 0.95

    def test_fix_with_scope(self):
        """Test fix commit with scope."""
        parser = IntentParser()
        result = parser.parse("fix(got): Edge deletion in transaction log")

        assert result.type == 'fix'
        assert result.scope == 'got'
        assert 'edge' in result.entities
        assert 'deletion' in result.entities

    def test_breaking_change_exclamation(self):
        """Test breaking change with exclamation mark."""
        parser = IntentParser()
        result = parser.parse("feat!: Remove deprecated API")

        assert result.type == 'feat'
        assert result.breaking is True
        assert result.priority == 'critical'
        assert result.action == 'remove'

    def test_breaking_change_with_scope(self):
        """Test breaking change with scope."""
        parser = IntentParser()
        result = parser.parse("feat(api)!: Change response format")

        assert result.type == 'feat'
        assert result.scope == 'api'
        assert result.breaking is True
        assert result.priority == 'critical'

    def test_empty_scope(self):
        """Test commit with empty scope parentheses."""
        parser = IntentParser()
        result = parser.parse("feat(): Add feature")

        # Empty scope should be parsed as None (no scope)
        assert result.type == 'feat'
        assert result.scope is None or result.scope == ''

    def test_scope_with_special_chars(self):
        """Test scope with dashes and underscores."""
        parser = IntentParser()
        result = parser.parse("feat(spark-utils): Add helper")

        assert result.type == 'feat'
        assert result.scope == 'spark-utils'

    def test_case_insensitive_type(self):
        """Test that commit types are case-insensitive."""
        parser = IntentParser()
        result = parser.parse("FEAT: Add feature")

        assert result.type == 'feat'
        assert result.confidence == 0.95

    def test_all_conventional_types(self):
        """Test all supported conventional commit types."""
        parser = IntentParser()
        types = ['feat', 'fix', 'refactor', 'docs', 'test', 'chore',
                 'perf', 'ci', 'build', 'style', 'revert', 'security']

        for commit_type in types:
            result = parser.parse(f"{commit_type}: Test message")
            assert result.type == commit_type.lower()
            assert result.confidence == 0.95


class TestFreeFormParsing(unittest.TestCase):
    """Test free-form commit message parsing."""

    def test_add_action(self):
        """Test detecting 'add' action."""
        parser = IntentParser()
        result = parser.parse("Add authentication support")

        assert result.action == 'add'
        assert result.type == 'feat'
        assert 'authentication' in result.entities

    def test_fix_action(self):
        """Test detecting 'fix' action."""
        parser = IntentParser()
        result = parser.parse("Fix PageRank convergence issue")

        assert result.action == 'fix'
        assert result.type == 'fix'

    def test_update_action(self):
        """Test detecting 'update' action."""
        parser = IntentParser()
        result = parser.parse("Update ML training data")

        assert result.action == 'update'
        assert result.type == 'chore'

    def test_remove_action(self):
        """Test detecting 'remove' action."""
        parser = IntentParser()
        result = parser.parse("Remove deprecated functions")

        assert result.action == 'remove'
        assert result.type == 'refactor'

    def test_implement_action(self):
        """Test detecting 'implement' action."""
        parser = IntentParser()
        result = parser.parse("Implement n-gram model")

        assert result.action == 'implement'
        assert result.type == 'feat'

    def test_optimize_action(self):
        """Test detecting 'optimize' action."""
        parser = IntentParser()
        result = parser.parse("Optimize search performance")

        assert result.action == 'optimize'
        assert result.type == 'perf'

    def test_entity_extraction(self):
        """Test entity extraction from free-form message."""
        parser = IntentParser()
        result = parser.parse("Add SparkSLM n-gram model for predictions")

        assert 'sparkslm' in result.entities
        assert 'ngram' in result.entities or 'gram' in result.entities
        assert 'model' in result.entities
        assert 'predictions' in result.entities

    def test_mixed_case_handling(self):
        """Test handling of mixed case in free-form messages."""
        parser = IntentParser()
        result = parser.parse("Add OAuth2 Authentication")

        # Entities should be lowercased
        assert all(e.islower() for e in result.entities)


class TestReferenceExtraction(unittest.TestCase):
    """Test reference extraction from commit messages."""

    def test_github_issue_reference(self):
        """Test extracting GitHub issue reference."""
        parser = IntentParser()
        result = parser.parse("feat: Add feature\n\nCloses #123")

        assert '123' in result.references

    def test_multiple_github_references(self):
        """Test extracting multiple GitHub references."""
        parser = IntentParser()
        result = parser.parse("fix: Fix bugs\n\nFixes #42, #43, #44")

        assert '42' in result.references
        assert '43' in result.references
        assert '44' in result.references

    def test_task_id_reference(self):
        """Test extracting GoT task ID."""
        parser = IntentParser()
        result = parser.parse("feat: Add feature\n\nRelated: T-20251222-093045-a1b2")

        assert 'T-20251222-093045-a1b2' in result.references

    def test_task_reference_keyword(self):
        """Test extracting Task #N reference."""
        parser = IntentParser()
        result = parser.parse("fix: Fix bug\n\nTask #456")

        assert '456' in result.references

    def test_mixed_references(self):
        """Test extracting multiple reference types."""
        parser = IntentParser()
        result = parser.parse(
            "feat: Add feature\n\n"
            "Closes #123\n"
            "Related: T-20251222-093045-a1b2\n"
            "Task #456"
        )

        assert '123' in result.references
        assert 'T-20251222-093045-a1b2' in result.references
        assert '456' in result.references

    def test_no_duplicate_references(self):
        """Test that duplicate references are removed."""
        parser = IntentParser()
        result = parser.parse("fix: Fix bug\n\nFixes #123\nCloses #123")

        # Should only have one instance of '123'
        assert result.references.count('123') == 1


class TestPriorityInference(unittest.TestCase):
    """Test priority level inference."""

    def test_breaking_change_priority(self):
        """Test that breaking changes get critical priority."""
        parser = IntentParser()
        result = parser.parse("feat!: Breaking change")

        assert result.priority == 'critical'

    def test_security_keyword_priority(self):
        """Test that security keywords trigger critical priority."""
        parser = IntentParser()
        result = parser.parse("fix: Security vulnerability in auth")

        assert result.priority == 'critical'

    def test_urgent_keyword_priority(self):
        """Test that urgent keywords trigger high priority."""
        parser = IntentParser()
        result = parser.parse("fix: Urgent bug blocking release")

        assert result.priority == 'high'

    def test_typo_keyword_priority(self):
        """Test that typo keywords trigger low priority."""
        parser = IntentParser()
        result = parser.parse("fix: Typo in error message")

        assert result.priority == 'low'

    def test_fix_type_default_priority(self):
        """Test that fix type defaults to high priority."""
        parser = IntentParser()
        result = parser.parse("fix: Fix bug")

        assert result.priority == 'high'

    def test_feat_type_default_priority(self):
        """Test that feat type defaults to medium priority."""
        parser = IntentParser()
        result = parser.parse("feat: Add feature")

        assert result.priority == 'medium'

    def test_docs_type_default_priority(self):
        """Test that docs type defaults to low priority."""
        parser = IntentParser()
        result = parser.parse("docs: Update documentation")

        assert result.priority == 'low'

    def test_breaking_change_in_body(self):
        """Test detecting BREAKING CHANGE in body."""
        parser = IntentParser()
        result = parser.parse("feat: Add feature\n\nBREAKING CHANGE: API changed")

        assert result.breaking is True
        assert result.priority == 'critical'


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_empty_message(self):
        """Test parsing empty message."""
        parser = IntentParser()
        result = parser.parse("")

        assert result.type == 'unknown'
        assert result.confidence == 0.0
        assert result.method == 'empty'

    def test_whitespace_only_message(self):
        """Test parsing whitespace-only message."""
        parser = IntentParser()
        result = parser.parse("   \n\t  ")

        assert result.type == 'unknown'
        assert result.confidence == 0.0

    def test_very_long_message(self):
        """Test parsing very long message."""
        parser = IntentParser()
        long_message = "feat: " + "Add " * 100 + "feature"
        result = parser.parse(long_message)

        assert result.type == 'feat'
        assert result.confidence == 0.95

    def test_unicode_characters(self):
        """Test parsing message with unicode characters."""
        parser = IntentParser()
        result = parser.parse("feat: Add 🚀 rocket feature")

        assert result.type == 'feat'
        assert result.confidence == 0.95

    def test_multiline_with_body(self):
        """Test parsing multiline message with body."""
        parser = IntentParser()
        message = """feat(spark): Add n-gram model

        - Implements trigram support
        - Adds smoothing
        - Includes tests

        Closes #42
        """
        result = parser.parse(message)

        assert result.type == 'feat'
        assert result.scope == 'spark'
        assert '42' in result.references

    def test_no_action_verb(self):
        """Test message with no recognizable action verb."""
        parser = IntentParser()
        result = parser.parse("chore: Something happened")

        # Should still parse as chore, action defaults to type when no verb found
        assert result.type == 'chore'
        assert result.action == 'chore'  # Defaults to type when no action verb found

    def test_scope_inference_from_entities(self):
        """Test inferring scope from entities when not in conventional format."""
        parser = IntentParser()
        result = parser.parse("Add feature to spark module")

        assert result.scope == 'spark'  # Inferred from MODULE_KEYWORDS


class TestSerialization(unittest.TestCase):
    """Test serialization to/from dict."""

    def test_to_dict(self):
        """Test converting IntentResult to dict."""
        parser = IntentParser()
        result = parser.parse("feat(auth): Add OAuth2")

        data = result.to_dict()

        assert isinstance(data, dict)
        assert data['type'] == 'feat'
        assert data['scope'] == 'auth'
        assert data['action'] == 'add'
        assert isinstance(data['entities'], list)
        assert isinstance(data['references'], list)

    def test_from_dict(self):
        """Test creating IntentResult from dict."""
        data = {
            'type': 'feat',
            'scope': 'auth',
            'action': 'add',
            'entities': ['oauth2'],
            'description': 'Add OAuth2',
            'breaking': False,
            'priority': 'medium',
            'references': ['123'],
            'confidence': 0.95,
            'method': 'conventional'
        }

        result = IntentResult.from_dict(data)

        assert result.type == 'feat'
        assert result.scope == 'auth'
        assert result.action == 'add'
        assert result.entities == ['oauth2']
        assert result.references == ['123']

    def test_roundtrip_serialization(self):
        """Test to_dict/from_dict roundtrip."""
        parser = IntentParser()
        original = parser.parse("feat(spark): Add n-gram model\n\nCloses #42")

        data = original.to_dict()
        restored = IntentResult.from_dict(data)

        assert restored.type == original.type
        assert restored.scope == original.scope
        assert restored.action == original.action
        assert restored.entities == original.entities
        assert restored.description == original.description
        assert restored.breaking == original.breaking
        assert restored.priority == original.priority
        assert restored.references == original.references
        assert restored.confidence == original.confidence
        assert restored.method == original.method
