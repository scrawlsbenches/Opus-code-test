"""
Tests for code_concepts module.

Tests the programming concept groups and expansion functions
used for semantic code search.
"""

import unittest
from cortical.code_concepts import (
    CODE_CONCEPT_GROUPS,
    get_related_terms,
    expand_code_concepts,
    get_concept_group,
    list_concept_groups,
    get_group_terms,
)


class TestCodeConceptGroups(unittest.TestCase):
    """Test the CODE_CONCEPT_GROUPS structure."""

    def test_groups_exist(self):
        """Test that concept groups are defined."""
        self.assertGreater(len(CODE_CONCEPT_GROUPS), 0)

    def test_retrieval_group(self):
        """Test the retrieval concept group."""
        self.assertIn('retrieval', CODE_CONCEPT_GROUPS)
        retrieval = CODE_CONCEPT_GROUPS['retrieval']
        self.assertIn('get', retrieval)
        self.assertIn('fetch', retrieval)
        self.assertIn('load', retrieval)
        self.assertIn('retrieve', retrieval)

    def test_storage_group(self):
        """Test the storage concept group."""
        self.assertIn('storage', CODE_CONCEPT_GROUPS)
        storage = CODE_CONCEPT_GROUPS['storage']
        self.assertIn('save', storage)
        self.assertIn('store', storage)
        self.assertIn('write', storage)
        self.assertIn('persist', storage)

    def test_auth_group(self):
        """Test the authentication concept group."""
        self.assertIn('auth', CODE_CONCEPT_GROUPS)
        auth = CODE_CONCEPT_GROUPS['auth']
        self.assertIn('login', auth)
        self.assertIn('credentials', auth)
        self.assertIn('token', auth)

    def test_error_group(self):
        """Test the error handling concept group."""
        self.assertIn('error', CODE_CONCEPT_GROUPS)
        error = CODE_CONCEPT_GROUPS['error']
        self.assertIn('exception', error)
        self.assertIn('catch', error)
        self.assertIn('throw', error)

    def test_groups_are_frozensets(self):
        """Test that groups are immutable frozensets."""
        for group_name, terms in CODE_CONCEPT_GROUPS.items():
            self.assertIsInstance(terms, frozenset)


class TestGetRelatedTerms(unittest.TestCase):
    """Test the get_related_terms function."""

    def test_fetch_related_terms(self):
        """Test getting terms related to 'fetch'."""
        related = get_related_terms('fetch', max_terms=10)
        self.assertIn('get', related)
        self.assertIn('load', related)
        self.assertNotIn('fetch', related)  # Should not include input term

    def test_save_related_terms(self):
        """Test getting terms related to 'save'."""
        related = get_related_terms('save', max_terms=12)
        self.assertIn('store', related)
        self.assertIn('write', related)
        self.assertNotIn('save', related)

    def test_unknown_term(self):
        """Test with a term not in any concept group."""
        related = get_related_terms('xyzabc123')
        self.assertEqual(related, [])

    def test_max_terms_limit(self):
        """Test that max_terms limits the output."""
        related = get_related_terms('get', max_terms=3)
        self.assertLessEqual(len(related), 3)

    def test_case_insensitive(self):
        """Test that lookup is case insensitive."""
        related_lower = get_related_terms('fetch')
        related_upper = get_related_terms('FETCH')
        related_mixed = get_related_terms('Fetch')
        self.assertEqual(set(related_lower), set(related_upper))
        self.assertEqual(set(related_lower), set(related_mixed))


class TestExpandCodeConcepts(unittest.TestCase):
    """Test the expand_code_concepts function."""

    def test_expand_single_term(self):
        """Test expanding a single term."""
        expanded = expand_code_concepts(['fetch'], max_expansions_per_term=10)
        self.assertIn('get', expanded)
        self.assertIn('load', expanded)
        self.assertNotIn('fetch', expanded)  # Input terms not in output

    def test_expand_multiple_terms(self):
        """Test expanding multiple terms."""
        expanded = expand_code_concepts(['fetch', 'save'], max_expansions_per_term=10)
        # Should have terms from both retrieval and storage groups
        self.assertIn('get', expanded)
        self.assertIn('store', expanded)

    def test_expand_empty_list(self):
        """Test expanding empty list."""
        expanded = expand_code_concepts([])
        self.assertEqual(expanded, {})

    def test_expand_unknown_terms(self):
        """Test expanding terms not in any group."""
        expanded = expand_code_concepts(['xyzabc123'])
        self.assertEqual(expanded, {})

    def test_weights_are_floats(self):
        """Test that expansion weights are floats."""
        expanded = expand_code_concepts(['fetch'])
        for term, weight in expanded.items():
            self.assertIsInstance(weight, float)
            self.assertGreater(weight, 0.0)
            self.assertLessEqual(weight, 1.0)

    def test_custom_weight(self):
        """Test custom weight parameter."""
        expanded = expand_code_concepts(['fetch'], weight=0.8)
        for term, weight in expanded.items():
            self.assertEqual(weight, 0.8)

    def test_max_expansions_per_term(self):
        """Test limiting expansions per term."""
        expanded = expand_code_concepts(['fetch'], max_expansions_per_term=2)
        self.assertLessEqual(len(expanded), 2)

    def test_no_duplicate_original_terms(self):
        """Test that original terms are not in expansions."""
        terms = ['get', 'fetch', 'load']
        expanded = expand_code_concepts(terms)
        for term in terms:
            self.assertNotIn(term, expanded)


class TestGetConceptGroup(unittest.TestCase):
    """Test the get_concept_group function."""

    def test_single_group_membership(self):
        """Test term that belongs to one group."""
        groups = get_concept_group('fetch')
        self.assertIn('retrieval', groups)

    def test_multiple_group_membership(self):
        """Test term that might belong to multiple groups."""
        # 'validate' is in both 'validation' and possibly 'testing'
        groups = get_concept_group('validate')
        self.assertIn('validation', groups)

    def test_unknown_term(self):
        """Test unknown term returns empty list."""
        groups = get_concept_group('xyzabc123')
        self.assertEqual(groups, [])

    def test_case_insensitive(self):
        """Test case insensitive lookup."""
        groups_lower = get_concept_group('fetch')
        groups_upper = get_concept_group('FETCH')
        self.assertEqual(groups_lower, groups_upper)


class TestListConceptGroups(unittest.TestCase):
    """Test the list_concept_groups function."""

    def test_returns_list(self):
        """Test that function returns a list."""
        groups = list_concept_groups()
        self.assertIsInstance(groups, list)

    def test_contains_known_groups(self):
        """Test that known groups are in the list."""
        groups = list_concept_groups()
        self.assertIn('retrieval', groups)
        self.assertIn('storage', groups)
        self.assertIn('auth', groups)
        self.assertIn('error', groups)

    def test_list_is_sorted(self):
        """Test that list is sorted alphabetically."""
        groups = list_concept_groups()
        self.assertEqual(groups, sorted(groups))


class TestGetGroupTerms(unittest.TestCase):
    """Test the get_group_terms function."""

    def test_retrieval_terms(self):
        """Test getting terms from retrieval group."""
        terms = get_group_terms('retrieval')
        self.assertIn('get', terms)
        self.assertIn('fetch', terms)

    def test_unknown_group(self):
        """Test unknown group returns empty list."""
        terms = get_group_terms('nonexistent_group')
        self.assertEqual(terms, [])

    def test_terms_are_sorted(self):
        """Test that terms are sorted alphabetically."""
        terms = get_group_terms('retrieval')
        self.assertEqual(terms, sorted(terms))


class TestQueryExpansionIntegration(unittest.TestCase):
    """Test code concepts integration with query expansion."""

    def setUp(self):
        """Set up test processor."""
        from cortical import CorticalTextProcessor
        self.processor = CorticalTextProcessor()
        # Use terms that won't be filtered as stop words
        self.processor.process_document("doc1", """
            The retrieve function obtains user information from the database.
            It will fetch data and load settings internally.
            The query method returns user profiles.
        """)
        self.processor.process_document("doc2", """
            The persist function stores user information to the database.
            It handles save operations and caching of user profiles.
            The store method writes data.
        """)
        self.processor.compute_all()

    def test_expand_query_with_code_concepts(self):
        """Test expand_query with use_code_concepts enabled."""
        expanded = self.processor.expand_query(
            "fetch data",
            use_code_concepts=True
        )
        # Should include original terms
        self.assertIn('fetch', expanded)
        self.assertIn('data', expanded)
        # With code concepts enabled, should also include related terms
        # like 'load', 'retrieve' (if expansion finds them)

    def test_expand_query_for_code(self):
        """Test the expand_query_for_code convenience method."""
        expanded = self.processor.expand_query_for_code("fetch data")
        self.assertIn('fetch', expanded)
        self.assertIn('data', expanded)

    def test_code_concepts_adds_synonyms(self):
        """Test that code concepts adds programming synonyms."""
        # Expand 'fetch' with code concepts - not a stop word
        expanded_with_code = self.processor.expand_query(
            "fetch",
            use_code_concepts=True,
            max_expansions=20
        )
        # Should include 'fetch' as original term
        self.assertIn('fetch', expanded_with_code)
        # Code concepts should add related retrieval terms
        # Check that at least one synonym is added
        retrieval_synonyms = {'load', 'retrieve', 'query', 'obtain'}
        has_synonym = any(s in expanded_with_code for s in retrieval_synonyms)
        self.assertTrue(has_synonym, f"Expected synonyms in {expanded_with_code}")

    def test_code_concepts_disabled_by_default(self):
        """Test that code concepts are disabled by default."""
        # This test verifies the parameter exists and doesn't crash
        expanded_default = self.processor.expand_query("fetch")
        self.assertIn('fetch', expanded_default)


if __name__ == '__main__':
    unittest.main()
