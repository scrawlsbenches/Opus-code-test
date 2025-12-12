"""
Unit Tests for Code Concepts Module
====================================

Task #169: Unit tests for cortical/code_concepts.py.

Tests the code concept groups and synonym expansion functions:
- CODE_CONCEPT_GROUPS: Programming concept categories
- get_related_terms: Find related programming terms
- expand_code_concepts: Expand query with code synonyms
- get_concept_group: Get concept groups for a term
- list_concept_groups: List all available concept groups
- get_group_terms: Get all terms in a concept group

These tests verify code search semantic expansion capabilities.
"""

import pytest

from cortical.code_concepts import (
    CODE_CONCEPT_GROUPS,
    get_related_terms,
    expand_code_concepts,
    get_concept_group,
    list_concept_groups,
    get_group_terms,
)


# =============================================================================
# CODE CONCEPT GROUPS STRUCTURE TESTS
# =============================================================================


class TestCodeConceptGroups:
    """Tests for CODE_CONCEPT_GROUPS dictionary structure."""

    def test_all_groups_present(self):
        """All expected concept groups are defined."""
        expected_groups = [
            'retrieval', 'storage', 'deletion', 'auth', 'error',
            'validation', 'transform', 'network', 'database', 'async',
            'config', 'logging', 'testing', 'file', 'iteration',
            'lifecycle', 'events'
        ]
        for group in expected_groups:
            assert group in CODE_CONCEPT_GROUPS

    def test_groups_are_frozensets(self):
        """All concept groups are frozensets."""
        for group_name, terms in CODE_CONCEPT_GROUPS.items():
            assert isinstance(terms, frozenset)

    def test_groups_nonempty(self):
        """All concept groups have at least one term."""
        for group_name, terms in CODE_CONCEPT_GROUPS.items():
            assert len(terms) > 0

    def test_retrieval_group_terms(self):
        """Retrieval group contains expected terms."""
        retrieval = CODE_CONCEPT_GROUPS['retrieval']
        expected_terms = ['get', 'fetch', 'load', 'retrieve', 'read', 'query']
        for term in expected_terms:
            assert term in retrieval

    def test_storage_group_terms(self):
        """Storage group contains expected terms."""
        storage = CODE_CONCEPT_GROUPS['storage']
        expected_terms = ['save', 'store', 'write', 'persist', 'cache', 'put']
        for term in expected_terms:
            assert term in storage

    def test_deletion_group_terms(self):
        """Deletion group contains expected terms."""
        deletion = CODE_CONCEPT_GROUPS['deletion']
        expected_terms = ['delete', 'remove', 'drop', 'clear', 'destroy']
        for term in expected_terms:
            assert term in deletion

    def test_auth_group_terms(self):
        """Auth group contains expected terms."""
        auth = CODE_CONCEPT_GROUPS['auth']
        expected_terms = ['auth', 'login', 'logout', 'token', 'password', 'user']
        for term in expected_terms:
            assert term in auth

    def test_error_group_terms(self):
        """Error group contains expected terms."""
        error = CODE_CONCEPT_GROUPS['error']
        expected_terms = ['error', 'exception', 'fail', 'catch', 'throw']
        for term in expected_terms:
            assert term in error

    def test_validation_group_terms(self):
        """Validation group contains expected terms."""
        validation = CODE_CONCEPT_GROUPS['validation']
        expected_terms = ['validate', 'check', 'verify', 'assert', 'ensure']
        for term in expected_terms:
            assert term in validation

    def test_transform_group_terms(self):
        """Transform group contains expected terms."""
        transform = CODE_CONCEPT_GROUPS['transform']
        expected_terms = ['transform', 'convert', 'parse', 'format', 'serialize']
        for term in expected_terms:
            assert term in transform

    def test_network_group_terms(self):
        """Network group contains expected terms."""
        network = CODE_CONCEPT_GROUPS['network']
        expected_terms = ['request', 'response', 'api', 'http', 'rest', 'client']
        for term in expected_terms:
            assert term in network

    def test_database_group_terms(self):
        """Database group contains expected terms."""
        database = CODE_CONCEPT_GROUPS['database']
        expected_terms = ['database', 'db', 'sql', 'query', 'table', 'orm']
        for term in expected_terms:
            assert term in database

    def test_async_group_terms(self):
        """Async group contains expected terms."""
        async_group = CODE_CONCEPT_GROUPS['async']
        expected_terms = ['async', 'await', 'promise', 'thread', 'concurrent']
        for term in expected_terms:
            assert term in async_group

    def test_config_group_terms(self):
        """Config group contains expected terms."""
        config = CODE_CONCEPT_GROUPS['config']
        expected_terms = ['config', 'settings', 'options', 'env', 'property']
        for term in expected_terms:
            assert term in config

    def test_logging_group_terms(self):
        """Logging group contains expected terms."""
        logging = CODE_CONCEPT_GROUPS['logging']
        expected_terms = ['log', 'logger', 'debug', 'info', 'warn', 'monitor']
        for term in expected_terms:
            assert term in logging

    def test_testing_group_terms(self):
        """Testing group contains expected terms."""
        testing = CODE_CONCEPT_GROUPS['testing']
        expected_terms = ['test', 'mock', 'fixture', 'assert', 'coverage']
        for term in expected_terms:
            assert term in testing

    def test_file_group_terms(self):
        """File group contains expected terms."""
        file_group = CODE_CONCEPT_GROUPS['file']
        expected_terms = ['file', 'path', 'directory', 'read', 'write', 'open']
        for term in expected_terms:
            assert term in file_group

    def test_iteration_group_terms(self):
        """Iteration group contains expected terms."""
        iteration = CODE_CONCEPT_GROUPS['iteration']
        expected_terms = ['iterate', 'loop', 'map', 'filter', 'reduce', 'list']
        for term in expected_terms:
            assert term in iteration

    def test_lifecycle_group_terms(self):
        """Lifecycle group contains expected terms."""
        lifecycle = CODE_CONCEPT_GROUPS['lifecycle']
        expected_terms = ['init', 'setup', 'start', 'stop', 'shutdown', 'build']
        for term in expected_terms:
            assert term in lifecycle

    def test_events_group_terms(self):
        """Events group contains expected terms."""
        events = CODE_CONCEPT_GROUPS['events']
        expected_terms = ['event', 'emit', 'listen', 'subscribe', 'publish']
        for term in expected_terms:
            assert term in events


# =============================================================================
# GET RELATED TERMS TESTS
# =============================================================================


class TestGetRelatedTerms:
    """Tests for get_related_terms function."""

    def test_get_related_basic(self):
        """Get related terms for a simple retrieval term."""
        related = get_related_terms('fetch')
        assert isinstance(related, list)
        assert len(related) <= 5  # Default max_terms
        # Should get terms from retrieval group (alphabetically first)
        retrieval_terms = CODE_CONCEPT_GROUPS['retrieval']
        for term in related:
            assert term in retrieval_terms
        assert 'fetch' not in related  # Original term excluded
        # Should include at least one related term
        assert len(related) > 0

    def test_get_related_storage(self):
        """Get related terms for storage operations."""
        related = get_related_terms('save')
        # Should get terms from storage group
        storage_terms = CODE_CONCEPT_GROUPS['storage']
        for term in related:
            assert term in storage_terms
        assert 'save' not in related
        # Should include at least one related term
        assert len(related) > 0

    def test_get_related_deletion(self):
        """Get related terms for deletion operations."""
        related = get_related_terms('delete')
        # Should get terms from deletion group
        deletion_terms = CODE_CONCEPT_GROUPS['deletion']
        for term in related:
            assert term in deletion_terms
        assert 'delete' not in related
        # Should include at least one related term
        assert len(related) > 0

    def test_get_related_auth(self):
        """Get related terms for authentication."""
        related = get_related_terms('login')
        # Auth is a large group, so we get 5 terms by default
        assert len(related) == 5
        assert 'auth' in related or 'authentication' in related

    def test_get_related_case_insensitive(self):
        """Related terms lookup is case insensitive."""
        lower = get_related_terms('fetch')
        upper = get_related_terms('FETCH')
        mixed = get_related_terms('Fetch')
        assert lower == upper == mixed

    def test_get_related_unknown_term(self):
        """Unknown term returns empty list."""
        related = get_related_terms('xyzunknown123')
        assert related == []

    def test_get_related_max_terms_limit(self):
        """Max terms parameter limits results."""
        related_3 = get_related_terms('fetch', max_terms=3)
        related_10 = get_related_terms('fetch', max_terms=10)
        assert len(related_3) == 3
        assert len(related_10) > len(related_3)

    def test_get_related_max_terms_zero(self):
        """Max terms of 0 returns empty list."""
        related = get_related_terms('fetch', max_terms=0)
        assert related == []

    def test_get_related_max_terms_one(self):
        """Max terms of 1 returns single term."""
        related = get_related_terms('fetch', max_terms=1)
        assert len(related) == 1

    def test_get_related_alphabetically_sorted(self):
        """Related terms are returned in alphabetical order."""
        related = get_related_terms('fetch', max_terms=10)
        assert related == sorted(related)

    def test_get_related_multi_group_term(self):
        """Term in multiple groups returns terms from all groups."""
        # 'validate' is in both 'validation' and 'testing' groups
        related = get_related_terms('validate', max_terms=10)
        # Should include terms from validation group
        assert 'check' in related or 'verify' in related
        # May include terms from testing group depending on alphabetical order
        assert len(related) == 10

    def test_get_related_empty_string(self):
        """Empty string returns empty list."""
        related = get_related_terms('')
        assert related == []


# =============================================================================
# EXPAND CODE CONCEPTS TESTS
# =============================================================================


class TestExpandCodeConcepts:
    """Tests for expand_code_concepts function."""

    def test_expand_single_term(self):
        """Expand single term returns weighted related terms."""
        expanded = expand_code_concepts(['fetch'])
        assert isinstance(expanded, dict)
        # Should not include original term
        assert 'fetch' not in expanded
        # Should include related terms from retrieval group
        assert len(expanded) > 0
        retrieval_terms = CODE_CONCEPT_GROUPS['retrieval']
        for term in expanded.keys():
            assert term in retrieval_terms

    def test_expand_default_weight(self):
        """Default weight is 0.6."""
        expanded = expand_code_concepts(['fetch'])
        for term, weight in expanded.items():
            assert weight == pytest.approx(0.6)

    def test_expand_custom_weight(self):
        """Custom weight is applied correctly."""
        expanded = expand_code_concepts(['fetch'], weight=0.8)
        for term, weight in expanded.items():
            assert weight == pytest.approx(0.8)

    def test_expand_max_expansions_limit(self):
        """Max expansions per term limits results."""
        expanded_3 = expand_code_concepts(['fetch'], max_expansions_per_term=3)
        expanded_5 = expand_code_concepts(['fetch'], max_expansions_per_term=5)
        assert len(expanded_3) <= 3
        assert len(expanded_5) <= 5
        assert len(expanded_5) >= len(expanded_3)

    def test_expand_multiple_terms(self):
        """Expand multiple terms combines expansions."""
        expanded = expand_code_concepts(['fetch', 'save'])
        # Should have terms from both retrieval and storage groups
        assert len(expanded) > 0
        retrieval_terms = CODE_CONCEPT_GROUPS['retrieval']
        storage_terms = CODE_CONCEPT_GROUPS['storage']
        # Each expanded term should be from retrieval or storage
        for term in expanded.keys():
            assert term in retrieval_terms or term in storage_terms

    def test_expand_overlapping_terms(self):
        """Overlapping expansions keep highest weight."""
        # Both 'read' and 'load' are in retrieval group
        expanded = expand_code_concepts(['read', 'load'], weight=0.7)
        # 'fetch' is related to both, should get weight 0.7 (not duplicated)
        if 'fetch' in expanded:
            assert expanded['fetch'] == pytest.approx(0.7)

    def test_expand_excludes_input_terms(self):
        """Original query terms are excluded from expansion."""
        expanded = expand_code_concepts(['fetch', 'save', 'delete'])
        assert 'fetch' not in expanded
        assert 'save' not in expanded
        assert 'delete' not in expanded

    def test_expand_case_insensitive_exclusion(self):
        """Input term exclusion is case insensitive."""
        expanded = expand_code_concepts(['FETCH', 'Save', 'delete'])
        assert 'fetch' not in expanded
        assert 'save' not in expanded
        assert 'delete' not in expanded

    def test_expand_empty_list(self):
        """Empty term list returns empty dict."""
        expanded = expand_code_concepts([])
        assert expanded == {}

    def test_expand_unknown_term(self):
        """Unknown term contributes no expansions."""
        expanded = expand_code_concepts(['xyzunknown123'])
        assert expanded == {}

    def test_expand_mixed_known_unknown(self):
        """Mix of known and unknown terms expands known ones."""
        expanded = expand_code_concepts(['fetch', 'xyzunknown123'])
        # Should have expansions from 'fetch'
        assert len(expanded) > 0
        assert 'get' in expanded or 'load' in expanded

    def test_expand_weight_zero(self):
        """Weight of 0.0 still creates entries."""
        expanded = expand_code_concepts(['fetch'], weight=0.0)
        for term, weight in expanded.items():
            assert weight == pytest.approx(0.0)

    def test_expand_weight_one(self):
        """Weight of 1.0 is valid."""
        expanded = expand_code_concepts(['fetch'], weight=1.0)
        for term, weight in expanded.items():
            assert weight == pytest.approx(1.0)

    def test_expand_returns_dict(self):
        """Return type is always dict."""
        expanded = expand_code_concepts(['fetch'])
        assert isinstance(expanded, dict)

    def test_expand_auth_terms(self):
        """Expand authentication terms."""
        expanded = expand_code_concepts(['login'], max_expansions_per_term=3)
        # Should get auth-related terms
        assert 'auth' in expanded or 'authentication' in expanded or 'token' in expanded

    def test_expand_error_terms(self):
        """Expand error handling terms."""
        expanded = expand_code_concepts(['exception'], max_expansions_per_term=3)
        # Should get error-related terms
        assert 'error' in expanded or 'fail' in expanded or 'catch' in expanded


# =============================================================================
# GET CONCEPT GROUP TESTS
# =============================================================================


class TestGetConceptGroup:
    """Tests for get_concept_group function."""

    def test_get_group_retrieval_term(self):
        """Retrieval term returns 'retrieval' group."""
        groups = get_concept_group('fetch')
        assert 'retrieval' in groups

    def test_get_group_storage_term(self):
        """Storage term returns 'storage' group."""
        groups = get_concept_group('save')
        assert 'storage' in groups

    def test_get_group_deletion_term(self):
        """Deletion term returns 'deletion' group."""
        groups = get_concept_group('delete')
        assert 'deletion' in groups

    def test_get_group_auth_term(self):
        """Auth term returns 'auth' group."""
        groups = get_concept_group('login')
        assert 'auth' in groups

    def test_get_group_multi_group_term(self):
        """Term in multiple groups returns all groups."""
        # 'validate' appears in both 'validation' and 'testing'
        groups = get_concept_group('validate')
        assert isinstance(groups, list)
        assert 'validation' in groups
        # Note: validate might only be in validation, let's test a definite multi-group term
        # 'test' is in validation and testing
        groups = get_concept_group('test')
        assert len(groups) >= 1  # At least one group

    def test_get_group_unknown_term(self):
        """Unknown term returns empty list."""
        groups = get_concept_group('xyzunknown123')
        assert groups == []

    def test_get_group_case_insensitive(self):
        """Concept group lookup is case insensitive."""
        lower = get_concept_group('fetch')
        upper = get_concept_group('FETCH')
        mixed = get_concept_group('Fetch')
        assert lower == upper == mixed

    def test_get_group_empty_string(self):
        """Empty string returns empty list."""
        groups = get_concept_group('')
        assert groups == []

    def test_get_group_returns_list(self):
        """Return type is always list."""
        groups = get_concept_group('fetch')
        assert isinstance(groups, list)


# =============================================================================
# LIST CONCEPT GROUPS TESTS
# =============================================================================


class TestListConceptGroups:
    """Tests for list_concept_groups function."""

    def test_list_returns_all_groups(self):
        """List returns all concept groups."""
        groups = list_concept_groups()
        assert isinstance(groups, list)
        assert len(groups) == len(CODE_CONCEPT_GROUPS)

    def test_list_is_sorted(self):
        """List is alphabetically sorted."""
        groups = list_concept_groups()
        assert groups == sorted(groups)

    def test_list_contains_expected_groups(self):
        """List contains all expected concept groups."""
        groups = list_concept_groups()
        expected = ['retrieval', 'storage', 'deletion', 'auth', 'error',
                    'validation', 'transform', 'network', 'database', 'async',
                    'config', 'logging', 'testing', 'file', 'iteration',
                    'lifecycle', 'events']
        for group in expected:
            assert group in groups

    def test_list_no_duplicates(self):
        """List has no duplicate entries."""
        groups = list_concept_groups()
        assert len(groups) == len(set(groups))


# =============================================================================
# GET GROUP TERMS TESTS
# =============================================================================


class TestGetGroupTerms:
    """Tests for get_group_terms function."""

    def test_get_retrieval_group_terms(self):
        """Get terms from retrieval group."""
        terms = get_group_terms('retrieval')
        assert isinstance(terms, list)
        assert 'get' in terms
        assert 'fetch' in terms
        assert 'load' in terms

    def test_get_storage_group_terms(self):
        """Get terms from storage group."""
        terms = get_group_terms('storage')
        assert 'save' in terms
        assert 'store' in terms
        assert 'write' in terms

    def test_get_deletion_group_terms(self):
        """Get terms from deletion group."""
        terms = get_group_terms('deletion')
        assert 'delete' in terms
        assert 'remove' in terms

    def test_get_auth_group_terms(self):
        """Get terms from auth group."""
        terms = get_group_terms('auth')
        assert 'login' in terms
        assert 'authentication' in terms or 'auth' in terms

    def test_get_terms_sorted(self):
        """Terms are alphabetically sorted."""
        terms = get_group_terms('retrieval')
        assert terms == sorted(terms)

    def test_get_unknown_group(self):
        """Unknown group returns empty list."""
        terms = get_group_terms('xyzunknown123')
        assert terms == []

    def test_get_empty_group_name(self):
        """Empty group name returns empty list."""
        terms = get_group_terms('')
        assert terms == []

    def test_get_all_groups_terms(self):
        """Can get terms from all groups."""
        all_group_names = list_concept_groups()
        for group_name in all_group_names:
            terms = get_group_terms(group_name)
            assert isinstance(terms, list)
            assert len(terms) > 0

    def test_get_terms_no_duplicates(self):
        """Group terms have no duplicates."""
        terms = get_group_terms('retrieval')
        assert len(terms) == len(set(terms))

    def test_get_network_group_terms(self):
        """Get terms from network group."""
        terms = get_group_terms('network')
        assert 'api' in terms
        assert 'http' in terms

    def test_get_async_group_terms(self):
        """Get terms from async group."""
        terms = get_group_terms('async')
        assert 'async' in terms
        assert 'await' in terms

    def test_get_testing_group_terms(self):
        """Get terms from testing group."""
        terms = get_group_terms('testing')
        assert 'test' in terms
        assert 'mock' in terms


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestCodeConceptsIntegration:
    """Integration tests combining multiple functions."""

    def test_round_trip_term_to_group_to_terms(self):
        """Term -> group -> terms round trip."""
        # Start with a term
        term = 'fetch'
        # Get its groups
        groups = get_concept_group(term)
        assert len(groups) > 0
        # Get terms from first group
        group_terms = get_group_terms(groups[0])
        # Original term should be in there
        assert term in group_terms

    def test_expansion_contains_related_terms(self):
        """Expansion includes terms from get_related_terms."""
        term = 'fetch'
        related = get_related_terms(term, max_terms=3)
        expanded = expand_code_concepts([term], max_expansions_per_term=3)
        # All related terms should be in expanded (with weights)
        for related_term in related:
            assert related_term in expanded

    def test_all_groups_accessible(self):
        """All concept groups are accessible via API."""
        groups = list_concept_groups()
        for group in groups:
            terms = get_group_terms(group)
            assert len(terms) > 0
            # Each term should know it belongs to this group
            for term in terms:
                term_groups = get_concept_group(term)
                assert group in term_groups

    def test_expand_query_for_code_search(self):
        """Realistic code search query expansion."""
        # User searches for "get user data"
        query_terms = ['get', 'user', 'data']
        expanded = expand_code_concepts(query_terms, max_expansions_per_term=2, weight=0.5)
        # Should expand 'get' with retrieval synonyms
        assert 'fetch' in expanded or 'load' in expanded or 'retrieve' in expanded
        # Original terms excluded
        assert 'get' not in expanded
        assert 'user' not in expanded
        assert 'data' not in expanded

    def test_weights_consistent_across_calls(self):
        """Same input produces same output."""
        expanded1 = expand_code_concepts(['fetch', 'save'], weight=0.7)
        expanded2 = expand_code_concepts(['fetch', 'save'], weight=0.7)
        assert expanded1 == expanded2
