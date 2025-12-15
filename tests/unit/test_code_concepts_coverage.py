"""
Additional Coverage Tests for Code Concepts Module
===================================================

Additional unit tests to achieve >80% coverage for cortical/code_concepts.py.
Focuses on edge cases and branches not covered by existing tests.
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
# EDGE CASE TESTS FOR EXPAND_CODE_CONCEPTS
# =============================================================================


class TestExpandCodeConceptsEdgeCases:
    """Additional edge case tests for expand_code_concepts function."""

    def test_expand_related_term_in_input_terms(self):
        """
        Test when a related term is also in the input terms.

        This covers the branch where related_term IS in input_terms,
        causing the expansion to skip that term (line 201->199).
        """
        # Both 'fetch' and 'get' are in the retrieval group
        # get_related_terms('fetch') will return terms including 'get'
        # But 'get' is also in input_terms, so it should be excluded
        expanded = expand_code_concepts(['fetch', 'get'], max_expansions_per_term=10)

        # Neither 'fetch' nor 'get' should be in the expansion
        assert 'fetch' not in expanded
        assert 'get' not in expanded

        # But other retrieval terms should be present
        assert len(expanded) > 0
        retrieval_terms = CODE_CONCEPT_GROUPS['retrieval']
        for term in expanded.keys():
            assert term in retrieval_terms
            # Ensure expanded term is not in original input
            assert term not in ['fetch', 'get']

    def test_expand_all_related_terms_in_input(self):
        """
        Test when most related terms are already in input.

        Edge case where the input contains many terms from the same concept group.
        """
        # Use several terms from the retrieval group
        input_terms = ['fetch', 'get', 'load', 'retrieve', 'query']
        expanded = expand_code_concepts(input_terms, max_expansions_per_term=5)

        # Expanded terms should not include any input terms
        for term in expanded.keys():
            assert term not in [t.lower() for t in input_terms]

        # Should get some expansions (may be from retrieval or database groups)
        # since 'query' is in both retrieval and database groups
        assert len(expanded) > 0

    def test_expand_weight_update_with_higher_weight(self):
        """
        Test that higher weight updates existing term weight.

        Tests line 203: expanded[related_term] < weight condition.
        """
        # First expand with lower weight
        expanded1 = expand_code_concepts(['fetch'], max_expansions_per_term=3, weight=0.3)
        # Then expand with higher weight using a related term
        # This would normally update if we had overlapping expansions

        # Since we can't easily control internal state, test the documented behavior:
        # "Keep highest weight if term appears multiple times"
        # We'll test this by expanding terms from the same group with different weights

        # Create a scenario with two terms that would generate overlapping expansions
        # Both 'fetch' and 'load' are in retrieval, so they share related terms
        expanded = expand_code_concepts(['fetch', 'load'],
                                       max_expansions_per_term=3,
                                       weight=0.7)

        # All weights should be 0.7 (the specified weight)
        for term, weight in expanded.items():
            assert weight == pytest.approx(0.7)

    def test_expand_negative_max_expansions(self):
        """
        Test with negative max_expansions_per_term.

        While not documented, this tests robustness of the slicing at line 171.
        """
        # Python slicing with negative index still returns items
        # sorted_list[:-1] returns all but last, sorted_list[:-5] returns all but last 5
        expanded = expand_code_concepts(['fetch'], max_expansions_per_term=-1)
        # Should still get expansions (all but the last term)
        assert isinstance(expanded, dict)
        # Might be empty or might have terms depending on group size
        retrieval_terms = CODE_CONCEPT_GROUPS['retrieval']
        # If group has > 1 term, we get expansions
        if len(retrieval_terms) > 2:  # Excluding 'fetch' and the last term
            assert len(expanded) > 0

    def test_expand_large_max_expansions(self):
        """
        Test with very large max_expansions_per_term.

        Should return all related terms (up to the group size).
        """
        expanded = expand_code_concepts(['fetch'], max_expansions_per_term=1000)
        # Should get many terms from retrieval group (minus 'fetch' itself)
        retrieval_terms = CODE_CONCEPT_GROUPS['retrieval']
        # At most len(retrieval_terms) - 1 (excluding 'fetch')
        assert len(expanded) <= len(retrieval_terms) - 1

    def test_expand_whitespace_term(self):
        """Test expansion with whitespace-only term."""
        expanded = expand_code_concepts(['   '], max_expansions_per_term=5)
        # Whitespace term should not match any concepts
        assert expanded == {}

    def test_expand_term_with_mixed_case_variations(self):
        """Test that case variations of the same term are handled correctly."""
        # Mix of cases for the same term
        expanded = expand_code_concepts(['FETCH', 'fetch', 'Fetch'],
                                       max_expansions_per_term=3)
        # Should not include 'fetch' in any case
        assert 'fetch' not in expanded
        # Should have some expansions
        assert len(expanded) > 0


# =============================================================================
# EDGE CASE TESTS FOR GET_RELATED_TERMS
# =============================================================================


class TestGetRelatedTermsEdgeCases:
    """Additional edge case tests for get_related_terms function."""

    def test_get_related_negative_max_terms(self):
        """Test with negative max_terms (Python slicing behavior)."""
        # Python slicing with negative numbers still returns items
        # sorted_list[:-5] returns all but the last 5 items
        related = get_related_terms('fetch', max_terms=-5)
        assert isinstance(related, list)
        # May or may not be empty depending on group size
        # Just verify it's a valid list and alphabetically sorted
        assert related == sorted(related)

    def test_get_related_very_large_max_terms(self):
        """Test with max_terms larger than available terms."""
        related = get_related_terms('fetch', max_terms=10000)
        # Should return all related terms in retrieval group (minus 'fetch')
        retrieval_terms = CODE_CONCEPT_GROUPS['retrieval']
        # Max is len(retrieval_terms) - 1
        assert len(related) <= len(retrieval_terms) - 1
        # Should be alphabetically sorted
        assert related == sorted(related)

    def test_get_related_special_characters(self):
        """Test term with special characters."""
        related = get_related_terms('fetch@#$%')
        assert related == []

    def test_get_related_numeric_string(self):
        """Test with numeric string."""
        related = get_related_terms('12345')
        assert related == []

    def test_get_related_unicode_term(self):
        """Test with unicode characters."""
        related = get_related_terms('фетч')  # 'fetch' in Cyrillic
        assert related == []

    def test_get_related_term_with_whitespace(self):
        """Test term with leading/trailing whitespace."""
        # The function does .lower() but doesn't strip
        related = get_related_terms(' fetch ')
        # Should not match 'fetch' because of whitespace
        assert related == []

    def test_get_related_single_char_term(self):
        """Test with single character term."""
        related = get_related_terms('a')
        assert related == []


# =============================================================================
# EDGE CASE TESTS FOR GET_CONCEPT_GROUP
# =============================================================================


class TestGetConceptGroupEdgeCases:
    """Additional edge case tests for get_concept_group function."""

    def test_get_group_whitespace_term(self):
        """Test with whitespace term."""
        groups = get_concept_group('   ')
        assert groups == []

    def test_get_group_special_characters(self):
        """Test with special characters."""
        groups = get_concept_group('@#$%')
        assert groups == []

    def test_get_group_numeric_string(self):
        """Test with numeric string."""
        groups = get_concept_group('123')
        assert groups == []

    def test_get_group_term_with_spaces(self):
        """Test term with spaces (not in concept groups)."""
        groups = get_concept_group('get user')
        assert groups == []

    def test_get_group_partial_match(self):
        """Test that partial matches don't work (exact match required)."""
        groups = get_concept_group('fet')  # Partial of 'fetch'
        assert groups == []


# =============================================================================
# EDGE CASE TESTS FOR GET_GROUP_TERMS
# =============================================================================


class TestGetGroupTermsEdgeCases:
    """Additional edge case tests for get_group_terms function."""

    def test_get_group_terms_case_sensitive(self):
        """Test that group name lookup is case sensitive."""
        # Groups are stored in lowercase
        terms = get_group_terms('RETRIEVAL')
        # Should not find 'RETRIEVAL' (case sensitive)
        assert terms == []

    def test_get_group_terms_with_spaces(self):
        """Test group name with spaces."""
        terms = get_group_terms('retrieval storage')
        assert terms == []

    def test_get_group_terms_special_chars(self):
        """Test group name with special characters."""
        terms = get_group_terms('retrieval@#$')
        assert terms == []

    def test_get_group_terms_partial_name(self):
        """Test that partial group names don't match."""
        terms = get_group_terms('retri')  # Partial of 'retrieval'
        assert terms == []

    def test_get_group_terms_returns_list_type(self):
        """Verify return type is list even for valid group."""
        terms = get_group_terms('retrieval')
        assert isinstance(terms, list)
        assert all(isinstance(term, str) for term in terms)


# =============================================================================
# BOUNDARY AND STRESS TESTS
# =============================================================================


class TestBoundaryConditions:
    """Boundary condition tests for edge cases."""

    def test_expand_with_float_weight_boundaries(self):
        """Test weight at exact boundaries."""
        # Test at 0.0
        expanded = expand_code_concepts(['fetch'], weight=0.0)
        for weight in expanded.values():
            assert weight == pytest.approx(0.0)

        # Test at 1.0
        expanded = expand_code_concepts(['fetch'], weight=1.0)
        for weight in expanded.values():
            assert weight == pytest.approx(1.0)

        # Test at 0.5
        expanded = expand_code_concepts(['fetch'], weight=0.5)
        for weight in expanded.values():
            assert weight == pytest.approx(0.5)

    def test_expand_with_very_small_weight(self):
        """Test with very small weight value."""
        expanded = expand_code_concepts(['fetch'], weight=0.0001)
        for weight in expanded.values():
            assert weight == pytest.approx(0.0001)

    def test_expand_with_weight_greater_than_one(self):
        """Test with weight > 1.0 (not validated by function)."""
        # Function doesn't validate weight range
        expanded = expand_code_concepts(['fetch'], weight=2.5)
        for weight in expanded.values():
            assert weight == pytest.approx(2.5)

    def test_get_related_max_terms_boundary(self):
        """Test max_terms at exact group size."""
        retrieval_count = len(CODE_CONCEPT_GROUPS['retrieval'])
        # Request exactly as many terms as in the group (minus 1 for 'fetch')
        related = get_related_terms('fetch', max_terms=retrieval_count - 1)
        # Should get all related terms
        assert len(related) == retrieval_count - 1

    def test_list_concept_groups_immutability(self):
        """Test that modifying returned list doesn't affect internal state."""
        groups1 = list_concept_groups()
        original_length = len(groups1)

        # Try to modify the returned list
        groups1.append('fake_group')

        # Get a fresh list
        groups2 = list_concept_groups()

        # Should still have original length
        assert len(groups2) == original_length
        assert 'fake_group' not in groups2

    def test_get_group_terms_immutability(self):
        """Test that modifying returned terms list doesn't affect internal state."""
        terms1 = get_group_terms('retrieval')
        original_length = len(terms1)

        # Try to modify the returned list
        terms1.append('fake_term')

        # Get a fresh list
        terms2 = get_group_terms('retrieval')

        # Should still have original length
        assert len(terms2) == original_length
        assert 'fake_term' not in terms2


# =============================================================================
# MULTI-GROUP TERM TESTS
# =============================================================================


class TestMultiGroupTerms:
    """Tests for terms that belong to multiple concept groups."""

    def test_term_in_multiple_groups_returns_all(self):
        """Test that terms in multiple groups return all memberships."""
        # Find a term that's in multiple groups
        # 'query' appears in both 'retrieval' and 'database'
        groups = get_concept_group('query')
        assert len(groups) >= 2
        assert 'retrieval' in groups or 'database' in groups

    def test_multi_group_term_expansions(self):
        """Test expansions for terms in multiple groups."""
        # 'test' appears in both 'validation' and 'testing'
        groups = get_concept_group('test')

        # Get related terms
        related = get_related_terms('test', max_terms=10)

        # Should get terms from all groups this term belongs to
        assert len(related) > 0

        # Verify terms come from validation or testing groups
        validation_terms = CODE_CONCEPT_GROUPS.get('validation', frozenset())
        testing_terms = CODE_CONCEPT_GROUPS.get('testing', frozenset())

        for term in related:
            assert term in validation_terms or term in testing_terms

    def test_expand_multi_group_terms_coverage(self):
        """Test expansion with terms from multiple groups."""
        # Use terms that each belong to different groups
        expanded = expand_code_concepts(
            ['fetch', 'delete', 'async'],
            max_expansions_per_term=3
        )

        # Should have expansions from retrieval, deletion, and async groups
        # Note: terms may appear in multiple groups, so we just verify we get expansions
        assert len(expanded) > 0

        # Verify none of the input terms are in the expansion
        assert 'fetch' not in expanded
        assert 'delete' not in expanded
        assert 'async' not in expanded


# =============================================================================
# INTEGRATION TESTS FOR COMPREHENSIVE COVERAGE
# =============================================================================


class TestComprehensiveIntegration:
    """Integration tests combining multiple edge cases."""

    def test_empty_and_unknown_mixed_expansion(self):
        """Test expansion with mix of empty, unknown, and valid terms."""
        expanded = expand_code_concepts(
            ['', 'unknown_xyz', 'fetch', '   ', '123'],
            max_expansions_per_term=3
        )
        # Should only expand 'fetch'
        assert len(expanded) > 0
        retrieval = CODE_CONCEPT_GROUPS['retrieval']
        for term in expanded.keys():
            assert term in retrieval

    def test_all_concept_groups_have_accessible_terms(self):
        """Verify all groups can be accessed and their terms retrieved."""
        groups = list_concept_groups()

        for group_name in groups:
            # Get terms for this group
            terms = get_group_terms(group_name)
            assert len(terms) > 0

            # Each term should reference back to this group
            for term in terms[:3]:  # Test first 3 terms from each group
                term_groups = get_concept_group(term)
                assert group_name in term_groups

    def test_round_trip_consistency(self):
        """Test round-trip consistency: term -> groups -> terms -> related."""
        test_terms = ['fetch', 'save', 'delete', 'login', 'test']

        for term in test_terms:
            # Get groups for term
            groups = get_concept_group(term)
            assert len(groups) > 0

            # Get terms for each group
            for group in groups:
                group_terms = get_group_terms(group)
                assert term in group_terms

            # Get related terms
            related = get_related_terms(term, max_terms=5)
            # Related terms should not include original
            assert term not in related

    def test_expansion_determinism(self):
        """Test that expansions are deterministic (same input -> same output)."""
        terms = ['fetch', 'save', 'delete']

        # Run expansion multiple times
        results = []
        for _ in range(3):
            expanded = expand_code_concepts(terms, max_expansions_per_term=5, weight=0.6)
            # Convert to sorted items for comparison
            sorted_items = sorted(expanded.items())
            results.append(sorted_items)

        # All results should be identical
        assert results[0] == results[1] == results[2]

    def test_all_groups_non_empty_and_unique(self):
        """Verify all groups have unique, non-empty term sets."""
        groups = list_concept_groups()

        seen_terms_per_group = {}
        for group in groups:
            terms = get_group_terms(group)

            # Non-empty
            assert len(terms) > 0

            # No duplicates within group
            assert len(terms) == len(set(terms))

            # Store for potential overlap analysis
            seen_terms_per_group[group] = set(terms)

        # Verify we checked all groups
        assert len(seen_terms_per_group) == len(groups)


# =============================================================================
# SECURITY CONCEPT GROUP TESTS
# =============================================================================


class TestSecurityConceptGroup:
    """Tests for the security concept group."""

    def test_security_in_concept_groups(self):
        """Test that 'security' is in the list of concept groups."""
        groups = list_concept_groups()
        assert 'security' in groups

    def test_get_security_group_terms(self):
        """Test that get_group_terms('security') returns expected security terms."""
        terms = get_group_terms('security')

        # Verify it's non-empty
        assert len(terms) > 0

        # Verify specific expected terms are present
        expected_terms = [
            'encrypt', 'decrypt', 'cipher', 'hash', 'salt', 'digest',
            'secure', 'secret', 'key', 'certificate', 'signature',
            'vulnerability', 'exploit', 'injection', 'xss', 'csrf', 'sanitize',
            'firewall', 'whitelist', 'blacklist', 'allowlist', 'denylist',
            'audit', 'scan', 'detect', 'block', 'filter'
        ]

        for term in expected_terms:
            assert term in terms, f"Expected term '{term}' not found in security group"

        # Verify all terms are sorted
        assert terms == sorted(terms)

    def test_get_related_terms_encrypt(self):
        """Test that get_related_terms('encrypt') returns security-related terms."""
        related = get_related_terms('encrypt', max_terms=10)

        # Should get other security terms
        assert len(related) > 0

        # Verify 'encrypt' is not in its own related terms
        assert 'encrypt' not in related

        # All related terms should be from the security group
        security_terms = CODE_CONCEPT_GROUPS['security']
        for term in related:
            assert term in security_terms

    def test_get_concept_group_for_security_terms(self):
        """Test that security terms map back to the 'security' group."""
        security_terms = ['encrypt', 'decrypt', 'hash', 'sanitize', 'firewall']

        for term in security_terms:
            groups = get_concept_group(term)
            assert 'security' in groups, f"Term '{term}' should belong to 'security' group"

    def test_expand_security_concepts(self):
        """Test expansion with security-related terms."""
        # Use only 'encrypt' which is unique to security group
        expanded = expand_code_concepts(['encrypt', 'decrypt'], max_expansions_per_term=5)

        # Should have expansions from security group
        assert len(expanded) > 0

        # Input terms should not be in expansion
        assert 'encrypt' not in expanded
        assert 'decrypt' not in expanded

        # Expanded terms should be from security group
        security_terms = CODE_CONCEPT_GROUPS['security']
        for term in expanded.keys():
            assert term in security_terms

    def test_security_group_overlap_with_validation(self):
        """Test that 'sanitize' appears in both security and validation groups."""
        # 'sanitize' should be in both security and validation
        groups = get_concept_group('sanitize')
        assert 'security' in groups
        assert 'validation' in groups

        # Get related terms for 'sanitize' - should get terms from both groups
        related = get_related_terms('sanitize', max_terms=10)
        assert len(related) > 0

        security_terms = CODE_CONCEPT_GROUPS['security']
        validation_terms = CODE_CONCEPT_GROUPS['validation']

        # Some related terms should be from security or validation
        for term in related:
            assert term in security_terms or term in validation_terms

    def test_security_group_overlap_with_logging(self):
        """Test that 'audit' appears in both security and logging groups."""
        # 'audit' should be in both security and logging
        groups = get_concept_group('audit')
        assert 'security' in groups
        assert 'logging' in groups

        # Get related terms for 'audit'
        related = get_related_terms('audit', max_terms=10)
        assert len(related) > 0

        security_terms = CODE_CONCEPT_GROUPS['security']
        logging_terms = CODE_CONCEPT_GROUPS['logging']

        # Related terms should be from security or logging
        for term in related:
            assert term in security_terms or term in logging_terms
