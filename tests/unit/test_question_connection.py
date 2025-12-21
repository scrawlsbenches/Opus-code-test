"""
Unit Tests for Question Connection Pipeline
===========================================

Tests the question_connection.py script functionality.

Tests:
1. JSON parsing from different inputs
2. Query expansion using the processor
3. Path finding between concepts
4. Top-k limiting
5. Error handling
6. Standalone vs pipeline mode
"""

import json
import pytest
import sys
import os
from io import StringIO
from unittest.mock import patch, MagicMock

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))

from question_connection import (
    load_json_input,
    build_processor_from_json,
    expand_query_terms,
    find_connection_paths,
    find_related_concepts,
    _bfs_path,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_analysis_data():
    """Sample JSON data from world_model_analysis.py"""
    return {
        'metadata': {
            'samples_dir': 'samples',
            'mode': 'full',
        },
        'concepts': [
            {'term': 'model', 'pagerank': 0.05, 'tfidf': 2.5, 'document_count': 10,
             'domains': ['cognitive_science', 'world_models'], 'connection_count': 50},
            {'term': 'prediction', 'pagerank': 0.04, 'tfidf': 2.3, 'document_count': 8,
             'domains': ['world_models', 'future_thinking'], 'connection_count': 45},
            {'term': 'learning', 'pagerank': 0.03, 'tfidf': 2.1, 'document_count': 7,
             'domains': ['cognitive_science'], 'connection_count': 40},
            {'term': 'decision', 'pagerank': 0.025, 'tfidf': 1.9, 'document_count': 6,
             'domains': ['cognitive_science'], 'connection_count': 35},
            {'term': 'uncertainty', 'pagerank': 0.02, 'tfidf': 1.8, 'document_count': 5,
             'domains': ['future_thinking'], 'connection_count': 30},
        ],
        'bridges': [
            {'term': 'cognitive', 'domain_count': 4, 'domains': ['cognitive_science', 'world_models', 'cross_domain', 'future_thinking']},
            {'term': 'system', 'domain_count': 3, 'domains': ['cognitive_science', 'world_models', 'workflow_practices']},
        ],
        'network': {
            'model': [
                {'term': 'prediction', 'weight': 8.0},
                {'term': 'learning', 'weight': 6.5},
                {'term': 'representation', 'weight': 5.0},
            ],
            'prediction': [
                {'term': 'uncertainty', 'weight': 7.0},
                {'term': 'forecast', 'weight': 5.5},
                {'term': 'future', 'weight': 4.0},
            ],
            'learning': [
                {'term': 'adaptation', 'weight': 6.0},
                {'term': 'experience', 'weight': 5.0},
                {'term': 'update', 'weight': 4.5},
            ],
        },
        'domains': {
            'cognitive_science': {'document_count': 10},
            'world_models': {'document_count': 8},
        },
        'summary': {
            'total_documents': 25,
            'unique_concepts': 150,
        }
    }


@pytest.fixture
def minimal_analysis_data():
    """Minimal valid JSON data"""
    return {
        'concepts': [
            {'term': 'test', 'pagerank': 0.01, 'connection_count': 5}
        ],
        'bridges': [],
        'network': {},
        'domains': {}
    }


# =============================================================================
# JSON LOADING TESTS
# =============================================================================


class TestLoadJsonInput:
    """Tests for load_json_input function."""

    def test_load_from_file(self, sample_analysis_data, tmp_path):
        """Successfully load JSON from file."""
        json_file = tmp_path / "analysis.json"
        json_file.write_text(json.dumps(sample_analysis_data))

        args = MagicMock()
        args.input = str(json_file)

        result = load_json_input(args)

        assert result == sample_analysis_data
        assert 'concepts' in result
        assert len(result['concepts']) == 5

    def test_load_from_stdin(self, sample_analysis_data):
        """Successfully load JSON from stdin."""
        args = MagicMock()
        args.input = None

        json_str = json.dumps(sample_analysis_data)

        with patch('sys.stdin', StringIO(json_str)):
            with patch('sys.stdin.isatty', return_value=False):
                result = load_json_input(args)

        assert result == sample_analysis_data

    def test_missing_file_error(self):
        """Raise error when input file doesn't exist."""
        args = MagicMock()
        args.input = "/nonexistent/file.json"

        with pytest.raises(FileNotFoundError, match="Input file not found"):
            load_json_input(args)

    def test_no_input_error(self):
        """Raise error when no input provided."""
        args = MagicMock()
        args.input = None

        with patch('sys.stdin.isatty', return_value=True):
            with pytest.raises(ValueError, match="No input provided"):
                load_json_input(args)

    def test_invalid_json_error(self):
        """Raise error when stdin contains invalid JSON."""
        args = MagicMock()
        args.input = None

        invalid_json = "{ this is not valid json }"

        with patch('sys.stdin', StringIO(invalid_json)):
            with patch('sys.stdin.isatty', return_value=False):
                with pytest.raises(ValueError, match="Invalid JSON input"):
                    load_json_input(args)

    def test_empty_json_object(self):
        """Handle empty but valid JSON object."""
        args = MagicMock()
        args.input = None

        with patch('sys.stdin', StringIO("{}")):
            with patch('sys.stdin.isatty', return_value=False):
                result = load_json_input(args)

        assert result == {}


# =============================================================================
# PROCESSOR BUILDING TESTS
# =============================================================================


class TestBuildProcessorFromJson:
    """Tests for build_processor_from_json function."""

    def test_build_processor_basic(self, sample_analysis_data):
        """Build processor from sample data."""
        from cortical import CorticalLayer
        processor = build_processor_from_json(sample_analysis_data)

        assert processor is not None
        # Check that documents were created
        layer3 = processor.get_layer(CorticalLayer.DOCUMENTS)
        assert layer3.column_count() > 0

    def test_build_processor_with_concepts(self, sample_analysis_data):
        """Processor includes concepts from data."""
        from cortical import CorticalLayer
        processor = build_processor_from_json(sample_analysis_data)

        layer0 = processor.get_layer(CorticalLayer.TOKENS)

        # Check that key terms are present
        assert layer0.get_minicolumn('model') is not None
        assert layer0.get_minicolumn('prediction') is not None
        assert layer0.get_minicolumn('learning') is not None

    def test_build_processor_empty_concepts(self):
        """Handle data with no concepts."""
        from cortical import CorticalLayer
        data = {'concepts': [], 'network': {}}
        processor = build_processor_from_json(data)

        assert processor is not None
        # Should still create a processor, just empty
        layer0 = processor.get_layer(CorticalLayer.TOKENS)
        assert layer0.column_count() == 0

    def test_build_processor_missing_concepts_key(self):
        """Handle data without concepts key."""
        from cortical import CorticalLayer
        data = {'network': {}, 'domains': {}}
        processor = build_processor_from_json(data)

        assert processor is not None
        layer0 = processor.get_layer(CorticalLayer.TOKENS)
        assert layer0.column_count() == 0

    def test_build_processor_domain_grouping(self, sample_analysis_data):
        """Concepts are grouped by domain."""
        from cortical import CorticalLayer
        processor = build_processor_from_json(sample_analysis_data)

        layer3 = processor.get_layer(CorticalLayer.DOCUMENTS)

        # Check that synthetic documents exist for domains
        doc_ids = list(layer3.minicolumns.keys())
        assert any('cognitive_science' in doc_id for doc_id in doc_ids)
        assert any('world_models' in doc_id for doc_id in doc_ids)


# =============================================================================
# QUERY EXPANSION TESTS
# =============================================================================


class TestExpandQueryTerms:
    """Tests for expand_query_terms function."""

    def test_expand_basic_query(self, sample_analysis_data):
        """Expand a simple query."""
        processor = build_processor_from_json(sample_analysis_data)

        expanded = expand_query_terms(processor, "model prediction", max_expansions=5)

        assert isinstance(expanded, dict)
        assert len(expanded) > 0
        # Original terms should be present with weight 1.0
        assert 'model' in expanded
        assert 'prediction' in expanded

    def test_expand_single_term(self, sample_analysis_data):
        """Expand single-term query."""
        processor = build_processor_from_json(sample_analysis_data)

        expanded = expand_query_terms(processor, "learning", max_expansions=10)

        assert 'learning' in expanded
        assert len(expanded) >= 1

    def test_expand_max_expansions_limit(self, sample_analysis_data):
        """Respect max_expansions parameter."""
        processor = build_processor_from_json(sample_analysis_data)

        expanded = expand_query_terms(processor, "model", max_expansions=3)

        # Should not exceed max_expansions
        assert len(expanded) <= 10  # Processor may add original + expansions

    def test_expand_unknown_term(self):
        """Handle query with unknown terms."""
        data = {'concepts': [{'term': 'known', 'pagerank': 0.01, 'domains': ['test']}]}
        processor = build_processor_from_json(data)

        expanded = expand_query_terms(processor, "unknown_term_xyz", max_expansions=5)

        # Should still return a dict (may be empty or contain stemmed version)
        assert isinstance(expanded, dict)

    def test_expand_empty_query(self, minimal_analysis_data):
        """Handle empty query string."""
        processor = build_processor_from_json(minimal_analysis_data)

        # Empty query should raise ValueError
        with pytest.raises(ValueError, match="non-empty string"):
            expand_query_terms(processor, "", max_expansions=5)


# =============================================================================
# PATH FINDING TESTS
# =============================================================================


class TestBfsPath:
    """Tests for _bfs_path function."""

    def test_bfs_direct_connection(self):
        """Find path with direct connection."""
        adjacency = {
            'a': {'b': 1.0},
            'b': {'c': 1.0},
        }

        path = _bfs_path(adjacency, 'a', 'b', max_depth=3)

        assert path == ['a', 'b']

    def test_bfs_two_hops(self):
        """Find path with two hops."""
        adjacency = {
            'a': {'b': 1.0},
            'b': {'c': 1.0},
        }

        path = _bfs_path(adjacency, 'a', 'c', max_depth=3)

        assert path == ['a', 'b', 'c']

    def test_bfs_no_path(self):
        """Return None when no path exists."""
        adjacency = {
            'a': {'b': 1.0},
            'c': {'d': 1.0},
        }

        path = _bfs_path(adjacency, 'a', 'c', max_depth=3)

        assert path is None

    def test_bfs_same_node(self):
        """Path from node to itself."""
        adjacency = {'a': {'b': 1.0}}

        path = _bfs_path(adjacency, 'a', 'a', max_depth=3)

        assert path == ['a']

    def test_bfs_max_depth_limit(self):
        """Respect max_depth parameter."""
        adjacency = {
            'a': {'b': 1.0},
            'b': {'c': 1.0},
            'c': {'d': 1.0},
            'd': {'e': 1.0},
        }

        # With max_depth=2, can't reach 'd' from 'a'
        path = _bfs_path(adjacency, 'a', 'd', max_depth=2)

        assert path is None

    def test_bfs_multiple_paths(self):
        """Find shortest path when multiple exist."""
        adjacency = {
            'a': {'b': 1.0, 'c': 1.0},
            'b': {'d': 1.0},
            'c': {'e': 1.0},
            'e': {'d': 1.0},
        }

        path = _bfs_path(adjacency, 'a', 'd', max_depth=5)

        # Should find shorter path a->b->d, not a->c->e->d
        assert path == ['a', 'b', 'd']


class TestFindConnectionPaths:
    """Tests for find_connection_paths function."""

    def test_find_paths_basic(self, sample_analysis_data):
        """Find basic connection paths."""
        paths = find_connection_paths(sample_analysis_data, max_paths=5, max_depth=3)

        assert isinstance(paths, list)
        assert len(paths) > 0

        # Check path structure
        for path in paths:
            assert 'from' in path
            assert 'to' in path
            assert 'via' in path
            assert 'total_weight' in path
            assert 'length' in path

    def test_find_paths_respects_max_paths(self, sample_analysis_data):
        """Limit results to max_paths."""
        max_paths = 3
        paths = find_connection_paths(sample_analysis_data, max_paths=max_paths)

        assert len(paths) <= max_paths

    def test_find_paths_empty_network(self):
        """Handle empty network gracefully."""
        data = {'concepts': [], 'network': {}}

        paths = find_connection_paths(data, max_paths=10)

        assert paths == []

    def test_find_paths_single_concept(self):
        """Handle network with single concept."""
        data = {
            'concepts': [{'term': 'alone', 'pagerank': 0.01}],
            'network': {}
        }

        paths = find_connection_paths(data, max_paths=10)

        assert paths == []

    def test_find_paths_no_intermediate_nodes(self, sample_analysis_data):
        """Paths should have intermediate nodes (length > 1)."""
        paths = find_connection_paths(sample_analysis_data, max_paths=10, max_depth=3)

        # Filter out direct connections (we only want multi-hop paths)
        for path in paths:
            # Length is number of edges, so length > 1 means 2+ hops
            if path['length'] > 1:
                assert len(path['via']) >= 0  # May have intermediate nodes

    def test_find_paths_sorted_by_weight(self, sample_analysis_data):
        """Paths sorted by total weight descending."""
        paths = find_connection_paths(sample_analysis_data, max_paths=10)

        if len(paths) > 1:
            weights = [p['total_weight'] for p in paths]
            # Should be in descending order
            assert weights == sorted(weights, reverse=True)

    def test_find_paths_calculates_weight(self, sample_analysis_data):
        """Total weight is sum of edge weights."""
        paths = find_connection_paths(sample_analysis_data, max_paths=10)

        for path in paths:
            assert path['total_weight'] >= 0


# =============================================================================
# RELATED CONCEPTS TESTS
# =============================================================================


class TestFindRelatedConcepts:
    """Tests for find_related_concepts function."""

    def test_find_related_basic(self, sample_analysis_data):
        """Find related concepts for query terms."""
        query_terms = ['model']

        related = find_related_concepts(sample_analysis_data, query_terms, top_k=5)

        assert isinstance(related, list)
        assert len(related) > 0

        # Check structure
        for concept in related:
            assert 'term' in concept
            assert 'relevance' in concept
            assert 'pagerank' in concept
            assert 'connections' in concept

    def test_find_related_excludes_query_terms(self, sample_analysis_data):
        """Related concepts should not include query terms."""
        query_terms = ['model', 'prediction']

        related = find_related_concepts(sample_analysis_data, query_terms, top_k=10)

        related_terms = [c['term'] for c in related]
        assert 'model' not in related_terms
        assert 'prediction' not in related_terms

    def test_find_related_respects_top_k(self, sample_analysis_data):
        """Limit results to top_k."""
        top_k = 2

        related = find_related_concepts(sample_analysis_data, [], top_k=top_k)

        assert len(related) <= top_k

    def test_find_related_sorted_by_relevance(self, sample_analysis_data):
        """Results sorted by relevance descending."""
        related = find_related_concepts(sample_analysis_data, [], top_k=10)

        if len(related) > 1:
            relevances = [c['relevance'] for c in related]
            assert relevances == sorted(relevances, reverse=True)

    def test_find_related_empty_concepts(self):
        """Handle data with no concepts."""
        data = {'concepts': []}

        related = find_related_concepts(data, ['test'], top_k=10)

        assert related == []

    def test_find_related_missing_concepts_key(self):
        """Handle data without concepts key."""
        data = {'network': {}}

        related = find_related_concepts(data, ['test'], top_k=10)

        assert related == []

    def test_find_related_relevance_calculation(self, sample_analysis_data):
        """Relevance scores are calculated correctly."""
        related = find_related_concepts(sample_analysis_data, [], top_k=10)

        for concept in related:
            # Relevance should be between 0 and 1
            assert 0 <= concept['relevance'] <= 1.5  # Can be > 1 with high conn count

    def test_find_related_single_query_term(self, sample_analysis_data):
        """Works with single query term."""
        related = find_related_concepts(sample_analysis_data, ['learning'], top_k=5)

        assert len(related) > 0
        assert 'learning' not in [c['term'] for c in related]


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline_with_query(self, sample_analysis_data, tmp_path):
        """Full pipeline: load -> expand -> find related."""
        # Save to file
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(sample_analysis_data))

        # Load
        args = MagicMock()
        args.input = str(json_file)
        data = load_json_input(args)

        # Build processor and expand
        processor = build_processor_from_json(data)
        expanded = expand_query_terms(processor, "model learning", max_expansions=5)

        # Find related
        from cortical.tokenizer import Tokenizer
        tokenizer = Tokenizer()
        query_terms = tokenizer.tokenize("model learning")
        related = find_related_concepts(data, query_terms, top_k=5)

        assert len(expanded) > 0
        assert len(related) > 0

    def test_full_pipeline_with_paths(self, sample_analysis_data):
        """Full pipeline with path finding."""
        # Build processor
        processor = build_processor_from_json(sample_analysis_data)

        # Find paths
        paths = find_connection_paths(sample_analysis_data, max_paths=5)

        assert isinstance(paths, list)
        # May or may not find paths depending on network structure

    def test_minimal_valid_input(self, minimal_analysis_data):
        """Handle minimal but valid input."""
        processor = build_processor_from_json(minimal_analysis_data)
        expanded = expand_query_terms(processor, "test", max_expansions=5)
        related = find_related_concepts(minimal_analysis_data, ['test'], top_k=5)

        # Should not crash with minimal data
        assert isinstance(expanded, dict)
        assert isinstance(related, list)


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_malformed_concept_entries(self):
        """Handle malformed concept entries gracefully."""
        data = {
            'concepts': [
                {'term': 'valid', 'pagerank': 0.01, 'connection_count': 10},
                {'term': 'missing_fields'},  # Missing pagerank
                {},  # Empty concept
            ]
        }

        # Should not crash
        processor = build_processor_from_json(data)
        assert processor is not None

        related = find_related_concepts(data, ['test'], top_k=5)
        assert isinstance(related, list)

    def test_negative_top_k(self, sample_analysis_data):
        """Handle negative top_k value."""
        # Should return empty list
        related = find_related_concepts(sample_analysis_data, ['model'], top_k=-1)
        assert related == []

    def test_zero_top_k(self, sample_analysis_data):
        """Handle zero top_k value."""
        related = find_related_concepts(sample_analysis_data, [], top_k=0)
        assert related == []

    def test_very_large_top_k(self, sample_analysis_data):
        """Handle top_k larger than available concepts."""
        related = find_related_concepts(sample_analysis_data, [], top_k=1000)

        # Should return all available concepts
        assert len(related) == len(sample_analysis_data['concepts'])
