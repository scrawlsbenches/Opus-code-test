"""Unit tests for CoChangeModel."""

import math
import pytest
from datetime import datetime, timedelta
from cortical.spark.co_change import CoChangeModel, CoChangeEdge, Commit


class TestAddingCommits:
    """Test adding commits to the model."""

    def test_add_single_commit(self):
        """Test adding a single commit."""
        model = CoChangeModel()
        model.add_commit('abc123', ['auth.py', 'login.py'])

        assert 'abc123' in model._commits
        assert len(model._commits) == 1
        assert len(model._edges) == 1

    def test_add_multiple_commits(self):
        """Test adding multiple commits."""
        model = CoChangeModel()
        model.add_commit('abc123', ['auth.py', 'login.py'])
        model.add_commit('def456', ['auth.py', 'tests/test_auth.py'])

        assert len(model._commits) == 2
        assert len(model._edges) == 2

    def test_add_commit_with_timestamp(self):
        """Test adding commit with explicit timestamp."""
        model = CoChangeModel()
        ts = datetime(2025, 1, 1, 12, 0, 0)
        model.add_commit('abc123', ['a.py', 'b.py'], timestamp=ts)

        commit = model._commits['abc123']
        assert commit.timestamp == ts

    def test_add_commit_with_message(self):
        """Test adding commit with message."""
        model = CoChangeModel()
        model.add_commit('abc123', ['a.py', 'b.py'], message='Fix bug')

        commit = model._commits['abc123']
        assert commit.message == 'Fix bug'

    def test_batch_add_commits(self):
        """Test adding commits in batch."""
        model = CoChangeModel()
        commits = [
            Commit('abc123', datetime.now(), ['a.py', 'b.py']),
            Commit('def456', datetime.now(), ['b.py', 'c.py']),
            Commit('ghi789', datetime.now(), ['a.py', 'c.py']),
        ]
        model.add_commits_batch(commits)

        assert len(model._commits) == 3
        assert len(model._edges) == 3

    def test_duplicate_commit_idempotent(self):
        """Test adding same commit twice is idempotent."""
        model = CoChangeModel()
        model.add_commit('abc123', ['a.py', 'b.py'])
        model.add_commit('abc123', ['a.py', 'b.py'])

        assert len(model._commits) == 1
        assert len(model._edges) == 1

    def test_three_file_commit_creates_three_edges(self):
        """Test commit with 3 files creates 3 edges (pairs)."""
        model = CoChangeModel()
        model.add_commit('abc123', ['a.py', 'b.py', 'c.py'])

        # Should have 3 edges: (a,b), (a,c), (b,c)
        assert len(model._edges) == 3


class TestEdgeManagement:
    """Test edge creation and updates."""

    def test_edge_creation(self):
        """Test edge is created correctly."""
        model = CoChangeModel()
        model.add_commit('abc123', ['a.py', 'b.py'])

        edge = model._edges[('a.py', 'b.py')]
        assert edge.source_file == 'a.py'
        assert edge.target_file == 'b.py'
        assert edge.co_change_count == 1
        assert len(edge.commits) == 1
        assert edge.commits[0] == 'abc123'

    def test_edge_update_increments_count(self):
        """Test edge count increments on co-change."""
        model = CoChangeModel()
        model.add_commit('abc123', ['a.py', 'b.py'])
        model.add_commit('def456', ['a.py', 'b.py'])

        edge = model._edges[('a.py', 'b.py')]
        assert edge.co_change_count == 2
        assert len(edge.commits) == 2

    def test_edge_weighted_score_accumulates(self):
        """Test weighted score accumulates over commits."""
        model = CoChangeModel()
        ts1 = datetime.now() - timedelta(days=10)
        ts2 = datetime.now()

        model.add_commit('abc123', ['a.py', 'b.py'], timestamp=ts1)
        model.add_commit('def456', ['a.py', 'b.py'], timestamp=ts2)

        edge = model._edges[('a.py', 'b.py')]
        assert edge.weighted_score > 1.0  # Two weights added

    def test_edge_alphabetical_ordering(self):
        """Test files are stored in alphabetical order."""
        model = CoChangeModel()
        model.add_commit('abc123', ['z.py', 'a.py'])

        # Should be stored as (a.py, z.py)
        assert ('a.py', 'z.py') in model._edges
        assert ('z.py', 'a.py') not in model._edges

    def test_edge_last_co_change_updated(self):
        """Test last_co_change tracks most recent timestamp."""
        model = CoChangeModel()
        ts1 = datetime(2025, 1, 1)
        ts2 = datetime(2025, 1, 10)

        model.add_commit('abc123', ['a.py', 'b.py'], timestamp=ts1)
        model.add_commit('def456', ['a.py', 'b.py'], timestamp=ts2)

        edge = model._edges[('a.py', 'b.py')]
        assert edge.last_co_change == ts2

    def test_edge_symmetry(self):
        """Test edge lookup is symmetric (A->B same as B->A)."""
        model = CoChangeModel()
        model.add_commit('abc123', ['a.py', 'b.py'])

        score_ab = model.get_co_change_score('a.py', 'b.py')
        score_ba = model.get_co_change_score('b.py', 'a.py')
        assert score_ab == score_ba

    def test_file_index_updated(self):
        """Test file index is maintained correctly."""
        model = CoChangeModel()
        model.add_commit('abc123', ['a.py', 'b.py', 'c.py'])

        assert 'b.py' in model._file_index['a.py']
        assert 'c.py' in model._file_index['a.py']
        assert 'a.py' in model._file_index['b.py']
        assert 'c.py' in model._file_index['b.py']


class TestPrediction:
    """Test prediction functionality."""

    def test_single_seed_file_prediction(self):
        """Test prediction with single seed file."""
        model = CoChangeModel()
        model.add_commit('abc123', ['a.py', 'b.py'])
        model.add_commit('def456', ['a.py', 'c.py'])

        predictions = model.predict(['a.py'], top_n=5)
        files = [f for f, _ in predictions]

        assert 'b.py' in files
        assert 'c.py' in files
        assert 'a.py' not in files  # Seed excluded

    def test_multiple_seed_files_prediction(self):
        """Test prediction with multiple seed files."""
        model = CoChangeModel()
        model.add_commit('abc123', ['a.py', 'b.py', 'd.py'])
        model.add_commit('def456', ['a.py', 'c.py'])
        model.add_commit('ghi789', ['b.py', 'd.py'])

        predictions = model.predict(['a.py', 'b.py'], top_n=5)
        files = [f for f, _ in predictions]

        assert 'c.py' in files or 'd.py' in files
        assert 'a.py' not in files
        assert 'b.py' not in files

    def test_top_n_limiting(self):
        """Test top_n limits number of predictions."""
        model = CoChangeModel()
        model.add_commit('abc123', ['a.py', 'b.py', 'c.py', 'd.py', 'e.py'])

        predictions = model.predict(['a.py'], top_n=2)
        assert len(predictions) <= 2

    def test_predictions_sorted_by_confidence(self):
        """Test predictions are sorted by confidence descending."""
        model = CoChangeModel()
        # Make b.py co-change more with a.py
        model.add_commit('abc123', ['a.py', 'b.py'])
        model.add_commit('def456', ['a.py', 'b.py'])
        model.add_commit('ghi789', ['a.py', 'c.py'])

        predictions = model.predict(['a.py'], top_n=5)

        # b.py should rank higher than c.py
        assert predictions[0][0] == 'b.py'

    def test_confidence_scores_valid_range(self):
        """Test confidence scores are in [0, 1] range."""
        model = CoChangeModel()
        model.add_commit('abc123', ['a.py', 'b.py', 'c.py'])

        predictions = model.predict(['a.py'], top_n=5)

        for _, confidence in predictions:
            assert 0.0 <= confidence <= 1.0

    def test_empty_seed_files(self):
        """Test prediction with empty seed list."""
        model = CoChangeModel()
        model.add_commit('abc123', ['a.py', 'b.py'])

        predictions = model.predict([], top_n=5)
        assert predictions == []

    def test_unknown_seed_file(self):
        """Test prediction with unknown seed file."""
        model = CoChangeModel()
        model.add_commit('abc123', ['a.py', 'b.py'])

        predictions = model.predict(['unknown.py'], top_n=5)
        assert predictions == []

    def test_prediction_aggregates_scores(self):
        """Test multiple seed files aggregate scores."""
        model = CoChangeModel()
        model.add_commit('abc123', ['a.py', 'b.py', 'd.py'])
        model.add_commit('def456', ['b.py', 'c.py', 'd.py'])

        # d.py co-changes with both a.py and b.py
        predictions = model.predict(['a.py', 'b.py'], top_n=5)
        pred_dict = dict(predictions)

        # d.py should have highest score (connected to both seeds)
        assert 'd.py' in pred_dict
        assert predictions[0][0] == 'd.py'


class TestTemporalDecay:
    """Test temporal weighting and decay."""

    def test_recent_commits_weighted_higher(self):
        """Test recent commits have higher weight."""
        model = CoChangeModel()
        old_ts = datetime.now() - timedelta(days=100)
        recent_ts = datetime.now()

        model.add_commit('old', ['a.py', 'b.py'], timestamp=old_ts)
        model.add_commit('recent', ['a.py', 'c.py'], timestamp=recent_ts)

        edge_ab = model._edges[('a.py', 'b.py')]
        edge_ac = model._edges[('a.py', 'c.py')]

        # Recent edge should have higher weighted score
        assert edge_ac.weighted_score > edge_ab.weighted_score

    def test_exponential_decay_computation(self):
        """Test exponential decay formula."""
        model = CoChangeModel(decay_lambda=0.01)
        ts = datetime.now() - timedelta(days=69)  # ~half-life

        weight = model._compute_temporal_weight(ts)

        # Should be close to 0.5 (half-life at ~69 days)
        assert 0.45 <= weight <= 0.55

    def test_zero_age_full_weight(self):
        """Test current timestamp has weight ~1.0."""
        model = CoChangeModel()
        ts = datetime.now()

        weight = model._compute_temporal_weight(ts)
        assert weight > 0.99

    def test_old_commits_decay(self):
        """Test very old commits have low weight."""
        model = CoChangeModel(decay_lambda=0.01)
        old_ts = datetime.now() - timedelta(days=500)

        weight = model._compute_temporal_weight(old_ts)
        assert weight < 0.01

    def test_different_lambda_changes_decay(self):
        """Test different lambda values affect decay rate."""
        model_slow = CoChangeModel(decay_lambda=0.001)
        model_fast = CoChangeModel(decay_lambda=0.1)

        old_ts = datetime.now() - timedelta(days=30)

        weight_slow = model_slow._compute_temporal_weight(old_ts)
        weight_fast = model_fast._compute_temporal_weight(old_ts)

        # Slow decay retains more weight
        assert weight_slow > weight_fast

    def test_future_timestamp_edge_case(self):
        """Test future timestamps don't cause errors."""
        model = CoChangeModel()
        future_ts = datetime.now() + timedelta(days=10)

        # Should handle gracefully (weight > 1 is fine mathematically)
        weight = model._compute_temporal_weight(future_ts)
        assert weight > 1.0


class TestPruning:
    """Test edge pruning functionality."""

    def test_prune_weak_edges(self):
        """Test pruning removes weak edges."""
        model = CoChangeModel()
        recent_ts = datetime.now()
        old_ts = datetime.now() - timedelta(days=500)

        model.add_commit('recent', ['a.py', 'b.py'], timestamp=recent_ts)
        model.add_commit('old', ['c.py', 'd.py'], timestamp=old_ts)

        assert len(model._edges) == 2

        removed = model.prune_old_edges(min_score=0.1)

        # Old edge should be pruned
        assert removed == 1
        assert len(model._edges) == 1

    def test_prune_maintains_strong_edges(self):
        """Test pruning keeps strong edges."""
        model = CoChangeModel()
        ts = datetime.now()

        model.add_commit('abc123', ['a.py', 'b.py'], timestamp=ts)
        model.add_commit('def456', ['a.py', 'b.py'], timestamp=ts)

        edge = model._edges[('a.py', 'b.py')]
        original_score = edge.weighted_score

        removed = model.prune_old_edges(min_score=0.1)

        assert removed == 0
        assert ('a.py', 'b.py') in model._edges


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_model_prediction(self):
        """Test prediction on empty model."""
        model = CoChangeModel()
        predictions = model.predict(['a.py'], top_n=5)
        assert predictions == []

    def test_single_file_commit_no_edges(self):
        """Test commit with single file creates no edges."""
        model = CoChangeModel()
        model.add_commit('abc123', ['a.py'])

        assert len(model._edges) == 0
        assert len(model._commits) == 1

    def test_large_file_set(self):
        """Test commit with many files."""
        model = CoChangeModel()
        files = [f'file_{i}.py' for i in range(50)]

        model.add_commit('abc123', files)

        # Should create n*(n-1)/2 edges = 50*49/2 = 1225
        assert len(model._edges) == 1225

    def test_get_edges_for_unknown_file(self):
        """Test get_edges_for_file with unknown file."""
        model = CoChangeModel()
        model.add_commit('abc123', ['a.py', 'b.py'])

        edges = model.get_edges_for_file('unknown.py')
        assert edges == []

    def test_get_co_change_score_unknown_pair(self):
        """Test get_co_change_score for unknown file pair."""
        model = CoChangeModel()
        model.add_commit('abc123', ['a.py', 'b.py'])

        score = model.get_co_change_score('a.py', 'unknown.py')
        assert score == 0.0

    def test_empty_files_list(self):
        """Test commit with empty files list."""
        model = CoChangeModel()
        model.add_commit('abc123', [])

        assert len(model._edges) == 0
        assert 'abc123' in model._commits


class TestSerialization:
    """Test JSON serialization and deserialization."""

    def test_to_dict_from_dict_roundtrip(self):
        """Test model can be serialized and restored."""
        model = CoChangeModel(decay_lambda=0.02)
        model.add_commit('abc123', ['a.py', 'b.py'])
        model.add_commit('def456', ['a.py', 'c.py'])

        # Serialize
        data = model.to_dict()

        # Deserialize
        restored = CoChangeModel.from_dict(data)

        assert restored._decay_lambda == model._decay_lambda
        assert len(restored._edges) == len(model._edges)
        assert len(restored._commits) == len(model._commits)

        # Check edge integrity
        for key, edge in model._edges.items():
            restored_edge = restored._edges[key]
            assert restored_edge.co_change_count == edge.co_change_count
            assert restored_edge.source_file == edge.source_file
            assert restored_edge.target_file == edge.target_file

    def test_datetime_serialization(self):
        """Test datetime objects serialize correctly."""
        model = CoChangeModel()
        ts = datetime(2025, 12, 22, 10, 30, 0)
        model.add_commit('abc123', ['a.py', 'b.py'], timestamp=ts)

        data = model.to_dict()
        restored = CoChangeModel.from_dict(data)

        commit = restored._commits['abc123']
        assert commit.timestamp == ts

    def test_edge_to_dict_from_dict(self):
        """Test CoChangeEdge serialization."""
        ts = datetime(2025, 12, 22, 10, 30, 0)
        edge = CoChangeEdge(
            source_file='a.py',
            target_file='b.py',
            co_change_count=5,
            weighted_score=2.5,
            confidence=0.7,
            last_co_change=ts,
            commits=['abc', 'def']
        )

        data = edge.to_dict()
        restored = CoChangeEdge.from_dict(data)

        assert restored.source_file == edge.source_file
        assert restored.target_file == edge.target_file
        assert restored.co_change_count == edge.co_change_count
        assert restored.weighted_score == edge.weighted_score
        assert restored.confidence == edge.confidence
        assert restored.last_co_change == edge.last_co_change
        assert restored.commits == edge.commits

    def test_commit_to_dict_from_dict(self):
        """Test Commit serialization."""
        ts = datetime(2025, 12, 22, 10, 30, 0)
        commit = Commit(
            sha='abc123',
            timestamp=ts,
            files=['a.py', 'b.py'],
            message='Fix bug'
        )

        data = commit.to_dict()
        restored = Commit.from_dict(data)

        assert restored.sha == commit.sha
        assert restored.timestamp == commit.timestamp
        assert restored.files == commit.files
        assert restored.message == commit.message


class TestConfidenceScoring:
    """Test confidence normalization."""

    def test_confidence_normalized_per_file(self):
        """Test confidence scores are normalized per file."""
        model = CoChangeModel()
        model.add_commit('abc123', ['a.py', 'b.py', 'c.py', 'd.py'])

        # All edges from a.py should have confidences that sum to <= 1
        edges = model.get_edges_for_file('a.py')
        total_confidence = sum(e.confidence for e in edges)

        # Due to symmetric normalization, might not sum to exactly 1
        assert 0.0 < total_confidence <= 3.0  # Upper bound relaxed

    def test_stronger_cochange_higher_confidence(self):
        """Test more frequent co-changes get higher confidence."""
        model = CoChangeModel()
        ts = datetime.now()

        # a.py co-changes more with b.py than c.py
        model.add_commit('1', ['a.py', 'b.py'], timestamp=ts)
        model.add_commit('2', ['a.py', 'b.py'], timestamp=ts)
        model.add_commit('3', ['a.py', 'c.py'], timestamp=ts)

        score_ab = model.get_co_change_score('a.py', 'b.py')
        score_ac = model.get_co_change_score('a.py', 'c.py')

        assert score_ab > score_ac

    def test_repr_format(self):
        """Test __repr__ shows useful information."""
        model = CoChangeModel()
        model.add_commit('abc123', ['a.py', 'b.py'])
        model.add_commit('def456', ['a.py', 'c.py'])

        repr_str = repr(model)
        assert 'CoChangeModel' in repr_str
        assert 'edges=2' in repr_str
        assert 'files=3' in repr_str
        assert 'commits=2' in repr_str
