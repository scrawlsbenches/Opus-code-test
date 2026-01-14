"""
Unit Tests for Experiment Checkpoint Save/Restore
==================================================

Tests for checkpoint functionality in cortical/experiments/:
- ExperimentLog.save_checkpoint()
- ExperimentLog.load_checkpoint()
- ExperimentLog.restore_parameters()
- Parameter naming uniqueness
- Forward pass consistency after restore
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from cortical.graph.attention import (
    AttentionGraph,
    AttentionLayer,
    create_causal_attention_graph,
)
from cortical.experiments.logging import ExperimentLog
from cortical.experiments.config import ExperimentConfig
from cortical.experiments.projection import VocabProjection


class TestAttentionLayerParameterNaming:
    """Tests for unique parameter naming in attention layers."""

    def test_single_layer_has_named_parameters(self):
        """Single attention layer has properly named parameters."""
        layer = AttentionLayer(embedding_dim=16, num_heads=2, name_prefix="layer_0")
        params = layer.parameters()

        names = [p.name for p in params]
        assert "layer_0_W_q" in names
        assert "layer_0_W_k" in names
        assert "layer_0_W_v" in names
        assert "layer_0_W_o" in names

    def test_different_layers_have_unique_names(self):
        """Different layers have distinct parameter names."""
        layer0 = AttentionLayer(embedding_dim=16, num_heads=2, name_prefix="layer_0")
        layer1 = AttentionLayer(embedding_dim=16, num_heads=2, name_prefix="layer_1")

        names0 = {p.name for p in layer0.parameters()}
        names1 = {p.name for p in layer1.parameters()}

        # No overlap
        assert names0.isdisjoint(names1)

    def test_graph_layers_have_unique_names(self):
        """AttentionGraph creates layers with unique parameter names."""
        graph = create_causal_attention_graph(
            seq_len=10, embedding_dim=16, num_heads=2, seed=42
        )
        # Trigger layer creation
        graph.forward(num_layers=3)

        all_params = graph.parameters()
        names = [p.name for p in all_params]

        # Check for uniqueness
        assert len(names) == len(set(names)), f"Duplicate names found: {[n for n in names if names.count(n) > 1]}"

        # Check layer naming pattern
        assert "layer_0_W_q" in names
        assert "layer_1_W_q" in names
        assert "layer_2_W_q" in names


class TestCheckpointSaveLoad:
    """Tests for checkpoint save and load functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_config(self):
        """Create sample experiment config."""
        return ExperimentConfig(
            name="test",
            input_path="test.txt",
            embedding_dim=16,
            num_heads=2,
            num_layers=2,
        )

    def test_save_checkpoint_creates_file(self, temp_dir, sample_config):
        """save_checkpoint creates checkpoint.pkl file."""
        log = ExperimentLog(config=sample_config, base_dir=temp_dir)

        # Create dummy parameters
        from cortical.graph.trainable import Parameter
        params = [
            Parameter(data=np.array([1.0, 2.0, 3.0]), name="test_param")
        ]

        checkpoint_path = log.save_checkpoint(params)

        assert checkpoint_path.exists()
        assert checkpoint_path.name == "checkpoint.pkl"

    def test_load_checkpoint_returns_data(self, temp_dir, sample_config):
        """load_checkpoint returns saved parameter data."""
        log = ExperimentLog(config=sample_config, base_dir=temp_dir)

        from cortical.graph.trainable import Parameter
        original_data = np.array([1.0, 2.0, 3.0])
        params = [Parameter(data=original_data.copy(), name="test_param")]

        checkpoint_path = log.save_checkpoint(params)
        loaded = ExperimentLog.load_checkpoint(checkpoint_path)

        assert "parameters" in loaded
        assert len(loaded["parameters"]) == 1
        assert loaded["parameters"][0]["name"] == "test_param"
        np.testing.assert_array_equal(loaded["parameters"][0]["data"], original_data)

    def test_restore_parameters_updates_values(self, temp_dir, sample_config):
        """restore_parameters correctly updates parameter values."""
        log = ExperimentLog(config=sample_config, base_dir=temp_dir)

        from cortical.graph.trainable import Parameter

        # Save trained parameters
        trained_data = np.array([10.0, 20.0, 30.0])
        trained_params = [Parameter(data=trained_data.copy(), name="test_param")]
        checkpoint_path = log.save_checkpoint(trained_params)

        # Create fresh parameters with different values
        fresh_data = np.array([1.0, 2.0, 3.0])
        fresh_params = [Parameter(data=fresh_data.copy(), name="test_param")]

        # Restore
        loaded = ExperimentLog.load_checkpoint(checkpoint_path)
        restored_count = ExperimentLog.restore_parameters(fresh_params, loaded)

        assert restored_count == 1
        np.testing.assert_array_equal(fresh_params[0].data, trained_data)


class TestAttentionGraphCheckpoint:
    """Tests for AttentionGraph checkpoint save/restore."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_config(self):
        return ExperimentConfig(
            name="test",
            input_path="test.txt",
            embedding_dim=16,
            num_heads=2,
            num_layers=2,
        )

    def test_graph_parameters_restore_correctly(self, temp_dir, sample_config):
        """Graph parameters restore to saved values."""
        # Create and initialize graph
        np.random.seed(42)
        graph1 = create_causal_attention_graph(
            seq_len=5, embedding_dim=16, num_heads=2, seed=42
        )
        input_nodes = {f"pos_{i}": np.random.randn(16) for i in range(5)}
        graph1.forward(num_layers=2, input_nodes=input_nodes)

        # Modify parameters (simulate training)
        for p in graph1.parameters():
            if "W_q" in p.name:
                p.data[:] = np.ones_like(p.data) * 999.0

        # Save checkpoint
        log = ExperimentLog(config=sample_config, base_dir=temp_dir)
        checkpoint_path = log.save_checkpoint(graph1.parameters())

        # Create fresh graph
        np.random.seed(42)
        graph2 = create_causal_attention_graph(
            seq_len=5, embedding_dim=16, num_heads=2, seed=42
        )
        graph2.forward(num_layers=2, input_nodes=input_nodes)

        # Verify fresh graph has different values
        for p in graph2.parameters():
            if "W_q" in p.name:
                assert not np.allclose(p.data, 999.0)

        # Restore
        loaded = ExperimentLog.load_checkpoint(checkpoint_path)
        ExperimentLog.restore_parameters(graph2.parameters(), loaded)

        # Verify restored values
        for p in graph2.parameters():
            if "W_q" in p.name:
                np.testing.assert_array_equal(p.data, np.ones((16, 16)) * 999.0)

    def test_forward_output_matches_after_restore(self, temp_dir, sample_config):
        """Forward pass output matches after checkpoint restore."""
        np.random.seed(42)

        # Create graph and get output
        graph1 = create_causal_attention_graph(
            seq_len=5, embedding_dim=16, num_heads=2, seed=42
        )
        input_nodes = {f"pos_{i}": np.random.randn(16) for i in range(5)}
        output1 = graph1.forward(num_layers=2, input_nodes=input_nodes)

        # Save checkpoint
        log = ExperimentLog(config=sample_config, base_dir=temp_dir)
        log.save_checkpoint(graph1.parameters())

        # Create fresh graph with same seed
        np.random.seed(42)
        graph2 = create_causal_attention_graph(
            seq_len=5, embedding_dim=16, num_heads=2, seed=42
        )
        input_nodes2 = {f"pos_{i}": np.random.randn(16) for i in range(5)}
        graph2.forward(num_layers=2, input_nodes=input_nodes2)

        # Restore
        loaded = ExperimentLog.load_checkpoint(log.checkpoint_path)
        ExperimentLog.restore_parameters(graph2.parameters(), loaded)

        # Get output with SAME input_nodes as graph1
        output2 = graph2.forward(num_layers=2, input_nodes=input_nodes)

        # Outputs should match
        for node_id in output1:
            np.testing.assert_array_almost_equal(
                output1[node_id], output2[node_id],
                err_msg=f"Output mismatch for {node_id}"
            )


class TestVocabProjectionCheckpoint:
    """Tests for VocabProjection checkpoint save/restore."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_config(self):
        return ExperimentConfig(
            name="test",
            input_path="test.txt",
            embedding_dim=16,
            num_heads=2,
            num_layers=2,
        )

    def test_vocab_projection_restores_correctly(self, temp_dir, sample_config):
        """VocabProjection parameters restore correctly."""
        # Create and modify vocab projection
        vocab_proj1 = VocabProjection(embedding_dim=16, vocab_size=10)
        vocab_proj1.W.data[:] = np.ones_like(vocab_proj1.W.data) * 123.0
        vocab_proj1.b.data[:] = np.ones_like(vocab_proj1.b.data) * 456.0

        # Save
        log = ExperimentLog(config=sample_config, base_dir=temp_dir)
        log.save_checkpoint(vocab_proj1.parameters())

        # Create fresh
        vocab_proj2 = VocabProjection(embedding_dim=16, vocab_size=10)

        # Restore
        loaded = ExperimentLog.load_checkpoint(log.checkpoint_path)
        ExperimentLog.restore_parameters(vocab_proj2.parameters(), loaded)

        np.testing.assert_array_equal(vocab_proj2.W.data, 123.0)
        np.testing.assert_array_equal(vocab_proj2.b.data, 456.0)


class TestEndToEndCheckpoint:
    """End-to-end tests for complete training checkpoint cycle."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_config(self):
        return ExperimentConfig(
            name="test",
            input_path="test.txt",
            embedding_dim=16,
            num_heads=2,
            num_layers=2,
        )

    def test_trained_model_accuracy_preserved(self, temp_dir, sample_config):
        """Trained model accuracy is preserved after checkpoint restore."""
        from cortical.experiments.kernel import ExperimentKernel
        from cortical.experiments.projection import CrossEntropyWithLogits
        from cortical.graph.trainable import Adam

        # Setup
        np.random.seed(42)
        tokens = ["a", "b", "c", "d", "e"]
        vocab = sorted(set(tokens))
        token_to_id = {t: i for i, t in enumerate(vocab)}
        id_to_token = {i: t for t, i in token_to_id.items()}
        token_ids = [token_to_id[t] for t in tokens]

        embeddings = np.random.randn(len(vocab), 16) * 0.35

        # Create and train
        graph = create_causal_attention_graph(
            seq_len=len(tokens), embedding_dim=16, num_heads=2, seed=42, use_residual=True
        )
        input_nodes = {f"pos_{i}": embeddings[token_ids[i]].copy() for i in range(len(tokens))}
        graph.forward(num_layers=2, input_nodes=input_nodes)

        vocab_proj = VocabProjection(embedding_dim=16, vocab_size=len(vocab))
        all_params = graph.parameters() + vocab_proj.parameters()
        optimizer = Adam(all_params, lr=0.01)

        kernel = ExperimentKernel(
            graph=graph, optimizer=optimizer,
            loss_fn=CrossEntropyWithLogits(), vocab_projection=vocab_proj
        )

        targets = {f"pos_{i}": np.eye(len(vocab))[token_ids[i + 1]] for i in range(len(tokens) - 1)}

        # Train until high accuracy
        for _ in range(100):
            kernel.train_step(targets=targets, num_layers=2, clip_grad=1.0, input_nodes=input_nodes)

        # Check accuracy before save
        outputs = graph.forward(num_layers=2, input_nodes=input_nodes)
        logits = vocab_proj.forward(outputs, apply_softmax=False)
        correct_before = sum(
            1 for i in range(len(tokens) - 1)
            if id_to_token[np.argmax(logits[f"pos_{i}"])] == tokens[i + 1]
        )

        # Save checkpoint
        log = ExperimentLog(config=sample_config, base_dir=temp_dir)
        checkpoint_data = {
            "parameters": [
                {"name": p.name, "data": p.data.copy(), "requires_grad": p.requires_grad}
                for p in all_params
            ]
        }

        # Create fresh model
        np.random.seed(42)
        embeddings2 = np.random.randn(len(vocab), 16) * 0.35

        graph2 = create_causal_attention_graph(
            seq_len=len(tokens), embedding_dim=16, num_heads=2, seed=42, use_residual=True
        )
        input_nodes2 = {f"pos_{i}": embeddings2[token_ids[i]].copy() for i in range(len(tokens))}
        graph2.forward(num_layers=2, input_nodes=input_nodes2)

        vocab_proj2 = VocabProjection(embedding_dim=16, vocab_size=len(vocab))
        all_params2 = graph2.parameters() + vocab_proj2.parameters()

        # Restore
        ExperimentLog.restore_parameters(all_params2, checkpoint_data)

        # Check accuracy after restore (using same input_nodes)
        outputs2 = graph2.forward(num_layers=2, input_nodes=input_nodes)
        logits2 = vocab_proj2.forward(outputs2, apply_softmax=False)
        correct_after = sum(
            1 for i in range(len(tokens) - 1)
            if id_to_token[np.argmax(logits2[f"pos_{i}"])] == tokens[i + 1]
        )

        assert correct_after == correct_before, f"Accuracy changed: {correct_before} -> {correct_after}"
