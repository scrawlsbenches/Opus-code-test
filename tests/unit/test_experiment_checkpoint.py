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


# =============================================================================
# Resume Training Tests (TDD)
# =============================================================================


class TestCheckpointResumeData:
    """Tests for checkpoint data required for resume training."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_config(self):
        return ExperimentConfig(
            name="resume-test",
            input_path="test.txt",
            embedding_dim=16,
            num_heads=2,
            num_layers=2,
        )

    def test_checkpoint_saves_epoch_number(self, temp_dir, sample_config):
        """Checkpoint should store the epoch number for resume."""
        from cortical.graph.trainable import Parameter

        log = ExperimentLog(config=sample_config, base_dir=temp_dir)
        params = [Parameter(data=np.array([1.0, 2.0]), name="test")]

        # Save with epoch
        log.save_checkpoint(params, epoch=150)

        # Load and verify
        loaded = ExperimentLog.load_checkpoint(log.checkpoint_path)
        assert "epoch" in loaded
        assert loaded["epoch"] == 150

    def test_checkpoint_saves_optimizer_state(self, temp_dir, sample_config):
        """Checkpoint should store optimizer state dict."""
        from cortical.graph.trainable import Parameter, Adam

        log = ExperimentLog(config=sample_config, base_dir=temp_dir)
        params = [Parameter(data=np.array([1.0, 2.0]), name="test")]
        optimizer = Adam(params, lr=0.01)

        # Simulate training steps to populate optimizer state
        for _ in range(5):
            params[0].grad = np.array([0.1, 0.2])
            optimizer.step()

        # Save checkpoint with optimizer
        log.save_checkpoint(params, optimizer=optimizer, epoch=5)

        # Load and verify
        loaded = ExperimentLog.load_checkpoint(log.checkpoint_path)
        assert "optimizer_state" in loaded
        assert loaded["optimizer_state"] is not None
        assert "lr" in loaded["optimizer_state"]
        assert "t" in loaded["optimizer_state"]  # Adam step counter
        assert loaded["optimizer_state"]["t"] == 5

    def test_checkpoint_saves_scheduler_state(self, temp_dir, sample_config):
        """Checkpoint should store scheduler state dict."""
        from cortical.graph.trainable import Parameter, Adam
        from cortical.experiments.scheduler import StepLR

        log = ExperimentLog(config=sample_config, base_dir=temp_dir)
        params = [Parameter(data=np.array([1.0, 2.0]), name="test")]
        optimizer = Adam(params, lr=0.01)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

        # Simulate steps
        for epoch in range(25):
            scheduler.step(epoch=epoch)

        # Save checkpoint with scheduler
        log.save_checkpoint(params, optimizer=optimizer, epoch=25, scheduler=scheduler)

        # Load and verify
        loaded = ExperimentLog.load_checkpoint(log.checkpoint_path)
        assert "scheduler_state" in loaded
        assert loaded["scheduler_state"] is not None
        assert loaded["scheduler_state"]["last_epoch"] == 24  # 0-indexed
        assert loaded["scheduler_state"]["base_lr"] == 0.01


class TestOptimizerStateRestoration:
    """Tests for optimizer state restoration from checkpoint."""

    def test_adam_state_dict_roundtrip(self):
        """Adam optimizer state survives save/load cycle."""
        from cortical.graph.trainable import Parameter, Adam

        params = [Parameter(data=np.random.randn(10), name="test")]
        optimizer1 = Adam(params, lr=0.01, betas=(0.9, 0.999))

        # Run several steps to build up momentum
        for _ in range(10):
            params[0].grad = np.random.randn(10) * 0.1
            optimizer1.step()

        # Save state
        state = optimizer1.state_dict()

        # Create fresh optimizer
        params2 = [Parameter(data=np.random.randn(10), name="test")]
        optimizer2 = Adam(params2, lr=0.05)  # Different initial LR

        # Restore state
        optimizer2.load_state_dict(state)

        # Verify state restored
        assert optimizer2.lr == optimizer1.lr
        assert optimizer2.t == optimizer1.t
        np.testing.assert_array_almost_equal(optimizer2.m[0], optimizer1.m[0])
        np.testing.assert_array_almost_equal(optimizer2.v[0], optimizer1.v[0])

    def test_optimizer_continues_from_saved_step(self):
        """Optimizer continues training correctly from saved state."""
        from cortical.graph.trainable import Parameter, Adam

        # Train for 100 steps
        params1 = [Parameter(data=np.zeros(5), name="test")]
        optimizer1 = Adam(params1, lr=0.01)

        np.random.seed(42)
        for _ in range(100):
            params1[0].grad = np.random.randn(5) * 0.1
            optimizer1.step()

        value_at_100 = params1[0].data.copy()
        state_at_100 = optimizer1.state_dict()

        # Continue training from step 100
        np.random.seed(43)  # Different seed for continuation
        for _ in range(50):
            params1[0].grad = np.random.randn(5) * 0.1
            optimizer1.step()

        value_at_150_original = params1[0].data.copy()

        # Now simulate resume: fresh optimizer, restore state, continue
        params2 = [Parameter(data=value_at_100.copy(), name="test")]
        optimizer2 = Adam(params2, lr=0.05)  # Different LR - should be overwritten
        optimizer2.load_state_dict(state_at_100)

        # Continue with same gradient sequence
        np.random.seed(43)
        for _ in range(50):
            params2[0].grad = np.random.randn(5) * 0.1
            optimizer2.step()

        # Should match original training
        np.testing.assert_array_almost_equal(
            params2[0].data, value_at_150_original,
            err_msg="Resumed optimizer didn't match original training"
        )


class TestSchedulerStateRestoration:
    """Tests for LR scheduler state restoration from checkpoint."""

    def test_step_lr_state_roundtrip(self):
        """StepLR scheduler state survives save/load cycle."""
        from cortical.graph.trainable import Parameter, Adam
        from cortical.experiments.scheduler import StepLR

        params = [Parameter(data=np.array([1.0]), name="test")]
        optimizer = Adam(params, lr=0.1)
        scheduler1 = StepLR(optimizer, step_size=10, gamma=0.5)

        # Advance to epoch 25
        for epoch in range(25):
            scheduler1.step(epoch=epoch)

        state = scheduler1.state_dict()

        # Create fresh scheduler
        optimizer2 = Adam(params, lr=0.1)
        scheduler2 = StepLR(optimizer2, step_size=10, gamma=0.5)

        # Restore
        scheduler2.load_state_dict(state)

        # Verify state
        assert scheduler2.last_epoch == scheduler1.last_epoch
        assert scheduler2._step_count == scheduler1._step_count
        assert scheduler2.base_lr == scheduler1.base_lr

    def test_cosine_lr_state_roundtrip(self):
        """CosineAnnealingLR scheduler state survives save/load cycle."""
        from cortical.graph.trainable import Parameter, Adam
        from cortical.experiments.scheduler import CosineAnnealingLR

        params = [Parameter(data=np.array([1.0]), name="test")]
        optimizer = Adam(params, lr=0.1)
        scheduler1 = CosineAnnealingLR(optimizer, T_max=100, lr_min=1e-6)

        # Advance to epoch 50
        for epoch in range(50):
            scheduler1.step(epoch=epoch)

        state = scheduler1.state_dict()
        lr_at_50 = optimizer.lr

        # Create fresh scheduler and restore
        optimizer2 = Adam(params, lr=0.1)
        scheduler2 = CosineAnnealingLR(optimizer2, T_max=100, lr_min=1e-6)
        scheduler2.load_state_dict(state)

        # Get LR at epoch 50 again
        scheduler2.step(epoch=49)  # Recompute LR for last_epoch=49

        assert optimizer2.lr == pytest.approx(lr_at_50, rel=1e-6)

    def test_plateau_lr_state_roundtrip(self):
        """ReduceLROnPlateau scheduler state survives save/load cycle."""
        from cortical.graph.trainable import Parameter, Adam
        from cortical.experiments.scheduler import ReduceLROnPlateau

        params = [Parameter(data=np.array([1.0]), name="test")]
        optimizer = Adam(params, lr=0.1)
        scheduler1 = ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5, min_lr=1e-6
        )

        # Simulate training with plateauing loss
        losses = [1.0, 0.9, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]  # Plateaus after 0.8
        for loss in losses:
            scheduler1.step(loss)

        state = scheduler1.state_dict()

        # Create fresh scheduler
        optimizer2 = Adam(params, lr=0.1)
        scheduler2 = ReduceLROnPlateau(
            optimizer2, patience=5, factor=0.5, min_lr=1e-6
        )

        # Restore
        scheduler2.load_state_dict(state)

        # Verify
        assert scheduler2.best == scheduler1.best
        assert scheduler2.num_bad_epochs == scheduler1.num_bad_epochs
        assert scheduler2.current_lr == scheduler1.current_lr


class TestResumeTrainingIntegration:
    """Integration tests for complete resume training workflow."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_config(self):
        return ExperimentConfig(
            name="resume-integration",
            input_path="test.txt",
            embedding_dim=16,
            num_heads=2,
            num_layers=2,
        )

    def test_full_resume_workflow(self, temp_dir, sample_config):
        """
        Complete resume workflow: train -> save -> restore -> continue.

        This test verifies that:
        1. Parameters are restored correctly
        2. Optimizer state is restored correctly
        3. Scheduler state is restored correctly
        4. Training continues from the correct epoch
        """
        from cortical.graph.trainable import Parameter, Adam
        from cortical.experiments.scheduler import StepLR

        # === Phase 1: Initial training for 50 epochs ===
        np.random.seed(42)
        params1 = [
            Parameter(data=np.random.randn(10), name="weights"),
            Parameter(data=np.random.randn(5), name="bias"),
        ]
        optimizer1 = Adam(params1, lr=0.01)
        scheduler1 = StepLR(optimizer1, step_size=25, gamma=0.5)

        # Train for 50 epochs
        for epoch in range(50):
            for p in params1:
                p.grad = np.random.randn(*p.data.shape) * 0.1
            optimizer1.step()
            scheduler1.step(epoch=epoch)

        # Save checkpoint at epoch 50
        log = ExperimentLog(config=sample_config, base_dir=temp_dir)
        checkpoint_path = log.save_checkpoint(
            params1,
            optimizer=optimizer1,
            epoch=50,
            scheduler=scheduler1,
        )

        # Record state at epoch 50
        params_at_50 = {p.name: p.data.copy() for p in params1}
        lr_at_50 = optimizer1.lr
        optimizer_t_at_50 = optimizer1.t

        # === Phase 2: Continue training to epoch 100 (original) ===
        np.random.seed(100)  # New seed for continuation
        for epoch in range(50, 100):
            for p in params1:
                p.grad = np.random.randn(*p.data.shape) * 0.1
            optimizer1.step()
            scheduler1.step(epoch=epoch)

        params_at_100_original = {p.name: p.data.copy() for p in params1}

        # === Phase 3: Resume from checkpoint and continue ===
        loaded = ExperimentLog.load_checkpoint(checkpoint_path)

        # Create fresh model
        np.random.seed(42)  # Same seed to create same initial structure
        params2 = [
            Parameter(data=np.random.randn(10), name="weights"),
            Parameter(data=np.random.randn(5), name="bias"),
        ]
        optimizer2 = Adam(params2, lr=0.05)  # Different initial LR
        scheduler2 = StepLR(optimizer2, step_size=25, gamma=0.5)

        # Restore from checkpoint
        ExperimentLog.restore_parameters(params2, loaded)
        optimizer2.load_state_dict(loaded["optimizer_state"])
        scheduler2.load_state_dict(loaded["scheduler_state"])

        # Verify restoration
        assert loaded["epoch"] == 50
        for p in params2:
            np.testing.assert_array_equal(p.data, params_at_50[p.name])
        assert optimizer2.lr == pytest.approx(lr_at_50)
        assert optimizer2.t == optimizer_t_at_50

        # Continue training from epoch 50 to 100 (resumed)
        np.random.seed(100)  # Same seed as original continuation
        for epoch in range(50, 100):
            for p in params2:
                p.grad = np.random.randn(*p.data.shape) * 0.1
            optimizer2.step()
            scheduler2.step(epoch=epoch)

        # === Verify: Resumed training matches original ===
        for p in params2:
            np.testing.assert_array_almost_equal(
                p.data, params_at_100_original[p.name],
                err_msg=f"Parameter {p.name} diverged after resume"
            )
