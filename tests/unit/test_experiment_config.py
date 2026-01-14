"""
Unit tests for ExperimentConfig: Experiment configuration dataclass.

Tests cover:
- Basic initialization
- Validation (embedding_dim divisible by num_heads, etc.)
- Serialization (to_dict, to_json, from_dict, from_json)
- File I/O (save, load)
- CLI argument conversion
- Summary generation
"""

import pytest
import json
import tempfile
import argparse
from pathlib import Path

from cortical.experiments.config import ExperimentConfig


# =============================================================================
# Basic Initialization Tests
# =============================================================================


class TestExperimentConfigInit:
    """Tests for basic ExperimentConfig initialization."""

    def test_minimal_config(self):
        """Test creating config with only required fields."""
        config = ExperimentConfig(
            name="test-experiment",
            input_path="data.txt"
        )

        assert config.name == "test-experiment"
        assert config.input_path == "data.txt"
        # Check defaults
        assert config.embedding_dim == 16
        assert config.num_heads == 2
        assert config.num_layers == 2
        assert config.epochs == 500
        assert config.lr == 0.03
        assert config.clip_grad == 1.0
        assert config.max_tokens == 50
        assert config.seed == 42

    def test_full_config(self):
        """Test creating config with all fields specified."""
        config = ExperimentConfig(
            name="full-experiment",
            input_path="full_data.txt",
            embedding_dim=64,
            num_heads=4,
            num_layers=6,
            epochs=1000,
            lr=0.001,
            clip_grad=0.5,
            max_tokens=100,
            seed=123,
            dropout=0.1,
            use_bias=True,
            residual=True,
            weight_decay=0.01,
            val_split=0.1,
            loss_fn="cross_entropy",
            position_encoding="learned",
        )

        assert config.name == "full-experiment"
        assert config.embedding_dim == 64
        assert config.num_heads == 4
        assert config.num_layers == 6
        assert config.epochs == 1000
        assert config.lr == 0.001
        assert config.clip_grad == 0.5
        assert config.max_tokens == 100
        assert config.seed == 123
        assert config.dropout == 0.1
        assert config.use_bias is True
        assert config.residual is True
        assert config.weight_decay == 0.01
        assert config.val_split == 0.1
        assert config.loss_fn == "cross_entropy"
        assert config.position_encoding == "learned"


# =============================================================================
# Validation Tests
# =============================================================================


class TestExperimentConfigValidation:
    """Tests for ExperimentConfig validation logic."""

    def test_invalid_embedding_dim_not_divisible_by_heads(self):
        """Test that embedding_dim must be divisible by num_heads."""
        with pytest.raises(ValueError, match="must be divisible"):
            ExperimentConfig(
                name="test",
                input_path="data.txt",
                embedding_dim=10,  # Not divisible by 4
                num_heads=4
            )

    def test_valid_embedding_dim_divisible_by_heads(self):
        """Test that valid divisible combination works."""
        config = ExperimentConfig(
            name="test",
            input_path="data.txt",
            embedding_dim=16,
            num_heads=4
        )
        assert config.embedding_dim == 16
        assert config.num_heads == 4

    def test_invalid_loss_function(self):
        """Test that invalid loss function raises error."""
        with pytest.raises(ValueError, match="not supported"):
            ExperimentConfig(
                name="test",
                input_path="data.txt",
                loss_fn="invalid_loss"
            )

    def test_valid_loss_functions(self):
        """Test that valid loss functions are accepted."""
        config_mse = ExperimentConfig(
            name="test", input_path="data.txt", loss_fn="mse"
        )
        config_ce = ExperimentConfig(
            name="test", input_path="data.txt", loss_fn="cross_entropy"
        )

        assert config_mse.loss_fn == "mse"
        assert config_ce.loss_fn == "cross_entropy"

    def test_invalid_position_encoding(self):
        """Test that invalid position encoding raises error."""
        with pytest.raises(ValueError, match="not supported"):
            ExperimentConfig(
                name="test",
                input_path="data.txt",
                position_encoding="invalid_encoding"
            )

    def test_valid_position_encodings(self):
        """Test that valid position encodings are accepted."""
        config_none = ExperimentConfig(
            name="test", input_path="data.txt", position_encoding="none"
        )
        config_learned = ExperimentConfig(
            name="test", input_path="data.txt", position_encoding="learned"
        )
        config_sinusoidal = ExperimentConfig(
            name="test", input_path="data.txt", position_encoding="sinusoidal"
        )

        assert config_none.position_encoding == "none"
        assert config_learned.position_encoding == "learned"
        assert config_sinusoidal.position_encoding == "sinusoidal"

    def test_invalid_val_split_negative(self):
        """Test that negative val_split raises error."""
        with pytest.raises(ValueError, match="between 0.0 and 0.5"):
            ExperimentConfig(
                name="test",
                input_path="data.txt",
                val_split=-0.1
            )

    def test_invalid_val_split_too_large(self):
        """Test that val_split > 0.5 raises error."""
        with pytest.raises(ValueError, match="between 0.0 and 0.5"):
            ExperimentConfig(
                name="test",
                input_path="data.txt",
                val_split=0.6
            )

    def test_valid_val_split_boundaries(self):
        """Test that val_split at boundaries is valid."""
        config_zero = ExperimentConfig(
            name="test", input_path="data.txt", val_split=0.0
        )
        config_half = ExperimentConfig(
            name="test", input_path="data.txt", val_split=0.5
        )

        assert config_zero.val_split == 0.0
        assert config_half.val_split == 0.5


# =============================================================================
# Serialization Tests
# =============================================================================


class TestExperimentConfigSerialization:
    """Tests for ExperimentConfig serialization methods."""

    @pytest.fixture
    def sample_config(self):
        """Create a sample config for serialization tests."""
        return ExperimentConfig(
            name="serialize-test",
            input_path="serialize_data.txt",
            embedding_dim=32,
            num_heads=4,
            num_layers=3,
            epochs=100,
            lr=0.02,
            dropout=0.1,
            use_bias=True,
            loss_fn="cross_entropy",
        )

    def test_to_dict(self, sample_config):
        """Test conversion to dictionary."""
        d = sample_config.to_dict()

        assert isinstance(d, dict)
        assert d["name"] == "serialize-test"
        assert d["input_path"] == "serialize_data.txt"
        assert d["embedding_dim"] == 32
        assert d["num_heads"] == 4
        assert d["num_layers"] == 3
        assert d["epochs"] == 100
        assert d["lr"] == 0.02
        assert d["dropout"] == 0.1
        assert d["use_bias"] is True
        assert d["loss_fn"] == "cross_entropy"

    def test_to_json(self, sample_config):
        """Test conversion to JSON string."""
        json_str = sample_config.to_json()

        assert isinstance(json_str, str)

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["name"] == "serialize-test"
        assert parsed["embedding_dim"] == 32

    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {
            "name": "from-dict-test",
            "input_path": "dict_data.txt",
            "embedding_dim": 64,
            "num_heads": 8,
        }

        config = ExperimentConfig.from_dict(d)

        assert config.name == "from-dict-test"
        assert config.input_path == "dict_data.txt"
        assert config.embedding_dim == 64
        assert config.num_heads == 8
        # Defaults should be filled in
        assert config.num_layers == 2

    def test_from_dict_ignores_unknown_fields(self):
        """Test that unknown fields are ignored."""
        d = {
            "name": "test",
            "input_path": "data.txt",
            "unknown_field": "should_be_ignored",
            "another_unknown": 123,
        }

        config = ExperimentConfig.from_dict(d)

        assert config.name == "test"
        assert not hasattr(config, "unknown_field")

    def test_from_json(self):
        """Test creation from JSON string."""
        json_str = '{"name": "from-json-test", "input_path": "json_data.txt", "embedding_dim": 32}'

        config = ExperimentConfig.from_json(json_str)

        assert config.name == "from-json-test"
        assert config.input_path == "json_data.txt"
        assert config.embedding_dim == 32

    def test_roundtrip_dict(self, sample_config):
        """Test roundtrip through dict serialization."""
        d = sample_config.to_dict()
        restored = ExperimentConfig.from_dict(d)

        assert restored.name == sample_config.name
        assert restored.input_path == sample_config.input_path
        assert restored.embedding_dim == sample_config.embedding_dim
        assert restored.dropout == sample_config.dropout
        assert restored.loss_fn == sample_config.loss_fn

    def test_roundtrip_json(self, sample_config):
        """Test roundtrip through JSON serialization."""
        json_str = sample_config.to_json()
        restored = ExperimentConfig.from_json(json_str)

        assert restored.name == sample_config.name
        assert restored.input_path == sample_config.input_path
        assert restored.embedding_dim == sample_config.embedding_dim


# =============================================================================
# File I/O Tests
# =============================================================================


class TestExperimentConfigFileIO:
    """Tests for ExperimentConfig file save/load."""

    @pytest.fixture
    def sample_config(self):
        """Create a sample config for file I/O tests."""
        return ExperimentConfig(
            name="file-io-test",
            input_path="file_data.txt",
            embedding_dim=32,
            num_heads=4,
        )

    def test_save_creates_file(self, sample_config):
        """Test that save creates a JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"

            sample_config.save(path)

            assert path.exists()
            content = path.read_text()
            assert "file-io-test" in content

    def test_save_creates_parent_directories(self, sample_config):
        """Test that save creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "deeply" / "config.json"

            sample_config.save(path)

            assert path.exists()

    def test_load_from_file(self, sample_config):
        """Test loading config from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            sample_config.save(path)

            loaded = ExperimentConfig.load(path)

            assert loaded.name == sample_config.name
            assert loaded.input_path == sample_config.input_path
            assert loaded.embedding_dim == sample_config.embedding_dim

    def test_roundtrip_file(self, sample_config):
        """Test complete save/load roundtrip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            sample_config.save(path)
            loaded = ExperimentConfig.load(path)

            # All fields should match
            assert loaded.to_dict() == sample_config.to_dict()


# =============================================================================
# CLI Arguments Tests
# =============================================================================


class TestExperimentConfigFromArgs:
    """Tests for ExperimentConfig.from_args method."""

    def test_from_args_basic(self):
        """Test creation from argparse namespace with basic args."""
        args = argparse.Namespace(
            name="cli-test",
            input="cli_data.txt",
            embedding_dim=16,
            num_heads=2,
            num_layers=2,
            epochs=100,
            lr=0.01,
            clip_grad=1.0,
            max_tokens=50,
            seed=42,
        )

        config = ExperimentConfig.from_args(args)

        assert config.name == "cli-test"
        assert config.input_path == "cli_data.txt"
        assert config.embedding_dim == 16
        assert config.epochs == 100

    def test_from_args_with_optional_fields(self):
        """Test creation from args with optional fields."""
        args = argparse.Namespace(
            name="cli-test",
            input="cli_data.txt",
            embedding_dim=32,
            num_heads=4,
            num_layers=4,
            epochs=200,
            lr=0.02,
            clip_grad=0.5,
            max_tokens=100,
            seed=123,
            dropout=0.2,
            use_bias=True,
            residual=True,
            weight_decay=0.01,
            val_split=0.2,
            loss_fn="cross_entropy",
            position_encoding="learned",
        )

        config = ExperimentConfig.from_args(args)

        assert config.dropout == 0.2
        assert config.use_bias is True
        assert config.residual is True
        assert config.weight_decay == 0.01
        assert config.val_split == 0.2
        assert config.loss_fn == "cross_entropy"
        assert config.position_encoding == "learned"

    def test_from_args_missing_optional_uses_defaults(self):
        """Test that missing optional args use defaults."""
        args = argparse.Namespace(
            name="cli-test",
            input="cli_data.txt",
            embedding_dim=16,
            num_heads=2,
            num_layers=2,
            epochs=100,
            lr=0.01,
            clip_grad=1.0,
            max_tokens=50,
            seed=42,
            # No optional fields
        )

        config = ExperimentConfig.from_args(args)

        # Should use defaults
        assert config.dropout == 0.0
        assert config.use_bias is False
        assert config.residual is False


# =============================================================================
# Summary Tests
# =============================================================================


class TestExperimentConfigSummary:
    """Tests for ExperimentConfig.summary method."""

    def test_summary_contains_all_fields(self):
        """Test that summary contains all important fields."""
        config = ExperimentConfig(
            name="summary-test",
            input_path="summary_data.txt",
            embedding_dim=64,
            num_heads=8,
            num_layers=6,
            epochs=500,
            lr=0.01,
            dropout=0.1,
            loss_fn="cross_entropy",
        )

        summary = config.summary()

        assert "summary-test" in summary
        assert "summary_data.txt" in summary
        assert "embedding_dim: 64" in summary
        assert "num_heads: 8" in summary
        assert "num_layers: 6" in summary
        assert "epochs: 500" in summary
        assert "lr: 0.01" in summary
        assert "dropout: 0.1" in summary
        assert "cross_entropy" in summary

    def test_summary_is_readable_string(self):
        """Test that summary is a readable multi-line string."""
        config = ExperimentConfig(
            name="test",
            input_path="data.txt",
        )

        summary = config.summary()

        assert isinstance(summary, str)
        assert "\n" in summary  # Multi-line
        assert "Experiment:" in summary
        assert "Architecture:" in summary
        assert "Training:" in summary


# =============================================================================
# Edge Cases
# =============================================================================


class TestExperimentConfigEdgeCases:
    """Tests for edge cases in ExperimentConfig."""

    def test_single_head_attention(self):
        """Test config with single attention head."""
        config = ExperimentConfig(
            name="test",
            input_path="data.txt",
            embedding_dim=16,
            num_heads=1
        )

        assert config.num_heads == 1

    def test_high_embedding_dimension(self):
        """Test config with high embedding dimension."""
        config = ExperimentConfig(
            name="test",
            input_path="data.txt",
            embedding_dim=512,
            num_heads=8
        )

        assert config.embedding_dim == 512

    def test_many_layers(self):
        """Test config with many layers."""
        config = ExperimentConfig(
            name="test",
            input_path="data.txt",
            num_layers=12
        )

        assert config.num_layers == 12

    def test_zero_dropout(self):
        """Test config with zero dropout (default)."""
        config = ExperimentConfig(
            name="test",
            input_path="data.txt",
            dropout=0.0
        )

        assert config.dropout == 0.0

    def test_high_dropout(self):
        """Test config with high dropout."""
        config = ExperimentConfig(
            name="test",
            input_path="data.txt",
            dropout=0.5
        )

        assert config.dropout == 0.5
