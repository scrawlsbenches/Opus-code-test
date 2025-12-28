"""
Tests for train_spark_from_git.py CLI script.

Following TDD: These tests define expected behavior before implementation.
"""

import pytest
import subprocess
import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestScriptHelp:
    """Test --help output."""

    def test_help_command(self):
        """Script should provide --help output."""
        result = subprocess.run(
            [sys.executable, 'scripts/train_spark_from_git.py', '--help'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert 'Train SparkSLM' in result.stdout
        assert 'train' in result.stdout
        assert 'stats' in result.stdout
        assert 'evaluate' in result.stdout

    def test_train_help(self):
        """train subcommand should have its own help."""
        result = subprocess.run(
            [sys.executable, 'scripts/train_spark_from_git.py', 'train', '--help'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert '--repo' in result.stdout
        assert '--branches' in result.stdout
        assert '--half-life' in result.stdout
        assert '--output' in result.stdout


class TestDryRun:
    """Test --dry-run mode."""

    def test_dry_run_no_file_creation(self, tmp_path):
        """dry-run should not create any files."""
        output = tmp_path / "model.json"

        result = subprocess.run(
            [
                sys.executable, 'scripts/train_spark_from_git.py', 'train',
                '--dry-run',
                '--output', str(output)
            ],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert not output.exists()
        assert 'DRY RUN' in result.stdout


class TestStatsCommand:
    """Test stats command."""

    def test_stats_runs_without_error(self):
        """stats command should run without error."""
        result = subprocess.run(
            [sys.executable, 'scripts/train_spark_from_git.py', 'stats'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert 'Git Training Statistics' in result.stdout

    def test_stats_shows_branch_weights(self):
        """stats should display branch weight configuration."""
        result = subprocess.run(
            [sys.executable, 'scripts/train_spark_from_git.py', 'stats'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert 'main' in result.stdout
        assert 'feature' in result.stdout
        assert 'claude' in result.stdout

    def test_stats_shows_quality_multipliers(self):
        """stats should display quality multipliers."""
        result = subprocess.run(
            [sys.executable, 'scripts/train_spark_from_git.py', 'stats'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert 'merged' in result.stdout
        assert 'tested' in result.stdout


class TestDemoMode:
    """Test --demo mode creates a valid model."""

    def test_demo_creates_model(self, tmp_path):
        """--demo should create a sample model file."""
        output = tmp_path / "test_model.json"

        result = subprocess.run(
            [
                sys.executable, 'scripts/train_spark_from_git.py', 'train',
                '--demo',
                '--output', str(output)
            ],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert output.exists()
        assert 'Demo model saved' in result.stdout

    def test_demo_model_is_valid_json(self, tmp_path):
        """Demo model should be valid JSON."""
        output = tmp_path / "test_model.json"

        subprocess.run(
            [
                sys.executable, 'scripts/train_spark_from_git.py', 'train',
                '--demo',
                '--output', str(output)
            ],
            capture_output=True,
            text=True
        )

        # Should be able to load as JSON
        with open(output) as f:
            data = json.load(f)

        assert 'n' in data  # NGramModel has 'n' field
        assert 'vocab' in data


class TestEvaluateCommand:
    """Test evaluate command."""

    def test_evaluate_nonexistent_model_fails(self, tmp_path):
        """evaluate should fail gracefully for nonexistent model."""
        nonexistent = tmp_path / "no_model.json"

        result = subprocess.run(
            [
                sys.executable, 'scripts/train_spark_from_git.py', 'evaluate',
                str(nonexistent)
            ],
            capture_output=True,
            text=True
        )

        assert result.returncode == 1
        assert 'not found' in result.stdout

    def test_evaluate_demo_model(self, tmp_path):
        """evaluate should work on a demo model."""
        output = tmp_path / "test_model.json"

        # Create demo model
        subprocess.run(
            [
                sys.executable, 'scripts/train_spark_from_git.py', 'train',
                '--demo',
                '--output', str(output)
            ],
            capture_output=True,
            text=True
        )

        # Evaluate it
        result = subprocess.run(
            [
                sys.executable, 'scripts/train_spark_from_git.py', 'evaluate',
                str(output)
            ],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert 'Loaded model' in result.stdout
        assert 'Vocabulary' in result.stdout
        assert 'Sample predictions' in result.stdout


class TestTrainArguments:
    """Test train command argument handling."""

    def test_custom_ngram_size(self, tmp_path):
        """Should accept custom n-gram size."""
        output = tmp_path / "model.json"

        result = subprocess.run(
            [
                sys.executable, 'scripts/train_spark_from_git.py', 'train',
                '--demo',
                '--ngram-size', '2',
                '--output', str(output)
            ],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0

        # Check model has correct n-gram size
        with open(output) as f:
            data = json.load(f)
        assert data['n'] == 2

    def test_custom_half_life(self):
        """Should accept custom temporal half-life."""
        result = subprocess.run(
            [
                sys.executable, 'scripts/train_spark_from_git.py', 'train',
                '--dry-run',
                '--half-life', '60.0'
            ],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert 'Half-life: 60.0' in result.stdout

    def test_custom_min_weight(self):
        """Should accept custom minimum weight."""
        result = subprocess.run(
            [
                sys.executable, 'scripts/train_spark_from_git.py', 'train',
                '--dry-run',
                '--min-weight', '0.2'
            ],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0


class TestNoCommand:
    """Test script behavior without a command."""

    def test_no_command_shows_help(self):
        """Running without command should show help."""
        result = subprocess.run(
            [sys.executable, 'scripts/train_spark_from_git.py'],
            capture_output=True,
            text=True
        )

        # Should exit with error code but show help
        assert result.returncode == 1
        assert 'usage:' in result.stdout.lower() or 'usage:' in result.stderr.lower()
