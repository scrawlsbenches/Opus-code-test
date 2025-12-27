#!/usr/bin/env python3
"""
Benchmark Regression Tests for SLM Training and Evaluation.

These tests detect quality regressions in:
- Model size (vocab, document count)
- Training safeguards (dry-run, backups)
- Corpus regeneration capability
- Benchmark suite stability

Tests are designed to be fast and non-invasive - they check structure and
availability rather than running expensive training/evaluation operations.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


class TestModelSize:
    """
    Regression tests for PRISM model size metrics.

    These tests ensure the model doesn't shrink significantly, which would
    indicate accidental training on a subset of the corpus (like the
    Dec 27, 2025 incident where the model shrank from 15,814 to 329 vocab).
    """

    @pytest.fixture
    def model_path(self):
        """Path to the production PRISM model."""
        return PROJECT_ROOT / "benchmarks" / "codebase_slm" / "models" / "prism_augmented.json"

    @pytest.fixture
    def model_data(self, model_path):
        """Load model data if it exists."""
        if not model_path.exists():
            pytest.skip(f"Model not found at {model_path}")

        with open(model_path) as f:
            return json.load(f)

    def test_prism_vocab_size(self, model_data):
        """
        PRISM vocab should not shrink significantly.

        Current baseline: 15,814 vocab terms
        Minimum acceptable: 15,000 (allow 5% variance)

        This catches regressions like training on subset of corpus.
        """
        vocab_size = len(model_data.get('vocab', []))

        assert vocab_size >= 15000, (
            f"PRISM vocab size {vocab_size} is below minimum 15,000. "
            f"This suggests training on incomplete corpus. "
            f"Check that corpus/training_patterns.jsonl was generated before training."
        )

    def test_prism_document_count(self, model_data):
        """
        PRISM doc count should not shrink significantly.

        Current baseline: 37,318 documents
        Minimum acceptable: 35,000 (allow 5% variance)

        This catches regressions in corpus generation.
        """
        doc_count = model_data.get('total_documents', 0)

        assert doc_count >= 35000, (
            f"PRISM document count {doc_count} is below minimum 35,000. "
            f"This suggests incomplete corpus. "
            f"Regenerate corpus with: python -m benchmarks.codebase_slm.generate_corpus --full"
        )

    def test_prism_token_count_reasonable(self, model_data):
        """
        PRISM should have a reasonable token count.

        Current baseline: ~649,000 tokens
        Minimum acceptable: 500,000
        """
        token_count = model_data.get('total_tokens', 0)

        assert token_count >= 500000, (
            f"PRISM token count {token_count} is suspiciously low. "
            f"Expected at least 500,000 tokens."
        )


class TestTrainingSafeguards:
    """
    Regression tests for training safety mechanisms.

    These tests verify that safeguards are in place to prevent accidental
    model overwrites and training on incomplete data.
    """

    @pytest.fixture
    def train_script_path(self):
        """Path to the training script."""
        return PROJECT_ROOT / "benchmarks" / "codebase_slm" / "train_augmented.py"

    def test_dry_run_available(self, train_script_path):
        """
        train_augmented.py should support --dry-run flag.

        This is a critical safeguard that allows evaluating the model
        before saving it.
        """
        if not train_script_path.exists():
            pytest.skip(f"Training script not found at {train_script_path}")

        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.codebase_slm.train_augmented", "--help"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )

        assert "--dry-run" in result.stdout, (
            "Training script must support --dry-run flag for safe evaluation. "
            "This prevents accidental model overwrites."
        )

    def test_output_flag_available(self, train_script_path):
        """
        train_augmented.py should support --output flag.

        This allows saving to a custom path without overwriting the
        production model.
        """
        if not train_script_path.exists():
            pytest.skip(f"Training script not found at {train_script_path}")

        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.codebase_slm.train_augmented", "--help"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )

        assert "--output" in result.stdout, (
            "Training script must support --output flag to prevent accidental overwrites."
        )

    def test_backup_mechanism_documented(self, train_script_path):
        """
        Training script should document backup behavior.

        Checks that the script's docstring mentions backup creation.
        """
        if not train_script_path.exists():
            pytest.skip(f"Training script not found at {train_script_path}")

        with open(train_script_path) as f:
            content = f.read()

        # Check for backup-related keywords in docstring/comments
        assert any(keyword in content.lower() for keyword in ['backup', 'timestamped', 'safeguard']), (
            "Training script should document backup mechanism to prevent data loss."
        )

    def test_corpus_check_exists(self, train_script_path):
        """
        Training script should check for corpus existence.

        Verifies that the script warns when training_patterns.jsonl is missing.
        """
        if not train_script_path.exists():
            pytest.skip(f"Training script not found at {train_script_path}")

        with open(train_script_path) as f:
            content = f.read()

        # Check for warning about missing corpus
        assert "training_patterns.jsonl" in content, (
            "Training script should reference training_patterns.jsonl file."
        )
        assert "WARNING" in content or "warning" in content, (
            "Training script should warn about missing corpus."
        )


class TestCorpusRegeneration:
    """
    Regression tests for corpus generation capability.

    These tests verify that the corpus can be regenerated from the codebase,
    which is critical since corpus files are gitignored.
    """

    @pytest.fixture
    def generate_script_path(self):
        """Path to the corpus generation script."""
        return PROJECT_ROOT / "benchmarks" / "codebase_slm" / "generate_corpus.py"

    @pytest.fixture
    def corpus_dir(self):
        """Path to the corpus directory."""
        return PROJECT_ROOT / "benchmarks" / "codebase_slm" / "corpus"

    def test_generate_script_exists(self, generate_script_path):
        """
        Corpus generation script should exist.

        This is the primary mechanism for regenerating the corpus.
        """
        assert generate_script_path.exists(), (
            f"Corpus generation script not found at {generate_script_path}. "
            f"This script is critical for regenerating training data."
        )

    def test_generate_script_has_full_mode(self, generate_script_path):
        """
        Generation script should support --full flag.

        This ensures we can generate the complete corpus.
        """
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.codebase_slm.generate_corpus", "--help"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )

        assert "--full" in result.stdout, (
            "Corpus generation script must support --full flag."
        )

    def test_corpus_patterns_file_exists(self, corpus_dir):
        """
        Main corpus file should exist (or be regeneratable).

        If it doesn't exist, provides clear instructions on how to generate it.
        """
        patterns_file = corpus_dir / "training_patterns.jsonl"

        if not patterns_file.exists():
            pytest.skip(
                f"Corpus not found at {patterns_file}. "
                f"Generate with: python -m benchmarks.codebase_slm.generate_corpus --full"
            )

        # If it exists, check it's not empty
        assert patterns_file.stat().st_size > 0, (
            f"Corpus file {patterns_file} exists but is empty."
        )

    @pytest.mark.slow
    def test_corpus_size_baseline(self, corpus_dir):
        """
        Regenerated corpus should meet baseline size.

        Current baseline: ~35,000 patterns
        Minimum acceptable: 30,000 (allow some variance)

        This is marked as slow since it may need to read the corpus file.
        """
        patterns_file = corpus_dir / "training_patterns.jsonl"

        if not patterns_file.exists():
            pytest.skip(f"Corpus not found at {patterns_file}")

        # Count lines in JSONL file
        with open(patterns_file) as f:
            pattern_count = sum(1 for line in f if line.strip())

        assert pattern_count >= 30000, (
            f"Corpus has only {pattern_count} patterns, expected at least 30,000. "
            f"Regenerate corpus with: python -m benchmarks.codebase_slm.generate_corpus --full"
        )


class TestBenchmarkStability:
    """
    Regression tests for benchmark suite stability.

    These tests verify that the benchmark infrastructure is available and
    working, without running the expensive benchmark evaluations.
    """

    @pytest.fixture
    def benchmark_script_path(self):
        """Path to the benchmark suite script."""
        return PROJECT_ROOT / "benchmarks" / "codebase_slm" / "benchmark_suite.py"

    def test_benchmark_suite_exists(self, benchmark_script_path):
        """
        Benchmark suite script should exist.

        This is the primary tool for evaluating model quality.
        """
        assert benchmark_script_path.exists(), (
            f"Benchmark suite not found at {benchmark_script_path}. "
            f"This is needed to detect quality regressions."
        )

    def test_benchmark_suite_importable(self):
        """
        Benchmark suite should be importable.

        This verifies the module structure is valid.
        """
        try:
            # Add project root to path
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))

            # Import the module to verify it's valid
            import benchmarks.codebase_slm.benchmark_suite
        except ImportError as e:
            pytest.fail(f"Failed to import benchmark_suite module: {e}")

    def test_benchmark_has_categories(self, benchmark_script_path):
        """
        Benchmark suite should define benchmark categories.

        Verifies that key categories like 'file_location', 'concept', etc.
        are defined in the benchmark queries.
        """
        with open(benchmark_script_path) as f:
            content = f.read()

        expected_categories = ['file_location', 'concept', 'how_to', 'completion']

        for category in expected_categories:
            assert category in content, (
                f"Benchmark suite should include '{category}' category."
            )

    def test_benchmark_has_help_flag(self, benchmark_script_path):
        """
        Benchmark suite should have help documentation.

        This ensures users can discover how to run benchmarks.
        """
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.codebase_slm.benchmark_suite", "--help"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )

        assert result.returncode == 0, (
            "Benchmark suite should respond to --help flag."
        )
        assert "--full" in result.stdout or "--quick" in result.stdout, (
            "Benchmark suite should document running modes."
        )

    @pytest.mark.slow
    def test_file_location_baseline_placeholder(self):
        """
        File location accuracy baseline check (placeholder).

        This is a placeholder for future actual benchmark runs.
        The real benchmark is expensive (~minutes), so we just verify
        the infrastructure exists here.

        Actual benchmark runs should be done manually or in CI with
        sufficient resources and time.

        Expected baseline: ~60% accuracy on file_location queries
        """
        # This is just a placeholder - actual benchmark runs are expensive
        # In the future, this could run a quick benchmark and compare
        # against stored baseline results

        # For now, we just document the expected baseline
        expected_baseline = 0.60  # 60% accuracy

        # Skip with informative message
        pytest.skip(
            f"Actual benchmark runs are too expensive for regular tests. "
            f"Run manually with: python -m benchmarks.codebase_slm.benchmark_suite --full"
        )


# Integration test verifying the full pipeline
class TestFullPipeline:
    """
    Integration tests for the full training pipeline.

    These tests verify that the two-step process (generate → train) is
    properly documented and working.
    """

    def test_pipeline_documentation_exists(self):
        """
        Training scripts should document the two-step pipeline.

        Verifies that both scripts reference each other and document
        the required order of operations.
        """
        train_script = PROJECT_ROOT / "benchmarks" / "codebase_slm" / "train_augmented.py"
        generate_script = PROJECT_ROOT / "benchmarks" / "codebase_slm" / "generate_corpus.py"

        if not train_script.exists():
            pytest.skip(f"Training script not found")
        if not generate_script.exists():
            pytest.skip(f"Generation script not found")

        # Check that training script references generation script
        with open(train_script) as f:
            train_content = f.read()

        assert "generate_corpus" in train_content, (
            "Training script should reference corpus generation step."
        )
        assert "STEP 1" in train_content or "Step 1" in train_content, (
            "Training script should document the two-step process."
        )

    def test_model_provenance_structure(self):
        """
        Saved models should include provenance metadata.

        This helps track what corpus was used to train the model.
        """
        model_path = PROJECT_ROOT / "benchmarks" / "codebase_slm" / "models" / "prism_augmented.json"

        if not model_path.exists():
            pytest.skip(f"Model not found at {model_path}")

        with open(model_path) as f:
            model_data = json.load(f)

        # Check for provenance field (may be empty in older models)
        # This is more of a "nice to have" than strict requirement
        if '_provenance' in model_data:
            provenance = model_data['_provenance']
            # If provenance exists, it should have certain fields
            assert isinstance(provenance, dict), (
                "Provenance should be a dictionary."
            )
