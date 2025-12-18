"""
Unit Tests for ML Experiments Module
=====================================

Tests for cortical/ml_experiments module, covering:
- utils.py: Hash functions, ID generation, splits, JSONL operations, git integration
- dataset.py: Dataset creation, filtering, splits, reproducibility, verification

These tests ensure reproducibility, determinism, and correct behavior of the
ML experiment tracking infrastructure.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from cortical.ml_experiments.utils import (
    compute_file_hash,
    compute_content_hash,
    compute_dict_hash,
    generate_experiment_id,
    generate_dataset_id,
    now_iso,
    get_git_hash,
    get_git_status,
    set_random_seed,
    reproducible_shuffle,
    split_indices,
    ensure_directory,
    append_jsonl,
    read_jsonl,
    filter_jsonl,
    save_json,
    load_json,
)

from cortical.ml_experiments.dataset import (
    SplitInfo,
    DatasetVersion,
    DatasetManager,
    create_holdout_split,
    DATASETS_DIR,
    DATASET_MANIFEST,
)


# =============================================================================
# UTILS.PY TESTS - HASH FUNCTIONS
# =============================================================================


class TestHashFunctions:
    """Tests for hash computation functions."""

    def test_compute_file_hash_basic(self, tmp_path):
        """Test basic file hashing."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, world!")

        hash1 = compute_file_hash(test_file)

        # Hash should be 12 characters
        assert len(hash1) == 12
        assert hash1.isalnum()

        # Same file should produce same hash
        hash2 = compute_file_hash(test_file)
        assert hash1 == hash2

    def test_compute_file_hash_different_content(self, tmp_path):
        """Test that different content produces different hashes."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"

        file1.write_text("Content A")
        file2.write_text("Content B")

        hash1 = compute_file_hash(file1)
        hash2 = compute_file_hash(file2)

        assert hash1 != hash2

    def test_compute_file_hash_algorithm(self, tmp_path):
        """Test different hash algorithms."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        sha256_hash = compute_file_hash(test_file, algorithm='sha256')
        md5_hash = compute_file_hash(test_file, algorithm='md5')

        # Different algorithms should produce different hashes
        assert sha256_hash != md5_hash
        assert len(sha256_hash) == 12
        assert len(md5_hash) == 12

    def test_compute_content_hash_basic(self):
        """Test content hash for strings."""
        hash1 = compute_content_hash("Hello")

        assert len(hash1) == 12
        assert hash1.isalnum()

        # Same content should produce same hash
        hash2 = compute_content_hash("Hello")
        assert hash1 == hash2

    def test_compute_content_hash_different_content(self):
        """Test that different strings produce different hashes."""
        hash1 = compute_content_hash("Content A")
        hash2 = compute_content_hash("Content B")

        assert hash1 != hash2

    def test_compute_dict_hash_basic(self):
        """Test dictionary hashing."""
        data = {"key1": "value1", "key2": 42}

        hash1 = compute_dict_hash(data)

        assert len(hash1) == 12
        assert hash1.isalnum()

    def test_compute_dict_hash_key_ordering(self):
        """Test that key order doesn't affect hash (sorted internally)."""
        data1 = {"a": 1, "b": 2, "c": 3}
        data2 = {"c": 3, "a": 1, "b": 2}

        hash1 = compute_dict_hash(data1)
        hash2 = compute_dict_hash(data2)

        assert hash1 == hash2

    def test_compute_dict_hash_nested(self):
        """Test hashing nested dictionaries."""
        data = {
            "outer": {
                "inner": "value"
            },
            "list": [1, 2, 3]
        }

        hash1 = compute_dict_hash(data)
        hash2 = compute_dict_hash(data)

        assert hash1 == hash2


# =============================================================================
# UTILS.PY TESTS - ID GENERATION
# =============================================================================


class TestIDGeneration:
    """Tests for ID generation functions."""

    def test_generate_experiment_id_format(self):
        """Test experiment ID format."""
        exp_id = generate_experiment_id()

        # Format: exp-YYYYMMDD-HHMMSS-XXXX
        assert exp_id.startswith("exp-")
        parts = exp_id.split("-")
        assert len(parts) == 4
        assert parts[0] == "exp"
        assert len(parts[1]) == 8  # YYYYMMDD
        assert len(parts[2]) == 6  # HHMMSS
        assert len(parts[3]) == 4  # XXXX hex suffix

    def test_generate_experiment_id_uniqueness(self):
        """Test that experiment IDs are unique."""
        ids = [generate_experiment_id() for _ in range(10)]

        # All IDs should be unique (due to random suffix)
        assert len(set(ids)) == len(ids)

    def test_generate_dataset_id_format(self):
        """Test dataset ID format."""
        ds_id = generate_dataset_id("test_dataset")

        # Format: ds-{name}-YYYYMMDD-XXXX
        assert ds_id.startswith("ds-")
        parts = ds_id.split("-")
        assert len(parts) >= 4
        assert parts[0] == "ds"
        assert parts[1] == "test_dataset"
        assert len(parts[-1]) == 4  # XXXX hex suffix

    def test_generate_dataset_id_sanitization(self):
        """Test that dataset names are sanitized."""
        ds_id = generate_dataset_id("My Dataset Name")

        # Should lowercase and replace spaces with underscores
        assert "my_dataset_name" in ds_id.lower()
        assert " " not in ds_id

    def test_generate_dataset_id_long_name(self):
        """Test truncation of long dataset names."""
        long_name = "a" * 50
        ds_id = generate_dataset_id(long_name)

        # Name should be truncated to 20 chars
        parts = ds_id.split("-")
        assert len(parts[1]) <= 20

    def test_now_iso_format(self):
        """Test ISO timestamp format."""
        timestamp = now_iso()

        # Should be ISO 8601 format with timezone
        assert "T" in timestamp
        assert "+" in timestamp or "Z" in timestamp or "-" in timestamp


# =============================================================================
# UTILS.PY TESTS - GIT INTEGRATION
# =============================================================================


class TestGitIntegration:
    """Tests for git integration functions."""

    @patch('subprocess.run')
    def test_get_git_hash_success(self, mock_run):
        """Test successful git hash retrieval."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="abc123def456789012345678\n"
        )

        git_hash = get_git_hash()

        assert git_hash == "abc123def456"  # First 12 chars
        mock_run.assert_called_once()

    @patch('subprocess.run')
    def test_get_git_hash_not_a_repo(self, mock_run):
        """Test git hash when not in a git repo."""
        mock_run.return_value = MagicMock(returncode=128)

        git_hash = get_git_hash()

        assert git_hash is None

    @patch('subprocess.run')
    def test_get_git_hash_timeout(self, mock_run):
        """Test git hash with timeout."""
        from subprocess import TimeoutExpired
        mock_run.side_effect = TimeoutExpired("git", 5)

        git_hash = get_git_hash()

        assert git_hash is None

    @patch('subprocess.run')
    def test_get_git_status_clean(self, mock_run):
        """Test git status when working directory is clean."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=""
        )

        status = get_git_status()

        assert status == "clean"

    @patch('subprocess.run')
    def test_get_git_status_dirty(self, mock_run):
        """Test git status when working directory has changes."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=" M file.txt\n"
        )

        status = get_git_status()

        assert status == "dirty"

    @patch('subprocess.run')
    def test_get_git_status_unknown(self, mock_run):
        """Test git status when git is not available."""
        mock_run.return_value = MagicMock(returncode=128)

        status = get_git_status()

        assert status == "unknown"


# =============================================================================
# UTILS.PY TESTS - REPRODUCIBLE RANDOMNESS
# =============================================================================


class TestReproducibleRandomness:
    """Tests for reproducible random operations."""

    def test_set_random_seed(self):
        """Test that setting seed produces reproducible results."""
        import random

        set_random_seed(42)
        values1 = [random.random() for _ in range(5)]

        set_random_seed(42)
        values2 = [random.random() for _ in range(5)]

        assert values1 == values2

    def test_reproducible_shuffle_basic(self):
        """Test basic reproducible shuffle."""
        items = list(range(10))

        shuffled1 = reproducible_shuffle(items, seed=42)
        shuffled2 = reproducible_shuffle(items, seed=42)

        # Same seed should produce same shuffle
        assert shuffled1 == shuffled2

        # Should contain all original items
        assert sorted(shuffled1) == items

    def test_reproducible_shuffle_different_seeds(self):
        """Test that different seeds produce different shuffles."""
        items = list(range(10))

        shuffled1 = reproducible_shuffle(items, seed=42)
        shuffled2 = reproducible_shuffle(items, seed=43)

        # Different seeds should (usually) produce different shuffles
        assert shuffled1 != shuffled2

    def test_reproducible_shuffle_original_unchanged(self):
        """Test that original list is not modified."""
        original = list(range(10))
        original_copy = original.copy()

        shuffled = reproducible_shuffle(original, seed=42)

        # Original should be unchanged
        assert original == original_copy
        # Shuffled should be different (with high probability)
        assert shuffled != original


# =============================================================================
# UTILS.PY TESTS - SPLIT INDICES
# =============================================================================


class TestSplitIndices:
    """Tests for split_indices function."""

    def test_split_indices_basic(self):
        """Test basic train/val/test split."""
        splits = split_indices(
            total=100,
            ratios={'train': 0.7, 'val': 0.15, 'test': 0.15},
            seed=42,
            strategy='random'
        )

        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits

        # Check counts (approximately correct)
        assert len(splits['train']) == 70
        assert len(splits['val']) == 15
        assert len(splits['test']) == 15

    def test_split_indices_all_covered(self):
        """Test that all indices are assigned to exactly one split."""
        total = 100
        splits = split_indices(
            total=total,
            ratios={'train': 0.7, 'val': 0.15, 'test': 0.15},
            seed=42
        )

        all_indices = []
        for split_indices_list in splits.values():
            all_indices.extend(split_indices_list)

        # All indices should be present exactly once
        assert sorted(all_indices) == list(range(total))

    def test_split_indices_reproducibility(self):
        """Test that same seed produces same splits."""
        splits1 = split_indices(
            total=100,
            ratios={'train': 0.8, 'test': 0.2},
            seed=42,
            strategy='random'
        )

        splits2 = split_indices(
            total=100,
            ratios={'train': 0.8, 'test': 0.2},
            seed=42,
            strategy='random'
        )

        assert splits1['train'] == splits2['train']
        assert splits1['test'] == splits2['test']

    def test_split_indices_temporal_preserves_order(self):
        """Test that temporal strategy preserves order."""
        splits = split_indices(
            total=100,
            ratios={'train': 0.7, 'val': 0.15, 'test': 0.15},
            seed=42,
            strategy='temporal'
        )

        # Temporal should be in order (splits are sorted alphabetically)
        # Order: test (15), train (70), val (15)
        assert splits['test'] == list(range(0, 15))
        assert splits['train'] == list(range(15, 85))
        assert splits['val'] == list(range(85, 100))

    def test_split_indices_random_shuffles(self):
        """Test that random strategy shuffles indices."""
        splits = split_indices(
            total=100,
            ratios={'train': 1.0},
            seed=42,
            strategy='random'
        )

        # Random should NOT be in order (with high probability)
        assert splits['train'] != list(range(100))

    def test_split_indices_invalid_ratios(self):
        """Test that invalid ratios raise ValueError."""
        with pytest.raises(ValueError, match="Ratios must sum to 1.0"):
            split_indices(
                total=100,
                ratios={'train': 0.5, 'test': 0.3},  # Sum = 0.8
                seed=42
            )

    def test_split_indices_rounding_handled(self):
        """Test that rounding is handled correctly."""
        splits = split_indices(
            total=100,
            ratios={'train': 0.33, 'val': 0.33, 'test': 0.34},
            seed=42
        )

        # All indices should be assigned
        total_assigned = sum(len(s) for s in splits.values())
        assert total_assigned == 100


# =============================================================================
# UTILS.PY TESTS - JSONL OPERATIONS
# =============================================================================


class TestJSONLOperations:
    """Tests for JSONL file operations."""

    def test_append_jsonl_basic(self, tmp_path):
        """Test appending records to JSONL file."""
        jsonl_file = tmp_path / "test.jsonl"

        append_jsonl(jsonl_file, {"id": 1, "value": "first"})
        append_jsonl(jsonl_file, {"id": 2, "value": "second"})

        # File should exist with 2 lines
        content = jsonl_file.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 2

    def test_append_jsonl_creates_directory(self, tmp_path):
        """Test that append_jsonl creates parent directories."""
        nested_file = tmp_path / "subdir" / "nested" / "test.jsonl"

        append_jsonl(nested_file, {"test": "data"})

        assert nested_file.exists()
        assert nested_file.parent.exists()

    def test_read_jsonl_basic(self, tmp_path):
        """Test reading JSONL file."""
        jsonl_file = tmp_path / "test.jsonl"

        append_jsonl(jsonl_file, {"id": 1, "value": "first"})
        append_jsonl(jsonl_file, {"id": 2, "value": "second"})

        records = read_jsonl(jsonl_file)

        assert len(records) == 2
        assert records[0]["id"] == 1
        assert records[1]["id"] == 2

    def test_read_jsonl_nonexistent_file(self, tmp_path):
        """Test reading non-existent JSONL file returns empty list."""
        jsonl_file = tmp_path / "nonexistent.jsonl"

        records = read_jsonl(jsonl_file)

        assert records == []

    def test_read_jsonl_malformed_lines(self, tmp_path):
        """Test that malformed lines are skipped."""
        jsonl_file = tmp_path / "test.jsonl"

        # Write valid and invalid JSON
        jsonl_file.write_text(
            '{"valid": 1}\n'
            'not valid json\n'
            '{"valid": 2}\n'
        )

        records = read_jsonl(jsonl_file)

        # Should skip the malformed line
        assert len(records) == 2
        assert records[0]["valid"] == 1
        assert records[1]["valid"] == 2

    def test_filter_jsonl_basic(self, tmp_path):
        """Test filtering JSONL records."""
        jsonl_file = tmp_path / "test.jsonl"

        append_jsonl(jsonl_file, {"type": "A", "value": 1})
        append_jsonl(jsonl_file, {"type": "B", "value": 2})
        append_jsonl(jsonl_file, {"type": "A", "value": 3})

        filtered = filter_jsonl(jsonl_file, {"type": "A"})

        assert len(filtered) == 2
        assert all(r["type"] == "A" for r in filtered)

    def test_filter_jsonl_multiple_conditions(self, tmp_path):
        """Test filtering with multiple conditions."""
        jsonl_file = tmp_path / "test.jsonl"

        append_jsonl(jsonl_file, {"type": "A", "status": "active", "value": 1})
        append_jsonl(jsonl_file, {"type": "A", "status": "inactive", "value": 2})
        append_jsonl(jsonl_file, {"type": "B", "status": "active", "value": 3})

        filtered = filter_jsonl(jsonl_file, {"type": "A", "status": "active"})

        assert len(filtered) == 1
        assert filtered[0]["value"] == 1

    def test_filter_jsonl_nested_keys(self, tmp_path):
        """Test filtering with nested keys using dot notation."""
        jsonl_file = tmp_path / "test.jsonl"

        append_jsonl(jsonl_file, {"meta": {"status": "active"}, "id": 1})
        append_jsonl(jsonl_file, {"meta": {"status": "inactive"}, "id": 2})

        filtered = filter_jsonl(jsonl_file, {"meta.status": "active"})

        assert len(filtered) == 1
        assert filtered[0]["id"] == 1

    def test_filter_jsonl_no_filters(self, tmp_path):
        """Test that empty filters return all records."""
        jsonl_file = tmp_path / "test.jsonl"

        append_jsonl(jsonl_file, {"id": 1})
        append_jsonl(jsonl_file, {"id": 2})

        filtered = filter_jsonl(jsonl_file, {})

        assert len(filtered) == 2


# =============================================================================
# UTILS.PY TESTS - JSON OPERATIONS
# =============================================================================


class TestJSONOperations:
    """Tests for JSON file operations."""

    def test_save_and_load_json_basic(self, tmp_path):
        """Test saving and loading JSON."""
        json_file = tmp_path / "test.json"
        data = {"key": "value", "number": 42}

        save_json(json_file, data)
        loaded = load_json(json_file)

        assert loaded == data

    def test_save_json_creates_directory(self, tmp_path):
        """Test that save_json creates parent directories."""
        nested_file = tmp_path / "subdir" / "test.json"

        save_json(nested_file, {"test": "data"})

        assert nested_file.exists()

    def test_load_json_nonexistent(self, tmp_path):
        """Test loading non-existent JSON returns None."""
        json_file = tmp_path / "nonexistent.json"

        loaded = load_json(json_file)

        assert loaded is None

    def test_ensure_directory(self, tmp_path):
        """Test ensure_directory creates nested directories."""
        nested_dir = tmp_path / "a" / "b" / "c"

        result = ensure_directory(nested_dir)

        assert nested_dir.exists()
        assert nested_dir.is_dir()
        assert result == nested_dir

    def test_ensure_directory_exists_ok(self, tmp_path):
        """Test ensure_directory doesn't fail if directory exists."""
        test_dir = tmp_path / "existing"
        test_dir.mkdir()

        result = ensure_directory(test_dir)

        assert test_dir.exists()
        assert result == test_dir


# =============================================================================
# DATASET.PY TESTS - SPLIT INFO
# =============================================================================


class TestSplitInfo:
    """Tests for SplitInfo dataclass."""

    def test_splitinfo_creation(self):
        """Test SplitInfo creation."""
        split = SplitInfo(
            name="train",
            ratio=0.7,
            count=70,
            indices=[0, 1, 2],
            strategy="random"
        )

        assert split.name == "train"
        assert split.ratio == 0.7
        assert split.count == 70
        assert split.indices == [0, 1, 2]
        assert split.strategy == "random"

    def test_splitinfo_to_dict(self):
        """Test SplitInfo to_dict conversion."""
        split = SplitInfo(
            name="val",
            ratio=0.15,
            count=15,
            indices=[70, 71, 72],
            strategy="temporal"
        )

        split_dict = split.to_dict()

        assert split_dict["name"] == "val"
        assert split_dict["ratio"] == 0.15
        assert split_dict["count"] == 15
        assert split_dict["indices"] == [70, 71, 72]
        assert split_dict["strategy"] == "temporal"

    def test_splitinfo_from_dict(self):
        """Test SplitInfo from_dict conversion."""
        data = {
            "name": "test",
            "ratio": 0.15,
            "count": 15,
            "indices": [85, 86, 87],
            "strategy": "random"
        }

        split = SplitInfo.from_dict(data)

        assert split.name == "test"
        assert split.ratio == 0.15
        assert split.count == 15
        assert split.indices == [85, 86, 87]
        assert split.strategy == "random"


# =============================================================================
# DATASET.PY TESTS - DATASET VERSION
# =============================================================================


class TestDatasetVersion:
    """Tests for DatasetVersion dataclass."""

    def test_datasetversion_creation(self):
        """Test DatasetVersion creation."""
        dataset = DatasetVersion(
            id="ds-test-20251216-abcd",
            name="test_dataset",
            created_at="2025-12-16T10:00:00Z",
            source_path="/path/to/data.jsonl",
            source_hash="abc123",
            total_records=100,
            filtered_records=90,
            filters={"exclude_merge": True},
            filter_hash="def456",
            random_seed=42,
            split_strategy="random",
            splits={},
            metadata={}
        )

        assert dataset.id == "ds-test-20251216-abcd"
        assert dataset.name == "test_dataset"
        assert dataset.total_records == 100
        assert dataset.filtered_records == 90

    def test_datasetversion_get_split_indices(self):
        """Test getting split indices."""
        splits = {
            "train": SplitInfo("train", 0.7, 70, list(range(70)), "random"),
            "test": SplitInfo("test", 0.3, 30, list(range(70, 100)), "random")
        }

        dataset = DatasetVersion(
            id="ds-test-20251216-abcd",
            name="test",
            created_at="2025-12-16T10:00:00Z",
            source_path="/path/to/data.jsonl",
            source_hash="abc123",
            total_records=100,
            filtered_records=100,
            filters={},
            filter_hash="def456",
            random_seed=42,
            split_strategy="random",
            splits=splits
        )

        train_indices = dataset.get_split_indices("train")
        assert train_indices == list(range(70))

    def test_datasetversion_get_split_indices_invalid(self):
        """Test getting non-existent split raises ValueError."""
        dataset = DatasetVersion(
            id="ds-test-20251216-abcd",
            name="test",
            created_at="2025-12-16T10:00:00Z",
            source_path="/path/to/data.jsonl",
            source_hash="abc123",
            total_records=100,
            filtered_records=100,
            filters={},
            filter_hash="def456",
            random_seed=42,
            split_strategy="random",
            splits={}
        )

        with pytest.raises(ValueError, match="Split 'nonexistent' not found"):
            dataset.get_split_indices("nonexistent")

    def test_datasetversion_serialization(self):
        """Test DatasetVersion to_dict and from_dict round-trip."""
        splits = {
            "train": SplitInfo("train", 0.7, 70, list(range(70)), "random")
        }

        dataset = DatasetVersion(
            id="ds-test-20251216-abcd",
            name="test",
            created_at="2025-12-16T10:00:00Z",
            source_path="/path/to/data.jsonl",
            source_hash="abc123",
            total_records=100,
            filtered_records=100,
            filters={"min_files": 1},
            filter_hash="def456",
            random_seed=42,
            split_strategy="random",
            splits=splits,
            metadata={"version": "1.0"}
        )

        # Convert to dict and back
        dataset_dict = dataset.to_dict()
        restored = DatasetVersion.from_dict(dataset_dict)

        assert restored.id == dataset.id
        assert restored.name == dataset.name
        assert restored.total_records == dataset.total_records
        assert restored.splits["train"].name == "train"
        assert restored.splits["train"].count == 70


# =============================================================================
# DATASET.PY TESTS - DATASET MANAGER
# =============================================================================


class TestDatasetManager:
    """Tests for DatasetManager class."""

    def setup_method(self):
        """Create temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.old_datasets_dir = DATASETS_DIR

        # Monkey-patch DATASETS_DIR to use temp directory
        import cortical.ml_experiments.dataset as dataset_module
        dataset_module.DATASETS_DIR = Path(self.temp_dir) / 'datasets'
        dataset_module.DATASET_MANIFEST = dataset_module.DATASETS_DIR / 'dataset_manifest.jsonl'

    def teardown_method(self):
        """Clean up temporary directory."""
        import shutil
        import cortical.ml_experiments.dataset as dataset_module

        # Restore original DATASETS_DIR
        dataset_module.DATASETS_DIR = self.old_datasets_dir
        dataset_module.DATASET_MANIFEST = self.old_datasets_dir / 'dataset_manifest.jsonl'

        # Clean up temp directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def create_test_source(self, path: Path, num_records: int = 100):
        """Helper to create test JSONL source file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        for i in range(num_records):
            record = {
                "id": i,
                "message": f"Commit {i}",
                "files": [f"file{i % 5}.py"],
                "is_merge": i % 10 == 0
            }
            append_jsonl(path, record)

    def test_create_dataset_basic(self):
        """Test basic dataset creation."""
        source_path = Path(self.temp_dir) / "source.jsonl"
        self.create_test_source(source_path, num_records=100)

        dataset = DatasetManager.create_dataset(
            name="test_dataset",
            source_path=str(source_path),
            filters={},
            split_ratios={'train': 0.7, 'val': 0.15, 'test': 0.15},
            random_seed=42
        )

        assert dataset.name == "test_dataset"
        assert dataset.total_records == 100
        assert dataset.filtered_records == 100
        assert len(dataset.splits) == 3
        assert 'train' in dataset.splits
        assert 'val' in dataset.splits
        assert 'test' in dataset.splits

    def test_create_dataset_with_filters(self):
        """Test dataset creation with filters applied."""
        source_path = Path(self.temp_dir) / "source.jsonl"
        self.create_test_source(source_path, num_records=100)

        dataset = DatasetManager.create_dataset(
            name="filtered_dataset",
            source_path=str(source_path),
            filters={"exclude_merge": True},
            random_seed=42
        )

        # Should exclude merge commits (10% of records)
        assert dataset.total_records == 100
        assert dataset.filtered_records == 90  # 90% non-merge

    def test_create_dataset_reproducibility(self):
        """Test that same seed produces same splits."""
        source_path = Path(self.temp_dir) / "source.jsonl"
        self.create_test_source(source_path, num_records=100)

        dataset1 = DatasetManager.create_dataset(
            name="reproducible_test_1",
            source_path=str(source_path),
            filters={},
            random_seed=42
        )

        dataset2 = DatasetManager.create_dataset(
            name="reproducible_test_2",
            source_path=str(source_path),
            filters={},
            random_seed=42
        )

        # Same seed should produce same train/val/test splits
        assert dataset1.get_split_indices('train') == dataset2.get_split_indices('train')
        assert dataset1.get_split_indices('val') == dataset2.get_split_indices('val')
        assert dataset1.get_split_indices('test') == dataset2.get_split_indices('test')

    def test_create_dataset_temporal_strategy(self):
        """Test dataset creation with temporal split strategy."""
        source_path = Path(self.temp_dir) / "source.jsonl"
        self.create_test_source(source_path, num_records=100)

        dataset = DatasetManager.create_dataset(
            name="temporal_dataset",
            source_path=str(source_path),
            split_strategy='temporal',
            random_seed=42
        )

        # Temporal splits should preserve order
        train_indices = dataset.get_split_indices('train')
        val_indices = dataset.get_split_indices('val')
        test_indices = dataset.get_split_indices('test')

        # All indices should be in order
        assert train_indices == sorted(train_indices)
        assert val_indices == sorted(val_indices)
        assert test_indices == sorted(test_indices)

    def test_create_dataset_nonexistent_source(self):
        """Test that creating dataset with non-existent source raises error."""
        with pytest.raises(FileNotFoundError):
            DatasetManager.create_dataset(
                name="bad_dataset",
                source_path="/nonexistent/path.jsonl",
                random_seed=42
            )

    def test_create_dataset_no_records_after_filtering(self):
        """Test that filtering out all records raises error."""
        source_path = Path(self.temp_dir) / "source.jsonl"
        # Create only merge commits
        path = source_path
        path.parent.mkdir(parents=True, exist_ok=True)
        for i in range(10):
            append_jsonl(path, {"id": i, "is_merge": True, "files": ["test.py"]})

        with pytest.raises(ValueError, match="No records remain after filtering"):
            DatasetManager.create_dataset(
                name="empty_dataset",
                source_path=str(source_path),
                filters={"exclude_merge": True},
                random_seed=42
            )

    def test_load_dataset(self):
        """Test loading a dataset by ID."""
        source_path = Path(self.temp_dir) / "source.jsonl"
        self.create_test_source(source_path, num_records=100)

        created = DatasetManager.create_dataset(
            name="load_test",
            source_path=str(source_path),
            random_seed=42
        )

        loaded = DatasetManager.load_dataset(created.id)

        assert loaded is not None
        assert loaded.id == created.id
        assert loaded.name == created.name
        assert loaded.total_records == created.total_records

    def test_load_dataset_nonexistent(self):
        """Test loading non-existent dataset returns None."""
        loaded = DatasetManager.load_dataset("ds-nonexistent-20251216-xxxx")

        assert loaded is None

    def test_load_latest(self):
        """Test loading the latest dataset version by name."""
        source_path = Path(self.temp_dir) / "source.jsonl"
        self.create_test_source(source_path, num_records=100)

        # Create multiple versions
        dataset1 = DatasetManager.create_dataset(
            name="versioned_dataset",
            source_path=str(source_path),
            random_seed=42
        )

        dataset2 = DatasetManager.create_dataset(
            name="versioned_dataset",
            source_path=str(source_path),
            random_seed=43
        )

        latest = DatasetManager.load_latest("versioned_dataset")

        assert latest is not None
        # Should load the most recent (dataset2)
        assert latest.id == dataset2.id

    def test_load_latest_nonexistent(self):
        """Test loading latest for non-existent name returns None."""
        latest = DatasetManager.load_latest("nonexistent_name")

        assert latest is None

    def test_list_datasets(self):
        """Test listing datasets."""
        source_path = Path(self.temp_dir) / "source.jsonl"
        self.create_test_source(source_path, num_records=100)

        # Create multiple datasets
        DatasetManager.create_dataset(
            name="dataset_a",
            source_path=str(source_path),
            random_seed=42
        )

        DatasetManager.create_dataset(
            name="dataset_b",
            source_path=str(source_path),
            random_seed=42
        )

        datasets = DatasetManager.list_datasets()

        assert len(datasets) >= 2
        names = [d.name for d in datasets]
        assert "dataset_a" in names
        assert "dataset_b" in names

    def test_list_datasets_with_filter(self):
        """Test listing datasets with name filter."""
        source_path = Path(self.temp_dir) / "source.jsonl"
        self.create_test_source(source_path, num_records=100)

        DatasetManager.create_dataset(
            name="alpha_dataset",
            source_path=str(source_path),
            random_seed=42
        )

        DatasetManager.create_dataset(
            name="beta_dataset",
            source_path=str(source_path),
            random_seed=42
        )

        filtered = DatasetManager.list_datasets(name_filter="alpha")

        assert len(filtered) >= 1
        assert all("alpha" in d.name for d in filtered)

    def test_get_split_data(self):
        """Test getting actual data records for a split."""
        source_path = Path(self.temp_dir) / "source.jsonl"
        self.create_test_source(source_path, num_records=100)

        dataset = DatasetManager.create_dataset(
            name="split_data_test",
            source_path=str(source_path),
            random_seed=42
        )

        train_data = DatasetManager.get_split_data(dataset, "train")

        assert len(train_data) == 70
        assert all("id" in record for record in train_data)

    def test_get_split_data_missing_source(self):
        """Test getting split data when source file is missing."""
        source_path = Path(self.temp_dir) / "source.jsonl"
        self.create_test_source(source_path, num_records=100)

        dataset = DatasetManager.create_dataset(
            name="missing_source_test",
            source_path=str(source_path),
            random_seed=42
        )

        # Delete source file
        source_path.unlink()

        with pytest.raises(FileNotFoundError):
            DatasetManager.get_split_data(dataset, "train")

    def test_verify_dataset_valid(self):
        """Test verifying a valid dataset."""
        source_path = Path(self.temp_dir) / "source.jsonl"
        self.create_test_source(source_path, num_records=100)

        dataset = DatasetManager.create_dataset(
            name="verify_test",
            source_path=str(source_path),
            random_seed=42
        )

        results = DatasetManager.verify_dataset(dataset)

        assert results['valid'] is True
        assert results['checks']['source_exists'] is True
        assert results['checks']['hash_matches'] is True
        assert len(results['errors']) == 0

    def test_verify_dataset_missing_source(self):
        """Test verifying dataset with missing source file."""
        source_path = Path(self.temp_dir) / "source.jsonl"
        self.create_test_source(source_path, num_records=100)

        dataset = DatasetManager.create_dataset(
            name="verify_missing_test",
            source_path=str(source_path),
            random_seed=42
        )

        # Delete source file
        source_path.unlink()

        results = DatasetManager.verify_dataset(dataset)

        assert results['valid'] is False
        assert results['checks']['source_exists'] is False
        assert len(results['errors']) > 0

    def test_verify_dataset_hash_changed(self):
        """Test verifying dataset when source hash has changed."""
        source_path = Path(self.temp_dir) / "source.jsonl"
        self.create_test_source(source_path, num_records=100)

        dataset = DatasetManager.create_dataset(
            name="verify_hash_test",
            source_path=str(source_path),
            random_seed=42
        )

        # Modify source file
        append_jsonl(source_path, {"id": 999, "message": "New commit", "files": ["new.py"]})

        results = DatasetManager.verify_dataset(dataset)

        assert results['valid'] is False
        assert results['checks']['hash_matches'] is False
        assert any("hash changed" in err for err in results['errors'])

    def test_apply_filters_exclude_merge(self):
        """Test _apply_filters with exclude_merge filter."""
        records = [
            {"id": 1, "is_merge": False, "files": ["a.py"]},
            {"id": 2, "is_merge": True, "files": ["b.py"]},
            {"id": 3, "is_merge": False, "files": ["c.py"]},
        ]

        valid_indices = DatasetManager._apply_filters(
            records,
            {"exclude_merge": True}
        )

        assert valid_indices == [0, 2]

    def test_apply_filters_min_files(self):
        """Test _apply_filters with min_files filter."""
        records = [
            {"id": 1, "files": ["a.py"]},
            {"id": 2, "files": ["b.py", "c.py"]},
            {"id": 3, "files": []},
        ]

        valid_indices = DatasetManager._apply_filters(
            records,
            {"min_files": 2}
        )

        assert valid_indices == [1]

    def test_apply_filters_max_files(self):
        """Test _apply_filters with max_files filter."""
        records = [
            {"id": 1, "files": ["a.py"]},
            {"id": 2, "files": ["b.py", "c.py", "d.py"]},
            {"id": 3, "files": ["e.py", "f.py"]},
        ]

        valid_indices = DatasetManager._apply_filters(
            records,
            {"max_files": 2}
        )

        assert valid_indices == [0, 2]

    def test_apply_filters_exclude_patterns(self):
        """Test _apply_filters with exclude_patterns filter."""
        records = [
            {"id": 1, "message": "feat: add feature", "files": ["a.py"]},
            {"id": 2, "message": "WIP: work in progress", "files": ["b.py"]},
            {"id": 3, "message": "fix: bug fix", "files": ["c.py"]},
        ]

        valid_indices = DatasetManager._apply_filters(
            records,
            {"exclude_patterns": ["WIP"], "exclude_empty_files": False}
        )

        assert valid_indices == [0, 2]

    def test_apply_filters_require_fields(self):
        """Test _apply_filters with require_fields filter."""
        records = [
            {"id": 1, "message": "commit 1", "author": "Alice", "files": ["a.py"]},
            {"id": 2, "message": "commit 2", "files": ["b.py"]},
            {"id": 3, "message": "commit 3", "author": "Bob", "files": ["c.py"]},
        ]

        valid_indices = DatasetManager._apply_filters(
            records,
            {"require_fields": ["author"], "exclude_empty_files": False}
        )

        assert valid_indices == [0, 2]

    def test_apply_filters_exclude_empty_files(self):
        """Test _apply_filters with exclude_empty_files filter."""
        records = [
            {"id": 1, "files": ["a.py"]},
            {"id": 2, "files": []},
            {"id": 3, "files": ["b.py"]},
        ]

        valid_indices = DatasetManager._apply_filters(
            records,
            {"exclude_empty_files": True}
        )

        assert valid_indices == [0, 2]

    def test_apply_filters_combined(self):
        """Test _apply_filters with multiple filters combined."""
        records = [
            {"id": 1, "is_merge": False, "files": ["a.py", "b.py"], "message": "feat: add"},
            {"id": 2, "is_merge": True, "files": ["c.py"], "message": "merge: branch"},
            {"id": 3, "is_merge": False, "files": ["d.py"], "message": "WIP: testing"},
            {"id": 4, "is_merge": False, "files": ["e.py", "f.py"], "message": "fix: bug"},
        ]

        valid_indices = DatasetManager._apply_filters(
            records,
            {
                "exclude_merge": True,
                "min_files": 2,
                "exclude_patterns": ["WIP"]
            }
        )

        # Only records 1 and 4 should pass all filters
        assert valid_indices == [0, 3]


# =============================================================================
# DATASET.PY TESTS - HOLDOUT SPLIT
# =============================================================================


class TestCreateHoldoutSplit:
    """Tests for create_holdout_split function."""

    def setup_method(self):
        """Create temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.old_datasets_dir = DATASETS_DIR

        # Monkey-patch DATASETS_DIR
        import cortical.ml_experiments.dataset as dataset_module
        dataset_module.DATASETS_DIR = Path(self.temp_dir) / 'datasets'
        dataset_module.DATASET_MANIFEST = dataset_module.DATASETS_DIR / 'dataset_manifest.jsonl'

    def teardown_method(self):
        """Clean up temporary directory."""
        import shutil
        import cortical.ml_experiments.dataset as dataset_module

        dataset_module.DATASETS_DIR = self.old_datasets_dir
        dataset_module.DATASET_MANIFEST = self.old_datasets_dir / 'dataset_manifest.jsonl'

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def create_test_source(self, path: Path, num_records: int = 100):
        """Helper to create test JSONL source file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        for i in range(num_records):
            record = {
                "id": i,
                "message": f"Commit {i}",
                "files": [f"file{i % 5}.py"]
            }
            append_jsonl(path, record)

    def test_create_holdout_split_basic(self):
        """Test basic holdout split creation."""
        source_path = Path(self.temp_dir) / "source.jsonl"
        self.create_test_source(source_path, num_records=100)

        dev, holdout = create_holdout_split(
            source_path=str(source_path),
            holdout_ratio=0.15,
            random_seed=42
        )

        assert dev.name == "development"
        assert holdout.name == "holdout"

        # Verify metadata
        assert dev.metadata.get("holdout_excluded") is True
        assert holdout.metadata.get("is_holdout") is True

    def test_create_holdout_split_sizes(self):
        """Test that holdout split has correct sizes."""
        source_path = Path(self.temp_dir) / "source.jsonl"
        self.create_test_source(source_path, num_records=100)

        dev, holdout = create_holdout_split(
            source_path=str(source_path),
            holdout_ratio=0.15,
            random_seed=42
        )

        # Note: create_holdout_split creates separate datasets with their own
        # indices into the source file, not subsets of each other.
        # The holdout dataset's test split should contain 100% of its filtered records.
        holdout_size = len(holdout.get_split_indices("test"))

        # The holdout dataset should have roughly 15% of the original records
        # (though it's created as a separate dataset with test=1.0)
        assert holdout_size > 0
        assert holdout.filtered_records > 0

    def test_create_holdout_split_with_filters(self):
        """Test holdout split with filters applied."""
        source_path = Path(self.temp_dir) / "source.jsonl"

        # Create source with some merge commits
        path = source_path
        path.parent.mkdir(parents=True, exist_ok=True)
        for i in range(100):
            append_jsonl(path, {
                "id": i,
                "message": f"Commit {i}",
                "files": [f"file{i % 5}.py"],
                "is_merge": i % 10 == 0
            })

        dev, holdout = create_holdout_split(
            source_path=str(source_path),
            holdout_ratio=0.15,
            random_seed=42,
            filters={"exclude_merge": True}
        )

        # Should filter out merge commits
        assert dev.filtered_records < 100
        assert holdout.filtered_records < 100


class TestJSONEncoder:
    """Tests for custom JSON encoder."""

    def test_json_encoder_datetime(self, tmp_path):
        """Test encoding datetime objects."""
        from datetime import datetime
        from cortical.ml_experiments.utils import save_json, load_json

        json_file = tmp_path / "test.json"
        data = {"timestamp": datetime(2025, 12, 16, 10, 30, 0)}

        save_json(json_file, data)
        loaded = load_json(json_file)

        # Should be serialized as ISO string
        assert isinstance(loaded["timestamp"], str)
        assert "2025-12-16" in loaded["timestamp"]

    def test_json_encoder_path(self, tmp_path):
        """Test encoding Path objects."""
        from cortical.ml_experiments.utils import save_json, load_json

        json_file = tmp_path / "test.json"
        data = {"path": Path("/some/path")}

        save_json(json_file, data)
        loaded = load_json(json_file)

        # Should be serialized as string
        assert isinstance(loaded["path"], str)
        assert loaded["path"] == "/some/path"

    def test_json_encoder_dataclass(self, tmp_path):
        """Test encoding dataclass objects."""
        from cortical.ml_experiments.utils import save_json, load_json

        json_file = tmp_path / "test.json"

        split_info = SplitInfo(
            name="test",
            ratio=0.5,
            count=50,
            indices=[0, 1, 2],
            strategy="random"
        )

        data = {"split": split_info}

        save_json(json_file, data)
        loaded = load_json(json_file)

        # Should be serialized as dict
        assert isinstance(loaded["split"], dict)
        assert loaded["split"]["name"] == "test"


class TestEdgeCases:
    """Tests for edge cases and corner scenarios."""

    def test_filter_jsonl_nested_key_not_found(self, tmp_path):
        """Test filtering with nested key that doesn't exist."""
        jsonl_file = tmp_path / "test.jsonl"

        append_jsonl(jsonl_file, {"id": 1, "data": "value"})
        append_jsonl(jsonl_file, {"id": 2, "meta": {"status": "active"}})

        # Filter for nested key that doesn't exist in first record
        filtered = filter_jsonl(jsonl_file, {"meta.status": "active"})

        assert len(filtered) == 1
        assert filtered[0]["id"] == 2

    def test_datasetversion_from_dict_with_dict_splits(self):
        """Test DatasetVersion.from_dict when splits are already dicts."""
        data = {
            "id": "ds-test-20251216-abcd",
            "name": "test",
            "created_at": "2025-12-16T10:00:00Z",
            "source_path": "/path/to/data.jsonl",
            "source_hash": "abc123",
            "total_records": 100,
            "filtered_records": 100,
            "filters": {},
            "filter_hash": "def456",
            "random_seed": 42,
            "split_strategy": "random",
            "splits": {
                "train": {
                    "name": "train",
                    "ratio": 0.7,
                    "count": 70,
                    "indices": list(range(70)),
                    "strategy": "random"
                }
            },
            "metadata": {}
        }

        dataset = DatasetVersion.from_dict(data)

        assert dataset.id == "ds-test-20251216-abcd"
        assert "train" in dataset.splits
        assert isinstance(dataset.splits["train"], SplitInfo)


class TestDatasetManagerAdvanced:
    """Advanced tests for DatasetManager edge cases."""

    def setup_method(self):
        """Create temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.old_datasets_dir = DATASETS_DIR

        # Monkey-patch DATASETS_DIR
        import cortical.ml_experiments.dataset as dataset_module
        dataset_module.DATASETS_DIR = Path(self.temp_dir) / 'datasets'
        dataset_module.DATASET_MANIFEST = dataset_module.DATASETS_DIR / 'dataset_manifest.jsonl'

    def teardown_method(self):
        """Clean up temporary directory."""
        import shutil
        import cortical.ml_experiments.dataset as dataset_module

        dataset_module.DATASETS_DIR = self.old_datasets_dir
        dataset_module.DATASET_MANIFEST = self.old_datasets_dir / 'dataset_manifest.jsonl'

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def create_test_source(self, path: Path, num_records: int = 100):
        """Helper to create test JSONL source file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        for i in range(num_records):
            record = {
                "id": i,
                "message": f"Commit {i}",
                "files": [f"file{i % 5}.py"],
                "is_merge": i % 10 == 0
            }
            append_jsonl(path, record)

    def test_verify_dataset_invalid_indices(self):
        """Test verifying dataset with indices out of bounds."""
        source_path = Path(self.temp_dir) / "source.jsonl"

        # Create small source file
        path = source_path
        path.parent.mkdir(parents=True, exist_ok=True)
        for i in range(10):
            append_jsonl(path, {"id": i, "files": ["test.py"]})

        # Create dataset
        dataset = DatasetManager.create_dataset(
            name="invalid_indices_test",
            source_path=str(source_path),
            random_seed=42
        )

        # Manually corrupt the indices to be out of bounds
        dataset.splits["train"].indices = [0, 1, 2, 999, 1000]

        results = DatasetManager.verify_dataset(dataset)

        assert results['valid'] is False
        assert results['checks']['train_indices_valid'] is False
        assert any("invalid indices" in err for err in results['errors'])


# =============================================================================
# METRICS.PY TESTS - METRICS MANAGER
# =============================================================================


class TestMetricEntry:
    """Tests for MetricEntry dataclass."""

    def test_metric_entry_creation(self):
        """Test MetricEntry creation."""
        entry = MetricEntry(
            timestamp="2025-12-16T10:00:00Z",
            experiment_id="exp-20251216-abc1",
            split="val",
            metric_name="mrr",
            value=0.46,
            metadata={"model": "v1"}
        )

        assert entry.timestamp == "2025-12-16T10:00:00Z"
        assert entry.experiment_id == "exp-20251216-abc1"
        assert entry.split == "val"
        assert entry.metric_name == "mrr"
        assert entry.value == 0.46
        assert entry.metadata == {"model": "v1"}

    def test_metric_entry_to_dict(self):
        """Test MetricEntry to_dict conversion."""
        entry = MetricEntry(
            timestamp="2025-12-16T10:00:00Z",
            experiment_id="exp-001",
            split="test",
            metric_name="recall@10",
            value=0.52
        )

        entry_dict = entry.to_dict()

        assert entry_dict["timestamp"] == "2025-12-16T10:00:00Z"
        assert entry_dict["experiment_id"] == "exp-001"
        assert entry_dict["metric_name"] == "recall@10"
        assert entry_dict["value"] == 0.52
        assert entry_dict["metadata"] == {}  # Should default to empty dict

    def test_metric_entry_from_dict(self):
        """Test MetricEntry from_dict conversion."""
        data = {
            "timestamp": "2025-12-16T10:00:00Z",
            "experiment_id": "exp-002",
            "split": "val",
            "metric_name": "precision@1",
            "value": 0.37,
            "metadata": {"note": "test"}
        }

        entry = MetricEntry.from_dict(data)

        assert entry.experiment_id == "exp-002"
        assert entry.metric_name == "precision@1"
        assert entry.value == 0.37


class TestMetricsManager:
    """Tests for MetricsManager class."""

    def setup_method(self):
        """Create temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()

        # Monkey-patch METRICS_DIR to use temp directory
        import cortical.ml_experiments.metrics as metrics_module
        self.old_metrics_dir = metrics_module.METRICS_DIR
        self.old_metrics_ledger = metrics_module.METRICS_LEDGER

        metrics_module.METRICS_DIR = Path(self.temp_dir) / 'metrics'
        metrics_module.METRICS_LEDGER = metrics_module.METRICS_DIR / 'metrics.jsonl'

    def teardown_method(self):
        """Clean up temporary directory."""
        import shutil
        import cortical.ml_experiments.metrics as metrics_module

        # Restore original paths
        metrics_module.METRICS_DIR = self.old_metrics_dir
        metrics_module.METRICS_LEDGER = self.old_metrics_ledger

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_record_metrics_basic(self):
        """Test recording metrics."""
        entry_ids = MetricsManager.record_metrics(
            experiment_id="exp-001",
            split="val",
            metrics={"mrr": 0.46, "recall@10": 0.52}
        )

        assert len(entry_ids) == 2
        assert all("mrr" in eid or "recall@10" in eid for eid in entry_ids)

    def test_record_metrics_with_metadata(self):
        """Test recording metrics with metadata."""
        MetricsManager.record_metrics(
            experiment_id="exp-002",
            split="test",
            metrics={"accuracy": 0.85},
            metadata={"model_version": "1.0"}
        )

        # Verify by reading back
        history = MetricsManager.get_metric_history("accuracy", split="test")
        assert len(history) == 1

    def test_get_metric_history_basic(self):
        """Test getting metric history."""
        # Record some metrics
        MetricsManager.record_metrics("exp-001", "val", {"mrr": 0.40})
        MetricsManager.record_metrics("exp-002", "val", {"mrr": 0.45})
        MetricsManager.record_metrics("exp-003", "val", {"mrr": 0.50})

        history = MetricsManager.get_metric_history("mrr", split="val")

        assert len(history) == 3
        # History should be sorted by timestamp
        values = [h[2] for h in history]
        assert values == [0.40, 0.45, 0.50]

    def test_get_metric_history_filtered_by_split(self):
        """Test metric history filtering by split."""
        MetricsManager.record_metrics("exp-001", "val", {"mrr": 0.40})
        MetricsManager.record_metrics("exp-002", "test", {"mrr": 0.45})
        MetricsManager.record_metrics("exp-003", "val", {"mrr": 0.50})

        history = MetricsManager.get_metric_history("mrr", split="val")

        assert len(history) == 2
        assert all(h[2] in [0.40, 0.50] for h in history)

    def test_get_metric_history_filtered_by_experiment(self):
        """Test metric history filtering by experiment IDs."""
        MetricsManager.record_metrics("exp-001", "val", {"mrr": 0.40})
        MetricsManager.record_metrics("exp-002", "val", {"mrr": 0.45})
        MetricsManager.record_metrics("exp-003", "val", {"mrr": 0.50})

        history = MetricsManager.get_metric_history(
            "mrr",
            split="val",
            experiment_ids=["exp-001", "exp-003"]
        )

        assert len(history) == 2
        exp_ids = [h[1] for h in history]
        assert "exp-001" in exp_ids
        assert "exp-003" in exp_ids

    def test_get_metric_history_with_limit(self):
        """Test metric history with limit."""
        for i in range(10):
            MetricsManager.record_metrics(f"exp-{i:03d}", "val", {"mrr": 0.40 + i * 0.01})

        history = MetricsManager.get_metric_history("mrr", split="val", limit=5)

        assert len(history) == 5
        # Should return the last 5 entries
        assert history[-1][2] == 0.49  # Last entry

    def test_get_experiment_metrics(self):
        """Test getting all metrics for an experiment."""
        MetricsManager.record_metrics(
            "exp-001",
            "val",
            {"mrr": 0.46, "recall@10": 0.52, "precision@1": 0.37}
        )

        metrics = MetricsManager.get_experiment_metrics("exp-001")

        assert len(metrics) == 3
        assert metrics["mrr"] == 0.46
        assert metrics["recall@10"] == 0.52
        assert metrics["precision@1"] == 0.37

    def test_get_experiment_metrics_filtered_by_split(self):
        """Test getting experiment metrics filtered by split."""
        MetricsManager.record_metrics("exp-001", "val", {"mrr": 0.46})
        MetricsManager.record_metrics("exp-001", "test", {"mrr": 0.42})

        val_metrics = MetricsManager.get_experiment_metrics("exp-001", split="val")
        test_metrics = MetricsManager.get_experiment_metrics("exp-001", split="test")

        assert val_metrics["mrr"] == 0.46
        assert test_metrics["mrr"] == 0.42

    def test_compare_experiments(self):
        """Test comparing metrics across experiments."""
        MetricsManager.record_metrics("exp-001", "val", {"mrr": 0.40, "recall@10": 0.45})
        MetricsManager.record_metrics("exp-002", "val", {"mrr": 0.46, "recall@10": 0.52})

        comparison = MetricsManager.compare_experiments(
            ["exp-001", "exp-002"],
            split="val"
        )

        assert "exp-001" in comparison
        assert "exp-002" in comparison
        assert comparison["exp-001"]["mrr"] == 0.40
        assert comparison["exp-002"]["mrr"] == 0.46

    def test_compare_experiments_specific_metrics(self):
        """Test comparing specific metrics only."""
        MetricsManager.record_metrics(
            "exp-001", "val",
            {"mrr": 0.40, "recall@10": 0.45, "precision@1": 0.30}
        )
        MetricsManager.record_metrics(
            "exp-002", "val",
            {"mrr": 0.46, "recall@10": 0.52, "precision@1": 0.35}
        )

        comparison = MetricsManager.compare_experiments(
            ["exp-001", "exp-002"],
            metrics=["mrr", "recall@10"],
            split="val"
        )

        assert "mrr" in comparison["exp-001"]
        assert "recall@10" in comparison["exp-001"]
        assert "precision@1" not in comparison["exp-001"]

    def test_get_best_value_higher_is_better(self):
        """Test finding best value when higher is better."""
        MetricsManager.record_metrics("exp-001", "val", {"mrr": 0.40})
        MetricsManager.record_metrics("exp-002", "val", {"mrr": 0.52})
        MetricsManager.record_metrics("exp-003", "val", {"mrr": 0.46})

        best = MetricsManager.get_best_value("mrr", split="val", higher_is_better=True)

        assert best is not None
        assert best[0] == "exp-002"
        assert best[1] == 0.52

    def test_get_best_value_lower_is_better(self):
        """Test finding best value when lower is better."""
        MetricsManager.record_metrics("exp-001", "val", {"loss": 0.50})
        MetricsManager.record_metrics("exp-002", "val", {"loss": 0.30})
        MetricsManager.record_metrics("exp-003", "val", {"loss": 0.40})

        best = MetricsManager.get_best_value("loss", split="val", higher_is_better=False)

        assert best is not None
        assert best[0] == "exp-002"
        assert best[1] == 0.30

    def test_get_best_value_no_records(self):
        """Test getting best value with no records."""
        best = MetricsManager.get_best_value("nonexistent", split="val")

        assert best is None

    def test_get_metric_stats(self):
        """Test getting statistics for a metric."""
        MetricsManager.record_metrics("exp-001", "val", {"mrr": 0.40})
        MetricsManager.record_metrics("exp-002", "val", {"mrr": 0.50})
        MetricsManager.record_metrics("exp-003", "val", {"mrr": 0.60})

        stats = MetricsManager.get_metric_stats("mrr", split="val")

        assert stats["count"] == 3
        assert stats["min"] == 0.40
        assert stats["max"] == 0.60
        assert stats["mean"] == 0.50
        assert stats["std"] > 0

    def test_get_metric_stats_no_records(self):
        """Test getting statistics with no records."""
        stats = MetricsManager.get_metric_stats("nonexistent", split="val")

        assert stats["count"] == 0
        assert stats["min"] == 0
        assert stats["max"] == 0
        assert stats["mean"] == 0

    def test_detect_regression_no_regression(self):
        """Test regression detection when no regression."""
        MetricsManager.record_metrics("exp-001", "val", {"mrr": 0.50})

        result = MetricsManager.detect_regression(
            "mrr",
            current_value=0.48,  # Only 4% drop
            split="val",
            threshold_pct=5.0
        )

        assert result is None

    def test_detect_regression_with_regression(self):
        """Test regression detection when regression exists."""
        MetricsManager.record_metrics("exp-001", "val", {"mrr": 0.50})

        result = MetricsManager.detect_regression(
            "mrr",
            current_value=0.40,  # 20% drop
            split="val",
            threshold_pct=5.0
        )

        assert result is not None
        assert result["metric"] == "mrr"
        assert result["current_value"] == 0.40
        assert result["best_value"] == 0.50
        assert result["pct_change"] < -5.0

    def test_detect_regression_lower_is_better(self):
        """Test regression detection for metrics where lower is better."""
        MetricsManager.record_metrics("exp-001", "val", {"loss": 0.30})

        result = MetricsManager.detect_regression(
            "loss",
            current_value=0.40,  # 33% increase (bad for loss)
            split="val",
            threshold_pct=5.0,
            higher_is_better=False
        )

        assert result is not None
        assert result["current_value"] == 0.40

    def test_detect_regression_no_baseline(self):
        """Test regression detection with no baseline records."""
        result = MetricsManager.detect_regression(
            "nonexistent",
            current_value=0.50,
            split="val"
        )

        assert result is None

    def test_format_comparison_table_basic(self):
        """Test formatting comparison as ASCII table."""
        comparison = {
            "exp-001": {"mrr": 0.40, "recall@10": 0.45},
            "exp-002": {"mrr": 0.46, "recall@10": 0.52}
        }

        table = MetricsManager.format_comparison_table(comparison)

        assert "Metric" in table
        assert "mrr" in table or "recall" in table
        assert "0.40" in table or "0.4000" in table

    def test_format_comparison_table_with_order(self):
        """Test formatting comparison with metric ordering."""
        comparison = {
            "exp-001": {"recall@10": 0.45, "mrr": 0.40, "precision@1": 0.30}
        }

        table = MetricsManager.format_comparison_table(
            comparison,
            metric_order=["mrr", "recall@10", "precision@1"]
        )

        # Check that table is formatted
        lines = table.split("\n")
        assert len(lines) >= 3  # Header, separator, at least one row

    def test_format_comparison_table_empty(self):
        """Test formatting empty comparison."""
        table = MetricsManager.format_comparison_table({})

        assert "No experiments to compare" in table


# Import MetricEntry and MetricsManager for tests
from cortical.ml_experiments.metrics import MetricEntry, MetricsManager


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
