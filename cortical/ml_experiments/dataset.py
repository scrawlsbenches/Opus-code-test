"""
Dataset versioning and reproducible train/val/test splits.

Provides:
- Versioned datasets with content hashing
- Reproducible splits (random or temporal)
- Holdout test set management
- Filter configuration tracking
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .utils import (
    compute_file_hash,
    compute_dict_hash,
    generate_dataset_id,
    now_iso,
    split_indices,
    append_jsonl,
    read_jsonl,
    save_json,
    load_json,
    ensure_directory,
)


# Default paths
DEFAULT_ML_DIR = Path('.git-ml')
DATASETS_DIR = DEFAULT_ML_DIR / 'datasets'
DATASET_MANIFEST = DATASETS_DIR / 'dataset_manifest.jsonl'


@dataclass
class SplitInfo:
    """Information about a single data split (train/val/test)."""
    name: str                          # 'train', 'val', 'test'
    ratio: float                       # 0.7, 0.15, 0.15
    count: int                         # Number of examples
    indices: List[int]                 # Indices into source data
    strategy: str                      # 'random' or 'temporal'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SplitInfo':
        return cls(**data)


@dataclass
class DatasetVersion:
    """
    A versioned dataset with reproducible splits.

    Tracks:
    - Source data and its hash (for change detection)
    - Filters applied during creation
    - Train/val/test split indices (for exact reproducibility)
    - Metadata for experiment tracking
    """
    id: str                                # Unique identifier
    name: str                              # Human-readable name
    created_at: str                        # ISO timestamp
    source_path: str                       # Path to source data
    source_hash: str                       # SHA256 of source file
    total_records: int                     # Total examples before filtering
    filtered_records: int                  # Examples after filtering
    filters: Dict[str, Any]                # Filter configuration
    filter_hash: str                       # Hash of filter config
    random_seed: int                       # For reproducibility
    split_strategy: str                    # 'random' or 'temporal'
    splits: Dict[str, SplitInfo] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['splits'] = {k: v.to_dict() if hasattr(v, 'to_dict') else v
                           for k, v in self.splits.items()}
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetVersion':
        splits_data = data.pop('splits', {})
        splits = {}
        for name, split_info in splits_data.items():
            if isinstance(split_info, dict):
                splits[name] = SplitInfo.from_dict(split_info)
            else:
                splits[name] = split_info
        return cls(splits=splits, **data)

    def get_split_indices(self, split_name: str) -> List[int]:
        """Get indices for a specific split."""
        if split_name not in self.splits:
            raise ValueError(f"Split '{split_name}' not found. Available: {list(self.splits.keys())}")
        return self.splits[split_name].indices


class DatasetManager:
    """
    Manage versioned datasets with reproducible splits.

    Example usage:
        # Create a new dataset version
        dataset = DatasetManager.create_dataset(
            name='commits_v1',
            source_path='.git-ml/tracked/commits.jsonl',
            filters={'exclude_merge': True, 'min_files': 1},
            split_ratios={'train': 0.7, 'val': 0.15, 'test': 0.15},
            split_strategy='random',
            random_seed=42
        )

        # Get indices for training
        train_indices = dataset.get_split_indices('train')

        # Load existing dataset
        dataset = DatasetManager.load_dataset('ds-commits_v1-20251216-abc1')
    """

    @staticmethod
    def create_dataset(
        name: str,
        source_path: str,
        filters: Optional[Dict[str, Any]] = None,
        split_ratios: Optional[Dict[str, float]] = None,
        split_strategy: str = 'random',
        random_seed: int = 42,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DatasetVersion:
        """
        Create a new versioned dataset with reproducible splits.

        Args:
            name: Human-readable dataset name
            source_path: Path to source JSONL file
            filters: Optional filter configuration
            split_ratios: Split ratios (default: 70/15/15 train/val/test)
            split_strategy: 'random' or 'temporal'
            random_seed: Seed for reproducible random splits
            metadata: Optional extra metadata

        Returns:
            DatasetVersion with split indices stored
        """
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        # Default split ratios
        if split_ratios is None:
            split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}

        filters = filters or {}

        # Load and count source data
        raw_records = read_jsonl(source)
        total_records = len(raw_records)

        # Apply filters to get valid indices
        valid_indices = DatasetManager._apply_filters(raw_records, filters)
        filtered_records = len(valid_indices)

        if filtered_records == 0:
            raise ValueError("No records remain after filtering")

        # Compute hashes for versioning
        source_hash = compute_file_hash(source)
        filter_hash = compute_dict_hash(filters)

        # Generate reproducible splits
        # We split the valid_indices, not the raw indices
        relative_splits = split_indices(
            total=filtered_records,
            ratios=split_ratios,
            seed=random_seed,
            strategy=split_strategy
        )

        # Map relative indices back to original indices
        splits = {}
        for split_name, relative_idx_list in relative_splits.items():
            original_indices = [valid_indices[i] for i in relative_idx_list]
            splits[split_name] = SplitInfo(
                name=split_name,
                ratio=split_ratios[split_name],
                count=len(original_indices),
                indices=original_indices,
                strategy=split_strategy
            )

        # Create dataset version
        dataset = DatasetVersion(
            id=generate_dataset_id(name),
            name=name,
            created_at=now_iso(),
            source_path=str(source),
            source_hash=source_hash,
            total_records=total_records,
            filtered_records=filtered_records,
            filters=filters,
            filter_hash=filter_hash,
            random_seed=random_seed,
            split_strategy=split_strategy,
            splits=splits,
            metadata=metadata or {}
        )

        # Save to manifest
        DatasetManager._save_to_manifest(dataset)

        # Save split indices to separate file (for quick loading)
        DatasetManager._save_splits(dataset)

        return dataset

    @staticmethod
    def _apply_filters(
        records: List[Dict[str, Any]],
        filters: Dict[str, Any]
    ) -> List[int]:
        """
        Apply filters to records and return valid indices.

        Supported filters:
        - exclude_merge: bool - Exclude merge commits
        - min_files: int - Minimum number of files changed
        - max_files: int - Maximum number of files changed
        - exclude_patterns: List[str] - Exclude if message matches patterns
        - require_fields: List[str] - Require these fields to be present

        Returns:
            List of indices that pass all filters
        """
        valid_indices = []

        for i, record in enumerate(records):
            # Default: include
            include = True

            # Filter: exclude merge commits
            if filters.get('exclude_merge', False):
                if record.get('is_merge', False):
                    include = False

            # Filter: minimum files
            min_files = filters.get('min_files', 0)
            if min_files > 0:
                files = record.get('files', [])
                if len(files) < min_files:
                    include = False

            # Filter: maximum files
            max_files = filters.get('max_files')
            if max_files is not None:
                files = record.get('files', [])
                if len(files) > max_files:
                    include = False

            # Filter: exclude patterns in message
            exclude_patterns = filters.get('exclude_patterns', [])
            message = record.get('message', '').lower()
            for pattern in exclude_patterns:
                if pattern.lower() in message:
                    include = False
                    break

            # Filter: require fields
            require_fields = filters.get('require_fields', [])
            for field_name in require_fields:
                if field_name not in record or record[field_name] is None:
                    include = False
                    break

            # Filter: exclude empty file lists
            if filters.get('exclude_empty_files', True):
                files = record.get('files', [])
                if not files:
                    include = False

            if include:
                valid_indices.append(i)

        return valid_indices

    @staticmethod
    def _save_to_manifest(dataset: DatasetVersion) -> None:
        """Append dataset to manifest (append-only)."""
        ensure_directory(DATASETS_DIR)
        append_jsonl(DATASET_MANIFEST, dataset.to_dict())

    @staticmethod
    def _save_splits(dataset: DatasetVersion) -> None:
        """Save split indices to a separate file for quick loading."""
        splits_file = DATASETS_DIR / f"{dataset.id}_splits.json"
        save_json(splits_file, {
            'dataset_id': dataset.id,
            'source_hash': dataset.source_hash,
            'splits': {name: info.to_dict() for name, info in dataset.splits.items()}
        })

    @staticmethod
    def load_dataset(dataset_id: str) -> Optional[DatasetVersion]:
        """
        Load a dataset version by ID.

        Args:
            dataset_id: Dataset identifier

        Returns:
            DatasetVersion or None if not found
        """
        # Search manifest for dataset
        records = read_jsonl(DATASET_MANIFEST)

        for record in reversed(records):  # Search newest first
            if record.get('id') == dataset_id:
                return DatasetVersion.from_dict(record)

        return None

    @staticmethod
    def load_latest(name: str) -> Optional[DatasetVersion]:
        """
        Load the most recent dataset version with a given name.

        Args:
            name: Dataset name to search for

        Returns:
            Most recent DatasetVersion with that name, or None
        """
        records = read_jsonl(DATASET_MANIFEST)

        # Find most recent by created_at
        matching = [r for r in records if r.get('name') == name]

        if not matching:
            return None

        latest = max(matching, key=lambda r: r.get('created_at', ''))
        return DatasetVersion.from_dict(latest)

    @staticmethod
    def list_datasets(
        name_filter: Optional[str] = None,
        limit: int = 20
    ) -> List[DatasetVersion]:
        """
        List available datasets.

        Args:
            name_filter: Optional name to filter by
            limit: Maximum number to return

        Returns:
            List of DatasetVersions, newest first
        """
        records = read_jsonl(DATASET_MANIFEST)

        if name_filter:
            records = [r for r in records if name_filter in r.get('name', '')]

        # Sort by created_at descending
        records.sort(key=lambda r: r.get('created_at', ''), reverse=True)

        return [DatasetVersion.from_dict(r) for r in records[:limit]]

    @staticmethod
    def get_split_data(
        dataset: DatasetVersion,
        split_name: str,
        source_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get actual data records for a split.

        Args:
            dataset: DatasetVersion to use
            split_name: Which split ('train', 'val', 'test')
            source_path: Override source path (optional)

        Returns:
            List of records for the split
        """
        source = Path(source_path or dataset.source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        all_records = read_jsonl(source)
        indices = dataset.get_split_indices(split_name)

        return [all_records[i] for i in indices if i < len(all_records)]

    @staticmethod
    def verify_dataset(dataset: DatasetVersion) -> Dict[str, Any]:
        """
        Verify a dataset's integrity.

        Checks:
        - Source file exists
        - Source hash matches
        - Split indices are valid

        Returns:
            Dict with verification results
        """
        results = {
            'valid': True,
            'checks': {},
            'errors': []
        }

        source = Path(dataset.source_path)

        # Check source exists
        if not source.exists():
            results['valid'] = False
            results['checks']['source_exists'] = False
            results['errors'].append(f"Source file missing: {source}")
        else:
            results['checks']['source_exists'] = True

            # Check hash
            current_hash = compute_file_hash(source)
            if current_hash != dataset.source_hash:
                results['valid'] = False
                results['checks']['hash_matches'] = False
                results['errors'].append(
                    f"Source hash changed: {dataset.source_hash} -> {current_hash}"
                )
            else:
                results['checks']['hash_matches'] = True

            # Check indices are valid
            records = read_jsonl(source)
            max_index = len(records) - 1

            for split_name, split_info in dataset.splits.items():
                invalid = [i for i in split_info.indices if i > max_index]
                if invalid:
                    results['valid'] = False
                    results['checks'][f'{split_name}_indices_valid'] = False
                    results['errors'].append(
                        f"Split '{split_name}' has {len(invalid)} invalid indices"
                    )
                else:
                    results['checks'][f'{split_name}_indices_valid'] = True

        return results


def create_holdout_split(
    source_path: str,
    holdout_ratio: float = 0.15,
    random_seed: int = 42,
    filters: Optional[Dict[str, Any]] = None
) -> Tuple[DatasetVersion, DatasetVersion]:
    """
    Create a holdout test set that should NEVER be used during development.

    This creates two datasets:
    1. Development dataset (train + val) - use for training and tuning
    2. Holdout dataset (test only) - use ONLY for final evaluation

    Args:
        source_path: Path to source JSONL
        holdout_ratio: Fraction to hold out (default 15%)
        random_seed: Seed for reproducibility
        filters: Optional filters

    Returns:
        Tuple of (dev_dataset, holdout_dataset)
    """
    # Create full dataset first
    full = DatasetManager.create_dataset(
        name='_temp_full',
        source_path=source_path,
        filters=filters,
        split_ratios={'dev': 1.0 - holdout_ratio, 'holdout': holdout_ratio},
        split_strategy='random',
        random_seed=random_seed
    )

    # Create dev dataset (train/val from dev split)
    dev_indices = full.get_split_indices('dev')
    dev = DatasetManager.create_dataset(
        name='development',
        source_path=source_path,
        filters=filters,
        split_ratios={'train': 0.824, 'val': 0.176},  # ~70/15 of original
        split_strategy='random',
        random_seed=random_seed + 1,  # Different seed for inner split
        metadata={'parent_split': 'dev', 'holdout_excluded': True}
    )

    # Create holdout dataset
    holdout = DatasetManager.create_dataset(
        name='holdout',
        source_path=source_path,
        filters=filters,
        split_ratios={'test': 1.0},
        split_strategy='random',
        random_seed=random_seed,
        metadata={'is_holdout': True, 'warning': 'DO NOT USE FOR TRAINING OR TUNING'}
    )

    return dev, holdout
