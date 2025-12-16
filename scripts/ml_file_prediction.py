#!/usr/bin/env python3
"""
ML File Prediction Module

Predicts which files are likely to be modified for a given task description.
Uses patterns learned from commit history:
- Commit type prefix patterns (feat:, fix:, docs:, etc.)
- File co-occurrence patterns
- Module keyword associations
- Task reference patterns

Usage:
    python scripts/ml_file_prediction.py train
    python scripts/ml_file_prediction.py predict "Add authentication feature"
    python scripts/ml_file_prediction.py evaluate --split 0.2
"""

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

from ml_collector.config import TRACKED_DIR, ML_DATA_DIR


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_DIR = ML_DATA_DIR / "models"
FILE_PREDICTION_MODEL = MODEL_DIR / "file_prediction.json"

# Commit type patterns
COMMIT_TYPE_PATTERNS = {
    'feat': r'^feat(?:\(.+?\))?:\s*',
    'fix': r'^fix(?:\(.+?\))?:\s*',
    'docs': r'^docs(?:\(.+?\))?:\s*',
    'refactor': r'^refactor(?:\(.+?\))?:\s*',
    'test': r'^test(?:\(.+?\))?:\s*',
    'chore': r'^chore(?:\(.+?\))?:\s*',
    'style': r'^style(?:\(.+?\))?:\s*',
    'perf': r'^perf(?:\(.+?\))?:\s*',
    'ci': r'^ci(?:\(.+?\))?:\s*',
    'build': r'^build(?:\(.+?\))?:\s*',
    'security': r'^security(?:\(.+?\))?:\s*',
    'merge': r'^Merge\s+',
    'task': r'[Tt]ask\s*#?\d+',
    'add': r'^[Aa]dd\s+',
    'update': r'^[Uu]pdate\s+',
    'implement': r'^[Ii]mplement\s+',
    'complete': r'^[Cc]omplete\s+',
}

# Module keyword to directory mappings
MODULE_KEYWORDS = {
    'test': ['tests/', 'test_'],
    'documentation': ['docs/', 'README', 'CLAUDE.md', '.md'],
    'config': ['config.py', 'pyproject.toml', 'setup.py', '.json'],
    'api': ['processor/', 'query/', '__init__.py'],
    'analysis': ['analysis.py', 'semantics.py', 'embeddings.py'],
    'persistence': ['persistence.py', 'chunk_index.py', 'state_storage.py'],
    'hooks': ['hooks/', '.git/hooks/', 'hook'],
    'ci': ['.github/', 'ci.yml', 'workflows/'],
    'ml': ['ml_', 'ml-', '.git-ml/'],
    'script': ['scripts/'],
    'core': ['cortical/', 'minicolumn.py', 'layers.py'],
    'tokenizer': ['tokenizer.py'],
    'fingerprint': ['fingerprint.py'],
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class TrainingExample:
    """A single training example from commit history."""
    commit_hash: str
    message: str
    files_changed: List[str]
    commit_type: Optional[str]
    keywords: List[str]
    timestamp: str
    insertions: int
    deletions: int


@dataclass
class FilePredictionModel:
    """Model for predicting files from task descriptions."""
    # File co-occurrence matrix: file -> {co_file: count}
    file_cooccurrence: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Commit type -> files mapping
    type_to_files: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Keyword -> files mapping
    keyword_to_files: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # File frequency (how often each file is changed)
    file_frequency: Dict[str, int] = field(default_factory=dict)

    # Total commits seen
    total_commits: int = 0

    # Training timestamp
    trained_at: str = ""

    # Model version
    version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FilePredictionModel':
        return cls(**d)


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_commit_type(message: str) -> Optional[str]:
    """Extract commit type from message using conventional commit patterns."""
    message_lower = message.lower()

    for commit_type, pattern in COMMIT_TYPE_PATTERNS.items():
        if re.search(pattern, message, re.IGNORECASE):
            return commit_type

    return None


def extract_keywords(message: str) -> List[str]:
    """Extract relevant keywords from commit message."""
    keywords = []
    message_lower = message.lower()

    # Check module keywords
    for keyword, _ in MODULE_KEYWORDS.items():
        if keyword in message_lower:
            keywords.append(keyword)

    # Extract task references
    task_match = re.search(r'[Tt]ask\s*#?(\d+)', message)
    if task_match:
        keywords.append(f'task_{task_match.group(1)}')

    # Extract action verbs
    action_verbs = ['add', 'fix', 'update', 'implement', 'refactor',
                    'remove', 'improve', 'optimize', 'complete']
    for verb in action_verbs:
        if verb in message_lower.split():
            keywords.append(f'action_{verb}')

    return keywords


def extract_file_keywords(files: List[str]) -> Set[str]:
    """Extract keywords associated with file paths."""
    keywords = set()

    for filepath in files:
        filepath_lower = filepath.lower()
        for keyword, patterns in MODULE_KEYWORDS.items():
            for pattern in patterns:
                if pattern.lower() in filepath_lower:
                    keywords.add(keyword)
                    break

    return keywords


def message_to_keywords(message: str) -> List[str]:
    """Convert a message/query into searchable keywords."""
    # Normalize
    words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', message.lower())

    # Filter stop words
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                  'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                  'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                  'can', 'need', 'to', 'of', 'in', 'for', 'on', 'with', 'at',
                  'by', 'from', 'as', 'into', 'through', 'during', 'before',
                  'after', 'above', 'below', 'between', 'under', 'again',
                  'further', 'then', 'once', 'here', 'there', 'when', 'where',
                  'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other',
                  'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
                  'so', 'than', 'too', 'very', 'just', 'and', 'but', 'if',
                  'or', 'because', 'until', 'while', 'this', 'that', 'these',
                  'those', 'what', 'which', 'who', 'whom', 'it', 'its', 'i',
                  'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
                  'you', 'your', 'yours', 'yourself', 'yourselves', 'he',
                  'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
                  'they', 'them', 'their', 'theirs', 'themselves'}

    keywords = [w for w in words if w not in stop_words and len(w) > 2]

    return keywords


# ============================================================================
# DATA LOADING
# ============================================================================

def load_commit_data() -> List[TrainingExample]:
    """Load commit data from JSONL file."""
    commits_file = TRACKED_DIR / "commits.jsonl"

    if not commits_file.exists():
        print(f"No commits file found at {commits_file}")
        return []

    examples = []

    with open(commits_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                commit = json.loads(line)

                # Skip merge commits and ML data commits
                if commit.get('is_merge', False):
                    continue
                if commit.get('message', '').startswith('data: ML'):
                    continue

                example = TrainingExample(
                    commit_hash=commit.get('hash', ''),
                    message=commit.get('message', ''),
                    files_changed=commit.get('files_changed', []),
                    commit_type=extract_commit_type(commit.get('message', '')),
                    keywords=extract_keywords(commit.get('message', '')),
                    timestamp=commit.get('timestamp', ''),
                    insertions=commit.get('insertions', 0),
                    deletions=commit.get('deletions', 0)
                )

                # Only include commits that changed files
                if example.files_changed:
                    examples.append(example)

            except json.JSONDecodeError:
                continue

    return examples


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_model(examples: List[TrainingExample]) -> FilePredictionModel:
    """Train file prediction model from commit examples."""
    model = FilePredictionModel(
        file_cooccurrence={},
        type_to_files={},
        keyword_to_files={},
        file_frequency={},
        total_commits=0,
        trained_at=datetime.now().isoformat(),
        version="1.0.0"
    )

    for example in examples:
        files = example.files_changed
        model.total_commits += 1

        # Update file frequency
        for f in files:
            model.file_frequency[f] = model.file_frequency.get(f, 0) + 1

        # Update file co-occurrence
        for i, f1 in enumerate(files):
            if f1 not in model.file_cooccurrence:
                model.file_cooccurrence[f1] = {}
            for f2 in files[i+1:]:
                # Bidirectional co-occurrence
                model.file_cooccurrence[f1][f2] = model.file_cooccurrence[f1].get(f2, 0) + 1
                if f2 not in model.file_cooccurrence:
                    model.file_cooccurrence[f2] = {}
                model.file_cooccurrence[f2][f1] = model.file_cooccurrence[f2].get(f1, 0) + 1

        # Update commit type -> files mapping
        if example.commit_type:
            if example.commit_type not in model.type_to_files:
                model.type_to_files[example.commit_type] = {}
            for f in files:
                model.type_to_files[example.commit_type][f] = \
                    model.type_to_files[example.commit_type].get(f, 0) + 1

        # Update keyword -> files mapping
        all_keywords = set(example.keywords)
        all_keywords.update(extract_file_keywords(files))
        all_keywords.update(message_to_keywords(example.message))

        for keyword in all_keywords:
            if keyword not in model.keyword_to_files:
                model.keyword_to_files[keyword] = {}
            for f in files:
                model.keyword_to_files[keyword][f] = \
                    model.keyword_to_files[keyword].get(f, 0) + 1

    return model


def save_model(model: FilePredictionModel, path: Path = None) -> str:
    """Save model to JSON file."""
    if path is None:
        path = FILE_PREDICTION_MODEL

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(model.to_dict(), f, indent=2)

    return str(path)


def load_model(path: Path = None) -> Optional[FilePredictionModel]:
    """Load model from JSON file."""
    if path is None:
        path = FILE_PREDICTION_MODEL

    if not path.exists():
        return None

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return FilePredictionModel.from_dict(data)


# ============================================================================
# PREDICTION
# ============================================================================

def predict_files(
    query: str,
    model: FilePredictionModel,
    top_n: int = 10,
    seed_files: List[str] = None
) -> List[Tuple[str, float]]:
    """
    Predict which files are likely to be modified for a given task.

    Args:
        query: Task description or commit message
        model: Trained file prediction model
        top_n: Number of files to return
        seed_files: Optional known files to use for co-occurrence boosting

    Returns:
        List of (file_path, score) tuples sorted by relevance
    """
    file_scores: Dict[str, float] = defaultdict(float)

    # Extract features from query
    commit_type = extract_commit_type(query)
    keywords = set(extract_keywords(query))
    keywords.update(message_to_keywords(query))

    # Score based on commit type
    if commit_type and commit_type in model.type_to_files:
        type_files = model.type_to_files[commit_type]
        type_total = sum(type_files.values())
        for f, count in type_files.items():
            # TF-IDF-like scoring
            tf = count / type_total
            idf = math.log(model.total_commits / (model.file_frequency.get(f, 1) + 1))
            file_scores[f] += tf * idf * 2.0  # Weight for type match

    # Score based on keywords
    for keyword in keywords:
        if keyword in model.keyword_to_files:
            kw_files = model.keyword_to_files[keyword]
            kw_total = sum(kw_files.values())
            for f, count in kw_files.items():
                tf = count / kw_total
                idf = math.log(model.total_commits / (model.file_frequency.get(f, 1) + 1))
                file_scores[f] += tf * idf * 1.5  # Weight for keyword match

    # Boost based on co-occurrence with seed files
    if seed_files:
        for seed in seed_files:
            if seed in model.file_cooccurrence:
                cooc = model.file_cooccurrence[seed]
                cooc_total = sum(cooc.values())
                for f, count in cooc.items():
                    # Jaccard-like similarity
                    union = model.file_frequency.get(seed, 1) + model.file_frequency.get(f, 1) - count
                    similarity = count / union if union > 0 else 0
                    file_scores[f] += similarity * 3.0  # Strong weight for co-occurrence

    # Apply file frequency penalty (avoid always recommending high-frequency files)
    max_freq = max(model.file_frequency.values()) if model.file_frequency else 1
    for f in file_scores:
        freq_penalty = 1.0 - (model.file_frequency.get(f, 0) / max_freq) * 0.3
        file_scores[f] *= freq_penalty

    # Sort and return top N (filtering out non-existent files)
    sorted_files = sorted(file_scores.items(), key=lambda x: -x[1])
    # Filter to only existing files - removes deleted/renamed files from predictions
    sorted_files = [(f, score) for f, score in sorted_files if Path(f).exists()]
    return sorted_files[:top_n]


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(
    model: FilePredictionModel,
    test_examples: List[TrainingExample],
    top_k: List[int] = [1, 3, 5, 10]
) -> Dict[str, Any]:
    """
    Evaluate model performance on test set.

    Metrics:
    - Recall@K: What fraction of actual files appear in top K predictions?
    - Precision@K: What fraction of top K predictions are correct?
    - MRR: Mean Reciprocal Rank
    """
    metrics = {
        f'recall@{k}': [] for k in top_k
    }
    metrics.update({
        f'precision@{k}': [] for k in top_k
    })
    metrics['mrr'] = []

    for example in test_examples:
        actual_files = set(example.files_changed)
        predictions = predict_files(example.message, model, top_n=max(top_k))
        predicted_files = [f for f, _ in predictions]

        # Calculate metrics for each K
        for k in top_k:
            top_k_files = set(predicted_files[:k])

            # Recall@K
            recall = len(actual_files & top_k_files) / len(actual_files) if actual_files else 0
            metrics[f'recall@{k}'].append(recall)

            # Precision@K
            precision = len(actual_files & top_k_files) / k if k > 0 else 0
            metrics[f'precision@{k}'].append(precision)

        # MRR - find rank of first correct prediction
        mrr = 0.0
        for i, f in enumerate(predicted_files):
            if f in actual_files:
                mrr = 1.0 / (i + 1)
                break
        metrics['mrr'].append(mrr)

    # Average metrics
    results = {}
    for metric_name, values in metrics.items():
        results[metric_name] = sum(values) / len(values) if values else 0.0

    results['total_examples'] = len(test_examples)

    return results


def train_test_split(
    examples: List[TrainingExample],
    test_ratio: float = 0.2,
    shuffle: bool = True
) -> Tuple[List[TrainingExample], List[TrainingExample]]:
    """Split examples into train and test sets."""
    if shuffle:
        import random
        examples = examples.copy()
        random.shuffle(examples)

    split_idx = int(len(examples) * (1 - test_ratio))
    return examples[:split_idx], examples[split_idx:]


# ============================================================================
# CLI
# ============================================================================

def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='ML File Prediction - Predict files to modify for a task'
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--output', '-o', type=str,
                             help='Output model path')

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict files for a task')
    predict_parser.add_argument('query', type=str, help='Task description')
    predict_parser.add_argument('--top', '-n', type=int, default=10,
                               help='Number of predictions')
    predict_parser.add_argument('--seed', '-s', type=str, nargs='*',
                               help='Seed files for co-occurrence boosting')

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    eval_parser.add_argument('--split', type=float, default=0.2,
                            help='Test split ratio')

    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show model statistics')

    args = parser.parse_args()

    if args.command == 'train':
        print("Loading commit data...")
        examples = load_commit_data()
        print(f"Loaded {len(examples)} training examples")

        print("Training model...")
        model = train_model(examples)

        output_path = Path(args.output) if args.output else None
        path = save_model(model, output_path)

        print(f"\nModel trained and saved to {path}")
        print(f"  Total commits: {model.total_commits}")
        print(f"  Unique files: {len(model.file_frequency)}")
        print(f"  Commit types: {len(model.type_to_files)}")
        print(f"  Keywords: {len(model.keyword_to_files)}")

    elif args.command == 'predict':
        model = load_model()
        if not model:
            print("No trained model found. Run 'train' first.")
            return 1

        predictions = predict_files(
            args.query,
            model,
            top_n=args.top,
            seed_files=args.seed
        )

        print(f"\nPredicted files for: '{args.query}'")
        print("-" * 60)
        for i, (filepath, score) in enumerate(predictions, 1):
            print(f"  {i:2}. {filepath:<45} ({score:.3f})")

    elif args.command == 'evaluate':
        print("Loading commit data...")
        examples = load_commit_data()

        print(f"Splitting {len(examples)} examples ({1-args.split:.0%} train, {args.split:.0%} test)...")
        train_examples, test_examples = train_test_split(examples, args.split)

        print(f"Training on {len(train_examples)} examples...")
        model = train_model(train_examples)

        print(f"Evaluating on {len(test_examples)} examples...")
        results = evaluate_model(model, test_examples)

        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        for metric, value in sorted(results.items()):
            if metric != 'total_examples':
                print(f"  {metric:<15}: {value:.4f}")
        print(f"\n  Test examples: {results['total_examples']}")

    elif args.command == 'stats':
        model = load_model()
        if not model:
            print("No trained model found. Run 'train' first.")
            return 1

        print("\n" + "=" * 60)
        print("FILE PREDICTION MODEL STATISTICS")
        print("=" * 60)
        print(f"  Version:       {model.version}")
        print(f"  Trained at:    {model.trained_at}")
        print(f"  Total commits: {model.total_commits}")
        print(f"  Unique files:  {len(model.file_frequency)}")
        print(f"  Commit types:  {len(model.type_to_files)}")
        print(f"  Keywords:      {len(model.keyword_to_files)}")

        if model.type_to_files:
            print("\n  Commit types distribution:")
            for ct, files in sorted(model.type_to_files.items(),
                                   key=lambda x: -sum(x[1].values()))[:10]:
                print(f"    {ct}: {sum(files.values())} commits")

        if model.file_frequency:
            print("\n  Most frequently changed files:")
            for f, count in sorted(model.file_frequency.items(),
                                   key=lambda x: -x[1])[:10]:
                print(f"    {f}: {count} commits")

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
