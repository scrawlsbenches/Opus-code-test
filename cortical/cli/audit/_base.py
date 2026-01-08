"""
Shared utilities for audit CLI commands.

This module provides common functionality used across audit commands,
including model paths, tokenization, and file utilities.
"""

from pathlib import Path
from typing import List, Any, Optional
import pickle

# Model storage paths
MODEL_DIR = Path(".audit_models")
BLOOM_MODEL = MODEL_DIR / "bloom_filter.pkl"
NAIVE_BAYES_MODEL = MODEL_DIR / "naive_bayes.pkl"
LSH_MODEL = MODEL_DIR / "lsh_index.pkl"
INDEX_MODEL = MODEL_DIR / "inverted_index.pkl"
TRIE_MODEL = MODEL_DIR / "marker_trie.pkl"

# Default output directory for training data
DEFAULT_TRAINING_DIR = Path("docs/audits")

# Default confidence threshold for classification
# Used by scan and classify commands to filter low-confidence predictions
DEFAULT_CONFIDENCE_THRESHOLD = 0.65


def save_model(obj: Any, path: Path) -> None:
    """
    Save a model using pickle.

    Args:
        obj: Object to save.
        path: Path to save to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Model saved to {path}")


def load_model(path: Path) -> Optional[Any]:
    """
    Load a model using pickle.

    Args:
        path: Path to load from.

    Returns:
        Loaded object or None if file doesn't exist.
    """
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def tokenize_comment(comment: str, tokenizer) -> List[str]:
    """
    Tokenize a comment using the Cortical tokenizer.

    Args:
        comment: Comment text to tokenize.
        tokenizer: Tokenizer instance.

    Returns:
        List of tokens.
    """
    import re
    # Remove common punctuation that adds noise
    cleaned = re.sub(r'[^\w\s]', ' ', comment)
    tokens = tokenizer.tokenize(cleaned, split_identifiers=True)
    return tokens


def print_header(title: str, width: int = 70) -> None:
    """Print a formatted header."""
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_separator(width: int = 70) -> None:
    """Print a separator line."""
    print("=" * width)
