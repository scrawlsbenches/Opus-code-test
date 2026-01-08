"""
Shared utilities for audit commands.

This module contains constants and helper functions used across multiple commands.
"""

import os
import re
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Any

# ==============================================================================
# PATHS
# ==============================================================================

MODEL_DIR = Path(".audit_models")
BLOOM_MODEL = MODEL_DIR / "bloom_filter.pkl"
NAIVE_BAYES_MODEL = MODEL_DIR / "naive_bayes.pkl"
LSH_MODEL = MODEL_DIR / "lsh_index.pkl"
INDEX_MODEL = MODEL_DIR / "inverted_index.pkl"
TRIE_MODEL = MODEL_DIR / "marker_trie.pkl"
DEFAULT_TRAINING_DIR = Path("docs/audits")

# ==============================================================================
# PATTERNS
# ==============================================================================

SUSPICIOUS_PATTERNS = [
    "will be implemented",
    "will be done",
    "will be replaced",
    "will be handled",
    "when cdg index is implemented",
    "when feature is ready",
    "placeholder",
    "stub",
    "not implemented yet",
    "coming soon",
    "tbd",
    "fixme later",
    "hack",
    "temporary fix",
    "workaround",
]

COMMENT_MARKERS = [
    "FUTURE:",
    "TODO:",
    "FIXME:",
    "HACK:",
    "XXX:",
    "NOTE:",
    "WARNING:",
    "BUG:",
    "OPTIMIZE:",
    "REFACTOR:",
]

MISLEADING_PATTERNS = [
    (r'will be implemented', 'speculative'),
    (r'will be added', 'speculative'),
    (r'will be handled', 'speculative'),
    (r'will be replaced', 'speculative'),
    (r'will be done', 'speculative'),
    (r'will be fixed', 'speculative'),
    (r'^FUTURE:', 'future_marker'),
    (r'FUTURE\s*when', 'future_marker'),
    (r'when .* is implemented', 'speculative'),
    (r'when .* is ready', 'speculative'),
    (r'when .* is done', 'speculative'),
    (r'when feature is', 'speculative'),
    (r'placeholder', 'placeholder'),
    (r'\bstub\b', 'placeholder'),
    (r'not implemented yet', 'placeholder'),
    (r'eventually', 'vague'),
    (r'someday', 'vague'),
    (r'in the future', 'vague'),
    (r'later we', 'vague'),
    (r'planned to', 'vague'),
    (r'See:.*\.md', 'doc_reference'),
    (r'see docs/', 'doc_reference'),
]

ACCURATE_PATTERNS = [
    (r'^Returns?\s+', 'returns'),
    (r'^Returns:\s+', 'returns'),
    (r'returns\s+(True|False|None|the|a|an)\b', 'returns'),
    (r'^Args?:\s*', 'args'),
    (r'^Parameters?:\s*', 'args'),
    (r'^Params?:\s*', 'args'),
    (r'^Raises?\s+', 'raises'),
    (r'^Raises:\s+', 'raises'),
    (r'raises\s+(ValueError|TypeError|KeyError|RuntimeError)', 'raises'),
    (r'^This (is|uses|implements|creates|computes|validates)', 'implementation'),
    (r'^Implements\s+', 'implementation'),
    (r'^Uses\s+', 'implementation'),
    (r'^Creates\s+', 'implementation'),
    (r'O\([nN1]\)', 'complexity'),
    (r'O\(n\s*(log\s*n)?\)', 'complexity'),
    (r'runs in O\(', 'complexity'),
    (r'time complexity', 'complexity'),
    (r'^type:\s*', 'type_hint'),
    (r'^NOTE:\s+\w', 'note'),
    (r'^IMPORTANT:\s+', 'note'),
    (r'^TODO:\s+[A-Z]', 'todo'),
    (r'^FIXME:\s+[A-Z]', 'todo'),
]

EXCLUDE_PATTERNS = [
    r'^-+$',
    r'^=+$',
    r'^\s*$',
    r'^#\s*$',
    r'^\d+$',
    r'^[a-z]$',
    r'^type:\s*ignore',
    r'^noqa',
    r'^pylint',
    r'^pragma',
]


# ==============================================================================
# FILE UTILITIES
# ==============================================================================

def find_python_files(directory: str) -> List[str]:
    """Recursively find all Python files in directory."""
    python_files = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files


def extract_comments_from_file(file_path: str) -> List[Tuple[int, str]]:
    """Extract comments from a Python file."""
    comments = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                match = re.search(r'#\s*(.+)$', line)
                if match:
                    comment_text = match.group(1).strip()
                    if comment_text:
                        comments.append((line_num, comment_text))
    except Exception:
        pass
    return comments


# ==============================================================================
# MODEL UTILITIES
# ==============================================================================

def save_model(obj: Any, path: Path) -> None:
    """Save a model using pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Model saved to {path}")


def load_model(path: Path) -> Optional[Any]:
    """Load a model using pickle."""
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


# ==============================================================================
# TOKENIZATION
# ==============================================================================

def tokenize_comment(comment: str, tokenizer) -> List[str]:
    """Tokenize a comment using the Cortical tokenizer."""
    cleaned = re.sub(r'[^\w\s]', ' ', comment)
    return tokenizer.tokenize(cleaned, split_identifiers=True)
