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

# =============================================================================
# HOW THIS MODEL WORKS (Knowledge for Future Developers)
# =============================================================================
#
# TRAINING ALGORITHM:
# 1. Parse commit messages to extract commit type (feat, fix, docs, etc.)
# 2. Build file co-occurrence matrix from commits (files changed together)
# 3. Extract keywords from commit messages and map to files touched
# 4. Weight by recency and frequency using TF-IDF-style scoring
#
# PREDICTION ALGORITHM:
# 1. Classify input as commit type (feat, fix, docs, etc.)
# 2. Extract keywords from task description
# 3. Look up keyword->file associations from training data
# 4. Boost files that frequently co-occur with matched files (--seed option)
# 5. Apply frequency penalty to avoid over-suggesting common files
#
# WHY THIS WORKS (INTUITION):
# - Commits touching auth.py often also touch login.py and tests/test_auth.py
# - "feat: add authentication" commits tend to modify similar file sets
# - Keywords like "authentication", "login", "user" map to specific modules
# - The model learns YOUR project's specific patterns, not generic ones
#
# METRICS EXPLANATION:
# - MRR (Mean Reciprocal Rank): Where does the first correct file appear?
#   MRR=0.5 means first correct file is typically at position 2
# - Recall@10: What percentage of actual files appear in top 10 predictions?
#   Recall@10=0.5 means half of files you'll touch are predicted
# - Precision@1: How often is the top prediction correct?
#   Precision@1=0.3 means 30% of top predictions are actually modified
#
# WHEN TO RETRAIN:
# - After major refactoring (file paths changed)
# - Every 50-100 new commits (patterns evolve)
# - When predictions feel stale or inaccurate
# - After merging branches with different file structures
#
# =============================================================================
"""

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

from ml_collector.config import TRACKED_DIR, ML_DATA_DIR, CALI_DIR

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# CALI support (high-performance ML storage)
try:
    from cortical.ml_storage import MLStore
    CALI_AVAILABLE = True
except ImportError:
    CALI_AVAILABLE = False


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_DIR = ML_DATA_DIR / "models"
MODEL_HISTORY_DIR = MODEL_DIR / "history"
FILE_PREDICTION_MODEL = MODEL_DIR / "file_prediction.json"
AI_META_CACHE = MODEL_DIR / "ai_meta_cache.json"

# Warning thresholds
WARNING_LOW_CONFIDENCE_THRESHOLD = 0.5
WARNING_STALE_COMMITS_THRESHOLD = 10
WARNING_MIN_TRAINING_COMMITS = 50
WARNING_NO_KEYWORD_MATCH_THRESHOLD = 0

# Default minimum confidence for predictions
DEFAULT_MIN_CONFIDENCE = 0.1  # Default minimum confidence threshold for predictions

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

# Development-specific stop words (common terms that pollute predictions)
# These are filtered out during keyword extraction from commit messages
DEVELOPMENT_STOP_WORDS = {
    # Action verbs (too generic)
    'add', 'change', 'clean', 'ensure', 'fix', 'handle', 'implement',
    'improve', 'make', 'move', 'refactor', 'remove', 'rename', 'support',
    'update',
    # Task/workflow terms
    'bug', 'epic', 'feature', 'issue', 'merge', 'pr', 'review', 'sprint',
    'story', 'task', 'ticket', 'todo', 'wip',
    # Generic terms
    'also', 'code', 'could', 'file', 'just', 'more', 'need', 'new', 'now',
    'old', 'should', 'some', 'still', 'test', 'use', 'work', 'would',
}

# File path migrations (old → new structure)
# Used to map historical commits to current file structure
FILE_PATH_MIGRATIONS = {
    'cortical/analysis.py': [
        'cortical/analysis/__init__.py',
        'cortical/analysis/pagerank.py',
        'cortical/analysis/tfidf.py',
        'cortical/analysis/clustering.py',
        'cortical/analysis/connections.py',
        'cortical/analysis/activation.py',
        'cortical/analysis/quality.py',
        'cortical/analysis/utils.py',
    ],
    'cortical/processor.py': [
        'cortical/processor/__init__.py',
        'cortical/processor/core.py',
        'cortical/processor/compute.py',
        'cortical/processor/query_api.py',
        'cortical/processor/documents.py',
    ],
    'cortical/query.py': [
        'cortical/query/__init__.py',
        'cortical/query/expansion.py',
        'cortical/query/search.py',
        'cortical/query/passages.py',
        'cortical/query/ranking.py',
    ],
    'TASK_LIST.md': [],  # Deleted, no replacement
    'TASK_ARCHIVE.md': [],  # Deleted, no replacement
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
    # Package-specific keywords
    'query': ['cortical/query/', 'query/expansion.py', 'query/search.py', 'query_api.py'],
    'expansion': ['expansion.py', 'query/expansion.py'],
    'search': ['search.py', 'query/search.py', 'find_documents'],
    'processor': ['cortical/processor/', 'processor/compute.py', 'processor/core.py'],
    'compute': ['compute.py', 'processor/compute.py', 'compute_all'],
    'ranking': ['ranking.py', 'query/ranking.py'],
    'passages': ['passages.py', 'query/passages.py'],
}

# Module to test file mapping
# Maps source modules/files to their primary test files
# Supports both exact file paths and directory prefixes
MODULE_TO_TEST_MAPPING = {
    'cortical/processor/': 'tests/test_processor.py',
    'cortical/analysis.py': 'tests/test_analysis.py',
    'cortical/query/': 'tests/test_query.py',
    'cortical/semantics.py': 'tests/test_semantics.py',
    'cortical/persistence.py': 'tests/test_persistence.py',
    'cortical/tokenizer.py': 'tests/test_tokenizer.py',
    'cortical/config.py': 'tests/test_config.py',
    'cortical/layers.py': 'tests/test_layers.py',
    'cortical/embeddings.py': 'tests/test_embeddings.py',
    'cortical/gaps.py': 'tests/test_gaps.py',
    'cortical/fingerprint.py': 'tests/test_fingerprint.py',
    'cortical/minicolumn.py': 'tests/test_minicolumn.py',
    'cortical/chunk_index.py': 'tests/test_chunk_indexing.py',
    'cortical/observability.py': 'tests/test_observability.py',
    'cortical/code_concepts.py': 'tests/test_code_concepts.py',
    'scripts/ml_file_prediction.py': 'tests/unit/test_ml_file_prediction.py',
    'scripts/ml_data_collector.py': 'tests/unit/test_ml_data_collector.py',
}

# Reverse mapping: test file -> source module
TEST_TO_MODULE_MAPPING = {v: k for k, v in MODULE_TO_TEST_MAPPING.items()}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class PredictionWarning:
    """A warning about prediction reliability."""
    level: str  # 'info', 'warning', 'error'
    code: str   # Machine-readable code
    message: str  # Human-readable message

    def __str__(self) -> str:
        icons = {'info': 'ℹ️', 'warning': '⚠️', 'error': '❌'}
        return f"{icons.get(self.level, '•')} {self.message}"


@dataclass
class PredictionResult:
    """Result of a file prediction with warnings."""
    files: List[Tuple[str, float]]
    warnings: List[PredictionWarning] = field(default_factory=list)
    query_keywords: List[str] = field(default_factory=list)
    matched_keywords: List[str] = field(default_factory=list)

    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    def get_warnings_by_level(self, level: str) -> List[PredictionWarning]:
        return [w for w in self.warnings if w.level == level]


# ============================================================================
# SEMANTIC SIMILARITY
# ============================================================================

def compute_semantic_similarity(text1: str, text2: str) -> float:
    """
    Compute simple semantic similarity using word overlap and bigrams.

    Uses Jaccard similarity for word sets and bigram overlap, combined
    with a weighted average. This is a lightweight alternative to
    embedding-based similarity that requires no external dependencies.

    Args:
        text1: First text to compare
        text2: Second text to compare

    Returns:
        Similarity score between 0.0 and 1.0
    """
    # Normalize texts to lowercase words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    # Remove stop words (common English words that add little semantic value)
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of',
                  'in', 'for', 'on', 'with', 'at', 'by', 'from'}
    words1 -= stop_words
    words2 -= stop_words

    if not words1 or not words2:
        return 0.0

    # Jaccard similarity for words
    word_intersection = words1 & words2
    word_union = words1 | words2
    word_sim = len(word_intersection) / len(word_union)

    # Bigram overlap for multi-word pattern matching
    def get_bigrams(words):
        """Extract bigrams from a set of words."""
        word_list = sorted(words)
        return set(zip(word_list[:-1], word_list[1:]))

    bigrams1 = get_bigrams(words1)
    bigrams2 = get_bigrams(words2)

    if bigrams1 and bigrams2:
        bigram_intersection = bigrams1 & bigrams2
        bigram_union = bigrams1 | bigrams2
        bigram_sim = len(bigram_intersection) / len(bigram_union)
    else:
        bigram_sim = 0.0

    # Weighted combination: favor word similarity over bigrams
    # Word similarity is more reliable for short texts
    return 0.7 * word_sim + 0.3 * bigram_sim


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
class ModelMetrics:
    """Evaluation metrics for a model version."""
    mrr: float = 0.0
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    precision_at_1: float = 0.0
    test_examples: int = 0


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

    # File to commit messages mapping (for semantic similarity)
    # Stores recent commit messages for each file
    file_to_commits: Dict[str, List[str]] = field(default_factory=dict)

    # Total commits seen
    total_commits: int = 0

    # Training timestamp
    trained_at: str = ""

    # Model version
    version: str = "1.1.0"

    # Git commit hash at training time (for staleness detection)
    git_commit_hash: str = ""

    # Metrics from evaluation (if available)
    metrics: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FilePredictionModel':
        # Handle backwards compatibility
        if 'git_commit_hash' not in d:
            d['git_commit_hash'] = ''
        if 'metrics' not in d:
            d['metrics'] = None
        if 'file_to_commits' not in d:
            d['file_to_commits'] = {}
        return cls(**d)

    def get_staleness_commits(self) -> int:
        """Get number of commits since model was trained."""
        if not self.git_commit_hash:
            return -1  # Unknown
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-list', '--count', f'{self.git_commit_hash}..HEAD'],
                capture_output=True, text=True, check=True
            )
            return int(result.stdout.strip())
        except Exception:
            return -1


@dataclass
class AIMetaData:
    """Parsed AI metadata for a Python file."""
    filepath: str
    sections: List[str] = field(default_factory=list)  # Section names like "Persistence", "Query"
    functions: List[str] = field(default_factory=list)  # Function names
    imports: List[str] = field(default_factory=list)  # Local imports (other files)
    see_also: Dict[str, List[str]] = field(default_factory=dict)  # Function -> related functions


# ============================================================================
# AI METADATA LOADING
# ============================================================================

def load_ai_meta_file(meta_path: Path) -> Optional[AIMetaData]:
    """Load and parse a single .ai_meta file."""
    if not YAML_AVAILABLE:
        return None

    if not meta_path.exists():
        return None

    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if not data:
            return None

        # Extract section names
        sections = []
        if 'sections' in data and isinstance(data['sections'], list):
            for section in data['sections']:
                if isinstance(section, dict) and 'name' in section:
                    sections.append(section['name'].lower())

        # Extract function names and see_also references
        functions = []
        see_also_map = {}
        if 'functions' in data and isinstance(data['functions'], dict):
            for func_name, func_data in data['functions'].items():
                # Clean function names (remove class prefixes)
                clean_name = func_name.split('.')[-1]
                functions.append(clean_name.lower())

                # Extract see_also references
                if isinstance(func_data, dict) and 'see_also' in func_data:
                    see_also_map[clean_name.lower()] = [
                        s.lower() for s in func_data['see_also']
                    ]

        # Extract local imports (file references)
        # Note: .ai_meta files list local imports in both 'local' and 'stdlib' sections
        # We need to check both and filter for cortical module imports
        imports = []
        if 'imports' in data and isinstance(data['imports'], dict):
            # Check local section
            if 'local' in data['imports'] and isinstance(data['imports']['local'], list):
                imports.extend([imp.lower() for imp in data['imports']['local']])

            # Check stdlib section for cortical module imports
            # E.g., "layers.HierarchicalLayer" -> "layers"
            if 'stdlib' in data['imports'] and isinstance(data['imports']['stdlib'], list):
                for imp in data['imports']['stdlib']:
                    # Filter for cortical module imports (without dots after module name)
                    if '.' in imp and not imp.startswith(('typing.', 'collections.', 'os.', 'sys.')):
                        module_name = imp.split('.')[0].lower()
                        if module_name not in ['typing', 'collections', 'os', 'sys', 'json',
                                                'logging', 'pathlib', 're', 'math', 'hashlib',
                                                'hmac', 'pickle', 'warnings', 'dataclasses']:
                            imports.append(module_name)

        # Get the source file path
        filepath = data.get('file', '')
        if not filepath and data.get('filename'):
            # Reconstruct from filename if needed
            filepath = str(meta_path).replace('.ai_meta', '')

        return AIMetaData(
            filepath=filepath,
            sections=sections,
            functions=functions,
            imports=imports,
            see_also=see_also_map
        )

    except Exception as e:
        # Gracefully handle parsing errors
        return None


def load_all_ai_meta() -> Dict[str, AIMetaData]:
    """
    Load all .ai_meta files from the cortical/ directory.

    Returns:
        Dict mapping Python file paths to their AIMetaData
    """
    meta_map = {}

    if not YAML_AVAILABLE:
        return meta_map

    # Search for .ai_meta files
    cortical_dir = Path(__file__).parent.parent / "cortical"
    if not cortical_dir.exists():
        return meta_map

    for meta_file in cortical_dir.rglob("*.ai_meta"):
        meta_data = load_ai_meta_file(meta_file)
        if meta_data:
            # Map to the source Python file path
            py_file = str(meta_file).replace('.ai_meta', '')
            meta_map[py_file] = meta_data

    return meta_map


def cache_ai_meta(meta_map: Dict[str, AIMetaData], cache_path: Path = None) -> None:
    """Cache loaded AI metadata to avoid re-parsing YAML."""
    if cache_path is None:
        cache_path = AI_META_CACHE

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    cache_data = {
        filepath: asdict(meta_data)
        for filepath, meta_data in meta_map.items()
    }

    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=2)


def load_cached_ai_meta(cache_path: Path = None) -> Optional[Dict[str, AIMetaData]]:
    """Load cached AI metadata."""
    if cache_path is None:
        cache_path = AI_META_CACHE

    if not cache_path.exists():
        return None

    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)

        return {
            filepath: AIMetaData(**meta_dict)
            for filepath, meta_dict in cache_data.items()
        }
    except Exception:
        return None


def build_import_graph(meta_map: Dict[str, AIMetaData]) -> Dict[str, Set[str]]:
    """
    Build an import graph from AI metadata.

    Returns:
        Dict mapping file paths to sets of imported file paths
    """
    import_graph = defaultdict(set)

    for filepath, meta_data in meta_map.items():
        for imported_module in meta_data.imports:
            # Try to resolve import to a file path
            # e.g., "persistence" -> "cortical/persistence.py"
            if '/' not in imported_module and '.' not in imported_module:
                # Simple module name - try to find it
                for other_filepath in meta_map.keys():
                    if imported_module in other_filepath.lower():
                        import_graph[filepath].add(other_filepath)

    return import_graph


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

    # Filter stop words (general English stop words)
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

    # Filter both general and development-specific stop words
    keywords = [w for w in words
                if w not in stop_words
                and w not in DEVELOPMENT_STOP_WORDS
                and len(w) > 2]

    return keywords


def get_associated_files(filepath: str) -> List[str]:
    """
    Get associated test or source files for a given file path.

    If filepath is a source file, returns its test file(s).
    If filepath is a test file, returns its source file(s).
    Handles both exact matches and directory prefix matches.

    Args:
        filepath: Path to match (e.g., 'cortical/processor/core.py' or 'tests/test_processor.py')

    Returns:
        List of associated file paths (empty if no mapping found)
    """
    associated = []

    # Check if this is a test file -> find source files
    for test_file, source in TEST_TO_MODULE_MAPPING.items():
        if filepath == test_file:
            # Exact match - if source is a directory, we can't predict a specific file
            # but we can boost the directory in general patterns
            if source.endswith('/'):
                # Directory mapping - any file in that directory is associated
                associated.append(source)
            else:
                # Exact file mapping
                associated.append(source)

    # Check if this is a source file -> find test files
    for source, test_file in MODULE_TO_TEST_MAPPING.items():
        if source.endswith('/'):
            # Directory prefix mapping
            if filepath.startswith(source):
                associated.append(test_file)
        else:
            # Exact file mapping
            if filepath == source:
                associated.append(test_file)

    return associated


# ============================================================================
# FILE PATH UTILITIES
# ============================================================================

def migrate_file_path(file_path: str) -> List[str]:
    """
    Migrate old file paths to current structure.

    Returns a list of current paths that the old path maps to.
    If no migration needed, returns [file_path].
    If file was deleted with no replacement, returns [].
    """
    if file_path in FILE_PATH_MIGRATIONS:
        return FILE_PATH_MIGRATIONS[file_path]
    return [file_path]


def filter_existing_files(files: List[str]) -> List[str]:
    """
    Filter file list to only include files that currently exist.
    Also migrates old paths to new structure.
    """
    result = []
    seen = set()

    for f in files:
        # Try migrating the path first
        migrated = migrate_file_path(f)

        for path in migrated:
            if path in seen:
                continue

            # Check if file exists
            if Path(path).exists():
                result.append(path)
                seen.add(path)

    return result


def get_existing_files_set() -> Set[str]:
    """
    Get a set of all files that currently exist in the repository.
    Used for efficient filtering during training.
    """
    existing = set()

    # Scan common directories
    for pattern in ['cortical/**/*.py', 'scripts/*.py', 'tests/**/*.py',
                    'docs/*.md', '*.md', '.claude/**/*.md', 'tasks/*.json']:
        for p in Path('.').glob(pattern):
            existing.add(str(p))

    return existing


# ============================================================================
# GIT UTILITIES
# ============================================================================

def get_current_git_hash() -> str:
    """Get the current git commit hash."""
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()[:12]  # Short hash
    except Exception:
        return ""


def get_commits_since(commit_hash: str) -> int:
    """Get number of commits since the given hash."""
    if not commit_hash:
        return -1
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-list', '--count', f'{commit_hash}..HEAD'],
            capture_output=True, text=True, check=True
        )
        return int(result.stdout.strip())
    except Exception:
        return -1


# ============================================================================
# DATA LOADING
# ============================================================================

def _load_commits_from_cali() -> List[Dict[str, Any]]:
    """Load commits from CALI store (O(n) sequential iteration)."""
    if not CALI_AVAILABLE or not CALI_DIR.exists():
        return []

    try:
        store = MLStore(CALI_DIR, rebuild_indices=False)  # Don't need indices for iteration
        commits = list(store.iterate('commit'))
        store.close()
        return commits
    except Exception as e:
        print(f"  Warning: CALI read failed, falling back to JSONL: {e}")
        return []


def _load_commits_from_jsonl() -> List[Dict[str, Any]]:
    """Load commits from legacy JSONL file."""
    commits_file = TRACKED_DIR / "commits.jsonl"

    if not commits_file.exists():
        return []

    commits = []
    with open(commits_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                commits.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return commits


def load_commit_data(filter_deleted: bool = True, use_cali: bool = True) -> List[TrainingExample]:
    """
    Load commit data from CALI store or JSONL file.

    Args:
        filter_deleted: If True, filter out files that no longer exist
                       and migrate old paths to new structure.
        use_cali: If True, try CALI first, then fall back to JSONL.
    """
    # Try CALI first (faster iteration, no index needed)
    commits = []
    if use_cali:
        commits = _load_commits_from_cali()

    # Fall back to JSONL
    if not commits:
        commits = _load_commits_from_jsonl()

    if not commits:
        print(f"No commits found in CALI or {TRACKED_DIR / 'commits.jsonl'}")
        return []

    # Pre-compute existing files for efficiency
    existing_files = get_existing_files_set() if filter_deleted else None

    examples = []
    filtered_count = 0

    for commit in commits:
        # Skip merge commits and ML data commits
        if commit.get('is_merge', False):
            continue
        if commit.get('message', '').startswith('data: ML'):
            continue

        files_changed = commit.get('files_changed', [])

        # Apply file path migrations and filter deleted files
        if filter_deleted:
            migrated_files = []
            for f_path in files_changed:
                # Migrate old paths to new structure
                migrated = migrate_file_path(f_path)
                for new_path in migrated:
                    # Check if file exists (either in our set or on disk)
                    if new_path in existing_files or Path(new_path).exists():
                        migrated_files.append(new_path)
            original_count = len(files_changed)
            files_changed = list(set(migrated_files))  # Dedupe
            if original_count > 0 and len(files_changed) == 0:
                filtered_count += 1
                continue  # Skip commits with no remaining files

        example = TrainingExample(
            commit_hash=commit.get('hash', ''),
            message=commit.get('message', ''),
            files_changed=files_changed,
            commit_type=extract_commit_type(commit.get('message', '')),
            keywords=extract_keywords(commit.get('message', '')),
            timestamp=commit.get('timestamp', ''),
            insertions=commit.get('insertions', 0),
            deletions=commit.get('deletions', 0)
        )

        # Only include commits that changed files
        if example.files_changed:
            examples.append(example)

    if filter_deleted and filtered_count > 0:
        print(f"  Filtered {filtered_count} commits with only deleted files")

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
        version="1.1.0",
        git_commit_hash=get_current_git_hash(),
        metrics=None
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

        # Store commit message for semantic similarity
        # Keep most recent 10 messages per file to limit model size
        for f in files:
            if f not in model.file_to_commits:
                model.file_to_commits[f] = []
            model.file_to_commits[f].append(example.message)
            # Keep only the 10 most recent (FIFO)
            if len(model.file_to_commits[f]) > 10:
                model.file_to_commits[f] = model.file_to_commits[f][-10:]

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

def generate_warnings(
    model: FilePredictionModel,
    query: str,
    predictions: List[Tuple[str, float]],
    query_keywords: List[str],
    matched_keywords: List[str]
) -> List[PredictionWarning]:
    """Generate warnings about prediction reliability."""
    warnings = []

    # Check model staleness
    staleness = get_commits_since(model.git_commit_hash)
    if staleness > WARNING_STALE_COMMITS_THRESHOLD:
        warnings.append(PredictionWarning(
            level='warning',
            code='STALE_MODEL',
            message=f"Model is {staleness} commits behind HEAD. Consider retraining."
        ))
    elif staleness == -1 and model.git_commit_hash:
        warnings.append(PredictionWarning(
            level='info',
            code='STALENESS_UNKNOWN',
            message="Could not determine model staleness (git hash not found)."
        ))

    # Check training data size
    if model.total_commits < WARNING_MIN_TRAINING_COMMITS:
        warnings.append(PredictionWarning(
            level='warning',
            code='LOW_TRAINING_DATA',
            message=f"Only {model.total_commits} training commits. Predictions may be unreliable."
        ))

    # Check keyword matching
    if len(matched_keywords) == 0:
        warnings.append(PredictionWarning(
            level='warning',
            code='NO_KEYWORD_MATCH',
            message=f"No keywords matched from query. Predictions based on general patterns only."
        ))
    elif len(matched_keywords) < len(query_keywords) // 2:
        warnings.append(PredictionWarning(
            level='info',
            code='PARTIAL_KEYWORD_MATCH',
            message=f"Only {len(matched_keywords)}/{len(query_keywords)} query keywords matched."
        ))

    # Check prediction confidence
    if predictions:
        max_score = predictions[0][1]
        if max_score < WARNING_LOW_CONFIDENCE_THRESHOLD:
            warnings.append(PredictionWarning(
                level='warning',
                code='LOW_CONFIDENCE',
                message=f"Top prediction score ({max_score:.2f}) is below threshold. Results may be unreliable."
            ))

        # Check for many low-scoring predictions (flat distribution)
        if len(predictions) >= 3:
            score_range = max_score - predictions[min(2, len(predictions)-1)][1]
            if score_range < 0.1 and max_score < 1.0:
                warnings.append(PredictionWarning(
                    level='info',
                    code='FLAT_DISTRIBUTION',
                    message="Prediction scores are similar. Consider providing seed files."
                ))

        # Check for non-existent predicted files (shouldn't happen with filtering, but good to check)
        missing_count = sum(1 for f, _ in predictions if not Path(f).exists())
        if missing_count > 0:
            warnings.append(PredictionWarning(
                level='error',
                code='MISSING_FILES',
                message=f"{missing_count} predicted files don't exist. Model may need retraining."
            ))
    else:
        warnings.append(PredictionWarning(
            level='error',
            code='NO_PREDICTIONS',
            message="No files predicted. Query may be too generic or model needs more training."
        ))

    return warnings


def predict_files(
    query: str,
    model: FilePredictionModel,
    top_n: int = 10,
    seed_files: List[str] = None,
    ai_meta_map: Optional[Dict[str, AIMetaData]] = None,
    use_ai_meta: bool = True,
    use_semantic: bool = False,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
) -> List[Tuple[str, float]]:
    """
    Predict which files are likely to be modified for a given task.

    Args:
        query: Task description or commit message
        model: Trained file prediction model
        top_n: Number of files to return
        seed_files: Optional known files to use for co-occurrence boosting
        ai_meta_map: Optional AI metadata map for enhanced predictions
        use_ai_meta: Whether to use AI metadata (default True)
        use_semantic: Whether to use semantic similarity boosting (default False)
        min_confidence: Minimum confidence threshold for predictions

    Returns:
        List of (file_path, score) tuples sorted by relevance
    """
    file_scores: Dict[str, float] = defaultdict(float)

    # Load AI metadata if requested and not provided
    if use_ai_meta and ai_meta_map is None:
        ai_meta_map = load_cached_ai_meta()
        if ai_meta_map is None and YAML_AVAILABLE:
            ai_meta_map = load_all_ai_meta()
            if ai_meta_map:
                cache_ai_meta(ai_meta_map)

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

    # Boost based on semantic similarity with commit messages
    if use_semantic and model.file_to_commits:
        for filepath in model.file_to_commits:
            # Compare query with recent commit messages for this file
            for commit_msg in model.file_to_commits[filepath][:5]:  # Top 5 recent
                sim = compute_semantic_similarity(query, commit_msg)
                if sim > 0.2:  # Only boost if meaningful similarity
                    file_scores[filepath] += sim * 0.5  # 50% weight

    # Boost based on AI metadata
    if ai_meta_map:
        query_words = set(message_to_keywords(query))

        # Build import graph for relationship boosting
        import_graph = build_import_graph(ai_meta_map)

        for filepath, meta_data in ai_meta_map.items():
            # Section keyword matching
            # E.g., "add persistence" -> boost files with "Persistence" section
            for section in meta_data.sections:
                if section in query_words or any(word in section for word in query_words):
                    file_scores[filepath] += 2.0  # Boost for section match

            # Function name matching
            # E.g., "compute_pagerank" in query -> boost files with that function
            for func_name in meta_data.functions:
                if func_name in query_words:
                    file_scores[filepath] += 1.5  # Boost for function match
                # Partial match (e.g., "pagerank" matches "compute_pagerank")
                elif any(word in func_name or func_name in word for word in query_words):
                    file_scores[filepath] += 0.75  # Partial boost

            # Import relationship boosting
            # If seed files are provided, boost files they import or that import them
            if seed_files and filepath in import_graph:
                for seed in seed_files:
                    # Normalize seed path for comparison
                    if any(seed in imported or imported in seed for imported in import_graph[filepath]):
                        file_scores[filepath] += 1.0  # Boost for import relationship

        # Reverse import boost: if a file imports files in our current candidates, boost it
        if seed_files:
            for filepath, imported_files in import_graph.items():
                for seed in seed_files:
                    if any(seed in imp or imp in seed for imp in imported_files):
                        file_scores[filepath] += 0.5  # Weaker boost for reverse imports

    # Boost test files when their source module is predicted, and vice versa
    # This ensures when a source file is predicted, its test file gets boosted
    # and when a test file is predicted, the source gets boosted
    current_files = set(file_scores.keys())
    for filepath in list(current_files):  # Iterate over copy to allow modifications
        associated = get_associated_files(filepath)
        for assoc_file in associated:
            if assoc_file.endswith('/'):
                # Directory mapping - boost all files in that directory that exist
                for candidate_file in current_files:
                    if candidate_file.startswith(assoc_file):
                        file_scores[candidate_file] += file_scores[filepath] * 0.4
            else:
                # Exact file mapping - boost the associated file
                if Path(assoc_file).exists():
                    # Boost with 40% of the source file's score
                    file_scores[assoc_file] += file_scores[filepath] * 0.4

    # Apply file frequency penalty (avoid always recommending high-frequency files)
    max_freq = max(model.file_frequency.values()) if model.file_frequency else 1
    for f in file_scores:
        freq_penalty = 1.0 - (model.file_frequency.get(f, 0) / max_freq) * 0.3
        file_scores[f] *= freq_penalty

    # Sort and return top N (filtering out non-existent files)
    sorted_files = sorted(file_scores.items(), key=lambda x: -x[1])
    # Filter to only existing files - removes deleted/renamed files from predictions
    sorted_files = [(f, score) for f, score in sorted_files if Path(f).exists()]
    # Filter by minimum confidence threshold
    sorted_files = [(f, score) for f, score in sorted_files if score >= min_confidence]
    return sorted_files[:top_n]


def predict_files_with_warnings(
    query: str,
    model: FilePredictionModel,
    top_n: int = 10,
    seed_files: List[str] = None,
    ai_meta_map: Optional[Dict[str, AIMetaData]] = None,
    use_ai_meta: bool = True,
    use_semantic: bool = False,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
) -> PredictionResult:
    """
    Predict files with system warnings about reliability.

    Returns a PredictionResult with files and any applicable warnings.
    """
    # Extract query keywords for warning generation
    query_keywords = list(set(extract_keywords(query) + message_to_keywords(query)))

    # Track which keywords matched
    matched_keywords = []
    for kw in query_keywords:
        if kw in model.keyword_to_files:
            matched_keywords.append(kw)

    # Get predictions (without confidence filtering for warning generation)
    all_predictions = predict_files(
        query, model, top_n, seed_files, ai_meta_map, use_ai_meta, use_semantic, min_confidence=0.0
    )

    # Apply confidence filtering
    predictions = [(f, score) for f, score in all_predictions if score >= min_confidence]

    # Generate warnings
    warnings = generate_warnings(
        model, query, predictions, query_keywords, matched_keywords
    )

    # Add warning if all predictions were filtered out by min_confidence
    if not predictions and all_predictions:
        highest_score = all_predictions[0][1]
        highest_file = all_predictions[0][0]
        warnings.append(PredictionWarning(
            level='warning',
            code='ALL_BELOW_THRESHOLD',
            message=f"All predictions below min_confidence ({min_confidence:.2f}). "
                    f"Highest was {highest_score:.2f} for {highest_file}"
        ))

    return PredictionResult(
        files=predictions,
        warnings=warnings,
        query_keywords=query_keywords,
        matched_keywords=matched_keywords
    )


# ============================================================================
# MODEL VERSIONING
# ============================================================================

def save_model_version(model: FilePredictionModel, metrics: Dict[str, float] = None) -> str:
    """
    Save model to history with timestamp and metrics.

    Returns the path to the saved version.
    """
    MODEL_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    # Update model metrics if provided
    if metrics:
        model.metrics = metrics

    # Create versioned filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    git_hash = model.git_commit_hash or "unknown"
    filename = f"model_{timestamp}_{git_hash}.json"
    version_path = MODEL_HISTORY_DIR / filename

    with open(version_path, 'w', encoding='utf-8') as f:
        json.dump(model.to_dict(), f, indent=2)

    return str(version_path)


def list_model_versions() -> List[Dict[str, Any]]:
    """List all saved model versions with metadata."""
    if not MODEL_HISTORY_DIR.exists():
        return []

    versions = []
    for model_file in sorted(MODEL_HISTORY_DIR.glob("model_*.json"), reverse=True):
        try:
            with open(model_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            versions.append({
                'filename': model_file.name,
                'path': str(model_file),
                'trained_at': data.get('trained_at', ''),
                'git_commit_hash': data.get('git_commit_hash', ''),
                'total_commits': data.get('total_commits', 0),
                'metrics': data.get('metrics'),
                'version': data.get('version', '1.0.0')
            })
        except Exception:
            continue

    return versions


def load_model_version(version_path: str) -> Optional[FilePredictionModel]:
    """Load a specific model version from history."""
    path = Path(version_path)
    if not path.exists():
        return None

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return FilePredictionModel.from_dict(data)


def compare_model_predictions(
    query: str,
    model1: FilePredictionModel,
    model2: FilePredictionModel,
    top_n: int = 10
) -> Dict[str, Any]:
    """
    Compare predictions between two models.

    Useful for seeing how predictions change over time.
    """
    pred1 = predict_files(query, model1, top_n)
    pred2 = predict_files(query, model2, top_n)

    files1 = set(f for f, _ in pred1)
    files2 = set(f for f, _ in pred2)

    return {
        'query': query,
        'model1': {
            'trained_at': model1.trained_at,
            'git_hash': model1.git_commit_hash,
            'predictions': pred1
        },
        'model2': {
            'trained_at': model2.trained_at,
            'git_hash': model2.git_commit_hash,
            'predictions': pred2
        },
        'common_files': list(files1 & files2),
        'only_in_model1': list(files1 - files2),
        'only_in_model2': list(files2 - files1),
        'jaccard_similarity': len(files1 & files2) / len(files1 | files2) if files1 | files2 else 0
    }


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
# DASHBOARD
# ============================================================================

def show_dashboard(as_json: bool = False):
    """Display comprehensive ML evaluation dashboard."""
    # Import ML data collector functions
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from ml_collector.stats import count_data, calculate_data_size
        ml_stats_available = True
    except ImportError:
        ml_stats_available = False

    dashboard_data = {}

    # Model Status
    model = load_model()
    if model:
        model_file = FILE_PREDICTION_MODEL
        staleness = model.get_staleness_commits()
        model_status = {
            'trained_date': model.trained_at,
            'training_data_size': model.total_commits,
            'model_file_size': model_file.stat().st_size if model_file.exists() else 0,
            'git_commit_hash': model.git_commit_hash or 'unknown',
            'version': model.version,
            'staleness_commits': staleness
        }
    else:
        model_status = {
            'trained_date': 'Not trained',
            'training_data_size': 0,
            'model_file_size': 0,
            'git_commit_hash': 'unknown',
            'version': 'N/A',
            'staleness_commits': -1
        }
    dashboard_data['model_status'] = model_status

    # Performance Metrics
    if model and model.metrics:
        performance = {
            'mrr': model.metrics.get('mrr', 0.0),
            'recall_at_10': model.metrics.get('recall@10', 0.0),
            'precision_at_1': model.metrics.get('precision@1', 0.0),
            'test_examples': model.metrics.get('total_examples', 0)
        }
    else:
        performance = {
            'mrr': 0.0,
            'recall_at_10': 0.0,
            'precision_at_1': 0.0,
            'test_examples': 0
        }
    dashboard_data['performance'] = performance

    # Data Health
    if ml_stats_available:
        try:
            counts = count_data()
            sizes = calculate_data_size()
            data_health = {
                'total_commits': counts.get('commits_lite', 0),
                'total_chats': counts['chats'],
                'session_count': counts.get('sessions_lite', 0),
                'data_size': sizes['total']
            }
        except Exception:
            data_health = {
                'total_commits': 0,
                'total_chats': 0,
                'session_count': 0,
                'data_size': 0
            }
    else:
        data_health = {
            'total_commits': 0,
            'total_chats': 0,
            'session_count': 0,
            'data_size': 0
        }
    dashboard_data['data_health'] = data_health

    # Feature Status
    cortical_dir = Path(__file__).parent.parent / "cortical"
    ai_meta_available = YAML_AVAILABLE and (AI_META_CACHE.exists() or
                                            (cortical_dir.exists() and any(cortical_dir.rglob('*.ai_meta'))))
    features = {
        'ai_metadata_integration': 'yes' if ai_meta_available else 'no',
        'semantic_similarity': 'yes',  # Always available
        'confidence_threshold': DEFAULT_MIN_CONFIDENCE
    }
    dashboard_data['features'] = features

    if as_json:
        print(json.dumps(dashboard_data, indent=2))
        return

    # ASCII dashboard output
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║              ML FILE PREDICTION DASHBOARD                    ║")
    print("╠══════════════════════════════════════════════════════════════╣")

    # Model Status section
    print("║ MODEL STATUS                                                 ║")
    print("╠══════════════════════════════════════════════════════════════╣")

    # Format trained date (truncate if too long)
    trained_str = str(model_status['trained_date'])[:41]
    print(f"║   Trained Date:      {trained_str:<41} ║")
    print(f"║   Training Commits:  {model_status['training_data_size']:<41} ║")

    # Format model file size
    size = model_status['model_file_size']
    if size > 1024 * 1024:
        size_str = f"{size / 1024 / 1024:.2f} MB"
    elif size > 1024:
        size_str = f"{size / 1024:.2f} KB"
    else:
        size_str = f"{size} bytes"
    print(f"║   Model File Size:   {size_str:<41} ║")

    # Staleness indicator
    staleness = model_status['staleness_commits']
    if staleness == -1:
        staleness_str = "Unknown"
    elif staleness == 0:
        staleness_str = "Up to date"
    elif staleness > 10:
        staleness_str = f"⚠️  {staleness} commits behind"
    else:
        staleness_str = f"{staleness} commits behind"
    print(f"║   Staleness:         {staleness_str:<41} ║")

    # Performance Metrics section
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║ PERFORMANCE METRICS                                          ║")
    print("╠══════════════════════════════════════════════════════════════╣")

    if performance['test_examples'] > 0:
        print(f"║   MRR:               {performance['mrr']:.4f}                                      ║")
        print(f"║   Recall@10:         {performance['recall_at_10']:.4f}                                      ║")
        print(f"║   Precision@1:       {performance['precision_at_1']:.4f}                                      ║")
        print(f"║   Test Examples:     {performance['test_examples']:<41} ║")
    else:
        print("║   No evaluation metrics available.                           ║")
        print("║   Run: python scripts/ml_file_prediction.py evaluate         ║")

    # Data Health section
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║ DATA HEALTH                                                  ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║   Total Commits:     {data_health['total_commits']:<41} ║")
    print(f"║   Total Chats:       {data_health['total_chats']:<41} ║")
    print(f"║   Sessions:          {data_health['session_count']:<41} ║")

    # Format data size
    data_size = data_health['data_size']
    if data_size > 1024 * 1024:
        data_size_str = f"{data_size / 1024 / 1024:.2f} MB"
    elif data_size > 1024:
        data_size_str = f"{data_size / 1024:.2f} KB"
    else:
        data_size_str = f"{data_size} bytes"
    print(f"║   Data Size:         {data_size_str:<41} ║")

    # Feature Status section
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║ FEATURE STATUS                                               ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║   AI Metadata:       {features['ai_metadata_integration']:<41} ║")
    print(f"║   Semantic Sim:      {features['semantic_similarity']:<41} ║")
    print(f"║   Confidence Thresh: {features['confidence_threshold']:<41} ║")

    print("╚══════════════════════════════════════════════════════════════╝")


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
    train_parser.add_argument('--save-version', '-v', action='store_true',
                             help='Save versioned copy to history')
    train_parser.add_argument('--evaluate', '-e', action='store_true',
                             help='Evaluate after training (20% holdout)')

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict files for a task')
    predict_parser.add_argument('query', type=str, help='Task description')
    predict_parser.add_argument('--top', '-n', type=int, default=10,
                               help='Number of predictions')
    predict_parser.add_argument('--seed', '-s', type=str, nargs='*',
                               help='Seed files for co-occurrence boosting')
    predict_parser.add_argument('--no-ai-meta', action='store_true',
                               help='Disable AI metadata enhancement')
    predict_parser.add_argument('--no-warnings', action='store_true',
                               help='Suppress system warnings')
    predict_parser.add_argument('--verbose', action='store_true',
                               help='Show detailed warning information')
    predict_parser.add_argument('--use-semantic', action='store_true',
                               help='Enable semantic similarity boosting')
    predict_parser.add_argument('--min-confidence', type=float, default=DEFAULT_MIN_CONFIDENCE,
                               help=f'Minimum confidence threshold (default: {DEFAULT_MIN_CONFIDENCE})')

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    eval_parser.add_argument('--split', type=float, default=0.2,
                            help='Test split ratio')

    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show model statistics')

    # AI metadata command
    ai_meta_parser = subparsers.add_parser('ai-meta', help='Show AI metadata statistics')
    ai_meta_parser.add_argument('--rebuild', action='store_true',
                               help='Rebuild AI metadata cache from source')

    # History command
    history_parser = subparsers.add_parser('history', help='List model version history')
    history_parser.add_argument('--limit', '-n', type=int, default=10,
                               help='Number of versions to show')

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare predictions between model versions')
    compare_parser.add_argument('query', type=str, help='Query to compare')
    compare_parser.add_argument('--version1', type=str, required=True,
                               help='Path to first model version')
    compare_parser.add_argument('--version2', type=str, default=None,
                               help='Path to second model (default: current)')
    compare_parser.add_argument('--top', '-n', type=int, default=10,
                               help='Number of predictions to compare')

    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Show ML evaluation dashboard')
    dashboard_parser.add_argument('--json', action='store_true',
                                 help='Output as JSON')

    args = parser.parse_args()

    if args.command == 'train':
        print("Loading commit data...")
        examples = load_commit_data()
        print(f"Loaded {len(examples)} training examples")

        # Optionally evaluate with holdout
        metrics = None
        if args.evaluate:
            print("Splitting data for evaluation (80/20)...")
            train_examples, test_examples = train_test_split(examples, 0.2)
            print(f"Training on {len(train_examples)} examples...")
            model = train_model(train_examples)
            print(f"Evaluating on {len(test_examples)} examples...")
            metrics = evaluate_model(model, test_examples)
            model.metrics = metrics
        else:
            print("Training model...")
            model = train_model(examples)

        output_path = Path(args.output) if args.output else None
        path = save_model(model, output_path)

        print(f"\nModel trained and saved to {path}")
        print(f"  Git commit: {model.git_commit_hash or 'unknown'}")
        print(f"  Total commits: {model.total_commits}")
        print(f"  Unique files: {len(model.file_frequency)}")
        print(f"  Commit types: {len(model.type_to_files)}")
        print(f"  Keywords: {len(model.keyword_to_files)}")

        if metrics:
            print(f"\n  Evaluation metrics:")
            print(f"    MRR: {metrics.get('mrr', 0):.4f}")
            print(f"    Recall@5: {metrics.get('recall@5', 0):.4f}")
            print(f"    Recall@10: {metrics.get('recall@10', 0):.4f}")

        # Save versioned copy if requested
        if args.save_version:
            version_path = save_model_version(model, metrics)
            print(f"\n  Saved to history: {version_path}")

    elif args.command == 'predict':
        model = load_model()
        if not model:
            print("No trained model found. Run 'train' first.")
            return 1

        use_ai_meta = not args.no_ai_meta
        use_semantic = args.use_semantic

        if use_ai_meta and not YAML_AVAILABLE:
            print("Warning: PyYAML not available. AI metadata enhancement disabled.")
            print("Install with: pip install pyyaml")
            use_ai_meta = False

        # Use warnings-enabled prediction
        result = predict_files_with_warnings(
            args.query,
            model,
            top_n=args.top,
            seed_files=args.seed,
            use_ai_meta=use_ai_meta,
            use_semantic=use_semantic,
            min_confidence=args.min_confidence
        )

        print(f"\nPredicted files for: '{args.query}'")
        enhancements = []
        if use_ai_meta:
            enhancements.append("AI metadata")
        if use_semantic:
            enhancements.append("semantic similarity")
        if enhancements:
            print(f"(Using {' + '.join(enhancements)} enhancement)")
        print("-" * 60)
        for i, (filepath, score) in enumerate(result.files, 1):
            print(f"  {i:2}. {filepath:<45} ({score:.3f})")

        # Show warnings unless suppressed
        if not args.no_warnings and result.has_warnings():
            print("\n" + "-" * 60)
            print("System Warnings:")
            for warning in result.warnings:
                print(f"  {warning}")

            if args.verbose:
                print(f"\n  Query keywords: {', '.join(result.query_keywords[:10])}")
                print(f"  Matched keywords: {', '.join(result.matched_keywords[:10])}")

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

    elif args.command == 'ai-meta':
        if not YAML_AVAILABLE:
            print("Error: PyYAML not available. AI metadata requires pyyaml.")
            print("Install with: pip install pyyaml")
            return 1

        if args.rebuild:
            print("Rebuilding AI metadata cache from source...")
            ai_meta_map = load_all_ai_meta()
            if ai_meta_map:
                cache_ai_meta(ai_meta_map)
                print(f"Cached metadata for {len(ai_meta_map)} files")
            else:
                print("No .ai_meta files found in cortical/ directory")
                return 1
        else:
            ai_meta_map = load_cached_ai_meta()
            if ai_meta_map is None:
                print("No cached AI metadata found. Loading from source...")
                ai_meta_map = load_all_ai_meta()
                if ai_meta_map:
                    cache_ai_meta(ai_meta_map)

        if not ai_meta_map:
            print("No AI metadata available.")
            return 1

        print("\n" + "=" * 60)
        print("AI METADATA STATISTICS")
        print("=" * 60)
        print(f"  Files with metadata: {len(ai_meta_map)}")

        # Count sections
        all_sections = set()
        for meta in ai_meta_map.values():
            all_sections.update(meta.sections)
        print(f"  Unique sections:     {len(all_sections)}")

        # Count functions
        total_functions = sum(len(meta.functions) for meta in ai_meta_map.values())
        print(f"  Total functions:     {total_functions}")

        # Count imports
        total_imports = sum(len(meta.imports) for meta in ai_meta_map.values())
        print(f"  Total imports:       {total_imports}")

        # Build import graph
        import_graph = build_import_graph(ai_meta_map)
        print(f"  Import relationships: {sum(len(v) for v in import_graph.values())}")

        # Section distribution
        if all_sections:
            print("\n  Section distribution:")
            section_counts = defaultdict(int)
            for meta in ai_meta_map.values():
                for section in meta.sections:
                    section_counts[section] += 1
            for section, count in sorted(section_counts.items(), key=lambda x: -x[1])[:10]:
                print(f"    {section}: {count} files")

        # Most connected files (by imports)
        if import_graph:
            print("\n  Most connected files (by imports):")
            for filepath, imported in sorted(import_graph.items(),
                                            key=lambda x: -len(x[1]))[:5]:
                print(f"    {Path(filepath).name}: imports {len(imported)} files")

    elif args.command == 'history':
        versions = list_model_versions()

        if not versions:
            print("No model versions found in history.")
            print(f"Train with --save-version to start tracking: python {__file__} train -v")
            return 0

        print("\n" + "=" * 60)
        print("MODEL VERSION HISTORY")
        print("=" * 60)

        for i, v in enumerate(versions[:args.limit], 1):
            print(f"\n  [{i}] {v['filename']}")
            print(f"      Trained: {v['trained_at']}")
            print(f"      Git: {v['git_commit_hash'] or 'unknown'}")
            print(f"      Commits: {v['total_commits']}")
            if v['metrics']:
                mrr = v['metrics'].get('mrr', 0)
                r5 = v['metrics'].get('recall@5', 0)
                print(f"      MRR: {mrr:.4f}, Recall@5: {r5:.4f}")

        print(f"\n  Total versions: {len(versions)}")
        print(f"  History dir: {MODEL_HISTORY_DIR}")

    elif args.command == 'compare':
        # Load model 1 (historical version)
        model1 = load_model_version(args.version1)
        if not model1:
            print(f"Could not load model: {args.version1}")
            return 1

        # Load model 2 (current or specified)
        if args.version2:
            model2 = load_model_version(args.version2)
            if not model2:
                print(f"Could not load model: {args.version2}")
                return 1
        else:
            model2 = load_model()
            if not model2:
                print("No current model found. Run 'train' first.")
                return 1

        # Compare predictions
        comparison = compare_model_predictions(
            args.query, model1, model2, args.top
        )

        print("\n" + "=" * 60)
        print("MODEL PREDICTION COMPARISON")
        print("=" * 60)
        print(f"Query: '{args.query}'")

        print(f"\nModel 1: {model1.trained_at} ({model1.git_commit_hash or 'unknown'})")
        for i, (f, score) in enumerate(comparison['model1']['predictions'][:5], 1):
            marker = "✓" if f in comparison['common_files'] else " "
            print(f"  {marker} {i}. {Path(f).name:<35} ({score:.3f})")

        print(f"\nModel 2: {model2.trained_at} ({model2.git_commit_hash or 'unknown'})")
        for i, (f, score) in enumerate(comparison['model2']['predictions'][:5], 1):
            marker = "✓" if f in comparison['common_files'] else " "
            print(f"  {marker} {i}. {Path(f).name:<35} ({score:.3f})")

        print(f"\nComparison:")
        print(f"  Common predictions: {len(comparison['common_files'])}")
        print(f"  Only in Model 1: {len(comparison['only_in_model1'])}")
        print(f"  Only in Model 2: {len(comparison['only_in_model2'])}")
        print(f"  Jaccard similarity: {comparison['jaccard_similarity']:.2%}")

        if comparison['only_in_model1']:
            print(f"\n  New in Model 2 (not in Model 1):")
            for f in comparison['only_in_model2'][:3]:
                print(f"    + {Path(f).name}")

        if comparison['only_in_model2']:
            print(f"\n  Removed in Model 2 (was in Model 1):")
            for f in comparison['only_in_model1'][:3]:
                print(f"    - {Path(f).name}")

    elif args.command == 'dashboard':
        show_dashboard(as_json=args.json if hasattr(args, 'json') else False)

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
