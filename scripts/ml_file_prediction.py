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

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_DIR = ML_DATA_DIR / "models"
FILE_PREDICTION_MODEL = MODEL_DIR / "file_prediction.json"
AI_META_CACHE = MODEL_DIR / "ai_meta_cache.json"

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

# File path migrations (old → new structure)
# Used to map historical commits to current file structure
FILE_PATH_MIGRATIONS = {
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
# DATA LOADING
# ============================================================================

def load_commit_data(filter_deleted: bool = True) -> List[TrainingExample]:
    """
    Load commit data from JSONL file.

    Args:
        filter_deleted: If True, filter out files that no longer exist
                       and migrate old paths to new structure.
    """
    commits_file = TRACKED_DIR / "commits.jsonl"

    if not commits_file.exists():
        print(f"No commits file found at {commits_file}")
        return []

    # Pre-compute existing files for efficiency
    existing_files = get_existing_files_set() if filter_deleted else None

    examples = []
    filtered_count = 0

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

            except json.JSONDecodeError:
                continue

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
    seed_files: List[str] = None,
    ai_meta_map: Optional[Dict[str, AIMetaData]] = None,
    use_ai_meta: bool = True
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
    predict_parser.add_argument('--no-ai-meta', action='store_true',
                               help='Disable AI metadata enhancement')

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

        use_ai_meta = not args.no_ai_meta

        if use_ai_meta and not YAML_AVAILABLE:
            print("Warning: PyYAML not available. AI metadata enhancement disabled.")
            print("Install with: pip install pyyaml")
            use_ai_meta = False

        predictions = predict_files(
            args.query,
            model,
            top_n=args.top,
            seed_files=args.seed,
            use_ai_meta=use_ai_meta
        )

        print(f"\nPredicted files for: '{args.query}'")
        if use_ai_meta:
            print("(Using AI metadata enhancement)")
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

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
