#!/usr/bin/env python3
"""
Refactor Expert

Micro-expert specialized in predicting which files or code regions
would benefit from refactoring.

Uses patterns learned from:
- Historical refactoring commits (commits with "refactor:" prefix)
- File characteristics (size, complexity proxies, churn rate)
- Code smell heuristics (long files, many functions, deep nesting)
- Co-refactoring patterns (files often refactored together)

This expert answers: "Which files are candidates for refactoring?"
(Different from FileExpert which answers: "Which files to modify for a task?")
"""

import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from micro_expert import MicroExpert, ExpertPrediction, ExpertMetrics


class RefactorExpert(MicroExpert):
    """
    Expert for predicting files that may benefit from refactoring.

    Combines two approaches:
    1. Commit-pattern learning: Learn from historical "refactor:" commits
    2. Heuristic analysis: Detect code smells from file characteristics

    Signal Types:
        EXTRACT: Long methods/files that should be split
        INLINE: Over-abstracted code that should be simplified
        RENAME: Unclear naming patterns
        MOVE: Code in wrong location (cross-module dependencies)
        DEDUPLICATE: Duplicated code patterns
        SIMPLIFY: Overly complex logic

    Model Data Structure:
        refactor_frequency: Dict[str, int] - How often each file was refactored
        co_refactor: Dict[str, Dict[str, int]] - Files refactored together
        refactor_keywords: Dict[str, Dict[str, int]] - Keywords associated with refactored files
        file_characteristics: Dict[str, Dict] - Cached file stats (size, age, etc.)
        total_refactor_commits: int - Total refactoring commits in training data
    """

    # Refactoring signal types
    SIGNAL_EXTRACT = 'extract'      # File/method too large, needs splitting
    SIGNAL_INLINE = 'inline'        # Over-abstracted, should simplify
    SIGNAL_RENAME = 'rename'        # Unclear naming
    SIGNAL_MOVE = 'move'            # Code in wrong location
    SIGNAL_DEDUPLICATE = 'dedupe'   # Duplicated patterns
    SIGNAL_SIMPLIFY = 'simplify'    # Complex logic

    # Heuristic thresholds (configurable)
    DEFAULT_THRESHOLDS = {
        'large_file_lines': 500,        # Files over this are candidates for EXTRACT
        'very_large_file_lines': 1000,  # Strong signal for EXTRACT
        'high_churn_commits': 10,       # Files changed this often may need refactoring
        'old_file_days': 180,           # Files not touched in this long may have tech debt
        'many_functions': 20,           # Files with many functions may need EXTRACT
        'long_function_lines': 50,      # Functions over this may need EXTRACT
    }

    def __init__(
        self,
        expert_id: str = "refactor_expert",
        version: str = "1.0.0",
        thresholds: Optional[Dict[str, int]] = None,
        **kwargs
    ):
        """
        Initialize RefactorExpert.

        Args:
            expert_id: Unique identifier (default: "refactor_expert")
            version: Expert version (default: "1.0.0")
            thresholds: Optional custom thresholds for heuristics
            **kwargs: Additional arguments passed to MicroExpert base class
        """
        # Remove expert_type from kwargs if present (avoids conflict when loading)
        kwargs.pop('expert_type', None)
        super().__init__(
            expert_id=expert_id,
            expert_type="refactor",
            version=version,
            **kwargs
        )

        # Merge custom thresholds with defaults
        self.thresholds = {**self.DEFAULT_THRESHOLDS}
        if thresholds:
            self.thresholds.update(thresholds)

        # Ensure model_data has required keys
        if not self.model_data:
            self.model_data = {
                'refactor_frequency': {},
                'co_refactor': {},
                'refactor_keywords': {},
                'refactor_types': {},  # Maps files to detected refactor types
                'file_characteristics': {},
                'total_refactor_commits': 0
            }

    def predict(self, context: Dict[str, Any]) -> ExpertPrediction:
        """
        Predict files that may benefit from refactoring.

        Args:
            context: Dictionary with:
                - query (str, optional): Task description for context
                - files (List[str], optional): Specific files to analyze
                - signal_type (str, optional): Filter by specific signal type
                - top_n (int, optional): Number of predictions (default: 10)
                - include_heuristics (bool, optional): Run heuristic analysis (default: True)
                - repo_root (str, optional): Repository root for file analysis

        Returns:
            ExpertPrediction with ranked (file, confidence) pairs and signal metadata
        """
        query = context.get('query', '')
        files_to_analyze = context.get('files', [])
        signal_filter = context.get('signal_type')
        top_n = context.get('top_n', 10)
        include_heuristics = context.get('include_heuristics', True)
        repo_root = context.get('repo_root', '.')

        scores: Dict[str, float] = defaultdict(float)
        signals: Dict[str, List[str]] = defaultdict(list)  # file -> signal types
        metadata = {
            'query': query,
            'scoring_sources': [],
            'signal_counts': defaultdict(int)
        }

        # Source 1: Historical refactoring patterns
        if self.model_data.get('refactor_frequency'):
            historical_scores = self._score_by_history(query)
            for filepath, score in historical_scores.items():
                scores[filepath] += score * 2.0  # Weight for historical patterns
            if historical_scores:
                metadata['scoring_sources'].append('historical_patterns')

        # Source 2: Co-refactoring patterns (if files provided)
        if files_to_analyze and self.model_data.get('co_refactor'):
            corefactor_scores = self._score_by_corefactoring(files_to_analyze)
            for filepath, score in corefactor_scores.items():
                scores[filepath] += score * 1.5
                signals[filepath].append('co_refactor')
            if corefactor_scores:
                metadata['scoring_sources'].append('co_refactoring')

        # Source 3: Heuristic analysis
        if include_heuristics:
            # If specific files provided, analyze those; otherwise use known files
            target_files = files_to_analyze or list(self.model_data.get('refactor_frequency', {}).keys())

            for filepath in target_files:
                heuristic_result = self._analyze_file_heuristics(filepath, repo_root)
                if heuristic_result['score'] > 0:
                    scores[filepath] += heuristic_result['score']
                    signals[filepath].extend(heuristic_result['signals'])
                    for sig in heuristic_result['signals']:
                        metadata['signal_counts'][sig] += 1

            if any(scores.values()):
                metadata['scoring_sources'].append('heuristics')

        # Source 4: Query-based matching
        if query:
            query_scores = self._score_by_query(query)
            for filepath, score in query_scores.items():
                scores[filepath] += score * 1.0
            if query_scores:
                metadata['scoring_sources'].append('query_match')

        # Filter by signal type if specified
        if signal_filter:
            scores = {f: s for f, s in scores.items()
                     if signal_filter in signals.get(f, [])}

        # Normalize scores
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {k: v / max_score for k, v in scores.items()}

        # Sort and limit
        sorted_files = sorted(scores.items(), key=lambda x: -x[1])[:top_n]

        # Add signal information to metadata
        metadata['file_signals'] = {f: signals.get(f, []) for f, _ in sorted_files}
        metadata['signal_counts'] = dict(metadata['signal_counts'])

        return ExpertPrediction(
            expert_id=self.expert_id,
            expert_type=self.expert_type,
            items=sorted_files,
            metadata=metadata
        )

    def _score_by_history(self, query: str) -> Dict[str, float]:
        """Score files based on historical refactoring frequency."""
        scores: Dict[str, float] = {}
        refactor_freq = self.model_data.get('refactor_frequency', {})
        total_commits = max(self.model_data.get('total_refactor_commits', 1), 1)

        if not refactor_freq:
            return scores

        # Base score from refactoring frequency
        for filepath, count in refactor_freq.items():
            # TF-IDF-like: files refactored often but not constantly
            tf = count / total_commits
            # Diminishing returns for very frequently refactored files
            scores[filepath] = tf * (1.0 - tf * 0.5)

        # Boost based on query keyword match
        if query:
            keywords = self._extract_keywords(query)
            keyword_files = self.model_data.get('refactor_keywords', {})

            for keyword in keywords:
                if keyword in keyword_files:
                    for filepath, count in keyword_files[keyword].items():
                        scores[filepath] = scores.get(filepath, 0) + count * 0.1

        return scores

    def _score_by_corefactoring(self, seed_files: List[str]) -> Dict[str, float]:
        """Score files based on co-refactoring patterns with seed files."""
        scores: Dict[str, float] = defaultdict(float)
        co_refactor = self.model_data.get('co_refactor', {})

        for seed in seed_files:
            if seed in co_refactor:
                for related_file, count in co_refactor[seed].items():
                    if related_file not in seed_files:
                        scores[related_file] += count

        return dict(scores)

    def _analyze_file_heuristics(
        self,
        filepath: str,
        repo_root: str = '.'
    ) -> Dict[str, Any]:
        """
        Analyze a file for refactoring signals using heuristics.

        Args:
            filepath: Path to file (relative to repo_root)
            repo_root: Repository root directory

        Returns:
            Dict with 'score' (float) and 'signals' (List[str])
        """
        result = {'score': 0.0, 'signals': []}
        full_path = Path(repo_root) / filepath

        # Skip if file doesn't exist or isn't a Python file
        if not full_path.exists() or not filepath.endswith('.py'):
            # Use cached characteristics if available
            cached = self.model_data.get('file_characteristics', {}).get(filepath)
            if cached:
                return self._score_from_cached_characteristics(cached)
            return result

        try:
            content = full_path.read_text(encoding='utf-8', errors='ignore')
            lines = content.split('\n')
            line_count = len(lines)

            # Heuristic 1: File size (EXTRACT signal)
            if line_count > self.thresholds['very_large_file_lines']:
                result['score'] += 0.8
                result['signals'].append(self.SIGNAL_EXTRACT)
            elif line_count > self.thresholds['large_file_lines']:
                result['score'] += 0.4
                result['signals'].append(self.SIGNAL_EXTRACT)

            # Heuristic 2: Function count (EXTRACT signal)
            function_count = len(re.findall(r'^\s*def\s+\w+', content, re.MULTILINE))
            class_count = len(re.findall(r'^\s*class\s+\w+', content, re.MULTILINE))

            if function_count > self.thresholds['many_functions']:
                result['score'] += 0.3
                if self.SIGNAL_EXTRACT not in result['signals']:
                    result['signals'].append(self.SIGNAL_EXTRACT)

            # Heuristic 3: Long functions (EXTRACT signal)
            long_functions = self._detect_long_functions(content)
            if long_functions:
                result['score'] += 0.2 * len(long_functions)
                if self.SIGNAL_EXTRACT not in result['signals']:
                    result['signals'].append(self.SIGNAL_EXTRACT)

            # Heuristic 4: Deep nesting (SIMPLIFY signal)
            max_indent = self._detect_max_indentation(lines)
            if max_indent > 6:  # More than 6 levels deep
                result['score'] += 0.3
                result['signals'].append(self.SIGNAL_SIMPLIFY)

            # Heuristic 5: Many imports (potential MOVE signal)
            import_count = len(re.findall(r'^(?:from|import)\s+', content, re.MULTILINE))
            if import_count > 20:
                result['score'] += 0.2
                result['signals'].append(self.SIGNAL_MOVE)

            # Heuristic 6: TODO/FIXME/HACK comments (general tech debt)
            debt_markers = len(re.findall(r'#\s*(TODO|FIXME|HACK|XXX)', content, re.IGNORECASE))
            if debt_markers > 5:
                result['score'] += 0.1 * min(debt_markers, 10)

            # Cache the characteristics
            self.model_data.setdefault('file_characteristics', {})[filepath] = {
                'line_count': line_count,
                'function_count': function_count,
                'class_count': class_count,
                'max_indent': max_indent,
                'import_count': import_count,
                'debt_markers': debt_markers,
                'long_functions': len(long_functions)
            }

        except Exception:
            pass  # File read error, skip heuristics

        return result

    def _score_from_cached_characteristics(self, cached: Dict[str, Any]) -> Dict[str, Any]:
        """Score based on cached file characteristics."""
        result = {'score': 0.0, 'signals': []}

        line_count = cached.get('line_count', 0)
        if line_count > self.thresholds['very_large_file_lines']:
            result['score'] += 0.8
            result['signals'].append(self.SIGNAL_EXTRACT)
        elif line_count > self.thresholds['large_file_lines']:
            result['score'] += 0.4
            result['signals'].append(self.SIGNAL_EXTRACT)

        if cached.get('function_count', 0) > self.thresholds['many_functions']:
            result['score'] += 0.3
            if self.SIGNAL_EXTRACT not in result['signals']:
                result['signals'].append(self.SIGNAL_EXTRACT)

        if cached.get('max_indent', 0) > 6:
            result['score'] += 0.3
            result['signals'].append(self.SIGNAL_SIMPLIFY)

        return result

    def _detect_long_functions(self, content: str) -> List[str]:
        """Detect functions that exceed the length threshold."""
        long_functions = []
        threshold = self.thresholds['long_function_lines']

        # Simple heuristic: count lines between def statements
        lines = content.split('\n')
        current_func = None
        func_start = 0

        for i, line in enumerate(lines):
            if re.match(r'^\s*def\s+(\w+)', line):
                # Check previous function
                if current_func and (i - func_start) > threshold:
                    long_functions.append(current_func)

                match = re.match(r'^\s*def\s+(\w+)', line)
                current_func = match.group(1) if match else None
                func_start = i

        # Check last function
        if current_func and (len(lines) - func_start) > threshold:
            long_functions.append(current_func)

        return long_functions

    def _detect_max_indentation(self, lines: List[str]) -> int:
        """Detect maximum indentation depth in code."""
        max_indent = 0
        for line in lines:
            if line.strip() and not line.strip().startswith('#'):
                # Count leading spaces (assuming 4-space indent)
                spaces = len(line) - len(line.lstrip())
                indent_level = spaces // 4
                max_indent = max(max_indent, indent_level)
        return max_indent

    def _score_by_query(self, query: str) -> Dict[str, float]:
        """Score files based on query keywords matching refactoring history."""
        scores: Dict[str, float] = {}
        keywords = self._extract_keywords(query)

        # Match keywords against file paths
        refactor_freq = self.model_data.get('refactor_frequency', {})
        for filepath in refactor_freq:
            path_lower = filepath.lower()
            matches = sum(1 for kw in keywords if kw in path_lower)
            if matches > 0:
                scores[filepath] = matches / len(keywords)

        return scores

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract meaningful keywords from text."""
        words = re.findall(r'\b[a-z_][a-z0-9_]*\b', text.lower())
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'to', 'of', 'in', 'for', 'on', 'with',
            'refactor', 'refactoring', 'file', 'code', 'this', 'that', 'and', 'or'
        }
        return set(w for w in words if w not in stop_words and len(w) > 2)

    def train(self, commits: List[Dict[str, Any]]) -> None:
        """
        Train the expert on commit history.

        Focuses on commits with "refactor:" prefix or refactoring-related messages.

        Args:
            commits: List of commit dictionaries with:
                - files (List[str]): Files changed in commit
                - message (str): Commit message
                - timestamp (str, optional): Commit timestamp
        """
        refactor_frequency: Dict[str, int] = defaultdict(int)
        co_refactor: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        refactor_keywords: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        refactor_types: Dict[str, Set[str]] = defaultdict(set)
        refactor_commit_count = 0

        for commit in commits:
            message = commit.get('message', '')
            files = commit.get('files', [])

            # Check if this is a refactoring commit
            if not self._is_refactor_commit(message):
                continue

            refactor_commit_count += 1

            # Extract refactoring type from message
            detected_types = self._detect_refactor_type(message)

            # Update frequency counts
            for filepath in files:
                refactor_frequency[filepath] += 1
                refactor_types[filepath].update(detected_types)

            # Update co-refactoring matrix
            for i, file1 in enumerate(files):
                for file2 in files[i+1:]:
                    co_refactor[file1][file2] += 1
                    co_refactor[file2][file1] += 1

            # Extract keywords and associate with files
            keywords = self._extract_keywords(message)
            for keyword in keywords:
                for filepath in files:
                    refactor_keywords[keyword][filepath] += 1

        # Store in model_data
        self.model_data = {
            'refactor_frequency': dict(refactor_frequency),
            'co_refactor': {k: dict(v) for k, v in co_refactor.items()},
            'refactor_keywords': {k: dict(v) for k, v in refactor_keywords.items()},
            'refactor_types': {k: list(v) for k, v in refactor_types.items()},
            'file_characteristics': self.model_data.get('file_characteristics', {}),
            'total_refactor_commits': refactor_commit_count
        }

        self.trained_on_commits = refactor_commit_count

    def _is_refactor_commit(self, message: str) -> bool:
        """Check if a commit message indicates refactoring."""
        msg_lower = message.lower()

        # Explicit refactor prefix
        if msg_lower.startswith('refactor:') or msg_lower.startswith('refactor('):
            return True

        # Keywords that suggest refactoring
        refactor_indicators = [
            'refactor', 'restructure', 'reorganize', 'cleanup', 'clean up',
            'split', 'extract', 'inline', 'rename', 'move', 'consolidate',
            'simplify', 'deduplicate', 'dedup', 'modularize'
        ]

        return any(indicator in msg_lower for indicator in refactor_indicators)

    def _detect_refactor_type(self, message: str) -> List[str]:
        """Detect refactoring type from commit message."""
        msg_lower = message.lower()
        types = []

        type_patterns = {
            self.SIGNAL_EXTRACT: ['split', 'extract', 'separate', 'break up', 'modularize'],
            self.SIGNAL_INLINE: ['inline', 'merge', 'combine', 'consolidate'],
            self.SIGNAL_RENAME: ['rename', 'naming'],
            self.SIGNAL_MOVE: ['move', 'relocate', 'reorganize'],
            self.SIGNAL_DEDUPLICATE: ['deduplicate', 'dedup', 'remove duplicate', 'dry'],
            self.SIGNAL_SIMPLIFY: ['simplify', 'cleanup', 'clean up', 'reduce complexity']
        }

        for signal_type, patterns in type_patterns.items():
            if any(pattern in msg_lower for pattern in patterns):
                types.append(signal_type)

        return types if types else [self.SIGNAL_SIMPLIFY]  # Default

    def analyze_codebase(
        self,
        repo_root: str = '.',
        file_patterns: List[str] = None,
        top_n: int = 20
    ) -> ExpertPrediction:
        """
        Analyze entire codebase for refactoring candidates.

        Convenience method that scans files and returns ranked suggestions.

        Args:
            repo_root: Repository root directory
            file_patterns: Glob patterns for files to include (default: ['**/*.py'])
            top_n: Number of top candidates to return

        Returns:
            ExpertPrediction with refactoring candidates
        """
        if file_patterns is None:
            file_patterns = ['**/*.py']

        # Collect files matching patterns
        files_to_analyze = []
        root = Path(repo_root)

        for pattern in file_patterns:
            for filepath in root.glob(pattern):
                if filepath.is_file():
                    rel_path = str(filepath.relative_to(root))
                    # Skip test files and hidden directories
                    if not rel_path.startswith('.') and 'test' not in rel_path.lower():
                        files_to_analyze.append(rel_path)

        return self.predict({
            'files': files_to_analyze,
            'top_n': top_n,
            'include_heuristics': True,
            'repo_root': repo_root
        })

    def get_file_report(self, filepath: str, repo_root: str = '.') -> Dict[str, Any]:
        """
        Get detailed refactoring report for a specific file.

        Args:
            filepath: Path to file
            repo_root: Repository root

        Returns:
            Detailed report with characteristics and recommendations
        """
        result = self._analyze_file_heuristics(filepath, repo_root)
        cached = self.model_data.get('file_characteristics', {}).get(filepath, {})
        history = self.model_data.get('refactor_frequency', {}).get(filepath, 0)
        past_types = self.model_data.get('refactor_types', {}).get(filepath, [])

        return {
            'filepath': filepath,
            'refactor_score': result['score'],
            'signals': result['signals'],
            'characteristics': cached,
            'historical_refactors': history,
            'past_refactor_types': past_types,
            'recommendations': self._generate_recommendations(result['signals'], cached)
        }

    def _generate_recommendations(
        self,
        signals: List[str],
        characteristics: Dict[str, Any]
    ) -> List[str]:
        """Generate human-readable recommendations based on signals."""
        recommendations = []

        if self.SIGNAL_EXTRACT in signals:
            line_count = characteristics.get('line_count', 0)
            func_count = characteristics.get('function_count', 0)
            if line_count > self.thresholds['very_large_file_lines']:
                recommendations.append(
                    f"File has {line_count} lines - consider splitting into smaller modules"
                )
            if func_count > self.thresholds['many_functions']:
                recommendations.append(
                    f"File has {func_count} functions - consider grouping related functions into separate files"
                )

        if self.SIGNAL_SIMPLIFY in signals:
            max_indent = characteristics.get('max_indent', 0)
            if max_indent > 6:
                recommendations.append(
                    f"Deep nesting detected ({max_indent} levels) - consider extracting nested logic"
                )

        if self.SIGNAL_MOVE in signals:
            import_count = characteristics.get('import_count', 0)
            if import_count > 20:
                recommendations.append(
                    f"Many imports ({import_count}) - consider if this file has too many responsibilities"
                )

        debt_markers = characteristics.get('debt_markers', 0)
        if debt_markers > 0:
            recommendations.append(
                f"Found {debt_markers} TODO/FIXME/HACK comments - review for tech debt"
            )

        return recommendations

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RefactorExpert':
        """Load RefactorExpert from dict."""
        metrics = None
        if data.get('metrics'):
            metrics = ExpertMetrics.from_dict(data['metrics'])

        return cls(
            expert_id=data.get('expert_id', 'refactor_expert'),
            version=data['version'],
            created_at=data['created_at'],
            trained_on_commits=data['trained_on_commits'],
            trained_on_sessions=data['trained_on_sessions'],
            git_hash=data['git_hash'],
            model_data=data['model_data'],
            metrics=metrics,
            calibration_curve=data.get('calibration_curve')
        )


# Convenience functions

def suggest_refactoring(
    files: Optional[List[str]] = None,
    repo_root: str = '.',
    model_path: Optional[Path] = None,
    top_n: int = 10
) -> List[Tuple[str, float, List[str]]]:
    """
    Suggest files that may need refactoring.

    Args:
        files: Specific files to analyze (None = scan codebase)
        repo_root: Repository root directory
        model_path: Optional path to saved RefactorExpert model
        top_n: Number of suggestions to return

    Returns:
        List of (filepath, score, signals) tuples
    """
    if model_path and model_path.exists():
        expert = RefactorExpert.load(model_path)
    else:
        expert = RefactorExpert()

    if files:
        prediction = expert.predict({
            'files': files,
            'top_n': top_n,
            'include_heuristics': True,
            'repo_root': repo_root
        })
    else:
        prediction = expert.analyze_codebase(repo_root=repo_root, top_n=top_n)

    # Combine with signal information
    file_signals = prediction.metadata.get('file_signals', {})
    return [
        (filepath, score, file_signals.get(filepath, []))
        for filepath, score in prediction.items
    ]


if __name__ == '__main__':
    # Demo usage
    print("RefactorExpert Demo")
    print("=" * 60)

    expert = RefactorExpert()

    # Simulate training data (refactoring commits)
    training_commits = [
        {
            'files': ['cortical/processor/compute.py', 'cortical/processor/query_api.py'],
            'message': 'refactor: Split large processor module into mixins'
        },
        {
            'files': ['cortical/analysis.py'],
            'message': 'refactor: Extract clustering logic to separate functions'
        },
        {
            'files': ['scripts/hubris/expert_consolidator.py'],
            'message': 'refactor: Simplify expert loading logic'
        },
        {
            'files': ['cortical/query/search.py', 'cortical/query/expansion.py'],
            'message': 'refactor: Move query expansion to dedicated module'
        },
    ]

    print("\nTraining on refactoring commits...")
    expert.train(training_commits)
    print(f"  Trained on {expert.trained_on_commits} refactoring commits")

    # Test prediction
    print("\nPredicting refactoring candidates...")
    prediction = expert.predict({
        'query': 'improve search module',
        'files': ['cortical/analysis.py', 'cortical/processor/compute.py'],
        'include_heuristics': True,
        'repo_root': '.'
    })

    print("\nTop refactoring candidates:")
    for filepath, score in prediction.items:
        signals = prediction.metadata.get('file_signals', {}).get(filepath, [])
        signals_str = ', '.join(signals) if signals else 'historical'
        print(f"  {filepath}: {score:.3f} [{signals_str}]")

    print(f"\nScoring sources: {prediction.metadata.get('scoring_sources', [])}")
