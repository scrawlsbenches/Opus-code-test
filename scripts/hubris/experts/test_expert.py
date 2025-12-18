#!/usr/bin/env python3
"""
Test Expert

Micro-expert specialized in predicting which test files to run
for a given code change or task description.

Uses patterns learned from:
- Source file to test file mappings (naming conventions)
- Historical test failures (which tests fail when files change)
- Import analysis (which tests import which modules)
- Co-change patterns (tests that often change with source)
"""

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from micro_expert import MicroExpert, ExpertPrediction, ExpertMetrics


class TestExpert(MicroExpert):
    """
    Expert for predicting which tests to run for code changes.

    Uses patterns learned from commit history and file structure:
    - Source-to-test naming conventions (foo.py -> test_foo.py)
    - Historical co-changes (tests that change with source files)
    - Test failure patterns (which tests fail when files change)
    - Module-to-test mappings (cortical/query/ -> tests/test_query.py)

    Model Data Structure:
        source_to_tests: Dict[str, Dict[str, int]] - Source file -> test files mapping
        test_to_sources: Dict[str, Set[str]] - Test file -> source files it covers
        test_failure_patterns: Dict[str, Dict[str, int]] - Source file -> tests that failed
        module_to_tests: Dict[str, Dict[str, int]] - Module path -> relevant tests
        test_cochange: Dict[str, Dict[str, int]] - Test co-change frequency
        total_commits: int - Total commits in training data
    """

    def __init__(
        self,
        expert_id: str = "test_expert",
        version: str = "1.0.0",
        **kwargs
    ):
        """
        Initialize TestExpert.

        Args:
            expert_id: Unique identifier (default: "test_expert")
            version: Expert version (default: "1.0.0")
            **kwargs: Additional arguments passed to MicroExpert base class
        """
        # Remove expert_type from kwargs if present (avoids conflict when loading)
        kwargs.pop('expert_type', None)
        super().__init__(
            expert_id=expert_id,
            expert_type="test",
            version=version,
            **kwargs
        )

        # Ensure model_data has required keys
        if not self.model_data:
            self.model_data = {
                'source_to_tests': {},
                'test_to_sources': {},
                'test_failure_patterns': {},
                'module_to_tests': {},
                'test_cochange': {},
                'total_commits': 0
            }

    def predict(self, context: Dict[str, Any]) -> ExpertPrediction:
        """
        Predict tests to run for given context.

        Args:
            context: Dictionary with:
                - changed_files (List[str]): Files being modified
                - query (str, optional): Task description
                - top_n (int, optional): Number of predictions (default: 10)

        Returns:
            ExpertPrediction with ranked test files
        """
        changed_files = context.get('changed_files', [])
        query = context.get('query', '')
        top_n = context.get('top_n', 10)

        scores: Dict[str, float] = defaultdict(float)
        metadata = {
            'changed_files': changed_files,
            'scoring_signals': []
        }

        # Signal 1: Naming convention mapping
        convention_tests = self._predict_by_convention(changed_files)
        for test_file, score in convention_tests.items():
            scores[test_file] += score * 3.0  # High weight for conventions
        if convention_tests:
            metadata['scoring_signals'].append('naming_convention')

        # Signal 2: Historical source-to-test mapping
        historical_tests = self._predict_by_history(changed_files)
        for test_file, score in historical_tests.items():
            scores[test_file] += score * 2.0
        if historical_tests:
            metadata['scoring_signals'].append('historical_mapping')

        # Signal 3: Module-level test mapping
        module_tests = self._predict_by_module(changed_files)
        for test_file, score in module_tests.items():
            scores[test_file] += score * 1.5
        if module_tests:
            metadata['scoring_signals'].append('module_mapping')

        # Signal 4: Test failure patterns
        failure_tests = self._predict_by_failures(changed_files)
        for test_file, score in failure_tests.items():
            scores[test_file] += score * 2.5  # High weight for failure history
        if failure_tests:
            metadata['scoring_signals'].append('failure_patterns')

        # Signal 5: Query-based matching (if provided)
        if query:
            query_tests = self._predict_by_query(query)
            for test_file, score in query_tests.items():
                scores[test_file] += score * 1.0
            if query_tests:
                metadata['scoring_signals'].append('query_match')

        # Normalize and sort
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {k: v / max_score for k, v in scores.items()}

        sorted_tests = sorted(scores.items(), key=lambda x: -x[1])[:top_n]

        return ExpertPrediction(
            expert_id=self.expert_id,
            expert_type=self.expert_type,
            items=sorted_tests,
            metadata=metadata
        )

    def _predict_by_convention(self, changed_files: List[str]) -> Dict[str, float]:
        """
        Predict tests based on naming conventions.

        Conventions:
        - foo.py -> test_foo.py, tests/test_foo.py, tests/unit/test_foo.py
        - cortical/foo.py -> tests/test_foo.py
        - foo/bar.py -> tests/foo/test_bar.py
        """
        predictions: Dict[str, float] = {}

        for file_path in changed_files:
            if self._is_test_file(file_path):
                continue

            path = Path(file_path)
            stem = path.stem
            parent = path.parent

            # Generate possible test file names
            possible_tests = [
                f"test_{stem}.py",
                f"tests/test_{stem}.py",
                f"tests/unit/test_{stem}.py",
                f"tests/integration/test_{stem}.py",
            ]

            # Add module-specific test paths
            if str(parent) != '.':
                possible_tests.extend([
                    f"tests/{parent}/test_{stem}.py",
                    f"tests/test_{parent.name}.py",
                ])

            # Check against known tests in model
            source_to_tests = self.model_data.get('source_to_tests', {})
            known_tests = set()
            for tests_dict in source_to_tests.values():
                known_tests.update(tests_dict.keys())

            for test_pattern in possible_tests:
                # Exact match
                if test_pattern in known_tests:
                    predictions[test_pattern] = 1.0
                else:
                    # Fuzzy match - check if any known test contains the pattern
                    for known_test in known_tests:
                        if stem in known_test and 'test' in known_test:
                            predictions[known_test] = max(
                                predictions.get(known_test, 0), 0.7
                            )

        return predictions

    def _predict_by_history(self, changed_files: List[str]) -> Dict[str, float]:
        """Predict tests based on historical source-to-test mappings."""
        predictions: Dict[str, float] = defaultdict(float)
        source_to_tests = self.model_data.get('source_to_tests', {})
        total_commits = max(self.model_data.get('total_commits', 1), 1)

        for file_path in changed_files:
            if file_path in source_to_tests:
                for test_file, count in source_to_tests[file_path].items():
                    # TF-IDF style scoring
                    tf = count / total_commits
                    predictions[test_file] += tf

        return dict(predictions)

    def _predict_by_module(self, changed_files: List[str]) -> Dict[str, float]:
        """Predict tests based on module-level mappings."""
        predictions: Dict[str, float] = defaultdict(float)
        module_to_tests = self.model_data.get('module_to_tests', {})

        for file_path in changed_files:
            path = Path(file_path)
            # Try different module paths
            modules_to_try = [
                str(path.parent),
                path.parts[0] if path.parts else '',
                '/'.join(path.parts[:2]) if len(path.parts) >= 2 else ''
            ]

            for module in modules_to_try:
                if module and module in module_to_tests:
                    for test_file, count in module_to_tests[module].items():
                        predictions[test_file] += count * 0.5

        return dict(predictions)

    def _predict_by_failures(self, changed_files: List[str]) -> Dict[str, float]:
        """Predict tests based on historical failure patterns."""
        predictions: Dict[str, float] = defaultdict(float)
        failure_patterns = self.model_data.get('test_failure_patterns', {})

        for file_path in changed_files:
            if file_path in failure_patterns:
                for test_file, failure_count in failure_patterns[file_path].items():
                    # Higher weight for tests that frequently fail
                    predictions[test_file] += failure_count * 1.5

        return dict(predictions)

    def _predict_by_query(self, query: str) -> Dict[str, float]:
        """Predict tests based on query keywords."""
        predictions: Dict[str, float] = {}
        query_lower = query.lower()
        keywords = set(re.findall(r'\b\w+\b', query_lower))

        # Get all known test files
        source_to_tests = self.model_data.get('source_to_tests', {})
        all_tests: Set[str] = set()
        for tests_dict in source_to_tests.values():
            all_tests.update(tests_dict.keys())

        for test_file in all_tests:
            test_lower = test_file.lower()
            # Count keyword matches
            matches = sum(1 for kw in keywords if kw in test_lower)
            if matches > 0:
                predictions[test_file] = matches / len(keywords)

        return predictions

    def _is_test_file(self, file_path: str) -> bool:
        """Check if a file is a test file."""
        path_lower = file_path.lower()
        return (
            'test' in path_lower or
            path_lower.startswith('tests/') or
            path_lower.endswith('_test.py')
        )

    def train(self, commits: List[Dict[str, Any]]) -> None:
        """
        Train the expert on commit history.

        Args:
            commits: List of commit dictionaries with:
                - files (List[str]): Files changed in commit
                - message (str): Commit message
                - test_results (Dict, optional): Test pass/fail info
        """
        source_to_tests: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        test_to_sources: Dict[str, Set[str]] = defaultdict(set)
        module_to_tests: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        test_failure_patterns: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        test_cochange: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for commit in commits:
            files = commit.get('files', [])
            test_results = commit.get('test_results', {})

            # Separate source and test files
            source_files = [f for f in files if not self._is_test_file(f)]
            test_files = [f for f in files if self._is_test_file(f)]

            # Build source-to-test mapping
            for source in source_files:
                for test in test_files:
                    source_to_tests[source][test] += 1
                    test_to_sources[test].add(source)

                # Module-level mapping
                module = str(Path(source).parent)
                for test in test_files:
                    module_to_tests[module][test] += 1

            # Test co-change patterns
            for i, test1 in enumerate(test_files):
                for test2 in test_files[i+1:]:
                    test_cochange[test1][test2] += 1
                    test_cochange[test2][test1] += 1

            # Failure patterns (if test results provided)
            if test_results:
                failed_tests = test_results.get('failed', [])
                for source in source_files:
                    for failed_test in failed_tests:
                        test_failure_patterns[source][failed_test] += 1

        # Store in model_data
        self.model_data = {
            'source_to_tests': {k: dict(v) for k, v in source_to_tests.items()},
            'test_to_sources': {k: list(v) for k, v in test_to_sources.items()},
            'module_to_tests': {k: dict(v) for k, v in module_to_tests.items()},
            'test_failure_patterns': {k: dict(v) for k, v in test_failure_patterns.items()},
            'test_cochange': {k: dict(v) for k, v in test_cochange.items()},
            'total_commits': len(commits)
        }

    def get_coverage_estimate(self, changed_files: List[str]) -> float:
        """
        Estimate test coverage for changed files.

        Returns a score 0-1 indicating how well the changed files
        are covered by known tests.
        """
        if not changed_files:
            return 1.0

        source_to_tests = self.model_data.get('source_to_tests', {})
        covered = sum(1 for f in changed_files if f in source_to_tests)
        return covered / len(changed_files)


# Convenience function for quick predictions
def suggest_tests(changed_files: List[str], model_path: Optional[Path] = None) -> List[Tuple[str, float]]:
    """
    Suggest tests to run for changed files.

    Args:
        changed_files: List of files being changed
        model_path: Optional path to saved TestExpert model

    Returns:
        List of (test_file, confidence) tuples
    """
    if model_path and model_path.exists():
        expert = TestExpert.load(model_path)
    else:
        expert = TestExpert()

    prediction = expert.predict({'changed_files': changed_files})
    return prediction.items


if __name__ == '__main__':
    # Demo usage
    expert = TestExpert()

    # Simulate some training data
    training_commits = [
        {
            'files': ['cortical/query/search.py', 'tests/test_query.py'],
            'message': 'feat: Add graph boosted search'
        },
        {
            'files': ['cortical/analysis.py', 'tests/test_analysis.py'],
            'message': 'fix: PageRank convergence'
        },
        {
            'files': ['cortical/processor/core.py', 'tests/test_processor.py'],
            'message': 'refactor: Split processor module'
        },
    ]

    expert.train(training_commits)

    # Test prediction
    prediction = expert.predict({
        'changed_files': ['cortical/query/search.py', 'cortical/analysis.py'],
        'query': 'search improvements'
    })

    print("Predicted tests to run:")
    for test_file, confidence in prediction.items:
        print(f"  {test_file}: {confidence:.3f}")
