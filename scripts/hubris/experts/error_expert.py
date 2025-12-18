#!/usr/bin/env python3
"""
Error Diagnosis Expert

Micro-expert specialized in diagnosing errors and suggesting fixes
based on error messages, stack traces, and historical error patterns.

Uses patterns learned from:
- Error message to fix mappings
- Stack trace patterns
- Common error categories
- Historical resolutions
"""

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from micro_expert import MicroExpert, ExpertPrediction, ExpertMetrics


class ErrorDiagnosisExpert(MicroExpert):
    """
    Expert for diagnosing errors and suggesting fixes.

    Uses patterns learned from error history:
    - Error type to common causes mapping
    - Stack trace patterns to file mappings
    - Error message keywords to solutions
    - Historical fix patterns

    Model Data Structure:
        error_to_files: Dict[str, Dict[str, int]] - Error type -> files that fixed it
        error_to_causes: Dict[str, List[str]] - Error type -> common causes
        keyword_to_fixes: Dict[str, List[str]] - Error keywords -> fix descriptions
        stack_patterns: Dict[str, Dict[str, int]] - Stack trace pattern -> relevant files
        error_categories: Dict[str, List[str]] - Category -> error types
        resolution_history: List[Dict] - Historical error resolutions
        total_errors: int - Total errors in training data
    """

    # Common error categories
    ERROR_CATEGORIES = {
        'import': ['ImportError', 'ModuleNotFoundError', 'ImportWarning'],
        'type': ['TypeError', 'AttributeError', 'ValueError'],
        'runtime': ['RuntimeError', 'RecursionError', 'MemoryError'],
        'syntax': ['SyntaxError', 'IndentationError', 'TabError'],
        'io': ['FileNotFoundError', 'IOError', 'PermissionError'],
        'key': ['KeyError', 'IndexError', 'LookupError'],
        'assertion': ['AssertionError'],
        'network': ['ConnectionError', 'TimeoutError', 'ConnectionRefusedError'],
    }

    def __init__(
        self,
        expert_id: str = "error_expert",
        version: str = "1.0.0",
        **kwargs
    ):
        """
        Initialize ErrorDiagnosisExpert.

        Args:
            expert_id: Unique identifier (default: "error_expert")
            version: Expert version (default: "1.0.0")
            **kwargs: Additional arguments passed to MicroExpert base class
        """
        # Remove expert_type from kwargs if present (avoids conflict when loading)
        kwargs.pop('expert_type', None)
        super().__init__(
            expert_id=expert_id,
            expert_type="error",
            version=version,
            **kwargs
        )

        # Ensure model_data has required keys
        if not self.model_data:
            self.model_data = {
                'error_to_files': {},
                'error_to_causes': {},
                'keyword_to_fixes': {},
                'stack_patterns': {},
                'error_categories': self.ERROR_CATEGORIES.copy(),
                'resolution_history': [],
                'total_errors': 0
            }

    def predict(self, context: Dict[str, Any]) -> ExpertPrediction:
        """
        Diagnose an error and suggest fixes.

        Args:
            context: Dictionary with:
                - error_message (str): The error message
                - stack_trace (str, optional): Full stack trace
                - error_type (str, optional): Exception type (e.g., 'TypeError')
                - file_context (str, optional): File where error occurred
                - top_n (int, optional): Number of suggestions (default: 5)

        Returns:
            ExpertPrediction with ranked fix suggestions
        """
        error_message = context.get('error_message', '')
        stack_trace = context.get('stack_trace', '')
        error_type = context.get('error_type', '')
        file_context = context.get('file_context', '')
        top_n = context.get('top_n', 5)

        # Auto-detect error type if not provided
        if not error_type:
            error_type = self._extract_error_type(error_message, stack_trace)

        scores: Dict[str, float] = defaultdict(float)
        metadata = {
            'error_type': error_type,
            'category': self._get_error_category(error_type),
            'diagnosis': [],
            'suggested_files': []
        }

        # Signal 1: Error type to files mapping
        file_suggestions = self._suggest_files_by_error(error_type)
        for file_path, score in file_suggestions.items():
            scores[f"file:{file_path}"] = score * 2.0
            metadata['suggested_files'].append(file_path)

        # Signal 2: Stack trace analysis
        if stack_trace:
            stack_files = self._analyze_stack_trace(stack_trace)
            for file_path, score in stack_files.items():
                scores[f"file:{file_path}"] = max(
                    scores.get(f"file:{file_path}", 0), score * 2.5
                )
                if file_path not in metadata['suggested_files']:
                    metadata['suggested_files'].append(file_path)

        # Signal 3: Keyword-based fix suggestions
        fix_suggestions = self._suggest_fixes_by_keywords(error_message)
        for fix_desc, score in fix_suggestions.items():
            scores[f"fix:{fix_desc}"] = score * 1.5

        # Signal 4: Common causes for error type
        causes = self._get_common_causes(error_type)
        for i, cause in enumerate(causes[:3]):
            scores[f"cause:{cause}"] = 1.0 - (i * 0.2)
            metadata['diagnosis'].append(cause)

        # Signal 5: Historical resolutions
        historical = self._find_similar_resolutions(error_message, error_type)
        for resolution, score in historical.items():
            scores[f"resolution:{resolution}"] = score * 1.8

        # Normalize and sort
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {k: v / max_score for k, v in scores.items()}

        sorted_items = sorted(scores.items(), key=lambda x: -x[1])[:top_n]

        return ExpertPrediction(
            expert_id=self.expert_id,
            expert_type=self.expert_type,
            items=sorted_items,
            metadata=metadata
        )

    def _extract_error_type(self, error_message: str, stack_trace: str) -> str:
        """Extract error type from message or stack trace."""
        # Common patterns for error types
        patterns = [
            r'(\w+Error):',
            r'(\w+Exception):',
            r'(\w+Warning):',
            r'raise (\w+)\(',
            r'^(\w+Error)',
            r'^(\w+Exception)',
        ]

        text = f"{error_message}\n{stack_trace}"
        for pattern in patterns:
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                return match.group(1)

        return 'UnknownError'

    def _get_error_category(self, error_type: str) -> str:
        """Get the category for an error type."""
        categories = self.model_data.get('error_categories', self.ERROR_CATEGORIES)
        for category, error_types in categories.items():
            if error_type in error_types:
                return category
        return 'unknown'

    def _suggest_files_by_error(self, error_type: str) -> Dict[str, float]:
        """Suggest files based on error type."""
        suggestions: Dict[str, float] = {}
        error_to_files = self.model_data.get('error_to_files', {})
        total_errors = max(self.model_data.get('total_errors', 1), 1)

        if error_type in error_to_files:
            for file_path, count in error_to_files[error_type].items():
                suggestions[file_path] = count / total_errors

        return suggestions

    def _analyze_stack_trace(self, stack_trace: str) -> Dict[str, float]:
        """Extract relevant files from stack trace."""
        files: Dict[str, float] = {}

        # Pattern to match file paths in stack traces
        patterns = [
            r'File "([^"]+\.py)", line \d+',
            r'at ([^\s]+\.py):\d+',
            r'([a-zA-Z0-9_/]+\.py)',
        ]

        lines = stack_trace.split('\n')
        total_lines = len(lines)

        for i, line in enumerate(lines):
            for pattern in patterns:
                matches = re.findall(pattern, line)
                for match in matches:
                    # Skip standard library and site-packages
                    if 'site-packages' in match or '/usr/lib' in match:
                        continue
                    # Weight by position (later in trace = more relevant)
                    weight = (i + 1) / total_lines
                    files[match] = max(files.get(match, 0), weight)

        return files

    def _suggest_fixes_by_keywords(self, error_message: str) -> Dict[str, float]:
        """Suggest fixes based on keywords in error message."""
        suggestions: Dict[str, float] = {}
        keyword_to_fixes = self.model_data.get('keyword_to_fixes', {})
        message_lower = error_message.lower()

        # Built-in keyword to fix mappings
        builtin_fixes = {
            'import': ['Check if module is installed', 'Verify import path', 'Check for circular imports'],
            'not defined': ['Check variable spelling', 'Ensure variable is initialized', 'Check scope'],
            'no attribute': ['Check attribute spelling', 'Verify object type', 'Check for None'],
            'type': ['Check argument types', 'Add type conversion', 'Validate input'],
            'key': ['Check dictionary keys', 'Use .get() with default', 'Verify key exists'],
            'index': ['Check list/array bounds', 'Verify index range', 'Handle empty collections'],
            'permission': ['Check file permissions', 'Run with elevated privileges', 'Verify path access'],
            'connection': ['Check network connectivity', 'Verify host/port', 'Check firewall'],
            'timeout': ['Increase timeout value', 'Check network latency', 'Retry with backoff'],
            'memory': ['Reduce batch size', 'Free unused resources', 'Use generators'],
            'recursion': ['Add base case', 'Increase recursion limit', 'Convert to iteration'],
        }

        # Check built-in fixes
        for keyword, fixes in builtin_fixes.items():
            if keyword in message_lower:
                for i, fix in enumerate(fixes):
                    suggestions[fix] = 1.0 - (i * 0.2)

        # Check learned fixes
        for keyword, fixes in keyword_to_fixes.items():
            if keyword in message_lower:
                for i, fix in enumerate(fixes[:3]):
                    suggestions[fix] = max(suggestions.get(fix, 0), 0.8 - (i * 0.2))

        return suggestions

    def _get_common_causes(self, error_type: str) -> List[str]:
        """Get common causes for an error type."""
        error_to_causes = self.model_data.get('error_to_causes', {})

        if error_type in error_to_causes:
            return error_to_causes[error_type]

        # Built-in common causes
        builtin_causes = {
            'ImportError': ['Module not installed', 'Incorrect import path', 'Circular import'],
            'ModuleNotFoundError': ['Package not installed', 'Wrong module name', 'Missing __init__.py'],
            'TypeError': ['Wrong argument type', 'Missing required argument', 'Incompatible operation'],
            'AttributeError': ['Misspelled attribute', 'Object is None', 'Wrong object type'],
            'ValueError': ['Invalid value', 'Out of range', 'Wrong format'],
            'KeyError': ['Key not in dictionary', 'Misspelled key', 'Key not initialized'],
            'IndexError': ['Index out of range', 'Empty list', 'Off-by-one error'],
            'FileNotFoundError': ['Wrong file path', 'File deleted', 'Relative path issue'],
            'PermissionError': ['Insufficient permissions', 'File locked', 'Directory not writable'],
            'AssertionError': ['Test condition failed', 'Invalid assumption', 'Contract violation'],
            'RuntimeError': ['Invalid state', 'Concurrent modification', 'Resource exhausted'],
            'RecursionError': ['Missing base case', 'Infinite recursion', 'Stack overflow'],
        }

        return builtin_causes.get(error_type, ['Unknown cause'])

    def _find_similar_resolutions(self, error_message: str, error_type: str) -> Dict[str, float]:
        """Find similar historical resolutions."""
        resolutions: Dict[str, float] = {}
        history = self.model_data.get('resolution_history', [])

        message_words = set(error_message.lower().split())

        for record in history[-100:]:  # Check last 100 resolutions
            if record.get('error_type') == error_type:
                resolution = record.get('resolution', '')
                if resolution:
                    # Calculate similarity
                    record_words = set(record.get('error_message', '').lower().split())
                    if record_words:
                        similarity = len(message_words & record_words) / len(message_words | record_words)
                        if similarity > 0.3:
                            resolutions[resolution] = max(resolutions.get(resolution, 0), similarity)

        return resolutions

    def train(self, error_records: List[Dict[str, Any]]) -> None:
        """
        Train the expert on error history.

        Args:
            error_records: List of error dictionaries with:
                - error_type (str): Type of error
                - error_message (str): Error message
                - stack_trace (str, optional): Stack trace
                - files_modified (List[str]): Files that were modified to fix
                - resolution (str, optional): Description of the fix
        """
        error_to_files: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        error_to_causes: Dict[str, List[str]] = defaultdict(list)
        keyword_to_fixes: Dict[str, List[str]] = defaultdict(list)
        stack_patterns: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        resolution_history: List[Dict] = []

        for record in error_records:
            error_type = record.get('error_type', 'Unknown')
            error_message = record.get('error_message', '')
            files_modified = record.get('files_modified', [])
            resolution = record.get('resolution', '')
            stack_trace = record.get('stack_trace', '')

            # Build error to files mapping
            for file_path in files_modified:
                error_to_files[error_type][file_path] += 1

            # Extract stack patterns
            if stack_trace:
                stack_files = self._analyze_stack_trace(stack_trace)
                for file_path in stack_files:
                    stack_patterns[error_type][file_path] += 1

            # Store resolution
            if resolution:
                resolution_history.append({
                    'error_type': error_type,
                    'error_message': error_message[:200],  # Truncate
                    'resolution': resolution,
                    'files': files_modified[:5]  # Limit files
                })

                # Extract keywords for fix mapping
                keywords = set(re.findall(r'\b\w+\b', error_message.lower()))
                for keyword in keywords:
                    if len(keyword) > 3:  # Skip short words
                        keyword_to_fixes[keyword].append(resolution)

        # Store in model_data
        self.model_data = {
            'error_to_files': {k: dict(v) for k, v in error_to_files.items()},
            'error_to_causes': dict(error_to_causes),
            'keyword_to_fixes': {k: list(set(v))[:5] for k, v in keyword_to_fixes.items()},
            'stack_patterns': {k: dict(v) for k, v in stack_patterns.items()},
            'error_categories': self.ERROR_CATEGORIES.copy(),
            'resolution_history': resolution_history[-500:],  # Keep last 500
            'total_errors': len(error_records)
        }

    def diagnose(self, error_message: str, stack_trace: str = "") -> Dict[str, Any]:
        """
        Convenience method for quick diagnosis.

        Args:
            error_message: The error message
            stack_trace: Optional stack trace

        Returns:
            Dictionary with diagnosis results
        """
        prediction = self.predict({
            'error_message': error_message,
            'stack_trace': stack_trace
        })

        # Parse prediction items into categories
        files = []
        fixes = []
        causes = []

        for item, score in prediction.items:
            if item.startswith('file:'):
                files.append((item[5:], score))
            elif item.startswith('fix:'):
                fixes.append((item[4:], score))
            elif item.startswith('cause:'):
                causes.append((item[6:], score))

        return {
            'error_type': prediction.metadata.get('error_type'),
            'category': prediction.metadata.get('category'),
            'likely_causes': causes,
            'suggested_fixes': fixes,
            'files_to_check': files,
            'confidence': prediction.items[0][1] if prediction.items else 0.0
        }


# Convenience function
def diagnose_error(error_message: str, stack_trace: str = "", model_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Diagnose an error and suggest fixes.

    Args:
        error_message: The error message
        stack_trace: Optional stack trace
        model_path: Optional path to saved ErrorDiagnosisExpert model

    Returns:
        Dictionary with diagnosis results
    """
    if model_path and model_path.exists():
        expert = ErrorDiagnosisExpert.load(model_path)
    else:
        expert = ErrorDiagnosisExpert()

    return expert.diagnose(error_message, stack_trace)


if __name__ == '__main__':
    # Demo usage
    expert = ErrorDiagnosisExpert()

    # Test diagnosis
    result = expert.diagnose(
        error_message="TypeError: 'NoneType' object is not subscriptable",
        stack_trace="""
Traceback (most recent call last):
  File "cortical/query/search.py", line 42, in find_documents
    return results[0]
TypeError: 'NoneType' object is not subscriptable
"""
    )

    print("Diagnosis:")
    print(f"  Error Type: {result['error_type']}")
    print(f"  Category: {result['category']}")
    print(f"  Likely Causes:")
    for cause, score in result['likely_causes']:
        print(f"    - {cause} ({score:.2f})")
    print(f"  Suggested Fixes:")
    for fix, score in result['suggested_fixes']:
        print(f"    - {fix} ({score:.2f})")
    print(f"  Files to Check:")
    for file, score in result['files_to_check']:
        print(f"    - {file} ({score:.2f})")
