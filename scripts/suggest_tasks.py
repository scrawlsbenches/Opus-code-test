#!/usr/bin/env python3
"""
Analyzes code changes and suggests follow-up tasks automatically.

This script examines git changes (staged and/or unstaged) and applies
pattern-based rules to detect when follow-up work is needed:
- Missing docstrings on new methods
- New tests that should be run
- Performance-sensitive code changes requiring profiling
- Validation logic changes requiring test verification
- Documentation updates needed

Usage:
    python scripts/suggest_tasks.py              # Analyze staged changes
    python scripts/suggest_tasks.py --all        # Include unstaged
    python scripts/suggest_tasks.py --json       # JSON output
    python scripts/suggest_tasks.py --create     # Create tasks in tasks/*.json

Examples:
    # Show suggestions for staged changes
    python scripts/suggest_tasks.py

    # Include unstaged changes and show detailed output
    python scripts/suggest_tasks.py --all --verbose

    # Create tasks from suggestions
    python scripts/suggest_tasks.py --create
"""

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

# Import task utilities if available
try:
    from task_utils import TaskSession
    TASK_UTILS_AVAILABLE = True
except ImportError:
    TASK_UTILS_AVAILABLE = False


class TaskSuggester:
    """Analyzes code changes and suggests follow-up tasks."""

    def __init__(self, repo_path: str = '.'):
        """
        Initialize task suggester.

        Args:
            repo_path: Path to git repository (default: current directory)
        """
        self.repo_path = Path(repo_path)
        self._is_git_repo = self._check_git_repo()

    def _check_git_repo(self) -> bool:
        """Check if the current directory is a git repository."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--git-dir'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def get_staged_changes(self) -> Dict[str, List[str]]:
        """
        Get staged file changes.

        Returns:
            Dict with keys 'added', 'modified', 'deleted' containing file paths

        Raises:
            RuntimeError: If not in a git repository
        """
        if not self._is_git_repo:
            raise RuntimeError("Not in a git repository")

        try:
            result = subprocess.run(
                ['git', 'diff', '--cached', '--name-status'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                return {'added': [], 'modified': [], 'deleted': []}

            return self._parse_git_status(result.stdout)

        except subprocess.SubprocessError as e:
            raise RuntimeError(f"Failed to get staged changes: {e}")

    def get_unstaged_changes(self) -> Dict[str, List[str]]:
        """
        Get unstaged file changes.

        Returns:
            Dict with keys 'added', 'modified', 'deleted' containing file paths

        Raises:
            RuntimeError: If not in a git repository
        """
        if not self._is_git_repo:
            raise RuntimeError("Not in a git repository")

        try:
            result = subprocess.run(
                ['git', 'diff', '--name-status'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                return {'added': [], 'modified': [], 'deleted': []}

            return self._parse_git_status(result.stdout)

        except subprocess.SubprocessError as e:
            raise RuntimeError(f"Failed to get unstaged changes: {e}")

    def _parse_git_status(self, output: str) -> Dict[str, List[str]]:
        """
        Parse git status output.

        Args:
            output: Output from git diff --name-status

        Returns:
            Dict with 'added', 'modified', 'deleted' lists
        """
        changes = {'added': [], 'modified': [], 'deleted': []}

        for line in output.strip().split('\n'):
            if not line:
                continue

            parts = line.split('\t', 1)
            if len(parts) != 2:
                continue

            status, filepath = parts
            status = status.strip()

            if status.startswith('A'):
                changes['added'].append(filepath)
            elif status.startswith('M'):
                changes['modified'].append(filepath)
            elif status.startswith('D'):
                changes['deleted'].append(filepath)

        return changes

    def analyze_file(self, filepath: str) -> List[Dict]:
        """
        Analyze a single file and return suggested tasks.

        Args:
            filepath: Path to file to analyze

        Returns:
            List of suggestion dicts with keys:
                - title: Task title
                - priority: high, medium, low
                - category: Task category (docs, test, perf, etc.)
                - reason: Why this task is suggested
                - files: Files related to this task
        """
        suggestions = []

        # Skip deleted files
        full_path = self.repo_path / filepath
        if not full_path.exists():
            return suggestions

        # Only analyze Python files for now
        if not filepath.endswith('.py'):
            return suggestions

        try:
            content = full_path.read_text(encoding='utf-8')
        except (IOError, UnicodeDecodeError):
            return suggestions

        # Rule 1: New public methods without docstrings
        if self._has_undocumented_methods(content):
            suggestions.append({
                'title': f'Add docstrings to {filepath}',
                'priority': 'low',
                'category': 'docs',
                'reason': 'New methods detected without docstrings',
                'files': [filepath]
            })

        # Rule 2: New test file
        if self._is_new_test_file(filepath, content):
            suggestions.append({
                'title': f'Run tests in {filepath}',
                'priority': 'high',
                'category': 'test',
                'reason': 'New test file added - should verify tests pass',
                'files': [filepath]
            })

        # Rule 3: Performance-sensitive code changes
        if self._has_performance_code(filepath, content):
            suggestions.append({
                'title': f'Profile performance of {filepath}',
                'priority': 'medium',
                'category': 'perf',
                'reason': 'Loop or performance-sensitive code detected',
                'files': [filepath]
            })

        # Rule 4: Validation logic changes
        if self._has_validation_logic(content):
            suggestions.append({
                'title': f'Verify validation tests for {filepath}',
                'priority': 'medium',
                'category': 'test',
                'reason': 'Validation logic detected - ensure tests exist',
                'files': [filepath]
            })

        # Rule 5: Missing type hints
        if self._missing_type_hints(content):
            suggestions.append({
                'title': f'Add type hints to {filepath}',
                'priority': 'low',
                'category': 'codequal',
                'reason': 'Functions without type hints detected',
                'files': [filepath]
            })

        # Rule 6: Configuration changes
        if self._is_config_change(filepath):
            suggestions.append({
                'title': f'Update documentation for config changes in {filepath}',
                'priority': 'medium',
                'category': 'docs',
                'reason': 'Configuration file modified',
                'files': [filepath]
            })

        # Rule 7: New public API methods
        if self._has_new_public_api(filepath, content):
            suggestions.append({
                'title': f'Document new public API in {filepath}',
                'priority': 'high',
                'category': 'docs',
                'reason': 'New public API methods detected',
                'files': [filepath]
            })

        return suggestions

    def _has_undocumented_methods(self, content: str) -> bool:
        """Check if file has methods without docstrings."""
        # Look for function definitions
        func_pattern = re.compile(r'^\s*(def|async def)\s+(\w+)\s*\(', re.MULTILINE)
        docstring_pattern = re.compile(r'^\s*"""', re.MULTILINE)

        functions = func_pattern.findall(content)
        docstrings = docstring_pattern.findall(content)

        # Simple heuristic: if we have functions but fewer docstrings
        return len(functions) > 0 and len(docstrings) < len(functions) * 0.5

    def _is_new_test_file(self, filepath: str, content: str) -> bool:
        """Check if this is a new test file."""
        return 'test_' in filepath and 'def test_' in content

    def _has_performance_code(self, filepath: str, content: str) -> bool:
        """Check if file contains performance-sensitive code."""
        # Only check core library files
        if not filepath.startswith('cortical/'):
            return False

        # Look for loops and performance indicators
        perf_indicators = [
            r'\bfor\b.*\bin\b',  # for loops
            r'\bwhile\b',  # while loops
            r'O\(n',  # Big-O notation
            r'\bslow\b',  # Comments about slow code
            r'\bperformance\b',  # Performance mentions
        ]

        return any(re.search(pattern, content, re.IGNORECASE) for pattern in perf_indicators)

    def _has_validation_logic(self, content: str) -> bool:
        """Check if file contains validation logic."""
        validation_patterns = [
            r'\braise ValueError\b',
            r'\braise TypeError\b',
            r'\bassert\b',
            r'\bif.*not\b.*:.*raise\b',
        ]

        return any(re.search(pattern, content) for pattern in validation_patterns)

    def _missing_type_hints(self, content: str) -> bool:
        """Check if file has functions missing type hints."""
        # Look for function definitions without -> or : annotations
        func_pattern = re.compile(r'^\s*def\s+(\w+)\s*\([^)]*\)\s*:', re.MULTILINE)
        typed_pattern = re.compile(r'^\s*def\s+(\w+)\s*\([^)]*:\s*\w+', re.MULTILINE)

        all_funcs = func_pattern.findall(content)
        typed_funcs = typed_pattern.findall(content)

        # If we have functions and less than 50% are typed
        return len(all_funcs) > 0 and len(typed_funcs) < len(all_funcs) * 0.5

    def _is_config_change(self, filepath: str) -> bool:
        """Check if this is a configuration file."""
        config_files = [
            'config.py',
            'settings.py',
            '.yml',
            '.yaml',
            '.json',
            '.toml',
            'setup.py',
            'pyproject.toml',
        ]
        return any(filepath.endswith(pattern) for pattern in config_files)

    def _has_new_public_api(self, filepath: str, content: str) -> bool:
        """Check if file has new public API methods."""
        # Only check main library files
        if not filepath.startswith('cortical/'):
            return False

        # Look for public class methods (not starting with _)
        public_method_pattern = re.compile(
            r'^\s*def\s+([a-z][a-z0-9_]*)\s*\(self',
            re.MULTILINE
        )
        matches = public_method_pattern.findall(content)

        # If we find public methods, suggest documentation
        return len(matches) > 0

    def suggest_from_changes(self, include_unstaged: bool = False) -> List[Dict]:
        """
        Analyze all changes and return deduplicated suggestions.

        Args:
            include_unstaged: If True, analyze both staged and unstaged changes

        Returns:
            List of unique suggestion dicts

        Raises:
            RuntimeError: If not in a git repository
        """
        all_suggestions = []

        # Get changes
        changes = self.get_staged_changes()
        if include_unstaged:
            unstaged = self.get_unstaged_changes()
            for key in changes:
                changes[key].extend(unstaged.get(key, []))

        # Analyze each file
        for filepath in changes.get('added', []) + changes.get('modified', []):
            all_suggestions.extend(self.analyze_file(filepath))

        # Deduplicate by title
        seen_titles = set()
        unique = []
        for suggestion in all_suggestions:
            if suggestion['title'] not in seen_titles:
                seen_titles.add(suggestion['title'])
                unique.append(suggestion)

        # Sort by priority (high > medium > low)
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        unique.sort(key=lambda s: priority_order.get(s['priority'], 1))

        # Limit to max 10 suggestions to avoid overwhelming
        return unique[:10]

    def format_suggestions(self, suggestions: List[Dict]) -> str:
        """
        Format suggestions as markdown.

        Args:
            suggestions: List of suggestion dicts

        Returns:
            Markdown-formatted string
        """
        if not suggestions:
            return "No suggestions at this time.\n"

        lines = [
            "# Suggested Follow-up Tasks",
            "",
            f"**Found {len(suggestions)} suggestion(s)**",
            "",
        ]

        # Group by category
        by_category = defaultdict(list)
        for suggestion in suggestions:
            by_category[suggestion['category']].append(suggestion)

        category_icons = {
            'docs': '📚',
            'test': '🧪',
            'perf': '⚡',
            'codequal': '✨',
        }

        for category, items in sorted(by_category.items()):
            icon = category_icons.get(category, '📋')
            lines.append(f"## {icon} {category.title()}")
            lines.append("")

            for item in items:
                priority_badge = {
                    'high': '🔴 HIGH',
                    'medium': '🟡 MEDIUM',
                    'low': '🟢 LOW'
                }.get(item['priority'], item['priority'])

                lines.append(f"### {item['title']}")
                lines.append(f"**Priority:** {priority_badge}")
                lines.append(f"**Reason:** {item['reason']}")
                lines.append(f"**Files:** {', '.join(item['files'])}")
                lines.append("")

        return '\n'.join(lines)

    def format_json(self, suggestions: List[Dict]) -> str:
        """
        Format suggestions as JSON.

        Args:
            suggestions: List of suggestion dicts

        Returns:
            JSON-formatted string
        """
        return json.dumps(suggestions, indent=2)

    def create_tasks(self, suggestions: List[Dict], tasks_dir: str = 'tasks') -> int:
        """
        Create tasks from suggestions using TaskSession.

        Args:
            suggestions: List of suggestion dicts
            tasks_dir: Directory for task files

        Returns:
            Number of tasks created

        Raises:
            ImportError: If task_utils is not available
        """
        if not TASK_UTILS_AVAILABLE:
            raise ImportError("task_utils module not available")

        if not suggestions:
            return 0

        session = TaskSession(tasks_dir=tasks_dir)

        for suggestion in suggestions:
            session.create_task(
                title=suggestion['title'],
                priority=suggestion['priority'],
                category=suggestion['category'],
                description=suggestion['reason'],
                context={'files': suggestion['files']}
            )

        session.save()
        return len(suggestions)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze code changes and suggest follow-up tasks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Analyze staged changes
  %(prog)s --all              # Include unstaged changes
  %(prog)s --json             # Output as JSON
  %(prog)s --create           # Create tasks from suggestions
  %(prog)s --all --verbose    # Detailed output with unstaged changes
        """
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Include unstaged changes (default: staged only)'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON instead of markdown'
    )
    parser.add_argument(
        '--create',
        action='store_true',
        help='Create tasks in tasks/ directory'
    )
    parser.add_argument(
        '--tasks-dir',
        default='tasks',
        help='Directory for task files (default: tasks/)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # Create suggester
    try:
        suggester = TaskSuggester()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Get suggestions
    try:
        suggestions = suggester.suggest_from_changes(include_unstaged=args.all)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Output
    if args.json:
        print(suggester.format_json(suggestions))
    else:
        print(suggester.format_suggestions(suggestions))

    # Create tasks if requested
    if args.create:
        if not TASK_UTILS_AVAILABLE:
            print("Error: task_utils module not available", file=sys.stderr)
            return 1

        try:
            count = suggester.create_tasks(suggestions, args.tasks_dir)
            print(f"\n✅ Created {count} task(s) in {args.tasks_dir}/")
        except Exception as e:
            print(f"Error creating tasks: {e}", file=sys.stderr)
            return 1

    # Verbose summary
    if args.verbose:
        print(f"\n--- Summary ---")
        print(f"Analyzed: {'staged + unstaged' if args.all else 'staged only'}")
        print(f"Suggestions: {len(suggestions)}")
        if suggestions:
            by_priority = defaultdict(int)
            for s in suggestions:
                by_priority[s['priority']] += 1
            for priority in ['high', 'medium', 'low']:
                if by_priority[priority] > 0:
                    print(f"  {priority}: {by_priority[priority]}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
