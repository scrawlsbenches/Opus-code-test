#!/usr/bin/env python3
"""
Validate tasks/ directory for the merge-friendly task system.

Usage:
    python scripts/validate_tasks.py           # Validate all task files
    python scripts/validate_tasks.py --strict  # Fail on warnings too
    python scripts/validate_tasks.py --json    # JSON output for CI

Checks for:
- Valid JSON structure in all task files
- Valid task IDs (T-YYYYMMDD-HHMMSS-XXXX-NNN or LEGACY-NNN format)
- Valid task statuses (pending, in_progress, completed, deferred)
- Required fields (id, title, status, priority)
- Orphaned task dependencies
- Duplicate task IDs across sessions
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


# Valid patterns
# Task IDs: T-YYYYMMDD-HHMMSS-XXXX-NN (e.g., T-20251213-223234-e233-01)
# Legacy IDs: LEGACY-NNN (e.g., LEGACY-001)
TASK_ID_PATTERN = re.compile(
    r'^(T-\d{8}-\d{6}-[a-f0-9]{4}-\d{2,3}|LEGACY-\d{1,3})$'
)
VALID_STATUSES = {'pending', 'in_progress', 'completed', 'deferred'}
VALID_PRIORITIES = {'critical', 'high', 'medium', 'low'}
REQUIRED_TASK_FIELDS = {'id', 'title', 'status', 'priority'}


@dataclass
class ValidationResult:
    """Result of validating a task file."""

    file_path: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    task_count: int = 0
    tasks_by_status: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    tasks_by_priority: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    task_ids: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0


def validate_task(task: dict, file_path: str) -> tuple[list[str], list[str]]:
    """
    Validate a single task dictionary.

    Returns (errors, warnings).
    """
    errors = []
    warnings = []

    # Check required fields
    for field_name in REQUIRED_TASK_FIELDS:
        if field_name not in task:
            errors.append(f"Missing required field '{field_name}'")

    # Validate task ID format
    task_id = task.get('id', '')
    if task_id and not TASK_ID_PATTERN.match(task_id):
        errors.append(f"Invalid task ID format: '{task_id}' (expected T-YYYYMMDD-HHMMSS-XXXX-NN or LEGACY-NNN)")

    # Validate status
    status = task.get('status', '')
    if status and status not in VALID_STATUSES:
        errors.append(f"Invalid status '{status}' for task {task_id} (valid: {', '.join(VALID_STATUSES)})")

    # Validate priority
    priority = task.get('priority', '')
    if priority and priority not in VALID_PRIORITIES:
        warnings.append(f"Non-standard priority '{priority}' for task {task_id} (standard: {', '.join(VALID_PRIORITIES)})")

    # Check for empty title
    title = task.get('title', '')
    if not title or not title.strip():
        errors.append(f"Empty title for task {task_id}")

    # Check depends_on references exist (will be validated later)
    depends_on = task.get('depends_on', [])
    if depends_on and not isinstance(depends_on, list):
        errors.append(f"depends_on must be a list for task {task_id}")

    # Warn about in_progress tasks without timestamps
    if status == 'in_progress' and not task.get('updated_at'):
        warnings.append(f"Task {task_id} is in_progress but has no updated_at timestamp")

    # Warn about completed tasks without completed_at
    if status == 'completed' and not task.get('completed_at'):
        warnings.append(f"Task {task_id} is completed but has no completed_at timestamp")

    return errors, warnings


def validate_task_file(file_path: Path) -> ValidationResult:
    """
    Validate a single task JSON file.

    Returns ValidationResult with all findings.
    """
    result = ValidationResult(file_path=str(file_path))

    # Try to load JSON
    try:
        with open(file_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        result.errors.append(f"Invalid JSON: {e}")
        return result
    except Exception as e:
        result.errors.append(f"Failed to read file: {e}")
        return result

    # Check for required top-level fields
    if 'tasks' not in data:
        result.errors.append("Missing 'tasks' array in file")
        return result

    if not isinstance(data['tasks'], list):
        result.errors.append("'tasks' must be an array")
        return result

    # Check session metadata
    if 'session_id' not in data:
        result.warnings.append("Missing 'session_id' field")

    if 'version' not in data:
        result.warnings.append("Missing 'version' field (schema versioning)")

    # Validate each task
    for i, task in enumerate(data['tasks']):
        if not isinstance(task, dict):
            result.errors.append(f"Task at index {i} is not an object")
            continue

        errors, warnings = validate_task(task, str(file_path))
        result.errors.extend(errors)
        result.warnings.extend(warnings)

        # Collect statistics
        task_id = task.get('id', f'unknown-{i}')
        result.task_ids.append(task_id)
        result.task_count += 1

        status = task.get('status', 'unknown')
        priority = task.get('priority', 'unknown')
        result.tasks_by_status[status] += 1
        result.tasks_by_priority[priority] += 1

    return result


def validate_dependencies(all_task_ids: set[str], all_tasks: list[dict]) -> list[str]:
    """
    Check for orphaned dependencies across all tasks.

    Returns list of error messages.
    """
    errors = []

    for task in all_tasks:
        task_id = task.get('id', 'unknown')
        depends_on = task.get('depends_on', [])

        if not isinstance(depends_on, list):
            continue

        for dep_id in depends_on:
            if dep_id not in all_task_ids:
                errors.append(f"Task {task_id} depends on non-existent task {dep_id}")

    return errors


def find_duplicate_ids(results: list[ValidationResult]) -> list[str]:
    """
    Find task IDs that appear in multiple files.

    Returns list of error messages.
    """
    errors = []
    id_to_files: dict[str, list[str]] = defaultdict(list)

    for result in results:
        for task_id in result.task_ids:
            id_to_files[task_id].append(result.file_path)

    for task_id, files in id_to_files.items():
        if len(files) > 1:
            errors.append(f"Duplicate task ID '{task_id}' found in: {', '.join(files)}")

    return errors


def aggregate_statistics(results: list[ValidationResult]) -> dict[str, Any]:
    """
    Aggregate statistics across all validated files.
    """
    total_tasks = 0
    by_status: dict[str, int] = defaultdict(int)
    by_priority: dict[str, int] = defaultdict(int)

    for result in results:
        total_tasks += result.task_count
        for status, count in result.tasks_by_status.items():
            by_status[status] += count
        for priority, count in result.tasks_by_priority.items():
            by_priority[priority] += count

    return {
        'total_tasks': total_tasks,
        'by_status': dict(by_status),
        'by_priority': dict(by_priority),
        'files_validated': len(results),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Validate merge-friendly task files in tasks/ directory'
    )
    parser.add_argument(
        '--strict', action='store_true',
        help='Fail on warnings too (not just errors)'
    )
    parser.add_argument(
        '--json', action='store_true',
        help='Output results as JSON'
    )
    parser.add_argument(
        '--tasks-dir', default='tasks',
        help='Path to tasks directory (default: tasks)'
    )
    args = parser.parse_args()

    # Find tasks directory
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    tasks_dir = repo_root / args.tasks_dir

    if not tasks_dir.exists():
        if args.json:
            print(json.dumps({'error': f'Tasks directory not found: {tasks_dir}'}))
        else:
            print(f"❌ Tasks directory not found: {tasks_dir}")
        return 1

    # Find all JSON files
    task_files = list(tasks_dir.glob('*.json'))

    if not task_files:
        if args.json:
            print(json.dumps({'warning': 'No task files found', 'files': 0}))
        else:
            print("⚠️  No task files found in tasks/ directory")
        return 0

    # Validate each file
    results: list[ValidationResult] = []
    all_tasks: list[dict] = []
    all_task_ids: set[str] = set()

    for task_file in sorted(task_files):
        result = validate_task_file(task_file)
        results.append(result)

        # Collect all tasks for cross-file validation
        try:
            with open(task_file) as f:
                data = json.load(f)
                tasks = data.get('tasks', [])
                all_tasks.extend(tasks)
                for task in tasks:
                    if 'id' in task:
                        all_task_ids.add(task['id'])
        except Exception:
            pass  # Already reported in individual validation

    # Cross-file validations
    cross_file_errors = []
    cross_file_errors.extend(find_duplicate_ids(results))
    cross_file_errors.extend(validate_dependencies(all_task_ids, all_tasks))

    # Aggregate statistics
    stats = aggregate_statistics(results)

    # Count totals
    total_errors = sum(len(r.errors) for r in results) + len(cross_file_errors)
    total_warnings = sum(len(r.warnings) for r in results)

    # JSON output
    if args.json:
        output = {
            'valid': total_errors == 0,
            'files_validated': len(results),
            'total_errors': total_errors,
            'total_warnings': total_warnings,
            'statistics': stats,
            'errors': [],
            'warnings': [],
        }

        for result in results:
            for error in result.errors:
                output['errors'].append({'file': result.file_path, 'message': error})
            for warning in result.warnings:
                output['warnings'].append({'file': result.file_path, 'message': warning})

        for error in cross_file_errors:
            output['errors'].append({'file': 'cross-file', 'message': error})

        print(json.dumps(output, indent=2))

        if total_errors > 0:
            return 1
        if args.strict and total_warnings > 0:
            return 1
        return 0

    # Human-readable output
    print("🔍 Task Validation Results\n")

    # Print errors and warnings per file
    has_issues = False
    for result in results:
        if result.errors or result.warnings:
            has_issues = True
            file_name = Path(result.file_path).name
            print(f"📄 {file_name}:")
            for error in result.errors:
                print(f"   ❌ {error}")
            for warning in result.warnings:
                print(f"   ⚠️  {warning}")
            print()

    # Print cross-file errors
    if cross_file_errors:
        has_issues = True
        print("📁 Cross-file issues:")
        for error in cross_file_errors:
            print(f"   ❌ {error}")
        print()

    # Print statistics
    print("📊 Statistics:")
    print(f"   Files validated: {stats['files_validated']}")
    print(f"   Total tasks: {stats['total_tasks']}")

    if stats['by_status']:
        status_str = ", ".join(f"{k}: {v}" for k, v in sorted(stats['by_status'].items()))
        print(f"   By status: {status_str}")

    if stats['by_priority']:
        priority_str = ", ".join(f"{k}: {v}" for k, v in sorted(stats['by_priority'].items()))
        print(f"   By priority: {priority_str}")

    print()

    # Final result
    if total_errors > 0:
        print(f"❌ Validation failed: {total_errors} error(s), {total_warnings} warning(s)")
        return 1
    elif args.strict and total_warnings > 0:
        print(f"❌ Validation failed (strict mode): {total_warnings} warning(s)")
        return 1
    elif total_warnings > 0:
        print(f"✅ Validation passed with {total_warnings} warning(s)")
        return 0
    else:
        print("✅ All task files are valid!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
