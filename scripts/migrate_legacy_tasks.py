#!/usr/bin/env python3
"""
Migrate legacy TASK_LIST.md tasks to the new merge-friendly task system.

This script:
1. Parses TASK_LIST.md and TASK_ARCHIVE.md
2. Creates a legacy migration file in tasks/
3. Corrects task statuses based on actual codebase state
4. Preserves historical context

Usage:
    python scripts/migrate_legacy_tasks.py
    python scripts/migrate_legacy_tasks.py --dry-run
"""

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.task_utils import generate_task_id

# Status corrections based on actual codebase state (verified 2025-12-14)
STATUS_CORRECTIONS = {
    # Listed as pending but actually complete
    206: ("completed", "2025-12-13", "cortical/state_storage.py exists - JSON state storage implemented"),
    134: ("completed", "2025-12-13", "cortical/proto/ directory with schema.proto and serialization.py"),
    184: ("completed", "2025-12-13", "cortical/mcp_server.py exists - MCP Server implemented"),

    # Partial completion
    133: ("pending", None, "Checkpointing added to compute_all() but full WAL not implemented"),

    # Truly pending
    135: ("pending", None, "Parallel processing not implemented - processor.py still sequential"),
    95: ("pending", None, "processor.py still 3115 lines - not split into modules"),
}

def parse_task_table(content: str, section_pattern: str) -> list:
    """Parse a markdown table section and extract tasks."""
    tasks = []

    # Find the section
    section_match = re.search(section_pattern, content, re.MULTILINE | re.DOTALL)
    if not section_match:
        return tasks

    section = section_match.group(0)

    # Parse table rows - handle both 3-column and 5-column formats
    # 5-column: | # | Task | Category | Depends | Effort |
    # 3-column: | # | Task | Category |
    for line in section.split('\n'):
        # Skip header, separator, and empty lines
        if not line.strip() or '|---' in line or '# |' in line or 'Task |' in line:
            continue

        parts = [p.strip() for p in line.split('|') if p.strip()]
        if len(parts) >= 3 and parts[0].isdigit():
            task_num = int(parts[0])
            title = parts[1]
            category = parts[2].lower().replace(' ', '-')
            tasks.append({
                'legacy_id': task_num,
                'title': title,
                'category': category,
            })

    return tasks

def parse_archive_tasks(archive_path: Path) -> dict:
    """Parse TASK_ARCHIVE.md for completed tasks with dates."""
    completed = {}

    if not archive_path.exists():
        return completed

    content = archive_path.read_text()

    # Parse the quick reference table
    table_pattern = r'\|\s*(\d+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*(\d{4}-\d{2}-\d{2})\s*\|'
    for match in re.finditer(table_pattern, content):
        task_num = int(match.group(1))
        title = match.group(2).strip()
        category = match.group(3).strip()
        completed_date = match.group(4).strip()
        completed[task_num] = {
            'title': title,
            'category': category.lower().replace(' ', '-'),
            'completed_date': completed_date,
        }

    return completed

def migrate_tasks(dry_run: bool = False) -> dict:
    """Migrate legacy tasks to new format."""

    project_root = Path(__file__).parent.parent
    task_list_path = project_root / "TASK_LIST.md"
    archive_path = project_root / "TASK_ARCHIVE.md"
    tasks_dir = project_root / "tasks"

    if not task_list_path.exists():
        print("ERROR: TASK_LIST.md not found")
        return {}

    content = task_list_path.read_text()

    # Parse archived (completed) tasks
    archived_tasks = parse_archive_tasks(archive_path)
    print(f"Found {len(archived_tasks)} completed tasks in archive")

    # Parse pending tasks from different sections
    # Use \Z for end-of-string instead of $ (which matches end-of-line in MULTILINE)
    high_priority = parse_task_table(content, r'### 🟠 High.*?(?=\n###|\Z)')
    medium_priority = parse_task_table(content, r'### 🟡 Medium.*?(?=\n###|\Z)')
    low_priority = parse_task_table(content, r'### 🟢 Low.*?(?=\n###|\Z)')
    future_tasks = parse_task_table(content, r'### 🔮 Future.*?(?=\n###|\Z)')

    # Parse deferred tasks (different format)
    deferred_pattern = r'\|\s*(\d+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|'
    deferred_section = re.search(r'### ⏸️ Deferred.*?(?=\n###|\Z)', content, re.DOTALL)
    deferred_tasks = []
    if deferred_section:
        for match in re.finditer(deferred_pattern, deferred_section.group(0)):
            deferred_tasks.append({
                'legacy_id': int(match.group(1)),
                'title': match.group(2).strip(),
                'category': 'deferred',
                'reason': match.group(3).strip(),
            })

    # Build migration data
    migration_tasks = []
    now = datetime.now().isoformat()

    # Add archived (completed) tasks
    for task_num, task_data in sorted(archived_tasks.items()):
        migration_tasks.append({
            'id': f"LEGACY-{task_num:03d}",
            'title': task_data['title'],
            'status': 'completed',
            'priority': 'medium',
            'category': task_data['category'],
            'description': f"Migrated from legacy TASK_LIST.md task #{task_num}",
            'depends_on': [],
            'effort': 'unknown',
            'created_at': '2025-12-10T00:00:00',  # Approximate start
            'updated_at': now,
            'completed_at': f"{task_data['completed_date']}T00:00:00",
            'context': {'legacy_task_number': task_num},
            'retrospective': None,
        })

    # Add pending tasks with status corrections
    def add_pending_tasks(tasks: list, priority: str, status: str = 'pending'):
        for task in tasks:
            task_num = task['legacy_id']

            # Apply status corrections if known
            if task_num in STATUS_CORRECTIONS:
                actual_status, completed_date, note = STATUS_CORRECTIONS[task_num]
                task_status = actual_status
                task_completed = f"{completed_date}T00:00:00" if completed_date else None
                description = f"{task.get('reason', '')} | Correction: {note}".strip(' |')
            else:
                task_status = status
                task_completed = None
                description = task.get('reason', f"Migrated from legacy TASK_LIST.md task #{task_num}")

            migration_tasks.append({
                'id': f"LEGACY-{task_num:03d}",
                'title': task['title'],
                'status': task_status,
                'priority': priority,
                'category': task['category'],
                'description': description,
                'depends_on': [],
                'effort': 'unknown',
                'created_at': '2025-12-10T00:00:00',
                'updated_at': now,
                'completed_at': task_completed,
                'context': {'legacy_task_number': task_num},
                'retrospective': None,
            })

    add_pending_tasks(high_priority, 'high')
    add_pending_tasks(medium_priority, 'medium')
    add_pending_tasks(low_priority, 'low')
    add_pending_tasks(future_tasks, 'low', 'pending')
    add_pending_tasks(deferred_tasks, 'low', 'deferred')

    # Sort by legacy task number
    migration_tasks.sort(key=lambda t: int(t['id'].split('-')[1]))

    # Create migration file
    migration_data = {
        'version': 1,
        'session_id': 'legacy-migration',
        'started_at': now,
        'saved_at': now,
        'migration_info': {
            'source': 'TASK_LIST.md + TASK_ARCHIVE.md',
            'migrated_at': now,
            'total_tasks': len(migration_tasks),
            'completed': sum(1 for t in migration_tasks if t['status'] == 'completed'),
            'pending': sum(1 for t in migration_tasks if t['status'] == 'pending'),
            'deferred': sum(1 for t in migration_tasks if t['status'] == 'deferred'),
            'status_corrections_applied': list(STATUS_CORRECTIONS.keys()),
        },
        'tasks': migration_tasks,
    }

    # Summary
    print(f"\n=== Migration Summary ===")
    print(f"Total tasks: {len(migration_tasks)}")
    print(f"  Completed: {migration_data['migration_info']['completed']}")
    print(f"  Pending: {migration_data['migration_info']['pending']}")
    print(f"  Deferred: {migration_data['migration_info']['deferred']}")
    print(f"\nStatus corrections applied to tasks: {STATUS_CORRECTIONS.keys()}")

    if dry_run:
        print("\n[DRY RUN] Would create: tasks/legacy_migration.json")
        print(json.dumps(migration_data, indent=2)[:2000] + "...\n")
    else:
        tasks_dir.mkdir(exist_ok=True)
        output_path = tasks_dir / "legacy_migration.json"
        with open(output_path, 'w') as f:
            json.dump(migration_data, f, indent=2)
        print(f"\nCreated: {output_path}")

    return migration_data

if __name__ == '__main__':
    dry_run = '--dry-run' in sys.argv
    migrate_tasks(dry_run=dry_run)
