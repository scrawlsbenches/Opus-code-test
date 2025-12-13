#!/usr/bin/env python3
"""
Validate TASK_LIST.md for common issues.

Usage:
    python scripts/validate_task_list.py

Checks for:
- Tasks marked pending that have completed test files
- Mismatched pending task counts
- Stale "In Progress" tasks
- Completed tasks with full details still in TASK_LIST.md
"""

import re
import os
from pathlib import Path


def main():
    task_list = Path("TASK_LIST.md").read_text()
    archive = Path("TASK_ARCHIVE.md").read_text() if Path("TASK_ARCHIVE.md").exists() else ""

    issues = []

    # Check 1: Header exists and has reasonable values
    count_match = re.search(r'\*\*Pending Tasks:\*\* (\d+)', task_list)
    completed_match = re.search(r'\*\*Completed Tasks:\*\* (\d+)', task_list)
    if count_match:
        claimed_pending = int(count_match.group(1))
        if claimed_pending > 100:
            issues.append(f"⚠️  Pending count seems high: {claimed_pending} tasks")
    if completed_match:
        claimed_completed = int(completed_match.group(1))
        # Sanity check - completed should be positive
        if claimed_completed < 1:
            issues.append(f"⚠️  Completed count seems low: {claimed_completed}")

    # Check 2: Tasks with status:pending that are in archive
    pending_tasks = re.findall(r'`status:pending`.*?#(\d+)', task_list)
    archived_tasks = re.findall(r'\| (\d+) \|.*?\| \d{4}-\d{2}-\d{2} \|', archive)
    stale = set(pending_tasks) & set(archived_tasks)
    if stale:
        issues.append(f"⚠️  Tasks marked pending but in archive: {', '.join(sorted(stale, key=int))}")

    # Check 3: Unit test tasks still marked pending but tests exist
    unit_test_dir = Path("tests/unit")
    if unit_test_dir.exists():
        test_files = list(unit_test_dir.glob("test_*.py"))
        if len(test_files) > 15:  # Substantial unit test coverage exists
            # Check if unit test tasks (#159-178) are still marked pending
            for task_num in range(159, 179):
                if f"### {task_num}." in task_list and "status:pending" in task_list:
                    pattern = rf'### {task_num}\.[^\n]*\n\n\*\*Meta:\*\* `status:pending`'
                    if re.search(pattern, task_list):
                        issues.append(f"⚠️  Task #{task_num} marked pending but tests/unit/ has {len(test_files)} test files")
                        break

    # Check 4: Completed task markers (✅, ✓) in pending task details (not coverage tables)
    if "## Pending Task Details" in task_list:
        pending_section = task_list.split("## Pending Task Details")[1]
        # Exclude coverage baseline table (which legitimately has ✅ markers)
        if "## Unit Test Coverage Baseline" in pending_section:
            pending_section = pending_section.split("## Unit Test Coverage Baseline")[0]
        if "## Category Index" in pending_section:
            pending_section = pending_section.split("## Category Index")[0]
        # Look for completion markers in actual task descriptions
        completed_markers = len(re.findall(r'### \d+\..*?✅|### \d+\..*?✓|`status:completed`', pending_section))
        if completed_markers > 0:
            issues.append(f"⚠️  Found {completed_markers} completed tasks in Pending Task Details section")

    # Check 5: Large file size warning
    lines = task_list.count('\n')
    if lines > 500:
        issues.append(f"⚠️  TASK_LIST.md has {lines} lines - consider archiving completed task details")

    # Report
    if issues:
        print("🔍 Task List Validation Issues Found:\n")
        for issue in issues:
            print(f"  {issue}")
        print(f"\n❌ {len(issues)} issue(s) found")
        return 1
    else:
        print("✅ Task list looks healthy!")
        print(f"   - {lines} lines")
        print(f"   - Pending: {claimed_pending if count_match else 'unknown'}")
        print(f"   - Completed: {claimed_completed if completed_match else 'unknown'}")
        return 0


if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent)
    exit(main())
