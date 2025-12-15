#!/usr/bin/env python3
"""
⚠️  DEPRECATED: This validator is for the legacy TASK_LIST.md file.

Use the new task validator instead:
    python scripts/validate_tasks.py

The project now uses a merge-friendly task system in tasks/ directory.
See docs/merge-friendly-tasks.md for documentation.

---

Legacy validator for TASK_LIST.md - kept for historical reference only.

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


def find_completed_tasks_in_pending_section(task_list: str) -> list[tuple[str, str, str]]:
    """
    Find tasks with completion markers in the Pending Task Details section.

    Returns list of (task_number, task_title, marker_found) tuples.
    """
    results = []

    if "## Pending Task Details" not in task_list:
        return results

    pending_section = task_list.split("## Pending Task Details")[1]

    # Exclude coverage baseline table (which legitimately has ✅ markers)
    if "## Unit Test Coverage Baseline" in pending_section:
        pending_section = pending_section.split("## Unit Test Coverage Baseline")[0]
    if "## Category Index" in pending_section:
        pending_section = pending_section.split("## Category Index")[0]

    # Find task headers with completion markers
    # Pattern: ### 123. Task Title ✅ or ### 123. Task Title
    task_pattern = r'### (\d+)\. ([^\n]*)'
    for match in re.finditer(task_pattern, pending_section):
        task_num = match.group(1)
        task_title = match.group(2).strip()

        # Check for completion markers in the title
        if '✅' in task_title:
            results.append((task_num, task_title.replace('✅', '').strip(), '✅ in title'))
        elif '✓' in task_title:
            results.append((task_num, task_title.replace('✓', '').strip(), '✓ in title'))
        else:
            # Check for status:completed in the task's section
            # Get the section content (until next ### or end)
            start = match.end()
            next_task = pending_section.find('### ', start)
            section_end = next_task if next_task != -1 else len(pending_section)
            section_content = pending_section[start:section_end]

            if '`status:completed`' in section_content:
                results.append((task_num, task_title, '`status:completed` in meta'))

    return results


def count_backlog_tasks(task_list: str) -> tuple[int, dict[str, list[str]]]:
    """
    Count actual pending tasks in backlog tables.

    Returns (total_count, {priority: [task_numbers]}).
    """
    tasks_by_priority = {}
    total = 0

    # Match table rows in backlog sections
    # Pattern: | 123 | Task name | Category | ... |
    in_backlog = False
    current_priority = None

    for line in task_list.split('\n'):
        # Detect priority section headers
        if '### 🟠 High' in line:
            current_priority = 'High'
            in_backlog = True
            tasks_by_priority['High'] = []
        elif '### 🟡 Medium' in line:
            current_priority = 'Medium'
            in_backlog = True
            tasks_by_priority['Medium'] = []
        elif '### 🟢 Low' in line:
            current_priority = 'Low'
            in_backlog = True
            tasks_by_priority['Low'] = []
        elif '### ⏸️ Deferred' in line or '### 🔮 Future' in line or '### 🔄 In Progress' in line:
            in_backlog = False
            current_priority = None
        elif line.startswith('---'):
            in_backlog = False
            current_priority = None
        elif in_backlog and current_priority and line.startswith('|'):
            # Match task row: | 123 | ... |
            match = re.match(r'\| (\d+) \|', line)
            if match:
                task_num = match.group(1)
                tasks_by_priority[current_priority].append(task_num)
                total += 1

    return total, tasks_by_priority


def main():
    task_list = Path("TASK_LIST.md").read_text()
    archive = Path("TASK_ARCHIVE.md").read_text() if Path("TASK_ARCHIVE.md").exists() else ""

    issues = []
    details = []  # Additional context for issues

    # Check 1: Header exists and has reasonable values
    count_match = re.search(r'\*\*Pending Tasks:\*\* (\d+)', task_list)
    completed_match = re.search(r'\*\*Completed Tasks:\*\* (\d+)', task_list)

    claimed_pending = int(count_match.group(1)) if count_match else None
    claimed_completed = int(completed_match.group(1)) if completed_match else None

    if claimed_pending and claimed_pending > 100:
        issues.append(f"⚠️  Pending count seems high: {claimed_pending} tasks")
        details.append("   Consider archiving completed tasks to TASK_ARCHIVE.md")

    if claimed_completed is not None and claimed_completed < 1:
        issues.append(f"⚠️  Completed count seems low: {claimed_completed}")
        details.append("   Update the header count to match archived tasks")

    # Check 1b: Verify claimed pending count matches actual backlog
    actual_pending, tasks_by_priority = count_backlog_tasks(task_list)
    if claimed_pending and actual_pending != claimed_pending:
        issues.append(f"⚠️  Pending task count mismatch: header says {claimed_pending}, actual is {actual_pending}")
        breakdown = ", ".join(f"{p}: {len(t)}" for p, t in tasks_by_priority.items() if t)
        details.append(f"   Breakdown: {breakdown}")
        details.append(f"   Fix: Update header '**Pending Tasks:** {actual_pending}'")

    # Check 2: Tasks with status:pending that are in archive
    pending_tasks = re.findall(r'`status:pending`.*?#(\d+)', task_list)
    archived_tasks = re.findall(r'\| (\d+) \|.*?\| \d{4}-\d{2}-\d{2} \|', archive)
    stale = set(pending_tasks) & set(archived_tasks)
    if stale:
        sorted_stale = sorted(stale, key=int)
        issues.append(f"⚠️  {len(stale)} task(s) marked pending but already in archive: #{', #'.join(sorted_stale)}")
        details.append("   Fix: Remove these task details from Pending Task Details section")
        details.append("        (they're already archived in TASK_ARCHIVE.md)")

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
                        details.append(f"   Fix: Update task #{task_num} status to completed and move to archive")
                        break

    # Check 4: Completed task markers in pending task details
    completed_in_pending = find_completed_tasks_in_pending_section(task_list)
    if completed_in_pending:
        issues.append(f"⚠️  Found {len(completed_in_pending)} completed task(s) in Pending Task Details section:")
        for task_num, task_title, marker in completed_in_pending:
            details.append(f"      - #{task_num}: {task_title[:50]}{'...' if len(task_title) > 50 else ''} ({marker})")
        details.append("   Fix: Move completed task details to TASK_ARCHIVE.md")
        details.append("        or remove the section if summary exists elsewhere")

    # Check 5: Large file size warning
    lines = task_list.count('\n')
    if lines > 500:
        issues.append(f"⚠️  TASK_LIST.md has {lines} lines (recommended: <500)")
        details.append("   Consider: Move completed task details to TASK_ARCHIVE.md")
        details.append("            Keep TASK_LIST.md focused on pending/active work")

    # Report
    if issues:
        print("🔍 Task List Validation Issues Found:\n")
        for i, issue in enumerate(issues):
            print(f"  {issue}")
            # Print related details
            for detail in details:
                if detail.startswith("   "):
                    print(f"  {detail}")
                    details.remove(detail)
                else:
                    break

        # Print any remaining details
        if details:
            print("\n  Additional context:")
            for detail in details:
                print(f"  {detail}")

        print(f"\n❌ {len(issues)} issue(s) found - see above for fix suggestions")
        return 1
    else:
        print("✅ Task list looks healthy!")
        print(f"   - {lines} lines")
        print(f"   - Pending: {claimed_pending if count_match else 'unknown'}")
        if tasks_by_priority:
            breakdown = ", ".join(f"{p}: {len(t)}" for p, t in tasks_by_priority.items() if t)
            print(f"   - Breakdown: {breakdown}")
        print(f"   - Completed: {claimed_completed if completed_match else 'unknown'}")
        return 0


if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent)
    exit(main())
