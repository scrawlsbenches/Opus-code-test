#!/usr/bin/env python3
"""
CI Task Reporter - Intelligent pending task output for CI pipelines.

This script outputs pending tasks from GoT (Graph of Thought) in a CI-friendly
format, suitable for:
- GitHub Actions job summaries
- Console output during CI runs
- Slack/Discord notifications

Features:
- Groups by priority (critical/high items first)
- Shows task status
- Provides actionable summary
- Exits with non-zero code if high-priority tasks exist (optional)

Usage:
    # Standard output
    python scripts/ci_task_report.py

    # GitHub Actions markdown format (writes to $GITHUB_STEP_SUMMARY)
    python scripts/ci_task_report.py --github

    # Fail CI if high-priority tasks pending
    python scripts/ci_task_report.py --fail-on-high

    # Quiet mode (summary only)
    python scripts/ci_task_report.py --quiet
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cortical.got.api import GoTManager
from cortical.got.types import Task
from cortical.core.bootstrap import create_container

# Default GoT directory
GOT_DIR = PROJECT_ROOT / ".got"


def get_pending_tasks(got_dir: Path = GOT_DIR) -> List[Task]:
    """Load only pending and in_progress tasks from GoT."""
    container = create_container(got_dir=got_dir)
    manager = container.resolve(GoTManager)
    pending = manager.list_tasks(status="pending")
    in_progress = manager.list_tasks(status="in_progress")
    return pending + in_progress


def group_by_priority(tasks: List[Task]) -> Dict[str, List[Task]]:
    """Group tasks by priority."""
    grouped = {"critical": [], "high": [], "medium": [], "low": []}
    for task in tasks:
        priority = task.priority if task.priority in grouped else "medium"
        grouped[priority].append(task)
    return grouped


def format_console_report(tasks: List[Task]) -> str:
    """Format tasks for console output."""
    if not tasks:
        return "✅ No pending tasks!\n"

    grouped = group_by_priority(tasks)
    lines = []

    # Summary header
    total = len(tasks)
    critical_count = len(grouped["critical"])
    high_count = len(grouped["high"])
    in_progress = sum(1 for t in tasks if t.status == "in_progress")

    lines.append("=" * 60)
    lines.append(f"📋 PENDING TASKS: {total} total ({critical_count} critical, {high_count} high)")
    if in_progress:
        lines.append(f"   🔄 {in_progress} currently in progress")
    lines.append("=" * 60)

    # Priority sections
    priority_config = [
        ("critical", "🔥 CRITICAL", "Immediate attention required!"),
        ("high", "🔴 HIGH PRIORITY", "These need attention!"),
        ("medium", "🟡 MEDIUM PRIORITY", ""),
        ("low", "🟢 LOW PRIORITY", ""),
    ]

    for priority, header, note in priority_config:
        if not grouped[priority]:
            continue
        lines.append("")
        lines.append(f"{header}" + (f" - {note}" if note else ""))
        lines.append("-" * 40)

        for task in grouped[priority]:
            status_marker = "🔄" if task.status == "in_progress" else "  "
            lines.append(f"  {status_marker} {task.id}")
            lines.append(f"       {task.title}")

    lines.append("")
    lines.append("=" * 60)

    # Actionable summary
    if critical_count > 0:
        lines.append("🔥 CRITICAL TASKS REQUIRE IMMEDIATE ATTENTION")
    elif high_count > 0:
        lines.append("⚠️  HIGH PRIORITY TASKS REQUIRE ATTENTION")

    return "\n".join(lines)


def format_github_markdown(tasks: List[Task]) -> str:
    """Format tasks as GitHub-flavored markdown for job summary."""
    if not tasks:
        return "## ✅ No Pending Tasks\n\nAll tasks have been completed!\n"

    grouped = group_by_priority(tasks)
    lines = []

    # Summary header
    total = len(tasks)
    critical_count = len(grouped["critical"])
    high_count = len(grouped["high"])
    in_progress = sum(1 for t in tasks if t.status == "in_progress")

    lines.append("## 📋 Pending Tasks Summary")
    lines.append("")
    lines.append("| Metric | Count |")
    lines.append("|--------|-------|")
    lines.append(f"| Total Pending | **{total}** |")
    lines.append(f"| 🔥 Critical | {critical_count} |")
    lines.append(f"| 🔴 High Priority | {high_count} |")
    lines.append(f"| 🟡 Medium Priority | {len(grouped['medium'])} |")
    lines.append(f"| 🟢 Low Priority | {len(grouped['low'])} |")
    lines.append(f"| 🔄 In Progress | {in_progress} |")
    lines.append("")

    # Critical/high priority callout
    if critical_count > 0:
        lines.append("> 🔥 **CRITICAL:** There are critical tasks that need immediate attention!")
        lines.append("")
    elif high_count > 0:
        lines.append("> ⚠️ **Attention:** There are high-priority tasks that need attention!")
        lines.append("")

    # Task tables by priority
    priority_config = [
        ("critical", "### 🔥 Critical"),
        ("high", "### 🔴 High Priority"),
        ("medium", "### 🟡 Medium Priority"),
        ("low", "### 🟢 Low Priority"),
    ]

    for priority, header in priority_config:
        if not grouped[priority]:
            continue

        lines.append(header)
        lines.append("")
        lines.append("| Status | ID | Title |")
        lines.append("|--------|----|----|")

        for task in grouped[priority]:
            status = "🔄" if task.status == "in_progress" else "📋"
            # Escape pipe characters in title
            title = task.title.replace("|", "\\|")
            # Truncate long titles
            if len(title) > 50:
                title = title[:47] + "..."
            lines.append(f"| {status} | `{task.id}` | {title} |")

        lines.append("")

    # Quick commands
    lines.append("<details>")
    lines.append("<summary>📌 Quick Commands</summary>")
    lines.append("")
    lines.append("```bash")
    lines.append("# List all tasks")
    lines.append("python -m cortical.got task list")
    lines.append("")
    lines.append("# Complete a task")
    lines.append("python -m cortical.got task complete T-XXXXX --notes \"...\"")
    lines.append("")
    lines.append("# Create new task")
    lines.append("python -m cortical.got task create \"Task title\" --priority high")
    lines.append("```")
    lines.append("</details>")

    return "\n".join(lines)


def format_quiet_report(tasks: List[Task]) -> str:
    """Minimal one-line summary."""
    if not tasks:
        return "Tasks: 0 pending"

    grouped = group_by_priority(tasks)
    return (
        f"Tasks: {len(tasks)} pending "
        f"(🔥{len(grouped['critical'])} 🔴{len(grouped['high'])} "
        f"🟡{len(grouped['medium'])} 🟢{len(grouped['low'])})"
    )


def main():
    parser = argparse.ArgumentParser(
        description="CI Task Reporter - Output pending tasks from GoT for CI pipelines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--github", action="store_true",
        help="Output GitHub-flavored markdown (writes to GITHUB_STEP_SUMMARY if available)"
    )
    parser.add_argument(
        "--fail-on-high", action="store_true",
        help="Exit with code 1 if critical or high-priority tasks exist"
    )
    parser.add_argument(
        "--fail-on-critical", action="store_true",
        help="Exit with code 1 only if critical tasks exist"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Minimal output (summary line only)"
    )
    parser.add_argument(
        "--got-dir",
        type=Path,
        default=GOT_DIR,
        help=f"GoT directory (default: {GOT_DIR})"
    )
    parser.add_argument(
        "--output", "-o",
        help="Write report to file instead of stdout"
    )

    args = parser.parse_args()

    # Load pending tasks
    try:
        tasks = get_pending_tasks(args.got_dir)
    except Exception as e:
        print(f"Warning: Could not load GoT tasks: {e}", file=sys.stderr)
        tasks = []

    # Format report
    if args.quiet:
        report = format_quiet_report(tasks)
    elif args.github:
        report = format_github_markdown(tasks)
    else:
        report = format_console_report(tasks)

    # Output report
    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report written to: {args.output}")
    else:
        print(report)

    # GitHub Actions: Write to step summary if available
    if args.github and "GITHUB_STEP_SUMMARY" in os.environ:
        summary_file = os.environ["GITHUB_STEP_SUMMARY"]
        with open(summary_file, "a") as f:
            f.write(report + "\n")

    # Exit code logic
    grouped = group_by_priority(tasks)

    if args.fail_on_critical and grouped["critical"]:
        print(f"\n❌ Failing: {len(grouped['critical'])} critical tasks pending")
        sys.exit(1)

    if args.fail_on_high and (grouped["critical"] or grouped["high"]):
        count = len(grouped["critical"]) + len(grouped["high"])
        print(f"\n❌ Failing: {count} critical/high-priority tasks pending")
        sys.exit(1)


if __name__ == "__main__":
    main()
