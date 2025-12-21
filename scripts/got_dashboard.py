#!/usr/bin/env python3
"""
GoT Dashboard - Comprehensive metrics and health indicators

Displays visual dashboard showing:
- Overview stats (tasks, edges, completion rate)
- Velocity metrics (tasks completed, avg completion time)
- Health indicators (blocked, stale, orphans)
- Agent performance (if available)
- Git integration status

Usage:
    python scripts/got_dashboard.py
    python scripts/got_utils.py dashboard
"""

import os
import sys
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cortical.reasoning.graph_of_thought import NodeType, EdgeType


# =============================================================================
# ANSI COLOR CODES
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"

    @staticmethod
    def enabled() -> bool:
        """Check if colors are supported."""
        return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def colorize(text: str, color: str) -> str:
    """Add color to text if colors are enabled."""
    if Colors.enabled():
        return f"{color}{text}{Colors.RESET}"
    return text


def bold(text: str) -> str:
    """Make text bold."""
    return colorize(text, Colors.BOLD)


def dim(text: str) -> str:
    """Make text dim."""
    return colorize(text, Colors.DIM)


# =============================================================================
# ASCII ART & FORMATTING
# =============================================================================

def draw_box(title: str, width: int = 80) -> Tuple[str, str, str]:
    """Draw a box around content.

    Returns:
        (top_border, title_line, bottom_border)
    """
    top = "┌" + "─" * (width - 2) + "┐"
    bottom = "└" + "─" * (width - 2) + "┘"

    # Center title
    title_len = len(title)
    padding = (width - title_len - 4) // 2
    title_line = "│ " + " " * padding + bold(title) + " " * (width - title_len - padding - 4) + " │"

    return top, title_line, bottom


def draw_separator(width: int = 80) -> str:
    """Draw a separator line."""
    return "├" + "─" * (width - 2) + "┤"


def draw_progress_bar(
    current: int,
    total: int,
    width: int = 40,
    filled_char: str = "█",
    empty_char: str = "░"
) -> str:
    """Draw a progress bar.

    Args:
        current: Current value
        total: Total value
        width: Width of the bar in characters
        filled_char: Character for filled portion
        empty_char: Character for empty portion

    Returns:
        Formatted progress bar string
    """
    if total == 0:
        percentage = 0
    else:
        percentage = (current / total) * 100

    filled = int((current / total) * width) if total > 0 else 0
    empty = width - filled

    bar = filled_char * filled + empty_char * empty

    # Color based on percentage
    if percentage >= 75:
        color = Colors.GREEN
    elif percentage >= 50:
        color = Colors.YELLOW
    elif percentage >= 25:
        color = Colors.BLUE
    else:
        color = Colors.RED

    colored_bar = colorize(bar, color)
    return f"{colored_bar} {current}/{total} ({percentage:.1f}%)"


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    elif seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    else:
        return f"{seconds / 86400:.1f}d"


def format_time_ago(timestamp_str: str) -> str:
    """Format timestamp as 'X ago'."""
    try:
        # Try parsing ISO format
        if "Z" in timestamp_str:
            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        else:
            timestamp = datetime.fromisoformat(timestamp_str)

        # Convert to naive datetime for comparison
        if timestamp.tzinfo is not None:
            timestamp = timestamp.replace(tzinfo=None)

        delta = datetime.now() - timestamp

        if delta.days > 0:
            return f"{delta.days}d ago"
        elif delta.seconds >= 3600:
            return f"{delta.seconds // 3600}h ago"
        elif delta.seconds >= 60:
            return f"{delta.seconds // 60}m ago"
        else:
            return "just now"
    except Exception:
        return "unknown"


# =============================================================================
# METRICS COLLECTION
# =============================================================================

class DashboardMetrics:
    """Collect and compute dashboard metrics from GoT graph."""

    def __init__(self, manager):
        """Initialize with a GoTProjectManager instance."""
        self.manager = manager
        self.graph = manager.graph

    def get_overview_stats(self) -> Dict[str, Any]:
        """Get overview statistics."""
        tasks = [n for n in self.graph.nodes.values() if n.node_type == NodeType.TASK]
        edges = self.graph.edges  # edges is a list, not a dict
        decisions = [n for n in self.graph.nodes.values() if n.node_type == NodeType.DECISION]

        # Count handoffs from events
        from scripts.got_utils import EventLog, HandoffManager
        events = EventLog.load_all_events(self.manager.events_dir)
        handoffs = HandoffManager.load_handoffs_from_events(events)

        # Task completion rate
        completed_tasks = [t for t in tasks if t.properties.get("status") == "completed"]
        total_tasks = len(tasks)
        completion_rate = (len(completed_tasks) / total_tasks * 100) if total_tasks > 0 else 0

        # Edge density (edges per node)
        total_nodes = len(self.graph.nodes)
        edge_density = len(edges) / total_nodes if total_nodes > 0 else 0

        return {
            "total_tasks": total_tasks,
            "total_edges": len(edges),
            "total_decisions": len(decisions),
            "total_handoffs": len(handoffs),
            "completed_tasks": len(completed_tasks),
            "completion_rate": completion_rate,
            "edge_density": edge_density,
            "total_nodes": total_nodes,
        }

    def get_velocity_metrics(self) -> Dict[str, Any]:
        """Get velocity and throughput metrics."""
        tasks = [n for n in self.graph.nodes.values() if n.node_type == NodeType.TASK]

        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = now - timedelta(days=7)

        # Tasks completed today
        completed_today = []
        completed_this_week = []

        for task in tasks:
            if task.properties.get("status") == "completed":
                completed_at_str = task.properties.get("completed_at")
                if completed_at_str:
                    try:
                        if "Z" in completed_at_str:
                            completed_at = datetime.fromisoformat(completed_at_str.replace("Z", "+00:00"))
                        else:
                            completed_at = datetime.fromisoformat(completed_at_str)

                        # Convert to naive
                        if completed_at.tzinfo is not None:
                            completed_at = completed_at.replace(tzinfo=None)

                        if completed_at >= today_start:
                            completed_today.append(task)
                        if completed_at >= week_start:
                            completed_this_week.append(task)
                    except Exception:
                        pass

        # Average completion time (for tasks with created_at and completed_at)
        completion_times = []
        for task in tasks:
            if task.properties.get("status") == "completed":
                created_at_str = task.metadata.get("created_at") or task.properties.get("created_at")
                completed_at_str = task.properties.get("completed_at")

                if created_at_str and completed_at_str:
                    try:
                        # Parse timestamps
                        if "Z" in created_at_str:
                            created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                        else:
                            created_at = datetime.fromisoformat(created_at_str)

                        if "Z" in completed_at_str:
                            completed_at = datetime.fromisoformat(completed_at_str.replace("Z", "+00:00"))
                        else:
                            completed_at = datetime.fromisoformat(completed_at_str)

                        # Convert to naive
                        if created_at.tzinfo is not None:
                            created_at = created_at.replace(tzinfo=None)
                        if completed_at.tzinfo is not None:
                            completed_at = completed_at.replace(tzinfo=None)

                        duration = (completed_at - created_at).total_seconds()
                        if duration >= 0:  # Sanity check
                            completion_times.append(duration)
                    except Exception:
                        pass

        avg_completion_time = sum(completion_times) / len(completion_times) if completion_times else 0

        # Sprint burndown (if active sprint exists)
        sprint_tasks = [t for t in tasks if t.properties.get("sprint_id")]
        sprint_completed = [t for t in sprint_tasks if t.properties.get("status") == "completed"]
        sprint_remaining = len(sprint_tasks) - len(sprint_completed)

        return {
            "completed_today": len(completed_today),
            "completed_this_week": len(completed_this_week),
            "avg_completion_time": avg_completion_time,
            "sprint_total": len(sprint_tasks),
            "sprint_completed": len(sprint_completed),
            "sprint_remaining": sprint_remaining,
        }

    def get_health_indicators(self) -> Dict[str, Any]:
        """Get health indicators."""
        tasks = [n for n in self.graph.nodes.values() if n.node_type == NodeType.TASK]

        # Blocked tasks
        blocked_tasks = [t for t in tasks if t.properties.get("status") == "blocked"]

        # Stale tasks (pending for > 7 days)
        now = datetime.now()
        stale_threshold = now - timedelta(days=7)
        stale_tasks = []

        for task in tasks:
            if task.properties.get("status") == "pending":
                created_at_str = task.metadata.get("created_at") or task.properties.get("created_at")
                if created_at_str:
                    try:
                        if "Z" in created_at_str:
                            created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                        else:
                            created_at = datetime.fromisoformat(created_at_str)

                        # Convert to naive
                        if created_at.tzinfo is not None:
                            created_at = created_at.replace(tzinfo=None)

                        if created_at < stale_threshold:
                            stale_tasks.append(task)
                    except Exception:
                        pass

        # Orphan nodes (no edges)
        orphan_nodes = []
        for node in self.graph.nodes.values():
            # Check if node has any edges
            has_edges = False
            for edge in self.graph.edges:  # edges is a list
                if edge.source_id == node.id or edge.target_id == node.id:
                    has_edges = True
                    break

            if not has_edges:
                orphan_nodes.append(node)

        return {
            "blocked_count": len(blocked_tasks),
            "stale_count": len(stale_tasks),
            "orphan_count": len(orphan_nodes),
            "blocked_tasks": blocked_tasks[:5],  # Top 5 for display
            "stale_tasks": stale_tasks[:5],
            "orphan_nodes": orphan_nodes[:5],
        }

    def get_agent_performance(self) -> Dict[str, Any]:
        """Get agent performance metrics from handoffs."""
        from scripts.got_utils import EventLog, HandoffManager
        events = EventLog.load_all_events(self.manager.events_dir)
        handoffs = HandoffManager.load_handoffs_from_events(events)

        if not handoffs:
            return {
                "total_handoffs": 0,
                "agent_stats": {},
            }

        # Group by agent
        agent_stats = defaultdict(lambda: {
            "total": 0,
            "completed": 0,
            "rejected": 0,
            "pending": 0,
            "durations": [],
        })

        for handoff in handoffs:
            target_agent = handoff.get("target_agent", "unknown")
            status = handoff.get("status", "pending")

            agent_stats[target_agent]["total"] += 1

            if status == "completed":
                agent_stats[target_agent]["completed"] += 1

                # Calculate duration
                initiated_at = handoff.get("initiated_at")
                completed_at = handoff.get("completed_at")

                if initiated_at and completed_at:
                    try:
                        if "Z" in initiated_at:
                            init = datetime.fromisoformat(initiated_at.replace("Z", "+00:00"))
                        else:
                            init = datetime.fromisoformat(initiated_at)

                        if "Z" in completed_at:
                            comp = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
                        else:
                            comp = datetime.fromisoformat(completed_at)

                        # Convert to naive
                        if init.tzinfo is not None:
                            init = init.replace(tzinfo=None)
                        if comp.tzinfo is not None:
                            comp = comp.replace(tzinfo=None)

                        duration = (comp - init).total_seconds()
                        if duration >= 0:
                            agent_stats[target_agent]["durations"].append(duration)
                    except Exception:
                        pass

            elif status == "rejected":
                agent_stats[target_agent]["rejected"] += 1
            else:
                agent_stats[target_agent]["pending"] += 1

        # Compute averages and success rates
        for agent, stats in agent_stats.items():
            total = stats["total"]
            completed = stats["completed"]
            rejected = stats["rejected"]

            stats["success_rate"] = (completed / total * 100) if total > 0 else 0
            stats["rejection_rate"] = (rejected / total * 100) if total > 0 else 0

            if stats["durations"]:
                stats["avg_duration"] = sum(stats["durations"]) / len(stats["durations"])
            else:
                stats["avg_duration"] = 0

        return {
            "total_handoffs": len(handoffs),
            "agent_stats": dict(agent_stats),
        }

    def get_commits_behind_origin(self) -> Dict[str, Any]:
        """Get how many commits local branch is behind origin.

        Returns:
            Dict with:
            - behind_count: int - commits behind origin
            - ahead_count: int - commits ahead of origin
            - status: str - 'up-to-date', 'behind', 'ahead', 'diverged', 'no-upstream', 'error'
            - message: str - human-readable status
            - last_fetch: str - time since last fetch (if available)
        """
        try:
            # Fetch to update refs (quick, no merge)
            # Use --quiet to suppress output
            fetch_result = subprocess.run(
                ["git", "fetch", "--quiet"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=PROJECT_ROOT
            )

            # Get current branch
            branch = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, check=True, cwd=PROJECT_ROOT
            ).stdout.strip()

            # Check if upstream is configured
            try:
                upstream = subprocess.run(
                    ["git", "rev-parse", "--abbrev-ref", "@{upstream}"],
                    capture_output=True,
                    text=True,
                    check=True,
                    cwd=PROJECT_ROOT
                ).stdout.strip()
            except subprocess.CalledProcessError:
                return {
                    "behind_count": 0,
                    "ahead_count": 0,
                    "status": "no-upstream",
                    "message": f"Branch '{branch}' has no upstream configured",
                    "last_fetch": None,
                }

            # Get behind/ahead counts
            # Format: "ahead\tbehind"
            result = subprocess.run(
                ["git", "rev-list", "--left-right", "--count", "HEAD...@{upstream}"],
                capture_output=True,
                text=True,
                check=True,
                cwd=PROJECT_ROOT
            )

            # Parse "ahead\tbehind"
            parts = result.stdout.strip().split()
            if len(parts) != 2:
                return {
                    "behind_count": 0,
                    "ahead_count": 0,
                    "status": "error",
                    "message": "Could not parse git output",
                    "last_fetch": None,
                }

            ahead, behind = map(int, parts)

            # Determine status
            if ahead == 0 and behind == 0:
                status = "up-to-date"
                message = f"Up-to-date with {upstream}"
            elif ahead > 0 and behind == 0:
                status = "ahead"
                message = f"Ahead of {upstream} by {ahead} commit{'s' if ahead != 1 else ''}"
            elif ahead == 0 and behind > 0:
                status = "behind"
                message = f"Behind {upstream} by {behind} commit{'s' if behind != 1 else ''}"
            else:
                status = "diverged"
                message = f"Diverged from {upstream}: +{ahead} -{behind}"

            # Try to get last fetch time
            last_fetch = None
            try:
                fetch_head = PROJECT_ROOT / ".git" / "FETCH_HEAD"
                if fetch_head.exists():
                    mtime = fetch_head.stat().st_mtime
                    fetch_time = datetime.fromtimestamp(mtime)
                    delta = datetime.now() - fetch_time

                    if delta.days > 0:
                        last_fetch = f"{delta.days}d ago"
                    elif delta.seconds >= 3600:
                        last_fetch = f"{delta.seconds // 3600}h ago"
                    elif delta.seconds >= 60:
                        last_fetch = f"{delta.seconds // 60}m ago"
                    else:
                        last_fetch = "just now"
            except Exception:
                pass

            return {
                "behind_count": behind,
                "ahead_count": ahead,
                "status": status,
                "message": message,
                "last_fetch": last_fetch,
            }

        except subprocess.TimeoutExpired:
            return {
                "behind_count": 0,
                "ahead_count": 0,
                "status": "error",
                "message": "Network timeout during fetch",
                "last_fetch": None,
            }
        except subprocess.CalledProcessError as e:
            return {
                "behind_count": 0,
                "ahead_count": 0,
                "status": "error",
                "message": f"Git error: {e.stderr.strip() if e.stderr else 'unknown'}",
                "last_fetch": None,
            }
        except Exception as e:
            return {
                "behind_count": 0,
                "ahead_count": 0,
                "status": "error",
                "message": f"Unexpected error: {str(e)}",
                "last_fetch": None,
            }

    def get_git_integration_status(self) -> Dict[str, Any]:
        """Get git integration status."""
        try:
            # Current branch
            branch = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, check=True, cwd=PROJECT_ROOT
            ).stdout.strip()

            # Check if we're on main
            is_main = branch in ("main", "master")

            # Commits ahead/behind main
            if not is_main:
                try:
                    # Get main branch name
                    main_branch = "main" if subprocess.run(
                        ["git", "rev-parse", "--verify", "main"],
                        capture_output=True, cwd=PROJECT_ROOT
                    ).returncode == 0 else "master"

                    # Commits ahead
                    ahead = subprocess.run(
                        ["git", "rev-list", "--count", f"{main_branch}..HEAD"],
                        capture_output=True, text=True, check=True, cwd=PROJECT_ROOT
                    ).stdout.strip()

                    # Commits behind
                    behind = subprocess.run(
                        ["git", "rev-list", "--count", f"HEAD..{main_branch}"],
                        capture_output=True, text=True, check=True, cwd=PROJECT_ROOT
                    ).stdout.strip()

                    drift = {"ahead": int(ahead), "behind": int(behind)}
                except Exception:
                    drift = None
            else:
                drift = None

            # Uncommitted changes
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, check=True, cwd=PROJECT_ROOT
            ).stdout.strip()

            uncommitted_files = len(status.split("\n")) if status else 0

            # Recent commits with task refs
            log = subprocess.run(
                ["git", "log", "--oneline", "-20"],
                capture_output=True, text=True, check=True, cwd=PROJECT_ROOT
            ).stdout.strip()

            commits_with_tasks = []
            for line in log.split("\n"):
                if "task:" in line.lower() or "T-" in line:
                    commits_with_tasks.append(line)

            # Check commits behind origin
            origin_status = self.get_commits_behind_origin()

            return {
                "branch": branch,
                "is_main": is_main,
                "drift": drift,
                "uncommitted_files": uncommitted_files,
                "recent_task_commits": commits_with_tasks[:5],
                "origin_status": origin_status,
            }
        except Exception as e:
            return {
                "branch": "unknown",
                "is_main": False,
                "drift": None,
                "uncommitted_files": 0,
                "recent_task_commits": [],
                "origin_status": {
                    "status": "error",
                    "message": str(e),
                },
                "error": str(e),
            }


# =============================================================================
# DASHBOARD RENDERING
# =============================================================================

def render_overview_section(stats: Dict[str, Any], width: int = 80) -> List[str]:
    """Render overview section."""
    lines = []

    top, title_line, bottom = draw_box("OVERVIEW", width)
    lines.append(colorize(top, Colors.CYAN))
    lines.append(title_line)
    lines.append(colorize(draw_separator(width), Colors.CYAN))

    # Stats table
    lines.append(f"│ {bold('Total Nodes:')} {stats['total_nodes']:>15}                                         │")
    lines.append(f"│   Tasks:      {colorize(str(stats['total_tasks']), Colors.BLUE):>15}                                         │")
    lines.append(f"│   Decisions:  {colorize(str(stats['total_decisions']), Colors.MAGENTA):>15}                                         │")
    lines.append(f"│   Edges:      {colorize(str(stats['total_edges']), Colors.YELLOW):>15}                                         │")
    lines.append(f"│   Handoffs:   {colorize(str(stats['total_handoffs']), Colors.GREEN):>15}                                         │")
    lines.append(f"│                                                                              │")

    # Completion rate with progress bar
    completion_bar = draw_progress_bar(stats['completed_tasks'], stats['total_tasks'], width=40)
    lines.append(f"│ {bold('Task Completion:')}                                                         │")
    lines.append(f"│   {completion_bar}  │")
    lines.append(f"│                                                                              │")

    # Edge density
    density_color = Colors.GREEN if stats['edge_density'] >= 2.0 else Colors.YELLOW
    density_value = f"{stats['edge_density']:.2f}"
    lines.append(f"│ {bold('Edge Density:')} {colorize(density_value, density_color)} edges/node                                    │")

    lines.append(colorize(bottom, Colors.CYAN))
    return lines


def render_velocity_section(stats: Dict[str, Any], width: int = 80) -> List[str]:
    """Render velocity metrics section."""
    lines = []

    top, title_line, bottom = draw_box("VELOCITY METRICS", width)
    lines.append(colorize(top, Colors.GREEN))
    lines.append(title_line)
    lines.append(colorize(draw_separator(width), Colors.GREEN))

    # Today's completions
    today_color = Colors.BRIGHT_GREEN if stats['completed_today'] > 0 else Colors.DIM
    lines.append(f"│ {bold('Completed Today:')} {colorize(str(stats['completed_today']), today_color):>15}                                    │")

    # This week's completions
    week_color = Colors.GREEN if stats['completed_this_week'] > 0 else Colors.DIM
    lines.append(f"│ {bold('Completed This Week:')} {colorize(str(stats['completed_this_week']), week_color):>15}                                │")

    # Average completion time
    avg_time = format_duration(stats['avg_completion_time'])
    lines.append(f"│ {bold('Avg Completion Time:')} {colorize(avg_time, Colors.CYAN):>15}                                │")
    lines.append(f"│                                                                              │")

    # Sprint burndown (if active)
    if stats['sprint_total'] > 0:
        lines.append(f"│ {bold('Sprint Progress:')}                                                        │")
        sprint_bar = draw_progress_bar(stats['sprint_completed'], stats['sprint_total'], width=40)
        lines.append(f"│   {sprint_bar}  │")
        lines.append(f"│   Remaining: {colorize(str(stats['sprint_remaining']), Colors.YELLOW):>15}                                         │")
    else:
        lines.append(f"│ {dim('No active sprint')}                                                         │")

    lines.append(colorize(bottom, Colors.GREEN))
    return lines


def render_health_section(stats: Dict[str, Any], width: int = 80) -> List[str]:
    """Render health indicators section."""
    lines = []

    top, title_line, bottom = draw_box("HEALTH INDICATORS", width)
    lines.append(colorize(top, Colors.YELLOW))
    lines.append(title_line)
    lines.append(colorize(draw_separator(width), Colors.YELLOW))

    # Blocked tasks
    blocked_color = Colors.RED if stats['blocked_count'] > 0 else Colors.GREEN
    lines.append(f"│ {bold('Blocked Tasks:')} {colorize(str(stats['blocked_count']), blocked_color):>15}                                      │")

    if stats['blocked_tasks']:
        for task in stats['blocked_tasks']:
            title = task.content[:50] + "..." if len(task.content) > 50 else task.content
            lines.append(f"│   • {colorize(title, Colors.RED):<50}                        │")

    lines.append(f"│                                                                              │")

    # Stale tasks
    stale_color = Colors.YELLOW if stats['stale_count'] > 0 else Colors.GREEN
    lines.append(f"│ {bold('Stale Tasks:')} {colorize(str(stats['stale_count']), stale_color):>15} (pending > 7 days)                        │")

    if stats['stale_tasks']:
        for task in stats['stale_tasks'][:3]:
            title = task.content[:50] + "..." if len(task.content) > 50 else task.content
            created = task.metadata.get("created_at", "")
            age = format_time_ago(created) if created else "unknown"
            lines.append(f"│   • {title:<50} ({age})       │")

    lines.append(f"│                                                                              │")

    # Orphan nodes
    orphan_color = Colors.YELLOW if stats['orphan_count'] > 0 else Colors.GREEN
    lines.append(f"│ {bold('Orphan Nodes:')} {colorize(str(stats['orphan_count']), orphan_color):>15} (no edges)                              │")

    lines.append(colorize(bottom, Colors.YELLOW))
    return lines


def render_agent_performance_section(stats: Dict[str, Any], width: int = 80) -> List[str]:
    """Render agent performance section."""
    lines = []

    top, title_line, bottom = draw_box("AGENT PERFORMANCE", width)
    lines.append(colorize(top, Colors.MAGENTA))
    lines.append(title_line)
    lines.append(colorize(draw_separator(width), Colors.MAGENTA))

    if stats['total_handoffs'] == 0:
        lines.append(f"│ {dim('No handoff data available')}                                                │")
    else:
        lines.append(f"│ {bold('Total Handoffs:')} {colorize(str(stats['total_handoffs']), Colors.CYAN):>15}                                    │")
        lines.append(f"│                                                                              │")

        # Agent stats table
        if stats['agent_stats']:
            lines.append(f"│ {bold('Agent'):25} {bold('Success'):>8} {bold('Rejected'):>8} {bold('Avg Time'):>12}        │")
            lines.append(f"│ {'-'*25} {'-'*8} {'-'*8} {'-'*12}        │")

            for agent, agent_stats in stats['agent_stats'].items():
                success_rate = agent_stats['success_rate']
                rejection_rate = agent_stats['rejection_rate']
                avg_time = format_duration(agent_stats['avg_duration'])

                # Color based on success rate
                if success_rate >= 80:
                    success_color = Colors.GREEN
                elif success_rate >= 50:
                    success_color = Colors.YELLOW
                else:
                    success_color = Colors.RED

                agent_name = agent[:23] + ".." if len(agent) > 25 else agent
                success_str = colorize(f"{success_rate:.1f}%", success_color)
                reject_str = colorize(f"{rejection_rate:.1f}%", Colors.RED if rejection_rate > 0 else Colors.DIM)

                lines.append(f"│ {agent_name:25} {success_str:>8} {reject_str:>8} {colorize(avg_time, Colors.CYAN):>12}        │")

    lines.append(colorize(bottom, Colors.MAGENTA))
    return lines


def render_git_integration_section(stats: Dict[str, Any], width: int = 80) -> List[str]:
    """Render git integration section."""
    lines = []

    top, title_line, bottom = draw_box("GIT INTEGRATION", width)
    lines.append(colorize(top, Colors.BLUE))
    lines.append(title_line)
    lines.append(colorize(draw_separator(width), Colors.BLUE))

    # Current branch
    branch_color = Colors.GREEN if stats['is_main'] else Colors.YELLOW
    lines.append(f"│ {bold('Current Branch:')} {colorize(stats['branch'], branch_color):<50}                 │")

    # Origin sync status
    origin_status = stats.get('origin_status', {})
    if origin_status:
        status_type = origin_status.get('status', 'error')
        message = origin_status.get('message', 'Unknown')
        behind_count = origin_status.get('behind_count', 0)
        ahead_count = origin_status.get('ahead_count', 0)
        last_fetch = origin_status.get('last_fetch')

        # Determine color and warning icon
        if status_type == 'up-to-date':
            status_color = Colors.GREEN
            icon = "✓"
        elif status_type == 'ahead':
            status_color = Colors.CYAN
            icon = "↑"
        elif status_type == 'behind':
            # Warn if significantly behind
            if behind_count >= 5:
                status_color = Colors.RED
                icon = "⚠️"
            else:
                status_color = Colors.YELLOW
                icon = "↓"
        elif status_type == 'diverged':
            status_color = Colors.YELLOW
            icon = "⇅"
        elif status_type == 'no-upstream':
            status_color = Colors.DIM
            icon = "ⓘ"
        else:  # error
            status_color = Colors.DIM
            icon = "✗"

        # Display status line
        status_line = f"{icon} {message}"
        lines.append(f"│ {bold('Origin Status:')} {colorize(status_line, status_color):<50}            │")

        # Show last fetch time if available
        if last_fetch:
            lines.append(f"│ {bold('Last Fetch:')} {colorize(last_fetch, Colors.DIM):<50}                 │")

        # Add helpful tip if significantly behind
        if status_type == 'behind' and behind_count >= 5:
            lines.append(f"│                                                                              │")
            lines.append(f"│ {colorize('⚠️  Tip: Run git pull to sync with origin', Colors.YELLOW):<68}      │")

    # Branch drift from main
    if stats['drift']:
        ahead = stats['drift']['ahead']
        behind = stats['drift']['behind']

        drift_status = f"+{ahead} -{behind} from main"
        drift_color = Colors.YELLOW if ahead > 0 or behind > 0 else Colors.GREEN

        lines.append(f"│ {bold('Branch Drift:')} {colorize(drift_status, drift_color):<50}            │")

    # Uncommitted changes
    uncommitted_color = Colors.YELLOW if stats['uncommitted_files'] > 0 else Colors.GREEN
    uncommitted_status = f"{stats['uncommitted_files']} file{'s' if stats['uncommitted_files'] != 1 else ''}"
    lines.append(f"│ {bold('Uncommitted:')} {colorize(uncommitted_status, uncommitted_color):<50}               │")

    lines.append(f"│                                                                              │")

    # Recent commits with task refs
    if stats['recent_task_commits']:
        lines.append(f"│ {bold('Recent Task Commits:')}                                                   │")
        for commit in stats['recent_task_commits'][:3]:
            commit_short = commit[:70] + ".." if len(commit) > 72 else commit
            lines.append(f"│   {colorize(commit_short, Colors.DIM):<72}  │")
    else:
        lines.append(f"│ {dim('No recent task commits')}                                                  │")

    lines.append(colorize(bottom, Colors.BLUE))
    return lines


def render_dashboard(manager) -> str:
    """Render complete dashboard.

    Args:
        manager: GoTProjectManager instance

    Returns:
        Formatted dashboard string
    """
    metrics = DashboardMetrics(manager)

    # Collect all metrics
    overview = metrics.get_overview_stats()
    velocity = metrics.get_velocity_metrics()
    health = metrics.get_health_indicators()
    agent_perf = metrics.get_agent_performance()
    git_status = metrics.get_git_integration_status()

    # Render sections
    lines = []
    lines.append("")
    lines.append(colorize("=" * 80, Colors.BOLD))
    lines.append(colorize("                          GoT DASHBOARD", Colors.BOLD + Colors.BRIGHT_CYAN))
    lines.append(colorize(f"                        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", Colors.DIM))
    lines.append(colorize("=" * 80, Colors.BOLD))
    lines.append("")

    lines.extend(render_overview_section(overview))
    lines.append("")
    lines.extend(render_velocity_section(velocity))
    lines.append("")
    lines.extend(render_health_section(health))
    lines.append("")
    lines.extend(render_agent_performance_section(agent_perf))
    lines.append("")
    lines.extend(render_git_integration_section(git_status))
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    # Import here to avoid circular imports
    from scripts.got_utils import GoTProjectManager

    manager = GoTProjectManager()
    dashboard = render_dashboard(manager)
    print(dashboard)
    return 0


if __name__ == "__main__":
    sys.exit(main())
