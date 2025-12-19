#!/usr/bin/env python3
"""
ASCII Art Codebase Visualization

Generates beautiful ASCII art visualizations of the codebase structure,
git history, and code metrics. Inspired by the cortical/neural theme.

Usage:
    python scripts/ascii_codebase_art.py              # All visualizations
    python scripts/ascii_codebase_art.py --skyline    # Code skyline only
    python scripts/ascii_codebase_art.py --heatmap    # Git activity heatmap
    python scripts/ascii_codebase_art.py --tree       # Module dependency tree
    python scripts/ascii_codebase_art.py --layers     # Cortical layers view
    python scripts/ascii_codebase_art.py --flames     # Hot files visualization
    python scripts/ascii_codebase_art.py --timeline   # Commit timeline
"""

import argparse
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ANSI color codes for terminal output
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    # Foreground colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    # Bright foreground
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'

    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def run_git_command(args: List[str], cwd: Optional[Path] = None) -> str:
    """Run a git command and return output."""
    try:
        result = subprocess.run(
            ['git'] + args,
            capture_output=True,
            text=True,
            cwd=cwd or get_project_root()
        )
        return result.stdout.strip()
    except Exception:
        return ""


def get_file_stats() -> Dict[str, int]:
    """Get line counts for Python files in cortical/."""
    stats = {}
    cortical_dir = get_project_root() / 'cortical'

    for py_file in cortical_dir.rglob('*.py'):
        if '__pycache__' in str(py_file):
            continue
        try:
            lines = len(py_file.read_text().splitlines())
            rel_path = py_file.relative_to(cortical_dir)
            stats[str(rel_path)] = lines
        except Exception:
            pass

    return stats


def get_git_activity(days: int = 365) -> Dict[str, int]:
    """Get commit counts per day for the last N days."""
    activity = defaultdict(int)
    since = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

    log = run_git_command([
        'log', '--format=%ad', '--date=short', f'--since={since}'
    ])

    for line in log.splitlines():
        if line:
            activity[line] += 1

    return dict(activity)


def get_hot_files(limit: int = 15) -> List[Tuple[str, int]]:
    """Get most frequently modified files."""
    log = run_git_command([
        'log', '--format=', '--name-only', '-n', '500'
    ])

    file_counts = defaultdict(int)
    for line in log.splitlines():
        if line and line.endswith('.py') and 'cortical/' in line:
            file_counts[line] += 1

    sorted_files = sorted(file_counts.items(), key=lambda x: -x[1])
    return sorted_files[:limit]


def get_commit_timeline(limit: int = 20) -> List[Tuple[str, str, str]]:
    """Get recent commits with date, hash, and message."""
    log = run_git_command([
        'log', '--format=%ad|%h|%s', '--date=short', '-n', str(limit)
    ])

    commits = []
    for line in log.splitlines():
        if '|' in line:
            parts = line.split('|', 2)
            if len(parts) == 3:
                commits.append(tuple(parts))

    return commits


def get_module_dependencies() -> Dict[str, List[str]]:
    """Analyze import dependencies between cortical modules."""
    deps = defaultdict(list)
    cortical_dir = get_project_root() / 'cortical'

    for py_file in cortical_dir.glob('*.py'):
        if py_file.name.startswith('__'):
            continue

        module_name = py_file.stem
        try:
            content = py_file.read_text()
            for line in content.splitlines():
                if line.startswith('from .') or line.startswith('from cortical.'):
                    # Extract imported module
                    parts = line.split()
                    if len(parts) >= 2:
                        import_path = parts[1]
                        if import_path.startswith('.'):
                            imported = import_path.split('.')[1] if '.' in import_path[1:] else import_path[1:]
                        else:
                            imported = import_path.split('.')[1] if '.' in import_path else import_path
                        if imported and imported != module_name:
                            deps[module_name].append(imported)
        except Exception:
            pass

    return dict(deps)


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def render_skyline(use_color: bool = True) -> str:
    """
    Render a city skyline where building heights represent file sizes.

    Each building is a Python file, height = lines of code.
    """
    stats = get_file_stats()
    if not stats:
        return "No files found."

    # Sort by size for visual effect
    sorted_files = sorted(stats.items(), key=lambda x: -x[1])[:20]
    max_lines = max(stats.values()) if stats else 1

    # Scale to max height of 15
    max_height = 15
    scale = max_height / max_lines if max_lines > 0 else 1

    # Building characters
    roof = '▄'
    wall = '█'
    window = '▓'
    base = '▀'

    # Colors for buildings (cycle through)
    building_colors = [
        Colors.BLUE, Colors.CYAN, Colors.GREEN,
        Colors.YELLOW, Colors.MAGENTA, Colors.RED
    ]

    lines = []
    lines.append("")
    lines.append(f"{Colors.BOLD if use_color else ''}╔══════════════════════════════════════════════════════════════════════════╗{Colors.RESET if use_color else ''}")
    lines.append(f"{Colors.BOLD if use_color else ''}║{'CODE SKYLINE - File Sizes as Buildings':^76}║{Colors.RESET if use_color else ''}")
    lines.append(f"{Colors.BOLD if use_color else ''}╚══════════════════════════════════════════════════════════════════════════╝{Colors.RESET if use_color else ''}")
    lines.append("")

    # Build the skyline row by row from top to bottom
    building_width = 3
    num_buildings = min(len(sorted_files), 20)

    for row in range(max_height, 0, -1):
        line_chars = []
        for i, (filename, line_count) in enumerate(sorted_files[:num_buildings]):
            height = int(line_count * scale)
            color = building_colors[i % len(building_colors)] if use_color else ''
            reset = Colors.RESET if use_color else ''

            if row == height:
                # Roof
                line_chars.append(f"{color} {roof} {reset}")
            elif row < height:
                # Wall with occasional window
                if row % 3 == 0:
                    line_chars.append(f"{color}{wall}{window}{wall}{reset}")
                else:
                    line_chars.append(f"{color}{wall}{wall}{wall}{reset}")
            else:
                # Sky
                line_chars.append("   ")

        lines.append("  " + " ".join(line_chars))

    # Ground line
    ground = "▀" * (num_buildings * 4 + num_buildings - 1)
    lines.append(f"  {Colors.DIM if use_color else ''}{ground}{Colors.RESET if use_color else ''}")

    # Labels (abbreviated)
    label_line = "  "
    for i, (filename, line_count) in enumerate(sorted_files[:num_buildings]):
        short_name = Path(filename).stem[:3]
        label_line += f"{short_name} "
    lines.append(f"{Colors.DIM if use_color else ''}{label_line}{Colors.RESET if use_color else ''}")

    # Legend
    lines.append("")
    lines.append(f"  {Colors.DIM if use_color else ''}Legend: Height = Lines of Code (max: {max_lines} lines){Colors.RESET if use_color else ''}")
    lines.append("")

    # Top 5 with actual numbers
    lines.append(f"  {Colors.BOLD if use_color else ''}Top Files:{Colors.RESET if use_color else ''}")
    for filename, line_count in sorted_files[:5]:
        bar_len = int(line_count / max_lines * 30)
        bar = "█" * bar_len
        lines.append(f"  {filename:<25} {Colors.GREEN if use_color else ''}{bar}{Colors.RESET if use_color else ''} {line_count}")

    return "\n".join(lines)


def render_heatmap(use_color: bool = True, weeks: int = 26) -> str:
    """
    Render a GitHub-style contribution heatmap.

    Shows commit activity over time as a grid of intensity levels.
    """
    activity = get_git_activity(days=weeks * 7)

    if not activity:
        return "No git activity found."

    lines = []
    lines.append("")
    lines.append(f"{Colors.BOLD if use_color else ''}╔══════════════════════════════════════════════════════════════════════════╗{Colors.RESET if use_color else ''}")
    lines.append(f"{Colors.BOLD if use_color else ''}║{'GIT ACTIVITY HEATMAP':^76}║{Colors.RESET if use_color else ''}")
    lines.append(f"{Colors.BOLD if use_color else ''}╚══════════════════════════════════════════════════════════════════════════╝{Colors.RESET if use_color else ''}")
    lines.append("")

    # Intensity levels
    levels = [' ', '░', '▒', '▓', '█']
    level_colors = [
        Colors.DIM,
        Colors.GREEN,
        Colors.BRIGHT_GREEN,
        Colors.YELLOW,
        Colors.BRIGHT_YELLOW
    ]

    # Calculate max for scaling
    max_commits = max(activity.values()) if activity else 1

    # Build week columns (most recent on right)
    today = datetime.now().date()
    start_date = today - timedelta(days=weeks * 7)

    # Align to start of week (Monday)
    start_date = start_date - timedelta(days=start_date.weekday())

    # Day labels
    day_labels = ['Mon', '   ', 'Wed', '   ', 'Fri', '   ', 'Sun']

    # Build the heatmap grid
    grid = []
    for day_of_week in range(7):
        row = []
        current = start_date + timedelta(days=day_of_week)
        while current <= today:
            date_str = current.strftime('%Y-%m-%d')
            commits = activity.get(date_str, 0)

            if commits == 0:
                level = 0
            else:
                # Scale to levels 1-4
                level = min(4, 1 + int(commits / max_commits * 3))

            row.append((level, commits))
            current += timedelta(days=7)
        grid.append(row)

    # Month labels
    month_line = "       "
    current = start_date
    last_month = None
    for week in range(len(grid[0])):
        date = start_date + timedelta(days=week * 7)
        month = date.strftime('%b')
        if month != last_month:
            month_line += month[:3]
            last_month = month
        else:
            month_line += "   "
    lines.append(f"{Colors.DIM if use_color else ''}{month_line}{Colors.RESET if use_color else ''}")

    # Render grid
    for day_idx, row in enumerate(grid):
        line = f"  {day_labels[day_idx]} "
        for level, commits in row:
            char = levels[level]
            if use_color:
                color = level_colors[level]
                line += f"{color}{char}{char}{Colors.RESET}"
            else:
                line += char + char
        lines.append(line)

    lines.append("")

    # Legend
    legend = "  Less "
    for i, (char, color) in enumerate(zip(levels, level_colors)):
        if use_color:
            legend += f"{color}{char}{char}{Colors.RESET}"
        else:
            legend += char + char
    legend += " More"
    lines.append(legend)

    # Stats
    total_commits = sum(activity.values())
    active_days = len([d for d, c in activity.items() if c > 0])
    lines.append("")
    lines.append(f"  {Colors.BOLD if use_color else ''}Stats:{Colors.RESET if use_color else ''} {total_commits} commits over {active_days} active days")

    return "\n".join(lines)


def render_layers(use_color: bool = True) -> str:
    """
    Render the 4-layer cortical architecture as ASCII art.

    Shows the hierarchical processing from tokens to documents.
    """
    lines = []
    lines.append("")
    lines.append(f"{Colors.BOLD if use_color else ''}╔══════════════════════════════════════════════════════════════════════════╗{Colors.RESET if use_color else ''}")
    lines.append(f"{Colors.BOLD if use_color else ''}║{'CORTICAL LAYER ARCHITECTURE':^76}║{Colors.RESET if use_color else ''}")
    lines.append(f"{Colors.BOLD if use_color else ''}╚══════════════════════════════════════════════════════════════════════════╝{Colors.RESET if use_color else ''}")
    lines.append("")

    # Layer definitions with visual metaphors
    layers = [
        ("Layer 3: DOCUMENTS", "IT Cortex - Objects", Colors.MAGENTA, "📄", [
            "┌─────────────────────────────────────────────────────────────────────┐",
            "│  ╔═══════════╗   ╔═══════════╗   ╔═══════════╗   ╔═══════════╗     │",
            "│  ║   doc_1   ║   ║   doc_2   ║   ║   doc_3   ║   ║   doc_4   ║     │",
            "│  ║  ▓▓▓▓▓▓▓  ║   ║  ▓▓▓▓▓▓▓  ║   ║  ▓▓▓▓▓▓▓  ║   ║  ▓▓▓▓▓▓▓  ║     │",
            "│  ╚═══════════╝   ╚═══════════╝   ╚═══════════╝   ╚═══════════╝     │",
            "└─────────────────────────────────────────────────────────────────────┘"
        ]),
        ("Layer 2: CONCEPTS", "V4 Cortex - Shapes", Colors.CYAN, "🧠", [
            "          ↑               ↑               ↑               ↑          ",
            "    ╔═══════════════════════════════════════════════════════════╗    ",
            "    ║     ┌───────────┐       ┌───────────┐       ┌───────────┐ ║    ",
            "    ║     │ cluster_1 │ ←───→ │ cluster_2 │ ←───→ │ cluster_3 │ ║    ",
            "    ║     │  ○ ○ ○ ○  │       │  ○ ○ ○ ○  │       │  ○ ○ ○ ○  │ ║    ",
            "    ║     └───────────┘       └───────────┘       └───────────┘ ║    ",
            "    ╚═══════════════════════════════════════════════════════════╝    "
        ]),
        ("Layer 1: BIGRAMS", "V2 Cortex - Patterns", Colors.YELLOW, "🔗", [
            "          ↑               ↑               ↑               ↑          ",
            "   ┌──────────────────────────────────────────────────────────────┐  ",
            "   │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │  ",
            "   │  │ neural   │──│ networks │──│ process  │──│  data    │     │  ",
            "   │  │ networks │  │ process  │  │  data    │  │  flow    │     │  ",
            "   │  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │  ",
            "   └──────────────────────────────────────────────────────────────┘  "
        ]),
        ("Layer 0: TOKENS", "V1 Cortex - Edges", Colors.GREEN, "📝", [
            "          ↑               ↑               ↑               ↑          ",
            "  ╔════════════════════════════════════════════════════════════════╗ ",
            "  ║ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐║ ",
            "  ║ │ the│ │data│ │flow│ │in  │ │this│ │test│ │file│ │word│ │more│║ ",
            "  ║ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘║ ",
            "  ╚════════════════════════════════════════════════════════════════╝ "
        ])
    ]

    for layer_name, analogy, color, icon, art in layers:
        c = color if use_color else ''
        r = Colors.RESET if use_color else ''
        b = Colors.BOLD if use_color else ''
        d = Colors.DIM if use_color else ''

        lines.append(f"  {b}{c}{layer_name}{r}  {d}({analogy}){r}")
        for line in art:
            lines.append(f"  {c}{line}{r}")
        lines.append("")

    # Add data flow arrows at bottom
    lines.append(f"  {Colors.DIM if use_color else ''}{'─' * 72}{Colors.RESET if use_color else ''}")
    lines.append(f"  {Colors.BOLD if use_color else ''}Data Flow:{Colors.RESET if use_color else ''} Text → Tokens → Bigrams → Concepts → Documents")
    lines.append(f"  {Colors.BOLD if use_color else ''}Algorithms:{Colors.RESET if use_color else ''} TF-IDF, PageRank, Louvain Clustering, Co-occurrence")

    return "\n".join(lines)


def render_hot_files(use_color: bool = True) -> str:
    """
    Render most frequently modified files as flames.

    Hotter files = more modifications = bigger flames.
    """
    hot_files = get_hot_files(15)

    if not hot_files:
        return "No file modification history found."

    lines = []
    lines.append("")
    lines.append(f"{Colors.BOLD if use_color else ''}╔══════════════════════════════════════════════════════════════════════════╗{Colors.RESET if use_color else ''}")
    lines.append(f"{Colors.BOLD if use_color else ''}║{'HOT FILES - Most Frequently Modified':^76}║{Colors.RESET if use_color else ''}")
    lines.append(f"{Colors.BOLD if use_color else ''}╚══════════════════════════════════════════════════════════════════════════╝{Colors.RESET if use_color else ''}")
    lines.append("")

    max_count = hot_files[0][1] if hot_files else 1

    # Flame characters and colors (from cool to hot)
    flame_chars = ['░', '▒', '▓', '█', '█']
    flame_colors = [
        Colors.YELLOW,
        Colors.BRIGHT_YELLOW,
        Colors.RED,
        Colors.BRIGHT_RED,
        Colors.BRIGHT_MAGENTA
    ]

    for filename, count in hot_files:
        # Calculate flame intensity
        intensity = count / max_count
        bar_len = int(intensity * 40)

        # Build gradient flame bar
        flame = ""
        for i in range(bar_len):
            pos = i / max(bar_len, 1)
            level = min(4, int(pos * 5))
            char = flame_chars[level]
            if use_color:
                flame += f"{flame_colors[level]}{char}{Colors.RESET}"
            else:
                flame += char

        # Truncate filename
        display_name = filename[-30:] if len(filename) > 30 else filename
        display_name = f"...{display_name}" if len(filename) > 30 else display_name

        lines.append(f"  {display_name:<35} {flame} {count}")

    lines.append("")
    lines.append(f"  {Colors.DIM if use_color else ''}Based on last 500 commits{Colors.RESET if use_color else ''}")

    return "\n".join(lines)


def render_timeline(use_color: bool = True) -> str:
    """
    Render a commit timeline showing project evolution.
    """
    commits = get_commit_timeline(25)

    if not commits:
        return "No commit history found."

    lines = []
    lines.append("")
    lines.append(f"{Colors.BOLD if use_color else ''}╔══════════════════════════════════════════════════════════════════════════╗{Colors.RESET if use_color else ''}")
    lines.append(f"{Colors.BOLD if use_color else ''}║{'COMMIT TIMELINE':^76}║{Colors.RESET if use_color else ''}")
    lines.append(f"{Colors.BOLD if use_color else ''}╚══════════════════════════════════════════════════════════════════════════╝{Colors.RESET if use_color else ''}")
    lines.append("")

    # Group by date
    current_date = None

    for date, hash_short, message in commits:
        if date != current_date:
            current_date = date
            date_color = Colors.CYAN if use_color else ''
            reset = Colors.RESET if use_color else ''
            lines.append(f"  {date_color}┌── {date} ──────────────────────────────────────────────────────┐{reset}")

        # Commit line with visual connector
        hash_color = Colors.YELLOW if use_color else ''
        msg_color = Colors.WHITE if use_color else ''
        reset = Colors.RESET if use_color else ''

        # Truncate message
        max_msg_len = 55
        display_msg = message[:max_msg_len] + "..." if len(message) > max_msg_len else message

        lines.append(f"  │ {hash_color}{hash_short}{reset} ● {msg_color}{display_msg}{reset}")

    lines.append(f"  └{'─' * 72}┘")

    return "\n".join(lines)


def render_dependency_tree(use_color: bool = True) -> str:
    """
    Render module dependencies as a neural network graph.
    """
    deps = get_module_dependencies()

    if not deps:
        return "No dependencies found."

    lines = []
    lines.append("")
    lines.append(f"{Colors.BOLD if use_color else ''}╔══════════════════════════════════════════════════════════════════════════╗{Colors.RESET if use_color else ''}")
    lines.append(f"{Colors.BOLD if use_color else ''}║{'MODULE DEPENDENCY NETWORK':^76}║{Colors.RESET if use_color else ''}")
    lines.append(f"{Colors.BOLD if use_color else ''}╚══════════════════════════════════════════════════════════════════════════╝{Colors.RESET if use_color else ''}")
    lines.append("")

    # Count incoming connections for each module
    incoming = defaultdict(int)
    for source, targets in deps.items():
        for target in targets:
            incoming[target] += 1

    # Sort modules by importance (most depended upon first)
    all_modules = set(deps.keys()) | set(incoming.keys())
    sorted_modules = sorted(all_modules, key=lambda m: -incoming.get(m, 0))

    # Node colors by importance
    def get_node_color(module: str) -> str:
        if not use_color:
            return ''
        count = incoming.get(module, 0)
        if count >= 5:
            return Colors.BRIGHT_RED
        elif count >= 3:
            return Colors.YELLOW
        elif count >= 1:
            return Colors.GREEN
        return Colors.DIM

    # Render as tree-like structure
    lines.append("  Core Modules (most dependencies):")
    lines.append("")

    for module in sorted_modules[:12]:
        node_color = get_node_color(module)
        reset = Colors.RESET if use_color else ''
        dim = Colors.DIM if use_color else ''

        in_count = incoming.get(module, 0)
        out_count = len(deps.get(module, []))

        # Visual node
        node = f"[{module:^15}]"

        # Connections
        connections = deps.get(module, [])[:4]  # Limit shown connections
        conn_str = " → " + ", ".join(connections) if connections else ""

        lines.append(f"  {node_color}{node}{reset} ←{in_count} →{out_count} {dim}{conn_str}{reset}")

    lines.append("")
    lines.append(f"  {Colors.DIM if use_color else ''}←N = depended on by N modules, →N = depends on N modules{Colors.RESET if use_color else ''}")

    return "\n".join(lines)


def render_brain_art(use_color: bool = True) -> str:
    """
    Render a brain visualization with module regions.
    """
    c = Colors.CYAN if use_color else ''
    m = Colors.MAGENTA if use_color else ''
    y = Colors.YELLOW if use_color else ''
    g = Colors.GREEN if use_color else ''
    r = Colors.RESET if use_color else ''
    b = Colors.BOLD if use_color else ''
    d = Colors.DIM if use_color else ''

    brain = f"""
{b}╔══════════════════════════════════════════════════════════════════════════╗{r}
{b}║{'CORTICAL TEXT PROCESSOR - Neural Architecture':^76}║{r}
{b}╚══════════════════════════════════════════════════════════════════════════╝{r}

                            {d}┌─────────────────────┐{r}
                      {d}┌─────┤{r}    {m}DOCUMENTS L3{r}     {d}├─────┐{r}
                      {d}│     └─────────────────────┘     │{r}
                      {d}│              ↑↓                 │{r}
               {d}┌──────┴──────┐              ┌──────┴──────┐{r}
               {d}│{r}  {c}CONCEPTS{r}   {d}│{r}              {d}│{r}  {c}SEMANTICS{r}  {d}│{r}
               {d}│{r}  {c}Layer 2{r}    {d}│{r}     {d}←──→{r}     {d}│{r}  {c}Relations{r}  {d}│{r}
               {d}└──────┬──────┘{r}              {d}└──────┬──────┘{r}
                      {d}│              ↑↓                 │{r}
         {d}┌────────────┴────────────┐      ┌────────────┴────────────┐{r}
         {d}│{r}       {y}BIGRAMS L1{r}        {d}│{r}      {d}│{r}     {y}FINGERPRINTS{r}      {d}│{r}
         {d}│{r}    {y}Pattern Detection{r}   {d}│{r}      {d}│{r}   {y}Similarity Hash{r}    {d}│{r}
         {d}└────────────┬────────────┘{r}      {d}└────────────┬────────────┘{r}
                      {d}│              ↑↓                 │{r}
    {d}┌────────────────┴────────────────┐  ┌────────────────┴────────────────┐{r}
    {d}│{r}          {g}TOKENS Layer 0{r}        {d}│{r}  {d}│{r}           {g}TOKENIZER{r}            {d}│{r}
    {d}│{r}       {g}Word-level units{r}        {d}│{r}  {d}│{r}    {g}Stemming, Stop words{r}      {d}│{r}
    {d}└────────────────┬────────────────┘{r}  {d}└────────────────┬────────────────┘{r}
                      {d}│                                   │{r}
                      {d}└───────────────┬───────────────────┘{r}
                                      {d}│{r}
                              {d}┌───────┴───────┐{r}
                              {d}│{r}  {b}RAW TEXT{r}    {d}│{r}
                              {d}│{r}    {b}INPUT{r}     {d}│{r}
                              {d}└───────────────┘{r}

    {b}Key Algorithms:{r}
    ├── {g}PageRank{r}      : Term importance scoring
    ├── {y}TF-IDF{r}        : Document relevance weighting
    ├── {c}Louvain{r}       : Concept community detection
    └── {m}Co-occurrence{r} : Lateral connection strength ("Hebbian")
"""
    return brain


def render_all(use_color: bool = True) -> str:
    """Render all visualizations."""
    sections = [
        render_brain_art(use_color),
        render_skyline(use_color),
        render_layers(use_color),
        render_heatmap(use_color),
        render_hot_files(use_color),
        render_timeline(use_color),
        render_dependency_tree(use_color),
    ]

    separator = "\n" + "═" * 78 + "\n"
    return separator.join(sections)


def main():
    parser = argparse.ArgumentParser(
        description="ASCII Art Codebase Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/ascii_codebase_art.py              # All visualizations
    python scripts/ascii_codebase_art.py --skyline    # Code skyline
    python scripts/ascii_codebase_art.py --heatmap    # Git activity
    python scripts/ascii_codebase_art.py --no-color   # Without colors
        """
    )

    parser.add_argument('--skyline', action='store_true', help='Show code skyline')
    parser.add_argument('--heatmap', action='store_true', help='Show git activity heatmap')
    parser.add_argument('--layers', action='store_true', help='Show cortical layers')
    parser.add_argument('--flames', '--hot', action='store_true', help='Show hot files')
    parser.add_argument('--timeline', action='store_true', help='Show commit timeline')
    parser.add_argument('--tree', '--deps', action='store_true', help='Show dependency tree')
    parser.add_argument('--brain', action='store_true', help='Show brain architecture')
    parser.add_argument('--no-color', action='store_true', help='Disable colors')
    parser.add_argument('--all', action='store_true', help='Show all visualizations')

    args = parser.parse_args()
    use_color = not args.no_color

    # If no specific visualization requested, show brain art
    if not any([args.skyline, args.heatmap, args.layers, args.flames,
                args.timeline, args.tree, args.brain, args.all]):
        args.brain = True
        args.skyline = True
        args.heatmap = True

    output = []

    if args.all:
        print(render_all(use_color))
        return

    if args.brain:
        output.append(render_brain_art(use_color))
    if args.skyline:
        output.append(render_skyline(use_color))
    if args.layers:
        output.append(render_layers(use_color))
    if args.heatmap:
        output.append(render_heatmap(use_color))
    if args.flames:
        output.append(render_hot_files(use_color))
    if args.timeline:
        output.append(render_timeline(use_color))
    if args.tree:
        output.append(render_dependency_tree(use_color))

    separator = "\n" + "═" * 78 + "\n"
    print(separator.join(output))


if __name__ == '__main__':
    main()
