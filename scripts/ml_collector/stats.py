"""
Statistics module for ML Data Collector

Handles data counting, size calculation, and progress estimation.
"""

from pathlib import Path
from typing import Dict

from .config import (
    COMMITS_DIR, COMMITS_LITE_DIR, COMMITS_LITE_FILE, SESSIONS_LITE_FILE,
    SESSIONS_DIR, CHATS_DIR, ACTIONS_DIR, GITHUB_DIR, MILESTONES
)
from .persistence import ensure_dirs


def count_jsonl_lines(filepath: Path) -> int:
    """Count non-empty lines in a JSONL file."""
    if not filepath.exists():
        return 0
    count = 0
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def count_data() -> Dict[str, int]:
    """Count collected data entries."""
    ensure_dirs()

    counts = {
        "commits": 0,
        "commits_lite": 0,
        "sessions_lite": 0,
        "chats": 0,
        "actions": 0,
        "sessions": 0,
        "prs": 0,
        "issues": 0,
    }

    # Count commits (full)
    if COMMITS_DIR.exists():
        counts["commits"] = len(list(COMMITS_DIR.glob("*.json")))

    # Count commits (lightweight - from JSONL)
    counts["commits_lite"] = count_jsonl_lines(COMMITS_LITE_FILE)
    # Also count legacy individual files if they exist
    if COMMITS_LITE_DIR.exists():
        counts["commits_lite"] += len(list(COMMITS_LITE_DIR.glob("*.json")))

    # Count sessions (lightweight - from JSONL)
    counts["sessions_lite"] = count_jsonl_lines(SESSIONS_LITE_FILE)

    # Count chats
    if CHATS_DIR.exists():
        counts["chats"] = len(list(CHATS_DIR.glob("**/*.json")))

    # Count actions
    if ACTIONS_DIR.exists():
        counts["actions"] = len(list(ACTIONS_DIR.glob("**/*.json")))

    # Count sessions
    if SESSIONS_DIR.exists():
        counts["sessions"] = len(list(SESSIONS_DIR.glob("*.json")))

    # Count GitHub data
    if GITHUB_DIR.exists():
        counts["prs"] = len(list(GITHUB_DIR.glob("pr_*.json")))
        counts["issues"] = len(list(GITHUB_DIR.glob("issue_*.json")))

    return counts


def calculate_data_size() -> Dict[str, int]:
    """Calculate total size of collected data."""
    ensure_dirs()

    sizes = {}
    for name, dir_path in [
        ("commits", COMMITS_DIR),
        ("chats", CHATS_DIR),
        ("actions", ACTIONS_DIR),
        ("sessions", SESSIONS_DIR),
    ]:
        total = 0
        if dir_path.exists():
            for f in dir_path.glob("**/*.json"):
                total += f.stat().st_size
        sizes[name] = total

    sizes["total"] = sum(sizes.values())
    return sizes


def estimate_progress() -> Dict[str, Dict]:
    """Estimate progress toward training milestones."""
    counts = count_data()

    progress = {}
    for milestone, requirements in MILESTONES.items():
        milestone_progress = {}
        for data_type, required in requirements.items():
            current = counts.get(data_type, 0)
            milestone_progress[data_type] = {
                "current": current,
                "required": required,
                "percent": min(100, int(100 * current / required)),
            }

        # Overall milestone progress (minimum of all types)
        overall = min(p["percent"] for p in milestone_progress.values())
        milestone_progress["overall"] = overall
        progress[milestone] = milestone_progress

    return progress


def print_stats():
    """Print collection statistics."""
    counts = count_data()
    sizes = calculate_data_size()
    progress = estimate_progress()

    print("\n" + "=" * 60)
    print("ML DATA COLLECTION STATISTICS")
    print("=" * 60)

    print("\n📊 Data Counts:")
    print(f"   Commits (full):  {counts['commits']:,}")
    print(f"   Commits (lite):  {counts.get('commits_lite', 0):,}  ← tracked in git")
    print(f"   Sessions (lite): {counts.get('sessions_lite', 0):,}  ← tracked in git")
    print(f"   Chats:           {counts['chats']:,}")
    print(f"   Actions:         {counts['actions']:,}")
    print(f"   Sessions (full): {counts['sessions']:,}")
    if counts.get('prs', 0) > 0 or counts.get('issues', 0) > 0:
        print(f"   PRs:             {counts.get('prs', 0):,}")
        print(f"   Issues:          {counts.get('issues', 0):,}")

    print("\n💾 Data Sizes:")
    for name, size in sizes.items():
        if size > 1024 * 1024:
            print(f"   {name.capitalize():10s}: {size / 1024 / 1024:.2f} MB")
        elif size > 1024:
            print(f"   {name.capitalize():10s}: {size / 1024:.2f} KB")
        else:
            print(f"   {name.capitalize():10s}: {size} bytes")

    print("\n🎯 Training Milestones:")
    for milestone, data in progress.items():
        overall = data.pop("overall")
        bar = "█" * (overall // 5) + "░" * (20 - overall // 5)
        print(f"\n   {milestone.replace('_', ' ').title()}: [{bar}] {overall}%")
        for data_type, info in data.items():
            print(f"      {data_type}: {info['current']}/{info['required']}")

    print("\n" + "=" * 60)


def estimate_project_size():
    """Estimate final project size when all milestones are reached."""
    # Current averages
    sizes = calculate_data_size()
    counts = count_data()

    # Calculate average sizes per entry type
    avg_commit_size = sizes["commits"] / max(1, counts["commits"])
    avg_chat_size = sizes["chats"] / max(1, counts["chats"]) if counts["chats"] > 0 else 2000  # estimate 2KB per chat
    avg_action_size = sizes["actions"] / max(1, counts["actions"]) if counts["actions"] > 0 else 500  # estimate 500B per action

    # Target for "code_suggestions" milestone (the highest)
    target_commits = MILESTONES["code_suggestions"]["commits"]
    target_chats = MILESTONES["code_suggestions"]["chats"]
    target_actions = target_chats * 10  # Estimate 10 actions per chat

    estimated_total = (
        target_commits * max(avg_commit_size, 5000) +  # ~5KB per commit if no data
        target_chats * max(avg_chat_size, 2000) +
        target_actions * max(avg_action_size, 500)
    )

    print("\n" + "=" * 60)
    print("PROJECT SIZE ESTIMATE (Full Collection)")
    print("=" * 60)

    print("\n📈 Target Data Points:")
    print(f"   Commits:  {target_commits:,}")
    print(f"   Chats:    {target_chats:,}")
    print(f"   Actions:  {target_actions:,} (estimated)")

    print("\n💾 Estimated Sizes:")
    print(f"   Commits data:  {target_commits * max(avg_commit_size, 5000) / 1024 / 1024:.1f} MB")
    print(f"   Chats data:    {target_chats * max(avg_chat_size, 2000) / 1024 / 1024:.1f} MB")
    print(f"   Actions data:  {target_actions * max(avg_action_size, 500) / 1024 / 1024:.1f} MB")
    print(f"   ─────────────────────────")
    print(f"   TOTAL:         {estimated_total / 1024 / 1024:.1f} MB")

    print("\n🧠 Model Training Estimates:")
    print(f"   Vocabulary size:     ~15,000 tokens (this project)")
    print(f"   Training examples:   ~{target_commits + target_chats:,}")
    print(f"   Micro-model size:    1-10 MB (1-10M parameters)")
    print(f"   Training time:       ~1-4 hours (single GPU)")
    print(f"   Inference:           <100ms on CPU")

    print("\n⏱️  Time to Collection Complete:")
    counts = count_data()
    commits_per_day = 20  # Based on current rate
    chats_per_day = 15  # Estimated

    days_for_commits = (target_commits - counts["commits"]) / commits_per_day
    days_for_chats = (target_chats - counts["chats"]) / chats_per_day

    days_needed = max(days_for_commits, days_for_chats)
    print(f"   At current rate:     ~{int(days_needed)} days ({int(days_needed/30)} months)")
    print(f"   With active use:     ~{int(days_needed * 0.5)} days (more chatting)")

    print("\n" + "=" * 60)
