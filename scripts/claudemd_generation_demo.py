#!/usr/bin/env python3
"""
CLAUDE.md Auto-Generation with GoT-Based Continuation Context.

This script implements the 5-layer CLAUDE.md system with dynamic Layer 4
generation from Graph of Thought (GoT) state. This enables:

- Session continuity across context windows
- Dynamic sprint/handoff awareness
- Checkpoint recovery for continuation

Layer 0 (Core): Quick Session Start - Always included
Layer 1 (Operational): Development Workflow - Always included
Layer 2 (Contextual): GoT Guide - When working on cortical/got/*
Layer 3 (Persona): ML Data Collection - For ML engineers
Layer 4 (Ephemeral): Current Session - DYNAMICALLY GENERATED from GoT

Usage:
    python scripts/claudemd_generation_demo.py [--verbose]
    python scripts/claudemd_generation_demo.py --dry-run
    python scripts/claudemd_generation_demo.py --team-demo
    python scripts/claudemd_generation_demo.py --generate-layer4   # Generate L4 from GoT
    python scripts/claudemd_generation_demo.py --continuation      # Full continuation context

Task: T-20251223-164818-f848d31d
Sprint: S-018 (Schema Evolution Foundation)
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

# Add project root to path
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cortical.got.types import ClaudeMdLayer, ClaudeMdVersion, PersonaProfile, Team
from cortical.utils.id_generation import (
    generate_claudemd_layer_id,
    generate_claudemd_version_id,
    generate_persona_profile_id,
    generate_team_id,
)


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subheader(title: str) -> None:
    """Print a formatted subsection header."""
    print()
    print(f"--- {title} ---")


# =============================================================================
# REAL CLAUDE.md CONTENT EXCERPTS
# =============================================================================

LAYER_0_QUICK_START = """## 🚀 Quick Session Start (READ THIS FIRST)

**New session? Start here to restore context fast.**

### 1. Check GoT State (30 seconds)
```bash
python scripts/got_utils.py validate      # Health check
python scripts/got_utils.py task list --status in_progress  # What's active?
```

### 2. Read Recent Knowledge Transfer (2 minutes)
```bash
ls -t samples/memories/*knowledge-transfer*.md | head -1 | xargs cat
```

### 3. Work Priority Order
1. **Security** → 2. **Bugs** → 3. **Features** → 4. **Documentation**
"""

LAYER_1_DEV_WORKFLOW = """## Development Workflow

### Before Writing Code
1. **Read the relevant module** - understand existing patterns
2. **Check existing tasks** - run `python scripts/got_utils.py task list`
3. **Run tests first** to establish baseline:
   ```bash
   python -m pytest tests/ -q
   ```

### After Writing Code
1. **Run the full test suite**
2. **Check coverage hasn't dropped** (baseline: 89%)
3. **Dog-food the feature** - test with real usage
4. **Create follow-up tasks** if issues discovered
"""

LAYER_2_GOT_GUIDE = """## Graph of Thought (GoT) Usage

GoT is our task, sprint, and decision tracking system:
- **Tasks**: `T-YYYYMMDD-HHMMSS-XXXXXXXX`
- **Sprints**: `S-NNN` (e.g., S-017)
- **Decisions**: `D-YYYYMMDD-HHMMSS-XXXXXXXX`
- **Handoffs**: `H-YYYYMMDD-HHMMSS-XXXXXXXX`

**Key Commands:**
| Command | Purpose |
|---------|---------|
| `got_utils.py dashboard` | Overview of all tasks |
| `got_utils.py task create "Title"` | Create task |
| `got_utils.py task complete T-XXX` | Mark complete |
| `got_utils.py validate` | Health check |

> ⚠️ **NEVER delete GoT files directly!** Use `got task delete` command.
"""

LAYER_3_ML_GUIDE = """## ML Data Collection Guide

**Fully automatic. Zero configuration required.**

ML data collection starts automatically when you open this project.
Every session is tracked, every commit is captured.

### What Gets Collected
| Data Type | Location | Contents |
|-----------|----------|----------|
| **Commits** | `.git-ml/commits/` | Git history with diffs |
| **Chats** | `.git-ml/chats/` | Query/response pairs |
| **Sessions** | `.git-ml/sessions/` | Development sessions |

### Quick Commands
```bash
python scripts/ml_data_collector.py stats      # Check progress
python scripts/ml_data_collector.py estimate   # Training viability
python scripts/ml_file_prediction.py train     # Train file predictor
```
"""

LAYER_4_SESSION_CONTEXT_FALLBACK = """## Current Session Context

**Note:** GoT state unavailable. Using fallback context.

### Quick Start
1. Check GoT: `python scripts/got_utils.py sprint status`
2. Check handoffs: `python scripts/got_utils.py handoff list`
3. Read recent knowledge transfer in `samples/memories/`
"""


# =============================================================================
# GOT INTEGRATION FOR DYNAMIC LAYER 4 GENERATION
# =============================================================================

def run_got_command(args: list) -> Optional[str]:
    """Run a got_utils.py command and return output."""
    try:
        result = subprocess.run(
            ["python", str(_PROJECT_ROOT / "scripts" / "got_utils.py")] + args,
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(_PROJECT_ROOT),
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def get_current_branch() -> str:
    """Get the current git branch name."""
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            cwd=str(_PROJECT_ROOT),
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def get_branch_from_handoff() -> Optional[str]:
    """Get the branch name from the most recent handoff."""
    handoffs_dir = _PROJECT_ROOT / ".got" / "entities" / "handoffs"
    if not handoffs_dir.exists():
        return None

    import json
    files = sorted(handoffs_dir.glob("H-*.json"), reverse=True)
    for f in files[:3]:  # Check recent handoffs
        try:
            with open(f) as fp:
                data = json.load(fp)
                # Look for branch in context or metadata
                context = data.get("context", {})
                if isinstance(context, dict):
                    branch = context.get("branch") or context.get("source_branch")
                    if branch:
                        return branch
        except (json.JSONDecodeError, IOError):
            continue
    return None


def get_last_sprint_branch() -> Optional[str]:
    """Get the branch from the active sprint's metadata."""
    sprint = get_sprint_status()
    if not sprint:
        return None

    sprint_id = sprint.get("id")
    if not sprint_id:
        return None

    # Check sprint entity for last known branch
    # Sprints are stored directly in .got/entities/ (not in a subdirectory)
    sprint_file = _PROJECT_ROOT / ".got" / "entities" / f"{sprint_id}.json"
    if sprint_file.exists():
        import json
        try:
            with open(sprint_file) as fp:
                data = json.load(fp)
                return data.get("metadata", {}).get("last_branch")
        except (json.JSONDecodeError, IOError):
            pass
    return None


def save_branch_to_sprint(branch: str) -> bool:
    """
    Save current branch to sprint metadata for future continuity tracking.

    This should be called when starting work on a sprint to record
    which branch the work is happening on.
    """
    sprint = get_sprint_status()
    if not sprint:
        return False

    sprint_id = sprint.get("id")
    if not sprint_id:
        return False

    # Sprints are stored directly in .got/entities/ (not in a subdirectory)
    sprint_file = _PROJECT_ROOT / ".got" / "entities" / f"{sprint_id}.json"
    if not sprint_file.exists():
        return False

    import json
    try:
        with open(sprint_file) as fp:
            data = json.load(fp)

        # Update metadata with current branch
        if "metadata" not in data:
            data["metadata"] = {}
        data["metadata"]["last_branch"] = branch
        data["metadata"]["branch_updated_at"] = datetime.now().isoformat()

        with open(sprint_file, "w") as fp:
            json.dump(data, fp, indent=2)
        return True
    except (json.JSONDecodeError, IOError):
        return False


def get_branch_state() -> dict:
    """
    Get raw branch state facts - NO determinations.

    Returns observable facts only. Agent decides what to do.
    """
    current = get_current_branch()
    saved = get_branch_from_handoff() or get_last_sprint_branch()

    # Get saved branch timestamp if available
    saved_at = None
    sprint = get_sprint_status()
    if sprint and sprint.get("id"):
        sprint_file = _PROJECT_ROOT / ".got" / "entities" / f"{sprint['id']}.json"
        if sprint_file.exists():
            import json
            try:
                with open(sprint_file) as fp:
                    data = json.load(fp)
                    saved_at = data.get("metadata", {}).get("branch_updated_at")
            except (json.JSONDecodeError, IOError):
                pass

    return {
        "current_branch": current,
        "saved_branch": saved,
        "saved_at": saved_at,
        "branches_differ": current != saved if saved else None,
    }


def get_sprint_status() -> dict:
    """Get current sprint status from GoT.

    Returns the first (most recent) sprint found in output.
    Multiple sprints may be in_progress; we take the first one listed.
    """
    output = run_got_command(["sprint", "status"])
    if not output:
        return {}

    status = {}
    for line in output.split("\n"):
        # Sprint: line indicates a new sprint block - if we already have data, stop
        if line.startswith("Sprint:") and status.get("id"):
            break
        elif line.startswith("Sprint:"):
            status["name"] = line.split(":", 1)[1].strip()
        elif line.startswith("ID:"):
            status["id"] = line.split(":", 1)[1].strip()
        elif line.startswith("Status:"):
            status["status"] = line.split(":", 1)[1].strip()
        elif line.startswith("Epic:"):
            status["epic"] = line.split(":", 1)[1].strip()
    return status


def get_sprint_tasks(sprint_id: str, limit: int = 5) -> list:
    """Get tasks for a sprint."""
    output = run_got_command(["sprint", "tasks", sprint_id])
    if not output:
        return []

    tasks = []
    for line in output.split("\n"):
        line = line.strip()
        if line.startswith("T-") or "T-" in line:
            # Parse task line format varies, extract what we can
            tasks.append(line)
        if len(tasks) >= limit:
            break
    return tasks


def get_pending_handoffs() -> list:
    """Get pending handoffs from GoT."""
    output = run_got_command(["handoff", "list", "--status", "initiated"])
    if not output or "No handoffs" in output:
        return []

    handoffs = []
    for line in output.split("\n"):
        line = line.strip()
        if line.startswith("H-") or "H-" in line:
            handoffs.append(line)
    return handoffs


def get_recent_decisions(limit: int = 3) -> list:
    """Get recent decisions from GoT."""
    # Decisions are stored in .got/entities/decisions/
    decisions_dir = _PROJECT_ROOT / ".got" / "entities" / "decisions"
    if not decisions_dir.exists():
        return []

    # Get most recent decision files
    import json
    decisions = []
    files = sorted(decisions_dir.glob("D-*.json"), reverse=True)[:limit]

    for f in files:
        try:
            with open(f) as fp:
                data = json.load(fp)
                decisions.append({
                    "id": data.get("id", f.stem),
                    "title": data.get("title", "Unknown"),
                    "created_at": data.get("created_at", ""),
                })
        except (json.JSONDecodeError, IOError):
            continue

    return decisions


def get_recent_knowledge_transfer() -> Optional[str]:
    """Get the most recent knowledge transfer file path."""
    memories_dir = _PROJECT_ROOT / "samples" / "memories"
    if not memories_dir.exists():
        return None

    # Sort by filename (not mtime) - filenames start with YYYY-MM-DD
    # so lexicographic sort gives chronological order
    kt_files = sorted(
        memories_dir.glob("*knowledge-transfer*.md"),
        reverse=True
    )

    if kt_files:
        return kt_files[0].name
    return None


def generate_layer4_from_got(verbose: bool = False) -> str:
    """
    Generate Layer 4 (Ephemeral) content from GoT state.

    Philosophy: Provide FACTS and INVESTIGATION COMMANDS.
    Agent runs commands, sees output, makes decisions.
    Script does NOT make determinations like "needs_merge".
    """
    lines = []

    # ==========================================================================
    # SECTION 1: OBSERVABLE FACTS (from GoT files)
    # ==========================================================================
    lines.append("## Observable Facts\n\n")

    # Branch state
    branch_state = get_branch_state()
    lines.append(f"**Current Branch:** `{branch_state['current_branch']}`\n")
    if branch_state["saved_branch"]:
        lines.append(f"**Saved Branch:** `{branch_state['saved_branch']}`")
        if branch_state["saved_at"]:
            lines.append(f" (saved: {branch_state['saved_at'][:19]})")
        lines.append("\n")
        if branch_state["branches_differ"]:
            lines.append("**Note:** Branches differ. Investigate before acting.\n")
    else:
        lines.append("**Saved Branch:** None (first session or no prior save)\n")

    # Sprint status
    sprint = get_sprint_status()
    if sprint:
        sprint_id = sprint.get("id", "unknown")
        sprint_name = sprint.get("name", "")
        sprint_status = sprint.get("status", "")
        lines.append(f"**Sprint:** {sprint_id}")
        if sprint_name:
            lines.append(f" ({sprint_name})")
        if sprint_status:
            lines.append(f" [{sprint_status}]")
        lines.append("\n")

        # Sprint tasks
        tasks = get_sprint_tasks(sprint_id)
        if tasks:
            lines.append("\n**Sprint Tasks:**\n")
            for task in tasks[:5]:
                lines.append(f"- {task}\n")
    else:
        lines.append("**Sprint:** None active\n")

    # Pending handoffs
    handoffs = get_pending_handoffs()
    if handoffs:
        lines.append("\n**Pending Handoffs:**\n")
        for handoff in handoffs:
            lines.append(f"- {handoff}\n")

    # Recent decisions
    decisions = get_recent_decisions(3)
    if decisions:
        lines.append("\n**Recent Decisions:**\n")
        for d in decisions:
            lines.append(f"- {d['id']}: {d['title']}\n")

    # Knowledge transfer
    kt_file = get_recent_knowledge_transfer()
    if kt_file:
        lines.append(f"\n**Latest Knowledge Transfer:** `samples/memories/{kt_file}`\n")

    # ==========================================================================
    # SECTION 2: VERIFY STATE (commands for agent to run)
    # ==========================================================================
    lines.append("\n---\n")
    lines.append("\n## Verify State\n\n")
    lines.append("Run these commands to understand current state:\n\n")
    lines.append("```bash\n")
    lines.append("# Git state\n")
    lines.append("git status\n")
    lines.append("git log --oneline -5\n")
    lines.append("\n# Divergence from main\n")
    lines.append("git log --oneline main..HEAD      # What's here but not in main\n")
    lines.append("git log --oneline HEAD..main      # What's in main but not here\n")

    # If branches differ, add commands to investigate
    if branch_state["saved_branch"] and branch_state["branches_differ"]:
        saved = branch_state["saved_branch"]
        lines.append(f"\n# Previous branch investigation\n")
        lines.append(f"git fetch origin\n")
        lines.append(f"git log --oneline origin/{saved} -5  # Commits on saved branch\n")
        lines.append(f"git log --oneline HEAD..origin/{saved}  # What's there but not here\n")

    lines.append("\n# GoT state\n")
    lines.append("python scripts/got_utils.py validate\n")
    if sprint:
        lines.append(f"python scripts/got_utils.py sprint tasks {sprint.get('id', 'SPRINT_ID')}\n")
    lines.append("```\n")

    # ==========================================================================
    # SECTION 3: QUESTIONS TO RESOLVE WITH HUMAN
    # ==========================================================================
    lines.append("\n## Questions to Resolve\n\n")
    lines.append("Before acting, ask the human:\n\n")

    # Generate contextual questions based on state
    q_num = 1

    # Branch question (only if branches differ or no saved branch)
    if branch_state["saved_branch"] and branch_state["branches_differ"]:
        lines.append(f"{q_num}. **Branch Continuity**\n")
        lines.append(f"   - Current: `{branch_state['current_branch']}`\n")
        lines.append(f"   - Saved: `{branch_state['saved_branch']}`\n")
        lines.append(f"   - *Is this a continuation? Should I pull from previous branch?*\n\n")
        q_num += 1
    elif not branch_state["saved_branch"]:
        lines.append(f"{q_num}. **Session Context**\n")
        lines.append(f"   - No saved branch found. Is this a fresh start or continuation?\n\n")
        q_num += 1

    # Sprint/task question
    if sprint:
        lines.append(f"{q_num}. **Work Focus**\n")
        lines.append(f"   - Sprint: {sprint.get('id')} ({sprint.get('name', '')})\n")
        if tasks:
            lines.append(f"   - Active tasks exist. Continue current work or switch focus?\n\n")
        else:
            lines.append(f"   - No tasks listed. What should I work on?\n\n")
        q_num += 1

    # Handoff question
    if handoffs:
        lines.append(f"{q_num}. **Pending Handoffs**\n")
        lines.append(f"   - {len(handoffs)} handoff(s) pending. Should I accept and read?\n\n")
        q_num += 1

    # Default question if nothing specific
    if q_num == 1:
        lines.append("1. What should I work on this session?\n")

    # ==========================================================================
    # SECTION 4: TRUST PROTOCOL
    # ==========================================================================
    lines.append("\n## Trust Protocol\n\n")
    lines.append("| Level | Meaning | Earned By |\n")
    lines.append("|-------|---------|----------|\n")
    lines.append("| L0 | Read-only | Default start |\n")
    lines.append("| L1 | Verified | Human confirms state summary |\n")
    lines.append("| L2 | Trusted | First task completed successfully |\n")
    lines.append("| L3 | Autonomous | Track record established |\n")
    lines.append("\n**Start at L0. Verify before acting.**\n")

    if verbose:
        lines.append("\n---\n")
        lines.append("\n### Debug Info\n")
        lines.append(f"- Branch state: {branch_state}\n")
        lines.append(f"- Sprint data: {sprint}\n")
        lines.append(f"- Handoffs found: {len(handoffs)}\n")
        lines.append(f"- Decisions found: {len(decisions)}\n")

    return "".join(lines)


def generate_continuation_context(verbose: bool = False) -> str:
    """
    Generate full continuation context for session resumption.

    Philosophy: Information + Investigation + Questions.
    No prescriptive guidance. Agent investigates and decides.
    """
    lines = ["# Continuation Context\n\n"]
    lines.append("> Facts from GoT. You investigate. You decide.\n\n")

    # Add Layer 4 content (facts + verify commands + questions)
    lines.append(generate_layer4_from_got(verbose))

    # Session ending protocol (informational, not prescriptive)
    lines.append("\n---\n")
    lines.append("\n## Session Ending (When Done)\n\n")
    lines.append("Before stopping, consider:\n\n")
    lines.append("```bash\n")
    lines.append("# Save state for next session\n")
    lines.append("git add -A && git commit -m 'checkpoint: [description]'\n")
    lines.append("git push -u origin $(git branch --show-current)\n")
    lines.append("python scripts/claudemd_generation_demo.py --save-branch\n")
    lines.append("\n# If handing off to another session\n")
    lines.append("python scripts/got_utils.py handoff initiate TASK_ID --target 'next-session' --instructions '...'\n")
    lines.append("```\n")

    lines.append("\n**Tell the human:**\n")
    lines.append("- What you were working on\n")
    lines.append("- What's done vs in-progress\n")
    lines.append("- What the next session should do\n")

    return "".join(lines)


def demo_layer_creation() -> list:
    """Demonstrate creating CLAUDE.md layers from real content."""
    print_header("1. Creating CLAUDE.md Layers (From Real CLAUDE.md)")

    layers = []

    # Layer 0: Core - Quick Session Start (always included)
    layer0 = ClaudeMdLayer(
        id=generate_claudemd_layer_id(0, "quick-session-start"),
        layer_type="core",
        layer_number=0,
        section_id="quick-session-start",
        title="Quick Session Start",
        content=LAYER_0_QUICK_START,
        freshness_decay_days=30,
        inclusion_rule="always",
    )
    layers.append(layer0)
    print(f"  ✓ Layer 0 (Core): {layer0.title}")
    print(f"    Source: CLAUDE.md lines 5-70")
    print(f"    Rule: {layer0.inclusion_rule} (every session needs this)")

    # Layer 1: Operational - Development Workflow (always included)
    layer1 = ClaudeMdLayer(
        id=generate_claudemd_layer_id(1, "dev-workflow"),
        layer_type="operational",
        layer_number=1,
        section_id="dev-workflow",
        title="Development Workflow",
        content=LAYER_1_DEV_WORKFLOW,
        freshness_decay_days=14,
        inclusion_rule="always",
    )
    layers.append(layer1)
    print(f"  ✓ Layer 1 (Operational): {layer1.title}")
    print(f"    Source: CLAUDE.md lines 1077-1258")
    print(f"    Rule: {layer1.inclusion_rule} (standard dev practices)")

    # Layer 2: Contextual - GoT Guide (only when working on GoT)
    layer2 = ClaudeMdLayer(
        id=generate_claudemd_layer_id(2, "got-usage"),
        layer_type="contextual",
        layer_number=2,
        section_id="got-usage",
        title="Graph of Thought Usage",
        content=LAYER_2_GOT_GUIDE,
        freshness_decay_days=7,
        inclusion_rule="context",
        context_modules=["cortical/got", "cortical/reasoning", ".got"],
        context_branches=["*got*", "*task*", "*sprint*"],
    )
    layers.append(layer2)
    print(f"  ✓ Layer 2 (Contextual): {layer2.title}")
    print(f"    Source: CLAUDE.md lines 21-62")
    print(f"    Rule: {layer2.inclusion_rule}")
    print(f"    When: Working on {layer2.context_modules}")

    # Layer 3: Persona - ML Data Collection (for ML engineers)
    layer3 = ClaudeMdLayer(
        id=generate_claudemd_layer_id(3, "ml-data-collection"),
        layer_type="persona",
        layer_number=3,
        section_id="ml-data-collection",
        title="ML Data Collection Guide",
        content=LAYER_3_ML_GUIDE,
        freshness_decay_days=30,
        inclusion_rule="user_pref",
        properties={"persona_ids": ["ml-engineer", "data-scientist"]},
    )
    layers.append(layer3)
    print(f"  ✓ Layer 3 (Persona): {layer3.title}")
    print(f"    Source: CLAUDE.md lines 2180-2430")
    print(f"    Rule: {layer3.inclusion_rule}")
    print(f"    For: {layer3.properties.get('persona_ids', [])}")

    # Layer 4: Ephemeral - Current Session Context (DYNAMICALLY GENERATED)
    # This is the key integration point for continuation protocol
    try:
        layer4_content = generate_layer4_from_got(verbose=False)
        layer4_source = "GoT (dynamic)"
    except Exception as e:
        layer4_content = LAYER_4_SESSION_CONTEXT_FALLBACK
        layer4_source = f"fallback (GoT error: {e})"

    layer4 = ClaudeMdLayer(
        id=generate_claudemd_layer_id(4, "session-context"),
        layer_type="ephemeral",
        layer_number=4,
        section_id="session-context",
        title="Current Session Context",
        content=layer4_content,
        freshness_decay_days=1,
        inclusion_rule="context",
        context_branches=["claude/*"],
    )
    layers.append(layer4)
    print(f"  ✓ Layer 4 (Ephemeral): {layer4.title}")
    print(f"    Source: {layer4_source}")
    print(f"    Decay: {layer4.freshness_decay_days} day (regenerate each session)")

    return layers


def demo_freshness_tracking(layers: list) -> None:
    """Demonstrate freshness tracking and staleness detection."""
    print_header("2. Freshness Tracking")

    print("  Each layer has a decay period after which it becomes stale:")
    print()
    for layer in layers:
        status = "FRESH" if not layer.is_stale() else "STALE"
        emoji = "🟢" if status == "FRESH" else "🔴"
        print(f"  {emoji} Layer {layer.layer_number} ({layer.layer_type}): {status}")
        print(f"       Decay: {layer.freshness_decay_days} days | Last: {layer.last_regenerated or 'Never'}")

    print_subheader("Simulating Layer Lifecycle")

    # Mark ephemeral layer stale (simulating day passed)
    layers[4].mark_stale("Session ended")
    print(f"  → Marked Layer 4 as stale (session ended)")
    print(f"    Status: {layers[4].freshness_status}")
    print(f"    Trigger: {layers[4].regeneration_trigger}")

    # Refresh it (simulating new session)
    layers[4].mark_fresh()
    print(f"  → Refreshed Layer 4 (new session started)")
    print(f"    Status: {layers[4].freshness_status}")
    print(f"    Timestamp: {layers[4].last_regenerated}")


def demo_versioning(layers: list) -> None:
    """Demonstrate layer versioning for audit trail."""
    print_header("3. Layer Versioning (Audit Trail)")

    layer = layers[0]

    print("  When CLAUDE.md content changes, we create version snapshots:")
    print()

    # Create version 1
    v1 = ClaudeMdVersion(
        id=generate_claudemd_version_id(layer.id, 1),
        layer_id=layer.id,
        version_number=1,
        content_snapshot=layer.content,
        change_rationale="Initial extraction from CLAUDE.md",
        changed_by="claude/auto-generate-claude-md",
    )

    print(f"  Version 1:")
    print(f"    ID: {v1.id[:50]}...")
    print(f"    Rationale: {v1.change_rationale}")
    print(f"    Content: {len(v1.content_snapshot)} chars")

    # Simulate version 2
    v2 = ClaudeMdVersion(
        id=generate_claudemd_version_id(layer.id, 2),
        layer_id=layer.id,
        version_number=2,
        content_snapshot=layer.content + "\n### 4. New Section Added\n",
        change_rationale="Added new quick-start section",
        changed_by="user-request",
        additions=2,
        deletions=0,
    )

    print(f"\n  Version 2:")
    print(f"    ID: {v2.id[:50]}...")
    print(f"    Rationale: {v2.change_rationale}")
    print(f"    Changes: +{v2.additions} -{v2.deletions} lines")


def demo_context_selection(layers: list) -> None:
    """Demonstrate context-aware layer selection."""
    print_header("4. Context-Aware Layer Selection")

    # Scenario 1: Working on GoT module
    print_subheader("Scenario: Working on cortical/got/api.py")
    context1 = {
        "branch": "claude/auto-generate-claude-md-LwdTl",
        "active_files": ["cortical/got/api.py", "cortical/got/types.py"],
        "sprint": "S-017",
        "persona": "developer",
    }

    selected1 = []
    for layer in layers:
        include = False
        reason = ""

        if layer.inclusion_rule == "always":
            include = True
            reason = "always included"
        elif layer.inclusion_rule == "context":
            # Check module context
            for ctx_mod in layer.context_modules:
                for active in context1["active_files"]:
                    if ctx_mod.replace("/*", "") in active or ctx_mod in active:
                        include = True
                        reason = f"matches {ctx_mod}"
                        break
            # Check branch context
            if not include:
                for pattern in layer.context_branches:
                    if pattern.replace("*", "") in context1["branch"]:
                        include = True
                        reason = f"branch matches {pattern}"
                        break

        status = "✓ INCLUDE" if include else "✗ SKIP"
        print(f"    Layer {layer.layer_number} ({layer.section_id}): {status}")
        if reason:
            print(f"      Reason: {reason}")
        if include:
            selected1.append(layer)

    print(f"\n    Result: {len(selected1)} layers selected for this context")

    # Scenario 2: Working on ML module
    print_subheader("Scenario: ML Engineer working on spark/ngram.py")
    context2 = {
        "branch": "feature/ml-training",
        "active_files": ["cortical/spark/ngram.py"],
        "persona": "ml-engineer",
    }

    selected2 = []
    for layer in layers:
        include = False

        if layer.inclusion_rule == "always":
            include = True
        elif layer.inclusion_rule == "user_pref":
            # Check persona
            persona_ids = layer.properties.get("persona_ids", [])
            if context2["persona"] in persona_ids:
                include = True

        if include:
            selected2.append(layer)
            print(f"    ✓ Layer {layer.layer_number}: {layer.title}")

    print(f"\n    Result: {len(selected2)} layers (includes ML guide for ml-engineer)")


def demo_persona_profiles() -> None:
    """Demonstrate PersonaProfile for role-based customization."""
    print_header("5. PersonaProfile (Role-Based Customization)")

    # Create personas matching our project
    senior_dev = PersonaProfile(
        id=generate_persona_profile_id(),
        name="Senior Developer",
        role="developer",
        layer_preferences={
            "quick-session-start": True,
            "dev-workflow": True,
            "got-usage": True,
            "ml-data-collection": False,  # Not their focus
            "architecture": True,
        },
        excluded_layers=["marketing-style", "branding-guide"],
    )

    ml_engineer = PersonaProfile(
        id=generate_persona_profile_id(),
        name="ML Platform Engineer",
        role="developer",
        inherits_from=senior_dev.id,
        layer_preferences={
            "ml-data-collection": True,  # Override parent
            "spark-slm": True,
        },
    )

    qa_engineer = PersonaProfile(
        id=generate_persona_profile_id(),
        name="QA Engineer",
        role="qa",
        layer_preferences={
            "quick-session-start": True,
            "testing-patterns": True,
            "regression-tests": True,
        },
        custom_layers=["qa-checklist", "test-coverage-report"],
    )

    print("  Project Personas:")
    print()
    for persona in [senior_dev, ml_engineer, qa_engineer]:
        print(f"  📋 {persona.name} ({persona.role})")
        print(f"     ID: {persona.id}")
        if persona.inherits_from:
            print(f"     Inherits from: {persona.inherits_from[:30]}...")
        prefs = [k for k, v in persona.layer_preferences.items() if v]
        print(f"     Includes: {', '.join(prefs[:3])}{'...' if len(prefs) > 3 else ''}")
        print()

    print_subheader("Layer Decisions for ML Engineer")
    test_layers = ["quick-session-start", "ml-data-collection", "marketing-style", "qa-checklist"]
    for layer in test_layers:
        include = ml_engineer.should_include_layer(layer)
        status = "✓ INCLUDE" if include else "✗ EXCLUDE"
        print(f"    {layer}: {status}")


def demo_team_hierarchy() -> None:
    """Demonstrate Team entity for SDLC pipeline support."""
    print_header("6. Team Hierarchy (SDLC Pipeline Support)")

    # Create teams matching potential SDLC pipeline
    engineering = Team(
        id=generate_team_id(),
        name="Engineering",
        description="Core development team",
        branch_patterns=["main", "develop", "feature/*", "bugfix/*"],
        module_scope=["cortical"],
        settings={
            "min_coverage": 89,
            "require_tests": True,
            "knowledge_domains": ["architecture", "algorithms", "testing"],
        },
    )

    ml_platform = Team(
        id=generate_team_id(),
        name="ML Platform",
        parent_team_id=engineering.id,
        description="ML/AI feature development",
        branch_patterns=["feature/ml-*", "feature/spark-*"],
        module_scope=["cortical/spark", "cortical/reasoning", ".git-ml"],
        settings={"knowledge_domains": ["ml-training", "data-pipelines"]},
    )

    qa_team = Team(
        id=generate_team_id(),
        name="QA",
        description="Quality assurance",
        branch_patterns=["qa/*", "release/*"],
        module_scope=["tests"],
        settings={
            "focus": "regression-testing",
            "knowledge_domains": ["test-patterns", "coverage"],
        },
    )

    print("  SDLC Team Structure:")
    print()
    print(f"  🏢 {engineering.name}")
    print(f"     Branches: {', '.join(engineering.branch_patterns)}")
    print(f"     Scope: {engineering.module_scope}")
    print()
    print(f"     └── 🤖 {ml_platform.name} (sub-team)")
    print(f"         Branches: {', '.join(ml_platform.branch_patterns)}")
    print(f"         Scope: {ml_platform.module_scope}")
    print()
    print(f"  🧪 {qa_team.name}")
    print(f"     Branches: {', '.join(qa_team.branch_patterns)}")
    print(f"     Scope: {qa_team.module_scope}")

    print_subheader("Branch → Team Mapping")
    test_branches = [
        "feature/ml-training",
        "feature/new-search",
        "qa/release-v2",
        "main",
    ]
    for branch in test_branches:
        matches = []
        if engineering.matches_branch(branch):
            matches.append("Engineering")
        if ml_platform.matches_branch(branch):
            matches.append("ML Platform")
        if qa_team.matches_branch(branch):
            matches.append("QA")

        teams_str = ", ".join(matches) if matches else "(no match)"
        print(f"    {branch} → {teams_str}")


def demo_generation_pipeline(layers: list) -> None:
    """Demonstrate the full generation pipeline."""
    print_header("7. Generation Pipeline")

    print("  The pipeline composes CLAUDE.md from selected layers:")
    print()
    print("    ┌─────────────────────────────────────────────────┐")
    print("    │  1. CONTEXT ANALYSIS                            │")
    print("    │     → Detect branch, sprint, active modules     │")
    print("    ├─────────────────────────────────────────────────┤")
    print("    │  2. LAYER SELECTION                             │")
    print("    │     → Filter by context + persona preferences   │")
    print("    ├─────────────────────────────────────────────────┤")
    print("    │  3. CONTENT COMPOSITION                         │")
    print("    │     → Assemble in layer order (L0→L4)           │")
    print("    ├─────────────────────────────────────────────────┤")
    print("    │  4. VALIDATION                                  │")
    print("    │     → Verify required sections present          │")
    print("    ├─────────────────────────────────────────────────┤")
    print("    │  5. OUTPUT                                      │")
    print("    │     → Write with backup (atomic + fallback)     │")
    print("    └─────────────────────────────────────────────────┘")

    print_subheader("Simulated Generation")

    # Current context
    context = {
        "branch": "claude/auto-generate-claude-md-LwdTl",
        "sprint": "S-017",
        "modules": ["cortical/got", "scripts"],
        "persona": "developer",
    }

    print(f"    Context: branch={context['branch']}")
    print(f"             sprint={context['sprint']}")
    print(f"             modules={context['modules']}")
    print()

    # Select layers
    selected = [l for l in layers if l.inclusion_rule == "always"]
    # Add GoT layer (context match)
    selected.append(layers[2])  # GoT guide
    selected.append(layers[4])  # Session context

    print("    Selected Layers:")
    total_lines = 0
    for layer in sorted(selected, key=lambda x: x.layer_number):
        lines = len(layer.content.strip().split('\n'))
        total_lines += lines
        print(f"      L{layer.layer_number}: {layer.title} ({lines} lines)")

    print()
    print(f"    📄 Composed output: {total_lines} lines from {len(selected)} layers")
    print(f"    💾 Original CLAUDE.md preserved as fallback")


def demo_fault_tolerance() -> None:
    """Demonstrate fault tolerance mechanisms."""
    print_header("8. Fault Tolerance & Recovery")

    print("  The system is designed to never break your workflow:")
    print()
    print("  🛡️  PROTECTION MECHANISMS:")
    print("      • Atomic writes (temp file → rename)")
    print("      • Automatic backup before any changes")
    print("      • Checksum verification on read")
    print("      • Validation before write")
    print()
    print("  🔄 FALLBACK CHAIN:")
    print("      1. Try generated CLAUDE.md")
    print("      2. If invalid → use backup")
    print("      3. If no backup → use original CLAUDE.md")
    print("      4. Original CLAUDE.md is NEVER modified")
    print()

    print_subheader("Recovery Scenarios")
    scenarios = [
        ("Layer file corrupted", "Skip layer, log warning, continue with others"),
        ("Required section missing", "Fall back to original CLAUDE.md"),
        ("Write fails mid-operation", "Restore from automatic backup"),
        ("GoT database unavailable", "Use cached layer content"),
    ]
    for scenario, response in scenarios:
        print(f"    ⚠️  {scenario}")
        print(f"       → {response}")
        print()


def main():
    """Run the CLAUDE.md generation demo."""
    import argparse

    parser = argparse.ArgumentParser(
        description="CLAUDE.md Auto-Generation with GoT-Based Continuation Context"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose output")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be generated")
    parser.add_argument("--team-demo", action="store_true", help="Focus on team/persona features")
    parser.add_argument(
        "--generate-layer4",
        action="store_true",
        help="Generate Layer 4 (Ephemeral) content from GoT and print it"
    )
    parser.add_argument(
        "--continuation",
        action="store_true",
        help="Generate full continuation context for session resumption"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path (default: stdout)"
    )
    parser.add_argument(
        "--save-branch",
        action="store_true",
        help="Save current branch to sprint metadata for continuity tracking"
    )
    parser.add_argument(
        "--check-continuity",
        action="store_true",
        help="Check branch continuity status and show guidance"
    )
    args = parser.parse_args()

    # Branch management
    if args.save_branch:
        branch = get_current_branch()
        if save_branch_to_sprint(branch):
            print(f"✓ Saved branch `{branch}` to active sprint metadata")
        else:
            print(f"✗ Failed to save branch (no active sprint or write error)")
        return

    if args.check_continuity:
        state = get_branch_state()
        print("## Branch State (Facts)")
        print(f"Current: {state['current_branch']}")
        print(f"Saved:   {state['saved_branch'] or 'None'}")
        if state['saved_at']:
            print(f"Saved at: {state['saved_at']}")
        if state['branches_differ']:
            print("\nBranches differ. Run these to investigate:")
            print(f"  git log --oneline origin/{state['saved_branch']} -5")
            print(f"  git log --oneline HEAD..origin/{state['saved_branch']}")
        elif state['saved_branch']:
            print("\nSame branch as saved.")
        else:
            print("\nNo saved branch. First session or no prior save.")
        return

    # Quick generation modes (no demo, just output)
    if args.generate_layer4:
        content = generate_layer4_from_got(verbose=args.verbose)
        if args.output:
            Path(args.output).write_text(content)
            print(f"Layer 4 written to: {args.output}")
        else:
            print(content)
        return

    if args.continuation:
        content = generate_continuation_context(verbose=args.verbose)
        if args.output:
            Path(args.output).write_text(content)
            print(f"Continuation context written to: {args.output}")
        else:
            print(content)
        return

    # Demo mode
    print()
    print("╔" + "═" * 68 + "╗")
    print("║  CLAUDE.md Auto-Generation with GoT-Based Continuation            ║")
    print("║  Cortical Text Processor - Dynamic Layer 4 from GoT State         ║")
    print("╚" + "═" * 68 + "╝")

    if args.team_demo:
        # Focus on multi-team features
        demo_persona_profiles()
        demo_team_hierarchy()
    else:
        # Full demo
        layers = demo_layer_creation()
        demo_freshness_tracking(layers)
        demo_versioning(layers)
        demo_context_selection(layers)
        demo_persona_profiles()
        demo_team_hierarchy()
        demo_generation_pipeline(layers)
        demo_fault_tolerance()

    print_header("Summary")
    print("  The CLAUDE.md auto-generation system provides:")
    print()
    print("  📚 5-Layer Architecture:")
    print("     L0 Core       → Quick Session Start (always)")
    print("     L1 Operational → Dev Workflow (always)")
    print("     L2 Contextual → GoT Guide (when working on GoT)")
    print("     L3 Persona    → ML Guide (for ML engineers)")
    print("     L4 Ephemeral  → Session Context (DYNAMIC from GoT)")
    print()
    print("  🔄 GoT-Based Continuation:")
    print("     • Layer 4 generated dynamically from GoT state")
    print("     • Sprint status, handoffs, decisions included")
    print("     • Trust protocol reminder embedded")
    print("     • Fallback to static content if GoT unavailable")
    print()
    print("  🎯 Key Features:")
    print("     • Context-aware layer selection")
    print("     • Freshness tracking with decay")
    print("     • PersonaProfile for role customization")
    print("     • Team hierarchy for SDLC pipelines")
    print("     • Fault-tolerant with fallback chain")
    print()
    print("  🚀 Quick Commands:")
    print("     python scripts/claudemd_generation_demo.py --generate-layer4")
    print("     python scripts/claudemd_generation_demo.py --continuation")
    print("     python scripts/claudemd_generation_demo.py --continuation -o context.md")
    print()
    print("  📖 Documentation:")
    print("     • .claude/commands/project:continuation.md (v2.0)")
    print("     • docs/architecture/GOT_DATABASE_ARCHITECTURE.md")
    print("     • cortical/got/types.py")
    print()


if __name__ == "__main__":
    main()
