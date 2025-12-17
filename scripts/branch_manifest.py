#!/usr/bin/env python3
"""
Branch Manifest System

Tracks which files each branch is touching to detect potential conflicts
before they happen. Part of the Continuous Consciousness framework.

Usage:
    python scripts/branch_manifest.py init      # Initialize manifest for current branch
    python scripts/branch_manifest.py touch FILE [FILE...]  # Mark files as touched
    python scripts/branch_manifest.py status    # Show current branch status
    python scripts/branch_manifest.py conflicts # Check for conflicts with other branches
    python scripts/branch_manifest.py archive   # Archive manifest (on session end)
    python scripts/branch_manifest.py checkpoint [MESSAGE]  # Create checkpoint commit

Checkpoint System Design:
    Checkpoints provide automatic work-in-progress commits at regular intervals
    to ensure no work is lost. They are designed to be:

    - Automatic: Can be triggered periodically (e.g., every 5-10 minutes)
    - Reversible: Checkpoints can be squashed/reset before final merge
    - Tracked: All checkpoints are recorded in the manifest for audit trail

    Implementation Notes:
    - Full automatic checkpoint system requires background process management
      (e.g., systemd timer, cron, or daemon process)
    - For now, provides manual checkpoint creation via CLI
    - Future: Could be integrated with IDE/editor plugins or git hooks
    - Checkpoints are tagged with 'checkpoint:' prefix for easy identification
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

# Directory structure
BRANCH_STATE_DIR = Path(".branch-state")
ACTIVE_DIR = BRANCH_STATE_DIR / "active"
MERGED_DIR = BRANCH_STATE_DIR / "merged"
CONFLICTS_FILE = BRANCH_STATE_DIR / "conflicts.json"


def run_git(args: List[str], check: bool = True) -> str:
    """Run a git command and return output."""
    result = subprocess.run(
        ["git"] + args,
        capture_output=True,
        text=True,
        cwd=str(Path.cwd())
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"git {args[0]} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def get_current_branch() -> str:
    """Get the current git branch name."""
    return run_git(["rev-parse", "--abbrev-ref", "HEAD"])


def get_main_branch() -> str:
    """Detect the main branch (main or master)."""
    branches = run_git(["branch", "-l", "main", "master"], check=False)
    if "main" in branches:
        return "main"
    return "master"


def get_last_main_sync() -> Optional[str]:
    """Get timestamp of last merge from main branch."""
    main_branch = get_main_branch()
    try:
        merge_base = run_git(["merge-base", "HEAD", main_branch])
        timestamp = run_git(["log", "-1", "--format=%ci", merge_base])
        return timestamp
    except RuntimeError:
        return None


def sanitize_branch_name(branch: str) -> str:
    """Convert branch name to safe filename."""
    return branch.replace("/", "-").replace("\\", "-")


def get_manifest_path(branch: Optional[str] = None) -> Path:
    """Get path to manifest file for a branch."""
    if branch is None:
        branch = get_current_branch()
    return ACTIVE_DIR / f"{sanitize_branch_name(branch)}.json"


def load_manifest(branch: Optional[str] = None) -> Dict:
    """Load manifest for a branch, or return empty dict."""
    manifest_path = get_manifest_path(branch)
    if manifest_path.exists():
        with open(manifest_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_manifest(manifest: Dict, branch: Optional[str] = None):
    """Save manifest for a branch."""
    manifest_path = get_manifest_path(branch)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)


def init_manifest() -> Dict:
    """Initialize a new manifest for the current branch."""
    branch = get_current_branch()

    # Check if manifest already exists
    existing = load_manifest()
    if existing:
        print(f"Manifest already exists for {branch}")
        return existing

    manifest = {
        "branch": branch,
        "started": datetime.now().isoformat(),
        "files_claimed": [],
        "files_touched": [],
        "last_main_sync": get_last_main_sync(),
        "sub_agents": [],
        "session_id": None,
        "checkpoints": [],  # Track checkpoint commits
        "last_checkpoint": None,  # Timestamp of last checkpoint
    }

    save_manifest(manifest)
    print(f"✓ Initialized manifest for {branch}")
    return manifest


def touch_files(files: List[str]) -> Dict:
    """Mark files as touched by this branch."""
    manifest = load_manifest()
    if not manifest:
        manifest = init_manifest()

    touched = set(manifest.get("files_touched", []))
    new_files = []

    for f in files:
        # Normalize path
        f = str(Path(f).resolve().relative_to(Path.cwd()))
        if f not in touched:
            touched.add(f)
            new_files.append(f)

    manifest["files_touched"] = sorted(touched)
    manifest["updated"] = datetime.now().isoformat()

    save_manifest(manifest)

    if new_files:
        print(f"✓ Marked {len(new_files)} file(s) as touched")

    return manifest


def claim_files(files: List[str]) -> Dict:
    """Claim exclusive access to files (for Director sub-agent coordination)."""
    manifest = load_manifest()
    if not manifest:
        manifest = init_manifest()

    claimed = set(manifest.get("files_claimed", []))

    for f in files:
        f = str(Path(f).resolve().relative_to(Path.cwd()))
        claimed.add(f)

    manifest["files_claimed"] = sorted(claimed)
    manifest["updated"] = datetime.now().isoformat()

    save_manifest(manifest)
    print(f"✓ Claimed {len(files)} file(s)")

    return manifest


def get_all_active_manifests() -> Dict[str, Dict]:
    """Load all active branch manifests."""
    manifests = {}
    if ACTIVE_DIR.exists():
        for f in ACTIVE_DIR.glob("*.json"):
            try:
                with open(f, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                    manifests[data.get("branch", f.stem)] = data
            except (json.JSONDecodeError, IOError):
                continue
    return manifests


def check_conflicts() -> Dict[str, List[str]]:
    """Check for file conflicts between active branches."""
    current_branch = get_current_branch()
    current_manifest = load_manifest()
    if not current_manifest:
        return {}

    current_files = set(current_manifest.get("files_touched", []))
    current_files.update(current_manifest.get("files_claimed", []))

    conflicts = {}

    for branch, manifest in get_all_active_manifests().items():
        if branch == current_branch:
            continue

        other_files = set(manifest.get("files_touched", []))
        other_files.update(manifest.get("files_claimed", []))

        overlap = current_files & other_files
        if overlap:
            conflicts[branch] = sorted(overlap)

    # Update conflicts file
    if conflicts:
        BRANCH_STATE_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFLICTS_FILE, 'w', encoding='utf-8') as f:
            json.dump({
                "checked_at": datetime.now().isoformat(),
                "current_branch": current_branch,
                "conflicts": conflicts
            }, f, indent=2)

    return conflicts


def show_status():
    """Show status of current branch manifest."""
    manifest = load_manifest()
    branch = get_current_branch()

    if not manifest:
        print(f"No manifest for branch: {branch}")
        print("Run 'python scripts/branch_manifest.py init' to create one")
        return

    print(f"Branch: {manifest.get('branch', branch)}")
    print(f"Started: {manifest.get('started', 'unknown')}")
    print(f"Last main sync: {manifest.get('last_main_sync', 'unknown')}")

    # Show checkpoint information
    checkpoints = manifest.get("checkpoints", [])
    if checkpoints:
        print(f"\nCheckpoints: {len(checkpoints)}")
        print(f"Last checkpoint: {manifest.get('last_checkpoint', 'unknown')}")
        # Show most recent checkpoints
        for cp in checkpoints[-3:]:
            print(f"  📍 {cp.get('commit', '???')}: {cp.get('message', 'WIP')}")

    touched = manifest.get("files_touched", [])
    claimed = manifest.get("files_claimed", [])

    print(f"\nFiles touched: {len(touched)}")
    for f in touched[:10]:
        print(f"  • {f}")
    if len(touched) > 10:
        print(f"  ... and {len(touched) - 10} more")

    if claimed:
        print(f"\nFiles claimed: {len(claimed)}")
        for f in claimed:
            print(f"  🔒 {f}")

    # Check for conflicts
    conflicts = check_conflicts()
    if conflicts:
        print(f"\n⚠️  Potential conflicts detected!")
        for other_branch, files in conflicts.items():
            print(f"  With {other_branch}:")
            for f in files[:5]:
                print(f"    • {f}")
            if len(files) > 5:
                print(f"    ... and {len(files) - 5} more")


def create_checkpoint(message: Optional[str] = None) -> bool:
    """
    Create a checkpoint commit to save work-in-progress.

    Checkpoints are lightweight commits that:
    - Save all current changes (staged + unstaged)
    - Are tagged with 'checkpoint:' prefix for easy identification
    - Can be squashed before final merge
    - Are tracked in the manifest for audit trail

    Args:
        message: Optional description of what's being checkpointed

    Returns:
        True if checkpoint was created, False if no changes to commit

    Design Notes:
        - This is a manual checkpoint trigger via CLI
        - Full automatic checkpoint system would require:
          * Background process (daemon/timer) to trigger periodically
          * Intelligent detection of "good checkpoint moments" (e.g., after test pass)
          * Integration with editor/IDE save events
        - Future enhancements could include:
          * Configurable checkpoint interval
          * Smart checkpointing based on file types/activity
          * Checkpoint cleanup (squashing old checkpoints automatically)
    """
    # Check for uncommitted changes
    try:
        status = run_git(["status", "--porcelain"], check=False)
        if not status.strip():
            print("✓ No changes to checkpoint")
            return False
    except RuntimeError:
        print("Error: Not in a git repository")
        return False

    # Load manifest
    manifest = load_manifest()
    if not manifest:
        manifest = init_manifest()

    # Add all changes (staged + unstaged)
    run_git(["add", "-A"], check=False)

    # Create checkpoint commit
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if message:
        commit_msg = f"checkpoint: {message} ({timestamp})"
    else:
        commit_msg = f"checkpoint: WIP at {timestamp}"

    try:
        commit_sha = run_git(["commit", "-m", commit_msg, "--no-verify"], check=False)

        # Record checkpoint in manifest
        checkpoints = manifest.get("checkpoints", [])
        checkpoint_record = {
            "timestamp": datetime.now().isoformat(),
            "message": message or "WIP",
            "commit": run_git(["rev-parse", "HEAD"], check=False).strip()[:8],
        }
        checkpoints.append(checkpoint_record)

        manifest["checkpoints"] = checkpoints
        manifest["last_checkpoint"] = datetime.now().isoformat()
        manifest["updated"] = datetime.now().isoformat()

        save_manifest(manifest)

        print(f"✓ Created checkpoint: {checkpoint_record['commit']}")
        if message:
            print(f"  Message: {message}")
        print(f"  Total checkpoints: {len(checkpoints)}")

        return True

    except RuntimeError as e:
        print(f"Error creating checkpoint: {e}")
        return False


def archive_manifest():
    """Archive the current branch manifest (on session end or merge)."""
    manifest = load_manifest()
    if not manifest:
        print("No manifest to archive")
        return

    branch = get_current_branch()
    manifest["archived"] = datetime.now().isoformat()

    # Move to merged directory
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    archive_path = MERGED_DIR / f"{sanitize_branch_name(branch)}_{timestamp}.json"

    with open(archive_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    # Remove from active
    active_path = get_manifest_path()
    if active_path.exists():
        active_path.unlink()

    print(f"✓ Archived manifest to {archive_path}")


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "init":
        init_manifest()
    elif command == "touch":
        if len(sys.argv) < 3:
            print("Usage: branch_manifest.py touch FILE [FILE...]")
            sys.exit(1)
        touch_files(sys.argv[2:])
    elif command == "claim":
        if len(sys.argv) < 3:
            print("Usage: branch_manifest.py claim FILE [FILE...]")
            sys.exit(1)
        claim_files(sys.argv[2:])
    elif command == "status":
        show_status()
    elif command == "conflicts":
        conflicts = check_conflicts()
        if conflicts:
            print("⚠️  Conflicts detected:")
            for branch, files in conflicts.items():
                print(f"\n  {branch}:")
                for f in files:
                    print(f"    • {f}")
        else:
            print("✓ No conflicts with other active branches")
    elif command == "checkpoint":
        # Optional message as remaining arguments
        message = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else None
        create_checkpoint(message)
    elif command == "archive":
        archive_manifest()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
