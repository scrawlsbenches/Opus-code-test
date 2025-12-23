#!/usr/bin/env python3
"""
Capture and restore diffs for sub-agent task delegation.

Usage:
    # Capture current changes and associate with a task
    python scripts/task_diff.py capture TASK_ID

    # Restore changes from a saved diff
    python scripts/task_diff.py restore TASK_ID

    # List saved diffs
    python scripts/task_diff.py list

    # Show a saved diff
    python scripts/task_diff.py show TASK_ID

This helps recover work when sub-agent changes don't persist.
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
_DIFFS_DIR = _PROJECT_ROOT / ".got" / "diffs"


def run_git(args: list) -> tuple:
    """Run git command and return (stdout, stderr, returncode)."""
    result = subprocess.run(
        ["git"] + args,
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True
    )
    return result.stdout, result.stderr, result.returncode


def capture_diff(task_id: str, message: str = None) -> bool:
    """Capture current git diff and associate with task."""
    _DIFFS_DIR.mkdir(parents=True, exist_ok=True)

    # Get both staged and unstaged changes
    stdout_staged, _, _ = run_git(["diff", "--cached"])
    stdout_unstaged, _, _ = run_git(["diff"])

    combined_diff = ""
    if stdout_staged:
        combined_diff += "# === STAGED CHANGES ===\n" + stdout_staged
    if stdout_unstaged:
        if combined_diff:
            combined_diff += "\n"
        combined_diff += "# === UNSTAGED CHANGES ===\n" + stdout_unstaged

    if not combined_diff.strip():
        print(f"No changes to capture for {task_id}")
        return False

    # Get list of changed files
    files_staged, _, _ = run_git(["diff", "--cached", "--name-only"])
    files_unstaged, _, _ = run_git(["diff", "--name-only"])
    all_files = set(files_staged.strip().split('\n') + files_unstaged.strip().split('\n'))
    all_files.discard('')

    # Create diff metadata
    metadata = {
        "task_id": task_id,
        "captured_at": datetime.now().isoformat(),
        "message": message or f"Diff capture for {task_id}",
        "files": sorted(all_files),
        "stats": {
            "files_changed": len(all_files),
            "staged_lines": len(stdout_staged.split('\n')) if stdout_staged else 0,
            "unstaged_lines": len(stdout_unstaged.split('\n')) if stdout_unstaged else 0,
        }
    }

    # Save diff file
    diff_file = _DIFFS_DIR / f"{task_id}.patch"
    diff_file.write_text(combined_diff)

    # Save metadata
    meta_file = _DIFFS_DIR / f"{task_id}.json"
    meta_file.write_text(json.dumps(metadata, indent=2))

    print(f"Captured diff for {task_id}")
    print(f"  Files: {', '.join(all_files)}")
    print(f"  Saved to: {diff_file}")
    return True


def restore_diff(task_id: str, dry_run: bool = False) -> bool:
    """Restore changes from a saved diff."""
    diff_file = _DIFFS_DIR / f"{task_id}.patch"
    meta_file = _DIFFS_DIR / f"{task_id}.json"

    if not diff_file.exists():
        print(f"No diff found for {task_id}")
        return False

    # Load metadata
    if meta_file.exists():
        metadata = json.loads(meta_file.read_text())
        print(f"Restoring diff from {metadata.get('captured_at', 'unknown')}")
        print(f"Files: {', '.join(metadata.get('files', []))}")

    if dry_run:
        print("\n[DRY RUN] Would apply:")
        print(diff_file.read_text()[:500])
        if diff_file.stat().st_size > 500:
            print("... (truncated)")
        return True

    # Apply the patch
    # Note: We apply with --reject to handle partial failures gracefully
    diff_content = diff_file.read_text()

    # Split into staged and unstaged if markers present
    if "# === STAGED CHANGES ===" in diff_content or "# === UNSTAGED CHANGES ===" in diff_content:
        # Apply without the comment markers
        clean_diff = '\n'.join(
            line for line in diff_content.split('\n')
            if not line.startswith('# ===')
        )

        # Write to temp file and apply
        temp_patch = _DIFFS_DIR / f"{task_id}.temp.patch"
        temp_patch.write_text(clean_diff)

        stdout, stderr, code = run_git(["apply", "--reject", "--whitespace=fix", str(temp_patch)])
        temp_patch.unlink()
    else:
        stdout, stderr, code = run_git(["apply", "--reject", "--whitespace=fix", str(diff_file)])

    if code == 0:
        print(f"Successfully restored diff for {task_id}")
        return True
    else:
        print(f"Partial restore for {task_id}")
        print(f"Some hunks may have failed: {stderr}")
        return False


def list_diffs() -> None:
    """List all saved diffs."""
    if not _DIFFS_DIR.exists():
        print("No diffs directory found")
        return

    diffs = sorted(_DIFFS_DIR.glob("*.json"))
    if not diffs:
        print("No saved diffs found")
        return

    print(f"{'Task ID':<40} {'Captured':<20} {'Files':<10}")
    print("-" * 70)

    for meta_file in diffs:
        try:
            metadata = json.loads(meta_file.read_text())
            task_id = metadata.get("task_id", meta_file.stem)
            captured = metadata.get("captured_at", "unknown")[:19]
            files = len(metadata.get("files", []))
            print(f"{task_id:<40} {captured:<20} {files:<10}")
        except json.JSONDecodeError:
            print(f"{meta_file.stem:<40} {'(invalid json)':<20} {'?':<10}")


def show_diff(task_id: str) -> None:
    """Show a saved diff."""
    diff_file = _DIFFS_DIR / f"{task_id}.patch"
    meta_file = _DIFFS_DIR / f"{task_id}.json"

    if not diff_file.exists():
        print(f"No diff found for {task_id}")
        return

    if meta_file.exists():
        metadata = json.loads(meta_file.read_text())
        print(f"Task: {metadata.get('task_id')}")
        print(f"Captured: {metadata.get('captured_at')}")
        print(f"Message: {metadata.get('message')}")
        print(f"Files: {', '.join(metadata.get('files', []))}")
        print("-" * 60)

    print(diff_file.read_text())


def main():
    parser = argparse.ArgumentParser(
        description="Capture and restore diffs for sub-agent task delegation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # capture
    capture_parser = subparsers.add_parser("capture", help="Capture current diff")
    capture_parser.add_argument("task_id", help="Task ID to associate with diff")
    capture_parser.add_argument("-m", "--message", help="Optional message")

    # restore
    restore_parser = subparsers.add_parser("restore", help="Restore saved diff")
    restore_parser.add_argument("task_id", help="Task ID to restore")
    restore_parser.add_argument("--dry-run", action="store_true", help="Show what would be applied")

    # list
    subparsers.add_parser("list", help="List saved diffs")

    # show
    show_parser = subparsers.add_parser("show", help="Show a saved diff")
    show_parser.add_argument("task_id", help="Task ID to show")

    args = parser.parse_args()

    if args.command == "capture":
        success = capture_diff(args.task_id, args.message)
        sys.exit(0 if success else 1)
    elif args.command == "restore":
        success = restore_diff(args.task_id, args.dry_run)
        sys.exit(0 if success else 1)
    elif args.command == "list":
        list_diffs()
    elif args.command == "show":
        show_diff(args.task_id)


if __name__ == "__main__":
    main()
