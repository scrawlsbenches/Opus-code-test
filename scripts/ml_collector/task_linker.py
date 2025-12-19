"""
Task Linker - Post-Commit Task Linking for ML Data Collection

Automatically updates task status and links commits when commit messages
reference task IDs.

Task ID Pattern: T-YYYYMMDD-HHMMSS-XXXX-NNN

Keywords:
- Completion: "completes", "fixes", "closes" → mark task complete
- Reference: "refs", "see", "related" → add commit link only

Examples:
    git commit -m "feat: Add authentication - completes T-20251213-143052-a1b2-001"
    git commit -m "refactor: Update tests - refs T-20251213-143052-a1b2-002"
    git commit -m "fix: Bug in login - fixes T-20251213-143052-a1b2-003"
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Task ID pattern with optional task number
# Matches: T-YYYYMMDD-HHMMSS-XXXX or T-YYYYMMDD-HHMMSS-XXXX-NNN
TASK_ID_PATTERN = re.compile(
    r'\bT-\d{8}-\d{6,14}-[a-f0-9]{4}(?:-\d{3})?\b',
    re.IGNORECASE
)

# Completion keywords that mark tasks as complete
COMPLETION_KEYWORDS = {
    'completes', 'complete',
    'fixes', 'fix', 'fixed',
    'closes', 'close', 'closed',
    'resolves', 'resolve', 'resolved',
    'implements', 'implement', 'implemented'
}

# Reference keywords that only link commits
REFERENCE_KEYWORDS = {
    'refs', 'ref', 'references', 'reference',
    'see', 'sees',
    'related', 'relates',
    'addresses', 'address',
    'updates', 'update',
    'touches', 'touch',
    'part-of', 'partial'
}


def find_task_ids(message: str) -> List[str]:
    """
    Extract all task IDs from a commit message.

    Args:
        message: Commit message text

    Returns:
        List of task IDs found (may be empty)

    Examples:
        >>> find_task_ids("feat: Add auth - completes T-20251213-143052-a1b2-001")
        ['T-20251213-143052-a1b2-001']
        >>> find_task_ids("See T-20251213-143052-a1b2-001 and T-20251213-143052-a1b2-002")
        ['T-20251213-143052-a1b2-001', 'T-20251213-143052-a1b2-002']
    """
    matches = TASK_ID_PATTERN.findall(message)
    # Normalize to uppercase T prefix
    return [m.upper() if m[0].islower() else m for m in matches]


def classify_task_references(message: str, task_ids: List[str]) -> Dict[str, Set[str]]:
    """
    Classify task IDs by whether they should be completed or just referenced.

    Args:
        message: Commit message text
        task_ids: List of task IDs found in the message

    Returns:
        Dictionary with 'complete' and 'reference' sets of task IDs

    Examples:
        >>> classify_task_references("completes T-20251213-143052-a1b2-001", ["T-20251213-143052-a1b2-001"])
        {'complete': {'T-20251213-143052-a1b2-001'}, 'reference': set()}
        >>> classify_task_references("refs T-20251213-143052-a1b2-001", ["T-20251213-143052-a1b2-001"])
        {'complete': set(), 'reference': {'T-20251213-143052-a1b2-001'}}
    """
    result = {'complete': set(), 'reference': set()}

    # Convert to lowercase for keyword matching
    message_lower = message.lower()

    for task_id in task_ids:
        # Find the position of this task ID
        task_pos = message.find(task_id)
        if task_pos == -1:
            task_pos = message.lower().find(task_id.lower())

        # Look for keywords before the task ID (within 50 chars)
        context_start = max(0, task_pos - 50)
        context = message_lower[context_start:task_pos]

        # Check for completion keywords
        found_completion = False
        for keyword in COMPLETION_KEYWORDS:
            if keyword in context:
                result['complete'].add(task_id)
                found_completion = True
                break

        # If no completion keyword, check for reference keywords
        if not found_completion:
            found_reference = False
            for keyword in REFERENCE_KEYWORDS:
                if keyword in context:
                    result['reference'].add(task_id)
                    found_reference = True
                    break

            # If no keyword found at all, treat as reference
            if not found_reference:
                result['reference'].add(task_id)

    return result


def find_task_session_file(task_id: str, tasks_dir: str = "tasks") -> Optional[Path]:
    """
    Find the session file containing a given task ID.

    Args:
        task_id: Task ID to search for
        tasks_dir: Directory containing task session files

    Returns:
        Path to session file, or None if not found
    """
    dir_path = Path(tasks_dir)
    if not dir_path.exists():
        return None

    for filepath in sorted(dir_path.glob("*.json")):
        try:
            with open(filepath) as f:
                data = json.load(f)

            # Check if this task is in this session
            for task in data.get('tasks', []):
                if task.get('id') == task_id:
                    return filepath
        except (json.JSONDecodeError, KeyError, OSError):
            continue

    return None


def update_task_with_commit(
    task_id: str,
    commit_hash: str,
    should_complete: bool,
    tasks_dir: str = "tasks"
) -> bool:
    """
    Update a task with a commit reference and optionally mark it complete.

    Args:
        task_id: Task ID to update
        commit_hash: Git commit SHA
        should_complete: Whether to mark the task as completed
        tasks_dir: Directory containing task session files

    Returns:
        True if task was updated successfully, False otherwise
    """
    # Find the session file containing this task
    session_file = find_task_session_file(task_id, tasks_dir)
    if not session_file:
        return False

    try:
        # Load the session
        with open(session_file) as f:
            data = json.load(f)

        # Find and update the task
        task_updated = False
        for task in data.get('tasks', []):
            if task.get('id') == task_id:
                # Initialize retrospective if it doesn't exist
                if not task.get('retrospective'):
                    task['retrospective'] = {
                        'notes': '',
                        'duration_minutes': 0,
                        'files_touched': [],
                        'tests_added': 0,
                        'commits': [],
                        'captured_at': datetime.now().isoformat()
                    }

                # Add commit if not already present
                commits = task['retrospective'].get('commits', [])
                if commit_hash not in commits:
                    commits.append(commit_hash)
                    task['retrospective']['commits'] = commits

                # Update timestamp
                task['updated_at'] = datetime.now().isoformat()

                # Mark as complete if requested
                if should_complete and task.get('status') != 'completed':
                    task['status'] = 'completed'
                    task['completed_at'] = datetime.now().isoformat()

                    # Add completion note if not present
                    if not task['retrospective'].get('notes'):
                        task['retrospective']['notes'] = f'Auto-completed by commit {commit_hash[:8]}'

                task_updated = True
                break

        if not task_updated:
            return False

        # Update saved_at timestamp
        data['saved_at'] = datetime.now().isoformat()

        # Write back atomically
        temp_file = session_file.with_suffix('.json.tmp')
        try:
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            temp_file.rename(session_file)
        except Exception:
            if temp_file.exists():
                temp_file.unlink()
            raise

        return True

    except (json.JSONDecodeError, KeyError, OSError) as e:
        print(f"Error updating task {task_id}: {e}")
        return False


def link_commit_to_tasks(
    commit_hash: str,
    commit_message: str,
    tasks_dir: str = "tasks",
    verbose: bool = False
) -> Dict[str, any]:
    """
    Link a commit to all referenced tasks and update their status.

    Args:
        commit_hash: Git commit SHA
        commit_message: Full commit message
        tasks_dir: Directory containing task session files
        verbose: Print detailed output

    Returns:
        Dictionary with:
        - task_ids: List of all task IDs found
        - completed: List of task IDs marked complete
        - referenced: List of task IDs just referenced
        - updated: List of task IDs successfully updated
        - failed: List of task IDs that failed to update
    """
    # Find all task IDs in the commit message
    task_ids = find_task_ids(commit_message)

    if not task_ids:
        return {
            'task_ids': [],
            'completed': [],
            'referenced': [],
            'updated': [],
            'failed': []
        }

    # Classify which tasks should be completed vs referenced
    classification = classify_task_references(commit_message, task_ids)

    completed = []
    referenced = []
    updated = []
    failed = []

    # Update tasks marked for completion
    for task_id in classification['complete']:
        success = update_task_with_commit(task_id, commit_hash, should_complete=True, tasks_dir=tasks_dir)
        if success:
            completed.append(task_id)
            updated.append(task_id)
            if verbose:
                print(f"✓ Completed task {task_id} (commit {commit_hash[:8]})")
        else:
            failed.append(task_id)
            if verbose:
                print(f"✗ Failed to complete task {task_id}")

    # Update tasks marked for reference only
    for task_id in classification['reference']:
        success = update_task_with_commit(task_id, commit_hash, should_complete=False, tasks_dir=tasks_dir)
        if success:
            referenced.append(task_id)
            updated.append(task_id)
            if verbose:
                print(f"✓ Linked commit to task {task_id} (commit {commit_hash[:8]})")
        else:
            failed.append(task_id)
            if verbose:
                print(f"✗ Failed to link commit to task {task_id}")

    return {
        'task_ids': task_ids,
        'completed': completed,
        'referenced': referenced,
        'updated': updated,
        'failed': failed
    }


def analyze_recent_commits(
    num_commits: int = 20,
    tasks_dir: str = "tasks",
    verbose: bool = True
) -> List[Dict[str, any]]:
    """
    Analyze recent commits for task references (dry run).

    Args:
        num_commits: Number of recent commits to analyze
        tasks_dir: Directory containing task session files
        verbose: Print detailed output

    Returns:
        List of commit analysis results
    """
    import subprocess

    try:
        # Get recent commit hashes and messages
        result = subprocess.run(
            ['git', 'log', f'-{num_commits}', '--format=%H|%s|%b'],
            capture_output=True,
            text=True,
            check=True
        )

        commits = []
        current_hash = None
        current_subject = None
        current_body = []

        for line in result.stdout.strip().split('\n'):
            if '|' in line:
                # Save previous commit if exists
                if current_hash:
                    full_message = f"{current_subject}\n{' '.join(current_body)}".strip()
                    task_ids = find_task_ids(full_message)
                    if task_ids:
                        classification = classify_task_references(full_message, task_ids)
                        commits.append({
                            'hash': current_hash,
                            'message': current_subject,
                            'task_ids': task_ids,
                            'complete': list(classification['complete']),
                            'reference': list(classification['reference'])
                        })

                # Parse new commit
                parts = line.split('|', 2)
                current_hash = parts[0]
                current_subject = parts[1]
                current_body = [parts[2]] if len(parts) > 2 else []
            else:
                current_body.append(line)

        # Don't forget the last commit
        if current_hash:
            full_message = f"{current_subject}\n{' '.join(current_body)}".strip()
            task_ids = find_task_ids(full_message)
            if task_ids:
                classification = classify_task_references(full_message, task_ids)
                commits.append({
                    'hash': current_hash,
                    'message': current_subject,
                    'task_ids': task_ids,
                    'complete': list(classification['complete']),
                    'reference': list(classification['reference'])
                })

        if verbose:
            print(f"\nAnalyzed {num_commits} recent commits:")
            print(f"Found {len(commits)} commits with task references\n")

            for commit in commits:
                print(f"Commit: {commit['hash'][:8]} - {commit['message'][:60]}")
                if commit['complete']:
                    print(f"  ✓ Would complete: {', '.join(commit['complete'])}")
                if commit['reference']:
                    print(f"  → Would reference: {', '.join(commit['reference'])}")
                print()

        return commits

    except subprocess.CalledProcessError as e:
        print(f"Error running git log: {e}")
        return []


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Link commits to tasks')
    parser.add_argument('--analyze', type=int, metavar='N',
                        help='Analyze last N commits (dry run)')
    parser.add_argument('--commit', metavar='HASH',
                        help='Link specific commit to tasks')
    parser.add_argument('--message', metavar='MSG',
                        help='Commit message (required with --commit)')
    parser.add_argument('--tasks-dir', default='tasks',
                        help='Tasks directory (default: tasks)')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    if args.analyze:
        analyze_recent_commits(args.analyze, args.tasks_dir, verbose=True)
    elif args.commit:
        if not args.message:
            print("Error: --message required with --commit")
            exit(1)
        result = link_commit_to_tasks(args.commit, args.message, args.tasks_dir, args.verbose)
        print(f"Linked commit {args.commit[:8]} to {len(result['updated'])} tasks")
        if result['completed']:
            print(f"  Completed: {', '.join(result['completed'])}")
        if result['referenced']:
            print(f"  Referenced: {', '.join(result['referenced'])}")
        if result['failed']:
            print(f"  Failed: {', '.join(result['failed'])}")
    else:
        parser.print_help()
