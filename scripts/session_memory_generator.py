#!/usr/bin/env python3
"""
Session Memory Generator for Automatic Knowledge Capture

Generates draft memory entries from session activity for the Continuous Consciousness
roadmap (Epic 2, Sprint 2). Reads from git log and .git-ml/sessions/ to create
structured memory documents.

Usage:
    # Generate memory for current session
    python scripts/session_memory_generator.py --session-id abc123

    # Generate memory from git log (last N commits)
    python scripts/session_memory_generator.py --commits 10

    # Specify custom output directory
    python scripts/session_memory_generator.py --session-id abc123 --output samples/memories/

Sprint Tasks:
    - Sprint-2.3: SessionEnd auto-memory generation
    - Sprint-2.4: Post-commit task linking
"""

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class CommitInfo:
    """Information about a git commit."""
    sha: str
    short_sha: str
    message: str
    author: str
    date: str
    files_changed: List[str] = field(default_factory=list)
    insertions: int = 0
    deletions: int = 0
    task_ids: List[str] = field(default_factory=list)

    @classmethod
    def from_git_log(cls, sha: str) -> Optional['CommitInfo']:
        """Create CommitInfo from a git commit SHA."""
        try:
            # Get commit details
            result = subprocess.run(
                ['git', 'show', '--no-patch', '--format=%H%n%h%n%an%n%aI%n%s%n%b', sha],
                capture_output=True,
                text=True,
                check=True
            )
            lines = result.stdout.strip().split('\n')
            if len(lines) < 5:
                return None

            full_sha = lines[0]
            short_sha = lines[1]
            author = lines[2]
            date = lines[3]
            subject = lines[4]
            body = '\n'.join(lines[5:]) if len(lines) > 5 else ''
            full_message = f"{subject}\n{body}".strip()

            # Get file stats
            stat_result = subprocess.run(
                ['git', 'show', '--stat', '--format=', sha],
                capture_output=True,
                text=True,
                check=True
            )

            files_changed = []
            insertions = 0
            deletions = 0

            for line in stat_result.stdout.strip().split('\n'):
                if '|' in line:
                    # Parse file change line: "path/to/file.py | 10 ++++----"
                    parts = line.split('|')
                    if len(parts) >= 2:
                        filepath = parts[0].strip()
                        files_changed.append(filepath)
                        # Extract insertions/deletions from the stats
                        stats_part = parts[1].strip()
                        plus_count = stats_part.count('+')
                        minus_count = stats_part.count('-')
                        insertions += plus_count
                        deletions += minus_count

            # Extract task IDs from commit message
            task_ids = extract_task_ids(full_message)

            return cls(
                sha=full_sha,
                short_sha=short_sha,
                message=full_message,
                author=author,
                date=date,
                files_changed=files_changed,
                insertions=insertions,
                deletions=deletions,
                task_ids=task_ids
            )
        except subprocess.CalledProcessError:
            return None


@dataclass
class SessionData:
    """Data collected from a development session."""
    session_id: str
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    commits: List[CommitInfo] = field(default_factory=list)
    chat_ids: List[str] = field(default_factory=list)
    action_ids: List[str] = field(default_factory=list)
    summary: Optional[str] = None
    all_files_modified: Set[str] = field(default_factory=set)
    task_ids_referenced: Set[str] = field(default_factory=set)


class SessionMemoryGenerator:
    """Generate draft memory entries from session activity."""

    def __init__(self, session_id: Optional[str] = None, git_ml_dir: str = ".git-ml"):
        """
        Initialize the session memory generator.

        Args:
            session_id: Session ID to generate memory for (optional)
            git_ml_dir: Path to .git-ml directory (default: .git-ml)
        """
        self.session_id = session_id
        self.git_ml_dir = Path(git_ml_dir)
        self.sessions_dir = self.git_ml_dir / "sessions"

    def collect_session_data(self, num_commits: Optional[int] = None) -> SessionData:
        """
        Gather commits, files changed, tasks updated from session.

        Args:
            num_commits: Number of recent commits to analyze (if no session_id)

        Returns:
            SessionData with collected information
        """
        session_data = SessionData(session_id=self.session_id or "unknown")

        # Try to load session metadata from .git-ml/sessions/
        if self.session_id:
            session_data = self._load_session_metadata(session_data)

        # Collect commits from git log
        commits = self._collect_commits(num_commits)
        session_data.commits = commits

        # Aggregate file changes
        for commit in commits:
            session_data.all_files_modified.update(commit.files_changed)
            session_data.task_ids_referenced.update(commit.task_ids)

        return session_data

    def _load_session_metadata(self, session_data: SessionData) -> SessionData:
        """Load session metadata from .git-ml/sessions/ if available."""
        if not self.sessions_dir.exists():
            return session_data

        # Try to find session file matching the session ID
        for session_file in self.sessions_dir.glob(f"*_{self.session_id}.json"):
            try:
                with open(session_file) as f:
                    data = json.load(f)
                    session_data.started_at = data.get('started_at')
                    session_data.ended_at = data.get('ended_at')
                    session_data.chat_ids = data.get('chat_ids', [])
                    session_data.action_ids = data.get('action_ids', [])
                    session_data.summary = data.get('summary')
                    break
            except (json.JSONDecodeError, IOError):
                continue

        return session_data

    def _collect_commits(self, num_commits: Optional[int] = None) -> List[CommitInfo]:
        """Collect commits from git log."""
        commits = []

        # Determine how many commits to collect
        if num_commits is None:
            # If session has start time, collect commits since then
            if self.session_id and self.sessions_dir.exists():
                num_commits = 20  # Default to last 20 commits
            else:
                num_commits = 10  # Default to last 10 commits

        try:
            # Get commit SHAs
            result = subprocess.run(
                ['git', 'log', f'-{num_commits}', '--format=%H'],
                capture_output=True,
                text=True,
                check=True
            )

            commit_shas = [sha.strip() for sha in result.stdout.strip().split('\n') if sha.strip()]

            # Load commit details
            for sha in commit_shas:
                commit = CommitInfo.from_git_log(sha)
                if commit:
                    commits.append(commit)

        except subprocess.CalledProcessError:
            pass  # Return empty list if git log fails

        return commits

    def generate_draft_memory(self, session_data: SessionData) -> str:
        """
        Generate markdown memory entry from session data.

        Args:
            session_data: Session data to convert to memory

        Returns:
            Markdown formatted memory entry
        """
        # Generate date string
        if session_data.started_at:
            try:
                dt = datetime.fromisoformat(session_data.started_at.replace('Z', '+00:00'))
                date_str = dt.strftime('%Y-%m-%d')
            except (ValueError, AttributeError):
                date_str = datetime.now().strftime('%Y-%m-%d')
        else:
            date_str = datetime.now().strftime('%Y-%m-%d')

        # Extract topic from commits
        topic = self._extract_topic(session_data)

        # Build memory content
        lines = [
            f"# Memory Entry: {date_str} Session - {topic}",
            "",
            f"**Session ID:** `{session_data.session_id}`",
        ]

        # Add session metadata if available
        if session_data.started_at:
            lines.append(f"**Started:** {session_data.started_at}")
        if session_data.ended_at:
            lines.append(f"**Ended:** {session_data.ended_at}")
        if session_data.summary:
            lines.append(f"**Summary:** {session_data.summary}")

        # Generate tags from commits and files
        tags = self._generate_tags(session_data)
        if tags:
            tags_str = ", ".join(f"`{tag}`" for tag in sorted(tags))
            lines.append(f"**Tags:** {tags_str}")

        lines.append("")

        # Section: What Happened
        lines.append("## What Happened")
        lines.append("")
        if session_data.commits:
            lines.append(f"This session included {len(session_data.commits)} commits:")
            lines.append("")
            for commit in session_data.commits[:10]:  # Limit to 10 commits
                # Format commit message (first line only)
                msg_first_line = commit.message.split('\n')[0]
                lines.append(f"- **[{commit.short_sha}]** {msg_first_line}")
                if commit.task_ids:
                    task_refs = ", ".join(commit.task_ids)
                    lines.append(f"  - Tasks: {task_refs}")
        else:
            lines.append("No commits recorded in this session.")

        lines.append("")

        # Section: Key Insights
        lines.append("## Key Insights")
        lines.append("")
        insights = self._extract_insights(session_data)
        if insights:
            for insight in insights:
                lines.append(f"- {insight}")
        else:
            lines.append("- [To be filled in manually]")

        lines.append("")

        # Section: Files Modified
        lines.append("## Files Modified")
        lines.append("")
        if session_data.all_files_modified:
            # Group by directory/category
            categorized = self._categorize_files(session_data.all_files_modified)
            for category, files in sorted(categorized.items()):
                lines.append(f"### {category}")
                lines.append("")
                for filepath in sorted(files)[:20]:  # Limit to 20 files per category
                    lines.append(f"- `{filepath}`")
                lines.append("")
        else:
            lines.append("No files modified.")
            lines.append("")

        # Section: Tasks Updated
        if session_data.task_ids_referenced:
            lines.append("## Tasks Updated")
            lines.append("")
            for task_id in sorted(session_data.task_ids_referenced):
                lines.append(f"- {task_id}")
            lines.append("")

        # Section: Related Documents (placeholder)
        lines.append("## Related Documents")
        lines.append("")
        lines.append("- [[CLAUDE.md]]")
        lines.append("")

        return '\n'.join(lines)

    def _extract_topic(self, session_data: SessionData) -> str:
        """Extract main topic from session data."""
        if session_data.summary:
            # Use summary if available
            return session_data.summary.split(':')[-1].strip()[:50]

        if session_data.commits:
            # Use most common words from commit messages
            words = []
            for commit in session_data.commits:
                # Extract first line of commit message
                first_line = commit.message.split('\n')[0]
                # Remove commit type prefix (feat:, fix:, docs:, etc.)
                first_line = re.sub(r'^[a-z]+:\s*', '', first_line, flags=re.IGNORECASE)
                words.extend(first_line.lower().split())

            # Filter common words and find most frequent
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'from'}
            words = [w for w in words if w not in common_words and len(w) > 3]

            if words:
                # Return most common word (simple approach)
                from collections import Counter
                most_common = Counter(words).most_common(3)
                return ' '.join([word for word, _ in most_common])

        return "Development Work"

    def _generate_tags(self, session_data: SessionData) -> Set[str]:
        """Generate tags from session data."""
        tags = set()
        tags.add('session')

        # Add tags from file types
        for filepath in session_data.all_files_modified:
            if filepath.endswith('.py'):
                tags.add('python')
            elif filepath.endswith('.md'):
                tags.add('docs')
            elif filepath.endswith('.json'):
                tags.add('config')

            # Add tags from directory names
            if 'test' in filepath.lower():
                tags.add('testing')
            if 'doc' in filepath.lower():
                tags.add('documentation')
            if 'script' in filepath.lower():
                tags.add('scripts')

        # Add tags from commit messages
        for commit in session_data.commits:
            msg_lower = commit.message.lower()
            if 'fix' in msg_lower or 'bug' in msg_lower:
                tags.add('bugfix')
            if 'test' in msg_lower:
                tags.add('testing')
            if 'refactor' in msg_lower:
                tags.add('refactoring')
            if 'doc' in msg_lower or 'memory' in msg_lower:
                tags.add('documentation')
            if 'feat' in msg_lower or 'feature' in msg_lower:
                tags.add('feature')

        return tags

    def _extract_insights(self, session_data: SessionData) -> List[str]:
        """Extract key insights from session data."""
        insights = []

        # Count commits by type
        commit_types = {}
        for commit in session_data.commits:
            # Extract commit type (feat, fix, docs, etc.)
            match = re.match(r'^([a-z]+):', commit.message.split('\n')[0], re.IGNORECASE)
            if match:
                ctype = match.group(1).lower()
                commit_types[ctype] = commit_types.get(ctype, 0) + 1

        if commit_types:
            # Report most common commit types
            for ctype, count in sorted(commit_types.items(), key=lambda x: -x[1])[:3]:
                if count > 1:
                    insights.append(f"{count} {ctype} commits made")

        # File statistics
        total_files = len(session_data.all_files_modified)
        if total_files > 0:
            insights.append(f"{total_files} files modified")

        # Task statistics
        total_tasks = len(session_data.task_ids_referenced)
        if total_tasks > 0:
            insights.append(f"{total_tasks} tasks referenced in commits")

        return insights

    def _categorize_files(self, files: Set[str]) -> Dict[str, List[str]]:
        """Categorize files by directory or type."""
        categorized = {}

        for filepath in files:
            # Determine category
            if filepath.startswith('cortical/'):
                category = 'Core Library'
            elif filepath.startswith('tests/'):
                category = 'Tests'
            elif filepath.startswith('scripts/'):
                category = 'Scripts'
            elif filepath.startswith('docs/'):
                category = 'Documentation'
            elif filepath.startswith('samples/'):
                category = 'Samples'
            elif filepath.startswith('.'):
                category = 'Configuration'
            else:
                category = 'Other'

            if category not in categorized:
                categorized[category] = []
            categorized[category].append(filepath)

        return categorized

    def save_draft(self, content: str, output_dir: str = "samples/memories") -> Path:
        """
        Save draft memory to samples/memories/[DRAFT]-YYYY-MM-DD-session-{id}.md

        Args:
            content: Memory content to save
            output_dir: Directory to save memory to

        Returns:
            Path to saved memory file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate filename
        date_str = datetime.now().strftime('%Y-%m-%d')
        session_suffix = self.session_id or "unknown"
        filename = f"[DRAFT]-{date_str}-session-{session_suffix}.md"

        filepath = output_path / filename

        # Write content
        with open(filepath, 'w') as f:
            f.write(content)

        return filepath


def extract_task_ids(text: str) -> List[str]:
    """
    Extract task IDs from text.

    Supports formats:
        - T-YYYYMMDD-HHMMSS-XXXX (old format without microseconds)
        - T-YYYYMMDD-HHMMSSffffff-XXXX (new format with microseconds)
        - T-YYYYMMDD-HHMMSSffffff-XXXX-NNN (session task)

    Args:
        text: Text to search for task IDs

    Returns:
        List of task IDs found
    """
    # Pattern matches both old and new formats
    pattern = r'T-\d{8}-\d{6,12}-[a-f0-9]{4}(?:-\d{3})?'
    matches = re.findall(pattern, text, re.IGNORECASE)
    return list(set(matches))  # Remove duplicates


def link_commit_to_tasks(commit_sha: str, commit_message: str, tasks_dir: str = "tasks") -> List[str]:
    """
    Extract task IDs from commit message and update task status.

    Args:
        commit_sha: Git commit SHA
        commit_message: Commit message to search
        tasks_dir: Directory containing task JSON files

    Returns:
        List of task IDs that were updated
    """
    task_ids = extract_task_ids(commit_message)
    if not task_ids:
        return []

    tasks_path = Path(tasks_dir)
    if not tasks_path.exists():
        return []

    updated_tasks = []

    # Search for tasks in all session files
    for task_file in tasks_path.glob("*.json"):
        try:
            with open(task_file, 'r') as f:
                session_data = json.load(f)

            modified = False
            for task in session_data.get('tasks', []):
                if task['id'] in task_ids:
                    # Add commit SHA to task context
                    if 'commits' not in task.get('context', {}):
                        if 'context' not in task:
                            task['context'] = {}
                        task['context']['commits'] = []

                    if commit_sha not in task['context']['commits']:
                        task['context']['commits'].append(commit_sha)
                        modified = True
                        updated_tasks.append(task['id'])

            # Save if modified
            if modified:
                with open(task_file, 'w') as f:
                    json.dump(session_data, f, indent=2)

        except (json.JSONDecodeError, IOError, KeyError):
            continue

    return updated_tasks


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate draft memory entries from session activity"
    )
    parser.add_argument(
        '--session-id',
        help='Session ID to generate memory for'
    )
    parser.add_argument(
        '--commits',
        type=int,
        default=10,
        help='Number of recent commits to analyze (default: 10)'
    )
    parser.add_argument(
        '--output',
        default='samples/memories',
        help='Output directory for draft memory (default: samples/memories)'
    )
    parser.add_argument(
        '--git-ml-dir',
        default='.git-ml',
        help='Path to .git-ml directory (default: .git-ml)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print memory to stdout instead of saving'
    )

    args = parser.parse_args()

    # Create generator
    generator = SessionMemoryGenerator(
        session_id=args.session_id,
        git_ml_dir=args.git_ml_dir
    )

    # Collect session data
    session_data = generator.collect_session_data(num_commits=args.commits)

    # Generate memory
    memory_content = generator.generate_draft_memory(session_data)

    if args.dry_run:
        # Print to stdout
        print(memory_content)
    else:
        # Save to file
        filepath = generator.save_draft(memory_content, output_dir=args.output)
        print(f"✅ Draft memory saved to: {filepath}")

        # Report on any task linking
        task_ids = session_data.task_ids_referenced
        if task_ids:
            print(f"📋 Found {len(task_ids)} task references in commits")
            for task_id in sorted(task_ids):
                print(f"   - {task_id}")


if __name__ == '__main__':
    main()
