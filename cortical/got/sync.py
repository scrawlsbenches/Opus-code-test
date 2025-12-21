"""
Sync manager for GoT git-based synchronization.

Git is used as TRANSPORT only - sharing state between agents.
Transactions must be complete before sync operations.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .errors import SyncError
from .conflict import SyncConflict


@dataclass
class SyncResult:
    """
    Result of a sync operation (push/pull).

    Attributes:
        success: Whether operation succeeded
        action: Type of sync ("push", "pull", "none")
        version: Version identifier after sync (e.g., git commit hash)
        conflicts: List of conflicts detected during sync
        error: Error message if sync failed
    """

    success: bool
    action: str  # "push", "pull", "none"
    version: Optional[str] = None
    conflicts: List[SyncConflict] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class SyncStatus:
    """
    Current sync status of the repository.

    Attributes:
        ahead: Number of commits ahead of remote
        behind: Number of commits behind remote
        dirty: Whether there are uncommitted changes
        has_active_tx: Whether there are active transactions (blocks sync)
    """

    ahead: int
    behind: int
    dirty: bool
    has_active_tx: bool


class SyncManager:
    """
    Manages git-based synchronization between agents.

    Git is used for TRANSPORT only - sharing state between agents.
    Transactions must be complete before sync operations.

    Safety guarantees:
    - Cannot sync with active transactions
    - Cannot push when behind remote (need pull first)
    - Detects merge conflicts during pull
    """

    def __init__(self, got_dir: Path, git_dir: Optional[Path] = None):
        """
        Initialize sync manager.

        Args:
            got_dir: Path to GoT data directory (e.g., .got/)
            git_dir: Path to git repository root (auto-detected if None)
        """
        self.got_dir = Path(got_dir)
        self.git_dir = Path(git_dir) if git_dir else self._find_git_root()

        if not self.git_dir:
            raise SyncError("Not in a git repository", got_dir=str(self.got_dir))

    def can_sync(self) -> bool:
        """
        Check if sync is allowed.

        Sync is blocked if there are active transactions.
        This prevents syncing incomplete/uncommitted work.

        Returns:
            True if sync is safe, False otherwise
        """
        # Check for active transaction files in WAL
        wal_dir = self.got_dir / "wal"
        if not wal_dir.exists():
            return True

        # Look for active transaction markers
        # Active transactions have WAL entries but no COMMIT/ROLLBACK
        try:
            result = subprocess.run(
                ["grep", "-l", "TX_BEGIN", str(wal_dir / "*.wal")],
                capture_output=True,
                text=True,
                timeout=5,
                shell=True
            )
            if result.returncode == 0:
                # Found TX_BEGIN, check if committed/rolled back
                # This is a simplified check - real implementation would
                # parse WAL to find incomplete transactions
                return False
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            # If we can't check, be conservative and block
            return False

        return True

    def push(self) -> SyncResult:
        """
        Push local changes to remote.

        Fails if:
        - Active transactions exist
        - Remote has changes (need pull first)

        Returns:
            SyncResult with success status and pushed version
        """
        if not self.can_sync():
            return SyncResult(
                success=False,
                action="push",
                error="Active transactions exist - complete or rollback before pushing"
            )

        try:
            # Fetch to check remote state
            self._run_git(["fetch"])

            # Check if we're behind
            status = self.get_status()
            if status.behind > 0:
                return SyncResult(
                    success=False,
                    action="push",
                    error=f"Remote has {status.behind} new commits - pull first"
                )

            # Check if there's anything to push
            if status.ahead == 0:
                return SyncResult(
                    success=True,
                    action="none",
                    version=self._get_current_commit()
                )

            # Push
            self._run_git(["push"])

            return SyncResult(
                success=True,
                action="push",
                version=self._get_current_commit()
            )

        except SyncError as e:
            return SyncResult(
                success=False,
                action="push",
                error=str(e)
            )

    def pull(self) -> SyncResult:
        """
        Pull remote changes to local.

        Fails if:
        - Active transactions exist
        - Merge conflicts detected

        Returns:
            SyncResult with success status and pulled version
        """
        if not self.can_sync():
            return SyncResult(
                success=False,
                action="pull",
                error="Active transactions exist - complete or rollback before pulling"
            )

        try:
            # Fetch first
            self._run_git(["fetch"])

            # Check if there's anything to pull
            status = self.get_status()
            if status.behind == 0:
                return SyncResult(
                    success=True,
                    action="none",
                    version=self._get_current_commit()
                )

            # Pull with rebase to avoid merge commits
            try:
                self._run_git(["pull", "--rebase"])
            except SyncError as e:
                # Check if it's a merge conflict
                if "conflict" in str(e).lower():
                    return SyncResult(
                        success=False,
                        action="pull",
                        error="Merge conflict detected - resolve manually"
                    )
                raise

            return SyncResult(
                success=True,
                action="pull",
                version=self._get_current_commit()
            )

        except SyncError as e:
            return SyncResult(
                success=False,
                action="pull",
                error=str(e)
            )

    def get_status(self) -> SyncStatus:
        """
        Get current sync status.

        Returns:
            SyncStatus with ahead/behind counts and dirty state
        """
        try:
            # Get ahead/behind count
            result = self._run_git(
                ["rev-list", "--left-right", "--count", "HEAD...@{u}"]
            )
            parts = result.strip().split()
            ahead = int(parts[0]) if len(parts) > 0 else 0
            behind = int(parts[1]) if len(parts) > 1 else 0

        except (SyncError, ValueError, IndexError):
            # No upstream configured or other error
            ahead = 0
            behind = 0

        # Check for uncommitted changes
        try:
            result = self._run_git(["status", "--porcelain"])
            dirty = bool(result.strip())
        except SyncError:
            dirty = False

        return SyncStatus(
            ahead=ahead,
            behind=behind,
            dirty=dirty,
            has_active_tx=not self.can_sync()
        )

    def _find_git_root(self) -> Optional[Path]:
        """
        Find git repository root by walking up from got_dir.

        Returns:
            Path to git root or None if not in a git repo
        """
        current = self.got_dir.resolve()

        while current != current.parent:
            if (current / ".git").exists():
                return current
            current = current.parent

        return None

    def _get_current_commit(self) -> str:
        """
        Get current commit hash.

        Returns:
            Short commit hash (7 chars)
        """
        try:
            result = self._run_git(["rev-parse", "--short", "HEAD"])
            return result.strip()
        except SyncError:
            return "unknown"

    def _run_git(self, args: List[str], timeout: int = 30) -> str:
        """
        Run git command and return output.

        Args:
            args: Git command arguments
            timeout: Command timeout in seconds

        Returns:
            Command stdout output

        Raises:
            SyncError: If command fails
        """
        try:
            result = subprocess.run(
                ["git"] + args,
                cwd=self.git_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True
            )
            return result.stdout

        except subprocess.TimeoutExpired:
            raise SyncError(
                f"Git command timed out after {timeout}s",
                command=" ".join(args)
            )

        except subprocess.CalledProcessError as e:
            raise SyncError(
                f"Git command failed: {e.stderr}",
                command=" ".join(args),
                returncode=e.returncode
            )

        except Exception as e:
            raise SyncError(
                f"Git command error: {e}",
                command=" ".join(args)
            )
