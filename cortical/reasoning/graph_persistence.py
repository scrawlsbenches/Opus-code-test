"""
Graph Persistence: Automatic Git Commits and Write-Ahead Logging.

This module provides tools for persisting ThoughtGraph state with:
- Automatic git commits after saves
- Write-ahead logging for crash recovery
- Safe push operations with protected branch detection
- Validation before commits

Key components:
- GitAutoCommitter: Automatic git commits with safety features
- GraphWAL: Write-ahead log for graph operations
- GraphWALEntry: WAL entry specific to graph operations

Usage:
    from cortical.reasoning.graph_persistence import GraphWAL, GraphWALEntry
    from cortical.reasoning.thought_graph import ThoughtGraph

    # Initialize WAL for graph persistence
    graph_wal = GraphWAL("reasoning_wal")

    # Log graph operations
    graph_wal.log_add_node("Q1", NodeType.QUESTION, "What is the best approach?")
    graph_wal.log_add_edge("Q1", "H1", EdgeType.EXPLORES)

    # Recover graph after crash
    graph = ThoughtGraph()
    for entry in graph_wal.get_all_entries():
        graph_wal.apply_entry(entry, graph)
"""

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from threading import Timer
from typing import Any, Dict, Iterator, List, Optional, Set

from cortical.wal import WALWriter, WALEntry, SnapshotManager
from .thought_graph import ThoughtGraph
from .graph_of_thought import NodeType, EdgeType, ThoughtNode, ThoughtEdge


class GitAutoCommitter:
    """
    Automatic git commits for graph persistence.

    Provides safe, configurable automatic commits when graphs are saved,
    with debouncing, validation, and protection against dangerous operations.

    Modes:
    - 'immediate': Commit immediately on save
    - 'debounced': Wait for debounce_seconds of inactivity before committing
    - 'manual': No automatic commits (validation only)

    Safety features:
    - Never force pushes
    - Protected branch detection (main/master by default)
    - Pre-commit graph validation
    - Backup branch creation for risky operations

    Example:
        >>> committer = GitAutoCommitter(mode='debounced', debounce_seconds=5)
        >>> committer.commit_on_save('/path/to/graph.json')
        # Waits 5 seconds, then commits if no more saves
    """

    def __init__(
        self,
        mode: str = 'immediate',
        debounce_seconds: int = 5,
        auto_push: bool = False,
        protected_branches: Optional[List[str]] = None,
        repo_path: Optional[str] = None,
    ):
        """
        Initialize GitAutoCommitter.

        Args:
            mode: Commit mode ('immediate', 'debounced', 'manual')
            debounce_seconds: Delay before debounced commit (default: 5)
            auto_push: Whether to auto-push after commit (default: False)
            protected_branches: Branches to never auto-push (default: ['main', 'master'])
            repo_path: Path to git repository (default: current directory)

        Raises:
            ValueError: If mode is invalid
        """
        if mode not in ('immediate', 'debounced', 'manual'):
            raise ValueError(f"Invalid mode: {mode}. Must be 'immediate', 'debounced', or 'manual'")

        self.mode = mode
        self.debounce_seconds = debounce_seconds
        self.auto_push = auto_push
        self.protected_branches: Set[str] = set(protected_branches or ['main', 'master'])
        self.repo_path = Path(repo_path or os.getcwd())

        # Debouncing state
        self._debounce_timer: Optional[Timer] = None
        self._pending_commit: Optional[tuple] = None

    def get_current_branch(self) -> Optional[str]:
        """
        Get the current git branch.

        Returns:
            Branch name or None if not in a git repo or on detached HEAD
        """
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
                timeout=5
            )
            branch = result.stdout.strip()
            return branch if branch != 'HEAD' else None
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, Exception):
            return None

    def is_protected_branch(self, branch: Optional[str] = None) -> bool:
        """
        Check if a branch is protected from auto-push.

        Args:
            branch: Branch name to check (default: current branch)

        Returns:
            True if branch is protected, False otherwise
        """
        if branch is None:
            branch = self.get_current_branch()

        if branch is None:
            return True  # Detached HEAD is protected

        return branch in self.protected_branches

    def validate_before_commit(self, graph: ThoughtGraph) -> tuple[bool, Optional[str]]:
        """
        Validate graph before committing.

        Checks:
        - Graph is not empty
        - No orphaned nodes (nodes with no edges)

        Args:
            graph: ThoughtGraph to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if graph is empty
        if not graph.nodes:
            return False, "Cannot commit empty graph"

        # Check for orphaned nodes (optional warning, not blocking)
        orphans = graph.find_orphans()
        if orphans and len(orphans) == len(graph.nodes):
            # All nodes are orphans - likely a problem
            return False, f"All {len(orphans)} nodes are orphaned (no edges)"

        # Validation passed
        return True, None

    def auto_commit(
        self,
        message: str,
        files: List[str],
        validate_graph: Optional[ThoughtGraph] = None
    ) -> bool:
        """
        Commit specified files with a message.

        Args:
            message: Commit message
            files: List of file paths to commit
            validate_graph: Optional graph to validate before commit

        Returns:
            True if commit succeeded, False otherwise
        """
        # Validate graph if provided
        if validate_graph is not None:
            valid, error = self.validate_before_commit(validate_graph)
            if not valid:
                print(f"[GitAutoCommitter] Validation failed: {error}")
                return False

        # Check if we're in a git repo
        try:
            subprocess.run(
                ['git', 'rev-parse', '--git-dir'],
                cwd=self.repo_path,
                capture_output=True,
                check=True,
                timeout=5
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, Exception):
            print("[GitAutoCommitter] Not in a git repository")
            return False

        try:
            # Add files
            for file_path in files:
                subprocess.run(
                    ['git', 'add', file_path],
                    cwd=self.repo_path,
                    capture_output=True,
                    check=True,
                    timeout=10
                )

            # Commit
            subprocess.run(
                ['git', 'commit', '-m', message],
                cwd=self.repo_path,
                capture_output=True,
                check=True,
                timeout=10
            )

            print(f"[GitAutoCommitter] Committed: {message}")
            return True

        except subprocess.CalledProcessError as e:
            # Check if it's a "nothing to commit" error (not a real error)
            if b'nothing to commit' in e.stdout or b'nothing to commit' in e.stderr:
                print("[GitAutoCommitter] No changes to commit")
                return True
            else:
                print(f"[GitAutoCommitter] Commit failed: {e.stderr.decode() if e.stderr else str(e)}")
                return False
        except subprocess.TimeoutExpired:
            print("[GitAutoCommitter] Commit timed out")
            return False
        except FileNotFoundError:
            print("[GitAutoCommitter] Git command not found")
            return False

    def push_if_safe(
        self,
        remote: str = 'origin',
        branch: Optional[str] = None,
        force_protected: bool = False
    ) -> bool:
        """
        Push to remote if safe (not a protected branch).

        Args:
            remote: Remote name (default: 'origin')
            branch: Branch to push (default: current branch)
            force_protected: Override protected branch check (default: False)

        Returns:
            True if push succeeded or was skipped safely, False on error
        """
        if branch is None:
            branch = self.get_current_branch()

        if branch is None:
            print("[GitAutoCommitter] Cannot push: detached HEAD")
            return False

        # Check protected branch
        if self.is_protected_branch(branch) and not force_protected:
            print(f"[GitAutoCommitter] Skipping push: {branch} is protected")
            return True  # Not an error, just skipped

        try:
            # Push without force
            subprocess.run(
                ['git', 'push', remote, branch],
                cwd=self.repo_path,
                capture_output=True,
                check=True,
                timeout=30
            )

            print(f"[GitAutoCommitter] Pushed to {remote}/{branch}")
            return True

        except subprocess.CalledProcessError as e:
            print(f"[GitAutoCommitter] Push failed: {e.stderr.decode() if e.stderr else str(e)}")
            return False
        except subprocess.TimeoutExpired:
            print("[GitAutoCommitter] Push timed out")
            return False
        except FileNotFoundError:
            print("[GitAutoCommitter] Git command not found")
            return False

    def commit_on_save(
        self,
        graph_path: str,
        graph: Optional[ThoughtGraph] = None,
        message: Optional[str] = None
    ) -> None:
        """
        Called after a graph save completes.

        Behavior depends on mode:
        - 'immediate': Commits immediately
        - 'debounced': Waits for debounce_seconds of inactivity
        - 'manual': No action (only validates if graph provided)

        Args:
            graph_path: Path to saved graph file
            graph: Optional ThoughtGraph for validation
            message: Optional custom commit message
        """
        if self.mode == 'manual':
            # Manual mode: only validate
            if graph is not None:
                valid, error = self.validate_before_commit(graph)
                if not valid:
                    print(f"[GitAutoCommitter] Validation warning: {error}")
            return

        # Default commit message
        if message is None:
            filename = Path(graph_path).name
            message = f"graph: Auto-save {filename}"

        files = [graph_path]

        if self.mode == 'immediate':
            # Immediate commit
            success = self.auto_commit(message, files, validate_graph=graph)

            # Auto-push if enabled
            if success and self.auto_push:
                self.push_if_safe()

        elif self.mode == 'debounced':
            # Cancel pending commit if any
            if self._debounce_timer is not None:
                self._debounce_timer.cancel()

            # Store pending commit
            self._pending_commit = (message, files, graph)

            # Schedule debounced commit
            def _do_debounced_commit():
                if self._pending_commit is not None:
                    msg, fls, grph = self._pending_commit
                    success = self.auto_commit(msg, fls, validate_graph=grph)

                    # Auto-push if enabled
                    if success and self.auto_push:
                        self.push_if_safe()

                    self._pending_commit = None
                    self._debounce_timer = None

            self._debounce_timer = Timer(self.debounce_seconds, _do_debounced_commit)
            self._debounce_timer.start()

    def create_backup_branch(self, prefix: str = 'backup') -> Optional[str]:
        """
        Create a backup branch before risky operations.

        Args:
            prefix: Prefix for backup branch name (default: 'backup')

        Returns:
            Backup branch name or None on failure
        """
        current_branch = self.get_current_branch()
        if current_branch is None:
            print("[GitAutoCommitter] Cannot create backup: detached HEAD")
            return None

        # Generate backup branch name with timestamp
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        backup_branch = f"{prefix}/{current_branch}/{timestamp}"

        try:
            subprocess.run(
                ['git', 'branch', backup_branch],
                cwd=self.repo_path,
                capture_output=True,
                check=True,
                timeout=10
            )

            print(f"[GitAutoCommitter] Created backup branch: {backup_branch}")
            return backup_branch

        except subprocess.CalledProcessError as e:
            print(f"[GitAutoCommitter] Failed to create backup: {e.stderr.decode() if e.stderr else str(e)}")
            return None
        except subprocess.TimeoutExpired:
            print("[GitAutoCommitter] Backup creation timed out")
            return None
        except FileNotFoundError:
            print("[GitAutoCommitter] Git command not found")
            return None

    def cleanup(self) -> None:
        """
        Clean up resources (cancel pending timers).

        Call this before destroying the committer instance.
        """
        if self._debounce_timer is not None:
            self._debounce_timer.cancel()
            self._debounce_timer = None
            self._pending_commit = None


# ==============================================================================
# GRAPH WAL ENTRY
# ==============================================================================

@dataclass
class GraphWALEntry:
    """
    Write-Ahead Log entry specific to graph operations.

    This wraps the base WALEntry with graph-specific fields and operations.
    Each entry represents a single atomic graph operation with integrity checksums.

    Attributes:
        operation: Type of graph operation (add_node, remove_node, add_edge, etc.)
        timestamp: ISO format timestamp of when operation occurred
        node_id: Primary node ID for node operations
        node_type: Type of node (for add_node operations)
        edge_type: Type of edge (for add_edge operations)
        source_id: Source node ID (for edge operations)
        target_id: Target node ID (for edge operations)
        cluster_id: Cluster ID (for cluster operations)
        payload: Flexible dict for operation-specific data
        checksum: SHA256 checksum for integrity verification
    """

    operation: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    node_id: Optional[str] = None
    node_type: Optional[str] = None
    edge_type: Optional[str] = None
    source_id: Optional[str] = None
    target_id: Optional[str] = None
    cluster_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    checksum: str = ""

    def __post_init__(self):
        """Compute checksum if not provided."""
        if not self.checksum:
            self.checksum = self._compute_checksum()

    def _compute_checksum(self) -> str:
        """
        Compute SHA256 checksum of entry content.

        Returns:
            First 16 characters of hex digest
        """
        content = json.dumps({
            'operation': self.operation,
            'timestamp': self.timestamp,
            'node_id': self.node_id,
            'node_type': self.node_type,
            'edge_type': self.edge_type,
            'source_id': self.source_id,
            'target_id': self.target_id,
            'cluster_id': self.cluster_id,
            'payload': self.payload,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_json(self) -> str:
        """
        Serialize to JSON string.

        Returns:
            JSON representation of the entry
        """
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> 'GraphWALEntry':
        """
        Deserialize from JSON string.

        Args:
            json_str: JSON string representation

        Returns:
            GraphWALEntry instance
        """
        data = json.loads(json_str)
        return cls(**data)

    def verify(self) -> bool:
        """
        Verify checksum matches content.

        Returns:
            True if checksum is valid, False otherwise
        """
        expected = self._compute_checksum()
        return self.checksum == expected

    def to_wal_entry(self) -> WALEntry:
        """
        Convert to base WALEntry for compatibility.

        Returns:
            WALEntry with graph data in payload
        """
        return WALEntry(
            operation=self.operation,
            timestamp=self.timestamp,
            doc_id=self.node_id,  # Reuse doc_id field
            payload={
                'node_id': self.node_id,
                'node_type': self.node_type,
                'edge_type': self.edge_type,
                'source_id': self.source_id,
                'target_id': self.target_id,
                'cluster_id': self.cluster_id,
                **self.payload,
            },
            # Don't pass checksum - let WALEntry compute it with its own algorithm
        )

    @classmethod
    def from_wal_entry(cls, entry: WALEntry) -> 'GraphWALEntry':
        """
        Convert from base WALEntry.

        Args:
            entry: Base WALEntry to convert

        Returns:
            GraphWALEntry instance
        """
        payload = entry.payload.copy()
        return cls(
            operation=entry.operation,
            timestamp=entry.timestamp,
            node_id=payload.pop('node_id', None),
            node_type=payload.pop('node_type', None),
            edge_type=payload.pop('edge_type', None),
            source_id=payload.pop('source_id', None),
            target_id=payload.pop('target_id', None),
            cluster_id=payload.pop('cluster_id', None),
            payload=payload,
            # Don't pass checksum - let GraphWALEntry compute it with its own algorithm
        )


# ==============================================================================
# GRAPH WAL
# ==============================================================================

class GraphWAL:
    """
    Write-Ahead Log for ThoughtGraph operations.

    Provides durable, fault-tolerant persistence for graph operations with:
    - Atomic operation logging with checksums
    - Crash recovery through replay
    - Snapshot support for fast recovery
    - Operation-specific logging methods

    Example:
        >>> graph_wal = GraphWAL("reasoning_wal")
        >>> graph_wal.log_add_node("Q1", NodeType.QUESTION, "What approach?",
        ...                         properties={'urgency': 'high'})
        >>> graph_wal.log_add_edge("Q1", "H1", EdgeType.EXPLORES, weight=0.9)
        >>> # Recover graph from WAL
        >>> graph = ThoughtGraph()
        >>> for entry in graph_wal.get_all_entries():
        ...     graph_wal.apply_entry(entry, graph)
    """

    def __init__(self, wal_dir: str):
        """
        Initialize GraphWAL.

        Args:
            wal_dir: Directory for WAL files and snapshots
        """
        self.wal_dir = Path(wal_dir)
        self._writer = WALWriter(wal_dir)
        self._snapshot_mgr = SnapshotManager(wal_dir)

    # ==========================================================================
    # LOGGING OPERATIONS
    # ==========================================================================

    def log_add_node(
        self,
        node_id: str,
        node_type: NodeType,
        content: str,
        properties: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a node addition operation.

        Args:
            node_id: Unique identifier for the node
            node_type: Type of thought node
            content: Main content/description
            properties: Optional type-specific properties
            metadata: Optional additional metadata
        """
        entry = GraphWALEntry(
            operation='add_node',
            node_id=node_id,
            node_type=node_type.value,
            payload={
                'content': content,
                'properties': properties or {},
                'metadata': metadata or {},
            }
        )
        self._writer.append(entry.to_wal_entry())

    def log_remove_node(self, node_id: str) -> None:
        """
        Log a node removal operation.

        Args:
            node_id: ID of node to remove
        """
        entry = GraphWALEntry(
            operation='remove_node',
            node_id=node_id,
        )
        self._writer.append(entry.to_wal_entry())

    def log_update_node(
        self,
        node_id: str,
        updates: Dict[str, Any],
    ) -> None:
        """
        Log a node update operation.

        Args:
            node_id: ID of node to update
            updates: Dictionary of field updates
                    Can include: 'content', 'properties', 'metadata'
        """
        entry = GraphWALEntry(
            operation='update_node',
            node_id=node_id,
            payload={'updates': updates}
        )
        self._writer.append(entry.to_wal_entry())

    def log_add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        weight: float = 1.0,
        confidence: float = 1.0,
        bidirectional: bool = False,
    ) -> None:
        """
        Log an edge addition operation.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Type of relationship
            weight: Strength of relationship (0.0-1.0)
            confidence: Confidence in relationship (0.0-1.0)
            bidirectional: Whether relationship goes both ways
        """
        entry = GraphWALEntry(
            operation='add_edge',
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type.value,
            payload={
                'weight': weight,
                'confidence': confidence,
                'bidirectional': bidirectional,
            }
        )
        self._writer.append(entry.to_wal_entry())

    def log_remove_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
    ) -> None:
        """
        Log an edge removal operation.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Type of relationship to remove
        """
        entry = GraphWALEntry(
            operation='remove_edge',
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type.value,
        )
        self._writer.append(entry.to_wal_entry())

    def log_add_cluster(
        self,
        cluster_id: str,
        name: str,
        node_ids: Optional[Set[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a cluster addition operation.

        Args:
            cluster_id: Unique cluster identifier
            name: Human-readable cluster name
            node_ids: Optional set of initial node IDs
            properties: Optional cluster properties
        """
        entry = GraphWALEntry(
            operation='add_cluster',
            cluster_id=cluster_id,
            payload={
                'name': name,
                'node_ids': list(node_ids or set()),
                'properties': properties or {},
            }
        )
        self._writer.append(entry.to_wal_entry())

    def log_merge_nodes(
        self,
        source_ids: List[str],
        target_id: str,
    ) -> None:
        """
        Log a node merge operation.

        Args:
            source_ids: List of node IDs to merge
            target_id: ID for the merged result node
        """
        entry = GraphWALEntry(
            operation='merge_nodes',
            node_id=target_id,
            payload={'source_ids': source_ids}
        )
        self._writer.append(entry.to_wal_entry())

    def log_graph_operation(
        self,
        operation_type: str,
        payload: Dict[str, Any],
    ) -> None:
        """
        Log a generic graph operation.

        This is a fallback for operations not covered by specific methods.

        Args:
            operation_type: Type of operation
            payload: Operation-specific data
        """
        entry = GraphWALEntry(
            operation=operation_type,
            payload=payload,
        )
        self._writer.append(entry.to_wal_entry())

    # ==========================================================================
    # REPLAY OPERATIONS
    # ==========================================================================

    def apply_entry(self, entry: GraphWALEntry, graph: ThoughtGraph) -> None:
        """
        Apply a single WAL entry to reconstruct graph state.

        Args:
            entry: GraphWALEntry to apply
            graph: ThoughtGraph to modify

        Raises:
            ValueError: If entry checksum verification fails
        """
        if not entry.verify():
            raise ValueError(f"Entry checksum verification failed: {entry.operation}")

        if entry.operation == 'add_node':
            # Add node to graph
            node_type = NodeType(entry.node_type)
            content = entry.payload.get('content', '')
            properties = entry.payload.get('properties', {})
            metadata = entry.payload.get('metadata', {})

            # Skip if node already exists
            if entry.node_id not in graph.nodes:
                graph.add_node(
                    entry.node_id,
                    node_type,
                    content,
                    properties,
                    metadata,
                )

        elif entry.operation == 'remove_node':
            # Remove node from graph
            if entry.node_id in graph.nodes:
                graph.remove_node(entry.node_id)

        elif entry.operation == 'update_node':
            # Update node properties
            if entry.node_id in graph.nodes:
                node = graph.nodes[entry.node_id]
                updates = entry.payload.get('updates', {})

                if 'content' in updates:
                    node.content = updates['content']
                if 'properties' in updates:
                    node.properties.update(updates['properties'])
                if 'metadata' in updates:
                    node.metadata.update(updates['metadata'])

        elif entry.operation == 'add_edge':
            # Add edge to graph
            edge_type = EdgeType(entry.edge_type)
            weight = entry.payload.get('weight', 1.0)
            confidence = entry.payload.get('confidence', 1.0)
            bidirectional = entry.payload.get('bidirectional', False)

            # Skip if either node doesn't exist
            if entry.source_id in graph.nodes and entry.target_id in graph.nodes:
                # Check if edge already exists to avoid duplicates
                existing_edges = graph.get_edges_from(entry.source_id)
                edge_exists = any(
                    e.target_id == entry.target_id and e.edge_type == edge_type
                    for e in existing_edges
                )

                if not edge_exists:
                    graph.add_edge(
                        entry.source_id,
                        entry.target_id,
                        edge_type,
                        weight,
                        confidence,
                        bidirectional,
                    )

        elif entry.operation == 'remove_edge':
            # Remove edge from graph
            edge_type = EdgeType(entry.edge_type)
            graph.remove_edge(entry.source_id, entry.target_id, edge_type)

        elif entry.operation == 'add_cluster':
            # Add cluster to graph
            name = entry.payload.get('name', '')
            node_ids = set(entry.payload.get('node_ids', []))
            properties = entry.payload.get('properties', {})

            # Skip if cluster already exists
            if entry.cluster_id not in graph.clusters:
                cluster = graph.add_cluster(entry.cluster_id, name, node_ids)
                cluster.properties.update(properties)

        elif entry.operation == 'merge_nodes':
            # Merge multiple nodes into target
            source_ids = entry.payload.get('source_ids', [])

            # Only merge if all nodes exist
            if all(nid in graph.nodes for nid in source_ids) and len(source_ids) >= 2:
                # Merge pairwise into target
                current_id = source_ids[0]
                for source_id in source_ids[1:]:
                    if source_id in graph.nodes:
                        current_id = entry.node_id if source_id == source_ids[-1] else current_id
                        try:
                            graph.merge_nodes(current_id, source_id, entry.node_id)
                            current_id = entry.node_id
                        except ValueError:
                            pass  # Node already merged or doesn't exist

        # For unknown operations, do nothing
        # This allows forward compatibility

    def get_all_entries(self) -> Iterator[GraphWALEntry]:
        """
        Get all WAL entries for replay.

        Yields:
            GraphWALEntry objects in chronological order
        """
        # Get entries from the beginning
        if self._writer.index.current_wal_file:
            for wal_entry in self._writer.get_entries_since(
                self._writer.index.current_wal_file, 0
            ):
                try:
                    yield GraphWALEntry.from_wal_entry(wal_entry)
                except (KeyError, ValueError):
                    # Skip malformed entries
                    continue

    def get_entries_since(
        self,
        wal_file: str,
        offset: int = 0
    ) -> Iterator[GraphWALEntry]:
        """
        Get WAL entries since a specific point.

        Args:
            wal_file: WAL file to start from
            offset: Line offset in that file

        Yields:
            GraphWALEntry objects
        """
        for wal_entry in self._writer.get_entries_since(wal_file, offset):
            try:
                yield GraphWALEntry.from_wal_entry(wal_entry)
            except (KeyError, ValueError):
                # Skip malformed entries
                continue

    # ==========================================================================
    # SNAPSHOT OPERATIONS
    # ==========================================================================

    def create_snapshot(
        self,
        graph: ThoughtGraph,
        compress: bool = True,
    ) -> str:
        """
        Create a snapshot of graph state.

        Args:
            graph: ThoughtGraph to snapshot
            compress: Whether to gzip the snapshot

        Returns:
            Snapshot ID
        """
        # Serialize graph to state dict
        state = self._graph_to_state(graph)

        # Get current WAL position
        wal_file = self._writer.index.current_wal_file
        wal_offset = self._writer.get_entry_count()

        return self._snapshot_mgr.create_snapshot(
            state,
            wal_file=wal_file,
            wal_offset=wal_offset,
            compress=compress,
        )

    def load_snapshot(
        self,
        snapshot_id: Optional[str] = None,
    ) -> Optional[ThoughtGraph]:
        """
        Load graph from snapshot.

        Args:
            snapshot_id: Snapshot to load (latest if None)

        Returns:
            Reconstructed ThoughtGraph, or None if no snapshot found
        """
        snapshot_data = self._snapshot_mgr.load_snapshot(snapshot_id)
        if not snapshot_data:
            return None

        state = snapshot_data.get('state', {})
        return self._state_to_graph(state)

    def _graph_to_state(self, graph: ThoughtGraph) -> Dict[str, Any]:
        """
        Convert ThoughtGraph to serializable state dict.

        Args:
            graph: ThoughtGraph to serialize

        Returns:
            State dictionary
        """
        return {
            'nodes': {
                node_id: {
                    'node_type': node.node_type.value,
                    'content': node.content,
                    'properties': node.properties,
                    'metadata': node.metadata,
                }
                for node_id, node in graph.nodes.items()
            },
            'edges': [
                {
                    'source_id': edge.source_id,
                    'target_id': edge.target_id,
                    'edge_type': edge.edge_type.value,
                    'weight': edge.weight,
                    'confidence': edge.confidence,
                    'bidirectional': edge.bidirectional,
                }
                for edge in graph.edges
            ],
            'clusters': {
                cluster_id: {
                    'name': cluster.name,
                    'node_ids': list(cluster.node_ids),
                    'properties': cluster.properties,
                }
                for cluster_id, cluster in graph.clusters.items()
            },
        }

    def _state_to_graph(self, state: Dict[str, Any]) -> ThoughtGraph:
        """
        Reconstruct ThoughtGraph from state dict.

        Args:
            state: State dictionary from snapshot

        Returns:
            Reconstructed ThoughtGraph
        """
        graph = ThoughtGraph()

        # Restore nodes
        for node_id, node_data in state.get('nodes', {}).items():
            node_type = NodeType(node_data['node_type'])
            graph.add_node(
                node_id,
                node_type,
                node_data['content'],
                node_data.get('properties', {}),
                node_data.get('metadata', {}),
            )

        # Restore edges (deduplicate by checking if already exists)
        seen_edges = set()
        for edge_data in state.get('edges', []):
            edge_key = (
                edge_data['source_id'],
                edge_data['target_id'],
                edge_data['edge_type']
            )
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                try:
                    edge_type = EdgeType(edge_data['edge_type'])
                    graph.add_edge(
                        edge_data['source_id'],
                        edge_data['target_id'],
                        edge_type,
                        edge_data.get('weight', 1.0),
                        edge_data.get('confidence', 1.0),
                        edge_data.get('bidirectional', False),
                    )
                except ValueError:
                    # Node doesn't exist, skip edge
                    pass

        # Restore clusters
        for cluster_id, cluster_data in state.get('clusters', {}).items():
            cluster = graph.add_cluster(
                cluster_id,
                cluster_data['name'],
                set(cluster_data.get('node_ids', [])),
            )
            cluster.properties.update(cluster_data.get('properties', {}))

        return graph

    # ==========================================================================
    # UTILITY METHODS
    # ==========================================================================

    def get_entry_count(self) -> int:
        """
        Get total number of entries in current WAL.

        Returns:
            Entry count
        """
        return self._writer.get_entry_count()

    def get_current_wal_path(self) -> Path:
        """
        Get path to current WAL file.

        Returns:
            Path to current WAL file
        """
        return self._writer.get_current_wal_path()

    def compact_wal(self, graph: ThoughtGraph) -> str:
        """
        Compact WAL by creating snapshot and removing old entries.

        Args:
            graph: Current graph state

        Returns:
            Snapshot ID
        """
        state = self._graph_to_state(graph)
        return self._snapshot_mgr.compact_wal(state)


# ==============================================================================
# GRAPH RECOVERY
# ==============================================================================

@dataclass
class GraphRecoveryResult:
    """Result of multi-level graph recovery."""

    success: bool
    level_used: int  # 1-4, indicates which recovery level succeeded
    nodes_recovered: int = 0
    edges_recovered: int = 0
    errors: List[str] = field(default_factory=list)
    graph: Optional[ThoughtGraph] = None
    recovery_method: str = ""  # Human-readable description of method used
    duration_ms: float = 0.0  # Time taken for recovery

    def __str__(self) -> str:
        """Human-readable recovery summary."""
        status = "SUCCESS" if self.success else "FAILED"
        lines = [
            f"Recovery {status}",
            f"Level: {self.level_used} ({self.recovery_method})",
            f"Nodes: {self.nodes_recovered}",
            f"Edges: {self.edges_recovered}",
            f"Duration: {self.duration_ms:.2f}ms",
        ]
        if self.errors:
            lines.append(f"Errors: {len(self.errors)}")
            for err in self.errors[:3]:  # Show first 3 errors
                lines.append(f"  - {err}")
        return "\n".join(lines)


@dataclass
class GraphSnapshot:
    """Metadata for a graph snapshot."""

    snapshot_id: str
    timestamp: str
    node_count: int
    edge_count: int
    size_bytes: int
    checksum: str
    path: Path

    def verify_checksum(self) -> bool:
        """
        Verify snapshot file integrity.

        Returns:
            True if checksum is valid, False otherwise
        """
        if not self.path.exists():
            return False

        # Compute SHA256 of file
        sha256 = hashlib.sha256()
        try:
            if self.path.suffix == '.gz':
                import gzip
                with gzip.open(self.path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b''):
                        sha256.update(chunk)
            else:
                with open(self.path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b''):
                        sha256.update(chunk)

            computed = sha256.hexdigest()[:16]
            return computed == self.checksum
        except (IOError, OSError):
            return False


class GraphRecovery:
    """
    Multi-level cascading recovery for ThoughtGraph.

    Implements 4-level fault-tolerant recovery:
    1. WAL Replay - fastest, most recent (load snapshot + replay WAL)
    2. Snapshot Rollback - try previous snapshots if WAL fails
    3. Git History Recovery - restore from source control
    4. Chunk Reconstruction - rebuild from chunk files (slowest, most thorough)

    Each level is attempted only if the previous level fails, ensuring
    graph state can always be recovered even after severe corruption.

    Example:
        >>> recovery = GraphRecovery(wal_dir='graph_wal', chunks_dir='graph_chunks')
        >>> if recovery.needs_recovery():
        ...     result = recovery.recover()
        ...     if result.success:
        ...         print(f"Recovered {result.nodes_recovered} nodes using Level {result.level_used}")
    """

    def __init__(
        self,
        wal_dir: str,
        chunks_dir: Optional[str] = None,
        max_snapshots: int = 3
    ):
        """
        Initialize graph recovery manager.

        Args:
            wal_dir: Directory containing WAL and snapshots
            chunks_dir: Optional directory containing chunk files for Level 4 recovery
            max_snapshots: Number of snapshot generations to keep (default: 3)
        """
        self.wal_dir = Path(wal_dir)
        self.chunks_dir = Path(chunks_dir) if chunks_dir else None
        self.max_snapshots = max_snapshots

        # Subdirectories
        self.snapshots_dir = self.wal_dir / "snapshots"
        self.logs_dir = self.wal_dir / "logs"

        # Create directories if needed
        self.wal_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)

        # Recovery helpers
        self.snapshot_mgr = SnapshotManager(str(self.wal_dir), max_snapshots)
        from cortical.wal import WALRecovery
        self.wal_recovery = WALRecovery(str(self.wal_dir))

    # ==========================================================================
    # PUBLIC API
    # ==========================================================================

    def needs_recovery(self) -> bool:
        """
        Check if recovery is needed.

        Returns:
            True if any recovery mechanism indicates need for recovery
        """
        # Check WAL
        if self.wal_recovery.needs_recovery():
            return True

        # Check for corrupted snapshots
        snapshots = self._list_graph_snapshots()
        if snapshots:
            latest = snapshots[-1]
            if not latest.verify_checksum():
                return True

        return False

    def recover(self) -> GraphRecoveryResult:
        """
        Perform multi-level cascading recovery.

        Attempts each level in order until one succeeds:
        1. WAL Replay (fastest, most recent)
        2. Snapshot Rollback (previous good states)
        3. Git History Recovery (source control restore)
        4. Chunk Reconstruction (slowest, most thorough)

        Returns:
            GraphRecoveryResult with recovery status and recovered graph
        """
        start_time = datetime.now()

        # Try Level 1: WAL Replay
        result = self._level1_wal_replay()
        if result.success:
            result.duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            return result

        # Try Level 2: Snapshot Rollback
        result = self._level2_snapshot_rollback()
        if result.success:
            result.duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            return result

        # Try Level 3: Git History Recovery
        result = self._level3_git_recovery()
        if result.success:
            result.duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            return result

        # Try Level 4: Chunk Reconstruction
        result = self._level4_chunk_reconstruct()
        result.duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        return result

    def verify_graph_integrity(self, graph: ThoughtGraph) -> List[str]:
        """
        Check graph for corruption and integrity issues.

        Performs comprehensive validation:
        - Orphaned edges (referencing non-existent nodes)
        - Index consistency (edge indices match actual edges)
        - Duplicate nodes
        - Self-loops (optional warning)
        - Cluster integrity (clusters reference valid nodes)

        Args:
            graph: ThoughtGraph to verify

        Returns:
            List of issue descriptions (empty list if no issues found)
        """
        issues = []

        # Check for orphaned edges (edges referencing non-existent nodes)
        for edge in graph.edges:
            if edge.source_id not in graph.nodes:
                issues.append(f"Edge references missing source node: {edge.source_id}")
            if edge.target_id not in graph.nodes:
                issues.append(f"Edge references missing target node: {edge.target_id}")

        # Check for index consistency
        for node_id in graph.nodes:
            # Verify outgoing edges in index
            indexed_out = graph._edges_from.get(node_id, [])
            actual_out = [e for e in graph.edges if e.source_id == node_id]
            if len(indexed_out) != len(actual_out):
                issues.append(
                    f"Node {node_id}: outgoing edge index mismatch "
                    f"(indexed: {len(indexed_out)}, actual: {len(actual_out)})"
                )

            # Verify incoming edges in index
            indexed_in = graph._edges_to.get(node_id, [])
            actual_in = [e for e in graph.edges if e.target_id == node_id]
            if len(indexed_in) != len(actual_in):
                issues.append(
                    f"Node {node_id}: incoming edge index mismatch "
                    f"(indexed: {len(indexed_in)}, actual: {len(actual_in)})"
                )

        # Check for duplicate nodes
        if len(graph.nodes) != len(set(graph.nodes.keys())):
            issues.append("Duplicate node IDs detected")

        # Check for self-loops (optional - may be valid in some cases)
        for edge in graph.edges:
            if edge.source_id == edge.target_id:
                issues.append(f"Self-loop detected: {edge.source_id}")

        # Check cluster integrity
        for cluster_id, cluster in graph.clusters.items():
            for node_id in cluster.node_ids:
                if node_id not in graph.nodes:
                    issues.append(
                        f"Cluster {cluster_id} references missing node: {node_id}"
                    )

        return issues

    # ==========================================================================
    # LEVEL 1: WAL REPLAY
    # ==========================================================================

    def _level1_wal_replay(self) -> GraphRecoveryResult:
        """
        Level 1 Recovery: Load latest snapshot and replay WAL entries.

        This is the fastest recovery method and preserves the most recent state.
        Loads the latest snapshot and applies all WAL entries since that snapshot.

        Returns:
            GraphRecoveryResult indicating success or failure
        """
        result = GraphRecoveryResult(
            success=False,
            level_used=1,
            recovery_method="WAL Replay"
        )

        try:
            # Load latest snapshot
            snapshot_data = self.snapshot_mgr.load_snapshot()
            if not snapshot_data:
                result.errors.append("No snapshot found for WAL replay")
                return result

            # Reconstruct graph from snapshot
            graph = self._graph_from_snapshot(snapshot_data)
            if not graph:
                result.errors.append("Failed to reconstruct graph from snapshot")
                return result

            # Get WAL entries to replay
            wal_ref = snapshot_data.get('wal_reference', {})
            wal_file = wal_ref.get('wal_file', '')
            wal_offset = wal_ref.get('wal_offset', 0)

            # Replay WAL entries
            replayed = 0
            if wal_file:
                wal_writer = WALWriter(str(self.wal_dir))

                for entry in wal_writer.get_entries_since(wal_file, wal_offset):
                    try:
                        self._apply_wal_entry(entry, graph)
                        replayed += 1
                    except (KeyError, ValueError, AttributeError) as e:
                        result.errors.append(f"Error replaying WAL entry: {e}")

            # Verify integrity
            integrity_issues = self.verify_graph_integrity(graph)
            if integrity_issues:
                result.errors.extend(integrity_issues)
                return result

            # Success!
            result.success = True
            result.graph = graph
            result.nodes_recovered = graph.node_count()
            result.edges_recovered = graph.edge_count()

            return result

        except Exception as e:
            result.errors.append(f"Level 1 failed: {e}")
            return result

    # ==========================================================================
    # LEVEL 2: SNAPSHOT ROLLBACK
    # ==========================================================================

    def _level2_snapshot_rollback(self) -> GraphRecoveryResult:
        """
        Level 2 Recovery: Try previous snapshots.

        If the latest snapshot is corrupted or WAL replay fails,
        try loading previous snapshot generations (up to max_snapshots).

        Returns:
            GraphRecoveryResult indicating success or failure
        """
        result = GraphRecoveryResult(
            success=False,
            level_used=2,
            recovery_method="Snapshot Rollback"
        )

        try:
            # List all available snapshots
            snapshots = self._list_graph_snapshots()
            if not snapshots:
                result.errors.append("No snapshots available for rollback")
                return result

            # Try each snapshot from newest to oldest
            for snapshot_info in reversed(snapshots):
                # Verify checksum
                if not snapshot_info.verify_checksum():
                    result.errors.append(
                        f"Snapshot {snapshot_info.snapshot_id} failed checksum verification"
                    )
                    continue

                # Try to load snapshot
                try:
                    snapshot_data = self.snapshot_mgr.load_snapshot(snapshot_info.snapshot_id)
                    if not snapshot_data:
                        result.errors.append(
                            f"Failed to load snapshot {snapshot_info.snapshot_id}"
                        )
                        continue

                    # Reconstruct graph
                    graph = self._graph_from_snapshot(snapshot_data)
                    if not graph:
                        result.errors.append(
                            f"Failed to reconstruct graph from {snapshot_info.snapshot_id}"
                        )
                        continue

                    # Verify integrity
                    integrity_issues = self.verify_graph_integrity(graph)
                    if integrity_issues:
                        result.errors.append(
                            f"Snapshot {snapshot_info.snapshot_id} has integrity issues"
                        )
                        continue

                    # Success with this snapshot!
                    result.success = True
                    result.graph = graph
                    result.nodes_recovered = graph.node_count()
                    result.edges_recovered = graph.edge_count()
                    result.recovery_method += f" (using {snapshot_info.snapshot_id})"

                    return result

                except Exception as e:
                    result.errors.append(
                        f"Error loading snapshot {snapshot_info.snapshot_id}: {e}"
                    )
                    continue

            # All snapshots failed
            result.errors.append("All snapshots failed integrity checks")
            return result

        except Exception as e:
            result.errors.append(f"Level 2 failed: {e}")
            return result

    # ==========================================================================
    # LEVEL 3: GIT HISTORY RECOVERY
    # ==========================================================================

    def _level3_git_recovery(self) -> GraphRecoveryResult:
        """
        Level 3 Recovery: Restore from git history.

        If snapshots are corrupted, search git history for the last
        commit with valid graph state and restore from there.

        Searches up to 50 recent commits that touched snapshot files.

        Returns:
            GraphRecoveryResult indicating success or failure
        """
        result = GraphRecoveryResult(
            success=False,
            level_used=3,
            recovery_method="Git History Recovery"
        )

        try:
            # Check if we're in a git repo
            if not self._is_git_repo():
                result.errors.append("Not in a git repository")
                return result

            # Find commits that modified graph-related files
            graph_commits = self._find_graph_commits()
            if not graph_commits:
                result.errors.append("No graph-related commits found in git history")
                return result

            # Try each commit from newest to oldest
            for commit_sha, commit_msg in graph_commits:
                try:
                    # Get snapshot content from this commit
                    snapshot_files = self._get_snapshot_files_at_commit(commit_sha)
                    if not snapshot_files:
                        continue

                    # Try each snapshot file
                    for snapshot_file in snapshot_files:
                        try:
                            snapshot_data = self._load_snapshot_from_commit(
                                commit_sha, snapshot_file
                            )
                            if not snapshot_data:
                                continue

                            # Reconstruct graph
                            graph = self._graph_from_snapshot(snapshot_data)
                            if not graph:
                                continue

                            # Verify integrity
                            integrity_issues = self.verify_graph_integrity(graph)
                            if integrity_issues:
                                continue

                            # Success!
                            result.success = True
                            result.graph = graph
                            result.nodes_recovered = graph.node_count()
                            result.edges_recovered = graph.edge_count()
                            result.recovery_method += f" (from commit {commit_sha[:8]})"

                            return result

                        except Exception as e:
                            result.errors.append(
                                f"Error loading snapshot from commit {commit_sha[:8]}: {e}"
                            )
                            continue

                except Exception as e:
                    result.errors.append(f"Error processing commit {commit_sha[:8]}: {e}")
                    continue

            # All commits failed
            result.errors.append("No valid graph state found in git history")
            return result

        except Exception as e:
            result.errors.append(f"Level 3 failed: {e}")
            return result

    # ==========================================================================
    # LEVEL 4: CHUNK RECONSTRUCTION
    # ==========================================================================

    def _level4_chunk_reconstruct(self) -> GraphRecoveryResult:
        """
        Level 4 Recovery: Rebuild graph from chunk files.

        This is the slowest but most thorough recovery method.
        Reads all chunk files in chronological order and replays
        all graph operations to rebuild state from scratch.

        Returns:
            GraphRecoveryResult indicating success or failure
        """
        result = GraphRecoveryResult(
            success=False,
            level_used=4,
            recovery_method="Chunk Reconstruction"
        )

        try:
            # Check if chunks directory exists
            if not self.chunks_dir or not self.chunks_dir.exists():
                result.errors.append("Chunks directory not available")
                return result

            # Find all chunk files
            chunk_files = sorted(self.chunks_dir.glob("*.json"))
            if not chunk_files:
                result.errors.append("No chunk files found")
                return result

            # Create empty graph
            graph = ThoughtGraph()

            # Replay each chunk in order
            chunks_processed = 0
            for chunk_file in chunk_files:
                try:
                    # Load chunk
                    with open(chunk_file, 'r') as f:
                        chunk_data = json.load(f)

                    # Process operations
                    operations = chunk_data.get('operations', [])
                    for op in operations:
                        try:
                            self._apply_chunk_operation(op, graph)
                        except Exception as e:
                            result.errors.append(
                                f"Error applying operation from {chunk_file.name}: {e}"
                            )

                    chunks_processed += 1

                except Exception as e:
                    result.errors.append(f"Error loading chunk {chunk_file.name}: {e}")
                    continue

            # Verify we got something
            if graph.node_count() == 0:
                result.errors.append("Chunk reconstruction produced empty graph")
                return result

            # Verify integrity
            integrity_issues = self.verify_graph_integrity(graph)
            if integrity_issues:
                result.errors.extend(integrity_issues)
                # Don't fail - partial reconstruction may be useful

            # Success (even if with errors)
            result.success = True
            result.graph = graph
            result.nodes_recovered = graph.node_count()
            result.edges_recovered = graph.edge_count()
            result.recovery_method += f" ({chunks_processed} chunks processed)"

            return result

        except Exception as e:
            result.errors.append(f"Level 4 failed: {e}")
            return result

    # ==========================================================================
    # HELPER METHODS
    # ==========================================================================

    def _list_graph_snapshots(self) -> List[GraphSnapshot]:
        """List all available graph snapshots with metadata."""
        import gzip

        snapshots = []

        for path in sorted(self.snapshots_dir.glob("snap_*.json*")):
            try:
                # Load to get metadata
                if path.suffix == '.gz':
                    with gzip.open(path, 'rt', encoding='utf-8') as f:
                        data = json.load(f)
                else:
                    with open(path, 'r') as f:
                        data = json.load(f)

                # Extract graph state
                state = data.get('state', {})
                graph_data = state.get('graph', {})

                # Compute checksum
                checksum = hashlib.sha256()
                if path.suffix == '.gz':
                    with gzip.open(path, 'rb') as f:
                        checksum.update(f.read())
                else:
                    with open(path, 'rb') as f:
                        checksum.update(f.read())

                snapshot = GraphSnapshot(
                    snapshot_id=data.get('snapshot_id', path.stem),
                    timestamp=data.get('timestamp', ''),
                    node_count=len(graph_data.get('nodes', {})),
                    edge_count=len(graph_data.get('edges', [])),
                    size_bytes=path.stat().st_size,
                    checksum=checksum.hexdigest()[:16],
                    path=path,
                )
                snapshots.append(snapshot)

            except (json.JSONDecodeError, KeyError, OSError):
                continue

        return snapshots

    def _graph_from_snapshot(self, snapshot_data: Dict[str, Any]) -> Optional[ThoughtGraph]:
        """
        Reconstruct ThoughtGraph from snapshot data.

        Args:
            snapshot_data: Snapshot dictionary containing state

        Returns:
            ThoughtGraph instance or None if reconstruction fails
        """
        try:
            state = snapshot_data.get('state', {})
            graph_data = state.get('graph', {})

            if not graph_data:
                return None

            # Create empty graph
            graph = ThoughtGraph()

            # Restore nodes
            nodes_data = graph_data.get('nodes', {})
            for node_id, node_dict in nodes_data.items():
                graph.add_node(
                    node_id=node_id,
                    node_type=NodeType(node_dict.get('node_type', 'concept')),
                    content=node_dict.get('content', ''),
                    properties=node_dict.get('properties', {}),
                    metadata=node_dict.get('metadata', {}),
                )

            # Restore edges
            edges_data = graph_data.get('edges', [])
            for edge_dict in edges_data:
                try:
                    graph.add_edge(
                        from_id=edge_dict.get('source_id'),
                        to_id=edge_dict.get('target_id'),
                        edge_type=EdgeType(edge_dict.get('edge_type', 'relates_to')),
                        weight=edge_dict.get('weight', 1.0),
                        confidence=edge_dict.get('confidence', 1.0),
                        bidirectional=edge_dict.get('bidirectional', False),
                    )
                except ValueError:
                    # Skip edges referencing non-existent nodes
                    continue

            return graph

        except (KeyError, TypeError, AttributeError, ValueError) as e:
            # Failed to reconstruct graph from snapshot data due to:
            # - KeyError: Missing expected keys in snapshot dict
            # - TypeError: Invalid types for NodeType/EdgeType enum values
            # - ValueError: Invalid enum values (e.g., 'invalid_type' for NodeType)
            # - AttributeError: Missing methods/attributes during reconstruction
            return None

    def _apply_wal_entry(self, entry: WALEntry, graph: ThoughtGraph) -> None:
        """
        Apply a WAL entry to graph state.

        Args:
            entry: WALEntry to apply
            graph: ThoughtGraph to modify
        """
        operation = entry.operation

        if operation == 'add_node':
            payload = entry.payload
            graph.add_node(
                node_id=payload.get('node_id'),
                node_type=NodeType(payload.get('node_type', 'concept')),
                content=payload.get('content', ''),
                properties=payload.get('properties', {}),
                metadata=payload.get('metadata', {}),
            )
        elif operation == 'add_edge':
            payload = entry.payload
            graph.add_edge(
                from_id=payload.get('from_id'),
                to_id=payload.get('to_id'),
                edge_type=EdgeType(payload.get('edge_type', 'relates_to')),
                weight=payload.get('weight', 1.0),
                confidence=payload.get('confidence', 1.0),
            )
        elif operation == 'remove_node':
            payload = entry.payload
            try:
                graph.remove_node(payload.get('node_id'))
            except ValueError:
                pass
        elif operation == 'remove_edge':
            payload = entry.payload
            graph.remove_edge(
                from_id=payload.get('from_id'),
                to_id=payload.get('to_id'),
                edge_type=EdgeType(payload.get('edge_type', 'relates_to')),
            )

    def _apply_chunk_operation(self, operation: Dict[str, Any], graph: ThoughtGraph) -> None:
        """
        Apply a chunk operation to graph.

        Args:
            operation: Operation dictionary from chunk file
            graph: ThoughtGraph to modify
        """
        op_type = operation.get('op')

        if op_type == 'add_node':
            graph.add_node(
                node_id=operation.get('node_id'),
                node_type=NodeType(operation.get('node_type', 'concept')),
                content=operation.get('content', ''),
                properties=operation.get('properties', {}),
                metadata=operation.get('metadata', {}),
            )
        elif op_type == 'add_edge':
            graph.add_edge(
                from_id=operation.get('from_id'),
                to_id=operation.get('to_id'),
                edge_type=EdgeType(operation.get('edge_type', 'relates_to')),
                weight=operation.get('weight', 1.0),
                confidence=operation.get('confidence', 1.0),
            )
        elif op_type == 'remove_node':
            try:
                graph.remove_node(operation.get('node_id'))
            except ValueError:
                pass
        elif op_type == 'remove_edge':
            graph.remove_edge(
                from_id=operation.get('from_id'),
                to_id=operation.get('to_id'),
                edge_type=EdgeType(operation.get('edge_type', 'relates_to')),
            )

    def _is_git_repo(self) -> bool:
        """Check if current directory is in a git repository."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--git-dir'],
                capture_output=True,
                timeout=5,
                cwd=self.wal_dir,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _find_graph_commits(self, limit: int = 50) -> List[tuple[str, str]]:
        """
        Find commits that modified graph-related files.

        Args:
            limit: Maximum number of commits to check

        Returns:
            List of (commit_sha, commit_message) tuples
        """
        try:
            # Look for commits touching snapshot files
            result = subprocess.run(
                [
                    'git', 'log',
                    f'-{limit}',
                    '--format=%H|%s',
                    '--',
                    str(self.snapshots_dir),
                ],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self.wal_dir,
            )

            if result.returncode != 0:
                return []

            commits = []
            for line in result.stdout.strip().split('\n'):
                if '|' in line:
                    sha, msg = line.split('|', 1)
                    commits.append((sha.strip(), msg.strip()))

            return commits

        except (subprocess.TimeoutExpired, FileNotFoundError):
            return []

    def _get_snapshot_files_at_commit(self, commit_sha: str) -> List[str]:
        """
        Get list of snapshot files at a specific commit.

        Args:
            commit_sha: Git commit SHA

        Returns:
            List of snapshot file paths
        """
        try:
            result = subprocess.run(
                [
                    'git', 'ls-tree', '--name-only', '-r',
                    commit_sha,
                    str(self.snapshots_dir.relative_to(self.wal_dir.parent)),
                ],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=self.wal_dir.parent,
            )

            if result.returncode != 0:
                return []

            return [
                line.strip() for line in result.stdout.strip().split('\n')
                if line.strip() and line.strip().endswith(('.json', '.json.gz'))
            ]

        except (subprocess.TimeoutExpired, FileNotFoundError):
            return []

    def _load_snapshot_from_commit(
        self,
        commit_sha: str,
        snapshot_file: str
    ) -> Optional[Dict[str, Any]]:
        """
        Load snapshot content from a specific git commit.

        Args:
            commit_sha: Git commit SHA
            snapshot_file: Path to snapshot file in repository

        Returns:
            Snapshot data dictionary or None if loading fails
        """
        try:
            import gzip
            import io

            result = subprocess.run(
                ['git', 'show', f'{commit_sha}:{snapshot_file}'],
                capture_output=True,
                timeout=10,
                cwd=self.wal_dir.parent,
            )

            if result.returncode != 0:
                return None

            # Decompress if needed
            if snapshot_file.endswith('.gz'):
                with gzip.open(io.BytesIO(result.stdout), 'rt', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return json.loads(result.stdout.decode('utf-8'))

        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
            return None
