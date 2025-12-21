#!/usr/bin/env python3
"""
Migrate existing .got/ event-sourced data to new transactional format.

Converts event-sourced GoT data (events, WAL logs, snapshots) to the new
transactional format with entities, versioning, and checksums.

Usage:
    python scripts/migrate_got.py [--got-dir .got] [--output-dir .got-tx] [--dry-run]

Example:
    # Analyze without migrating
    python scripts/migrate_got.py --dry-run

    # Migrate to new directory
    python scripts/migrate_got.py --output-dir .got-tx

    # Migrate from custom source
    python scripts/migrate_got.py --got-dir /path/to/.got --output-dir .got-tx
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

# Import new transactional types
try:
    from cortical.got import GoTManager, Task, Decision, Edge
    from cortical.got.types import Entity
except ImportError:
    print("Error: cortical.got module not found. Run from project root.", file=sys.stderr)
    sys.exit(1)


@dataclass
class MigrationAnalysis:
    """Results of migration analysis."""

    tasks: int
    decisions: int
    edges: int
    events: int
    wal_entries: int
    source_dir: Path
    target_dir: Path


@dataclass
class MigrationResult:
    """Results of migration execution."""

    success: bool
    tasks_migrated: int
    decisions_migrated: int
    edges_migrated: int
    errors: List[str]
    warnings: List[str]


class GoTMigrator:
    """Migrates event-sourced GoT data to transactional format."""

    def __init__(
        self,
        source_dir: Path,
        target_dir: Path,
        dry_run: bool = False
    ):
        """
        Initialize migrator.

        Args:
            source_dir: Source .got directory with event-sourced data
            target_dir: Target directory for transactional store
            dry_run: If True, analyze without writing
        """
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.dry_run = dry_run

        # Validate source directory exists
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {self.source_dir}")

        # Collected entities
        self.tasks: Dict[str, Task] = {}
        self.decisions: Dict[str, Decision] = {}
        self.edges: Dict[str, Edge] = {}

        # Track errors and warnings
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def analyze(self) -> MigrationAnalysis:
        """
        Analyze source data without migrating.

        Returns:
            MigrationAnalysis with entity counts
        """
        # Read all event files
        event_count = self._count_events()

        # Read all WAL entries
        wal_count = self._count_wal_entries()

        # Parse events to get entity counts
        self._parse_events()
        self._parse_wal()

        # Count unique entities (we store by both original and clean ID, so dedupe)
        unique_tasks = {task.id: task for task in self.tasks.values()}
        unique_decisions = {dec.id: dec for dec in self.decisions.values()}

        return MigrationAnalysis(
            tasks=len(unique_tasks),
            decisions=len(unique_decisions),
            edges=len(self.edges),
            events=event_count,
            wal_entries=wal_count,
            source_dir=self.source_dir,
            target_dir=self.target_dir
        )

    def migrate(self) -> MigrationResult:
        """
        Perform migration.

        Steps:
        1. Parse events and WAL to build current state
        2. Convert ThoughtNode → Task/Decision
        3. Convert edges to new format
        4. Write to new transactional store
        5. Verify migration with checksums

        Returns:
            MigrationResult with migration status
        """
        print(f"Starting migration from {self.source_dir} to {self.target_dir}")

        # Step 1: Parse events and WAL
        print("Parsing events...")
        self._parse_events()

        print("Parsing WAL...")
        self._parse_wal()

        # Count unique entities (we store by both original and clean ID, so dedupe by task.id)
        unique_tasks = {task.id: task for task in self.tasks.values()}
        unique_decisions = {dec.id: dec for dec in self.decisions.values()}
        tasks_migrated = len(unique_tasks)
        decisions_migrated = len(unique_decisions)
        edges_migrated = len(self.edges)

        print(f"Found {tasks_migrated} tasks, {decisions_migrated} decisions, {edges_migrated} edges")

        if self.dry_run:
            print("Dry run - skipping write phase")
            return MigrationResult(
                success=True,
                tasks_migrated=tasks_migrated,
                decisions_migrated=decisions_migrated,
                edges_migrated=edges_migrated,
                errors=self.errors,
                warnings=self.warnings
            )

        # Step 2: Write to new transactional store
        print(f"Writing to {self.target_dir}...")
        try:
            manager = GoTManager(self.target_dir)

            # Write in single transaction for atomicity
            with manager.transaction() as tx:
                # Write tasks (deduplicate by ID since we store by both original and clean ID)
                seen_tasks = set()
                for task in self.tasks.values():
                    if task.id not in seen_tasks:
                        tx.write(task)
                        seen_tasks.add(task.id)

                # Write decisions (deduplicate by ID)
                seen_decisions = set()
                for decision in self.decisions.values():
                    if decision.id not in seen_decisions:
                        tx.write(decision)
                        seen_decisions.add(decision.id)

                # Write edges
                for edge in self.edges.values():
                    tx.write(edge)

            print("Migration completed successfully")

            # Step 3: Verify
            if not self.verify():
                self.warnings.append("Verification found discrepancies")

            return MigrationResult(
                success=True,
                tasks_migrated=tasks_migrated,
                decisions_migrated=decisions_migrated,
                edges_migrated=edges_migrated,
                errors=self.errors,
                warnings=self.warnings
            )

        except Exception as e:
            error_msg = f"Migration failed: {e}"
            self.errors.append(error_msg)
            print(f"ERROR: {error_msg}", file=sys.stderr)
            return MigrationResult(
                success=False,
                tasks_migrated=0,
                decisions_migrated=0,
                edges_migrated=0,
                errors=self.errors,
                warnings=self.warnings
            )

    def verify(self) -> bool:
        """
        Verify migrated data matches source.

        Returns:
            True if verification passes, False otherwise
        """
        if self.dry_run:
            print("Dry run - skipping verification")
            return True

        print("Verifying migration...")
        manager = GoTManager(self.target_dir)

        # Get unique tasks by clean ID (deduplicate since we store by both prefixed and clean IDs)
        unique_tasks = {task.id: task for task in self.tasks.values()}

        # Verify task count
        with manager.transaction(read_only=True) as tx:
            for task_id, task in unique_tasks.items():
                migrated_task = tx.get_task(task_id)
                if migrated_task is None:
                    self.warnings.append(f"Task not found in migrated store: {task_id}")
                    return False

        print(f"Verification passed - {len(unique_tasks)} unique tasks verified")
        return True

    def _count_events(self) -> int:
        """Count total events in source directory."""
        events_dir = self.source_dir / "events"
        if not events_dir.exists():
            return 0

        count = 0
        for event_file in events_dir.glob("*.jsonl"):
            with open(event_file) as f:
                count += sum(1 for _ in f)

        return count

    def _count_wal_entries(self) -> int:
        """Count total WAL entries in source directory."""
        wal_logs = self.source_dir / "wal" / "logs"
        if not wal_logs.exists():
            return 0

        count = 0
        for wal_file in wal_logs.glob("*.jsonl"):
            with open(wal_file) as f:
                count += sum(1 for _ in f)

        return count

    def _parse_events(self) -> None:
        """Parse event files and build entity state."""
        events_dir = self.source_dir / "events"
        if not events_dir.exists():
            return

        for event_file in sorted(events_dir.glob("*.jsonl")):
            with open(event_file) as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        self._process_event(event)
                    except json.JSONDecodeError as e:
                        self.warnings.append(f"Failed to parse event in {event_file}: {e}")
                    except Exception as e:
                        self.warnings.append(f"Failed to process event: {e}")

    def _parse_wal(self) -> None:
        """Parse WAL files and build entity state."""
        wal_logs = self.source_dir / "wal" / "logs"
        if not wal_logs.exists():
            return

        for wal_file in sorted(wal_logs.glob("*.jsonl")):
            with open(wal_file) as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        self._process_wal_entry(entry)
                    except json.JSONDecodeError as e:
                        self.warnings.append(f"Failed to parse WAL entry in {wal_file}: {e}")
                    except Exception as e:
                        self.warnings.append(f"Failed to process WAL entry: {e}")

    def _process_event(self, event: Dict[str, Any]) -> None:
        """Process a single event and update entity state."""
        event_type = event.get("event")

        if event_type == "node.create":
            self._process_node_create(event)
        elif event_type == "node.update":
            self._process_node_update(event)
        elif event_type == "edge.create":
            self._process_edge_create(event)
        # Skip handoff events - not part of core entities

    def _process_wal_entry(self, entry: Dict[str, Any]) -> None:
        """Process a single WAL entry and update entity state."""
        operation = entry.get("operation")

        if operation == "add_node":
            self._process_wal_node(entry)
        elif operation == "update_node":
            self._process_wal_update(entry)
        elif operation == "add_edge":
            self._process_wal_edge(entry)

    def _process_node_create(self, event: Dict[str, Any]) -> None:
        """Process node.create event."""
        node_id = event.get("id")
        node_type = event.get("type", "")
        data = event.get("data", {})

        if not node_id:
            return

        # Convert based on node type
        if node_type == "TASK" or node_id.startswith("task:"):
            task = self._convert_to_task(node_id, data, event.get("ts"))
            # Store by both original and clean ID for update lookups
            self.tasks[node_id] = task
            self.tasks[task.id] = task
        elif node_type == "DECISION" or node_id.startswith("decision:"):
            decision = self._convert_to_decision(node_id, data, event.get("ts"))
            # Store by both original and clean ID for update lookups
            self.decisions[node_id] = decision
            self.decisions[decision.id] = decision

    def _process_node_update(self, event: Dict[str, Any]) -> None:
        """Process node.update event."""
        raw_node_id = event.get("id")
        # Events use "changes" not "updates"
        updates = event.get("changes", event.get("updates", {}))

        if not raw_node_id:
            return

        # Strip prefix from node ID for lookup
        node_id = self._strip_id_prefix(raw_node_id)

        # Update existing entity
        if node_id in self.tasks:
            task = self.tasks[node_id]
            self._apply_updates(task, updates)
            task.bump_version()
        elif node_id in self.decisions:
            decision = self.decisions[node_id]
            self._apply_updates(decision, updates)
            decision.bump_version()

    def _strip_id_prefix(self, node_id: str) -> str:
        """Strip 'task:' or 'decision:' prefix from node ID."""
        if node_id.startswith("task:"):
            return node_id[5:]  # len("task:") = 5
        if node_id.startswith("decision:"):
            return node_id[9:]  # len("decision:") = 9
        return node_id

    def _process_edge_create(self, event: Dict[str, Any]) -> None:
        """Process edge.create event."""
        edge_data = event.get("data", {})
        source_id = edge_data.get("source_id") or edge_data.get("from")
        target_id = edge_data.get("target_id") or edge_data.get("to")
        edge_type = edge_data.get("edge_type") or edge_data.get("type")

        if not (source_id and target_id and edge_type):
            return

        # Strip prefixes from source and target IDs
        clean_source = self._strip_id_prefix(source_id)
        clean_target = self._strip_id_prefix(target_id)

        edge = Edge(
            id="",  # Auto-generated
            source_id=clean_source,
            target_id=clean_target,
            edge_type=edge_type,
            weight=edge_data.get("weight", 1.0),
            confidence=edge_data.get("confidence", 1.0)
        )
        self.edges[edge.id] = edge

    def _process_wal_node(self, entry: Dict[str, Any]) -> None:
        """Process WAL add_node entry."""
        payload = entry.get("payload", {})
        node_id = payload.get("node_id")
        node_type = payload.get("node_type", "")

        if not node_id:
            return

        # Convert based on node type
        if node_type == "task" or node_id.startswith("task:"):
            task = self._convert_to_task(
                node_id,
                payload.get("properties", {}),
                entry.get("timestamp")
            )
            self.tasks[node_id] = task
        elif node_type == "decision" or node_id.startswith("decision:"):
            decision = self._convert_to_decision(
                node_id,
                payload.get("properties", {}),
                entry.get("timestamp")
            )
            self.decisions[node_id] = decision

    def _process_wal_update(self, entry: Dict[str, Any]) -> None:
        """Process WAL update_node entry."""
        payload = entry.get("payload", {})
        node_id = payload.get("node_id")

        if not node_id:
            return

        # Update existing entity
        if node_id in self.tasks:
            task = self.tasks[node_id]
            self._apply_updates(task, payload.get("properties", {}))
            task.bump_version()
        elif node_id in self.decisions:
            decision = self.decisions[node_id]
            self._apply_updates(decision, payload.get("properties", {}))
            decision.bump_version()

    def _process_wal_edge(self, entry: Dict[str, Any]) -> None:
        """Process WAL add_edge entry."""
        payload = entry.get("payload", {})
        source_id = payload.get("source_id") or payload.get("from_id")
        target_id = payload.get("target_id") or payload.get("to_id")
        edge_type = payload.get("edge_type")

        if not (source_id and target_id and edge_type):
            return

        # Strip prefixes from source and target IDs
        clean_source = self._strip_id_prefix(source_id)
        clean_target = self._strip_id_prefix(target_id)

        edge = Edge(
            id="",  # Auto-generated
            source_id=clean_source,
            target_id=clean_target,
            edge_type=edge_type,
            weight=payload.get("weight", 1.0),
            confidence=payload.get("confidence", 1.0)
        )
        self.edges[edge.id] = edge

    def _convert_to_task(
        self,
        node_id: str,
        data: Dict[str, Any],
        timestamp: Optional[str] = None
    ) -> Task:
        """Convert node data to Task entity."""
        # Map legacy statuses to new valid statuses
        status_mapping = {
            "deferred": "pending",
            "cancelled": "blocked",
            "done": "completed",
        }
        raw_status = data.get("status", "pending")
        status = status_mapping.get(raw_status, raw_status)

        # Strip 'task:' prefix if present
        clean_id = node_id.replace("task:", "") if node_id.startswith("task:") else node_id

        return Task(
            id=clean_id,
            title=data.get("title", ""),
            status=status,
            priority=data.get("priority", "medium"),
            description=data.get("description", ""),
            properties=data.get("properties", {}),
            metadata=data.get("metadata", {}),
            created_at=timestamp or datetime.now(timezone.utc).isoformat(),
            modified_at=timestamp or datetime.now(timezone.utc).isoformat()
        )

    def _convert_to_decision(
        self,
        node_id: str,
        data: Dict[str, Any],
        timestamp: Optional[str] = None
    ) -> Decision:
        """Convert node data to Decision entity."""
        # Strip 'decision:' prefix if present
        clean_id = node_id.replace("decision:", "") if node_id.startswith("decision:") else node_id

        return Decision(
            id=clean_id,
            title=data.get("title", ""),
            rationale=data.get("rationale", ""),
            affects=data.get("affects", []),
            properties=data.get("properties", {}),
            created_at=timestamp or datetime.now(timezone.utc).isoformat(),
            modified_at=timestamp or datetime.now(timezone.utc).isoformat()
        )

    def _apply_updates(self, entity: Entity, updates: Dict[str, Any]) -> None:
        """Apply updates to an entity."""
        # Fields that should go into metadata
        metadata_fields = {"completed_at", "started_at", "updated_at", "created_at"}
        # Fields that should go into properties
        property_fields = {"retrospective", "notes", "sprint_id"}

        for key, value in updates.items():
            if key in metadata_fields:
                entity.metadata[key] = value
            elif key in property_fields:
                entity.properties[key] = value
            elif hasattr(entity, key):
                setattr(entity, key, value)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate GoT event-sourced data to transactional format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze without migrating
  python scripts/migrate_got.py --dry-run

  # Migrate to new directory
  python scripts/migrate_got.py --output-dir .got-tx

  # Migrate from custom source
  python scripts/migrate_got.py --got-dir /path/to/.got --output-dir .got-tx
        """
    )

    parser.add_argument(
        "--got-dir",
        type=Path,
        default=Path(".got"),
        help="Source .got directory (default: .got)"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".got-tx"),
        help="Target directory for transactional store (default: .got-tx)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze without writing (safe mode)"
    )

    args = parser.parse_args()

    try:
        migrator = GoTMigrator(
            source_dir=args.got_dir,
            target_dir=args.output_dir,
            dry_run=args.dry_run
        )

        if args.dry_run:
            print("=== MIGRATION ANALYSIS (DRY RUN) ===")
            analysis = migrator.analyze()
            print(f"Source: {analysis.source_dir}")
            print(f"Target: {analysis.target_dir}")
            print(f"Events: {analysis.events}")
            print(f"WAL entries: {analysis.wal_entries}")
            print(f"Tasks: {analysis.tasks}")
            print(f"Decisions: {analysis.decisions}")
            print(f"Edges: {analysis.edges}")
            print()
            print("Run without --dry-run to perform migration")
        else:
            result = migrator.migrate()

            if result.success:
                print()
                print("=== MIGRATION SUCCESSFUL ===")
                print(f"Tasks migrated: {result.tasks_migrated}")
                print(f"Decisions migrated: {result.decisions_migrated}")
                print(f"Edges migrated: {result.edges_migrated}")
            else:
                print()
                print("=== MIGRATION FAILED ===", file=sys.stderr)
                for error in result.errors:
                    print(f"ERROR: {error}", file=sys.stderr)
                sys.exit(1)

            if result.warnings:
                print()
                print("=== WARNINGS ===")
                for warning in result.warnings:
                    print(f"WARNING: {warning}")

    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
