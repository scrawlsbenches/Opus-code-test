"""
CDG configuration types.

Provides configuration dataclasses for the Cortical Distributed Graph,
controlling durability modes, partition settings, and operational parameters.

This module defines the configuration options that make CDG a flexible,
configurable storage layer that can serve different use cases:

DEFAULT (ACID-safe):
- transactions=True, wal=True, recovery=full
- WAL-first model: commits are durable before entity files are modified
- Use CDGConfig() for standard safe operation

PRESETS:
- CDGConfig.for_got(): Same as default (full ACID)
- CDGConfig.for_simple_storage(): transactions=False, wal=False (fast, ephemeral)
- CDGConfig.for_high_performance(): Maximum speed, no safety guarantees
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable, Any


class DurabilityMode(Enum):
    """
    Durability mode controlling fsync behavior.

    Controls the trade-off between write performance and data safety:

    - FAST: No fsync, maximum performance, data loss possible on crash
    - BALANCED: Fsync on commit, good balance of safety and performance
    - PARANOID: Fsync on every write, maximum safety, slower writes

    Recommendation:
    - Development: FAST (quick iteration, data is disposable)
    - Testing: BALANCED (catch timing issues, reasonable performance)
    - Production: BALANCED or PARANOID (based on data criticality)
    """
    FAST = "fast"           # No fsync, maximum performance
    BALANCED = "balanced"   # Fsync on commit
    PARANOID = "paranoid"   # Fsync on every write


class IsolationLevel(Enum):
    """
    Transaction isolation level.

    Controls how transactions see concurrent modifications:

    - SNAPSHOT: Transactions see a consistent snapshot from start time
                (recommended, prevents dirty reads and non-repeatable reads)
    - READ_COMMITTED: Transactions see committed changes from other transactions
                      (allows non-repeatable reads but prevents dirty reads)

    Note: CDG currently only implements SNAPSHOT isolation.
    READ_COMMITTED is reserved for future use.
    """
    SNAPSHOT = "snapshot"
    READ_COMMITTED = "read_committed"


class RecoveryMode(Enum):
    """
    Recovery strategy on startup.

    Controls how aggressively CDG attempts to recover from crashes:

    - NONE: No recovery, fastest startup (use for ephemeral data)
    - CHECKSUM: Verify entity checksums, quarantine corrupt (basic safety)
    - FULL: WAL replay + checksum verification + orphan repair (maximum safety)

    Recommendation:
    - Development: NONE (fast iteration)
    - Testing: CHECKSUM (catch corruption)
    - Production: FULL (complete crash recovery)
    """
    NONE = "none"
    CHECKSUM = "checksum"
    FULL = "full"


class OrphanStrategy(Enum):
    """
    Strategy for handling orphaned entities.

    Orphans are entity files that exist on disk but have no corresponding
    WAL record. This can happen from:
    - Pre-transaction era data
    - Manual file edits
    - Crashes during non-WAL writes

    Strategies:
    - FAIL: Raise error, refuse to start (strict mode)
    - DELETE: Remove orphaned files (clean slate)
    - REPAIR: Adopt orphans by creating synthetic WAL entries (preserve data)
    """
    FAIL = "fail"
    DELETE = "delete"
    REPAIR = "repair"


@dataclass
class CDGConfig:
    """
    Configuration for CDG store instances.

    Controls all operational parameters for a CDG store, including
    durability, partitioning, and validation settings.

    Example:
        config = CDGConfig(
            durability=DurabilityMode.BALANCED,
            partition_count=4,
            validate_on_write=True
        )
        store = CDGStore(Path("./data"), config)

    Attributes:
        durability: Fsync behavior mode
        partition_count: Number of partitions (1 = single partition)
        validate_on_write: Validate entities against schema before writing
        enable_wal: Enable Write-Ahead Log for crash recovery
        enable_history: Enable historical snapshots for MVCC
        compression_enabled: Compress stored entities
        encryption_enabled: Encrypt stored entities
    """

    # Durability settings
    durability: DurabilityMode = DurabilityMode.BALANCED

    # Partition settings
    partition_count: int = 1
    partition_strategy: str = "hash"  # "hash" or "range"

    # Validation settings
    validate_on_write: bool = True
    strict_edge_types: bool = True

    # Transaction settings
    # Default: ACID transactions enabled for data safety
    # Disable for ephemeral data or maximum performance
    transactions_enabled: bool = True  # Enable begin/commit/rollback semantics
    isolation_level: IsolationLevel = IsolationLevel.SNAPSHOT
    transaction_timeout_seconds: int = 300  # 5 minutes default

    # WAL settings
    # Default: WAL enabled for crash recovery
    # WAL-first model: commit is durable in WAL before entities are modified
    enable_wal: bool = True  # Enable write-ahead log for crash recovery
    wal_archive_enabled: bool = True
    wal_archive_threshold: int = 1000  # Archive after N entries

    # Recovery settings
    # Default: FULL recovery for maximum safety
    recovery_mode: RecoveryMode = RecoveryMode.FULL
    orphan_strategy: OrphanStrategy = OrphanStrategy.REPAIR
    auto_recover_on_startup: bool = True

    # History settings (for MVCC)
    enable_history: bool = True
    history_retention_days: int = 30

    # Index callback (for GoT integration)
    # Called during recovery to rebuild indexes
    # Signature: Callable[[Path], int] where Path is store_dir, returns count
    index_rebuild_callback: Optional[Callable[[Any], int]] = None

    # Storage optimization
    compression_enabled: bool = False
    encryption_enabled: bool = False

    # Super-node handling
    super_node_warning_threshold: int = 10_000
    super_node_overflow_threshold: int = 100_000
    super_node_partition_threshold: int = 1_000_000

    # Performance tuning
    read_cache_enabled: bool = True
    read_cache_max_items: int = 10_000
    write_buffer_size: int = 1000

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.partition_count < 1:
            raise ValueError("partition_count must be at least 1")

        if self.partition_strategy not in ("hash", "range"):
            raise ValueError("partition_strategy must be 'hash' or 'range'")

        if self.super_node_warning_threshold >= self.super_node_overflow_threshold:
            raise ValueError(
                "super_node_warning_threshold must be less than overflow_threshold"
            )

        if self.super_node_overflow_threshold >= self.super_node_partition_threshold:
            raise ValueError(
                "super_node_overflow_threshold must be less than partition_threshold"
            )

        # Validate transaction/WAL consistency
        if self.enable_wal and not self.transactions_enabled:
            # WAL without transactions is allowed but unusual
            # WAL can still be used for crash recovery of non-transactional writes
            pass

    @classmethod
    def for_got(cls) -> "CDGConfig":
        """
        Pre-configured for GoT workloads.

        Enables full ACID transactions with WAL-based crash recovery.
        This is the configuration that GoT uses when delegating to CDG.

        Features enabled:
        - Transactions (begin/commit/rollback)
        - Write-Ahead Log (crash recovery)
        - Full recovery mode (WAL replay + orphan repair)
        - Snapshot isolation

        Returns:
            CDGConfig configured for GoT
        """
        return cls(
            transactions_enabled=True,
            isolation_level=IsolationLevel.SNAPSHOT,
            enable_wal=True,
            wal_archive_enabled=True,
            recovery_mode=RecoveryMode.FULL,
            orphan_strategy=OrphanStrategy.REPAIR,
            auto_recover_on_startup=True,
            durability=DurabilityMode.BALANCED,
            enable_history=True,
        )

    @classmethod
    def for_simple_storage(cls) -> "CDGConfig":
        """
        Pre-configured for simple storage without transactions.

        Provides basic entity storage with checksum verification
        but no transaction overhead. Good for simple applications.

        Features enabled:
        - Checksum verification (data integrity)
        - History (MVCC for snapshot reads)
        - No transactions (writes are auto-committed)
        - No WAL (no crash recovery)

        Returns:
            CDGConfig for simple storage
        """
        return cls(
            transactions_enabled=False,
            enable_wal=False,
            recovery_mode=RecoveryMode.CHECKSUM,
            orphan_strategy=OrphanStrategy.REPAIR,
            auto_recover_on_startup=True,
            durability=DurabilityMode.BALANCED,
            enable_history=True,
        )

    @classmethod
    def for_high_performance(cls) -> "CDGConfig":
        """
        Pre-configured for maximum throughput, ephemeral data.

        Disables all safety features for maximum write speed.
        Use only when data loss is acceptable.

        Features disabled:
        - Transactions
        - WAL
        - Recovery
        - Fsync

        Returns:
            CDGConfig for high-performance ephemeral storage
        """
        return cls(
            transactions_enabled=False,
            enable_wal=False,
            recovery_mode=RecoveryMode.NONE,
            orphan_strategy=OrphanStrategy.DELETE,
            auto_recover_on_startup=False,
            durability=DurabilityMode.FAST,
            enable_history=False,
            validate_on_write=False,
        )


@dataclass
class PerformanceContract:
    """
    Performance contract for a specific environment tier.

    Defines expected latency and throughput targets that should
    be monitored and defended in CI.

    Attributes:
        tier: Environment tier (development, staging, production)
        read_p50_ms: 50th percentile read latency target
        read_p99_ms: 99th percentile read latency target
        write_p50_ms: 50th percentile write latency target
        write_p99_ms: 99th percentile write latency target
        throughput_ops_per_sec: Minimum throughput target
    """

    tier: str = "development"

    # Read latency targets (milliseconds)
    read_p50_ms: int = 50
    read_p99_ms: int = 200

    # Write latency targets (milliseconds)
    write_p50_ms: int = 100
    write_p99_ms: int = 500

    # Throughput targets
    throughput_ops_per_sec: int = 1000

    # Whether violations block the build
    violations_block_build: bool = False

    @classmethod
    def development(cls) -> "PerformanceContract":
        """Best-effort development contract."""
        return cls(
            tier="development",
            read_p50_ms=100,
            read_p99_ms=500,
            write_p50_ms=200,
            write_p99_ms=1000,
            throughput_ops_per_sec=500,
            violations_block_build=False,
        )

    @classmethod
    def staging(cls) -> "PerformanceContract":
        """Soft targets for staging."""
        return cls(
            tier="staging",
            read_p50_ms=50,
            read_p99_ms=200,
            write_p50_ms=100,
            write_p99_ms=500,
            throughput_ops_per_sec=1000,
            violations_block_build=False,
        )

    @classmethod
    def production(cls) -> "PerformanceContract":
        """Hard targets for production."""
        return cls(
            tier="production",
            read_p50_ms=20,
            read_p99_ms=100,
            write_p50_ms=50,
            write_p99_ms=200,
            throughput_ops_per_sec=2000,
            violations_block_build=True,
        )
