"""
CDG configuration types.

Provides configuration dataclasses for the Cortical Distributed Graph,
controlling durability modes, partition settings, and operational parameters.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


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

    # WAL settings
    enable_wal: bool = True
    wal_archive_enabled: bool = True

    # History settings (for MVCC)
    enable_history: bool = True
    history_retention_days: int = 30

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
