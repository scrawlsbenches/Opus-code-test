"""
Configuration for the Cognitive Event Lattice.

Centralizes all magic numbers, strings, thresholds, and tunable parameters.
Designed for dependency injection via the IoC container.

Design Principles:
    1. No magic numbers scattered in code
    2. Sensible defaults that work out of the box
    3. Environment-aware (can load from env vars)
    4. Validation on construction
    5. Immutable after creation (frozen dataclass)

Timezone Policy:
    All timestamps are stored as ISO 8601 with explicit timezone.
    Internal processing uses UTC. Display can be localized.

Example:
    config = CELConfig(
        max_events_before_compaction=5000,
        bloom_filter_size=10000,
    )
    container.register_instance(CELConfig, config)
    lattice = create_lattice(container, config=config)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import timezone
from typing import Any, Dict, FrozenSet, Optional


# =============================================================================
# TIMESTAMP UTILITIES (Timezone-Safe)
# =============================================================================

def utc_now_iso() -> str:
    """
    Get current time as ISO 8601 string with explicit UTC timezone.

    ALWAYS use this instead of datetime.now().isoformat() to ensure
    timezone consistency across distributed systems.

    Returns:
        ISO 8601 string like "2025-12-28T22:30:45.123456+00:00"
    """
    from datetime import datetime
    return datetime.now(timezone.utc).isoformat()


def parse_iso_timestamp(iso_string: str) -> 'datetime':
    """
    Parse ISO 8601 timestamp, handling timezone variations.

    Handles:
        - "2025-12-28T22:30:45.123456+00:00" (explicit UTC)
        - "2025-12-28T22:30:45.123456Z" (Z suffix)
        - "2025-12-28T22:30:45" (naive, assumed UTC)

    Returns:
        datetime with timezone info (always UTC-normalized)
    """
    from datetime import datetime

    # Handle Z suffix (Zulu time = UTC)
    if iso_string.endswith('Z'):
        iso_string = iso_string[:-1] + '+00:00'

    # Try parsing with timezone
    try:
        dt = datetime.fromisoformat(iso_string)
        if dt.tzinfo is None:
            # Naive datetime - assume UTC
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        # Last resort: basic parsing
        from datetime import datetime
        dt = datetime.fromisoformat(iso_string.replace('Z', ''))
        return dt.replace(tzinfo=timezone.utc)


def timestamp_for_storage(dt: 'datetime' = None) -> str:
    """
    Convert datetime to storage format (ISO 8601 with UTC).

    Args:
        dt: datetime to convert (None = now)

    Returns:
        ISO 8601 string with +00:00 suffix
    """
    from datetime import datetime as dt_class

    if dt is None:
        return utc_now_iso()

    # Ensure timezone-aware
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    # Convert to UTC
    utc_dt = dt.astimezone(timezone.utc)
    return utc_dt.isoformat()


# =============================================================================
# CEL CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class CELConfig:
    """
    Configuration for the Cognitive Event Lattice.

    Frozen dataclass - immutable after creation for thread safety.
    All timing values are in milliseconds unless noted.

    Attributes:
        # Event Storage
        max_events_before_compaction: Trigger compaction suggestion
        max_event_content_bytes: Maximum size for event content
        event_id_hash_length: Length of content-addressed IDs

        # Bloom Filter (Semantic Index)
        bloom_filter_size: Number of bits in bloom filter
        bloom_hash_count: Number of hash functions
        bloom_false_positive_rate: Target false positive rate

        # Materialization Cache
        cache_size: Maximum cached entities
        cache_ttl_seconds: Time-to-live for cache entries

        # Health Monitoring
        health_check_interval_seconds: Time between auto-checks
        health_event_count_warn: Warn at this event count
        health_event_count_critical: Critical at this count
        health_orphan_ratio_warn: Warn at this orphan ratio

        # Compaction
        compaction_min_age_days: Don't compact newer events
        compaction_window_hours: Time window for grouping
        compaction_semantic_threshold: Similarity for merging

        # Tracing
        enable_tracing: Enable debug tracing
        trace_sample_rate: Fraction of events to trace (0-1)
        max_trace_depth: Maximum causal chain depth to trace

        # Distributed (future)
        node_id: Unique identifier for this node
        cluster_size_hint: Expected cluster size for tuning
    """

    # -------------------------------------------------------------------------
    # Event Storage
    # -------------------------------------------------------------------------
    max_events_before_compaction: int = 10_000
    max_event_content_bytes: int = 1_048_576  # 1MB
    event_id_hash_length: int = 16  # Characters of SHA256 to use

    # -------------------------------------------------------------------------
    # Bloom Filter (Semantic Index)
    # -------------------------------------------------------------------------
    bloom_filter_size: int = 10_000
    bloom_hash_count: int = 3
    bloom_false_positive_rate: float = 0.01  # 1%

    # -------------------------------------------------------------------------
    # Materialization Cache
    # -------------------------------------------------------------------------
    cache_size: int = 1_000
    cache_ttl_seconds: int = 300  # 5 minutes

    # -------------------------------------------------------------------------
    # Health Monitoring
    # -------------------------------------------------------------------------
    health_check_interval_seconds: int = 300  # 5 minutes
    health_event_count_warn: int = 10_000
    health_event_count_critical: int = 50_000
    health_orphan_ratio_warn: float = 0.01  # 1%
    health_orphan_ratio_critical: float = 0.05  # 5%

    # -------------------------------------------------------------------------
    # Compaction
    # -------------------------------------------------------------------------
    compaction_min_age_days: int = 7
    compaction_window_hours: int = 24
    compaction_semantic_threshold: float = 0.8  # 80% concept overlap
    compaction_chain_max_length: int = 10

    # -------------------------------------------------------------------------
    # Tracing & Debugging
    # -------------------------------------------------------------------------
    enable_tracing: bool = False
    trace_sample_rate: float = 1.0  # Trace all when enabled
    max_trace_depth: int = 100
    trace_include_content: bool = False  # Include event content in traces

    # -------------------------------------------------------------------------
    # Distributed / Extended (Future)
    # -------------------------------------------------------------------------
    node_id: str = "local"
    cluster_size_hint: int = 1

    # -------------------------------------------------------------------------
    # Environment Overrides
    # -------------------------------------------------------------------------

    @classmethod
    def from_environment(cls, prefix: str = "CEL_") -> 'CELConfig':
        """
        Create config from environment variables.

        Args:
            prefix: Environment variable prefix (default: CEL_)

        Example:
            CEL_MAX_EVENTS_BEFORE_COMPACTION=5000
            CEL_ENABLE_TRACING=true
            CEL_NODE_ID=worker-1
        """
        overrides = {}

        # Map env vars to config fields
        env_mappings = {
            f"{prefix}MAX_EVENTS_BEFORE_COMPACTION": ("max_events_before_compaction", int),
            f"{prefix}BLOOM_FILTER_SIZE": ("bloom_filter_size", int),
            f"{prefix}CACHE_SIZE": ("cache_size", int),
            f"{prefix}ENABLE_TRACING": ("enable_tracing", lambda x: x.lower() in ('true', '1', 'yes')),
            f"{prefix}TRACE_SAMPLE_RATE": ("trace_sample_rate", float),
            f"{prefix}NODE_ID": ("node_id", str),
            f"{prefix}CLUSTER_SIZE_HINT": ("cluster_size_hint", int),
        }

        for env_var, (field_name, converter) in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                try:
                    overrides[field_name] = converter(value)
                except (ValueError, TypeError):
                    pass  # Ignore invalid values

        return cls(**overrides)

    def validate(self) -> None:
        """
        Validate configuration values.

        Raises:
            ValueError: If any value is out of valid range
        """
        if not (0 < self.bloom_false_positive_rate < 1):
            raise ValueError("bloom_false_positive_rate must be between 0 and 1")

        if not (0 <= self.trace_sample_rate <= 1):
            raise ValueError("trace_sample_rate must be between 0 and 1")

        if self.event_id_hash_length < 8:
            raise ValueError("event_id_hash_length must be at least 8")

        if self.cache_size < 0:
            raise ValueError("cache_size must be non-negative")

        if self.compaction_semantic_threshold < 0 or self.compaction_semantic_threshold > 1:
            raise ValueError("compaction_semantic_threshold must be between 0 and 1")


# =============================================================================
# TRACING CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class TraceConfig:
    """
    Configuration specific to tracing and debugging.

    Separated from main config for cleaner organization
    and independent tuning.
    """

    # What to trace
    trace_event_creation: bool = True
    trace_event_retrieval: bool = True
    trace_materialization: bool = True
    trace_index_operations: bool = False  # High volume
    trace_health_checks: bool = True
    trace_compaction: bool = True

    # How to trace
    output_format: str = "json"  # "json" or "text"
    include_timestamps: bool = True
    include_stack_traces: bool = False
    max_content_preview: int = 200  # Characters

    # Where to trace
    trace_to_stderr: bool = True
    trace_to_file: Optional[str] = None
    trace_to_events: bool = True  # Meta-cognition events

    # Filtering
    min_duration_ms: float = 0.0  # Only trace slow operations
    concept_filter: FrozenSet[str] = field(default_factory=frozenset)


# =============================================================================
# BENCHMARK CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class BenchmarkConfig:
    """
    Configuration for CEL benchmarks.

    Used by the benchmark suite to ensure consistent measurement.
    """

    # Warmup
    warmup_iterations: int = 3
    warmup_events: int = 100

    # Measurement
    measurement_iterations: int = 10
    measurement_timeout_seconds: int = 60

    # Scaling
    scale_factors: tuple = (1, 10, 100, 1000)  # Event counts to test

    # Thresholds (for pass/fail)
    max_event_append_ms: float = 1.0  # Per event
    max_materialize_ms: float = 10.0  # Per entity
    max_search_ms: float = 50.0  # Per query
    min_throughput_events_per_sec: float = 1000.0

    # Baseline comparison
    baseline_file: Optional[str] = None
    regression_threshold_percent: float = 10.0  # Fail if >10% slower


# =============================================================================
# CONSTANTS (Non-Configurable)
# =============================================================================

class CELConstants:
    """
    Constants that should NOT be configurable.

    These are fundamental to the architecture and changing them
    would break compatibility.
    """

    # Hash algorithm
    HASH_ALGORITHM = "sha256"

    # Event type prefixes (for ID namespacing)
    EVENT_PREFIX = "E-"
    ENTITY_PREFIX = "N-"  # Node
    HORIZON_PREFIX = "H-"
    COMPACTION_PREFIX = "C-"

    # Special event IDs
    GENESIS_EVENT_ID = "GENESIS"

    # Timestamp format
    TIMESTAMP_FORMAT = "ISO8601"  # Always ISO 8601 with timezone

    # Version for storage format
    STORAGE_VERSION = 1

    # Maximum sizes (hard limits)
    MAX_CAUSAL_PARENTS = 100
    MAX_CONCEPTS_PER_EVENT = 50
    MAX_CONTENT_DEPTH = 10  # Nesting depth in content dict


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_config(
    profile: str = "default",
    **overrides: Any,
) -> CELConfig:
    """
    Create a configuration with optional profile presets.

    Profiles:
        "default" - Balanced settings for typical use
        "development" - More tracing, smaller limits
        "production" - Optimized for performance
        "testing" - Fast iterations, minimal overhead
        "distributed" - Settings for multi-node deployment

    Args:
        profile: Preset profile name
        **overrides: Additional overrides

    Returns:
        Configured CELConfig instance
    """
    profiles = {
        "default": {},
        "development": {
            "enable_tracing": True,
            "trace_sample_rate": 1.0,
            "max_events_before_compaction": 1000,
            "health_check_interval_seconds": 60,
        },
        "production": {
            "enable_tracing": False,
            "cache_size": 5000,
            "bloom_filter_size": 100_000,
            "max_events_before_compaction": 50_000,
        },
        "testing": {
            "max_events_before_compaction": 100,
            "cache_size": 100,
            "bloom_filter_size": 1000,
            "health_check_interval_seconds": 10,
            "enable_tracing": True,
        },
        "distributed": {
            "cluster_size_hint": 3,
            "cache_size": 2000,
            "enable_tracing": True,
            "trace_sample_rate": 0.1,  # Sample 10%
        },
    }

    if profile not in profiles:
        raise ValueError(f"Unknown profile: {profile}. Options: {list(profiles.keys())}")

    # Start with profile defaults
    config_dict = profiles[profile].copy()

    # Apply overrides
    config_dict.update(overrides)

    # Create and validate
    config = CELConfig(**config_dict)
    config.validate()

    return config
