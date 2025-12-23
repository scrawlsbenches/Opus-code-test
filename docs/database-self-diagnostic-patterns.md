# Self-Diagnostic and Self-Healing Patterns for Database Systems

## Overview

This document synthesizes patterns for database systems that maintain themselves with minimal human intervention. The patterns are based on research into existing implementations and best practices for:

- **Health checks**: What to monitor, frequency, acceptable bounds
- **Diagnostic logs**: Smart logging without log explosion
- **Anomaly detection**: Early problem detection before critical failure
- **Self-repair**: Automatic recovery from common issues
- **Investigation triggers**: Decision logic for alerts vs auto-fix vs log-and-continue
- **Constraint definitions**: Expressing "acceptable" system bounds

These patterns enable systems to self-diagnose problems, determine severity, take corrective action, and escalate intelligently when human intervention is needed.

---

## 1. Health Checks: Monitoring Framework

### 1.1 The Four Levels of Health Status

Health checks form a hierarchy from low-level operational metrics to high-level system state:

```
┌──────────────────────────────────────────────────────────────┐
│  LEVEL 4: SYSTEM HEALTH (5-minute intervals)                │
│  - Overall system operational? (binary)                       │
│  - Cascading failures? (how many components down)            │
│  - Recovery in progress?                                     │
├──────────────────────────────────────────────────────────────┤
│  LEVEL 3: SUBSYSTEM HEALTH (1-minute intervals)             │
│  - Component operational? (persistence, search, clustering)  │
│  - Component staleness (data freshness)                      │
│  - Recovery completeness                                     │
├──────────────────────────────────────────────────────────────┤
│  LEVEL 2: OPERATIONAL METRICS (10-second intervals)         │
│  - Core operations functioning? (reads, writes, computes)    │
│  - Error rates by operation type                             │
│  - Latency percentiles (p50, p99)                            │
│  - Constraint violations                                     │
├──────────────────────────────────────────────────────────────┤
│  LEVEL 1: FINE-GRAINED TELEMETRY (per-operation)           │
│  - Individual operation timing                               │
│  - Detailed error context                                    │
│  - State transitions                                         │
│  - Anomalies detected                                        │
└──────────────────────────────────────────────────────────────┘
```

### 1.2 Health Check Categories

#### Category A: Data Integrity Checks

**What to monitor:**
- Checksum verification on critical data structures
- Orphaned references (edges without source/target nodes)
- Index-data consistency (index references valid items)
- Constraint violation counts

**How often:**
- **On-demand**: After every transaction that modifies state (checkpoint)
- **Periodic**: Every 5 minutes for background verification
- **On-repair**: After any self-healing operation

**Pattern implementation:**
```python
class IntegrityChecker:
    """Verify data consistency without expensive full scans."""

    def __init__(self):
        self.last_full_check = None
        self.last_sample_check = None

    def quick_check(self) -> bool:
        """Sample-based check (1% of data, <100ms)."""
        # Check random samples instead of entire dataset
        sample_size = max(100, self.total_items // 100)
        for item in random.sample(self.items, sample_size):
            if not self._verify_item(item):
                return False
        return True

    def full_check(self) -> IntegrityReport:
        """Complete check (expensive, run occasionally)."""
        report = IntegrityReport()
        # Check every item, every reference
        # Takes O(n) but ensures 100% coverage
        return report

    def incremental_check(self) -> bool:
        """Check only changed items since last check."""
        # Keep set of items modified since last check
        # Only verify those
        pass
```

#### Category B: Performance Metrics

**What to monitor:**
- Operation latency (p50, p95, p99)
- Throughput (operations per second)
- Queue depths (if operations queue)
- Memory usage trends
- Cache hit rates

**How often:**
- Per-operation: Every operation records timing
- Aggregated: Every 10-30 seconds
- Alerts: On anomalous latency spike (>2x baseline)

**Pattern implementation:**
```python
@dataclass
class LatencyBucket:
    """Track latency distribution."""
    operation: str
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    count: int

    def is_anomalous(self) -> bool:
        """Compare against baseline."""
        # If p99 > 2x typical baseline, it's concerning
        baseline_p99 = BASELINE_LATENCIES.get(self.operation, 1000)
        return self.p99_ms > 2 * baseline_p99

class LatencyMonitor:
    """Track operation latencies efficiently."""

    def __init__(self):
        self.baselines: Dict[str, float] = {}
        self.current_bucket = {}
        self.bucket_duration = 30  # seconds

    def record_operation(self, op_name: str, duration_ms: float):
        """Record individual operation."""
        if op_name not in self.current_bucket:
            self.current_bucket[op_name] = deque(maxlen=10000)
        self.current_bucket[op_name].append(duration_ms)

    def get_metrics(self) -> Dict[str, LatencyBucket]:
        """Get current latency metrics."""
        metrics = {}
        for op_name, timings in self.current_bucket.items():
            metrics[op_name] = LatencyBucket(
                operation=op_name,
                p50_ms=percentile(timings, 50),
                p95_ms=percentile(timings, 95),
                p99_ms=percentile(timings, 99),
                min_ms=min(timings),
                max_ms=max(timings),
                count=len(timings)
            )
        return metrics
```

#### Category C: Resource Constraints

**What to monitor:**
- Disk space available for WAL/snapshots
- Memory used for caches/indexes
- File descriptor count
- Concurrent operation count
- Transaction size (bytes/rows modified)

**How often:**
- Every 30 seconds for major resources
- Before operation if approaching limit
- Alert threshold: 80% utilization

**Pattern implementation:**
```python
@dataclass
class ResourceConstraint:
    """Define a monitored resource constraint."""
    name: str
    current: float  # Current usage
    limit: float    # Maximum allowed
    warning_threshold: float = 0.8  # Alert at 80%
    critical_threshold: float = 0.95  # Emergency at 95%

    @property
    def utilization(self) -> float:
        """Percentage of limit in use."""
        return self.current / self.limit if self.limit > 0 else 0.0

    def status(self) -> str:
        """Current status."""
        util = self.utilization
        if util > self.critical_threshold:
            return "CRITICAL"
        elif util > self.warning_threshold:
            return "WARNING"
        else:
            return "OK"

class ConstraintMonitor:
    """Monitor system resource constraints."""

    def __init__(self):
        self.constraints: Dict[str, ResourceConstraint] = {}

    def register_constraint(self, constraint: ResourceConstraint):
        """Register a constraint to monitor."""
        self.constraints[constraint.name] = constraint

    def check_before_operation(self, op_type: str) -> bool:
        """Check if operation can proceed."""
        # Different operations have different constraint requirements
        if op_type == "write":
            # Need disk space and memory
            return (self._check_space() and
                    self._check_memory() and
                    self._check_concurrency())
        elif op_type == "snapshot":
            # Need significant disk space
            return self._check_space(multiplier=2.0)
        return True

    def get_alert_status(self) -> List[str]:
        """Get list of constraint violations."""
        alerts = []
        for constraint in self.constraints.values():
            status = constraint.status()
            if status != "OK":
                alerts.append(
                    f"{constraint.name}: {status} "
                    f"({constraint.utilization:.1%})"
                )
        return alerts
```

### 1.3 Health Check Scheduling

```python
class HealthCheckScheduler:
    """Coordinate different health checks at appropriate frequencies."""

    def __init__(self):
        self.checks = {
            'quick': {
                'fn': self.quick_integrity_check,
                'interval_sec': 60,
                'timeout_sec': 5,
                'last_run': None
            },
            'metrics': {
                'fn': self.gather_metrics,
                'interval_sec': 30,
                'timeout_sec': 2,
                'last_run': None
            },
            'resource': {
                'fn': self.check_resources,
                'interval_sec': 30,
                'timeout_sec': 1,
                'last_run': None
            },
            'full': {
                'fn': self.full_integrity_check,
                'interval_sec': 3600,  # 1 hour
                'timeout_sec': 30,
                'last_run': None
            }
        }

    def should_run(self, check_name: str) -> bool:
        """Check if a health check should run now."""
        check = self.checks[check_name]
        if check['last_run'] is None:
            return True
        elapsed = time.time() - check['last_run']
        return elapsed >= check['interval_sec']

    def run_due_checks(self) -> HealthReport:
        """Run all checks that are due."""
        report = HealthReport()
        for check_name, check_config in self.checks.items():
            if self.should_run(check_name):
                try:
                    result = timeout_call(
                        check_config['fn'],
                        timeout=check_config['timeout_sec']
                    )
                    report.add_check(check_name, result)
                    check_config['last_run'] = time.time()
                except TimeoutError:
                    report.add_timeout(check_name)
        return report
```

---

## 2. Diagnostic Logs: Smart Logging Without Explosion

The key challenge: Capture enough detail to debug problems without creating log volume that masks important signals.

### 2.1 The Diagnostic Log Strategy

Instead of logging everything, use **strategic sampling** and **context-aware verbosity**:

```
┌──────────────────────────────────────────────────────────────┐
│  NORMAL OPERATION (99% of time)                              │
│  - Sampled: 1 in 1000 operations                             │
│  - Info level: Summaries only                                │
│  - Exception: Any error/anomaly                              │
├──────────────────────────────────────────────────────────────┤
│  ELEVATED OPERATION (anomaly detected)                       │
│  - Sampled: 1 in 100 operations                              │
│  - Info level: Detailed context                              │
│  - Include: Operation parameters, intermediate state         │
├──────────────────────────────────────────────────────────────┤
│  ACTIVE INCIDENT                                             │
│  - Sampled: All operations (1 in 1)                          │
│  - Debug level: Full traces                                  │
│  - Include: Stack traces, state dumps                        │
│  - Duration: Until issue resolved + 10 minutes               │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 Adaptive Log Verbosity

```python
class AdaptiveLogger:
    """Adjust log verbosity based on system state."""

    def __init__(self):
        self.base_sample_rate = 0.001  # 1 in 1000
        self.anomaly_detected = False
        self.anomaly_until = None
        self.incident_mode = False

    def get_sample_rate(self) -> float:
        """Get current sampling rate."""
        now = time.time()

        # Emergency mode: log everything
        if self.incident_mode:
            return 1.0

        # Anomaly mode: higher sampling
        if self.anomaly_detected:
            if self.anomaly_until and now > self.anomaly_until:
                self.anomaly_detected = False
            else:
                return 0.01  # 1 in 100

        return self.base_sample_rate

    def should_log(self, operation: str, context: dict) -> bool:
        """Decide if operation should be logged."""
        rate = self.get_sample_rate()

        # Always log errors
        if context.get('error'):
            return True

        # Always log anomalies
        if context.get('anomalous'):
            return True

        # Sample based on rate
        return random.random() < rate

    def log_operation(self, operation: str, **context):
        """Conditional operation logging."""
        if not self.should_log(operation, context):
            return

        # Determine verbosity
        if self.incident_mode:
            self._log_full_trace(operation, context)
        elif self.anomaly_detected:
            self._log_detailed(operation, context)
        else:
            self._log_summary(operation, context)

    def detect_anomaly(self, anomaly_type: str):
        """Escalate logging for detected anomaly."""
        self.anomaly_detected = True
        self.anomaly_until = time.time() + 600  # 10 minutes
        logger.warning(f"Anomaly detected: {anomaly_type}. "
                      f"Increasing log verbosity for next 10 minutes.")

    def enter_incident_mode(self, reason: str):
        """Enter full logging mode for incident."""
        self.incident_mode = True
        logger.critical(f"INCIDENT MODE: {reason}. Logging all operations.")

    def exit_incident_mode(self):
        """Return to normal logging."""
        self.incident_mode = False
        logger.info("Incident resolved. Returning to normal logging.")
```

### 2.3 Contextual Diagnostic Bundles

Instead of individual log lines, create **diagnostic bundles** when an event occurs:

```python
@dataclass
class DiagnosticBundle:
    """Complete diagnostic information for an event."""
    timestamp: datetime
    event_type: str  # 'error', 'anomaly', 'recovery', 'constraint_violation'
    summary: str  # One-line summary
    context: Dict[str, Any]  # Operation-specific context
    metrics_snapshot: Dict[str, float]  # Current metrics
    state_snapshot: Dict[str, Any]  # Relevant system state
    stack_trace: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)

    def to_log(self) -> str:
        """Format for logging."""
        lines = [
            f"[{self.event_type.upper()}] {self.summary}",
            f"Context: {json.dumps(self.context, default=str)}",
            f"Metrics: {json.dumps(self.metrics_snapshot)}",
        ]
        if self.stack_trace:
            lines.append(f"Stack:\n{self.stack_trace}")
        if self.suggestions:
            lines.append(f"Suggestions: {'; '.join(self.suggestions)}")
        return "\n".join(lines)

class DiagnosticsCollector:
    """Gather diagnostic information efficiently."""

    def create_bundle(self, event_type: str, summary: str,
                     context: dict) -> DiagnosticBundle:
        """Create diagnostic bundle without expensive operations."""
        # Quick snapshot of current metrics
        metrics = {
            'ops_per_sec': self._get_ops_per_sec(),
            'p95_latency_ms': self._get_p95_latency(),
            'error_rate': self._get_error_rate(),
            'memory_mb': psutil.Process().memory_info().rss / 1024 / 1024,
        }

        # Only include state that's relevant to this event
        state = {}
        if event_type == 'constraint_violation':
            state['constraints'] = self._get_violated_constraints()
        elif event_type == 'recovery':
            state['recovery_progress'] = self._get_recovery_status()

        bundle = DiagnosticBundle(
            timestamp=datetime.now(),
            event_type=event_type,
            summary=summary,
            context=context,
            metrics_snapshot=metrics,
            state_snapshot=state
        )

        # Auto-suggest fixes based on event type
        if event_type == 'constraint_violation':
            bundle.suggestions = self._suggest_fixes_for_constraint(context)

        return bundle
```

### 2.4 Log Retention and Compaction

```python
class DiagnosticLogManager:
    """Manage diagnostic logs with automatic compaction."""

    def __init__(self, log_dir: Path, retention_days: int = 7):
        self.log_dir = log_dir
        self.retention_days = retention_days
        self.current_log = None

    def append_bundle(self, bundle: DiagnosticBundle):
        """Append diagnostic bundle to current log."""
        if not self.current_log:
            self.current_log = self._open_daily_log()

        self.current_log.write(bundle.to_log() + "\n")
        self.current_log.flush()

    def compact_old_logs(self):
        """Compress and archive old logs."""
        now = datetime.now()
        cutoff = now - timedelta(days=self.retention_days)

        for log_file in self.log_dir.glob("diagnostic-*.log"):
            # Extract date from filename
            file_date = self._parse_log_date(log_file)
            if file_date < cutoff:
                # Compress old logs
                with open(log_file, 'rb') as f_in:
                    with gzip.open(f"{log_file}.gz", 'wb') as f_out:
                        f_out.writelines(f_in)
                log_file.unlink()
```

---

## 3. Anomaly Detection: Early Problem Detection

### 3.1 Multi-Method Anomaly Detection

Combine statistical, pattern-based, and semantic anomaly detection:

```python
@dataclass
class AnomalyResult:
    """Result of anomaly detection."""
    is_anomalous: bool
    confidence: float  # 0.0 = definitely normal, 1.0 = definitely anomalous
    reasons: List[str]  # Why we think it's anomalous
    methods: Dict[str, float]  # Score from each detection method
    severity: str  # 'low', 'medium', 'high'

class MultiMethodAnomalyDetector:
    """Detect anomalies using multiple strategies."""

    def __init__(self):
        self.baseline_metrics = {}
        self.recent_history = defaultdict(deque)
        self.patterns = PatternMatcher()

    def check_operation(self, op_name: str,
                       duration_ms: float,
                       result: dict) -> AnomalyResult:
        """Check if operation is anomalous."""
        reasons = []
        methods = {}

        # Method 1: Statistical deviation
        stat_score = self._check_statistical_deviation(op_name, duration_ms)
        methods['statistical'] = stat_score
        if stat_score > 0.7:
            reasons.append(f"Latency {stat_score:.1%} above baseline")

        # Method 2: Pattern matching
        pattern_score = self._check_patterns(op_name, result)
        methods['pattern'] = pattern_score
        if pattern_score > 0.7:
            reasons.append(f"Matches anomalous pattern (score {pattern_score:.2f})")

        # Method 3: Behavioral change
        behavior_score = self._check_behavior_change(op_name, result)
        methods['behavior'] = behavior_score
        if behavior_score > 0.7:
            reasons.append("Behavior differs from recent history")

        # Method 4: Constraint violations
        constraint_score = self._check_constraints(op_name, result)
        methods['constraints'] = constraint_score
        if constraint_score > 0.7:
            reasons.append("Violates system constraints")

        # Aggregate scores
        avg_score = sum(methods.values()) / len(methods)
        is_anomalous = avg_score > 0.5 or len(reasons) >= 2

        return AnomalyResult(
            is_anomalous=is_anomalous,
            confidence=avg_score,
            reasons=reasons,
            methods=methods,
            severity=self._classify_severity(avg_score, reasons)
        )

    def _check_statistical_deviation(self, op_name: str,
                                     duration_ms: float) -> float:
        """Check if latency is statistically anomalous."""
        baseline = self.baseline_metrics.get(op_name)
        if not baseline:
            return 0.0

        # Standard deviation based detection
        z_score = abs((duration_ms - baseline['mean']) / baseline['std'])
        # Scores: z > 3 is very unusual (0.99 confidence)
        return min(1.0, max(0.0, (z_score - 2) / 2))  # 0 at z=2, 1 at z=4

    def _check_patterns(self, op_name: str, result: dict) -> float:
        """Check for known anomalous patterns."""
        # Examples of patterns:
        # - Error types that correlate with problems
        # - Result structures that indicate issues
        # - Operation chains that precede failures
        return self.patterns.match(op_name, result)

    def _check_behavior_change(self, op_name: str, result: dict) -> float:
        """Detect significant changes from recent behavior."""
        history = self.recent_history[op_name]
        if len(history) < 10:
            return 0.0  # Not enough history

        # Compare recent distribution to current result
        recent_results = list(history)[-100:]  # Last 100 operations

        # Check if current result differs significantly
        # This detects things like:
        # - Suddenly returning different result structure
        # - Significantly different state after operation
        return self._kl_divergence(result, recent_results)
```

### 3.2 Adaptive Baselines

```python
class AdaptiveBaseline:
    """Learn and adapt anomaly detection baselines."""

    def __init__(self, operation: str):
        self.operation = operation
        self.measurements = deque(maxlen=10000)
        self.baseline = None
        self.last_update = None

    def record(self, duration_ms: float):
        """Record an operation measurement."""
        self.measurements.append(duration_ms)

        # Update baseline periodically
        if not self.last_update or \
           time.time() - self.last_update > 300:  # 5 minutes
            self._update_baseline()

    def _update_baseline(self):
        """Recalculate baseline from recent measurements."""
        measurements = list(self.measurements)
        if not measurements:
            return

        # Use robust statistics (resistant to outliers)
        self.baseline = {
            'mean': statistics.mean(measurements),
            'median': statistics.median(measurements),
            'stdev': statistics.stdev(measurements) if len(measurements) > 1 else 0,
            'p95': percentile(measurements, 95),
            'p99': percentile(measurements, 99),
            'min': min(measurements),
            'max': max(measurements),
        }
        self.last_update = time.time()

    def is_anomalous(self, duration_ms: float) -> bool:
        """Check if measurement is anomalous."""
        if not self.baseline:
            return False

        # Use median absolute deviation (robust to outliers)
        median = self.baseline['median']
        median_abs_dev = statistics.median(
            [abs(x - median) for x in self.measurements]
        )

        # Anomalous if more than 3 MAD from median
        return abs(duration_ms - median) > 3 * median_abs_dev
```

### 3.3 Cascading Anomaly Alerts

```python
class AnomalyAlertManager:
    """Manage anomaly alerts with smart escalation."""

    def __init__(self):
        self.anomalies: Dict[str, List[AnomalyResult]] = defaultdict(list)
        self.incident_threshold = 5  # Alert if 5+ anomalies in window
        self.window_sec = 60

    def report_anomaly(self, result: AnomalyResult):
        """Report a detected anomaly."""
        event_type = self._classify_event(result)
        self.anomalies[event_type].append(result)

        # Clean old anomalies
        cutoff = time.time() - self.window_sec
        for event_type in self.anomalies:
            self.anomalies[event_type] = [
                a for a in self.anomalies[event_type]
                if a.timestamp > cutoff
            ]

        # Check for escalation
        self._check_escalation()

    def _check_escalation(self):
        """Check if we should escalate anomalies to incident."""
        total_anomalies = sum(len(v) for v in self.anomalies.values())

        if total_anomalies >= self.incident_threshold:
            logger.warning(
                f"Anomaly surge detected: {total_anomalies} anomalies "
                f"in {self.window_sec}s. Entering elevated diagnostics."
            )
            self._enter_elevated_diagnostics()

        # Check for specific patterns
        error_anomalies = len(self.anomalies['error_rate_spike'])
        latency_anomalies = len(self.anomalies['latency_spike'])

        if error_anomalies > 10:
            logger.critical("High error rate detected. Entering incident mode.")
            self._enter_incident_mode()
```

---

## 4. Self-Repair: Automatic Recovery

### 4.1 Recovery Cascade Framework

Design recovery as a **cascade of increasingly thorough methods**, trying faster methods first:

```
Recovery Cascade Philosophy:
- Level 1 (fastest): Try to recover in-place
- Level 2 (faster): Restore from recent snapshot
- Level 3 (moderate): Replay from transaction log
- Level 4 (slowest): Rebuild from scratch

Key principle: Always try to recover most recent state.
Each level loses a little data but guarantees recovery.
```

### 4.2 Self-Repair Decision Framework

```python
@dataclass
class RepairDecision:
    """Decision to repair or escalate."""
    should_repair: bool
    repair_type: str  # 'in_place', 'snapshot', 'replay', 'rebuild', 'escalate'
    confidence: float  # How confident we are repair will work
    estimated_duration_sec: float
    estimated_data_loss_percent: float
    reasons: List[str]

class SelfRepairManager:
    """Decide when and how to self-repair."""

    def __init__(self):
        self.repair_history = []
        self.recent_failures = defaultdict(int)

    def decide_repair(self,
                     problem_type: str,
                     severity: str,
                     error_context: dict) -> RepairDecision:
        """Decide whether to self-repair or escalate."""

        # Check if this has been failing repeatedly
        self.recent_failures[problem_type] += 1
        if self.recent_failures[problem_type] > 3:
            return RepairDecision(
                should_repair=False,
                repair_type='escalate',
                confidence=0.0,
                estimated_duration_sec=0,
                estimated_data_loss_percent=0,
                reasons=["Problem has failed 3+ repair attempts"]
            )

        # Determine repair strategy by problem type
        if problem_type == 'stale_lock':
            return self._decide_lock_repair(error_context)
        elif problem_type == 'corrupted_wal':
            return self._decide_wal_repair(error_context)
        elif problem_type == 'missing_file':
            return self._decide_file_repair(error_context)
        elif problem_type == 'constraint_violation':
            return self._decide_constraint_repair(error_context)
        else:
            return RepairDecision(
                should_repair=False,
                repair_type='escalate',
                confidence=0.0,
                estimated_duration_sec=0,
                estimated_data_loss_percent=0,
                reasons=[f"Unknown problem type: {problem_type}"]
            )

    def _decide_lock_repair(self, context: dict) -> RepairDecision:
        """Decide if stale lock can be auto-repaired."""
        # Stale locks are safe to force-unlock if:
        # 1. Lock holder hasn't updated it in >5 minutes
        # 2. We have a recent snapshot (can rollback if needed)
        # 3. No concurrent operations might be affected

        lock_age_sec = context.get('lock_age_sec', 0)
        has_snapshot = context.get('has_recent_snapshot', False)
        concurrent_count = context.get('concurrent_operations', 0)

        if (lock_age_sec > 300 and
            has_snapshot and
            concurrent_count == 0):
            return RepairDecision(
                should_repair=True,
                repair_type='in_place',
                confidence=0.95,
                estimated_duration_sec=0.5,
                estimated_data_loss_percent=0,
                reasons=[
                    f"Lock is {lock_age_sec}s old (> 5min threshold)",
                    "Recent snapshot available for rollback",
                    "No concurrent operations"
                ]
            )
        else:
            return RepairDecision(
                should_repair=False,
                repair_type='escalate',
                confidence=0.0,
                estimated_duration_sec=0,
                estimated_data_loss_percent=0,
                reasons=[
                    f"Lock too fresh ({lock_age_sec}s)" if lock_age_sec < 300 else "",
                    "No snapshot available" if not has_snapshot else "",
                    f"Concurrent ops ({concurrent_count})" if concurrent_count > 0 else ""
                ]
            )

    def _decide_wal_repair(self, context: dict) -> RepairDecision:
        """Decide if WAL corruption can be auto-repaired."""
        corruption_point = context.get('corruption_at_entry', 0)
        total_entries = context.get('total_entries', 0)

        # Safe to skip corrupted entries if it's near the end
        # (recent changes we can re-apply)
        if corruption_point > total_entries * 0.95:
            return RepairDecision(
                should_repair=True,
                repair_type='replay',
                confidence=0.8,
                estimated_duration_sec=5,
                estimated_data_loss_percent=0.5,
                reasons=[
                    f"Corruption near end ({corruption_point}/{total_entries})",
                    "Recent changes can be re-applied after restart"
                ]
            )
        else:
            return RepairDecision(
                should_repair=False,
                repair_type='escalate',
                confidence=0.0,
                estimated_duration_sec=0,
                estimated_data_loss_percent=0,
                reasons=[
                    f"Corruption too early ({corruption_point}/{total_entries})",
                    "Risk of significant data loss"
                ]
            )

    def execute_repair(self, decision: RepairDecision) -> RepairResult:
        """Execute the decided repair."""
        start_time = time.time()

        try:
            if decision.repair_type == 'in_place':
                result = self._repair_in_place()
            elif decision.repair_type == 'snapshot':
                result = self._restore_snapshot()
            elif decision.repair_type == 'replay':
                result = self._replay_wal()
            elif decision.repair_type == 'rebuild':
                result = self._rebuild_from_scratch()
            else:
                result = RepairResult(success=False, reason="Unknown repair type")

            # Record repair in history
            duration = time.time() - start_time
            self.repair_history.append({
                'type': decision.repair_type,
                'success': result.success,
                'duration_sec': duration,
                'timestamp': datetime.now()
            })

            # Reset failure counter on success
            if result.success:
                self.recent_failures.clear()

            return result

        except Exception as e:
            logger.error(f"Repair failed: {e}")
            return RepairResult(success=False, reason=str(e))
```

### 4.3 Safe Repair Operations

Key pattern: **Always have a rollback plan**

```python
class SafeRepairOperation:
    """Encapsulate repair with automatic rollback."""

    def __init__(self, name: str, backup_dir: Path):
        self.name = name
        self.backup_dir = backup_dir
        self.backup_created = False

    def __enter__(self):
        """Create backup before starting repair."""
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        # Copy current state to backup
        shutil.copytree(
            self.data_dir,
            self.backup_dir / "pre_repair",
            dirs_exist_ok=True
        )
        self.backup_created = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Rollback if repair failed."""
        if exc_type is not None and self.backup_created:
            logger.error(f"Repair failed, rolling back...")
            shutil.rmtree(self.data_dir)
            shutil.copytree(
                self.backup_dir / "pre_repair",
                self.data_dir
            )

    def execute(self):
        """Execute the repair operation."""
        # Repair logic here
        # If exception is raised, __exit__ will rollback
        pass

# Usage:
with SafeRepairOperation("rebuild_index", backup_dir) as repair:
    repair.rebuild_index()
    # If exception occurs, automatically rolls back
```

---

## 5. Investigation Triggers: Alert vs Auto-Fix Decision Logic

### 5.1 Failure Classification Hierarchy

```
Classify every failure into a level that determines response:

LEVEL 1: SELF-RECOVERABLE (HICCUP)
- Definition: Can be fixed automatically with high confidence
- Examples: Stale lock, transient network timeout
- Response: Fix automatically, log it, monitor for patterns
- Threshold: Need 3+ successful repairs before considering "solved"

LEVEL 2: NEEDS ADAPTATION (OBSTACLE)
- Definition: Solvable but requires strategy adjustment
- Examples: Constraint temporarily violated, queue backing up
- Response: Trigger mitigation, increase monitoring, prepare escalation
- Threshold: If manual mitigation needed 2+ times, escalate

LEVEL 3: NEEDS HUMAN (WALL)
- Definition: Fundamentally unsolvable without external input
- Examples: Configuration error, incompatible dependency version
- Response: Generate detailed report, escalate to human
- Threshold: Alert human immediately

LEVEL 4: STOP IMMEDIATELY (CRISIS)
- Definition: Continuing causes damage
- Examples: Data corruption spreading, security breach
- Response: Halt operations, preserve state, alert
- Threshold: Stop first, understand later
```

### 5.2 Progressive Escalation Framework

```python
class ProgressiveEscalation:
    """Escalate problems progressively, not immediately."""

    def __init__(self):
        self.current_level = 0  # 0=normal, 1=watch, 2=warn, 3=escalate
        self.level_entered_at = time.time()
        self.evidence: List[str] = []

    def report_issue(self, issue_type: str, severity: float):
        """Report a potential issue."""
        new_level = self._classify_level(issue_type, severity)

        if new_level > self.current_level:
            self._escalate_level(new_level)
        elif new_level < self.current_level:
            self._check_deescalate()

        self.evidence.append({
            'issue': issue_type,
            'severity': severity,
            'timestamp': datetime.now()
        })

    def _classify_level(self, issue_type: str, severity: float) -> int:
        """Classify issue into escalation level."""
        if issue_type in ('stale_lock', 'transient_error') and severity < 0.5:
            return 0  # Normal
        elif issue_type in ('constraint_violated', 'slow_query') and severity < 0.7:
            return 1  # Watch
        elif issue_type in ('corruption_detected', 'repeated_failure'):
            return 2 if severity < 0.8 else 3
        elif issue_type in ('data_loss', 'security_issue'):
            return 3  # Escalate
        else:
            return 0

    def _escalate_level(self, new_level: int):
        """Escalate to higher level."""
        old_level = self.current_level
        self.current_level = new_level
        self.level_entered_at = time.time()

        if new_level == 1:
            logger.info("Escalating to WATCH level. Increasing monitoring.")
        elif new_level == 2:
            logger.warning("Escalating to WARN level. Generating report for human.")
            self._generate_escalation_report()
        elif new_level == 3:
            logger.critical("Escalating to ESCALATE level. Human intervention required.")
            self._escalate_to_human()

    def _check_deescalate(self):
        """Check if we can return to lower escalation level."""
        elapsed = time.time() - self.level_entered_at

        # Deescalate if no issues for threshold period
        if self.current_level == 1 and elapsed > 300:  # 5 min
            self.current_level = 0
            logger.info("Returning to NORMAL level. Issue appears resolved.")
        elif self.current_level == 2 and elapsed > 600:  # 10 min
            self.current_level = 1
            logger.info("Returning to WATCH level. No new evidence.")

    def _generate_escalation_report(self):
        """Generate report for human review."""
        report = {
            'timestamp': datetime.now(),
            'escalation_level': self.current_level,
            'evidence_count': len(self.evidence),
            'issue_types': list(set(e['issue'] for e in self.evidence)),
            'timeline': self.evidence[-10:],  # Last 10 events
            'suggestions': self._generate_suggestions()
        }
        return report

    def _generate_suggestions(self) -> List[str]:
        """Auto-generate suggestions for human."""
        suggestions = []
        issue_types = set(e['issue'] for e in self.evidence)

        if 'repeated_failure' in issue_types:
            suggestions.append(
                "Problem has repeated multiple times. "
                "May indicate deeper issue. Consider full system check."
            )
        if 'constraint_violated' in issue_types:
            suggestions.append(
                "Constraints are being violated. May need to adjust limits "
                "or investigate root cause."
            )
        if 'slow_query' in issue_types:
            suggestions.append(
                "Performance degradation. Consider profiling or adding indexes."
            )

        return suggestions
```

### 5.3 Smart Alert Rules

```python
@dataclass
class AlertRule:
    """Define when to alert based on observable conditions."""
    name: str
    condition: Callable[[dict], bool]  # Returns True if should alert
    severity: str  # 'info', 'warning', 'critical'
    escalate_after_count: int = 1  # Alert after N violations
    window_sec: int = 300  # Check within this time window

    def __call__(self, metrics: dict) -> bool:
        """Check if alert should fire."""
        return self.condition(metrics)

class SmartAlertManager:
    """Manage alerts with smart de-duplication and escalation."""

    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.violations: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=10)
        )
        self.alerts_sent: Dict[str, datetime] = {}

    def register_rule(self, rule: AlertRule):
        """Register an alert rule."""
        self.rules[rule.name] = rule

    def check_metrics(self, metrics: dict) -> List[str]:
        """Check all rules against metrics."""
        alerts = []

        for rule_name, rule in self.rules.items():
            if rule(metrics):
                self.violations[rule_name].append(time.time())

                # Check if we should alert
                violation_count = len(self.violations[rule_name])
                if violation_count >= rule.escalate_after_count:
                    # Check if we've already alerted recently
                    last_alert = self.alerts_sent.get(rule_name)
                    if not last_alert or \
                       time.time() - last_alert > 300:  # 5 min cooldown
                        alerts.append(rule_name)
                        self.alerts_sent[rule_name] = datetime.now()

        return alerts

# Example rules:
alert_mgr = SmartAlertManager()

# Alert on high error rate
alert_mgr.register_rule(AlertRule(
    name='high_error_rate',
    condition=lambda m: m.get('error_rate', 0) > 0.05,  # >5%
    severity='warning',
    escalate_after_count=3,  # Alert after 3 consecutive checks
    window_sec=60
))

# Alert on resource exhaustion
alert_mgr.register_rule(AlertRule(
    name='disk_almost_full',
    condition=lambda m: m.get('disk_free_percent', 100) < 10,
    severity='critical',
    escalate_after_count=1,  # Alert immediately
    window_sec=30
))

# Alert on data anomaly
alert_mgr.register_rule(AlertRule(
    name='data_corruption_detected',
    condition=lambda m: m.get('corruption_count', 0) > 0,
    severity='critical',
    escalate_after_count=1,
    window_sec=5
))
```

---

## 6. Constraint Definitions: Expressing Acceptable Bounds

### 6.1 Constraint Specification Pattern

Constraints should be **explicit, validated, and monitored**:

```python
@dataclass
class Constraint:
    """Define a system constraint."""
    name: str
    metric: str  # What to measure
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    unit: str = ""
    description: str = ""

    # Response configuration
    warning_threshold: float = 0.8  # Alert at 80% of limit
    critical_threshold: float = 0.95  # Emergency at 95%
    auto_fix_available: bool = False  # Can this be auto-repaired

    def check(self, value: float) -> tuple[str, float]:
        """
        Check if value violates constraint.

        Returns:
            (status, utilization_percent)
            status: 'ok', 'warning', 'critical'
        """
        if self.upper_bound is not None:
            utilization = value / self.upper_bound
            if value > self.upper_bound:
                return ('critical', utilization)
            elif value > self.upper_bound * self.critical_threshold:
                return ('critical', utilization)
            elif value > self.upper_bound * self.warning_threshold:
                return ('warning', utilization)

        if self.lower_bound is not None:
            utilization = 1.0 - (value / self.lower_bound)
            if value < self.lower_bound:
                return ('critical', utilization)
            elif value < self.lower_bound * (1 - self.critical_threshold):
                return ('critical', utilization)
            elif value < self.lower_bound * (1 - self.warning_threshold):
                return ('warning', utilization)

        return ('ok', 0.0)

class ConstraintSet:
    """Define all constraints for a system."""

    def __init__(self):
        self.constraints: Dict[str, Constraint] = {}
        self._setup_defaults()

    def _setup_defaults(self):
        """Set up standard constraints."""

        # Disk space
        self.add_constraint(Constraint(
            name='disk_space',
            metric='disk_free_bytes',
            lower_bound=1_000_000_000,  # 1GB minimum
            unit='bytes',
            description='Minimum disk space for operations',
            auto_fix_available=False
        ))

        # Memory
        self.add_constraint(Constraint(
            name='memory_available',
            metric='memory_free_bytes',
            lower_bound=500_000_000,  # 500MB minimum
            unit='bytes',
            description='Minimum free memory',
            auto_fix_available=True  # Can drop caches
        ))

        # Error rate
        self.add_constraint(Constraint(
            name='error_rate',
            metric='error_rate_percent',
            upper_bound=5.0,  # Max 5% error rate
            unit='percent',
            description='Maximum acceptable error rate',
            warning_threshold=0.6,  # Warn at 3%
            auto_fix_available=False
        ))

        # Latency
        self.add_constraint(Constraint(
            name='p99_latency',
            metric='p99_latency_ms',
            upper_bound=1000,  # Max 1 second
            unit='ms',
            description='99th percentile operation latency',
            warning_threshold=0.7,  # Warn at 700ms
            auto_fix_available=True  # Can clear caches, rebuild indexes
        ))

        # WAL size
        self.add_constraint(Constraint(
            name='wal_size',
            metric='wal_bytes',
            upper_bound=1_000_000_000,  # 1GB WAL max
            unit='bytes',
            description='WAL should not grow unbounded',
            warning_threshold=0.7,
            auto_fix_available=True  # Can create snapshot
        ))

        # Concurrent transactions
        self.add_constraint(Constraint(
            name='concurrent_tx',
            metric='concurrent_transaction_count',
            upper_bound=100,
            unit='count',
            description='Maximum concurrent transactions',
            auto_fix_available=False  # Must queue or reject
        ))

    def add_constraint(self, constraint: Constraint):
        """Register a constraint."""
        self.constraints[constraint.name] = constraint

    def check_all(self, metrics: dict) -> dict:
        """Check all constraints against current metrics."""
        results = {}
        violations = []

        for name, constraint in self.constraints.items():
            if constraint.metric not in metrics:
                continue

            value = metrics[constraint.metric]
            status, utilization = constraint.check(value)

            results[name] = {
                'status': status,
                'utilization': utilization,
                'value': value,
                'constraint': constraint
            }

            if status != 'ok':
                violations.append(name)

        return {'constraints': results, 'violations': violations}

# Usage:
constraints = ConstraintSet()
current_metrics = gather_metrics()
check_results = constraints.check_all(current_metrics)

for violation in check_results['violations']:
    constraint = check_results['constraints'][violation]['constraint']
    print(f"VIOLATION: {constraint.description}")

    if constraint.auto_fix_available:
        print(f"  → Auto-fixing by clearing caches")
        clear_caches()
    else:
        print(f"  → Escalating to human")
        alert_human(constraint)
```

### 6.2 Constraint Validation in Configuration

```python
@dataclass
class CorticalConfig:
    """System configuration with constraint validation."""

    # Algorithm parameters with bounds
    pagerank_damping: float = 0.85
    pagerank_iterations: int = 20
    louvain_resolution: float = 2.0
    chunk_size: int = 512

    def __post_init__(self):
        """Validate all configuration constraints."""
        self._validate_constraints()

    def _validate_constraints(self):
        """Validate configuration is within acceptable bounds."""

        # PageRank damping: must be in (0, 1)
        if not (0 < self.pagerank_damping < 1):
            raise ValueError(
                f"pagerank_damping must be in (0, 1), got {self.pagerank_damping}"
            )

        # PageRank iterations: must be positive
        if self.pagerank_iterations < 1:
            raise ValueError(
                f"pagerank_iterations must be >= 1, got {self.pagerank_iterations}"
            )

        # Louvain resolution: must be positive
        if self.louvain_resolution <= 0:
            raise ValueError(
                f"louvain_resolution must be > 0, got {self.louvain_resolution}"
            )

        # Chunk size: must be reasonable
        if self.chunk_size < 10:
            raise ValueError(
                f"chunk_size must be >= 10, got {self.chunk_size}"
            )

        # Warnings for unusual (but valid) values
        if self.louvain_resolution > 20:
            import warnings
            warnings.warn(
                f"louvain_resolution={self.louvain_resolution} is very high. "
                f"This may produce hundreds of clusters. "
                f"Typical range: 1.0-10.0"
            )
```

### 6.3 Runtime Constraint Monitoring

```python
class RuntimeConstraintMonitor:
    """Monitor runtime behavior against defined constraints."""

    def __init__(self, constraints: ConstraintSet):
        self.constraints = constraints
        self.violation_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )

    def on_operation_complete(self, op_result: dict):
        """Called after each operation."""
        metrics = self._extract_metrics(op_result)
        check_results = self.constraints.check_all(metrics)

        # Record violations
        for violation in check_results['violations']:
            self.violation_history[violation].append(time.time())
            constraint = self.constraints.constraints[violation]

            # Decide action
            self._handle_violation(violation, constraint, metrics)

    def _handle_violation(self, name: str,
                         constraint: Constraint,
                         metrics: dict):
        """Handle a constraint violation."""

        # Count recent violations
        recent = [
            t for t in self.violation_history[name]
            if time.time() - t < 60  # Last minute
        ]

        if len(recent) >= 5:
            # Repeated violation
            if constraint.auto_fix_available:
                logger.warning(f"Repeated violation of {name}. Auto-fixing...")
                self._auto_fix(name, constraint)
            else:
                logger.error(f"Repeated violation of {name}. Escalating...")
                alert_human(constraint)

    def _auto_fix(self, constraint_name: str, constraint: Constraint):
        """Apply auto-fix for constraint violation."""
        if constraint_name == 'memory_available':
            logger.info("Clearing caches...")
            self.clear_all_caches()
        elif constraint_name == 'p99_latency':
            logger.info("Rebuilding index...")
            self.rebuild_index()
        elif constraint_name == 'wal_size':
            logger.info("Creating snapshot...")
            self.create_snapshot()
```

---

## 7. Integration: Complete Health Monitoring System

### 7.1 The Central Health Monitor

```python
class DatabaseHealthMonitor:
    """
    Central monitoring system coordinating all health checks,
    diagnostics, anomaly detection, and self-repair.
    """

    def __init__(self, config: CorticalConfig):
        self.config = config

        # Components
        self.health_checks = HealthCheckScheduler()
        self.diagnostics = DiagnosticsCollector()
        self.anomaly_detector = MultiMethodAnomalyDetector()
        self.repair_manager = SelfRepairManager()
        self.constraint_monitor = RuntimeConstraintMonitor(ConstraintSet())
        self.logger = AdaptiveLogger()
        self.escalation = ProgressiveEscalation()

        # State
        self.system_status = "healthy"
        self.incidents: List[Incident] = []

    def on_operation_start(self, op_name: str):
        """Called at start of operation."""
        self.logger.check_verbosity()  # Update log level if needed

    def on_operation_complete(self, op_name: str, result: dict,
                            duration_ms: float):
        """Called after operation completes."""

        # Record metrics
        self.anomaly_detector.record_operation(op_name, duration_ms)
        self.constraint_monitor.on_operation_complete({
            'operation': op_name,
            'duration_ms': duration_ms,
            'result': result
        })

        # Check for anomalies
        anomaly_result = self.anomaly_detector.check_operation(
            op_name, duration_ms, result
        )

        if anomaly_result.is_anomalous:
            self.logger.detect_anomaly(anomaly_result)
            bundle = self.diagnostics.create_bundle(
                'anomaly',
                f"Anomalous operation detected: {anomaly_result.reasons}",
                {'operation': op_name, 'result': anomaly_result}
            )
            self.diagnostics.append_bundle(bundle)
            self.escalation.report_issue('anomaly', anomaly_result.confidence)

    def on_operation_error(self, op_name: str, error: Exception, context: dict):
        """Called when operation fails."""

        # Classify error
        severity = self._classify_error_severity(error)

        # Create diagnostic bundle
        bundle = self.diagnostics.create_bundle(
            'error',
            f"{type(error).__name__}: {str(error)}",
            {'operation': op_name, 'context': context}
        )
        bundle.stack_trace = traceback.format_exc()
        self.diagnostics.append_bundle(bundle)

        # Try to decide on recovery
        decision = self.repair_manager.decide_repair(
            problem_type=self._infer_problem_type(error),
            severity=severity,
            error_context=context
        )

        if decision.should_repair:
            logger.info(f"Attempting auto-repair: {decision.repair_type}")
            repair_result = self.repair_manager.execute_repair(decision)
            if repair_result.success:
                self.escalation.report_issue('transient_error', 0.3)
            else:
                self.escalation.report_issue('repair_failed', 0.8)
        else:
            self.escalation.report_issue('unrecoverable_error', severity)

    def periodic_check(self):
        """Called periodically to run health checks."""

        # Run all scheduled health checks
        health_report = self.health_checks.run_due_checks()

        # Check constraints
        metrics = health_report.get_metrics()
        constraint_results = self.constraint_monitor.constraints.check_all(metrics)

        # Update system status
        self._update_system_status(health_report, constraint_results)

        # Log health summary
        logger.debug(
            f"Health check complete: {self.system_status} | "
            f"Issues: {constraint_results['violations']}"
        )

    def _update_system_status(self, health_report, constraint_results):
        """Update overall system status."""
        violations = constraint_results['violations']
        failed_checks = health_report.get_failed_checks()

        if not violations and not failed_checks:
            self.system_status = "healthy"
        elif len(violations) == 1 and len(failed_checks) == 0:
            self.system_status = "degraded"
        elif len(violations) > 1 or any(c in failed_checks for c in ['quick', 'resource']):
            self.system_status = "unhealthy"
        else:
            self.system_status = "checking"
```

---

## 8. Practical Implementation Examples

### Example 1: Self-Healing a Stale Lock

```python
def handle_stale_lock(lock_holder_id: str, lock_age_sec: float):
    """Handle stale transaction lock automatically."""

    # Diagnose
    context = {
        'lock_holder_id': lock_holder_id,
        'lock_age_sec': lock_age_sec,
        'has_recent_snapshot': check_for_snapshot(),
        'concurrent_operations': count_concurrent_ops()
    }

    # Decide
    decision = repair_manager.decide_repair(
        'stale_lock',
        'medium',
        context
    )

    if not decision.should_repair:
        logger.warning(
            f"Cannot auto-repair stale lock "
            f"({decision.reasons}). Escalating..."
        )
        return False

    # Execute with rollback capability
    try:
        with SafeRepairOperation("unlock_stale", backup_dir) as repair:
            # Force-unlock with diagnostics
            logger.info(f"Force-unlocking stale lock holder {lock_holder_id}")
            clear_lock(lock_holder_id)

            # Verify repair
            if verify_no_locks_held(lock_holder_id):
                logger.info("Stale lock successfully cleared")
                return True
            else:
                raise RuntimeError("Lock is still held after unlock attempt")

    except Exception as e:
        # SafeRepairOperation context manager will rollback
        logger.error(f"Lock repair failed: {e}")
        return False
```

### Example 2: Self-Healing WAL Corruption

```python
def handle_corrupted_wal(corruption_entry: int, total_entries: int):
    """Handle WAL corruption by skipping and rebuilding."""

    # Diagnose
    context = {
        'corruption_at_entry': corruption_entry,
        'total_entries': total_entries,
        'percent_affected': 100 * corruption_entry / total_entries
    }

    # Decide
    decision = repair_manager.decide_repair(
        'corrupted_wal',
        'high',
        context
    )

    if not decision.should_repair:
        logger.critical(
            f"Cannot auto-repair WAL corruption "
            f"(entries {corruption_entry}:{total_entries}). "
            f"Would lose {decision.estimated_data_loss_percent:.1%} of data. "
            f"Escalating..."
        )
        return False

    logger.warning(
        f"Skipping corrupted WAL entries "
        f"({total_entries - corruption_entry} recent entries). "
        f"Data loss: ~{decision.estimated_data_loss_percent:.1%}"
    )

    # Execute repair
    wal.truncate_at(corruption_entry)
    wal.compact()  # Rebuild from entries up to corruption point

    logger.info("WAL repair complete")
    return True
```

### Example 3: Self-Healing Resource Exhaustion

```python
def handle_disk_space_low():
    """Handle low disk space by triggering compaction."""

    disk_free_percent = get_disk_free_percent()

    if disk_free_percent > 20:
        return  # Not critical

    if disk_free_percent > 10:
        logger.warning(
            f"Disk space low ({disk_free_percent:.1%} free). "
            f"Starting compaction..."
        )
        trigger_compaction('aggressive')
    else:
        logger.critical(
            f"Disk space critical ({disk_free_percent:.1%} free). "
            f"Halting writes..."
        )
        set_readonly_mode()
```

---

## 9. Testing Self-Diagnostic Systems

### Testing Anomaly Detection

```python
def test_anomaly_detection():
    """Test anomaly detection catches known anomalies."""

    detector = MultiMethodAnomalyDetector()

    # Train on normal operations
    for i in range(1000):
        detector.record_operation('query', duration_ms=random.gauss(50, 5))

    # Normal operation should not be flagged
    result = detector.check_operation('query', duration_ms=52, result={})
    assert not result.is_anomalous

    # Extreme latency should be flagged
    result = detector.check_operation('query', duration_ms=500, result={})
    assert result.is_anomalous
    assert result.confidence > 0.8

    # Pattern-based anomaly should be flagged
    result = detector.check_operation(
        'query',
        duration_ms=50,
        result={'error': 'SQL injection attempt detected'}
    )
    assert result.is_anomalous
```

### Testing Self-Repair Decisions

```python
def test_repair_decision_logic():
    """Test repair decision making."""

    manager = SelfRepairManager()

    # Should auto-repair fresh stale lock with snapshot
    decision = manager.decide_repair(
        'stale_lock',
        'medium',
        {
            'lock_age_sec': 500,
            'has_recent_snapshot': True,
            'concurrent_operations': 0
        }
    )
    assert decision.should_repair
    assert decision.confidence > 0.9

    # Should NOT auto-repair active lock
    decision = manager.decide_repair(
        'stale_lock',
        'medium',
        {
            'lock_age_sec': 10,  # Fresh lock
            'has_recent_snapshot': True,
            'concurrent_operations': 5  # Active
        }
    )
    assert not decision.should_repair
```

---

## 10. Deployment Checklist

Before deploying self-healing systems:

- [ ] **Health checks configured** - All critical metrics have monitoring
- [ ] **Baselines established** - Normal operation profiles documented
- [ ] **Anomaly detection tested** - Can detect known failure modes
- [ ] **Repair procedures validated** - All auto-fix procedures tested in staging
- [ ] **Escalation rules verified** - Human alerts work correctly
- [ ] **Diagnostics captured** - Sufficient logging for post-incident analysis
- [ ] **Rollback procedures tested** - Can always revert from failed repairs
- [ ] **Constraints documented** - All system bounds explicitly specified
- [ ] **On-call documentation** - Clear escalation paths for incidents
- [ ] **Monitoring dashboards** - Real-time visibility into system health

---

## Summary

Self-diagnostic and self-healing databases require **five coordinated systems**:

1. **Health Checks**: Continuous, layered monitoring with appropriate frequency
2. **Diagnostic Logs**: Strategic sampling that increases during anomalies
3. **Anomaly Detection**: Multi-method detection with adaptive baselines
4. **Self-Repair**: Cascade of recovery methods with automatic rollback
5. **Investigation Triggers**: Progressive escalation with constraint-based decisions

The key insight: **Make systems observable, smart about what to fix, and honest about when to escalate**. Systems that try to fix everything fail catastrophically; systems that know their limits survive.

