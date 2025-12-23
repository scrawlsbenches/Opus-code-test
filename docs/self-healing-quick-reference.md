# Self-Healing Patterns: Quick Reference

A cheat sheet for implementing and understanding self-diagnostic and self-healing database systems.

---

## The Self-Healing Stack (5 Layers)

```
┌─────────────────────────────────────────────────────┐
│ Layer 5: ESCALATION                                 │
│ When to alert humans, with full context             │
├─────────────────────────────────────────────────────┤
│ Layer 4: REPAIR DECISION                            │
│ Classify problems, decide fix or escalate           │
├─────────────────────────────────────────────────────┤
│ Layer 3: SELF-REPAIR                                │
│ Automatic recovery strategies (with rollback)       │
├─────────────────────────────────────────────────────┤
│ Layer 2: ANOMALY DETECTION                          │
│ Catch problems early before critical failure        │
├─────────────────────────────────────────────────────┤
│ Layer 1: OBSERVABILITY                              │
│ Metrics, health checks, diagnostic logs             │
└─────────────────────────────────────────────────────┘
```

---

## Quick Reference by Layer

### Layer 1: OBSERVABILITY - "What's Happening?"

**What to measure:**
```
Operation latencies     → p50, p95, p99 (captures tail)
Operation counts        → Throughput in ops/sec
Error rates            → Errors / total operations
Resource usage         → Memory, disk, connections
Data integrity         → Checksum failures, orphaned refs
```

**Sampling strategy:**
```
Normal operation     → Sample 0.1% (1 in 1000)
Anomaly detected    → Sample 1% (1 in 100)
Incident mode       → Sample 100% (all operations)
```

**Code pattern:**
```python
@dataclass
class Metrics:
    operation: str
    p50_ms: float
    p95_ms: float
    p99_ms: float
    count: int
    errors: int

# Record everything
metrics.record_timing("search", duration_ms=45.3)

# Query metrics
if metrics.get("search").p99_ms > baseline * 2:
    escalate("Latency spike")
```

**Where in Cortical:**
- `cortical/observability.py` - MetricsCollector
- `cortical/processor` - enable_metrics=True
- `scripts/corpus_health.py` - Health analysis

---

### Layer 2: ANOMALY DETECTION - "Is Something Wrong?"

**Detection methods (combine all):**

```
Statistical       → Z-score > 3 (99.7% confidence)
Pattern-based     → Known problem signatures
Distribution      → Vocabulary/behavior change
Constraint        → Violates resource limits
```

**Example thresholds:**
```
Latency anomalous if     → duration > baseline + 3*stdev
Error rate anomalous if  → errors/sec > 0.05
Memory anomalous if      → usage > 0.95 * limit
```

**Code pattern:**
```python
# Train on normal
baseline = []
for _ in range(1000):
    baseline.append(measure_operation())

# Detect anomaly
current = measure_operation()
z_score = (current - mean(baseline)) / stdev(baseline)

if z_score > 3:  # Anomalous
    logging.warning(f"Anomalous: {z_score:.1f}σ deviation")
```

**Where in Cortical:**
- `cortical/spark/anomaly.py` - AnomalyDetector
- `cortical/reasoning/crisis_manager.py` - Crisis classification

---

### Layer 3: SELF-REPAIR - "Can We Fix It?"

**Repair strategies by problem type:**

| Problem | Strategy | Safety | Downtime |
|---------|----------|--------|----------|
| Stale computation | Recompute | Safe | ~doc_count * 0.01s |
| Stale lock | Force unlock | Safe if old | ~0.1s |
| Corrupted index | Rebuild | Safe | ~doc_count * 0.05s |
| WAL corruption | Truncate+recover | Risky if early | ~5s |
| Memory pressure | Clear caches | Safe | ~0.5s |
| Disk full | Compact+vacuum | Safe | ~30s |

**Code pattern (ALWAYS use safe wrapper):**
```python
try:
    with safe_repair("rebuild_index", backup_dir):
        # Repair happens here
        clear_index()
        rebuild_from_documents()
        # If exception → automatic rollback

except RepairFailed:
    # Failed to auto-repair
    alert_human("Index rebuild failed")
```

**Decision logic:**
```python
repair = decide_repair(problem_type, context)

if repair.should_repair and repair.confidence > 0.7:
    execute_repair(repair)
else:
    escalate_to_human(repair.reason)
```

**Where in Cortical:**
- `cortical/reasoning/graph_persistence.py` - GraphRecovery (4-level cascade)
- `cortical/got/recovery.py` - Transaction recovery
- `cortical/wal.py` - Write-ahead log recovery

---

### Layer 4: REPAIR DECISION - "Should We Fix It?"

**Decision tree:**
```
Is this a known, low-risk fix?
  YES → Can we do it safely (rollback available)?
    YES → Execute repair
    NO  → Escalate ("Not safe without rollback")
  NO  → Is this time-critical?
    YES → Try repair with high monitoring
    NO  → Escalate ("Would rather be safe")

Can repair succeed? (confidence check)
  <70% → Escalate ("Too risky")
  70-90% → Try with close monitoring
  >90% → Execute
```

**Problem classification:**

```
HICCUP (low risk)       → Auto-fix (stale lock, retry)
OBSTACLE (medium risk)  → Try fix, monitor, be ready to rollback
WALL (high risk)        → Escalate to human (corruption, config)
CRISIS (danger)         → STOP NOW (data loss, security)
```

**Code pattern:**
```python
# Classify problem
severity = classify_problem(error)

if severity == Severity.HICCUP:
    retry_operation()  # Safe to retry
elif severity == Severity.OBSTACLE:
    decision = repair_manager.decide_repair(...)
    if decision.confidence > 0.8:
        execute_repair(...)  # Try it
    else:
        escalate(...)  # Don't risk it
else:  # WALL or CRISIS
    escalate_to_human(full_context)  # Explain what happened
```

---

### Layer 5: ESCALATION - "When to Call Humans?"

**Escalation triggers:**

```
Immediate (CRISIS)     → Data corruption, security issue
High (within 5 min)    → Repair failed 3+ times, repeated errors
Medium (within 30 min) → Performance degradation, constraints violated
Low (monitoring)       → Anomalies detected, preparing report
```

**Alert rules:**
```
Single issue       → Log it (sample rate 0.01)
2-3 issues/min    → Warn (sample rate 0.1, increase monitoring)
5+ issues/min     → Alert human (sample rate 1.0, full context)
Pattern detected  → Alert + report + suggestions
```

**What to include in escalation:**

```
1. WHAT
   - What is the problem (issue_type)
   - What evidence (metrics, timestamps)
   - When it started

2. IMPACT
   - Is system still working?
   - Performance degradation?
   - Data at risk?

3. WHAT WE TRIED
   - Auto-repair attempts (success/failure)
   - Why they didn't work
   - Confidence levels

4. WHAT YOU SHOULD DO
   - Recommended actions
   - Documentation links
   - Escalation contacts
```

**Code pattern:**
```python
class AlertManager:
    def report_issue(self, issue, severity):
        # Count recent issues
        count = count_recent(issue)

        if count >= 5:  # Threshold
            report = generate_escalation_report(issue)
            alert_human(report)

# Example report:
report = {
    'issue': 'latency_anomaly',
    'evidence': [
        'p99 latency: 1000ms (baseline 50ms)',
        '15 anomalies in last 5 minutes',
        'Memory usage: 850MB / 1000MB'
    ],
    'impact': 'Searches slow but working',
    'attempted': 'Cache clear (unsuccessful)',
    'suggestions': [
        'Check for index corruption',
        'Consider backup restore if corruption detected',
        'Review system load'
    ]
}
```

---

## Constraint Examples

**System constraints (define bounds):**

```python
# Example constraints
constraints = {
    'disk_space': {'minimum': 1_000_000_000, 'unit': 'bytes'},  # 1GB
    'memory_available': {'minimum': 500_000_000},  # 500MB
    'error_rate': {'maximum': 0.05},  # 5%
    'p99_latency': {'maximum': 1000},  # 1 second
    'concurrent_transactions': {'maximum': 100},
    'wal_size': {'maximum': 1_000_000_000},  # 1GB
}

# Check constraints before operation
if not constraint_check('write'):
    raise ResourceExhausted("Disk < 1GB")

# Monitor after operation
violations = constraint_monitor.check_all(current_metrics)
if violations:
    take_action(violations)
```

**Constraint response:**

```
Status       % of Limit    Action
─────────────────────────────────────
OK           0-80%         Continue normally
WARNING      80-95%        Log, increase monitoring
CRITICAL     95%+          Alert, consider limiting operations
EXCEEDED     >100%         Alert, restrict operations
```

---

## Problem Classification Matrix

```
┌──────────────────────┬──────────────────┬──────────────┐
│ PROBLEM              │ CONFIDENCE       │ ACTION       │
├──────────────────────┼──────────────────┼──────────────┤
│ Stale lock           │ High (95%)       │ Auto-fix     │
│ Stale computation    │ Very high (99%)  │ Auto-fix     │
│ Memory pressure      │ Medium (70%)     │ Try fix      │
│ Slow query           │ Low (50%)        │ Escalate     │
│ Index corruption     │ Very low (30%)   │ Escalate     │
│ WAL corruption       │ Low (40%)        │ Escalate     │
│ Data corruption      │ Very low (10%)   │ STOP + alert │
│ Security issue       │ Very low (20%)   │ STOP + alert │
└──────────────────────┴──────────────────┴──────────────┘
```

---

## Implementation Priorities

### MVP (Minimum Viable Self-Healing)

**Priority 1: Observability**
```python
processor = CorticalTextProcessor(enable_metrics=True)
# Now every operation is timed and counted
```

**Priority 2: Health Checks**
```python
health = analyze_corpus_health(processor)
if health['layers']['tokens']['avg_connections'] == 0:
    print("ERROR: No connections, rebuild needed")
```

**Priority 3: Error Handling**
```python
try:
    result = processor.compute_all()
except Exception as e:
    # Instead of just logging, classify severity
    severity = classify_error(e)
    if severity == 'low':
        retry_operation()
    else:
        alert_human(e)
```

### Production (Full Self-Healing)

Add in order:
1. Anomaly detection (catch problems early)
2. Repair decision logic (know what's safe)
3. Safe repair operations (with rollback)
4. Progressive escalation (smart alerts)
5. Escalation reports (context for humans)

---

## Common Patterns

### Pattern: Stateless Operation with Fallback
```python
def operation_with_fallback():
    try:
        return primary_method()
    except Exception:
        return fallback_method()
```

### Pattern: Repair with Checkpoint
```python
try:
    checkpoint = create_checkpoint()
    execute_repair()
    return True
except Exception:
    restore_checkpoint(checkpoint)
    return False
```

### Pattern: Bounded Retry
```python
max_attempts = 3
for attempt in range(max_attempts):
    try:
        return attempt_operation()
    except TransientError:
        if attempt < max_attempts - 1:
            time.sleep(2 ** attempt)  # Exponential backoff
        else:
            raise PermanentError("Repeated failures")
```

### Pattern: Adaptive Sampling
```python
def should_log(operation):
    if incident_mode:
        return True  # Log everything
    elif anomaly_detected:
        return random.random() < 0.1  # 10%
    else:
        return random.random() < 0.001  # 0.1%
```

### Pattern: Constraint Check Before Operation
```python
def safe_operation():
    if not constraint_check('write'):
        return SKIP  # Or raise exception

    # Safe to proceed
    return do_operation()
```

---

## Metric Formulas

```
Error Rate = Total Errors / Total Operations
Availability = (Total Time - Downtime) / Total Time
Latency P99 = Percentile(All Latencies, 0.99)
Memory Utilization = Memory Used / Memory Limit
Staleness = Time Since Last Compute
Anomaly Score = |value - mean| / stdev (Z-score)
Confidence = Correct Predictions / Total Predictions
```

---

## Troubleshooting Self-Healing Systems

| Symptom | Cause | Fix |
|---------|-------|-----|
| Too many alerts | Alert threshold too low | Increase threshold |
| Alerts ignored | Threshold too high | Decrease threshold |
| Repairs fail | Decision logic too aggressive | Increase confidence requirement |
| Repairs don't happen | Decision logic too conservative | Decrease confidence requirement |
| No improvement after repair | Wrong diagnosis | Better anomaly detection |
| Data corruption spreads | Repair not rolling back | Add checkpoints |
| Memory keeps growing | Cache not clearing | Check cache clear implementation |

---

## Files to Create/Modify

```
cortical/
├── observability.py               ✅ (metrics, exists)
├── config.py                      ✅ (constraints, exists)
├── spark/anomaly.py               ✅ (detection, exists)
└── reasoning/
    ├── crisis_manager.py          ✅ (levels, exists)
    └── graph_persistence.py       ✅ (recovery, exists)

New files to create:
├── health_monitor.py              (health checks)
├── anomaly_detector.py            (multi-method detection)
├── repair_manager.py              (auto-repair)
├── repair_decision.py             (decision logic)
├── escalation_manager.py          (alert management)
└── escalation_report.py           (human-readable reports)
```

---

## Key Takeaways

1. **Observability is foundational** - You can't fix what you can't measure
2. **Detect problems early** - Anomalies are warnings, catch them before failure
3. **Classify by severity** - Not all problems need the same response
4. **Repair safely** - Always have a rollback plan
5. **Escalate intelligently** - Alert humans when you hit your limits
6. **Document decisions** - Humans need context to debug

---

## Related Documentation

- **Full patterns**: `docs/database-self-diagnostic-patterns.md`
- **Codebase mapping**: `docs/self-healing-codebase-patterns.md`
- **Implementation guide**: `docs/implementing-self-healing.md`
- **Crisis management**: `cortical/reasoning/crisis_manager.py`
- **Graph recovery**: `cortical/reasoning/graph_persistence.py`
- **Observability**: `cortical/observability.py`

