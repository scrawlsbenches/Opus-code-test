# Self-Healing Patterns: What's Implemented in Cortical

This document maps the self-diagnostic and self-healing patterns from `database-self-diagnostic-patterns.md` to actual implementations in the Cortical codebase.

---

## 1. Health Checks: Existing Implementations

### 1.1 Corpus Health Analysis (IMPLEMENTED)

**File**: `cortical/processor/` + `scripts/corpus_health.py`

**What exists**:
- `analyze_corpus_health()` function computes:
  - Document count, size distribution, document types
  - Layer statistics (tokens, bigrams, concepts, documents)
  - Connection statistics (average, max connections per layer)
  - Semantic relation statistics
  - Concept cluster quality metrics

**How to use**:
```python
from cortical.processor import CorticalTextProcessor
from scripts.corpus_health import analyze_corpus_health

processor = CorticalTextProcessor()
# ... load data ...

health = analyze_corpus_health(processor, check_concepts=True)
print(f"Document count: {health['document_count']}")
print(f"Layers: {health['layers']}")
```

**Extend with**:
- Staleness tracking integration (check `processor.is_stale()`)
- Resource constraint monitoring
- Latency metrics collection

### 1.2 Observability & Metrics Collection (IMPLEMENTED)

**File**: `cortical/observability.py`

**What exists**:
```python
class MetricsCollector:
    - record_timing(operation, duration_ms, trace_id, context)
    - record_count(operation, count)
    - get_operation_stats()
    - get_percentile_stats()
    - trace_operation() context manager

# Usage:
processor = CorticalTextProcessor(enable_metrics=True)
processor.process_document("doc1", "text")
processor.compute_all()

metrics = processor.get_metrics()
summary = processor.get_metrics_summary()
```

**Characteristics**:
- Per-operation timing tracking
- Percentile calculations (p50, p95, p99)
- Trace context linking
- Configurable history limits (memory bounded)

**Extend with**:
- Automated baseline learning
- Anomaly threshold detection
- Alert firing based on latency spikes

### 1.3 Staleness Tracking (IMPLEMENTED)

**File**: `cortical/processor/core.py`

**What exists**:
```python
class CorticalTextProcessor:
    COMP_TFIDF = 'tfidf'
    COMP_PAGERANK = 'pagerank'
    COMP_ACTIVATION = 'activation'
    # ... other computation types ...

    def is_stale(self, computation: str) -> bool:
        """Check if computation needs refresh."""

    def get_stale_computations(self) -> Set[str]:
        """Get all stale computations."""

    def _mark_all_stale(self):
    def _mark_fresh(self, computation: str):
```

**Pattern**:
- Every computation type tracked separately
- Marked stale when data changes
- Marked fresh after computing

**Use case**: Health check can verify which computations are current

---

## 2. Diagnostic Logs: Existing Implementations

### 2.1 Adaptive Logging Foundation (PARTIALLY IMPLEMENTED)

**What exists**:
- Python standard logging throughout
- Observability module for timing context
- Exception handling with context preservation

**What's missing**:
- Adaptive sampling based on anomaly state
- Diagnostic bundles that consolidate information
- Context-aware log verbosity escalation

**How to extend**:
```python
# Create adaptive logger
from cortical.observability import AdaptiveLogger

logger = AdaptiveLogger()
logger.detect_anomaly('constraint_violation')  # Escalates verbosity
logger.log_operation('search', duration_ms=150)
```

### 2.2 Transaction Logging (IMPLEMENTED)

**File**: `cortical/wal.py` + `cortical/got/wal.py`

**What exists**:
```python
@dataclass
class BaseWALEntry:
    operation: str
    timestamp: str
    payload: Dict[str, Any]
    checksum: str  # SHA256 for integrity

    def verify() -> bool:
        """Verify checksum matches content."""

@dataclass
class TransactionWALEntry(BaseWALEntry):
    seq: int  # Sequence number
    tx_id: str  # Transaction ID

# Usage:
writer = WALWriter("corpus_wal")
writer.append(WALEntry(operation="add_document", doc_id="doc1", ...))
```

**Characteristics**:
- Append-only log with checksums
- Transaction boundaries tracked
- Replays available for recovery

---

## 3. Anomaly Detection: Existing Implementations

### 3.1 Prompt Injection Detection (IMPLEMENTED)

**File**: `cortical/spark/anomaly.py`

**What exists**:
```python
class AnomalyDetector:
    def __init__(self, ngram_model, perplexity_threshold=2.0, ...):
        """Initialize with customizable thresholds."""

    @dataclass
    class AnomalyResult:
        is_anomalous: bool
        confidence: float  # 0.0 = normal, 1.0 = anomalous
        reasons: List[str]
        metrics: Dict[str, float]

    def check(query: str) -> AnomalyResult:
        """Multi-method anomaly detection."""

# Detection methods:
# 1. Perplexity-based (statistical)
# 2. Pattern-based (injection patterns)
# 3. Distribution-based (vocabulary coverage)
# 4. Length anomalies (too short/long)
```

**Extends to**:
- Known injection patterns (XSS, SQL injection, template injection)
- N-gram model for statistical deviation
- Unknown vocabulary detection

**How to use**:
```python
from cortical.spark import AnomalyDetector, NGramModel

model = NGramModel()
model.train(documents)
detector = AnomalyDetector(model)

result = detector.check("user query")
print(f"Anomalous: {result.is_anomalous}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Reasons: {result.reasons}")
```

### 3.2 Crisis Management Framework (IMPLEMENTED)

**File**: `cortical/reasoning/crisis_manager.py`

**What exists**:
```python
class CrisisLevel(Enum):
    HICCUP = 1      # Self-recoverable
    OBSTACLE = 2    # Needs adaptation
    WALL = 3        # Needs human intervention
    CRISIS = 4      # Immediate stop required

@dataclass
class CrisisEvent:
    id: str
    level: CrisisLevel
    description: str
    context: Dict[str, Any]
    timestamp: datetime
    action_taken: Optional[RecoveryAction]
    resolution: Optional[str]
    resolved_at: Optional[datetime]
    lessons_learned: List[str]

class RepeatedFailureTracker:
    """Detect repeated failures and escalate."""
    def record_attempt(self, hypothesis: str, action: str, result: str):
    def should_escalate(self) -> bool:  # After 3+ failures
    def get_escalation_report(self) -> str:

class ScopeCreepDetector:
    """Detect project scope drift."""
    def detect_creep(self) -> Tuple[bool, List[str]]:  # Warning signs
```

**Level definitions**:
- **HICCUP**: Fix and continue (stale lock, minor config issue)
- **OBSTACLE**: Pause and adapt (constraint exceeded, performance issue)
- **WALL**: Stop and escalate (fundamental assumption wrong)
- **CRISIS**: Immediate halt (data loss, security issue)

**Integration pattern**:
Tie this to health checks - escalate when:
- Anomaly detected with high confidence → OBSTACLE
- Multiple repair attempts fail → WALL
- Data corruption detected → CRISIS

---

## 4. Self-Repair: Existing Implementations

### 4.1 Graph Persistence & Recovery (IMPLEMENTED)

**File**: `cortical/reasoning/graph_persistence.py`

**Recovery cascade** (4 levels):
```
Level 1: WAL Replay          (Fastest, ~10-100ms)
  └─ Load snapshot + replay WAL entries

Level 2: Snapshot Rollback   (Fast, ~100-500ms)
  └─ Load previous snapshot generation

Level 3: Git History         (Moderate, ~1-5s)
  └─ Restore from git commit

Level 4: Chunk Reconstruction (Slowest, ~5-30s)
  └─ Rebuild from operation logs
```

**Implementation**:
```python
from cortical.reasoning.graph_persistence import GraphRecovery

recovery = GraphRecovery(wal_dir='reasoning_wal')

if recovery.needs_recovery():
    result = recovery.recover()  # Tries levels in order
    print(f"Recovered at level {result.level_used}")
    print(f"Nodes: {result.nodes_recovered}, Edges: {result.edges_recovered}")
```

**Key features**:
- Automatic escalation through levels
- Integrity verification after each level
- Minimal data loss (recovers most recent valid state)

### 4.2 GoT Transaction Recovery (IMPLEMENTED)

**File**: `cortical/got/recovery.py`

**What exists**:
```python
@dataclass
class RecoveryResult:
    success: bool
    recovered_transactions: int
    rolled_back: List[str]
    corrupted_entities: List[str]
    corrupted_wal_entries: int
    actions_taken: List[str]

class RecoveryManager:
    """Crash recovery for transaction system."""

    def __init__(self, got_dir: Path):
    def needs_recovery(self) -> bool:
    def recover(self) -> RecoveryResult:
    def verify_checksums(self) -> bool:
    def repair_orphans(self) -> RepairResult:
```

**Recovery process**:
1. Check WAL for incomplete transactions
2. Rollback incomplete transactions
3. Verify entity checksums
4. Repair orphaned entities

**Stale lock recovery**:
```python
def handle_stale_locks(self) -> int:
    """Force-unlock stale transaction holders."""
    unlocked = 0
    for holder_id, lock_age in self.get_stale_locks():
        if lock_age > 300:  # 5 minutes
            self.force_unlock(holder_id)
            unlocked += 1
    return unlocked
```

### 4.3 Safe Repair with Rollback (PATTERN READY)

**File**: `cortical/reasoning/` (example pattern)

**Pattern to implement**:
```python
class SafeRepairOperation:
    """Encapsulate repair with automatic rollback."""

    def __enter__(self):
        # Create backup
        shutil.copytree(self.data_dir, self.backup_dir)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Rollback on exception
            shutil.rmtree(self.data_dir)
            shutil.copytree(self.backup_dir, self.data_dir)

# Usage:
with SafeRepairOperation("rebuild_index") as repair:
    repair.rebuild_index()  # Rolls back if exception
```

---

## 5. Investigation Triggers: Existing Implementations

### 5.1 Crisis Escalation Levels (IMPLEMENTED)

**File**: `cortical/reasoning/crisis_manager.py`

**Current implementation**:
```python
class CrisisLevel(Enum):
    HICCUP = 1      # Auto-fix (test failure, minor misunderstanding)
    OBSTACLE = 2    # Adapt (verification failing, blocked dependency)
    WALL = 3        # Escalate (assumption false, multiple failures)
    CRISIS = 4      # Stop (causing damage, security issue)

@dataclass
class CrisisEvent:
    level: CrisisLevel
    action_taken: RecoveryAction  # CONTINUE, ADAPT, ROLLBACK, ESCALATE, STOP
    lessons_learned: List[str]
```

**Escalation patterns**:
- Repeated failures → escalate level
- Pattern analysis in escalation reports
- Actionable suggestions for humans

### 5.2 Alert Rules (READY TO IMPLEMENT)

**Pattern from code**:
```python
@dataclass
class AlertRule:
    name: str
    condition: Callable[[dict], bool]  # Checks metrics
    severity: str  # 'info', 'warning', 'critical'
    escalate_after_count: int = 1
    window_sec: int = 300

class SmartAlertManager:
    def register_rule(self, rule: AlertRule):
    def check_metrics(self, metrics: dict) -> List[str]:
        # Returns alerts that should fire
```

**To implement**:
Create rule definitions for:
- High error rate (>5%)
- Disk almost full (<10%)
- Constraint violations
- Data corruption detected
- Latency anomalies (p99 > 2x baseline)

---

## 6. Constraint Definitions: Existing Implementations

### 6.1 Configuration Validation (IMPLEMENTED)

**File**: `cortical/config.py`

**What exists**:
```python
@dataclass
class CorticalConfig:
    """Configuration with validation."""

    pagerank_damping: float = 0.85
    pagerank_iterations: int = 20
    louvain_resolution: float = 2.0
    # ... 30+ parameters ...

    def __post_init__(self):
        """Validate configuration values."""
        self._validate()

    def _validate(self):
        """Validate all constraints."""
        # PageRank damping: must be in (0, 1)
        if not (0 < self.pagerank_damping < 1):
            raise ValueError(...)

        # Louvain resolution: must be > 0
        if self.louvain_resolution <= 0:
            raise ValueError(...)

        # Warnings for unusual values
        if self.louvain_resolution > 20:
            warnings.warn("Very high resolution...")
```

**Validation patterns**:
- **Range bounds**: `0 < value < 1`
- **Minimum values**: `value >= minimum`
- **Relative constraints**: `chunk_overlap < chunk_size`
- **Warnings for unusual values**: High resolution → many clusters

### 6.2 GoT Durability Constraints (IMPLEMENTED)

**File**: `cortical/got/config.py`

**What exists**:
```python
class DurabilityMode(Enum):
    """Durability controls fsync behavior."""
    PARANOID = "paranoid"    # fsync every op, slowest, safest
    BALANCED = "balanced"    # fsync on commit (recommended)
    RELAXED = "relaxed"      # no fsync, fastest, least safe

@dataclass
class GoTConfig:
    durability: DurabilityMode = DurabilityMode.BALANCED
```

**Trade-offs defined**:
- PARANOID: ~36 ops/s, zero data loss
- BALANCED: ~150-200 ops/s (recommended)
- RELAXED: ~500+ ops/s, power loss risk

---

## 7. Implementation Status: Self-Healing Features

### Implemented ✅

| Feature | File | Status | Usage |
|---------|------|--------|-------|
| Health checks | `scripts/corpus_health.py` | Implemented | `analyze_corpus_health(processor)` |
| Metrics collection | `cortical/observability.py` | Implemented | `enable_metrics=True` |
| Staleness tracking | `cortical/processor/core.py` | Implemented | `is_stale(COMP_TFIDF)` |
| WAL logging | `cortical/wal.py` | Implemented | `WALWriter.append()` |
| Anomaly detection | `cortical/spark/anomaly.py` | Implemented | `AnomalyDetector.check()` |
| Crisis levels | `cortical/reasoning/crisis_manager.py` | Implemented | `CrisisLevel.HICCUP/OBSTACLE/WALL/CRISIS` |
| Graph recovery | `cortical/reasoning/graph_persistence.py` | Implemented | `GraphRecovery.recover()` |
| GoT recovery | `cortical/got/recovery.py` | Implemented | `RecoveryManager.recover()` |
| Config validation | `cortical/config.py` | Implemented | `CorticalConfig._validate()` |

### Ready to Implement 🟡

| Feature | Location | Notes |
|---------|----------|-------|
| Adaptive logging | Log it in `AdaptiveLogger` pattern | Strategic sampling based on anomaly state |
| Diagnostic bundles | Create in diagnostics module | Consolidate context when errors occur |
| Baseline learning | Extend `ObservabilityModule` | Track normal operation baseline |
| Smart alert rules | Create `AlertManager` + rules | Multi-signal escalation |
| Constraint monitoring | Extend `ConstraintSet` | Monitor resources in real-time |
| Progressive escalation | Implement in crisis manager | Don't escalate immediately |
| Auto-fix routing | Create `SelfRepairRouter` | Route problems to appropriate fix |

---

## 8. Extension Recommendations

### Priority 1: Complete Anomaly Detection Pipeline

**Current gap**: We detect anomalies but don't react to them systematically.

**Implementation**:
1. Add multi-method anomaly detection:
   - Statistical (latency deviation)
   - Pattern-based (known problem patterns)
   - Behavioral (change from history)

2. Create adaptive alert escalation:
   - Single anomaly → log (sample rate 0.01)
   - Multiple anomalies → warn (sample rate 0.1)
   - Spike pattern → alert human (sample rate 1.0)

3. Link to crisis manager:
```python
anomaly_result = detector.check_operation(...)
if anomaly_result.is_anomalous:
    # Escalate if repeated
    crisis_mgr.report_anomaly(anomaly_result)
    crisis_mgr.check_escalation()  # → OBSTACLE or WALL
```

### Priority 2: Constraint Monitoring Loop

**Current gap**: Constraints defined but not actively monitored during operations.

**Implementation**:
1. Create `ConstraintMonitor` that wraps operations:
```python
class ConstraintMonitor:
    def pre_operation_check(self, op_type: str) -> bool:
        """Check if operation can proceed."""
        # Check disk space for writes
        # Check memory for large queries
        # Check concurrent count

    def post_operation_check(self, op_result: dict) -> List[str]:
        """Check operation didn't violate constraints."""
        # Check latency isn't anomalous
        # Check error rate isn't spiking
        # Return list of violations if any
```

2. Integrate with operations:
```python
def find_documents_for_query(self, query: str):
    if not constraint_monitor.pre_operation_check('search'):
        raise ResourceExhausted("Disk space < 1GB")

    result = self._search_impl(query)

    violations = constraint_monitor.post_operation_check({
        'operation': 'search',
        'duration_ms': ...,
        'result_count': len(result)
    })
```

### Priority 3: Smart Escalation Framework

**Current gap**: Crisis levels exist but no progressive escalation.

**Implementation**:
```python
class ProgressiveEscalation:
    def __init__(self):
        self.current_level = 0  # 0=normal, 1=watch, 2=warn, 3=escalate

    def report_issue(self, issue_type: str, severity: float):
        # Classify to level
        # Only escalate if level increases and reasons accumulate
        # Deescalate after threshold period of no issues

escalation = ProgressiveEscalation()
escalation.report_issue('latency_spike', 0.8)  # Level 1
escalation.report_issue('latency_spike', 0.9)  # Still level 1 (same issue)
escalation.report_issue('high_error_rate', 0.7)  # Level 2 (different issue)
escalation.report_issue('high_error_rate', 0.9)  # Level 3 (escalate to human)
```

### Priority 4: Auto-Fix Decision Routing

**Current gap**: We can recover in various ways but no smart routing.

**Implementation**:
Create `SelfRepairRouter` that decides:
- Which repair strategy to use
- Whether repair is safe (confidence check)
- When to escalate instead of repair
- How to validate repair success

```python
class SelfRepairRouter:
    def route_repair(self, problem_type: str,
                     context: dict) -> RepairDecision:
        """Decide how to repair or escalate."""

        if problem_type == 'stale_lock':
            return self._decide_lock_repair(context)
        elif problem_type == 'corrupted_wal':
            return self._decide_wal_repair(context)
        # ... etc ...
```

---

## 9. Testing Self-Healing Features

### Test Anomaly Detection

```python
def test_detects_latency_anomaly():
    detector = MultiMethodAnomalyDetector()

    # Train on normal latencies (50±5ms)
    for _ in range(100):
        detector.record_operation('search', duration_ms=50)

    # Normal operation
    result = detector.check_operation('search', 52, {})
    assert not result.is_anomalous

    # Anomalous (10x slower)
    result = detector.check_operation('search', 500, {})
    assert result.is_anomalous
    assert result.confidence > 0.8
    assert 'statistical' in result.reasons
```

### Test Crisis Escalation

```python
def test_crisis_escalation():
    mgr = CrisisManager()

    # Single hiccup
    mgr.report_crisis(CrisisLevel.HICCUP, "test failure")
    assert mgr.current_level == CrisisLevel.HICCUP

    # Multiple obstacles
    mgr.report_crisis(CrisisLevel.OBSTACLE, "constraint violated")
    mgr.report_crisis(CrisisLevel.OBSTACLE, "constraint violated")
    assert mgr.should_escalate()  # After 2+ same level
```

### Test Safe Repair

```python
def test_safe_repair_rollback():
    """Test repair rolls back on failure."""

    initial_state = copy_state()

    try:
        with SafeRepairOperation("test") as repair:
            modify_state()  # Simulate repair
            raise RuntimeError("Repair failed")
    except RuntimeError:
        pass

    # State should be restored
    assert copy_state() == initial_state
```

---

## 10. Deployment Checklist

- [ ] Enable metrics collection in production
- [ ] Configure anomaly detector baselines from production traffic
- [ ] Set up alert rules for each constraint
- [ ] Create escalation runbooks for each WALL-level issue
- [ ] Test recovery procedures in staging
- [ ] Document alert response procedures
- [ ] Set up monitoring dashboard
- [ ] Configure log retention
- [ ] Train on-call team

---

## Summary: The Self-Healing Cortical System

The Cortical codebase **already has the foundational layers**:

1. **Observable**: Metrics, health checks, staleness tracking ✅
2. **Diagnosable**: Anomaly detection, crisis levels ✅
3. **Recoverable**: WAL, snapshots, 4-level cascade ✅
4. **Validated**: Config constraints, durability modes ✅

**To complete the self-healing vision**:

5. **Smart escalation** (in progress)
6. **Constraint monitoring** (ready to implement)
7. **Adaptive alerts** (ready to implement)
8. **Auto-fix routing** (ready to implement)

The system is designed to fail gracefully: it observes problems early, understands their severity, repairs what it can, and escalates intelligently when it can't.

