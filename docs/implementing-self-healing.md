# Implementing Self-Healing: Step-by-Step Guide

This guide walks through implementing self-healing features in a database system, using the Cortical system as a concrete example.

---

## Part 1: Foundation - Make It Observable

### Step 1.1: Set Up Metrics Collection

**Goal**: Establish baseline understanding of normal operation.

**Code**:
```python
# In your main processor initialization
from cortical.processor import CorticalTextProcessor

processor = CorticalTextProcessor(enable_metrics=True)

# Process your normal workload
for doc in documents:
    processor.process_document(doc['id'], doc['text'])

# Get baseline metrics
metrics = processor.get_metrics()
for op_name, stats in metrics.items():
    print(f"{op_name}:")
    print(f"  p99: {stats['p99_ms']:.2f}ms")
    print(f"  avg: {stats['avg_ms']:.2f}ms")
    print(f"  count: {stats['count']}")
```

**What to capture**:
- Operation latencies (p50, p95, p99)
- Operation counts (throughput)
- Error counts
- Resource usage (memory, disk)

**Best practice**: Run this for at least 1000 operations to establish reliable baselines.

### Step 1.2: Health Check Function

**Goal**: Create function that runs periodically to assess health.

**Code**:
```python
# health_monitor.py
import time
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class HealthReport:
    timestamp: float
    processor_ok: bool
    metrics: Dict[str, Any]
    violations: list = None

class HealthMonitor:
    def __init__(self, processor, baseline_metrics):
        self.processor = processor
        self.baseline = baseline_metrics

    def run_health_check(self) -> HealthReport:
        """Run periodic health check."""
        report = HealthReport(
            timestamp=time.time(),
            processor_ok=True,
            metrics={},
            violations=[]
        )

        # Check 1: Staleness
        stale = self.processor.get_stale_computations()
        if stale:
            report.metrics['stale_computations'] = list(stale)

        # Check 2: Data integrity
        try:
            # Sample-based integrity check
            sample_docs = list(self.processor.documents.items())[:min(10, len(self.processor.documents))]
            for doc_id, content in sample_docs:
                # Verify document is indexed correctly
                layer0 = self.processor.get_layer(0)  # Tokens
                # Check that doc_id appears in some tokens
                found = any(doc_id in col.document_ids
                           for col in layer0.minicolumns.values())
                if not found:
                    report.violations.append(f"Doc {doc_id} not indexed")
        except Exception as e:
            report.violations.append(f"Integrity check failed: {e}")
            report.processor_ok = False

        # Check 3: Resource constraints
        import psutil
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        report.metrics['memory_mb'] = memory_mb
        if memory_mb > 1024:  # > 1GB
            report.violations.append(f"Memory high: {memory_mb:.0f}MB")

        # Check 4: Performance anomalies
        current_metrics = self.processor.get_metrics()
        for op_name, stats in current_metrics.items():
            baseline = self.baseline.get(op_name)
            if baseline:
                # Check if p99 is 2x baseline
                if stats['p99_ms'] > baseline['p99_ms'] * 2:
                    report.violations.append(
                        f"{op_name} p99 latency spike: "
                        f"{stats['p99_ms']:.0f}ms "
                        f"(baseline {baseline['p99_ms']:.0f}ms)"
                    )

        report.processor_ok = len(report.violations) == 0
        return report

# Usage:
from cortical import CorticalTextProcessor

processor = CorticalTextProcessor(enable_metrics=True)
# ... load data ...

# Record baseline
baseline = processor.get_metrics()

# Create monitor
monitor = HealthMonitor(processor, baseline)

# Run periodic checks
while True:
    report = monitor.run_health_check()
    if report.violations:
        print(f"Health issues detected: {report.violations}")
    time.sleep(60)  # Check every minute
```

---

## Part 2: Diagnosis - Detect Problems Early

### Step 2.1: Add Anomaly Detection

**Goal**: Detect unusual operation patterns.

**Code**:
```python
# anomaly_detector.py
from cortical.spark.anomaly import AnomalyDetector, NGramModel
from typing import List
import statistics

class OperationAnomalyDetector:
    """Detect anomalous operations."""

    def __init__(self):
        self.baseline_latencies = {}
        self.recent_latencies = {}
        self.anomaly_count = 0

    def record_operation(self, op_name: str, duration_ms: float):
        """Record operation timing."""
        if op_name not in self.recent_latencies:
            self.recent_latencies[op_name] = []

        self.recent_latencies[op_name].append(duration_ms)

        # Keep last 1000 measurements
        if len(self.recent_latencies[op_name]) > 1000:
            self.recent_latencies[op_name].pop(0)

    def check_operation(self, op_name: str,
                       duration_ms: float) -> bool:
        """Check if operation latency is anomalous."""

        recent = self.recent_latencies.get(op_name, [])
        if len(recent) < 10:
            return False  # Need baseline

        # Statistical test: is this > 3 std devs from mean?
        mean = statistics.mean(recent)
        stdev = statistics.stdev(recent) if len(recent) > 1 else 0

        if stdev == 0:
            return False  # No variation in baseline

        z_score = abs((duration_ms - mean) / stdev)
        is_anomalous = z_score > 3.0  # 99.7% confidence

        if is_anomalous:
            self.anomaly_count += 1

        return is_anomalous

# Usage in processor:
detector = OperationAnomalyDetector()

start = time.time()
result = processor.find_documents_for_query("test query")
duration = time.time() - start

detector.record_operation('search', duration * 1000)
if detector.check_operation('search', duration * 1000):
    logger.warning(f"Anomalous search latency: {duration*1000:.0f}ms")
    detector.anomaly_count += 1

    # If anomalies cluster, escalate
    if detector.anomaly_count > 10:
        logger.error("High anomaly count. Increasing diagnostics.")
```

### Step 2.2: Integrate Crisis Levels

**Goal**: Classify problems by severity.

**Code**:
```python
# crisis_handler.py
from cortical.reasoning.crisis_manager import (
    CrisisLevel, CrisisEvent, CrisisManager
)

class DatabaseCrisisHandler:
    def __init__(self):
        self.crisis_mgr = CrisisManager()

    def handle_operation_error(self, op_name: str, error: Exception):
        """Classify and handle operation error."""

        # Classify error severity
        severity = self._classify_error(error)

        # Create crisis event
        event = CrisisEvent(
            level=severity,
            description=f"{op_name} failed: {error}",
            context={
                'operation': op_name,
                'error_type': type(error).__name__,
                'error_msg': str(error)
            }
        )

        # Decide action based on level
        if severity == CrisisLevel.HICCUP:
            logger.info(f"Minor issue in {op_name}. Retrying...")
            # Retry the operation
            return self._retry_operation(op_name)

        elif severity == CrisisLevel.OBSTACLE:
            logger.warning(f"Operation blocked: {error}. Adapting...")
            event.action_taken = RecoveryAction.ADAPT
            # Adapt strategy
            return self._adapt_strategy(op_name, error)

        elif severity == CrisisLevel.WALL:
            logger.error(f"Fundamental issue: {error}. Escalating...")
            event.action_taken = RecoveryAction.ESCALATE
            # Generate report for human
            return self._escalate_to_human(event)

        elif severity == CrisisLevel.CRISIS:
            logger.critical(f"CRISIS: {error}. Halting...")
            event.action_taken = RecoveryAction.STOP
            # Halt operations
            return self._stop_operations(event)

    def _classify_error(self, error: Exception) -> CrisisLevel:
        """Classify error by severity."""

        error_type = type(error).__name__

        # Transient errors → HICCUP
        if error_type in ['TimeoutError', 'ConnectionError', 'IOError']:
            return CrisisLevel.HICCUP

        # Resource errors → OBSTACLE
        if error_type in ['MemoryError', 'ResourceError']:
            return CrisisLevel.OBSTACLE

        # Configuration errors → WALL
        if error_type in ['ValueError', 'ConfigurationError']:
            return CrisisLevel.WALL

        # Data corruption → CRISIS
        if 'corruption' in str(error).lower():
            return CrisisLevel.CRISIS

        # Default
        return CrisisLevel.WALL

# Usage:
handler = DatabaseCrisisHandler()

try:
    processor.compute_all()
except Exception as e:
    handler.handle_operation_error('compute_all', e)
```

---

## Part 3: Repair - Fix Problems Automatically

### Step 3.1: Implement Safe Repair Operations

**Goal**: Create repairs that can't cause worse problems.

**Code**:
```python
# safe_repairs.py
import shutil
from pathlib import Path
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

@contextmanager
def safe_repair(repair_name: str, backup_dir: Path):
    """
    Context manager for safe repair operations.

    If exception occurs inside context, automatically rolls back.
    """
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"backup_{repair_name}_{time.time()}"

    try:
        logger.info(f"Starting repair: {repair_name}")
        logger.info(f"Creating backup at {backup_path}")

        # Create backup of current state
        shutil.copytree("data", backup_path)

        yield backup_path

        logger.info(f"Repair {repair_name} completed successfully")

    except Exception as e:
        logger.error(f"Repair {repair_name} failed: {e}. Rolling back...")

        # Restore from backup
        if backup_path.exists():
            shutil.rmtree("data")
            shutil.copytree(backup_path, "data")
            logger.info("Rollback complete")
        else:
            logger.critical("Backup not found! Cannot rollback.")

        raise

class SelfRepairManager:
    """Manage automatic repair operations."""

    def __init__(self, processor, backup_dir="backups"):
        self.processor = processor
        self.backup_dir = Path(backup_dir)
        self.repair_log = []

    def repair_stale_computation(self, computation_type: str) -> bool:
        """Repair stale computation by recomputing."""

        if computation_type == 'tfidf':
            logger.info("Recomputing TF-IDF...")
            self.processor.compute_tfidf()
            return True
        elif computation_type == 'pagerank':
            logger.info("Recomputing PageRank...")
            self.processor.compute_importance()
            return True
        elif computation_type == 'concepts':
            logger.info("Rebuilding concept clusters...")
            self.processor.build_concept_clusters()
            return True
        else:
            logger.warning(f"Don't know how to repair {computation_type}")
            return False

    def repair_index_corruption(self, layer_id: int) -> bool:
        """Repair index corruption by rebuilding layer."""

        logger.info(f"Rebuilding layer {layer_id}...")

        with safe_repair(f"rebuild_layer_{layer_id}", self.backup_dir):
            # Rebuild layer from documents
            layer = self.processor.get_layer(layer_id)

            # Clear and rebuild
            layer.minicolumns.clear()
            layer._id_index.clear()

            # Re-index documents
            for doc_id in self.processor.documents:
                # Re-tokenize/process document
                # This rebuilds the layer
                pass

            logger.info(f"Layer {layer_id} rebuilt successfully")

        return True

    def repair_wal_corruption(self) -> bool:
        """Repair corrupted WAL by truncating and recovering."""

        logger.warning("Detected WAL corruption. Recovering...")

        try:
            # Try to recover from WAL
            from cortical.wal import WALRecovery
            recovery = WALRecovery("corpus_wal")

            if recovery.needs_recovery():
                logger.info("Running WAL recovery...")
                result = recovery.recover()

                if result.success:
                    logger.info(f"WAL recovery successful")
                    return True
                else:
                    logger.error("WAL recovery failed")
                    return False

        except Exception as e:
            logger.error(f"WAL repair failed: {e}")
            return False

    def repair_memory_leak(self) -> bool:
        """Repair memory leak by clearing caches."""

        logger.warning("Memory usage high. Clearing caches...")

        try:
            # Clear query cache
            self.processor.clear_query_cache()

            # Compact structures
            # (processor-specific)

            logger.info("Caches cleared")
            return True

        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
            return False

# Usage:
repair_mgr = SelfRepairManager(processor)

# Repair stale computation
if processor.is_stale('pagerank'):
    repair_mgr.repair_stale_computation('pagerank')

# Repair corruption
repair_mgr.repair_wal_corruption()
```

### Step 3.2: Smart Repair Decision Logic

**Goal**: Decide what can be safely auto-repaired.

**Code**:
```python
# repair_decision.py
from dataclasses import dataclass

@dataclass
class RepairDecision:
    should_repair: bool
    confidence: float  # 0-1, how sure we are
    repair_type: str  # What to do
    estimated_downtime_sec: float
    estimated_data_loss_percent: float
    reason: str

class RepairDecisionEngine:
    """Decide whether to auto-repair or escalate."""

    def decide(self, problem_type: str, context: dict) -> RepairDecision:
        """Decide repair strategy."""

        if problem_type == 'stale_computation':
            return self._decide_stale_computation_repair(context)
        elif problem_type == 'wal_corruption':
            return self._decide_wal_repair(context)
        elif problem_type == 'memory_pressure':
            return self._decide_memory_repair(context)
        else:
            return RepairDecision(
                should_repair=False,
                confidence=0.0,
                repair_type='escalate',
                estimated_downtime_sec=0,
                estimated_data_loss_percent=0,
                reason=f"Unknown problem type: {problem_type}"
            )

    def _decide_stale_computation_repair(self, context: dict) -> RepairDecision:
        """Decide if stale computation can be repaired."""

        # Stale computations are safe to repair
        # They just need recomputation (no data loss)
        computation = context.get('computation_type')
        doc_count = context.get('document_count', 0)

        # Estimate time based on doc count
        # (rough estimate: ~10ms per doc)
        estimated_time = doc_count * 0.01

        return RepairDecision(
            should_repair=True,
            confidence=0.99,
            repair_type='recompute',
            estimated_downtime_sec=estimated_time,
            estimated_data_loss_percent=0,
            reason=f"Recomputing {computation} is safe"
        )

    def _decide_wal_repair(self, context: dict) -> RepairDecision:
        """Decide if WAL corruption can be repaired."""

        corruption_point = context.get('corruption_byte_offset', 0)
        total_size = context.get('total_wal_size', 0)
        has_snapshot = context.get('has_recent_snapshot', False)

        # If corruption is near end, can skip recent entries
        corruption_ratio = corruption_point / total_size if total_size > 0 else 0

        if corruption_ratio > 0.9 and has_snapshot:
            # Can skip recent entries and recover
            return RepairDecision(
                should_repair=True,
                confidence=0.8,
                repair_type='wal_truncate',
                estimated_downtime_sec=5,
                estimated_data_loss_percent=1.0,  # Lost ~1% of recent changes
                reason="Corruption near end, can truncate and recover"
            )
        else:
            # Too risky
            return RepairDecision(
                should_repair=False,
                confidence=0.0,
                repair_type='escalate',
                estimated_downtime_sec=0,
                estimated_data_loss_percent=0,
                reason=f"Corruption too early ({corruption_ratio:.1%}), too risky"
            )

    def _decide_memory_repair(self, context: dict) -> RepairDecision:
        """Decide if memory pressure can be auto-repaired."""

        memory_mb = context.get('memory_mb', 0)
        memory_limit = context.get('memory_limit_mb', 1024)
        memory_ratio = memory_mb / memory_limit if memory_limit > 0 else 0

        if memory_ratio > 0.95:
            # Critical - try cache clear
            return RepairDecision(
                should_repair=True,
                confidence=0.7,  # May not help much
                repair_type='clear_caches',
                estimated_downtime_sec=0.5,
                estimated_data_loss_percent=0,
                reason="Memory critical, clearing caches"
            )
        elif memory_ratio > 0.8:
            # Warning - can try
            return RepairDecision(
                should_repair=True,
                confidence=0.8,
                repair_type='clear_caches',
                estimated_downtime_sec=0.1,
                estimated_data_loss_percent=0,
                reason="Memory high, clearing caches may help"
            )
        else:
            return RepairDecision(
                should_repair=False,
                confidence=0.0,
                repair_type='monitor',
                estimated_downtime_sec=0,
                estimated_data_loss_percent=0,
                reason="Memory usage acceptable"
            )

# Usage:
decision_engine = RepairDecisionEngine()

# Check if we should repair
decision = decision_engine.decide('stale_computation', {
    'computation_type': 'pagerank',
    'document_count': 150
})

if decision.should_repair and decision.confidence > 0.7:
    logger.info(f"Auto-repairing: {decision.reason}")
    # Execute repair
    repair_mgr.repair_stale_computation('pagerank')
else:
    logger.warning(f"Escalating to human: {decision.reason}")
    # Alert operator
```

---

## Part 4: Escalation - Know When to Ask for Help

### Step 4.1: Progressive Alert System

**Goal**: Alert humans intelligently, not for every problem.

**Code**:
```python
# escalation.py
import time
from collections import defaultdict

class ProgressiveAlertSystem:
    """Escalate alerts intelligently."""

    def __init__(self, human_contact="ops@example.com"):
        self.human_contact = human_contact
        self.recent_alerts = defaultdict(list)
        self.alert_log = []

    def report_issue(self, issue_type: str, severity: float):
        """Report a potential issue."""

        now = time.time()
        self.recent_alerts[issue_type].append({
            'timestamp': now,
            'severity': severity
        })

        # Keep only recent (5 minute window)
        cutoff = now - 300
        self.recent_alerts[issue_type] = [
            a for a in self.recent_alerts[issue_type]
            if a['timestamp'] > cutoff
        ]

        # Check if should alert
        count = len(self.recent_alerts[issue_type])
        max_severity = max(a['severity'] for a in self.recent_alerts[issue_type])

        if count >= 5 or max_severity > 0.9:
            # Alert human
            self._send_alert(issue_type, count, max_severity)

    def _send_alert(self, issue_type: str, count: int, severity: float):
        """Send alert to human."""

        message = (
            f"Alert: {issue_type} (severity {severity:.1%})\n"
            f"Seen {count} times in last 5 minutes\n"
            f"Escalating to {self.human_contact}"
        )

        logger.critical(message)

        # In production, send email/slack/pagerduty
        # self._email(self.human_contact, message)
        # self._slack(message)

        self.alert_log.append({
            'timestamp': time.time(),
            'issue_type': issue_type,
            'count': count,
            'severity': severity,
            'alerted': True
        })

# Usage:
alerts = ProgressiveAlertSystem()

# Each anomaly increases alert pressure
for i in range(5):
    # On 5th anomaly, human gets alerted
    anomaly_detected = True
    if anomaly_detected:
        alerts.report_issue('latency_anomaly', severity=0.7)
```

### Step 4.2: Human-Readable Escalation Reports

**Goal**: When escalating to humans, give them context they need.

**Code**:
```python
# escalation_report.py
from datetime import datetime, timedelta
import json

class EscalationReportGenerator:
    """Generate reports for human review."""

    def __init__(self, processor, health_monitor, anomaly_detector):
        self.processor = processor
        self.monitor = health_monitor
        self.detector = anomaly_detector

    def generate_full_report(self, issue_type: str) -> str:
        """Generate comprehensive escalation report."""

        lines = [
            "=" * 70,
            f"ESCALATION REPORT: {issue_type.upper()}",
            f"Generated: {datetime.now().isoformat()}",
            "=" * 70,
            ""
        ]

        # Section 1: Current issue
        lines.extend(self._section_current_issue(issue_type))

        # Section 2: System health
        lines.extend(self._section_system_health())

        # Section 3: Recent anomalies
        lines.extend(self._section_recent_anomalies())

        # Section 4: Recovery attempts
        lines.extend(self._section_recovery_attempts())

        # Section 5: Suggestions
        lines.extend(self._section_suggestions(issue_type))

        return "\n".join(lines)

    def _section_current_issue(self, issue_type: str) -> list:
        """Details about current issue."""
        return [
            "## CURRENT ISSUE",
            f"Type: {issue_type}",
            f"First seen: (timestamp)",
            f"Severity: HIGH (escalated to human)",
            ""
        ]

    def _section_system_health(self) -> list:
        """Current system health status."""
        report = self.monitor.run_health_check()

        lines = [
            "## SYSTEM HEALTH",
            f"Status: {'HEALTHY' if report.processor_ok else 'UNHEALTHY'}",
            f"Stale computations: {self.processor.get_stale_computations()}",
            "Metrics:"
        ]

        for op, stats in list(self.processor.get_metrics().items())[:5]:
            lines.append(
                f"  {op}: p99={stats['p99_ms']:.0f}ms "
                f"avg={stats['avg_ms']:.0f}ms count={stats['count']}"
            )

        lines.append("")
        return lines

    def _section_recent_anomalies(self) -> list:
        """Recent detected anomalies."""
        return [
            "## RECENT ANOMALIES",
            f"Count in last 5min: {self.detector.anomaly_count}",
            f"Types: (list of anomaly types)",
            ""
        ]

    def _section_recovery_attempts(self) -> list:
        """Attempts to recover automatically."""
        return [
            "## RECOVERY ATTEMPTS",
            "Auto-repair tried: (list what was tried)",
            "Results: (what worked, what didn't)",
            ""
        ]

    def _section_suggestions(self, issue_type: str) -> list:
        """Suggestions for human action."""

        suggestions = {
            'latency_anomaly': [
                "Check system load and resource usage",
                "Look for index corruption (run integrity check)",
                "Consider cache rebuild if stale"
            ],
            'memory_pressure': [
                "Clear old cache entries",
                "Check for memory leaks in application",
                "Consider reducing corpus size"
            ],
            'corruption_detected': [
                "DO NOT attempt repairs, escalate to database team",
                "Restore from latest clean backup",
                "Investigate root cause of corruption"
            ]
        }

        lines = ["## SUGGESTIONS", "Recommended actions:"]
        for suggestion in suggestions.get(issue_type, ["Review system state"]):
            lines.append(f"- {suggestion}")

        return lines

# Usage:
generator = EscalationReportGenerator(processor, monitor, detector)

if should_escalate:
    report = generator.generate_full_report('latency_anomaly')
    print(report)
    # Email to ops team
```

---

## Part 5: Putting It All Together

### Complete Example: Self-Healing Search System

```python
# complete_example.py
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SelfHealingSearchSystem:
    """
    Complete self-healing search system integrating:
    - Health monitoring
    - Anomaly detection
    - Automatic repair
    - Smart escalation
    """

    def __init__(self, processor_config):
        self.processor = CorticalTextProcessor(
            config=processor_config,
            enable_metrics=True
        )

        # Component setup
        self.health_monitor = HealthMonitor(self.processor, {})
        self.anomaly_detector = OperationAnomalyDetector()
        self.crisis_handler = DatabaseCrisisHandler()
        self.repair_mgr = SelfRepairManager(self.processor)
        self.repair_decision = RepairDecisionEngine()
        self.alerts = ProgressiveAlertSystem()
        self.report_gen = EscalationReportGenerator(
            self.processor, self.health_monitor, self.anomaly_detector
        )

        # Monitoring loop
        self.running = False

    def add_documents(self, documents: list):
        """Add documents with error handling."""
        for doc in documents:
            try:
                self.processor.process_document(doc['id'], doc['text'])
            except Exception as e:
                logger.error(f"Failed to add document {doc['id']}: {e}")
                self.crisis_handler.handle_operation_error(
                    'process_document', e
                )

    def search(self, query: str) -> list:
        """Search with anomaly detection."""
        start = time.time()

        try:
            results = self.processor.find_documents_for_query(query)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            self.crisis_handler.handle_operation_error('search', e)
            return []

        duration = time.time() - start

        # Record operation
        self.anomaly_detector.record_operation('search', duration * 1000)

        # Check for anomaly
        is_anomalous = self.anomaly_detector.check_operation(
            'search', duration * 1000
        )

        if is_anomalous:
            logger.warning(f"Anomalous search latency: {duration*1000:.0f}ms")
            self.alerts.report_issue('latency_anomaly', severity=0.8)

        return results

    def periodic_health_check(self):
        """Run periodic health checks."""
        while self.running:
            try:
                # Run health check
                health = self.health_monitor.run_health_check()

                if not health.processor_ok:
                    logger.warning(f"Health issues: {health.violations}")

                    # Try to repair
                    for violation in health.violations:
                        self._handle_health_violation(violation)

                # Check for repairs needed
                stale = self.processor.get_stale_computations()
                if stale:
                    logger.info(f"Stale: {stale}")
                    for comp in stale:
                        self.repair_mgr.repair_stale_computation(comp)

            except Exception as e:
                logger.error(f"Health check failed: {e}")

            # Sleep before next check
            time.sleep(60)

    def _handle_health_violation(self, violation: str):
        """Handle a health violation."""

        if 'memory' in violation.lower():
            # Try to clear caches
            self.repair_mgr.repair_memory_leak()
        elif 'corruption' in violation.lower():
            # Try WAL repair
            if not self.repair_mgr.repair_wal_corruption():
                # If repair fails, escalate
                report = self.report_gen.generate_full_report('corruption')
                logger.critical(report)

    def start(self):
        """Start monitoring."""
        self.running = True
        import threading
        monitor_thread = threading.Thread(target=self.periodic_health_check)
        monitor_thread.daemon = True
        monitor_thread.start()

    def stop(self):
        """Stop monitoring."""
        self.running = False

# Usage:
from cortical import CorticalConfig

config = CorticalConfig()
system = SelfHealingSearchSystem(config)
system.start()

# Add documents
system.add_documents([
    {'id': 'doc1', 'text': 'Neural networks process information'},
    {'id': 'doc2', 'text': 'Machine learning requires data'},
    # ... more documents ...
])

# Do searches
results = system.search("neural networks")

# Let it monitor
time.sleep(300)  # 5 minutes

system.stop()
```

---

## Checklist: Implementing Self-Healing

- [ ] **Observable**: Set up metrics collection
- [ ] **Diagnosed**: Implement anomaly detection
- [ ] **Classified**: Add crisis levels for problems
- [ ] **Repairable**: Create safe repair operations
- [ ] **Decided**: Implement repair decision logic
- [ ] **Escalatable**: Set up alert system
- [ ] **Reportable**: Generate escalation reports
- [ ] **Integrated**: Tie all components together
- [ ] **Tested**: Verify in staging environment
- [ ] **Documented**: Document escalation procedures

---

## Summary

A self-healing system follows this flow:

1. **Observe** everything (metrics, health, anomalies)
2. **Diagnose** problems early (before they cause damage)
3. **Decide** what to do (repair, adapt, escalate)
4. **Repair** what you can (safely, with rollback)
5. **Escalate** when you can't (with full context)

The key insight: **Make each decision at the lowest level possible**, but **escalate intelligently when you hit limits**.

