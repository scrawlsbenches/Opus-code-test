# Tiered Locking: Performance Modeling & Benchmarks

This document provides performance models and benchmarking methodologies for tiered locking systems.

---

## Part 1: Analytical Models

### 1.1 Amdahl's Law Applied to Locking

The speedup from parallelism is limited by the sequential portion (locks):

```
Speedup(p, n) = 1 / (1 - p + p/n)

Where:
  p = Parallelizable fraction (0 to 1)
  n = Number of cores/threads
```

**Example 1: Database-level lock (p = 0.1, only 10% parallelizable)**
```
Speedup(0.1, 10) = 1 / (1 - 0.1 + 0.1/10) = 1 / 0.91 = 1.1x

Even with 10 cores, only 1.1x speedup. Lock is the bottleneck!
```

**Example 2: Hierarchical locks (p = 0.9, 90% parallelizable)**
```
Speedup(0.9, 10) = 1 / (1 - 0.9 + 0.9/10) = 1 / 0.19 = 5.3x

Much better! 10 agents can achieve ~5x speedup.
```

**Example 3: Lock-free (p = 0.99, 99% parallelizable)**
```
Speedup(0.99, 10) = 1 / (1 - 0.99 + 0.99/10) = 1 / 0.109 = 9.2x

Near-linear scaling with lock-free approach.
```

### 1.2 Lock Contention Model

Lock wait time grows exponentially with contention:

```
Wait_time = service_time × (utilization / (1 - utilization))

Where:
  service_time = Time to hold lock
  utilization = (arrival_rate × service_time)
```

**Example: Task update under database lock**
```
service_time = 100ms per task update
arrival_rate = 10 tasks/sec
utilization = 10 × 0.1 = 1.0 (100% saturated!)

Wait_time = 0.1 × (1.0 / (1 - 1.0)) = INFINITE (queue explodes)
```

**Same scenario with row-level locks**
```
Now only 1 task is locked at a time:
utilization ≈ 0.01 (1% of tasks contending)
Wait_time ≈ 0.1 × (0.01 / 0.99) = 0.001ms

Massive improvement! No queue.
```

### 1.3 Transaction Conflict Rate Model

Under optimistic locking, conflict rate depends on:

```
conflict_rate = 1 - (1 - p_overlap)^n

Where:
  p_overlap = Probability two transactions access same data
  n = Number of concurrent transactions

Example: 10 concurrent agents, each touching 10% of task dataset

p_overlap = 0.1 (10% overlap)
conflict_rate = 1 - (1 - 0.1)^10 = 1 - 0.349 = 65%

With independent tasks (1% overlap):
p_overlap = 0.01
conflict_rate = 1 - (1 - 0.01)^10 = 1 - 0.904 = 9.6%
```

---

## Part 2: Benchmarking Methodology

### 2.1 YCSB-Inspired Workload Profiles

```python
# benchmarks/ycsb_workloads.py

from enum import Enum
from dataclasses import dataclass


class WorkloadType(Enum):
    """YCSB workload profiles adapted for GoT."""

    # Workload A: 50% reads, 50% writes (typical)
    A = {'reads': 0.5, 'writes': 0.5, 'name': 'balanced'}

    # Workload B: 95% reads, 5% writes (read-heavy)
    B = {'reads': 0.95, 'writes': 0.05, 'name': 'read_heavy'}

    # Workload C: 100% reads (read-only, like queries)
    C = {'reads': 1.0, 'writes': 0.0, 'name': 'read_only'}

    # Workload D: 95% reads, 5% appends (time-series, ML logs)
    D = {'reads': 0.95, 'appends': 0.05, 'name': 'append_heavy'}

    # Workload E: 10% reads, 90% writes (write-heavy)
    E = {'reads': 0.1, 'writes': 0.9, 'name': 'write_heavy'}


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    workload: WorkloadType           # YCSB profile
    num_agents: int                  # Number of concurrent agents
    duration_seconds: int            # How long to run
    num_entities: int                # Total entities in database
    hot_entity_fraction: float       # Fraction of entities that are "hot" (frequently accessed)
    read_only: bool = False          # If True, only reads


class YCSBBench:
    """YCSB-style benchmarking for tiered locking."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = {
            'throughput': 0,           # ops/sec
            'latency_p50': 0,          # milliseconds
            'latency_p99': 0,
            'latency_max': 0,
            'conflicts': 0,            # Transaction aborts
            'deadlocks': 0,
            'lock_wait_ms': 0,         # Time spent waiting for locks
            'io_wait_ms': 0,           # Time spent in I/O
        }

    def run(self):
        """Execute benchmark."""
        import time
        import threading
        from concurrent.futures import ThreadPoolExecutor

        start_time = time.time()
        end_time = start_time + self.config.duration_seconds

        operations = []

        def agent_worker():
            """Single agent executing workload."""
            while time.time() < end_time:
                op_type = self._choose_operation()

                op_start = time.time()
                try:
                    if op_type == 'read':
                        self._do_read()
                    else:  # write
                        self._do_write()

                    op_latency = (time.time() - op_start) * 1000
                    operations.append({
                        'type': op_type,
                        'latency_ms': op_latency,
                        'success': True,
                    })

                except ConflictError:
                    op_latency = (time.time() - op_start) * 1000
                    operations.append({
                        'type': op_type,
                        'latency_ms': op_latency,
                        'success': False,
                        'error': 'conflict',
                    })

                except DeadlockError:
                    operations.append({
                        'type': op_type,
                        'success': False,
                        'error': 'deadlock',
                    })

        # Run agents in parallel
        with ThreadPoolExecutor(max_workers=self.config.num_agents) as executor:
            futures = [
                executor.submit(agent_worker)
                for _ in range(self.config.num_agents)
            ]
            for future in futures:
                future.result()

        # Analyze results
        self._analyze_operations(operations)

    def _choose_operation(self) -> str:
        """Choose operation type based on workload."""
        import random

        rand = random.random()
        workload = self.config.workload.value

        if rand < workload['reads']:
            return 'read'
        else:
            return 'write'

    def _do_read(self):
        """Execute a read operation."""
        # Implementation: read random entity
        pass

    def _do_write(self):
        """Execute a write operation."""
        # Implementation: write random entity
        pass

    def _analyze_operations(self, operations: List[Dict]):
        """Analyze benchmark results."""
        from statistics import mean, median, quantiles

        successful = [op for op in operations if op['success']]
        failed = [op for op in operations if not op['success']]

        latencies = [op['latency_ms'] for op in successful]

        self.results['throughput'] = len(successful) / self.config.duration_seconds
        self.results['latency_p50'] = median(latencies)
        self.results['latency_p99'] = quantiles(latencies, n=100)[98]
        self.results['latency_max'] = max(latencies)
        self.results['conflicts'] = len([op for op in failed if op.get('error') == 'conflict'])
        self.results['deadlocks'] = len([op for op in failed if op.get('error') == 'deadlock'])

    def print_results(self):
        """Print benchmark results in human-readable format."""
        print(f"""
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        BENCHMARK RESULTS: {self.config.workload.value['name']}
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        Configuration:
          Agents:           {self.config.num_agents}
          Duration:         {self.config.duration_seconds}s
          Entities:         {self.config.num_entities}
          Hot fraction:     {self.config.hot_entity_fraction * 100:.1f}%

        Results:
          Throughput:       {self.results['throughput']:.0f} ops/sec
          Latency p50:      {self.results['latency_p50']:.2f}ms
          Latency p99:      {self.results['latency_p99']:.2f}ms
          Latency max:      {self.results['latency_max']:.2f}ms

        Failures:
          Conflicts:        {self.results['conflicts']}
          Deadlocks:        {self.results['deadlocks']}

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """)
```

### 2.2 Comparative Benchmarking

```python
# benchmarks/compare_locking_strategies.py

class LockingStrategyBenchmark:
    """Compare different locking strategies."""

    STRATEGIES = {
        'database_lock': {
            'description': 'Single database-level lock',
            'implementation': DatabaseLockImpl,
        },
        'hierarchical_row': {
            'description': 'Hierarchical row-level locks',
            'implementation': HierarchicalLockImpl,
        },
        'optimistic': {
            'description': 'Optimistic locking + versioning',
            'implementation': OptimisticLockImpl,
        },
        'lockfree': {
            'description': 'Lock-free append-only',
            'implementation': LockFreeImpl,
        },
    }

    def benchmark_all(self, config: BenchmarkConfig):
        """Run benchmark for all locking strategies."""
        results = {}

        for strategy_name, strategy_info in self.STRATEGIES.items():
            print(f"Benchmarking {strategy_name}...")

            impl = strategy_info['implementation'](config)
            bench = YCSBBench(config)
            bench._do_read = impl.read
            bench._do_write = impl.write
            bench.run()

            results[strategy_name] = bench.results

        return self._compare_results(results)

    def _compare_results(self, results: Dict) -> pd.DataFrame:
        """Create comparison table."""
        import pandas as pd

        comparison = pd.DataFrame({
            strategy: results[strategy]
            for strategy in results
        }).T

        # Add improvement ratios
        baseline = results['database_lock']['throughput']
        comparison['speedup'] = comparison['throughput'] / baseline

        return comparison
```

---

## Part 3: Expected Benchmark Results

### Scenario 1: Balanced Workload (YCSB A)

```
┌──────────────────────────────────────────────────────────────┐
│ YCSB Workload A: 50% Reads, 50% Writes (10 agents, 1000 tasks)│
├──────────────────────────────────────────────────────────────┤
│                                                               │
│ Locking Strategy         │ Throughput  │ p99 Latency │ Speedup│
│ ─────────────────────────┼─────────────┼─────────────┼────────│
│ Database-level lock      │ 10 ops/sec  │ 95ms       │ 1.0x  │
│ Hierarchical row-lock    │ 85 ops/sec  │ 12ms       │ 8.5x  │
│ Optimistic locking       │ 65 ops/sec  │ 8ms        │ 6.5x  │
│ Lock-free (metrics only) │ 950 ops/sec │ 2ms        │ 95x   │
│                                                               │
│ Note: Lock-free is only for BEST_EFFORT tier.               │
│ Expected hybrid (all tiers): ~100 ops/sec, 15ms p99         │
└──────────────────────────────────────────────────────────────┘
```

### Scenario 2: Read-Heavy Workload (YCSB B)

```
┌──────────────────────────────────────────────────────────────┐
│ YCSB Workload B: 95% Reads, 5% Writes (10 agents, 1000 tasks)│
├──────────────────────────────────────────────────────────────┤
│                                                               │
│ Locking Strategy         │ Throughput  │ p99 Latency │ Speedup│
│ ─────────────────────────┼─────────────┼─────────────┼────────│
│ Database-level lock      │ 50 ops/sec  │ 20ms       │ 1.0x  │
│ Hierarchical row-lock    │ 400 ops/sec │ 3ms        │ 8.0x  │
│ Optimistic locking       │ 450 ops/sec │ 2ms        │ 9.0x  │
│ Lock-free (read-only)    │ 950 ops/sec │ 1ms        │ 19x   │
│                                                               │
│ Note: Optimistic excels in read-heavy workloads.            │
│ Only 5% writes means few conflicts.                         │
└──────────────────────────────────────────────────────────────┘
```

### Scenario 3: Write-Heavy Workload (YCSB E)

```
┌──────────────────────────────────────────────────────────────┐
│ YCSB Workload E: 10% Reads, 90% Writes (10 agents, 1000 tasks)│
├──────────────────────────────────────────────────────────────┤
│                                                               │
│ Locking Strategy         │ Throughput  │ p99 Latency │ Speedup│
│ ─────────────────────────┼─────────────┼─────────────┼────────│
│ Database-level lock      │ 11 ops/sec  │ 85ms       │ 1.0x  │
│ Hierarchical row-lock    │ 85 ops/sec  │ 12ms       │ 7.7x  │
│ Optimistic locking       │ 15 ops/sec  │ 60ms       │ 1.4x  │
│ Lock-free (metrics only) │ 950 ops/sec │ 2ms        │ 86x   │
│                                                               │
│ Note: Optimistic locking performs poorly (high conflicts).  │
│ Pessimistic (row-lock) is better for writes.               │
└──────────────────────────────────────────────────────────────┘
```

### Scenario 4: ML Metric Logging (YCSB D)

```
┌──────────────────────────────────────────────────────────────┐
│ YCSB Workload D: 95% Appends (Lock-Free Metrics, 100 agents) │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│ Locking Strategy         │ Throughput    │ p99 Latency│ Notes │
│ ─────────────────────────┼───────────────┼──────────────────────┤
│ Database-level lock      │ 95 metrics/s  │ 1050ms     │ SERIAL│
│ CALI lock-free storage   │ 95K metrics/s │ 0.1ms      │ IDEAL │
│                                                               │
│ Improvement: 1000x! (from 95 to 95,000 per second)          │
└──────────────────────────────────────────────────────────────┘
```

---

## Part 4: Lock Contention Visualization

### 4.1 Lock Wait Time Over Time

```
Database-level lock (contention model):
┌────────────────────────────────────────────────────────────┐
│ Lock Wait Time (ms)                                         │
│ ▲                                                           │
│ │                                        ╱╱╱╱╱╱╱╱╱╱╱╱     │
│ │                                    ╱╱╱╱╱        (SATURATED)│
│ │                                ╱╱╱╱╱                      │
│ │  1000 ├─────────────────────╱╱╱╱╱                       │
│ │       │               ╱╱╱╱╱                              │
│ │       │           ╱╱╱╱╱                                  │
│ │   100 ├───────╱╱╱╱╱                                     │
│ │       │   ╱╱╱╱╱                                          │
│ │    10 ├──╱╱╱                                            │
│ │       │╱╱                                                │
│ │     1 └───────────────────────────────────────────────  │
│ │  0 │                                                      │
│ └────┴──────┬──────┬──────┬──────┬──────┬──────┬──────────│
│             1      2      3      4      5      6      10 ops/sec
│                        Arrival Rate
│
│ At 3 ops/sec, wait time starts climbing exponentially
│ At 10 ops/sec, queue is infinite (system collapse)
└────────────────────────────────────────────────────────────┘


Hierarchical locks (row-level):
┌────────────────────────────────────────────────────────────┐
│ Lock Wait Time (ms)                                         │
│ ▲                                                           │
│ │   10 ├─────────────────────────────────────────────────│
│ │      │                                                   │
│ │    1 ├─────────                                        │
│ │      │           ──────                                  │
│ │  0.1 ├─────────────────────────────────────────────────│
│ │      │                                                   │
│ │    0 └───────────────────────────────────────────────  │
│ │  0 │                                                      │
│ └────┴──────┬──────┬──────┬──────┬──────┬──────┬──────────│
│             10     20     30     40     50     60     100 ops/sec
│                        Arrival Rate
│
│ Linear scaling, no collapse point
│ At 100 ops/sec, wait time still < 0.5ms
└────────────────────────────────────────────────────────────┘
```

---

## Part 5: Real-World Measurement

### 5.1 Metrics to Track

```python
# cortical/got/benchmark_metrics.py

class BenchmarkMetrics:
    """Track metrics for performance analysis."""

    def __init__(self):
        self.operations = []
        self.lock_acquisitions = []

    def record_operation(
        self,
        op_type: str,
        entity_id: str,
        start_time: float,
        end_time: float,
        success: bool,
        error: Optional[str] = None,
    ):
        """Record single operation."""
        self.operations.append({
            'type': op_type,
            'entity_id': entity_id,
            'duration_ms': (end_time - start_time) * 1000,
            'success': success,
            'error': error,
            'timestamp': start_time,
        })

    def record_lock_acquisition(
        self,
        lock_level: str,
        entity_id: str,
        wait_time_ms: float,
        hold_time_ms: float,
    ):
        """Record lock timing."""
        self.lock_acquisitions.append({
            'level': lock_level,
            'entity_id': entity_id,
            'wait_ms': wait_time_ms,
            'hold_ms': hold_time_ms,
            'total_ms': wait_time_ms + hold_time_ms,
        })

    def print_summary(self):
        """Print summary statistics."""
        import statistics as stats

        if not self.operations:
            print("No operations recorded")
            return

        durations = [op['duration_ms'] for op in self.operations if op['success']]
        successful = len([op for op in self.operations if op['success']])
        failed = len([op for op in self.operations if not op['success']])

        print(f"""
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        OPERATION METRICS
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        Total operations:  {len(self.operations)}
        Successful:        {successful}
        Failed:            {failed}
        Success rate:      {successful / len(self.operations) * 100:.1f}%

        Latency (successful ops):
          Mean:            {stats.mean(durations):.2f}ms
          Median:          {stats.median(durations):.2f}ms
          Stdev:           {stats.stdev(durations):.2f}ms
          Min:             {min(durations):.2f}ms
          Max:             {max(durations):.2f}ms

        Lock Metrics:
          Total lock acquis: {len(self.lock_acquisitions)}
          Avg wait time:     {stats.mean([l['wait_ms'] for l in self.lock_acquisitions]):.2f}ms
          Avg hold time:     {stats.mean([l['hold_ms'] for l in self.lock_acquisitions]):.2f}ms
        """)
```

### 5.2 Instrumentation Points

Add timing measurements at critical points:

```python
# cortical/got/instrumented_transaction.py

class InstrumentedTransaction:
    """Transaction with built-in performance instrumentation."""

    def __init__(self, metrics: BenchmarkMetrics):
        self.metrics = metrics
        self.start_time = time.time()

    def write(self, entity_id: str, field: str, value: any):
        """Write with instrumentation."""
        lock_start = time.time()

        # Acquire lock
        lock_wait = time.time() - lock_start

        # Execute write
        write_start = time.time()
        self._do_write(entity_id, field, value)
        write_time = time.time() - write_start

        # Record metrics
        self.metrics.record_lock_acquisition(
            lock_level='record_id',
            entity_id=entity_id,
            wait_time_ms=lock_wait * 1000,
            hold_time_ms=write_time * 1000,
        )

    def commit(self) -> bool:
        """Commit with instrumentation."""
        commit_start = time.time()

        try:
            success = self._do_commit()

            self.metrics.record_operation(
                op_type='transaction',
                entity_id='multiple',
                start_time=self.start_time,
                end_time=time.time(),
                success=success,
            )

            return success

        except Exception as e:
            self.metrics.record_operation(
                op_type='transaction',
                entity_id='multiple',
                start_time=self.start_time,
                end_time=time.time(),
                success=False,
                error=str(e),
            )
            raise
```

---

## Part 6: Running Benchmarks

### 6.1 Benchmark Script

```python
# benchmarks/run_benchmarks.py

#!/usr/bin/env python3

import argparse
from pathlib import Path
from benchmarks.ycsb_workloads import WorkloadType, BenchmarkConfig, YCSBBench
from benchmarks.compare_locking_strategies import LockingStrategyBenchmark


def main():
    parser = argparse.ArgumentParser(description='Run tiered locking benchmarks')
    parser.add_argument(
        '--workload',
        choices=['a', 'b', 'c', 'd', 'e'],
        default='a',
        help='YCSB workload (default: A)'
    )
    parser.add_argument(
        '--agents',
        type=int,
        default=10,
        help='Number of concurrent agents (default: 10)'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=60,
        help='Benchmark duration in seconds (default: 60)'
    )
    parser.add_argument(
        '--entities',
        type=int,
        default=1000,
        help='Number of entities in database (default: 1000)'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare all locking strategies'
    )

    args = parser.parse_args()

    # Map workload letter to type
    workload_map = {
        'a': WorkloadType.A,
        'b': WorkloadType.B,
        'c': WorkloadType.C,
        'd': WorkloadType.D,
        'e': WorkloadType.E,
    }

    config = BenchmarkConfig(
        workload=workload_map[args.workload],
        num_agents=args.agents,
        duration_seconds=args.duration,
        num_entities=args.entities,
        hot_entity_fraction=0.1,  # 10% hot entities
    )

    if args.compare:
        print("Comparing locking strategies...")
        bench = LockingStrategyBenchmark()
        results = bench.benchmark_all(config)
        print(results)
    else:
        print(f"Running YCSB Workload {args.workload.upper()}...")
        bench = YCSBBench(config)
        bench.run()
        bench.print_results()


if __name__ == '__main__':
    main()
```

**Usage:**

```bash
# Benchmark with 10 agents, 60 seconds
python benchmarks/run_benchmarks.py --agents 10 --duration 60

# Compare all strategies
python benchmarks/run_benchmarks.py --compare --agents 10

# Run workload E (write-heavy) with 50 agents
python benchmarks/run_benchmarks.py --workload e --agents 50
```

---

## Part 7: Performance Targets

After implementing tiered locking, you should expect:

```
┌────────────────────────────────────────────────────────────┐
│        TIERED LOCKING PERFORMANCE TARGETS                  │
├────────────────────────────────────────────────────────────┤
│                                                             │
│ Metric                  │ Before      │ After    │ Gain    │
│ ────────────────────────┼─────────────┼──────────┼─────────│
│ Single task update      │ 500ms       │ 50ms     │ 10x    │
│ Latency p99             │             │          │         │
│                                                             │
│ 10 concurrent updates   │ 5 sec       │ 500ms    │ 10x    │
│ (serialized)            │             │          │         │
│                                                             │
│ 100 concurrent metrics  │ >10 sec     │ 100ms    │ 100x   │
│ logs                    │ (blocked!)  │ (parallel)          │
│                                                             │
│ Transaction abort rate  │ 5%          │ <1%      │ 5x     │
│ (due to conflicts)      │             │          │         │
│                                                             │
│ Lock contention (% time)│ 80%         │ 10%      │ 8x     │
│ waiting for locks       │             │          │         │
│                                                             │
│ Query throughput        │ 100 qps     │ 500 qps  │ 5x    │
│ (find tasks)            │             │          │         │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

---

## Conclusion

Tiered locking systems can achieve 10-100x improvements in throughput and latency, depending on the workload and locking strategy. The key is to:

1. **Use hierarchical locks** to reduce contention (10x improvement)
2. **Apply optimistic locking to read-heavy workloads** (better for reads)
3. **Use pessimistic locking for write-heavy workloads** (avoids retries)
4. **Implement lock-free storage for best-effort data** (unbounded throughput)
5. **Measure and profile** to identify bottlenecks

Expected improvement from tiered locking: **10-15x throughput increase** on typical mixed workloads.

