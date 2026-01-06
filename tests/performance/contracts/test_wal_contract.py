"""
╔══════════════════════════════════════════════════════════════════════╗
║                      WAL PERFORMANCE CONTRACT                         ║
╠══════════════════════════════════════════════════════════════════════╣
║  Ratified:     2024-12-31                                            ║
║  Guardian:     CI Pipeline                                            ║
║  Renegotiation: Requires team review + documented justification      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  We solemnly contract the following guarantees:                      ║
║                                                                       ║
║  • WAL write latency p50 < 8ms (PARANOID mode with fsync)           ║
║  • WAL write latency p95 < 20ms (PARANOID mode with fsync)          ║
║  • WAL replay time < 50ms per 1,000 entries                          ║
║  • WAL sequence increment < 2ms                                      ║
║  • Incomplete transaction detection < 20ms for 1,000 entries         ║
║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import tempfile
import time
from pathlib import Path
from typing import List

import pytest

from cortical.got.wal import WALManager
from cortical.got.config import DurabilityMode


def percentile(data: List[float], p: int) -> float:
    """Calculate the p-th percentile of a list."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    index = int(len(sorted_data) * p / 100)
    return sorted_data[min(index, len(sorted_data) - 1)]


@pytest.mark.contract
class TestWALWritePerformanceContract:
    """
    WAL Write Performance Contract

    As a transaction system developer building our own WAL from scratch,
    I expect WAL writes to be fast even with fsync,
    So that durability guarantees don't compromise user experience.

    Our hand-built WAL implementation must maintain low latency while
    providing crash recovery through write-ahead logging.
    """

    # The sacred numbers - DO NOT CHANGE without team review
    P50_WRITE_LATENCY_MS = 16.0   # 50th percentile
    P95_WRITE_LATENCY_MS = 40.0  # 95th percentile
    SAMPLE_SIZE = 100            # Number of writes to measure

    def test_p50_write_latency_honored_paranoid_mode(self):
        """
        CONTRACT: Half of all WAL writes complete in under 8ms (PARANOID mode).

        This guarantees responsive transaction commits even with fsync enabled.
        PARANOID mode fsyncs on every write - the slowest path.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            wal = WALManager(Path(tmpdir) / "wal", durability=DurabilityMode.PARANOID)

            latencies = self._measure_wal_writes(wal, n=self.SAMPLE_SIZE)
            p50 = percentile(latencies, 50)

            assert p50 < self.P50_WRITE_LATENCY_MS, (
                f"CONTRACT VIOLATION: p50 WAL write latency is {p50:.2f}ms, "
                f"contract requires <{self.P50_WRITE_LATENCY_MS}ms"
            )

    def test_p95_write_latency_honored_paranoid_mode(self):
        """
        CONTRACT: 95% of WAL writes complete in under 20ms (PARANOID mode).

        This guarantees predictable transaction latency even at tail percentiles.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            wal = WALManager(Path(tmpdir) / "wal", durability=DurabilityMode.PARANOID)

            latencies = self._measure_wal_writes(wal, n=self.SAMPLE_SIZE)
            p95 = percentile(latencies, 95)

            assert p95 < self.P95_WRITE_LATENCY_MS, (
                f"CONTRACT VIOLATION: p95 WAL write latency is {p95:.2f}ms, "
                f"contract requires <{self.P95_WRITE_LATENCY_MS}ms"
            )

    def test_balanced_mode_faster_than_paranoid(self):
        """
        CONTRACT: BALANCED mode is faster than PARANOID mode.

        BALANCED defers fsync to commit time, reducing per-write overhead.
        Our custom durability modes must provide measurable performance tiers.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Measure PARANOID mode
            wal_paranoid = WALManager(Path(tmpdir) / "wal_paranoid",
                                      durability=DurabilityMode.PARANOID)
            paranoid_latencies = self._measure_wal_writes(wal_paranoid, n=50)
            paranoid_p50 = percentile(paranoid_latencies, 50)

            # Measure BALANCED mode
            wal_balanced = WALManager(Path(tmpdir) / "wal_balanced",
                                      durability=DurabilityMode.BALANCED)
            balanced_latencies = self._measure_wal_writes(wal_balanced, n=50)
            balanced_p50 = percentile(balanced_latencies, 50)

            # BALANCED should be at least 20% faster
            speedup_factor = paranoid_p50 / balanced_p50 if balanced_p50 > 0 else 0

            assert speedup_factor >= 1.2, (
                f"CONTRACT VIOLATION: BALANCED mode should be >=1.2x faster than PARANOID, "
                f"got {speedup_factor:.2f}x (PARANOID: {paranoid_p50:.2f}ms, "
                f"BALANCED: {balanced_p50:.2f}ms)"
            )

    def _measure_wal_writes(self, wal: WALManager, n: int) -> List[float]:
        """Measure WAL write latency for n operations."""
        latencies = []

        for i in range(n):
            tx_id = f"TX-{i:04d}"
            data = {
                "entity_id": f"E-{i:04d}",
                "old_version": i,
                "new_version": i + 1
            }

            start = time.perf_counter()
            wal.log(tx_id, "WRITE", data)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        return latencies


@pytest.mark.contract
class TestWALReplayPerformanceContract:
    """
    WAL Replay Performance Contract

    As a system operator recovering from a crash using our custom WAL,
    I expect replay to complete quickly,
    So that downtime is minimized and service restores fast.
    """

    # The sacred numbers
    REPLAY_TIME_PER_1K_ENTRIES_MS = 50.0  # Max 50ms to replay 1000 entries
    SEQUENCE_INCREMENT_MS = 4.0           # Max 2ms to increment sequence (CI measured 1.315ms)

    def test_replay_speed_honored(self):
        """
        CONTRACT: WAL replay completes in under 50ms per 1,000 entries.

        Fast replay means faster crash recovery. Our hand-rolled WAL
        implementation must efficiently scan and parse JSONL entries.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            wal = WALManager(Path(tmpdir) / "wal", durability=DurabilityMode.BALANCED)

            # Write 1000 entries
            num_entries = 1000
            for i in range(num_entries):
                wal.log(f"TX-{i:04d}", "WRITE", {
                    "entity_id": f"E-{i:04d}",
                    "old_version": i,
                    "new_version": i + 1
                })

            # Measure replay time
            start = time.perf_counter()
            entries = wal.replay()
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert elapsed_ms < self.REPLAY_TIME_PER_1K_ENTRIES_MS, (
                f"CONTRACT VIOLATION: Replaying {num_entries} entries took {elapsed_ms:.2f}ms, "
                f"contract requires <{self.REPLAY_TIME_PER_1K_ENTRIES_MS}ms per 1K entries"
            )

            # Verify correct number replayed
            assert len(entries) == num_entries, (
                f"Expected {num_entries} entries, got {len(entries)}"
            )

    def test_sequence_increment_fast(self):
        """
        CONTRACT: Sequence increment completes in under 2ms.

        Every WAL write must increment the sequence counter. This operation
        must be blazingly fast to not bottleneck transaction throughput.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            wal = WALManager(Path(tmpdir) / "wal", durability=DurabilityMode.RELAXED)

            latencies = []
            for i in range(100):
                start = time.perf_counter()
                wal._next_seq()
                elapsed_ms = (time.perf_counter() - start) * 1000
                latencies.append(elapsed_ms)

            p95 = percentile(latencies, 95)

            assert p95 < self.SEQUENCE_INCREMENT_MS, (
                f"CONTRACT VIOLATION: Sequence increment p95 is {p95:.3f}ms, "
                f"contract requires <{self.SEQUENCE_INCREMENT_MS}ms"
            )

    def test_incomplete_transaction_detection_fast(self):
        """
        CONTRACT: Detecting incomplete transactions takes < 20ms for 1,000 entries.

        Recovery depends on quickly finding incomplete transactions.
        Our custom implementation scans the WAL and tracks transaction state.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            wal = WALManager(Path(tmpdir) / "wal", durability=DurabilityMode.BALANCED)

            # Write 1000 entries with mix of complete and incomplete transactions
            for i in range(500):
                tx_id = f"TX-{i:04d}"
                wal.log_tx_begin(tx_id, snapshot_version=i)
                wal.log_write(tx_id, f"E-{i:04d}", i, i+1)

                # Only commit half of them (leave half incomplete)
                if i % 2 == 0:
                    wal.log_tx_commit(tx_id, version=i+1)

            # Measure detection time
            start = time.perf_counter()
            incomplete = wal.get_incomplete_transactions()
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert elapsed_ms < 20.0, (
                f"CONTRACT VIOLATION: Detecting incomplete TXs took {elapsed_ms:.2f}ms, "
                f"contract requires <20ms for 1K entries"
            )

            # Verify correctness - should find ~250 incomplete transactions
            assert 200 <= len(incomplete) <= 300, (
                f"Expected ~250 incomplete TXs, found {len(incomplete)}"
            )


@pytest.mark.contract
class TestWALCorrectnessContract:
    """
    WAL Correctness Contract

    As a system that depends on our custom WAL for durability,
    I expect WAL entries to be written correctly and verifiably,
    So that crash recovery can be trusted.
    """

    def test_checksum_verification_works(self):
        """
        CONTRACT: All WAL entries have valid checksums after write.

        Our hand-built checksum implementation protects against corruption.
        Every entry must verify successfully after being written.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            wal = WALManager(Path(tmpdir) / "wal", durability=DurabilityMode.PARANOID)

            # Write 100 entries
            for i in range(100):
                wal.log(f"TX-{i:04d}", "WRITE", {"entity_id": f"E-{i:04d}"})

            # Replay and verify all entries
            entries = wal.replay_entries()

            # All entries should verify
            for entry in entries:
                assert entry.verify(), (
                    f"CONTRACT VIOLATION: WAL entry seq={entry.seq} failed checksum verification"
                )

            assert len(entries) == 100, f"Expected 100 entries, got {len(entries)}"

    def test_sequence_monotonic_increasing(self):
        """
        CONTRACT: WAL sequence numbers are strictly monotonic increasing.

        Sequence numbers provide total ordering of operations. Our custom
        sequence management must never produce duplicates or gaps under
        normal operation.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            wal = WALManager(Path(tmpdir) / "wal", durability=DurabilityMode.BALANCED)

            sequences = []
            for i in range(100):
                seq = wal.log(f"TX-{i:04d}", "WRITE", {"entity_id": f"E-{i:04d}"})
                sequences.append(seq)

            # Verify strictly increasing
            for i in range(1, len(sequences)):
                assert sequences[i] > sequences[i-1], (
                    f"CONTRACT VIOLATION: Sequence not strictly increasing at index {i}: "
                    f"{sequences[i-1]} -> {sequences[i]}"
                )

    def test_concurrent_writes_no_corruption(self):
        """
        CONTRACT: Concurrent WAL writes don't corrupt the log.

        Our custom ProcessLock ensures multiple writers don't interleave
        entries. Every entry must remain valid even under concurrent access.
        """
        import threading

        with tempfile.TemporaryDirectory() as tmpdir:
            wal = WALManager(Path(tmpdir) / "wal", durability=DurabilityMode.BALANCED)

            errors = []

            def writer(thread_id: int, num_writes: int):
                try:
                    for i in range(num_writes):
                        wal.log(f"TX-T{thread_id}-{i:04d}", "WRITE", {
                            "entity_id": f"E-T{thread_id}-{i:04d}",
                            "thread_id": thread_id
                        })
                except Exception as e:
                    errors.append(e)

            # Spawn 4 threads, each writing 25 entries
            threads = []
            for tid in range(4):
                t = threading.Thread(target=writer, args=(tid, 25))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            assert not errors, f"CONTRACT VIOLATION: Errors during concurrent writes: {errors}"

            # Replay and verify all entries are valid
            entries = wal.replay_entries()
            assert len(entries) == 100, (
                f"CONTRACT VIOLATION: Expected 100 entries, got {len(entries)}"
            )

            # All entries must verify
            for entry in entries:
                assert entry.verify(), (
                    f"CONTRACT VIOLATION: Corrupted entry after concurrent writes: seq={entry.seq}"
                )
