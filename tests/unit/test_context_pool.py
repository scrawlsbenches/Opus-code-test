"""
Unit tests for ContextPool multi-agent coordination.
"""

import unittest
import time
import tempfile
from pathlib import Path

from cortical.reasoning.context_pool import (
    ContextPool,
    ContextFinding,
    ConflictResolutionStrategy
)


class TestContextFinding(unittest.TestCase):
    """Test ContextFinding dataclass."""

    def test_creation(self):
        """Test creating a finding."""
        finding = ContextFinding(
            topic="test",
            content="test content",
            source_agent="agent_a",
            timestamp=time.time(),
            confidence=0.9,
            finding_id="test123",
            metadata={"key": "value"}
        )

        self.assertEqual(finding.topic, "test")
        self.assertEqual(finding.confidence, 0.9)
        self.assertEqual(finding.metadata["key"], "value")

    def test_immutability(self):
        """Test that findings are immutable."""
        finding = ContextFinding(
            topic="test",
            content="test",
            source_agent="agent_a",
            timestamp=time.time(),
            confidence=1.0,
            finding_id="id",
            metadata={}
        )

        with self.assertRaises(AttributeError):
            finding.content = "modified"  # type: ignore

    def test_conflicts_with_same_topic_different_content(self):
        """Test conflict detection."""
        f1 = ContextFinding(
            topic="location",
            content="File is in dir_a",
            source_agent="agent_a",
            timestamp=time.time(),
            confidence=1.0,
            finding_id="id1",
            metadata={}
        )

        f2 = ContextFinding(
            topic="location",
            content="File is in dir_b",
            source_agent="agent_b",
            timestamp=time.time(),
            confidence=1.0,
            finding_id="id2",
            metadata={}
        )

        self.assertTrue(f1.conflicts_with(f2))

    def test_no_conflict_same_agent(self):
        """Same agent can publish multiple findings without conflict."""
        f1 = ContextFinding(
            topic="location",
            content="File is in dir_a",
            source_agent="agent_a",
            timestamp=time.time(),
            confidence=1.0,
            finding_id="id1",
            metadata={}
        )

        f2 = ContextFinding(
            topic="location",
            content="File is in dir_b",
            source_agent="agent_a",  # Same agent
            timestamp=time.time(),
            confidence=1.0,
            finding_id="id2",
            metadata={}
        )

        self.assertFalse(f1.conflicts_with(f2))

    def test_no_conflict_different_topic(self):
        """Different topics don't conflict."""
        f1 = ContextFinding(
            topic="location",
            content="File is in dir_a",
            source_agent="agent_a",
            timestamp=time.time(),
            confidence=1.0,
            finding_id="id1",
            metadata={}
        )

        f2 = ContextFinding(
            topic="status",  # Different topic
            content="File is in dir_a",
            source_agent="agent_b",
            timestamp=time.time(),
            confidence=1.0,
            finding_id="id2",
            metadata={}
        )

        self.assertFalse(f1.conflicts_with(f2))

    def test_serialization(self):
        """Test to_dict and from_dict."""
        original = ContextFinding(
            topic="test",
            content="content",
            source_agent="agent",
            timestamp=123.456,
            confidence=0.8,
            finding_id="abc",
            metadata={"key": "value"}
        )

        # Serialize
        data = original.to_dict()
        self.assertIsInstance(data, dict)

        # Deserialize
        restored = ContextFinding.from_dict(data)
        self.assertEqual(restored.topic, original.topic)
        self.assertEqual(restored.confidence, original.confidence)
        self.assertEqual(restored.metadata, original.metadata)


class TestContextPool(unittest.TestCase):
    """Test ContextPool functionality."""

    def setUp(self):
        """Create fresh pool for each test."""
        self.pool = ContextPool()

    def test_publish_and_query(self):
        """Test basic publish and query."""
        finding = self.pool.publish(
            topic="test",
            content="test content",
            source_agent="agent_a",
            confidence=0.9
        )

        self.assertIsInstance(finding, ContextFinding)
        self.assertEqual(finding.topic, "test")

        results = self.pool.query("test")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].content, "test content")

    def test_query_nonexistent_topic(self):
        """Query for nonexistent topic returns empty list."""
        results = self.pool.query("nonexistent")
        self.assertEqual(results, [])

    def test_query_all(self):
        """Test querying all findings."""
        self.pool.publish("topic1", "content1", "agent_a")
        self.pool.publish("topic2", "content2", "agent_b")

        all_findings = self.pool.query_all()
        self.assertEqual(len(all_findings), 2)

    def test_multiple_findings_same_topic(self):
        """Multiple findings on same topic."""
        self.pool.publish("bugs", "Bug 1", "agent_a")
        self.pool.publish("bugs", "Bug 2", "agent_b")

        results = self.pool.query("bugs")
        self.assertEqual(len(results), 2)

    def test_confidence_validation(self):
        """Confidence must be in [0.0, 1.0]."""
        with self.assertRaises(ValueError):
            self.pool.publish("test", "content", "agent", confidence=1.5)

        with self.assertRaises(ValueError):
            self.pool.publish("test", "content", "agent", confidence=-0.1)

    def test_metadata(self):
        """Test metadata storage."""
        finding = self.pool.publish(
            topic="test",
            content="content",
            source_agent="agent",
            metadata={"task_id": "T-001", "priority": "high"}
        )

        self.assertEqual(finding.metadata["task_id"], "T-001")
        self.assertEqual(finding.metadata["priority"], "high")

    def test_get_topics(self):
        """Test getting all topics."""
        self.pool.publish("topic1", "content1", "agent_a")
        self.pool.publish("topic2", "content2", "agent_b")
        self.pool.publish("topic1", "content3", "agent_c")

        topics = self.pool.get_topics()
        self.assertEqual(set(topics), {"topic1", "topic2"})

    def test_count(self):
        """Test counting findings."""
        self.pool.publish("topic1", "content1", "agent_a")
        self.pool.publish("topic1", "content2", "agent_b")
        self.pool.publish("topic2", "content3", "agent_c")

        self.assertEqual(self.pool.count(), 3)
        self.assertEqual(self.pool.count("topic1"), 2)
        self.assertEqual(self.pool.count("topic2"), 1)
        self.assertEqual(self.pool.count("nonexistent"), 0)

    def test_clear(self):
        """Test clearing the pool."""
        self.pool.publish("test", "content", "agent")
        self.assertEqual(self.pool.count(), 1)

        self.pool.clear()
        self.assertEqual(self.pool.count(), 0)
        self.assertEqual(self.pool.query_all(), [])


class TestConflictDetection(unittest.TestCase):
    """Test conflict detection and resolution."""

    def test_manual_conflict_detection(self):
        """Test MANUAL strategy keeps both findings."""
        pool = ContextPool(conflict_strategy=ConflictResolutionStrategy.MANUAL)

        pool.publish("location", "File is in dir_a", "agent_a")
        pool.publish("location", "File is in dir_b", "agent_b")

        # Both findings kept
        self.assertEqual(pool.count("location"), 2)

        # Conflict recorded
        conflicts = pool.get_conflicts()
        self.assertEqual(len(conflicts), 1)

    def test_last_write_wins(self):
        """Test LAST_WRITE_WINS strategy."""
        pool = ContextPool(conflict_strategy=ConflictResolutionStrategy.LAST_WRITE_WINS)

        pool.publish("location", "File is in dir_a", "agent_a")
        pool.publish("location", "File is in dir_b", "agent_b")

        # Only newest finding kept
        findings = pool.query("location")
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].content, "File is in dir_b")

    def test_highest_confidence_wins(self):
        """Test HIGHEST_CONFIDENCE strategy."""
        pool = ContextPool(conflict_strategy=ConflictResolutionStrategy.HIGHEST_CONFIDENCE)

        pool.publish("perf", "Takes 100ms", "agent_a", confidence=0.9)
        pool.publish("perf", "Takes 200ms", "agent_b", confidence=0.7)

        # Higher confidence kept
        findings = pool.query("perf")
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].content, "Takes 100ms")

    def test_no_conflict_same_content(self):
        """Same content from different agents is not a conflict."""
        pool = ContextPool(conflict_strategy=ConflictResolutionStrategy.MANUAL)

        pool.publish("location", "File is in dir_a", "agent_a")
        pool.publish("location", "File is in dir_a", "agent_b")  # Same content

        conflicts = pool.get_conflicts()
        self.assertEqual(len(conflicts), 0)


class TestSubscriptions(unittest.TestCase):
    """Test subscription callbacks."""

    def test_subscribe_and_notify(self):
        """Test subscription notifications."""
        pool = ContextPool()
        received = []

        def callback(finding: ContextFinding):
            received.append(finding)

        pool.subscribe("test_topic", callback)
        pool.publish("test_topic", "test content", "agent_a")

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].content, "test content")

    def test_multiple_subscribers(self):
        """Multiple subscribers receive notifications."""
        pool = ContextPool()
        received_a = []
        received_b = []

        pool.subscribe("topic", lambda f: received_a.append(f))
        pool.subscribe("topic", lambda f: received_b.append(f))

        pool.publish("topic", "content", "agent")

        self.assertEqual(len(received_a), 1)
        self.assertEqual(len(received_b), 1)

    def test_subscribe_wrong_topic(self):
        """Subscribers only notified for their topic."""
        pool = ContextPool()
        received = []

        pool.subscribe("topic_a", lambda f: received.append(f))
        pool.publish("topic_b", "content", "agent")

        self.assertEqual(len(received), 0)


class TestTTL(unittest.TestCase):
    """Test TTL expiration."""

    def test_ttl_expiration(self):
        """Findings expire after TTL."""
        pool = ContextPool(ttl_seconds=0.5)  # 500ms TTL

        pool.publish("test", "content", "agent")
        self.assertEqual(pool.count(), 1)

        # Wait for expiration
        time.sleep(0.6)

        # Expired findings removed on query
        self.assertEqual(pool.count(), 0)
        self.assertEqual(pool.query("test"), [])

    def test_no_ttl(self):
        """Findings don't expire when ttl_seconds=None."""
        pool = ContextPool(ttl_seconds=None)

        pool.publish("test", "content", "agent")
        time.sleep(0.1)

        self.assertEqual(pool.count(), 1)

    def test_ttl_prunes_conflicts(self):
        """Expired findings removed from conflicts."""
        pool = ContextPool(
            ttl_seconds=0.5,
            conflict_strategy=ConflictResolutionStrategy.MANUAL
        )

        pool.publish("topic", "content_a", "agent_a")
        pool.publish("topic", "content_b", "agent_b")

        self.assertEqual(len(pool.get_conflicts()), 1)

        time.sleep(0.6)
        pool.query("topic")  # Trigger pruning

        self.assertEqual(len(pool.get_conflicts()), 0)


class TestPersistence(unittest.TestCase):
    """Test save/load functionality."""

    def test_save_and_load(self):
        """Test saving and loading pool state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_dir = Path(tmpdir)

            # Create and populate pool
            pool = ContextPool(storage_dir=storage_dir)
            pool.publish("topic1", "content1", "agent_a", confidence=0.9)
            pool.publish("topic2", "content2", "agent_b", confidence=0.8)

            # Save
            pool.save()

            # Load into new pool
            new_pool = ContextPool(storage_dir=storage_dir)
            new_pool.load()

            # Verify state restored
            self.assertEqual(new_pool.count(), 2)
            self.assertEqual(len(new_pool.query("topic1")), 1)
            self.assertEqual(len(new_pool.query("topic2")), 1)

    def test_load_nonexistent(self):
        """Loading from nonexistent file is a no-op."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pool = ContextPool(storage_dir=Path(tmpdir))
            pool.load()  # Should not raise

            self.assertEqual(pool.count(), 0)

    def test_save_load_conflicts(self):
        """Test saving and loading conflicts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_dir = Path(tmpdir)

            pool = ContextPool(
                storage_dir=storage_dir,
                conflict_strategy=ConflictResolutionStrategy.MANUAL
            )

            pool.publish("topic", "content_a", "agent_a")
            pool.publish("topic", "content_b", "agent_b")

            pool.save()

            new_pool = ContextPool(storage_dir=storage_dir)
            new_pool.load()

            self.assertEqual(len(new_pool.get_conflicts()), 1)

    def test_save_without_storage_dir(self):
        """Save without storage_dir raises ValueError."""
        pool = ContextPool()

        with self.assertRaises(ValueError):
            pool.save()

    def test_load_without_storage_dir(self):
        """Load without storage_dir raises ValueError."""
        pool = ContextPool()

        with self.assertRaises(ValueError):
            pool.load()


if __name__ == '__main__':
    unittest.main()
