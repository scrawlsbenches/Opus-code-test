"""
Unit tests for the inter-agent pub/sub messaging system.

Tests cover:
- Basic publish/subscribe operations
- Topic pattern matching with wildcards
- Message priority queues
- Message expiration (TTL)
- At-least-once delivery with acknowledgments
- Dead letter queue handling
- Subscription filtering
- Message persistence
- Statistics and inspection
"""

import json
import os
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from cortical.reasoning.pubsub import (
    Message,
    MessageStatus,
    PubSubBroker,
    Subscription,
    create_payload_filter,
    create_topic_filter,
)


class TestMessage(unittest.TestCase):
    """Test Message dataclass functionality."""

    def test_message_creation(self):
        """Test creating a message with all fields."""
        msg = Message(
            id="msg-123",
            topic="agent.task.completed",
            payload={"result": "success"},
            sender="worker1",
            timestamp=datetime.now(),
            ttl_seconds=60,
            priority=5,
        )
        self.assertEqual(msg.id, "msg-123")
        self.assertEqual(msg.topic, "agent.task.completed")
        self.assertEqual(msg.priority, 5)
        self.assertEqual(msg.status, MessageStatus.PENDING)

    def test_message_expiration(self):
        """Test message TTL expiration."""
        # Non-expiring message
        msg = Message(
            id="msg-1",
            topic="test",
            payload={},
            sender="agent1",
            timestamp=datetime.now(),
            ttl_seconds=None,
        )
        self.assertFalse(msg.is_expired())

        # Expired message
        msg_expired = Message(
            id="msg-2",
            topic="test",
            payload={},
            sender="agent1",
            timestamp=datetime.now() - timedelta(seconds=10),
            ttl_seconds=5,
        )
        self.assertTrue(msg_expired.is_expired())

        # Not yet expired
        msg_valid = Message(
            id="msg-3",
            topic="test",
            payload={},
            sender="agent1",
            timestamp=datetime.now(),
            ttl_seconds=60,
        )
        self.assertFalse(msg_valid.is_expired())

    def test_message_serialization(self):
        """Test message serialization and deserialization."""
        original = Message(
            id="msg-123",
            topic="test.topic",
            payload={"key": "value"},
            sender="agent1",
            timestamp=datetime.now(),
            ttl_seconds=300,
            priority=7,
        )

        # Serialize
        data = original.to_dict()
        self.assertIsInstance(data['timestamp'], str)
        self.assertEqual(data['status'], 'PENDING')

        # Deserialize
        restored = Message.from_dict(data)
        self.assertEqual(restored.id, original.id)
        self.assertEqual(restored.topic, original.topic)
        self.assertEqual(restored.payload, original.payload)
        self.assertEqual(restored.sender, original.sender)
        self.assertEqual(restored.priority, original.priority)
        self.assertEqual(restored.status, original.status)


class TestSubscription(unittest.TestCase):
    """Test Subscription dataclass functionality."""

    def test_subscription_creation(self):
        """Test creating a subscription."""
        sub = Subscription(
            id="sub-123",
            subscriber_id="worker1",
            topic_pattern="agent.*",
        )
        self.assertEqual(sub.id, "sub-123")
        self.assertEqual(sub.subscriber_id, "worker1")
        self.assertEqual(sub.topic_pattern, "agent.*")
        self.assertIsNone(sub.filter_fn)

    def test_topic_pattern_matching(self):
        """Test topic pattern matching with wildcards."""
        # Wildcard at end
        sub = Subscription(
            id="sub-1",
            subscriber_id="worker1",
            topic_pattern="agent.*",
        )
        self.assertTrue(sub.matches("agent.task"))
        self.assertTrue(sub.matches("agent.completed"))
        self.assertFalse(sub.matches("task.agent"))
        self.assertFalse(sub.matches("agent.task.error"))  # Single * doesn't match multiple segments

        # Wildcard at start
        sub2 = Subscription(
            id="sub-2",
            subscriber_id="worker2",
            topic_pattern="*.completed",
        )
        self.assertTrue(sub2.matches("task.completed"))
        self.assertTrue(sub2.matches("agent.completed"))
        self.assertFalse(sub2.matches("task.error"))

        # Multiple wildcards
        sub3 = Subscription(
            id="sub-3",
            subscriber_id="worker3",
            topic_pattern="agent.*.error",
        )
        self.assertTrue(sub3.matches("agent.task.error"))
        self.assertTrue(sub3.matches("agent.workflow.error"))
        self.assertFalse(sub3.matches("agent.error"))
        self.assertFalse(sub3.matches("task.workflow.error"))

        # Match all
        sub4 = Subscription(
            id="sub-4",
            subscriber_id="monitor",
            topic_pattern="*",
        )
        self.assertTrue(sub4.matches("anything"))
        self.assertFalse(sub4.matches("dotted.topic"))  # Single * doesn't match dots

    def test_subscription_filtering(self):
        """Test subscription message filtering."""
        msg = Message(
            id="msg-1",
            topic="agent.task.completed",
            payload={"priority": 10},
            sender="worker1",
            timestamp=datetime.now(),
        )

        # No filter - should receive
        sub = Subscription(
            id="sub-1",
            subscriber_id="monitor",
            topic_pattern="agent.*",
        )
        self.assertFalse(sub.should_receive(msg))  # Topic doesn't match pattern

        sub2 = Subscription(
            id="sub-2",
            subscriber_id="monitor",
            topic_pattern="agent.task.completed",
        )
        self.assertTrue(sub2.should_receive(msg))

        # With filter
        def high_priority_filter(m: Message) -> bool:
            return m.payload.get("priority", 0) > 5

        sub3 = Subscription(
            id="sub-3",
            subscriber_id="urgent",
            topic_pattern="agent.task.completed",
            filter_fn=high_priority_filter,
        )
        self.assertTrue(sub3.should_receive(msg))

        msg_low = Message(
            id="msg-2",
            topic="agent.task.completed",
            payload={"priority": 3},
            sender="worker1",
            timestamp=datetime.now(),
        )
        self.assertFalse(sub3.should_receive(msg_low))


class TestPubSubBroker(unittest.TestCase):
    """Test PubSubBroker core functionality."""

    def setUp(self):
        """Create a fresh broker for each test."""
        self.broker = PubSubBroker()

    def test_broker_initialization(self):
        """Test broker initialization."""
        self.assertEqual(len(self.broker.messages), 0)
        self.assertEqual(len(self.broker.subscriptions), 0)
        self.assertEqual(self.broker.stats['messages_published'], 0)

    def test_publish_message(self):
        """Test publishing a message."""
        msg_id = self.broker.publish(
            topic="test.topic",
            payload={"data": "value"},
            sender="publisher1",
        )
        self.assertIsNotNone(msg_id)
        self.assertIn(msg_id, self.broker.messages)
        self.assertEqual(self.broker.stats['messages_published'], 1)

    def test_publish_with_priority(self):
        """Test publishing with different priorities."""
        msg1 = self.broker.publish("test", {}, "sender", priority=0)
        msg2 = self.broker.publish("test", {}, "sender", priority=10)
        msg3 = self.broker.publish("test", {}, "sender", priority=5)

        self.assertEqual(self.broker.messages[msg1].priority, 0)
        self.assertEqual(self.broker.messages[msg2].priority, 10)
        self.assertEqual(self.broker.messages[msg3].priority, 5)

    def test_publish_invalid_priority(self):
        """Test that invalid priorities raise ValueError."""
        with self.assertRaises(ValueError):
            self.broker.publish("test", {}, "sender", priority=-1)

        with self.assertRaises(ValueError):
            self.broker.publish("test", {}, "sender", priority=11)

    def test_subscribe_and_poll(self):
        """Test basic subscribe and poll flow."""
        # Subscribe first
        sub = self.broker.subscribe("test.*", "subscriber1")
        self.assertIsNotNone(sub)
        self.assertEqual(sub.subscriber_id, "subscriber1")

        # Publish message
        msg_id = self.broker.publish("test.message", {"data": "hello"}, "publisher1")

        # Poll for messages
        messages = self.broker.poll("subscriber1")
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].id, msg_id)
        self.assertEqual(messages[0].payload["data"], "hello")

    def test_subscribe_after_publish(self):
        """Test that subscribing after publishing delivers existing messages."""
        # Publish first
        msg_id = self.broker.publish("test.topic", {"data": "value"}, "publisher1")

        # Subscribe after
        self.broker.subscribe("test.*", "subscriber1")

        # Should receive the message
        messages = self.broker.poll("subscriber1")
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].id, msg_id)

    def test_wildcard_subscription(self):
        """Test wildcard pattern matching in subscriptions."""
        self.broker.subscribe("agent.*", "subscriber1")

        self.broker.publish("agent.task", {}, "publisher1")
        self.broker.publish("agent.completed", {}, "publisher1")
        self.broker.publish("task.agent", {}, "publisher1")  # Doesn't match

        messages = self.broker.poll("subscriber1")
        self.assertEqual(len(messages), 2)

    def test_multiple_subscribers(self):
        """Test that multiple subscribers receive the same message."""
        self.broker.subscribe("test.topic", "subscriber1")
        self.broker.subscribe("test.topic", "subscriber2")

        msg_id = self.broker.publish("test.topic", {"data": "shared"}, "publisher1")

        msgs1 = self.broker.poll("subscriber1")
        msgs2 = self.broker.poll("subscriber2")

        self.assertEqual(len(msgs1), 1)
        self.assertEqual(len(msgs2), 1)
        self.assertEqual(msgs1[0].id, msg_id)
        self.assertEqual(msgs2[0].id, msg_id)

    def test_message_priority_ordering(self):
        """Test that higher priority messages are delivered first."""
        self.broker.subscribe("test", "subscriber1")

        # Publish in random priority order
        self.broker.publish("test", {"order": 3}, "sender", priority=5)
        self.broker.publish("test", {"order": 1}, "sender", priority=10)
        self.broker.publish("test", {"order": 2}, "sender", priority=7)
        self.broker.publish("test", {"order": 4}, "sender", priority=0)

        messages = self.broker.poll("subscriber1", max_messages=10)

        # Should be ordered by priority (highest first)
        self.assertEqual(messages[0].payload["order"], 1)  # priority 10
        self.assertEqual(messages[1].payload["order"], 2)  # priority 7
        self.assertEqual(messages[2].payload["order"], 3)  # priority 5
        self.assertEqual(messages[3].payload["order"], 4)  # priority 0

    def test_message_acknowledgment(self):
        """Test message acknowledgment flow."""
        self.broker.subscribe("test", "subscriber1")
        msg_id = self.broker.publish("test", {}, "sender")

        # Poll message
        messages = self.broker.poll("subscriber1")
        self.assertEqual(len(messages), 1)

        # Message should be delivered
        msg = self.broker.messages[msg_id]
        self.assertEqual(msg.status, MessageStatus.DELIVERED)

        # Acknowledge
        ack_result = self.broker.acknowledge(msg_id, "subscriber1")
        self.assertTrue(ack_result)

        # Message should be acknowledged
        self.assertEqual(msg.status, MessageStatus.ACKNOWLEDGED)
        self.assertEqual(self.broker.stats['messages_acknowledged'], 1)

    def test_unacknowledged_redelivery(self):
        """Test that unacknowledged messages are redelivered."""
        self.broker.subscribe("test", "subscriber1")
        msg_id = self.broker.publish("test", {}, "sender")

        # First poll
        messages1 = self.broker.poll("subscriber1")
        self.assertEqual(len(messages1), 1)

        # Don't acknowledge - poll again
        messages2 = self.broker.poll("subscriber1")
        self.assertEqual(len(messages2), 1)
        self.assertEqual(messages2[0].id, msg_id)  # Same message redelivered

    def test_message_expiration(self):
        """Test that expired messages are cleaned up."""
        self.broker.subscribe("test", "subscriber1")

        # Publish message with very short TTL
        msg_id = self.broker.publish("test", {}, "sender", ttl_seconds=0)

        # Wait a moment
        time.sleep(0.1)

        # Poll should get no messages (expired)
        messages = self.broker.poll("subscriber1")
        self.assertEqual(len(messages), 0)

        # Message should be in dead letter queue
        self.assertIn(msg_id, self.broker.dead_letters)
        self.assertEqual(self.broker.stats['messages_expired'], 1)

    def test_unsubscribe(self):
        """Test unsubscribing from topics."""
        sub = self.broker.subscribe("test", "subscriber1")
        self.assertEqual(len(self.broker.subscriptions), 1)

        # Unsubscribe
        result = self.broker.unsubscribe(sub.id)
        self.assertTrue(result)
        self.assertEqual(len(self.broker.subscriptions), 0)

        # Publish message
        self.broker.publish("test", {}, "sender")

        # Should not receive
        messages = self.broker.poll("subscriber1")
        self.assertEqual(len(messages), 0)

    def test_unsubscribe_nonexistent(self):
        """Test unsubscribing from non-existent subscription."""
        result = self.broker.unsubscribe("fake-sub-id")
        self.assertFalse(result)

    def test_get_subscriptions(self):
        """Test retrieving subscriptions for a subscriber."""
        self.broker.subscribe("test1", "subscriber1")
        self.broker.subscribe("test2", "subscriber1")
        self.broker.subscribe("test3", "subscriber2")

        subs1 = self.broker.get_subscriptions("subscriber1")
        subs2 = self.broker.get_subscriptions("subscriber2")

        self.assertEqual(len(subs1), 2)
        self.assertEqual(len(subs2), 1)

    def test_subscription_filtering(self):
        """Test subscription with custom filter function."""
        def high_priority_filter(msg: Message) -> bool:
            return msg.priority > 5

        self.broker.subscribe("test", "subscriber1", filter_fn=high_priority_filter)

        self.broker.publish("test", {"data": "low"}, "sender", priority=3)
        self.broker.publish("test", {"data": "high"}, "sender", priority=8)

        messages = self.broker.poll("subscriber1")
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].payload["data"], "high")

    def test_dead_letter_queue(self):
        """Test dead letter queue operations."""
        self.broker.subscribe("test", "subscriber1")

        # Publish expired message
        msg_id = self.broker.publish("test", {"data": "expired"}, "sender", ttl_seconds=0)
        time.sleep(0.1)

        # Trigger cleanup
        self.broker.poll("subscriber1")

        # Check dead letter queue
        dead_letters = self.broker.get_dead_letters()
        self.assertEqual(len(dead_letters), 1)
        self.assertEqual(dead_letters[0].id, msg_id)

    def test_dead_letter_filtering(self):
        """Test filtering dead letters by topic and sender."""
        self.broker.subscribe("*", "subscriber1")

        # Create some dead letters
        msg1 = self.broker.publish("topic1", {}, "sender1", ttl_seconds=0)
        msg2 = self.broker.publish("topic1", {}, "sender2", ttl_seconds=0)
        msg3 = self.broker.publish("topic2", {}, "sender1", ttl_seconds=0)

        time.sleep(0.1)
        self.broker.poll("subscriber1")  # Trigger cleanup

        # Filter by topic
        topic1_dead = self.broker.get_dead_letters(topic="topic1")
        self.assertEqual(len(topic1_dead), 2)

        # Filter by sender
        sender1_dead = self.broker.get_dead_letters(sender="sender1")
        self.assertEqual(len(sender1_dead), 2)

        # Filter by both
        specific = self.broker.get_dead_letters(topic="topic1", sender="sender1")
        self.assertEqual(len(specific), 1)

    def test_retry_dead_letter(self):
        """Test retrying delivery of a dead letter message."""
        self.broker.subscribe("test", "subscriber1")

        # Create dead letter
        msg_id = self.broker.publish("test", {"data": "retry"}, "sender", ttl_seconds=0)
        time.sleep(0.1)
        self.broker.poll("subscriber1")  # Move to dead letter

        self.assertIn(msg_id, self.broker.dead_letters)

        # Retry
        result = self.broker.retry_dead_letter(msg_id)
        self.assertTrue(result)

        # Should be back in active queue
        self.assertNotIn(msg_id, self.broker.dead_letters)
        self.assertIn(msg_id, self.broker.messages)

        # Should be deliverable again
        messages = self.broker.poll("subscriber1")
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].id, msg_id)

    def test_cleanup_old_messages(self):
        """Test cleaning up old acknowledged and dead letter messages."""
        self.broker.subscribe("test", "subscriber1")

        # Create and acknowledge a message
        msg_id = self.broker.publish("test", {}, "sender")
        self.broker.poll("subscriber1")
        self.broker.acknowledge(msg_id, "subscriber1")

        # Manually age the message
        self.broker.messages[msg_id].timestamp = datetime.now() - timedelta(hours=25)

        # Clean up
        removed = self.broker.cleanup_old_messages(acknowledged_ttl_hours=24)
        self.assertEqual(removed['acknowledged'], 1)

    def test_get_stats(self):
        """Test retrieving broker statistics."""
        self.broker.subscribe("test", "subscriber1")

        self.broker.publish("test", {}, "sender1")
        self.broker.publish("test", {}, "sender2")

        self.broker.poll("subscriber1")

        stats = self.broker.get_stats()
        self.assertEqual(stats['messages_published'], 2)
        self.assertEqual(stats['messages_delivered'], 2)  # Same msg delivered to one subscriber
        self.assertEqual(stats['active_subscriptions'], 1)
        self.assertEqual(stats['active_subscribers'], 1)

    def test_get_message(self):
        """Test retrieving a message by ID."""
        msg_id = self.broker.publish("test", {}, "sender")

        msg = self.broker.get_message(msg_id)
        self.assertIsNotNone(msg)
        self.assertEqual(msg.id, msg_id)

        # Non-existent message
        fake_msg = self.broker.get_message("fake-id")
        self.assertIsNone(fake_msg)

    def test_list_topics(self):
        """Test listing all topics."""
        self.broker.publish("topic1", {}, "sender")
        self.broker.publish("topic2", {}, "sender")
        self.broker.publish("topic1", {}, "sender")  # Duplicate

        topics = self.broker.list_topics()
        self.assertEqual(len(topics), 2)
        self.assertIn("topic1", topics)
        self.assertIn("topic2", topics)

    def test_get_pending_count(self):
        """Test getting pending message count for a subscriber."""
        self.broker.subscribe("test", "subscriber1")

        self.broker.publish("test", {}, "sender")
        self.broker.publish("test", {}, "sender")

        count = self.broker.get_pending_count("subscriber1")
        self.assertEqual(count, 2)

        # Poll and acknowledge one
        messages = self.broker.poll("subscriber1", max_messages=1)
        self.broker.acknowledge(messages[0].id, "subscriber1")

        count = self.broker.get_pending_count("subscriber1")
        self.assertEqual(count, 1)

    def test_max_messages_limit(self):
        """Test that poll respects max_messages limit."""
        self.broker.subscribe("test", "subscriber1")

        for i in range(10):
            self.broker.publish("test", {"seq": i}, "sender")

        # Poll with limit
        messages = self.broker.poll("subscriber1", max_messages=5)
        self.assertEqual(len(messages), 5)

        # Acknowledge the first batch so they're not redelivered
        for msg in messages:
            self.broker.acknowledge(msg.id, "subscriber1")

        # Poll remaining
        messages2 = self.broker.poll("subscriber1", max_messages=10)
        self.assertEqual(len(messages2), 5)


class TestPersistence(unittest.TestCase):
    """Test message persistence to disk."""

    def setUp(self):
        """Create temporary directory for persistence."""
        self.temp_dir = tempfile.mkdtemp()
        self.broker = PubSubBroker(persist_dir=self.temp_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_persistence_directory_creation(self):
        """Test that persistence directory is created."""
        self.assertTrue(Path(self.temp_dir).exists())

    def test_message_persistence(self):
        """Test that messages are persisted to disk."""
        msg_id = self.broker.publish("test", {"data": "persisted"}, "sender")

        # Check file exists
        msg_file = Path(self.temp_dir) / f"msg_{msg_id}.json"
        self.assertTrue(msg_file.exists())

        # Check content
        with open(msg_file) as f:
            data = json.load(f)
            self.assertEqual(data['id'], msg_id)
            self.assertEqual(data['payload']['data'], "persisted")

    def test_load_persisted_state(self):
        """Test loading persisted messages on broker creation."""
        # Create and persist messages
        msg_id1 = self.broker.publish("test1", {"data": "msg1"}, "sender")
        msg_id2 = self.broker.publish("test2", {"data": "msg2"}, "sender")
        self.broker.save_state()

        # Create new broker with same persist_dir
        new_broker = PubSubBroker(persist_dir=self.temp_dir)

        # Should load persisted messages
        self.assertIn(msg_id1, new_broker.messages)
        self.assertIn(msg_id2, new_broker.messages)
        self.assertEqual(new_broker.messages[msg_id1].payload['data'], "msg1")

    def test_save_state(self):
        """Test saving complete broker state."""
        self.broker.subscribe("test", "subscriber1")
        self.broker.publish("test", {"data": "value"}, "sender")

        self.broker.save_state()

        # Check metadata file
        meta_file = Path(self.temp_dir) / "broker_meta.json"
        self.assertTrue(meta_file.exists())

        with open(meta_file) as f:
            meta = json.load(f)
            self.assertIn('stats', meta)


class TestHelperFunctions(unittest.TestCase):
    """Test helper functions for creating filters."""

    def test_create_topic_filter(self):
        """Test creating a topic filter by sender."""
        filter_fn = create_topic_filter({"sender1", "sender2"})

        msg1 = Message("id1", "test", {}, "sender1", datetime.now())
        msg2 = Message("id2", "test", {}, "sender3", datetime.now())

        self.assertTrue(filter_fn(msg1))
        self.assertFalse(filter_fn(msg2))

    def test_create_topic_filter_allow_all(self):
        """Test topic filter with no restrictions."""
        filter_fn = create_topic_filter(None)

        msg = Message("id1", "test", {}, "anyone", datetime.now())
        self.assertTrue(filter_fn(msg))

    def test_create_payload_filter(self):
        """Test creating a payload filter by required keys."""
        filter_fn = create_payload_filter({"task_id", "status"})

        msg1 = Message("id1", "test", {"task_id": "123", "status": "done"}, "sender", datetime.now())
        msg2 = Message("id2", "test", {"task_id": "456"}, "sender", datetime.now())  # Missing status

        self.assertTrue(filter_fn(msg1))
        self.assertFalse(filter_fn(msg2))


class TestIntegrationScenarios(unittest.TestCase):
    """Test realistic integration scenarios."""

    def test_agent_coordination_scenario(self):
        """Test a realistic agent coordination scenario."""
        broker = PubSubBroker()

        # Orchestrator subscribes to all task updates (using two separate patterns)
        # Note: With single-segment wildcards, "task.*" only matches "task.completed"
        # To match "task.assigned.*", we need a separate subscription or use "task.*.*"
        broker.subscribe("task.completed", "orchestrator")
        broker.subscribe("task.assigned.*", "orchestrator")

        # Workers subscribe to their specific task assignments
        broker.subscribe("task.assigned.worker1", "worker1")
        broker.subscribe("task.assigned.worker2", "worker2")

        # Monitor subscribes to all high-priority messages
        def high_priority(msg: Message) -> bool:
            return msg.priority >= 8

        # Monitor subscribes to task topics with high priority filter
        broker.subscribe("task.completed", "monitor", filter_fn=high_priority)
        broker.subscribe("task.assigned.*", "monitor", filter_fn=high_priority)

        # Orchestrator assigns tasks
        broker.publish("task.assigned.worker1", {"work": "process data"}, "orchestrator", priority=5)
        broker.publish("task.assigned.worker2", {"work": "generate report"}, "orchestrator", priority=3)

        # Worker1 completes task
        broker.publish("task.completed", {"worker": "worker1", "result": "success"}, "worker1", priority=9)

        # Check deliveries
        orch_msgs = broker.poll("orchestrator")
        self.assertEqual(len(orch_msgs), 3)  # All task messages

        worker1_msgs = broker.poll("worker1")
        self.assertEqual(len(worker1_msgs), 1)  # Only their assignment

        worker2_msgs = broker.poll("worker2")
        self.assertEqual(len(worker2_msgs), 1)  # Only their assignment

        monitor_msgs = broker.poll("monitor")
        self.assertEqual(len(monitor_msgs), 1)  # Only high priority completion

    def test_crisis_broadcast_scenario(self):
        """Test broadcasting crisis alerts to all agents."""
        broker = PubSubBroker()

        # All agents subscribe to crisis alerts
        broker.subscribe("crisis.*", "agent1")
        broker.subscribe("crisis.*", "agent2")
        broker.subscribe("crisis.*", "agent3")

        # Broadcast critical alert
        msg_id = broker.publish(
            "crisis.critical",
            {"message": "System overload detected", "action": "pause non-essential tasks"},
            "monitor",
            priority=10,
        )

        # All agents receive
        for agent_id in ["agent1", "agent2", "agent3"]:
            messages = broker.poll(agent_id)
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0].id, msg_id)
            broker.acknowledge(msg_id, agent_id)

        # After all acknowledge, message is fully processed
        msg = broker.get_message(msg_id)
        self.assertEqual(msg.status, MessageStatus.ACKNOWLEDGED)


if __name__ == '__main__':
    unittest.main()
