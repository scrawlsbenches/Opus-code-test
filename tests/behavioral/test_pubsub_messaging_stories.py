"""
Behavioral Tests for Pub/Sub Messaging System.

This module tests the topic-based publish-subscribe messaging system
we built ourselves for inter-agent coordination.

Epic: System architect coordinates distributed agents
Story: As a system architect building distributed coordination,
       I want publish-subscribe messaging we implemented ourselves,
       So that agents can communicate through channels we control completely.
"""

import pytest
import time
from datetime import datetime, timedelta
from cortical.reasoning.pubsub import (
    PubSubBroker,
    Message,
    MessageStatus,
    Subscription,
)


class TestDistributedSystemCoordinatesAgents:
    """
    Epic: Distributed System Coordinates Agents

    As a system architect building distributed coordination,
    I want messaging infrastructure we built ourselves,
    So that I control agent communication completely.
    """

    def test_scenario_agent_publishes_message_to_topic(self):
        """
        Scenario: Agent publishes messages to topics

        Given a messaging broker we built
        When an agent publishes to a topic
        Then message is stored and routable
        Because we implemented pub/sub ourselves
        """
        # Given messaging broker
        broker = PubSubBroker()

        # When publishing message
        msg_id = broker.publish(
            topic="agent.task.completed",
            payload={"task_id": "123", "result": "success"},
            sender="worker-001"
        )

        # Then message is stored
        assert msg_id is not None
        message = broker.get_message(msg_id)
        assert message is not None
        assert message.topic == "agent.task.completed"
        assert message.sender == "worker-001"

    def test_scenario_agent_subscribes_to_topic_pattern(self):
        """
        Scenario: Agents subscribe to topic patterns with wildcards

        Given a broker routing messages we built
        When an agent subscribes to a pattern
        Then matching messages are delivered
        Because we implemented pattern matching ourselves
        """
        # Given broker
        broker = PubSubBroker()

        # When subscribing to pattern
        subscription = broker.subscribe("agent.task.*", "monitor-001")

        # Then subscription is active
        assert subscription is not None
        assert subscription.topic_pattern == "agent.task.*"
        assert subscription.subscriber_id == "monitor-001"

    def test_scenario_message_routes_to_matching_subscribers(self):
        """
        Scenario: Published messages route to matching subscribers

        Given subscribers to different patterns we configured
        When messages are published
        Then only matching subscribers receive them
        Because we built routing logic ourselves
        """
        # Given subscribers
        broker = PubSubBroker()
        broker.subscribe("agent.task.*", "worker-001")
        broker.subscribe("agent.error.*", "monitor-001")
        broker.subscribe("agent.*.*", "logger-001")  # Catches all

        # When publishing messages
        task_msg = broker.publish("agent.task.started", {"id": "1"}, "orchestrator")
        error_msg = broker.publish("agent.error.critical", {"msg": "fail"}, "worker-002")

        # Then routing works correctly
        worker_msgs = broker.poll("worker-001")
        monitor_msgs = broker.poll("monitor-001")
        logger_msgs = broker.poll("logger-001")

        # Worker should only get task messages
        assert len(worker_msgs) == 1
        assert worker_msgs[0].topic == "agent.task.started"

        # Monitor should only get error messages
        assert len(monitor_msgs) == 1
        assert monitor_msgs[0].topic == "agent.error.critical"

        # Logger should get both
        assert len(logger_msgs) == 2

    def test_scenario_priority_messages_delivered_first(self):
        """
        Scenario: High priority messages bypass queue

        Given a queue with normal messages
        When high priority message arrives
        Then it's delivered first
        Because we implemented priority queuing ourselves
        """
        # Given normal messages
        broker = PubSubBroker()
        broker.subscribe("alerts.*", "responder-001")

        broker.publish("alerts.info", {"msg": "low"}, "system", priority=0)
        broker.publish("alerts.warn", {"msg": "medium"}, "system", priority=5)
        broker.publish("alerts.critical", {"msg": "urgent"}, "system", priority=10)

        # When polling
        messages = broker.poll("responder-001", max_messages=3)

        # Then high priority first
        assert len(messages) == 3
        assert messages[0].priority == 10  # Critical first
        assert messages[1].priority == 5   # Warning second
        assert messages[2].priority == 0   # Info last


class TestMessagingSystemEnsuresReliability:
    """
    Epic: Messaging System Ensures Reliable Delivery

    As a system requiring reliable communication,
    I want guaranteed delivery and acknowledgment,
    So that no messages are lost.
    """

    def test_scenario_message_remains_pending_until_acknowledged(self):
        """
        Scenario: Unacknowledged messages stay in queue

        Given messages delivered to subscriber
        When subscriber doesn't acknowledge
        Then messages remain for redelivery
        Because we implement at-least-once delivery ourselves
        """
        # Given delivered messages
        broker = PubSubBroker()
        broker.subscribe("work.queue", "worker-001")
        msg_id = broker.publish("work.queue", {"task": "process"}, "scheduler")

        # When polling without acknowledgment
        messages1 = broker.poll("worker-001")
        assert len(messages1) == 1

        # Then message redelivered on next poll
        messages2 = broker.poll("worker-001")
        assert len(messages2) == 1
        assert messages2[0].id == messages1[0].id

    def test_scenario_acknowledged_message_not_redelivered(self):
        """
        Scenario: Acknowledged messages leave the queue

        Given a message delivered and acknowledged
        When subscriber polls again
        Then message is not redelivered
        Because we track acknowledgments ourselves
        """
        # Given delivered and acknowledged
        broker = PubSubBroker()
        broker.subscribe("work.done", "collector-001")
        msg_id = broker.publish("work.done", {"id": "1"}, "worker")

        messages = broker.poll("collector-001")
        broker.acknowledge(messages[0].id, "collector-001")

        # When polling again
        messages2 = broker.poll("collector-001")

        # Then not redelivered
        assert len(messages2) == 0

    def test_scenario_expired_messages_move_to_dead_letter(self):
        """
        Scenario: TTL enforcement prevents stale messages

        Given messages with time-to-live we set
        When TTL expires
        Then messages move to dead letter queue
        Because we implement expiration ourselves
        """
        # Given messages with TTL
        broker = PubSubBroker()
        broker.subscribe("ephemeral.*", "subscriber-001")

        # Publish with very short TTL
        msg_id = broker.publish(
            "ephemeral.event",
            {"data": "time-sensitive"},
            "publisher",
            ttl_seconds=1  # 1 second TTL
        )

        # Wait for expiration
        time.sleep(2)

        # When polling
        messages = broker.poll("subscriber-001")

        # Then message expired (moved to dead letter)
        assert len(messages) == 0

        # Check dead letter queue
        dead_letters = broker.get_dead_letters()
        assert len(dead_letters) > 0
        assert dead_letters[0].status == MessageStatus.EXPIRED

    def test_scenario_dead_letter_messages_can_be_retried(self):
        """
        Scenario: Failed messages can be retried

        Given a message in dead letter queue
        When I retry delivery with new TTL
        Then message returns to active queue
        Because we built retry mechanism ourselves
        """
        # Given dead letter message (create by expiring)
        broker = PubSubBroker()
        broker.subscribe("retryable", "worker-001")

        msg_id = broker.publish(
            "retryable",
            {"attempt": 1},
            "scheduler",
            ttl_seconds=1
        )

        time.sleep(2)  # Let it expire
        broker.poll("worker-001")  # Trigger expiration

        # When retrying
        success = broker.retry_dead_letter(msg_id, new_ttl_seconds=300)

        # Then back in active queue
        assert success is True
        messages = broker.poll("worker-001")
        assert len(messages) == 1


class TestMessagingSystemSupportsFiltering:
    """
    Epic: Messaging System Supports Advanced Filtering

    As a subscriber wanting specific messages,
    I want content-based filtering,
    So that I only receive relevant messages.
    """

    def test_scenario_subscriber_filters_by_sender(self):
        """
        Scenario: Filter messages by sender identity

        Given messages from various senders
        When I filter by allowed senders
        Then only permitted messages arrive
        Because we built filtering ourselves
        """
        # Given messages from various senders
        from cortical.reasoning.pubsub import create_topic_filter

        broker = PubSubBroker()
        filter_fn = create_topic_filter(allowed_senders={"trusted-agent"})

        broker.subscribe("notifications", "receiver-001", filter_fn=filter_fn)

        # When publishing from different senders
        broker.publish("notifications", {"msg": "from trusted"}, "trusted-agent")
        broker.publish("notifications", {"msg": "from untrusted"}, "random-agent")

        # Then only trusted messages received
        messages = broker.poll("receiver-001")
        assert len(messages) == 1
        assert messages[0].sender == "trusted-agent"

    def test_scenario_subscriber_filters_by_payload_content(self):
        """
        Scenario: Filter messages by payload fields

        Given messages with varying payloads
        When I filter by required keys
        Then only complete messages arrive
        Because we implemented payload filtering ourselves
        """
        # Given varying payloads
        from cortical.reasoning.pubsub import create_payload_filter

        broker = PubSubBroker()
        filter_fn = create_payload_filter(required_keys={"task_id", "priority"})

        broker.subscribe("tasks", "worker-001", filter_fn=filter_fn)

        # When publishing with different payloads
        broker.publish("tasks", {"task_id": "1", "priority": "high"}, "scheduler")  # Has both
        broker.publish("tasks", {"task_id": "2"}, "scheduler")  # Missing priority

        # Then only complete messages received
        messages = broker.poll("worker-001")
        assert len(messages) == 1
        assert "priority" in messages[0].payload


class TestMessagingSystemProvidesObservability:
    """
    Epic: Messaging System Provides Observability

    As a system operator monitoring message flow,
    I want statistics and inspection tools,
    So that I understand system health.
    """

    def test_scenario_broker_reports_comprehensive_statistics(self):
        """
        Scenario: Statistics reveal message flow health

        Given a broker handling messages we built
        When I query statistics
        Then comprehensive metrics are available
        Because we track everything ourselves
        """
        # Given active broker
        broker = PubSubBroker()
        broker.subscribe("stats.test", "sub-001")
        broker.publish("stats.test", {"n": 1}, "pub-001")
        messages = broker.poll("sub-001")
        broker.acknowledge(messages[0].id, "sub-001")

        # When querying stats
        stats = broker.get_stats()

        # Then metrics available
        assert 'messages_published' in stats
        assert 'messages_delivered' in stats
        assert 'messages_acknowledged' in stats
        assert 'active_subscriptions' in stats
        assert stats['messages_published'] >= 1
        assert stats['messages_acknowledged'] >= 1

    def test_scenario_topic_listing_shows_active_channels(self):
        """
        Scenario: List all active topics

        Given messages on various topics
        When I list topics
        Then all active channels are shown
        Because we track topics ourselves
        """
        # Given various topics
        broker = PubSubBroker()
        broker.publish("channel.a", {}, "agent1")
        broker.publish("channel.b", {}, "agent2")
        broker.publish("channel.c", {}, "agent3")

        # When listing topics
        topics = broker.list_topics()

        # Then all shown
        assert len(topics) >= 3
        assert "channel.a" in topics
        assert "channel.b" in topics
        assert "channel.c" in topics

    def test_scenario_pending_count_monitors_queue_depth(self):
        """
        Scenario: Monitor queue depth per subscriber

        Given subscribers with pending messages
        When I check queue depth
        Then backlog is visible
        Because we monitor queues ourselves
        """
        # Given pending messages
        broker = PubSubBroker()
        broker.subscribe("work.*", "worker-001")

        for i in range(5):
            broker.publish("work.task", {"id": i}, "scheduler")

        # When checking depth
        pending = broker.get_pending_count("worker-001")

        # Then backlog visible
        assert pending == 5


class TestMessagingSystemSupportsPersistence:
    """
    Epic: Messaging System Supports Durability

    As a system requiring durability,
    I want message persistence,
    So that restarts don't lose data.
    """

    def test_scenario_persisted_messages_survive_restart(self, tmp_path):
        """
        Scenario: Messages persist across restarts

        Given a broker with persistence enabled
        When messages are published and broker restarts
        Then messages are recovered
        Because we built persistence ourselves
        """
        # Given broker with persistence
        persist_dir = tmp_path / "pubsub_test"
        broker1 = PubSubBroker(persist_dir=str(persist_dir))

        broker1.subscribe("persistent", "sub-001")
        msg_id = broker1.publish("persistent", {"data": "important"}, "publisher")
        broker1.save_state()

        # When restarting
        broker2 = PubSubBroker(persist_dir=str(persist_dir))

        # Then messages recovered
        recovered_msg = broker2.get_message(msg_id)
        assert recovered_msg is not None
        assert recovered_msg.payload["data"] == "important"

    def test_scenario_old_messages_cleaned_up_automatically(self):
        """
        Scenario: Old acknowledged messages are cleaned up

        Given acknowledged messages aging out
        When cleanup runs
        Then disk space is reclaimed
        Because we implement cleanup ourselves
        """
        # Given old acknowledged messages
        broker = PubSubBroker()
        broker.subscribe("cleanup.test", "sub-001")

        msg_id = broker.publish("cleanup.test", {"old": True}, "pub")
        messages = broker.poll("sub-001")
        broker.acknowledge(messages[0].id, "sub-001")

        # When cleaning up (0 hours = immediate)
        removed = broker.cleanup_old_messages(acknowledged_ttl_hours=0)

        # Then old messages removed
        assert removed['acknowledged'] >= 1
