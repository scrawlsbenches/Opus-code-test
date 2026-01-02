"""
Behavioral tests for inter-agent pub/sub messaging system.

As an agent in a multi-agent system,
I want to publish and subscribe to messages asynchronously,
So that agents can coordinate without tight coupling.

Based on: examples/pubsub_demo.py
"""

import pytest
import time
from cortical.reasoning import (
    PubSubBroker,
    Message,
    create_topic_filter,
    create_payload_filter,
)


class TestAgentsPublishAndSubscribeToTopics:
    """
    Epic: Basic Messaging

    As an agent needing to notify others,
    I want to publish messages to topics,
    So that interested subscribers receive notifications.
    """

    def test_scenario_agent_receives_published_messages(self):
        """
        Scenario: Subscriber receives published message

        Given an agent subscribed to a topic
        When another agent publishes to that topic
        Then the subscriber receives the message
        Because pub/sub enables decoupled communication
        """
        # Given: agent subscribed to topic
        broker = PubSubBroker()
        broker.subscribe("task.completed", "worker1")

        # When: publishing to topic
        msg_id = broker.publish(
            topic="task.completed",
            payload={"task_id": "123", "result": "success"},
            sender="orchestrator",
        )

        # Then: subscriber receives message
        messages = broker.poll("worker1")
        assert len(messages) == 1
        assert messages[0].topic == "task.completed"
        assert messages[0].payload["task_id"] == "123"

    def test_scenario_unsubscribed_agents_receive_nothing(self):
        """
        Scenario: Non-subscribers don't receive messages

        Given an agent not subscribed to a topic
        When a message is published to that topic
        Then the agent receives nothing
        Because subscriptions are explicit
        """
        # Given: agent not subscribed
        broker = PubSubBroker()
        broker.subscribe("topic.a", "agent1")

        # When: publishing to different topic
        broker.publish("topic.b", {"data": "test"}, "sender")

        # Then: agent1 receives nothing
        messages = broker.poll("agent1")
        assert len(messages) == 0

    def test_scenario_messages_require_acknowledgment(self):
        """
        Scenario: Messages must be acknowledged

        Given a subscriber that receives a message
        When acknowledging the message
        Then the message is marked delivered
        Because at-least-once delivery requires acks
        """
        # Given: subscriber receives message
        broker = PubSubBroker()
        broker.subscribe("events", "agent1")
        msg_id = broker.publish("events", {"event": "test"}, "sender")
        messages = broker.poll("agent1")

        # When: acknowledging
        result = broker.acknowledge(messages[0].id, "agent1")

        # Then: acknowledged
        assert result is True


class TestAgentsUseWildcardSubscriptions:
    """
    Epic: Pattern-Based Subscription

    As an agent monitoring multiple related topics,
    I want to subscribe using wildcard patterns,
    So that I don't need individual subscriptions.
    """

    def test_scenario_wildcard_matches_multiple_topics(self):
        """
        Scenario: Wildcard pattern matches topic hierarchy

        Given an agent subscribed to "task.*"
        When messages published to "task.started" and "task.completed"
        Then agent receives both messages
        Because wildcards match topic patterns
        """
        # Given: wildcard subscription
        broker = PubSubBroker()
        broker.subscribe("task.*", "monitor")

        # When: publishing to matching topics
        broker.publish("task.started", {"task": "A"}, "worker1")
        broker.publish("task.completed", {"task": "B"}, "worker2")
        broker.publish("status.update", {"status": "idle"}, "worker1")  # Doesn't match

        # Then: receives matching messages only
        messages = broker.poll("monitor")
        assert len(messages) == 2
        topics = {msg.topic for msg in messages}
        assert "task.started" in topics
        assert "task.completed" in topics

    def test_scenario_wildcard_filters_non_matching_topics(self):
        """
        Scenario: Wildcard doesn't match unrelated topics

        Given an agent subscribed to "agent.task.*"
        When message published to "system.alert"
        Then agent receives nothing
        Because pattern doesn't match
        """
        # Given: specific pattern subscription
        broker = PubSubBroker()
        broker.subscribe("agent.task.*", "agent1")

        # When: publishing to non-matching topic
        broker.publish("system.alert", {"alert": "test"}, "system")

        # Then: no messages received
        messages = broker.poll("agent1")
        assert len(messages) == 0


class TestAgentsReceiveMessagesByPriority:
    """
    Epic: Priority-Based Delivery

    As an agent processing messages,
    I want high-priority messages delivered first,
    So that urgent work is handled immediately.
    """

    def test_scenario_higher_priority_messages_delivered_first(self):
        """
        Scenario: Messages ordered by priority descending

        Given messages published with different priorities
        When subscriber polls for messages
        Then messages are ordered by priority (highest first)
        Because urgent messages need immediate attention
        """
        # Given: messages with different priorities
        broker = PubSubBroker()
        broker.subscribe("alerts.*", "monitor")

        broker.publish("alerts.info", {"msg": "Info"}, "system", priority=2)
        broker.publish("alerts.critical", {"msg": "Critical"}, "system", priority=10)
        broker.publish("alerts.warning", {"msg": "Warning"}, "system", priority=6)

        # When: polling
        messages = broker.poll("monitor")

        # Then: ordered by priority
        assert len(messages) == 3
        assert messages[0].priority == 10  # Critical first
        assert messages[1].priority == 6   # Warning second
        assert messages[2].priority == 2   # Info last

    def test_scenario_default_priority_is_standard(self):
        """
        Scenario: Messages without priority get default value

        Given a message published without priority specified
        When checking message priority
        Then priority has reasonable default
        Because not all messages need explicit priority
        """
        # Given: message without explicit priority
        broker = PubSubBroker()
        broker.subscribe("events", "agent1")
        broker.publish("events", {"data": "test"}, "sender")

        # When: checking priority
        messages = broker.poll("agent1")

        # Then: has default priority
        assert len(messages) == 1
        assert messages[0].priority >= 0  # Has some priority value


class TestMultipleAgentsReceiveSameMessage:
    """
    Epic: Broadcast Messaging

    As a coordinator broadcasting alerts,
    I want all subscribed agents to receive messages,
    So that system-wide notifications work.
    """

    def test_scenario_multiple_subscribers_all_receive_message(self):
        """
        Scenario: Broadcast to all subscribers

        Given multiple agents subscribed to same topic
        When a message is published
        Then all subscribers receive the message
        Because broadcasts reach all interested parties
        """
        # Given: multiple subscribers
        broker = PubSubBroker()
        broker.subscribe("crisis.alert", "agent1")
        broker.subscribe("crisis.alert", "agent2")
        broker.subscribe("crisis.alert", "agent3")

        # When: publishing
        msg_id = broker.publish(
            "crisis.alert",
            {"message": "System overload"},
            "monitor",
        )

        # Then: all receive it
        for agent_id in ["agent1", "agent2", "agent3"]:
            messages = broker.poll(agent_id)
            assert len(messages) == 1
            assert messages[0].payload["message"] == "System overload"

    def test_scenario_each_subscriber_acknowledges_independently(self):
        """
        Scenario: Subscribers acknowledge independently

        Given multiple subscribers receiving same message
        When each acknowledges the message
        Then each acknowledgment is tracked separately
        Because delivery is confirmed per-subscriber
        """
        # Given: multiple subscribers with message
        broker = PubSubBroker()
        broker.subscribe("event", "agent1")
        broker.subscribe("event", "agent2")
        msg_id = broker.publish("event", {"data": "test"}, "sender")

        messages1 = broker.poll("agent1")
        messages2 = broker.poll("agent2")

        # When: each acknowledges
        ack1 = broker.acknowledge(msg_id, "agent1")
        ack2 = broker.acknowledge(msg_id, "agent2")

        # Then: both successful
        assert ack1 is True
        assert ack2 is True


class TestAgentsFilterMessagesOnSubscription:
    """
    Epic: Selective Message Reception

    As an agent with specific interests,
    I want to filter messages on subscription,
    So that I only receive relevant messages.
    """

    def test_scenario_filter_function_selects_messages(self):
        """
        Scenario: Subscription filter limits messages

        Given an agent subscribed with high-priority filter
        When messages of varying priority are published
        Then only high-priority messages are received
        Because filters reduce noise
        """
        # Given: subscription with priority filter
        broker = PubSubBroker()

        def high_priority(msg: Message) -> bool:
            return msg.priority >= 8

        broker.subscribe("task.*", "urgent_handler", filter_fn=high_priority)
        broker.subscribe("task.*", "all_handler")  # No filter

        # When: publishing varied priorities
        broker.publish("task.started", {"task": "low"}, "worker", priority=3)
        broker.publish("task.started", {"task": "high"}, "worker", priority=9)

        # Then: filtered subscriber gets only high-priority
        urgent_msgs = broker.poll("urgent_handler")
        all_msgs = broker.poll("all_handler")

        assert len(urgent_msgs) == 1  # Only high priority
        assert urgent_msgs[0].priority == 9
        assert len(all_msgs) == 2  # Gets both

    def test_scenario_filter_receives_no_messages_when_none_match(self):
        """
        Scenario: Filter blocks all non-matching messages

        Given an agent with restrictive filter
        When messages not matching filter are published
        Then agent receives no messages
        Because filters enforce criteria
        """
        # Given: very restrictive filter
        broker = PubSubBroker()

        def only_critical(msg: Message) -> bool:
            return msg.priority >= 10

        broker.subscribe("alerts.*", "critical_handler", filter_fn=only_critical)

        # When: publishing lower priority
        broker.publish("alerts.warning", {"msg": "Warning"}, "system", priority=5)

        # Then: no messages received
        messages = broker.poll("critical_handler")
        assert len(messages) == 0


class TestBrokerHandlesExpiredMessages:
    """
    Epic: Message Lifecycle Management

    As a broker managing message lifetime,
    I want expired messages moved to dead letter queue,
    So that stale messages don't clog the system.
    """

    def test_scenario_expired_messages_not_delivered(self):
        """
        Scenario: TTL-expired messages go to DLQ

        Given a message published with short TTL
        When TTL expires before delivery
        Then message is not delivered to subscriber
        Because expired messages are invalid
        """
        # Given: message with very short TTL
        broker = PubSubBroker()
        broker.subscribe("urgent.task", "worker1")

        msg_id = broker.publish(
            "urgent.task",
            {"task": "time-sensitive"},
            "orchestrator",
            ttl_seconds=0,  # Expires immediately
        )

        time.sleep(0.1)

        # When: polling
        messages = broker.poll("worker1")

        # Then: not delivered
        assert len(messages) == 0

    def test_scenario_dead_letters_can_be_inspected(self):
        """
        Scenario: Dead letter queue is accessible

        Given messages that expired
        When checking dead letter queue
        Then expired messages are listed
        Because operators need to see failures
        """
        # Given: expired message
        broker = PubSubBroker()
        broker.subscribe("task.urgent", "worker1")
        broker.publish("task.urgent", {"task": "expired"}, "sender", ttl_seconds=0)
        time.sleep(0.1)
        broker.poll("worker1")  # Triggers DLQ move

        # When: checking DLQ
        dead_letters = broker.get_dead_letters()

        # Then: message in DLQ
        assert len(dead_letters) > 0

    def test_scenario_dead_letters_can_be_retried(self):
        """
        Scenario: Dead letter messages can be retried

        Given a message in dead letter queue
        When retrying the message
        Then message becomes deliverable again
        Because some failures are transient
        """
        # Given: message in DLQ
        broker = PubSubBroker()
        broker.subscribe("task.retry", "worker1")
        msg_id = broker.publish("task.retry", {"task": "retry-test"}, "sender", ttl_seconds=0)
        time.sleep(0.1)
        broker.poll("worker1")

        dead_letters = broker.get_dead_letters()
        assert len(dead_letters) > 0

        # When: retrying
        result = broker.retry_dead_letter(dead_letters[0].id)

        # Then: deliverable again
        assert result is True
        messages = broker.poll("worker1")
        assert len(messages) > 0


class TestBrokerProvidesStatistics:
    """
    Epic: Observability and Monitoring

    As a system operator,
    I want to inspect broker statistics,
    So that I can monitor message flow.
    """

    def test_scenario_broker_tracks_message_counts(self):
        """
        Scenario: Statistics include message counts

        Given a broker with message activity
        When requesting statistics
        Then message counts are reported
        Because operators need volume metrics
        """
        # Given: broker with activity
        broker = PubSubBroker()
        broker.subscribe("events", "agent1")
        broker.publish("events", {"data": "test1"}, "sender")
        broker.publish("events", {"data": "test2"}, "sender")
        broker.poll("agent1")

        # When: requesting stats
        stats = broker.get_stats()

        # Then: counts present
        assert "messages_published" in stats
        assert "messages_delivered" in stats
        assert stats["messages_published"] >= 2

    def test_scenario_broker_tracks_active_subscriptions(self):
        """
        Scenario: Statistics include subscription counts

        Given a broker with multiple subscriptions
        When requesting statistics
        Then subscription counts are reported
        Because operators need connectivity metrics
        """
        # Given: multiple subscriptions
        broker = PubSubBroker()
        broker.subscribe("topic.a", "agent1")
        broker.subscribe("topic.b", "agent1")
        broker.subscribe("topic.a", "agent2")

        # When: requesting stats
        stats = broker.get_stats()

        # Then: subscription info present
        assert "active_subscriptions" in stats
        assert "active_subscribers" in stats
        assert stats["active_subscriptions"] >= 3
        assert stats["active_subscribers"] >= 2

    def test_scenario_broker_lists_active_topics(self):
        """
        Scenario: Broker reports which topics have messages

        Given messages published to various topics
        When listing topics
        Then all published topics are listed
        Because operators need topic visibility
        """
        # Given: messages to different topics
        broker = PubSubBroker()
        broker.publish("topic.alpha", {}, "sender")
        broker.publish("topic.beta", {}, "sender")
        broker.publish("topic.gamma", {}, "sender")

        # When: listing topics
        topics = broker.list_topics()

        # Then: all topics present
        assert "topic.alpha" in topics
        assert "topic.beta" in topics
        assert "topic.gamma" in topics


class TestAgentsCoordinateViaMessaging:
    """
    Epic: Real-World Coordination

    As agents in a distributed system,
    I want to coordinate via pub/sub,
    So that we can work together without tight coupling.
    """

    def test_scenario_task_completion_notifies_waiting_agents(self):
        """
        Scenario: Agent broadcasts task completion

        Given agents waiting for task completion
        When worker completes task and publishes
        Then waiting agents are notified
        Because dependent work can now proceed
        """
        # Given: agents waiting for completion
        broker = PubSubBroker()
        broker.subscribe("task.*.completed", "agent1")
        broker.subscribe("task.*.completed", "agent2")

        # When: worker completes task
        broker.publish(
            "task.auth.completed",
            {"task_id": "auth-123", "result": "success"},
            "worker1",
        )

        # Then: waiting agents notified
        for agent in ["agent1", "agent2"]:
            messages = broker.poll(agent)
            assert len(messages) == 1
            assert "auth-123" in messages[0].payload["task_id"]

    def test_scenario_crisis_alert_broadcasts_to_all_agents(self):
        """
        Scenario: System-wide crisis notification

        Given multiple agents subscribed to crisis alerts
        When crisis is detected
        Then all agents receive immediate notification
        Because system-wide events need broadcast
        """
        # Given: agents subscribed to alerts
        broker = PubSubBroker()
        agents = ["agent1", "agent2", "agent3", "agent4"]
        for agent in agents:
            broker.subscribe("crisis.alert", agent)

        # When: crisis detected
        broker.publish(
            "crisis.alert",
            {"level": "critical", "action": "pause tasks"},
            "monitor",
            priority=10,  # Highest priority
        )

        # Then: all agents receive alert
        for agent in agents:
            messages = broker.poll(agent)
            assert len(messages) == 1
            assert messages[0].priority == 10
            assert messages[0].payload["level"] == "critical"
