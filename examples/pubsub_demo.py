"""
Demo of the inter-agent pub/sub messaging system.

This example demonstrates:
- Basic publish/subscribe operations
- Topic wildcards and pattern matching
- Message priorities
- At-least-once delivery with acknowledgments
- Dead letter queue handling
- Message filtering
"""

from cortical.reasoning import (
    PubSubBroker,
    Message,
    create_topic_filter,
    create_payload_filter,
)


def demo_basic_pubsub():
    """Demonstrate basic publish/subscribe."""
    print("=" * 70)
    print("DEMO 1: Basic Publish/Subscribe")
    print("=" * 70)

    broker = PubSubBroker()

    # Subscribe to a topic
    broker.subscribe("agent.task.completed", "worker1")

    # Publish a message
    msg_id = broker.publish(
        topic="agent.task.completed",
        payload={"task_id": "123", "result": "success"},
        sender="orchestrator",
    )
    print(f"✓ Published message: {msg_id}")

    # Poll for messages
    messages = broker.poll("worker1")
    print(f"✓ Received {len(messages)} message(s)")
    for msg in messages:
        print(f"  - Topic: {msg.topic}")
        print(f"  - Payload: {msg.payload}")
        broker.acknowledge(msg.id, "worker1")

    print()


def demo_wildcard_subscriptions():
    """Demonstrate wildcard topic patterns."""
    print("=" * 70)
    print("DEMO 2: Wildcard Topic Patterns")
    print("=" * 70)

    broker = PubSubBroker()

    # Subscribe to pattern
    broker.subscribe("agent.task.*", "monitor")

    # Publish to various topics
    broker.publish("agent.task.started", {"task": "A"}, "worker1")
    broker.publish("agent.task.completed", {"task": "B"}, "worker2")
    broker.publish("agent.task.failed", {"task": "C"}, "worker3")
    broker.publish("agent.status", {"status": "idle"}, "worker1")  # Won't match

    messages = broker.poll("monitor")
    print(f"✓ Pattern 'agent.task.*' matched {len(messages)} messages:")
    for msg in messages:
        print(f"  - {msg.topic}: {msg.payload}")
        broker.acknowledge(msg.id, "monitor")

    print()


def demo_message_priorities():
    """Demonstrate priority-based message delivery."""
    print("=" * 70)
    print("DEMO 3: Priority-Based Delivery")
    print("=" * 70)

    broker = PubSubBroker()
    broker.subscribe("alerts.*", "monitor")

    # Publish messages with different priorities
    broker.publish("alerts.info", {"msg": "System update available"}, "system", priority=2)
    broker.publish("alerts.critical", {"msg": "Database connection lost"}, "system", priority=10)
    broker.publish("alerts.warning", {"msg": "High memory usage"}, "system", priority=6)

    messages = broker.poll("monitor")
    print("✓ Messages delivered in priority order (highest first):")
    for i, msg in enumerate(messages, 1):
        print(f"  {i}. [Priority {msg.priority}] {msg.topic}: {msg.payload['msg']}")
        broker.acknowledge(msg.id, "monitor")

    print()


def demo_multiple_subscribers():
    """Demonstrate broadcast to multiple subscribers."""
    print("=" * 70)
    print("DEMO 4: Broadcast to Multiple Subscribers")
    print("=" * 70)

    broker = PubSubBroker()

    # Multiple agents subscribe to the same topic
    broker.subscribe("crisis.alert", "agent1")
    broker.subscribe("crisis.alert", "agent2")
    broker.subscribe("crisis.alert", "agent3")

    # Broadcast critical alert
    msg_id = broker.publish(
        "crisis.alert",
        {"message": "System overload detected", "action": "pause non-essential tasks"},
        "monitor",
        priority=10,
    )

    # Each agent receives the message
    for agent_id in ["agent1", "agent2", "agent3"]:
        messages = broker.poll(agent_id)
        print(f"✓ {agent_id} received: {messages[0].payload['message']}")
        broker.acknowledge(msg_id, agent_id)

    print()


def demo_message_filtering():
    """Demonstrate subscription filtering."""
    print("=" * 70)
    print("DEMO 5: Subscription Filtering")
    print("=" * 70)

    broker = PubSubBroker()

    # Subscribe with high-priority filter
    def high_priority(msg: Message) -> bool:
        return msg.priority >= 8

    broker.subscribe("task.*", "urgent_handler", filter_fn=high_priority)
    broker.subscribe("task.*", "all_handler")  # No filter

    # Publish messages
    broker.publish("task.started", {"task": "low"}, "worker", priority=3)
    broker.publish("task.started", {"task": "high"}, "worker", priority=9)

    urgent_msgs = broker.poll("urgent_handler")
    all_msgs = broker.poll("all_handler")

    print(f"✓ Urgent handler (priority >= 8): {len(urgent_msgs)} message(s)")
    print(f"✓ All handler (no filter): {len(all_msgs)} message(s)")

    for msg in urgent_msgs:
        broker.acknowledge(msg.id, "urgent_handler")
    for msg in all_msgs:
        broker.acknowledge(msg.id, "all_handler")

    print()


def demo_dead_letter_queue():
    """Demonstrate dead letter queue for expired messages."""
    print("=" * 70)
    print("DEMO 6: Dead Letter Queue")
    print("=" * 70)

    broker = PubSubBroker()
    broker.subscribe("task.urgent", "worker1")

    # Publish message with very short TTL
    msg_id = broker.publish(
        "task.urgent",
        {"task": "time-sensitive operation"},
        "orchestrator",
        ttl_seconds=0,  # Expires immediately
    )

    import time
    time.sleep(0.1)

    # Try to poll (message will be expired)
    messages = broker.poll("worker1")
    print(f"✓ Received {len(messages)} messages (expired message not delivered)")

    # Check dead letter queue
    dead_letters = broker.get_dead_letters()
    print(f"✓ Dead letter queue has {len(dead_letters)} message(s)")
    for dl in dead_letters:
        print(f"  - Topic: {dl.topic}, Status: {dl.status.name}")

    # Retry the dead letter
    if dead_letters:
        retry_result = broker.retry_dead_letter(dead_letters[0].id)
        print(f"✓ Retry dead letter: {retry_result}")

        # Now it should be deliverable
        messages = broker.poll("worker1")
        print(f"✓ After retry: {len(messages)} message(s) delivered")
        for msg in messages:
            broker.acknowledge(msg.id, "worker1")

    print()


def demo_statistics():
    """Demonstrate broker statistics and inspection."""
    print("=" * 70)
    print("DEMO 7: Broker Statistics")
    print("=" * 70)

    broker = PubSubBroker()

    # Set up some activity
    broker.subscribe("task.*", "worker1")
    broker.subscribe("task.*", "worker2")

    broker.publish("task.started", {}, "orchestrator")
    broker.publish("task.completed", {}, "orchestrator")

    broker.poll("worker1")
    broker.poll("worker2")

    # Get statistics
    stats = broker.get_stats()
    print("✓ Broker Statistics:")
    print(f"  - Messages published: {stats['messages_published']}")
    print(f"  - Messages delivered: {stats['messages_delivered']}")
    print(f"  - Active messages: {stats['active_messages']}")
    print(f"  - Active subscriptions: {stats['active_subscriptions']}")
    print(f"  - Active subscribers: {stats['active_subscribers']}")

    # List topics
    topics = broker.list_topics()
    print(f"✓ Topics published to: {sorted(topics)}")

    print()


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "PUB/SUB MESSAGING SYSTEM DEMO" + " " * 23 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    demo_basic_pubsub()
    demo_wildcard_subscriptions()
    demo_message_priorities()
    demo_multiple_subscribers()
    demo_message_filtering()
    demo_dead_letter_queue()
    demo_statistics()

    print("=" * 70)
    print("All demos completed successfully!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
