"""
Inter-Agent Pub/Sub Messaging System.

This module implements a topic-based publish-subscribe messaging system for
coordination between multiple agents in the Cortical Text Processor's reasoning
framework.

Key features:
- Topic-based routing with wildcard pattern matching
- Message priority queues and expiration (TTL)
- At-least-once delivery guarantee with acknowledgments
- Dead letter queue for failed deliveries
- Optional disk persistence for durability
- Subscription filtering by message type and content

Example:
    >>> broker = PubSubBroker()
    >>> broker.subscribe("agent.task.*", "worker1")
    >>> msg_id = broker.publish("agent.task.completed", {"result": "success"}, "worker1")
    >>> messages = broker.poll("worker1")
    >>> broker.acknowledge(messages[0].id, "worker1")

Integration with reasoning framework:
    - ParallelCoordinator can use this for agent coordination
    - CrisisManager can use this for broadcasting alerts
    - CollaborationManager can use this for status updates
"""

import fnmatch
import heapq
import json
import os
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


class MessageStatus(Enum):
    """Status of a message in the system."""
    PENDING = auto()  # Published but not delivered
    DELIVERED = auto()  # Delivered to at least one subscriber
    ACKNOWLEDGED = auto()  # Acknowledged by all subscribers
    EXPIRED = auto()  # TTL expired before delivery
    DEAD_LETTER = auto()  # Failed delivery attempts exhausted


@dataclass
class Message:
    """
    A message in the pub/sub system.

    Attributes:
        id: Unique message identifier
        topic: Topic the message was published to
        payload: Message content as dictionary
        sender: Agent ID of the sender
        timestamp: When the message was created
        ttl_seconds: Time-to-live in seconds (None = no expiration)
        priority: Higher priority messages delivered first (0-10, default 0)
        retry_count: Number of delivery attempts (for dead letter tracking)
        status: Current message status
    """
    id: str
    topic: str
    payload: Dict[str, Any]
    sender: str
    timestamp: datetime
    ttl_seconds: Optional[int] = None
    priority: int = 0
    retry_count: int = 0
    status: MessageStatus = MessageStatus.PENDING

    def is_expired(self) -> bool:
        """Check if message has exceeded its TTL."""
        if self.ttl_seconds is None:
            return False
        elapsed = (datetime.now() - self.timestamp).total_seconds()
        return elapsed > self.ttl_seconds

    def to_dict(self) -> Dict[str, Any]:
        """Serialize message to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['status'] = self.status.name
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Deserialize message from dictionary."""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['status'] = MessageStatus[data['status']]
        return cls(**data)


@dataclass
class Subscription:
    """
    A subscription to a topic pattern.

    Attributes:
        id: Unique subscription identifier
        subscriber_id: Agent ID of the subscriber
        topic_pattern: Topic pattern with wildcard support (e.g., "agent.*", "*.completed")
        filter_fn: Optional function to filter messages (msg -> bool)
        created_at: When subscription was created
        message_count: Number of messages received
    """
    id: str
    subscriber_id: str
    topic_pattern: str
    filter_fn: Optional[Callable[[Message], bool]] = None
    created_at: datetime = field(default_factory=datetime.now)
    message_count: int = 0

    def matches(self, topic: str) -> bool:
        """
        Check if topic matches this subscription's pattern.

        Topic matching rules:
        - "*" matches a single segment (e.g., "agent.*" matches "agent.task" but not "agent.task.error")
        - Use "agent.*.*" to match two segments
        - Exact match required if no wildcards
        """
        # Convert pattern to regex where * only matches non-dot characters
        pattern = self.topic_pattern.replace('.', r'\.')  # Escape dots
        pattern = pattern.replace('*', r'[^.]+')  # * matches one or more non-dots
        pattern = f'^{pattern}$'  # Anchor to start and end

        import re
        return bool(re.match(pattern, topic))

    def should_receive(self, message: Message) -> bool:
        """Check if subscriber should receive this message."""
        if not self.matches(message.topic):
            return False
        if self.filter_fn and not self.filter_fn(message):
            return False
        return True


@dataclass
class DeliveryRecord:
    """
    Record of message delivery to a subscriber.

    Tracks delivery attempts and acknowledgments for at-least-once delivery.
    """
    message_id: str
    subscriber_id: str
    delivered_at: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None


class PubSubBroker:
    """
    A topic-based publish-subscribe message broker.

    Manages message routing, delivery, acknowledgment, and persistence for
    inter-agent communication in the reasoning framework.

    Features:
    - Topic wildcards: "agent.*", "*.completed", "agent.*.error"
    - Priority queues: Higher priority messages delivered first
    - Message expiration: TTL-based automatic cleanup
    - At-least-once delivery: Messages redelivered until acknowledged
    - Dead letter queue: Failed messages for analysis
    - Optional persistence: Durable storage to disk

    Example:
        >>> broker = PubSubBroker(persist_dir=".got/pubsub")
        >>> broker.subscribe("task.*", "worker1")
        >>> broker.publish("task.completed", {"task_id": "123"}, "orchestrator")
        >>> messages = broker.poll("worker1")
        >>> for msg in messages:
        ...     process_message(msg)
        ...     broker.acknowledge(msg.id, "worker1")
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        max_retries: int = 3,
        dead_letter_ttl_days: int = 7,
    ):
        """
        Initialize the pub/sub broker.

        Args:
            persist_dir: Optional directory for message persistence
            max_retries: Maximum delivery attempts before moving to dead letter queue
            dead_letter_ttl_days: How long to keep dead letter messages
        """
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self.max_retries = max_retries
        self.dead_letter_ttl_days = dead_letter_ttl_days

        # Message storage
        self.messages: Dict[str, Message] = {}  # All messages by ID
        self.dead_letters: Dict[str, Message] = {}  # Failed messages

        # Subscription management
        self.subscriptions: Dict[str, Subscription] = {}  # By subscription ID
        self.subscriber_subs: Dict[str, Set[str]] = defaultdict(set)  # subscriber_id -> sub_ids

        # Delivery tracking (for at-least-once guarantee)
        self.pending_delivery: Dict[str, Set[str]] = defaultdict(set)  # subscriber_id -> message_ids
        self.delivery_records: Dict[str, List[DeliveryRecord]] = defaultdict(list)  # message_id -> records

        # Priority queues per subscriber (heapq with negative priority for max-heap)
        self.subscriber_queues: Dict[str, List[Tuple[int, float, str]]] = defaultdict(list)
        # Format: (-priority, timestamp, message_id)

        # Statistics
        self.stats = {
            'messages_published': 0,
            'messages_delivered': 0,
            'messages_acknowledged': 0,
            'messages_expired': 0,
            'messages_dead_lettered': 0,
        }

        # Load persisted state if available
        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._load_state()

    # =========================================================================
    # PUBLISHING
    # =========================================================================

    def publish(
        self,
        topic: str,
        payload: Dict[str, Any],
        sender: str,
        ttl_seconds: Optional[int] = None,
        priority: int = 0,
    ) -> str:
        """
        Publish a message to a topic.

        Args:
            topic: Topic name (e.g., "agent.task.completed")
            payload: Message content as dictionary
            sender: Agent ID of the publisher
            ttl_seconds: Time-to-live in seconds (None = no expiration)
            priority: Message priority (0-10, higher = more urgent, default 0)

        Returns:
            Message ID

        Raises:
            ValueError: If priority is out of range [0, 10]
        """
        if not 0 <= priority <= 10:
            raise ValueError(f"Priority must be 0-10, got {priority}")

        msg_id = str(uuid.uuid4())
        message = Message(
            id=msg_id,
            topic=topic,
            payload=payload,
            sender=sender,
            timestamp=datetime.now(),
            ttl_seconds=ttl_seconds,
            priority=priority,
        )

        self.messages[msg_id] = message
        self.stats['messages_published'] += 1

        # Route to matching subscribers
        self._route_message(message)

        # Persist if enabled
        if self.persist_dir:
            self._persist_message(message)

        return msg_id

    def _route_message(self, message: Message) -> None:
        """Route message to all matching subscribers."""
        for sub_id, subscription in self.subscriptions.items():
            if subscription.should_receive(message):
                subscriber_id = subscription.subscriber_id

                # Add to subscriber's queue (priority queue)
                heapq.heappush(
                    self.subscriber_queues[subscriber_id],
                    (-message.priority, message.timestamp.timestamp(), message.id)
                )

                # Track pending delivery
                self.pending_delivery[subscriber_id].add(message.id)

                # Update subscription stats
                subscription.message_count += 1

    # =========================================================================
    # SUBSCRIBING
    # =========================================================================

    def subscribe(
        self,
        topic_pattern: str,
        subscriber_id: str,
        filter_fn: Optional[Callable[[Message], bool]] = None,
    ) -> Subscription:
        """
        Subscribe to a topic pattern.

        Args:
            topic_pattern: Topic pattern with wildcard support
                          - "*" matches any segment: "agent.*" matches "agent.task"
                          - "**" not supported (use multiple subscriptions)
            subscriber_id: Unique identifier for the subscriber
            filter_fn: Optional function to filter messages (msg -> bool)

        Returns:
            Subscription object

        Examples:
            >>> broker.subscribe("agent.*", "worker1")  # All agent.* topics
            >>> broker.subscribe("*.completed", "logger")  # All *.completed topics
            >>> broker.subscribe("task.error.*", "monitor")  # All task.error.* topics
            >>> broker.subscribe("*", "debug", lambda m: m.priority > 5)  # High priority only
        """
        sub_id = str(uuid.uuid4())
        subscription = Subscription(
            id=sub_id,
            subscriber_id=subscriber_id,
            topic_pattern=topic_pattern,
            filter_fn=filter_fn,
        )

        self.subscriptions[sub_id] = subscription
        self.subscriber_subs[subscriber_id].add(sub_id)

        # Retroactively deliver existing matching messages that are still pending
        self._deliver_existing_messages(subscription)

        return subscription

    def _deliver_existing_messages(self, subscription: Subscription) -> None:
        """Deliver existing messages that match the new subscription."""
        for message in self.messages.values():
            if message.status == MessageStatus.PENDING and subscription.should_receive(message):
                subscriber_id = subscription.subscriber_id
                if message.id not in self.pending_delivery[subscriber_id]:
                    heapq.heappush(
                        self.subscriber_queues[subscriber_id],
                        (-message.priority, message.timestamp.timestamp(), message.id)
                    )
                    self.pending_delivery[subscriber_id].add(message.id)

    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Remove a subscription.

        Args:
            subscription_id: ID of the subscription to remove

        Returns:
            True if subscription existed and was removed, False otherwise
        """
        if subscription_id not in self.subscriptions:
            return False

        subscription = self.subscriptions[subscription_id]
        subscriber_id = subscription.subscriber_id

        # Remove from indexes
        del self.subscriptions[subscription_id]
        self.subscriber_subs[subscriber_id].discard(subscription_id)

        # Clean up if no more subscriptions for this subscriber
        if not self.subscriber_subs[subscriber_id]:
            del self.subscriber_subs[subscriber_id]
            if subscriber_id in self.subscriber_queues:
                del self.subscriber_queues[subscriber_id]
            if subscriber_id in self.pending_delivery:
                del self.pending_delivery[subscriber_id]

        return True

    def get_subscriptions(self, subscriber_id: str) -> List[Subscription]:
        """Get all subscriptions for a subscriber."""
        sub_ids = self.subscriber_subs.get(subscriber_id, set())
        return [self.subscriptions[sub_id] for sub_id in sub_ids]

    # =========================================================================
    # POLLING & DELIVERY
    # =========================================================================

    def poll(
        self,
        subscriber_id: str,
        timeout: float = 0,
        max_messages: int = 100,
    ) -> List[Message]:
        """
        Poll for messages from subscribed topics.

        Args:
            subscriber_id: ID of the subscriber polling
            timeout: Seconds to wait for messages (0 = non-blocking)
            max_messages: Maximum number of messages to return

        Returns:
            List of messages in priority order (highest priority first)

        Note:
            Messages remain in pending state until acknowledged.
            Unacknowledged messages will be redelivered on next poll.
        """
        # Clean up expired messages first
        self._cleanup_expired()

        messages = []
        pending_msgs = self.pending_delivery.get(subscriber_id, set())

        # If timeout > 0, wait for messages
        if timeout > 0 and not pending_msgs:
            start_time = time.time()
            while time.time() - start_time < timeout:
                if self.pending_delivery.get(subscriber_id):
                    break
                time.sleep(0.1)  # Poll every 100ms

        # Rebuild priority queue from pending messages
        # (This ensures unacknowledged messages are redelivered)
        queue = []
        for msg_id in pending_msgs:
            if msg_id in self.messages:
                msg = self.messages[msg_id]
                if not msg.is_expired():
                    heapq.heappush(queue, (-msg.priority, msg.timestamp.timestamp(), msg_id))

        # Retrieve messages from priority queue
        delivered_count = 0
        while queue and delivered_count < max_messages:
            _, _, msg_id = heapq.heappop(queue)

            # Skip if message no longer exists
            if msg_id not in self.messages:
                self.pending_delivery[subscriber_id].discard(msg_id)
                continue

            message = self.messages[msg_id]

            if message.is_expired():
                self._handle_expired_message(message, subscriber_id)
                continue

            # Deliver message
            messages.append(message)
            delivered_count += 1

            # Record delivery (only if not already delivered to this subscriber)
            existing_records = [
                r for r in self.delivery_records.get(msg_id, [])
                if r.subscriber_id == subscriber_id
            ]
            if not existing_records:
                record = DeliveryRecord(
                    message_id=msg_id,
                    subscriber_id=subscriber_id,
                )
                self.delivery_records[msg_id].append(record)

            if message.status == MessageStatus.PENDING:
                message.status = MessageStatus.DELIVERED
                self.stats['messages_delivered'] += 1

        return messages

    def acknowledge(self, message_id: str, subscriber_id: str) -> bool:
        """
        Acknowledge receipt of a message.

        Args:
            message_id: ID of the message to acknowledge
            subscriber_id: ID of the subscriber acknowledging

        Returns:
            True if acknowledgment was recorded, False if message not found

        Note:
            Once all subscribers acknowledge, message moves to ACKNOWLEDGED status
            and can be cleaned up.
        """
        if message_id not in self.messages:
            return False

        # Find delivery record
        records = self.delivery_records.get(message_id, [])
        for record in records:
            if record.subscriber_id == subscriber_id and not record.acknowledged:
                record.acknowledged = True
                record.acknowledged_at = datetime.now()
                break

        # Remove from pending
        self.pending_delivery[subscriber_id].discard(message_id)

        # Check if all subscribers have acknowledged
        message = self.messages[message_id]
        if all(r.acknowledged for r in records):
            message.status = MessageStatus.ACKNOWLEDGED
            self.stats['messages_acknowledged'] += 1

            # Optionally clean up acknowledged messages
            # (kept for now for audit trail)

        return True

    # =========================================================================
    # DEAD LETTER QUEUE
    # =========================================================================

    def get_dead_letters(
        self,
        topic: Optional[str] = None,
        sender: Optional[str] = None,
    ) -> List[Message]:
        """
        Retrieve messages from the dead letter queue.

        Args:
            topic: Optional topic filter (exact match)
            sender: Optional sender filter

        Returns:
            List of dead letter messages matching filters
        """
        messages = list(self.dead_letters.values())

        if topic:
            messages = [m for m in messages if m.topic == topic]

        if sender:
            messages = [m for m in messages if m.sender == sender]

        return sorted(messages, key=lambda m: m.timestamp, reverse=True)

    def retry_dead_letter(self, message_id: str, new_ttl_seconds: Optional[int] = 300) -> bool:
        """
        Retry delivery of a dead letter message.

        Args:
            message_id: ID of the dead letter message
            new_ttl_seconds: New TTL for the message (default: 300 seconds, None = no expiration)

        Returns:
            True if message was moved back to active queue, False if not found
        """
        if message_id not in self.dead_letters:
            return False

        message = self.dead_letters[message_id]
        del self.dead_letters[message_id]

        # Reset retry count, status, and TTL
        message.retry_count = 0
        message.status = MessageStatus.PENDING
        message.ttl_seconds = new_ttl_seconds
        message.timestamp = datetime.now()  # Reset timestamp for new TTL

        # Re-add to active messages
        self.messages[message_id] = message

        # Re-route to subscribers
        self._route_message(message)

        return True

    def _move_to_dead_letter(self, message: Message) -> None:
        """Move a message to the dead letter queue."""
        message.status = MessageStatus.DEAD_LETTER
        self.dead_letters[message.id] = message
        if message.id in self.messages:
            del self.messages[message.id]
        self.stats['messages_dead_lettered'] += 1

    # =========================================================================
    # CLEANUP & MAINTENANCE
    # =========================================================================

    def _cleanup_expired(self) -> None:
        """Remove expired messages from queues."""
        expired_ids = []

        for msg_id, message in self.messages.items():
            if message.is_expired():
                expired_ids.append(msg_id)

        for msg_id in expired_ids:
            message = self.messages[msg_id]
            message.status = MessageStatus.EXPIRED
            self.stats['messages_expired'] += 1

            # Remove from all subscriber queues
            for subscriber_id in self.pending_delivery:
                self.pending_delivery[subscriber_id].discard(msg_id)

            # Move to dead letter queue for audit
            self._move_to_dead_letter(message)

    def _handle_expired_message(self, message: Message, subscriber_id: str) -> None:
        """Handle an expired message during poll."""
        self.pending_delivery[subscriber_id].discard(message.id)
        message.status = MessageStatus.EXPIRED
        self.stats['messages_expired'] += 1
        self._move_to_dead_letter(message)

    def cleanup_old_messages(
        self,
        acknowledged_ttl_hours: int = 24,
        dead_letter_ttl_days: Optional[int] = None,
    ) -> Dict[str, int]:
        """
        Clean up old acknowledged and dead letter messages.

        Args:
            acknowledged_ttl_hours: Remove acknowledged messages older than this
            dead_letter_ttl_days: Remove dead letters older than this (None = use default)

        Returns:
            Dictionary with counts of removed messages by category
        """
        if dead_letter_ttl_days is None:
            dead_letter_ttl_days = self.dead_letter_ttl_days

        now = datetime.now()
        removed = {'acknowledged': 0, 'dead_letters': 0}

        # Clean up acknowledged messages
        ack_cutoff = now - timedelta(hours=acknowledged_ttl_hours)
        to_remove = [
            msg_id for msg_id, msg in self.messages.items()
            if msg.status == MessageStatus.ACKNOWLEDGED and msg.timestamp < ack_cutoff
        ]
        for msg_id in to_remove:
            del self.messages[msg_id]
            if msg_id in self.delivery_records:
                del self.delivery_records[msg_id]
            removed['acknowledged'] += 1

        # Clean up old dead letters
        dl_cutoff = now - timedelta(days=dead_letter_ttl_days)
        to_remove = [
            msg_id for msg_id, msg in self.dead_letters.items()
            if msg.timestamp < dl_cutoff
        ]
        for msg_id in to_remove:
            del self.dead_letters[msg_id]
            removed['dead_letters'] += 1

        return removed

    # =========================================================================
    # STATISTICS & INSPECTION
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """
        Get broker statistics.

        Returns:
            Dictionary with message counts, subscriber counts, and delivery stats
        """
        pending_count = sum(
            1 for msg in self.messages.values()
            if msg.status == MessageStatus.PENDING
        )

        return {
            **self.stats,
            'active_messages': len(self.messages),
            'pending_messages': pending_count,
            'dead_letter_messages': len(self.dead_letters),
            'active_subscriptions': len(self.subscriptions),
            'active_subscribers': len(self.subscriber_subs),
        }

    def get_message(self, message_id: str) -> Optional[Message]:
        """Get a message by ID (checks both active and dead letter queues)."""
        return self.messages.get(message_id) or self.dead_letters.get(message_id)

    def list_topics(self) -> Set[str]:
        """Get all topics that have been published to."""
        topics = set()
        for msg in self.messages.values():
            topics.add(msg.topic)
        for msg in self.dead_letters.values():
            topics.add(msg.topic)
        return topics

    def get_pending_count(self, subscriber_id: str) -> int:
        """Get number of pending messages for a subscriber."""
        return len(self.pending_delivery.get(subscriber_id, set()))

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def _persist_message(self, message: Message) -> None:
        """Persist a message to disk."""
        if not self.persist_dir:
            return

        msg_file = self.persist_dir / f"msg_{message.id}.json"
        with open(msg_file, 'w') as f:
            json.dump(message.to_dict(), f, indent=2)

    def _load_state(self) -> None:
        """Load persisted messages from disk."""
        if not self.persist_dir or not self.persist_dir.exists():
            return

        # Load messages
        for msg_file in self.persist_dir.glob("msg_*.json"):
            try:
                with open(msg_file) as f:
                    data = json.load(f)
                    message = Message.from_dict(data)
                    self.messages[message.id] = message
            except Exception as e:
                print(f"Warning: Failed to load message from {msg_file}: {e}")

    def save_state(self) -> None:
        """Save current state to disk."""
        if not self.persist_dir:
            return

        # Clear old message files
        for msg_file in self.persist_dir.glob("msg_*.json"):
            msg_file.unlink()

        # Save active messages
        for message in self.messages.values():
            self._persist_message(message)

        # Save dead letters
        for message in self.dead_letters.values():
            self._persist_message(message)

        # Save metadata
        meta_file = self.persist_dir / "broker_meta.json"
        with open(meta_file, 'w') as f:
            json.dump({
                'stats': self.stats,
                'max_retries': self.max_retries,
                'dead_letter_ttl_days': self.dead_letter_ttl_days,
            }, f, indent=2)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_topic_filter(allowed_senders: Optional[Set[str]] = None) -> Callable[[Message], bool]:
    """
    Create a filter function that only accepts messages from specific senders.

    Args:
        allowed_senders: Set of sender IDs to accept

    Returns:
        Filter function for use in subscribe()

    Example:
        >>> filter_fn = create_topic_filter({"orchestrator", "monitor"})
        >>> broker.subscribe("task.*", "worker1", filter_fn=filter_fn)
    """
    def filter_fn(message: Message) -> bool:
        if allowed_senders is None:
            return True
        return message.sender in allowed_senders

    return filter_fn


def create_payload_filter(required_keys: Set[str]) -> Callable[[Message], bool]:
    """
    Create a filter function that only accepts messages with specific payload keys.

    Args:
        required_keys: Set of required keys in message payload

    Returns:
        Filter function for use in subscribe()

    Example:
        >>> filter_fn = create_payload_filter({"task_id", "status"})
        >>> broker.subscribe("task.updates", "monitor", filter_fn=filter_fn)
    """
    def filter_fn(message: Message) -> bool:
        return all(key in message.payload for key in required_keys)

    return filter_fn
