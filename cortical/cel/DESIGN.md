# Cognitive Event Lattice (CEL) Design Document

## Overview

The Cognitive Event Lattice is a self-referential, self-maintaining cognitive substrate for machine reasoning. It provides a foundation for systems that need to reason about themselves while modifying themselves, without falling into paradoxes or inconsistencies.

## The Problem

Traditional database-centric systems face several challenges when used as a cognitive substrate:

1. **Self-Reference Paradox**: A system querying itself while modifying itself creates race conditions and inconsistent views.
2. **Temporal Blindness**: Standard CRUD operations lose history, making it impossible to ask "what did the system know when it made decision X?"
3. **Rigid Schemas**: Database schemas are difficult to evolve without data migration.
4. **Merge Conflicts**: Git-tracked JSON files create conflicts when multiple agents work in parallel.
5. **Duplicate Data**: Entity-centric storage creates redundancy (ID in filename AND content, repeated timestamps, etc.)

## The Solution: Event Sourcing + Temporal References

### Core Insight

> **Events are primary. Entities are derived.**

Instead of storing entities (tasks, decisions, sprints), we store *events* - immutable records of observations and intentions. Entities are computed projections of events, materialized on demand.

This is similar to:
- Git (commits are events, working tree is derived)
- Redux (actions are events, state is derived)
- Double-entry bookkeeping (transactions are events, balances are derived)

### The Double Helix Architecture

```
    ╭─────────╮         ╭─────────╮
    │ WISDOM  │─────────│ SANITY  │
    │ Strand  │  ╲   ╱  │ Strand  │
    ╰────┬────╯   ╲ ╱   ╰────┬────╯
         │         ╳         │
         │        ╱ ╲        │
    ╭────┴────╮  ╱   ╲  ╭────┴────╯
    │  DAG    │─────────│ Health  │
    │ Events  │  ╲   ╱  │ Monitor │
    ╰────┬────╯   ╲ ╱   ╰────┬────╯
         │         ╳         │
         │        ╱ ╲        │
    ╭────┴────╮  ╱   ╲  ╭────┴────╯
    │Material-│─────────│Migration│
    │  izer   │  ╲   ╱  │ Engine  │
    ╰────┬────╯   ╲ ╱   ╰────┬────╯
         │         ╳         │
         │        ╱ ╲        │
    ╭────┴────╮  ╱   ╲  ╭────┴────╯
    │Semantic │─────────│Compactor│
    │ Index   │         │         │
    ╰─────────╯         ╰─────────╯
```

**WISDOM Strand** (cortical/cel/wisdom/)
- Knowledge, memory, relationships - *what the system knows*
- Events, DAG, Materialization, Semantic indexing

**SANITY Strand** (cortical/cel/sanity/)
- Validation, health, evolution - *keeping the system coherent*
- Health monitoring, Migration, Compaction

The two strands are intertwined:
- Wisdom without sanity leads to corruption (inconsistent state)
- Sanity without wisdom leads to empty process (no actual knowledge)

## Key Concepts

### 1. Events (Immutable Records)

Events are immutable records of what happened. Each event has:

```python
@dataclass(frozen=True)
class CognitiveEvent:
    timestamp: str                    # When it happened
    event_type: EventType             # What kind of event
    causal_parents: tuple[str, ...]   # Content hashes of parent events
    content: Dict[str, Any]           # Event-specific data
    concepts: tuple[str, ...]         # Semantic tags for indexing

    @property
    def id(self) -> str:
        # SHA256 hash of content - content-addressed ID
```

Event types:
- **Observation**: External fact observed by the system
- **Intention**: Task/goal to be accomplished
- **Fulfillment**: Intention was completed
- **Invalidation**: Entity is no longer valid
- **Compaction**: Events were compressed
- **MetaCognition**: System observing itself

### 2. Merkle DAG (Directed Acyclic Graph)

Events form a DAG through `causal_parents`:

```
    [Event A]
         │
         ▼
    [Event B] ◄─────┐
         │          │
         ▼          │
    [Event C]  [Event D]
         │          │
         └────┬─────┘
              ▼
         [Event E]  (merge)
```

Properties:
- **Content-addressed**: Event ID is hash of content (Merkle root)
- **Immutable**: Same content = same ID forever
- **Causally ordered**: Parents happened before children
- **Verifiable**: Can recompute hashes to verify integrity

### 3. Temporal References (Solving Self-Reference)

The key insight that enables self-reference without paradox:

> **Reference "the entity at event E", not "the entity"**

```python
@dataclass(frozen=True)
class TemporalReference:
    entity_id: str      # What entity
    horizon: EventHorizon  # As of which event

    def resolve(self, lattice) -> Entity:
        return lattice.materialize(self.entity_id, at=self.horizon)
```

This means:
- A task can reference "the system configuration as it was when this task was created"
- Even if configuration changes later, the reference is stable
- No race conditions, no paradoxes

### 4. Deferred References (Task Dependencies)

For tasks that depend on other tasks completing:

```python
@dataclass
class DeferredReference:
    entity_id: str
    after: list[str]  # Event IDs that must complete first

    def resolve_now(self, lattice) -> None:
        # Captures current horizon as resolution point
        self.resolved_horizon = lattice.current_horizon
```

Lifecycle:
1. Created with `mode=DEFERRED`, no horizon yet
2. Dependencies complete
3. `resolve_now()` called, captures horizon
4. Now behaves like TemporalReference (stable)

### 5. Materialization (Events → Entities)

Entities don't exist in storage. They are computed projections:

```python
def materialize(entity_id: str, at: EventHorizon = None) -> Entity:
    state = None
    for event in events_up_to(at):
        if event_affects(entity_id):
            state = reducer(state, event)
    return state
```

This is like Redux reducers or functional fold/reduce.

Benefits:
- **Temporal queries**: "What was task X at time T?"
- **No stale state**: Always computed from source of truth
- **Caching**: Can cache materialized entities, invalidate on new events

### 6. Semantic Indexing (Fast Lookup)

For fast "does X exist?" and "find events about X":

```python
class BloomSemanticIndex:
    _bloom: BloomFilter        # O(1) probabilistic existence
    _inverted: InvertedIndex   # Term → event mapping

    def probably_contains(self, concept: str) -> bool:
        return concept in self._bloom  # May have false positives

    def search(self, query: str) -> List[str]:
        # Full search using inverted index
```

Trade-off: Speed vs accuracy
- Bloom filter: Fast but probabilistic
- Inverted index: Exact but slower

### 7. Dependency Injection (Adaptability)

All components are protocol-based:

```python
class EventStore(Protocol):
    def append(self, event: CognitiveEvent) -> MerkleRoot: ...
    def get(self, event_id: str) -> Optional[CognitiveEvent]: ...
    def iterate(self) -> Iterator[CognitiveEvent]: ...
```

Container manages wiring:

```python
container = Container()
container.register(EventStore, FileSystemEventStore)
container.register(Materializer, CachingMaterializer)

lattice = create_lattice(container, path=".got")
```

Benefits:
- Swap implementations without changing consumers
- Test with mocks
- Evolve without breaking

## Migration from GoT

### Phase 1: Bridge (Current)

```
┌──────────────┐      ┌─────────────────┐
│   GoT        │ ◄─── │  GotBridgeStore │
│  .got/       │      │                 │
│  entities/   │      │  Reads GoT      │
│              │ ───► │  Writes Both    │
└──────────────┘      └─────────────────┘
                              │
                              ▼
                      ┌─────────────────┐
                      │      CEL        │
                      │  .cel/events/   │
                      └─────────────────┘
```

- Read from GoT, convert to CEL events
- Write to both GoT and CEL
- Existing tools still work

### Phase 2: Parallel

- Read primarily from CEL
- Write to both
- GoT becomes backup

### Phase 3: CEL Only

- Read/write CEL only
- GoT retired

## Compaction Strategies

Over time, events accumulate. Compaction reduces storage while preserving meaning:

### Time Window Compaction

```
Events in 24h window          After compaction
─────────────────────         ──────────────────
Task created     ─┐
Task updated     ─┤
Task updated     ─┼──────►     Compaction summary +
Task updated     ─┤            Final state
Task completed   ─┘
```

### Semantic Compaction

Events with >80% concept overlap are merged:

```
"Implement auth"  ─┐
"Add login page"  ─┼──────►  "Auth feature" (merged)
"Auth tests"      ─┘
```

### Causal Chain Compaction

Long A→B→C→D chains become A→D with summary:

```
A ──► B ──► C ──► D    After:  A ────► D
                               (summary of B,C)
```

## Health Monitoring

The system watches itself:

```python
health = lattice.health_monitor.check()

print(health.status)  # HEALTHY, DEGRADED, UNHEALTHY, CRITICAL
print(health.metrics)  # DAG consistency, storage size, etc.
print(health.recommendations)  # "Consider compaction", etc.
```

Health checks are themselves MetaCognition events, creating recursive self-awareness.

## Directory Structure

```
cortical/cel/
├── __init__.py          # Public API
├── container.py         # DI container and factories
├── DESIGN.md            # This document
│
├── core/                # Core abstractions
│   ├── __init__.py
│   ├── protocols.py     # Interface definitions
│   ├── events.py        # Event types
│   └── references.py    # Temporal reference types
│
├── wisdom/              # Knowledge strand
│   ├── __init__.py
│   ├── dag.py           # Merkle DAG implementation
│   ├── materializer.py  # Event → Entity projection
│   └── semantic.py      # Bloom filter, inverted index
│
├── sanity/              # Health strand
│   ├── __init__.py
│   ├── health.py        # Health monitoring
│   ├── migration.py     # Schema evolution
│   └── compaction.py    # Storage compression
│
└── adapters/            # External integrations
    ├── __init__.py
    └── got.py           # GoT bridge adapter
```

## Usage Examples

### Creating a Task

```python
from cortical.cel import Container, create_lattice, Intention

container = Container()
lattice = create_lattice(container, path=".got")

# Create task with temporal reference to current state
task = Intention(
    title="Optimize storage format",
    description="Reduce JSON file sizes",
    priority="high",
    references_at=lattice.current_horizon,  # Snapshot of current state
)

# Append as event
event = task.to_event()
root = lattice.event_store.append(event)
print(f"Task created: {root.value[:8]}")
```

### Querying Past State

```python
# Get task as it was at a specific point
old_task = lattice.materializer.materialize(
    "T-xxx",
    at=some_horizon,  # Past event horizon
)

# Even after many changes, this returns the same result
```

### Running Health Checks

```python
health = lattice.health_monitor
report = health.check()

if report.status != HealthStatus.HEALTHY:
    print("Issues detected:")
    for issue in report.issues:
        print(f"  - {issue}")

    print("Recommendations:")
    for rec in report.recommendations:
        print(f"  - {rec}")
```

### Compacting Storage

```python
from cortical.cel.sanity.compaction import TimeWindowCompactor

compactor = TimeWindowCompactor(
    lattice.event_store,
    window_size=timedelta(hours=24),
    min_age=timedelta(days=7),
)

if compactor.should_compact():
    result = compactor.compact()
    print(f"Compacted {result.original_count} → {result.compacted_count}")
    print(f"Saved {result.bytes_saved} bytes")
```

## Design Decisions

### Why Events Instead of Entities?

| Approach | Pros | Cons |
|----------|------|------|
| Entity-centric | Simple queries | Loses history, merge conflicts |
| Event-sourcing | Full history, no conflicts | More complex queries |

We chose events because cognitive systems need to reason about their own history.

### Why Content-Addressed IDs?

- Same content = same ID everywhere
- No coordination needed between agents
- Integrity verification built in
- Natural deduplication

### Why Temporal References?

The only way to reference a changing system without paradox:
- "The system" is ambiguous
- "The system at event E" is concrete and stable

### Why Bloom Filters?

Trade-off: False positives vs. speed

- Fast "probably exists" checks
- Falls back to exact search if needed
- Perfect for "does concept X exist?" queries

## Future Directions

1. **Embeddings**: Vector embeddings for semantic similarity
2. **Distributed**: Multi-agent consensus on event ordering
3. **Pruning**: Age-based event archival
4. **Visualization**: DAG viewer and time-travel debugging
5. **PRISM Integration**: Connecting to the reasoning framework
