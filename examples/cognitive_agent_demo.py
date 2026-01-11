#!/usr/bin/env python3
"""
Cognitive Agent Demo
====================

Demonstrates the 7-layer bio-inspired cognitive architecture:
1. Knowledge (CognitiveGraph) - Hypergraph with links as atoms
2. Attention (STI/LTI) - Short/long-term importance with decay
3. Working Memory - LRU bounded buffer (Cowan's 4+-1)
4. Prediction - Co-occurrence based associative learning
5. Goals - Control theory with urgency = importance * (1 - progress)
6. Exploration - Epsilon-greedy adaptation
7. Episodic Memory - Experience storage and replay

Uses event hooks to show all operations as they happen.

Usage:
    python examples/cognitive_agent_demo.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cortical.cognitive import (
    CognitiveAgent,
    CognitiveGraph,
    AtomType,
    TruthValue,
    Goal,
    EventType,
    CognitiveEvent,
)


# ANSI color codes for pretty output
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def colorize(text: str, color: str) -> str:
    """Wrap text in ANSI color codes."""
    return f"{color}{text}{Colors.RESET}"


def print_header(title: str):
    """Print a formatted section header."""
    width = 70
    print()
    print(colorize("=" * width, Colors.BOLD))
    print(colorize(f" {title}", Colors.BOLD + Colors.CYAN))
    print(colorize("=" * width, Colors.BOLD))
    print()


def print_subheader(title: str):
    """Print a formatted subsection header."""
    print()
    print(colorize(f"--- {title} ---", Colors.YELLOW))
    print()


# Event type to color mapping
EVENT_COLORS = {
    # Working Memory events
    EventType.ATOM_LOADED: Colors.BLUE,
    EventType.ATOM_EVICTED: Colors.YELLOW,
    # Attention events
    EventType.ATTENTION_FOCUSED: Colors.CYAN,
    EventType.ATTENTION_DECAYED: Colors.DIM,
    # Prediction events
    EventType.PREDICTION_MADE: Colors.CYAN,
    EventType.SURPRISE_RECORDED: Colors.RED,
    # Goal events
    EventType.GOAL_ADDED: Colors.GREEN,
    EventType.GOAL_PROGRESS: Colors.BLUE,
    EventType.GOAL_COMPLETED: Colors.GREEN + Colors.BOLD,
    # Exploration events
    EventType.EXPLORE_DECISION: Colors.YELLOW,
    EventType.STRATEGY_ADAPTED: Colors.CYAN,
    # Agent lifecycle
    EventType.STEP_STARTED: Colors.DIM,
    EventType.STEP_COMPLETED: Colors.DIM,
    # Episodic Memory events
    EventType.EPISODE_STORED: Colors.GREEN,
    EventType.EPISODE_EVICTED: Colors.YELLOW,
    EventType.EPISODE_RETRIEVED: Colors.CYAN,
    EventType.EXPERIENCE_REPLAY: Colors.BLUE,
}


def format_event(event: CognitiveEvent) -> str:
    """Format an event for display."""
    color = EVENT_COLORS.get(event.event_type, Colors.RESET)
    event_name = event.event_type.name

    # Extract key data for display
    data_str = ""
    if "atom_id" in event.data:
        data_str = f"atom={event.data['atom_id']}"
    elif "node_id" in event.data:
        data_str = f"node={event.data['node_id']}"
    elif "goal_id" in event.data:
        data_str = f"goal={event.data['goal_id']}"
    elif "predicted" in event.data:
        data_str = f"predicted={event.data['predicted'][:3]}"
    elif "surprise" in event.data:
        data_str = f"surprise={event.data['surprise']:.2f}"
    elif "episode_step" in event.data:
        data_str = f"step={event.data['episode_step']}"
    elif "epsilon" in event.data:
        data_str = f"epsilon={event.data['epsilon']:.2f}"
    elif "episodes_replayed" in event.data:
        data_str = f"count={event.data['episodes_replayed']}"

    if data_str:
        return colorize(f"  [{event_name}] {data_str}", color)
    return colorize(f"  [{event_name}]", color)


def create_event_listener():
    """Create an event listener that prints events."""
    def listener(event: CognitiveEvent):
        print(format_event(event))
    return listener


def demo_knowledge_layer(agent: CognitiveAgent):
    """Demonstrate the knowledge graph layer."""
    print_subheader("Layer 1: Knowledge Graph")
    print("Adding nodes and links to the hypergraph...")
    print("(Links are first-class atoms - they can be linked to!)")
    print()

    # Create concept nodes
    cat = agent.graph.node("cat")
    animal = agent.graph.node("animal")
    mammal = agent.graph.node("mammal")
    pet = agent.graph.node("pet")

    # Create inheritance links with truth values
    cat_is_animal = agent.graph.link(
        AtomType.INHERITANCE,
        [cat, animal],
        TruthValue(0.99, 0.95)  # High confidence
    )
    cat_is_mammal = agent.graph.link(
        AtomType.INHERITANCE,
        [cat, mammal],
        TruthValue(0.99, 0.90)
    )
    cat_is_pet = agent.graph.link(
        AtomType.INHERITANCE,
        [cat, pet],
        TruthValue(0.80, 0.70)  # Most but not all cats
    )

    # Meta-link: linking to a link (demonstrating hypergraph)
    print()
    print(colorize("Creating meta-link (link pointing to another link)...", Colors.DIM))
    evidence = agent.graph.node("veterinary_study")
    supports = agent.graph.link(
        AtomType.EVALUATION,
        [evidence, cat_is_mammal],
        TruthValue(0.95, 0.85)
    )

    print()
    all_atoms = agent.graph._storage.all_atoms()
    print(f"Graph now has {len(all_atoms)} atoms")


def demo_attention_layer(agent: CognitiveAgent):
    """Demonstrate attention dynamics."""
    print_subheader("Layer 2: Attention (STI/LTI)")
    print("Boosting attention on 'cat' and running decay...")
    print()

    # Boost attention
    agent.attend("cat", amount=50)

    # Show current attention
    cat_atom = agent.graph.get_node("cat")
    if cat_atom:
        print(f"  'cat' STI after boost: {cat_atom.sti}")

    # Run a step to show decay
    print()
    print("Running agent step (causes attention decay)...")
    agent.step()

    if cat_atom:
        print(f"  'cat' STI after decay: {cat_atom.sti}")


def demo_working_memory_layer(agent: CognitiveAgent):
    """Demonstrate working memory dynamics."""
    print_subheader("Layer 3: Working Memory (LRU Bounded)")
    print(f"Working memory capacity: {agent.working_memory.capacity}")
    print()

    # Add items to working memory by attending to them
    concepts = ["dog", "bird", "fish", "snake", "lizard", "frog", "spider"]

    print("Attending to multiple concepts to fill working memory...")
    for concept in concepts:
        agent.graph.node(concept)  # Create if not exists
        agent.attend(concept, amount=30)

    # Show what's in working memory
    print()
    wm_items = agent.working_memory._slots
    print(f"Working memory contents ({len(wm_items)} items):")
    for atom in wm_items:
        name = atom.name if atom.name else atom.id[:8]
        print(f"  - {name}")


def demo_prediction_layer(agent: CognitiveAgent):
    """Demonstrate co-occurrence prediction."""
    print_subheader("Layer 4: Prediction (Co-occurrence Learning)")
    print("Recording co-occurrences and making predictions...")
    print()

    # Record some co-occurrences (using atom IDs - simulating learning from experience)
    agent.predictor.record_co_occurrence("cat", "meow")
    agent.predictor.record_co_occurrence("cat", "whiskers")
    agent.predictor.record_co_occurrence("cat", "purr")
    agent.predictor.record_co_occurrence("dog", "bark")
    agent.predictor.record_co_occurrence("dog", "tail")

    # Strengthen cat associations (repeated exposure)
    for _ in range(5):
        agent.predictor.record_co_occurrence("cat", "meow")
        agent.predictor.record_co_occurrence("cat", "purr")

    # Get predictions using atoms (the predict method needs Atom objects)
    # For demo purposes, show the raw co-occurrence data
    print("Co-occurrence associations for 'cat':")
    if "cat" in agent.predictor._co_occurrences:
        for target, weight in sorted(
            agent.predictor._co_occurrences["cat"].items(),
            key=lambda x: -x[1]
        )[:5]:
            print(f"  {target}: {weight:.1f}")

    print()
    print("Co-occurrence associations for 'dog':")
    if "dog" in agent.predictor._co_occurrences:
        for target, weight in sorted(
            agent.predictor._co_occurrences["dog"].items(),
            key=lambda x: -x[1]
        )[:5]:
            print(f"  {target}: {weight:.1f}")


def demo_surprise_and_learning(agent: CognitiveAgent):
    """Demonstrate surprise detection and learning."""
    print_subheader("Surprise Detection & Learning")
    print("Encountering unexpected outcomes triggers learning...")
    print()

    # Get what we'd predict after "cat" (from co-occurrences)
    predicted_ids = []
    if "cat" in agent.predictor._co_occurrences:
        predicted_ids = list(agent.predictor._co_occurrences["cat"].keys())[:3]

    print(f"Expected after 'cat': {predicted_ids}")
    print("Actual outcome: 'bark' (unexpected!)")

    # Learn from surprise (using the actual API)
    surprise = agent.learn_from_surprise(
        context_ids=["cat"],
        actual_id="bark",
    )

    # Show that learning occurred
    print()
    print("Surprise tracker stats:")
    print(f"  Returned surprise: {surprise:.2f}")
    print(f"  Mean surprise: {agent.surprise_tracker.mean_surprise():.2f}")


def demo_goals_layer(agent: CognitiveAgent):
    """Demonstrate goal tracking with urgency."""
    print_subheader("Layer 5: Goals (Control Theory)")
    print("Adding goals with importance and tracking progress...")
    print()

    # Add goals using Goal objects
    agent.goals.add_goal(Goal(
        id="learn_animal_taxonomy",
        description="Learn the animal taxonomy",
        target_state=1.0,
        importance=0.8
    ))
    agent.goals.add_goal(Goal(
        id="find_pet_store",
        description="Find a pet store",
        target_state=1.0,
        importance=0.5
    ))
    agent.goals.add_goal(Goal(
        id="feed_cat",
        description="Feed the cat",
        target_state=1.0,
        importance=0.9
    ))

    # Show goals (sorted by urgency internally)
    print("Current goals:")
    for goal in agent.goals.get_active_goals():
        print(f"  {goal.id}: importance={goal.importance:.1f}, "
              f"progress={goal.progress:.1f}, urgency={goal.urgency:.2f}")

    # Update progress (state goes from 0 towards target)
    print()
    print("Updating 'feed_cat' progress to 50%...")
    agent.goals.update_progress("feed_cat", current_state=0.5)

    print()
    print("Goals after update:")
    for goal in agent.goals.get_active_goals():
        print(f"  {goal.id}: progress={goal.progress:.1f}, urgency={goal.urgency:.2f}")


def demo_exploration_layer(agent: CognitiveAgent):
    """Demonstrate exploration/exploitation balance."""
    print_subheader("Layer 6: Exploration (Epsilon-Greedy)")
    print(f"Initial epsilon: {agent.exploration.epsilon:.2f}")
    print()

    # Simulate some decisions
    explore_count = 0
    exploit_count = 0

    print("Making 20 explore/exploit decisions...")
    for _ in range(20):
        if agent.exploration.should_explore():
            explore_count += 1
        else:
            exploit_count += 1

    print(f"  Explored: {explore_count} times")
    print(f"  Exploited: {exploit_count} times")

    # Show epsilon adaptation
    print()
    print("Recording successes (epsilon will decay)...")
    for _ in range(10):
        agent.exploration.record_success()

    print(f"Epsilon after successes: {agent.exploration.epsilon:.2f}")

    print()
    print("Recording failures (epsilon will increase)...")
    for _ in range(5):
        agent.exploration.record_failure()

    print(f"Epsilon after failures: {agent.exploration.epsilon:.2f}")


def demo_episodic_memory_layer(agent: CognitiveAgent):
    """Demonstrate episodic memory and experience replay."""
    print_subheader("Layer 7: Episodic Memory (Experience Replay)")
    print("Storing experiences and replaying them for learning...")
    print()

    # The surprising event earlier should have been stored
    print(f"Episodes stored: {len(agent.episodic_memory._episodes)}")

    # Generate more surprising experiences
    print()
    print("Generating more surprising experiences...")
    for i in range(5):
        context = [f"context_{i}"]
        agent.learn_from_surprise(
            context_ids=context,
            actual_id=f"unexpected_{i}",
        )

    print(f"Episodes after learning: {len(agent.episodic_memory._episodes)}")

    # Experience replay
    print()
    print("Running experience replay (reinforces learning)...")
    n_replayed = agent.experience_replay(n_episodes=3, prioritized=True)
    print(f"Replayed {n_replayed} episodes")

    # Content-addressable retrieval
    print()
    print("Recalling episodes similar to ['context_0']...")
    similar = agent.recall_similar(["context_0"], top_k=2)
    for ep in similar:
        print(f"  Step {ep.step}: outcome={ep.outcome_id}, surprise={ep.surprise:.2f}")


def demo_integrated_step(agent: CognitiveAgent):
    """Demonstrate the integrated cognitive loop."""
    print_subheader("Integrated Cognitive Loop")
    print("Running multiple integrated steps...")
    print()

    for i in range(3):
        print(f"Step {i+1}:")
        metrics = agent.step()
        print(f"  Working memory size: {metrics['working_memory_size']}")
        print(f"  Epsilon: {metrics['epsilon']:.2f}")
        print()


def demo_persistence(agent: CognitiveAgent):
    """Demonstrate save/load capability."""
    print_subheader("Persistence")
    print("Agent state can be saved and restored...")
    print()

    # Get current state
    state = agent.to_dict()
    all_atoms = agent.graph._storage.all_atoms()
    print(f"State version: {state['version']}")
    print(f"Total atoms: {len(all_atoms)}")
    print(f"Episodic memories: {len(agent.episodic_memory._episodes)}")

    # Create new agent from state
    print()
    print("Creating new agent from saved state...")
    restored = CognitiveAgent.from_dict(state)
    restored_atoms = restored.graph._storage.all_atoms()
    print(f"Restored atoms: {len(restored_atoms)}")
    print(f"Restored memories: {len(restored.episodic_memory._episodes)}")


def main():
    print(colorize("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║              COGNITIVE AGENT - 7 LAYER ARCHITECTURE DEMO             ║
    ║                                                                      ║
    ║  Bio-inspired cognition with attention, memory, prediction, goals   ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """, Colors.CYAN))

    # Create agent with small capacity for demo
    print("Creating CognitiveAgent with event hooks...")
    agent = CognitiveAgent(
        working_memory_size=5,
        episodic_memory_size=100,
    )

    # Subscribe to events
    event_listener = create_event_listener()
    for event_type in EventType:
        agent.events.subscribe(event_type, event_listener)

    print(colorize("Event hooks active - watching all cognitive events", Colors.GREEN))

    # Run all demonstrations
    print_header("LAYER DEMONSTRATIONS")

    demo_knowledge_layer(agent)
    demo_attention_layer(agent)
    demo_working_memory_layer(agent)
    demo_prediction_layer(agent)
    demo_surprise_and_learning(agent)
    demo_goals_layer(agent)
    demo_exploration_layer(agent)
    demo_episodic_memory_layer(agent)
    demo_integrated_step(agent)
    demo_persistence(agent)

    # Summary
    print_header("DEMO COMPLETE")
    print("""
This demo showed all 7 layers of the cognitive architecture:

  1. KNOWLEDGE      Hypergraph where links are first-class atoms
  2. ATTENTION      STI/LTI with decay - what deserves focus
  3. WORKING MEMORY LRU bounded buffer - active concepts
  4. PREDICTION     Co-occurrence learning - what comes next
  5. GOALS          Control theory - urgency drives action
  6. EXPLORATION    Epsilon-greedy - balance novelty/certainty
  7. EPISODIC       Experience storage and replay for learning

Event hooks provided visibility into every operation. This same
infrastructure supports:
  - Debugging cognitive behavior
  - Logging for analysis
  - Integration with external systems
  - Real-time monitoring dashboards
    """)

    return agent


if __name__ == "__main__":
    agent = main()
