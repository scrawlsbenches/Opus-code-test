#!/usr/bin/env python3
"""
DEMO: Learning System with REAL GoT Data

This demonstrates pulling actual data from the .got/ directory
and using our semantic matching + file risk tracking.
"""

import json
from pathlib import Path
from datetime import datetime

from llm_orchestration.learning import (
    LearningCycle,
    Context,
    Action,
    Outcome,
    OutcomeType,
    ExperienceType,
)


def load_got_tasks(got_dir: Path) -> list:
    """Load real tasks from .got/ directory."""
    tasks = []
    for task_file in got_dir.glob("T-*.json"):
        try:
            with open(task_file) as f:
                data = json.load(f)
                task = data.get("data", data)
                tasks.append({
                    "id": task.get("id"),
                    "title": task.get("title"),
                    "description": task.get("description", ""),
                    "status": task.get("status"),
                    "priority": task.get("priority"),
                    "category": task.get("properties", {}).get("category", "general"),
                    "created_at": task.get("created_at"),
                })
        except (json.JSONDecodeError, KeyError) as e:
            continue
    return tasks


def load_got_experiences(got_dir: Path) -> list:
    """Load real experiences from .got/learning/experiences/."""
    experiences = []
    exp_dir = got_dir / "learning" / "experiences"
    if not exp_dir.exists():
        return experiences

    for exp_file in exp_dir.glob("exp_*.json"):
        try:
            with open(exp_file) as f:
                exp = json.load(f)
                experiences.append(exp)
        except (json.JSONDecodeError, KeyError):
            continue
    return experiences


def load_got_edges(got_dir: Path) -> list:
    """Load edge relationships from .got/ directory."""
    edges = []
    for edge_file in got_dir.glob("E-*.json"):
        try:
            with open(edge_file) as f:
                data = json.load(f)
                edge = data.get("data", data)
                edges.append(edge)
        except (json.JSONDecodeError, KeyError):
            continue
    return edges


def main():
    got_dir = Path(".got")

    if not got_dir.exists():
        print("ERROR: .got directory not found!")
        return

    print("=" * 70)
    print("   LEARNING SYSTEM DEMO - REAL GoT DATA")
    print("=" * 70)

    # ========================================================================
    # PHASE 1: Load Real GoT Data
    # ========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: Loading Real Data from .got/")
    print("=" * 70)

    tasks = load_got_tasks(got_dir)
    experiences = load_got_experiences(got_dir)
    edges = load_got_edges(got_dir)

    print(f"\n📊 GoT Statistics:")
    print(f"   Tasks: {len(tasks)}")
    print(f"   Experiences: {len(experiences)}")
    print(f"   Edges: {len(edges)}")

    # Show some real tasks
    print(f"\n📋 Sample Tasks (first 5):")
    for task in tasks[:5]:
        status_icon = "✓" if task["status"] == "completed" else "○"
        print(f"   [{status_icon}] {task['title'][:60]}")
        print(f"       Priority: {task['priority']}, Category: {task['category']}")

    # ========================================================================
    # PHASE 2: Analyze Real Experiences
    # ========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: Analyzing Real Experiences")
    print("=" * 70)

    # Count outcomes
    success_count = sum(1 for e in experiences if e.get("outcome", {}).get("outcome_type") == "SUCCESS")
    failure_count = sum(1 for e in experiences if e.get("outcome", {}).get("outcome_type") == "FAILURE")

    print(f"\n📈 Experience Outcomes:")
    print(f"   Successes: {success_count}")
    print(f"   Failures: {failure_count}")
    if experiences:
        print(f"   Success Rate: {success_count/len(experiences)*100:.1f}%")

    # Show experiences with what_worked insights
    print(f"\n💡 Real Insights from Experiences:")
    for exp in experiences[:5]:
        intent = exp.get("intent", "Unknown")[:50]
        outcome = exp.get("outcome", {}).get("outcome_type", "?")
        what_worked = exp.get("what_worked", [])

        icon = "✓" if outcome == "SUCCESS" else "✗"
        print(f"\n   [{icon}] {intent}")
        if what_worked:
            # Parse the what_worked string for actual insights
            for insight in what_worked[:1]:
                if "What worked" in insight:
                    lines = insight.split("\n")
                    for line in lines[1:4]:  # First few insight lines
                        if line.strip().startswith("-"):
                            print(f"       {line.strip()}")

    # ========================================================================
    # PHASE 3: Apply Semantic Matching to Real Data
    # ========================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: Semantic Matching on Real Intents")
    print("=" * 70)

    # Create a temporary learning cycle and load real experiences
    import tempfile
    import shutil
    temp_dir = Path(tempfile.mkdtemp())

    try:
        cycle = LearningCycle(temp_dir)

        # Convert real experiences to our format
        loaded_count = 0
        for exp_data in experiences:
            try:
                context = Context(
                    goal_type=exp_data.get("context", {}).get("goal_type", "general"),
                    goal_complexity=exp_data.get("context", {}).get("goal_complexity", "moderate"),
                    domain=exp_data.get("context", {}).get("domain", "general"),
                )

                exp = cycle.start_experience(
                    context=context,
                    intent=exp_data.get("intent", "Unknown task"),
                    experience_type=ExperienceType.TASK_EXECUTION,
                )

                # Add what_worked/what_didnt_work
                what_worked = exp_data.get("what_worked", [])
                what_didnt = exp_data.get("what_didnt_work", [])
                exp.reflect(
                    what_worked=what_worked if isinstance(what_worked, list) else [],
                    what_didnt_work=what_didnt if isinstance(what_didnt, list) else [],
                    would_do_differently=exp_data.get("would_do_differently", [])
                )

                outcome_type = OutcomeType.SUCCESS if exp_data.get("outcome", {}).get("outcome_type") == "SUCCESS" else OutcomeType.FAILURE
                cycle.complete_experience(exp, Outcome(
                    outcome_type=outcome_type,
                    description=exp_data.get("outcome", {}).get("description", ""),
                ))
                loaded_count += 1
            except Exception as e:
                continue

        print(f"\n📥 Loaded {loaded_count} real experiences into learning cycle")

        # Now test semantic matching with real queries
        test_queries = [
            "Fix index save failure",
            "ML metrics collection",
            "test isolation bug",
            "WAL recovery",
        ]

        for query in test_queries:
            print(f"\n🔍 Query: \"{query}\"")
            print("-" * 50)

            results = cycle.find_by_intent(query, min_similarity=0.1, limit=3)

            if results:
                for exp in results:
                    similarity = cycle.intent_similarity(query, exp.intent)
                    outcome = "✓" if exp.outcome and exp.outcome.was_successful() else "✗"
                    print(f"   [{outcome}] {exp.intent[:55]}")
                    print(f"       Similarity: {similarity:.1%}")
            else:
                print("   No matches found")

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    # ========================================================================
    # PHASE 4: Show Edge Relationships
    # ========================================================================
    print("\n" + "=" * 70)
    print("PHASE 4: Real Edge Relationships")
    print("=" * 70)

    if edges:
        print(f"\n🔗 Task Relationships ({len(edges)} edges):")
        for edge in edges[:5]:
            edge_id = edge.get("id", "?")
            # Parse edge ID format: E-{status}-{date}-{hash}-{from}-{type}.json
            parts = edge_id.split("-")
            if len(parts) >= 6:
                edge_type = parts[-1].replace(".json", "")
                print(f"   {edge_type}")
    else:
        print("\n   No edges found")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("""
Key Findings:
1. GoT contains real task execution history with outcomes
2. Experiences include rich "what worked/didn't work" insights
3. Semantic matching finds related tasks across the real data
4. This data can inform future agent decisions
""")


if __name__ == "__main__":
    main()
