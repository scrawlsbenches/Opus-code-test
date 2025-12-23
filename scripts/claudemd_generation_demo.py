#!/usr/bin/env python3
"""
Demo script for the CLAUDE.md auto-generation system.

This script demonstrates:
1. Creating and managing CLAUDE.md layers
2. Context-aware layer selection
3. Freshness tracking and staleness detection
4. PersonaProfile and Team entities for multi-team support
5. Full generation pipeline with fault tolerance

Usage:
    python scripts/claudemd_generation_demo.py [--verbose]
    python scripts/claudemd_generation_demo.py --dry-run
    python scripts/claudemd_generation_demo.py --team-demo
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cortical.got.types import ClaudeMdLayer, ClaudeMdVersion, PersonaProfile, Team
from cortical.utils.id_generation import (
    generate_claudemd_layer_id,
    generate_claudemd_version_id,
    generate_persona_profile_id,
    generate_team_id,
)


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subheader(title: str) -> None:
    """Print a formatted subsection header."""
    print()
    print(f"--- {title} ---")


def demo_layer_creation() -> list:
    """Demonstrate creating CLAUDE.md layers."""
    print_header("1. Creating CLAUDE.md Layers")

    layers = []

    # Layer 0: Core (always included)
    layer0 = ClaudeMdLayer(
        id=generate_claudemd_layer_id(0, "quick-start"),
        layer_type="core",
        layer_number=0,
        section_id="quick-start",
        title="Quick Session Start",
        content="""## Quick Session Start (READ THIS FIRST)

**New session? Start here to restore context fast.**

1. Check GoT State: `python scripts/got_utils.py validate`
2. Read Recent Knowledge Transfer
3. Work Priority: Security > Bugs > Features > Docs
""",
        freshness_decay_days=30,
        inclusion_rule="always",
    )
    layers.append(layer0)
    print(f"  Created Layer 0 (Core): {layer0.title}")
    print(f"    ID: {layer0.id}")
    print(f"    Inclusion: {layer0.inclusion_rule}")

    # Layer 1: Operational
    layer1 = ClaudeMdLayer(
        id=generate_claudemd_layer_id(1, "dev-workflow"),
        layer_type="operational",
        layer_number=1,
        section_id="dev-workflow",
        title="Development Workflow",
        content="""## Development Workflow

### Before Writing Code
1. Read the relevant module
2. Check existing tasks
3. Run tests first to establish baseline

### After Writing Code
1. Run the full test suite
2. Check coverage hasn't dropped
3. Dog-food the feature
""",
        freshness_decay_days=14,
        inclusion_rule="always",
    )
    layers.append(layer1)
    print(f"  Created Layer 1 (Operational): {layer1.title}")

    # Layer 2: Contextual (included based on context)
    layer2 = ClaudeMdLayer(
        id=generate_claudemd_layer_id(2, "got-guide"),
        layer_type="contextual",
        layer_number=2,
        section_id="got-guide",
        title="Graph of Thought Guide",
        content="""## Graph of Thought (GoT) Usage

When working with the GoT system:
- Tasks: `T-YYYYMMDD-HHMMSS-XXXXXXXX`
- Decisions: `D-YYYYMMDD-HHMMSS-XXXXXXXX`
- Use `python scripts/got_utils.py` for all operations
""",
        freshness_decay_days=7,
        inclusion_rule="context",
        context_modules=["cortical/got", "cortical/reasoning"],
        context_branches=["feature/got-*", "claude/*"],
    )
    layers.append(layer2)
    print(f"  Created Layer 2 (Contextual): {layer2.title}")
    print(f"    Context modules: {layer2.context_modules}")

    # Layer 3: Persona-specific
    layer3 = ClaudeMdLayer(
        id=generate_claudemd_layer_id(3, "ml-engineer"),
        layer_type="persona",
        layer_number=3,
        section_id="ml-engineer",
        title="ML Engineering Guidelines",
        content="""## ML Engineering Guidelines

- Always validate data quality before training
- Use reproducible random seeds
- Track experiments with clear naming conventions
- Monitor for data drift in production
""",
        freshness_decay_days=30,
        inclusion_rule="user_pref",
        properties={"persona_ids": ["ml-engineer", "data-scientist"]},
    )
    layers.append(layer3)
    print(f"  Created Layer 3 (Persona): {layer3.title}")
    print(f"    Persona IDs: {layer3.properties.get('persona_ids', [])}")

    # Layer 4: Ephemeral (session-specific)
    layer4 = ClaudeMdLayer(
        id=generate_claudemd_layer_id(4, "session-context"),
        layer_type="ephemeral",
        layer_number=4,
        section_id="session-context",
        title="Current Session Context",
        content="""## Current Session Context

**Active Sprint:** S-017 (Spark SLM)
**Current Branch:** claude/auto-generate-claude-md
**Focus:** CLAUDE.md auto-generation system
""",
        freshness_decay_days=1,
        inclusion_rule="context",
    )
    layers.append(layer4)
    print(f"  Created Layer 4 (Ephemeral): {layer4.title}")
    print(f"    Decay: {layer4.freshness_decay_days} day(s)")

    return layers


def demo_freshness_tracking(layers: list) -> None:
    """Demonstrate freshness tracking and staleness detection."""
    print_header("2. Freshness Tracking")

    for layer in layers:
        status = "FRESH" if not layer.is_stale() else "STALE"
        print(f"  Layer {layer.layer_number} ({layer.layer_type}): {status}")
        print(f"    Decay period: {layer.freshness_decay_days} days")
        print(f"    Last regenerated: {layer.last_regenerated or 'Never'}")

    # Simulate marking a layer stale
    print_subheader("Simulating Staleness")
    layers[4].mark_stale()
    print(f"  Marked Layer 4 as stale: {layers[4].freshness_status}")

    # Refresh it
    layers[4].mark_fresh()
    print(f"  Refreshed Layer 4: {layers[4].freshness_status}")
    print(f"    New regeneration time: {layers[4].last_regenerated}")


def demo_versioning(layers: list) -> None:
    """Demonstrate layer versioning."""
    print_header("3. Layer Versioning")

    # Create a version snapshot
    layer = layers[0]
    version = ClaudeMdVersion(
        id=generate_claudemd_version_id(layer.id, 1),
        layer_id=layer.id,
        version_number=1,
        content_snapshot=layer.content,
        change_rationale="Initial version",
    )

    print(f"  Created version snapshot:")
    print(f"    Version ID: {version.id}")
    print(f"    Layer ID: {version.layer_id}")
    print(f"    Version Number: {version.version_number}")
    print(f"    Rationale: {version.change_rationale}")
    print(f"    Content length: {len(version.content_snapshot)} chars")


def demo_serialization(layers: list) -> None:
    """Demonstrate serialization and deserialization."""
    print_header("4. Serialization Round-Trip")

    layer = layers[2]  # Use the contextual layer

    # Serialize
    data = layer.to_dict()
    print(f"  Serialized layer to dict:")
    print(f"    Keys: {list(data.keys())[:8]}...")

    # Deserialize
    restored = ClaudeMdLayer.from_dict(data)
    print(f"  Deserialized back to ClaudeMdLayer:")
    print(f"    ID match: {layer.id == restored.id}")
    print(f"    Content match: {layer.content == restored.content}")
    print(f"    Context modules match: {layer.context_modules == restored.context_modules}")


def demo_persona_profiles() -> None:
    """Demonstrate PersonaProfile entity for multi-team support."""
    print_header("5. PersonaProfile Entity (Multi-Team Support)")

    # Create a developer persona
    dev_persona = PersonaProfile(
        id=generate_persona_profile_id(),
        name="Senior Developer",
        role="developer",
        layer_preferences={
            "quick-start": True,
            "dev-workflow": True,
            "got-guide": True,
            "ml-engineer": False,  # Not relevant for general dev
        },
        custom_layers=["testing-guidelines"],
        excluded_layers=["marketing-style"],
    )

    print(f"  Created Developer Persona:")
    print(f"    ID: {dev_persona.id}")
    print(f"    Name: {dev_persona.name}")
    print(f"    Role: {dev_persona.role}")

    # Test layer inclusion
    print_subheader("Layer Inclusion Decisions")
    test_sections = ["quick-start", "ml-engineer", "testing-guidelines", "marketing-style"]
    for section in test_sections:
        include = dev_persona.should_include_layer(section)
        print(f"    {section}: {'INCLUDE' if include else 'EXCLUDE'}")

    # Create an ML engineer persona that inherits from developer
    ml_persona = PersonaProfile(
        id=generate_persona_profile_id(),
        name="ML Engineer",
        role="developer",
        inherits_from=dev_persona.id,
        layer_preferences={
            "ml-engineer": True,  # Override parent's False
        },
    )

    print_subheader("Inheritance Demo")
    print(f"  ML Engineer inherits from: {ml_persona.inherits_from}")
    print(f"  ML Engineer's own preferences: {ml_persona.layer_preferences}")
    print(f"  Effective preferences would include parent's + overrides")


def demo_team_hierarchy() -> None:
    """Demonstrate Team entity for organizational hierarchy."""
    print_header("6. Team Entity (Organizational Hierarchy)")

    # Create Engineering team (parent)
    engineering = Team(
        id=generate_team_id(),
        name="Engineering",
        branch_patterns=["main", "develop", "feature/*", "bugfix/*"],
        module_scope=["cortical/*"],
        settings={
            "code_review_required": True,
            "min_coverage": 85,
            "knowledge_domains": ["architecture", "testing", "performance"],
        },
    )

    print(f"  Created Engineering Team:")
    print(f"    ID: {engineering.id}")
    print(f"    Name: {engineering.name}")
    print(f"    Branch patterns: {engineering.branch_patterns}")

    # Create ML sub-team
    ml_team = Team(
        id=generate_team_id(),
        name="ML Platform",
        parent_team_id=engineering.id,
        branch_patterns=["feature/ml-*", "feature/spark-*"],
        module_scope=["cortical/spark/*", "cortical/reasoning/*"],
        settings={"knowledge_domains": ["ml-training", "data-pipelines"]},
    )

    print(f"\n  Created ML Platform Sub-Team:")
    print(f"    ID: {ml_team.id}")
    print(f"    Parent: {ml_team.parent_team_id}")
    print(f"    Module scope: {ml_team.module_scope}")

    # Test branch matching
    print_subheader("Branch Matching Demo")
    test_branches = ["main", "feature/ml-training", "feature/auth", "release/v1.0"]
    for branch in test_branches:
        eng_match = engineering.matches_branch(branch)
        ml_match = ml_team.matches_branch(branch)
        print(f"    {branch}:")
        print(f"      Engineering: {'MATCH' if eng_match else 'NO MATCH'}")
        print(f"      ML Platform: {'MATCH' if ml_match else 'NO MATCH'}")

    # Test module scope
    print_subheader("Module Scope Demo")
    test_modules = ["cortical/got/api.py", "cortical/spark/ngram.py", "scripts/demo.py"]
    for module in test_modules:
        eng_scope = engineering.is_in_scope(module)
        ml_scope = ml_team.is_in_scope(module)
        print(f"    {module}:")
        print(f"      Engineering: {'IN SCOPE' if eng_scope else 'OUT OF SCOPE'}")
        print(f"      ML Platform: {'IN SCOPE' if ml_scope else 'OUT OF SCOPE'}")


def demo_generation_pipeline(layers: list, dry_run: bool = True) -> None:
    """Demonstrate the generation pipeline concept."""
    print_header("7. Generation Pipeline (Conceptual)")

    print("  Pipeline Phases:")
    print("    1. Context Analysis - Detect branch, sprint, active files")
    print("    2. Layer Selection - Filter by context and persona")
    print("    3. Content Composition - Assemble in layer order")
    print("    4. Validation - Check required sections")
    print("    5. Output - Write with backup (or dry-run)")

    print_subheader("Simulated Context")
    context = {
        "branch": "claude/auto-generate-claude-md",
        "sprint": "S-017",
        "active_modules": ["cortical/got", "cortical/reasoning"],
        "persona": "developer",
    }
    for key, value in context.items():
        print(f"    {key}: {value}")

    print_subheader("Layer Selection Results")
    selected = []
    for layer in layers:
        # Simulate context-based selection
        include = False
        if layer.inclusion_rule == "always":
            include = True
        elif layer.inclusion_rule == "context":
            # Check if any context module matches
            for ctx_mod in layer.context_modules:
                for active in context["active_modules"]:
                    if active.startswith(ctx_mod.replace("/*", "")):
                        include = True
                        break

        status = "SELECTED" if include else "SKIPPED"
        print(f"    Layer {layer.layer_number} ({layer.section_id}): {status}")
        if include:
            selected.append(layer)

    print_subheader("Composed Output Preview")
    if dry_run:
        print("  [DRY RUN - No file written]")

    total_lines = 0
    for layer in selected:
        lines = len(layer.content.strip().split('\n'))
        total_lines += lines
        print(f"    + {layer.title}: {lines} lines")

    print(f"\n  Total composed content: {total_lines} lines from {len(selected)} layers")


def demo_fault_tolerance() -> None:
    """Demonstrate fault tolerance concepts."""
    print_header("8. Fault Tolerance")

    print("  Built-in protections:")
    print("    - Atomic writes with temp file + rename")
    print("    - Automatic backup before overwrite")
    print("    - Fallback to original CLAUDE.md on failure")
    print("    - Validation before write")
    print("    - Checksum verification on read")

    print_subheader("Recovery Scenarios")
    scenarios = [
        ("Corrupted layer file", "Skip layer, log warning, continue"),
        ("Missing required section", "Fall back to original CLAUDE.md"),
        ("Write failure", "Restore from backup"),
        ("Invalid layer content", "Validation catches before write"),
    ]
    for scenario, response in scenarios:
        print(f"    {scenario}")
        print(f"      -> {response}")


def main():
    """Run the CLAUDE.md generation demo."""
    import argparse

    parser = argparse.ArgumentParser(description="CLAUDE.md Generation System Demo")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose output")
    parser.add_argument("--dry-run", action="store_true", help="Run generation in dry-run mode")
    parser.add_argument("--team-demo", action="store_true", help="Focus on team/persona demo")
    args = parser.parse_args()

    print()
    print("*" * 70)
    print("*  CLAUDE.md Auto-Generation System Demo")
    print("*  Cortical Text Processor - Graph of Thought Integration")
    print("*" * 70)

    if args.team_demo:
        # Focus on multi-team features
        demo_persona_profiles()
        demo_team_hierarchy()
    else:
        # Full demo
        layers = demo_layer_creation()
        demo_freshness_tracking(layers)
        demo_versioning(layers)
        demo_serialization(layers)
        demo_persona_profiles()
        demo_team_hierarchy()
        demo_generation_pipeline(layers, dry_run=args.dry_run)
        demo_fault_tolerance()

    print_header("Demo Complete")
    print("  The CLAUDE.md auto-generation system provides:")
    print("    - 5-layer architecture (Core -> Ephemeral)")
    print("    - Context-aware layer selection")
    print("    - Freshness tracking with configurable decay")
    print("    - Version history for audit trails")
    print("    - PersonaProfile for role-based customization")
    print("    - Team hierarchy for organizational scoping")
    print("    - Fault-tolerant generation pipeline")
    print()
    print("  For more details, see:")
    print("    - docs/claude-md-generation-design.md")
    print("    - cortical/got/claudemd.py")
    print("    - cortical/got/types.py")
    print()


if __name__ == "__main__":
    main()
