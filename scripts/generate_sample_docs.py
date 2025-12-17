#!/usr/bin/env python3
"""
Sample Document Generator

Generates sample documents for specified domains to help balance the corpus.
This is a template-based generator that creates structured documents with
domain-specific terminology and patterns.

Usage:
    python scripts/generate_sample_docs.py --domain philosophy --count 10
    python scripts/generate_sample_docs.py --list-domains
    python scripts/generate_sample_docs.py --batch gaps.json
    python scripts/generate_sample_docs.py --interactive

NOTE: This generator creates placeholder documents. For high-quality training,
replace these with authentic domain content or use AI-assisted generation
with human review.
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


# Domain definitions with seed content for generation
DOMAIN_SEEDS = {
    # Sciences
    "physics_mechanics": {
        "concepts": ["force", "motion", "momentum", "energy", "acceleration", "velocity", "friction", "gravity", "torque", "equilibrium"],
        "activities": ["analyzing motion", "calculating forces", "measuring velocity", "studying collisions", "examining rotational dynamics"],
        "applications": ["automotive engineering", "sports science", "aerospace design", "structural analysis", "robotics"],
    },
    "physics_quantum": {
        "concepts": ["wave-particle duality", "superposition", "entanglement", "uncertainty principle", "quantum tunneling", "spin", "quantum states"],
        "activities": ["measuring quantum states", "observing interference patterns", "detecting particles", "calculating probabilities"],
        "applications": ["quantum computing", "cryptography", "medical imaging", "semiconductor design", "laser technology"],
    },
    "chemistry_organic": {
        "concepts": ["carbon bonds", "functional groups", "isomers", "polymers", "reactions", "catalysis", "stereochemistry"],
        "activities": ["synthesizing compounds", "analyzing structures", "studying reaction mechanisms", "purifying substances"],
        "applications": ["drug development", "materials science", "agriculture", "food chemistry", "petroleum processing"],
    },
    "biology_ecology": {
        "concepts": ["ecosystem", "food web", "biodiversity", "habitat", "population dynamics", "symbiosis", "succession"],
        "activities": ["studying species interactions", "measuring biodiversity", "tracking populations", "analyzing ecosystems"],
        "applications": ["conservation biology", "environmental management", "agriculture", "fisheries", "urban planning"],
    },

    # Law
    "contract_law": {
        "concepts": ["offer", "acceptance", "consideration", "breach", "damages", "remedies", "performance", "termination"],
        "activities": ["drafting agreements", "negotiating terms", "reviewing contracts", "enforcing obligations"],
        "applications": ["business transactions", "employment", "real estate", "intellectual property", "commercial sales"],
    },
    "intellectual_property": {
        "concepts": ["patent", "copyright", "trademark", "trade secret", "licensing", "infringement", "fair use", "prior art"],
        "activities": ["filing applications", "prosecuting patents", "licensing technology", "enforcing rights"],
        "applications": ["technology licensing", "brand protection", "content distribution", "pharmaceutical patents"],
    },

    # Humanities
    "philosophy": {
        "concepts": ["ethics", "metaphysics", "epistemology", "logic", "aesthetics", "existentialism", "consciousness", "free will"],
        "activities": ["analyzing arguments", "examining assumptions", "questioning beliefs", "constructing theories"],
        "applications": ["ethical decision-making", "critical thinking", "policy analysis", "artificial intelligence ethics"],
    },
    "history": {
        "concepts": ["causation", "periodization", "primary sources", "historiography", "cultural context", "social change"],
        "activities": ["analyzing sources", "interpreting events", "tracing developments", "comparing civilizations"],
        "applications": ["policy analysis", "cultural preservation", "education", "journalism", "diplomacy"],
    },
    "literature": {
        "concepts": ["narrative", "character", "theme", "symbolism", "genre", "style", "voice", "perspective"],
        "activities": ["close reading", "literary analysis", "comparing texts", "examining context"],
        "applications": ["creative writing", "criticism", "education", "publishing", "screenwriting"],
    },

    # Arts
    "music_composition": {
        "concepts": ["melody", "harmony", "rhythm", "counterpoint", "orchestration", "dynamics", "form", "timbre"],
        "activities": ["composing pieces", "arranging music", "analyzing scores", "studying techniques"],
        "applications": ["film scoring", "game audio", "advertising", "live performance", "music education"],
    },
    "visual_arts": {
        "concepts": ["composition", "color theory", "perspective", "texture", "form", "space", "light", "movement"],
        "activities": ["creating works", "studying techniques", "analyzing art", "developing style"],
        "applications": ["illustration", "fine art", "advertising", "product design", "animation"],
    },

    # Practical Skills
    "culinary_arts": {
        "concepts": ["flavor profiles", "cooking techniques", "mise en place", "heat transfer", "emulsification", "fermentation"],
        "activities": ["preparing dishes", "developing recipes", "plating presentations", "managing kitchens"],
        "applications": ["restaurant cooking", "catering", "food product development", "culinary education"],
    },
    "woodworking": {
        "concepts": ["joinery", "grain direction", "wood selection", "finishing", "measurement", "tool maintenance"],
        "activities": ["cutting joints", "assembling pieces", "applying finishes", "designing furniture"],
        "applications": ["furniture making", "cabinetry", "construction", "restoration", "crafts"],
    },

    # Sports
    "team_sports": {
        "concepts": ["strategy", "teamwork", "positioning", "tactics", "conditioning", "game management"],
        "activities": ["practicing skills", "analyzing opponents", "executing plays", "building team chemistry"],
        "applications": ["professional sports", "coaching", "youth development", "sports analytics"],
    },
    "martial_arts": {
        "concepts": ["technique", "form", "discipline", "respect", "self-defense", "competition", "tradition"],
        "activities": ["practicing forms", "sparring", "conditioning", "studying philosophy"],
        "applications": ["self-defense", "competition", "fitness", "discipline training", "performance"],
    },
}

# Document templates
TEMPLATES = {
    "concept_explanation": """# {title}

## Overview

{overview}

## Key Principles

{principles}

## Applications

{applications}

## Related Concepts

{related}

---
*Generated for corpus balancing - {domain} domain*
""",

    "procedural_guide": """# How to {task}

## Prerequisites

{prerequisites}

## Step-by-Step Process

{steps}

## Common Challenges

{challenges}

## Tips for Success

{tips}

---
*Generated for corpus balancing - {domain} domain*
""",

    "glossary": """# {domain_title} Terminology

## Core Terms

{core_terms}

## Advanced Concepts

{advanced_concepts}

## Practical Applications

{applications}

---
*Generated for corpus balancing - {domain} domain*
""",
}


def generate_concept_doc(domain: str, seed: Dict, index: int) -> str:
    """Generate a concept explanation document."""
    concept = seed['concepts'][index % len(seed['concepts'])]
    related_concepts = random.sample(seed['concepts'], min(3, len(seed['concepts'])))

    title = f"{concept.title()} in {domain.replace('_', ' ').title()}"

    overview = f"""The concept of {concept} is fundamental to understanding {domain.replace('_', ' ')}.
It encompasses the principles and practices that define how practitioners approach core challenges in the field.
Understanding {concept} requires familiarity with both theoretical foundations and practical applications."""

    principles = "\n".join([
        f"1. **Foundation**: The basic principle of {concept} establishes the groundwork for advanced study.",
        f"2. **Application**: Practical use of {concept} requires systematic approaches.",
        f"3. **Integration**: {concept.title()} connects with other concepts in {domain.replace('_', ' ')}.",
        f"4. **Evolution**: The understanding of {concept} continues to develop with new research.",
    ])

    applications = "\n".join([
        f"- **{app.title()}**: {concept.title()} plays a crucial role in {app}."
        for app in seed['applications'][:4]
    ])

    related = "\n".join([
        f"- [[{rc}]]: Connection through shared principles"
        for rc in related_concepts if rc != concept
    ])

    return TEMPLATES['concept_explanation'].format(
        title=title,
        overview=overview,
        principles=principles,
        applications=applications,
        related=related,
        domain=domain
    )


def generate_procedural_doc(domain: str, seed: Dict, index: int) -> str:
    """Generate a procedural guide document."""
    activity = seed['activities'][index % len(seed['activities'])]

    task = activity.replace('ing', '').strip()

    prerequisites = "\n".join([
        f"- Understanding of {seed['concepts'][i]}"
        for i in range(min(3, len(seed['concepts'])))
    ])

    steps = "\n".join([
        f"### Step {i+1}: {phase.title()}\n\n{phase.title()} requires attention to detail and systematic approach. "
        f"Begin by reviewing the relevant {seed['concepts'][i % len(seed['concepts'])]} principles.\n"
        for i, phase in enumerate(["preparation", "execution", "verification", "documentation"])
    ])

    challenges = "\n".join([
        f"- **Challenge {i+1}**: Common difficulty related to {seed['concepts'][i % len(seed['concepts'])]}"
        for i in range(3)
    ])

    tips = "\n".join([
        f"- Practice with simpler examples before attempting complex {activity}",
        f"- Document your process for future reference",
        f"- Seek feedback from experienced practitioners",
        f"- Review related concepts regularly",
    ])

    return TEMPLATES['procedural_guide'].format(
        task=task,
        prerequisites=prerequisites,
        steps=steps,
        challenges=challenges,
        tips=tips,
        domain=domain
    )


def generate_glossary_doc(domain: str, seed: Dict, index: int) -> str:
    """Generate a glossary document."""
    domain_title = domain.replace('_', ' ').title()

    core_terms = "\n\n".join([
        f"**{concept.title()}**\n: A fundamental concept in {domain_title} that relates to "
        f"the systematic study and application of principles in the field."
        for concept in seed['concepts'][:5]
    ])

    advanced_concepts = "\n\n".join([
        f"**Advanced {concept.title()}**\n: Building on basic {concept}, this advanced topic explores "
        f"deeper implications and specialized applications."
        for concept in seed['concepts'][5:8] if len(seed['concepts']) > 5
    ]) or "See core terms for foundational concepts."

    applications = "\n".join([
        f"- **{app.title()}**: Practical application of {domain_title} principles"
        for app in seed['applications'][:5]
    ])

    return TEMPLATES['glossary'].format(
        domain_title=domain_title,
        core_terms=core_terms,
        advanced_concepts=advanced_concepts,
        applications=applications,
        domain=domain
    )


def generate_document(domain: str, index: int) -> Optional[str]:
    """Generate a single document for a domain."""
    if domain not in DOMAIN_SEEDS:
        return None

    seed = DOMAIN_SEEDS[domain]
    template_type = index % 3

    if template_type == 0:
        return generate_concept_doc(domain, seed, index)
    elif template_type == 1:
        return generate_procedural_doc(domain, seed, index)
    else:
        return generate_glossary_doc(domain, seed, index)


def generate_documents(domain: str, count: int, output_dir: Path, dry_run: bool = False) -> List[Path]:
    """Generate multiple documents for a domain."""
    if domain not in DOMAIN_SEEDS:
        print(f"Warning: No seed data for domain '{domain}'. Using generic template.")
        return []

    created = []
    domain_dir = output_dir / domain
    if not dry_run:
        domain_dir.mkdir(parents=True, exist_ok=True)

    for i in range(count):
        content = generate_document(domain, i)
        if content is None:
            continue

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"{domain}_{timestamp}_{i:03d}.txt"
        filepath = domain_dir / filename

        if dry_run:
            print(f"Would create: {filepath}")
        else:
            filepath.write_text(content)
            print(f"Created: {filepath}")

        created.append(filepath)

    return created


def list_domains():
    """List all available domains."""
    print("\nAvailable domains for document generation:\n")
    for domain, seed in sorted(DOMAIN_SEEDS.items()):
        print(f"  {domain}")
        print(f"    Concepts: {', '.join(seed['concepts'][:5])}...")
        print()


def interactive_mode(output_dir: Path):
    """Interactive document generation mode."""
    print("\n=== Interactive Document Generator ===\n")
    print("Commands:")
    print("  list          - List available domains")
    print("  generate <domain> <count> - Generate documents")
    print("  quit          - Exit\n")

    while True:
        try:
            cmd = input(">>> ").strip().split()
            if not cmd:
                continue

            if cmd[0] == "quit":
                break
            elif cmd[0] == "list":
                list_domains()
            elif cmd[0] == "generate" and len(cmd) >= 3:
                domain = cmd[1]
                count = int(cmd[2])
                generate_documents(domain, count, output_dir)
            else:
                print("Unknown command. Try 'list', 'generate <domain> <count>', or 'quit'")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Generate sample documents for corpus balancing')
    parser.add_argument('--domain', '-d', help='Domain to generate documents for')
    parser.add_argument('--count', '-n', type=int, default=10, help='Number of documents to generate')
    parser.add_argument('--output', '-o', default='samples/generated', help='Output directory')
    parser.add_argument('--list-domains', '-l', action='store_true', help='List available domains')
    parser.add_argument('--batch', '-b', help='JSON file with domain:count mappings')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be created')
    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.list_domains:
        list_domains()
        return

    if args.interactive:
        interactive_mode(output_dir)
        return

    if args.batch:
        with open(args.batch) as f:
            batch = json.load(f)
        for domain, count in batch.items():
            print(f"\nGenerating {count} documents for {domain}...")
            generate_documents(domain, count, output_dir, args.dry_run)
        return

    if args.domain:
        generate_documents(args.domain, args.count, output_dir, args.dry_run)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
