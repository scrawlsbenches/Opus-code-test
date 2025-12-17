#!/usr/bin/env python3
"""
Corpus Balance Analyzer

Analyzes sample documents to measure domain coverage and balance.
Used to track progress toward a balanced Mixture of Experts training corpus.

Usage:
    python scripts/analyze_corpus_balance.py
    python scripts/analyze_corpus_balance.py --verbose
    python scripts/analyze_corpus_balance.py --json > balance_report.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import re


# Target taxonomy with keyword patterns for classification
TAXONOMY = {
    # Sciences
    "physics_mechanics": ["mechanics", "force", "motion", "newton", "momentum", "velocity", "acceleration", "friction", "gravity"],
    "physics_quantum": ["quantum", "particle", "wave", "photon", "electron", "superposition", "entanglement"],
    "chemistry_organic": ["organic chemistry", "carbon", "molecule", "polymer", "compound", "reaction", "synthesis"],
    "chemistry_materials": ["materials", "alloy", "composite", "ceramic", "polymer material", "metal", "crystal structure"],
    "biology_cellular": ["cell", "cellular", "mitochondria", "nucleus", "membrane", "organelle", "cytoplasm"],
    "biology_ecology": ["ecology", "ecosystem", "habitat", "species", "biodiversity", "population", "environment"],
    "biology_genetics": ["genetics", "dna", "gene", "chromosome", "mutation", "heredity", "genome"],
    "astronomy": ["astronomy", "star", "planet", "galaxy", "telescope", "cosmic", "celestial", "orbit"],
    "geology": ["geology", "rock", "mineral", "earthquake", "volcanic", "sediment", "tectonic"],
    "meteorology": ["meteorology", "weather", "climate", "atmosphere", "precipitation", "storm", "temperature"],
    "oceanography": ["ocean", "marine", "sea", "tide", "current", "coral", "submarine"],
    "pharmacology": ["pharmacology", "drug", "medication", "dose", "pharmaceutical", "therapeutic", "toxicity"],
    "neuroscience": ["neuroscience", "brain", "neuron", "synapse", "cortex", "neural", "cognitive"],
    "psychology": ["psychology", "behavior", "cognitive bias", "perception", "memory", "emotion", "mental"],
    "medicine_clinical": ["clinical", "diagnosis", "treatment", "patient", "symptom", "disease", "therapy"],
    "medicine_surgery": ["surgery", "surgical", "operation", "incision", "transplant", "anesthesia"],

    # Engineering
    "mechanical_engineering": ["mechanical", "machine", "gear", "bearing", "shaft", "torque", "mechanism"],
    "electrical_engineering": ["electrical", "circuit", "voltage", "current", "resistor", "capacitor", "inductor"],
    "civil_engineering": ["civil engineering", "bridge", "foundation", "concrete", "structural", "load bearing"],
    "chemical_engineering": ["chemical engineering", "reactor", "distillation", "separation", "process engineering"],
    "aerospace_engineering": ["aerospace", "aircraft", "flight", "aerodynamic", "propulsion", "wing", "fuselage"],
    "robotics": ["robot", "actuator", "sensor", "autonomous", "manipulator", "kinematics", "end effector"],
    "manufacturing": ["manufacturing", "cnc", "machining", "assembly", "production", "fabrication", "tooling"],
    "quality_control": ["quality control", "inspection", "defect", "tolerance", "specification", "qc", "qa"],
    "industrial_design": ["industrial design", "ergonomic", "user centered", "prototype", "form factor"],
    "systems_engineering": ["systems engineering", "requirement", "integration", "verification", "validation"],
    "environmental_engineering": ["environmental engineering", "pollution", "waste", "remediation", "sustainability"],
    "biomedical_engineering": ["biomedical", "prosthetic", "implant", "medical device", "bioinstrumentation"],

    # Computing & Technology
    "algorithms": ["algorithm", "complexity", "data structure", "sorting", "searching", "graph algorithm", "dynamic programming"],
    "operating_systems": ["operating system", "kernel", "process", "thread", "scheduling", "memory management", "file system"],
    "networking": ["network", "protocol", "tcp", "ip", "router", "packet", "bandwidth", "latency"],
    "databases": ["database", "sql", "query optimization", "index", "transaction", "schema", "normalization"],
    "distributed_systems": ["distributed system", "consensus", "replication", "partition", "cap theorem", "eventual consistency"],
    "computer_graphics": ["graphics", "rendering", "shader", "texture", "3d model", "ray tracing", "rasterization"],
    "hci": ["hci", "user interface", "usability", "interaction design", "ux", "accessibility"],
    "cybersecurity": ["security", "encryption", "vulnerability", "authentication", "firewall", "malware", "penetration"],
    "cloud_computing": ["cloud", "virtualization", "container", "kubernetes", "serverless", "iaas", "paas"],
    "devops": ["devops", "ci cd", "deployment", "monitoring", "infrastructure", "automation", "pipeline"],
    "embedded_systems": ["embedded", "microcontroller", "firmware", "real time", "rtos", "iot"],
    "quantum_computing": ["quantum computing", "qubit", "quantum gate", "quantum algorithm", "decoherence"],

    # AI/ML
    "ml_fundamentals": ["machine learning", "supervised", "unsupervised", "classification", "regression", "feature"],
    "deep_learning": ["deep learning", "neural network", "layer", "backpropagation", "gradient", "loss function"],
    "nlp": ["nlp", "natural language", "tokenization", "parsing", "sentiment", "text classification", "language model"],
    "computer_vision": ["computer vision", "image", "object detection", "segmentation", "cnn", "feature extraction"],
    "reinforcement_learning": ["reinforcement learning", "reward", "policy", "agent", "environment", "q learning"],
    "knowledge_graphs": ["knowledge graph", "ontology", "triple", "relation extraction", "entity", "semantic web"],
    "expert_systems": ["expert system", "rule based", "inference engine", "knowledge base", "forward chaining"],
    "neural_architecture": ["architecture", "transformer", "attention", "encoder", "decoder", "embedding"],
    "ai_ethics": ["ai ethics", "fairness", "bias", "transparency", "accountability", "responsible ai"],
    "generative_ai": ["generative", "gan", "diffusion", "autoencoder", "generation", "synthesis"],

    # Business & Finance
    "accounting": ["accounting", "ledger", "debit", "credit", "balance sheet", "income statement", "audit"],
    "financial_markets": ["market", "trading", "stock", "bond", "derivative", "options", "futures"],
    "investment_banking": ["investment banking", "ipo", "merger", "acquisition", "underwriting", "capital"],
    "corporate_finance": ["corporate finance", "valuation", "cash flow", "capital structure", "dividend"],
    "risk_management": ["risk management", "hedging", "var", "exposure", "mitigation", "volatility"],
    "marketing": ["marketing", "brand", "campaign", "customer acquisition", "segmentation", "positioning"],
    "operations_management": ["operations", "supply chain", "inventory", "logistics", "procurement", "lean"],
    "supply_chain": ["supply chain", "logistics", "warehouse", "distribution", "procurement", "supplier"],
    "human_resources": ["human resources", "hr", "recruitment", "onboarding", "performance", "compensation"],
    "project_management": ["project management", "scope", "schedule", "budget", "stakeholder", "milestone"],

    # Law & Governance
    "contract_law": ["contract", "agreement", "breach", "damages", "consideration", "offer", "acceptance"],
    "intellectual_property": ["patent", "copyright", "trademark", "intellectual property", "infringement", "licensing"],
    "criminal_law": ["criminal", "prosecution", "defense", "verdict", "sentencing", "felony", "misdemeanor"],
    "corporate_law": ["corporate law", "incorporation", "shareholder", "board", "fiduciary", "governance"],
    "international_law": ["international law", "treaty", "sovereignty", "jurisdiction", "extradition"],
    "regulatory_compliance": ["compliance", "regulation", "regulatory", "audit", "enforcement", "policy"],
    "constitutional_law": ["constitutional", "amendment", "rights", "judicial review", "due process"],
    "privacy_law": ["privacy", "gdpr", "data protection", "consent", "personal data", "right to be forgotten"],

    # Humanities
    "philosophy": ["philosophy", "ethics", "metaphysics", "epistemology", "logic", "existential"],
    "history": ["history", "historical", "century", "era", "civilization", "dynasty", "revolution"],
    "literature": ["literature", "novel", "poetry", "narrative", "author", "literary", "prose"],
    "linguistics": ["linguistics", "grammar", "syntax", "semantics", "phonology", "morphology"],
    "anthropology": ["anthropology", "culture", "ethnography", "kinship", "ritual", "tribe"],
    "archaeology": ["archaeology", "excavation", "artifact", "stratigraphy", "ancient", "ruins"],
    "religious_studies": ["religion", "theology", "sacred", "ritual", "worship", "scripture"],
    "art_history": ["art history", "renaissance", "baroque", "impressionism", "sculpture", "painting"],

    # Creative & Performing Arts
    "visual_arts": ["visual art", "painting", "drawing", "sculpture", "canvas", "composition"],
    "music_composition": ["music", "composition", "melody", "harmony", "rhythm", "chord", "notation"],
    "film_cinema": ["film", "cinema", "director", "screenplay", "cinematography", "editing"],
    "theater_drama": ["theater", "drama", "stage", "performance", "actor", "playwright"],
    "architecture": ["architecture", "building", "facade", "floor plan", "structural", "design"],
    "photography": ["photography", "exposure", "aperture", "shutter", "lens", "focal length"],
    "graphic_design": ["graphic design", "typography", "layout", "visual identity", "branding"],
    "creative_writing": ["creative writing", "fiction", "character", "plot", "dialogue", "narrative"],

    # Practical Skills & Trades
    "culinary_arts": ["culinary", "cooking", "recipe", "ingredient", "flavor", "cuisine", "chef"],
    "agriculture": ["agriculture", "farming", "crop", "harvest", "soil", "irrigation", "cultivation"],
    "woodworking": ["woodworking", "carpentry", "joinery", "lumber", "saw", "chisel", "cabinet"],
    "metalworking": ["metalworking", "welding", "forging", "casting", "machining", "lathe"],
    "textiles_fashion": ["textile", "fabric", "sewing", "pattern", "fashion", "garment", "weaving"],
    "construction": ["construction", "framing", "foundation", "roofing", "drywall", "contractor"],
    "automotive": ["automotive", "engine", "transmission", "brake", "suspension", "mechanic"],
    "plumbing_electrical": ["plumbing", "pipe", "fixture", "wiring", "electrical installation"],

    # Sports & Recreation
    "team_sports": ["team sport", "soccer", "basketball", "football", "hockey", "volleyball", "baseball"],
    "individual_sports": ["tennis", "golf", "swimming", "track", "cycling", "running", "athletics"],
    "martial_arts": ["martial art", "karate", "judo", "taekwondo", "boxing", "mma", "wrestling"],
    "outdoor_recreation": ["hiking", "camping", "climbing", "kayaking", "fishing", "outdoor"],
    "board_games": ["board game", "chess", "strategy game", "puzzle", "game theory", "competition"],
    "fitness_training": ["fitness", "workout", "exercise", "strength training", "cardio", "gym"],
    "extreme_sports": ["extreme sport", "skateboarding", "surfing", "snowboarding", "bmx", "parkour"],
    "esports": ["esports", "competitive gaming", "streaming", "tournament", "pro gaming"],
}

# Target documents per domain
TARGET_PER_DOMAIN = 30
MINIMUM_PER_DOMAIN = 15


def classify_document(filepath: Path, content: str) -> List[str]:
    """Classify a document into domains based on keyword matching."""
    content_lower = content.lower()
    filename_lower = filepath.stem.lower()

    matches = []
    for domain, keywords in TAXONOMY.items():
        score = 0
        for keyword in keywords:
            # Check filename (higher weight)
            if keyword in filename_lower:
                score += 3
            # Check content
            if keyword in content_lower:
                score += content_lower.count(keyword)

        if score > 2:  # Threshold for classification
            matches.append((domain, score))

    # Return top matches
    matches.sort(key=lambda x: -x[1])
    return [m[0] for m in matches[:3]]  # Allow up to 3 domain classifications


def analyze_samples(samples_dir: Path) -> Dict:
    """Analyze all sample documents and compute balance metrics."""
    domain_counts = defaultdict(list)
    unclassified = []
    total_docs = 0
    total_words = 0

    # Walk through samples directory
    for root, dirs, files in os.walk(samples_dir):
        for filename in files:
            if not filename.endswith(('.txt', '.md', '.py')):
                continue

            filepath = Path(root) / filename
            try:
                content = filepath.read_text(encoding='utf-8')
            except Exception:
                continue

            total_docs += 1
            total_words += len(content.split())

            domains = classify_document(filepath, content)
            if domains:
                for domain in domains:
                    domain_counts[domain].append(str(filepath.relative_to(samples_dir)))
            else:
                unclassified.append(str(filepath.relative_to(samples_dir)))

    # Compute metrics
    counts = {k: len(v) for k, v in domain_counts.items()}
    total_classified = sum(counts.values())

    # Balance metrics
    if counts:
        max_count = max(counts.values())
        min_count = min(counts.values()) if counts else 0
        mean_count = total_classified / len(counts) if counts else 0

        # Gini coefficient
        sorted_counts = sorted(counts.values())
        n = len(sorted_counts)
        if n > 0 and sum(sorted_counts) > 0:
            cumulative = sum((i + 1) * c for i, c in enumerate(sorted_counts))
            gini = (2 * cumulative) / (n * sum(sorted_counts)) - (n + 1) / n
        else:
            gini = 0
    else:
        max_count = min_count = mean_count = gini = 0

    # Identify gaps
    gaps = {}
    for domain in TAXONOMY:
        current = counts.get(domain, 0)
        gap = max(0, TARGET_PER_DOMAIN - current)
        if gap > 0:
            gaps[domain] = {
                'current': current,
                'target': TARGET_PER_DOMAIN,
                'gap': gap,
                'status': 'critical' if current < 5 else 'low' if current < MINIMUM_PER_DOMAIN else 'ok'
            }

    return {
        'summary': {
            'total_documents': total_docs,
            'total_words': total_words,
            'avg_doc_length': total_words / total_docs if total_docs > 0 else 0,
            'domains_covered': len(counts),
            'domains_total': len(TAXONOMY),
            'unclassified_count': len(unclassified),
        },
        'balance': {
            'max_domain_count': max_count,
            'min_domain_count': min_count,
            'mean_domain_count': mean_count,
            'gini_coefficient': gini,
            'balance_score': 1 - gini,  # Higher is better
        },
        'domain_counts': dict(sorted(counts.items(), key=lambda x: -x[1])),
        'domain_files': {k: v for k, v in domain_counts.items()},
        'gaps': dict(sorted(gaps.items(), key=lambda x: -x[1]['gap'])),
        'unclassified': unclassified,
    }


def print_report(analysis: Dict, verbose: bool = False):
    """Print a human-readable report."""
    print("=" * 70)
    print("CORPUS BALANCE ANALYSIS REPORT")
    print("=" * 70)

    summary = analysis['summary']
    print(f"\n📊 SUMMARY")
    print(f"   Total documents: {summary['total_documents']}")
    print(f"   Total words: {summary['total_words']:,}")
    print(f"   Average doc length: {summary['avg_doc_length']:.0f} words")
    print(f"   Domains covered: {summary['domains_covered']}/{summary['domains_total']}")
    print(f"   Unclassified: {summary['unclassified_count']}")

    balance = analysis['balance']
    print(f"\n⚖️  BALANCE METRICS")
    print(f"   Max domain count: {balance['max_domain_count']}")
    print(f"   Min domain count: {balance['min_domain_count']}")
    print(f"   Mean domain count: {balance['mean_domain_count']:.1f}")
    print(f"   Gini coefficient: {balance['gini_coefficient']:.3f} (lower is more balanced)")
    print(f"   Balance score: {balance['balance_score']:.3f} (higher is better)")

    print(f"\n📈 TOP DOMAINS")
    for domain, count in list(analysis['domain_counts'].items())[:15]:
        bar = '█' * min(count, 30)
        status = '✓' if count >= TARGET_PER_DOMAIN else '○' if count >= MINIMUM_PER_DOMAIN else '✗'
        print(f"   {status} {domain[:30]:<30} {count:>3} {bar}")

    print(f"\n🔴 CRITICAL GAPS (< 5 docs)")
    critical = [(d, g) for d, g in analysis['gaps'].items() if g['status'] == 'critical']
    for domain, gap in critical[:20]:
        print(f"   {domain:<35} {gap['current']:>2} → {gap['target']} (need {gap['gap']})")

    if len(critical) > 20:
        print(f"   ... and {len(critical) - 20} more")

    print(f"\n🟡 LOW COVERAGE (5-14 docs)")
    low = [(d, g) for d, g in analysis['gaps'].items() if g['status'] == 'low']
    for domain, gap in low[:10]:
        print(f"   {domain:<35} {gap['current']:>2} → {gap['target']} (need {gap['gap']})")

    # Calculate total gap
    total_gap = sum(g['gap'] for g in analysis['gaps'].values())
    print(f"\n📉 TOTAL GAP: {total_gap} documents needed to reach target")

    if verbose and analysis['unclassified']:
        print(f"\n❓ UNCLASSIFIED DOCUMENTS")
        for doc in analysis['unclassified'][:20]:
            print(f"   - {doc}")
        if len(analysis['unclassified']) > 20:
            print(f"   ... and {len(analysis['unclassified']) - 20} more")


def main():
    parser = argparse.ArgumentParser(description='Analyze corpus balance')
    parser.add_argument('--samples-dir', default='samples', help='Samples directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir)
    if not samples_dir.exists():
        print(f"Error: Samples directory not found: {samples_dir}", file=sys.stderr)
        sys.exit(1)

    analysis = analyze_samples(samples_dir)

    if args.json:
        # Remove file lists for cleaner JSON output
        output = {
            'summary': analysis['summary'],
            'balance': analysis['balance'],
            'domain_counts': analysis['domain_counts'],
            'gaps': analysis['gaps'],
            'unclassified_count': len(analysis['unclassified']),
        }
        print(json.dumps(output, indent=2))
    else:
        print_report(analysis, verbose=args.verbose)


if __name__ == '__main__':
    main()
