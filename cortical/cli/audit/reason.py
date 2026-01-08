"""
Reason command - PLN-based audit reasoning.

Uses Probabilistic Logic Networks (PLN) for:
- Multi-rule aggregation for combining evidence
- Attention-based focus for prioritized inference
- Natural language query parsing
- Explainability for inference chains
- WovenMind integration for pattern discovery
"""

import os
from typing import Any

from ._base import (
    print_header,
    print_separator,
)


def setup_args(subparsers) -> None:
    """Set up command arguments."""
    parser = subparsers.add_parser(
        'reason',
        help='PLN-based audit reasoning'
    )
    parser.add_argument(
        'query',
        nargs='?',
        default=None,
        help='Natural language query (e.g., "risky files in reasoning/")'
    )
    parser.add_argument(
        '--directory',
        '-d',
        help='Directory to analyze'
    )
    parser.add_argument(
        '--explain',
        '-e',
        type=str,
        help='Explain risk for specific file'
    )
    parser.add_argument(
        '--load-rules',
        action='store_true',
        help='Load rules from WovenMind'
    )
    parser.add_argument(
        '--save-state',
        action='store_true',
        help='Save reasoning state for persistence'
    )
    parser.add_argument(
        '--vlti',
        type=str,
        help='Mark file as Very Long Term Important (pinned)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.3,
        help='Minimum risk threshold (default: 0.3)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed output'
    )


def run(args: Any) -> None:
    """Execute the reason command."""
    from cortical.audits import (
        AuditReasoner,
        AuditQuery,
        translate_audit_query,
        is_natural_language_query,
        analyze_directory,
    )

    verbose = getattr(args, 'verbose', False)
    threshold = getattr(args, 'threshold', 0.3)

    print("PLN Audit Reasoning")
    print_separator()

    # Initialize reasoner
    reasoner = AuditReasoner()
    reasoner.add_default_rules()

    # Load WovenMind rules if requested
    if getattr(args, 'load_rules', False):
        count = reasoner.load_rules_from_woven_mind()
        print(f"Loaded {count} rules from WovenMind")
        count = reasoner.load_manual_rules()
        print(f"Loaded {count} manual rules")

    # Mark VLTI if requested
    if args.vlti:
        reasoner.set_vlti(args.vlti, True)
        print(f"Marked {args.vlti} as Very Long Term Important")

    # Explain specific file
    if args.explain:
        print(f"\nExplaining risk for: {args.explain}")
        print_separator()
        explanation = reasoner.explain_file_risk(args.explain)

        if explanation['facts']:
            print("\nFacts:")
            for fact in explanation['facts']:
                print(f"  {fact['atom']}")
                print(f"    strength={fact['strength']:.2f}, confidence={fact['confidence']:.2f}")

        if explanation['risk_level']:
            rl = explanation['risk_level']
            print(f"\nRisk Level: {rl['mean']:.2%}")
            print(f"  strength={rl['strength']:.2f}, confidence={rl['confidence']:.2f}")

        if explanation['suggestions']:
            print("\nSuggestions:")
            for sugg in explanation['suggestions']:
                print(f"  • {sugg}")

        return

    # Handle natural language query or directory
    directory = args.directory
    query_str = args.query

    if query_str and is_natural_language_query(query_str):
        query = translate_audit_query(query_str)
        directory = directory or query.directory
        print(f"\nParsed query:")
        print(f"  Directory: {query.directory or '(current)'}")
        print(f"  Intent: {query.intent}")
        if query.negations:
            print(f"  Exclude: {query.negations}")
        if query.include_traits:
            print(f"  Traits: {query.include_traits}")
        if query.min_risk > 0:
            print(f"  Min risk: {query.min_risk}")
        threshold = max(threshold, query.min_risk)
    elif query_str and not directory:
        # Assume query_str is a directory
        directory = query_str

    if not directory:
        print("Error: No directory specified. Use --directory or provide a path.")
        return

    # Analyze directory
    print(f"\nAnalyzing {directory}...")
    result = analyze_directory(
        directory=directory,
        with_git=True,
        verbose=False,
    )

    if result.error:
        print(f"Error: {result.error}")
        return

    print(f"Found {result.files_analyzed} files, {len(result.findings)} findings")

    # Assert facts for each file
    file_patterns = {}
    for finding in result.findings:
        # Findings use 'id' field in format "filename:line"
        finding_id = finding.get('id', finding.get('file', ''))
        filepath = finding_id.split(':')[0] if ':' in finding_id else finding_id
        pattern = finding.get('pattern', '')
        if filepath and filepath not in file_patterns:
            file_patterns[filepath] = {'patterns': [], 'traits': [], 'dirs': []}
        if filepath and pattern:
            file_patterns[filepath]['patterns'].append(pattern)

    # Add traits from git analysis
    if result.git_analysis:
        high_churn = result.git_analysis.get('high_churn_files', {})
        for filepath in file_patterns:
            if filepath in high_churn:
                file_patterns[filepath]['traits'].append('high_churn')

    # Assert facts and query risk
    risky_files = []
    for filepath, data in file_patterns.items():
        # Extract directory name
        parts = filepath.split(os.sep)
        dirs = [p for p in parts[:-1] if p and p != '.']

        reasoner.assert_file_facts(
            file_path=filepath,
            patterns=data['patterns'],
            traits=data['traits'],
            directories=dirs,
        )

        # Query risk
        risk = reasoner.query_risk(filepath, aggregate=True)
        if risk and risk.mean() >= threshold:
            risky_files.append((filepath, risk.mean(), data['patterns']))

    # Sort by risk
    risky_files.sort(key=lambda x: -x[1])

    # Report
    print(f"\nRISKY FILES (threshold={threshold:.0%}):")
    print_separator()

    if not risky_files:
        print("  No files exceed the risk threshold.")
    else:
        for filepath, risk, patterns in risky_files[:20]:
            if filepath:
                rel_path = os.path.relpath(filepath, directory)
                print(f"\n  {rel_path}")
            else:
                print(f"\n  (unknown file)")
            print(f"    Risk: {risk:.1%}")
            if patterns and verbose:
                unique_patterns = list(set(patterns))[:5]
                print(f"    Patterns: {', '.join(unique_patterns)}")

        if len(risky_files) > 20:
            print(f"\n  ... and {len(risky_files) - 20} more risky files")

    # Focus on high importance
    focused = reasoner.focus_on_high_importance(threshold=0.5)
    if focused and verbose:
        print(f"\nAttention focused on {len(focused)} high-importance files")

    # Save state if requested
    if getattr(args, 'save_state', False):
        reasoner.save_state()
        print("\nState saved to persistence")

    print_separator()
    print("Reasoning complete!")
