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
    parser.add_argument(
        '--show-rules',
        action='store_true',
        help='Display all PLN rules (default, manual, and derived)'
    )
    parser.add_argument(
        '--show-state',
        action='store_true',
        help='Show persistence state (session count, file importance, etc.)'
    )
    parser.add_argument(
        '--clear-state',
        action='store_true',
        help='Clear persistence state and start fresh'
    )
    parser.add_argument(
        '--file-history',
        type=str,
        metavar='FILE',
        help='Show importance history for a specific file'
    )
    parser.add_argument(
        '--add-rule',
        nargs=3,
        metavar=('ANTECEDENT', 'CONSEQUENT', 'STRENGTH'),
        help='Add a manual PLN rule (e.g., --add-rule "test(X)" "flagged(X)" "0.8")'
    )
    parser.add_argument(
        '--aggregate',
        type=str,
        choices=['mean', 'max', 'min', 'product'],
        help='Set aggregation strategy for combining evidence'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save state after execution'
    )


def run(args: Any) -> None:
    """Execute the reason command."""
    import json
    from cortical.audits import (
        AuditReasoner,
        AuditQuery,
        translate_audit_query,
        is_natural_language_query,
        analyze_directory,
    )
    from cortical.audits.persistence import FilePersistenceBackend

    verbose = getattr(args, 'verbose', False)
    threshold = getattr(args, 'threshold', 0.3)

    # Handle --clear-state before initializing reasoner
    if getattr(args, 'clear_state', False):
        from pathlib import Path
        from cortical.audits.persistence import DEFAULT_PERSISTENCE_FILE

        print("PLN AUDIT STATE")
        print_separator()
        persistence_file = Path.cwd() / DEFAULT_PERSISTENCE_FILE
        if persistence_file.exists():
            persistence_file.unlink()
            print("Persistence state cleared.")
        else:
            print("No persistence state to clear.")
        return

    print("PLN Audit Reasoning")
    print_separator()

    # Initialize reasoner
    reasoner = AuditReasoner()
    reasoner.add_default_rules()

    # Handle --show-rules
    if getattr(args, 'show_rules', False):
        print("\nPLN AUDIT RULES")
        print_separator()

        stats = reasoner.get_stats()
        print(f"Total rules: {stats['rules']}")
        print(f"Aggregate strategy: {stats['aggregate_strategy']}")

        rules_config = reasoner.rules_config
        manual_rules = rules_config.get("manual_rules", [])
        derived_rules = rules_config.get("derived_rules", [])

        if manual_rules:
            print("\nManual Rules:")
            for rule in manual_rules:
                print(f"  {rule['antecedent']} → {rule['consequent']} (strength={rule.get('strength', 0.7):.2f})")

        if derived_rules:
            print("\nDerived Rules (from WovenMind):")
            for rule in derived_rules[:10]:
                print(f"  {rule.get('antecedent', '?')} → {rule.get('consequent', '?')} (strength={rule.get('strength', 0.7):.2f})")
            if len(derived_rules) > 10:
                print(f"  ... and {len(derived_rules) - 10} more")

        return

    # Handle --show-state
    if getattr(args, 'show_state', False):
        print("\nAUDIT PLN PERSISTENCE STATE")
        print_separator()

        state = reasoner._persistence_state
        if state:
            print(f"Session count: {state.session_count}")
            print(f"Created: {state.created}")
            print(f"Updated: {state.updated}")
            print(f"Files tracked: {len(state.file_importance)}")

            if state.global_stats:
                print("\nGlobal Stats:")
                for key, value in state.global_stats.items():
                    print(f"  {key}: {value}")

            if state.attention_focus:
                print(f"\nAttention focus: {len(state.attention_focus)} atoms")
        else:
            print("No persistence state loaded.")

        return

    # Handle --file-history
    if getattr(args, 'file_history', None):
        file_path = args.file_history
        # Normalize file path to ID format
        file_id = file_path.replace('.', '_').replace('/', '_').replace('\\', '_')

        print(f"\nImportance History for: {file_path}")
        print_separator()

        history = reasoner.get_importance_history(file_id)
        if history:
            print(f"File ID: {file_id}")
            for entry in history:
                print(f"  {entry.get('timestamp', 'unknown')}: STI={entry.get('sti', 0):.2f}, LTI={entry.get('lti', 0):.2f}")

            trend = reasoner.get_importance_trend(file_id)
            if trend:
                print(f"\nTrend: {trend}")
        else:
            print(f"No history found for {file_path}")

        return

    # Handle --add-rule
    if getattr(args, 'add_rule', None):
        antecedent, consequent, strength_str = args.add_rule
        strength = float(strength_str)

        print(f"\nAdding rule: {antecedent} → {consequent} (strength={strength:.2f})")

        # Add to rules config
        if "manual_rules" not in reasoner.rules_config:
            reasoner.rules_config["manual_rules"] = []

        reasoner.rules_config["manual_rules"].append({
            "antecedent": antecedent,
            "consequent": consequent,
            "strength": strength,
            "confidence": 0.8
        })

        # Save to persistence
        reasoner._persistence.save_rules(reasoner.rules_config)
        print(f"Added rule and saved to persistence.")

        return

    # Set aggregate strategy if specified
    if getattr(args, 'aggregate', None):
        reasoner.aggregate_strategy = args.aggregate
        print(f"Using aggregation strategy: {args.aggregate}")

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
        explanation = reasoner.explain_file_risk(args.explain, verbose=verbose)

        if explanation['facts']:
            print("\nFacts:")
            for fact_name in explanation['facts']:
                print(f"  • {fact_name}")

        if explanation['risk_level']:
            rl = explanation['risk_level']
            print(f"\nRisk Level: {rl['mean']:.2%}")
            print(f"  strength={rl['strength']:.2f}, confidence={rl['confidence']:.2f}")

        if explanation['suggestions']:
            print("\nSuggestions:")
            for sugg in explanation['suggestions']:
                print(f"  • {sugg}")

        if verbose and explanation.get('raw_traces'):
            print("\nInference Traces:")
            for query_type, trace_data in explanation['raw_traces'].items():
                print(f"  {query_type}:")
                if isinstance(trace_data, dict) and trace_data.get('final_result'):
                    fr = trace_data['final_result']
                    print(f"    Result: {fr.get('strength', 0):.2%} (conf: {fr.get('confidence', 0):.2%})")

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
                # Build readable path: prepend directory if filepath is just a filename
                if os.path.isabs(filepath):
                    display_path = os.path.relpath(filepath, directory)
                elif os.sep not in filepath and '/' not in filepath:
                    # Just a filename - prepend directory for context
                    display_path = os.path.join(directory, filepath)
                else:
                    display_path = filepath
                print(f"\n  {display_path}")
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

    # Save state (unless --no-save is specified)
    no_save = getattr(args, 'no_save', False)
    save_state = getattr(args, 'save_state', False)
    if save_state or (not no_save):
        reasoner.save_state()
        if verbose:
            print("\nState saved to persistence")

    print_separator()
    print("Reasoning complete!")
