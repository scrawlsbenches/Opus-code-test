"""
Discover command - WovenMind pattern discovery for audits.

Uses WovenMind dual-process cognitive architecture to discover
emergent patterns in codebase audits through:
- Hebbian learning for co-occurrence patterns
- Abstraction formation for higher-level concepts
- Surprise detection for anomalies

Note: This is an experimental research tool. Treat outputs as
hints to investigate, not definitive findings.
"""

from typing import Any

from ._base import (
    print_separator,
)


def setup_args(subparsers) -> None:
    """Set up command arguments."""
    parser = subparsers.add_parser(
        'discover',
        help='WovenMind pattern discovery (experimental)'
    )
    parser.add_argument(
        'directory',
        nargs='?',
        default='cortical/',
        help='Directory to analyze'
    )
    parser.add_argument(
        '--with-git',
        action='store_true',
        help='Include git history analysis'
    )
    parser.add_argument(
        '--show-mind',
        action='store_true',
        help='Show learned WovenMind state'
    )
    parser.add_argument(
        '--reset-mind',
        action='store_true',
        help='Clear learned state and start fresh'
    )
    parser.add_argument(
        '--consolidate',
        action='store_true',
        help='Run learning consolidation cycle'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed output'
    )


def run(args: Any) -> None:
    """Execute the discover command."""
    from cortical.audits import (
        analyze_directory,
    )
    from cortical.audits.discovery import (
        WovenMindDiscovery,
        DiscoveryConfig,
        generate_discovery_report,
    )

    print("WovenMind Pattern Discovery")
    print_separator()
    print("⚠️  EXPERIMENTAL: Treat outputs as hints to investigate")
    print_separator()

    # Initialize discovery system
    config = DiscoveryConfig()
    discovery = WovenMindDiscovery(config=config)

    # Handle reset-mind command
    if getattr(args, 'reset_mind', False):
        discovery.reset()
        print("Mind state cleared. Fresh start on next run.")
        return

    # Load existing state
    discovery.load_or_create_mind()

    # Handle show-mind command
    if getattr(args, 'show_mind', False):
        stats = discovery.get_mind_stats()
        if not stats:
            print("No saved mind state found.")
            return

        print(f"\nMind State:")
        print(f"  Session: {discovery.session_num}")
        print(f"  Mode: {stats.get('mode', 'unknown')}")

        cortex = stats.get('cortex', {})
        print(f"\nCortex:")
        print(f"  Observations: {cortex.get('total_observations', 0)}")
        print(f"  Unique patterns: {cortex.get('unique_patterns', 0)}")
        print(f"  Abstractions: {cortex.get('total_abstractions', 0)}")

        abstractions = discovery.get_abstractions()
        if abstractions:
            print(f"\nFormed Abstractions:")
            for a in abstractions[:5]:
                print(f"  [{a['id']}] strength={a['strength']:.2f}, freq={a['frequency']}")
                print(f"    → {a['interpretation']}")
        return

    # Handle consolidation
    if getattr(args, 'consolidate', False):
        print("\nRunning consolidation cycle...")
        result = discovery.consolidate()
        print(f"  Patterns transferred: {result['patterns_transferred']}")
        print(f"  Abstractions formed: {result['abstractions_formed']}")
        print(f"  Cycle duration: {result['cycle_duration_ms']:.1f}ms")
        discovery.save_state()
        print("\nState saved.")
        return

    # Normal analysis run
    directory = args.directory
    with_git = getattr(args, 'with_git', False)
    verbose = getattr(args, 'verbose', False)

    print(f"\nAnalyzing: {directory}")

    # Analyze codebase
    result = analyze_directory(directory, with_git=with_git, verbose=False)

    if not result.findings:
        print("No findings to analyze.")
        return

    print(f"Found {len(result.findings)} findings")

    # Run discovery
    git_analysis = result.git_analysis if hasattr(result, 'git_analysis') else {}
    discovery_result = discovery.run_discovery(result.findings, git_analysis)

    # Generate and print report
    report = generate_discovery_report(discovery_result, discovery.mind)
    print(report)

    # Save state
    discovery.save_state()
    print(f"\nState saved for session {discovery.session_num}")

    print_separator()
    print("Discovery complete!")
