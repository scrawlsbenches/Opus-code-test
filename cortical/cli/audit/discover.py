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

import sys
from pathlib import Path
from typing import Any

from ._base import (
    print_header,
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
    # Import from the script module (business logic stays there for now)
    # TODO(migration): Extract discovery logic to cortical/audits/discovery.py
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'scripts'))

    try:
        from scripts.woven_audit_discovery import (
            load_mind_state,
            save_mind_state,
            load_discovery_log,
            feed_findings_to_mind,
            extract_discoveries,
            generate_discovery_report,
            MIND_STATE_FILE,
        )
        from scripts.codebase_health import analyze_directory
    except ImportError as e:
        print(f"Error: Could not import discovery modules: {e}")
        print("Make sure scripts/woven_audit_discovery.py exists.")
        return

    from cortical.reasoning.woven_mind import WovenMind, WovenMindConfig
    from cortical.reasoning.loom import ThinkingMode

    print("WovenMind Pattern Discovery")
    print_separator()
    print("⚠️  EXPERIMENTAL: Treat outputs as hints to investigate")
    print_separator()

    # Handle show-mind command
    if getattr(args, 'show_mind', False):
        state = load_mind_state()
        if state is None:
            print("No saved mind state found.")
            return

        metadata = state.get("metadata", {})
        mind_data = state.get("mind", {})

        print(f"\nMind State:")
        print(f"  Last saved: {metadata.get('saved_at', 'unknown')}")
        print(f"  Sessions: {metadata.get('session_count', 0)}")
        print(f"  Total findings processed: {metadata.get('total_findings', 0)}")

        # Show abstractions
        cortex = mind_data.get("cortex_state", {})
        engine = cortex.get("engine_state", {})
        abstractions = engine.get("abstractions", {})

        if abstractions:
            print(f"\nLearned Abstractions: {len(abstractions)}")
            for abs_id, abs_data in list(abstractions.items())[:5]:
                nodes = abs_data.get("source_nodes", [])
                strength = abs_data.get("strength", 0)
                print(f"  {abs_id}: {nodes[:3]}... (strength={strength:.2f})")
        return

    # Handle reset-mind command
    if getattr(args, 'reset_mind', False):
        if MIND_STATE_FILE.exists():
            MIND_STATE_FILE.unlink()
            print("Mind state cleared.")
        else:
            print("No mind state to clear.")
        return

    # Run discovery
    directory = args.directory
    with_git = getattr(args, 'with_git', False)
    verbose = getattr(args, 'verbose', False)

    print(f"\nAnalyzing: {directory}")

    # Analyze codebase
    result = analyze_directory(directory, with_git=with_git, verbose=verbose)

    if not result.findings:
        print("No findings to analyze.")
        return

    print(f"Found {len(result.findings)} findings")

    # Initialize or load WovenMind
    state = load_mind_state()
    if state:
        config = WovenMindConfig(
            thinking_mode=ThinkingMode.AUTOMATIC,
            max_tokens=10000,
        )
        mind = WovenMind(config)
        mind.from_dict(state["mind"])
        session_count = state.get("metadata", {}).get("session_count", 0) + 1
        print(f"Loaded existing mind state (session {session_count})")
    else:
        config = WovenMindConfig(
            thinking_mode=ThinkingMode.AUTOMATIC,
            max_tokens=10000,
        )
        mind = WovenMind(config)
        session_count = 1
        print("Starting fresh mind")

    # Feed findings to mind
    git_analysis = result.git_analysis if hasattr(result, 'git_analysis') else {}
    feed_findings_to_mind(mind, result.findings, git_analysis)

    # Consolidate if requested
    if getattr(args, 'consolidate', False):
        print("\nRunning consolidation...")
        mind.consolidate()

    # Extract and report discoveries
    discoveries = extract_discoveries(mind)

    if discoveries.get("abstractions"):
        print(f"\nDiscovered {len(discoveries['abstractions'])} patterns:")
        for i, abs_data in enumerate(discoveries["abstractions"][:10]):
            interpretation = abs_data.get("interpretation", "Unknown pattern")
            strength = abs_data.get("strength", 0)
            print(f"  {i+1}. {interpretation} (strength={strength:.2f})")

    if discoveries.get("surprises"):
        print(f"\nSurprising findings: {len(discoveries['surprises'])}")
        for surprise in discoveries["surprises"][:5]:
            print(f"  - {surprise}")

    # Save state
    metadata = {
        "session_count": session_count,
        "total_findings": len(result.findings),
        "directory": directory,
    }
    save_mind_state(mind, metadata)
    print(f"\nMind state saved to {MIND_STATE_FILE}")

    print_separator()
    print("Discovery complete!")
