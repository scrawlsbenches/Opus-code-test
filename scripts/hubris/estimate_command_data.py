#!/usr/bin/env python3
"""
Estimate time to collect sufficient data for CommandExpert.

Analyzes current data collection rate and estimates when we'll have
enough diverse command patterns for useful predictions.

Usage:
    python scripts/hubris/estimate_command_data.py
    python scripts/hubris/estimate_command_data.py --target 500
    python scripts/hubris/estimate_command_data.py --verbose
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple


def get_actions_dir() -> Path:
    """Get the .git-ml/actions directory."""
    script_dir = Path(__file__).parent.parent.parent
    actions_dir = script_dir / '.git-ml' / 'actions'
    if actions_dir.exists():
        return actions_dir
    return Path('.git-ml/actions')


def analyze_collection_rate(actions_dir: Path) -> Dict:
    """
    Analyze command collection rate over time.

    Returns:
        Dict with collection statistics and rate estimates
    """
    stats = {
        'dates': defaultdict(lambda: {'total': 0, 'bash': 0, 'unique_commands': set(), 'python_c': 0}),
        'sessions': set(),
        'all_commands': defaultdict(int),
        'python_c_patterns': defaultdict(int),
        'first_date': None,
        'last_date': None
    }

    if not actions_dir.exists():
        return {'error': f'Actions directory not found: {actions_dir}'}

    for date_dir in sorted(actions_dir.iterdir()):
        if not date_dir.is_dir():
            continue

        date_str = date_dir.name

        for action_file in date_dir.glob('A-*.json'):
            try:
                with open(action_file) as f:
                    action = json.load(f)

                stats['dates'][date_str]['total'] += 1
                stats['sessions'].add(action.get('session_id', ''))

                # Track first/last dates
                if stats['first_date'] is None or date_str < stats['first_date']:
                    stats['first_date'] = date_str
                if stats['last_date'] is None or date_str > stats['last_date']:
                    stats['last_date'] = date_str

                ctx = action.get('context', {})
                if ctx.get('tool') != 'Bash':
                    continue

                stats['dates'][date_str]['bash'] += 1

                input_data = ctx.get('input', {})
                if isinstance(input_data, dict):
                    input_data = input_data.get('input', input_data)

                command = input_data.get('command', '') if isinstance(input_data, dict) else ''
                if not command:
                    continue

                # Normalize for uniqueness tracking
                normalized = command[:100].lower()
                stats['dates'][date_str]['unique_commands'].add(normalized)
                stats['all_commands'][normalized] += 1

                if 'python -c' in command or 'python3 -c' in command:
                    stats['dates'][date_str]['python_c'] += 1
                    stats['python_c_patterns'][normalized] += 1

            except (json.JSONDecodeError, KeyError):
                continue

    return stats


def estimate_time_to_target(stats: Dict, target_unique: int, target_python_c: int) -> Dict:
    """
    Estimate time to reach target data levels.

    Args:
        stats: Collection statistics from analyze_collection_rate
        target_unique: Target number of unique commands
        target_python_c: Target number of python -c patterns

    Returns:
        Dict with estimates
    """
    if 'error' in stats:
        return stats

    # Calculate current totals
    current_unique = len(stats['all_commands'])
    current_python_c = len(stats['python_c_patterns'])
    total_bash = sum(d['bash'] for d in stats['dates'].values())

    # Calculate collection period
    if stats['first_date'] and stats['last_date']:
        first = datetime.strptime(stats['first_date'], '%Y-%m-%d')
        last = datetime.strptime(stats['last_date'], '%Y-%m-%d')
        days_elapsed = max((last - first).days, 1)
    else:
        days_elapsed = 1

    # Calculate rates
    unique_per_day = current_unique / days_elapsed
    python_c_per_day = current_python_c / days_elapsed
    bash_per_day = total_bash / days_elapsed
    sessions_per_day = len(stats['sessions']) / days_elapsed

    # Estimate days to target (accounting for diminishing returns on uniqueness)
    # As we collect more, fewer will be unique - use logarithmic growth model
    if unique_per_day > 0:
        # Simple linear estimate (optimistic)
        days_to_unique_linear = max(0, (target_unique - current_unique) / unique_per_day)
        # Adjusted estimate (more realistic - assumes 50% diminishing returns)
        days_to_unique_adjusted = days_to_unique_linear * 1.5
    else:
        days_to_unique_linear = float('inf')
        days_to_unique_adjusted = float('inf')

    if python_c_per_day > 0:
        days_to_python_c = max(0, (target_python_c - current_python_c) / python_c_per_day)
    else:
        days_to_python_c = float('inf')

    return {
        'current': {
            'unique_commands': current_unique,
            'python_c_patterns': current_python_c,
            'total_bash_commands': total_bash,
            'sessions': len(stats['sessions']),
            'days_elapsed': days_elapsed
        },
        'rates': {
            'unique_per_day': round(unique_per_day, 2),
            'python_c_per_day': round(python_c_per_day, 2),
            'bash_per_day': round(bash_per_day, 2),
            'sessions_per_day': round(sessions_per_day, 2)
        },
        'targets': {
            'unique_commands': target_unique,
            'python_c_patterns': target_python_c
        },
        'estimates': {
            'days_to_unique_target': round(days_to_unique_adjusted, 1),
            'days_to_python_c_target': round(days_to_python_c, 1) if days_to_python_c != float('inf') else 'N/A (no python -c collected)',
            'estimated_date_unique': (datetime.now() + timedelta(days=days_to_unique_adjusted)).strftime('%Y-%m-%d') if days_to_unique_adjusted != float('inf') else 'N/A'
        },
        'recommendations': []
    }


def generate_recommendations(estimates: Dict) -> List[str]:
    """Generate actionable recommendations."""
    recs = []

    current = estimates.get('current', {})
    rates = estimates.get('rates', {})

    # Check python -c rate
    if rates.get('python_c_per_day', 0) < 1:
        recs.append("⚠️  Python -c collection rate is very low. Consider:")
        recs.append("   - Intentionally using python -c commands during sessions")
        recs.append("   - Adding template-based synthetic examples")
        recs.append("   - Creating a python -c usage guide for common tasks")

    # Check overall rate
    if rates.get('unique_per_day', 0) < 5:
        recs.append("📊 Unique command discovery rate is low. This is normal as common commands saturate.")
        recs.append("   - Focus on diverse task types to discover new patterns")

    # Check if we have enough for basic model
    if current.get('unique_commands', 0) < 100:
        recs.append("🎯 Need 100+ unique commands for basic pattern matching (currently: {})".format(
            current.get('unique_commands', 0)))
    elif current.get('unique_commands', 0) < 500:
        recs.append("🎯 Need 500+ unique commands for useful predictions (currently: {})".format(
            current.get('unique_commands', 0)))
    else:
        recs.append("✅ Sufficient unique commands for useful model")

    # Check python -c specifically
    if current.get('python_c_patterns', 0) < 10:
        recs.append("🐍 Need 10+ python -c patterns for that feature (currently: {})".format(
            current.get('python_c_patterns', 0)))

    return recs


def print_report(estimates: Dict, verbose: bool = False):
    """Print formatted report."""
    print("\n" + "=" * 60)
    print("CommandExpert Data Collection Estimate")
    print("=" * 60)

    current = estimates.get('current', {})
    rates = estimates.get('rates', {})
    targets = estimates.get('targets', {})
    est = estimates.get('estimates', {})

    print(f"\n📈 Current Data ({current.get('days_elapsed', 0)} days of collection):")
    print(f"   Unique commands:     {current.get('unique_commands', 0)}")
    print(f"   Python -c patterns:  {current.get('python_c_patterns', 0)}")
    print(f"   Total Bash commands: {current.get('total_bash_commands', 0)}")
    print(f"   Sessions:            {current.get('sessions', 0)}")

    print(f"\n📊 Collection Rates (per day):")
    print(f"   New unique commands: {rates.get('unique_per_day', 0)}")
    print(f"   Python -c patterns:  {rates.get('python_c_per_day', 0)}")
    print(f"   Bash commands:       {rates.get('bash_per_day', 0)}")
    print(f"   Sessions:            {rates.get('sessions_per_day', 0)}")

    print(f"\n🎯 Targets:")
    print(f"   Unique commands:     {targets.get('unique_commands', 0)}")
    print(f"   Python -c patterns:  {targets.get('python_c_patterns', 0)}")

    print(f"\n⏱️  Estimates:")
    print(f"   Days to unique target:    {est.get('days_to_unique_target', 'N/A')}")
    print(f"   Days to python -c target: {est.get('days_to_python_c_target', 'N/A')}")
    print(f"   Estimated ready date:     {est.get('estimated_date_unique', 'N/A')}")

    recs = generate_recommendations(estimates)
    if recs:
        print(f"\n💡 Recommendations:")
        for rec in recs:
            print(f"   {rec}")

    print("\n" + "=" * 60)

    if verbose:
        print("\nRaw data:")
        print(json.dumps(estimates, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(description='Estimate CommandExpert data collection timeline')
    parser.add_argument('--target', type=int, default=500,
                       help='Target unique commands (default: 500)')
    parser.add_argument('--target-python-c', type=int, default=50,
                       help='Target python -c patterns (default: 50)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show verbose output')
    parser.add_argument('--json', action='store_true',
                       help='Output as JSON')

    args = parser.parse_args()

    actions_dir = get_actions_dir()
    stats = analyze_collection_rate(actions_dir)

    if 'error' in stats:
        print(f"Error: {stats['error']}")
        return

    estimates = estimate_time_to_target(stats, args.target, args.target_python_c)
    estimates['recommendations'] = generate_recommendations(estimates)

    if args.json:
        print(json.dumps(estimates, indent=2, default=str))
    else:
        print_report(estimates, args.verbose)


if __name__ == '__main__':
    main()
