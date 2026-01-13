"""
CLI for managing audit patterns.

Usage:
    python -m cortical.cli.audit pattern list
    python -m cortical.cli.audit pattern add "monkeypatch" --scope code --implies needs_di_review
    python -m cortical.cli.audit pattern remove monkeypatch
"""

import argparse
import json
from pathlib import Path
from typing import Optional
from datetime import datetime

from cortical.audits.health import (
    AuditPattern,
    get_all_patterns,
    load_custom_patterns,
    DEFAULT_PATTERNS_FILE,
    DEFAULT_SUSPICIOUS_PATTERNS,
)


def print_separator():
    print("=" * 70)


def run(args: argparse.Namespace) -> int:
    """Run the pattern command."""
    if args.pattern_command == 'list':
        return cmd_list(args)
    elif args.pattern_command == 'add':
        return cmd_add(args)
    elif args.pattern_command == 'remove':
        return cmd_remove(args)
    else:
        print(f"Unknown command: {args.pattern_command}")
        return 1


def cmd_list(args: argparse.Namespace) -> int:
    """List all patterns (default + custom)."""
    print("AUDIT PATTERNS")
    print_separator()

    # Show default patterns
    print("\nDefault Patterns (comment scope):")
    print("-" * 40)
    for p in DEFAULT_SUSPICIOUS_PATTERNS:
        print(f"  {p}")

    # Show custom patterns
    custom = load_custom_patterns()
    if custom:
        print(f"\nCustom Patterns ({len(custom)}):")
        print("-" * 40)
        for p in custom:
            scope_tag = f"[{p.scope}]"
            regex_tag = "[regex]" if p.regex else ""
            implies = f" → {p.implies}" if p.implies else ""
            print(f"  {p.id:<20} {scope_tag:<10} {regex_tag:<8} {implies}")
            if p.description and args.verbose:
                print(f"    {p.description}")
    else:
        print("\nNo custom patterns defined.")
        print(f"  Add patterns to: {DEFAULT_PATTERNS_FILE}")

    print()
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    """Add a new custom pattern."""
    pattern_id = args.id or args.match.replace(" ", "_").lower()

    # Load existing patterns
    patterns_file = DEFAULT_PATTERNS_FILE
    if patterns_file.exists():
        with open(patterns_file, 'r') as f:
            data = json.load(f)
    else:
        data = {
            "version": 2,
            "description": "Custom audit patterns",
            "patterns": [],
            "created": datetime.now().isoformat(),
        }

    # Check for duplicate
    for p in data.get('patterns', []):
        if p.get('id') == pattern_id:
            print(f"Error: Pattern '{pattern_id}' already exists.")
            return 1

    # Add new pattern
    new_pattern = {
        "id": pattern_id,
        "match": args.match,
        "scope": args.scope,
    }
    if args.implies:
        new_pattern["implies"] = args.implies
    if args.strength:
        new_pattern["strength"] = args.strength
    if args.regex:
        new_pattern["regex"] = True
    if args.description:
        new_pattern["description"] = args.description

    data['patterns'].append(new_pattern)
    data['updated'] = datetime.now().isoformat()

    # Ensure directory exists
    patterns_file.parent.mkdir(parents=True, exist_ok=True)

    # Save
    with open(patterns_file, 'w') as f:
        json.dump(data, f, indent=2)

    scope_tag = f"[{args.scope}]"
    implies = f" → {args.implies}" if args.implies else ""
    print(f"Added pattern: {pattern_id} {scope_tag}{implies}")
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    """Remove a custom pattern."""
    patterns_file = DEFAULT_PATTERNS_FILE
    if not patterns_file.exists():
        print("No custom patterns file found.")
        return 1

    with open(patterns_file, 'r') as f:
        data = json.load(f)

    # Find and remove
    patterns = data.get('patterns', [])
    original_count = len(patterns)
    data['patterns'] = [p for p in patterns if p.get('id') != args.pattern_id]

    if len(data['patterns']) == original_count:
        print(f"Pattern '{args.pattern_id}' not found.")
        return 1

    data['updated'] = datetime.now().isoformat()

    with open(patterns_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Removed pattern: {args.pattern_id}")
    return 0


def setup_args(subparsers) -> argparse.ArgumentParser:
    """Set up the pattern subcommand parser."""
    parser = subparsers.add_parser(
        'pattern',
        help='Manage audit patterns',
        description='Add, list, or remove custom audit patterns'
    )

    pattern_subparsers = parser.add_subparsers(
        dest='pattern_command',
        help='Pattern commands'
    )

    # List command
    list_parser = pattern_subparsers.add_parser('list', help='List all patterns')
    list_parser.add_argument('-v', '--verbose', action='store_true',
                            help='Show descriptions')

    # Add command
    add_parser = pattern_subparsers.add_parser('add', help='Add a custom pattern')
    add_parser.add_argument('match', help='Pattern to match (string or regex)')
    add_parser.add_argument('--id', help='Pattern ID (default: derived from match)')
    add_parser.add_argument('--scope', choices=['comments', 'code', 'all'],
                           default='code', help='Where to search (default: code)')
    add_parser.add_argument('--implies', help='What the pattern implies (e.g., needs_review)')
    add_parser.add_argument('--strength', type=float, help='Confidence strength (0-1)')
    add_parser.add_argument('--regex', action='store_true', help='Treat match as regex')
    add_parser.add_argument('--description', help='Pattern description')

    # Remove command
    remove_parser = pattern_subparsers.add_parser('remove', help='Remove a custom pattern')
    remove_parser.add_argument('pattern_id', help='Pattern ID to remove')

    return parser
