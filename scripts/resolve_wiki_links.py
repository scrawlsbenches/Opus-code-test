#!/usr/bin/env python3
"""Wiki-link cross-reference resolution for memory documents."""

import re
import sys
import json
import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict


def extract_wiki_links(content: str) -> List[str]:
    """Parse [[link]] patterns from markdown content."""
    return re.findall(r'\[\[([^\]]+)\]\]', content)


def resolve_link(link: str, source_file: str, search_dirs: List[str]) -> Optional[str]:
    """Resolve wiki-link to file path (exact, fuzzy, or date-based match)."""
    source_dir = Path(source_file).resolve().parent

    # Try exact path match (relative to source file)
    candidate = source_dir / link
    if candidate.exists() and candidate.is_file():
        return str(candidate.resolve())

    # Try fuzzy filename match in search directories
    link_name = Path(link).name
    for search_dir in search_dirs:
        search_path = Path(search_dir)
        if search_path.exists():
            for file_path in search_path.rglob('*.md'):
                if file_path.name == link_name:
                    return str(file_path.resolve())

    # Try date-based match (YYYY-MM-DD)
    if re.match(r'^\d{4}-\d{2}-\d{2}$', link):
        for search_dir in search_dirs:
            search_path = Path(search_dir)
            if search_path.exists():
                for file_path in search_path.rglob('*.md'):
                    if link in file_path.name:
                        return str(file_path.resolve())

    return None


def find_backlinks(target_file: str, search_dirs: List[str]) -> List[Tuple[str, int]]:
    """Find all files that link to the target file."""
    target_path = Path(target_file).resolve()
    target_name = target_path.name
    backlinks = []

    for search_dir in search_dirs:
        search_path = Path(search_dir)
        if not search_path.exists():
            continue
        for file_path in search_path.rglob('*.md'):
            if file_path.resolve() == target_path:
                continue
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, start=1):
                        for link in extract_wiki_links(line):
                            resolved = resolve_link(link, str(file_path), search_dirs)
                            if (resolved and Path(resolved) == target_path) or Path(link).name == target_name:
                                backlinks.append((str(file_path.resolve()), line_num))
            except (IOError, UnicodeDecodeError):
                continue

    return backlinks


def generate_link_report(file_path: str, search_dirs: List[str]) -> Dict:
    """Generate a report of all wiki-links in a file."""
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        return {'error': f'File not found: {file_path}'}

    try:
        with open(file_path_obj, 'r', encoding='utf-8') as f:
            content = f.read()
    except (IOError, UnicodeDecodeError) as e:
        return {'error': f'Error reading file: {e}'}

    links = extract_wiki_links(content)
    resolved, broken = {}, []
    for link in links:
        target = resolve_link(link, file_path, search_dirs)
        (resolved.__setitem__(link, target) if target else broken.append(link))

    return {'file': str(file_path_obj.resolve()), 'links': links, 'resolved': resolved, 'broken': broken}


def main():
    """CLI for wiki-link resolution."""
    parser = argparse.ArgumentParser(description='Parse and resolve wiki-style links in markdown files')
    parser.add_argument('file', nargs='?', help='File to analyze')
    parser.add_argument('--backlinks', action='store_true', help='Show files that link to the specified file')
    parser.add_argument('--check', metavar='DIR', help='Check all links in directory')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    parser.add_argument('--search-dirs', nargs='+', default=['samples/memories', 'samples/decisions', 'docs'],
                        help='Directories to search for link targets')
    args = parser.parse_args()

    if args.check:
        check_dir = Path(args.check)
        if not check_dir.exists():
            print(f"Error: Directory not found: {args.check}", file=sys.stderr)
            return 1
        results = {str(fp): r for fp in check_dir.rglob('*.md')
                   if (r := generate_link_report(str(fp), args.search_dirs)).get('broken')}
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print(f"✓ All links in {args.check} are valid" if not results else
                  f"Broken links found in {args.check}:")
            for file_path, report in results.items():
                print(f"\n{file_path}:")
                for broken in report['broken']:
                    print(f"  ✗ [[{broken}]] → NOT FOUND")
        return 0

    if not args.file:
        parser.print_help()
        return 1

    if args.backlinks:
        backlinks = find_backlinks(args.file, args.search_dirs)
        if args.json:
            print(json.dumps({'target': args.file, 'backlinks': [{'file': f, 'line': ln} for f, ln in backlinks]}, indent=2))
        else:
            print(f"Backlinks to {args.file}:")
            print("  (none)" if not backlinks else '\n'.join(f"  - {f}:{ln}" for f, ln in backlinks))
    else:
        report = generate_link_report(args.file, args.search_dirs)
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            if 'error' in report:
                print(f"Error: {report['error']}", file=sys.stderr)
                return 1
            print(f"Links in {report['file']}:")
            print("  (none)" if not report['links'] else '\n'.join(
                f"  ✓ [[{link}]] → {report['resolved'][link]}" if link in report['resolved']
                else f"  ✗ [[{link}]] → NOT FOUND" for link in report['links']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
