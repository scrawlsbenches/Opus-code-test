"""
Export module for ML Data Collector

Handles exporting collected data in various formats for training.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any
import logging

from .config import COMMITS_DIR, CHATS_DIR

logger = logging.getLogger(__name__)


# ============================================================================
# DATA EXPORT FOR TRAINING
# ============================================================================

def _summarize_diff(hunks: List[Dict]) -> str:
    """Summarize diff hunks into a concise description for training."""
    if not hunks:
        return ""

    # Group by file
    files = {}
    for hunk in hunks:
        file = hunk.get('file', 'unknown')
        if file not in files:
            files[file] = {'add': 0, 'delete': 0, 'modify': 0}
        change_type = hunk.get('change_type', 'modify')
        files[file][change_type] = files[file].get(change_type, 0) + 1

    # Create summary
    parts = []
    for file, changes in files.items():
        change_desc = []
        if changes['add'] > 0:
            change_desc.append(f"+{changes['add']}")
        if changes['delete'] > 0:
            change_desc.append(f"-{changes['delete']}")
        if changes['modify'] > 0:
            change_desc.append(f"~{changes['modify']}")
        parts.append(f"{file}: {' '.join(change_desc)}")

    return '; '.join(parts[:10])  # Limit to first 10 files


def _export_jsonl(records: List[Dict], output_path: Path):
    """Export records as JSONL (one JSON per line)."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def _export_csv(records: List[Dict], output_path: Path):
    """Export records as CSV."""
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'type', 'timestamp', 'input', 'output',
            'session_id', 'files', 'tools_used'
        ])
        writer.writeheader()

        for record in records:
            context = record.get('context', {})
            row = {
                'type': record.get('type', ''),
                'timestamp': record.get('timestamp', ''),
                'input': record.get('input', '')[:1000],  # Truncate for CSV
                'output': record.get('output', '')[:1000],
                'session_id': context.get('session_id', ''),
                'files': '; '.join(context.get('files', []))[:500],
                'tools_used': '; '.join(context.get('tools_used', [])),
            }
            writer.writerow(row)


def _export_huggingface(records: List[Dict], output_path: Path):
    """Export records in HuggingFace Dataset dict format."""
    # HuggingFace datasets format: dict of lists
    dataset = {
        'type': [],
        'timestamp': [],
        'input': [],
        'output': [],
        'session_id': [],
        'files': [],
        'tools_used': [],
    }

    for record in records:
        context = record.get('context', {})
        dataset['type'].append(record.get('type', ''))
        dataset['timestamp'].append(record.get('timestamp', ''))
        dataset['input'].append(record.get('input', ''))
        dataset['output'].append(record.get('output', ''))
        dataset['session_id'].append(context.get('session_id', ''))
        dataset['files'].append(context.get('files', []))
        dataset['tools_used'].append(context.get('tools_used', []))

    # Save as JSON in HuggingFace format
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)


def export_data(format: str, output_path: Path) -> Dict[str, Any]:
    """Export collected ML data in training-ready formats.

    Args:
        format: Output format (jsonl, csv, huggingface)
        output_path: Path to write the exported data

    Returns:
        Stats dict with counts and file paths

    Raises:
        ValueError: If format is invalid
    """
    from .persistence import ensure_dirs

    ensure_dirs()

    # Collect all data
    all_records = []

    # Load commits
    if COMMITS_DIR.exists():
        for commit_file in COMMITS_DIR.glob("*.json"):
            try:
                with open(commit_file, 'r', encoding='utf-8') as f:
                    commit_data = json.load(f)

                # Transform commit to training format
                record = {
                    "type": "commit",
                    "timestamp": commit_data.get('timestamp', ''),
                    "input": commit_data.get('message', ''),
                    "output": _summarize_diff(commit_data.get('hunks', [])),
                    "context": {
                        "files": commit_data.get('files_changed', []),
                        "session_id": commit_data.get('session_id', ''),
                        "tools_used": [],
                        "insertions": commit_data.get('insertions', 0),
                        "deletions": commit_data.get('deletions', 0),
                        "branch": commit_data.get('branch', ''),
                    }
                }
                all_records.append(record)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error reading commit {commit_file}: {e}")

    # Load chats
    if CHATS_DIR.exists():
        for chat_file in CHATS_DIR.glob("**/*.json"):
            try:
                with open(chat_file, 'r', encoding='utf-8') as f:
                    chat_data = json.load(f)

                record = {
                    "type": "chat",
                    "timestamp": chat_data.get('timestamp', ''),
                    "input": chat_data.get('query', ''),
                    "output": chat_data.get('response', ''),
                    "context": {
                        "files": chat_data.get('files_referenced', []) + chat_data.get('files_modified', []),
                        "session_id": chat_data.get('session_id', ''),
                        "tools_used": chat_data.get('tools_used', []),
                    }
                }
                all_records.append(record)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error reading chat {chat_file}: {e}")

    # Sort by timestamp
    all_records.sort(key=lambda r: r['timestamp'])

    # Export based on format
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "jsonl":
        _export_jsonl(all_records, output_path)
    elif format == "csv":
        _export_csv(all_records, output_path)
    elif format == "huggingface":
        _export_huggingface(all_records, output_path)
    else:
        raise ValueError(f"Unknown format: {format}")

    return {
        "format": format,
        "output_path": str(output_path),
        "records": len(all_records),
        "commits": sum(1 for r in all_records if r['type'] == 'commit'),
        "chats": sum(1 for r in all_records if r['type'] == 'chat'),
    }
