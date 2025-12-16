"""
Data quality analysis module for ML Data Collector

Analyzes completeness, diversity, and anomalies in collected data.
"""

import json
import hashlib
import logging
from typing import Dict, Any

from .config import COMMITS_DIR, CHATS_DIR, SESSIONS_DIR, CHAT_SCHEMA, validate_schema
from .persistence import ensure_dirs

logger = logging.getLogger(__name__)

def analyze_data_quality() -> Dict[str, Any]:
    """Analyze data quality across all collected ML data.
    
    Returns:
        Dictionary with completeness, diversity, anomalies, and quality score.
    """
    ensure_dirs()
    
    # Initialize metrics containers
    completeness = {
        'chats_complete': 0,
        'chats_total': 0,
        'commits_with_ci': 0,
        'commits_total': 0,
        'sessions_with_commits': 0,
        'sessions_total': 0,
        'chats_with_feedback': 0,
    }
    
    diversity = {
        'unique_files': set(),
        'unique_tools': {},
        'query_lengths': [],
        'response_lengths': [],
    }
    
    anomalies = {
        'empty_responses': 0,
        'zero_file_commits': 0,
        'empty_sessions': 0,
        'potential_duplicates': 0,
    }
    
    # Track duplicates (timestamp + content hash)
    seen_entries = set()
    
    # Analyze commits
    if COMMITS_DIR.exists():
        for commit_file in COMMITS_DIR.glob("*.json"):
            try:
                with open(commit_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                completeness['commits_total'] += 1
                
                # Check CI results
                if data.get('ci_result'):
                    completeness['commits_with_ci'] += 1
                
                # Track files
                diversity['unique_files'].update(data.get('files_changed', []))
                
                # Check anomalies
                if not data.get('files_changed'):
                    anomalies['zero_file_commits'] += 1
                
                # Check duplicates (timestamp + message hash)
                entry_key = (data.get('timestamp', ''),
                           hashlib.md5(data.get('message', '').encode()).hexdigest()[:8])
                if entry_key in seen_entries:
                    anomalies['potential_duplicates'] += 1
                else:
                    seen_entries.add(entry_key)
                    
            except (json.JSONDecodeError, IOError):
                continue
    
    # Analyze chats
    if CHATS_DIR.exists():
        for chat_file in CHATS_DIR.glob("**/*.json"):
            try:
                with open(chat_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                completeness['chats_total'] += 1
                
                # Check completeness (all required fields from CHAT_SCHEMA)
                errors = validate_schema(data, CHAT_SCHEMA, "chat")
                if not errors:
                    completeness['chats_complete'] += 1
                
                # Check feedback
                if data.get('user_feedback'):
                    completeness['chats_with_feedback'] += 1
                
                # Track diversity
                diversity['unique_files'].update(data.get('files_referenced', []))
                diversity['unique_files'].update(data.get('files_modified', []))
                
                for tool in data.get('tools_used', []):
                    diversity['unique_tools'][tool] = diversity['unique_tools'].get(tool, 0) + 1
                
                query = data.get('query', '')
                response = data.get('response', '')
                
                diversity['query_lengths'].append(len(query))
                diversity['response_lengths'].append(len(response))
                
                # Check anomalies
                if not response or len(response.strip()) == 0:
                    anomalies['empty_responses'] += 1
                
                # Check duplicates (timestamp + query hash)
                entry_key = (data.get('timestamp', ''),
                           hashlib.md5(query.encode()).hexdigest()[:8])
                if entry_key in seen_entries:
                    anomalies['potential_duplicates'] += 1
                else:
                    seen_entries.add(entry_key)
                    
            except (json.JSONDecodeError, IOError):
                continue
    
    # Analyze sessions
    if SESSIONS_DIR.exists():
        for session_file in SESSIONS_DIR.glob("*.json"):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                completeness['sessions_total'] += 1
                
                # Check if session has chats
                if not data.get('chat_ids'):
                    anomalies['empty_sessions'] += 1
                    
            except (json.JSONDecodeError, IOError):
                continue
    
    # Count sessions with commits by checking commits with session_id
    session_ids_with_commits = set()
    if COMMITS_DIR.exists():
        for commit_file in COMMITS_DIR.glob("*.json"):
            try:
                with open(commit_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    session_id = data.get('session_id')
                    if session_id:
                        session_ids_with_commits.add(session_id)
            except (json.JSONDecodeError, IOError):
                continue
    
    completeness['sessions_with_commits'] = len(session_ids_with_commits)
    
    # Calculate percentages for completeness
    completeness_metrics = {
        'chats_complete_pct': (completeness['chats_complete'] / max(1, completeness['chats_total'])) * 100,
        'commits_with_ci_pct': (completeness['commits_with_ci'] / max(1, completeness['commits_total'])) * 100,
        'sessions_with_commits_pct': (completeness['sessions_with_commits'] / max(1, completeness['sessions_total'])) * 100,
        'chats_with_feedback_pct': (completeness['chats_with_feedback'] / max(1, completeness['chats_total'])) * 100,
    }
    
    # Calculate diversity statistics
    diversity_stats = {
        'unique_files': len(diversity['unique_files']),
        'unique_tools': len(diversity['unique_tools']),
        'tool_distribution': diversity['unique_tools'],
        'query_length_min': min(diversity['query_lengths']) if diversity['query_lengths'] else 0,
        'query_length_avg': sum(diversity['query_lengths']) / max(1, len(diversity['query_lengths'])) if diversity['query_lengths'] else 0,
        'query_length_max': max(diversity['query_lengths']) if diversity['query_lengths'] else 0,
        'response_length_min': min(diversity['response_lengths']) if diversity['response_lengths'] else 0,
        'response_length_avg': sum(diversity['response_lengths']) / max(1, len(diversity['response_lengths'])) if diversity['response_lengths'] else 0,
        'response_length_max': max(diversity['response_lengths']) if diversity['response_lengths'] else 0,
    }
    
    # Calculate quality score (0-100)
    # Weighted components:
    # - Completeness: 40%
    # - Low anomalies: 30%
    # - Diversity: 30%
    
    # Completeness score (average of all completeness metrics)
    completeness_score = (
        completeness_metrics['chats_complete_pct'] * 0.4 +
        completeness_metrics['commits_with_ci_pct'] * 0.2 +
        completeness_metrics['sessions_with_commits_pct'] * 0.3 +
        completeness_metrics['chats_with_feedback_pct'] * 0.1
    )
    
    # Anomaly score (penalize based on anomaly percentage)
    total_entries = completeness['chats_total'] + completeness['commits_total'] + completeness['sessions_total']
    total_anomalies = (anomalies['empty_responses'] + anomalies['zero_file_commits'] +
                      anomalies['empty_sessions'] + anomalies['potential_duplicates'])
    anomaly_rate = total_anomalies / max(1, total_entries)
    anomaly_score = max(0, 100 - (anomaly_rate * 200))  # Cap at 0, scale anomalies harshly
    
    # Diversity score (based on having diverse tools and files)
    # Good diversity: >5 tools, >50 files = 100%, scale down from there
    tool_score = min(100, (diversity_stats['unique_tools'] / 5.0) * 100)
    file_score = min(100, (diversity_stats['unique_files'] / 50.0) * 100)
    diversity_score = (tool_score + file_score) / 2
    
    # Overall quality score
    quality_score = int(
        completeness_score * 0.4 +
        anomaly_score * 0.3 +
        diversity_score * 0.3
    )
    
    return {
        'completeness': {
            'chats_complete': completeness['chats_complete'],
            'chats_total': completeness['chats_total'],
            'chats_complete_pct': completeness_metrics['chats_complete_pct'],
            'commits_with_ci': completeness['commits_with_ci'],
            'commits_total': completeness['commits_total'],
            'commits_with_ci_pct': completeness_metrics['commits_with_ci_pct'],
            'sessions_with_commits': completeness['sessions_with_commits'],
            'sessions_total': completeness['sessions_total'],
            'sessions_with_commits_pct': completeness_metrics['sessions_with_commits_pct'],
            'chats_with_feedback': completeness['chats_with_feedback'],
            'chats_with_feedback_pct': completeness_metrics['chats_with_feedback_pct'],
        },
        'diversity': diversity_stats,
        'anomalies': anomalies,
        'quality_score': quality_score,
    }


def print_quality_report():
    """Print a comprehensive data quality report."""
    result = analyze_data_quality()
    
    comp = result['completeness']
    div = result['diversity']
    anom = result['anomalies']
    score = result['quality_score']
    
    print("\n" + "=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)
    
    print("\n📊 Completeness:")
    print(f"   Chats with all fields:    {comp['chats_complete_pct']:>3.0f}% ({comp['chats_complete']}/{comp['chats_total']})")
    print(f"   Commits with CI results:  {comp['commits_with_ci_pct']:>3.0f}% ({comp['commits_with_ci']}/{comp['commits_total']})")
    print(f"   Sessions with commits:    {comp['sessions_with_commits_pct']:>3.0f}% ({comp['sessions_with_commits']}/{comp['sessions_total']})")
    print(f"   Chats with feedback:      {comp['chats_with_feedback_pct']:>3.0f}% ({comp['chats_with_feedback']}/{comp['chats_total']})")
    
    print("\n📈 Diversity:")
    print(f"   Unique files:             {div['unique_files']}")
    print(f"   Unique tools:             {div['unique_tools']}")
    if div['tool_distribution']:
        print("   Tool usage:")
        for tool, count in sorted(div['tool_distribution'].items(), key=lambda x: -x[1])[:8]:
            print(f"      {tool}: {count}")
        if len(div['tool_distribution']) > 8:
            print(f"      ... and {len(div['tool_distribution']) - 8} more")
    print(f"   Query length:             min={div['query_length_min']}, avg={div['query_length_avg']:.0f}, max={div['query_length_max']} chars")
    print(f"   Response length:          min={div['response_length_min']}, avg={div['response_length_avg']:.0f}, max={div['response_length_max']} chars")
    
    print("\n⚠️  Anomalies:")
    print(f"   Empty responses:          {anom['empty_responses']}")
    print(f"   Zero-file commits:        {anom['zero_file_commits']}")
    print(f"   Empty sessions:           {anom['empty_sessions']}")
    print(f"   Potential duplicates:     {anom['potential_duplicates']}")
    
    print(f"\n🎯 Quality Score: {score}/100")
    print("=" * 60 + "\n")
