#!/usr/bin/env python3
"""
Demo script showing different origin status states in the GoT dashboard.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.got_dashboard import render_git_integration_section


def demo_all_states():
    """Demonstrate all possible origin status states."""

    scenarios = [
        {
            'name': 'Up-to-date',
            'stats': {
                'branch': 'main',
                'is_main': True,
                'drift': None,
                'uncommitted_files': 0,
                'recent_task_commits': [],
                'origin_status': {
                    'status': 'up-to-date',
                    'behind_count': 0,
                    'ahead_count': 0,
                    'message': 'Up-to-date with origin/main',
                    'last_fetch': '2m ago',
                }
            }
        },
        {
            'name': 'Behind origin (minor)',
            'stats': {
                'branch': 'feature-branch',
                'is_main': False,
                'drift': {'ahead': 2, 'behind': 1},
                'uncommitted_files': 1,
                'recent_task_commits': [],
                'origin_status': {
                    'status': 'behind',
                    'behind_count': 3,
                    'ahead_count': 0,
                    'message': 'Behind origin/feature-branch by 3 commits',
                    'last_fetch': '10m ago',
                }
            }
        },
        {
            'name': 'Behind origin (SIGNIFICANT - shows warning)',
            'stats': {
                'branch': 'feature-branch',
                'is_main': False,
                'drift': {'ahead': 2, 'behind': 1},
                'uncommitted_files': 1,
                'recent_task_commits': [],
                'origin_status': {
                    'status': 'behind',
                    'behind_count': 10,  # >= 5 shows warning
                    'ahead_count': 0,
                    'message': 'Behind origin/feature-branch by 10 commits',
                    'last_fetch': '1h ago',
                }
            }
        },
        {
            'name': 'Ahead of origin',
            'stats': {
                'branch': 'feature-branch',
                'is_main': False,
                'drift': {'ahead': 2, 'behind': 1},
                'uncommitted_files': 2,
                'recent_task_commits': [],
                'origin_status': {
                    'status': 'ahead',
                    'behind_count': 0,
                    'ahead_count': 5,
                    'message': 'Ahead of origin/feature-branch by 5 commits',
                    'last_fetch': '5m ago',
                }
            }
        },
        {
            'name': 'Diverged from origin',
            'stats': {
                'branch': 'feature-branch',
                'is_main': False,
                'drift': {'ahead': 2, 'behind': 1},
                'uncommitted_files': 1,
                'recent_task_commits': [],
                'origin_status': {
                    'status': 'diverged',
                    'behind_count': 3,
                    'ahead_count': 5,
                    'message': 'Diverged from origin/feature-branch: +5 -3',
                    'last_fetch': '15m ago',
                }
            }
        },
        {
            'name': 'No upstream configured',
            'stats': {
                'branch': 'local-only-branch',
                'is_main': False,
                'drift': None,
                'uncommitted_files': 0,
                'recent_task_commits': [],
                'origin_status': {
                    'status': 'no-upstream',
                    'behind_count': 0,
                    'ahead_count': 0,
                    'message': "Branch 'local-only-branch' has no upstream configured",
                    'last_fetch': None,
                }
            }
        },
        {
            'name': 'Network error',
            'stats': {
                'branch': 'feature-branch',
                'is_main': False,
                'drift': None,
                'uncommitted_files': 0,
                'recent_task_commits': [],
                'origin_status': {
                    'status': 'error',
                    'behind_count': 0,
                    'ahead_count': 0,
                    'message': 'Network timeout during fetch',
                    'last_fetch': None,
                }
            }
        },
    ]

    for scenario in scenarios:
        print(f"\n{'='*80}")
        print(f"SCENARIO: {scenario['name']}")
        print('='*80)

        lines = render_git_integration_section(scenario['stats'])
        for line in lines:
            print(line)


if __name__ == '__main__':
    demo_all_states()
