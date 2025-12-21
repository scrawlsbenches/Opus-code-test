# Implementation Summary: Commits Behind Origin Indicator

## Task
**T-20251220-194437-f898** - Add 'commits behind origin' indicator to GoT dashboard

## Overview
Added a comprehensive origin tracking feature to the GoT dashboard that shows how many commits the local branch is behind/ahead of its upstream origin, helping users know when to pull changes.

## Files Modified

### 1. `/home/user/Opus-code-test/scripts/got_dashboard.py`

#### New Function: `get_commits_behind_origin()`
- **Location**: Lines 462-596 in `DashboardMetrics` class
- **Features**:
  - Fetches from origin with 10-second timeout
  - Checks upstream tracking configuration
  - Calculates ahead/behind counts using `git rev-list --left-right --count`
  - Tracks last fetch time from `.git/FETCH_HEAD`
  - Returns structured dict with status, counts, and message

- **Status Values**:
  - `up-to-date`: Branch is synced with origin
  - `behind`: Branch is behind origin
  - `ahead`: Branch is ahead of origin
  - `diverged`: Branch has diverged (both ahead and behind)
  - `no-upstream`: No upstream tracking configured
  - `error`: Git error or network timeout

- **Error Handling**:
  - Network timeouts (10s max)
  - No upstream configured
  - Git command failures
  - Unparseable output
  - Generic exceptions

#### Modified Function: `get_git_integration_status()`
- **Location**: Lines 598-679
- **Change**: Added call to `get_commits_behind_origin()` and includes result in return dict as `origin_status`

#### Modified Function: `render_git_integration_section()`
- **Location**: Lines 837-924
- **Changes**:
  - Added origin status display with visual indicators
  - Shows last fetch time when available
  - Displays warning with tip when ≥5 commits behind
  - Color-coded status indicators:
    - ✓ Green (up-to-date)
    - ↑ Cyan (ahead)
    - ↓ Yellow (behind, <5 commits)
    - ⚠️ Red (behind, ≥5 commits)
    - ⇅ Yellow (diverged)
    - ⓘ Dim (no upstream)
    - ✗ Dim (error)

### 2. `/home/user/Opus-code-test/tests/unit/test_got_dashboard.py` (NEW)
- **13 comprehensive tests** covering:
  - Up-to-date scenario
  - Behind origin (3 commits)
  - Ahead of origin (2 commits)
  - Diverged scenario (5 ahead, 3 behind)
  - No upstream configured
  - Network timeout handling
  - Git error handling
  - Last fetch time calculation
  - Singular/plural grammar ("1 commit" vs "2 commits")
  - Integration with `get_git_integration_status()`
  - Rendering with warning (≥5 behind)
  - Rendering up-to-date
  - Rendering no upstream

### 3. `/home/user/Opus-code-test/examples/got_dashboard_origin_demo.py` (NEW)
- **Demo script** showcasing all possible status states
- Visual examples of each scenario

## Test Results
```
============================= test session starts ==============================
collected 13 items

tests/unit/test_got_dashboard.py::TestGetCommitsBehindOrigin::test_ahead_of_origin PASSED
tests/unit/test_got_dashboard.py::TestGetCommitsBehindOrigin::test_behind_origin PASSED
tests/unit/test_got_dashboard.py::TestGetCommitsBehindOrigin::test_diverged PASSED
tests/unit/test_got_dashboard.py::TestGetCommitsBehindOrigin::test_git_error PASSED
tests/unit/test_got_dashboard.py::TestGetCommitsBehindOrigin::test_last_fetch_time PASSED
tests/unit/test_got_dashboard.py::TestGetCommitsBehindOrigin::test_network_timeout PASSED
tests/unit/test_got_dashboard.py::TestGetCommitsBehindOrigin::test_no_upstream_configured PASSED
tests/unit/test_got_dashboard.py::TestGetCommitsBehindOrigin::test_single_commit_grammar PASSED
tests/unit/test_got_dashboard.py::TestGetCommitsBehindOrigin::test_up_to_date PASSED
tests/unit/test_got_dashboard.py::TestGitIntegrationStatus::test_integration_includes_origin_status PASSED
tests/unit/test_got_dashboard.py::TestDashboardRendering::test_render_handles_no_upstream PASSED
tests/unit/test_got_dashboard.py::TestDashboardRendering::test_render_shows_origin_warning PASSED
tests/unit/test_got_dashboard.py::TestDashboardRendering::test_render_shows_up_to_date PASSED

============================== 13 passed in 0.37s
```

## Example Dashboard Output

### Up-to-date
```
│ Current Branch: main                                                               │
│ Origin Status: ✓ Up-to-date with origin/main                                 │
│ Last Fetch: 2m ago                                                             │
```

### Behind (minor)
```
│ Current Branch: feature-branch                                                     │
│ Origin Status: ↓ Behind origin/feature-branch by 3 commits                   │
│ Last Fetch: 10m ago                                                            │
```

### Behind (significant - with warning)
```
│ Current Branch: feature-branch                                                     │
│ Origin Status: ⚠️ Behind origin/feature-branch by 10 commits                 │
│ Last Fetch: 1h ago                                                             │
│                                                                              │
│ ⚠️  Tip: Run git pull to sync with origin                                 │
```

### Ahead
```
│ Current Branch: feature-branch                                                     │
│ Origin Status: ↑ Ahead of origin/feature-branch by 5 commits                 │
│ Last Fetch: 5m ago                                                             │
```

### Diverged
```
│ Current Branch: feature-branch                                                     │
│ Origin Status: ⇅ Diverged from origin/feature-branch: +5 -3                  │
│ Last Fetch: 15m ago                                                            │
```

### No Upstream
```
│ Current Branch: local-only-branch                                                  │
│ Origin Status: ⓘ Branch 'local-only-branch' has no upstream configured            │
```

### Error
```
│ Current Branch: feature-branch                                                     │
│ Origin Status: ✗ Network timeout during fetch                                │
```

## Implementation Details

### Git Commands Used
1. `git fetch --quiet` - Update refs from origin (with 10s timeout)
2. `git rev-parse --abbrev-ref HEAD` - Get current branch name
3. `git rev-parse --abbrev-ref @{upstream}` - Check upstream tracking
4. `git rev-list --left-right --count HEAD...@{upstream}` - Get ahead/behind counts

### Design Decisions

1. **Automatic fetch**: The dashboard fetches on every run to ensure fresh data
   - Uses `--quiet` flag to suppress output
   - 10-second timeout to prevent hanging
   - Graceful degradation on network errors

2. **Warning threshold**: Shows red warning icon and tip when ≥5 commits behind
   - Balances between being helpful and avoiding noise
   - Users working on long-running branches won't see false alarms

3. **Visual indicators**: Different icons for each status
   - Makes status immediately recognizable at a glance
   - Color-coding provides additional context

4. **Last fetch time**: Helps users understand data freshness
   - Only shown when `.git/FETCH_HEAD` exists
   - Human-readable format (just now, 5m ago, 1h ago, 2d ago)

5. **Error handling**: Comprehensive error cases
   - Network timeouts
   - No upstream configured
   - Git command failures
   - All errors return structured dict (never crash)

## Demo
Run the demo to see all status states:
```bash
python examples/got_dashboard_origin_demo.py
```

## Testing
Run the test suite:
```bash
python -m pytest tests/unit/test_got_dashboard.py -v
```

## Requirements Met

✅ Add function to get commits behind count
✅ Display in dashboard under Git Integration section
✅ Show warning if significantly behind (5+ commits)
✅ Handle offline/no-remote gracefully
✅ Comprehensive tests for all scenarios
✅ Visual indicators with icons and colors
✅ Last fetch time tracking
✅ Grammar correctness (1 commit vs 2 commits)

## Additional Features

Beyond the requirements, also implemented:
- Ahead of origin indicator
- Diverged state indicator
- Last fetch time display
- Automatic fetch on dashboard run
- Network timeout protection
- Demo script showcasing all states
- 13 comprehensive unit tests
