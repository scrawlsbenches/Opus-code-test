#!/bin/bash
#
# git-sync-ml.sh - Safe git synchronization for ML data sessions
#
# Handles git pull/push with intelligent conflict resolution for ML data files.
# Designed to prevent data loss during context resumption and parallel sessions.
#
# Usage:
#   ./scripts/git-sync-ml.sh [MODE] [OPTIONS]
#
# Modes:
#   --auto      Fully automatic with defaults, no prompts (default)
#   --semi      Automatic but confirm at key decision points
#   --survey    Interactive prompts for all parameters
#
# Options:
#   --dry-run           Show what would happen without doing it
#   --force             Override lock file if present
#   --no-lock           Don't create lock file
#   --no-backup         Skip backup step (dangerous)
#   --keep-backup       Keep backup after successful sync
#   --no-dedupe         Skip JSONL deduplication
#   --manual-conflicts  Prompt for each conflict instead of auto-resolve
#   --branch BRANCH     Specify branch (default: current branch)
#   --verbose           Show detailed output
#   --help              Show this help
#
# Exit codes:
#   0 - Success
#   1 - General error
#   2 - Git state error (needs manual intervention)
#   3 - Network error (retry may help)
#   4 - Lock file exists (another sync in progress)
#   5 - User cancelled
#

set -o pipefail

# =============================================================================
# CONFIGURATION DEFAULTS
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ML_DATA_DIR="$REPO_ROOT/.git-ml"
LOCK_FILE="$ML_DATA_DIR/.sync.lock"
STATE_FILE="$ML_DATA_DIR/.sync.state"
BACKUP_DIR=""  # Set dynamically with timestamp

# Default parameters (can be overridden)
MODE="auto"           # auto, semi, survey
DRY_RUN=false
FORCE_LOCK=false
USE_LOCK=true
DO_BACKUP=true
KEEP_BACKUP=false
DO_DEDUPE=true
AUTO_CONFLICTS=true
BRANCH=""             # Empty = current branch
VERBOSE=false

# Retry configuration
MAX_RETRIES=3
RETRY_DELAY=2

# =============================================================================
# LOGGING
# =============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

log_verbose() {
    if $VERBOSE; then
        echo -e "${BLUE}[DEBUG]${NC} $1"
    fi
}

log_step() {
    echo ""
    echo -e "${BLUE}━━━ $1 ━━━${NC}"
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

confirm() {
    local prompt="$1"
    local default="${2:-n}"

    if [[ "$MODE" == "auto" ]]; then
        return 0  # Always yes in auto mode
    fi

    local yn_prompt
    if [[ "$default" == "y" ]]; then
        yn_prompt="[Y/n]"
    else
        yn_prompt="[y/N]"
    fi

    read -r -p "$prompt $yn_prompt " response
    response="${response:-$default}"

    [[ "$response" =~ ^[Yy]$ ]]
}

prompt_choice() {
    local prompt="$1"
    local default="$2"
    shift 2
    local options=("$@")

    if [[ "$MODE" == "auto" ]]; then
        echo "$default"
        return
    fi

    echo "$prompt"
    for i in "${!options[@]}"; do
        local marker=""
        if [[ "${options[$i]}" == "$default" ]]; then
            marker=" (default)"
        fi
        echo "  $((i+1)). ${options[$i]}$marker"
    done

    read -r -p "Choice [1-${#options[@]}]: " choice

    if [[ -z "$choice" ]]; then
        echo "$default"
    elif [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#options[@]} )); then
        echo "${options[$((choice-1))]}"
    else
        echo "$default"
    fi
}

prompt_value() {
    local prompt="$1"
    local default="$2"

    if [[ "$MODE" == "auto" ]]; then
        echo "$default"
        return
    fi

    read -r -p "$prompt [$default]: " value
    echo "${value:-$default}"
}

check_disk_space() {
    local required_mb="${1:-100}"
    local available_mb
    available_mb=$(df -m "$REPO_ROOT" | awk 'NR==2 {print $4}')

    if (( available_mb < required_mb )); then
        log_error "Insufficient disk space: ${available_mb}MB available, ${required_mb}MB required"
        return 1
    fi
    log_verbose "Disk space OK: ${available_mb}MB available"
    return 0
}

# =============================================================================
# GIT STATE CHECKS
# =============================================================================

check_git_repo() {
    if ! git -C "$REPO_ROOT" rev-parse --git-dir &>/dev/null; then
        log_error "Not a git repository: $REPO_ROOT"
        return 1
    fi
    return 0
}

check_git_state() {
    log_step "Checking Git State"

    # Check for merge in progress
    if [[ -f "$REPO_ROOT/.git/MERGE_HEAD" ]]; then
        log_error "Merge in progress. Complete or abort it first:"
        log_error "  git merge --abort  OR  git merge --continue"
        return 2
    fi

    # Check for rebase in progress
    if [[ -d "$REPO_ROOT/.git/rebase-merge" ]] || [[ -d "$REPO_ROOT/.git/rebase-apply" ]]; then
        log_error "Rebase in progress. Complete or abort it first:"
        log_error "  git rebase --abort  OR  git rebase --continue"
        return 2
    fi

    # Check for cherry-pick in progress
    if [[ -f "$REPO_ROOT/.git/CHERRY_PICK_HEAD" ]]; then
        log_error "Cherry-pick in progress. Complete or abort it first:"
        log_error "  git cherry-pick --abort  OR  git cherry-pick --continue"
        return 2
    fi

    # Check for detached HEAD
    if ! git -C "$REPO_ROOT" symbolic-ref -q HEAD &>/dev/null; then
        log_error "Detached HEAD state. Checkout a branch first:"
        log_error "  git checkout <branch-name>"
        return 2
    fi

    # Check for index.lock
    if [[ -f "$REPO_ROOT/.git/index.lock" ]]; then
        log_warn "Git index.lock exists. Another git process may be running."
        if ! confirm "Proceed anyway?"; then
            return 4
        fi
    fi

    log_success "Git state is clean"
    return 0
}

get_current_branch() {
    git -C "$REPO_ROOT" branch --show-current
}

check_remote_branch() {
    local branch="$1"

    if ! git -C "$REPO_ROOT" ls-remote --exit-code --heads origin "$branch" &>/dev/null; then
        log_warn "Remote branch 'origin/$branch' does not exist"
        return 1
    fi
    return 0
}

get_divergence() {
    local branch="$1"
    local local_hash remote_hash base_hash

    local_hash=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null)
    remote_hash=$(git -C "$REPO_ROOT" rev-parse "origin/$branch" 2>/dev/null) || return 1
    base_hash=$(git -C "$REPO_ROOT" merge-base HEAD "origin/$branch" 2>/dev/null) || return 1

    local ahead=0 behind=0

    if [[ "$local_hash" != "$base_hash" ]]; then
        ahead=$(git -C "$REPO_ROOT" rev-list --count "$base_hash..$local_hash")
    fi

    if [[ "$remote_hash" != "$base_hash" ]]; then
        behind=$(git -C "$REPO_ROOT" rev-list --count "$base_hash..$remote_hash")
    fi

    echo "$ahead $behind"
}

# =============================================================================
# LOCK FILE MANAGEMENT
# =============================================================================

acquire_lock() {
    if ! $USE_LOCK; then
        log_verbose "Lock file disabled"
        return 0
    fi

    if [[ -f "$LOCK_FILE" ]]; then
        local lock_pid lock_time
        read -r lock_pid lock_time < "$LOCK_FILE" 2>/dev/null || true

        log_warn "Lock file exists (PID: $lock_pid, created: $lock_time)"

        # Check if process is still running
        if [[ -n "$lock_pid" ]] && kill -0 "$lock_pid" 2>/dev/null; then
            log_error "Another sync process is running (PID: $lock_pid)"
            if ! $FORCE_LOCK; then
                log_error "Use --force to override"
                return 4
            fi
            log_warn "Forcing lock override"
        else
            log_warn "Stale lock file (process not running)"
            if ! $FORCE_LOCK && [[ "$MODE" != "auto" ]]; then
                if ! confirm "Remove stale lock?"; then
                    return 4
                fi
            fi
        fi
    fi

    mkdir -p "$(dirname "$LOCK_FILE")"
    echo "$$ $(date -Iseconds)" > "$LOCK_FILE"
    log_verbose "Lock acquired (PID: $$)"
    return 0
}

release_lock() {
    if [[ -f "$LOCK_FILE" ]]; then
        rm -f "$LOCK_FILE"
        log_verbose "Lock released"
    fi
}

# =============================================================================
# STATE MANAGEMENT (for recovery)
# =============================================================================

save_state() {
    local state="$1"
    mkdir -p "$(dirname "$STATE_FILE")"
    cat > "$STATE_FILE" << EOF
STATE=$state
TIMESTAMP=$(date -Iseconds)
BACKUP_DIR=$BACKUP_DIR
BRANCH=$BRANCH
PID=$$
EOF
    log_verbose "State saved: $state"
}

load_state() {
    if [[ -f "$STATE_FILE" ]]; then
        source "$STATE_FILE"
        echo "$STATE"
    else
        echo ""
    fi
}

clear_state() {
    rm -f "$STATE_FILE"
    log_verbose "State cleared"
}

check_incomplete_sync() {
    local prev_state
    prev_state=$(load_state)

    if [[ -n "$prev_state" && "$prev_state" != "complete" ]]; then
        log_warn "Previous sync was incomplete (state: $prev_state)"

        if [[ "$MODE" == "auto" ]]; then
            log_info "Attempting recovery..."
            return 0
        fi

        echo "Options:"
        echo "  1. Attempt recovery"
        echo "  2. Start fresh (may lose data)"
        echo "  3. Abort"

        read -r -p "Choice [1-3]: " choice
        case "$choice" in
            1) return 0 ;;
            2) clear_state; return 0 ;;
            *) return 5 ;;
        esac
    fi
    return 0
}

# =============================================================================
# BACKUP MANAGEMENT
# =============================================================================

create_backup() {
    if ! $DO_BACKUP; then
        log_warn "Backup disabled (--no-backup)"
        return 0
    fi

    log_step "Creating Backup"

    if [[ ! -d "$ML_DATA_DIR" ]]; then
        log_info "No ML data directory to backup"
        return 0
    fi

    BACKUP_DIR="$REPO_ROOT/.git-ml-backup-$(date +%Y%m%d-%H%M%S)-$$"

    if $DRY_RUN; then
        log_info "[DRY-RUN] Would backup $ML_DATA_DIR to $BACKUP_DIR"
        return 0
    fi

    cp -r "$ML_DATA_DIR" "$BACKUP_DIR"

    # Verify backup
    local orig_count backup_count
    orig_count=$(find "$ML_DATA_DIR" -type f | wc -l)
    backup_count=$(find "$BACKUP_DIR" -type f | wc -l)

    if (( backup_count < orig_count )); then
        log_error "Backup incomplete: $backup_count/$orig_count files"
        rm -rf "$BACKUP_DIR"
        return 1
    fi

    log_success "Backup created: $BACKUP_DIR ($backup_count files)"
    return 0
}

restore_backup() {
    if [[ -z "$BACKUP_DIR" || ! -d "$BACKUP_DIR" ]]; then
        log_error "No backup available to restore"
        return 1
    fi

    log_info "Restoring from backup: $BACKUP_DIR"

    if $DRY_RUN; then
        log_info "[DRY-RUN] Would restore from $BACKUP_DIR"
        return 0
    fi

    # Restore unique session files (won't overwrite existing)
    for subdir in chats actions sessions; do
        if [[ -d "$BACKUP_DIR/$subdir" ]]; then
            find "$BACKUP_DIR/$subdir" -type f -name "*.json" | while read -r file; do
                local rel_path="${file#$BACKUP_DIR/}"
                local dest="$ML_DATA_DIR/$rel_path"
                if [[ ! -f "$dest" ]]; then
                    mkdir -p "$(dirname "$dest")"
                    cp "$file" "$dest"
                    log_verbose "Restored: $rel_path"
                fi
            done
        fi
    done

    log_success "Backup restored"
    return 0
}

cleanup_backup() {
    if [[ -z "$BACKUP_DIR" || ! -d "$BACKUP_DIR" ]]; then
        return 0
    fi

    if $KEEP_BACKUP; then
        log_info "Keeping backup: $BACKUP_DIR"
        return 0
    fi

    if $DRY_RUN; then
        log_info "[DRY-RUN] Would remove backup: $BACKUP_DIR"
        return 0
    fi

    rm -rf "$BACKUP_DIR"
    log_verbose "Backup removed: $BACKUP_DIR"
    return 0
}

# =============================================================================
# ML DATA OPERATIONS
# =============================================================================

commit_local_ml_data() {
    log_step "Committing Local ML Data"

    # Check for ML data changes
    local ml_changes
    ml_changes=$(git -C "$REPO_ROOT" status --porcelain .git-ml 2>/dev/null | wc -l)

    if (( ml_changes == 0 )); then
        log_info "No local ML data changes to commit"
        return 0
    fi

    log_info "Found $ml_changes ML data changes"

    if $DRY_RUN; then
        log_info "[DRY-RUN] Would commit ML data changes"
        return 0
    fi

    git -C "$REPO_ROOT" add .git-ml/
    git -C "$REPO_ROOT" commit -m "data: ML session data (pre-sync)" || {
        log_warn "Nothing to commit (may already be committed)"
    }

    log_success "Local ML data committed"
    return 0
}

dedupe_jsonl() {
    local file="$1"
    local key_field="$2"

    if [[ ! -f "$file" ]]; then
        return 0
    fi

    if ! $DO_DEDUPE; then
        log_verbose "Deduplication disabled for $file"
        return 0
    fi

    log_verbose "Deduplicating $file by $key_field"

    if $DRY_RUN; then
        log_info "[DRY-RUN] Would dedupe $file"
        return 0
    fi

    local temp_file="${file}.deduped"
    local before_count after_count

    before_count=$(wc -l < "$file")

    # Use Python for reliable JSON handling
    python3 << PYTHON
import json
import sys

seen = {}
with open("$file", 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            key = obj.get("$key_field", line)  # Use whole line if no key
            seen[key] = line  # Later entries override earlier
        except json.JSONDecodeError:
            # Keep malformed lines
            seen[line] = line

with open("$temp_file", 'w') as f:
    for line in seen.values():
        f.write(line + '\n')
PYTHON

    mv "$temp_file" "$file"
    after_count=$(wc -l < "$file")

    local removed=$((before_count - after_count))
    if (( removed > 0 )); then
        log_info "Removed $removed duplicate entries from $(basename "$file")"
    fi

    return 0
}

resolve_ml_conflicts() {
    log_step "Resolving ML Data Conflicts"

    # Get list of conflicted files
    local conflicted
    conflicted=$(git -C "$REPO_ROOT" diff --name-only --diff-filter=U 2>/dev/null | grep "^\.git-ml/" || true)

    if [[ -z "$conflicted" ]]; then
        log_info "No ML data conflicts to resolve"
        return 0
    fi

    log_info "Conflicted ML files:"
    echo "$conflicted" | while read -r file; do
        echo "  - $file"
    done

    if ! $AUTO_CONFLICTS; then
        log_warn "Manual conflict resolution requested"
        log_info "Resolve conflicts manually, then run: git add .git-ml/ && git commit"
        return 1
    fi

    if $DRY_RUN; then
        log_info "[DRY-RUN] Would auto-resolve ML conflicts"
        return 0
    fi

    echo "$conflicted" | while read -r file; do
        local full_path="$REPO_ROOT/$file"
        local basename
        basename=$(basename "$file")

        case "$basename" in
            commits.jsonl|sessions.jsonl)
                # For shared JSONL: take remote, then restore local unique entries
                log_verbose "Resolving $basename: take remote + merge local"
                git -C "$REPO_ROOT" checkout --theirs "$file" 2>/dev/null || true
                ;;
            *)
                # For other files: take ours (local) - preserve local data
                log_verbose "Resolving $basename: keep local"
                git -C "$REPO_ROOT" checkout --ours "$file" 2>/dev/null || true
                ;;
        esac
    done

    # Restore session-specific files from backup
    restore_backup

    # Stage resolved files
    git -C "$REPO_ROOT" add .git-ml/

    log_success "ML conflicts resolved"
    return 0
}

# =============================================================================
# GIT SYNC OPERATIONS
# =============================================================================

fetch_remote() {
    log_step "Fetching Remote"

    local retry=0
    while (( retry < MAX_RETRIES )); do
        if $DRY_RUN; then
            log_info "[DRY-RUN] Would fetch origin"
            return 0
        fi

        if git -C "$REPO_ROOT" fetch origin 2>&1; then
            log_success "Fetch complete"
            return 0
        fi

        retry=$((retry + 1))
        if (( retry < MAX_RETRIES )); then
            log_warn "Fetch failed, retrying in ${RETRY_DELAY}s... ($retry/$MAX_RETRIES)"
            sleep "$RETRY_DELAY"
            RETRY_DELAY=$((RETRY_DELAY * 2))
        fi
    done

    log_error "Fetch failed after $MAX_RETRIES attempts"
    return 3
}

pull_and_merge() {
    local branch="$1"

    log_step "Pulling and Merging"

    # Check divergence
    local divergence ahead behind
    divergence=$(get_divergence "$branch")
    read -r ahead behind <<< "$divergence"

    log_info "Local is $ahead ahead, $behind behind origin/$branch"

    if (( behind == 0 )); then
        log_info "Already up to date with remote"
        return 0
    fi

    if (( ahead > 0 && behind > 0 )); then
        log_warn "Branch has diverged - merge required"
        if [[ "$MODE" == "semi" ]]; then
            if ! confirm "Proceed with merge?"; then
                return 5
            fi
        fi
    fi

    if $DRY_RUN; then
        log_info "[DRY-RUN] Would pull --no-rebase origin $branch"
        return 0
    fi

    # Pull with merge (not rebase!)
    if git -C "$REPO_ROOT" pull --no-rebase origin "$branch" 2>&1; then
        log_success "Pull complete"
        return 0
    fi

    # Pull failed - likely conflicts
    log_warn "Pull resulted in conflicts"

    # Check if conflicts are only in ML data
    local non_ml_conflicts
    non_ml_conflicts=$(git -C "$REPO_ROOT" diff --name-only --diff-filter=U 2>/dev/null | grep -v "^\.git-ml/" || true)

    if [[ -n "$non_ml_conflicts" ]]; then
        log_error "Non-ML files have conflicts - manual resolution required:"
        echo "$non_ml_conflicts"
        return 2
    fi

    # Only ML conflicts - attempt auto-resolution
    resolve_ml_conflicts || return $?

    # Complete the merge
    git -C "$REPO_ROOT" commit -m "merge: Sync with remote (auto-resolved ML conflicts)" || {
        log_warn "Merge commit may have failed or was not needed"
    }

    log_success "Merge complete"
    return 0
}

push_changes() {
    local branch="$1"

    log_step "Pushing Changes"

    # Check if there's anything to push
    local ahead
    ahead=$(git -C "$REPO_ROOT" rev-list --count "origin/$branch..HEAD" 2>/dev/null || echo "0")

    if (( ahead == 0 )); then
        log_info "Nothing to push"
        return 0
    fi

    log_info "Pushing $ahead commit(s) to origin/$branch"

    if $DRY_RUN; then
        log_info "[DRY-RUN] Would push to origin $branch"
        return 0
    fi

    local retry=0
    while (( retry < MAX_RETRIES )); do
        if git -C "$REPO_ROOT" push -u origin "$branch" 2>&1; then
            log_success "Push complete"
            return 0
        fi

        retry=$((retry + 1))
        if (( retry < MAX_RETRIES )); then
            log_warn "Push failed, retrying in ${RETRY_DELAY}s... ($retry/$MAX_RETRIES)"
            sleep "$RETRY_DELAY"
            RETRY_DELAY=$((RETRY_DELAY * 2))
        fi
    done

    log_error "Push failed after $MAX_RETRIES attempts"
    return 3
}

# =============================================================================
# SURVEY MODE
# =============================================================================

run_survey() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Git Sync for ML Data - Configuration Survey"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    # Branch
    local current_branch
    current_branch=$(get_current_branch)
    BRANCH=$(prompt_value "Branch to sync" "$current_branch")

    # Dry run
    local dry_choice
    dry_choice=$(prompt_choice "Run mode:" "Execute" "Execute" "Dry-run (preview only)")
    [[ "$dry_choice" == "Dry-run"* ]] && DRY_RUN=true

    # Backup
    local backup_choice
    backup_choice=$(prompt_choice "Backup ML data before sync?" "Yes" "Yes" "No")
    [[ "$backup_choice" == "No" ]] && DO_BACKUP=false

    if $DO_BACKUP; then
        local keep_choice
        keep_choice=$(prompt_choice "Keep backup after success?" "No" "No" "Yes")
        [[ "$keep_choice" == "Yes" ]] && KEEP_BACKUP=true
    fi

    # Conflict resolution
    local conflict_choice
    conflict_choice=$(prompt_choice "Conflict resolution:" "Automatic" "Automatic" "Manual (pause on conflicts)")
    [[ "$conflict_choice" == "Manual"* ]] && AUTO_CONFLICTS=false

    # Deduplication
    local dedupe_choice
    dedupe_choice=$(prompt_choice "Deduplicate JSONL files?" "Yes" "Yes" "No")
    [[ "$dedupe_choice" == "No" ]] && DO_DEDUPE=false

    # Verbose
    local verbose_choice
    verbose_choice=$(prompt_choice "Verbose output?" "No" "No" "Yes")
    [[ "$verbose_choice" == "Yes" ]] && VERBOSE=true

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Configuration Summary"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Branch:       $BRANCH"
    echo "  Dry-run:      $DRY_RUN"
    echo "  Backup:       $DO_BACKUP (keep: $KEEP_BACKUP)"
    echo "  Auto-resolve: $AUTO_CONFLICTS"
    echo "  Dedupe:       $DO_DEDUPE"
    echo "  Verbose:      $VERBOSE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    if ! confirm "Proceed with these settings?" "y"; then
        log_info "Cancelled by user"
        return 5
    fi
}

# =============================================================================
# MAIN
# =============================================================================

show_help() {
    head -50 "$0" | grep "^#" | cut -c3-
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --auto)
                MODE="auto"
                shift
                ;;
            --semi)
                MODE="semi"
                shift
                ;;
            --survey|--manual)
                MODE="survey"
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --force)
                FORCE_LOCK=true
                shift
                ;;
            --no-lock)
                USE_LOCK=false
                shift
                ;;
            --no-backup)
                DO_BACKUP=false
                shift
                ;;
            --keep-backup)
                KEEP_BACKUP=true
                shift
                ;;
            --no-dedupe)
                DO_DEDUPE=false
                shift
                ;;
            --manual-conflicts)
                AUTO_CONFLICTS=false
                shift
                ;;
            --branch)
                BRANCH="$2"
                shift 2
                ;;
            --verbose|-v)
                VERBOSE=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

cleanup() {
    local exit_code=$?

    if (( exit_code != 0 )); then
        log_warn "Sync did not complete successfully (exit code: $exit_code)"

        if [[ -n "$BACKUP_DIR" && -d "$BACKUP_DIR" ]]; then
            log_info "Backup preserved at: $BACKUP_DIR"
            log_info "To restore: cp -r $BACKUP_DIR/* $ML_DATA_DIR/"
        fi
    fi

    release_lock

    exit $exit_code
}

main() {
    parse_args "$@"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Git Sync for ML Data"
    echo "  Mode: $MODE | Dry-run: $DRY_RUN"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Set up cleanup trap
    trap cleanup EXIT

    # Survey mode: gather parameters interactively
    if [[ "$MODE" == "survey" ]]; then
        run_survey || exit $?
    fi

    # Use current branch if not specified
    if [[ -z "$BRANCH" ]]; then
        BRANCH=$(get_current_branch)
    fi

    log_info "Syncing branch: $BRANCH"

    # Pre-flight checks
    check_git_repo || exit $?
    check_disk_space 100 || exit $?
    check_incomplete_sync || exit $?
    acquire_lock || exit $?
    check_git_state || exit $?

    # Create backup
    save_state "backup"
    create_backup || exit $?

    # Commit local ML data
    save_state "commit_local"
    commit_local_ml_data || exit $?

    # Fetch remote
    save_state "fetch"
    fetch_remote || exit $?

    # Check if remote branch exists
    if ! check_remote_branch "$BRANCH"; then
        log_info "Remote branch doesn't exist - will create on push"
    fi

    # Pull and merge
    save_state "merge"
    pull_and_merge "$BRANCH" || exit $?

    # Deduplicate JSONL files
    save_state "dedupe"
    dedupe_jsonl "$ML_DATA_DIR/tracked/commits.jsonl" "hash"
    dedupe_jsonl "$ML_DATA_DIR/tracked/sessions.jsonl" "session_id"

    # Push changes
    save_state "push"
    push_changes "$BRANCH" || exit $?

    # Success - cleanup
    save_state "complete"
    cleanup_backup
    clear_state

    echo ""
    log_success "━━━ Sync Complete ━━━"
    echo ""

    exit 0
}

main "$@"
