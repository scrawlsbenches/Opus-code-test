"""
Git hooks module for ML Data Collector

Handles installation of git hooks for automatic data collection.
"""

from pathlib import Path


# Git hook configuration
ML_HOOK_MARKER = "# ML-DATA-COLLECTOR-HOOK"

POST_COMMIT_SNIPPET = '''
# ML-DATA-COLLECTOR-HOOK
# ML Data Collection - Post-Commit Hook
# Automatically collects enriched commit data for model training
python scripts/ml_data_collector.py commit 2>/dev/null || true
# END-ML-DATA-COLLECTOR-HOOK
'''

PRE_PUSH_SNIPPET = '''
# ML-DATA-COLLECTOR-HOOK
# ML Data Collection - Pre-Push Hook
# Validates data collection is working before push
if [ -d ".git-ml/commits" ]; then
    count=$(ls -1 .git-ml/commits/*.json 2>/dev/null | wc -l)
    echo "📊 ML Data: $count commits collected"
fi
# END-ML-DATA-COLLECTOR-HOOK
'''


def install_hooks():
    """Install git hooks for data collection, merging with existing hooks."""
    hooks_dir = Path(".git/hooks")

    for hook_name, snippet in [("post-commit", POST_COMMIT_SNIPPET), ("pre-push", PRE_PUSH_SNIPPET)]:
        hook_path = hooks_dir / hook_name

        if hook_path.exists():
            existing = hook_path.read_text(encoding="utf-8")

            # Check if our hook is already installed
            if ML_HOOK_MARKER in existing:
                print(f"✓ {hook_name}: ML hook already installed")
                continue

            # Append to existing hook
            with open(hook_path, "a", encoding="utf-8") as f:
                f.write(snippet)
            print(f"✓ {hook_name}: Added ML hook to existing hook")

        else:
            # Create new hook with shebang
            with open(hook_path, "w", encoding="utf-8") as f:
                f.write("#!/bin/bash\n")
                f.write(snippet)
                f.write("\nexit 0\n")
            hook_path.chmod(0o755)
            print(f"✓ {hook_name}: Created new hook")

    print("\nML hooks installed! Commit data will be collected automatically.")
