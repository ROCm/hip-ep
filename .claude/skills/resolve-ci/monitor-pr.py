#!/usr/bin/env python3
#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""
Complete PR lifecycle orchestrator for /resolve-ci skill
1. Updates branch (sync fork, rebase main) - detects conflicts
2. Monitors CI status every 30 seconds - auto-fixes pre-commit
3. Cleans up after merge - switches to main, deletes branches
Returns to AI only when intelligent intervention needed
"""

import subprocess
import json
import time
import sys
import os
import argparse

# Fix Windows console encoding for emojis
if os.name == "nt":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Monitor PR for /resolve-ci skill")
parser.add_argument(
    "--attempt", type=int, default=1, help="Retry attempt number (default: 1)"
)
args = parser.parse_args()


def run_command(cmd, check=True):
    """Run shell command and return output."""
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        return None
    return result.stdout.strip()


def get_current_branch():
    """Get current git branch."""
    return run_command("git branch --show-current")


def update_branch(branch):
    """Update branch: sync fork and rebase main. Returns (success, status)."""
    print("🔄 Phase 1: Updating branch...")
    print("")

    # Step 1: Sync with upstream fork branch
    print("📥 Step 1: Syncing with fork branch...")
    run_command(f'git fetch fork "{branch}"')

    result = subprocess.run(
        f'git rebase "fork/{branch}"',
        shell=True,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("⚠️  Conflicts with upstream fork branch detected")
        print("STATUS:FORK_CONFLICT")
        return False, "FORK_CONFLICT"

    print("✅ Synced with fork")
    print("")

    # Step 2: Rebase onto origin/main
    print("📥 Step 2: Rebasing onto origin/main...")
    run_command("git fetch origin main")

    result = subprocess.run(
        "git rebase origin/main",
        shell=True,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("⚠️  Conflicts with base branch detected")
        print("STATUS:BASE_CONFLICT")
        return False, "BASE_CONFLICT"

    print("✅ Rebased onto main")
    print("")

    # Push
    print("📤 Pushing updated branch...")
    result = subprocess.run(
        f'git push fork "{branch}" --force-with-lease',
        shell=True,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("❌ Failed to push")
        print(result.stderr)
        return False, "PUSH_FAILED"

    print("✅ Branch updated successfully")
    print("")
    return True, "UPDATED"


def cleanup_after_merge(branch):
    """Cleanup: switch to main, pull, delete branches."""
    print("")
    print("🧹 Phase 3: Cleaning up after merge...")
    print("")

    # Don't delete main
    if branch == "main":
        print("ℹ️  Already on main branch - nothing to clean up")
        return

    # Switch to main
    print("📥 Switching to main branch...")
    run_command("git checkout main")
    print("✅ Switched to main")
    print("")

    # Pull latest
    print("📥 Pulling latest from origin/main...")
    run_command("git pull origin main")
    print("✅ Main branch updated")
    print("")

    # Delete local branch
    print(f"🗑️  Deleting local branch: {branch}...")
    run_command(f'git branch -d "{branch}"')
    print("✅ Local branch deleted")
    print("")

    # Delete remote branch
    print(f"🗑️  Deleting remote branch: {branch}...")
    run_command(f'git push fork --delete "{branch}"')
    print("✅ Remote branch deleted")
    print("")

    print("🎉 Cleanup complete!")


def get_pr_info(branch):
    """Get PR info for current branch."""
    output = run_command(
        f'gh pr list --head "{branch}" --json number,state,isDraft', check=False
    )
    if not output:
        return None
    try:
        prs = json.loads(output)
        return prs[0] if prs else None
    except (json.JSONDecodeError, IndexError):
        return None


def get_pr_status(pr_number):
    """Get detailed PR status."""
    output = run_command(
        f"gh pr view {pr_number} --json statusCheckRollup,autoMergeRequest",
        check=False,
    )
    if not output:
        return None
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return None


def check_if_merged(branch):
    """Check if branch is merged using git."""
    # Fetch latest
    run_command("git fetch origin main", check=False)

    # Check if our commits are in origin/main
    result = run_command(f"git log origin/main..{branch}", check=False)
    return result == ""  # Empty means all commits are in main


def run_precommit_fix(branch):
    """Auto-fix pre-commit failures."""
    print("")
    print("🔧 Pre-commit failure detected - auto-fixing...")

    # Run pre-commit
    result = subprocess.run(
        "pre-commit run --all-files", shell=True, capture_output=True
    )

    if result.returncode == 0:
        print("✅ Pre-commit passed after auto-fix")
    else:
        print("⚠️  Pre-commit made changes - staging and committing...")

    # Stage all changes
    run_command("git add -u", check=False)

    # Check if there are changes to commit
    diff_result = subprocess.run("git diff --cached --quiet", shell=True)

    if diff_result.returncode == 0:
        print("ℹ️  No changes to commit after pre-commit run")
    else:
        # Commit and push
        run_command('git commit -m "style: apply pre-commit fixes"', check=False)
        run_command(f'git push fork "{branch}"', check=False)
        print("✅ Pre-commit fixes pushed - restarting monitoring...")


def enable_auto_merge(pr_number):
    """Enable auto-merge for PR."""
    print("")
    print("✅ All checks passed - enabling auto-merge (squash)...")

    result = subprocess.run(
        f"gh pr merge {pr_number} --auto --squash",
        shell=True,
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print("✅ Auto-merge enabled successfully")
        print("   Continuing to monitor for merge...")
        return True
    else:
        print("❌ Failed to enable auto-merge")
        return False


def main():
    # Auto-detect PR from current branch
    current_branch = get_current_branch()
    if not current_branch:
        print("❌ Failed to get current branch")
        sys.exit(1)

    print(f"🔍 Detecting PR for branch: {current_branch}")

    pr_info = get_pr_info(current_branch)
    if not pr_info:
        print(f"❌ No PR found for branch: {current_branch}")
        sys.exit(1)

    pr_number = pr_info.get("number")
    pr_state = pr_info.get("state")
    is_draft = pr_info.get("isDraft")

    # Validate PR is open
    if pr_state != "OPEN":
        print(f"ℹ️  PR #{pr_number} is {pr_state} (not OPEN)")
        sys.exit(0)

    # Validate PR is not draft
    if is_draft:
        print(f"❌ PR #{pr_number} is still DRAFT")
        print(f"Mark it ready for review first: gh pr ready {pr_number}")
        sys.exit(1)

    print(f"✅ Found PR #{pr_number} (OPEN, ready for review)")
    print("")

    # Phase 1: Update branch (upfront, before monitoring)
    success, status = update_branch(current_branch)
    if not success:
        # Conflict detected - return to AI for resolution
        sys.exit(0)

    # Phase 2: Monitor CI until merged
    if args.attempt > 1:
        print(
            f"🔍 Phase 2: Starting monitoring loop (Attempt {args.attempt}/6, 10-minute timeout)..."
        )
    else:
        print("🔍 Phase 2: Starting monitoring loop (10-minute timeout)...")
    print("")

    cycle_count = 0

    while True:
        cycle_count += 1
        print(f"[Cycle {cycle_count}] Checking status...")

        # Check if merged using git
        if check_if_merged(current_branch):
            print(f"✅ PR #{pr_number} merged successfully!")
            cleanup_after_merge(current_branch)
            print("STATUS:CLEANUP_COMPLETE")
            sys.exit(0)

        # Get PR CI status from GitHub
        pr_data = get_pr_status(pr_number)
        if not pr_data:
            print("❌ Failed to fetch PR data")
            sys.exit(1)

        # Check CI status
        status_checks = pr_data.get("statusCheckRollup", [])

        # Find failed checks (ignore checks with no conclusion - they're still running)
        failed_checks = [
            check
            for check in status_checks
            if check.get("conclusion")
            and check.get("conclusion") not in ["SUCCESS", "SKIPPED"]
        ]

        if failed_checks:
            print("❌ CI failures detected:")
            for check in failed_checks:
                print(f"  - {check.get('name')}: {check.get('conclusion')}")

            # Check if it's a pre-commit failure (auto-fixable)
            precommit_failed = any(
                check.get("name") == "pre-commit"
                and check.get("conclusion") == "FAILURE"
                for check in status_checks
            )

            if precommit_failed:
                run_precommit_fix(current_branch)
                # Continue monitoring immediately (don't wait 30s)
                continue

            # Complex CI failure - needs intelligent analysis
            print("STATUS:NEEDS_FIX_CI")
            sys.exit(0)

        # Check if CI is still running
        pending_checks = [
            check
            for check in status_checks
            if check.get("status") in ["IN_PROGRESS", "QUEUED"]
        ]

        if pending_checks:
            print("⏳ CI still running:")
            for check in pending_checks:
                print(f"  - {check.get('name')}: {check.get('status')}")
        else:
            # All checks passed, enable auto-merge if not already
            auto_merge_enabled = pr_data.get("autoMergeRequest") is not None

            if not auto_merge_enabled:
                if enable_auto_merge(pr_number):
                    # Auto-merge enabled, continue monitoring
                    pass
                else:
                    print("STATUS:AUTO_MERGE_FAILED")
                    sys.exit(0)
            else:
                print("✅ All checks passed - auto-merge enabled, waiting for merge...")

        # Wait 30 seconds before next check
        print("⏳ Waiting 30 seconds before next check...")
        print("")
        time.sleep(30)


if __name__ == "__main__":
    main()
