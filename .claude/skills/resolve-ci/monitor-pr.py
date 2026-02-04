#!/usr/bin/env python3
#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""
PR monitoring script for /resolve-ci skill
Monitors PR status every 30 seconds and auto-fixes common issues
Only returns to skill when complex issues need intelligent analysis
"""

import subprocess
import json
import time
import sys
import os

# Fix Windows console encoding for emojis
if os.name == "nt":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


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
        f"gh pr view {pr_number} --json mergeable,statusCheckRollup,autoMergeRequest,mergedAt",
        check=False,
    )
    if not output:
        return None
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return None


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
        print("   Continuing to monitor for cascade conflicts...")
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
    print("🔍 Starting monitoring loop...")
    print("")

    cycle_count = 0

    while True:
        cycle_count += 1
        print(f"[Cycle {cycle_count}] Checking PR status...")

        # Get PR status
        pr_data = get_pr_status(pr_number)
        if not pr_data:
            print("❌ Failed to fetch PR data")
            sys.exit(1)

        # Check if merged
        if pr_data.get("mergedAt"):
            print(f"✅ PR #{pr_number} merged successfully!")
            print("STATUS:MERGED")
            sys.exit(0)

        # Check mergeable status
        mergeable = pr_data.get("mergeable")
        if mergeable == "CONFLICTING":
            print("⚠️  Merge conflicts detected")
            print("STATUS:NEEDS_FIX_CONFLICTS")
            sys.exit(0)

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
