#!/usr/bin/env python3
#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""
PR Monitoring and Conflict Resolution Script for /resolve-ci Skill

OVERVIEW:
This script autonomously monitors a PR until it merges, handling common failures
without AI intervention. It runs in a loop with 30-second polling intervals,
exiting only when intelligent intervention is needed or the PR merges.

KEY DESIGN DECISIONS:

1. UPDATE BRANCH ON EVERY CYCLE
   - Keeps branch current with origin/main throughout monitoring
   - Prevents "branch behind" issues when other PRs merge during monitoring
   - Git rebase is a no-op if already up-to-date (very fast)
   - Relies on git, not `gh`, for conflict detection (simpler, more reliable)

2. TWO TYPES OF CONFLICTS DETECTED
   - FORK_CONFLICT: Local branch conflicts with fork/branch (rare: someone else pushed)
   - BASE_CONFLICT: Local branch conflicts with origin/main (common: other PRs merged)
   Both handled via git rebase, exit with status code for AI to resolve

3. AUTO-FIX PRE-COMMIT FAILURES
   - Pre-commit failures (formatting) are mechanical, safe to auto-fix
   - Run pre-commit, commit changes, push to fork
   - Complex CI failures (build/test) require AI analysis

4. TOKEN EFFICIENCY
   - Python script polls GitHub API (no AI token cost)
   - Only returns to AI when intelligent intervention needed
   - Status codes tell AI what happened: conflicts, CI failures, auto-merge issues

WORKFLOW:
  Loop every 30 seconds:
    1. Update branch (rebase onto origin/main) → Exit if conflicts
    2. Check if PR merged → Cleanup and exit
    3. Check CI status → Auto-fix pre-commit, exit if complex failures
    4. Enable auto-merge when CI passes
    5. Wait 30 seconds

EXIT STATUS CODES:
  - STATUS:FORK_CONFLICT - Conflict with fork branch (AI resolves)
  - STATUS:BASE_CONFLICT - Conflict with origin/main (AI resolves)
  - STATUS:NEEDS_FIX_CI - Complex CI failure (AI analyzes logs)
  - STATUS:AUTO_MERGE_FAILED - Auto-merge permission issue (user checks settings)
  - STATUS:CLEANUP_COMPLETE - PR merged, branches cleaned up (success!)
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
parser.add_argument(
    "--pr-number",
    type=int,
    default=None,
    help="PR number (optional, auto-detected from branch if not provided)",
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
    """
    Update branch: sync with fork, rebase onto origin/main, push if needed.

    Returns: (success: bool, remote_updated: bool, status: str)
    - success: True if no intervention needed, False if AI/user help required
    - remote_updated: True if remote was pushed (CI restarts), False if unchanged
    - status: "UPDATED" | "UP_TO_DATE" | "FORK_CONFLICT" | "BASE_CONFLICT" | "PUSH_FAILED"

    This handles TWO types of conflicts via git rebase:
    1. FORK_CONFLICT: Local branch diverged from fork/branch
       - Rare case: someone else pushed to the same feature branch
       - Detected by: git rebase fork/branch failure
    2. BASE_CONFLICT: Local branch conflicts with origin/main
       - Common case: other PRs merged to main while we're waiting
       - Detected by: git rebase origin/main failure

    WHY USE GIT INSTEAD OF `gh`:
    - Git rebase directly detects conflicts (simple, reliable)
    - `gh pr view --json mergeStateStatus` is indirect and complex
    - Git is the source of truth for branch relationships

    WHY RETURN remote_updated:
    - If remote pushed → CI restarts → should break loop and restart monitoring
    - If unchanged → CI still running on same commit → continue checking CI status
    """
    print("🔄 Updating branch...")
    print("")

    # Step 1: Sync with upstream fork branch
    # WHY: Ensure we have latest commits from fork (handles FORK_CONFLICT)
    print("📥 Step 1: Syncing with fork branch...")
    run_command(f'git fetch fork "{branch}"')

    result = subprocess.run(
        f'git rebase "fork/{branch}"',
        shell=True,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        # FORK_CONFLICT: Someone else pushed to our feature branch
        print("⚠️  Conflicts with upstream fork branch detected")
        print("STATUS:FORK_CONFLICT")
        # Return: (success=False: needs AI help, remote_updated=False: remote not changed, status=FORK_CONFLICT)
        return False, False, "FORK_CONFLICT"

    print("✅ Synced with fork")
    print("")

    # Capture HEAD before rebasing onto main
    # WHY: We need to know if rebase actually changed anything
    # If no changes → no need to push → CI doesn't restart
    old_head = run_command("git rev-parse HEAD")
    if not old_head:
        print("❌ Failed to get current HEAD")
        sys.exit(1)

    # Step 2: Rebase onto origin/main
    # WHY: Keep branch current with main (handles BASE_CONFLICT)
    # This is the KEY operation that prevents "branch behind" infinite loops
    print("📥 Step 2: Rebasing onto origin/main...")
    run_command("git fetch origin main")

    result = subprocess.run(
        "git rebase origin/main",
        shell=True,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        # BASE_CONFLICT: Branch conflicts with origin/main
        # Common when other PRs merge while we're waiting for CI
        print("⚠️  Conflicts with base branch detected")
        print("STATUS:BASE_CONFLICT")
        # Return: (success=False: needs AI help, remote_updated=False: remote not changed, status=BASE_CONFLICT)
        return False, False, "BASE_CONFLICT"

    print("✅ Rebased onto main")
    print("")

    # Check if HEAD moved (rebase changed anything)
    new_head = run_command("git rev-parse HEAD")
    if not new_head:
        print("❌ Failed to get current HEAD after rebase")
        sys.exit(1)

    if old_head == new_head:
        # Branch was already up-to-date, nothing changed
        # WHY NOT PUSH: No point pushing if nothing changed
        # WHY IMPORTANT: Tells monitoring loop CI is still running on same commit
        print("✅ Branch already up-to-date")
        print("")
        # Return: (success=True: no conflicts, remote_updated=False: no push needed, status=UP_TO_DATE)
        # Monitoring loop will fall through to check current CI status
        return True, False, "UP_TO_DATE"

    # HEAD moved - we have new commits after rebasing
    # Push rebased branch to fork
    # WHY --force-with-lease: After rebase, history changed (force needed)
    # But --force-with-lease is safer than --force: fails if someone else
    # pushed to fork/branch since our last fetch (prevents overwriting work)
    print("📤 Pushing updated branch...")
    result = subprocess.run(
        f'git push fork "{branch}" --force-with-lease',
        shell=True,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        # PUSH_FAILED: Corner case (network issue, auth problem, or race condition)
        # WHY return False: Need AI/user intervention to diagnose
        # WHY remote_updated=False: Remote was NOT updated despite local changes
        print("❌ Failed to push")
        print(result.stderr)
        print("STATUS:PUSH_FAILED")
        # Return: (success=False: needs AI/user help, remote_updated=False: push failed, status=PUSH_FAILED)
        # Monitoring loop will exit and return to AI
        return False, False, "PUSH_FAILED"

    print("✅ Branch updated successfully")
    print("")
    # Return: (success=True: no conflicts, remote_updated=True: pushed to fork, status=UPDATED)
    # WHY remote_updated=True: We pushed, CI will restart on new commit
    # Monitoring loop will continue (restart) to wait for CI to start on new commit
    return True, True, "UPDATED"


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
        f'gh pr list --head "{branch}" --json number,state,isDraft,title', check=False
    )
    if not output:
        return None
    try:
        prs = json.loads(output)
        return prs[0] if prs else None
    except (json.JSONDecodeError, IndexError):
        return None


def extract_issue_number(pr_title):
    """Extract issue number from PR title like 'Issue #038: ...'"""
    import re

    match = re.match(r"Issue\s+#(\d+)", pr_title)
    return match.group(1) if match else None


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


def check_if_merged(pr_number):
    """Check if PR is merged using GitHub API."""
    output = run_command(f"gh pr view {pr_number} --json state,mergedAt", check=False)
    if not output:
        return False
    try:
        data = json.loads(output)
        return data.get("state") == "MERGED"
    except json.JSONDecodeError:
        return False


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
    # Fix Python output buffering for background execution
    # Without this, print() output doesn't appear in output file until script exits
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

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
    pr_title = pr_info.get("title", "")

    # Validate PR is open
    if pr_state != "OPEN":
        print(f"ℹ️  PR #{pr_number} is {pr_state} (not OPEN)")
        sys.exit(0)

    # Check 1: Validate PR is not draft
    if is_draft:
        print(f"❌ PR #{pr_number} is still DRAFT")
        print(f"Mark it ready for review first: gh pr ready {pr_number}")
        print("STATUS:NEEDS_READY")
        sys.exit(1)

    # Check 2: Validate PR is finalized (title updated from [WIP])
    if "[WIP]" in pr_title:
        issue_num = extract_issue_number(pr_title)
        print(f"❌ PR #{pr_number} needs finalization")
        print(f"Title still contains [WIP]: {pr_title}")
        print("")
        print("AI must complete Phase 0 finalization:")
        print(f"1. Read issue #{issue_num} file for context")
        print("2. Craft PR title (remove [WIP], add proper type)")
        print("3. Write comprehensive PR body from issue + implementation")
        print("4. Update completed-issues.md, backlog.md")
        print("5. Delete issue/plan files")
        print(f'6. Commit "docs: complete issue #{issue_num}" and push')
        print("7. Re-run monitor-pr.py")
        print("")
        print("STATUS:NEEDS_FINALIZATION")
        print(f"PR_NUMBER:{pr_number}")
        print(f"ISSUE_NUMBER:{issue_num}")
        sys.exit(1)

    print(f"✅ Found PR #{pr_number} (OPEN, ready for review)")
    print("")

    # Validate if --pr-number was explicitly provided
    if args.pr_number and args.pr_number != pr_number:
        print(
            f"⚠️  Warning: Provided PR #{args.pr_number} doesn't match detected PR #{pr_number}"
        )
        print(f"Using auto-detected PR #{pr_number} from branch {current_branch}")
        print("")

    # Start monitoring loop
    if args.attempt > 1:
        print(f"🔍 Starting monitoring loop (Attempt {args.attempt}/6)...")
    else:
        print("🔍 Starting monitoring loop...")
    print("")

    cycle_count = 0

    while True:
        cycle_count += 1
        print(f"[Cycle {cycle_count}] Checking status...")

        # CRITICAL: Update branch on every cycle
        # WHY: When another PR merges to main during monitoring, our branch falls
        # behind and GitHub won't auto-merge. By rebasing on every cycle, we stay
        # current. Git rebase is a no-op if already up-to-date (very fast).
        # This prevents the infinite loop bug where we wait forever for a merge
        # that will never happen because branch is behind.
        success, remote_updated, _ = update_branch(current_branch)

        if not success:
            # Conflict or push failed - needs AI/user intervention
            # STATUS already printed by update_branch (FORK_CONFLICT, BASE_CONFLICT, or PUSH_FAILED)
            # WHY exit: Can't proceed until conflicts resolved or push issue fixed
            sys.exit(0)

        if remote_updated:
            # Remote branch was pushed - CI will restart on new commit
            # WHY continue: Need to restart monitoring from top, give CI time to start
            # Don't check current CI status - it's for old commit, will be stale
            print("⏳ Waiting 30 seconds for CI to start on updated branch...")
            print("")
            time.sleep(30)
            continue

        # Branch was already up-to-date (UP_TO_DATE status)
        # WHY fall through: CI is still running on current commit
        # Check CI status in this same iteration

        # Check if merged using GitHub API
        if check_if_merged(pr_number):
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

            # Auto-fix pre-commit failures (mechanical, safe)
            # WHY: Pre-commit failures are just formatting issues. We can safely
            # run the formatter, commit, and push without AI intervention.
            # Complex failures (build errors, test failures) need AI analysis.
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
            # Return to AI for log analysis and fix
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
            # All checks passed - enable auto-merge if not already enabled
            # WHY AUTO-MERGE: Once CI passes and branch is up-to-date, GitHub can
            # auto-merge when maintainer approves. This eliminates manual clicking.
            auto_merge_enabled = pr_data.get("autoMergeRequest") is not None

            if not auto_merge_enabled:
                if enable_auto_merge(pr_number):
                    # Auto-merge enabled, continue monitoring for merge
                    pass
                else:
                    # Auto-merge failed (likely permissions issue)
                    # Return to AI so user can check repository settings
                    print("STATUS:AUTO_MERGE_FAILED")
                    sys.exit(0)
            else:
                print("✅ All checks passed - auto-merge enabled, waiting for merge...")

        # Wait 30 seconds before next check
        # WHY 30 SECONDS: Balance between responsiveness and API rate limits
        # GitHub has rate limits, and CI checks typically take minutes, not seconds
        print("⏳ Waiting 30 seconds before next check...")
        print("")
        time.sleep(30)


if __name__ == "__main__":
    main()
