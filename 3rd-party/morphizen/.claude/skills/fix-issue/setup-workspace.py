#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""
Workspace setup script for /fix-issue skill

WHAT IT DOES:
Creates a complete workspace for implementing a backlog issue - automates all
mechanical setup steps so developer can immediately start implementation work.

WORKFLOW:
1. Validates prerequisites (on main branch, no uncommitted changes)
2. Syncs with origin/main
3. Checks if someone else is already working on this issue (#063 fix)
4. Creates feature branch (feature/issue-NNN-slug)
5. Makes initial commit (modifies issue file to have something to commit)
6. Pushes to fork with upstream tracking
7. Creates draft PR
8. Returns structured output for /fix-issue skill to continue

INTEGRATION:
- Called by /fix-issue skill at Phase 3 (after user confirms to proceed)
- NOT meant for direct user invocation (use /fix-issue skill instead)
- Output is parsed by skill to continue with implementation phase

WHY THIS DESIGN:
- Enforces git workflow requirements (documented in docs/workflows/)
- Automates mechanical steps (branch, commit, PR creation)
- Prevents duplicate work (checks if branch exists on fork)
- Ensures compliance (pre-commit hooks enforced)
- Leaves PR as draft for author review before finalization

See docs/workflows/issue-resolution-workflow.md for full context.
"""

import sys
import subprocess
import json
import glob
from datetime import date
from pathlib import Path

# Ensure UTF-8 encoding on Windows
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def run(cmd, capture=False, check=True):
    """Run a shell command and optionally capture output."""
    result = subprocess.run(cmd, shell=True, capture_output=capture, text=True, check=check)
    if capture:
        return result.stdout.strip()
    return result.returncode == 0


def check_remote_branch_exists(branch_name):
    """
    Check if branch exists on fork remote.

    Added in #063 to prevent duplicate work - detects if someone else
    is already working on the same issue. Fails fast with clear message
    instead of confusing push error later.

    Returns:
        bool: True if branch exists on fork, False otherwise
    """
    try:
        result = subprocess.run(
            ["git", "ls-remote", "--heads", "fork", f"refs/heads/{branch_name}"],
            capture_output=True,
            text=True,
            check=False,
        )
        return bool(result.stdout.strip())
    except Exception as e:
        print(f"⚠️  Warning: Could not check remote branch: {e}")
        return False


def main():
    # ============================================================================
    # Phase 1: Prerequisites Check
    # ============================================================================

    print("🔍 Checking prerequisites...")

    # Check current branch
    current_branch = run("git branch --show-current", capture=True)
    if current_branch != "main":
        print(f"❌ Not on main branch (currently on: {current_branch})")
        print("   Run: git checkout main && git pull origin main")
        sys.exit(1)

    # Check uncommitted changes
    status = run("git status --porcelain", capture=True)
    if status:
        print("❌ Uncommitted changes detected:")
        run("git status --short")
        print("   Commit or stash changes first")
        sys.exit(1)

    # Sync main branch
    print("📥 Syncing main branch...")
    run("git pull origin main")

    # ============================================================================
    # Phase 2: Workspace Setup
    # ============================================================================

    # Parse issue number from argument
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <issue_number>")
        print(f"Example: {sys.argv[0]} 042")
        sys.exit(1)

    issue_num = sys.argv[1]

    # Find issue file
    pattern = f"docs/project/issues/{issue_num}-*.md"
    matches = glob.glob(pattern)
    if not matches:
        print(f"❌ Issue file not found: {pattern}")
        sys.exit(1)

    issue_file = matches[0]

    # Extract slug from filename
    issue_slug = Path(issue_file).stem.replace(f"{issue_num}-", "", 1)

    print(f"📝 Issue #{issue_num}: {issue_slug}")

    # Create feature branch
    branch_name = f"feature/issue-{issue_num}-{issue_slug}"

    # Check if branch already exists on fork
    if check_remote_branch_exists(branch_name):
        print(f"\n❌ Branch '{branch_name}' already exists on fork.")
        print(f"   Someone else may be working on issue #{issue_num}.")
        print("\n   To check the existing PR:")
        print(f"   gh pr list --head {branch_name}")
        print("\n   To view all PRs for this issue:")
        print(f"   gh pr list --search 'issue #{issue_num}' --state all")
        sys.exit(1)

    print(f"🌿 Creating branch: {branch_name}")
    run(f"git checkout -b {branch_name}")

    # Modify issue file to create initial commit
    # WHY: Need *something* to commit so we can:
    #   1. Create the feature branch with at least one commit
    #   2. Push to fork and establish upstream tracking
    #   3. Create draft PR (GitHub requires at least one commit)
    # NOTE: This is just a throwaway modification - issue file gets deleted
    #       when work completes anyway, so no need to check if "Started:" exists
    today = date.today().strftime("%Y-%m-%d")
    with open(issue_file, "a", encoding="utf-8") as f:
        f.write(f"\nStarted: {today}\n")
    print(f"✅ Added Started date: {today}")

    # Stage the issue file
    run(f'git add "{issue_file}"')

    # Pre-commit retry loop
    # Why loop: Formatters (trailing whitespace, line endings) may modify files
    # Retry until hooks pass without changes, then commit
    # Documented requirement: docs/workflows/issue-resolution-workflow.md:21
    print("🔧 Running pre-commit hooks...")
    while True:
        if run("pre-commit run", check=False):
            break
        print("⚠️  Pre-commit made changes - re-staging...")
        run("git add -u")

    # Create initial commit
    commit_msg = f"docs: start work on issue #{issue_num}"
    run(f'git commit -m "{commit_msg}"')
    print(f"✅ Committed: {commit_msg}")

    # Push with -u flag
    print("📤 Pushing to fork...")
    try:
        run(f'git push -u fork "{branch_name}"')
    except subprocess.CalledProcessError as e:
        if "rejected" in str(e):
            print("\n❌ Push rejected - branch exists on fork despite check")
            print("   This shouldn't happen. The branch may have been created between checks.")
            print(f"   Check: gh pr list --head {branch_name}")
        raise

    # Create draft PR
    print("📋 Creating draft PR...")
    pr_title = f"Issue #{issue_num}: [WIP] {issue_slug}"
    pr_body = f"Work in progress - implementing issue #{issue_num}"

    run(f'gh pr create --draft --title "{pr_title}" --body "{pr_body}"')

    # Get PR number
    pr_json = run(f'gh pr list --head "{branch_name}" --json number', capture=True)
    try:
        data = json.loads(pr_json)
        pr_number = str(data[0]["number"]) if data else ""
    except (json.JSONDecodeError, IndexError, KeyError) as e:
        print(f"❌ Failed to parse PR number: {e}")
        sys.exit(1)

    if not pr_number:
        print("❌ Failed to get PR number")
        sys.exit(1)

    # Output results (parsed by skill)
    print()
    print("STATUS:WORKSPACE_READY")
    print(f"BRANCH:{branch_name}")
    print(f"PR_NUMBER:{pr_number}")
    print(f"ISSUE_FILE:{issue_file}")
    print(f"ISSUE_NUM:{issue_num}")
    print(f"ISSUE_SLUG:{issue_slug}")


if __name__ == "__main__":
    main()
