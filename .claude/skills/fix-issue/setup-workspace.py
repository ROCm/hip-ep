#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""
Workspace setup script for /fix-issue skill
Handles Phase 1 (prerequisites) + Phase 2 (branch, commit, PR)
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
    print(f"🌿 Creating branch: {branch_name}")
    run(f"git checkout -b {branch_name}")

    # Add "Started:" date to issue file
    today = date.today().strftime("%Y-%m-%d")
    with open(issue_file, "r", encoding="utf-8") as f:
        content = f.read()

    if "Started:" in content:
        print("⚠️  Issue already has Started date - skipping")
    else:
        # Append Started date to end of file
        with open(issue_file, "a", encoding="utf-8") as f:
            f.write(f"\nStarted: {today}\n")

        print(f"✅ Added Started date: {today}")

    # Stage the issue file
    run(f'git add "{issue_file}"')

    # Pre-commit retry loop
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
    run(f'git push -u fork "{branch_name}"')

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
