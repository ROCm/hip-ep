#!/usr/bin/env python3
#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Git and PR status summary"""

import subprocess
import json
import sys
import io

# Ensure UTF-8 encoding for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def run(cmd):
    """Run command and return output"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip(), result.returncode
    except subprocess.TimeoutExpired:
        return "", 1


# Branch
branch, _ = run("git branch --show-current")
print(f"🌿 Branch: {branch}")

# Status
status, _ = run("git status --short")
if not status:
    print("✨ Status: clean")
else:
    lines = status.split("\n")
    staged = sum(
        1
        for line in lines
        if line.startswith("M ") or line.startswith("A ") or line.startswith("D ")
    )
    modified = sum(1 for line in lines if line.startswith(" M"))
    untracked = sum(1 for line in lines if line.startswith("??"))
    total = len(lines)
    print(
        f"📝 Status: {total} changes (staged: {staged}, modified: {modified}, untracked: {untracked})"
    )

# Unpushed commits
unpushed, rc = run("git log @{upstream}.. --oneline 2>/dev/null | wc -l")
if rc == 0 and unpushed and int(unpushed) > 0:
    print(f"📤 Commits: {unpushed.strip()} unpushed")
else:
    _, rc = run("git rev-parse @{upstream}")
    if rc == 0:
        print("✅ Commits: up to date")
    else:
        print("⚠️  Commits: no remote tracking")

# Check if branch needs updating from origin/main
if branch != "main":
    # Fetch origin/main quietly
    run("git fetch origin main")
    commits, rc = run("git log HEAD..origin/main --oneline")
    if rc == 0:
        if commits:
            behind_count = len(commits.split("\n"))
            print(f"⬇️  Branch update: {behind_count} commits behind origin/main")
        else:
            print("✅ Branch update: up to date with origin/main")

# Show file changes compared to merge base
if branch != "main":
    merge_base, rc = run("git merge-base HEAD origin/main")
    if rc == 0 and merge_base:
        # Get file changes with stats
        numstat, rc = run(f"git diff --numstat {merge_base}..HEAD")
        name_status, _ = run(f"git diff --name-status {merge_base}..HEAD")

        if rc == 0 and numstat:
            lines = numstat.strip().split("\n")
            total_add = 0
            total_del = 0
            file_changes = []

            # Parse numstat
            name_status_dict = {}
            if name_status:
                for line in name_status.strip().split("\n"):
                    parts = line.split("\t", 1)
                    if len(parts) == 2:
                        status_code = parts[0]
                        filename = parts[1]
                        name_status_dict[filename] = status_code

            for line in lines:
                parts = line.split("\t")
                if len(parts) >= 3:
                    adds = parts[0]
                    dels = parts[1]
                    filename = parts[2]

                    # Handle binary files (shows - -)
                    if adds != "-":
                        total_add += int(adds)
                    if dels != "-":
                        total_del += int(dels)

                    status = name_status_dict.get(filename, "M")
                    status_text = {"M": "Modified", "A": "Added", "D": "Deleted"}.get(
                        status, "Modified"
                    )

                    if adds == "-" and dels == "-":
                        file_changes.append(f"  - {status_text}: {filename} (binary)")
                    else:
                        file_changes.append(
                            f"  - {status_text}: {filename} (+{adds}/-{dels})"
                        )

            print(f"Changes ({len(lines)} files):")
            for fc in file_changes:
                print(fc)
            print(f"📊 Net: +{total_add}/-{total_del}")

# PR status - handle fork-based workflow
# Search by head branch in repo (works for fork-based workflow)
pr_json, rc = run(
    f'gh pr list --repo ROCm/MorphiZen --head "{branch}" --json number,state,isDraft,url,autoMergeRequest --limit 1'
)
if rc == 0 and pr_json:
    # pr list returns array, get first element
    try:
        prs = json.loads(pr_json)
        if prs and len(prs) > 0:
            pr_json = json.dumps(prs[0])
    except json.JSONDecodeError:
        pr_json = None

if rc == 0 and pr_json:
    try:
        pr = json.loads(pr_json)
        number = pr.get("number")
        state = pr.get("state", "")
        is_draft = pr.get("isDraft", False)
        url = pr.get("url", "")
        auto_merge = pr.get("autoMergeRequest")

        if number:
            draft_text = "DRAFT" if is_draft else "READY"
            if is_draft:
                state_emoji = "📝"
            else:
                state_emoji = {"OPEN": "🔓", "CLOSED": "🔒", "MERGED": "🎉"}.get(
                    state, "📋"
                )
            print(f"{state_emoji} PR: #{number} ({state}, {draft_text})")
            if url:
                print(f"🔗 URL: {url}")

            # Show auto-merge status
            if auto_merge:
                print("🤖 Auto-merge: enabled")
            else:
                print("🚫 Auto-merge: disabled")

            # Get CI check status
            checks_json, rc = run(
                f"gh pr view {number} --repo ROCm/MorphiZen --json statusCheckRollup"
            )
            if rc == 0 and checks_json:
                try:
                    checks_data = json.loads(checks_json)
                    status_checks = checks_data.get("statusCheckRollup", [])

                    if status_checks:
                        passed = []
                        failed = []
                        pending = []

                        for check in status_checks:
                            name = check.get("name", "unknown")
                            conclusion = check.get("conclusion", "")
                            status = check.get("status", "")

                            if conclusion == "SUCCESS":
                                passed.append(name)
                            elif conclusion in ["FAILURE", "CANCELLED", "TIMED_OUT"]:
                                failed.append(name)
                            elif status in ["IN_PROGRESS", "QUEUED", "PENDING"]:
                                pending.append(name)

                        # Print summary
                        total = len(status_checks)
                        if failed:
                            print(
                                f"❌ Checks: {len(failed)} failed, {len(passed)} passed, {len(pending)} pending ({total} total)"
                            )
                            for name in failed:
                                print(f"   ❌ {name}")
                        elif pending:
                            print(
                                f"⏳ Checks: {len(pending)} pending, {len(passed)} passed ({total} total)"
                            )
                            for name in pending:
                                print(f"   ⏳ {name}")
                        else:
                            print(f"✅ Checks: All {total} checks passed")
                except json.JSONDecodeError:
                    pass

        else:
            print("❌ PR: none")
    except json.JSONDecodeError:
        print("❌ PR: none")
else:
    print("❌ PR: none")
