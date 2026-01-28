<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
---
name: status
description: Quick health check of git and PR status. Use when user says "status", "/status", or asks about current branch/PR state.
---

# /status - Git and PR Health Check

## Purpose

Provide a quick, clean summary of:
- Current git branch
- Working directory status (uncommitted/staged changes)
- Pull request status for current branch

## Instructions

When invoked, execute these checks and display results:

### 1. Current Branch
```bash
git branch --show-current
```

### 2. Git Status Summary
```bash
git status --short
```

Count and categorize:
- Uncommitted changes (modified, deleted)
- Staged changes (ready to commit)
- Untracked files

### 3. Unpushed Commits
```bash
git log @{upstream}.. --oneline 2>/dev/null || echo "No tracking branch"
```

Count commits ahead of remote.

### 4. PR Status
```bash
gh pr status
```

Check if current branch has:
- Open PR (draft or ready for review)
- Merged PR
- No PR yet

### 5. Display Clean Summary

Format output as:

```
Branch: <branch-name>
Status: <clean | N changes (M staged, K untracked)>
Commits: <N unpushed | up to date | no remote tracking>
PR: <#123 (DRAFT) | #123 (OPEN) | #123 (MERGED) | none>
URL: <PR URL if exists>
```

## Example Output

```
Branch: feature/add-caching
Status: 3 changes (1 staged, 1 untracked)
Commits: 2 unpushed
PR: #42 (DRAFT)
URL: https://github.com/ROCm/MorphiZen/pull/42
```

Or for clean state:

```
Branch: feature/refactor-api
Status: clean
Commits: up to date
PR: #38 (OPEN)
URL: https://github.com/ROCm/MorphiZen/pull/38
```

## Notes

- This is a **read-only** check - makes no modifications
- Safe to run anytime
- Helps quickly assess current state without running multiple commands
- If `gh` command fails, note that PR status is unavailable
