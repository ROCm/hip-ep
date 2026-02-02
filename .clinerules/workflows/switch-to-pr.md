<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the Apache License, Version 2.0.
-->
# Switch to Pull Request Branch

Switch to a PR branch, rebase on latest main, and sync with remote.

## Workflow Steps

### Step 1: Check Current Branch

First, identify what branch you're currently on:

```bash
git branch --show-current
```

- If already on the target PR branch → skip to Step 4 (fetch and rebase)
- If on a different branch → continue to Step 2

### Step 2: Check for Uncommitted Changes

```bash
git status --short
```

If there are uncommitted changes (output is not empty), stash them:

```bash
git stash push -m "WIP: stashed before switching to PR"
```

### Step 3: Fetch Latest and Checkout PR Branch

Fetch all remote updates and checkout the PR:

```bash
git fetch origin
gh pr checkout <PR_NUMBER>
```

The `gh pr checkout` command fetches the PR branch and switches to it in one step.

### Step 4: Rebase on Latest Main

Update the PR branch with the latest changes from main:

```bash
git fetch origin main
git rebase origin/main
```

If conflicts occur:
1. Resolve conflicts in each file
2. Stage resolved files: `git add <file>`
3. Continue rebase: `git rebase --continue`
4. Or abort if needed: `git rebase --abort`

### Step 5: Push to Remote (Sync)

After rebasing, push the updated branch to remote:

```bash
git push --force-with-lease
```

Use `--force-with-lease` instead of `--force` for safety - it will fail if someone else pushed changes you don't have.

## Quick Command (PowerShell)

For switching to a PR and syncing in one operation:

```powershell
git stash push -m "WIP: auto-stash"; git fetch origin; gh pr checkout <PR_NUMBER>; git rebase origin/main; git push --force-with-lease
```

## Check PR Status

View the current branch's PR status:

```bash
gh pr view --json state,merged,title
```

View a specific PR:

```bash
gh pr view <PR_NUMBER> --json state,merged,mergedAt
```

List your open PRs:

```bash
gh pr list --author @me --state open
```

## Clean Up After Merge

After a PR is merged, delete the local and remote branches:

```bash
# Switch to main first
git checkout main

# Delete local branch
git branch -d <branch-name>

# Delete remote branch (if not auto-deleted)
git push origin --delete <branch-name>

# Prune stale remote-tracking refs
git fetch --prune
```

## When to Use

- When you need to continue work on an existing PR
- When reviewing and testing someone else's PR locally
- When updating a PR with the latest main branch changes
- When syncing local PR changes with remote

## Notes

- Always check your current branch before switching
- Stash uncommitted changes to avoid losing work
- Rebasing rewrites history - use `--force-with-lease` when pushing
- Enable "Automatically delete head branches" in GitHub repo settings to auto-clean merged branches
