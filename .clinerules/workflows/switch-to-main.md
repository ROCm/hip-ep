<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Switch to Main Branch and Sync

Switch to the main branch and update it with the latest remote changes.

## Workflow Steps

### Step 1: Stash Local Changes (if any)

If the working directory is dirty (has uncommitted changes), stash them first:

```bash
git stash push -m "WIP: stashed before switching to main"
```

### Step 2: Switch to Main Branch

```bash
git checkout main
```

### Step 3: Sync with Remote

Fetch the latest changes and reset to match origin/main exactly:

```bash
git fetch origin main
git reset --hard origin/main
```

This ensures your local main branch and working directory are in perfect sync with the remote, discarding any local-only commits on main.

## Quick Command

For stashing, switching, and syncing in one operation (PowerShell):

```powershell
git stash push -m "WIP: auto-stash"; git checkout main; git fetch origin main; git reset --hard origin/main
```

## When to Use

- Before starting a new feature branch (to branch from latest main)
- When you need to review the current state of main
- After completing work on a feature branch and merging via PR

## Restoring Stashed Changes

After returning to your feature branch, restore stashed changes:

```bash
git checkout <your-branch>
git stash pop
```

To list all stashes:

```bash
git stash list
```

## Notes

- This workflow is for updating your local main branch only
- Do not make commits directly on main - use feature branches for changes
- Always stash uncommitted changes before switching branches to avoid conflicts
- Use descriptive stash messages to identify changes later
