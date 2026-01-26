<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the Apache License, Version 2.0.
-->
# Commit Changes and Sync Git Repository

Commit local changes and synchronize with the remote repository.

## Pre-flight Checks

### Step 1: Verify Current Branch

**CRITICAL:** Never commit directly to `main` branch.

```bash
git branch --show-current
```

If on `main`, create or switch to a feature branch first:
```bash
git checkout -b feature/<descriptive-name>
# or switch to existing branch
git checkout <branch-name>
```

### Step 2: Check Repository Status

```bash
git status
```

Review:
- Staged changes (green)
- Unstaged changes (red)
- Untracked files

## Commit Changes

### Step 3: Stage Changes

Stage specific files:
```bash
git add <file1> <file2>
```

Stage all changes (use with caution):
```bash
git add -A
```

Interactive staging (recommended for selective commits):
```bash
git add -p
```

### Step 4: Review Staged Changes

```bash
git diff --staged
```

Verify only intended changes are staged before committing.

### Step 5: Commit with Message

```bash
git commit -m "<type>: <brief description>"
```

Commit message conventions:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks
- `style:` - Code style/formatting changes

Examples:
```bash
git commit -m "feat: add MLIR transformation pass"
git commit -m "fix: resolve test model generation issue"
git commit -m "docs: update TESTING.md with environment variables"
```

For detailed commit messages:
```bash
git commit
# Opens editor for multi-line message
```

## Sync with Remote

### Step 6: Fetch Remote Changes

```bash
git fetch origin
```

Check if there are upstream changes:
```bash
git log HEAD..origin/main --oneline
```

### Step 6.5: Sync Feature Branch with Remote (if exists)

If your branch already exists on remote, sync it first to get any commits pushed from another machine or by collaborators:

```bash
git pull --rebase
```

To check if your branch has a remote tracking branch:
```bash
git branch -vv
```

If the branch doesn't have a remote yet (output shows no `[origin/branch-name]`), skip this step.

### Step 7: Sync with Main Branch

Option A - Rebase (preferred for cleaner history):
```bash
git fetch origin main
git rebase origin/main
```

Option B - Merge:
```bash
git fetch origin main
git merge origin/main
```

### Step 8: Handle Conflicts (if any)

If conflicts occur during rebase/merge:

1. Check conflicting files:
   ```bash
   git status
   ```

2. Resolve conflicts in each file (look for `<<<<<<<`, `=======`, `>>>>>>>` markers)

3. After resolving, stage the fixed files:
   ```bash
   git add <resolved-file>
   ```

4. Continue rebase/merge:
   ```bash
   git rebase --continue
   # or
   git merge --continue
   ```

5. If needed, abort and start over:
   ```bash
   git rebase --abort
   # or
   git merge --abort
   ```

### Step 9: Push Changes

Push to remote (feature branch):
```bash
git push origin <branch-name>
```

If branch doesn't exist on remote:
```bash
git push -u origin <branch-name>
```

If rebase changed history (force push with lease for safety):
```bash
git push --force-with-lease origin <branch-name>
```

### Step 10: Create or Update Pull Request

If no PR exists for this branch, create one. **On Windows, use `--body-file` to avoid quoting issues:**

1. Create a temporary file `pr_body.md` with the PR description:
```markdown
# Summary of Changes

* <High-level description of what this request adds/changes/improves/fixes>

Closes #<issue-number> (if applicable)

# Motivation

<Why this change is needed/useful>

# Implementation

<Any implementation details or design choices worth noting>
```

2. Create the PR using the file:
```bash
gh pr create --title "<type>: <brief description>" --body-file pr_body.md
```

3. Delete the temporary file after PR creation.

To update an existing PR description:
```bash
gh pr edit <PR-number> --body-file pr_body.md
```

If a PR already exists, the push will automatically update it (no need to edit unless scope changed).

## Quick Workflow Summary

For a typical commit and sync cycle:

```bash
# 1. Check branch (must NOT be main)
git branch --show-current

# 2. Stage changes
git add -A

# 3. Commit
git commit -m "feat: your change description"

# 4. Sync feature branch with remote (if exists)
git pull --rebase

# 5. Sync with main
git fetch origin main
git rebase origin/main

# 6. Push
git push origin <branch-name>

# 7. Create PR (if not exists) - use --body-file on Windows to avoid quoting issues
# First write PR body to pr_body.md, then:
gh pr create --title "feat: your change description" --body-file pr_body.md
```

## After Pushing

If PR already exists:
- Update PR description if scope changed significantly (see git-rules.md)
- Request review if ready: `gh pr ready`

## Notes

- **Small, frequent commits** are preferred over large, infrequent ones
- Always verify you're on a feature branch before committing
- Use `--force-with-lease` instead of `--force` for safer force pushes
- Run tests before pushing when possible
