<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
---
name: update-branch
description: Update current feature branch with latest changes from upstream main. Use when user says "update-branch", "/update-branch", or wants to sync with main.
---

# /update-branch - Update Feature Branch with Upstream Main

## Purpose

Safely update the current feature branch with the latest changes from upstream `origin/main`. Performs fetch, rebase, and force-push-with-lease to fork.

## Pre-flight Safety Checks

Before any operations, verify:

1. **Not on main branch**:
   ```bash
   git branch --show-current
   ```
   If on `main`, error and stop. This skill is only for feature branches.

2. **Clean working directory**:
   ```bash
   git status --porcelain
   ```
   If uncommitted changes exist, error and stop. User must commit or stash first.

3. **Confirm with user**:
   Display current branch name and ask user to confirm they want to update it.

## PR Status Check

Before updating, check if a Pull Request exists for the current branch and its state.

```bash
gh pr view --json state,number
```

### If PR is CLOSED or MERGED:

**DO NOT update the branch.** Instead, run the cleanup workflow per `.clinerules/git-rules.md`:

1. **Switch to main**:
   ```bash
   git checkout main
   ```

2. **Pull latest from origin/main**:
   ```bash
   git pull origin main
   ```

3. **Delete local feature branch**:
   ```bash
   git branch -D <feature-branch>
   ```

4. **Delete remote branch on fork**:
   ```bash
   git push fork --delete <feature-branch>
   ```

5. **Verify clean state**:
   ```bash
   git status
   ```
   Should show "nothing to commit, working tree clean"

6. **Inform user**:
   ```
   PR #<number> is <MERGED|CLOSED>.
   Cleaned up feature branch per git-rules.md:
   ✓ Switched to main
   ✓ Updated with latest from origin/main
   ✓ Deleted local branch: <feature-branch>
   ✓ Deleted remote branch: fork/<feature-branch>
   ✓ Working directory clean

   You're now on main with latest changes.
   ```

7. **STOP** - Do not proceed with update sequence

### If PR is OPEN or DRAFT:

Continue with the update sequence below.

### If No PR Exists:

Warn user but allow update to proceed (they may be working locally before creating PR).

```
WARNING: No PR found for branch <feature-branch>.
Consider creating a draft PR after updating: gh pr create --draft

Proceeding with branch update...
```

## Update Sequence

### 1. Fetch Latest from Origin
```bash
git fetch origin
```
Fetch latest commits from upstream without modifying local branches.

### 2. Sync with Fork's Remote Branch
```bash
git pull --rebase
```
Pull and rebase any changes that exist on fork's remote version of current branch.
This ensures we have the latest state from the fork before rebasing onto origin/main.

### 3. Rebase onto Origin/Main
```bash
git rebase origin/main
```
Rebase current branch's commits on top of the latest origin/main.

### 4. Handle Conflicts (if any)

If rebase conflicts occur:

**A. Show conflict details:**
```bash
git status
```
Display which files have conflicts.

**B. Read conflict files:**
Use Read tool to examine files with conflict markers:
```
<<<<<<< HEAD
current branch changes
=======
origin/main changes
>>>>>>> origin/main
```

**C. Attempt intelligent resolution:**

- **Documentation files** (.md, .txt, README): Consider accepting upstream version
- **Configuration files** (.gitignore, package.json): Attempt merge if safe
- **Code files** (.c, .cpp, .h, .py, .js, etc.): Provide detailed guidance to user

**D. Resolution options:**

1. **Auto-resolve if safe**: For simple conflicts in documentation
   - Use Edit tool to resolve conflicts
   - Stage resolved files: `git add <file>`
   - Continue rebase: `git rebase --continue`

2. **Guide user for complex conflicts**:
   - Show exact file paths and conflict sections
   - Explain what each side of the conflict represents
   - Provide clear steps for manual resolution
   - Offer to continue after user resolves

3. **Offer abort option**:
   ```bash
   git rebase --abort
   ```
   Returns branch to state before rebase started.

### 5. Force Push to Fork
```bash
git push --force-with-lease fork <branch-name>
```

**Why --force-with-lease?**
- Safer than `--force`
- Only succeeds if remote branch hasn't been updated by someone else
- Prevents accidentally overwriting others' work

## After Update

Display summary:
- Number of commits rebased
- Any conflicts encountered and how they were resolved
- Confirmation of successful push to fork
- Current branch status

## Example Output

```
Branch: feature/add-caching
Checks: Not on main ✓, Working tree clean ✓

Fetching from origin...
Fetched 15 new commits from upstream

Syncing with fork's remote branch...
Already up to date with fork/feature/add-caching

Rebasing onto origin/main...
Applying 3 commits:
  - feat: add caching layer
  - test: add cache tests
  - docs: update caching documentation

Rebase successful!

Force pushing to fork (--force-with-lease)...
Done!

Summary:
✓ Branch updated with latest changes from origin/main
✓ 3 commits rebased successfully
✓ Pushed to fork/feature/add-caching
```

## Error Cases

### Error: On main branch
```
ERROR: Cannot run /update-branch on main branch.
This skill is only for feature branches.
Current branch: main
```

### Error: Uncommitted changes
```
ERROR: Working directory has uncommitted changes.
Please commit or stash your changes first.

Uncommitted files:
  M src/cache.cpp
  M tests/cache_test.cpp

Run: git status
```

### Error: Rebase conflicts (complex)
```
CONFLICT: Rebase conflicts in code files require manual resolution.

Conflicted files:
  src/cache.cpp (CONFLICT - code changes)
  README.md (AUTO-RESOLVED - accepted upstream)

Next steps:
1. Review conflicts in src/cache.cpp
2. Edit the file to resolve conflict markers
3. Stage the file: git add src/cache.cpp
4. Continue: git rebase --continue

Or abort the rebase: git rebase --abort
```

## Notes

- This is a **destructive operation** (rewrites branch history)
- Only run when you're ready to update with latest upstream changes
- Requires force push, which is safe with `--force-with-lease`
- Best practice: Run this regularly to avoid large divergences
- DO NOT run on `main` branch - this skill is feature-branch only
