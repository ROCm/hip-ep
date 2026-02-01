<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
---
name: update-branch
description: Update feature branch with latest from origin/main
---

# /update-branch

Update current feature branch by rebasing onto origin/main and force-pushing to fork.

## Safety Checks

1. **Not on main**: Error if on main branch
2. **Clean tree**: Error if uncommitted changes exist
3. **PR status**: If PR is MERGED or CLOSED, auto-cleanup and STOP:
   - `git checkout main && git pull origin main && git branch -D <feature-branch> && git push fork --delete <feature-branch>`
   - Show cleanup summary (don't proceed with update)

## Steps

1. Run: `git fetch origin main && git pull --rebase && git rebase origin/main`
2. If conflicts:
   - Read conflicted files with Read tool
   - Intelligently resolve using Edit tool (accept upstream for docs, merge carefully for code)
   - Stage: `git add <files>`
   - Continue: `git rebase --continue`
5. Force push: `git push --force-with-lease fork <branch>`

## Output

Show progress with emojis:
- 🌿 Branch name
- ✅ Safety checks passed
- 📥 Fetching
- 🔄 Syncing with fork
- 🔀 Rebasing
- ⚠️ Conflicts (if any) + resolution steps
- 📤 Force pushing
- 🎉 Success

## Final Step

After successful update (or cleanup), invoke `/status` to show final state
