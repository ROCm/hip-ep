<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Git Workflow - Supplemental Guidance

See `CLAUDE.md` for main git workflow. This doc contains additional guidance.

---

## Resuming Work: Check Context First

Before starting work (especially when resuming from previous sessions):

1. **Check current branch:**
   ```bash
   git branch --show-current
   ```

2. **Look for existing PR branches:**
   ```bash
   git branch -a | grep feature/
   gh pr list --repo ROCm/MorphiZen
   ```

3. **When uncertain, ask user:**
   - If multiple branches exist or context is unclear
   - Prevents creating duplicate branches for the same task

---

## Keeping PR Updated with Main

When origin/main advances while PR is in review:

```bash
# Check what changed
git fetch origin main
git log HEAD..origin/main --oneline

# Update branch
git rebase origin/main

# Resolve conflicts if needed (see CLAUDE.md for conflict resolution)

# Force push
git push --force-with-lease fork <branch-name>
```

**When to update:**
- GitHub shows branch is out-of-date
- Before requesting final review if main has advanced
- If CI checks fail due to main branch changes

---

## PR Description Updates

Update PR description/title **only when scope/approach changes meaningfully** - not for every push.

**Update when:**
- Scope changes (covers different functionality)
- Significant refactoring (approach changed)
- Ready for review (Draft → Ready state)

**Don't update for:**
- Small bug fixes within same scope
- Code style/formatting changes
- Minor refactoring that doesn't change intent
