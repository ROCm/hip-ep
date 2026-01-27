<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Check if Pull Request is Merged and Cleanup

Check if your current branch's PR has been merged and perform cleanup.

## Quick Check Command

Check PR merge status:
```powershell
gh pr view --json mergedAt,number,state,title
```

## Cleanup After Merge (Interactive)

If your PR is merged, run this workflow to clean up:
```powershell
# 1. Stash uncommitted changes (if any)
git stash push -m "WIP: auto-stash before cleanup"

# 2. Switch to main and sync
git checkout main
git fetch origin main
git reset --hard origin/main

# 3. Delete the merged feature branch (replace <branch-name> with your branch)
git branch -d <branch-name>
git push fork --delete <branch-name>
```

## One-Liner Auto-Cleanup

**WARNING**: This will automatically cleanup if PR is merged. Use with caution.

### PowerShell:
```powershell
$mergedAt = (gh pr view --json mergedAt --jq '.mergedAt' 2>$null); if ($mergedAt -and $mergedAt -ne 'null') { $branch = git branch --show-current; git stash push -m "WIP: auto-stash"; git checkout main; git fetch origin main; git reset --hard origin/main; git branch -d $branch; git push fork --delete $branch 2>$null; echo "✓ Cleaned up $branch" } else { echo "PR not merged yet" }
```

### Bash:
```bash
mergedAt=$(gh pr view --json mergedAt --jq '.mergedAt' 2>/dev/null); if [ "$mergedAt" != "null" ] && [ -n "$mergedAt" ]; then branch=$(git branch --show-current); git stash push -m "WIP: auto-stash"; git checkout main; git fetch origin main; git reset --hard origin/main; git branch -d $branch; git push fork --delete $branch 2>/dev/null; echo "✓ Cleaned up $branch"; else echo "PR not merged yet"; fi
```

## When to Use

- After receiving PR merge notification
- Before starting new work (to ensure clean state)
- When switching between multiple feature branches

## Manual Check (Safest)

Always check status first before cleanup:
```bash
# Check if merged (mergedAt will have a timestamp if merged, null if not)
gh pr view --json mergedAt,state

# If mergedAt has a timestamp, then manually run cleanup commands above
```
