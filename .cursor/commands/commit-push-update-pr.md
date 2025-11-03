<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
Please commit the current changes, push to the fork remote, and update the PR description.

## Settings to Remember:
- **Remote**: `fork` (git@gitenterprise.xilinx.com:<your-username>/MorphiZen.git)
- **Repository**: `VitisAI/MorphiZen`
- **GitHub Enterprise Host**: `gitenterprise.xilinx.com`

## Steps:

1. **Check current branch and staged changes:**
   ```powershell
   git branch --show-current
   git status
   git diff --cached --name-only
   ```

2. **Stage changes if needed:**
   ```powershell
   # Stage specific files
   git add <file1> <file2> ...
   # Or stage all changes
   git add -A
   ```

3. **Commit with descriptive message:**
   ```powershell
   git commit -m "type: brief description

   - Detailed change 1
   - Detailed change 2
   - Detailed change 3"
   ```

   Commit types:
   - `fix:` - Bug fixes
   - `feat:` - New features
   - `docs:` - Documentation changes
   - `refactor:` - Code refactoring
   - `test:` - Adding or updating tests
   - `chore:` - Maintenance tasks

4. **Push to fork remote:**
   ```powershell
   $currentBranch = git branch --show-current
   git push fork $currentBranch
   ```

5. **Find PR number for current branch:**
   ```powershell
   $env:GH_HOST = "gitenterprise.xilinx.com"
   gh pr list --repo VitisAI/MorphiZen --head <your-username>:$currentBranch
   ```

6. **Update PR description:**
   ```powershell
   # Create temp file with updated PR body
   $prBodyFile = Join-Path $env:TEMP "pr_update_$(Get-Date -Format 'yyyyMMdd_HHmmss').md"

   # Edit the PR body (open in editor)
   notepad $prBodyFile

   # Or write directly with PowerShell
   @"
   # Summary of Changes

   <Describe your changes here>

   # Motivation

   <Why these changes are needed>

   # Implementation

   <How you implemented the changes>
   "@ | Set-Content -Path $prBodyFile

   # Update the PR
   $env:GH_HOST = "gitenterprise.xilinx.com"
   gh pr edit <PR_NUMBER> --repo VitisAI/MorphiZen --body-file $prBodyFile

   # Clean up temp file
   Remove-Item $prBodyFile -ErrorAction SilentlyContinue
   ```

## Quick Workflow Script:

For quick updates, run all steps together:

```powershell
# Get current branch and PR number
$currentBranch = git branch --show-current
$env:GH_HOST = "gitenterprise.xilinx.com"

# Commit (assumes files are already staged)
git commit -m "Your commit message"

# Push to fork
git push fork $currentBranch

# Find PR number
$prNumber = (gh pr list --repo VitisAI/MorphiZen --head <your-username>:$currentBranch --json number --jq '.[0].number')

# Create and edit PR body
$prBodyFile = Join-Path $env:TEMP "pr_update_$(Get-Date -Format 'yyyyMMdd_HHmmss').md"
notepad $prBodyFile

# Update PR
gh pr edit $prNumber --repo VitisAI/MorphiZen --body-file $prBodyFile

# Clean up
Remove-Item $prBodyFile -ErrorAction SilentlyContinue
```

## Alternative: Update PR via Web:

If you prefer to edit the PR description in the browser:

```powershell
$currentBranch = git branch --show-current
$env:GH_HOST = "gitenterprise.xilinx.com"
gh pr view --repo VitisAI/MorphiZen --head <your-username>:$currentBranch --web
```

## Notes:
- Always review changes before committing (`git diff --cached`)
- Use descriptive commit messages following conventional commits format
- PR descriptions should follow the template structure
- Clean up temp files after updating PR
- Verify PR was updated successfully with `gh pr view <PR_NUMBER>`
