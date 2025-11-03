<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
Please create a pull request comparing the current branch with the main branch.

## Settings to Remember:
- **Remote**: `fork` (git@gitenterprise.xilinx.com:<your-username>/MorphiZen.git)
- **Base Branch**: `main` (from VitisAI/MorphiZen)
- **GitHub Enterprise Host**: `gitenterprise.xilinx.com`

## Steps:

1. **Check current branch and changes:**
   ```powershell
   git branch --show-current
   git diff main...HEAD --name-only
   git log main..HEAD --oneline
   ```

2. **Create PR description in temp file:**
   ```powershell
   # Create temp file based on the PR template
   $prBodyFile = Join-Path $env:TEMP "pr_body_$(Get-Date -Format 'yyyyMMdd_HHmmss').md"
   Copy-Item .github/pull_request_template.md $prBodyFile
   # Open in default editor to fill in the template
   notepad $prBodyFile
   ```

   Fill in the template with:
   - **Summary of Changes**: High-level description of what this PR adds/changes/improves/fixes
   - **Motivation**: Why this change is needed
   - **Implementation**: Details about how it was implemented and any design choices

3. **Push branch to fork:**
   ```powershell
   git push fork <current-branch-name>
   ```

4. **Create PR using GitHub CLI:**

   ```powershell
   $env:GH_HOST = "gitenterprise.xilinx.com"
   gh pr create --repo VitisAI/MorphiZen --base main --head <your-username>:<current-branch-name> --title "Your PR Title" --body-file $prBodyFile
   ```

   Or use `--web` flag to open browser:
   ```powershell
   gh pr create --repo VitisAI/MorphiZen --base main --head <your-username>:<current-branch-name> --title "Your PR Title" --body-file $prBodyFile --web
   ```

   Note: The `$prBodyFile` variable is set from step 2. If you're running this in a new shell session, you'll need to set it again:
   ```powershell
   $prBodyFile = (Get-ChildItem $env:TEMP\pr_body_*.md | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
   ```

5. **Verify PR was created:**
   ```powershell
   gh pr list --repo VitisAI/MorphiZen --author <your-username> --limit 5
   ```

6. **Clean up temp file (optional):**
   ```powershell
   Remove-Item $prBodyFile -ErrorAction SilentlyContinue
   ```

## PR Body Template Structure:

```markdown
# Summary of Changes

* High-level description of changes
* What this PR adds/changes/improves/fixes
* Each request should have an overarching theme

# Motivation

Describe why this request is needed/good/useful.
Link to related issues if applicable.

# Implementation

Details about:
- How it was implemented
- Design choices made
- What alternatives were tried
- Testing performed
```

## Alternative: Manual Creation

If GitHub CLI doesn't work, open this URL:
```
https://gitenterprise.xilinx.com/VitisAI/MorphiZen/compare/main...<your-username>:<current-branch-name>
```

## Notes:
- Always follow the PR template structure
- Include testing details and verification
- Reference related issues if any
- Keep PR focused on a single theme
- The temp file approach keeps your working directory clean
- Temp files are automatically created with timestamps to avoid conflicts
