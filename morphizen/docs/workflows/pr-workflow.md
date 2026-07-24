<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Pull Request Workflow

Complete guide for working with pull requests in the MorphiZen project.

## Creating a Pull Request

See [git-workflow.md](git-workflow.md#required-workflow) for the complete workflow.

Quick summary:
1. Create feature branch from `main`
2. Make initial commit
3. Push to fork: `git push fork <branch>`
4. Create DRAFT PR: `gh pr create --draft`
5. Continue working, commit and push frequently
6. Mark as ready for review when complete

**PR Title Convention for Backlog Issues:**

When creating PRs for backlog issues, ALWAYS include the issue number in the title:

- ✅ CORRECT: `Issue #006: Remove legacy cache_dir system (~220-280 LOC)`
- ❌ WRONG: `Remove legacy cache_dir system (~220-280 LOC)`

Example:
```bash
gh pr create --draft --title "Issue #006: Remove legacy cache_dir system" --body "..."
```

This ensures PRs are easily linked to their corresponding issues in docs/project/backlog.md.

## Switching to a PR Branch

### Quick Check: Current Branch

```bash
git branch --show-current
```

- If already on the target PR branch → skip to "Fetch and Rebase" below
- If on a different branch → continue with steps below

### Step 1: Check for Uncommitted Changes

```bash
git status --short
```

If there are uncommitted changes, stash them:

```bash
git stash push -m "WIP: stashed before switching to PR"
```

### Step 2: Fetch and Checkout PR Branch

Fetch all remote updates and checkout the PR:

```bash
git fetch origin
gh pr checkout <PR_NUMBER>
```

The `gh pr checkout` command fetches the PR branch and switches to it in one step.

### Step 3: Rebase on Latest Main

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

### Step 4: Push to Remote (Sync)

After rebasing, push the updated branch to fork:

```bash
git push --force-with-lease fork
```

Use `--force-with-lease` instead of `--force` for safety - it will fail if someone else pushed changes you don't have.

### Quick Command (All Steps)

For switching to a PR and syncing in one operation:

```powershell
git stash push -m "WIP: auto-stash"; git fetch origin; gh pr checkout <PR_NUMBER>; git rebase origin/main; git push --force-with-lease fork
```

### When to Use

- When you need to continue work on an existing PR
- When reviewing and testing someone else's PR locally
- When updating a PR with the latest main branch changes
- When syncing local PR changes with remote

## Reviewing a Pull Request

### Step 1: Get PR Number

Ask for the PR number to review, or extract it from context.

### Step 2: Analyze the PR

#### View PR Details
```bash
gh pr view <PR_NUMBER>
```

#### View the Diff
```bash
gh pr diff <PR_NUMBER>
```

#### Check CI Status
```bash
gh pr checks <PR_NUMBER>
```

#### View Files Changed
```bash
gh pr view <PR_NUMBER> --json files --jq '.files[].path'
```

#### View Commit Messages
```bash
gh pr view <PR_NUMBER> --json commits --jq '.commits[].messageHeadline'
```

#### Get Line Count Changes
```bash
gh pr view <PR_NUMBER> --json additions,deletions --jq '"+\(.additions) -\(.deletions)"'
```

### Step 3: Review the PR

#### Review the Description
Analyze the PR description for:
- Clear explanation of what the PR does and why
- Compliance with PR template (if applicable)
- Links to related issues or documentation
- Breaking changes or migration notes (if needed)

If the description is incomplete or missing, offer to generate/update it:
```bash
gh pr edit <PR_NUMBER> --body "Your generated description here"
```

Generate a description based on:
1. The diff output (what changed)
2. Commit messages (`gh pr view <PR_NUMBER> --json commits --jq '.commits[].messageHeadline'`)
3. The PR template from `.github/pull_request_template.md` (if exists)

#### Review the Code
Analyze the diff output and provide feedback on:
- Code correctness
- Best practices
- Potential bugs or issues
- Suggestions for improvement

Read related files to understand the context:
- Check how similar patterns are implemented elsewhere
- Verify consistency with existing code
- Look for potential side effects in related files

### Step 4: Generate Review Comment

Use the following template to generate a structured review comment:

```markdown
# Code Review - PR #<NUMBER>: <TITLE>

## Summary
[Brief overview of changes and purpose]

## Changes
- `file1.ext` - [description]
- `file2.ext` - [description]

## Findings

### ✅ Strengths
- [Strength 1]
- [Strength 2]

### 🔴 Issues (if any)
- **[Critical/Medium/Low]:** [Issue and recommendation]

## Verdict
**Status:** ✅ APPROVED / ⚠️ CHANGES REQUESTED

**Risk:** LOW/MEDIUM/HIGH

[Brief closing statement]
```

For complex PRs requiring detailed analysis, expand with optional sections:
- **Code Quality Table:** Score aspects like correctness, readability, testing (1-5 stars)
- **Before/After:** Document specific behavior changes with impact
- **Edge Cases:** List considered edge cases and how they're handled
- **Testing Notes:** Checklist of test cases to verify

### Step 5: Check PR Authorship

Before submitting a review, check if you are the PR author:

```bash
gh pr view <PR_NUMBER> --json author --jq '.author.login'
```

**Important:** GitHub does not allow self-approval. If you are the PR author:
- Use `gh pr comment` instead of `gh pr review --approve`
- Or use the global `approve-my-pr` workflow with a separate token if self-approval is needed

### Step 6: Submit Review

#### If NOT the PR author:

##### Approve
```bash
gh pr review <PR_NUMBER> --approve --body-file review_comment.md
```

##### Request Changes
```bash
gh pr review <PR_NUMBER> --request-changes --body-file review_comment.md
```

##### Comment Only
```bash
gh pr review <PR_NUMBER> --comment --body-file review_comment.md
```

#### If you ARE the PR author:

Post as a comment instead (self-approval is not allowed by GitHub):
```bash
gh pr comment <PR_NUMBER> --body-file review_comment.md
```

### Step 7: Incremental Review (Optional)

If deeper investigation reveals additional findings, post a follow-up comment:

```markdown
# Incremental Review - PR #<NUMBER>: <Finding Title>

## Summary
[Brief overview of what was discovered during deeper investigation]

## Investigation Results

### Current Usage
- [Finding 1]
- [Finding 2]
- [Finding 3]

### Files Checked
- ✅ `path/to/file1.ext` - [What was found]
- ✅ `path/to/file2.ext` - [What was found]
- ⚠️ `path/to/file3.ext` - [Potential issue found]

## Requested Changes

### Option 1: [Recommended Approach]
[Detailed explanation of the recommended approach with code examples]

### Option 2: [Alternative Approach]
[Detailed explanation of the alternative approach with code examples]

## Impact Assessment

**Current PR Fix:** ✅/❌ [Assessment of the current fix]

**Additional Issue:** ⚠️/🔴 [Priority] - [Description of any additional issues found]

## Recommendation

**Request Changes:** Please either:
1. [Option 1 action]
2. [Option 2 action]

## Questions for Author

1. [Question 1]?
2. [Question 2]?
3. [Question 3]?

---

**Note:** [Any additional notes or clarifications]
```

Post follow-up comment:
```bash
gh pr comment <PR_NUMBER> --body "$(cat incremental_review.md)"
```

## Checking PR Status

### View Current Branch's PR Status

```bash
gh pr view --json state,merged,title
```

### View a Specific PR

```bash
gh pr view <PR_NUMBER> --json state,merged,mergedAt
```

### List Your Open PRs

```bash
gh pr list --author @me --state open
```

## Checking if PR is Merged

### Quick Check Command

Check PR merge status:
```powershell
gh pr view --json mergedAt,number,state,title
```

### Cleanup After Merge (Interactive)

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

### One-Liner Auto-Cleanup

**WARNING**: This will automatically cleanup if PR is merged. Use with caution.

#### PowerShell:
```powershell
$mergedAt = (gh pr view --json mergedAt --jq '.mergedAt' 2>$null); if ($mergedAt -and $mergedAt -ne 'null') { $branch = git branch --show-current; git stash push -m "WIP: auto-stash"; git checkout main; git fetch origin main; git reset --hard origin/main; git branch -d $branch; git push fork --delete $branch 2>$null; echo "✓ Cleaned up $branch" } else { echo "PR not merged yet" }
```

#### Bash:
```bash
mergedAt=$(gh pr view --json mergedAt --jq '.mergedAt' 2>/dev/null); if [ "$mergedAt" != "null" ] && [ -n "$mergedAt" ]; then branch=$(git branch --show-current); git stash push -m "WIP: auto-stash"; git checkout main; git fetch origin main; git reset --hard origin/main; git branch -d $branch; git push fork --delete $branch 2>/dev/null; echo "✓ Cleaned up $branch"; else echo "PR not merged yet"; fi
```

### When to Check and Cleanup

- After receiving PR merge notification
- Before starting new work (to ensure clean state)
- When switching between multiple feature branches

### Manual Check (Safest)

Always check status first before cleanup:
```bash
# Check if merged (mergedAt will have a timestamp if merged, null if not)
gh pr view --json mergedAt,state

# If mergedAt has a timestamp, then manually run cleanup commands above
```

## Notes

- Always check your current branch before switching
- Stash uncommitted changes to avoid losing work
- Rebasing rewrites history - use `--force-with-lease` when pushing
- Enable "Automatically delete head branches" in GitHub repo settings to auto-clean merged branches
- After cleanup, your working directory should be clean on main branch
