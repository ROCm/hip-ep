<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Review Pull Request

Review a GitHub pull request using the GitHub CLI (gh).

## Step 1: Get PR Number

Ask the user for the PR number to review, or extract it from context.

## Step 2: Analyze the PR

### View PR Details
```bash
gh pr view <PR_NUMBER>
```

### View the Diff
```bash
gh pr diff <PR_NUMBER>
```

### Check CI Status
```bash
gh pr checks <PR_NUMBER>
```

### View Files Changed
```bash
gh pr view <PR_NUMBER> --json files --jq '.files[].path'
```

### View Commit Messages
```bash
gh pr view <PR_NUMBER> --json commits --jq '.commits[].messageHeadline'
```

### Get Line Count Changes
```bash
gh pr view <PR_NUMBER> --json additions,deletions --jq '"+\(.additions) -\(.deletions)"'
```

## Step 3: Review the PR

### Review the Description
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

### Review the Code
Analyze the diff output and provide feedback on:
- Code correctness
- Best practices
- Potential bugs or issues
- Suggestions for improvement

Read related files to understand the context:
- Check how similar patterns are implemented elsewhere
- Verify consistency with existing code
- Look for potential side effects in related files

## Step 4: Generate Review Comment

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

## Step 5: Submit Review

### Approve
```bash
gh pr review <PR_NUMBER> --approve --body "$(cat review_comment.md)"
```

### Request Changes
```bash
gh pr review <PR_NUMBER> --request-changes --body "$(cat review_comment.md)"
```

### Comment Only
```bash
gh pr review <PR_NUMBER> --comment --body "$(cat review_comment.md)"
```

## Step 6: Incremental Review (Optional)

If deeper investigation reveals additional findings, post a follow-up comment:

```markdown
<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
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

## Additional References

- Full Guide: `doc/pr-review/README.md`
- Command Reference: `doc/pr-review/QUICK_REFERENCE.md`
