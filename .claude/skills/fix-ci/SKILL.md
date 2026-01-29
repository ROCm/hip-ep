<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
---
name: fix-ci
description: Monitor and automatically fix CI check failures for the current branch's PR. Watches checks in real-time and auto-remediates fixable issues like pre-commit formatting.
allowed-tools: [Bash, Read, Grep, Glob, Edit]
---

# /fix-ci - Monitor and Fix CI Check Failures

## Purpose

Monitor CI check status for the current branch's Pull Request and automatically fix common failures like pre-commit formatting issues. Provides continuous monitoring until all checks pass or timeout.

## When to Use

- After pushing changes to a PR and CI checks are running
- When CI checks fail and you need to identify and fix issues
- To avoid manual back-and-forth of checking status and applying fixes
- For automated monitoring until all checks pass

## Workflow Overview

1. **PR Detection**: Find and validate PR for current branch
2. **Initial Status**: Display current CI check states
3. **Auto-Fix**: Attempt to automatically fix common failures (pre-commit)
4. **Monitor**: Continuously watch checks until all pass or timeout
5. **Report**: Provide final summary of results

---

## Phase 1: PR Detection

### Step 1: Check Current Branch

```bash
git branch --show-current
```

**Validation**:
- If on `main` branch: Error and stop - must be on feature branch
- Store branch name for later use

### Step 2: Find Pull Request

```bash
gh pr view --json number,state,isDraft,url,title
```

**Scenarios**:

**A. PR exists and is OPEN**:
- Continue to Phase 2
- Display: "Found PR #<number>: <title>"
- Display: "URL: <url>"
- Display: "Status: <DRAFT | OPEN>"

**B. PR exists but is MERGED or CLOSED**:
- Error and stop: "PR #<number> is <MERGED|CLOSED>. Switch to a feature branch with an open PR."

**C. No PR found**:
- Ask user: "No PR found for branch <branch-name>. Would you like to create a draft PR?"
- If yes:
  ```bash
  gh pr create --draft --title "WIP: <branch-name>" --body "Work in progress"
  ```
  Display: "Created draft PR #<number>"
  Continue to Phase 2

- If no:
  Stop and display: "Run /fix-ci after creating a PR for this branch."

---

## Phase 2: Initial CI Status Check

### Step 1: Fetch CI Check Status

```bash
gh pr view --json statusCheckRollup
```

Parse JSON output to extract:
- Check name (e.g., "pre-commit", "build-and-test-lnx", "build-and-test-win")
- Check status (SUCCESS, FAILURE, PENDING, QUEUED, IN_PROGRESS, EXPECTED, SKIPPED)
- Check conclusion (if completed)

### Step 2: Categorize Checks

Group checks into categories:
- **Passing**: SUCCESS status
- **Failing**: FAILURE status
- **Pending**: PENDING, QUEUED, IN_PROGRESS, EXPECTED
- **Skipped**: SKIPPED status

### Step 3: Display Status Summary

```
CI Status for PR #<number>: <title>

✓ Passing (N):
  ✓ build-and-test-lnx
  ✓ build-and-test-win

✗ Failing (M):
  ✗ pre-commit

○ Pending (K):
  ○ code-coverage

Summary: N passing, M failing, K pending
```

### Step 4: Determine Next Action

**If all checks passing**:
- Display: "All CI checks passed! PR is ready for review."
- Skip to Phase 5 (Final Report)

**If any checks failing**:
- Continue to Phase 3 (Auto-Fix)

**If only pending (no failures)**:
- Continue to Phase 4 (Monitor) - wait for pending checks to complete

---

## Phase 3: Auto-Fix Attempts

For each failing check, determine if it's auto-fixable:

### Auto-Fixable: pre-commit Check

**When**: Check name is "pre-commit" and status is FAILURE

**Safety Checks Before Auto-Fix**:

1. **Verify branch is not main**:
   ```bash
   git branch --show-current
   ```
   If `main`, error and skip auto-fix

2. **Check for uncommitted changes**:
   ```bash
   git status --porcelain
   ```
   If uncommitted changes exist, display:
   ```
   WARNING: Uncommitted changes detected.
   Auto-fix will commit all changes together with formatting fixes.

   Uncommitted files:
   <list files>

   Continue with auto-fix? (This will stage and commit all changes)
   ```
   - If user confirms: Continue
   - If user declines: Skip to Phase 4 (Monitor)

**Auto-Fix Procedure**:

1. **Run pre-commit locally**:
   ```bash
   pre-commit run --all-files --hook-stage manual
   ```

   Display output showing:
   - Which hooks ran
   - Which files were modified
   - Any errors encountered

2. **Check what changed**:
   ```bash
   git status --porcelain
   git diff --cached --numstat
   ```

   **Binary detection**:
   ```bash
   git diff --cached --numstat | grep '^-'
   ```
   If binaries detected:
   - Error: "Binary files detected in changes. Manual review required."
   - Skip auto-commit
   - Continue to Phase 4 (Monitor)

3. **Review changes**:
   Display summary:
   ```
   Pre-commit made the following changes:
   - Modified N files (line ending normalization, trailing whitespace, etc.)
   - No binaries detected

   Files to be committed:
   <list files with change counts>
   ```

4. **Stage all changes**:
   ```bash
   git add -A
   ```

5. **Commit with descriptive message**:
   ```bash
   git commit -m "style: apply pre-commit formatting fixes"
   ```

6. **Push to fork**:
   ```bash
   git push fork $(git branch --show-current)
   ```

   Display: "Pushed formatting fixes to fork/<branch-name>"

7. **Record attempt**:
   - Track that pre-commit auto-fix was attempted
   - Increment attempt counter for pre-commit check
   - Max 3 attempts per check (prevent infinite loops)

### Manual Fix Required: Build/Test Failures

**When**: Check name contains "build" or "test" and status is FAILURE

**Cannot auto-fix** - requires code changes. Provide guidance:

1. **Fetch failure logs**:
   ```bash
   gh run view --log-failed
   ```
   Or use check URL from statusCheckRollup

2. **Display relevant errors**:
   Parse logs for error messages, extract key information:
   - Compiler errors (file:line: error: message)
   - Test failures (FAILED test_name - message)
   - Link errors (undefined reference, multiple definition)

3. **Provide guidance**:
   ```
   Build failure detected in: <check-name>

   Error summary:
   <extracted error messages>

   This requires manual fixing:
   1. Review the error messages above
   2. Fix the code issues locally
   3. Commit and push your changes
   4. /fix-ci will continue monitoring automatically

   Check logs: gh run view --log-failed
   ```

4. **Continue to Phase 4 (Monitor)** - watch for user's fix

### Other Failing Checks

For other check types (code-coverage, security scans, etc.):
- Display check name and failure status
- Show check URL if available
- Provide generic guidance: "Manual review required. See check details at <URL>"
- Continue to Phase 4 (Monitor)

---

## Phase 4: Continuous Monitoring

After initial status and any auto-fix attempts, continuously monitor checks.

### Monitoring Loop

1. **Wait interval**: 30 seconds between status checks

2. **Fetch updated status**:
   ```bash
   gh pr view --json statusCheckRollup
   ```

3. **Detect state changes**:
   Compare with previous status:
   - Check moved from PENDING → SUCCESS: Display "✓ <check> passed"
   - Check moved from PENDING → FAILURE: Display "✗ <check> failed"
   - Check moved from FAILURE → PENDING: Display "○ <check> restarted"
   - New check appeared: Display "○ New check: <check>"

4. **Handle new failures**:
   If a check that was passing/pending now fails:
   - If pre-commit and retry count < 3: Attempt auto-fix (Phase 3)
   - Otherwise: Show failure message and guidance

5. **Check termination conditions**:

   **A. All checks passing**:
   - Display: "All CI checks passed! PR is ready for review."
   - Go to Phase 5 (Final Report)

   **B. Timeout reached** (30 minutes since start):
   - Display: "Monitoring timeout reached (30 minutes)"
   - Go to Phase 5 (Final Report)

   **C. User interruption**:
   - Allow Ctrl+C to stop monitoring
   - Display: "Monitoring stopped by user"
   - Go to Phase 5 (Final Report)

6. **Display periodic updates**:
   Every 5 minutes, show current status summary:
   ```
   [15:30] Still monitoring... N passing, M failing, K pending
   ```

### Retry Limits

**Per-check retry limit**: Max 3 auto-fix attempts per check
- Track attempts in memory during monitoring session
- If pre-commit fails after 3 auto-fix attempts:
  - Display: "Pre-commit still failing after 3 auto-fix attempts. Manual review required."
  - Stop attempting auto-fix for pre-commit
  - Continue monitoring other checks

### Error Handling During Monitoring

**Network/API errors**:
```bash
gh pr view --json statusCheckRollup
```
If command fails:
- Display: "Network error fetching CI status. Retrying in 30s..."
- Continue monitoring (don't exit)
- After 3 consecutive network errors:
  - Display: "Repeated network errors. Check connection and try again."
  - Go to Phase 5 (Final Report)

**Push failures during auto-fix**:
If `git push` fails:
- Display error message
- Suggest fixes:
  - Authentication: "Run: gh auth login"
  - Conflicts: "Run: git pull --rebase fork <branch>"
  - Permissions: "Verify fork remote is configured correctly"
- Continue monitoring (don't exit)

---

## Phase 5: Final Report

Display comprehensive summary of monitoring session.

### Success Scenario

All checks passed:

```
✓ SUCCESS: All CI checks passed!

Final status:
✓ pre-commit            PASSED (auto-fixed: line endings, trailing whitespace)
✓ build-and-test-lnx    PASSED
✓ build-and-test-win    PASSED

Auto-fixes applied:
- 1 commit: "style: apply pre-commit formatting fixes"
  Modified 8 files (line ending normalization)

PR is ready for review!
Run: gh pr ready (if currently draft)
```

### Partial Success Scenario

Some checks passed, others still failing:

```
⚠ PARTIAL: Some checks passed, others require manual fixes

Final status:
✓ pre-commit            PASSED (auto-fixed)
✓ build-and-test-lnx    PASSED
✗ build-and-test-win    FAILED (manual fix required)

Auto-fixes applied:
- 1 commit: "style: apply pre-commit formatting fixes"

Remaining failures:
✗ build-and-test-win: Compiler error in src/model.cpp:127
  Review and fix manually, then push changes.

Run /fix-ci again to continue monitoring after your fix.
```

### Timeout Scenario

Monitoring timed out before all checks completed:

```
⏱ TIMEOUT: Monitoring timed out after 30 minutes

Final status:
✓ pre-commit            PASSED (auto-fixed)
○ build-and-test-lnx    PENDING (still running)
○ build-and-test-win    PENDING (still running)

Auto-fixes applied:
- 1 commit: "style: apply pre-commit formatting fixes"

Checks are still running. Options:
1. Wait and run /fix-ci again to resume monitoring
2. Check manually: gh pr view --web
```

### Failure Scenario

All auto-fix attempts exhausted, manual intervention required:

```
✗ FAILED: CI checks require manual fixes

Final status:
✗ pre-commit            FAILED (3 auto-fix attempts exhausted)
✗ build-and-test-lnx    FAILED
✓ build-and-test-win    PASSED

Auto-fix attempts:
- pre-commit: 3 attempts made, still failing
  Last error: hook 'check-yaml' failed on .github/workflows/ci.yml

Manual fixes required:
1. pre-commit: Review .github/workflows/ci.yml for YAML syntax errors
2. build-and-test-lnx: Compiler error in src/cache.cpp:45

After fixing, commit and push, then run /fix-ci to monitor.
```

---

## Safety Guarantees

### Branch Protection

- **Never** run on `main` branch
- Always verify current branch before any git operations
- Abort if branch check fails

### Binary Detection

Before any commit:
```bash
git diff --cached --numstat | grep '^-'
```
If binaries detected, abort commit and warn user.

### Diff Review

Before committing auto-fixes:
- Show file list and change counts
- Verify changes are reasonable (formatting only, not code logic)
- User can see exactly what will be committed

### Retry Limits

- Max 3 auto-fix attempts per check
- Prevents infinite loop of fix attempts
- After limit reached, require manual intervention

### Timeout

- Maximum 30 minutes of continuous monitoring
- Prevents indefinite running
- User can restart monitoring after timeout

### Error Recovery

- All network errors are retried (up to 3 times)
- Push failures don't crash the monitoring loop
- User always gets clear guidance on what to do next

---

## Example Usage Scenarios

### Scenario 1: Pre-commit Failure (Auto-Fixed)

```
$ /fix-ci

Found PR #42: feature/add-logging
URL: https://github.com/ROCm/MorphiZen/pull/42
Status: DRAFT

CI Status for PR #42:
✓ Passing (2):
  ✓ build-and-test-lnx
  ✓ build-and-test-win

✗ Failing (1):
  ✗ pre-commit

Summary: 2 passing, 1 failing

Attempting auto-fix for pre-commit...

Running: pre-commit run --all-files --hook-stage manual

[INFO] Trim trailing whitespace........................Passed
[INFO] Fix end of files.................................Passed
[INFO] Check for added large files......................Passed
[INFO] Mixed line ending................................Failed
- hook id: mixed-line-ending
- files were modified by this hook
  src/cache.cpp
  src/cache.hpp
  tests/cache_test.cpp

Pre-commit made the following changes:
- Modified 3 files (line ending normalization CRLF → LF)
- No binaries detected

Files to be committed:
  3 files changed, 0 insertions(+), 0 deletions(-)

Committing: style: apply pre-commit formatting fixes
Pushing to fork/feature/add-logging...

Pushed formatting fixes to fork/feature/add-logging

Monitoring checks (poll every 30s)...

[30s] ○ pre-commit restarted (new run triggered)
[2m]  ✓ pre-commit passed

✓ SUCCESS: All CI checks passed!

Final status:
✓ pre-commit            PASSED (auto-fixed: line endings)
✓ build-and-test-lnx    PASSED
✓ build-and-test-win    PASSED

Auto-fixes applied:
- 1 commit: "style: apply pre-commit formatting fixes"
  Modified 3 files (line ending normalization)

PR is ready for review!
```

### Scenario 2: Build Failure (Manual Fix Required)

```
$ /fix-ci

Found PR #43: fix/memory-leak
URL: https://github.com/ROCm/MorphiZen/pull/43
Status: OPEN

CI Status for PR #43:
✗ Failing (1):
  ✗ build-and-test-lnx

✓ Passing (2):
  ✓ build-and-test-win
  ✓ pre-commit

Summary: 2 passing, 1 failing

Build failure detected in: build-and-test-lnx

Error summary:
  morphizen-core/src/model.cpp:127:15: error: use of undeclared identifier 'config'
    auto cache_dir = config.get_cache_dir();
                     ^
  1 error generated.

This requires manual fixing:
1. Review the error messages above
2. Fix the code issues locally
3. Commit and push your changes
4. /fix-ci will continue monitoring automatically

Check logs: gh run view --log-failed

Monitoring checks (poll every 30s)...

[User fixes the code and pushes]

[2m]  ○ build-and-test-lnx restarted (new run triggered)
[7m]  ✓ build-and-test-lnx passed

✓ SUCCESS: All CI checks passed!

Final status:
✓ build-and-test-lnx    PASSED
✓ build-and-test-win    PASSED
✓ pre-commit            PASSED

No auto-fixes applied (manual fix was required)

PR is ready for review!
```

### Scenario 3: Multiple Failures, Mixed Auto/Manual

```
$ /fix-ci

Found PR #44: refactor/api-cleanup
URL: https://github.com/ROCm/MorphiZen/pull/44
Status: DRAFT

CI Status for PR #44:
✗ Failing (2):
  ✗ pre-commit
  ✗ build-and-test-win

✓ Passing (1):
  ✓ build-and-test-lnx

Summary: 1 passing, 2 failing

Attempting auto-fix for pre-commit...
[... pre-commit auto-fix succeeds ...]

Build failure detected in: build-and-test-win

Error summary:
  error LNK2019: unresolved external symbol "class Cache * __cdecl create_cache(void)"
  onnxruntime_morphizen_ep.dll : fatal error LNK1120: 1 unresolved externals

This requires manual fixing:
1. Review the error messages above
2. Fix the code issues locally
3. Commit and push your changes
4. /fix-ci will continue monitoring automatically

Monitoring checks (poll every 30s)...

[1m]  ○ pre-commit restarted
[3m]  ✓ pre-commit passed

[User fixes linker error and pushes]

[5m]  ○ build-and-test-win restarted
[12m] ✓ build-and-test-win passed

✓ SUCCESS: All CI checks passed!

Final status:
✓ pre-commit            PASSED (auto-fixed: formatting)
✓ build-and-test-lnx    PASSED
✓ build-and-test-win    PASSED (manual fix required)

Auto-fixes applied:
- 1 commit: "style: apply pre-commit formatting fixes"

PR is ready for review!
```

---

## Commands Reference

### Check Current Branch
```bash
git branch --show-current
```

### Find PR for Current Branch
```bash
gh pr view --json number,state,isDraft,url,title
```

### Create Draft PR
```bash
gh pr create --draft --title "WIP: <branch-name>" --body "Work in progress"
```

### Fetch CI Check Status
```bash
gh pr view --json statusCheckRollup
```

### Run Pre-commit Locally
```bash
pre-commit run --all-files --hook-stage manual
```

### Check for Binaries in Staged Changes
```bash
git diff --cached --numstat | grep '^-'
```

### Fetch Failure Logs
```bash
gh run view --log-failed
```

### Mark PR as Ready for Review
```bash
gh pr ready
```

---

## Limitations

### Cannot Auto-Fix

- Build failures (compiler errors, linker errors)
- Test failures (failing unit tests, integration tests)
- Code coverage issues
- Security scan findings
- Custom checks requiring code changes

### Requires Manual Intervention

- Complex pre-commit failures (e.g., YAML syntax errors, Python import issues)
- Merge conflicts during rebase
- Authentication issues with gh or git
- Network connectivity problems

### Best Practices

- Run `/fix-ci` immediately after pushing to PR
- Don't leave PRs with failing checks unattended
- Fix build/test issues promptly when notified
- Review auto-fix commits to understand what changed
- Use `/fix-ci` iteratively as you work on fixes

---

## Related Skills and Documentation

- `/status` - Quick check of current branch and PR status
- `/update-branch` - Update feature branch with latest from main
- `/build-and-test` - Local build and test execution
- `docs/workflows/git-workflow.md` - Git workflow rules and best practices
- `docs/workflows/pr-workflow.md` - Pull request creation and review process
- `docs/technical/pre-commit.md` - Pre-commit hook configuration and usage
