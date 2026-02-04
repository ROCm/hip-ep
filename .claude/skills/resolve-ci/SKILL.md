<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
---
name: resolve-ci
description: Autonomous loop to resolve merge conflicts and CI failures until PR merges
allowed-tools: [Bash, Read, Grep, Glob, Edit, Write, AskUserQuestion]
---

# /resolve-ci - Autonomous Conflict and CI Failure Resolution

## Purpose

Autonomously resolve merge conflicts and CI failures until PR merges, then clean up workspace.

## Main Loop

1. Launch monitor-pr.py
2. If error → stop
3. Parse status → execute phase → repeat until merged

---

## Phase 2: Conflict Resolution (Linear History)

When script returns `STATUS:NEEDS_FIX_CONFLICTS`:

**Rebase onto latest main:**
```bash
git fetch origin main
git rebase origin/main  # Linear history - replays commits on top of main
```

**Analyze conflicts:**
```bash
git diff --name-only --diff-filter=U
Read <conflicting_file>
```

**Resolve intelligently:**
- Simple patterns (whitespace, imports) → auto-merge
- Complex logic conflicts → analyze context and choose best resolution
- Too complex → mark as stuck, ask user

**Apply and continue rebase:**
```bash
Edit <conflicting_file>  # Remove conflict markers
git add <conflicting_file>
git rebase --continue  # Continue rebase (may have more conflicts if multiple commits)
```

**Push (force required since history rewritten):**
```bash
git push fork "$CURRENT_BRANCH" --force-with-lease  # Safer than --force
```

Then restart monitoring script.

---

## Phase 3: CI Failure Resolution

When script returns `STATUS:NEEDS_FIX_CI` (complex failures only - pre-commit handled in bash):

**Fetch logs:**
```bash
RUN_ID=$(gh run list --branch "$CURRENT_BRANCH" --limit 1 --json databaseId --jq '.[0].databaseId')
gh run view $RUN_ID --log-failed > /tmp/ci_logs.txt
Read /tmp/ci_logs.txt
```

**Categorize:**
- Build failures → extract error location, read code, fix syntax/types/includes
- Test failures → read test + implementation, fix logic bugs
- Other → analyze and attempt fix

**Fix build errors:**
```bash
Read <file_with_error>
Edit <file>  # Fix the issue
cmake --build ../../build/$(basename $PWD) --config Debug --parallel  # Verify
git add <file>
git commit -m "fix: resolve build error in <file>"
git push fork "$CURRENT_BRANCH"
```

**Fix test failures:**
```bash
Read test_file.cpp
Read implementation_file.cpp
Edit <file>  # Fix logic bug
../../build/$(basename $PWD)/bin/morphizen-unit-tests.exe --gtest_filter="<test_name>"  # Verify
git add <file>
git commit -m "fix: resolve test failure in <test_name>"
git push fork "$CURRENT_BRANCH"
```

Then restart monitoring script.

---

## Phase 4: Stuck Detection

**Track attempts:**
- Same error 3 times → stuck
- 7 total attempts → stuck
- Critical errors (git push failure, broken CMake) → stuck

**Ask user when stuck:**
```
❌ Stuck on error (tried 3 approaches):
[error details]

What should I do?
- retry: Try current approach again
- different: Try different approach (you describe)
- skip: Continue monitoring, ignore this
- manual: Exit, you'll fix manually
```

---

## Phase 5: Cleanup After Merge

When script returns `STATUS:MERGED`:

```bash
git checkout main
git pull origin main
git branch -d "$CURRENT_BRANCH"
git push fork --delete "$CURRENT_BRANCH"
```

Done!
