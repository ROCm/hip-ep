<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #063: Fix /fix-issue Skill Bugs

## Problem

The `/fix-issue` skill has two bugs that prevent it from working correctly:

### Bug 1: setup-workspace.py fails when remote branch already exists

**Symptom:**
```
git push -u fork "feature/issue-062-cleanup-root-config-files"
! [rejected] ... (fetch first)
error: failed to push some refs
```

**Root cause:**
- `setup-workspace.py` doesn't check if branch already exists on fork
- Fails with push error if someone else is working on the same issue
- Should detect existing branch and either:
  - Warn user and exit cleanly, OR
  - Use unique branch names (e.g., append timestamp)

**Impact:** Skill fails when trying to work on issues that have active PRs

### Bug 2: Confusing error messages

**Current behavior:**
When script fails, error points to wrong directory:
```
C:\ProgramData\miniconda3\python.exe: can't open file
'C:\\Develop\\m\\source\\morphizen.github.1\\scripts\\setup-workspace.py'
```

**Root cause:**
- Error message shows incorrect path (`.1` instead of `.2`)
- Path should be `.claude/skills/fix-issue/setup-workspace.py`
- Confusing for users trying to debug

**Impact:** Users get misleading error messages

## Solution

### Fix 1: Handle existing remote branches gracefully

**Update setup-workspace.py to:**

1. **Before creating branch, check if it exists on fork:**
   ```python
   # Check if branch exists on remote
   result = subprocess.run(
       ['git', 'ls-remote', '--heads', 'fork', branch_name],
       capture_output=True, text=True
   )
   if result.stdout.strip():
       print(f"❌ Branch '{branch_name}' already exists on fork.")
       print(f"   Someone else may be working on issue #{issue_num}.")
       print(f"   Check: gh pr list --head {branch_name}")
       sys.exit(1)
   ```

2. **Alternative: Use unique branch names**
   ```python
   import time
   timestamp = int(time.time())
   branch_name = f"feature/issue-{issue_num:03d}-{slug}-{timestamp}"
   ```

**Recommendation:** Use **Option 1** (check and exit cleanly)
- Prevents duplicate work
- User can check existing PR
- Simpler than managing timestamps

### Fix 2: Improve error context

**Already correct in current implementation:**
- Script is at `.claude/skills/fix-issue/setup-workspace.py`
- Skill invokes it correctly
- No documentation fix needed (SKILL.md just says "Run `setup-workspace.py`")

**Only fix needed:** Better error message on push failure showing:
```python
except subprocess.CalledProcessError as e:
    if 'rejected' in e.stderr:
        print(f"\n❌ Push failed - branch may already exist on fork")
        print(f"   Check: gh pr list --head {branch_name}")
        print(f"   To force: git push -u fork {branch_name} --force-with-lease")
    raise
```

## Implementation Plan

**File:** `.claude/skills/fix-issue/setup-workspace.py`

### Step 1: Add remote branch check function

After line 30 (after `run()` function), add:

```python
def check_remote_branch_exists(branch_name):
    """Check if branch exists on fork remote."""
    try:
        result = subprocess.run(
            ['git', 'ls-remote', '--heads', 'fork', f'refs/heads/{branch_name}'],
            capture_output=True,
            text=True,
            check=False
        )
        return bool(result.stdout.strip())
    except Exception as e:
        print(f"⚠️  Warning: Could not check remote branch: {e}")
        return False
```

### Step 2: Check before creating branch

In `main()` function, after line ~75 (after branch name is generated), add:

```python
    branch_name = f"feature/issue-{issue_num:03d}-{slug}"

    # Check if branch already exists on fork
    if check_remote_branch_exists(branch_name):
        print(f"\n❌ Branch '{branch_name}' already exists on fork.")
        print(f"   Someone else may be working on issue #{issue_num}.")
        print(f"\n   To check the existing PR:")
        print(f"   gh pr list --head {branch_name}")
        print(f"\n   To view PRs for this issue:")
        print(f"   gh pr list --search 'issue #{issue_num}' --state all")
        sys.exit(1)

    print(f"🌿 Creating branch: {branch_name}")
```

### Step 3: Improve push error handling

Replace the push command section (around line 127) with:

```python
    print("📤 Pushing to fork...")
    try:
        run(f'git push -u fork "{branch_name}"')
    except subprocess.CalledProcessError as e:
        if 'rejected' in str(e):
            print(f"\n❌ Push rejected - branch exists on fork despite check")
            print(f"   This shouldn't happen. The branch may have been created between checks.")
            print(f"   Check: gh pr list --head {branch_name}")
        raise
```

### Step 4: Test

**Test case 1: Normal flow (no conflict)**
```bash
python .claude/skills/fix-issue/setup-workspace.py 055
# Should work normally
```

**Test case 2: Existing branch**
```bash
# Create a dummy branch on fork first
git checkout -b feature/issue-055-test
git push fork feature/issue-055-test
git checkout main
git branch -D feature/issue-055-test

# Now try to use the skill
python .claude/skills/fix-issue/setup-workspace.py 055
# Should exit cleanly with helpful message
```

**Test case 3: Cleanup**
```bash
git push fork --delete feature/issue-055-test
```

## Success Criteria

- [ ] Script checks if remote branch exists before creating
- [ ] Clear error message if branch already exists
- [ ] Suggests commands to check existing PR
- [ ] Exits cleanly (exit code 1) without leaving dirty state
- [ ] Works correctly when no conflict exists
- [ ] Error messages are helpful and actionable

## Metadata

- **Type:** Bug Fix
- **Priority:** MEDIUM
- **Estimated time:** 30 minutes
- **Component:** /fix-issue skill
- **Files affected:** `.claude/skills/fix-issue/setup-workspace.py`
