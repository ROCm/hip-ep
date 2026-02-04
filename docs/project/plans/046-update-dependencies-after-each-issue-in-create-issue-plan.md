<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Update Dependencies After Each Issue in /create-issue

**Issue:** #046
**Created:** 2026-02-03
**Status:** READY

## Objective

Add automatic dependency tracking to `/create-issue` skill so it updates issue relationships after creating each issue, before committing.

## Background

**Problem discovered:** While creating issues #043, #044, #045, the skill didn't document that #043 and #045 are related (both improve `/create-issue` skill). This relationship information was lost.

**Why now matters:** When an issue is just created, the AI has fresh context about how it relates to previously created issues. This is the optimal time to capture dependencies.

## Implementation Steps

### Step 1: Add "Update Dependencies" Section

**File:** `.claude/skills/create-issue/SKILL.md`

**Location:** In the new "Discussion and Approval Workflow" section (from Issue #045), update Step 5 to add dependency tracking

**Find Step 5:**
```markdown
### Step 5: Create Issue
Only after user selects an option, create the issue with appropriate detail level.
```

**Replace with:**
```markdown
### Step 5: Create Issue with Dependency Tracking

Only after user selects an option:

1. **Create issue and plan files** (based on selected detail level)
2. **Analyze and update dependencies** (new step - see below)
3. **Commit all changes together**

#### 5.1: Create Issue and Plan Files
[Existing content about Option 1/2/3...]

#### 5.2: Analyze and Update Dependencies

**Before committing**, analyze how the new issue relates to:
- Issues created in this session
- Existing issues in backlog

**Relationship types to identify:**

**Blocks:** New issue must complete before another can start
- Example: "Remove dead code" blocks "Refactor using that code"
- Indicator: Issue description says "must complete X first" or "depends on"
- Format: `#NEW blocks #OLD`

**Blocked by:** Another issue must complete before new issue can start
- Example: "Add feature X" blocked by "Refactor API first"
- Indicator: New issue says "requires" or "needs" another issue
- Format: `#OLD blocks #NEW`

**Influences:** Should coordinate (shared component, might conflict)
- Example: Two issues modifying same class
- Indicator: Overlapping files, related functionality
- Format: `#NEW influences #OLD, #OTHER`

**Related:** Work on same feature/component (can work independently)
- Example: Multiple issues improving same skill or component
- Indicator: Same keywords, same file, same feature area
- Format: `#NEW relates to #OLD, #OTHER`

**Analysis approach:**

1. **Check issue content:**
   - Read the new issue title and description
   - Extract key components/files mentioned
   - Identify the feature area or component

2. **Compare with recent issues:**
   - Check issues created in this session (stored in task metadata)
   - Look for matching components, files, or feature areas
   - Determine relationship type

3. **Compare with backlog issues:**
   - Check if new issue mentions any existing issue numbers
   - Look for related feature areas in backlog
   - Limit to top 5-10 most relevant backlog issues (don't scan all)

4. **Determine relationship type:**
   ```
   If new issue explicitly mentions "depends on" or "requires" → Blocked by
   If new issue says "must complete before" → Blocks
   If same files AND different changes → Influences
   If same component/feature area → Related
   ```

#### 5.3: Update Backlog Quick Dependencies

**File to modify:** `docs/project/backlog.md`

**Location:** "Quick dependencies" section (after line 17)

**Add entry if relationship found:**

```markdown
**Quick dependencies:**
- **#003 blocks #004** ⚠️ - Must complete #003 first
- **#003 influences #005, #007** - Should coordinate
- **#006 relates to #009, #010, #011** - Cache cleanup group
[NEW ENTRY HERE]
- **#046 relates to #043, #045** - All improve /create-issue skill
```

**Format:**
- Blocks: `- **#X blocks #Y** ⚠️ - [Reason]`
- Influences: `- **#X influences #Y, #Z** - [Reason]`
- Related: `- **#X relates to #Y, #Z** - [Reason]`

**Reason should be concise:**
- "Must complete X first"
- "Both modify same component"
- "All improve [feature]"

#### 5.4: Update Issue File Metadata

**Add dependency line to the new issue file:**

**Before:**
```markdown
## Metadata
- **Type:** Skill Improvement
- **Priority:** MEDIUM
- **Created:** 2026-02-03
```

**After:**
```markdown
## Metadata
- **Type:** Skill Improvement
- **Priority:** MEDIUM
- **Created:** 2026-02-03
- **Dependencies:** Related to #043, #045 (all improve /create-issue skill)
```

**Format patterns:**
- Blocks: `Blocks #X (must complete before X)`
- Blocked by: `Issue #X (must complete first)`
- Influences: `Influences #X, #Y (overlapping changes)`
- Related: `Related to #X, #Y (same feature/component)`

#### 5.5: Commit All Together

**Single commit containing:**
- Issue file (with dependency metadata)
- Plan file (if created)
- Backlog.md (issue entry + dependency update)

**Commit message format unchanged:**
```
docs: add issue #046 - Update dependencies in /create-issue
```

---
```

### Step 2: Add Dependency Analysis Examples

**File:** `.claude/skills/create-issue/SKILL.md`

**Location:** After the Step 5 content, add examples subsection

**Content:**

````markdown
#### Examples: Dependency Analysis

**Example 1: Related Issues (Same Component)**

```
Session context:
- Issue #043: Fix /create-issue - prevent implementation
- Issue #045: Add approval workflow to /create-issue
- NEW: Issue #046: Update dependencies in /create-issue

Analysis:
- All three mention "/create-issue skill"
- Same component (skill improvement)
- Can work independently (different aspects)
- Relationship: RELATED

Quick dependencies entry:
- **#046 relates to #043, #045** - All improve /create-issue skill

Issue #046 metadata:
- **Dependencies:** Related to #043, #045 (all improve /create-issue skill)
```

**Example 2: Blocking Relationship**

```
Session context:
- Issue #050: Remove dead helper function foo()
- NEW: Issue #051: Refactor class using helper foo()

Analysis:
- Issue #051 uses foo() which #050 will remove
- Must remove dead code first, then refactor
- Relationship: BLOCKS

Quick dependencies entry:
- **#050 blocks #051** ⚠️ - Must remove dead code before refactoring

Issue #051 metadata:
- **Dependencies:** Issue #050 (must complete first - removes dependency)
```

**Example 3: Influences Relationship**

```
Session context:
- Issue #060: Extract duplicate code in TarFile::add_entry
- NEW: Issue #061: Simplify TarFile constructor initialization

Analysis:
- Both modify TarFile class
- Different methods, but might affect each other
- Should coordinate to avoid merge conflicts
- Relationship: INFLUENCES

Quick dependencies entry:
- **#060 influences #061** - Both modify TarFile, should coordinate

Issue #061 metadata:
- **Dependencies:** Influences #060 (overlapping TarFile changes)
```

**Example 4: No Dependencies**

```
Session context:
- Issue #070: Fix typo in graph.cpp
- NEW: Issue #071: Document pattern in tar_file.cpp

Analysis:
- Completely different files
- Different components
- No overlap
- Relationship: NONE

No quick dependencies entry added
No dependency metadata added
```

---
````

### Step 3: Update Phase 2 Step 4 Option A

**File:** `.claude/skills/create-issue/SKILL.md`

**Location:** In "## Phase 2: Interactive Exploration Loop" → Step 4 → Option A

**Find step 5-7:**
```markdown
  5. Create issue file based on selected detail level
  6. Create plan file if user selected Option 2 or 3
  7. Update backlog
```

**Replace with:**
```markdown
  5. Create issue file based on selected detail level
  6. Create plan file if user selected Option 2 or 3
  7. Analyze dependencies and update:
     - Issue file metadata (add Dependencies line)
     - Backlog.md quick dependencies section (add relationship entry)
     - Backlog.md issue list (already done in previous step)
  8. Update backlog (issue entry - already includes dependency metadata)
```

### Step 4: Add Dependency Tracking to Session State

**File:** `.claude/skills/create-issue/SKILL.md`

**Location:** In "## Task Metadata Schema" section

**Find:**
```markdown
Each task stores metadata:

```json
{
  "topic_type": "exploration" | "subtopic",
  "issue_number": "031",
  "exploration_notes": "Found 3 unused methods...",
  "parent_task": "1",
  "depth": 2
}
```
```

**Update to:**
```markdown
Each task stores metadata:

```json
{
  "topic_type": "exploration" | "subtopic",
  "issue_number": "031",
  "issue_component": "TarFile",  // Component/feature area
  "issue_files": ["tar_file.cpp", "tar_file.hpp"],  // Files affected
  "exploration_notes": "Found 3 unused methods...",
  "parent_task": "1",
  "depth": 2
}
```

**New fields for dependency tracking:**
- `issue_component`: Component or feature area (used to find related issues)
- `issue_files`: List of files affected (used to detect influences)

**When completing task with issue created:**
```javascript
TaskUpdate(taskId, {
  status: "completed",
  metadata: {
    issue_number: "046",
    issue_component: "/create-issue skill",
    issue_files: [".claude/skills/create-issue/SKILL.md"]
  }
})
```

**Use during dependency analysis:**
```javascript
// Get all completed tasks with issues
TaskList → filter by completed + has issue_number

// For each completed task:
//   - Check if issue_component matches
//   - Check if issue_files overlap
//   - Determine relationship type
```
```

### Step 5: Update Example Session Flow

**File:** `.claude/skills/create-issue/SKILL.md`

**Location:** In "## Example Session Flow" → Example 1

**After "Skill: [Creates Issue #031 with detailed plan]", add:**

```markdown
Skill: Analyzing dependencies...
       - Checking issues created in this session: none yet
       - Checking backlog: no overlaps found
       - No dependencies to add

Skill: ✓ Created Issue #031: Remove tar_ball.cpp
       (No dependencies)

[Later in session, after creating Issue #032...]

Skill: Analyzing dependencies...
       - Issue #032: Remove unused TarFile methods
       - Issue #031: Remove tar_ball.cpp
       - Both are code removal in tar-related code
       - Relationship: RELATED

       Updating dependencies:
       - Quick dependencies: #032 relates to #031
       - Issue #032 metadata: Related to #031

Skill: ✓ Created Issue #032: Remove unused TarFile methods
       Dependencies: Related to #031 (both tar code cleanup)
```

## Verification

### Test Case 1: Related Issues
**Scenario:** Create two issues improving same component
```
1. Create Issue A: Fix /create-issue prevent implementation
2. Create Issue B: Add approval workflow to /create-issue
```
**Expected:**
- Quick dependencies: `#B relates to #A`
- Issue B metadata: `Related to #A`

### Test Case 2: Blocking Relationship
**Scenario:** Create issue that depends on another
```
1. Create Issue A: Remove helper function
2. Create Issue B: Refactor using that helper (mentions depends on A)
```
**Expected:**
- Quick dependencies: `#A blocks #B ⚠️`
- Issue B metadata: `Issue #A (must complete first)`

### Test Case 3: No Dependencies
**Scenario:** Create unrelated issues
```
1. Create Issue A: Fix typo in graph.cpp
2. Create Issue B: Document pattern in tar_file.cpp
```
**Expected:**
- No quick dependencies entry
- No dependency metadata

### Test Case 4: Multiple Related Issues
**Scenario:** Create third issue in same feature area
```
1. Issue A: /create-issue improvement
2. Issue B: /create-issue improvement
3. Issue C: /create-issue improvement
```
**Expected:**
- Quick dependencies: `#C relates to #A, #B`
- Issue C metadata: `Related to #A, #B`

## Success Criteria

- [ ] Dependency analysis step added to Step 5 in workflow
- [ ] Relationship types (blocks/blocked-by/influences/related) documented
- [ ] Analysis approach clearly explained
- [ ] Quick dependencies update automated
- [ ] Issue metadata update automated
- [ ] Task metadata schema includes component and files
- [ ] Examples show all relationship types
- [ ] Example session flow demonstrates dependency tracking
- [ ] All updates in single commit

## Files Modified

- `.claude/skills/create-issue/SKILL.md` - Add dependency tracking to workflow

## Notes

**Why track in task metadata:**
Task system provides easy access to all issues created in session. Storing component and files in metadata allows quick comparison without re-reading issue files.

**Why limit backlog scanning:**
Don't scan all 40+ backlog issues - only check recent or explicitly mentioned ones. Focus on session-created issues (where context is freshest).

**Relationship type priority:**
1. Blocks/Blocked-by (strongest - prevents work)
2. Influences (medium - coordination needed)
3. Related (weakest - nice to know)

**Edge case - multiple types:**
If issue both "relates to" and "influences" others, choose the stronger relationship (influences).
