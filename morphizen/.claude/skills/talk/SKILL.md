<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
---
name: talk
description: Fast iteration workflow - discuss issues, document externally, then implement immediately
allowed-tools: [Read, Grep, Glob, Edit, Write, TaskCreate, TaskUpdate, TaskList, TaskGet]
---

# /talk - Fast Issue Creation and Implementation

## Purpose

**Fast iteration workflow** for discussing issues, documenting them externally, then implementing immediately in one session.

**Key Capabilities**:
- **Interactive brainstorming**: Ask questions one-by-one, free discussion
- **Hierarchical topic exploration**: Navigate topic trees like `/create-issue`
- **External documentation**: Create issues/plans in `../notes/` (outside repo)
- **Immediate implementation**: After discussion, directly fix all issues
- **No git operations**: Stay on current branch, no commits during discussion

**Prerequisite**: User wants quick iteration - discuss, document, implement in one go.

## When to Use

Use this skill when:

- **Fast prototyping**: "Let's quickly fix these issues"
- **External notes**: Want to document outside the repo
- **Continuous flow**: Discuss → document → implement without interruption
- **No PR preparation**: Not ready for formal issue tracking in repo
- **Stay on branch**: Want to work on current branch without switching

**Different from `/create-issue`**:
- `/create-issue`: Exploration → formal issues in repo → separate implementation
- `/talk`: Exploration → external notes → immediate implementation

## Workflow Overview

1. **Phase 1: Discussion & Documentation**
   - Interactive exploration (like `/create-issue`)
   - Ask clarifying questions one-by-one
   - Brainstorm and discuss freely
   - Summarize understanding
   - Get approval
   - Create issues/plans in `../notes/`

2. **Phase 2: Implementation**
   - After all issues documented
   - Directly implement all issues
   - Make actual code changes
   - Fix everything in one go

---

## Phase 1: Discussion & Documentation

This phase is **identical to `/create-issue`** in terms of interaction style, but with different output location.

### Step 1: Skip Git Checks

**NO git operations** - skip all branch checks, no switching to main.

Start directly with scope exploration.

### Step 2: Understand Initial Scope

Ask user about exploration scope:
```
What area do you want to explore and fix?

Examples:
- "Let's fix the tar code issues"
- "I want to improve the graph API"
- "Clean up the morphizen-core module"
```

Store the **exploration scope** for context.

### Step 3: Initial Code Exploration

Based on scope, explore relevant code:
- Use Glob to find relevant files
- Use Grep to search for patterns
- Use Read to analyze key files
- Look for: bugs, complexity, inconsistencies, tech debt

### Step 4: Identify Top-Level Topics

From initial exploration, identify 2-5 high-level topics.

**Example**: Exploring "tar code" might reveal:
1. Dead code cleanup
2. TarFile class simplification
3. Documentation updates

### Step 5: Create Top-Level Topic Tasks

For each top-level topic, create a task:

```bash
TaskCreate(
  subject: "Topic 1: Dead code cleanup",
  description: "Explore tar code for dead/unused code that can be removed",
  activeForm: "Exploring dead code"
)
```

**Show the initial tree**:
```
Exploration scope: tar code

Initial topics identified:
  Task 1: Dead code cleanup (pending)
  Task 2: TarFile class simplification (pending)
  Task 3: Documentation updates (pending)

Use /tasks anytime to see the full tree.

Which topic should we explore first? (or describe another topic to add)
```

---

## Interactive Exploration Loop

This is the **core brainstorming phase** - identical interaction to `/create-issue`.

### Navigation Model

**Topic tree structure**:
```
Task 1: Dead code cleanup
  Task 1.1: Remove tar_ball.cpp
  Task 1.2: Remove unused TarFile functions
    Task 1.2.1: Remove public methods
    Task 1.2.2: Remove private helpers
  Task 1.3: Remove unused includes
Task 2: TarFile class simplification
Task 3: Documentation updates
```

**User navigation commands** (understand natural language):
- "Focus on task 1" / "Let's explore topic 1"
- "Drill into 1.2" / "Tell me more about 1.2"
- "Go back" / "Back to parent" / "Up one level"
- "Next topic" / "Skip this"
- "Show me the tree" / "/tasks"
- "Let's work on 2"

### Loop Structure

```
WHILE user wants to continue exploration:
  1. Get current topic context
  2. Explore code/docs for this topic
  3. Present findings and suggestions
  4. User responds (create issue, drill deeper, skip, etc.)
  5. Update task status based on action
  6. Navigate to next topic based on user input
```

### Step-by-Step: Exploring a Topic

#### Step 1: Mark Topic as In Progress

When user focuses on a topic:
```bash
TaskUpdate(taskId: "current_topic", status: "in_progress")
```

#### Step 2: Deep Code Exploration

**For current topic**, explore thoroughly:

**Example: Topic "Remove tar_ball.cpp"**
```bash
# Find the files
Glob pattern="**/tar_ball.*"

# Search for references
Grep pattern="tar_ball" output_mode="files_with_matches"

# Read the file
Read tar_ball.cpp

# Check includes
Grep pattern="#include.*tar_ball" output_mode="content"

# Check CMakeLists
Grep pattern="tar_ball" path="CMakeLists.txt"
```

**Analyze findings**:
- Is it referenced anywhere?
- Is it in build system?
- Does it have tests?
- What's the risk of removing it?

#### Step 3: Present Findings and Start Discussion

Share findings naturally:
```
I explored tar_ball.cpp/hpp:

Findings:
- 250 lines of code
- No references in codebase (grep found nothing)
- Not in any CMakeLists.txt
- No tests reference it
- Last modified 2 years ago

Looks like dead code. This seems like a candidate for an issue.

Before documenting it, I have 3 questions:
1. What exactly should we do with this file?
2. Why does this matter / what's the impact?
3. Should we implement this immediately after discussion?

I will ask one by one. Let's start with:

**Question 1:** What exactly should we do with this file?
```

**Pattern:**
1. List all clarifying questions upfront (2-4 questions)
2. Tell user you'll ask one-by-one
3. Ask each question and wait for answer
4. Move to next question only after receiving answer

**Common questions to ask:**
- What is this? (Ensure understanding of the problem)
- Why does this matter? (Understand impact/priority)
- What's the solution? (Clarify approach)
- What are the deliverables? (What files/changes)
- Any edge cases or considerations? (Completeness check)

#### Step 4: Summarize Understanding

After discussion, present your understanding back to the user:

```
Let me summarize what I understood about this issue:

**Problem:**
[2-3 sentence description of the problem]

**Why it matters:**
[Impact, why this needs fixing]

**Solution:**
[High-level approach]

**Files affected:**
- path/to/file1.cpp (lines X-Y)
- path/to/file2.hpp (lines A-B)

**Implementation complexity:**
[Simple/Medium/Complex - brief justification]

Does this match what you're thinking?
```

**Wait for user confirmation.** If user corrects something, update your understanding and re-summarize.

#### Step 5: Get Approval with Plan Options

Only after user confirms your understanding, ask how to document:

```
How should I document this issue?

1. Brief issue only (no plan)
   - Problem description + high-level solution
   - Use for: Trivial fixes (1-line changes, typos, obvious fixes)

2. Issue with simple plan
   - Problem + solution + basic implementation steps
   - Use for: Straightforward changes (single file, clear approach)

3. Issue with detailed plan (Recommended)
   - Complete implementation guide with full context
   - Use for: Complex changes, multiple files, design decisions
   - Contains enough info for implementation

4. Skip this issue
   - Not worth documenting right now

I recommend: **Option [X]** because [specific reason based on complexity]

Which option do you prefer?
```

**Recommendation logic:**

**Choose Option 1 (Brief)** when:
- Single line change (e.g., remove duplicate declaration)
- Obvious fix with no decisions needed
- Trivial typo or formatting fix

**Choose Option 2 (Simple plan)** when:
- Single file modification
- Straightforward refactoring (e.g., use standard idiom)
- Clear approach, minimal context needed

**Choose Option 3 (Detailed plan)** when:
- Multiple files affected
- Design decisions involved
- Requires understanding context/rationale
- Complex refactoring or architectural change
- **Default: When in doubt, choose Option 3**

**Choose Option 4 (Skip)** when:
- User explicitly says not worth documenting
- Issue is too vague or needs more investigation
- Duplicate of existing issue

#### Step 6: Create Issue and Plan in `../notes/`

**Directory structure:**
```
../notes/
  issues/
    001-remove-tar-ball.md
    002-simplify-tarfile.md
  plans/
    001-remove-tar-ball-plan.md
    002-simplify-tarfile-plan.md
```

**Get next issue number:**
```bash
# Check if ../notes/issues/ exists, create if not
mkdir -p ../notes/issues ../notes/plans

# Get highest issue number
HIGHEST=$(ls ../notes/issues/ 2>/dev/null | grep -E '^[0-9]+-' | sed 's/-.*//' | sort -n | tail -1)
NEXT_ISSUE=$((HIGHEST + 1))
```

**Create issue file:**
```markdown
# Issue #001: Remove tar_ball.cpp

## Problem

[Description based on exploration]

## Why It Matters

[Impact based on discussion]

## Solution

[Approach based on discussion]

## Files Affected

- path/to/file1.cpp (lines X-Y)
- path/to/file2.hpp (lines A-B)

## Metadata

- **Type:** [Bug/Feature/Refactoring/Documentation]
- **Priority:** [HIGH/MEDIUM/LOW]
- **Complexity:** [Simple/Medium/Complex]
- **Created:** YYYY-MM-DD
```

**Create plan file (if Option 2 or 3 selected):**

**For Option 2 (Simple plan):**
```markdown
# Plan: Issue #001 - Remove tar_ball.cpp

## Implementation Steps

1. [Step 1]
2. [Step 2]
3. [Step 3]

## Files to Modify

- path/to/file1.cpp
- path/to/file2.hpp

## Verification

```bash
[verification command]
```
```

**For Option 3 (Detailed plan):**
```markdown
# Plan: Issue #001 - Remove tar_ball.cpp

## Context

[Full background and rationale]

## Step-by-Step Implementation

### Step 1: [Action]

**File:** path/to/file.cpp (lines X-Y)

**Before:**
```cpp
[code before]
```

**After:**
```cpp
[code after]
```

**Rationale:** [why this change]

### Step 2: [Action]

[Continue for all steps...]

## Verification Steps

1. [Verification step 1]
2. [Verification step 2]

## Success Criteria

- [ ] [Criterion 1]
- [ ] [Criterion 2]
```

**Report creation:**
```
✓ Created issue #001 in ../notes/issues/001-remove-tar-ball.md
✓ Created plan in ../notes/plans/001-remove-tar-ball-plan.md

Continue exploring or ready to implement?
```

#### Step 7: Continue or Finish Discussion

After creating issue/plan, ask:
```
Current status:
  Task 1: Dead code cleanup
    Task 1.1: Remove tar_ball.cpp → Issue #001 ✓
    Task 1.2: Remove unused TarFile functions (pending)
  Task 2: TarFile class simplification (pending)

What's next?
- Continue exploring (drill into 1.2, move to 2, add new topic)
- Ready to implement all documented issues
- Done (wrap up)
```

### Sub-topic Breakdown Workflow

**When exploration reveals a topic is too complex for a single issue, break it down:**

**Indicators that a topic is too complex:**
- Multiple distinct responsibilities identified
- Would require changing 5+ files
- Architectural refactoring needed
- Estimated 500+ lines of code changes
- Mix of high complexity and low complexity changes
- Topic naturally splits into independent sub-problems

**Breakdown steps:**

1. **Identify complexity** during exploration
2. **Identify sub-topics** (2-5 concrete, actionable sub-topics)
3. **Summarize breakdown** to user
4. **Get approval** from user
5. **Create tasks automatically** for all sub-topics
6. **Ask which sub-topic first**
7. Each sub-topic goes through **normal discussion workflow**

---

## Phase 2: Implementation

**After all issues documented**, implement them immediately.

### Step 1: Confirm Implementation

When user says "ready to implement" or "let's implement":

```
I documented N issues:
  #001: Remove tar_ball.cpp
  #002: Simplify TarFile API
  #003: Update documentation

Ready to implement all of them now?

Or select specific issues to implement (e.g., "implement #001 and #003")?
```

### Step 2: Implementation Approach

**For each issue to implement:**

1. **Read the plan** from `../notes/plans/NNN-name-plan.md`
2. **Follow the plan step-by-step**
3. **Make code changes** using Edit/Write tools
4. **Verify changes** per plan verification steps

**Example:**

```
Implementing Issue #001: Remove tar_ball.cpp

Step 1: Remove tar_ball.cpp and tar_ball.hpp
[Use Edit/Write to remove files or remove references]

Step 2: Remove from CMakeLists.txt
[Edit CMakeLists.txt]

Step 3: Remove test references
[Edit test files]

Verification: Build and run tests
[Use Bash to build and test]

✓ Issue #001 implemented successfully
```

### Step 3: Implementation Status

After each issue:

```
Implementation status:
  ✓ Issue #001: Remove tar_ball.cpp (implemented)
  ⊙ Issue #002: Simplify TarFile API (in progress)
  ⊘ Issue #003: Update documentation (pending)

Continue with #002?
```

### Step 4: Wrap Up

When all issues implemented:

```
All issues implemented!

Summary:
  ✓ #001: Remove tar_ball.cpp
  ✓ #002: Simplify TarFile API
  ✓ #003: Update documentation

Changes made to:
  - morphizen-core/src/tar_ball.cpp (removed)
  - morphizen-core/src/tar_ball.hpp (removed)
  - morphizen-core/src/tar_file.cpp (simplified)
  - docs/api/tar.md (updated)

Next steps:
  - Test changes manually if needed
  - Commit changes when ready
  - Issues and plans saved in ../notes/ for reference
```

---

## Key Differences from `/create-issue`

**Similarities (keep the good parts):**
- ✅ Ask questions one-by-one
- ✅ Brainstorming and free discussion
- ✅ Hierarchical topic exploration
- ✅ Task system for progress tracking
- ✅ Discussion → Summarize → Approve workflow
- ✅ Same markdown format for issues/plans

**Differences:**

| Feature | `/create-issue` | `/talk` |
|---------|----------------|---------|
| Git checks | ✅ Check branch, require main | ❌ Skip all git checks |
| Branch switching | ✅ Switch to main | ❌ Stay on current branch |
| Output location | `docs/project/` | `../notes/` |
| Backlog updates | ✅ Update backlog.md | ❌ No backlog updates |
| Git commits | ✅ Commit issue docs | ❌ No commits during discussion |
| PR creation | ✅ Create draft PR | ❌ No PR |
| Implementation | ❌ Separate phase | ✅ Immediate after discussion |
| Dependency tracking | ✅ Track in backlog | ❌ External notes only |

---

## Task Metadata Schema

Each task stores metadata (same as `/create-issue`):

```json
{
  "topic_type": "exploration" | "subtopic",
  "issue_number": "001",  // If issue created
  "issue_component": "TarFile",  // Component/feature area
  "issue_files": ["tar_file.cpp", "tar_file.hpp"],  // Files affected
  "exploration_notes": "Found 3 unused methods...",
  "parent_task": "1",  // For hierarchical structure
  "implemented": true  // NEW: Track if implemented
}
```

**New field:**
- `implemented`: `true` if issue was implemented in Phase 2

---

## Navigation Commands (Natural Language)

User can say:
- "Focus on task 1" / "Let's work on 1" / "Start with 1"
- "Drill into 1.2" / "Explore 1.2 deeper" / "Break down 1.2"
- "Go back" / "Back to parent" / "Up one level"
- "Next" / "Move to next topic" / "Skip this"
- "Show the tree" / "/tasks"
- "Add a new topic: [description]"
- "Ready to implement" / "Let's implement" / "Implement all"
- "Implement #001" / "Just implement #001 and #003"

---

## Example Session Flow

```
User: "/talk - fix tar code issues"

Skill: "What area do you want to explore and fix?

       Based on your request, I'll explore the tar code.

       [Explores tar code...]

       Initial topics identified:
         Task 1: Dead code cleanup (pending)
         Task 2: TarFile class simplification (pending)

       Which topic should we explore first?"

User: "Let's start with task 1"

Skill: [Marks Task 1 in_progress, explores dead code]

       "I found tar_ball.cpp/hpp - no references anywhere.

       Before documenting this, I have 3 questions:
       1. What exactly should we do with this file?
       2. Why does this matter?
       3. Any concerns about removing it?

       I will ask one by one. Let's start with:

       **Question 1:** What exactly should we do with this file?"

User: "Remove it completely"

Skill: "**Question 2:** Why does this matter?"

User: "It's dead code, clutters the codebase"

Skill: "**Question 3:** Any concerns about removing it?"

User: "No concerns, it's clearly unused"

Skill: "Let me summarize what I understood:

       **Problem:** tar_ball.cpp/hpp is dead code (no references)
       **Why it matters:** Clutters codebase
       **Solution:** Remove completely
       **Files affected:** tar_ball.cpp, tar_ball.hpp
       **Complexity:** Simple

       Does this match what you're thinking?"

User: "Yes"

Skill: "How should I document this issue?

       1. Brief issue only (no plan)
       2. Issue with simple plan
       3. Issue with detailed plan (Recommended)
       4. Skip this issue

       I recommend: **Option 2** because it's straightforward deletion

       Which option do you prefer?"

User: "Option 2"

Skill: [Creates ../notes/issues/001-remove-tar-ball.md]
       [Creates ../notes/plans/001-remove-tar-ball-plan.md]

       "✓ Created issue #001 in ../notes/issues/001-remove-tar-ball.md
       ✓ Created plan in ../notes/plans/001-remove-tar-ball-plan.md

       Continue exploring or ready to implement?"

User: "Continue, let's look at task 2"

Skill: [Explores TarFile class simplification...]
       [Similar discussion workflow...]
       [Creates issue #002]

       "✓ Created issue #002

       Ready to implement or continue exploring?"

User: "Ready to implement all"

Skill: "I documented 2 issues:
         #001: Remove tar_ball.cpp
         #002: Simplify TarFile API

       Ready to implement all of them now?"

User: "Yes"

Skill: [Reads plan #001]
       [Makes code changes]
       [Verifies]

       "✓ Issue #001 implemented successfully

       Implementing Issue #002..."

       [Implements #002]

       "All issues implemented!

       Summary:
         ✓ #001: Remove tar_ball.cpp
         ✓ #002: Simplify TarFile API

       Changes made to:
         - morphizen-core/src/tar_ball.cpp (removed)
         - morphizen-core/src/tar_file.cpp (simplified)

       Next steps:
         - Test changes
         - Commit when ready
         - Issues/plans saved in ../notes/ for reference"
```

---

## Error Handling

### Notes Directory Creation

If `../notes/` doesn't exist:
```bash
mkdir -p ../notes/issues ../notes/plans
```

No error - just create it.

### No Topics Found

```
"I explored [scope] but didn't find obvious issues.

Would you like to:
- Explore a different area
- Describe specific concerns to investigate
- Cancel
```

### User Cancels Mid-Session

```
User: "Actually, let's cancel"

Skill: "No problem!

       Created N issues so far in ../notes/:
         #001: Remove tar_ball.cpp
         #002: Simplify TarFile API

       No implementation done yet. Issues saved for later reference."
```

### Implementation Fails

If implementation of an issue fails (e.g., tests fail):

```
"⚠️ Issue #001 implementation encountered an error:
   [error details]

   Options:
   - Skip this issue for now
   - Try a different approach
   - Stop implementation

   What would you like to do?"
```

---

## Notes

**Design Principles**:
- **Fast iteration**: Discuss → Document → Implement in one flow
- **Interactive**: Natural conversation, ask questions one-by-one
- **External tracking**: Use `../notes/` for personal issue tracking
- **No git friction**: Stay on current branch, no branch switching
- **Flexible**: Can create 0 to N issues, implement all or selectively

**When to Use `/talk` vs `/create-issue`**:
- **Use `/talk`**: Fast prototyping, external notes, immediate fixes
- **Use `/create-issue`**: Formal issue tracking, PR preparation, team collaboration

**Output Location**:
- Issues: `../notes/issues/NNN-name.md`
- Plans: `../notes/plans/NNN-name-plan.md`
- No changes to repo during discussion phase
- Code changes only during implementation phase
