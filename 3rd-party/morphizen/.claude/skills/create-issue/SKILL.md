<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
---
name: create-issue
description: Interactive exploration and brainstorming to create issues with hierarchical topic navigation
allowed-tools: [Bash, Read, Grep, Glob, Edit, Write, TaskCreate, TaskUpdate, TaskList, TaskGet]
---

# /create-issue - Interactive Issue Creation Through Exploration

## Purpose

**Brainstorming-style workflow** for exploring codebases and creating issues incrementally through natural conversation.

**Key Capabilities**:
- **Hierarchical topic exploration**: Navigate topic trees (Topic 1 → 1.1 → 1.1.1)
- **Incremental issue creation**: Create issues one-at-a-time during exploration
- **Code exploration**: Read/analyze code like plan mode to find insights
- **Visual progress tracking**: Use task system to show exploration tree
- **Flexible outcomes**: Create 0 to N issues based on discoveries

**Prerequisite**: User wants to explore a codebase area and document findings as issues.

## When to Use

Use this skill when:

- **Exploratory work**: "Let's explore the tar code and see what needs cleaning up"
- **Brainstorming**: "I want to brainstorm improvements to the graph API"
- **Discovery-driven**: Don't know exactly what issues to create yet, will discover during exploration
- **Hierarchical planning**: Big topic that needs to be broken down into sub-topics
- **Interactive session**: Want to create issues incrementally during discussion

**Single issue mode**: Can still create just one issue if that's all you find.

**Multi-issue mode**: Discover and create multiple issues during exploration.

## Workflow Overview

1. **Initialize Session**: Create top-level exploration tasks, check prerequisites
2. **Interactive Exploration Loop**:
   - Navigate topic tree (drill down, go back, skip)
   - Explore code/docs for current topic
   - Suggest issues based on findings
   - Create issues immediately when user agrees
   - Mark topics complete and continue
3. **Wrap Up**: Show summary of all created issues

---

## CRITICAL: Issue Creation vs Implementation

**The `/create-issue` skill creates documentation about work, NOT the work itself.**

### During Issue Creation - DO THIS:

✅ **Create issue documentation:**
- Issue file (`docs/project/issues/NNN-name.md`) describing the problem and solution
- Plan file (`docs/project/plans/NNN-name-plan.md`) describing HOW to implement
- Update backlog.md with issue entry
- Commit ONLY the documentation files

✅ **Explore code thoroughly:**
- Read files with Read tool
- Search with Grep/Glob
- Analyze patterns and problems
- Document findings in issue/plan

### During Issue Creation - DO NOT DO THIS:

❌ **Do NOT implement the work described in the issue:**
- Do NOT create deliverable files (technical docs, config files, etc.)
- Do NOT modify source code
- Do NOT make any changes beyond issue documentation
- Do NOT commit implementation artifacts

### Why This Matters

**Wrong approach (what NOT to do):**
```
User: "yes we need to add some comment and tech doc"
Skill: [Creates docs/technical/pattern.md]        ❌ WRONG - this is implementation
Skill: [Updates morphizen-core/src/file.hpp]      ❌ WRONG - this is implementation
Skill: [Commits everything]                       ❌ WRONG - mixed issue + implementation
```

**Right approach (what TO do):**
```
User: "yes we need to add some comment and tech doc"
Skill: [Creates docs/project/issues/042-doc-pattern.md describing WHAT needs to be done]  ✅ CORRECT
Skill: [Creates docs/project/plans/042-doc-pattern-plan.md describing HOW to do it]      ✅ CORRECT
Skill: [Updates backlog.md]                                                               ✅ CORRECT
Skill: [Commits only the issue documentation]                                             ✅ CORRECT
Skill: [Does NOT create docs/technical/pattern.md]                                        ✅ CORRECT
Skill: [Does NOT modify source code]                                                      ✅ CORRECT
```

Later, when someone (or you) implements the issue, THEN create the tech doc and modify code.

### Plan File Content

The plan file should describe the implementation steps, including:
- What files to create (with example content structure)
- What code to modify (with before/after examples)
- Verification steps
- Success criteria

But **do not actually create those files or make those changes** during issue creation.

### Exception: Documentation-Only Issues

If the issue is specifically about creating documentation AND the user explicitly confirms "create the doc now during issue creation", then it's acceptable to create the documentation as part of the issue.

**Ask for confirmation:**
```
This issue is about creating documentation. Should I:
- A) Just create the issue/plan (normal workflow)
- B) Create the issue/plan AND the actual documentation file (exception)
```

By default, assume **option A** unless user explicitly chooses B.

---

## Discussion and Approval Workflow

**Before creating any issue, follow this 5-step workflow:**

### Step 1: Explore (Existing)
Use Read, Grep, Glob to find and analyze the problem. This is your investigation phase.

### Step 2: Discuss
**DO NOT immediately create an issue after exploration.** Instead, discuss with the user.

**Pattern:**
1. List all clarifying questions upfront (2-4 questions)
2. Tell user you'll ask one-by-one
3. Ask each question and wait for answer
4. Move to next question only after receiving answer

**Example:**
```
I found [problem]. Before documenting this, I have 3 questions:
1. What exactly is this issue?
2. Why does this matter?
3. What should the solution approach be?

I will ask one by one. Let's start with:

**Question 1:** What exactly is this issue?

Is it:
- A) [Option]
- B) [Option]
- C) [Option]
```

**Common questions to ask:**
- What is this? (Ensure understanding of the problem)
- Why does it matter? (Understand impact/priority)
- What's the solution? (Clarify approach)
- What are the deliverables? (What files/changes)
- Any edge cases or considerations? (Completeness check)

### Step 3: Summarize
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

### Step 4: Get Approval with Plan Options
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
   - Contains enough info for fresh Claude session to implement

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

### Step 5: Create Issue with Dependency Tracking

Only after user selects an option:

1. **Create issue and plan files** (based on selected detail level)
2. **Analyze and update dependencies** (see below)
3. **Commit all changes together**

#### 5.1: Create Issue and Plan Files

**For Option 1 (Brief):**
- Create issue file only
- No plan file needed
- 1-2 paragraphs describing problem
- 1 paragraph describing solution
- List of affected files

**For Option 2 (Simple plan):**
- Create issue file
- Create simple plan file with:
  - Basic implementation steps (5-10 bullet points)
  - File locations
  - Simple before/after examples
  - Verification command

**For Option 3 (Detailed plan):**
- Create issue file
- Create detailed plan file with:
  - Complete step-by-step guide
  - Exact file paths and line numbers
  - Detailed before/after code examples
  - Multiple verification steps
  - Success criteria checklist
  - Full context for fresh Claude session

**Never skip steps 2-4.** Always discuss, summarize, and get approval before creating.

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

**Location:** "Quick dependencies" section

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

**Commit message format:**
```
docs: add issue #046 - Update dependencies in /create-issue
```

---

## Sub-topic Breakdown Workflow

**When exploration reveals a topic is too complex for a single issue, use this workflow to break it down:**

### When to Break Down a Topic

**Indicators that a topic is too complex:**
- Multiple distinct responsibilities identified
- Would require changing 5+ files
- Architectural refactoring needed
- Estimated 500+ lines of code changes
- Mix of high complexity and low complexity changes
- Topic naturally splits into independent sub-problems

**When NOT to break down:**
- Single clear problem to solve
- Changes localized to 1-3 files
- Straightforward refactoring
- All changes at similar complexity level

### Breakdown Workflow Steps

**Step 1: Explore and Identify Complexity**

During exploration, recognize complexity indicators:
```
I explored TarFile class:

Findings:
- 676 lines total (218 header + 458 implementation)
- 7 factory methods with overlapping logic
- Manages streams, entries, I/O, and serialization
- Multiple responsibilities mixed together

This is too complex for a single issue.
```

**Step 2: Identify Sub-topics**

Break down into concrete, actionable sub-topics:

**Criteria for good sub-topics:**
- Each addresses a single responsibility
- Can be implemented independently
- Has clear scope and deliverables
- Estimated as Simple/Medium/High complexity

**Format:**
```
I've identified 5 sub-topics:

**Sub-topic 5.2.1: Factory Method Proliferation**
- Problem: 7 different create() factory methods with overlapping logic
- Impact: Hard to understand which factory to use
- Complexity: Medium
- Files: tar_file.hpp, tar_file.cpp

**Sub-topic 5.2.2: Stream Management Responsibility**
- Problem: TarFile manages both iostream and mmap-specific MemStream
- Impact: Mixed concerns, harder to test
- Complexity: High
- Files: tar_file.hpp, tar_file.cpp

[Continue for all sub-topics...]
```

**Step 3: Summarize Breakdown**

Present the full breakdown to user:
```
Let me summarize the breakdown:

**Parent Topic:** TarFile God Class Pattern

**Sub-topics identified:** 5

1. Factory Method Proliferation (Medium)
2. Stream Management Responsibility (High)
3. Entry Management Coupling (Medium-High)
4. Serialization Responsibility (Low-Medium)
5. Public API Surface (Low)

Each sub-topic can be addressed independently as a separate issue.

Does this breakdown make sense?
```

**Wait for user confirmation.**

**Step 4: Get Approval**

User reviews and approves (or suggests changes):
```
User: "Yes, good breakdown"
```

**Step 5: Create Tasks Automatically**

**CRITICAL:** After user approves, immediately create tasks for ALL sub-topics:

```javascript
// For each sub-topic:
TaskCreate({
  subject: "Sub-topic X.Y.Z: [Title]",
  description: "[Problem description, complexity, files]",
  activeForm: "Analyzing [topic name]",
  metadata: {
    parent_task: "[parent_task_id]",
    topic_type: "subtopic"
  }
})
```

**Minimal metadata fields:**
- `parent_task`: ID of parent task (links sub-topic to parent)
- `topic_type`: Always "subtopic" (for filtering)

**Report task creation:**
```
Created tasks for sub-topics:
- Task #10: Sub-topic 5.2.1: Factory Method Proliferation
- Task #11: Sub-topic 5.2.2: Stream Management Responsibility
- Task #12: Sub-topic 5.2.3: Entry Management Coupling
- Task #13: Sub-topic 5.2.4: Serialization Responsibility
- Task #14: Sub-topic 5.2.5: Public API Surface
```

**Step 6: Ask Which Sub-topic First**

Prompt user to select:
```
Which sub-topic would you like to discuss first?
```

User selects a sub-topic, then that sub-topic goes through the **normal 5-step workflow**:
1. Explore (the specific sub-topic in detail)
2. Discuss (ask questions one-by-one)
3. Summarize (confirm understanding)
4. Get Approval (offer plan options: brief/simple/detailed)
5. Create Issue (document that sub-topic)

### Integration with Normal Workflow

**Two distinct workflows:**

**A) Sub-topic Breakdown Workflow** (this section)
- Used when: Topic too complex
- Creates: Tasks (one per sub-topic)
- Outcome: User selects which sub-topic to start with

**B) Normal 5-step Workflow** (existing, from Discussion and Approval Workflow)
- Used when: Topic has clear scope OR working on a sub-topic
- Creates: Issues (one issue with optional plan)
- Outcome: Documented issue ready for implementation

**Flow:**
```
Explore Topic → Too complex?
  YES → Sub-topic Breakdown Workflow
        → Create tasks
        → User selects sub-topic
        → Normal 5-step Workflow for that sub-topic
        → Creates Issue #X for sub-topic
  NO  → Normal 5-step Workflow
        → Creates Issue #Y for topic
```

### Decision Tree: Break Down or Single Issue?

Use this decision tree during exploration:

```
Is the topic complex?
├─ YES: Check indicators
│   ├─ Multiple responsibilities? (God Class, Mixed Concerns)
│   ├─ 5+ files affected?
│   ├─ Architectural refactoring?
│   ├─ 500+ lines of changes?
│   └─ Mix of High/Low complexity?
│       ├─ YES (2+ indicators) → Use Sub-topic Breakdown Workflow
│       └─ NO (0-1 indicators) → Use Normal 5-step Workflow
└─ NO: Single clear problem
    └─ Use Normal 5-step Workflow
```

**Examples:**

**Needs breakdown:**
- "TarFile God Class" - Multiple responsibilities, 676 lines, 7 factory methods
- "Refactor configuration system" - Architectural change, many files
- "Clean up authentication module" - Mixed concerns, high+low complexity tasks

**Single issue:**
- "Extract duplicate tmpfile code" - 4 locations, clear pattern, ~20 lines
- "Document PrivateTag pattern" - Single responsibility, 1-2 files
- "Fix erase-remove idiom" - Straightforward refactoring, 2 locations

---

## Phase 0: Pre-Selection Sync

Ensure on main branch with no uncommitted changes, then sync local main with origin/main via `git pull origin main`.

---

## Phase 1: Initialize Exploration Session

### Step 1: Check Prerequisites

**Check current branch**:
```bash
git branch --show-current
```

**If not on main**:
- Ask: "You're on branch `<branch>`. Switch to main first? (y/n)"
- If yes: `git checkout main && git pull origin main`
- If no: Exit

**Check uncommitted changes**:
```bash
git status --porcelain
```

**If uncommitted changes exist**:
- Display: "❌ You have uncommitted changes. Commit or stash them first."
- List files
- Exit

### Step 2: Understand Initial Scope

Ask user about exploration scope:
```
What area do you want to explore?

Examples:
- "Let's explore the tar code"
- "I want to brainstorm improvements to the graph API"
- "Clean up the morphizen-core module"
```

Store the **exploration scope** for context.

### Step 3: Initial Code Exploration

Based on scope, explore relevant code:
- Use Glob to find relevant files
- Use Grep to search for patterns
- Use Read to analyze key files
- Look for: dead code, complexity, inconsistencies, tech debt

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

## Phase 2: Interactive Exploration Loop

This is the **core brainstorming phase** - iterate through topics, drill down hierarchically, create issues incrementally.

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

#### Step 3: Present Findings and Suggestions

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

Before documenting it, let me ask some questions to ensure I understand correctly...
```

**DO NOT immediately ask "Should we create an issue?"**

Instead, transition to discussion phase (Step 2 of Discussion and Approval Workflow).

**Reminder: Check Complexity**

After presenting findings, evaluate:
- Is this too complex for a single issue?
- Check decision tree in "Sub-topic Breakdown Workflow"
- If YES → Use Sub-topic Breakdown Workflow
- If NO → Continue with normal workflow (ask questions, summarize, approve, create)

#### Step 4: Handle User Response

**User can respond in multiple ways:**

**Option A: User wants to create issue**
```
User: "Yes, let's create an issue for this"

You:
  1. **STOP - Do not create issue yet**
  2. Follow the 5-step approval workflow:
     - Step 2: Discuss (ask questions one-by-one)
     - Step 3: Summarize understanding
     - Step 4: Get approval with plan options
     - Step 5: Create issue (only after user selects option)
  3. Find next issue number
  4. If first issue: Create branch (feature/exploration-session-XXX)
  5. Create issue file based on selected detail level
  6. Create plan file if user selected Option 2 or 3
  7. Analyze dependencies and update:
     - Issue file metadata (add Dependencies line)
     - Backlog.md quick dependencies section (add relationship entry)
     - Backlog.md issue list (add issue entry)
  8. ❌ DO NOT implement anything (see "Issue Creation vs Implementation")
  9. Commit ONLY issue documentation
  10. Push to fork
  11. If first issue: Create draft PR
  12. Update task metadata: issue_number, issue_component, issue_files
  13. Mark task as completed

**Critical:** Steps 2-4 (Discuss → Summarize → Get Approval) are MANDATORY.
Never skip directly from exploration to issue creation.
```

**Option B: Drill deeper (create subtasks)**
```
User: "Let's break this down - separate issue for .cpp and .hpp?"

You: "Sure! Creating subtasks:"
  TaskCreate(subject: "Remove tar_ball.cpp", ...)  → Task 1.1
  TaskCreate(subject: "Remove tar_ball.hpp", ...)  → Task 1.2

  "Which one to explore first?"
```

**Option C: Skip for now**
```
User: "Skip this, let's look at topic 2"

You:
  TaskUpdate(current_task, status: "pending")  # Mark as pending
  TaskUpdate(topic_2, status: "in_progress")   # Move to topic 2
  [Continue exploration loop with topic 2]
```

**Option D: Not needed**
```
User: "Actually, this might be used by external code. Skip it."

You:
  TaskUpdate(current_task, status: "completed")  # Mark done (no issue)

  "Got it. Moving on..."
```

**Option E: Add new topic**
```
User: "Good point, but I also just realized we should check for unused includes"

You:
  TaskCreate(subject: "Remove unused includes", ...)  → New task

  "Added! Now exploring 'Remove unused includes'..."
```

#### Step 5: Navigate and Continue

After handling response, ask what's next:
```
Current status (use /tasks to see full tree):
  Task 1: Dead code cleanup
    Task 1.1: Remove tar_ball.cpp → Issue #031 ✓
    Task 1.2: Remove unused TarFile functions (pending)
  Task 2: TarFile class simplification (pending)

What's next? (drill into 1.2, move to 2, add new topic, or done)
```

---

## Phase 3: Git Workflow for Incremental Issue Creation

### Strategy: One Branch for All Issues

**First issue**:
```bash
# Get next issue number
HIGHEST=$(ls docs/project/issues/ | grep -E '^[0-9]+-' | sed 's/-.*//' | sort -n | tail -1)
FIRST_ISSUE=$((HIGHEST + 1))

# Create branch
git checkout -b feature/exploration-session-${FIRST_ISSUE}

# Create issue files, commit, push
git push -u fork feature/exploration-session-${FIRST_ISSUE}

# Create draft PR
gh pr create --draft --title "docs: exploration session - issues #${FIRST_ISSUE}+" --body "..."

# Store PR number for future commits
```

**Subsequent issues** (same session):
```bash
# Create issue files on same branch
# Commit
# Push to same branch (updates PR automatically)
git push fork feature/exploration-session-${FIRST_ISSUE}
```

### Issue File Creation (Same as Before)

For each issue:

1. **Generate issue number and filename**
2. **Create plan file**: Extract from exploration findings
3. **Create issue file**: Use exploration findings for problem/solution
4. **Update backlog**: Add entry
5. **Commit**: One commit per issue
6. **Push**: Updates the PR

---

## Phase 4: Wrap Up Session

When user is done exploring (says "done", "that's all", etc.):

### Step 1: Show Task Tree Summary

```bash
TaskList
```

Display summary:
```
Exploration session complete!

Topic tree:
  ✓ Task 1: Dead code cleanup
    ✓ Task 1.1: Remove tar_ball.cpp → Issue #031
    ✓ Task 1.2: Remove unused TarFile functions → Issue #032
  ✓ Task 2: TarFile class simplification → Issue #033
  ⊘ Task 3: Documentation updates (skipped)

Created 3 issues total: #031, #032, #033
```

### Step 2: Show PR Information

```
Branch: feature/exploration-session-031
PR: https://github.com/ROCm/MorphiZen/pull/99 (draft)

Commits:
- docs: add issue #031 - Remove tar_ball.cpp
- docs: add issue #032 - Remove unused TarFile functions
- docs: add issue #033 - Simplify TarFile class

Next steps:
- Review issues in PR
- Run pre-commit if needed
- Mark PR ready when satisfied
```

### Step 3: Optionally Clean Up Tasks

Ask user:
```
Keep the task tree for reference, or clean it up?
```

If clean up: Mark all tasks as completed or delete them.

---

## Example Session Flow

### Example 1: Simple Exploration (2 Issues)

```
User: "/create-issue - explore tar code"

Skill: [Checks prerequisites, explores tar code]

Skill: "Exploration scope: tar code

       Initial topics identified:
         Task 1: Dead code cleanup (pending)
         Task 2: Code quality improvements (pending)

       Which topic should we explore first?"

User: "Let's start with task 1"

Skill: [Marks Task 1 in_progress, explores dead code]

Skill: "I found tar_ball.cpp/hpp - no references anywhere.
       Should we create an issue to remove it?"

User: "Yes, make it #031"

Skill: [Creates branch, issue #031, commits, pushes, creates PR]
       "✓ Created issue #031: Remove tar_ball.cpp

       Continue exploring dead code or move to task 2?"

User: "Are there other dead files?"

Skill: [Explores more]
       "I also found old_archive.cpp - deprecated since v2.0. Remove this too?"

User: "Yes, #032"

Skill: [Creates issue #032, commits, pushes]
       "✓ Created issue #032

       That's 2 issues so far. Continue?"

User: "No, that's enough"

Skill: "Exploration complete!

       Created 2 issues: #031, #032
       PR: https://github.com/.../pull/99 (draft)"
```

### Example 2: Hierarchical Exploration (Drilling Down)

```
User: "/create-issue - let's brainstorm TarFile improvements"

Skill: [Explores TarFile class]

Skill: "Initial topics:
         Task 1: Remove unused methods (pending)
         Task 2: Simplify API (pending)

       Start with task 1?"

User: "Yes"

Skill: [Explores unused methods]
       "I found 8 methods in TarFile. 3 are never called:
        - extract_header()
        - verify_checksum()
        - get_file_count()

       Should we create subtasks for each, or one issue for all 3?"

User: "Let's break it down - one issue per method"

Skill: [Creates subtasks]
       "Created subtasks:
         Task 1.1: Remove extract_header() (pending)
         Task 1.2: Remove verify_checksum() (pending)
         Task 1.3: Remove get_file_count() (pending)

       Explore 1.1 first?"

User: "Yes"

Skill: [Analyzes extract_header()]
       "extract_header() findings:
        - 45 lines
        - No callers
        - Has unit test (will need updating)

       Create issue?"

User: "Yes, #031"

Skill: [Creates issue #031]
       "✓ Created issue #031

       Move to task 1.2?"

User: "Actually, let's skip 1.2 and 1.3 - they might be needed later. Go to task 2."

Skill: [Marks 1.2, 1.3 as pending, starts task 2]
       [Explores API simplification...]
```

### Example 3: Discover New Topics During Exploration

```
User: "/create-issue - explore morphizen-graph module"

Skill: [Initial exploration]
       "Topics identified:
         Task 1: API cleanup (pending)
         Task 2: Performance improvements (pending)"

User: "Start with 1"

Skill: [Explores API]
       "I see many C-style functions like graph_create(), graph_destroy()...
       Should we modernize to C++ classes?"

User: "Hmm, that's a big topic. Let's add it as a separate task. Continue with current topic."

Skill: [Creates Task 3: Modernize to C++ API]

       "Added Task 3. Continuing with API cleanup...

       I found duplicate functions: get_node() and get_node_by_id() do the same thing.
       Remove one?"

User: "Yes, create issue for removing get_node_by_id()"

Skill: [Creates issue #031]

User: "Also, while you were exploring, I just realized we should document
      the breaking changes for v3.0. Add that as a topic."

Skill: [Creates Task 4: Document v3.0 breaking changes]

       "Added Task 4. Current tree:
         ✓ Task 1: API cleanup → Issue #031
         ⊘ Task 2: Performance improvements (pending)
         ⊘ Task 3: Modernize to C++ API (pending)
         ⊘ Task 4: Document v3.0 breaking changes (pending)

       Continue with task 1, or explore another?"
```

---

## Key Commands and Patterns

### Navigation Commands (Natural Language)

User can say:
- "Focus on task 1" / "Let's work on 1" / "Start with 1"
- "Drill into 1.2" / "Explore 1.2 deeper" / "Break down 1.2"
- "Go back" / "Back to parent" / "Up one level"
- "Next" / "Move to next topic" / "Skip this"
- "Show the tree" / "/tasks" / "What's the status?"
- "Add a new topic: [description]"

### Issue Creation Commands

User can say:
- "Create issue #031" / "Make that #031" / "Yes, issue for that"
- "Skip this" / "Not needed" / "Let's skip"
- "Let me check first" / "Not sure yet"

### Session Control

User can say:
- "Done" / "That's all" / "Finish"
- "Pause" / "Let's continue later"
- "Cancel" / "Never mind"

---

## Task Metadata Schema

Each task stores metadata:

```json
{
  "topic_type": "exploration" | "subtopic",
  "issue_number": "031",  // If issue created
  "issue_component": "TarFile",  // Component/feature area (for dependency tracking)
  "issue_files": ["tar_file.cpp", "tar_file.hpp"],  // Files affected (for dependency tracking)
  "exploration_notes": "Found 3 unused methods...",
  "parent_task": "1"  // For hierarchical structure
}
```

**Field usage:**

**For sub-topic tasks (topic_type: "subtopic"):**
- `parent_task`: Required - ID of parent task
- `topic_type`: Always "subtopic"
- Other fields: Added later when creating the issue

**Minimal metadata on creation:**
```javascript
TaskCreate({
  metadata: {
    parent_task: "1",
    topic_type: "subtopic"
  }
})
```

**After creating issue for sub-topic:**
```javascript
TaskUpdate(taskId, {
  status: "completed",
  metadata: {
    issue_number: "050",
    issue_component: "TarFile",
    issue_files: ["tar_file.hpp"]
  }
})
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

---

## Error Handling

### Not on Main Branch
- Ask to switch
- If user declines: exit

### Uncommitted Changes
- Display error
- Exit (don't continue)

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

       Created N issues so far: #031, #032
       Branch: feature/exploration-session-031
       PR: [url] (draft)

       You can continue later or abandon the PR."
```

---

## Notes

**Design Principles**:
- **Exploratory**: Like plan mode - read code, analyze deeply
- **Interactive**: Natural conversation, not rigid workflow
- **Incremental**: Create issues during exploration, not at end
- **Hierarchical**: Support topic trees with drill-down
- **Visual**: Task system shows big picture
- **Flexible**: 0 to N issues, user drives the flow

**Task System Benefits**:
- `/tasks` shows exploration progress anytime
- Resume exploration if session interrupted
- Clear parent-child relationships in tree
- Track which topics led to issues
- Metadata links tasks to created issues

**Git Workflow**:
- One branch per exploration session
- One PR (draft) for all issues
- One commit per issue (clean history)
- Can continue adding issues to same PR

**When to Use This vs Traditional Planning**:
- Use this skill: Exploratory, don't know what issues exist yet
- Use plan mode: Know what to build, need implementation plan
