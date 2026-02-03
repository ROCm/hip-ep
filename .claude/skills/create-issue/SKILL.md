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

Looks like dead code. Should we create an issue to remove it?
```

#### Step 4: Handle User Response

**User can respond in multiple ways:**

**Option A: Create issue immediately**
```
User: "Yes, create issue #031"

You:
  1. Find next issue number
  2. Create branch (feature/add-issue-031-remove-tar-ball)
  3. Create plan file based on exploration findings
  4. Create issue file with problem/solution from findings
  5. Update backlog
  6. Commit
  7. Push to fork
  8. Create draft PR (or add to existing PR if already created)
  9. Update task metadata: issue_number = "031"
  10. Mark task as completed

  Display: "✓ Created issue #031: Remove tar_ball.cpp dead code"
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
  "exploration_notes": "Found 3 unused methods...",
  "parent_task": "1",  // For hierarchical structure
  "depth": 2  // Tree depth (1 = top-level)
}
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
