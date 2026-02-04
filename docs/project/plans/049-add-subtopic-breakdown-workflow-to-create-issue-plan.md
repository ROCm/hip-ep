<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Add Sub-topic Breakdown Workflow to /create-issue

**Issue:** #049
**Created:** 2026-02-03
**Status:** READY

## Objective

Add a structured sub-topic breakdown workflow to `/create-issue` skill for handling complex topics that need to be split into multiple issues. Automatically create tasks for approved sub-topics.

## Background

**Problem discovered:** While discussing TarFile God Class Pattern (Topic 5.2), the AI identified 5 sub-topics but didn't automatically create tasks to track them until the user noticed: "I don't see todos updated."

**Why this matters:** Complex topics need structured breakdown into manageable pieces. Without automatic task creation, sub-topics are identified but not tracked, leading to lost context.

## Implementation Steps

### Step 1: Add "Sub-topic Breakdown Workflow" Section

**File:** `.claude/skills/create-issue/SKILL.md`

**Location:** After "Discussion and Approval Workflow" section (from Issue #045) and before "## Phase 2: Interactive Exploration Loop"

**Content to add:**

````markdown
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

**B) Normal 5-step Workflow** (existing, from Issue #045)
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

### Example Session Flow

````
User: "Let's discuss Topic 5.2: TarFile God Class Pattern"

AI: [Explores TarFile]
    I explored TarFile class - 676 lines, 7 factory methods, multiple
    responsibilities. This is too complex for a single issue.

    I've identified 5 sub-topics:

    Sub-topic 5.2.1: Factory Method Proliferation
    - Problem: 7 different create() methods
    - Complexity: Medium

    [Lists all 5 sub-topics...]

    Does this breakdown make sense?

User: "Yes, good breakdown"

AI: [Creates tasks automatically]
    Created tasks for sub-topics:
    - Task #10: Sub-topic 5.2.1: Factory Method Proliferation
    - Task #11: Sub-topic 5.2.2: Stream Management Responsibility
    - Task #12: Sub-topic 5.2.3: Entry Management Coupling
    - Task #13: Sub-topic 5.2.4: Serialization Responsibility
    - Task #14: Sub-topic 5.2.5: Public API Surface

    Which sub-topic would you like to discuss first?

User: "The simplest one"

AI: Sub-topic 5.2.5: Public API Surface is the simplest (Low complexity).

    [Starts normal 5-step workflow for this sub-topic]

    Step 1: Explore
    [Explores public API surface in detail...]

    Step 2: Discuss
    Before documenting, I have 3 questions:
    1. What is the problem with the API surface?

    [One-by-one questions...]

    Step 3: Summarize
    [Confirms understanding...]

    Step 4: Get Approval
    How should I document this issue?
    1. Brief | 2. Simple plan | 3. Detailed plan

    User: "2"

    Step 5: Create Issue
    [Creates Issue #049: Improve TarFile Public API Documentation]
````

---
````

### Step 2: Add Decision Tree for Complexity

**File:** `.claude/skills/create-issue/SKILL.md`

**Location:** In the new "Sub-topic Breakdown Workflow" section, before "Example Session Flow"

**Add subsection:**

````markdown
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
```
````

### Step 3: Update Task Metadata Schema

**File:** `.claude/skills/create-issue/SKILL.md`

**Location:** In "## Task Metadata Schema" section

**Find:**
```markdown
Each task stores metadata:

```json
{
  "topic_type": "exploration" | "subtopic",
  "issue_number": "031",
  "issue_component": "TarFile",
  "issue_files": ["tar_file.cpp", "tar_file.hpp"],
  "exploration_notes": "Found 3 unused methods...",
  "parent_task": "1",
  "depth": 2
}
```
```

**Update to clarify sub-topic metadata:**
```markdown
Each task stores metadata:

```json
{
  "topic_type": "exploration" | "subtopic",
  "issue_number": "031",
  "issue_component": "TarFile",
  "issue_files": ["tar_file.cpp", "tar_file.hpp"],
  "exploration_notes": "Found 3 unused methods...",
  "parent_task": "1"
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
```

### Step 4: Add Reminder to Phase 2

**File:** `.claude/skills/create-issue/SKILL.md`

**Location:** In "## Phase 2: Interactive Exploration Loop" → After "Step 3: Present Findings"

**Add reminder:**

```markdown
**Reminder: Check Complexity**

After presenting findings, evaluate:
- Is this too complex for a single issue?
- Check decision tree in "Sub-topic Breakdown Workflow"
- If YES → Use Sub-topic Breakdown Workflow
- If NO → Continue with normal workflow (ask questions, summarize, approve, create)
```

## Verification

### Test Scenario 1: Complex Topic Discovery

1. User asks to explore a complex topic (e.g., "Refactor configuration system")
2. AI explores and finds: multiple files, architectural changes, mixed complexity
3. Verify AI recognizes complexity and suggests sub-topic breakdown
4. Verify AI lists concrete sub-topics with complexity estimates
5. User approves breakdown
6. Verify AI automatically creates tasks for all sub-topics
7. Verify task metadata includes parent_task and topic_type
8. Verify AI asks which sub-topic to start with

### Test Scenario 2: Simple Topic (No Breakdown Needed)

1. User asks to explore a simple topic (e.g., "Fix typo in header")
2. AI explores and finds: single file, clear fix, low complexity
3. Verify AI does NOT suggest breakdown
4. Verify AI proceeds with normal 5-step workflow
5. Creates single issue

### Test Scenario 3: Sub-topic Goes Through Normal Workflow

1. After breakdown, user selects a sub-topic
2. Verify AI starts normal 5-step workflow for that sub-topic:
   - Explore (in detail)
   - Discuss (questions one-by-one)
   - Summarize (confirm understanding)
   - Get Approval (plan options)
   - Create Issue (with dependencies updated)
3. Verify issue created for that sub-topic
4. Verify task updated with issue_number

## Success Criteria

- [ ] "Sub-topic Breakdown Workflow" section added
- [ ] Decision tree for complexity assessment added
- [ ] Task metadata schema clarified for sub-topics
- [ ] Reminder added to Phase 2 to check complexity
- [ ] Example session flow shows complete breakdown workflow
- [ ] Automatic task creation documented
- [ ] Integration with normal 5-step workflow explained
- [ ] Minimal metadata approach (parent_task + topic_type) documented

## Files Modified

- `.claude/skills/create-issue/SKILL.md` - Add sub-topic breakdown workflow

## Notes

**Why separate workflow:**
Sub-topic breakdown creates TASKS (for tracking), not ISSUES (for documentation). Each sub-topic then goes through the normal workflow to create its own issue.

**Why automatic task creation:**
Without automatic creation, sub-topics are identified but not tracked. User has to remind AI, which breaks flow and risks losing context.

**Why minimal metadata:**
Start simple with just parent_task and topic_type. Additional fields (issue_component, issue_files) are added later when creating the issue for that sub-topic.

**When to use:**
- God Class patterns (multiple responsibilities)
- Large refactorings (architectural changes)
- Feature areas with mixed complexity (some trivial, some complex)

**When NOT to use:**
- Code duplication (even if in many places, single pattern)
- Simple refactorings (even if multi-file, straightforward)
- Single responsibility issues

**Example from this session:**
- Parent: Topic 5.2: TarFile God Class Pattern
- Breakdown created 5 tasks (#10-#14)
- Each will become a separate issue when explored
