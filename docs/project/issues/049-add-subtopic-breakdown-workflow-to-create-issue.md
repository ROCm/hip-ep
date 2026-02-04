<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #049: Add Sub-topic Breakdown Workflow to /create-issue

## Metadata
- **Type:** Skill Improvement
- **Priority:** MEDIUM
- **Created:** 2026-02-03
- **Dependencies:** Related to #043, #045, #046 (all improve /create-issue skill)

## Description

Add a structured workflow for breaking down complex topics into manageable sub-topics in the `/create-issue` skill. When exploration reveals a topic is too complex for a single issue, the skill should guide the breakdown into trackable sub-topics.

## Problem

**Current behavior:**

When the `/create-issue` skill discovers a complex topic (like "TarFile God Class Pattern"), there's no structured workflow for breaking it down:

```
User: Let's discuss Topic 5.2: TarFile God Class Pattern

AI: [Explores and finds it's complex]
    This is complex, has multiple responsibilities...

User: "Let's break it into sub-topics"

AI: [Informally identifies sub-topics]
    Here are 5 sub-topics... which one first?

User: "But I don't see todos updated"

AI: Oh! [Manually creates tasks after being reminded]
```

**Problems:**

1. **No structured workflow** - Breaking down happens informally
2. **Tasks not automatically created** - User has to remind AI to create sub-topic tasks
3. **Inconsistent pattern** - Sometimes tasks created, sometimes not
4. **Loses context** - Sub-topics identified but not tracked
5. **Poor integration** - Unclear how breakdown relates to normal 5-step workflow

## Solution

Add a **Sub-topic Breakdown Workflow** that branches from exploration when complexity is discovered:

### Workflow Steps

**Step 1: Explore** (existing)
- During exploration, AI discovers topic is too complex for single issue
- Indicators: Multiple responsibilities, large scope, architectural changes, 500+ lines of code

**Step 2: Identify Sub-topics**
- List concrete, actionable sub-topics
- Each sub-topic should be independently solvable
- Include brief description and complexity estimate

**Step 3: Summarize Breakdown**
- Present all sub-topics to user for review
- Format:
  ```
  I found this topic is too complex for a single issue.

  I've identified 5 sub-topics:

  Sub-topic 5.2.1: Factory Method Proliferation
  - Problem: 7 different create() factory methods
  - Complexity: Medium

  Sub-topic 5.2.2: Stream Management Responsibility
  - Problem: TarFile manages iostream and MemStream
  - Complexity: High

  ...

  Does this breakdown make sense?
  ```

**Step 4: Get Approval**
- Wait for user confirmation
- User can suggest changes to breakdown

**Step 5: Create Tasks Automatically**
- After user approves, create task for each sub-topic
- Use minimal metadata:
  ```json
  {
    "parent_task": "1",
    "topic_type": "subtopic"
  }
  ```
- Report task creation:
  ```
  Created tasks for sub-topics:
  - Task #10: Sub-topic 5.2.1: Factory Method Proliferation
  - Task #11: Sub-topic 5.2.2: Stream Management Responsibility
  ...
  ```

**Step 6: Ask Which First**
- Prompt user to select which sub-topic to explore
- Selected sub-topic then goes through normal 5-step workflow

### Integration with Normal Workflow

**After breakdown, each sub-topic follows:**
1. Explore (the specific sub-topic)
2. Discuss (ask questions one-by-one)
3. Summarize (confirm understanding)
4. Get Approval (offer plan options)
5. Create Issue (document that sub-topic)

**Result:** Parent topic gets multiple issues, each addressing one sub-topic.

### When to Use Sub-topic Breakdown

**Use breakdown when:**
- Multiple distinct responsibilities identified
- Would require changing 5+ files
- Architectural refactoring needed
- Estimated 500+ lines of code changes
- Mix of high/low complexity changes

**Don't use breakdown when:**
- Single clear problem to solve
- Changes localized to 1-3 files
- Straightforward refactoring

## Plans

- [049-add-subtopic-breakdown-workflow-to-create-issue-plan.md](../plans/049-add-subtopic-breakdown-workflow-to-create-issue-plan.md) - Created 2026-02-03

## Notes

**Discovery:** While discussing TarFile God Class Pattern (Topic 5.2), identified 5 sub-topics but forgot to create tasks automatically. User noticed: "I don't see todos updated."

**Related issues:**
- Issue #043: Prevent implementation during discussion
- Issue #045: Add approval workflow (this extends that pattern)
- Issue #046: Update dependencies after each issue

All four issues (#043, #045, #046, #049) improve the `/create-issue` skill workflow.

**Example - TarFile God Class:**
- Parent: Topic 5.2: TarFile God Class Pattern
- Sub-topics created:
  - Task #10: Factory Method Proliferation
  - Task #11: Stream Management Responsibility
  - Task #12: Entry Management Coupling
  - Task #13: Serialization Responsibility
  - Task #14: Public API Surface
