<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #045: Add Approval Workflow to /create-issue Skill

## Metadata
- **Type:** Skill Improvement
- **Priority:** HIGH
- **Created:** 2026-02-03
- **Dependencies:** Issue #043 (related skill improvements)

## Description

Add approval workflow to `/create-issue` skill to ensure AI properly understands issues before documenting them, and allow user to control plan detail level. Currently the skill jumps from exploration directly to issue creation without discussion or confirmation.

## Problem

**Current behavior:**

```
AI: [Explores code, finds duplication]
AI: "Perfect! I found the duplication. Lines 303-316..."
AI: "Now let me create the issue documenting this:"
User: [Has to interrupt] ❌
```

**Two critical problems:**

### Problem A: Creates Issues Without Proper Understanding
- AI explores code and immediately creates issue
- Skips verification that understanding is correct
- User has to interrupt to prevent premature issue creation
- Results in issues that may not capture the real problem

### Problem B: No Discussion Phase
- No opportunity to ask clarifying questions
- No summary presented for user confirmation
- User can't verify AI understood correctly before documenting
- Creates issues with incomplete or incorrect understanding

**Example from real usage:**
When discussing TarFile code duplication, AI said "Now let me create the issue" immediately after finding duplicated code, without discussing what the duplication was, why it mattered, or what the solution should be. User had to interrupt.

## Solution

Add **5-step approval workflow** between exploration and issue creation:

### Step 1: Explore
Find and analyze the problem (existing behavior, keep as-is)

### Step 2: Discuss
Ask clarifying questions one-by-one:
- What is this problem exactly?
- Why does it matter?
- What should the solution be?
- Any other considerations?

Use the pattern from Issue #043:
```
I have 3 questions:
1. What is this?
2. Why does it matter?
3. What's the solution?

I will ask one by one. Let's start with:

Question 1: What is this?
[Wait for answer]

Question 2: Why does it matter?
[Wait for answer]
...
```

### Step 3: Summarize
Present understanding back to user for confirmation:
```
Let me summarize what I understood:

**Problem:** [description]
**Why it matters:** [impact]
**Solution:** [approach]
**Files affected:** [list]

Does this match what you're thinking?
```

Wait for user to confirm or correct.

### Step 4: Get Approval
Offer plan options with recommendation:
```
How should I document this issue?

1. Brief issue only (no plan) - For trivial fixes
2. Issue with simple plan - Basic implementation steps
3. Issue with detailed plan (Recommended) - Complete guide for clean session
4. Skip this issue - Not worth documenting

I recommend: Option [X] because [reason]

Which option do you prefer?
```

**Recommendation logic:**
- Trivial (1-line, typo, obvious) → Option 1
- Straightforward (single file, clear steps) → Option 2
- Complex (multiple files, design decisions, needs context) → Option 3
- Default: Option 3 (when in doubt, more detail is better)

### Step 5: Create
Only after user selects an option, create the issue with appropriate detail level.

## Plan Options Detail

### Option 1: Brief Issue Only
**Content:**
- Problem description (2-3 paragraphs)
- High-level solution (1-2 paragraphs)
- Files affected (list)
- No step-by-step plan

**When to use:** Trivial fixes like typos, removing single line, obvious changes

**Example:** Issue #041 (duplicate friend declaration)

### Option 2: Issue with Simple Plan
**Content:**
- Problem description with examples
- Solution approach
- Basic implementation steps (5-10 bullet points)
- File locations
- Simple before/after examples
- Verification command

**When to use:** Straightforward changes, single file modifications, clear approach

**Example:** Issue #044 (erase-remove idiom)

### Option 3: Issue with Detailed Plan
**Content:**
- Comprehensive problem analysis
- Solution design with rationale
- Detailed step-by-step implementation guide:
  - Exact file paths and line numbers
  - Complete before/after code examples
  - Multi-step procedures
  - Edge cases to handle
- Verification commands with expected output
- Success criteria checklist
- **Critical:** Enough context for fresh Claude session to implement without this conversation

**When to use:**
- Complex changes affecting multiple files
- Design decisions involved
- Requires understanding of context/rationale
- Default choice when uncertain

**Example:** Issue #043 (fix /create-issue skill)

## Plans

- [045-add-approval-workflow-to-create-issue-plan.md](../plans/045-add-approval-workflow-to-create-issue-plan.md) - Created 2026-02-03

## Notes

**This issue discovered while using `/create-issue`:** During TarFile organizational improvements session, the skill repeatedly jumped from exploration to issue creation without discussion, requiring user interruption.

**Related to Issue #043:** Issue #043 addresses preventing implementation during discussion. This issue (#045) addresses ensuring proper understanding before documentation.

**Impact:** High priority because this affects the core workflow of the skill and directly impacts issue quality.
