<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Fix /create-issue Skill - Prevent Implementation During Discussion

**Issue:** #043
**Created:** 2026-02-03
**Status:** READY

## Objective

Fix two critical problems in `/create-issue` skill:
1. Stop implementing work during issue creation (should only create documentation)
2. Ask questions one-by-one with proper pattern (not all at once)

## Background

**Discovery:** While using `/create-issue` to create Issue #042 (Document PrivateTag pattern), the skill:
- Created the technical documentation file during discussion
- Modified source code during discussion
- Committed implementation along with issue documentation

This violates the principle that issue creation should only document WHAT needs to be done, not DO the work.

Also, during discussion the skill asked multiple compound questions at once, overwhelming the user who previously gave feedback: "did you forget to ask question one by one?"

## Implementation Steps

### Step 1: Add "Issue Creation vs Implementation" Section

**File:** `.claude/skills/create-issue/SKILL.md`

**Location:** Add new section after "## Phase 1: Initialize Exploration Session" and before "## Phase 2: Interactive Exploration Loop"

**Content to add:**

```markdown
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

Later, when someone (or you) implements Issue #042, THEN create the tech doc and modify code.

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
```

### Step 2: Update "Step 4: Handle User Response - Option A"

**File:** `.claude/skills/create-issue/SKILL.md`

**Location:** In "## Phase 2: Interactive Exploration Loop" → "### Step-by-Step: Exploring a Topic" → "#### Step 4: Handle User Response"

**Find this section:**
```markdown
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
```

**Replace with:**
```markdown
**Option A: Create issue immediately**
```
User: "Yes, create issue #031"

You:
  1. Find next issue number
  2. If first issue: Create branch (feature/exploration-session-031)
  3. Create plan file based on exploration findings (describes HOW to implement)
  4. Create issue file with problem/solution from findings (describes WHAT to do)
  5. Update backlog
  6. ❌ CRITICAL: DO NOT implement anything described in issue/plan
     - Do NOT create deliverable files
     - Do NOT modify source code
     - ONLY create issue documentation
  7. Commit ONLY the issue documentation files
  8. Push to fork
  9. If first issue: Create draft PR
  10. Update task metadata: issue_number = "031"
  11. Mark task as completed

  Display: "✓ Created issue #031: Remove tar_ball.cpp dead code"

**Remember:** The plan describes implementation steps, but you do NOT execute those steps.
The issue/plan is documentation for future work, not the work itself.
```
```

### Step 3: Add "Asking Questions One-by-One" Section

**File:** `.claude/skills/create-issue/SKILL.md`

**Location:** Add new subsection under "## Phase 2: Interactive Exploration Loop" → after "### Loop Structure"

**Content to add:**

```markdown
### Asking Questions During Discussion

When you need multiple answers to create a complete issue, use the one-by-one pattern:

#### Pattern

1. **Preview all questions** - List them numbered so user knows what's coming
2. **State the approach** - Tell user you'll ask one-by-one
3. **Ask first question** - Present first question with options
4. **Wait for answer** - User responds
5. **Acknowledge and continue** - Summarize answer, ask next question
6. **Repeat** until all questions answered

#### Example: Good (One-by-One)

```
I have 3 questions to clarify this issue:
1. What's the actual problem we're solving?
2. What should the deliverables be?
3. Should this have a plan or just brief issue?

I will ask you questions one by one. Let's start with:

**Question 1:** What's the actual problem we're solving?

Is it:
- A) Developers are confused by the PrivateTag pattern when reading code?
- B) Missing documentation makes the code hard to maintain?
- C) It's a learning opportunity - documenting a useful C++ idiom?
- D) Something else?
```

[User answers: "A, B, and C - all correct"]

```
Got it - it addresses developer confusion, maintainability, AND serves as learning resource.

**Question 2:** What should the deliverables be?

Should we:
- A) Add the technical doc to the repository (docs/technical/pattern.md)
- B) Update the comment in tar_file.hpp to reference the doc
- C) Both A and B
- D) Something more?
```

[User answers: "C - both"]

```
Perfect! Both the technical doc AND comment update.

**Question 3:** What should the updated comment say?

Should it be:
- A) // PrivateTag pattern - see docs/technical/privatetag-factory-pattern.md
- B) // Enables factory pattern with std::make_unique - see docs/technical/...
- C) Something more detailed in the comment itself?
```

#### Example: Bad (All At Once) ❌

```
I need to understand a few things:

Question 1: What's the actual problem?
- A) Developer confusion
- B) Maintainability
- C) Learning opportunity

Question 2: What deliverables?
- A) Tech doc
- B) Comment update
- C) Both

Question 3: What should comment say?
- A) Short reference
- B) Detailed explanation

Please answer all three.
```

**Why this is bad:**
- Overwhelming - user has to track 3 questions + 9 options
- Hard to respond - unclear format for answering multiple questions
- Violates user preference for one-by-one approach

#### When to Use This Pattern

Use one-by-one questioning when:
- Creating an issue requires clarifying multiple aspects
- Exploring a topic raises several design questions
- User needs to make multiple decisions

**Prefer:**
- 2-4 questions max (if more, issue might be too complex)
- Clear, focused questions with 2-4 options each
- Explicit acknowledgment of each answer before proceeding

---
```

### Step 4: Update Example Session Flows

**File:** `.claude/skills/create-issue/SKILL.md`

**Location:** "## Example Session Flow" section

**Add note to examples:**

After each example showing issue creation, add:

```markdown
**Note:** Issue #031 documentation created (issue + plan files). The actual removal
of tar_ball.cpp will happen later when someone implements Issue #031, NOT during
this issue creation session.
```

### Step 5: Add Warning in Issue File Creation Section

**File:** `.claude/skills/create-issue/SKILL.md`

**Location:** "## Phase 3: Git Workflow for Incremental Issue Creation" → "### Issue File Creation"

**Add before "For each issue:":**

```markdown
### Issue File Creation (Documentation Only)

**CRITICAL REMINDER:** During issue creation, you ONLY create documentation files:
- Issue file (`docs/project/issues/NNN-name.md`)
- Plan file (`docs/project/plans/NNN-name-plan.md`)
- Backlog update (`docs/project/backlog.md`)

**Do NOT create or modify:**
- Source code files
- Technical documentation
- Configuration files
- Any deliverables described in the issue

Those will be created LATER when the issue is implemented.

---

For each issue:
```

## Verification

### Test Case 1: Documentation Issue

**Scenario:** Creating an issue to document a pattern (like Issue #042)

**Expected behavior:**
1. ✅ Create issue file describing what doc is needed
2. ✅ Create plan file describing doc structure and where to add it
3. ✅ Commit only issue/plan files
4. ❌ Do NOT create the actual technical documentation
5. ❌ Do NOT modify source code comments

### Test Case 2: Code Refactoring Issue

**Scenario:** Creating an issue to extract duplicate code

**Expected behavior:**
1. ✅ Create issue file describing the duplication problem
2. ✅ Create plan file showing before/after code examples
3. ✅ Commit only issue/plan files
4. ❌ Do NOT create the helper function
5. ❌ Do NOT modify the source files

### Test Case 3: Multiple Questions

**Scenario:** Need to ask 3 questions to create complete issue

**Expected behavior:**
```
Skill: I have 3 questions:
       1. What's the problem?
       2. What are deliverables?
       3. Any other considerations?

       I will ask one by one. Let's start with:

       Question 1: What's the problem?
       [options]

[User answers]

Skill: Got it. Question 2: What are deliverables?
       [options]

[User answers]

Skill: Perfect. Question 3: Any other considerations?
       [options]
```

## Success Criteria

- [ ] Skill documentation clearly states "issue creation = documentation only"
- [ ] Explicit "DO NOT implement" warnings added in multiple places
- [ ] Option A updated with reminder not to implement
- [ ] One-by-one question pattern documented with examples
- [ ] All example sessions include notes about implementation happening later
- [ ] Verification section confirms skill only creates docs, not implementation

## Files Modified

- `.claude/skills/create-issue/SKILL.md` - Add sections and update existing content

## Notes

**Why multiple reminders?**
The skill is long (~800 lines). Adding reminders in multiple strategic locations ensures the AI sees the rule regardless of which section it's focusing on during execution.

**Strategic reminder locations:**
1. High-level section after Phase 1 (sets context early)
2. In Option A handling (most common creation path)
3. In git workflow section (right before committing)
4. In examples (reinforces through demonstration)
