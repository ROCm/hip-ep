<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #043: Fix /create-issue Skill - Prevent Implementation During Discussion

## Metadata
- **Type:** Bug Fix / Skill Improvement
- **Priority:** MEDIUM
- **Created:** 2026-02-03
- **Dependencies:** None

## Description

Fix two critical problems with `/create-issue` skill that cause poor user experience during issue creation discussions:
1. **Implementation during discussion** - Skill creates implementation artifacts (files, code changes) when it should only create issue documentation
2. **Overwhelming questions** - Asks multiple questions in one response instead of one-by-one

## Problem

### Problem #1: Implementation During Discussion

**What happens:**
During issue creation discussion, when user agrees on what needs to be done, the skill immediately:
- ❌ Creates implementation files (e.g., technical documentation)
- ❌ Modifies source code (e.g., updates comments)
- ❌ Commits the implementation
- ✅ Creates issue file (correct)

**Example - Issue #042:**
```
User: "yes we need to add some comment references..."
Skill: [Immediately creates docs/technical/privatetag-factory-pattern.md]
Skill: [Immediately updates morphizen-core/src/tar_file.hpp]
Skill: [Commits everything together]
```

**What should happen:**
Issue creation phase should ONLY create documentation about the work:
- ✅ Create issue file describing WHAT needs to be done
- ✅ Create plan file describing HOW to do it
- ✅ Update backlog.md
- ✅ Commit the issue documentation
- ⏹️ STOP - Don't implement anything!

The actual work (creating files, modifying code) should happen LATER when someone implements the issue.

**Why this is wrong:**
1. Conflates "planning" with "doing"
2. Issue becomes immediately obsolete (work is already done)
3. PR contains both issue documentation AND implementation (should be separate)
4. Violates single responsibility - issue creation should only document work, not do it

### Problem #2: Overwhelming Multiple Questions

**What happens:**
During issue discussion, skill asks multiple questions in one response:

```
Question 1: What's the actual problem we're solving?
Is it:
- A) Developers are confused?
- B) Missing documentation?
- C) Learning opportunity?
- D) Something else?

Question 2: What should the deliverables be?
Should we:
- A) Add technical doc?
- B) Update comment?
- C) Both?

Question 3: What should the comment say?
- A) Option 1
- B) Option 2
- C) Something else?
```

User previously gave explicit feedback: "did you forget to ask question one by one?"

**What should happen:**
1. Preview all questions upfront
2. Ask one question at a time
3. Wait for answer before proceeding

**Better format:**
```
I have 3 questions:
1. What's the actual problem we're solving?
2. What should the deliverables be?
3. What should the comment say?

I will ask you questions one by one. Let's start with:

Question 1: What's the actual problem we're solving?
Is it:
- A) Developers are confused?
- B) Missing documentation?
- C) Learning opportunity?
- D) Something else?
```

**Why this is important:**
1. Reduces cognitive load - user focuses on one thing at a time
2. Prevents user from feeling overwhelmed
3. Follows user's explicit preference
4. Matches the skill's stated "one by one" principle

## Solution

Update `/create-issue` skill instructions to:

### Fix #1: Prevent Implementation During Discussion

**Add explicit rule in skill instructions:**
```markdown
## CRITICAL: Issue Creation vs Implementation

**During issue creation, ONLY create documentation:**
- ✅ Create issue file (docs/project/issues/NNN-name.md)
- ✅ Create plan file (docs/project/plans/NNN-name-plan.md) - if needed
- ✅ Update backlog.md
- ✅ Commit issue documentation

**DO NOT implement the work:**
- ❌ Do NOT create technical documentation files
- ❌ Do NOT modify source code
- ❌ Do NOT create any deliverables described in the issue
- ❌ Do NOT do anything beyond documenting what needs to be done

**The plan file should describe HOW to implement, not actually implement.**

**Exception:** Only if the issue is about creating documentation AND the user
explicitly says "create the doc now as part of the issue", then it's okay.
But by default, assume issue creation = documentation only.
```

**Add reminder in exploration loop:**
```markdown
#### Step 4: Handle User Response - Option A: Create issue immediately

When user agrees to create an issue:

1. Find next issue number
2. Create issue file with problem/solution
3. Create plan file (if needed) - describes HOW to implement
4. Update backlog
5. Commit ONLY the issue documentation
6. ❌ DO NOT implement anything described in the issue/plan
7. Push to fork
```

### Fix #2: One-by-One Question Pattern

**Add question handling guideline:**
```markdown
## Asking Questions During Discussion

When you need to ask multiple questions to clarify an issue:

**Pattern:**
1. List all questions upfront (numbered)
2. Tell user you'll ask one-by-one
3. Ask first question and WAIT for answer
4. After answer, ask next question
5. Continue until all answered

**Example:**
```
I have 3 questions to clarify this issue:
1. What's the actual problem we're solving?
2. What should the deliverables be?
3. Should this have a plan or just brief issue?

I will ask you questions one by one. Let's start with:

**Question 1:** What's the actual problem we're solving?
Is it:
- A) Developers are confused by the PrivateTag pattern?
- B) Missing documentation makes code hard to maintain?
- C) Learning opportunity for the team?
- D) Something else?
```

**After user answers Question 1:**
```
Got it - all three (A, B, C) are correct.

**Question 2:** What should the deliverables be?
Should we:
- A) Add technical doc only?
- B) Update code comment only?
- C) Both A and B?
- D) Something more?
```

**Do NOT do this (bad):**
```
Question 1: What's the problem?
Question 2: What are deliverables?
Question 3: What should the comment say?

[Waiting for answers to all 3 at once]
```
```

## Plans

- [043-fix-create-issue-implementation-during-discussion-plan.md](../plans/043-fix-create-issue-implementation-during-discussion-plan.md) - Created 2026-02-03

## Notes

**Root cause:** The `/create-issue` skill instructions don't clearly separate "issue creation" from "implementation". It focuses on exploring code and creating issues, but doesn't explicitly prohibit implementing the work during the discussion phase.

**Impact:** Users create "issues" that are already implemented, making the issue tracking system less useful. PRs mix documentation with implementation.

**Related:** This was discovered while using `/create-issue` to create TarFile organizational improvement issues (Issue #041, #042).
