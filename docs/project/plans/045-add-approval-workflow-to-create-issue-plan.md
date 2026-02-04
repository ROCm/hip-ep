<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Add Approval Workflow to /create-issue Skill

**Issue:** #045
**Created:** 2026-02-03
**Status:** READY

## Objective

Add 5-step approval workflow to `/create-issue` skill: Explore → Discuss → Summarize → Get Approval → Create. This ensures AI properly understands issues before documenting them and allows user to control plan detail level.

## Background

**Problem discovered during usage:** While creating TarFile organizational issues, the skill repeatedly jumped from exploration directly to issue creation without discussion:

```
AI: [Finds code duplication]
AI: "Now let me create the issue documenting this:"
User: [Interrupts] ❌
```

**Root cause:** The skill's Phase 2 (Interactive Exploration Loop) goes directly from "Present findings" to "Handle User Response - Create issue" without a discussion/approval step.

## Implementation Steps

### Step 1: Add "Discussion and Approval Workflow" Section

**File:** `.claude/skills/create-issue/SKILL.md`

**Location:** After the "CRITICAL: Issue Creation vs Implementation" section (added in Issue #043) and before "## Phase 2: Interactive Exploration Loop"

**Content to add:**

````markdown
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

### Step 5: Create Issue
Only after user selects an option, create the issue with appropriate detail level.

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

---
````

### Step 2: Update Phase 2 "Step 3: Present Findings"

**File:** `.claude/skills/create-issue/SKILL.md`

**Location:** In "## Phase 2: Interactive Exploration Loop" → "### Step-by-Step: Exploring a Topic" → "#### Step 3: Present Findings and Suggestions"

**Find this:**
```markdown
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
```

**Replace with:**
```markdown
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

Instead, transition to discussion phase (Step 2 of approval workflow).
```

### Step 3: Update "Step 4: Handle User Response - Option A"

**File:** `.claude/skills/create-issue/SKILL.md`

**Location:** In "## Phase 2: Interactive Exploration Loop" → "### Step-by-Step: Exploring a Topic" → "#### Step 4: Handle User Response"

**Find the "Option A: Create issue immediately" section:**
```markdown
**Option A: Create issue immediately**
```
User: "Yes, create issue #031"

You:
  1. Find next issue number
  ...
```
```

**Replace with:**
```markdown
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
  7. Update backlog
  8. ❌ DO NOT implement anything (see "Issue Creation vs Implementation")
  9. Commit ONLY issue documentation
  10. Push to fork
  11. If first issue: Create draft PR
  12. Update task metadata
  13. Mark task as completed

**Critical:** Steps 2-4 (Discuss → Summarize → Get Approval) are MANDATORY.
Never skip directly from exploration to issue creation.
```
```

### Step 4: Add Examples to Workflow Section

**File:** `.claude/skills/create-issue/SKILL.md`

**Location:** In the new "Discussion and Approval Workflow" section, after Step 5

**Add subsection:**

````markdown
### Workflow Examples

#### Example 1: Complex Issue (Detailed Plan Recommended)

```
[After exploration finds code duplication]

AI: I found duplicate entry deduplication logic in tar_file.cpp.
    Before documenting this, I have 3 questions:
    1. What exactly is being duplicated?
    2. Why does this duplication matter?
    3. What's the solution approach?

    I will ask one by one. Let's start with:

    **Question 1:** What exactly is being duplicated?

    I see similar code in:
    - add_regular_entry() (lines 303-316)
    - add_symlink_entry() (lines 362-376)

    Is this the duplication you're concerned about?

User: Yes, that's it.

AI: **Question 2:** Why does this duplication matter?

    Is it:
    - A) Maintenance burden (changes need to be made twice)
    - B) Violates DRY principle
    - C) Risk of divergence (one gets fixed, other doesn't)
    - D) All of the above

User: D - all of the above

AI: **Question 3:** What's the solution approach?

    Should we:
    - A) Extract to a private helper function
    - B) Use a template function
    - C) Refactor the class structure

User: A - extract to helper

AI: Let me summarize what I understood:

    **Problem:**
    Entry deduplication logic is duplicated in add_regular_entry() and
    add_symlink_entry() (~14 lines each). Same std::remove_if + erase pattern.

    **Why it matters:**
    - Maintenance burden (update in two places)
    - Violates DRY principle
    - Risk of divergence if one location gets fixed without the other

    **Solution:**
    Extract to private helper function (e.g., remove_entry_by_path())

    **Files affected:**
    - morphizen-core/src/tar_file.cpp (lines 303-316, 362-376)
    - morphizen-core/src/tar_file.hpp (add helper declaration)

    **Implementation complexity:**
    Medium - straightforward extraction but need to preserve logging behavior

    Does this match what you're thinking?

User: Yes

AI: How should I document this issue?

    1. Brief issue only (no plan)
    2. Issue with simple plan
    3. Issue with detailed plan (Recommended) ✅
    4. Skip this issue

    I recommend: **Option 3 (detailed plan)** because this involves:
    - Two separate locations to modify
    - Helper function signature design
    - Logging behavior preservation
    - A fresh Claude session will benefit from detailed context

    Which option do you prefer?

User: Option 3

AI: [Creates Issue #045 with detailed plan]
```

#### Example 2: Simple Issue (Simple Plan)

```
[After exploration finds verbose code pattern]

AI: I found verbose erase-remove pattern in tar_file.cpp with unnecessary if-check.
    Before documenting, I have 2 questions:
    1. What's the issue with the current pattern?
    2. What's the standard alternative?

    **Question 1:** What's the issue with the current pattern?

    Current code:
    ```cpp
    auto it = std::remove_if(...);
    if (it != entries_.end()) {  // This check
      entries_.erase(it, entries_.end());
    }
    ```

    Is the problem the unnecessary if-check?

User: Yes

AI: **Question 2:** What's the standard alternative?

    Should we use the one-liner erase-remove idiom:
    ```cpp
    entries_.erase(std::remove_if(...), entries_.end());
    ```

User: Correct

AI: Let me summarize:

    **Problem:**
    Verbose erase-remove pattern with unnecessary if-check in two locations

    **Why it matters:**
    - Not following standard C++ idiom
    - More code than needed (4 lines vs 1)

    **Solution:**
    Use standard erase-remove idiom (one-liner)

    **Files affected:**
    - morphizen-core/src/tar_file.cpp (lines 303-316, 362-376)

    **Implementation complexity:**
    Simple - straightforward pattern replacement

    Does this match your understanding?

User: Yes

AI: How should I document this issue?

    1. Brief issue only (no plan)
    2. Issue with simple plan (Recommended) ✅
    3. Issue with detailed plan
    4. Skip this issue

    I recommend: **Option 2 (simple plan)** because:
    - Straightforward pattern replacement
    - Well-known C++ idiom
    - Basic before/after example is sufficient

    Which option do you prefer?

User: Option 2

AI: [Creates Issue #044 with simple plan]
```

#### Example 3: Trivial Issue (Brief Only)

```
[After exploration finds duplicate line]

AI: I found duplicate friend class declaration in tar_entry.hpp line 127.
    One quick question: Should we just remove the duplicate line?

User: Yes

AI: Let me summarize:

    **Problem:**
    Duplicate "friend class TarFile;" on line 127

    **Solution:**
    Delete line 127

    **Files affected:**
    - morphizen-core/src/tar_entry.hpp (line 127)

    Correct?

User: Yes

AI: How should I document this issue?

    1. Brief issue only (Recommended) ✅
    2. Issue with simple plan
    3. Issue with detailed plan
    4. Skip this issue

    I recommend: **Option 1 (brief)** because:
    - Trivial 1-line deletion
    - No decisions needed
    - Obvious fix

    Which option do you prefer?

User: Option 1

AI: [Creates Issue #041 with brief description only]
```

---
````

### Step 5: Update Example Session Flows

**File:** `.claude/skills/create-issue/SKILL.md`

**Location:** "## Example Session Flow" section

**For each existing example, add the discussion phase.** For instance, update Example 1:

**Before:**
```
Skill: "I found tar_ball.cpp/hpp - no references anywhere.
       Should we create an issue to remove it?"

User: "Yes, make it #031"
```

**After:**
```
Skill: "I found tar_ball.cpp/hpp - no references anywhere.
       Before documenting, let me ask: What exactly should we remove?
       Just the .cpp/.hpp files, or also from build system?"

User: "Both - files and any CMakeLists references"

Skill: "Summary: Remove tar_ball.{cpp,hpp} files and CMakeLists entries.
       Correct?"

User: "Yes"

Skill: "How should I document this?
       1. Brief | 2. Simple plan | 3. Detailed plan ✅ | 4. Skip

       Recommend: Option 3 - multiple files + build system

       Which option?"

User: "Option 3"

Skill: [Creates Issue #031 with detailed plan]
```

### Step 6: Add Reminder in Task Update Section

**File:** `.claude/skills/create-issue/SKILL.md`

**Location:** Before "#### Step 5: Navigate and Continue"

**Add:**

```markdown
**Reminder:** Before creating any issue, always:
1. Discuss (ask questions)
2. Summarize (confirm understanding)
3. Get approval (offer plan options)
4. Create (only after user selects option)

Never skip the discussion phase.
```

## Verification

### Test Scenario 1: Complex Issue
1. Explore code and find complex problem (multiple files)
2. Verify skill asks clarifying questions one-by-one
3. Verify skill summarizes understanding
4. Verify skill recommends Option 3 (detailed plan)
5. Verify detailed plan contains enough context for fresh session

### Test Scenario 2: Simple Issue
1. Explore and find straightforward issue
2. Verify discussion happens
3. Verify skill recommends Option 2 (simple plan)
4. Verify simple plan has basic steps

### Test Scenario 3: Trivial Issue
1. Explore and find 1-line fix
2. Verify brief discussion
3. Verify skill recommends Option 1 (brief)
4. Verify only issue file created (no plan)

### Test Scenario 4: User Correction
1. AI presents summary
2. User corrects understanding
3. Verify AI updates and re-summarizes
4. Verify only proceeds after confirmation

## Success Criteria

- [ ] "Discussion and Approval Workflow" section added to skill
- [ ] Phase 2 Step 3 updated to transition to discussion (not ask for creation)
- [ ] Phase 2 Step 4 Option A enforces mandatory discussion steps
- [ ] Examples added showing all 3 plan detail levels
- [ ] Existing example sessions updated with discussion phase
- [ ] Reminder added to prevent skipping discussion
- [ ] Fresh Claude session can follow the updated workflow correctly

## Files Modified

- `.claude/skills/create-issue/SKILL.md` - Add workflow section, update Phase 2, add examples

## Notes

**Why 5 steps are necessary:**
1. Explore - Understand the code
2. Discuss - Ensure AI understands the problem correctly
3. Summarize - User confirms understanding before documentation
4. Get approval - User controls detail level
5. Create - Only after explicit approval

**Why detailed plans matter:**
A detailed plan allows a completely fresh Claude session (without this conversation's context) to implement the issue correctly. This is crucial for async work where the implementing session is different from the planning session.

**Integration with Issue #043:**
- Issue #043: Don't implement during discussion (what NOT to do)
- Issue #045: Discuss before creating issue (what TO do)

Both issues work together to improve the `/create-issue` skill workflow.
