<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #018: Make ConfigProto const Member - Enforce Immutability at Compile Time

## Metadata
- **Status:** BACKLOG
- **Priority:** LOW
- **Type:** Tech Debt / Refactoring
- **Dependencies:** Issue #004, #005, #007, #014, #015, #017 (ALL must complete first)
- **Strategic Goal:** Immutable ConfigProto

## Description

Change `ConfigProto config_` to `const ConfigProto config_` in PassContextImp. This is the final step to enforce ConfigProto immutability at compile-time after all mutation sites are eliminated.

## Problem

**After Issue #003 (ConfigProto runtime-only member):**
```cpp
class PassContextImp {
  ContextProto context_proto;  // OUTPUT only
  ConfigProto config_;         // Runtime INPUT (non-const)
};
```

**Why non-const is necessary during migration:**

Issue #003 adds `config_` as non-const to allow incremental fixes:

```cpp
// Issue #005 - needs mutable ConfigProto during migration
*context->config_.mutable_cache_key() = new_cache_key;  // Still mutating

// Issue #007 - needs mutable ConfigProto during migration
context->config_.set_target(*target);  // Still mutating

// Issue #014 - needs mutable ConfigProto during migration
config_.add_passes();  // Still mutating
```

**If we made it const in Issue #003, all these issues would fail to compile!**

**After ALL mutation issues complete:**

After #004, #005, #007, #014, #015, #017 are done:
- ✅ No more cache_key mutations (#005 fixed)
- ✅ No more target mutations (#007 fixed)
- ✅ No more pass mutations (#014 fixed)
- ✅ No more version info mutations (#015 fixed)
- ✅ No more encryption_key mutations (#004 fixed)
- ✅ update_config_by_target() removed (#017 fixed)

**ConfigProto is never mutated after construction → can be const**

## Solution

Change `config_` from non-const to const.

**Step 1: Update PassContextImp member (pass_context_imp.hpp)**

```cpp
// OLD (after Issue #003):
class PassContextImp : public PassContext {
  ContextProto context_proto;
  ConfigProto config_;  // Non-const
};

// NEW (this issue):
class PassContextImp : public PassContext {
  ContextProto context_proto;
  const ConfigProto config_;  // ← CONST enforces immutability
};
```

**Step 2: Verify all access is const-correct**

All existing code should already be const-correct (readonly access):
```cpp
// Read access (already works with const)
auto& passes = config_.passes();
auto target = config_.target();
auto& provider_opts = config_.provider_options();
```

**Step 3: Verify no mutable access exists**

Search for mutable access:
```bash
grep -r "config_\.mutable_" morphizen-core/src/
grep -r "config_\.set_" morphizen-core/src/
grep -r "config_\.add_" morphizen-core/src/
```

**Expected result:** NONE (all mutations eliminated by prerequisites)

**Step 4: Compile and verify**

```bash
cmake --build ../../build/Morphizen --config Debug
```

**If compilation succeeds:** ConfigProto is truly immutable ✓
**If compilation fails:** Found remaining mutation site (prerequisite incomplete)

**Benefits:**
- ✅ **Compile-time enforcement** - Compiler error if anyone tries to mutate
- ✅ **Self-documenting** - Type clearly shows immutability
- ✅ **Prevents accidental mutations** - Future code cannot mutate
- ✅ **Matches strategic goal** - "Immutable ConfigProto"

## Evidence

**Member declaration:**
- pass_context_imp.hpp:~208 - `ContextProto context_proto;`
- pass_context_imp.hpp:~??? - `ConfigProto config_;` (to be added in #003, to be made const in this issue)

**Note:** After Issue #003, the exact line number for `config_` will be known.

## Context

### Why Two-Step Process?

**Step 1 (Issue #003): Add non-const member**
- Allows incremental fixes of mutation sites
- Other issues (#004-#017) can proceed without compiler errors
- Gradual migration path

**Step 2 (This issue): Make it const**
- After all mutations eliminated
- Compile-time verification of immutability
- Final enforcement step

### Why const Member is Better Than Workaround

**Alternative (BAD): Return const reference**
```cpp
class PassContextImp {
  ConfigProto config_;  // Non-const member

  const ConfigProto& get_config() const {  // Return const reference
    return config_;
  }
};
```

**Problem:** Member is still mutable internally, only external access is const.

**This issue (GOOD): const member**
```cpp
class PassContextImp {
  const ConfigProto config_;  // Const member

  const ConfigProto& get_config() const {  // Already const
    return config_;
  }
};
```

**Benefit:** Immutable everywhere (internal AND external). Compiler enforces.

### Part of Strategic Goal: Immutable ConfigProto

This is the **final step** in achieving immutable ConfigProto:

1. ✅ Issue #003: Make ConfigProto runtime-only (not persisted)
2. ✅ Issues #004-#017: Eliminate all mutation sites
3. **This issue (#018):** Enforce immutability at compile-time

**After this issue:** ConfigProto is immutable by design AND by enforcement.

## Acceptance Criteria

**Prerequisites (MUST complete ALL first):**
- [ ] Issue #003 completed (config_ member added as non-const)
- [ ] Issue #004 completed (encryption_key mutations eliminated)
- [ ] Issue #005 completed (cache_key moved to ContextProto)
- [ ] Issue #007 completed (target mutations eliminated)
- [ ] Issue #014 completed (dynamic pass mutations eliminated)
- [ ] Issue #015 completed (version info moved to ContextProto)
- [ ] Issue #017 completed (update_config_by_target removed)

**Implementation:**
- [ ] `config_` changed from non-const to const in pass_context_imp.hpp
- [ ] Code compiles without errors
- [ ] All tests pass

**Verification:**
- [ ] Search for mutable access: `grep -r "config_\.mutable_\|config_\.set_\|config_\.add_" morphizen-core/src/` finds NONE
- [ ] Compiler enforces immutability (attempting mutation causes compilation error)
- [ ] All existing const access still works
- [ ] ConfigProto truly immutable (compile-time verification)

## Notes

### Simple but Critical

**Effort:** 5 minutes (one-line change)

**Impact:** HIGH (enforces entire immutability strategy)

This is a trivial change (adding `const` keyword) but has high strategic value:
- Prevents future regressions (compiler catches mutations)
- Documents design intent in type system
- Completes the immutability journey

### Testing Strategy

**Positive test (should compile):**
```cpp
// Readonly access - should work
auto target = context->config_.target();
auto& passes = context->config_.passes();
```

**Negative test (should NOT compile):**
```cpp
// Try to mutate - should fail compilation
context->config_.set_target("xyz");  // ERROR: cannot call on const
context->config_.add_passes();        // ERROR: cannot call on const
```

**If negative test compiles:** Bug found! Mutation site missed by prerequisites.

### Dependency Chain

This issue is the **leaf node** in the dependency tree:

```
#003 (Add config_ non-const)
  ↓
#004, #005, #007, #014, #015 (Eliminate mutations)
  ↓
#017 (Remove obsolete function)
  ↓
#018 (Make config_ const) ← THIS ISSUE (final step)
```

**Cannot be done until ALL prerequisites complete.**

### Final Verification Command

After this issue completes:

```bash
# Verify ConfigProto is const
grep "const ConfigProto config_" morphizen-core/src/pass_context_imp.hpp

# Verify no mutations exist
grep -r "config_\.mutable_\|config_\.set_\|config_\.add_" morphizen-core/src/

# Both should confirm: ConfigProto is immutable
```

**Success:** ConfigProto immutable by design, immutable by enforcement, immutable by verification. Strategic goal achieved.
