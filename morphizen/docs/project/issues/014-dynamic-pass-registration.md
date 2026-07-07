<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #014: Dynamic Pass Registration at Runtime

## Metadata
- Status: BACKLOG
- Priority: MEDIUM
- Type: Tech Debt / Refactoring / Architecture
- Updated: 2026-01-31 (Solution designed after discussion)
- Related: Issue #003 (ConfigProto immutability)
- Strategic Goal: Immutable ConfigProto

## Description

Passes added dynamically to ConfigProto via add_passes() at runtime - violates ConfigProto immutability goal.

**Solution:** Compute effective pass list on-demand instead of mutating ConfigProto.

## Problem

**Current design (BAD - mutates ConfigProto):**

**Pattern 1: Target-based pass selection (config.cpp:110-115)**
```cpp
// TargetProto has list of pass names to run
for (auto pass : target_proto->pass()) {
  auto new_pass = proto.add_passes();  // MUTATES ConfigProto!
  new_pass->CopyFrom(pass_from_library);
}
```

**Pattern 2: Plugin-based anonymous pass (pass_imp.cpp:453-456)**
```cpp
auto pass_proto = context_proto.mutable_config()->add_passes();  // MUTATES ConfigProto!
pass_proto->set_name("annonymous_pass");
pass_proto->set_plugin("<annonymous_plugin>");
```

### Historical Context

**Original design:**
- ConfigProto.passes = simple list of passes to run
- Just execute what's specified

**Later evolution (multiple compiler backends):**
- ConfigProto.passes = ALL available passes (library of passes)
- TargetProto.passes = list of pass NAMES (references which passes to use)
- Target-based selection copies passes from library into ConfigProto

**Problem:**
- Pattern 2 (pass_imp.cpp:453-456) is OLD code from before TargetProto.passes existed
- It wasn't updated when design changed to TargetProto.passes approach
- Both patterns still mutate ConfigProto instead of computing effective pass list

**Design flaw:** ConfigProto should be immutable INPUT (pass library), but passes are mutated at runtime based on target selection and plugin loading.

## Solution

**Compute effective pass list on-demand** instead of mutating ConfigProto.

### Architecture

```cpp
// config.proto - Pass library (immutable)
message ConfigProto {
  repeated PassProto passes = 1;  // Library of ALL available passes (NEVER mutated at runtime)
}
```

**IMPORTANT: Add proto comment!**
```protobuf
message ConfigProto {
  // Library of all available passes.
  // At runtime, target selection (TargetProto.pass) determines which passes to execute.
  // This field is immutable after initialization - never use add_passes() at runtime.
  repeated PassProto passes = 1;
}
```

### Implementation

```cpp
class PassContextImp {
  ConfigProto config_;  // Immutable pass library

  // Compute passes to run (called once during compilation)
  std::vector<PassProto> compute_effective_passes() const {
    std::vector<PassProto> result;

    // Target-based selection
    for (auto pass_name : target_proto_->pass()) {
      const PassProto* pass = find_pass_by_name(config_.passes(), pass_name);
      CHECK(pass != nullptr) << "Pass not found in library: " << pass_name;
      result.push_back(*pass);
    }

    // Plugin anonymous passes (if needed)
    if (needs_anonymous_pass) {
      PassProto anonymous_pass;
      anonymous_pass.set_name("anonymous_pass");
      anonymous_pass.set_plugin("<plugin_name>");
      result.push_back(anonymous_pass);
    }

    return result;  // Local variable (not stored)
  }

  void compile() {
    auto effective_passes = compute_effective_passes();  // Compute once when needed
    auto passes = IPass::create_passes(this, effective_passes);
    IPass::run_passes(passes, graph);
  }
};
```

### Implementation Steps

**Step 1: Add compute_effective_passes() method**
```cpp
// pass_context_imp.hpp
std::vector<PassProto> compute_effective_passes() const;

// pass_context_imp.cpp
std::vector<PassProto> PassContextImp::compute_effective_passes() const {
  // Implementation above
}
```

**Step 2: Update config.cpp target-based selection**
```cpp
// OLD (config.cpp:110-115) - DELETE
static void add_target_pass(ConfigProto& proto, ...) {
  for (auto pass : target_proto->pass()) {
    auto new_pass = proto.add_passes();  // BAD - mutates
  }
}

// NEW - No mutation, just use compute_effective_passes()
// This function can be removed - logic moves to compute_effective_passes()
```

**Step 3: Update pass_imp.cpp plugin-based pass**
```cpp
// OLD (pass_imp.cpp:453-456) - UPDATE
auto pass_proto = context_ptr->context_proto.mutable_config()->add_passes();  // BAD
pass_proto->set_name("annonymous_pass");

// NEW - Build PassProto locally, don't mutate ConfigProto
// This logic moves into compute_effective_passes()
```

**Step 4: Update compilation flow**
```cpp
// Use compute_effective_passes() instead of config_.passes() directly
void compile() {
  auto effective_passes = compute_effective_passes();
  auto passes = IPass::create_passes(this, effective_passes);
  IPass::run_passes(passes, graph);
}
```

**Step 5: Add proto comment (REQUIRED)**
```protobuf
// config.proto
message ConfigProto {
  // Library of all available passes.
  // At runtime, target selection (TargetProto.pass) determines which passes to execute.
  // This field is immutable after initialization - never use add_passes() at runtime.
  repeated PassProto passes = 1;
}
```

## Evidence

**Code locations:**
- config.cpp:110-115 - add_target_pass() (target-based selection - MUTATES)
- pass_imp.cpp:453-456 - create_pass() (plugin-based creation - MUTATES)
- config.cpp:130+ - update_config_by_target() (calls add_target_pass)

**Related proto:**
- config.proto:X - PassProto definition
- config.proto:Y - ConfigProto.passes field

## Acceptance Criteria

**Implementation:**
- [ ] compute_effective_passes() method implemented in PassContextImp
- [ ] Target-based selection updated (no proto.add_passes())
- [ ] Plugin-based pass updated (no proto.add_passes())
- [ ] Compilation flow uses compute_effective_passes()
- [ ] Proto comment added to ConfigProto.passes field
- [ ] All tests pass

**Verification:**
- [ ] ConfigProto.passes never mutated after initialization
- [ ] grep for "add_passes()" shows no runtime mutations
- [ ] Target-based pass selection works correctly
- [ ] Plugin anonymous passes work correctly
- [ ] Proto comment clearly documents immutability

## Sessions

### 2026-01-31: Solution Design Discussion

**Context:** Discussed how to eliminate ConfigProto mutations from dynamic pass registration.

**Key questions explored:**

**Q1: When does target-based pass selection happen?**
- A: During target auto-discovery (when hardware detected, target-specific passes selected)

**Q2: When does plugin-based pass creation happen?**
- A: This is OLD code from before TargetProto.passes existed
- Not properly updated when design evolved to support multiple backends
- Still used, but bad implementation

**Q3: Historical context - how did design evolve?**
- Original: ConfigProto.passes = simple list to run
- Later: ConfigProto.passes = library of ALL passes
- Later: TargetProto.passes = names of passes to run (indirect reference)
- Problem: Pattern 2 (anonymous pass) wasn't updated to match this evolution

**Q4: Should effective_passes_ be member variable or local variable?**
- Discussion: Member variable (compute once, store) vs local variable (compute when needed)
- Conclusion: **Local variable** - pass list only needed once during compilation
- Benefits: Simpler, no storage, compute once when needed

**Q5: Must we add proto comments?**
- User: "BTW 'Add comment to proto for clarity:' this sounds great. we must have good comment."
- **CRITICAL: Proto comment is REQUIRED to document immutability**

**Solution decided:**
```cpp
std::vector<PassProto> compute_effective_passes() const {
  // Build from target selection + plugins
  // Return local variable (not stored)
}
```

**Benefits:**
- ✅ ConfigProto.passes never mutated (immutable library)
- ✅ No extra member variable needed
- ✅ Computed once when needed
- ✅ Simple and clean

## Notes

**Part of strategic goal:** Achieving immutable ConfigProto.

**Key design principle:**
- ConfigProto.passes = **library** (all available passes, immutable)
- Runtime pass list = **computed** (based on target + plugins, not stored)
- Clear separation: library vs execution list

**Why local variable, not member variable:**
- Effective pass list only needed once during compilation
- No need to store it as member variable
- Simpler: compute when needed, return

**CRITICAL: Proto comments required:**
- ConfigProto.passes must have clear comment explaining it's a library
- Prevents future developers from mutating it
- Self-documenting code

**Related issues:**
- Issue #003: ConfigProto becomes runtime-only member (natural fit with this design)
- Issue #015: Version info mutations (similar pattern - should be computed, not mutated)
