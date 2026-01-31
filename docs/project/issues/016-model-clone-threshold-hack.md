<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #016: Remove dirty_hack_for_model_clone_external_data_threshold

## Metadata
- Status: BACKLOG
- Priority: MEDIUM
- Type: Tech Debt / Refactoring
- Created: 2026-01-31
- Related: Issue #014 (both involve pass-dependent configuration)
- Strategic Goal: Eliminate global state mutations, improve testability

## Description

Remove `dirty_hack_for_model_clone_external_data_threshold()` function that mutates global ENV_PARAM based on pass detection. Replace with clean pass-dependent threshold configuration.

**Current problem:** Global state mutation based on runtime pass detection - hard to test, non-obvious side effects.

**Solution:** Pass threshold as parameter to `model_clone()`, or make it PassContext configuration.

## Problem

**Current implementation (morphizen_compile_model.cpp:1001-1012):**

```cpp
static void dirty_hack_for_model_clone_external_data_threshold(
    const ConfigProto& config_proto) {
  //  check each pass in passes, if vaiml plugin is eanbled , disable the
  //  optimization for model clone.
  for (auto& pass : config_proto.passes()) {
    if (pass.plugin() == ENV_PARAM(XLNX_VAIML_LEVEL_1_NAME)) {
      // effective disable the optimization for model clone.
      ENV_PARAM(XLNX_model_clone_external_data_threshold) = 17179869184;
      break;
    }
  }
}
```

**Called at line 1040:**
```cpp
dirty_hack_for_model_clone_external_data_threshold(
    context->get_config_proto());
```

**Why this is wrong:**

1. **Mutates global state** - Changes `ENV_PARAM(XLNX_model_clone_external_data_threshold)` which is global/thread-local
2. **Side effects** - Function has hidden side effects on global configuration
3. **Pass-dependent configuration** - Threshold depends on which passes will run (VAIML needs all constants cloned)
4. **Tight coupling** - Couples pass detection logic with threshold configuration
5. **Hard to test** - Global state mutation makes unit testing difficult
6. **Non-obvious** - Function name suggests threshold adjustment, but actually does pass plugin detection

**Design flaw:** Configuration that depends on passes should be determined during pass planning, not as global state mutation.

## Solution

**Option 1: Pass threshold as parameter (Preferred)**

```cpp
// morphizen_compile_model.cpp:225-232
static int64_t compute_model_clone_threshold(const ConfigProto& config_proto,
                                              PassContext* context) {
  // Check provider option first (user override)
  auto po_threshold = context->get_provider_option(
      "XLNX_model_clone_external_data_threshold");
  if (po_threshold) {
    return std::stoll(po_threshold.value());
  }

  // Check if VAIML pass is enabled
  for (auto& pass : config_proto.passes()) {
    if (pass.plugin() == ENV_PARAM(XLNX_VAIML_LEVEL_1_NAME)) {
      return 17179869184;  // Large threshold for VAIML (all constants cloned)
    }
  }

  // Default from ENV_PARAM
  return ENV_PARAM(XLNX_model_clone_external_data_threshold);
}

// In compile_onnx_model_2:
int64_t threshold = compute_model_clone_threshold(
    context->get_config_proto(), context.get());
auto cloned_model = morphizen::model_clone(model, threshold);
```

**Option 2: Make it PassContext configuration**

```cpp
class PassContextImp {
  int64_t model_clone_threshold_;  // Computed during initialization

  void compute_model_clone_threshold() {
    // Same logic as Option 1, store result in member variable
  }
};
```

**Benefits:**

- ✅ **No global state mutation** - Pure function computing threshold
- ✅ **Clear dependencies** - Threshold computation logic is explicit
- ✅ **Testable** - Easy to test with different pass configurations
- ✅ **Obvious** - Clear relationship between passes and threshold
- ✅ **Respects user override** - provider_option still takes precedence

## Evidence

**Code locations:**

- morphizen_compile_model.cpp:1001-1012 - `dirty_hack_for_model_clone_external_data_threshold()` definition
- morphizen_compile_model.cpp:1040 - Called before `compile_onnx_model_2()`
- morphizen_compile_model.cpp:225-232 - Where threshold is actually used in `compile_onnx_model_2()`
- morphizen_compile_model.cpp:62-73 - `ENV_PARAM(XLNX_model_clone_external_data_threshold)` definition

**Related ENV_PARAM definition:**

```cpp
#ifdef _WIN32
#  ifdef ENABLE_PYTHON
// Python is only enabled for VAIML compilation on Windows, which requires
// this threshold to be set to a large value so all constants are cloned for the
// compilation.
DEF_ENV_PARAM_2(XLNX_model_clone_external_data_threshold, "17179869184", int64_t)
#  else
// Set the threshold to small value to save memory usage for Windows runtime
// package
DEF_ENV_PARAM_2(XLNX_model_clone_external_data_threshold, "128", int64_t)
#  endif
#else
// Set the threshold to a large value on Linux for VAIML compilation
DEF_ENV_PARAM_2(XLNX_model_clone_external_data_threshold, "17179869184", int64_t)
#endif
```

## Context

**Why VAIML needs large threshold:**

VAIML (Python-based compilation) requires all model constants to be cloned into memory. The threshold controls whether external data is cloned during `model_clone()` operation:

- **Small threshold (128)** - External data stays external (saves memory for runtime)
- **Large threshold (17179869184)** - All data cloned into memory (needed for VAIML Python compilation)

**Current approach:**

1. Default threshold varies by platform (#ifdef ENABLE_PYTHON)
2. Runtime hack checks if VAIML pass is enabled
3. If VAIML enabled, mutate global ENV_PARAM to large value
4. `model_clone()` reads global ENV_PARAM

**Problem:** Pass-dependent configuration shouldn't mutate global state.

## Implementation Steps

**Step 1: Create compute_model_clone_threshold() function**

```cpp
// morphizen_compile_model.cpp - NEW function
static int64_t compute_model_clone_threshold(const ConfigProto& config_proto,
                                              PassContext* context) {
  // Implementation as shown in Solution section
}
```

**Step 2: Update compile_onnx_model_2 to use computed threshold**

```cpp
// morphizen_compile_model.cpp:225-232 - UPDATE
void compile_onnx_model_2(std::shared_ptr<PassContextImp> context,
                          const Graph& onnx_graph) {
  bool cache_hit = check_cache_hit(*context);
  if (!cache_hit) {
    auto& model = graph_get_model(onnx_graph);

    // NEW: Compute threshold without mutating global state
    int64_t threshold = compute_model_clone_threshold(
        context->get_config_proto(), context.get());

    auto cloned_model = morphizen::model_clone(model, threshold);
    // ... rest unchanged
  }
}
```

**Step 3: Remove dirty_hack_for_model_clone_external_data_threshold**

```cpp
// morphizen_compile_model.cpp:1001-1012 - DELETE entire function
```

**Step 4: Remove call site**

```cpp
// morphizen_compile_model.cpp:1040 - DELETE this line
dirty_hack_for_model_clone_external_data_threshold(
    context->get_config_proto());
```

**Step 5: Verify tests pass**

- Ensure VAIML models still compile correctly (threshold computed correctly)
- Ensure non-VAIML models use appropriate threshold
- Ensure provider_option override still works

## Acceptance Criteria

**Implementation:**
- [ ] `compute_model_clone_threshold()` function implemented
- [ ] `compile_onnx_model_2()` updated to use computed threshold
- [ ] `dirty_hack_for_model_clone_external_data_threshold()` function removed
- [ ] Call site at line 1040 removed
- [ ] All tests pass

**Verification:**
- [ ] No global state mutation (ENV_PARAM not modified at runtime)
- [ ] VAIML models compile correctly (threshold = 17179869184)
- [ ] Non-VAIML models use default threshold
- [ ] provider_option override works: `get_provider_option("XLNX_model_clone_external_data_threshold")`
- [ ] Code is more testable (pure function)

## Notes

**Part of broader effort to eliminate global state mutations.**

**Why this is better:**

- **Pure function** - No side effects, easy to reason about
- **Explicit dependencies** - Clear relationship between passes and threshold
- **Testable** - Can test threshold computation with different pass configurations
- **Respects layering** - Threshold computed at the right layer (where passes are known)

**Related to Issue #014:**

Both issues involve pass-dependent configuration:
- Issue #014: Pass selection at runtime (compute effective passes)
- Issue #016: Threshold configuration based on passes (compute threshold)

**Pattern:** Configuration that depends on passes should be computed, not stored or mutated globally.
