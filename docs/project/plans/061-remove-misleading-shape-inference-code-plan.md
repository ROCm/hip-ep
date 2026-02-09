<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Implementation Plan: Issue #061 - Remove Misleading Shape Inference Code

## Overview

Remove misleading references to shape inference functionality that is non-functional in MorphiZen.

## Step-by-Step Implementation

### Step 1: Delete Commented-Out Test Function

**File:** `ort-bridge/test/src/test-morphizen-ort-implementation.cpp`

**Location:** Lines 1048-1064

**Action:** Delete the entire `Test15_ShapeInferenceOperations()` function

**Before:**
```cpp
void MorphizenOrtApiTest::Test15_ShapeInferenceOperations() {
  // try {
  //   // Test shape inference from file path
  //   std::filesystem::path temp_path =
  //       std::filesystem::temp_directory_path() / "test_inference.onnx";
  //   std::string save_path =
  //       (std::filesystem::temp_directory_path() /
  //       "test_inference_output.onnx")
  //           .string();
  //
  //  // This will likely fail without actual models
  //  // wrapped_api_->graph_infer_shapes_from_filepath(temp_path.string(),
  //  //                                               save_path);
  //} catch (...) {
  //  LOG(INFO) << "Shape inference operations tested";
  //}
}
```

**After:**
```cpp
// Function deleted entirely (lines 1048-1064)
```

**Also check for:**
- Function declaration in class header (search for `Test15_ShapeInferenceOperations`)
- References in test runner code

---

### Step 2: Update resolve() Documentation in graph.hpp

**File:** `morphizen-graph/include/morphizen/graph.hpp`

**Location:** Around line 630

**Action:** Remove "1. Shape inference" from the list of operations

**Before:**
```cpp
  /**
   * @brief Resolves the graph.
   *
   * This function resolves the graph by performing necessary computations and
   * updates.
   *
   * @param force If set to true, the resolution will be forced even if it's
   * not necessary.
   * @return True if the resolution is successful, false otherwise.
   *
   * @note before a graph is properly resolved, some functions like
   * get_consumers get_producer topological_sorted_nodes() are not functional.
   * It  is a heavy calculation includes
   *
   * 1. Shape inference
   * 2. Build edge/node relationship
   * 3. clean up all internal data structure and rebuild everything from
   * sratch.
   * 4. Others
   */
  bool resolve(bool force = false);
```

**After:**
```cpp
  /**
   * @brief Resolves the graph.
   *
   * This function resolves the graph by performing necessary computations and
   * updates.
   *
   * @param force If set to true, the resolution will be forced even if it's
   * not necessary.
   * @return True if the resolution is successful, false otherwise.
   *
   * @note before a graph is properly resolved, some functions like
   * get_consumers get_producer topological_sorted_nodes() are not functional.
   * It is a heavy calculation that includes:
   *
   * 1. Build edge/node relationship
   * 2. Clean up all internal data structure and rebuild everything from
   * scratch.
   * 3. Other backend-specific internal operations
   */
  bool resolve(bool force = false);
```

**Changes:**
- Removed "1. Shape inference"
- Renumbered remaining items
- Fixed typo: "sratch" → "scratch"
- Improved clarity: "Others" → "Other backend-specific internal operations"

---

### Step 3: Update Test Documentation (If Needed)

**Check these files for references to Test15:**

1. **`ort-bridge/test/src/README-test-runner.md`**
   - Search for "Test15" or "ShapeInferenceOperations"
   - Remove or update any references

2. **`ort-bridge/test/src/SUMMARY-comprehensive-tests.md`**
   - Search for "Test15" or "ShapeInferenceOperations"
   - Remove or update any references

**Action:** If found, remove the test from the list of tests.

---

### Step 4: Search for Other References

**Global search for:**
```bash
grep -r "Test15_ShapeInferenceOperations" .
grep -r "Test15" ort-bridge/test/
```

**Remove any remaining references to this test.**

---

## Verification Steps

1. **Build test:**
   ```bash
   cmake --build ../../build/$(basename $PWD) --config Debug --target morphizen-unit-tests
   ```

2. **Check for compilation errors** (function declaration mismatches, etc.)

3. **Run tests:**
   ```bash
   ../../build/$(basename $PWD)/bin/morphizen-unit-tests.exe
   ```

4. **Verify test suite still passes** without Test15

5. **Check documentation:**
   - Read updated `graph.hpp` documentation
   - Ensure it makes sense without shape inference mention

---

## Success Criteria

- [ ] `Test15_ShapeInferenceOperations()` function deleted
- [ ] Function declaration removed (if separate)
- [ ] `graph.hpp` resolve() documentation updated (no mention of shape inference)
- [ ] Test documentation files updated (README, SUMMARY)
- [ ] No compilation errors
- [ ] All remaining tests pass
- [ ] Code is clearer and doesn't mislead about shape inference support

---

## Estimated Effort

**Time:** 30-60 minutes
**Complexity:** Simple
**Risk:** Low (only removing dead/commented code and updating docs)
