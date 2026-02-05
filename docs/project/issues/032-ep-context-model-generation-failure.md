<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #032: EP Context Model Generation Failure Causing E2E Test Failures

## Metadata
- **Type:** Bug
- **Priority:** MEDIUM
- **Dependencies:** #028, #030, #031 (potentially related)

## Description

ONNX Runtime graph partitioner unable to compile any nodes, resulting in EP context model file not being generated, causing E2E tests to fail.

## Problem

**Current design/code:**
```cpp
// E2E test flow:
// 1. Create session with context_enable=1
//    → Should generate context model (e.g., "single_session_gen_and_run_embed_ctx.onnx")
// 2. Create session to load context model
//    → Fails: file doesn't exist
```

**Why this is problematic:**
1. ONNX Runtime warning: `Unable to compile any nodes`
2. Expected EP context model file not generated
3. Second session cannot load non-existent context model
4. E2E test workflow interrupted

**Code locations:**
- `unit-test/morphizen-e2e-test/session.cpp:97` - Check model exists
- Graph partitioner warning from ONNX Runtime internal

**Error output:**
```
2026-02-03 01:59:54.6829417 [W:onnxruntime:, graph_partitioner.cc:819] Unable to compile any nodes. ONNX Runtime will not generate a compiled model. Either the session EPs do not support compilation or the model is already compiled.

ONNXRuntime session create :  107609 us
Running model...
ONNXRuntime session run :  37077
done

F20260203 01:59:54.746500 session.cpp:97] Check failed: std::filesystem::exists(model_path)
Model path does not exist: "single_session_gen_and_run_embed_ctx.onnx"
```

## Solution

**Root cause analysis:**

Possible causes (need sequential investigation):

1. **MLIR backend not properly implementing node compilation interface** (related to #028)
   - ONNX Runtime calling EP's `GetCapability()` returns empty
   - No nodes assigned to MorphiZen EP

2. **Target configuration missing** (related to #031)
   - Cannot find valid compilation target
   - Compilation flow exits early

3. **Plugin not loaded** (related to #030)
   - Required Pass plugin missing
   - Compilation flow cannot complete

4. **E2E test configuration issue**
   - Incorrect provider options
   - Context mode misconfiguration

**Proposed approach:**

### Step 1: Verify EP Registration and Capability
```cpp
// Check if GetCapability() returns nodes
// ort-bridge/src/morphizen-ep.cpp
```

### Step 2: Fix Dependency Issues
- First resolve Issue #028 (MLIR model support)
- First resolve Issue #030 (plugin loading)
- First resolve Issue #031 (Target configuration)

### Step 3: Verify Context Generation Logic
```cpp
// Confirm CreateEpContextModel is correctly called
// Confirm compilation flow executes completely
```

## Approach

1. **Dependencies first**: Resolve #028, #030, #031 first
2. **Debug EP capability**: Verify MorphiZen EP correctly declares node support
3. **Trace compilation flow**: Add logging to view compilation stage status
4. **Verify Context generation**: Confirm EP context file is created
5. Verify 4 E2E tests pass

## Benefits

- ✅ 4 E2E tests restored
- ✅ EP context functionality integrity verification
- ✅ End-to-end compilation and inference flow verification

## Evidence

**Affected tests (4 total):**
- MorphizenE2ETestSuite/MorphizenE2ETest.RunE2ETests/single_session_gen_and_run_embed_ctx
- MorphizenE2ETestSuite/MorphizenE2ETest.RunE2ETests/multiple_session_gen_and_run_embed_ctx
- MorphizenE2ETestSuite/MorphizenE2ETest.RunE2ETests/single_session_gen_and_run_non_embed_no_prefix_ctx
- (Note: v2 test has different error - see #033)

**Test results:**
- Session creation succeeds (using CPU EP fallback)
- Model inference succeeds
- But EP context model file not generated
- Second session cannot load non-existent file

**Notes:**
- This is a compound issue, may need to fix multiple dependency Issues
- Priority set to MEDIUM because other issues need to be resolved first
- May need ONNX Runtime side debugging support
