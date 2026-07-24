<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #034: MLIR Backend Shape Nullptr Check Failure

## Metadata
- **Type:** Bug
- **Priority:** MEDIUM
- **Dependencies:** None

## Description

MLIR backend encounters shape nullptr when processing certain operations, causing assertion failures. Affects 2 test cases.

## Problem

**Current design/code:**
```cpp
// mlir-graph.cpp:480
CHECK(shape != nullptr);  // Assertion fails

// LLVM Casting.h:650
assert(detail::isPresent(Val) && "dyn_cast on a non-existent value");
```

**Why this is problematic:**
1. Shape information missing or incorrectly set for MLIR operations
2. May be accessing invalid MLIR Value during type conversion
3. Causes CHECK failure or LLVM assertion
4. Affects dynamic operation invocation and model creation functionality

**Code locations:**
- `mlir-imp/src/mlir-graph.cpp:480` - Shape nullptr CHECK
- LLVM `Casting.h:650` - dyn_cast assertion
- `unit-test/morphizen/test_op_invoker.cpp` - OpInvokerTest.CreateAndInvoke
- `unit-test/morphizen/test_model.cpp` - ModelTest.ModelCreationTest

**Error output:**
```
F20260203 01:59:54.530774 mlir-graph.cpp:480] Check failed: shape != nullptr

Assertion failed: detail::isPresent(Val) && "dyn_cast on a non-existent value",
file D:\ROCm\llvm-project\llvm\include\llvm/Support/Casting.h, line 650
```

## Solution

**Investigation needed:**

### Issue 1: OpInvokerTest.CreateAndInvoke

Check code around `mlir-graph.cpp:480`:
```cpp
// Possibly getTensorShape() or similar function
auto shape = getTensorShape(value);
CHECK(shape != nullptr);  // Fails here
```

**Possible causes**:
- MLIR Value doesn't have ranked tensor type
- Dynamic shape information missing
- Type inference failure

### Issue 2: ModelTest.ModelCreationTest

LLVM Casting assertion failure:
```cpp
auto tensor_type = dyn_cast<RankedTensorType>(value.getType());
// dyn_cast fails because value is not valid RankedTensorType
```

**Possible causes**:
- Type not correctly set during MLIR model creation
- Some operations return NoneType or other non-tensor types

**Proposed approaches:**

### Option A: Add Type Checking and Error Handling
```cpp
// mlir-graph.cpp
auto tensor_type = value.getType().dyn_cast<RankedTensorType>();
if (!tensor_type) {
    LOG(WARNING) << "Value does not have a ranked tensor type";
    return default_shape;  // Return default instead of CHECK
}
auto shape = getTensorShape(tensor_type);
```

### Option B: Ensure Correct Type Setting During Operation Creation
```cpp
// Explicitly set output type when creating MLIR operation
builder.create<SomeOp>(loc,
    RankedTensorType::get(shape, elementType),  // Explicit type
    operands);
```

### Option C: Fix Test Case Input
```cpp
// test_op_invoker.cpp or test_model.cpp
// Ensure test provides correct shape information
```

**Recommended:** Option A + Option B (defensive programming + fix root cause)

## Approach

1. Analyze code logic around `mlir-graph.cpp:480`
2. Add type checking, avoid direct CHECK nullptr
3. Review OpInvoker and ModelCreation implementation
4. Ensure MLIR operations correctly set type information during creation
5. Add more detailed error logging
6. Verify 2 tests pass

## Benefits

- ✅ 2 tests restored
- ✅ More robust type handling
- ✅ Better error diagnostics

## Evidence

**Affected tests (2 total):**
- OpInvokerTest.CreateAndInvoke - `Check failed: shape != nullptr`
- ModelTest.ModelCreationTest - `Assertion failed: isPresent(Val)`

**Test results:** Both tests crash with assertion failures

**Notes:**
- This may be related to MLIR version or API changes
- Need to view complete stack trace to locate exact position
- May need to reference MLIR documentation or example code
