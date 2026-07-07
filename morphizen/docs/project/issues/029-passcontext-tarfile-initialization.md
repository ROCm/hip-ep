<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #029: PassContext Test Cases Missing tar_file_ Initialization

## Metadata
- **Type:** Bug
- **Priority:** MEDIUM
- **Dependencies:** None

## Description

Instances created by `PassContext::create()` do not properly initialize the `tar_file_` member, causing 3 PassContextTest test cases to fail.

## Problem

**Current design/code:**
```cpp
// test_pass_context.cpp
void SetUp() override {
    passContext = morphizen::PassContext::create();
    // tar_file_ is not initialized
}

TEST_F(PassContextTest, ReadFileTest) {
    passContext->write_file(filename, data);  // Internally uses tar_file_
}
```

**Why this is problematic:**
1. `PassContext::create()` creates an "empty" context, but many methods depend on `tar_file_` member
2. In normal compilation flow, `tar_file_` is initialized in `initialize_context()`
3. Unit tests using `create()` directly bypass the complete initialization flow
4. Causes CHECK failure: `tar_file_ != nullptr`

**Code locations:**
- `morphizen-core/src/pass_context_imp.cpp:514` - CHECK failure point
- `morphizen-core/src/pass_context_imp.cpp:469` - `open_file_for_read()` call site
- `unit-test/morphizen/test_pass_context.cpp:32` - Test SetUp() creation point

**Error output:**
```
F20260203 01:59:52.415457 pass_context_imp.cpp:514] Check failed: tar_file_ != nullptr tar_file_ should always exist
```

## Solution

**Proposed approaches:**

### Option A: Modify PassContext::create() Default Behavior
```cpp
// PassContext.hpp
static std::unique_ptr<PassContext> create() {
    auto context = std::make_unique<PassContextImp>();
    context->tar_file_ = TarFile::create();  // Add default initialization
    return context;
}
```

**Pros**:
- Simple and direct, benefits all users of create()
- Minimal changes

**Cons**:
- May not align with "empty context" semantics
- May affect other use cases

### Option B: Add Test Helper Function
```cpp
// test_environment.hpp
static std::unique_ptr<PassContext> create_test_context() {
    auto context = PassContext::create();
    // Initialize tar_file_ for testing
    context->initialize_tar_file();  // New public method
    return context;
}

// test_pass_context.cpp
void SetUp() override {
    passContext = create_test_context();
}
```

**Pros**:
- Does not affect production code
- Clear test intent

**Cons**:
- Need to expose initialization method in PassContext
- Additional test helper code

### Option C: Use Complete Initialization Flow
```cpp
// test_pass_context.cpp
void SetUp() override {
    // Use real initialization path
    auto model = Model::load(RESNET_50_PATH);
    auto graph = model->main_graph();
    passContext = morphizen::initialize_context(
        model_path, graph, {}, provider_options, logger);
}
```

**Pros**:
- Tests real usage scenario
- No special handling needed

**Cons**:
- Test depends on more components
- May not suit pure PassContext unit tests

**Recommended:** Option B

## Approach

1. Add `initialize_tar_file()` public method in `PassContextImp`
2. Add `create_test_context()` helper function in `test_environment.hpp`
3. Modify `test_pass_context.cpp` SetUp() to use new function
4. Verify 3 tests pass

## Benefits

- ✅ 3 PassContextTest tests restored
- ✅ Test code more clearly expresses intent
- ✅ Does not affect production code API design

## Evidence

**Affected tests (3 total):**
- PassContextTest.ReadFileTest
- PassContextTest.UntarCacheTest
- PassContextTest.TestEmptyFiles

**Test results:** All 3 tests fail with `tar_file_ != nullptr` CHECK failure
