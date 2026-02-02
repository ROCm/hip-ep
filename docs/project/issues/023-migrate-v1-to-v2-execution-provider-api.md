<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #023: Migrate from VitisAI v1 API to V2 Execution Provider API

## Metadata
- **Status:** BACKLOG
- **Priority:** CRITICAL
- **Type:** Tech Debt / Refactoring
- **Owner:** TBD
- **Created:** 2026-02-01
- **Dependencies:** None
- **Strategic Goal:** Enable MLIR backend by migrating to the new V2 Execution Provider API

## Description

Migrate all usages of the deprecated `AppendExecutionProvider_VitisAI()` (v1 API) to the new `AppendExecutionProvider_V2()` API with mlir-backend. This blocks all unit tests from running because the v1 API attempts to load `onnxruntime_providers_vitisai.dll` which no longer exists.

## Problem

**Current design/code:**
```cpp
// unit-test/morphizen_unit_test_main.cpp:87
Ort::SessionOptions().AppendExecutionProvider_VitisAI();

// common/initialize_morphizen.hpp
Ort::SessionOptions().AppendExecutionProvider_VitisAI();

// graph-opt/graph-opt.cpp
Ort::SessionOptions().AppendExecutionProvider_VitisAI();
```

**Why this is wrong:**
1. The v1 API `AppendExecutionProvider_VitisAI()` tries to load `onnxruntime_providers_vitisai.dll`
2. This DLL no longer exists in the new mlir-backend architecture
3. All tests crash during initialization with exit code 0xc0000409 (STATUS_STACK_BUFFER_OVERRUN)

**Current behavior:**
- Test executable crashes before any test can run
- Error: `Error loading "D:\ROCm\build\MorphiZen\bin\onnxruntime_providers_vitisai.dll" which is missing. (Error 126)`
- 38+ tests fail due to this initialization failure

## Solution

**Proposed design:**
```cpp
// Replace v1 API:
// Ort::SessionOptions().AppendExecutionProvider_VitisAI();

// With v2 API (default options):
Ort::SessionOptions session_options;
session_options.AppendExecutionProvider_V2("MorphiZen", nullptr, nullptr, 0);
```

**Approach:**
1. Create a helper function `initialize_morphizen_ep()` in `common/initialize_morphizen.hpp`
2. Update all 6 source files to use the new API
3. Update documentation in 2 files
4. Run tests to verify the migration

**Benefits:**
- ✅ Tests will pass with the new mlir-backend
- ✅ Removes dependency on deprecated VitisAI provider DLL
- ✅ Enables use of new MLIR-based execution provider

**Migration path:**
1. Implement helper function with v2 API
2. Replace all v1 API calls
3. Update documentation
4. Verify tests pass

## Evidence

Files requiring code changes:
- `unit-test/morphizen_unit_test_main.cpp:87` - Test initialization
- `common/initialize_morphizen.hpp` - Shared initialization helper
- `graph-opt/graph-opt.cpp` - Graph optimizer tool
- `pattern-gen/onnx_pattern_gen.cpp` - Pattern generator tool
- `onnx-grep/onnx_grep.cpp` - ONNX grep tool
- `unit-test/morphizen-e2e-test/session-options.cpp` - E2E test session options

Files requiring documentation update:
- `docs/architecture.md` - Architecture documentation
- `morphizen-core/include/morphizen/onnxruntime_morphizen_ep.hpp` - API header documentation

## Context

The project is transitioning from the VitisAI execution provider architecture (v1) to a new MLIR-based backend (v2). The v1 API (`AppendExecutionProvider_VitisAI`) was designed for the Xilinx/AMD VitisAI NPU backend, while the v2 API (`AppendExecutionProvider_V2`) is a generic mechanism that supports multiple backends including the new MLIR backend.

This migration is blocking all CI/CD testing as no tests can run until the initialization code is updated.
