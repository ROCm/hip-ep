<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #026: MLIR Model Export API

## Metadata
- **Status:** BACKLOG
- **Priority:** MEDIUM
- **Type:** Feature
- **Dependencies:** None
- **Related:** #025

## Description

Add a new API `model_save_mlir()` to export MorphiZen internal MLIR Model pointer directly to standard MLIR file format. This feature is only available in MLIR backend; ONNX-IR backend calls return NOT_SUPPORTED.

## Problem

**Current gap:**
1. No direct API to export MLIR format
2. Difficult to inspect intermediate representation when debugging MLIR backend
3. Cannot easily save processed models in MLIR format

**Current behavior:**
- Models can only be saved in ONNX format
- Cannot directly output MLIR IR for debugging or external tool processing

## Solution

**Proposed API:**
```cpp
// morphizen-ort-api-ext.hpp
struct MorphizenOrtApiExt : public morphizen::OrtApiForMorphizen {
  // ... existing members ...

  // Save Model to MLIR file
  // @param model: MorphiZen Model pointer
  // @param filepath: Output file path (.mlir or .mlirbc)
  // @param binary: true=binary format(.mlirbc), false=text format(.mlir)
  // @return: 0=success, non-zero=error code
  // Note: Only valid for MLIR backend, ONNX-IR backend returns NOT_SUPPORTED
  int (*model_save_mlir)(const morphizen::Model* model,
                         const char* filepath,
                         int binary);
};
```

**MLIR backend implementation:**
```cpp
// mlir-imp/src/morphizen-ort-api.cpp
the_mlir_instance_of_morphizen_ort_api.model_save_mlir =
    [](const morphizen::Model* model, const char* filepath, int binary) -> int {
  auto* mlir_model = reinterpret_cast<const mlir_impl::MLIRModel*>(model);

  std::error_code ec;
  llvm::raw_fd_ostream os(filepath, ec);
  if (ec) {
    LOG(ERROR) << "Failed to open file: " << filepath;
    return -1;
  }

  auto& module = mlir_model->get_module();
  if (binary) {
    mlir::BytecodeWriterConfig config;
    if (mlir::failed(mlir::writeBytecodeToFile(module, os, config))) {
      return -2;
    }
  } else {
    module.print(os);
  }
  return 0;
};
```

**ONNX-IR backend implementation:**
```cpp
// onnx-ir-imp/src/morphizen-ort-api.cpp
the_onnx_instance_of_morphizen_ort_api.model_save_mlir =
    [](const morphizen::Model*, const char*, int) -> int {
  LOG(WARNING) << "model_save_mlir is not supported in ONNX-IR backend";
  return -100;  // NOT_SUPPORTED
};
```

**Approach:**
1. Add `model_save_mlir` function pointer to `MorphizenOrtApiExt`
2. Implement export logic in MLIR backend
3. Return NOT_SUPPORTED in ONNX-IR backend
4. Add unit tests to verify functionality

**Benefits:**
- ✅ Easy debugging of MLIR backend intermediate representation
- ✅ Integration with external MLIR toolchain
- ✅ Standard MLIR export capability
- ✅ Clear backend compatibility handling

## Evidence

- `morphizen-ort-api-ext/include/morphizen/morphizen-ort-api-ext.hpp` - API declaration
- `mlir-imp/src/morphizen-ort-api.cpp` - MLIR backend implementation
- `mlir-imp/src/mlir-model.hpp` - MLIRModel class definition
