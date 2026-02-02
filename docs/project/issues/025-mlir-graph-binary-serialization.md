<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #025: MLIR Graph Binary Serialization

## Metadata
- **Status:** BACKLOG
- **Priority:** MEDIUM
- **Type:** Feature
- **Owner:** TBD
- **Created:** 2026-02-02
- **Dependencies:** None
- **Related:** #026

## Description

Change the MLIR Graph save format from MLIR text format (.mlir) to MLIR binary bytecode format (.mlirbc). Binary format significantly improves serialization/deserialization performance and reduces file size.

## Problem

**Current design/code:**
```cpp
// mlir-imp/src/mlir-graph.cpp
void MLIRGraph::save(const std::string& filename, ...) {
  std::error_code ec;
  llvm::raw_fd_ostream os(filename, ec);
  // Uses text format printing
  module_->print(os);  // Outputs .mlir text format
}
```

**Why this is wrong:**
1. Text format parsing is slow - requires full lexical/syntax analysis
2. Text format files are large - no compression, all identifiers stored verbatim
3. Text format serialization has high overhead - requires formatted output

**Current behavior:**
- Saves as .mlir text files
- Reading requires full text parsing
- Large models have long save/load times

## Solution

**Proposed design:**
```cpp
// mlir-imp/src/mlir-graph.cpp
void MLIRGraph::save(const std::string& filename, ..., bool binary = true) {
  std::error_code ec;
  llvm::raw_fd_ostream os(filename, ec);

  if (binary) {
    // Use MLIR bytecode format
    mlir::BytecodeWriterConfig config;
    if (mlir::failed(mlir::writeBytecodeToFile(module_, os, config))) {
      LOG(ERROR) << "Failed to write MLIR bytecode";
    }
  } else {
    // Preserve text format option for debugging
    module_->print(os);
  }
}
```

**Approach:**
1. Modify `MLIRGraph::save()` method to use binary format by default
2. Add `binary` parameter to allow output format selection
3. Update `MLIRModel::load()` to support reading binary format
4. Update related test cases

**Benefits:**
- ✅ Faster read/write speed (expected 3-5x improvement)
- ✅ Smaller file size (expected 30-50% reduction)
- ✅ Reduced serialization/deserialization overhead
- ✅ Preserved text format option for debugging

## Evidence

- `mlir-imp/src/mlir-graph.cpp` - `save()` method
- `mlir-imp/src/mlir-model.cpp` - `load()` method
- MLIR documentation: https://mlir.llvm.org/docs/BytecodeFormat/
