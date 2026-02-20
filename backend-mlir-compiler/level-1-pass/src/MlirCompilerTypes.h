// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>

// Local type definitions for morphizen-mlir-compiler C API
// NOTE: These must stay in sync with morphizen-mlir-compiler/compiler_types.h
// to ensure ABI compatibility across the DLL boundary.

namespace mlir_compiler_local {

// Error codes - must match morphizen-mlir-compiler/compiler_types.h
enum CompilerErrorCode {
    COMPILER_SUCCESS = 0,
    COMPILER_ERROR_INVALID_INPUT = 1,
    COMPILER_ERROR_PARSE_FAILED = 2,
    COMPILER_ERROR_PASS_FAILED = 3,
    COMPILER_ERROR_TRANSLATION_FAILED = 4,
    COMPILER_ERROR_LINKING_FAILED = 5,
    COMPILER_ERROR_IO_ERROR = 6,
    COMPILER_ERROR_COMPILATION_FAILED = 7,
    COMPILER_ERROR_VALIDATION_FAILED = 8,
    COMPILER_ERROR_INTERNAL = 99
};

// Error struct - must match binary layout in morphizen-mlir-compiler/compiler_types.h
struct CompilerError {
    CompilerErrorCode code;
    char message[1024];  // Fixed-size buffer for DLL safety
};

// JSON options format (documentation only, not a type):
// Example: {"opt_level": 2, "output_mode": "OUTPUT_MODE_DLL"}
// Fields:
//   - opt_level: 0-3 (default: 2)
//   - output_mode: "OUTPUT_MODE_DLL" | "OUTPUT_MODE_LLVM_IR" (default: DLL)

} // namespace mlir_compiler_local
