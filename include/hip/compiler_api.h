/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_COMPILER_API_H
#define HIP_COMPILER_API_H

#include "compiler_types.h"

/* Export macro for DLL visibility */
#ifdef _WIN32
#ifdef HIP_COMPILER_EXPORTS
#define COMPILER_API __declspec(dllexport)
#else
#define COMPILER_API __declspec(dllimport)
#endif
#else
#define COMPILER_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Compile MLIR input to DLL/object/IR file.
 *
 * onnx.Constant data is written to "constants.bin" via the provided
 * FileSystem. External constants are identified by the `location`
 * attribute on onnx.Constant ops in the MLIR bytecode.
 *
 * `outputs` (may be NULL) is the channel by which MLIR-refined output
 * shapes and per-output-dim SSA origins flow back to the caller. This
 * replaces the previous thread-local stash that was exposed via
 * `hip_get_last_compile_output_*` accessors — no thread-local lifetime
 * to manage, no follow-up calls, single source of truth: the data is
 * captured inside `runMLIRPasses` while the function is still
 * `func::FuncOp`, written into module attributes that survive
 * `func.func → llvm.func` conversion, then read back here and packed
 * into the caller-allocated buffers in `outputs`. See
 * `compiler_types.h::CompilationOutputs` for the buffer layout.
 *
 * @param input_mlir      Input MLIR data (text or bytecode)
 * @param input_size      Size of input data in bytes
 * @param output_path     Output DLL/object/IR file path
 * @param options_json    Compilation options as JSON string (can be NULL)
 * @param error           Error information output (can be NULL)
 * @param fs              morphizen::FileSystem* (cast to void* for C ABI)
 * @param outputs         InferOnnxShapes results out-channel (can be NULL)
 * @return                COMPILER_SUCCESS or error code
 */
COMPILER_API CompilerErrorCode hip_compile_with_fs(
    const void *input_mlir, size_t input_size, const char *output_path,
    const char *options_json, CompilerError *error, void *fs,
    CompilationOutputs *outputs);

/**
 * Get compiler version string.
 *
 * @return Static version string (e.g., "1.0.0")
 */
COMPILER_API const char *hip_get_version(void);

#ifdef __cplusplus
}
#endif

#endif /* HIP_COMPILER_API_H */
