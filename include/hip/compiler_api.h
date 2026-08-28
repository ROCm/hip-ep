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
 * @param input_mlir      Input MLIR data (text or bytecode)
 * @param input_size      Size of input data in bytes
 * @param output_path     Output DLL/object/IR file path
 * @param options_json    Compilation options as JSON string (can be NULL)
 * @param error           Error information output (can be NULL)
 * @param fs              morphizen::FileSystem* (cast to void* for C ABI)
 * @return                COMPILER_SUCCESS or error code
 */
COMPILER_API CompilerErrorCode hip_compile_with_fs(
    const void *input_mlir, size_t input_size, const char *output_path,
    const char *options_json, CompilerError *error, void *fs);

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
