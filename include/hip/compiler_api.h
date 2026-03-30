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
 * Compilation entry point with external constant storage.
 *
 * Compiles MLIR input to DLL/object/IR file. onnx.Constant data is written
 * to "constants.bin" via the provided FileSystem. The DLL contains only code;
 * all weight data lives in the external file.
 *
 * constants.bin format: raw concatenated bytes of each constant in
 * discovery order. Sizes are hardcoded in the generated inference_init.
 *
 * The generated inference_init signature becomes:
 *   int inference_init(void** out_state, void* fs)
 * where fs is a morphizen::FileSystem* passed as void* for C ABI.
 *
 * @param input_mlir   Input MLIR data (text or bytecode)
 * @param input_size   Size of input data in bytes
 * @param output_path  Output DLL/object/IR file path
 * @param options_json Compilation options as JSON string (can be NULL)
 * @param error        Error information output (can be NULL)
 * @param fs           morphizen::FileSystem* (cast to void* for C ABI).
 *                     Used to create "constants.bin" for writing.
 * @return             COMPILER_SUCCESS or error code
 */
COMPILER_API CompilerErrorCode hip_compile_with_fs(
    const void *input_mlir, size_t input_size, const char *output_path,
    const char *options_json, CompilerError *error, void *fs);

/**
 * Compilation entry point with zero-copy constant data.
 *
 * Same as hip_compile_with_fs, but constant data is passed directly via
 * an array of HipConstantRef instead of being embedded in the MLIR bytecode.
 * The MLIR input contains onnx.Constant ops with "hip.constant_ref"
 * attributes (name only, no DenseElementsAttr). The compiler resolves
 * each name against the provided constant_refs array.
 *
 * @param input_mlir      Input MLIR data (text or bytecode)
 * @param input_size      Size of input data in bytes
 * @param output_path     Output DLL/object/IR file path
 * @param options_json    Compilation options as JSON string (can be NULL)
 * @param error           Error information output (can be NULL)
 * @param fs              morphizen::FileSystem* (cast to void* for C ABI)
 * @param constant_refs   Array of HipConstantRef (name + data pointer + size)
 * @param num_constants   Number of entries in constant_refs
 * @return                COMPILER_SUCCESS or error code
 */
COMPILER_API CompilerErrorCode hip_compile_with_constants(
    const void *input_mlir, size_t input_size, const char *output_path,
    const char *options_json, CompilerError *error, void *fs,
    const void *constant_refs, size_t num_constants);

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
