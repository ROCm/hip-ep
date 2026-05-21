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

/**
 * Retrieve the MLIR-refined output shapes from the most recent
 * `hip_compile_with_fs` call on the same thread. Used by the EP-side
 * metadata builder to populate `Output.shape[d]` in metadata.proto for
 * dims that the original ONNX export left symbolic but the
 * `InferOnnxShapes` pre-lowering pass tightened to static values. This
 * is the pass-through channel that lets the EP work on models whose
 * output `dim_param`s don't match any input `dim_param` WITHOUT
 * modifying the ONNX file.
 *
 * Output buffer layout (packed int64):
 *   [num_outputs,
 *    num_dims_0, dim_0_0, dim_0_1, ..., dim_0_{n-1},
 *    num_dims_1, dim_1_0, ..., dim_1_{m-1},
 *    ...]
 * where each dim is positive for a static value and -1 for genuinely
 * dynamic. `num_outputs == 0` means the previous compile produced no
 * outputs or no compile has run on this thread.
 *
 * Two-call discovery pattern (matches dlsym-style ABI design):
 *   1. Call with `out_buffer == NULL` to get the required `num_int64`s
 *      via the return value.
 *   2. Allocate, call again with `out_buffer` sized appropriately.
 * The return value is always the number of int64s the buffer needs;
 * the call writes at most `buffer_size` int64s and returns the full
 * required size so the caller can detect truncation.
 *
 * @param out_buffer  Caller-allocated int64 buffer, or NULL to query
 *                    size.
 * @param buffer_size Capacity of `out_buffer` in int64 elements; ignored
 *                    when `out_buffer == NULL`.
 * @return            Number of int64s required to fully serialize the
 *                    refined shapes (may exceed `buffer_size`).
 */
COMPILER_API int64_t hip_get_last_compile_output_shapes(int64_t *out_buffer,
                                                        int64_t buffer_size);

/**
 * Retrieve per-output-dim SSA origins from the most recent compile on
 * the same thread. Each origin is a `(arg_index, dim_idx)` pair into
 * the function arguments; `(-1, -1)` if the trace couldn't find a
 * passthrough origin. Used by `pass_main.cpp::build_metadata_json` to
 * populate `DimSource.input_idx + dim_idx` for dynamic output dims
 * whose `dim_param` names don't match any input's `dim_param` — i.e.
 * the as-shipped Gemma-3 `vision.onnx` whose output dim_params are
 * `num_image_tokens` / `MatMulimage_features_dim_1` while the input
 * has `num_images`.
 *
 * Output buffer layout (packed int64):
 *   [num_outputs,
 *    num_dims_0, arg_0_0, dim_0_0, arg_0_1, dim_0_1, ...,
 *    num_dims_1, arg_1_0, dim_1_0, ...]
 *
 * Two-call discovery: same as `hip_get_last_compile_output_shapes`.
 *
 * @param out_buffer  Caller-allocated int64 buffer, or NULL to query.
 * @param buffer_size Capacity in int64s; ignored when out_buffer is NULL.
 * @return            Number of int64s required for the full payload.
 */
COMPILER_API int64_t hip_get_last_compile_output_dim_origins(
    int64_t *out_buffer, int64_t buffer_size);

#ifdef __cplusplus
}
#endif

#endif /* HIP_COMPILER_API_H */
