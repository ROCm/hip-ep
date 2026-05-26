/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_COMPILER_TYPES_H
#define HIP_COMPILER_TYPES_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Error codes for compiler operations */
typedef enum {
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
} CompilerErrorCode;

/* Error information (DLL-safe, fixed-size buffer) */
#define COMPILER_ERROR_MESSAGE_SIZE 1024
typedef struct {
  CompilerErrorCode code;
  char message[COMPILER_ERROR_MESSAGE_SIZE];
} CompilerError;

/* Caller-allocated buffers for InferOnnxShapes results returned by
 * `hip_compile_with_fs`. Replaces the previous thread-local stash that
 * was exposed via `hip_get_last_compile_output_*` C ABI accessors —
 * eliminates the thread-local lifetime hazard and the two-call discovery
 * round trips. May be NULL on the call site when the caller does not
 * need the data.
 *
 * Memory ownership: caller allocates `shapes_buf` and `origins_buf`,
 * passes their capacities (in int64 elements), and reads back the
 * `*_needed` fields after the call to detect truncation. The compiler
 * writes the full packed-int64 layout up to `*_capacity` int64s and
 * always reports the full required size in `*_needed` (so the caller
 * can compare and re-allocate if necessary — though re-running compile
 * is expensive; typical callers pre-allocate generously).
 *
 * Shapes layout (matches the prior C-ABI encoding):
 *   [num_outputs,
 *    num_dims_0, d_0_0, ..., d_0_{n-1},
 *    num_dims_1, d_1_0, ..., d_1_{m-1},
 *    ...]
 *
 * Origins layout (triple per dim: arg_idx, dim_idx, mult_bits — the
 * IEEE 754 binary64 bit pattern of the per-Reshape mult composed across
 * the SSA trace):
 *   [num_outputs,
 *    num_dims_0, arg_0_0, dim_0_0, mult_bits_0_0,
 *                arg_0_1, dim_0_1, mult_bits_0_1, ...,
 *    num_dims_1, arg_1_0, dim_1_0, mult_bits_1_0, ...]
 */
typedef struct {
  int64_t *shapes_buf;     /* caller-allocated; may be NULL */
  int64_t shapes_capacity; /* in int64 elements; 0 if shapes_buf is NULL */
  int64_t shapes_needed;   /* OUT: required size; > capacity => truncated */
  int64_t *origins_buf;    /* caller-allocated; may be NULL */
  int64_t origins_capacity;
  int64_t origins_needed;
} CompilationOutputs;

#ifdef __cplusplus
}
#endif

#endif /* HIP_COMPILER_TYPES_H */
