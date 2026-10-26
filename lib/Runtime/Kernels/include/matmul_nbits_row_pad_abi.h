/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
/* Row-pad policy of a MatMulNBits autotune table, as custom_kernels_<arch>
 * exports it (hip_matmul_nbits_u4_row_pads) and the compiler reads it for
 * hip-pad-matmul-nbits-weights. Plain C with no HIP types, so the compiler
 * includes it without the kernels' headers. */
#ifndef HIP_MATMUL_NBITS_ROW_PAD_ABI_H
#define HIP_MATMUL_NBITS_ROW_PAD_ABI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* One int4 weight shape the table lays out with its own row pad. */
typedef struct HipMatmulNBitsRowPad {
  int32_t n;
  int32_t k;
  int32_t block_size;
  int32_t zero_points; /* 1: a zero_points tensor (asymmetric), 0: symmetric */
  int32_t pad_bytes;   /* multiple of 16; 0 keeps the weight dense */
} HipMatmulNBitsRowPad;

/* The row pads of the embedded table for `arch` ("gfx1151"; NULL or "": the
 * running device). Writes *default_pad (the pad for an aliasing weight no
 * entry lists; 0 pads none at load) and the first min(count, capacity)
 * entries, and returns count; -1 when the library has no compatible table for
 * `arch`. */
typedef int (*HipMatmulNBitsRowPadsFn)(const char *arch, int32_t *default_pad,
                                       HipMatmulNBitsRowPad *entries,
                                       int32_t capacity);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // HIP_MATMUL_NBITS_ROW_PAD_ABI_H
