/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Q4_K, Q5_K, Q6_K, and Q8_1 storage layouts are adapted from
// ggml-common.h in llama.cpp at commit
// b21e4de74567f5eef213765c9476a843c2e43f0d. llama.cpp is Copyright (c)
// 2023-2026 The ggml authors and is distributed under the MIT License; see
// lib/Runtime/Kernels/licenses/llama.cpp/LICENSE.

#ifndef HIP_KQUANT_ABI_H
#define HIP_KQUANT_ABI_H

// The K-quant and Q8_1 block layouts the kquant kernels and their runtime
// wrappers share, usable from C and C++. The compiler's view of the same
// layouts is include/hip/Dialect/IR/HipKQuant.h; the two must agree.

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  HIP_KQUANT_FORMAT_Q4_K = 0,
  HIP_KQUANT_FORMAT_Q6_K = 1,
  HIP_KQUANT_FORMAT_Q5_K = 2,
} hip_kquant_format_t;

typedef enum {
  HIP_Q8_1_SUM_PREQUANTIZATION = 0,
  HIP_Q8_1_SUM_SCALE_TIMES_QUANTIZED = 1,
  // Multi-token inputs store d * sum(q) as F16, as llama.cpp's Q8_1 and MMQ
  // producers do: the minimum term of Q4_K/Q5_K then sees the same quantized
  // activation as the product, which the pre-quantization sum does not under
  // large activations. Single-token inputs store the exact signed sum(q)
  // integer bits in the same 16-bit field. The logical token count is total
  // prepared rows divided by rows_per_token. This is a prepared layout, not
  // canonical Q8_1.
  HIP_Q8_1_SUM_MATRIX_SCALED_VECTOR_INTEGER_SUM = 2,
} hip_q8_1_sum_mode_t;

#define HIP_KQUANT_BLOCK_ELEMENTS 256
#define HIP_KQUANT_Q4_K_SCALE_BYTES 12
#define HIP_KQUANT_Q4_K_BLOCK_SIZE 144
#define HIP_KQUANT_Q4_K_BLOCK_ALIGNMENT 4
#define HIP_KQUANT_Q5_K_BLOCK_SIZE 176
#define HIP_KQUANT_Q5_K_BLOCK_ALIGNMENT 4
#define HIP_KQUANT_Q6_K_BLOCK_SIZE 210
#define HIP_KQUANT_Q6_K_BLOCK_ALIGNMENT 2
#define HIP_Q8_1_BLOCK_ELEMENTS 32
#define HIP_Q8_1_BLOCK_SIZE 36
#define HIP_Q8_1_BLOCK_ALIGNMENT 4

// MMQ tiles (kquant_mmq_kernel.hip).
#define HIP_KQUANT_MMQ_OUTPUT_TILE 128
#define HIP_KQUANT_MMQ_INPUT_TILE 128
#define HIP_KQUANT_MMQ_MIN_INPUT_ROWS 9
#define HIP_KQUANT_MMQ_REDUCTION_TILE 256
#define HIP_KQUANT_MMQ_Q8_BLOCKS_PER_TILE 4
#define HIP_KQUANT_MMQ_F32_ALIGNMENT 16

// Head dimensions hip_token_major_gqa has a tile kernel for.
static inline int hip_token_major_gqa_head_dimension_valid(int64_t head_dimension) {
  return head_dimension == 40 || head_dimension == 64 || head_dimension == 72 ||
         head_dimension == 80 || head_dimension == 96 ||
         head_dimension == 112 || head_dimension == 128 ||
         head_dimension == 256;
}

#if defined(__cplusplus)
#define HIP_KQUANT_ALIGNAS(value) alignas(value)
#define HIP_KQUANT_ALIGNOF(type) alignof(type)
#define HIP_KQUANT_STATIC_ASSERT(condition, message)                             \
  static_assert(condition, message)
#elif defined(_MSC_VER)
#define HIP_KQUANT_ALIGNAS(value) __declspec(align(value))
#define HIP_KQUANT_ALIGNOF(type) __alignof(type)
#define HIP_KQUANT_STATIC_ASSERT(condition, message)                             \
  static_assert(condition, message)
#else
#define HIP_KQUANT_ALIGNAS(value) _Alignas(value)
#define HIP_KQUANT_ALIGNOF(type) _Alignof(type)
#define HIP_KQUANT_STATIC_ASSERT(condition, message)                             \
  _Static_assert(condition, message)
#endif

typedef struct {
  HIP_KQUANT_ALIGNAS(HIP_KQUANT_Q4_K_BLOCK_ALIGNMENT) uint16_t d;
  uint16_t dmin;
  uint8_t scales[HIP_KQUANT_Q4_K_SCALE_BYTES];
  uint8_t qs[HIP_KQUANT_BLOCK_ELEMENTS / 2];
} hip_kquant_q4_k_block_t;

typedef struct {
  HIP_KQUANT_ALIGNAS(HIP_KQUANT_Q5_K_BLOCK_ALIGNMENT) uint16_t d;
  uint16_t dmin;
  uint8_t scales[HIP_KQUANT_Q4_K_SCALE_BYTES];
  uint8_t qh[HIP_KQUANT_BLOCK_ELEMENTS / 8];
  uint8_t qs[HIP_KQUANT_BLOCK_ELEMENTS / 2];
} hip_kquant_q5_k_block_t;

typedef struct {
  uint8_t ql[HIP_KQUANT_BLOCK_ELEMENTS / 2];
  uint8_t qh[HIP_KQUANT_BLOCK_ELEMENTS / 4];
  int8_t scales[HIP_KQUANT_BLOCK_ELEMENTS / 16];
  uint16_t d;
} hip_kquant_q6_k_block_t;

typedef struct {
  HIP_KQUANT_ALIGNAS(HIP_Q8_1_BLOCK_ALIGNMENT) uint16_t d;
  uint16_t s;
  int8_t qs[HIP_Q8_1_BLOCK_ELEMENTS];
} hip_q8_1_block_t;

HIP_KQUANT_STATIC_ASSERT(sizeof(hip_kquant_q4_k_block_t) ==
                           HIP_KQUANT_Q4_K_BLOCK_SIZE,
                       "Q4_K ABI block size mismatch");
HIP_KQUANT_STATIC_ASSERT(HIP_KQUANT_ALIGNOF(hip_kquant_q4_k_block_t) ==
                           HIP_KQUANT_Q4_K_BLOCK_ALIGNMENT,
                       "Q4_K ABI block alignment mismatch");
HIP_KQUANT_STATIC_ASSERT(offsetof(hip_kquant_q4_k_block_t, d) == 0,
                       "Q4_K ABI d offset mismatch");
HIP_KQUANT_STATIC_ASSERT(offsetof(hip_kquant_q4_k_block_t, dmin) == 2,
                       "Q4_K ABI dmin offset mismatch");
HIP_KQUANT_STATIC_ASSERT(offsetof(hip_kquant_q4_k_block_t, scales) == 4,
                       "Q4_K ABI scales offset mismatch");
HIP_KQUANT_STATIC_ASSERT(offsetof(hip_kquant_q4_k_block_t, qs) == 16,
                       "Q4_K ABI quants offset mismatch");

HIP_KQUANT_STATIC_ASSERT(sizeof(hip_kquant_q5_k_block_t) ==
                           HIP_KQUANT_Q5_K_BLOCK_SIZE,
                       "Q5_K ABI block size mismatch");
HIP_KQUANT_STATIC_ASSERT(HIP_KQUANT_ALIGNOF(hip_kquant_q5_k_block_t) ==
                           HIP_KQUANT_Q5_K_BLOCK_ALIGNMENT,
                       "Q5_K ABI block alignment mismatch");
HIP_KQUANT_STATIC_ASSERT(offsetof(hip_kquant_q5_k_block_t, d) == 0,
                       "Q5_K ABI d offset mismatch");
HIP_KQUANT_STATIC_ASSERT(offsetof(hip_kquant_q5_k_block_t, dmin) == 2,
                       "Q5_K ABI dmin offset mismatch");
HIP_KQUANT_STATIC_ASSERT(offsetof(hip_kquant_q5_k_block_t, scales) == 4,
                       "Q5_K ABI scales offset mismatch");
HIP_KQUANT_STATIC_ASSERT(offsetof(hip_kquant_q5_k_block_t, qh) == 16,
                       "Q5_K ABI high-quants offset mismatch");
HIP_KQUANT_STATIC_ASSERT(offsetof(hip_kquant_q5_k_block_t, qs) == 48,
                       "Q5_K ABI low-quants offset mismatch");

HIP_KQUANT_STATIC_ASSERT(sizeof(hip_kquant_q6_k_block_t) ==
                           HIP_KQUANT_Q6_K_BLOCK_SIZE,
                       "Q6_K ABI block size mismatch");
HIP_KQUANT_STATIC_ASSERT(HIP_KQUANT_ALIGNOF(hip_kquant_q6_k_block_t) ==
                           HIP_KQUANT_Q6_K_BLOCK_ALIGNMENT,
                       "Q6_K ABI block alignment mismatch");
HIP_KQUANT_STATIC_ASSERT(offsetof(hip_kquant_q6_k_block_t, ql) == 0,
                       "Q6_K ABI low-quants offset mismatch");
HIP_KQUANT_STATIC_ASSERT(offsetof(hip_kquant_q6_k_block_t, qh) == 128,
                       "Q6_K ABI high-quants offset mismatch");
HIP_KQUANT_STATIC_ASSERT(offsetof(hip_kquant_q6_k_block_t, scales) == 192,
                       "Q6_K ABI scales offset mismatch");
HIP_KQUANT_STATIC_ASSERT(offsetof(hip_kquant_q6_k_block_t, d) == 208,
                       "Q6_K ABI scale offset mismatch");

HIP_KQUANT_STATIC_ASSERT(sizeof(hip_q8_1_block_t) ==
                           HIP_Q8_1_BLOCK_SIZE,
                       "Q8_1 ABI block size mismatch");
HIP_KQUANT_STATIC_ASSERT(HIP_KQUANT_ALIGNOF(hip_q8_1_block_t) ==
                           HIP_Q8_1_BLOCK_ALIGNMENT,
                       "Q8_1 ABI block alignment mismatch");
HIP_KQUANT_STATIC_ASSERT(offsetof(hip_q8_1_block_t, d) == 0,
                       "Q8_1 ABI scale offset mismatch");
HIP_KQUANT_STATIC_ASSERT(offsetof(hip_q8_1_block_t, s) == 2,
                       "Q8_1 ABI sum offset mismatch");
HIP_KQUANT_STATIC_ASSERT(offsetof(hip_q8_1_block_t, qs) == 4,
                       "Q8_1 ABI quants offset mismatch");

#undef HIP_KQUANT_ALIGNAS
#undef HIP_KQUANT_ALIGNOF
#undef HIP_KQUANT_STATIC_ASSERT

#ifdef __cplusplus
} // extern "C"
#endif

#endif // HIP_KQUANT_ABI_H
