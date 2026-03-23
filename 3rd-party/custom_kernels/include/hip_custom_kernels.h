/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_CUSTOM_KERNELS_H
#define HIP_CUSTOM_KERNELS_H

/*
 * Pure C interface for HIP custom kernels.
 *
 * This header declares host-side launcher functions that are implemented
 * in .hip files compiled by hipcc. The interface uses only standard C types
 * (no HIP-specific types) so it can be included by:
 *   - Clang during bitcode compilation (lib/Runtime/real/)
 *   - MSVC for any host-side C/C++ code
 *
 * The .hip implementations (compiled by hipcc into a static library) define
 * these functions with extern "C" linkage. At model-DLL link time, the static
 * library is linked alongside MIOpen/hipBLASLt/amdhip64 import libs.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Data Type Enum
 * =========================================================================
 *
 * Unambiguous type identifiers for kernels that need to dispatch on data type.
 * Unlike element_size_bytes, this distinguishes types of the same byte size
 * (e.g. int64 vs float64, int32 vs float32).
 *
 * Existing GQA/RoPE kernels continue using element_size_bytes since they only
 * support float32/float16 (no ambiguity). New kernels should prefer hip_dtype.
 */
typedef enum {
    HIP_DTYPE_FLOAT32  = 0,
    HIP_DTYPE_FLOAT16  = 1,
    HIP_DTYPE_INT64    = 2,
    HIP_DTYPE_INT32    = 3,
    HIP_DTYPE_FLOAT64  = 4,
    HIP_DTYPE_BFLOAT16 = 5,
} hip_dtype_t;

/* =========================================================================
 * Elementwise Subtraction
 * =========================================================================
 *
 * Computes output[i] = lhs[i] - rhs[i] for num_elements elements.
 *
 * Parameters:
 *   stream       - hipStream_t cast to void*
 *   lhs          - GPU pointer to left-hand operand
 *   rhs          - GPU pointer to right-hand operand
 *   output       - GPU pointer to output
 *   num_elements - number of elements in each tensor
 *   hip_dtype    - data type (hip_dtype_t value cast to int)
 *
 * Currently supported types: HIP_DTYPE_INT64
 * Returns: 0 on success (hipSuccess), non-zero hipError_t on failure
 */
int hip_elementwise_sub(
    void* stream,
    const void* lhs,
    const void* rhs,
    void* output,
    int64_t num_elements,
    int hip_dtype);

/* =========================================================================
 * Rotary Position Embedding (RoPE)
 * =========================================================================
 *
 * Applies rotary position embeddings to the input tensor.
 *
 * For each (batch, seq_pos, head, dim_pair d):
 *   cos_val = cos_cache[pos, d]
 *   sin_val = sin_cache[pos, d]
 *
 *   Non-interleaved (half-rotated):
 *     x0 = input[..., d],  x1 = input[..., d + rotary_dim/2]
 *     output[..., d]                 = x0 * cos_val - x1 * sin_val
 *     output[..., d + rotary_dim/2]  = x0 * sin_val + x1 * cos_val
 *
 *   Interleaved:
 *     x0 = input[..., 2*d],  x1 = input[..., 2*d+1]
 *     output[..., 2*d]   = x0 * cos_val - x1 * sin_val
 *     output[..., 2*d+1] = x0 * sin_val + x1 * cos_val
 *
 * Parameters:
 *   stream             - hipStream_t cast to void*
 *   input              - GPU pointer [batch, seq_len, num_heads * head_dim]
 *   position_ids       - GPU pointer [batch, seq_len] (int64 or int32)
 *   cos_cache          - GPU pointer [max_seq, rotary_dim/2]
 *   sin_cache          - GPU pointer [max_seq, rotary_dim/2]
 *   output             - GPU pointer (same shape as input)
 *   batch_size         - batch dimension
 *   seq_len            - sequence length
 *   num_heads          - number of attention heads
 *   head_dim           - dimension per head
 *   rotary_dim         - number of dimensions to rotate (<=head_dim)
 *   max_seq_len        - max sequence length in cos/sin cache (for bounds clamping)
 *   interleaved        - 0 = half-rotated, 1 = interleaved
 *   element_size_bytes - 2 for fp16, 4 for fp32
 *
 * Returns: 0 on success, non-zero on error
 */
int hip_rope_forward(
    void* stream,
    const void* input,
    const void* position_ids,
    const void* cos_cache,
    const void* sin_cache,
    void* output,
    int64_t batch_size,
    int64_t seq_len,
    int64_t num_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t max_seq_len,
    int64_t interleaved,
    int64_t element_size_bytes);

/* =========================================================================
 * GQA Device Kernel Launchers
 * =========================================================================
 *
 * Individual kernel launchers for the 11-step GQA pipeline.
 * All FP16 only. The orchestration (hipBLASLt GEMMs, workspace, temp
 * buffers) lives in the runtime wrapper (real/gqa.cpp).
 */

/* KV cache update: copy new K/V from BSHD [B,sq,G,d] into BNSD cache
 * [B,G,max_seq,d] at positions [past_len .. past_len+sq). */
int hip_gqa_kv_cache_update(
    void* stream, const void* src, void* cache,
    int batch_size, int sq, int G, int d, int max_seq, int past_len);

/* Internal GQA RoPE (half-rotated, FP16):
 * out[d] = in[d]*cos - in[d+half]*sin
 * out[d+half] = in[d+half]*cos + in[d]*sin */
int hip_gqa_rope(
    void* stream, const void* input, void* output,
    const void* cos_cache, const void* sin_cache,
    int batch_size, int seq_len, int num_heads,
    int head_dim, int half_rot, int past_len);

/* Transpose middle two dims of 4D tensor:
 * [B, dim1, dim2, D] -> [B, dim2, dim1, D] */
int hip_gqa_transpose_mid_dims(
    void* stream, const void* src, void* dst,
    int batch_size, int dim1, int dim2, int D);

/* KV group expansion: replicate G groups -> H heads.
 * For head h, copies from group g = h / heads_per_group. */
int hip_gqa_expand_kv(
    void* stream, const void* src, void* dst,
    int total_heads, int heads_per_group,
    int src_stride, int dst_stride, int copy_elems);

/* Causal mask (prefill only): S[k,q] = -inf where k > past_len + q */
int hip_gqa_causal_mask(
    void* stream, void* S,
    int total_heads, int skv, int sq,
    int batch_stride, int past_len);

/* Column-wise softmax in-place. One threadblock per (head, query). */
int hip_gqa_softmax_inplace(
    void* stream, void* data,
    int total_head_queries, int rows, int cols,
    int batch_stride);

/* =========================================================================
 * Cast (Element Type Conversion)
 * =========================================================================
 *
 * Converts each element from input_dtype to output_dtype.
 *   output[i] = (output_type)input[i]
 *
 * Parameters:
 *   stream       - hipStream_t cast to void*
 *   input        - GPU pointer to source data
 *   output       - GPU pointer to destination
 *   num_elements - number of elements to convert
 *   input_dtype  - source data type (hip_dtype_t value cast to int)
 *   output_dtype - destination data type (hip_dtype_t value cast to int)
 *
 * Currently supported conversions: INT64 -> INT32
 * Returns: 0 on success, non-zero on failure
 */
int hip_cast(
    void* stream,
    const void* input,
    void* output,
    int64_t num_elements,
    int input_dtype,
    int output_dtype);

/* =========================================================================
 * Gather (Index-Based Element Selection)
 * =========================================================================
 *
 * Gathers slices from data along the given axis using indices.
 * For axis=0 with scalar index:
 *   output[i] = data[index * output_num_elements + i]
 *
 * Parameters:
 *   stream              - hipStream_t cast to void*
 *   data                - GPU pointer to source tensor
 *   indices             - GPU pointer to index tensor (i64 values)
 *   output              - GPU pointer to output
 *   axis                - axis along which to gather
 *   data_num_elements   - total elements in data tensor
 *   output_num_elements - total elements in output tensor
 *   element_size_bytes  - byte size per element (used for raw copy)
 *
 * Currently supports: axis=0, scalar (single-element) index
 * Supported element sizes: 2 (f16/bf16), 4 (f32/i32), 8 (i64/f64)
 * Returns: 0 on success, non-zero on failure
 */
int hip_gather(
    void* stream,
    const void* data,
    const void* indices,
    void* output,
    int64_t axis,
    int64_t data_num_elements,
    int64_t output_num_elements,
    int element_size_bytes);

/* =========================================================================
 * ReduceSum (Parallel Sum Reduction)
 * =========================================================================
 *
 * Reduces input by summing over contiguous blocks.
 * reduce_size = num_input_elements / num_output_elements
 * For each output element j:
 *   output[j] = sum(input[j*reduce_size .. (j+1)*reduce_size - 1])
 *
 * Parameters:
 *   stream              - hipStream_t cast to void*
 *   data                - GPU pointer to input tensor
 *   output              - GPU pointer to output tensor
 *   num_input_elements  - total input elements
 *   num_output_elements - total output elements
 *   hip_dtype           - data type (hip_dtype_t value cast to int)
 *
 * Currently supported types: HIP_DTYPE_INT64
 * Returns: 0 on success, non-zero on failure
 */
int hip_reduce_sum(
    void* stream,
    const void* data,
    void* output,
    int64_t num_input_elements,
    int64_t num_output_elements,
    int hip_dtype);

#ifdef __cplusplus
}
#endif

#endif /* HIP_CUSTOM_KERNELS_H */
