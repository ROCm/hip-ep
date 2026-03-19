/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef hip_CUSTOM_KERNELS_H
#define hip_CUSTOM_KERNELS_H

/*
 * Pure C interface for hip custom HIP kernels.
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
    hip_DTYPE_FLOAT32  = 0,
    hip_DTYPE_FLOAT16  = 1,
    hip_DTYPE_INT64    = 2,
    hip_DTYPE_INT32    = 3,
    hip_DTYPE_FLOAT64  = 4,
    hip_DTYPE_BFLOAT16 = 5,
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
 *   hip_dtype   - data type (hip_dtype_t value cast to int)
 *
 * Currently supported types: hip_DTYPE_INT64
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
 * Group Query Attention (GQA)
 * =========================================================================
 *
 * Performs scaled dot-product attention with grouped key/value heads and
 * optional KV cache management.
 *
 * Computation:
 *   1. If past_key/past_value provided: concatenate with key/value into
 *      present_key/present_value (KV cache update)
 *   2. If do_rotary: apply RoPE to query and key
 *   3. Compute attention: softmax(scale * Q @ K^T) @ V
 *   4. Write output [batch, seq_q, num_heads * head_dim]
 *
 * Parameters:
 *   stream              - hipStream_t cast to void*
 *   query               - GPU [batch, seq_q, num_heads * head_dim]
 *   key                 - GPU [batch, seq_q, kv_num_heads * head_dim]
 *   value               - GPU [batch, seq_q, kv_num_heads * head_dim]
 *   past_key            - GPU [batch, past_seq, kv_num_heads * head_dim] or NULL
 *   past_value          - GPU [batch, past_seq, kv_num_heads * head_dim] or NULL
 *   seqlens_k           - GPU [batch] past sequence lengths per batch (int32)
 *   total_seq_len       - GPU scalar: total KV sequence length
 *   cos_cache           - GPU [max_seq, rotary_dim/2] (if do_rotary)
 *   sin_cache           - GPU [max_seq, rotary_dim/2] (if do_rotary)
 *   output              - GPU [batch, seq_q, num_heads * head_dim]
 *   present_key         - GPU [batch, total_seq, kv_num_heads * head_dim]
 *   present_value       - GPU [batch, total_seq, kv_num_heads * head_dim]
 *   batch_size          - batch dimension
 *   seq_len_q           - query sequence length
 *   seq_len_kv          - total KV sequence length (past + current)
 *   num_heads           - number of query heads
 *   kv_num_heads        - number of key/value heads (num_heads % kv_num_heads == 0)
 *   head_dim            - dimension per head
 *   scale               - attention scale (typically 1/sqrt(head_dim))
 *   softcap             - soft capping value (0 = disabled)
 *   do_rotary           - 1 to apply RoPE, 0 to skip
 *   rotary_interleaved  - 0 = half-rotated, 1 = interleaved
 *   element_size_bytes  - 2 for fp16, 4 for fp32
 *
 * Returns: 0 on success, non-zero on error
 */
int hip_gqa_forward(
    void* stream,
    const void* query,
    const void* key,
    const void* value,
    const void* past_key,
    const void* past_value,
    const void* seqlens_k,
    const void* total_seq_len,
    const void* cos_cache,
    const void* sin_cache,
    void* output,
    void* present_key,
    void* present_value,
    int64_t batch_size,
    int64_t seq_len_q,
    int64_t seq_len_kv,
    int64_t num_heads,
    int64_t kv_num_heads,
    int64_t head_dim,
    float scale,
    float softcap,
    int64_t do_rotary,
    int64_t rotary_interleaved,
    int64_t element_size_bytes);

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
 *   hip_dtype          - data type (hip_dtype_t value cast to int)
 *
 * Currently supported types: hip_DTYPE_INT64
 * Returns: 0 on success, non-zero on failure
 */
int hip_reduce_sum(
    void* stream,
    const void* data,
    void* output,
    int64_t num_input_elements,
    int64_t num_output_elements,
    int hip_dtype);

/* =========================================================================
 * MatMulNBits (Fused Dequant + MatMul)
 * =========================================================================
 *
 * Computes Y = A @ dequant(B)^T + bias, where B holds packed int4 weights.
 *
 * Dequantization (per-block): dequant = (quant_val - zero_point) * scale
 * For 4-bit: lower nibble = first value, upper nibble = second.
 * Default zero_point = 8 (when zero_points is NULL).
 *
 * Parameters:
 *   stream             - hipStream_t cast to void*
 *   A                  - GPU [batch, M, K]
 *   B                  - GPU [N, k_blocks, blob_size] uint8 packed int4
 *   scales             - GPU [N, k_blocks] (same type as A)
 *   zero_points        - GPU [N, k_blocks] uint8 (nullable, default zp=8)
 *   bias               - GPU [N] (nullable, same type as A)
 *   output             - GPU [batch, M, N]
 *   M                  - rows per batch
 *   N                  - output columns
 *   K                  - inner dimension
 *   batch_count        - number of batches
 *   bits               - quantization bit-width (must be 4)
 *   block_size         - quantization block size (e.g. 32)
 *   element_size_bytes - 2 for fp16, 4 for fp32
 *
 * Returns: 0 on success, non-zero on failure
 */
int hip_matmul_nbits(
    void* stream,
    const void* A,
    const void* B,
    const void* scales,
    const void* zero_points,
    const void* bias,
    void* output,
    int64_t M, int64_t N, int64_t K,
    int64_t batch_count,
    int64_t bits,
    int64_t block_size,
    int64_t element_size_bytes);

/* =========================================================================
 * QMoE Sub-Kernels
 * =========================================================================
 *
 * Individual kernel launchers for QMoE (Quantized Mixture-of-Experts).
 * These only launch GPU kernels — no memory allocation, no stream sync.
 * The runtime wrapper (wrap_qmoe) orchestrates the expert loop.
 *
 * All functions take element_size_bytes: 2 for fp16, 4 for fp32.
 */

/* Top-k routing: find top-k experts per token from router_probs.
 *   router_probs   - GPU [num_tokens, num_experts]
 *   expert_indices - GPU [num_tokens, k] int32 (output)
 *   expert_weights - GPU [num_tokens, k] (output, same type as probs)
 *   normalize      - 1 to normalize selected weights (sum-to-one)
 */
int hip_qmoe_topk_routing(
    void* stream,
    const void* router_probs,
    void* expert_indices,
    void* expert_weights,
    int64_t num_tokens,
    int64_t num_experts,
    int64_t k,
    int64_t normalize,
    int64_t element_size_bytes);

/* Gather rows: gathered[i,:] = input[token_ids[i],:]
 *   token_ids - GPU [count] int32
 */
int hip_qmoe_gather_tokens(
    void* stream,
    const void* input,
    void* gathered,
    const void* token_ids,
    int64_t width,
    int64_t count,
    int64_t element_size_bytes);

/* In-place bias: data[i,j] += bias[j]
 *   No-op if bias is NULL.
 */
int hip_qmoe_add_bias(
    void* stream,
    void* data,
    const void* bias,
    int64_t n,
    int64_t width,
    int64_t element_size_bytes);

/* SwiGLU activation (fused, swiglu_fusion=1):
 *   input  [n, 2*inter_size] -> output [n, inter_size]
 *   G = min(gate, limit)
 *   L = clamp(linear, -limit, limit)
 *   out = G * sigmoid(alpha*G) * (L + beta)
 */
int hip_qmoe_swiglu(
    void* stream,
    const void* input,
    void* output,
    int64_t n,
    int64_t inter_size,
    float alpha,
    float beta,
    float limit,
    int64_t element_size_bytes);

/* Weighted scatter-add: output[token_ids[i],:] += weights[i] * expert_out[i,:]
 *   token_ids - GPU [count] int32
 *   weights   - GPU [count] (same type as output)
 */
int hip_qmoe_scatter_add(
    void* stream,
    void* output,
    const void* expert_out,
    const void* token_ids,
    const void* weights,
    int64_t width,
    int64_t count,
    int64_t element_size_bytes);

#ifdef __cplusplus
}
#endif

#endif /* hip_CUSTOM_KERNELS_H */
