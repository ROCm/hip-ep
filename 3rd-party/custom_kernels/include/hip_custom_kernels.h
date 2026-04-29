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
 * Elementwise reciprocal (1 / x)
 * =========================================================================
 *
 * ONNX Reciprocal over the full IEEE domain (including negative x and x=0
 * per IEEE rules). Implemented as a plain HIP kernel. MIOpen
 * miopenActivationPOWER (used for other wrap_power cases) does not match
 * ONNX 1/x for negative inputs; see lib/Runtime/real/power.cpp.
 *
 * Supported hip_dtype: HIP_DTYPE_FLOAT32, HIP_DTYPE_FLOAT16, HIP_DTYPE_BFLOAT16
 * Returns: 0 on success (hipSuccess), non-zero hipError_t on failure
 */
int hip_elementwise_reciprocal(
    void* stream,
    const void* input,
    void* output,
    int64_t num_elements,
    int hip_dtype);

/* =========================================================================
 * Elementwise sqrt (ONNX Sqrt / IEEE)
 * =========================================================================
 *
 * Element-wise square root via HIP (sqrtf / promote half and bf16 to float).
 * Matches ONNX: negative inputs yield NaN; positive domain follows IEEE.
 * hip.sqrt lowers to wrap_power(0, 1, 0.5) which dispatches here instead of
 * MIOpen miopenActivationPOWER (see lib/Runtime/real/power.cpp).
 *
 * Supported hip_dtype: HIP_DTYPE_FLOAT32, HIP_DTYPE_FLOAT16, HIP_DTYPE_BFLOAT16
 * Returns: 0 on success (hipSuccess), non-zero hipError_t on failure
 */
int hip_elementwise_sqrt(
    void* stream,
    const void* input,
    void* output,
    int64_t num_elements,
    int hip_dtype);

/* =========================================================================
 * Elementwise GELU (Gaussian Error Linear Unit)
 * =========================================================================
 *
 * Element-wise GELU activation via HIP with support for exact and approximate modes.
 *
 * Approximate mode (approximate=1, tanh):
 *   Formula: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
 *   Standard approximation used in PyTorch, TensorFlow, and ONNX.
 *
 * Exact mode (approximate=0, erf, default):
 *   Formula: GELU(x) = x * 0.5 * (1.0 + erf(x / sqrt(2.0)))
 *   Matches ONNX Gelu operator spec exactly.
 *
 * Parameters:
 *   stream       - hipStream_t cast to void*
 *   input        - GPU pointer to input
 *   output       - GPU pointer to output
 *   num_elements - number of elements
 *   hip_dtype    - data type (HIP_DTYPE_FLOAT32, HIP_DTYPE_FLOAT16,
 *                  HIP_DTYPE_BFLOAT16, HIP_DTYPE_FLOAT64)
 *   approximate  - 0 for exact (erf), 1 for tanh approximation
 *
 * Supported data types (per ONNX spec):
 *   - HIP_DTYPE_FLOAT32 (float32)
 *   - HIP_DTYPE_FLOAT16 (float16)
 *   - HIP_DTYPE_BFLOAT16 (bfloat16)
 *   - HIP_DTYPE_FLOAT64 (double/float64)
 *
 * Returns: 0 on success (hipSuccess), non-zero hipError_t on failure
 */
int hip_elementwise_gelu(
    void* stream,
    const void* input,
    void* output,
    int64_t num_elements,
    int hip_dtype,
    int64_t approximate);

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
 * Individual kernel launchers for the 12-step GQA pipeline (Step 0 + Steps 1-11).
 * All FP16 only. The orchestration (hipBLASLt GEMMs, workspace, temp
 * buffers) lives in the runtime wrapper (real/gqa.cpp).
 */

/* KV cache append: scatter new K/V from BSHD [B,sq,G,d] into an existing
 * BNSD cache [B,G,present_seq,d] at positions [past_len .. past_len+sq).
 * present_seq is the actual sequence dimension (stride) of the present buffer,
 * which may be larger than past_len+sq if the buffer is pre-allocated.
 * Use when past and present share the same buffer (aliased / in-place).
 * seqlens_k: optional device pointer [B] int32. When non-null, past_len is
 * derived from seqlens_k[b]+1-sq (per-batch) and the host past_len is ignored.
 * Pass NULL for host-side past_len. */
int hip_gqa_kv_cache_append(
    void* stream, const void* src, void* cache,
    int batch_size, int sq, int G, int d, int present_seq, int past_len,
    const void* seqlens_k);

/* KV cache concat: concatenate past data and new tokens into a fresh present
 * buffer.  Fills present [B,G,present_seq,d] by copying past data from
 * past [B,G,past_seq,d] at positions [0,past_len) AND transposing new tokens
 * from current BSHD [B,sq,G,d] at [past_len,past_len+sq).
 * past_seq and present_seq are the actual sequence dimensions (strides) of the
 * respective buffers.  Handles the stride mismatch (past_seq != present_seq)
 * in a single kernel launch. */
int hip_gqa_kv_cache_concat(
    void* stream, const void* past, const void* current, void* present,
    int batch_size, int past_len, int sq, int G, int d,
    int past_seq, int present_seq);

/* Internal GQA RoPE (half-rotated, FP16):
 * out[d] = in[d]*cos - in[d+half]*sin
 * out[d+half] = in[d+half]*cos + in[d]*sin
 * seqlens_k: optional device pointer [B] int32. When non-null, past_len is
 * derived from seqlens_k[b]+1-seq_len and the host past_len is ignored. */
int hip_gqa_rope(
    void* stream, const void* input, void* output,
    const void* cos_cache, const void* sin_cache,
    int batch_size, int seq_len, int num_heads,
    int head_dim, int half_rot, int past_len,
    const void* seqlens_k);

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

/* Split packed QKV [B*S, (H+2*G)*d] into separate Q, K, V buffers.
 * Q: [B*S, H*d], K: [B*S, G*d], V: [B*S, G*d] */
int hip_gqa_split_qkv(
    void* stream, const void* packed, void* Q, void* K, void* V,
    int batch_size, int seq_len, int num_heads, int kv_num_heads, int head_dim);

/* Causal mask (prefill only): S[k,q] = -inf where k > past_len + q.
 * When local_window_size > 0, also masks k < past_len + q - local_window_size + 1. */
int hip_gqa_causal_mask(
    void* stream, void* S,
    int total_heads, int skv, int sq,
    int batch_stride, int past_len, int local_window_size);

/* Causal mask on fp32 Score matrix.  Same semantics as hip_gqa_causal_mask
 * but operates on float* and writes -INFINITY instead of -65504. */
int hip_gqa_causal_mask_f32(
    void* stream, void* S,
    int total_heads, int skv, int sq,
    int batch_stride, int past_len, int local_window_size);

/* Column-wise softmax in-place. One threadblock per (head, query).
 * Smooth softmax is activated when head_sink is non-null OR use_smooth_softmax
 * is set.  When head_sink is non-null, uses per-head sink factors:
 *   softmax_i = exp(x_i) / (exp(head_sink[h]) + sum_j exp(x_j))
 * When head_sink is null but use_smooth_softmax is set, uses sink = 0:
 *   softmax_i = exp(x_i) / (exp(0) + sum_j exp(x_j)) */
int hip_gqa_softmax_inplace(
    void* stream, void* data,
    int total_head_queries, int rows, int cols,
    int batch_stride, const void* head_sink, int num_heads,
    int use_smooth_softmax);

/* Column-wise softmax: fp32 input -> fp16 output.
 * Reads fp32 Score matrix (no fp16 overflow/inf), writes fp16 probabilities.
 * input_batch_stride is in float elements, output_batch_stride in half elements. */
int hip_gqa_softmax_f32_to_f16(
    void* stream, const void* input_f32, void* output_f16,
    int total_head_queries, int rows, int cols,
    int input_batch_stride, int output_batch_stride,
    const void* head_sink, int num_heads, int use_smooth_softmax);

/* Fused GQA decode (sq == 1, d in {64, 128, 256}): single-token attention
 * via cooperative dot product + online softmax in log2e space, one block
 * per (batch, head_q) with D threads (D == d).
 * Replaces steps 3, 6-11 of the decomposed pipeline.
 * Returns -1 for unsupported d values.
 * seqlens_k: optional device pointer [B] int32. When non-null, the kernel
 * reads total_seq = seqlens_k[b]+1 as the loop bound (instead of skv).
 * NOTE: assumes wave32 (RDNA); not portable to CDNA/wave64 without changes
 * to the warp shuffle reduction tree. */
int hip_gqa_fused_decode(
    void* stream, const void* Q, const void* Kcache, const void* Vcache,
    void* O, int B, int H, int G, int d, int skv, int max_seq,
    float scale, const void* seqlens_k);

/* Fused GQA prefill (sq > 1, d == 128): Flash Attention 2 with WMMA
 * tile GEMMs and online softmax. Double-buffered KV tiles, causal mask.
 * Replaces steps 3, 6-11 of the decomposed pipeline. */
int hip_gqa_fused_prefill(
    void* stream, const void* Q, const void* Kcache, const void* Vcache,
    void* O, int B, int H, int G, int sq, int skv, int max_seq, int past_len,
    float scale);

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
 *   output               - GPU pointer to output
 *   axis                 - axis along which to gather
 *   data_num_elements    - total elements in data tensor
 *   indices_num_elements - total elements in indices tensor
 *   output_num_elements  - total elements in output tensor
 *   element_size_bytes   - byte size per element (used for raw copy)
 *
 * Currently supports: axis=0
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
    int64_t indices_num_elements,
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
    int64_t element_size_bytes,
    int64_t zp_elem_size);  // 1=uint8 packed nibbles, 2=fp16

/* Release internal scratch buffers (transpose / dequant).
 * Useful for benchmarking cold-cache scenarios. */
void hip_matmul_nbits_release_buffers(void);

/* Returns a string describing the autotune config used by the last call. */
const char* hip_matmul_nbits_last_tune_info(void);

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

/* =========================================================================
 * WMMA GEMM (Small-M Matrix Multiply via Wave Matrix Multiply-Accumulate)
 * =========================================================================
 *
 * Computes C[M,N] = A[M,K] * B[K,N] using RDNA 3+ WMMA instructions.
 * FP16 inputs, FP32 accumulation, FP16 output. All matrices row-major.
 *
 * Designed for M <= 512 where hipBLASLt's register-heavy tiling (256 VGPRs,
 * 4/16 occupancy) underperforms. This kernel targets ~30 VGPRs and 16/16
 * occupancy via 16x16 WMMA tiles.
 *
 * Requires K and N to be multiples of 16.
 *
 * Parameters:
 *   stream - hipStream_t cast to void*
 *   A      - GPU pointer to activation matrix [M, K] row-major (fp16)
 *   B      - GPU pointer to weight matrix [K, N] row-major (fp16)
 *   C      - GPU pointer to output matrix [M, N] (fp16)
 *   M      - number of rows in A / output
 *   K      - inner dimension (reduction axis), must be multiple of 16
 *   N      - number of columns in B / output, must be multiple of 16
 *
 * Returns: 0 on success, non-zero on failure
 */
int hip_gemm_wmma_fp16(void* stream, const void* A, const void* B,
                       void* C, int M, int K, int N);

#ifdef __cplusplus
}
#endif

#endif /* HIP_CUSTOM_KERNELS_H */
