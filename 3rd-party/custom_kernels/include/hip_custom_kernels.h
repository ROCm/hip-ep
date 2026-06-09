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
    HIP_DTYPE_INT16    = 6,
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
 * Elementwise Where (NumPy-style multidirectional broadcasting, arbitrary rank)
 * =========================================================================
 *
 * Computes output[i] = condition[idx_cond(i)] ? x[idx_x(i)] : y[idx_y(i)]
 * for each element of the output tensor. Each operand is described by its
 * own (shape_ptr, rank) pair; operand shapes are left-padded with 1s up to
 * the output rank to implement ONNX multidirectional broadcasting. Dims of
 * size 1 are broadcast against the corresponding larger output dim.
 *
 * No fixed layout is assumed; operands may have any rank <= HIP_WHERE_MAX_RANK.
 *
 * Parameters:
 *   stream      - hipStream_t cast to void*
 *   condition   - GPU pointer to bool tensor (1 byte per element)
 *   x           - GPU pointer to X tensor (selected when condition is true)
 *   y           - GPU pointer to Y tensor (selected when condition is false)
 *   output      - GPU pointer to output tensor (broadcast shape)
 *   cond_shape  - host pointer to condition shape array (length == cond_rank)
 *   cond_rank   - rank of condition tensor
 *   x_shape     - host pointer to X shape array (length == x_rank)
 *   x_rank      - rank of X tensor
 *   y_shape     - host pointer to Y shape array (length == y_rank)
 *   y_rank      - rank of Y tensor
 *   out_shape   - host pointer to output shape array (length == out_rank)
 *   out_rank    - rank of output tensor (max of input ranks)
 *   hip_dtype   - element type of x/y/output (hip_dtype_t value cast to int)
 *
 * Currently supported types for x/y/output: HIP_DTYPE_FLOAT32, HIP_DTYPE_FLOAT16,
 * HIP_DTYPE_BFLOAT16, HIP_DTYPE_INT32, HIP_DTYPE_INT64
 * Returns: 0 on success (hipSuccess), non-zero on failure (including rank > max)
 */
int hip_elementwise_where(
    void* stream,
    const void* condition,
    const void* x,
    const void* y,
    void* output,
    const int64_t* cond_shape, int64_t cond_rank,
    const int64_t* x_shape,    int64_t x_rank,
    const int64_t* y_shape,    int64_t y_rank,
    const int64_t* out_shape,  int64_t out_rank,
    int hip_dtype);

/* =========================================================================
 * Elementwise Unary (Neg / Sign / Cos / Sin / Not)
 * =========================================================================
 *
 * Per-op launchers for the 5 ONNX unary ops added for the Qwen3.5 vision
 * model. All five share a single .hip translation unit
 * (3rd-party/custom_kernels/hip/elementwise_unary_kernel.hip).
 *
 * Supported hip_dtype (per op, may differ):
 *   Neg/Sign  : FLOAT16, INT32, INT64 (+ FLOAT32 for free)
 *   Cos/Sin   : FLOAT16, FLOAT32
 *   Not       : bool (i.e. 1-byte; pass element_size_bytes is unused -- the
 *               kernel reads/writes 1 byte unconditionally and ignores
 *               hip_dtype)
 * Returns: 0 on success (hipSuccess), non-zero hipError_t on failure.
 */
int hip_elementwise_neg(
    void* stream,
    const void* input,
    void* output,
    int64_t num_elements,
    int hip_dtype);

int hip_elementwise_sign(
    void* stream,
    const void* input,
    void* output,
    int64_t num_elements,
    int hip_dtype);

int hip_elementwise_cos(
    void* stream,
    const void* input,
    void* output,
    int64_t num_elements,
    int hip_dtype);

int hip_elementwise_sin(
    void* stream,
    const void* input,
    void* output,
    int64_t num_elements,
    int hip_dtype);

int hip_elementwise_exp(
    void* stream,
    const void* input,
    void* output,
    int64_t num_elements,
    int hip_dtype);

int hip_elementwise_not(
    void* stream,
    const void* input,
    void* output,
    int64_t num_elements);

/* =========================================================================
 * Elementwise Binary (Div / Mod / Equal / Less)
 * =========================================================================
 *
 * Same-shape binary elementwise ops added for the Qwen3.5 vision model.
 * All four share a single .hip TU (elementwise_binary_kernel.hip).
 *
 * Important: broadcasting is NOT performed in these kernels. lhs and rhs
 * must already have identical shape (broadcasting is materialised
 * upstream via Expand).
 *
 * Output dtype for Equal/Less is bool (1 byte); their hip_dtype refers to
 * the INPUT type. For Div/Mod, output dtype matches input dtype.
 */
int hip_elementwise_div(
    void* stream,
    const void* lhs,
    const void* rhs,
    void* output,
    int64_t num_elements,
    int hip_dtype);

/* Same-shape commutative elementwise ops (Add/Mul/Min/Max). Cover integer
 * dtypes (i32/i64) that miopenOpTensor cannot describe; the runtime
 * materialises any broadcast via hip_expand before calling these. Supported
 * hip_dtype: FLOAT16, FLOAT32, INT32, INT64. */
int hip_elementwise_add(
    void* stream,
    const void* lhs,
    const void* rhs,
    void* output,
    int64_t num_elements,
    int hip_dtype);

int hip_elementwise_mul(
    void* stream,
    const void* lhs,
    const void* rhs,
    void* output,
    int64_t num_elements,
    int hip_dtype);

int hip_elementwise_min(
    void* stream,
    const void* lhs,
    const void* rhs,
    void* output,
    int64_t num_elements,
    int hip_dtype);

int hip_elementwise_max(
    void* stream,
    const void* lhs,
    const void* rhs,
    void* output,
    int64_t num_elements,
    int hip_dtype);

int hip_elementwise_mod(
    void* stream,
    const void* lhs,
    const void* rhs,
    void* output,
    int64_t num_elements,
    int hip_dtype,
    int fmod_flag);

int hip_elementwise_equal(
    void* stream,
    const void* lhs,
    const void* rhs,
    void* output,
    int64_t num_elements,
    int hip_dtype);

int hip_elementwise_less(
    void* stream,
    const void* lhs,
    const void* rhs,
    void* output,
    int64_t num_elements,
    int hip_dtype);

/* And over bool (1-byte) tensors. No hip_dtype: bool is the only supported
 * input/output type (mirrors ORT v1.22.2 SPECIALIZED_BINARY_ELEMENTWISE_IMPL(And, bool)).
 */
int hip_elementwise_and(
    void* stream,
    const void* lhs,
    const void* rhs,
    void* output,
    int64_t num_elements);

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
 * LeakyRelu Activation
 * =========================================================================
 *
 * Applies LeakyRelu element-wise: y = x >= 0 ? x : alpha * x
 *
 * Parameters:
 *   stream       - HIP stream (cast to hipStream_t internally)
 *   input        - Device pointer to input tensor
 *   output       - Device pointer to output tensor
 *   num_elements - Total number of elements
 *   hip_dtype    - Data type enum (HIP_DTYPE_FLOAT32, HIP_DTYPE_FLOAT16,
 *                  HIP_DTYPE_FLOAT64)
 *   alpha        - Slope for negative values (default 0.01 per ONNX spec)
 *
 * Returns: 0 on success (hipSuccess), non-zero hipError_t on failure
 */
int hip_leaky_relu(
    void* stream,
    const void* input,
    void* output,
    int64_t num_elements,
    int hip_dtype,
    double alpha);

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
 * When rotary_dim < head_dim, dimensions [rotary_dim, head_dim) are passed
 * through unchanged (the half-rotated kernel writes them in the d>=rotary_dim
 * branch; the interleaved path uses a separate copy kernel).
 *
 * Parameters:
 *   stream             - hipStream_t cast to void*
 *   input              - GPU pointer; layout depends on is_bnsh:
 *                          is_bnsh=0 -> BSNH [batch, seq_len, num_heads, head_dim]
 *                                       (also the 3D [B, S, num_heads*head_dim])
 *                          is_bnsh=1 -> BNSH [batch, num_heads, seq_len, head_dim]
 *   position_ids       - GPU pointer [batch, seq_len] (int64)
 *   cos_cache          - GPU pointer [max_seq, rotary_dim/2]
 *   sin_cache          - GPU pointer [max_seq, rotary_dim/2]
 *   output             - GPU pointer (same shape/layout as input)
 *   batch_size         - batch dimension
 *   seq_len            - sequence length
 *   num_heads          - number of attention heads
 *   head_dim           - dimension per head (>= rotary_dim)
 *   rotary_dim         - number of dimensions to rotate (<= head_dim)
 *   max_seq_len        - max sequence length in cos/sin cache (for bounds clamping)
 *   interleaved        - 0 = half-rotated, 1 = interleaved
 *   element_size_bytes - 2 for fp16, 4 for fp32
 *   is_bnsh            - layout flag, see input above (0 = BSNH/3D, 1 = BNSH)
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
    int64_t element_size_bytes,
    int64_t is_bnsh);

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

/* FA-2 split-K GQA decode (sq == 1, d in {64, 128}, HPG=H/G==4):
 * GQA-aware kernel that loads K/V tiles into LDS once and reuses them
 * across the 4 query heads of each KV group, then a second kernel
 * merges K_SPLITS partial (m, l, O) per query head.
 *
 * Depth-gated alternative to hip_gqa_fused_decode for skv >= ~256 where
 * Llama-3.x family shows large bandwidth headroom over the existing
 * one-block-per-head fused decode.
 *
 * Workspace: float scratch sized B*H*K_SPLITS*(d+2)*sizeof(float) bytes.
 * Caller is responsible for allocating and passing it in.
 *
 * K_SPLITS: only 8 supported in V1. Returns -1 on unsupported (HPG, d, K_SPLITS).
 *
 * seqlens_k: optional device pointer [B] int32. When non-null, total_seq
 * = seqlens_k[b]+1 is read on-device (no host sync).
 *
 * local_window_size: when > 0, restricts each query to attend only to the
 * last `local_window_size` KV positions (sliding-window attention, e.g.
 * gpt-oss-20b's 128-token sliding layers). When <= 0, full attention.
 *
 * head_sink: optional device pointer [num_heads] fp16, attention-sink
 * (smooth-softmax) per-head bias. When non-null, the final softmax
 * denominator gains an exp(s_h - global_m) term per head (no V contribution
 * for the sink). When null and use_smooth_softmax != 0, behaves as if
 * s_h = 0 for all heads. This is the gpt-oss-20b / Mistral-style attention
 * sink. The partials are unaffected; the term is folded in by the reduce
 * kernel. */
int hip_gqa_flash_decode(
    void* stream,
    const void* Q, const void* Kcache, const void* Vcache,
    void* O,
    void* partials_workspace,
    int B, int H, int G, int d, int max_seq, int K_SPLITS,
    float scale,
    const void* seqlens_k,
    int local_window_size,
    const void* head_sink,
    int use_smooth_softmax);

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
    int64_t axis_size,
    int64_t inner_size,
    int element_size_bytes,
    int indices_element_size_bytes);

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
 * Currently supported types: HIP_DTYPE_INT64, HIP_DTYPE_INT32, HIP_DTYPE_FLOAT16
 *   - INT32 accumulates in int64 internally to avoid overflow on large slices.
 *   - FLOAT16 accumulates in float internally to preserve precision; the
 *     final result is narrowed back to half.
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
 * Global pool (avg / max / lp)
 * =========================================================================
 *
 * Reduces each contiguous `reduce_size`-element slice into a single value.
 * Data is viewed as `[outer, reduce_size]` where
 *   outer       = N * C
 *   reduce_size = D_1 * D_2 * ... * D_k   (product of all spatial dims)
 *
 * `mode` selects the reduction (must match HIPDNN_EP_GLOBAL_POOL_* in
 * lib/Runtime/hipdnn_ep_runtime.h):
 *   0 (AVERAGE): Y = mean(slice)
 *   1 (MAX)    : Y = max(slice)
 *   2 (LP)     : Y = pow(sum(pow(|slice|, p)), 1/p)
 *
 * `p` is the LP-norm exponent; ignored for AVG / MAX. Caller must guarantee
 * `p >= 1` for LP (the runtime wrapper rejects values below that).
 *
 * One reduction block per output element (per (n, c) slice). Accumulation
 * happens in float (regardless of input dtype) to keep precision on long
 * spatial reductions of fp16 / bf16 inputs.
 *
 * Supported hip_dtypes: HIP_DTYPE_FLOAT32, HIP_DTYPE_FLOAT16,
 * HIP_DTYPE_BFLOAT16, HIP_DTYPE_FLOAT64.
 * Returns: 0 on success, non-zero on failure
 */
int hip_global_pool(
    void* stream,
    const void* data,
    void* output,
    int64_t outer,
    int64_t reduce_size,
    int hip_dtype,
    int mode,
    int p);

/* =========================================================================
 * Pool — MaxPool / AveragePool / LpPool (1D / 2D / 3D)
 * =========================================================================
 *
 * Generic ONNX window pooling over an `(N, C, D_1[, D_2[, D_3]])` input.
 * Lays the output `(N, C, O_1[, O_2[, O_3]])` out in row-major order matching
 * the input layout.
 *
 * `mode` selects the per-window reduction (must match HIPDNN_EP_POOL_* in
 * lib/Runtime/hipdnn_ep_runtime.h):
 *   0 (AVERAGE): Y = sum(window) / divisor
 *   1 (MAX)    : Y = max(window)
 *   2 (LP)     : Y = pow(sum(pow(|window|, p)), 1/p)
 *
 * Pad positions are never read (they fall outside the input bounds). For
 * AVERAGE, `count_include_pad` picks the divisor: 0 = number of in-bounds
 * window elements, 1 = full kernel volume (pad cells contribute 0 to the
 * sum). `p` is the LP norm exponent (>= 1); both are ignored for the modes
 * that don't use them.
 *
 * Optional `indices` (i64 buffer the same shape as the output) records the
 * row-major flat index in the *unpadded* input that each max came from —
 * MAX mode only; matches ONNX MaxPool spec for storage_order = 0. Pass NULL
 * for AVERAGE / LP.
 *
 * `spatial_rank` selects how many of the per-axis arrays are read; for
 * spatial_rank < 3 the trailing slots in `in_d`, `out_d`, `kernel`,
 * `strides`, `pads_begin`, `dilations` must be set to 1 / 0 by the caller
 * (the lowering does this).
 *
 * Supported hip_dtypes: HIP_DTYPE_FLOAT32, HIP_DTYPE_FLOAT16,
 * HIP_DTYPE_BFLOAT16, HIP_DTYPE_FLOAT64.
 * Returns: 0 on success, non-zero on failure.
 */
int hip_pool(
    void* stream,
    const void* input,
    void* output,
    void* indices,            /* int64_t* — nullable, MAX only */
    int hip_dtype,
    int mode,
    int spatial_rank,
    int64_t N, int64_t C,
    int64_t in_d0, int64_t in_d1, int64_t in_d2,
    int64_t out_d0, int64_t out_d1, int64_t out_d2,
    int64_t k0, int64_t k1, int64_t k2,
    int64_t s0, int64_t s1, int64_t s2,
    int64_t p0, int64_t p1, int64_t p2,
    int64_t dil0, int64_t dil1, int64_t dil2,
    int count_include_pad,
    int p);

/* =========================================================================
 * Resize (1D / 2D / 3D spatial)
 * =========================================================================
 *
 * Resamples the trailing spatial axes of an `(N, C, D_1, ..., D_k)` input
 * onto an `(N, C, O_1, ..., O_k)` output grid.  Per-axis scale is computed
 * inside the kernel as `scale = in_dim / out_dim`.  The (N, C) prefix is
 * pass-through.
 *
 *  mode:               0 = nearest, 1 = linear (N-linear)
 *  coord_transform:    0 = half_pixel, 1 = asymmetric, 2 = align_corners
 *  nearest_mode:       0 = round_prefer_floor (only used when mode=nearest)
 *
 * Supported hip_dtypes: HIP_DTYPE_FLOAT32, HIP_DTYPE_FLOAT16,
 * HIP_DTYPE_BFLOAT16, HIP_DTYPE_FLOAT64.
 * Returns: 0 on success, non-zero on failure.
 */
int hip_resize(
    void* stream,
    const void* input,
    void* output,
    int hip_dtype,
    int spatial_rank,
    int64_t N, int64_t C,
    int64_t in_d0, int64_t in_d1, int64_t in_d2,
    int64_t out_d0, int64_t out_d1, int64_t out_d2,
    int mode,
    int coord_transform,
    int nearest_mode);

/* =========================================================================
 * Block reductions (Max / Prod) -- same layout convention as hip_reduce_sum.
 * =========================================================================
 *
 * Both share the structure: one block reduces `reduce_size = num_input /
 * num_output` consecutive input elements into a single output. The reduce
 * axes must already be collapsed into the trailing dimension of `data`
 * (the upstream lowering arranges this for us).
 *
 * - hip_reduce_max  : Max op, init = -INF (FP) / TYPE_MIN (INT). NaN propagating
 *                     on the FP path (matches ORT _Max<float>).
 * - hip_reduce_prod : Mul op, init = 1.
 *
 * Supported hip_dtypes: HIP_DTYPE_INT32, HIP_DTYPE_INT64, HIP_DTYPE_FLOAT16
 * (FP16 accumulates in float, narrows on write).
 */
int hip_reduce_max(
    void* stream,
    const void* data,
    void* output,
    int64_t num_input_elements,
    int64_t num_output_elements,
    int hip_dtype);

int hip_reduce_prod(
    void* stream,
    const void* data,
    void* output,
    int64_t num_input_elements,
    int64_t num_output_elements,
    int hip_dtype);

/* Mean reduction (ONNX ReduceMean): sum then divide by reduce_size. */
int hip_reduce_mean(
    void* stream,
    const void* data,
    void* output,
    int64_t num_input_elements,
    int64_t num_output_elements,
    int hip_dtype);

/* =========================================================================
 * Tile / Expand (shape replication)
 * =========================================================================
 *
 * Both ops copy `input` into a larger `output`. Shapes are passed as
 * host-side int64 arrays from the lowering, so neither op needs to D2H
 * the GPU-side shape / repeats tensors.
 *
 * - Tile  : output_shape[d] = input_shape[d] * repeats[d].
 *           in_coord[d] = out_coord[d] % input_shape[d].
 * - Expand: output_shape[d] is the broadcast result; any input dim that is 1
 *           is replicated. in_coord[d] = (in_shape[d] == 1) ? 0
 *                                       : out_coord[d - rank_diff].
 *
 * Both kernels are bounded to kTileMaxRank = 8 input/output dimensions
 * (matches ORT's TArray<int64, 8> default).
 */
int hip_tile(
    void* stream,
    const void* input,
    void* output,
    const int64_t* input_shape_host,
    const int64_t* output_shape_host,
    int rank,
    int hip_dtype);

int hip_expand(
    void* stream,
    const void* input,
    void* output,
    const int64_t* input_shape_host,
    int input_rank,
    const int64_t* output_shape_host,
    int output_rank,
    int hip_dtype);

/* =========================================================================
 * GatherND
 * =========================================================================
 *
 * Pick slices of `data` along the first K = indices.shape[-1] dims (after
 * `batch_dims`), one slice per row of `indices`. INT64 indices only.
 *
 * Shapes pass as host int64 arrays (no GPU shape D2H). K and the rank
 * decomposition are computed on the host; the kernel runs one thread per
 * output element and reads K indices inline (no scratch buffer).
 */
int hip_gather_nd(
    void* stream,
    const void* input,
    const void* indices,
    void* output,
    const int64_t* data_shape_host,
    int data_rank,
    const int64_t* indices_shape_host,
    int indices_rank,
    int batch_dims,
    int hip_dtype);

/* =========================================================================
 * Slice (ONNX-13+ — non-constant indices / negative-step fallback)
 * =========================================================================
 *
 * The compile-time-constant + positive-stride case is folded to
 * `tensor.extract_slice` upstream of the runtime, so this kernel only
 * services slices whose `starts` / `ends` / `axes` / `steps` are NOT
 * graph-constant (or have negative steps).
 *
 * The host wrapper D2Hs the (typically tiny) index tensors and resolves
 * them into per-axis `(start, step)` pairs in INPUT-space, one entry per
 * data dimension. Axes not listed default to `(0, 1)`. The kernel runs
 * one thread per output element and computes:
 *
 *     in_offset = sum_d ( start[d] + out_coord[d] * step[d] ) * input_stride[d]
 *     output[out_idx] = input[in_offset]
 *
 * `step[d]` may be negative; correctness relies on the host wrapper
 * having already resolved start / end to absolute positions per ONNX's
 * negative-index and clamping rules (see lib/Runtime/real/slice.cpp).
 *
 * Bounded to rank <= 8 (matches kPadMaxRank / kGatherNDMaxRank).
 *
 * Supported dtypes: f16, f32, i32, i64.
 */
int hip_slice(
    void* stream,
    const void* input,
    void* output,
    const int64_t* input_shape_host,
    const int64_t* output_shape_host,
    const int64_t* starts_per_axis_host,  /* length = rank */
    const int64_t* steps_per_axis_host,   /* length = rank */
    int rank,
    int hip_dtype);

/* =========================================================================
 * ScatterND (ONNX-13+ with optional `reduction`)
 * =========================================================================
 *
 * Produces an output tensor with the shape of `data` whose values are
 * `data` copied, then `updates` overwritten / reduced into at positions
 * specified by `indices`.
 *
 * The host wrapper does the data->output D2D copy first (one
 * hipMemcpyAsync). The kernel then runs one thread per (updates_slice,
 * inner) pair = num_updates_slices * slice_size threads total. Each
 * thread reads K = indices.shape[-1] int64 indices inline and writes
 * one element into output.
 *
 *   num_updates_slices = product(indices.shape[:-1])
 *                      = product(updates.shape[:indices_rank-1])
 *   slice_size         = product(data.shape[K:])
 *
 * `reduction_id`:
 *   0 = none ("replace")  — last-writer-wins for duplicate indices,
 *                           matching ONNX's "undefined" guarantee.
 *   1 = add               — atomicAdd (or CAS-emulation for fp16).
 *   2 = mul               — CAS-emulated atomic multiply.
 *   3 = min               — CAS-emulated atomic min.
 *   4 = max               — CAS-emulated atomic max.
 *
 * Bounded to rank <= 8.
 *
 * Supported dtypes: f16, f32, i32, i64. INT64 indices only.
 */
int hip_scatter_nd(
    void* stream,
    const void* data,
    const void* indices,
    const void* updates,
    void* output,
    const int64_t* data_shape_host,
    int data_rank,
    const int64_t* indices_shape_host,
    int indices_rank,
    int reduction_id,
    int hip_dtype);

/* =========================================================================
 * CumSum
 * =========================================================================
 *
 * One thread per (outer, inner) slice; each thread sequentially scans
 * `axis_size` elements with stride `inner`. The host wrapper decomposes
 *   outer = product(shape[:axis]); axis_size = shape[axis];
 *   inner = product(shape[axis+1:])
 * and synchronously D2H-reads the axis scalar.
 *
 * FP16 accumulates in float to avoid precision loss for long axes.
 */
int hip_cumsum(
    void* stream,
    const void* x,
    void* y,
    int64_t outer,
    int64_t axis_size,
    int64_t inner,
    int hip_dtype,
    int exclusive,
    int reverse);

/* =========================================================================
 * Pad (constant / reflect / edge / wrap)
 * =========================================================================
 *
 * One thread per output element. For each output coord, walk the dims and
 * either copy input or fill from the pad_value depending on mode.
 *
 * `pad_mode`:    0 = Constant, 1 = Reflect, 2 = Edge, 3 = Wrap.
 * `lower_pads_host`: per-dim begin pad (length = rank), already filtered
 *                    by the `axes` attribute (defaults to 0 for unaffected
 *                    dims). Upper bound implied by output_shape.
 * `pad_value_host` : host pointer to a scalar of the data type (used only
 *                    when pad_mode == Constant). May be null -> default 0.
 */
int hip_pad(
    void* stream,
    const void* input,
    void* output,
    const int64_t* input_shape_host,
    const int64_t* output_shape_host,
    const int64_t* lower_pads_host,
    int rank,
    int hip_dtype,
    int pad_mode,
    const void* pad_value_host);

/* =========================================================================
 * LayerNormalization (ONNX-17)
 * =========================================================================
 *
 *   y = (x - mean) * rsqrt(var + epsilon) * scale + bias
 *
 * Per-row reduction with FP32 accumulators. Bias and the optional `mean` /
 * `inv_std` outputs may be null.
 *
 * `hip_dtype`   : I/O type for input/scale/bias/output -- FLOAT16 or FLOAT32.
 * `mean_dtype`  : type of mean/inv_std output buffers -- FLOAT16 or FLOAT32.
 */
int hip_layer_norm(
    void* stream,
    const void* input,
    const void* scale,
    const void* bias,         // optional
    void* output,
    void* mean_out,           // optional
    void* inv_std_out,        // optional
    int64_t outer,
    int64_t norm_size,
    float epsilon,
    int hip_dtype,
    int mean_dtype);

/* =========================================================================
 * Range (1-D sequence generation)
 * =========================================================================
 *
 * Writes output[i] = start + i * delta for i in [0, output_num_elements).
 *
 * start, limit, delta are scalar pointers in device memory (limit is accepted
 * for interface symmetry and runtime validation; the kernel only needs start
 * and delta once output_num_elements is known).
 *
 * Parameters:
 *   stream              - hipStream_t cast to void*
 *   start               - GPU pointer to scalar start
 *   limit               - GPU pointer to scalar limit
 *   delta               - GPU pointer to scalar delta
 *   output              - GPU pointer to output tensor
 *   output_num_elements - total output elements
 *   hip_dtype           - element type (hip_dtype_t value)
 *   device_error_flag   - GPU pointer to int error flag (nullable)
 *
 * Supported types: HIP_DTYPE_INT16, HIP_DTYPE_INT32, HIP_DTYPE_INT64,
 *                  HIP_DTYPE_FLOAT32, HIP_DTYPE_FLOAT64
 * Returns: 0 on success, non-zero on failure
 */
int hip_range(
    void* stream,
    const void* start,
    const void* limit,
    const void* delta,
    void* output,
    int64_t output_num_elements,
    int64_t hip_dtype,
    void* device_error_flag);

/* =========================================================================
 * Transpose (Generic N-D Permutation)
 * =========================================================================
 *
 * Permutes the dimensions of `input` according to `perm` and writes the
 * result to `output`.  Implements full ONNX Transpose semantics: any valid
 * permutation of [0, rank) is supported.  For each output linear index i:
 *   - decompose i into output coordinates using output shape derived from
 *     input_shape[perm[k]];
 *   - map to input coordinates via the supplied `perm`;
 *   - linearize using the row-major strides of input_shape and copy.
 *
 * Parameters:
 *   stream             - hipStream_t cast to void*
 *   input              - GPU pointer to source tensor (contiguous, row-major)
 *   output             - GPU pointer to destination tensor (contiguous,
 *                        row-major after permutation)
 *   rank               - number of dimensions (must be in [1, 8])
 *   input_shape        - host pointer to int64_t[rank] with the input shape
 *   perm               - host pointer to int64_t[rank] permutation; output
 *                        dim i corresponds to input dim perm[i]
 *   num_elements       - total elements in the tensor (product of input_shape)
 *   element_size_bytes - 1, 2, 4, or 8 (selects the typed memcpy kernel)
 *
 * Returns: 0 on success, non-zero hipError_t / -1 on failure.
 */
int hip_transpose(
    void* stream,
    const void* input,
    void* output,
    int64_t rank,
    const int64_t* input_shape,
    const int64_t* perm,
    int64_t num_elements,
    int element_size_bytes);

/* =========================================================================
 * MatMulNBits (Fused Dequant + MatMul)
 * =========================================================================
 *
 * Computes Y = A @ dequant(B)^T + bias, where B holds packed quantized
 * weights.  Supports bits=4 (packed nibbles) and bits=8 (1 byte per
 * weight); other widths return an error.
 *
 * Dequantization (per-block): dequant = (quant_val - zero_point) * scale
 * For 4-bit: lower nibble = first value, upper nibble = second.
 *            Default zero_point = 8 (when zero_points is NULL).
 * For 8-bit: B is unpacked uint8 of shape [N, K]; zero_points (when
 *            provided) is uint8 [N, k_blocks]; default zero_point = 128.
 *
 * Parameters:
 *   stream             - hipStream_t cast to void*
 *   A                  - GPU [batch, M, K]
 *   B                  - GPU packed weights:
 *                          bits=4: [N, k_blocks, blob_size] uint8 packed int4
 *                          bits=8: [N, K] uint8 (no packing)
 *   scales             - GPU [N, k_blocks] (same type as A)
 *   zero_points        - GPU [N, k_blocks] uint8 (nullable; default zp=8
 *                        for bits=4, zp=128 for bits=8)
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
    int64_t zp_elem_size,    // 1=uint8 packed nibbles, 2=fp16
    // Optional pre-unpacked zero_points buffers (matmul_nbits.cpp pointer-keyed
    // cache). When non-null, the kernel skips its own unpack/convert kernel
    // launches and reads from these directly. zp_u8 must be valid whenever
    // zero_points is non-null and zp_elem_size==1; zp_fp16 is only consumed
    // by the WMMA / col-major-GEMV (M>1) paths and may be null otherwise.
    const void* pre_unpacked_zp_u8,
    const void* pre_unpacked_zp_fp16);

/* Stand-alone launchers for the zero_points unpack/convert kernels, used by
 * the asym matmul_nbits cache in lib/Runtime/real/matmul_nbits.cpp.
 *
 *   zp_packed: GPU [N, ceil(K/block_size/2)] packed nibbles
 *   dst_*:     GPU output buffer, caller-allocated
 *   N:         output rows
 *   groups_k:  K / block_size (round-up)
 */
void hip_matmul_nbits_unpack_zp_u8(
    void* stream, const void* zp_packed, void* dst_u8, int N, int groups_k);
void hip_matmul_nbits_convert_zp_fp16(
    void* stream, const void* zp_packed, void* dst_fp16, int N, int groups_k);

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

/* GPU-side expert bucketing (Phase 2 foundation).
 *
 * Reorders (expert_indices, expert_weights) into per-expert contiguous slices
 * on the device, eliminating the D2H + hipStreamSynchronize that the host
 * bucket loop would otherwise need. Outputs the per-expert count and exclusive
 * prefix-sum offsets; downstream per-expert dispatch can read these directly
 * via device pointers (or as a tiny D2H of just the counts when needed).
 *
 *   expert_indices   - GPU [num_tokens * k] int32 (input from topk_routing)
 *   expert_weights   - GPU [num_tokens * k] fp16  (input from topk_routing)
 *   expert_counts    - GPU [num_experts]      int32 (output)
 *   expert_offsets   - GPU [num_experts + 1]  int32 (output, exclusive scan)
 *   sorted_token_ids - GPU [num_tokens * k]   int32 (output, grouped by eid)
 *   sorted_weights   - GPU [num_tokens * k]   fp16  (output, aligned w/ ids)
 *
 * Constraints: fp16 only; num_experts <= 1024.
 */
int hip_qmoe_bucket_tokens(
    void* stream,
    const void* expert_indices,
    const void* expert_weights,
    void* expert_counts,
    void* expert_offsets,
    void* sorted_token_ids,
    void* sorted_weights,
    int64_t num_tokens,
    int64_t num_experts,
    int64_t k,
    int64_t element_size_bytes);

/* -------------------------------------------------------------------------
 * Fully fused MoE decode (num_tokens == 1).
 *
 * Replaces the multi-pass topk -> bucket -> per-expert (gather, FC1, SwiGLU,
 * FC2, scatter_add) sequence with three back-to-back kernel launches and
 * zero hipStreamSynchronize calls per layer. Caller still issues the topk
 * (hip_qmoe_topk_routing) before invoking this; the fused launcher reads
 * expert_indices/expert_weights and dispatches all k experts inline.
 *
 * Layout (single token):
 *   input            - GPU [hidden]              fp16
 *   expert_indices   - GPU [k]                   int32 (from topk_routing)
 *   expert_weights   - GPU [k]                   fp16  (from topk_routing)
 *   fc1_weights      - GPU [E, 2*inter, K_pack]  uint8 (per-expert nibbles)
 *   fc1_scales       - GPU [E, 2*inter, n_blk]   fp16
 *   fc1_zero_points  - GPU [E, 2*inter, ceil(n_blk/2)] uint8 (packed nibbles)
 *   fc1_bias         - GPU [E, 2*inter] or null  fp16
 *   fc2_weights      - GPU [E, hidden, K_pack]   uint8
 *   fc2_scales       - GPU [E, hidden, n_blk]    fp16
 *   fc2_zero_points  - GPU [E, hidden, ceil(n_blk/2)] uint8
 *   fc2_bias         - GPU [E, hidden] or null   fp16
 *   slot_buf         - GPU [k, hidden]           fp16  (transient scratch)
 *   act_out          - GPU [k, inter]            fp16  (transient scratch)
 *   output           - GPU [hidden]              fp16  (final, weighted sum)
 *
 * Constraints: fp16 only (element_size_bytes == 2); hidden_size and
 * inter_size both multiples of 32; block_size > 0 and even.
 */
int hip_qmoe_decode_fused(
    void* stream,
    const void* input,
    const void* expert_indices,
    const void* expert_weights,
    const void* fc1_weights, const void* fc1_scales,
    const void* fc1_zero_points, const void* fc1_bias,
    const void* fc2_weights, const void* fc2_scales,
    const void* fc2_zero_points, const void* fc2_bias,
    void* slot_buf,
    void* act_out,
    void* output,
    int64_t hidden_size, int64_t inter_size,
    int64_t k, int64_t block_size,
    float swiglu_alpha, float swiglu_beta, float swiglu_limit,
    int64_t element_size_bytes);

/* =========================================================================
 * Linear Attention Decode (Single-Token Recurrence, Prefill-Friendly)
 * =========================================================================
 *
 * Performs one step of the linear attention recurrence for a single query
 * token. Updates state in-place and writes the attention output for that
 * token. The caller is expected to invoke this once per time step for
 * prefill (seq_len > 1); no batching across the time dimension is
 * performed inside the kernel.
 *
 * For each (batch, kv_head) pair, the recurrence is:
 *   linear:       S = S + k (x) v
 *   gated:        S = diag(exp(g)) * S + k (x) v
 *   delta:        S = S + beta * k (x) (v - S^T k)
 *   gated_delta:  S = diag(exp(g)) * S + beta * k (x) (v - diag(exp(g)) * S^T k)
 *   output_h = scale * S^T q_h   (for each query head h mapped to this KV head)
 *
 * Input tensors are views into the packed [B, T, H*D] layout of the full
 * sequence: query/key/value/output/decay/beta pointers must already point
 * at the start of the current time step (i.e. the caller has pre-advanced
 * the pointer by t * token_bytes). seq_len is the original T dimension
 * of the packed layout and is used by the kernel to compute the per-batch
 * stride (seq_len * H*D). Pass seq_len = 1 in the pure decode case where
 * the tensors are already shaped [B, 1, H*D].
 *
 * Head counts are three-way and subject to the following divisibility
 * constraints:
 *   - n_k_heads | kv_num_heads
 *       When n_k_heads < kv_num_heads multiple KV heads share the same key
 *       head (mapping: h_k = h_kv * n_k_heads / kv_num_heads).
 *   - Either q_num_heads % kv_num_heads == 0  (standard GQA, H_q >= H_kv)
 *       or   kv_num_heads % q_num_heads == 0  (inverse GQA, H_q < H_kv)
 *
 * Parameters:
 *   stream             - hipStream_t cast to void*
 *   query              - GPU [batch, T, q_num_heads * head_dim_k]
 *                        pointing at time step t
 *   key                - GPU [batch, T, n_k_heads * head_dim_k]
 *                        pointing at time step t
 *                        n_k_heads may differ from kv_num_heads; it must
 *                        divide kv_num_heads.
 *   value              - GPU [batch, T, kv_num_heads * head_dim_v]
 *                        pointing at time step t
 *   decay              - GPU decay tensor in log-space, or nullptr.
 *                        Layout is selected by decay_per_key_dim:
 *                          1 -> [batch, T, kv_num_heads * head_dim_k]
 *                               per-key-dimension decay (GLA / RWKV-6)
 *                          0 -> [batch, T, kv_num_heads]
 *                               per-head scalar decay (DeltaNet / RetNet),
 *                               broadcast across the head_dim_k axis.
 *                        Pointer must already be advanced to time step t.
 *                        Required for gated and gated_delta modes.
 *   beta               - GPU update-rate tensor, or nullptr.
 *                        Layout is selected by beta_per_head:
 *                          1 -> [batch, T, kv_num_heads]
 *                               per-head update rate.
 *                          0 -> [batch, T, 1]
 *                               single scalar update rate per (batch, T),
 *                               broadcast across all kv heads.
 *                        Pointer must already be advanced to time step t.
 *                        Required for delta and gated_delta modes.
 *   state              - GPU [batch, kv_num_heads, head_dim_k, head_dim_v]
 *                        Read/write. Must be pre-initialized (from past_state
 *                        or zeros) before the first time step.
 *   output             - GPU [batch, T, max(q_num_heads, kv_num_heads) *
 *                             head_dim_v], pointing at time step t.
 *                        Standard GQA: heads packed in Q-head order.
 *                        Inverse GQA: heads packed in KV-head order.
 *   B                  - batch dimension
 *   seq_len            - length of the T dimension in the packed layout;
 *                        used to compute per-batch stride. Use 1 when the
 *                        tensors are already shaped [B, 1, H*D].
 *   Hq                 - number of query heads
 *   Hkv                - number of key/value state heads
 *   Nk                 - number of key heads packed in the key tensor;
 *                        must divide Hkv
 *   dk                 - key dimension per head
 *   dv                 - value dimension per head
 *   scale              - output scaling factor (typically 1/sqrt(d_k))
 *   update_rule        - 0=linear, 1=gated, 2=delta, 3=gated_delta
 *   decay_per_key_dim  - decay layout flag (see `decay` above). Ignored when
 *                        decay == nullptr. Any non-zero value is treated as 1.
 *   beta_per_head      - beta  layout flag (see `beta`  above). Ignored when
 *                        beta  == nullptr. Any non-zero value is treated as 1.
 *   type               - element type enum: 0=float, 1=float16, 2=bfloat16
 *                        (HIPDNN_EP_DATATYPE_* in hipdnn_ep_runtime.h)
 *
 * Returns: 0 on success, non-zero on failure
 */
int hip_linear_attention_decode(
    void* stream,
    const void* query,
    const void* key,
    const void* value,
    const void* decay,
    const void* beta,
    void* state,
    void* output,
    int64_t B,
    int64_t seq_len,
    int64_t Hq,
    int64_t Hkv,
    int64_t Nk,
    int64_t dk,
    int64_t dv,
    float scale,
    int64_t update_rule,
    int64_t decay_per_key_dim,
    int64_t beta_per_head,
    int64_t type);

/* =========================================================================
 * Causal Depthwise 1D Conv -- single-step "decode" path
 * =========================================================================
 *
 * Fused fast path for the seq_len == 1 case of CausalConvWithState used by
 * Mamba / Gated DeltaNet decoders. Replaces the MIOpen virtual-buffer +
 * convolution + bias + activation chain with one compute kernel that:
 *   - reads past_state[b,c,0..k-2] (or zero if past_state==nullptr),
 *   - reads input[b,c,0],
 *   - computes the depthwise convolution dot product:
 *       output[b,c,0] = sum_{j=0..k-2} weight[c,0,j] * past_state[b,c,j]
 *                     + weight[c,0,k-1] * input[b,c,0]
 *                     + (bias ? bias[c] : 0)
 *   - applies optional SiLU (activation == 1):
 *       output[b,c,0] *= 1 / (1 + exp(-output[b,c,0]))
 *   - writes the new state by shifting forward by one step:
 *       present_state[b,c,0..k-3] = past_state[b,c,1..k-2]
 *       present_state[b,c,k-2]    = input[b,c,0]
 *
 * Bypasses hipMemcpy2DAsync entirely: at decode-shape (rows=B*C, width=k-1
 * elements) the 2D copy has thousands of pathologically thin rows and is
 * massively slower than a single launch with the same arithmetic.
 *
 * Shapes (matching wrap_causal_conv_with_state layout):
 *   input         [B, C, 1]           (past_state is [B, C, k-1])
 *   weight        [C, 1, k]           (depthwise: one k-tap filter per channel)
 *   bias          [C] or nullptr
 *   output        [B, C, 1]
 *   past_state    [B, C, k-1] or nullptr (treated as zeros)
 *   present_state [B, C, k-1]
 *
 * Constraints:
 *   - kernel_size in [1, 8]   (k-1 fits in a small register array)
 *   - element_size_bytes in {2, 4} (fp16 or fp32; matches wrapper validation)
 *   - activation in {0, 1}   (0=none, 1=SiLU)
 *
 * Returns: 0 on success, non-zero on failure.
 */
int hip_causal_conv_step_decode(
    void* stream,
    const void* input,
    const void* weight,
    const void* bias,
    const void* past_state,
    void* output,
    void* present_state,
    int64_t batch_size,
    int64_t channels,
    int64_t kernel_size,
    int64_t activation,
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
