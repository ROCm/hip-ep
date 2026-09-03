/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Device half of the GroupQueryAttention kernels, split out so the AOT build
// and hipRTC compile the same text; if the two diverge their code objects stop
// being comparable.
//
// This header is not self-contained by design: it relies on hip_arch_compat.h
// having been included first. The AOT build gets that from gqa_kernel.hip; for
// hipRTC the build concatenates the two files, since hipRTC resolves no
// include paths.

#ifndef HIPDNN_EP_RTC_GQA_DEVICE_H
#define HIPDNN_EP_RTC_GQA_DEVICE_H

#include <cstdint>

#if defined(__HIPCC_RTC__)
  // hipRTC supplies the math functions but not the <math.h> macros, and <cmath>
  // conflicts with clang's HIP mode.
  #define INFINITY __builtin_inff()
#endif

// Wavefront size: 32 on RDNA (wave32), 64 on CDNA (wave64, e.g. MI350). Keyed
// off the device-pass arch macros via hip_arch_compat.h so the shuffle-based
// reductions and per-wave shared-memory fan-in below span the whole hardware
// wave on both families.
static constexpr int WAVE_SIZE = HIPDNN_WAVE_SIZE;
static constexpr float LOG2E = 1.4426950408889634f;

// -------------------------------------------------------------------------
// Step 1-2: RoPE -- Rotary Positional Embedding (half-rotated)
// -------------------------------------------------------------------------
// Applies to Q (Step 1) and K (Step 2) independently.
// Layout: BSHD [B, seq, num_heads, head_dim]
//
// For position s, head h, dimension d_i (d_i < half_rot):
//   pos = past_len + s
//   out[d_i]      = in[d_i]*cos[pos,d_i] - in[d_i+half]*sin[pos,d_i]
//   out[d_i+half] = in[d_i+half]*cos[pos,d_i] + in[d_i]*sin[pos,d_i]
//
// Thread: one per (b,s,h,d_i) where d_i in [0, half_rot). blockIdx.y = batch.
// Ref: onnxruntime/core/mlas/lib/rotary_embedding.cpp (MlasRotaryEmbedOneRow)
// -------------------------------------------------------------------------

// Scalar load/store helpers that promote the element type to float for math
// and narrow back on store. Specialized for __half (HW convert) and float
// (identity) so the data-movement / RoPE kernels can be templated on T.
template <typename T>
__device__ __forceinline__ float gqa_to_float(T v);
template <>
__device__ __forceinline__ float gqa_to_float<__half>(__half v) {
  return __half2float(v);
}
template <>
__device__ __forceinline__ float gqa_to_float<float>(float v) {
  return v;
}

template <typename T>
__device__ __forceinline__ T gqa_from_float(float v);
template <>
__device__ __forceinline__ __half gqa_from_float<__half>(float v) {
  return __float2half(v);
}
template <>
__device__ __forceinline__ float gqa_from_float<float>(float v) {
  return v;
}

template <typename T>
__global__ void rope_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const T* __restrict__ cos_cache,
    const T* __restrict__ sin_cache,
    int seq_len, int num_heads, int head_dim,
    int half_rot, int past_len,
    const int* __restrict__ seqlens_k) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int batch_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);
  // When seqlens_k is provided, read past_len from device memory
  // (ORT convention: seqlens_k[b] = total_tokens - 1, so past_len = total - sq).
  // All threads read the same address; hardware coalesces into a broadcast.
  int eff_past_len = seqlens_k ? (seqlens_k[batch_idx] + 1 - seq_len) : past_len;
  int elems = seq_len * num_heads * half_rot;
  if (idx >= elems) return;

  int d = idx % half_rot;
  int remaining = idx / half_rot;
  int h = remaining % num_heads;
  int s = remaining / num_heads;
  if (s >= seq_len) return;

  int pos = eff_past_len + s;
  int base = ((batch_idx * seq_len + s) * num_heads + h) * head_dim;

  float x1 = gqa_to_float<T>(input[base + d]);
  float x2 = gqa_to_float<T>(input[base + d + half_rot]);
  float c  = gqa_to_float<T>(cos_cache[pos * half_rot + d]);
  float sn = gqa_to_float<T>(sin_cache[pos * half_rot + d]);

  output[base + d]            = gqa_from_float<T>(x1 * c - x2 * sn);
  output[base + d + half_rot] = gqa_from_float<T>(x2 * c + x1 * sn);
}

// -------------------------------------------------------------------------
// Step 3 / Step 11: Middle-dimension transpose
// -------------------------------------------------------------------------
// Transposes dims 1 and 2 of a 4D tensor [B, dim1, dim2, D]:
//   Step 3  (Q):  BSHD [B,S,H,d] -> BNSD [B,H,S,d]  (dim1=S, dim2=H)
//   Step 11 (O):  BNSD [B,H,S,d] -> BSHD [B,S,H,d]  (dim1=H, dim2=S)
//
// Thread: one per element in dim1*dim2*D sub-tensor. blockIdx.y = batch.
// -------------------------------------------------------------------------

template <typename T>
__global__ void transpose_mid_dims_kernel(
    const T* __restrict__ src,
    T* __restrict__ dst,
    int dim1, int dim2, int D) {
  const int batch_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int per_batch = dim1 * dim2 * D;
  if (idx >= per_batch) return;

  int di = idx % D;
  int remaining = idx / D;
  int d2 = remaining % dim2;
  int d1 = remaining / dim2;

  int src_off = batch_idx * per_batch + idx;
  int dst_off = batch_idx * per_batch + (d2 * dim1 + d1) * D + di;
  dst[dst_off] = src[src_off];
}

// -------------------------------------------------------------------------
// Step 4-5: KV Cache Append (fused transpose + scatter)
// -------------------------------------------------------------------------
// Appends new K (Step 4) or V (Step 5) tokens into an existing BNSD cache.
//   Src: BSHD [B, sq, G, d]              (new tokens from current input)
//   Dst: BNSD [B, G, present_seq, d]     (pre-allocated KV cache)
//   Write offset: [past_len .. past_len + sq)
//
// present_seq is the sequence dimension of the present buffer (its stride),
// which may be larger than past_len + sq if the buffer is pre-allocated.
//
// Fuses the BSHD -> BNSD layout transpose with the cache scatter,
// avoiding a separate transpose pass.  Past data is assumed to already be
// at the correct positions (same-buffer / aliased KV cache).
// Thread: one per (b,s,g,d_i) element. blockIdx.y = batch.
// Ref: onnxruntime/contrib_ops/cpu/bert/gqa_attention_base.h (ConcatStateChunkGQA)
// -------------------------------------------------------------------------

// Widest copy unit for the KV data-movement kernels. Both append and concat are
// pure relocation, so the element type only matters for indexing -- the payload
// can move as opaque 16 B chunks. One fp16 per lane leaves the copy issue-rate
// limited well short of the memory roofline (measured ~99 GB/s of 256 GB/s at
// 12K context); eight per lane is what closes that gap.
struct __align__(16) KvVec16 {
  uint32_t x, y, z, w;
};

template <typename T>
__global__ void kv_cache_append_kernel(
    const T* __restrict__ src,
    T* __restrict__ cache,
    int sq, int G, int d, int present_seq, int past_len,
    const int* __restrict__ seqlens_k) {
  const int batch_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);
  int eff_past_len = seqlens_k ? (seqlens_k[batch_idx] + 1 - sq) : past_len;
  // Skip malformed batches rather than write past this slice.
  if (eff_past_len < 0 || eff_past_len + sq > present_seq) return;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_per_batch = sq * G * d;
  if (idx >= total_per_batch) return;

  int di = idx % d;
  int remaining = idx / d;
  int g = remaining % G;
  int s = remaining / G;

  int src_idx = ((batch_idx * sq + s) * G + g) * d + di;
  int dst_idx = ((batch_idx * G + g) * present_seq + (eff_past_len + s)) * d + di;
  cache[dst_idx] = src[src_idx];
}

// 16 B-per-lane form of kv_cache_append_kernel. VEC elements of T are moved as
// one KvVec16, so the thread count drops by VEC and each lane issues a single
// dwordx4. Indexing is identical, just in vector units along d.
template <typename T, int VEC>
__global__ void kv_cache_append_vec_kernel(
    const T* __restrict__ src,
    T* __restrict__ cache,
    int sq, int G, int d, int present_seq, int past_len,
    const int* __restrict__ seqlens_k) {
  const int batch_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);
  int eff_past_len = seqlens_k ? (seqlens_k[batch_idx] + 1 - sq) : past_len;
  if (eff_past_len < 0 || eff_past_len + sq > present_seq) return;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int dv = d / VEC;
  int total_per_batch = sq * G * dv;
  if (idx >= total_per_batch) return;

  int dvi = idx % dv;
  int remaining = idx / dv;
  int g = remaining % G;
  int s = remaining / G;

  int src_idx = ((batch_idx * sq + s) * G + g) * d + dvi * VEC;
  int dst_idx =
      ((batch_idx * G + g) * present_seq + (eff_past_len + s)) * d + dvi * VEC;
  *reinterpret_cast<KvVec16*>(cache + dst_idx) =
      *reinterpret_cast<const KvVec16*>(src + src_idx);
}

// -------------------------------------------------------------------------
// Step 4-5 (separate-buffer variant): KV Cache Concat
// -------------------------------------------------------------------------
// Concatenates past KV data and new tokens into a fresh present buffer.
// Used when past and present are separate allocations with different strides.
//   - Positions [0, past_len): copy from past BNSD [B, G, past_seq, d]
//     (different stride: past has stride past_seq*d, present has present_seq*d)
//   - Positions [past_len, past_len+sq): transpose new tokens from
//     BSHD [B, sq, G, d] into BNSD present buffer
//
// past_seq and present_seq are the actual sequence dimensions (strides) of
// the respective buffers, which may differ from the active token counts.
//
// Performs both the strided past copy and the BSHD->BNSD new-token transpose
// in a single kernel launch, avoiding host-side loop overhead.
// Thread: one per (b, s_present, g, d_i) element. blockIdx.y = batch.
//
// s_lo is the first present position written; [0, s_lo) is left untouched, for
// a caller that will not read it back (a sliding-window layer at decode). The
// grid is sized to (total_seq - s_lo) so the skipped positions cost no threads,
// which is the entire point -- the indices below stay absolute in s.
// -------------------------------------------------------------------------

template <typename T>
__global__ void kv_cache_concat_kernel(
    const T* __restrict__ past,
    const T* __restrict__ current,
    T* __restrict__ present,
    int past_len, int sq, int G, int d,
    int past_seq, int present_seq, int s_lo) {
  const int batch_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_seq = past_len + sq;
  int total_per_batch = (total_seq - s_lo) * G * d;
  if (idx >= total_per_batch) return;

  int di = idx % d;
  int remaining = idx / d;
  int g = remaining % G;
  int s = s_lo + remaining / G;

  int dst_idx = ((batch_idx * G + g) * present_seq + s) * d + di;

  if (s < past_len) {
    int src_idx = ((batch_idx * G + g) * past_seq + s) * d + di;
    present[dst_idx] = past[src_idx];
  } else {
    int cs = s - past_len;
    int src_idx = ((batch_idx * sq + cs) * G + g) * d + di;
    present[dst_idx] = current[src_idx];
  }
}

// 16 B-per-lane form of kv_cache_concat_kernel. A VEC-wide chunk never straddles
// the past/new boundary (the split is along s, the chunk runs along d), so the
// source branch stays uniform per lane exactly as in the scalar kernel.
template <typename T, int VEC>
__global__ void kv_cache_concat_vec_kernel(
    const T* __restrict__ past,
    const T* __restrict__ current,
    T* __restrict__ present,
    int past_len, int sq, int G, int d,
    int past_seq, int present_seq, int s_lo) {
  const int batch_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int dv = d / VEC;
  int total_seq = past_len + sq;
  int total_per_batch = (total_seq - s_lo) * G * dv;
  if (idx >= total_per_batch) return;

  int dvi = idx % dv;
  int remaining = idx / dv;
  int g = remaining % G;
  int s = s_lo + remaining / G;

  int dst_idx = ((batch_idx * G + g) * present_seq + s) * d + dvi * VEC;

  if (s < past_len) {
    int src_idx = ((batch_idx * G + g) * past_seq + s) * d + dvi * VEC;
    *reinterpret_cast<KvVec16*>(present + dst_idx) =
        *reinterpret_cast<const KvVec16*>(past + src_idx);
  } else {
    int cs = s - past_len;
    int src_idx = ((batch_idx * sq + cs) * G + g) * d + dvi * VEC;
    *reinterpret_cast<KvVec16*>(present + dst_idx) =
        *reinterpret_cast<const KvVec16*>(current + src_idx);
  }
}

// -------------------------------------------------------------------------
// INT8 KV cache: quantized append / concat + dequant (per-channel symmetric)
// -------------------------------------------------------------------------
// For the quantized-KV GroupQueryAttention variant (k/v_quant_type=PER_CHANNEL,
// kv_cache_bit_width=8): the KV cache is stored as symmetric INT8 with a static
// per-channel fp32 scale [G, d] (no zero point). These mirror the fp16
// append/concat above, but fold in QuantizeLinear on the incoming fp16 tokens
// (q = clamp(round(x / scale[g*d+di]), -128, 127)) and a matching DequantizeLinear
// used to rebuild an fp16 view for the compute-bound prefill (x = q * scale).

// Quantize-append (in-place cache): fp16 BSHD [B,sq,G,d] -> INT8 BNSD
// [B,G,present_seq,d] at offset past_len (or seqlens_k-derived).
__global__ void kv_cache_append_quant_i8_kernel(
    const _Float16* __restrict__ src,     // BSHD [B, sq, G, d]
    int8_t* __restrict__ cache,           // BNSD [B, G, present_seq, d]
    const float* __restrict__ scale,      // [G, d]
    int sq, int G, int d, int present_seq, int past_len,
    const int* __restrict__ seqlens_k) {
  const int batch_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);
  int eff_past_len = seqlens_k ? (seqlens_k[batch_idx] + 1 - sq) : past_len;
  if (eff_past_len < 0 || eff_past_len + sq > present_seq) return;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_per_batch = sq * G * d;
  if (idx >= total_per_batch) return;
  int di = idx % d;
  int remaining = idx / d;
  int g = remaining % G;
  int s = remaining / G;
  int src_idx = ((batch_idx * sq + s) * G + g) * d + di;
  int dst_idx = ((batch_idx * G + g) * present_seq + (eff_past_len + s)) * d + di;
  float inv = 1.0f / scale[g * d + di];
  float q = rintf((float)src[src_idx] * inv);
  q = fminf(fmaxf(q, -128.0f), 127.0f);
  cache[dst_idx] = (int8_t)q;
}

// Quantize-concat (separate past/present buffers): copy past INT8 [0,past_len)
// then quantize new fp16 tokens [past_len, past_len+sq) into present INT8.
__global__ void kv_cache_concat_quant_i8_kernel(
    const int8_t* __restrict__ past,      // BNSD int8 [B, G, past_seq, d]
    const _Float16* __restrict__ current, // BSHD fp16 [B, sq, G, d]
    int8_t* __restrict__ present,         // BNSD int8 [B, G, present_seq, d]
    const float* __restrict__ scale,      // [G, d]
    int past_len, int sq, int G, int d, int past_seq, int present_seq,
    int s_lo) {
  const int batch_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_seq = past_len + sq;
  if (idx >= (total_seq - s_lo) * G * d) return;
  int di = idx % d;
  int r = idx / d;
  int g = r % G;
  int s = s_lo + r / G;
  int dst_idx = ((batch_idx * G + g) * present_seq + s) * d + di;
  if (s < past_len) {
    present[dst_idx] = past[((batch_idx * G + g) * past_seq + s) * d + di];
  } else {
    int cs = s - past_len;
    float inv = 1.0f / scale[g * d + di];
    float q = rintf((float)current[((batch_idx * sq + cs) * G + g) * d + di] * inv);
    q = fminf(fmaxf(q, -128.0f), 127.0f);
    present[dst_idx] = (int8_t)q;
  }
}

// Dequant INT8 BNSD [B,G,src_seq,d] -> fp16 BNSD [B,G,dst_seq,d] over the first
// total_seq positions (rebuilds an fp16 cache view for the fp16 prefill kernel).
__global__ void dequant_kv_i8_to_fp16_kernel(
    const int8_t* __restrict__ src,   // BNSD int8 [B, G, src_seq, d]
    _Float16* __restrict__ dst,       // BNSD fp16 [B, G, dst_seq, d]
    const float* __restrict__ scale,  // [G, d]
    int total_seq, int G, int d, int src_seq, int dst_seq) {
  const int batch_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_seq * G * d) return;
  int di = idx % d;
  int r = idx / d;
  int g = r % G;
  int s = r / G;
  int src_idx = ((batch_idx * G + g) * src_seq + s) * d + di;
  int dst_idx = ((batch_idx * G + g) * dst_seq + s) * d + di;
  dst[dst_idx] = (_Float16)((float)src[src_idx] * scale[g * d + di]);
}

// -------------------------------------------------------------------------
// Step 6-7: KV Group Expansion (G groups -> H heads)
// -------------------------------------------------------------------------
// Replicates each KV group for HPG = H/G query heads.
//   Src: [B*G groups, present_seq*d each]   (KV cache)
//   Dst: [B*H heads,  skv*d each]       (expanded for batched GEMM)
//   For head h: source group g = h / HPG
//
// blockIdx.x = head (0..B*H-1)
// blockIdx.y * blockDim.x + threadIdx.x = element within the skv*d slice
// Ref: onnxruntime/contrib_ops/cpu/bert/gqa_attention_base.h line 238
// -------------------------------------------------------------------------

template <typename T>
__global__ void expand_kv_kernel(
    const T* __restrict__ src,
    T* __restrict__ dst,
    int heads_per_group,
    int src_stride,
    int dst_stride,
    int copy_elems) {
  const int h = __builtin_amdgcn_readfirstlane(blockIdx.x);
  const int g = __builtin_amdgcn_readfirstlane(h / heads_per_group);
  int idx = blockIdx.y * blockDim.x + threadIdx.x;
  if (idx < copy_elems) {
    dst[(size_t)h * dst_stride + idx] =
        src[(size_t)g * src_stride + idx];
  }
}

// -------------------------------------------------------------------------
// Step 0: Split packed QKV into separate Q, K, V buffers
// -------------------------------------------------------------------------
// When GQA receives a single packed query tensor [B*S, (H + 2*G)*d],
// this kernel splits it into:
//   Q [B*S, H*d],  K [B*S, G*d],  V [B*S, G*d]
//
// Thread: one per element in the packed tensor.
// -------------------------------------------------------------------------

template <typename T>
__global__ void split_qkv_kernel(
    const T* __restrict__ packed,
    T* __restrict__ Q,
    T* __restrict__ K,
    T* __restrict__ V,
    int BS, int D_total, int Q_dim, int K_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= BS * D_total) return;

  int row = idx / D_total;
  int col = idx % D_total;
  T val = packed[idx];

  if (col < Q_dim)
    Q[row * Q_dim + col] = val;
  else if (col < Q_dim + K_dim)
    K[row * K_dim + (col - Q_dim)] = val;
  else
    V[row * K_dim + (col - Q_dim - K_dim)] = val;
}

// -------------------------------------------------------------------------
// Step 9a: Causal Mask (prefill only, skipped when seq_q == 1)
// -------------------------------------------------------------------------
// Masks future positions so each query attends only to past + itself.
//   Score matrix: col-major [skv, sq] per head
//   S[k, q] = -inf where k > past_len + q
//
// When local_window_size > 0, also masks distant past positions:
//   S[k, q] = -inf where k < past_len + q - local_window_size + 1
//
// blockIdx.y = head (0..B*H-1)
//
// Templated on T ∈ {__half, float}. The mask value is -65504 for fp16
// (largest finite negative) and -INFINITY for fp32.
// -------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ T causal_mask_fill_value();

template <>
__device__ __forceinline__ __half causal_mask_fill_value<__half>() {
  return __float2half(-65504.0f);
}

template <>
__device__ __forceinline__ float causal_mask_fill_value<float>() {
  return -INFINITY;
}

template <typename T>
__global__ void causal_mask_kernel_impl(
    T* __restrict__ S,
    int skv, int sq, int batch_stride, int past_len,
    int local_window_size) {
  const int head = __builtin_amdgcn_readfirstlane(blockIdx.y);
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = skv * sq;
  if (idx >= total) return;

  int k = idx % skv;
  int q = idx / skv;
  int abs_q = past_len + q;
  const T fill = causal_mask_fill_value<T>();
  if (k > abs_q) {
    S[(size_t)head * batch_stride + (size_t)q * skv + k] = fill;
  } else if (local_window_size > 0 && k < abs_q - local_window_size + 1) {
    S[(size_t)head * batch_stride + (size_t)q * skv + k] = fill;
  }
}

// -------------------------------------------------------------------------
// Step 9b: Column-wise Softmax (numerically stable, in-place)
// -------------------------------------------------------------------------
// Per column q: softmax(S[:,q]) = exp(S[:,q]-max) / sum(exp(S[:,q]-max))
//
// Smooth softmax (matching ORT behaviour):
//   Activated when head_sink is non-null OR use_smooth_softmax is set.
//   softmax_i = exp(x_i - m) / (exp(s - m) + sum_j exp(x_j - m))
//   where s = head_sink[h] when provided, or 0 when head_sink is null.
//
// One threadblock per (head, query) pair.
// Uses shared memory for parallel max-reduction and sum-reduction.
// -------------------------------------------------------------------------

// Warp-level reduction helpers (wave-size portable: 32 on RDNA, 64 on CDNA)

__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
  for (int offset = WAVE_SIZE >> 1; offset > 0; offset >>= 1)
    val = fmaxf(val, __shfl_xor(val, offset));
  return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = WAVE_SIZE >> 1; offset > 0; offset >>= 1)
    val += __shfl_xor(val, offset);
  return val;
}

// Column-wise Softmax in-place
// One threadblock per (head, query) pair.
// Uses warp shuffles for intra-wave reductions and LDS only for cross-wave.
// Uses exp2f(x * LOG2E) instead of expf(x) for native HW instruction.

__global__ void softmax_inplace_kernel(
    __half* data, int rows, int cols, int batch_stride,
    const __half* head_sink, int num_heads, int use_smooth_softmax) {
  const int linear = __builtin_amdgcn_readfirstlane(blockIdx.x);
  const int head = __builtin_amdgcn_readfirstlane(linear / cols);
  const int col  = __builtin_amdgcn_readfirstlane(linear % cols);
  __half* col_ptr = data + (size_t)head * batch_stride + (size_t)col * rows;

  const int tid = threadIdx.x;
  const int wave_id = tid / WAVE_SIZE;
  const int num_waves = blockDim.x / WAVE_SIZE;

  extern __shared__ float smem[];

  // Pass 1: find max
  float local_max = -INFINITY;
  for (int i = tid; i < rows; i += blockDim.x)
    local_max = fmaxf(local_max, __half2float(col_ptr[i]));

  local_max = warp_reduce_max(local_max);
  if (tid % WAVE_SIZE == 0)
    smem[wave_id] = local_max;
  __syncthreads();

  if (tid < WAVE_SIZE) {
    float v = (tid < num_waves) ? smem[tid] : -INFINITY;
    v = warp_reduce_max(v);
    smem[0] = v;
  }
  __syncthreads();
  float max_val = smem[0];

  // Pass 2: exp2f + sum
  float local_sum = 0.0f;
  for (int i = tid; i < rows; i += blockDim.x) {
    float e = exp2f((__half2float(col_ptr[i]) - max_val) * LOG2E);
    col_ptr[i] = __float2half(e);
    local_sum += e;
  }

  local_sum = warp_reduce_sum(local_sum);
  if (tid % WAVE_SIZE == 0)
    smem[wave_id] = local_sum;
  __syncthreads();

  if (tid < WAVE_SIZE) {
    float v = (tid < num_waves) ? smem[tid] : 0.0f;
    v = warp_reduce_sum(v);
    smem[0] = v;
  }
  __syncthreads();
  float sum_val = smem[0];

  // Smooth softmax: add exp(s - max) to the denominator.
  if (head_sink != nullptr) {
    int actual_head = head % num_heads;
    float s = __half2float(head_sink[actual_head]);
    sum_val += expf(s - max_val);
  } else if (use_smooth_softmax) {
    sum_val += expf(0.0f - max_val);
  }

  float inv_sum = 1.0f / sum_val;
  for (int i = tid; i < rows; i += blockDim.x)
    col_ptr[i] = __float2half(__half2float(col_ptr[i]) * inv_sum);
}

// The score matrix and the bias tensor are indexed independently on both axes,
// because `scores` can be a sub-block of the full [sq, total_seq] logical
// matrix while the bias always describes the whole thing.
//
// Query axis: `sq` counts the query rows present in `scores`, which is a chunk
// of the full query range when the caller is tiling the prefill. `bias_sq` is
// the full query extent of the bias tensor and `bias_row_offset` locates the
// chunk inside it.
//
// Key axis: `score_cols` counts the key columns present in `scores`, which is
// just the sliding window when the caller narrowed the decode key range.
// `bias_total_seq` is the bias tensor's true last dim and `bias_col_offset`
// locates the narrowed range inside it.
//
// Neither offset can be folded into the `bias` pointer: `bias_plane` must stay
// on the bias tensor's own full extents or every batch/head plane after the
// first mis-strides, so with bias_batch > 1 or bias_heads > 1 a pointer bump
// would be wrong.
template <typename TBias>
__global__ void add_attention_bias_kernel(
    float* __restrict__ scores, const TBias* __restrict__ bias,
    int total_heads, int num_heads, int bias_batch, int bias_heads, int sq,
    int score_cols, int score_batch_stride, int bias_sq, int bias_row_offset,
    int bias_total_seq, int bias_col_offset) {
  int head = blockIdx.y;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int n = sq * score_cols;
  if (tid >= n)
    return;
  int row = tid / score_cols;
  int col = tid % score_cols;
  int b = head / num_heads;
  int h = head % num_heads;
  int bias_b = (bias_batch == 1) ? 0 : b;
  int bias_h = (bias_heads == 1) ? 0 : h;
  size_t bias_plane = static_cast<size_t>(bias_sq) * bias_total_seq;
  size_t bias_idx = static_cast<size_t>(bias_b) * bias_heads * bias_plane +
                    static_cast<size_t>(bias_h) * bias_plane +
                    static_cast<size_t>(bias_row_offset + row) * bias_total_seq +
                    (bias_col_offset + col);
  scores[static_cast<size_t>(head) * score_batch_stride + tid] +=
      static_cast<float>(bias[bias_idx]);
}

// Step 9b (fp32->fp16 variant): Softmax reading fp32 scores, writing fp16 output
// Avoids inf-inf=NaN by operating entirely in fp32 before the final fp16 store.

// Output-type-templated: reads fp32 scores, writes TOut ∈ {__half, float}.
// fp16 output (TOut=__half) feeds the fp16 Value GEMM; fp32 output (TOut=float)
// feeds the fp32 Value GEMM on the Whisper no_causal fp32 decomposed path.
template <typename TOut>
__global__ void softmax_f32_to_out_kernel(
    const float* input, TOut* output,
    int rows, int cols, int input_batch_stride, int output_batch_stride,
    const __half* head_sink, int num_heads, int use_smooth_softmax) {
  const int linear = __builtin_amdgcn_readfirstlane(blockIdx.x);
  const int head = __builtin_amdgcn_readfirstlane(linear / cols);
  const int col  = __builtin_amdgcn_readfirstlane(linear % cols);
  const float* in_col = input + (size_t)head * input_batch_stride + (size_t)col * rows;
  TOut* out_col = output + (size_t)head * output_batch_stride + (size_t)col * rows;

  const int tid = threadIdx.x;
  const int wave_id = tid / WAVE_SIZE;
  const int num_waves = blockDim.x / WAVE_SIZE;

  extern __shared__ float smem[];

  // Pass 1: find max (fp32, no overflow risk)
  float local_max = -INFINITY;
  for (int i = tid; i < rows; i += blockDim.x)
    local_max = fmaxf(local_max, in_col[i]);

  local_max = warp_reduce_max(local_max);
  if (tid % WAVE_SIZE == 0)
    smem[wave_id] = local_max;
  __syncthreads();

  if (tid < WAVE_SIZE) {
    float v = (tid < num_waves) ? smem[tid] : -INFINITY;
    v = warp_reduce_max(v);
    smem[0] = v;
  }
  __syncthreads();
  float max_val = smem[0];

  // Pass 2: exp2f + sum (fp32 throughout, values recomputed in Pass 3)
  float local_sum = 0.0f;
  for (int i = tid; i < rows; i += blockDim.x)
    local_sum += exp2f((in_col[i] - max_val) * LOG2E);

  local_sum = warp_reduce_sum(local_sum);
  if (tid % WAVE_SIZE == 0)
    smem[wave_id] = local_sum;
  __syncthreads();

  if (tid < WAVE_SIZE) {
    float v = (tid < num_waves) ? smem[tid] : 0.0f;
    v = warp_reduce_sum(v);
    smem[0] = v;
  }
  __syncthreads();
  float sum_val = smem[0];

  if (head_sink != nullptr) {
    int actual_head = head % num_heads;
    float s = __half2float(head_sink[actual_head]);
    sum_val += expf(s - max_val);
  } else if (use_smooth_softmax) {
    sum_val += expf(0.0f - max_val);
  }

  // Pass 3: normalize and write output (fp16 or fp32 per TOut).
  float inv_sum = 1.0f / sum_val;
  for (int i = tid; i < rows; i += blockDim.x) {
    float e = exp2f((in_col[i] - max_val) * LOG2E);
    out_col[i] = gqa_from_float<TOut>(e * inv_sum);
  }
}

//===----------------------------------------------------------------------===//
// WMMA types for fused kernels (RDNA 3 / 3.5 / 4, wave32)
//===----------------------------------------------------------------------===//
// Two WMMA f16 16x16x16 encodings exist across these architectures (see
// hip_arch_compat.h for how each was empirically verified): the original
// gfx11 encoding (half16 fragments, K replicated across the wave's two
// lane-halves) and the newer gfx12-style encoding gfx1170/gfx12xx use (half8
// fragments, K split by pair = lane/16, no replication). HIPDNN_WMMA_FRAG_ELEMS
// / HIPDNN_WMMA_F32_16X16X16_F16 / hipdnn_wmma_k_off / hipdnn_wmma_acc_row
// abstract the three points of difference so the kernels below are written
// once and compile correctly for either encoding.

static constexpr int kWmmaTile = 16;  // WMMA tile dimension (16x16x16)

// half16 holds one WMMA A/B fragment; its vector width matches
// HIPDNN_WMMA_FRAG_ELEMS (16 on gfx11, 8 on the gfx12-style encoding). The
// name is kept for continuity with the existing RDNA3-era call sites even
// though the width is arch-dependent.
typedef _Float16 half16 __attribute__((ext_vector_type(HIPDNN_WMMA_FRAG_ELEMS)));
typedef float    float8 __attribute__((ext_vector_type(8)));
// Loads one A/B fragment of HIPDNN_WMMA_FRAG_ELEMS contiguous K-major
// elements starting at `pointer`, offset by this lane's pair-half
// (hipdnn_wmma_k_off(pair): 0 on gfx11, pair*8 on the gfx12-style encoding).
// Only valid when the source is contiguous along K, as it is at every call
// site below (Q rows, K/P cache rows are all D- or BKV-contiguous).
#define HALF16(pointer, pair)                                                  \
  (reinterpret_cast<half16*>(                                                  \
       (void *)(&(pointer) + hipdnn_wmma_k_off(pair)))[0])

// 16-byte opaque vector, used only to widen KV cache global loads to a single
// dwordx4 per lane. The element type is irrelevant; the bytes are memcpy'd
// into the typed prefetch registers.
typedef unsigned int bytes16 __attribute__((ext_vector_type(4)));

// Where neither WMMA builtin is available (non-RDNA3/3.5/4 arch), trap
// instead so this file still compiles and a WMMA launch on an unsupported
// arch aborts loudly rather than returning garbage.
#if !HIPDNN_HAS_WMMA
static __device__ __forceinline__ float8
hipdnn_wmma_f32_16x16x16_f16_unavailable(half16, half16, float8 c) {
  __builtin_trap();
  return c;
}
#undef HIPDNN_WMMA_F32_16X16X16_F16
#define HIPDNN_WMMA_F32_16X16X16_F16(a, b, c)                                  \
  hipdnn_wmma_f32_16x16x16_f16_unavailable((a), (b), (c))
#endif

//===----------------------------------------------------------------------===//
// FA-2 Split-K GQA Decode Kernel (Phase 1)
//===----------------------------------------------------------------------===//
// Goal: 4-15x speedup over hip_gqa_fused_decode at high KV depth (skv >= 256)
// by exploiting:
//   1. GQA-aware KV reuse: one block per (batch, head_kv) instead of per
//      (batch, head_q). HPG=4 query heads share K/V loaded once into LDS,
//      reducing global KV bandwidth by 4x.
//   2. Split-K parallelism: K_SPLITS blocks per (batch, head_kv) cover
//      different KV ranges, raising occupancy for long sequences.
//   3. LDS-staged K/V tiles (BKV=64): K[t] is loaded once and reused HPG
//      times for dot products across 4 query heads.
//
// Algorithm:
//   Each block computes partial (m, l, O) for its slice [kv_start, kv_end)
//   using online softmax in log2e space. Partials written to workspace
//   [B, H, K_SPLITS, D+2] floats. A second kernel (gqa_flash_decode_reduce)
//   merges the K_SPLITS partials per (batch, head_q) into the final output.
//
// Block geometry:
//   Block = HPG waves of 32 = HPG * 32 threads. Each wave handles ONE query
//   head out of HPG. Thread `lane` in wave handles d-elements
//   [lane * EPT, (lane+1) * EPT) where EPT = D / WAVE_SIZE.
//
// Grid:
//   gridX = B * G        - one block per (batch, head_kv)
//   gridY = K_SPLITS     - split-K dimension
//
// Supported HPG specializations (template param):
//   - HPG=4  (Llama-3.x family at d=64 and d=128)
//   - HPG=8  (gpt-oss-20b at d=64; 64 q heads / 8 kv heads)
//
// Constraints (asserted at launcher):
//   - D must be 64, 128 or 256 (EPT = D/32 in 1..8; D=256 -> BKV=16 to keep
//     the LDS tile ~16KB). The scalar kernel is otherwise D-generic.
//   - RDNA wave32 (uses __shfl_xor across 32 lanes).
//===----------------------------------------------------------------------===//

static constexpr int FLASH_DECODE_BKV     = 64;

// Partial layout per (b, h_q, k_split): {m: float, l: float, O[D]: float}
__device__ __forceinline__ size_t flash_decode_partial_offset(
    int b, int h_q, int k_split, int H, int K_SPLITS, int D) {
  return ((size_t)((b * H + h_q) * K_SPLITS + k_split)) * (size_t)(D + 2);
}

// KV-cache element format for the unified decode kernels. Extensible: to add a
// new quantized cache (INT4, FP8, per-tensor int8, ...) add an enum value here
// plus the matching `if constexpr` branch inside the kernels -- no new template
// parameter is needed (a bool per format would not scale).
enum class KvDtype { kF16, kI8 };

// std::conditional stand-in: hipRTC resolves no include paths, so
// <type_traits> is unavailable on that path.
template <bool B, class T, class F>
struct GqaCond {
  using type = T;
};
template <class T, class F>
struct GqaCond<false, T, F> {
  using type = F;
};
template <bool B, class T, class F>
using GqaCondT = typename GqaCond<B, T, F>::type;

// Element type of the KV cache for a given format. Lets the launchers below be
// written once against the format rather than once per element type.
template <KvDtype FMT>
using KvElem = GqaCondT<FMT == KvDtype::kI8, int8_t, __half>;

// FMT==kF16: fp16 K/V cache (k_scale/v_scale ignored, pass nullptr).
// FMT==kI8 : symmetric per-channel INT8 K/V cache with fp32 scales [G, D]:
//   k_fp16 = k_i8 * k_scale[head_kv*D + channel]  (folded into Q here)
//   v_fp16 = v_i8 * v_scale[head_kv*D + channel]  (deferred to the epilogue)
// The int8 read stays 1 byte/elem, halving the fp16 path's DRAM traffic on the
// bandwidth-bound decode. All divergent code is a compile-time `if constexpr`
// branch, so FMT==kF16 lowers to byte-identical SASS as the original fp16
// kernel. Both paths emit the SAME [B, H, K_SPLITS, D+2] partials.
template <int D, int HPG, KvDtype FMT = KvDtype::kF16>
__global__ void __launch_bounds__(HPG * WAVE_SIZE)
gqa_flash_decode_kernel(
    const __half* __restrict__ Q_roped,    // [B, 1, H, D] (BSHD with sq=1)
    const GqaCondT<FMT == KvDtype::kI8, int8_t, __half>* __restrict__ Kcache,  // [B, G, max_seq, D]
    const GqaCondT<FMT == KvDtype::kI8, int8_t, __half>* __restrict__ Vcache,  // [B, G, max_seq, D]
    const float* __restrict__ k_scale,     // [G, D] per-channel; FMT==kI8 only
    const float* __restrict__ v_scale,     // [G, D] per-channel; FMT==kI8 only
    float* __restrict__ partials,          // [B, H, K_SPLITS, D+2]
    int H, int G, int max_seq,
    float scale,
    const int* __restrict__ seqlens_k,
    int local_window_size,
    int K_SPLITS) {  // runtime split-K count (was a template param)
  constexpr bool KV_I8 = (FMT == KvDtype::kI8);
  using KVt = GqaCondT<KV_I8, int8_t, __half>;
  // Size the K+V LDS tile for block residency, since this kernel is bandwidth
  // bound and the split-K grid always supplies more blocks than fit. 2*BKV*D*2
  // bytes: D=128 -> BKV=32 (16KB), D=64 -> BKV=64 (16KB). At BKV=64,D=128 the
  // tile would be 32KB -> only 2 resident blocks -> latency-bound (decode
  // stalls ~1.6x).
  //
  // D>=256 uses 8KB (BKV=8) rather than the 16KB the other widths get, which
  // doubles resident blocks per WGP. Measured on gfx1151 with
  // test_gqa_decode --iters 500, vs BKV=16, taking the min of 5-7 runs
  // (the absolute numbers drift a few % with clocks, so only compare within a
  // run):
  //   Qwen3.5-9B   H16 G4 d256 skv3860   0.0440 -> 0.0409 ms  (7-9% faster)
  //   Qwen3.6-35B  H16 G2 d256 skv18646  0.1772 -> 0.1745 ms  (neutral..+3%)
  // Going further the wrong way in either direction is worse: BKV=32 costs 37%
  // on the 9B (halved residency), and BKV=4 costs 6% on the 35B (the tile stops
  // amortizing the per-tile __syncthreads and staging setup).
  constexpr int BKV = (D >= 256) ? 8 : (D == 128) ? 32 : FLASH_DECODE_BKV;
  constexpr int THREADS = HPG * WAVE_SIZE;  // total threads in block
  constexpr int EPT = D / WAVE_SIZE;  // elements-per-thread along d

  static_assert(D % WAVE_SIZE == 0, "D must be a multiple of WAVE_SIZE");
  static_assert(EPT >= 1 && EPT <= 8, "EPT must be 1..8");

  const int batch     = __builtin_amdgcn_readfirstlane(blockIdx.x / G);
  const int head_kv   = __builtin_amdgcn_readfirstlane(blockIdx.x % G);
  const int split_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);
  const int tid       = threadIdx.x;
  const int wave_id   = tid / WAVE_SIZE;       // [0, HPG)
  const int lane_id   = tid % WAVE_SIZE;
  const int head_q    = head_kv * HPG + wave_id;

  // total_seq = seqlens_k[batch] + 1, clamped to [0, max_seq].
  const int raw_skv = seqlens_k
      ? __builtin_amdgcn_readfirstlane(seqlens_k[batch] + 1)
      : max_seq;
  const int eff_skv = (raw_skv > 0)
      ? (raw_skv < max_seq ? raw_skv : max_seq) : 0;

  // Sliding-window: at decode (sq==1) the query position is total_seq-1, so
  // it attends to KV positions [max(0, total_seq - window), total_seq). When
  // window <= 0 the attention is full (kv_lo = 0). gpt-oss-20b alternates
  // sliding (local_window_size=128) and full layers; clamping kv_lo here lets
  // sliding layers skip the full KV cache scan and process only ~window slots.
  int kv_lo = 0;
  if (local_window_size > 0 && eff_skv > local_window_size) {
    kv_lo = eff_skv - local_window_size;
  }
  const int kv_range = eff_skv - kv_lo;

  // Partition KV range into K_SPLITS roughly-equal chunks.
  const int chunk    = (kv_range + K_SPLITS - 1) / K_SPLITS;
  const int kv_start = kv_lo + chunk * split_idx;
  int kv_end_tmp     = kv_start + chunk;
  if (kv_end_tmp > eff_skv) kv_end_tmp = eff_skv;
  const int kv_end   = kv_end_tmp;

  // Output partials destination for THIS wave's query head.
  float* partial_dst = partials +
      flash_decode_partial_offset(batch, head_q, split_idx, H, K_SPLITS, D);

  // Empty slice: write neutral identity so the merge ignores this split.
  // exp2(-INFINITY - global_m) == 0, so l/O contributions vanish.
  if (kv_start >= kv_end) {
    if (lane_id == 0) {
      partial_dst[0] = -INFINITY;  // m
      partial_dst[1] = 0.0f;       // l
    }
    #pragma unroll
    for (int e = 0; e < EPT; ++e)
      partial_dst[2 + lane_id * EPT + e] = 0.0f;
    return;
  }

  // LDS tiles for K/V (shared across HPG=4 query heads for dedup). 16B-aligned
  // so the cooperative load can move 8 fp16 per float4 (see staging below).
  // The __half2-vectorized staging below assumes an even EPT. Flash-decode is a
  // wave32 (RDNA) path only -- wrap_group_query_attention clears fused_supported
  // on wave64 (hipdnn_device_is_wave64()), so this invariant only needs to hold
  // there. On wave64 (CDNA/MI350) D=64 yields EPT=1; the kernel is then dead
  // code that must still compile, so exempt the wave64 device pass.
  static_assert(EPT % 2 == 0 || WAVE_SIZE != 32,
                "EPT must be even for __half2 vectorized loads (wave32)");
  // HpG==1 (MHA) runs a single wave per block, so the K/V tile in LDS is never
  // reused across waves -- staging is pure overhead (32KB LDS + a __syncthreads
  // every tile). Stream straight from global instead (kStage=false): the LDS
  // arrays shrink to 1 elem and the inner loop reads the global tile directly.
  // GQA (HpG>1) keeps the shared tile so HpG query heads load K/V once. The
  // per-lane inner math is byte-identical either way (OPTIMIZATION.md §3.1).
  constexpr bool kStage = (HPG > 1);
  constexpr int kShElems = kStage ? (BKV * D) : 1;
  __shared__ __align__(16) KVt K_lds[kShElems];
  __shared__ __align__(16) KVt V_lds[kShElems];

  // Load this wave's Q into registers, pre-converted to fp32 (the per-key dot
  // is fp32; converting once here avoids EPT half->float casts per key). For
  // INT8 the per-channel K scale is folded into Q here (dot = sum_e
  // (Q[e]*Ksc[e]) * K_i8[e]) so the inner loop is a plain int8*fp32 MAC and the
  // K-scale registers are freed; the per-channel V scale is deferred to the
  // epilogue (loaded once there instead of held live across the KV loop).
  float Q_f[EPT];
  {
    const __half* Q_ptr = Q_roped + (size_t)(batch * H + head_q) * D;
    if constexpr (KV_I8) {
      const float* ks = k_scale + (size_t)head_kv * D;
      #pragma unroll
      for (int e = 0; e < EPT; ++e)
        Q_f[e] = __half2float(Q_ptr[lane_id * EPT + e]) * ks[lane_id * EPT + e];
    } else {
      #pragma unroll
      for (int e = 0; e < EPT; ++e)
        Q_f[e] = __half2float(Q_ptr[lane_id * EPT + e]);
    }
  }

  // KV cache base offset for this (batch, head_kv).
  const size_t kv_base = (size_t)__builtin_amdgcn_readfirstlane(
      (unsigned)((batch * G + head_kv) * max_seq)) * (size_t)D;

  const float scale_log2e = scale * LOG2E;

  // Online softmax accumulators (per query head, in registers).
  float m_acc = -INFINITY;
  float l_acc = 0.0f;
  float O_acc[EPT];
  #pragma unroll
  for (int e = 0; e < EPT; ++e) O_acc[e] = 0.0f;

  // Iterate KV tiles within this block's split.
  for (int tile_start = kv_start; tile_start < kv_end; tile_start += BKV) {
    const int tile_n = (tile_start + BKV <= kv_end)
        ? BKV : (kv_end - tile_start);

    // Stage the K/V tile into LDS (GQA: reused across the HpG waves) or point
    // straight at the global tile (MHA HpG==1: single wave, no reuse).
    const KVt* Kt;
    const KVt* Vt;
    if constexpr (kStage) {
      if constexpr (KV_I8) {
        // INT8 tile: tile_n*D is a multiple of 16 bytes (D % 16 == 0), so a
        // single int4 (16 int8) vectorized copy tiles it exactly -- no tail.
        const int total_vec = (tile_n * D) / 16;
        const int4* Kv = reinterpret_cast<const int4*>(
            Kcache + kv_base + (size_t)tile_start * D);
        const int4* Vv = reinterpret_cast<const int4*>(
            Vcache + kv_base + (size_t)tile_start * D);
        int4* Ksv = reinterpret_cast<int4*>(K_lds);
        int4* Vsv = reinterpret_cast<int4*>(V_lds);
        for (int i = tid; i < total_vec; i += THREADS) {
          Ksv[i] = Kv[i];
          Vsv[i] = Vv[i];
        }
      } else {
        // Cooperative vectorized LDS load: one float4 moves 8 fp16 (D % 8 == 0,
        // tile base kv_base+tile_start*D is 16B-aligned since tile_start*D*2 is a
        // multiple of D*2 >= 128B), cutting load instructions 8x vs per-element.
        const int total_vec = (tile_n * D) / 8;
        const float4* Kv = reinterpret_cast<const float4*>(
            Kcache + kv_base + (size_t)tile_start * D);
        const float4* Vv = reinterpret_cast<const float4*>(
            Vcache + kv_base + (size_t)tile_start * D);
        float4* Ksv = reinterpret_cast<float4*>(K_lds);
        float4* Vsv = reinterpret_cast<float4*>(V_lds);
        for (int i = tid; i < total_vec; i += THREADS) {
          Ksv[i] = Kv[i];
          Vsv[i] = Vv[i];
        }
      }
      __syncthreads();
      Kt = K_lds;
      Vt = V_lds;
    } else {
      // &Kt[t*D + ...] == Kcache[kv_base + (tile_start + t)*D + ...].
      Kt = Kcache + kv_base + (size_t)tile_start * D;
      Vt = Vcache + kv_base + (size_t)tile_start * D;
    }

    // Each wave processes the tile against its own query head.
    for (int t = 0; t < tile_n; ++t) {
      // Per-lane partial dot. fp16: __half2 reads (2 fp16/load + half22float2).
      // int8: K scale is pre-folded into Q_f, so this is a plain int8*fp32 MAC;
      // when EPT % 4 == 0 a single char4 (32-bit) LDS load beats two 16-bit reads.
      float dot_partial = 0.0f;
      if constexpr (KV_I8) {
        if constexpr (EPT % 4 == 0) {
          #pragma unroll
          for (int e = 0; e < EPT; e += 4) {
            const char4 kk =
                *reinterpret_cast<const char4*>(&Kt[t * D + lane_id * EPT + e]);
            dot_partial += Q_f[e]     * (float)kk.x + Q_f[e + 1] * (float)kk.y
                         + Q_f[e + 2] * (float)kk.z + Q_f[e + 3] * (float)kk.w;
          }
        } else {
          #pragma unroll
          for (int e = 0; e < EPT; e += 2) {
            const char2 kk =
                *reinterpret_cast<const char2*>(&Kt[t * D + lane_id * EPT + e]);
            dot_partial += Q_f[e] * (float)kk.x + Q_f[e + 1] * (float)kk.y;
          }
        }
      } else {
        #pragma unroll
        for (int e = 0; e < EPT; e += 2) {
          const __half2 kk =
              *reinterpret_cast<const __half2*>(&Kt[t * D + lane_id * EPT + e]);
          const float2 kf = __half22float2(kk);
          dot_partial += Q_f[e] * kf.x + Q_f[e + 1] * kf.y;
        }
      }
      // Warp reduce across the wave -> all lanes hold the full dot.
#pragma unroll
      for (int offset = WAVE_SIZE >> 1; offset > 0; offset >>= 1)
        dot_partial += __shfl_xor(dot_partial, offset);

      const float score = dot_partial * scale_log2e;
      const float m_new = fmaxf(m_acc, score);
      const float correction = exp2f(m_acc - m_new);
      const float p = exp2f(score - m_new);
      l_acc = correction * l_acc + p;

      // fp16: dequant-on-read via __half2. int8: accumulate raw p*V_i8; the
      // per-channel V scale is applied once in the epilogue.
      if constexpr (KV_I8) {
        if constexpr (EPT % 4 == 0) {
          #pragma unroll
          for (int e = 0; e < EPT; e += 4) {
            const char4 vv =
                *reinterpret_cast<const char4*>(&Vt[t * D + lane_id * EPT + e]);
            O_acc[e]     = correction * O_acc[e]     + p * (float)vv.x;
            O_acc[e + 1] = correction * O_acc[e + 1] + p * (float)vv.y;
            O_acc[e + 2] = correction * O_acc[e + 2] + p * (float)vv.z;
            O_acc[e + 3] = correction * O_acc[e + 3] + p * (float)vv.w;
          }
        } else {
          #pragma unroll
          for (int e = 0; e < EPT; e += 2) {
            const char2 vv =
                *reinterpret_cast<const char2*>(&Vt[t * D + lane_id * EPT + e]);
            O_acc[e]     = correction * O_acc[e]     + p * (float)vv.x;
            O_acc[e + 1] = correction * O_acc[e + 1] + p * (float)vv.y;
          }
        }
      } else {
        #pragma unroll
        for (int e = 0; e < EPT; e += 2) {
          const __half2 vv =
              *reinterpret_cast<const __half2*>(&Vt[t * D + lane_id * EPT + e]);
          const float2 vf = __half22float2(vv);
          O_acc[e]     = correction * O_acc[e]     + p * vf.x;
          O_acc[e + 1] = correction * O_acc[e + 1] + p * vf.y;
        }
      }
      m_acc = m_new;
    }
    if constexpr (kStage) __syncthreads();
  }

  // Write partial (m, l, O) for this wave's query head.
  if (lane_id == 0) {
    partial_dst[0] = m_acc;
    partial_dst[1] = l_acc;
  }
  if constexpr (KV_I8) {
    // Apply the deferred per-channel V scale on the way out (loaded here, not
    // held in registers across the KV loop).
    const float* vs = v_scale + (size_t)head_kv * D;
    #pragma unroll
    for (int e = 0; e < EPT; ++e)
      partial_dst[2 + lane_id * EPT + e] = O_acc[e] * vs[lane_id * EPT + e];
  } else {
    #pragma unroll
    for (int e = 0; e < EPT; ++e)
      partial_dst[2 + lane_id * EPT + e] = O_acc[e];
  }
}

//===----------------------------------------------------------------------===//
// WMMA Split-K Decode Kernel (sq == 1) -- optimized TPS path
//===----------------------------------------------------------------------===//
// Drop-in producer of the SAME [B, H, K_SPLITS, D+2] partials as the scalar
// gqa_flash_decode_kernel above, so it shares the identical FA-2 reduce (and
// therefore seqlens_k / sliding-window / head-sink / smooth-softmax handling).
//
// Why WMMA: the scalar kernel's per-key Q.K dot ends in a 5-step __shfl_xor
// cross-lane reduction that is a serialized VALU latency chain (~41% of the
// decode runtime for high-GQA D=64, see OPTIMIZATION.md ch.8). This kernel
// instead packs the HPG query heads of one KV group into the M=16 rows of a
// WMMA 16x16x16 tile and runs the QK / PV GEMMs on the matrix unit, removing
// the reduction entirely. A software pipeline keeps the NEXT KV tile's global
// loads in flight (in registers) across the current tile's GEMMs to hide DRAM
// latency on the single resident wave.
//
// Grid: (G, K_SPLITS, B). Block: 32 (one wave). M=16 rows == HPG query heads
// (rows >= HPG are inert / never written). Split math mirrors the scalar
// kernel exactly (eff_skv, kv_lo, chunk).
//===----------------------------------------------------------------------===//
// FMT==kF16: fp16 K/V cache (k_scale/v_scale ignored, pass nullptr).
// FMT==kI8 : symmetric per-channel INT8 K/V cache -- the int8 tile is loaded
// at 1 byte/elem (HALF the fp16 DRAM traffic) and dequantized to fp16 only on
// the LDS write, so the 16x16x16 WMMA score/value GEMMs stay byte-identical to
// the fp16 path. All divergent code is a compile-time `if constexpr` branch, so
// FMT==kF16 lowers to the same SASS as the original fp16 kernel. Both paths
// emit the SAME [B, H, K_SPLITS, D+2] partials.
template <int D, int HPG, int BKV, KvDtype FMT = KvDtype::kF16>
__global__ void __launch_bounds__(32)
gqa_flash_decode_wmma_kernel(
    const __half* __restrict__ Q_in,       // [B, 1, H, D] (sq=1) == [B, H, D]
    const GqaCondT<FMT == KvDtype::kI8, int8_t, __half>* __restrict__ K_in,  // [B, G, max_seq, D]
    const GqaCondT<FMT == KvDtype::kI8, int8_t, __half>* __restrict__ V_in,  // [B, G, max_seq, D]
    const float* __restrict__ k_scale,     // [G, D] per-channel; FMT==kI8 only
    const float* __restrict__ v_scale,     // [G, D] per-channel; FMT==kI8 only
    float* __restrict__ partials,          // [B, H, K_SPLITS, D+2]
    int H, int G, int max_seq, float scale,
    const int* __restrict__ seqlens_k,
    int local_window_size,
    int K_SPLITS) {  // runtime split-K count (was a template param)
  constexpr bool KV_I8 = (FMT == KvDtype::kI8);
  using KVt = GqaCondT<KV_I8, int8_t, __half>;
  constexpr int DSTR    = D + 2;
  constexpr int S_STR   = BKV + 1;
  constexpr int P_STR   = BKV + 2;
  constexpr int Q_TILES = D / kWmmaTile;
  constexpr int O_TILES = D / kWmmaTile;
  constexpr int N_TILES = BKV / kWmmaTile;

  const _Float16* Q = reinterpret_cast<const _Float16*>(Q_in);
  // fp16: KVt==_Float16 (reinterpret of __half*); int8: KVt==int8_t (identity).
  const KVt* Ksrc = reinterpret_cast<const KVt*>(K_in);
  const KVt* Vsrc = reinterpret_cast<const KVt*>(V_in);

  const int head_kv = __builtin_amdgcn_readfirstlane(blockIdx.x);
  const int split   = __builtin_amdgcn_readfirstlane(blockIdx.y);
  const int batch   = __builtin_amdgcn_readfirstlane(blockIdx.z);

  const int lane      = threadIdx.x;
  const int wmma_lane = lane % kWmmaTile;
  const int pair      = lane / kWmmaTile;

  // total_seq = seqlens_k[batch] + 1, clamped to [0, max_seq] (mirrors scalar).
  const int raw_skv = seqlens_k
      ? __builtin_amdgcn_readfirstlane(seqlens_k[batch] + 1)
      : max_seq;
  const int eff_skv = (raw_skv > 0)
      ? (raw_skv < max_seq ? raw_skv : max_seq) : 0;
  int kv_lo = 0;
  if (local_window_size > 0 && eff_skv > local_window_size) {
    kv_lo = eff_skv - local_window_size;
  }
  const int kv_range = eff_skv - kv_lo;
  const int chunk    = (kv_range + K_SPLITS - 1) / K_SPLITS;
  const int kv_start = kv_lo + chunk * split;
  int kv_end_tmp     = kv_start + chunk;
  if (kv_end_tmp > eff_skv) kv_end_tmp = eff_skv;
  const int kv_end   = kv_end_tmp;

  // Empty slice: emit identity partials so the reduce ignores this split.
  if (kv_start >= kv_end) {
    if (lane < HPG) {
      const int head_q = head_kv * HPG + lane;
      float* pd = partials +
          flash_decode_partial_offset(batch, head_q, split, H, K_SPLITS, D);
      pd[0] = -INFINITY;  // m
      pd[1] = 0.0f;       // l
      #pragma unroll
      for (int c = 0; c < D; ++c) pd[2 + c] = 0.0f;
    }
    return;
  }

  extern __shared__ char smem_dw[];
  float*    S_lds = reinterpret_cast<float*>(smem_dw);
  _Float16* K_lds = reinterpret_cast<_Float16*>(smem_dw + 16 * S_STR * sizeof(float));
  _Float16* V_lds = K_lds + BKV * DSTR;
  _Float16* P_lds = V_lds + BKV * DSTR;
  float*    ml    = reinterpret_cast<float*>(P_lds + 16 * P_STR);  // [16*3]

  const size_t kv_base = (static_cast<size_t>(batch) * G + head_kv)
                         * static_cast<size_t>(max_seq) * D;

  // Q fragments: lane wmma_lane holds row wmma_lane (head_kv*HPG + wmma_lane).
  // For INT8 the per-channel K scale is folded in here, exactly as the scalar
  // kernel does: dot = sum_e (Q[e]*Ksc[e]) * K_i8[e]. Q is 16xD per block and
  // read once, whereas scaling K would cost a multiply per staged element on
  // every tile. Fragment element e is D-index tk*16+e, which is the channel the
  // scale is indexed by.
  half16 Q_reg[Q_TILES];
  {
    const bool valid = wmma_lane < HPG;
    const int head = head_kv * HPG + wmma_lane;
    const size_t q_row = (static_cast<size_t>(batch) * H + head) * D;
    if constexpr (KV_I8) {
      const float* ks = k_scale + (size_t)head_kv * D;
      const int k_off = hipdnn_wmma_k_off(pair);
      #pragma unroll
      for (int tk = 0; tk < Q_TILES; ++tk) {
        half16 q{};
        if (valid) {
          #pragma unroll
          for (int e = 0; e < HIPDNN_WMMA_FRAG_ELEMS; ++e)
            q[e] = (_Float16)((float)Q[q_row + tk * 16 + k_off + e] *
                               ks[tk * 16 + k_off + e]);
        }
        Q_reg[tk] = q;
      }
    } else {
      #pragma unroll
      for (int tk = 0; tk < Q_TILES; ++tk)
        Q_reg[tk] = valid ? HALF16(Q[q_row + tk * 16], pair) : half16{};
    }
  }

  if (lane < 16) {
    ml[lane * 3]     = -INFINITY;
    ml[lane * 3 + 1] = 0.0f;
    ml[lane * 3 + 2] = 1.0f;
  }

  float8 O_frag[O_TILES] = {};

  const float scale_log2e = scale * LOG2E;
  const int num_tiles = (kv_end - kv_start + BKV - 1) / BKV;

  // Software pipeline: prefetch the next tile's K/V global loads into registers
  // while the current tile's GEMMs run. For INT8 the prefetch registers stay
  // int8 (KVt) and widen to fp16 only on the LDS write, so the global read
  // stays at 1 byte per element.
  constexpr int kPET = (BKV * D) / 32;          // elements per lane per tile
  // Budget the prefetch by the register footprint it costs, not by element
  // count: K and V together hold 2*kPFBytes per lane, so 128 B caps them at
  // 64 VGPRs. Element count alone would reject D=64/BKV=32, which fits.
  constexpr int kPFBytes = kPET * (int)sizeof(KVt);
  constexpr bool kPrefetch = (kPFBytes <= 128);

  // A KV row is D*sizeof(KVt) bytes; for D=64 fp16 that is exactly one 128 B
  // gfx1151 cache line. Loading one element per lane only reaches 64 B across
  // the wave, so every line is requested in two halves and RGP shows twice the
  // DRAM traffic the kernel actually needs. Loading 16 B per lane makes one
  // instruction cover four whole lines. Safe as long as the cache base is 16 B
  // aligned (hipMalloc guarantees 256 B) since every lane offset here is a
  // multiple of kVecElems.
  constexpr int kVecElems = KV_I8 ? 16 : 8;
  constexpr bool kVecLoad =
      kPrefetch && (D % kVecElems == 0) && (kPET % kVecElems == 0);
  constexpr int kSlotElems = kVecLoad ? kVecElems : 1;
  // Zero when there is no prefetch, so the staging loops below collapse away
  // instead of indexing the placeholder one-element register arrays.
  constexpr int kSlots     = kPrefetch ? (kPET / kSlotElems) : 0;

  KVt k_pf[kPrefetch ? kPET : 1];
  KVt v_pf[kPrefetch ? kPET : 1];

  // First element this lane owns in slot s. kSlotElems divides D, so a slot
  // never straddles two KV rows and the bounds check stays per row.
  auto slot_first = [&](int s) { return (lane + s * 32) * kSlotElems; };

  auto prefetch_tile = [&](int tile_kv0) {
    #pragma unroll
    for (int s = 0; s < kSlots; ++s) {
      const int i = slot_first(s);
      const int row = i / D, col = i % D;
      const int kv_pos = tile_kv0 + row;
      const bool ok = kv_pos < kv_end;
      const size_t off = kv_base + static_cast<size_t>(kv_pos) * D + col;
      if constexpr (kVecLoad) {
        bytes16 kv{}, vv{};
        if (ok) {
          kv = *reinterpret_cast<const bytes16*>(Ksrc + off);
          vv = *reinterpret_cast<const bytes16*>(Vsrc + off);
        }
        __builtin_memcpy(&k_pf[s * kSlotElems], &kv, sizeof(bytes16));
        __builtin_memcpy(&v_pf[s * kSlotElems], &vv, sizeof(bytes16));
      } else {
        k_pf[s] = ok ? Ksrc[off] : (KVt)0;
        v_pf[s] = ok ? Vsrc[off] : (KVt)0;
      }
    }
  };

  if constexpr (kPrefetch) prefetch_tile(kv_start);

  for (int t = 0; t < num_tiles; ++t) {
    const int kv0 = kv_start + t * BKV;

    if constexpr (kPrefetch) {
      // Write the prefetched tile to LDS (int8 widens to fp16 here; its scales
      // are applied outside the KV loop).
      #pragma unroll
      for (int s = 0; s < kSlots; ++s) {
        const int i0 = slot_first(s);
        const int row = i0 / D, col0 = i0 % D;
        #pragma unroll
        for (int e = 0; e < kSlotElems; ++e) {
          const int col = col0 + e;
          const KVt kval = k_pf[s * kSlotElems + e];
          const KVt vval = v_pf[s * kSlotElems + e];
          if constexpr (KV_I8) {
            // Raw int8 magnitudes; both scales are applied outside the KV loop.
            K_lds[row * DSTR + col] = (_Float16)(float)kval;
            V_lds[row * DSTR + col] = (_Float16)(float)vval;
          } else {
            K_lds[row * DSTR + col] = kval;
            V_lds[row * DSTR + col] = vval;
          }
        }
      }
      __syncthreads();
      if (t + 1 < num_tiles) prefetch_tile(kv_start + (t + 1) * BKV);
    } else {
      for (int i = lane; i < BKV * D; i += 32) {
        const int row = i / D, col = i % D;
        const int kv_pos = kv0 + row;
        if (kv_pos < kv_end) {
          const size_t off = kv_base + static_cast<size_t>(kv_pos) * D + col;
          if constexpr (KV_I8) {
            K_lds[row * DSTR + col] = (_Float16)(float)Ksrc[off];
            V_lds[row * DSTR + col] = (_Float16)(float)Vsrc[off];
          } else {
            K_lds[row * DSTR + col] = Ksrc[off];
            V_lds[row * DSTR + col] = Vsrc[off];
          }
        } else {
          K_lds[row * DSTR + col] = (_Float16)0;
          V_lds[row * DSTR + col] = (_Float16)0;
        }
      }
      __syncthreads();
    }

    // scoreGEMM: S[16, BKV] = Q . K^T (scaled, log2 domain) -> S_lds.
    #pragma unroll
    for (int sj = 0; sj < N_TILES; ++sj) {
      float8 s_frag = {};
      #pragma unroll
      for (int tk = 0; tk < Q_TILES; ++tk) {
        half16 kf = HALF16(K_lds[(sj * 16 + wmma_lane) * DSTR + tk * 16], pair);
        s_frag = HIPDNN_WMMA_F32_16X16X16_F16(Q_reg[tk], kf, s_frag);
      }
      #pragma unroll
      for (int ele = 0; ele < 8; ++ele) {
        const int r = hipdnn_wmma_acc_row(pair, ele);
        const int c = sj * 16 + wmma_lane;
        S_lds[r * S_STR + c] = s_frag[ele] * scale_log2e;
      }
    }
    __syncthreads();

    // Online softmax over the BKV columns (no causal mask at decode).
    {
      const int row  = lane / 2;
      const int half = lane & 1;
      const int col_start = half * (BKV / 2);

      float local_max = -INFINITY;
      #pragma unroll
      for (int c = 0; c < BKV / 2; ++c) {
        const int kv_pos = kv0 + col_start + c;
        float s = (kv_pos < kv_end) ? S_lds[row * S_STR + col_start + c]
                                    : -INFINITY;
        S_lds[row * S_STR + col_start + c] = s;
        local_max = fmaxf(local_max, s);
      }
      local_max = fmaxf(local_max, __shfl_xor(local_max, 1));

      const float m_old = ml[row * 3];
      const float l_old = ml[row * 3 + 1];
      const float m_new = fmaxf(m_old, local_max);
      const float correction = exp2f(m_old - m_new);

      float local_sum = 0.0f;
      #pragma unroll
      for (int c = 0; c < BKV / 2; ++c) {
        const float p = exp2f(S_lds[row * S_STR + col_start + c] - m_new);
        local_sum += p;
        P_lds[row * P_STR + col_start + c] = static_cast<_Float16>(p);
      }
      local_sum += __shfl_xor(local_sum, 1);

      if (half == 0) {
        ml[row * 3]     = m_new;
        ml[row * 3 + 1] = correction * l_old + local_sum;
        ml[row * 3 + 2] = correction;
      }
    }
    __syncthreads();

    // valueGEMM: rescale running O by correction, then O += P . V.
    #pragma unroll
    for (int tj = 0; tj < O_TILES; ++tj)
      #pragma unroll
      for (int ele = 0; ele < 8; ++ele)
        O_frag[tj][ele] *= ml[hipdnn_wmma_acc_row(pair, ele) * 3 + 2];

    #pragma unroll
    for (int tj = 0; tj < O_TILES; ++tj) {
      #pragma unroll
      for (int tk = 0; tk < N_TILES; ++tk) {
        half16 p = HALF16(P_lds[wmma_lane * P_STR + tk * 16], pair);
        half16 v;
        const int v_k_off = hipdnn_wmma_k_off(pair);
        #pragma unroll
        for (int e = 0; e < HIPDNN_WMMA_FRAG_ELEMS; ++e)
          v[e] = V_lds[(tk * 16 + v_k_off + e) * DSTR + tj * 16 + wmma_lane];
        O_frag[tj] = HIPDNN_WMMA_F32_16X16X16_F16(p, v, O_frag[tj]);
      }
    }
    __syncthreads();
  }

  // Epilogue: write UN-normalized partials (m, l, O) per query head row; the
  // shared reduce kernel performs the /l normalization + sink terms.
  // INT8 applies the deferred per-channel V scale here. O accumulates in fp32
  // over the KV range for a fixed channel, so one multiply at the end is both
  // cheaper and more accurate than scaling every staged V element.
  #pragma unroll
  for (int tj = 0; tj < O_TILES; ++tj)
    #pragma unroll
    for (int ele = 0; ele < 8; ++ele) {
      const int r = hipdnn_wmma_acc_row(pair, ele);
      const int c = tj * 16 + wmma_lane;
      if (r < HPG) {
        const int head_q = head_kv * HPG + r;
        float* pd = partials +
            flash_decode_partial_offset(batch, head_q, split, H, K_SPLITS, D);
        if constexpr (KV_I8) {
          pd[2 + c] = O_frag[tj][ele] * v_scale[(size_t)head_kv * D + c];
        } else {
          pd[2 + c] = O_frag[tj][ele];
        }
      }
    }
  if (lane < HPG) {
    const int head_q = head_kv * HPG + lane;
    float* pd = partials +
        flash_decode_partial_offset(batch, head_q, split, H, K_SPLITS, D);
    pd[0] = ml[lane * 3];
    pd[1] = ml[lane * 3 + 1];
  }
}

// FA-2 Split-K Reduction
//===----------------------------------------------------------------------===//
// Combines K_SPLITS partial (m, l, O) into the final output per (B, head_q).
//
//   global_m = max_k m_k
//   global_l = sum_k (exp2(m_k - global_m) * l_k)
//   global_O = sum_k (exp2(m_k - global_m) * O_k) / global_l
//
// Empty splits write m=-INFINITY, l=0, O=0; exp2(-INF - global_m) = 0 so
// those splits contribute nothing.
//
// Smooth softmax (attention sink): when head_sink is non-null OR
// use_smooth_softmax is set, an extra term is added to the denominator only:
//   global_l += exp(s - global_m_natural)
// where s is in natural (qk*scale) units. The partials are in log2-scaled
// space (m and l use exp2/LOG2E), so the natural identity becomes
//   exp(s - m_nat) = exp2((s - m_nat) * LOG2E) = exp2(s*LOG2E - global_m).
// The sink contributes to L only — there is no V term for it (gpt-oss-20b /
// Mistral-style attention sink).
//
// Grid: (B * H, 1) - one block per query head.
// Block: D threads. Each thread reduces one element of the d-dimension.
//===----------------------------------------------------------------------===//

template <int D>
__global__ void __launch_bounds__(D)
gqa_flash_decode_reduce_kernel(
    const float* __restrict__ partials,    // [B, H, K_SPLITS, D+2]
    __half* __restrict__ O,                // [B, 1, H, D] (BSHD with sq=1)
    int H,
    const __half* __restrict__ head_sink,  // [num_heads] or null
    int num_heads,
    int use_smooth_softmax,
    int K_SPLITS) {  // runtime split-K count (was a template param)
  const int linear = __builtin_amdgcn_readfirstlane(blockIdx.x);
  const int batch  = __builtin_amdgcn_readfirstlane(linear / H);
  const int head_q = __builtin_amdgcn_readfirstlane(linear % H);
  const int tid    = threadIdx.x;

  // Pass 1: find global_m. K_SPLITS is small (8..64) - serial per thread is fine.
  float global_m = -INFINITY;
  for (int k = 0; k < K_SPLITS; ++k) {
    const float* p = partials +
        flash_decode_partial_offset(batch, head_q, k, H, K_SPLITS, D);
    const float m_k = p[0];
    global_m = fmaxf(global_m, m_k);
  }

  // When all splits are -INF (eff_skv == 0) but a sink is configured, the
  // sink term still defines a valid (zero) output: softmax over empty +
  // sink ⇒ all weight on the sink, which contributes 0 to V. We therefore
  // write a zero output here either way.
  if (!isfinite(global_m)) {
    O[(batch * H + head_q) * D + tid] = __float2half(0.0f);
    return;
  }

  // Pass 2: accumulate l and O contributions.
  float l_sum = 0.0f;
  float O_sum = 0.0f;
  for (int k = 0; k < K_SPLITS; ++k) {
    const float* p = partials +
        flash_decode_partial_offset(batch, head_q, k, H, K_SPLITS, D);
    const float m_k = p[0];
    const float l_k = p[1];
    const float scale_k = exp2f(m_k - global_m);
    l_sum += scale_k * l_k;
    O_sum += scale_k * p[2 + tid];
  }

  // Smooth softmax: add the sink term to the denominator only. global_m is
  // in log2-scaled units (it's the max of dot*scale*LOG2E across splits), so
  // the natural-space exp(s - m_nat) becomes exp2(s*LOG2E - global_m).
  if (head_sink != nullptr) {
    const int actual_head = head_q % num_heads;
    const float s = __half2float(head_sink[actual_head]);
    l_sum += exp2f(s * (float)LOG2E - global_m);
  } else if (use_smooth_softmax) {
    // s = 0 in natural units; exp2(0*LOG2E - global_m) = exp2(-global_m).
    l_sum += exp2f(-global_m);
  }

  const float inv_l = 1.0f / fmaxf(l_sum, 1e-6f);
  O[(batch * H + head_q) * D + tid] = __float2half(O_sum * inv_l);
}

// half16_t is a WMMA A/B fragment; width matches HIPDNN_WMMA_FRAG_ELEMS (see
// the shared WMMA block above). half8_t is unrelated: a genuinely fixed
// 8-wide fp16 vector used for O storage further below, not a WMMA operand.
typedef _Float16 half16_t __attribute__((ext_vector_type(HIPDNN_WMMA_FRAG_ELEMS)));
typedef _Float16 half8_t __attribute__((ext_vector_type(8)));
typedef float float8_t __attribute__((ext_vector_type(8)));
#ifndef HALF16_LOAD
// Loads one fragment of HIPDNN_WMMA_FRAG_ELEMS contiguous K-major elements
// starting at `ptr`, offset by this lane's pair-half (see HALF16 above).
#define HALF16_LOAD(ptr, pair)                                                 \
  (*reinterpret_cast<const half16_t*>((ptr) + hipdnn_wmma_k_off(pair)))
#endif
// Always-16-wide vector, for coalesced bulk copies of a 16-element D-chunk
// from global into LDS staging (e.g. the V tile below). This is plain memory
// movement, not a WMMA operand, so unlike half16_t its width must NOT track
// HIPDNN_WMMA_FRAG_ELEMS.
typedef _Float16 half16_raw_t __attribute__((ext_vector_type(16)));
#define HALF16_RAW_LOAD(ptr) (*reinterpret_cast<const half16_raw_t*>(ptr))

// ---------------------------------------------------------------------
// Warp-private prefill kernel (v5). Grid: (num_q_tiles*B, Hq). Block: 32.
// Each wave owns M_TILES 16-row q-subtiles x all BKV columns. Only LDS is
// P_lds (C->A transpose); only sync is single-wave around it.
// ---------------------------------------------------------------------
// HAS_WINDOW is a template parameter rather than a runtime test because this
// kernel runs at the VGPR ceiling: carrying the window's per-row state live in
// the full-attention instantiation pushed M_TILES=2 from 120 to 180 bytes/lane
// of scratch (38 -> 68 spills) and cost that path 21% at deep KV.
template <int M_TILES, int BKV, int D, bool HAS_WINDOW>
__global__ void __launch_bounds__(32)
gqa_flash_prefill_v5_kernel(
    const _Float16* __restrict__ Q,       // [B, sq, Hq, D]
    const _Float16* __restrict__ Kcache,  // [B, G, max_seq, D]
    const _Float16* __restrict__ Vcache,  // [B, G, max_seq, D]
    _Float16* __restrict__ O,             // [B, sq, Hq, D]
    int B_count, int Hq, int G, int sq, int skv, int max_seq, int past_len,
    float scale, int ablate,
    const _Float16* __restrict__ head_sink,  // [num_heads] or null
    int num_heads, int smooth_softmax,
    int local_window_size)                   // <= 0 for full attention
{
    // ablation bitmask (perf diagnostics only; wrong numerics):
    //   bit0 skip softmax, bit1 skip valueGEMM, bit2 skip scoreGEMM,
    //   bit3 skip global loads (K frags + V_lds staging).
    const int abl = __builtin_amdgcn_readfirstlane(ablate);
    constexpr int kWmmaTile = 16;
    constexpr int ROWS      = M_TILES * kWmmaTile;
    constexpr int S_TILES_J = BKV / kWmmaTile;   // 16-col score tiles
    constexpr int D_TILES   = D / kWmmaTile;     // 16-col D tiles
    constexpr int P_STR     = BKV + 2;
    constexpr int V_STR     = D + 2;             // V_lds row stride (pad)
    constexpr float kLog2e  = 1.4426950408889634f;

    const int num_q_tiles = (sq + ROWS - 1) / ROWS;
    const int q_tile_idx  = __builtin_amdgcn_readfirstlane(blockIdx.x % num_q_tiles);
    const int batch       = __builtin_amdgcn_readfirstlane(blockIdx.x / num_q_tiles);
    const int head_q      = __builtin_amdgcn_readfirstlane(blockIdx.y);
    const int head_kv     = __builtin_amdgcn_readfirstlane(head_q / (Hq / G));
    const int q_start     = q_tile_idx * ROWS;

    const int lane_id   = threadIdx.x;          // 0..31
    const int wmma_lane = lane_id % kWmmaTile;   // 0..15
    const int pair      = lane_id / kWmmaTile;   // 0 -> even rows, 1 -> odd

    // Per-wave LDS: P scratch for the WMMA C->A transpose of P, and a V tile
    // staged with coalesced HALF16 loads (the direct-from-global V fragment is
    // a stride-D gather that wrecks D=128; K stays direct since it is already
    // coalesced HALF16 along D).
    extern __shared__ char smem_v5[];
    _Float16* P_lds = reinterpret_cast<_Float16*>(smem_v5);
    _Float16* V_lds = reinterpret_cast<_Float16*>(
        smem_v5 + ROWS * P_STR * sizeof(_Float16));

    const size_t kv_base =
        (static_cast<size_t>(batch) * G + head_kv) * max_seq * D;

    // ---- Q in registers: Q_reg[mi][tk], row = q_start + mi*16 + wmma_lane ----
    half16_t Q_reg[M_TILES][D_TILES];
    #pragma unroll
    for (int mi = 0; mi < M_TILES; ++mi) {
        const int q_pos = q_start + mi * 16 + wmma_lane;
        const bool valid = q_pos < sq;
        const size_t q_row_base =
            (static_cast<size_t>(batch * sq + (valid ? q_pos : 0)) * Hq + head_q) * D;
        #pragma unroll
        for (int tk = 0; tk < D_TILES; ++tk)
            Q_reg[mi][tk] = valid ? HALF16_LOAD(&Q[q_row_base + tk * 16], pair) : half16_t{};
    }

    // ---- running state in registers (per owned row) ----
    float8_t O_frag[M_TILES][D_TILES] = {};
    float    m_reg[M_TILES][8];
    float    l_reg[M_TILES][8];
    #pragma unroll
    for (int mi = 0; mi < M_TILES; ++mi)
        #pragma unroll
        for (int e = 0; e < 8; ++e) { m_reg[mi][e] = -INFINITY; l_reg[mi][e] = 0.0f; }

    // Score GEMM for one kv tile -> S_reg (registers), scaled by scale*log2e.
    // K is loaded straight from global as B-fragments (reused across mi).
    auto scoreGEMM = [&](int kv0, float8_t S_reg[M_TILES][S_TILES_J]) {
        #pragma unroll
        for (int mi = 0; mi < M_TILES; ++mi)
            #pragma unroll
            for (int sj = 0; sj < S_TILES_J; ++sj) S_reg[mi][sj] = float8_t{};
        if (abl & 4) return;  // skip scoreGEMM (also its K loads)
        #pragma unroll
        for (int sj = 0; sj < S_TILES_J; ++sj) {
            const int key = kv0 + sj * 16 + wmma_lane;
            const bool kvalid = key < skv;
            const size_t k_row = kv_base + static_cast<size_t>(kvalid ? key : 0) * D;
            #pragma unroll
            for (int tk = 0; tk < D_TILES; ++tk) {
                half16_t k_frag = (kvalid && !(abl & 8))
                                         ? HALF16_LOAD(&Kcache[k_row + tk * 16], pair)
                                         : half16_t{};
                #pragma unroll
                for (int mi = 0; mi < M_TILES; ++mi)
                    S_reg[mi][sj] = HIPDNN_WMMA_F32_16X16X16_F16(
                        Q_reg[mi][tk], k_frag, S_reg[mi][sj]);
            }
        }
        #pragma unroll
        for (int mi = 0; mi < M_TILES; ++mi)
            #pragma unroll
            for (int sj = 0; sj < S_TILES_J; ++sj)
                #pragma unroll
                for (int e = 0; e < 8; ++e)
                    S_reg[mi][sj][e] *= scale * kLog2e;
    };

    const int kv_max    = (skv - 1 < past_len + q_start + ROWS - 1)
                              ? (skv - 1) : (past_len + q_start + ROWS - 1);
    const int num_tiles = kv_max / BKV + 1;

    // With a window, every row in this Q tile has a lower bound at least as high
    // as the first row's, so starting from the first row's bound covers the whole
    // tile and the per-row mask trims the rest. This is what makes the window
    // cheaper rather than merely masked: the number of tiles walked stops growing
    // with skv once the window is filled.
    int t_start = 0;
    if constexpr (HAS_WINDOW) {
        const int kv_lo_tile = past_len + q_start - local_window_size + 1;
        t_start = (kv_lo_tile > 0) ? (kv_lo_tile / BKV) : 0;
    }

    for (int t = t_start; t < num_tiles; ++t) {
        const int kv0 = t * BKV;

        // Stage V tile into LDS with coalesced HALF16 loads along D.
        if (!(abl & 8)) {
            #pragma unroll
            for (int i = lane_id; i < BKV * D_TILES; i += 32) {
                const int kv_local = i / D_TILES;
                const int dt       = i % D_TILES;
                const int kv_pos   = kv0 + kv_local;
                half16_raw_t vv = (kv_pos < skv)
                    ? HALF16_RAW_LOAD(&Vcache[kv_base + (size_t)kv_pos * D + dt * 16])
                    : half16_raw_t{};
                *reinterpret_cast<half16_raw_t*>(&V_lds[kv_local * V_STR + dt * 16]) = vv;
            }
        }

        float8_t S_reg[M_TILES][S_TILES_J];
        scoreGEMM(kv0, S_reg);

        // ---- online softmax, fully intra-wave; writes P_lds (row-major) ----
        float corr_reg[M_TILES][8];
        #pragma unroll
        for (int mi = 0; mi < M_TILES; ++mi)
            #pragma unroll
            for (int e = 0; e < 8; ++e) corr_reg[mi][e] = 1.0f;
        if (!(abl & 1))
        #pragma unroll
        for (int mi = 0; mi < M_TILES; ++mi) {
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int row    = mi * 16 + hipdnn_wmma_acc_row(pair, e);  // local q row
                const int q_pos  = q_start + row;
                // causal mask, sliding-window lower bound, and per-row partial
                // max over this lane's columns
                const int abs_q = past_len + q_pos;
                int kv_lo = 0;
                if constexpr (HAS_WINDOW) kv_lo = abs_q - local_window_size + 1;
                float rmax = -INFINITY;
                #pragma unroll
                for (int sj = 0; sj < S_TILES_J; ++sj) {
                    const int kv_pos = kv0 + sj * 16 + wmma_lane;
                    bool masked = (kv_pos >= skv || kv_pos > abs_q);
                    if constexpr (HAS_WINDOW) masked = masked || (kv_pos < kv_lo);
                    if (masked) S_reg[mi][sj][e] = -INFINITY;
                    rmax = fmaxf(rmax, S_reg[mi][sj][e]);
                }
                rmax = fmaxf(rmax, __shfl_xor(rmax, 1));
                rmax = fmaxf(rmax, __shfl_xor(rmax, 2));
                rmax = fmaxf(rmax, __shfl_xor(rmax, 4));
                rmax = fmaxf(rmax, __shfl_xor(rmax, 8));

                const float m_old = m_reg[mi][e];
                const float m_new = fmaxf(m_old, rmax);
                // A window can mask a whole tile for a row while that row has
                // still seen nothing, leaving m_old and m_new both at -inf.
                // exp2f(-inf - -inf) is NaN, so short-circuit: contribute
                // nothing and leave the running state untouched. Causal-only
                // never reaches this, since tile 0 always holds key 0, so the
                // guard folds away in the full-attention instantiation.
                bool empty = false;
                if constexpr (HAS_WINDOW) empty = (m_new == -INFINITY);
                const float corr  = empty ? 1.0f : exp2f(m_old - m_new);
                corr_reg[mi][e]   = corr;

                float rsum = 0.0f;
                #pragma unroll
                for (int sj = 0; sj < S_TILES_J; ++sj) {
                    const float p =
                        empty ? 0.0f : exp2f(S_reg[mi][sj][e] - m_new);
                    rsum += p;
                    P_lds[row * P_STR + sj * 16 + wmma_lane] = static_cast<_Float16>(p);
                }
                rsum += __shfl_xor(rsum, 1);
                rsum += __shfl_xor(rsum, 2);
                rsum += __shfl_xor(rsum, 4);
                rsum += __shfl_xor(rsum, 8);

                m_reg[mi][e] = m_new;
                l_reg[mi][e] = corr * l_reg[mi][e] + rsum;
            }
        }
        __syncthreads();   // single-wave: P_lds + V_lds visible for valueGEMM

        // ---- value GEMM: rescale O by correction, O += P . V (V from global) ----
        #pragma unroll
        for (int mi = 0; mi < M_TILES; ++mi)
            #pragma unroll
            for (int tj = 0; tj < D_TILES; ++tj)
                #pragma unroll
                for (int e = 0; e < 8; ++e)
                    O_frag[mi][tj][e] *= corr_reg[mi][e];

        if (!(abl & 2))
        #pragma unroll
        for (int tj = 0; tj < D_TILES; ++tj) {
            #pragma unroll
            for (int tk = 0; tk < S_TILES_J; ++tk) {
                // V as B-fragment from LDS: v[e=kv] for column d = tj*16+wmma_lane
                half16_t v_frag;
                const int v_k_off = hipdnn_wmma_k_off(pair);
                #pragma unroll
                for (int e = 0; e < HIPDNN_WMMA_FRAG_ELEMS; ++e)
                    v_frag[e] = V_lds[(tk * 16 + v_k_off + e) * V_STR + tj * 16 + wmma_lane];
                #pragma unroll
                for (int mi = 0; mi < M_TILES; ++mi) {
                    half16_t p_frag = HALF16_LOAD(
                        &P_lds[(mi * 16 + wmma_lane) * P_STR + tk * 16], pair);
                    O_frag[mi][tj] = HIPDNN_WMMA_F32_16X16X16_F16(
                        p_frag, v_frag, O_frag[mi][tj]);
                }
            }
        }
        __syncthreads();   // WAR: P_lds reused by next tile's softmax
    }

    // ---- Attention sink: one extra term in the softmax denominator ----
    // m_reg / l_reg are in log2 space (scores were premultiplied by
    // scale * kLog2e above and exp2f is used for both), so m_reg == m_nat *
    // kLog2e and the natural-units sink term exp(s - m_nat) becomes
    // exp2(s * kLog2e - m_reg) -- the same identity the decode reduce kernel
    // documents. Row-invariant in the D dimension, so it is folded into l once
    // here rather than inside the D_TILES loop below.
    //
    // smooth_softmax with a null sink means s = 0, matching
    // softmax_f32_to_out_kernel. A row where every key was masked keeps
    // m_reg == -INFINITY; it must be skipped, because the exponent would
    // otherwise be +inf and the whole row would come out zero instead of
    // sink-only.
    float l_final[M_TILES][8];
    const bool apply_sink = (head_sink != nullptr) || (smooth_softmax != 0);
    float sink_logit = 0.0f;
    if (head_sink != nullptr)
        sink_logit = static_cast<float>(
            head_sink[num_heads > 0 ? (head_q % num_heads) : head_q]);
    #pragma unroll
    for (int mi = 0; mi < M_TILES; ++mi) {
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            float l = l_reg[mi][e];
            if (apply_sink && m_reg[mi][e] > -INFINITY)
                l += exp2f(sink_logit * kLog2e - m_reg[mi][e]);
            l_final[mi][e] = fmaxf(l, 1e-6f);
        }
    }

    // ---- Epilogue: O / l ----
    #pragma unroll
    for (int mi = 0; mi < M_TILES; ++mi) {
        #pragma unroll
        for (int tj = 0; tj < D_TILES; ++tj) {
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int row   = mi * 16 + hipdnn_wmma_acc_row(pair, e);
                const int c     = tj * 16 + wmma_lane;
                const int q_pos = q_start + row;
                if (q_pos < sq && c < D) {
                    const float out_val = O_frag[mi][tj][e] / l_final[mi][e];
                    O[(static_cast<size_t>(batch * sq + q_pos) * Hq + head_q) * D + c] =
                        static_cast<_Float16>(out_val);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// M-register-blocked shared-KV prefill kernel (v7). Grid: (num_q_tiles*B, Hq).
// Block: NW*32 threads. Each wave owns MT 16-row q-subtiles; K/V tiles are
// staged once into block-shared LDS, and within a wave each K/V fragment feeds
// all MT M-tiles -> the LDS read of K/V is amortized MT x.
// ---------------------------------------------------------------------
template <int NW, int BKV, int D, int MT>
__global__ void __launch_bounds__(NW * 32)
gqa_flash_prefill_v7_kernel(
    const _Float16* __restrict__ Q,       // [B, sq, Hq, D]
    const _Float16* __restrict__ Kcache,  // [B, G, max_seq, D]
    const _Float16* __restrict__ Vcache,  // [B, G, max_seq, D]
    _Float16* __restrict__ O,             // [B, sq, Hq, D]
    int B_count, int Hq, int G, int sq, int skv, int max_seq, int past_len,
    float scale, int ablate)
{
    const int abl = __builtin_amdgcn_readfirstlane(ablate);
    constexpr int kWmmaTile = 16;
    constexpr int ROWS      = NW * MT * kWmmaTile;   // query rows per block
    constexpr int S_TILES_J = BKV / kWmmaTile;
    constexpr int D_TILES   = D / kWmmaTile;
    constexpr int P_STR     = BKV + 2;
    constexpr int KV_STR    = D + 2;
    constexpr float kLog2e  = 1.4426950408889634f;

    const int num_q_tiles = (sq + ROWS - 1) / ROWS;
    const int q_tile_idx  = __builtin_amdgcn_readfirstlane(blockIdx.x % num_q_tiles);
    const int batch       = __builtin_amdgcn_readfirstlane(blockIdx.x / num_q_tiles);
    const int head_q      = __builtin_amdgcn_readfirstlane(blockIdx.y);
    const int head_kv     = __builtin_amdgcn_readfirstlane(head_q / (Hq / G));

    const int tid       = threadIdx.x;
    const int wave_id   = tid / 32;
    const int lane_id   = tid % 32;
    const int wmma_lane = lane_id % kWmmaTile;
    const int pair      = lane_id / kWmmaTile;       // 0 -> even rows, 1 -> odd

    const int q_start   = q_tile_idx * ROWS;
    const int row_base  = q_start + wave_id * MT * kWmmaTile;  // wave's first row
    const int pbase     = wave_id * MT * kWmmaTile;           // wave's P_lds offset

    // K streams from global (cache-resident) straight into WMMA fragments; only
    // V (stride-D gather) and the per-wave P transpose need LDS.
    extern __shared__ char smem_v7[];
    _Float16* V_lds = reinterpret_cast<_Float16*>(smem_v7);
    _Float16* P_lds = reinterpret_cast<_Float16*>(
        smem_v7 + (size_t)BKV * KV_STR * sizeof(_Float16));

    const size_t kv_base =
        (static_cast<size_t>(batch) * G + head_kv) * max_seq * D;

    // ---- Q in registers: Q_reg[mt][tk], row = row_base + mt*16 + wmma_lane.
    //      (Tried Q->LDS to free VGPRs -- net loss: occupancy didn't rise but the
    //       per-WMMA LDS re-reads hit the LDS port; registers win here. Also
    //       tried streaming Q from global at D=256 to shed the resident
    //       Q_reg[MT][16] -- also a net loss: the per-KV-tile Q reloads (xnum_tiles)
    //       cost more than the spill they avoid.) ----
    half16_t Q_reg[MT][D_TILES];
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        const int q_pos = row_base + mt * kWmmaTile + wmma_lane;
        const bool valid = q_pos < sq;
        const size_t q_row_base =
            (static_cast<size_t>(batch * sq + (valid ? q_pos : 0)) * Hq + head_q) * D;
        #pragma unroll
        for (int tk = 0; tk < D_TILES; ++tk)
            Q_reg[mt][tk] = valid ? HALF16_LOAD(&Q[q_row_base + tk * 16], pair) : half16_t{};
    }

    // ---- running state: O in fp32 (avoids fp16 pack/unpack per KV tile, safe under clang 23) ----
    float8_t O_acc[MT][D_TILES] = {};
    float   m_reg[MT][8];
    float   l_reg[MT][8];
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt)
        #pragma unroll
        for (int e = 0; e < 8; ++e) { m_reg[mt][e] = -INFINITY; l_reg[mt][e] = 0.0f; }

    const int kv_max    = (skv - 1 < past_len + q_start + ROWS - 1)
                              ? (skv - 1) : (past_len + q_start + ROWS - 1);
    const int num_tiles = kv_max / BKV + 1;

    for (int t = 0; t < num_tiles; ++t) {
        const int kv0 = t * BKV;

        // ---- cooperative V staging into shared LDS (K streams from global) ----
        if (!(abl & 8)) {
            #pragma unroll
            for (int i = tid; i < BKV * D_TILES; i += NW * 32) {
                const int kv_local = i / D_TILES;
                const int dt       = i % D_TILES;
                const int kv_pos   = kv0 + kv_local;
                const bool kvalid  = kv_pos < skv;
                const size_t off   = kv_base + (size_t)(kvalid ? kv_pos : 0) * D + dt * 16;
                half16_raw_t vv = kvalid ? HALF16_RAW_LOAD(&Vcache[off]) : half16_raw_t{};
                *reinterpret_cast<half16_raw_t*>(&V_lds[kv_local * KV_STR + dt * 16]) = vv;
            }
        }
        // NB: no barrier here. v7 streams K from global and softmax only touches
        // S_reg/P_lds -- nothing between here and B2 reads V_lds, so the V staging
        // overlaps the scoreGEMM+softmax work and is made visible by B2.
        // (The previous B1 was a v6 leftover from when scoreGEMM read K from LDS.)

        // ---- score GEMM: K streamed from global (cache-resident), reused / MT ----
        float8_t S_reg[MT][S_TILES_J];
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt)
            #pragma unroll
            for (int sj = 0; sj < S_TILES_J; ++sj) S_reg[mt][sj] = float8_t{};
        if (!(abl & 4)) {
            #pragma unroll
            for (int sj = 0; sj < S_TILES_J; ++sj) {
                const int key = kv0 + sj * 16 + wmma_lane;
                const bool kvalid = key < skv;
                const size_t k_row = kv_base + (size_t)(kvalid ? key : 0) * D;
                for (int tk = 0; tk < D_TILES; ++tk) {
                    half16_t k_frag = (kvalid && !(abl & 16))
                        ? HALF16_LOAD(&Kcache[k_row + tk * 16], pair) : half16_t{};
                    #pragma unroll
                    for (int mt = 0; mt < MT; ++mt)
                        S_reg[mt][sj] = HIPDNN_WMMA_F32_16X16X16_F16(
                            Q_reg[mt][tk], k_frag, S_reg[mt][sj]);
                }
            }
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt)
                #pragma unroll
                for (int sj = 0; sj < S_TILES_J; ++sj)
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) S_reg[mt][sj][e] *= scale * kLog2e;
        }

        // ---- online softmax (per M-tile, intra-wave); fold corr into O_h ----
        if (!(abl & 1)) {
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                float corr_e[8];
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int row    = hipdnn_wmma_acc_row(pair, e);
                    const int q_pos  = row_base + mt * kWmmaTile + row;
                    float rmax = -INFINITY;
                    #pragma unroll
                    for (int sj = 0; sj < S_TILES_J; ++sj) {
                        const int kv_pos = kv0 + sj * 16 + wmma_lane;
                        if (kv_pos >= skv || kv_pos > past_len + q_pos)
                            S_reg[mt][sj][e] = -INFINITY;
                        rmax = fmaxf(rmax, S_reg[mt][sj][e]);
                    }
                    rmax = fmaxf(rmax, __shfl_xor(rmax, 1));
                    rmax = fmaxf(rmax, __shfl_xor(rmax, 2));
                    rmax = fmaxf(rmax, __shfl_xor(rmax, 4));
                    rmax = fmaxf(rmax, __shfl_xor(rmax, 8));

                    const float m_old = m_reg[mt][e];
                    const float m_new = fmaxf(m_old, rmax);
                    const float corr  = exp2f(m_old - m_new);
                    corr_e[e]         = corr;

                    float rsum = 0.0f;
                    #pragma unroll
                    for (int sj = 0; sj < S_TILES_J; ++sj) {
                        const float p = exp2f(S_reg[mt][sj][e] - m_new);
                        rsum += p;
                        P_lds[(pbase + mt * kWmmaTile + row) * P_STR + sj * 16 + wmma_lane] =
                            static_cast<_Float16>(p);
                    }
                    rsum += __shfl_xor(rsum, 1);
                    rsum += __shfl_xor(rsum, 2);
                    rsum += __shfl_xor(rsum, 4);
                    rsum += __shfl_xor(rsum, 8);

                    m_reg[mt][e] = m_new;
                    l_reg[mt][e] = corr * l_reg[mt][e] + rsum;
                }
                // rescale running O by correction
                #pragma unroll
                for (int tj = 0; tj < D_TILES; ++tj)
                    #pragma unroll
                    for (int e = 0; e < 8; ++e)
                        O_acc[mt][tj][e] *= corr_e[e];
            }
        }
        __syncthreads();   // B2

        // ---- value GEMM: V gathered once per (tj,tk), reused across MT ----
        if (!(abl & 2))
        #pragma unroll
        for (int tj = 0; tj < D_TILES; ++tj) {
            #pragma unroll
            for (int tk = 0; tk < S_TILES_J; ++tk) {
                half16_t v_frag;
                const int v_k_off = hipdnn_wmma_k_off(pair);
                #pragma unroll
                for (int e = 0; e < HIPDNN_WMMA_FRAG_ELEMS; ++e)
                    v_frag[e] = V_lds[(tk * 16 + v_k_off + e) * KV_STR + tj * 16 + wmma_lane];
                #pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    half16_t p_frag = HALF16_LOAD(
                        &P_lds[(pbase + mt * kWmmaTile + wmma_lane) * P_STR + tk * 16], pair);
                    O_acc[mt][tj] = HIPDNN_WMMA_F32_16X16X16_F16(
                        p_frag, v_frag, O_acc[mt][tj]);
                }
            }
        }
        __syncthreads();   // B3: WAR -- V_lds reused next tile
    }

    // ---- Epilogue: O / l ----
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt)
        #pragma unroll
        for (int tj = 0; tj < D_TILES; ++tj) {
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int row   = hipdnn_wmma_acc_row(pair, e);
                const int c     = tj * 16 + wmma_lane;
                const int q_pos = row_base + mt * kWmmaTile + row;
                if (q_pos < sq && c < D) {
                    const float l_val   = fmaxf(l_reg[mt][e], 1e-6f);
                    const float out_val = O_acc[mt][tj][e] / l_val;
                    O[(static_cast<size_t>(batch * sq + q_pos) * Hq + head_q) * D + c] =
                        static_cast<_Float16>(out_val);
                }
            }
        }
}

// =====================================================================
// Prefill v8: D=256-specialized "output-D split + score reduction".
//
// A block runs ND waves that share the SAME query rows but each owns
// D_TILES/ND output-D tiles. The score needs the full-D contraction, so
// each wave computes a PARTIAL score over its own D slice; the partials
// are summed across waves through LDS (the secondary reduction) to form
// the full score. Every wave then runs an identical online softmax and a
// value GEMM for its own (disjoint) output-D tiles. Halving each wave's
// Q_reg/O_h footprint is what lifts occupancy at D=256 (the bottleneck
// that capped v7); the smaller ROWS also spawns more blocks, filling the
// GPU better on short prompts.
// Grid: (num_q_tiles*B, Hq). Block: ND*32.
// =====================================================================
template <int ND, int MT, int BKV, int D>
__global__ void __launch_bounds__(ND * 32)
gqa_flash_prefill_v8_kernel(
    const _Float16* __restrict__ Q,       // [B, sq, Hq, D]
    const _Float16* __restrict__ Kcache,  // [B, G, max_seq, D]
    const _Float16* __restrict__ Vcache,  // [B, G, max_seq, D]
    _Float16* __restrict__ O,             // [B, sq, Hq, D]
    int B_count, int Hq, int G, int sq, int skv, int max_seq, int past_len,
    float scale, int ablate)
{
    const int abl = __builtin_amdgcn_readfirstlane(ablate);
    constexpr int kWmmaTile = 16;
    constexpr int ROWS      = MT * kWmmaTile;
    constexpr int S_TILES_J = BKV / kWmmaTile;
    constexpr int D_TILES   = D / kWmmaTile;
    constexpr int DPG       = D_TILES / ND;    // D-tiles owned per wave
    // A WMMA accumulator holds 8 rows per lane (row = hipdnn_wmma_acc_row(pair, e)),
    // and the softmax over those rows is row-independent, so it splits ND ways
    // instead of being repeated ND times or dumped on one wave while the rest wait.
    constexpr int EPW       = 8 / ND;          // accumulator rows per wave
    static_assert(EPW * ND == 8, "ND must divide 8 for the softmax row split");
    constexpr int P_STR     = BKV + 2;
    constexpr int KV_STR    = D + 2;
    constexpr int SP_STR    = BKV;             // partial-score LDS row stride
    constexpr float kLog2e  = 1.4426950408889634f;

    const int num_q_tiles = (sq + ROWS - 1) / ROWS;
    const int q_tile_idx  = __builtin_amdgcn_readfirstlane(blockIdx.x % num_q_tiles);
    const int batch       = __builtin_amdgcn_readfirstlane(blockIdx.x / num_q_tiles);
    const int head_q      = __builtin_amdgcn_readfirstlane(blockIdx.y);
    const int head_kv     = __builtin_amdgcn_readfirstlane(head_q / (Hq / G));

    const int tid       = threadIdx.x;
    const int group     = __builtin_amdgcn_readfirstlane(tid / 32);   // 0..ND-1
    const int lane_id   = tid % 32;
    const int wmma_lane = lane_id % kWmmaTile;
    const int pair      = lane_id / kWmmaTile;
    const int dt_base   = group * DPG;         // this wave's first D-tile
    const int q_start   = q_tile_idx * ROWS;

    extern __shared__ char smem_v8[];
    _Float16* V_lds  = reinterpret_cast<_Float16*>(smem_v8);
    float*    Sp_lds = reinterpret_cast<float*>(
        smem_v8 + (size_t)BKV * KV_STR * sizeof(_Float16));
    _Float16* P_lds  = reinterpret_cast<_Float16*>(
        Sp_lds + (size_t)ND * ROWS * SP_STR);
    // Per-row softmax correction, published by the one wave that owns the
    // softmax and read by all of them; reused after the KV loop to hand out the
    // final l. ROWS floats, so 64 B at MT=1.
    float*    Cr_lds  = reinterpret_cast<float*>(
        P_lds + (size_t)ROWS * P_STR);

    const size_t kv_base =
        (static_cast<size_t>(batch) * G + head_kv) * max_seq * D;

    // ---- Q in registers: only THIS wave's D slice (DPG tiles) ----
    half16_t Q_reg[MT][DPG];
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        const int q_pos = q_start + mt * kWmmaTile + wmma_lane;
        const bool valid = q_pos < sq;
        const size_t q_row_base =
            (static_cast<size_t>(batch * sq + (valid ? q_pos : 0)) * Hq + head_q) * D;
        #pragma unroll
        for (int tk = 0; tk < DPG; ++tk)
            Q_reg[mt][tk] = valid
                ? HALF16_LOAD(&Q[q_row_base + (dt_base + tk) * 16], pair) : half16_t{};
    }

    // Accumulate O in fp32 registers. The WMMA accumulator is already fp32, so
    // the previous half8_t store meant unpacking to float, accumulating, and
    // repacking on every KV tile -- 2 conversions per element per tile, plus the
    // same round-trip again in the rescale, for no register saving that mattered
    // at the winning config. Holding fp32 also drops the per-tile rounding of a
    // running sum.
    float8_t O_acc[MT][DPG] = {};
    // Running (m, l) only for the rows this wave owns in the softmax split.
    float   m_reg[MT][EPW];
    float   l_reg[MT][EPW];
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt)
        #pragma unroll
        for (int ei = 0; ei < EPW; ++ei) { m_reg[mt][ei] = -INFINITY; l_reg[mt][ei] = 0.0f; }

    const int kv_max    = (skv - 1 < past_len + q_start + ROWS - 1)
                              ? (skv - 1) : (past_len + q_start + ROWS - 1);
    const int num_tiles = kv_max / BKV + 1;

    for (int t = 0; t < num_tiles; ++t) {
        const int kv0 = t * BKV;

        // ---- cooperative V staging (all ND waves), row-major V[kv][d] ----
        if (!(abl & 8)) {
            for (int i = tid; i < BKV * D_TILES; i += ND * 32) {
                const int kv_local = i / D_TILES;
                const int dt       = i % D_TILES;
                const int kv_pos   = kv0 + kv_local;
                const bool kvalid  = kv_pos < skv;
                const size_t off   = kv_base + (size_t)(kvalid ? kv_pos : 0) * D + dt * 16;
                half16_raw_t vv = kvalid ? HALF16_RAW_LOAD(&Vcache[off]) : half16_raw_t{};
                *reinterpret_cast<half16_raw_t*>(&V_lds[kv_local * KV_STR + dt * 16]) = vv;
            }
        }

        // ---- partial score GEMM over THIS wave's D slice (K from global) ----
        float8_t S_reg[MT][S_TILES_J];
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt)
            #pragma unroll
            for (int sj = 0; sj < S_TILES_J; ++sj) S_reg[mt][sj] = float8_t{};
        if (!(abl & 4)) {
            #pragma unroll
            for (int sj = 0; sj < S_TILES_J; ++sj) {
                const int key = kv0 + sj * 16 + wmma_lane;
                const bool kvalid = key < skv;
                const size_t k_row = kv_base + (size_t)(kvalid ? key : 0) * D;
                #pragma unroll
                for (int tk = 0; tk < DPG; ++tk) {
                    half16_t k_frag = (kvalid && !(abl & 16))
                        ? HALF16_LOAD(&Kcache[k_row + (dt_base + tk) * 16], pair) : half16_t{};
                    #pragma unroll
                    for (int mt = 0; mt < MT; ++mt)
                        S_reg[mt][sj] = HIPDNN_WMMA_F32_16X16X16_F16(
                            Q_reg[mt][tk], k_frag, S_reg[mt][sj]);
                }
            }
        }

        // ---- secondary reduction: publish partials, then sum across waves ----
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt)
            #pragma unroll
            for (int sj = 0; sj < S_TILES_J; ++sj)
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int rl  = hipdnn_wmma_acc_row(pair, e);
                    const int col = sj * 16 + wmma_lane;
                    Sp_lds[((size_t)group * ROWS + mt * 16 + rl) * SP_STR + col] =
                        S_reg[mt][sj][e];
                }
        // abl bit 32 drops all three per-tile barriers. The result is wrong, but
        // it prices the barriers before any restructuring is done to remove them:
        // this kernel runs 3 barriers per KV tile and ~583 tiles per block at
        // BKV=16, so "reduce the barrier count" is only worth doing if that shows
        // up as time.
        if (!(abl & 32)) __syncthreads();   // b_a: partials + V staging visible

        // ---- softmax, rows split ND ways across the waves ----
        // Every wave used to sum the ND partials and run a bit-identical softmax,
        // with only group 0's P store skipped for the others. Ablating this block
        // (ablate bit 1, which also drops the cross-wave partial sum below, so the
        // figure covers that read-back too) cost 136 ms of 348 ms at
        // sq=18646, the largest single item in the kernel, so repeating it ND times
        // was the main waste -- and it is why raising ND used to make this kernel
        // slower rather than faster despite the extra occupancy.
        //
        // Handing the whole softmax to group 0 instead only moves the problem: the
        // other waves then idle at b_b while group 0's copy sits on the critical
        // path. The rows are independent, so wave g takes accumulator rows
        // [g*EPW, (g+1)*EPW) and the critical path shrinks with ND instead of
        // growing. Each wave publishes the per-row correction and, after the loop,
        // its share of l; both are read back by every wave, since a wave writes
        // all 16 rows of the D tiles it owns.
        //
        // corr and l are uniform across wmma_lane (both come out of a 16-lane
        // shuffle reduction), so the 16 lanes sharing a row write the same value
        // to Cr_lds -- a same-value race, not a data race.
        if (!(abl & 1)) {
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                #pragma unroll
                for (int ei = 0; ei < EPW; ++ei) {
                    const int e     = group * EPW + ei;
                    const int rl    = hipdnn_wmma_acc_row(pair, e);
                    const int q_pos = q_start + mt * 16 + rl;

                    // Sum the ND partial scores for this row only.
                    float s_row[S_TILES_J];
                    #pragma unroll
                    for (int sj = 0; sj < S_TILES_J; ++sj) {
                        const int col = sj * 16 + wmma_lane;
                        float acc = 0.0f;
                        #pragma unroll
                        for (int g = 0; g < ND; ++g)
                            acc += Sp_lds[((size_t)g * ROWS + mt * 16 + rl) * SP_STR + col];
                        s_row[sj] = acc * scale * kLog2e;
                    }

                    float rmax = -INFINITY;
                    #pragma unroll
                    for (int sj = 0; sj < S_TILES_J; ++sj) {
                        const int kv_pos = kv0 + sj * 16 + wmma_lane;
                        if (kv_pos >= skv || kv_pos > past_len + q_pos)
                            s_row[sj] = -INFINITY;
                        rmax = fmaxf(rmax, s_row[sj]);
                    }
                    rmax = fmaxf(rmax, __shfl_xor(rmax, 1));
                    rmax = fmaxf(rmax, __shfl_xor(rmax, 2));
                    rmax = fmaxf(rmax, __shfl_xor(rmax, 4));
                    rmax = fmaxf(rmax, __shfl_xor(rmax, 8));

                    const float m_old = m_reg[mt][ei];
                    const float m_new = fmaxf(m_old, rmax);
                    const float corr  = exp2f(m_old - m_new);

                    float rsum = 0.0f;
                    #pragma unroll
                    for (int sj = 0; sj < S_TILES_J; ++sj) {
                        const float p = exp2f(s_row[sj] - m_new);
                        rsum += p;
                        P_lds[(mt * 16 + rl) * P_STR + sj * 16 + wmma_lane] =
                            static_cast<_Float16>(p);
                    }
                    rsum += __shfl_xor(rsum, 1);
                    rsum += __shfl_xor(rsum, 2);
                    rsum += __shfl_xor(rsum, 4);
                    rsum += __shfl_xor(rsum, 8);

                    m_reg[mt][ei] = m_new;
                    l_reg[mt][ei] = corr * l_reg[mt][ei] + rsum;
                    Cr_lds[mt * 16 + rl] = corr;
                }
            }
        }
        if (!(abl & 32)) __syncthreads();   // b_b: P_lds + Cr_lds visible

        // ---- every wave rescales its own O tiles by the published correction ----
        // Skipping this when corr == 1.0 (i.e. no row's running max grew, which is
        // most tiles past the first few) was tried and measured 0.3%, inside run
        // noise: these fp32 multiplies hide behind the tile's memory traffic, so
        // the ballot and the branch bought nothing and are not here.
        if (!(abl & 1)) {
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                float corr_e[8];
                #pragma unroll
                for (int e = 0; e < 8; ++e)
                    corr_e[e] = Cr_lds[mt * 16 + hipdnn_wmma_acc_row(pair, e)];
                #pragma unroll
                for (int tj = 0; tj < DPG; ++tj)
                    #pragma unroll
                    for (int e = 0; e < 8; ++e)
                        O_acc[mt][tj][e] *= corr_e[e];
            }
        }

        // ---- value GEMM: this wave's owned D tiles only ----
        if (!(abl & 2))
        for (int tj = 0; tj < DPG; ++tj) {
            const int d_tile = dt_base + tj;
            #pragma unroll
            for (int tk = 0; tk < S_TILES_J; ++tk) {
                half16_t v_frag;
                const int v_k_off = hipdnn_wmma_k_off(pair);
                #pragma unroll
                for (int e = 0; e < HIPDNN_WMMA_FRAG_ELEMS; ++e)
                    v_frag[e] = V_lds[(tk * 16 + v_k_off + e) * KV_STR + d_tile * 16 + wmma_lane];
                #pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    half16_t p_frag = HALF16_LOAD(
                        &P_lds[(mt * 16 + wmma_lane) * P_STR + tk * 16], pair);
                    O_acc[mt][tj] = HIPDNN_WMMA_F32_16X16X16_F16(
                        p_frag, v_frag, O_acc[mt][tj]);
                }
            }
        }
        if (!(abl & 32)) __syncthreads();   // b_c: WAR on V_lds/Sp_lds/P_lds
    }

    // ---- pool the per-wave l so every wave can normalise all 16 rows ----
    // The softmax split gives each wave l for only EPW of the 8 accumulator rows,
    // but a wave writes all 16 rows of the D tiles it owns. b_c above already
    // separated this from the last tile's reads of Cr_lds, so one barrier does.
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt)
        #pragma unroll
        for (int ei = 0; ei < EPW; ++ei)
            Cr_lds[mt * 16 + hipdnn_wmma_acc_row(pair, group * EPW + ei)] = l_reg[mt][ei];
    __syncthreads();
    float l_fin[MT][8];
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt)
        #pragma unroll
        for (int e = 0; e < 8; ++e)
            l_fin[mt][e] = Cr_lds[mt * 16 + hipdnn_wmma_acc_row(pair, e)];

    // ---- Epilogue: this wave writes its owned D tiles ----
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt)
        #pragma unroll
        for (int tj = 0; tj < DPG; ++tj) {
            const int c = (dt_base + tj) * 16 + wmma_lane;
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int rl    = hipdnn_wmma_acc_row(pair, e);
                const int q_pos = q_start + mt * 16 + rl;
                if (q_pos < sq && c < D) {
                    const float l_val   = fmaxf(l_fin[mt][e], 1e-6f);
                    const float out_val = O_acc[mt][tj][e] / l_val;
                    O[(static_cast<size_t>(batch * sq + q_pos) * Hq + head_q) * D + c] =
                        static_cast<_Float16>(out_val);
                }
            }
        }
}

// Fused GQA decode (sq == 1, d in {64,128,256}). One block per (batch,head_q),
// D threads; LDS-tiles TILE=8 KV rows to drive memory-level parallelism.
template <int D>
__global__ void __launch_bounds__(D)
legacy_fused_gqa_decode_kernel(
    const __half* __restrict__ Q_roped,
    const __half* __restrict__ Kcache,
    const __half* __restrict__ Vcache,
    __half* __restrict__ O,
    int H, int G, int d, int skv, int max_seq,
    float scale,
    const int* __restrict__ seqlens_k)
{
    constexpr int NUM_WAVES = D / WAVE_SIZE;
    constexpr int TILE = 8;

    const int head_q  = __builtin_amdgcn_readfirstlane(blockIdx.x % H);
    const int batch   = __builtin_amdgcn_readfirstlane(blockIdx.x / H);
    const int head_kv = __builtin_amdgcn_readfirstlane(head_q / (H / G));
    const int tid     = threadIdx.x;
    const int wave_id = tid / WAVE_SIZE;
    const int lane_id = tid % WAVE_SIZE;

    const int raw_skv = seqlens_k
        ? __builtin_amdgcn_readfirstlane(seqlens_k[batch] + 1) : skv;
    const int clamp_hi = raw_skv < max_seq ? raw_skv : max_seq;
    const int eff_skv = clamp_hi > 0 ? clamp_hi : 0;

    __shared__ float smem[NUM_WAVES];
    __shared__ _Float16 K_tile[TILE * D];
    __shared__ _Float16 V_tile[TILE * D];

    const float scale_log2e = scale * LOG2E;

    const float Q_val = __half2float(Q_roped[(batch * H + head_q) * d + tid]);

    const size_t kv_base = (size_t)__builtin_amdgcn_readfirstlane(
        (unsigned)((batch * G + head_kv) * max_seq)) * d;

    float m = -INFINITY;
    float l = 0.0f;
    float O_acc = 0.0f;

    for (int t_base = 0; t_base < eff_skv; t_base += TILE) {
        const int tile_n = (t_base + TILE <= eff_skv) ? TILE : (eff_skv - t_base);

        #pragma unroll
        for (int i = 0; i < TILE; ++i) {
            if (i < tile_n) {
                K_tile[i * D + tid] = Kcache[kv_base + (size_t)(t_base + i) * d + tid];
                V_tile[i * D + tid] = Vcache[kv_base + (size_t)(t_base + i) * d + tid];
            }
        }

        for (int t_off = 0; t_off < tile_n; ++t_off) {
            float dot = Q_val * __half2float(K_tile[t_off * D + tid]);

#pragma unroll
            for (int offset = WAVE_SIZE >> 1; offset > 0; offset >>= 1)
                dot += __shfl_xor(dot, offset);

            if (lane_id == 0)
                smem[wave_id] = dot;
            __syncthreads();

            float score = 0.0f;
            for (int w = 0; w < NUM_WAVES; ++w)
                score += smem[w];
            score *= scale_log2e;

            __syncthreads();

            float m_new = fmaxf(m, score);
            float correction = exp2f(m - m_new);
            float p = exp2f(score - m_new);
            l = correction * l + p;

            float V_val = __half2float(V_tile[t_off * D + tid]);
            O_acc = correction * O_acc + p * V_val;
            m = m_new;
        }
    }

    O_acc /= fmaxf(l, 1e-6f);
    O[(batch * H + head_q) * d + tid] = __float2half(O_acc);
}

#endif  // HIPDNN_EP_RTC_GQA_DEVICE_H
