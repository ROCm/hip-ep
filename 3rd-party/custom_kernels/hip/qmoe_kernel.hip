/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * QMoE sub-kernels — pure kernel launches, no memory or stream management.
 * The orchestration logic (hipMalloc, hipMemcpy, expert loop, hipFree) lives
 * in the runtime wrapper (lib/Runtime/real/qmoe.cpp).
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>

static constexpr int kBlock = 256;
static constexpr int kMaxK = 16;

// ---------------------------------------------------------------------------
// Top-k routing (fp16 only): for each token, find top-k experts by
// probability. Uses insertion sort — k is small (typically 2-8).
// ---------------------------------------------------------------------------
__global__ void topk_routing_kernel(
    const __half* __restrict__ router_probs,
    int32_t* __restrict__ expert_indices,
    __half* __restrict__ expert_weights,
    int64_t num_tokens,
    int64_t num_experts,
    int64_t k,
    int64_t normalize) {
  int64_t token = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (token >= num_tokens) return;

  const __half* logits = router_probs + token * num_experts;
  int ik = static_cast<int>(k < kMaxK ? k : kMaxK);

  // Pass 1: find max logit for numerically stable softmax
  float max_logit = -1e30f;
  for (int64_t e = 0; e < num_experts; e++) {
    float val = static_cast<float>(logits[e]);
    if (val > max_logit) {
      max_logit = val;
    }
  }

  // Pass 2: compute softmax denominator and find top-k by logit value.
  // Since exp() is monotonic, top-k by logit == top-k by probability.
  float exp_sum = 0.0f;
  float top_vals[kMaxK];
  int32_t top_idx[kMaxK];
  for (int i = 0; i < ik; i++) {
    top_vals[i] = -1e30f;
    top_idx[i] = -1;
  }

  for (int64_t e = 0; e < num_experts; e++) {
    float val = static_cast<float>(logits[e]);
    exp_sum += expf(val - max_logit);

    if (val > top_vals[ik - 1]) {
      top_vals[ik - 1] = val;
      top_idx[ik - 1] = static_cast<int32_t>(e);
      for (int i = ik - 2; i >= 0; i--) {
        if (top_vals[i + 1] > top_vals[i]) {
          float tv = top_vals[i];
          top_vals[i] = top_vals[i + 1];
          top_vals[i + 1] = tv;
          int32_t ti = top_idx[i];
          top_idx[i] = top_idx[i + 1];
          top_idx[i + 1] = ti;
        }
      }
    }
  }

  // Convert top-k logit values to softmax probabilities
  for (int i = 0; i < ik; i++) {
    top_vals[i] = expf(top_vals[i] - max_logit) / exp_sum;
  }

  // Optionally re-normalize top-k weights to sum to 1
  if (normalize) {
    float sum = 0.0f;
    for (int i = 0; i < ik; i++) {
      sum += top_vals[i];
    }
    if (sum > 0.0f) {
      for (int i = 0; i < ik; i++) {
        top_vals[i] /= sum;
      }
    }
  }

  for (int i = 0; i < ik; i++) {
    expert_indices[token * k + i] = top_idx[i];
    expert_weights[token * k + i] = static_cast<__half>(top_vals[i]);
  }
}

// ---------------------------------------------------------------------------
// Gather (fp16 only): collect rows by index.
// output[i, :] = input[token_ids[i], :]
// ---------------------------------------------------------------------------
__global__ void gather_tokens_kernel(
    const __half* __restrict__ input,
    __half* __restrict__ gathered,
    const int32_t* __restrict__ token_ids,
    int64_t width,
    int64_t count) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count * width) return;

  int64_t row = idx / width;
  int64_t col = idx % width;
  int64_t src_row = static_cast<int64_t>(token_ids[row]);

  gathered[idx] = input[src_row * width + col];
}

// ---------------------------------------------------------------------------
// In-place bias addition (fp16 only): data[i, j] += bias[j]
// ---------------------------------------------------------------------------
__global__ void add_bias_kernel(
    __half* __restrict__ data,
    const __half* __restrict__ bias,
    int64_t n,
    int64_t width) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n * width) return;

  int64_t col = idx % width;
  float val = static_cast<float>(data[idx]) + static_cast<float>(bias[col]);
  data[idx] = static_cast<__half>(val);
}

// ---------------------------------------------------------------------------
// SwiGLU activation (fp16 only, fused interleaved, swiglu_fusion=1).
//
// input:  [n, 2 * inter_size]  — interleaved: gate[0], up[0], gate[1], up[1]...
// output: [n, inter_size]
//
//   G = min(gate, limit)
//   L = clamp(linear, -limit, limit)
//   out = G * sigmoid(alpha * G) * (L + beta)
// ---------------------------------------------------------------------------
__global__ void swiglu_kernel(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int64_t n,
    int64_t inter_size,
    float alpha,
    float beta,
    float limit) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n * inter_size) return;

  int64_t row = idx / inter_size;
  int64_t col = idx % inter_size;

  float g = static_cast<float>(input[row * 2 * inter_size + 2 * col]);
  float l = static_cast<float>(input[row * 2 * inter_size + 2 * col + 1]);

  float G = fminf(g, limit);
  float L = fmaxf(fminf(l, limit), -limit);

  float sigmoid_val = 1.0f / (1.0f + expf(-alpha * G));
  output[idx] = static_cast<__half>(G * sigmoid_val * (L + beta));
}

// ---------------------------------------------------------------------------
// Weighted scatter-add (fp16 only):
//   output[token_ids[i], j] += weights[i] * expert_out[i, j]
//
// Safe when experts are processed serially (no concurrent writes to same row).
// ---------------------------------------------------------------------------
__global__ void scatter_add_kernel(
    __half* __restrict__ output,
    const __half* __restrict__ expert_out,
    const int32_t* __restrict__ token_ids,
    const __half* __restrict__ weights,
    int64_t width,
    int64_t count) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count * width) return;

  int64_t row = idx / width;
  int64_t col = idx % width;
  int64_t dst_row = static_cast<int64_t>(token_ids[row]);

  float w = static_cast<float>(weights[row]);
  float val = static_cast<float>(expert_out[idx]);
  float out = static_cast<float>(output[dst_row * width + col]);

  output[dst_row * width + col] = static_cast<__half>(out + w * val);
}

// ---------------------------------------------------------------------------
// GPU-side expert bucketing (Phase 2 foundation).
//
// Replaces the host-side bucket loop in wrap_qmoe that today requires a D2H
// readback of expert_indices + expert_weights followed by hipStreamSynchronize
// (24 stalls/inference for gpt-oss-20b). With this kernel, the data is
// reorganized on the device into per-expert contiguous slices that downstream
// expert dispatch can consume directly via expert_counts / expert_offsets.
//
// Inputs:
//   expert_indices [num_tokens * k] int32 (output of topk_routing)
//   expert_weights [num_tokens * k] fp16
// Outputs:
//   expert_counts    [num_experts]      int32 — count per expert
//   expert_offsets   [num_experts + 1]  int32 — exclusive prefix sum
//   sorted_token_ids [num_tokens * k]   int32 — token ids grouped by expert
//   sorted_weights   [num_tokens * k]   fp16  — weights aligned with sorted ids
//
// One block, one thread per expert. num_experts is small in practice (32 for
// gpt-oss-20b); each thread does num_tokens*k cheap comparisons. Two passes
// (count + scatter) guarantee deterministic in-expert ordering identical to
// the host version (token-major), which keeps numerical reproducibility for
// the downstream weighted scatter_add.
// ---------------------------------------------------------------------------
__global__ void bucket_tokens_kernel(
    const int32_t* __restrict__ expert_indices,
    const __half* __restrict__ expert_weights,
    int32_t* __restrict__ expert_counts,
    int32_t* __restrict__ expert_offsets,
    int32_t* __restrict__ sorted_token_ids,
    __half* __restrict__ sorted_weights,
    int num_tokens,
    int k,
    int num_experts) {
  int eid = static_cast<int>(threadIdx.x);
  if (eid >= num_experts) return;

  // Pass 1: count occurrences of this eid across all (token, slot) pairs.
  int local_count = 0;
  int total_pairs = num_tokens * k;
  for (int i = 0; i < total_pairs; i++) {
    if (expert_indices[i] == eid) local_count++;
  }
  expert_counts[eid] = local_count;

  __syncthreads();

  // Pass 2: serial prefix sum on a single thread. num_experts is small (~32)
  // so a parallel scan would cost more in __syncthreads + LDS than it saves.
  if (eid == 0) {
    int sum = 0;
    expert_offsets[0] = 0;
    for (int e = 0; e < num_experts; e++) {
      sum += expert_counts[e];
      expert_offsets[e + 1] = sum;
    }
  }

  __syncthreads();

  // Pass 3: scatter (token_id, weight) pairs into the sorted layout.
  // Iterate (t, s) in the same order as the host loop being replaced so the
  // per-expert ordering is bit-identical (matters for scatter_add reductions).
  int write_pos = expert_offsets[eid];
  for (int t = 0; t < num_tokens; t++) {
    for (int s = 0; s < k; s++) {
      int idx = t * k + s;
      if (expert_indices[idx] == eid) {
        sorted_token_ids[write_pos] = t;
        sorted_weights[write_pos] = expert_weights[idx];
        write_pos++;
      }
    }
  }
}

// ===== Host launchers (extern "C") — pure kernel launches, no alloc/sync ====

extern "C" int hip_qmoe_topk_routing(
    void* stream,
    const void* router_probs,
    void* expert_indices,
    void* expert_weights,
    int64_t num_tokens,
    int64_t num_experts,
    int64_t k,
    int64_t normalize,
    int64_t element_size_bytes) {
  if (num_tokens <= 0) return 0;

  hipStream_t s = static_cast<hipStream_t>(stream);
  int grid = static_cast<int>((num_tokens + kBlock - 1) / kBlock);

  // Only fp16 (__half, element_size_bytes==2) is supported. All QMoE sub-
  // kernels assume fp16 I/O; the target models use fp16 for router logits,
  // expert weights, and activations throughout the MoE layer.
  if (element_size_bytes != 2) {
    fprintf(stderr,
            "[custom_kernels] hip_qmoe_topk_routing: only fp16 supported, "
            "got element_size %lld\n",
            (long long)element_size_bytes);
    return -1;
  }

  hipLaunchKernelGGL(topk_routing_kernel, dim3(grid), dim3(kBlock),
                     0, s, static_cast<const __half*>(router_probs),
                     static_cast<int32_t*>(expert_indices),
                     static_cast<__half*>(expert_weights), num_tokens,
                     num_experts, k, normalize);

  hipError_t err = hipGetLastError();
  if (err != hipSuccess)
    fprintf(stderr, "[custom_kernels] hip_qmoe_topk_routing failed: %s\n",
            hipGetErrorString(err));
  return static_cast<int>(err);
}

extern "C" int hip_qmoe_gather_tokens(
    void* stream,
    const void* input,
    void* gathered,
    const void* token_ids,
    int64_t width,
    int64_t count,
    int64_t element_size_bytes) {
  if (count <= 0 || width <= 0) return 0;

  hipStream_t s = static_cast<hipStream_t>(stream);
  int64_t total = count * width;
  int grid = static_cast<int>((total + kBlock - 1) / kBlock);

  // fp16 only — see hip_qmoe_topk_routing for rationale.
  if (element_size_bytes != 2) {
    fprintf(stderr,
            "[custom_kernels] hip_qmoe_gather_tokens: only fp16 supported, "
            "got element_size %lld\n",
            (long long)element_size_bytes);
    return -1;
  }

  hipLaunchKernelGGL(gather_tokens_kernel, dim3(grid), dim3(kBlock),
                     0, s, static_cast<const __half*>(input),
                     static_cast<__half*>(gathered),
                     static_cast<const int32_t*>(token_ids), width, count);

  hipError_t err = hipGetLastError();
  if (err != hipSuccess)
    fprintf(stderr, "[custom_kernels] hip_qmoe_gather_tokens failed: %s\n",
            hipGetErrorString(err));
  return static_cast<int>(err);
}

extern "C" int hip_qmoe_add_bias(
    void* stream,
    void* data,
    const void* bias,
    int64_t n,
    int64_t width,
    int64_t element_size_bytes) {
  if (n <= 0 || width <= 0 || !bias) return 0;

  hipStream_t s = static_cast<hipStream_t>(stream);
  int64_t total = n * width;
  int grid = static_cast<int>((total + kBlock - 1) / kBlock);

  // fp16 only — see hip_qmoe_topk_routing for rationale.
  if (element_size_bytes != 2) {
    fprintf(stderr,
            "[custom_kernels] hip_qmoe_add_bias: only fp16 supported, "
            "got element_size %lld\n",
            (long long)element_size_bytes);
    return -1;
  }

  hipLaunchKernelGGL(add_bias_kernel, dim3(grid), dim3(kBlock), 0, s,
                     static_cast<__half*>(data),
                     static_cast<const __half*>(bias), n, width);

  hipError_t err = hipGetLastError();
  if (err != hipSuccess)
    fprintf(stderr, "[custom_kernels] hip_qmoe_add_bias failed: %s\n",
            hipGetErrorString(err));
  return static_cast<int>(err);
}

extern "C" int hip_qmoe_swiglu(
    void* stream,
    const void* input,
    void* output,
    int64_t n,
    int64_t inter_size,
    float alpha,
    float beta,
    float limit,
    int64_t element_size_bytes) {
  if (n <= 0 || inter_size <= 0) return 0;

  hipStream_t s = static_cast<hipStream_t>(stream);
  int64_t total = n * inter_size;
  int grid = static_cast<int>((total + kBlock - 1) / kBlock);

  // fp16 only — see hip_qmoe_topk_routing for rationale.
  if (element_size_bytes != 2) {
    fprintf(stderr,
            "[custom_kernels] hip_qmoe_swiglu: only fp16 supported, "
            "got element_size %lld\n",
            (long long)element_size_bytes);
    return -1;
  }

  hipLaunchKernelGGL(swiglu_kernel, dim3(grid), dim3(kBlock), 0, s,
                     static_cast<const __half*>(input),
                     static_cast<__half*>(output), n, inter_size, alpha,
                     beta, limit);

  hipError_t err = hipGetLastError();
  if (err != hipSuccess)
    fprintf(stderr, "[custom_kernels] hip_qmoe_swiglu failed: %s\n",
            hipGetErrorString(err));
  return static_cast<int>(err);
}

extern "C" int hip_qmoe_scatter_add(
    void* stream,
    void* output,
    const void* expert_out,
    const void* token_ids,
    const void* weights,
    int64_t width,
    int64_t count,
    int64_t element_size_bytes) {
  if (count <= 0 || width <= 0) return 0;

  hipStream_t s = static_cast<hipStream_t>(stream);
  int64_t total = count * width;
  int grid = static_cast<int>((total + kBlock - 1) / kBlock);

  // fp16 only — see hip_qmoe_topk_routing for rationale.
  if (element_size_bytes != 2) {
    fprintf(stderr,
            "[custom_kernels] hip_qmoe_scatter_add: only fp16 supported, "
            "got element_size %lld\n",
            (long long)element_size_bytes);
    return -1;
  }

  hipLaunchKernelGGL(scatter_add_kernel, dim3(grid), dim3(kBlock), 0,
                     s, static_cast<__half*>(output),
                     static_cast<const __half*>(expert_out),
                     static_cast<const int32_t*>(token_ids),
                     static_cast<const __half*>(weights), width, count);

  hipError_t err = hipGetLastError();
  if (err != hipSuccess)
    fprintf(stderr, "[custom_kernels] hip_qmoe_scatter_add failed: %s\n",
            hipGetErrorString(err));
  return static_cast<int>(err);
}

extern "C" int hip_qmoe_bucket_tokens(
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
    int64_t element_size_bytes) {
  if (num_tokens <= 0 || num_experts <= 0 || k <= 0) return 0;

  hipStream_t s = static_cast<hipStream_t>(stream);

  // fp16 only — see hip_qmoe_topk_routing for rationale.
  if (element_size_bytes != 2) {
    fprintf(stderr,
            "[custom_kernels] hip_qmoe_bucket_tokens: only fp16 supported, "
            "got element_size %lld\n",
            (long long)element_size_bytes);
    return -1;
  }

  // One block, one thread per expert. Block dim must be a multiple of the
  // wave size (32 on RDNA3) and <= 1024 (max block size). gpt-oss-20b has 32.
  if (num_experts > 1024) {
    fprintf(stderr,
            "[custom_kernels] hip_qmoe_bucket_tokens: num_experts=%lld > 1024 "
            "not supported (would exceed max block size)\n",
            (long long)num_experts);
    return -1;
  }
  int block_dim = static_cast<int>(((num_experts + 31) / 32) * 32);

  hipLaunchKernelGGL(bucket_tokens_kernel, dim3(1), dim3(block_dim), 0, s,
                     static_cast<const int32_t*>(expert_indices),
                     static_cast<const __half*>(expert_weights),
                     static_cast<int32_t*>(expert_counts),
                     static_cast<int32_t*>(expert_offsets),
                     static_cast<int32_t*>(sorted_token_ids),
                     static_cast<__half*>(sorted_weights),
                     static_cast<int>(num_tokens), static_cast<int>(k),
                     static_cast<int>(num_experts));

  hipError_t err = hipGetLastError();
  if (err != hipSuccess)
    fprintf(stderr, "[custom_kernels] hip_qmoe_bucket_tokens failed: %s\n",
            hipGetErrorString(err));
  return static_cast<int>(err);
}

// ===========================================================================
// Fused decode QMoE: 4 kernels per layer (fc1, swiglu, fc2, reduce).
// Eliminates the bucket_tokens + D2H + hipStreamSynchronize + host expert
// loop that the multi-pass path needs. Decode-only (num_tokens == 1).
//
// Inner GEMV ported from matmul_nbits_gemv_kernel:
//   - Cooperative LDS preload of input/activation vector (one global read
//     per element, then BLOCK_SIZE threads share access)
//   - uint4 (128-bit) vectorized loads of packed-int4 B = 32 nibbles/load
//   - Factored dequant: (dot - a_sum * zp) * scale  saves ~40% FLOPs
//   - Warp shuffle reduction across BLOCK_SIZE threads
//
// Grid: (N/TILE_N, k). Each block computes TILE_N output elements for one
// (slot) -- looking up the expert id from expert_indices[slot]. This gives
// (5760/8) * 4 = 2880 blocks for FC1 at gpt-oss decode -- 720x more
// parallelism than the per-(token, slot) layout my first attempt used.
//
// FP16 only. INT4 (bits=4) packed nibbles, packed-zp layout matching
// matmul_nbits convention. Requires hidden % 32 == 0 (uint4 main loop).
// ===========================================================================

// FC1 + SwiGLU fused: input [hidden] -> act_out [k, inter]. Per (slot,
// output-tile). Each block computes TILE_N gate/linear matmul outputs
// (TILE_N must be even), then pair-wise SwiGLU into TILE_N/2 act_out
// elements at indices [n_base/2 .. n_base/2 + TILE_N/2). Skips the
// separate qmoe_decode_swiglu_kernel launch and the [k, fusion_inter]
// fc1_out scratch round-trip.
template <int BLOCK_SIZE, int TILE_N>
__global__ void qmoe_decode_fc1_kernel(
    const __half*  __restrict__ input,           // [hidden]
    const int32_t* __restrict__ expert_indices,  // [k]
    const uint8_t* __restrict__ fc1_w,           // [E, fusion_inter, hidden/2]
    const __half*  __restrict__ fc1_s,           // [E, fusion_inter, n_blk_in]
    const uint8_t* __restrict__ fc1_zp_packed,   // [E, fusion_inter, ceil(n_blk_in/2)] or null
    const __half*  __restrict__ fc1_b,           // [E, fusion_inter] or null
    __half*        __restrict__ act_out,         // [k, inter]
    int hidden,
    int fusion_inter,
    int n_blk_in,
    int block_size,
    float swiglu_alpha,
    float swiglu_beta,
    float swiglu_limit) {
  const int n_base = blockIdx.x * TILE_N;
  const int slot   = blockIdx.y;
  const int tid    = threadIdx.x;
  const int E      = expert_indices[slot];

  // No LDS preload of the input vector: each input element is read exactly
  // once per block (threads stride through K-tiles uniquely), so an LDS
  // copy just adds a __syncthreads + duplicated traffic without
  // amortization. The 5760-byte input is shared across all 720 blocks of
  // this slot via L2/MALL cache instead.

  // Per-expert weight slice.
  const size_t  W_row_bytes = static_cast<size_t>(hidden) / 2;
  const uint8_t* W = fc1_w + static_cast<size_t>(E) * fusion_inter * W_row_bytes;
  const __half*  S = fc1_s + static_cast<size_t>(E) * fusion_inter * n_blk_in;
  const int      packed_zp_stride = (n_blk_in + 1) / 2;
  const uint8_t* Z = fc1_zp_packed
                       ? fc1_zp_packed + static_cast<size_t>(E) * fusion_inter * packed_zp_stride
                       : nullptr;
  const __half*  Bv = fc1_b ? fc1_b + static_cast<size_t>(E) * fusion_inter : nullptr;

  const int bs_shift = __builtin_ctz(static_cast<unsigned>(block_size));

  float partial[TILE_N];
  #pragma unroll
  for (int t = 0; t < TILE_N; t++) partial[t] = 0.0f;

  // Pre-cache row pointers / scale rows for each output column in tile.
  const uint8_t* W_rows[TILE_N];
  const __half*  S_rows[TILE_N];
  const uint8_t* Z_rows[TILE_N];
  int            n_idx[TILE_N];
  #pragma unroll
  for (int t = 0; t < TILE_N; t++) {
    int n = n_base + t;
    int sn = (n < fusion_inter) ? n : 0;
    n_idx[t]  = sn;
    W_rows[t] = W + static_cast<size_t>(sn) * W_row_bytes;
    S_rows[t] = S + static_cast<size_t>(sn) * n_blk_in;
    Z_rows[t] = Z ? Z + static_cast<size_t>(sn) * packed_zp_stride : nullptr;
  }
  const bool has_zp = (Z != nullptr);

  // ---- Phase 1: 32-nibble (uint4 = 128-bit) main loop ----
  const int num_u128 = hidden / 32;
  for (int idx = tid; idx < num_u128; idx += BLOCK_SIZE) {
    const int k32 = idx * 32;

    float a_vals[32];
    float a_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
      a_vals[i] = static_cast<float>(input[k32 + i]);
      a_sum += a_vals[i];
    }
    const int   grp = k32 >> bs_shift;
    const float a_sum_x_default_zp = a_sum * 8.0f;

    uint4 b_pk[TILE_N];
    float sv[TILE_N];
    #pragma unroll
    for (int t = 0; t < TILE_N; t++) {
      b_pk[t] = *reinterpret_cast<const uint4*>(&W_rows[t][idx * 16]);
      sv[t]   = static_cast<float>(S_rows[t][grp]);
    }

    #pragma unroll
    for (int t = 0; t < TILE_N; t++) {
      float dot = 0.0f;
      #pragma unroll
      for (int i = 0; i < 8; i++)
        dot = __fmaf_rn(a_vals[i],
                        static_cast<float>((b_pk[t].x >> (i * 4)) & 0xF), dot);
      #pragma unroll
      for (int i = 0; i < 8; i++)
        dot = __fmaf_rn(a_vals[8 + i],
                        static_cast<float>((b_pk[t].y >> (i * 4)) & 0xF), dot);
      #pragma unroll
      for (int i = 0; i < 8; i++)
        dot = __fmaf_rn(a_vals[16 + i],
                        static_cast<float>((b_pk[t].z >> (i * 4)) & 0xF), dot);
      #pragma unroll
      for (int i = 0; i < 8; i++)
        dot = __fmaf_rn(a_vals[24 + i],
                        static_cast<float>((b_pk[t].w >> (i * 4)) & 0xF), dot);

      if (has_zp) {
        uint8_t zp_byte = Z_rows[t][grp >> 1];
        float   zp = static_cast<float>(
                       (grp & 1) ? (zp_byte >> 4) : (zp_byte & 0x0F));
        partial[t] += (dot - a_sum * zp) * sv[t];
      } else {
        partial[t] += (dot - a_sum_x_default_zp) * sv[t];
      }
    }
  }

  // ---- Reduction: warp shuffle then LDS for cross-warp ----
  // After reduction we pair (gate, linear) outputs at indices (2*i, 2*i+1)
  // within the TILE_N matmul outputs and write the SwiGLU result directly to
  // act_out[slot, sw_base+i] where sw_base = n_base/2. This eliminates the
  // separate qmoe_decode_swiglu_kernel launch (saves 1 launch per layer).
  // Requires TILE_N to be even and n_base to be even (callers ensure both).
  constexpr int WARP_SIZE = 32;
  const int warp_id = tid / WARP_SIZE;
  const int lane    = tid % WARP_SIZE;
  constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
  static_assert((TILE_N & 1) == 0, "TILE_N must be even for SwiGLU pairing");
  constexpr int TILE_OUT = TILE_N / 2;
  const int sw_base = n_base / 2;
  const int inter   = fusion_inter / 2;

  #pragma unroll
  for (int t = 0; t < TILE_N; t++) {
    for (int off = WARP_SIZE >> 1; off > 0; off >>= 1)
      partial[t] += __shfl_down(partial[t], off);
  }

  if constexpr (BLOCK_SIZE <= WARP_SIZE) {
    if (lane == 0) {
      #pragma unroll
      for (int p = 0; p < TILE_OUT; p++) {
        int n_g = n_base + 2 * p;
        int n_l = n_g + 1;
        if (n_g < fusion_inter && n_l < fusion_inter) {
          float g = partial[2 * p];
          float l = partial[2 * p + 1];
          if (Bv) {
            g += static_cast<float>(Bv[n_g]);
            l += static_cast<float>(Bv[n_l]);
          }
          float G = fminf(g, swiglu_limit);
          float L = fmaxf(fminf(l, swiglu_limit), -swiglu_limit);
          float sig = 1.0f / (1.0f + expf(-swiglu_alpha * G));
          act_out[slot * inter + sw_base + p] =
              static_cast<__half>(G * sig * (L + swiglu_beta));
        }
      }
    }
  } else {
    __shared__ float warp_sums[TILE_N * NUM_WARPS];
    if (lane == 0) {
      #pragma unroll
      for (int t = 0; t < TILE_N; t++)
        warp_sums[t * NUM_WARPS + warp_id] = partial[t];
    }
    __syncthreads();
    if (warp_id == 0) {
      float reduced[TILE_N];
      #pragma unroll
      for (int t = 0; t < TILE_N; t++) {
        float v = (lane < NUM_WARPS) ? warp_sums[t * NUM_WARPS + lane] : 0.0f;
        for (int off = WARP_SIZE >> 1; off > 0; off >>= 1)
          v += __shfl_down(v, off);
        reduced[t] = v;
      }
      if (lane == 0) {
        #pragma unroll
        for (int p = 0; p < TILE_OUT; p++) {
          int n_g = n_base + 2 * p;
          int n_l = n_g + 1;
          if (n_g < fusion_inter && n_l < fusion_inter) {
            float g = reduced[2 * p];
            float l = reduced[2 * p + 1];
            if (Bv) {
              g += static_cast<float>(Bv[n_g]);
              l += static_cast<float>(Bv[n_l]);
            }
            float G = fminf(g, swiglu_limit);
            float L = fmaxf(fminf(l, swiglu_limit), -swiglu_limit);
            float sig = 1.0f / (1.0f + expf(-swiglu_alpha * G));
            act_out[slot * inter + sw_base + p] =
                static_cast<__half>(G * sig * (L + swiglu_beta));
          }
        }
      }
    }
  }
}

// FC2: act_out [k, inter] -> slot_buf [k, hidden] (pre-multiplied by routing
// weight so the reduce kernel is a pure sum). Per (slot, output-tile).
template <int BLOCK_SIZE, int TILE_N>
__global__ void qmoe_decode_fc2_kernel(
    const __half*  __restrict__ act_out,         // [k, inter]
    const int32_t* __restrict__ expert_indices,  // [k]
    const __half*  __restrict__ expert_weights,  // [k]
    const uint8_t* __restrict__ fc2_w,           // [E, hidden, inter/2]
    const __half*  __restrict__ fc2_s,           // [E, hidden, n_blk_mid]
    const uint8_t* __restrict__ fc2_zp_packed,   // [E, hidden, ceil(n_blk_mid/2)] or null
    const __half*  __restrict__ fc2_b,           // [E, hidden] or null
    __half*        __restrict__ slot_buf,        // [k, hidden]
    int inter,
    int hidden,
    int n_blk_mid,
    int block_size) {
  const int n_base = blockIdx.x * TILE_N;
  const int slot   = blockIdx.y;
  const int tid    = threadIdx.x;
  const int E      = expert_indices[slot];
  const float wt   = static_cast<float>(expert_weights[slot]);

  // No LDS preload of the activation vector: each element is read once per
  // block (threads stride uniquely through K-tiles); LDS would just add a
  // __syncthreads + duplicated traffic. The 5760-byte slot activation is
  // shared across the (hidden/TILE_N) blocks of this slot via L2/MALL cache.
  const __half* a_in = act_out + slot * inter;

  const size_t  W_row_bytes = static_cast<size_t>(inter) / 2;
  const uint8_t* W = fc2_w + static_cast<size_t>(E) * hidden * W_row_bytes;
  const __half*  S = fc2_s + static_cast<size_t>(E) * hidden * n_blk_mid;
  const int      packed_zp_stride = (n_blk_mid + 1) / 2;
  const uint8_t* Z = fc2_zp_packed
                       ? fc2_zp_packed + static_cast<size_t>(E) * hidden * packed_zp_stride
                       : nullptr;
  const __half*  Bv = fc2_b ? fc2_b + static_cast<size_t>(E) * hidden : nullptr;

  const int bs_shift = __builtin_ctz(static_cast<unsigned>(block_size));

  float partial[TILE_N];
  #pragma unroll
  for (int t = 0; t < TILE_N; t++) partial[t] = 0.0f;

  const uint8_t* W_rows[TILE_N];
  const __half*  S_rows[TILE_N];
  const uint8_t* Z_rows[TILE_N];
  int            n_idx[TILE_N];
  #pragma unroll
  for (int t = 0; t < TILE_N; t++) {
    int n = n_base + t;
    int sn = (n < hidden) ? n : 0;
    n_idx[t]  = sn;
    W_rows[t] = W + static_cast<size_t>(sn) * W_row_bytes;
    S_rows[t] = S + static_cast<size_t>(sn) * n_blk_mid;
    Z_rows[t] = Z ? Z + static_cast<size_t>(sn) * packed_zp_stride : nullptr;
  }
  const bool has_zp = (Z != nullptr);

  const int num_u128 = inter / 32;
  for (int idx = tid; idx < num_u128; idx += BLOCK_SIZE) {
    const int k32 = idx * 32;

    float a_vals[32];
    float a_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
      a_vals[i] = static_cast<float>(a_in[k32 + i]);
      a_sum += a_vals[i];
    }
    const int   grp = k32 >> bs_shift;
    const float a_sum_x_default_zp = a_sum * 8.0f;

    uint4 b_pk[TILE_N];
    float sv[TILE_N];
    #pragma unroll
    for (int t = 0; t < TILE_N; t++) {
      b_pk[t] = *reinterpret_cast<const uint4*>(&W_rows[t][idx * 16]);
      sv[t]   = static_cast<float>(S_rows[t][grp]);
    }

    #pragma unroll
    for (int t = 0; t < TILE_N; t++) {
      float dot = 0.0f;
      #pragma unroll
      for (int i = 0; i < 8; i++)
        dot = __fmaf_rn(a_vals[i],
                        static_cast<float>((b_pk[t].x >> (i * 4)) & 0xF), dot);
      #pragma unroll
      for (int i = 0; i < 8; i++)
        dot = __fmaf_rn(a_vals[8 + i],
                        static_cast<float>((b_pk[t].y >> (i * 4)) & 0xF), dot);
      #pragma unroll
      for (int i = 0; i < 8; i++)
        dot = __fmaf_rn(a_vals[16 + i],
                        static_cast<float>((b_pk[t].z >> (i * 4)) & 0xF), dot);
      #pragma unroll
      for (int i = 0; i < 8; i++)
        dot = __fmaf_rn(a_vals[24 + i],
                        static_cast<float>((b_pk[t].w >> (i * 4)) & 0xF), dot);

      if (has_zp) {
        uint8_t zp_byte = Z_rows[t][grp >> 1];
        float   zp = static_cast<float>(
                       (grp & 1) ? (zp_byte >> 4) : (zp_byte & 0x0F));
        partial[t] += (dot - a_sum * zp) * sv[t];
      } else {
        partial[t] += (dot - a_sum_x_default_zp) * sv[t];
      }
    }
  }

  constexpr int WARP_SIZE = 32;
  const int warp_id = tid / WARP_SIZE;
  const int lane    = tid % WARP_SIZE;
  constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

  #pragma unroll
  for (int t = 0; t < TILE_N; t++) {
    for (int off = WARP_SIZE >> 1; off > 0; off >>= 1)
      partial[t] += __shfl_down(partial[t], off);
  }

  if constexpr (BLOCK_SIZE <= WARP_SIZE) {
    if (lane == 0) {
      #pragma unroll
      for (int t = 0; t < TILE_N; t++) {
        int n = n_base + t;
        if (n < hidden) {
          float v = partial[t];
          if (Bv) v += static_cast<float>(Bv[n]);
          slot_buf[slot * hidden + n] = static_cast<__half>(wt * v);
        }
      }
    }
  } else {
    __shared__ float warp_sums[TILE_N * NUM_WARPS];
    if (lane == 0) {
      #pragma unroll
      for (int t = 0; t < TILE_N; t++)
        warp_sums[t * NUM_WARPS + warp_id] = partial[t];
    }
    __syncthreads();
    if (warp_id == 0) {
      #pragma unroll
      for (int t = 0; t < TILE_N; t++) {
        float v = (lane < NUM_WARPS) ? warp_sums[t * NUM_WARPS + lane] : 0.0f;
        for (int off = WARP_SIZE >> 1; off > 0; off >>= 1)
          v += __shfl_down(v, off);
        if (lane == 0) {
          int n = n_base + t;
          if (n < hidden) {
            if (Bv) v += static_cast<float>(Bv[n]);
            slot_buf[slot * hidden + n] = static_cast<__half>(wt * v);
          }
        }
      }
    }
  }
}

// Reduce: output[n] = sum over slot of slot_buf[slot, n]. Pure sum.
__global__ void qmoe_decode_reduce_kernel(
    const __half* __restrict__ slot_buf,  // [k, hidden]
    __half*       __restrict__ output,    // [hidden]
    int hidden,
    int k) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= hidden) return;
  float acc = 0.0f;
  #pragma unroll 4
  for (int s = 0; s < k; s++) {
    acc += static_cast<float>(slot_buf[s * hidden + n]);
  }
  output[n] = static_cast<__half>(acc);
}

// ---------------------------------------------------------------------------
// Decode launchers. Hand-picked BLOCK_SIZE=256, TILE_N=8 (a strong default
// for matmul_nbits autotune at gpt-oss-20b shapes; can later wrap in a
// tiny per-shape autotune like matmul_nbits if needed).
// ---------------------------------------------------------------------------

extern "C" int hip_qmoe_decode_fused(
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
    int64_t element_size_bytes) {
  if (element_size_bytes != 2) {
    fprintf(stderr, "[custom_kernels] hip_qmoe_decode_fused: only fp16 "
                    "supported (got element_size %lld)\n",
            (long long)element_size_bytes);
    return -1;
  }
  if (block_size <= 0 || (block_size & 1) != 0) {
    fprintf(stderr, "[custom_kernels] hip_qmoe_decode_fused: invalid "
                    "block_size=%lld\n",
            (long long)block_size);
    return -1;
  }
  if (hidden_size <= 0 || inter_size <= 0 || k <= 0) {
    fprintf(stderr, "[custom_kernels] hip_qmoe_decode_fused: invalid sizes\n");
    return -1;
  }
  if ((hidden_size % 32) != 0 || (inter_size % 32) != 0) {
    fprintf(stderr, "[custom_kernels] hip_qmoe_decode_fused: requires "
                    "hidden_size %% 32 == 0 and inter_size %% 32 == 0 "
                    "(got %lld, %lld)\n",
            (long long)hidden_size, (long long)inter_size);
    return -1;
  }

  hipStream_t s = static_cast<hipStream_t>(stream);
  // BS=64 is the sweet spot here: K-tile count num_u128 = hidden/32 = 90
  // for gpt-oss-20b. With BS=256 ~65% of threads do zero iterations of the
  // main loop (only tid<90 enters the loop). BS=64 gives every thread 1-2
  // iterations and fits 2 wave32 per block (no LDS preload needed; the
  // cross-warp warp_sums array is only TILE_N*NUM_WARPS = 16 floats).
  constexpr int BS = 64;
  // TILE_N must be even for FC1 (gate/linear paired into SwiGLU). 8 keeps
  // register pressure modest while computing 4 SwiGLU outputs/block.
  constexpr int TN = 8;
  const int fusion_inter = static_cast<int>(2 * inter_size);
  const int n_blk_in  = static_cast<int>((hidden_size + block_size - 1) / block_size);
  const int n_blk_mid = static_cast<int>((inter_size  + block_size - 1) / block_size);

  // FC1 + SwiGLU fused: grid (fusion_inter/TN, k). Each block produces TN
  // matmul outputs internally and pair-wise SwiGLUs them down to TN/2
  // act_out elements. No dynamic LDS (input read straight from L2/MALL).
  {
    dim3 grid(static_cast<unsigned>((fusion_inter + TN - 1) / TN),
              static_cast<unsigned>(k));
    hipLaunchKernelGGL(HIP_KERNEL_NAME(qmoe_decode_fc1_kernel<BS, TN>),
                       grid, dim3(BS), 0, s,
                       static_cast<const __half*>(input),
                       static_cast<const int32_t*>(expert_indices),
                       static_cast<const uint8_t*>(fc1_weights),
                       static_cast<const __half*>(fc1_scales),
                       static_cast<const uint8_t*>(fc1_zero_points),
                       static_cast<const __half*>(fc1_bias),
                       static_cast<__half*>(act_out),
                       static_cast<int>(hidden_size), fusion_inter,
                       n_blk_in, static_cast<int>(block_size),
                       swiglu_alpha, swiglu_beta, swiglu_limit);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) {
      fprintf(stderr, "[custom_kernels] qmoe_decode_fc1 failed: %s\n",
              hipGetErrorString(e));
      return static_cast<int>(e);
    }
  }

  // FC2 + weighted write to slot_buf: grid (hidden/TN, k). No LDS preload.
  {
    dim3 grid(static_cast<unsigned>((hidden_size + TN - 1) / TN),
              static_cast<unsigned>(k));
    hipLaunchKernelGGL(HIP_KERNEL_NAME(qmoe_decode_fc2_kernel<BS, TN>),
                       grid, dim3(BS), 0, s,
                       static_cast<const __half*>(act_out),
                       static_cast<const int32_t*>(expert_indices),
                       static_cast<const __half*>(expert_weights),
                       static_cast<const uint8_t*>(fc2_weights),
                       static_cast<const __half*>(fc2_scales),
                       static_cast<const uint8_t*>(fc2_zero_points),
                       static_cast<const __half*>(fc2_bias),
                       static_cast<__half*>(slot_buf),
                       static_cast<int>(inter_size),
                       static_cast<int>(hidden_size),
                       n_blk_mid, static_cast<int>(block_size));
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) {
      fprintf(stderr, "[custom_kernels] qmoe_decode_fc2 failed: %s\n",
              hipGetErrorString(e));
      return static_cast<int>(e);
    }
  }

  // Reduce: grid (hidden/kBlock).
  {
    int grid = static_cast<int>((hidden_size + kBlock - 1) / kBlock);
    hipLaunchKernelGGL(qmoe_decode_reduce_kernel, dim3(grid), dim3(kBlock), 0, s,
                       static_cast<const __half*>(slot_buf),
                       static_cast<__half*>(output),
                       static_cast<int>(hidden_size), static_cast<int>(k));
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) {
      fprintf(stderr, "[custom_kernels] qmoe_decode_reduce failed: %s\n",
              hipGetErrorString(e));
      return static_cast<int>(e);
    }
  }

  return 0;
}
