/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * com.amd QMoE (Nemotron-H LatentMoE) sub-kernels — pure kernel launches, no
 * memory or stream management. The orchestration logic (scratch sizing,
 * bucketing, per-expert dispatch loop) lives in the runtime wrapper
 * (lib/Runtime/real/qmoe_amd.cpp). This file only adds the two kernels that
 * are genuinely new for this op (sigmoid + correction-bias routing, and the
 * relu2 activation); everything else (gather/bucket/scatter/matmul_nbits)
 * is reused unmodified from qmoe_kernel.hip / matmul_nbits_kernel.hip.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>

static constexpr int kBlock = 256;
static constexpr int kAmdRoutingBlock = 256;
// com.amd QMoE routing epsilon added to the weight-sum denominator before
// normalization, matching the sigmoid-gate MoE convention (e.g. DeepSeek-V3
// style routers): w = probs[sel] / (sum(probs[sel]) + eps).
static constexpr float kRoutingEps = 1e-20f;

__host__ __device__ static inline size_t align_up_4(size_t s) { return (s + 3) & ~size_t(3); }

// ---------------------------------------------------------------------------
// com.amd QMoE routing: dense (unquantized) logits = hidden_states @
// router_weight, sigmoid, optional correction-bias for top-k *selection*
// only, raw-sigmoid for the selected experts' *weight*, optional
// sum-normalize, then scale by routed_scaling_factor.
//
// One block per token. Threads cooperate to:
//   1. Stage the token's hidden vector into shared memory (read once).
//   2. Compute per-expert dense dot products (each thread owns a strided
//      subset of experts), sigmoid, and the correction-biased selection
//      score, both cached in shared memory.
//   3. Run k rounds of block-wide argmax over the biased score (same
//      masked-argmax approach as topk_routing_kernel in qmoe_kernel.hip),
//      recording the *raw* (unbiased) sigmoid probability as the weight.
//   4. Optionally normalize the k selected weights to sum to 1 (+eps), then
//      scale by routed_scaling_factor.
// ---------------------------------------------------------------------------
__global__ void qmoe_amd_route_kernel(
    const __half* __restrict__ hidden_states,   // [num_tokens, hidden_size]
    const __half* __restrict__ router_weight,   // [hidden_size, num_experts]
    const __half* __restrict__ correction_bias, // [num_experts] or null
    int32_t* __restrict__ expert_indices,        // [num_tokens, k]
    __half* __restrict__ expert_weights,         // [num_tokens, k]
    int64_t num_tokens,
    int64_t hidden_size,
    int64_t num_experts,
    int64_t k,
    int64_t normalize,
    int64_t use_correction_bias,
    float routed_scaling_factor) {
  const int token = static_cast<int>(blockIdx.x);
  if (token >= num_tokens) return;

  const int tid = static_cast<int>(threadIdx.x);
  const int nthreads = static_cast<int>(blockDim.x);
  const int ik = static_cast<int>(k);
  const int ne = static_cast<int>(num_experts);
  const int hs = static_cast<int>(hidden_size);

  extern __shared__ unsigned char s_raw_mem[];
  __half* s_x = reinterpret_cast<__half*>(s_raw_mem);
  float* s_prob = reinterpret_cast<float*>(
      s_raw_mem + align_up_4(static_cast<size_t>(hs) * sizeof(__half)));
  float* s_biased = s_prob + ne;

  __shared__ float s_red[kAmdRoutingBlock];
  __shared__ int s_idx[kAmdRoutingBlock];

  // Stage this token's hidden vector once; every expert's dot product reads
  // it from shared memory instead of re-fetching from global each time.
  const __half* x = hidden_states + static_cast<int64_t>(token) * hs;
  for (int h = tid; h < hs; h += nthreads) {
    s_x[h] = x[h];
  }
  __syncthreads();

  // Dense logits + sigmoid + correction-biased selection score, per expert.
  // router_weight is [hidden_size, num_experts] row-major, so for fixed h,
  // consecutive experts (== consecutive threads here) read consecutive
  // memory -- the inner loop is coalesced across the warp.
  for (int e = tid; e < ne; e += nthreads) {
    float dot = 0.0f;
    for (int h = 0; h < hs; h++) {
      dot += static_cast<float>(s_x[h]) *
             static_cast<float>(router_weight[static_cast<int64_t>(h) * ne + e]);
    }
    float prob = 1.0f / (1.0f + expf(-dot));
    s_prob[e] = prob;
    float biased = prob;
    if (use_correction_bias && correction_bias) {
      biased += static_cast<float>(correction_bias[e]);
    }
    s_biased[e] = biased;
  }
  __syncthreads();

  // k rounds of block argmax over the biased score. Ties resolve to the
  // lower expert index (matches topk_routing_kernel's convention). The
  // weight recorded is the *raw* (unbiased) sigmoid probability of the
  // winning expert, not the biased score used to select it.
  for (int i = 0; i < ik; i++) {
    float bv = -1e30f;
    int bi = ne;
    for (int e = tid; e < ne; e += nthreads) {
      float v = s_biased[e];
      if (v > bv || (v == bv && e < bi)) {
        bv = v;
        bi = e;
      }
    }
    s_red[tid] = bv;
    s_idx[tid] = bi;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
      if (tid < s) {
        if (s_red[tid + s] > s_red[tid] ||
            (s_red[tid + s] == s_red[tid] && s_idx[tid + s] < s_idx[tid])) {
          s_red[tid] = s_red[tid + s];
          s_idx[tid] = s_idx[tid + s];
        }
      }
      __syncthreads();
    }
    if (tid == 0) {
      const int win = s_idx[0];
      expert_indices[token * ik + i] = win;
      expert_weights[token * ik + i] = static_cast<__half>(s_prob[win]);
      s_biased[win] = -1e30f;
    }
    __syncthreads();
  }

  // Optional sum-normalization (+eps), then unconditional routed-scaling.
  if (tid == 0) {
    if (normalize) {
      float sum = 0.0f;
      for (int i = 0; i < ik; i++) {
        sum += static_cast<float>(expert_weights[token * ik + i]);
      }
      const float denom = sum + kRoutingEps;
      for (int i = 0; i < ik; i++) {
        float w = static_cast<float>(expert_weights[token * ik + i]) / denom;
        expert_weights[token * ik + i] =
            static_cast<__half>(w * routed_scaling_factor);
      }
    } else {
      for (int i = 0; i < ik; i++) {
        float w = static_cast<float>(expert_weights[token * ik + i]);
        expert_weights[token * ik + i] =
            static_cast<__half>(w * routed_scaling_factor);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// relu2 activation (fp16 only): out = relu(x)^2, elementwise.
// ---------------------------------------------------------------------------
__global__ void qmoe_amd_relu2_kernel(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int64_t total) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  float v = fmaxf(0.0f, static_cast<float>(input[idx]));
  output[idx] = static_cast<__half>(v * v);
}

// ===== Host launchers (extern "C") — pure kernel launches, no alloc/sync ====

extern "C" int hip_qmoe_amd_route(
    void* stream,
    const void* hidden_states,
    const void* router_weight,
    const void* correction_bias,
    void* expert_indices,
    void* expert_weights,
    int64_t num_tokens,
    int64_t hidden_size,
    int64_t num_experts,
    int64_t k,
    int64_t normalize,
    int64_t use_correction_bias,
    float routed_scaling_factor,
    int64_t element_size_bytes) {
  if (num_tokens <= 0) return 0;

  // fp16 only -- matches every other QMoE sub-kernel's constraint (the
  // target models use fp16 throughout the MoE layer).
  if (element_size_bytes != 2) {
    fprintf(stderr,
            "[custom_kernels] hip_qmoe_amd_route: only fp16 supported, got "
            "element_size %lld\n",
            (long long)element_size_bytes);
    return -1;
  }
  if (hidden_size <= 0 || num_experts <= 0 || k <= 0 || k > num_experts) {
    fprintf(stderr,
            "[custom_kernels] hip_qmoe_amd_route: invalid sizes (hidden=%lld "
            "experts=%lld k=%lld)\n",
            (long long)hidden_size, (long long)num_experts, (long long)k);
    return -1;
  }

  hipStream_t s = static_cast<hipStream_t>(stream);

  const size_t shmem =
      align_up_4(static_cast<size_t>(hidden_size) * sizeof(__half)) +
      2 * static_cast<size_t>(num_experts) * sizeof(float);
  // Guard against pathological (hidden_size, num_experts) combinations that
  // would exceed typical AMD GPU per-block LDS capacity (64 KB); fail loudly
  // instead of launching a kernel HIP itself would reject.
  constexpr size_t kMaxDynamicShmem = 64 * 1024;
  if (shmem > kMaxDynamicShmem) {
    fprintf(stderr,
            "[custom_kernels] hip_qmoe_amd_route: shared memory requirement "
            "%zu bytes exceeds %zu byte budget (hidden=%lld experts=%lld)\n",
            shmem, kMaxDynamicShmem, (long long)hidden_size,
            (long long)num_experts);
    return -1;
  }

  hipLaunchKernelGGL(
      qmoe_amd_route_kernel, dim3(static_cast<unsigned>(num_tokens)),
      dim3(kAmdRoutingBlock), shmem, s,
      static_cast<const __half*>(hidden_states),
      static_cast<const __half*>(router_weight),
      static_cast<const __half*>(correction_bias),
      static_cast<int32_t*>(expert_indices),
      static_cast<__half*>(expert_weights), num_tokens, hidden_size,
      num_experts, k, normalize, use_correction_bias, routed_scaling_factor);

  hipError_t err = hipGetLastError();
  if (err != hipSuccess)
    fprintf(stderr, "[custom_kernels] hip_qmoe_amd_route failed: %s\n",
            hipGetErrorString(err));
  return static_cast<int>(err);
}

extern "C" int hip_qmoe_amd_relu2(
    void* stream,
    const void* input,
    void* output,
    int64_t n,
    int64_t width,
    int64_t element_size_bytes) {
  if (n <= 0 || width <= 0) return 0;

  if (element_size_bytes != 2) {
    fprintf(stderr,
            "[custom_kernels] hip_qmoe_amd_relu2: only fp16 supported, got "
            "element_size %lld\n",
            (long long)element_size_bytes);
    return -1;
  }

  hipStream_t s = static_cast<hipStream_t>(stream);
  int64_t total = n * width;
  int grid = static_cast<int>((total + kBlock - 1) / kBlock);

  hipLaunchKernelGGL(qmoe_amd_relu2_kernel, dim3(grid), dim3(kBlock), 0, s,
                     static_cast<const __half*>(input),
                     static_cast<__half*>(output), total);

  hipError_t err = hipGetLastError();
  if (err != hipSuccess)
    fprintf(stderr, "[custom_kernels] hip_qmoe_amd_relu2 failed: %s\n",
            hipGetErrorString(err));
  return static_cast<int>(err);
}
