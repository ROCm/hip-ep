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
// Top-k routing: for each token, find top-k experts by probability.
// Uses insertion sort — k is small (typically 2-8).
// ---------------------------------------------------------------------------
template <typename T>
__global__ void topk_routing_kernel(
    const T* __restrict__ router_probs,
    int32_t* __restrict__ expert_indices,
    T* __restrict__ expert_weights,
    int64_t num_tokens,
    int64_t num_experts,
    int64_t k,
    int64_t normalize) {
  int64_t token = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (token >= num_tokens) return;

  const T* logits = router_probs + token * num_experts;
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
    expert_weights[token * k + i] = static_cast<T>(top_vals[i]);
  }
}

// ---------------------------------------------------------------------------
// Gather: collect rows by index.
// output[i, :] = input[token_ids[i], :]
// ---------------------------------------------------------------------------
template <typename T>
__global__ void gather_tokens_kernel(
    const T* __restrict__ input,
    T* __restrict__ gathered,
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
// In-place bias addition: data[i, j] += bias[j]
// ---------------------------------------------------------------------------
template <typename T>
__global__ void add_bias_kernel(
    T* __restrict__ data,
    const T* __restrict__ bias,
    int64_t n,
    int64_t width) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n * width) return;

  int64_t col = idx % width;
  float val = static_cast<float>(data[idx]) + static_cast<float>(bias[col]);
  data[idx] = static_cast<T>(val);
}

// ---------------------------------------------------------------------------
// SwiGLU activation (fused interleaved, swiglu_fusion=1).
//
// input:  [n, 2 * inter_size]  — interleaved: gate[0], up[0], gate[1], up[1]...
// output: [n, inter_size]
//
//   G = min(gate, limit)
//   L = clamp(linear, -limit, limit)
//   out = G * sigmoid(alpha * G) * (L + beta)
// ---------------------------------------------------------------------------
template <typename T>
__global__ void swiglu_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
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
  output[idx] = static_cast<T>(G * sigmoid_val * (L + beta));
}

// ---------------------------------------------------------------------------
// Weighted scatter-add:
//   output[token_ids[i], j] += weights[i] * expert_out[i, j]
//
// Safe when experts are processed serially (no concurrent writes to same row).
// ---------------------------------------------------------------------------
template <typename T>
__global__ void scatter_add_kernel(
    T* __restrict__ output,
    const T* __restrict__ expert_out,
    const int32_t* __restrict__ token_ids,
    const T* __restrict__ weights,
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

  output[dst_row * width + col] = static_cast<T>(out + w * val);
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

  if (element_size_bytes == 2) {
    hipLaunchKernelGGL(topk_routing_kernel<__half>, dim3(grid), dim3(kBlock),
                       0, s, static_cast<const __half*>(router_probs),
                       static_cast<int32_t*>(expert_indices),
                       static_cast<__half*>(expert_weights), num_tokens,
                       num_experts, k, normalize);
  } else if (element_size_bytes == 4) {
    hipLaunchKernelGGL(topk_routing_kernel<float>, dim3(grid), dim3(kBlock),
                       0, s, static_cast<const float*>(router_probs),
                       static_cast<int32_t*>(expert_indices),
                       static_cast<float*>(expert_weights), num_tokens,
                       num_experts, k, normalize);
  } else {
    fprintf(stderr,
            "[custom_kernels] hip_qmoe_topk_routing: unsupported elem_size "
            "%lld\n",
            (long long)element_size_bytes);
    return -1;
  }

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

  if (element_size_bytes == 2) {
    hipLaunchKernelGGL(gather_tokens_kernel<__half>, dim3(grid), dim3(kBlock),
                       0, s, static_cast<const __half*>(input),
                       static_cast<__half*>(gathered),
                       static_cast<const int32_t*>(token_ids), width, count);
  } else if (element_size_bytes == 4) {
    hipLaunchKernelGGL(gather_tokens_kernel<float>, dim3(grid), dim3(kBlock),
                       0, s, static_cast<const float*>(input),
                       static_cast<float*>(gathered),
                       static_cast<const int32_t*>(token_ids), width, count);
  } else {
    fprintf(stderr,
            "[custom_kernels] hip_qmoe_gather_tokens: unsupported elem_size "
            "%lld\n",
            (long long)element_size_bytes);
    return -1;
  }

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

  if (element_size_bytes == 2) {
    hipLaunchKernelGGL(add_bias_kernel<__half>, dim3(grid), dim3(kBlock), 0, s,
                       static_cast<__half*>(data),
                       static_cast<const __half*>(bias), n, width);
  } else if (element_size_bytes == 4) {
    hipLaunchKernelGGL(add_bias_kernel<float>, dim3(grid), dim3(kBlock), 0, s,
                       static_cast<float*>(data),
                       static_cast<const float*>(bias), n, width);
  } else {
    fprintf(stderr,
            "[custom_kernels] hip_qmoe_add_bias: unsupported elem_size "
            "%lld\n",
            (long long)element_size_bytes);
    return -1;
  }

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

  if (element_size_bytes == 2) {
    hipLaunchKernelGGL(swiglu_kernel<__half>, dim3(grid), dim3(kBlock), 0, s,
                       static_cast<const __half*>(input),
                       static_cast<__half*>(output), n, inter_size, alpha,
                       beta, limit);
  } else if (element_size_bytes == 4) {
    hipLaunchKernelGGL(swiglu_kernel<float>, dim3(grid), dim3(kBlock), 0, s,
                       static_cast<const float*>(input),
                       static_cast<float*>(output), n, inter_size, alpha,
                       beta, limit);
  } else {
    fprintf(stderr,
            "[custom_kernels] hip_qmoe_swiglu: unsupported elem_size %lld\n",
            (long long)element_size_bytes);
    return -1;
  }

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

  if (element_size_bytes == 2) {
    hipLaunchKernelGGL(scatter_add_kernel<__half>, dim3(grid), dim3(kBlock), 0,
                       s, static_cast<__half*>(output),
                       static_cast<const __half*>(expert_out),
                       static_cast<const int32_t*>(token_ids),
                       static_cast<const __half*>(weights), width, count);
  } else if (element_size_bytes == 4) {
    hipLaunchKernelGGL(scatter_add_kernel<float>, dim3(grid), dim3(kBlock), 0,
                       s, static_cast<float*>(output),
                       static_cast<const float*>(expert_out),
                       static_cast<const int32_t*>(token_ids),
                       static_cast<const float*>(weights), width, count);
  } else {
    fprintf(stderr,
            "[custom_kernels] hip_qmoe_scatter_add: unsupported elem_size "
            "%lld\n",
            (long long)element_size_bytes);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess)
    fprintf(stderr, "[custom_kernels] hip_qmoe_scatter_add failed: %s\n",
            hipGetErrorString(err));
  return static_cast<int>(err);
}
