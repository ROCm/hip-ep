/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * com.amd QMoE (LatentMoE) sub-kernels — pure kernel launches, no
 * memory or stream management. The orchestration logic (scratch sizing,
 * per-expert dispatch loop) lives in the runtime wrapper
 * (lib/Runtime/real/qmoe_amd.cpp). This file adds the kernels that are
 * genuinely new for this op:
 *   - sigmoid + correction-bias routing,
 *   - the relu2 activation,
 *   - expert bucketing sized for this op's much larger (num_tokens * k) pair
 *     space than the shared qmoe_kernel.hip version targets,
 *   - a fully fused routed branch for decode (num_tokens == 1).
 * Everything else (gather/scatter/matmul_nbits) is reused unmodified from
 * qmoe_kernel.hip / matmul_nbits_kernel.hip.
 */

#include "hip_custom_kernels.h"
#include "hip_arch_compat.h"
#include "debug_log.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>

static constexpr int kBlock = 256;
static constexpr int kAmdRoutingBlock = 256;
static_assert((kAmdRoutingBlock & (kAmdRoutingBlock - 1)) == 0,
              "the block-wide argmax reduction halves the active thread count "
              "each step, so the routing block size must be a power of two");
// Experts per routing thread. 8 fp16 weights are exactly one uint4, which is
// what lets the logits loop amortize one 16-byte load over 8 MACs.
static constexpr int kAmdRouteExpertsPerLane = 8;
// com.amd QMoE routing epsilon added to the weight-sum denominator before
// normalization, matching the sigmoid-gate MoE convention:
//   w = probs[sel] / (sum(probs[sel]) + eps).
static constexpr float kRoutingEps = 1e-20f;

// ---------------------------------------------------------------------------
// Routing thread geometry.
//
// The routing block is viewed as a 2-D (expert-lane, hidden-chunk) grid:
// lane `el` owns the 8-expert group `el` (and every `lanes`-th group after
// it), chunk `hc` owns the hidden slice h = hc, hc + chunks, ... The two
// degrees of freedom exist because router_weight is [hidden_size,
// num_experts]: coalescing wants consecutive threads to read consecutive
// experts, while keeping all threads busy needs the reduction axis split as
// well once the expert count drops below the block size.
//
// Host and device must agree exactly (the host sizes the dynamic LDS from
// this), hence the shared __host__ __device__ helper.
// ---------------------------------------------------------------------------
struct AmdRouteGeom {
  int groups;  // ceil(num_experts / kAmdRouteExpertsPerLane)
  int lanes;   // expert lanes per hidden chunk
  int chunks;  // hidden chunks summed together per expert
};

__host__ __device__ static inline AmdRouteGeom amd_route_geom(int num_experts,
                                                              int nthreads) {
  int groups = (num_experts + kAmdRouteExpertsPerLane - 1) /
               kAmdRouteExpertsPerLane;
  if (groups < 1) groups = 1;
  const int lanes = groups < nthreads ? groups : nthreads;
  int chunks = nthreads / lanes;
  if (chunks < 1) chunks = 1;
  AmdRouteGeom g;
  g.groups = groups;
  g.lanes = lanes;
  g.chunks = chunks;
  return g;
}

// ---------------------------------------------------------------------------
// com.amd QMoE routing: dense (unquantized) logits = hidden_states @
// router_weight, sigmoid, optional correction-bias for top-k *selection*
// only, raw-sigmoid for the selected experts' *weight*, optional
// sum-normalize, then scale by routed_scaling_factor.
//
// One block per token. Threads cooperate to:
//   1. Compute per-expert dense dot products (see AmdRouteGeom), reduce the
//      per-chunk partials, and cache sigmoid + correction-biased selection
//      score in shared memory. Everything stays in fp32 up to the top-k
//      comparison; only the emitted weight is narrowed to fp16.
//   2. Run k rounds of block-wide argmax over the biased score (same
//      masked-argmax approach as topk_routing_kernel in qmoe_kernel.hip),
//      recording the *raw* (unbiased) sigmoid probability as the weight.
//   3. Optionally normalize the k selected weights to sum to 1 (+eps), then
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

  const AmdRouteGeom geom = amd_route_geom(ne, nthreads);
  const size_t part_floats =
      static_cast<size_t>(geom.chunks) * geom.groups * kAmdRouteExpertsPerLane;

  extern __shared__ float s_dyn[];
  float* s_part = s_dyn;                 // [chunks][groups][8] partial dots
  float* s_prob = s_dyn + part_floats;   // [num_experts] raw sigmoid
  float* s_biased = s_prob + ne;         // [num_experts] selection score

  __shared__ float s_red[kAmdRoutingBlock];
  __shared__ int s_idx[kAmdRoutingBlock];

  // Dense logits. Each lane holds 8 fp32 accumulators in registers and reads
  // one uint4 of router_weight per hidden element, so the inner loop costs one
  // global load per 8 MACs instead of one per MAC. Every lane of a wave shares
  // the same h, which makes hidden_states[h] a broadcast read and removes the
  // need to stage the token's hidden vector in LDS at all.
  const int el = tid % geom.lanes;
  const int hc = tid / geom.lanes;

  if (hc < geom.chunks) {
    const __half* x = hidden_states + static_cast<int64_t>(token) * hs;
    // The uint4 load needs a 16-byte aligned row offset, which holds only when
    // the expert count is itself a multiple of the 8-expert lane width.
    const bool vectorized = (ne % kAmdRouteExpertsPerLane) == 0;

    for (int g = el; g < geom.groups; g += geom.lanes) {
      const int e_base = g * kAmdRouteExpertsPerLane;

      float acc[kAmdRouteExpertsPerLane];
#pragma unroll
      for (int j = 0; j < kAmdRouteExpertsPerLane; j++) acc[j] = 0.0f;

      for (int h = hc; h < hs; h += geom.chunks) {
        const float xv = static_cast<float>(x[h]);
        const __half* wrow =
            router_weight + static_cast<int64_t>(h) * ne + e_base;
        if (vectorized) {
          union {
            uint4 packed;
            __half lane[kAmdRouteExpertsPerLane];
          } w;
          w.packed = *reinterpret_cast<const uint4*>(wrow);
#pragma unroll
          for (int j = 0; j < kAmdRouteExpertsPerLane; j++)
            acc[j] = __fmaf_rn(xv, static_cast<float>(w.lane[j]), acc[j]);
        } else {
          for (int j = 0; j < kAmdRouteExpertsPerLane && e_base + j < ne; j++)
            acc[j] = __fmaf_rn(xv, static_cast<float>(wrow[j]), acc[j]);
        }
      }

      float* dst = s_part + (static_cast<size_t>(hc) * geom.groups + g) *
                                kAmdRouteExpertsPerLane;
#pragma unroll
      for (int j = 0; j < kAmdRouteExpertsPerLane; j++) dst[j] = acc[j];
    }
  }
  __syncthreads();

  for (int e = tid; e < ne; e += nthreads) {
    const int group = e / kAmdRouteExpertsPerLane;
    const int slot = e % kAmdRouteExpertsPerLane;
    float dot = 0.0f;
    for (int c = 0; c < geom.chunks; c++) {
      dot += s_part[(static_cast<size_t>(c) * geom.groups + group) *
                        kAmdRouteExpertsPerLane +
                    slot];
    }
    const float prob = 1.0f / (1.0f + expf(-dot));
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
      if (v > bv) {
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
// relu2 activation (fp16 only): out = relu(x)^2, elementwise. Callers activate
// in place (input == output), so neither pointer is __restrict__.
// ---------------------------------------------------------------------------
__global__ void qmoe_amd_relu2_kernel(
    const __half* input,
    __half* output,
    int64_t total) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  float v = fmaxf(0.0f, static_cast<float>(input[idx]));
  output[idx] = static_cast<__half>(v * v);
}

// ---------------------------------------------------------------------------
// Expert bucketing, parallelized over experts.
//
// The shared hip_qmoe_bucket_tokens runs one thread per expert in a single
// block, so every thread serially scans all num_tokens * k pairs twice. That
// is sized for com.microsoft::QMoE (~32 experts, k <= 4); this op's pair space
// is one to two orders of magnitude larger, which leaves the whole scan on one
// compute unit. Here each expert gets its own block and the scan is split
// across the block's threads.
//
// The scatter preserves the serial version's exact token-major in-expert
// ordering: threads own *contiguous* slices of the pair space and a block-wide
// exclusive scan of their match counts places each thread's writes so that
// concatenating them in thread order reproduces ascending pair order.
// ---------------------------------------------------------------------------
__global__ void qmoe_amd_bucket_count_kernel(
    const int32_t* __restrict__ expert_indices,
    int32_t* __restrict__ expert_counts,
    int total_pairs) {
  const int e = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const int nthreads = static_cast<int>(blockDim.x);

  // Counting is order-independent, so this pass can stride (coalesced).
  int count = 0;
  for (int i = tid; i < total_pairs; i += nthreads) {
    if (expert_indices[i] == e) count++;
  }

  __shared__ int s_count[kBlock];
  s_count[tid] = count;
  __syncthreads();
  for (int off = nthreads / 2; off > 0; off >>= 1) {
    if (tid < off) s_count[tid] += s_count[tid + off];
    __syncthreads();
  }
  if (tid == 0) expert_counts[e] = s_count[0];
}

__global__ void qmoe_amd_bucket_offsets_kernel(
    const int32_t* __restrict__ expert_counts,
    int32_t* __restrict__ expert_offsets,
    int num_experts) {
  if (threadIdx.x != 0) return;
  int sum = 0;
  expert_offsets[0] = 0;
  for (int e = 0; e < num_experts; e++) {
    sum += expert_counts[e];
    expert_offsets[e + 1] = sum;
  }
}

__global__ void qmoe_amd_bucket_scatter_kernel(
    const int32_t* __restrict__ expert_indices,
    const __half* __restrict__ expert_weights,
    const int32_t* __restrict__ expert_offsets,
    int32_t* __restrict__ sorted_token_ids,
    __half* __restrict__ sorted_weights,
    int total_pairs,
    int k) {
  const int e = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const int nthreads = static_cast<int>(blockDim.x);

  // Contiguous (not strided) slices: required for the ordering guarantee.
  const int chunk = (total_pairs + nthreads - 1) / nthreads;
  const int lo = tid * chunk;
  const int hi = (lo + chunk < total_pairs) ? (lo + chunk) : total_pairs;

  int count = 0;
  for (int i = lo; i < hi; i++) {
    if (expert_indices[i] == e) count++;
  }

  // Exclusive scan of the per-thread counts. A serial scan by one thread over
  // nthreads entries is negligible next to the pair scan above and keeps the
  // ordering argument obvious.
  __shared__ int s_count[kBlock];
  __shared__ int s_base[kBlock];
  s_count[tid] = count;
  __syncthreads();
  if (tid == 0) {
    int running = 0;
    for (int t = 0; t < nthreads; t++) {
      s_base[t] = running;
      running += s_count[t];
    }
  }
  __syncthreads();

  int w = expert_offsets[e] + s_base[tid];
  for (int i = lo; i < hi; i++) {
    if (expert_indices[i] == e) {
      sorted_token_ids[w] = i / k;  // pair index i == token * k + slot
      sorted_weights[w] = expert_weights[i];
      w++;
    }
  }
}

// ---------------------------------------------------------------------------
// Fused routed branch for decode (num_tokens == 1).
//
// Derived from the com.microsoft::QMoE decode GEMVs in qmoe_kernel.hip (same
// uint4 packed-nibble main loop, same factored dequant, same wave-shuffle
// reduction), specialized for this op: relu2 instead of SwiGLU (so no
// gate/linear pairing), no zero_points, and no bias.
//
// Grid is (N / TILE_N, k) -- one block per (routing slot, output tile), with
// the expert id looked up from expert_indices on the device. That is what
// removes the host round-trip: the general path has to know the active expert
// set to compute weight base pointers on the host.
// ---------------------------------------------------------------------------

// Reduces TILE_N per-thread partials across the block. Returns true for the
// single thread holding the reduced values in `partial`.
template <int BLOCK_SIZE, int TILE_N>
__device__ static inline bool amd_decode_tile_reduce(int tid,
                                                    float (&partial)[TILE_N]) {
  // Wave size is 32 on RDNA and 64 on CDNA (HIPDNN_WAVE_SIZE). __shfl_down
  // defaults to the full hardware wave, so the geometry must be expressed in
  // terms of the real wave size to keep the reduction inside one wave. When
  // BLOCK_SIZE < WARP_SIZE only the first BLOCK_SIZE lanes are active, so cap
  // the span to avoid shuffling in an inactive lane.
  constexpr int WARP_SIZE = HIPDNN_WAVE_SIZE;
  constexpr int NUM_WARPS = BLOCK_SIZE > WARP_SIZE ? BLOCK_SIZE / WARP_SIZE : 1;
  constexpr int RED_SPAN = BLOCK_SIZE < WARP_SIZE ? BLOCK_SIZE : WARP_SIZE;
  const int warp_id = tid / WARP_SIZE;
  const int lane = tid % WARP_SIZE;

#pragma unroll
  for (int t = 0; t < TILE_N; t++) {
    for (int off = RED_SPAN >> 1; off > 0; off >>= 1)
      partial[t] += __shfl_down(partial[t], off);
  }

  // A single-wave block is already fully reduced by the shuffles above. Plain
  // `if` on a constexpr condition rather than `if constexpr` because this file
  // is compiled as C++14.
  constexpr bool kCrossWave = BLOCK_SIZE > WARP_SIZE;
  if (!kCrossWave) return lane == 0;

  __shared__ float warp_sums[TILE_N * NUM_WARPS];
  if (lane == 0) {
#pragma unroll
    for (int t = 0; t < TILE_N; t++)
      warp_sums[t * NUM_WARPS + warp_id] = partial[t];
  }
  __syncthreads();
  if (warp_id != 0) return false;
#pragma unroll
  for (int t = 0; t < TILE_N; t++) {
    float v = (lane < NUM_WARPS) ? warp_sums[t * NUM_WARPS + lane] : 0.0f;
    for (int off = WARP_SIZE >> 1; off > 0; off >>= 1)
      v += __shfl_down(v, off);
    partial[t] = v;
  }
  return lane == 0;
}

// Accumulates TILE_N dot products of `a` against TILE_N packed-int4 weight
// rows. Symmetric int4 with MatMulNBits' default zero point of 8, so the
// dequantized dot factors to (sum(a * b_q) - 8 * sum(a)) * scale; this op's
// schema has no zero_points input, so that factoring is unconditional.
template <int BLOCK_SIZE, int TILE_N>
__device__ static inline void amd_decode_gemv_tile(
    const __half* __restrict__ a,
    const uint8_t* (&w_rows)[TILE_N],
    const __half* (&s_rows)[TILE_N],
    int k_dim,
    int bs_shift,
    int tid,
    float (&partial)[TILE_N]) {
  const int num_u128 = k_dim / 32;
  for (int idx = tid; idx < num_u128; idx += BLOCK_SIZE) {
    const int k32 = idx * 32;

    float a_vals[32];
    float a_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 32; i++) {
      a_vals[i] = static_cast<float>(a[k32 + i]);
      a_sum += a_vals[i];
    }
    const int grp = k32 >> bs_shift;
    const float a_sum_x_zp = a_sum * 8.0f;

    uint4 b_pk[TILE_N];
    float sv[TILE_N];
#pragma unroll
    for (int t = 0; t < TILE_N; t++) {
      b_pk[t] = *reinterpret_cast<const uint4*>(&w_rows[t][idx * 16]);
      sv[t] = static_cast<float>(s_rows[t][grp]);
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
      partial[t] += (dot - a_sum_x_zp) * sv[t];
    }
  }
}

// FC1 + relu2: latent [latent_size] -> act_out [k, moe_inter].
template <int BLOCK_SIZE, int TILE_N>
__global__ void qmoe_amd_decode_fc1_kernel(
    const __half* __restrict__ latent,           // [latent_size]
    const int32_t* __restrict__ expert_indices,  // [k]
    const uint8_t* __restrict__ fc1_w,           // [E, moe_inter, latent/2]
    const __half* __restrict__ fc1_s,            // [E, moe_inter, n_blk_in]
    __half* __restrict__ act_out,                // [k, moe_inter]
    int latent_size,
    int moe_inter,
    int n_blk_in,
    int block_size) {
  const int n_base = static_cast<int>(blockIdx.x) * TILE_N;
  const int slot = static_cast<int>(blockIdx.y);
  const int tid = static_cast<int>(threadIdx.x);
  const int e = expert_indices[slot];

  const size_t w_row_bytes = static_cast<size_t>(latent_size) / 2;
  const uint8_t* w = fc1_w + static_cast<size_t>(e) * moe_inter * w_row_bytes;
  const __half* s = fc1_s + static_cast<size_t>(e) * moe_inter * n_blk_in;

  float partial[TILE_N];
  const uint8_t* w_rows[TILE_N];
  const __half* s_rows[TILE_N];
#pragma unroll
  for (int t = 0; t < TILE_N; t++) {
    partial[t] = 0.0f;
    const int n = n_base + t;
    const int sn = (n < moe_inter) ? n : 0;
    w_rows[t] = w + static_cast<size_t>(sn) * w_row_bytes;
    s_rows[t] = s + static_cast<size_t>(sn) * n_blk_in;
  }

  amd_decode_gemv_tile<BLOCK_SIZE, TILE_N>(
      latent, w_rows, s_rows, latent_size,
      __builtin_ctz(static_cast<unsigned>(block_size)), tid, partial);

  if (amd_decode_tile_reduce<BLOCK_SIZE, TILE_N>(tid, partial)) {
#pragma unroll
    for (int t = 0; t < TILE_N; t++) {
      const int n = n_base + t;
      if (n < moe_inter) {
        const float v = fmaxf(0.0f, partial[t]);
        act_out[static_cast<size_t>(slot) * moe_inter + n] =
            static_cast<__half>(v * v);
      }
    }
  }
}

// FC2: act_out [k, moe_inter] -> slot_buf [k, latent_size], pre-multiplied by
// the routing weight so the reduce below is a pure sum.
template <int BLOCK_SIZE, int TILE_N>
__global__ void qmoe_amd_decode_fc2_kernel(
    const __half* __restrict__ act_out,          // [k, moe_inter]
    const int32_t* __restrict__ expert_indices,  // [k]
    const __half* __restrict__ expert_weights,   // [k]
    const uint8_t* __restrict__ fc2_w,           // [E, latent, moe_inter/2]
    const __half* __restrict__ fc2_s,            // [E, latent, n_blk_mid]
    __half* __restrict__ slot_buf,               // [k, latent_size]
    int moe_inter,
    int latent_size,
    int n_blk_mid,
    int block_size) {
  const int n_base = static_cast<int>(blockIdx.x) * TILE_N;
  const int slot = static_cast<int>(blockIdx.y);
  const int tid = static_cast<int>(threadIdx.x);
  const int e = expert_indices[slot];
  const float wt = static_cast<float>(expert_weights[slot]);

  const size_t w_row_bytes = static_cast<size_t>(moe_inter) / 2;
  const uint8_t* w = fc2_w + static_cast<size_t>(e) * latent_size * w_row_bytes;
  const __half* s = fc2_s + static_cast<size_t>(e) * latent_size * n_blk_mid;

  float partial[TILE_N];
  const uint8_t* w_rows[TILE_N];
  const __half* s_rows[TILE_N];
#pragma unroll
  for (int t = 0; t < TILE_N; t++) {
    partial[t] = 0.0f;
    const int n = n_base + t;
    const int sn = (n < latent_size) ? n : 0;
    w_rows[t] = w + static_cast<size_t>(sn) * w_row_bytes;
    s_rows[t] = s + static_cast<size_t>(sn) * n_blk_mid;
  }

  amd_decode_gemv_tile<BLOCK_SIZE, TILE_N>(
      act_out + static_cast<size_t>(slot) * moe_inter, w_rows, s_rows, moe_inter,
      __builtin_ctz(static_cast<unsigned>(block_size)), tid, partial);

  if (amd_decode_tile_reduce<BLOCK_SIZE, TILE_N>(tid, partial)) {
#pragma unroll
    for (int t = 0; t < TILE_N; t++) {
      const int n = n_base + t;
      if (n < latent_size) {
        slot_buf[static_cast<size_t>(slot) * latent_size + n] =
            static_cast<__half>(wt * partial[t]);
      }
    }
  }
}

// Reduce: acc[n] = sum over slots of slot_buf[slot, n]. All k slots accumulate
// in fp32 before the single fp16 store; the general path's scatter_add instead
// rounds to fp16 after every expert.
__global__ void qmoe_amd_decode_reduce_kernel(
    const __half* __restrict__ slot_buf,  // [k, latent_size]
    __half* __restrict__ acc,             // [latent_size]
    int latent_size,
    int k) {
  const int n = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (n >= latent_size) return;
  float sum = 0.0f;
#pragma unroll 4
  for (int slot = 0; slot < k; slot++) {
    sum += static_cast<float>(slot_buf[static_cast<size_t>(slot) * latent_size + n]);
  }
  acc[n] = static_cast<__half>(sum);
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

  const AmdRouteGeom geom =
      amd_route_geom(static_cast<int>(num_experts), kAmdRoutingBlock);
  const size_t part_floats =
      static_cast<size_t>(geom.chunks) * geom.groups * kAmdRouteExpertsPerLane;
  const size_t shmem =
      (part_floats + 2 * static_cast<size_t>(num_experts)) * sizeof(float);
  // The kernel also declares kAmdRoutingBlock floats + ints of *static* shared
  // memory for the argmax reduction, which draws on the same per-block LDS
  // budget as the dynamic request -- so check the total, not just `shmem`.
  constexpr size_t kStaticShmem =
      kAmdRoutingBlock * (sizeof(float) + sizeof(int));
  constexpr size_t kMaxShmem = 64 * 1024;
  if (shmem + kStaticShmem > kMaxShmem) {
    fprintf(stderr,
            "[custom_kernels] hip_qmoe_amd_route: shared memory requirement "
            "%zu bytes (%zu dynamic + %zu static) exceeds %zu byte budget "
            "(experts=%lld)\n",
            shmem + kStaticShmem, shmem, kStaticShmem, kMaxShmem,
            (long long)num_experts);
    return -1;
  }

  // Clear any stale error so the check below attributes a launch failure to
  // THIS kernel. hipGetLastError() reports the last error from any prior HIP
  // call on the thread, and the caller aborts the whole op on a non-zero
  // return -- without this clear, an error left sticky by an earlier op's
  // unchecked launch fails QMoE with a bogus diagnostic.
  (void)hipGetLastError();
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

  (void)hipGetLastError(); // clear stale error for correct attribution
  hipLaunchKernelGGL(qmoe_amd_relu2_kernel, dim3(grid), dim3(kBlock), 0, s,
                     static_cast<const __half*>(input),
                     static_cast<__half*>(output), total);

  hipError_t err = hipGetLastError();
  if (err != hipSuccess)
    fprintf(stderr, "[custom_kernels] hip_qmoe_amd_relu2 failed: %s\n",
            hipGetErrorString(err));
  return static_cast<int>(err);
}

extern "C" int hip_qmoe_amd_bucket_tokens(
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
  if (num_tokens <= 0) return 0;

  if (element_size_bytes != 2) {
    fprintf(stderr,
            "[custom_kernels] hip_qmoe_amd_bucket_tokens: only fp16 supported, "
            "got element_size %lld\n",
            (long long)element_size_bytes);
    return -1;
  }
  if (num_experts <= 0 || k <= 0) {
    fprintf(stderr,
            "[custom_kernels] hip_qmoe_amd_bucket_tokens: invalid sizes "
            "(experts=%lld k=%lld)\n",
            (long long)num_experts, (long long)k);
    return -1;
  }
  const int64_t total_pairs = num_tokens * k;
  if (total_pairs > 0x7fffffffLL) {
    fprintf(stderr,
            "[custom_kernels] hip_qmoe_amd_bucket_tokens: num_tokens * k = "
            "%lld exceeds the int32 pair index range\n",
            (long long)total_pairs);
    return -1;
  }

  hipStream_t s = static_cast<hipStream_t>(stream);
  const unsigned experts = static_cast<unsigned>(num_experts);
  const int pairs = static_cast<int>(total_pairs);

  (void)hipGetLastError();
  hipLaunchKernelGGL(qmoe_amd_bucket_count_kernel, dim3(experts), dim3(kBlock),
                     0, s, static_cast<const int32_t*>(expert_indices),
                     static_cast<int32_t*>(expert_counts), pairs);
  hipLaunchKernelGGL(qmoe_amd_bucket_offsets_kernel, dim3(1), dim3(1), 0, s,
                     static_cast<const int32_t*>(expert_counts),
                     static_cast<int32_t*>(expert_offsets),
                     static_cast<int>(num_experts));
  hipLaunchKernelGGL(qmoe_amd_bucket_scatter_kernel, dim3(experts),
                     dim3(kBlock), 0, s,
                     static_cast<const int32_t*>(expert_indices),
                     static_cast<const __half*>(expert_weights),
                     static_cast<const int32_t*>(expert_offsets),
                     static_cast<int32_t*>(sorted_token_ids),
                     static_cast<__half*>(sorted_weights), pairs,
                     static_cast<int>(k));

  hipError_t err = hipGetLastError();
  if (err != hipSuccess)
    fprintf(stderr, "[custom_kernels] hip_qmoe_amd_bucket_tokens failed: %s\n",
            hipGetErrorString(err));
  return static_cast<int>(err);
}

extern "C" int hip_qmoe_amd_decode_fused_supported(
    int64_t latent_size,
    int64_t moe_intermediate_size,
    int64_t block_size,
    int64_t expert_weight_bits,
    int64_t element_size_bytes) {
  if (element_size_bytes != 2 || expert_weight_bits != 4) return 0;
  if (latent_size <= 0 || moe_intermediate_size <= 0) return 0;
  // The packed-nibble main loop consumes K in 32-element tiles and derives the
  // quant-block index by shifting the tile offset, so the quant block must be
  // a power of two no smaller than a tile. K % block_size == 0 additionally
  // makes the per-expert row stride exactly K/2 bytes (and implies K % 32 == 0,
  // which the uint4 loads need).
  if (block_size < 32 || (block_size & (block_size - 1)) != 0) return 0;
  if ((latent_size % block_size) != 0) return 0;
  if ((moe_intermediate_size % block_size) != 0) return 0;
  return 1;
}

// TILE_N output columns per block. Even/odd does not matter here (unlike the
// SwiGLU-paired com.microsoft variant) because relu2 is elementwise.
static constexpr int kAmdDecodeTileN = 8;
#define QMOE_AMD_DECODE_BS_LIST(X) X(32) X(64) X(128) X(256)

// Match the block size to the number of 32-element K tiles: threads beyond the
// tile count never enter the main loop but still pay the reduction.
static inline int amd_decode_block_size(int64_t k_dim) {
  const int64_t units = k_dim / 32;
  if (units >= 256) return 256;
  if (units >= 128) return 128;
  if (units >= 64) return 64;
  return 32;
}

template <int BS>
static hipError_t launch_amd_decode_fc1(
    hipStream_t s, const void* latent, const void* expert_indices,
    const void* fc1_w, const void* fc1_s, void* act_out, int latent_size,
    int moe_inter, int n_blk_in, int block_size, int k) {
  dim3 grid(static_cast<unsigned>((moe_inter + kAmdDecodeTileN - 1) /
                                  kAmdDecodeTileN),
            static_cast<unsigned>(k));
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(qmoe_amd_decode_fc1_kernel<BS, kAmdDecodeTileN>), grid,
      dim3(BS), 0, s, static_cast<const __half*>(latent),
      static_cast<const int32_t*>(expert_indices),
      static_cast<const uint8_t*>(fc1_w), static_cast<const __half*>(fc1_s),
      static_cast<__half*>(act_out), latent_size, moe_inter, n_blk_in,
      block_size);
  return hipGetLastError();
}

template <int BS>
static hipError_t launch_amd_decode_fc2(
    hipStream_t s, const void* act_out, const void* expert_indices,
    const void* expert_weights, const void* fc2_w, const void* fc2_s,
    void* slot_buf, int moe_inter, int latent_size, int n_blk_mid,
    int block_size, int k) {
  dim3 grid(static_cast<unsigned>((latent_size + kAmdDecodeTileN - 1) /
                                  kAmdDecodeTileN),
            static_cast<unsigned>(k));
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(qmoe_amd_decode_fc2_kernel<BS, kAmdDecodeTileN>), grid,
      dim3(BS), 0, s, static_cast<const __half*>(act_out),
      static_cast<const int32_t*>(expert_indices),
      static_cast<const __half*>(expert_weights),
      static_cast<const uint8_t*>(fc2_w), static_cast<const __half*>(fc2_s),
      static_cast<__half*>(slot_buf), moe_inter, latent_size, n_blk_mid,
      block_size);
  return hipGetLastError();
}

extern "C" int hip_qmoe_amd_decode_fused(
    void* stream,
    const void* latent,
    const void* expert_indices,
    const void* expert_weights,
    const void* fc1_weights,
    const void* fc1_scales,
    const void* fc2_weights,
    const void* fc2_scales,
    void* act_scratch,
    void* slot_scratch,
    void* acc,
    int64_t latent_size,
    int64_t moe_intermediate_size,
    int64_t k,
    int64_t expert_weight_bits,
    int64_t block_size,
    int64_t element_size_bytes) {
  if (k <= 0) return 0;
  if (!hip_qmoe_amd_decode_fused_supported(latent_size, moe_intermediate_size,
                                           block_size, expert_weight_bits,
                                           element_size_bytes)) {
    fprintf(stderr,
            "[custom_kernels] hip_qmoe_amd_decode_fused: unsupported geometry "
            "(latent=%lld moe_inter=%lld block=%lld bits=%lld elem=%lld)\n",
            (long long)latent_size, (long long)moe_intermediate_size,
            (long long)block_size, (long long)expert_weight_bits,
            (long long)element_size_bytes);
    return -1;
  }

  hipStream_t s = static_cast<hipStream_t>(stream);
  const int i_latent = static_cast<int>(latent_size);
  const int i_moe_inter = static_cast<int>(moe_intermediate_size);
  const int i_block = static_cast<int>(block_size);
  const int i_k = static_cast<int>(k);
  const int n_blk_in = i_latent / i_block;
  const int n_blk_mid = i_moe_inter / i_block;

  (void)hipGetLastError();

  const int bs1 = amd_decode_block_size(latent_size);
  hipError_t err = hipErrorInvalidValue;
#define X(B)                                                                   \
  if (bs1 == (B))                                                              \
    err = launch_amd_decode_fc1<B>(s, latent, expert_indices, fc1_weights,     \
                                   fc1_scales, act_scratch, i_latent,          \
                                   i_moe_inter, n_blk_in, i_block, i_k);
  QMOE_AMD_DECODE_BS_LIST(X)
#undef X
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_qmoe_amd_decode_fused fc1 failed: %s\n",
            hipGetErrorString(err));
    return static_cast<int>(err);
  }

  const int bs2 = amd_decode_block_size(moe_intermediate_size);
  err = hipErrorInvalidValue;
#define X(B)                                                                   \
  if (bs2 == (B))                                                              \
    err = launch_amd_decode_fc2<B>(                                            \
        s, act_scratch, expert_indices, expert_weights, fc2_weights,           \
        fc2_scales, slot_scratch, i_moe_inter, i_latent, n_blk_mid, i_block,   \
        i_k);
  QMOE_AMD_DECODE_BS_LIST(X)
#undef X
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_qmoe_amd_decode_fused fc2 failed: %s\n",
            hipGetErrorString(err));
    return static_cast<int>(err);
  }

  const int reduce_grid = (i_latent + kBlock - 1) / kBlock;
  hipLaunchKernelGGL(qmoe_amd_decode_reduce_kernel, dim3(reduce_grid),
                     dim3(kBlock), 0, s,
                     static_cast<const __half*>(slot_scratch),
                     static_cast<__half*>(acc), i_latent, i_k);

  err = hipGetLastError();
  if (err != hipSuccess)
    fprintf(stderr,
            "[custom_kernels] hip_qmoe_amd_decode_fused reduce failed: %s\n",
            hipGetErrorString(err));
  return static_cast<int>(err);
}
