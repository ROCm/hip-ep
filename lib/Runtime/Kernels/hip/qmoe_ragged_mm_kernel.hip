/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Device-driven ragged MatMulNBits for com.microsoft::QMoE prefill.
 *
 * One launch covers every active expert.  Work is described by compact
 * row-groups produced by qmoe's bucket kernel:
 *   groups[g] = {expert_id, sorted_row_begin, row_count, reserved}.
 * Persistent blocks dequeue (group, N-tile) tasks from a device counter, so
 * the host never reads expert counts and never launches one GEMM per expert.
 */

#include "hip_custom_kernels.h"
#include "hip_arch_compat.h"
#include "matmul_nbits_wmma_core.h"
#include "debug_log.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace {

constexpr int kRowsPerGroup = 16;
constexpr int kRowsPerLargeGroup = 64;
constexpr int kColsPerBlock = 64;
constexpr int kKTile = 32;
constexpr int kThreadsPerBlock = 128;
// HIP reports WGPs in multiProcessorCount on RDNA; each WGP contains two CUs.
constexpr int kComputeUnitsPerWgp = 2;
constexpr int kMaxTopK = 16;
constexpr int kGemvTileN = 16;
constexpr int64_t kTwoWindowMinRows = 512;

constexpr int kGridBlockCandidates[] = {20, 40, 80, 160, 320, 640, 4096};
constexpr int kGridFactorCandidates[] = {1, 2, 4, 8, 16};

enum class RaggedGridOverrideKind {
  kDefault,
  kAbsoluteBlocks,
  kOccupancyFactor,
};

struct RaggedGridOverride {
  RaggedGridOverrideKind kind;
  int value;
};

enum class RaggedQueuePolicy : int {
  kAtomic1,
  kChunk2,
  kChunk4,
  kChunk8,
  kGridStride,
};

enum class RaggedTaskPolicy : int {
  kP0Current = 0,
  kP1EarlyWmma16 = 1,
  kP2EarlyWmma64 = 2,
  kP3EarlyBoth = 3,
};

struct RaggedTaskPolicyConfig {
  RaggedTaskPolicy policy;
  int gemv_max_rows;
  int wmma16_max_rows;
  const char* name;
};

enum RaggedTaskKind : int {
  kTaskGemv = 0,
  kTaskWmma16 = 1,
  kTaskWmma64 = 2,
};

template <size_t N>
bool is_grid_candidate(int value, const int (&candidates)[N]) {
  return std::find(candidates, candidates + N, value) != candidates + N;
}

bool parse_positive_int(const char* text, int* value) {
  if (!text || !*text || !value) {
    return false;
  }
  errno = 0;
  char* end = nullptr;
  const long parsed = std::strtol(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || parsed <= 0 ||
      parsed > std::numeric_limits<int>::max()) {
    return false;
  }
  *value = static_cast<int>(parsed);
  return true;
}

RaggedGridOverride load_ragged_grid_override() {
  const char* blocks = std::getenv("HIPDNN_QMOE_RAGGED_GRID_BLOCKS");
  if (blocks) {
    int value = 0;
    if (parse_positive_int(blocks, &value) &&
        is_grid_candidate(value, kGridBlockCandidates)) {
      fprintf(stderr,
              "[custom_kernels] qmoe ragged benchmark grid override: "
              "HIPDNN_QMOE_RAGGED_GRID_BLOCKS=%d\n",
              value);
      return {RaggedGridOverrideKind::kAbsoluteBlocks, value};
    }
    fprintf(stderr,
            "[custom_kernels] invalid "
            "HIPDNN_QMOE_RAGGED_GRID_BLOCKS='%s'; expected one of "
            "20,40,80,160,320,640,4096; using the production shape "
            "policy\n",
            blocks);
    return {RaggedGridOverrideKind::kDefault, 1};
  }

  const char* factor = std::getenv("HIPDNN_QMOE_RAGGED_GRID_FACTOR");
  if (factor) {
    int value = 0;
    if (parse_positive_int(factor, &value) &&
        is_grid_candidate(value, kGridFactorCandidates)) {
      fprintf(stderr,
              "[custom_kernels] qmoe ragged benchmark grid override: "
              "HIPDNN_QMOE_RAGGED_GRID_FACTOR=%d\n",
              value);
      return {RaggedGridOverrideKind::kOccupancyFactor, value};
    }
    fprintf(stderr,
            "[custom_kernels] invalid HIPDNN_QMOE_RAGGED_GRID_FACTOR='%s'; "
            "expected one of 1,2,4,8,16; using the production shape "
            "policy\n",
            factor);
  }
  return {RaggedGridOverrideKind::kDefault, 1};
}

const RaggedGridOverride& get_ragged_grid_override() {
  // Benchmark-only knobs are process-latched so production launches do not
  // repeatedly enter the CRT environment lock. Absolute blocks take
  // precedence whenever that variable is present, including invalid values.
  static const RaggedGridOverride grid_override = load_ragged_grid_override();
  return grid_override;
}

RaggedQueuePolicy load_ragged_queue_policy() {
  const char* policy = std::getenv("HIPDNN_QMOE_RAGGED_QUEUE_POLICY");
  if (!policy) {
    return RaggedQueuePolicy::kAtomic1;
  }
  if (std::strcmp(policy, "atomic1") == 0) {
    return RaggedQueuePolicy::kAtomic1;
  }
  if (std::strcmp(policy, "chunk2") == 0) {
    return RaggedQueuePolicy::kChunk2;
  }
  if (std::strcmp(policy, "chunk4") == 0) {
    return RaggedQueuePolicy::kChunk4;
  }
  if (std::strcmp(policy, "chunk8") == 0) {
    return RaggedQueuePolicy::kChunk8;
  }
  if (std::strcmp(policy, "grid-stride") == 0) {
    return RaggedQueuePolicy::kGridStride;
  }
  fprintf(stderr,
          "[custom_kernels] invalid HIPDNN_QMOE_RAGGED_QUEUE_POLICY='%s'; "
          "expected atomic1,chunk2,chunk4,chunk8,grid-stride; using atomic1\n",
          policy);
  return RaggedQueuePolicy::kAtomic1;
}

const RaggedQueuePolicy& get_ragged_queue_policy() {
  // Benchmark-only policy selection is process-latched for the same reason as
  // the grid override: do not enter the CRT environment lock on every launch.
  static const RaggedQueuePolicy queue_policy = load_ragged_queue_policy();
  return queue_policy;
}

RaggedTaskPolicyConfig load_ragged_task_policy(const char* env_name,
                                                const char* stage_name) {
  const char* policy = std::getenv(env_name);
  if (!policy || std::strcmp(policy, "p0-current") == 0) {
    return {RaggedTaskPolicy::kP0Current, 7, 31, "p0-current"};
  }
  if (std::strcmp(policy, "p1-early-wmma16") == 0) {
    fprintf(stderr,
            "[custom_kernels] qmoe ragged benchmark %s task policy "
            "override: %s=%s\n",
            stage_name, env_name, policy);
    return {RaggedTaskPolicy::kP1EarlyWmma16, 3, 31,
            "p1-early-wmma16"};
  }
  if (std::strcmp(policy, "p2-early-wmma64") == 0) {
    fprintf(stderr,
            "[custom_kernels] qmoe ragged benchmark %s task policy "
            "override: %s=%s\n",
            stage_name, env_name, policy);
    return {RaggedTaskPolicy::kP2EarlyWmma64, 7, 15,
            "p2-early-wmma64"};
  }
  if (std::strcmp(policy, "p3-early-both") == 0) {
    fprintf(stderr,
            "[custom_kernels] qmoe ragged benchmark %s task policy "
            "override: %s=%s\n",
            stage_name, env_name, policy);
    return {RaggedTaskPolicy::kP3EarlyBoth, 3, 15, "p3-early-both"};
  }
  fprintf(stderr,
          "[custom_kernels] invalid %s='%s'; expected one of "
          "p0-current,p1-early-wmma16,p2-early-wmma64,p3-early-both; "
          "using p0-current\n",
          env_name, policy);
  return {RaggedTaskPolicy::kP0Current, 7, 31, "p0-current"};
}

const RaggedTaskPolicyConfig& get_ragged_fc1_task_policy() {
  // Process-latched benchmark override. Production defaults to P0.
  static const RaggedTaskPolicyConfig policy = load_ragged_task_policy(
      "HIPDNN_QMOE_RAGGED_FC1_TASK_POLICY", "FC1");
  return policy;
}

const RaggedTaskPolicyConfig& get_ragged_fc2_task_policy() {
  // Process-latched independently from FC1 so every Bxy bundle is reproducible.
  static const RaggedTaskPolicyConfig policy = load_ragged_task_policy(
      "HIPDNN_QMOE_RAGGED_FC2_TASK_POLICY", "FC2");
  return policy;
}

__host__ __device__ __forceinline__ int select_ragged_task_kind(
    int row_count, int gemv_max_rows, int wmma16_max_rows) {
  return row_count <= gemv_max_rows
             ? kTaskGemv
             : row_count <= wmma16_max_rows ? kTaskWmma16 : kTaskWmma64;
}

template <RaggedQueuePolicy Policy>
__host__ __device__ __forceinline__ uint32_t queue_chunk_size() {
  if constexpr (Policy == RaggedQueuePolicy::kChunk2) {
    return 2;
  } else if constexpr (Policy == RaggedQueuePolicy::kChunk4) {
    return 4;
  } else if constexpr (Policy == RaggedQueuePolicy::kChunk8) {
    return 8;
  }
  return 1;
}

__device__ __forceinline__ void qmoe_wmma16_task(
    const _Float16* input, const int32_t* input_row_ids,
    const uint8_t* weights, const _Float16* scales,
    const uint8_t* zero_points, const _Float16* bias, _Float16* output,
    int total_rows, int N, int K, int block_size, int row_begin,
    int row_count, int n_begin, int weight_row_bytes, int k_blocks) {
  const int first_rows = min(row_count, kRowsPerGroup);
  GemmFp16U4Impl<kRowsPerGroup, kColsPerBlock, 1, 1, /*USE_ZEROS=*/true,
                 /*BOUNDS_CHECK=*/true, /*FUSED_DQ=*/true, kKTile,
                 /*ZEROS_U8=*/true, /*ADD_BIAS=*/true,
                 /*EXPLICIT_TILE=*/true, /*CHECK_K_TAIL=*/true,
                 /*RUNTIME_OPTIONAL=*/true>(
      total_rows, N, K, input, K, weights, scales, zero_points, block_size,
      k_blocks, output, N, /*swizzle_n=*/1, weight_row_bytes, row_begin,
      n_begin, first_rows, input_row_ids, bias);
  if (row_count > kRowsPerGroup) {
    GemmFp16U4Impl<kRowsPerGroup, kColsPerBlock, 1, 1,
                   /*USE_ZEROS=*/true, /*BOUNDS_CHECK=*/true,
                   /*FUSED_DQ=*/true, kKTile, /*ZEROS_U8=*/true,
                   /*ADD_BIAS=*/true, /*EXPLICIT_TILE=*/true,
                   /*CHECK_K_TAIL=*/true, /*RUNTIME_OPTIONAL=*/true>(
        total_rows, N, K, input, K, weights, scales, zero_points, block_size,
        k_blocks, output, N, /*swizzle_n=*/1, weight_row_bytes,
        row_begin + kRowsPerGroup, n_begin, row_count - kRowsPerGroup,
        input_row_ids, bias);
  }
}

__device__ __forceinline__ void qmoe_wmma64_task(
    const _Float16* input, const int32_t* input_row_ids,
    const uint8_t* weights, const _Float16* scales,
    const uint8_t* zero_points, const _Float16* bias, _Float16* output,
    int total_rows, int N, int K, int block_size, int row_begin,
    int row_count, int n_begin, int weight_row_bytes, int k_blocks) {
  GemmFp16U4Impl<kRowsPerLargeGroup, kColsPerBlock, 1, 4,
                 /*USE_ZEROS=*/true, /*BOUNDS_CHECK=*/true,
                 /*FUSED_DQ=*/true, kKTile, /*ZEROS_U8=*/true,
                 /*ADD_BIAS=*/true, /*EXPLICIT_TILE=*/true,
                 /*CHECK_K_TAIL=*/true, /*RUNTIME_OPTIONAL=*/true>(
      total_rows, N, K, input, K, weights, scales, zero_points, block_size,
      k_blocks, output, N, /*swizzle_n=*/1, weight_row_bytes, row_begin,
      n_begin, row_count, input_row_ids, bias);
}

__device__ __forceinline__ void qmoe_gemv_task(
    const _Float16* input, const int32_t* input_row_ids,
    const uint8_t* weights, const _Float16* scales,
    const uint8_t* zero_points, const _Float16* bias, _Float16* output,
    int N, int K, int block_size, int row_begin, int row_count, int n_begin,
    int weight_row_bytes, int k_blocks) {
  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / 32;
  const int lane = tid & 31;
  const int wave_n_begin = n_begin + wave * kGemvTileN;

  for (int r = 0; r < row_count; ++r) {
    const int sorted_row = row_begin + r;
    const int input_row = input_row_ids ? input_row_ids[sorted_row] : sorted_row;
    const _Float16* a_row = input + static_cast<size_t>(input_row) * K;
    float partial[kGemvTileN] = {};

    const int chunks32 = K / 32;
    for (int chunk = lane; chunk < chunks32; chunk += 32) {
      const int k32 = chunk * 32;
      const int group = k32 / block_size;
      float a_values[32];
      float a_sum = 0.0f;
#pragma unroll
      for (int i = 0; i < 32; ++i) {
        a_values[i] = static_cast<float>(a_row[k32 + i]);
        a_sum += a_values[i];
      }

      uint4 packed[kGemvTileN];
      float scale_values[kGemvTileN];
#pragma unroll
      for (int tn = 0; tn < kGemvTileN; ++tn) {
        const int n = wave_n_begin + tn;
        const int safe_n = n < N ? n : 0;
        packed[tn] = *reinterpret_cast<const uint4*>(
            weights + static_cast<size_t>(safe_n) * weight_row_bytes + k32 / 2);
        scale_values[tn] = static_cast<float>(
            scales[static_cast<size_t>(safe_n) * k_blocks + group]);
      }

#pragma unroll
      for (int tn = 0; tn < kGemvTileN; ++tn) {
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i)
          dot = __fmaf_rn(a_values[i],
                          static_cast<float>((packed[tn].x >> (i * 4)) & 15u),
                          dot);
#pragma unroll
        for (int i = 0; i < 8; ++i)
          dot = __fmaf_rn(a_values[8 + i],
                          static_cast<float>((packed[tn].y >> (i * 4)) & 15u),
                          dot);
#pragma unroll
        for (int i = 0; i < 8; ++i)
          dot = __fmaf_rn(a_values[16 + i],
                          static_cast<float>((packed[tn].z >> (i * 4)) & 15u),
                          dot);
#pragma unroll
        for (int i = 0; i < 8; ++i)
          dot = __fmaf_rn(a_values[24 + i],
                          static_cast<float>((packed[tn].w >> (i * 4)) & 15u),
                          dot);
        const int n = wave_n_begin + tn;
        const int safe_n = n < N ? n : 0;
        const float zp = zero_points
            ? static_cast<float>(zero_points[
                  static_cast<size_t>(safe_n) * k_blocks + group])
            : 8.0f;
        partial[tn] += (dot - a_sum * zp) * scale_values[tn];
      }
    }

    const int k16 = chunks32 * 32;
    if (K - k16 >= 16 && lane == 0) {
      const int group = k16 / block_size;
      float a_values[16];
      float a_sum = 0.0f;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        a_values[i] = static_cast<float>(a_row[k16 + i]);
        a_sum += a_values[i];
      }
#pragma unroll
      for (int tn = 0; tn < kGemvTileN; ++tn) {
        const int n = wave_n_begin + tn;
        const int safe_n = n < N ? n : 0;
        const uint2 packed = *reinterpret_cast<const uint2*>(
            weights + static_cast<size_t>(safe_n) * weight_row_bytes + k16 / 2);
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i)
          dot = __fmaf_rn(a_values[i],
                          static_cast<float>((packed.x >> (i * 4)) & 15u), dot);
#pragma unroll
        for (int i = 0; i < 8; ++i)
          dot = __fmaf_rn(a_values[8 + i],
                          static_cast<float>((packed.y >> (i * 4)) & 15u), dot);
        const float scale = static_cast<float>(
            scales[static_cast<size_t>(safe_n) * k_blocks + group]);
        const float zp = zero_points
            ? static_cast<float>(zero_points[
                  static_cast<size_t>(safe_n) * k_blocks + group])
            : 8.0f;
        partial[tn] += (dot - a_sum * zp) * scale;
      }
    }

    const int tail_begin = k16 + ((K - k16) >= 16 ? 16 : 0);
    for (int k = tail_begin + lane; k < K; k += 32) {
      const float a = static_cast<float>(a_row[k]);
      const int group = k / block_size;
#pragma unroll
      for (int tn = 0; tn < kGemvTileN; ++tn) {
        const int n = wave_n_begin + tn;
        const int safe_n = n < N ? n : 0;
        const uint8_t packed = weights[
            static_cast<size_t>(safe_n) * weight_row_bytes + k / 2];
        const float q = static_cast<float>(
            (k & 1) ? (packed >> 4) : (packed & 15u));
        const float scale = static_cast<float>(
            scales[static_cast<size_t>(safe_n) * k_blocks + group]);
        const float zp = zero_points
            ? static_cast<float>(zero_points[
                  static_cast<size_t>(safe_n) * k_blocks + group])
            : 8.0f;
        partial[tn] = __fmaf_rn(a, (q - zp) * scale, partial[tn]);
      }
    }

#pragma unroll
    for (int tn = 0; tn < kGemvTileN; ++tn)
      for (int offset = 16; offset > 0; offset >>= 1)
        partial[tn] += __shfl_down(partial[tn], offset);

    if (lane == 0) {
#pragma unroll
      for (int tn = 0; tn < kGemvTileN; ++tn) {
        const int n = wave_n_begin + tn;
        if (n < N) {
          float value = partial[tn];
          if (bias) value += static_cast<float>(bias[n]);
          output[static_cast<size_t>(sorted_row) * N + n] =
              static_cast<_Float16>(value);
        }
      }
    }
  }
}

// `groups` is a flat int32 array, four values per group:
//   [expert_id, sorted_row_begin, row_count, reserved].
// `input_row_ids == nullptr` means input is already in sorted-row order;
// otherwise input_row_ids[sorted_row] selects the source row (FC1 gather).
//
// One persistent workgroup owns one descriptor x 64-column task. Small groups
// use a wave-reduction GEMV; medium/large groups use the shared MatMulNBits
// double-buffered WMMA core. Every task kind uses the same 128-thread block.
template <RaggedQueuePolicy QueuePolicy>
__global__ __attribute__((amdgpu_flat_work_group_size(kThreadsPerBlock,
                                                       kThreadsPerBlock)))
void qmoe_ragged_matmul_nbits_adaptive_kernel(
    const _Float16 *__restrict__ input,
    const int32_t *__restrict__ input_row_ids,
    const int32_t *__restrict__ groups,
    const int32_t *__restrict__ group_count,
    uint32_t *__restrict__ queue_head,
    const uint8_t *__restrict__ weights,
    const _Float16 *__restrict__ scales,
    const uint8_t *__restrict__ zero_points,
    const _Float16 *__restrict__ bias,
    _Float16 *__restrict__ output,
    int64_t total_rows,
    int64_t num_experts,
    int64_t N,
    int64_t K,
    int64_t block_size,
    int gemv_max_rows,
    int wmma16_max_rows) {
  __shared__ uint32_t task_id_shared;
  __shared__ int group_count_shared;

  const int tid = static_cast<int>(threadIdx.x);
  const int n_tiles = static_cast<int>((N + kColsPerBlock - 1) /
                                       kColsPerBlock);
  if (tid == 0) {
    group_count_shared = *group_count;
  }
  __syncthreads();

  const uint32_t task_count =
      static_cast<uint32_t>(group_count_shared) *
      static_cast<uint32_t>(n_tiles);

  const uint32_t kQueueChunk = queue_chunk_size<QueuePolicy>();
  uint32_t chunk_offset = kQueueChunk;
  uint32_t grid_stride_task = static_cast<uint32_t>(blockIdx.x);

  while (true) {
    uint32_t task_id = 0;
    if constexpr (QueuePolicy == RaggedQueuePolicy::kGridStride) {
      task_id = grid_stride_task;
      grid_stride_task += static_cast<uint32_t>(gridDim.x);
    } else {
      if (chunk_offset == kQueueChunk) {
        if (tid == 0) {
          task_id_shared = atomicAdd(queue_head, kQueueChunk);
        }
        __syncthreads();
        chunk_offset = 0;
      }
      task_id = task_id_shared + chunk_offset;
      ++chunk_offset;
    }

    if (task_id >= task_count) {
      return;
    }

    const int group_id = static_cast<int>(task_id / n_tiles);
    const int n_tile = static_cast<int>(task_id % n_tiles);
    const int expert = groups[group_id * 4 + 0];
    const int row_begin = groups[group_id * 4 + 1];
    const int row_count = groups[group_id * 4 + 2];
    const int task_kind = select_ragged_task_kind(
        row_count, gemv_max_rows, wmma16_max_rows);
    const int n_begin = n_tile * kColsPerBlock;
    const bool valid_expert = expert >= 0 && expert < num_experts;

    const int k_blocks = static_cast<int>((K + block_size - 1) / block_size);
    const int weight_row_bytes = k_blocks * static_cast<int>(block_size / 2);
    if (valid_expert) {
      const size_t expert_cols = static_cast<size_t>(expert) * N;
      const uint8_t* expert_weights =
          weights + expert_cols * weight_row_bytes;
      const _Float16* expert_scales = scales + expert_cols * k_blocks;
      const uint8_t* expert_zeros =
          zero_points ? zero_points + expert_cols * k_blocks : nullptr;
      const _Float16* expert_bias = bias ? bias + expert_cols : nullptr;

      if (task_kind == kTaskGemv) {
        qmoe_gemv_task(input, input_row_ids, expert_weights, expert_scales,
                        expert_zeros, expert_bias, output, static_cast<int>(N),
                        static_cast<int>(K), static_cast<int>(block_size),
                        row_begin, row_count, n_begin, weight_row_bytes,
                        k_blocks);
      } else if (task_kind == kTaskWmma16) {
        qmoe_wmma16_task(
            input, input_row_ids, expert_weights, expert_scales, expert_zeros,
            expert_bias, output, static_cast<int>(total_rows),
            static_cast<int>(N), static_cast<int>(K),
            static_cast<int>(block_size), row_begin, row_count, n_begin,
            weight_row_bytes, k_blocks);
      } else if (task_kind == kTaskWmma64) {
        qmoe_wmma64_task(
            input, input_row_ids, expert_weights, expert_scales, expert_zeros,
            expert_bias, output, static_cast<int>(total_rows),
            static_cast<int>(N), static_cast<int>(K),
            static_cast<int>(block_size), row_begin, row_count, n_begin,
            weight_row_bytes, k_blocks);
      }
    }
    __syncthreads();
  }
}

struct RaggedLaunchConfig {
  int device_cu_count;
  int resident_blocks_per_cu;
  int grid_blocks;
  int64_t upper_tasks;
  RaggedQueuePolicy queue_policy;
};

hipError_t get_ragged_resident_blocks(RaggedQueuePolicy policy,
                                      int* resident_blocks_per_cu) {
  switch (policy) {
    case RaggedQueuePolicy::kAtomic1:
      return hipOccupancyMaxActiveBlocksPerMultiprocessor(
          resident_blocks_per_cu,
          qmoe_ragged_matmul_nbits_adaptive_kernel<
              RaggedQueuePolicy::kAtomic1>,
          kThreadsPerBlock, /*dynamicSMemSize=*/0);
    case RaggedQueuePolicy::kChunk2:
      return hipOccupancyMaxActiveBlocksPerMultiprocessor(
          resident_blocks_per_cu,
          qmoe_ragged_matmul_nbits_adaptive_kernel<
              RaggedQueuePolicy::kChunk2>,
          kThreadsPerBlock, /*dynamicSMemSize=*/0);
    case RaggedQueuePolicy::kChunk4:
      return hipOccupancyMaxActiveBlocksPerMultiprocessor(
          resident_blocks_per_cu,
          qmoe_ragged_matmul_nbits_adaptive_kernel<
              RaggedQueuePolicy::kChunk4>,
          kThreadsPerBlock, /*dynamicSMemSize=*/0);
    case RaggedQueuePolicy::kChunk8:
      return hipOccupancyMaxActiveBlocksPerMultiprocessor(
          resident_blocks_per_cu,
          qmoe_ragged_matmul_nbits_adaptive_kernel<
              RaggedQueuePolicy::kChunk8>,
          kThreadsPerBlock, /*dynamicSMemSize=*/0);
    case RaggedQueuePolicy::kGridStride:
      return hipOccupancyMaxActiveBlocksPerMultiprocessor(
          resident_blocks_per_cu,
          qmoe_ragged_matmul_nbits_adaptive_kernel<
              RaggedQueuePolicy::kGridStride>,
          kThreadsPerBlock, /*dynamicSMemSize=*/0);
  }
  return hipErrorInvalidValue;
}

hipError_t get_ragged_launch_config(int64_t total_rows, int64_t N,
                                    RaggedLaunchConfig* config) {
  if (!config || total_rows <= 0 || N <= 0) {
    return hipErrorInvalidValue;
  }

  const int64_t n_tiles = (N + kColsPerBlock - 1) / kColsPerBlock;
  if (total_rows > std::numeric_limits<int64_t>::max() / n_tiles) {
    return hipErrorInvalidValue;
  }

  int device = 0;
  hipError_t err = hipGetDevice(&device);
  if (err != hipSuccess) {
    return err;
  }
  hipDeviceProp_t properties{};
  err = hipGetDeviceProperties(&properties, device);
  if (err != hipSuccess) {
    return err;
  }

  const RaggedQueuePolicy queue_policy = get_ragged_queue_policy();
  int resident_blocks_per_cu = 0;
  err = get_ragged_resident_blocks(queue_policy, &resident_blocks_per_cu);
  if (err != hipSuccess) {
    return err;
  }
  if (properties.multiProcessorCount <= 0 || resident_blocks_per_cu <= 0) {
    return hipErrorInvalidConfiguration;
  }

  const int64_t upper_tasks = total_rows * n_tiles;
  const int device_cu_count =
      properties.multiProcessorCount * kComputeUnitsPerWgp;
  const int64_t resident_capacity =
      static_cast<int64_t>(device_cu_count) * resident_blocks_per_cu;
  const RaggedGridOverride& grid_override = get_ragged_grid_override();
  const int production_factor = total_rows >= kTwoWindowMinRows ? 2 : 1;
  int64_t grid_limit = resident_capacity;
  if (grid_override.kind == RaggedGridOverrideKind::kAbsoluteBlocks) {
    grid_limit = grid_override.value;
  } else {
    const int factor =
        grid_override.kind == RaggedGridOverrideKind::kOccupancyFactor
            ? grid_override.value
            : production_factor;
    if (resident_capacity > std::numeric_limits<int64_t>::max() /
                                factor) {
      grid_limit = std::numeric_limits<int64_t>::max();
    } else {
      grid_limit = resident_capacity * factor;
    }
  }
  config->device_cu_count = device_cu_count;
  config->resident_blocks_per_cu = resident_blocks_per_cu;
  config->grid_blocks = static_cast<int>(std::min(
      std::min(upper_tasks, grid_limit),
      static_cast<int64_t>(std::numeric_limits<int>::max())));
  config->upper_tasks = upper_tasks;
  config->queue_policy = queue_policy;
  return hipSuccess;
}

template <RaggedQueuePolicy QueuePolicy>
hipError_t launch_ragged_matmul_nbits(
    hipStream_t stream, const void* input, const void* input_row_ids,
    const void* row_groups, const void* row_group_count, void* queue_head,
    const void* weights, const void* scales, const void* zero_points_u8,
    const void* bias, void* output, int64_t total_rows, int64_t num_experts,
    int64_t N, int64_t K, int64_t block_size, int grid_blocks,
    int gemv_max_rows, int wmma16_max_rows) {
  hipLaunchKernelGGL(
      qmoe_ragged_matmul_nbits_adaptive_kernel<QueuePolicy>,
      dim3(grid_blocks), dim3(kThreadsPerBlock), 0, stream,
      static_cast<const _Float16 *>(input),
      static_cast<const int32_t *>(input_row_ids),
      static_cast<const int32_t *>(row_groups),
      static_cast<const int32_t *>(row_group_count),
      static_cast<uint32_t *>(queue_head),
      static_cast<const uint8_t *>(weights),
      static_cast<const _Float16 *>(scales),
      static_cast<const uint8_t *>(zero_points_u8),
      static_cast<const _Float16 *>(bias), static_cast<_Float16 *>(output),
      total_rows, num_experts, N, K, block_size, gemv_max_rows,
      wmma16_max_rows);
  return hipGetLastError();
}

__global__ void qmoe_reduce_sorted_pairs_kernel(
    const __half *__restrict__ sorted_expert_output,
    const int32_t *__restrict__ pair_to_sorted,
    const int32_t *__restrict__ expert_indices,
    const __half *__restrict__ expert_weights,
    __half *__restrict__ output,
    int64_t num_tokens,
    int64_t width,
    int k) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= num_tokens * width) {
    return;
  }

  const int64_t token = idx / width;
  const int64_t col = idx - token * width;
  uint32_t used = 0;
  __half accum = __float2half(0.0f);

  // Match the old host loop: contributions are accumulated in ascending
  // expert-id order, with an fp16 store/round after every contribution.
  for (int rank = 0; rank < k; ++rank) {
    int best_slot = -1;
    int32_t best_expert = INT32_MAX;
    for (int slot = 0; slot < k; ++slot) {
      if ((used & (1u << slot)) != 0) {
        continue;
      }
      const int32_t expert = expert_indices[token * k + slot];
      if (expert < best_expert) {
        best_expert = expert;
        best_slot = slot;
      }
    }
    if (best_slot < 0) {
      break;
    }
    used |= 1u << best_slot;
    const int64_t pair = token * k + best_slot;
    const int32_t sorted_row = pair_to_sorted[pair];
    const float old_value = static_cast<float>(accum);
    const float weight = static_cast<float>(expert_weights[pair]);
    const float value =
        static_cast<float>(sorted_expert_output[
            static_cast<int64_t>(sorted_row) * width + col]);
    accum = __float2half(old_value + weight * value);
  }
  output[idx] = accum;
}

} // namespace

extern "C" int hip_qmoe_ragged_matmul_nbits_get_launch_config(
    int64_t total_rows, int64_t N, int32_t* device_cu_count,
    int32_t* resident_blocks_per_cu, int32_t* grid_blocks,
    int64_t* upper_tasks) {
  if (!device_cu_count || !resident_blocks_per_cu || !grid_blocks ||
      !upper_tasks) {
    return -1;
  }
  RaggedLaunchConfig config{};
  const hipError_t err = get_ragged_launch_config(total_rows, N, &config);
  if (err != hipSuccess) {
    return static_cast<int>(err);
  }
  *device_cu_count = config.device_cu_count;
  *resident_blocks_per_cu = config.resident_blocks_per_cu;
  *grid_blocks = config.grid_blocks;
  *upper_tasks = config.upper_tasks;
  return static_cast<int>(hipSuccess);
}

extern "C" int hip_qmoe_ragged_matmul_nbits_get_task_kind(
    int64_t is_fc1, int64_t row_count, int32_t* policy_id,
    int32_t* task_kind) {
  if ((is_fc1 != 0 && is_fc1 != 1) || row_count <= 0 ||
      row_count > kRowsPerLargeGroup || !policy_id || !task_kind) {
    return -1;
  }
  const RaggedTaskPolicyConfig& policy =
      is_fc1 ? get_ragged_fc1_task_policy() : get_ragged_fc2_task_policy();
  *policy_id = static_cast<int32_t>(policy.policy);
  *task_kind = select_ragged_task_kind(
      static_cast<int>(row_count), policy.gemv_max_rows,
      policy.wmma16_max_rows);
  return static_cast<int>(hipSuccess);
}

extern "C" int hip_qmoe_ragged_matmul_nbits(
    void *stream, const void *input, const void *input_row_ids,
    const void *row_groups, const void *row_group_count, void *queue_head,
    const void *weights, const void *scales, const void *zero_points_u8,
    const void *bias, void *output, int64_t total_rows, int64_t num_experts,
    int64_t N, int64_t K, int64_t block_size, int64_t element_size_bytes) {
  if (!stream || !input || !row_groups || !row_group_count || !queue_head ||
      !weights || !scales || !output) {
    fprintf(stderr,
            "[custom_kernels] hip_qmoe_ragged_matmul_nbits: null required "
            "argument\n");
    return -1;
  }
  if (element_size_bytes != 2) {
    fprintf(stderr,
            "[custom_kernels] hip_qmoe_ragged_matmul_nbits: fp16 only\n");
    return -1;
  }
  if (total_rows <= 0 || num_experts <= 0 || N <= 0 || K <= 0 ||
      block_size <= 0 || (block_size & 1) != 0) {
    fprintf(stderr,
            "[custom_kernels] hip_qmoe_ragged_matmul_nbits: invalid shape\n");
    return -1;
  }
  if (hipdnn_device_is_wave64()) {
    fprintf(stderr,
            "[custom_kernels] hip_qmoe_ragged_matmul_nbits: WMMA requires "
            "a wave32 RDNA target\n");
    return -1;
  }

  RaggedLaunchConfig config{};
  hipError_t err = get_ragged_launch_config(total_rows, N, &config);
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_qmoe_ragged_matmul_nbits launch config "
            "failed: %s\n",
            hipGetErrorString(err));
    return static_cast<int>(err);
  }
  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  const RaggedTaskPolicyConfig& task_policy = input_row_ids
                                                   ? get_ragged_fc1_task_policy()
                                                   : get_ragged_fc2_task_policy();
  switch (config.queue_policy) {
    case RaggedQueuePolicy::kAtomic1:
      err = launch_ragged_matmul_nbits<RaggedQueuePolicy::kAtomic1>(
          hip_stream, input, input_row_ids, row_groups, row_group_count,
          queue_head, weights, scales, zero_points_u8, bias, output, total_rows,
          num_experts, N, K, block_size, config.grid_blocks,
          task_policy.gemv_max_rows, task_policy.wmma16_max_rows);
      break;
    case RaggedQueuePolicy::kChunk2:
      err = launch_ragged_matmul_nbits<RaggedQueuePolicy::kChunk2>(
          hip_stream, input, input_row_ids, row_groups, row_group_count,
          queue_head, weights, scales, zero_points_u8, bias, output, total_rows,
          num_experts, N, K, block_size, config.grid_blocks,
          task_policy.gemv_max_rows, task_policy.wmma16_max_rows);
      break;
    case RaggedQueuePolicy::kChunk4:
      err = launch_ragged_matmul_nbits<RaggedQueuePolicy::kChunk4>(
          hip_stream, input, input_row_ids, row_groups, row_group_count,
          queue_head, weights, scales, zero_points_u8, bias, output, total_rows,
          num_experts, N, K, block_size, config.grid_blocks,
          task_policy.gemv_max_rows, task_policy.wmma16_max_rows);
      break;
    case RaggedQueuePolicy::kChunk8:
      err = launch_ragged_matmul_nbits<RaggedQueuePolicy::kChunk8>(
          hip_stream, input, input_row_ids, row_groups, row_group_count,
          queue_head, weights, scales, zero_points_u8, bias, output, total_rows,
          num_experts, N, K, block_size, config.grid_blocks,
          task_policy.gemv_max_rows, task_policy.wmma16_max_rows);
      break;
    case RaggedQueuePolicy::kGridStride:
      err = launch_ragged_matmul_nbits<RaggedQueuePolicy::kGridStride>(
          hip_stream, input, input_row_ids, row_groups, row_group_count,
          queue_head, weights, scales, zero_points_u8, bias, output, total_rows,
          num_experts, N, K, block_size, config.grid_blocks,
          task_policy.gemv_max_rows, task_policy.wmma16_max_rows);
      break;
  }
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_qmoe_ragged_matmul_nbits launch failed: "
            "%s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}

extern "C" int hip_qmoe_reduce_sorted_pairs(
    void *stream, const void *sorted_expert_output,
    const void *pair_to_sorted, const void *expert_indices,
    const void *expert_weights, void *output, int64_t num_tokens,
    int64_t width, int64_t k, int64_t element_size_bytes) {
  if (!stream || !sorted_expert_output || !pair_to_sorted ||
      !expert_indices || !expert_weights || !output) {
    fprintf(stderr,
            "[custom_kernels] hip_qmoe_reduce_sorted_pairs: null required "
            "argument\n");
    return -1;
  }
  if (element_size_bytes != 2 || num_tokens <= 0 || width <= 0 || k <= 0 ||
      k > kMaxTopK) {
    fprintf(stderr,
            "[custom_kernels] hip_qmoe_reduce_sorted_pairs: invalid shape or "
            "dtype\n");
    return -1;
  }

  constexpr int block = 256;
  const int64_t elems = num_tokens * width;
  const int grid = static_cast<int>((elems + block - 1) / block);
  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  hipLaunchKernelGGL(
      qmoe_reduce_sorted_pairs_kernel, dim3(grid), dim3(block), 0, hip_stream,
      static_cast<const __half *>(sorted_expert_output),
      static_cast<const int32_t *>(pair_to_sorted),
      static_cast<const int32_t *>(expert_indices),
      static_cast<const __half *>(expert_weights),
      static_cast<__half *>(output), num_tokens, width, static_cast<int>(k));

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_qmoe_reduce_sorted_pairs launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
