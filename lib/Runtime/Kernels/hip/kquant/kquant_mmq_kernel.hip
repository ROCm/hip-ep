/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "kquant_mmq_launch.h"

#include "hip/kquant_abi.h"
#include "hip_custom_kernels.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

using kquant_mmq::ActivationBlock;
using kquant_mmq::kActivationBlockValues;
using kquant_mmq::MmqProblem;

namespace {

constexpr int kMaximumCachedDevices = 16;
// Up to eight rows go to MMVQ (hip_kquant_matmul_q8_1), as in llama.cpp.
constexpr int64_t kMinInputRows = 9;

static_assert(HIP_KQUANT_MMQ_PADDING_BYTES ==
              kquant_mmq::kMaxTileWidth * sizeof(ActivationBlock));
static_assert(alignof(float4) == HIP_KQUANT_MMQ_F32_ALIGNMENT);
static_assert(alignof(hip_q8_1_block_t) == HIP_Q8_1_BLOCK_ALIGNMENT);

std::atomic<int> deviceSupport[kMaximumCachedDevices]{};

bool currentDeviceSupportsMmq() {
  int device = -1;
  if (hipGetDevice(&device) != hipSuccess || device < 0 ||
      device >= kMaximumCachedDevices)
    return false;
  int cached = deviceSupport[device].load(std::memory_order_acquire);
  if (cached != 0)
    return cached > 0;

  hipDeviceProp_t properties{};
  const bool supported =
      hipGetDeviceProperties(&properties, device) == hipSuccess &&
      properties.warpSize == 32 &&
      properties.sharedMemPerBlock >=
          kquant_mmq::sharedBytes(HIP_KQUANT_FORMAT_Q6_K, 128) &&
      std::strncmp(properties.gcnArchName, "gfx1151", 7) == 0 &&
      (properties.gcnArchName[7] == '\0' || properties.gcnArchName[7] == ':');
  deviceSupport[device].store(supported ? 1 : -1, std::memory_order_release);
  return supported;
}

/// Whether MMQ runs a weight format.
bool isMmqFormat(int formatId) {
  return kquant_mmq::tileThreads(formatId, 16) != 0;
}

/// A weight block's size and the alignment MMQ's tile loads need of it.
struct BlockLayout {
  size_t bytes;
  size_t alignment;
};

template <typename Block> constexpr BlockLayout layoutOf() {
  return {sizeof(Block), alignof(Block)};
}

BlockLayout blockLayout(int formatId) {
  switch (formatId) {
  case HIP_KQUANT_FORMAT_Q4_K:
    return layoutOf<hip_kquant_q4_k_block_t>();
  case HIP_KQUANT_FORMAT_Q5_K:
    return layoutOf<hip_kquant_q5_k_block_t>();
  case HIP_KQUANT_FORMAT_Q6_K:
    return layoutOf<hip_kquant_q6_k_block_t>();
  default:
    return {0, 16};
  }
}

bool isWholeSteps(int64_t K) {
  return K % kquant_mmq::kReductionStep == 0;
}

#define HIP_KQUANT_MMQ_FORMATS(X) X(Q4_K) X(Q5_K) X(Q6_K)

int pickTileWidth(int formatId, int columns, int rows) {
  switch (formatId) {
#define HIP_KQUANT_MMQ_PICK(F)                                                   \
  case HIP_KQUANT_FORMAT_##F:                                            \
    return kquant_mmq::pickTileWidth<HIP_KQUANT_FORMAT_##F>(columns,   \
                                                                    rows);
    HIP_KQUANT_MMQ_FORMATS(HIP_KQUANT_MMQ_PICK)
#undef HIP_KQUANT_MMQ_PICK
  default:
    return 0;
  }
}

int launchMmq(hipStream_t stream, int formatId, const MmqProblem &problem) {
  switch (formatId) {
#define HIP_KQUANT_MMQ_LAUNCH(F)                                                 \
  case HIP_KQUANT_FORMAT_##F:                                            \
    return kquant_mmq::launchMmqFormat<HIP_KQUANT_FORMAT_##F>(stream,  \
                                                                      problem);
    HIP_KQUANT_MMQ_FORMATS(HIP_KQUANT_MMQ_LAUNCH)
#undef HIP_KQUANT_MMQ_LAUNCH
  default:
    return -1;
  }
}

__device__ int loadWord(const void *address) {
  int value;
  __builtin_memcpy(&value, address, sizeof(value));
  return value;
}

__device__ float halfBitsToFloat(uint16_t bits) {
  __half_raw raw;
  raw.x = bits;
  return __half2float(__half(raw));
}

/// A dense product: one channel, weights shared across samples.
int launchDense(void *stream, int formatId, const void *mmqActivation,
                const void *packedWeights, float *output, int64_t M, int64_t N,
                int64_t K, int64_t batchCount) {
  MmqProblem problem;
  problem.weights = packedWeights;
  problem.activation = static_cast<const int *>(mmqActivation);
  problem.output = output;
  problem.rows = static_cast<int>(N);
  problem.K = static_cast<int>(K);
  problem.columns = static_cast<int>(M);
  problem.samples = static_cast<int>(batchCount);
  problem.activationSampleStride =
      M * (K / kActivationBlockValues) * (sizeof(ActivationBlock) / sizeof(int));
  problem.outputSampleStride = M * N;
  const int status =
      launchMmq(static_cast<hipStream_t>(stream), formatId, problem);
  if (status < 0)
    std::fprintf(stderr, "[custom_kernels] hip_kquant_mmq: no MMQ "
                         "kernel for format %d\n",
                 formatId);
  return status;
}

/// Copies standard Q8_1 rows into the MMQ layout,
/// `[sample][K / 128][row]`.
template <bool StoreSums>
__global__ void
packMmqQ8_1Kernel(const hip_q8_1_block_t *__restrict__ source,
                  ActivationBlock *__restrict__ destination, int64_t M,
                  int64_t K) {
  const int64_t valueBase =
      (static_cast<int64_t>(blockDim.x) * blockIdx.y + threadIdx.x) * 4;
  if (valueBase >= K)
    return;

  const int64_t batch = blockIdx.z;
  const int64_t row = blockIdx.x;
  const int64_t sourceRow = batch * M + row;
  const int64_t sourceBlock = valueBase / HIP_Q8_1_BLOCK_ELEMENTS;
  const int quantIndex = static_cast<int>(valueBase % kActivationBlockValues);
  const int blockInMmq = quantIndex / HIP_Q8_1_BLOCK_ELEMENTS;
  const int wordInBlock =
      (quantIndex % HIP_Q8_1_BLOCK_ELEMENTS) / sizeof(int);
  const hip_q8_1_block_t &sourceValue =
      source[sourceRow * (K / HIP_Q8_1_BLOCK_ELEMENTS) + sourceBlock];
  ActivationBlock &destinationValue =
      destination[(batch * (K / kActivationBlockValues) +
                   valueBase / kActivationBlockValues) *
                      M +
                  row];
  reinterpret_cast<int *>(destinationValue.qs)[quantIndex / sizeof(int)] =
      loadWord(sourceValue.qs + wordInBlock * sizeof(int));
  if (wordInBlock == 0) {
    if constexpr (StoreSums) {
      reinterpret_cast<int *>(destinationValue.ds4)[blockInMmq] =
          loadWord(&sourceValue);
    } else {
      destinationValue.d4[blockInMmq] = halfBitsToFloat(sourceValue.d);
    }
  }
}

int packMmqQ8_1(void *stream, bool storeSums, const void *quantizedActivation,
                void *mmqActivation, int64_t M, int64_t K,
                int64_t batchCount) {
  constexpr int threads = 128;
  const int64_t gridY = (K + threads * 4 - 1) / (threads * 4);
  if (gridY <= 0 || gridY > std::numeric_limits<unsigned>::max() ||
      M > std::numeric_limits<unsigned>::max() ||
      batchCount > std::numeric_limits<unsigned>::max())
    return -1;
  const dim3 grid(static_cast<unsigned>(M), static_cast<unsigned>(gridY),
                  static_cast<unsigned>(batchCount));
  const auto *source =
      static_cast<const hip_q8_1_block_t *>(quantizedActivation);
  auto *destination = static_cast<ActivationBlock *>(mmqActivation);
  if (storeSums)
    hipLaunchKernelGGL((packMmqQ8_1Kernel<true>), grid, dim3(threads), 0,
                       static_cast<hipStream_t>(stream), source, destination,
                       M, K);
  else
    hipLaunchKernelGGL((packMmqQ8_1Kernel<false>), grid, dim3(threads), 0,
                       static_cast<hipStream_t>(stream), source, destination,
                       M, K);
  return static_cast<int>(hipGetLastError());
}

__global__ void quantizeMmqQ8_1D4Kernel(const float *__restrict__ activation,
                                        ActivationBlock *__restrict__ quantized,
                                        int64_t M, int64_t K) {
  const int64_t valueBase =
      (static_cast<int64_t>(blockDim.x) * blockIdx.y + threadIdx.x) * 4;
  if (valueBase >= K)
    return;

  const int64_t batch = blockIdx.z;
  const int64_t row = blockIdx.x;
  const float4 values = reinterpret_cast<const float4 *>(
      activation + (batch * M + row) * K)[valueBase / 4];
  float maximum = fabsf(values.x);
  maximum = fmaxf(maximum, fabsf(values.y));
  maximum = fmaxf(maximum, fabsf(values.z));
  maximum = fmaxf(maximum, fabsf(values.w));
#pragma unroll
  for (int offset = 4; offset > 0; offset >>= 1)
    maximum = fmaxf(maximum, __shfl_xor(maximum, offset, 32));

  const float inverseScale = maximum == 0.0f ? 0.0f : 127.0f / maximum;
  const float scale = inverseScale == 0.0f ? 0.0f : 1.0f / inverseScale;
  char4 packed;
  packed.x = static_cast<int8_t>(roundf(values.x * inverseScale));
  packed.y = static_cast<int8_t>(roundf(values.y * inverseScale));
  packed.z = static_cast<int8_t>(roundf(values.z * inverseScale));
  packed.w = static_cast<int8_t>(roundf(values.w * inverseScale));

  const int64_t block = valueBase / kActivationBlockValues;
  const int quantIndex = static_cast<int>(valueBase % kActivationBlockValues);
  ActivationBlock &destination =
      quantized[(batch * (K / kActivationBlockValues) + block) * M + row];
  reinterpret_cast<char4 *>(destination.qs)[quantIndex / 4] = packed;
  if (quantIndex % HIP_Q8_1_BLOCK_ELEMENTS == 0) {
    const int scaleIndex = quantIndex / HIP_Q8_1_BLOCK_ELEMENTS;
    destination.d4[scaleIndex] = scale;
  }
}

int launchFromF32(void *stream, int formatId, const float *activation,
                  void *q8Scratch, const void *packedWeights, float *output,
                  int64_t M, int64_t N, int64_t K, int64_t batchCount) {
  constexpr int quantizeThreads = 128;
  const int64_t quantizeGridY =
      (K + quantizeThreads * 4 - 1) / (quantizeThreads * 4);
  if (quantizeGridY <= 0 ||
      quantizeGridY > std::numeric_limits<unsigned>::max()) {
    std::fprintf(stderr,
                 "[custom_kernels] hip_kquant_mmq_f32: quantization "
                 "grid overflow\n");
    return -1;
  }
  hipLaunchKernelGGL(
      quantizeMmqQ8_1D4Kernel,
      dim3(static_cast<unsigned>(M), static_cast<unsigned>(quantizeGridY),
           static_cast<unsigned>(batchCount)),
      dim3(quantizeThreads), 0, static_cast<hipStream_t>(stream), activation,
      static_cast<ActivationBlock *>(q8Scratch), M, K);
  const hipError_t quantizeError = hipGetLastError();
  if (quantizeError != hipSuccess)
    return static_cast<int>(quantizeError);
  return launchDense(stream, formatId, q8Scratch, packedWeights, output, M, N,
                     K, batchCount);
}

bool hasValidPackedMmqArguments(const void *mmqActivation,
                                const void *packedWeights, const float *output,
                                int64_t M, int64_t N, int64_t K,
                                int64_t batchCount, int formatId) {
  return mmqActivation && packedWeights && output && isMmqFormat(formatId) &&
         reinterpret_cast<uintptr_t>(packedWeights) %
                 blockLayout(formatId).alignment ==
             0 &&
         reinterpret_cast<uintptr_t>(mmqActivation) %
                 HIP_Q8_1_BLOCK_ALIGNMENT ==
             0 &&
         reinterpret_cast<uintptr_t>(output) % alignof(float) == 0 &&
         hip_kquant_mmq_is_supported(M, N, K, batchCount, formatId);
}

} // namespace

extern "C" int hip_kquant_mmq_is_supported(int64_t M, int64_t N,
                                                 int64_t K, int64_t batchCount,
                                                 int formatId) {
  if (!isMmqFormat(formatId) ||
      M < kMinInputRows || N <= 0 || K <= 0 ||
      batchCount <= 0 || !isWholeSteps(K) ||
      M > std::numeric_limits<int>::max() ||
      N > std::numeric_limits<int>::max() ||
      K > std::numeric_limits<int>::max() ||
      batchCount > std::numeric_limits<uint16_t>::max())
    return 0;
  if (pickTileWidth(formatId, static_cast<int>(M), static_cast<int>(N)) ==
      0)
    return 0;
  const uint64_t activationStride = static_cast<uint64_t>(M) *
                                    (K / kActivationBlockValues) *
                                    (sizeof(ActivationBlock) / sizeof(int));
  const uint64_t outputStride = static_cast<uint64_t>(M) * N;
  const uint64_t weightBlocks = static_cast<uint64_t>(N) * (K / 256);
  return weightBlocks <=
             static_cast<uint64_t>(std::numeric_limits<int>::max()) &&
         activationStride * batchCount <=
             static_cast<uint64_t>(std::numeric_limits<int>::max()) &&
         outputStride * batchCount <=
             static_cast<uint64_t>(std::numeric_limits<int>::max()) &&
         currentDeviceSupportsMmq();
}

extern "C" int hip_kquant_mmq_reads_q8_1(int formatId) {
  return isMmqFormat(formatId) && kquant_mmq::readsSums(formatId);
}

extern "C" int
hip_kquant_mmq_packed_q8_1(void *stream, const void *mmqActivation,
                                 const void *packedWeights, float *output,
                                 int64_t M, int64_t N, int64_t K,
                                 int64_t batchCount, int formatId) {
  if (!isMmqFormat(formatId) || !isWholeSteps(K) ||
      !hasValidPackedMmqArguments(mmqActivation, packedWeights, output, M, N, K,
                                  batchCount, formatId)) {
    return -1;
  }
  return launchDense(stream, formatId, mmqActivation, packedWeights, output, M,
                     N, K, batchCount);
}

extern "C" int
hip_kquant_mmq_from_q8_1(void *stream, const void *quantizedActivation,
                               void *mmqScratch, const void *packedWeights,
                               float *output, int64_t M, int64_t N, int64_t K,
                               int64_t batchCount, int formatId) {
  if (!quantizedActivation || !isMmqFormat(formatId) ||
      reinterpret_cast<uintptr_t>(quantizedActivation) %
              HIP_Q8_1_BLOCK_ALIGNMENT !=
          0 ||
      !hasValidPackedMmqArguments(mmqScratch, packedWeights, output, M, N, K,
                                  batchCount, formatId)) {
    return -1;
  }
  const int status =
      packMmqQ8_1(stream, kquant_mmq::readsSums(formatId),
                  quantizedActivation, mmqScratch, M, K, batchCount);
  if (status != 0)
    return status;
  return launchDense(stream, formatId, mmqScratch, packedWeights, output, M,
                     N, K, batchCount);
}

extern "C" int hip_kquant_mmq_f32(void *stream, const float *activation,
                                        void *q8Scratch,
                                        const void *packedWeights,
                                        float *output, int64_t M, int64_t N,
                                        int64_t K, int64_t batchCount,
                                        int formatId) {
  if (!activation || !q8Scratch || !isMmqFormat(formatId) ||
      kquant_mmq::readsSums(formatId) || !isWholeSteps(K) ||
      reinterpret_cast<uintptr_t>(activation) %
              HIP_KQUANT_MMQ_F32_ALIGNMENT !=
          0 ||
      !hasValidPackedMmqArguments(q8Scratch, packedWeights, output, M, N, K,
                                  batchCount, formatId)) {
    std::fprintf(stderr, "[custom_kernels] hip_kquant_mmq_f32: invalid "
                         "arguments\n");
    return -1;
  }
  return launchFromF32(stream, formatId, activation, q8Scratch,
                       packedWeights, output, M, N, K, batchCount);
}
